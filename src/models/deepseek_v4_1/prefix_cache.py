"""A prompt's state kept on the host, keyed by the prompt's own tokens.

Every v41 request forward-passes its whole prompt, and a chat loop resends the history each turn, so
turn N pays for turns 1..N-1 again. This module is the store that makes turn N pay for the user turn
only: the caches are snapshotted at a *prefix length*, the snapshot is keyed by the tokens that
produced it, and a later request that starts with the same tokens restores the snapshot and forwards
the remainder instead of the whole prompt.

**What a snapshot is.** `CACHE_NAMES` below is the set of buffers a step writes and a later step
reads, and it is the same set `graphs.snapshot` walks. Two of them are second-axis *position* tables
-- `compress_kv_cache` and `k_cache` carry one row per `compress_ratio` positions -- so a prefix of
`p` tokens has written their first `p // ratio` rows and nothing else. The rest are stored whole: the
window ring is a fixed 128 slots keyed by `position % 128`, `kv_state`/`score_state` are the group
still filling, and all three are small enough that cutting them would be arithmetic for its own sake.
Not stored at all is `freqs_cis`: it is sliced by absolute position, so a resume needs no re-base.

**Why a resume is exact.** A restore is followed by a *continuation* forward at `start_pos = p`,
which is a chunk boundary and nothing more -- `tests/test_models_deepseek_v4_1_chunked_prefill.py`
pins every chunk boundary bit-equal to a one-shot prefill with `torch.equal`. What makes the state
match rather than merely be unread is `reset_state`: it zeroes the three position tables and refills
the compressor state, and the forward that produced the snapshot started from that same reset, so
the rows a sliced snapshot does not carry are zeros on both sides. A restore therefore reconstructs
the state a cold prefill of exactly `p` tokens leaves, not a state that is written over.

**Why the key is the tokens.** `thinking_mode`, `reasoning_effort`, tools and `response_format` all
reach the model as tokens, so keying on the rendered prompt folds every one of them in for free: a
different render is a different token stream and cannot false-hit. The key itself is a chain over
64-token blocks with the length mixed in, `h_i = blake2b(h_{i-1} || length || block_i)`, which is
vLLM's shape (`hash_request_tokens`'s block chain) and gives the same two properties: a prefix's key
is computed once and extended, and the key for a length identifies the whole prefix below it.

**What is not borrowed is vLLM's paged KV.** The caches here are pre-allocated contiguous buffers
with a partial group that is not block-aligned, so a block-granular store would be a rewrite of the
cache layout rather than a store on top of it. The unit is therefore a whole-prefix anchor: the
prompt's exact end, plus a fixed-length head anchor for the case where two conversations share a
rendered header but diverge after it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = [
    "BLOCK_TOKENS",
    "CACHE_NAMES",
    "Entry",
    "GROUPED_AXES",
    "PrefixCache",
    "extend_hash",
    "prefix_hashes",
    "restore",
    "restore_rows",
    "snapshot",
    "snapshot_rows",
]

# Every buffer a decode step writes to and a later step reads. `named_buffers` matches by substring,
# and none of these names contains another: `compress_kv_cache` has no `k_cache` in it and
# `window_kv_cache` has no `k_cache` either, both because the character before the `k` is a `_`.
CACHE_NAMES = ("window_kv_cache", "compress_kv_cache", "k_cache", "kv_state", "score_state")

# The two of those whose second axis is a group of positions rather than a fixed slot table -- one
# row per `compress_ratio` positions. Everything else in `CACHE_NAMES` is stored whole, and the
# partition has to be by name: `window_kv_cache`'s second axis is 128 slots, which read off its own
# width would look like a ratio of `max_seq_len // 128`.
GROUPED_AXES = ("compress_kv_cache", "k_cache")

# The block the hash chain steps by. vLLM uses the same size for the same reason: it is what a
# `by_length` index is walked in, so the chain for a candidate length is usually already computed and
# only a length that is not a multiple of it costs one extra hash.
BLOCK_TOKENS = 64

# 64 bits. vLLM's bar as well -- a collision needs two prompts whose first `L` tokens hash alike, and
# a collision is not a crash: it restores a state that was produced by different tokens, which is why
# the key is a chain over *all* the blocks below `L` and not a hash of the length alone.
HASH_BYTES = 8

_DOMAIN = b"pocketllm.v41.prefix-cache.1"


def _tokens_bytes(tokens) -> bytes:
    """The ids as little-endian 32-bit words, so the hash is over the tokens and not their spelling.

    Accepts a list, an `array`, or a tensor. A tensor is moved off the card first, which costs a copy
    the size of the block: `prompt_ids` reaches the model as a Python list and a caller that has a
    tensor has already paid more than this to build it.
    """
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().to(torch.int64).cpu().to(torch.int32).numpy().astype("<i4", copy=False).tobytes()
    return b"".join(int(token).to_bytes(4, "little", signed=True) for token in tokens)


def _seed(tag: bytes = b"") -> bytes:
    """The chain's starting value. `tag` is the geometry; see `geometry_tag`."""
    return hashlib.blake2b(_DOMAIN + tag, digest_size=HASH_BYTES).digest()


EMPTY_SEED = _seed()


def extend_hash(previous: bytes, length: int, tokens) -> bytes:
    """One step of the chain: the key for a prefix that ends `length` tokens in.

    `previous` is the chain at the last whole block below `length` (or the seed at zero), and
    `tokens` is the block's own tokens -- the length is mixed in beside them so that a block at
    position 64 and the same tokens at position 128 are different keys.
    """
    digest = hashlib.blake2b(previous, digest_size=HASH_BYTES)
    digest.update(int(length).to_bytes(8, "little"))
    digest.update(_tokens_bytes(tokens))
    return digest.digest()


def prefix_hashes(ids, seed: bytes = EMPTY_SEED) -> list[bytes]:
    """The chain after each whole block of `ids`: `out[j - 1]` keys the first `64 * j` tokens.

    A candidate length that is a multiple of `BLOCK_TOKENS` is answered out of this list and costs no
    hashing at all, which is every length the head anchor ever has and half of the ones the end
    anchor has. Lengths that are not are one `extend_hash` each.
    """
    out: list[bytes] = []
    previous = seed
    for blocks in range(1, len(ids) // BLOCK_TOKENS + 1):
        length = blocks * BLOCK_TOKENS
        previous = extend_hash(previous, length, ids[length - BLOCK_TOKENS : length])
        out.append(previous)
    return out


def _key_at(ids, length: int, seed: bytes, chain: list[bytes]) -> bytes:
    """The key for `ids[:length]`, given the whole-block chain for `ids`.

    `length` must be positive and no longer than `ids`; `chain` must reach `length // BLOCK_TOKENS`.
    """
    blocks, remaining = divmod(length, BLOCK_TOKENS)
    if not remaining:
        return chain[blocks - 1]
    previous = chain[blocks - 1] if blocks else seed
    return extend_hash(previous, length, ids[blocks * BLOCK_TOKENS : length])


def geometry_tag(model: torch.nn.Module, world_size: int, max_seq_len: int) -> bytes:
    """A tag over the geometry a snapshot was taken in, for the chain's seed.

    The store lives in the process, so the tags below cannot survive a restart; what they catch is a
    process that is not the one they think it is -- a different `world_size` writing to a
    tensor-parallel layout these bytes were not shaped for, or a `max_seq_len` that moved under the
    caches. Reading them off the tree rather than off the config is the same discipline
    `_cache_device` follows: the shapes are the thing that would actually be wrong.
    """
    parts = [f"world={int(world_size)}", f"max_seq_len={int(max_seq_len)}"]
    for name, buffer in sorted(model.named_buffers(), key=lambda pair: pair[0]):
        if any(cache in name for cache in CACHE_NAMES) and buffer.numel():
            parts.append(f"{name}={'x'.join(str(size) for size in buffer.shape)}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).digest()


# -- the snapshot itself --------------------------------------------------------------------------


def _ratio_of(columns: int, max_seq_len: int) -> int:
    """How many positions one row of a grouped buffer stands for, off the buffer's own width.

    Nothing is read from the config: the buffer was allocated with `max_seq_len // ratio` rows, so the
    ratio is `max_seq_len // columns` and the rule cannot drift from the tree that allocated it.
    """
    return max(1, max_seq_len // columns)


def snapshot_rows(
    model: torch.nn.Module,
    limit: int | None = None,
    max_seq_len: int | None = None,
    to_host: bool = True,
) -> dict[str, torch.Tensor]:
    """Every cache buffer in the tree, cut to the rows a prefix of `limit` tokens wrote.

    `limit=None` is each buffer whole, which is what `graphs.snapshot` takes around a capture pass:
    there the point is to put back exactly what was there, and the whole buffer is the cheapest way
    to be sure of it. A `limit` cuts only the two grouped buffers, to `limit // ratio` rows, and the
    result must therefore be restored into a state that has been reset -- see `restore_rows`.

    `to_host` is what the two callers differ by, and it is the difference between a store and a
    rewind rather than a preference. A snapshot the store keeps has to be pageable host memory: it
    outlives the request, and `DecodeGraphs.release` calls `torch.cuda.empty_cache` on it. Pinned
    memory is not the alternative it looks like either -- `/dev/shm` is held at 91% by the resident
    expert bank. The capture pass wants the opposite: its snapshot is taken and put back inside one
    call and never leaves it, so at 256K the caches are a gigabyte that would cross PCIe twice per
    step for nothing.
    """
    if limit is not None and max_seq_len is None:
        raise ValueError("a row-limited snapshot needs the `max_seq_len` the buffers were built with")
    # Read once, so the guard above stays the only place the two arguments can disagree.
    span = 0 if max_seq_len is None else int(max_seq_len)
    out: dict[str, torch.Tensor] = {}
    for name, buffer in model.named_buffers():
        if not any(cache in name for cache in CACHE_NAMES) or not buffer.numel():
            continue
        rows = None
        if limit is not None and any(axis in name for axis in GROUPED_AXES):
            rows = min(buffer.shape[1], limit // _ratio_of(buffer.shape[1], span))
        block = buffer if rows is None else buffer[:, :rows]
        out[name] = block.detach().to("cpu", copy=True) if to_host else block.detach().clone()
    return out


def restore_rows(model: torch.nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """Copy a snapshot back over the model's own buffers.

    `copy_` and never a rebind: the recorded graphs hold raw pointers into these buffers, so replacing
    one would leave every graph writing to memory nothing reads -- a wrong answer that still looks
    like a number, and one that only shows up as a divergence several steps later.

    A sliced snapshot fills the leading rows and leaves the rest alone, which is exact only because
    the caller reset first: `Attention.reset_state` zeroes the three position tables and refills the
    compressor state, and the forward that produced the snapshot started from that same reset, so the
    rows it did not carry were zeros on both sides.
    """
    by_name = dict(model.named_buffers())
    for name, value in saved.items():
        buffer = by_name[name]
        if value.shape == buffer.shape:
            buffer.copy_(value)
        else:
            buffer[:, : value.shape[1]].copy_(value)


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every cache buffer in the tree, cloned where it lies.

    Nothing reads it but `restore`, one forward later and inside the same call, so it never leaves the
    card; `snapshot_rows` is where that choice and the store's opposite one are explained.
    """
    return snapshot_rows(model, to_host=False)


def restore(model: torch.nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """`snapshot` put back, for the capture pass's rewind."""
    restore_rows(model, saved)


# -- the store ------------------------------------------------------------------------------------


@dataclass(slots=True)
class Entry:
    """One snapshot, and what it is keyed by.

    `length` is a position in the prompt: the snapshot is the state after its first `length` tokens,
    so a request that restores it forwards the prompt from `length` on. `used` is the eviction clock
    and not a timestamp -- an `int` comparison that cannot be moved by the wall clock or by a
    monotonic one that wraps.
    """

    length: int
    key: bytes
    saved: dict[str, torch.Tensor]
    nbytes: int
    used: int = 0


class PrefixCache:
    """Whole-prefix anchors on the host, evicted least-recently-used under a byte budget.

    The anchors are lengths, not blocks: `store` is told the position a snapshot was taken at and the
    entry is keyed by exactly that. An "aligned" anchor would not be free -- a position's state only
    exists at a forward boundary -- and keying by the exact length is what lets an exact repeat hit
    the end anchor rather than the block below it.

    Lookup walks the stored lengths from the longest down and stops at the first whose key matches, so
    the answer is the longest prefix of the request the store has, which is SGLang's rule
    (`RadixCache.match_prefix`) with anchors standing in for the radix tree. There is deliberately no
    partial credit: a length whose key does not match is skipped, and the walk continues below it.

    Threading is the caller's: the backend holds one request lock across a generation, and every
    method here is called from inside it.
    """

    def __init__(
        self,
        budget_bytes: int,
        max_seq_len: int,
        tag: bytes = b"",
        min_tokens: int = BLOCK_TOKENS,
    ) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes is {budget_bytes}, which is negative")
        self._budget = int(budget_bytes)
        self._max_seq_len = int(max_seq_len)
        self._seed_bytes = _seed(tag)
        self._min_tokens = int(min_tokens)
        self._by_length: dict[int, dict[bytes, Entry]] = {}
        self._bytes = 0
        self._clock = 0
        self._hits = 0
        self._misses = 0
        self._reused = 0

    # -- what the caller asks ----------------------------------------------------------------

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def bytes(self) -> int:
        """What the store holds, which is at or under the budget after every `store`."""
        return self._bytes

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._by_length.values())

    def lengths(self) -> list[int]:
        """The anchor lengths held, longest first. Diagnostics and tests; the walk is `lookup`."""
        return sorted(self._by_length, reverse=True)

    def lookup(self, ids: Iterable[int]) -> tuple[int, Entry] | None:
        """The longest stored prefix of `ids`, as `(cached_len, entry)`; `None` if there is none.

        `cached_len` is the entry's own length clamped to `len(ids) - 1`. The clamp is not defensive:
        a forward is always needed for the logits the first new token comes from, so an exact repeat
        forwards exactly one token rather than none, and that one token is the decode path at a
        position past zero -- the case the chunked-prefill suite pins separately.
        """
        # A one-token prompt has no prefix to reuse: the clamp below would leave nothing to forward.
        if len(ids) < 2 or not self._by_length:
            return None
        limit = min(len(ids), self._max_seq_len)
        if limit < self._min_tokens:
            return None
        chain = prefix_hashes(ids, self._seed_bytes)
        for length in self.lengths():
            if length > limit:
                continue
            entry = self._by_length[length].get(_key_at(ids, length, self._seed_bytes, chain))
            if entry is None:
                continue
            self._hits += 1
            self._reused += length
            self._touch(entry)
            return min(length, len(ids) - 1), entry
        self._misses += 1
        return None

    def store(self, ids: Iterable[int], length: int, saved: dict[str, torch.Tensor]) -> Entry | None:
        """Record `saved` as the state after the first `length` tokens of `ids`.

        Returns the entry, or `None` when it was not taken: shorter than the floor, longer than the
        prompt, past the context, or a payload that does not fit the budget on its own. Not storing is
        never an error -- a request that cannot be cached is a request the next forward pays for
        again, and the caller has no decision to make about it.

        The floor exists because an entry costs the window ring whatever its length: 128 slots by 512
        by 40 layers is 5 MiB, and a store of a handful of tokens would spend it on the ring. It is a
        whole block by default, which is the smallest length the chain can key in one step.
        """
        if length < self._min_tokens or length > len(ids) or length > self._max_seq_len:
            return None
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in saved.values())
        if nbytes > self._budget:
            return None
        chain = prefix_hashes(ids[:length], self._seed_bytes)
        entry = Entry(length, _key_at(ids, length, self._seed_bytes, chain), saved, nbytes)
        return self._admit(entry)

    def clear(self) -> None:
        self._by_length.clear()
        self._bytes = 0

    def stats(self) -> dict[str, int]:
        """Counters for the server's metrics, and the one shape a benchmark reads."""
        return {
            "entries": len(self),
            "bytes": self._bytes,
            "budget_bytes": self._budget,
            "hits": self._hits,
            "misses": self._misses,
            "reused_tokens": self._reused,
        }

    # -- the index ---------------------------------------------------------------------------

    def _admit(self, entry: Entry) -> Entry:
        """Put `entry` in the index and evict until the budget holds again.

        Storing the same `(length, key)` twice replaces rather than adds: the tokens are the same, so
        the payload is the same state, and a chat loop that resends its history would otherwise fill
        the budget with one conversation's copies.
        """
        entries = self._by_length.setdefault(entry.length, {})
        previous = entries.get(entry.key)
        if previous is not None:
            self._bytes -= previous.nbytes
        entries[entry.key] = entry
        self._bytes += entry.nbytes
        self._touch(entry)
        # The new entry fits on its own -- `store` refused it otherwise -- so this terminates on the
        # entries that were already here, oldest first, and never on the one just admitted.
        while self._bytes > self._budget:
            self._evict(self._oldest())
        return entry

    def _touch(self, entry: Entry) -> None:
        entry.used = self._clock
        self._clock += 1

    def _oldest(self) -> Entry:
        return min(
            (entry for entries in self._by_length.values() for entry in entries.values()),
            key=lambda entry: entry.used,
        )

    def _evict(self, entry: Entry) -> None:
        entries = self._by_length[entry.length]
        del entries[entry.key]
        if not entries:
            del self._by_length[entry.length]
        self._bytes -= entry.nbytes
