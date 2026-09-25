"""A prompt's state kept on the host, keyed by the prompt's own tokens.

Every request on either of the two runtimes that serve a chat resends its whole history, and both
of them wipe their caches at the top of the loop, so turn N pays for turns 1..N-1 again. This module
is the half of the fix that does not know which model it is serving: it keys a snapshot by the
tokens that produced it, walks the stored anchors longest-first for the longest prefix a request
shares with one already served, and evicts least-recently-used under a byte budget. What a snapshot
*is* -- which buffers, how they are cut, how they are put back -- belongs to the model, and
:mod:`src.models.deepseek_v4_1.prefix_cache` and :mod:`src.models.mimo_v2.prefix_cache` are those two
halves.

**Why the key is the tokens.** Every serving option a client can set -- thinking mode, reasoning
effort, tools, response format -- reaches the model as tokens, so keying on the rendered prompt
folds all of them in for free: a different render is a different token stream and cannot false-hit.
The key is a chain over 64-token blocks with the length mixed in, ``h_i = blake2b(h_{i-1} || length
|| block_i)``, which is vLLM's shape (``hash_request_tokens``'s block chain) and gives the same two
properties: a prefix's key is computed once and extended, and the key for a length identifies the
whole prefix below it.

**Why the seed is the geometry.** A snapshot is a set of shaped buffers, so a chain is only
meaningful inside the layout it was taken in. ``geometry_tag`` is each model's own reading of that
layout and it seeds the chain, which makes the same tokens under two geometries two keys. The store
lives in the process, so the tag cannot survive a restart and is not trying to: what it catches is a
process that is not the one it thinks it is.

**What is not borrowed is vLLM's paged KV.** Both runtimes hold pre-allocated contiguous buffers
with a partial group that is not block-aligned, so a block-granular store would be a rewrite of the
cache layout rather than a store on top of it. The unit is therefore a whole-prefix anchor: the
prompt's exact end, plus a fixed-length head anchor for the case where two conversations share a
rendered header but diverge after it.

**An exact repeat forwards nothing, and the entry says why.** A request whose prompt *is* a stored
prefix -- the second turn of a chat that has not changed, a retry -- needs the distribution its first
new token comes from and nothing else, and that row was already computed when the anchor was taken.
So :attr:`Entry.logits` carries it and the repeat is a restore and a sample. The rule that looks
cheaper, restoring the state and forwarding the prompt's last token again at its own position, is
*unsound* wherever any buffer is a recurrence rather than a function of the position; both models
here have one, and :meth:`PrefixCache.lookup` therefore does not clamp the match to
``len(ids) - 1``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = [
    "BLOCK_TOKENS",
    "EMPTY_SEED",
    "Entry",
    "HASH_BYTES",
    "PrefixCache",
    "extend_hash",
    "prefix_hashes",
    "tokens_bytes",
]

# The block the hash chain steps by. vLLM uses the same size for the same reason: it is what a
# `by_length` index is walked in, so the chain for a candidate length is usually already computed and
# only a length that is not a multiple of it costs one extra hash.
BLOCK_TOKENS = 64

# 64 bits. vLLM's bar as well -- a collision needs two prompts whose first `L` tokens hash alike, and
# a collision is not a crash: it restores a state that was produced by different tokens, which is why
# the key is a chain over *all* the blocks below `L` and not a hash of the length alone.
HASH_BYTES = 8

_DOMAIN = b"pocketllm.prefix-cache.1"


def tokens_bytes(tokens) -> bytes:
    """The ids as little-endian 32-bit words, so the hash is over the tokens and not their spelling.

    Accepts a list, an `array`, or a tensor. A tensor is moved off the card first, which costs a copy
    the size of the block: `prompt_ids` reaches the model as a Python list and a caller that has a
    tensor has already paid more than this to build it.
    """
    if isinstance(tokens, torch.Tensor):
        return (
            tokens.detach()
            .to(torch.int64)
            .cpu()
            .to(torch.int32)
            .numpy()
            .astype("<i4", copy=False)
            .tobytes()
        )
    return b"".join(int(token).to_bytes(4, "little", signed=True) for token in tokens)


# The old spelling, which the V4.1 module's own `__all__` carried before the store moved here.
_tokens_bytes = tokens_bytes


def _seed(tag: bytes = b"") -> bytes:
    """The chain's starting value. `tag` is the geometry; see each model's `geometry_tag`."""
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
    digest.update(tokens_bytes(tokens))
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


@dataclass(slots=True)
class Entry:
    """One snapshot, and what it is keyed by.

    `length` is a position in the prompt: the snapshot is the state after its first `length` tokens,
    so a request that restores it forwards the prompt from `length` on. `used` is the eviction clock
    and not a timestamp -- an `int` comparison that cannot be moved by the wall clock or by a
    monotonic one that wraps.

    `logits` is the distribution at position `length - 1`, on the host in the dtype the head produced
    it in. It is not an optimization the caller may skip: a request whose prompt is exactly `length`
    tokens has nothing left to forward, and the rule that would let it forward one token anyway is
    the replay the module docstring calls out as unsound. Every entry has one, including the head
    anchor, whose chunk is a forward *of* `ids[:p_head]` and therefore already holds the row a prompt
    of exactly that length would sample from.

    `saved` is the model's own payload and the store never looks inside it; it is counted in bytes
    and handed back to whoever took it. See each model's `snapshot_rows`/`restore_rows`.
    """

    length: int
    key: bytes
    saved: dict
    logits: torch.Tensor
    nbytes: int
    used: int = 0


def payload_bytes(saved: dict) -> int:
    """What a snapshot costs, which is the sum of its own tensors and nothing else.

    Every payload either runtime takes is a flat ``name -> tensor`` mapping -- a tensor for each
    buffer and a small one for the position counters, which a restore has to write back and so cannot
    be a plain Python int. Both halves of that are the *restore*'s requirement rather than the
    store's, which is why the store can read one number off the whole mapping and move on.
    """
    return sum(tensor.numel() * tensor.element_size() for tensor in saved.values())


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

    Threading is the caller's: each backend holds one request lock across a generation, and every
    method here is called from inside it.
    """

    def __init__(
        self,
        budget_bytes: int,
        max_seq_len: int,
        tag: bytes = b"",
        min_tokens: int = BLOCK_TOKENS,
        head_tokens: int = 0,
    ) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes is {budget_bytes}, which is negative")
        if head_tokens < 0:
            raise ValueError(f"head_tokens is {head_tokens}, which is negative")
        self._budget = int(budget_bytes)
        self._max_seq_len = int(max_seq_len)
        self._seed_bytes = _seed(tag)
        self._min_tokens = int(min_tokens)
        self._head_tokens = int(head_tokens)
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
    def max_seq_len(self) -> int:
        return self._max_seq_len

    @property
    def head_tokens(self) -> int:
        """The fixed-length anchor a prefill also stores, or 0 for an end anchor only.

        Read by the prefill rather than by the store, because it is the prefill that has to put a
        chunk boundary where the snapshot goes. It lives here because it is one of the three numbers
        that describe an anchor -- with `min_tokens` and `max_seq_len`, both of which the store also
        enforces -- and because a caller that configures the store is the caller that knows whether a
        head anchor is worth an extra chunk.
        """
        return self._head_tokens

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

        `cached_len` is the entry's own length, which is never more than `len(ids)`: the key is over
        the first `length` tokens of the request, so a match is a statement about a prefix that is
        already there. It may be *equal* to `len(ids)`, which is the exact repeat and is the reason
        there is no clamp to `len(ids) - 1` here -- the caller takes `entry.logits` and forwards
        nothing. See `Entry.logits` and the module docstring for why the one-token forward is not the
        fallback.

        Nothing is stored or cleared: `lookup` only reads the index and moves the eviction clock, so
        a caller that abandons the hit (a forward that raises, a cancelled request) leaves the store
        exactly as it found it.
        """
        # A one-token prompt has no prefix to reuse: it is shorter than the floor, either way.
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
            return length, entry
        self._misses += 1
        return None

    def store(
        self,
        ids: Iterable[int],
        length: int,
        saved: dict,
        logits: torch.Tensor,
    ) -> Entry | None:
        """Record `saved` as the state after the first `length` tokens of `ids`.

        Returns the entry, or `None` when it was not taken: shorter than the floor, longer than the
        prompt, past the context, or a payload that does not fit the budget on its own. Not storing is
        never an error -- a request that cannot be cached is a request the next forward pays for
        again, and the caller has no decision to make about it.

        The floor exists because an entry costs its fixed tables whatever its length: a ring of
        window slots and a group still filling cost the same for eight tokens as for eight thousand.
        It is a whole block by default, which is the smallest length the chain can key in one step.

        `logits` is the entry's own last row and is taken as it is: the callers are the forwards that
        produced the state, and none of them can be missing the row -- they sampled from it.
        """
        if length < self._min_tokens or length > len(ids) or length > self._max_seq_len:
            return None
        row = logits.detach().reshape(-1).to("cpu", copy=True)
        nbytes = payload_bytes(saved) + row.numel() * row.element_size()
        if nbytes > self._budget:
            return None
        chain = prefix_hashes(ids[:length], self._seed_bytes)
        entry = Entry(length, _key_at(ids, length, self._seed_bytes, chain), saved, row, nbytes)
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
