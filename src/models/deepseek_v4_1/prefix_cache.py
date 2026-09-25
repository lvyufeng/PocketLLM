"""The store's V4.1 half: which buffers a snapshot carries, and how they are cut and put back.

:mod:`src.models.prefix_cache` is the store -- the token-chain key, the longest-prefix walk, the
byte-budget LRU, and :attr:`~src.models.prefix_cache.Entry.logits`. What is here is everything that
knows this model: `CACHE_NAMES` is the set of buffers a step writes and a later step reads, two of
them are second-axis *position* tables that a prefix cuts, the Engram hash cache rides beside them
under a key of its own, and `geometry_tag` reads the layout off the tree so the chain's seed changes
when the shapes do.

**What a snapshot is.** `CACHE_NAMES` below is the same set `graphs.snapshot` walks. Two of them are
second-axis *position* tables -- `compress_kv_cache` and `k_cache` carry one row per `compress_ratio`
positions -- so a prefix of `p` tokens has written their first `p // ratio` rows and nothing else. The
rest are stored whole: the window ring is a fixed 128 slots keyed by `position % 128`,
`kv_state`/`score_state` are the group still filling, and all three are small enough that cutting them
would be arithmetic for its own sake. Not stored at all is `freqs_cis`: it is sliced by absolute
position, so a resume needs no re-base.

**Why a resume is exact.** A restore is followed by a *continuation* forward at `start_pos = p`,
which is a chunk boundary and nothing more -- `tests/test_models_deepseek_v4_1_chunked_prefill.py`
pins every chunk boundary bit-equal to a one-shot prefill with `torch.equal`. What makes the state
match rather than merely be unread is `reset_state`: it zeroes the three position tables and refills
the compressor state, and the forward that produced the snapshot started from that same reset, so
the rows a sliced snapshot does not carry are zeros on both sides. A restore therefore reconstructs
the state a cold prefill of exactly `p` tokens leaves, not a state that is written over.

**Why the exact repeat is a sample and not a replay.** The ring and the Engram hash cache are pure
functions of the position and would survive a replay of the last token at its own position, but the
compressor is a *recurrence* over a group: a replay re-emits the group's row from a state that has
already counted the position or has already flushed it.
`test_a_snapshot_exactly_the_prompt_cannot_be_resumed_with_a_forward` asserts the divergence that
produces, so the shortcut cannot be reintroduced as an optimization.
"""

from __future__ import annotations

import hashlib

import torch

from src.models.prefix_cache import (  # noqa: F401 - the store, re-exported as this module's own
    BLOCK_TOKENS,
    Entry,
    PrefixCache,
    extend_hash,
    prefix_hashes,
)

__all__ = [
    "BLOCK_TOKENS",
    "CACHE_NAMES",
    "Entry",
    "GROUPED_AXES",
    "HASH_CACHE",
    "PrefixCache",
    "extend_hash",
    "geometry_tag",
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

# The Engram hash cache rides in a snapshot under its own key. It is a plain attribute of
# `EngramHashIds` rather than a registered buffer, so `named_buffers` never sees it and the tree
# cannot supply it -- but a continuation's first token reads the previous `max_ngram_size - 1`
# positions through it, and `reset` fills it with `DEAD`, so a restore that left it out would answer
# the first new token with a truncated n-gram. The spelling carries a dot and cannot collide with a
# buffer name.
HASH_CACHE = "engram.hash_cache"


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

    The walk is over the tree's buffers and not over the snapshot's keys, so a snapshot that carries
    something the tree does not -- `HASH_CACHE`, which is restored by whoever took it -- is passed
    over rather than looked up.

    A sliced snapshot fills the leading rows and leaves the rest alone, which is exact only because
    the caller reset first: `Attention.reset_state` zeroes the three position tables and refills the
    compressor state, and the forward that produced the snapshot started from that same reset, so the
    rows it did not carry were zeros on both sides.
    """
    for name, buffer in model.named_buffers():
        value = saved.get(name)
        if value is None:
            continue
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
