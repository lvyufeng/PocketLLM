"""The store's MiMo half: what a snapshot of this model's cache carries, and how it goes back.

:mod:`src.models.prefix_cache` is the store -- the token-chain key, the longest-prefix walk, the
byte-budget LRU, and :attr:`~src.models.prefix_cache.Entry.logits`. What is here is everything that
knows this model: one key buffer and one value buffer a layer, a write head a layer beside them, and
`geometry_tag` reading that layout off the cache so the chain's seed changes when the shapes do.

**What a snapshot is.** `MimoV2KVCache` holds `[kv_heads, slots, head_dim]` keys and
`[kv_heads, slots, v_head_dim]` values a layer, where `slots` is the layer's own geometry: a
windowed layer's buffer is `min(window, capacity)` slots and a ring, and a global layer's is the
context. A snapshot therefore carries two different things:

* **A ring whole.** A windowed layer is written modulo its length, so after a wrap the positions it
  holds are scattered across every slot -- there is no leading run of rows to cut, and the ring is a
  fixed 128 slots whatever the prompt is. All thirty-nine of them together are 6.1 MiB a rank
  against the gigabytes the nine global layers cost, so there is nothing to gain by being clever
  about them.
* **A global layer's written prefix.** Its buffer is the context appended from zero, so a prefix of
  `p` positions lives in rows `[0, p)` and the rest of the buffer is nothing. This is the cut that
  makes the store worth having: at 262144 the nine global layers are 1.41 GiB a rank and a
  twelve-hundred-token prompt is 6.6 MiB of them. Every figure here is the served shape's -- four
  ranks, the attention split along the checkpoint's own partition -- and a rank that holds the whole
  attention pays four times as much.

**Why the rows a snapshot does not carry are not a correctness problem.** A restore leaves the tail
of a global layer holding whatever the previous request wrote there, and that is sound because every
read of the cache is bounded by the layer's write head: `prefix(layer, upto)` refuses `upto` above
`written`, and the positions it returns are `[upto - slots, upto)`, all below the head. The
continuation's own appends land in that tail before anything reads it -- `append` places position
`q` at slot `q % slots`, which for a global layer is row `q` -- so the stale rows are overwritten
rather than read. The head, not the buffer, is what says which rows mean anything, which is why the
snapshot carries it: `written` below is one row a layer.

**Why a resume is not bit-exact here, and what that costs.** V4.1 can claim a resumed prefill is the
one-shot prefill with `torch.equal`, because its chunk boundaries are not a source of difference. This
model's are: the attention's gemms change shape with a chunk, the float sums reassociate, and
`tests/test_models_mimo_v2_device_attention.py::test_a_chunked_prefill_is_a_one_shot_prefill` holds
that at `atol=1e-4` rather than at zero. A resume is a chunk boundary at the stored length -- which
is generally not a multiple of `--backend-option prefill_chunk` -- so the tokens past it are the
continuation's arithmetic and not the one-shot's, to that tolerance and no more. What the store is
*not* allowed to do is change the state: the rows it restores are the rows a cold prefill of the same
tokens wrote, and `tests/test_models_mimo_v2_prefix_cache.py` holds a resume to that, at the same
tolerance the chunked-prefill test uses and no looser.
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
    "Entry",
    "PrefixCache",
    "WRITTEN",
    "extend_hash",
    "geometry_tag",
    "layer_key",
    "layer_value",
    "prefix_hashes",
    "restore_rows",
    "snapshot_rows",
]

#: The write head's own key in a snapshot: one `int64` a layer, in `MimoV2KVCache.layers` order.
#: It is not a buffer of the cache and no walk would find it, but a continuation reads it on its
#: first call -- `prefix` bounds every read by it -- so a snapshot that dropped it would answer the
#: first new token out of whatever span the previous request had left written. The same argument the
#: V4.1 half makes for carrying its Engram hash cache beside the buffers.
WRITTEN = "written"


def layer_key(layer: int) -> str:
    return f"layers.{layer}.key"


def layer_value(layer: int) -> str:
    return f"layers.{layer}.value"


def _rows_to_store(cache, layer: int) -> int:
    """The rows of `layer`'s buffers a prefix has written, or all of them if it is a ring.

    A global layer's state is its leading rows, which is where the cut is; a ring's is scattered
    across every slot it has the moment it wraps, so it is stored whole. Read off the layer's own
    geometry rather than off `slots < capacity`, because a window whose length is at or above the
    capacity is a ring that cannot wrap and is stored whole either way.
    """
    slots = cache.slots(layer)
    if cache.shape(layer).sliding_window is not None:
        return slots
    return min(slots, cache.written(layer))


def geometry_tag(cache, world_size: int, max_seq_len: int) -> bytes:
    """A tag over the geometry a snapshot was taken in, for the chain's seed.

    The store lives in the process, so this tag cannot survive a restart; what it catches is a
    process that is not the one it thinks it is -- a different `world_size`, whose attention split
    over a different number of ranks gives every buffer a different head count, or a `max_seq_len`
    that moved under the cache. Reading the shapes off the buffers rather than off the config is the
    same discipline the V4.1 half follows: the shapes are the thing that would actually be wrong, and
    at four ranks a `[4, slots, 192]` buffer restored under a `[1, slots, 192]` layout is a shape
    error only if someone checks, and a wrong answer if nobody does.
    """
    parts = [f"world={int(world_size)}", f"max_seq_len={int(max_seq_len)}"]
    for layer in cache.layers:
        parts.append(
            f"L{layer}="
            f"{'x'.join(str(size) for size in cache.key_buffer(layer).shape)}"
            f"/{'x'.join(str(size) for size in cache.value_buffer(layer).shape)}"
        )
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).digest()


def snapshot_rows(cache, *, to_host: bool = True) -> dict[str, torch.Tensor]:
    """The cache's state as a flat `name -> tensor` mapping, cut to what a prefix has written.

    Flat and named because that is what the store counts bytes over and what a test can compare a
    layer at a time; `WRITTEN` rides in the same mapping for the reason it documents.

    `to_host` is not a preference. A snapshot the store keeps has to be pageable host memory: it
    outlives the request that produced it, and the rank's card is holding a 5.65 GiB cache, a 408 MiB
    expert arena a slot and the resident rows a deployment asked for. A snapshot taken for a caller
    that puts it back inside the same call wants the opposite and there is no such caller here -- the
    MiMo runtime has no decode graphs -- so `to_host=False` exists for the measurements that take a
    snapshot without keeping it.
    """
    out: dict[str, torch.Tensor] = {}
    written = []
    for layer in cache.layers:
        rows = _rows_to_store(cache, layer)
        for name, buffer in (
            (layer_key(layer), cache.key_buffer(layer)),
            (layer_value(layer), cache.value_buffer(layer)),
        ):
            block = buffer[:, :rows]
            out[name] = block.detach().to("cpu", copy=True) if to_host else block.detach().clone()
        written.append(cache.written(layer))
    out[WRITTEN] = torch.tensor(written, dtype=torch.int64, device="cpu")
    return out


def restore_rows(cache, saved: dict[str, torch.Tensor]) -> None:
    """Copy a snapshot back over the cache's own buffers and put its write heads back.

    `copy_` into the leading rows and never a rebind: the buffers outlive every request -- one cache
    is allocated for the life of the process -- and every span the attention reads is a view of them,
    so a rebind would leave the caller building spans out of memory that this cache no longer owns.
    A layer's rows past what the snapshot carries are left exactly as they were; the module docstring
    is the argument for why that is sound rather than convenient.

    The heads go back **after** the buffers, because a snapshot is a state and not a pair of halves:
    a head moved onto a buffer that had not been copied yet would be a window in which a
    concurrent reader sees a prefix the rows do not hold. Nothing here is concurrent -- the backend
    holds one request lock across a generation -- and the order is written down anyway, since the
    next reader of this function will not be able to tell from the call sites.
    """
    heads = saved.get(WRITTEN)
    for index, layer in enumerate(cache.layers):
        for name, buffer in (
            (layer_key(layer), cache.key_buffer(layer)),
            (layer_value(layer), cache.value_buffer(layer)),
        ):
            value = saved.get(name)
            if value is None:
                continue
            buffer[:, : value.shape[1]].copy_(value)
        if heads is not None:
            cache.set_written(layer, int(heads[index].item()))
