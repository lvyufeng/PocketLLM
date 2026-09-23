"""One layer's attention on a card, out of the released weights, against a KV cache.

`layers.py` is the reference and stays the reference; this module is the same
arithmetic moved onto a device, and everything it gets right it gets right because
the reference says so. Three of those are easy to move and wrong to move:

* The fused `qkv_proj` is stored as four tensor-parallel shards of `[q | k | v]`,
  so the projection is run in the checkpoint's own row order and the *output* is
  cut by `split_fused_qkv`. Reordering the weight instead is the same answer for
  2x the bytes and one more place to be wrong.
* The value scale is applied to the values before attention and before the cache,
  so a cached value has already been scaled and a second scale is a bug.
* The sink is a softmax column, not a logit bias, and the row maximum is taken
  over the extended row. `blocked_attention` starts its running maximum at the
  sink and its running denominator at one, which is that same arithmetic without
  ever materialising the column.

What is deliberately not here yet: tensor parallelism, expert parallelism, and a
CUDA kernel. The attention is torch, so it is a correctness artifact and a baseline
rather than a performance one, and it is two implementations of one softmax: a single
pass while the scores fit in memory, which is every decode step, and an online block
loop for the prefill where they do not, which is what keeps 256k affordable at all.

The cache is two buffers a layer. A sliding-window layer's window is 128, so its
buffer is 128 slots and a ring; a global layer's is the whole context, appended.
That is not a tuning choice but what the two families read: a windowed layer at
256k never looks further back than 128, and a global layer looks at everything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from src.models.mimo_v2.config import MimoV2AttentionShape, MimoV2TextConfig
from src.models.mimo_v2.layers import (
    apply_partial_rope,
    build_rope_cos_sin,
    build_rope_inv_freq,
    split_fused_qkv,
)
from src.models.mimo_v2.quant import QKV_SHARDS

__all__ = [
    "AttentionStats",
    "MimoV2DeviceAttention",
    "MimoV2KVCache",
    "attention",
    "blocked_attention",
    "DEFAULT_TILE_SCORES",
    "fused_qkv_row_order",
    "group_qkv_order",
    "shard_shape",
    "single_pass_attention",
    "tile_step",
]


def fused_qkv_row_order(shape: MimoV2AttentionShape, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """The permutation that turns the stored `qkv_proj` rows into `[q | k | v]`.

    Exists for callers that want the weight itself reordered -- a kernel that wants
    three contiguous matrices, or a single-rank path that wants to skip the
    four-way split per call. `MimoV2DeviceAttention` uses it as a gather on the one
    row of activations instead: `qkv.index_select(-1, order)` is the same three
    tensors `split_fused_qkv` builds, from one kernel rather than the nineteen
    slices and six concatenations it takes.

    The permutation is the one `split_fused_qkv` implies, read off the same
    description rather than derived a second time: shard `i` holds
    `[q_i | k_i | v_i]` at `stride * i`, and the canonical order gathers the four
    query sections, then the four (or one) key sections, then the value ones.
    """
    groups = 4
    if shape.qkv_row_layout == "contiguous":
        return torch.arange(shape.qkv_out, dtype=dtype)
    if shape.qkv_row_layout not in ("tp4_interleaved", "tp4_interleaved_vk"):
        raise ValueError(
            f"no row order is known for {shape.qkv_row_layout!r}; the released checkpoint's "
            f"fused projection is 'tp4_interleaved'"
        )
    q_head = shape.q_size // groups
    k_head = shape.k_size // groups
    v_head = shape.v_size // groups
    stride = q_head + k_head + v_head
    # Where each of the three sections sits inside a group, in the canonical `[q | k | v]` order
    # the name promises. `tp4_interleaved_vk` stores the value section first, so its key section
    # is at `q_head + v_head` and the widths are not interchangeable.
    sections = (
        ((0, q_head), (q_head, k_head), (q_head + k_head, v_head))
        if shape.qkv_row_layout == "tp4_interleaved"
        else ((0, q_head), (q_head + v_head, k_head), (q_head, v_head))
    )
    rows = []
    for lo, width in sections:
        for group in range(groups):
            base = group * stride + lo
            rows.append(torch.arange(base, base + width, dtype=dtype))
    return torch.cat(rows)


def shard_shape(
    shape: MimoV2AttentionShape, shard: int, shards: int
) -> MimoV2AttentionShape:
    """One share's geometry: a quarter of the query heads, the key heads and the widths.

    The checkpoint's fused projection is stored as `shards` groups of `[q | k | v]` -- that is
    what `tp4_interleaved` says and what `quant.QKV_SHARDS` records -- so a share is a contiguous
    quarter of the projection's rows, and the query heads, key heads and value heads those rows
    hold go with it. The division is exact for both families of the released checkpoint:

    * a global layer is 64 query heads over 4 key heads, so 16 query heads share one key head and
      a quarter is 16 query heads with 1 key head -- the groups line up with the key heads;
    * a sliding-window layer is 64 over 8, so 16 query heads span 2 key heads and a quarter is 16
      with 2 -- still no group boundary is crossed.

    `num_key_value_groups` is unchanged and that is the whole reason the split is a split: a
    share's 16 query heads divide over its own key heads exactly as the layer's 64 divide over its
    4, so the score computation, the softmax and the sink are the same arithmetic on fewer heads.

    `o_in` becomes *this share's* width -- the rows of `pre_o` a share produces -- and the layer's
    own `o_in` is what the projection after the join reads. `qkv_row_layout` becomes `"contiguous"`
    because inside a group the three sections are stored in that order; the permutation out of a
    group's stored order is `group_qkv_order`'s, and it is the identity for `tp4_interleaved`.
    """
    if shards < 1:
        raise ValueError(f"shards must be at least 1, got {shards}")
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} is not a share of {shards}")
    if shards == 1:
        return shape
    for name in ("num_q_heads", "num_kv_heads", "q_size", "k_size", "v_size"):
        if getattr(shape, name) % shards:
            raise ValueError(
                f"layer {shape.layer_idx} ({shape.family}): {name} is "
                f"{getattr(shape, name)} and does not divide into {shards} shares"
            )
    heads = shape.num_q_heads // shards
    return replace(
        shape,
        num_q_heads=heads,
        num_kv_heads=shape.num_kv_heads // shards,
        q_size=shape.q_size // shards,
        k_size=shape.k_size // shards,
        v_size=shape.v_size // shards,
        qkv_out=(shape.q_size + shape.k_size + shape.v_size) // shards,
        o_in=heads * shape.v_head_dim,
        qkv_row_layout="contiguous",
    )


def group_qkv_order(
    shape: MimoV2AttentionShape, shard: int, shards: int, dtype: torch.dtype = torch.int64
) -> torch.Tensor | None:
    """The permutation from a share's stored `qkv` rows into that share's `[q | k | v]`.

    `None` when there is nothing to permute, which is the case a caller should hope for: the
    forward then takes `split_fused_qkv`'s three-way cut and pays no gather at all, where a
    permutation is an `index_select` of the whole fused row once a layer.

    Three cases, and they are the three the checkpoint and its readers make:

    * **whole, `contiguous`** -- nothing to do.
    * **whole, `tp4_interleaved`** -- the layer's own permutation, `fused_qkv_row_order`, applied to
      the activations. This is the released reading and the reason the helper exists.
    * **a share of `tp4_interleaved`** -- the identity. A group *is* `[q | k | v]` contiguously, so
      the quarter of the projection this rank holds is already in the order the split wants, and
      the four-way permutation the whole tensor needs is exactly what a share does not.
    * **a share of `tp4_interleaved_vk`** -- the key and value sections swap inside the group:
      `[q | v | k]` stored, `[q | k | v]` wanted.
    """
    if shards == 1:
        order = fused_qkv_row_order(shape, dtype)
        return None if bool(torch.equal(order, torch.arange(shape.qkv_out, dtype=dtype))) else order
    if shape.qkv_row_layout == "contiguous":
        raise ValueError(
            f"layer {shape.layer_idx}: a `contiguous` fused projection is not stored in shares "
            f"and cannot be read as one; its first quarter is all query and no key"
        )
    if shape.qkv_row_layout not in ("tp4_interleaved", "tp4_interleaved_vk"):
        raise ValueError(f"no share order is known for {shape.qkv_row_layout!r}")
    q_head = shape.q_size // shards
    k_head = shape.k_size // shards
    v_head = shape.v_size // shards
    if shape.qkv_row_layout == "tp4_interleaved":
        return None
    rows = (
        list(range(q_head))
        + list(range(q_head + v_head, q_head + v_head + k_head))
        + list(range(q_head, q_head + v_head))
    )
    return torch.tensor(rows, dtype=dtype)


def _qkv_weight(
    checkpoint, layer: "MimoV2DeviceAttention", key: str, dtype: torch.dtype
) -> torch.Tensor:
    """The fused `qkv_proj`, whole or this rank's share of it.

    A share is read *as* a share rather than read whole and sliced, because the weight is FP8 with
    its scale blocked inside each share (`quant._dequant_fp8_block_sharded`): asking the loader for
    one share dequantizes a quarter of the rows and moves a quarter of the bytes, and the rows it
    produces are the rows a full read would have produced for them.
    """
    if layer.shards == 1:
        return checkpoint.dense_tensor(key, dtype, layer.device)
    return checkpoint.dense_tensor(
        key, dtype, layer.device, shard=layer.shard, shards=layer.shards
    )


@dataclass(frozen=True)
class AttentionStats:
    """What the last attention call actually did.

    `blocks` is a launch count and `pairs` is work: the number of query-key products
    the call evaluated, against `full_pairs`, which is what a dense pass over the same
    bounds would have evaluated. For a global layer the two are equal, and for a
    windowed layer the ratio is what the bounds bought -- which is the only number
    here that says anything about the skipping.
    """

    path: str
    blocks: int
    pairs: int
    full_pairs: int

    @property
    def saving(self) -> float:
        return self.full_pairs / self.pairs if self.pairs else 0.0


def _check_bounds(query, key, value, lower, upper, sink, *, validate: bool = True):
    """The checks both attention paths make, and the geometry they both fold with.

    **The last two checks read the bounds back to the host, and that is a device sync.** They are
    worth it for a caller that hands in bounds it computed somewhere else, which is every direct
    caller and every test; they are not worth it for the model, which derives its bounds from
    `prefix_len` and `sequence` two functions above and has already established both facts about
    them in Python -- so `validate=False` skips exactly those two and keeps every shape check.
    Measured on a four-rank decode: 48 layers x 2 round trips are 96 of the 145
    `cudaStreamSynchronize` calls a token makes, and a token is 235 ms.
    """
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError("attention takes [heads, sequence, width] tensors")
    heads, queries, head_dim = query.shape
    kv_heads = key.shape[0]
    keys = key.shape[1]
    if value.shape[0] != kv_heads or value.shape[1] != keys:
        raise ValueError(
            f"key is {tuple(key.shape)} and value {tuple(value.shape)}; they are one key a value"
        )
    if kv_heads == 0 or heads % kv_heads:
        raise ValueError(f"{heads} query heads do not group over {kv_heads} key heads")
    if key.shape[-1] != head_dim:
        raise ValueError(f"query width {head_dim} and key width {key.shape[-1]} differ")
    if lower.shape != (queries,) or upper.shape != (queries,):
        raise ValueError("the bounds are one entry a query")
    if validate:
        if keys and int(upper.max()) >= keys:
            raise ValueError(
                f"a query is allowed key {int(upper.max())} of {keys}; the bounds index the "
                f"concatenated key tensor, not the chunk"
            )
        if bool((upper < lower).any()):
            raise ValueError("a query's upper bound is below its lower bound")
    if not keys and sink is None:
        raise ValueError("an empty key tensor leaves nothing to attend to and no sink to absorb it")
    return heads, queries, head_dim, kv_heads, keys, heads // kv_heads


#: Above this many keys the single pass stops folding the query groups into a batch
#: dimension and walks the key heads instead. A batch dimension of one expanded over
#: sixteen is not a batch dimension to cuBLAS, it is a copy: at 65k keys the folded
#: product is 162 ms and four per-head gemms are 5.8 ms. Below it the fold is the
#: cheaper of the two, because the loop pays a launch a head and a window is 128 keys.
FOLD_KEYS = 1024


def _probabilities(scores: torch.Tensor, visible: torch.Tensor, column: torch.Tensor | None):
    """The masked, sunk softmax over the last axis of a float32 score block.

    `column` is the sink, already broadcast to `[rows, queries, 1]`, and it enters the
    same way it does in `blocked_attention`: as part of the row maximum, and as one
    extra share of the denominator that the output never sees. That is the reference's
    concatenated column and its `probs[..., :-1]`, without the column.
    """
    scores = scores.masked_fill(~visible, float("-inf"))
    if column is None:
        running_max = scores.amax(dim=-1, keepdim=True)
    else:
        running_max = torch.maximum(scores.amax(dim=-1, keepdim=True), column)
    probabilities = torch.exp(scores - running_max)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    if column is not None:
        denominator = denominator + torch.exp(column - running_max)
    return probabilities / denominator


def single_pass_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    scaling: float,
    sink: torch.Tensor | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, AttentionStats]:
    """The same softmax as `blocked_attention`, materialised in one piece.

    Cheaper than the block loop whenever the scores fit in memory, which is every
    decode step and any short prefill, because the loop's advantage is memory and its
    cost is a launch a block: sixty-five blocks of a few microseconds each is a
    millisecond of nothing. The caller's budget is on the product, not the sequence, so
    a decode step at 256k -- 16.7M scores, 67 MiB in float32 -- takes this path and a
    prefill chunk at 256k does not.

    Two ways to write the product, and the key count picks (`FOLD_KEYS`). Both are the
    reference's arithmetic, including the sink as a concatenated column and the row
    maximum taken over the extended row. What neither does that the loop does is skip:
    a masked entry is computed and then discarded, so the pair count reported is the
    dense one.

    `validate` is `_check_bounds`'s two value checks; see there for why a caller that
    built its own bounds turns them off.
    """
    heads, queries, head_dim, kv_heads, keys, groups = _check_bounds(
        query, key, value, lower, upper, sink, validate=validate
    )
    width = value.shape[-1]

    positions = torch.arange(keys, device=query.device)
    visible = (positions.unsqueeze(0) >= lower.unsqueeze(1)) & (
        positions.unsqueeze(0) <= upper.unsqueeze(1)
    )
    flat_key = key.to(torch.float32)
    flat_value = value.to(torch.float32)

    if keys <= FOLD_KEYS:
        scores = (
            torch.matmul(
                query.reshape(kv_heads, groups, queries, head_dim).to(torch.float32),
                flat_key.transpose(1, 2).unsqueeze(1),
            )
            * scaling
        )
        column = (
            None
            if sink is None
            else sink.reshape(kv_heads, groups, 1, 1)
            .to(torch.float32)
            .expand(kv_heads, groups, queries, 1)
        )
        probabilities = _probabilities(scores, visible.unsqueeze(0).unsqueeze(0), column)
        out = torch.matmul(probabilities, flat_value.unsqueeze(1)).reshape(heads, queries, width)
    else:
        out = torch.zeros((heads, queries, width), dtype=torch.float32, device=query.device)
        for head in range(kv_heads):
            rows = slice(head * groups, (head + 1) * groups)
            scores = (
                torch.matmul(query[rows].to(torch.float32), flat_key[head].transpose(0, 1))
                * scaling
            )
            column = (
                None
                if sink is None
                else sink[rows].to(torch.float32).reshape(groups, 1, 1).expand(groups, queries, 1)
            )
            probabilities = _probabilities(scores, visible, column)
            out[rows] = torch.matmul(probabilities, flat_value[head])

    pairs = heads * queries * keys
    return out.to(query.dtype), AttentionStats("single", 1 if keys else 0, pairs, pairs)


def blocked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    scaling: float,
    sink: torch.Tensor | None = None,
    block: int = 1024,
    row_step: int | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, AttentionStats]:
    """Softmax attention where query `i` sees keys `lower[i] .. upper[i]`, inclusive.

    `query` is `[heads, queries, head_dim]`, `key`/`value` are `[kv_heads, keys, *]`,
    and the two bounds are `[queries]` and non-decreasing. Being non-decreasing is what
    makes a key block cheap: it is answered by a contiguous *slice* of the query rows,
    found with two binary searches. A causal layer's slice stops at the diagonal, which
    is half the grid; a windowed layer's starts a window below the block, so its loop
    visits a block plus a window and not the whole chunk. A global layer's bounds are
    `[0, i]` and its slices are the prefix below each row, which is the quadratic case
    and honestly is one.

    The query heads are grouped over the key heads rather than repeated: the model is
    64 query heads to 4 or 8 key heads, and the reference's `repeat_kv` would
    materialise 64 heads of keys -- 6.4 GiB at 256k, 8x the cache itself. Folded into
    `[kv_heads, groups, ...]` and broadcast against one key head, the same product
    costs nothing extra and the answer is identical because the head order is
    key-major.

    The softmax is online (a running maximum and a running denominator), so the block
    loop holds no `[queries, keys]` tensor and 256k costs memory like 128 does. The
    sink enters as the initial maximum and as an initial denominator of one, which is
    the reference's concatenated column without the column: with the running maximum
    started at the sink, `exp(sink - max)` is exactly one whenever the sink still
    dominates.

    **`row_step` is the memory bound and `block` is not.** The score tile is
    `[kv_heads, groups, rows_in_slice, block]` float32 and its width is the *chunk*, so a
    commit at 256k wants a chunk that fits and a chunk is what a caller has: 2048 rows at a
    thousand keys is 536 MiB of float32 before the exponentials, and three widths of it are
    live at once. Splitting the *rows* is what bounds that, and it is arithmetic rather than
    an approximation: a row still sees its own key blocks in the same order with the same
    running maximum, and the only thing that changes is the row count of a gemm -- which is
    a summation order inside a dot product, so the answer moves by a rounding and not by a
    tolerance. Measured against the whole-slice loop at `[8 heads, 2 kv, 37 rows, 211 keys]`
    the largest difference any step below made was 4.5e-08 on outputs of order 0.05.
    `None` keeps the whole slice, which is what every caller but a long-context prefill
    wants.
    """
    heads, queries, head_dim, kv_heads, keys, groups = _check_bounds(
        query, key, value, lower, upper, sink, validate=validate
    )
    width = value.shape[-1]
    span = queries if row_step is None else max(1, min(int(row_step), queries))

    flat_out = torch.zeros((heads, queries, width), dtype=torch.float32, device=query.device)
    if sink is None:
        flat_max = torch.full((heads, queries), float("-inf"), dtype=torch.float32, device=query.device)
        flat_sum = torch.zeros((heads, queries), dtype=torch.float32, device=query.device)
    else:
        # A `clone` and not a `contiguous`: with one query row the expansion of a
        # `[heads, 1]` sink is already contiguous, so `contiguous()` returns the sink
        # itself and the running maximum below writes into the layer's own sink
        # parameter -- which is silent, lasts one call, and only shows up in the next.
        flat_max = sink.reshape(heads, 1).to(torch.float32).expand(heads, queries).clone()
        flat_sum = torch.ones((heads, queries), dtype=torch.float32, device=query.device)

    # Every accumulator is written and read through the grouped view, so a block
    # touches a contiguous run of rows in all three at once.
    out = flat_out.view(kv_heads, groups, queries, width)
    running_max = flat_max.view(kv_heads, groups, queries)
    running_sum = flat_sum.view(kv_heads, groups, queries)
    folded_query = query.reshape(kv_heads, groups, queries, head_dim).to(torch.float32)

    seen_blocks = 0
    pairs = 0
    # The query rows that can see any key in [start, stop) are `upper >= start` and
    # `lower <= stop - 1`, and both bounds are sorted, so they are two searches. **Both are taken
    # for every block at once and read back once**, because a search per block per bound is a
    # device-to-host round trip and a round trip here stalls the pipeline on the very kernel that
    # is being queued next: a 64k chunk against a 1k block is 128 of them a layer, and one
    # `searchsorted` over the block starts is the same two answers in one read.
    starts = torch.arange(0, keys, block, device=query.device)
    stops = torch.clamp(starts + block, max=keys)
    edges = torch.stack(
        [
            torch.searchsorted(upper, starts),
            torch.searchsorted(lower, stops - 1, right=True),
        ]
    ).tolist()
    for index, start in enumerate(range(0, keys, block)):
        stop = min(start + block, keys)
        first, last = edges[0][index], edges[1][index]
        if last <= first:
            continue
        seen_blocks += 1
        pairs += heads * (last - first) * (stop - start)

        rows_q_all = folded_query[:, :, first:last]
        block_k = key[:, start:stop].to(torch.float32)
        block_v = value[:, start:stop].to(torch.float32)
        positions = torch.arange(start, stop, device=query.device)

        # The row slices of this key block, which are disjoint and cover it exactly. The key and
        # value conversions above are hoisted out of the loop and not repeated per slice, so a
        # narrower step costs the tile and nothing else.
        for lo in range(first, last, span):
            hi = min(lo + span, last)
            rows_q = rows_q_all[:, :, lo - first : hi - first]
            # [kv_heads, groups, rows, head_dim] x [kv_heads, 1, head_dim, block]
            scores = torch.matmul(rows_q, block_k.transpose(1, 2).unsqueeze(1)) * scaling

            visible = (positions.unsqueeze(0) >= lower[lo:hi].unsqueeze(1)) & (
                positions.unsqueeze(0) <= upper[lo:hi].unsqueeze(1)
            )
            scores = scores.masked_fill(~visible.unsqueeze(0).unsqueeze(0), float("-inf"))

            block_max = scores.amax(dim=-1)
            previous = running_max[:, :, lo:hi]
            merged = torch.maximum(previous, block_max)
            # A row whose every key in this block is masked has a `-inf` maximum, and
            # `exp(-inf - -inf)` is a nan rather than a zero. Such a row keeps its
            # running state instead: the substitution only ever feeds the exponentials.
            infinite = torch.isinf(merged)
            finite = torch.where(infinite, torch.zeros_like(merged), merged)
            alpha = torch.exp(previous - finite).masked_fill(infinite, 1.0)
            probs = torch.exp(scores - finite.unsqueeze(-1))
            probs = torch.where(infinite.unsqueeze(-1), torch.zeros_like(probs), probs)

            running_sum[:, :, lo:hi] = running_sum[:, :, lo:hi] * alpha + probs.sum(dim=-1)
            out[:, :, lo:hi] = out[:, :, lo:hi] * alpha.unsqueeze(-1) + torch.matmul(
                probs, block_v.unsqueeze(1)
            )
            running_max[:, :, lo:hi] = merged

    flat_out = flat_out / flat_sum.unsqueeze(-1)
    return flat_out.to(query.dtype), AttentionStats("blocked", seen_blocks, pairs, heads * queries * keys)


#: Scores the block loop's tile may hold when the caller does not name one. A count and not a
#: byte count, so it reads in the same units as `attention`'s `budget`: 2**26 of them is 256 MiB
#: of float32 scores and about half a gibibyte with their exponentials alive beside them, which
#: fits on a card holding a 256k cache.
#:
#: **A constant, and not the card's free memory.** That was tried and removed, and the reason is
#: the lockstep rather than the memory: `mem_get_info` is read per *rank*, four ranks of a layer
#: therefore pick four different steps and do four different amounts of work, and every routed
#: layer's `all_reduce` charges all four of them the slowest one's time. Measured on a 256k prompt:
#: the four ranks' attention columns spread **19.47 to 25.47 ms a token** where the run at this
#: fixed step spread 15.03 to 15.53, and the prompt took **8246.7 s against 5424.9** -- 52% slower
#: for a tile that was supposed to be a memory bound and not a schedule.
#:
#: The width is measured in the same place and in the same direction: 2**23, which is 128 rows a
#: step here, took **6446.8 s** and 19.36 ms of attention a token on that prompt. So this is wide
#: deliberately and not by default -- a narrow step pays for its masks and its launches once per
#: step, and the memory it saves is memory the run's own cache has already spoken for.
#:
#: A step that is a constant is the same step on every rank, and it is also the only kind a CUDA
#: graph could capture. It is not a determinism knob in the other direction either: `tile_step`'s
#: split is a rounding order and not an approximation, so two runs that pick different steps
#: produce the same answer and different timings.
DEFAULT_TILE_SCORES = 1 << 26


def tile_step(kv_heads: int, groups: int, block: int, scores: int) -> int:
    """The query rows one key block may be scored against at once, for a tile of `scores` scores.

    `scores` is a count and not a byte count, in the same units as `attention`'s `budget`, so the
    two knobs read alike. The tile `blocked_attention` holds is `kv_heads x groups x rows x block`,
    so the row that many scores buy is `scores / (kv_heads * groups * block)` and never less than
    one -- a caller can make the tile too small to be a tile, and the loop then takes a row at a
    time, which is slow and still correct.
    """
    per_row = max(1, int(kv_heads) * int(groups) * int(block))
    return max(1, int(scores) // per_row)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    scaling: float,
    sink: torch.Tensor | None = None,
    block: int = 1024,
    budget: int = 1 << 25,
    tile_budget: int | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, AttentionStats]:
    """One piece, or a block loop, whichever the scores can afford.

    `budget` is a count of scores and not of bytes, so one number covers both dtypes:
    at the default a call may materialise 33.5M of them, which is 134 MiB of float32 and
    the whole reason the block loop exists above it.

    Both paths are the same softmax and the tests hold them to the same answer; what
    decides is that a decode step is one query against up to 256k keys, which is a
    single gemm and no loop worth writing, and a prefill chunk is thousands of queries
    against the same keys, which is a grid nothing should materialise at once.

    `tile_budget` bounds the *tile* the block loop holds at one time, which `budget`
    does not: a prefill chunk passes `budget` on its very first key block and then runs
    a loop whose tile is `kv_heads x groups x rows x block`, a number that grows with the
    chunk and is the one thing at 256k that a card can run out of. It defaults to
    `DEFAULT_TILE_SCORES` and **not** to the card's free memory, which was tried and
    removed; the constant's own comment has the measurement and the reason, and it is a
    schedule and not a memory bound. The row count of a call is the caller's and the
    block is the loop's, so the price of staying under the budget is a narrower *step*
    and not a narrower chunk -- see `blocked_attention`, where the split costs a
    rounding-order difference and nothing else.
    """
    if query.shape[0] * query.shape[1] * key.shape[1] <= budget:
        return single_pass_attention(
            query, key, value, lower, upper, scaling=scaling, sink=sink, validate=validate
        )
    kv_heads = key.shape[0]
    groups = query.shape[0] // kv_heads
    scores = DEFAULT_TILE_SCORES if tile_budget is None else int(tile_budget)
    return blocked_attention(
        query,
        key,
        value,
        lower,
        upper,
        scaling=scaling,
        sink=sink,
        block=block,
        row_step=tile_step(kv_heads, groups, block, scores),
        validate=validate,
    )


class MimoV2KVCache:
    """Key and value states for a stack of layers, one buffer a layer.

    A windowed layer's buffer is `window` slots and a ring: it is written modulo its
    length, so a ring of 128 holds the last 128 positions whatever the sequence
    length, and a prefill chunk longer than the window simply overwrites it. That is
    correct rather than lossy, because a query only ever reads the window below
    itself -- and the queries *inside* the chunk are answered from the chunk's own
    key and value tensors, never from the cache, which is also why appending after
    the attention rather than before changes nothing.

    A global layer's buffer is the context and is appended under a hard limit, since
    there is no window to make an overwrite harmless.

    `shards`/`shard` hold a *share* of the key and value heads, which is what a rank of a split
    attention needs and nothing else: the heads a rank attends over are the heads it has to keep,
    so a four-way split divides the cache by four -- 5.65 GiB of global layers at 256k becoming
    1.4 -- and every rank's `prefix` is the same number of *positions* over its own heads.
    """

    def __init__(
        self,
        config: MimoV2TextConfig,
        capacity: int,
        layers: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        *,
        shards: int = 1,
        shard: int = 0,
    ) -> None:
        self.config = config
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        self.shards = int(shards)
        self.shard = int(shard)
        self.layers = tuple(range(config.num_hidden_layers)) if layers is None else tuple(layers)
        if not self.layers:
            raise ValueError("a cache with no layers holds nothing")

        self._shapes: dict[int, MimoV2AttentionShape] = {}
        self._slots: dict[int, int] = {}
        self._key: dict[int, torch.Tensor] = {}
        self._value: dict[int, torch.Tensor] = {}
        self._written: dict[int, int] = {}
        for layer in self.layers:
            shape = shard_shape(config.attention(layer), self.shard, self.shards)
            slots = self.capacity if shape.sliding_window is None else min(
                int(shape.sliding_window), self.capacity
            )
            self._shapes[layer] = shape
            self._slots[layer] = slots
            self._key[layer] = torch.zeros(
                (shape.num_kv_heads, slots, shape.head_dim), dtype=dtype, device=self.device
            )
            self._value[layer] = torch.zeros(
                (shape.num_kv_heads, slots, shape.v_head_dim), dtype=dtype, device=self.device
            )
            self._written[layer] = 0

    def __contains__(self, layer: int) -> bool:
        return layer in self._key

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def memory_bytes(self) -> int:
        return sum(
            self._key[layer].numel() * self._key[layer].element_size()
            + self._value[layer].numel() * self._value[layer].element_size()
            for layer in self.layers
        )

    @property
    def context_capacity(self) -> int:
        """How far the *global* layers can go; a windowed layer never binds."""
        return self.capacity

    def slots(self, layer: int) -> int:
        return self._slots[layer]

    def shape(self, layer: int) -> MimoV2AttentionShape:
        """What this cache holds for `layer`: its geometry, or this rank's *share* of it.

        The cache is the authority on this and a caller that fills or reads a cache has to ask it
        rather than the config: when the attention is split over the ranks, a rank's buffer holds
        `1 / shards` of the layer's key and value heads, and a caller that wrote the layer's own
        count would be refused by `append` -- which is the good outcome, and what the check is for.
        """
        if layer not in self._shapes:
            raise KeyError(f"this cache holds no layer {layer}; it holds {sorted(self._shapes)}")
        return self._shapes[layer]

    def written(self, layer: int) -> int:
        """Positions appended so far, whether or not the ring still holds them."""
        return self._written[layer]

    def length(self, layer: int) -> int:
        """Positions the ring currently holds -- `min(written, slots)`."""
        return min(self._written[layer], self._slots[layer])

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> int:
        """Write `[kv_heads, n, *]` key and value states and return the new length.

        Called with the *post-RoPE, post-value-scale* states, which is what a caller
        has just computed and what a later call must read back.
        """
        if layer not in self._key:
            raise KeyError(f"layer {layer} is not in this cache; it holds {sorted(self._key)}")
        shape = self._shapes[layer]
        if key.shape[0] != shape.num_kv_heads or key.shape[1] != value.shape[1]:
            raise ValueError(
                f"layer {layer} takes {shape.num_kv_heads} key heads and one value a key, got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        slots = self._slots[layer]
        start = self._written[layer]
        if slots == self.capacity and start + key.shape[1] > slots:
            raise ValueError(
                f"layer {layer} is a global-attention layer and the cache holds "
                f"{self.capacity} positions; appending {key.shape[1]} at {start} runs past it"
            )
        if key.shape[1] > slots:
            # More positions than the ring has slots: only the last `slots` of them
            # can survive, and they are exactly the ones a later query can read. The
            # trim is what keeps the slot indices distinct, so `index_copy_` is not
            # asked to write the same slot twice in one call.
            trim = key.shape[1] - slots
            key, value = key[:, trim:], value[:, trim:]
            start += trim
        destination = torch.arange(start, start + key.shape[1], device=self.device) % slots
        self._key[layer].index_copy_(1, destination, key.to(self.dtype))
        self._value[layer].index_copy_(1, destination, value.to(self.dtype))
        self._written[layer] = start + key.shape[1]
        return self._written[layer]

    def prefix(self, layer: int, upto: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The last `min(upto, slots)` positions below `upto`, in time order.

        The ring is unrolled here rather than in the attention, so a caller gets a
        contiguous tensor and never has to know the buffer wrapped. `upto` is a
        position and not an index: it is what a chunk's `start_pos` is.

        A ring that has not wrapped is not unrolled but *sliced*, because the unrolling is
        an `index_select` and an `index_select` is a copy: at 256k a global layer's prefix
        is 400 MiB of keys and 267 MiB of values, and the caller concatenates the chunk
        onto it -- so a copy here is a copy over and above the one the caller is about to
        make, taken at the shallowest moment of the card's headroom. Unwrapped, the wanted
        indices are `arange(0, upto)` and the slice is the same tensor for nothing.
        """
        if layer not in self._key:
            raise KeyError(f"layer {layer} is not in this cache; it holds {sorted(self._key)}")
        self._check_readable(layer, upto)
        slots = self._slots[layer]
        start = max(0, upto - slots)
        if start == 0:
            # Nothing has been overwritten yet, so time order is slot order and the prefix is a
            # window onto the buffer rather than a rearrangement of it.
            return self._key[layer][:, :upto], self._value[layer][:, :upto], upto
        wanted = torch.arange(start, upto, device=self.device) % slots
        return (
            self._key[layer].index_select(1, wanted),
            self._value[layer].index_select(1, wanted),
            upto - start,
        )

    def append_and_span(
        self, layer: int, key: torch.Tensor, value: torch.Tensor, *, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Append a chunk and hand back the span an attention is about to read, if it is a view.

        The span is `[0, start_pos + n)`: the prefix below the chunk with the chunk's own keys on
        the end of it, which is what a caller would build with a `cat` of `prefix(layer,
        start_pos)` and its own `key`. While nothing has wrapped, the buffer *is* that span in
        slot order -- the chunk's keys land at slots `[start_pos, start_pos + n)`, which is where
        the concatenation would have put them -- so appending first and reading back is the same
        bytes in the same order with no copy of the prefix at all. At 262144 with the attention
        split over four ranks a global layer's prefix is 160 MiB of keys and values, nine layers,
        read *and* written every token; it was 640 MiB a layer before the split.

        `None` says the buffer is not that span in order and the caller should read its prefix
        before appending as it always has: a ring that has wrapped holds the *last* `slots`
        positions, so the span would be a rearrangement (an `index_select`, a copy) and the bounds
        would have to be rebased onto a shorter key tensor, which is where the arithmetic moves.
        Nothing is appended in that case, and the caller's own append still has to happen.

        The read is checked here for the reason `prefix` checks it: a span that starts above what
        the cache has written would otherwise be answered with the zeros the buffer was allocated
        with.
        """
        self._check_readable(layer, start_pos)
        upto = start_pos + key.shape[1]
        if upto > self._slots[layer]:
            return None
        self.append(layer, key, value)
        return self._key[layer][:, :upto], self._value[layer][:, :upto]

    def _check_readable(self, layer: int, upto: int) -> None:
        if upto > self._written[layer]:
            raise ValueError(
                f"layer {layer} has {self._written[layer]} positions cached and {upto} were asked "
                f"for; a chunk cannot read positions it has not appended"
            )

    def reset(self) -> None:
        """Forget every position without giving the memory back."""
        for layer in self.layers:
            self._written[layer] = 0


class MimoV2DeviceAttention:
    """One layer's attention on a card: the released weights, both families.

    The interface matches `layers.MimoV2DecoderLayer.attend` down to the returned
    dictionary, minus the probabilities -- the block loop never holds them, which is
    the point of it -- and minus `qkv_in`, which is the stream the caller passed in.
    So a caller can diff one against the other at three boundaries instead of at the
    output alone. `hidden_states` is the *post* input-norm stream, as it is there, and
    the returned tensors are `[sequence, width]` whether the input carried a batch axis
    or not.

    The batch axis is not carried: a decode step and a prefill chunk are both one
    sequence, and every tensor in the model is `[sequence, width]`.
    """

    #: What the last call's attention did -- which path it took, and how much of the
    #: grid a dense pass would have walked against what it walked.
    last_stats: AttentionStats = AttentionStats("none", 0, 0, 0)

    def __init__(
        self,
        checkpoint,
        layer_idx: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block: int = 1024,
        budget: int = 1 << 25,
        tile_budget: int | None = None,
        *,
        shard: int = 0,
        shards: int = 1,
        gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        self.config: MimoV2TextConfig = checkpoint.layer
        #: The layer's own geometry, and the geometry *this rank* computes. `shards=1` is the
        #: whole layer and is the control; `shards=4` is one share of the checkpoint's own
        #: partition. `MimoV2AttentionShape`, `group_qkv_order` and `ep.attention_shards` are
        #: where the split is defined.
        self.full_shape: MimoV2AttentionShape = self.config.attention(self.layer_idx)
        self.shard = int(shard)
        self.shards = int(shards)
        if self.shards < 1 or not 0 <= self.shard < self.shards:
            raise ValueError(
                f"shard {shard} is not a share of {shards} for layer {self.layer_idx}"
            )
        self.gather = gather
        if self.shards > 1 and gather is None:
            raise ValueError(
                f"layer {self.layer_idx} is split over {self.shards} ranks and has no way to "
                f"join them; `ep.make_all_gather` is one"
            )
        self.shape = shard_shape(self.full_shape, self.shard, self.shards)
        self.device = torch.device(device)
        self.dtype = dtype
        self.block = int(block)
        #: Scores a single pass may materialise, and the block loop's tile, both **this share's**.
        #: The whole layer's counts are `shards` times these, so dividing them here is what makes a
        #: split call take the same steps as an unshapen one: the single-pass test is
        #: `heads * queries * keys` and the tile is `kv_heads * groups * rows * block`, and both
        #: are a quarter of the whole's at a share of four -- so a quarter of the budget is the
        #: same step, and it is what makes a four-rank prefill agree with a one-rank one.
        self.budget = int(budget) // self.shards
        #: Scores the block loop's tile may hold, or `None` for `DEFAULT_TILE_SCORES // shards`. It
        #: is a constant because a per-rank step makes the four ranks of a lockstep do four
        #: different amounts of work; the constant's comment has the measurement.
        self.tile_budget = (
            DEFAULT_TILE_SCORES // self.shards
            if tile_budget is None
            else int(tile_budget) // self.shards
        )
        if self.shape.projection_layout != "fused_qkv":
            raise NotImplementedError(
                f"layer {self.layer_idx}: the released checkpoint's fused qkv layout is the "
                f"only one implemented, got {self.shape.projection_layout!r}"
            )

        root = f"model.layers.{self.layer_idx}.self_attn"
        self.qkv_proj = _qkv_weight(checkpoint, self, f"{root}.qkv_proj.weight", dtype)
        #: The projection that mixes the heads back together, **whole on every rank**: the split
        #: is joined by concatenating the shares' `pre_o`, not by summing projected partials, and
        #: the join's own comment says why. So this tensor is the one weight a split attention
        #: does not divide and the one GEMM it does not divide either.
        self.o_proj = checkpoint.dense_tensor(f"{root}.o_proj.weight", dtype, self.device)
        sink_key = f"{root}.attention_sink_bias"
        self.sink = (
            checkpoint.dense_tensor(sink_key, torch.float32, self.device)
            if sink_key in checkpoint
            else None
        )
        if self.sink is not None and not self.shape.has_sink:
            raise ValueError(
                f"layer {self.layer_idx} ({self.shape.family}) holds a sink tensor but its "
                f"family does not carry one; the checkpoint and the config disagree"
            )
        if self.shape.has_sink and self.sink is None:
            raise ValueError(
                f"layer {self.layer_idx} ({self.shape.family}) is meant to carry a sink and the "
                f"checkpoint has no {sink_key}"
            )
        if self.sink is not None and self.shards > 1:
            self.sink = self.sink[
                self.shard * self.shape.num_q_heads : (self.shard + 1) * self.shape.num_q_heads
            ]

        self._inv_freq = build_rope_inv_freq(
            self.shape.rope_dim, self.shape.rope_theta, device=self.device
        )
        #: The permutation `split_fused_qkv` implies, as an index, or `None` for the layouts that
        #: need no permutation. One `index_select` builds the three tensors the split builds out
        #: of nineteen slices and six concatenations, and a decode step makes forty-eight of them
        #: a token, on a host that is the whole of the step's cost. A *share* of the checkpoint's
        #: partition is already in `[q | k | v]` order and takes the plain split.
        self._qkv_order = group_qkv_order(self.full_shape, self.shard, self.shards)
        if self._qkv_order is not None:
            self._qkv_order = self._qkv_order.to(self.device)
        #: A `[capacity, rope_dim]` cos/sin table this layer shares with the rest of its family,
        #: or `None` to build the table per call. See `share_rope_table`.
        self._rope_table: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def rope_key(self) -> tuple[int, float]:
        """What two layers have to agree on to share one RoPE table.

        The windowed layers use a different theta from the global ones, which is the whole of the
        difference between the two families' tables; the dimension is in the key because a config
        that changed `partial_rotary_factor` halfway down the stack would otherwise index a table
        built for another width.
        """
        return (int(self.shape.rope_dim), float(self.shape.rope_theta))

    def build_rope_table(self, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[capacity, rope_dim]` cos and sin for every absolute position below `capacity`.

        Float32, which is what `build_rope_cos_sin` returns anyway, so a lookup hands back the
        same bits the call would have computed.
        """
        positions = torch.arange(int(capacity), device=self.device)
        cos, sin = build_rope_cos_sin(self._inv_freq, positions.reshape(1, -1))
        return cos.reshape(-1, self.shape.rope_dim), sin.reshape(-1, self.shape.rope_dim)

    def share_rope_table(self, table: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Take a `[capacity, rope_dim]` table built by another layer of the same family.

        **Why a table at all.** `build_rope_cos_sin` is an outer product, two transcendental
        kernels and a concatenation, to produce one row of numbers -- and for a decode step the row
        it produces for position p is the row it produced last token. It is about half of what
        this and `_qkv_order` are worth together, which is 6.4% of a decode token on four ranks;
        the measurement is in `docs/models/mimo-v2.6-flash.md`, "The host is the step".
        """
        self._rope_table = table

    @property
    def memory_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qkv_proj, self.o_proj, self.sink)
            if tensor is not None
        )

    def rope(
        self,
        positions: torch.Tensor,
        *,
        within: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's family's cos/sin for absolute `positions`, `[1, n, rope_dim]`.

        `within` is one past the highest position the caller promises to ask for, and it is what
        lets a shared table be used without reading a position back to the host:
        `int(positions[-1])` would be a device sync, and a decode step would make forty-eight of
        them. Without a table, or with one that does not reach `within`, the table is built the
        long way -- so a caller that passes no `within` still gets the right answer, just not the
        cheap one.
        """
        if within is not None and self._rope_table is not None:
            cos, sin = self._rope_table
            if within <= cos.shape[0]:
                index = positions.reshape(-1).to(self.device)
                return cos[index].unsqueeze(0), sin[index].unsqueeze(0)
        return build_rope_cos_sin(self._inv_freq, positions.reshape(1, -1).to(self.device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: torch.Tensor | None = None,
        cache: MimoV2KVCache | None = None,
    ) -> dict[str, torch.Tensor]:
        """Attention for one chunk of one sequence, joined and projected.

        The split is a two-line thing around `attention_output`: a share produces its own heads'
        output and the four shares are concatenated, and the projection that mixes the heads runs
        on the whole of it. Nothing else about the layer is sharded and nothing else needs to be.
        """
        pre_o, qkv = self.attention_output(
            hidden_states, start_pos=start_pos, positions=positions, cache=cache
        )
        if self.gather is not None:
            pre_o = self.gather(pre_o)
            if pre_o.shape[-1] != self.full_shape.o_in:
                raise ValueError(
                    f"the gathered attention output is {pre_o.shape[-1]} wide and this layer's "
                    f"projection reads {self.full_shape.o_in}"
                )
        post_o = F.linear(pre_o.to(self.o_proj.dtype), self.o_proj)
        return {
            "qkv_raw": qkv,
            "attn_out_pre_o": pre_o,
            "attn_out_post_o": post_o,
        }

    def attention_output(
        self,
        hidden_states: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: torch.Tensor | None = None,
        cache: MimoV2KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """**This share's** attention output, `[sequence, o_in / shards]`, and the fused qkv.

        `attn_out_pre_o` before the join: the query heads this rank holds, attended, with the
        projection that mixes heads not yet applied. A caller with `shards == 1` gets the whole
        layer's `pre_o` and the dictionary `forward` returns is exactly what it always was.
        """
        if hidden_states.dim() == 2:
            flat = hidden_states
        elif hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
            flat = hidden_states[0]
        else:
            raise ValueError(
                f"expected [sequence, hidden] or [1, sequence, hidden], got "
                f"{tuple(hidden_states.shape)}"
            )
        sequence = flat.shape[0]
        if positions is None:
            positions = torch.arange(start_pos, start_pos + sequence, device=self.device)
        else:
            positions = positions.reshape(-1).to(self.device)
            if positions.numel() != sequence:
                raise ValueError(
                    f"{sequence} rows and {positions.numel()} positions; they are one to one"
                )
            if sequence and (
                int(positions[0]) != start_pos
                or not bool((positions.diff() == 1).all())
            ):
                raise ValueError(
                    f"positions must run contiguously from start_pos={start_pos}, got "
                    f"{positions[:4].tolist()}.."
                )
        shape = self.shape

        cos, sin = self.rope(positions, within=start_pos + sequence)
        qkv = F.linear(flat, self.qkv_proj)
        if self._qkv_order is None:
            query, key, value = split_fused_qkv(qkv, shape, shape.qkv_row_layout)
        else:
            query, key, value = qkv.index_select(-1, self._qkv_order).split(
                [shape.q_size, shape.k_size, shape.v_size], dim=-1
            )

        heads, kv_heads = shape.num_q_heads, shape.num_kv_heads
        query = query.view(sequence, heads, shape.head_dim).transpose(0, 1)
        key = key.view(sequence, kv_heads, shape.head_dim).transpose(0, 1)
        value = value.view(sequence, kv_heads, shape.v_head_dim).transpose(0, 1)

        if shape.value_scale is not None:
            value = value * shape.value_scale
        query = apply_partial_rope(query.unsqueeze(0), cos, sin, shape.rope_dim)[0]
        key = apply_partial_rope(key.unsqueeze(0), cos, sin, shape.rope_dim)[0]

        # The cache holds keys and values at the compute dtype, and the chunk's own
        # keys and values are cast to it before they are used. That is what makes a
        # token's key the same bits whether it was read back from the cache or
        # computed in the call that consumed it -- so a chunked prefill agrees with a
        # one-shot prefill to the rounding of its own summations and not to the
        # cache's. The query is not cached and stays at the width the rope left it
        # at.
        if cache is not None and cache.dtype != key.dtype:
            key, value = key.to(cache.dtype), value.to(cache.dtype)
        else:
            key, value = key.to(self.dtype), value.to(self.dtype)

        # The attention reads one span, `[0, prefix_len + sequence)`: the prefix below this chunk
        # with the chunk's own keys on the end of it. Where the cache's buffer *is* that span in
        # order -- nothing has wrapped -- the chunk is appended first and the span read back as a
        # view of the buffer, which is the same bytes the concatenation below would build and is
        # not a copy of the whole prefix: at 262144 a global layer's prefix is 160 MiB of keys and
        # values with the attention split over four ranks, read and written every token. A ring
        # that has wrapped is read before its append as it always was; `append_and_span` says
        # which of the two this call is and has the reason.
        span = (
            None
            if cache is None
            else cache.append_and_span(self.layer_idx, key, value, start_pos=start_pos)
        )
        if span is not None:
            all_key, all_value = span
            prefix_len = start_pos
        else:
            prefix_key = prefix_value = None
            prefix_len = 0
            if cache is not None:
                prefix_key, prefix_value, prefix_len = cache.prefix(self.layer_idx, start_pos)
            if prefix_len:
                all_key = torch.cat([prefix_key, key], dim=1)
                all_value = torch.cat([prefix_value, value], dim=1)
            else:
                all_key, all_value = key, value

        rows = torch.arange(sequence, device=self.device)
        upper = prefix_len + rows
        if shape.sliding_window is None:
            lower = torch.zeros_like(upper)
        else:
            lower = (upper - int(shape.sliding_window) + 1).clamp_min(0)

        # The two bounds a low-level caller has to be checked against are known here, in Python,
        # and not on the card: `upper` runs from `prefix_len` to `prefix_len + sequence - 1` and
        # the key tensor is exactly `prefix_len + sequence` long, and `lower` is either zero or
        # `upper` clamped down from below. So the value checks have nothing to find, and asking
        # them anyway is two device-to-host round trips a layer -- 96 of a decode token's 145
        # `cudaStreamSynchronize` calls, each one stalling the host on the kernel it queued last.
        attn_output, stats = attention(
            query.to(torch.float32),
            all_key,
            all_value,
            lower,
            upper,
            scaling=shape.scaling,
            sink=self.sink,
            block=self.block,
            budget=self.budget,
            tile_budget=self.tile_budget,
            validate=False,
        )
        self.last_stats = stats
        if cache is not None and span is None:
            cache.append(self.layer_idx, key, value)

        pre_o = attn_output.transpose(0, 1).reshape(sequence, shape.o_in).contiguous()
        return pre_o, qkv
