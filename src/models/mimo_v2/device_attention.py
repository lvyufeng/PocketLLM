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

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.mimo_v2.config import MimoV2AttentionShape, MimoV2TextConfig
from src.models.mimo_v2.layers import (
    apply_partial_rope,
    build_rope_cos_sin,
    build_rope_inv_freq,
    split_fused_qkv,
)

__all__ = [
    "AttentionStats",
    "MimoV2DeviceAttention",
    "MimoV2KVCache",
    "attention",
    "blocked_attention",
    "fused_qkv_row_order",
    "single_pass_attention",
]


def fused_qkv_row_order(shape: MimoV2AttentionShape, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """The permutation that turns the stored `qkv_proj` rows into `[q | k | v]`.

    Exists for callers that want the weight itself reordered -- a kernel that wants
    three contiguous matrices, or a single-rank path that wants to skip the
    four-way split per call. The attention below does not need it: it runs the
    projection in the stored row order and cuts the output, which is the same
    arithmetic with the permutation applied to 1 row of activations instead of
    `qkv_out` rows of weights.

    The permutation is the one `split_fused_qkv` implies, read off the same
    description rather than derived a second time: shard `i` holds
    `[q_i | k_i | v_i]` at `stride * i`, and the canonical order gathers the four
    query sections, then the four (or one) key sections, then the value ones.
    """
    groups = 4
    if shape.qkv_row_layout == "contiguous":
        return torch.arange(shape.qkv_out, dtype=dtype)
    if shape.qkv_row_layout != "tp4_interleaved":
        raise ValueError(
            f"no row order is known for {shape.qkv_row_layout!r}; the released checkpoint's "
            f"fused projection is 'tp4_interleaved'"
        )
    q_head = shape.q_size // groups
    k_head = shape.k_size // groups
    v_head = shape.v_size // groups
    stride = q_head + k_head + v_head
    rows = []
    for lo, width in ((0, q_head), (q_head, k_head), (q_head + k_head, v_head)):
        for group in range(groups):
            base = group * stride + lo
            rows.append(torch.arange(base, base + width, dtype=dtype))
    return torch.cat(rows)


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


def _check_bounds(query, key, value, lower, upper, sink):
    """The checks both attention paths make, and the geometry they both fold with."""
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
    """
    heads, queries, head_dim, kv_heads, keys, groups = _check_bounds(
        query, key, value, lower, upper, sink
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
    """
    heads, queries, head_dim, kv_heads, keys, groups = _check_bounds(
        query, key, value, lower, upper, sink
    )
    width = value.shape[-1]

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
    for start in range(0, keys, block):
        stop = min(start + block, keys)
        # The query rows that can see any key in [start, stop): `upper >= start` and
        # `lower <= stop - 1`. Both bounds are sorted, so these are two searches.
        first = int(torch.searchsorted(upper, torch.tensor(start, device=query.device)))
        last = int(
            torch.searchsorted(lower, torch.tensor(stop - 1, device=query.device), right=True)
        )
        if last <= first:
            continue
        seen_blocks += 1
        pairs += heads * (last - first) * (stop - start)

        rows_q = folded_query[:, :, first:last]
        block_k = key[:, start:stop].to(torch.float32)
        # [kv_heads, groups, rows, head_dim] x [kv_heads, 1, head_dim, block]
        scores = torch.matmul(rows_q, block_k.transpose(1, 2).unsqueeze(1)) * scaling

        positions = torch.arange(start, stop, device=query.device)
        visible = (positions.unsqueeze(0) >= lower[first:last].unsqueeze(1)) & (
            positions.unsqueeze(0) <= upper[first:last].unsqueeze(1)
        )
        scores = scores.masked_fill(~visible.unsqueeze(0).unsqueeze(0), float("-inf"))

        block_max = scores.amax(dim=-1)
        previous = running_max[:, :, first:last]
        merged = torch.maximum(previous, block_max)
        # A row whose every key in this block is masked has a `-inf` maximum, and
        # `exp(-inf - -inf)` is a nan rather than a zero. Such a row keeps its
        # running state instead: the substitution only ever feeds the exponentials.
        infinite = torch.isinf(merged)
        finite = torch.where(infinite, torch.zeros_like(merged), merged)
        alpha = torch.exp(previous - finite).masked_fill(infinite, 1.0)
        probs = torch.exp(scores - finite.unsqueeze(-1))
        probs = torch.where(infinite.unsqueeze(-1), torch.zeros_like(probs), probs)

        running_sum[:, :, first:last] = running_sum[:, :, first:last] * alpha + probs.sum(dim=-1)
        out[:, :, first:last] = out[:, :, first:last] * alpha.unsqueeze(-1) + torch.matmul(
            probs, value[:, start:stop].to(torch.float32).unsqueeze(1)
        )
        running_max[:, :, first:last] = merged

    flat_out = flat_out / flat_sum.unsqueeze(-1)
    return flat_out.to(query.dtype), AttentionStats("blocked", seen_blocks, pairs, heads * queries * keys)


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
) -> tuple[torch.Tensor, AttentionStats]:
    """One piece, or a block loop, whichever the scores can afford.

    `budget` is a count of scores and not of bytes, so one number covers both dtypes:
    at the default a call may materialise 33.5M of them, which is 134 MiB of float32 and
    the whole reason the block loop exists above it.

    Both paths are the same softmax and the tests hold them to the same answer; what
    decides is that a decode step is one query against up to 256k keys, which is a
    single gemm and no loop worth writing, and a prefill chunk is thousands of queries
    against the same keys, which is a grid nothing should materialise at once.
    """
    if query.shape[0] * query.shape[1] * key.shape[1] <= budget:
        return single_pass_attention(query, key, value, lower, upper, scaling=scaling, sink=sink)
    return blocked_attention(
        query, key, value, lower, upper, scaling=scaling, sink=sink, block=block
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
    """

    def __init__(
        self,
        config: MimoV2TextConfig,
        capacity: int,
        layers: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        self.layers = tuple(range(config.num_hidden_layers)) if layers is None else tuple(layers)
        if not self.layers:
            raise ValueError("a cache with no layers holds nothing")

        self._shapes: dict[int, MimoV2AttentionShape] = {}
        self._slots: dict[int, int] = {}
        self._key: dict[int, torch.Tensor] = {}
        self._value: dict[int, torch.Tensor] = {}
        self._written: dict[int, int] = {}
        for layer in self.layers:
            shape = config.attention(layer)
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
        """
        if layer not in self._key:
            raise KeyError(f"layer {layer} is not in this cache; it holds {sorted(self._key)}")
        if upto > self._written[layer]:
            raise ValueError(
                f"layer {layer} has {self._written[layer]} positions cached and {upto} were asked "
                f"for; a chunk cannot read positions it has not appended"
            )
        slots = self._slots[layer]
        start = max(0, upto - slots)
        wanted = torch.arange(start, upto, device=self.device) % slots
        return (
            self._key[layer].index_select(1, wanted),
            self._value[layer].index_select(1, wanted),
            upto - start,
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
    ) -> None:
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        self.config: MimoV2TextConfig = checkpoint.layer
        self.shape: MimoV2AttentionShape = self.config.attention(self.layer_idx)
        self.device = torch.device(device)
        self.dtype = dtype
        self.block = int(block)
        self.budget = int(budget)
        if self.shape.projection_layout != "fused_qkv":
            raise NotImplementedError(
                f"layer {self.layer_idx}: the released checkpoint's fused qkv layout is the "
                f"only one implemented, got {self.shape.projection_layout!r}"
            )

        root = f"model.layers.{self.layer_idx}.self_attn"
        self.qkv_proj = checkpoint.dense_tensor(f"{root}.qkv_proj.weight", dtype, self.device)
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

        self._inv_freq = build_rope_inv_freq(
            self.shape.rope_dim, self.shape.rope_theta, device=self.device
        )

    @property
    def memory_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qkv_proj, self.o_proj, self.sink)
            if tensor is not None
        )

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's family's cos/sin for absolute `positions`, `[1, n, rope_dim]`."""
        return build_rope_cos_sin(self._inv_freq, positions.reshape(1, -1).to(self.device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        start_pos: int = 0,
        positions: torch.Tensor | None = None,
        cache: MimoV2KVCache | None = None,
    ) -> dict[str, torch.Tensor]:
        """Attention for one chunk of one sequence.

        `start_pos` is the absolute position of the chunk's first token and is what
        the cache reads and the RoPE table is built from. `positions`, when given, is
        the chunk's absolute positions and must be contiguous from `start_pos` -- a
        decode step passes one position and a prefill passes a run, and anything else
        is a different feature rather than a different call.
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

        cos, sin = self.rope(positions)
        qkv = F.linear(flat, self.qkv_proj)
        query, key, value = split_fused_qkv(qkv, shape, shape.qkv_row_layout)

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
        )
        self.last_stats = stats
        if cache is not None:
            cache.append(self.layer_idx, key, value)

        pre_o = attn_output.transpose(0, 1).reshape(sequence, shape.o_in).contiguous()
        post_o = F.linear(pre_o.to(self.o_proj.dtype), self.o_proj)
        return {
            "qkv_raw": qkv,
            "attn_out_pre_o": pre_o,
            "attn_out_post_o": post_o,
        }
