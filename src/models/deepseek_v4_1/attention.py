"""The CSA2 attention half of the DeepSeek-V4.1 backbone, in pure PyTorch.

CSA2 assigns every attention layer one of three static modes and makes the layers share work
instead of each deriving it. Four layers — `kv_source_layers` — pool their own KV with a
`Compressor`; eight — `index_source_layers` — run an `Indexer` that picks the `index_topk`
compressed positions each query attends to; the rest read what those layers published. The eighth
index source, `candidate_source_layer`, additionally narrows the field to `candidate_topk_blocks`
blocks first, and the index sources after it score inside that narrowed field rather than over the
whole context — the two-level hierarchy the model card calls a Hierarchical Sparse Indexer.

`Attention` then attends over two KV sources at once, concatenated into a single `sparse_attn`
call: a sliding window of raw KV, and the compressed positions reaching further back.

This module follows the released `inference/model.py` rather than the V4-Flash runtime in
`src/models/deepseek_v4/`, because the two disagree in ways the layer sees: V4.1 gives
`compress_ratios` a different meaning (a layer with a non-zero ratio *reads* compressed positions;
only four layers write them), it adds the candidate-block level, and its decoder projects the global
KV cache from the encoder. The four helpers at the top of this file — `RMSNorm`,
`precompute_freqs_cis`, `apply_rotary_emb`, `get_window_topk_idxs` — are the reference's, written
here rather than imported from the V4-Flash runtime for the same reason: the V4-Flash copy has
since grown a speculative-verify branch and no longer states the same function.

No weights are loaded here and none of this claims to reproduce the released model's tokens: it
defines the arithmetic, and `tests/test_models_deepseek_v4_1_attention.py` holds it to the
properties that are checkable without a checkpoint. Tensor parallelism is not implemented — the
reference splits heads and groups across ranks and all_reduces the indexer scores; at one rank
those paths are the identity, which is what is written here.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache, partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.ops import act_quant, fp4_act_quant, sparse_attn
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.decode_pos import Pos, publish, write_row
from src.models.deepseek_v4_1.kernels import fp4_act_quant_e4m3
from src.models.deepseek_v4_1.tp import indexer_row_split

__all__ = [
    "Attention",
    "AttentionStack",
    "Compressor",
    "Indexer",
    "SharedAttentionRuntime",
    "canonical_device",
    "get_window_topk_idxs",
    "precompute_freqs_cis",
    "select_candidate_blocks",
]

# The reference's module globals, from its `kernel.py`. The window KV is quantized to FP8 over 32
# elements; the indexer's queries and keys to FP4 with E8M0 scales over 32; the compressed KV to
# FP4 with E4M3 scales over 16 -- that last one is the branch `fp4_act_quant_e4m3` exists for.
FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
COMPRESS_KV_BLOCK_SIZE = 16
SCALE_FMT = "ue8m0"
SCALE_DTYPE = torch.float8_e8m0fnu

# What this module holds its dense weights and caches in, where the reference's choice is not
# structural. The reference's `Linear` defaults to `torch.float8_e4m3fn`, so most of these layers
# store fp8 weights and quantize their *activations* through `act_quant` before an fp8 GEMM; only the
# few it names explicitly (`wo_a` bf16, the compressor's fp32 pooling above ratio 1) are otherwise.
# None of that is reproducible without the checkpoint's block scales, and none of it changes a shape
# or a sharing decision -- so this module keeps one dense-precision dtype and says so, rather than
# inventing a weight-quantization pass over random parameters. Where the reference's dtype choice
# *is* structural -- the compressor's fp32 softmax pooling -- the code below keeps it.
#
# fp16 rather than bf16, because sm_75 has no bf16 tensor core: cuBLAS answers a bf16 GEMM with the
# fp32 SIMT kernel `magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16>` at 6.4-7.5 TFLOP/s,
# where the identical shapes in fp16 take `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f` at 47-58.
# The projections this applies to are stored fp8 with E8M0 block scales, so their weight is formed in
# fp32 and narrowed to whichever carrier either way -- and fp16 is the wider container of the two,
# carrying 11 significand bits to bf16's 8. Everything else in the stack (the compressor's `wkv`,
# the indexer's `wk`, the router, the embedding) is bf16 in the checkpoint, and bf16-to-fp16 is
# lossless for values in fp16's exponent range, which measured inputs are: the largest input arriving
# at any site the change carries is 100, 0.153 % of fp16's largest finite. Decode is not traded away
# for this -- at one row there is no GEMM left, only a matvec whose cost is the weight bytes it
# reads, and both widths are two bytes -- so it is a prefill lever. The per-site lever, the parity
# diff and the decode arm are recorded in `docs/performance/v41_dense_gemm_dtype.md`.
LINEAR_DTYPE = torch.float16
# The KV-side caches. Not an independent knob: every sparse-attention entry point in
# `src/csrc/cuda_kernel_impl.cu` dispatches on `q.scalar_type()` and reads its `kv` at that same
# dtype, so a cache at a different width from the projections filling it is silent corruption rather
# than an error. It is a separate name only because it also sizes three persistent buffers.
CACHE_DTYPE = torch.float16


class RMSNorm(nn.Module):
    """The checkpoint stores these in bf16; the parameter is kept fp32 and the input upcast, which
    is what the reference does unconditionally rather than only when the dtype is low."""

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | str | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def canonical_device(device: torch.device | str | None) -> torch.device | None:
    """`device` with an unindexed `cuda` resolved to the card we are actually on.

    Every node in the tree builds its own copy of the rope table, and they share one by asking for
    the same key. `torch.device("cuda")` is not the same key as `torch.device("cuda", 0)` and lands
    wherever the current device happens to point, so it is resolved here rather than at each cache
    lookup -- otherwise the first node to be built would decide, silently, which card the other
    thirty-nine read their roperies from.
    """
    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


@lru_cache(16)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With `original_seq_len > 0` this is YaRN: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`, and the
    band between `beta_fast` and `beta_slow` is faded across with a linear ramp. The V4.1 config
    turns YaRN on for the layers that read compressed positions (`compress_rope_theta`, 160,000)
    and off for the pure sliding-window layers (`rope_theta`, 10,000).

    `device` is part of the cache key and not just of the result, and that is the whole point of it.
    The table is large -- `max_position_embeddings` is 1,048,576 and `rope_head_dim` 64, so one
    variant is 268 MB of complex64 -- and all forty layers of a backbone want the same one or the
    other of exactly two. `nn.Module.to` does not dedupe a buffer two modules share (measured, torch
    2.9.1: `_apply` rebuilds each module's own copy), so a `.to(cuda)` on a host-built tree would
    turn those 536 MB into 10.7 GiB per card. Keyed on the device, the tree gets one object per
    variant and the modules alias it.

    The size is not two. Two variants times one card per process is four entries, but a process that
    drives four cards itself asks for all eight, and an eviction is worse than a miss: the modules
    still hold the object they were handed, so the next call for the same key builds a *second* 268 MB
    table beside it rather than reusing the live one. Sixteen leaves room for that without ever
    reaching for it.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # the dimension whose wavelength completes `rotations` across the training context
        def corrected_dim(rotations: int) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    table = torch.polar(
        torch.ones_like(torch.outer(torch.arange(seqlen), freqs)), torch.outer(torch.arange(seqlen), freqs)
    )
    return table if device is None else table.to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate `x` in place, taking adjacent element pairs as complex numbers.

    Accepts `[b, s, d]` and `[b, s, h, d]`; `inverse` conjugates the rotation, which is how the
    attention output has the query's rotation removed again so the KV cache can stay in one shared
    rotated form rather than two.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y.copy_(torch.view_as_real(x * freqs_cis).flatten(-2))
    return y


def _is_continuation(pos: Pos, seqlen: int) -> bool:
    """Whether this forward is a later chunk of a prompt whose earlier chunks have already run.

    `start_pos > 0` alone does not say which of two bodies this is and they are different bodies: a
    decode step has one query and reads the ring buffer as its whole window, while a continuation
    chunk has `seqlen` queries whose windows reach *back past the chunk*, into rows this forward has
    to keep in front of its own. The count of queries is the whole of the difference -- a prompt is
    never one token and a decode step is never more than one -- and it is the caller's `seqlen`
    rather than anything about the position, which is why it is passed in.
    """
    return not pos.first() and seqlen > 1


def _group_freqs(freqs_cis: torch.Tensor, pos: Pos, n_groups: int, ratio: int, seqlen: int) -> torch.Tensor:
    """The RoPE row for the first token of each of the `n_groups` groups a forward closes.

    A latent stands for the first token of its group, so group `g` takes position `g * ratio`. A
    chunk starting at 0 indexes the table from 0 in steps of `ratio`; a later chunk starts at the
    first multiple of `ratio` at or below its own start -- the position the group it completes began
    on -- and takes `n_groups` steps from there; a decode step has one group and *picks* its row,
    because it is on the device path and a Python `slice` is a value a capture would freeze.
    """
    if pos.first():
        return freqs_cis[: n_groups * ratio : ratio]
    if _is_continuation(pos, seqlen):
        base = pos.host - pos.host % ratio
        return freqs_cis[base : base + n_groups * ratio : ratio]
    return pos.pick(freqs_cis, 1 - ratio)


def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    pos: int | Pos,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Which sliding-window cache slots each query attends to; `-1` marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.

    `device` is where the table has to end up, because `Attention.forward` concatenates it against
    the card's own tensors -- a CPU table fails there the moment the tree runs anywhere but the host,
    and the shapes are equal enough that only the `cat` catches it. The int path *builds* the table
    on the host and moves it, not in place: at the config's window of 128 a decode step's table is
    512 bytes, while the eight `arange`/`clamp`/`where` kernels it takes to build one are eight
    launches on a stream whose busy fraction a decode round exists to raise.

    A *later chunk* of a prompt is the third case and its own expression, because the cache is no
    longer the chunk. A chunk of more than one query starting past zero reads its window out of two
    tensors: the ring holds the `window_size` positions that precede the chunk, and the chunk's own K
    is read beside it, so `Attention._window_kv` hands `sparse_attn` the two concatenated and a
    position is named by which half it fell in -- a ring slot below `window_size`, or an offset into
    the chunk above it. Emitted oldest position first, which is the order the prefill branch emits and
    the order the sparse attention's denominator sums in; a chunk and the same tokens prefilled in one
    call therefore select the same vectors in the same order.

    The device path is in the `pos.on_device` branch below the prefill one, and it is not that build
    moved to the card, it is a different expression with the same value: the ring's listing
    `cat([arange(oldest, win), arange(oldest)])` is the rotation `(j + oldest) % window_size`, so it
    is one `arange` and one modulo on shapes the position cannot change -- which is what a capture
    needs, and what removes the pageable H2D along with the host build. The mask that follows is
    the *same* comparison on both paths, because it tests a slot index against the position and not
    the other way round. The prefill branch and the host build stay exactly as they were for the int
    path, and the prefill branch is checked first: it is the one case the device path does not cover
    (its table has one row per query, and prefill stays eager), so a device `Pos` at 0 -- which no
    decode step builds -- reaches the prefill build rather than a rotation that would be wrong for
    it.
    """
    pos = Pos.of(pos)
    if pos.first():
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    elif _is_continuation(pos, seqlen):
        # Positions, not slots: the row is one query's whole window by absolute position, and which
        # half of the concatenated K a position landed in is decided after. `k0` is the first
        # position the query can still see, which is 0 for the first rows of a prompt shorter than
        # the window and `end - window_size + 1` once the window is full.
        assert not pos.on_device, "a chunk of more than one query is the eager path"
        end = pos.host + torch.arange(seqlen).unsqueeze(1)
        start = (end - window_size + 1).clamp(0)
        at = start + torch.arange(window_size)
        # a position before the chunk is a ring slot, one inside the chunk is an offset past the
        # ring's `window_size` columns; a position the query has not reached is not attended to
        idxs = torch.where(at < pos.host, at % window_size, window_size + at - pos.host)
        idxs = torch.where(at > end, -1, idxs)
    elif pos.on_device:
        oldest = pos.slot(window_size) + 1
        idxs = (torch.arange(window_size, device=pos.torch_device) + oldest) % window_size
        idxs = torch.where(idxs > pos.row(), -1, idxs)  # ring still filling
    else:
        oldest = pos.slot(window_size) + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > pos.host, -1, idxs)  # ring still filling
    # `sparse_attn` needs real [b, m, topk] int32 memory, hence the materializing expand
    idxs = idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
    return idxs if device is None else idxs.to(device)


# There is deliberately no cache here. The one this replaced was `@lru_cache(1)` keyed on the
# arguments and not on the device, so a second device silently received the first device's tensor --
# and it could not have helped anyway: `start_pos` is in the key and advances every decode step, so
# the steady state was a miss per layer per step. What is worth caching is the ring listing, which
# depends on `start_pos % window_size` alone, and that belongs with the module that owns the
# per-step launch budget rather than with a free function two callers share.


class SharedAttentionRuntime:
    """What attention layers hand down the stack instead of recomputing.

    Layers run in order and every source writes before its consumers read, so one slot each is
    enough and nothing needs resetting between forwards. Sources: `compress_kv` and `index_k` from
    `kv_source_layers`, `topk_idxs` from `index_source_layers`, `candidates` from
    `candidate_source_layer`.

    `candidates` names *blocks*, and a block is `candidate_block_size` compressed positions -- so the
    ids only mean anything in the ratio they were built in. `candidates_ratio` carries that ratio
    beside them, and a consumer asserts it: the dense mask this replaced was shaped like the
    consumer's own score and so could only ever have been consumed by a layer of the same ratio, and
    the ids would instead be silently wrong.

    The reference holds this in a module-level singleton because there is one model per process.
    It is passed explicitly here so that two stacks in one process cannot silently share a cache.
    """

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.candidates_ratio: int | None = None


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up — so during decode it
    only yields every `compress_ratio` steps, holding the partial group in `kv_state`/
    `score_state`. Pre-RoPE is deliberate: the indexer needs the unrotated form, so `Attention`
    rotates afterwards.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        layer_id: int,
        max_batch_size: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        ratio = _compress_ratio_at(cfg, layer_id)
        head_dim = _required_int(cfg, "head_dim")
        self.compress_ratio = ratio
        self.head_dim = head_dim
        # Which decode body this forward is, when it is being *recorded* rather than run: `None`
        # derives it from the position, `True` records the emitting body and `False` the filling
        # one. What the body contains -- the quantizer and the cache write, or neither -- is
        # otherwise `(start_pos + 1) % ratio == 0`, a host answer a capture freezes; `DecodeGraphs`
        # therefore captures the two bodies as two graphs, and this is what lets it record either
        # of them at a position that would have chosen the other. `None` everywhere else, so the
        # eager path takes exactly the branch it always took.
        self.compress_variant: bool | None = None
        self.norm = RMSNorm(head_dim, _required_float(cfg, "norm_eps"), device=device)
        # ratio 1 is a plain projection, so it is the checkpoint's own bf16 -- carried at
        # `LINEAR_DTYPE`, which holds that value exactly; the softmax pooling above ratio 1 runs in
        # fp32, so those weights are promoted to fp32 to match
        # `bias=False` throughout this module, as the reference's `Linear` defaults to and as the
        # checkpoint is: no V4.1 projection carries one.
        self.wkv = nn.Linear(
            _required_int(cfg, "dim"),
            head_dim,
            bias=False,
            dtype=torch.float32 if ratio > 1 else LINEAR_DTYPE,
            device=device,
        )
        if ratio <= 1:
            return
        self.wgate = nn.Linear(_required_int(cfg, "dim"), head_dim, bias=False, dtype=torch.float32, device=device)

        state_shape = (max_batch_size, ratio, head_dim)
        self.register_buffer(
            "kv_state", torch.zeros(state_shape, dtype=torch.float32, device=device), persistent=False
        )
        self.register_buffer(
            "score_state",
            torch.full(state_shape, -torch.inf, dtype=torch.float32, device=device),
            persistent=False,
        )

    def reset_state(self, batch_size: int) -> None:
        """Forget the partial group on `batch_size` rows. The reference never does this -- it is one
        conversation per process -- but two forwards over the same module are otherwise not
        independent, which is a trap for a test rather than a property of the model."""
        if self.compress_ratio <= 1:
            return
        self.kv_state[:batch_size].zero_()
        self.score_state[:batch_size].fill_(-torch.inf)

    def forward(self, x: torch.Tensor, pos: int | Pos) -> torch.Tensor | None:
        pos = Pos.of(pos)
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if pos.first():
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        elif _is_continuation(pos, seqlen):  # a later chunk of a prompt: pool in groups, not per token
            # The chunk closes every group that ends inside it. A group is `ratio` consecutive
            # positions, so the group the previous chunk left open -- if this chunk does not start on
            # a boundary -- is closed by this chunk's first tokens, and those first tokens complete it
            # *in the state*, which is where its earlier tokens are; the groups that begin and end
            # inside the chunk pool directly. Both are the same softmax over the same `ratio` rows,
            # which is why a chunk and the same tokens in one call agree bit for bit.
            assert not pos.on_device, "a chunk of more than one query is the eager path"
            carry = pos.host % ratio
            take = min(ratio - carry, seqlen) if carry else 0
            if take:
                self.kv_state[:bsz, carry : carry + take] = kv[:, :take]
                self.score_state[:bsz, carry : carry + take] = score[:, :take]
                kv, score = kv[:, take:], score[:, take:]
            if carry and take < ratio - carry:  # the chunk ends inside the group it did not close
                return None
            rest = seqlen - take
            whole = rest - rest % ratio
            pooled = []
            if carry:
                pooled.append((self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True))
            if whole:
                grouped_kv = kv[:, :whole].unflatten(1, (-1, ratio))
                grouped_score = score[:, :whole].unflatten(1, (-1, ratio))
                pooled.append((grouped_kv * grouped_score.softmax(dim=2)).sum(dim=2))
            if rest % ratio:  # trailing partial group waits in the state for the next chunk
                self.kv_state[:bsz, : rest % ratio] = kv[:, whole:]
                self.score_state[:bsz, : rest % ratio] = score[:, whole:]
            if not pooled:  # nothing closed: the whole chunk is still one open group
                return None
            should_compress = True
            kv = torch.cat(pooled, dim=1)
        else:  # one token per step: fill a slot, and pool only when the group just completed
            variant = self.compress_variant
            should_compress = pos.emits(ratio) if variant is None else variant
            slot = pos.slot(ratio)
            write_row(self.kv_state[:bsz], slot, kv.squeeze(1))
            write_row(self.score_state[:bsz], slot, score.squeeze(1))
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))


# The indexer is the one place in V4.1 whose cost is quadratic in the sequence length, and it used to
# build the whole `[bsz, seqlen, index_heads, width]` score in a single `einsum`. That matrix is what
# puts a long context out of reach on a 22 GiB card: at 256K positions one ratio-1 index source's is
# 4.3 TB in bf16, and layers 2/8/14/20 cannot narrow `width` at all -- the candidate level applies
# only strictly after layer 20. Nothing downstream reads the matrix. A layer wants `index_topk` key
# ids per query and, at the candidate source, `candidate_topk_blocks` block ids, so the score is built
# one tile at a time and merged into a running top-k of fixed width.
#
# The tiling is exact rather than a threshold approximation: `_TopKStream.push` keeps the top-k of the
# union of what it holds with the tile, and the top-k of a union is the top-k of the merged top-k
# lists, so a value that survives is the true k-th best of everything pushed so far. The head
# reduction runs inside a tile over the head axis it always ran over, and no tile straddles the
# accumulation axis. What is *not* pinned is the order of equal scores -- `torch.topk(sorted=False)`
# picks among ties by an order this merge need not share with a one-shot call -- which is the same tie
# artifact `tests/test_models_deepseek_v4_1_attention.py` already records for `index_topk` truncation.
# The picks are re-sorted by position before they leave, which *is* pinned: `sparse_attn` sums over
# that axis, so a different order there is different arithmetic rather than a different tie.
#
# The tile is sized from a budget on the *score* tensor rather than fixed, because the two shapes the
# indexer is asked for are three orders of magnitude apart. A decode step has one query, so the budget
# buys a single tile spanning the whole width and the loop runs once: the decode graph records the
# same one pass it used to and the launch count does not change. A 256K prefill has 256K queries
# against a 256K-wide cache, and there the tiles are what keep the peak finite. Measured peak per
# index layer at 4096/16384/65536/262144 before this: 0.63/9.87/... GiB, OOM at 24576.
INDEXER_SCORE_BUDGET = int(os.getenv("DEEPSEEK_V41_INDEXER_SCORE_BUDGET", str(1 << 26)))
INDEXER_QUERY_TILE = int(os.getenv("DEEPSEEK_V41_INDEXER_QUERY_TILE", "2048"))
INDEXER_MIN_KEY_TILE = int(os.getenv("DEEPSEEK_V41_INDEXER_MIN_KEY_TILE", "1024"))
# The candidate path gathers its keys per query, so its tiles are bounded by the gathered copy
# (`q_tile * blocks * block_size * index_head_dim`) rather than by the score alone.
INDEXER_CAND_QUERY_TILE = int(os.getenv("DEEPSEEK_V41_INDEXER_CAND_QUERY_TILE", "512"))
INDEXER_CAND_TILE = int(os.getenv("DEEPSEEK_V41_INDEXER_CAND_TILE", "64"))
# Whether the candidate level's top-k stream keeps `_TopKStream.push`'s early-out. It is **off**,
# which is the answer a capture already gets -- `_TopKStream` refuses that read inside one, so this
# makes the eager path agree with the captured one rather than differ from it. The reason is the
# geometry: the stream is built with `k = min(index_topk, width)`, which is exactly this tiling's
# `span`, so the field being narrowed is already the top-k's own width and the running k-th value sits
# inside every incoming tile rather than above it. The guard is still *live*, but a hit is worth one
# merge and the test costs a device->host synchronization on every push: 0 hits in 255 synthetic
# pushes at 174 us of a 1524 us c-iteration on the fabric
# (`/tmp/bench_indexer_cand_overlap.py`, four ranks, both cache widths), and 1 hit in 896 pushes
# against 0.14 s of a 1.12 s `stream_candidates` row in situ
# (`/tmp/probe_cand_skip_inproc.py --at 8192 --chunk 4096 --arms 0 1 1 0`, both brackets agreeing in
# sign). The read also drains the compute stream every c-iteration, which is what hides the
# collective `_ReducePipeline` would defer on this path. Setting this to 1 restores the tested
# branch, which is what the guard is for.
INDEXER_CAND_SKIP_TEST = int(os.getenv("DEEPSEEK_V41_INDEXER_CAND_SKIP_TEST", "0"))
# A key tile's collective is a blocking `all_reduce` on the compute stream, and a `[2048, 4096]` tile
# is 5.4 ms of wire against 4.4 ms of arithmetic on this fabric -- so the loop spends most of its time
# waiting for the wire with the SMs idle. Deferring the join by `depth` tiles puts that wire under the
# following tiles' arithmetic; 0 is the shipped order and 2 is where the lookahead saturates. See
# `_ReducePipeline`.
INDEXER_REDUCE_DEPTH = int(os.getenv("DEEPSEEK_V41_INDEXER_REDUCE_DEPTH", "0"))


def _as_column(lens: torch.Tensor | int, bsz: int, seqlen: int, device, dtype) -> torch.Tensor:
    """`compress_lens` widened to one column per query: `[bsz, seqlen, 1]`.

    The indexer carries it as a per-query column during prefill -- a count per query, counted from the
    position the forward starts at -- and as a 0-dim tensor during a decode step, where the single
    number is a read of the recorded position. Every use below wants it aligned with the query axis of
    a score matrix, and one caller needs it in a *shape*, which is why this exists rather than the
    `masked_fill` broadcast the other call sites get away with. Widening by `expand` rather than
    indexing is also what keeps a 0-dim tensor out of `slice`, which refuses it.
    """
    if torch.is_tensor(lens):
        return lens.to(dtype).expand(bsz, seqlen, 1)
    return torch.full((bsz, seqlen, 1), int(lens), dtype=dtype, device=device)


def _recording() -> bool:
    """Whether a CUDA graph is being recorded on the current stream.

    Asked by the site that would otherwise read a tensor back to the host: a capture forbids that
    read (`cudaErrorStreamCaptureUnsupported`), and the answer cannot be hoisted out of the capture,
    because what the site branches on is a value the body under capture is still computing. So the
    host is asked what *it* is doing instead of what the card holds -- the stream query enqueues
    nothing -- and the recorded body takes the branch that needs no answer. Eager, the query is False
    and the branch is the one the same code has always taken, so nothing about the prefill changes.

    Not `torch.cuda.is_current_stream_capturing()` alone: that raises on a host with no CUDA at all,
    and the CPU path is a real one for these tests.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class _TopKStream:
    """Exact top-`k` along the last axis, merged one tile at a time into a fixed-width buffer.

    `push` keeps the top `k` of the union of what it holds with the tile it is given, which is exact:
    the top-k of a union equals the top-k of the merged top-k lists, so the k-th value held after any
    push is the k-th value of everything pushed so far, and the buffer never has to see the whole
    axis at once. The alternative -- hold a threshold and drop what falls below it -- is exact only if
    the threshold is the running k-th value, which is what `values` holds anyway.

    A tile whose largest value is below the running k-th cannot contribute, so `push` returns without
    touching the buffer when `amax(tile) < amin(held)`. That test reads one boolean to the host and is
    therefore a synchronization; it is worth it because on a long prefill most key tiles of most query
    tiles are below the running k-th while the merge is the expensive part. It is skipped until the
    buffer is saturated, because before that `amin(held)` is not yet the k-th value. How often it can
    pay is a property of the caller's geometry rather than of the test: the prefix level narrows a
    4096-key tile into a 512-wide buffer, while `_stream_candidates` builds its stream with an `k`
    equal to its own tile width -- there a real 4096-token chunk gave **1 hit in 896 pushes**, and one
    hit in 896 is a merge saved against a synchronization paid on every push. See
    `INDEXER_CAND_SKIP_TEST` for the reading and for why the candidate level answers the question with
    `False`.

    **The skip is exact, not a tie-break, and that is what lets a capture drop it.** Every value held
    is above every value in the tile, so the k largest of the union are the k that `held` already has:
    the merge would return the same multisets and the `topk` below would pick exactly the entries the
    buffer already holds. It is a pure cost saving, and the reason it is not taken inside a capture --
    see `host_branch` -- is that a capture cannot branch on a value it is still computing, so the
    recorded body merges and then drops the tile instead of never merging it. The values and the
    positions that come out are the same either way, tie order aside, and `Indexer` sorts by position
    at the end.

    Tie order among equal values is whatever `torch.topk(sorted=False)` returns, so two streams fed
    the same values in a different order may name different positions. Both are valid top-k results
    and `Indexer` re-sorts by position afterwards, so the difference never reaches a caller.
    """

    __slots__ = ("k", "values", "keys", "host_branch")

    def __init__(self, k: int, host_branch: bool | None = None):
        self.k = k
        self.values: torch.Tensor | None = None
        self.keys: torch.Tensor | None = None
        # False inside a capture, where the early-out's read of device data is not permitted. Asked
        # here rather than by each caller so that every stream a captured body builds is right without
        # a caller having to remember, and overridable so a test can hold both behaviours side by side.
        self.host_branch = (not _recording()) if host_branch is None else host_branch

    def push(self, values: torch.Tensor, keys: torch.Tensor) -> None:
        k = self.k
        if self.values is None:
            if values.size(-1) <= k:
                self.values, self.keys = values, keys
            else:
                self.values, sel = values.topk(k, dim=-1, sorted=False)
                self.keys = keys.gather(-1, sel)
            return
        if (
            self.host_branch
            and self.values.size(-1) == k
            and bool((values.amax(dim=-1) < self.values.amin(dim=-1)).all())
        ):
            return
        merged = torch.cat([self.values, values], dim=-1)
        # fewer than `k` values seen so far means the buffer is still filling and there is nothing to
        # drop; only once it is saturated is its width the top-k's and the buffer stops growing.
        self.values, sel = merged.topk(min(k, merged.size(-1)), dim=-1, sorted=False)
        self.keys = torch.cat([self.keys, keys], dim=-1).gather(-1, sel)

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.values is not None and self.keys is not None, "a stream nothing was pushed into"
        return self.values, self.keys


_SIDE: dict[int, torch.cuda.Stream] = {}


def _side_stream(device) -> torch.cuda.Stream:
    """One side stream a device, kept across calls. A fresh `torch.cuda.Stream` brings its own event
    pool, and a 262144-token prefill reaches `_ReducePipeline` 8 indexer layers times 64 chunks."""
    index = device.index if device is not None and device.index is not None \
        else torch.cuda.current_device()
    stream = _SIDE.get(index)
    if stream is None:
        stream = _SIDE[index] = torch.cuda.Stream(device=device)
    return stream


class _ReducePipeline:
    """`tp.reduce` with a `depth`-tile lookahead: tile k's collective is issued on a side stream and
    joined `depth` tiles later, once those tiles' arithmetic has been enqueued on the compute stream.

    The loop this replaces is one blocking `all_reduce` a key tile, and on this fabric a `[2048, 4096]`
    fp32 tile is 5.4 ms of wire against 4.4 ms of arithmetic, so most of the loop is spent waiting for
    the fabric with the SMs idle. Nothing below the reduce reads *its own* tile's score: the next
    tile's einsum reads `index_k`, not `score`, and the one edge that has to be kept is reduce -> push.
    So the collective may be issued early, and the pushes still land in tile order because the joins
    are FIFO -- which is what keeps the top-k selection identical rather than merely equivalent.

    `/tmp/bench_indexer_reduce_overlap.py` prices it on the real fabric at these shapes: 9.83 ms a
    tile serial, 8.44 at a depth of one, **6.91 at two**, 6.98 at four and 7.11 at eight, against a
    4.40 ms floor with no collective at all. Two is where it saturates, and every depth's result is
    elementwise identical to the serial arm's.

    A capture gets the serial order, because a captured body may branch on nothing and can record
    neither an event nor a second stream.
    """

    __slots__ = ("reduce", "depth", "side", "pending", "events")

    def __init__(self, reduce, depth: int, device):
        self.reduce = reduce
        self.depth = depth
        self.side = _side_stream(device)
        self.pending: list[tuple[torch.Tensor, torch.cuda.Event, int, int]] = []
        # an event is only ever waited on before it is recorded again, so `depth` of them is all the
        # pipeline can have in flight and there is no reason to allocate one a tile
        self.events: list[torch.cuda.Event] = []

    @classmethod
    def build(cls, reduce, device) -> "_ReducePipeline | None":
        """`None` -- the serial call -- unless a lookahead was asked for outside a capture."""
        if reduce is None or INDEXER_REDUCE_DEPTH <= 1 or _recording():
            return None
        return cls(reduce, INDEXER_REDUCE_DEPTH, device)

    def push(self, score: torch.Tensor, k0: int, k1: int):
        """Issue this tile's collective; return the tile whose turn it now is, if any."""
        side = self.side
        stream = torch.cuda.current_stream(score.device)
        # the side stream has to see the tile's arithmetic before it reduces it
        side.wait_stream(stream)
        ready = self.events.pop() if self.events else torch.cuda.Event()
        with torch.cuda.stream(side):
            out = self.reduce(score)
            ready.record(side)
        # `out` is born on `side` and read on the compute stream, so it is the compute stream's use
        # that its memory cannot be handed out ahead of
        out.record_stream(stream)
        self.pending.append((out, ready, k0, k1))
        if len(self.pending) < self.depth:
            return None
        return self.pop()

    def pop(self):
        out, ready, k0, k1 = self.pending.pop(0)
        torch.cuda.current_stream(out.device).wait_event(ready)
        self.events.append(ready)
        return out, k0, k1

    def drain(self):
        """The tiles still in flight when the loop ends, in order."""
        while self.pending:
            yield self.pop()


def select_candidate_blocks(
    scores: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: which blocks of compressed positions a query may look in.

    `scores` is `[..., n_blocks]`, one entry per block, already the best score inside it, with a block
    the query cannot reach at `-inf`. Returns the ids it keeps, ascending and `-1` padded out to
    `min(topk_blocks, n_blocks)`.

    Ids rather than the dense mask this used to return: the mask is `[bsz, seqlen, width]`, which at
    256K positions is 68 GB of bool per candidate source, and level two only ever *reads* the mask to
    gather the blocks it names. The ids carry the same selection in 32 bytes a query, and they drop
    the consumer's need to score a width it is going to throw away -- the gather only touches the
    blocks that survived.

    Not what runs: this is the whole-axis form of the selection, kept because it is the one that can
    be read. `Indexer._stream_prefix` applies the same rule a tile at a time, and the pin below is
    the reason it cannot simply be a top-k there -- inside a stream, a pinned block has to earn its
    `+inf` from the tile that owns it rather than be appended afterwards, or a block that also scored
    its way in would appear twice and level two would gather its positions twice.
    """
    num_blocks = scores.size(-1)
    keep = min(topk_blocks, num_blocks)
    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=scores.device) == last, torch.inf)
    top = scores.topk(keep, dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: they are not
    # blocks the query may look in, so they become padding rather than a narrow mask
    picked = torch.where(top.values > -torch.inf, top.indices, -1)
    return picked.sort(dim=-1).values.int()


class Indexer(nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: FP4 query heads against one shared key per compressed position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        layer_id: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
        world: int = 1,
    ):
        super().__init__()
        kv_sources = _required(cfg, "kv_source_layers")
        candidate_source = cfg.candidate_source_layer
        self.layer_id = layer_id
        # the index keys are derived from the compressor's latent, so only a layer that compresses
        # its own KV can produce them; every other indexer reads them from that layer's cache
        self.owns_k = layer_id in kv_sources
        self.compress_ratio = _compress_ratio_at(cfg, layer_id)
        self.is_candidate_source = layer_id == candidate_source
        self.uses_candidates = candidate_source is not None and 0 <= candidate_source < layer_id
        self.candidate_topk_blocks = cfg.candidate_topk_blocks or 0
        self.candidate_block_size = cfg.candidate_block_size or 0
        # Which axis the world cuts: heads (`indexer_row_split` off) or query rows (on). The head
        # split can be taken out of a replicated `wq_b` at run time, so the module is built with all
        # the heads whenever the row split is available -- `_wq_b_local` below is that cut, and it is
        # the same tensor the loader would have written, row for row.
        self.row_split = indexer_row_split(world)
        idx_world = 1 if self.row_split else world
        self.n_heads = _required_int(cfg, "index_n_heads") // idx_world
        self.n_heads_global = _required_int(cfg, "index_n_heads")
        self.index_head_dim = _required_int(cfg, "index_head_dim")
        self.rope_head_dim = _required_int(cfg, "rope_head_dim")
        self.index_topk = _required_int(cfg, "index_topk")
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            _required_int(cfg, "q_lora_rank"),
            self.n_heads * self.index_head_dim,
            bias=False,
            dtype=LINEAR_DTYPE,
            device=device,
        )
        # The output width is global while the *parameter* is not cut at all: `weights_proj` is 32
        # numbers against a 5120-wide input, so slicing it would mean this rank's 8 output columns
        # are produced by heads 0-31's weights -- the same 8 numbers, for the sake of a slice that
        # saves 64 bytes. The forward takes the head offset out of the output instead.
        self.weights_proj = nn.Linear(
            _required_int(cfg, "dim"), self.n_heads_global, bias=False, dtype=LINEAR_DTYPE, device=device
        )
        self.tp = None
        # The head-split view of a replicated `wq_b`, cut once on the first forward that needs it.
        # Built lazily rather than in `__init__` because the parameter is filled by a load that runs
        # after the constructor, and cached rather than re-cut per step because a decode step is the
        # one place it is used and a per-step `contiguous` would be an allocation inside a capture.
        self._wq_b_local: torch.Tensor | None = None
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = nn.Linear(
                _required_int(cfg, "head_dim"),
                self.index_head_dim,
                bias=False,
                dtype=LINEAR_DTYPE,
                device=device,
            )

            self.k_norm = RMSNorm(self.index_head_dim, _required_float(cfg, "norm_eps"), device=device)
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len // self.compress_ratio,
                    self.index_head_dim,
                    dtype=CACHE_DTYPE,
                    device=device,
                ),
                persistent=False,
            )

    def _local_heads(self) -> int:
        """How wide this rank's slice of the index heads is, i.e. the head-split output width."""
        tp = self.tp
        return self.n_heads if tp is None else tp.index_heads

    def _wq_b_for_heads(self) -> torch.Tensor:
        """`wq_b`'s weight cut to this rank's index heads -- the tensor the loader would have written.

        Only reached under `indexer_row_split`, where the parameter is replicated and the *rows* are
        the thing that is sharded. A prefill takes the whole weight; a step whose length is below the
        world cannot be cut into bands, so it keeps the head split, and that has to be the same cut
        `ShardPlan.local_value` makes -- `[n_heads, index_head_dim, -1]` view, then a contiguous band
        of whole heads -- or the decode column would move for a reason that has nothing to do with
        the split.
        """
        tp = self.tp
        assert tp is not None, "the head split is only ever taken by a rank of a world"
        cached = self._wq_b_local
        if cached is None:
            lo = tp.rank * tp.index_heads
            cached = (
                self.wq_b.weight.view(self.n_heads_global, self.index_head_dim, -1)[
                    lo : lo + tp.index_heads
                ]
                .reshape(tp.index_heads * self.index_head_dim, -1)
                .contiguous()
            )
            self._wq_b_local = cached
        return cached

    def row_band(self, seqlen: int) -> tuple[int, int]:
        """Which query rows this rank's indexer computes, as `(row, length)`, or `(0, seqlen)`.

        The single place the row split is decided. `Indexer.forward` does not re-derive it -- it reads
        `seqlen_full` and knows it was handed a band -- so there is one expression that can be wrong
        here rather than two that can disagree, and this one is checked against the parameter the
        loader cut by the assertion in `forward`.

        `(0, seqlen)` covers every call that is not a row split: no world at all, the split switched
        off, and the chunk that cannot be cut. That last one is not an edge case to reason about but a
        length the split has nothing to say about -- a decode step is one query against a world of
        four, and a chunk of three is three -- and both keep the head split, which is the layout that
        works at any length. The band has to be whole for the gather to be a fixed-shape call, so a
        chunk that does not divide the world takes the whole-chunk path instead of a ragged band that
        would need an all-gather of a size the other ranks do not have.
        """
        tp = self.tp
        if not self.row_split or tp is None or seqlen < tp.world or seqlen % tp.world:
            return 0, seqlen
        band = seqlen // tp.world
        return tp.rank * band, band

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        pos: int | Pos,
        offset: int,
        shared: SharedAttentionRuntime,
        row: int = 0,
        seqlen_full: int | None = None,
    ) -> torch.Tensor:
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before `Attention` overwrites that same storage with
        the RoPE'd, quantized values.

        `row` and `seqlen_full` are the row split's: a rank takes the query rows
        `[row, row + seqlen)` of a `seqlen_full`-long chunk, so `x`/`qr` arrive pre-sliced and only
        the query side is local. The key side is not -- this layer's compressed latent and the index
        cache it writes are whole on every rank either way, and the cache is replicated, so the write
        is the same write and stays unconditional."""
        assert self.freqs_cis is not None, "the owning Attention sets this before the first forward"
        pos = Pos.of(pos)
        bsz, seqlen, _ = x.size()
        full = seqlen if seqlen_full is None else seqlen_full
        ratio, rd = self.compress_ratio, self.rope_head_dim

        # A key owner publishes its cache even when `latent` is None. The reference publishes only
        # inside the write, which leaves `index_k` unset on a forward that starts mid-group with no
        # earlier forward in the process to have set it -- the states differ, the tensor does not:
        # it is the same buffer either way, so publishing unconditionally cannot feed a consumer
        # anything the reference would not have.
        if self.owns_k:
            shared.index_k = self.k_cache
        # latent is None while a group is still filling up, so there is nothing to publish yet
        if self.owns_k and latent is not None:
            # The key side is the *chunk's* on both paths: `latent` arrives whole and the cache write
            # below is the chunk's, so the group count it has to be dated against is the chunk's as
            # well. A band shorter than the world can be one row long, and asking about the band's own
            # length would put a continuation chunk on the decode branch for a reason that is an
            # artifact of how the rows were dealt out.
            freqs = _group_freqs(self.freqs_cis, pos, latent.size(1), ratio, full)
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant(k, FP4_BLOCK_SIZE, True)
            self.k_cache[:bsz, pos.span(k.size(1), pos.group(ratio))] = k

        assert shared.index_k is not None, "an indexer needs a key cache, and no kv source has published one"
        # Everything below is the *query* side, and it is what a row band moves: `pos_q` is where
        # this rank's first row sits, so the RoPE rows, the reachable-group counts and the compressed
        # prefix are all this rank's own. A whole-chunk call has `row=0` and is byte for byte the
        # call this made before the split existed.
        pos_q = pos + row
        # The two ways the query side can be laid out. A row band completes its own score, so it
        # needs all the heads and no collective; anything narrower than the world keeps the head
        # split, which needs the local heads out of a replicated weight and the score reduced.
        # `tp.reduce` is the same call the head-split path has always made, and on the row path it is
        # not skipped-and-still-issued -- there is no partial score to complete.
        #
        # Which one this call is does not have to be re-derived here: `row_band` is the one place the
        # split is decided, and a caller that took a band says so by naming the chunk it came out of.
        # A second derivation from `world` and `full` would be a second chance to disagree with the
        # parameter the loader cut, and the disagreement would be a wrong answer rather than a crash.
        banded = seqlen_full is not None
        if banded:
            assert full % seqlen == 0 and full > seqlen, "a row band is one of several equal bands"
            assert self.n_heads == self.n_heads_global, (
                "a row band reads every index head of its own rows, so `wq_b` has to be whole: this "
                "module was built with the head split, which means the constructor's world and the "
                "loader's cut disagree about the axis"
            )
            q = self.wq_b(qr).unflatten(-1, (self.n_heads_global, self.index_head_dim))
            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads_global**-0.5)
            tp = None
        else:
            # `self.wq_b` is this rank's own width in the head split -- the constructor divided it --
            # and the *whole* width under the row split, where the parameter replicated and a call
            # narrower than the world has to take the head band out of it. `world=1` is the control
            # in both, and its module is the whole width, so it goes through the module either way.
            q = (
                F.linear(qr, self._wq_b_for_heads())
                if self.row_split and self.tp is not None
                else self.wq_b(qr)
            ).unflatten(-1, (self._local_heads(), self.index_head_dim))
            # `weights` is one number per index head, and the sum below runs over heads -- so this
            # rank needs its own slice of the output while the scale stays the *global* head count.
            # Using the local 8 in `n_heads**-0.5` would scale every index-source layer's partial by
            # 2x. A row band has all 32, so its slice is the whole output and the same scale is the
            # local count.
            tp = self.tp
            if tp is None:
                weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads_global**-0.5)
            else:
                lo = tp.rank * tp.index_heads
                weights = self.weights_proj(x)[..., lo : lo + tp.index_heads] * (
                    self.softmax_scale * self.n_heads_global**-0.5
                )
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[pos_q.span(seqlen)])
        fp4_act_quant(q, FP4_BLOCK_SIZE, True)

        # `end_pos // ratio` groups are reachable. Both paths read the whole cache and mask the
        # tail out of the score, rather than slicing to the reachable prefix: the prefix is a
        # step by step count, and reading one width on both paths is what keeps the graphed column
        # and the eager column the same arithmetic instead of two score matrices over different
        # widths. The prefill's `upto` is its own group count, so there the two coincide.
        #
        # The count is the *chunk's*, read off `pos` and not `pos_q`. `__add__(0)` returns `self`, so
        # on the whole-chunk path the two are the same position and this is the expression that has
        # always been here -- but on a row band they are not, and a band's last row is not the chunk's
        # last row. A rank that read up to its own last row would compute a smaller `width`, and with
        # it a smaller `topk`: the picks would come out the right length only by accident, and the
        # fixed-shape all-gather that puts the bands back together would refuse the two. Each band
        # reading the chunk's prefix is also what the reference reads, so nothing is masked twice and
        # nothing extra is scored -- the extra columns past a band's own reach are the same columns the
        # whole-chunk forward masks with the `compress_lens` below.
        reach = seqlen if seqlen_full is None else seqlen_full
        index_k = shared.index_k[:bsz, pos.upto(pos.group(ratio, reach), shared.index_k.size(1), reach)]

        # How many compressed positions each query can see: a block becomes visible once the query has
        # passed its last token, so a query at an absolute position `p` sees `(p + 1) // ratio` of
        # them. The prefill's is therefore one count per query counted from where the forward starts,
        # which on a later chunk is not 0 -- a chunk's first query can see the whole history in front
        # of it, and without that its blocks would all be masked away. A decode step has one query and
        # one number, and it stays a 0-dim *tensor*: it is a read of the recorded position, and
        # `_as_column` is what widens it where a shape needs it.
        if pos_q.on_device:
            compress_lens = pos_q.group(ratio, seqlen)
        else:
            compress_lens = ((pos_q.host + torch.arange(1, seqlen + 1, device=x.device)) // ratio).unsqueeze(-1)

        width = index_k.size(1)
        topk = min(self.index_topk, width)
        if topk == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)

        if self.uses_candidates:
            # level two: score with our own weights, but only inside the source's blocks. The assert
            # is the one thing the old dense mask checked for free -- it was shaped like this layer's
            # own scores, so a source with a different ratio could not have been consumed.
            assert shared.candidates is not None, "a candidate source must run before a candidate user"
            assert shared.candidates_ratio == ratio, "the candidate blocks are the source's positions, in the source's ratio"
            values, keys = self._stream_candidates(q, weights, index_k, shared.candidates, compress_lens, tp)
        else:
            values, keys, blocks = self._stream_prefix(
                q, weights, index_k, compress_lens, tp,
                self.candidate_block_size if self.is_candidate_source else 0,
            )
            if self.is_candidate_source:
                shared.candidates = publish(shared.candidates, blocks)
                shared.candidates_ratio = ratio

        # Re-sorted by position, and unreachable picks pushed to the end of the row rather than left
        # where they fell: the reference's `topk(...).indices.sort()` puts every `-1` after every
        # reachable id, and `sparse_attn` sums along this axis, so keeping that order is what makes
        # this the same arithmetic rather than merely the same set. `tail` is past every position the
        # row can name, so the ones that sort above it are exactly the unreachable ones.
        tail = width + offset + 1
        idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
        idx = torch.where(idxs < tail, idxs, -1).int()
        if banded:
            # A row band is complete *for its own rows*, which is what removes the score collective --
            # but `sparse_attn` is head-parallel, so every rank reads the picks of every query in the
            # chunk. They come back together here, and this is the only message the indexer sends on
            # this path. It is per index source and per chunk: `[bsz, chunk, index_topk]` int32, one
            # all-gather, against a fp32 score tile per key tile over the whole compressed prefix.
            #
            # `cat` in rank order reassembles the order `row_band` cut the bands in, so this is the
            # tensor a head-split rank would have produced rather than a permutation of it. Nothing
            # else on the row path needs one: `shared.candidates` names *blocks* and is read by the
            # indexer alone, on this rank, for this rank's own rows. Its two ends are in the same
            # coordinates for the same reason -- the source publishes from its band and the user
            # indexes `[:, q0:q1]` in its own, and `row_band` is a function of the chunk's length and
            # the rank alone, so every layer on this rank cuts the same band out of the same chunk
            # and both rows `i` are the same absolute query. A band cut per layer, or a source that
            # published its whole chunk, would be the one place the two could disagree.
            tp = self.tp
            assert tp is not None and tp.gather is not None, "a row band is a rank of a world with a gather"
            idx = tp.gather(idx)
        return idx

    def _stream_prefix(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        compress_lens: torch.Tensor | int,
        tp,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One tiled pass over the reachable prefix, yielding both levels of the two-level top-k.

        Returns the `index_topk` best positions per query with their key ids, plus -- when this layer
        is the candidate source, i.e. `block_size` is set -- the level-one block ids. Both come out of
        the same score, which is the expensive part; scoring twice to answer the two questions would
        double the cost of the one layer that is already the most expensive in the model.
        """
        bsz, seqlen, n_heads, _ = q.shape
        width = index_k.size(1)
        topk = min(self.index_topk, width)
        q_tile = min(seqlen, INDEXER_QUERY_TILE)
        # the budget is on `q_tile * n_heads * key_tile`, which is what the einsum materializes
        key_tile = max(INDEXER_MIN_KEY_TILE, INDEXER_SCORE_BUDGET // (n_heads * q_tile))
        if block_size:
            # a block may not straddle two tiles, or its amax would be split across two pushes
            key_tile = -(-key_tile // block_size) * block_size
        key_tile = min(key_tile, width)

        # The one thing level one does that a plain top-k does not: the block holding a query's newest
        # position is only partly filled, so it holds the most recent tokens and could otherwise be
        # outscored by an older, full block. It is pinned by giving it `+inf` as its tile goes past --
        # inside the same top-k rather than appended to it, because a block that also earned its place
        # on score would then be in the list twice and level two would gather its positions twice.
        pin = (
            (_as_column(compress_lens, bsz, seqlen, q.device, torch.int32) - 1) // block_size
            if block_size
            else None
        )

        # One stream per query tile, concatenated along the query axis at the end. A stream merges
        # along the key axis only, so a second query tile pushed into the same buffer would be merged
        # as more keys of the first: its rows would be dropped and the buffer would keep the first
        # tile's row count. That hides for as long as the whole chunk fits in one tile, which is
        # every length a toy test uses and none of the lengths this exists for.
        out_values: list[torch.Tensor] = []
        out_keys: list[torch.Tensor] = []
        out_blocks: list[torch.Tensor] = []
        # one pipeline for the whole call: it is drained at the foot of every query tile, so nothing
        # of one tile's is still in flight when the next tile's first collective is issued
        #
        # `tp is None` is two cases and both want the same answer: there is no world at all, or there
        # is one and this call is a row band -- whose score is whole already, because the sum above ran
        # over every index head of its own rows. `forward` is the one place that decides, by not
        # carrying a world into its banded branch, so a band reaches the loop below with nothing to
        # reduce rather than with a reduce it then declines to issue.
        # The reduce is bound to `discrete` here and not at the call below, because the pipeline holds
        # the closure rather than the site: a score's top-k reads the sum, so this message stays fp32
        # whatever the activation wire is set to, and `tp.py` says why the two kinds differ.
        pipe = _ReducePipeline.build(None if tp is None else partial(tp.reduce, discrete=True),
                                     q.device)
        for q0 in range(0, seqlen, q_tile):
            q1 = min(q0 + q_tile, seqlen)
            q_tile_t = q[:, q0:q1]
            weights_t = weights[:, q0:q1]
            # widened through `_as_column` rather than sliced, so that the shapes it can be handed --
            # a per-query column on the host path, a 0-dim tensor on the device one -- all arrive as
            # one column per query, which is what the masks below are shaped against
            lens_t = _as_column(compress_lens, bsz, seqlen, q.device, torch.int32)[:, q0:q1]
            positions = _TopKStream(topk)
            blocks = _TopKStream(self.candidate_topk_blocks) if block_size else None

            def flush(score: torch.Tensor, k0: int, k1: int) -> None:
                """Everything after the collective, for one tile: the mask and the two pushes.

                A closure rather than inline code because the pipelined order calls it on a tile whose
                index is not the loop's, and the tail has to be the same code in both orders.
                """
                ids = torch.arange(k0, k1, dtype=torch.int32, device=q.device).reshape(1, 1, -1)
                score = score.masked_fill(ids >= lens_t, -torch.inf)
                if blocks is not None:
                    # `-inf` pads the tail of the last tile out, so that a block is always whole. The
                    # pad belongs to the block push alone: the position push below carries its own
                    # `ids`, and padding the score it shares would leave values one column wider than
                    # the keys they are picked with.
                    pad = -(-(k1 - k0) // block_size) * block_size - (k1 - k0)
                    block_score = F.pad(score, (0, pad), value=-torch.inf) if pad else score
                    block_ids = torch.arange(
                        k0 // block_size, -(-k1 // block_size), dtype=torch.int32, device=q.device
                    ).reshape(1, 1, -1)
                    blocks.push(
                        block_score.unflatten(-1, (-1, block_size))
                        .amax(dim=-1)
                        .masked_fill(block_ids == pin[:, q0:q1], torch.inf),
                        block_ids.expand(bsz, q1 - q0, -1),
                    )
                positions.push(score, ids.expand(bsz, q1 - q0, -1))

            for k0 in range(0, width, key_tile):
                k1 = min(k0 + key_tile, width)
                score = torch.einsum("bqhd,btd->bqht", q_tile_t, index_k[:, k0:k1])
                score = (score.relu_() * weights_t.unsqueeze(-1)).sum(dim=2)
                if pipe is not None:
                    # the collective is issued now and joined `depth` tiles from here, so this tile's
                    # tail is not the one the loop is on
                    due = pipe.push(score, k0, k1)
                    if due is not None:
                        flush(*due)
                else:
                    # `tp` is this rank's world on the head split, where the sum above ran over this
                    # rank's heads only and the score is a partial. It has to be whole before any
                    # top-k, because a top-k over partials is a different selection and the difference
                    # is discrete -- it does not average away downstream the way a rounding does.
                    # Tiled, this is one collective a tile, which is the same volume as the one it
                    # replaced and a fixed count for a fixed width. On a row band `tp` is None and the
                    # score is complete for exactly these rows, so the top-k below is the unsharded
                    # top-k and no collective happens at any tile.
                    if tp is not None:
                        score = tp.reduce(score, discrete=True)
                    flush(score, k0, k1)
            if pipe is not None:
                for due in pipe.drain():
                    flush(*due)

            tile_values, tile_keys = positions.result()
            out_values.append(tile_values)
            out_keys.append(tile_keys)
            if blocks is not None:
                block_values, block_keys = blocks.result()
                # a block the query cannot reach yet scored -inf and is not a block it may look in,
                # the same way `select_candidate_blocks` drops the picks that came back -inf
                out_blocks.append(
                    torch.where(block_values > -torch.inf, block_keys, -1).sort(dim=-1).values.int()
                )

        values = torch.cat(out_values, dim=1)
        keys = torch.cat(out_keys, dim=1)
        if blocks is None:
            return values, keys, None
        return values, keys, torch.cat(out_blocks, dim=1)

    def _stream_candidates(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k: torch.Tensor,
        candidates: torch.Tensor,
        compress_lens: torch.Tensor | int,
        tp,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Level two: the same score, computed only inside the blocks level one kept.

        `index_k` is *gathered* per query here rather than read as a prefix, because each query kept
        a different set of blocks. That gather is the whole point of the level: the version this
        replaces scored the full width with its own weights and then masked more than 99% of it away,
        so the multiplies outside the mask were paid for and discarded.
        """
        bsz, seqlen, _, _ = q.shape
        width = index_k.size(1)
        block_size = self.candidate_block_size
        q_tile = min(seqlen, INDEXER_CAND_QUERY_TILE)
        span = INDEXER_CAND_TILE * block_size
        k = min(self.index_topk, width)

        # Same reason as `_stream_prefix`: one stream per query tile, joined along the query axis.
        # This tile is the smaller of the two, so the whole chunk stops fitting in one as soon as the
        # chunk is longer than `INDEXER_CAND_QUERY_TILE` -- which is the length every real prefill has.
        out_values: list[torch.Tensor] = []
        out_keys: list[torch.Tensor] = []
        for q0 in range(0, seqlen, q_tile):
            q1 = min(q0 + q_tile, seqlen)
            q_tile_t = q[:, q0:q1]
            weights_t = weights[:, q0:q1]
            # widened through `_as_column` rather than sliced, so that the shapes it can be handed --
            # a per-query column on the host path, a 0-dim tensor on the device one -- all arrive as
            # one column per query, which is what the masks below are shaped against
            lens_t = _as_column(compress_lens, bsz, seqlen, q.device, torch.int32)[:, q0:q1]
            # the positions the surviving blocks name, `[bsz, q_tile, n_kept * block_size]`, with a
            # `-1` for a padded block. Built per query tile rather than for the whole sequence: at
            # 256K this is 16384 ids a query and the full-sequence form is 34 GB.
            keys = torch.add(
                candidates[:, q0:q1].unsqueeze(-1) * block_size,
                torch.arange(block_size, dtype=torch.int32, device=q.device),
            ).flatten(-2)
            # `host_branch=None` leaves the decision to `_TopKStream`, which is what the shipped code
            # did; `False` is this level's own answer -- see `INDEXER_CAND_SKIP_TEST`, which is also
            # what a capture builds, so the eager path matches the recorded one instead of paying for
            # a test whose hits are worth less than its read. The knob is read here rather than at the
            # top of the loop so that a probe can set it between arms.
            positions = _TopKStream(k, host_branch=None if INDEXER_CAND_SKIP_TEST else False)
            for c0 in range(0, keys.size(-1), span):
                c1 = min(c0 + span, keys.size(-1))
                tile = keys[:, :, c0:c1]
                # a block at the end of the cache covers positions past its width, and a padded
                # block is a `-1`: both are gathered from a valid slot and masked out below, so the
                # gather index never leaves the cache
                gathered = index_k[
                    torch.arange(bsz, device=q.device).reshape(bsz, 1, 1), tile.clamp(0, width - 1)
                ]
                score = torch.einsum("bqhd,bqmd->bqhm", q_tile_t, gathered)
                score = (score.relu_() * weights_t.unsqueeze(-1)).sum(dim=2)
                # As in `_stream_prefix`: `tp` is the world on the head split and None on a row band,
                # whose score is whole for its own rows and needs no collective before the top-k. The
                # reduce is `discrete`, as it is there and for the same reason.
                if tp is not None:
                    score = tp.reduce(score, discrete=True)
                # a padded block names position -1 and a query cannot see past its own group count;
                # both are the same `-inf` the prefix path masks with
                score = score.masked_fill((tile < 0) | (tile >= lens_t), -torch.inf)
                positions.push(score, tile)

            tile_values, tile_keys = positions.result()
            out_values.append(tile_values)
            out_keys.append(tile_keys)

        return torch.cat(out_values, dim=1), torch.cat(out_keys, dim=1)


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attn` call: a
    sliding window of raw KV, plus -- when `compress_ratio > 0` -- `index_topk` compressed positions
    reaching further back. Q and the output projection are both low-rank, the latter grouped.

    `compress_ratio > 0` does not mean the layer compresses its own KV: only `kv_source_layers` do,
    the rest read that same cache.
    """

    def __init__(
        self,
        layer_id: int,
        cfg: V41TextConfig,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        world: int = 1,
    ):
        super().__init__()
        if max_seq_len is None:
            max_seq_len = cfg.original_seq_len or cfg.max_position_embeddings
        if not max_seq_len:
            raise ValueError("a max_seq_len is required: the config states neither original_seq_len nor max_position_embeddings")
        device = canonical_device(device)
        self.layer_id = layer_id
        self.dim = _required_int(cfg, "dim")
        # `world` divides the heads and the groups, and nothing else here: wq_a, wkv and the whole
        # compressor replicate, for the reasons in `tp.py`. The two counts are local so that every
        # `unflatten`/`view` below is local too, and there is no place a global head count still
        # needs to be right except `attn_sink`, whose width is the thing being cut.
        self.world = world
        self.n_heads = _required_int(cfg, "n_heads") // world
        self.head_dim = _required_int(cfg, "head_dim")
        self.rope_head_dim = _required_int(cfg, "rope_head_dim")
        self.n_groups = _required_int(cfg, "o_groups") // world
        self.o_lora_rank = _required_int(cfg, "o_lora_rank")
        self.window_size = _required_int(cfg, "window_size")
        self.softmax_scale = self.head_dim**-0.5
        self.eps = _required_float(cfg, "norm_eps")

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32, device=device))
        self.wq_a = nn.Linear(
            self.dim, _required_int(cfg, "q_lora_rank"), bias=False, dtype=LINEAR_DTYPE, device=device
        )
        self.q_norm = RMSNorm(self.wq_a.out_features, self.eps, device=device)
        self.wq_b = nn.Linear(
            self.wq_a.out_features, self.n_heads * self.head_dim, bias=False, dtype=LINEAR_DTYPE, device=device
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=LINEAR_DTYPE, device=device)
        self.kv_norm = RMSNorm(self.head_dim, self.eps, device=device)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            dtype=LINEAR_DTYPE,
            device=device,
        )
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, self.dim, bias=False, dtype=LINEAR_DTYPE, device=device
        )
        self.tp = None

        n_layers = cfg.n_layers if cfg.n_layers is not None else len(cfg.compress_ratios)
        is_backbone = layer_id < n_layers
        self.compress_ratio = _compress_ratio_at(cfg, layer_id)
        self.is_kv_source = is_backbone and layer_id in _required(cfg, "kv_source_layers")
        self.is_index_source = is_backbone and layer_id in _required(cfg, "index_source_layers")
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(cfg, layer_id, max_batch_size, device=device)
        if self.is_index_source:
            self.indexer = Indexer(cfg, layer_id, max_batch_size, max_seq_len, device=device, world=world)

        self.register_buffer(
            "window_kv_cache",
            torch.zeros(max_batch_size, self.window_size, self.head_dim, dtype=CACHE_DTYPE, device=device),
            persistent=False,
        )
        if self.is_kv_source:
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(
                    max_batch_size,
                    max_seq_len // self.compress_ratio,
                    self.head_dim,
                    dtype=CACHE_DTYPE,
                    device=device,
                ),
                persistent=False,
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = _required_int(cfg, "original_seq_len"), _required_float(
                cfg, "compress_rope_theta"
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, _required_float(cfg, "rope_theta")
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.rope_head_dim,
                max_seq_len,
                original_seq_len,
                rope_theta,
                _required_float(cfg, "rope_factor"),
                _required_int(cfg, "beta_fast"),
                _required_int(cfg, "beta_slow"),
                device,
            ),
            persistent=False,
        )

    def reset_state(self, batch_size: int) -> None:
        """Drop the cached window, compressed KV and index keys for `batch_size` rows.

        The reference never resets -- one conversation per process, and every slot is overwritten
        before it is read. That is only true of a forward that continues the previous one, so a
        second independent forward over the same module is not, and neither is a decode step that
        skips ahead. This exists for tests and for callers that reuse a module across sequences.
        """
        self.window_kv_cache[:batch_size].zero_()
        if self.is_kv_source:
            self.compress_kv_cache[:batch_size].zero_()
            assert self.compressor is not None
            self.compressor.reset_state(batch_size)
        if self.indexer is not None and self.indexer.owns_k:
            self.indexer.k_cache[:batch_size].zero_()

    def _window_kv(self, x, freqs_cis, pos):
        """This layer's sliding-window K and the window positions every query may attend to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included."""
        pos = Pos.of(pos)
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(kv, FP8_BLOCK_SIZE, SCALE_FMT, SCALE_DTYPE, True)
        if pos.first():  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            window_kv = kv
        elif _is_continuation(pos, seqlen):  # a later chunk: the window spans the ring and the chunk
            # The K handed on is the ring and the chunk concatenated rather than the ring alone,
            # because this chunk's queries attend to their own rows as well; `get_window_topk_idxs`
            # names a position by which half it fell in, and the ring half has to be read *before* the
            # ring is advanced. The chunk's rows go into the slots their absolute positions name --
            # `slot = position % window_size` -- and those are exactly the slots the `window_size`
            # positions just in front of the chunk live in, so writing first would overwrite the rows
            # this chunk's own queries are about to read. `cat` materializes, so the ring can be
            # advanced after it. Only the last `window_size` rows are kept: a chunk longer than the
            # ring wraps away rows no later query can see again.
            window_kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)
            keep = min(seqlen, win)
            slots = (pos.host + torch.arange(seqlen - keep, seqlen, device=kv.device)) % win
            self.window_kv_cache[:bsz].index_copy_(1, slots, kv[:, seqlen - keep :])
        else:  # decode: one token into the ring buffer, attend over the whole window
            write_row(self.window_kv_cache[:bsz], pos.slot(win), kv.squeeze(1))
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, pos, device=x.device)

    def _compress_topk_idxs(self, x, qr, latent, pos, offset, shared):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published."""
        if not self.is_index_source:
            assert shared.topk_idxs is not None, "an index source must run before a layer that reuses its indices"
            return shared.topk_idxs

        pos = Pos.of(pos)
        bsz, seqlen, _ = x.size()
        if (pos.host + seqlen) // self.compress_ratio == 0:
            # no group has closed yet, so there is nothing to index. Asked on the host from `pos.host`
            # rather than from the device-side group count, which on the graph path is a tensor and
            # cannot be tested -- and the answer is the same one, because it is the same arithmetic.
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            # The row split's band, which is `(0, seqlen)` for every configuration that is not one.
            # `x`/`qr` are handed over as views of the chunk's own rows -- a band moves the query side
            # only, so the *key* side (`latent`, the cache write, the window) stays whole and stays
            # this layer's, and slicing here rather than inside keeps the compressor the one caller
            # that reads the whole chunk. `pos`/`offset` are the chunk's, not the band's: what a band
            # changes is where its first row sits, and `forward` asks for that with `row`.
            row, band = self.indexer.row_band(seqlen)
            if band == seqlen:
                idxs = self.indexer(x, qr, latent, pos, offset, shared)
            else:
                idxs = self.indexer(
                    x[:, row : row + band],
                    qr[:, row : row + band],
                    latent,
                    pos,
                    offset,
                    shared,
                    row=row,
                    seqlen_full=seqlen,
                )
        shared.topk_idxs = publish(shared.topk_idxs, idxs)
        return shared.topk_idxs

    def _compress_kv(self, x, qr, pos, offset, shared):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache."""
        pos = Pos.of(pos)
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        latent = None
        if self.is_kv_source:
            assert self.compressor is not None
            latent = self.compressor(x, pos)
            shared.compress_kv = self.compress_kv_cache
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, pos, offset, shared)
        if latent is not None:
            freqs = _group_freqs(self.freqs_cis, pos, latent.size(1), ratio, seqlen)
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
            fp4_act_quant_e4m3(latent, COMPRESS_KV_BLOCK_SIZE, True)
            self.compress_kv_cache[:bsz, pos.span(latent.size(1), pos.group(ratio))] = latent
        # read after the write, so this does not depend on the slice aliasing the cache
        assert shared.compress_kv is not None, "a kv source must run before a layer that reads its cache"
        return shared.compress_kv[:bsz, pos.upto(pos.group(ratio, seqlen), shared.compress_kv.size(1), seqlen)], idxs

    def forward(self, x: torch.Tensor, start_pos: int | Pos, shared: SharedAttentionRuntime) -> torch.Tensor:
        pos = Pos.of(start_pos)
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[pos.span(seqlen)]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, pos, kv.size(1), shared)
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. The checkpoint stores it fp8; a loader would dequantize it to bf16. The local
        # heads are exactly the local groups -- o_groups divides n_heads and this rank took a
        # contiguous run of heads, so 16 heads from 16r is groups 2r and 2r+1 whole -- which is why
        # wo_a needs no collective of its own.
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        y = self.wo_b(o.flatten(2))
        # wo_b is row-parallel, so this is the partial sum over groups and the one collective the
        # attention half of a layer needs.
        tp = self.tp
        return y if tp is None else tp.reduce(y)


class AttentionStack(nn.Module):
    """The attention half of the V4.1 backbone: one `Attention` per layer, run in order.

    The reference interleaves these with MoE, Engram and mHC residual mixing inside `Block`; none of
    that is here, so this runs attention over a stream that only attention has touched. It is the
    unit the smoke test drives, not a model.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        world: int = 1,
    ):
        super().__init__()
        n_layers = cfg.n_layers if cfg.n_layers is not None else len(cfg.compress_ratios)
        self.layers = nn.ModuleList(
            Attention(layer_id, cfg, max_batch_size, max_seq_len, device=device, world=world)
            for layer_id in range(n_layers)
        )
        self.shared = SharedAttentionRuntime()

    def reset_state(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset_state(batch_size)
        self.shared.__init__()

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, start_pos, self.shared)
        return x


def _required(cfg: V41TextConfig, name: str):
    value = getattr(cfg, name)
    if value is None:
        raise ValueError(f"the V4.1 attention layers need `{name}`, and this config does not state it")
    return value


def _compress_ratio_at(cfg: V41TextConfig, layer_id: int) -> int:
    """`compress_ratios[layer_id]`, with the two ways that lookup can miss a layer called out.

    The tuple defaults to empty rather than to None, so `_required` does not catch a config that
    never set it -- it just indexes off the end, which is a worse error message than this one.
    """
    ratios = _required(cfg, "compress_ratios")
    if not ratios:
        raise ValueError("the V4.1 attention layers need `compress_ratios`, and this config states none")
    if not 0 <= layer_id < len(ratios):
        raise ValueError(f"layer {layer_id} has no compress_ratio: the tuple covers {len(ratios)} layers")
    return ratios[layer_id]


def _required_int(cfg: V41TextConfig, name: str) -> int:
    value = _required(cfg, name)
    if not isinstance(value, int):
        raise ValueError(f"`{name}` is {value!r}, which is not an int")
    return value


def _required_float(cfg: V41TextConfig, name: str) -> float:
    value = _required(cfg, name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{name}` is {value!r}, which is not a number")
    return float(value)
