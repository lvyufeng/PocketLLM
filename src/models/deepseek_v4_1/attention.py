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
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.ops import act_quant, fp4_act_quant, sparse_attn
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.kernels import fp4_act_quant_e4m3

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
LINEAR_DTYPE = torch.bfloat16
CACHE_DTYPE = torch.bfloat16


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


def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Which sliding-window cache slots each query attends to; `-1` marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.

    `device` is where the table has to end up, because `Attention.forward` concatenates it against
    the card's own tensors -- a CPU table fails there the moment the tree runs anywhere but the host,
    and the shapes are equal enough that only the `cat` catches it. The table is *built* on the host
    and moved, not built in place: at the config's window of 128 a decode step's table is 512 bytes,
    while the eight `arange`/`clamp`/`where` kernels it takes to build one are eight launches on a
    stream whose busy fraction this round exists to raise. `None` -- the host's own answer -- stays
    the default, so the host path issues exactly what it issued before.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
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

    The reference holds this in a module-level singleton because there is one model per process.
    It is passed explicitly here so that two stacks in one process cannot silently share a cache.
    """

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


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
        self.norm = RMSNorm(head_dim, _required_float(cfg, "norm_eps"), device=device)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        # `bias=False` throughout this module, as the reference's `Linear` defaults to and as the
        # checkpoint is: no V4.1 projection carries one.
        self.wkv = nn.Linear(
            _required_int(cfg, "dim"),
            head_dim,
            bias=False,
            dtype=torch.float32 if ratio > 1 else torch.bfloat16,
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

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # trailing partial group waits in the state
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # one token per step: fill a slot, and pool only when the group just completed
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is `[..., n_positions]` with positions the query cannot reach already at `-inf`, which
    is what makes a block score of `-inf` mean "not reachable yet". `compress_lens` is a plain int
    during decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask
    shaped like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


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
        self.n_heads = _required_int(cfg, "index_n_heads") // world
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

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
        shared: SharedAttentionRuntime,
    ) -> torch.Tensor:
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before `Attention` overwrites that same storage with
        the RoPE'd, quantized values."""
        assert self.freqs_cis is not None, "the owning Attention sets this before the first forward"
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # A key owner publishes its cache even when `latent` is None. The reference publishes only
        # inside the write, which leaves `index_k` unset on a forward that starts mid-group with no
        # earlier forward in the process to have set it -- the states differ, the tensor does not:
        # it is the same buffer either way, so publishing unconditionally cannot feed a consumer
        # anything the reference would not have.
        if self.owns_k:
            shared.index_k = self.k_cache
        # latent is None while a group is still filling up, so there is nothing to publish yet
        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant(k, FP4_BLOCK_SIZE, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k

        assert shared.index_k is not None, "an indexer needs a key cache, and no kv source has published one"
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant(q, FP4_BLOCK_SIZE, True)

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        # `weights` is one number per index head, and the sum below runs over heads -- so this rank
        # needs its own slice of the output while the scale stays the *global* head count. Using the
        # local 8 in `n_heads**-0.5` would scale every index-source layer's partial by 2x.
        tp = self.tp
        if tp is None:
            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads_global**-0.5)
        else:
            lo = tp.rank * tp.index_heads
            weights = self.weights_proj(x)[..., lo : lo + tp.index_heads] * (
                self.softmax_scale * self.n_heads_global**-0.5
            )
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if tp is not None:
            # the sum above ran over this rank's heads only, so the score is a partial. It has to be
            # whole before the top-k, because a top-k over partials is a different selection and the
            # difference is discrete -- it does not average away downstream the way a rounding does.
            index_score = tp.reduce(index_score)

        # how many compressed positions each query can see: a block becomes visible once the query
        # has passed its last token. One query per decode step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # level two: score with our own weights, but only inside the source's candidate blocks
            assert shared.candidates is not None, "a candidate source must run before a candidate user"
            index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


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

    def _window_kv(self, x, freqs_cis, start_pos):
        """This layer's sliding-window K and the window positions every query may attend to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(kv, FP8_BLOCK_SIZE, SCALE_FMT, SCALE_DTYPE, True)
        if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            window_kv = kv
        else:  # decode: one token into the ring buffer, attend over the whole window
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, device=x.device)

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len, shared):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published."""
        if not self.is_index_source:
            assert shared.topk_idxs is not None, "an index source must run before a layer that reuses its indices"
            return shared.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(x, qr, latent, start_pos, offset, shared)
        shared.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset, shared):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            assert self.compressor is not None
            latent = self.compressor(x, start_pos)
            shared.compress_kv = self.compress_kv_cache
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len, shared)
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
            fp4_act_quant_e4m3(latent, COMPRESS_KV_BLOCK_SIZE, True)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        # read after the write, so this does not depend on the slice aliasing the cache
        assert shared.compress_kv is not None, "a kv source must run before a layer that reads its cache"
        return shared.compress_kv[:bsz, :compress_len], idxs

    def forward(self, x: torch.Tensor, start_pos: int, shared: SharedAttentionRuntime) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1), shared)
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
