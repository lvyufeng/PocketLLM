from __future__ import annotations

import math
import os as _os
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.components.gguf.tp_logits import distributed_argmax_local_logits, gather_sharded_logits
from src.components.moe.spec import metadata_float, metadata_int
from src.loader.gguf.bundle import GGUFBundle
from src.models.glm_dsa.spec import GLMDSASpec


# ---------------------------------------------------------------------------
# Opt-in decode profiler (GLM_PROFILE=1). Accumulates wall time per named
# section, synchronizing CUDA around each so the numbers are real (not just
# launch-queue time). Zero overhead when disabled: _prof_section short-circuits
# before touching the clock or the CUDA device. Print the breakdown via
# glm_profile_report() (the CLI does this on rank 0 after decode).
# ---------------------------------------------------------------------------
_GLM_PROFILE = _os.getenv("GLM_PROFILE", "0") == "1"
_GLM_PROFILE_TOTALS: dict[str, float] = {}
_GLM_PROFILE_COUNTS: dict[str, int] = {}
_GLM_PROFILE_PHASE = "decode"


def _glm_profile_set_phase(seqlen: int) -> None:
    # seqlen==1 is a decode step; anything longer is prefill. Sections are keyed
    # by phase so the prefill forward's per-layer time does not skew the decode
    # per-call averages (decode is what we're optimizing).
    global _GLM_PROFILE_PHASE
    _GLM_PROFILE_PHASE = "decode" if int(seqlen) == 1 else "prefill"


@contextmanager
def _prof_section(name: str):
    if not _GLM_PROFILE:
        yield
        return
    key = f"{_GLM_PROFILE_PHASE}.{name}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = _time.perf_counter() - t0
        _GLM_PROFILE_TOTALS[key] = _GLM_PROFILE_TOTALS.get(key, 0.0) + dt
        _GLM_PROFILE_COUNTS[key] = _GLM_PROFILE_COUNTS.get(key, 0) + 1


def glm_profile_enabled() -> bool:
    return _GLM_PROFILE


# Per-(phase) tally of how many local experts each MoE layer activated, so we
# can quantify TP rank-skew: at decode T=1 the top_k=8 experts scatter across
# the 4 EP ranks unevenly, and the slowest rank's staging gates the per-layer
# all_reduce.  Recording the local active count per layer-call lets us report
# the distribution (and, gathered across ranks, the per-step max-vs-mean skew).
_GLM_PROFILE_ACTIVE: list[int] = []


def _glm_profile_record_active(n_local: int) -> None:
    if _GLM_PROFILE and _GLM_PROFILE_PHASE == "decode":
        _GLM_PROFILE_ACTIVE.append(int(n_local))


def glm_profile_reset() -> None:
    _GLM_PROFILE_TOTALS.clear()
    _GLM_PROFILE_COUNTS.clear()
    _GLM_PROFILE_ACTIVE.clear()


def glm_profile_report() -> str:
    if not _GLM_PROFILE_TOTALS:
        return "GLM_PROFILE: no sections recorded"
    lines = ["GLM_PROFILE per-section totals (seconds / calls / ms-per-call):"]
    total = sum(_GLM_PROFILE_TOTALS.values())
    for name in sorted(_GLM_PROFILE_TOTALS, key=lambda k: -_GLM_PROFILE_TOTALS[k]):
        secs = _GLM_PROFILE_TOTALS[name]
        n = _GLM_PROFILE_COUNTS.get(name, 0)
        per = 1000.0 * secs / n if n else 0.0
        pct = 100.0 * secs / total if total else 0.0
        lines.append(f"  {name:20s} {secs:9.4f}s  {n:6d}x  {per:8.3f}ms  {pct:5.1f}%")
    return "\n".join(lines)


def glm_profile_skew_report() -> str:
    """Quantify EP rank-skew from THIS rank's samples only (no collective, so it
    is safe to call after the process group is torn down).  At decode T=1 the
    top_k=8 experts scatter unevenly across the 4 EP ranks; the slowest
    (max-active) rank's staging gates that layer's all_reduce.  Each rank prints
    its own mean/max/min local active count tagged by rank; comparing the four
    lines gives the cross-rank skew (max-rank-mean / avg-rank-mean)."""
    local = list(_GLM_PROFILE_ACTIVE)
    if not local:
        return "  [skew] no decode activation samples"
    # RANK from the env is stable even after the process group is destroyed.
    rank = int(_os.environ.get("RANK", "0"))
    import statistics
    mean = statistics.fmean(local)
    mx = max(local)
    mn = min(local)
    # top_k=8 is the global activation budget; on 4 ranks a perfectly balanced
    # (TP) split would stage 8/world per rank each layer.
    return (
        f"  [skew] rank={rank} decode layer-calls={len(local)} "
        f"local_active mean={mean:.3f} min={mn} max={mx} "
        f"(TP-balanced would be top_k/world; compare mean across the 4 rank lines)"
    )


@dataclass(frozen=True)
class GLMDSAArgs:
    n_layers: int
    leading_dense_layers: int
    dim: int
    vocab_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    key_mla_dim: int
    value_dim: int
    value_mla_dim: int
    rope_dim: int
    rope_base: float
    indexer_heads: int
    indexer_key_dim: int
    dense_inter_dim: int
    n_routed_experts: int
    top_k: int
    moe_inter_dim: int
    n_shared_experts: int
    norm_eps: float
    context_length: int
    expert_weights_norm: bool
    expert_weights_scale: float

    @classmethod
    def from_bundle(cls, bundle: GGUFBundle, *, n_layers: int | None = None) -> "GLMDSAArgs":
        spec = GLMDSASpec()
        params = spec.parse_params(bundle)
        md = bundle.metadata
        # ``params.n_layers`` is the physical block_count (e.g. 79), which
        # includes the trailing NextN/MTP block(s).  Those are speculative-decode
        # layers, not part of the main transformer trunk, and must NOT be run in
        # the forward residual stream (llama.cpp skips them: "unused tensor
        # blk.<last>").  The trunk depth is block_count - nextn_predict_layers.
        nextn_layers = metadata_int(md, "glm-dsa.nextn_predict_layers", 0)
        full_layers = max(0, int(params.n_layers) - int(nextn_layers))
        requested_layers = full_layers if n_layers is None else int(n_layers)
        return cls(
            n_layers=max(0, min(requested_layers, full_layers)),
            leading_dense_layers=metadata_int(md, "glm-dsa.leading_dense_block_count", 0),
            dim=int(params.hidden_size),
            vocab_size=int(params.vocab_size),
            n_heads=int(params.n_heads),
            n_kv_heads=int(params.n_kv_heads),
            head_dim=int(params.head_dim),
            q_lora_rank=metadata_int(md, "glm-dsa.attention.q_lora_rank", 0),
            kv_lora_rank=metadata_int(md, "glm-dsa.attention.kv_lora_rank", 0),
            key_mla_dim=metadata_int(md, "glm-dsa.attention.key_length_mla", 0),
            value_dim=metadata_int(md, "glm-dsa.attention.value_length", 0),
            value_mla_dim=metadata_int(md, "glm-dsa.attention.value_length_mla", 0),
            rope_dim=int(params.rope_dim or 0),
            rope_base=float(params.rope_base or 10000.0),
            indexer_heads=metadata_int(md, "glm-dsa.attention.indexer.head_count", 0),
            indexer_key_dim=metadata_int(md, "glm-dsa.attention.indexer.key_length", 0),
            dense_inter_dim=metadata_int(md, "glm-dsa.feed_forward_length", int(params.hidden_size) * 2),
            n_routed_experts=int(params.n_routed_experts),
            top_k=int(params.top_k),
            moe_inter_dim=int(params.expert_intermediate_size),
            n_shared_experts=int(params.n_shared_experts),
            norm_eps=float(params.norm_eps or 1.0e-6),
            context_length=int(params.context_length),
            expert_weights_norm=bool(md.get("glm-dsa.expert_weights_norm", False)),
            expert_weights_scale=metadata_float(md, "glm-dsa.expert_weights_scale", 1.0),
        )


class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, *, out_dtype: torch.dtype = torch.float16):
        self.weight = weight.float().contiguous()
        self.eps = float(eps)
        self.out_dtype = out_dtype
        import os as _os

        from src.kernels.cuda_loader import load_cuda_kernel

        # OFF by default: a real e2e showed the fused RMSNorm kernel is neutral
        # for GLM decode (0.66->0.64 tok/s, within noise) because GLM's per-token
        # time is ~1.5s, so the ~44ms of fp32 norm ops is only ~3% -- unlike
        # MiniMax (~100ms/token) where it was ~20% and fusing gave +79%.  Keeping
        # it off preserves bit-identical greedy output vs the DP4A baseline.
        # Opt in with GLM_FUSED_RMSNORM=1.
        self._use_fused = _os.getenv("GLM_FUSED_RMSNORM", "0") == "1"
        self._cuda = load_cuda_kernel() if self._use_fused else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._use_fused
            and self._cuda is not None
            and hasattr(self._cuda, "fused_rms_norm_forward")
            and self.out_dtype == torch.float16
            and x.is_cuda
            and x.dim() >= 2
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            x_flat = x.reshape(-1, x.size(-1)).contiguous()
            y_flat = self._cuda.fused_rms_norm_forward(x_flat, self.weight, self.eps)
            return y_flat.view(x.shape)
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv * self.weight).to(self.out_dtype)


class ReferenceLinear:
    def __init__(self, weight: torch.Tensor, *, out_dtype: torch.dtype = torch.float16, row_start: int = 0):
        # Weight is stored as [out_features, in_features] after GGUF dequant/read.
        if weight.dim() != 2:
            raise ValueError(f"ReferenceLinear expects a 2D weight, got {tuple(weight.shape)}")
        self.weight = weight.float().contiguous()
        self.out_dtype = out_dtype
        self.row_start = int(row_start)

    @property
    def in_dim(self) -> int:
        return int(self.weight.size(1))

    @property
    def out_dim(self) -> int:
        return int(self.weight.size(0))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_dim:
            raise ValueError(f"linear expected input dim {self.in_dim}, got {x.size(-1)}")
        y = F.linear(x.float(), self.weight, None)
        return y.to(self.out_dtype)


class ReferenceEmbedding:
    def __init__(self, weight: torch.Tensor, *, out_dtype: torch.dtype = torch.float16):
        # Weight is [vocab, dim].
        if weight.dim() != 2:
            raise ValueError(f"ReferenceEmbedding expects a 2D weight, got {tuple(weight.shape)}")
        self.weight = weight.float().contiguous()
        self.out_dtype = out_dtype

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.to(torch.long), self.weight).to(self.out_dtype)


class GLMDSAAttention:
    def __init__(
        self,
        args: GLMDSAArgs,
        layer_id: int,
        q_a_proj: ReferenceLinear,
        q_a_norm_weight: torch.Tensor,
        q_b_proj: ReferenceLinear,
        kv_a_proj: ReferenceLinear,
        kv_a_norm_weight: torch.Tensor,
        k_b_weight: torch.Tensor,
        v_b_weight: torch.Tensor,
        o_proj: ReferenceLinear,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.q_a_proj = q_a_proj
        self.q_a_norm = RMSNorm(q_a_norm_weight, args.norm_eps, out_dtype=dtype)
        self.q_b_proj = q_b_proj
        self.kv_a_proj = kv_a_proj
        self.kv_a_norm = RMSNorm(kv_a_norm_weight, args.norm_eps, out_dtype=dtype)
        self.k_b_weight = k_b_weight.float().contiguous()  # [k_nope, kv_lora, heads]
        self.v_b_weight = v_b_weight.float().contiguous()  # [value_dim, value_mla, heads]
        self.o_proj = o_proj
        self.device = device
        self.dtype = dtype
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None
        self.cache_batch = 0
        self.cache_len = 0

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.cache_batch = int(batch_size)
        self.cache_len = int(max_seq_len)
        self.cache_k = torch.empty(
            (self.cache_batch, self.cache_len, self.args.n_heads, self.args.key_mla_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.cache_v = torch.empty(
            (self.cache_batch, self.cache_len, self.args.n_heads, self.args.value_mla_dim),
            device=self.device,
            dtype=self.dtype,
        )

    def _ensure_cache(self, batch_size: int, needed_len: int) -> None:
        if self.cache_k is None or self.cache_v is None or self.cache_batch < batch_size or self.cache_len < needed_len:
            self.reset_cache(batch_size, max(int(needed_len), max(1, self.cache_len) * 2))

    def _apply_rope(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Apply RoPE using interleaved (NEOX/ROPE_TYPE_NEOX) style.

        GLM-5.2 (glm-dsa) uses interleaved rotary embeddings where adjacent
        pairs (x[0],x[1]), (x[2],x[3]), ... are rotated together. This is
        ROPE_TYPE_NEOX=2 in llama.cpp, NOT ROPE_TYPE_NORM (split-half).
        Verified against llama.cpp golden output: token 1 k_pe element 0
        transforms -2.0848 → -1.0771 via NEOX rotation, NOT split-half.
        """
        rope_dim = int(self.args.rope_dim)
        if rope_dim < 2:
            return x
        half = rope_dim // 2
        positions = torch.arange(start_pos, start_pos + x.size(1), device=x.device, dtype=torch.float32)
        j = torch.arange(half, device=x.device, dtype=torch.float32)
        inv = torch.pow(torch.full_like(j, float(self.args.rope_base)), -(2.0 * j) / float(rope_dim))
        freqs = positions[:, None] * inv[None, :]
        sin = torch.sin(freqs).to(x.dtype)[None, :, None, :]
        cos = torch.cos(freqs).to(x.dtype)[None, :, None, :]
        # Interleaved: pair adjacent elements (x[2i], x[2i+1])
        x1 = x[..., 0::2][..., :half]   # even indices
        x2 = x[..., 1::2][..., :half]   # odd indices
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        # Interleave y1/y2 back: (y1[0], y2[0], y1[1], y2[1], ...)
        y = torch.stack((y1, y2), dim=-1).flatten(-2)   # [..., rope_dim]
        if rope_dim < x.size(-1):
            return torch.cat((y, x[..., rope_dim:]), dim=-1)
        return y

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        end_pos = int(start_pos) + int(seqlen)
        self._ensure_cache(bsz, end_pos)
        assert self.cache_k is not None and self.cache_v is not None

        q_latent = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_latent).view(bsz, seqlen, self.args.n_heads, self.args.key_mla_dim)

        kv_rope = self.kv_a_proj(x).float()
        kv_latent = kv_rope[..., : self.args.kv_lora_rank]
        k_rope = kv_rope[..., self.args.kv_lora_rank : self.args.kv_lora_rank + self.args.rope_dim]
        kv_latent = self.kv_a_norm(kv_latent)

        # k_b: [k_nope, kv_lora, heads] -> [B,S,H,k_nope]
        k_nope = torch.einsum("bsk,dkh->bshd", kv_latent.float(), self.k_b_weight)
        # v_b: [value_dim, value_mla, heads].  GLM stores value_dim=512 but the
        # projection output used by attention is value_mla=256 per head.
        v = torch.einsum("bsk,kdh->bshd", kv_latent.float(), self.v_b_weight[: self.args.kv_lora_rank])
        if v.size(-1) != self.args.value_mla_dim:
            v = v[..., : self.args.value_mla_dim]
        k_rope = k_rope.view(bsz, seqlen, 1, self.args.rope_dim).expand(-1, -1, self.args.n_heads, -1)
        k_rope = self._apply_rope(k_rope.to(self.dtype), int(start_pos)).float()
        q_rope = q[..., -self.args.rope_dim :]
        q_nope = q[..., : self.args.key_mla_dim - self.args.rope_dim]
        q_rope = self._apply_rope(q_rope.to(self.dtype), int(start_pos)).float()
        q = torch.cat((q_nope.float(), q_rope), dim=-1).to(self.dtype)
        k = torch.cat((k_nope, k_rope), dim=-1).to(self.dtype)
        v = v.to(self.dtype)

        self.cache_k[:bsz, start_pos:end_pos].copy_(k)
        self.cache_v[:bsz, start_pos:end_pos].copy_(v)
        k_full = self.cache_k[:bsz, :end_pos]
        v_full = self.cache_v[:bsz, :end_pos]

        q_t = q.transpose(1, 2).contiguous()
        k_t = k_full.transpose(1, 2).contiguous()
        v_t = v_full.transpose(1, 2).contiguous()
        if seqlen == 1:
            attn_mask = None
            is_causal = False
        elif int(start_pos) == 0:
            attn_mask = None
            is_causal = True
        else:
            q_pos = torch.arange(start_pos, end_pos, device=x.device)
            k_pos = torch.arange(0, end_pos, device=x.device)
            allowed = k_pos[None, :] <= q_pos[:, None]
            attn_mask = torch.zeros((seqlen, end_pos), device=x.device, dtype=q_t.dtype)
            attn_mask.masked_fill_(~allowed, float("-inf"))
            is_causal = False
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(float(self.args.key_mla_dim)),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.args.n_heads * self.args.value_mla_dim)
        return self.o_proj(out)


class GLMDSADenseMLP:
    def __init__(self, gate_proj: ReferenceLinear, up_proj: ReferenceLinear, down_proj: ReferenceLinear, *, dtype: torch.dtype = torch.float16):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()
        return self.down_proj(hidden.to(self.dtype))


class GLMDSAMoE:
    """Reference GLM-DSA routed+shared MoE (fp32 fallback path).

    Selected experts are dequantized on CPU by the GGUF loader, moved to CUDA,
    and evaluated with PyTorch matmuls.  This is the fallback used when the routed
    dtypes or block layout are not eligible for the raw-block CUDA kernels; the
    optimized path is ``GLMDSARawBlockMoE``.
    """

    def __init__(
        self,
        args: GLMDSAArgs,
        layer_id: int,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        shared_gate_proj: ReferenceLinear | None,
        shared_up_proj: ReferenceLinear | None,
        shared_down_proj: ReferenceLinear | None,
        expert_loader,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        expert_start: int = 0,
        expert_count: int | None = None,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.gate_weight = gate_weight.float().contiguous()
        self.gate_bias = gate_bias.float().contiguous() if gate_bias is not None else None
        self.shared_gate_proj = shared_gate_proj
        self.shared_up_proj = shared_up_proj
        self.shared_down_proj = shared_down_proj
        self.expert_loader = expert_loader
        self.device = device
        self.dtype = dtype
        self.expert_start = int(expert_start)
        available = max(0, int(args.n_routed_experts) - self.expert_start)
        self.expert_count = available if expert_count is None else int(expert_count)
        if self.expert_start < 0 or self.expert_count < 0 or self.expert_start + self.expert_count > int(args.n_routed_experts):
            raise ValueError(
                f"invalid GLM-DSA local expert range [{self.expert_start}, {self.expert_start + self.expert_count}) "
                f"for expert_count={args.n_routed_experts}"
            )
        self._expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x_flat.float() @ self.gate_weight.t()
        probs = torch.sigmoid(logits)
        select_scores = probs if self.gate_bias is None else probs + self.gate_bias
        _, indices = torch.topk(select_scores, k=int(self.args.top_k), dim=-1)
        weights = torch.gather(probs, dim=-1, index=indices)
        if self.args.expert_weights_norm:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        weights = weights * float(self.args.expert_weights_scale)
        return indices.to(torch.long).contiguous(), weights.float().contiguous()

    def _expert_weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expert = int(expert)
        cached = self._expert_cache.get(expert)
        if cached is not None:
            return cached
        w1, w3, w2 = self.expert_loader(expert)
        cached = (
            w1.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous(),
            w3.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous(),
            w2.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous(),
        )
        self._expert_cache[expert] = cached
        return cached

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1]).contiguous()
        indices, weights = self.route(x_flat)
        out = torch.zeros((x_flat.size(0), self.args.dim), device=x.device, dtype=torch.float32)
        try:
            local_start = self.expert_start
            local_end = self.expert_start + self.expert_count
            for expert in torch.unique(indices).tolist():
                expert = int(expert)
                if expert < local_start or expert >= local_end:
                    continue
                mask = indices == expert
                route_pos = mask.nonzero(as_tuple=False)
                if route_pos.numel() == 0:
                    continue
                token_ids = route_pos[:, 0]
                route_ids = route_pos[:, 1]
                x_sel = x_flat.index_select(0, token_ids).float()
                w1, w3, w2 = self._expert_weights(expert)
                hidden = F.silu(F.linear(x_sel, w1)) * F.linear(x_sel, w3)
                y = F.linear(hidden, w2) * weights[token_ids, route_ids, None]
                out.index_add_(0, token_ids, y.float())
        finally:
            # Keep this reference path heterogeneous/active-expert: do not retain
            # dequantized routed experts across layers or decode calls.
            self._expert_cache.clear()

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)

        if self.shared_gate_proj is not None and self.shared_up_proj is not None and self.shared_down_proj is not None:
            shared = F.silu(self.shared_gate_proj(x_flat).float()) * self.shared_up_proj(x_flat).float()
            out = out + self.shared_down_proj(shared.to(self.dtype)).float()
        return out.reshape(original_shape).to(self.dtype)


class GLMDSARawBlockMoE:
    """GLM-DSA routed+shared MoE over raw GGUF quantized blocks.

    Routed experts (iq2_xs w1/w3, iq3_xxs w2) are staged per active expert and
    evaluated with the CUDA grouped MoE kernel; no fp32 weight expansion happens
    on the hot path.  Shared experts run through raw quantized GGUF linears
    (q5_k gate/up, q6_k down).  Router math matches ``GLMDSAMoE.route``.
    """

    def __init__(
        self,
        args: GLMDSAArgs,
        layer_id: int,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        *,
        gguf_path: str,
        w1_name: str,
        w3_name: str,
        w2_name: str,
        signed_grid: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        expert_start: int = 0,
        expert_count: int | None = None,
        swiglu_limit: float = -1.0,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.gate_weight = gate_weight.float().contiguous()
        self.gate_bias = gate_bias.float().contiguous() if gate_bias is not None else None
        self.shared_gate_proj = shared_gate_proj
        self.shared_up_proj = shared_up_proj
        self.shared_down_proj = shared_down_proj
        self.gguf_path = str(gguf_path)
        self.w1_name = str(w1_name)
        self.w3_name = str(w3_name)
        self.w2_name = str(w2_name)
        self.device = device
        self.dtype = dtype
        self.swiglu_limit = float(swiglu_limit)
        self.expert_start = int(expert_start)
        available = max(0, int(args.n_routed_experts) - self.expert_start)
        self.expert_count = available if expert_count is None else int(expert_count)
        if self.expert_start < 0 or self.expert_count < 0 or self.expert_start + self.expert_count > int(args.n_routed_experts):
            raise ValueError(
                f"invalid GLM-DSA local expert range [{self.expert_start}, {self.expert_start + self.expert_count}) "
                f"for expert_count={args.n_routed_experts}"
            )
        self._grid = signed_grid.to(device=device, dtype=torch.int8).contiguous()
        from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS

        self._type_ids = GGUF_DENSE_TYPE_IDS
        # Resident staging cache: on first forward we read all local experts'
        # raw blocks once into contiguous (optionally pinned) CPU tensors, so the
        # decode hot path only does an index + async H2D instead of re-reading
        # from mmap + torch.stack + a CPU sync every step/layer.  ~0.77 GB/layer.
        self._resident_w1: torch.Tensor | None = None
        self._resident_w3: torch.Tensor | None = None
        self._resident_w2: torch.Tensor | None = None
        self._resident_meta: tuple | None = None
        # Reusable pinned host staging buffers (fallback path). The active expert
        # SET changes every step so we cannot cache contents, but we CAN reuse a
        # small pinned buffer (<=top_k experts, ~tens of MB) so each step's H2D is
        # a fast pinned->GPU DMA instead of a pageable copy. Keyed by active count
        # + block shape; rebuilt only when those change (rare at decode).  This is
        # NOT the forbidden full-mmap pin: it is a bounded per-layer buffer.
        self._pin_stage_w1: torch.Tensor | None = None
        self._pin_stage_w3: torch.Tensor | None = None
        self._pin_stage_w2: torch.Tensor | None = None
        self._pin_stage_key: tuple | None = None
        import os as _os
        # Pinned staging is OFF by default: a profiled e2e showed the per-expert
        # copy into pinned host memory (stage_read) costs more than the faster
        # pinned->GPU DMA (stage_h2d) saves — a net staging regression.  The mmap
        # page-cache read into pinned is no cheaper than into pageable, so there
        # is no free lunch here.  Kept as an opt-in experiment switch only.
        self._use_pinned_stage = _os.getenv("GLM_PINNED_STAGE", "0") == "1"

        # Resident-cache staging is OFF by default: an e2e measurement showed it
        # does NOT help decode once the page cache is warm (the original per-step
        # mmap read already hits RAM, so torch.stack and index_select cost the
        # same ~1.8GB host copy), while the one-time 58GB/rank build regresses
        # prefill.  Kept as an opt-in experiment switch only.
        self._disable_resident = _os.getenv("GLM_ENABLE_RESIDENT_EXPERTS", "0") != "1"
        # Pinning ~58 GB/rank is non-pageable host memory; opt-in only to avoid
        # over-committing pinned memory across TP ranks.
        self._pin_resident = _os.getenv("GLM_PIN_RESIDENT_EXPERTS", "0") == "1"
        # GLM routed experts are iq2_xs (w1/w3) + iq3_xxs (w2); enable their
        # DP4A grouped-MoE kernels by default (numerically parity-checked vs the
        # fp32 general branch).  The CUDA kernel reads these env gates, so set
        # them here unless the user explicitly opted out with "0".
        for _flag in ("DEEPSEEK_GGUF_IQ2_XS_W13_DP4A", "DEEPSEEK_GGUF_IQ3_XXS_W2_DP4A"):
            if _os.getenv(_flag) is None:
                _os.environ[_flag] = "1"

    def route(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x_flat.float() @ self.gate_weight.t()
        probs = torch.sigmoid(logits)
        select_scores = probs if self.gate_bias is None else probs + self.gate_bias
        _, indices = torch.topk(select_scores, k=int(self.args.top_k), dim=-1)
        weights = torch.gather(probs, dim=-1, index=indices)
        if self.args.expert_weights_norm:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        weights = weights * float(self.args.expert_weights_scale)
        return indices.to(torch.long).contiguous(), weights.float().contiguous()

    def _build_resident_experts(self, reader) -> None:
        """Read this rank's ALL local experts' raw blocks once into contiguous
        CPU tensors of shape ``[expert_count, N, K_blocks, block_bytes]`` for
        w1/w3/w2.  Done lazily on first forward so the decode hot path can index
        + async-H2D instead of re-reading mmap + torch.stack + a CPU sync every
        step.  ~0.77 GB/layer resident on host (system has ample RAM)."""
        w1_list, w3_list, w2_list = [], [], []
        w1_tn = w3_tn = w2_tn = None
        w1_in = w3_in = w2_in = None
        for local_id in range(self.expert_count):
            expert_id = local_id + self.expert_start
            b1, w1_tn, w1_in = reader.read_routed_expert_blocks(self.w1_name, expert_id)
            b3, w3_tn, w3_in = reader.read_routed_expert_blocks(self.w3_name, expert_id)
            b2, w2_tn, w2_in = reader.read_routed_expert_blocks(self.w2_name, expert_id)
            # Clone: read_routed_expert_blocks returns a view over the shared mmap;
            # we must own the memory to keep it resident and contiguous.
            w1_list.append(b1.clone())
            w3_list.append(b3.clone())
            w2_list.append(b2.clone())
        w1 = torch.stack(w1_list, dim=0).contiguous()
        w3 = torch.stack(w3_list, dim=0).contiguous()
        w2 = torch.stack(w2_list, dim=0).contiguous()
        if self._pin_resident:
            w1 = w1.pin_memory()
            w3 = w3.pin_memory()
            w2 = w2.pin_memory()
        self._resident_w1 = w1
        self._resident_w3 = w3
        self._resident_w2 = w2
        self._resident_meta = (
            (self._type_ids[w1_tn], int(w1_in)),
            (self._type_ids[w3_tn], int(w3_in)),
            (self._type_ids[w2_tn], int(w2_in)),
        )

    def _stage_active_experts(self, reader, local_ids_cpu: list[int]):
        """Return stacked ``[E_active, N, K_blocks, block_bytes]`` tensors for
        w1/w3/w2 plus their (type_id, in_dim) metadata, staged on device.

        Fast path: index the resident CPU cache by the active local ids and do a
        single async H2D per weight.  Fallback (resident disabled): re-read from
        mmap + torch.stack every call (original behavior)."""
        if not self._disable_resident:
            if self._resident_w1 is None:
                self._build_resident_experts(reader)
            idx = torch.as_tensor(local_ids_cpu, dtype=torch.long)
            w1_cpu = self._resident_w1.index_select(0, idx)
            w3_cpu = self._resident_w3.index_select(0, idx)
            w2_cpu = self._resident_w2.index_select(0, idx)
            w1_blocks = w1_cpu.to(self.device, non_blocking=True).contiguous()
            w3_blocks = w3_cpu.to(self.device, non_blocking=True).contiguous()
            w2_blocks = w2_cpu.to(self.device, non_blocking=True).contiguous()
            return w1_blocks, w3_blocks, w2_blocks, self._resident_meta

        with _prof_section("stage_read"):
            w1_vals, w3_vals, w2_vals = [], [], []
            for local_id in local_ids_cpu:
                expert_id = int(local_id) + self.expert_start
                w1_b, w1_tn, w1_in = reader.read_routed_expert_blocks(self.w1_name, expert_id)
                w3_b, w3_tn, w3_in = reader.read_routed_expert_blocks(self.w3_name, expert_id)
                w2_b, w2_tn, w2_in = reader.read_routed_expert_blocks(self.w2_name, expert_id)
                w1_vals.append((w1_b, w1_tn, w1_in))
                w3_vals.append((w3_b, w3_tn, w3_in))
                w2_vals.append((w2_b, w2_tn, w2_in))
            if getattr(self, "_use_pinned_stage", False):
                w1_host = self._fill_pinned_stage("w1", [v[0] for v in w1_vals])
                w3_host = self._fill_pinned_stage("w3", [v[0] for v in w3_vals])
                w2_host = self._fill_pinned_stage("w2", [v[0] for v in w2_vals])
            else:
                w1_host = torch.stack([v[0] for v in w1_vals], dim=0)
                w3_host = torch.stack([v[0] for v in w3_vals], dim=0)
                w2_host = torch.stack([v[0] for v in w2_vals], dim=0)
        with _prof_section("stage_h2d"):
            # Three independent small H2D copies. A merged single-H2D variant was
            # tried (GLM_MERGED_H2D) and REGRESSED: the host torch.cat memcpy of
            # ~1.8MB costs more than the 2 launches it saves (stage_h2d 4->12ms),
            # so decode staging is not launch-overhead bound. Kept as 3 copies.
            w1_blocks = w1_host.to(self.device, non_blocking=True).contiguous()
            w3_blocks = w3_host.to(self.device, non_blocking=True).contiguous()
            w2_blocks = w2_host.to(self.device, non_blocking=True).contiguous()
        meta = (
            (self._type_ids[w1_vals[0][1]], int(w1_vals[0][2])),
            (self._type_ids[w3_vals[0][1]], int(w3_vals[0][2])),
            (self._type_ids[w2_vals[0][1]], int(w2_vals[0][2])),
        )
        return w1_blocks, w3_blocks, w2_blocks, meta

    def _fill_pinned_stage(self, which: str, blocks: list[torch.Tensor]) -> torch.Tensor:
        """Copy per-expert raw-block views into a reusable pinned host tensor and
        return it.  The pinned buffer makes the subsequent H2D a fast DMA; it is
        reused across steps whenever the active-expert count and block shape match
        (the common decode case), so we avoid re-pinning every step."""
        n = len(blocks)
        b0 = blocks[0]
        key = (which, n, tuple(b0.shape), b0.dtype)
        attr = f"_pin_stage_{which}"
        buf = getattr(self, attr)
        # w1/w3/w2 shapes are stable per layer; track the active key per buffer.
        keys = getattr(self, "_pin_stage_keys", None)
        if keys is None:
            keys = {}
            self._pin_stage_keys = keys
        if buf is None or keys.get(which) != key:
            buf = torch.empty((n, *b0.shape), dtype=b0.dtype).pin_memory()
            setattr(self, attr, buf)
            keys[which] = key
        for i, b in enumerate(blocks):
            buf[i].copy_(b)
        return buf

    def _routed_forward(self, x_flat: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        from src.kernels.cuda_loader import load_cuda_kernel
        from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

        cuda_mod = load_cuda_kernel()
        if cuda_mod is None or not hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward"):
            raise RuntimeError("CUDA grouped MoE kernel is required for GLMDSARawBlockMoE")

        out = torch.zeros((x_flat.size(0), self.args.dim), device=x_flat.device, dtype=torch.float32)
        with _prof_section("moe_route_group"):
            grouped = cuda_mod.moe_group_routes(
                indices.contiguous(),
                weights.contiguous().to(torch.float32),
                int(self.expert_start),
                int(self.expert_count),
            )
        if grouped is None:
            return out
        local_ids, route_tokens, route_weights, seg_starts = grouped
        if local_ids.numel() == 0:
            _glm_profile_record_active(0)
            return out
        with _prof_section("moe_ids_sync"):
            local_ids_cpu = local_ids.detach().cpu().to(torch.long).tolist()
        _glm_profile_record_active(len(local_ids_cpu))

        with _prof_section("moe_stage_experts"):
            reader = get_cached_gguf_tensor_reader(self.gguf_path)
            w1_blocks, w3_blocks, w2_blocks, meta = self._stage_active_experts(reader, local_ids_cpu)
        (w1_type_id, w1_in_dim), (w3_type_id, w3_in_dim), (w2_type_id, w2_in_dim) = meta

        seg_i32 = seg_starts if seg_starts.dtype == torch.int32 else seg_starts.to(torch.int32)
        compact_starts = torch.cat(
            [seg_i32[local_ids.to(seg_i32.device)], seg_i32[-1:]], dim=0
        ).contiguous()

        with _prof_section("moe_kernel"):
          y = cuda_mod.gguf_moe_prefill_grouped_forward(
            x_flat.to(torch.float16).contiguous(),
            route_tokens,
            route_weights.contiguous(),
            compact_starts,
            w1_blocks,
            w3_blocks,
            w2_blocks,
            w1_in_dim,
            w1_type_id,
            w3_in_dim,
            w3_type_id,
            w2_in_dim,
            w2_type_id,
            self._grid,
            float(self.swiglu_limit),
        )
        # The grouped kernel already scatter-adds each route into y[token] and
        # multiplies route weights internally, so y is the [T, D] routed output.
        return y.float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1]).contiguous()
        with _prof_section("moe_route"):
            indices, weights = self.route(x_flat)
        out = self._routed_forward(x_flat, indices, weights)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            with _prof_section("moe_all_reduce"):
                dist.all_reduce(out)

        if self.shared_gate_proj is not None and self.shared_up_proj is not None and self.shared_down_proj is not None:
            with _prof_section("moe_shared"):
                shared = F.silu(self.shared_gate_proj(x_flat).float()) * self.shared_up_proj(x_flat).float()
                out = out + self.shared_down_proj(shared.to(self.dtype)).float()
        return out.reshape(original_shape).to(self.dtype)


class GLMDSAMoEPlaceholder:
    def __init__(self, layer_id: int):
        self.layer_id = int(layer_id)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"GLM-DSA MoE layer {self.layer_id} was reached with MoE disabled; "
            "MoE is implemented but this run opted out (allow_moe_layers=False). "
            "Request n_layers beyond leading_dense_block_count to enable MoE layers."
        )


class GLMDSABlock:
    def __init__(
        self,
        args: GLMDSAArgs,
        layer_id: int,
        attn_norm_weight: torch.Tensor,
        ffn_norm_weight: torch.Tensor,
        attention: GLMDSAAttention,
        mlp,
        *,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.layer_id = int(layer_id)
        self.attn_norm = RMSNorm(attn_norm_weight, args.norm_eps, out_dtype=dtype)
        self.ffn_norm = RMSNorm(ffn_norm_weight, args.norm_eps, out_dtype=dtype)
        self.attention = attention
        self.mlp = mlp
        self.dtype = dtype

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        self.attention.reset_cache(batch_size, max_seq_len)

    def __call__(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        with _prof_section("attention"):
            x_attn = (x + self.attention(self.attn_norm(x), start_pos)).to(self.dtype)
        with _prof_section("mlp"):
            mlp_in = self.ffn_norm(x_attn)
            mlp_out = self.mlp(mlp_in)
            return (x_attn + mlp_out).to(self.dtype)


class GLMDSATransformer:
    def __init__(
        self,
        args: GLMDSAArgs,
        embedding: ReferenceEmbedding,
        layers: list[GLMDSABlock],
        final_norm_weight: torch.Tensor,
        lm_head: ReferenceLinear,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.args = args
        self.embedding = embedding
        self.layers = layers
        self.final_norm = RMSNorm(final_norm_weight, args.norm_eps, out_dtype=dtype)
        self.lm_head = lm_head
        self.device = device
        self.dtype = dtype
        self.max_seq_len = args.context_length

    def reset_cache(self, batch_size: int, max_seq_len: int) -> None:
        for layer in self.layers:
            layer.reset_cache(batch_size, max_seq_len)

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        *,
        return_next_token: bool = False,
        return_hidden: bool = False,
        keep_all_positions: bool = False,
    ):
        if tokens.device != self.device:
            tokens = tokens.to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        import os as _os

        _dump = bool(_os.environ.get("GLM_DUMP_LAYERS"))

        def _d(name: str, t: torch.Tensor) -> None:
            # Per-layer numeric dump for cross-checking against a reference
            # implementation (e.g. llama.cpp eval-callback). Prints sum + first
            # 3 values of the last position, matching the reference format.
            if not _dump:
                return
            tf = t.detach().float()
            last = tf[0, -1] if tf.dim() == 3 else tf.reshape(-1)
            head = ", ".join(f"{v:12.4f}" for v in last[:3].tolist())
            print(f"GLM_DUMP {name:16s} sum={tf.sum().item():16.6f}  first3=[{head}]", flush=True)

        _glm_profile_set_phase(int(tokens.size(-1)))
        h = self.embedding(tokens).to(self.dtype)
        _d("embd", h)
        for _li, layer in enumerate(self.layers):
            h = layer(h, int(start_pos))
            _d(f"l_out-{_li}", h)
        h = self.final_norm(h)
        _d("result_norm", h)
        logits_input = h if keep_all_positions else h[:, -1:, :]
        logits = self.lm_head(logits_input).float()
        if return_next_token:
            next_token = distributed_argmax_local_logits(logits, row_start=int(self.lm_head.row_start))
            if not keep_all_positions:
                next_token = next_token[:, -1]
            if return_hidden:
                return next_token, (h if keep_all_positions else h[:, -1:, :])
            return next_token
        logits = gather_sharded_logits(logits, full_out_dim=int(self.args.vocab_size), row_start=int(self.lm_head.row_start))
        if return_hidden:
            return logits, h
        return logits

    __call__ = forward
