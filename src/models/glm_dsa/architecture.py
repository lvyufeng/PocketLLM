from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.components.gguf.tp_logits import distributed_argmax_local_logits, gather_sharded_logits
from src.components.moe.spec import metadata_float, metadata_int
from src.loader.gguf.bundle import GGUFBundle
from src.models.glm_dsa.spec import GLMDSASpec


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
        full_layers = int(params.n_layers)
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
        x1 = x[..., :half]
        x2 = x[..., half:rope_dim]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        if rope_dim < x.size(-1):
            return torch.cat((y1, y2, x[..., rope_dim:]), dim=-1)
        return torch.cat((y1, y2), dim=-1)

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


class GLMDSAMoEPlaceholder:
    def __init__(self, layer_id: int):
        self.layer_id = int(layer_id)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"GLM-DSA MoE runtime is not implemented for layer {self.layer_id}; "
            "use --n-layers within leading_dense_block_count for the first dense-prefix smoke"
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
        x = (x + self.attention(self.attn_norm(x), start_pos)).to(self.dtype)
        x = (x + self.mlp(self.ffn_norm(x))).to(self.dtype)
        return x


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
        h = self.embedding(tokens).to(self.dtype)
        for layer in self.layers:
            h = layer(h, int(start_pos))
        h = self.final_norm(h)
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
