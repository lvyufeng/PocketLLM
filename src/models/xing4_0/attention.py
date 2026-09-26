"""Xing4.0-29B-A4B's MLA attention, in both of the forms its two releases imply.

The checkpoint ships the same attention twice.  Its own remote code
(`modeling_xing4_0.py`) is the **expanded** form: `kv_b_proj` up-projects the
512-wide latent to `num_heads * (qk_nope + v_head_dim)` and the cache holds
32 heads x (192 + 128) per token.  Its released GGUF is the **absorbed** form: it
carries `attn_k_b` and `attn_v_b` as separate tensors and its
`attention.head_count_kv` is 1, so the cache holds one 512-wide latent and its
64-wide shared rotary key.  This module writes both, because the parity test is
only meaningful against the first and the engine wants the second.

What was reused, and what could not be:

- **Reused**: `glm_dsa`'s MLA skeleton.  The GGUF tensor names are identical
  (`attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_output`), and so are the
  orientations -- `k_b` is `[qk_nope, kv_lora, heads]` and `v_b` is
  `[kv_lora, v_head, heads]`, which is what `src/models/glm_dsa/architecture.py`
  already sums against.  The einsums below are that file's, with this
  checkpoint's numbers.
- **Not reusable**: the rotary embedding.  `glm_dsa` reads interleaved pairs and
  writes them back interleaved; this checkpoint reads interleaved pairs and
  writes the two halves contiguously, and it is YaRN rather than plain RoPE.
  See `src/models/xing4_0/rope.py`.
- **Not reusable**: the attention scale.  `1/sqrt(192)`, then multiplied by the
  YaRN magnitude scale *squared* -- `1.4159 ** 2` for this config.  `glm_dsa`
  uses `1/sqrt(key_mla_dim)` and nothing else.
- **Not reusable**: the decode path.  `glm_dsa` expands per token; the absorbed
  form below is what the released file is shaped for.

## The absorbed form, and what absorbing costs

Absorption folds the up-projections into the attention rather than the cache:

    score[h] = (W_kb[h]^T q_nope[h]) . c  +  q_rope . k_rope
    out[h]   = (softmax(score)[h] . c) @ W_vb[h]

which is the same arithmetic as the expanded form, re-associated.  It is what
makes the cache 576 elements per token per layer instead of 32 x 320, i.e. 17.8x
smaller, and the price is that the query has to be absorbed once per token, which
widens it from 32 x 192 = 6144 to 32 x 512 = 16384.  At decode that is one
small GEMM against the cache being read at a seventeenth of the width, so it
pays; at prefill it does not, which is why both are here.

One detail is worth stating because it is where an absorbed port diverges: the
rope key is *shared across heads* in both forms.  `kv_a_proj` produces one
64-wide `k_rope`, RoPE rotates it once, and the expanded form then broadcasts it
with `.expand(...)`.  A port that ropes it per head still gets the right answer
if the rotation is identical -- but a port that ropes the *latent* instead of the
rope key, or that folds the rope dimension into the 512-wide latent's dot
product, does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.rope import cos_sin, inv_freq, rotate_interleaved

__all__ = ["MLAAttention", "MLAAttentionWeights", "KVLatentCache"]


@dataclass
class MLAAttentionWeights:
    """The attention's weights, in torch's own orientation.

    Every 2-D weight is `(out, in)`, which is what `F.linear` wants and what a
    GGUF tensor already is once its dimension list is reversed -- GGML's `ne[0]`
    is the *fastest-varying* axis, so a tensor the header lists as `[3584, 768]`
    is 768 rows of 3584 in memory.  The three-dimensional ones keep the head axis
    first, for the same reason: the header's `[128, 512, 32]` is `(32, 512, 128)`
    in memory.

    `k_b` and `v_b` are the absorbed halves of `kv_b_proj`: the GGUF ships them
    split so that an absorbed runtime never has to build the (512, 8192)
    concatenation the reference's `kv_b_proj` is.  Their `(out, in)` orientation
    is the absorbed one -- `k_b[h]` is `(kv_lora, qk_nope)` and applies to a
    query, `v_b[h]` is `(v_head, kv_lora)` and applies to the latent -- and the
    non-absorbed form reads the same tensors transposed, which is a contraction
    order and not a different weight.
    """

    q_a_proj: torch.Tensor  # [q_lora, hidden]
    q_a_norm: torch.Tensor  # [q_lora]
    q_b_proj: torch.Tensor  # [n_heads * qk_head_dim, q_lora]
    kv_a_proj: torch.Tensor  # [kv_lora + qk_rope, hidden]
    kv_a_norm: torch.Tensor  # [kv_lora]
    k_b: torch.Tensor  # [n_heads, kv_lora, qk_nope]
    v_b: torch.Tensor  # [n_heads, v_head_dim, kv_lora]
    o_proj: torch.Tensor  # [hidden, n_heads * v_head_dim]

    @classmethod
    def from_hf(cls, tensors: dict[str, torch.Tensor], params: Xing4_0Params) -> "MLAAttentionWeights":
        """Build from the released BF16 checkpoint's own tensor names.

        Two names differ from what a reader might expect from the GGUF:

        - `kv_a_proj_with_mqa` is one `(kv_lora + qk_rope, hidden)` weight.  The
          "mqa" half is the single 64-wide rotary key the release produces once
          and broadcasts across all 32 heads, and it is the *last* 64 rows.
        - `kv_b_proj` is one `(heads * (qk_nope + v_head), kv_lora)` weight whose
          row order is **head-major**: head 0's 128 `nope` rows, then head 0's 128
          `v` rows, then head 1's.  The released GGUF ships that same weight split
          into `attn_k_b` and `attn_v_b`, so splitting it here is what makes the
          two sources produce the same `k_b`/`v_b`; the split is per head and not
          a cut in half, and `concat(split)` is the original weight exactly.
        """
        heads = params.n_heads
        nope, v_head = params.qk_nope_head_dim, params.v_head_dim
        kv_b = tensors["self_attn.kv_b_proj.weight"]
        if tuple(kv_b.shape) != (heads * (nope + v_head), params.kv_lora_rank):
            raise ValueError(f"kv_b_proj is {tuple(kv_b.shape)}, expected {(heads * (nope + v_head), params.kv_lora_rank)}")
        per_head = kv_b.reshape(heads, nope + v_head, params.kv_lora_rank)
        return cls(
            q_a_proj=tensors["self_attn.q_a_proj.weight"],
            q_a_norm=tensors["self_attn.q_a_layernorm.weight"],
            q_b_proj=tensors["self_attn.q_b_proj.weight"],
            kv_a_proj=tensors["self_attn.kv_a_proj_with_mqa.weight"],
            kv_a_norm=tensors["self_attn.kv_a_layernorm.weight"],
            # `k_b[h][j, i]` multiplies latent `j` into key channel `i`, so the
            # (nope, kv_lora) block is this head's first `nope` rows transposed.
            k_b=per_head[:, :nope, :].transpose(1, 2).contiguous(),
            # `v_b[h]` already applies to the latent in the (out, in) orientation.
            v_b=per_head[:, nope:, :].contiguous(),
            o_proj=tensors["self_attn.o_proj.weight"],
        )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, as the reference computes it."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


class KVLatentCache:
    """The absorbed cache: one 512-wide latent and its 64-wide rope key per token.

    Not a `(k, v)` pair.  There is no separate value tensor to store, because the
    value is `latent @ v_b` and is only ever needed after the attention weights.
    """

    def __init__(self, batch: int, capacity: int, params: Xing4_0Params, *, device="cpu", dtype=torch.float32):
        self.params = params
        self.batch = int(batch)
        self.capacity = int(capacity)
        self.width = int(params.kv_lora_rank) + int(params.qk_rope_head_dim)
        self.latent = torch.zeros((self.batch, self.capacity, self.width), device=device, dtype=dtype)
        self.length = 0

    def append(self, rows: torch.Tensor, start_pos: int) -> None:
        """Write one chunk at `start_pos`; rows are `(batch, seq, width)`."""
        end = int(start_pos) + rows.shape[1]
        if end > self.capacity:
            raise ValueError(f"cache holds {self.capacity} tokens, asked for {end}")
        self.latent[:, start_pos:end] = rows
        self.length = max(self.length, end)

    def view(self, length: int | None = None) -> torch.Tensor:
        return self.latent[:, : length if length is not None else self.length]


class MLAAttention:
    """One attention layer.  No weights are loaded here."""

    def __init__(
        self,
        params: Xing4_0Params,
        weights: MLAAttentionWeights,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.params = params
        self.weights = weights
        self.dtype = dtype
        self.device = device
        self.scale = float(params.attention_scale)
        self._frequencies = inv_freq(params, device=device)

    # -- shared prefix ------------------------------------------------------- #

    def _q_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        return _rms_norm(
            F.linear(hidden, self.weights.q_a_proj),
            self.weights.q_a_norm,
            self.params.rms_norm_eps,
        )

    def _compressed_kv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(latent, rope_key)` -- the 512 and the 64, before either is used."""
        p = self.params
        projected = F.linear(hidden, self.weights.kv_a_proj)
        latent = projected[..., : p.kv_lora_rank]
        rope_key = projected[..., p.kv_lora_rank :]
        latent = _rms_norm(latent, self.weights.kv_a_norm, p.rms_norm_eps)
        return latent, rope_key

    def rope_queries(self, q_rot: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate a `(batch, seq, heads, qk_rope)` tensor.

        `cos`/`sin` are built at `(seq, head_dim)` and inserted at the head axis,
        which is where the reference's `unsqueeze_dim=1` puts them for a
        `(batch, heads, seq, dim)` layout.
        """
        cos, sin = cos_sin(self.params, positions, frequencies=self._frequencies, device=self.device, dtype=self.dtype)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return rotate_interleaved(q_rot, cos, sin).to(self.dtype)

    # -- the reference's form: expanded -------------------------------------- #

    def forward_expanded(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        cache: KVLatentCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """The checkpoint's own remote-code arithmetic, expanded per head.

        `hidden` is `(1, seq, hidden)`.  Returns `(1, seq, hidden)`.
        """
        p = self.params
        b, seq, _ = hidden.shape
        nope, rope, heads = p.qk_nope_head_dim, p.qk_rope_head_dim, p.n_heads
        v_head = p.v_head_dim

        q = F.linear(self._q_latent(hidden), self.weights.q_b_proj).view(b, seq, heads, nope + rope)
        q_nope, q_rot = q[..., :nope], q[..., nope:]
        q_rot = self.rope_queries(q_rot, positions)

        latent, rope_key = self._compressed_kv(hidden)
        if cache is not None:
            rope_key = self.rope_queries(rope_key.unsqueeze(2), positions).squeeze(2)
            cache.append(torch.cat((latent, rope_key), dim=-1), start_pos)
            latent_full = cache.view()[:, :, : p.kv_lora_rank]
            rope_full = cache.view()[:, :, p.kv_lora_rank :]
            # The reference ropes the key once and broadcasts it across heads.
            k_rot = rope_full.unsqueeze(2).expand(-1, -1, heads, -1)
            tokens, offset = latent_full.shape[1], 0
        else:
            rope_key = self.rope_queries(rope_key.unsqueeze(2), positions).squeeze(2)
            latent_full, k_rot = latent, rope_key.unsqueeze(2).expand(-1, -1, heads, -1)
            tokens, offset = seq, 0

        # k_nope: `k_b[h]` is (kv_lora, qk_nope), so it contracts the latent's
        # 512 down to this head's 128 without a transpose.
        k_nope = torch.einsum("bsk,hkd->bshd", latent_full, self.weights.k_b)
        v = torch.einsum("bsk,hdk->bshd", latent_full, self.weights.v_b)

        query = torch.cat((q_nope, q_rot), dim=-1).transpose(1, 2)
        key = torch.cat((k_nope, k_rot), dim=-1).transpose(1, 2)
        value = v.transpose(1, 2)

        out = _attend(query, key, value, self.scale, seq, tokens, start_pos, dtype=self.dtype)
        return F.linear(out, self.weights.o_proj)

    # -- the released file's form: absorbed ---------------------------------- #

    def forward_absorbed(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        cache: KVLatentCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """The same arithmetic, re-associated so the cache stays a latent.

        `hidden` is `(1, seq, hidden)`.  Returns `(1, seq, hidden)`.
        """
        p = self.params
        b, seq, _ = hidden.shape
        nope, rope, heads = p.qk_nope_head_dim, p.qk_rope_head_dim, p.n_heads

        q = F.linear(self._q_latent(hidden), self.weights.q_b_proj).view(b, seq, heads, nope + rope)
        q_nope, q_rot = q[..., :nope], q[..., nope:]
        q_rot = self.rope_queries(q_rot, positions)

        latent, rope_key = self._compressed_kv(hidden)
        rope_key = self.rope_queries(rope_key.unsqueeze(2), positions).squeeze(2)
        if cache is not None:
            cache.append(torch.cat((latent, rope_key), dim=-1), start_pos)
            stored = cache.view()
            latent_full = stored[:, :, : p.kv_lora_rank]
            rope_full = stored[:, :, p.kv_lora_rank :]
        else:
            latent_full, rope_full = latent, rope_key

        # W_kb^T applied to the query instead of W_kb applied to the key: the same
        # tensor as `forward_expanded` reads, contracted the other way.
        q_absorbed = torch.einsum("bshd,hkd->bhsk", q_nope, self.weights.k_b)

        # The two score terms are written as one GEMM each rather than as a
        # broadcast batch.  `(b, h, s, k) @ (b, k, tokens)` broadcasts the *cache*
        # across heads, which makes every head re-read the whole latent -- 32x the
        # traffic, and at 32K that is 1.07 GiB per layer instead of 33 MiB.  Folded
        # into the head axis there is one GEMM whose B operand is read once.  Each
        # output element is the same per-head sum over the same 512 terms; only the
        # accumulation order inside the GEMM is the kernel's business.
        batch, heads_axis = q_absorbed.shape[0], heads * seq
        latent_t = latent_full.transpose(1, 2)  # (b, kv_lora, tokens)
        scores = (q_absorbed.reshape(batch, heads_axis, p.kv_lora_rank) @ latent_t).reshape(
            batch, heads, seq, latent_full.shape[1]
        )
        # The rope key is shared across heads, so one dot product serves all of
        # them -- and the same fold applies to that one.
        rope_t = rope_full.transpose(1, 2)  # (b, qk_rope, tokens)
        scores = scores + (
            q_rot.transpose(1, 2).reshape(batch, heads_axis, rope) @ rope_t
        ).reshape(batch, heads, seq, latent_full.shape[1])
        scores = _mask(scores * self.scale, seq, latent_full.shape[1], start_pos)
        # As `eager_attention_forward` does: an fp32 softmax, cast back before the
        # value contraction.  The contractions stay in the model dtype.
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)

        # The value side has the same shape problem in reverse: `(b, h, s, tokens)
        # @ (b, tokens, kv_lora)` broadcasts the latent across heads again.  Folded
        # the same way, the latent is read once for all 32 heads.
        attended = (probs.reshape(batch, heads_axis, latent_full.shape[1]) @ latent_full).reshape(
            batch, heads, seq, p.kv_lora_rank
        )
        out = torch.einsum("bhsk,hdk->bhsd", attended, self.weights.v_b)
        out = out.transpose(1, 2).reshape(b, seq, heads * p.v_head_dim)
        return F.linear(out.to(self.dtype), self.weights.o_proj)


def _mask(scores: torch.Tensor, seq: int, tokens: int, start_pos: int) -> torch.Tensor:
    """Causal mask, expressed against absolute positions so a resume is one rule."""
    if seq == 1:
        return scores
    q_pos = torch.arange(start_pos, start_pos + seq, device=scores.device).view(1, 1, seq, 1)
    k_pos = torch.arange(0, tokens, device=scores.device).view(1, 1, 1, tokens)
    return scores.masked_fill(k_pos > q_pos, float("-inf"))


def _attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    seq: int,
    tokens: int,
    offset: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """`scaled_dot_product_attention` over an explicitly built causal mask.

    The reference uses `eager_attention_forward` with the mask the caller passed
    and `repeat_kv`, which at `num_key_value_groups == 1` is the identity -- this
    checkpoint has `num_key_value_heads == num_attention_heads`, so there is
    nothing to repeat and the expanded form's key already carries `heads`.
    """
    mask = None
    is_causal = False
    if seq == 1:
        is_causal = False
    elif offset == 0 and tokens == seq:
        is_causal = True
    else:
        q_pos = torch.arange(offset, offset + seq, device=query.device).view(seq, 1)
        k_pos = torch.arange(0, tokens, device=query.device).view(1, tokens)
        mask = torch.zeros((seq, tokens), device=query.device, dtype=query.dtype)
        mask.masked_fill_(k_pos > q_pos, float("-inf"))
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=is_causal, scale=scale
    )
    return out.transpose(1, 2).reshape(query.shape[0], seq, -1).to(dtype)
