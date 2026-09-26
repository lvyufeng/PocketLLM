"""Xing4.0-29B-A4B's decoder layer: the hyper-connection around the sublayers.

The audit read this out of `Xing4_0DecoderLayer.forward`, and the shape of it is
what makes the hyper-connection more than a wrapper:

    post, comb, collapsed = attn_hc(hidden)          # hidden is [tokens, 4, hidden]
    collapsed = input_layernorm(collapsed)           # the pre-attention norm is inside
    attn_out  = self_attn(collapsed)
    hidden    = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + comb @ hidden

    post, comb, collapsed = ffn_hc(hidden)
    collapsed = post_attention_layernorm(collapsed)
    mlp_out   = self.mlp(collapsed)
    hidden    = post.unsqueeze(-1) * mlp_out.unsqueeze(-2) + comb @ hidden

Two things follow, and both are places a port goes wrong quietly:

- **The sublayer sees one stream, the plumbing sees four.** Attention and the FFN
  take `[tokens, hidden]`; only `hc_*` and the residual are `[tokens, 4, 4]` and
  `[tokens, 4, hidden]`. On this card that is the difference between a decode step
  that reads 17.8 GiB and one that reads 71 GiB.
- **`post` and `comb` are recomputed from the current state each sublayer**, so
  there is no residual accumulator to carry. The block's output is a rebuild, not
  an addition.

`hidden_size` here is one stream's width: the layer's state is `hc_mult` of them,
which the model's `hidden_states.mean(dim=2)` collapses to one before the final
norm -- **mean**, not sum, and with no learned head-side gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from src.models.xing4_0.attention import MLAAttention, MLAAttentionWeights
from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.decode_pos import Pos
from src.models.xing4_0.hyper_connection import HyperConnection, HyperConnectionWeights

__all__ = ["DecoderLayer", "DecoderLayerWeights", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`Xing4_0RMSNorm`: fp32 statistics, the learned scale applied, cast back.

    The weighted counterpart of the hyper-connection's norm.  Both exist and they
    are not interchangeable.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


@dataclass
class DecoderLayerWeights:
    attn_hc: HyperConnectionWeights
    ffn_hc: HyperConnectionWeights
    attention: MLAAttentionWeights
    input_layernorm: torch.Tensor  # [hidden]
    post_attention_layernorm: torch.Tensor  # [hidden]


class DecoderLayer:
    """One trunk block.  The sublayer is passed in, so the dense FFN and the MoE
    are the same plumbing with a different callable in it.
    """

    def __init__(
        self,
        params: Xing4_0Params,
        weights: DecoderLayerWeights,
        mlp: Callable[[torch.Tensor], torch.Tensor],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        use_kernel: bool = False,
        residual_dtype: torch.dtype | None = None,
    ):
        self.params = params
        self.weights = weights
        self.mlp = mlp
        self.dtype = dtype
        # The four residual streams are carried at their own width, which is
        # `dtype` unless a caller says otherwise.  See `forward` for why this
        # checkpoint needs the wider one.
        self.residual_dtype = dtype if residual_dtype is None else residual_dtype
        self.device = device
        self.attn_hc = HyperConnection(params, weights.attn_hc, dtype=dtype, use_kernel=use_kernel)
        self.ffn_hc = HyperConnection(params, weights.ffn_hc, dtype=dtype, use_kernel=use_kernel)
        self.attention = MLAAttention(params, weights.attention, dtype=dtype, device=device)

    @classmethod
    def weights_from_hf(
        cls, tensors: dict[str, torch.Tensor], params: Xing4_0Params
    ) -> DecoderLayerWeights:
        """Read one layer out of the released BF16 checkpoint's own names."""
        return DecoderLayerWeights(
            attn_hc=HyperConnectionWeights.from_hf(tensors, params, "attn_hc"),
            ffn_hc=HyperConnectionWeights.from_hf(tensors, params, "ffn_hc"),
            attention=MLAAttentionWeights.from_hf(tensors, params),
            input_layernorm=tensors["input_layernorm.weight"],
            post_attention_layernorm=tensors["post_attention_layernorm.weight"],
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        cache=None,
        start_pos: "int | Pos" = 0,
        absorbed: bool = False,
    ) -> torch.Tensor:
        """`hidden` is `(*batch, tokens, hc, hidden)`; returns the same shape.

        `start_pos` is an `int` on the eager path and a `Pos` on the one a graph replays.  Nothing in
        this method reads it — the two sublayers both hand it to the attention, which is the only
        place in a block that a position means anything.  See
        :mod:`src.models.xing4_0.decode_pos`.
        """
        p = self.params
        # The sublayers work in `dtype` -- the width the GEMMs and the attention
        # were built for -- and the residual streams are carried in
        # `residual_dtype`.  The two are the same on a model whose activations fit
        # in fp16 and they are not on this one: a routed expert's SwiGLU output
        # passes 1e5, which fp16 saturates at 65504 and calls inf, and the inf
        # lands in the residual where every later layer reads it.  What makes the
        # split safe is that a sublayer only ever *sees* the collapsed stream,
        # and that stream goes through `rms_norm` first -- so the sublayer's own
        # arithmetic is on O(1) numbers however large the residual has grown.
        residual = self.residual_dtype
        post, comb, collapsed = self.attn_hc.forward(hidden)
        collapsed = rms_norm(collapsed, self.weights.input_layernorm, p.rms_norm_eps)
        attend = self.attention.forward_absorbed if absorbed else self.attention.forward_expanded
        attn_out = attend(collapsed.to(self.dtype), positions, cache=cache, start_pos=start_pos)
        hidden = post.unsqueeze(-1) * attn_out.to(residual).unsqueeze(-2) + torch.matmul(comb, hidden)

        post, comb, collapsed = self.ffn_hc.forward(hidden)
        collapsed = rms_norm(collapsed, self.weights.post_attention_layernorm, p.rms_norm_eps)
        mlp_out = self.mlp(collapsed.to(self.dtype))
        return post.unsqueeze(-1) * mlp_out.to(residual).unsqueeze(-2) + torch.matmul(comb, hidden)

    def collapse(self, hidden: torch.Tensor) -> torch.Tensor:
        """`hidden_states.mean(dim=2)` -- what the head does before the final norm."""
        return hidden.mean(dim=-2)
