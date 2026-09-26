"""Xing4.0-29B-A4B's matrix hyper-connection: four residual streams, not one.

Every model in this repository before now has had **one** residual stream of width
`hidden`, and a block added its sublayer's output to it.  Xing4.0 has **four**
streams of width 3584, the sublayers still see one, and the residual is not a
tensor an implementation carries but a pair it rebuilds each call:

    hidden_streams  [tokens, 4, 3584]     the state
    pre, post, comb [tokens, 4], [tokens, 4], [tokens, 4, 4]     from the state
    collapsed       [tokens, 3584]        = sum_s pre[s] * hidden[s]
    hidden'[d]      = post[d] * sublayer_out + sum_s comb[d, s] * hidden[s]

So `post[d]` is how much of the sublayer's output stream `d` takes and `comb` is
how the old streams mix into the new ones.  Both are produced by the *same* call
that produces the collapsed input.  An implementation that keeps a conventional
residual accumulator and adds to it produces plausible, wrong text — which is why
`tests/test_xing4_0_hyper_connection.py` pins the forward pass rather than the
shape.

The coefficients come from one projection over all four streams flattened:

    flat = rms_norm(hidden.flatten(-2))             unweighted, over 4*3584 = 14336
    w    = hc_fn @ flat                             [24, 14336] -> [24, tokens]
    pre_w, post_w, comb_w = w.split([4, 4, 16])

`hc_scale` (three scalars, one per gate) and `hc_base` (24, read at those offsets)
are part of the arithmetic: they are initialised to ones and zeros during
training, so a reimplementation that drops either diverges from the released
weights on the first token.

`comb` is a Sinkhorn iterate and ends up approximately doubly stochastic, which
is what makes four parallel streams stable instead of a four-fold gain.  The
iteration count is a config value, not a performance knob.

Stage 4 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).  The reading
this is written from is
[docs/architecture/xing4_0_29b_a4b_audit.md](../../../docs/architecture/xing4_0_29b_a4b_audit.md) §1,
which also cross-checks it against the open llama.cpp port.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.xing4_0.config import Xing4_0Params

__all__ = ["HyperConnection", "HyperConnectionWeights", "sinkhorn"]


@dataclass
class HyperConnectionWeights:
    """`attn_hc`'s or `ffn_hc`'s own three tensors.

    Small: 24 x 14336 is 672 KiB in bf16, and the release deliberately leaves it
    unquantized -- 240 such tensors across the 40 blocks are 52.51 MiB, 0.27% of
    the file, and the quantizer excludes them by name as precision-sensitive
    alongside the router.  So the gate can be fp16 or bf16 with fp32 accumulation
    for the sigmoid, the clamp and the Sinkhorn at no memory cost.
    """

    hc_fn: torch.Tensor  # [24, hc_mult * hidden]
    hc_base: torch.Tensor  # [24]
    hc_scale: torch.Tensor  # [3], one scalar per gate

    @classmethod
    def from_hf(cls, tensors: dict[str, torch.Tensor], params: Xing4_0Params, prefix: str) -> "HyperConnectionWeights":
        """`prefix` is `attn_hc` or `ffn_hc` -- the release names them per sublayer."""
        mix = (2 + params.hc_mult) * params.hc_mult
        wide = params.hc_mult * params.hidden_size
        fn = tensors[f"{prefix}.hc_fn"]
        if tuple(fn.shape) != (mix, wide):
            raise ValueError(f"{prefix}.hc_fn is {tuple(fn.shape)}, expected {(mix, wide)}")
        return cls(
            hc_fn=fn,
            hc_base=tensors[f"{prefix}.hc_base"],
            hc_scale=tensors[f"{prefix}.hc_scale"],
        )


def sinkhorn(comb: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Twenty (row, column) normalizations, in this order, with this epsilon.

    Written as one batched loop over a `(..., hc, hc)` tensor rather than as 20
    separate dispatches per row, because at decode the batch is one row and the
    eager version of this is 40 launches of a 4x4 kernel.  See
    `tests/test_xing4_0_hyper_connection.py` for what depends on both the count
    and the order.
    """
    for _ in range(iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


class HyperConnection:
    """One `hc_*` block.  Stateless apart from the weights it is handed."""

    def __init__(self, params: Xing4_0Params, weights: HyperConnectionWeights, *, dtype: torch.dtype = torch.float32):
        self.params = params
        self.weights = weights
        self.dtype = dtype
        self.hc = int(params.hc_mult)
        self.eps = float(params.hc_eps)
        # `Xing4_0UnweightedRMSNorm(eps=config.rms_norm_eps)` -- the block's norm
        # epsilon, not a constant of its own.
        self.norm_eps = float(params.rms_norm_eps)
        self.iters = int(params.hc_sinkhorn_iters)
        self.clamp_min = float(params.hc_clamp_min)
        self.clamp_max = float(params.hc_clamp_max)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(..., hc, hidden)` -> `(post, comb, collapsed)`.

        `post` is `(..., hc)`, `comb` is `(..., hc, hc)` and `collapsed` is
        `(..., hidden)`.  The reference's own function, with the same splits, the
        same clamp and the same normalization order.
        """
        hc = self.hc
        original_dtype = hidden_streams.dtype
        # The reference writes `flatten(start_dim=2)` because its state is
        # `[batch, seq, hc, hidden]`; flattening the last two dims is the same
        # thing without requiring the batch and sequence axes to be separate.
        # It is not a reshape to `hidden`: the unweighted norm's mean is over the
        # flattened 4 * 3584, so the four streams share one scale.  Unweighted is
        # the point -- `Xing4_0UnweightedRMSNorm` has no learnable scale, unlike
        # the block's `input_layernorm`, and reaching for the weighted one here
        # multiplies the coefficients by 14336 numbers that are not in this
        # gate's weights.
        x = hidden_streams.flatten(start_dim=-2).float()
        flat = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.norm_eps)

        # The reference runs the projection in the model dtype and only then casts
        # to fp32 for the gates.
        w = F.linear(flat.to(original_dtype), self.weights.hc_fn.to(original_dtype)).float()
        pre_w, post_w, comb_w = w.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.weights.hc_base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.weights.hc_scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b)
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        # `dst * hc + src`, which is what `view(hc, hc)` on the split's 16 gives.
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        # The clamp is on the logits, before the exponential, and it is what keeps
        # the Sinkhorn iterate from saturating: +-30 is far outside float32's
        # ability to represent exp, so it is a guard, not a shaping.
        comb_logits = torch.clamp(comb_logits, min=self.clamp_min, max=self.clamp_max)
        comb = torch.exp(comb_logits - comb_logits.amax(dim=-1, keepdim=True))
        comb = sinkhorn(comb, self.iters, self.eps)

        collapsed = (pre.unsqueeze(-1).to(original_dtype) * hidden_streams).sum(dim=-2)
        return post.to(original_dtype), comb.to(original_dtype), collapsed.to(original_dtype)
