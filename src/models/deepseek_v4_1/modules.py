"""The DeepSeek-V4.1 backbone above the attention layers, in pure PyTorch.

`attention.py` covers the CSA2 stack; this module is everything the reference's `Block` and
`Transformer` wrap around it -- the Hyper-Connections residual stream, the Engram n-gram memory,
the routed and shared experts, the gating, and the head. It follows the released
`inference/model.py` op for op, because the point of it is to be checkable against that file
rather than to be fast.

Three things are deliberately not here.

**Expert parallelism and tensor parallelism are absent.** The reference shards experts across ranks
and splits attention heads; at one rank every one of those paths is the identity, which is what is
written. `MoE` keeps the reference's routed/shared split, and the routed half walks experts in id
order exactly as the reference's `for i in range(experts_start_idx, experts_end_idx)` does, so
adding a rank slice later is a change to the range and not to the loop.

**The MTP / DSpark draft head and the ViT tower are absent.** Neither runs in a plain text forward
-- `Transformer.forward` only records `main_hiddens`, and `merge_image_embeddings` is a no-op
without images -- so a text-only runtime does not need them. `Backbone` takes an `image_mask` and
passes it to the gate, which is the whole of what the decoder does with it.

**Weights are ordinary dense tensors.** The released checkpoint stores them fp8 and fp4 with block
scales, and the reference consumes them quantized in a GEMM that needs a tensor core this host does
not have. Here they are dequantized once, at load, and the arithmetic runs in bf16 -- so what this
module reproduces is the reference's *order of operations*, which is what a numerics comparison can
actually be made against. The two big host-resident sets, the 384 routed experts per layer and the
two Engram tables, are behind the `RoutedExperts` and `EngramTable` interfaces: the module tree
asks for one expert or one row at a time and never learns whether it came from device memory or
from a mapped shard.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deepseek_v4_1.attention import (
    Attention,
    RMSNorm,
    SharedAttentionRuntime,
    canonical_device,
)
from src.models.deepseek_v4_1.config import V41TextConfig
from src.kernels.ops import hc_split_sinkhorn

__all__ = [
    "Backbone",
    "Block",
    "Embedding",
    "Engram",
    "EngramTable",
    "Expert",
    "Gate",
    "Head",
    "MoE",
    "ResidentEngramTable",
    "ResidentRoutedExperts",
    "RoutedExperts",
    "check_activation_matches_experts",
    "expert_forward",
    "make_identity_pre_mix",
    "sample",
]


# The compute dtype for every dense weight this module holds. The checkpoint's fp8/fp4 weights are
# expanded into it at load; bf16 is what the reference's norms and residual stream already use, and
# sm_75 has no fp8 or fp4 tensor core to keep them quantized for.
LINEAR_DTYPE = torch.bfloat16


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4 activations
    in range: the up branch is clamped on both sides, the gate branch only from above."""

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        dtype: torch.dtype = LINEAR_DTYPE,
        swiglu_limit: float = 0.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        # `nn.Linear` rather than bare parameters so that the names line up with the checkpoint's:
        # `w1.weight` here is `w1.weight` there, once the fp4 packing is expanded.
        self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=dtype, device=device)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        return expert_forward(x, self.w1.weight, self.w2.weight, self.w3.weight, self.swiglu_limit, weights)


def expert_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """The reference's `Expert.forward`, with the three matrices passed in rather than owned.

    Split out because the routed experts are not `nn.Module`s: a layer holds 384 of them and only
    six are ever read at once, so both of the stores below want to hand in three tensors they just
    produced rather than own a module per expert.

    The gate and up projections are computed in fp32 and the clamp is applied there. That is not
    decoration -- `swiglu_limit` is 10.0 and the expert activations routinely exceed it, so doing
    the clamp in bf16 would round a value that is about to be clamped and change the result.
    """
    dtype = x.dtype
    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    out = F.silu(gate) * up
    if weights is not None:
        out = weights * out
    return F.linear(out.to(dtype), w2)


class RoutedExperts:
    """One layer's routed experts, however they are held.

    An interface rather than a base class: `MoE.forward` needs exactly one thing from it, and the
    two implementations have nothing else in common -- one owns tensors, the other owns a byte
    range of a mapped checkpoint.
    """

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """`x` is `[n, dim]` bf16, `weights`/`indices` are `[n, topk]`. Returns `[n, dim]` fp32."""
        raise NotImplementedError


def check_activation_matches_experts(x: torch.Tensor, where: torch.device, what: str) -> None:
    """Refuse a card-side activation at a host-side expert bank, by name.

    The dense tree can be built on a card while the routed experts stay in host memory -- that is the
    shape this model is served in -- but the two halves do not meet here. A bank that owns host
    tensors can only run the expert on the host, and the path that stages host rows onto the card is
    `DeviceRoutedExperts`, which consumes the checkpoint's packed fp4 rather than an expanded bf16
    matrix. Left unchecked this surfaces as a device mismatch inside `F.linear`, forty layers and
    several minutes into a run, in a kernel that says nothing about which of the two halves is the
    wrong one.
    """
    if x.device != where:
        raise RuntimeError(
            f"{what} holds its experts on {where} but was handed an activation on {x.device}. "
            "A dense tree on a card needs the staging path: build this layer's bank as "
            "`DeviceRoutedExperts` (loader.py, `expert_device=`), which keeps the experts in host "
            "memory and uploads the rows a token routes to."
        )


class ResidentRoutedExperts(RoutedExperts, nn.Module):
    """Every routed expert of one layer held on one device.

    Correct and convenient, and not what the released checkpoint needs: 384 experts is 16.9 MiB of
    packed fp4 each, plus 1.1 MiB of scales, which is 6.7 GiB of codes per layer and 25.3 GiB once
    every one of them is expanded to bf16, so a 40-layer model cannot hold them. This exists for the
    small configs the tests build and for a future device-side expert cache; the checkpoint path is
    `CheckpointRoutedExperts` in `loader.py`.
    """

    def __init__(
        self,
        n_experts: int,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        dtype: torch.dtype = LINEAR_DTYPE,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.swiglu_limit = swiglu_limit
        # One leading axis per matrix rather than one `Expert` module per expert. The checkpoint's
        # names are `experts.{i}.w{1,2,3}.weight`; a bank indexes instead, so a loader mapping one
        # onto the other has to slice. That is the whole of the difference, and it is what keeps a
        # 384-expert layer from being 1152 modules.
        self.w1 = nn.Parameter(torch.empty(n_experts, inter_dim, dim, dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty(n_experts, dim, inter_dim, dtype=dtype, device=device))
        self.w3 = nn.Parameter(torch.empty(n_experts, inter_dim, dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        check_activation_matches_experts(x, self.w1.device, "ResidentRoutedExperts")
        y = torch.zeros_like(x, dtype=torch.float32)
        # The reference walks experts in id order and not in token order, so a token's contribution
        # from expert 7 lands before its contribution from expert 300. The accumulation is a sum, so
        # the order only shows up in the last bits -- but it is the order the reference produces.
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += expert_forward(x[idx], self.w1[i], self.w2[i], self.w3[i], self.swiglu_limit, weights[idx, top, None])
        return y


class EngramTable:
    """One Engram layer's n-gram table, addressed by row id."""

    def lookup(self, indices: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Dequantize the rows `indices` names into bf16 `[..., head_dim]`."""
        raise NotImplementedError


class ResidentEngramTable(EngramTable, nn.Module):
    """A whole Engram table on one device.

    Same caveat as `ResidentRoutedExperts`: the released tables are 183.11 GiB of rows plus 5.72
    GiB of scales across the two layers, which is the single largest thing in the checkpoint. This
    is for the tests and for a truncated table.
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, block_size: int = 32):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

    def lookup(self, indices: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Gather where the table is, then hand the rows to `device`.

        The table stays where it was built -- that is the point of this class and of
        `CheckpointEngramTable` both, the table being 91.55 GiB a layer -- so the indices have to come
        to it before the gather, and only the rows that were asked for travel back. `Engram.wkv` is
        on the card whenever the dense tree is, so a lookup that returned the table's own device
        would put a host tensor into a card-side `Linear`; the explicit move is what makes the two
        halves of that split meet.
        """
        where = self.weight.device
        indices = indices.to(where)
        gathered = dequantize_rows(
            F.embedding(indices, self.weight),
            F.embedding(indices, self.scale),
            self.block_size,
        )
        return gathered if device is None else gathered.to(device)


def dequantize_rows(values: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """The reference's `ParallelEngramEmbedding.forward` dequantization, without the sharding.

    The values come back fp8 and the scales E8M0, one per `block_size` columns; the product is
    formed in fp32 and only then narrowed, which is the order the reference uses. Doing it in bf16
    would round the scale away -- these are 256-wide rows with e8m0 scales that reach 2**-13.
    """
    values = values.float().unflatten(-1, (-1, block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key, signed-sqrt'd before the
    sigmoid, which is what the training kernel does and not what a plain sigmoid would.
    """

    def __init__(self, cfg: V41TextConfig, layer_id: int, layout, table: EngramTable, device=None):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = list(layout.layer_ids).index(layer_id)
        self.dim = cfg.dim
        self.hc_mult = cfg.hc_mult
        self.clamp_value = 1e-6

        self.embed = table
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            cfg.dim * (cfg.hc_mult + 1),
            bias=False,
            dtype=LINEAR_DTYPE,
            device=device,
        )
        self.eps = cfg.norm_eps
        # bf16 rather than the default dtype the reference inherits here: the checkpoint holds them
        # bf16, and `weight` is only ever used as `q_weight * k_weight`, so the width buys nothing.
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim, dtype=LINEAR_DTYPE, device=device))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim, dtype=LINEAR_DTYPE, device=device))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed.lookup(hash_ids, x.device).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from
    the unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`).

    `norm_topk_prob` is the one field the reference always has and one of the two released configs never
    states: `inference/config.json` omits it, and a config that omits it means "unsaid", not "false"
    (`config.py` documents that contract). The reference's own `ModelArgs` defaults it True, so an
    unstated field has to default True here as well -- reading it as a bare truthiness test would
    silently drop the top-k renormalization for every run configured from the flat file, which is a
    routing change and not a rounding one. `gate_temp` is the same kind of field: not in either
    released config, divided by unconditionally in the reference, and defaulted to 1.0 by its
    `ModelArgs`, so it is written here as a divisor defaulting to the same value.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        n_routed_experts: int,
        n_activated_experts: int,
        gate_temp: float = 1.0,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.dim = cfg.dim
        self.topk = n_activated_experts
        self.score_func = cfg.score_func
        self.gate_temp = gate_temp
        self.norm_topk_prob = True if cfg.norm_topk_prob is None else cfg.norm_topk_prob
        self.route_scale = cfg.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, cfg.dim, dtype=LINEAR_DTYPE, device=device))
        self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32, device=device))
        self.bias_vl = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through.

    The routed half sits behind `RoutedExperts` because that is where the whole checkpoint's size
    is: a layer's 384 experts are 7.06 GiB of fp4 and the shared expert is 0.23, so what this class
    decides is the shape of the interface, not the storage.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        layer_id: int,
        n_routed_experts: int,
        n_activated_experts: int,
        expert_dtype: torch.dtype = LINEAR_DTYPE,
        routed: RoutedExperts | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = cfg.dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate = Gate(cfg, n_routed_experts, n_activated_experts, device=device)
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, expert_dtype, cfg.swiglu_limit, device=device)
        self.routed = routed if routed is not None else ResidentRoutedExperts(
            n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, expert_dtype, device=device
        )

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = self.routed.forward(x, weights, indices)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """The mix the first block's attention reads: copy 0 at full weight, the rest at zero."""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    def __init__(self, cfg: V41TextConfig, layer_id: int, max_batch_size: int, max_seq_len: int, layout=None,
                 engram_table: EngramTable | None = None, routed: RoutedExperts | None = None, device=None):
        super().__init__()
        device = canonical_device(device)
        self.layer_id = layer_id
        self.norm_eps = cfg.norm_eps
        self.attn = Attention(layer_id, cfg, max_batch_size, max_seq_len, device=device)
        self.ffn = MoE(cfg, layer_id, *_moe_shape(cfg, layer_id), routed=routed, device=device)
        self.engram = None
        if layout is not None and layer_id in layout.layer_ids:
            if engram_table is None:
                raise ValueError(f"layer {layer_id} carries an Engram memory but no table was supplied")
            self.engram = Engram(cfg, layer_id, layout, engram_table, device=device)
        self.attn_norm = RMSNorm(cfg.dim, self.norm_eps, device=device)
        self.ffn_norm = RMSNorm(cfg.dim, self.norm_eps, device=device)
        self.hc_mult = hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * cfg.dim
        # fp32 in the checkpoint, not bf16: these are the residual coefficients themselves.
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
        pre / post / comb coefficients, split out of one projection of the flattened stream."""
        # normalized over the whole flattened hc*d stream, one statistic per token
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        shared=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias)."""
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, shared)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


def _moe_shape(cfg: V41TextConfig, layer_id: int) -> tuple[int, int]:
    """The reference's `ModelArgs.get_moe_config`: the MTP layers are a different MoE entirely.

    Layer ids past `n_layers` are MTP, and its fallback is `dspark_x or x` rather than `dspark_x` --
    a config that sets `dspark_block_size` and leaves the draft MoE at zero gets the backbone's
    expert counts. This backbone builds no MTP layers, but the lookup is the reference's and
    reading it from the same place is what keeps the two in step if one changes.
    """
    if layer_id < cfg.n_layers:
        return cfg.n_routed_experts, cfg.n_activated_experts
    return (
        cfg.dspark_n_routed_experts or cfg.n_routed_experts,
        cfg.dspark_n_activated_experts or cfg.n_activated_experts,
    )


class Embedding(nn.Module):
    """`ParallelEmbedding` at one rank. The checkpoint stores it bf16."""

    def __init__(self, vocab_size: int, dim: int, device: torch.device | str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=LINEAR_DTYPE, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class Head(nn.Module):
    """`ParallelHead` at one rank: the last position only, and fp32 logits directly.

    The reference keeps this weight fp32 rather than bf16 even though the checkpoint stores bf16,
    because the logits come out of it and the sampler's Gumbel-max reads them at full width.
    """

    def __init__(self, vocab_size: int, dim: int, device: torch.device | str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


class Backbone(nn.Module):
    """DeepSeek-V4.1 text: embed -> expand to hc_mult copies -> blocks -> collapse -> logits.

    The reference's `Transformer` minus the parts a text forward does not execute: the ViT and
    aligner, and the DSpark MTP stack. `main_hidden` -- what `forward` returns for the MTP head --
    is still collected, because it is read *inside* the layer loop (a target layer contributes the
    attention input of the layer itself, not its output) and dropping it would make the loop a
    different loop.
    """

    def __init__(
        self,
        cfg: V41TextConfig,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        layout=None,
        engram_tables: dict[int, EngramTable] | None = None,
        routed: dict[int, RoutedExperts] | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        device = canonical_device(device)
        n_layers = cfg.n_layers if cfg.n_layers is not None else len(cfg.compress_ratios)
        max_seq_len = cfg.max_position_embeddings if max_seq_len is None else max_seq_len
        self.max_seq_len = max_seq_len
        self.hc_mult = cfg.hc_mult
        self.norm_eps = cfg.norm_eps
        self.temperature = getattr(cfg, "temperature", 1.0)
        self.target_layer_ids = tuple(cfg.dspark_target_layer_ids or ())
        self.layers = nn.ModuleList(
            Block(
                cfg,
                layer_id,
                max_batch_size,
                max_seq_len,
                layout=layout,
                engram_table=(engram_tables or {}).get(layer_id),
                routed=(routed or {}).get(layer_id),
                device=device,
            )
            for layer_id in range(n_layers)
        )
        self.embed = Embedding(cfg.vocab_size, cfg.dim, device=device)
        self.norm = RMSNorm(cfg.dim, self.norm_eps, device=device)
        self.head = Head(cfg.vocab_size, cfg.dim, device=device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        hash_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """input_ids: [b, s]. Returns (output_ids, logits, main_hidden).

        `hash_ids` is [b, s, n_engram_layers, n_hash_cols], the row ids `src.encoding.engram` hands
        out. It is required exactly when the model has Engram layers, checked here rather than
        raised from inside the loop: a missing table would otherwise first show up as an attribute
        error 40 layers in, after minutes of expert staging.
        """
        if any(block.engram is not None for block in self.layers) and hash_ids is None:
            raise ValueError("this model has Engram layers, so `hash_ids` is required")
        # image tokens take no part in an n-gram and get no engram contribution; text-only needs no mask
        engram_mask = None if image_mask is None else ~image_mask

        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        shared = SharedAttentionRuntime()
        for i, block in enumerate(self.layers):
            if block.engram is not None:
                h = block.engram(h, hash_ids[:, :, block.engram.layer_hash_index, :], engram_mask)
            # the MTP head reads the attention input of its target layers, not their output
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = block(h, start_pos, pre_mix, image_mask, shared)
        h = self.layers[-1].hc_pre(h, pre_mix)
        logits = self.head(self.norm(h))
        output_ids = sample(logits, self.temperature)
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return output_ids, logits, main_hidden

    def reset_state(self, batch_size: int) -> None:
        for block in self.layers:
            block.attn.reset_state(batch_size)


def sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
