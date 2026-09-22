"""MiMo-V2 text decoder layer, in plain torch, as the host reference.

Numerics follow the checkpoint's own remote code (`modeling_mimo_v2.py`) exactly,
including the places where it forces float32, because the parity tests compare
against goldens captured from that file. Nothing here is tensor-parallel or
quantized: weights arrive as dense tensors, and the quantized layouts (MXFP4
experts, FP8 E4M3 linears under 128x128 block scales) are the caller's problem.
Keeping this module dense is deliberate -- it is the thing a kernel port is
diffed against, so it has to be readable rather than fast.

Two details are easy to get subtly wrong and are called out where they live:

* The value scale is applied to the value states *before* attention, inside the
  attention block, so a caller that scales after the output projection is wrong.
* The attention sink is a concatenated softmax column, not an additive logit
  bias, and the row maximum that precedes the softmax is taken over the
  *extended* row -- so the sink changes the normalisation of every other entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from src.models.mimo_v2.config import MimoV2AttentionShape, MimoV2TextConfig

__all__ = [
    "MimoV2ExpertWeights",
    "MimoV2QuantizedExperts",
    "MimoV2LayerWeights",
    "MimoV2DecoderLayer",
    "MimoV2HostModel",
    "build_attention_masks",
    "build_rope_cos_sin",
    "build_rope_inv_freq",
    "gate_and_route",
    "repeat_kv",
    "rms_norm",
    "rope_dim",
    "rotate_half",
    "route_and_mix",
    "swiglu_mlp",
]

#: The reference's `compute_default_rope_parameters` returns a scaling of 1.0 for
#: the `default` rope type, which is what the released config uses
#: (`rope_parameters.rope_type: "default"`, no `rope_scaling`).
DEFAULT_ATTENTION_SCALING = 1.0

#: `silu` is the released config's `hidden_act`.
_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"silu": F.silu}


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference's `MiMoV2RMSNorm`.

    The variance is accumulated in float32, then the result is cast back before the
    weight is applied -- so the multiply by `weight` happens in the input dtype,
    not in float32. Casting after the multiply instead is within rounding of the
    same answer for fp32 weights, and visibly different for bf16 ones.
    """
    input_dtype = hidden_states.dtype
    upcast = hidden_states.to(torch.float32)
    variance = upcast.pow(2).mean(-1, keepdim=True)
    upcast = upcast * torch.rsqrt(variance + eps)
    return weight * upcast.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings: one table per attention family
# ---------------------------------------------------------------------------


def rope_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """How much of each head rotates; the reference rejects an odd value."""
    dim = int(head_dim * partial_rotary_factor)
    if dim % 2 != 0:
        raise ValueError(
            f"rotary dimension must be even, got {dim} from head_dim={head_dim} "
            f"and partial_rotary_factor={partial_rotary_factor}"
        )
    return dim


def build_rope_inv_freq(
    dim: int,
    base: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """`1 / base ** (arange(0, dim, 2) / dim)` over `dim // 2` frequencies."""
    return 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )


def build_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: float = DEFAULT_ATTENTION_SCALING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[batch, seq, dim]` cos and sin, one entry per rotated coordinate.

    `position_ids` is `[batch, seq]`. The frequencies are duplicated rather than
    halved because the rotation interleaves as `[-x2, x1]` over a split in the
    middle of the rotated span, which needs a full-width table.
    """
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    inv_freq_expanded = inv_freq_expanded.to(position_ids.device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Rotate the leading `dim` of each head, leave the rest untouched.

    `states` is `[batch, heads, seq, head_dim]`; `cos`/`sin` are `[batch, seq, dim]`
    and are broadcast over heads, which is why the head axis is unsqueezed at 1.
    """
    rope, nope = states.split([dim, states.shape[-1] - dim], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rope = (rope * cos) + (rotate_half(rope) * sin)
    return torch.cat([rope, nope], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Materialise the grouped-query repetition rather than relying on broadcast."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention with an optional softmax sink column.

    The sink is concatenated as one extra key whose "value" does not exist: the
    softmax runs over `seq + 1` entries and the extra column is then discarded, so
    the sink only absorbs probability mass. Because the row maximum is taken after
    the concatenation, the sink also shifts the normalisation of every real entry.
    """
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    if sink is not None:
        expanded = sink.reshape(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1
        )
        attn_weights = torch.cat([attn_weights, expanded], dim=-1)

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if sink is not None:
        probs = probs[..., :-1]

    attn_output = torch.matmul(probs, value_states)
    return attn_output.transpose(1, 2).contiguous(), probs


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def gate_and_route(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    scoring_func: str = "sigmoid",
    topk_method: str = "noaux_tc",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score tokens, choose experts on corrected scores, weight them by raw ones.

    Returns `(topk_idx, topk_weight, logits, scores, scores_for_choice)`.

    The split between the two score tensors is the whole point: `noaux_tc` adds a
    learned correction bias to decide *which* experts a token takes, and then
    weights the chosen ones by the uncorrected sigmoid score. The reference also
    discards the top-k ordering (`sorted=False`) and renormalises by the actual
    sum, so the weights are a partition of 1 across the chosen experts.

    The reference is only ever called with `[batch, seq, hidden]`, and its first
    act is `.view(-1, h)`. A flat `[rows, hidden]` is accepted here because that is
    the same arithmetic on the same rows, and it is the shape a decode step has.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    bsz, seq_len, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)

    # Both operands are upcast explicitly; the released config stores the router
    # weight in bf16, so this upcast is part of the arithmetic and not an
    # optimisation.
    logits = F.linear(flat.to(torch.float32), weight.to(torch.float32), None)
    if scoring_func != "sigmoid":
        raise NotImplementedError(f"unsupported MiMo-V2 scoring function: {scoring_func}")
    scores = logits.sigmoid()

    if topk_method != "noaux_tc":
        raise NotImplementedError(f"unsupported MiMo-V2 topk method: {topk_method}")
    if correction_bias is None:
        raise ValueError("noaux_tc routing needs the expert-score correction bias")

    rows = bsz * seq_len
    scores_for_choice = scores.view(rows, -1) + correction_bias.unsqueeze(0)

    # Group selection: each group is scored by its best two experts.
    group_scores = scores_for_choice.view(rows, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(rows, n_group, scores_for_choice.shape[-1] // n_group)
        .reshape(rows, -1)
    )
    masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_idx = torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]
    topk_weight = scores.gather(1, topk_idx)

    if top_k > 1 and norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * routed_scaling_factor
    return topk_idx, topk_weight, logits, scores, scores_for_choice


def route_and_mix(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_fn: Callable[[int, torch.Tensor], torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """Apply each expert to the tokens routed to it and scatter the results back.

    `expert_fn(expert_idx, tokens)` is the seam a quantized port replaces, and it
    is per-expert on purpose: the summed layer output can agree while one expert is
    wrong, or while two experts' errors cancel. Accumulation follows the
    reference's `index_add_` in float32 (`topk_weight.dtype`) and is cast to the
    hidden dtype only at the end.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    accum = torch.zeros_like(flat, dtype=topk_weight.dtype)

    expert_mask = F.one_hot(topk_idx, num_classes=num_experts).permute(2, 0, 1)
    for expert_idx in range(num_experts):
        token_indices, weight_indices = torch.where(expert_mask[expert_idx])
        if token_indices.numel() == 0:
            continue
        weights = topk_weight[token_indices, weight_indices]
        out = expert_fn(expert_idx, flat[token_indices])
        accum.index_add_(0, token_indices, out * weights.unsqueeze(-1))

    return accum.to(hidden_states.dtype).reshape(*hidden_states.shape)


def swiglu_mlp(
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act: str = "silu",
) -> torch.Tensor:
    """`down(act(gate(x)) * up(x))`, the shape both the dense FFN and an expert use."""
    if act not in _ACTIVATIONS:
        raise NotImplementedError(f"unsupported MiMo-V2 activation: {act}")
    inner = _ACTIVATIONS[act](F.linear(hidden_states, gate_proj))
    return F.linear(inner * F.linear(hidden_states, up_proj), down_proj)


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------


def build_attention_masks(
    seq_len: int,
    sliding_window: int | None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """The two masks the reference builds with `transformers.masking_utils`.

    Both are additive `[batch, 1, seq, seq]` with `finfo(dtype).min` where a key is
    forbidden, and both are causal. The window keeps `sliding_window` positions
    counting the query itself, so key `j` is visible to query `i` when
    `0 <= i - j < sliding_window`.

    Satisfying the window is not the same as satisfying causality here: the
    reference builds the two masks through separate helpers and hands each layer
    the one its attention type names, so a global layer is not windowed and a
    windowed layer does not see the full prefix.
    """
    device = device if device is not None else torch.device("cpu")
    queries = torch.arange(seq_len, device=device).unsqueeze(1)
    keys = torch.arange(seq_len, device=device).unsqueeze(0)

    forbidden = torch.finfo(dtype).min
    causal = (keys > queries).expand(batch_size, 1, seq_len, seq_len)
    masks = {"full_attention": torch.where(causal, forbidden, 0.0).to(dtype)}
    if sliding_window is not None:
        windowed = causal | ((queries - keys) >= sliding_window)
        masks["sliding_window_attention"] = torch.where(windowed, forbidden, 0.0).to(dtype)
    return masks


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimoV2ExpertWeights:
    """One routed expert. Dense here; MXFP4 blocks in the checkpoint."""

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


class MimoV2QuantizedExperts:
    """The seam a packed expert layout implements.

    `layers.py` never needs to know how an expert is stored -- only that it can run
    one on a set of tokens. Implementations are `weights.py`'s `MimoV2Mxfp4Experts`
    (host, torch, for the reference) and the device pool that replaces it.
    """

    #: How many experts this source can serve. The router's ids must stay below it.
    n_experts: int

    def expert_fn(self, act: str = "silu") -> Callable[[int, torch.Tensor], torch.Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class MimoV2LayerWeights:
    """One decoder layer's tensors, in the checkpoint's own layout.

    `qkv_proj` is the fused projection, whose row count differs by family
    (13568 on a global layer, 14848 on a sliding-window one) and whose sections are
    q, then k, then v. `sink` is present only where the config says that layer's
    family carries one. `experts` is empty on a dense layer, and `gate`/`correction_bias`
    are `None` there.
    """

    qkv_proj: torch.Tensor
    o_proj: torch.Tensor
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    sink: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    correction_bias: torch.Tensor | None = None
    experts: tuple[MimoV2ExpertWeights, ...] = ()
    mlp_gate_proj: torch.Tensor | None = None
    mlp_up_proj: torch.Tensor | None = None
    mlp_down_proj: torch.Tensor | None = None
    #: A routed expert that is already quantized -- the released checkpoint's MXFP4
    #: experts are, and expanding them to dense tensors costs 12.75 MiB of codes
    #: becoming 100 MiB of float32. When set, it replaces `experts` entirely:
    #: `expert_fn` runs it, and the dense tuple is ignored.
    expert_source: "MimoV2QuantizedExperts | None" = None
    #: Where these tensors came from, for error messages.
    source: str = ""

    @classmethod
    def from_tensors(
        cls,
        tensors: Mapping[str, torch.Tensor],
        layer_idx: int,
        config: MimoV2TextConfig,
        prefix: str = "model.layers",
    ) -> "MimoV2LayerWeights":
        """Assemble one layer from a flat name -> tensor mapping.

        The names are the checkpoint's, so the same mapping serves the fixture and
        the released shards; only the tensors behind them differ.
        """
        root = f"{prefix}.{layer_idx}"
        kind = config.ffn_kind(layer_idx)

        def get(name: str) -> torch.Tensor:
            key = f"{root}.{name}"
            if key not in tensors:
                raise KeyError(f"missing tensor {key!r} for layer {layer_idx} ({kind})")
            return tensors[key]

        def maybe(name: str) -> torch.Tensor | None:
            return tensors.get(f"{root}.{name}")

        experts: tuple[MimoV2ExpertWeights, ...] = ()
        if kind == "moe":
            experts = tuple(
                MimoV2ExpertWeights(
                    gate_proj=get(f"mlp.experts.{e}.gate_proj.weight"),
                    up_proj=get(f"mlp.experts.{e}.up_proj.weight"),
                    down_proj=get(f"mlp.experts.{e}.down_proj.weight"),
                )
                for e in range(config.n_routed_experts)
            )

        return cls(
            qkv_proj=get("self_attn.qkv_proj.weight"),
            o_proj=get("self_attn.o_proj.weight"),
            input_layernorm=get("input_layernorm.weight"),
            post_attention_layernorm=get("post_attention_layernorm.weight"),
            sink=maybe("self_attn.attention_sink_bias"),
            gate=maybe("mlp.gate.weight") if kind == "moe" else None,
            correction_bias=maybe("mlp.gate.e_score_correction_bias") if kind == "moe" else None,
            experts=experts,
            mlp_gate_proj=maybe("mlp.gate_proj.weight") if kind == "dense" else None,
            mlp_up_proj=maybe("mlp.up_proj.weight") if kind == "dense" else None,
            mlp_down_proj=maybe("mlp.down_proj.weight") if kind == "dense" else None,
            source=root,
        )

    def expert_fn(self, config: MimoV2TextConfig) -> Callable[[int, torch.Tensor], torch.Tensor]:
        """The `route_and_mix` seam: `(expert_idx, tokens) -> expert output`.

        A quantized expert source wins over the dense tuple, because a released
        checkpoint's experts are MXFP4 and expanding them here would defeat the
        point of keeping them packed.
        """
        if self.expert_source is not None:
            return self.expert_source.expert_fn(config.hidden_act)
        act = config.hidden_act
        experts = self.experts
        if not experts:
            raise ValueError(f"{self.source} is a dense layer and has no routed experts")

        def run(expert_idx: int, tokens: torch.Tensor) -> torch.Tensor:
            expert = experts[expert_idx]
            return swiglu_mlp(tokens, expert.gate_proj, expert.up_proj, expert.down_proj, act)

        return run


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


@dataclass
class MimoV2DecoderLayer:
    """One pre-norm decoder layer: attention, a residual add, an FFN, another add."""

    config: MimoV2TextConfig
    layer_idx: int
    weights: MimoV2LayerWeights
    shape: MimoV2AttentionShape = field(init=False)

    def __post_init__(self) -> None:
        self.shape = self.config.attention(self.layer_idx)
        rows = self.weights.qkv_proj.shape[0]
        if rows != self.shape.qkv_out:
            raise ValueError(
                f"{self.weights.source}: qkv_proj has {rows} rows but layer "
                f"{self.layer_idx} ({self.shape.family}) needs {self.shape.qkv_out}"
            )

    @property
    def kind(self) -> str:
        return self.config.ffn_kind(self.layer_idx)

    def rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's family's cos/sin table."""
        inv_freq = build_rope_inv_freq(self.shape.rope_dim, self.shape.rope_theta)
        return build_rope_cos_sin(inv_freq, position_ids)

    def attend(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Attention alone, returning the boundaries a kernel port is diffed at.

        `hidden_states` is the *post* input-norm stream, exactly as `forward`
        passes it: the reference's `qkv_proj` is fed the normalised activations, so
        comparing a `qkv_proj` output against one produced from the pre-norm stream
        is off by the whole RMSNorm.
        """
        shape = self.shape
        if shape.projection_layout != "fused_qkv":
            raise NotImplementedError(
                f"{self.weights.source}: only the fused qkv layout is implemented, "
                f"got {shape.projection_layout!r}"
            )
        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0)
            position_embeddings = self.rope(position_ids)
        cos, sin = position_embeddings

        input_shape = hidden_states.shape[:-1]
        qkv = F.linear(hidden_states, self.weights.qkv_proj)
        query, key, value = qkv.split([shape.q_size, shape.k_size, shape.v_size], dim=-1)

        heads = shape.num_q_heads
        kv_heads = shape.num_kv_heads
        query = query.view(*input_shape, heads, shape.head_dim).transpose(1, 2)
        key = key.view(*input_shape, kv_heads, shape.head_dim).transpose(1, 2)
        value = value.view(*input_shape, kv_heads, shape.v_head_dim).transpose(1, 2)

        # The value scale belongs here, before attention and before any cache.
        if shape.value_scale is not None:
            value = value * shape.value_scale

        query = apply_partial_rope(query, cos, sin, shape.rope_dim)
        key = apply_partial_rope(key, cos, sin, shape.rope_dim)

        attn_output, probs = attention(
            query,
            key,
            value,
            attention_mask,
            shape.scaling,
            shape.num_key_value_groups,
            self.weights.sink,
        )
        pre_o = attn_output.reshape(*input_shape, -1).contiguous()
        post_o = F.linear(pre_o, self.weights.o_proj)
        return {
            "qkv_raw": qkv,
            "qkv_in": hidden_states,
            "attn_out_pre_o": pre_o,
            "attn_out_post_o": post_o,
            "probs": probs,
        }

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.weights.gate is None:
            raise ValueError(f"{self.weights.source} has no router")
        return gate_and_route(
            hidden_states,
            self.weights.gate,
            self.weights.correction_bias,
            top_k=self.config.num_experts_per_tok,
            n_group=self.config.n_group,
            topk_group=self.config.topk_group,
            norm_topk_prob=self.config.resolved_norm_topk_prob,
            routed_scaling_factor=self.config.resolved_routed_scaling_factor,
            scoring_func=self.config.scoring_func,
            topk_method=self.config.topk_method,
        )

    def mlp(
        self,
        hidden_states: torch.Tensor,
        capture_experts: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The FFN, dense or routed.

        `capture_experts`, when given, is filled with each routed expert's own
        output so a test can compare experts individually.
        """
        if self.kind == "dense":
            return swiglu_mlp(
                hidden_states,
                self.weights.mlp_gate_proj,
                self.weights.mlp_up_proj,
                self.weights.mlp_down_proj,
                self.config.hidden_act,
            )

        topk_idx, topk_weight = self.route(hidden_states)[:2]
        run = self.weights.expert_fn(self.config)

        def observed(expert_idx: int, tokens: torch.Tensor) -> torch.Tensor:
            out = run(expert_idx, tokens)
            if capture_experts is not None:
                capture_experts[expert_idx] = out
            return out

        return route_and_mix(
            hidden_states, topk_idx, topk_weight, observed, self.config.n_routed_experts
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        capture_experts: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = rms_norm(
            hidden_states, self.weights.input_layernorm, self.config.layernorm_epsilon
        )
        hidden_states = self.attend(
            hidden_states, attention_mask, position_embeddings, position_ids
        )["attn_out_post_o"]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = rms_norm(
            hidden_states, self.weights.post_attention_layernorm, self.config.layernorm_epsilon
        )
        hidden_states = self.mlp(hidden_states, capture_experts=capture_experts)
        return residual + hidden_states

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# The backbone
# ---------------------------------------------------------------------------


class MimoV2HostModel:
    """The whole text backbone on the host, assembled from a flat tensor mapping.

    Exists so the stack can be exercised end to end without a GPU and without a
    built native engine -- the fixture and the released checkpoint differ only in
    the tensors handed to the constructor.
    """

    def __init__(
        self,
        config: MimoV2TextConfig,
        tensors: Mapping[str, torch.Tensor],
        prefix: str = "model",
        head_prefix: str = "lm_head",
        strict: bool = True,
    ) -> None:
        self.config = config
        self.tensors = dict(tensors)
        self.prefix = prefix
        self.layers = [
            MimoV2DecoderLayer(
                config,
                layer_idx,
                MimoV2LayerWeights.from_tensors(
                    tensors, layer_idx, config, f"{prefix}.layers"
                ),
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.embed_tokens = _require(tensors, f"{prefix}.embed_tokens.weight")
        self.norm = _require(tensors, f"{prefix}.norm.weight")
        self.lm_head = tensors.get(f"{head_prefix}.weight")
        if strict:
            self._check_all_tensors_used()

    def _check_all_tensors_used(self) -> None:
        """Every backbone tensor in the mapping should have been claimed.

        A silently unused tensor is worse than a missing one: a checkpoint that
        stores something this module does not read would run and be wrong.
        """
        used = {f"{self.prefix}.embed_tokens.weight", f"{self.prefix}.norm.weight"}
        for layer in self.layers:
            root = f"{self.prefix}.layers.{layer.layer_idx}"
            used |= {
                f"{root}.self_attn.qkv_proj.weight",
                f"{root}.self_attn.o_proj.weight",
                f"{root}.input_layernorm.weight",
                f"{root}.post_attention_layernorm.weight",
            }
            if layer.weights.sink is not None:
                used.add(f"{root}.self_attn.attention_sink_bias")
            if layer.weights.gate is not None:
                used.add(f"{root}.mlp.gate.weight")
                used.add(f"{root}.mlp.gate.e_score_correction_bias")
            for e in range(len(layer.weights.experts)):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    used.add(f"{root}.mlp.experts.{e}.{proj}.weight")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if getattr(layer.weights, f"mlp_{proj}") is not None:
                    used.add(f"{root}.mlp.{proj}.weight")
        if self.lm_head is not None:
            used.add("lm_head.weight")
        backbone_prefixes = (
            f"{self.prefix}.layers",
            f"{self.prefix}.embed",
            f"{self.prefix}.norm",
        )
        unused = sorted(
            name
            for name in self.tensors
            if name not in used and name.startswith(backbone_prefixes)
        )
        if unused:
            raise ValueError(
                "the tensor mapping carries backbone weights this module does not read: "
                + ", ".join(unused[:8])
                + (" ..." if len(unused) > 8 else "")
            )

    def rope_tables(self, position_ids: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """The two tables the reference builds once and hands to every layer."""
        by_family: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for shape in (self.config.attention(i) for i in range(self.config.num_hidden_layers)):
            if shape.family in by_family:
                continue
            inv_freq = build_rope_inv_freq(shape.rope_dim, shape.rope_theta)
            by_family[shape.family] = build_rope_cos_sin(inv_freq, position_ids)
        return by_family

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Mapping[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        capture_experts: dict[int, dict[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Token ids to logits, unmasked by default.

        The embedding is not scaled by `sqrt(hidden)`, and the head is not tied to
        it -- both are the reference's behaviour and both are checked.
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        if attention_mask is None:
            attention_mask = build_attention_masks(
                input_ids.shape[1], self.config.resolved_window
            )

        hidden_states = F.embedding(input_ids, self.embed_tokens)
        tables = self.rope_tables(position_ids)
        for layer in self.layers:
            family = layer.shape.family
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask[_mask_key(family)],
                position_embeddings=tables[family],
                position_ids=position_ids,
                capture_experts=None
                if capture_experts is None
                else capture_experts.setdefault(layer.layer_idx, {}),
            )

        hidden_states = rms_norm(hidden_states, self.norm, self.config.layernorm_epsilon)
        if self.lm_head is None:
            return hidden_states
        return F.linear(hidden_states, self.lm_head)


def _mask_key(family: str) -> str:
    return "sliding_window_attention" if family == "swa" else "full_attention"


def _require(tensors: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in tensors:
        raise KeyError(f"missing tensor {key!r}")
    return tensors[key]
