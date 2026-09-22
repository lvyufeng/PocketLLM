"""The released checkpoint's tensors, assembled into the shapes `layers.py` runs on.

`layers.py` is the reference the port is diffed against, and it is only useful if
it can run the *real* weights -- there is no second implementation of MiMo to
compare against, and no 4 x 80 GB host to run vLLM on. This module is the bridge:
it turns `MimoV2Checkpoint`'s mmap views into `MimoV2LayerWeights`, dequantizing
FP8 dense linears once (sm_75 has no FP8, and neither does the host reference) and
leaving the MXFP4 experts **packed**.

That last part is the important one. An expert is 12.75 MiB of codes and scales
that expands to 100 MiB of float32, so a layer's 256 experts would be 25 GB and the
model's would be 4.7 TB. `MimoV2Mxfp4Experts` instead holds the checkpoint's own
uint8 views for one layer and dequantizes an expert only when the router selects
it, which for a token is eight experts and not 256. The consequence is that a
whole-model forward on the real checkpoint fits in about 12 GB of host RAM and
never materialises an expert nobody used.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Mapping, Sequence

import torch

from src.models.mimo_v2.layers import (
    MimoV2DecoderLayer,
    MimoV2HostModel,
    MimoV2LayerWeights,
    MimoV2QuantizedExperts,
    swiglu_mlp,
)
from src.models.mimo_v2.quant import dequant_mxfp4

__all__ = [
    "MimoV2Mxfp4Experts",
    "host_model_from_checkpoint",
    "layer_weights_from_checkpoint",
    "router_weights",
]


class MimoV2Mxfp4Experts(MimoV2QuantizedExperts):
    """One layer's routed experts, still packed, dequantized on selection.

    The codes and scales are the checkpoint's mapped bytes, so constructing this
    costs nothing and holding it costs no more than the layer costs on disk. Each
    `expert_fn` call dequantizes the three projections of one expert -- the
    cheapest thing that can answer the question `route_and_mix` asks.

    A dequantized expert is held for the life of the object when `cache` is set, so
    a layer whose tokens keep hitting the same expert pays for it once. Without it
    each call re-dequantizes, which is what makes the memory bound independent of
    how many tokens are routed.
    """

    def __init__(
        self,
        checkpoint,
        layer: int,
        dtype: torch.dtype = torch.float32,
        cache: bool = False,
        experts: Iterable[int] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.layer = layer
        self.dtype = dtype
        self.n_experts = checkpoint.layer.n_routed_experts
        self._cache = cache
        self._dense: dict[int, dict[str, torch.Tensor]] = {}
        #: When set, only these experts can be served; a router that selects any
        #: other one is a routing bug and raises rather than reading a zero expert.
        self._allowed = None if experts is None else frozenset(experts)

    def dense_expert(self, expert: int) -> dict[str, torch.Tensor]:
        """The three projections of one expert as dense float tensors."""
        if self._allowed is not None and expert not in self._allowed:
            raise KeyError(
                f"layer {self.layer} expert {expert} was not loaded into this source; "
                f"loaded: {len(self._allowed)} of {self.n_experts}"
            )
        if self._cache and expert in self._dense:
            return self._dense[expert]
        arrays = self.checkpoint.expert_arrays(self.layer, expert)
        dense = {
            proj: dequant_mxfp4(
                arrays[(proj, "weight")], arrays[(proj, "weight_scale")], 32, self.dtype
            )
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        if self._cache:
            self._dense[expert] = dense
        return dense

    def expert_fn(self, act: str = "silu") -> Callable[[int, torch.Tensor], torch.Tensor]:
        def run(expert_idx: int, tokens: torch.Tensor) -> torch.Tensor:
            dense = self.dense_expert(expert_idx)
            return swiglu_mlp(
                tokens, dense["gate_proj"], dense["up_proj"], dense["down_proj"], act
            )

        return run

    def dropped_cache(self) -> None:
        """Release any cached experts, so a long run does not grow with hotness."""
        self._dense.clear()


def router_weights(checkpoint, layer: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One layer's router: the gate matrix and the `noaux_tc` correction bias."""
    root = f"model.layers.{layer}.mlp.gate"
    gate = checkpoint.dense_tensor(f"{root}.weight", torch.float32)
    bias_key = f"{root}.e_score_correction_bias"
    bias = checkpoint.dense_tensor(bias_key, torch.float32) if bias_key in checkpoint else None
    return gate, bias


def layer_weights_from_checkpoint(
    checkpoint,
    layer_idx: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    experts: Iterable[int] | None = None,
    expert_cache: bool = False,
) -> MimoV2LayerWeights:
    """One decoder layer's tensors, from the release.

    The FP8 `qkv_proj` is dequantized on the way through, including the released
    global-attention layer's two idle scale rows. The routed experts stay packed:
    `expert_source` carries them and `experts` stays empty, so a caller that wants a
    dense expert tuple is asking for something this loader deliberately does not do.
    """
    config = checkpoint.layer
    root = f"model.layers.{layer_idx}"
    kind = config.ffn_kind(layer_idx)

    sink_key = f"{root}.self_attn.attention_sink_bias"
    weights = MimoV2LayerWeights(
        qkv_proj=checkpoint.dense_tensor(f"{root}.self_attn.qkv_proj.weight", dtype, device),
        o_proj=checkpoint.dense_tensor(f"{root}.self_attn.o_proj.weight", dtype, device),
        input_layernorm=checkpoint.dense_tensor(f"{root}.input_layernorm.weight", dtype, device),
        post_attention_layernorm=checkpoint.dense_tensor(
            f"{root}.post_attention_layernorm.weight", dtype, device
        ),
        sink=(
            checkpoint.dense_tensor(sink_key, torch.float32, device)
            if sink_key in checkpoint
            else None
        ),
        source=root,
    )

    if kind == "moe":
        gate, bias = router_weights(checkpoint, layer_idx)
        return replace(
            weights,
            gate=gate.to(device),
            correction_bias=None if bias is None else bias.to(device),
            expert_source=MimoV2Mxfp4Experts(
                checkpoint, layer_idx, dtype, cache=expert_cache, experts=experts
            ),
        )
    return replace(
        weights,
        mlp_gate_proj=checkpoint.dense_tensor(f"{root}.mlp.gate_proj.weight", dtype, device),
        mlp_up_proj=checkpoint.dense_tensor(f"{root}.mlp.up_proj.weight", dtype, device),
        mlp_down_proj=checkpoint.dense_tensor(f"{root}.mlp.down_proj.weight", dtype, device),
    )


def host_model_from_checkpoint(
    checkpoint,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    layers: Sequence[int] | None = None,
    expert_cache: bool = False,
    expert_layers: Mapping[int, Iterable[int]] | None = None,
) -> MimoV2HostModel:
    """The whole text backbone on the real weights, or a slice of it.

    `layers` and `expert_layers` exist because a single layer is what a kernel port
    is actually diffed on, and materializing all 48 layers' attention just to reach
    layer 23 costs several minutes of I/O for nothing. When `layers` is given the
    returned model is a `MimoV2HostModel` whose `layers` list holds only those, so
    `forward` would run a truncated stack; the intended use is calling the layers
    directly, which is what the tests do.

    `expert_layers` restricts which experts each MoE layer may serve. A test that
    pre-computes the router's selection can pass exactly that set and turn a
    256-expert layer into an 8-expert one.
    """
    config = checkpoint.layer
    wanted = list(range(config.num_hidden_layers)) if layers is None else list(layers)
    expert_layers = expert_layers or {}

    model = MimoV2HostModel.__new__(MimoV2HostModel)
    model.config = config
    model.tensors = {}
    model.prefix = "model"
    model.embed_tokens = checkpoint.dense_tensor("model.embed_tokens.weight", dtype, device)
    model.norm = checkpoint.dense_tensor("model.norm.weight", dtype, device)
    model.lm_head = (
        checkpoint.dense_tensor("lm_head.weight", dtype, device)
        if "lm_head.weight" in checkpoint
        else None
    )
    model.layers = [
        MimoV2DecoderLayer(
            config,
            layer_idx,
            layer_weights_from_checkpoint(
                checkpoint,
                layer_idx,
                dtype,
                device,
                experts=expert_layers.get(layer_idx),
                expert_cache=expert_cache,
            ),
        )
        for layer_idx in wanted
    ]
    return model
