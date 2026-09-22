"""The whole MiMo-V2.6 text backbone on one card, a token at a time.

This is the stage where the pieces become a model: `device_attention.py` runs a
layer's attention against a cache, `device_experts.py` runs a layer's routed experts
out of host memory, and this file runs forty-eight of those layers over a KV cache and
ends at the logits. Nothing here is new arithmetic -- the layer's own arithmetic is
`layers.py`'s, moved -- so the value of the stage is that it can now be *measured* and
that every later optimization has somewhere to land.

What it is not is fast, and the first measurement says why. A decode step draws eight
experts a layer out of the bank, which is 102 MiB of host-to-device traffic a layer and
**4.68 GiB a token**; at the 10.4 GiB/s the pinned link sustains that is 450 ms of a
610 ms token, and the attention, at 64 ms, is nowhere in it. That is a property of
running every expert of every layer on one rank, and it is what the expert-parallel
stage is for.

Three details of the assembly are the model's and are easy to get wrong:

* The embedding is not scaled by `sqrt(hidden)`, and the head is not tied to it. Both
  are the reference's behaviour and both are checked.
* The two attention families want two different masks, and the windowed mask is not a
  prefix: a windowed layer at 256k reads 128 keys and a global layer reads all of them.
  `MimoV2DeviceAttention` takes the bounds instead of a mask, so this file passes a
  position and nothing else -- there is no mask tensor to build at 256k.
* A layer's residual is added in the hidden dtype on both sides. The routed sum leaves
  the kernel in float32 and is rounded once, where the reference rounds it once.

A decode step is the only step this file can run. The expert kernel is the single-token
one, so a chunk of tokens has to go through the grouped-prefill kernel, which is not
routed into yet -- a prefill here would draw every token's experts one at a time and
cost a token per token. The cache is what makes the *decode* a stream rather than a
re-run of the prefix.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.mimo_v2.config import MimoV2TextConfig
from src.models.mimo_v2.device_attention import MimoV2DeviceAttention, MimoV2KVCache
from src.models.mimo_v2.device_experts import MimoV2DeviceExperts, MimoV2ExpertSource
from src.models.mimo_v2.layers import gate_and_route, rms_norm, swiglu_mlp

__all__ = [
    "MimoV2DeviceLayer",
    "MimoV2DeviceModel",
]


class MimoV2DeviceLayer:
    """One decoder layer on a card: norm, attention, norm, FFN.

    The FFN is dense on layer 0 and routed on the other forty-seven, and the routed one
    is a *draw*: the router picks eight experts, the shared arena stages those eight,
    and the kernel sums their weighted outputs. The arena is shared across layers --
    forty-seven of them at 102 MiB would be 4.8 GiB of a 22 GiB card for state that is
    read once a layer a token -- which is why `experts` is handed in rather than built
    here.
    """

    def __init__(
        self,
        checkpoint,
        layer_idx: int,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        experts: MimoV2DeviceExperts | None = None,
        block: int = 1024,
        budget: int = 1 << 25,
    ) -> None:
        self.checkpoint = checkpoint
        self.config: MimoV2TextConfig = checkpoint.layer
        self.layer_idx = int(layer_idx)
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = self.config.attention(self.layer_idx)
        self.kind = self.config.ffn_kind(self.layer_idx)

        self.attention = MimoV2DeviceAttention(
            checkpoint, self.layer_idx, self.device, dtype, block=block, budget=budget
        )
        root = f"model.layers.{self.layer_idx}"
        self.input_layernorm = checkpoint.dense_tensor(
            f"{root}.input_layernorm.weight", dtype, self.device
        )
        self.post_attention_layernorm = checkpoint.dense_tensor(
            f"{root}.post_attention_layernorm.weight", dtype, self.device
        )

        self.gate = None
        self.correction_bias = None
        self.mlp_gate_proj = self.mlp_up_proj = self.mlp_down_proj = None
        self.experts = None
        if self.kind == "moe":
            if experts is None:
                raise ValueError(
                    f"layer {self.layer_idx} routes to 256 experts and no expert module was "
                    f"handed in; a dense tuple of them is 3.2 GiB and not what this path is for"
                )
            self.experts = experts
            self.gate = checkpoint.dense_tensor(f"{root}.mlp.gate.weight", torch.float32, self.device)
            bias_key = f"{root}.mlp.gate.e_score_correction_bias"
            if bias_key not in checkpoint:
                raise ValueError(f"layer {self.layer_idx} routes by `noaux_tc` and holds no {bias_key}")
            self.correction_bias = checkpoint.dense_tensor(bias_key, torch.float32, self.device)
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(
                    self,
                    f"mlp_{name}",
                    checkpoint.dense_tensor(f"{root}.mlp.{name}.weight", dtype, self.device),
                )

    @property
    def memory_bytes(self) -> int:
        return self.attention.memory_bytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.input_layernorm,
                self.post_attention_layernorm,
                self.gate,
                self.correction_bias,
                self.mlp_gate_proj,
                self.mlp_up_proj,
                self.mlp_down_proj,
            )
            if tensor is not None
        )

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The reference's router, on the device: `(topk_idx, topk_weight)`."""
        return gate_and_route(
            hidden,
            self.gate,
            self.correction_bias,
            top_k=self.config.num_experts_per_tok,
            n_group=self.config.n_group,
            topk_group=self.config.topk_group,
            norm_topk_prob=self.config.resolved_norm_topk_prob,
            routed_scaling_factor=self.config.resolved_routed_scaling_factor,
            scoring_func=self.config.scoring_func,
            topk_method=self.config.topk_method,
        )[:2]

    def mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "dense":
            return swiglu_mlp(
                hidden,
                self.mlp_gate_proj,
                self.mlp_up_proj,
                self.mlp_down_proj,
                self.config.hidden_act,
            )
        if hidden.shape[0] != 1:
            raise ValueError(
                f"the routed path runs one token: {hidden.shape[0]} rows would draw every "
                f"token's experts one at a time, which is the grouped-prefill kernel's job"
            )
        indices, weights = self.route(hidden)
        out = self.experts.forward(hidden, indices[0], weights[0], layer_id=self.layer_idx)
        return out.to(hidden.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int = 0,
        cache: MimoV2KVCache | None = None,
    ) -> torch.Tensor:
        """`[sequence, hidden]` in, `[sequence, hidden]` out, the reference's two adds."""
        residual = hidden
        normed = rms_norm(hidden, self.input_layernorm, self.config.layernorm_epsilon)
        attended = self.attention.forward(normed, start_pos=start_pos, cache=cache)[
            "attn_out_post_o"
        ]
        hidden = residual + attended

        residual = hidden
        normed = rms_norm(hidden, self.post_attention_layernorm, self.config.layernorm_epsilon)
        return residual + self.mlp(normed)


class MimoV2DeviceModel:
    """The text backbone on a card: embedding, forty-eight layers, norm, head.

    `layers` exists for the same reason it does in the host bridge: a test that diffs one
    layer should not pay for forty-eight, and a machine that cannot hold the whole model
    should still be able to run a slice of it. A truncated stack's logits mean nothing, so
    the model reports which layers it holds rather than pretending.

    A source that can pin itself is asked to, because an unpinned bank stages every draw
    through PyTorch's own pinned ring instead of reading its pages in place. `pin=False`
    says not to, which is what a test that builds many models over one 149.81 GiB mapping
    wants; the result is kept on `pin_result` so a caller can report what the driver said.
    """

    def __init__(
        self,
        checkpoint,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        layers: Sequence[int] | None = None,
        expert_source: MimoV2ExpertSource | None = None,
        slots: int = 2,
        block: int = 1024,
        budget: int = 1 << 25,
        pin: bool | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.config: MimoV2TextConfig = checkpoint.layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.layers_in_model = list(range(self.config.num_hidden_layers))
        wanted = self.layers_in_model if layers is None else [int(i) for i in layers]

        self.embed_tokens = checkpoint.dense_tensor("model.embed_tokens.weight", dtype, self.device)
        self.norm = checkpoint.dense_tensor("model.norm.weight", dtype, self.device)
        head_key = "lm_head.weight"
        if head_key not in checkpoint:
            raise ValueError(
                f"{head_key} is not in the checkpoint and `tie_word_embeddings` is false; an "
                f"untied head is not optional here"
            )
        # The head is the sampler's input and it stays in float32, at 2.5 GiB and about two
        # milliseconds a token. Its weights are bf16 in the checkpoint either way, but a
        # bf16 *output* is quantised to 0.125 at a logit of twenty -- and a near-tie at that
        # distance is what the two-token comparison in the docs runs into.
        self.lm_head = checkpoint.dense_tensor(head_key, torch.float32, self.device)

        routed = [i for i in wanted if self.config.ffn_kind(i) == "moe"]
        self.experts = None
        self.pin_result = None
        if routed:
            if expert_source is None:
                raise ValueError(
                    f"layers {routed[:4]} route to experts and no source was handed in; the "
                    f"bank is `bank.open_expert_bank` and the fallback is `MmapExpertSource`"
                )
            pinner = getattr(expert_source, "pin_if_enabled", None)
            if pin is False:
                pinner = None
            elif pin and pinner is None:
                raise ValueError(
                    "the caller asked for the expert source to be pinned and it has no "
                    "`pin_if_enabled`; only a resident bank can be registered in place"
                )
            if pinner is not None:
                self.pin_result = pinner()
            self.experts = MimoV2DeviceExperts(
                expert_source,
                top_k=self.config.num_experts_per_tok,
                dim=self.config.hidden_size,
                inter_dim=self.config.resolved_moe_intermediate_size,
                device=self.device,
                slots=slots,
            )
        self.layers = [
            MimoV2DeviceLayer(
                checkpoint,
                layer_idx,
                device=self.device,
                dtype=dtype,
                experts=self.experts,
                block=block,
                budget=budget,
            )
            for layer_idx in wanted
        ]
        self.vocab_size = self.embed_tokens.shape[0]

    @property
    def is_complete(self) -> bool:
        """Whether `forward` runs the whole stack, which is what makes its logits mean something."""
        return [layer.layer_idx for layer in self.layers] == self.layers_in_model

    @property
    def memory_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.embed_tokens, self.norm, self.lm_head)
        ) + sum(layer.memory_bytes for layer in self.layers) + (
            self.experts.arena_bytes if self.experts is not None else 0
        )

    def cache(self, capacity: int, *, dtype: torch.dtype | None = None) -> MimoV2KVCache:
        """A cache sized for this model's layers, on this model's device."""
        return MimoV2KVCache(
            self.config,
            capacity,
            [layer.layer_idx for layer in self.layers],
            device=self.device,
            dtype=dtype or self.dtype,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        cache: MimoV2KVCache | None = None,
        final_norm: bool = True,
    ) -> torch.Tensor:
        """Token ids to logits, `[sequence, vocab]`, at the positions `start_pos` onwards."""
        ids = input_ids.reshape(-1).to(self.device)
        hidden = F.embedding(ids, self.embed_tokens)
        for layer in self.layers:
            hidden = layer.forward(hidden, start_pos=start_pos, cache=cache)
        if not final_norm:
            return hidden
        hidden = rms_norm(hidden, self.norm, self.config.layernorm_epsilon)
        return F.linear(hidden.to(self.lm_head.dtype), self.lm_head)

    @torch.no_grad()
    def step(
        self,
        token_id: int,
        *,
        start_pos: int,
        cache: MimoV2KVCache,
    ) -> torch.Tensor:
        """One decode step: one token in, `[1, vocab]` logits out, at position `start_pos`.

        One row, so that `[-1]` is the row and not its last element -- a caller that indexed
        a squeezed `[vocab]` with `[-1]` would get a scalar and feed `argmax` of *that* back
        in, which produces token zero forever and a plausible-looking token a second.
        """
        return self.forward(
            torch.tensor([int(token_id)], dtype=torch.int64), start_pos=start_pos, cache=cache
        )

    @torch.no_grad()
    def greedy(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int,
        cache: MimoV2KVCache | None = None,
    ) -> list[int]:
        """Decode `max_tokens` tokens greedily, one at a time, through the cache.

        One at a time is the only shape this path has: the prompt goes through the expert
        kernel a token at a time too, which is correct and is not a prefill.

        The loop stops at the config's own end-of-turn tokens, and it returns the one it
        stopped on. Turning a finished turn back into the model produces a continuation of
        the *template* rather than of the answer -- `<|im_start|><|im_start|><|im_end|>` and
        so on -- which is a plausible thing to print and not something a caller asked for.
        """
        if not prompt_ids:
            raise ValueError("a prompt with no tokens has no logits to start from")
        ids = [int(token) for token in prompt_ids]
        cache = cache if cache is not None else self.cache(len(ids) + max_tokens)
        logits = None
        for position, token in enumerate(ids):
            logits = self.forward(
                torch.tensor([token], dtype=torch.int64), start_pos=position, cache=cache
            )[-1]
        if self.experts is not None:
            self.experts.drain()
        end = set(self.config.eos_token_ids)
        generated = []
        for step in range(max_tokens):
            token = int(logits.argmax())
            generated.append(token)
            if token in end:
                break
            logits = self.step(token, start_pos=len(ids) + step, cache=cache)[-1]
        if self.experts is not None:
            self.experts.drain()
        return generated
