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
running every expert of every layer on one rank, and dealing the experts out is what the
`ep` argument is for: on four ranks a token is 275 ms and the copy is a fifth of it.

Three details of the assembly are the model's and are easy to get wrong:

* The embedding is not scaled by `sqrt(hidden)`, and the head is not tied to it. Both
  are the reference's behaviour and both are checked.
* The two attention families want two different masks, and the windowed mask is not a
  prefix: a windowed layer at 256k reads 128 keys and a global layer reads all of them.
  `MimoV2DeviceAttention` takes the bounds instead of a mask, so this file passes a
  position and nothing else -- there is no mask tensor to build at 256k.
* A layer's residual is added in the hidden dtype on both sides. The routed sum leaves
  the kernel in float32 and is rounded once, where the reference rounds it once.

**Across ranks, the experts are dealt and everything else is replicated.** `ep.py` holds
the deal; this file holds the one place the deal becomes an answer, which is `mlp` summing
this rank's share of the routed experts before the residual. The attention, the router,
the embedding, the head and the dense layer are the same on every rank -- so a four-rank
run does four times the attention work to divide the copy by four, and the copy is the
larger half. That is a stage's boundary and not a design: the dense stack is what the
tensor-parallel stage is for, and until it lands the honest description of a four-rank
decode is *the expert traffic, divided*.

A step is one row and a prefill is a chunk, and the two are different paths through the
same layers. The attention does not care: it takes `[sequence, hidden]`, reads the prefix
out of the cache and bounds each row's keys by its position, so a chunk is one call with
the rows attending to each other. The routed experts do care, and `mlp` dispatches on the
row count -- one row is a draw of eight experts and the single-token kernel, many rows are
many draws and `moe_multi_token_fp4_forward`, whose pairs are grouped by expert rather than
by drawing. `prefill` is the chunked entry point; without a chunk band the experts module
refuses a chunk instead of drawing every token's experts one at a time, which is correct and
is not a prefill.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.mimo_v2.config import MimoV2TextConfig
from src.models.mimo_v2.device_attention import MimoV2DeviceAttention, MimoV2KVCache
from src.models.mimo_v2.device_experts import MimoV2DeviceExperts, MimoV2ExpertSource
from src.models.mimo_v2.ep import EpGroup, deal_rule
from src.models.mimo_v2.layers import gate_and_route, rms_norm, swiglu_mlp

__all__ = [
    "MimoV2DeviceLayer",
    "MimoV2DeviceModel",
    "normalise",
]


def normalise(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`layers.rms_norm` on the card, through the release's fused kernel, bit for bit.

    `layers.rms_norm` is the reference and stays the reference: it is six tensor operations -- an
    upcast, a square, a mean, an `rsqrt`, a downcast and a multiply -- because that is readable and
    because the reference has nothing to be fast for. On a card it is a decode token's biggest
    single Python chain after the attention, and 48 layers pay it twice.

    **The weight is passed as `None` and multiplied afterwards, and that is the whole trick.** The
    reference's own docstring is explicit that the cast to the input width happens *before* the
    weight is applied, and `F.rms_norm` applies its weight in float32 and casts once at the end --
    so handing it the weight buys a fused kernel whose rounding point is not the reference's, and
    measured, that is one to two bfloat16 ulps (`test_a_fused_norm_is_the_reference_norm` used to
    hold it to a bound and now holds it to equality). Without a weight there is nothing to round
    late: the fused kernel produces exactly the float32 `x * rsqrt(mean(x^2) + eps)` the reference
    produces before its cast, and the multiply by a bfloat16 weight in bfloat16 is the reference's
    own last step.

    Measured on this box, a `[1, 4096]` bfloat16 row: 27 us against the reference chain's 112, and
    the two are the same tensor. The nine seeds in the test are the evidence; the ones with a
    weight of one are what isolated the cause, since they agreed before the fix and a random weight
    did not. In float32 the two differ in the last bits of the reduction and not in the arithmetic
    -- a few ulps, which is the same freedom every matmul in this model already takes.

    A host caller and a weight of another width from the input get the reference, because a host
    caller is asking for the reference and `F.rms_norm` will not dispatch across widths.
    """
    if not hidden.is_cuda or weight.dtype != hidden.dtype:
        return rms_norm(hidden, weight, eps)
    normalised = torch.nn.functional.rms_norm(
        hidden, (hidden.shape[-1],), None, eps
    )
    return normalised * weight


class MimoV2DeviceLayer:
    """One decoder layer on a card: norm, attention, norm, FFN.

    The FFN is dense on layer 0 and routed on the other forty-seven, and the routed one
    is a *draw*: the router picks eight experts, the shared arena stages those eight,
    and the kernel sums their weighted outputs. The arena is shared across layers --
    forty-seven of them at 102 MiB would be 4.8 GiB of a 22 GiB card for state that is
    read once a layer a token -- which is why `experts` is handed in rather than built
    here.

    `ep` is how the draw's experts are shared out over the ranks. `None` is one rank
    holding the whole draw; a group of more than one makes the kernel's output a summand,
    and the sum happens here because this is where the reference's arithmetic is: the
    routed sum is completed in fp32 and rounded once, on the way into the residual.
    """

    def __init__(
        self,
        checkpoint,
        layer_idx: int,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        experts: MimoV2DeviceExperts | None = None,
        chunk_experts: MimoV2DeviceExperts | None = None,
        ep: EpGroup | None = None,
        block: int = 1024,
        budget: int = 1 << 25,
        tile_budget: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.config: MimoV2TextConfig = checkpoint.layer
        self.layer_idx = int(layer_idx)
        self.device = torch.device(device)
        self.dtype = dtype
        self.ep = ep
        #: The layer's own geometry, whichever way the attention is cut. `self.attention.shape` is
        #: the *share* this rank computes once the attention is split; a caller asking what comes out
        #: of the layer wants this one, and a caller asking which quarter of the projection this rank
        #: holds wants that one. `shards=1` makes them the same object's numbers.
        self.shape = self.config.attention(self.layer_idx)
        self.kind = self.config.ffn_kind(self.layer_idx)

        #: How the attention is cut, answered by the group rather than chosen here: a group that
        #: cannot join the pieces does not cut them (`EpGroup.attention_shards`), so a world of two,
        #: or a caller that injects a collective of its own, keeps the whole attention on every rank
        #: -- which is what this layer did before the split existed. A group that does not cut them
        #: is not handed a `gather` either: the join is the width of the pieces, and applying it to a
        #: piece that is already the whole output would double the tensor.
        shards = 1 if ep is None else ep.attention_shards
        self.attention = MimoV2DeviceAttention(
            checkpoint,
            self.layer_idx,
            self.device,
            dtype,
            block=block,
            budget=budget,
            tile_budget=tile_budget,
            shard=0 if ep is None else ep.attention_shard,
            shards=shards,
            gather=ep.gather if ep is not None and shards > 1 else None,
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
        self._route_ops = None
        self.mlp_gate_proj = self.mlp_up_proj = self.mlp_down_proj = None
        self.experts = None
        #: The module a *chunk* goes through when it is not the module a step goes through.
        #: The two deals are not interchangeable and they are not a preference: `id` partitions
        #: the experts, so a chunk stages its quarter of them, and `sorted` partitions the
        #: drawings, so a chunk reaches every expert and stages all of them four times over.
        #: Only a decode step can be dealt by drawing, so a model that serves both keeps both
        #: arenas -- the second one costs two rows a slot, which is 51 MiB, against a prefill
        #: that would otherwise be refused or four times the bytes.
        self.chunk_experts = chunk_experts
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
            #: The router as one C++ call, or `None` for a config the transcription does not
            #: cover -- in which case `route` takes the reference path, which is the one that
            #: refuses a scoring function or a `topk` method this checkpoint does not use.
            ops = load_cuda_kernel()
            self._route_ops = (
                ops
                if (
                    ops is not None
                    and self.config.scoring_func == "sigmoid"
                    and self.config.topk_method == "noaux_tc"
                )
                else None
            )
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
        """The reference's router, on the device: `(topk_idx, topk_weight)`.

        The call is `layers.gate_and_route`'s arithmetic -- the same ATen operations in the same
        order, which is why the two agree to the bit and not to a tolerance -- reached from C++
        rather than from Python. That is the whole of the difference, and it is most of what the
        router costs, because the router is two dozen small operations on one row and what those
        cost is the asking and not the arithmetic.

        `tests/probe_mimo_v2_router_ab.py` measures one call: 498.5 us from Python against 190.2 us
        from C++, which at forty-seven routed layers is 23.4 ms of a token's host time against 8.9.
        `tests/probe_mimo_v2_host_phases.py --python-router` measures the same thing inside a real
        step, where the region reads 23.0 ms against 9.2. Neither is the token: at sixteen resident
        rows the card is the limit and the step's own host has slack, so the token does not move
        yet. What this buys is the fourteen milliseconds of headroom that the device-side work is
        about to need. See `src/csrc/mimo_decode_ops.cpp` for the transcription -- including the one
        operation it skips and why the released configuration cannot see it -- and
        `tests/test_models_mimo_v2_device_router.py` for the equality.
        """
        if self._route_ops is None:
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
        return self._route_ops.mimo_noaux_tc_route(
            hidden,
            self.gate,
            self.correction_bias,
            self.config.num_experts_per_tok,
            self.config.n_group,
            self.config.topk_group,
            self.config.resolved_norm_topk_prob,
            self.config.resolved_routed_scaling_factor,
        )

    def mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        """The FFN of one row or of a chunk: dense, or a draw's experts, or a chunk's.

        The routed path has two shapes and the row count is the whole of the choice. One row
        is a decode step: one draw of `top_k`, an arena two rows wide and the single-token
        kernel. More than one row is a chunk: every row's own draw, the arena holding this
        rank's share of the layer's experts, and the grouped kernel. They are the same
        arithmetic, so this is a dispatch and not a decision about results.
        """
        if self.kind == "dense":
            return swiglu_mlp(
                hidden,
                self.mlp_gate_proj,
                self.mlp_up_proj,
                self.mlp_down_proj,
                self.config.hidden_act,
            )
        indices, weights = self.route(hidden)
        if hidden.shape[0] == 1:
            out = self.experts.forward(hidden, indices[0], weights[0], layer_id=self.layer_idx)
        else:
            chunk = self.experts if self.chunk_experts is None else self.chunk_experts
            out = chunk.forward_chunk(hidden, indices, weights, layer_id=self.layer_idx)
        # The world's sum, from this rank's share: the kernel's output is a partial as soon as
        # the draw is dealt out, and the reference's own rounding point is here -- once, in
        # float32, before the residual. `EpGroup` refuses to exist without a way to sum.
        if self.ep is not None and self.ep.partial:
            out = self.ep.reduce(out)
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
        normed = normalise(hidden, self.input_layernorm, self.config.layernorm_epsilon)
        attended = self.attention.forward(normed, start_pos=start_pos, cache=cache)[
            "attn_out_post_o"
        ]
        hidden = residual + attended

        residual = hidden
        normed = normalise(hidden, self.post_attention_layernorm, self.config.layernorm_epsilon)
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

    `ep` deals the routed experts over the ranks. `deal` is which shape of deal that is and
    defaults to the environment's, `ep.py`'s `POCKETLLM_MIMO_EXPERT_DEAL`; it is a parameter
    as well as a variable so a test can hold still what a process-wide variable would make
    depend on the order tests ran in.

    `chunk_rows` is the prefill: without it this model is a decode model, and a prompt is fed
    through `greedy` one token at a time. With it, `prefill` runs a chunk of tokens through
    the grouped expert kernel, and the width of the arena is that many of this rank's share of
    the experts. The two paths want different deals -- a chunk needs the experts partitioned,
    which is `id` -- so a model that is asked to prefill and built with `sorted` at a world
    over one is refused by the experts module rather than served slowly.

    `resident_rows` is the decode path's only lever on the bytes, and the bytes are the floor.
    A token copies `top_k / world` experts a layer -- 1198.5 MiB a rank at a world of four --
    and the layers are a chain, so each layer's copy waits on the one before and the hundred
    milliseconds the link needs is a hundred milliseconds of the step no host work can hide.
    Holding the hottest `resident_rows` experts of every routed layer on the card removes their
    copies from the chain; which experts those are is learned as the draws arrive, because the
    device has to be told and every sentinel was computed on the same layer. The cost is
    `resident_rows` experts of arena a layer, and it is the reason this is a parameter: at a
    short context there is room for a dozen a layer and at 262144 there is not.
    """

    def __init__(
        self,
        checkpoint,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        layers: Sequence[int] | None = None,
        expert_source: MimoV2ExpertSource | None = None,
        ep: EpGroup | None = None,
        deal: str | None = None,
        slots: int = 2,
        block: int = 1024,
        budget: int = 1 << 25,
        pin: bool | None = None,
        chunk_rows: int | None = None,
        tile_budget: int | None = None,
        resident_rows: int = 0,
    ) -> None:
        self.checkpoint = checkpoint
        self.config: MimoV2TextConfig = checkpoint.layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.ep = ep
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
        self.chunk_experts = None
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
            routed_common = dict(
                source=expert_source,
                top_k=self.config.num_experts_per_tok,
                dim=self.config.hidden_size,
                inter_dim=self.config.resolved_moe_intermediate_size,
                device=self.device,
                slots=slots,
                world=1 if ep is None else ep.world,
                rank=0 if ep is None else ep.rank,
                n_experts=self.config.n_routed_experts,
            )
            # A chunk's deal when the step's is not the one a chunk can use: `id` partitions the
            # experts and `sorted` partitions the drawings, so a model built for `sorted` decode
            # has no module that can take a chunk at a world over one -- `forward_chunk` refuses
            # rather than staging every expert on every rank.
            #
            # When there are two, the *step* module is built without a band, because a band is a
            # chunk's arena: sixteen rows it would never fill is 408 MiB of a card that at 262144
            # positions has two gigabytes free, and the module that steps holds one draw.
            resolved = deal_rule() if deal is None else str(deal)
            separate_chunk = (
                chunk_rows is not None
                and routed_common["world"] > 1
                and resolved != "id"
            )
            # The step module's own arena width. It is a *decode* module when a chunk has a
            # module of its own to go to -- either because the step never takes a chunk, or
            # because a second arena was built for one -- and a decode module is the only kind
            # that can hold a resident set, since a chunk's arena is its share of the layer's
            # experts rather than a draw's hottest few.
            step_chunk_rows = None if separate_chunk else chunk_rows
            self.experts = MimoV2DeviceExperts(
                deal=deal,
                chunk_rows=step_chunk_rows,
                # The resident region is one block of rows a routed layer in the arena, so the
                # module has to be told how many layers will draw from it -- exactly the routed
                # ones, because a block is claimed by the first draw a layer makes.
                resident_rows=resident_rows if step_chunk_rows is None else 0,
                resident_layers=len(routed) if resident_rows else 1,
                **routed_common,
            )
            self.chunk_experts = None
            if separate_chunk:
                self.chunk_experts = MimoV2DeviceExperts(
                    deal="id", chunk_rows=chunk_rows, **routed_common
                )
        self.layers = [
            MimoV2DeviceLayer(
                checkpoint,
                layer_idx,
                device=self.device,
                dtype=dtype,
                experts=self.experts,
                chunk_experts=self.chunk_experts,
                ep=ep,
                block=block,
                budget=budget,
                tile_budget=tile_budget,
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
        ) + sum(layer.memory_bytes for layer in self.layers) + sum(
            module.arena_bytes
            for module in (self.experts, self.chunk_experts)
            if module is not None
        )

    def cache(self, capacity: int, *, dtype: torch.dtype | None = None) -> MimoV2KVCache:
        """A cache sized for this model's layers, on this model's device.

        It is also where the RoPE tables are sized: `capacity` is the highest position this model
        will be asked for, so it is the one place that knows how long the tables have to be and
        the last one before a token needs them.
        """
        self.share_rope_tables(capacity)
        # The cache holds a *share* of the key and value heads once the attention is cut, so it asks
        # the group the same way the layers do -- the rule is `EpGroup.attention_shards` and is not
        # restated here, because a cache that disagreed with its attention would be off by a quarter
        # of the heads at the first token and would look like a model bug.
        shards = 1 if self.ep is None else self.ep.attention_shards
        return MimoV2KVCache(
            self.config,
            capacity,
            [layer.layer_idx for layer in self.layers],
            device=self.device,
            dtype=dtype or self.dtype,
            shard=0 if self.ep is None else self.ep.attention_shard,
            shards=shards,
        )

    def share_rope_tables(self, capacity: int) -> int:
        """Give every attention layer a cos/sin table for `capacity` positions.

        One table a *family* and not one a layer: the nine global layers share a theta and the
        thirty-nine windowed ones another, so two tables cover the stack. Forty-eight of them
        would be 3.2 GiB at 128k positions against the 268 MiB the two cost, and the tables are
        identical rows of the same two functions.
        """
        tables: dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in self.layers:
            attention = layer.attention
            key = attention.rope_key
            if key not in tables:
                tables[key] = attention.build_rope_table(capacity)
            attention.share_rope_table(tables[key])
        return len(tables)

    # `inference_mode` and not `no_grad`, on all four entry points. They are the same promise about
    # autograd and they are not the same amount of work: a decode step is several thousand eager
    # dispatches, and `inference_mode` drops the version-counter bump and the view tracking on each
    # one. A trivial `torch.add` on this box is 15.9 us under `no_grad`, 9.8 under `inference_mode`
    # and a `matmul` 31.9 against 22.1, and on a real step at sixteen resident rows the arm in
    # `probe_mimo_v2_ablate.py` reads **7.8 ms** of a token. It changes no arithmetic -- the same
    # kernels on the same values -- so the two modes are held to `torch.equal` rather than to a
    # tolerance, and nothing here saves a tensor for a backward pass that does not exist.
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        cache: MimoV2KVCache | None = None,
        final_norm: bool = True,
        rows: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Token ids to logits, `[sequence, vocab]`, at the positions `start_pos` onwards.

        `rows` selects which of the sequence's rows are carried through the final norm and the
        head, and is applied *after* the layers and not before them: a chunk's rows attend to
        each other, so a caller that wants one row's logits still has to run every row through
        the stack. Slicing here rather than at the head's output is the whole of the saving --
        `[2048, 152576]` float32 is 1.16 GiB and one row of it is 0.6 MiB, and the norm is
        per-row, so the row that leaves is the same row to the bit.
        """
        ids = input_ids.reshape(-1).to(self.device)
        hidden = F.embedding(ids, self.embed_tokens)
        for layer in self.layers:
            hidden = layer.forward(hidden, start_pos=start_pos, cache=cache)
        if not final_norm:
            return hidden
        if rows is not None:
            hidden = hidden[list(rows)]
        hidden = normalise(hidden, self.norm, self.config.layernorm_epsilon)
        return F.linear(hidden.to(self.lm_head.dtype), self.lm_head)

    @torch.inference_mode()
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

    @torch.inference_mode()
    def prefill(
        self,
        prompt_ids: Sequence[int],
        *,
        cache: MimoV2KVCache | None = None,
        chunk: int = 512,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """A prompt through the chunked path: `[vocab]` logits for its last row.

        The prompt is one sequence and the cache is the prefix, so this is `forward` in
        chunks rather than a second implementation of it -- but it is `forward` in chunks
        for a reason that is not performance: one call over a whole prompt builds
        `[sequence, vocab]` of logits, which is 61 GB at 256k on a card that has 22, so a
        caller cannot ask for a long prompt's logits and get them. The rows of a chunk
        attend to each other and to the prefix the cache holds, and only the last chunk's
        last row leaves.

        `chunk` is the caller's because it trades host work for memory and nothing else: the
        bytes a chunk moves are the experts a chunk draws, so a chunk wide enough to draw
        most of the layer's experts is amortising the *per-call* work of the layer and not
        the copy, and a chunk too wide for the arena is banded rather than refused. The
        arithmetic is the same at every width.

        `start_pos` is where the prompt starts in the sequence, and it is what a resumed request
        passes: the cache already holds the first `start_pos` positions, so `prompt_ids` is the
        *remainder* and every row of it goes through the stack at its own absolute position --
        which is what the rotation, the attention bounds and the ring's slot arithmetic all read.
        The default is zero, which is the cold prompt this call was written for.
        """
        ids = [int(token) for token in prompt_ids]
        if not ids:
            raise ValueError("a prompt with no tokens has no logits to start from")
        if chunk <= 0:
            raise ValueError(f"a chunk of {chunk} tokens is not a chunk")
        cache = cache if cache is not None else self.cache(len(ids) + 8)
        logits = None
        for start in range(0, len(ids), chunk):
            piece = torch.tensor(ids[start : start + chunk], dtype=torch.int64)
            if start + piece.shape[0] < len(ids):
                # Every chunk but the last leaves through the stack alone. Asking for its logits
                # would build `[chunk, vocab]` -- 1.16 GiB at 2048 rows -- which is the same
                # `[sequence, vocab]` the docstring above refuses for the whole prompt, arriving
                # one chunk at a time and twice over once the head's arithmetic is counted.
                self.forward(
                    piece, start_pos=start_pos + start, cache=cache, final_norm=False
                )
                continue
            logits = self.forward(
                piece, start_pos=start_pos + start, cache=cache, rows=[-1]
            )[-1]
        for module in (self.experts, self.chunk_experts):
            if module is not None:
                module.drain()
        return logits

    @torch.inference_mode()
    def greedy(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int,
        cache: MimoV2KVCache | None = None,
    ) -> list[int]:
        """Decode `max_tokens` tokens greedily, one at a time, through the cache.

        One at a time is this loop's shape: the prompt goes through the expert kernel a token
        at a time too, because `greedy` is the reference-shaped path and a token a time is
        what its arithmetic is. A caller that wants the prompt chunked wants `prefill`, which
        is the same layers over a chunk and is what a serving loop runs.

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
