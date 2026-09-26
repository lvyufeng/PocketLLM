"""Xing4.0-29B-A4B's MoE: 64 routed experts, top-4, and one shared expert.

The architecture is DeepSeek's mixture-of-experts with a sigmoid router, but
three things about it decide the implementation and none of them are the
expert count:

- **The bulk of the checkpoint is here, and the whole of it fits on one card.**
  64 experts x 3584 x 1024 x 3 tensors is 396 MiB per layer in IQ4_NL and
  14.7 GiB across the 38 MoE blocks, so the expert weights are *resident* --
  read once, held in device memory, and indexed by expert.  Every other MoE in
  this repository streams its active experts from a host bank because its
  checkpoint is far larger than a card; this one does not, and an
  implementation that copied per token would pay a PCIe round trip for weights
  that are already where they are needed.

- **The router's selection and its weights are different numbers.**  The
  checkpoint's own `Xing4_0TopkRouter` picks by `sigmoid(logits) +
  e_score_correction_bias` and then weights by the *unbiased* `sigmoid(logits)`,
  normalized over the top-k and scaled by `routed_scaling_factor`.  Reading the
  bias into the weights, or dropping it from the selection, both produce a model
  that runs and is wrong.

- **top-k is four, and four is where float addition stops being reproducible.**
  The grouped kernel this reuses ends in an `atomicAdd` per output element, so
  its result depends on the order the routes' blocks finish in.  Two summands
  have one order; four do not, and the repository has this measured.  So the
  route assembly below aims each route at its *own* row of the kernel's output
  and sums them in slot order afterwards -- the atomics then have one writer
  each and the reduction is a fixed-order sum, with no change to the kernel.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.loader.gguf.iq4_nl import fold_to_runtime_span
from src.loader.gguf.quant_types import IQ4_NL_RUNTIME_SPAN
from src.models.xing4_0.config import Xing4_0Params

__all__ = [
    "DenseExpertStack",
    "GroupedExpertStack",
    "MoEWeights",
    "RoutePlan",
    "RoutedMoE",
    "SwiGLUMLP",
    "plan_routes",
    "swiglu",
]


@dataclass
class MoEWeights:
    """One MoE block's router and its four expert groups.

    The routed groups are either raw IQ4_NL blocks in the grouped kernel's own
    shape -- `(experts, out_dim, blocks_per_row, 144)`, which is what
    `src/loader/gguf/iq4_nl.fold_to_runtime_span` produces -- or dense fp32
    matrices for the host reference.  `w1` is the gate projection, `w3` the up
    projection and `w2` the down one, which is the naming the GGUF and the
    kernels both use; the checkpoint calls them `gate_proj`, `up_proj` and
    `down_proj`.
    """

    router_weight: torch.Tensor  # [experts, hidden]
    router_bias: torch.Tensor  # [experts], the selection-only correction
    w1: object
    w3: object
    w2: object
    shared_gate: object
    shared_up: object
    shared_down: object

    @property
    def is_quantized(self) -> bool:
        return isinstance(self.w1, torch.Tensor) and self.w1.dtype == torch.uint8

    #: GGUF name suffixes, in the order the four expert groups are built.  The
    #: routed trio is stacked over experts and the shared trio is not, which is
    #: why they are listed apart rather than as three names used twice.
    GGUF_ROUTED = (("ffn_gate_exps.weight", "w1"), ("ffn_up_exps.weight", "w3"), ("ffn_down_exps.weight", "w2"))
    GGUF_SHARED = (("ffn_gate_shexp.weight", "shared_gate"), ("ffn_up_shexp.weight", "shared_up"), ("ffn_down_shexp.weight", "shared_down"))

    @classmethod
    def from_gguf(
        cls, loader, prefix: str, device: str = "cuda", *, out_dtype: torch.dtype = torch.float32
    ) -> "MoEWeights":
        """Build one block's MoE weights, resident, from the released GGUF.

        The routed trio arrives as the file's own stacked-expert tensors and is
        folded into the 256-weight row element the grouped kernel indexes rows
        by; the shared trio stays raw blocks behind `QuantizedGGUFLinear`, since
        it is one expert and the dense GEMM already knows how to run it.

        The router is read dense.  Its GGUF dimensions are `(3584, 64)` -- the
        file's fastest-varying-first order, i.e. [hidden, experts] -- and the
        reader's `read_dense` reverses that into the storage shape, so what comes
        back is already `[experts, hidden]`.  Reading it with a transpose as well
        would give a matrix that multiplies, is wrong, and passes every shape
        check that does not know which axis is which.

        `out_dtype` is the width the shared expert's GEMMs emit at, and it
        defaults to fp32 rather than to the fp16 the rest of the file runs at:
        the shared expert is a real expert on the same weights and its
        `silu(gate) * up` reaches values fp16 saturates.  Narrowing it here would
        put an inf into the sum the routed experts are added to.
        """
        from src.components.gguf.quantized_ops import QuantizedGGUFLinear

        def _folded(suffix: str) -> torch.Tensor:
            reference = loader.tensor_ref(f"{prefix}{suffix}")
            blocks, type_name, in_dim = loader.reader_for(reference).read_routed_layer_blocks(
                reference.name
            )
            if type_name != "iq4_nl":
                raise NotImplementedError(
                    f"{reference.name} is {type_name}; the resident expert path folds iq4_nl"
                )
            # `in_dim` is the row width the kernel indexes rows by, so it is the
            # span count and not the weight count that has to be a multiple of 256.
            return fold_to_runtime_span(blocks, int(in_dim)).to(device=device, non_blocking=False).contiguous()

        values: dict[str, object] = {}
        for suffix, key in cls.GGUF_ROUTED:
            values[key] = _folded(suffix)
        for suffix, key in cls.GGUF_SHARED:
            values[key] = QuantizedGGUFLinear(
                loader.read_quant(f"{prefix}{suffix}", "iq4_nl"), out_dtype=out_dtype
            )

        return cls(
            router_weight=loader.read_dense(f"{prefix}ffn_gate_inp.weight").float(),
            router_bias=loader.read_dense(f"{prefix}exp_probs_b.bias").float(),
            w1=values["w1"],
            w3=values["w3"],
            w2=values["w2"],
            shared_gate=values["shared_gate"],
            shared_up=values["shared_up"],
            shared_down=values["shared_down"],
        )

    @classmethod
    def from_hf(cls, tensors: dict[str, torch.Tensor], params: Xing4_0Params, *, dtype: torch.dtype = torch.float32) -> "MoEWeights":
        """The checkpoint's own tensors, dense, for parity against the reference.

        `tensors` is one layer's worth, keyed without the `model.layers.N.`
        prefix -- the same convention `MLAAttentionWeights.from_hf` uses.  The
        expert matrix it builds is 64 x 1024 x 3584 fp32, 0.94 GiB, so this is a
        few-blocks-at-a-time constructor rather than a runtime one.
        """
        experts = int(params.n_routed_experts)
        stacked = {
            key: torch.stack(
                [tensors[f"mlp.experts.{e}.{name}.weight"] for e in range(experts)], dim=0
            ).to(dtype)
            for key, name in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))
        }
        return cls(
            router_weight=tensors["mlp.gate.weight"].float(),
            router_bias=tensors["mlp.gate.e_score_correction_bias"].float(),
            w1=stacked["w1"],
            w3=stacked["w3"],
            w2=stacked["w2"],
            shared_gate=_dense_linear(tensors, "mlp.shared_experts.gate_proj", dtype),
            shared_up=_dense_linear(tensors, "mlp.shared_experts.up_proj", dtype),
            shared_down=_dense_linear(tensors, "mlp.shared_experts.down_proj", dtype),
        )


def _dense_linear(tensors: dict[str, torch.Tensor], name: str, dtype: torch.dtype):
    """A `nn.Linear`-shaped callable over a stored `[out, in]` weight."""

    weight = tensors[f"{name}.weight"].to(dtype).contiguous()

    def linear(x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)

    return linear


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
    """`silu(gate) * up`, with the kernel's optional clamp.

    The clamp is off in this checkpoint -- `swiglu_limit = 0.0` means "no clamp"
    in `gguf_route_swiglu_quantize_hidden_16_kernel` -- but it is written here
    because the grouped kernel applies it *before* the activation and a port that
    applied it only to the up half would differ from the kernel it calls.
    """
    if limit > 0.0:
        up = torch.clamp(up, -limit, limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


class SwiGLUMLP:
    """The checkpoint's `Xing4_0MLP`: gate, up, silu, multiply, down.

    Takes any three callables, so the same class serves the raw-block kernel path
    (whose linears quantize their own input as the kernel requires) and the host
    reference (whose linears are plain matmuls in fp32).
    """

    def __init__(self, gate, up, down, *, swiglu_limit: float = 0.0, out_dtype: torch.dtype = torch.float32):
        self.gate = gate
        self.up = up
        self.down = down
        self.swiglu_limit = float(swiglu_limit)
        self.out_dtype = out_dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = swiglu(self.gate(x).float(), self.up(x).float(), self.swiglu_limit)
        return self.down(hidden.to(self.out_dtype))


@dataclass(frozen=True)
class RoutePlan:
    """A `[tokens, top_k]` route table in the grouped kernel's CSR form.

    The kernel counts routes, not tokens: it is handed `[routes, dim]` of
    activations and writes `[routes, dim]` of outputs, with `seg_starts` naming
    the expert of each route.  So the plan is two index vectors and the
    permutation between them:

    - `order` sorts the routes by expert (`argsort(..., stable=True)`, so equal
      experts keep `(token, slot)` order).  Route `r` of the kernel's call is the
      flat route `order[r]` of the table.
    - `token_of_route` is `flat_token[order]`: the activation row route `r` reads.

    Sorting by expert is what the kernel wants for locality -- a block of routes
    then reads one expert's weights instead of four -- and it is the reason the
    output has to be permuted back rather than written in place.
    """

    seg_starts: torch.Tensor  # [experts + 1] int32
    order: torch.Tensor  # [routes] int64
    token_of_route: torch.Tensor  # [routes] int64
    route_weights: torch.Tensor  # [routes] float32, in kernel order
    n_routes: int

    def gather_input(self, x: torch.Tensor) -> torch.Tensor:
        """`[tokens, dim]` -> `[routes, dim]`, one row per route."""
        return x.index_select(0, self.token_of_route).contiguous()

    def scatter_output(self, y: torch.Tensor, tokens: int, top_k: int, dim: int) -> torch.Tensor:
        """`[routes, dim]` -> `[tokens, dim]`, summed in slot order.

        Both steps are deterministic and neither uses an atomic: `index_copy_`
        writes each row from exactly one source, and the final `sum(dim=1)`
        reduces the top-k in slot order.  Doing this in the kernel's `atomicAdd`
        instead would combine a token's four routes in block-completion order,
        which is a different sum every run -- see the module docstring.
        """
        flat = y.new_zeros(self.n_routes, dim)
        flat.index_copy_(0, self.order, y)
        return flat.view(tokens, top_k, dim).sum(dim=1)


def plan_routes(indices: torch.Tensor, weights: torch.Tensor, n_experts: int) -> RoutePlan:
    """Build the plan for a `[tokens, top_k]` route table."""
    _, top_k = indices.shape
    flat_expert = indices.reshape(-1)
    counts = torch.bincount(flat_expert, minlength=n_experts)
    seg_starts = torch.zeros(n_experts + 1, dtype=torch.int32, device=indices.device)
    torch.cumsum(counts, dim=0, out=seg_starts[1:])
    order = torch.argsort(flat_expert, stable=True)
    flat_token = (
        torch.arange(indices.size(0), device=indices.device)
        .unsqueeze(1)
        .expand_as(indices)
        .reshape(-1)
    )
    return RoutePlan(
        seg_starts=seg_starts,
        order=order,
        token_of_route=flat_token[order].contiguous(),
        route_weights=weights.reshape(-1)[order].contiguous(),
        n_routes=int(flat_expert.numel()),
    )


class GroupedExpertStack:
    """The routed experts, resident, through `gguf_moe_prefill_grouped_forward`.

    One call handles every route in one launch, which is what makes this usable
    at prefill as well as decode: 64 experts x 3 projections is 192 GEMMs a layer
    if each is a launch of its own, and 256 tokens of prefill is 1024 routes.
    """

    def __init__(
        self,
        params: Xing4_0Params,
        weights: MoEWeights,
        cuda_mod,
        *,
        in_dim: int,
        inter_dim: int,
        n_experts: int | None = None,
    ):
        self.params = params
        self.weights = weights
        self.cuda = cuda_mod
        self.in_dim = int(in_dim)
        self.inter_dim = int(inter_dim)
        # How many experts this stack *holds*, which is the config's count in the
        # model and fewer in a test that wants eight of them on a card.
        self.n_experts = int(params.n_routed_experts if n_experts is None else n_experts)
        self.top_k = int(params.n_experts_per_tok)
        # An empty int8 tensor: iq4_nl reads its values from a 16-entry codebook the
        # kernel holds in registers, so it needs no signed grid, and the wrapper's
        # grid check has no iq4_nl branch.
        self._grid = torch.empty(0, dtype=torch.int8, device=weights.w1.device)
        self._type_id = 20

    def run(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        tokens = x.size(0)
        plan = plan_routes(indices, weights, self.n_experts)
        route_index = torch.arange(plan.n_routes, dtype=torch.int64, device=x.device)
        y = self.cuda.gguf_moe_prefill_grouped_forward(
            plan.gather_input(x),
            route_index,
            plan.route_weights,
            plan.seg_starts,
            self.weights.w1,
            self.weights.w3,
            self.weights.w2,
            self.in_dim,
            self._type_id,
            self.in_dim,
            self._type_id,
            self.inter_dim,
            self._type_id,
            self._grid,
            0.0,
        )
        return plan.scatter_output(y, tokens, self.top_k, self.in_dim)


class DenseExpertStack:
    """The same arithmetic without the kernel: the reference's own loop.

    Present because parity needs it -- a test that compares the kernel against
    the kernel proves nothing -- and because 64 experts of 1024 x 3584 in fp32 is
    4.7 GiB a layer, so it is only ever used on a few blocks at a time.  It is
    the checkpoint's `Xing4_0MoE.moe`, with the same `index_add_` and the same
    per-expert loop, which is also what makes it an independent statement of the
    routing rather than a second copy of the grouped path.
    """

    def __init__(self, params: Xing4_0Params, weights: MoEWeights, *, n_experts: int | None = None):
        self.weights = weights
        self.n_experts = int(params.n_routed_experts if n_experts is None else n_experts)
        self.top_k = int(params.n_experts_per_tok)

    def run(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w1, w3, w2 = (t.float() for t in (self.weights.w1, self.weights.w3, self.weights.w2))
        out = torch.zeros(x.size(0), w2.size(-2), dtype=torch.float32, device=x.device)
        xf = x.float()
        for expert in range(self.n_experts):
            # Ascending expert, ascending token within an expert: the same
            # summation order the plan above produces, arrived at the slow way.
            mask = indices == expert
            token_idx, slot_idx = torch.where(mask)
            if token_idx.numel() == 0:
                continue
            rows = xf[token_idx]
            hidden = swiglu(rows @ w1[expert].T, rows @ w3[expert].T)
            weighted = (hidden @ w2[expert].T) * weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, weighted)
        return out


class RoutedMoE:
    """`Xing4_0MoE`: route, run the routed experts, add the shared expert."""

    def __init__(
        self,
        params: Xing4_0Params,
        weights: MoEWeights,
        stack,
        *,
        dtype: torch.dtype = torch.float32,
        swiglu_limit: float = 0.0,
        out_dtype: torch.dtype | None = None,
    ):
        self.params = params
        self.weights = weights
        self.stack = stack
        self.dtype = dtype
        # What this block hands back.  `dtype` unless the caller names something
        # wider, which the model does: the routed sum is fp32 out of the kernel
        # and narrowing it to fp16 here is what turned a 1e5 activation into the
        # inf the whole trunk then carried.
        self.out_dtype = dtype if out_dtype is None else out_dtype
        self.top_k = int(params.n_experts_per_tok)
        self.scaling = float(params.routed_scaling_factor)
        self.norm_topk_prob = bool(params.norm_topk_prob)
        if params.scoring_func != "sigmoid":
            raise NotImplementedError(
                f"Xing4.0's router scores with sigmoid; {params.scoring_func!r} is not implemented"
            )
        # The shared expert's activation product is the same magnitude as a routed
        # expert's, so it is handed to its down projection in fp32 and never in
        # the model's own narrow dtype.
        self.shared = SwiGLUMLP(
            weights.shared_gate,
            weights.shared_up,
            weights.shared_down,
            swiglu_limit=swiglu_limit,
            out_dtype=torch.float32,
        )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The checkpoint's `Xing4_0TopkRouter.forward`, selection and weights apart.

        `sorted=False` is the reference's own argument and is kept: the set of
        experts is what the model defines, and a different order among the chosen
        four is a different summation order rather than a different answer.
        """
        logits = F.linear(x.float(), self.weights.router_weight.float())
        scores = logits.sigmoid()
        _, indices = torch.topk(
            scores + self.weights.router_bias.float(), self.top_k, dim=-1, sorted=False
        )
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.scaling

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        indices, weights = self.route(xf)
        routed = self.stack.run(xf, indices, weights)
        return (routed + self.shared(xf)).reshape(*shape[:-1], routed.size(-1)).to(self.out_dtype)
