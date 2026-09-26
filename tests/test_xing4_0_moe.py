"""Xing4.0-29B-A4B's MoE, against the checkpoint's own router and experts.

The block is DeepSeek's mixture-of-experts with three differences that each get
a test rather than a comment:

- the router selects with a sigmoid score plus a correction bias and weights by
  the *unbiased* score, which is a distinction that survives every shape check;
- the routed experts are IQ4_NL blocks run through the shared grouped kernel, so
  they need a parity test against a dense evaluation of the same weights;
- top-k is four, and four is the smallest count at which the kernel's
  `atomicAdd` reduction stops being reproducible, so the determinism is asserted
  rather than assumed.

The reference for the first is a line-by-line port of `Xing4_0TopkRouter` and
`Xing4_0MoE`, both read from the checkpoint's own `modeling_xing4_0.py`.  The
device tests need a CUDA build and the released GGUF; the host test needs a real
shard.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.mlp import (
    DenseExpertStack,
    GroupedExpertStack,
    MoEWeights,
    RoutePlan,
    RoutedMoE,
    plan_routes,
    swiglu,
)


CHECKPOINT_DIR_ENV = "POCKETLLM_XING4_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2"
GGUF_NAME = "Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
HF_DIR = "Xing4.0-29B-A4B"

#: Layer 2 is the first MoE block (`first_k_dense_replace` is 2), and its whole
#: expert stack is 396 MiB in IQ4_NL.
MOE_LAYER = 2
GGUF_PREFIX = f"blk.{MOE_LAYER}."
#: How many experts the parity test decodes to fp32.  Each is 11 MiB non-resident
#: and 44 MiB dense, against 0.9 GiB for all 64.
PARITY_EXPERTS = 8


def _gguf_path() -> Path:
    path = Path(os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT)) / GGUF_NAME
    if not path.exists():
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {path}")
    return path


def _hf_dir() -> Path:
    path = Path(os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT)) / HF_DIR
    if not (path / "model.safetensors.index.json").exists():
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {path}")
    return path


def _params() -> Xing4_0Params:
    return Xing4_0Params.from_json(_hf_dir() / "config.json")


def _cuda_mod():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.kernels.cuda_loader import load_cuda_kernel

    module = load_cuda_kernel()
    if module is None or not hasattr(module, "gguf_moe_prefill_grouped_forward"):
        pytest.skip("grouped MoE kernel is not built for this interpreter")
    return module


# --------------------------------------------------------------------------- #
# The router, against a port of the checkpoint's own
# --------------------------------------------------------------------------- #


def _reference_router(hidden, weight, bias, top_k, *, norm_topk_prob, scaling):
    """`Xing4_0TopkRouter.forward`, transcribed.

    The two numbers that matter are both here: the top-k is chosen on
    `scores + bias` and weighted by `scores` alone, and the normalization
    divides by the sum of the *gathered* scores.
    """
    router_logits = F.linear(hidden.float(), weight.float())
    scores = router_logits.sigmoid()
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return router_logits, weights * scaling, indices


def _real_router(tensors: dict[str, torch.Tensor], params: Xing4_0Params) -> MoEWeights:
    """Only the router half of a `MoEWeights`; the expert half is never touched."""
    return MoEWeights(
        router_weight=tensors["mlp.gate.weight"].float(),
        router_bias=tensors["mlp.gate.e_score_correction_bias"].float(),
        w1=None,
        w3=None,
        w2=None,
        shared_gate=None,
        shared_up=None,
        shared_down=None,
    )


def _dense_moe(params: Xing4_0Params, weights: MoEWeights) -> RoutedMoE:
    """A `RoutedMoE` whose routed half is the dense loop, for host-only tests."""
    moe = RoutedMoE(params, weights, DenseExpertStack(params, weights), dtype=torch.float32)
    return moe


def test_the_router_matches_the_checkpoint() -> None:
    """Selection on `scores + bias`, weighting on `scores`.

    A port that used the biased scores for the weights would pick the same
    experts and produce different weights, so the indices alone are not the
    assertion -- both are compared, and the weights are the strict one.
    """
    tensors = _layer_tensors(["mlp.gate.weight", "mlp.gate.e_score_correction_bias"])
    params = _params()
    weights = _real_router(tensors, params)
    # The reference's own MoE only needs the router to be exercised, but building
    # a `RoutedMoE` does too, so the object under test is the class the model uses.
    moe = _dense_moe(params, weights)

    x = torch.randn(37, params.hidden_size, generator=torch.Generator().manual_seed(0))
    indices, got = moe.route(x)
    _, want, want_indices = _reference_router(
        x,
        weights.router_weight,
        weights.router_bias,
        params.n_experts_per_tok,
        norm_topk_prob=params.norm_topk_prob,
        scaling=params.routed_scaling_factor,
    )
    # `topk(..., sorted=False)` does not promise a slot order, so compare the
    # route *sets* and the weight multiset rather than the slots.
    assert torch.equal(indices.sort(dim=-1).values, want_indices.sort(dim=-1).values)
    assert torch.allclose(got.sort(dim=-1).values, want.sort(dim=-1).values, rtol=1e-6, atol=1e-7)
    # The bias really is doing something on this layer, or the test above would
    # pass for a port that ignored it.
    biased = F.linear(x, weights.router_weight).sigmoid() + weights.router_bias
    plain = F.linear(x, weights.router_weight).sigmoid()
    assert not torch.equal(
        biased.topk(params.n_experts_per_tok, dim=-1).indices.sort(dim=-1).values,
        plain.topk(params.n_experts_per_tok, dim=-1).indices.sort(dim=-1).values,
    )


def test_the_weights_are_normalized_then_scaled() -> None:
    """`norm_topk_prob` divides and `routed_scaling_factor` multiplies, in that order.

    The factor is 2.0 for this checkpoint, so a port that applied the scale
    before the normalization would be right here and wrong on any checkpoint
    where the normalized weights do not sum to one -- which is all of them.
    """
    params = _params()
    tensors = _layer_tensors(["mlp.gate.weight", "mlp.gate.e_score_correction_bias"])
    weights = _real_router(tensors, params)
    moe = _dense_moe(params, weights)
    x = torch.randn(16, params.hidden_size, generator=torch.Generator().manual_seed(3))
    _, got = moe.route(x)
    assert torch.allclose(got.sum(dim=-1), torch.full((16,), params.routed_scaling_factor))
    assert params.routed_scaling_factor != 1.0


@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 6, 8])
def test_the_reference_router_port_is_the_reference(top_k: int) -> None:
    """The port is checked against itself at several top-k, so the tests above are
    a comparison and not a tautology: a weight bug in both would still show here
    as a failure against the checkpoint's `Xing4_0TopkRouter` semantics asserted
    directly -- the biased score selects and the unbiased one weights."""
    scores = torch.rand(5, 16, generator=torch.Generator().manual_seed(top_k))
    bias = torch.rand(16, generator=torch.Generator().manual_seed(top_k + 100)) - 0.5
    _, weights, indices = _reference_router(
        scores, torch.eye(16), bias, top_k, norm_topk_prob=True, scaling=1.0
    )
    assert weights.sum(dim=-1).allclose(torch.ones(5))
    # The router weighs by `sigmoid(logits)`, so that -- and not the pre-sigmoid
    # score -- is what the normalized weights have to be proportional to.
    biased_scores = scores.sigmoid()
    for row in range(5):
        chosen = indices[row].tolist()
        assert len(set(chosen)) == top_k
        expected = biased_scores[row][chosen]
        assert torch.allclose(weights[row].sort().values, (expected / expected.sum()).sort().values)
        # And the selection used the bias: when the bias reorders anything, the
        # chosen set beats the unbiased top-k on the biased score.
        unbiased = biased_scores[row].topk(top_k).indices.tolist()
        if set(unbiased) != set(chosen):
            assert (biased_scores[row][chosen] + bias[chosen]).min() >= (
                biased_scores[row][unbiased] + bias[unbiased]
            ).min()


# --------------------------------------------------------------------------- #
# The route plan, and the determinism it exists for
# --------------------------------------------------------------------------- #


def _plan_fixture(tokens: int = 9, experts: int = 5, top_k: int = 4, seed: int = 0):
    """A route table with `top_k` *distinct* experts per token.

    Distinct because that is what `topk` guarantees and what the grouped kernel's
    segment layout assumes; a fixture that repeated an expert would test a shape
    the model cannot produce.  When `top_k` exceeds `experts` the fixture is
    rejected rather than silently truncated, which is why the parametrisation
    below keeps `top_k <= experts`.
    """
    assert top_k <= experts, "a route table cannot name more distinct experts than exist"
    generator = torch.Generator().manual_seed(seed)
    indices = torch.stack(
        [torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)]
    ).to(torch.int64)
    weights = torch.rand(tokens, top_k, generator=generator)
    return indices, weights, experts


@pytest.mark.parametrize("tokens,experts,top_k", [(9, 5, 4), (1, 64, 4), (64, 64, 4), (17, 8, 8)])
def test_the_plan_is_a_csr_over_the_route_table(tokens: int, experts: int, top_k: int) -> None:
    """`seg_starts` partitions the routes, and the two index vectors are the table.

    A CSR that is off by one still runs -- the kernel reads whatever expert the
    segment says -- so the assertion is against the route table itself: every
    route's expert, reconstructed from the segments, must be the route's own.
    """
    indices, weights, experts = _plan_fixture(tokens, experts, top_k)
    plan = plan_routes(indices, weights, experts)
    flat_expert = indices.reshape(-1)

    assert int(plan.seg_starts[-1]) == plan.n_routes == tokens * top_k
    assert torch.equal(plan.seg_starts[1:] - plan.seg_starts[:-1], torch.bincount(flat_expert, minlength=experts))
    assert sorted(plan.order.tolist()) == list(range(tokens * top_k))
    # The segments, and the kernel's own lookup, must name the route's expert.
    recovered = torch.empty(plan.n_routes, dtype=torch.int64)
    for expert in range(experts):
        recovered[plan.seg_starts[expert] : plan.seg_starts[expert + 1]] = expert
    assert torch.equal(recovered, flat_expert[plan.order])
    # And the activation row must be the token that owns the route.
    flat_token = torch.arange(tokens).unsqueeze(1).expand_as(indices).reshape(-1)
    assert torch.equal(plan.token_of_route, flat_token[plan.order])
    # The routes arrive sorted by expert, which is what the kernel wants.
    assert torch.equal(flat_expert[plan.order], flat_expert[plan.order].sort().values)


def test_the_scatter_is_the_inverse_of_the_order() -> None:
    """`plan.order` maps kernel rows back to route-table rows, invertibly."""
    indices, weights, experts = _plan_fixture()
    plan = plan_routes(indices, weights, experts)
    tokens, top_k = indices.shape
    # Label each route with the token it belongs to and its slot in the table, both
    # read off `order` and independently of the code under test: a flat route index
    # is `token * top_k + slot`, so `order[r] // top_k` and `order[r] % top_k` are
    # the route's token and slot.  If the scatter put a route back anywhere but its
    # own cell, the token column would mix and this sum would not factor.
    label = (plan.order // top_k).float() * 10.0 + (plan.order % top_k).float()
    out = plan.scatter_output(label.unsqueeze(1).expand(plan.n_routes, 2), tokens, top_k, 2)
    want_token = (torch.arange(tokens).float() * 10.0 * top_k + sum(range(top_k)))
    assert torch.allclose(out.cpu(), want_token.unsqueeze(1).expand(tokens, 2))


def test_the_gather_reads_the_token_that_owns_the_route() -> None:
    """`gather_input` is a row selection, not a permutation of values."""
    indices, weights, experts = _plan_fixture()
    plan = plan_routes(indices, weights, experts)
    x = torch.arange(9, dtype=torch.float32).unsqueeze(1).expand(9, 4)
    gathered = plan.gather_input(x)
    assert gathered.shape == (plan.n_routes, 4)
    for route in range(plan.n_routes):
        assert torch.all(gathered[route] == float(plan.token_of_route[route]))


# --------------------------------------------------------------------------- #
# The routed experts: the kernel against a dense evaluation of the same weights
# --------------------------------------------------------------------------- #


def _quantized_moe(params: Xing4_0Params, experts: int) -> MoEWeights:
    from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader

    with GGUFQuantizedTensorLoader(str(_gguf_path()), device="cuda") as loader:
        weights = MoEWeights.from_gguf(loader, GGUF_PREFIX)
    # Trim the routed stack to the experts this test decodes.  The router is kept
    # whole; only the routed half is indexed by expert id.
    weights = MoEWeights(
        router_weight=weights.router_weight,
        router_bias=weights.router_bias,
        w1=weights.w1[:experts].contiguous(),
        w3=weights.w3[:experts].contiguous(),
        w2=weights.w2[:experts].contiguous(),
        shared_gate=weights.shared_gate,
        shared_up=weights.shared_up,
        shared_down=weights.shared_down,
    )
    return weights


def _decoded_dense(weights: MoEWeights, in_dim: int, inter_dim: int) -> dict[str, torch.Tensor]:
    """The same blocks, decoded to fp32 on the host, in the dense stacks' layout."""
    from src.loader.gguf import iq4_nl

    out = {}
    for key, row_elems in (("w1", in_dim), ("w3", in_dim), ("w2", inter_dim)):
        blocks = weights.__dict__[key].cpu().numpy()
        experts = blocks.shape[0]
        # The folded form is (experts, out_dim, spans, 144); decoding it needs the
        # native 18-byte blocks back, which is the same bytes reshaped.
        native = blocks.reshape(experts, blocks.shape[1], -1, iq4_nl.IQ4_NL_BLOCK_BYTES)
        decoded = iq4_nl.dequantize_blocks(native).reshape(experts, blocks.shape[1], -1)
        out[key] = torch.from_numpy(decoded.copy()).float()
    return out


def test_the_grouped_kernel_matches_a_dense_evaluation() -> None:
    """The run path, against the same weights evaluated by torch.

    Two things this catches that a kernel-vs-kernel test cannot: a fold that
    regroups bytes correctly but in the wrong order, and a route plan that feeds
    the kernel the wrong activation row -- both of which survive the block-dot
    tests in `test_gguf_iq4nl_kernel.py`.
    """
    cuda_mod = _cuda_mod()
    params = _params()
    in_dim, inter_dim = params.hidden_size, params.moe_intermediate_size
    quantized = _quantized_moe(params, PARITY_EXPERTS)
    dense_weights = MoEWeights(
        router_weight=quantized.router_weight,
        router_bias=quantized.router_bias,
        **_decoded_dense(quantized, in_dim, inter_dim),
        shared_gate=None,
        shared_up=None,
        shared_down=None,
    )

    generator = torch.Generator().manual_seed(11)
    indices = torch.stack(
        [torch.randperm(PARITY_EXPERTS, generator=generator)[: params.n_experts_per_tok] for _ in range(6)]
    ).to(torch.int64)
    route_weights = torch.rand(6, params.n_experts_per_tok, generator=generator)

    # The kernel quantizes its activation to fp16, so the dense side is handed the
    # same rounded values; that and the fp32 accumulation order are the tolerance.
    x = torch.randn(6, in_dim, generator=generator) * 0.1
    x_half = x.half().float()

    grouped = GroupedExpertStack(
        params, quantized, cuda_mod, in_dim=in_dim, inter_dim=inter_dim, n_experts=PARITY_EXPERTS
    )
    dense = DenseExpertStack(params, dense_weights, n_experts=PARITY_EXPERTS)

    got = grouped.run(x_half.cuda(), indices.cuda(), route_weights.cuda()).cpu()
    want = dense.run(x_half, indices, route_weights)

    scale = want.abs().max().clamp(min=1e-6)
    assert (got - want).abs().max() / scale < 2e-2, (float((got - want).abs().max()), float(scale))
    # And a wrong route plan would show up as a much larger difference, so the
    # output is not simply dominated by the shared expert or by zero.
    assert want.abs().max() > 1e-3


def test_the_grouped_kernel_is_bit_reproducible_at_top_four() -> None:
    """Four routes per token summed by `atomicAdd` is the repository's known
    divergence, so this is the assertion that the plan removed it.

    Five repeats, bit-for-bit.  Everything upstream is fixed -- the kernel's
    launch order is a function of the inputs and nothing else -- so any run that
    differs means a route's contribution was summed in a different order.
    """
    cuda_mod = _cuda_mod()
    params = _params()
    in_dim, inter_dim = params.hidden_size, params.moe_intermediate_size
    weights = _quantized_moe(params, PARITY_EXPERTS)
    stack = GroupedExpertStack(
        params, weights, cuda_mod, in_dim=in_dim, inter_dim=inter_dim, n_experts=PARITY_EXPERTS
    )
    generator = torch.Generator().manual_seed(5)
    indices = torch.stack(
        [torch.randperm(PARITY_EXPERTS, generator=generator)[: params.n_experts_per_tok] for _ in range(4)]
    ).to(torch.int64)
    route_weights = torch.rand(4, params.n_experts_per_tok, generator=generator)
    x = (torch.randn(4, in_dim, generator=generator) * 0.1).half().cuda()

    first = stack.run(x, indices.cuda(), route_weights.cuda())
    for _ in range(4):
        assert torch.equal(first, stack.run(x, indices.cuda(), route_weights.cuda()))


def test_a_permuted_route_table_gives_the_same_answer() -> None:
    """The same expert assignments in a different slot order, same bits.

    This is the property that makes the plan worth its permutation: the kernel is
    handed the routes in an order derived from the *experts* rather than from the
    table, so a table whose slots are permuted produces the same route order and
    therefore the same summand order.  Without the plan, this is exactly the case
    `atomicAdd` makes irreproducible.
    """
    cuda_mod = _cuda_mod()
    params = _params()
    in_dim, inter_dim = params.hidden_size, params.moe_intermediate_size
    weights = _quantized_moe(params, PARITY_EXPERTS)
    stack = GroupedExpertStack(
        params, weights, cuda_mod, in_dim=in_dim, inter_dim=inter_dim, n_experts=PARITY_EXPERTS
    )
    generator = torch.Generator().manual_seed(9)
    indices = torch.stack(
        [torch.randperm(PARITY_EXPERTS, generator=generator)[: params.n_experts_per_tok] for _ in range(4)]
    ).to(torch.int64)
    route_weights = torch.rand(4, params.n_experts_per_tok, generator=generator)
    x = (torch.randn(4, in_dim, generator=generator) * 0.1).half().cuda()

    base = stack.run(x, indices.cuda(), route_weights.cuda())
    # Reverse the slots of every token.  Each token keeps its four experts and the
    # weights that go with them, so the answer is the same number.
    flipped = indices.flip(-1)
    flipped_weights = route_weights.flip(-1)
    other = stack.run(x, flipped.cuda(), flipped_weights.cuda())
    # The sum's terms are the same but its order is not, so this is a fp32
    # associativity gap and not an exact statement; what it pins is that the gap
    # is at the last-bit scale and not the atomicAdd scale.
    gap = (base - other).abs().max() / base.abs().max()
    assert gap < 1e-5, float(gap)


def test_the_shared_expert_is_added_after_the_routed_ones() -> None:
    """`moe(...) + shared_experts(residuals)`, with both on the same input.

    The shared expert sees the block's normalized input, not the routed output --
    the checkpoint passes `residuals`, which is `hidden_states` unmodified.  A
    port that chained them would run and be wrong, and the two differ by exactly
    the shared expert's output here.
    """
    params = _params()
    generator = torch.Generator().manual_seed(0)
    hidden, inter, experts = 32, 8, 3
    routed = {key: torch.randn(experts, inter, hidden, generator=generator) * 0.1 for key in ("w1", "w3")}
    routed["w2"] = torch.randn(experts, hidden, inter, generator=generator) * 0.1
    shared = {
        "gate": torch.randn(inter, hidden, generator=generator) * 0.1,
        "up": torch.randn(inter, hidden, generator=generator) * 0.1,
        "down": torch.randn(hidden, inter, generator=generator) * 0.1,
    }
    weights = MoEWeights(
        router_weight=torch.randn(experts, hidden, generator=generator),
        router_bias=torch.randn(experts, generator=generator),
        w1=routed["w1"],
        w3=routed["w3"],
        w2=routed["w2"],
        shared_gate=_linear(shared["gate"]),
        shared_up=_linear(shared["up"]),
        shared_down=_linear(shared["down"]),
    )
    small = replace(
        params,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=experts,
        n_experts_per_tok=min(params.n_experts_per_tok, experts),
    )
    moe = RoutedMoE(small, weights, DenseExpertStack(small, weights), dtype=torch.float32)
    x = torch.randn(5, hidden, generator=generator)
    indices, route_weights = moe.route(x)
    want = moe.stack.run(x, indices, route_weights) + F.linear(
        swiglu(F.linear(x, shared["gate"]), F.linear(x, shared["up"])), shared["down"]
    )
    assert torch.allclose(moe(x), want, rtol=1e-5, atol=1e-5)
    # The shared expert is not negligible, so a port that dropped it fails here.
    assert torch.abs(moe(x) - moe.stack.run(x, indices, route_weights)).max() > 1e-3


def _linear(weight: torch.Tensor):
    def linear(x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)

    return linear


# --------------------------------------------------------------------------- #
# Reading the real weights
# --------------------------------------------------------------------------- #


def test_the_gguf_block_has_the_shape_the_kernel_wants() -> None:
    """The folded routed stack, and a router that is not still [hidden, experts].

    The GGUF stores the router transposed relative to the checkpoint's linear, so
    reading it without the transpose gives a matrix that multiplies and is wrong
    -- and shape-checks pass, because 64 x 3584 and 3584 x 64 are both square
    against their own inputs in the wrong order only at the *end*.
    """
    from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader

    params = _params()
    with GGUFQuantizedTensorLoader(str(_gguf_path()), device="cuda") as loader:
        weights = MoEWeights.from_gguf(loader, GGUF_PREFIX)
    assert tuple(weights.router_weight.shape) == (params.n_routed_experts, params.hidden_size)
    assert tuple(weights.router_bias.shape) == (params.n_routed_experts,)
    assert weights.is_quantized
    assert tuple(weights.w1.shape) == (64, params.moe_intermediate_size, params.hidden_size // 256, 144)
    assert tuple(weights.w3.shape) == tuple(weights.w1.shape)
    assert tuple(weights.w2.shape) == (64, params.hidden_size, params.moe_intermediate_size // 256, 144)
    # The router is fp32 in the file because the quantizer excludes it, and the
    # bias is the CHECKPOINT's rather than a zero buffer -- a release that shipped
    # zeros would leave the selection tests above passing and hide a real
    # difference here.  It is small by construction (a correction, not a score),
    # so the assertion is "every expert has one" and not a magnitude.
    assert weights.router_weight.dtype == torch.float32
    assert int((weights.router_bias != 0).sum()) == params.n_routed_experts


# --------------------------------------------------------------------------- #
# Shard access, for the host-only tests
# --------------------------------------------------------------------------- #


def _layer_tensors(names: list[str]) -> dict[str, torch.Tensor]:
    """Read the named layer-2 tensors out of whichever shards hold them."""
    root = _hf_dir()
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    safetensors = pytest.importorskip("safetensors.torch")
    wanted = {f"model.layers.{MOE_LAYER}.{name}": name for name in names}
    shards = {index["weight_map"][key] for key in wanted}
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(shards):
        with safetensors.safe_open(str(root / shard), framework="pt") as handle:
            for key, short in wanted.items():
                if index["weight_map"][key] == shard:
                    out[short] = handle.get_tensor(key)
    missing = set(names) - set(out)
    assert not missing, missing
    return out
