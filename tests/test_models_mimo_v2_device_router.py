"""The C++ router against the reference it was transcribed from, to the bit.

`layers.gate_and_route` is the definition of MiMo-V2.6's router and `src/csrc/mimo_decode_ops.cpp`
is a transcription of it: the same ATen calls, in the same order, with the same defaults, reached
without a Python frame apiece. That is the entire claim the transcription makes -- it is not a
faster approximation of the router, it *is* the router -- and it is a claim a test can hold it to
exactly, because a path that runs the same kernels on the same arguments cannot have rounded
anything differently.

Why it is worth doing at all: a decode step routes forty-seven times and each routing is two dozen
small operations on one row. `tests/probe_mimo_v2_host_phases.py` puts the router at 23.5 ms of a
token's host time, which at these shapes is the cost of asking rather than of arithmetic.

The shapes below are the release's -- a 4096-wide hidden, 256 experts, the released grouping -- and
the rows are 1, 2 and 9 so that the decode row and a small batch are both covered. The values are
random, which is what makes the test about the arithmetic rather than about a checkpoint.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402
from src.models.mimo_v2.layers import gate_and_route  # noqa: E402

DIM = 4096
N_EXPERTS = 256
TOP_K = 8
SCALING = 2.5

#: `(n_group, topk_group)`. The released config is `(1, 1)` -- every expert in one group, so the
#: group mask keeps all of them and the routing is a plain top-k over corrected scores. The other
#: is the non-degenerate grouping the reference also implements, and it is here because the kernel
#: takes the grouping as an argument and must not be exact for one of them by accident.
GROUPINGS = ((8, 4), (1, 1))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def ops():
    kernel = load_cuda_kernel()
    if kernel is None or not hasattr(kernel, "mimo_noaux_tc_route"):
        pytest.skip("the `cuda_kernel` extension is not built for this interpreter")
    return kernel


def inputs(rows: int, seed: int):
    """A hidden row and the router's gate and correction bias, at the release's shapes.

    The gate is float32 here because that is how a device layer holds it -- the checkpoint stores
    it in bf16 and the upcast to float32 is part of the arithmetic, which is why the kernel takes
    whatever dtype it is handed and upcasts it itself, exactly as the reference does.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(rows, DIM, generator=generator, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(N_EXPERTS, DIM, generator=generator, device="cuda", dtype=torch.float32)
    bias = torch.randn(N_EXPERTS, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    return hidden, gate, bias


def both_ways(hidden, gate, bias, top_k, n_group, topk_group, norm, scaling):
    """The reference's answer and the kernel's, from the same arguments."""
    reference = gate_and_route(
        hidden,
        gate,
        bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm,
        routed_scaling_factor=scaling,
    )[:2]
    got = ops().mimo_noaux_tc_route(
        hidden, gate, bias, top_k, n_group, topk_group, norm, float(scaling)
    )
    return reference, tuple(got)


@needs_cuda
@pytest.mark.parametrize("rows", [1, 2, 9])
@pytest.mark.parametrize("n_group,topk_group", GROUPINGS)
def test_the_kernel_is_the_reference_router_bit_for_bit(rows, n_group, topk_group):
    """A decode row, a pair and a small batch, under both groupings, all `torch.equal`."""
    hidden, gate, bias = inputs(rows, 500 + rows)
    reference, got = both_ways(hidden, gate, bias, TOP_K, n_group, topk_group, True, SCALING)
    assert torch.equal(got[0], reference[0]), "the chosen experts moved"
    assert torch.equal(got[1], reference[1]), "the weights moved"


@needs_cuda
def test_a_batch_row_is_the_row_a_decode_step_would_have_seen():
    """The rows of a batch do not interact -- and where they seem to, the reference does it first.

    A decode step hands over one row of a sequence whose prefill routed many rows at once, so the
    two had better agree. Which experts are chosen does agree exactly. The weights do not, and the
    reason is upstream of this kernel: `F.linear` on four rows is a different GEMM from `F.linear`
    on one row four times, and the last bit of a float32 sum is not the same for both. That is true
    of `layers.gate_and_route` by itself, with no kernel involved, which is what this test pins --
    the kernel's divergence between shapes is the reference's divergence and not an extra one.
    """
    hidden, gate, bias = inputs(4, 909)
    reference, got = both_ways(hidden, gate, bias, TOP_K, 1, 1, True, SCALING)
    assert torch.equal(got[0], reference[0])
    assert torch.equal(got[1], reference[1])
    for row in range(4):
        one = hidden[row : row + 1]
        alone_reference, alone_got = both_ways(one, gate, bias, TOP_K, 1, 1, True, SCALING)
        assert torch.equal(alone_got[0], alone_reference[0])
        assert torch.equal(alone_got[1], alone_reference[1])
        assert torch.equal(alone_got[0], reference[0][row : row + 1]), "the choice moved by shape"
        assert torch.allclose(
            alone_got[1],
            reference[1][row : row + 1],
            rtol=1e-6,
            atol=1e-7,
        ), "a row's weights moved by more than the gemm's own reassociation"


@needs_cuda
def test_a_three_dimensional_hidden_and_a_bfloat16_gate_are_the_reference_too():
    """The reference's own first act is a reshape, and its own upcast is of the checkpoint's gate.

    A caller that hands the kernel the checkpoint's bf16 gate, or a `[1, rows, hidden]` hidden, is
    asking for the reference's answer and not for the answer at bf16 precision.
    """
    hidden, gate, bias = inputs(3, 3131)
    for shaped in (hidden, hidden.unsqueeze(0)):
        for weight in (gate, gate.to(torch.bfloat16)):
            reference, got = both_ways(shaped, weight, bias, TOP_K, 1, 1, True, SCALING)
            assert torch.equal(got[0], reference[0])
            assert torch.equal(got[1], reference[1])


@needs_cuda
@pytest.mark.parametrize("norm,scaling", [(False, 1.0), (True, 1.0), (True, 0.5), (False, 3.0)])
def test_the_unnormalised_and_the_unscaled_router_are_the_reference_too(norm, scaling):
    """`norm_topk_prob` and the scaling factor come from the config, and both are honoured."""
    hidden, gate, bias = inputs(2, 77)
    reference, got = both_ways(hidden, gate, bias, TOP_K, 1, 1, norm, scaling)
    assert torch.equal(got[0], reference[0])
    assert torch.equal(got[1], reference[1])


@needs_cuda
@pytest.mark.parametrize("top_k,n_group,topk_group", [(8, 8, 4), (4, 4, 2), (1, 2, 1)])
def test_another_grouping_or_k_is_the_reference_too(top_k, n_group, topk_group):
    """The grouping is the config's, not this kernel's, so a different one must still be exact.

    `top_k=1` is the degenerate k, where the reference's renormalisation is skipped by its own
    `top_k > 1` guard -- a difference a test that only ever asked for eight could not see.
    """
    hidden, gate, bias = inputs(3, 4242 + n_group)
    reference, got = both_ways(hidden, gate, bias, top_k, n_group, topk_group, True, SCALING)
    assert torch.equal(got[0], reference[0])
    assert torch.equal(got[1], reference[1])


@needs_cuda
def test_the_released_configuration_is_the_one_this_kernel_implements():
    """The grouping the checkpoint routes by, read from its own config when it is present.

    The kernel is only ever dispatched to for `sigmoid` scoring and `noaux_tc` selection -- a layer
    built by `MimoV2DeviceModel` checks both before setting `_route_ops` -- so this is the test
    that says the tests above are exercising the shipped path and not a neighbour of it.
    """
    release = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
    if not os.path.isfile(os.path.join(release, "config.json")):
        pytest.skip(f"no MiMo-V2.6 checkpoint at {release}")
    from src.models.mimo_v2.config import load_config

    config = load_config(os.path.join(release, "config.json")).text
    assert config.scoring_func == "sigmoid"
    assert config.topk_method == "noaux_tc"
    assert 1 <= config.topk_group <= config.n_group
    hidden, gate, bias = inputs(2, 6060)
    reference, got = both_ways(
        hidden,
        gate,
        bias,
        config.num_experts_per_tok,
        config.n_group,
        config.topk_group,
        config.resolved_norm_topk_prob,
        config.resolved_routed_scaling_factor,
    )
    assert torch.equal(got[0], reference[0])
    assert torch.equal(got[1], reference[1])
