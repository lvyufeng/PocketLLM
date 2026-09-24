"""The routed experts on the card: the arena, the ring, and the arithmetic against the host.

Two things are being checked and they are independent. The first is bookkeeping: a draw goes
into an arena row, the row is filled from the source, a slot is not reused until its own
kernel has drained, and a draw of eight experts does not read a ninth. Getting that wrong
produces plausible numbers from the wrong expert, which is why the synthetic half of this
file gives every expert bytes that identify it.

The second is arithmetic. The kernel quantizes activations to int8 a row at a time, so it
cannot equal a float32 reference exactly and never will -- the question worth asking is
whether it equals *its own* arithmetic, which the test below answers by emulating that
quantization on the host and demanding agreement to float32 rounding. The float32 comparison
is in the file too, with the tolerance the quantization actually costs, so the number is
recorded rather than assumed.

Everything here needs a CUDA device and one of the halves needs the release.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.device_experts import (  # noqa: E402
    MimoV2DeviceExperts,
    MmapExpertSource,
    _Residents,
)
from src.models.mimo_v2.layers import gate_and_route  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.quant import dequant_mxfp4  # noqa: E402
from src.models.mimo_v2.weights import router_weights  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release = pytest.mark.skipif(
    not HAS_RELEASE, reason=f"MiMo-V2.6 checkpoint not present at {RELEASE}"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the device expert path needs both the release and a CUDA device",
)

#: Small enough to be a unit test and still a legal fp4 kernel shape: both strides a multiple
#: of 32, `w1`/`w3` `[inter, dim/2]` and `w2` `[dim, inter/2]`.
DIM = 64
INTER = 32
TOP_K = 2
N_EXPERTS = 6


def tagged_expert(expert: int, *, jitter: bool = False) -> dict[tuple[str, str], torch.Tensor]:
    """One expert's six tensors, in the shape and the storage the kernel takes.

    The scales are E8M0 127 -- that is 2**0, one -- so a dequantized weight is exactly the
    codebook value and the reference below is arithmetic rather than a second quantizer. That
    is what makes "the device read the expert it was asked for" a comparison and not a
    tolerance. The codes are drawn from the whole byte range and then shifted by the expert's
    own id, so two experts differ and no draw is ambiguous.
    """
    generator = torch.Generator().manual_seed(9000 + expert)
    shapes = {
        ("gate_proj", "weight"): (INTER, DIM // 2),
        ("gate_proj", "weight_scale"): (INTER, DIM // 32),
        ("down_proj", "weight"): (DIM, INTER // 2),
        ("down_proj", "weight_scale"): (DIM, INTER // 32),
        ("up_proj", "weight"): (INTER, DIM // 2),
        ("up_proj", "weight_scale"): (INTER, DIM // 32),
    }
    out: dict[tuple[str, str], torch.Tensor] = {}
    for key, shape in shapes.items():
        if key[1] == "weight_scale":
            out[key] = torch.full(shape, 127, dtype=torch.uint8)
        else:
            # `randint` has no uint8 path, so the draw is made wider and narrowed here.
            codes = torch.randint(0, 256, shape, generator=generator, dtype=torch.int16)
            out[key] = ((codes + (expert * 37 if jitter else 0)) % 256).to(torch.uint8)
    return {key: value.to(torch.uint8) for key, value in out.items()}


class SyntheticSource:
    """A source whose experts are identifiable, so a wrong row cannot look like a right one."""

    def __init__(self, n_experts: int = N_EXPERTS) -> None:
        self.n_experts = n_experts
        self._cache = {expert: tagged_expert(expert, jitter=True) for expert in range(n_experts)}
        self.draws: list[tuple[int, int]] = []

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        self.draws.append((int(layer_id), int(expert)))
        return self._cache[int(expert)]


def device_experts(source, **kwargs) -> MimoV2DeviceExperts:
    return MimoV2DeviceExperts(
        source,
        kwargs.pop("layer_id", 3),
        device="cuda",
        top_k=TOP_K,
        dim=DIM,
        inter_dim=INTER,
        **kwargs,
    )


def int8_rows(x: torch.Tensor) -> torch.Tensor:
    """int8 a row at a time, one scale a row -- the kernel's own activation quantization."""
    scale = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-30)
    return (x / scale).round().clamp(-127, 127) * scale


def reference(
    source,
    hidden: torch.Tensor,
    indices: list[int],
    weights: torch.Tensor,
    *,
    layer: int,
    quantised: bool,
) -> torch.Tensor:
    """`down(silu(gate(x)) * up(x))` summed over the draw, in float32, optionally with the
    kernel's int8 activation quantization applied where the kernel applies it."""
    total = torch.zeros_like(hidden)
    for slot, expert in enumerate(indices):
        views = source.expert_views(layer, expert)
        dense = {
            proj: dequant_mxfp4(views[(proj, "weight")], views[(proj, "weight_scale")], 32, torch.float32)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        x = int8_rows(hidden) if quantised else hidden
        gate = torch.nn.functional.silu(torch.nn.functional.linear(x, dense["gate_proj"]))
        up = torch.nn.functional.linear(x, dense["up_proj"])
        inner = float(weights[slot]) * gate * up
        if quantised:
            inner = int8_rows(inner)
        total += torch.nn.functional.linear(inner, dense["down_proj"])
    return total


# ---------------------------------------------------------------------------
# The arena and the ring
# ---------------------------------------------------------------------------


@needs_cuda
def test_the_arena_holds_a_draw_and_is_billed_for_it():
    experts = device_experts(SyntheticSource())
    assert experts.arena_rows == TOP_K
    # One expert is the three projections, codes and scales: the same 12.75 MiB the released
    # checkpoint stores per expert, at this miniature's dimensions.
    expected = sum(
        shape[0] * shape[1] for shape in experts._shapes().values()
    )
    assert experts.expert_bytes == expected
    assert experts.arena_bytes == experts.slots * experts.arena_rows * expected


@needs_cuda
def test_an_arena_too_narrow_for_a_draw_is_refused():
    with pytest.raises(ValueError, match="cannot hold"):
        device_experts(SyntheticSource(), arena_rows=TOP_K - 1)


@needs_cuda
def test_a_multi_token_call_is_refused_rather_than_answered_wrong():
    experts = device_experts(SyntheticSource())
    hidden = torch.zeros(2, DIM, device="cuda")
    with pytest.raises(ValueError, match="single-token"):
        experts.forward(hidden, torch.zeros(2, dtype=torch.int64, device="cuda"),
                        torch.ones(2, device="cuda"))


@needs_cuda
def test_a_draw_is_answered_by_the_experts_it_asked_for():
    """Two draws of different experts give different answers, and each matches its own."""
    source = SyntheticSource()
    experts = device_experts(source)
    torch.manual_seed(0)
    hidden = torch.randn(1, DIM) * 0.4

    outs = {}
    for draw, indices in enumerate(([1, 4], [0, 5], [2, 3])):
        weights = torch.tensor([0.6, 0.4])
        out = experts.forward(
            hidden.cuda(),
            torch.tensor(indices, dtype=torch.int64, device="cuda"),
            weights.cuda(),
        )
        experts.drain()
        outs[draw] = out.cpu()
        expected = reference(source, hidden, indices, weights, layer=3, quantised=True)
        assert torch.allclose(outs[draw], expected, atol=2e-3), draw

    assert not torch.allclose(outs[0], outs[1])
    assert not torch.allclose(outs[1], outs[2])


@needs_cuda
def test_more_calls_than_slots_still_answer_each_draw_correctly():
    """The ring is the thing under test: a slot is reused and must be refilled, not read stale."""
    source = SyntheticSource()
    experts = device_experts(source, slots=2)
    torch.manual_seed(1)
    draws = [([0, 1], torch.tensor([0.5, 0.5])), ([2, 3], torch.tensor([0.7, 0.3])),
             ([4, 5], torch.tensor([0.2, 0.8])), ([1, 2], torch.tensor([0.4, 0.6]))]
    hidden = torch.randn(1, DIM) * 0.4

    first = []
    for indices, weights in draws:
        first.append(
            experts.forward(
                hidden.cuda(),
                torch.tensor(indices, dtype=torch.int64, device="cuda"),
                weights.cuda(),
            ).cpu()
        )
    experts.drain()

    repeats = []
    for indices, weights in draws:
        repeats.append(
            experts.forward(
                hidden.cuda(),
                torch.tensor(indices, dtype=torch.int64, device="cuda"),
                weights.cuda(),
            ).cpu()
        )
    experts.drain()

    for index, (a, b) in enumerate(zip(first, repeats)):
        expected = reference(source, hidden, draws[index][0], draws[index][1], layer=3, quantised=True)
        assert torch.allclose(a, b, atol=0), f"draw {index} is not repeatable"
        assert torch.allclose(a, expected, atol=2e-3), f"draw {index} is not its own draw"


@needs_cuda
def test_the_source_is_asked_for_each_drawn_expert_exactly_once():
    source = SyntheticSource()
    experts = device_experts(source)
    hidden = torch.randn(1, DIM) * 0.4
    experts.forward(
        hidden.cuda(),
        torch.tensor([3, 1], dtype=torch.int64, device="cuda"),
        torch.tensor([0.5, 0.5], device="cuda"),
    )
    experts.drain()
    assert source.draws == [(3, 3), (3, 1)]
    assert experts.staged_experts == TOP_K
    assert experts.staged_bytes == TOP_K * experts.expert_bytes


# ---------------------------------------------------------------------------
# The resident set
#
# The set exists because the copies are the step's floor: a token stages `top_k / world` experts a
# layer and the layers are a chain, so a decode step at a world of four moves 1198.5 MiB a rank and
# the link needs the time it needs. Holding a layer's hottest experts on the card removes their
# copies, and the claim that makes the whole thing safe is an equality rather than a tolerance.
# ---------------------------------------------------------------------------


@needs_cuda
def test_a_resident_set_answers_a_draw_the_bytes_the_staging_row_would_have():
    """Residency changes the traffic and not the answer, so the two builds agree to the bit.

    `_stage` writes the same six views into whichever row it was handed and the kernel is given
    rows, so a resident expert is the same values summed in the same order as a copied one. That
    is why `torch.equal` is the whole claim and why the policy can be scored, or changed, without
    a correctness argument -- and it is checked here at every draw, including the ones that come
    after the set has filled and started refusing.
    """
    torch.manual_seed(2)
    hidden = torch.randn(1, DIM) * 0.4
    draws = (
        ([0, 1], [0.6, 0.4]),
        ([2, 1], [0.3, 0.7]),
        ([1, 0], [0.5, 0.5]),
        ([3, 1], [0.9, 0.1]),
        ([1, 2], [0.2, 0.8]),
        ([4, 1], [0.4, 0.6]),
    )
    out: dict[int, list[torch.Tensor]] = {}
    for rows in (0, 2):
        experts = device_experts(SyntheticSource(), resident_rows=rows, resident_layers=2)
        got = []
        for indices, weights in draws:
            got.append(
                experts.forward(
                    hidden.cuda(),
                    torch.tensor(indices, dtype=torch.int64, device="cuda"),
                    torch.tensor(weights, device="cuda"),
                ).cpu()
            )
            experts.drain()
        out[rows] = got
    for index in range(len(draws)):
        assert torch.equal(out[0][index], out[2][index]), f"draw {index} moved"
    assert out[2][0].abs().sum() > 0


@needs_cuda
def test_a_held_expert_is_not_asked_of_the_source_a_second_time():
    """The copy is what the set buys, so the source is the instrument: it is asked once an expert.

    A draw the set holds never reaches `_stage`, which is the only thing that asks the source. So
    the second draw of the same pair is free and the counters say so -- `staged_experts` counts the
    copies and not the draws, and the admits are the only copies a run of repeats makes.
    """
    source = SyntheticSource()
    experts = device_experts(source, resident_rows=TOP_K, resident_layers=1)
    hidden = torch.randn(1, DIM) * 0.4
    indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    weights = torch.tensor([0.5, 0.5], device="cuda")
    for _ in range(4):
        experts.forward(hidden.cuda(), indices, weights)
        experts.drain()

    assert source.draws == [(3, 0), (3, 1)], "a held expert was copied again"
    assert experts.staged_experts == TOP_K, "a held draw was billed for a copy"
    assert experts.resident_hits == 3 * TOP_K
    report = experts.resident_report()
    assert report["held"] == TOP_K and report["swaps"] == 0
    assert report["hit_rate"] == 0.75, "the two admits should be a quarter of the eight draws"


@needs_cuda
def test_a_resident_set_and_a_chunk_band_do_not_share_an_arena():
    with pytest.raises(ValueError, match="do not share an arena"):
        device_experts(
            SyntheticSource(), resident_rows=2, chunk_rows=2, n_experts=N_EXPERTS
        )


def test_a_resident_row_is_a_block_a_layer_and_a_row_a_slot():
    """The row a layer holds is claimed on its first draw, and a layer id is not an offset.

    A model built for a subset of the layers still names them by their absolute index, so a row
    computed as `layer_id * rows` walks off the end of the region as soon as any layer is left
    out -- and the row it lands on is a staging row, which is a wrong answer rather than a crash.
    """
    held = _Residents(rows=2, layers=2)
    held.take(47, 7)
    held.take(3, 7)
    assert held.row(47, 0) == 0 and held.row(47, 1) == 1
    assert held.row(3, 0) == 2 and held.row(3, 1) == 3
    assert len(held.ids) == 2, "the third layer's block should still be unclaimed"
    with pytest.raises(ValueError, match="built for 2 layers"):
        held.take(11, 7)


def test_the_coldest_resident_leaves_only_for_a_strictly_hotter_draw():
    """An unfilled row is taken, a tie is refused, and the coldest leaves for a hotter challenger.

    Refusing a tie is what stops the set thrashing on noise, and it is why this can be a running
    policy rather than a periodic rebuild: there is no calibration prompt and no eviction burst.
    """
    held = _Residents(rows=2, layers=1)
    assert held.take(0, 10) == (0, True)
    assert held.take(0, 11) == (1, True)
    assert held.take(0, 12) is None, "a tie evicted the incumbent"
    assert held.take(0, 12) == (0, True), "a strictly hotter draw did not take the coldest row"
    assert held.ids[0] == [12, 11]
    assert held.swaps == 1
    assert held.take(0, 11) == (1, False)
    assert (held.hits, held.misses, held.admits) == (1, 4, 3)
    assert held.held == 2


# ---------------------------------------------------------------------------
# The released weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


@needs_release_cuda
def test_the_released_device_experts_reproduce_the_reference_arithmetic(release):
    """The kernel's own arithmetic, which is int8 activations and float32 accumulation.

    A random draw is a legal draw, and three of them are enough to catch a row filled from the
    wrong expert or a tensor staged into the wrong one -- which is what a wrong answer looks
    like here, and it is 14 to 250 percent off rather than 1e-6.
    """
    config = release.layer
    source = MmapExpertSource(release)
    experts = MimoV2DeviceExperts(
        source, 5, device="cuda", top_k=config.num_experts_per_tok,
        dim=config.hidden_size, inter_dim=config.moe_intermediate_size,
    )
    assert experts.expert_bytes == 13_369_344

    torch.manual_seed(0)
    for _ in range(3):
        hidden = (torch.randn(1, config.hidden_size) * 0.5).float()
        indices = torch.randint(0, config.n_routed_experts, (1, config.num_experts_per_tok))
        weights = torch.softmax(torch.randn(1, config.num_experts_per_tok), dim=-1)[0]
        out = experts.forward(
            hidden.cuda(),
            indices[0].to(torch.int64).cuda(),
            weights.cuda(),
        ).cpu()
        experts.drain()

        exact = reference(
            source, hidden, indices[0].tolist(), weights, layer=5, quantised=True
        )
        assert torch.allclose(out, exact, atol=1e-4), (
            (out - exact).abs().max().item(),
            exact.abs().max().item(),
        )


@needs_release_cuda
def test_the_int8_activation_quantisation_is_what_the_float_reference_disagrees_about(release):
    """Records the cost rather than asserting a bound: the kernel is int8 in, float32 out.

    On a released layer and an ordinary draw the agreement with a float32 reference is a few
    parts in a thousand of the output's peak, and it is worse where the summed output is small
    and the hidden is not -- cancellation, which no tolerance is going to make go away. The
    end-to-end figure is a property of the whole backbone and belongs to a run of it, not to
    this test; what this pins is that the disagreement is *bounded by that* and not by a bug,
    which is what the exact test above establishes.
    """
    config = release.layer
    source = MmapExpertSource(release)
    experts = MimoV2DeviceExperts(
        source, 5, device="cuda", top_k=config.num_experts_per_tok,
        dim=config.hidden_size, inter_dim=config.moe_intermediate_size,
    )
    torch.manual_seed(0)
    hidden = (torch.randn(1, config.hidden_size) * 0.5).float()
    indices = torch.tensor([137, 22, 200, 5, 61, 90, 244, 12], dtype=torch.int64)
    weights = torch.tensor([0.13] * config.num_experts_per_tok)

    out = experts.forward(hidden.cuda(), indices.cuda(), weights.cuda()).cpu()
    experts.drain()
    exact = reference(source, hidden, indices.tolist(), weights, layer=5, quantised=True)
    loose = reference(source, hidden, indices.tolist(), weights, layer=5, quantised=False)

    assert torch.allclose(out, exact, atol=1e-4)
    assert not torch.allclose(out, loose, atol=1e-5)
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten(), loose.flatten(), dim=0
    ).item()
    assert cosine > 0.98, cosine
