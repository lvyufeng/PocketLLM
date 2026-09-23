"""The draw, dealt out: who stages what, and whether the shares are the draw.

Two halves, and neither needs a process group.

The first is the deal itself, which is arithmetic and not communication: `id` gives a drawing to
`expert % world` and `sorted` gives sorted position `p` to rank `p % world`, and the difference
between them is the whole reason both exist. `sorted` deals a top-8 draw as exactly 2, 2, 2, 2 --
fixed, with no variance and no rank left out -- which is the decode shape. `id` partitions the
*experts*, so a rank is ever dealt from a quarter of them, which is what makes a prefill chunk stage
a quarter of its bytes; and it leaves a rank of four with nothing in 10% of top-8 draws, which is
not a corner case and is why the zero has to be a value rather than a skipped call.

The second is the arithmetic the deal is supposed to partition. Four ranks' partials are summed
*in this process* and held against the one-rank answer, which is the strongest statement available
without four cards: if the deal dropped a drawing, double-counted one, or paired a weight with the
wrong expert, the sum is not the draw. It is also the cheapest one, because a rank's module is just
another `MimoV2DeviceExperts` and the sum is four additions rather than a fabric.

What this does *not* test is the collective. `make_all_reduce` is a closure around
`dist.all_reduce` and there is nothing in it to get wrong; whether four processes on four cards
still agree is what the multi-rank bench and the four-rank decode comparison answer, and neither
is something a single-process test can stand in for. `make_all_gather` is the other one, and it
*does* have something to get wrong: the primitive joins along the first axis where the attention
wants the last, and a caller that allocates the shape it wants gets a silent scramble for any
`rows` over one. So the collective is faked here -- one function that writes the layout the real
one writes -- and the only question asked of it is where each rank's piece ended up.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.device_experts import MimoV2DeviceExperts  # noqa: E402
from src.models.mimo_v2.ep import (  # noqa: E402
    DEAL_ENV,
    EpGroup,
    deal_card,
    deal_rule,
    make_all_reduce,
    owned_positions,
    rows_per_card,
)
from tests.test_models_mimo_v2_device_model import (  # noqa: E402
    DEVICE,
    HIDDEN,
    INTER,
    TOP_K,
    SyntheticCheckpoint,
    SyntheticSource,
    delta,
    synthetic_tensors,
    tiny_config,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

WORLD = 4

#: A top-8 draw of distinct experts, which is the shape the release actually makes.
TOP_EIGHT = [3, 41, 7, 200, 12, 88, 130, 5]


def assert_partition(positions: list[list[int]], width: int) -> None:
    """The ranks' shares of a draw: disjoint, and together the whole draw."""
    flat = sorted(position for share in positions for position in share)
    assert flat == list(range(width)), f"the shares {positions} are not a partition of {width}"


# ---------------------------------------------------------------------------
# The deal
# ---------------------------------------------------------------------------


def test_the_id_deal_reads_the_experts_own_number():
    """`id` is a property of the expert, so the drawn position does not enter it."""
    assert [deal_card(expert, position, deal="id", world=WORLD) for position, expert in enumerate(TOP_EIGHT)] == [
        expert % WORLD for expert in TOP_EIGHT
    ]
    # The same expert at two different positions goes to the same rank, which is the whole
    # difference from `sorted` and the reason a rank's staged set stays narrow under `id`.
    assert deal_card(41, 0, deal="id", world=WORLD) == deal_card(41, 5, deal="id", world=WORLD)


def test_the_sorted_deal_reads_the_position_and_not_the_expert():
    """Two draws that route to the same experts in a different order are the same deal."""
    shuffled = [TOP_EIGHT[index] for index in (4, 1, 7, 0, 5, 2, 6, 3)]
    by_id = [deal_card(e, p, deal="sorted", world=WORLD) for p, e in enumerate(sorted(TOP_EIGHT))]
    by_shuffle = [
        deal_card(e, p, deal="sorted", world=WORLD) for p, e in enumerate(sorted(shuffled))
    ]
    assert by_id == by_shuffle


def test_the_sorted_deal_splits_a_top_eight_draw_two_two_two_two():
    """The deal that makes a rank's traffic predictable, and it is exact at top 8 over 4."""
    positions = [
        owned_positions(TOP_EIGHT, rank=rank, world=WORLD, deal="sorted") for rank in range(WORLD)
    ]
    assert [len(share) for share in positions] == [2, 2, 2, 2]
    assert_partition(positions, len(TOP_EIGHT))


def test_the_sorted_deal_does_not_care_that_the_experts_are_all_congruent():
    """The draw `id` cannot spread: eight experts that are all the same rank's.

    Four ranks and a draw of eight experts congruent mod four is rank 0's whole draw under `id`,
    and 2, 2, 2, 2 under `sorted`. This is what `sorted` is for.
    """
    congruent = [4 * step for step in range(8)]
    assert owned_positions(congruent, rank=0, world=WORLD, deal="id") == list(range(8))
    assert [len(owned_positions(congruent, rank=r, world=WORLD, deal="sorted")) for r in range(WORLD)] == [2, 2, 2, 2]


def test_the_id_deal_can_hand_a_rank_nothing():
    """One draw in ten at top 8 over 4, and the rank still owes the collective a zero."""
    congruent = [WORLD * step + 1 for step in range(8)]
    assert owned_positions(congruent, rank=0, world=WORLD, deal="id") == []
    assert len(owned_positions(congruent, rank=1, world=WORLD, deal="id")) == 8
    for rank in (2, 3):
        assert owned_positions(congruent, rank=rank, world=WORLD, deal="id") == []
    # And the same draw is spread evenly by the other deal, which is what the two are for.
    assert [len(owned_positions(congruent, rank=r, world=WORLD, deal="sorted")) for r in range(WORLD)] == [2, 2, 2, 2]


def test_the_two_deals_partition_the_same_draw():
    for deal in ("id", "sorted"):
        assert_partition(
            [
                owned_positions(TOP_EIGHT, rank=rank, world=WORLD, deal=deal)
                for rank in range(WORLD)
            ],
            len(TOP_EIGHT),
        )


def test_a_world_of_one_owns_the_whole_draw_and_sends_nothing():
    assert owned_positions(TOP_EIGHT, rank=0, world=1, deal="id") == list(range(8))
    assert owned_positions(TOP_EIGHT, rank=0, world=1, deal="sorted") == list(range(8))
    assert make_all_reduce(1) is None
    assert EpGroup().partial is False


def test_rows_per_card_is_a_quarter_under_sorted_and_all_of_it_under_id():
    """The `id` deal's cost, paid in arena rows: nothing bounds a draw's congruences."""
    assert rows_per_card("id", 8, WORLD) == 8
    assert rows_per_card("sorted", 8, WORLD) == 2
    assert rows_per_card("id", 2, WORLD) == 2
    # A top-2 draw over four ranks is one position a rank at most, so one row.
    assert rows_per_card("sorted", 2, WORLD) == 1
    assert rows_per_card("sorted", 6, 4) == 2
    assert rows_per_card("sorted", 1, 1) == 1
    with pytest.raises(ValueError):
        rows_per_card("congruent", 8, WORLD)


def test_the_deal_is_the_environment_unless_a_caller_says_otherwise(monkeypatch):
    """`sorted` unless something says otherwise, and something can only say `id` or `sorted`.

    The default is the decode deal because decode is the only path there is; `id` is a word a
    caller has to use on purpose.
    """
    monkeypatch.delenv(DEAL_ENV, raising=False)
    assert deal_rule() == "sorted"
    monkeypatch.setenv(DEAL_ENV, "id")
    assert deal_rule() == "id"
    # Anything else is the default rather than an error, because a variable a run does not set
    # correctly must not be able to change what a run computes.
    monkeypatch.setenv(DEAL_ENV, "interleaved")
    assert deal_rule() == "sorted"


def test_a_group_of_more_than_one_without_a_sum_is_refused():
    """A world of four and no way to sum is a wrong answer, not a configuration."""
    with pytest.raises(ValueError):
        EpGroup(world=WORLD, rank=0)
    with pytest.raises(ValueError):
        EpGroup(world=WORLD, rank=WORLD, reduce=lambda tensor: tensor)
    with pytest.raises(ValueError):
        EpGroup(world=0, rank=0)
    group = EpGroup(world=WORLD, rank=3, reduce=lambda tensor: tensor)
    assert group.partial is True and group.rank == 3


# ---------------------------------------------------------------------------
# The shares are the draw
# ---------------------------------------------------------------------------


def expert_module(
    *, world: int = 1, rank: int = 0, deal: str = "id", top_k: int = TOP_K, experts: int = 4
):
    """One rank's routed experts over the miniature source. Two slots, as a model builds it."""
    return MimoV2DeviceExperts(
        SyntheticSource(n_experts=experts),
        top_k=top_k,
        dim=HIDDEN,
        inter_dim=INTER,
        device=DEVICE,
        slots=2,
        world=world,
        rank=rank,
        deal=deal,
    )


@needs_cuda
@pytest.mark.parametrize("deal", ["id", "sorted"])
def test_the_ranks_shares_sum_to_the_one_rank_answer(deal):
    """The partition the deal claims, held against the arithmetic it is supposed to divide.

    The one-rank module is the oracle: same weights, same draw, same hidden state, and no
    collective anywhere. Four partials summed in this process have to land on it -- and the
    association differs, so this is a tolerance and not equality.
    """
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(1, HIDDEN, generator=generator).to(DEVICE)
    indices = torch.arange(TOP_K, dtype=torch.int64, device=DEVICE)
    weights = torch.softmax(torch.randn(TOP_K, generator=generator), dim=0).to(DEVICE)

    whole = expert_module()
    shares = [expert_module(world=WORLD, rank=rank, deal=deal) for rank in range(WORLD)]
    reference = whole.forward(hidden, indices, weights, layer_id=0)
    assert reference.shape == (1, HIDDEN)

    parts = [module.forward(hidden, indices, weights, layer_id=0) for module in shares]
    total = sum(part.float() for part in parts)
    assert delta(total, reference) < 1e-5
    # Two experts over four ranks is one drawing a rank and two ranks with nothing, whichever
    # deal is in force -- and two draws staged in all, which is the draw and not four copies
    # of it.
    assert sorted(module.staged_experts for module in shares) == [0, 0, 1, 1]
    assert sum(module.staged_bytes for module in shares) == whole.staged_bytes


@needs_cuda
def test_a_rank_that_owns_nothing_returns_a_zero_rather_than_skipping_its_turn():
    """The 10% case, and the reason it is a value: the collective is unconditional.

    The draw is two experts that are both rank 1's under `id`, so rank 0 has nothing staged,
    nothing computed, and a zero to contribute -- with the counters unmoved, which is what says
    the zero did not cost a copy.
    """
    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(1, HIDDEN, generator=generator).to(DEVICE)
    indices = torch.tensor([1, 5], dtype=torch.int64, device=DEVICE)
    weights = torch.tensor([0.5, 0.5], device=DEVICE)

    empty = expert_module(world=WORLD, rank=0, experts=8)
    owned = expert_module(world=WORLD, rank=1, experts=8)
    out = empty.forward(hidden, indices, weights, layer_id=0)
    assert out.shape == (1, HIDDEN) and float(out.abs().sum()) == 0.0
    assert empty.staged_experts == 0 and empty.staged_bytes == 0

    other = owned.forward(hidden, indices, weights, layer_id=0)
    assert owned.staged_experts == 2
    whole = expert_module(experts=8).forward(hidden, indices, weights, layer_id=0)
    assert delta(other.float() + out.float(), whole) < 1e-5


@needs_cuda
def test_the_arena_is_as_wide_as_the_deal_can_hand_a_rank():
    """A `sorted` top-8 over four ranks is two rows, and an `id` one is eight.

    The difference is 51 MiB against 204 MiB of card per slot, which is small beside a model and
    is not small beside the arena.
    """
    sorted_module = expert_module(world=WORLD, rank=0, deal="sorted", top_k=8)
    id_module = expert_module(world=WORLD, rank=0, deal="id", top_k=8)
    assert sorted_module.arena_rows == 2
    assert id_module.arena_rows == 8
    assert sorted_module.arena_bytes * 4 == id_module.arena_bytes
    with pytest.raises(ValueError):
        MimoV2DeviceExperts(
            SyntheticSource(n_experts=8),
            top_k=8,
            dim=HIDDEN,
            inter_dim=INTER,
            device=DEVICE,
            world=WORLD,
            rank=0,
            deal="id",
            arena_rows=2,
        )


@needs_cuda
def test_an_eight_wide_draw_over_four_sorted_ranks_stages_two_experts_each():
    """The width a decode actually hands a rank, staged as rows and computed as rows."""
    source = SyntheticSource(n_experts=8)
    modules = [
        MimoV2DeviceExperts(
            source,
            top_k=8,
            dim=HIDDEN,
            inter_dim=INTER,
            device=DEVICE,
            world=WORLD,
            rank=rank,
            deal="sorted",
        )
        for rank in range(WORLD)
    ]
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(1, HIDDEN, generator=generator).to(DEVICE)
    indices = torch.tensor([3, 7, 1, 6, 4, 5, 0, 2], dtype=torch.int64, device=DEVICE)
    weights = torch.full((8,), 0.125, device=DEVICE)
    for module in modules:
        assert module.forward(hidden, indices, weights, layer_id=0).shape == (1, HIDDEN)
        assert module.staged_experts == 2


@needs_cuda
def test_a_layer_sums_four_shares_into_the_answer_one_rank_gives():
    """The same partition one level up, where the layer's own rounding point is.

    The reduce here is the identity, so each model returns its share and the sum is taken by the
    test. That is what the collective does on a fabric, and doing it in this process is what lets
    a single card check the deal's arithmetic without four of them.
    """
    from src.models.mimo_v2.device_model import MimoV2DeviceModel

    config = tiny_config(routed=(0, 1))
    assert config.ffn_kind(1) == "moe"
    source = SyntheticSource()
    checkpoint = SyntheticCheckpoint(config, synthetic_tensors(config, source))

    def build(ep=None):
        return MimoV2DeviceModel(
            checkpoint,
            device=DEVICE,
            dtype=torch.float32,
            layers=[1],
            expert_source=source,
            ep=ep,
            pin=False,
        )

    def identity(tensor):
        """The collective, replaced by the caller's own addition: what the reduce does, minus NCCL."""
        return tensor

    whole = build()
    shares = [build(ep=EpGroup(world=WORLD, rank=rank, reduce=identity)) for rank in range(WORLD)]
    generator = torch.Generator().manual_seed(23)
    normed = (torch.randn(1, HIDDEN, generator=generator) * 0.2).to(DEVICE).to(torch.float32)

    reference = whole.layers[0].mlp(normed)
    parts = [model.layers[0].mlp(normed) for model in shares]
    for part in parts:
        assert part.shape == reference.shape
    assert delta(sum(parts), reference) < 1e-5


def test_the_join_writes_each_rank_down_its_own_columns(monkeypatch):
    """`all_gather_into_tensor` joins along the first axis; the attention wants the last.

    A caller that allocates the tensor it wants -- `[rows, world * width]` -- hands the collective a
    buffer it reads as `world * rows` rows, and for any `rows` over one every piece lands in the
    wrong place: the same numbers, a different tensor, no error and no shape mismatch. A decode step
    is where it hides, because `rows` is one there and the flat layout happens to be the
    concatenation; a prompt prefill is where it goes wrong, and it took a layer-0 comparison to see
    it at all.

    The collective is faked here rather than skipped: one function that writes the layout the real
    one writes, so the question -- which columns did rank `r`'s piece come out in -- is the real
    one, and it is answerable without a fabric or a second card.
    """
    import torch.distributed as dist

    from src.models.mimo_v2.ep import make_all_gather

    world, rows, width = 4, 3, 2

    def fake_all_gather_into_tensor(out, part, *args, **kwargs):
        flat, source = out.reshape(-1), part.reshape(-1)
        assert flat.numel() == world * source.numel(), (flat.numel(), source.numel())
        for rank in range(world):
            flat[rank * source.numel() : (rank + 1) * source.numel()] = source + rank

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    gather = make_all_gather(world)
    part = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
    joined = gather(part)
    assert joined.shape == (rows, world * width)
    for rank in range(world):
        assert torch.equal(joined[:, rank * width : (rank + 1) * width], part + rank), rank


def test_a_single_rank_joins_nothing():
    from src.models.mimo_v2.ep import make_all_gather

    assert make_all_gather(1) is None
