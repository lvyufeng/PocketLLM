"""A chunk of tokens through the routed experts: the prefill path, and what it has to agree with.

`forward` is one token and `forward_chunk` is a chunk, and the difference is not the batch -- the
kernel has always been batched -- but the layout. A decode step's draw is eight experts out of a
quarter of them, so the arena holds the draw and the kernel reads one row a drawing. A chunk draws
nearly every expert there is, so what a rank computes is the *subset of the chunk's pairs whose
expert it owns*, and the kernel wants those pairs grouped by expert with an arena row a group. Two
things follow and both are checked here: the layout is a gather, a stable sort and a `searchsorted`,
which is easy to get right in a way that is silently wrong about *which* pair a weight belongs to;
and the chunk path is a different kernel from the single-token one, so "the same arithmetic" is a
claim with a measured number behind it rather than an identity.

The number is this: the two kernels agree to about 1e-7 of the answer's own peak, not exactly,
because the grouped kernel tiles K in two stages of shared memory where the single-token one tiles
it in one, and reduces a token's pairs in a separate pass. The miniature here has a peak of 2.1e3
-- its experts are random codes at scale one, so their outputs are large -- and the two paths differ
by 1.2e-4 on that, which is 6e-8 relative. Every comparison in this file is against that peak and
not against an absolute tolerance, because the same kernel on the released checkpoint's experts
produces a stream of a very different size.

The other half of the file is the model: `mlp` dispatches on the row count, `prefill` runs a prompt
in chunks of the caller's width, and a chunked prompt has to produce the same tokens as a prompt fed
one token at a time. That last one is the end-to-end claim, and it is checked both on the miniature
and on two released layers -- the miniature's experts are random, so only the release can say the
chunk path is right about real weights.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.device_experts import MimoV2DeviceExperts  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup, owned_experts  # noqa: E402
from src.models.mimo_v2.layers import gate_and_route  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.test_models_mimo_v2_device_model import (  # noqa: E402
    DEVICE,
    HIDDEN,
    INTER,
    Fixture,
    SyntheticSource,
    tiny_config,
)

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the released prefill needs both the release and a CUDA device",
)

#: The module-level miniature: sixteen experts, a draw of four, so four ranks of the `id` deal own
#: four apiece and a chunk's pairs divide evenly.
TOP_K = 4
N_EXPERTS = 16
LAYER = 3
ROWS = 6
WORLD = 4

#: What the two kernels actually cost each other. The grouped kernel is not the single-token one:
#: it tiles K over two stages of shared memory and reduces a token's pairs in its own pass, so the
#: float32 accumulation is reassociated and the answers differ in the last bits. Measured at the
#: miniature's dimensions it is 1.2e-4 against a peak of 2.1e3 -- 6e-8 -- and the tightest
#: comparison in this file is one order above that.
ROUNDING = 1e-6


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The largest disagreement between two answers, as a fraction of the answer's own peak."""
    return float((actual - expected).abs().max() / expected.abs().max())


def expert_module(
    source, *, world=1, rank=0, deal="id", chunk_rows=0, top_k=TOP_K, n_experts=N_EXPERTS, **kwargs
):
    """A module at the miniature's dimensions, with the chunk path on unless told otherwise."""
    return MimoV2DeviceExperts(
        source,
        LAYER,
        device=DEVICE,
        top_k=top_k,
        dim=HIDDEN,
        inter_dim=INTER,
        world=world,
        rank=rank,
        deal=deal,
        n_experts=n_experts,
        chunk_rows=chunk_rows,
        **kwargs,
    )


def routing(rows: int = ROWS, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A chunk's routing, drawn from a seed: `[rows, top_k]` ids and the weights with them.

    Random and not the miniature model's own router, because a chunk's value is that its rows
    draw *different* experts: the model's one-hot gate draws the same two for every token, which
    is a layout with two slots and no grouping to speak of.
    """
    torch.manual_seed(seed)
    hidden = (torch.randn(rows, HIDDEN) * 0.4).to(DEVICE)
    gate = torch.randn(N_EXPERTS, HIDDEN).to(DEVICE)
    bias = torch.zeros(N_EXPERTS, device=DEVICE)
    indices, weights = gate_and_route(
        hidden,
        gate,
        bias,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )[:2]
    return hidden, indices, weights


# ---------------------------------------------------------------------------
# What a chunk stages, and what it refuses
# ---------------------------------------------------------------------------


@needs_cuda
def test_the_deal_partitions_the_experts_a_chunk_needs():
    """`owned_experts` is the whole reading of a chunk's share, and the two deals differ.

    Under `id` the four ranks' shares are disjoint and cover the layer exactly once, which is
    what makes a dealt chunk's bytes a quarter of one rank's. Under `sorted` every rank's share
    is the whole set -- a chunk's positions reach every expert -- which is the same answer for
    four times the copy, and is why the chunk path refuses it.
    """
    shares = [owned_experts(N_EXPERTS, rank=rank, world=WORLD, deal="id") for rank in range(WORLD)]
    assert [len(share) for share in shares] == [N_EXPERTS // WORLD] * WORLD
    assert sorted(expert for share in shares for expert in share) == list(range(N_EXPERTS))
    for rank, share in enumerate(shares):
        assert share == [expert for expert in range(N_EXPERTS) if expert % WORLD == rank]
    assert owned_experts(N_EXPERTS, rank=0, world=1, deal="id") == list(range(N_EXPERTS))
    assert owned_experts(N_EXPERTS, rank=0, world=1, deal="sorted") == list(range(N_EXPERTS))
    for rank in range(WORLD):
        assert owned_experts(N_EXPERTS, rank=rank, world=WORLD, deal="sorted") == list(
            range(N_EXPERTS)
        )
    with pytest.raises(ValueError, match="not a deal"):
        owned_experts(N_EXPERTS, rank=0, world=WORLD, deal="round-robin")


@needs_cuda
def test_a_chunk_without_a_band_is_refused():
    """The arena a decode module carries is one draw wide, and a chunk does not fit in it.

    Refused rather than answered, because the alternative is a rank reading a chunk's experts
    out of two rows: the kernel would be handed an arena that does not hold what the layout says
    it holds, and a wrong expert's bytes are the kind of wrong that still produces a number.
    """
    module = expert_module(SyntheticSource(n_experts=N_EXPERTS), chunk_rows=None)
    hidden, indices, weights = routing()
    assert module.chunk_rows is None
    with pytest.raises(ValueError, match="chunk band"):
        module.forward_chunk(hidden, indices, weights)


@needs_cuda
def test_a_chunk_under_a_sorted_deal_is_refused_at_a_world_over_one():
    """The refusal that keeps a prefill from quietly costing four times its copy.

    `sorted` is the decode default and it is the right deal there -- two experts a rank, no
    variance -- but a chunk's drawings reach every expert on every rank, so a `sorted` chunk
    stages all sixteen on all four of them and saves nothing. The message has to name the deal
    that works, because the caller who reads it is choosing between two defaults.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    for rank in range(WORLD):
        module = expert_module(source, world=WORLD, rank=rank, deal="sorted")
        assert module.n_local == N_EXPERTS
        with pytest.raises(ValueError, match="id"):
            module.forward_chunk(hidden, indices, weights)
    # One rank has nothing to divide, so the deal does not matter and the chunk is served.
    alone = expert_module(source, world=1, deal="sorted")
    assert alone.forward_chunk(hidden, indices, weights).shape == (ROWS, HIDDEN)


@needs_cuda
def test_the_arena_a_chunk_asks_for_is_the_share_it_owns():
    """`chunk_rows=0` is the whole share in one band, and the arena is exactly that.

    Four ranks over sixteen experts is four rows a rank, which is a quarter of the arena one
    rank would need for the same chunk -- and the module bills the caller for it, since the
    arena is card memory and the width is a decision about how much of it to spend.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    for rank in range(WORLD):
        module = expert_module(source, world=WORLD, rank=rank)
        assert module.n_local == N_EXPERTS // WORLD
        assert module.chunk_rows == N_EXPERTS // WORLD
        assert module.arena_rows == max(TOP_K, N_EXPERTS // WORLD)
    banded = expert_module(source, chunk_rows=2)
    assert banded.chunk_rows == 2 and banded.n_local == N_EXPERTS
    assert banded.arena_rows == TOP_K
    # A band wider than the share is the share, because there is nothing else to hold.
    assert expert_module(source, chunk_rows=N_EXPERTS + 8).chunk_rows == N_EXPERTS


@needs_cuda
def test_a_chunk_band_needs_to_be_told_how_many_experts_there_are():
    with pytest.raises(ValueError, match="n_experts"):
        expert_module(SyntheticSource(n_experts=N_EXPERTS), n_experts=None)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


@needs_cuda
def test_a_chunk_agrees_with_the_same_rows_one_at_a_time():
    """The chunk path against the path a decode step takes, row by row.

    This is the claim the whole stage rests on: a chunk of `ROWS` rows through the grouped
    kernel is the same answer as `ROWS` single-token draws, and the same *weights* land on the
    same experts. The layouts are entirely different -- one call over the chunk's pairs grouped
    by expert against one call a row over an eight-wide draw -- so a `pair_weights` that followed
    the wrong pair would show up here as a large disagreement rather than a small one.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    module = expert_module(source)
    chunk = module.forward_chunk(hidden, indices, weights)
    steps = torch.cat(
        [module.forward(hidden[t : t + 1], indices[t], weights[t]) for t in range(ROWS)], 0
    )
    assert chunk.shape == (ROWS, HIDDEN)
    assert relative(chunk, steps) < ROUNDING
    # And the two are not the same call by accident: every row drew its own experts.
    assert len({int(expert) for row in indices for expert in row}) > TOP_K


@needs_cuda
def test_a_chunk_of_one_row_is_the_single_token_call():
    """The degenerate width, which is what a serving loop's last chunk looks like."""
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    module = expert_module(source)
    chunk = module.forward_chunk(hidden[:1], indices[:1], weights[:1])
    single = module.forward(hidden[:1], indices[0], weights[0])
    assert chunk.shape == single.shape
    assert relative(chunk, single) < ROUNDING


@needs_cuda
def test_the_bands_of_a_chunk_sum_to_the_same_answer():
    """A band is a slice of one sorted pair list, so banding is a memory decision and not a
    different computation.

    A band of one expert is *bit-identical* to a single band, and that is not luck: a top-k draw
    names distinct experts, a token's pairs in slot order are therefore in ascending expert
    order, and the kernel's own reduction adds them in that order. Summing one-expert bands in
    slot order is the same sequence of adds. Wider bands regroup those adds and round differently
    in the last bits, which is what the tolerance is for, and it is the reason the width can be
    chosen for arena bytes without a parity question behind it.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    whole = expert_module(source).forward_chunk(hidden, indices, weights)
    one = expert_module(source, chunk_rows=1).forward_chunk(hidden, indices, weights)
    assert torch.equal(one, whole)
    for width in (2, 3, 5):
        banded = expert_module(source, chunk_rows=width)
        out = banded.forward_chunk(hidden, indices, weights)
        assert relative(out, whole) < ROUNDING


@needs_cuda
def test_the_shares_of_a_dealt_chunk_sum_to_the_one_rank_answer():
    """Four ranks of the `id` deal, one chunk: the partials are the answer.

    The same property the decode path has, at a different shape -- and here the shares are much
    more unequal, because a chunk's rows are routed independently and a rank can be drawn by
    every row of one expert group and none of another. The sum is taken in fp32 where the model
    takes it, so the comparison is against the one-rank answer with the same tolerance the two
    kernels have.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    whole = expert_module(source, world=1).forward_chunk(hidden, indices, weights)
    ranks = [expert_module(source, world=WORLD, rank=rank) for rank in range(WORLD)]
    parts = [module.forward_chunk(hidden, indices, weights) for module in ranks]
    assert relative(sum(parts), whole) < ROUNDING
    # Each rank computed a strict subset, and every pair was computed exactly once.
    owned = sum(int((module._row_of_local[indices] < module.n_local).sum()) for module in ranks)
    assert owned == ROWS * TOP_K


@needs_cuda
def test_a_rank_that_owns_none_of_a_chunks_experts_answers_zero():
    """A zero and not a skipped call: the collective a dealt layer ends with is unconditional.

    Under `id` with a draw of four over sixteen, a rank owns one expert in four, so a chunk whose
    rows happen to draw only another rank's experts is not a corner -- and the answer it has to
    arrive at the collective with is exactly zero, not the empty tensor its pair list would give.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing()
    assert indices.shape == (ROWS, TOP_K)
    # Every drawing moved into rank 1's share, so rank 0's pair list is empty and rank 1's is
    # everything, which is the widest skew a chunk can have. The drawing ids are rank 1's own,
    # so this is a routing the router could have produced and not a hand-built layout.
    theirs_pool = owned_experts(N_EXPERTS, rank=1, world=WORLD, deal="id")
    theirs = torch.tensor(
        [
            [theirs_pool[(row * TOP_K + column) % len(theirs_pool)] for column in range(TOP_K)]
            for row in range(ROWS)
        ],
        dtype=torch.int64,
        device=DEVICE,
    )
    rank0 = expert_module(source, world=WORLD, rank=0)
    out = rank0.forward_chunk(hidden, theirs, weights)
    assert out.shape == (ROWS, HIDDEN)
    assert bool((out == 0).all())
    assert rank0.staged_experts == 0
    # Rank 1 owns the whole chunk and answers with all of it.
    rank1 = expert_module(source, world=WORLD, rank=1)
    assert bool((rank1.forward_chunk(hidden, theirs, weights) != 0).any())
    assert rank1.staged_experts == rank1.n_local
    # And rank 1's answer is the one rank's answer, because it owns every drawing.
    whole = expert_module(source, world=1).forward_chunk(hidden, theirs, weights)
    assert relative(rank1.forward_chunk(hidden, theirs, weights), whole) < ROUNDING


@needs_cuda
def test_a_dealt_chunk_stages_a_quarter_of_the_experts():
    """The prefill's win, counted rather than timed: bytes are a quarter and so is the arithmetic.

    One rank's whole share is sixteen experts; four ranks hold four apiece, and a chunk that
    draws every expert gives each of them all four of its own. So the four ranks' staged bytes
    are the sixteen one rank would have staged, once, and no expert crosses the bus twice.
    """
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing(rows=64)
    whole = expert_module(source, world=1)
    whole.forward_chunk(hidden, indices, weights)
    assert whole.staged_experts == N_EXPERTS
    ranks = [expert_module(source, world=WORLD, rank=rank) for rank in range(WORLD)]
    for module in ranks:
        module.forward_chunk(hidden, indices, weights)
        assert module.staged_experts == N_EXPERTS // WORLD
    assert sum(module.staged_bytes for module in ranks) == whole.staged_bytes
    drawn = {int(expert) for row in indices for expert in row}
    assert len(drawn) == N_EXPERTS, "the chunk has to draw every expert for this to be a quarter"


@needs_cuda
def test_a_banded_chunk_stages_only_the_bands_it_draws():
    """A band whose experts the chunk never drew costs nothing, which is what makes a narrow
    band a way to spend less memory rather than a way to do more work."""
    source = SyntheticSource(n_experts=N_EXPERTS)
    hidden, indices, weights = routing(rows=2)
    drawn = {int(expert) for row in indices for expert in row}
    module = expert_module(source, chunk_rows=1)
    out = module.forward_chunk(hidden, indices, weights)
    assert module.staged_experts == len(drawn)
    assert module.staged_experts < N_EXPERTS
    assert relative(out, expert_module(source).forward_chunk(hidden, indices, weights)) < ROUNDING


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@needs_cuda
def test_the_layer_routes_a_chunk_by_its_row_count():
    """One row is a step, many rows are a chunk, and the dispatch is the row count.

    A model built without a chunk band is a decode model, and its routed layer says so when it is
    handed a chunk -- which is a better failure than drawing every token's experts one at a time,
    because that path is correct and would look like a model that is merely slow.
    """
    config = tiny_config(routed=(0, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source, layers=[1])
    layer = fixture.model.layers[0]
    assert layer.kind == "moe"

    hidden = torch.randn(1, config.hidden_size, device=DEVICE)
    assert layer.mlp(hidden).shape == hidden.shape
    chunk = torch.randn(3, config.hidden_size, device=DEVICE)
    with pytest.raises(ValueError, match="chunk band"):
        layer.mlp(chunk)

    banded = MimoV2DeviceModel(
        fixture.checkpoint,
        device=DEVICE,
        dtype=torch.float32,
        layers=[1],
        expert_source=source,
        pin=False,
        deal="id",
        chunk_rows=0,
    )
    assert banded.layers[0].mlp(chunk).shape == chunk.shape
    assert banded.experts.n_local == config.n_routed_experts


@needs_cuda
def test_a_sorted_deal_cannot_be_asked_to_prefill_at_a_world_over_one():
    """The deal is a property of the module and not of the call, so the refusal is the module's.

    A four-rank run that wants to decode with `sorted` and prefill at all has to be built with
    `id`, and the error says so. `EpGroup` holds the collective and nothing else; the deal is
    the experts module's, and a caller who got the two out of step learns it here rather than
    from a prefill that is four times its copy.
    """
    config = tiny_config(routed=(0, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source, layers=[1])
    group = EpGroup(world=2, rank=0, reduce=lambda tensor: tensor, device=DEVICE)
    model = MimoV2DeviceModel(
        fixture.checkpoint,
        device=DEVICE,
        dtype=torch.float32,
        layers=[1],
        expert_source=source,
        pin=False,
        ep=group,
        deal="sorted",
        chunk_rows=0,
    )
    ids = torch.tensor([3, 17, 42], dtype=torch.int64)
    with pytest.raises(ValueError, match="id"):
        model.prefill(ids.tolist(), chunk=3)


def banded(fixture: Fixture, **kwargs) -> MimoV2DeviceModel:
    """A model over the fixture's checkpoint with the chunk path on.

    `Fixture` builds a decode model, which is the right default -- an arena a rank's whole share
    wide is card memory a run that never prefills should not spend -- so a test that wants a
    prefill builds its own, out of the same checkpoint and source.
    """
    options = {
        "device": DEVICE,
        "dtype": torch.float32,
        "layers": [1],
        "expert_source": fixture.source,
        "pin": False,
        "deal": "id",
        "chunk_rows": 0,
    }
    options.update(kwargs)
    return MimoV2DeviceModel(fixture.checkpoint, **options)


@needs_cuda
def test_a_prefill_returns_the_last_rows_logits_and_only_those():
    """One row out, whatever the chunk width: a prompt's logits are 61 GB at 256k."""
    config = tiny_config(routed=(0, 1))
    fixture = Fixture(config, source=SyntheticSource(), layers=[1])
    model = banded(fixture)
    ids = [3, 17, 42, 8, 5]
    cache = model.cache(len(ids) + 4)
    whole = model.forward(torch.tensor(ids, dtype=torch.int64), start_pos=0, cache=cache)[-1]
    assert whole.shape == (config.vocab_size,)
    for chunk in (1, 2, 3, len(ids)):
        cache.reset()
        row = model.prefill(ids, cache=cache, chunk=chunk)
        assert row.shape == (config.vocab_size,)
        assert relative(row, whole) < ROUNDING
    with pytest.raises(ValueError, match="no tokens"):
        model.prefill([], chunk=2)
    with pytest.raises(ValueError, match="not a chunk"):
        model.prefill(ids, chunk=0)
    # The cache the prefill left behind is what the next token reads, and the window is a ring:
    # a decode step after a chunked prefill lands where one pass over the whole stream lands.
    cache.reset()
    model.prefill(ids, cache=cache, chunk=2)
    stepped = model.step(11, start_pos=len(ids), cache=cache)[-1]
    cache.reset()
    whole = model.forward(torch.tensor(ids + [11], dtype=torch.int64), start_pos=0, cache=cache)[-1]
    assert relative(stepped, whole) < ROUNDING


@needs_cuda
def test_a_chunked_prompt_gives_the_same_tokens_as_one_token_at_a_time():
    """The end-to-end claim, on the miniature: a chunked prefill is a prefill.

    Two paths over the same five tokens -- one call a chunk through the grouped kernel against
    five calls through the single-token one -- and the comparison is the logits *and* the token
    the sampler would take, because a disagreement in the last bits of the stream is only a
    problem when it flips an `argmax`, and on this model's logits it does not.
    """
    config = tiny_config(routed=(0, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source, layers=[1])
    model = banded(fixture)
    ids = [3, 17, 42, 8, 5]
    cache = model.cache(len(ids) + 4)
    stepped = model.forward(torch.tensor(ids, dtype=torch.int64), start_pos=0, cache=cache)[-1]
    cache.reset()
    chunked = model.prefill(ids, cache=cache, chunk=3)
    assert relative(chunked, stepped) < ROUNDING
    assert int(chunked.argmax()) == int(stepped.argmax())
    # The two paths do not agree by being flat: the row has a decided answer and a runner-up.
    order = torch.argsort(chunked, descending=True)[:2]
    assert float(chunked[order[0]] - chunked[order[1]]) > 0


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------


@needs_release_cuda
def test_the_released_two_layers_prefill_as_one_token_at_a_time():
    """Two released layers, a chunk of three tokens against three single-token calls.

    The miniature's experts are random codes, so it can show that the layout is the layout but
    not that the chunk path reads real fp4 weights the way the release means them. Layer 0 is
    the dense layer and layer 1 is the routed one, so this is one real MoE layer fed a chunk,
    out of the checkpoint's own mapped pages rather than the 149.81 GiB bank -- which is why
    the band is one expert wide: a rank's whole share is 256 experts, and only the bands a
    drawing lands in are staged, so this reads about fifty experts and not all of them.

    What can differ between the two paths is the attention's summation order, which changes
    when a row is computed in a chunk instead of alone, and the expert kernel's, which changes
    between the single-token and grouped kernels. Both are the same layer stack otherwise, so
    the disagreement is a rounding and not a different model: measured, it is 3.0e-7 of the
    logits' own peak of 17.45, three hundred times inside the bound below, and the two paths
    take the same token at every one of the six positions.
    """
    checkpoint = MimoV2Checkpoint(RELEASE)
    config = checkpoint.layer
    from src.models.mimo_v2.device_experts import MmapExpertSource

    model = MimoV2DeviceModel(
        checkpoint,
        device=DEVICE,
        dtype=torch.float32,
        layers=[0, 1],
        expert_source=MmapExpertSource(checkpoint),
        pin=False,
        deal="id",
        chunk_rows=1,
    )
    ids = [1024, 2048, 4096, 8192, 512, 256]
    cache = model.cache(len(ids) + 4)
    logits = []
    for position, token in enumerate(ids):
        row = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
        logits.append(row.to(torch.float32).cpu())
    stepped = torch.stack(logits)
    cache.reset()
    staged = model.experts.staged_experts
    chunked = []
    for start in range(0, len(ids), 3):
        out = model.forward(
            torch.tensor(ids[start : start + 3], dtype=torch.int64), start_pos=start, cache=cache
        )
        chunked.append(out.to(torch.float32).cpu())
    chunked = torch.cat(chunked)
    assert relative(chunked, stepped) < 1e-4
    assert [int(row.argmax()) for row in chunked] == [int(row.argmax()) for row in stepped]
    # One rank holds every expert, so its share is the whole set and the band is the knob.
    assert model.experts.n_local == config.n_routed_experts
    # The chunk's two calls drew 24 experts each and read only those: a band of one expert
    # stages the experts the drawings land in, not the share they came from.
    assert 24 <= model.experts.staged_experts - staged < config.n_routed_experts
