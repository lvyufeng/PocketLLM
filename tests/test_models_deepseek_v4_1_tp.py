"""The dense tree cut four ways: what a rank owns, and that four ranks are one.

`tp.py` splits the tree *by construction* -- every module derives its own `1/world` slice from a
`world` -- while the loader has to cut the file's tensors the same way to fill it. Those are two
spellings of one arithmetic and they cannot be shared, because a constructor divides `n_heads` where
the loader has to divide `n_heads * head_dim` and land on a whole number of heads. So the tests
here are written to make the two halves fail against *each other* rather than against the released
checkpoint, which no test can hold:

* `test_the_loader_cuts_the_file_the_way_the_constructor_cut_the_parameter` -- every parameter of
  every rank against a post-hoc slice of the one-rank tree, name for name. This is the pair that
  drifts: a rank whose `wq_b` was built 16 heads wide and filled from a 32-head band still loads,
  still runs, and is wrong.
* `test_the_shards_partition_the_tensor_they_were_cut_from` -- the far end of the same claim: the
  pieces have to *cover* the whole, with no gap and no overlap. The reassembly below is spelled out
  from the projections' own shapes rather than read out of `tp.py`, so the two agree rather than
  agree with themselves.
* `test_a_sharded_forward_is_the_unsharded_forward` -- the tree run four ways at once and against one
  rank, on the same activation and the same weights, through a collective that is four threads and a
  barrier instead of a process group.

The forward test has to be honest about its own reach, and three of its limits are the fixture's or
the dtype's rather than the split's. The routed experts are not dealt out here -- the mini
checkpoint's store holds all 384 of a layer and sums them on the host -- so four ranks compute the
same routed sum and the ffn's all-reduce completes the shared expert's partial *alone*; that is why
`MoE.forward` reduces the shared half and not the total, and
`test_the_ffn_completes_the_routed_partial_only_when_the_store_was_dealt_out` is what covers the
other branch, where the two halves travel on one message. The mini geometry's `index_topk` is wide
enough that no block is ever truncated, so a tie in the indexer's selection cannot arise here: the
discrete half of `probe_tp4_block.py`'s parity check has nothing to catch on this checkpoint, and
what the logits comparison below covers is the arithmetic. And the arithmetic is bf16, where a
row-parallel partial is rounded before it is summed, so the comparison is between two roundings of
one value rather than between two spellings of it -- `_close` states the bound that follows and how
it is derived.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from types import SimpleNamespace

import pytest
import torch

from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.loader import V41Checkpoint, load_backbone
from src.models.deepseek_v4_1.modules import RoutedExperts
from src.models.deepseek_v4_1.tp import (
    INDEXER_ROW_SPLIT_ENV,
    ShardPlan,
    attach_tp,
    make_all_gather,
    make_all_reduce,
    split_names,
)
from tests.test_models_deepseek_v4_1_loader import (
    MINI,
    N_LAYERS,
    _cfg,
    _layout,
    _mini_checkpoint,
    _toy_hasher,
)

# The mini geometry divides by two everywhere a split needs it: 4 heads, 2 o-groups, 2 index heads,
# a 64-wide shared expert. Four would divide too and would cost four trees; two is enough to tell a
# split from a slice, because a rank's band is then neither the whole tensor nor a single row.
WORLD = 2

# Which axis of the *file's* tensor a rank's band runs along, for the six splits that are a plain
# band. The two that are not are handled by the reassembly below. Written from what each projection
# computes -- a row-parallel half keeps the output axis, a column-parallel one keeps the input -- and
# not read out of `tp.py`, which is the point of having it here at all.
_AXIS = {
    "attn.wq_b.weight": 0,
    "attn.attn_sink": 0,
    "attn.wo_b.weight": 1,
    "ffn.shared_experts.w1.weight": 0,
    "ffn.shared_experts.w3.weight": 0,
    "ffn.shared_experts.w2.weight": 1,
}


def _reassemble(tail: str, parts: list[torch.Tensor], cfg) -> torch.Tensor:
    """Put the ranks' pieces back together, on the axis the original was cut along.

    `wo_a` and the indexer's `wq_b` are the two that are not a band of the flat matrix: both stack
    one block per head group (`o_groups` of them, each projecting only its own heads) or per index
    head, so the pieces go back on the *stack's* axis and the flat matrix comes out of a reshape.
    """
    if tail == "attn.wo_a.weight":
        per = cfg.o_groups // WORLD
        stacked = torch.cat([p.view(per, cfg.o_lora_rank, -1) for p in parts], dim=0)
        return stacked.reshape(-1, parts[0].size(-1))
    if tail == "attn.indexer.wq_b.weight":
        per = cfg.index_n_heads // WORLD
        stacked = torch.cat([p.view(per, cfg.index_head_dim, -1) for p in parts], dim=0)
        return stacked.reshape(-1, parts[0].size(-1))
    return torch.cat(parts, dim=_AXIS[tail])


class _MessageBoard:
    """`dist.all_reduce` for one process: `world` threads and a barrier per collective.

    The three collectives in a layer are all-reduces over the rank axis, so a test that wants to see
    what four ranks compute does not need four processes -- it needs four threads that meet at the
    same points in the same order. Each thread posts its partial, waits, reads every slot and sums
    them in rank order, which is what `make_all_reduce`'s fp32 `SUM` does, deterministically.

    The second barrier is not decoration: without it a fast thread could post the *next* collective's
    partial while a slow one is still reading the slots of the previous one, and the sum would be
    over two different layers. It is also why a thread that raises aborts the barrier -- the rest are
    blocked on a point the failure will never reach, and a hung test is worse than a failed one.
    """

    def __init__(self, world: int):
        self.world = world
        self._barrier = threading.Barrier(world)
        self._slots: list[torch.Tensor | None] = [None] * world

    def reduce_for(self, rank: int):
        def collective(tensor: torch.Tensor) -> torch.Tensor:
            self._slots[rank] = tensor.detach().float()
            self._barrier.wait()
            total = torch.stack(list(self._slots)).sum(dim=0)
            self._barrier.wait()
            return total.to(tensor.dtype)

        return collective

    def gather_for(self, rank: int):
        """`dist.all_gather` along the sequence axis: the row split's one message.

        No two-phase trick here, unlike `reduce_for`'s documented case. An all-gather is not a
        reduction -- every rank's own slice *is* part of the answer, whole and already final -- so the
        first barrier is what makes the slots complete and the second is the same "do not post the
        next message over a slow rank's read" that `reduce_for` needs.
        """

        def collective(tensor: torch.Tensor) -> torch.Tensor:
            self._slots[rank] = tensor.detach().clone()
            self._barrier.wait()
            parts = list(self._slots)
            self._barrier.wait()
            return torch.cat(parts, dim=1)

        return collective

    def run(self, body) -> list:
        """Run `body(rank)` on every rank at once, and give back their results in rank order."""
        results: list = [None] * self.world
        failure: list[BaseException] = []

        def one(rank: int) -> None:
            try:
                results[rank] = body(rank)
            except BaseException as error:  # noqa: BLE001 -- re-raised on the test's thread
                failure.append(error)
                self._barrier.abort()

        threads = [threading.Thread(target=one, args=(rank,), daemon=True) for rank in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        assert not any(thread.is_alive() for thread in threads), "a rank never reached the end"
        if failure:
            raise failure[0]
        return results


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """One mini checkpoint, loaded once whole and once per rank: the two halves of the claim."""
    root = str(tmp_path_factory.mktemp("v41-tp"))
    cfg, layout = _cfg(), _layout()
    _mini_checkpoint(root, cfg, layout)
    checkpoint = V41Checkpoint(root)
    whole = load_backbone(cfg, checkpoint, layout=layout, hasher=_toy_hasher(layout))
    shards = [
        load_backbone(cfg, checkpoint, layout=layout, hasher=_toy_hasher(layout), world=WORLD, rank=rank)
        for rank in range(WORLD)
    ]
    yield SimpleNamespace(cfg=cfg, layout=layout, ckpt=checkpoint, whole=whole, shards=shards)
    checkpoint.close()


# -- what is cut, and what that means for the names -----------------------------------------------


def test_the_split_names_are_the_parameters_the_tree_actually_has(trees) -> None:
    """`split_names` is a table of parameter tails, so a typo in it is silent: the loader would pass
    a name it never matched straight through, and the sharded parameter would be filled from the
    whole tensor or refused by shape depending on which axis the typo moved."""
    tails = Counter(
        ".".join(name.split(".")[2:])
        for name, _ in trees.whole.model.named_parameters()
        if name.startswith("layers.")
    )
    expected = {tail: N_LAYERS for tail in split_names()}
    # only a layer that indexes owns an indexer's wq_b, which is why its count is not N_LAYERS
    expected["attn.indexer.wq_b.weight"] = len(MINI["index_source_layers"])
    assert {tail: tails[tail] for tail in split_names()} == expected

    plan = ShardPlan.build(trees.cfg, 0, WORLD)
    # The one that is easy to get wrong: `attn.wq_b` and `attn.indexer.wq_b` end in the same three
    # components and are cut on different axes, so the table is an exact match and not a suffix test.
    # A row band of the indexer's `wq_b` would be half of heads 0 and half of head 1.
    assert plan.local_shape("layers.2.attn.wq_b.weight", (128, 32)) == (64, 32)
    assert plan.local_shape("layers.2.attn.indexer.wq_b.weight", (64, 32)) == (32, 32)
    assert plan.local_shape("layers.2.attn.wkv.weight", (32, 64)) == (32, 64), "wkv replicates"


def test_one_rank_is_the_whole_tree_and_attaches_nothing(trees) -> None:
    """`world=1` is the control column and has to be the *same* code path, not a branch: no division
    by anything but one, no collective, and every split name a no-op."""
    plan = ShardPlan.build(trees.cfg, 0, 1)
    assert plan.reduce is None and make_all_reduce(1) is None
    assert plan.gather is None and make_all_gather(1) is None
    assert attach_tp(trees.whole.model, plan) == 0
    assert getattr(trees.whole.model, "tp", None) is None

    # every collective site answers `None` rather than a no-op closure, so the host forward is the
    # host forward with no branch taken
    sites = [
        module
        for module in trees.whole.model.modules()
        if type(module).__name__ in ("Attention", "Indexer", "MoE")
    ]
    assert len(sites) > N_LAYERS, "the walk has to have found the modules it is checking"
    assert all(site.tp is None for site in sites)

    parameters = dict(trees.whole.model.named_parameters())
    for layer in range(N_LAYERS):
        for tail in split_names():
            name = f"layers.{layer}.{tail}"
            if name in parameters:
                assert torch.equal(plan.local_value(name, parameters[name]), parameters[name])


def test_the_plan_refuses_a_split_that_is_not_a_partition(trees) -> None:
    """Every division in `build` has to be exact, and the failures are named rather than left to a
    shape check forty layers into a load."""
    cfg = trees.cfg
    with pytest.raises(ValueError, match="do not split 3 ways"):
        ShardPlan.build(cfg, 0, 3)
    with pytest.raises(ValueError, match="outside a world"):
        ShardPlan.build(cfg, 2, 2)
    with pytest.raises(ValueError, match="not a split of anything"):
        ShardPlan.build(cfg, 0, 0)

    # 6 heads into 4 groups: `heads % groups` is the check that a rank's slice of the heads is a
    # whole number of groups, which `wo_a`'s block structure requires. It is *not* implied by the
    # divisions above -- 6 and 4 both divide by 2 -- and nothing else would catch it, because the
    # blocks would still have the right total width.
    awkward = SimpleNamespace(**{**cfg.__dict__, "n_heads": 6, "o_groups": 4})
    with pytest.raises(ValueError, match="o-groups"):
        ShardPlan.build(awkward, 0, WORLD)


# -- the cut itself -------------------------------------------------------------------------------


def test_the_shards_partition_the_tensor_they_were_cut_from(trees) -> None:
    """No gap and no overlap: put the ranks' pieces back and the file's tensor comes out.

    Equality with the original is the whole assertion, and it catches the failure a shape check
    cannot -- `wo_a` sliced by *rows* instead of by group has the right shape and takes an
    `o_lora_rank`-wide band across all `o_groups` of heads instead of two whole groups, which is a
    different tensor that a shape check is happy with.
    """
    parameters = dict(trees.whole.model.named_parameters())
    generator = torch.Generator().manual_seed(7)
    for layer in range(N_LAYERS):
        for tail in split_names():
            name = f"layers.{layer}.{tail}"
            if name not in parameters:
                continue
            source = torch.randn(parameters[name].shape, generator=generator)
            parts = [
                ShardPlan.build(trees.cfg, rank, WORLD).local_value(name, source) for rank in range(WORLD)
            ]
            rebuilt = _reassemble(tail, parts, trees.cfg)
            assert rebuilt.shape == source.shape, f"{name} came back {tuple(rebuilt.shape)}"
            assert torch.equal(rebuilt, source), f"{name} is not the concatenation of its shards"


def test_the_loader_cuts_the_file_the_way_the_constructor_cut_the_parameter(trees) -> None:
    """The pair that has to agree, parameter by parameter and name for name.

    One side is built: `Attention`/`Indexer`/`MoE` each divide their own widths by `world` in
    `__init__`. The other side is sliced: `checkpoint_weights` cuts the file's tensor with
    `ShardPlan.local_value`. Nothing in the tree compares them, and the failure mode is quiet --
    a rank holding 16 heads filled from heads 0-31 runs and produces numbers.
    """
    whole = dict(trees.whole.model.named_parameters())
    for rank, shard in enumerate(trees.shards):
        plan = ShardPlan.build(trees.cfg, rank, WORLD)
        loaded = dict(shard.model.named_parameters())
        assert set(loaded) == set(whole), "a sharded tree has to name the same parameters"
        cut = 0
        for name, parameter in loaded.items():
            local = plan.local_value(name, whole[name])
            assert torch.equal(parameter, local), (
                f"rank {rank}'s {name} is {tuple(parameter.shape)} and the loader's cut of "
                f"{tuple(whole[name].shape)} is {tuple(local.shape)}"
            )
            cut += tuple(parameter.shape) != tuple(whole[name].shape)
        # a load that sharded nothing would pass every equality above, so the count is checked too
        assert cut == len(split_names()) * N_LAYERS - (N_LAYERS - len(MINI["index_source_layers"]))


def test_a_sharded_forward_is_the_unsharded_forward(trees) -> None:
    """Four ranks with a real all-reduce against one rank, on the same tokens.

    The whole model is the oracle and it is the strongest one available: it was loaded from the same
    file with every division by one, and the sharded tree reproduces its logits position by position
    -- prefill and then stepwise, so the compressed-key cache and the indexer's are both exercised.

    What this does *not* compare is the indexer's selected blocks, which the module keeps local. A
    flipped selection would show up here as a logits difference rather than as a discrete mismatch,
    which is a weaker statement than `probe_tp4_block.py` can make with a model whose indexer actually
    truncates.
    """
    board = _MessageBoard(WORLD)
    for rank, shard in enumerate(trees.shards):
        plan = ShardPlan.build(trees.cfg, rank, WORLD, reduce=board.reduce_for(rank))
        sites = [
            module
            for module in shard.model.modules()
            if type(module).__name__ in ("Attention", "Indexer", "MoE")
        ]
        # a plan that reached no module would look exactly like a plan that worked, and the run
        # would be four ranks each computing a quarter of the answer and printing it
        assert attach_tp(shard.model, plan) == len(sites)
        assert len(sites) == 2 * N_LAYERS + len(MINI["index_source_layers"])
        assert plan.reduce is not None

    tokens = [5, 3, 17, 11]
    trees.whole.reset_state(1)
    _, want, _ = trees.whole(torch.tensor([tokens]), 0)

    # the dtype the row-parallel partials are rounded to, which is what `_close` bounds against
    dtype = trees.whole.model.layers[0].attn.wo_b.weight.dtype

    def prefill(rank: int):
        trees.shards[rank].reset_state(1)
        return trees.shards[rank](torch.tensor([tokens]), 0)[1]

    _close(want, board.run(prefill), "prefill", dtype)

    steps = [11, 61, 42, 8]
    for step, token in enumerate(steps):
        position = len(tokens) + step
        _, want, _ = trees.whole(torch.tensor([[token]]), position)
        got = board.run(
            lambda rank, t=token, p=position: trees.shards[rank](torch.tensor([[t]]), p)[1]
        )
        _close(want, got, f"decode step {step}", dtype)


def _close(want: torch.Tensor, got: list[torch.Tensor], what: str, dtype: torch.dtype) -> None:
    """Every rank against the oracle, within what the tree's own dtype costs.

    **The two paths are not the same arithmetic, and the difference is a rounding, not a split.**
    The split is exact and that is a separate test: `test_the_shards_partition_the_tensor_they_were_cut_from`
    rebuilds each tensor from its shards with `torch.equal`, and a scratch run of the tree in fp64
    gives `0.000e+00` between the whole forward and the sum of the sharded ones on the same
    activation and weights. What is left in bf16 is that a row-parallel projection is a *partial*
    here: `wo_b` and the shared expert's `w2` each round their own half to bf16 before the
    all-reduce sums the two, where the one-rank tree rounds the finished sum once. A logit therefore
    passes two extra roundings a layer, and the residual stream carries them to the head.

    That fixes the bound rather than leaving it to taste: `N_LAYERS` roundings of half an ulp each
    is `N_LAYERS * eps`, with `eps` read out of the dtype the projections are *stored* in -- not out
    of the logits, which the head narrows from an fp32 accumulation and which would put the bound
    six orders of magnitude below the thing it is bounding. It is not fitted to the measurement --
    the measurement is 9.2e-3 prefill and 2.9e-2 at the worst decode step, and both ranks land on
    the same digit every time, since the fixture is seeded and the two halves of a row-parallel
    projection are summed in rank order.

    A *wrong* split does not live in this margin. A `wo_a` sliced by rows instead of by group, an
    all-reduce that sums the routed experts twice, or an indexer scaled by a rank's head count
    instead of the global one are all errors of order one, and the exact tests above catch the ones
    that are also visible in the weights.
    """
    scale = want.abs().max().item()
    assert scale > 0, f"{what}: the oracle produced nothing to compare against"
    bound = N_LAYERS * torch.finfo(dtype).eps
    for rank, logits in enumerate(got):
        worst = (logits - want).abs().max().item() / scale
        assert worst <= bound, (
            f"{what}: rank {rank} is {worst:.3e} of a max |logit| of {scale:.3e} from the whole "
            f"tree, and {N_LAYERS} layers of {torch.finfo(dtype).eps:.3e} roundings allow "
            f"{bound:.3e}"
        )


# -- the indexer's other axis: query rows instead of index heads -----------------------------------


def test_the_row_split_replicates_one_parameter_and_moves_no_other_cut(trees) -> None:
    """`wq_b` is what the row split costs, and it is the whole of what it costs.

    A row band reads every index head of its own rows, so the weight cannot be cut however many ranks
    there are. Nothing else changes axis: `wq_a` is not in the table at all -- so `qr` is already full
    width on every rank -- and `weights_proj` is already computed at `index_n_heads` wide and sliced on
    the way out. Written as a comparison against the head split rather than against a literal, so a
    table edited in either direction fails here.
    """
    whole = dict(trees.whole.model.named_parameters())
    # built outside the patch, so this is the head split and the comparison below is a comparison of
    # two layouts rather than of a plan with itself. That holds because `build` is where the switch is
    # read: a plan answers the same way however long after it was made it is asked.
    head = ShardPlan.build(trees.cfg, 0, WORLD)
    # every split name of every layer, less the indexer's own on the layers that have no indexer --
    # counted for each rank, because the loop below re-checks the whole list on every rank and it is
    # the re-check that would catch a cut that is right on one rank and wrong on the next
    expected = WORLD * (len(split_names()) * N_LAYERS - (N_LAYERS - len(MINI["index_source_layers"])))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(INDEXER_ROW_SPLIT_ENV, "1")
        checked = 0
        for rank in range(WORLD):
            rows = ShardPlan.build(trees.cfg, rank, WORLD)
            assert rows.index_heads == head.index_heads, "the head count is not what moved"
            for layer in range(N_LAYERS):
                for tail in split_names():
                    name = f"layers.{layer}.{tail}"
                    if name not in whole:
                        continue
                    shape = tuple(whole[name].shape)
                    if tail == "attn.indexer.wq_b.weight":
                        assert rows.local_shape(name, shape) == shape, "a row band needs all the heads"
                        assert head.local_shape(name, shape) != shape, "and the head split does not"
                    else:
                        assert rows.local_shape(name, shape) == head.local_shape(name, shape), tail
                    checked += 1
        assert checked == expected


def test_a_row_band_is_the_whole_chunk_for_every_length_it_cannot_cut(trees) -> None:
    """The four lengths `row_band` is not a band for, and the one it is.

    A chunk that does not divide the world is not a corner to reason about but a length the split has
    nothing to say about, and the answer for it has to be "the head split", which works at any length.
    The gather is a fixed-shape call, so a ragged band would need a second code path for a case this
    model does not have. The bands that *are* taken have to cover the chunk exactly once, which is the
    property the gather's `cat` depends on for its order.
    """
    from src.models.deepseek_v4_1.attention import Indexer

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(INDEXER_ROW_SPLIT_ENV, "1")
        plans = [ShardPlan.build(trees.cfg, rank, WORLD) for rank in range(WORLD)]
        indexers = [
            Indexer(trees.cfg, MINI["index_source_layers"][0], 1, 64, world=WORLD) for _ in range(WORLD)
        ]
        for rank, indexer in enumerate(indexers):
            indexer.tp = plans[rank]
            # the module was built for the row split, so it holds the whole weight and `forward` takes
            # the head band out of it on the calls that need one -- and asserts the two agree
            assert indexer.row_split and indexer.n_heads == indexer.n_heads_global
            assert indexer.row_band(1) == (0, 1), "one query is below the world"
            assert indexer.row_band(3) == (0, 3), "three does not divide two"
            assert indexer.row_band(4) == (rank * 2, 2)
            assert indexer.row_band(2) == (rank, 1)
        covered = [
            row
            for row0, length in (indexers[rank].row_band(4) for rank in range(WORLD))
            for row in range(row0, row0 + length)
        ]
        assert covered == [0, 1, 2, 3], "the bands have to cover the chunk once, in rank order"

        # and the switch is a switch: with the split off, `world=4` is four bands of heads and the
        # indexer keeps every query row
        patch.setenv(INDEXER_ROW_SPLIT_ENV, "0")
        off = Indexer(trees.cfg, MINI["index_source_layers"][0], 1, 64, world=WORLD)
        off.tp = ShardPlan.build(trees.cfg, 1, WORLD)
        assert not off.row_split and off.n_heads == off.n_heads_global // WORLD
        assert off.row_band(4) == (0, 4)


def test_a_row_split_shard_is_the_unsharded_tree_row_for_row(trees) -> None:
    """The row split against the oracle, on the two things it claims.

    The *discrete* claim is the picks. Every rank computes the ids for its own rows and one all-gather
    puts them back in rank order, so the concatenation has to be the tensor the whole tree's indexer
    produced -- not a permutation of it. That is what a band off by a row fails, and it fails here
    rather than as a logit margin. The hook is on `Indexer.forward`'s return value, which is the
    gathered tensor: a rank's band is never visible outside the module.

    The *continuous* one is `_close`, at the same bound the head split is held to, because the row
    split changes nothing about `wo_b` or the shared expert -- still column-parallel partials rounded
    to bf16 before the all-reduce. What it does change is the indexer's score, which is now summed
    over all 32 heads in one rank, in the oracle's own order, instead of over 8 heads on each of four
    ranks and then rounded by a collective. The indexer is closer to the oracle here, not further.

    The decode steps in the loop are the *head* split: one query is below the world, so `row_band`
    hands back the whole chunk and the module takes its `_wq_b_for_heads` cut. That is deliberate --
    it is the layout that works at any length -- and it means both axes of one module are covered.
    """
    board = _MessageBoard(WORLD)

    def picks_of(loaded):
        """`Indexer.forward`'s return value, per index source, for the next forward only."""
        caught: dict[int, torch.Tensor] = {}
        handles = []
        for layer in loaded.model.layers:
            indexer = getattr(layer.attn, "indexer", None)
            if indexer is None:
                continue
            handles.append(
                indexer.register_forward_hook(
                    lambda _module, _inputs, output, lid=layer.layer_id: caught.__setitem__(
                        lid, output.detach().clone()
                    )
                )
            )
        return caught, handles

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(INDEXER_ROW_SPLIT_ENV, "1")
        shards = [
            load_backbone(
                trees.cfg,
                trees.ckpt,
                layout=trees.layout,
                hasher=_toy_hasher(trees.layout),
                world=WORLD,
                rank=rank,
            )
            for rank in range(WORLD)
        ]
        for rank, shard in enumerate(shards):
            plan = ShardPlan.build(
                trees.cfg, rank, WORLD, reduce=board.reduce_for(rank), gather=board.gather_for(rank)
            )
            sites = [
                module
                for module in shard.model.modules()
                if type(module).__name__ in ("Attention", "Indexer", "MoE")
            ]
            assert attach_tp(shard.model, plan) == len(sites)
            assert plan.gather is not None, "a row split with no gather cannot put the picks back"

        # the parameter the split costs, against the oracle's own tensor rather than against a shape
        whole = dict(trees.whole.model.named_parameters())
        for layer in MINI["index_source_layers"]:
            name = f"layers.{layer}.attn.indexer.wq_b.weight"
            for rank, shard in enumerate(shards):
                assert torch.equal(dict(shard.model.named_parameters())[name], whole[name]), (
                    f"rank {rank}'s {name} is not the file's whole weight"
                )

        tokens = [5, 3, 17, 11]
        trees.whole.reset_state(1)
        want_picks, handles = picks_of(trees.whole)
        _, want, _ = trees.whole(torch.tensor([tokens]), 0)
        for handle in handles:
            handle.remove()
        assert want_picks, "no indexer ran on the oracle, so there is nothing to compare picks to"

        dtype = trees.whole.model.layers[0].attn.wo_b.weight.dtype

        def prefill(rank: int):
            shards[rank].reset_state(1)
            picks, rank_handles = picks_of(shards[rank])
            try:
                return shards[rank](torch.tensor([tokens]), 0)[1], picks
            finally:
                for handle in rank_handles:
                    handle.remove()

        got = board.run(prefill)
        for rank, (logits, picks) in enumerate(got):
            assert set(picks) == set(want_picks), "the same layers have to have run their own indexer"
            for layer, picked in picks.items():
                assert picked.dtype == want_picks[layer].dtype
                assert torch.equal(picked, want_picks[layer]), (
                    f"rank {rank}'s layer {layer} picked {picked.tolist()} and the whole tree picked "
                    f"{want_picks[layer].tolist()}"
                )
        # the logits come out of the *tuples* `board.run` collected, without the picks attached
        _close(want, [logits for logits, _ in got], "row-split prefill", dtype)


# -- the routed experts, once a rank owns a share of them -----------------------------------------


class _DealtRoutedExperts(RoutedExperts):
    """A routed store that says it was dealt out, over the fixture's whole-expert store.

    `MoE.forward` reads two things off its store: a tensor, and the bit that says whether that
    tensor is one rank's share of the layer's routed sum or all of it. `DeviceRoutedExperts` is the
    only store that deals one out and it needs the packed checkpoint and a card, so what this varies
    is the bit: the same routed sum arrives either whole or cut into `world` equal shares whose sum
    is the whole, which is the arithmetic a real deal produces without the checkpoint to produce it
    from. A clone, because the whole-sum case is added into in place by the `MoE` above.
    """

    def __init__(self, inner: RoutedExperts, world: int, dealt: bool):
        self.inner = inner
        self.world = world
        self.partial = dealt

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        whole = self.inner.forward(x, weights, indices).clone()
        return whole / self.world if self.partial else whole


def test_the_ffn_completes_the_routed_partial_only_when_the_store_was_dealt_out(trees) -> None:
    """Which of the two all-reduce arrangements a layer's ffn uses, decided by the store and not by
    the split.

    Both arrangements are correct in their own configuration: reduce `routed + shared` when the
    routed store dealt the experts out, reduce `shared` alone when every rank computed the same
    routed sum. They are also a factor of `world` apart on the routed term, so taking the wrong one
    is an order-one error and not a rounding -- which is what makes this test worth its four extra
    forwards. Same fixture, same sharded trees, same message board, one flipped bit.
    """
    board = _MessageBoard(WORLD)
    for rank, shard in enumerate(trees.shards):
        attach_tp(shard.model, ShardPlan.build(trees.cfg, rank, WORLD, reduce=board.reduce_for(rank)))

    layer = 0
    whole_ffn = trees.whole.model.layers[layer].ffn
    ffns = [shard.model.layers[layer].ffn for shard in trees.shards]
    stores = [ffn.routed for ffn in ffns]

    x = torch.randn(3, trees.cfg.dim, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(3))
    # The inner `no_grad` is not redundant with the outer one: grad mode is thread-local, and the
    # board runs each rank on a thread of its own, which starts with it enabled. `load_backbone`
    # fills the tree inside `inference_mode`, so a parameter there is an inference tensor and a
    # call that builds a graph on one is an error rather than a cost.
    def step(rank: int) -> torch.Tensor:
        with torch.no_grad():
            return ffns[rank](x)

    with torch.no_grad():
        want = whole_ffn(x).clone()

    # the routed term is a factor of two apart between the two cases, against a difference between
    # two bf16 roundings of one value; anything in between is a margin, not a coincidence
    bound = 4 * torch.finfo(torch.bfloat16).eps
    for dealt in (False, True):
        for ffn, store in zip(ffns, stores):
            ffn.routed = _DealtRoutedExperts(store, WORLD, dealt)
        got = board.run(step)
        scale = want.abs().max().item()
        for rank, y in enumerate(got):
            worst = (y - want).abs().max().item() / scale
            assert worst <= bound, (
                f"dealt_out={dealt}: rank {rank} is {worst:.3e} of a max |ffn out| of "
                f"{scale:.3e} from the whole tree's ffn, so the routed partial was completed "
                f"{'wrongly' if dealt else 'twice'}"
            )

    # the fixture is module-scoped, so put the trees back the way the test found them
    for ffn, store in zip(ffns, stores):
        ffn.routed = store


def test_the_deal_is_the_same_partition_whichever_way_it_is_driven(trees) -> None:
    """`DeviceRoutedExperts._split` under one rank per process and under one process for all.

    Round-robin over the *global* expert ordering and `ranks` selecting from it, rather than dealing
    over `len(ranks)`: the two are the same for a one-process run and differ for a rank that owns
    one share, and only the first one comes out the same in every process. A process that dealt the
    layer from its own position would stage a different subset than its neighbours, each subset
    would be a plausible partial, and the all-reduce would return a number.

    Called unbound, on a stub, because the deal is arithmetic on two integers and everything else in
    the class needs a packed checkpoint and four cards to exist at all.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts

    ids = [30, 10, 60, 20, 50, 40]
    order = sorted(range(len(ids)), key=lambda slot: ids[slot])

    for world in (1, 2, 3, 4):
        # one rank per process, four processes: the pieces tile the routes, once each
        pieces: list[tuple[int, int]] = []
        for rank in range(world):
            cards = DeviceRoutedExperts._split(
                SimpleNamespace(world=world, ranks=[rank], deal="sorted"), ids
            )
            assert len(cards) == 1, "one rank per process drives one card, not a world of them"
            pieces += cards[0]
        assert sorted(pieces) == sorted(
            (position // world, slot) for position, slot in enumerate(order)
        )

        # one process for all of them: the same deal, one card per rank, 2/2/1/1 over four
        cards = DeviceRoutedExperts._split(
            SimpleNamespace(world=world, ranks=list(range(world)), deal="sorted"), ids
        )
        assert cards == [[(p // world, order[p]) for p in range(r, len(order), world)] for r in range(world)]

    # and the roster is what selects, not the order it is written in
    picked = DeviceRoutedExperts._split(SimpleNamespace(world=4, ranks=[2], deal="sorted"), ids)
    assert picked[0] == [(p // 4, order[p]) for p in (2,)]


def test_the_resident_set_is_this_cards_own_deal_counted() -> None:
    """`DeviceRoutedExperts._hot_rows`, checked against `_split` and against the layer's routing.

    An expert is resident iff the layer asks *this card* for it at least twice, and the deal hands a
    card a fixed set of columns of each row's sorted ids -- so the rule is a count over the card's
    own multiset and not over the layer's routing. Both halves of that are quiet when they are
    wrong. A set counted over the whole routing holds experts the card is never dealt: arena rows
    and page-locked bytes for a row nothing reads. A set counted off another rank's columns names
    arena rows nothing reads at all, and the cards that do read them each pay full price. The first
    is checked below against the deal `_split` really makes, and the second against a routing where
    the two rules provably differ.

    Called unbound, on a stub, the way the deal test above calls `_split`: this is arithmetic on a
    host tensor, and everything else in the class needs the packed checkpoint and four cards.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts

    world, topk, n_experts = 4, 6, 512

    def stub(rank: int, hot_rows: int) -> SimpleNamespace:
        # `deal` pinned to the default: this file's expectations are the sorted deal's round-robin
        # columns, and `test_models_deepseek_v4_1_expert_deal.py` is where the id deal is checked.
        return SimpleNamespace(
            world=world,
            topk=topk,
            ranks=[rank],
            deal="sorted",
            hot_rows=hot_rows,
            capped_layers=0,
            capped_rows=0,
        )

    # The control column first: zero is not a small set, it is no set, and it must count nothing --
    # including the cap counters, which are what a run reports its set as having been cut by.
    blank = torch.zeros((2, topk), dtype=torch.int64)
    for rank in range(world):
        empty = stub(rank, 0)
        assert DeviceRoutedExperts._hot_rows(empty, blank, 0) == []
        assert (empty.capped_layers, empty.capped_rows) == (0, 0)

    generator = torch.Generator().manual_seed(11)
    # 128 rows of six ids off 512 experts: enough rows that a card's draws repeat, few enough
    # experts that the rule has a tail to leave out -- the shape the sweep was measured on.
    whole = torch.randint(0, n_experts, (128, topk), generator=generator)
    ordered = whole.sort(dim=1).values

    def dealt(rank: int) -> list[int]:
        """This rank's draws, read off `_split` rather than off the columns the rule reads."""
        who = stub(rank, 0)
        out: list[int] = []
        for row in whole:
            ids = [int(e) for e in row]
            out += [ids[slot] for _, slot in DeviceRoutedExperts._split(who, ids)[0]]
        return out

    for rank in range(world):
        counts = Counter(dealt(rank))
        want = sorted(e for e, n in counts.items() if n >= 2)
        assert want, "the fixture's deal does not repeat and this test would count nothing"

        roomy = stub(rank, len(want))
        assert DeviceRoutedExperts._hot_rows(roomy, ordered, 0) == want
        assert (roomy.capped_layers, roomy.capped_rows) == (0, 0), (
            "an arena exactly the width of the rule's own set is not a cap, so a run sized this way "
            "must not report its set as having been cut"
        )

        # A cap is a truncation of the same rule rather than a second one: it keeps the hottest, it
        # counts itself, and a larger arena can then only stage fewer rows.
        cap = max(1, len(want) // 2)
        cut = stub(rank, cap)
        got = DeviceRoutedExperts._hot_rows(cut, ordered, 0)
        assert len(got) == cap
        assert (cut.capped_layers, cut.capped_rows) == (1, len(want) - cap)
        assert set(got) < set(want)
        dropped = {e: n for e, n in counts.items() if n >= 2 and e not in set(got)}
        assert min(counts[e] for e in got) >= max(dropped.values()), (
            "the cap dropped an expert drawn more often than one it kept, which is what would make "
            "a larger arena not a monotone step towards the rule"
        )

    # Two rows where the two rules provably differ. Expert 5 is drawn twice by rank 1 and expert 1
    # once by each of two ranks, so a rule that counted the layer's routing -- `{1, 5}` below --
    # would keep both while a rule that counts a card's keeps only the one with a second draw to
    # hit. Expert 1 is asked twice and neither of the cards that asked should hold it.
    hand = [[0, 1, 2, 3, 4, 5], [5, 1, 12, 13, 14, 15]]
    twice = Counter(e for row in hand for e in row)
    assert {e for e, n in twice.items() if n >= 2} == {1, 5}, "the fixture is not the case it exists for"

    small = torch.tensor([sorted(row) for row in hand], dtype=torch.int64)
    assert [DeviceRoutedExperts._hot_rows(stub(r, 8), small, 0) for r in range(world)] == [
        [],
        [5],
        [],
        [],
    ], "the resident sets are not each card's own draws, so a rank is holding an arena row it is never dealt or missing one it is dealt twice"


def test_a_row_names_the_arena_row_of_every_route_it_was_dealt() -> None:
    """`DeviceRoutedExperts._stage_row`'s arena arithmetic, which no counter and no shape reports.

    A resident route has to name the row `_fill` wrote its expert into, and a route that missed has
    to name a row of the arena's *tail* -- `hot_rows` and up, not `len(hot)` and up, because the tail
    is a fixed place in a fixed-height arena so that every layer's misses start it at the same
    offset. Both are off-by-`hot_rows` errors that hand the kernel another expert's weights: a wrong
    answer with the right shape, in a kernel that reads bare pointers. `_upload` copies each miss
    into the arena row it names, so the row index and the bytes that row will read are one fact.

    The pool adds a third answer beside the set and the tail, and it is asked with a key that has to
    carry the layer: the weights are `layers.{layer}.ffn.experts.{expert}.{which}.weight`, so a pool
    holding another layer's expert under the same id must not answer this layer's draw. That was the
    bug the 512-token sweep found (`pool_row`'s own docstring has the numbers) and the last two cases
    below are its regression test, at the level where it is decidable without four cards.

    Checked hand in hand with the miss list, because the number of rows a row is dealt is also what
    `drawn_rows` counts -- the denominator the coverage of a resident set is reported over, which is
    the number that was wrong the first time this was measured.

    Called on a stub, the way the two tests above are: the split, the map and the counters are host
    arithmetic, and everything else in the class needs the packed checkpoint and four cards. The stub
    binds `_resolve_row`/`_stage_misses` and not only `_stage_row`, so what it drives is the same
    two halves the batched path runs for a whole batch -- the resolution that must move nothing and
    the staging that must happen exactly once a row.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts, ResidentSet

    world, topk, hot_rows = 4, 6, 3
    # Sorted: 10(slot1), 20(slot3), 30(slot0), 40(slot5), 50(slot4), 60(slot2), dealt round-robin
    # over four ranks, so this rank's two routes are 20 and 60 and the siblings hold the other four.
    ids = [30, 10, 60, 20, 50, 40]
    layer_id = 7
    staged: list[list[tuple[int, int]]] = []
    # The two ends of `_stage_misses`, recorded as separate lists because they are separately
    # skipped: a row with nothing to move takes no buffer *and* stages nothing, and the wait a
    # buffer carries is the half of a pool hit's cost that `staged` alone cannot show.
    takes: list[int] = []

    def pool(held: dict[tuple[int, int], int], free: list[int], width: int = 4) -> SimpleNamespace:
        """A `ResidentSet` carrying only what the pool's own arithmetic reads.

        The real `pool_row` is bound to it rather than re-spelled here, so the eviction order, the
        counters and the key are the shipped ones. `free` is given in pop order -- `pool_free.pop()`
        takes the last -- and the rows are the arena's pool band, `rows_per_card + hot_rows` and up,
        which with no rows-per-card is `hot_rows`.
        """
        one = SimpleNamespace(
            pool_rows=width,
            pool_staged=0,
            pool_evicted=0,
            pool_free=[list(free)],
            pool_map=[dict(held)],
            pool_lru=[OrderedDict((key, None) for key in held)],
        )
        one.pool_row = lambda card, key: ResidentSet.pool_row(one, card, key)
        return one

    def stub(
        rank: int, residents: list[int], hot: int = hot_rows, pooled: SimpleNamespace | None = None
    ) -> SimpleNamespace:
        one = SimpleNamespace(
            world=world,
            topk=topk,
            ranks=[rank],
            deal="sorted",
            hot_rows=hot,
            rows=0,
            drawn_rows=0,
            expert_rows=0,
            layer_id=layer_id,
            # The map `_fill` leaves behind, rebuilt here from the ids it filled, in arena order.
            _hot_map=[{expert: row for row, expert in enumerate(residents)}],
            _residents=pooled,
        )
        one._split = lambda ids: DeviceRoutedExperts._split(one, ids)
        one._take_buffer = lambda: takes.append(1) or 0
        # The shipped composition, with only its two ends stubbed: `_take_buffer` does not have to
        # wait on an event that a stub never recorded, and `_stage` records the triples instead of
        # copying the checkpoint's bytes. Everything between them -- `_resolve_row`'s arena
        # arithmetic and `_stage_misses`' buffer, counters and H2D -- is the real thing, which is
        # what the batched path shares and what must not drift under it.
        one._resolve_row = lambda ids_row: DeviceRoutedExperts._resolve_row(one, ids_row)
        # One card in `ranks`, so the row's staging is `misses[0]`: `(miss row, route slot, arena
        # row)` triples, the arena row being where `_upload` will put those bytes.
        one._stage = lambda buffer, ids, misses: staged.append(misses[0])
        one._upload = lambda buffer, misses: None
        one._stage_misses = lambda ids, misses: DeviceRoutedExperts._stage_misses(one, ids, misses)
        one._pool_row = lambda card, expert: DeviceRoutedExperts._pool_row(one, card, expert)
        return one

    # Expert 20 is resident in arena row 1 and 60 is not, so the row is dealt one of each. The hit
    # names row 1 -- neither the miss row nor the row of the other resident expert -- and the miss
    # names `hot_rows` and not 0, which is the offset the whole tail hangs on.
    one = stub(1, [10, 20])
    got = DeviceRoutedExperts._stage_row(one, None, None, torch.tensor(ids))
    assert one.drawn_rows == 2, "a row of two routes is two draws, however many of them missed"
    assert got == [[(1, 3), (hot_rows, 2)]], (
        "the row's routes do not name the arena rows it staged into: 20 is the second resident "
        "expert and 60 is the row's first miss, so the kernel would read 10's weights for 20 or the "
        "resident block for a miss"
    )
    assert staged == [[(0, 2, hot_rows)]], (
        "the pointer written for expert 60 is not its own arena row, so `_upload` and the indices "
        "handed to the kernel disagree about where the bytes are"
    )

    # With no set the arena is the row's own deal, and the tail is what the misses ride on: the same
    # routes off the same empty map name rows 3 and 4 when `hot_rows` is 3 and rows 0 and 1 when it
    # is zero -- which is the offset, stated as the one number it is.
    tail = stub(1, [])
    assert DeviceRoutedExperts._stage_row(tail, None, None, torch.tensor(ids)) == [
        [(hot_rows, 3), (hot_rows + 1, 2)]
    ], "an empty resident map moved the deal's own rows, or the tail is not the offset it is"
    assert staged[-1] == [(0, 3, hot_rows), (1, 2, hot_rows + 1)], (
        "with no set the whole deal is staged, once each, into the rows the kernel was handed"
    )

    none = stub(1, [], hot=0)
    assert DeviceRoutedExperts._stage_row(none, None, None, torch.tensor(ids)) == [[(0, 3), (1, 2)]], (
        "zero resident rows is not a small set, it is the configuration before any of this existed, "
        "so its arena rows have to be the deal's own"
    )

    # The pool is asked after the set and before the tail, and it may answer without a stage: expert
    # 60 is this layer's and the pool holds it in row 3, so the route names row 3 and the row stages
    # nothing at all -- which is the whole of what the mechanism buys. The set still answers first,
    # so expert 20 names row 1 rather than anything the pool has.
    before, buffers = len(staged), len(takes)
    hit = stub(1, [10, 20], pooled=pool({(layer_id, 60): 3}, free=[4, 5, 6]))
    assert DeviceRoutedExperts._stage_row(hit, None, None, torch.tensor(ids)) == [[(1, 3), (3, 2)]], (
        "a pooled route did not name the row the pool holds it in, or the pool was asked before the "
        "set and took a draw the set had already answered"
    )
    assert len(staged) == before and len(takes) == buffers, (
        "a pool hit staged, which is the copy this mechanism exists to not pay -- and it must not "
        "take a buffer either, or the wait that buffer has been holding covers no staging at all"
    )
    assert hit._residents.pool_staged == 0 and hit.drawn_rows == 2

    # The key carries the layer. The pool below holds layer 6's expert 60 in the row a layer 7 draw
    # would have been given, and the draw is *not* answered: it takes the next free row and stages
    # into it. Keyed on the id alone -- which is what shipped first, and what the 512-token sweep
    # caught -- this route would have named row 3 and read another layer's bytes for the expert.
    other = pool({(layer_id - 1, 60): 3}, free=[5, 6])
    miss = stub(1, [10, 20], pooled=other)
    assert DeviceRoutedExperts._stage_row(miss, None, None, torch.tensor(ids)) == [[(1, 3), (6, 2)]], (
        "another layer's expert under the same id answered this layer's draw, so the route names the "
        "row of a tensor that belongs to a different layer"
    )
    assert staged[-1] == [(0, 2, 6)], (
        "the draw that the pool's other-layer key must not answer was not staged into its own row"
    )
    assert (layer_id, 60) in other.pool_map[0] and (layer_id - 1, 60) in other.pool_map[0], (
        "the pool did not record the key it was asked with, so it cannot be telling two layers' "
        "experts apart in the first place"
    )

    # And an eviction, which is the other way a wrong row could be handed over: one row of pool, no
    # free rows, holding layer 6's expert 60. The layer 7 draw takes that row -- the eviction is
    # allowed, the bytes in it are overwritten -- and the entry it replaces is gone rather than
    # shadowed by a key that differs only in its layer.
    squeezed = pool({(layer_id - 1, 60): 3}, free=[], width=1)
    evicted = stub(1, [10, 20], pooled=squeezed)
    assert DeviceRoutedExperts._stage_row(evicted, None, None, torch.tensor(ids)) == [[(1, 3), (3, 2)]]
    assert squeezed.pool_evicted == 1 and squeezed.pool_staged == 1, (
        "a pool at its width did not evict, so the row it handed over is one the map still believes "
        "holds the key it took the row from"
    )
    assert list(squeezed.pool_map[0]) == [(layer_id, 60)], (
        "the evicted key survived the key that replaced it"
    )

    # And every rank's own share, summed, is every route the layer made -- the other half of what
    # `drawn_rows` is reported over.
    across = [stub(rank, [], hot=0) for rank in range(world)]
    for one in across:
        DeviceRoutedExperts._stage_row(one, None, None, torch.tensor(ids))
    assert [one.drawn_rows for one in across] == [2, 2, 1, 1]
    assert sum(one.drawn_rows for one in across) == topk


def test_the_stage_copies_nothing_and_the_upload_reads_the_bank(monkeypatch) -> None:
    """A miss is one `packed` tensor a projection a kind, into the arena row the row named.

    This is the pair that replaced the pinned staging arena, and both halves are asserted here
    because either one alone can look right while the mechanism is gone. `_stage` must ask the
    checkpoint for **nothing** -- no key, no tensor -- since a staging call that touches the bank is
    a copy, and the copy is what made every byte cross host DRAM three times (read the bank, write
    the pinned row, DMA read the pinned row) instead of once. `_upload` must ask `packed` for exactly
    the six (projection, kind) pairs of every miss and put each one in the arena row that miss named,
    which is the fact the kernel reads through a bare pointer -- wrong, it is a right-shaped answer
    out of another expert's weights.

    The arena rows here are neither the pool's nor the pool-off tail's but a hand-drawn pair, `4` and
    `6`, because the two configurations differ only in which rows those are: with the pool off they
    are `hot_rows .. hot_rows + k` and with it on they are whatever `pool_row` handed back, and both
    are the same loop. A contiguous block is covered by the empty-set case above; what this covers is
    that a row that is *not* the miss's index still receives the right bytes.

    Called on a stub, and the stub is the two stream calls and nothing else: `_stage`, `_upload`,
    `_key` and `scale_key` are the shipped ones, and the three CUDA entry points `_upload` uses are
    replaced with no-ops so this runs with no card in the machine.
    """
    import contextlib

    from src.models.deepseek_v4_1 import device_experts
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts
    from src.models.deepseek_v4_1.loader import scale_key

    kinds = (("w1", "q"), ("w1", "s"), ("w2", "q"), ("w2", "s"), ("w3", "q"), ("w3", "s"))
    rows, width, layer_id = 8, 8, 7

    class Bank:
        """`Checkpoint.packed` as `_upload` uses it: one tensor a key, and a record of every key.

        The bytes are the ordinal of the call, so where they land says which key they came from
        without the test having to predict the order the six pairs are asked in.
        """

        def __init__(self) -> None:
            self.asked: list[str] = []

        def packed(self, key: str) -> torch.Tensor:
            self.asked.append(key)
            return torch.full((1, width), len(self.asked), dtype=torch.uint8)

    class Stream:
        """The copy stream, seen only through what `_upload` does to it: wait, and order behind."""

        def __init__(self) -> None:
            self.waited: list[object] = []

        def wait_stream(self, other: object) -> None:
            self.waited.append(other)

    class Event:
        """The one thing `_upload` leaves behind for `_take_buffer`: that the copies were issued."""

        def __init__(self) -> None:
            self.recorded = 0

        def record(self, stream: object) -> None:
            self.recorded += 1

    opened: dict[object, Stream] = {}
    monkeypatch.setattr(
        device_experts, "_copy_stream", lambda device: opened.setdefault(device, Stream())
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "compute")

    bank = Bank()
    arena = [{kind: torch.zeros((rows, width), dtype=torch.uint8) for kind in kinds}]
    events = [[Event(), Event()]]
    one = SimpleNamespace(
        layer_id=layer_id,
        devices=[torch.device("cpu")],
        _carried={},
        _uploaded=[{}],
        _events=events,
        expert_rows=0,
        _on_device=arena,
        checkpoint=bank,
    )
    one._key = lambda expert, which: DeviceRoutedExperts._key(one, expert, which)
    one._scale_key = scale_key
    one._stage = lambda buffer, ids, misses: DeviceRoutedExperts._stage(one, buffer, ids, misses)
    one._upload = lambda buffer, misses: DeviceRoutedExperts._upload(one, buffer, misses)

    # Sorted: 10(slot1), 20(slot3), 30(slot0), 40(slot5), 50(slot4), 60(slot2). This rank's row is
    # dealt 10 and 60, and the two misses name arena rows 4 and 6 -- so the second miss's bytes must
    # reach row 6 even though it is the first miss's index plus two.
    ids = [30, 10, 60, 20, 50, 40]
    misses = [[(0, 1, 4), (1, 2, 6)]]

    one._stage(0, ids, misses)
    assert bank.asked == [], (
        "the staging call read the checkpoint, so the copy the bank exists to delete is back: every "
        "byte would cross host DRAM twice more than it has to"
    )
    assert one.expert_rows == 2, "the misses are the draws the resident set and the pool did not answer"

    one._upload(0, misses)
    assert one.expert_rows == 2, "the upload recounted the row instead of counting what it moved"
    assert one._carried == {}, "the row's ids were left parked under the buffer for the next row"

    for card in range(1):
        for (which, kind), target in arena[card].items():
            for slot, arena_row, expert in ((1, 4, 10), (2, 6, 60)):
                weight = f"layers.{layer_id}.ffn.experts.{expert}.{which}.weight"
                key = weight if kind == "q" else scale_key(weight)
                assert key in bank.asked, (
                    f"{kind} of {which} for expert {expert} was never asked of the checkpoint, so "
                    f"the kernel will read whatever the arena row held before"
                )
                ordinal = bank.asked.index(key) + 1
                assert target[arena_row].tolist() == [ordinal] * width, (
                    f"{key}'s bytes did not land in arena row {arena_row}, which is the row the "
                    f"kernel is handed for expert {expert}"
                )
            untouched = [row for row in range(rows) if row not in {4, 6}]
            assert all(int(target[row].abs().sum()) == 0 for row in untouched), (
                "an upload wrote outside the arena rows its misses named"
            )
    assert len(bank.asked) == 2 * len(kinds), (
        "the upload moved an expert's whole six tensors a miss, or moved some of them twice"
    )
    assert events[0][0].recorded == 1, (
        "the buffer's event was not recorded, so `_take_buffer` cannot tell the copies were issued"
    )

    # And the pair cannot be taken apart: `_upload` with no `_stage` before it has no ids and no way
    # to name an expert, and it has already been asked for the buffer by `_take_buffer`.
    with pytest.raises(RuntimeError, match="no staging call claimed"):
        one._upload(0, misses)


def test_a_chunk_is_cut_where_a_later_row_would_re_draw_an_arena_row() -> None:
    """`DeviceRoutedExperts._chunk_bounds`' rule -- the one thing the batched path can be wrong by.

    A batched call reads each arena row it is handed as one expert's bytes *for the whole call*,
    where the per-row path reads a row once and moves on. So a batch is the same computation as the
    rows it replaces only while every arena row it names still holds what it held when the row that
    named it was staged -- and the pool is what breaks that, because a draw that misses can evict a
    row an earlier row of the same batch is still going to read. Wrong, it is a right-shaped answer
    out of the wrong expert's weights, in a kernel that reads them through a bare pointer.

    The cut is decidable because `_forward_chunked` resolves the whole batch before it stages any of
    it: the batch's claim on the pool is a property of the routing and not of a race, so the rule is
    a pure function of the resolve lists and is checked as one here. Hand-built entries rather than a
    pool driven through `_resolve_row`, because what is under test is where the line falls and not
    how a row comes to be on one side of it -- the pool would have to be made to evict on cue.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts

    one = SimpleNamespace(ranks=[0])
    two = SimpleNamespace(ranks=[0, 1])

    def row(*arena_rows: int) -> tuple:
        """One resolved row on one card whose draws all missed, into the arena rows given.

        Both halves of what `_resolve_row` returns are filled and they agree, because the rule reads
        both: `cards` is what the row *reads* -- every route names an arena row, hit or miss -- and
        `misses` is the subset of it that has to be moved in first. A fixture that filled only
        `misses` would be a row that reads nothing, which no row is.
        """
        return (
            [],
            [[(arena, i) for i, arena in enumerate(arena_rows)]],
            [[(i, 0, arena) for i, arena in enumerate(arena_rows)]],
            len(arena_rows),
        )

    def hit(*arena_rows: int) -> tuple:
        """One resolved row whose draws all hit the pool: it reads rows and stages none of them."""
        return ([], [[(arena, i) for i, arena in enumerate(arena_rows)]], [[]], len(arena_rows))

    # Nothing to cut. A batch whose draws were all pool hits staged nothing, and one whose misses are
    # into rows no other row of the batch names is reading what it wrote -- both are one call, which
    # is the case the whole mechanism is for.
    assert DeviceRoutedExperts._chunk_bounds(one, [row(), row(), row()]) == [(0, 3)], (
        "a batch that moved nothing was cut, which is exactly the batch this exists for"
    )
    assert DeviceRoutedExperts._chunk_bounds(one, [row(4), row(5), row(6)]) == [(0, 3)], (
        "disjoint misses were cut: the pool hands out fresh rows, so a row taking one is not reading "
        "another row's"
    )

    # The case the rule is for: row 2 draws an expert the pool puts back in arena row 4, which row 0
    # staged, so one call over both would apply row 0's bytes to row 2's token.
    assert DeviceRoutedExperts._chunk_bounds(one, [row(4), row(5), row(4)]) == [(0, 2), (2, 3)], (
        "a row that re-draws an arena row an earlier row of the batch staged was left in the same "
        "call as that row, which reads the wrong expert for whichever of the two it is not holding"
    )
    # A row cannot clash with itself, and every row after the first here clashes with the one before.
    assert DeviceRoutedExperts._chunk_bounds(one, [row(4, 5), row(4), row(4)]) == [(0, 1), (1, 2), (2, 3)], (
        "a row's own two misses were treated as a clash, or two rows that both re-draw row 4 were "
        "not"
    )
    # A row that hit the pool moves nothing, so it is free to sit in the chunk -- but it still *reads*
    # the row it was pooled into, and a later row's miss can be handed that same row. A claim built
    # from the misses alone sees a row with nothing to stage and lets it sit beside the row that is
    # about to overwrite what it reads, and the failure is silent: the right shapes out of the wrong
    # expert's weights. Both halves of the resolve entry are therefore claims, which is why `row()` and
    # `hit()` above fill `cards` and `misses` consistently.
    assert DeviceRoutedExperts._chunk_bounds(one, [hit(4), row(5), row(4)]) == [(0, 2), (2, 3)], (
        "a row whose draws were all pool hits was not counted as a claim on the arena, so a later "
        "row's miss can be handed the row it is reading"
    )
    # And the reverse is not a clash: a hit row reads rows that no later row claims.
    assert DeviceRoutedExperts._chunk_bounds(one, [hit(4), row(5), row(6)]) == [(0, 3)], (
        "a hit row's read cut a batch that never recycled the row it reads"
    )

    # The rule is per card. A clash on the second card cuts even when the first card's rows are clean,
    # because the two cards' kernels are two calls -- but a clash on card 0 does not by itself make
    # card 1's rows clash, and the cut is on the row and not on the card.
    def pair(first: list[int], second: list[int]) -> tuple:
        return (
            [],
            [
                [(arena, i) for i, arena in enumerate(first)],
                [(arena, i) for i, arena in enumerate(second)],
            ],
            [
                [(i, 0, arena) for i, arena in enumerate(first)],
                [(i, 0, arena) for i, arena in enumerate(second)],
            ],
            len(first) + len(second),
        )

    assert DeviceRoutedExperts._chunk_bounds(two, [pair([4], []), pair([5], [9]), pair([6], [9])]) == [
        (0, 2),
        (2, 3),
    ], "a clash on the second card's arena did not cut, so that card's kernel would read its own row 9 twice"


def test_a_batched_pass_resolves_the_whole_batch_before_it_stages_any_of_it() -> None:
    """`DeviceRoutedExperts._forward_chunked`'s schedule, which is the row loop's at a wider unit.

    Two things have to hold for the batched path to be the per-row path's answer and not a different
    one, and neither is a shape. **Every row is resolved before any row is staged**, in row order,
    because the pool's state after the batch has to be the state the row loop would have left it in
    -- and at `hot_rows > 0` more than that, since `_fill` chose which experts deserve a resident row
    from the whole batch's routing. **And the pipeline survives the wider unit**: a chunk's drain is
    entered after the *next* chunk has been issued, which is the only ordering in the class that is
    not free to change. **A row with nothing to move is not a stage**: it takes no buffer and issues
    no copy, so a batch the pool answers entirely neither waits nor writes, and the buffer rotation
    advances only over the rows that do -- which is what keeps a slot's copy a whole *staging* old.

    Counted on a stub whose resolve, chunk and stage are the real ones and whose issue and drain are
    recorded instead of run, since the point is the order of the calls and not what is on the cards.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts, ResidentSet

    world, topk, layer_id = 4, 6, 7

    def pool(held: dict[tuple[int, int], int], free: list[int], width: int) -> SimpleNamespace:
        out = SimpleNamespace(
            pool_rows=width,
            pool_staged=0,
            pool_evicted=0,
            pool_free=[list(free)],
            pool_map=[dict(held)],
            pool_lru=[OrderedDict((key, None) for key in held)],
        )
        out.pool_row = lambda card, key: ResidentSet.pool_row(out, card, key)
        return out

    def stub(pooled: SimpleNamespace | None) -> tuple[SimpleNamespace, list, list, list]:
        staged: list = []
        takes: list = []
        calls: list = []
        one = SimpleNamespace(
            world=world,
            topk=topk,
            ranks=[1],
            deal="sorted",
            hot_rows=0,
            layer_id=layer_id,
            partial=True,
            rows=0,
            drawn_rows=0,
            expert_rows=0,
            _hot_map=[{}],
            _residents=pooled,
        )
        one._split = lambda ids: DeviceRoutedExperts._split(one, ids)
        one._pool_row = lambda card, expert: DeviceRoutedExperts._pool_row(one, card, expert)
        one._resolve_row = lambda ids_row: DeviceRoutedExperts._resolve_row(one, ids_row)
        one._take_buffer = lambda: takes.append(1) or 0
        one._stage = lambda buffer, ids, misses: staged.append(misses[0])
        one._upload = lambda buffer, misses: None
        one._stage_misses = lambda ids, misses: DeviceRoutedExperts._stage_misses(one, ids, misses)
        one._chunk_bounds = lambda resolved: DeviceRoutedExperts._chunk_bounds(one, resolved)
        one._issue_chunk = lambda row0, row1, resolved, x, weights: (
            calls.append(("issue", row0, row1)) or [0]
        )
        one._drain_chunk = lambda row0, row1, issued, y: calls.append(("drain", row0, row1))
        return one, staged, takes, calls

    # Everything the two rows ask for is already in the pool, so the batch is one chunk that stages
    # nothing at all -- and a chunk that stages nothing is still a chunk: the resolve still ran, in
    # row order, and the pass still ends with exactly one drain.
    held = pool({(layer_id, 20): 2, (layer_id, 60): 3}, free=[4, 5], width=4)
    one, staged, takes, calls = stub(held)
    route = torch.tensor([[30, 10, 60, 20, 50, 40]] * 2)
    DeviceRoutedExperts._forward_chunked(one, None, None, route, None, 2)
    assert calls == [("issue", 0, 2), ("drain", 0, 2)], (
        "a batch that staged nothing was not one issue and one drain, so the unit is not the batch"
    )
    assert staged == [] and takes == [], (
        "a pool hit staged, which is the copy this mechanism exists to not pay -- and a run of hits "
        "must leave the buffer rotation where it found it, or the slot it hands out has a copy "
        "issued immediately before it with no staging in between to cover it"
    )
    assert one.drawn_rows == 4 and one.rows == 2

    # Two rows that want two different experts out of a pool with one free row: the second row's
    # draws evict what the first staged, so the batch is cut and the cut is what the calls show. The
    # drain of the first chunk is entered after the second chunk has been issued, which is the
    # pipeline -- a drain before its own issue would be a wait the row loop does not take either.
    squeezed = pool({(layer_id, 20): 2}, free=[4], width=2)
    one, staged, takes, calls = stub(squeezed)
    route = torch.tensor([[30, 10, 60, 20, 50, 40], [30, 10, 61, 21, 50, 40]])
    DeviceRoutedExperts._forward_chunked(one, None, None, route, None, 2)
    assert calls == [
        ("issue", 0, 1),
        ("drain", 0, 1),
        ("issue", 1, 2),
        ("drain", 1, 2),
    ], (
        "a batch whose second row re-draws the row the first staged was not cut, or the drain stopped "
        "being one chunk behind the issue"
    )
    assert staged == [[(0, 2, 4)], [(0, 3, 2), (1, 2, 4)]], (
        "the second row's evictions do not name the rows its own draws were put in, so the kernel "
        "would read the first row's experts"
    )
    assert len(takes) == len(staged) == 2, (
        "a buffer was taken for a row that moved nothing, or two staging rows shared one -- either "
        "way a slot's copy is not a whole staging old when it is waited on"
    )
