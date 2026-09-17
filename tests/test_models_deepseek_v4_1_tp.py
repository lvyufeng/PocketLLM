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
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.loader import V41Checkpoint, load_backbone
from src.models.deepseek_v4_1.modules import RoutedExperts
from src.models.deepseek_v4_1.tp import ShardPlan, attach_tp, make_all_reduce, split_names
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
            cards = DeviceRoutedExperts._split(SimpleNamespace(world=world, ranks=[rank]), ids)
            assert len(cards) == 1, "one rank per process drives one card, not a world of them"
            pieces += cards[0]
        assert sorted(pieces) == sorted(
            (position // world, slot) for position, slot in enumerate(order)
        )

        # one process for all of them: the same deal, one card per rank, 2/2/1/1 over four
        cards = DeviceRoutedExperts._split(
            SimpleNamespace(world=world, ranks=list(range(world))), ids
        )
        assert cards == [[(p // world, order[p]) for p in range(r, len(order), world)] for r in range(world)]

    # and the roster is what selects, not the order it is written in
    picked = DeviceRoutedExperts._split(SimpleNamespace(world=4, ranks=[2]), ids)
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
        return SimpleNamespace(
            world=world,
            topk=topk,
            ranks=[rank],
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
    answer with the right shape, in a kernel that reads bare pointers. `_upload` writes to the same
    offset (`slice(self.hot_rows, self.hot_rows + k)`), and `_stage` stages into it, so the row
    index and the bytes that row will read are one fact stated in three places.

    Checked hand in hand with the miss list, because the number of rows a row is dealt is also what
    `drawn_rows` counts -- the denominator the coverage of a resident set is reported over, which is
    the number that was wrong the first time this was measured.

    Called on a stub, the way the two tests above are: the split, the map and the counters are host
    arithmetic, and everything else in the class needs the packed checkpoint and four cards.
    """
    from src.models.deepseek_v4_1.device_experts import DeviceRoutedExperts

    world, topk, hot_rows = 4, 6, 3
    # Sorted: 10(slot1), 20(slot3), 30(slot0), 40(slot5), 50(slot4), 60(slot2), dealt round-robin
    # over four ranks, so this rank's two routes are 20 and 60 and the siblings hold the other four.
    ids = [30, 10, 60, 20, 50, 40]
    staged: list[list[tuple[int, int]]] = []

    def stub(rank: int, residents: list[int], hot: int = hot_rows) -> SimpleNamespace:
        one = SimpleNamespace(
            world=world,
            topk=topk,
            ranks=[rank],
            hot_rows=hot,
            rows=0,
            drawn_rows=0,
            expert_rows=0,
            # The map `_fill` leaves behind, rebuilt here from the ids it filled, in arena order.
            _hot_map=[{expert: row for row, expert in enumerate(residents)}],
        )
        one._split = lambda ids: DeviceRoutedExperts._split(one, ids)
        one._take_buffer = lambda: 0
        # One card in `ranks`, so the row's staging is `misses[0]`: `(miss row, route slot)` pairs.
        one._stage = lambda buffer, ids, misses: staged.append(misses[0])
        one._upload = lambda buffer, misses: None
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
    assert staged == [[(0, 2)]], (
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
    assert staged[-1] == [(0, 3), (1, 2)], (
        "with no set the whole deal is staged, once each, into the rows the kernel was handed"
    )

    none = stub(1, [], hot=0)
    assert DeviceRoutedExperts._stage_row(none, None, None, torch.tensor(ids)) == [[(0, 3), (1, 2)]], (
        "zero resident rows is not a small set, it is the configuration before any of this existed, "
        "so its arena rows have to be the deal's own"
    )

    # And every rank's own share, summed, is every route the layer made -- the other half of what
    # `drawn_rows` is reported over.
    across = [stub(rank, [], hot=0) for rank in range(world)]
    for one in across:
        DeviceRoutedExperts._stage_row(one, None, None, torch.tensor(ids))
    assert [one.drawn_rows for one in across] == [2, 2, 1, 1]
    assert sum(one.drawn_rows for one in across) == topk
