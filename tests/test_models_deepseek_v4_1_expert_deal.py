"""Contract tests for the V4.1 routed-expert deal: `sorted` and `id`.

The deal is which card owns which of a row's six routed experts. It is decided in `_split` and read
back in `_hot_rows`: the first says where a drawing is staged, the second says which experts count as
resident for a card, and nothing downstream re-derives it -- `_resolve_row` renumbers a row from the
pairs `_split` placed and `_issue`/`_upload` read the row they are handed. That is what makes those
two the whole contract, and it is also what makes it testable **without a card**: both are walks of
`self.deal`, `self.world`, `self.ranks` and `self.topk`, and the arena they fill is the only part of
the class that needs a device. The methods are therefore called unbound against a stub.

**The failure these pin is quiet.** A `_hot_rows` that walked a different deal from `_split` would
fill its resident map under keys the split never names, so a card would look up an expert it is
already holding, miss, and stage a row it did not need -- a plausible number and a wrong answer, with
no counter that reads wrong. So what is checked is the pair rather than either half: on a routing, the
experts `_hot_rows` returns for a card must be exactly the experts that card's own `_split` drawings
name, kept by the resident rule. The same file is the regression guard for `sorted`, the deal the
numbers taken before the default were measured on: it has to come out of this tree exactly as it
comes out of the tree before it, which is how a change to `rows_per_card` or to `_hot_rows`' columns
gets caught rather than measured.

**And what the deal costs.** `rows_per_card` is the width of a card's staging tail, so it is the one
number that says `id` is not free: `sorted` deals a card at most `ceil(topk / world)` of a row's
slots, while `id` can hand one card all `topk` of them, and the tests below construct a row that
makes both deals hit their own width exactly. That width is what `pool_rows` has to clear and what
every arena row above it -- `hot_rows` resident, `pool_rows` pooled -- is added to.

The balance the `id` deal is for is priced on the real routing in
`docs/performance/deepseek_v4_1_flash_device_experts.md`; what is asserted here is only the
structural half of it, that under `id` the six drawings of a row can all land on one card and that
the resident set of a saturated pool does not depend on which card it is.
"""

from __future__ import annotations

import inspect
import random

import pytest
import torch

from src.cli.generate_v41 import build_arg_parser
from src.models.deepseek_v4_1 import device_experts as de
from src.models.deepseek_v4_1.loader import load_backbone

# The geometry every number in the module docstring's deal section and on the device-experts page was
# taken at: six routed experts per row over four cards.
TOP_K = 6
WORLD = 4
RANKS = [0, 1, 2, 3]


class _Deal:
    """The four fields `_split` and `_hot_rows` read, for a class whose arenas need a card.

    Standing in for `DeviceRoutedExperts` rather than constructing one is the whole reason this file
    runs without a GPU: the constructor loads the extension and allocates `arena_rows` rows of expert
    bytes per card, and none of that is read by either method under test. The two counters are here
    because `_hot_rows` writes them when the arena is the ceiling and cuts the resident set.
    """

    def __init__(self, *, deal: str, hot_rows: int = 0, topk: int = TOP_K, world: int = WORLD):
        self.deal = deal
        self.hot_rows = hot_rows
        self.topk = topk
        self.world = world
        self.ranks = list(RANKS)
        self.capped_layers = 0
        self.capped_rows = 0


def _routing(seed: int, rows: int, n_experts: int = 384) -> list[list[int]]:
    """A gate's output: `rows` x `TOP_K` ids, repeats included on purpose."""
    rng = random.Random(seed)
    return [[rng.randrange(n_experts) for _ in range(TOP_K)] for _ in range(rows)]


def _hot(self_, ordered: torch.Tensor, card: int) -> list[int]:
    return sorted(de.DeviceRoutedExperts._hot_rows(self_, ordered, card))


def _split(self_, ids: list[int]):
    return de.DeviceRoutedExperts._split(self_, ids)


def test_the_id_deal_reads_the_expert_and_the_sorted_deal_reads_the_position():
    """`sorted` is a function of the drawing's place among the sorted ids, `id` of the id itself.

    Both halves matter. A `sorted` that read the id would make the deal depend on which experts a
    row happened to route to, so two runs to the same six experts would stage them into different
    rows -- the resident set's map and the pool's key are both per card, and a row's arena rows are
    what the next pass's probes name. An `id` that read the position would not be the deal at all.
    """
    assert de.deal_card(17, 3, deal="id", world=WORLD) == 17 % WORLD
    for position in range(TOP_K):
        assert de.deal_card(position, position, deal="sorted", world=WORLD) == position % WORLD
    # The expert's number does not enter `sorted`'s rule, and the position does not enter `id`'s.
    assert len({de.deal_card(expert, 2, deal="sorted", world=WORLD) for expert in range(50)}) == 1
    assert len({de.deal_card(17, position, deal="id", world=WORLD) for position in range(TOP_K)}) == 1


def test_an_unknown_deal_is_refused_and_the_environment_defaults_to_id(monkeypatch):
    """The default is the `id` deal, and an unreadable name is not silently one of the two.

    `DEAL_ENV` is the lever the A/B pulls, so a typo in it has to be a failed run rather than a
    quietly shipped run: the environment reads as `id` when it is unset or empty, and a name that is
    not a deal is refused by the two functions that would otherwise apply it. `sorted` still has to
    be reachable through the same lever, because it is the deal the recorded numbers were taken on
    and the one an A/B switches back to.
    """
    monkeypatch.delenv(de.DEAL_ENV, raising=False)
    assert de.deal_rule() == "id"
    for value in ("", "  ", "ID", "id "):
        monkeypatch.setenv(de.DEAL_ENV, value)
        assert de.deal_rule() == "id"
    for value in ("sorted", " SORTED "):
        monkeypatch.setenv(de.DEAL_ENV, value)
        assert de.deal_rule() == "sorted"
    monkeypatch.setenv(de.DEAL_ENV, "positions")
    assert de.deal_rule() == "id", "an unknown deal in the environment is not a third deal"
    assert de.DEALS == ("sorted", "id")
    for bad in ("positions", "", "ID"):
        with pytest.raises(ValueError):
            de.deal_card(0, 0, deal=bad, world=WORLD)
        with pytest.raises(ValueError):
            de.rows_per_card(bad, TOP_K, WORLD)


def test_the_width_a_deal_needs_is_the_widest_row_it_can_deal():
    """`rows_per_card` for both deals, and a row that makes each one reach its own width.

    A row whose six experts are all congruent mod `world` is legal and is what an `id` arena has to
    hold: rank `expert % world` is dealt all six drawings, so the staging tail has to be `topk` wide.
    The same row is the one that leaves `sorted` at `ceil(topk / world)`, because the split sorts
    first and deals positions -- so a shared arena's width cannot be justified by the routing.
    """
    assert de.rows_per_card("sorted", TOP_K, WORLD) == 2
    assert de.rows_per_card("id", TOP_K, WORLD) == TOP_K
    # The same rule read at the widths the arithmetic has to round at, including the degenerate ones.
    for topk, world in ((6, 4), (5, 4), (8, 4), (7, 3), (1, 4), (6, 1), (6, 6), (3, 2)):
        assert de.rows_per_card("sorted", topk, world) == -(-topk // world)
        assert de.rows_per_card("id", topk, world) == topk

    congruent = [WORLD, 2 * WORLD, 3 * WORLD, 4 * WORLD, 5 * WORLD, 6 * WORLD]
    for deal, widest in (("sorted", 2), ("id", TOP_K)):
        self_ = _Deal(deal=deal)
        cards = _split(self_, congruent)
        assert max(len(members) for members in cards) == widest
        assert de.rows_per_card(deal, TOP_K, WORLD) == widest


def test_every_route_slot_of_a_row_lands_on_exactly_one_card():
    """A row is partitioned by the deal, not merely covered by it or fanned out over it.

    The all-reduce sums the cards' partials, so a slot dealt to two cards is a layer counted twice and
    a slot dealt to none is a layer counted never -- neither of which a shape check can see.
    """
    for deal in ("sorted", "id"):
        self_ = _Deal(deal=deal)
        for seed in range(8):
            for ids in _routing(seed, 32):
                cards = _split(self_, ids)
                slots = sorted(slot for members in cards for _, slot in members)
                assert slots == list(range(TOP_K)), f"{deal} dropped or repeated a slot of {ids}"
                assert len(cards) == WORLD


def test_the_sorted_deal_fills_the_cards_unevenly_and_the_id_deal_does_not():
    """The reason a second deal exists, read off the resident set's widths.

    Six sorted slots over four cards is positions `0, 4` / `1, 5` / `2` / `3`: ranks 0 and 1 are dealt
    two columns of every row and ranks 2 and 3 one, so on a routing wide enough for the repeats to
    show, the two-slot cards are holding about twice the residents of the one-slot cards. That
    imbalance is what a chunk's expert H2D pays for twice -- once as the wider staged set and once as
    the all-reduce's rendezvous with the card that is still copying -- and it is structural rather
    than a property of this routing, because the columns are.

    Under `id` a card's drawings can only be experts congruent to its rank mod `world`, so its pool is
    a disjoint `n_experts / world` and every card's pool saturates the same way at the same width.
    The `hot_rows` here is above `n_experts / world` on purpose: the equality is the saturated case,
    where each card draws every expert of its own pool at least twice, and it is the case the arena
    is sized for.
    """
    routing = _routing(5, 200, 128)
    ordered = torch.tensor(routing, dtype=torch.int64).sort(dim=1).values

    sorted_widths = [len(_hot(_Deal(deal="sorted", hot_rows=148), ordered, card))
                     for card in range(WORLD)]
    assert min(sorted_widths[:2]) > max(sorted_widths[2:]), sorted_widths

    id_widths = [len(_hot(_Deal(deal="id", hot_rows=148), ordered, card))
                 for card in range(WORLD)]
    assert len(set(id_widths)) == 1, id_widths


def test_the_launcher_offers_the_deal_and_the_loader_takes_it():
    """The three places the deal is written, pinned so the flag cannot drift off the rule.

    `--expert-deal` defaults to `None` rather than to `"id"`, and that is the contract: `None` means
    ask `DEEPSEEK_V41_EXPERT_DEAL`, which is the lever the A/B pulls, so a launcher default of
    `"id"` would silently out-prioritise the environment and leave the recorded `sorted` numbers
    unreachable without a flag on every command line. The loader keyword has to exist and has to
    default to `None` for the same reason -- a run reaches the class through `load_backbone`, and a
    deal that is not threaded through there is a deal the flag cannot set.

    `format_help` is called because the help text names the rule, and a bare `%` in an argparse help
    string is a format spec, not a percent sign: it raises out of `--help` rather than printing
    `expert % world`. Both deal spellings are rendered here so that the check is on the text and not
    on the flag's existence.
    """
    parser = build_arg_parser()
    action = next(one for one in parser._actions if one.dest == "expert_deal")
    assert action.default is None
    assert tuple(action.choices or ()) == de.DEALS
    parser.parse_args(["--checkpoint", "x", "--expert-deal", "id"])
    for deal in de.DEALS:
        assert deal in parser.format_help() or deal in action.help
    assert "%%" not in parser.format_help(), "the help text is showing argparse's escape"
    assert inspect.signature(load_backbone).parameters["expert_deal"].default is None


@pytest.mark.parametrize("deal", ["sorted", "id"])
@pytest.mark.parametrize("hot_rows", [0, 4, 16, 148])
def test_the_resident_set_is_a_walk_of_the_same_deal_the_split_makes(deal, hot_rows):
    """The pair, on routings that include a single-expert row and a 24-expert world.

    `_hot_rows` is handed the layer's routing sorted along each row, because that sort is what makes
    `sorted`'s columns mean anything; the splits are read on the raw ids, which is what `_resolve_row`
    walks. The oracle counts the drawings each card's own split names -- so it is the same walk, read
    the other way -- keeps the experts drawn at least twice, and truncates to the arena the way the
    class truncates: by count, hottest kept.
    """
    for seed, n_experts, rows in ((1, 384, 64), (2, 384, 8), (3, 24, 64), (4, 6, 32), (5, 128, 200)):
        routing = _routing(seed, rows, n_experts)
        ordered = torch.tensor(routing, dtype=torch.int64).sort(dim=1).values
        self_ = _Deal(deal=deal, hot_rows=hot_rows)
        splits = [_split(self_, ids) for ids in routing]
        for card in range(WORLD):
            hot = _hot(self_, ordered, card)
            if hot_rows == 0:
                assert hot == [], "no arena rows are resident when the arena has none"
                continue
            counted: dict[int, int] = {}
            for index, members in enumerate(splits):
                for _, slot in members[card]:
                    counted[routing[index][slot]] = counted.get(routing[index][slot], 0) + 1
            repeated = {expert: count for expert, count in counted.items() if count >= 2}
            if len(repeated) <= hot_rows:
                assert hot == sorted(repeated), f"{deal} hot set is not the split's repeats"
            else:
                # The arena is binding and `topk` picks the survivors, so what is pinned is the
                # width and that the ones kept are the hottest -- the truncation stays monotone in
                # the arena, which is what makes the knob sweepable rather than a step function.
                assert len(hot) == hot_rows
                kept = sorted(repeated[expert] for expert in hot)
                assert kept == sorted(repeated.values())[-hot_rows:]
