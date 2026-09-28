"""Which card a rank runs on, and the one name U3 split to say it.

`--device` and `--device-ids` are two concepts that were one flag: the platform, and the cards. The
tests here are about the rule that reads the second, because it is the one every runtime shares and
the one a wrong answer costs a collective rather than a number -- and about the refusals, because
the old spelling is well formed and belongs to the other flag, which is the whole of the migration.
"""

from __future__ import annotations

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.backends import cli_surface
from pocketllm.backends.runtime_engine import card_for_rank, visible_card_count


def _args(**overrides) -> EngineArgs:
    base = dict(model="checkpoint", backend="cpp")
    base.update(overrides)
    return EngineArgs(**base)


# ------------------------------------------------------------------------- the card rule


def test_a_named_card_list_is_indexed_by_rank() -> None:
    """The rule `--device-ids` exists for: one flag names every rank's card.

    It replaces the pair a launcher used to write per rank -- `CUDA_VISIBLE_DEVICES=$rank` and
    `--device 0` beside it -- so rank 1 of `2,3` is card 3, and the flag itself does not have to be
    rewritten for each process the launcher starts.
    """
    assert card_for_rank((2, 3), rank=0, world=2) == 2
    assert card_for_rank((2, 3), rank=1, world=2) == 3
    assert card_for_rank((4, 5, 6, 7), rank=3, world=4) == 7
    # A single process names one card and gets it: this is `--device cuda:2` said the new way.
    assert card_for_rank((2,), rank=0, world=1) == 2


def test_an_unnamed_card_list_keeps_the_rule_every_runtime_had() -> None:
    """Nothing named: a single process takes card 0, a sharded one takes its rank.

    The supervisor hands every rank the same visible device list, so a sharded rank that took card 0
    would stack the whole world onto one GPU. A launcher that narrowed visibility to a single card
    per rank has already renumbered that card to 0, which is why the offset is not applied on top of
    it -- and that case is the migration's escape hatch, still correct after U3.
    """
    assert card_for_rank((), rank=0, world=1) == 0
    assert card_for_rank((), rank=2, world=4) == 2
    assert card_for_rank((), rank=2, world=4, visible=1) == 0
    assert card_for_rank((), rank=2, world=4, visible=4) == 2


def test_a_list_shorter_than_the_rank_is_refused_rather_than_clamped() -> None:
    """Rank 3 of a two-card list has no card, and card 1 would be someone else's.

    `EngineArgs` refuses this against the world before a runtime is constructed; this is the same
    statement for a caller whose rank the arguments never saw -- an EP group's, say.
    """
    with pytest.raises(ConfigurationError, match="rank 3 has none"):
        card_for_rank((2, 3), rank=3, world=4)


def test_the_visible_set_is_read_from_either_platforms_variable() -> None:
    """`None` and `0` are different answers: a variable nobody set is the whole host, and an empty
    one is no card at all -- which is how the suite runs a CPU-only test on a card-bearing host."""
    assert visible_card_count({}) is None
    assert visible_card_count({"CUDA_VISIBLE_DEVICES": "2"}) == 1
    assert visible_card_count({"CUDA_VISIBLE_DEVICES": "2,3"}) == 2
    assert visible_card_count({"CUDA_VISIBLE_DEVICES": "2,3,"}) == 2
    assert visible_card_count({"CUDA_VISIBLE_DEVICES": ""}) == 0
    assert visible_card_count({"ASCEND_RT_VISIBLE_DEVICES": "0,1"}) == 2


# ------------------------------------------------------------------------- the two flags


def test_the_platform_and_the_cards_are_two_fields_with_two_questions() -> None:
    args = _args(device="cuda", device_ids=(2, 3), tensor_parallel_size=2)
    assert args.device == "cuda"
    assert args.device_ids == (2, 3)


def test_the_platform_is_one_of_four_and_a_card_is_not_one_of_them() -> None:
    """The migration, at the field: `0` and `cuda:2` are refused with the flag that answers them.

    They are well formed values of the flag they used to belong to, so a refusal that only said
    "invalid" would be true and useless. The hint is the whole of what an affected launch changes.
    """
    assert _args(device="cpu").device == "cpu"
    for value, hint in (("cuda:2", "--device-ids 2"), ("0", "--device-ids 0"), ("npu:1", "--device-ids 1")):
        with pytest.raises(ConfigurationError, match="names a card") as raised:
            _args(device=value)
        assert hint in str(raised.value)
    with pytest.raises(ConfigurationError, match="must be one of"):
        _args(device="tpu")


def test_a_card_list_is_read_from_a_string_or_a_sequence() -> None:
    """A command line carries `2,3` and a caller building the args by hand carries a list; both have
    to land on the same value, or comparing a parent against the rank it spawned compares spellings."""
    assert _args(device_ids="2,3").device_ids == (2, 3)
    assert _args(device_ids=[2, 3]).device_ids == (2, 3)
    assert _args(device_ids=("2", "3")).device_ids == (2, 3)
    assert _args().device_ids == ()


def test_a_card_list_that_cannot_hold_the_world_is_refused() -> None:
    """Two cards for four ranks leaves the last two nowhere, and one card twice is worse: two ranks
    on one card is a collective that never completes rather than a run that is merely slow."""
    with pytest.raises(ConfigurationError, match="one per rank"):
        _args(tensor_parallel_size=4, device_ids=(2, 3))
    with pytest.raises(ConfigurationError, match="cannot hold two ranks"):
        _args(tensor_parallel_size=2, device_ids=(2, 2))
    with pytest.raises(ConfigurationError, match="non-negative card indices"):
        _args(device_ids="2,x")


def test_cards_and_a_host_only_platform_are_refused_together() -> None:
    """`cpu:0` is not a device, so the pair has no reading -- refused where both are still words."""
    with pytest.raises(ConfigurationError, match="cpu"):
        _args(device="cpu", device_ids=(0,))


# ------------------------------------------------------------------------- the surface


def test_no_declaration_generates_a_device_flag_any_more() -> None:
    """`device` was the one entry in `NO_FLAG`, and U3 removed the declaration rather than the
    exception: the card list means the same thing on every runtime, including the native one that
    has no options list, so it is a fact about the launch and lives beside `tensor_parallel_size`."""
    assert "device" not in cli_surface.declarations()
    assert "device_ids" not in cli_surface.declarations()
    assert cli_surface.NO_FLAG == frozenset()


def test_the_host_owns_both_flags_and_the_help_says_what_a_card_is(capsys) -> None:
    """Both are the host's, so `--help` has to answer the question the split raises: which is which.

    It reads the rendered text rather than the actions because the rendering is the part an operator
    meets -- a `--device-ids` that took a space-separated list would be as invisible to a reader of
    the parser as to the reader of `--help`, and only one of them is who it is for.
    """
    from pocketllm.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--help"])
    printed = " ".join(capsys.readouterr().out.split())
    assert "--device {auto,cuda,ascend,cpu}" in printed
    assert "--device-ids L" in printed
    assert "in rank order" in printed


def test_the_placement_travels_to_a_worker_rank() -> None:
    """A worker that defaulted the card list would bind a card rank 0 did not name.

    The failure is a collective that never completes rather than a wrong number, and it shows up at
    the first all-reduce rather than at startup -- so the list travels with the rest of the launch's
    placement, through the same `POCKETLLM_WORKER_ARGS` the other shared arguments use.
    """
    from pocketllm.backends.factory import _worker_arg_overrides

    args = _args(device="cuda", device_ids=(2, 3), tensor_parallel_size=2)
    overrides = _worker_arg_overrides(args)
    assert overrides["device"] == "cuda"
    assert overrides["device_ids"] == (2, 3)


def test_the_environment_bridge_spells_the_same_two_things(monkeypatch) -> None:
    """`DEVICE` was the bridge's word for a card and `POCKETLLM_DEVICE_IDS` is its word for the list.

    The bridge exists so a launcher that sets variables instead of a command line reaches the same
    fields; a split that only moved the flag would leave that launcher with no way to say the new
    thing, and one that kept `DEVICE` meaning a card would leave two spellings of one field.
    """
    monkeypatch.setenv("CKPT_PATH", "/tmp/checkpoint")
    monkeypatch.setenv("POCKETLLM_DEVICE_IDS", "2,3")
    assert EngineArgs.from_env().device_ids == (2, 3)

    monkeypatch.setenv("DEVICE", "cuda:1")
    with pytest.raises(ConfigurationError, match="names a card"):
        EngineArgs.from_env()

    monkeypatch.setenv("DEVICE", "cuda")
    assert EngineArgs.from_env().device == "cuda"
