"""The golden-fixture harness itself: the schema, the flag reader, the child protocol, the skip rule.

The end-to-end tests in `test_served_path_golden.py` all skip without a checkpoint, so without these
the harness would be code that runs nowhere on a machine with no weights -- which is most machines.
What they cover is the part that decides *what* an end-to-end run means: how a fixture is read and
written, how a child hands an outcome back, and when a fixture is skipped rather than run.

The skip rule is the one worth being explicit about, and it has exactly two reasons, both about the
machine rather than about the answer: the checkpoint is not here, and a resource the fixture states
it needs is not free. Both are asserted below so that nothing else can quietly join that list: a
fixture that skipped because its *answer* moved would be a regression wearing a skip.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest
from tests.golden_fixtures import (
    CHILD_RESULT_ENV,
    DEFAULT_SAMPLING,
    ENTRY_POINTS,
    GOLDEN_MODULE,
    REPO_ROOT,
    GoldenFixture,
    Outcome,
    _with_repo_root_on_the_path,
    fixture_path,
    load_fixture,
    load_fixtures,
    run_isolated,
)

PAYLOAD = {
    "entry": "xing4",
    # Synthetic, and deliberately a path that does not exist: this module must not depend on which
    # checkpoints this machine happens to have, and one test below asserts the skip rule on it.
    "checkpoint": "/mnt/nope/xing4_0-29b-IQ4_NL.gguf",
    "argv": ["serve", "--backend", "xing4", "--model", "/mnt/nope/xing4_0-29b-IQ4_NL.gguf"],
    "env": {"CUDA_VISIBLE_DEVICES": "2"},
    "prompt": "Reply with the single line: golden fixture.",
    "sampling": {"temperature": 0.0, "max_tokens": 16},
    "expected": {"prompt_tokens": 9, "token_ids": [35, 48972], "text": "\ngolden fixture"},
    "commit": "abc1234",
    "taken_at": "2026-09-26",
    "notes": "why this checkpoint",
}


def test_a_fixture_round_trips_through_json() -> None:
    fixture = GoldenFixture.from_json(PAYLOAD)

    assert GoldenFixture.from_json(json.loads(json.dumps(fixture.to_json()))) == fixture


def test_the_expected_side_is_read_by_name_and_not_by_position() -> None:
    """The three fields are separate claims and each has its own accessor."""
    fixture = GoldenFixture.from_json(PAYLOAD)

    assert fixture.expected_token_ids == (35, 48972)
    assert fixture.expected_text == "\ngolden fixture"
    assert fixture.expected_prompt_tokens == 9


def test_a_fixture_without_the_required_fields_is_refused() -> None:
    """Every one of these is load-bearing, so a truncated fixture fails at load, not at compare."""
    for missing in ("entry", "checkpoint", "argv", "prompt", "expected"):
        payload = {k: v for k, v in PAYLOAD.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            GoldenFixture.from_json(payload)


def test_a_fixture_is_skipped_only_for_a_missing_checkpoint() -> None:
    fixture = GoldenFixture.from_json(PAYLOAD)

    reason = fixture.unwritable_reason()

    assert reason is not None and "checkpoint not present" in reason


def test_a_fixture_whose_checkpoint_exists_is_not_skipped(tmp_path: pathlib.Path) -> None:
    present = tmp_path / "checkpoint"
    present.mkdir()
    fixture = GoldenFixture.from_json({**PAYLOAD, "checkpoint": str(present)})

    # No GPU, no built extension, no engine -- and none of that is a reason to skip. Those belong to
    # the entry point, which is expected to raise, because an answer that moved has to be reported.
    assert fixture.unwritable_reason() is None


def test_the_entry_point_set_is_the_one_the_readme_promises() -> None:
    """A new backend cannot be added without either recording a fixture or failing this test."""
    assert ENTRY_POINTS == ("cpp", "v41", "mimo", "xing4", "torch")


def test_the_default_sampling_is_greedy() -> None:
    """A fixture that samples is a fixture that fails one run in ten for no reason."""
    assert DEFAULT_SAMPLING["temperature"] == 0.0


def test_a_recorded_fixture_names_its_own_entry_point() -> None:
    """The file name and the `entry` field have to agree, or the completeness test lies.

    `load_fixture("mimo")` reads `mimo.json`; a file whose contents say `xing4` would satisfy the
    completeness check for a backend that has no fixture at all.
    """
    for entry in ENTRY_POINTS:
        fixture = load_fixture(entry)
        if fixture is None:
            continue
        assert fixture.entry == entry, f"{fixture_path(entry)} records entry={fixture.entry!r}"
        assert fixture_path(entry).name == f"{entry}.json"


def test_every_loaded_fixture_can_be_compared() -> None:
    """A fixture with no recorded answer cannot fail, and a thing that cannot fail is not a test."""
    for entry, fixture in load_fixtures().items():
        has_answer = bool(fixture.expected_token_ids) or fixture.expected_text is not None
        assert has_answer, f"{entry} records neither token ids nor text"


def test_the_outcome_survives_the_trip_through_the_child() -> None:
    """The parent reads the child's answer off disk, so every field has to round trip exactly.

    `None` is the interesting case: an entry point that produces no token ids is a fact about the
    entry point, not a missing field, and a round trip that turned it into `[]` would make `torch`'s
    fixture look like it had been compared when it had not.
    """
    for outcome in (
        Outcome(token_ids=[35, 48972], text="\ngolden fixture", prompt_tokens=9, elapsed_seconds=1.5),
        Outcome(text="golden fixture.", prompt_tokens=21, elapsed_seconds=16.0),
    ):
        assert Outcome.from_json(json.loads(json.dumps(outcome.to_json()))) == outcome


def test_the_child_module_is_launchable() -> None:
    """`run_isolated` spawns this file by path, so it has to work as a program and not only a module.

    It is checked here because the alternative is finding out during a fixture run, where the failure
    arrives as a non-zero exit from a forty-minute process.
    """
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--help"], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert "--entry" in completed.stdout


def test_the_child_refuses_an_entry_point_with_no_fixture() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "definitely-not-an-entry"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


def test_the_child_needs_somewhere_to_write_the_outcome() -> None:
    """Without the env var there is nowhere for the answer to go, so it stops before loading weights.

    Required rather than defaulted: a child that defaulted to stdout would print its JSON into a
    stream the entry point is also writing to, and the parent would parse whatever came first.
    """
    env = {k: v for k, v in __import__("os").environ.items() if k != CHILD_RESULT_ENV}
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "xing4"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert CHILD_RESULT_ENV in completed.stderr


def test_a_fixture_that_cannot_run_here_raises_rather_than_spawning_a_child() -> None:
    """The parent checks the host first, so an unrunnable fixture costs no process and no model load.

    `run_isolated` is what the suite calls; a fixture whose checkpoint is absent must raise there --
    the test that calls it skips earlier, and this is the belt to that pair of braces.
    """
    fixture = GoldenFixture.from_json(PAYLOAD)

    with pytest.raises(RuntimeError, match="checkpoint not present"):
        run_isolated(fixture)


def test_the_child_is_given_the_repository_root_on_its_import_path(monkeypatch) -> None:
    """The suite finds `pocketllm` because the root is the working directory; the child does not.

    The child is launched by path, so the interpreter puts `tests/` on `sys.path` instead -- which
    made every Python fixture fail with `ModuleNotFoundError: No module named 'pocketllm'` the first
    time the runs were isolated. It goes in the environment rather than the child's `sys.path`
    because the entry points start their own ranks, and those inherit only the environment.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _with_repo_root_on_the_path() == str(REPO_ROOT)

    # An existing path is kept, behind the root: the root has to win, or a stale install shadows it.
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    assert _with_repo_root_on_the_path().split(os.pathsep) == [str(REPO_ROOT), "/somewhere/else"]
