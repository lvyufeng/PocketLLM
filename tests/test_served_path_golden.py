"""Golden fixtures: the served path, end to end, per entry point.

Three tests, and they answer different questions.

- `test_every_entry_point_has_a_fixture` is a statement about the repository and needs no weights,
  no card and no engine: every entry point `tests/README.md` names has a recorded fixture. It is the
  test that makes a missing entry point visible, because a fixture that does not exist skips rather
  than fails and a suite of skips reads as green.
- `test_every_entry_point_that_is_recorded_parses` says the recorded files are loadable and carry an
  answer. Also free.
- `test_served_path_matches_the_recorded_answer` runs each fixture and compares what comes back. It
  is **opt-in**, gated on `POCKETLLM_GOLDEN=1`, and it skips when the checkpoint is not on this
  machine -- the fixture names a path, and the only honest outcome without it is a skip.

Each fixture runs in its own child process, for the reason `run_isolated` documents: five engines in
one interpreter is a configuration nothing else here uses, and `xing4` after `mimo` and `torch`
after `mimo` both fail in it for reasons that are not about the checkpoint. The comparison stays
here, so the failure still names the fixture.

Why that gate rather than running the fixture always: these are the only tests here that load real
weights, and the fixtures span three orders of magnitude in cost. `xing4` answers in 18 seconds; the
`v41` fixture pins a 457.8 GiB resident expert bank and takes forty minutes on a cold segment. A
suite that a person runs before a commit cannot include the second one by default, and a gate with a
documented reason is better than either a silently cheap subset or a suite nobody runs.

What a mismatch means: this checkpoint, through this entry point, with these flags, no longer
produces these tokens. That is the claim a request-lifecycle refactor has to keep, and it is the one
claim no kernel-local test can make.
"""

from __future__ import annotations

import os

import pytest

from tests.golden_fixtures import (
    ENTRY_POINTS,
    GOLDEN_GATE_ENV,
    GoldenFixture,
    fixture_path,
    load_fixture,
    run_isolated,
)


def _recorded() -> list[tuple[str, GoldenFixture]]:
    return [(entry, fixture) for entry in ENTRY_POINTS if (fixture := load_fixture(entry)) is not None]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_every_entry_point_has_a_fixture(entry: str) -> None:
    path = fixture_path(entry)
    assert path.is_file(), (
        f"no golden fixture for the {entry!r} entry point. Record one with\n"
        f"    python scripts/record_golden_fixture.py --entry {entry} --checkpoint <path>\n"
        f"which writes {path.relative_to(path.parents[2])}. A path that does not exist is not a "
        f"reason to leave it out: the fixture records where the checkpoint lives and the test skips "
        f"when it is absent, so the file is a statement about this repository rather than about the "
        f"machine reading it."
    )


def test_every_recorded_fixture_belongs_to_a_known_entry_point() -> None:
    """A file in `fixtures/golden/` that is not an entry point is a fixture nobody will ever run.

    The completeness test above goes the other way -- an entry point with no file -- so the two
    together pin the set, and this one is what notices `fixtures/golden/troch.json`, which would be
    a typo'd fixture that skipped nothing and checked nothing.
    """
    from tests.golden_fixtures import FIXTURE_DIR, load_fixtures

    on_disk = {path.stem for path in FIXTURE_DIR.glob("*.json")}
    orphans = sorted(on_disk - set(ENTRY_POINTS))

    assert not orphans, (
        f"{', '.join(orphans)} in {FIXTURE_DIR.relative_to(FIXTURE_DIR.parents[1])} name no entry "
        f"point, so nothing loads them. Either the entry point is missing from ENTRY_POINTS or the "
        f"file is a typo"
    )
    assert set(load_fixtures()) == on_disk


@pytest.mark.parametrize("entry,fixture", _recorded(), ids=[entry for entry, _ in _recorded()])
def test_served_path_matches_the_recorded_answer(entry: str, fixture: GoldenFixture) -> None:
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(
            f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them. They load "
            f"real checkpoints and the cost ranges from seconds to the better part of an hour"
        )
    reason = fixture.unwritable_reason()
    if reason is not None:
        pytest.skip(reason)

    outcome = run_isolated(fixture)

    # The prompt's own token count first, because it is the half of a fixture a moved *tokenizer*
    # shows up in, and a comparison of the answer alone would blame the model for it.
    if fixture.expected_prompt_tokens is not None:
        assert outcome.prompt_tokens == fixture.expected_prompt_tokens, (
            f"{entry}: the prompt rendered to {outcome.prompt_tokens} tokens, not "
            f"{fixture.expected_prompt_tokens}. The text is unchanged, so this is the tokenizer or "
            f"the chat template -- both of which change what the model is asked"
        )

    compared = False
    if fixture.expected_token_ids:
        assert outcome.token_ids is not None, (
            f"the {entry} fixture records token ids but this entry point produced none; either the "
            f"entry point stopped returning ids or the fixture was recorded under a different kind"
        )
        assert list(outcome.token_ids) == list(fixture.expected_token_ids), (
            f"{entry}: token ids moved. Recorded {list(fixture.expected_token_ids)}, got "
            f"{list(outcome.token_ids)}. The fixture was taken at {fixture.commit or 'an unknown'} "
            f"with {' '.join(fixture.argv)}; a mismatch is either a regression or a fixture that "
            f"needs re-recording, and `git log -1 --format=%H` is what tells them apart"
        )
        compared = True
    if fixture.expected_text is not None:
        assert outcome.text is not None, f"the {entry} fixture records text but none came back"
        assert outcome.text == fixture.expected_text, (
            f"{entry}: served text moved. Recorded {fixture.expected_text!r}, got {outcome.text!r}"
        )
        compared = True
    assert compared, (
        f"the {entry} fixture records neither token ids nor text, so it cannot fail and is not a "
        f"fixture"
    )
