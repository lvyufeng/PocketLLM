"""The baseline diff itself: what it calls a regression, and what it calls a fix.

`scripts/check_test_baseline.py` is the instrument that is supposed to make the suite's known
failures visible as a *set*, and an instrument that is only ever run against the whole suite is
tested by nothing. These tests drive `diff_outcomes` and `parse_baseline` directly, on outcome maps
that are short enough to read, which is how the two behaviours the refactor depends on are pinned:

- **a new failure is reported by node id**, not by a count that moved -- including when the count
  moved *down*, which is the case a summary line reports as an improvement;
- **a node id that stops failing is reported separately**, and does not on its own fail the check,
  because a test that has become skipped on this host and a test that has been fixed are
  indistinguishable from one suite run.

The maps below are `nodeid -> outcome`, the shape `scripts/_baseline_recorder.py` writes and the
shape `--observed` reads, so a change to either format breaks these tests rather than the check.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_test_baseline.py"


def _load_checker():
    """Import scripts/check_test_baseline.py by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("check_test_baseline_under_test", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


BASE = {"tests/test_a.py::test_one": "failed", "tests/test_b.py::test_two": "error"}


def test_an_unchanged_run_is_clean() -> None:
    observed = dict(BASE, **{"tests/test_c.py::test_three": "passed"})

    result = checker.diff_outcomes(BASE, observed)

    assert result.ok
    assert result.new == ()
    assert result.fixed == ()
    assert result.still_failing == ("tests/test_a.py::test_one", "tests/test_b.py::test_two")


def test_a_new_failure_is_reported_by_node_id() -> None:
    observed = dict(BASE, **{"tests/test_c.py::test_three": "failed"})

    result = checker.diff_outcomes(BASE, observed)

    assert not result.ok, "a new failure has to fail the check"
    assert result.new == ("tests/test_c.py::test_three",)


def test_a_falling_count_does_not_hide_a_new_failure() -> None:
    """The whole reason this is a set and not a number.

    Two baseline entries heal and one new failure appears: 2 -> 1, which any count-based summary
    reports as progress. Pinned as the acceptance criterion of issue #438.
    """
    observed = {"tests/test_c.py::test_three": "failed"}

    result = checker.diff_outcomes(BASE, observed)

    assert not result.ok
    assert result.new == ("tests/test_c.py::test_three",)
    assert result.fixed == ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
    assert len(result.new) < len(BASE), "the count fell, which is exactly the trap"


def test_a_healed_entry_is_reported_but_does_not_fail_the_check() -> None:
    """A fix and a skip look the same from here, so neither is enforced.

    `--fail-on-fixed` is the opt-in for a caller that knows the difference (a release run on the
    host the baseline was taken on), and it is asserted here so the flag cannot be dropped.
    """
    observed = {"tests/test_a.py::test_one": "failed"}

    result = checker.diff_outcomes(BASE, observed)

    assert result.ok
    assert result.fixed == ("tests/test_b.py::test_two",)
    assert not checker.diff_outcomes(BASE, observed, fail_on_fixed=True).ok


def test_an_error_and_a_failure_are_both_failing_outcomes() -> None:
    """Both have to count, or a fixture that blows up would read as a passing test."""
    observed = {"tests/test_c.py::test_three": "error"}

    result = checker.diff_outcomes(BASE, observed)

    assert result.new == ("tests/test_c.py::test_three",)
    assert checker.FAILING_OUTCOMES == {"failed", "error"}


def test_a_skip_is_never_a_regression() -> None:
    """This suite skips itself for want of a card, a checkpoint or a built extension.

    A skip that was reported as a new failure would make the check useless on any host but this
    one, which is the state the suite was already in.
    """
    observed = dict(BASE, **{"tests/test_c.py::test_three": "skipped"})

    assert checker.diff_outcomes(BASE, observed).ok


def test_the_baseline_round_trips_through_its_own_format() -> None:
    rendered = checker.render_baseline(BASE)

    assert checker.parse_baseline(rendered) == BASE
    assert all(line.startswith("#") or line.split()[0] in BASE for line in rendered.splitlines() if line.strip())


def test_the_baseline_file_refuses_a_passing_entry() -> None:
    """A green test in the baseline would silently license a future failure of the same id."""
    with pytest.raises(ValueError, match="only failing outcomes|failed\\|error"):
        checker.parse_baseline("tests/test_a.py::test_one passed\n")


def test_the_baseline_file_tolerates_comments_and_blank_lines() -> None:
    entries = checker.parse_baseline(
        "# a comment\n"
        "\n"
        "tests/test_a.py::test_one failed   # the reason, which is not part of the id\n"
        "tests/test_b.py::test_two error\n"
    )

    assert entries == BASE


def test_the_baseline_file_refuses_a_malformed_line() -> None:
    with pytest.raises(ValueError):
        checker.parse_baseline("tests/test_a.py::test_one\n")


def test_the_recorded_baseline_parses_and_is_sorted() -> None:
    """The file checked in has to be readable by the checker that reads it.

    It is asserted as sorted because an unsorted file makes a one-line change look like a rewrite,
    and the diff is the entire point of recording a set.

    The empty set is a legal state -- it is the state this host is in -- so the guard cannot be
    "there is at least one entry". What has to hold instead is that nothing was silently dropped:
    every non-comment, non-blank line of the file is a parsed entry. A parser that stopped
    understanding the format returns `{}` for a file full of entries, which is the regression the
    count-based assertion used to catch and this one still does.
    """
    text = checker.BASELINE_PATH.read_text(encoding="utf-8")
    entries = checker.parse_baseline(text)

    recorded = [
        line.split("#", 1)[0].strip()
        for line in text.splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert [f"{nodeid} {outcome}" for nodeid, outcome in sorted(entries.items())] == recorded
    assert list(entries) == sorted(entries)
    assert all(outcome in checker.FAILING_OUTCOMES for outcome in entries.values())


def test_a_missing_baseline_is_refused_and_an_empty_one_is_not(tmp_path, monkeypatch) -> None:
    """The two states are different, and reading them as one made the check unusable.

    An empty file is the state this host is in -- its own header says so -- and the check has
    nothing to report there. A missing file cannot be told apart from a file that lost its entries,
    and treating it as empty would report every known failure as new, burying the one fact worth
    reading under a wall of node ids.
    """
    observed = tmp_path / "observed.json"
    observed.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(checker, "BASELINE_PATH", tmp_path / "gone" / "baseline_failures.txt")
    with pytest.raises(SystemExit) as refused:
        checker.main(["--observed", str(observed)])
    assert "missing" in str(refused.value)

    empty = tmp_path / "baseline_failures.txt"
    empty.write_text("# The set is empty: every test this host collects and runs passes.\n", "utf-8")
    monkeypatch.setattr(checker, "BASELINE_PATH", empty)
    assert checker.main(["--observed", str(observed)]) == 0
