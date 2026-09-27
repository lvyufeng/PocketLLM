#!/usr/bin/env python3
"""Diff the suite's outcomes against the recorded baseline, as a set of node ids.

`python -m pytest tests/` has a non-empty baseline: the failures that are known on this host are
recorded in `tests/baseline_failures.txt` by node id, and this script is the thing that keeps that
set honest. No count is quoted here on purpose -- a number that lives in a docstring goes stale the
first time somebody fixes a test, and the file is the authority.

Two failure modes it exists for, both of which a summary line hides:

- **A silent pass.** The deleted `tests/test_cpp_backend_batching.py` branched on `sys.argv` and
  `return`ed when under-specified, so a run with no path argument reported a green line for a test
  that never ran. A node id that is *expected* to fail and now passes is worth as much attention as
  one that broke, so the diff is symmetric.
- **A falling count.** Three passes gained and one failure introduced is a net improvement in the
  count and a regression in the tree. Counting is not diffing.

Usage::

    python scripts/check_test_baseline.py                # run the suite, diff, exit 1 on anything new
    python scripts/check_test_baseline.py --observed FILE # diff an already-recorded run
    python scripts/check_test_baseline.py --update       # rewrite the baseline from a fresh run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "tests" / "baseline_failures.txt"
RECORDER_PLUGIN = REPO_ROOT / "scripts" / "_baseline_recorder.py"

# Outcomes that are a defect of the tree rather than a statement about the host. A skip is not a
# failure here: this suite skips itself when a checkpoint, a GPU or a built extension is absent, and
# a skip is a statement that the test could not be run, which the baseline is not the place to
# record -- membership in the baseline is a claim that the test runs and fails.
FAILING_OUTCOMES = frozenset({"failed", "error"})

# `tests/test_x.py::test_y[param - with dash]` is a legal node id, so nothing here splits on
# whitespace or punctuation: the whole stripped line is the id. The comment form has to match a
# whole-line comment as well as a trailing one, or the file's own header becomes a malformed entry.
_COMMENT = re.compile(r"(?:^|\s)#.*$")


def parse_baseline(text: str) -> dict[str, str]:
    """Read the baseline file into `nodeid -> outcome`.

    The format is one entry per line so that a change to the set is a line change in a diff, which
    is the property a count cannot have.
    """
    entries: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _COMMENT.sub("", raw).strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[1] not in FAILING_OUTCOMES:
            raise ValueError(
                f"{BASELINE_PATH}:{number}: expected `<nodeid> <failed|error>`, got {raw!r}. "
                f"Only failing outcomes belong in the baseline; a passing test is written nowhere."
            )
        entries[parts[0]] = parts[1]
    return entries


def render_baseline(entries: dict[str, str]) -> str:
    header = (
        "# The suite's known failures, by node id. Regenerate with\n"
        "#   python scripts/check_test_baseline.py --update\n"
        "#\n"
        "# One entry a line, `<nodeid> <failed|error>`, sorted. This file is a *set*, not a count:\n"
        "# the check in scripts/check_test_baseline.py reports a node id that is new and a node id\n"
        "# that is no longer failing separately, because three tests fixed and one broken is a net\n"
        "# improvement in a count and a regression in the tree.\n"
        "#\n"
        "# A test added to this file is a test that runs on this host and fails. Anything skipped for\n"
        "# want of a GPU, a checkpoint or a built extension is not here and must not be.\n"
    )
    if not entries:
        # An empty file under a header that says "the suite's known failures" is ambiguous: it
        # reads either as "every test passes" or as "nobody has recorded anything", and the two
        # call for opposite responses. The line says which one it is, and it is written by the same
        # function that writes the entries, so it cannot go stale independently of them.
        header += (
            "#\n"
            "# The set is empty: every test this host collects and runs passes. The suite still\n"
            "# skips the ones needing a checkpoint, a card or an extension this build has not got,\n"
            "# and a skip is not a pass.\n"
        )
    body = "".join(f"{nodeid} {entries[nodeid]}\n" for nodeid in sorted(entries))
    return header + body


@dataclass(frozen=True)
class Diff:
    new: tuple[str, ...]
    fixed: tuple[str, ...]
    still_failing: tuple[str, ...]
    fail_on_fixed: bool = False

    @property
    def ok(self) -> bool:
        return not self.new and not (self.fail_on_fixed and self.fixed)


def diff_outcomes(
    baseline: dict[str, str], observed: dict[str, str], *, fail_on_fixed: bool = False
) -> Diff:
    """Compare two outcome maps by node id.

    `new` is the only thing that fails the check by default. A node id that has stopped failing is
    reported rather than enforced, because deleting it from the baseline is a decision about the
    tree (the test may be newly skipped on a host without the hardware) and this script cannot tell
    a fix from a skip without the suite's full outcome map, which it deliberately does not keep.
    `fail_on_fixed` is that decision made explicitly, by a caller that knows which host it is on.
    """
    observed_failing = {k for k, v in observed.items() if v in FAILING_OUTCOMES}
    baseline_failing = set(baseline)
    new = tuple(sorted(observed_failing - baseline_failing))
    fixed = tuple(sorted(baseline_failing - observed_failing))
    return Diff(
        new=new,
        fixed=fixed,
        still_failing=tuple(sorted(observed_failing & baseline_failing)),
        fail_on_fixed=fail_on_fixed,
    )


def run_suite(extra_args: list[str]) -> dict[str, str]:
    """Run the suite with the recorder plugin attached and read its JSON back out."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "outcomes.json"
        env = dict(os.environ, POCKETLLM_BASELINE_OUT=str(out))
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-p",
            "_baseline_recorder",
            *extra_args,
        ]
        # `-p _baseline_recorder` needs `scripts/` importable, which is not the repository root.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(RECORDER_PLUGIN.parent), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        if not out.exists():
            raise SystemExit(
                f"the suite produced no outcome record (pytest exited {completed.returncode}); "
                f"its own output above is the reason"
            )
        return dict(json.loads(out.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--observed",
        type=Path,
        help="diff a run already recorded by scripts/_baseline_recorder.py instead of running one",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite tests/baseline_failures.txt from a fresh run",
    )
    parser.add_argument(
        "--fail-on-fixed",
        action="store_true",
        help="also exit 1 when a baseline entry has stopped failing",
    )
    parser.add_argument("pytest_args", nargs="*", help="extra arguments passed through to pytest")
    args = parser.parse_args(argv)

    baseline = (
        parse_baseline(BASELINE_PATH.read_text(encoding="utf-8"))
        if BASELINE_PATH.exists()
        else {}
    )
    if not baseline and not args.update:
        # An empty baseline is the state a fresh checkout would be in if the file were lost, and it
        # would report every known failure as new. That is a loud enough failure to be worth its own
        # message rather than a wall of node ids.
        raise SystemExit(f"{BASELINE_PATH} is missing; run with --update to record it")
    observed = (
        dict(json.loads(args.observed.read_text(encoding="utf-8")))
        if args.observed
        else run_suite(args.pytest_args)
    )

    if args.update:
        entries = {k: v for k, v in observed.items() if v in FAILING_OUTCOMES}
        BASELINE_PATH.write_text(render_baseline(entries), encoding="utf-8")
        print(f"{BASELINE_PATH}: {len(entries)} entries")
        return 0

    result = diff_outcomes(baseline, observed, fail_on_fixed=args.fail_on_fixed)
    print(f"baseline {len(baseline)} failing, observed {len(result.still_failing) + len(result.new)}")
    if result.fixed:
        print(f"\nno longer failing ({len(result.fixed)}) -- remove from the baseline:")
        for nodeid in result.fixed:
            print(f"  {nodeid}")
    if result.new:
        print(f"\nNEW failures ({len(result.new)}) -- these are regressions:")
        for nodeid in result.new:
            print(f"  {nodeid}")
        return 1
    if result.fixed and args.fail_on_fixed:
        return 1
    print("\nthe failure set matches the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
