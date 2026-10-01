"""Record the outcome of every collected test as a machine-readable list.

Loaded by `scripts/check_test_baseline.py` with `-p`, so this file is never imported by the suite
itself and a normal `pytest` run is unaffected by it. It exists because the baseline the suite is
diffed against has to be a *set of node ids* and not a count: the two failures that motivated the
check were a silent pass (an uncollected outcome that reads as green) and a count that can fall
while a new failure appears, and no summary line distinguishes either.

Written to `$POCKETLLM_BASELINE_OUT` as JSON, a list of `[nodeid, outcome]` sorted by node id, so
two runs of the same tree produce byte-identical files.
"""

from __future__ import annotations

import json
import os

_OUTCOME_BY_PHASE = ("setup", "call", "teardown")


class BaselineRecorder:
    """A pytest plugin: no test is modified, the outcome is only observed."""

    def __init__(self) -> None:
        # nodeid -> the worst thing that happened to it. "error" outranks "failed" because a test
        # whose fixture blew up never ran its body, and reporting it as a failure would claim an
        # assertion that was never reached.
        self.outcomes: dict[str, str] = {}
        self._rank = {"passed": 0, "skipped": 1, "xfailed": 1, "failed": 2, "error": 3}

    def _record(self, nodeid: str, outcome: str) -> None:
        if outcome not in self._rank:
            outcome = "failed" if outcome == "failed" else "error"
        previous = self.outcomes.get(nodeid)
        if previous is None or self._rank[outcome] > self._rank[previous]:
            self.outcomes[nodeid] = outcome

    def pytest_runtest_logreport(self, report) -> None:  # type: ignore[no-untyped-def]
        if report.when not in _OUTCOME_BY_PHASE:
            return
        if report.failed:
            # A setup failure is an error, and the distinction matters: it is the shape a missing
            # dependency takes, and the shape this suite's collection errors took.
            self._record(report.nodeid, "error" if report.when != "call" else "failed")
        elif report.when == "call":
            self._record(report.nodeid, report.outcome)

    def pytest_collectreport(self, report) -> None:  # type: ignore[no-untyped-def]
        if report.failed:
            self._record(report.nodeid, "error")

    def pytest_sessionfinish(self, session, exitstatus) -> None:  # type: ignore[no-untyped-def]
        path = os.environ.get("POCKETLLM_BASELINE_OUT")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sorted(self.outcomes.items()), handle, indent=1)
            handle.write("\n")


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.pluginmanager.register(BaselineRecorder(), "pocketllm-baseline-recorder")
