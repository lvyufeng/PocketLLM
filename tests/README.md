# The test suite

`tests/` holds two different things and pytest collects only one of them.

- **`test_*.py`** — the suite. Run it from the repository root, because there is no `conftest.py`
  and no pytest configuration: modules import from the root and nothing puts it on `sys.path`.
- **`bench_*.py`, `probe_*.py`, `profile_*.py`, `debug_*.py`, `summarize_*.py`** — measurement
  scripts. pytest does not collect them, and the numbers they print are quoted in `docs/` with the
  command that produced them. `tests/fixtures/` and `tests/data/` are their inputs.

```bash
python -m pytest tests/ -q          # the whole suite
python -m pytest tests/test_x.py -q # one module
```

There is no `conftest.py` by design rather than by accident: a root `conftest.py` would change how
every module is imported, and the suite's modules are run as scripts often enough (see the skip
policy below) that the two entry points have to agree. Shared fixtures are defined in the module
that uses them.

## The baseline, and why it is a set

**The suite is not green on `master`.** The failures that are known are recorded, one per line by
node id, in [`baseline_failures.txt`](baseline_failures.txt), and
[`scripts/check_test_baseline.py`](../scripts/check_test_baseline.py) is what diffs a run against it:

```bash
python scripts/check_test_baseline.py                 # run the suite, diff, exit 1 on anything new
python scripts/check_test_baseline.py --observed FILE # diff a run recorded earlier
python scripts/check_test_baseline.py --update        # rewrite the baseline from a fresh run
python scripts/check_test_baseline.py --fail-on-fixed # also fail when an entry stops failing
```

It is a **set of node ids and not a count**, and that distinction is the whole point. Three tests
fixed and one broken is a *net improvement* in a count and a regression in the tree, and reporting
that as "-2" hides the one that matters. It is also symmetric in the other direction: a test that
used to fail and now passes silently is reported on its own line, because a fixed test and a test
that has started skipping itself look identical from one run.

`--fail-on-fixed` is off by default because those two really are indistinguishable from here: this
suite skips itself for want of a card, a checkpoint or a built extension, and a failure recorded on
this host may legitimately become a skip on another. Deciding otherwise is a decision about the
host, so it is a flag rather than the default.

The recorder itself, [`scripts/_baseline_recorder.py`](../scripts/_baseline_recorder.py), is loaded
with `-p` by that script and never by the suite, so a plain `pytest` run is unaffected by any of
this.

## Skips are not passes

A module that needs a GPU, a real checkpoint or a built extension skips itself — `pytest.skip`,
`pytest.importorskip` or a fixture that skips — and **a skip is not a pass**. It says the test could
not run here, which is a different claim from the test having run and succeeded.

That is why skips are absent from the baseline. An entry in `baseline_failures.txt` is a claim that
the test *runs on this host and fails*; a test that cannot run here belongs nowhere in that file, and
a skip that was reported as a new failure would make the check useless on any machine but the one it
was recorded on.

The skipped set is a coverage claim, so it is worth reading before trusting a green run:

```bash
python -m pytest tests/ -q -rs        # show why each test was skipped
```

## The no-checkpoint tests and the served-path tests

Most of the suite is hermetic: it tests a kernel against a reference, a parser against a fixture, or
an invariant against the source tree, and it runs in seconds. Those are the tests to keep fast.

A smaller set needs a **real checkpoint** and is the only place an end-to-end claim is checked — a
prompt in, token ids out, through a real entry point. Those tests find their checkpoint through the
environment and skip when it is absent, so the suite is runnable on a machine with no weights at
all. Each golden fixture records the checkpoint, the flags and the prompt it was taken under, in
[`fixtures/golden/`](fixtures/golden/), together with the token ids that came out; see
[the golden fixtures section](#golden-fixtures) below.

## What CI runs

**Nothing here.** `.github/workflows/publish-pypi.yml` builds and uploads a release, and
`.github/workflows/pages.yml` builds the documentation site with `mkdocs build --strict`. No
workflow runs pytest, so `python scripts/check_test_baseline.py` is a manual step and the baseline
is a record rather than a gate. Anything that has to be enforced therefore has to be enforced by a
check that runs somewhere — which, in this repository, means the `mkdocs build --strict` link check
for documentation and the `check_layering` CMake target for the C++ layering.

## Golden fixtures

A golden fixture is one real request through one real entry point, recorded with everything needed
to reproduce it:

| Field | What it is |
| --- | --- |
| `entry` | the entry point: `xing4` — the only runtime this build has |
| `checkpoint` | the checkpoint directory or GGUF, as an absolute path |
| `env` | the environment variables the run needs (`CUDA_VISIBLE_DEVICES`, …) |
| `argv` | the command line, **as an operator types it**, minus `--prompt` |
| `prompt` | the prompt **as text** |
| `sampling` | greedy is `temperature: 0.0`; a fixture that samples is a fixture that fails one run in ten |
| `requires` | a resource the run needs from this host, currently `dev_shm_bytes` |
| `expected.prompt_tokens` | how many tokens the prompt rendered to for this checkpoint's tokenizer |
| `expected.token_ids` | what came out |
| `expected.text` | what came out |
| `commit`, `taken_at` | the revision and the day it was recorded |

Four things about that list are load-bearing, and the third is about how a fixture *runs* rather
than what it records.

**`argv` is a command line and the Python entries are driven through the CLI parser**, so
`build_parser()` → `_args()` → `EngineArgs` is exercised on every run of the fixture. A flag that
stops being read, or a default that moves, changes the tokens the fixture produces. Driving
`EngineArgs` directly would leave untested the one layer a serving refactor is about.

**`expected.prompt_tokens` is compared before the answer**, because the tokenizer and the chat
template are half of what a fixture pins: the same text rendering to a different number of tokens
means the model was asked a different question, and comparing only the answer would blame the model
for it.

**Each fixture runs in its own process.** The test spawns `tests/golden_fixtures.py --entry <name>`
and compares what the child hands back, rather than loading the engine inside pytest's interpreter.
That is not tidiness — a model loaded into the pytest process is a configuration nothing else in this
repository uses, and one process per rank per GPU is the rule in
[`docs/guides/benchmarking.md`](../docs/guides/benchmarking.md). So the child is the unit of
execution and the comparison stays in the parent, where a failure can name the fixture.

**`requires` is how a fixture states what it needs from the host**, and it is why a missing resource
is a skip and not a failure: a fixture that cannot fit skips and names the size it wanted, rather
than dying partway through a load with an error that reads like a model bug.

A fixture is a smoke test, not a benchmark: it answers "does this checkpoint still produce this
answer through this entry point", which is the question a refactor of the request lifecycle must not
break. It is deliberately not a rate measurement — rates belong in `docs/performance/` with the
command that produced them, and they move with the host.

Recording one is a deliberate act rather than a side effect, because re-recording is how a real
regression gets quietly blessed as correct:

```bash
python scripts/record_golden_fixture.py --entry xing4 \
    --checkpoint /mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf \
    --max-tokens 16 -- --max-model-len 4096
```

### Running them is opt-in

The fixture loads a real checkpoint, so it is not free: the `xing4` entry answers in 18 seconds on a
loaded model. The *runs* are gated:

```bash
python -m pytest tests/test_served_path_golden.py -q                    # skips, one per fixture
POCKETLLM_GOLDEN=1 python -m pytest tests/test_served_path_golden.py -q # now they run
```

Everything else in that module always runs, and it is the part that matters for coverage: the
fixture set is complete, every file parses, and every one carries an answer. What the gate defers is
the end-to-end execution, not the claim that the fixture exists.

The file is a statement about the repository, so it is checked in even when the checkpoint lives only
on one machine; a run skips when the path is absent, and `-rs` prints the path it wanted.
