# The test suite

Run it from the repository root, because the modules under test live at `python/pocketllm/`: the
import path is shortened by the `pythonpath = ["python"]` line in `pyproject.toml`'s
`[tool.pytest.ini_options]`, and `conftest.py` at this level exports the same path to the fresh
interpreters several probes spawn. Both only take effect when pytest finds them — that is, from the
root:

```bash
python -m pytest tests/ -q
```

## What pytest collects

| Directory | What it covers |
| --- | --- |
| `abi/` | The kernel ABI: op schemas, tensor and buffer descriptors, graph verification, dispatch resolution, the reference backend's completeness, and the guarantee that importing the package pulls in no device runtime. |
| `backends/` | The backend registry and the conformance harness: every declared op runs on every available backend, and the numerics match the reference. `backends/conftest.py` holds the harness. |
| `architectures/` | The model IR and the `toy` architecture — a graph built and verified through the same builder a real model uses. |
| `engine/` | The execution layer: the op-by-op executor, the memory planner, region planning and capture, session lifecycle, and the `LLM` facade over them. |
| `loader/` | The GGUF decoders — one test per quant format — and the vendored GGML header's resolution order and pinned hash. |
| `serving/` | The ported serving layer: the OpenAI-compatible contract, chat templating, request parsing, choice fan-out, and the HTTP server against a fake backend. |
| `test_package_boundaries.py` | The layering rules — which package may import which — plus the two subprocess checks that `import pocketllm` and `import pocketllm.cli` pull in no runtime. |

There is no `fixtures/` directory and no golden fixture yet: a golden fixture records one real request
through one real entry point, and there is no entry point that can produce an answer until an
architecture and a backend exist. `scripts/record_golden_fixture.py` is kept and ready; the first
fixture belongs in the commit that makes `pocketllm run` work, together with the test that replays it.

A few `bench_*` / `probe_*` / `summarize_*` scripts have not been carried over. What is left is
pytest-collectable only.

## The baseline

`tests/baseline_failures.txt` records the known failures as a **set** of node ids. Diff a run against
it with:

```bash
python scripts/check_test_baseline.py        # --update to re-record
```

It is a set and not a count on purpose: three tests fixed and one broken is a net improvement in a
count and a regression in the tree. The comparison is symmetric, so a node id that was expected to
fail and now passes is reported too — a silently-skipped test that reports green is exactly the
failure mode this catches. The set is currently **empty**: every test this host collects and runs
passes.

## Skip policy

A module that needs a card, a checkpoint or a device runtime **skips itself**, and a skip is **not a
pass**. A test that fails on this host belongs in the baseline; one skipped for want of hardware does
not.

The `reference` backend is pure numpy and always available, so the ABI and the executor are never
skipped wholesale — what skips is an accelerated backend's conformance run, and the numeric comparison
for a format with no fast kernel.

## Why the boundary tests are here and not in review

`test_package_boundaries.py` parses each module's imports with `ast` rather than importing it, so a
violation is reported by file and line even when the module cannot be imported on this host. That
matters more here than usual: the whole point is to police backends whose runtimes are absent, and a
rule only checked on machines that have the runtime is a rule that does not hold on the machines that
do not.

## What CI runs

**CI runs no tests.**

- `.github/workflows/pages.yml` builds this documentation site with `mkdocs build --strict`, and that
  build is also the repository's link checker and the `docs/llms.txt` staleness check.
- `.github/workflows/publish-pypi.yml` builds and uploads a release.

Neither runs pytest, so the baseline check is a manual step and a gate only where somebody runs it.