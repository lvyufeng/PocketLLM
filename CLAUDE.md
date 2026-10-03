# PocketLLM

Single-card inference engine: run a large model on **one accelerator** — edge, mobile, phone. The rule
the scope reduces to is *if it does not fit, quantize it* — never offload, never split across devices.
The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary.

This repository is one of four:

| Repository | What it is |
|---|---|
| **PocketLLM** (this tree) | Single-card / edge / mobile runtime. One process owns one device. |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | Multi-GPU PyTorch runtime and serving shell. |
| [relic-core](https://github.com/lvyufeng/relic-core) | The shared torch operator library (CUDA sm_75 + CPU host ops). **Optional here, not a dependency.** |
| relic-engine | The retired `cpp_engine` tree, kept only as a frozen archive. Nothing here builds against it. |

`docs/README.md` indexes the documentation — the kernel ABI, the backend model and the device targets
are documented there.

## Current state: two halves that have not met

The tree has two implementations of the same thing, at different levels of done, and **the exact
statement of what runs is the point of this section** — the status table in `README.md` is the
authority and is checked against `pocketllm devices` and the registry rather than maintained by hand.

**The C core (`src/`) runs Qwen3-0.6B, in f16 and in `q4_k_m`.** `libpocketllm.so` reads a GGUF,
tokenizes with the checkpoint's own BPE, walks the Qwen3 graph, and decodes greedily — on widened
`f32`/`f16` weights, and on packed `q4_k`/`q6_k` that `src/quant/blocks.h` decodes inside the kernel
and never expands. Verified token-for-token against llama.cpp on `cpu` and on one `cuda` card, for
both checkpoints. That is real, it is tested, and it is what the `src/` row of the README's status
table claims.

**The Python package has no device backend of its own.** `pocketllm.kernels` (the ABI),
`pocketllm.quant`, `pocketllm.loader.gguf`, `pocketllm.engine`, `pocketllm.protocol`,
`pocketllm.server` and `pocketllm.cli` are real and tested — and the CLI is no longer a stub, since
`run` and `serve` both drive the C core through `native.py` — but **no Python backend implements a
kernel**: every backend except `reference` is a declaration with a session that raises
`BackendNotImplementedError` naming the runtime it waits for. The serving adapter in
`server/native_backend.py` is not an exception to that: it implements `EngineBackend`, the *serving*
contract, over the same ctypes bridge, and never touches the kernel ABI.

**These are two separate answers to two separate questions, and the trap is reading either one as the
other.** `pocketllm.kernels` declares 17 ops; the C engine implements 12 of them over its own backend
interface, which is a *different* interface from `pocketllm.backends` and shares no code with it. A
statement about `pocketllm devices` — which lists the *Python* backends and their stubs — says
nothing about what `src/` can do, and `pocketllm devices` on this host does not know the C core
exists.

They meet at exactly one place: `python/pocketllm/native.py`, the `ctypes` bridge, which
**`pocketllm run` and `pocketllm serve` both drive** — each opens a session, tokenizes, runs the
graph and decodes through the C core, and `tests/native/test_cli_run.py` checks the first against
llama.cpp's sequence. The second reaches the same engine through
`python/pocketllm/server/native_backend.py`, the `EngineBackend` adapter over the HTTP surface. So
both host entry points run a model, and neither is a *Python* backend: what they drive is the C core.
Both offer `--temperature/--top-k/--top-p/--min-p/--seed` in some form — the flags on `run`, the
per-request OpenAI fields on `serve` — and greedy is the default in both. The draw is the host's on
every side — `cli.py` holds a `random.Random`, `run.cpp` a `std::mt19937_64`, the adapter a
`random.Random` derived from `(seed, position)` — because the engine takes a uniform variate and
holds no RNG of its own.

**`serve` serves one request at a time, and says so.** A C `Session` holds one `position_` and one KV
cache with no lock in `src/` or `native.py`, while the HTTP server is a thread-per-request
`ThreadingHTTPServer`; the adapter serializes behind a lock and declares `supports_batch = False`
rather than letting the handler promise overlap the engine cannot provide. Measured with six
concurrent requests: 6/6 correct with the lock, 0/6 without, and **no error either way** — the
failure is silent wrong text with a 200, which is why this is a correctness constraint and not a
performance note. `supports_cancellation = False` for the same kind of reason: `Session::forward`
runs to completion and nothing in the ABI observes a flag mid-call.

So: do not describe the *Python package* as able to run a model, do not describe *the tree* as unable
to, and do not describe the C core and the Python backends as one implementation — they are two, at
different levels of done, joined at the host shell (two entry points, one bridge). Prefer the README's
status table to any prose here.

## Language convention

**All Markdown documents and code comments in this project must be written in English**, unless a
Chinese version is explicitly requested as an additional deliverable.

When a Chinese version is requested, keep it as a separate file (see `README.md` / `README_CN.md`)
rather than mixing languages inside one file.

This applies to commit messages, code comments, docstrings, and all `.md` files.

## Documentation layout

**New documents go into the existing topic directory. Do not add a file at the top level of
`docs/`.** The top level holds the three *site entry points* and nothing else:

- `docs/README.md`, which is the published site's home page,
- `docs/getting-started.md`, which the `mkdocs.yml` nav pins at that path,
- `docs/llms.txt`, the machine-readable index of every published page.

`llms.txt` is not a document — it is a generated artifact of the nav, and it is the one file at that
level nobody writes by hand. Regenerate it with `scripts/gen_llms_txt.py` whenever a nav entry
changes, in the same commit. `mkdocs build --strict` fails when it is stale
(`.mkdocs/hooks/llms_txt_staleness.py`), and that build is the only check CI runs on a documentation
change.

Four directories, each with one subject:

| Directory | What belongs in it |
|---|---|
| `docs/guides/` | Rules and procedures — the release flow today |
| `docs/architecture/` | Design documents — the kernel ABI, the backend model, execution, roadmaps |
| `docs/models/` | What a model *is* here: the architecture registry, per-checkpoint guides |
| `docs/reports/` | Rendered long-form reports |

Filenames are lowercase `snake_case`. `docs/guides/` has an `index.md` and `docs/models/` has a
`README.md`; a new document has to be listed in one of those and in the nav, because one that is not
is unreachable except by guessing a path — so **update the index in the same commit**, add the page
to `mkdocs.yml`'s `nav`, and re-run `scripts/gen_llms_txt.py` in the same one. Relative links between
directories need the `../` prefix; **links back to the repository root cannot be relative** —
`README.md` and `README_CN.md` are outside `docs_dir`, so link them by absolute URL
(`https://github.com/lvyufeng/PocketLLM#…`), which is also how every cross-repository link is written.
Moving a file means fixing every inbound reference in the same commit (source comments and test
headers link here too, not just other Markdown).

This is not only a filing convention: `docs/` is the published site and the build runs
`mkdocs build --strict`, so a link that no longer resolves fails the page build rather than just
looking untidy.

## Repository layout

The Python tree lives under `python/`, which is where the C++ engine's `src/` will sit beside it.
That split is the design: `python/` is the *host side* — the spec, the numeric oracle and the CLI —
and `src/` is the *device side*, the native library that does the work.

| Path | What it is |
|---|---|
| `python/` | The Python package tree, and the only tree `pip install` builds. Every path below is relative to it. |
| `python/pocketllm/kernels/` | **The kernel ABI, and the heart of the tree.** Descriptors (`dtypes`, `device`, `buffer`, `tensor`), declarations (`schema`, `ops/*`, `registry`), and the machinery a backend is driven through (`backend`, `dispatch`, `graph`). **Stdlib-only at every level** — no numpy, no torch, no I/O — because if reading a shape needed numpy, a phone build could not be trimmed of numpy. It is also the **spec** the C ABI is derived from. |
| `python/pocketllm/quant/` | The GGML block decoders. A **leaf**: numpy and the vendored table header, and nothing else in `pocketllm`. Loader and reference backend both need it, and neither may own it. |
| `python/pocketllm/loader/gguf/` | The GGUF reader and quantized-tensor loader. numpy only — **no torch, no `relic_core`**. `vendor/ggml-common.h` is the vendored codebook table; see below. |
| `python/pocketllm/backends/` | Device implementations, one directory each, plus `registry.py` (the static table and the entry-point discovery), `base.py` (`RuntimeProbe`, `DeclaredBackend`, `UnimplementedSession`) and the conformance harness's subject. `reference/` is implemented; the other six are stubs. |
| `python/pocketllm/engine/` | Execution: `session.py` (device selection and policy), `planner.py` (regions), `executor.py` (the op-by-op walk), `captured.py`, `memory.py` (the arena and liveness) and `llm.py` (the `LLM`/`AsyncLLM` facade). |
| `python/pocketllm/architectures/` | The model IR (`ir.py`, `cache.py`, `registry.py`) and the builders. Only `toy` ships. |
| `python/pocketllm/api/`, `protocol/`, `server/`, `choices.py`, `tokenizer/`, `cli.py` | The intent types, the OpenAI-compatible HTTP surface, n-choice fan-out, the GGUF-vocabulary tokenizer skeleton, and the CLI. The last three are the **host shell** over the C core: `cli.py` and `server/` call the engine through `ctypes` rather than running a graph in Python. |
| `tests/` | pytest suite — see **Testing** below; `tests/README.md` is its own documentation. |
| `.mkdocs/`, `docs/`, `scripts/`, `mkdocs.yml` | The documentation site and the three scripts that keep it and the baseline honest. `.mkdocs/` holds the two **build-only** inputs — `overrides/` (`theme.custom_dir`) and `hooks/` (the `llms.txt` staleness guard); they are configuration, not content, so they sit in a dot-directory rather than at the repo root, and both paths are resolved relative to `mkdocs.yml`. |

**`python/pocketllm` is the whole installable package, and the only top-level package name the wheel
claims.** Its second root is not a Python one: `src/` is native source that no wheel ships and no
`pip install` compiles, so two trees coexist without a wheel-ownership question — a wheel that claims
a name another wheel also claims is the problem the single Python root avoids.

Three invariants the layout exists to protect:

- **The Python package is pure Python.** `pyproject.toml` declares no `ext_modules` and the package
  vendors no compiled artifact. The engine is a **separate native library** (`src/`, CMake) that the
  package loads at runtime through `ctypes`; it is not a Python extension, and no build step runs at
  `pip install`. `tests/test_package_boundaries.py` enforces the package half of that.
- **One process owns one device.** Nothing in this tree launches a second rank, a collective or a
  worker. `EngineArgs` has no `tensor_parallel_size`, no rank and no device-id list, and adding one
  back would resurrect a deleted feature. A checkpoint that does not fit is quantized further.
- **The ABI imports nothing.** `pocketllm.kernels` is stdlib-only at every level, enforced by
  `tests/test_package_boundaries.py`, which now walks `python/pocketllm/`. The dependency directions
  that file encodes (`kernels` → nothing, `quant` → numpy, `backends`/`loader` → kernels+quant,
  `engine` → the lot, `architectures` → kernels) are the design, not a preference.

### The vendored GGML header

`python/pocketllm/loader/gguf/vendor/ggml-common.h` **is** how this tree reads a codebook table. Before the
rebuild the loader resolved it through `Path(relic_core.__file__).parent / "csrc" / ...`, which made
reading a checkpoint require the kernel library — unacceptable for a phone install. It is resolved in
order by `pocketllm.quant.ggml_tables.header_path()`: `$POCKETLLM_GGML_COMMON` → the vendored copy →
relic-core's, last, with its absence not an error. The vendored copy's sha256 is pinned in
`ggml_tables.VENDORED_SHA256` and a local edit is refused rather than silently changing what every
decoder reads.

Non-`.py` payload needs both a `[tool.setuptools.package-data]` entry and a `MANIFEST.in` line, or an
installed wheel's loader dies at the first codebook read — a failure that does not appear in a source
checkout at all. `tests/test_package_boundaries.py` fails if a non-Python file appears in the package
without being listed.

## Hardware and toolchain

Two machines this project has development notes for. **Determine which one you are on before
concluding anything about what can be built, run, or measured.** The CUDA path does not exist on the
Ascend machine at all, and neither machine can run a model from this tree today.

### x86_64 CUDA machine — 4 x RTX 2080 Ti

- **GPUs**: 4 x RTX 2080 Ti, 22528 MiB each, compute capability **7.5 (Turing / sm_75)**. relic-core
  builds for sm_75; do not drop sm_75-specific kernel paths there. `src/kernel/cuda/` is built for
  sm_75 too (`POCKETLLM_CUDA_ARCHITECTURES`), and one process uses exactly one of these cards.
- **CPU / RAM**: 2 x Xeon E5-2696 v4, 22 cores each (88 hardware threads), ~1 TiB RAM.
- **OS / Python**: Ubuntu 22.04.5, x86_64, kernel 5.15. Python 3.10.10 (conda).
- **CUDA**: `nvcc` on `PATH` is **13.0** while `CUDA_HOME` points at **`/usr/local/cuda-12.4`**;
  11.8, 12.4 and 13.0 are installed. `/usr/local/cuda` resolves through
  `/etc/alternatives/cuda` to `cuda-13.0`, which is what CMake's search finds, so the CUDA backend
  builds against 13.0 without any path being pinned. Do not pin one anyway: the mismatch between
  `CUDA_HOME` and `nvcc` is a fact about this host, not about the next one.
  - **Historical trap, now resolved**: a pip-installed torch used to be built against CUDA 12.4
    while `nvcc` was 13.0, and `torch.utils.cpp_extension` hard-fails on that mismatch
    (`The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch
    (12.4)`). Torch is now `2.13.0+cu130` and the two agree. relic-core still compiles CUDA and
    would still hit this if torch were ever downgraded.
- **No NPU here**: no `/dev/davinci*` and no CANN.
- One relevant `git` note: `origin` is HTTP**S** (`https://github.com/lvyufeng/PocketLLM.git`) and
  there is no SSH key this host can authenticate to GitHub with. `gh` is authenticated separately as
  `lvyufeng` over `api.github.com`, which intermittently times out — retry a `gh` command that fails.
  `github.com` over HTTPS answers.

### aarch64 Ascend machine — 8 x Ascend 910B (historical notes)

**Nothing in this tree runs there, and the Ascend notes below describe a chip that is not this
project's target.** The `ascend` backend targets the **310B** (Orange Pi AIpro / Atlas 200I), a
different generation with different CANN support and different `aclgraph` behaviour. The recorded
910B facts are kept only because they are the sole description of that host in these notes, and
**none of them can be verified from the x86_64 machine** — treat them as claims to re-check in place,
and check them against the machine only if Ascend work resumes.

- **NPU**: 8 x Ascend `910B` with no trailing digit, i.e. **1st generation**
  (`Short_SoC_version=Ascend910`); 32 GB HBM per card, `/dev/davinci0-7`. See
  [the naming convention](#ascend-chip-naming-convention) before trusting that name.
- **CANN**: 9.0.0, `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`. Driver 25.5.2.
- **OS**: Ubuntu 22.04.5, aarch64, kernel 5.15. No CUDA toolchain.
- One warning worth carrying: an ACL binary launched without CANN's own `set_env.sh` does not fail —
  it *hangs* before `aclInit` returns.

## Ascend chip naming convention

**`910B` with no trailing digit is first generation; `910B1`–`910B4` are second generation.** The
name `npu-smi info` prints is not the SoC generation, so read `Short_SoC_version` from
`$ASCEND_TOOLKIT_HOME/<arch>-linux/data/platform_config/*.ini` before making any judgement about which
hardware you are on. The two generations need **separate AscendC kernel implementations, not retuned
parameters**.

The 310B target that `backends/ascend/` names is a third case again, and the board itself — Orange Pi
AIpro or Atlas 200I — plus its CANN version is an open question recorded in
[the device targets page](docs/architecture/devices.md#ascend-310b-not-910b).

## Git workflow

**Never commit directly to `main`.** Every change goes on a branch and through a pull request.

> **Bootstrap exception.** The `main` branch is an **orphan**: its first commit is a seed with no
> ancestry to the previous tree. A pull request needs a shared base ref, so that one commit could not
> come through one. From the second commit on, the rule has no exceptions.

Branch prefixes:

- `feature/<description>` — new features (e.g. `feature/cuda-gemm-quant-kernel`)
- `fix/<description>` — bug fixes (e.g. `fix/decode-eos-handling`)
- `refactor/<description>` — refactoring (e.g. `refactor/dispatch-trace`)
- `docs/<description>` — documentation (e.g. `docs/kernel-abi-v1`)
- `perf/<description>` — performance work (e.g. `perf/qnn-graph-granularity`)

Pull requests: a title under 72 characters, a body covering the summary, implementation details and
testing status, and **one concern per PR** — break large features into several. Every PR body must
end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Commits: a one-line summary under 72 characters, a blank line, then the explanation starting on line
3. Every commit message must end with
`Co-Authored-By: Claude Code <noreply@anthropic.com>`.

The commit trailer's address is `noreply@anthropic.com`, **not** the `anthropic.com` one this file
used to spell — commits carrying the old address are attributed to the wrong identity. Do not
"normalize" the two toward each other in either direction: the old address is still correct on the
history that already has it, and rewriting it would be a force-push over other people's commits.

Merged branches are **not** reliably deleted on `origin`, so delete yours yourself, locally and
remotely.

**Emergency hotfixes** may go directly to `main` for critical production issues only: a clear commit
message explaining the emergency, an immediate follow-up PR, and a post-mortem if it was severe.
This should be well under 1% of commits.

### The `legacy` branch

The previous tree is preserved as `legacy`: full history, both tags, and everything that was cut. It
is **not** deleted. If it is ever retired, move it to `refs/archive/` rather than deleting it, so
GitHub does not eventually garbage-collect the tag targets. The roughly thirty-five stale `origin/*`
branches predate the rebuild and are unreachable from `main`.

## Testing

The suite is `tests/`, which is pytest-collectable only — the `bench_*` / `probe_*` / `summarize_*`
scripts the old tree carried are gone. The package moved to `python/`, so the import path comes from
`[tool.pytest.ini_options] pythonpath = ["python"]` in `pyproject.toml` plus a `tests/conftest.py`
that exports the same path to the fresh interpreters several probes spawn. Both are read from the
root, so **run pytest from the repository root**:

```bash
python -m pytest tests/ -q
```

`tests/README.md` is the suite's own documentation: what pytest collects, the skip policy, the
baseline, and what CI does and does not run. Read it before changing how the suite is invoked.

- **The baseline is empty**, and `tests/baseline_failures.txt` records the known failures as a set of
  node ids. Diff a run against it with `python scripts/check_test_baseline.py` (`--update` to
  re-record). It is a set and not a count on purpose: three tests fixed and one broken is a net
  improvement in a count and a regression in the tree.
- Modules that need a GPU or a real checkpoint **skip** themselves — and a skip is not a pass. A test
  that fails on this host belongs in the file; one skipped for want of a card or a checkpoint does
  not. The `reference` backend is numpy and always available, so the ABI and the executor never skip
  wholesale.
- **A declared op needs a reference implementation in the same commit.**
  `tests/abi/test_reference_completeness.py` asserts `OPS.names() ⊆ reference.capabilities()`.
- **A new backend must be conformance-clean**: every op in `capabilities()` runs and matches the
  reference's numerics. `tests/backends/conftest.py` is the harness.
- **There are no golden fixtures yet**, and no test that replays them. A fixture records one real
  request through one real entry point; there is no entry point that can produce an answer until an
  architecture and a backend exist. `scripts/record_golden_fixture.py` is ready and its docstring
  says so.
- **CI runs no tests.** `.github/workflows/pages.yml` builds the documentation site with
  `mkdocs build --strict` (which is also the link check and the `llms.txt` staleness gate), and
  `.github/workflows/publish-pypi.yml` builds and uploads a release. The baseline check is a manual
  step and a gate only where somebody runs it.

## Documentation build

```bash
pip install -r requirements-docs.txt   # mkdocs-material, pinned
mkdocs build --strict
python scripts/gen_llms_txt.py --check
```

`--strict` promotes warnings to errors, which is what makes the build the repository's link checker.
Do not "fix" a failing link by adding an exemption — the exemption is how a 404 ships.