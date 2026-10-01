# PocketLLM

Single-card inference engine: run a large model on **one accelerator**, edge and mobile. The rule
the scope reduces to is *if it does not fit, quantize it* — never offload, never split across cards.
The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary.

This repository is one of four:

| Repository | What it is |
|---|---|
| **PocketLLM** (this tree) | Single-card / edge runtime. One process owns one card. |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | Multi-GPU PyTorch runtime and serving shell. |
| [relic-core](https://github.com/lvyufeng/relic-core) | The shared torch operator library (CUDA sm_75 + CPU host ops), installed here as the `relic_core` package. |
| relic-engine | The retired `cpp_engine` tree, kept only as a frozen archive. Nothing here builds against it. |

`docs/README.md` indexes the documentation — model support status, benchmarking rules, and the
release procedure are documented there.

## Language convention

**All Markdown documents and code comments in this project must be written in English**, unless a
Chinese version is explicitly requested as an additional deliverable.

When a Chinese version is requested, keep it as a separate file (see `README.md` / `README_CN.md`)
rather than mixing languages inside one file.

This applies to commit messages, code comments, docstrings, and all `.md` files.

## Documentation layout

**New documents go into the existing topic directory. Do not add a file at the top level of
`docs/`.** The top level holds the three *site entry points* and nothing else:

- `docs/README.md`, which indexes the directories,
- `docs/getting-started.md`, which the `mkdocs.yml` nav pins at that path,
- `docs/llms.txt`, the machine-readable index of every published page.

`llms.txt` is not a document — it is a generated artifact of the nav, and it is the one file at that
level nobody writes by hand. Regenerate it with `scripts/gen_llms_txt.py` whenever a nav entry
changes, in the same commit. `mkdocs build --strict` fails when it is stale
(`hooks/llms_txt_staleness.py`), and that build is the only check CI runs on a documentation change.

Four directories survive the multi-card cut. The topics that left with it — `performance/`,
`migration/` and `archive/` — went with the code they measured, which is now in RelicLLM, relic-core
and relic-engine; do not recreate them here.

| Directory | What belongs in it |
|---|---|
| `docs/guides/` | Rules and procedures — release flow today |
| `docs/architecture/` | Design documents, refactor plans, **roadmaps**, engine comparisons |
| `docs/models/` | Per-checkpoint guides and the support matrix |
| `docs/reports/` | Rendered long-form reports |

Filenames are lowercase `snake_case`. `docs/guides/` has an `index.md` and `docs/models/` has a
`README.md`; a new document has to be listed in one of those and in `docs/README.md`, because one
that is not is unreachable except by guessing a path — so **update the index in the same commit**,
and add the page to `mkdocs.yml`'s `nav` and re-run `scripts/gen_llms_txt.py` in the same one.
Relative links between directories need the `../` prefix; **links back to the repository root
cannot be relative** — `README.md` and `README_CN.md` are outside `docs_dir`, so link them by
absolute URL (`https://github.com/lvyufeng/PocketLLM#…`), which is also how every cross-repository
link is written. Moving a file means fixing every inbound reference in the same commit (source
comments and test headers link here too, not just other Markdown).

This is not only a filing convention: `docs/` is the published site and the build runs
`mkdocs build --strict`, so a link that no longer resolves fails the page build rather than just
looking untidy.

## Repository layout

| Path | What it is |
|---|---|
| `pocketllm/` | **The whole installable package, and the only top-level package name this wheel claims.** `models/xing4_0/` (the one model runtime), `loader/gguf/` (the GGUF decoders), `components/gguf/`, plus the CLI, the HTTP server and `backends/` — one runtime, `backends/xing4_backend.py`; the rest of `backends/` is the dispatch machinery (declarations, refusals, option surfaces) that the single runtime still reads. There is no `src/` tree any more: it was folded in here so the tree stops claiming a generic package name RelicLLM's wheel also claimed. |
| `tests/` | pytest suite — see **Testing** below. |
| `hooks/` | MkDocs build-time checks, registered under `hooks:` in `mkdocs.yml`. One file: it fails the docs build when `docs/llms.txt` is stale. |
| `docs/` | The three topic directories above, indexed by `docs/README.md`. Also the source of the published site: `mkdocs.yml` points `docs_dir` at it and `.github/workflows/pages.yml` builds it to <https://lvyufeng.github.io/PocketLLM/>, where `docs/llms.txt` is published as the machine-readable index. New files go in a topic directory, never at the top level; see **Documentation layout** above. |

Two invariants the layout exists to protect:

- **The install is pure Python.** `setup.py` has no `ext_modules`; every native kernel lives in
  relic-core, reached through the installed `relic_core` package. Do not reintroduce a compile step
  here, and do not vendor a kernel into `pocketllm/` — the vendored GGML tables (for example
  `ggml-common.h`, read by `pocketllm/loader/gguf/iq4_nl.py`) are resolved *through*
  `relic_core.__file__`, not by a path relative to this tree.
- **One process owns one card.** Nothing in this tree launches a second rank or a collective. A
  checkpoint that does not fit is quantized further, not split.
- **One top-level package name.** `pocketllm/` is the only root `setup.py` packages. Do not
  reintroduce a second root (the generic `src` was one) — a wheel that claims a name another wheel
  also claims has no defined owner, and install order silently decides which tree wins.

Quantized kernel dispatch on the Python side goes through `relic_core.kernels.ops`
(`_auto_impl` / `_resolve_impl`), with paired `*_torch` / `*_triton` implementations behind it —
that module is relic-core's now, not this tree's.

## Hardware and toolchain

Two development machines. **Determine which one you are on before concluding anything about what can
be built, run, or measured.** A command that is correct on one is usually wrong on the other — most
visibly, the CUDA path does not exist on the Ascend machine at all.

### x86_64 CUDA machine — 4 x RTX 2080 Ti

- **GPUs**: 4 x RTX 2080 Ti, 22528 MiB each, compute capability **7.5 (Turing / sm_75)**.
  relic-core builds for sm_75; do not drop sm_75-specific kernel paths there.
- **Topology**: `GPU0-GPU1` are PHB (PCIe, same NUMA node); **`GPU2-GPU3` are NV2 (NVLink)**; every
  cross-pair is SYS. This tree runs one card per process, so the topology matters only when several
  processes are started on the same box and their cards are chosen by `--device-ids` — a fair
  side-by-side comparison still wants physical GPUs **2 and 3**.
- **CPU / RAM**: 2 x Xeon E5-2696 v4, 22 cores each (88 hardware threads), 2 NUMA nodes, ~1 TiB RAM.
  GPUs 0-1 sit on NUMA node 0, GPUs 2-3 on node 1.
- **OS / Python**: Ubuntu 22.04.5, x86_64, kernel 5.15. Python 3.10.10 (conda).
- **Compilers**: gcc 11.4.0, cmake 3.26.3 (conda's, first on `PATH`). Neither is needed here any
  more — relic-core does the compiling.
- **CUDA**: `nvcc` on `PATH` is **13.0** (`/usr/local/cuda` → 13.0) while `CUDA_HOME` points at
  **`/usr/local/cuda-12.4`**; 11.8, 12.4 and 13.0 are all installed.
  - **Trap**: a pip-installed torch is built against CUDA 12.4, and `torch.utils.cpp_extension`
    hard-fails on the mismatch (`The detected CUDA version (13.0) mismatches the version that was
    used to compile PyTorch (12.4)`). This tree no longer compiles anything, but relic-core does:
    keep `CUDA_HOME` on a 12.x toolkit, and do not put `/usr/local/cuda-13.0/bin` ahead on `PATH`
    when building an extension against that torch.
- **No NPU here**: no `/dev/davinci*` and no CANN. The Ascend notes below are recorded from that
  machine, not verifiable from this host.

### aarch64 Ascend machine — 8 x Ascend 910B

**Nothing in this tree runs on that machine any more.** The Ascend runtime was the C++ engine's, and
it is archived in relic-engine; the build scripts that drove it (`scripts/ascend_env.sh`,
`scripts/build_ascend.sh`) went with it. The recorded facts below are kept because they are the only
description of that host these notes have — **none of them can be verified from the x86_64 host, so
treat them as claims to re-check in place**, and check them against the machine only if Ascend work
resumes.

- **NPU**: 8 x Ascend `910B` with no trailing digit, i.e. **1st generation**
  (`Short_SoC_version=Ascend910`); 32 GB HBM per card, `/dev/davinci0-7`. See **Ascend chip naming
  convention** below before trusting that name.
- **CANN**: 9.0.0, `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`. Driver 25.5.2
  (`ascendhal 7.35.23`).
- **OS**: Ubuntu 22.04.5, aarch64, kernel 5.15. Compilers: gcc 11.4.0.
- **No CUDA toolchain**, so CUDA builds and 2080 Ti regression runs cannot happen there.
- **Build (historical)**: `source scripts/ascend_env.sh` first — CANN's own `set_env.sh` is required,
  not just `LD_LIBRARY_PATH`, because an ACL binary launched without it does not fail but *hangs*
  before `aclInit` returns. Then `scripts/build_ascend.sh`. Both scripts are in relic-engine now.
- `/etc/hccn.conf` exists but is empty: multi-card **HCCL over RDMA** needs it configured first.
  Intra-server SDMA does not depend on it.

## Network access

- **`origin` is HTTPS**: `https://github.com/lvyufeng/PocketLLM.git`. There is no SSH key this host
  can authenticate to GitHub with — `~/.ssh/` holds `id_ed25519` and no `_github` key,
  `~/.ssh/config` has no `github.com` entry, and `ssh -T git@github.com` is refused on port 22
  (checked 2026-09-29). A refused handshake says this host has no accepted key, not that a keyed
  route does not exist, so a host with the entry described in this bullet's earlier revision would
  still push over SSH.
- `gh` is authenticated separately, as the account `lvyufeng`. It uses `api.github.com`, which is a
  different path from git's — so a `gh` command can fail while a push succeeds.
- `github.com` over HTTPS answers, including the API's own paths (checked 2026-09-29; it was
  unreachable at some point before that, and the earlier SNI-filtering workaround no longer
  applies). **README badges and raw-file links over `raw.githubusercontent.com` and `camo.` have not
  been checked** and are not covered by any of the above.
- `api.github.com` is reachable but **intermittently times out**. `gh` commands — `gh pr list
  --json` in particular — may need a retry.
- PyPI and Test PyPI are reachable over HTTPS. `docs/guides/pypi_release.md` documents the release flow and
  where the credentials live.

## Ascend chip naming convention

**`910B` with no trailing digit is first generation; `910B1`–`910B4` are second generation.** The
name `npu-smi info` prints is not the SoC generation, so read `Short_SoC_version` from
`$ASCEND_TOOLKIT_HOME/<arch>-linux/data/platform_config/*.ini` before making any judgement about
which hardware you are on. The two generations need **separate AscendC kernel implementations, not
retuned parameters**.

Full table, platform_config layout, and the CMake variable that consumed it:
[docs/guides/ascend_soc_generations.md](https://github.com/lvyufeng/relic-core/blob/master/docs/guides/ascend_soc_generations.md)
— that page went to relic-core with the hardware notes.

## Git workflow

**Never commit directly to `master`.** Every change goes on a branch and through a pull request.

Branch prefixes:

- `feature/<description>` — new features (e.g. `feature/cpp-engine-batch-scheduler`)
- `fix/<description>` — bug fixes (e.g. `fix/decode-eos-handling`)
- `refactor/<description>` — refactoring (e.g. `refactor/unified-api-phase1`)
- `docs/<description>` — documentation (e.g. `docs/phase3-completion-summary`)
- `perf/<description>` — performance work (e.g. `perf/gqa-tensor-core`)

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

**Emergency hotfixes** may go directly to `master` for critical production issues only: a clear
commit message explaining the emergency, an immediate follow-up PR, and a post-mortem if it was
severe. This should be well under 1% of commits.

## Testing

The suite is `tests/`, which also still carries a few `bench_*` / `probe_*` / `summarize_*` scripts
that pytest does not collect (`tests/README.md` lists what is left). There is no `conftest.py` and
no pytest configuration; modules import from the repository root, so
**run pytest from the repository root**:

```bash
python -m pytest tests/ -q
```

`tests/README.md` is the suite's own documentation: what pytest collects and what it does not, the
skip policy, the golden fixtures, and what CI does and does not run. Read it before changing how the
suite is invoked.

- **The baseline is empty**, and `tests/baseline_failures.txt` records the known failures as a set of
  node ids. Diff a run against it with `python scripts/check_test_baseline.py` (`--update` to
  re-record). It is a set and not a count on purpose: three tests fixed and one broken is a net
  improvement in a count and a regression in the tree. A test that fails on this host belongs in the
  file; one skipped for want of a card or a checkpoint does not.
- Modules that need a GPU or a real checkpoint **skip** themselves — and a skip is not a pass.
- The **golden fixtures** in `tests/fixtures/golden/` are the only end-to-end claims: one real
  request through one real entry point per backend, compared against recorded token ids. One entry
  point survives (`xing4`); record its fixture with `scripts/record_golden_fixture.py`.
- **CI runs no tests.** `.github/workflows/publish-pypi.yml` builds and uploads a release, and
  `.github/workflows/pages.yml` builds the documentation site with `mkdocs build --strict`. No
  workflow runs the pytest suite, so the baseline check is a manual step and a gate only where
  somebody runs it.
