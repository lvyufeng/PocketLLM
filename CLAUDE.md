# PocketLLM

Multi-backend inference engine for DeepSeek-V4, MiniMax, GLM, and Qwen checkpoints. One repository,
two engines: a device-agnostic core shared by both, and a per-vendor kernel layer.

`docs/README.md` indexes the documentation — model support status, benchmarking rules, and the
release procedure are documented there.

## Language convention

**All Markdown documents and code comments in this project must be written in English**, unless a
Chinese version is explicitly requested as an additional deliverable.

When a Chinese version is requested, keep it as a separate file (see `README.md` / `README_CN.md`)
rather than mixing languages inside one file.

This applies to commit messages, code comments, docstrings, and all `.md` files.

## Repository layout

| Path | What it is |
|---|---|
| `cpp_engine/` | C++ engine. `core/` is device-agnostic, `engine/` is the layer model, and `backends/{api,cuda,ascend}/` holds the vendor code. Build and run instructions: `cpp_engine/README.md`. |
| `src/` | Python/PyTorch implementation and kernel library. |
| `pocketllm/` | The installed package: CLI, HTTP server, supervisor. Imports `pocketllm_cpp` when the native engine was built. |
| `tests/` | pytest suite — see **Testing** below. |
| `docs/` | Model guides, benchmark definitions, release procedure. |

Two invariants the layout exists to protect:

- **Kernels stay behind the C ABI.** `cpp_engine/include/cuda_ops.hpp` declares 93 ops, 91 of which
  take `void* stream`, and it includes no CUDA headers at all. `cpp_engine/engine/` likewise
  includes no CUDA headers and contains **zero `<<<` launches** — all 359 of them are under
  `backends/`. Adding a vendor-specific type to an op signature breaks the other vendor's build.
- **Backend selection happens at build time.** Per-hardware tuning is not traded away for
  portability, so do not "unify" a 2080 Ti or 910B specific kernel in the name of sharing code.

Quantized kernel dispatch on the Python side goes through `src/kernels/ops.py`
(`_auto_impl` / `_resolve_impl`), with paired `*_torch` / `*_triton` implementations behind it.

## Hardware and toolchain

Two development machines, one per backend. **Determine which one you are on before concluding
anything about what can be built, run, or measured.** A command that is correct on one is usually
wrong on the other — most visibly, the CUDA path does not exist on the Ascend machine at all.

### x86_64 CUDA machine — 4 x RTX 2080 Ti

- **GPUs**: 4 x RTX 2080 Ti, 22528 MiB each, compute capability **7.5 (Turing / sm_75)**.
  `cpp_engine` defaults to `CMAKE_CUDA_ARCHITECTURES=75`; do not drop sm_75-specific paths.
- **Topology**: `GPU0-GPU1` are PHB (PCIe, same NUMA node); **`GPU2-GPU3` are NV2 (NVLink)**; every
  cross-pair is SYS. NVLink-sensitive work and fair TP2 comparisons must run on physical GPUs
  **2 and 3**.
- **CPU / RAM**: 2 x Xeon E5-2696 v4, 22 cores each (88 hardware threads), 2 NUMA nodes, ~1 TiB RAM.
  GPUs 0-1 sit on NUMA node 0, GPUs 2-3 on node 1.
- **OS / Python**: Ubuntu 22.04.5, x86_64, kernel 5.15. Python 3.10.10 (conda).
- **Compilers**: gcc 11.4.0, cmake 3.26.3 (conda's, first on `PATH`).
- **CUDA**: `nvcc` on `PATH` is **13.0** (`/usr/local/cuda` → 13.0) while `CUDA_HOME` points at
  **`/usr/local/cuda-12.4`**; 11.8, 12.4 and 13.0 are all installed.
  - **Trap**: a pip-installed torch is built against CUDA 12.4, and `torch.utils.cpp_extension`
    hard-fails on the mismatch (`The detected CUDA version (13.0) mismatches the version that was
    used to compile PyTorch (12.4)`). Keep `CUDA_HOME` on a 12.x toolkit, and do not put
    `/usr/local/cuda-13.0/bin` ahead on `PATH` when building an extension against that torch.
- **No NPU here**: no `/dev/davinci*` and no CANN. `cpp_engine/build_ascend/` is a synced artifact
  tree; its presence tells you nothing about this host.

### aarch64 Ascend machine — 8 x Ascend 910B

These facts are recorded from that machine, not measured on the x86_64 host, so **none of them can
be verified from here** — treat them as claims to re-check in place.

- **NPU**: 8 x Ascend `910B` with no trailing digit, i.e. **1st generation**
  (`Short_SoC_version=Ascend910`); 32 GB HBM per card, `/dev/davinci0-7`. See **Ascend chip naming
  convention** below before trusting that name.
- **CANN**: 9.0.0, `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`. Driver 25.5.2
  (`ascendhal 7.35.23`).
- **OS**: Ubuntu 22.04.5, aarch64, kernel 5.15. Compilers: gcc 11.4.0.
- **No CUDA toolchain**, so CUDA builds and 2080 Ti regression runs cannot happen there.
- **Build**: `source scripts/ascend_env.sh` first — CANN's own `set_env.sh` is required, not just
  `LD_LIBRARY_PATH`, because an ACL binary launched without it does not fail but *hangs* before
  `aclInit` returns. Then `scripts/build_ascend.sh`.
- `/etc/hccn.conf` exists but is empty: multi-card **HCCL over RDMA** needs it configured first.
  Intra-server SDMA does not depend on it.

## Network access

- **`origin` is HTTPS**: `https://github.com/lvyufeng/PocketLLM.git`, with `gh` authenticated as
  `lvyufeng`. There is no SSH remote, no `~/.ssh/config` entry for `github.com`, and no deploy key;
  `ssh -T git@github.com` is refused on port 22. Use HTTPS.
- `github.com` over 443 works (checked 2026-09-14). The SNI-filtering workaround documented here
  previously no longer applies.
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

Full table, platform_config layout, and the CMake variable that consumes it:
[docs/guides/ascend_soc_generations.md](docs/guides/ascend_soc_generations.md).

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
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

Merged branches are **not** reliably deleted on `origin`, so delete yours yourself, locally and
remotely.

**Emergency hotfixes** may go directly to `master` for critical production issues only: a clear
commit message explaining the emergency, an immediate follow-up PR, and a post-mortem if it was
severe. This should be well under 1% of commits.

## Testing

The suite is `tests/`, alongside `bench_*` and `profile_*` scripts that pytest does not collect.
There is no `conftest.py` and no pytest configuration; modules import from the repository root, so
**run pytest from the repository root**:

```bash
python -m pytest tests/ -q
```

- `tests/test_gguf_q2_precision.py` fails at **collection**: it still imports `src.gguf.reader`,
  which moved to `src/loader/gguf/reader.py`. A bare `python -m pytest tests/` aborts on it, so add
  `--continue-on-collection-errors` to run the rest. Porting or deleting that module is the better
  fix.
- Modules that need a GPU, a real checkpoint, or a built `pocketllm_cpp` skip themselves via
  `pytest.importorskip`. A skip is not a pass.
- **CI runs no tests.** `.github/workflows/publish-pypi.yml` is the only workflow, and it only
  builds and uploads a release.
