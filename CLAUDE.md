# PocketLLM

## Language convention

**All Markdown documents and code comments in this project must be written in English**, unless a
Chinese version is explicitly requested as an additional deliverable.

When a Chinese version is requested, keep it as a separate file (see `README.md` / `README_CN.md`)
rather than mixing languages inside one file.

This applies to commit messages, code comments, docstrings, and all `.md` files.

## Ascend chip naming convention

**This is the most error-prone detail in this project. Read this section before making any judgement
about which hardware generation you are on.**

For the Ascend 910 series, the product name and the SoC generation are not the same thing. The name
shown by `npu-smi info` is misleading.

### The rule

**`910B` with no trailing digit belongs to the same generation as 910A (first generation).
`910B1` / `910B2` / `910B3` / `910B4` with a trailing digit are 910B (second generation).**

The only reliable discriminator is `Short_SoC_version` in the CANN `platform_config` files, not the
product name:

| platform_config | `Short_SoC_version` | Generation | AI Core | L2 | Cube freq |
|---|---|---|---|---|---|
| `Ascend910A` | **`Ascend910`** | 1st | 32 | 32 MB | 1000 MHz |
| `Ascend910B` | **`Ascend910`** | **1st** | 30 | 32 MB | 900 MHz |
| `Ascend910ProA` | **`Ascend910`** | 1st | 32 | 32 MB | 1100 MHz |
| `Ascend910ProB` | **`Ascend910`** | 1st | - | 32 MB | - |
| `Ascend910PremiumA` | **`Ascend910`** | 1st | - | 32 MB | - |
| `Ascend910B1` | **`Ascend910B`** | 2nd | 24 | 192 MB | 1850 MHz |
| `Ascend910B2` | **`Ascend910B`** | 2nd | 24 | 192 MB | 1800 MHz |
| `Ascend910B3` | **`Ascend910B`** | 2nd | 20 | 192 MB | 1800 MHz |
| `Ascend910B4` | **`Ascend910B`** | 2nd | 20 | 96 MB | 1500 MHz |

Platform config location (varies with the CANN install path):

```
$ASCEND_TOOLKIT_HOME/aarch64-linux/data/platform_config/*.ini
```

### Why this matters

Kernel tuning strategies cannot be shared across the two generations:

- **L2 differs by 6x** (32 MB vs 192 MB), which drives weight/KV L2 residency strategy and block sizes
- **Cube frequency differs by ~2x**, which shifts the compute/memory balance point and therefore the tiling
- **Only the 2nd generation has `cube_vector_combine=split`**, where Cube and Vector are independent
  units that can be pipelined in parallel. The 1st generation cannot do this.

So AscendC kernels must branch on `Short_SoC_version` with separate implementations, not merely
retuned parameters. This mirrors the principle on the CUDA side of not giving up 2080 Ti (sm_75)
specific optimizations.

### Do not detect it this way

```bash
npu-smi info | grep 910B      # WRONG: "910B" without a digit is actually 1st generation
```

Read `Short_SoC_version` instead, or resolve the exact model via `npu-smi` and look it up in the
table above.

## Hardware and toolchain (current dev machine)

- **NPU**: 8 x Ascend `910B` (i.e. **1st generation**, `Short_SoC_version=Ascend910`), 32 GB HBM per
  card, `/dev/davinci0-7`
- **CANN**: 9.0.0, `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`
- **Driver**: 25.5.2 (`ascendhal 7.35.23`)
- **OS**: Ubuntu 22.04.5, aarch64, kernel 5.15
- **Compilers**: gcc 11.4.0, cmake 3.22.1
- **No CUDA toolchain** on this machine (no `nvcc` / `nvidia-smi`). CUDA / 2080 Ti builds and
  regression runs must happen on a different machine.
- `/etc/hccn.conf` exists but is empty. Multi-card HCCL over RDMA needs it configured first.
  Intra-server SDMA (die-to-die) does not depend on it; the actual topology still needs to be
  measured.

## Network access

Port 443 on `github.com` is **blocked by SNI filtering** (DNS resolves fine, the TLS handshake
hangs). The same IP returns HTTP 200 when the SNI is `api.github.com` but times out with SNI
`github.com`, so switching IPs or omitting SNI does not help.

`git` is configured to bypass this via SSH over port 443:

```
# ~/.ssh/config
Host github.com
    HostName ssh.github.com
    Port 443
```

- `origin` uses SSH: `git@github.com:lvyufeng/PocketLLM.git`
- Authentication uses a repository-level **deploy key** (write-enabled), `~/.ssh/id_ed25519_github`
- `api.github.com` is reachable but flaky; `gh` commands may need a retry
- Read-only fetches can also use the mirror prefix `https://ghfast.top/https://github.com/...`
  (the mirror does not support push)

The repository was renamed from `deepseek-v4-2080ti` to `PocketLLM`.

## Architecture

Two largely independent engines with almost no cross-dependency:

- `cpp_engine/` — C++/CUDA engine (~52k lines). Kernels already sit behind a vendor-neutral C ABI
  (`include/cuda_ops.hpp` and friends: 91 of 93 declarations take `void* stream`, and the headers
  pull in no CUDA headers). The engine layer contains **zero `<<<` kernel launches**, and 13 of the
  22 files under `src/` have no CUDA references at all.
- `src/` — PyTorch implementation (~44k lines). `src/kernels/ops.py` already has an
  `_auto_impl` / `_resolve_impl` dispatch seam with paired `*_torch` / `*_triton` implementations.

The multi-backend refactor uses a **single repository with layered separation** rather than splitting
repos: a device-agnostic core is shared (GGUF parsing, tokenizer, HTTP server, scheduling skeleton),
while each vendor owns its own kernel implementations so that per-hardware optimization is never
compromised. Backend selection happens at build time.
