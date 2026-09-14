# Ascend SoC generations

For the Ascend 910 series the product name and the SoC generation are not the same thing, and the
name `npu-smi info` prints is the misleading one. Kernel tuning strategies cannot be shared across
the two generations, so this has to be resolved before judging which hardware you are on.

## The rule

**`910B` with no trailing digit belongs to the same generation as `910A` (first generation).
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

## Where the config lives

```
$ASCEND_TOOLKIT_HOME/<arch>-linux/data/platform_config/*.ini
```

The arch directory is not always the `uname` value — `arm64-linux` exists alongside `aarch64-linux`
— so probe rather than compose the path. `cpp_engine/CMakeLists.txt` does exactly that, trying
`aarch64-linux`, `arm64-linux`, then `x86_64-linux`, and failing with a named error if none holds
`include/acl/acl.h`.

The table above is not verifiable from this repository — no `platform_config` tree is checked in, and
the x86_64 development host has no CANN install. Read the files on the Ascend machine before relying
on a value for a decision.

## What consumes the value

`cpp_engine/CMakeLists.txt` sets `ASCEND_SOC_VERSION`, defaulting to `"ascend910b"`, and the comment
there records why that default is the *first* generation: CANN's own `host_config.cmake` groups
`ascend910b` with `ascend910a` and keeps `ascend910b1`..`b4` in a separate group, so the string
selects the first-generation codegen path. The tiling is tuned for 30 AI cores / 32 MB L2 / 900 MHz
cube and should not be changed without re-tuning.

## Why the generations are not interchangeable

- **L2 differs by 6x** (32 MB vs 192 MB), which drives weight/KV L2 residency and block sizes
- **Cube frequency differs by ~2x**, which shifts the compute/memory balance point and therefore the
  tiling
- **Only the 2nd generation has `cube_vector_combine=split`**, where Cube and Vector are independent
  units that can be pipelined in parallel. The 1st generation cannot do this.

AscendC kernels must therefore branch on `Short_SoC_version` with separate implementations, not
merely retuned parameters. This mirrors the CUDA side's refusal to give up 2080 Ti (sm_75) specific
optimizations for portability.

## Do not detect it this way

```bash
npu-smi info | grep 910B      # WRONG: "910B" without a digit is actually 1st generation
```

Read `Short_SoC_version` instead, or resolve the exact model via `npu-smi` and look it up above.
