# Ascend 310B custom ops

[Device targets](devices.md#ascend-310b-not-910b) gives the `ascend` backend's shape — 310B, not
910B, `STREAM_CAPTURE`/`STEP`, gated on `libascendcl.so` and a `/dev/davinci*` node. This page is
the one measurement-backed result behind that entry: what actually runs on a 310B, what had to be
built to make it run, and what the numbers are.

Everything here was measured on an Orange Pi AIpro 20T (`orangepiaipro-20t`, Ascend **310B1**,
aarch64, CANN **8.3.RC2**, npu-smi 23.0.0) on 2026-10-09. Whether a number transfers to another
310B board is a claim to re-check in place, not an assumption — the CANN version in particular is
part of the result.

## The built-in matmul has no 310B kernel

`aclnnMm` and the other aclnn *built-in* ops on this CANN ship no `ascend310b` kernel binary, so
the built-in matmul path fails on the board. The 310B is a small-CANN board and the ops the
910B takes for granted were simply not compiled for it.

The workaround is a set of **AscendC custom ops** with their own 310B kernel binaries. They are
not in this repository — they live in the adjacent
[minicpm-o-4.5-orangepi](https://github.com/lvyufeng/minicpm-o-4.5-orangepi) tree, under
`src/csrc/custom_ops/`, and the relevant ones are:

| Op | What it is |
|---|---|
| `MatmulW4a16Custom` | GPTQ int4 weight, fp16 activation matmul (M=1 fast path) |
| `MatmulW8a8I32Custom` | int8 × int8 → int32 matmul |
| `MatmulCubeCustom` | the cube-unit fp16 matmul (`MatmulImpl<half,half,half>`) |
| `RmsNorm1024Custom` | RMSNorm |
| `SiluMulCustom` | SiLU-gated multiply |
| `AttentionStepCustom` | one decode attention step |

## Installation, and the env the two-phase call requires

The package is a makeself archive (`custom_opp_ubuntu_aarch64.run`). It installs for the current
user, with no root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh          # installer reads ASCEND_OPP_PATH
bash custom_opp_ubuntu_aarch64.run --quiet --install-path=$HOME/Ascend/custom_opp
source $HOME/Ascend/custom_opp/vendors/customize/bin/set_env.bash
```

The installer writes `vendors/customize/` under the given path and generates that second
`set_env.bash`, which sets exactly two things: `ASCEND_CUSTOM_OPP_PATH` to
`<path>/vendors/customize`, and `<path>/vendors/customize/op_api/lib` onto `LD_LIBRARY_PATH`.

**That env is not optional, and its failure mode is a lie.** With `ASCEND_CUSTOM_OPP_PATH` unset —
or pointing one level too high, at `<path>/custom_opp` instead of `<path>/custom_opp/vendors/customize`
— the op is never registered with Nnopbase. The two-phase aclnn call then fails at phase 1:

```
phase1 GetWorkspaceSize: status=161001  executor=NULL
```

`161001` is `ACLNN_ERR_PARAM_NULLPTR`, which reads as "you passed a null tensor". It is not: every
`aclTensor` is non-null and the shapes are correct. What is null is the executor that
`NnopbaseGetExecutor` returned, because the op type was never found. Exported correctly, the same
call returns `status=0`. The variable must be set **before `aclInit`** — the registry is read when
the runtime initialises — and `source .../bin/set_env.bash` is the reliable way to set it.

## Shape constraints

From the host tiling and the call-site guards:

- `K % 128 == 0`, `N % 128 == 0` (group size is fixed at 128)
- `tileLen % 16 == 0` and `tileLen <= 448`; `tileLen` is the caller's choice, taken from `w.shape[1]`
- `blockLen = ceil(N/8)`, `tilesPerBlock = ceil(blockLen / tileLen)`
- `w` (int8) packed rows `= (K/128) * 8 * tilesPerBlock * 128`
- `x` fp16 `[1,K]`, `scales` fp16 `[K/128, N]`, `out` fp16 `[1,N]`

The int4 weight is repacked once, by `(group, core, tile, k)`, so each 128-row group/tile is one
contiguous DMA instead of 128 strided row copies.

## Precision: fp32 accumulate is required

The op originally accumulated in fp16 — `partial` and `acc` were `LocalTensor<half>` and the inner
loop was `Axpy`/`Mul`/`Add` on `half`. Measured against a float64 CPU reference at K=512, N=128,
G=128 (`max|op − fp64| / max|ref|`):

| accumulate | K=512 | K=2048 | time K=512 | time K=2048 |
|---|---|---|---|---|
| fp16 | 2.5e-3 | 2.5e-3 | 0.159 ms | 0.53 ms |
| **fp32** | **3.0e-4** | **2.8e-4** | 0.219 ms | 0.70 ms |

Both are **flat in K** — the fp16 error is not drift that grows with depth, it is a per-step
rounding floor: the loop rounded the product through fp16 on every one of the K steps, where a
cube-style fp16 accumulate keeps the product in fp32.

**fp32 accumulation is what a Qwen3 310B backend must use** — 2.5e-3 fails a 1e-3 budget, 3.0e-4
clears it by ~3×, and it holds as K grows. The cost is 1.4× the fp16 loop, because fp32 `Axpy`
processes half the elements per cycle.

The change is small and is preserved in the minicpm tree as
`op_kernel/matmul_w4a16_custom.cpp` (branch `fix/w4a16-fp32-accumulate`): make `partial`/`acc`/the
scale row fp32 and cast to fp16 once at the store. What makes it fit in the 192 KB UB is that the
weight tile stays fp16-sized — `Axpy` with an fp32 destination and an fp16 source dispatches to the
dav_m310 mixed-width path, which converts and multiplies in fp32. Converting the weight buffer
itself to fp32 would need 224 KB at the widest tile.

## The cube path

`MatmulCubeCustom` is the other candidate and is the numerically correct one on paper: it is
`MatmulImpl<half,half,half>`, so the cube's L0C accumulator is fp32, and it costs 1.4× *less* than
the fp16 vector loop — a different order of magnitude from either W4A16 variant. It takes
dequantized fp16 weights (`a[M,K] · b[K,N] → out[M,N]`), so a backend would need a host-side or
fused int4→fp16 dequantize first, which adds `K·N·2` bytes of GM per layer.

It is the likely long-term answer, but it is **not yet independently verified**: the standalone
cube probe currently fails phase 1 with the same `161001` as above, and until that runs there is no
trustworthy cube number to compare against the table. The W4A16 row is the measured one.
