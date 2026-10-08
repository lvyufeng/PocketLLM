# Ascend 310B custom ops

[Device targets](devices.md#ascend-310b-not-910b) gives the `ascend` backend's shape — 310B, not
910B, `STREAM_CAPTURE`/`STEP`, gated on `libascendcl.so` and a `/dev/davinci*` node. This page is
the one measurement-backed result behind that entry: what actually runs on a 310B, what had to be
built to make it run, and what the numbers are.

Everything here was measured on an Orange Pi AIpro 20T (`orangepiaipro-20t`, Ascend **310B1**,
aarch64, CANN **8.3.RC2**, npu-smi 23.0.0) on 2026-10-09. Whether a number transfers to another
310B board is a claim to re-check in place, not an assumption — the CANN version in particular is
part of the result.

## The built-in *matmul* has no 310B kernel

`aclnnMm` and the other built-in **matmul** ops on this CANN ship no `ascend310b` kernel binary, so
the built-in matmul path fails on the board. The 310B is a small-CANN board and the matmul the
910B takes for granted was simply not compiled for it.

**This is a fact about the matmul, not about every built-in, and the difference was measured.** The
tempting generalisation — "the built-ins have no 310B kernel" — is false: `aclnnEmbedding`,
`aclnnSoftmax` and `aclnnArgMax` all run on the board (phase-1 and phase-2 status 0, correct
values), and the backend drives them. What is true is that a header's *existence* proves nothing
and each built-in has to be run to learn whether a 310B binary is behind it — which is exactly why
the built-in matmul looked usable and was not. (The kernel binaries do not even live under the
expected name: the op `aclnnArgMax` runs from an `arg_max_v2` directory, `aclnnSoftmax` from
`softmax_v2`. The `ascend310b` factory listing is a bad index of what can run.)

The workaround for matmul is a set of **AscendC custom ops** with their own 310B kernel binaries.
They are not in this repository — they live in the adjacent
[minicpm-o-4.5-orangepi](https://github.com/lvyufeng/minicpm-o-4.5-orangepi) tree, under
`src/csrc/custom_ops/`, and the relevant ones are:

| Op | What it is |
|---|---|
| `MatmulW4a16Custom` | GPTQ int4 weight, fp16 activation matmul (M=1 fast path) |
| `MatmulW8a8I32Custom` | int8 × int8 → int32 matmul |
| `MatmulCubeCustom` | the cube-unit fp16 matmul (`MatmulImpl<half,half,half>`) |
| `RmsNorm1024Custom` | RMSNorm (bakes `HIDDEN_SIZE = 1024`, so only that width) |
| `RmsNormNdCustom` | RMSNorm with the row width from the shape (≤ 4096) |
| `SiluMulCustom` | SiLU-gated multiply |
| `AttentionStepCustom` | one decode attention step |
| `RopeCustom` | split-half rotary embedding, cos/sin table indexed by an INT32 row |

Three of them are `half`-only kernels (`RmsNormNdCustom`, `RopeCustom`, and the matmuls), which is
why the backend narrows f32 operands to fp16 for those ops and widens the result — the fp16-wide
tolerance those cases use is the format's width, not slack. Two of them (`RmsNormNdCustom`,
`RopeCustom`) move their operands in the AscendC vector pipe's 32-byte (16-half-word) repeat, so a
width that is not a whole number of repeats is refused by name: `rms_norm` needs `d % 16 == 0` and
`rope_neox` needs `d % 32 == 0`. Head dim 128 and hidden 1024 satisfy both; the suite's `d = 1`
norm and `d = 4` rope cases do not, and they are the two shape exclusions.

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

## The cube path is the backend's default

`MatmulCubeCustom` is `MatmulImpl<half,half,half>`, so the cube's L0C accumulator is fp32, and it
costs 1.4× *less* than the fp16 vector loop — a different order of magnitude from either W4A16
variant. It takes dequantized fp16 weights (`a[M,K] · b[K,N] → out[M,N]`), so the backend decodes a
packed checkpoint tensor to a persistent f32 buffer once per weight and drives the cube as
`a[M,K] · b[K,N]` with `b` the transposed weight plane.

**The backend uses it, and it is the measured correct path** — `gemm` and `gemm_quant` both route
here, and the whole Qwen3 forward runs on it. Measured against the CPU kernel at real model shapes
(1×1024×1024, 1 or 16 × 1024×3072) it is **~2.7e-4 relative**, the same order as the fp32-accumulate
W4A16 row above and for the same underlying reason (fp32 accumulate, fp16 operands). The table's
fp16-*vector*-loop row is not what the backend runs; it is the sibling work's measurement of a
different kernel.

**The standalone cube probe still fails phase 1 with `161001`**, and that discrepancy is
unexplained: the same op the backend drives successfully will not come up through the probe's
`aclnnMm`-style path. Until someone reconciles it, treat "the probe fails" as a fact about the
probe, not about the op — the backend's result is the one to trust, because it is the one a
token-for-token forward pass exercises end to end.

### The cube is not correct at every N

`MatmulCubeCustom` returns a right-shaped tensor of *wrong* numbers above an N this backend had to
find the hard way. The tied output projection is `gemm_quant` q6_K at **n = 151936**, and measured
against the CPU kernel the op came back **131% off** there, while n = 32768 and below matched to
5e-4. Bisected on the board, the unchunked op crosses from ~5e-4 to 100%+ between **n = 49152 and
n = 65536** (q4_k and q6_k alike — the failure is in the shared cube drive above the decode, not in
either format). It is a silent failure: right shape, plausible magnitude, wrong values.

The backend therefore walks N in 8192-column blocks, each a shape the op answers correctly, and
places the columns into the caller's row-major output by hand (`run_cube`, `kCubeChunkN`). The bound
is an order of magnitude under the failing edge so a tiling change on another board is far more
likely to land inside the safe range than outside it. This is pinned by
`case_gemm_quant_large_n_{q4_k,q6_k}` at n = 65600, which is over the edge and ends on a partial
final chunk — the only conformance cases whose N reaches the failing region.

## End to end: a full Qwen3 forward runs

The backend runs a complete Qwen3-0.6B forward — prefill and decode — and produces text **identical
to the CPU backend**:

```
pocketllm-run <Qwen3-0.6B q4_k_m> --device ascend --prompt "The capital of France is" --steps 8
  -> [12095 Paris 13. 576 The 6722 capital 315 of 9625 France 374 is 1083 also]   # == --device cpu
```

Every op on the greedy path is implemented: `rms_norm`, `gemm`, `gemm_quant` (q4_k and q6_k),
`embedding`/`embedding_quant`, `silu_mul`, `rope_neox`, `attention` (decode and prefill), and the KV
append. `softmax`, `argmax`, `topk_sample` and `logits_temperature` are host-side in `run.cpp`, not
graph ops, so the sampler tail is not a backend gap.

Wrong answers are still possible where the trilogy of numeric width, tuned kernel, and API surface
disagree — this is the page that documented the W4A16 fp16-accumulate floor and the cube's
large-N edge — but the *coverage* gap is closed: there is no op the graph calls that the backend
refuses. The paths the whole forward still lacks are the ones the graph never calls: `gemm` bias,
attention with a sliding window (`first_key != 0`), and `topk_sample`/`logits_temperature`.

The cost is real. Decoding a packed checkpoint to f32 weights is ~2 GB resident on the NPU for a
0.6B q4_k_m file (~0.4 GB packed), and the prefill of a five-token prompt takes minutes against the
CPU's milliseconds. This is a correctness path, not a fast one.
