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

## Precision: two different errors, and which one is the graph's

There are **two** error numbers in play for this op, and the sign of the whole "should the backend
drive the packed weights?" question turns on keeping them apart. They are measured at different
things and they are three orders of magnitude apart.

**(1) The op's own arithmetic** — how far the kernel's sum is from a float64 reference computed
from the *same* int8 weights. The op originally accumulated in fp16 — `partial` and `acc` were
`LocalTensor<half>` and the inner loop was `Axpy`/`Mul`/`Add` on `half`. Measured at K=512, N=128,
G=128 (`max|op − fp64| / max|ref|`), and again at two real model matrices:

| accumulate | K=512 | K=2048 | 1024×3072 `ffn_gate` | 3072×1024 `ffn_down` |
|---|---|---|---|---|
| fp16 | 2.5e-3 | 2.5e-3 | **1.19e-3** | **1.82e-3** |
| **fp32** | **3.0e-4** | **2.8e-4** | **3.05e-4** | **3.33e-4** |

(The K=512/2048 columns are the op's original N=128 probe; the two matrix columns are the real
weights and shapes, measured this round. `ffn_down` is q6_k and sits a little higher — same order.)
fp16 accumulation fails a 1e-3 budget at the toy shape *and* at the real ones; fp32 accumulation
passes at both. Both are **flat in K** and flat in shape — it is a per-step rounding floor, not
drift: the fp16 loop rounded the product through fp16 on every one of the K steps. The fp16 loop is
the *faster* one, 0.159 ms (K=512) and 0.53 ms (K=2048) against fp32's 0.219 and 0.70 ms — 1.4×,
because fp32 `Axpy` processes half the elements per cycle.

**(2) Requantization** — how far the int4 weights the op is *given* are from the q4_K/q6_K weights
the checkpoint actually holds. The op's input is GPTQ int4 packed as int8 with a per-128 fp16 scale;
the checkpoint's tensor is a q4_K super-block with six-bit sub-scales. Bridging them means decoding
the block faithfully and rounding to `[-8, 7]` per 128 — **a coarser quantizer**, and this is where
the real error lives. Measured against the tree's own `dequant_q4_k`/`dequant_q6_k`, at the model's
own matrices (weights loaded straight from `Qwen3-0.6B-Q4_K_M.gguf`):

| matrix | shape (k×n) | type | requant ‖Wr−Wf‖/‖Wf‖ | op arithmetic (fp32 acc) | **total** |
|---|---|---|---|---|---|
| `blk.0.ffn_gate` | 1024×3072 | q4_k | 1.19e-1 | 3.05e-4 | **9.4e-2** |
| `blk.0.attn_q` | 1024×2048 | q4_k | 1.23e-1 | 4.24e-4 | **1.7e-1** |
| `blk.0.ffn_down` | 3072×1024 | q6_k | 1.24e-1 | 3.33e-4 | **1.3e-1** |
| `token_embd` (tied head) | 1024×151936 | q6_k | 1.17e-1 | 3.91e-4 | **1.4e-1** |

**The two numbers do not contradict each other — they answer different questions.** `3.0e-4` is row
(1), the arithmetic, and it *does* meet a 1e-3 budget. The `~1e-1` is row (2), the requantization,
and it does *not*: the total error a model built on this path sees is dominated by the quantizer,
so it fails 1e-3 by two orders of magnitude at every real shape, and it is flat in n (the 151936
tied head is the same ~1.2e-1 as a k=1024 projection — the failure is the quantizer, not any
large-N behaviour, unlike the cube's silent edge above).

**Even int8 does not rescue it.** The kernel's weight load is `Cast(wFp16, wInt8, CAST_NONE)` — a
plain int8→fp16 cast with no 4-bit mask — so the op can in fact be handed full int8 weights; the
"W4" is the kernel's *name*, not a constraint it enforces. Requantizing to `[-127, 127]` per 128
instead of `[-8, 7]` drops the requantization from 1.19e-1 to **6.6e-3** and the total to 4.9e-3 on
`ffn_gate` — better by ~20×, still 5–7× past 1e-3. The int8 requantization is a *finer* quantizer
than q4_K's six-bit sub-scales only in step size, not in total error: q4_K already carries a per-32
six-bit scale, so re-expressing the same values through int8 does not recover the resolution q4_K's
own encode has.

**So the W4A16 op is not the decode GEMM's answer, and the packed weights cannot be fed to it.**
`$POCKETLLM_ASCEND_W4A16=1` still selects it, for an experiment where the requantization is the
thing being measured rather than a surprise. The graph's path stays the dense cube on the decoded
plane — the fp16 plane is 2× smaller than the f32 it replaced, and the two numbers above are why it
is decoded faithfully instead of requantized.

**The op arithmetic table is still true and still worth having** — it is why the fp32-accumulate
rebuild exists. fp16 accumulation (2.5e-3) alone fails a 1e-3 budget; the rebuild is preserved in
the minicpm tree as `op_kernel/matmul_w4a16_custom.cpp` (branch `fix/w4a16-fp32-accumulate`): make
`partial`/`acc`/the scale row fp32 and cast to fp16 once at the store. What makes it fit in the
192 KB UB is that the weight tile stays fp16-sized — `Axpy` with an fp32 destination and an fp16
source dispatches to the dav_m310 mixed-width path, which converts and multiplies in fp32.
Its cost is 1.4× the fp16 loop, because fp32 `Axpy` processes half the elements per cycle. It is
the right *op* — but it is fed weights it cannot be given without losing more than it saves.

### Where the names came from, and how this page got it wrong

`#582` chose the cube "because W4A16 requantizes to int8 at ~1e-1". The `~1e-1` was right and the
reason was right, but the page recorded it next to the `2.5e-3`/`3.0e-4` arithmetic table without
saying the two measure different things — so a reader saw "W4A16 ~1e-1" and "W4A16 fp32-acc ~3.0e-4"
in the same tree and read them as a contradiction. They are not: **`3.0e-4` is the kernel's
arithmetic on already-int4 weights, `~1e-1` is the int4 quantization of the checkpoint's q4_K
weights, and the second is 300× larger, so it is the one the model sees.** Both rows are measured
above, from the same probe, at the model's real shapes.

## The cube path is the backend's default

`MatmulCubeCustom` is `MatmulImpl<half,half,half>`, so the cube's L0C accumulator is fp32, and it
costs 1.4× *less* than the fp16 vector loop — a different order of magnitude from either W4A16
variant. It takes dequantized fp16 weights (`a[M,K] · b[K,N] → out[M,N]`), so the backend decodes a
packed checkpoint tensor to a persistent buffer once per weight and drives the cube as
`a[M,K] · b[K,N]` with `b` the transposed weight plane.

**The transposed plane is the thing to cache, not the f32 weight.** The op's B operand is `[K,N]`
where the graph holds `[N,K]`, and it is *fp16*, so it is not the decoded weight's bytes with a
different stride — it is a transposed copy in a narrower format. Building that plane on the host
inside every GEMM call was **~85% of a decode step** (31.5 s of ~37 s over three tokens): a scalar
walk over every element of every weight, per token. The plane is a pure function of the weight, so
it is built once and cached; the decode step does no weight-side host work at all. As a side effect
the cached artifact is *half* the size of the f32 buffer it replaced (~1 GB instead of ~2 GB for a
0.6B q4_k_m checkpoint).

**The embedding table is the same cache, for the same reason.** `embedding_quant` decodes the packed
`token_embd.weight` table (`151_936 × 1024`) on the host to gather from it, and it used to do that
in full **on every call** — once per token. It was the largest cost *left* after the cube plane was
cached. The decoded table is now cached once and only the (device `aclnnEmbedding`) gather runs per
token.

**The backend uses the cube, and it is the measured correct path** — `gemm` and `gemm_quant` both
route here, and the whole Qwen3 forward runs on it. Measured against the CPU kernel at real model
shapes (1×1024×1024, 1 or 16 × 1024×3072) it is **~2.7e-4 relative**, the same order as the
fp32-accumulate W4A16 row above and for the same underlying reason (fp32 accumulate, fp16
operands). The table's fp16-*vector*-loop row is not what the backend runs; it is the sibling work's
measurement of a different kernel.

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

**The cost was real, and the two caches above are what paid it down.**
`Qwen3-0.6B-Q4_K_M`, `--prompt "The capital of France is"`, measured on the board:

| | decode, marginal | prefill | `--steps 8` wall | weights resident |
|---|---|---|---|---|
| before | **14.6 s/token** | 21.3 s/token | 150.5 s | ~2 GB f32 |
| after | **0.26 s/token** | 1.63 s/token | 27.6 s | ~1 GB fp16 plane |

Decode went **~56×** faster (14.6 → 0.26 s/token, 0.07 → 3.8 t/s) and prefill **~13×**
(21.3 → 1.63 s/token). Both numbers come from the same binary at two step counts — the decode rate
is the marginal cost between `--steps 8` and `--steps 32`, so it is not carrying a fixed prefill
term; the prefill rate is the residual after that marginal is removed. The step-8 wall time
includes the **one-time** cost of building the cube planes and decoding the embedding table, ~18.5 s
for this checkpoint, which is why 8 steps is 27.6 s and not 5 × 1.63 + 3 × 0.26.

Where the *remaining* time goes, from an env-gated per-stage clock run (wall times, so they carry
the profiler's own overhead — read them as shares, not as the totals above):

| stage | before | after |
|---|---|---|
| weight transpose + fp16 convert | 31.5 s | 0 (cached) |
| embedding-table decode | 53.8 s | 0 (cached) † |
| cube | 0.69 s | 0.69 s |
| rms_norm / silu / rope / kv / softmax | ~1.0 s | ~1.0 s |

† the embedding table was the largest cost *after* the weight plane was cached, and it is a
per-token cost until it is cached in turn — the two together are the change. The activation-side
transfers the ops still do (each op brings f32 in, converts to fp16, drives the op, widens back) are
what is left; that is the next thing to attack, not something this change touched.

This is a correctness path that is now also ~fast enough to use (3.8 t/s decode), but it is still
the fp16 cube on ~1 GB of resident plane rather than the device's own quantized ops. Feeding
`MatmulW8a8I32Custom` / `MatmulW4a16Custom` the *packed* weights would remove both the ~1 GB and the
per-op f32↔fp16 round trip, and is the larger remaining win — it is not this change.
