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
to the CPU backend**. This is the board's **regression gate** and it still passes; the 1.7B section
below is a second size and where it does not.

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
large-N edge, and the second checkpoint below is a third instance — but the *coverage* gap is closed:
there is no op the graph calls that the backend refuses. The paths the whole forward still lacks are
the ones the graph never calls: `gemm` bias, attention with a sliding window (`first_key != 0`), and
`topk_sample`/`logits_temperature`.

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
what is left; that is the next thing to attack, not something this change touched. **That "next
thing" has since been measured and is not worth attacking**: at 4B the convert and transfer are
~9% of a decode step, and the step is ~86% the NPU waiting on the cube itself — see
[Where a 4B decode step actually goes](#where-a-4b-decode-step-actually-goes-and-why-nothing-was-changed).

This is a correctness path that is now also ~fast enough to use (3.8 t/s decode), but it is still
the fp16 cube on ~1 GB of resident plane rather than the device's own quantized ops. Feeding
`MatmulW8a8I32Custom` / `MatmulW4a16Custom` the *packed* weights would remove both the ~1 GB and the
per-op f32↔fp16 round trip, and is the larger remaining win — it is not this change.

## A second checkpoint: Qwen3-1.7B, and where it does not match

The board runs a second Qwen3 size, and it is a different shape family rather than a bigger copy of
the first. `Qwen3-1.7B-Q4_K_M` is a `q4_k_m` file like the 0.6B's — the same quantizer mix, so the
same faithful q4_K/q6_K decode path, over ~2.8× the weights — but its hidden width is **2048**, not
1024; its `ffn_down` is 6144 wide; and its output projection is a distinct q4_k tensor where the 0.6B
ties `token_embd`. None of that was known to work: the backend's shape guards (`rms_norm` ≤ 4096 and
a multiple of 16, the cube's 8192-column chunk, `n % 128`) had only ever been exercised at the 0.6B
family.

**It runs, and it fits.** The 1.7B fp16 plane is ~3.4 GB against the 0.6B's ~1 GB, on a board whose
`MemTotal` is 23.7 GiB (`MemAvailable` 21.4 GiB at rest) — the process peak RSS is **3.6 GiB**
(`VmHWM` 3776924 kB) and no swap is touched. Resident weights scale linearly with the checkpoint, as
they should: the plane is a pure function of the weights and is built once.

| checkpoint | decode, marginal | `--steps 8` wall | `--steps 32` wall | peak RSS |
|---|---|---|---|---|
| Qwen3-0.6B | 0.26 s/token | 27.6 s | — | ~1 GB plane |
| Qwen3-1.7B | **0.39 s/token** | 92.0 s | 101.4 s | **3.6 GiB** |

Decode is the marginal between `--steps 8` and `--steps 32`: (101.38 − 92.03) / 24 = **0.39 s/token**,
1.5× the 0.6B's — the ratio the 2.8× weight count over the same 28 layers predicts. The ~92 s wall at
either step count is mostly a one-time cost, not decode: at `--steps 1` the floor is **~91.5 s, flat
from 11 prompt tokens up** (92.19 s at 11 tokens), which is the plane build — the 1.7B plane is ~3.4×
the 0.6B's and takes ~4.7× as long to build. Prefill is *not* that fixed cost, and it is
**superlinear**: `--steps 1` costs 98.26 s at 101 prompt tokens (+0.067 s/token over the floor),
113.63 s at 201, and **178.15 s at 401** — the marginal rate rises from ~0.07 to ~0.32 s/token, because
prefill is the model's loop of one `AttentionStepCustom` per position and every position attends to
its whole prefix.

**The identity gate is against the board's own CPU backend, and it does not hold for this checkpoint
on the canonical prompt — but it fails the way the page already predicts it can, not in a new way.**
The two sequences:

```
--prompt "The capital of France is" --steps 8
  cpu          -> [12095 Paris 13. 576 The 6722 capital 315 of 17689 Spain 374 is 24081 Madrid]
  ascend 1.7B  -> [12095 Paris 13. 576 The 6722 capital 315 of  279 the   3639 United 4180 States]
```

They agree for **five** generated tokens and first differ at **generated index 5**: the CPU picks
`17689` ("Spain"), the board picks `279` ("the"). The two sides' logits at that position:

| rank | CPU (f32) | | ascend (fp16 cube) | |
|---|---|---|---|---|
| 1 | 17689 "Spain" | 22.0520 | **279 "the"** | 22.0469 |
| 2 | **279 "the"** | 22.0486 | 17689 "Spain" | 22.0469 |
| 3–8 | 9856, 15344, 6323, 15948, 32961, 6864 | 21.87 … 20.55 | the same six ids, same order | within 0.013 |

**It is the same two tokens swapped, on a margin of 0.0034.** Ranks 3–8 are identical ids in
identical order, so nothing is structurally wrong — this is one near-tie the two backends resolve
differently, and past it every token is a different question because the two are then reading
different text. 0.0034 on a ~22-magnitude logit is **1.5e-4 relative**, below the **~2.7e-4** the cube
is itself good to (the fp16-operand, fp32-accumulate figure this page measured for the cube and, in
the W4A16 section above, for the same arithmetic). A backend 2.7e-4 off cannot be *expected* to break
a 1.5e-4 tie the way f32 does, so the flip sits inside the backend's documented accuracy rather than
outside it.

**Two controls say the same thing.** A second prompt — `"Name three colors of the rainbow"`,
`--steps 8` — is **token-identical** on both backends
(`[13. 576 The 47613 rainbow 374 is 1865 made 705 up 315 of 8094 seven]`); and the 0.6B regression
gate above still passes. The divergence is therefore not "1.7B is broken" and not "the ascend backend
is broken" — it is this one prompt landing a near-tie under the fp16 floor. (The 0.6B is not immune
in principle; it simply has fewer near-ties to land, and its canonical prompt does not land one in 32
steps — which the 0.6B was checked at too, identical on both backends.)

So the honest statement of coverage is: **one Qwen3 size is token-identical on the board and a second
runs, fits, and matches everywhere the two backends are not asked to break a tie smaller than the
fp16 plane's own error.** It is not a coverage gap — every op the graph calls runs at hidden 2048, so
no shape guard is in the way — and it is not a new op. It is the precision of the fp16 plane, showing
up as a token. The path to "identical at 1.7B" is the same as everywhere else on this page: give the
graph more precision (the packed-weights path the W4A16 section closed, or an fp32 accumulation
upgrade), not a new op.

## A third checkpoint: Qwen3-4B, and the memory the RSS does not count

`Qwen3-4B-Q4_K_M` is a third size family: `q4_k_m` like the other two, on **36** layers with hidden
**2560** and `ffn` **9728**, 32 attention heads over 8 KV heads. It **ties** `token_embd` (there is
no distinct `output.weight`, as on the 0.6B and unlike the 1.7B), so its output projection is the
same `q6_k` `2560×151936` table the embedding reads. The file is 2.33 GiB.

**It runs, and the identity gate holds.** Unlike the 1.7B, the canonical prompt does not land a
near-tie under the fp16 floor: `--prompt "The capital of France is" --steps 8` is **token-identical**
to the board's own CPU backend.

```
--prompt "The capital of France is" --steps 8
  cpu     -> [12095 Paris 13. 576 The 6722 capital 315 of 9856 Germany 374 is 19846 Berlin]
  ascend  -> [12095 Paris 13. 576 The 6722 capital 315 of 9856 Germany 374 is 19846 Berlin]
```

At `--steps 32` it stays coherent — the continuation walks France → Italy → Spain → Portugal →
Lisbon, correct capitals throughout — which is the same evidence the 0.6B gate carries: the tokens
are not just equal, they are *right*.

**No shape guard trips.** Hidden 2560 is ≤ 4096 and a multiple of 16 (the `rms_norm` guard); every
GEMM width is a multiple of 128 (2560, 9728, 151936 — the `kGemmAlign` guard); the cube's 8192-column
chunk divides them all. The 4B family fits inside the same guards the 0.6B was written against.

**The memory bound — and the number `VmHWM` alone gets wrong.** The board's `MemTotal` is 23.72 GiB
with `MemAvailable` 21.45 GiB at rest, so headroom was never the question for 4B and the process peak
host RSS is **5.12 GiB** (`VmHWM` 5364240 kB, exact via `getrusage`). But that is only *half* the
footprint, and this is the correction the 1.7B row above needed too: **the process RSS does not
contain the weights.** The checkpoint is `mmap`'d (`MAP_PRIVATE`, `src/gguf/reader.cpp`), so its
~2.3 GiB of touched pages sit in RSS as reclaimable file pages; the **cube planes and the embedding
table are device allocations** (`aclrtMalloc`) in the NPU's own memory pool, which is a separate
carve-out of the same LPDDR and never appears in `/proc/<pid>/status`. The two halves:

| what | where | 4B |
|---|---|---|
| checkpoint pages (mmap) | host RSS, file-backed | ~2.3 GiB |
| layer cube planes, fp16 (`cube_for`, non-embedding weights) | device pool | **6.06 GiB** |
| head plane, fp16 (`cube_for`, `token_embd` driven as the GEMM head) | device pool | **0.72 GiB** |
| embedding table, f32 (`dense_table_for`: `vocab × d × 4`) | device pool | **1.45 GiB** |
| host transient (the f32 `dense` decode of the embedding table) | host RSS | ~1.5 GiB peak |
| **host peak RSS** | | **5.12 GiB** |
| **device residency** | | **~8.2 GiB** |

The 6.06 GiB is not an estimate: it is the checkpoint's non-embedding quantized element count
(3,255,828,480) at two bytes. The other two rows are the *same* `token_embd` tensor held twice — once
as a fp16 GEMM plane for the tied head, once as the f32 gather table — so a tied checkpoint pays for
its vocabulary matrix both ways; the 1.45 GiB is `2560 × 151936 × 4`, twice the fp16 plane of the same
tensor, because the gather table is cached f32. A checkpoint with a **distinct** `output.weight` pays
the head plane on *that* tensor instead: the 1.7B's device residence is 2.41 GiB of layer plane plus a
0.58 GiB head plane on `output.weight` plus a 1.16 GiB f32 table on `token_embd`, **~4.2 GiB** — which
its "3.6 GiB peak RSS" row above does **not** count, because the RSS is not where the weights live.
Read the 1.7B row's "peak RSS" as host RSS and this table as what the device is actually holding.
These device figures are computed from the checkpoints' element counts, not sampled: `npu-smi`
returns nothing once the process holds the device, so they are the sizes the allocator is asked for.
`npu-smi info` reports the pool as **23.73 GB**, with **~7.0 GB in use at idle and no process
running** (the driver/firmware carve-out), leaving ~16.7 GB for the plane; the 4B's 8.2 GiB sits
comfortably inside it.

**Timings — this is a coverage result, not a speed one.** Decode is the marginal between `--steps 8`
and `--steps 32`, paired in one session: (234.88 − 220.80) / 24 = **0.587 s/token** (1.7 t/s), 1.5×
the 1.7B's 0.39 s/token and, as on that checkpoint, ~the weight ratio over the same session length.
The walls are dominated by the **one-time plane build**, not decode:

| checkpoint | plane, fp16 | build floor (`--steps 1`) | decode, marginal | `--steps 8` wall |
|---|---|---|---|---|
| Qwen3-0.6B | 0.71 GiB | ~18.5 s | 0.26 s/token | 27.6 s |
| Qwen3-1.7B | 2.41 GiB | ~91.5 s | 0.39 s/token | 92.0 s |
| Qwen3-4B | 6.06 GiB | **~212–225 s** | **0.587 s/token** | **~218–221 s** |

The 4B build floor is ~3.5 minutes, and at 8 or 32 steps the wall is essentially that floor — 220.80 s
at 8, 234.88 s at 32 — so **decoding is free next to building the plane**. The build scales with the
plane, not the layer count: 6.06 / 2.41 = 2.51× the 1.7B plane, 212 / 91.5 ≈ 2.3× the time. (The
`--steps 1` wall came in at 225.05 s, *above* the 8-step 220.80 s — a ~2% run-to-run spread on a
host build that is a host walk over every block, not a signal.)

**And the next size up runs too — this is where the board stops being comfortable.** An 8192-wide
`Qwen3-8B-Q4_K_M` needs **11.81 GiB** of layer plane, a **1.16 GiB** head plane on its distinct
`output.weight`, and a **2.32 GiB** f32 embedding table — **~15.3 GiB device** — against the ~16.7 GB
the pool leaves after the driver's 7 GB carve-out, and 4.7 GiB of mmap'd checkpoint on the host.
Measured, it **completes** and, on the same prompt, is **token-identical to CPU** at 8 steps:

```
--prompt "The capital of France is" --steps 8
  cpu     -> [12095 Paris 13. 576 The 6722 capital 315 of 15344 Italy 374 is 21718 Rome]
  ascend  -> [12095 Paris 13. 576 The 6722 capital 315 of 15344 Italy 374 is 21718 Rome]
```

But the host is at the edge, and the *spread itself* is the honest number: peak RSS was **13.23 GiB**
on one run and **8.28 GiB** on the next — the difference is how much the kernel paged out, not how
much the process wanted — and `MemAvailable` fell to **~0.6 GB** with **~2 GB of swap** touched. The
wall reflected it: **~500 s** on the first run and **~1197 s** on the second, and that 697 s gap over
the same 8 tokens is **swap, not decode** — which is why no clean 8B decode marginal is claimed here.
The 500 s clean wall is, as at 4B, almost entirely the one-time plane build. What the run establishes
is a **coverage** fact: *an 8B runs on one 310B*, which nothing in this tree's history claimed and
the S600's 2 GiB ION pool makes impossible. Above 8B the layer plane alone would exceed the pool, so
this is where the ladder ends at `q4_k_m` fp16 planes — the packed-weight path the
[W4A16 section](#precision-two-different-errors-and-which-one-is-the-graphs) closed is what would
move it, and it is not this change.

The honest statement of coverage is now four sizes: **0.6B, 4B and 8B token-identical, and 1.7B runs
and fits with one documented near-tie** — the same fp16-plane precision limit this whole page is
about, met at one prompt at 1.7B and not met at the other three, with no new op and no shape guard in
the way at any of them.

## Where a 4B decode step actually goes — and why nothing was changed

The paragraph above the caches names the candidate: "the activation-side transfers the ops still do
(each op brings f32 in, converts to fp16, drives the op, widens back) are what is left". A 4B decode
step is ~343 cube drives and ~253 `gemm_quant` calls — the 36 layers' seven GEMMs plus the head, each
split at the 8192-column `kCubeChunkN` — so *if* that guess were right the fix would be to thread fp16
through the graph, or to reuse the per-call `aclTensor` descriptors, or both. **Measured, the guess is
wrong, and the honest answer to "fix it" is no.** The splits, from `$POCKETLLM_ASCEND_PROFILE=1`
(a per-stage clock now in `backend.cpp`, printing at exit):

| stage | steps=1 | steps=8 | **per decode step** | share |
|---|---|---|---|---|
| **op_sync** — `aclrtSynchronizeStream`, the NPU's own work | 913.6 ms | 4083.0 ms | **452.8 ms** | **85.5%** |
| to_f16 — the f32→fp16 convert of an activation | 128.5 ms | 421.1 ms | 41.8 ms | 7.9% |
| sdma_down — the D2H read-back of each chunk | 696.1 ms | 776.0 ms | 11.4 ms | 2.2% |
| to_f32 — the fp16→f32 widen | 49.5 ms | 114.6 ms | 9.3 ms | 1.8% |
| op_enqueue — the host cost of the op call itself | 27.1 ms | 70.2 ms | 6.2 ms | 1.2% |
| sdma_up — H2D | 2687.8 ms | 2717.9 ms | 4.3 ms | 0.8% |
| aclTensor create+destroy, `GetWorkspaceSize`, ws alloc | 19.9 ms | 45.1 ms | ~3.6 ms | 0.7% |

The per-decode-step column is the two run totals differenced and divided by seven: `--steps 1` and
`--steps 8` share the whole one-time prologue (the plane build, the embedding-table decode, the
~0.38 s of `embedding_quant`) and the same 5-token prefill, so their difference **is** seven decode
steps. That differencing is what makes the split readable at all — the steps=1 column is 59% `sdma_up`
because the plane build is a host→device `aclrtMemcpy` of every weight, and the table decode is
`embedding_quant`'s 190 ms/call. Neither is a decode cost, and neither paginates a fix. The tracked
stages sum to **~0.53 s** of the ~0.59–0.63 s headline marginal; the ~0.07 s gap is the host work that
sits *between* the slots — chiefly `run_cube_weight`'s chunk-accumulate loop (343 × 8192 f32 adds per
step) and the graph's own inter-op walk — which no slot measures and this split does not claim.

**The activation-side work is 14.5% of the step — everything that is not `op_sync` — and the *transfers* are the smaller half of it.**
The convert is 41.8 ms and the widen 9.3 ms — CPU work on one token's `k`-element row, and the two
halves that a fp16-threaded graph would remove — while the `sdma_*` beside them is only 15.7 ms, and
its 4360-up/4122-down call counts are dominated by the *plane build's* copies, not decode. Threading
fp16 through the graph would take the step from **~530 ms to ~480 ms: a ~9% win**, for a graph-wide
dtype change that touches every op and every boundary. The `aclTensor` glue the task flagged as "the
sleeper" came in at **0.4%** — ~5.2 µs per create/destroy across 343 drives; it is noise, and
descriptor reuse would not pay for its own risk.

**What the 452.8 ms is.** 343 chunks × (m=1, n=8192, k=~2560) is 14.4 GFLOP/step, so the cube is
moving **~32 GFLOP/s** — a small fraction of what its fp16 units can do, and far under what the
earlier per-op measurement saw (that one ran a single large gemm, not 343 launches of a 24×8K tile).
The cost is therefore **per-op launch/scheduling overhead on the NPU**, not arithmetic: the host
enqueues in 6.2 ms and then waits 452.8 ms for work that is mostly fixed-cost. The lever is fewer,
larger cube drives — a batched or wider-N launch, or not re-launching per chunk — not the activation
dtype. That is a different change than the one the page's own text proposed, and it is the one a
profile actually points at; it is left for its own PR rather than bundled here.

> **This reading was wrong, and the correction is below.** The 14.4 GFLOP number double-counts: a
> decode step is ~8.0 GFLOP, and `op_sync` is the cube *executing* that, not launching it. The
> "1.3 ms/launch" that follows from 452.8 / 343 does not survive the chunk and sync sweeps in
> [the next section](#the-343-launches-are-not-the-cost-correcting-the-profile-above).

**No behavior changed, and the gate says so.** The profiler is env-gated and, off, adds not one clock
read: `StageClock` does not call `now_ms()` and `OpScope` holds a `const char *` rather than a
`std::string`, so a call site allocates nothing and the timed path is the binary it was before. The
4B decode — the `--steps 8`/`32` marginal — reads **0.587 s/token** before (this page's row above,
measured in the coverage change) and **0.634 s/token** after (walls 209.0 s and 224.2 s), and that
8% is this host's run-to-run spread rather than a cost of the profiler: the walls at *both* step
counts are **lower** after (220.8→209.0 and 234.9→224.2 s), so the two moved independently. There is
no mechanism for the profiling code to change a run with `POCKETLLM_ASCEND_PROFILE` unset, and the
measurement is reported as it landed rather than re-rolled to the prior number. All three identity
gates hold as documented above: **0.6B identical, 1.7B the one near-tie, 4B identical.** The
deliverable is the split above — the measured answer that the transfers are a ninth of the step, so
the change the page named is not worth making, and the profile names a different bottleneck instead.

## The 343 launches are not the cost — correcting the profile above

The section above ends on a reading that the next two experiments falsify. It said the 452.8 ms of
`op_sync` was **per-op launch overhead** — "~1.3 ms of fixed cost per cube drive, 343 times a token"
— and named "fewer, larger cube drives" as the lever. Both halves of that are wrong, and each is
wrong for a reason worth writing down: the arithmetic was double-counted, and the lever was never
tested against a control.

**First, the arithmetic.** The "14.4 GFLOP/step" above comes from charging *every* one of the 343
drives a full n = 8192, k = 2560 tile. But only the ffn and head chunks are that wide — the 144
hidden-matrix drives (`q`/`k`/`v`/`o`, n = 2560) and the 36 `ffn_down` drives are a *single* 2560-wide
chunk each. Summed at each drive's real shape a decode step is **~8.0 GFLOP** (36 layers × 202 MFLOP +
a 778 MFLOP head), so 452.8 ms is **~18 GFLOP/s**, not 32.

A cube doing ~18 GFLOP/s on an **m=1** activation is not obviously launch-bound: an m=1 GEMM is a
GEMV, it exercises one row of the cube per tile and leaves the M dimension of the array idle, so a
small fraction of the rated fp16 throughput is exactly what a decode-shaped matmul should produce.
"A small fraction of peak" was read as "overhead" when it is more likely the shape.

**The chunk sweep removes the launch count and nothing happens.** `kCubeChunkN` is now overridable
(`$POCKETLLM_ASCEND_CUBE_CHUNK_N`) so the width can be swept on the board, and the 4B identity gate
is the correctness oracle at each width — the head is n = 151936, so a 32768-wide sweep drives the
cube at N = 32768 where an 8192 sweep never does:

| chunk N | cube drives / token | gate | decode (`--steps 8`/`32` marginal) |
|---|---|---|---|
| 8192 (shipped) | **343** | ✅ identical | 0.516 s/token |
| 16384 | 262 | ✅ identical | 0.513 s/token |
| 32768 | 257 | ✅ identical | 0.513 s/token |
| 65536 | 255 | ❌ **garbage** | — |
| 151936 | 255 | ❌ **garbage** | — |

**Cutting the launch count by 25% (343 → 257) moved decode by 0.6% — noise.** If each drive carried
~1.3 ms of fixed cost, removing 86 of them would have saved ~112 ms/step, a fifth of the step. It
saved nothing. The launches are not the cost.

The sweep also **tightens the cube's N wall** this page has carried since the first backend: the op
was known to be right at n = 32768 and wrong at n = 151936, and 65536 now fails too, with the same
degenerate output (`[119332呻 55101edu 92695 negativity …]` — one token repeated, the signature of a
silently mis-tiled result). The boundary is therefore **32768 < N ≤ 65536**, and `kCubeChunkN = 8192`
is the conservative side of it, as its comment claims.

**Second, the sync A/B.** The chunk loop syncs *after every chunk*, which on its face is a host-side
cost the compute could hide: the host enqueues, waits, reads back, and only then enqueues the next
chunk. `run_cube_weight` can instead enqueue a GEMM's chunks before one sync — each writes its own
output buffer, so nothing forces the interleave — selected by `$POCKETLLM_ASCEND_CUBE_SYNC=one`
(the shipped default is the per-chunk sync, so the binary is unchanged; the override is for
measurement). A/B, same binary, chunk 8192:

| | `op_sync` calls | `op_sync` time (steps=8) | gate |
|---|---|---|---|
| sync per chunk (shipped) | 3087 | 4070.1 ms | identical |
| sync once per GEMM | **2277** (−26%) | **4023.3 ms** (−1.2%) | identical |

**26% fewer syncs, 1.2% less time.** Removing the per-chunk sync removes 26% of the `op_sync`
*calls* and none of its *time*, which is the definition of the time not being the sync. `op_sync` is
the cube executing; the sync is where the host waits for it. (The end-to-end walls say the same and
no more: `--steps 8` was 215.012 s and 215.022 s for the two, identical to the millisecond, and the
`--steps 32` marginal read 0.641 vs 0.770 s/t — a spread this host produced on the *same* binary
[earlier](#where-a-4b-decode-step-actually-goes-and-why-nothing-was-changed), so the honest reading
is the profile's `op_sync` time, which is flat, not the marginal, which is noise.)

**So the corrected reading.** The 452.8 ms per decode step is **the cube doing the GEMMs** — ~8.0
GFLOP at ~18 GFLOP/s on an m=1 shape — not 343 × 1.3 ms of per-launch host overhead. Neither lever
touches it: not the launch count (the chunk sweep), not the sync placement (the A/B). The levers that
*can* reach device GEMV throughput are a different class and are not host-side at all — feeding the
cube a wider m so the array is not half-idle (a batched decode), or the device's own quantized matmul
(`MatmulW8a8I32Custom`/`MatmulW4a16Custom`, which the [W4A16
section](#precision-two-different-errors-and-which-one-is-the-graphs) closed on accuracy grounds) —
and the honest statement is that **this is where a host-side launch-and-transfer optimization path
ends**, because the measured cost was never on the host side of it.

**No behavior changed.** Both knobs default to the shipped values (`kCubeChunkN = 8192`, the
per-chunk sync), so the delivered binary is byte-for-byte the timed path it was; the 4B gate holds at
`[12095 Paris 13. 576 The 6722 capital 315 of 9856 Germany 374 is 19846 Berlin]` at every chunk width
above the wall, and 0.6B/1.7B are unchanged as documented.

## The m=1 rate is the shape, not the device — so batching is the lever

The correction above ended by saying the ~18 GFLOP/s was "more likely the shape" of an m=1 GEMV than
overhead. That was a hypothesis; this is the test. `pocketllm-mscale` drives the same public
`gemm_quant` the engine uses, on synthetic q4_K weights and a **fixed** (n, k), varying only m:

| m | n = k = 2560 | n = 9728 | n = 151936 |
|---|---|---|---|
| 1 | **19.0 GFLOP/s** | 15.3 | 15.4 |
| 2 | 36.1 | | |
| 4 | 65.2 | | |
| 8 | 107.9 | 101.9 | 107.0 |
| 16 | 169.4 | | |
| 32 | **224.2** | 228.0 | 245.7 |

**Throughput is not flat in m — it rises ~12× from m=1 to m=32 and then saturates near 225–245
GFLOP/s.** So the 18 GFLOP/s is the **shape**, exactly as the correction guessed: an m=1 GEMM is a
GEMV that leaves the cube's M dimension idle, and the rate climbs steeply once m ≥ 8 fills it. The
m=1 row is the cross-check that ties this to the decode profile — 19 GFLOP/s here is the ~18 GFLOP/s
the decode step runs at, from a completely different harness, which is what makes the two numbers the
same fact.

It is also **shape-saturated, not `n`-limited**: the same m gives the same rate at n = 2560, 9728 and
151936, so the chunking the last two sections argued about does not touch throughput either (it never
did — that was already the chunk sweep's conclusion).

**What batching would require — and how much of the gap it is.** The engine walks **one token at a
time on the decode path**: `Qwen3Model::forward` calls `matmul(..., n, ...)` with `n` = the tokens in
the call, so a **prefill already drives m = the prompt length** (it is batched), and only the
per-token decode is m=1 — the head is even called at literal `m = 1`. Every decode step is therefore
36 layers plus a head of m=1 GEMVs at ~19 GFLOP/s, when the same weights do ~108 GFLOP/s at m=8.
Restructuring decode to process a batch of tokens would need **the serving layer to batch requests**:
the runtime is `supports_batch = False` — one request at a time behind a lock, one KV cache, one
position — so there is nothing on this board to fill m with today. That is the whole gap, and it is
why "batch a few requests" is the real 310B decode lever: it would take the GEMM half of a decode
step toward the m=8 rate, though the attention and memory traffic that share the step (the ~12% of
the profile that is not `op_sync`) bound the end-to-end win well below the pure-GEMM 5.7×.

Greedy decode cannot batch within one sequence, so the honest framing is: **the 310B's decode GEMM
rate is a shape limit that batch serving could lift, not a device limit that caps the board.** The
device itself does ~230 GFLOP/s fp16 at m ≥ 16. Nothing on this board is at that width today.

**No behavior changed.** The harness is a new tool (`src/tools/mscale.cpp`, `pocketllm-mscale`) that
does not touch the backend, so no kernel or graph code moved and no identity gate was at risk; the
0.6B/1.7B/4B gates are as documented above.
