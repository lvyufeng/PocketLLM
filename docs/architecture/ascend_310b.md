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

## End-to-end accuracy: the 310B sits at the quantizer's own floor

The precision section above is **op-level**: it measures the cube's arithmetic and the W4A16
requantization against each other, not what a token stream does. This section is the
token-level counterpart. It answers the project's accuracy question — agreement with the
**2080ti C-engine oracle** — end to end, using the same method and the same 50-prompt corpus
the S600 accuracy work established ([s600_qwen3.md](../models/s600_qwen3.md#accuracy-against-the-2080ti-oracle)).

**The method, and one structural advantage.** Each prompt's ChatML-templated ids are fed to
the board's engine and to the oracle; both decode greedily and the two id sequences are
diffed. The 310B is *the C engine*, so it has a **native token-id surface**
(`Engine.encode` / `Engine.next_token`) — the same API the oracle drives — so both sides see
identical ids with **no text round-trip**. That is cleaner than the S600 path, whose delegate
is text-in / text-out and must reconstruct the ChatML text and re-tokenize it. Two reports
are read off each run: the **shared prefix** before the first divergence (bigger = the two
engines agree longer), and the **oracle's top-1-minus-top-2 margin** at the split (how
confident the oracle was where they part, in logits).

"At the floor" is the yardstick the [S600 page](../models/s600_qwen3.md#the-f16-oracle-the-gap-is-real-against-truth-and-a-floor-separates-w8-from-w4)
established: score the same capture against a near-lossless **f16** checkpoint (truth) and
against the **q4_k_m** checkpoint the board runs (the floor). A build whose agreement with
truth equals the q4_k_m checkpoint's own agreement with truth is measuring the *quantizer's*
error, not the board's; a build below that floor is a real, board-owned deficit.

Fifty prompts, four sizes, three axes (columns: shared-prefix min / p25 / median / p75 / max,
summed prefix Σ, and splits with a ≥5-logit oracle margin):

| size | axis | pref min/p25/**med**/p75/max | Σ prefix | splits ≥5 |
|---|---|---|---|---|
| 0.6B | 310B vs `q4_k_m` | 2 / 8 / **16** / 24 / 67 | 887 | 0 / 50 |
| 0.6B | 310B vs `f16` | 2 / 11 / **19** / 25 / 63 | 1078 | 0 / 50 |
| 0.6B | **floor** (`q4_k_m` vs `f16`) | 4 / 8 / **20** / 27 / 56 | 995 | 0 / 50 |
| 1.7B | 310B vs `q4_k_m` | 2 / 16 / **33** / 49 / 128 | 1908 | 3 / 46 |
| 1.7B | 310B vs `f16` | 2 / 8 / **17** / 27 / 79 | 939 | **5** / 49 |
| 1.7B | **floor** (`q4_k_m` vs `f16`) | 2 / 9 / **19** / 26 / 79 | 956 | 3 / 49 |
| 4B | 310B vs `q4_k_m` | 13 / 35 / **71** / 102 / 128 | 3495 | 5 / 49 |
| 4B | 310B vs `f16` | 4 / 17 / **25** / 47 / 112 | 1620 | 2 / 50 |
| 4B | **floor** (`q4_k_m` vs `f16`) | 4 / 17 / **25** / 47 / 96 | 1552 | 2 / 50 |
| 8B | 310B vs `q4_k_m` | 5 / 31 / **61** / 91 / 128 | 3158 | 9 / 48 |
| 8B | 310B vs `f16` | 4 / 17 / **32** / 49 / 100 | 1832 | 1 / 50 |
| 8B | **floor** (`q4_k_m` vs `f16`) | 4 / 19 / **34** / 53 / 119 | 1901 | 0 / 50 |

**The 310B sits at the floor at every size — at 8B, the furthest under it.** Against
the f16 truth the board reaches **Σ 1078 / 939 / 1620 / 1832** (0.6B / 1.7B / 4B / 8B) where the
`q4_k_m` checkpoint's own distance from truth is the floor **Σ 995 / 956 / 1552 / 1901** — ratios
**1.08 / 0.98 / 1.04 / 0.96**. Through 4B the board's agreement with a near-lossless reference is
**as good as the quantized checkpoint's own agreement with it** — within run-to-run noise of each
other, and at 4B fractionally better. The reading through 4B is direct: **the board is not adding a
token-agreement deficit of its own** — what disagreement there is, is the `q4_k_m` quantization's,
which the board inherits by running the same weights. The 0.6B floor Σ 995 is the S600 page's own
measured 0.6B floor reproduced **exactly**, which cross-checks the corpus and the oracle across two
boards.

**The 8B is the size where the board's Σ lands furthest under the floor's — 69 tokens over the 50
prompts (−3.6%, Σ 1832 vs 1901; the 1.7B's own low-side miss was −1.8%) — and the row-level
structure says what that deficit is.** On **35 of the 50 prompts the board and the checkpoint
part from truth at the *same* token with the *same* oracle margin**: the board's own arithmetic
contributes nothing there. On 7 prompts the board holds on longer than the checkpoint (+136
tokens together) and on 8 it parts earlier (−205), which is the whole −69. Six of those eight
early splits are near-ties — board-side oracle margins **0.02, 0.16, 0.16, 0.53, 0.92, 1.66** —
the class of prompt the fp16 plane's arithmetic can tip either way;
`instr_03` parts at 3.7; and one, `code_03`, parts at 49 against the floor's 119 with a
**7.8-logit** margin, the run's only wide-margin disagreement (the floor has none). Where the
board does part it is typically *less* sure than the floor is at its own split — gap median
**0.36** against the floor's **0.45**. So the 8B deficit is a scattered increment of board
arithmetic, mostly near-ties plus one confident row, not a systematic drift — and it is what a
−3.6% Σ at this size is made of.

**This is the token-level confirmation of the section above.** The precision section measured a
**~1.2e-1** error for the W4A16 op — the int4 requantization of the checkpoint's q4_K weights —
and concluded the graph's path stays the dense cube on the faithfully-decoded plane, not W4A16.
The end-to-end result is that conclusion restated at the token level: W4A16 is env-gated off
(`$POCKETLLM_ASCEND_W4A16`, `backend.cpp:762`), the shipped path decodes `q4_k`/`q6_k` with the
tree's own dequant and drives `MatmulCubeCustom` (~2.7e-4), and the board lands **at the floor** —
the 8B's −3.6% is the largest shortfall in the table and it is arithmetic scatter, above. Had the
~1e-1 requantization been the path, the suite would sit far under the floor — it does not. So the
token agreement is the end-to-end evidence that **the faithful decode + cube is what actually
executes.**

**The one place the 1.7B is *not* merely at the floor is its confident disagreements.** Its
310B-vs-f16 row carries **5** splits with a ≥5-logit margin against the floor's **3** — the
board parts from truth with more conviction, on a handful of prompts, than the `q4_k_m`
checkpoint itself does. `n` is small (five of fifty) so this is a note, not a finding, but it is
the row to watch if a wider corpus is ever run.

**The cross-board contrast is a quantizer-provenance effect, not a ranking of the boards.**
Against the `q4_k_m` oracle at 1.7B the board reaches **Σ 1908 (median 33)**, where the S600's
1.7B `w4` graph reached **Σ 636 (median 11)** with nine wide-margin splits. The reason is the
weights: the 310B decodes **the same `q4_k_m` GGUF** the oracle runs, so the two agree by
construction on most tokens; the S600's `.hbm` is a **different quantizer** (`leap_llm`
symmetric per-channel int4, not `q4_k_m`'s super-block k-quant), so it agrees far less. This
says nothing about which board is "better" — it says the 310B row is a *same-quantizer*
comparison and the S600 `w4` row is a *cross-quantizer* one.

**Scope, stated rather than implied.** Two limitations bound what this table means:

- **No logits surface on the board path.** The 310B exposes ids, not a distribution, so the
  project's "aligned to 1e-3" is expressible **only as token agreement** — the same limitation
  the S600 page documents. This measures how long two engines agree, not a logit distance.
- **The scheme confound.** `310B vs q4_k_m` is **same-quantizer** (hence the high Σ); the
  `vs f16` rows and the floor are **cross-quantizer**. The floor is what makes the second
  readable, and it is why the "at the floor" verdict — not the raw Σ — is the result.

**The 8B was omitted for a memory reason, and the release removed it.** The row was left out
while the board sat at its memory edge at 8B (swap, stalls), which would have confounded token
agreement with board instability. With the #676 packed-bytes release — the default since, and the
build this capture ran (the kill switch `$POCKETLLM_ASCEND_KEEP_PACKED` unset) — the 4.68 GiB of
packed `q4_k`/`q6_k` bytes are freed once the 16.42 GiB of planes and table are built at load.
The capture then ran to **50 of 50 rows** (47 capped at the 128-token ceiling) in ~4 h 10 min:
`MemAvailable` bottoms at **4,145–4,153 MiB** in the load's shadow (minutes 8–14; the pre-release
probe's dip was ~0.58 GB), recovers to ~4,614 MiB, and holds a **4,573–4,638 MiB** band for the
remaining four hours; `SwapFree` gives up **539 MiB (0.53 GiB)** — one step at t+15 min, the
trough at t+1.5 h — and is flat after, back at its pre-run level when the process exits: no
stall, no OOM, no thrash. The 8B accuracy cell is measurable on this board precisely because the
packed bytes are released.

The corpus, the two capture/score scripts and every artifact are recorded on the harness page's
convention: the corpus is byte-identical to the S600 one (md5 `0feef55b9b879a6ec4529317a3162312`),
and the captures and their `q4_k_m`/`f16`/floor scorings live beside the S600 artifacts on the
oracle host.

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
`npu-smi info` reports the pool as **23673 MB**, with **5814 MB (5.8 GiB) in use at idle and no
process running** (the driver/firmware carve-out), leaving **~17.4 GiB** for a model; the 4B's 8.2 GiB
sits comfortably inside it. (The 8B's planes and table are 16.42 GiB and its measured device peak is
21.7 GiB — see [the 8B subsection](#why-the-8b-is-at-the-edge-the-fp16-plane-not-the-24-gib) — which
is why the 8B, and not the 4B, is the size the pool cannot hold.)

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
`Qwen3-8B-Q4_K_M` needs a **12.94 GiB** layer plane, a **1.16 GiB** head plane on its distinct
`output.weight`, and a **2.32 GiB** f32 embedding table — **16.4 GiB of planes and table**, and the
**~21.7 GiB** measured device peak once the resident packed weights are counted (both derived below) — against the
~17.4 GiB the pool leaves after the ~5.8 GiB the driver holds at idle, and 4.7 GiB of mmap'd
checkpoint on the host.
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

### Where the 8B's host memory actually goes — and the one piece of it that was ours to give back

That "~0.6 GB `MemAvailable` with swap touched" paragraph invited an investigation that assumed the
4.7 GiB mmap'd checkpoint was resident and could be released. **Half of that was right and half was
wrong, and the wrong half is why the peak did not move.**

Measured on this board (23.7 GiB `MemTotal`, 22.9 GiB available at rest), the host memory during an
8B run is **not** dominated by the mapping or by the process's own heap. The readings below are
`/proc/<pid>` interiors taken *at the moment `MemAvailable` is lowest*, so this is the peak, not a
quiet point in it:

| what | how counted | at peak |
|---|---|---|
| `libpocketllm` stack + the Ascend runtime's own buffers, as one anonymous `aclrtMalloc` region | `[anon]` Rss in `smaps` | **~2.9 GiB** |
| the storage/driver hugepage pool (`/sp_group_100001_nohuge`) | `sp_group` Rss + Swap in `smaps` | **~4.1 GiB** (plus ~2 GiB swapped) |
| the checkpoint, `mmap`'d `MAP_PRIVATE` | the `gguf` mapping's `Rss` | **~4 kB** — fully evicted |
| the process heap | `[heap]` Rss | **~33 MiB** |
| the process's stated `VmRSS` | `/proc/<pid>/status` | **~7.6 GiB** |
| `MemAvailable` | `/proc/meminfo` | **~0.58 GB** |

Three things fall out of this and each one closed a door:

- **The checkpoint mapping is not the problem at the peak.** `/proc/<pid>/smaps` reports a `Rss` of
  **0** for the 4.7 GiB `gguf` mapping: the kernel had already evicted every unreferenced page of it,
  because `MAP_PRIVATE` file pages are clean and first to go under pressure. Releasing the mapping
  did **not** lower the peak and could not — the peak was already being reached without it.
- **The process's own host memory is negligible.** ~33 MiB of heap and a handful of MiB of library
  text. There is no large `std::vector` in the ascend path: every staging buffer (`cube_for`,
  `dense_table_for`, `read_f32`, `to_f16`) is a local that dies before the next tensor, so none
  accumulates. The "which host buffer can be shrunk" half of the question has no answer because there
  is no such buffer.
- **What `MemAvailable` sees is the device pool.** The 310B has no separate device LPDDR — the NPU
  carves its allocations out of the same 23.7 GiB the host uses, which is why `MemAvailable` ≈
  23.7 − (driver reserve) − (device allocations) − (host Rss). The device side, not the host side,
  is what the number is reporting. The memtrace confirms the device peak: `aclrtMalloc` reaches
  **21.7 GiB live**, of which 14.1 GiB is the layer+head cube planes, 2.3 GiB the f32 embedding
  table, and the rest is the **resident raw packed GGUF weights** (~4.68 GiB, held for the
  model's life), the KV cache at a 256-row initial capacity, and the context-length cos/sin
  tables. That is 92% of `MemTotal` before a single byte of swap is used, and the swap
  is the direct consequence.

**What was changed anyway, because it is correct in its own right.** `GgufReader` now exposes
`release_mapping()`, and both engine entry points (`Qwen3Model::load` in `run.cpp` and
`Session::open` in `session.cpp`) call it once every tensor has been bound into device memory — the
tokenizer has already copied its vocabulary and merges, and nothing reads the file again. The
directory (names, shapes, offsets, metadata) is parsed into the reader's own storage, so `size()`,
`tensors()` and `tensor()` keep working; only `tensor_data()` needs the bytes, and it now throws a
named error rather than returning a dangling pointer if it is called after the release. This frees
the mapping the instant the build ends instead of holding it for the session's life.

**What the change does and does not do, measured.** One clean A/B on the same binary, same prompt,
same 8 steps:

| | peak host RSS | peak swap | wall |
|---|---|---|---|
| before (mapping held for the session) | 8.68 GiB | ~3.47 GiB | 1139 s |
| after (mapping released after the build) | 7.87 GiB | ~3.47 GiB | 1189 s |

The peak RSS is ~0.8 GiB lower; the peak swap and the wall are unchanged. The wall difference is
within the run-to-run spread this board has shown all along (~1100–1200 s for the same 8 tokens).
**So the honest outcome is that the 8B is still swap-bound and its decode is still not a clean
marginal — the mapping release is a real, small reduction in resident set, not a fix for the
thrashing.** The 8B does not fit in the 310B's single shared pool at `q4_k_m` fp16 planes, and
nothing that touches only the *host* side can make it: the thing that does not fit is 21.7 GiB of
device allocation against a 23.7 GiB pool with a ~7 GiB driver carve-out. The ladder's exit for a
checkpoint this size is the packed-weight path ([W4A16](#precision-two-different-errors-and-which-one-is-the-graphs)),
which would cut the 14.1 GiB of fp16 planes to the ~4.4 GiB of packed weights — a change to what the
*cube* consumes, not to what the host holds, and not this change. **The fit moved anyway, by
neither of those levers:** the 4.68 GiB of packed bytes are only needed until their decoded form
exists, and releasing them at load takes the resident set from 21.10 GiB to the 16.42 GiB of
planes and table — measured, the same 8B now loads in 507.7 s and decodes at 1.73 s/token with
the swapfile untouched **in that probe run** (the release section below has the numbers); the
four-hour accuracy capture above is the longer-run reading of the same release-on configuration,
and there `SwapFree` gives up **539 MiB (0.53 GiB)** once and holds it flat
([the accuracy section](#end-to-end-accuracy-the-310b-sits-at-the-quantizers-own-floor)).

The honest statement of coverage is now four sizes: **0.6B, 4B and 8B token-identical, and 1.7B runs
and fits with one documented near-tie** — the same fp16-plane precision limit this whole page is
about, met at one prompt at 1.7B and not met at the other three, with no new op and no shape guard in
the way at any of them.

### Why the 8B is at the edge: the fp16 plane, not the 24 GiB

The section above answers "where did the memory go" from the host side. The question it invites —
*why is an 8B model at the edge on a board with ~24 GiB of unified memory?* — has a different answer,
and it is the pool's size and the cube's operand width, not the 8B's own footprint in bytes.

**The pool is ~17 GiB, not 24.** The 310B has no separate device LPDDR: the NPU allocates out of the
same memory the host uses (`enable_ascend_share_pool` in `/proc/cmdline`), and the driver/firmware
carve-out is charged *before* any model is loaded. Read at idle, with no model bound, on this board:

| reading | value | source |
|---|---|---|
| board total (host-visible) | 23.1 GiB (`24241728 kB`) | `/proc/meminfo` `MemTotal` |
| `npu-smi` total | 23673 MB | `npu-smi info` |
| **`npu-smi` charged at idle** | **5814 MB** | `npu-smi info`, no process running |
| **pool left for a model** | **~17.4 GiB** (23673 − 5814) | the two above |

So the number to reason about is ~17 GiB, and a comfortable 8B has to fit inside it — the ~5.7 GiB
idle charge is gone before the first weight is read.

**The 8B's device footprint is the measured 21.7 GiB, and two thirds of it is the fp16 *expansion* of
a 5.0 GB packed file.** Parsed from the checkpoint's own tensor table (`Qwen3-8B-Q4_K_M.gguf`,
5,027,784,512 B, 36 layers, hidden 4096, a *distinct* `output.weight` — Qwen3-8B does not tie the
head):

| what | packed on disk | on device | factor |
|---|---|---|---|
| 36 layers' quantized weights (`q4_k`/`q6_k`) | 4.160 GB | **13.892 GB (12.94 GiB)** fp16 | **3.34×** |
| `output.weight` head (`q6_k`, not tied) | 0.511 GB | 1.245 GB (1.16 GiB) fp16 | 2.44× |
| — *planes alone* | *4.671 GB* | ***14.10 GiB*** *(the page's measured "14.1")* | *3.02×* |
| `token_embd.weight` → f32 embedding table | 0.350 GB | 2.489 GB **(2.32 GiB)** f32 | 7.11× |
| *planes + table* | *5.021 GB* | ***16.42 GiB*** | *3.51×* |
| **resident raw packed weights** (the on-disk `q4_k`/`q6_k` bytes, held for the model's life) | 5.021 GB | **4.68 GiB** (the same bytes, verbatim) | 1.00× |
| *planes + table + packed* | — | ***21.10 GiB*** | — |
| **measured device peak** (`aclrtMalloc`, the section above) | — | **21.7 GiB** | — |

**The two numbers on this page are the same number, split — and the missing term is the packed
weights, not an arena.** Planes + table is **16.42 GiB**, and the paragraph above records the
**21.7 GiB** measured `aclrtMalloc` peak. The ascend backend reserves **no arena and no persistent
workspace** — grep `src/kernel/ascend/backend.cpp` and the only scratch is the per-op
`aclnn…GetWorkspaceSize` temporary, `aclrtMalloc`'d and freed inside the single call. The ~4.68 GiB
between the two figures is the **raw packed GGUF bytes, resident for the model's whole life**:
`bind_packed` (`src/model/qwen3.cpp`) does exactly `backend_->allocate(nbytes)` followed by
`copy_to_device` of the *compressed* tensor, and stores it in `Weight.blocks`; nothing frees it. The
graph then decodes each weight to its fp16 plane **in addition** (`cube_for`, cached), reading *from*
those packed blocks, so the 8B holds both the ~4.68 GiB of packed `q4_k`/`q6_k` bytes *and* the
~14.1 GiB of fp16 planes built from them. Per tensor, from the checkpoint's own table:
`token_embd.weight` (`q4_k`) is **0.326 GiB** packed, `output.weight` (`q6_k`) **0.475 GiB**, and the
396 layer tensors **3.876 GiB** — **4.677 GiB** together. Add the KV cache and rotary tables at the
256-position initial capacity (~0.04 GiB) and the computed resident set is **~21.1 GiB**, within ~0.6 GiB
of the measured 21.7 GiB peak (that residual is transient per-op workspace the memtrace caught mid-run,
not anything the model keeps). So the 8B's settled footprint is the *measured* **21.7 GiB**, not any one
computed column — and either figure is already over the ~17.4 GiB pool, which is the point: the planes
alone (14.10 GiB) are **~81% of the pool**, before the weights they were decoded from.

The expansion is not an implementation detail, it is `MatmulCubeCustom`'s contract: the cube is
`half × half → half`, so a packed `q4_k`/`q6_k` block cannot be fed to it — the graph decodes each
weight to fp16 once (`cube_for`, cached) and the plane it keeps is the fp16 one.

**Why it is swap-bound rather than OOM-killed.** The pool *is* host DRAM, and there is a 24 GiB
swapfile (`/mnt/data/swapfile`, on `nvme0n1p1`) with `vm.swappiness = 60`. An allocation that does not
fit in the pool does not fail — it succeeds from the host's point of view and pages churn to NVMe,
which is the 500 s clean run versus the ~1197 s one: the wall is the paging, not the decode. (Swap at
rest, with no model, is already ~0.9 GiB used of 24 GiB.)

**The one missing lever here was a *release*, not an arena shrink — and it is pulled.** The ~4.68 GiB
of resident packed blocks is needed only until its fp16 plane is built, so it is freed then:
`Qwen3Model::load` ends with `materialize_packed_weights()`, which asks the backend
(`Backend::materializes_packed_weights()`, true only here) to build every decode the forward path will
read — `cube_for`'s fp16 plane per GEMM weight, `dense_table_for`'s f32 table for the embedding, and
on a *tied* checkpoint both forms of the one buffer the table and the head share — and then releases
the packed bytes, per weight, largest first. `$POCKETLLM_ASCEND_KEEP_PACKED=1` turns the whole pass
off, and `$POCKETLLM_ASCEND_W4A16=1` turns it off implicitly because that op decodes the packed bytes
on every call and has no cached plane to fall back on.

**What the release measures out to.** On the 8B (`Qwen3-8B-Q4_K_M.gguf`, 36 layers, untied head) the
load now ends holding the 16.42 GiB of decoded forms and none of the 4.68 GiB of packed bytes: it
takes **507.7 s** and leaves **1.06 GiB** of the 23.12 GiB pool free, with `SwapFree` unmoved to
within 1.5 MiB across the whole probe run. (The four-hour accuracy capture above is the longer-run
reading of this configuration: `SwapFree` gives up **539 MiB (0.53 GiB)** once, holds it flat, and
returns it at exit.) The decode that follows runs at **1.73 s per token**
(1.726-1.733 s across eight greedy steps after the 5-token canonical prompt) — the 8B's first clean
*rate* on this board; the same probe reads 0.129 s per token on the 0.6B, which is the canonical
table's 0.129 s/token for that checkpoint, so it is the same instrument measuring both. The contrast
arm is the backend's own kill switch: with `$POCKETLLM_ASCEND_KEEP_PACKED=1` the load is quick
(53.1 s — no planes are built) and the wall moves into the *prefill*, which takes **443.1 s** while
the lazy plane build runs the pool down to 3.73 GiB and puts **4.05 GiB into swap**; the two decoded
tokens then cost 24.0 s and 55.7 s against the release's 1.73 s, with 3.09 GiB of swap faulting back
in. Same checkpoint, same board: the term being released is the 4.68 GiB of packed bytes, and it is
the difference between a 0.58 tok/s decode and a swap measurement.

**The instrument is not the process's resident set.** The pool figures above are
`aclrtGetMemInfo(ACL_DDR_MEM)`, which on this board answers with exactly `/proc/meminfo`'s `MemFree`,
byte for byte. `VmRSS` and `VmHWM` do **not** contain the NPU's `aclrtMalloc` memory: a peak-RSS
comparison of the two arms measures the host's staging and reports the released run peaking *higher*
(2,233,487,360 B against 1,868,619,776 B on the 0.6B). The release's own gate
(`tests/native/test_packed_release.py`) reads the pool for that reason — the same nine greedy tokens
with the release on and off, and **510,107,648 B** of pool back against the 0.6B's 390,491,136 B of
packed weights.

The handle-lifetime hazard this page named is real, and the resolution is *not* to drop the cache
entry: the entry **is** the decode, the handle stays its key for the model's life, and the rule is
order — materialize first, release second, and never decode again. The caches are keyed on the handle
*and* the shape the caller asked for, so a lookup that misses after the release is a shape the load
did not materialize; the backend refuses it by throwing (`require_packed_live`) rather than decoding
freed memory into a plane that would then be used as a weight. That failure would be fluent wrong
text, not a crash, which is why it is a guard and not a comment.

`token_embd.blocks` is read on every forward only in the sense that `embedding_quant` *asks* for it
every forward: `dense_table_for` caches the f32 table under the same handle, so the decode happens
once and the embedding's packed bytes are dead with every other weight's.

**The other lever is the plane's *width*, and the measured middle is untried.** Shrinking host-side buffers moves
nothing (the prior section: the heap is ~33 MiB and every staging buffer is a local). The W4A16 packed
path would cut the 14.1 GiB of fp16 planes to ~4.4 GiB — but it *requantizes* `q4_K → int4` at
~1.2e-1 relative error ([#590](#precision-two-different-errors-and-which-one-is-the-graphs)), which
fails the page's 1e-3 bar and produces incoherent text; that is measured, not speculative. A faithful
int8 plane (~3.8 GiB, `q4_K` decoded exactly and stored as int8 rather than fp16) would need a new
cube op or a new dtype the op accepts, and it is the one middle rung nobody has built. Until then the
8B's home on this board is a ~17 GiB pool that its fp16 planes fill, and with the packed bytes
released the 8B's decode is a rate after all — **1.73 s per token** on the probe above — and
the canonical table's 8B row now carries the post-release `--steps 8` vs `32` marginal,
**1.55 s/token** (see below).

*Verified on the board, read-only, for this section: the idle `npu-smi` charge, `/proc/meminfo`,
`/proc/cmdline`, the swap device and swappiness, and the full GGUF tensor table. The plane, head and
f32-table figures come from the per-tensor element counts; the **packed-weights term is independent of
the peak** — it is the direct sum of every `q4_k`/`q6_k` tensor's `nbytes` in the checkpoint's own
table (5.021 GB = 4.677 GiB), and `bind_packed` is what makes those bytes resident. The **21.7 GiB**
`aclrtMalloc` peak is the **prior** measurement recorded in the section above — it was not re-taken
here, and no 8B run was made, because a run thrashes swap for 500–1200 s.*

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

### The per-op stage matrix (the profiler's second table)

The per-stage table above answers "how much conversion is there in a step" but not **"which op's
conversion"** — and the second is the question a boundary change asks. The clock now records each
op scope's *own* stage deltas, and the profile prints a second table at exit:

```
[pocketllm ascend profile] by op x stage (ms)
  op                  to_f16      to_f32     sdma_up   sdma_down  tensor_create ... op_sync   op_total     other
  attention             ...          ...         ...         ...            ...        ...        ...        ...
  gemm_quant            ...          ...         ...         ...            ...        ...        ...        ...
```

One row per op, one column per stage, plus the op's own-scope `op_total` and an **`other`** column —
the part of that op's wall the stage clocks do not name, which is the host glue around the op (the
`attention` memcpy repack, the `aclTensor`/`GetWorkspaceSize` setup, the loops that feed a launch).
It is the raw material for the split in
[the batched-decode bound page](ascend_310b_batch_decode_bound.md#5-what-the-term-actually-is-split):
that page reads the residual it used to leave at 56% off this table.

It is the same `POCKETLLM_ASCEND_PROFILE` gate as the per-stage table: with the flag unset no timer
is armed and no clock is read, and it was landed under the same byte-identical gate. Instrumenting
the last nine `aclrtSynchronizeStream` call sites (the per-stage table's `op_sync` column had only
`gemm_quant`'s two) is what makes the matrix honest — before it, six ops reported a zero kernel and
their host wall was over-counted into `other`.

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

## How much a batch would actually buy, and what blocks it

The section above answered "is batching the lever?" with the m-sweep. This one decomposes that sweep
into a fixed and a marginal part, reads the runtime for what a batched decode would have to change,
puts an honest multiplier on B = 4 and B = 8, and records **a measured single-request inefficiency
that is worth more than the batch path** and needs none of it.

**The m-sweep, re-measured (`pocketllm-mscale --device ascend --n 2560 --k 2560 --rep 50`).** The
rates reproduce the section above to a few percent; what is new is the time, not just the rate:

| m | ms per call | GFLOP/s | slower than m = 1 | B rows in one call vs B sequential m = 1 |
|---|---|---|---|---|
| 1 | 0.694 | 18.9 | 1.00× | 1.00× |
| 2 | 0.735 | 35.7 | 1.06× | **1.89×** |
| 4 | 0.824 | 63.6 | 1.19× | **3.37×** |
| 8 | 0.976 | 107.5 | 1.41× | **5.69×** |
| 16 | 1.268 | 165.5 | 1.83× | **8.76×** |
| 32 | 1.922 | 218.2 | 2.77× | **11.55×** |

**The op is a fixed cost plus a small per-row cost: `ms(m) ≈ 0.657 + 0.0393·m`.** At m = 1 that fixed
part is **95% of the call**, which is the m-sweep's own proof that the m=1 decode GEMM is paying for
the *tile*, not for the row: a width-m GEMM costs almost the same as a width-1 one until m fills the
cube. The rate is flat in n (same m gives the same rate at n = 2560, 9728 and 151936), so this is the
M dimension of the array sitting idle, nothing else.

**What actually blocks a batch — from the code, not from the flag.** `supports_batch = False` is a
*conclusion*; these are its causes. A B-row decode step needs, in order of how large the change is:

1. **A KV cache per sequence.** `Qwen3Model` holds one `k_cache_`/`v_cache_`, sized `[layer][position]
   [kv_head][head_dim]` (`qwen3.h`), and `Session` advances one `position_` by `cache_length_`
   (`session.cpp`). A batch of B independent sequences needs B such caches (or one cache with a B-wide
   position axis), plus a per-sequence position vector and per-sequence `cache_length_` — because the
   sequences in a batch are at *different* positions, which is the same fact that makes attention
   non-uniform. This is the largest piece: it changes the cache layout, `forward`'s write offsets,
   and the `Session` contract in one go.
2. **A batched attention.** `attention` (`backend.cpp`) is driven **one query at a time** —
   `for (t = 0; t < q_len; ++t) run_attention(...)` — because `AttentionStepCustom` scores one query
   against its own causal window and the loop is where the causal structure lives. A batch of B rows
   at B different positions is B such steps with B different windows; the op is already a
   one-query-against-a-window kernel, so batching it means either B drives per layer (no kernel
   change, no amortization of the window read) or a new multi-query path in the op.
3. **A batched prefill/decode seam.** `forward` sizes its scratch to `n` and its cache to `end_pos`,
   and the head is projected at literal `m = 1` (`x_last` is one row). Batched decode is `m = B`, so
   every downstream shape and the head's row selection move with it.
4. **The serving seam.** `native_backend.py` serializes behind one lock and declares
   `supports_batch = False` precisely because the C `Session` has one cache and one position; a batch
   path means the adapter collecting B requests into one `pocketllm_forward`, which the header has no
   entry point for.

None of these is a kernel change; the cube, the norms and the rope are shape-agnostic already (the
m-sweep *is* the proof). The change is the **runtime's state model** — one cache and one position
becoming B — which is why "batch serving" is a project and not a flag flip.

**The honest ceiling.** The batchable half of a decode step is the weight-reading (the GEMMs); the
other half — attention, the KV traffic, the per-op host round-trips — does **not** shrink when the
batch grows and in fact grows with B. Writing `G` for the GEMM share of the m = 1 step and `R_m` for
the sweep's rate ratio, the speedup is `1 / (G/R_m + (1 − G))`:

| G (GEMM share of the m=1 step) | source | B = 4 | B = 8 |
|---|---|---|---|
| 0.855 | 4B profile, `op_sync` = 85.5% | **2.5×** | **3.4×** |
| 0.68 | 0.6B profile, this run | **1.9×** | **2.3×** |
| 1.00 | the pure-GEMM ideal | 3.4× | 5.7× |

The 4B number is the more favorable one — a bigger model is more GEMM-bound — and the assumption is
stated: **the non-GEMM half is held constant**, which is optimistic, because attention and the KV
reads scale with B and are themselves host round-trips here. So the expected end-to-end factor is
**~2× at B = 4 and ~2.5–3× at B = 8, not the 3.4×/5.7× the pure GEMM suggests**, and the honest
reading is that the number this measurement supports is a *ceiling*, not a projection. It is still a
real lever — nothing else on this board moves the decode rate by 2× — but it is a runtime project,
and the single-request win below is cheaper and already measured.

**The single-request win — and it is bigger than the batch's first step.** While reading the ops for
the blocker list, `rope_neox` (`backend.cpp`) showed up as a round-trip: it reads the **entire**
cos/sin table to the host and re-uploads it on **every call**, when the op only indexes the
`n_tokens · n_heads` rows the batch actually uses. The table is `position_capacity` rows (256
initially, doubling), so at short context it is 128 KB read + 64 KB re-uploaded *per call*, 56 calls
per decode step — and the read is of rows the batch never touches.

The A/B is the check that this is the table and not the token count: the same one-token batch, at a
short vs a ~300-token context, so the table is 256 vs 512 rows —

| rope table | rows | rope cost per decode step | per call |
|---|---|---|---|
| short context | 256 | 24.1 ms | 430 µs |
| ~300-token context | 512 | **49.3 ms** | **880 µs** |

**The per-call cost doubles exactly when the table doubles, at a fixed one-token batch** — so the
cost is the whole-table round-trip, and it *worsens as the context grows*. At 256 rows that is
**~25 ms of a 0.6B decode step: over a fifth of it**, before the table has grown at all.

**The fix, and its measured win.** The tables are a pure function of their buffer — the model builds
them once and only reallocates on a grow — so the correct fix is not to read fewer rows but to stop
re-converting at all: the fp16 tables are now built once and **cached on the device**, keyed on the
source buffers' handles and shape, exactly the contract `cube_for` already relies on for the weight
plane (`rope_tables_for` in `backend.cpp`). The key includes the row count, so a grown table is a
miss and re-converts rather than returning a stale hit; the cached buffers are owned by the cache,
not by the call, so the op no longer releases them. A/B on the board, the same two table sizes, the
profiled binary built both ways and diffed:

| rope table | rows | rope before | rope after | decode step before → after |
|---|---|---|---|---|
| short context | 256 | 25.3 ms/step (21.7%) | **1.5 ms/step (1.6%)** | 116.8 → 93.2 ms |
| ~300-token context | 512 | 48.4 ms/step (19.4%) | **1.3 ms/step (0.6%)** | 249.3 → 218.3 ms |

**The round-trip is gone and the doubling with it** — rope is now flat in the table size (1.5 vs 1.3
ms, noise), where it used to double. The cost was *all* round-trip: the op's own work is ~1.4 ms.
That is a **~24 ms/decode step cut at 256 rows and ~47 ms at 512** — a ~12–20% step reduction on the
0.6B, from one bounded change to one op, with no session or graph change.

**The gate: byte-identical, which is the whole requirement.** This is a pure cost removal, so the
tokens must not move. Built both ways and diffed: the 0.6B canonical prompt is byte-identical before
and after, ids and text (`[12095 13 576 6722 315 9625 374 1083 …]` for 16 steps), and the 1.7B is
too (`[12095 13 576 6722 315 279 3639 4180]`) — 1.7B is the one near-tie this page already carries,
so its gate is pre-change-against-post-change, not against cpu. The 4B is identical as well. The
conformance suite also carries rope cases that compare the kernels against the reference on the
tables the graph builds; they need the tool at the repo-root `build/` path, which this board's
out-of-tree `src/build` does not provide, so on this host they skip and the identity diffs above are
what stands.

**What this does not do.** No batch path was built — the ceiling above is a model with its assumption
named rather than a measured B-way run. And `pocketllm-mscale` is unchanged, so no gate is at risk:
the 0.6B/1.7B/4B identities are as documented above.

### The round-trip class, swept: what else reads a stable buffer

The rope table was not a one-off. It was one instance of a class worth naming exactly: **an op that
reads a whole device buffer to the host and re-uploads it on every call, when the buffer is (a) a pure
function of a stable, bound tensor and (b) larger than the slice the call uses.** Swept against that
definition, every op in the ascend backend falls into one of three groups.

**Stable buffers already cached** — the weight plane (`cube_for`), the embedding table
(`dense_table_for`), the rope tables (`rope_tables_for`). Each reads only its activation per call.

**The one stable buffer that was not** — `rms_norm` narrowed its gamma on the host on every call
(`read_f32(weight, d)` + `to_f16` + `copy_to_device`), but the gamma is a bound f32 tensor the graph
never rewrites: a pure function of a stable buffer, exactly the rope case. It is now cached the same
way, keyed on the source handle and the element count (`narrow_weight_for`). The win is real but
**small**: differencing `--steps 1` against `--steps 31` on the 0.6B, `rms_norm` fell from
**3.55 ms/step to 2.95 ms/step** (two repeats: 3.567→3.023, 3.547→2.907), and the per-step `to_f16`
call count fell from 563 to 450 (the ~28 layers × 2 norms × the gamma). That is **~0.6 ms of a ~95 ms
decode step, under 1%** — well under the materiality threshold the rope fix cleared, because the
gamma is `d = 1024` elements against the rope table's `rows × d/2`. It was taken anyway: same class,
same contract, one op, and the 0.6B/1.7B/4B identity gate is byte-identical.

**The one other stable read, not hot** — the dense `gemm` builds its weight plane per call from the
f32 weight (`run_cube` → `build_cube_weight_from_f32`). It is the same class, and the graph's decode
never reaches it: every decode GEMM is `gemm_quant`, whose plane is `cube_for`-cached. Left for a
follow-up rather than fixed here.

**What is *not* this class, and how its cost actually splits.** `attention` reads its whole KV window
to the host on every call (`read_kv` of `context × width`), and that traffic is both material and
context-scaling — measured on the 0.6B with matched `--steps 1` against `--steps 11` windows,
attention is **~5.6 ms/step at a ~30-token context and ~240 ms/step at ~600 tokens, a 43× scaling
with context**. But the buffer is the **KV cache, which changes every decode step** — not a pure
function of anything stable, so no cache can hold it. That cost is the attention step's operand
plumbing, and it is not the rope pattern.

Scoping it for a fix, the cost is **not** transfer-dominated, which changes what the fix should be.
Decomposing the ~240 ms/step: the host f32→f16 conversion of the q/k/v operands is **~177 ms (74%)**,
the device↔host transfer of the window is **~46 ms (19%)**, and the op's own NPU work plus the host
repack scatter is **~17 ms (7%)**. The A/B that proves the split is the existing
`$POCKETLLM_ASCEND_KV_F16` lever, which halves the wire bytes — and the step gets *worse*
(325 → 438 ms/step), because the narrow slab adds a `half_to_f32` sandwich on the read side while the
transfer it saves is the minority term.

So the fix is **not** a device-side attention kernel: that removes only the 19 % transfer term. It is
to keep the K/V **projections** f16 across `kv_append` → `attention`, so the operand boundary never
round-trips through f32; that removes the 74 % conversion term and reuses the NPU kernel that already
exists. **That fix is now implemented and measured.**

It needed no new op and no change to what any kernel computes. The board has no device f32→f16 cast,
but the f16 cache path already narrows the projection *once* per token in `kv_append` (an O(n·width)
walk, ~0.1 ms at a decode), and `f32_to_f16` is round-to-nearest-even — so the f16 cache holds
*exactly* the bytes `attention`'s own `to_f16` would have computed from the f32 window. Pointing the
graph at the f16 cache therefore changes only *where* the cast happens, never its result, and
`attention` reads the window and feeds the op with no widen-then-narrow. The switch is a new
default-OFF capability on the interface (`prefers_kv_projection_in_cache_dtype()`, alongside
`preferred_kv_dtype()`) so every other backend and the reference are untouched; the ascend backend
overrides it and `$POCKETLLM_ASCEND_KV_PROJ_F16=0` turns it back off for the A/B.

**Measured on the 0.6B at ~600 tokens, byte-identical before and after.** The window is `--steps 11`
against `--steps 41` — the marginal over 30 steps, *after* the one-time weight load and plane build,
which a `--steps 1`-vs-N window divides by N and so inflates (see the wall note below). f32 cache vs
f16 projection:

| | f32 cache | f16 projection | |
|---|---:|---:|---|
| `attention` op (profiled, marginal) | 235 ms | **31 ms** | **7.6×** |
| `to_f16` (host convert) | 193 ms | **3 ms** | gone |
| decode step (profiled marginal) | 321 ms | **116 ms** | **2.8×** |
| decode step (wall marginal) | 615 ms | **309 ms** | **2.0×** |
| short context (~30 tok, wall) | 102 ms | 102 ms | unchanged |

The `to_f16` term — the widest single column, the host narrow of the whole window every call —
collapses to near zero, and with it the widening that fed it; the `attention` op drops 7.6×. The wire
traffic (`sdma_down`) *falls* too (the window is read as f16 rather than as f32), and the profiled step
total drops 2.8×. At short context the step is **unchanged** (102 vs 102 ms/step over the same window):
attention is a small share there, so a fix to it should do nothing, and it does nothing — which is the
check that this is a scaling fix and not a constant-time win.

**The wall clock: one honest number, and why an earlier draft stated a different one.** The
long-context decode wall is **~615 ms/step (f32) → ~309 ms/step (f16), a 2.0× step**; the fix halves
the ~600-token decode step and the win grows with context, since attention's share does. #638 reported
~620 ms/step here from walls of 202.69 / 208.87 / 215.03 s at 1 / 11 / 21 steps — **+618 then +616,
linear, correct** — and this re-measurement reproduces it at 615. The fix's first draft instead printed
**1234** ms/step, from `(wall₁₁ − wall₁)/10`. That is the *same* window and the *same* one-time cost
divided by 10 rather than 20: the two campaigns' `wall₁₁ − wall₁` were 12.34 s and 12.36 s — identical
— so the page was never reporting two different walls, only one cost at two scales. Both windows are
short enough to still contain the one-time build; 1234 is the artifact of the smaller divisor.
**~615–620 ms/step is the honest long-context marginal**, and it is what this page now states.

**Short context is the same windowing story, and there the fix changes nothing.** At ~30 tokens the
step must not move, and it does not: 102 vs 102 ms/step over a matched 30-step window. #638's
~307 ms/step came from a 10-step window (`(wall₂₁ − wall₁₁)/10`) and this page's earlier ~101 from a
30-step one — the board's one-time cost seen through different divisors, not two decode rates. A
short-context run here is barely longer than its own load: one session measured steps 1 / 11 / 21 at
24.56 / 24.55 / 24.54 s and steps 31 / 41 at 27.62 / 27.61 s — flat, then one ~3.06 s step, then flat
again — so *any* short run's "ms/step" is an artifact of where that step lands in the window. The
short-context result is therefore stated only as **unchanged**, which is the claim that matters and is
divisor-independent.

**The gate is the whole requirement, and it holds: byte-identical on the 0.6B, 1.7B and 4B** (token
ids and text), because the f16 cache stores the same rounding the f32 path applies. That is what makes
this a cost removal rather than a numerics change; had a projection cast moved the rounding, the gate
would have failed and the change would have been dropped, not retuned.

So the sweep found one more instance, fixed it, and confirmed the class is otherwise exhausted: the
next material round-trip was attention — as an operand-conversion boundary, and the conversion is now
gone. `pocketllm-mscale` is unchanged.

## What the short-context decode step is — and where it is *not*

The #643 fix left the ~30-token step unchanged at ~102 ms/step, which is the correct result: at short
context attention is a small share, so a fix to attention should not move it. That leaves the
**context-independent** part of the step to name — the per-step weight stream, the logits read-back,
argmax/sampling, and launch. Profiling a short-context decode on the current build (`--steps 11`
against `--steps 41`, the marginal over 30 steps, so the one-time plane build cancels the way the
canonical table above does it):

| stage | 0.6B ms/step | share | 1.7B ms/step | share |
|---|---:|---:|---:|---:|
| **`op_sync`** — the cube executing the 36-layer GEMMs | **66.7** | **77.4%** | **185.6** | **87.7%** |
| `sdma_down` — *all* D2H, incl. the 607 KB logits read-back | 4.55 | 5.3% | 5.60 | 2.6% |
| `to_f32` — fp16→f32 widen of op activations | 3.81 | 4.4% | 5.36 | 2.5% |
| `to_f16` — f32→fp16 narrow of op activations | 3.39 | 3.9% | 5.24 | 2.5% |
| `op_enqueue` — host cost of the op call | 3.20 | 3.7% | 3.79 | 1.8% |
| `sdma_up` — H2D | 2.63 | 3.1% | 3.72 | 1.8% |
| `ws_get` + `aclTensor` create/destroy | 1.88 | 2.2% | 2.22 | 1.0% |
| **profiled total** | **86.2** | | **211.6** | |

The per-op table says the same thing from the other side: `gemm_quant` is **77.7 ms/step of the 0.6B's
86.2** and **200.7 of the 1.7B's 211.6** — 90% and 95% — and every other op is a millisecond or two.
`attention` is 2.5 ms/step (2.9%), `rms_norm` 2.9 (3.3%), `rope_neox` 1.6, `kv_append` 0.5, `silu_mul`
1.0. **The short-context step is the weight stream and essentially nothing else**, and that is not a
surprise at 30 tokens — it is the same fact the batch section already established, that a decode step
is one m=1 GEMV per weight tensor at ~19 GFLOP/s.

**Is it at the roofline? Yes, by the same arithmetic as the m-sweep.** A 0.6B decode step is ~1.07
GFLOP (28 layers × ~13.6 MFLOP + a 155.6 MFLOP head), so 66.7 ms is **~16 GFLOP/s**; the 1.7B step is
~3.68 GFLOP over 185.6 ms, **~20 GFLOP/s**. The m-scale table's m = 1 row measured **19.0 GFLOP/s** on
a synthetic GEMV. The three agree, from three different harnesses, so the short-context step is
**weight-streaming-bound at the board's m = 1 GEMV rate** and there is **no host-side lever in it** —
the same conclusion the 4B section reached, now for the small model at short context. The only lever
that reaches a decode GEMM rate is a wider m (batch serving), which is the batch section's subject and
not a change to this step.

**Item 2 — the argmax read-back: the ABI advertises four bytes and the *device* does four bytes, but
no caller reaches it, and on the C path reaching it would save nothing.** `backend.h` says `argmax`
returns a device address "so the caller transfers four bytes instead of the whole logit vector when it
only wants the token". The ascend backend honors that: `argmax` drives `aclnnArgMax` and reads back 8
bytes. But the whole-vocab transfer does **not** come from the argmax path — it comes from
`pocketllm_forward`, whose signature is `(session, tokens, n, float *logits, logits_cap)` and which
**must** write the logits, because the ABI's contract is "return the vocabulary size — which is also
the number of floats written". Measured, that transfer is not a cost:

- **`aclrtMemcpy` D2H of 607,744 bytes (the 151,936-float vocab) moves at 8.7 GB/s = 0.070 ms**, and
  8 bytes moves in 0.0018 ms — measured directly with a standalone ACL benchmark on this board. On the
  C path the host then does a 0.080 ms memcpy and a **0.20 ms** `kernel::argmax` scan. **Total, the
  entire logits read-back and greedy pick on the C path is ~0.28 ms of a ~102 ms step: 0.3%.** The
  607 KB sits inside the profile's `sdma_down` (4.55 ms, and that column also carries every op's
  read-back), so it is not even individually visible.
- **On the Python path it is a different number, and this is the finding.** `native.py`'s `forward`
  declares `out = (ctypes.c_float * vocab)()` and returns `[float(out[i]) for i in range(vocab)]` —
  it **materializes the whole vocabulary as a 151,936-element Python list**, and `Engine.argmax` then
  **repacks that list back into a ctypes array** to hand it to the C scan. Measured on the board:
  the list build is **57 ms/step** and the repack is **72 ms/step** (the C scan itself is 0.20 ms).
  End to end, `pocketllm run --device ascend` (which crosses this path) reads **308 ms/step where the
  C binary reads 102** over the same `--steps 11`/`41` window — a **~205 ms/step** host-shell cost that
  is almost entirely the Python list round-trip, larger than the C decode step it wraps.

So the answer is split, and both halves are useful. On the **C path there is no lever**: the read-back
is 0.3% and the step is at the GEMV roofline. On the **host shell there is a large one** — the Python
bridge materializes the full vocab twice per step for a token that is one integer, when
`Session::forward` already has an `argmax_out` that fills the id from the device in 8 bytes.
**The interface change landed as the follow-up (#646)**, and it is small and mirrors #643's shape. A
new additive ABI call, `pocketllm_forward_argmax(session, tokens, n, int32_t *out_token)`, returns the
greedy id through a 4-byte out-parameter and leaves `pocketllm_forward` untouched for samplers;
`native.Engine.next_token` calls it and never builds a list; the CLI's greedy loop and the serving
adapter's greedy path both route through it. It is a *host-shell* change, not a kernel one — the
device `argmax` was already there — so the sampling path still calls `forward` and still needs the
logits. Measured on this board, 0.6B `q4_k_m`, the same `--steps 11`/`41` window:

| greedy path | ms/step | note |
|---|---|---|
| before — `argmax(forward())`, vocabulary materialized on the host | **266–269** | list build 57 + repack 72 measured on the shell |
| after — `next_token()`, id read back in 8 bytes | **137** | the whole-vocabulary round trip is gone |

That delta is **~130 ms/step**, measured directly as the two-loop difference on one binary, and
#644's isolated measurement of the two host terms the fast path removes — the 151,936-element list
build (57 ms) and the repack back to a ctypes array (72 ms) — accounts for essentially all of it.
The C path's own vocab memcpy and 0.20 ms scan, which the fast path also skips, are the negligible
remainder. The new path is **byte-identical** to the old — it is the same C `argmax`,
so the token sequence is unchanged — verified on `cpu` and `ascend` over 16 greedy steps from the same
prompt. The win is greedy-only, which is the default and the documented fast path.

**On the two before-numbers.** They describe the same before-state but are not the same
measurement. **266–269 ms/step** is this change's own figure, and it is a *difference of two
loops on one binary* — greedy-through-`forward` minus greedy-through-`next_token`, with
nothing else inside the timed region — so it is read against the after-number from the same
instrument. **308 ms/step**, in the paragraph above, is #644's figure for the same command: a *sum*
of two separately-measured terms (102 for the C step plus ~205 for the host-shell overhead), not a
single timed loop. The ~40 ms between them is that difference — a summed estimate against
a timed whole — and the directly-measured **266–269** is the before-number the after-number
in the table below is measured against.

**#644 itself changed no behavior.** This section is measurement, not code: no op and no graph moved,
and the ABI change it recommends is #646's, above, not this section's. The identity gates (#643's
byte-identical 0.6B/1.7B/4B) stand because nothing was rebuilt. The D2H and
sampler figures come from standalone benchmarks against the shipped `libpocketllm.so`; the per-stage
and per-op tables come from `POCKETLLM_ASCEND_PROFILE=1` on the same binary the canonical section's
numbers were taken from.

**Validated on the serving path (#646).** The fast path is not only the CLI's: `serve`'s greedy
loop reaches the same `_advance`, so it was measured through the real HTTP surface on this board
(`pocketllm serve --device ascend`, 0.6B `q4_k_m`, one greedy `/v1/completions`, 16 tokens).
Before and after were the **same binary** with only `native_backend.py` swapped between the
pre-#646 loop and `_advance` — the change is pure Python, so the C engine is held fixed.

| metric | before (pre-#646 loop) | after (`_advance`) | note |
|---|---|---|---|
| greedy request, total | **4430 ms** | **2304 ms** | Δ 2126 ms = **132.9 ms/step** over 16 tokens |
| TTFT (streamed) | **334.8 ms** | **210.4 ms** | Δ 124.4 ms |
| sampling request (T=0.7) | 7076 ms | 7242 ms | unchanged (noise), still 200 |

The per-step saving, **132.9 ms**, is the CLI's ~130 ms — the same `_advance`, reached over HTTP.
The response **bodies are byte-identical** before and after (greedy and sampling; only `created`
differs, as it does between any two requests), and the served greedy text is *exactly* the
engine-level greedy output the identity gate produced — tokens `[12095, 13, 576, … 9625,
13]`, decoded ` Paris. The capital of France is also the capital of the Republic of France.` —
so the fast path is transparent on the surface a client sees, not just under it. The sampling
path is unchanged because it still calls `forward`: same binary, same answer, no saving, as
expected.

## Serving on the 310B: coming up, and the thread-bound context that had to be attached

Everything above this point was measured through `pocketllm-run` (the C binary) or the `ctypes`
bridge on the **opening thread**. The HTTP surface — `pocketllm serve` — had never been run on this
board, and it did not come up, for two stacked and independently fixable reasons. Both are fixed now,
and the path is proven through the real server.

**The two gaps.** (1) The CLI's runtime dispatch table (`_RUNTIMES` in `python/pocketllm/cli.py`)
mapped `cpu`/`cuda` → the C engine and `horizon` → the S600 delegate, with no `ascend` entry, so
`pocketllm serve --device ascend` refused at the parser. That was a table omission, not a backend gap
— `pocketllm-run --device ascend` already drove the whole forward and answered `Paris`. (2) Even with
the CLI bypassed, every request 500'd with the opaque `pocketllm_forward failed (-1)`, because the ACL
context is **thread-bound**: `AscendBackend`'s constructor called `aclrtSetDevice(0)` and
`aclrtCreateStream`, which bind the current context and stream to the **calling thread**, and
`forward` never re-bound one on any other. `serve` is a `ThreadingHTTPServer` — one thread per
request — and the adapter opens the engine on the main thread, so the handler thread had no current
context and every op launch failed. The `cpu` backend has no such problem (forward from any thread is
fine), which is why it went unnoticed: the C binary is single-threaded and read the engine on the
thread that built it.

**The fix, and why this shape.** `_RUNTIMES` gains `"ascend": "native"` — the C engine carries the
ascend backend, so `--device ascend` drives the same ctypes runtime as `cpu`/`cuda`; the CLI names a
*runtime*, not a kernel. In the backend, the constructor now creates an explicit context
(`aclrtCreateContext`) and `ensure_context()` — a `thread_local` guard — attaches it with
`aclrtSetCurrentContext` before each device op. CANN's rule is exactly why: `acl_rt.h` says
`aclrtSetDevice` "implicitly create[s] the default context … for the calling thread", so a context is
per-thread and must be re-attached on each new one; `aclrtSetCurrentContext` is idempotent ("the last
one prevails"), so the guard reduces it to once per thread while still rebinding a thread that meets a
second backend. Sharing one context and one *stream* across threads is what CANN permits "if the user
guarantees the execution order of tasks in the same stream" — which this backend does: every op
synchronizes the stream before returning and its callers serialize, so no two threads ever have work
in flight at once. That is also why no per-thread stream is created: a second stream would be a way to
*overlap* work the session (one KV cache, one position) cannot support.

**The proof, through the real HTTP server** (0.6B `q4_k_m`, `pocketllm serve --device ascend`):

| | result |
|---|---|
| `pocketllm serve --device ascend` starts | **yes** — `/ready` → 200, banner `Qwen3-0.6B-Q4_K_M.gguf on ascend` |
| one `/v1/completions` | **200**, `finish_reason: length`, 19.5 s — ` Paris. The capital` |
| **four concurrent, distinct prompts** | **4/4 correct, all 200**, wall 5.9 s — `Paris` / `Tokyo` / `4\n$$` / `the first president of` |

The concurrent row is the point: each of the four returned its **own** correct text, none garbled —
the worker-thread path now works, where before every one of them 500'd. The **0.6B gate is
unchanged**: `pocketllm-run --device ascend --prompt "The capital of France is" --steps 8` still
prints `[12095 Paris 13. 576 The 6722 capital 315 of 9625 France 374 is 1083 also]`, bit-identical to
`--device cpu` — the change is in the op entry points, so that identity is the check that it is
transparent.

**The over-cap path, since fixed to a 400.** At the time of the serving fix an over-cap prompt was
still a plain **500** (`pocketllm_forward failed (-1)`, 0.2 s) rather than a client error:
`Qwen3Model::forward` throws before any GEMM when the sequence would pass the checkpoint's 40960
window, and `c_api.cpp` flattened every `std::exception` to `-1`. That is distinct from the #622
over-cap *crash* (which was the S600/horizon delegate), and it is now its own fix. The engine's
verdict is made distinguishable across the ABI instead of being guessed at on the host: the
context-length case throws `ContextLengthError` and `pocketllm_forward` returns
`POCKETLLM_ERR_CONTEXT_LENGTH` (`-3`) for it, ahead of the generic `-1`. `native.py`'s
`ContextLengthExceeded` is raised from that code, and `native_backend.py` maps it to
`ConfigurationError`, which `_error_status` renders as **400 `invalid_request_error`** naming the
limit: `{"error":{"message":"the prompt is 50000 tokens, past this model's context window of
40960","type":"invalid_request_error"}}`. A return code was chosen over a host pre-flight check
because the C engine's `capacity_` is the authority on the window — the host's `_context_length`
falls back to a hardcoded 4096 when the GGUF lacks the key — and because the same code catches a
decode that walks past the window one token at a time, which a prompt-length check cannot see.

## Current state — the canonical numbers after the series

The numbers above were taken across five separate branches (#597–#601), each adding an env knob or a
tool, so they do not all come from one binary and two of them (the 0.6B and 4B decode rates in their
own sections) were measured with the residual *method* this section corrects. This is the single
consistent measurement, on merged `main` at `d16f182`, from one build with **every knob at its
shipped default** (see the check below). It supersedes the per-section figures where they differ, with one exception: **the 8B
row is a post-release re-take** on the #676 build with the release on and the same shipped
defaults, because the pre-release binary could not separate its marginal (see below).

**Method.** `pocketllm-run`, `--prompt "The capital of France is"` (5 prompt tokens), `--device
ascend`, on a quiet board. *Decode* is the `--steps N` / `--steps M` marginal — the walls differenced
and divided by the steps between them — so it carries neither the one-time plane build nor the
prefill. *Prefill* is the slope between a 5-token and a 100-token prompt at `--steps 1`: **both runs
pay the identical plane build, so the difference cancels it** and the slope is the true prefill rate.
(The page's earlier 0.6B "1.63 s/token prefill" was the residual of an *under-estimated* build floor,
not a measured slope; this method does not have that failure mode.) *Plane build* is `--steps 1` at
the 5-token prompt minus that prompt's own prefill.

| checkpoint | decode, marginal | prefill (5→100 tok) | one-time plane build | `--steps 8` wall |
|---|---|---|---|---|
| Qwen3-0.6B | **0.129 s/token** | 0.065 s/token | 2.1 s | 12.8 s ⚠ |
| Qwen3-1.7B | **0.390 s/token** | 0.065 s/token | 5.4 s | 18.9 s |
| Qwen3-4B | **0.640 s/token** | 0.129 s/token | 14.4 s | 41.2 s |
| Qwen3-8B | **1.55 s/token** | — | — | — |

⚠ **The 0.6B wall is not comparable to the other rows, and neither is the column across
sessions** — the process wall is quantized in ~3.07 s bands and the 0.6B sits near the *bottom*
band's edge, so the same binary reports 9.7 or 12.8 s for the same work. The plane-build column is
the reproducible one (it reproduced 3.3 / 8.1 / 21.7 s exactly before the register transpose below,
one session later). See [the wall is quantized](#the-process-wall-is-quantized-in-307-s-bands).

**The plane-build column was re-measured twice on 2026-10-14.** First it went 4–8× down — the
`--steps 8` walls read 10.0 / 19.2 / 43.8 s against the earlier 24.5 / 85.9 / 208.9 s, output
byte-identical at every size. Then the transpose described below took the prologue another 1.5× down
(3.3 → 2.1 s at 0.6B, 8.1 → 5.4 s at 1.7B, 21.7 → 14.4 s at 4B), which moved the 1.7B and 4B walls
by one and two bands. The decode and prefill columns are unchanged — no work below touches either.

`cube_for` and `dense_table_for` decoded each weight with `quant::dequant_block` **once per weight**,
so every one of a super-block's 256 weights re-derived the block header — two fp16 loads, the
group/run/nibble index arithmetic, and the packed six-bit scale/min unpack. That decoder is written
for a packed GEMM, which consumes a weight as it decodes and never wants a row in memory; the plane
build is the opposite shape, and it was paying the header 256× too often. Decoding a whole block per
call (`dequant_q4_k_block`/`dequant_q6_k_block`, the same expression and association, verified
bit-identical over 10.24M weights) took `decode_cube` from 6834 to 1204 ms at 0.6B. `build_cube_weight_from_f32`
then transposed with a scalar software `f32_to_f16` and a column-major scatter; the A55's four-wide
`vcvt_f16_f32` (bit-identical to the software round, checked over 2M values) walked in 64-square tiles
took `transpose_cube` from 7922 to 4132 ms. Finally, since decoding a weight is a pure function of its
bytes, both loops stride their rows across threads (`POCKETLLM_ASCEND_BUILD_THREADS`, default the
hardware concurrency; a row writes only its own output, so no lock).

The measured prologue split at 0.6B, from the `one-time prologue` table the profile now prints (it
was previously outside every op scope, which is why the per-stage total read 1.7 s against a 28.4 s
wall):

| phase (0.6B) | before | serial fix | threaded | register transpose |
|---|---|---|---|---|
| `transpose_cube` — f32→f16 + transposed upload | 7922 ms | 4132 ms | 2217 ms | **1060 ms** |
| `decode_cube` — packed block → f32 | 6834 ms | 1204 ms | 662 ms | 662 ms |
| `decode_table` — the embedding gather table | 1707 ms | 672 ms | 289 ms | 289 ms |
| `upload_table` | 125 ms | 125 ms | 125 ms | 125 ms |
| **PROLOGUE** | **16588 ms** | **6133 ms** | **3293 ms** | **2140 ms** |

The register transpose is the last column and is the subject of the next subsection; the three
columns before it are the series described below. The threaded prologue was 8.1 s at 1.7B and 21.7 s
at 4B, and is now 5.4 s and 14.4 s. The threading step's own gain is ~1.9× (6133 → 3293 ms) rather
than the 4× of the board's core count, because the build is now partly bound by the host↔device
copies, not the decode. The build is a plain *host CPU* cost
with `transpose_cube` still the largest single phase left, so the next step for a workload that runs
the same checkpoint repeatedly is to **persist the plane to disk** and skip the build entirely on the
second run; nothing on the board does that today. (Its size is the fp16 plane, ~1.2 GiB at 0.6B.) The
decode rates move with the checkpoint exactly as the weight count predicts (0.6B → 1.7B is 2.8× the
weights, 0.129 → 0.390 is 3.0×; 1.7B → 4B is 2.35×, 0.390 → 0.640 is 1.6× because the 4B's layer
count rises to 36 and its prefill-heavy head is a smaller share of a slower step).

**The transpose is done in registers now, and that is the change to `transpose_cube`.** The old
kernel converted four `f32` at a time with `vcvt_f16_f32` and then scattered the four halves with
four scalar stores to `dst[c*n + r]` — four destinations, four cache lines, per four weights. The new
one walks eight source rows at a time, converts them, and transposes the 8×8 block *in the vector
registers* with the standard three-stage `vtrnq` butterfly (16-bit lanes, then 32-bit, then 64-bit),
so a destination row leaves as one 16-byte `vst1q_u16` of eight contiguous halves instead of eight
two-byte stores. The values are the same round-to-nearest-even conversions — the two kernels are
checked bit-identical, not assumed to be — and only the order the halves reach memory differs. It is
2.10× at 0.6B, 1.86× at 1.7B, 1.73× at 4B: the ratio falls as the weight grows because the larger
planes are closer to the bandwidth ceiling, where the stores were never the bound. An isolated
microbenchmark put the same pair at 356 → 734 Mweights/s on a `1024×3072` tensor.

### The process wall is quantized in 3.07 s bands

A short `pocketllm-run` on this board has a **fixed process-exit tail of about 3.25 s that is armed by
`aclrtSetDevice` and cannot be removed.** It is not the ACL/device teardown in this repository's
destructors, and it is not an op timeout: a standalone probe that called `aclInit` and `aclrtSetDevice`
and nothing else still took 3.6 s to disappear, and eighteen variants — explicit
`aclrtResetDevice`+`aclFinalize`, no release at all, `std::_Exit(0)` straight after `SetDevice`,
`aclrtSetOpWaitTimeout(1)`, `aclrtSetOpExecuteTimeOut(1)`, `aclrtSetOpExecuteTimeOutV2`, and
`aclrtSynchronizeDeviceWithTimeout(1000)` and `(0)` — all landed inside **3.556–3.606 s**, a 50 ms
spread. Setting the knobs before `aclInit` changed nothing either. `aclInit` alone costs 46 ms to exit;
it is `aclrtSetDevice` that arms the tax. In the tool the same tail is ~4.9 s, so ~1.6 s of it *is* our
own userspace teardown on top of the ~3.3 s the attach costs — which is why `std::_Exit(0)` at the end
of `main` saves a real **1.77 s** at 0.6B, 0.90 s at 1.7B and 0.98 s at 4B. Those saves are unchanged
in the kernel-side part, which no userspace change reaches.

**The wall is therefore a staircase, not a line.** `--steps` at fixed 5-token prefill, two runs each:

| steps | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| wall (s) | 9.67 | 9.72 | 9.73 | 12.77 | 12.80 | 15.87 | 18.94 |

Steps 1, 2 and 4 are the *same* wall to within 80 ms even though each does 133 ms more decode —
the extra work hides inside the band. Every band edge adds **3.07 s** (9.72 → 12.79 → 15.87 → 18.94),
which is the fixed ACL tax, not anything the step did. So a short run's wall carries a ~3 s quantum
that swamps any host-side win smaller than a band, and it is the reason the canonical table's wall
column cannot be compared across sessions: the same binary on the same work reads 9.7 or 12.8 s
depending on which side of an edge it lands.

**The consequence for measuring.** A change that moved the *work* by 1227 ms at 0.6B — `model_loaded`
5303 → 4089 ms, `main_end` 7781 → 6536 ms, measured by stamping the phases in the tool — left the wall
inside 0.04 s of where it was, because both arms sat in the same band. The prologue table and a
`main_end` stamp are the instruments that resolve it; the wall is not. That also means the plane cache
mooted at the end of the previous subsection is worth less on the 0.6B short run than its build cost —
a cache that saves a second cannot save a band, though it still pays at 4B and for repeated runs.

**Decode is not one rate — it rises with context.** The marginal is a function of where the window
sits, because every decode step attends to a KV cache that has grown by one. On the 0.6B, holding the
`--steps 8` anchor at 24.56 s across the series:

| bracket | marginal |
|---|---|
| 8 → 32 | 0.127 s/token |
| 32 → 64 | 0.193 s/token |
| 64 → 128 | 0.240 s/token |

So "0.6B = 0.13 s/token" is the *short-context* figure and the page's older "0.26 s/token" is a
longer-context one; both are real points on the same curve, and neither is wrong. The table above
quotes the 8→32 bracket because it is the range every checkpoint shares and the one a generation
actually opens in.

**The 8B row is a post-release re-take, and the release is what makes its marginal separable.**
Before the release that marginal was a memory fact, not a decode one: at 8 steps the run completes
in **486.5 s** and is token-identical to CPU (the gate above), but its **~21.7 GiB** measured device
peak (16.42 GiB of planes and table) against the pool's ~17.4 GiB usable, plus a 4.7 GiB mmap'd
checkpoint on a 23.7 GiB host, pushes `MemAvailable` to ~0.6 GB and the process into swap: the same
`--steps 8` run takes **486.5 s** on a quiet board (nearly all plane build) and its `--steps 32` leg
was timed at **2402 s**, with an intermediate `--steps 1`/`8` pair at 634 s / ~16 min, and the
marginal between two such walls is dominated by how much the kernel paged out, not by decode (it can
even come out negative, a longer run whose plane build reclaimed more cleanly beating a shorter one
that paid the eviction). With the packed bytes released the same protocol is clean: on the #676
build with the release on (`POCKETLLM_ASCEND_KEEP_PACKED` unset, every knob at its shipped default),
two repeat pairs of the `--steps 8`/`32` walls read 443.95 / 444.13 s and 481.15 / 481.12 s against a
431.40 s `--steps 1` wall, giving 1.54985 and 1.54096 s/token — **1.55 s/token** — with
`MemAvailable` held at ~23.1 GB and `SwapFree` moving under 16 MB. That is the differenced marginal,
the 8-to-32 window every checkpoint shares. The release probe's **1.73 s/step** and the marginal are
different quantities, not revisions of each other: the probe times each of the first eight steps
through the Python bridge, so its step walls keep the bridge's per-call host work; the marginal
differences two CLI walls 24 steps apart, where per-call host work and per-run constants cancel
alike. Both are real points on the 8B's curve; the probe's per-step record above stays as it is.

**Every knob defaults to the shipped path.** The series added five env knobs; each is checked here to
default to the value the page's numbers were taken at, so the default binary *is* the measured one:

| variable | default | effect when unset |
|---|---|---|
| `POCKETLLM_ASCEND_PROFILE` | unset | no timers, no clock read (#597) |
| `POCKETLLM_ASCEND_CUBE_CHUNK_N` | `8192` | the shipped chunk width (#599) |
| `POCKETLLM_ASCEND_CUBE_SYNC` | unset | sync per chunk, the shipped shape (#599) |
| `POCKETLLM_ASCEND_W4A16` | unset | the faithful dense path, not the requantized op (#599) |
| `POCKETLLM_ASCEND_KV_F16` | unset | f32 cache (`preferred_kv_dtype` = kF32) (#582) |

The four identity gates hold on this build, from the binary each table row came from (the 8B row is its
post-release re-take and re-prints the same 8-step list):
**0.6B token-identical**, **1.7B the documented near-tie** (`17689 Spain` vs `279 the` at generated
index 5 — reproduced exactly, same two ids, same order), **4B exactly**
`[12095 Paris 13. 576 The 6722 capital 315 of 9856 Germany 374 is 19846 Berlin]`, and **8B
token-identical**. So the series is behavior-preserving: nothing #597–#601 added moves the output,
and the default path is the shipped one.
