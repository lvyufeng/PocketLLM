# DeepSeek-V4.1-Flash: the dense stack in fp16, measured and withdrawn

> **The width is not shipped.** `LINEAR_DTYPE` and `CACHE_DTYPE` are **bf16** in this tree. fp16 is a
> real 4.34× lever on the dense calls of a 4096-token chunk, and everything below measures it, but
> the width cannot hold what this model's own prompts carry: on the prompt the chat renderer builds,
> `[0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822]`, the Hyper-Connections residual stream
> reaches **2.04e+06** at bf16 and the input to a layer's norm **4.57e+05** — 31× and 7× fp16's
> largest finite, 65504. Narrowed to fp16 that forward has **415 non-finite sites of 956**, all-NaN
> logits and eight begin-of-sentence tokens where bf16 answers `The capital of France is
> **Paris**.` The section that missed it is **Headroom** below, and the mistake was the chunk it
> metered rather than the arithmetic: the probe tokenizes *prose*, which behaves like this prompt
> with its leading BOS removed — the same sequence without that token stays under 200 at every site
> at both widths. Evidence: `/tmp/probe_v41_nan_site.py` (the site trace) and
> `/tmp/cherry_bos_bf16.sh` (the same two prompts on both widths), with `LINEAR_DTYPE` back at bf16
> by `fix/v41-dense-stack-carrier`.

[The remaining-bottlenecks page](deepseek_v4_1_flash_remaining_bottlenecks.md) left the tree's dense
arithmetic where it started: every projection, the embedding, the router and the shared expert at
`torch.bfloat16`, because that is what the checkpoint holds and what the reference computes in. On
this card that is the expensive choice rather than the safe one. **sm_75 has no bf16 tensor core**,
so cuBLAS does not answer a bf16 GEMM with a tensor-core kernel at all — it falls through to the fp32
SIMT path — and fp16 at the identical shapes runs 6.3–9.2× faster. Setting `LINEAR_DTYPE` and
`CACHE_DTYPE` to `torch.float16` is worth **1.92 s a rank on a 4096-token chunk over the 716 dense
calls that chunk makes, 4.34×** — on a chunk of prose, which is the scope the measurement has and the
reason it was withdrawn — and it moves nothing else: the same tree at bf16 (the patch with the
two constants flipped back) produces **bit-identical logits**, `max |diff| = 0.000e+00` on eight
position/rank pairs, while on a real prompt the fp16 arm keeps the argmax on the same token at every
position measured and the top-8 losing at most one member — a claim this page scopes to the prefill
geometry, because on a one-token input it does not hold and saying so is the decode section's job.

This page is the evidence for that paragraph, and the two limitations that come with it. The per-site
lever is the reproducible instrument; the wall it lands in is not (0.46 s to 3.5 s depending on how it
is measured). And the arm-level logit difference between the two widths is **not** a measurement of
the width: an arm of this probe differs from its own next chunk by as much as it differs from the
other dtype, in a patch-free tree too, so that comparison is an upper bound and is reported as one.

| Configuration | TP4, one process a card, one process a rank on 4 x RTX 2080 Ti, `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `OMP_NUM_THREADS=22` |
| Geometry | A 4096-token chunk from position 0, `--chunk 4096`, `--pool-rows 148`, `_buffers 2` — the shipping configuration's own chunk width |
| Decode arm | The same probe at `--chunk 1`: a one-row forward, 461 dense calls against the chunk's 716. The lever there is **zero** and the argmax moves; both are recorded under **The decode column**. |
| Arms | The in-process A-B-A-B probe: `bf16,fp16,fp16,bf16` (and an eight-arm `bf16,fp16,fp16,bf16,bf16,fp16,fp16,bf16` run), the dtype switched by swapping the buffer objects on every `Parameter` at that dtype, one arm at a time |
| Instrument | `/tmp/probe_v41_dense_dtype_abab.py` — a CUDA event pair around every `nn.Linear` call and every `bsgd,grd->bsgr` `einsum`, read back after the chunk's own synchronize, plus the chunk's wall |
| Trees | `/tmp/prefill_csr` (master at bf16) for the in-process runs; `/tmp/prefill_fp16` (the same tree with the three constants flipped) for the separate-process cross-checks; `/tmp/baseline_master` (a `git archive master` extraction) for the patch-free control |
| Control | The bf16 arm of the patched tree and the bf16 arm of `/tmp/baseline_master` are **bit-identical** at both metered chunks on all four ranks — `equal=True`, `max |diff| = 0.000e+00` |
| Logs | `/tmp/abab_dense.log` (8 arms), `/tmp/abab_seq.log` (4 arms interleaved), `/tmp/abab_master.log` (the patch-free control), `/tmp/dense_cost.log` and `/tmp/dense_cost_fp16.log` (the separate-process pair); reduced by `/tmp/reduce_abab_sites.py` and diffed by `/tmp/diff_abab_seq.py` |

## Why a bf16 GEMM is an fp32 GEMM here

The kernel tells the whole story. A bf16 GEMM at these shapes lands on
`magma_sgemmEx_kernel<float, __nv_bfloat16, __nv_bfloat16, false, 6, 4, 6, 3, 4>` — a SIMT kernel
that upcasts to fp32, multiplies and narrows back — at **6.4–7.5 TFLOP/s**, which is 2.398 s of a
28.14 s 4096-token chunk over 397 calls and the fourth heaviest kernel on the device. The identical
shapes in fp16 take `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` at **47–58
TFLOP/s**, an 8× difference that comes from the hardware and not from the shapes, the tiling or the
call pattern. Turing's tensor cores do fp16 and int8; bf16 arrived with Ampere. So on this card a
bf16 GEMM is not a slightly slower fp16 GEMM, it is a different kind of kernel.

Two things make the swap cheap for *this* model rather than merely fast:

- **The weights are fp8 already.** The projections this applies to are stored F8_E4M3 with E8M0 block
  scales and `max|w| = 0.875`. The GEMM's weight is formed in fp32 from the quantized blocks either
  way and then narrowed to whichever carrier, so the storage width of the *dense* stack is a
  destination and not a source — narrow it to fp16 and 11 significand bits survive where bf16 kept 8.
- **Everything else is bf16 in the checkpoint, and bf16→fp16 is lossless inside fp16's exponent
  range.** The compressor's `wkv`, the indexer's `wk`, the router, the embedding and the Hyper
  Connections coefficients all arrive bf16, and a bf16 value's 8 significand bits fit inside fp16's
  11 — so the *weights* of every site the change carries are converted exactly, down to fp16's
  denormal floor of 2⁻¹⁴.

What is not exact is the activations. A GEMM casts its inputs to its weight's width, so a bf16 GEMM
rounds them to 8 bits where an fp16 one keeps 11; that is the arithmetic difference the parity section
below is about, and it is a difference in the *carried* precision rather than an error introduced.

## Headroom

fp16 trades exponent range for mantissa width — 65504 is the largest finite fp16 against bf16's
3.4e38 — so the change stands or falls on whether any activation arrives near it. The probe measures
`x.abs().max()` at every site on the metered chunk:

- **nothing metered anywhere exceeds 65504**; the largest input at any site at all is **202.5**;
- of the sites the change actually carries, **13** would change dtype, and the largest input among
  those is **100** — 0.153 % of fp16's largest finite;
- the sites with the widest inputs are the ones that stay fp32 anyway: the Hyper Connections mixing
  projection at 266 and its own residual stream, and the router at 14.4.

**This is the section that was wrong, and the fault is the chunk rather than the arithmetic.**
`probe_v41_chunk_scaling.prose(...)` tokenizes prose with `add_special_tokens=False`, so the metered
chunk is the *no-BOS* case, and every number above is a no-BOS number. The model's own serving path
does not produce that case: `encode_messages` embeds a leading `<｜begin▁of▁sentence｜>` and tokenizes
with `add_special_tokens=False`, so the first token of every chat prompt is id 0. On that prompt the
same `x.abs().max()` measurement reads **2.04e+06** — the largest tensor in the forward is the Hyper
Connections mixing projection's own fp32 `F.linear (24, 20480)` output at site 879 of 956 — with
**4.57e+05** arriving at layer 30's `attn_norm`, both past 65504 by the time the stack is
three-quarters done. The same sequence with the leading BOS removed reads **200** and **63** at its
two widest sites, which is the number the table above is really reporting: one token of a nine-token
prompt moves this arithmetic by four orders of magnitude, and it is the token every chat prompt
starts with.

The failure is a trace rather than an inference. At fp16, site by site
(`/tmp/probe_v41_nan_site.py`, 956 sites in execution order, run on both widths):

```
  536 bare F.linear (576, 5120)     fp16  in 24.39    out 3.289       finite
  538 bare F.linear (5120, 576)     fp16  in 14.77    out 3.338       finite
  539 layers.21.ffn.shared_experts  fp16  in 24.39    out 3.338       finite
  540 layers.21.ffn                 fp16  in 24.39    out 1.356e+04   finite
  541 bare F.linear (24, 20480)     fp32  in inf      out nan         <-- first non-finite
  542 layers.22.attn_norm           fp16  in inf      out nan
```

The last site holding a value over the ceiling is the Hyper Connections mixing projection's output at
**7.152e+04**, which reads a stream of **5.309e+04**; the site after it is the *same* projection one
layer down, whose input is already `inf`. Nothing writes that stream between them except layer 21's
own `hc_post`, so the overflow is the expansion's sum: `post * x` with the FFN's 1.356e+04 plus the
mixed residual whose copies stand at 5.3e+04, all fp32 in the arithmetic and narrowed to fp16 by the
return. Everything downstream of 541 is a consequence — 415 of the 956 sites are non-finite, 1 of
them carries an `inf` (the site that received it) and the other 414 carry the NaNs an `inf` produces
— and `argmax` over all-NaN logits returns 0 eight times, which the serving adapter faithfully
reports as eight special tokens.

At bf16 the same instrument reads **every one of the 956 sites finite, 0 carrying inf**, the mixing
projection peaking at **2.038e+06** and the final norm reading 3.297e+05: the values are the model's
own, and bf16's 3.4e38 holds them. So the defect is the carrier and not a runaway, and the fix is the
two constants back to bf16 — the width of the *weights* is what buys the 4.34×, but the width of the
*activations* is what has to hold the stream, and at these geometries the weights are fp8 formed in
fp32 either way. `tests/test_models_deepseek_v4_1_modules.py` pins the invariant at the measured
magnitude (2.038e+06 through `hc_post`, 4.567e+05 through `hc_pre`), and it fails at fp16.

## The lever, per site

Per rank, per chunk, seconds. The 8-arm run is 64 chunk samples and the interleaved 4-arm run is 32;
both are reduced from the dumps by `/tmp/reduce_abab_sites.py`, because the probe's own printed
per-site column is **mis-scaled** — its numerator already sums over every rank and every arm while
the print divides by `n_arm * world`, so that column reads *chunks an arm* times the per-rank
per-chunk figure, and its summary `sites` column divides an already-per-rank mean by `world` a second
time and so reads a quarter of it. The dump is the authority: one row is one chunk on one rank, and
each row's site total matches the per-chunk line the probe prints beside it (`716 calls, 2.44 s in
them`). The figures below are those rows, averaged over samples.

| site | bf16 s | fp16 s | ratio | calls |
|---|---|---|---|---|
| `attn.wo_b` | 0.448 | 0.064 | 6.99 | 32 |
| `attn.wq_b` | 0.402 | 0.061 | 6.59 | 32 |
| `wo_a einsum` | 0.364 | 0.054 | 6.76 | 32 |
| `engram.wkv` | 0.318 | 0.044 | 7.30 | 32 |
| `F.linear (576, 5120)` | 0.251 | 0.052 | 4.86 | 32 |
| `attn.wq_a` | 0.248 | 0.044 | 5.69 | 32 |
| `F.linear (24, 20480)` | 0.157 | 0.157 | 1.00 | 32 |
| `F.linear (5120, 576)` | 0.113 | 0.017 | 6.70 | 32 |
| `attn.wkv` | 0.109 | 0.015 | 7.40 | 32 |
| `F.linear (384, 5120)` | 0.051 | 0.052 | 0.99 | 32 |
| `attn.indexer.wq_b` | 0.012 | 0.002 | 5.21 | 32 |
| `attn.compressor.wkv` | 0.008 | 0.005 | 1.52 | 32 |
| `attn.indexer.weights_proj` | 0.005 | 0.001 | 5.90 | 32 |
| `attn.compressor.wgate` | 0.005 | 0.005 | 1.07 | 32 |
| `F.linear (129280, 5120)` | 0.005 | 0.005 | 1.01 | 32 |
| `attn.indexer.wk` | 0.000 | 0.000 | 3.45 | 32 |

**bf16 2.497 a chunk over these sites, fp16 0.576 — the site lever is 1.921 s a rank a chunk
(4.34×).** The interleaved four-arm run reads 2.487 / 0.576 and 1.911 s (4.32×), within a few
milliseconds on every row, and its per-arm walls are the balanced ones (bf16 at positions 0 and 3,
fp16 at 1 and 2).

Everything below `attn.indexer.weights_proj` is under 10 ms a chunk and the ratios there are rounding
noise rather than a lever; they are kept in the table so the call count of 716 can be seen to be
accounted for rather than assumed. The whole tail is 0.017 s of the chunk's 0.576.

**The independent cross-check is on the same basis and agrees to 3 %.** `/tmp/probe_v41_dense_cost.py
--at 0 --chunk 4096` meters the same sites with its own event pairs in three separate processes (bf16,
fp16, and a bf16 run-to-run control) and reads **`-- all sites (716 calls) 2.488 bf16 / 0.631 fp16,
3.94×`**, saving **1.857 s a rank a chunk**; the sites over 0.10 s a chunk, 2.402 against 0.178. Two
instruments, two processes, two code paths, one answer: **≈1.9 s a rank a chunk at ≈4.3×**.

### The wall is not the instrument

The site total is clean because it is a per-call constant: the four fp16 arms read 0.144, 0.144, 0.144,
0.144 s a chunk and the four bf16 arms 0.616, 0.627, 0.627, 0.626 — a spread of 11 ms on a 0.48 s
figure, immune to where in the run the arm sits. The wall is not. The eight-arm run:

| arm | dtype | chunks | wall mean | wall min | sites mean | sites min |
|---|---|---|---|---|---|---|
| 0 | bf16 | 8 | 24.50 | 24.13 | 0.616 | 0.610 |
| 1 | fp16 | 8 | 23.92 | 23.74 | 0.144 | 0.143 |
| 2 | fp16 | 8 | 24.44 | 24.35 | 0.144 | 0.143 |
| 3 | bf16 | 8 | 27.28 | 26.74 | 0.627 | 0.618 |
| 4 | bf16 | 8 | 26.94 | 26.49 | 0.627 | 0.618 |
| 5 | fp16 | 8 | 24.55 | 24.43 | 0.144 | 0.143 |
| 6 | fp16 | 8 | 25.10 | 24.36 | 0.144 | 0.143 |
| 7 | bf16 | 8 | 27.20 | 26.84 | 0.626 | 0.616 |
| — | bf16 | 4 | 26.48 | 24.13 | 0.624 | 0.610 |
| — | fp16 | 4 | 24.50 | 23.74 | 0.144 | 0.143 |

The bf16 arms are 2.78 s apart from each other (24.50 against 27.28), which is larger than the lever
the run is trying to read, and the drift is upward with position in the run and affects both dtypes.
The interleaved four-arm run, where the two dtypes average the same position (1.5), collapses the wall
to **0.46 s** — against a site lever of 1.911 s, a four-fold disagreement. Across all four
measurements of the same thing:

| measurement | wall lever |
|---|---|
| 8-arm in-process, mean of 4 arms each | 1.98 s |
| separate processes, `--at 0`, mean of 6 chunks | 2.36 s |
| separate processes, `--at 32768`, mean of 6 chunks | 1.56 s |
| interleaved 4-arm in-process | 0.46 s |

So the wall reads 1.0–2.4 s on three of the four and 0.46 s on the fourth, and **the page claims the
site lever and reports the wall as corroboration with that spread**. A 1.9 s difference inside a
25 s chunk is 7.6 %, and this host's four ranks contend for one memory system; the run-to-run wall
spread of the *same* arm is the size of the effect being read. The site total is a measurement of
device time inside the dense calls and is what the change is; whether all of it leaves the critical
path is a question this probe cannot answer, and a served end-to-end A/B is what would.

## What does not move, and why

Four of the metered rows are flat. Each is pinned by a line that upcasts both operands before the
GEMM, or by a parameter that is fp32 in the checkpoint, so the storage width never reaches cuBLAS:

| site | ratio | why |
|---|---|---|
| `F.linear (384, 5120)` | 0.99 | The router's `Gate.weight`. `modules.py:443` reads `scores = F.linear(x.float(), self.weight.float()) / self.gate_temp` — both operands are upcast, so this is a storage width and not a routing precision, and the fp32 GEMM is unchanged. |
| `F.linear (24, 20480)` | 1.00 | The Hyper Connections mixing projection: `mix_hc = (2 + hc_mult) * hc_mult = 24`, `hc_dim = hc_mult * cfg.dim = 20480`. `hc_attn_fn` and `hc_ffn_fn` are `torch.float32` parameters **by design** (`modules.py:563-564`, "fp32 in the checkpoint, not bf16: these are the residual coefficients themselves"), and `_hc_mixes_pass` flattens and casts the stream to fp32 before `F.linear`. 258 `hc_*` keys in the checkpoint, all fp32. |
| `attn.compressor.wgate` | 1.07 | Constructed `dtype=torch.float32` unconditionally, and only when the layer's compress ratio is above 1. |
| `F.linear (129280, 5120)` | 1.01 | The embedding/head, whose `forward` is `modules.py:721`'s `F.linear(x.float(), self.weight)` — a **third** fp32-upcasting linear site beside the router and the HC mixing. |

`attn.compressor.wkv` is a fifth case that only half-moves (1.52×): it is fp32 above a compress ratio
of 1 and `LINEAR_DTYPE` at ratio 1, so the arm changes only the ratio-1 layers it has.

One row of the table needs its shape explained, because no checkpoint tensor is that wide. `F.linear
(576, 5120)` and `F.linear (5120, 576)` are the **shared expert** — `Expert` is built with `world=4`,
so `inter = inter_dim // world = 2304 // 4 = 576` (`modules.py:116-119`). A scan of all 48 safetensors
headers finds **no tensor with a 576 dimension at all**, which is the check that says the width is a
sharding artifact and not a stored width. The routed experts, which are the other three matrices an
`Expert` holds, do not appear in this table: on the resident path they never reach `F.linear` — they
go through the fp4 kernels on the device — so the 4.86× and 6.70× above are the *shared* expert's.

### A routed expert's width is the activation's

The change also had to settle what width a routed expert's expanded weight should be, and the answer
is not "whatever the bank holds". `expert_forward` multiplies the activation by the weight, so the two
have to agree; the previous tree had a module-local `EXPERT_DTYPE` constant alongside an
`EXPERT_DTYPE`-free `device_experts.py`, which is one place for a width to be decided and another for
it to be assumed. The patch deletes the module-local constant (there is one `LINEAR_DTYPE`, bound in
`attention.py` and imported by `modules.py` and `loader.py`), converts every expansion
(`dequantize_rows` at `modules.py:322`, `CheckpointRoutedExperts.expert()` at `loader.py:474`) to it,
and puts a guard where the two widths meet: `check_activation_matches_experts` takes a fourth `dtype`
argument and raises *"holds its experts at {dtype} but was handed an activation at {x.dtype}. The two
are multiplied, so they have to agree; expand the experts at the activation's width (`LINEAR_DTYPE`)
or cast the activation at this boundary, not only in one place."* The routed **bank** is untouched by
the width change by construction — `resident_bank.py` stores `uint8` views of the pinned checkpoint
bytes and neither it nor `device_experts.py` names a float dtype — so the levers on the
[device-experts page](deepseek_v4_1_flash_device_experts.md) are not re-priced by this one.

## Parity

The two widths do not produce the same logits, and the question is whether they produce the same
*model*. Three separate readings, strongest first.

**Cross-tree at bf16 is exactly zero.** The patched tree with both constants flipped back to bf16
against `/tmp/baseline_master` (a patch-free `git archive master`) reads `equal=True` and
`max |diff| = 0.000e+00`, `mean |diff| = 0.000e+00`, same argmax, at both metered chunks on all four
ranks — eight position/rank pairs. The patch at bf16 is a bit-exact no-op: it changes which constants
exist and where they are bound, and the guard, and nothing about the arithmetic. That is also what
makes the arm table's bf16 rows comparable across trees at all.

**The separate-process parity matrix.** `/tmp/probe_v41_dense_cost.py --at 0` over three positions and
four ranks, identical on all four ranks:

| position | max \|logit\| | max \|diff\| | mean \|diff\| | ratio to scale | top-8 differing | argmax | margin A | margin B |
|---|---|---|---|---|---|---|---|---|
| 0 | 26.22 | 3.395 | 0.4478 | 0.129 | 1 of 8 | same | 11.66 | 12.40 |
| 4096 | 26.37 | 2.801 | 0.4610 | 0.106 | 1 of 8 | same | 11.97 | 11.35 |
| 8192 | 25.75 | 2.515 | 0.4021 | 0.0977 | 0 of 8 | same | 11.53 | 11.63 |

`/tmp/diff_logits.py par_bf16.pt par_bf16.pt` reads exactly 0, so the diff script is sound, and
`par_bf16.pt` against the control `par_bf16b.pt` — the same tree run twice — reads **exactly 0 at all
12 position/rank pairs on every column**. The largest disagreement anywhere is 0.13 of the logit
scale, the argmax never moves, the top-8 loses at most one member, and the top-1 margin is 11.5–12.4,
i.e. 3.4–4.8× the largest disagreement anywhere. Read as a direction: fp16 carries three more
significand bits, so the difference is dominated by the bf16 rounding the fp16 arm no longer makes,
and the 3.4 is error *removed* rather than introduced — stated as an interpretation, because the
measurement that would settle the direction (a third arm at `LINEAR_DTYPE = torch.float32`, which is a
proxy for the model's own arithmetic rather than a reference) has not been run.

**The arm-level comparison in the A-B-A-B probe is an upper bound and is reported as one.** Every
`bf16 × fp16` arm pair at both metered chunks reads 8.4e-02 to 1.15e-01 of the logit scale, argmax
223 against 223 everywhere, top-8 sharing 6–8 of 8, margins 11.8–13.3. That looks like the same
answer as the matrix above until the control is put beside it: **an arm of this probe does not
reproduce its own next chunk.**

```
bf16 arm 0: chunk 1 vs chunk 2: max|d| 2.872e+00  equal=False  argmax 223==223  top8 7/8
fp16 arm 1: chunk 1 vs chunk 2: max|d| 2.835e+00  equal=False  argmax 223==223  top8 8/8
fp16 arm 2: chunk 1 vs chunk 2: max|d| 2.706e+00  equal=False  argmax 223==223  top8 6/8
bf16 arm 3: chunk 1 vs chunk 2: max|d| 2.430e+00  equal=False  argmax 223==223  top8 6/8
```

Both chunks are the same computation: the same token ids, position 0, and `front.reset_state(1)`
before every forward, so nothing is being read out of the previous forward's caches. Same-dtype
chunk-to-chunk (2.4–2.9) is as wide as cross-dtype arm-to-arm (1.7–3.2), so the arm comparison cannot
attribute its difference to the width.

**The patch-free control says that difference is the configuration's.** Running the identical probe on
`/tmp/baseline_master` — no patch, every parameter bf16, the tree that ships — reads
**`2.872e+00`, `equal=False`, `argmax 223==223`, `top8 7/8` on all four ranks**: the same number as the
patched tree's bf16 arm 0 to four significant digits, and its logits are bit-identical to that arm's
(`max |diff| = 0.000e+00`, `equal=True`, both chunks, all four ranks). So the chunk-to-chunk
difference is a property of this probe's configuration, present in both trees and reproducible across
processes — it is not the patch and it is not noise. The candidate mechanism is the same one the
[chunked-prefill page](deepseek_v4_1_flash_chunked_prefill.md) localizes the first `1e-6` boundary
difference to: the prefill MoE's reduction is an **unordered `atomicAdd`**
(`src/csrc/cuda_kernel_impl.cu:3700`, `:3912`, `:4144`, with no `DEEPSEEK_MOE_DETERMINISTIC_REDUCE`
guard, unlike the single- and multi-token entry points at `:2837` and `:3398`), so a chunk's routed
sum is a function of the order the experts' partials land in. A routing tie resolved differently would
move a token's expert set, which is far more than a rounding difference and is the size seen here.

**The one-token arm isolates that mechanism, and it is what makes it the explanation rather than a
candidate.** The same probe at `--chunk 1` — the decode geometry, which reaches the MoE's *guarded*
single-token entry points (`cuda_kernel_impl.cu:2837`, `:3398`) instead of the unguarded batched
scatter — makes the same-dtype chunk-to-chunk difference **exactly zero**: `max |diff| = 0.000e+00`,
`equal=True`, for all four arms at both chunks on all four ranks, with bf16's argmax 7805 and fp16's
271 each holding. The spread is therefore specific to the 4096-token prefill geometry and not to
re-running a forward from position 0 as such — the deterministic reduction the guard buys is exactly
what removes it, and the same guard on the prefill scatter is the fix the
[CSR page](deepseek_v4_1_flash_moe_reduce_csr.md) shipped for the reduction's cost and that
[the unguarded sites](deepseek_v4_1_flash_chunked_prefill.md) still need for its order.

The consequence for this page is the bound and not the mechanism: **the arm-level logit diff bounds
the dtype's contribution from above and does not measure it.** The evidence that the width change is
acceptable is the cross-tree zero, the separate-process parity matrix with its own zero control, and
the invariable argmax — not the arm table.

## Acceptance at the kernels

fp16 is not being smuggled past the CUDA layer. Every sparse-attention entry point dispatches
`AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, ...)`
(`src/csrc/cuda_kernel_impl.cu:5396`, `:5440`, `:5485`, `:5532`, `:5578`, `:5626`), as do the MoE fp4
kernels (`:2334`, `:2787`, `:3355`, `:4188`, `:4196`, `:4427`, `:4677`, `:4685`, `:4913`), and
`src/kernels/ops.py`'s `sparse_attn` gates every CUDA path on shape and never on dtype.

`CACHE_DTYPE` is not an independent knob and is set with `LINEAR_DTYPE` for that reason: every
sparse-attention entry point dispatches on `q.scalar_type()` and reads `kv` at that same dtype, so a
cache at one width under a linear stack at another is silent corruption rather than an error — the
cache's dtype is a *consequence* of the query's. It keeps a separate name only because it also sizes
three persistent buffers (`Indexer.k_cache`, `window_kv_cache`, `compress_kv_cache`).

## The decode column

At one row there is no GEMM left to speed up: a `Linear` at m=1 is a matvec whose cost is the weight
bytes it reads, and both candidate widths are two bytes. Run on the same instrument, that is what it
measures — **the lever is zero**.

| | bf16 | fp16 |
|---|---|---|
| forwards | 16 | 16 |
| calls a forward | 461 | 461 |
| wall mean / min, s | 0.33 / 0.28 | 0.34 / 0.29 |
| site total mean / min, s | 0.040 / 0.029 | 0.040 / 0.031 |

**`the site lever is 0.000 s (1.01×)`** — the fp16 arm is nominally one percent *slower* — and every
per-site ratio sits between 0.97 and 1.12 with every per-site figure under 13 ms a forward. The
largest pair, the shared expert's `F.linear (576, 5120)`, reads 6 ms in both widths. So the header
comment's argument is now a measurement: the change is a prefill lever, and nothing about a decode
step's dense arithmetic moves with it. The context for the number is that a one-row forward is not
where a decode step's seconds are anyway — 0.040 s of the 0.33 s forward is dense arithmetic, 12%,
the rest being the eager expert call the
[live decode-graph page](deepseek_v4_1_flash_decode_graph_live.md) prices at 89% of a graphed step.

The same run also bounds a claim the parity section above cannot make on its own. On a **one-token
input**, the two widths disagree where they did not on a 4096-token one:

```
bf16 vs fp16 @chunk 1: max|d| 7.964e+00 of max|logit| 19.521 (4.08e-01 of scale)  mean|d| 1.059e+00
    argmax 7805 vs 271   <-- MOVED   top8 shared 4/8   margin top1-top2 bf16 2.077e-01 fp16 1.887e+00
```

identical at all eight arm pairs (two chunks, four ranks). The bf16 arm's top-1 is a **near-tie at a
0.207 margin** and the fp16 arm's is 1.887, so on this geometry the model is nearly indifferent and
the argmax is not a decision either width is making; the wider carrier being the more decisive one is
the direction the "difference is error removed" reading predicts. But the absolute difference is
2.3× the prefill geometry's — 7.96 against 3.40 — on a comparable logit scale, and this page reports
both rather than fitting them: a one-row forward is a different computation (one query row, no window
context to enter, the compressor emitting where a chunk fills), so the two are not extrapolations of
each other. **The consequence is a scope, not a defect**: this page claims the widths pick the same
token on a 4096-token residual stream with a 12-margin top-1, and does not claim it for a one-token
input, where the top-1 is unformed. A served token-for-token comparison is what would settle the
decode case, and the served numbers themselves are not re-derived here — the
[served gate page](deepseek_v4_1_flash_served_gate.md)'s 4.45–4.53 tok/s at a 1364-token prompt and
3.48–3.54 at 262144 are the *unpatched* tree's.

## What the tests found

Three things the test suite turned up while the constants moved, kept here because each is a fact
about the tree rather than about the patch:

- **`Engram.lookup` narrows per call.** The Engram gather casts its output to the module's width on
  every call, so the test that drives a real backbone has to pin the whole test's input width to the
  tree's `LINEAR_DTYPE` rather than pick one per tensor; a mixed-width fixture fails at the gather and
  not where the width was chosen.
- **A GEMM's row grouping is not what separates the arms.** `/tmp/gemm_m_grouping.py` compares a
  batched `F.linear` against the same rows one at a time — m=4 and m=12, k=5120, n=8192/5120/1024,
  both dtypes — and reads `max |batch - row-at-a-time|` of exactly `0.000e+00` with `torch.equal`
  true in all eight cases. Whatever an arm difference is, it is not how the rows were grouped.
- **A one-ULP difference in the Hyper Connections is present in both trees.** A miniature backbone
  with 130 hooked modules drives them in both orders under bf16 and under fp16. At bf16, exactly two
  modules differ and both are fp32 and both are ~1.8e-07 (`layers.2.attn.compressor.wkv` and
  `wgate`); the fp16 tree shows the same two plus a ladder of fp16-ULP differences whose largest,
  `7.341e-04` at logit scale 1.298, is inside one fp16 ULP (`finfo(float16).eps = 9.77e-04`). Both
  orderings repeat bit-identically at 1 and at 22 threads. This is the control behind the loader
  test's one-ULP bound: it is the smallest bound that admits a value sitting exactly on a rounding
  boundary, so a disagreement wider than it is a disagreement about the arithmetic.

## Reproducing

```
# the lever, in process, dtype switched by swapping buffers on every Parameter at that dtype
DEEPSEEK_V41_RESIDENT_EXPERTS=1 V41_TREE=/tmp/prefill_csr \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  /tmp/probe_v41_dense_dtype_abab.py --chunk 4096 --repeats 2 --arms bf16,fp16,fp16,bf16

# the same basis from the dumps -- not the probe's own printed column, which is mis-scaled
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/reduce_abab_sites.py /tmp/abab_seq.pt

# the independent cross-check, one arm a process
DEEPSEEK_V41_RESIDENT_EXPERTS=1 V41_TREE=/tmp/prefill_csr \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  /tmp/probe_v41_dense_cost.py --at 0 --chunk 4096 --repeats 1 --top 6 \
  --logits /tmp/par_bf16.pt --out /tmp/dense_cost.pt

# the patch-free control, same geometry, the tree as it ships
DEEPSEEK_V41_RESIDENT_EXPERTS=1 V41_TREE=/tmp/baseline_master \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  /tmp/probe_v41_dense_dtype_abab.py --chunk 4096 --repeats 2 --arms bf16 \
  --out /tmp/abab_master.pt --logits /tmp/abab_master_logits.pt

# the decode arm: one row, and the check that the lever is zero
DEEPSEEK_V41_RESIDENT_EXPERTS=1 V41_TREE=/tmp/prefill_csr \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  /tmp/probe_v41_dense_dtype_abab.py --chunk 1 --repeats 2 --arms bf16,fp16,fp16,bf16 \
  --out /tmp/abab_dec.pt --logits /tmp/abab_dec_logits.pt
```

`/tmp/baseline_master` is a `git archive` extraction and has no built extensions: it needs the three
`.so` files copied in from a built tree (`cuda_kernel`, `deepseek_cpu_moe_ext`,
`moe_dispatch_cuda_ext`) or it will silently fall back to host experts and fail on the first
activation with `moe_single_token_fp4_forward is not available in the built extension`.
