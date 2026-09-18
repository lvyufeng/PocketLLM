# DeepSeek-V4.1-Flash: the routed experts on the four cards

`src/models/deepseek_v4_1/device_experts.py` holds one backbone layer's routed experts as fixed
arenas on `world` cards and consumes the checkpoint's packed fp4 directly: the kernel
`moe_single_token_fp4_forward` takes `[E, N, K/2]` uint8 codes beside `[E, N, K/32]` E8M0 scales and
dequantizes inside the kernel, so no bf16 expert matrix is ever built anywhere. That is the whole
point of the path — the host's 0.122 s per expert is 99.7% the expansion, and this arithmetic is the
one the checkpoint ships with.

This page is what the path costs and what it is worth. The companion page,
[what the released checkpoint costs to run on one host](deepseek_v4_1_flash_host_run.md), is the
other half: the same model with the experts on the CPU, at 15 to 42 s per generated token.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB (SMR disk, `/dev/sda`) |
| Runtime | PyTorch resident, `src/models/deepseek_v4_1`, no native engine; `moe_single_token_fp4_forward` from the built `pocketllm_cpp` CUDA extension |
| Commit | `df3ed3d` on `feature/v41-backbone-runtime` plus the uncommitted `device_experts.py`; the ordering fix and its re-measured step are in [the launch](#the-launch-was-four-kernels-serialized-not-one-plus-copies) |
| TP4 | The same path with the dense tree cut across the four cards — one process per card under `torchrun --nproc_per_node=4`, `src/cli/generate_v41.py`. It is a different configuration of the same measurements, not a later commit of them, and [the section below](#the-dense-tree-across-the-four-cards-tp4) is what it changes |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each; expert-parallel `world=4` across all four and `world=1` on `cuda:0`, both measured |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Prompt | `The capital of France is` (5 tokens), greedy, 4 new tokens |
| Warm/cold | Both: the cost probe runs the same prompt twice in one process, and its `GiB/s` column is what separates a warm pass from a cold one |

Scripts: `/tmp/probe_device_experts.py` (the class against `expert_forward` on a captured
activation), `/tmp/probe_fp4_parity.py` (the kernel against the host expert on real activations, and
the EP4 decomposition), `/tmp/probe_stage.py` (the staging and H2D terms), `/tmp/probe_h2d.py` (the
PCIe rates), `/tmp/probe_first_tokens.py` (the top five at each of the first tokens, on either path),
`/tmp/probe_device_cost.py` (a whole step, phase by phase), and `/tmp/probe_launch_cost.py` and
`/tmp/probe_launch_split.py`, which are the two that priced and then fixed the launch ordering. All
but the first-token probe read `/tmp/v41_activations.pt`, which `/tmp/probe_capture.py` writes; that
one runs the loop itself and reads nothing. They are throwaway probes, not checked-in benchmarks; the
numbers they produced are what this page records. The TP4 sections add the probes written for the four
cards and for the row loop: `/tmp/probe_v41_tp4_e2e.py`, and `/tmp/probe_v41_tp4_pipe_matrix.py`,
which is [the row loop, serial against one row deep](#the-row-loop-runs-one-row-deep-and-it-is-worth-110-on-a-prefill).
The resident set adds two more: `/tmp/probe_v41_hot_ab.py`, which is the width sweep and the logit
comparison behind [the section below](#a-per-layer-resident-set-is-worth-27-on-a-prefill-and-it-is-the-fill-that-pays-for-it),
and `/tmp/probe_v41_resident_fill.py`, which is the only instrument here that times a per-*layer*
cost and the only one that reads the process's own `smaps_rollup`. The pool is the same probe under
`--pool-rows`, driven three times: `/tmp/run_poolfix.sh` is the six-sitting A-B-C-C-B-A behind its
first table, `/tmp/run_poolwidth.sh` the eight-sitting width sweep behind the second, and
`/tmp/run_poolshort.sh` the five-sitting A-B-C-D-A that shortens the pass instead of moving the width
— the last of which records one sampled token that disagrees with the other twenty-nine, and
[the subsection on the probe's sampler](#the-probes-tokens-are-the-samplers-draw-and-the-logit-column-is-the-parity-check)
is what that is. The width pair is re-run by
`/tmp/run_quiet_width.sh`, which is not a probe but the driver of the A-B-A-B: it runs
`probe_v41_hot_ab.py` at 148 and 192 twice in one sitting, waits on `WAIT_FOR=<pid>` so the four
cards are free, and echoes `/proc/loadavg` at every leg — that load trace is what says whether a
sitting's numbers are worth comparing to the last one's, and the answer for the sittings recorded
here is that they are worth comparing only to themselves. `/tmp/probe_v41_resident.py` is the bank's
acceptance instrument and the only one here that reads `/proc/self/io`'s `read_bytes`: it brackets the
load, each prefill and each decode with the kernel's own block-device counter, which is the one figure
a wall clock cannot produce. The dense tree is `/tmp/probe_dense_tree.py`, per op and per layer and at five thread counts — and with `--tree cuda`, under a `torchrun` driver (`/tmp/run_quiet_dtree.sh`), the
same per-op table with the tree on the four cards, which is the sitting where `hc_mixes` stops being a
fallback and becomes the thing the fusion was measured for. The launcher's default adds the two
drivers it was accepted on: `/tmp/v41_default_accept.sh`, four legs of one configuration a process
with the default and its off state alternating, and `/tmp/v41_default_accept2.sh`, which is the third
prefill leg that splits the pool's worth from the batch's and then the decode A-B-A-B at the width
that ships, with `/tmp/probe_v41_hot_ab.py --compare` pairing the two decode legs' top-32 logits at
`|dlogit| 0.000e+00`. Both echo `/proc/loadavg` at every leg, for the reason `run_quiet_width.sh`
does. `/tmp/probe_hc_split_card.py` prices that
one op against the loop it replaced on an idle card, with no checkpoint and no second rank in the
process. The same probe under `--launches` then counts what the step launches, under a CUPTI window
(`/tmp/run_quiet_dtree_launches.sh`); that report is taken twice, to `/tmp/dt_cuda_launches.out` and
`/tmp/dt_cuda_launches2.out`, because only the pair says which of its columns is a count and which is
a sample. The last one closes the bank's own open number: `/tmp/probe_engram_bank.py` attaches the
filled segment read-only and prices a 512-token prefill's 12,288 gathers a table — the call, the
scatter under it and the dequant over it — against the per-row control the host page took, on a host
with no card in the process at all.

## The split

Per row, the layer's 6 routed experts are sorted by global id and dealt round-robin, so card `c` owns
sorted positions `c` and `c + world`: **2, 2, 1, 1** over four cards, from the real routing of layer
0's captured prefill row, card 0 took experts 128 and 251, card 1 took 137 and 277, card 2 took 155
and card 3 took 206. The split is a property of the routing and not of the order the gate emitted it
in, which is what makes two runs that route to the same six experts stage the same bytes into the
same rows.

Each card's arena is a fixed `ceil(topk / world) = 2` rows, so nothing resizes between steps and
nothing is allocated per token. The pinned arena holds every card's rows laid end to end — one
allocation rather than one per card — **143.4 MiB per side** at `world=4` against **107.6 MiB** at
`world=1`, both with two buffers so a row can stage while the DMA that read the previous one is still
in flight. The four-card arena is the *larger* of the two because 6 does not divide by 4: two of its
eight rows are empty in every token. That is 0.3 GiB of pinned RAM for the four-card path, which is
not a reason to rebalance it.

**The cards never talk to each other.** Each holds its own arena, is handed the same `[1, 5120]`
activation, and returns `[1, 5120]` fp32; the host sums the `world` partials. That is 20 KiB back per
card per layer, 3.2 MB per token across four cards, and it is why there is no NCCL, no all-to-all and
no collective here to debug. It also makes `world=1` the single-card configuration rather than a
second implementation, which is what Verification item 3 in the plan asked for: the same code, one
flag.

**The dense tree is on the host in this configuration and on the cards in the next one.** It is
16.79 GiB, it worked first, and moving it is worth 0.4–0.6 s/token on its own; doing both at once
would have put two independent sources of divergence inside one debugging session. It has since been
cut across the four cards — a different launcher and a different process shape,
`src/cli/generate_v41.py` under `torchrun --nproc_per_node=4` — and [that section](#the-dense-tree-across-the-four-cards-tp4)
has the numbers. Everything above it and everything under *What a step costs* is the
tree-on-host configuration.

## Two conventions that had to be settled before the first run

Both looked like blockers first, and both came out in the kernel's favour. They are recorded here
because a page that only reports the numbers leaves the next reader to re-derive them.

**The scale.** `fp4_block_scale` (`cuda_kernel_impl.cu:2388`) is
`__int_as_float(max(0, byte - 1) << 23)`. `__int_as_float` places its argument in the *exponent
field*, so the value is `2**(byte - 1 - 127)` and not `2**(byte - 1)`; with the LUT's doubled e2m1
levels that is `level * 2**(byte - 127)`, bit for bit what torch's `float8_e8m0fnu` reads on the same
byte and what `cpp_engine/backends/cuda/kernels/fp4_ops.cu:243` computes as `exp2f(code - 127)`. The
`- 1` absorbs the LUT's `* 2`. `tests/test_moe_single_token_fp4.py` draws scales 124..130 against a
reference that reads them as `2**(byte - 127)` and reproduces the kernel to 2.9e-5 relative. **An
earlier reading of this file recorded the opposite and was a misreading of the shift**: there is no
kernel bug here and no rebasing anywhere in the path.

**The layout.** The release stores an expert's `w1.weight` as `(2304, 2560)` `I8` beside `w1.scale`
`(2304, 160)` `F8_E8M0` — `[N, K/2]` beside `[N, K/32]` with `N = inter_dim` and `K = dim`, which is
the op's ABI verbatim — and `w2` as `(5120, 1152)` beside `(5120, 72)` as `[dim, inter/2]`. So an
arena row is the checkpoint's own tensor: no transpose, no per-element scale expansion, no rebasing.
`DeviceRoutedExperts._check_shapes` refuses the transposed release rather than reading it, because
`w1` and `w2` swapped would still give every arena a plausible shape and every kernel a plausible
answer.

## Correctness

Three measurements, from the outside in. The activations are real — `/tmp/probe_capture.py` captures
a row at each layer while the actual checkpoint runs — because the kernel quantizes a row with a
single scale, so what the precision costs depends on that row's dynamic range and a Gaussian of the
wrong width would answer a different question.

| Comparison | Max abs | Relative | Cosine | Argmax |
| --- | ---: | ---: | ---: | --- |
| kernel vs `expert_forward`, layer 20 row 2 (worst of 69 rows) | 0.08572, **6.715% of the output scale** | — | 0.997356 | differs on **7 of 69** rows |
| kernel vs `expert_forward`, typical row | — | 2–5% of scale | 0.997–0.9998 | — |
| four arenas summed vs one call over all six experts | 5.960e-08 | 5.607e-08 | — | same |
| `world=4` vs `world=1`, whole class, same activation | 5.960e-08 | — | — | same |
| `world=4` vs the host path | — | 1.5–2.4% of scale | 0.9993–0.9997 | — |

The first row is the whole cost of the device path and it is the arithmetic the checkpoint ships
with: int8 activations × fp4 weights against the host's bf16 × bf16, two different precision classes
by construction. 6.7% of the output scale on the worst of 69 rows, with the argmax moving on 7 of
them, is what that costs at the layer level; it does not accumulate into a wrong token (below).

The next two rows are the ones that say the *implementation* is exact. `5.960e-08` is fp32
association order across four partial sums and nothing else — the number recurs because both
comparisons are summing the same four partials in a different order — and it is 2^-24 of a unit-scale
result. So the split, the deal, the staging, the arena row assignment and the weight permutation are
all accounted for: what is left between this path and the host is the kernel's own precision.

And the end of it, the checked-in generation loop, greedy, `--max-new-tokens 4`, both worlds, the
verbatim tail of each run:

```text
$ ... -m src.models.deepseek_v4_1.generate --checkpoint /mnt/data3/DeepSeek-V4.1-Flash \
      --prompt "The capital of France is" --max-new-tokens 4 --expert-device cuda --expert-world 4
loaded in 73.7 s
routed experts: DeviceRoutedExperts on 40 layers, world 4
prompt 5 tokens: ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
3 tokens in 7.2 s (2.41 s/token), stopped on eos
The capital of France is Paris.<｜end▁of▁sentence｜>

$ ... --expert-device cuda --expert-world 1
loaded in 72.2 s
routed experts: DeviceRoutedExperts on 40 layers, world 1
prompt 5 tokens: ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
3 tokens in 8.6 s (2.87 s/token), stopped on eos
The capital of France is Paris.<｜end▁of▁sentence｜>
```

`--max-new-tokens 4` and 3 tokens, because token 3 is the EOS the model actually chose. The `routed
experts:` line is the flag's own report of what it built, and it exists because "it silently fell
back to the host" is the failure this flag has: it walks the layers and prints the class it found
plus the world read off the object rather than off the argument.

The two worlds agree with each other token for token. The host path does not agree with them at
token 3, so that token gets its own probe rather than its own sentence: `/tmp/probe_first_tokens.py`
runs the same loop, same prefill, same greedy pick, with the distribution printed at every step.

| Path | Token 1 | Token 2 | Token 3 |
| --- | --- | --- | --- |
| `world=4`, on the cards | `' Paris'` **22.295** | `'.'` **22.933** | EOS **17.953** against `' The'` 17.452 |
| the host, same loop | `' Paris'` 20.605 | `'.'` 20.873 | `' The'` **17.513** against EOS 17.051 |

**The third token is a half-logit knife edge on both paths, and they fall on opposite sides of it.**
The device's margin is 0.501 logits and the host's is 0.462, on a distribution whose leader sits at 18
to 23 — this is not a token either path has an opinion about, and the difference that decides it is
the fp4 kernel's: the first two tokens score 1.690 and 2.060 logits higher on the cards than on the
host, which is the 6.7%-of-scale layer difference above showing up where it can flip an argmax.

**And the host path is not a single answer at that token either.** `probe_accept.py`'s stepwise decode
of the same prompt — the host page's recorded `' Paris.<｜end▁of▁sentence｜>'` — emits EOS there from
the same host weights, and the host page measures a prefill against a stepwise decode of this prompt
at 6.0748 max abs logits, two orders of magnitude above the 1.69 the cards move it. So the side of a
0.5-logit tie this token lands on was never a property of the host weights; both device worlds land on
the side the host's own stepwise decode lands on.

That is the acceptance bar the plan set, met: **all three configurations agree on the first two
tokens and each continues coherently** — the cards end the sentence at the period, the host builds
`' The Eiffel'` out of it — and the one token they disagree about is a near-tie on a distribution
that does not distinguish its two candidates. Both device worlds and both device runs put EOS on the
same token, and neither is a wrong answer at it.

**Every number in the table above is a comparison inside one run, and that is what makes it hold.**
The two `5.960e-08` rows are one process, one activation, the same six experts summed four ways and
one way; the kernel rows are one process on a captured row. Nothing there is a comparison *between*
two runs. That used to matter a great deal: **the tree was not reproducible from one run to the
next**, and two sittings of one configuration kept the greedy tokens and 30 of the top 32 ids while
moving 30 of the 32 out of position, which is why every configuration below was compared on its
tokens and its tok/s and not on a logit column. It does not matter any more, because the cause was
found and it was not a tolerance: **one missing `__syncthreads()` in the six sparse-attention
kernels**, and with it in place two runs of one configuration are **32 of 32 in position at `|dlogit|
0.000e+00`** — as is a pair that differs by a whole configuration. Being able to say that is a fact
about the code rather than a choice of tolerance, and it restores the columns as a measurement, so
the configurations below are still separated on their tok/s — not because the logits are noisy, but
because what separates them is a staging cost and the logits are exactly where it does not show. The
evidence chain, and what the old drifting pairs measured before it existed, are in *What this does
not do yet*.

## What a step costs

`/tmp/probe_device_cost.py` instruments `DeviceRoutedExperts` itself and runs the prompt through the
backbone twice in one process. Each row is one MoE row: a decode step is 40 of them (one per layer, 6
experts each) and a 5-token prefill is 200.

These two tables are the run that established the shape of the step, and they predate the ordering
fix in [the launch](#the-launch-was-four-kernels-serialized-not-one-plus-copies) — read their `Launch`
column as the 0.27 s that fix took to 0.15 s. They are kept as measured rather than back-edited; the
re-measured step is in that section.

`world=4`, routed experts on `cuda:0..3`, 2 rows per card:

| Pass | Step | Wall | Stage | Upload | Launch | Other | GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 (prefill, 200 rows) | 5.00 s | 1.94 s | 0.29 s | 1.34 s | 1.42 s | 10.83 |
| 1 | 1 | 1.26 s | 0.34 s | 0.03 s | 0.27 s | 0.62 s | 12.46 |
| 1 | 2 | 1.28 s | 0.33 s | 0.04 s | 0.27 s | 0.65 s | 12.79 |
| 2 | 0 (prefill, 200 rows) | 4.50 s | 1.89 s | 0.17 s | 1.34 s | 1.11 s | 11.13 |
| 2 | 1 | 1.42 s | 0.34 s | 0.03 s | 0.27 s | 0.78 s | 12.32 |
| 2 | 2 | 1.34 s | 0.33 s | 0.03 s | 0.27 s | 0.71 s | 12.71 |

`world=1`, routed experts on `cuda:0`, 6 rows per card:

| Pass | Step | Wall | Stage | Upload | Launch | Other | GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 (prefill, 200 rows) | 13.74 s | 9.20 s | 0.10 s | 2.38 s | 2.07 s | 2.28 |
| 1 | 1 | 2.36 s | 0.93 s | 0.02 s | 0.48 s | 0.94 s | 4.53 |
| 1 | 2 | 2.22 s | 1.02 s | 0.02 s | 0.48 s | 0.70 s | 4.11 |
| 2 | 0 (prefill, 200 rows) | 5.52 s | 1.88 s | 0.06 s | 2.31 s | 1.27 s | 11.18 |
| 2 | 1 | 1.40 s | 0.32 s | 0.01 s | 0.46 s | 0.60 s | 12.94 |
| 2 | 2 | 1.39 s | 0.33 s | 0.01 s | 0.46 s | 0.59 s | 12.91 |

**Before the ordering fix a decode step was 1.23–1.42 s at `world=4` across three runs of two probes,
and 1.40 s at `world=1`, warm**; a 5-token prefill is 4.50–5.00 s and 5.52 s. After it the step is
**1.06–1.14 s** at `world=4`, which is [measured below](#the-launch-was-four-kernels-serialized-not-one-plus-copies).
Every one of those runs returns `'The capital of France is Paris.<｜end▁of▁sentence｜>'` with tokens
`[11111, 16, 1]`. Against the host path's 15.3 s for a warm decode step and 14.5–29 s of expert
expansion alone, the step is an order of magnitude and the expansion is gone entirely.

The second probe that measures a step — `/tmp/probe_token_cost.py --expert-device cuda
--expert-world 4`, the same file the host page uses for the host path — attributes it in the
coarser pair the two paths share: **1.23 s of step, 0.73 s of it inside `DeviceRoutedExperts` and
0.51 s of it the dense tree, the head and the layer glue**. That probe also reports zero misses and
then divides by them, because its miss counters are `CheckpointRoutedExperts`'s and this path has no
window at all; the device path's per-step cost is the same whether a step repeats an expert or not,
which is the second thing it says. Both of its figures predate the ordering fix and both keep their
shape after it: the class's 0.73 s falls to **0.48 s** (0.30 staging + 0.15 launch + 0.03 upload) and
the 0.51 s outside it is unchanged at 0.55–0.67 s, which sums to the 1.06–1.14 s step measured below.

**The checked-in loop's own numbers are whole-request and do not decompose**, which is the one thing
to read them for and not for anything else: its timer wraps the entire `generate()` call — the 5-token
prefill plus three decode steps — and divides by the three new tokens, so the `2.41 s/token` it
prints is not a step and this page does not quote it as one. What it does give is a second,
independent instrument on the difference between the two worlds: **8.6 s against 7.2 s is 1.4 s**,
while the probes' own world-1-minus-world-4 is 0.5–1.0 s of prefill (5.52 against 4.50–5.00) plus
nothing to 0.5 s over three decode steps (1.40 against 1.23–1.42) — 0.5 s to 1.5 s. Same sign, same
size, from code that shares nothing with the probes. On the absolute the loop is 1.0–2.1 s under what
the probes' per-step figures sum to; the probes are the instrumented side of that pair, so the step
figure this page quotes is theirs.

Whole-request, the same command with the flag unset is the comparison the flag exists to make, and it
is measured on the same prompt in the same session: **28.00–42.37 s per new token on the host against
2.41–2.87 s on the cards**, five runs, prefill included at both ends. The host figures are `168.0 s
for 6 tokens`, `142.0 s for 4` and the companion page's `169.5 s` and `124.6 s`; the flag-unset run in
this session is the second of those, so the host path is unchanged by this work and still the default.

The `GiB/s` column is the staging rate and it is what makes the page-cache state readable off the
run. The `world=1` pass 1 is the only column that caught a cold cache — 2.28 GiB/s at the prefill and
4.1–4.5 for its decode steps, against 11.2–12.9 in the same run's pass 2 — so a cold first token
surfaces as a staging time and not as anything else. The `world=4` run found the same pages resident
in both passes, which earlier device probes had made them. **The controlled comparison is pass 1
against pass 2 within a run**, and where the two runs disagree about what was resident the `GiB/s`
column says so rather than averaging it away.

### The four cards are worth 0.1–0.2 s, and that is the one prediction the plan got wrong

The plan projected a token of 0.6–1.0 s on four cards against 1.2–2.0 s on one, with the H2D falling
from 0.51 s to 0.13 s as the four links aggregate. The aggregation is real — measured, one link moves
10.47 GiB/s and four move **38.56 GiB/s** together, so the 4.20 GiB of a step is 0.11 s of transfer
against 0.51 s — but it is not what the step spends its time on, and the measured split says why:

| Term | `world=1` | `world=4` |
| --- | ---: | ---: |
| staging, host, per decode step | 0.33 s | 0.33 s |
| the copy chains, enqueued | 0.01 s | 0.03–0.04 s |
| `_launch`, four kernels plus the H2D they wait on | 0.46 s | 0.27 s → **0.15 s** |
| **wall** | **1.40 s** | **1.23–1.42 s → 1.06–1.14 s** |

**Staging is identical on both and it is host work.** It does not care how many cards read the bytes,
so what the split buys is the kernels and the partials they hand back — 0.19 s of a step before the
ordering fix and 0.31 s of the re-measured 1.06 s step after it — and nothing at all of the 0.33 s that
is now the largest single term. The plan's table had H2D on the critical path and staging as the
unmeasured question; the measurement says the reverse. That is the honest reading of four-against-one
and it is why the follow-on that would matter keeps the packed rows on the card rather than staging
them again — the wider arena, below.

### The launch was four kernels serialized, not one plus copies

Everything from here to the pipeline below was measured while the row's card side was a single method
named `_launch`, and that is the name this section keeps. It is now two methods — `_issue`, which is
all of the above and returns without waiting, and `_drain`, which is the waits and the host sum — and
the split exists so that a row's card time and the next row's staging can be in flight together. The
numbers this section records are the two halves of that, and none of them moved when it was split; see
[the pipeline below](#the-row-loop-runs-one-row-deep-and-it-is-worth-110-on-a-prefill).

The phase this page named as "the obvious next thing to instrument" was the 0.27 s of `launch`: 160
card-calls of 1.7 ms each against the 0.83 ms an isolated call of that shape costs on one device. The
guess was that the per-call allocation and the pageable, synchronous copies were the missing
millisecond. `/tmp/probe_launch_split.py` priced it by measuring one whole row of four cards three
ways, one change apart, on layer 0's real activation and real arenas:

| A row of four cards | Per row | Per token, 40 layers |
| --- | ---: | ---: |
| as it was: allocate per call, pageable transfers, D2H drained inside the card loop | 3,633.4 µs | 145.3 ms |
| preallocated, pinned, still one card drained at a time | 3,452.5 µs | 138.1 ms |
| preallocated, every card issued before any card is drained | **1,009.8 µs** | 40.4 ms |

**The allocation and the pageable transfers were worth 1.05x. The serialization was worth 3.42x, and
3.60x end to end.** `_launch` ran a *pageable, synchronous* D2H of card 0's partial **inside** the card
loop, and a pageable D2H cannot return until the kernel that produced it has finished — so the host
could not launch card 1's kernel until card 0 was done, and the four cards were four kernels added up
rather than four kernels in flight. The fix is ordering, and it fits inside the arenas and the pinned
buffers the class already had: no arena growth, no wider arena, no multi-token kernel.

`_launch` is now issue-then-drain. A per-instance `_row_scratch` holds one pinned activation, one
pinned weight vector, per-card device copies of both, per-card device index vectors, and per-card
pinned results; the row's route weights are gathered once with `index_select` into card order so each
card's slice is contiguous; every card's H2D and kernel are issued first, with no drain anywhere in
that loop; then each card's D2H into pinned memory, one `torch.cuda.Event` recorded per card and
waited on once each. One blocking call per row instead of four blocking calls interleaved with the
launches. The result is `clone()`d, so a caller that keeps it is not handed a view of a reused buffer.

It is also **bit-neutral**: `probe_launch_split.py`'s part 3 compares the pipelined row against the
row it replaced and gets `exact True`, `max|d| 0.000e+00` for both the preallocated and the pipelined
versions. The class-level parity numbers are unchanged to the last digit — `world=4` 1.532% / 2.146% /
1.913% of scale at cosine 0.999267 / 0.999327 / 0.999700, argmax correct on all three rows, and
`world=4` against `world=1` still never above 5.960e-08.

Re-measured in a real step, warm (`/tmp/probe_launch_cost.py --world 4`, which also wraps
`_take_buffer` — no other probe had):

| Phase | Step 1 | Step 2 |
| --- | ---: | ---: |
| wall | 1.06 s | 1.14 s |
| `_take_buffer` | **0.00 s** | **0.00 s** |
| `_stage` | 0.30 s | 0.29 s |
| `_upload` | 0.03 s | 0.04 s |
| `_launch` | **0.16 s** | **0.15 s** |
| everything else | 0.57 s | 0.66 s |

Both passes decode `'The capital of France is Paris.<｜end▁of▁sentence｜>'`, and the 0.16 s is the
0.27 s the phase measured before. In the class's own isolation the same change is `world=4`
17.4 ms → **14.8 ms** per row-layer, 0.70 s → **0.59 s** per token, and `world=1` now measures
**21.9 ms**, 0.88 s per token. The one-card configuration moves too and by more of its own total,
which is the tell that the change is not about the four cards: its `launch` phase was the 0.46–0.48 s
of a single card paying a pageable synchronous D2H per row four times over, and that cost is the same
whether one card is behind it or four.

**Two things that had to be checked rather than assumed.** First, the hypothesis was wrong in an
informative way: pinned-and-reused, the change this work was originally scoped as, is 1.05x of a row
and does not reach the target on its own. Second, a real step's `_take_buffer` is **0.00 s** — on
every decode step and on the cold prefill too — so the wait for the previous upload's DMA is already
satisfied and this page's earlier suspicion that part of the 0.33 s staging figure was really PCIe is
retired, not confirmed. With two pinned buffers the buffer being staged was read by a DMA issued a row
and a launch earlier; a third buffer would buy nothing.

**A measurement hazard, stated because it is in the numbers above.** The "before" run of
`probe_launch_cost.py` caught a cold page cache — its pass 1 prefill measured 185.92 s, 167.23 s of it
staging at 0.13 GiB/s, against the 3.92 s and 11.94 GiB/s of the "after" run where the pages were
already resident. So `stage` is **not** controlled between the two runs and its 0.30 s comes from the
after run alone; `launch` and `take` are the controlled terms, and they are the ones the change is
about. `_launch` decomposes the same way: of its 0.15 s, one kernel's worth of arithmetic is the
40.4 ms the isolated row costs, and the remaining **~0.11 s** is the H2D the kernels wait on — a card's
arena copy is ordered behind that card's previous kernel, so the transfer, unlike the issue, is on the
device's critical path and not the host's.

### Where the 1.06 s goes

This section used to be an inference — three phases of a 1.3 s step with a tenth left unattributed,
and a closing admission that "the waits inside `_take_buffer`" were the obvious next thing to
instrument. `/tmp/probe_launch_cost.py` is that instrument, and the answer is measured:

| Term | Per decode step | Per token |
| --- | ---: | ---: |
| `_stage`, host page cache → pinned, 4.20 GiB | 0.30 s | 0.30 s |
| `_launch`, four kernels plus the H2D they wait on | 0.15 s | 0.15 s |
| `_upload`, the copy chains enqueued | 0.03 s | 0.03 s |
| `_take_buffer`, the wait for the previous DMA | **0.00 s** | **0.00 s** |
| the host dense tree, the gate, the shared experts, the head, the layer glue | 0.55–0.67 s | 0.55–0.67 s |
| **wall** | **1.06–1.14 s** | |

Three of those need their numbers held apart from the isolated ones, and each is now a measurement
rather than a caveat. `_take_buffer` is zero on every decode step *and* on the cold prefill, so the
wait for the previous upload's DMA is always already satisfied; a third pinned buffer would buy
nothing. `_upload`'s 0.03 s is issue and not transfer, which the zero above confirms rather than
assumes — the transfer is not hiding in the buffer handshake. And `_launch`'s 0.15 s is the 1,009.8 µs
row measured above against the 0.83 ms an isolated call of that shape costs, so roughly a quarter of
it is arithmetic and the rest is the pinned H2D each kernel's arena copy is ordered behind.

What that leaves is the honest headline of this configuration: **the host's own dense stack is the
largest single term of the step**, 0.55–0.67 s against the 0.30 s of staging, and it is the same
0.51 s the earlier coarse probe measured from the other direction. It has since been moved — the tree
is cut across the four cards and the step is
[722–747 ms rather than 1.06–1.14 s](#the-dense-tree-across-the-four-cards-tp4) — which leaves the
staging in this table as the largest term of what is left.

On one device and serialized, the same arithmetic measures: **2,077.3 µs per layer** for one call
over all six experts against **3,302.6 µs** for four calls of 2/2/1/1 — 83.1 ms against 132.1 ms per
token. That was the upper bound the four cards were supposed to beat by overlapping, and it turned out
they were not overlapping at all; the 1,009.8 µs row above is what they cost once they do, which is
below even the single-call figure because the four device chains run concurrently.

## The staging rate, and the number that was wrong

The plan's gate was this term, and the figure it had to retire was 0.33 GiB/s, from a probe that
timed `ckpt.reader.load(k).pin_memory()` per tensor — a fresh `cudaHostAlloc` and copy, 1,920 times
per token, inside the timed region. A device path allocates its arena once and `copy_`s into it, and
that is what this class does. Measured, warm:

| Operation | Rate |
| --- | ---: |
| `copy_` into a pre-allocated pinned arena — **what the path does** | **11.49 GiB/s** |
| `copy_` between two pinned buffers, the ceiling | 13.77 GiB/s |
| `pin_memory()` on a freshly loaded tensor | 12.45 GiB/s |
| `reader.load` alone, allocate and clone | 8.10 GiB/s |
| H2D, one card | 10.47 GiB/s |
| H2D, four cards at once | 38.56 GiB/s |
| the same staging across four threads and four arenas | **regresses**: 733 ms against 365 ms |

**The staging rate in a real step is 12.3–14.0 GiB/s**, which is the isolated `copy_` and not the
allocation storm: a decode step stages 40 rows × 6 experts × 17.9 MiB = 4.20 GiB in 0.30–0.34 s
across the runs that measured it. Prefill is the same rate over 200 rows — 21.0 GiB in 1.89–1.94 s,
10.8–11.1 GiB/s. The 0.33 GiB/s figure measured an allocation pattern no device path uses, and an
earlier reading of the host page drew a conclusion from it; **the staging model in the plan was right
and no correction was needed.**

Threading was measured and regresses — 733 ms against 365 ms — because a 3 MiB `copy_` is already at
what one core pulls out of the page cache. The loop here is deliberately single-threaded; the
parallelism that pays is the four links, not four threads.

The staging is a read of `/mnt/data3`, an SMR disk, so the floor under a cold first token is the
disk: one scattered expert row is **1308.0 ms** cold against 11.2 ms warm.

## The dense tree across the four cards (TP4)

Everything above this section is the tree on the host. The configuration the round was for is the tree
cut across the cards — one process per card, the experts dealt to the same cards — and this is what it
measures.

The launcher is `src/cli/generate_v41.py`:

```bash
torchrun --nproc_per_node=4 -m src.cli.generate_v41 \
    --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" \
    --max-new-tokens 8 --threads 22
```

The split is the one settled in advance: `wq_b` by head, `wo_a` by `o_groups` so a rank owning 16 of
64 heads owns whole groups and needs no collective of its own, `wo_b` row-parallel, the indexer's 32
heads to 8 a rank, and three all-reduces a layer — the attention output, the ffn output, and a small
`[1, 1, t]` for the indexer, whose score sums over heads. `wq_a` and `wkv` stay whole, for the
reasons [the split](#the-split) gives. `DeviceRoutedExperts` is unchanged in shape and changed in
owner: rank `r` holds experts `r`, `r + 4`, … by global id and returns its own partial, which the ffn
all-reduce completes, with the host's summation moved into the collective.

**The step.** `/tmp/probe_v41_tp4_e2e.py` runs the phases `/tmp/probe_v41_e2e.py` clocks with the tree
on the cards, so the two tables read line for line. Uninstrumented, 22 threads, warm:

| Context | Prefill | Prefill tok/s | Decode | in `DeviceRoutedExperts` | in the tree | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 tokens | 3.05 s | 2.6 | **746.8 ms** | 444.3 ms | 302.5 ms | 1.34 |
| 128 tokens | 42.86 s | 3.0 | **722.0 ms** | 408.8 ms | 313.2 ms | 1.39 |

against the **1.06–1.14 s a step** [the section above](#where-the-106-s-goes) records with the tree on
the host. The tree and the glue around it went from 550–670 ms to **300–310 ms**, while the class's
half is 409–444 ms here against the 0.48 s (0.30 staging, 0.03 upload, 0.15 launch) it was inside that
1.06–1.14 s step — the same staging and the same launch, with the host's summation replaced by the
ffn's collective. The tree's factor is **1.9× and not 4×**, which is what `probe_tp4_block.py` said
before any of this ran: that cost is Python dispatch and small GEMVs, and a rank running a quarter of
the launches on a quarter of the weights still runs a quarter of the launches at the same price each.
Both lengths measure the same step — the 8-token row is the run's first and carries the per-process
first-frame cost, which is why it is the larger of the two — because the experts are re-staged every
row whatever is in the KV cache.

**The same step, op by op.** `/tmp/probe_dense_tree.py --tree cuda` is the per-op instrument the host
section below is built on, with the tree on each rank's own card — one process a card, the experts dealt
to the same four, so it is the configuration above and not a variant of it. It clocks the step at
**688.9 ms** — 22 threads, the resident bank on, `/tmp/dt_cuda.out` — the class at **414.45 ms** against
the e2e probe's 444.3 and 408.8, and the named dense calls at **191.32 ms**:

| card, decode | ms/step | | card, decode | ms/step |
| --- | ---: | --- | --- | ---: |
| `routed` (the class) | 414.45 | | `hc_post` + `hc_pre` | 17.15 |
| `hc_mixes` | 27.72 | | the four norms | 25.73 |
| `csa2` | 23.64 | | `engram` | 6.78 |
| `kv_quant` | 21.42 | | 5 GEMVs + `compressor` | 11.84 |
| `gate` | 15.30 | | `head` + `norm` + `embed` | 0.29 |
| `shared_experts` | 14.19 | | | |
| `rope` | 13.64 | | **dense, all of it** | **191.32** |
| `indexer` | 13.62 | | **whole step** | **688.9** |

The 191.32 is the sum of the named calls, so it is narrower than the 302–313 ms "in the tree" column
above: that one is a phase clock over the whole block, and it carries the block's own glue and its three
collectives, which the named-call wrappers here do not own. The two agree on the step — 688.9 against
746.8 and 722.0, where the 8-token row is a run's first frame — and on the class to 1.3%.

**Where it goes.** With `_stage`, `_upload` and `_launch` wrapped as well (`--lengths 8,128`):

| | 8 tokens | 128 tokens |
| --- | ---: | ---: |
| `_stage` | 224.7 ms | 256.0 ms |
| `_upload` | 16.0 ms | 17.8 ms |
| `_launch` | 192.9 ms | 177.6 ms |
| unattributed inside the class | 59.3 ms | 39.7 ms |
| whole step | 804.5 ms | 815.9 ms |

Instrumenting costs 60–90 ms a step (746.8 uninstrumented against 804.5 instrumented, same probe, same
prompt), so the unqualified numbers are the table above and this one is read for the split rather than
the total: **the staging is the largest term inside the class** — 225–256 ms of the 805 ms step,
ahead of the launch's 178–193 — and the class is the larger half of the step, 493 ms against the 312
the tree and the glue around it cost. Prefill is unchanged and still the problem — 3.0 tok/s at 128
tokens, 41.85 of its 42.86 s inside the class — because a prefill of `n` rows stages 4.2 GiB `n`
times, and 128 rows is what it says it is.

**The four ranks agree on the token by construction rather than by broadcast.** Each collective's
result is identical on all four, the head and the embedding are replicated rather than cut, and a ring
all-reduce sums in one order everywhere, so all four compute the same logits; greedy decoding is an
argmax over that and needs no message to enforce what is already true. Every run above prints
`The capital of France is Paris.<｜end▁of▁sentence｜>` on all four ranks, 3 tokens, `stopped on eos`,
and the same text comes out of the `world=1` host run. That is a statement about the four ranks within
one run and not about two runs of the same configuration; [the caveats](#what-this-does-not-do-yet)
carry the second question.

### The tree's largest launch source was one Python loop, and cutting the tree could not cut it

The 300–310 ms above is with `hc_split_sinkhorn` fused, and part of the reason the tree's factor is 1.9×
rather than 4× is that this op was never the tree's to cut. `Block.hc_mixes` calls it twice a layer, it
normalizes a `[b, s, 4, 4]` matrix, and it did so through a Python loop of `hc_sinkhorn_iters - 1`
iterations — 19 of them at V4.1's `hc_sinkhorn_iters` of 20 — of up to six elementwise ops. That is
about 114 ATen dispatches a call and **~9,100 launches a step for arithmetic that fits in a register
file**. `hc_mixes` is not one of the tensors the split cuts: it is glue, the gate and the `hc_*`
coefficients replicate on every rank, and a quarter of the tree on a quarter of the cards is nothing to
an op whose cost is its dispatch count rather than its bytes — 0.6 GB/s on the line this round was
planned against, the least traffic in the tree and the most launches.

Measured on the card, one decode row: **1.931 ms a call, 154.45 ms over a step's 80 calls**, at the same
price for batch 1 and batch 512 — which is the evidence that the number is dispatch and not arithmetic.
The fix is one kernel that runs all `ITERS` normalizations in registers, 32 rows a program, placed
behind `_resolve_impl` with the loop kept as `hc_split_sinkhorn_torch`; it falls back to that loop when
`hc_mult` is not a power of two, when the input is not CUDA, or when triton is unavailable — so a host
tensor still takes the loop and the host-tree configuration above keeps this cost by design. Agreement
with the loop over batch 1/8/128/512/2048 against iterations 1/5/20 is **2.98e-07 worst relative
error**, and the speedup runs from 3.8× at one iteration to **25.4× at the twenty this model uses**: in
the step, the loop's 154.45 ms becomes the kernel's **7.19**. Those are the two endpoints of that
bench, and the card sitting below measures each of them again on this branch.

**The dense-tree sitting the rest of this section is built on cannot see any of that, because it is the
host tree.** The kernel's third fallback is the decisive one here: `Block.hc_mixes` on a CPU block
hands it a CPU tensor, `flat.is_cuda` is false, and the op takes the loop — so `hc_mixes` is 86.13
ms/step in that table (`/tmp/probe_dense_tree.out6`, 1152.9 ms/step, 1.077 ms a call) and 88.93 in the
sitting before it, not because the kernel is missing but because nothing on the host path can call it.
Nothing in that table changes when the fusion lands. What the fusion changes is what the term costs
**once the block is on a card**, and that is the opposite of a detail in this round: the same 80 calls
go from **1.077 ms each to 1.931** the moment the input is a CUDA tensor, so a move that is otherwise a
wash would have *added* about 68 ms a step to this term alone before the kernel and removes about 79
with it. The move needs the fusion to be affordable; the fusion needs the move to be reachable.

**The card sitting has now been taken, and the prediction holds.** `/tmp/probe_dense_tree.py --tree
cuda` builds the tree on each rank's own card under `torchrun` — one process a card, the experts dealt
to the same four — which is the configuration the paragraph above is about and the only one in which
`Block.hc_mixes` is handed a CUDA tensor. Its per-op table puts `hc_mixes` at **27.72 ms/step over 80
calls**, 0.347 ms a call. That row times the whole method, and the method is more than the op, so
`/tmp/probe_hc_split_card.py` prices the pieces on an idle card with no model in the process, at the
shape this model runs — 1 row, `hc_mult` 4, 24 flattened, 20 iterations:

| card, 1 row | µs/call | ms/step at 80 calls |
| --- | ---: | ---: |
| the fused kernel alone | 98.6 | **7.89** |
| the loop it replaced, same card, same shape | 1476.1 | 118.09 |
| the rest of the method: `flatten`+`float`, `square`+`mean`, `rsqrt`, `F.linear`, `mul` | 151.2 | 12.10 |
| the whole method, fused | 249.8 | 19.99 |
| the whole method, loop | 1623.1 | 129.85 |

**7.19 predicted against 7.89 measured**, and flat across the batch, which is the part only the card
can say: 1 row and 512 rows cost the fused kernel 98.6 and 98.3 µs, the loop 1476.1 and 1499.9 — the
same dispatch-bound behaviour the commit's bench reported, at the shapes the model actually uses. The
run's 0.347 a call against the idle card's 0.250 is four-rank contention and the probe's own wrapper,
not a second kernel. What is left is **12.10 ms/step** of six ATen launches over a tensor of 24
numbers, and no fusion inside this op reaches them — that is the next thing the term has to give, and
it is the graph's to give rather than this kernel's. The comparison that matters for the move is the
two whole-method rows on the card: **129.85 ms/step to 19.99**. Read against the host's 86.13, the
fusion is what keeps the card from being *slower* than the host at this term — an unfused card method
would be half again the host's — and so what makes the move a win there rather than a wash. The
corresponding arithmetic for the tree is that without the kernel its half of the card step would be
191.32 + (118.09 − 7.89) ≈ 302 ms, which is arithmetic over two measured terms rather than a sitting.

The lineage is worth stating precisely because it was got wrong once here: all six sittings of
`/tmp/probe_dense_tree.*` were written between 15:17 and 16:01, and the Triton compiler's own cache —
`~/.triton/cache/*/_hc_split_sinkhorn_kernel.cubin` — is stamped **17:59**, seven minutes before the
`d21d018` commit. So the kernel first ran at 17:59 and every one of the six probe outputs predates it,
which makes the 262.73-to-86 spread across those sittings a page-cache and contention difference and
not the fusion. It is also why the 262.73 ms/step figure in the plan this round was written against
carries no weight: that sitting is a 19,137.6 ms/step configuration.

### The step launches 7,076 kernels, where the sinkhorn loop alone used to launch 9,100

The plan's last unreported number is this one, and it is the number the fusion exists to move: the loop
above was **~9,100 launches a step** for arithmetic that fits in a register file, and what the step
launches now is the question that figure leaves open. `/tmp/probe_dense_tree.py --launches` takes it —
one more decode trajectory under a `torch.profiler` window on CUDA activity, the same instrument the
single-block breakdown uses. Its rows are split on `device_type`, because `cudaLaunchKernel` and
`cudaDeviceSynchronize` are rows in that report too and counting them would count the launches rather
than be one. Every rank runs the pass and rank 0 holds the window: the step's collectives need both
ends, and the first attempt at this number had three ranks return while the fourth was still in the
window — what was left was a rank 0 hanging in an all-reduce with no peer, no report and nothing in
the log.

| one decode step, 4 ranks, rank 0 | |
| --- | ---: |
| kernel launches | **7,076** |
| memcpy/memset | 770 |
| distinct kernels / distinct copies | 122 / 5 |
| the 16 heaviest kernels' share of the launches | 47% |
| device-busy under the window | 24% and 49% |
| the same step without the window | 692.7, 770.4 ms |

Two sittings of that command, back to back, and the counts are identical to the tenth: 7,076 launches
and 770 memcpy/memset both times, 122 distinct kernels and 5 distinct copies both times, the sixteen
heaviest at 47% of the launches both times, and every per-kernel call count in the table unchanged —
`ncclDevKernel_AllReduce` 88.0, `moe_single_w1w3_fp4_kernel` 40.0 and `moe_single_w2_partial_fp4_kernel`
40.0, the ATen elementwise kernels with the most calls 809.5 and 596.5, the 280 pinned→device copies.
That is what makes this a count rather than a sample, and it is the only part of the report that
survives a second sitting.

**The device-time columns do not, and the largest of them moves by a factor of 73.**
`ncclDevKernel_AllReduce_Sum_f32_RING_LL` is charged **166,991 µs a step in one sitting and 2,278.7 in
the next** — 46.9% of the recorded device time against 1.2%, at 88.0 calls a step in both — and the
step's busy fraction follows it, 355.9 of 731.6 ms against 197.5 of 814.1. The ring kernel's device
time is a peer wait rather than work, and how much of a wait a per-rank CUPTI window charges to the
kernel instead of to the gap around it is not stable across sittings. The copies on the same report
are — 132,829 against 133,068 µs a step, 0.2% apart — so what swings is the collective's own
attribution and not the window's arithmetic, and the practical reading is that nothing here may be
priced off that table. The stable row says the same thing on its own: 280 pinned→device copies a step
is the expert upload, the class's own `_upload` phase prices that traffic at **16.0–17.8 ms**, and the
window prices the same copies at 133 — a factor of eight on a row that reproduces to 0.2%. The one
time number this sitting does contribute is its unprofiled wall, 692.7 and 770.4 ms a step against the
**688.9** the sitting above times the same step at; the second is 12% high, on a host that is not idle.

**The fusion's share is 8,640 of them.** `hc_split_sinkhorn` was ~114 ATen dispatches a call at 80
calls a step — 19 iterations of up to six elementwise ops, twice a layer — and the kernel that
replaced it is one launch plus the six ATen calls `Block.hc_mixes` makes around it: 480 a step against
the loop's ~9,600. Put the loop back and the step is ~15,700 launches; it is 7,076. That is a
derivation and not a second sitting — the 9,100 is arithmetic over the loop's body and the 7,076 is
counted — and it is the cleanest statement of what the round did: the term that was cut was larger by
itself than everything the step now launches.

**The rest of the shape is why the count is still the number that matters.** 122 distinct kernels
launch 7,076 times and the sixteen heaviest are 47% of the launches, so **~3,750 of them are a long
tail** spread over the other 106, most of them ATen elementwise and reduce kernels over a few hundred
numbers. That is `hc_mixes`' signature one level up: the arithmetic is small and the launches are
many, and no kernel inside any one op reaches them. It is what a per-layer graph is for, and it is the
one thing here a second sitting confirms rather than contradicts.

**The largest kernel in the step is the collective, and its count is what to keep.**
`ncclDevKernel_AllReduce_Sum_f32_RING_LL` runs **88 times a step** — two a layer, plus one in each of
the eight layers that run the indexer — and each is a device-wide rendezvous inside a step of 40
layers. That is the reason the routed experts' partial and the shared expert's partial were made to
land in one message rather than two, and it is the other two fifths of the reconciliation above: the
tree's phase clock is 302–313 ms, the named calls account for 191.32 of it, and the wrappers do not own
the collectives — so the **111–122 ms** that gap leaves is the block's own glue and these 88 issued,
not their device time.

Beside it the experts' arithmetic is visible and small, and stable enough to read across the two
sittings: `moe_single_w1w3_fp4_kernel` **19.9 then 22.7 ms/step over 40 calls** and
`moe_single_w2_partial_fp4_kernel` **5.45 then 6.27 over 40** — 25 to 29 ms of device time a step,
against the class's 414.45, which is the staging and the issue around those products and not the
products. And the row that is neither is `hc_mixes`: six ATen launches on 24 numbers, 80 times a step,
which is this section's tail rather than a term with a kernel left to find.

### The 747 ms is a page-cache number, and this host does not keep the working set

`DeviceRoutedExperts` stages out of the checkpoint mapping, so its cost is a function of what the page
cache holds, and the class's own `_stage` line is the instrument that reads it off a run. That was
worth one deliberate experiment, because the same 8-token row has since been measured on both sides of
the line. `/tmp/fadvise_drop.py` issues `POSIX_FADV_DONTNEED` over the 48 shards — 475.25 GiB dropped
in 13.9 s, and `/tmp/mincore_resident.py`, which counts resident pages with `mincore`, confirms
**0.00 GiB** of the checkpoint left — and the same probe then measures:

| 8 tokens, 22 threads | resident | dropped |
| --- | ---: | ---: |
| load, per rank | 23.3–24.4 s | 85.0–85.1 s |
| `_stage` | 302.9–319.9 ms | **9911.7 ms** |
| `_upload` | 15.4–16.0 ms | 25.7 ms |
| `_launch` | 161.0–161.2 ms | 156.9 ms |
| unattributed inside the class | 46.8–47.1 ms | 84.1 ms |
| **whole step** | **826.7–878.0 ms** | **17013.4 ms** |

`_upload` and `_launch` do not move — 16 ms and 161 ms either way — so the difference is not the
cards, not the kernels and not the four links. It is one phase reading the same bytes off a disk
instead of out of RAM, and at 40 rows a step it is the step. The same effect showed up first as an
unexplained **15,654.7 ms** in a run whose page cache had gone cold on its own, with `stage` at
8,998.8 ms and everything else where it belongs; the drop above is that reading made deliberate.

The page cache on this host holds **68.39–99.93 GiB of the checkpoint's 475.25 GiB** (14.4–21.0%),
measured with `mincore` after and before a run, and the reason is arithmetic: **457.78 GiB of this
host's 1007 GiB is the resident bank's tmpfs segment**, which is not reclaimable with 8 GB of swap
fully used, so the 475.25 GiB the expert path reads cannot fit beside it. The class's docstring
already names the floor — a scattered expert row is **1308.0 ms** cold against 11.2 ms warm — and this
is the step-level version of it: **19.4×**, on the same row, from the cache alone.

None of that changes the measurement above it; it says which claim it is. The 722–747 ms step is the
configuration where the expert source is in RAM, which is the configuration the resident bank exists
to produce — and the stage had to be pointed at that bank before the number meant anything on a host
that does not keep the working set. That is the next section, and between the two this table is what
the wiring is worth.

### The resident bank takes the disk out of `_stage`, and not the copy into pinned

`V41Checkpoint.packed` already falls through to `resident_bank` when one is attached, so the class's
staging has read from that segment since the bank was written; what was missing was a run that had
both. The segment is filled once per boot of it — one process, 457.78 GiB, 36.6 minutes of
`/mnt/data3`, `src/models/deepseek_v4_1/resident_bank.py` is the module and its docstring is the
account of it — and any later run attaches in milliseconds by environment:

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
    /tmp/probe_v41_tp4_e2e.py --lengths 8 --steps 8
```

All four ranks attach the same 457.8 GiB segment and report it, and the experts come back
`DeviceRoutedExperts` on all 40 layers rather than falling back. The same 8-token row, instrumented,
bank on against bank off — both with the page cache as it was:

| 8 tokens, 22 threads | bank on | bank off |
| --- | ---: | ---: |
| load, per rank | 37.2–39.4 s | 23.3–24.4 s |
| `_stage` | 244.0 ms | 224.7 ms |
| `_upload` | 15.4 ms | 16.0 ms |
| `_launch` | 165.5 ms | 192.9 ms |
| unattributed inside the class | 37.2 ms | 59.3 ms |
| inside the class | 462.1 ms | 492.9 ms |
| whole step | 754.0 ms | 804.5 ms |

**Warm, the bank is neither a win nor a cost.** Its `_stage` — 242.1 and 244.0 ms across the two runs
here — lands inside the spread the page-cache-sourced stage already shows, 224.7 ms in the run above
against 302.9–319.9 ms in the deliberate pair, and nothing else in the table moves outside the 20–50
ms this host's runs differ by. That is the class's own docstring being right rather than corrected:
`resident_bank` says in as many words that it does not remove the page cache → pinned copy, and
`_stage` is that copy — 4.2 GiB a row into the pinned arena, a memcpy at ~17 GiB/s whether it reads
tmpfs or the page cache. What the bank removes is the *disk*, and the disk only appears when the cache
does not hold the working set.

So the experiment that shows what it is worth is the cold one. `/tmp/fadvise_drop.py` over the 48
shards — **0.00 GiB** confirmed resident by `mincore` — and then the same banked run:

| 8 tokens, 22 threads | bank on, cache dropped | bank off, cache dropped |
| --- | ---: | ---: |
| load, per rank | 83.6–83.9 s | 85.0–85.1 s |
| `_stage` | **242.1 ms** | 9911.7 ms |
| `_upload` | 17.4 ms | 25.7 ms |
| `_launch` | 174.2 ms | 156.9 ms |
| unattributed inside the class | 24.7 ms | 84.1 ms |
| inside the class | 458.4 ms | 10178.4 ms |
| whole step | **782.9 ms** | **17013.4 ms** |

**21.7× on the step and 41× on `_stage`**, on the same row, the same prompt and the same probe, one
environment variable apart. The step with the cache emptied is 782.9 ms against the 754.0 ms the warm
banked run measures — 3.7% — where without the bank the same pair is 19.4×. That is the whole of what
the bank buys, and it is the whole of what was at stake: **the 722–747 ms headline now holds on a host
whose page cache holds none of the checkpoint**, which is the state this host is in after a reboot and
the state the 457.78 GiB tmpfs segment keeps it in besides.

Two costs come with it, both recorded rather than argued. Loading is slower when the cache is cold —
83.6 s against 37.2 s — because the dense tree is deliberately not in the segment: the fill reads its
16.79 GiB *whole* on each of four ranks and slices, so a cold start pays ~67 GiB of SMR reads for
16.79 GiB of parameters, and only the warm case has those pages already. And the run leaves the cache
where it found it: `mincore` after the cold banked run reads **9.60 GiB of 475.25 GiB (2.0%)**, which
is the tree's own fill and rounding, against the 68–100 GiB a page-cache-sourced run warms. The expert
path contributed nothing to it, which is the same statement as the 242.1 ms.

**The long row says the same thing, and it is the row where the Engram tables are on the route.** The
same probe at `--lengths 128 --steps 4` on the same emptied cache, banked:

| 128 tokens, 22 threads, cache dropped | |
| --- | ---: |
| load, per rank | 82.7–82.9 s |
| prefill | **53.42 s** (2.4 tok/s), 52.95 s of it inside the class |
| decode | 802.2 ms, 487.4 in the class and 314.8 in the tree |
| decode phases | stage 265.0, upload 16.8, launch 159.7, unattributed 45.9 ms |
| `mincore` after | **9.61 GiB** of 475.25 GiB |

The prefill is 99.1% inside the class and its staging is 265.0 ms a row against the 242.1–244.0 ms the
8-token rows measure — the same copy, paid 128 times — so the bank does not reach it and was never
going to. The number to read is the last one: a cold banked run that gathers **3,072 Engram rows per
table** leaves the page cache at 9.61 GiB, one hundredth of a GiB above the 9.60 GiB the 8-token run
left. The tables are in the segment and `rows` reads them out of it, so a banked prefill's gather is
not on the 253.4 s/table cold path the host page records. 53.42 s against the 42.86 s the warm
uninstrumented run records is a 25% difference this pair does not separate into the instrumentation
and the cold host's own reads, and neither of those is the bank.

### A pass over the banked checkpoint reads zero bytes from the device, and `read_bytes` is what says so

Every number above this line is a wall clock, and a wall clock cannot tell a run that read its
experts out of the segment from one that read them off `/mnt/data3` and found them in the page cache.
`/proc/self/io`'s `read_bytes` can: it counts what the kernel fetched from the block device and not
what it served from cache, so a checkpoint that is resident reads **zero** there whatever the clock
does. `/tmp/probe_v41_resident.py` brackets each phase with it, one process, four cards attached, the
457.78 GiB segment attached by `DEEPSEEK_V41_RESIDENT_EXPERTS=1`:

```
load_backbone                   74.98 s       0.00 GiB read       45.99 GiB rss

residency, counted:
  process rss 46.74 GiB, peak 49.21 GiB
  /dev/shm 457.78 GiB
  torch pinned 0.00 GiB
  cuda:0 allocated 1.49 GiB, reserved 1.52 GiB
  cuda:1 allocated 1.49 GiB, reserved 1.52 GiB
  cuda:2 allocated 1.49 GiB, reserved 1.52 GiB
  cuda:3 allocated 1.49 GiB, reserved 1.52 GiB
  device expert class DeviceRoutedExperts, world 4

  prompt 128 warmup             66.94 s       0.00 GiB read       96.95 GiB rss
  prompt 128 prefill            62.33 s       0.00 GiB read        0.07 GiB rss
  prompt 128 decode              5.51 s       0.02 GiB read        6.04 GiB rss
  prompt 512 warmup            264.47 s       0.00 GiB read       48.59 GiB rss
  prompt 512 prefill           261.95 s       0.00 GiB read        0.10 GiB rss
  prompt 512 decode              4.99 s       0.03 GiB read        0.91 GiB rss
```

**Zero across both warmups, both prefills and the load**, which is the acceptance the plan asked for
and the first time it has been read off the kernel's own counter rather than inferred: the 41040
packed rows a 512-token prefill stages on a two-route rank and the 12,288 Engram gathers it makes per
table come out of the segment, and the 0.00 on `load_backbone` is the *dense tree*, which is
deliberately not in the
segment, being served out of the page cache on this host — the 83.6 s / 67 GiB read the cold banked
table above records is what that line reads when the cache is dropped. Decode is 0.02 and 0.03 GiB
over four steps each, 5–8 MiB a step: real, three orders below the 17.9 MiB a row a disk-sourced
stage would be, and this run does not attempt to attribute it, so it is recorded rather than
explained.

The residency block is verification 5 and it is the number the directive is about. The process's own
`VmRSS` is **46.74 GiB**, its peak 49.21 GiB; `/dev/shm` holds **457.78 GiB** — one copy of the
segment, counted on disk, not four; `torch` pinned host memory is **0.00 GiB**; and each of the four
cards holds **1.49 GiB allocated, 1.52 GiB reserved**. Against the plan's ~477 GiB budget, the
checkpoint's share is the 457.78 GiB of shared memory plus the 16.79 GiB of tree that lives in the
process's own RSS and nowhere else — no replica of it exists on any other rank, which is the failure
mode this measurement exists to rule out.

Two things the numbers are not. They are **one process driving four cards**, not four processes under
`torchrun`, so the clock columns here are not comparable with the ones above — 261.95 s for a
512-token prefill against the sweep's 137.9–143.9 s at 0 rows is that difference and not the bank.
And every prompt is measured after a warmup pass over the same prompt at the same length, so the
`prefill` row is the second one by construction; that the *warmup* also reads 0.00 is the point —
there is no cold pass left for the bank to fix.

### Zero disk reads is also, at this size, near-zero cost: 1.4–2.3 ms a table for a 512-token prefill

`read_bytes` says the gathers read nothing off the device, and that is a route rather than a price —
the segment is tmpfs, and `V41Checkpoint.rows` reaches it through a `torch.frombuffer` view rather
than through a privately allocated array, which is not the copy the host page prices. So
`/tmp/probe_engram_bank.py` attaches the same 457.78 GiB segment, draws the released geometry's ids
— **12,288 rows a table**, 512 tokens by the config's 24 hash columns, uniform over each table's
whole range so the scatter is not flattered — and times the call a forward makes:

| segment, one 512-token prefill's gathers, per table | layer 1 | layer 14 |
| --- | ---: | ---: |
| codes, `view[rows]`, scattered | 0.22 ms | 0.22–0.26 ms |
| scales, the same | 0.03–0.04 ms | 0.04 ms |
| `V41Checkpoint.rows`, the real call | 0.28–0.29 ms | 0.29–0.42 ms |
| `dequantize_rows` over the gathered rows | 1.19–1.39 ms | 1.03–1.88 ms |
| **a table, gather and dequant** | **1.5–1.7 ms** | **1.4–2.3 ms** |
| the same 3.00 MiB copied contiguously | 0.07–0.26 ms | 0.05 ms |

Two attachments on a host that was not idle (a `soong_build` at 833% CPU), so each row is a band
rather than a point; the conclusion survives it — **both tables together are 3–4 ms**, against the
**253.4 s** the same rows cost from a cold shingled disk and the **3.0 tok/s** the 128-token card
prefill above measures. Zero disk reads is also, at this size, near-zero cost.

The scatter is free and the dequant is not. Scattered and sequential row ids cost the same
0.20–0.26 ms, because 3 MiB is a working set rather than a stream and the segment is RAM either
way — the 91.55 GiB the ids are drawn across never enters it. What is left is
`dequantize_rows`' three passes over a 12.6 MB fp32 expansion, and it is two thirds of the table's
cost at a size where nothing is bytes-bound.

The last control is the one that reconciles this with the host page's **0.009 s** for the same
12,288 rows out of a `resident_engram=True` copy, which reads as a 31× difference in the segment's
favour and is not one. The same rows gathered one at a time, a Python call each, are **19.96 and
19.61 ms** here against the 9 ms recorded there — the same operation in two writings, the gap being
the loop's own overhead and not the array — where ATen's batched `index_select` is **0.22–0.26 ms**
for them. The batched number is the one a forward pays; the per-row pair is an artefact of how each
page took its control, and the two pages agree once they are read that way.

The card's own `engram` row says the same from the other side. It is **6.78 ms a decode step** over
two calls in the op-by-op table above, and the gather is not what that buys: a decode step asks for
24 rows a table against the 12,288 measured here, so the host half of an engram call is a fraction
of a millisecond, and what is left is the card's own `wkv` — `[1, 6144] x [6144, 25600]`, 314 MB of
bf16 read once a call — plus the gate and the handover of 12 KiB. Engram is on the host because its
*tables* are 189.13 GiB and cannot be anywhere else, not because the arithmetic wants to be there.

### `--threads` is worth 1.13× with the source resident, and 6.8× without it

`torch.distributed.run` sets `OMP_NUM_THREADS` to 1 for every worker unless the environment already
had one, so the launcher takes `--threads` and says so out loud. Three tokens of `The capital of
France is`, two runs each way in both orders, in one session:

| `--threads` | resident bank on | off |
| --- | ---: | ---: |
| 22 | 5.7 s, 5.8 s | 6.5 s |
| 1 | 6.6 s, 6.5 s | **44.1 s** |

With the source resident the flag is worth **1.13×** — 5.7–5.8 s against 6.5–6.6 s, reproduced in both
orders — and what it buys is host work: the gate, the head, the layer glue. Without the bank the same
three tokens are 6.5 s at 22 threads and **44.1 s at one**, 14.70 s/token, because the per-row read
out of `/mnt/data3` is serialized on a single thread. The launcher's docstring used to record this
flag as 5.1 s against 6.6 s as if it were a thread effect; those two runs differed in the resident
bank as well as the thread count, so what the pair measured was the disk. The table above is the
controlled pair, and the 6.8× belongs to the configuration with no resident source rather than to the
flag.

### The row loop runs one row deep, and it is worth 1.10× on a prefill

A row used to be strictly serialized: the host could not stage row `k+1` until the card side of row
`k` had returned, and that returned only once row `k`'s partials had landed. `forward` now holds one
row — it stages row `k+1` and issues its kernels, and only then drains row `k`. **A decode step is one
row a layer, so a row-deep pipeline has nothing to overlap there** and the whole of what it can be
worth is a prefill's, `n` rows a layer with `n` the prompt length. That is what reframes this: not a
lever on the headline number, a lever on the prefill column beside it.

`/tmp/probe_v41_tp4_pipe_matrix.py` runs both orders over the same prompt in one process — the second
engine in a process is ~10% slower on this host, which rules out comparing across processes — and
alternates them, so the node's own ~20% drift between sittings lands on both columns rather than on
one. Four ranks under `torchrun`, one process a card, the resident bank on, 22 threads. All phase
columns are totals over the whole prefill. The serial column's 53.61 s at 128 tokens is the same
sitting-drift figure the banked table above records as 53.42 s, and not the 42.86 s the warm
uninstrumented run does — the page separates those two [there](#the-resident-bank-takes-the-disk-out-of-_stage-and-not-the-copy-into-pinned)
and the columns here are the pair to read, not either against a table from another sitting:

| prompt | order | prefill | tok/s | `_stage_row` | `_stage` | `_upload` | `_issue` | `_drain` |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | serial | 13.51 s, 13.77 s | 2.4 | 7488 ms | 6850 ms | 482 ms | 849 ms | 4792 ms |
| 32 | pipelined | 12.43 s, 12.42 s | 2.6 | 8720 ms | 8048 ms | 594 ms | 989 ms | 2305 ms |
| 128 | serial | 53.61 s, 53.45 s | 2.4 | 30096 ms | 27542 ms | 1964 ms | 3434 ms | 19264 ms |
| 128 | pipelined | 48.44 s, 48.18 s | 2.7 | 34975 ms | 32717 ms | 2007 ms | 3501 ms | 9033 ms |

**1.098×** at 32 tokens and **1.108×** at 128, and the per-row-layer account is flat across both
prompts: the drain falls from **3.75 ms to 1.8 ms** a row-layer and the staging rises from **5.35 ms
to 6.3 ms**, for a net **~1.0 ms** off a row-layer that costs about 12. The rise is the pipeline
ceasing to hide a cost rather than creating one: the staging is a memcpy out of tmpfs, and with a row
in flight it now shares the memory system with that row's H2D reading the *other* pinned arena. The
residual 1.8 ms drain is the row's own arena copy plus its kernel, which is the floor one row deep —
a two-row pipeline would have to issue row `k+1`'s H2D while row `k`'s kernel ran, and it needs a
third pinned arena to do it. The pinned pool is not what limits this: the same probe swept two, three
and four arenas a layer and `_take_buffer` is 10–29 ms against a stage of 4–8 s, so `pinned_buffers`
stays at 2. **That 10–29 ms is this configuration's, not the rotation's** — with no pool every row
stages, so the rotation and the row count agree; a pool makes them disagree and turns this wait into
the pass's second-largest term. See [the buffer
rotation](#a-pool-hit-used-to-take-a-staging-buffer-and-the-copy-behind-it-was-left-uncovered) below
for the 6.89 s and the fix.

**Two things had to change for it to pay, and the first version did not.** It measured **1.012×**, and
the cause was the routing: `indices_row.tolist()` inside the row loop synchronizes the card's stream,
and under the pipeline that stream already has the *previous* row's kernel on it, so those reads cost
**2.9 ms a row** where the same reads cost 0.1 ms in a loop that has already drained. That is as much
as the drain saving, handed straight back. `_route_ids` now takes the whole `[n, topk]` to host memory
in one pinned copy ahead of the loop, so every row's read is a read of host memory; and the route
weights, which `_issue` gathers on the card, are copied back `non_blocking`, so nothing in the loop
waits on a pageable transfer.

**It does not change the answer, and the probe is arranged to show that rather than assert it.** A
prefill of this path is not bit-reproducible — the same order twice differs by 3.4e-02–7.1e-02 max abs
on the last token's logits, which is the run-to-run spread this path has on its own — and the
pipelined run differs from the serial one by 2.0e-02, inside that spread. Reading a serial-against-
pipelined logit difference as the pipeline's would be reading noise.

**The 1.10× is measured on the un-resident path and is not claimed for the resident one.** What that
pipeline overlaps is a drain of 3.75 ms a row-layer against a staging of 6.3 ms, and the next section
takes 7.8× of that staging away: a prefill staging a fifth of these bytes has a fifth as much to hide
behind a row. The two configurations have not been run against each other, so this number belongs to
the column it was measured on (`--hot-rows 0`, which is the default) and not to the one below.

### A per-layer resident set is worth 2.7× on a prefill, and it is the fill that pays for it

`--expert-hot-rows N` is the knob the two sections above set up. An expert is resident iff the layer
asks this rank for it at least twice over the pass, so the set is a property of the routing and not of
a policy; the arena and the pinned block are one a card, shared by all forty layers and refilled by
whichever layer is running, which is what makes 148 rows affordable at all — forty layers holding
their own would be 105 GiB of a 22 GiB card. The class docstring is the mechanism
(`ResidentSet`, `_hot_rows`, `_fill`); this is what it measures.

`/tmp/probe_v41_hot_ab.py`, four ranks under `torchrun`, one process a card, 22 threads, the resident
bank on, one 512-token prompt, `--length 512 --decode 1`, the four configurations run one after
another on the same cards in one session. `drawn_rows` is every route a rank was dealt and
`expert_rows` the ones that missed, so `drawn - staged` is the coverage the set bought. Every counter
below is the whole run — the 512-row prefill plus the one decode row `--decode 1` asks for — so the
staged column carries 80 draws on ranks 0 and 1 and 40 on ranks 2 and 3 that the prefill did not make,
and the tok/s column is the mean of the four ranks:

| `--hot-rows` | prefill, ranks 0-3 | prefill tok/s | draws, r0/r1 | staged, r0/r1/r2/r3 | % resident | filled, r0 | layers cut |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 143.9 / 140.1 / 138.0 / 137.9 s | 3.7 | 41040 | 41040 / 41040 / 20520 / 20520 | 0.0 / 0.0 / 0.0 / 0.0 | 0 | 0 |
| 64 | **52.8 / 51.1 / 53.2 / 51.8 s** | 9.8 | 41040 | 5307 / 5419 / 1696 / 1681 | 87.1 / 86.8 / 91.7 / 91.8 | 2547 | 39 / 40 / 19 / 19 |
| 148 | **49.1 / 50.7 / 52.5 / 49.0 s** | 10.2 | 41040 | 1730 / 1743 / 1308 / 1322 | 95.8 / 95.8 / 93.6 / 93.6 | 3932 | 0 / 0 / 0 / 0 |
| 192 | 55.0 / 52.9 / 54.3 / 54.5 s | 9.5 | 41040 | 1724 / 1685 / 1361 / 1301 | 95.8 / 95.9 / 93.4 / 93.7 | 3929 | 0 / 0 / 0 / 0 |

**2.6–2.7× on every rank at 64 rows**, and **7.8× fewer packed rows staged** on rank 0 — 41040 to
5307 — against the **6.54×** the offline sweep predicted for this length. The prediction was a
four-rank sum; the deal is not even, and the per-rank columns are what says so. `_split` sorts the
row's six slots by global expert id, walks that order round-robin over four ranks and keeps whichever
ranks this process names, so a rank holding two of the six is dealt two routes a row and a rank
holding one is dealt one: ranks 0 and 1 draw 41040 over the pass and ranks 2 and 3 draw 20520. The
narrower deal repeats more, so ranks 2 and 3 come out **91.7% resident against rank 0's 87.1%** at the
same `hot_rows`. Every rank prints `first token 2413` and `[2413, 21779]` in every column.

**64 is where the curve flattens, and the counters said otherwise.** The 39 and 40 layers cut on ranks
0 and 1 at 64 rows are the arena being the binding constraint — that is what `capped_layers` is for —
and uncapping it at 148 rows takes the staged count down a further **3.1×**, 5307 to 1730, with no
layer cut anywhere. It buys **6–7%** of the prefill and costs **2.3×** the arena. Past that it is a
loss: at 192 rows the staged and filled rows are within 60 of 148's on every rank — the arena is not
doing less work, it is only wider — and the prefill costs more. The pair was re-run three times
because a single sitting is worth what the node's own drift says it is worth:

| sitting | `--decode` | order | prefill at 148, ranks 0-3 | at 192, ranks 0-3 | rank mean |
| --- | ---: | --- | ---: | ---: | ---: |
| the width sweep above | 1 | 0, 64, 148, 192 | 49.1 / 50.7 / 52.5 / 49.0 | 55.0 / 52.9 / 54.3 / 54.5 | 50.3 → 54.2 s, +7.6% |
| `/tmp/r512_{a,b}.pt` | 4 | 148 then 192 | 50.1 / 48.5 / 50.0 / 49.0 | 57.3 / 53.4 / 54.3 / 54.3 | 49.4 → 54.8 s, +11.0% |
| `/tmp/q512_1{a,b}.pt` | 4 | 148 then 192 | 48.2 / 48.6 / 49.4 / 49.8 | 49.7 / 53.0 / 51.7 / 50.6 | 49.0 → 51.3 s, +4.6% |
| `/tmp/q512_2{a,b}.pt` | 4 | 148 then 192 | 49.0 / 49.4 / 52.9 / 51.9 | 54.4 / 50.3 / 51.6 / 50.1 | 50.8 → 51.6 s, +1.6% |

The last two rows are the A-B-A-B: 148, 192, then 148, 192 again, one prompt, only the arena width
differing. On the rank mean 192 is the slower column in all four sittings, and in three of the four
every rank is on the same side of it — by 0.8 s at the narrowest and 7.2 s at the widest. The fourth
is the A-B-A-B's second round, and it splits two ranks each way for a mean margin of 1.6%, which is
less than half the 3.7% the *same* 148-row configuration drifted between the two rounds of that same
pair (`/tmp/q512_1a.pt` to `/tmp/q512_2a.pt`, 49.0 to 50.8 s, at load average 25 and 34). So the
direction is measured and the size is not: without a quiet machine this is worth 1.6% to 11.0%, and
the honest headline is that the wider arena is not buying anything the counters can see while it
costs more every time it has been run.

The four counters this class keeps are identical across the pair to within a few dozen rows, so what
the extra 789 MiB of arena and 789 MiB of pinned block cost is not anything this class counts. It is
reported here as measured and unexplained rather than argued away, and it is the reason `hot_rows` is
sized to the routing and not above it.

**What it costs is `_fill`, and it is 16–25% of the prefill it sits in.** `_fill` is the one thing the
class does per *layer* rather than per row, so no counter has a line for it: it is invisible in the
A/B probe's wall clock and the counters only report what it moved (`filled_rows`). It is timed by
wrapping the method — `/tmp/probe_v41_resident_fill.py`, one `perf_counter` a layer — on the top of
the same 512-token prefill:

| `--hot-rows` | rank | fill, 40 layers | a layer | rows filled |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0–3 | 0.00 s | — | 0 |
| 64 | 0 / 1 / 2 / 3 | 8.64 / 7.95 / 7.92 / 8.20 s | 216.0 / 198.8 / 198.0 / 205.1 ms | 2548 / 2560 / 2311 / 2282 |
| 148 | 0 / 1 / 2 / 3 | 13.06 / 12.83 / 9.98 / 9.85 s | 326.6 / 320.7 / 249.4 / 246.2 ms | 3865 / 4019 / 2491 / 2400 |

That is near enough **3.4 ms an expert row** in both columns — 8.64 s over 2548 rows and 13.06 s over
3865 — so the fill is linear in the set's width and indifferent to everything else, and it is charged
every layer whether or not that layer has anything new to say. It is a **2.6×** correction to the 82 ms
a layer the class docstring used to claim from arithmetic: the fill pays *two* copies a row where
`_stage` pays one, the bank into the pinned block at 17.9 MiB and the pinned block into the arena at
17.9 MiB again, and the second reads bytes the first has just written, so the two halves do not add up
to the one-directional 14 GiB/s `_stage` reaches.

**Decode is the column the set does not help, and the sweep prices it at 10–11%.** 0.668 s a token at 0
rows against **0.743** at 64 and **0.736** at 148, all three in the same sitting at `--decode 1`, and
inside each column the four ranks agree to within 0.006 s. A decode step is one row a layer and one
row asks
each expert once, so nothing of it can be resident and `_fill` returns before it looks at anything;
the only per-row work the set adds is the index copy `_issue` makes because the arena rows are no
longer `0..k`, four 16-byte H2Ds a layer, which is not 1.9 ms of one. This page does not have the
mechanism for the 10–11%, and says so rather than crediting the fill with a cost it does not pay.

**It is also the column where the readings disagree, so read the 10–11% as what one sitting found.**
The widths do not separate from each other here at all, and the A-B-A-B is what says so. Within the
`--decode 4` instrument, 148 measures **0.788, 0.717 and 0.857** across its three sittings and 192
measures **0.888, 0.756 and 0.739** across its three — each configuration drifting about 20% between
its own sittings, which is wider than the widest gap between the two configurations in any one
sitting. Worse for a one-sitting reading, the sign reverses: 192 is 5.5% slower in the A-B-A-B's first
round (0.717 against 0.756) and 13.8% faster in its second (0.857 against 0.739), and in that second
round it is the 148-row column that carries the outlier. The sweep's own 0.949 at 192 rows is the same
story from the `--decode 1` side: 192 measures 0.756, 0.739 and 0.888 on the other instrument, so
0.949 is the node and not the arena.

The corollary is that the set-off control was only ever run as the *first* column of one sweep, so the
11% step from it to the 64-row column cannot be separated from a warm-up that had finished by the
second column. The check that would settle it is that sweep with the control repeated at the end as
well as the start, and it has not been run; until it is, `--expert-hot-rows` is recommended for
prefill and left off for decode on the strength of the mechanism — a decode row repeats nothing, so a
resident set cannot help it — rather than on the strength of this column.

**The one column the set does not move is the logits, and that is now measured as zero rather than
inferred.** `--hot-rows 0` against `--hot-rows 64`, same prompt, same sitting: the set takes rank 0
from 41280 staged rows to 5613 (86.4% resident against 0.0%), ranks 2 and 3 from 20640 to 1807 and
1792 (91.2 and 91.3%), and the pair then agrees on **32 of the top 32 ids, in position, at `|dlogit|
0.000e+00` of a max |logit| of 29.15 — on all four ranks**, ranks 1–3 included. Both legs route
identically, so every expert the prefill asked for was computed from the same packed weights whether
it came out of the arena or off the checkpoint.

That is the answer a cache has to give and it is worth stating as one: an 86–91% reduction in staged
rows that moves not one bit of the tail is what makes `_hot_rows` a pure cost knob. It also disposes
of the reading this page carried before the fix, in which *two different configurations* agreeing on
28 of 32 ids at a median 0.30 looked like a tighter match than two runs of the *same* one — a
"comparison inside a ±3% envelope" that would have made the logit columns a repeat. The same
configuration's floor was 30 of the 32 out of position at the time, so that 28 was noise wearing the
shape of a result, and the envelope it implied does not exist: both floors are now exactly zero, and
the columns separate nothing about the resident set because there is nothing there to separate.

**Residency, counted.** `VmRSS` at the end of a 512-token pass, per rank, against the same pass with
the set off: rank 0 **119.84 GiB → 122.36 GiB at 64 rows and 123.55 at 148**, rank 2 **86.03 → 88.75 →
90.63**. The pinned block is 1.15 GiB and 2.65 GiB of that, so the set's own cost is its pinned block
plus 1–2 GiB, which is the honest answer to a question the earlier A/B could not ask: `ru_maxrss` over
a whole process moves by tens of GiB with the *prompt length*, and reading that as the set was reading
the prompt. `VmLck` is 0 kB throughout — the driver's pinning does not appear as `mlock` here — and one
arena a card, not forty, is what the 2690 MiB in the table above already says.

Reproduce with:

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_hot_ab.py \
    --length 512 --hot-rows 64 --out /tmp/s512_h64.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_resident_fill.py \
    --length 512 --hot-rows 64 --out /tmp/fill_h64.out
```

### The pool spends the same arena on what the pass draws, and its first key answered the wrong layer

`--expert-pool-rows` is the same arena spent the other way round. A fill row is chosen up front for the
whole layer and paid for whether that layer asks for it or not; a pool row is handed to an expert on
its first sight *in that layer* and taken back least-recently-used when the pool runs out, so it is
paid for exactly the draws the pass makes. Both are `expert_hot_rows`' arena, both cost 18.8 MB a row,
and what differs is which repeats they keep and when the copy is made.

**The first version of it was keyed on the bare expert id, and that is a wrong answer rather than a
slower path.** An expert id is not an identity: the weights are
`layers.{layer}.ffn.experts.{expert}.{which}.weight`, so layer 6's expert 7 is a different tensor from
layer 5's, and a pool keyed on the id alone answers the later draw with an earlier layer's bytes. It
does so *more* often the wider the pool is, and it does so silently — the rows are the right shape and
the bytes are a real expert's. The counters could not see it: the pre-fix sweep's counters were
exactly reproducible (1434 rows staged at 288 on rank 0, 1434 again; 3037 at 192, 3037 again, on the
same counter the tables below read) while the tokens were not. Same prompt, same cards, per-rank
tokens as this probe drew them — and they *are*
draws rather than an argmax, which is [its own subsection below](#the-probes-tokens-are-the-samplers-draw-and-the-logit-column-is-the-parity-check):

| `--expert-pool-rows` | r0 | r1 | r2 | r3 | staged, r0/r1 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `[2413, 21779]` | `[2413, 21779]` | `[2413, 21779]` | `[2413, 21779]` | 41040 / 41040 |
| 192 | `[2413, 45539]` | `[2413, 82761]` | `[2413, 3373]` | `[2413, 63767]` | 3037 / 3010 |
| 288 | `[2413, 21779]` | `[2413, 21779]` | `[2413, 21779]` | `[2413, 21779]` | 1434 / 1386 |

Four ranks of one run disagreeing with each other is not a race — the tree is cut so that every rank
computes the same logits — and the top-32 comparison prices it: against the control, **1 of the top 32
ids in position at a worst |dlogit| of 4.717e+00 at 192 rows and 6.787e+00 at 288, against a max
|logit| of 2.915e+01**, with 288 landing on the control's second token by luck and 192 missing it on
every rank but a different way each time. The fix is the key: `pool_map`/`pool_lru` are keyed on
`(layer_id, expert)` and `_pool_row` composes the tuple. And the verification is the column that
priced it: the logit comparison over **six sittings and four ranks is 32 of 32 ids in position at
`|dlogit| 0.000e+00`: 24 payloads, 276 pairs, every one of them exactly zero and 32 of 32 ids in
position**, pooled runs against the control and against
each other. The token columns of every pooled run above agree too, but a token column is a weak
instrument under this probe's sampler — [the subsection below](#the-probes-tokens-are-the-samplers-draw-and-the-logit-column-is-the-parity-check)
prices how weak — and what convicted the key was the logit comparison behind the pre-fix table rather
than its tokens.

**What the corrected key means is that a pool row belongs to a layer, not to the model.** With the
layer in the key nothing carries across a layer boundary, so the pool holds the working set of *one*
layer, and that is visible as an identity rather than an inference: `pool_staged - pool_evicted` is
the width exactly, in every pooled run (5700 - 5412 = 288; 5761 - 5613 = 148; 6579 - 6483 = 96). A
512-token pass of 40 layers asks rank 0 for **~142 distinct experts a layer** out of the 384 the model
has — 5700 rows staged over the pass at 288 rows — and it asks 41040 times, so **86.1% of the pass's
draws are answered out of the arena and every row the pool pays for is a row the pass asked for at
least once**.

**It is 4.0–4.2× on the prefill against no arena, and it beats the resident set at the set's own
width.** Six sittings, one 512-token prompt, the same four cards, the configurations alternating in
time — control, 148-hot, 288-pool, 288-pool, 148-hot, control — with the 512-row prefill and one
decode row, both counted, so the staged column carries 80 draws on ranks 0 and 1 and 40 on ranks 2 and
3 that the prefill did not make:

| configuration | prefill, ranks 0-3 | rank mean | staged, r0/r1/r2/r3 | % resident | filled | arena | pinned |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| none | 185.71 / 187.18 / 185.91 / 185.72 s | 186.13 | 41040 / 41040 / 20520 / 20520 | 0.0 / 0.0 / 0.0 / 0.0 | 0 | 36 MiB | 0 |
| `--hot-rows 148` | 48.57 / 48.75 / 48.57 / 49.60 s | 48.87 | 1768 / 1691 / 1371 / 1346 | 95.7 / 95.9 / 93.3 / 93.4 | 3935 | 2690 MiB | 2654 MiB |
| `--pool-rows 288` | 46.32 / 44.80 / 44.45 / 45.51 s | 45.27 | 5700 / 5749 / 3828 / 3778 | 86.1 / 86.0 / 81.3 / 81.6 | 0 | 5200 MiB | 0 |
| `--pool-rows 288` | 42.10 / 45.17 / 47.24 / 42.23 s | 44.19 | 5700 / 5749 / 3828 / 3778 | 86.1 / 86.0 / 81.3 / 81.6 | 0 | 5200 MiB | 0 |
| `--hot-rows 148` | 52.20 / 47.34 / 47.50 / 49.44 s | 49.12 | 1768 / 1691 / 1371 / 1346 | 95.7 / 95.9 / 93.3 / 93.4 | 3935 | 2690 MiB | 2654 MiB |
| none | 186.72 / 186.36 / 186.22 / 187.10 s | 186.60 | 41040 / 41040 / 20520 / 20520 | 0.0 / 0.0 / 0.0 / 0.0 | 0 | 36 MiB | 0 |

The pool is **4.0–4.2×** the control — 186.4 to 44.7 on the rank mean of the pair — and **1.10×** the
set, 49.0 to 44.7, and the row counts are what explains the second rather than contradicting it:
**both mechanisms move about 5700 packed rows.** The set stages 1768 and *fills* 3935, and a fill row
is a row moved, so its pass costs 5703 row-moves against the pool's 5700 at the same width and 5761 at
148. What differs is where in the layer they are paid: `_fill` is one serial block a layer, written
before that layer's first draw, at the 3.4 ms an expert row the [fill table
above](#a-per-layer-resident-set-is-worth-27-on-a-prefill-and-it-is-the-fill-that-pays-for-it)
measures, while a pool row is copied on the draw that needs it and never if the layer does not draw
it. The same bytes in a different place in the layer are worth 4.4 s of this prefill, and the pool
reaches them without a hotness model at all: no prediction, no `_fill`, and `capped_layers` is 0
everywhere because a pool that runs out evicts rather than cutting a layer.

**The width has a floor at one layer's working set, and the counters were read for it before the
sweep ran.** A B C D D C B A — 148-hot, 96-pool, 148-pool, 288-pool, and back — one prompt, one
sitting, one process a card, rank means of the two sittings each configuration got:

| configuration | arena | staged, r0 | % resident, r0/r1 | prefill, rank mean | pair mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `--hot-rows 148` | 2690 MiB | 1768 (+ 3935 filled) | 95.7 / 95.9 | 48.03 / 48.93 s | 48.48 s |
| `--pool-rows 96` | 1757 MiB | 6579 | 84.0 / 83.7 | 47.73 / 48.28 s | 48.01 s |
| `--pool-rows 148` | 2690 MiB | 5761 | 86.0 / 85.8 | 45.80 / 41.60 s | 43.70 s |
| `--pool-rows 288` | 5200 MiB | 5700 | 86.1 / 86.0 | 46.03 / 45.98 s | 46.01 s |

The prediction, stated in the sweep's own header before it ran: at 148 rows the pool should stage the
same ~5700 rows as at 288 rows, because 148 is already more rows than a layer's ~142 distinct experts;
at 96 it should stage **more**, because below a layer's working set the pool evicts its own keys inside
one layer and pays to stage the same expert twice in that layer. Both hold — 5761 rows, 1.1% above
288's, and 6579, 15.4% above it — and the second is the floor showing: 96 rows answers 84.0% of rank
0's draws where 148 answers 86.0%, costing 818 rows of re-staging over the pass, 20 a layer.

**At 148 rows the pool and the set are the same arena, and that is the comparison the earlier sweeps
could not make.** Same 2690 MiB, same prompt, same cards: the pool is faster in both of its sittings —
45.80 and 41.60 s against 48.03 and 48.93 — and it carries **no pinned block at all** where the set
holds 2654 MiB of one, because a pool row is copied straight out of the bank into its arena row where
a fill row goes bank → pinned → arena. 288 rows then buys nothing over 148 — 61 fewer rows staged out
of 5761, and the two inside each other's drift on the wall clock — for 2.5 GiB more arena, which is
the conclusion the width sweep above reached about 148 rows against 192 for the same reason: on this
prompt the routing's working set is ~142 a layer and anything above it is holding rows nobody asks for
again.

**One caveat on that number, and it is about the floor rather than the measurement.** 142 distinct
experts a layer is what *this* prompt draws in a layer, and the floor moves with the prompt: a longer
prefill draws more distinct experts a layer, up to one per expert the model has, so the width that
cannot be below the floor at any prompt is 384 rows — **6.9 GiB a card**, more than the four cards
have to give once the tree and the caches are on them. `--pool-rows 148` is sized to a 512-token
prefill and is not a constant of the mechanism; what is a constant is that below the layer's distinct
count the pool pays twice for the same expert. The sweep below takes the arrow the other way — a pass
a quarter the length — and the floor moves **less** than proportionally, which is the same
concentration that makes 148 and 288 the same width here; read the two together before sizing a width
for a prompt that is neither.

**The sitting drifts in a U, and the pair means are what say so.** The rank means in time order are
48.03, 47.73, 45.80, 46.03, 45.98, 41.60, 48.28, 48.93: the two 148-row sittings sit on the fast
stretch and the two control sittings on the slow one at either end, so the size of the 148-against-288
and 148-against-set gaps is partly the order. What survives the order is the direction and the
counters: both 148-row sittings beat both set sittings, both 288-row sittings beat both set sittings,
both 96-row sittings match the set's, and the row counts are identical between the two sittings of
every configuration — 6579, 5761 and 5700 staged on rank 0 twice each, to the row.

**Decode is not this knob either, and this sweep is the clearest evidence yet that its column is the
node — because every decode column in it is one token wide.** A decode step asks a layer for one
row's worth of experts and nothing in it repeats, so a pool evicts everything it stages there, and
the two 288-row sittings — the same configuration, adjacent in time — read **0.698 and 0.807 s a
token, 15.6% apart**, which is wider than the gap between any two configurations in the sweep. That
is true of one step. A pool *survives* the step, and the next step asks for much the same set, so a
decode read as a single token cannot show it: [the subsection
below](#the-pool-is-a-decode-lever-and-a-one-token-decode-is-the-measurement-that-hid-it) runs the
same 128-token prompt with 64 and 256 decode steps a leg and the pool is worth 1.27x and 1.40x
there, with the first step as expensive as the control. The corrected six-sitting sweep says the
same thing about the set: control 0.717 and 0.756, `--hot-rows 148` 0.714 and 0.669, so the set is on
the fast side of the control in one pairing and on the slow side in the other and the 10–11% the
width sweep measured does not reproduce. A one-step decode prices neither mechanism, and the set is
the one that does not amortise.

Reproduce with:

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_hot_ab.py \
    --length 512 --decode 1 --hot-rows 148 --out /tmp/pw_h148.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_hot_ab.py \
    --length 512 --decode 1 --pool-rows 148 --out /tmp/pw_p148.pt
python /tmp/probe_v41_hot_ab.py --compare /tmp/pw_h148.pt.r0 /tmp/pw_p148.pt.r0
```

`/tmp/run_poolfix.sh` is the six-sitting A-B-C-C-B-A behind the first table — the control at both ends,
which the resident-set section above says had not been run and which comes back 186.13 against 186.60 s
on the rank mean, 0.25% apart, with the decode column 5.5% apart on the same two runs.
`/tmp/run_poolwidth.sh` is the eight-sitting width sweep behind the second, and `/tmp/run_poolshort.sh`
the five-sitting A-B-C-D-A on the subsection below, which moves the length rather than the width.

### A quarter-length pass moves the floor, and by less than the draw count moves

**The corrected key makes the floor a property of the pass, and the way to test that is to hold the
width and shorten the pass.** Every pool measurement above sits on a 512-token prompt. This one is the
same probe, the same four cards and three widths at `--length 128` — a quarter of the tokens, the
control at both ends, A B C D A, one sitting. Both predictions were in the sweep's own header before
it ran: **(1)** a layer is drawn ~256 times at 128 tokens against ~1024 at 512, so if the distinct
count scaled with the draw count the floor would land near 40 and `--pool-rows 64` would be sitting on
it while 32 staged strictly more; **(2)** the pool should still beat the control, but by less than the
4.0–4.2× at 512, because both mechanisms are paid per row staged and this pass stages a quarter as
many.

| configuration | arena | staged, r0/r1/r2/r3 | % resident, r0/r1 | prefill, ranks 0-3 | rank mean |
| --- | ---: | --- | ---: | --- | ---: |
| `--pool-rows 0` | 36 MiB | 10320 / 10320 / 5160 / 5160 | 0.0 / 0.0 | 51.1 / 52.2 / 48.5 / 49.4 s | 50.30 s |
| `--pool-rows 32` | 610 MiB | 4085 / 4165 / 2150 / 2170 | 60.4 / 59.6 | 22.5 / 20.1 / 19.8 / 24.9 s | 21.84 s |
| `--pool-rows 64` | 1183 MiB | 3335 / 3368 / 1977 / 2003 | 67.7 / 67.4 | 21.3 / 19.8 / 20.6 / 20.0 s | 20.42 s |
| `--pool-rows 148` | 2690 MiB | 3175 / 3217 / 1976 / 2003 | 69.2 / 68.8 | 21.4 / 21.9 / 22.0 / 19.5 s | 21.20 s |
| `--pool-rows 0` | 36 MiB | 10320 / 10320 / 5160 / 5160 | 0.0 / 0.0 | 49.0 / 49.5 / 48.8 / 48.8 s | 49.04 s |

The staged column is the cumulative counter, so it carries the decode row's 80 draws on ranks 0 and 1
and 40 on ranks 2 and 3 as the 512-token tables do. The prefill's own draw count falls **exactly 4×**:
the control stages 10320 rows on rank 0 here against 41040 over the same forty layers at 512 — **258
draws a layer against 1026**, of which the last two a layer are the decode row's, so the prefill itself
makes 256 a layer at 128 tokens against 1024 at 512.

**Prediction 2 holds. The pool is 2.3–2.4× here, not 4.0–4.2×.** The control's two sittings come back
50.30 and 49.04 s, 2.5% apart, and against their mean of 49.67 the three pooled legs read 21.84,
20.42 and 21.20 s — 2.27×, 2.43× and 2.34×. The mechanism reproduces; the size of it is a property of
how much staging the pass does, which is what the three knobs on this page all have in common.

**Prediction 1 holds in its second half and the first half is where it fails.** Below the floor a
width does stage strictly more — 32 rows stages 4085 on rank 0 against 64's 3335 and 148's 3175, **+23%
and +29%**, with the coverage column reading 60.4% against 67.7% and 69.2% — but the floor is **not**
near 40. At 148 rows rank 0 stages 3175 rows over the pass, and over forty layers that is **~79
distinct experts a layer against the ~142 distinct a 512-token pass draws** where a floor that scaled
with the draw count would have been ~36. The two widths at and above it are within 5% of each other
(3335 and 3175) and the one below is 23% over both, so ~79 is a floor and not a slope on this pass:
64 rows is already more rows than a layer asks for, and 32 is below the line.

**The floor is sub-linear in the draw count because the routing is concentrated, and that is the same
concentration the 512-token sweep ran into from the other side.** A layer's 258 draws at 128 tokens
over 384 experts would cover ~188 distinct if the gate were uniform — `384 · (1 − e^{−258/384})` — and
a 512-token pass's 1026 draws would cover ~357. Measured, they cover ~79 and ~142. Both are far below
the uniform estimate, and the consequence is that four times the draws buys 1.8× the distinct experts
rather than 4×: the tail of the routing accrues new experts slowly, which is why a width sized to this
prompt is neither comparable to a quarter of the 512-token width nor useful as a fraction of anything.

**Everything else the mechanism is measured on holds at this length too.** The identity does:
`pool_staged − pool_evicted` is the width exactly in all three pooled legs (4085 − 4053 = 32,
3335 − 3271 = 64, 3175 − 3027 = 148), so the eviction counter remains the one that says whether the
width was the binding constraint. And the decode column separates nothing **at one token**, as
before: 0.710–0.757 s a token across all five sittings, with the two controls — one configuration —
at 0.757 and 0.712, 6% apart, and every pooled leg inside that band. That is a one-step decode, which
is the width at which this pool cannot show anything at all; the same probe at the same length with
256 decode steps a leg reads 0.780/0.750 against 0.528/0.566, which is [the subsection
below](#the-pool-is-a-decode-lever-and-a-one-token-decode-is-the-measurement-that-hid-it).

**What it says about sizing is that the width belongs to the prompt.** At 128 tokens 64 rows buys the
prefill 148 rows buys — 20.42 against 21.20 s on the rank mean, inside the drift — for 1183 MiB of
arena instead of 2690, which is the same conclusion the 512-token sweep reached at 148 rows against
192 and 288. There is no width a *prefill* can be handed as a constant, so what the two sweeps
together establish is the shape of the curve the flag moves along, and that the mechanism's worth is
the staging it removes and nothing else; the size the launcher ships is [the decode's knee, not this
table's](#the-launchers-default-is-288-and-the-acceptance-sitting-behind-it).

Reproduce with:

```bash
# the width against a quarter-length pass: A B C D A at `--length 128`, control at both ends so the
# node's own drift lands on the column it would otherwise be credited to
/tmp/run_poolshort.sh
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_v41_hot_ab.py --compare \
  /tmp/ps_1_p0.pt.r0 /tmp/ps_4_p148.pt.r0
```

### The pool is a decode lever, and a one-token decode is the measurement that hid it

**Every decode column on this page is one token wide, and that is the one width at which a pool cannot
show anything.** The paragraph above states the mechanism as if it settled the question — a decode
step asks a layer for one row's worth of experts and nothing in it repeats — and the measurement it
rests on is a two-token decode added to a 512-token prefill. The first half is true of *a step*. It is
not true of a *generation*: the pool outlives the step, so the step after it asks for much the same
experts, and a cache that amortises over steps is invisible to a column read at step one. This
subsection is the same probe, the same four cards, the same bank and the same arena with the decode
run long enough to be measured: **256 steps a leg, A-B-A-B, control at both ends.**

| leg | `--expert-pool-rows` | prefill, rank mean (ranks 0-3) | decode | tok/s | draws / staged, rank 0 | decode-phase hit | arena | cuda |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 600 | **16.90 s** (18.6 / 16.5 / 16.2 / 16.3) | **0.528 s** | 1.89 | 30720 / 14303 | 45.5% | 10794 MiB | 18.76 GiB |
| 2 | 0 | 46.38 s (47.4 / 48.0 / 45.0 / 45.1) | 0.780 s | 1.28 | 30720 / 30720 | 0.0% | 36 MiB | 9.71 GiB |
| 3 | 600 | **20.02 s** (21.8 / 19.4 / 19.8 / 19.1) | **0.566 s** | 1.77 | 30720 / 14417 | 44.9% | 10794 MiB | 18.76 GiB |
| 4 | 0 | 42.40 s (42.6 / 42.8 / 42.6 / 41.6) | 0.750 s | 1.33 | 30720 / 30720 | 0.0% | 36 MiB | 9.71 GiB |

The controls are 0.780 and 0.750 s a token — 765.0 ms on the mean, 3.9% apart — and the pooled legs
are 0.528 and 0.566, 547.0 ms on the mean, 6.9% apart. **1.399×, and the two configurations do not
overlap on either end of the sitting.** The prefill moves with them and by more — 44.4 against 16.9
and 20.0 s on the rank means, 2.4× — which is the 4.0–4.2× mechanism above, read at a prompt a
quarter the length. The classification is the logits: **top-32 ids in position at
`|dlogit| 0.000e+00` on all four ranks of all four legs.** A pool is a cache, not an approximation,
and this sitting reads the last prefill position's logits the same way the six-sitting A-B-C-C-B-A
did.

**The win is the hit rate and not the length of the run, which is why the phases have to be split.**
The run-level `% resident` column mixes a prefill — where a layer's whole draw is in flight and the
pool is at 69–86% — with a decode, which is the question here. Rank 0's own log lines separate them:
the prefill prints its own staged count, and the run's total minus it is the decode phase's.

| prompt | decode steps | pool | control | pooled | speedup | decode hit, r0 | prefill, rank mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 0 | 0.743 s | — | — | 0.0% | 5.95 s |
| 8 | 64 | 600 | 0.743 s | **0.451 s** | **1.647×** | **61.3%** | 5.82 s |
| 8 | 128 | 600 | — | 0.333 s | — | 86.2% | 5.23 s |
| 128 | 64 | 0 | 0.706 / 0.765 s | — | — | 0.0% | 38.02 / 46.48 s |
| 128 | 64 | 300 | 0.7355 s (pair mean) | 0.601 s | 1.224× | 29.9% | 18.85 s |
| 128 | 64 | 600 | 0.7355 s (pair mean) | **0.557 s** | **1.320×** | 44.6% | 20.20 s |
| 128 | 256 | 0 | 0.780 / 0.750 s | — | — | 0.0% | 46.38 / 42.40 s |
| 128 | 256 | 600 | 0.765 s (pair mean) | **0.547 s** | **1.399×** | 45.2% | 16.90 / 20.02 s |
| 512 | 64 | 0 | 0.734 s | — | — | 0.0% | 190.12 s |
| 512 | 64 | 600 | 0.734 s | **0.564 s** | **1.301×** | 42.9% | 44.67 s |
| 512 | 128 | 0 | 0.764 s | — | — | 0.0% | 189.30 s |
| 512 | 128 | 600 | 0.764 s | **0.565 s** | **1.352×** | 43.8% | 45.62 s |
| 512 | 256 | 600 | — | 0.537 s | — | 44.5% | 43.45 s |

**The decode phase's hit rate is 42.9–45.5% at every prompt length, and 61.3% at eight tokens.** The
prompt length does not change it; what it changes is how much of the run is prefill, which is what
moves the run-level column the earlier tables print. The 8-token row is the outlier and it is the
mechanism, not noise: an eight-token context routes through a narrower set of experts, so the pool
covers 61.3% of its decode draws instead of ~45% and the same width is worth 1.647× instead of
1.30–1.40×. Two rows in that table have no control beside them — `8 × 128` and `512 × 256` — so they
are the trend of the pooled column and not speedups, and the 8 × 128 leg's 86.2% is a hit rate from a
run whose prefill is only 640 of its 10880 draws.

**In the decode window a hit is worth 5.0–6.4 ms of the step, and that is what prices the lever rather
than the hit rate alone.** Divide each pair's saving by the rows a step stops staging (the decode
phase's staged count over its step count, both on rank 0) and the eight sittings land in one narrow
band: **292 ms over 49.0 rows 5.96, 178.5 over 35.7 5.00, 134.5 over 23.9 5.62, 218.0 over 36.2 6.03,
170.0 over 34.3 4.96, 199.0 over 35.0 5.68, 227.0 over 35.6 6.38 ms a row**, at 17.93 MiB a row. That
is 2.9–3.9 GiB/s of effective source rate *inside the step*, against the 14 GiB/s the same copy loop
reaches when it is replayed on its own. The two legs at their own measured rates do not reach it
either: `_stage` in situ is 2.48 ms a row (198.5 ms over 40 calls, two rows a call), the row's H2D is
1.67 ms at the 10.47 GiB/s a card measures, and whether that second leg is chargeable depends on
whether the profiler's 16.0–17.8 ms `_upload` is the DMA or its issue — so the legs price a hit at
2.7–4.2 ms against the 5.0–6.4 the decode phase drops. **The difference is the loop and not the
bytes**, which is the same conclusion `_stage`'s own docstring reaches: "It is not a faster `copy_`:
14 GiB/s either way." A row is 12 `copy_` through `at::parallel_for` and a set of non-blocking H2D
issues, and a step that stops staging 36 of them stops paying 432 fork-joins; the per-row cost of that
is the term Lever 1 measured and could not attribute, and a profile of the pooled step is what would
close it. The figure the lever's price rests on is the measured 5.0–6.4, not the decomposition.

**Both figures above are decode-window figures, and that is a limit of the instrument rather than of
the mechanism.** Every one of the eight sittings divides a *decode* phase's saving by the rows that
phase stopped staging, because only the decode has a per-step wall to divide by; a prefill's saving
arrives at a different step count and the same division has not been run on its rows. So read
5.0–6.4 as what a decode step stops paying, and do not carry it to a prefill row or to "a hit" in
general.

**Half the arena buys 69% of the win, and that is the sizing datum.** At 128 tokens and 64 steps the
same sitting carries `--expert-pool-rows 300` against 600: 0.601 against 0.557 s a token, a decode hit
rate of 29.9% against 44.6%, for 5415 MiB of arena against 10794 and 13.51 GiB of card memory
against 18.76. So the win is not linear in the width — the first 300 rows carry most of it and the
second 300 carry the rest — and the arena a decode wants is one it can fill and refill, not the
prompt-wide width a prefill wants. This is the one place the two readings of the mechanism disagree
about sizing, and it is where the shipped default comes from: a prefill's width is set by a layer's
distinct experts, which rises with the prompt, and a decode's is set by the working set of the experts
a generation re-draws, which is flat in the prompt length and is what the 42.9–45.5% column above is.
So the width a prefill would hard-code is the wrong one to ship — it scales with a prompt the run has
not seen yet — and 288 is sized to this flat column instead, between 300's 1.224× and 600's 1.399× and
under the arena charge that keeps 600 from being the default ([why
288](#the-launchers-default-is-288-and-the-acceptance-sitting-behind-it)).
The control's own step is also flat across the whole sweep — 0.706–0.780 s over 8 to 512 tokens of
context and 64 to 256 steps — so nothing in the ratio is the node's drift on the control side.

**What it costs, and what it does not change.** 10794 MiB of arena a card, which is the 18.76 GiB of
`cuda` against the control's 9.71 — and 22 GiB less 9.7 leaves the KV cache 3–4 GiB at this width,
which is the same charge the remaining-bottlenecks page measures as 9.05 GiB above the step's own. The
prefill column is the same mechanism
at a different step count and not a second effect. And the bank is on in every leg above, as it is in
every number on this page's device sections: `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, so `_stage` reads
`/dev/shm` and not `/mnt/data3`, which is what makes the control's 0.75 s a compute number rather
than a disk one.

Reproduce with:

```bash
# the A-B-A-B: 600 / 0 / 600 / 0 at `--length 128 --decode 256`, control at both ends so the node's
# own drift lands on the column it would otherwise be credited to
for pool in 600 0 600 0; do
  DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_hot_ab.py \
    --length 128 --decode 256 --threads 22 --hot-rows 0 --pool-rows "$pool" --out /tmp/pr_$pool.pt
done
# the prompt-length and step-count sweep behind the second table, one sitting each
sed -n '1,40p' /tmp/run_pooldepth.sh
# and the phase split, which is arithmetic off the run's own two counters
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/tab_pool.py
```

### A pool hit used to take a staging buffer, and the copy behind it was left uncovered

The two pinned buffers exist so that the host can stage row `k+1` while the card still reads row `k`.
The guarantee is the wait: `_take_buffer` hands out the next slot *and waits on the event that last
read it*, so what makes the wait free is that a whole **staging** has happened between the copy being
issued into that slot and the wait — not that a whole *row* has. The rotation used to advance once a
row, and against a pool that answers most of a pass's rows the two are not the same thing: at
`--expert-pool-rows 288`, a 512-token prefill resolves **20480 rows** and stages in **~141 a layer**,
so **72.5% of the rows move nothing at all**. Those rows still took a slot, so the rotation walked
through empty rows and handed out the slot the previous *working* row had uploaded immediately
before, with the copy it issued still in flight and no staging in between to cover it.

It is measurable as one number inside `_take_buffer`: **6.89 / 6.87 / 4.11 / 4.02 s** of a
**30.35 / 30.55 / 18.97 / 18.62 s** class wall — 21.6–22.7% of the pass, in a wait whose own DMA is
~2.5 s. The expected value from the counters alone is 512 rows a layer less 141 staging ones, one
1.71 ms DMA each, 40 layers: **7.0 s against the 6.89 measured**, a 1.6% match. It also explains the
one column that never showed it — the decode step recorded **0.00 s** here, because a decode row
misses by construction and so was the only configuration where the wait always had a row's staging in
front of it.

`_stage_misses` now returns before `_take_buffer` when a row has nothing to move, so the rotation
advances only over rows that stage. Consecutive working rows take consecutive slots, which restores
the guarantee by construction rather than by luck:

| | before | after |
| --- | ---: | ---: |
| `_take_buffer` calls, rank 0–3 | 20480 each | **4807 / 4872 / 3791 / 3740** |
| `_take_buffer`, ranks 2–3 | 4.11 / 4.02 s | **0.09–0.25 s** |

**The end-to-end gain is not cleanly separable from this host's noise, and the honest reading is the
mechanism rather than the percentage.** An interleaved A/B — `before`, `before2`, `after`, `before3`,
`after2`, back to back, same leg, the file swapped under the same path and `diff`-verified at the end
— gives **BEFORE 33.95 s mean over 12 rank-runs against AFTER 31.90 over 8**, i.e. −6%, but the two
ranges are 28.3–35.5 s and **they overlap**. The un-fixed legs' own `_take_buffer` ranges **0.11–9.92 s
within the same code**, and a fixed leg still reads 6.26 s on one rank, so the column is worth less
than the effect it is supposed to show. What does not move in either direction is `_stage` (16.91 /
17.12 / 10.16 / 9.97 s) and `_upload` (2.16 / 2.18 / 1.53 / 1.50): **the byte cost is the floor and
the wait was only ever a fraction of it.** And the explanation for that is this host's own shape — the
four ranks share one memory system, so rank 0 waiting on an event is time the other three are spending
on their own staging, and removing rank 0's exposed wait relocates contention rather than removing
work.

The change is bit-exact, which is what makes it a pipeline fix and not a numerical one: across all
five legs, on all four ranks, **32/32 of the top 32 identical, worst `|dlogit| 0.000e+00`**, the
sampled token equal, and every counter equal — staged 5623 / 5673 / 3791 / 3740, `pool_staged`,
`pool_evicted`, and the batched path's 40 chunks / 40 calls.

Two things this pass ruled out on the way, both worth not retrying:

- **Caching the key strings and `checkpoint.packed()` in `_stage`.** The proposition was that
  `_stage`'s 501 µs a copy had ~230 µs of Python in it. `/tmp/probe_v41_stage_micro.py` decomposes it
  over 768 copies a round: `_key` + `scale_key` **0.5 µs**, `+ packed() + view` **8.3 µs**, `+`
  precomputed keys **7.7 µs**, the full `_stage` body **130.2 µs**, and `copy_` alone into pinned
  **77.3 µs**. So the Python is **~16 µs of the 501**, a ceiling of **~0.3 s** and not the 8 s the
  proposition implied.
- **The 501 µs being Python at all.** `/tmp/probe_v41_stage_contend.py` runs the identical loop solo
  and then four processes at once on disjoint expert bands: solo **133.8 µs a copy**, its `copy_`
  alone **78.6 µs (37.14 GiB/s in-pinned)** — and four at once **5353 / 5416 / 7258 / 6180 µs**, a
  **40–54×** spread that is asymmetric across the four and so is not a shared constant. What the four
  ranks contend for is the host's memory bandwidth, which is why a fix that removes a *wait* cannot
  move a wall that a *bandwidth* sets.

Reproduce with:

```bash
# the interleaved A/B on the guard: three un-fixed legs and two fixed ones, back to back, the file
# swapped under the shipped path between legs and diffed afterwards
bash /tmp/v41_guard_ab.sh
# the 501 us decomposed: key strings, the checkpoint lookup, and the copy alone
DEEPSEEK_V41_RESIDENT_EXPERTS=1 /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  /tmp/probe_v41_stage_micro.py
# the same loop solo against four processes on disjoint bands -- the contention measurement
DEEPSEEK_V41_RESIDENT_EXPERTS=1 /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  /tmp/probe_v41_stage_contend.py --tag solo
for i in 0 1 2 3; do DEEPSEEK_V41_RESIDENT_EXPERTS=1 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_v41_stage_contend.py \
  --tag "q$i" --offset "$i" & done; wait
```

### The probe's tokens are the sampler's draw, and the logit column is the parity check

**The tokens this probe records are not an argmax.** `Backbone.__init__` reads
`self.temperature = getattr(cfg, "temperature", 1.0)` (`modules.py:604`), the released V4.1
`config.json` carries no `temperature` field in either its top level or its `text_config` (checked),
and `sample(logits, self.temperature)` (`modules.py:668`) therefore takes its non-zero branch —
`probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)` (`modules.py:677`), an unseeded
Gumbel-max over the whole vocabulary at temperature 1.0. The probe calls `front(...)` directly, so the
model samples at its own default; the checked-in loop is the greedy one because
`src/models/deepseek_v4_1/generate.py:105` zeroes `model.temperature` around `_decode`, and the token
columns on this page that come out of *that* loop are argmaxes. Every token column taken with
`probe_v41_hot_ab.py` — the pre-fix table above, the four sittings under "the same configuration was
run four times", the pool's own runs — is a draw.

**The 128-token sweep is where that shows, and it is one payload of twenty.** The pool-32 leg's rank 2
drew `[1613, 270]` where the other three ranks of its own run and the other 19 payloads drew
`[14, 270]`. Nothing in the mechanism can move it and nothing in the logits did: that rank's top-32 at
the last prefill position is `[14, 1613, 305, 16]` at 27.8221 / 22.2702 / 20.9814 / 20.9665,
bit-identical to every other rank's and to every other run's — `torch.equal` on the stored top-32
values is true for all twenty of them, not merely close — and the pairwise comparison over all five
sittings × four ranks is **32 of 32 in position at `|dlogit| 0.000e+00`**. The top-2 margin is
5.55 logits and at temperature 1.0 the draw is `logit + Gumbel(0, 1)`, so 1613 wins about
`e^{−5.55}` = 0.39% of the time on this row, with its closer competitors (305 at 20.98, 16 at 20.97,
295 at 20.96) adding about a tenth of a percent each and the tail beyond them a little more — call it
1% a draw that the argmax is not the top-1. The 40 sampled tokens those twenty payloads hold would
carry 0.4 deviations at that rate, so one of them is the rate's ordinary outcome rather than a second
defect to chase.

**So the token columns are a weak instrument and the logit columns are the ones the claims rest on.**
The rate above is what makes an agreeing token column evidence about a margin rather than about
reproduction, and it cuts both ways: the pre-fix table's four ranks disagreeing with each other is
worth reading only because the logits behind it moved by 4.717 and 6.787 against a 2.915e+01 maximum,
which is far outside anything the sampler does. Every parity statement on this page — the six-sitting
pair, the pool's runs, the two post-fix legs — is the `|dlogit|` comparison, and the tokens are
reported beside it because they were recorded, not because they decide anything.

### The launcher's default is 288, and the acceptance sitting behind it

**The width is a construction-time decision, and that is why the default is a number and not a
per-run choice.** `ResidentSet.arena_rows` is `rows_per_card + hot_rows + pool_rows`, the arena is
allocated at that height once, and `DeviceRoutedExperts.__init__` accepts a shared set only when its
own width matches and raises otherwise — so nothing raises the pool after the fact, and the number the
launcher hands `load_backbone` is the number the run gets. The library level is deliberately
unchanged: `load_backbone(expert_pool_rows=...)` still defaults to 0, because it cannot tell whether
its caller has an arena to spend, and `src/cli/generate_v41.py` is the caller that can. It also
resolves the flag to 0 on the host path, where `--expert-device` is unset and there is no arena to
pool rows in, so the size becomes the off state there instead of a number the loader would warn about
and then ignore.

**The direction of the default is forced by a coupling, and the size of it is what the acceptance
sitting measured.** The pool's own two sweeps price the width from opposite ends: a prefill saturates
at ~148 rows (a 512-token pass stages 5700 rows at 288, 5761 at 148 and 6579 at 96, so the rows above
148 are bought for the decode), and a decode keeps paying (1.224× at 300 rows, 1.399× at 600). 288
sits above the saturation and below the point where the arena starts eating the KV cache — 5200 MiB of
it, 13.3 GiB of a 22 GiB card, against 600's 10794 MiB and 18.76 GiB — which is the trade a *default*
has to make rather than the one a named flag on a decode-heavy short-context run should. 600 stays
reachable by name. What is not arguable is that 0 would not be neutral: `expert_batched` cannot run
without a pool, because the batched call reads each arena row it is handed as one expert's bytes for a
whole chunk, so `--expert-pool-rows 0` drops the prefill's two best-measured mechanisms at once.

One configuration a process, a real 512-token prompt, the resident bank on in all of them, the default
and its off state run as a pair a header apart so a drift on the node lands across the pair rather
than inside it, and the separating leg added in a second sitting:

| leg | `--expert-pool-rows` | batched | 512-token prefill | pooled rows staged, ranks 0-3 | call shape |
| --- | ---: | --- | ---: | --- | --- |
| default | 288 | yes | **21.28 s** | 3997 / 4078 / 2643 / 2709 of 41040 / 41040 / 20520 / 20520 (90.3–87.1%) | 40 calls, 20480 rows, **512.0 rows a call** |
| control | 0 | — | **179.67 s** | every draw | 20480 calls, one row each |
| the missing point | 288 | no | **38.40 s** | 3997 / 4078 / 2643 / 2709, the same | 20480 calls, one row each |

**So the third leg is the one that separates the two mechanisms, and it is the leg the control
cannot be.** `--expert-pool-rows 0` drops the batch with the pool, so the default's own pair prices
both at once — **179.67 against 21.28 s, 8.44×** — and only the third leg says which is which: the
pool is **4.68×** (179.67 → 38.40 s) and the batch is **1.80×** on top of it (38.40 → 21.28 s), which
is 17.12 s over 20,440 fewer calls, **0.84 ms a call**. The staged rows are equal across the two
288-row legs to the row, so the batch half is the call count and not the floor, as
[the batch shape's own A/B](deepseek_v4_1_flash_remaining_bottlenecks.md#lever-5--the-prefill-is-a-batch-shape-problem-and-the-fix-is-in-27-at-512-tokens)
already had it.

**The decode is priced at the default's own width rather than at 600's.** A-B-A-B at 128 tokens and
256 decode steps a leg, the control at both ends, per-row so that this A/B isolates the pool:

| | pool 288 | pool 0 |
| --- | ---: | ---: |
| decode, s a token | **0.601 / 0.578** | **0.769 / 0.769** |
| 128-token prefill, rank mean | 20.2 s | 50.8 s |
| arena | 5200 MiB | 36 MiB |
| `cuda` | 13.3 GiB | 9.7 GiB |
| pooled share of the run's draws | 44.6% / 52.5% | 0.0% |

**0.5895 against 0.769, 1.305×, 180 ms a token**, for 5200 MiB of arena and 3.6 GiB of the card, and
the top-32 logits are identical at `|dlogit| 0.000e+00` on rank 0 with both counters equal. The two
controls read 0.769 twice, to the digit, which is what makes the 1.305× readable; the two pooled legs
differ by 3.8% between themselves, so quote the pair means and not one of them. The 0.601 s a token at
288 rows is the page's own 300-row column to the digit — 288 is twelve rows under that sitting — and
the control at 0.769 sits inside the 0.706–0.780 band the earlier 600-against-0 sitting recorded, so
this is the same measurement at the width that now ships and not a new one.

**Two limits on the numbers above.** The 512-token prompt is one paragraph repeated thirteen times, so
its routing is more concentrated than a document's: the pool answers 90.3% of rank 0's draws where the
512-token width sweep above answered 86.1%, and it stages 3997 distinct rows against that sweep's
5700 — which is why 21.28 s here is below the 34.75 s the same configuration records in
[the batch shape's own A/B](deepseek_v4_1_flash_remaining_bottlenecks.md#lever-5--the-prefill-is-a-batch-shape-problem-and-the-fix-is-in-27-at-512-tokens).
Within the sitting the A-B is clean; across sittings the prompt is the difference. And the batch's
share is larger here (**−44.6%** against the recorded −27.0% at the same two configurations), which
is 0.84 ms a call against 0.63 — a host-state difference the two sittings cannot separate, so read the
batch's worth as a band rather than as either figure.

Reproduce with:

```bash
# the acceptance: four legs, one configuration a process, default/off then default/off. The prompt is
# 512 tokens exactly and re-tokenizes to itself; --threads 22 because torchrun sets OMP_NUM_THREADS=1.
bash /tmp/v41_default_accept.sh      # default_long and off_long are the two legs of the table above
bash /tmp/v41_default_accept2.sh     # the third leg, and the decode A-B-A-B at the default's width
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_v41_hot_ab.py --compare \
  /tmp/dd_288.pt.r0 /tmp/dd_0.pt.r0
```

## What this does not do yet

All of these are separate measurements rather than separate opinions.

- **Not caching across rows on the device was the first of these, and it is now two knobs, both
  measured.** See
  [the resident set](#a-per-layer-resident-set-is-worth-27-on-a-prefill-and-it-is-the-fill-that-pays-for-it)
  and [the pool](#the-pool-spends-the-same-arena-on-what-the-pass-draws-and-its-first-key-answered-the-wrong-layer):
  `--expert-hot-rows` keeps a layer's hot experts in the shared arena and is worth 2.6–2.7× on a
  512-token prefill; `--expert-pool-rows` spends the same arena on the draws the pass actually makes
  and reads **43.7 s against the set's 48.5 s** at the two mechanisms' own 148-row width — the same
  2690 MiB either way, and the pool with no pinned block at all, which is where the set's 2654 MiB
  go.
- **The batch shape was the follow-on, and it has landed without moving the floor.** `--expert-batched`
  resolves the whole pass before staging any of it and then issues `moe_multi_token_fp4_forward` once a
  chunk — one slot per distinct expert the chunk hit, its tokens contiguous — instead of
  `moe_single_token_fp4_forward` once a row. The staging is **unchanged**: 512 tokens at 288 rows stage
  5623/5673/3791/3740 rows a rank batched, exactly what the per-row path stages, and the pool's
  took-in/evicted columns with them. What collapses is the call count — 512 rows a layer becomes **40
  calls, one a layer, 512.0 rows a call** — and that is worth **47.63 → 34.75 s a rank (−27.0%,
  10.75 → 14.73 tok/s)** with the top-32 logits bit-identical on all four ranks. So the floor is still
  *a layer's distinct experts* and the width table above still stands: `--expert-pool-rows 148` is
  still sized to a 512-token prefill because 148 and 288 stage the same 5700 rows and 96 stages 6579
  ([the width table
  above](#the-pool-spends-the-same-arena-on-what-the-pass-draws-and-its-first-key-answered-the-wrong-layer)),
  and a shorter pass still lowers the floor by less than the draw count falls —
  [a quarter-length pass puts it at ~79 experts a layer and the width that sits on it at 64 rows
  rather than 148](#a-quarter-length-pass-moves-the-floor-and-by-less-than-the-draw-count-moves). The
  path is default on and falls back to the per-row call without a pool, since the batched call reads
  each arena row it is handed as one expert's bytes for a whole chunk; the full A/B and the chunk rule
  are on [the bottlenecks
  page](deepseek_v4_1_flash_remaining_bottlenecks.md#lever-5--the-prefill-is-a-batch-shape-problem-and-the-fix-is-in-27-at-512-tokens).
- **The bank removes the disk from `_stage`, not the copy out of it.** With
  `DEEPSEEK_V41_RESIDENT_EXPERTS=1` the step is 782.9 ms on an emptied page cache against 17.01 s
  without it, so the 722–747 ms headline holds on a host that has forgotten the checkpoint —
  [measured above](#the-resident-bank-takes-the-disk-out-of-_stage-and-not-the-copy-into-pinned). What
  it does not remove is the join: the segment is not the pinned arena, so a row is copied into that
  arena whichever source it came from, and `_stage` is still 242 ms of a 458 ms class. The pipeline
  above does not take that copy away either — it hides 2.0 ms a row-layer of a 3.75 ms drain and gives
  back 0.95 ms of it in staging, which is a 1.10× and not the 7.3-against-1.0 the ceiling suggested.
- **The Engram tables are in the segment, and a banked prefill's gather is priced: 1.4–2.3 ms a
  table, 3–4 ms for both.** `rows` reaches the segment under the same environment variable
  (`resident_bank.parse_engram_key`), so the 253.4 s/table cold path the host page records is not on
  a banked run's route, and the `read_bytes` acceptance above puts a 512-token prefill's 12,288
  gathers a table at zero bytes off the device. **Zero disk reads turned out not to be a cost either
  at this size** — the gather is 0.22–0.26 ms of the 1.4–2.3, and `dequantize_rows` over the
  gathered rows is the rest ([measured
  above](#zero-disk-reads-is-also-at-this-size-near-zero-cost-1423-ms-a-table-for-a-512-token-prefill)).
  What that does not settle is a corpus that moves onto n-grams it has not hashed before: the ids
  are uniform over the whole table here and a real one is not, and this measurement does not say
  what the hot set is.
- **The determinism check the plan lists was run, and what it found was a real bug rather than a
  tolerance.** The same configuration twice — `--length 512 --hot-rows 148 --decode 4`, `q512_1a`
  against `q512_2a` on rank 0 — keeps **all five greedy tokens** and **30 of the top 32 ids**, and
  moves **30 of the 32 out of position**: positional agreement is 2 of 32 there, 3 of 32 on the
  192-row repeat, and 2 to 10 of 32 across every pair recorded on this page, the pairs that differ by
  a whole configuration included. The band is not association order either (5.960e-08 is what that
  costs on this path): the same id's logit moved by **0.46 median and 1.21 at worst** on that pair,
  and by up to **2.93** on the other, against a leader at 28.8. Read positionally, slot against slot,
  the same pair is 0.195 median and 0.875 at worst, smaller precisely because the permutation is most
  of the positional difference. It was the first hard evidence that the path did not repeat itself,
  and everything below is what it led to — **the fix is one missing barrier in the sparse-attention
  kernels, and it takes this pair to 32 of 32 in position at `|dlogit| = 0`.**
- **The same configuration was run four times in one sitting, no two of them routed alike, and that is
  what localized the drift to the gate's input.** `d1`, `r1`, `r2` and `d2` are one configuration —
  `--length 512 --hot-rows 148 --decode 4` — and they disagreed on `expert_rows` (1990, 1963, 2001,
  1975) and on `filled_rows` (3946, 3969, 3904, 3900), while all four keep the five greedy tokens
  `[2413, 21779, 14, 305, 270]`. The pair this page was written around, `q512_1a` against `q512_2a`,
  is the same disagreement in another sitting (2008 against 1983 staged, 3916 against 3909 filled).
  `_fill` and `_hot_rows` are pure functions of the layer's own routing — the count is of that
  layer's own draws and nothing is carried between layers — so runs that filled different counts
  selected different experts: the gate asked for experts the others did not. **Re-run after the fix,
  the same pair agrees on every one of those numbers, on every rank**: rank 0 stages 2008 and fills
  3935 in both legs (`/tmp/pf_p1.pt.r0` against `/tmp/pf_p2.pt.r0`), ranks 1, 2 and 3 are identical
  run to run as well, and the counters that had four different values now have one. The one kernel in
  the chain whose nondeterminism *is* documented turns out not to be on it: the op the device path
  calls is `moe_single_token_fp4_forward` (`src/csrc/cuda_kernel_impl.cu`), and its default is the
  fixed-order `partials` reduce (`DEEPSEEK_MOE_DETERMINISTIC_REDUCE` defaults to 1), whose `atomicAdd`
  twin is the one whose own comment records `topk>=3` diverging on 60 of 60 repeats. With
  `n_activated_experts` at 6 this path is in exactly the regime that default exists for.
- **It is not the collective, and all three ways of asking that agree.** Every rank is bitwise
  identical to every other *within* a run: on four recorded payload sets, ranks 1, 2 and 3 against
  rank 0 are 32 of 32 positional, 32 of 32 in set, worst |dlogit| **0.00e+00**. On its own that
  excludes a rank-asymmetric race and very little else, because an all-reduce hands every rank the
  same answer by construction — the thing it cannot see is the collective choosing a different
  reduction order on a different run, which is why the second test pins it rather than observing it.
  Two of the four runs carried `NCCL_ALGO=Ring` (accepted, with no `Invalid value` warning), and the
  pinned pair drifts neither more nor less than the unpinned control: all six pairs among the four
  runs land between **2 and 8 of 32 positions at a median 0.336 to 0.796 logits**, `r1` against `r2`
  at 6 of 32 and median 0.407 sitting between the control pair's 6 of 32 at 0.336 and the widest
  pair's 2 of 32 at 0.796. `NCCL_ALGO` is not the whole order, though — the protocol and the channel
  count are left free, and the partition a ring reduces over is a function of the channel count — so
  the pin that closes this is the one that fixes those too. **It does not close it.** Two more runs
  under `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1` — one channel on
  a ring under the simple protocol, which leaves NCCL no freedom in the order its arithmetic happens
  in — are **6 of 32 positions at a median 0.243 and 0.970 at worst**, and they are the *closest*
  pair of the six, nearer each other than two of the four unpinned runs are. The pin was accepted
  (no `Invalid value` warning on either leg) and it bought nothing. What settled it is that the
  pinned pair also **routed differently** — 1988 staged and 3927 filled against 1952 and 3922 — so
  the order the collective reduces in is not what the counters move with. A collective is the same
  answer on every rank and that is all it is; the drift is in the host arithmetic the four ranks
  each do privately, which the next measurement goes after directly.
- **The first block diverges with its input bitwise identical, and the seed is a swapped pair of
  tied experts.** `/tmp/probe_v41_routes.py` records, per layer, a bf16 **digest** of the block's own
  input and output and the gate's own `indices` captured in place — a probe that re-derived the
  routing would be checking its own arithmetic — and ran twice at `--length 512 --hot-rows 0`, so the
  resident set is out of the picture by construction and the expert source is the checkpoint it always
  was. Layer 0 is the answer: **input digest equal, last-row digest equal, output digest not**, and of
  its 3,072 route entries exactly **2 differ, on 1 row of 512** — every row keeps the same six
  experts. The row is 64, and the two runs read `[40, 134, 315, 204, 62, 41]` and
  `[40, 134, 315, 62, 204, 41]`: the same six experts, with **204 and 62 exchanging slots 5 and 6**.
  What that does not license is the reading that the gate saw two *equal* scores and returned them in
  the other order: the block's input is the attention's input, and the gate reads the attention's
  *output* — which moves between two calls on one frozen activation in the same process, one layer
  further down. The gate therefore scored 204 and 62 differently, the two landed in the other order,
  and the (weight, expert) pairs were then summed along the row in the other sequence. Two float sums
  of the same six terms that differ only in order differ in the last bits, so the block output
  differs, and the page's own reproduction is the amplification: from layer 1 on the input is
  already different and the divergence compounds without any second cause — **45 experts whose counts
  disagree at layer 2, 64 at layer 3, 85 at layer 4, and 172 of the layer's 384 by layer 38**, against
  none at all at layer 0. Layer 1 shows both halves of the same coin: four rows differ, three of them
  pure swaps (94↔63, 139↔270, 109↔33) and one
  a **real change of expert** (127 against 252), which is a near-tie not on the ordering boundary but
  on the cut itself. The whole route is the same six experts in the other order, or a 127th expert
  swapped for a 252nd, and 40 layers of that is the 2-to-8-of-32 positional spread and the 0.24-to-1.25
  logit band every pair on this page shows. The counters move for the same reason: `_fill` reads the
  layer's own routing, and one row's worth of a swapped expert is one more or one fewer row staged.
- **The producer is a missing `__syncthreads()` in the sparse-attention kernels — one shared buffer,
  two reductions, and no fence between them.** Every one of the six sparse-attention kernels in
  `src/csrc/cuda_kernel_impl.cu` reduces twice into the same shared `float*` and reuses index 0 for
  both. The max tree ends, its result is read out as `max_score = fmaxf(reduce[0], attn_sink[h])`
  (single-head; `max_score0`/`max_score1` on the two head-pair kernels), and then the *denominator*
  pass opens with `reduce[tid] = local_denom` — index 0 included — with nothing between the read and
  the write. A thread that loses that race reads a denominator where it expects the maximum, and every
  `expf(scores[t] - max_score)` after it is computed against the wrong constant.
  - The six sites, and no others: `prefill_sparse_attn_kernel` (the read at
    `cuda_kernel_impl.cu:888`), `prefill_sparse_attn_headpair_kernel` (`:1004`/`:1005`),
    `fused_decode_sparse_attn_kernel` (`:1118`), `fused_decode_sparse_attn_wmma_kernel` (`:1229`),
    `flashinfer_style_sparse_attn_kernel` (`:1335`), and `flashinfer_style_sparse_attn_headpair_kernel`
    (`:1455`/`:1456`).
  - `compute-sanitizer --tool racecheck` names exactly that pair and nothing else: **Read at `+0x2d10`
    racing Write at `+0x3350`, and Read at `+0x2d30` racing Write at `+0x3330`, 16,384 hazards
    each**. With `-lineinfo` they resolve to the two `max_score` reads against the two
    `reduce[tid] = local_denom` stores. SASS agrees and rules out a compiler artifact: `BAR.SYNC 0x0`
    sits at `0x3360`, *after* both stores.
  - **Proof it is the cause and the whole cause.** The kernel text lifted byte-for-byte out of the
    repository and compiled standalone (`/tmp/probe_headpair_standalone.py`), 50 calls on one frozen
    input, in one process with one thread configuration — so OpenMP, thread count and the four-card
    collective are not variables in this test at all:

    | variant | outputs over 50 calls | worst max \|d\| |
    | --- | ---: | ---: |
    | as the source reads today, fence in place | **1** | **0.000e+00** |
    | that one fence deleted | 50 | 1.797e+00 |
    | same, before the fix, at `-O3` and no `--use_fast_math` | 50 | 1.984e+00 |

    The only difference between the first two rows is one `__syncthreads()`; the third row is the
    pre-fix source and settles the build flags — `setup.py` does compile this extension with
    `--use_fast_math`, but a plain `-O3` build drifts 50 of 50 on its own. The same command with the
    fence deleted reproduces the sanitizer's hazard report exactly (the same two addresses, the same
    16,384 each), and with the fence in place reports **0 hazards** and returns the same digest on
    every call.
  - **Neither of the two fixes the previous draft of this bullet proposed is the fix.** There is no
    tie to break: `/tmp/probe_v41_gate_ties.py --report` over 12,288 rows finds **0** rows holding two
    of the six returned weights at a bitwise-equal score, with `redo_mismatch = 0` everywhere; only 3
    rows carry a tie at the selection cut at all (6th and 7th bitwise equal, layers 3/38/39 and layer
    39). And the thread count is not the variable: `--threads 1` with `OMP_NUM_THREADS=1` still
    drifts, and the standalone test above has no OpenMP and no thread-count change in it whatsoever.
    A `(score, expert id)` tie-break and a single-threaded reduction would each have cost time and
    fixed nothing.
  - **One thing that looked like a second hazard is not.** Dropping the barrier between the score
    loop and the max tree changes nothing (1 distinct over 50 too), and it should not: with
    `for (t = tid; t < topk; t += blockDim.x)` on both the write and the read, every thread reads
    back only the entries it wrote itself, so that barrier is redundant by construction.
  - **Measured after the source fix and the rebuild.** The built module
    (`setup.py build_ext --inplace`) goes from **4 distinct outputs over 10 calls, worst max |d|
    5.972e-01 against a max |value| of 3.203e+00** to **1 distinct over 20 calls, 0 of 4,194,304
    elements moved** on the same probe, and the model's own repeat report goes from
    `sparse_attn: DIFFERS, 4 distinct over 10 calls … this is the producer` to
    `sparse_attn: same, the kernel is not the producer`, with every sub-op of layer 0's attention
    (`sparse_rope_out`, `einsum_wo_a`, `wo_b`, `out`) now `same` across ten repeats on all four ranks,
    on two separate legs. The fix is one `__syncthreads()` per site, in all six kernels, with a comment
    saying why; it is in the working tree on `perf/v41-dense-tree` and not yet on a branch of its own.
- **The second check the plan lists is now run, and it is the segment ids permuted.** Section 2b of
  `/tmp/probe_fp4_parity.py` calls the op with the *route order reversed*: the arena untouched, and
  `indices` and `weights` permuted together, so the six (expert, weight) pairs are the same set and
  the only thing that moved is the order the kernel reads them in — `[277, 128, 155, 137, 206, 251]`
  named as `[251, 206, 137, 155, 128, 277]` on layer 0. Reading the arena by arrival position instead
  of by `indices[route]` fails this by orders of magnitude, not in the last decimal. It does not:
  **`5.960e-08` against the identity-order call**, which is the same fp32 association class the EP4
  decomposition costs, and **`1.532e-02` of the output scale against `expert_forward` — the
  digit-for-digit figure the identity order gives**, argmax agreeing. The reduce is a fixed ascending
  route order precisely so that a permutation of the routes is nothing but a rounding re-association,
  and that is what the pair measures. What it does not add is a test of the *gate*: the segments it
  permutes arrive from a route that has already been shown to flip order, so a pass clears the kernel
  of an ordering dependence and not of being downstream of one.
- **The per-layer graph the move exists to enable is still not taken, and the number it was gated on is
  now in — as a count and a shape, not yet as a size.** The tree on the cards is **191.32 ms a decode
  step** with the step at **688.9**, of which the class is 414.45 — so three fifths of the step is the
  class's staging, and most of the tree's 191.32 is dispatch over small tensors: 12.10 ms of `hc_mixes`
  alone is six ATen launches on 24 numbers, and no kernel inside that op reaches them. The step launches
  **7,076 kernels** and only **47%** of them are the sixteen heaviest, so the tail is exactly the shape a
  graph addresses. What this sitting does not give is the prize: the profiler's device times do not
  survive a second sitting, so the upside is bounded by host work that has to be timed directly. The
  **111–122 ms** the phase clock leaves outside the named calls is the part of it that is currently
  unattributed, and the 88 `ncclDevKernel_AllReduce` calls a step are the part a graph would have to
  capture as collective nodes or leave outside it.

## Reproducing

```bash
# the kernel against the host expert on real activations, the EP4 decomposition, the same six experts
# with the segment ids permuted and the weights riding with them, and the 2/2/1/1-against-one-call
# kernel cost -- needs /tmp/v41_activations.pt from probe_capture.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_capture.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_fp4_parity.py

# the class against expert_forward, and world=4 against world=1 on the same activation
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_device_experts.py --world 4
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_device_experts.py --world 1

# staging, threading, pinning and H2D, each priced on its own
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_stage.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_h2d.py

# a whole step, phase by phase, twice through the same prompt so pass 2 is a warm page cache
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_device_cost.py --world 4
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_device_cost.py --world 1

# the same step with `_take_buffer` wrapped as well, which probe_device_cost does not do -- and the
# one that caught a cold page cache in its first pass, so read its `GiB/s` column before its `stage`
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_launch_cost.py --world 4

# one row of four cards, three ways, one change apart, and the three checked against each other
PYTHONPATH=. /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_launch_split.py --layer 0

# the checked-in loop on the device path, which is what the text above is produced by -- the
# `routed experts:` line it prints is the flag's own report that it did not fall back
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -u -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" \
  --max-new-tokens 4 --expert-device cuda --expert-world 4
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -u -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" \
  --max-new-tokens 4 --expert-device cuda --expert-world 1

# the same flags off, so the host path and the default -- this is what the whole-request comparison
# above is against, and the check that this work did not move it
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -u -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" --max-new-tokens 4

# the distribution behind the first three tokens, on either path -- this is what says the one token
# the two disagree about is a half-logit tie rather than a wrong answer
PYTHONPATH=. /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_first_tokens.py \
  --device cuda --world 4
PYTHONPATH=. /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_first_tokens.py

# the TP4 configuration: the tree on the four cards, one process per card, the same phase
# attribution as above so the two tables read line for line
torchrun --nproc_per_node=4 /tmp/probe_v41_tp4_e2e.py --lengths 8,128 --steps 8

# ... and the same step per op instead of per phase, with the tree on the cards -- this is the sitting
# `hc_mixes` is read off, and the one that says the host tree's `hc_mixes` row is a fallback and not a
# price. `PYTHONPATH` because `src` is a repo-root package and the probe lives in /tmp, so `sys.path[0]`
# is /tmp under `torchrun` exactly as it is under `python`. `--no-sweep` is the default here: the thread
# pool does not govern a tree on a card, so the sweep and the replay would re-time the same step
DEEPSEEK_V41_RESIDENT_EXPERTS=1 PYTHONPATH=/mnt/data1/dsv4_inference \
  torchrun --nproc_per_node=4 /tmp/probe_dense_tree.py --tree cuda --threads 22 --steps 4 \
  --out /tmp/dt_cuda.out

# the `hc_mixes` row above, taken apart: the fused kernel, the loop it replaced and the six ATen ops
# around them, on one idle card, no checkpoint and no second rank. The shape is the decode shape -- 1
# row, hc_mult 4 -- and it also runs 512 to show that neither implementation is arithmetic-bound
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_hc_split_card.py

# how many kernels the step launches: one more decode trajectory under a CUPTI window, the same
# `torch.profiler` instrument the single-block breakdown uses, rows split on `device_type` so the
# CUDA runtime API rows are not counted as launches. Same configuration as the sitting above, because
# the count is only comparable to the ~9,100 the sinkhorn loop cost if it is the same step. Every
# rank must run this pass and only rank 0 reports it -- three ranks returning early leaves rank 0 in
# an all-reduce with no peer, which is how the first attempt at this number died
DEEPSEEK_V41_RESIDENT_EXPERTS=1 PYTHONPATH=/mnt/data1/dsv4_inference \
  torchrun --nproc_per_node=4 /tmp/probe_dense_tree.py --tree cuda --threads 22 --steps 4 --launches \
  --out /tmp/dt_cuda_launches.out

# ... and once more, which is the point of the pair: the launch counts come out identical and the
# device-time columns do not, so a single sitting of this command cannot tell a count from a sample
DEEPSEEK_V41_RESIDENT_EXPERTS=1 PYTHONPATH=/mnt/data1/dsv4_inference \
  torchrun --nproc_per_node=4 /tmp/probe_dense_tree.py --tree cuda --threads 22 --steps 4 --launches \
  --out /tmp/dt_cuda_launches2.out

# ... with the experts staged from the resident bank instead of the checkpoint mapping. The bank is
# filled once by `resident_bank` and every later run attaches it in milliseconds; this flag is the
# whole wiring. Warm it is a wash, cold it is the difference between 782.9 ms and 17.01 s a step
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_tp4_e2e.py --lengths 8 --steps 8
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_tp4_e2e.py --lengths 128 --steps 4

# ... and the same step with the page cache deliberately emptied first, which is the 17 s one --
# `mincore_resident.py` counts resident pages, so run it before and after to see the two states
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/mincore_resident.py /mnt/data3/DeepSeek-V4.1-Flash
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/fadvise_drop.py /mnt/data3/DeepSeek-V4.1-Flash
torchrun --nproc_per_node=4 /tmp/probe_v41_tp4_e2e.py --lengths 8 --steps 8

# ... and the acceptance: `read_bytes` out of `/proc/self/io` around every phase, plus `/dev/shm`,
# the process's own RSS and `torch.cuda` host bytes. One process, four cards, a warmup pass before
# each measured prompt, so the prefill row is the second pass over that prompt by construction
DEEPSEEK_V41_RESIDENT_EXPERTS=1 /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  /tmp/probe_v41_resident.py --lengths 128,512 --steps 4 --threads 22

# what that acceptance's zero means as a price: the same segment attached read-only, 12,288 row ids a
# table drawn uniform over the whole 91.5 GiB, and the gather, the surrounding call, the dequant and
# a per-row loop each timed. No card, no checkpoint read past the 0.24 s layout scan, and it never
# fills the segment -- if `bank.ready` is there it attaches in 0.3 s
DEEPSEEK_V41_RESIDENT_EXPERTS=1 /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  /tmp/probe_engram_bank.py

# the checked-in TP4 launcher, which is what `--threads` is about
torchrun --nproc_per_node=4 -m src.cli.generate_v41 \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" \
  --max-new-tokens 3 --threads 22

# ... and the same launcher with a per-layer resident expert set, which is the prefill column's
# whole configuration. It reports its own hit rate and its capping at the end of the run
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 -m src.cli.generate_v41 \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" \
  --max-new-tokens 3 --threads 22 --expert-hot-rows 64

# the row loop, serial against one row deep, both orders in one process and alternated so the node's
# own drift lands on both columns. The class's own `forward` is the pipelined one; the probe defines
# the serial loop beside it, because that is a baseline and not a configuration. `--pinned 2,3,4` is
# the arena sweep that says a third and a fourth are worth nothing
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_tp4_pipe_matrix.py --lengths 32,128 --pinned 2 --threads 22
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_tp4_pipe_matrix.py --lengths 32 --pinned 2,3,4 --threads 22

# the resident set, four widths, one sitting, one prompt. `--out` takes a rank suffix, so the four
# ranks do not write the same file
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_hot_ab.py --length 512 --hot-rows 0 --out /tmp/s512_h0.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_hot_ab.py --length 512 --hot-rows 64 --out /tmp/s512_h64.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_hot_ab.py --length 512 --hot-rows 148 --out /tmp/s512_h148.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_hot_ab.py --length 512 --hot-rows 192 --out /tmp/s512_h192.pt

# ... and 148 against 192 again, twice, A-B-A-B, which is what separates the arena's effect from the
# node's: the same script at `--decode 4` with nothing between the two widths but the width. The
# wrapper waits on `WAIT_FOR=<pid>` first, because it wants the four cards to itself, and echoes the
# load average at each leg so the sitting can be judged
WAIT_FOR=0 /tmp/run_quiet_width.sh

# ... and the two logit columns against the control. Read all three: the set takes rank 0 from 41280
# staged rows to 5613 and ranks 2-3 from 20640 to ~1800, and it moves none of the 32 -- 32 of 32 in
# position at `|dlogit| 0.000e+00` of a max |logit| of 29.15, on every rank. So the columns separate
# the set's cost and nothing else; it is a cache, not an approximation
python /tmp/probe_v41_hot_ab.py --compare /tmp/s512_h0.pt.r0 /tmp/s512_h64.pt.r0
# ... and the same two legs with the compare attached, as one command, so the pair can be re-taken
# against a sitting rather than against the cards
WAIT_FOR=0 /tmp/run_quiet_configcmp.sh

# what the fill costs, which no counter has a line for: it wraps `_fill` and charges a
# `perf_counter` a layer, then reads the process's own RSS and smaps_rollup at four points
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_resident_fill.py --length 512 --hot-rows 64 --out /tmp/fill_h64.out
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_resident_fill.py --length 512 --hot-rows 148 --out /tmp/fill_h148.out

# --- the run-to-run drift, and the one barrier that caused it ---

# layer 0's attention re-run ten times on the activation it actually consumed, every sub-op digested.
# The repeats are inside one leg, so one leg is enough to name a producer; the second leg is for the
# case where all ten agree and the question becomes whether the *process's* conditions differ
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_gate_ties.py --length 512 --hot-rows 148 --threads 22 --detail-layers 1 \
  --repeat 10 --out /tmp/gt_r1.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
  /tmp/probe_v41_gate_ties.py --length 512 --hot-rows 148 --threads 22 --detail-layers 1 \
  --repeat 10 --out /tmp/gt_r2.pt

# ... and the tie this page used to blame for it. `--report` answers from the gate's own captured
# `indices` and scores, and prints `redo_mismatch` as the check that it recomputed them the way the
# gate did; zero rows in 12,288 hold two of the six returned weights at a bitwise-equal score
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_v41_gate_ties.py --report \
  /tmp/gt_r1.pt.r0 /tmp/gt_r2.pt.r0

# the kernel alone, twenty times: does `prefill_sparse_attn_headpair_forward` repeat itself on frozen
# arguments? It digests the arguments on every call too, so an instrument that is not in fact holding
# them fixed says so instead of blaming the kernel. Ten seconds the first time, three minutes after
PYTHONPATH=. /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_sparse_attn_repeat.py \
  --repeats 20

# where in the output it disagrees, and against a float64 reference: which rows, which heads, how many
# of the 512 lanes move inside a pair, and whether the calls straddle the truth or all sit on one side
PYTHONPATH=. /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_sparse_attn_where.py

# the same kernel text lifted byte-for-byte out of `src/csrc/cuda_kernel_impl.cu` and compiled
# standalone, so the answer cannot be a fact about the model: as the source reads today, with the
# max-read fence deleted, and with the *other* barrier deleted (the control that a stable result is
# not a harness that cannot see a race). The `nvcc` on `PATH` is 13.0 while this torch is a 12.4
# build, so the toolkit has to be pinned the same way the extension build pins it
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.5 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_headpair_standalone.py --repeats 50

# ... and the sanitizer that named the pair in the first place: 16,384 hazards on the max-read against
# the denominator store with the fence deleted, `0 hazards` and one digest on every call with it in
# place. `--unfenced` builds the variant; `-lineinfo` is what puts the addresses on source lines
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.5 \
  compute-sanitizer --tool racecheck --print-limit 4 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/run_headpair_once.py --unfenced
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.5 \
  compute-sanitizer --tool racecheck --print-limit 4 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/run_headpair_once.py

# the rebuild every model-level leg above needs before it can see the fix. `build_ext` builds *every*
# extension in `setup.py`, and `pocketllm_cpp` does not survive `TORCH_CUDA_ARCH_LIST=7.5` -- its
# qwen NVFP4 kernels are sm_80+ -- but `cuda_kernel` is linked and copied before that failure, so
# read the timestamp on `cuda_kernel.cpython-311-x86_64-linux-gnu.so` rather than the exit code
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.5 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python setup.py build_ext --inplace

# ... and the pair of full legs that closes it, the same configuration twice, compared on the counters
# and on the logits at every rank: 32/32 in position at `|dlogit| 0.000e+00`, and every counter equal
WAIT_FOR=0 /tmp/run_quiet_postfix.sh
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_v41_hot_ab.py --compare \
  /tmp/pf_p1.pt.r0 /tmp/pf_p2.pt.r0
```

Each takes about 90 s, most of it the ~70 s load; the standalone and sanitizer runs at the end take
minutes, because they compile CUDA from scratch, and `racecheck` serializes execution on top of that.
The probes that touch a card want all four
of them free. `probe_stage.py` and `probe_h2d.py` report whichever page-cache state they find, and
`probe_device_cost.py` prints a `GiB/s` column so the same is true of it and readable rather than
silent.

The host path stays the default: the device path needs `--expert-device` (or
`DEEPSEEK_V41_EXPERT_DEVICE`) and falls back to `CheckpointRoutedExperts` with one line on `progress`
if the extension is unloadable, the card is missing or the checkpoint's expert is laid out the other
way round. The *resident set*'s width, `--expert-hot-rows`, is off at 0 and is a knob rather than a
default for a measured reason: it is worth **2.6–2.7×** on a 512-token prefill, and the one sitting
that priced its decode against a no-set control put that at **10–11%** — so `N` here is a prefill knob
and the launcher that leaves it unset is the decode configuration (the decode column is also the one
the readings disagree on; see the resident-set section above). It costs no memory when it is zero —
the arena is the same two rows — and the flag prints its own hit rate, its capping and its misses when
the run ends. The **pool**'s width, `--expert-pool-rows`, is the other way round and ships at 288:
its decode column is the larger of the two and it is the mechanism the batched prefill runs on
([why 288](#the-launchers-default-is-288-and-the-acceptance-sitting-behind-it)).

**Read that fallback line, and run the device path from the `deepseek` environment.** The repository
root holds two builds of the same extension, `cuda_kernel.cpython-310-x86_64-linux-gnu.so` and
`cuda_kernel.cpython-311-x86_64-linux-gnu.so`, and `cuda_loader._find_built_extension` resolves by the
running interpreter's cpython tag: whichever one matches is loaded, silently, whether or not it is the
newer build. The 3.10 one predates the fp4 MoE ops, so a device-path run under the base conda
environment (3.10.10) reports `moe_single_token_fp4_forward is not available in the built extension`
on all 40 layers and quietly keeps the hosts' experts — a full run of the wrong configuration that
looks like a run. The `deepseek` environment is 3.11 and picks the 3.11 build, which carries both
`moe_single_token_fp4_forward` and `moe_multi_token_fp4_forward`; every command above that touches a
card is written with it for that reason.
