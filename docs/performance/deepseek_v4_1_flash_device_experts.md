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
numbers they produced are what this page records.

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

The dense tree stays on the host in this phase. It is 16.79 GiB, it works, and moving it is worth
0.4–0.6 s/token on its own; doing both at once would put two independent sources of divergence inside
one debugging session.

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

What that leaves is the honest headline: **the host's own dense stack is now the largest single term
of a device step**, 0.55–0.67 s against the 0.30 s of staging, and it is the same 0.51 s the earlier
coarse probe measured from the other direction. Moving it is worth 0.4–0.6 s/token on its own and is
[still not done](#what-this-does-not-do-yet).

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

## What this does not do yet

All three are separate measurements rather than separate opinions.

- **Nothing is cached on the device between rows.** The arena is the two rows this row needs and it
  is refilled every row, so a prefill of `n` rows pays 4.20 GiB `n` times: the 5-token prefill above
  stages 21.0 GiB, five times a decode step's traffic, because two tokens that route to the same
  expert each stage it. A wider arena plus `moe_multi_token_fp4_forward` — one slot per distinct
  expert the batch hit, its tokens contiguous — is the shape that fixes it, and it is a follow-on
  rather than a knob, because an arena size and an eviction policy only mean something once that
  measurement exists.
- **The staging does not overlap the launch.** A row is strictly serialized today: the host cannot
  stage row `k+1` until `_launch` has returned for row `k`. With the launch at 0.15 s and the staging
  at 0.30 s, a one-row-deep pipeline — stage row `k+1` while row `k`'s kernels run, drain row `k` at
  the top of row `k+1` — is worth up to the launch, and the isolated row puts a ceiling on it:
  **1,009.8 µs** of device work against the **7.3 ms** the same row's staging costs, so the device
  side would be fully hidden. It needs one more generation of the activation, the weights and the
  partials and it changes the shape of the row loop rather than any of its parts, so it is the next
  follow-on with its own measurement.
- **The dense tree is still host code**, now the largest single term at 0.55–0.67 s of the 1.06–1.14 s
  step — worth 0.4–0.6 s/token on its own.

## Reproducing

```bash
# the kernel against the host expert on real activations, the EP4 decomposition, and the
# 2/2/1/1-against-one-call kernel cost -- needs /tmp/v41_activations.pt from probe_capture.py
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
```

Each takes about 90 s, most of it the ~70 s load. The three probes that touch a card want all four
of them free. `probe_stage.py` and `probe_h2d.py` report whichever page-cache state they find, and
`probe_device_cost.py` prints a `GiB/s` column so the same is true of it and readable rather than
silent.

The host path stays the default: the device path needs `--expert-device` (or
`DEEPSEEK_V41_EXPERT_DEVICE`) and falls back to `CheckpointRoutedExperts` with one line on `progress`
if the extension is unloadable, the card is missing or the checkpoint's expert is laid out the other
way round.
