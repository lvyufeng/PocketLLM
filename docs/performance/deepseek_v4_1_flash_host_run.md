# DeepSeek-V4.1-Flash: what the released checkpoint costs to run on one host

The released V4.1-Flash weights now load into `src/models/deepseek_v4_1` and decode correct text on
this machine. This page is the measured cost of doing it, phase by phase, and the arithmetic that
says which of those phases the four RTX 2080 Ti can and cannot move.

**The forward timed here is entirely host code over a memory-mapped checkpoint**, so for everything
below the four cards are idle and every number is a CPU, RAM and disk number. That is the point: it
establishes what the host half costs, and it is the first measurement of this checkpoint anywhere in
this repository. Two things here touch a card at all — the PCIe rows of the carrier table, and the
`/tmp/probe_token_cost.py` line that measures the same prompt through the device path — and the
device path itself has its own page,
[deepseek_v4_1_flash_device_experts.md](deepseek_v4_1_flash_device_experts.md).

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB (SMR disk, `/dev/sda`) |
| Runtime | PyTorch resident, `src/models/deepseek_v4_1`, no native engine, no CUDA tensors |
| Commit | `688d803` on `feature/v41-backbone-runtime` |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each — **idle** for every host number below; EP world size 1 on the host path, one card timed for PCIe in the carrier table, and the device page's world 1 and 4 |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM, 930 GiB available |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Prompt | `The capital of France is` (5 tokens), and one token at position 0 for the phase table |
| Warm/cold | Both reported; the phase table gives three consecutive forwards of the same token |

Scripts: `/tmp/probe_where.py` (phase timing), `/tmp/probe_footprint.py` (what the tree occupies),
`/tmp/probe_engram_cost.py` (the Engram gather), `/tmp/probe_accept.py` (correctness),
`/tmp/probe_expand_vs_read.py` and `/tmp/probe_dequant_floor.py` (the read/expansion split and the
expansion's headroom), `/tmp/probe_top5_label.py` (which prefix length each quoted top-5 belongs to),
`/tmp/probe_capture.py` (the routing and the miss rate against the 16-slot window),
`/tmp/probe_token_cost.py` (the per-step table that replaced the page's miss arithmetic, on the host
path and on the device one) and `/tmp/probe_h2d.py` (the PCIe and card-side arithmetic in the carrier
table — the only one of them that wants a GPU). They are throwaway probes rather than checked-in
benchmarks; the numbers they produced are what this page records. The commit above is the code the
phase table, the byte census and the Engram numbers ran against; this page itself, the comment
corrections it prompted, and the checked-in generation path land in later commits, and the two
generated-token runs in the next section were made with
`src/models/deepseek_v4_1/generate.py` on top of `c9694b3`. The per-step table under them and the
corrected carrier rows were measured after that, on the same branch.

## What is in the 475 GiB

Byte census over all 96,085 tensors, from `V41Checkpoint.nbytes`:

| Group | GiB | Share |
| --- | ---: | ---: |
| `layers.ffn.experts` (routed experts, packed fp4) | 268.95 | 56.6% |
| `layers.engram` (two n-gram tables and their scales) | 189.13 | 39.8% |
| `mtp` (three DSpark draft layers, not loaded) | 7.39 | 1.6% |
| `layers.attn` | 4.80 | 1.0% |
| `layers.ffn.shared_experts` | 1.32 | 0.3% |
| `embed` + `head` | 2.46 | 0.5% |
| `vision` + `aligner` (not loaded) | 0.91 | 0.2% |
| everything else (norms, gates, hyper-connections) | 0.28 | 0.1% |
| **total** | **475.24** | |

**96.4% of this checkpoint is two things that are not arithmetic**: the routed experts and the two
Engram tables. Any plan for this model on this hardware is a plan for those two, and the rest of the
model is a rounding error that fits on the cards several times over.

## Loading it

Into `modules.Backbone` with `resident_engram=False`:

```text
924 tensors (330 quantized), 9.16 GiB read; not asked for: aligner 4, image_end 1,
image_newline 1, image_start 1, layers.0 2312, ... mtp 2401, vision 259;
no parameter left unfilled
```

- **924 of the tree's parameters filled, 330 of them quantized, none missing.** Those 924 names plus
  the 330 scales beside them are the only 1,254 of the checkpoint's 96,085 tensors the loader touches;
  the other 94,831 stay in the shards, 92,160 of them expert projections. The report prints 95,161
  because it counts a quantized name as one tensor rather than two — the point of the field is that
  the difference is counted at all, not passed over in silence.
- **9.16 GiB is read**, because everything quantized is dequantized at load and the experts and Engram
  tables are not read at all.
- **65.5 s** to load warm (`65.5`, `65.7`, `64.9` across runs) and **115.4 s** cold. The difference is
  the page cache, not the code.

What the process then holds:

| | GiB |
| --- | ---: |
| Tree parameters (14.12 bf16 + 2.67 fp32) | 16.79 |
| Expert window, if all 40 layers were full (16/layer) | 42.19 |
| Engram tables, resident | 189.13 |
| Process peak RSS after a plain load | 35.29 |

**At TP4 the tree is 4.20 GiB per rank — 19.1% of a 22 GiB card.** The dense model, including the
KV cache, is not the problem.

## The Engram tables: 189.13 GiB that must be resident

The group is two 94.41 GiB tables — 91.55 / 91.56 GiB of codes plus 2.86 GiB of scales each — and
0.29 GiB of `wkv` companions on the two Engram layers. A forward touches one row
per hash column, 24 per position, so a 512-token prefill gathers 12,288 rows per table. What a gather
costs, measured on this disk:

| Access | Cost |
| --- | ---: |
| One scattered row, page not resident, isolated | 48.6 ms |
| One scattered row, page not resident, batched | 20.6 ms |
| One scattered row, page resident | 0.004 ms |
| One scattered row, out of the resident copy | 0.001 ms |
| 12,288 scattered rows (one 512-token prefill), cold | 253.4 s |
| The same 12,288 rows out of the resident copy | 0.009 s |

The disk sustains **271 MiB/s** on a long sequential read (125 MiB/s on a 1 GiB one), so copying both
tables in costs **373 s per table, 747 s for both** — once. Against 253 s *per table per cold
prefill*, and a corpus keeps paying it as it moves onto n-grams it has not seen.

This is why `resident_engram=True` exists: the host has 930 GiB available for 189.13 GiB of tables,
and the alternative is not slower by a factor but unusable for prefill. Streaming remains the default
because a 189 GiB allocation should be asked for, not inherited.

## What a token costs

Three consecutive forwards of the same token at position 0, in one process:

| Phase | cold | warm | warm again |
| --- | ---: | ---: | ---: |
| attention stack (all 40 layers) | 0.78 s | 0.61 s | 0.37 s |
| ffn, the MoE (all 40 layers) | 26.01 s | 1.25 s | 0.44 s |
| Engram gather (layers 1, 14) | 0.02 s | 0.04 s | 0.01 s |
| `norm` + `head` | 0.18 s | 0.14 s | 0.04 s |
| per-layer overhead and `embed`, unattributed | 0.24 s | 0.19 s | 0.14 s |
| **total** | **27.23 s** | **2.23 s** | **1.00 s** |

The cold column is every routed expert of the token — 6 per layer, all 240 — expanded to bf16 for the
first time: 4.20 GiB read and 15.82 GiB written out. The third column is the *same* forward a third
time, and by then the layer's 16-slot window holds all six of the experts that token routes to, so its
0.44 s of MoE is arithmetic over cached bf16 with no expansion in it at all. **The third column is a
repeat cost, not a decode cost**, and the difference between the two is the whole of the next section.

The 16-expert FIFO window in `CheckpointRoutedExperts` turns out to be nearly worthless and is
measured rather than assumed: a token routes to 6 of a layer's 384 experts, and against a window that
holds the last 16, **about half of them miss — 2.67, 3.30, 2.92 and 3.02 misses per layer over the
four decode steps of `/tmp/probe_capture.py`, 6 distinct experts per layer in every one of them**. It
saves about half of a step's expert expansions for 42 GiB of host RAM across the backbone.
`DEFAULT_EXPERT_CACHE = 16` bounds a correctness path; it is not a cache policy.

### What a generated token costs, and of what

`src/models/deepseek_v4_1/generate.py` against the complete checkpoint, greedy, from the 5-token
prompt, two runs in separate processes:

| Run | New tokens | Wall | Per token | Text out |
| --- | ---: | ---: | ---: | --- |
| 1 | 4 | 169.5 s | 42.37 s | `The capital of France is Paris. The E` |
| 2 | 4 | 124.6 s | 31.15 s | `The capital of France is Paris. The E` |

The same command run three more times, once in the device page's session, gives a wider sample —
`168.0 s for 6`, `142.0 s for 4` and a second `142.0 s for 4`, so the whole-request per-token figure
is **28 to 42 seconds** over the five runs. A *step* whose expert pages are warm is the floor of that
range and it is measured separately, at **15.3 s** (below). Call a generated token **15 to 42
seconds** on this host, then, against the 1.00 s the repeated-forward column reports. The spread is the
page cache and the gap to the repeated forward is exactly the expansion a repeated forward does not
have to do. Splitting one miss into its two halves, over a layer whose pages are already resident:

| Step of one miss | Per projection tensor |
| --- | ---: |
| the mapping read (`reader.load`) | 0.1 ms |
| the expansion alone (`dequantize`, fp4 → fp32) | 39.4 ms |
| read + expand + cast (`weight(..., dtype=bf16)`) | 40.7 ms |

**The read is 0.3% of a miss and the expansion is 99.7% of it.** One expert is three of those tensors,
so 0.122 s per expert: a token's **240** experts are **29.3 s** when a step misses every one of them,
and that is what the 31 s of run 2 is made of. A warm step misses fewer, and the number is measured
rather than assumed — `/tmp/probe_token_cost.py`, one run, per step:

| Step | Wall | MoE | other | misses of 240 | s per miss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (5-token prefill) | 84.14 s | 82.41 s | 1.72 s | **759 of 1,200** | 0.1086 |
| 1 | 14.98 s | 13.59 s | 1.40 s | 107 | 0.1270 |
| 2 | 16.72 s | 15.85 s | 0.86 s | 132 | 0.1201 |
| 3 | 14.43 s | 14.08 s | 0.35 s | 117 | 0.1204 |
| 4 | 15.01 s | 14.22 s | 0.79 s | 121 | 0.1175 |

The prefill routes 5 tokens × 6 experts × 40 layers = **1,200** rows and misses **759** of them; a
decode step routes 240 and misses **107–132**, a mean of **119**. The per-miss cost is
**0.1175–0.1270 s**, which is the 0.122 s above and not something near it, so the decomposition holds
and the miss count is what was wrong: a warm decode step is **15.3 s** wall, **14.4 s** of it MoE and
**0.9 s** everything else — 119 misses × 0.122 s = 14.5 s of the 14.4 s.

**The miss *count* is the routing's and the miss *cost* is the page cache's.** 119 is what the
6-experts-against-a-16-slot window gives at any speed; this probe's steps are the fastest the page
has recorded because it ran behind probes that had already read the same expert rows, so its reads
were page-cache hits. The two `generate.py` runs above were slower and the two of them differ by 11 s
per token from each other in the same conditions, so what separates 15 s from 31 s and 42 s is how
much of the 4.20 GiB of expert pages was resident, not anything in the loop.

The expansion is not a bandwidth floor. The same probe times the bf16 cast of the already-expanded
matrix — the cheapest operation that writes the same 45 MiB — at **1.28 ms**, so the expansion runs at
304 Mparam/s against 9,223 Mparam/s for the cast alone: **30.4x the cheapest op of the same output
size**. It reaches `src/kernels/ops.py`'s `soft_fp4_blockfp4_weight_dequant`, a general elementwise
torch path that materializes intermediates at fp32. That says where the host's token time is spent; it
does not say the host could not spend less of it.

## Correctness

Greedy decode from `The capital of France is`, stepped through one token at a time rather than
prefilled, at temperature 0, through the host-offload path:

```text
The capital of France is  ->  ' Paris.<｜end▁of▁sentence｜>\n\n\n\n\n'
engram rows gathered:    {1: 456, 14: 456}
```

Those 456 gathers per table are 24 rows for each of the 19 positions that probe forwards in one
process — a profile forward, four prefill/stepwise comparisons and the thirteen-step decode — not
456 for the prompt on its own.

That block used to carry a third line, `expert misses per layer: min 26 max 52 total 1648`, and it
is gone because it was wrong: the probe stepped one token at a time at **8** experts per layer where
the checkpoint activates **6**, and its `1648` over 19 forwards reconciles with neither count. The
measured replacement is the table above — **119 of 240** expert rows missed at a decode step and
**759 of 1,200** at a five-token prefill, both from `/tmp/probe_token_cost.py`, which counts what
the class actually did rather than what a constant in a probe said it would.

The distribution at the end of a prefill is coherent rather than flat, and `/tmp/probe_top5_label.py`
pins which prefix each one belongs to, because the prompt is five tokens and not four:

| Prefill | Top-5 |
| --- | --- |
| 4 tokens: `The capital of France` | `' is'` 25.014, `','` 21.838, `' and'` 20.167, `' was'` 19.553, `' ('` 19.430 |
| 5 tokens: `The capital of France is` | `' Paris'` 20.605, `' ...'` 18.368, `'...'` 18.224, `' a'` 17.956, `' ______'` 17.505 |

The second row is the one the generation path starts from, and `' Paris'`, its argmax, is the first
token it emits — followed at once by the end-of-sentence token, which is the `' Paris.'` above.

**A prefill and the equivalent stepwise decode do not agree bit for bit, and this is understood
rather than tolerated.** From position 0:

- Length 1 is **bit-exact**: max absolute difference `0.0000`, same argmax (201), same top-5.
- Length 2 differs by `6.0748` max / `0.9915` mean with a different argmax.

A per-layer bisect of the length-2 case finds the first nonzero difference at `layers.0.attn`
(`max 0.000002`), and it grows monotonically — `layers.1 0.000244`, `layers.5 0.1377`,
`layers.20 1.625`, `layers.39 155.0` — which is what fp32 reduction-order differences do across 40
layers, not what a missing or misapplied term does. Length 1 being exact rules out the structural
explanations. The acceptance evidence is the generated text.

No reference implementation was run for comparison; the correctness claim here is that the text is
right, not that the logits match the reference's.

## The wall, and what the cards can do about it

A decode step routes to 6 experts per layer, so it must move

```text
6 experts x 40 layers x 17.9 MiB of packed fp4 and its scales  =  4.20 GiB of expert bytes per token
```

and all 384 experts of all 40 layers are **268.95 GiB**, against 88 GiB of VRAM on the four cards
combined. The experts cannot be resident on the cards. Everything else in the model — the whole
924-parameter tree, attention and shared experts and embeddings and norms — is 16.79 GiB, 3.5% of the
checkpoint, and it fits at 19.1% of a card per rank.

So the floor is set by the expansion, not by the bytes and not by FLOPS. Every carrier that could
bring the 4.20 GiB to a consumer is now measured rather than estimated:

| Carrier | Rate on this host | Floor per token |
| --- | ---: | ---: |
| host CPU today: read free, expand a step's missed experts | 0.122 s per expert measured | **14.5–29 s** (119 misses warm, 240 cold) |
| PCIe Gen3 x16, packed fp4 as stored, pinned | 8.29 GiB/s measured | 0.51 s |
| PCIe Gen3 x16, expanded to bf16 first, pinned | 10.19 GiB/s measured | 1.55 s |
| the SMR disk behind both | 271 MiB/s | 16 s |

These are `/tmp/probe_h2d.py` on `cuda:2`, over the real 1,440-tensor token load and not a synthetic
buffer. The raw slot sustains 6.4 GiB/s pageable and 10.6 GiB/s pinned on a 512 MiB buffer; the token
load reaches 5.70 and 8.29 GiB/s of that — 11% and 22% below the slot — so 1,440 separate copies cost
a fifth and not a factor. The `I8` packing costs nothing on the wire: the 4.20 GiB that crosses is the
bytes as stored. The bf16 row is the one that decides whether a device-side expert cache is worth
building on the dequant path that already exists, and it costs 3.0x the time: 4.20 GiB of packed fp4
becomes 15.82 GiB of bf16, which is also why its rate is the higher of the two.

The first row is the one that surprised this page, and the one that decides the rest of it: the bytes
are already in host RAM, and reading them is 0.3% of what a miss costs. The other 99.7% is arithmetic
— the fp4 code turned into a bf16 number at 304 Mparam/s, where the cheapest op of the same output
size runs at 9,223 Mparam/s. Every other row here is a hardware rate; that one is an implementation
cost, and it is what the host's token time is made of.

**The cards' arithmetic is not the problem, and neither is the transfer out to them.** The same probe
times an isolated batched MoE for one layer — 8 experts through all three projections — at 1,564 µs,
and the device path is what a real token pays for the same work: **0.27 s** for all 40 layers' four
call sets, D2H partials and host sum included, measured, against the **14.5–29 s** the host spends on
the same 40 layers. The card is two orders of magnitude the cheaper place to do the multiplication,
and it stays the cheaper place after being handed the operands. The whole device step — staging the
packed rows, four links, four kernels a layer, and the dense tree still on the host — measures
**1.23–1.42 s**, against the 1.00 s the host needs merely to *repeat* a forward it has already cached
and the 15 to 42 s it needs for a real one. That 0.27 s was four kernels serialized rather than one
plus copies — the drain sat inside the card loop — and the device page's
[ordering fix](../performance/deepseek_v4_1_flash_device_experts.md#the-launch-was-four-kernels-serialized-not-one-plus-copies)
takes it to **0.15 s** and the step to **1.06–1.14 s**; the 0.27 s is kept here as what this page
measured. Moving the tree itself off the host, onto the same four cards under `torchrun
--nproc_per_node=4`, is the device page's
[later section](../performance/deepseek_v4_1_flash_device_experts.md#the-dense-tree-across-the-four-cards-tp4)
and takes that step to **722–747 ms** — again with the caveat that the staging underneath it is a
page-cache read, 19× worse on the same row once the cache is emptied.

That reverses the reading this page first drew from the carrier table. It took the repeated forward's
0.44 s of MoE as the host-RAM carrier's floor and set it against 0.68 s of PCIe, and concluded that
the bytes were the wall. The fair comparison is against what a real step costs, and a real step
expands 119 experts warm and 240 cold: **the bytes are not the wall, the expansion is.** The four
cards are the fastest way measured to pay it.

What does not change is the constraint underneath: 268.95 GiB of experts against 88 GiB of VRAM, so
every token still moves 4.20 GiB. A device-side expert cache holding whole layers resident fails on
arithmetic rather than on engineering — one layer is 25.3 GiB expanded, so all 88 GiB of VRAM holds
three of the forty and the remaining 37 layers would still pay the 0.68 s. Only a format that makes a
layer small enough to sit on a card removes the per-token transfer.

The host path's own 15 to 42 s is left where it is, and the measurement says where the room is: an
expansion running at 30.4x the cheapest op of the same output size is a kernel problem, not a
bandwidth one, so a host that kept the experts on the CPU has that much in front of it without
touching PCIe at all.

## Reproducing

```bash
# the load report, the phase table, and the three forwards of one token at position 0
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_where.py

# one expert miss split into its mapping read and its fp4 expansion, and the expansion against the
# cheapest operation that writes the same output size
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_expand_vs_read.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_dequant_floor.py

# a generated token, which is the number the expansion actually sets: ~60 s to load, then 15-42 s
# per token
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --prompt "The capital of France is" --max-new-tokens 4

# the correctness run: the profile forward, the prefill/stepwise comparison, and the greedy decode
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_accept.py

# which prefix length each quoted top-5 belongs to, one prefill per length
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_top5_label.py

# the byte census, what the tree occupies, and the TP4 per-rank figure
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_footprint.py

# the Engram gather, cold, warm, and out of the copy
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_engram_cost.py

# the per-step routes, misses and wall clock that replaced this page's miss arithmetic, and the same
# prompt through the device path with `--expert-device cuda --expert-world 4`
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_token_cost.py --max-new-tokens 6

# the routing and the miss rate on their own, one layer at a time
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_capture.py

# the PCIe rows of the carrier table and the card's MoE arithmetic -- the only probe here that
# wants a GPU, and the only one that reads 21 GiB instead of 4.2
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_h2d.py
```

Each takes one to fifteen minutes and reads the checkpoint off `/mnt/data3`. The Engram probe's copy
step alone is 373 s and 94.5 GiB of page cache, so run it alone. `probe_h2d.py` reports whichever
page-cache state it finds the checkpoint in; run it twice for a warm H2D column. The two expansion
probes want the opposite: they warm the pages on purpose, so that what they time is the arithmetic
and not the disk. `probe_top5_label.py` pays the ~60 s load again for five prefixes at one forward
each, which is worth it once — the prefix a quoted distribution belongs to is not recoverable from
the numbers. `probe_token_cost.py` is the one whose page-cache state has to be read off the run: its
per-miss column is a property of the class and its wall clock is a property of what is resident, and
the two are reported side by side for that reason.
