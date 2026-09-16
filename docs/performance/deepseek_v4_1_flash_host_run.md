# DeepSeek-V4.1-Flash: what the released checkpoint costs to run on one host

The released V4.1-Flash weights now load into `src/models/deepseek_v4_1` and decode correct text on
this machine. This page is the measured cost of doing it, phase by phase, and the arithmetic that
says which of those phases the four RTX 2080 Ti can and cannot move.

**Nothing in this run used a GPU.** The forward is entirely host code over a memory-mapped
checkpoint, so the four cards are idle throughout and every number below is a CPU, RAM and disk
number. That is the point: it establishes what the host half costs before any device work, and it is
the first measurement of this checkpoint anywhere in this repository.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB (SMR disk, `/dev/sda`) |
| Runtime | PyTorch resident, `src/models/deepseek_v4_1`, no native engine, no CUDA tensors |
| Commit | `688d803` on `feature/v41-backbone-runtime` |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each — **not used**; TP/EP world size 1 CPU rank |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM, 930 GiB available |
| Software | Python 3.10.10, torch 2.9.1+cu128, `deepseek` conda env |
| Prompt | `The capital of France is` (4 tokens), and one token at position 0 for the phase table |
| Warm/cold | Both reported; the phase table gives three consecutive forwards of the same token |

Scripts: `/tmp/probe_where.py` (phase timing), `/tmp/probe_footprint.py` (what the tree occupies),
`/tmp/probe_engram_cost.py` (the Engram gather), `/tmp/probe_accept.py` (correctness). They are
throwaway probes rather than checked-in benchmarks; the numbers they produced are what this page
records. The commit above is the code they ran against — this page itself, and the comment
corrections it prompted, land in a later documentation-only commit.

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
| Expert window, if all 40 layers were full (16/layer) | 21.09 |
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

The cold column is 240 routed experts each expanded to bf16 for the first time, off a disk that has
never seen them. The third column is the steady state: **1.00 s per token**, of which the attention
stack is 0.37 s and the MoE 0.44 s.

The 16-expert FIFO window in `CheckpointRoutedExperts` turns out to be nearly worthless and is
measured rather than assumed: a token routes to 8 of a layer's 384 experts, and **6 of those 8 are
misses at every layer, every token, warm**. It saves 25% of the expert reads for 21 GiB of host RAM
across the backbone. `DEFAULT_EXPERT_CACHE = 16` bounds a correctness path; it is not a cache policy.

## Correctness

Greedy decode from the 4-token prompt, at temperature 0, through the host-offload path:

```text
The capital of France is  ->  ' Paris.<｜end▁of▁sentence｜>\n\n\n\n\n'
expert misses per layer: min 26 max 52 total 1648
engram rows gathered:    {1: 456, 14: 456}
```

A 4-token prefill's top-5 is `' is'` 25.014, `','` 21.838, `' and'` 20.167, `' was'` 19.553,
`' ('` 19.430 — a coherent distribution, not a flat one.

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

A decode step routes to 8 experts per layer, so it must move

```text
8 experts x 40 layers x 17.9 MiB of packed fp4 and its scales  =  5.60 GiB of expert bytes per token
```

and all 384 experts of all 40 layers are **268.95 GiB**, against 88 GiB of VRAM on the four cards
combined. The experts cannot be resident on the cards. Everything else in the model — the whole
924-parameter tree, attention and shared experts and embeddings and norms — is 16.79 GiB, 3.5% of the
checkpoint, and it fits at 19.1% of a card per rank.

So the floor is set by bytes per token, not by FLOPS. 5.60 GiB per token is:

| Carrier | Rate on this host | Floor per token |
| --- | ---: | ---: |
| host RAM, read and expanded to bf16 on the CPU | measured | **0.44 s** (the MoE column above) |
| PCIe Gen3 x16 to a 2080 Ti | ~10 GB/s realistic | ~0.60 s |
| the SMR disk behind both | 271 MiB/s | 21 s |

The measured 0.44 s is the fastest of the three, and that is the finding: **moving the experts to the
GPU does not move this wall, because the wall is the bytes and the GPU cannot hold them.** Pushing
the same 5.60 GiB across PCIe is not cheaper than reading it from host RAM and expanding it on the
CPU — that the two are within 40% of each other is what makes the point, not that they are equal.
The four cards can only help with the 0.37 s attention stack and the 0.04 s of `norm` + `head`: the
part of the model that is neither the MoE nor the Engram tables, 4.20 GiB per rank at TP4, and the
one part that is a normal port.

Reaching a materially higher token rate on this hardware needs fewer bytes per token — a lower-bit
expert format, or fewer active experts — and not a faster kernel.

## Reproducing

```bash
# the load report, the phase table, and the first token's cold/warm columns
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_where.py

# the byte census, what the tree occupies, and the TP4 per-rank figure
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_footprint.py

# the Engram gather, cold, warm, and out of the copy
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/probe_engram_cost.py
```

Each takes one to fifteen minutes and reads the checkpoint off `/mnt/data3`; none of them needs a
GPU. The Engram probe's copy step alone is 373 s and 94.5 GiB of page cache, so run it alone.
