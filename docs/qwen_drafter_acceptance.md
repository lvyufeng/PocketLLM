# Qwen speculative drafter acceptance and speedup

Measured comparison of the three speculative drafters available for the Qwen
GPU-resident FP8 path: native MTP, external DSpark, and external DFlash2.

## Setup

| Field | Value |
| --- | --- |
| Checkpoint | `Qwen3.8-27B-FP8` (GPU-resident FP8) |
| Drafter checkpoints | `Qwen3.8-27B-DSpark`, `Qwen3.8-27B-DFlash2` |
| Commit | `9fe6952` |
| Runtime | `cpp_engine`, `--resident-bench` |
| Hardware | 4x RTX 2080 Ti, 22528 MiB each, TP world 4 |
| CPU / RAM | 2x Xeon E5-2696 v4, 88 threads |
| Driver / CUDA | 580.173.02 / 13.0.88 |
| Dataset | gsm8k, 8 prompts, real chat-template fixtures |
| Prompt / generated | 133 prompt tokens (request 0), 256 generated |
| KV cache | fp16, prefill chunk 512, max context 32768 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20, seed 42 |
| Prefix cache | off |

Each drafter was benchmarked in a separate serial run with its own plain
baseline, with the GPUs confirmed idle beforehand. Ranks run concurrently but
occupy one card each.

## Results

Plain baseline: 27.95 ms/token decode, 309.5 prefill tok/s.

| Drafter | Draft width | acc_len | accept rate | draft ms/step | verify ms/step | decode ms/tok | decode speedup | wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mtp | 2 | 1.792 | 0.792 | 7.69 | 30.49 | 22.20 | 1.251x | 1.231x |
| dspark | 8 | 3.762 | 0.395 | 39.03 | 63.54 | 30.02 | 0.931x | 0.935x |
| dflash2 | 8 | 3.955 | 0.422 | 40.27 | 62.69 | 28.30 | 0.988x | 0.991x |

`acc_len` is mean accepted tokens per verify step. Draft width is proposed
drafts plus one. Speedups are means over the 8 per-request ratios, so they do
not divide out exactly against the aggregate ms/token column.

Sampled mode draws a fresh uniform each step, so drafter and plain streams
diverge by design and the harness reports no token parity. Correctness for this
commit rests on the separate 4-rank identity check, where all four ranks emit
byte-identical streams in both plain and mtp modes.

## Why the longer drafters lose

DSpark and DFlash2 reach more than double MTP's accepted length yet fail to beat
plain decode. Two costs, both following from the draft width of 8.

Draft generation alone exceeds a full plain decode step. At 39-40 ms per step it
costs more than plain's 27.95 ms/token, consuming the entire budget before
verification starts. MTP proposes one token for 7.69 ms.

Wide verification is not free. Verify cost per row does fall with batch width,
so the batching works:

```text
plain    width=1   27.95 ms/step  ->  27.95 ms/row
mtp      width=2   30.49 ms/step  ->  15.25 ms/row
dspark   width=8   63.54 ms/step  ->   7.94 ms/row
dflash2  width=8   62.69 ms/step  ->   7.84 ms/row
```

But width 8 still costs 2.3x a width-1 step. A 27B decode step should be
memory-bound with weight reads amortized across the batch, so paying 2.3x for 8
rows means the verify path is not running in that regime.

Per useful token, including both phases: DSpark spends (39.03+63.54)/3.762 =
27.3 ms, DFlash2 (40.27+62.69)/3.955 = 26.0 ms, MTP (7.69+30.49)/1.792 =
21.3 ms. MTP's short draft is cheap enough to win despite the lower accepted
length. The wide drafters hit roughly 0.40 accept rate, so about 5 of every 8
drafted tokens are discarded and paid for.

The bottleneck is draft width relative to current kernel behaviour, not
acceptance quality. Sweeping width downward is the obvious next step.

## Not covered

- Whether the 39-40 ms draft step is dominated by the external drafter forward
  or by surrounding top-k/sampling work; the logs do not separate these.
- Whether the 2.3x wide-batch verify cost is an attention kernel selection
  problem; this needs a separate profile.
- Greedy (`--temperature 0`) parity numbers for the two external drafters.
