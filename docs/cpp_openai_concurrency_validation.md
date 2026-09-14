# Native C++ OpenAI concurrency validation

This document records the end-to-end HTTP acceptance test for issue #106 and the
vLLM head-to-head comparison that followed it. It is separate from the lower-level
batched-kernel benchmark: the test starts the native OpenAI-compatible server,
sends real HTTP requests, and checks request isolation, streaming, cancellation
recovery, and concurrent throughput.

## Scope

The validated path is:

- Qwen3.8-27B-FP8 safetensors
- native `pocketllm_engine`
- CUDA, TP4, four RTX 2080 Ti devices
- OpenAI `/v1/chat/completions`
- FP16 KV cache
- full model depth

The following remain outside this acceptance result:

- TP2 automatic supervision (tracked in #159)
- DeepSeek-V4/PersistentEngine multi-slot execution
- MTP, DSpark, and DFlash2 batched decode
- Ascend batched decode
- FP8/TurboQuant batched KV decode
- SGLang: not installed and not measured. The SGLang comparison in
  [vllm_sglang_architecture_analysis.md](vllm_sglang_architecture_analysis.md) is
  an architecture-level reading of its scheduler and cache, not a benchmark.

## Reproduction

Run one configuration at a time. The harness starts and stops all four TP ranks
itself, waits for `/health`, and writes a JSON record before shutting down:

```bash
python scripts/bench_cpp_openai_concurrency.py \
  --ckpt /mnt/data2/Qwen3.8-27B-FP8 \
  --binary cpp_engine/build-python/pocketllm_engine \
  --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  --devices 0,1,2,3 \
  --mode batch \
  --max-batch-size 8 \
  --prefill-token-budget 4096 \
  --max-context 8192 \
  --max-tokens 32 \
  --short-prompt-words 128 \
  --concurrency 2 4 8 \
  --json-out /tmp/http-concurrency-batch.json
```

The serial control uses the same command with `--mode serial` and
`--max-batch-size 1`. Each mode gets a separate server process and model load.

`--warmup-rounds` (default 1) makes a discarded pass over the measured ladder before
recording. The engine's first request is not free — kernel module load, allocator
growth, and, at TP4, the first collective on a freshly built NCCL communicator — and
without the warmup that cost lands on the `count=1` case the report reads as
"single-request latency", which makes every other level look faster by comparison.

The vLLM side uses the same harness functions through a separate driver, so both
engines send byte-identical payloads and compute their metrics the same way:

```bash
python scripts/bench_qwen_vllm_concurrency.py \
  --python /home/lvyufeng/miniconda3/envs/vllm-2080ti-v015/bin/python \
  --checkpoint /mnt/data2/Qwen3.8-27B-FP8 \
  --devices 0,1,2,3 \
  --mode batch --max-batch-size 8 \
  --max-num-batched-tokens 4096 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-tokens 32 --short-prompt-words 128 --concurrency 2 4 8 \
  --json-out /tmp/vllm-http-concurrency-batch.json
```

The harness covers:

1. Single-request latency.
2. 2/4/8 simultaneous non-streaming requests.
3. A long request started before a short request, exercising scheduler admission
   and chunked prefill.
4. Two simultaneous streaming requests, including SSE framing and `[DONE]`.
5. A client that disconnects after two stream events, followed by a fresh request
   to verify slot/KV cleanup and recovery.

Prompts are natural text rather than arbitrary token IDs. Every completion is
checked for a valid response object, positive usage counts, a valid finish reason,
and `total_tokens == prompt_tokens + completion_tokens`. The tests use a fixed
32-token generation budget and verify that every request reaches that budget, so
an early EOS cannot look like a speedup.

## Results

### Current run

Commit `f8a4fb8`, checkpoint `/mnt/data2/Qwen3.8-27B-FP8`, TP4, four RTX 2080 Ti,
128-word prompts, 32 generated tokens, one run per concurrency level in each
separately launched server process. These are acceptance measurements rather than a
statistical performance sweep.

PocketLLM: `max_batch_size=8`, `--prefill-token-budget 4096`, `--max-context 8192`,
FP16 KV, physical GPUs 0-3.
vLLM: 0.1.15, `max_num_seqs=8`, `--max-num-batched-tokens 4096`,
`--max-model-len 8192`, `--gpu-memory-utilization 0.90`,
`gdn_prefill_backend=triton`, `--no-enable-prefix-caching`, chunked prefill enabled
(vLLM's default, and the setting that matters at concurrency; the single-request
A/B below used `--no-enable-chunked-prefill`).

All four records report `status: pass`, and every request in every level returned
exactly 32 completion tokens with `finish_reason=length`.

#### Single request

| Server | Wall (s) | Output tok/s | Result |
|---|---:|---:|---|
| PocketLLM `max_batch_size=8` | 0.763 | 41.93 | pass |
| PocketLLM `max_batch_size=1` | 0.916 | 34.92 | pass |
| vLLM `max_num_seqs=8` | 0.928 | 34.47 | pass |
| vLLM `max_num_seqs=1` | 0.887 | 36.09 | pass |

Batch mode was **0.833x** of the serial wall time on PocketLLM (17% faster) and
**1.046x** on vLLM (5% slower). On PocketLLM the batch arm also reuses the prompt
prefix across the slots it allocates, so part of that 17% is cache reuse rather than
scheduling; treat it as a smoke bound, not a permanent gain.

#### Concurrent non-streaming requests

Wall seconds, and the batch/serial speedup each engine gets from its own scheduler:

| Concurrent requests | PocketLLM batch | PocketLLM serial | PocketLLM speedup | vLLM batch | vLLM serial | vLLM speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.854 | 1.825 | **2.14x** | 1.147 | 1.746 | 1.52x |
| 4 | 1.009 | 3.647 | **3.61x** | 1.328 | 3.502 | 2.64x |
| 8 | 1.561 | 7.300 | **4.68x** | 1.834 | 7.000 | 3.82x |

Head-to-head at the same concurrency level, batch mode on both sides:

| Concurrent requests | PocketLLM (s) | vLLM (s) | PocketLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 0.763 | 0.928 | **0.82x** |
| 2 | 0.854 | 1.147 | **0.74x** |
| 4 | 1.009 | 1.328 | **0.76x** |
| 8 | 1.561 | 1.834 | **0.85x** |

Aggregate output throughput (output tokens/s across all in-flight requests) and
average per-request latency:

| Concurrent requests | P batch tok/s | P serial tok/s | V batch tok/s | V serial tok/s | P batch latency (s) | V batch latency (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.93 | 34.92 | 34.47 | 36.09 | 0.763 | 0.928 |
| 2 | 74.98 | 35.07 | 55.78 | 36.66 | 0.840 | 1.134 |
| 4 | 126.81 | 35.10 | 96.39 | 36.55 | 1.000 | 1.319 |
| 8 | 163.95 | 35.07 | 139.61 | 36.57 | 1.530 | 1.825 |

Two things are worth separating here.

PocketLLM's concurrency scaling is the stronger of the two: 2.14x/3.61x/4.68x wall
speedup at 2/4/8 against vLLM's 1.52x/2.64x/3.82x. Its serial arm is also flatter —
35.07 output tok/s at n=8 versus 34.92 at n=1 — which confirms the serial control is
genuinely serializing rather than accidentally batching.

vLLM's batch arm reaches a lower throughput ceiling at every level on this hardware,
and its absolute batch wall time is 17-35% higher than PocketLLM's at 1-8 concurrent
requests. That is consistent with the single-request A/B below: the two engines are
within a few percent on this model at TP4, so the gap here is scheduler and
per-request overhead rather than one engine's kernels being faster.

The end-to-end HTTP speedup is lower than the native decode-kernel speedup from #131
(1.88x/3.09x/3.94x at batch 2/4/8) because this test includes prefill, request
handling, response serialization, and the scheduler's variable active set. It
nevertheless confirms that HTTP requests are actually multiplexed rather than merely
queued behind a single generation.

#### Long/short interleave

A long request was submitted first, followed 0.5 seconds later by a short request.
Both completed successfully in every mode:

| Mode | Short latency (s) | Wall (s) | Long completion tokens | Short completion tokens |
|---|---:|---:|---:|---:|
| PocketLLM batch/chunked | 1.650 | 2.152 | 32 | 32 |
| PocketLLM serial | 2.348 | 2.849 | 32 | 32 |
| vLLM batch/chunked | 1.363 | 1.865 | 32 | 32 |
| vLLM serial | 2.322 | 2.824 | 32 | 32 |

The batch server's short request completed while the long request was still in
flight, demonstrating that the scheduler regained control between prefill chunks.

#### Streaming and disconnect recovery

Both PocketLLM modes passed:

- 2 concurrent streams, with valid `chat.completion.chunk` events and `[DONE]`;
- role marker and terminal `finish_reason` present;
- no cross-request response mixing;
- a client disconnect after two events;
- a subsequent non-streaming request completing successfully after that
  disconnect, with 32 completion tokens.

The vLLM side of this run did not exercise streaming; the harness's streaming cases
were run on PocketLLM only.

The reduced harness smoke additionally passed the same cases with a four-layer
model, confirming the test's lifecycle and cleanup behavior independently of the
full-depth timing run.

### Single-request long-context A/B

Same two servers, one request at a time, no concurrency, 128 generated tokens,
`max_context` matched to the prompt, greedy sampling on both sides, FP16 KV. vLLM
ran with `--no-enable-chunked-prefill` here so that `--max-model-len` pins the exact
prompt length; PocketLLM used `--prefill-chunk-tokens 8192`.

Ratios in the last three columns are PocketLLM/vLLM, so > 1.00x means PocketLLM is
faster. Decode tokens are 127: the first token is produced by prefill.

| Prompt tokens | PocketLLM prefill | vLLM prefill | Prefill ratio | PocketLLM decode | vLLM decode | Decode ratio | PocketLLM wall | vLLM wall | Wall ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,722.63 tok/s (2.378 s) | 1,799.47 tok/s (2.276 s) | 0.96x | 44.07 tok/s | 41.45 tok/s | 1.06x | 5.260 s | 5.340 s | 1.02x |
| 8,192 | 1,826.99 tok/s (4.484 s) | 1,772.19 tok/s (4.623 s) | 1.03x | 44.23 tok/s | 41.07 tok/s | 1.08x | 7.355 s | 7.715 s | 1.05x |
| 32,768 | 1,681.88 tok/s (19.483 s) | 1,653.32 tok/s (19.820 s) | 1.02x | 42.15 tok/s | 39.87 tok/s | 1.06x | 22.496 s | 23.005 s | 1.02x |
| 65,536 | 1,457.11 tok/s (44.977 s) | 1,515.75 tok/s (43.237 s) | 0.96x | 39.32 tok/s | 38.07 tok/s | 1.03x | 48.207 s | 46.573 s | 0.97x |

All four PocketLLM runs report rank token parity PASS across the four TP ranks.
PocketLLM's peak per-rank memory rose from 10.86 GiB at 4K to 12.29 GiB at 64K
(66 MiB to 1,026 MiB of that is KV cache), against 22 GiB per device.

The summary: **prefill is at parity within ±4%**, PocketLLM's **decode is 3-8% ahead
at every length**, and end-to-end wall time is within ±5% in both directions — with
PocketLLM faster at 4K/8K/32K and vLLM 3.4% faster at 64K, where prefill dominates.

### Superseded run

The original acceptance run was taken at commit `6b762a3` and reported 1.43x/2.64x/
2.93x wall speedup at 2/4/8 with 1.108 s single-request latency. It is superseded
for two reasons and is not comparable row by row with the table above:

- the harness has changed (prompt construction, the warmup pass, and the JSON record
  layout), so its absolute numbers were never on the same footing;
- its batch arm stopped being reproducible at `dc07d1e` (#183), which handed
  per-request sampling parameters into the TP prefill path and made the first HTTP
  request hang forever on a mismatched NCCL collective. Fixed in `f8a4fb8`. The
  historical numbers predate that regression and were valid when taken: running the
  pre-fix code now does not produce a slow number at all, it produces a hang.

## Comparison against vLLM

Across both the long-context single-request A/B and the short-prompt concurrency
ladder, PocketLLM and vLLM are the same class of engine on this model and this
hardware — four RTX 2080 Ti at TP4, Qwen3.8-27B-FP8:

- Single request, 4K-64K: prefill parity, decode 1.03-1.08x in PocketLLM's favour,
  wall time within ±5%.
- 1-8 concurrent: PocketLLM is 1.17-1.34x faster on wall time, and its scheduler
  extracts more speedup from the same request ladder (4.68x versus 3.82x at n=8).

Where vLLM still has structural advantages that these numbers do not measure:
grammar-constrained (GBNF/regex) decoding, LoRA multiplexing, multimodal input,
multi-node serving, and a broader production feature set. PocketLLM's JSON mode and
JSON Schema constraints exist in the native engine but are refused at TP > 1, so
they do not apply to the configurations measured here.

## Verdict for issue #106

The Qwen CUDA native OpenAI server's core continuous-batching behavior is
validated end to end: request admission, multi-request batched decode, chunked
prefill interleaving, concurrent streaming, and post-disconnect recovery all pass.
The issue should be considered **substantially complete for this target path**,
but not a claim that every PocketLLM engine or decoding mode supports batching.
The limitations listed in the Scope section should remain explicit follow-up work.
