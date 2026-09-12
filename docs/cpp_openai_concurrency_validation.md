# Native C++ OpenAI concurrency validation

This document records the end-to-end HTTP acceptance test for issue #106. It is
separate from the lower-level batched-kernel benchmark: the test starts the native
OpenAI-compatible server, sends real HTTP requests, and checks request isolation,
streaming, cancellation recovery, and concurrent throughput.

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

Commit `6b762a3` (`master` after PR #160), checkpoint
`/mnt/data2/Qwen3.8-27B-FP8`, TP4, 128-token prompts, 32 generated tokens.
Measurements are one run per concurrency level in each separately launched
server process; the numbers are acceptance measurements rather than a statistical
performance sweep.

### Single request

| Server | Wall (s) | Output tok/s | Result |
|---|---:|---:|---|
| `max_batch_size=1` | 1.186 | 27.0 | pass |
| `max_batch_size=8` | 1.108 | 28.9 | pass |

Batch mode was **0.934x** of the serial wall time in this run (6.6% faster), so
there is no observed single-request latency regression. Treat this as a smoke
bound, not a claim of a permanent 6.6% gain.

### Concurrent non-streaming requests

| Concurrent requests | Serial wall (s) | Batch wall (s) | Wall speedup | Serial req/s | Batch req/s |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.621 | 1.137 | **1.43x** | 1.23 | 1.76 |
| 4 | 3.401 | 1.287 | **2.64x** | 1.18 | 3.11 |
| 8 | 7.100 | 2.423 | **2.93x** | 1.13 | 3.30 |

The 8-request run produced 105.7 output tokens/s versus 36.1 tokens/s for the
serial server. Every request returned exactly 32 completion tokens with
`finish_reason=length`.

The end-to-end HTTP speedup is lower than the native decode-kernel speedup from
#131 (1.88x/3.09x/3.94x at batch 2/4/8) because this test includes prefill,
request handling, response serialization, and the scheduler's variable active
set. It nevertheless confirms that HTTP requests are actually multiplexed rather
than merely queued behind a single generation.

### Long/short interleave

A long request was submitted first, followed 0.5 seconds later by a short request.
Both completed successfully:

| Mode | Short latency (s) | Long completion tokens | Short completion tokens |
|---|---:|---:|---:|
| Serial | 2.304 | 32 | 32 |
| Batch/chunked | 1.629 | 32 | 32 |

The batch server's short request completed while the long request was still in
flight, demonstrating that the scheduler regained control between prefill chunks.

### Streaming and disconnect recovery

Both modes passed:

- 2 concurrent streams, with valid `chat.completion.chunk` events and `[DONE]`;
- role marker and terminal `finish_reason` present;
- no cross-request response mixing;
- a client disconnect after two events;
- a subsequent non-streaming request completing successfully after that
  disconnect.

The reduced harness smoke additionally passed the same cases with a four-layer
model, confirming the test's lifecycle and cleanup behavior independently of the
full-depth timing run.

## Verdict for issue #106

The Qwen CUDA native OpenAI server's core continuous-batching behavior is
validated end to end: request admission, multi-request batched decode, chunked
prefill interleaving, concurrent streaming, and post-disconnect recovery all pass.
The issue should be considered **substantially complete for this target path**,
but not a claim that every PocketLLM engine or decoding mode supports batching.
The limitations listed in the Scope section should remain explicit follow-up work.
