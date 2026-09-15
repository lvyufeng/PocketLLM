# Phase 3.5 Validation Results

Measured on commit `c8c3b15`, 4× RTX 2080 Ti (22 GB), TP4, checkpoint
`/mnt/data2/Qwen3.8-27B-FP8`, 16-token prompts, 32 new tokens, 1 warmup + 5 runs
per measurement. Driver: `scripts/run_phase35_benchmarks.sh`.

## Verdict

| Target | Result | |
|---|---|---|
| Single-request latency ≤1.05× serial | **1.002×** | PASS |
| 2 concurrent ≥1.7× throughput | **1.00×** | FAIL |
| 4 concurrent ≥3.0× throughput | **0.99×** | FAIL |
| 8 concurrent ≥4.5× throughput | **0.99×** | FAIL |

Batching costs nothing for a lone request, and it buys nothing for concurrent
ones. The concurrency targets cannot be met by tuning the scheduler; they need a
batched decode kernel that does not exist yet.

## Concurrent throughput detail

| Level | Mode | Wall per round | Avg request latency | Throughput |
|---|---|---|---|---|
| 2 | serial | 1.344s | 1.007s | 1.488 req/s |
| 2 | batch | 1.350s | 1.339s | 1.481 req/s |
| 4 | serial | 2.703s | 1.689s | 1.480 req/s |
| 4 | batch | 2.717s | 2.700s | 1.472 req/s |
| 8 | serial | 5.471s | 3.075s | 1.462 req/s |
| 8 | batch | 5.517s | 5.494s | 1.450 req/s |

Throughput is flat at ~1.46 req/s regardless of mode or concurrency: wall time
per round scales linearly with the request count in both modes. Batch mode does
change the latency *distribution* — every request finishes at about the same time
(5.494s of 5.517s wall at 8 concurrent) instead of early ones finishing first
(3.075s average) — which is what fair scheduling looks like, but it moves no
total work.

The scheduler itself is doing its job. Instrumenting `get_stats()` during an
8-request run shows peak `running_requests = 8` with 7 waiting, so all slots are
genuinely occupied concurrently rather than silently serializing.

## Why throughput is flat

`QwenEngine::batch_decode_step` (`cpp_engine/engine/qwen_engine.cpp:4281`) loops
over the batch and calls the single-sequence `decode_step` once per request, with
its own comment saying so:

```cpp
// Phase 3.1: Process decode steps sequentially
// True batched decode kernel is deferred to Phase 3.2
for (QwenBatchedRequest* req : requests) {
    worker_command_decode(req->last_token, req->slot_id);
    QwenForwardResult fwd_result = decode_step(req->last_token, req->slot_id);
```

N concurrent requests therefore do N times the sequential work per step, plus a
TP broadcast each. 1.00× is the structural ceiling of this design, so the ≥1.7×
/ ≥3.0× / ≥4.5× targets were never reachable at this commit — the batched decode
kernel deferred from Phase 3.2 is the missing prerequisite. Weight loading is
what these targets were meant to amortize, and that only happens with a kernel
that computes several sequences per pass.

## Two measurement bugs found and fixed

**Serial TTFT was hardcoded to `0.0`.** `_generate_serial` filled in a
placeholder, so the latency benchmark divided by zero comparing TTFT ratios. The
stepped path now measures it (`pocketllm/backends/cpp_backend.py`), since it
drives prefill and decode itself and knows when the first token appeared. The
single-rank fast path still reports `0.0`, because native `generate()` runs its
own loop and gives no first-token boundary — that is a real absence, not a
placeholder. Covered by `tests/test_cpp_backend_serial_ttft.py`.

**Building two engines in one process penalizes the second by ~10%, whichever
mode it is.** This invalidated the original comparison, which always built serial
first and batch second. Four *serial* engines in a row in one process:

| engine | wall |
|---|---|
| #0 | 0.6718s |
| #1 | 0.7352s |
| #2 | 0.6746s |
| #3 | 0.6741s |

Only #1 is slow, and the effect follows build position rather than mode or slot
count — an 8-slot batch engine measured 1.102× when built second and 1.004× when
built third with identical configuration. GPU clocks (1860 MHz), power state, and
host thread/child counts are identical across the slow and fast windows, and a
20-second sleep between engines does not remove it, so it is neither thermal
throttling nor teardown overlap. Root cause not identified; a leaked NCCL
communicator is a candidate, since `cached_comm`
(`cpp_engine/backends/cuda/collective/tp_comm.cpp:184`) caches per-communicator
state in a `static` map keyed partly on `id_path` and never destroys entries, but
I could not isolate it (TP=1 cannot hold this 27B checkpoint, so there is no
NCCL-free control on this hardware).

Given one config per process the penalty disappears entirely and both modes agree
to within 0.3%, which is why the driver now spawns a separate process per
measurement and `scripts/summarize_phase35_benchmarks.py` aggregates the JSON.
The earlier in-process 1.117× "batching overhead" was this artifact, not a real
scheduler cost.

## Scheduler fix included

`QwenBatchScheduler::run_prefill_batch` and `run_decode_batch` returned `void`,
so the scheduler loop could not tell a completed forward pass from an empty one
and slept 1ms unconditionally between iterations. They now return whether a
forward pass ran, and the loop only waits when there was no work. This is worth
~18ms of the 32-step single-request measurement. Idle cost after the change is
0.2% of one core over a 10-second idle window, so the condition-variable wait
still parks correctly rather than busy-spinning.

## Reproducing

```bash
TP=4 bash scripts/run_phase35_benchmarks.sh /mnt/data2/Qwen3.8-27B-FP8 out/
```

Each mode and concurrency level runs in its own process; the summary applies the
targets. `scripts/profile_batch_single_request_overhead.py` reproduces the
build-order artifact by running several engines in one process.

## Test status

`tests/test_cpp_backend_serial_ttft.py` (4 new tests) passes. Of the existing
suites, `test_cpp_backend.py`, `test_cpp_backend_tp.py`,
`test_cpp_backend_worker.py`, `test_factory_tp_supervision_reuse.py`, and
`test_supervisor_orphan_guard.py` pass; 3 failures in
`test_cpp_backend_batching.py::test_serial_mode`, `::test_batch_mode`, and
`test_cpp_backend_tp.py::test_tp_requires_nccl_id_path` predate these changes and
reproduce on a clean tree.
