# Ascend 910B serving through the Python front end

4 × Ascend 910B (first generation, `Short_SoC_version=Ascend910`, 32 GB HBM each), TP4, one card a
rank, CANN 9.0.0, commit `b5f791c`. Checkpoint `prism-ml/Ternary-Bonsai-2-27B-gguf` — see
[the model guide](../models/ternary-bonsai-2-27b.md). Workload: `scripts/bench_serving.py`,
`--dataset-name custom`, 16 prompts of 243 input tokens and 64 output tokens each.

This is the first record of `pocketllm serve` answering requests on the 910B. Until the four changes
below, the host had no HTTP serving path at all: the C++ binary's own front end was deleted in
[#479](https://github.com/lvyufeng/PocketLLM/pull/479) and `scripts/build_ascend.sh` did not build
`pocketllm_cpp`, the pybind module `--backend cpp` loads — the runtime that declares `ascend` in
`pocketllm/backends/capabilities.py` was the one runtime the Ascend host could not build. The gap was
recorded as [the migration note's "Ascend consequence"](../migration/native-serve-front-end-removed.md#the-ascend-consequence);
what closes it is [issue #478](https://github.com/lvyufeng/PocketLLM/issues/478).

## 1. The bring-up

```bash
source scripts/ascend_env.sh
python -m pocketllm serve \
    --model /mnt/data1/modelscope/prism-ml/Ternary-Bonsai-2-27B-gguf \
    --backend cpp --device ascend --device-ids 0,1,2,3 \
    --tensor-parallel-size 4 --served-model-name ternary-bonsai-27b \
    --host 127.0.0.1 --port 8123
```

The supervisor forks four rank children, one a card, each printing `POCKETLLM_RANK_READY rank=<n>`;
`/health` answered `{"status": "ready", "backend": "cpp", "ready": true}` within 50 s of the launch
(polled at 5 s intervals).

Four changes stand between the deletion in #479 and that bring-up, and they are three different kinds
of failure:

| Change | What it fixed |
| --- | --- |
| [#486](https://github.com/lvyufeng/PocketLLM/pull/486) | `scripts/build_ascend.sh` did not build the pybind module at all. It now names the interpreter and the NCCL root. |
| [#487](https://github.com/lvyufeng/PocketLLM/pull/487) | The GGUF-only checkpoint did not reach the engine: the front end handed it a path the native loader could not open. |
| [#485](https://github.com/lvyufeng/PocketLLM/pull/485) | **The one that looked like a hardware fault.** `ThreadingHTTPServer` runs one thread a request, and an ACL context is per-thread state; a request thread that had never called `aclrtSetDevice` entered the engine unbound. |
| [#488](https://github.com/lvyufeng/PocketLLM/pull/488) | `logprobs: true` was admitted and then answered with a 500 rather than refused by name; see §6. |

`--device-ids 0,1,2,3` is one card a rank; nothing in this record exercised the other four.

## 2. The ladder

`--max-concurrency` was the only thing that changed between arms. No `--max-batch-size` was passed,
so the width is `DEFAULT_BATCH_SLOTS = 8` (`pocketllm/backends/cpp_backend.py:70`).

| arm | concurrency | output tok/s | req/s | TTFT | TPOT | E2EL | `requests_running` | `requests_waiting` | `slots_free` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| non-streaming | 1 | 19.80 | 0.309 | 3231.6 ms | — | 3231.6 ms | 1 | 0 | 7 |
| non-streaming | 2 | 27.12 | 0.424 | 4715.2 ms | — | 4715.2 ms | 2 | 1 | 6 |
| non-streaming | 4 | 42.35 | 0.662 | 6019.7 ms | — | 6019.7 ms | 4 | 3 | 4 |
| non-streaming | 8 | **57.06** | 0.892 | 8938.8 ms | — | 8938.8 ms | 8 | 7 | 0 |
| non-streaming | 16 | 57.15 | 0.893 | 13575.1 ms | — | 13575.1 ms | **8** | 12 | 0 |
| non-streaming | 32 | 57.21 | 0.894 | 13573.0 ms | — | 13573.0 ms | **8** | 13 | 0 |
| streaming | 1 | 20.02 | 0.313 | 385.1 ms | 44.63 ms | 3196.6 ms | **0** | 0 | 8 |
| streaming | 2 | 19.69 | 0.308 | 3453.0 ms | 45.09 ms | 6293.4 ms | **0** | 0 | 8 |
| streaming | 4 | 19.91 | 0.311 | 8823.6 ms | 44.91 ms | 11652.8 ms | **0** | 0 | 8 |
| streaming | 8 | 19.87 | 0.310 | 17269.3 ms | 45.02 ms | 20105.5 ms | **0** | 0 | 8 |

Throughput is the client's `output_throughput` — 1024 generated tokens over the benchmark's own
measured duration, the vLLM convention of [Serving latency metrics](../guides/latency_metrics.md).
The three scheduler columns are the peak (or, for `slots_free`, the floor) of a `/metrics` sampler
reading once every 0.4 s for the length of the arm.

The same 1024 tokens counted on the server's own counter over the arm's wall clock read 19.69, 26.36,
39.31, 50.76, 50.93, 50.79 for the non-streaming arms and 19.93, 19.58, 19.80, 19.74 for the
streaming ones — within 1% of the client at concurrency 1 and 11% below it at 8, the gap being the
client's fixed startup and drain around a shorter measured window.

**The non-streaming path scales 2.89× from 1 to 8, and then stops.** The streaming path does not
scale at all: 20.02 → 19.87 tok/s across the whole range while TTFT rises 385 ms → 17.3 s. Those are
two different facts and the rest of this page is the two of them.

## 3. Where the ladder's shape comes from

Every request in this workload has the same prompt length and generates exactly 64 tokens, so the
scheduler's behaviour is legible from the per-request timestamps the harness records. The
non-streaming arms run in **waves the width of the batch**:

| concurrency | waves | requests a wave | wave wall | aggregate tok/s | speedup | a client's share of the solo rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 1 | 3.23 s | 19.80 | 1.00× | 100% |
| 2 | 8 | 2 | 4.71 s | 27.18 | 1.37× | 68.6% |
| 4 | 4 | 4 | 5.99 s | 42.74 | 2.16× | 53.9% |
| 8 | 2 | 8 | 8.93 s | 57.34 | 2.90× | 36.1% |
| 16 | 2 | 8 | 8.94 s | 57.15 | 2.89× | 18.0% |
| 32 | 2 | 8 | 8.94 s | 57.21 | 2.89× | 9.0% |

At 16 and 32 client requests the wave count stays at **two**: eight are admitted and run to
completion, the rest wait, and the second wave begins when the first ends. That is the plateau — 8 is
the slot count, not a property of the hardware. `requests_waiting` rising to 12 and then 13 while
`requests_running` stays pinned at 8 is the same fact from the server's side. The last column is the
aggregate rate divided by the number of clients, so the 16- and 32-client rows price the queueing for
a second wave on top of the batching; only the first four rows are batching alone.

The one-row row of that table decomposes, because the streaming arm measures the same request's two
halves separately: 385.1 ms of prefill for 243 prompt tokens, then 63 intervals at 44.63 ms — so
64 × 44.6 ms of decode plus the prefill is the 3.20 s the non-streaming arm sees end to end. The
higher rows cannot be decomposed this way: the non-streaming arm reports one number, and the batched
prefill's own cost is inside it.

What the shapes say together is that a decode step gets more expensive faster than it gets wider.
Eight rows take 2.8× as long a step as one row for 8× the work, which is why the aggregate rate
improves 2.9× while a single request's own rate falls to 36% of what it was alone. The ladder's
throughput and the ladder's latency are the same measurement read in two directions.

## 4. The streaming arm never reaches the scheduler

`requests_running` is **0 on every streaming arm**, `requests_waiting` is 0, and `slots_free` never
leaves 8. Not one streaming request entered the batch scheduler, and the throughput is flat at ~19.9
tok/s — which is the concurrency-1 rate, held no matter how many clients are attached.

The cause is a lock. `CppBackend.stream` holds `self._request_lock` around the entire generator
(`pocketllm/backends/cpp_backend.py:1995`), where `generate` reaches `_generate_batched` without it:

```python
def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
    # A primitive lock can span generator yields even when AsyncLLM resumes
    # the generator on a different executor thread. It serializes all access
    # to the native engine's single mutable KV-cache session.
    self._begin_request(request.request_id)
    with self._request_lock:
        ...
        yield from self._stream_native(request)
```

The comment says why, and it is a correct description of the serialized session: the native engine
has one mutable KV-cache session and the lock is what stops two generators from interleaving into it.
What the lock also does is make the *streaming* path a serialized-path measurement on a build whose
engine has eight slots and a working scheduler. The 20 tok/s and the 17.3 s TTFT in the table above
are therefore the cost of that lock and not the cost of the machine — the same server, asked the same
way but with `stream: false`, answers 8 requests at 57 tok/s. `generate`, by contrast, dispatches to
`_generate_batched` (`:1615`) with no lock in the way.

Anything that measures streaming must say which of the two it measured. This page's streaming arms
are the contrast case, not a ceiling.

## 5. A repeated prompt is not cheaper

`reads_prefix_cache` is `bool(self.args.enable_prefix_caching) and not self._batching_enabled`
(`pocketllm/backends/cpp_backend.py:794`), because the serialized session's per-slot cache is reached
from `QwenEngine::prefill` and the batch scheduler prefills through `QwenEngine::batch_prefill`,
which never enters that lookup. Batching is on by default, so on this server the cache is off by
construction — and it measures that way. One 608-token prompt sent four times back to back:

| call | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| wall | 1.555 s | 1.548 s | 1.572 s | 1.575 s |

and three *different* prompts of the same length: 1.557 s, 1.533 s, 1.584 s. A repeat costs what a
novel prompt costs, to within the 27 ms spread of the three repeats. The cache is not failing here;
it is not being consulted, and a serving record that claimed a prefix-cache win through this front
end would be claiming a lookup that does not happen on the path it measured.

## 6. The field surface

Refusals use the OpenAI error shape with `type` `invalid_request_error`, `param` naming the field and
`code` `unsupported_feature`, checked before dispatch so a refused request is refused whole. That
envelope and the host-side rules behind it are
[documented in the API guide](../guides/pocketllm_api.md#refused-with-http-400) and are shared with
every runtime.

| request | status | `param` | `code` |
| --- | ---: | --- | --- |
| chat `logprobs: true` | 400 | `logprobs` | `unsupported_feature` |
| chat `logprobs: true, top_logprobs: 3` | 400 | `logprobs` | `unsupported_feature` |
| completions `logprobs: 3` | 400 | `logprobs` | `unsupported_feature` |
| chat `response_format: {"type": "json_object"}` | 400 | `response_format` | `unsupported_feature` |
| chat `response_format: {"type": "json_schema", ...}` | 400 | `response_format` | `unsupported_feature` |
| chat `top_logprobs: 3` | 400 | `top_logprobs` | `unsupported_feature` |
| chat `frequency_penalty: 0.5` | 400 | `frequency_penalty` | `unsupported_feature` |
| chat `stop: 5` | 400 | `stop` | `unsupported_feature` |
| completions `echo: true` | 400 | `echo` | `unsupported_feature` |
| chat `logprobs: false` | 200 | — | — |

The first five rows are the engine's own declaration; the three after them are the host's shape
rules and read identically on every backend; the last row is a value the server already implements,
kept so a client that sends the documented default explicitly is not punished for it.

The engine-declared five are read off the engine rather than special-cased for Ascend, and the
machinery is worth naming because it is exactly where this used to be wrong. `caps()` returns
`c.structured_outputs = false` and `c.logprobs = false` under `POCKET_BACKEND_ASCEND`
(`cpp_engine/engine/qwen_engine.cpp:5309`, `:5322`), the Ascend build having no device sampler to
constrain or rank with. `_served_fields` (`:1174`) asks the *engine* that question, through
`_rankings_available` (`:1223`) and `_constraints_available` (`:1254`), rather than asking the
scheduler alone: the scheduler is what carries the value, but it carries only what the engine
produced, so a build that admits the field on the scheduler's answer and then has nothing to fill it
with returns a truncated ranking under a 200. That is what `logprobs: true` did here before
[#488](https://github.com/lvyufeng/PocketLLM/pull/488) — a 500, not a 400 — and the fix was to ask
the second question, not to add an Ascend branch to the audit. The CUDA host reaches the same code
with `caps().logprobs` true, so it answers `logprobs: true` with a 200 and the two backends differ
only in what their engines declare.

`/metrics` reports the twelve scheduler series — `pocketllm_requests_{total,active,running,waiting}`,
`pocketllm_slots_free`, `pocketllm_generation_tokens_total`, `pocketllm_prompt_tokens_total`,
`pocketllm_request_errors_total` and the four histograms (`inter_token_latency_seconds`,
`request_duration_seconds`, `request_time_per_output_token_seconds`, `ttft_seconds`) — with the same
names and the same definitions as the CUDA host, which is what makes §2's scheduler columns readable
against the CUDA records in this directory.

## 7. What this ladder cannot say

1. **It cannot tell continuous refill from static batching.** Every request generates exactly 64
   tokens, so every slot in a wave frees at the same moment and the two policies are
   indistinguishable on this workload. A mixed-length workload is the measurement that separates
   them, and the plateau's shape at 16 and 32 clients would look different under each.
2. **The non-streaming arms carry no TPOT or ITL.** The host latches both on the streaming path, and
   a non-streaming response has one timestamp; the zeros in `bench_serving.py`'s output for those
   arms are the harness declining to invent a decode rate, not a measurement of zero. The streaming
   arm is where a decode rate lives, and on this build it is a serialized-path decode rate (§4).
3. **There is no CUDA comparison in this record.** The ladder says what the 910B does; it does not
   say what the same checkpoint does on a 2080 Ti, and a cross-backend ratio would need that arm
   measured rather than assumed.
4. **Nothing here exercised more than four cards**, and nothing here is a long-context result: 243
   prompt tokens says nothing about a 4966-token prefill, which on this backend is a different
   measurement with its own record.
5. **The scheduler never had to evict.** With 8 slots and a 64-token output there is no contention
   for KV, and no arm reports what happens when a wave's rows would not fit the arena.

## 8. Reproducing

```bash
source scripts/ascend_env.sh
python -m pocketllm serve \
    --model /mnt/data1/modelscope/prism-ml/Ternary-Bonsai-2-27B-gguf \
    --backend cpp --device ascend --device-ids 0,1,2,3 \
    --tensor-parallel-size 4 --served-model-name ternary-bonsai-27b \
    --host 127.0.0.1 --port 8123

python scripts/bench_serving.py --base-url http://127.0.0.1:8123 \
    --endpoint /v1/chat/completions --model ternary-bonsai-27b \
    --dataset-name custom --random-input-len 256 --random-output-len 64 \
    --num-prompts 16 --num-warmups 1 --seed 0 \
    --max-concurrency 8 --json-out /tmp/ladder.json
```

`--no-stream` selects the non-streaming arm. One arm per `--max-concurrency` value; the arms in §2
were run serially against one server, with the scheduler columns read from `/metrics` on a sampler
thread rather than out of the response bodies.

Numbers from a single run each. Nothing here was repeated often enough to put error bars on, and the
differences that matter — 2.89× across the ladder, 0 scheduler requests on every streaming arm — are
far outside the run-to-run spread this backend has shown elsewhere.
