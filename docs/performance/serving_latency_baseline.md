# Serving latency baseline (vLLM convention)

This is the first record taken with `scripts/bench_serving.py`, the client that
reports TTFT, TPOT, ITL, E2EL, throughput and goodput on the definitions in
[Serving latency metrics](../guides/latency_metrics.md). Everything else under
`docs/performance/` measures the engine from the inside — `prefill_tps` and
`decode_tps` at concurrency 1 with no queue
([Benchmarking and reporting rules](../guides/benchmarking.md)). This page is the
other half: what a client sees over HTTP, queueing included.

The numbers are a baseline, not a ceiling. The workload is small and
rate-limited so that it is quick to reproduce and hard to misread; a saturating
run is a different record with a different TTFT.

## Scope

The measured path:

- Qwen3.8-27B safetensors, `/mnt/data1/modelscope/Qwen/Qwen3.8-27B`
- native `pocketllm_engine`, TP4 across four Ascend 910B (first generation)
- OpenAI `/v1/chat/completions`, streaming SSE
- `--no-kv-paged`, `max_context` 8192, batch width 8, prefill budget 4096
  (the last three as printed by the server at startup)
- commit `0de9d41`, binary `cpp_engine/build-ascend/pocketllm_engine`

The recorded tree is **identical to master's** — `git diff 0de9d41 1638d07` is
empty — so the record applies to master as it stands. The CUDA path was not
measured: no 2080 Ti host was involved.

## Reproduction

```bash
source scripts/ascend_env.sh     # without it an ACL binary hangs before aclInit returns
python scripts/bench_serving.py \
    --ckpt /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
    --binary cpp_engine/build-ascend/pocketllm_engine \
    --devices 0,1,2,3 --device-style ascend \
    --endpoint /v1/chat/completions \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 24 --request-rate 2 --max-concurrency 8 --num-warmups 1 \
    --goodput ttft:2000 tpot:200 e2el:30000 \
    --log-dir /tmp/serve-logs --json-out /tmp/serve.json
```

One warmup request is discarded before the measured run; the server is launched
and torn down by the harness. `--token-latch content` is the default and is what
was used, so `ttft` is latched on the first chunk carrying content, not on the
first chunk.

## Results

24 requests, 0 failures, 60.259 s of measured wall time. Latencies in
milliseconds; E2EL in seconds.

| Metric | Mean | Median | Std | P99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 1018.02 | 977.17 | 473.21 | 2276.78 |
| TPOT | 151.30 | 155.60 | 10.40 | 165.64 |
| ITL | 151.33 | 132.70 | 103.13 | 561.78 |
| E2EL | 18.831 | 19.498 | 2.111 | 21.620 |

| Quantity | Value |
| --- | ---: |
| Completed / failed | 24 / 0 |
| Generated tokens | 2849 (per-request output lengths 97–128) |
| Request throughput | 0.398 req/s |
| Output throughput | 47.279 tok/s |
| Peak output tokens/s | 64 |
| Peak concurrent requests | 10 |
| Request goodput | 0.382 req/s |
| Total token throughput | not reported — see below |

The goodput figure is 23 of the 24 requests meeting *all three* of
`ttft:2000 tpot:200 e2el:30000` (milliseconds); the remaining request is inside
the latency means above and is only excluded from goodput, not from them.

Per stream, TPOT 151.30 ms is 6.61 tokens/s; the aggregate 47.28 tokens/s is what
the concurrent streams add up to. The per-token figure is well above the 53 ms a
single request costs on this hardware, because a packed batch is charged to every
stream in it — the client's measurement and the engine's own are reconciled in
[The client and the engine agree on the per-token time](#the-client-and-the-engine-agree-on-the-per-token-time).

### The role-chunk cost, measured

The one number that only a chat-endpoint run can produce is the gap between the
two latching rules:

| Latch | Mean | Median | P99 |
| --- | ---: | ---: | ---: |
| `ttft` (first chunk with content) | 1018.02 | 977.17 | 2276.78 |
| `ttft_first_chunk` (vLLM's rule) | 4.76 | 4.59 | 7.82 |

The difference is **1013.26 ms** on average. That is the price of the role-only
delta both servers write before the backend is called: a client that latches TTFT
on the first chunk of the SSE stream reports 4.76 ms here, which is the time to
open the socket and read the role chunk, and none of the queue-and-prefill cost
that follows. Strict vLLM parity and honest measurement are 213× apart on this
workload, which is exactly the reason the harness reports both rather than
picking one silently.

In a first-token sense TTFT is the *only* per-request latency that can be
distorted this way: TPOT and ITL are differences between later tokens, where both
latching rules agree.

## What these numbers are not

- **TTFT includes queueing and prefill.** The arrivals are rate-limited to 2 req/s
  with up to 8 concurrent, so part of the mean 1018.02 ms is waiting for a slot,
  not prefill. A TTFT taken at concurrency 1 is a different number and neither is
  wrong; the arrival rate is what makes them comparable, and it is above.
- **Input tokens are unavailable to the client.** The servers emit no usage chunk
  while streaming and `--tokenizer` was not passed, so `total_input_tokens` is
  `null` and `total_token_throughput` is not reported. The server's own log
  records `prompt_tokens=320` for every request, which is the real count — the
  512 in the command line is the random dataset's *target* length before
  tokenization.
- **Output token counts are the client's**, taken from the number of
  token-bearing chunks because no usage chunk arrives mid-stream. This run makes
  that visible: the length budget was 128 and the requests returned 97 to 128
  tokens, ending early on EOS in some cases.
- **The peak-concurrency figure exceeds the configured cap** (10 against
  `--max-concurrency 8`). The cap is enforced by the client's rate limiter and
  this run shows it is not strict at the boundary; treat the column as an
  observation rather than a guarantee.
- **P99 over 24 requests is nearly the maximum.** With one percentile requested
  over 24 samples, the P99 column is the top one or two observations,
  interpolated — it says "the slowest request", not "one request in a hundred".
  Read the median for the centre and the P99 only as a bound.
- **This does not measure the engine's decode rate.** The serving path adds HTTP,
  chat templating, detokenization and scheduling, and the launcher differs from
  the engine-internal records (no paged KV cache on this path, batch width 8).
  The `prefill_tps` / `decode_tps` pages are the ones to compare against a kernel
  change; this page is the one to compare against an SLO.

### The client and the engine agree on the per-token time

The client's TPOT is a socket-arrival measurement, so it is worth checking against
the engine's own clock rather than trusting it. On the same commit, a scrape taken
around a single streamed `/v1/chat/completions` with `max_tokens=8` records

| Server-side series | Value | Implied |
| --- | ---: | ---: |
| `pocket_inter_token_latency_seconds_sum` | 0.751991 s | 107.4 ms per interval (7 intervals) |
| `pocket_request_time_per_output_token_seconds_sum` | 0.107427 s | the per-request mean of the same gaps |
| `pocket_request_prefill_time_seconds` | 0.122785 s | prefill, queue 0.000011 s |
| `pocket_request_duration_seconds` | 0.884885 s | 0.010080 s above queue+prefill+decode |

107 ms of production-side gap against the 128–151 ms the client reports at similar
concurrency is the expected direction: the client also pays HTTP, detokenization
and the drain loop. The headline numbers on this page are therefore not an
artifact of the client, and the engine-internal `decode_tps` pages can be
reconciled with them by the batch width and the serving overhead.

## A superseded smoke record

An earlier, smaller smoke run of the same client — 4 prompts, 8 output tokens
each, `/v1/completions` — recorded a mean TTFT of **54.39 ms** and a mean TPOT of
**15.78 ms** at commit `4530e4b`. Those numbers are superseded, and not merely
because that tree is old.

**15.78 ms per token is faster than this engine has ever been measured going, on
this checkpoint, on this hardware, at any row count.** The single-request record
for Qwen3.8-27B on the same four 910B cards is **103.8 ms per decode step**
(9.63 TPS) at the shipped configuration, **53.0–53.9 ms** (18.56–18.86 TPS) with
the hand-written collective and its device-side wait — opt-in when that pair was
measured, both the backend's defaults now — and **39.4–39.8 ms**
(25.15–25.37 TPS) once the Cube's row replication is added on top — the fastest
step measured anywhere on this checkpoint
([Ascend 910A single-request decode](ascend_single_request_tps.md)).
The smoke reports a step 2.5× faster than the fastest one ever recorded, while
carrying four concurrent streams rather than one. A measurement that beats the
platform's own floor by that margin is not measuring this platform's decode.

Re-running the identical workload on the tree this page records gives, on both
endpoints:

| Endpoint | `max_context` | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: |
| `/v1/completions` | 8192 | 476.9 ms | 129.4 ms |
| `/v1/completions` | 4096 | 470.2 ms | 127.5 ms |
| `/v1/chat/completions` | 8192 | 471.4 ms | 128.4 ms |

The 4096 row also rules out `max_context` as the cause, since the smoke ran at
4096. The three agree with each other to within 2%, with the server-side scrape
above, and with the engine-internal record's order of magnitude once the batch
width is accounted for.

The cause of the earlier pair was not identified. The engine sources changed
between the two runs (`e20b703`, the native server's latency-metrics alignment),
so they are not the same binary — but that commit moved where the server takes
its samples; it did not make the engine eight times slower, and the argument
above does not rest on it. No number on this page or on any other depends on the
earlier pair: the 24-request run is the one to cite.
