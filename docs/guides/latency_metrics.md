# Serving latency metrics (vLLM convention)

[Benchmarking and reporting rules](benchmarking.md) define PocketLLM's
**engine-internal** numbers: `prefill_tps` and `decode_tps`, measured at
concurrency 1 with no queue in front of the request. They answer "how fast is one
forward pass", not "what does a client see".

This page defines the second set: the **serving** numbers a client observes at a
controlled arrival rate — TTFT, TPOT, ITL, E2EL, throughput and goodput. They are
defined exactly the way `vllm bench serve` defines them, so a PocketLLM row and a
vLLM row can go in one table without an argument about what the words mean. The
formulas below were read from the upstream source, not from documentation prose;
the file and line each came from is in [Provenance](#provenance).

Client: `scripts/bench_serving.py`. Server: `/metrics` on both servers.

## The metrics

| Metric | Definition | Unit |
| --- | --- | --- |
| **TTFT** | Request start → first token-bearing chunk at the client. **Includes queueing and prefill.** | s |
| **TPOT** | `(E2EL − TTFT) / (output_tokens − 1)`. Excluded from the summary when `output_tokens ≤ 1`. | s |
| **ITL** | Each token-bearing chunk's arrival minus the previous one. Pooled across requests for the summary. | s |
| **E2EL** | Request start → last token-bearing chunk. | s |

Throughput is reported four ways, all over the benchmark duration:

```text
request_throughput       = completed / duration                      (req/s)
output_throughput        = total_output_tokens / duration            (tok/s)
total_token_throughput   = (total_input + total_output) / duration   (tok/s)
max_output_tokens_per_s  = peak of the per-second token histogram
max_concurrent_requests  = peak of the per-second concurrency histogram
```

**Goodput** is the throughput of requests that met *every* configured service
level objective, not the throughput of requests that succeeded:

```bash
--goodput ttft:2000 tpot:60 e2el:30000    # values in milliseconds
```

A request that meets two of the three SLOs is not goodput. The keys are `ttft`,
`tpot` and `e2el`; a request with `output_tokens ≤ 1` is treated as `tpot = 0`
for this test, so a single-token response is never excluded by its TPOT.

Every metric is reported as mean, median, standard deviation and percentiles.
The percentile set defaults to `99` (`--metric-percentiles 25,50,99` for more),
and which metrics get percentiles defaults to `ttft,tpot,itl`
(`--percentile-metrics`, add `e2el`). Standard deviation is the **population**
standard deviation and percentiles are **linearly interpolated**, because those
are numpy's defaults and vLLM's numbers are numpy's.

## Two things that make a naive reading wrong

### 1. TTFT includes queueing

TTFT is measured from the moment the client sends the request. If the request
waits behind a full batch before it is admitted, that wait is inside TTFT. A TTFT
measured at concurrency 1 is therefore **not** comparable with a TTFT measured at
`--request-rate 16`: same server, same prompt, different number. Always publish
the arrival rate and `--max-concurrency` alongside it.

This is also why `--num-warmups` exists. The engine's first request is not free —
kernel module load, allocator growth, and on multi-rank runs the first collective —
so a measurement taken without it charges that cost to whichever request happened
to be first. It defaults to 0, matching vLLM; the concurrency acceptance harness
defaults to one discarded round for the same reason.

### 2. The first SSE chunk is not the first token

Both PocketLLM servers write a role-only delta before the model is called at all,
so the first chunk on the wire arrives before any prefill work has happened. A
client that latches TTFT on the first chunk therefore reports a TTFT short by the
entire queue-and-prefill cost.

vLLM's own client has this blind spot: it latches on the first chunk with a
non-empty `choices` array and explicitly tolerates an empty `text`
(`endpoint_request_func.py:236-245`). So strict parity and honest measurement are
**different numbers**, and `scripts/bench_serving.py` reports both:

| Reported field | Latches on |
| --- | --- |
| `ttft` | First chunk carrying non-empty `delta.content`, `delta.reasoning_content` or `text` — the headline |
| `ttft_first_chunk` | First chunk with a non-empty `choices` array, empty text included — what a vLLM client would report |

`--token-latch {content,first-chunk}` selects which one drives the headline and
the ITL series; `content` is the default. Their difference is itself the
diagnostic: it prices the server's role-chunk emission, and it is what lets a
head-to-head against a real vLLM server be defended either way.

Note that `/v1/completions` emits no role chunk, so there the two coincide. The
distinction is specific to `/v1/chat/completions`.

## Naming map: vLLM series → PocketLLM series

The names stay `pocket_*` / `pocketllm_*`; only the definitions are aligned.
There are no `vllm:` aliases. The native server prefixes with `pocket_` and the
Python server with `pocketllm_`, and a series exists only where the table names
one:

| vLLM series | Native server (`cpp_engine`) | Python server (`pocketllm`) |
| --- | --- | --- |
| `vllm:time_to_first_token_seconds` | `pocket_ttft_seconds` | `pocketllm_ttft_seconds` |
| `vllm:e2e_request_latency_seconds` | `pocket_request_duration_seconds` | `pocketllm_request_duration_seconds` |
| `vllm:inter_token_latency_seconds` | `pocket_inter_token_latency_seconds` | `pocketllm_inter_token_latency_seconds` |
| `vllm:request_time_per_output_token_seconds` | `pocket_request_time_per_output_token_seconds` | `pocketllm_request_time_per_output_token_seconds` |
| `vllm:request_queue_time_seconds` | `pocket_request_queue_time_seconds` | — |
| `vllm:request_prefill_time_seconds` | `pocket_request_prefill_time_seconds` | — |
| `vllm:request_decode_time_seconds` | `pocket_request_decode_time_seconds` | — |

Bucket vectors are copied verbatim from upstream rather than chosen here, so a
`_bucket` line from a PocketLLM scrape can be compared with the vLLM line of the
same name line by line. Both servers use the same three vectors:

| Upstream family | Bounds | Used by |
| --- | --- | --- |
| `request_latency` | 21, `0.3` … `7680` | E2EL, queue, prefill, decode |
| `time_to_first_token` | 22, `0.001` … `2560` | TTFT |
| `inter_token_latency` | 19, `0.01` … `80` | ITL, and the per-request TPOT mean |

Copying the vector is not cosmetic. The families this repository had before the
alignment topped out at 5 s for TTFT, which is below a single 6497-token prefill
on the 2080 Ti baseline, so on that workload every real sample landed only in
`+Inf` and the histogram could not be quantiled at all.

!!! note "The `le` label is spelled differently by the two servers"
    The Python exporter writes ``le="1.0"`` and the native one writes
    ``le="1"``, because C++'s default float formatting drops the trailing
    zero. The bucket *sets* are identical, and both are valid Prometheus; a
    dashboard that matches on the label text rather than on the parsed bucket
    bound will see the difference.

### What each server observes, and what it does not

**Native server.** All seven series. It owns the scheduler's clock, so it
reports the three request phases directly and derives TTFT from the same
scheduler result. The phase columns have independent counts, unlike vLLM's,
which observes all three for every finished request: a request cancelled before
its first token has a real queue wait and no prefill interval, and it records
only what it has.

**Python server.** The four latency series, and only the four. It has no queue
term in its timing surface and its streaming path yields no final result to
read a phase split from, so the three phase series are absent rather than
approximated — a permanently-zero `_count` would read as "no queueing" instead
of "not measured".

Two further caveats on the Python server:

- **`pocketllm_ttft_seconds` and the two per-token series are streaming-only.**
  A non-streaming request has no per-token boundary to observe, so it records
  `pocketllm_request_duration_seconds` and nothing else. The families are still
  exported from process start, at zero, so a rate over the first scrape window
  is defined; read them together with their `_count`.
- **A single-token response contributes no ITL and no TPOT sample**, matching
  vLLM's `output_len ≤ 1` exclusion on both sides.

## Server-side vs client-side TPOT

Upstream observes server-side TPOT as the **per-request mean**
(`finished_request.mean_time_per_output_token`), and this repository follows: one
sample per request that produced at least two tokens, not one sample per gap. So

```text
<prefix>_inter_token_latency_seconds_count ≡ Σ(n_i − 1)
<prefix>_request_time_per_output_token_seconds_count ≡ #{requests with n_i ≥ 2}
```

where `<prefix>` is `pocket_` on the native server and `pocketllm_` on the
Python one. Summed over one stream that produced `n ≥ 2` tokens,

```text
inter_token_latency_seconds_sum == request_time_per_output_token_seconds_sum × (n − 1)
```

— because the gaps the first family sums are the very intervals the second
averages. The identity is per request: a scrape pools several requests, and
requests with different output lengths have no single multiplier between the two
sums. Both servers satisfy it by construction, since both derive the two samples
from one set of intervals. A client-side ITL computed from chunk arrival times
will **not** match the server-side sum, because the client's clock starts at the
socket and the server's at token production; report which side a number came
from.

## Relation to the prefill/decode convention

The two conventions are complementary, not competing:

| | `prefill_tps` / `decode_tps` | TTFT / TPOT / ITL |
| --- | --- | --- |
| Measured by | the engine, internally | a client, over HTTP |
| Answers | how fast is one forward pass | what latency does a request see |
| Includes queueing | no | yes, for TTFT |
| Includes HTTP and detokenization | no | yes |
| Best for | kernel and model optimization | capacity planning, SLO compliance |

The one place they touch is the phase boundary. The engine convention assigns the
first generated token to prefill; the serving convention assigns it to TTFT and
starts TPOT at the second token. Both are stated in their own terms, and neither
replaces the other. A benchmark that reports one must say which — the rule in
[Benchmarking and reporting rules](benchmarking.md#timing-convention) still
applies to any number that claims to be `prefill_tps` or `decode_tps`.

## Invocation

```bash
# Launch a native server and measure it. --device-style picks how the ranks
# select a card: cuda (the default) uses CUDA_VISIBLE_DEVICES, ascend passes the
# absolute card index and disables the HCCL whitelist.
python scripts/bench_serving.py \
    --ckpt /path/to/checkpoint \
    --binary cpp_engine/build-ascend/pocketllm_engine \
    --devices 0,1,2,3 --device-style ascend \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 32 --request-rate 4 --max-concurrency 8 \
    --goodput ttft:2000 tpot:60 --json-out /tmp/serve.json

# Measure a server somebody else started (the vLLM head-to-head mode).
python scripts/bench_serving.py --base-url http://127.0.0.1:8000
```

On Ascend the launch adds `--no-kv-paged` by itself, because the engine rejects a
paged KV cache on the batched decode path outright
(`cpp_engine/engine/qwen_engine.cpp:4501`); the CUDA launch is unchanged.

`--num-prompts 1000` and `--request-rate inf` are the defaults, matching vLLM:
by default the harness saturates the server. `--dataset-name random` (the default)
samples prompt lengths from `len × (1 ± --random-range-ratio)`; `--dataset-name
custom` uses the same synthetic prompts as the concurrency harness, so a serving
record and an acceptance record can share a workload.

Two caveats on the random dataset, both recorded in the JSON output: without a
tokenizer it reproduces vLLM's length *distribution* rather than its exact token
ids (its own sampler draws real vocabulary ids), and streaming responses carry no
`usage` chunk, so the input token count is reported as unavailable unless the
server supplies one. Pass `--tokenizer <path>` to count generated tokens by
re-tokenizing the text, which is exactly what vLLM falls back to.

## Provenance

Read from upstream `main` on 2026-09-18 via `raw.githubusercontent.com`
(`docs.vllm.ai` is not reachable from the development hosts):

| Definition | Source |
| --- | --- |
| TPOT formula, `output_len ≤ 1` exclusion, goodput rule, throughput fields | `vllm/benchmarks/serve.py:588-782` |
| Streaming TTFT/ITL latching, `output_tokens` fallback | `vllm/benchmarks/lib/endpoint_request_func.py:226-262, 403-436` |
| Arrival intervals (gamma, `burstiness`), rescaling | `vllm/benchmarks/serve.py:438-490` |
| CLI defaults | `vllm/benchmarks/serve.py:1609-1850` |
| Server series and bucket vectors | `vllm/v1/metrics/loggers.py`, `vllm/v1/metrics/buckets.py` |

The client behaviour is pinned by `tests/test_bench_serving_metrics.py`, which
drives a stub SSE server that emits the role chunk early and the token late — the
shape that breaks a naive latch.
