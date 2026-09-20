# Serving concurrency, throughput and the TFLOPS behind them

This page pushes on the workload
[serving_latency_optimized.md](serving_latency_optimized.md) left at 70.29 tok/s:
how many streams the engine actually runs at once, what raising that number
costs in latency, where the throughput curve stops paying, and what the resulting
decode and prefill steps are worth in TFLOP/s against the card's peak.

It is a **run record of the opt-in configuration**. Every timing below was taken
with `POCKET_ASCEND_IPC_ALLREDUCE=1 POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1`, which
is not the shipped path — `ipc_allreduce.hpp:49` documents both as opt-in and the
engine prints a WARNING for the device-side wait at startup. The shipped
configuration is the [baseline page](serving_latency_baseline.md)'s.

Headline: **112 concurrent streams is where this configuration stops.** At
`--max-context 2048` the KV arena stops fitting one rank at 120 slots, the rank
exits during `device_malloc`, and the server keeps accepting requests while it
emits nothing — a silent hang rather than an error. Throughput peaks earlier, at
96 slots.

## Scope

- Qwen3.8-27B safetensors, `/mnt/data1/modelscope/Qwen/Qwen3.8-27B`
- native `pocketllm_engine`, TP4 across devices 0-3 of an 8 x Ascend 910B
  (first generation, `Short_SoC_version=Ascend910`; 32 GB HBM a card), CANN 9.0.0
- OpenAI `/v1/chat/completions`, streaming SSE
- **KV cache is the contiguous, unpaged arena, not the paged pool the CLI
  defaults to.** The default is the other way (`main.cpp:104`:
  `bool kv_paged = true;`), and reading that is what led an earlier revision of
  this page to say paging was on. It is not: `bench_serving.py` hands the launch
  to `start_server`, which appends `--no-kv-paged` whenever `--device-style
  ascend` and `--kv-paged` is unset (`bench_cpp_openai_concurrency.py:281-290`,
  added in `9c1b017`, an ancestor of `645e36b`). It has to, because
  `run_batched_decode` rejects that combination outright
  (`qwen_engine.cpp:4501`): the batched Ascend entry points address a slot by a
  constant element stride and do no block-table translation, so a paged arena
  would read the wrong rows rather than fail. The `--no-kv-paged` of the revision
  this page "corrected" was right, and
  [latency_metrics.md](../guides/latency_metrics.md) already documents the
  behaviour. **Nothing below moves**: the arena reserves the identical
  `max_batch_size x max_context x bytes a token x full-attention layers` the paged
  pool's zero-budget branch derives (`qwen_engine.cpp:1700` against `1517-1521`,
  whose own comment calls turning paging on "memory-neutral"), so only the name of
  the thing that stops fitting a rank at 120 slots was wrong.
- commit `645e36b` (master as of 2026-09-19), binary
  `cpp_engine/build-ascend/pocketllm_engine`
- every ladder row: `--random-input-len 512 --random-output-len 512
  --num-warmups 0 --request-rate inf --prefill-token-budget 4096
  --max-context 2048`, with `--max-batch-size` equal to the client's
  `--max-concurrency`; the two saturating runs and the two concurrency-limit
  probes are named where they differ
- `--random-range-ratio 0.0`, so every prompt is nominally 512 tokens and `inf`
  request rate, so the whole group is offered at once. **The nominal length is
  not the length the engine ran**: the harness has no tokenizer, so it builds
  512 tokens' worth of words at `--chars-per-token` and the server re-tokenizes.
  The engine logs `prompt_tokens=320` or `327`, a mean of **324.8** over `L32`'s
  32 requests, and that is the count every FLOP figure below is computed from.
  `analyze_serving_roofline.py --prefill-tokens` takes the engine's number and
  rejects `--random-input-len` by name for exactly this reason.

The engine sources at `645e36b` are the ones the numbers come from. The two
`report_phase_profile()` calls PR #294 added to the `--batch-decode` bench
(merged as `0d122ea`) were applied for the profiled runs only; with
`QWEN_PHASE_PROFILE` unset both return immediately, so the serving numbers are
unaffected by them.

### Reproduction

The driver is `scripts/run_serving_sweep.sh`, which fixes the arguments every
point of the sweep shares and varies the width:

```bash
source scripts/ascend_env.sh     # without it an ACL binary hangs before aclInit returns
export POCKET_SWEEP_CKPT=/mnt/data1/modelscope/Qwen/Qwen3.8-27B
scripts/run_serving_sweep.sh ladder      # L1 ... L112, then L16x64, L48x192
scripts/run_serving_sweep.sh limit       # L120, L128, L128c1024
scripts/run_serving_sweep.sh ab          # the replicate-rows A/B, interleaved
# one point on its own, which is `L32` in the ladder table:
#   point <tag> <slots> <concurrency> <prompts> <in> <out> <rate> <ctx> [K=V ...]
scripts/run_serving_sweep.sh point L32 32 32 32 512 512 inf 2048
```

It exports `POCKET_ASCEND_IPC_ALLREDUCE=1` and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1` for every point, and wraps each run in
`scripts/_serving_metrics_scrape.py`, which polls the engine's `/metrics` while
the bench is alive. That scrape is the only source of the queue / prefill /
decode split, because `bench_serving.py`'s record carries no server-side series.
It is taken from outside the bench, so it can only see counters the engine has
published while its server is up, and the bench stops that server as soon as the
last request returns. `point` therefore passes `--server-drain-seconds 1.0`,
which holds the server for a second past the measured window and moves no figure
the bench reports, because it runs after `run_measured` has returned. Without it
the scrape is short by the requests still in flight: the artifacts this page was
written from predate the flag, and `L4` recorded 3 of its 4 requests while `L1`
recorded 0 of its 1.
A point writes four artifacts under `$POCKET_SWEEP_DIR` (default `./sweep-out`):
`<tag>.json` (the bench's record), `<tag>.metrics` (the last scrape), `<tag>.out`
(the console output, kept for provenance) and `logs-<tag>/` (the per-rank engine
logs, which is where `prompt_tokens=` above comes from).

Three readers, each taking those artifacts and nothing else:

```bash
python scripts/summarize_serving_sweep.py --client    sweep-out/L*.json
python scripts/summarize_serving_sweep.py --server    sweep-out/L*.metrics
python scripts/summarize_serving_sweep.py --occupancy sweep-out/L96.json
python scripts/analyze_serving_roofline.py --prefill-ms 411 --prefill-tokens 325 sweep-out/L*.json
```

`--client` reproduces the ladder table, `--server` the phase split, and
`--occupancy` the time-weighted batch width. All figures are read from the
`--json-out` file, never from the console table, which has no `Std` column and
truncates when piped. `--occupancy` needs the per-request `start_seconds` the
record carries: `ttft_seconds` and `itl_seconds` are relative to their own
request, so a refilled run such as `L16x64` cannot be placed on a shared axis
without it. `analyze_serving_roofline.py` prints the ceilings it uses and
derives [Where the FLOPs go](#where-the-flops-go) from the ladder JSONs plus one
measured prefill.

**The output length is not the 512 that was asked for, and it is not a cap
either.** Every request in every run below ends at EOS between 57 and 130 tokens,
so the cap was never reached and no row is truncated. The `out tok` column is the
real generated total, not `prompts x 512`. TPOT is per token and is unaffected;
throughput and E2EL are not, which is why they are reported per run rather than
compared against a nominal workload.

## The concurrency ladder

One wave of prompts, `--request-rate inf`, so every prompt is offered at once and
the client holds no more than `--max-concurrency` in flight. `slots` is
`--max-batch-size`; `prompts` is `--num-prompts`. Latencies in milliseconds, E2EL
in seconds. Every row is the same command line on the same commit with the same
warm-up count; the only differences are the width and the prompt count.

| run | slots | prompts | TTFT | TTFT P99 | TPOT | TPOT std | E2EL | tok/s | req/s | goodput | out tok | out len min/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `L1` | 1 | 1 | 513.4 | 513.4 | 54.31 | 0.00 | 7.085 | 17.06 | 0.140 | 0.140 | 122 | 122/122 |
| `L4` | 4 | 4 | 1460.8 | 1777.7 | 68.28 | 4.31 | 9.789 | 48.90 | 0.398 | 0.398 | 492 | 122/126 |
| `L8` | 8 | 8 | 3032.8 | 3396.3 | 84.67 | 7.52 | 13.398 | 71.37 | 0.579 | 0.072 | 987 | 114/128 |
| `L16` | 16 | 16 | 6248.4 | 6636.1 | 153.65 | 12.12 | 24.350 | 74.22 | 0.625 | 0.000 | 1901 | 98/129 |
| `L32` | 32 | 32 | 12790.1 | 13196.4 | 212.55 | 17.68 | 37.943 | 96.14 | 0.806 | 0.000 | 3818 | 77/130 |
| `L48` | 48 | 48 | 19338.3 | 19751.9 | 261.55 | 38.08 | 46.850 | 100.65 | 0.937 | 0.000 | 5155 | 66/130 |
| `L64` | 64 | 64 | 22997.6 | 26529.0 | 341.59 | 29.51 | 57.895 | 105.39 | 1.013 | 0.000 | 6656 | 66/130 |
| `L96` | 96 | 96 | 36077.3 | 39929.3 | 471.00 | 44.86 | 80.055 | 105.07 | 1.099 | 0.000 | 9181 | 66/130 |
| `L112` | 112 | 112 | 39153.6 | 46329.1 | 683.84 | 80.86 | 100.969 | 95.15 | 1.019 | 0.000 | 10459 | 57/130 |
| `L16x64` | 16 | 64 | 2373.1 | 6631.5 | 191.94 | 26.90 | 22.177 | 71.93 | 0.688 | 0.194 | 6686 | 66/130 |
| `L48x192` | 48 | 192 | 5972.2 | 19722.1 | 432.89 | 95.47 | 41.871 | 93.43 | 1.095 | 0.000 | 16379 | 57/130 |

`L1` and `L4` are inside the control band of the `QWEN_ASCEND_REPLICATE_ROWS` A/B
further down (control 54.05-54.85 ms), so the ladder's floor reproduces an
independently measured arm.

Goodput is a **rate**, not a count, against `ttft:2000 tpot:200 e2el:30000`.
Nothing at 16 slots or above clears the 200 ms TPOT budget, so from `L16` up the
goodput column is zero however high the throughput column goes. The best goodput
on the page is `L4`'s 0.398, and among the runs that keep the batch full it is
`L16x64`'s 0.194.

**The budget is met at a lower offered concurrency, not by a lower slot count.**
[serving_latency_optimized.md](serving_latency_optimized.md) clears it — 0.564 of
0.589 req/s — with the same opt-in stack and the same nominal 512-token prompts, at 24
prompts, `--request-rate 2` and `--max-concurrency 8`. Its peak concurrency was
11 and it measured 70.29 tok/s at 99.49 ms TPOT, against `L8`'s 71.37 tok/s at
84.67 ms and `L16`'s 74.22 at 153.65. Throughput matches at the same width;
the TPOT lands between the 8- and 16-row rows because a slow arrival never holds
a full-width batch for the whole decode. The distinction that matters for
capacity planning is therefore **concurrency offered at once** versus
**concurrency sustained**, and this page measures the first.

## Maximum concurrency: the KV pool, not the flag

`--max-batch-size` is the concurrency ceiling — the engine logs it and the
scheduler admits only while `slot_to_request_.size() < max_batch_size_`
(`batch_scheduler.cpp:288`). Raising it is not free, and the failure when it is
raised too far is the worst kind:

| run | slots | ctx | engine | client | what happened |
| --- | ---: | ---: | --- | --- | --- |
| `L112` | 112 | 2048 | `batch width 112`, four ranks up | 112/112 pass | full run, 95.15 tok/s |
| `L120` | 120 | 2048 | `batch width 120`, **rank 2 dies in `device_malloc`** | 105/120 "pass", 15 failures | every request stubs out after 2 tokens; 901 s wall |
| `L128` | 128 | 2048 | `batch width 128`, **rank 2 dies in `device_malloc`** | 1/128 pass, 127 failures | one request served, 127 client timeouts at 900 s |
| `L128c1024` | 128 | 1024 | `batch width 128`, four ranks up | 128/128 pass | full run, 95.72 tok/s |

The failure is not a rejected request. `logs-L120/rank2.log` and
`logs-L128/rank2.log` each end in

```
error: device_malloc Qwen runtime tensor: [PID: ...] Memory_Allocation_Failure(EL0004)
        alloc device memory failed, runtime result = 207001
```

while rank 0 logs `[server] batch width 120 ... [server] listening on
127.0.0.1:18280` and starts reading requests. The three surviving ranks then wait
for a peer that has already exited, so the server accepts, admits and counts
every request and never produces a token — the client's only symptom is
`Never received a valid chunk to calculate TTFT.` after its 900 s timeout.

**It is the KV cache's size that decides this, and `--max-context` is half of
it.** The arena reserves
`max_batch_size x max_context x bytes a token x full_attention_layers`
(`qwen_engine.cpp:1700`; the paged pool's zero-budget branch at `1517-1521` derives
the same product), and for this checkpoint at TP4 that is 1 local KV
head x 256 head_dim x 2 tensors x 2 bytes = 1024 B a token a layer across 16
full-attention layers:

| slots | ctx | arena a rank | |
| ---: | ---: | ---: | --- |
| 112 | 2048 | 3.50 GiB | runs |
| 120 | 2048 | 3.75 GiB | rank 2 fails, +268 MiB |
| 128 | 2048 | 4.00 GiB | rank 2 fails, +537 MiB |
| 128 | 1024 | 2.00 GiB | runs |

`L128c1024` is the discriminating run: with the same `--max-batch-size 128` and
half the context it completes 128/128, so the ceiling is the product and not the
width flag. `npu-smi` reports device 2's HBM as 32741 MB against 32768 MB on
devices 0, 1 and 3, a 27 MB difference that cannot explain a 268 MiB step — the
margin is tight for a reason this record does not establish, and if the ceiling
matters to a deployment it is worth re-measuring per host rather than quoting
112 as a property of the engine. What is established is the shape: **the arena is
sized for worst-case `max_context` on every slot, so concurrency and context
length buy from the same budget**, and almost all of it is spent on slots that
never reach 130 tokens. `L112` reserves 3.500 GiB a rank and ever holds 0.160 GiB
of generation, 0.715 GiB once the re-tokenized prompts are counted — 2.8 GiB of
over-reservation in that one row. Summed over the eleven ladder points it is
13.906 GiB reserved against 4.102 GiB ever held: 9.8 GiB of reservation for
tokens that were never there. (Generation is the records' own `output_tokens`;
the prompt term is the 325-token mean the engine logs, because these artifacts
predate the harness recording `prompt_tokens`, which reads 0 in all of them.)

## Where the curve stops paying

The quantity that saturates is the **row-step rate**, rows x steps/s, which is
the work the decoder retires while the batch is full. It is not the same as the
measured aggregate, which also counts the partial-width first and last steps and
the prefill:

| slots | TPOT | ms/row marginal | row-steps/s | steps/s | measured tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.31 | — | 18.41 | 18.4 | 17.06 |
| 4 | 68.28 | 4.66 | 58.58 | 14.6 | 48.90 |
| 8 | 84.67 | 4.10 | 94.49 | 11.8 | 71.37 |
| 16 | 153.65 | 8.62 | 104.13 | 6.5 | 74.22 |
| 32 | 212.55 | 3.68 | 150.56 | 4.7 | 96.14 |
| 48 | 261.55 | 3.06 | 183.52 | 3.8 | 100.65 |
| 64 | 341.59 | 5.00 | 187.36 | 2.9 | 105.39 |
| 96 | 471.00 | 4.04 | **203.82** | 2.1 | 105.07 |
| 112 | 683.84 | 13.30 | 163.78 | 1.5 | 95.15 |

**The knee is at 96 slots, not at 48.** Rows 49 to 64 buy 3.8 row-steps/s and
rows 65 to 96 buy another 16.5; only past 96 does the step rate fall faster than
the width grows, and 97 to 112 rows *lose* 40 row-steps/s. So a 112-slot server
pays 45% more TPOT than a 96-slot one for 9% fewer tokens, and the last row of
the ladder is past the point where the width is doing anything but costing time.

The marginal ms/row column is the slope of TPOT between consecutive widths, and
it is not monotone: 4.66, 4.10, 8.62, 3.68, 3.06, 5.00, 4.04, 13.30 ms. The
16-slot entry is where the step stops being able to hide its fixed costs, and the
112-slot entry is where the arena's worst-case reservation starts to show. The
nominal peak of the measured aggregate is `L64`/`L96` at 105.4/105.1 tok/s — a
dead tie across a 50% wider batch, which is the honest statement of where this
configuration tops out.

### Saturation does not help

`L48x192` offers 192 prompts to 48 slots, so the batch is refilled as it drains
rather than being allowed to empty. It is **worse than the single wave at the
same width**: 93.43 tok/s against 100.65, TPOT 432.89 ms against 261.55. The
scheduler runs `run_prefill_batch()` and `run_decode_batch()` in the same pass
(`batch_scheduler.cpp:226`), and a continuous supply of new prompts means a
prefill is always waiting to run in front of the next decode, so the step rate
drops from 3.8 to 2.3 a second.

`L16x64` is the opposite case and is a genuine gain on TTFT: 64 prompts against
16 slots gives mean TTFT 2373.1 ms against `L16`'s 6248.4, because slots free one
at a time and each prefill group is small, at the same throughput (71.93 against
74.22 tok/s) and the best goodput among the full-batch runs. The mechanism is the
one in the next section.

## TTFT is the whole wave's prefill

TTFT in the ladder is not a queueing term. The server's own histograms separate
the two: in `L32` the queue is 443.2 ms a request while the prefill is 12241.2
ms, so **96% of TTFT is prefill**, and the prefill time grows with the width of
the wave rather than with the size of the prompt.

Per request of a ~325-token prompt, the prefill the server charged is nearly flat
in the width — 240 ms at 4 rows, 311 at 8, then 356, 383, 391, 304, 339, 287 at
16, 32, 48, 64, 96 and 112 — while mean TTFT is *exactly* linear:

| run | slots | TTFT mean | first request | the rest | spread among the rest |
| --- | ---: | ---: | ---: | ---: | --- |
| `L16` | 16 | 6248.4 ms | 517 ms | 6630 ms | 8 ms, over 15 requests |
| `L32` | 32 | 12790.1 ms | 517 ms | 13186 ms | 14 ms, over 31 requests |
| `L48` | 48 | 19338.3 ms | 523 ms | 19740 ms | 19 ms, over 47 requests |
| `L112` | 112 | 39153.6 ms | 504 ms | 38099 ms | 29 ms, over 111 requests |

Server-side the means are 6146.3, 12684.4, 19179.7 and 38809.4 ms, so the client
and the engine agree to 2%.

**The distribution is bimodal, not a ramp, and that fixes the model.** In `L48`
the 47 requests behind the first one all latch their first token within 19 ms of
each other after a 19.7 s wait — 0.1% — while exactly one latches at 523 ms,
which is `L1`'s single-request TTFT. So the engine serves the first request
through its own prefill and one decode step, and then runs the remaining N-1
prefills inside a **single blocking `batch_prefill` call** that only returns when
the last of them is done. Nobody in that group decodes before it returns, so
nobody in it sees a token before the slowest member.

TTFT is therefore `517 ms + c x (N - 1)`, where `c` is the per-prompt prefill
cost, and not the `slots/2 x prefill` of an earlier revision of this page — the
per-request distributions falsify the half-wave form. The slope `c` and the
server's own per-request prefill histogram are the same measurement taken two
ways and they agree — 411 against 356 ms at 16 rows, 412 against 383 at 32, 411
against 391 at 48, 309 against 304 at 64, 283 against 287 at 112. So the prefill
of a ~325-token prompt costs the same whether it shares a wave with 15 others or
111, with a mild drift down at the widest rows: the wave never shares a forward
pass, which is the next section's finding stated as a number.

At `L64` and above the queue term also wakes up — 3264.7 ms at 64, 3267.9 at 96,
6688.9 at 112 — because admission, not prefill, has become the wait.

### The root cause, from the engine

**`QwenEngine::batch_prefill` runs its requests one at a time.** The comment at
`qwen_engine.cpp:5087` says so — "Requests still run one at a time" — and it is
what the phase profile shows. With `QWEN_PHASE_PROFILE=1` and the batch bench's
`--batch-decode`, each request prefills through the single-sequence path, which
reports under the tag `prefill` once per request, so the number of post-reference
`prefill` blocks counts the requests:

| `QWEN_BATCH_ROWS` | post-reference `prefill` blocks | each, seconds |
| ---: | ---: | --- |
| 1 | 1 | 1.75524 |
| 4 | 4 | 1.75624, 1.71228, 1.68776, 1.71922 |
| 16 | 16 | 0.566094, and 15 more between 0.509331 and 0.533169 |

Sixteen rows give sixteen blocks, and the profiler synchronizes the device on
entry and exit of every scope (`qwen_engine.cpp:2555`), so the seconds are
inflated relative to the ~400 ms the server measures unprofiled and the bench's
own per-row prompts are shorter than the serving prompts. What transfers is the
**count**, and therefore the serial structure. A wave of N prompts is N
single-request prefills issued back to back before the scheduler next runs a
decode.

That is the whole of the TTFT scaling. The prefill GEMMs never see more than one
prompt's rows, and the 325-token prefill runs at 325/0.411 = 791 tokens/s
against a weight-read floor of 11.7 ms a pass.

### The levers that were tried against it

**A smaller prefill token budget makes TTFT worse, not better.** The budget
documents itself as the thing that stops "a long prompt holding the device"
(`qwen_engine.cpp:5092`), but it also caps how far each request advances per
call, so a 128-token budget turns every ~325-token prompt into three passes:

| run | budget | TTFT mean | TTFT P99 | TPOT | tok/s | goodput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L16` | 4096 | 6248.4 | 6636.1 | 153.65 | 74.22 | 0.000 |
| `pf128` | 128 | 12443.8 | 12580.2 | 148.90 | 59.74 | 0.000 |

TTFT doubles and throughput drops 20%, while TPOT is unchanged within its spread
— the cost is entirely in the extra passes through the 64-layer stack. For prompt
lengths near the budget the budget is a latency tax. It would earn its keep on
prompts far longer than 4096 tokens, which this record does not measure.

**Padding the batch with the client does work, and is the usable lever.** `L16x64`
is `L16` with 64 prompts offered to 16 slots, so the client holds a wave down to
16 and refills as slots free: mean TTFT 2373.1 ms against 6248.4 for a 62%
reduction, at the same throughput and the best goodput among the full-batch runs.
It costs nothing but a slot count below the offered concurrency, which is the
opposite of what a throughput-first configuration does.

## The TPOT lever at concurrency one

`QWEN_ASCEND_REPLICATE_ROWS=16` fills the Cube's sixteen-row M tile by
broadcasting a decode step's single activation row, and
[ascend_single_request_tps.md](ascend_single_request_tps.md) records 53.2-53.6 ms
falling to 39.4-39.8 ms inside the engine. It had never been measured over HTTP.
Three interleaved pairs, control and lever alternating, at `--max-concurrency 1`
and `--max-batch-size 1`:

| pair | control TPOT | replicate TPOT | change | control tok/s | replicate tok/s | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.85 ms | 40.87 ms | -25.5% | 16.91 | 21.01 | +24.3% |
| 2 | 54.05 ms | 40.33 ms | -25.4% | 17.17 | 22.47 | +30.9% |
| 3 | 54.33 ms | 40.78 ms | -24.9% | 17.04 | 21.09 | +23.8% |

The control spans 54.05-54.85 ms and the lever 40.33-40.87 ms across the three
pairs, so the arms do not overlap and every pair moves the same 25%. TTFT is
unmoved (495.8-513.4 ms in both arms) — the lever is in the decode step, and at
concurrency one the decode step is the whole of the run between the first token
and the last.

**At 16 rows the same lever does nothing**, which is what the mechanism predicts:
the M tile is already full of real rows, so a broadcast row adds no work.
`rep16` reads 152.36 ms and 72.86 tok/s against `L16`'s 153.65 and 74.22, a
difference inside the ladder's own spread.

## Where the FLOPs go

The card's ceiling is read from CANN's own platform config, not assumed:
`Ascend910B.ini` gives `ai_core_cnt=30`, `cube_m/n/k_size=16`, `cube_freq=900`,
so 30 x 16^3 x 900 MHz x 2 = **221.2 TFLOP/s a card**, 884.7 across the four.
The model's arithmetic, per token and for the whole model — the parameter row is
one rank of four, and `analyze_serving_roofline.py` prints this same table:

| quantity | value |
| --- | ---: |
| FLOP a token | 51.244 GFLOP (48 x 766.4 + 16 x 744.5 linear/full + 2542.8 head, MFLOP) |
| parameters | 26.893 B, 12.523 GiB fp16 a rank |
| engine's own `resident_weight_bytes` | 13,449,011,456 B = 12.525 GiB |
| machine balance | 192.7 FLOP/byte (at a measured 1148 GB/s HBM read) |
| decode arithmetic intensity | **0.953 FLOP/byte, 202x below the ridge** |
| prefill arithmetic intensity, 325 tokens | **1238 FLOP/byte, 6.4x above the ridge** |

Decode is therefore irreducibly memory-bound and prefill is not, and the measured
decode table says the same thing:

| rows | TPOT | row-steps/s | TFLOP/s, 4 cards | % of peak | GB/s a card | % of the 1148 GB/s probe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.31 | 18.41 | 0.944 | 0.11% | 247.6 | 21.6% |
| 4 | 68.28 | 58.58 | 3.002 | 0.34% | 197.0 | 17.2% |
| 8 | 84.67 | 94.49 | 4.842 | 0.55% | 158.8 | 13.8% |
| 16 | 153.65 | 104.13 | 5.336 | 0.60% | 87.5 | 7.6% |
| 32 | 212.55 | 150.56 | 7.715 | 0.87% | 63.3 | 5.5% |
| 48 | 261.55 | 183.52 | 9.404 | 1.06% | 51.4 | 4.5% |
| 64 | 341.59 | 187.36 | 9.601 | 1.09% | 39.4 | 3.4% |
| 96 | 471.00 | 203.82 | **10.445** | **1.18%** | 28.6 | 2.5% |
| 112 | 683.84 | 163.78 | 8.393 | 0.95% | 19.7 | 1.7% |
| 128 @ ctx 1024 | 811.14 | 157.80 | 8.086 | 0.91% | 16.6 | 1.4% |

The GB/s column is the step's own traffic if every resident weight byte were read
once, over the measured step time. It falls from 21.6% of the probe at one row to
1.7% at 112, so **the weights stop being the binding constraint between 8 and 16
rows** and the step time keeps growing anyway: 12.6x from one row to 112 while
the memory floor stays at 11.7 ms. Only ~1.2% of the Cube is ever doing work, and
the peak of that curve is 96 rows.

What grows instead is the collective and the stack. At 16 rows the engine
attributes a 64-step batched decode of 37.0834 s as 41.45% in the 64-layer stack
scope `STACK.b`, 12.01% in `tp_all_reduce` over 8256 calls — **129 a step**, the
48/16 projections plus the embedding reduce — and 10.30% in `full_attention`.
Between 4 and 16 rows the stack's share falls 45.05% -> 41.45% while the
collectives' share rises 5.21% -> 12.01%, which is the whole of what width buys:
the per-step kernel work amortizes and the 129 fixed-latency round trips do not.
That is why row-steps/s plateaus a little past 190 and why it turns over at 96,
while the FLOPs available would allow far more.

Prefill sits on the other side of the ridge and is equally far from its ceiling.
A 324.8-token prompt costs 411 ms of prefill — the ladder's TTFT slope, which is
that quantity measured directly — and a token of forward pass is 51.244 GFLOP, so
the prompt is 324.8 x 51.244 = 16.64 TFLOP for the whole model and 16.64/0.411 =
40.5 TFLOP/s across the four cards:

| | ms | tokens/s | TFLOP/s, 4 cards | % of peak |
| --- | ---: | ---: | ---: | ---: |
| prefill, one 325-token prompt | 411 | 791 | 40.5 | 4.6% |
| decode at 96 rows | 471.00 | — | 10.445 | 1.18% |

So prefill is 3.9x more efficient than the widest decode and still leaves 95% of
the Cube idle. That gap is the same serial structure the TTFT section identifies:
the GEMMs never see more than one prompt's rows, and a 325-row M is small enough
that per-op overhead dominates. **Both curves are bounded by the same defect, and
it is a batching defect rather than a bandwidth one.**

## Caveats

- **The HTTP decode path is not token-reproducible on this build.** With
  `--seed 0` the prompt dataset is rebuilt identically every run, and
  `build_payload` sends `temperature: 0.0, top_p: 1.0`, so the completion should
  be a function of the prompt alone. It is not. Seven control runs of the identical
  command produce two different completions — 494 generated characters (77 tokens)
  and 795 (122 tokens) — and so does the replicate arm, so the divergence is not
  attributable to the lever under test. It is visible in the ladder too: the
  512-token output cap is never reached but the generated length varies from 57 to
  130 tokens for prompts drawn from the same fixed-length distribution. The
  engine-side bench agrees from the other direction: at a 512-token prompt both
  arms report `batch_decode_seed=7660`, the same first token, and both fail its
  `seed_mismatches` gate only because a one-row run has no reference to compare
  against (`reference_seed=-1`). This page therefore records latency, which
  separates cleanly, and claims nothing about which tokens were emitted. It is
  consistent with the workspace race that
  [ascend_rope_table_workspace_aliasing.md](ascend_rope_table_workspace_aliasing.md)
  and [ascend_gated_delta_slice.md](ascend_gated_delta_slice.md) chased to zero
  spread on the single-request path, and it says that the batched HTTP path has
  not had the same treatment.
- **Both switches are opt-in.** Every number here is the IPC collective with the
  device-side wait, and the engine says at startup that the device-side wait is
  not the shipped path. The ladder is therefore a ladder of this stack, and the
  concurrency ceiling it finds is a ceiling of this stack's memory budget; the
  shipped path's KV pool is the same size, so the ceiling should transfer, but it
  was not re-measured there.
- **112 is a measurement, not a specification.** The failing rank is the one with
  27 MB less HBM on this host, and 268 MiB is all that separated a working run
  from a hung one. Treat the number as "this host, this checkpoint, ctx 2048" and
  re-derive it from the pool formula anywhere else.
- **Phase shares, not phase times.** `PhaseScope` synchronizes the device on
  entry and exit (`qwen_engine.cpp:2555`), so the percentages in the profile are
  meaningful and the seconds are not. The prefill block counts above are used for
  their count.
- **Prompt lengths are nominal, not tokenizer-exact.** `--random-input-len 512`
  is what the harness asked for; the engine forwarded 320 or 327 tokens, which is
  the count every FLOP figure on this page uses. A run that measures the same
  checkpoint with a tokenizer-exact 512-token prompt should expect the prefill
  and decode rates to move by roughly 325/512 on the token side and not at all on
  the time side.
- **One pair is not a series.** The ladder is one run a width; the repeatability
  it has is the `L16`/`rep16` pair, which matched to 1.8% in throughput and 0.8%
  in TPOT, and the `L1`/`ctl1_r*` match against the A/B's control arm. The
  replicate A/B was run interleaved three times for exactly this reason and is the
  only lever on this page quoted with its spread.
- **The two runs that lost a rank are excluded from every mean on this page.**
  `L120` and `L128` appear only in the concurrency-limit table, as pass/fail
  counts. Their latency figures are the 900 s client timeout, not the engine:
  `L120`'s mean TPOT of 911.42 ms is what averaging 105 requests that produced a
  median of 2 tokens gives, and `L128`'s single success is one request. The
  `128 @ ctx 1024` row of the roofline table is `L128c1024`, a different run that
  completed 128/128.
- **Every server-side figure is a mean over all but the last requests of its
  run.** The archived `.metrics` artifacts predate `--server-drain-seconds`, so
  the scrape that ended each run was taken with requests still in flight: 3 of
  `L4`'s 4, 45 of `L48`'s 48, 61 of `L64`'s 64, 110 of `L112`'s 112, 0 of `L1`'s
  1. The queue, prefill, decode and TTFT means quoted here are therefore over
  the requests that had completed by the last poll. The ones left out are the
  tail of a single wave, and the spread among the requests behind the first is
  8-29 ms, so dropping one or two moves a mean by under a millisecond. The two
  that lost a rank are the exception and are excluded already. Re-running a
  point with the drain makes its scrape exact.
- **No claim about other checkpoints.** 48 of the 64 layers are linear attention,
  and the gated-delta recurrence is what the prefill profile spends 11.7% of its
  time in. A stack without that recurrence would move.

## What this says to do next

In order of what the measurements support, and none of it done here:

1. **Size the KV pool by concurrency actually granted, not by worst-case
   context.** The arena reserves `max_context` for every slot, so at 112 slots it
   holds 3.500 GiB for generations that end at 130 tokens — 2.8 GiB of
   over-reservation in that one row, 9.8 GiB summed over the ladder, and the
   reason 120 slots hangs a rank instead of admitting fewer. The token-based
   budget itself already exists: `ModelOptions::kv_cache_bytes`
   (`model_registry.hpp:47`) is bridged to the engine
   (`engine_registry_builtin.cpp:40`), the paged budget reads it and treats 0 as
   "reserve what the arena would" (`qwen_engine.cpp:1517`), and the Python
   bindings expose it (`bindings.cpp:323`). **`main.cpp` has no
   `--kv-cache-bytes`** — only `--kv-paged`, `--no-kv-paged` and
   `--kv-block-size` — so the server has no way to set it today; that part is a
   flag. The rest is not: a budget below the worst case is a pool handed out on
   demand, and `run_batched_decode` rejects that combination on Ascend
   (`qwen_engine.cpp:4501`) precisely because the batched entry points address a
   slot by a constant element stride rather than through a block table. Exposing
   the knob without teaching those kernels the translation would produce a
   server that starts and answers wrongly, which is worse than the hang above.
2. **Merge the admission wave into one prefill forward.** It is the only finding
   on this page that moves TTFT and prefill TFLOPS together — both are bounded by
   the same one-request-at-a-time loop in `batch_prefill`. The engine's own
   comment cites a saturation sweep that measured 1890 tok/s at a 4096-token
   chunk against 1330 at 512 — a chunk of one or two prompts, not a question
   about several; the cost of *not* merging them is the linear
   TTFT above. The obstacle is real and is named in the code: the linear-attention
   layers carry a per-sequence state, so a merged forward needs a segmented
   recurrence, and the 16 full-attention layers need a block-diagonal mask.
3. **Make the row-step rate rather than the FLOP rate the target, and stop at
   96.** At 96 slots the step is 471.00 ms for 203.82 row-steps/s and 4.04 ms of
   new row each; 129 collectives a step are 12% of it and none of it amortizes
   with width. Going 96 -> 112 costs 45% more TPOT for 9% fewer tokens.
4. **Default the operating point to a slot count below the offered concurrency.**
   `L16x64` gets 62% less TTFT than `L16` at the same throughput and the best
   goodput among the full-batch runs. This is a configuration change and not an
   engine one.
5. **Leave the Cube kernels alone.** Decode is 0.953 FLOP/byte, 202x below the
   ridge. A faster multiply changes a term that is 1.18% of peak and would have
   to be paid for against 129 collectives and a 41.45% stack.
