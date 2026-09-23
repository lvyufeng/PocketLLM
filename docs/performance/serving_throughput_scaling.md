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
emits nothing — a silent hang rather than an error. Throughput stops paying far
earlier: from 32 slots to 112 the measured rate is a dead tie inside 1.8%, at a
nominal peak of 128.59 tok/s, while TPOT grows 3.0x.

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
- commit `dc11490` (master as of 2026-09-21) plus this PR's one-line raise of the
  IPC all-reduce element ceiling from 40960 to 2 621 440 (`512 x 5120`), binary
  `cpp_engine/build-ascend/pocketllm_engine` built from that tree. Every ladder
  row, every
  `prefill` point, the concurrency-limit probes and `pf128` below were measured
  under that ceiling at this commit. The branch has since been rebased onto
  `bf15be7` (master as of 2026-09-23), and **`cpp_engine/` is identical between
  the two**: `git diff dc11490 origin/master -- cpp_engine/` is empty, because
  the thirty-one commits in between are the Python V4.1 and MiMo-V2.6 backends,
  their tests and the documentation around them. `scripts/bench_serving.py` is
  unchanged over the same range too, so the binary and the harness this page
  measured are the ones the rebased branch still builds, and nothing below is
  re-measured for the rebase. Three things on the page were not measured at
  either commit, and one
  of them is qualified where it is used: the `QWEN_ASCEND_REPLICATE_ROWS` A/B's
  concurrency-one pairs, whose planes are one row wide and inside both ceilings,
  so their comparison stands as measured; that A/B's `rep16` point, which is 16
  rows wide and is therefore only compared against the `L16` it ran beside; and
  [serving_latency_optimized.md](serving_latency_optimized.md), whose figures are
  left as they are and named where they are compared.
- **This branch adds one engine change on top of that range, and it moves TTFT.**
  `BatchScheduler::run_prefill_batch()` now issues one `batch_prefill` call a
  request and hands each row's token over before the next prompt starts
  (`batch_scheduler.cpp:372`), instead of calling it once for the whole wave and
  delivering nothing until the last prompt is done. Everything on this page above
  is the pre-change record and stands as the control arm: the change re-writes no
  row's device work, so the prefill decomposition, the FLOP tables and the
  concurrency limit are untouched by it. The ladder rows it moves are re-measured
  against a same-day control in
  [Handing each row its token when it is produced](#handing-each-row-its-token-when-it-is-produced),
  and `L8`, `L16` and `L32` reproduce this page's numbers to 0.4% on that control.
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

The engine sources at `dc11490` are the ones the numbers come from. The two
`report_phase_profile()` calls PR #294 added to the `--batch-decode` bench
(merged as `0d122ea`) were applied for the profiled runs only; with
`QWEN_PHASE_PROFILE` unset both return immediately, so the serving numbers are
unaffected by them.

**This revision is a re-run, not an addition.** The page was first written at
`645e36b` with the IPC all-reduce ceiling at its old default of 40960 elements,
under which every prefill plane and every decode plane wider than eight rows went
to HCCL — a 325-token prompt is 1.66 M elements, forty times that ceiling.
Raising the ceiling to `512 x 5120` puts all of them on the hand-written barrier,
and everything the ceiling reaches is re-measured here: the ladder, the prefill
decomposition, `pf128`, the concurrency limit and the FLOP tables. Across that
range the serving path changed in exactly one other way — the `+20` lines PR #294
added to `main.cpp`, which return immediately when `QWEN_PHASE_PROFILE` is unset;
`qwen_engine.cpp` changed only comments — so where a figure is quoted against its
old value below, the ceiling is the difference. The one place a *conclusion*
moves is the `~1.06x` in `qwen_engine.cpp:5091`, now worth 1.6x at the ladder's
prompt length; see [What one prefill call costs](#what-one-prefill-call-costs).

### Reproduction

The driver is `scripts/run_serving_sweep.sh`, which fixes the arguments every
point of the sweep shares and varies the width:

```bash
source scripts/ascend_env.sh     # without it an ACL binary hangs before aclInit returns
export POCKET_SWEEP_CKPT=/mnt/data1/modelscope/Qwen/Qwen3.8-27B
scripts/run_serving_sweep.sh ladder      # L1 ... L112, then L16x64, L48x192
scripts/run_serving_sweep.sh limit       # L120, L128, L128c1024
scripts/run_serving_sweep.sh ab          # the replicate-rows A/B, interleaved
scripts/run_serving_sweep.sh prefill     # one call's fixed cost vs its per-token cost
# one point on its own, which is `L32` in the ladder table:
#   point <tag> <slots> <concurrency> <prompts> <in> <out> <rate> <ctx> [K=V ...]
scripts/run_serving_sweep.sh point L32 32 32 32 512 512 inf 2048
```

`prefill` is [What one prefill call costs](#what-one-prefill-call-costs):
nine one-call length points and two arms that hold the prompt fixed while the
budget cuts it into 1/2/4/8 calls. The four wave points the decomposition is
checked against are `ladder`'s `L1`, `L4`, `L16` and `L32`.

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
python scripts/analyze_serving_roofline.py --prefill-ms 331 --prefill-tokens 325 sweep-out/L*.json
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
| `L1` | 1 | 1 | 417.4 | 417.4 | 54.40 | 0.00 | 7.000 | 17.26 | 0.142 | 0.142 | 122 | 122/122 |
| `L4` | 4 | 4 | 1177.4 | 1431.3 | 67.45 | 3.41 | 9.404 | 50.86 | 0.413 | 0.413 | 492 | 122/126 |
| `L8` | 8 | 8 | 2422.2 | 2709.5 | 84.35 | 6.06 | 12.658 | 74.50 | 0.609 | 0.076 | 979 | 114/128 |
| `L16` | 16 | 16 | 4961.5 | 5268.2 | 100.39 | 13.86 | 16.547 | 106.37 | 0.903 | 0.056 | 1885 | 77/129 |
| `L32` | 32 | 32 | 10018.0 | 10335.6 | 155.70 | 14.21 | 28.584 | 128.59 | 1.070 | 0.000 | 3847 | 98/130 |
| `L48` | 48 | 48 | 15119.5 | 15443.6 | 212.42 | 29.54 | 37.736 | 126.33 | 1.163 | 0.000 | 5212 | 66/130 |
| `L64` | 64 | 64 | 18000.1 | 20754.1 | 284.58 | 23.69 | 46.843 | 128.50 | 1.244 | 0.000 | 6611 | 66/130 |
| `L96` | 96 | 96 | 27282.0 | 31013.1 | 422.04 | 54.37 | 66.791 | 126.56 | 1.313 | 0.000 | 9253 | 66/130 |
| `L112` | 112 | 112 | 32159.1 | 36257.3 | 472.97 | 48.19 | 75.506 | 127.35 | 1.351 | 0.000 | 10561 | 57/130 |
| `L16x64` | 16 | 64 | 1786.9 | 5254.8 | 129.34 | 18.49 | 14.827 | 105.21 | 1.021 | 0.782 | 6594 | 66/130 |
| `L48x192` | 48 | 192 | 4854.3 | 15391.5 | 347.13 | 77.58 | 33.426 | 116.10 | 1.367 | 0.043 | 16303 | 57/130 |

**These eleven rows are the re-measurement under the raised ceiling, and the two
columns that moved are the ones the change predicts.** TPOT at 1, 4 and 8 rows is
54.40, 67.45 and 84.35 ms against 54.31, 68.28 and 84.67 before — flat to 1.1%,
because 8 rows x 5120 columns is 40960 elements and the old ceiling admitted
exactly that. From 16 rows up every decode plane is wider than the old ceiling and
moved: `L16` 153.65 to 100.39 ms, `L32` 212.55 to 155.70, `L48` 261.55 to 212.42,
`L64` 341.59 to 284.58, `L96` 471.00 to 422.04, `L112` 683.84 to 472.97. Nine rows is the first width the
old ceiling refused, so `L16` is the narrowest point in this ladder above the
boundary and the last two rows below it are the control.

**The lever does not decay with width.** The ladder stops at 112 rows, but the
`128 @ ctx 1024` probe — re-measured on both sides of the change like the rest —
moves its TPOT from 811.14 to 536.98 ms, **1.51x**, against 1.53x at 16 rows and
1.45x at 112. Putting every decode and prefill plane of a serving run on the
hand-written barrier is therefore worth about the same factor at every width this
record reaches, and it is paid for out of nothing but the crossing measured under
[What one prefill call costs](#what-one-prefill-call-costs).

`L1` is also the row the change cannot move at all — a one-row plane is 5120
elements, inside both ceilings — so its 54.40 ms against the 54.31 ms recorded
before is a cross-check between the two runs rather than a result of the change.
It sits inside the control band of the `QWEN_ASCEND_REPLICATE_ROWS` A/B further
down (54.05-54.85 ms), so the ladder's floor reproduces an independently measured
arm.

Goodput is a **rate**, not a count, against `ttft:2000 tpot:200 e2el:30000`. The
200 ms TPOT budget is cleared to 16 slots and not past them: `L16` is 100.39 ms
and `L32` is 155.70, and no run at 32 slots or above clears it however high the
throughput column goes. The best goodput on the page is `L16x64`'s 0.782, and
among the single waves it is `L4`'s 0.413.

**The budget is met at a lower offered concurrency, not by a lower slot count.**
[serving_latency_optimized.md](serving_latency_optimized.md) clears it — 0.564 of
0.589 req/s — with the same opt-in stack and the same nominal 512-token prompts, at 24
prompts, `--request-rate 2` and `--max-concurrency 8`. Its peak concurrency was
11 and it measured 70.29 tok/s at 99.49 ms TPOT, against `L8`'s 74.50 tok/s at
84.35 ms and `L16`'s 106.37 at 100.39. The two configurations no longer meet at
the same width: `L16` now delivers 1.5x that throughput at a TPOT within 1% of
it. Part of that gap is the arrival pattern the older page chose — a slow stream
never holds a full-width batch for a whole decode — and part of it is this
change, because that page's peak concurrency of 11 puts its decode planes above
the ceiling this page raises. **Its figures are pre-change numbers and are left
as they are**; re-running that configuration is a measurement this page does not
carry. The distinction that matters for capacity planning is still **concurrency
offered at once** versus **concurrency sustained**, and this page measures the
first.

## Maximum concurrency: the KV pool, not the flag

`--max-batch-size` is the concurrency ceiling — the engine logs it and the
scheduler admits only while `slot_to_request_.size() < max_batch_size_`
(`batch_scheduler.cpp:264`). Raising it is not free, and the failure when it is
raised too far is the worst kind:

| run | slots | ctx | engine | client | what happened |
| --- | ---: | ---: | --- | --- | --- |
| `L112` | 112 | 2048 | `batch width 112`, four ranks up | 112/112 pass | full run, 127.35 tok/s |
| `L120` | 120 | 2048 | `batch width 120`, **rank 2 dies in `device_malloc`** | 1/120 pass, 119 failures | one request served, 119 timeouts at 900 s; 964 s wall |
| `L128` | 128 | 2048 | `batch width 128`, **rank 2 dies in `device_malloc`** | 1/128 pass, 127 failures | the same, 964 s wall |
| `L128c1024` | 128 | 1024 | `batch width 128`, four ranks up | 128/128 pass | full run, 126.72 tok/s |

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

**The raised ceiling changes what the survivors say about it, and not the wall
clock.** Both probes take 964 s, which is the client's timeout and not the
engine's, and `L128` is unchanged at one success. But the hand-written barrier
has a status latch the host reads once every 32 calls, and that read is the one
place "a peer that never arrives" can become a reported failure
(`ipc_allreduce.cpp:974-995`); prefill planes are on that barrier now, so rank 0's
schedule loop ends in

```
BatchScheduler: prefill failed: Ascend IPC all-reduce: cannot read the device arrival wait's status
BatchScheduler: decode failed: CmdChannel: send_to_workers write_all failed: Broken pipe
```

and ranks 1 and 3 report the same arrival-wait error as their last line, instead
of spinning to the end of the run. The client outcome moves with it at `L120`:
under the old ceiling that width produced 105 *successes* — 211 output tokens
across them, two apiece before each request stopped — and it produces 1 now, 2
tokens. The engine's own counter is the same verdict in both revisions, 0
successes and every request an error, so the old run's 105 passes were the client
counting a stub as a reply.

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
| 1 | 54.40 | — | 18.38 | 18.4 | 17.26 |
| 4 | 67.45 | 4.35 | 59.30 | 14.8 | 50.86 |
| 8 | 84.35 | 4.22 | 94.85 | 11.9 | 74.50 |
| 16 | 100.39 | 2.01 | 159.37 | 10.0 | 106.37 |
| 32 | 155.70 | 3.46 | 205.52 | 6.4 | 128.59 |
| 48 | 212.42 | 3.55 | 225.96 | 4.7 | 126.33 |
| 64 | 284.58 | 4.51 | 224.89 | 3.5 | 128.50 |
| 96 | 422.04 | 4.30 | 227.47 | 2.4 | 126.56 |
| 112 | 472.97 | 3.18 | **236.80** | 2.1 | 127.35 |

**The knee is at 32 slots.** Rows 17 to 32 buy 46.2 row-steps/s and rows 33 to 48
buy another 20.4; past 48 the rate is flat inside 5% — 225.96, 224.89, 227.47,
236.80 — while TPOT grows from 212.42 to 472.97 ms. So a 112-slot server pays
2.2x the TPOT of a 48-slot one for the same token rate, and every row past 48 is
width that costs time and buys none.

The marginal ms/row column is the slope of TPOT between consecutive widths, and
it is now flat: 4.35, 4.22, 2.01, 3.46, 3.55, 4.51, 4.30, 3.18 ms. The two spikes
the table carried before this re-run — 8.62 ms at 16 rows and 13.30 at 112 — do
not reproduce. The first was the ceiling's edge: 8 rows is exactly 40960
elements, so moving from 8 to 16 doubled the plane *and* switched it from the
hand-written barrier to HCCL, and both halves of that are now gone. The second
does not survive a re-run in which `L112`'s own TPOT spread is 48.19 ms. The
nominal peak of the measured aggregate is `L32` at 128.59 tok/s, with `L48`,
`L64`, `L96` and `L112` at 126.33, 128.50, 126.56 and 127.35 — a dead tie inside
1.8% across a batch 3.5x wider, which is the honest statement of where this
configuration tops out.

### Saturation does not help

`L48x192` offers 192 prompts to 48 slots, so the batch is refilled as it drains
rather than being allowed to empty. It is **worse than the single wave at the
same width**: 116.10 tok/s against 126.33, TPOT 347.13 ms against 212.42. The
scheduler runs `run_prefill_batch()` and `run_decode_batch()` in the same pass
(`batch_scheduler.cpp:226`), and a continuous supply of new prompts means a
prefill is always waiting to run in front of the next decode, so the step rate
drops from 4.71 to 2.88 a second.

`L16x64` is the opposite case and is a genuine gain on TTFT: 64 prompts against
16 slots gives mean TTFT 1786.9 ms against `L16`'s 4961.5, because slots free one
at a time and each prefill group is small, at the same throughput (105.21 against
106.37 tok/s) and the best goodput on the page. The mechanism is the one in the
next section.

## TTFT is the whole wave's prefill

TTFT in the ladder is not a queueing term. The server's own histograms separate
the two: in `L32` the queue is 346.3 ms a request while the prefill is 9566.0
ms, so **95.5% of TTFT is prefill**, and the prefill time grows with the width of
the wave rather than with the size of the prompt.

Per request of a ~325-token prompt, mean TTFT is *exactly* linear in the width up
to 48 slots, and the distribution behind the mean is bimodal rather than a ramp:

| run | slots | TTFT mean | first request | the rest, p50 | the rest, max | of the rest, within 100 ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `L4` | 4 | 1177.4 ms | 417.3 ms | 1430.5 ms | 1431.3 ms | 3 of 3 |
| `L8` | 8 | 2422.2 ms | 424.0 ms | 2707.3 ms | 2709.5 ms | 7 of 7 |
| `L16` | 16 | 4961.5 ms | 410.9 ms | 5264.8 ms | 5268.2 ms | 15 of 15 |
| `L32` | 32 | 10018.0 ms | 415.4 ms | 10327.6 ms | 10336.1 ms | 31 of 31 |
| `L48` | 48 | 15119.5 ms | 417.9 ms | 15432.0 ms | 15443.7 ms | 47 of 47 |
| `L64` | 64 | 18000.1 ms | 418.7 ms | 17922.3 ms | 20754.4 ms | 55 of 63 |
| `L96` | 96 | 27282.0 ms | 413.2 ms | 27116.9 ms | 31013.7 ms | 84 of 95 |
| `L112` | 112 | 32159.1 ms | 418.6 ms | 31986.2 ms | 36257.7 ms | 99 of 111 |

Exactly one request latches at 411-424 ms in every row — `L1`'s single-request
TTFT, reproduced to 3% — and up to 48 slots every one of the others latches
within 23 ms of the pack. So the engine serves the first request through its own
prefill and one decode step, and then runs the remaining N-1 prefills inside a
**single blocking `batch_prefill` call** that only returns when the last of them
is done. Nobody in that group decodes before it returns, so nobody in it sees a
token before the slowest member.

TTFT is therefore `417 ms + c x (N - 1)`, where `c` is the per-prompt prefill
cost, and not the `slots/2 x prefill` of an earlier revision of this page — the
per-request distributions falsify the half-wave form. Solving each row for `c`
gives 337.8 ms at 4 slots, 326.2 at 8, 323.6 at 16, 319.8 at 32 and 319.5 at 48:
**a ~325-token prompt costs 320 ms of prefill whether it shares a wave with three
others or 47.** The server's own prefill histogram is the same measurement taken
from the engine's side — 4523.6, 9566.0 and 14663.5 ms a request at 16, 32 and 48
slots, whose successive differences are 315.2 and 318.6 ms — so the client's
distribution and the engine's histogram agree on `c` to 1% from two sources that
do not read each other.

The widest run on this page extends the same line past where the decode ladder
stops. `L128c1024` admits 128 requests into one schedule and its median TTFT is
36.2 s, which is `417 + c x 127` for **282 ms a request** — 12% below the
ladder's 320 ms, on the same nominal prompts. The only argument the two runs do
not share is `--max-context`, which sizes the KV arena and not the prefill
compute, and if that is not the explanation then this is where the linear form
starts to soften; what it does not do is break. **The wave's prefill is not
spread across the batch at any width measured here**, so a deployment that wants
a wide batch pays for it once per request before any of them decode.

From 64 slots the form breaks, and the table's last two columns show how: the
pack is still tight — 55 of 63 requests within 100 ms of the median at `L64` — but
a tail group of 8 to 12 requests latches 2.8-4.3 s later. That is the queue term
waking up: 2424.6 ms a request at 64, 3264.9 at 96, 3576.1 at 112, against
346-359 ms up to 48. Past that point admission, not prefill, is what the last
requests wait for.

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
prompt's rows, and the 325-token prefill runs at 325/0.331 = 981 tokens/s
against a weight-read floor of 11.7 ms a pass.

### What one prefill call costs

The slope `c` above is one number, but it is the sum of two: a cost that does not
depend on the prompt at all, and a cost that does. Holding the prompt fixed and
varying the **number of calls** the server makes separates them, because a
reduced `--prefill-token-budget` turns one prompt into several passes through the
same 64-layer stack:

| prompt | budget | calls | TTFT p50 | ms a call |
| ---: | ---: | ---: | ---: | ---: |
| 3703 tok | 4096 | 1 | 2420.8 ms | — |
| 3703 tok | 2048 | 2 | 2614.1 ms | 195 |
| 3703 tok | 1024 | 4 | 2913.2 ms | 164 |
| 3703 tok | 512 | 8 | 3424.1 ms | 143 |
| 1242 tok | 2048 | 1 | 888.5 ms | — |
| 1242 tok | 1024 | 2 | 1011.8 ms | 128 |
| 1242 tok | 512 | 3 | 1150.8 ms | 130 |
| 1242 tok | 256 | 5 | 1295.2 ms | 110 |

Total tokens are constant within each arm, so **a call is worth 110-195 ms
whatever is in it**. A joint least squares over all seventeen points — the eight
above and the nine below — reads

```text
TTFT = 131.4 ms x n_calls + 637.4 us x prompt_tokens - 7.2
```

with a maximum residual of 95.5 ms and an rms of 37.5 ms.

A one-call length sweep agrees from the other side. Nine single-slot runs at a
budget above the prompt, so every point is exactly one call, over 67, 115, 219,
320, 421, 629, 1242, 2474 and 3703 real tokens (the engine's own
`prompt_tokens=`, not the harness's nominal length):

| nominal | real tokens | TTFT mean | TTFT p50 | server prefill |
| ---: | ---: | ---: | ---: | ---: |
| 90 | 67 | 158.1 | 144.4 | 143.1 ms |
| 170 | 115 | 191.3 | 178.2 | 175.7 ms |
| 340 | 219 | 269.7 | 257.3 | 252.9 ms |
| 512 | 320 | 356.5 | 334.2 | 337.3 ms |
| 680 | 421 | 414.0 | 400.8 | 395.6 ms |
| 1024 | 629 | 583.9 | 566.8 | 563.8 ms |
| 2048 | 1242 | 905.7 | 887.8 | 880.0 ms |
| 4096 | 2474 | 1731.5 | 1713.8 | 1694.9 ms |
| 6144 | 3703 | 2438.7 | 2420.3 | 2391.0 ms |

with every residual inside ±47 ms of

```text
TTFT = 142.6 ms + 627.4 us x prompt_tokens
```

fitted on those nine alone. The two fits put the fixed term at 131.4 and 142.6 ms
from disjoint data: the calls arms never send a 325-token prompt and the length
sweep never makes more than one call.

One measurement already on this page lands on the same number without being
asked to. The ladder's own marginal prefill `c` is 319.5-337.8 ms a request,
against the model's 329.8 ms for the 322.6-token mean prompt `L16` actually sent.

The `pf128` arm below is the one that does **not** land on it, and that is worth
stating rather than burying. It gives each of sixteen requests two extra calls,
and if the fixed term were per call the way the fit has it, the mean TTFT would
move by `k x (32 x 15/16 + 2 x 1/16) = 30.1 k`, or 4.0 s at `k` = 131.4 ms. It
moves 1.56 s, which reads 52 ms a call — a factor of 2.5 below what the two
direct arms measure. The wave's own shape says why: `L16`'s bimodality is gone,
the fifteen pack requests all latch between 6625 and 6632 ms and the first at
4905.1 ms where `L16`'s first was 410.9 ms. **A budget that cuts a prompt into
three passes does not add three independent calls to each request; it changes the
order the scheduler runs them in.** The direct arms stay the measurement of the
per-call term — they vary the call count at one slot and one prompt, so a call is
the only one in flight — and this arm bounds what budgeting costs a whole wave
rather than being a second estimate of the same constant.

**So a 325-token prefill costs 331 ms and 131 of it — 40% — is paid before the
first token-dependent FLOP.** Every estimate of that fixed term here — 131.4 and
142.6 from the two fits, and the 110-195 ms the individual arms span — comes from
one of three different manipulations of the same server.

**It is the collectives.** A forward pass issues 129 of them whatever the prompt's
width, and the barrier the engine ships for them is not the one it could use.
Ceiling raised past the prompt against the shipped default, one call, one slot,
TTFT p50 over six requests:

| rows | plane | default, on HCCL | hand-written | change |
| ---: | ---: | ---: | ---: | ---: |
| 67 | 686 KB | 165.0 ms | 145.2 ms | -12.0% |
| 219 | 2.24 MB | 369.6 ms | 257.0 ms | -30.5% |
| 421 | 4.31 MB | 465.0 ms | 401.0 ms | -13.8% |
| 629 | 6.44 MB | 564.9 ms | 569.4 ms | +0.8% |
| 1242 | 12.7 MB | 889.6 ms | 1021.1 ms | +14.8% |

At 219 rows that is 112.6 ms over the 129 collectives a pass issues, 0.87 ms a
call, for a barrier that computes the same numbers. **`tp_all_reduce` is serialized
work on the critical path rather than latency the ranks absorb**, and a cheaper
call is worth the whole difference.

The two barriers cross between 421 and 629 rows, which is the part of this an
element ceiling gets wrong. [ascend_single_request_tps.md](ascend_single_request_tps.md)
§2.2 gives HCCL's side: its 0.481 ms a call is host latency and not wire time, so it
is flat from 10 KB to 640 KB. The hand-written barrier is cheaper there and costs
more per element, so it wins where HCCL's fixed price dominates and loses once the
plane is large enough for that price to stop dominating. The gap below the crossing
is wider than the barriers alone account for, because the hand-written path also
skips `all_reduce_half`'s bracket: it runs on the caller's stream, so there is no
default-stream drain around it, which §5.5.1 of that page prices at 0.100 ms a call.
**The ceiling is therefore pinned at 512 rows**, between the largest measured win and
the smallest measured loss, and it stays a count of elements because the collective
layer does not know the hidden size.

The crossing is also the only place in this change where a barrier could have been
wrong rather than slow, and it is not: over a 432-token prompt — 2.21 M elements, so
on the hand-written path under the new ceiling and on HCCL under the old one — 24
greedy steps emit identical ids. That run is worth 391.9 ms of prefill against
471.6 ms, **-16.9%**, against the same prompt the length sweep could only bound.

### What merging would and would not buy

A merged forward pays the fixed term once and the per-token term once per token;
the serial loop pays both once per request. `L16`'s sixteen prompts are 5162 real
tokens, ten at 320 and six at 327:

| | wave prefill | the 16th request |
| --- | ---: | ---: |
| `L16` today, 16 serial calls | 16 x 131.4 + 5162 x 0.6374 - 7.2 = 5385 ms | 5264.8 ms measured |
| merged, one 5162-row forward | 131.4 + 5162 x 0.6374 - 7.2 = 3414 ms | one decode step later |
| merged, at the one-call slope 0.6274 | 142.6 + 5162 x 0.6274 = 3381 ms | one decode step later |

The measured column is the check, and the delivery fix below does not touch it:
the 16th request is handed its token at 5235.2 ms after it against 5249.0 before,
because what that fix removes is the wait the *other* fifteen were doing. The
serial model's own last request is
15 x 329.8 + 410.9 = 5358 ms, against 5264.8 ms measured, so the per-request
prefill the model uses reproduces the wave to 2% — the first request's TTFT is
410.9 ms against its 329.8 ms of modelled prefill plus one decode step. The
merged rows are the same 5162 tokens with the call term paid once instead of
sixteen times, and either would put each request's first token one decode step
past it: `L16`'s own TPOT at 16 rows, 100.39 ms.

**That is 1.6x, not the ~1.06x the engine's comment quotes.** The two slopes
are the joint call-count fit's and the one-call length fit's; both are measured
to 3703 rows and the merged call is 5162, so the table's exposure is that
extrapolation and not the decomposition.

The comment is not wrong, it answers a different question. The saturation sweep
that produced it merges prompts into a chunk that is *already* 2048 or 4096
tokens, where the fixed term is 131.4/(131.4 + 0.6374 x 2048 - 7.2) = 9.2% of a
call and merging sixteen of them is worth 1.09x. The ladder sends 322-token
prompts, where the same term is 40% of a call. The smaller the prompt, the larger
the share of it that is the fixed cost, and the ladder is on the small end.

In FLOP terms the fixed term is what separates the delivered rate from the
marginal one. 51.244 GFLOP a token over a 637.4 us slope is **80.4 TFLOP/s of
884.7, 9.1% of the Cube**, against 50.3 TFLOP/s, 5.7%, delivered on a 325-token
prompt. Merging recovers that 1.6x — the same 1.6x as the TTFT figure, because it
is the same term.

### The levers that were tried against it

**A smaller prefill token budget makes TTFT worse, not better.** The budget
documents itself as the thing that stops "a long prompt holding the device"
(`qwen_engine.cpp:5097`), but it also caps how far each request advances per
call, so a 128-token budget turns every ~325-token prompt into three passes:

| run | budget | TTFT mean | TTFT P99 | TPOT | tok/s | goodput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L16` | 4096 | 4961.5 | 5268.2 | 100.39 | 106.37 | 0.056 |
| `pf128` | 128 | 6521.1 | 6631.8 | 96.85 | 97.60 | 0.000 |

TTFT is 31% higher and throughput 8% lower, while TPOT is *better* by 3.5 ms —
inside its own 13.86 ms spread, so the cost is entirely in the extra passes
through the 64-layer stack and not in the step. For prompt lengths near the
budget the budget is a latency tax. It would earn its keep on prompts far longer
than 4096 tokens, which this record does not measure.

**Padding the batch with the client does work, and is the usable lever.** `L16x64`
is `L16` with 64 prompts offered to 16 slots, so the client holds a wave down to
16 and refills as slots free: mean TTFT 1786.9 ms against 4961.5 for a 64%
reduction, at the same throughput (105.21 against 106.37 tok/s) and the best
goodput among the full-batch runs. It costs nothing but a slot count below the
offered concurrency, which is the opposite of what a throughput-first
configuration does.

### Handing each row its token when it is produced

The wave is serial, but the *handover* was serialized with it, and nothing about
the handover is expensive. `batch_prefill` is a loop over requests
(`qwen_engine.cpp:5163`), and each iteration ends in `prefill_bounded`, which
ranks the prompt's last position and returns `step.result.top_token`. That token
exists from that moment. What did not exist was a delivery: `run_prefill_batch`
gathered the whole wave, called the engine **once**, and only then updated every
request's state and called `deliver_tokens` — so the token request 0 produced at
410 ms sat inside a `BatchPrefillResult` until request 15 had finished too.

Making that loop the scheduler's is the whole change: one `batch_prefill` call a
request, with the state update and the token callback in between
(`batch_scheduler.cpp:372`). The engine still runs its prompts one at a time, in
the same order, at the same budget, so no row's device work moves; what changes is
when its caller hears about it. Three interleaved pairs, control and treatment
alternating at each width:

| run | TTFT mean | change | TTFT p50 | change | TTFT p99 | change | last row | tok/s | e2el |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L8` control | 2421.1 ms | — | 2707.8 ms | — | 2710.0 ms | — | 2710.1 ms | 74.94 | 12679.5 ms |
| `L8` treatment | 1575.6 ms | **-34.9%** | 1581.3 ms | -41.6% | 2677.3 ms | -1.2% | 2699.6 ms | 74.45 | 12826.1 ms |
| `L16` control | 4943.7 ms | — | 5245.3 ms | — | 5248.9 ms | — | 5249.0 ms | 110.18 | 16090.5 ms |
| `L16` treatment | 2847.4 ms | **-42.4%** | 2852.6 ms | -45.6% | 5187.5 ms | -1.2% | 5235.2 ms | 110.72 | 16244.0 ms |
| `L32` control | 10004.3 ms | — | 10313.2 ms | — | 10321.0 ms | — | 10321.2 ms | 128.75 | 28799.3 ms |
| `L32` treatment | 5386.9 ms | **-46.2%** | 5396.8 ms | -47.7% | 10210.1 ms | -1.1% | 10307.8 ms | 125.67 | 28731.6 ms |

The bimodal distribution the page opened with becomes a ramp, which is what the
fix is:

```text
L8 control    [412, 2707, 2707, 2707, 2708, 2709, 2709, 2710]
L8 treatment  [413,  784, 1103, 1422, 1740, 2062, 2381, 2700]
L32 treatment [412, 772, 1090, 1407, ... , 9675, 9992, 10308]
```

The control's fifteen rows all latch on the moment the wave's prefill ends. The
treatment's each latch one per-prompt prefill after the one before — the `L32`
successive differences are 360 then 317-321 ms, against the 319.8 ms per-prompt
prefill this page solved for at 32 slots. The first row is unchanged (413 against
412) and so is the last (2700 against 2710): **the wave still takes the same time,
and only the order of the departures changed.**

Nothing is slower for it. The second token of every row lands at the same absolute
time in both arms, and so does every token after it:

| run | token 1, p50 | token 2, p50 | first gap, p50 | every later gap, p50 |
| --- | ---: | ---: | ---: | ---: |
| `L8` control | 2707.8 ms | 2793.3 ms | 84.8 ms | 81.9 ms |
| `L8` treatment | 1581.3 ms | 2784.4 ms | 883.4 ms | 82.9 ms |
| `L16` control | 5245.3 ms | 5345.2 ms | 99.1 ms | 90.9 ms |
| `L16` treatment | 2852.6 ms | 5339.3 ms | 2169.1 ms | 91.1 ms |
| `L32` control | 10313.2 ms | 10484.0 ms | 170.1 ms | 152.2 ms |
| `L32` treatment | 5396.8 ms | 10480.5 ms | 4770.5 ms | 152.2 ms |

The engine's own histogram agrees from the other side: the reported prefill
interval a request falls from 2006.5 to 1159.4 ms at `L8`, 4502.5 to 2408.4 at
`L16` and 9554.6 to 4939.5 at `L32`, while the queue term behind it does not move
(346.8 against 348.2 ms at `L8`, 346.2 against 345.3 at `L32`).

**TPOT rises by 9% to 27% and that is the metric, not the step.** The control
charges a row its whole wait for the wave to the *second* token: token 1 and
token 2 are 84.8 ms apart because token 1 was held back to the moment decode
started. Delivered honestly, the same row's first two tokens are 883.4 ms apart,
and `(e2el - ttft)/(n-1)` reads that gap out. The gap *after* the first is
identical in both arms — 81.9 against 82.9 ms, 90.9 against 91.1, 152.2 against
152.2 — so the decode step is untouched and nothing had its steadier cadence taken
away. The whole of the TPOT difference is a first-token wait that the control was
hiding inside a first-token delay.

**This does not do what a merge would.** The wave's own prefill is still N serial
calls; the treatment removes the handover, not the loop, and step 2 below is
still worth its 1.6x on the wave. What it does is stop the wave's last prompt from
billing every other prompt for its own duration. `L16x64`, which holds a wave
down to the slot count, gets its 64% a different way and the two compose: it is a
client-side setting, this is the engine delivering what it already computed.

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
the M tile is already full of real rows, so a broadcast row adds no work. The
`rep16` run is from the earlier revision — it is at `645e36b`, above the old
ceiling and below the new one — and it reads 152.36 ms and 72.86 tok/s against
that revision's `L16` of 153.65 ms and 74.22 tok/s, a difference inside the
ladder's own spread. Compared against the re-measured `L16` it would look like a
52% loss, and that difference is the ceiling, not the lever: `rep16`'s decode
planes are 16 rows wide, so under the old ceiling they went to HCCL while the
current `L16` keeps them on the hand-written barrier. **The arm is not re-run
here**, so its only valid comparison is against the `L16` it was measured beside.

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
| 1 | 54.40 | 18.38 | 0.942 | 0.11% | 247.2 | 21.5% |
| 4 | 67.45 | 59.30 | 3.039 | 0.34% | 199.4 | 17.4% |
| 8 | 84.35 | 94.85 | 4.860 | 0.55% | 159.5 | 13.9% |
| 16 | 100.39 | 159.37 | 8.167 | 0.92% | 134.0 | 11.7% |
| 32 | 155.70 | 205.52 | 10.532 | 1.19% | 86.4 | 7.5% |
| 48 | 212.42 | 225.96 | 11.579 | 1.31% | 63.3 | 5.5% |
| 64 | 284.58 | 224.89 | 11.524 | 1.30% | 47.3 | 4.1% |
| 96 | 422.04 | 227.47 | 11.656 | 1.32% | 31.9 | 2.8% |
| 112 | 472.97 | **236.80** | **12.135** | **1.37%** | 28.4 | 2.5% |
| 128 @ ctx 1024 | 536.98 | 238.37 | 12.215 | 1.38% | 25.0 | 2.2% |

The GB/s column is the step's own traffic if every resident weight byte were read
once, over the measured step time. It falls from 21.5% of the probe at one row to
2.5% at 112, and it is under a quarter of the card's measured read bandwidth at
every width on the ladder: at 112 rows the step takes 472.97 ms against the
11.7 ms a full weight read costs. **Memory bandwidth is not what bounds this
ladder**, and the step time nonetheless grows 8.7x from one row to 112 while that
floor stays flat. Only 1.4% of the Cube is ever doing work, and the widest rung
of the ladder is where the ladder's own curve peaks — it has not turned over at
112. The `128 @ ctx 1024` row sits just above the 112-row one, 12.215 TFLOP/s and
238.37 row-steps/s, but it is a run at half the context and so a different
operating point rather than the next rung.

What grows instead is the collective and the stack. At 16 rows the engine
attributes a 64-step batched decode of 37.0834 s as 41.45% in the 64-layer stack
scope `STACK.b`, 12.01% in `tp_all_reduce` over 8256 calls — **129 a step**, the
48/16 projections plus the embedding reduce — and 10.30% in `full_attention`.
Between 4 and 16 rows the stack's share falls 45.05% -> 41.45% while the
collectives' share rises 5.21% -> 12.01%, which is the whole of what width buys:
the per-step kernel work amortizes and the 129 fixed-latency round trips do not.
That is why the row-step rate is flat from 48 slots on while TPOT is not —
225.96, 224.89, 227.47 and 236.80 row-steps/s at 48, 64, 96 and 112 against
212.42 to 472.97 ms of TPOT — so past the knee every extra slot is bought from
the step time and paid for out of nothing. The FLOPs available would allow far
more, and the distance between them is the 129 round trips.

Those shares are the **shipped** barrier's, and the profile is the only place on
this page that measures it. The profiled runs are `--batch-decode` invocations of
`run_qwen_ascend_tp4.sh`, which exports neither `POCKET_ASCEND_IPC_ALLREDUCE` nor
its DEVWAIT, so every collective in them is HCCL and the element ceiling this
revision raises is not on their path at all — with the switch unset the ceiling
is never consulted. The ladder's numbers are the opt-in stack's; the profile's
12.01% is what the shipped configuration pays for the same 129 calls.

Prefill sits on the other side of the ridge and is equally far from its ceiling.
A 325-token prompt costs 331 ms of a single call — the fits above are that
quantity measured directly — and a token of forward pass is 51.244 GFLOP, so the
prompt is 325 x 51.244 = 16.65 TFLOP for the whole model and 16.65/0.331 =
50.3 TFLOP/s across the four cards:

| | ms | tokens/s | TFLOP/s, 4 cards | % of peak |
| --- | ---: | ---: | ---: | ---: |
| prefill, one 325-token prompt | 331 | 982 | 50.3 | 5.7% |
| decode at 112 rows, the widest | 472.97 | — | 12.135 | 1.37% |

So prefill is 4.1x more efficient than the widest decode and still leaves 94% of
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
  in TPOT — both of them at `645e36b`, and `rep16` has no counterpart under the
  raised ceiling — and the `L1`/`ctl1_r*` match against the A/B's control arm.
  The replicate A/B was run interleaved three times for exactly this reason and is
  the only lever on this page quoted with its spread.
- **The two runs that lost a rank are excluded from every mean on this page.**
  `L120` and `L128` appear only in the concurrency-limit table, as pass/fail
  counts. Their latency figures are what one surviving request reads, not what
  the run is: the re-measured `L120` completed a single request at 421.85 ms TTFT
  and 55.59 ms TPOT for 2 output tokens, and the engine's own counter for that run
  is 120 errors and 0 successes. The `128 @ ctx 1024` row of the roofline table
  is `L128c1024`, a different run that completed 128/128.
- **Every server-side figure on this revision is exact.** The re-measured runs
  pass `--server-drain-seconds 1.0`, so the scrape that ends a run is taken after
  the last request has returned: every `.metrics` artifact behind the ladder, the
  prefill sweep and the limit probes has a `pocket_request_prefill_time_seconds_count`
  equal to the number of requests the bench completed — 1, 4, 8, 16, 32, 48, 64,
  96, 112, 16x64, 48x192 on the ladder and 4 of 4 at every prefill point. The
  `QWEN_ASCEND_REPLICATE_ROWS` A/B arms predate the flag and are the one place it
  still bites: `rep16` recorded 15 of its 16 requests, `rep1` and the `ctl1_*`
  controls 0 of 1. An earlier revision of this page read the whole ladder from
  artifacts of that kind, with `L4` at 3 of its 4 and `L1` at 0 of its 1. The
  figures the A/B is quoted for are the bench's own record rather than the
  scrape, so nothing in that table moves.
- **No claim about other checkpoints.** 48 of the 64 layers are linear attention,
  and the gated-delta recurrence is what the prefill profile spends 11.7% of its
  time in. A stack without that recurrence would move.

## What this says to do next

In order of what the measurements support. Item 2 is partly done on this branch —
its handover half — and is marked there; the rest is not.

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
2. **Merge the admission wave into one prefill forward, and expect 1.6x rather
   than the ~1.06x the engine's comment quotes.** It is the only finding on this
   page that moves TTFT and prefill TFLOPS *together* — both are bounded by the
   same one-request-at-a-time loop in `batch_prefill`, and the term that merging
   removes is 131 of the 331 ms a 325-token prompt costs. At `L16` that is
   5385 ms of wave prefill to 3414 ms, and 49 to ~78 TFLOP/s on four cards. The
   `~1.06x` in that comment is the same arithmetic at 2048 tokens, where the
   fixed term is 9% of a call instead of 40%; it is the ladder's short prompts,
   not the sweep's, that the merge is worth doing for. **The half of this that is
   a handover rather than a forward is done** — the change in
   [Handing each row its token when it is produced](#handing-each-row-its-token-when-it-is-produced)
   is worth -42% mean TTFT at `L16` and touches no kernel, but it leaves the
   wave's own prefill exactly as long as it was, so the 1.6x above is still
   entirely unclaimed. The obstacle to claiming it is real and
   is named in the code: the linear-attention layers carry a per-sequence state,
   so a merged forward needs a segmented recurrence, and the 16 full-attention
   layers need a block-diagonal mask — without which a 5162-row forward would
   spend 16x on the dense attention matrix what sixteen 322-row forwards spend in
   total.
3. **Make the row-step rate rather than the FLOP rate the target, and stop at
   48.** At 48 slots the step is 212.42 ms for 225.96 row-steps/s and 3.55 ms of
   new row each; 129 collectives a step are 12% of the shipped step and none of
   it amortizes with width. Going 48 -> 112 more than doubles TPOT, 212.42 to
   472.97 ms, for 4.8% more row-steps/s and 0.8% more tokens. If the step rate is
   the target rather than the latency, the lever is those 129 round trips, and
   the ceiling this revision raises is a first step at it: the same TPOT points on
   the hand-written barrier instead of HCCL are worth 1.53x at 16 rows and 1.45x
   at 112.
4. **Default the operating point to a slot count below the offered concurrency.**
   `L16x64` gets 64% less TTFT than `L16` at the same throughput and the best
   goodput among the full-batch runs. This is a configuration change and not an
   engine one.
5. **Leave the Cube kernels alone.** Decode is 0.953 FLOP/byte, 202x below the
   ridge. A faster multiply changes a term that is 1.37% of peak and would have
   to be paid for against 129 collectives and a 41.45% stack.
