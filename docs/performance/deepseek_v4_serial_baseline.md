# DeepSeek-V4 PersistentEngine serial baseline

Issue #163 asks the PersistentEngine / DeepSeek-V4 path to run multiple requests
concurrently without breaking TP ordering, token correctness, or single-request
latency. Its acceptance criteria name throughput and single-request latency, and
neither can be judged — nor can the effect of any later batching change be
measured — without a recorded baseline for this engine on this checkpoint. This
page is that baseline.

It is also the control for the batching question, and that question has two
answers on this page because the code changed between them. The first batch arm
was run before a real batched forward existed, and it came back
indistinguishable from serial because the scheduler clamped every batch to width
1 — that is [why the first batch arm was throughput-neutral](#why-the-first-batch-arm-was-throughput-neutral). The second
was run after the batched forward landed (#239, #241) and after `caps()` was
changed to report it, and it does move: 1.62x wall time at eight concurrent
requests, with the single-request phase split unchanged. That is
[batch mode after the batched forward](#batch-mode-after-the-batched-forward).

Read [Benchmarking and reporting rules](../guides/benchmarking.md) before
comparing these numbers with anything else. Prefill and decode are reported
separately and are not interchangeable; every figure below states its prompt
length, its generation length, and the environment it was taken in.

## Scope

The measured path is:

- `DeepSeek-V4-Flash-0731` safetensors, FP8 (`e4m3`) block-quantized weights,
  `quant_method fp8`, `weight_block_size [128, 128]`, 43 layers
- the native `pocketllm_engine --serve` process (`server_architecture=deepseek_v4`),
  not the Python server
- CUDA, TP4, four RTX 2080 Ti devices, full model depth
- OpenAI `/v1/chat/completions`, greedy, non-streaming for the throughput ladder
  and streaming for the phase split, `--max-context 8192`

The following are outside this result:

- **Multi-slot execution.** The engine declared up to 8 slots but ran at width 1;
  see [the clamps](#why-the-first-batch-arm-was-throughput-neutral). Nothing in
  the baseline tables above demonstrates two requests executing in one forward
  pass. That changed afterwards, and
  [the second batch arm](#batch-mode-after-the-batched-forward) does run at width
  8 — a later measurement on later code, not part of this baseline's result.
- **Token parity against a reference implementation.** No PyTorch or GGUF
  reference run was made for this baseline, so no parity claim is available here.
  The harnesses check that each response is well-formed, reaches its token
  budget, and reports `finish_reason: length`; they do not compare token ids
  across ranks or against another engine.
- **The GGUF Q2 path**, which has its own engine path and its own baseline.
- **Ascend**, which cannot run this checkpoint.
- **Long context.** The longest prompt measured is 1446 tokens against the
  checkpoint's 1M-token position range; 8192 is only the server's configured
  context limit. No claim about 32K or 64K prefill is made here.

## Configuration

| Item | Value |
| --- | --- |
| Checkpoint | `/mnt/data3/DeepSeek-V4-Flash-0731` (156 GB, `deepseek_v4`, 43 layers, 256 routed experts, vocab 129280, `torch_dtype bfloat16`) |
| Engine | `cpp_engine/build/pocketllm_engine --serve`, built `Release`, `CMAKE_CUDA_ARCHITECTURES=75` |
| PocketLLM commit | `15366816ae95983163e824ccb0d1c2eeffab7560` |
| TP / EP world | TP4, one rank per device, devices `0,1,2,3` |
| GPU | 4 x RTX 2080 Ti, 22528 MiB each, driver 580.173.02 |
| GPU topology | `GPU0-GPU1` PHB, `GPU2-GPU3` NV2 (NVLink), every cross pair SYS. TP4 therefore spans both NUMA nodes and moves three of its six rank pairs over PCIe |
| CPU / host | 2 x Xeon E5-2696 v4, 88 hardware threads, 2 NUMA nodes, 1007 GiB RAM |
| Toolchain | gcc 11.4.0, cmake 3.26.3, `nvcc` CUDA 13.0 |
| Server args | `--max-context 8192 --max-batch-size 1 --prefill-token-budget 0` (serial) or `--max-batch-size 8 --prefill-token-budget 4096` (batch arm) |
| Sampling | greedy, `--max-tokens 32`, prompts of 128 and 1600 words (120 and 1446 tokens) |

No functional C++ change has landed since `326ff26` (2026-09-14); the commits
between it and `1536681` touched documentation, one comment, and site
configuration. Both build trees present on this host (`build/` and
`build-python/`) are therefore the same engine. The runs below used
`cpp_engine/build/pocketllm_engine`.

### Environment

The engine reads four switches that matter for this checkpoint. `default` means
the variable is unset, and the parenthesised value is the built-in default:

| Variable | Default | Tuned | What it does |
| --- | --- | --- | --- |
| `POCKETLLM_CPP_DECODE_SPARSE_ARENA` | `0` | `12` | Use a sparse per-layer arena for decode, and drop the dense prefill arenas so all 43 layer arenas stay resident |
| `DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS` | `0` | `43` | LRU cap on the resident per-layer active-expert arena cache |
| `POCKETLLM_CPP_PREFILL_MOE_PREFETCH` | `0` | `1` | Prefetch the next layer's experts during prefill; also enables the copy stream |
| `POCKETLLM_CPP_PREFILL_BATCHED_ATTN` | `1` | `1` | Row-batched prefill attention (already on by default) |

The names in the 2026-08-03 record — `DSV4_CPP_DECODE_SPARSE_ARENA`,
`DSV4_CPP_PREFILL_MOE_PREFETCH`, `DSV4_CPP_PREFILL_BATCHED_ATTN` — no longer
exist in the source. They were renamed to the `POCKETLLM_CPP_` prefix, and
`DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS` is the one that kept the old prefix.
A command copied from that record sets nothing.

## Reproduction

Two harnesses are used, because they answer different questions and the native
server cannot answer both from one.

`scripts/bench_cpp_openai_concurrency.py` is the acceptance harness. It starts
all four ranks, waits for `/health`, runs a warmup ladder, then measures a
single request, a 2/4/8 concurrency ladder, a long/short interleave, streaming,
and disconnect recovery:

```bash
python scripts/bench_cpp_openai_concurrency.py \
  --ckpt /mnt/data3/DeepSeek-V4-Flash-0731 \
  --binary cpp_engine/build/pocketllm_engine \
  --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  --devices 0,1,2,3 --layers 0 --max-context 8192 \
  --mode serial --max-batch-size 1 --prefill-token-budget 0 \
  --max-tokens 32 --short-prompt-words 128 --long-prompt-words 1600 \
  --json-out /tmp/dsv4-serial.json
```

The batch arm is the same command with `--mode batch --max-batch-size 8
--prefill-token-budget 4096`. Each configuration gets its own server process and
its own model load.

`scripts/bench_cpp_openai_phases.py` splits a single request into prefill and
decode. A whole-request `output_tokens_per_second` is not enough to compare
configurations, because it mixes a prefill term that scales with the prompt
against a decode term that does not:

```bash
python scripts/bench_cpp_openai_phases.py \
  --ckpt /mnt/data3/DeepSeek-V4-Flash-0731 \
  --binary cpp_engine/build/pocketllm_engine \
  --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
  --devices 0,1,2,3 --layers 0 --max-context 8192 \
  --max-batch-size 1 --prefill-token-budget 0 \
  --max-tokens 32 --short-prompt-words 128 --long-prompt-words 1600 \
  --repeats 3 --warmup-rounds 1 \
  --json-out /tmp/dsv4-phases.json
```

Both harnesses run one configuration at a time. Per
[the benchmarking rules](../guides/benchmarking.md) and this host's memory
behaviour, configurations are never run concurrently: a second engine in the same
process, or a second configuration on the same devices, changes the number.

### Why the phase split comes from `/metrics` and not from the stream

The native server emits no per-token timing on the wire. Its stream cannot be
used to delimit the phases either: `handle_stream` writes the role chunk
*before* it calls `sched.submit_request` (`cpp_engine/engine/openai_server.cpp`),
so the client sees a first event within milliseconds of the request regardless of
prompt length. The `deepseek_timings` field belongs to the *Python* server
(`src/server/openai.py`) and never appears on this path.

The engine does keep exact per-request accounting in
`cpp_engine/core/metrics.cpp` and exports it as Prometheus counters at
`/metrics`. Reading them before and after one streamed request gives that
request's split from the engine's own clock:

```
prefill_seconds = pocket_ttft_seconds_sum delta
decode_tokens   = pocket_tokens_total{type="generation"} delta - 1
decode_seconds  = pocket_request_duration_seconds_sum delta - prefill_seconds
```

!!! note "True of the commit this run was taken on"
    The subtraction above is how the phases had to be recovered when this page
    was measured: `/metrics` exported a TTFT family and an end-to-end family and
    nothing else. The engine now records `pocket_request_queue_time_seconds`,
    `pocket_request_prefill_time_seconds` and `pocket_request_decode_time_seconds`
    directly, from the scheduler's own clock, so neither phase is derived any
    more and the decode interval no longer absorbs the duration family's
    response assembly. TTFT comes from the same scheduler result rather than
    from a latch in the HTTP handler. The results below are unchanged: they are a
    record of commit `1536681`, not of the current tree.

The first generated token belongs to prefill, so it is subtracted from the
decode count. The deltas are attributed to one request by asserting that the
TTFT, duration, and success counters each moved by exactly 1; without that check
the numbers could silently belong to a neighbour.

Two consequences are recorded rather than hidden:

- `duration` runs to the end of the response, which includes response assembly
  after the last token, so `decode_seconds` is an **upper bound** on the decode
  intervals and every decode rate on this page is a **floor**.
- The client-observed total is reported alongside as a cross-check on the
  engine's clock. The two agree to about a millisecond (8.640 s against 8.641 s
  for one tuned short-prompt sample), which is the expected result: the client
  cannot finish earlier than the engine that started before it.

Only the streaming handler records TTFT at the first token.
`handle_nonstream` passes a null token callback to `sched.submit_request`, so the
engine has no first-token instant on that path and its TTFT equals its full
duration. That was measured, not assumed — the first attempt at this harness
failed on `duration 8.8587 does not exceed ttft 8.8588`. The phase harness
therefore streams, and asserts that the first stream event arrives before TTFT.

!!! note "Fixed since"
    That equality was a property of the pop-time latch, not of the engine: the
    latch fired on the first read that did not time out, which for a
    non-streaming request is the completion itself. TTFT now comes from the
    scheduler's result — the instant the first token was produced, on the
    engine's own clock — so a non-streaming request reports a TTFT well below its
    duration and the failure quoted above can no longer occur. Why this page's
    harness streams is unchanged: the non-streaming path still has no
    token-bearing event to delimit the phases on the wire.

## Results

### Serial, whole request

Each row is one server process. `n=2/4/8` are the concurrency ladder, and the
interleave figure is the latency of a short request started while a long request
is in flight. Commit `1536681`, tuned or default environment as marked.

| Run | Env | Mode | Single request | n=2 | n=4 | n=8 | Interleave, short latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| serial  | default | serial, mbs=1 | 11.922 s / 2.684 tok/s | 2.686 | 2.686 | 2.686 | 26.384 s |
| serial2 | default | serial, mbs=1 | 11.335 s / 2.823 tok/s | 2.798 | 2.806 | 2.803 | 25.390 s |
| tuned   | tuned | serial, mbs=1 | 9.330 s / 3.430 tok/s | 3.443 | 3.449 | 3.444 | 20.439 s |
| tuned2  | tuned | serial, mbs=1 | 8.682 s / 3.686 tok/s | 3.683 | 3.775 | 3.770 | 19.289 s |
| batch   | tuned | batch, mbs=8, budget 4096 | 8.845 s / 3.618 tok/s | 3.582 | 3.538 | 3.544 | 20.187 s |

Throughput figures are output tokens per second over the whole request. The
ladder columns are per-request-multiplied aggregate rates.

Two things to read out of this table:

- **Throughput is flat inside every row.** Doubling concurrency does not change
  the aggregate rate at all (2.684 → 2.686 → 2.686). Requests are served one at
  a time; adding load lengthens the queue and every client's latency
  proportionally — the `count=8` requests in the serial run finish at 11.9 s,
  23.8 s, 35.7 s, …, 95.3 s, i.e. a pure serialization.
- **The batch arm is not faster.** Its single-request figure (3.618) and its
  ladder (3.582/3.538/3.544) both sit inside the spread between `tuned` and
  `tuned2`, which are the *same command run twice*: 3.430 against 3.686 is 7.5%
  of run-to-run variation on identical inputs. A difference smaller than the
  spread between two identical runs is not a result, and it would be wrong to
  present `--max-batch-size 8` as a throughput win on the strength of it.

### Prefill and decode separately

Same checkpoint, commit, and devices. Each cell is the median of three samples
after one discarded warmup round; the brackets are the min and max of those
three. `ptok` is the prompt token count the engine counted, `ctok` the generated
count. Decode rates are floors, per [above](#why-the-phase-split-comes-from-metrics-and-not-from-the-stream).

| Env | Prompt | ptok | Prefill | Prefill tok/s | ctok | Decode tok/s | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned | short | 120 | 2.699 s | 44.5 (43.0–44.7) | 32 | **5.248** (5.205–5.371) | 8.606 s |
| tuned | long | 1446 | 5.139 s | 281.4 (276.2–284.3) | 32 | **4.868** (4.785–4.936) | 11.516 s |
| default | short | 120 | 2.580 s | 46.5 (46.2–46.7) | 32 | **3.554** (3.529–3.571) | 11.303 s |
| default | long | 1446 | 5.914 s | 244.5 (244.0–245.9) | 32 | **3.328** (3.248–3.387) | 15.117 s |
| batch | short | 120 | 2.775 s | 43.2 (42.9–43.8) | 32 | 4.927 (4.872–5.114) | 9.066 s |
| batch | long | 1446 | 5.148 s | 280.9 (274.4–282.9) | 32 | 4.446 (4.441–4.602) | 12.120 s |

`batch` here is the tuned environment with `--max-batch-size 8
--prefill-token-budget 4096`; the other two are `--max-batch-size 1
--prefill-token-budget 0`.

- **The tuned environment's whole advantage is in decode**: 5.248 against 3.554
  tok/s at the short prompt and 4.868 against 3.328 at the long one, i.e.
  **1.48x and 1.46x**. Its prefill is marginally *worse* at the short prompt
  (44.5 against 46.5 tok/s) and better at the long one (281.4 against 244.5).
- **Decode degrades with context depth**: 5.248 → 4.868 tok/s (tuned) and
  3.554 → 3.328 (default) as the prompt grows from 120 to 1446 tokens. A decode
  figure without its context is meaningless, and this is the width of that
  dependence over the range measured here.
- **The batch arm again lands inside the tuned arm**, on every field including
  the ones the whole-request table cannot separate. Its prefill and its decode
  are both within the spread of the two tuned serial runs.
- **The spread within one configuration is a few percent.** The short-prompt
  prefill range is 43.0–44.7 tok/s (4%) and the long-prompt decode range is
  4.785–4.936 (3%). Differences below that are not measurable with three
  samples on this host.

The prefill column needs one caveat: 120 tokens at 44.5 tok/s is a bounded-cost
measurement, not a throughput measurement. These two points give a marginal rate
of 543 tok/s over the 120 → 1446 token interval, but a marginal rate over an
interval that contains the fixed per-request cost is not a throughput, and the
2026-08-03 record below shows the rate is not constant over a wider range. No
curve was fitted and no third length was measured, so no marginal or asymptotic
prefill rate is claimed.

### Peak memory

Each row is one server process in a run of its own: model load, one 128-word
request, one 1600-word request, and a sampler reading
`nvidia-smi --query-gpu=index,memory.used` and `/proc/<pid>/smaps_rollup` every
two seconds from before the load to after the last response. The GPU columns are
per rank, and steady is the median over the second half of the loaded samples.

| Env | Mode | Peak GPU per rank | Steady GPU per rank | Host |
| --- | --- | ---: | ---: | ---: |
| default | serial, mbs=1 | 8.19 GiB | 6.29 GiB | 137 GiB |
| tuned | serial, mbs=1 | 15.57 GiB | 8.15 GiB | 137 GiB |
| tuned | batch, mbs=8 | 16.42 GiB | 9.73 GiB | 137 GiB |

The tuned environment costs 7.4 GiB per rank over the default at peak and 3 GiB
at steady, which is the price of keeping all 43 per-layer arenas resident, and
the heaviest configuration still leaves this host's 22 GiB per card room for a
KV cache. The ladder runs' own sampler read the batch arm at 15.68 GiB; the two
runs sampled different moments of the same configuration, and 16.42 GiB is the
figure to plan against.

The host column does not move between the three configurations — summed `Pss`
over the four rank processes is 285 GiB in each, and the three values agree to
within 15 MB — and it is the one number on this page that is easy to get wrong,
so the measurement behind it is worth stating:

- With the tuned server loaded, `Shmem` in `/proc/meminfo` is 137.1 GiB, and it
  is 0.0 GiB once the engines exit. That is the staging arenas: 34.4 GiB of
  shared-writable mapping per rank, in four rank processes, not five.
- `MemAvailable` falls from 997.1 GiB idle to 853.0 GiB loaded — 144.1 GiB, of
  which 137.1 GiB is the arenas and the rest is anonymous and pinned memory. It
  returns to 997.1 GiB when the server stops.
- Summed `Pss` reads 285 GiB and is **not** a footprint. Each rank maps the
  whole checkpoint read-only from `/mnt/data3`, and `smaps` charges those
  page-cache pages to every process that maps them while `Pss` divides only the
  mappings it can tell are shared. The checkpoint pages are page-cache backed
  and reclaimable, and on this host they were resident before the load. Measured
  against `MemAvailable`, which cannot double-count, and against `Shmem`, which
  returns to zero, the engine's own host allocation is the 137 GiB of arenas.

"About 137 GiB of host memory per server" is therefore the statement to carry
forward, with the GPU column above as the resource that actually constrains the
slot count.

## Why the first batch arm was throughput-neutral

**Every measurement in this section is from commit `1536681`.** It is kept because
it is the explanation for the batch rows of the tables above, which were taken
there. It is no longer the behaviour of the code: `caps()` now asks the engine,
and the [arm below](#batch-mode-after-the-batched-forward) runs at width 8. Read
it as the record of why the 2026-09-14 batch column was a control.

`--max-batch-size 8` does reach the engine —
`engine_registry_builtin.cpp` wires it into `max_slots`, and the batch arm's log
line says `engine declares max_slots=8`. The scheduler then declines to use it.
`cpp_engine/engine/batch_scheduler.cpp`:

```cpp
int allowed = caps_.continuous_batching ? caps_.max_slots : 1;
```

and `PersistentEngineAdapter::caps()`
(`persistent_engine_adapter.cpp`) hardcoded `continuous_batching = false`. So
every batch arm logged:

```
[server] batch width 1 (engine declares max_slots=8, continuous_batching=no), prefill budget 0
```

There was a second clamp in the same file, `if (!caps_.chunked_prefill)
prefill_token_budget_ = 0;`, which is why `--prefill-token-budget 4096` also had
no effect: the same `false` zeroed the budget. Both clamps came from one
capability flag, and only the first of them was about batching: `chunked_prefill`
still reports false, which is why the budget is still zero in the arm below.

The reason the flag was `false` is visible in the implementation rather than the
scheduler. `PersistentEngine::batch_decode_step`
(`deepseek_v4_engine.cpp`) validates that there are no more than 8 requests with
unique slots and then decodes them in a `for` loop, one `worker_command_decode`
plus one `run_safetensors_token_forward_impl` per request. The row-batched
implementation it would need, `run_safetensors_continuation_batch_impl`, exists
and is genuinely row-batched, but it assumes contiguous positions:
`head_rmsnorm_rope_freqs_rows_kernel` computes `position = start_position + token`
from a single scalar, and the per-row KV index build, compressor update, and ring
publish all derive their positions from `start_position + row`. Independent slots
need those to become per-row, which is the actual work item behind #163.

So the honest reading of the batch column is not "batching is neutral" but
"batching is not yet switched on, and these numbers are the width-1 control the
real implementation will be measured against."

## Batch mode after the batched forward

Three commits separate this arm from the one above. #239 gave
`PersistentEngine::batch_decode_step` a real row-batched forward behind
`POCKETLLM_CPP_BATCHED_DECODE` (default off, serial kept as the reference), #241
scoped that forward's decode state to its own slot and its own rank, and the third
is the one recorded on this page: `PersistentEngineAdapter::caps()` now reports
`continuous_batching = max_slots > 1 && engine_->batched_decode_enabled()` instead
of the hardcoded `false` fixed earlier in the file, which is what makes the switch
visible from outside the engine. `BatchScheduler` clamps on that flag, so the arm
runs at width 8 instead of being silently reduced to 1.

The startup line is the evidence that it did:

```
# switch unset, --max-batch-size 8
[server] batch width 1 (engine declares max_slots=8, continuous_batching=no), prefill budget 0
# POCKETLLM_CPP_BATCHED_DECODE=1, --max-batch-size 8
[server] batch width 8 (engine declares max_slots=8, continuous_batching=yes), prefill budget 0
```

The prefill budget is still zero in both arms. It is zeroed by `chunked_prefill`,
which is a separate capability — this engine has no block allocator to resume an
unfinished prompt against — and it is deliberately not part of the change.

Method: commit `ccb339d` plus the `caps()` change recorded here, the tuned
environment, the same two harnesses as above, one server process per
configuration, run one after another. The batch arm is the same command with
`POCKETLLM_CPP_BATCHED_DECODE=1` exported; the harnesses copy the parent
environment into the ranks they start, so nothing else differs.

### Whole request

| Arm | Single request | n=2 | n=4 | n=8 | Interleave, short latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial, mbs=1 | 8.656 s / 3.697 tok/s | 17.734 s / 3.609 | 34.760 s / 3.682 | 69.495 s / 3.684 | 19.660 s |
| batched, mbs=8 | 8.978 s / 3.564 tok/s | 16.581 s / 3.860 | 27.624 s / 4.634 | 42.927 s / 5.964 | 18.538 s |
| batched / serial | 0.96x | 1.07x | 1.26x | **1.62x** | 1.06x |

Rates are output tokens per second over the whole request; the ladder columns are
per-request-multiplied aggregate rates, as in the serial table above.

- **The ladder is no longer flat.** Serial holds 3.609 → 3.682 → 3.684 across
  n=2/4/8, which is the serialization recorded above; the batch arm rises 3.860 →
  4.634 → 5.964, a 1.62x wall-time improvement at n=8. Requests in that arm are
  genuinely running in the same forward pass.
- **The serial row reproduces the baseline table**, which is the check that
  nothing on this branch moved the width-1 path: 3.697 against the baseline's
  3.686 at a single request, 3.684 against 3.770 at n=8, and a single-request
  wall of 8.656 s against 8.682 s.
- **The gain is an amortization, not free parallelism.** At n=8 the batch arm
  spends 42.927 s on 32 decode steps, 1.342 s per step for the whole batch; one
  row alone spends 8.656 s, 0.271 s per step. Eight rows therefore cost 4.96
  single-row steps, an effective width of 1.62 — the same number as the speedup,
  as it must be. Rows share the step but do not disappear inside it.
- **Prefill is still one request at a time** (`batch_prefill` loops over the
  requests it is handed), so the wall-time speedup understates the decode gain.
  If the 8 short prefills cost the same 2.716 s each that the phase table
  measures, the n=8 arm spends ~21.7 s in prefill and decodes 256 tokens in
  ~21.2 s against serial's 8 x 8.656 - 21.7 = ~47.6 s, which would put the
  decode-only gain near 2.2x. That is an estimate built on a prefill cost
  measured one request at a time, not a second measurement, and it is written
  here as an estimate.
- **The queue changes shape before it gets shorter.** At n=8 the serial arm's
  average latency is 39.113 s with a worst case of 69.490 s; the batch arm's is
  42.830 s with a worst case of 42.925 s. Every row in a batch finishes with the
  batch, so a request that would have been near the front of the queue now waits
  for its slowest neighbour — a lower ceiling, a higher floor. Whether that is an
  improvement depends on which end of the distribution the caller cares about.

### Single request, phase split

| Arm | Prompt | ptok | Prefill | Prefill tok/s | ctok | Decode tok/s | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | short | 120 | 2.716 s | 44.2 (44.0–44.4) | 32 | **5.370** (5.314–5.423) | 8.489 s |
| serial | long | 1446 | 5.163 s | 280.1 (279.8–282.0) | 32 | **4.685** (4.442–4.872) | 11.780 s |
| batched | short | 120 | 2.739 s | 43.8 (43.4–44.8) | 32 | **5.422** (5.199–5.450) | 8.482 s |
| batched | long | 1446 | 5.113 s | 282.8 (281.2–283.9) | 32 | **4.988** (4.796–5.099) | 11.173 s |

Median of three samples after one discarded warmup round, ranges in parentheses,
decode rates are floors.

This is the acceptance criterion that had to survive, and it does: **turning the
batched forward on does not cost single-request latency.** The four pairs overlap
on every field. The one gap worth naming is long-prompt decode, 4.685 → 4.988, or
+6.5%, and 6.5% is smaller than the 7.5% spread recorded above between two
identical serial runs of this command, so it is not a result. A single-row batch
takes the serial branch of the switch in any case: `batch_decode_step` falls back
to its per-request reference loop at `requests.size() == 1`.

### Peak memory

Each row is one server process in a run of its own, sampling
`nvidia-smi --query-gpu=index,memory.used` every two seconds from before the
model load to after the last response. The comparison is at the concurrency
ladder, because that is where eight rows are in flight at once and therefore
where width changes what is resident.

| Arm | Scheduler width | Peak GPU per rank | `Shmem` |
| --- | ---: | ---: | ---: |
| serial, mbs=1 | 1 | 15.90 GiB | 137.1 GiB |
| batched, mbs=8 | 8 | **17.52 GiB** | 137.1 GiB |

Four and a half GiB of the 22 GiB card remain at the highest configuration
measured here, and the host figure does not move. Widening the scheduler from 1
to 8 costs about 1.6 GiB per rank, which is the eight slots' KV and the
row-batched decode workspace being live together rather than one at a time.

## Environment ablation

The tuned environment is four switches, one of which
(`POCKETLLM_CPP_PREFILL_BATCHED_ATTN=1`) is the built-in default and therefore not
a variable at all. The other three were applied one at a time on top of the
otherwise default environment, each in its own server process, with the same
command and the same harness as the phase table above. `sparse + cached` is the
pair the source comment predicts is the operative combination.

| Arm | Short prefill | Long prefill | Short decode | Long decode |
| --- | ---: | ---: | ---: | ---: |
| default (nothing set) | 46.5 (46.2–46.7) | 244.5 (244.0–245.9) | 3.554 (3.529–3.571) | 3.328 (3.248–3.387) |
| `POCKETLLM_CPP_DECODE_SPARSE_ARENA=12` | 45.2 (44.5–45.5) | 241.4 (241.0–244.2) | 3.598 (3.592–3.602) | 3.420 (3.356–3.435) |
| `POCKETLLM_CPP_PREFILL_MOE_PREFETCH=1` | 45.2 (43.6–45.3) | **263.3** (259.0–273.5) | 3.428 (3.416–3.442) | 3.296 (3.295–3.304) |
| `DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=43` | 46.6 (45.8–46.8) | 228.4 (221.6–239.7) | 3.577 (3.573–3.577) | 3.352 (3.328–3.359) |
| `SPARSE_ARENA=12` + `MAX_CACHED_LAYERS=43` | 42.6 (38.2–43.3) | 212.2 (211.4–218.8) | **5.104** (4.751–5.239) | **4.925** (4.876–4.932) |
| all four (the tuned environment) | 44.5 (43.0–44.7) | **281.4** (276.2–284.3) | **5.248** (5.205–5.371) | **4.868** (4.785–4.936) |

Decode rates in tok/s, prefill rates in tok/s, each a median of three samples
with the min and max in parentheses.

**No single switch produces the decode gain.** All three single-switch arms land
within noise of the default environment — 3.598, 3.428 and 3.577 short-prompt
decode tok/s against the default's 3.554, and 3.420, 3.296 and 3.352 long-prompt
against 3.328. Given the 7.5% spread between two identical default runs reported
above, none of these is a difference.

**The pair does.** `POCKETLLM_CPP_DECODE_SPARSE_ARENA=12` together with
`DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=43` reaches 5.104 and 4.925 decode
tok/s, which brackets the full tuned environment's 5.248 and 4.868. The tuned
environment's entire decode advantage, and therefore the control that #163 has to
preserve, is the interaction of these two switches.

That is what the source says to expect. The sparse-arena switch carries this
comment in `deepseek_v4_engine.cpp`:

> When decode uses a sparse per-layer arena (matching PyTorch's lazy per-layer
> cache), prefill's dense (full n_local_experts) arenas would double the
> per-layer footprint and OOM at MAX_CACHED_LAYERS=43. Drop them here so decode
> can keep all 43 sparse arenas resident.

Setting the cache depth without the sparse arena evicts the arenas during prefill
and leaves nothing resident for decode; setting the sparse arena alone never asks
for the 43 layers to be kept. Only together do the 43 layers stay on the device,
and decode stops paying a per-step host transfer for the experts it re-reads.

The practical consequence is that these are **not independent knobs**, and
neither may be reported alone as "the decode setting". A future change that
touches the arena allocator, the LRU, or the prefill arena release has to be
measured as this pair, because breaking the interaction would silently return
decode to the default rate.

**`POCKETLLM_CPP_PREFILL_MOE_PREFETCH` buys prefill.** On its own it lifts the
long-prompt prefill from 244.5 to 263.3 tok/s, +7.7%, with ranges that do not
overlap. It also repairs the regression the arena pair introduces: the pair
prefills at 212.2 tok/s, below even the default, and the tuned environment's
281.4 tok/s is that pair plus prefetch. The mechanism is consistent with what the
switch does — overlapping the next layer's expert transfer with the current
layer's compute — and with the fact that the pair makes prefill the phase that
pays for the arenas being held.

**Prefill is the noisy column here.** The prefetch arm's long-prompt range spans
259.0–273.5 and the cached arm's 221.6–239.7, several times the spread of the
default arm's 244.0–245.9 on the same harness. The short-prompt prefill column is
the least trustworthy of all: at 120 tokens it measures the fixed per-request
cost rather than a rate, and the pair's range there is 38.2–43.3. The decode
columns are tight (often under 1%) and are the part of this table worth acting
on.

## Comparison with the 2026-08-03 record

An earlier end-to-end baseline exists for this checkpoint and TP4 on these four
cards (`dsv4_0731_e2e_perf_baseline`, 2026-08-03). It is preserved; these numbers
do not replace it, and they are not subtractable from it.

| Quantity | 2026-08-03 | This page (tuned) |
| --- | --- | --- |
| Prefill clock | client-side, role marker to first content | engine-side, `pocket_ttft_seconds_sum` |
| Prefill, ~1440 tokens | ~5.2 s (interpolated from 786 → 3.57 s and 1602 → 5.40 s) | 5.139 s at 1446 tokens |
| Prefill, fixed cost | 0.989 s measured with a 6-token prompt | not separable from the 120-token point |
| Prefill marginal | ~378 tok/s over the 6396 → 9609 token range | 543 tok/s over 120 → 1446 |
| Decode, low context | 6.34 tok/s at context 22 (steady, first 4 tokens skipped) | 5.248 tok/s at 120 (all decode steps, upper-bounded interval) |
| Decode, ~1600 context | 5.20 tok/s at context 1602 | 4.868 tok/s at 1446 |

At the one prompt length where the two overlap, prefill agrees: 5.139 s here
against an interpolation of the old curve at 1446 tokens. Nothing else is
comparable. The old prefill clock started when the role marker reached the
client, which is *before* the request was submitted, so it measures a longer span
than TTFT and cannot be subtracted from these numbers; the old decode figures
skip four warm-up tokens, which this page does not. The 363 tok/s asymptote and
the 0.989 s fixed cost are properties of that measurement path, and the marginal
column above is deliberately left as an interval rate rather than converted into
either.

## Verdict for #163

What this establishes:

1. **A serial baseline exists** for the PersistentEngine on the real checkpoint.
   Whole-request throughput is 2.684–2.823 output tok/s in the default
   environment and 3.430–3.686 in the tuned one; split by phase, the tuned
   environment decodes at 5.248 and 4.868 tok/s at 120 and 1446 prompt tokens
   and prefills the long prompt at 281.4 tok/s. Its 44.5 tok/s at 120 prompt
   tokens is a fixed-cost measurement rather than a rate. It re-measures today at
   3.697 tok/s whole-request, 5.370 and 4.685 tok/s of decode, and 280.1 tok/s of
   long-prompt prefill.
2. **The default environment is not the baseline to beat.** It decodes 1.5x
   slower. Any change to the decode path has to be measured against the tuned
   environment, not against an unset shell.
3. **Batching is on, behind a switch.** The 2026-09-14 batch arm was clamped to
   width 1 by a hardcoded `caps().continuous_batching`; that is fixed, and with
   `POCKETLLM_CPP_BATCHED_DECODE=1` the scheduler admits eight requests and runs
   them in one batched forward, for 1.62x wall time at n=8 (3.684 → 5.964 output
   tok/s aggregate). The switch is off by default, so an unset environment still
   gets the serial reference path.
4. **Single-request latency is unchanged by batching**, which is the criterion
   that had to hold: the phase split overlaps across the whole table and the
   whole-request difference is 3.7%, inside this host's run-to-run spread. What
   does change at saturation is the shape of the queue — a lower worst case and a
   higher average, because a row now finishes with its batch.
5. **The gain is bounded by amortization.** Eight rows cost 4.96 single-row steps
   of work, so the effective width is 1.62, not 8. Nothing here shows a batched
   forward that scales with the number of rows; it shows one whose step cost grows
   with them and is worth paying from about n=4 upward.

What this does **not** claim: paged KV, chunked prefill, or a prefill token
budget. `caps()` still reports `chunked_prefill = false` and `paged_kv = false`,
so `--prefill-token-budget` remains a no-op on this engine and prefills stay one
request at a time — which is also why the decode-only gain is larger than the
wall-time one. Slot isolation, slot reuse, cancellation, and TP parity across
both arms of the switch are covered by
`cpp_engine/tests/test_multi_slot_decode_parity.cpp`, which runs at `tp_world` 1
and 4 and whose tp4 and tp1 chains are token-identical over eight steps.
