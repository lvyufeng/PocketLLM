# Serving latency with IPC all-reduce and device-side wait

This record measures the same workload as
[serving_latency_baseline.md](serving_latency_baseline.md) with two optimizations
enabled: the hand-written IPC all-reduce and the device-side arrival wait, then
selected with `POCKET_ASCEND_IPC_ALLREDUCE=1` and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1`. Both are documented in
[Ascend 910A single-request decode](ascend_single_request_tps.md).

**Both switches have since become the shipped defaults, and neither prints
anything at startup.** The run below was taken before either flip and is left as
it was measured, so its two exports are the record of the configuration it ran
under rather than something a reader has to set. An unset environment runs both
levers; `POCKET_ASCEND_IPC_ALLREDUCE=0` is what reaches this page's
[baseline](serving_latency_baseline.md); and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0` reaches the host poll the device-side wait
replaced. `serving_throughput_scaling.md` is where the second flip's own
measurement lives, and it is the one that carries the split between the two.

## Scope

The measured path:

- Qwen3.8-27B safetensors, `/mnt/data1/modelscope/Qwen/Qwen3.8-27B`
- native `pocketllm_engine`, TP4 across four Ascend 910B (first generation)
- OpenAI `/v1/chat/completions`, streaming SSE
- `--no-kv-paged`, `max_context` 8192, batch width 8, prefill budget 4096
- commit `4fc72a1` (master as of 2026-09-19), binary `cpp_engine/build-ascend/pocketllm_engine`
- **`POCKET_ASCEND_IPC_ALLREDUCE=1 POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1`** —
  both were required at the time of the run and both are the defaults now, so an
  unset environment reproduces this record and neither export is needed
- `QWEN_ASCEND_REPLICATE_ROWS` at its default of 1, i.e. the Cube's row
  replication is **off** — see [the ladder](#the-same-levers-inside-the-engine)

The engine sources are identical to the baseline record's: `git diff 0de9d41
4fc72a1 -- src pocketllm cpp_engine scripts tests` is empty, and only `docs/` and
`mkdocs.yml` differ between the two commits. The baseline run's own commit
`0de9d41` has a tree identical to `1638d07`. So the comparison isolates the two
environment variables.

## Reproduction

```bash
source scripts/ascend_env.sh     # without it an ACL binary hangs before aclInit returns
export POCKET_ASCEND_IPC_ALLREDUCE=1        # the default now; kept as it was run
export POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1   # the default now too
python scripts/bench_serving.py \
    --ckpt /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
    --binary cpp_engine/build-ascend/pocketllm_engine \
    --devices 0,1,2,3 --device-style ascend \
    --endpoint /v1/chat/completions \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 24 --request-rate 2 --max-concurrency 8 --num-warmups 1 \
    --goodput ttft:2000 tpot:200 e2el:30000 \
    --log-dir /tmp/serve-opt-logs --json-out /tmp/serve-optimized.json
```

The command line is the baseline's with the two exports added and the log and
JSON destinations renamed. The recorded artifact is `/tmp/serve-optimized.json`,
whose `git_commit` field is `4fc72a1`; the server logs are under
`/tmp/serve-opt-logs/`. `--token-latch content` is the default and is what was
used, so `ttft` is latched on the first chunk carrying content, not on the first
chunk.

## Results

24 requests, 0 failures, 40.744 s of measured wall time. Latencies in
milliseconds; E2EL in seconds.

| Metric | Mean | Median | Std | P99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 927.26 | 901.26 | 490.70 | 2165.50 |
| TPOT | 99.49 | 102.15 | 10.08 | 114.50 |
| ITL | 99.51 | 82.08 | 97.56 | 497.65 |
| E2EL | 12.702 | 13.085 | 1.547 | 15.047 |

| Quantity | Value |
| --- | ---: |
| Completed / failed | 24 / 0 |
| Generated tokens | 2864 (per-request output lengths 98–128) |
| Request throughput | 0.589 req/s |
| Output throughput | 70.291 tok/s |
| Peak output tokens/s | 104 |
| Peak concurrent requests | 11 |
| Request goodput | 0.564 req/s |

The goodput figure is a **rate**, 0.564 of 0.589 req/s, so it is 23 of the 24
requests meeting all three of `ttft:2000 tpot:200 e2el:30000`; one request
exceeded the TTFT budget.

Per stream, TPOT 99.49 ms is **10.05 tokens/s**; the aggregate 70.29 tokens/s is
what the concurrent streams add up to.

The role-chunk cost the baseline page measures is unchanged on this tree: mean
`ttft_first_chunk` is 5.01 ms against the 927.26 ms latched on content, a
922.25 ms mean difference. The optimization moves the second number and not the
first, which is the point — the role chunk is written before the backend is
called.

## Comparison against baseline

Against the [baseline record](serving_latency_baseline.md), same workload, same
client, default configuration:

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| TPOT mean | 151.30 ms | 99.49 ms | **-34.2%** |
| TPOT median | 155.60 ms | 102.15 ms | **-34.4%** |
| TPOT P99 | 165.64 ms | 114.50 ms | **-30.9%** |
| TTFT mean | 1018.02 ms | 927.26 ms | **-8.9%** |
| TTFT median | 977.17 ms | 901.26 ms | **-7.8%** |
| TTFT P99 | 2276.78 ms | 2165.50 ms | -4.9% |
| E2EL mean | 18.831 s | 12.702 s | **-32.5%** |
| Output throughput | 47.279 tok/s | 70.291 tok/s | **+48.7%** |
| Request throughput | 0.398 req/s | 0.589 req/s | **+47.9%** |
| Duration | 60.26 s | 40.74 s | **-32.4%** |

**Per-stream decode rate improved from 6.61 tok/s to 10.05 tok/s**, a 52%
increase. The total output throughput across concurrent streams rose 48.7%.

TTFT improved by a smaller margin (8.9%) than TPOT did. Prefill is not what these
two switches change — the collective sits in the decode step — so the TTFT
reduction is consistent with lower queue wait rather than with a faster prefill:
requests finish sooner and free their slots sooner, and at 2 req/s with 8 slots
the queue is a real term in the baseline's 1018.02 ms. This run does not separate
the two contributions, and a concurrency-1 repeat would be the way to.

The comparison is one pair of runs, not an interleaved series, so the change
column carries this host's run-to-run spread as well as the lever. The effect is
far outside it — TPOT differs by a third — but the smaller entries (TTFT P99 at
-4.9%) should be read as "unchanged" rather than as a measured improvement.

## What the optimizations do

Both target the all-reduce collective that synchronizes activations across TP
ranks during decode. A decode step issues **129** of them — 64 `ar.mlp`, 48
`ar.lin.out`, 16 `ar.full.out` from the three projection sites in
`qwen_layer_components.inl`, plus 1 `ar.hidden_a` for the embedding output — each
reducing the same 5120-element fp16 plane, 10240 B. One per-call saving is
therefore multiplied by 129.

1. **The hand-written IPC all-reduce**, which replaced `HcclAllReduce` as the
   backend's default (`POCKET_ASCEND_IPC_ALLREDUCE=0` is the way back to it). It
   is a copy-and-barrier that reads peer buffers directly and synchronizes with an
   arrival signal carried in the payload itself. The empty-loop cost drops from
   `HcclAllReduce`'s 0.4810 ms to 0.26 ms per call.

2. **The device-side arrival wait** (`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT`, written
   `=1` when this run was taken and the default now) moves the arrival poll off the
   host and into a kernel-side busy-wait (`qwen_ipc_arrive_wait_kernel`),
   eliminating the host round trip per collective. This stacks with the hand-written
   barrier rather than replacing it; `=0` is the way back to the host poll.

### The same levers inside the engine

The engine-internal record measures the same two switches in the same session at
rows=1, with the TP4 launcher and one process per rank
([Ascend 910A single-request decode](ascend_single_request_tps.md), §6.3):

| collective | `step_ms` | decode TPS |
| --- | ---: | ---: |
| `HcclAllReduce` (the `POCKET_ASCEND_IPC_ALLREDUCE=0` arm) | 103.8 | 9.63 |
| IPC all-reduce, host poll (the `..._DEVWAIT=0` arm) | 76.6-77.1 | 12.96-13.06 |
| IPC all-reduce + device wait — the shipped pair | **53.2-53.6** | **18.66-18.79** |

23.4 ms of the intermediate 77.1 ms step is the host's round trip through the
runtime, which is what the device-side wait removes. The serving numbers above
confirm that the engine-internal improvement survives at batch width 8 over HTTP.

**The 39.4-39.8 ms / 25.15-25.37 TPS step that page records as the fastest on
this checkpoint is a third lever, and this run does not have it.** Those numbers
are the two switches above *plus* `QWEN_ASCEND_REPLICATE_ROWS=16`, which fills the
Cube's sixteen-row M tile from a decode step's single activation row by
broadcasting it. That variable defaults to 1, i.e. off, and it was not set here —
so 53.2-53.6 ms, not 39.4-39.8 ms, is the engine-internal counterpart of the TPOT
on this page.

All configurations in that table emit the reference's own tokens, so the lever is
not paid for with accuracy.
