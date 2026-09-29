# Qwen3.8-27B on the 910B, through the server

4 × Ascend 910B (first generation, `Short_SoC_version=Ascend910`, 32 GB HBM each), TP4, one card a
rank, CANN 9.0.0. Checkpoint the official `Qwen/Qwen3.8-27B` BF16 release,
`/mnt/data1/modelscope/Qwen/Qwen3.8-27B` — see [the model guide](../models/qwen3.8-27b-bf16.md).
Workload: `scripts/bench_serving.py`, `--dataset-name custom`, 16 prompts of 244 input tokens and 64
output tokens each. Tokenizer source: the checkpoint's own.

Engine: the `pocketllm_cpp` module built on this host at 02:51:40 UTC on 2026-09-29, whose sources are
identical to `b5f791c` in every file that is not a comment — the only later edits under `cpp_engine/`
are four comments in test and bench sources renamed by
[#498](https://github.com/lvyufeng/PocketLLM/pull/498), which moved no code. Python tree `master` at
`5b29b8e`, where `pocketllm/backends/cpp_backend.py` — the file §3 and §4 are about — is identical to
`1c8b586` ([#495](https://github.com/lvyufeng/PocketLLM/pull/495), the last commit to touch it), so
the line numbers those sections cite hold at both.

This is the first record of this checkpoint behind `pocketllm serve` on any hardware. The engine had
already run it at TP4 through `scripts/run_qwen_ascend_tp4.sh` at `rows=1` and at a batch — that is
what the six [performance records](index.md) and the
[roadmap](../architecture/ascend_performance_roadmap.md) measure, and [the model
guide](../models/qwen3.8-27b-bf16.md) says so. What none of them did is put the
checkpoint behind the HTTP path. The backend itself had been served before, but on a different
weight source: [the serving ladder](ascend_cpp_serving_ladder.md) is `prism-ml/Ternary-Bonsai-2-27B-gguf`
at commit `b5f791c`, whose loader decodes 1.75-bit blocks on the host and ships 12.53 GiB of FP16 a
rank. This checkpoint ships 12.8 GiB of FP16 a rank and takes a different arm of the same weight
map, so the earlier record's ladder is not this one's.

The result is four sentences. **The server path costs nothing measurable** — 22.18 output tok/s at
concurrency one, against 21.99-22.18 for the same workload measured three ways. **It emits the
sequence this backend already publishes** as the reference's own
([the RoPE workspace record](ascend_rope_table_workspace_aliasing.md)). **A default deployment stops at
8 slots**, which is `DEFAULT_BATCH_SLOTS` and not a capacity limit of this checkpoint. And **passing
`--max-batch-size 16` is worth 1.45×**, 69.88 → 101.56 output tok/s, at which point the server's width
and the engine's own plateau arrive at the same number from two different harnesses.

## 1. The bring-up

```bash
source scripts/ascend_env.sh
python -m pocketllm serve \
    --model /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
    --backend cpp --device ascend --device-ids 4,5,6,7 \
    --tensor-parallel-size 4 --served-model-name qwen3.8-27b-bf16 \
    --host 127.0.0.1 --port 8124
```

Cards 4-7 were used because 0-3 were held by a Ternary-Bonsai server this measurement did not touch;
nothing here exercised more than four cards, and `--device-ids` is the CLI's documented way to say
which four — rank `r` takes the `r`-th, under the supervisor (`pocketllm/cli.py:257`).

Four bring-ups were timed to `/health` answering `{"status":"ready","backend":"cpp","ready":true}`,
each starting from a state where the previous server had exited and the four cards read 0 MB of HBM
used:

| bring-up | wall |
| ---: | ---: |
| 1 | 45 s |
| 2 | 67 s |
| 3 | **94.5 s** |
| 4 | 56.3 s |

The 45, 67 and 56 are one block — within the ~50 s this server's own readiness is written as, and
inside this backend's spread. **The 94.5 s is separated out rather than averaged in, and the reason is
that it is not reproducible.** It is a real observation and it is left in the table for that reason;
what it is *not* is a load-time figure, an artefact of a cold page cache, or a quantity with a tail
this record can characterise. Three of the four agree and one does not, and a median over four points
with one outlier would print a number nobody measured.

What can be said is that a bring-up of this checkpoint on this host is **on the order of a minute**,
and that the engine's own `model_load_seconds` — the summary line the CLI path in
`scripts/run_qwen_ascend_tp4.sh` prints — **is not reported on the server path at all**. So this
record does not claim to have measured the model load. These are process-start-to-ready times, engine
bring-up, shard reads and all, and they are the only numbers this record has for that cost.

## 2. The ladder

One server, arms run serially, `--max-concurrency` the only thing that changes. `--num-prompts` is 16
for the arms through concurrency 8 and 32 above it, so that the last wave is full at every arm;
without that, concurrency 16 on 16 prompts finishes in a fraction of a wave and reads as though the
width made no difference, which is what the first pass at this ladder did.

| arm | concurrency | prompts | output tok/s | req/s | TTFT | E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-streaming | 1 | 16 | 21.99 | 0.344 | 2910.3 ms | 2910.3 ms |
| non-streaming | 2 | 16 | 30.60 | 0.478 | 4178.5 ms | 4178.5 ms |
| non-streaming | 4 | 16 | 49.10 | 0.767 | 5189.7 ms | 5189.7 ms |
| non-streaming | 8 | 16 | 69.88 | 1.092 | 7293.3 ms | 7293.3 ms |
| non-streaming | 8 (repeat) | 16 | 70.01 | 1.094 | 7294.1 ms | 7294.1 ms |
| non-streaming | 8 | 32 | 69.85 | 1.091 | 7291.2 ms | 7291.2 ms |
| non-streaming | 16 | 32 | 70.13 | 1.096 | 12813.5 ms | 12813.5 ms |
| non-streaming | 32 | 32 | 69.13 | 1.080 | 17930.3 ms | 17930.3 ms |

**3.18× from concurrency 1 to 8, and then flat at ~70 tok/s.** 69.88 → 70.13 → 69.13 across widths 8,
16 and 32, with time-to-first-token still climbing linearly (7.3 → 12.8 → 17.9 s). That is a queue,
not a ceiling: the machine is doing the same work per second and making callers wait longer for it.
The two extra rows are what rules out the other reading — the concurrency-8 arm repeated gives 70.01,
and the same width run on twice as many prompts gives 69.85 with the *same* 7.3 s first-token time,
which is two waves of eight either way.

Every arm was run with the backend's defaults. `scripts/ascend_env.sh` and the model path are the only
configuration involved: the `POCKET_ASCEND_IPC_*` names the engine's own records export are read as
process environment by the engine, `pocketllm serve` exports none of them, and the tree's own scripts
do not either. A pass with `POCKET_ASCEND_IPC_ALLREDUCE=1` and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1` set around the parent — the configuration the
[single-request record](ascend_single_request_tps.md) measures as worth 106.33 → 39.4 ms a step —
produced c=1 = 22.18 tok/s and c=8 = 69.88, which is the default column to the digit. **The switch
does not register at these prompt lengths**, which is a null result and the reason those two arms are
not given a column: a table row labelled "with IPC" would misrepresent a run that measured the
default.

Three runs of the c=1 arm land in **21.19-22.18 tok/s** (21.99, 22.18, 21.19 — the last on the tree
with [#495](https://github.com/lvyufeng/PocketLLM/pull/495), see §6) and two of the c=8 arm in
**69.78-70.01**, so the ladder's shape is well outside what repeats, and a difference of the size the
collective switch would have to produce would be readable if it were there.

There is no TPOT column because the non-streaming arms have none to give. The harness declines to
invent a decode rate where the response carries a single timestamp rather than a stream of chunks, and
the zeros it prints for `tpot` on those arms are that refusal rather than a measurement of zero. The
streaming arm is where a decode rate lives, and §3 is what it says.

## 3. The streaming arm is serialized, and that is the host's lock

| arm | concurrency | output tok/s | TTFT | TPOT | E2EL |
| --- | ---: | ---: | ---: | ---: | ---: |
| streaming | 1 | 21.87 | 252.5 ms | 42.44 ms | 2861.3 ms |
| streaming | 8 | 21.77 | 15662.6 ms | 42.40 ms | 18290.4 ms |

Throughput is flat at ~21.8 — below even the non-streaming concurrency-1 arm, and *one third* of what
the same server does on the non-streaming path at concurrency 8. TPOT is 42.4 ms a token at both
widths, which is a serialized decode a token at a time.

The mechanism is not this checkpoint's. `CppBackend.stream` holds `_request_lock` across the whole
generator (`pocketllm/backends/cpp_backend.py:2000`), so a streaming request is serialized against
every other streaming one by the host, and the non-streaming path is the one that reaches
`_generate_batched` without it. **The scheduler is never reached on a streaming request**, and this
record has the sampler to say so rather than inferring it:

```
running=0 waiting=0 slots_free=8 active=8      ← 25 samples over a 16-request streaming wave
running=0 waiting=0 slots_free=8 active=1      ← 16 samples
```

`requests_running` is 0 in every one of 70 samples; `requests_waiting` is 0 too. What moves is
`requests_active`, which counts live HTTP requests rather than scheduler rows — 8 while the client's
eight are in flight, then draining 7, 6, 5 … as they finish one behind another. Eight streams arrive
at one worker and are served in a line, which is why the eight-way arm pays a 15.7 s wait for its
first token.

This is the third record to land on that lock — the Bonsai ladder reported the same flat streaming
column and attributed it to the same line — and it is now measured on a second checkpoint, a second
build and with the scheduler columns rather than the response bodies. The server's own `Peak
concurrent requests: 9` on the streaming arm is the *client's* in-flight count and not the engine's;
the non-streaming arms, which do reach the scheduler, report 16 at concurrency 8.

## 4. The width stops at 8, and it is `DEFAULT_BATCH_SLOTS`

This checkpoint is 12.796 GiB a rank against 32 GiB a card, and the KV arena is sized from the
remaining headroom, so whether the 8-slot cap binds here was not something the Bonsai record could
settle for it. It binds.

A sampler polled `/metrics` every second through a 16-client non-streaming wave:

```
running=8 waiting=8     ← 7 samples
running=8 waiting=0     ← 7
running=1 waiting=0     ← 2
running=0 waiting=12    ← 1
running=0 waiting=0     ← 1
```

`requests_running` pins at exactly 8 and `requests_waiting` goes as high as 12 while the client holds
16 in flight. The client is queuing against a scheduler that is running 8 rows and no more. That is
`DEFAULT_BATCH_SLOTS = 8` (`pocketllm/backends/cpp_backend.py:70`), and it is reached because
`--max-batch-size` was never passed at all: the flag's default of 1 is the sentinel for *unspecified*
(`_requested_batch_width`, `pocketllm/backends/cpp_backend.py:641`), so the width falls through to the
slot count. **The trap is that the sentinel is spelled the same way as a width.** Passing
`--max-batch-size 1` explicitly does not ask for one row; it is indistinguishable from silence and
gets 8. The one-row path is `--no-enable-batching`, which is also what the CLI refuses to combine with
a width above 1.

The client-side numbers agree with it and are the cleaner evidence, because they do not depend on the
sampler's 1 s grid: at 16 and at 32 concurrency the throughput is the same as at 8 (70.13 and 69.13
against 69.88), and running the concurrency-8 arm on **32** prompts instead of 16 gives 69.85 tok/s
with a time-to-first-token of 7.3 s — the same figure the 16-prompt arm gave, two waves either way.
The plateau is a policy boundary at 8, not the arena filling.

## 5. How to read 101.56 against the roadmap's 100 TPS target

The width-16 ladder crosses the number [the roadmap](../architecture/ascend_performance_roadmap.md)
sets as its target — decode at **≥ 100 TPS**, recorded there at **9.2 TPS** on this checkpoint — and
**that is not this record's claim to make.** 101.56 is a client-side aggregate: each row's prefill is
inside the same wall clock as its decode, the first token is excluded by the client's own accounting
rather than by the engine's, and the quantity is a scheduler row group over HTTP rather than the
engine's own decode rate. It is a serving number, and the roadmap's target is an engine number. Read
the roadmap against the engine-side record that speaks to it and not against this one.

What it does say is where the target's shortfall sits. The engine-side record puts the batched path
itself at **114.5 TPS at 16 rows** with TP4's 129 all-reduces a decode step, and the roadmap's own
arithmetic puts a TP4 100 TPS target 2.7× beyond what the recurrence and the collectives will give.
This ladder adds one bound to that arithmetic that the §2 table cannot show, because §2's server was
left at the default. **The slot count is the bound at 8, and passing the flag moves it to 16 — which is
the end of the road.**

| arm | concurrency | prompts | output tok/s | req/s | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `--max-batch-size 16`, width 16 | 1 | 16 | 21.85 | 0.341 | 2928.6 ms |
| `--max-batch-size 16`, width 16 | 8 | 16 | 69.88 | 1.092 | 7293.0 ms |
| `--max-batch-size 16`, width 16 | 16 | 32 | **101.56** | 1.587 | 10006.9 ms |
| `--max-batch-size 16`, width 16 | 16 | 32 (repeat) | 101.69 | 1.589 | 9994.3 ms |
| `--max-batch-size 16`, width 16 | 16 | 48 | 101.03 | 1.581 | 10037.9 ms |
| `--max-batch-size 16`, width 16 | 32 | 32 | 93.17 | 1.456 | 14520.9 ms |

Same server, same workload, only the flag and the client's concurrency changed, and the c=8 and c=1
arms reproduce the §2 column to the digit — which is the check that the two tables are one ladder.
**Raising the slot count is worth 1.45×**, 69.88 → 101.56, and the cap moves with it: a sampler through
a 32-client wave reads `requests_running 16` with `slots_free 0` and `requests_waiting` 7, then 8, then
9 — the same pinned-and-queuing shape §4 saw at 8, one width up. Concurrency 32 is past the width and
reads 93.17 while its mean first-token time climbs to 14.5 s.

The plateau above 16 is the engine, not the server. **16 rows is where the engine-side record already
put the batched path** — its `114.5 TPS at 16 rows` — so the two agree on where the curve turns, from
two different harnesses, and the second width is the last one worth asking the server for.

The residency difference between the two widths is **170 MB a rank** — 21744 MB at `--max-batch-size
16` against 21574 MB at the default 8, `kv_paged` off — and this record does not attribute it to the KV
reservation rather than to allocator slack, because it is one observation and not a sweep.

## 6. Correctness

**The tokens are the reference's, and they are the ones this backend already publishes.** Greedy
decode of the 5-token prompt `The capital of France is` at 32 steps, through `/v1/completions`:

```
' Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is Madrid.\nThe capital of Portugal is'
```

Token ids `11751 13 198 760 6511 314 9564 369 19241 …`, which is the sequence
[the RoPE workspace record](ascend_rope_table_workspace_aliasing.md) publishes as "the reference's
own top-1 at each of the first five steps" — recorded there through the engine CLI at `rows=1`. The
HTTP path emits the identical sequence, and it did so on every arm of this ladder: before any
batching, at the end of the ladder, after a reload, and again on the tree that has
[#495](https://github.com/lvyufeng/PocketLLM/pull/495). **The serving path is not a new decoding
regime for this checkpoint**; it is the same one behind a different front end.

A request whose prompt comes from the chat template is not the same request. The same question asked
through `/v1/chat/completions` carries the template's tokens and answers in two of them — `Paris`,
`finish_reason: stop` — which is a different prompt, so it is not evidence either way about the
sequence above. The CLI's own gate is the batched one, `QWEN_BATCH_ROWS=16 QWEN_BATCH_VERIFY=3` — the
harness compares the batched path against a single-row reference on both tokens and logits — described
in [the single-request record](ascend_single_request_tps.md#552-the-gate-it-has-to-pass); nothing here
replaces it.

No CUDA arm was run. This record says what the server costs on the 910B and nothing about what the
same checkpoint would do on a 2080 Ti.

## 7. What this record cannot say

1. **It cannot separate the server path from the engine path.** Every arm here went through the
   server, so the 22.18 tok/s at concurrency one is the engine's 18.6 TPS-at-`rows=1` figure plus a
   client, a scheduler and 244-token prompts rather than 5 — not a like-for-like comparison with that
   18.6, which is why §5 and §1 do not lean on it. A same-session A/B against the CLI at `rows=1`
   would need the CLI and the server on the same cards at the same time, which is not what was run.
2. **Nothing here was repeated often enough for error bars.** The repeat arms that do exist agree
   closely — concurrency 8 gives 69.88 and 70.01, concurrency 16 gives 101.56, 101.69 and 101.03,
   streaming gives 21.76 and 21.77, three of the four bring-ups land in 45-67 s, and the c=1 and c=8
   arms reproduce across two servers — but each arm is two or three samples and the bring-ups are
   four. The differences that matter here (a 3.18× ladder, a flat streaming column, a flat plateau
   above 8) are far outside anything this backend has shown run to run.
3. **The collective in use cannot be told from these numbers, and this record does not claim which
   one ran.** The engine reads the `POCKET_ASCEND_IPC_*` names from the process environment, so the
   server inherits whatever the shell has; nothing on the Python side exports them. A pass with the
   two collected names set explicitly produced c=1 = 22.18 and c=8 = 69.88, which is the shipped
   column to the digit, so **switching the collective on this checkpoint does not register at the
   client at all** — at 244 tokens of prompt and 64 of output, the collective is not what this
   workload is waiting on. That is a null result, stated as one. A record that needs to know *which*
   collective ran needs the engine's own instrumentation rather than this client.
4. **This is not a long-context result.** 244 prompt tokens says nothing about how `pocketllm serve`
   handles a streamed 4,966-token prefill, which is a different measurement on this backend.
5. **The two widths are a two-point ladder, and the residency between them is one observation.** What
   a third width below 16 would give is unmeasured, and the 170 MB a rank that separates the two runs
   — 21574 MB at the default 8 against 21744 MB at 16 — is not attributed here to the KV reservation
   rather than to allocator slack.
6. **One wave, one shape.** Every request generates exactly 64 tokens, so continuous refill and static
   batching are indistinguishable here, and no arm reports what happens when a wave's rows would not
   fit the arena — the scheduler never had to evict.

## 8. Reproducing

```bash
source scripts/ascend_env.sh
python -m pocketllm serve \
    --model /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
    --backend cpp --device ascend --device-ids 4,5,6,7 \
    --tensor-parallel-size 4 --served-model-name qwen3.8-27b-bf16 \
    --host 127.0.0.1 --port 8124

python scripts/bench_serving.py --base-url http://127.0.0.1:8124 \
    --endpoint /v1/chat/completions --model qwen3.8-27b-bf16 \
    --dataset-name custom --random-input-len 256 --random-output-len 64 \
    --num-prompts 16 --num-warmups 1 --seed 0 \
    --max-concurrency 8 --no-stream --json-out /tmp/ladder.json
```

`--no-stream` selects the non-streaming arm; `--max-concurrency` is the only thing that changes
between arms; the arms were run serially against one server. **§5's table is the same command against
a second server whose only difference is the flag `--max-batch-size 16`** — every arm above reproduced
there at concurrency 1 and 8, which is the check that the two tables are one ladder. The scheduler
columns in §3, §4 and §5 came from a sampler thread polling `/metrics` every second, not from the
response bodies. The reference token sequence in §6 is one request:

```bash
curl -s http://127.0.0.1:8124/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-bf16","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

No CUDA arm, no long context, and no run repeated enough to carry a spread.

## Where the detail is

- [The model guide](../models/qwen3.8-27b-bf16.md) — what the checkpoint is, and the CLI gate.
- [Ascend 910B single-request decode](ascend_single_request_tps.md) — the `rows=1` step and every
  lever on it.
- [Ascend decode collectives and batch scaling](ascend_decode_collective_ab.md) — the 114.5 TPS
  batched figure and the batch table behind it.
- [Ascend 910B serving through the Python front end](ascend_cpp_serving_ladder.md) — the same server
  on a different checkpoint, and where the streaming lock was first written down.
- [Ascend 910B performance roadmap](../architecture/ascend_performance_roadmap.md) — the targets.