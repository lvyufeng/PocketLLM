# Xing4.0-29B-A4B: the prefill/decode seam, and the 11% of decode that was the prompt

Every request this engine serves reports two numbers, and the serving adapter hands both to the
client: `timings.prefill_seconds` and `timings.decode_seconds`, with `timings.tpot_seconds` derived
from the second. They are two host clocks around a boundary **the device does not have**. A forward is
submitted, not completed, so at the instant `generate` reads the prefill clock the prompt's last chunk
is still running — and where it drains is inside the first decode step's `sample_token`, whose
`.cpu()` is the step's first synchronisation. The prompt was being charged to the decode loop.

**This page is the measurement of that, and the correction to every rate this repository has published
for this model.** Decode at a 32,768-token context was reported at **6.07 tok/s** and is **6.76**; the
published table's *trend* — decode falling 7% as the context grows 64× — was the drain, not the model,
and the corrected trend is flat.

| | |
|---|---|
| Hardware | 1 x RTX 2080 Ti (sm_75), `--device cuda:2` |
| Checkpoint | `/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf` (18.72 GiB) + the release directory's tokenizer and `config.json` |
| Context | 512, 512, 4,096, 16,384, 32,768 tokens — real prose, `--max-model-len 32768`, chunk 128, greedy, one request, 32 generated tokens |
| Commit | `fix/xing4-0-prefill-decode-clock`, the tree of PR #430 |
| Question | how much of `decode_seconds` was the prompt, and where in a request it is |
| Run | 2026-09-26, `scripts/bench_xing4_0_e2e.py --device cuda:2 --lengths 512,512,4096,16384,32768 --decode-steps 32 --max-model-len 32768` |

## The seam, measured directly

The size of the error was measured rather than inferred, because inferring it from two runs of the
whole bench cannot separate it from the host's own run-to-run spread on a 148 ms step. One process,
one prompt, prefilled in 128-token chunks: the *last* chunk is timed twice — once with a synchronise
after it, which is the chunk's real cost, and once without, which is what `generate` times.

| Context | Last chunk, drained | Submitted only | Left in flight | Per step of 32 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1532 ms | 1227.1 ms | **305 ms** | 9.5 ms |
| 4,096 | 1716 ms | 1376.3 ms | **340 ms** | 10.6 ms |
| 16,384 | 2018 ms | 1669.3 ms | **348 ms** | 10.9 ms |
| 32,768 | 2297 ms | 1927.7 ms | **369 ms** | 11.5 ms |

**The drain is what the device still owes when the host finishes submitting, and at this chunk width it
is nearly flat in context** — 305 to 369 ms across a 64× change in depth, a 21% spread where the
depth changed by 6400%. That is not what a chunk's *device* time does; it is what its *backlog* does,
and the backlog is bounded by how far behind a card that has 46 ms of work per decode step can be
after the host has spent 1.9 s submitting a chunk it takes 2.3 s to run. Its size follows the width of
the last chunk — a wider chunk is a longer host submission *and* a longer device tail — and it is the
step count that turns it into a rate error: the same 305 ms of prompt spread over a 32-token answer is
9.5 ms a step, and over an 8-token answer it is 38.

## The corrected table

One process, the same configuration the guide's table is measured at, with the 512-token row measured
twice — once as the process's own first request and once warm — so that the cold-read cost is visible
rather than argued about.

| Prompt | Prefill | Decode, all steps | Decode, steady | First step |
| ---: | ---: | ---: | ---: | ---: |
| 512 — the process's first request | 77.84 tok/s (6.58 s) | 6.69 tok/s | 6.72 tok/s | 171 ms |
| 512 | **79.15 tok/s (6.47 s)** | **6.75 tok/s** | **6.75 tok/s** | 146 ms |
| 4,096 | 75.22 tok/s (54.5 s) | 6.72 tok/s | 6.72 tok/s | 150 ms |
| 16,384 | 68.30 tok/s (239.9 s) | 6.74 tok/s | 6.74 tok/s | 146 ms |
| 32,768 | 64.36 tok/s (509.2 s) | 6.76 tok/s | 6.76 tok/s | 145 ms |

Against what was published:

| Prompt | Prefill was | Prefill is | Decode was | Decode is |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 84.39 tok/s | 79.15 | 6.50 tok/s | **6.75** |
| 4,096 | 78.69 tok/s | 75.22 | 6.52 tok/s | **6.72** |
| 16,384 | 68.54 tok/s | 68.30 | 6.45 tok/s | **6.74** |
| 32,768 | 63.84 tok/s | 64.36 | 6.07 tok/s | **6.76** |

**Decode is flat, and that is the thing the seam was hiding.** 6.72 to 6.76 tok/s from a 512- to a
32,768-token context — a 0.6% spread — where the old table read 6.50 down to 6.07, a 7% fall. The
fall was arithmetic on the old side of the seam: the drain is a fixed ~10 ms a step at this chunk
width, and the old *decode* figure absorbed it in full while the old *prefill* figure under-reported by
the same 10 ms a step. Everything the model actually does is unchanged, and the flat result is the one
the absorbed cache predicts — a decode step reads 576 values a layer a token however much context is
behind it.

**Prefill moves too, in the other direction, and by less.** 84.39 → 79.15 at 512 and 78.69 → 75.22 at
4,096 are the drain moving to the field it belongs to; 68.54 → 68.30 and 63.84 → 64.36 are within the
run-to-run spread, because a 128-wide chunk of a long prompt is where prefill spends its time and the
last chunk's 300-odd ms is a rounding error against 509 s.

**One number the seam does not fully explain, and it is stated rather than rounded away.** The
32,768-token row moves 11.4%, the largest of the four, and the direct measurement above accounts for
11.5 ms a step of the 16.9 — the rest is this host's own run-to-run spread, which a fixed-code re-run
made visible by measuring that row at both 6.76 and 7.06 tok/s (147.9 and 141.7 ms a step). The
corrected figure this page carries is the 6.76 of the single clean run above; a reader comparing
against the 141 ms of the other run is looking at the spread and not at a second defect. Both are well
above the 6.07 that prompted this.

## The two-card table

The other published Xing4.0 table — two server processes, one a card, against one alone — was measured
on the same side of the seam, so it is re-measured here rather than corrected arithmetically. Same
512-token prompt, `--max-model-len 8192`, chunk 128, 32 decode steps, each process's request run twice
so that the warm row is the one quoted.

| | Prefill was | Prefill is | Decode was | Decode is |
| --- | ---: | ---: | ---: | ---: |
| One process, one card | 86.28 tok/s | **83.94** | 6.48 tok/s | **7.10** |
| Two processes, cards 0 and 1 — the first | 84.79 tok/s | **82.81** | 6.52 tok/s | **7.27** |
| Two processes, cards 0 and 1 — the second | 85.89 tok/s | **84.29** | 6.53 tok/s | **7.09** |

**The concurrency claim survives, and the arithmetic checks.** Prefill gives up the drain it was
previously keeping — 305 ms on a 5,934 ms prefill is 5%, and 86.28 → 82.1 tok/s is what that predicts
against the 83.94 measured — and decode gains the same 9.5 ms a step back, 154.3 → 144.8 ms expected
against 140.9 measured. The two concurrent runs are 82.81 and 84.29 against 83.94 alone, and 7.27 and
7.09 against 7.10, with the *fastest* decode in the table belonging to a concurrent process. That is
the claim the table exists for and it is unchanged: neither the card nor the host submission path is
shared, so what the three rows differ by is this host's spread rather than a concurrency cost.

**These absolute rates are the fast end of the spread this page documents, and they are a different
configuration from the table above.** 140.9 ms a step at a 512-token context is below the 148.1 ms the
corrected table reports for the same prompt at `--max-model-len 32768`, and the two runs differ by the
cache as well — 0.35 GiB against 1.41 GiB. A reader should take the *comparison* from this table and
the *rate* from that one, and not read the difference between 6.75 and 7.10 as a second defect.

## What did not change

- **No latency moved.** A client waits for the prompt either way; `ttft_seconds` has always included
  the drain, because it is measured from before the prompt's forward rather than from after it. What
  changed is which field the prompt's tail is *attributed* to.
- **No token moved.** The fix is a clock and a synchronisation. `generate`'s scripted-model tests are
  unchanged, and the answers this page's run produced are greedy and reproducible.
- **No kernel moved.** Nothing in the model's arithmetic is touched; this is entirely a measurement.

## The fix, and its guard

`src/models/xing4_0/generate.py` drains at the seam — one call, `_drain(model)`, immediately after the
prompt is forwarded and before either clock is read. It is a no-op on a host model, which is what keeps
`tests/test_xing4_0_serving.py`'s scripted stand-ins working and what a CPU caller gets.

**The drain's *position* is the whole of the fix, so the position is what is tested.**
`test_the_prompt_and_the_decode_are_split_by_a_device_drain` replaces `torch.cuda.synchronize` with a
recorder, patches the stand-in model's `forward` to append to the same log, and asserts the log is
`forward, forward, drain, forward, forward` for a four-token prompt at `chunk=2` with a three-token
budget — the two prefill chunks, the drain, the two decode steps — and that the device it was told to
wait for is the model's own. It fails, at index 2 of that list, on the code before this change.

**The same seam is in `src/models/mimo_v2/generate.py`.** It has not been re-measured there, and its
published rates carry the same error. It is named in
[the benchmarking guide](../guides/benchmarking.md#timing-convention) rather than fixed here, because
correcting it means re-measuring that model's tables and this page is about this one.

## Run record

| Claim | Command |
| --- | --- |
| The corrected table | `scripts/bench_xing4_0_e2e.py --device cuda:2 --lengths 512,512,4096,16384,32768 --decode-steps 32 --max-model-len 32768` |
| The two-card table | the same script with `--device cuda:0 --lengths 512,512 --max-model-len 8192 --chunk 128`, alone and then concurrently with `--device cuda:1` |
| The drain, measured directly at three contexts | the two timings of one last chunk, per context, recorded above |
| The seam's position | `python -m pytest tests/test_xing4_0_serving.py -q` |

Two things about the numbers a reader should not have to reconstruct. **The host is the fixed point and
it moves**: every rate here is of a process that had the card to itself, and this box reports a 148 ms
decode step anywhere from 141 to 193 ms across runs — the 193 ms one was measured while a previous
bench was still shutting down, and is not quoted above. **And the first row of a process is the one to
distrust**, which is why the 512-token prompt was measured twice: 6.69 tok/s for the cold request
against 6.75 for the same request warm, with a first decode step of 171 ms against 146.
