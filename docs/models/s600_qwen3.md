# Qwen3 on the RDK S600 (Horizon Nash BPU)

This page is the durable record of what this tree can actually run on the **RDK S600** board and how
to run it: the command, the two environment variables it cannot start without, the measured
performance ladder, the memory mode the board has to boot in to reach its large models, and the
determinism contract. Every
number and every failure below was measured on the board — first on 2026-10-09, then again on
2026-10-10 **after the `balanced` memory switch and its reboot**, when all four Qwen3 sizes became
runnable, and the `w8` rebuilds — under the `bpu_first` pool — once more on 2026-10-13; whether either
transfers to another S600 is a claim to re-check in place, and the SDK version is part of the result.

Two facts frame the rest, and both are easy to get wrong:

- The S600 is **not the C engine's target.** There is no CUDA, no compiled `libpocketllm.so` for
  aarch64 here, and `--device horizon` does not select a [Python backend](../architecture/devices.md)
  either — those are stubs. It selects the **`xlm` delegate**, an `EngineBackend` adapter
  (`python/pocketllm/server/xlm_backend.py`) over D-Robotics' `libxlm.so`, which runs a
  prebuilt `.hbm` graph on the BPU. See [Serving](../architecture/serving.md) for the adapter
  contract and [#571](https://github.com/lvyufeng/PocketLLM/pull/571)/[#572](https://github.com/lvyufeng/PocketLLM/pull/572)
  for how `run` and `serve` reach it.
- **The delegate is text-in / text-out.** It tokenizes, applies its own chat template, decodes on the
  BPU, and hands text back. There is no logits surface and no token-id surface (the SDK's
  `XLM_INPUT_TOKEN` path is "not support yet"), which is why the sampling rules in
  [Sampling and caps](#sampling-and-caps-the-cli-enforces) are what they are: there is nothing on
  this path to sample from.

## The board and what it runs

RDK S600 — Horizon **Nash** BPU, aarch64, **4 BPU cores**, Linux 6.1.158-rt, Python 3.12.3. The
D-Robotics LLM SDK (`D-Robotics_LLM_S600_1.0.2_SDK`) ships the runtime (`oellm_runtime/`), a
compiler tree (`oellm_build/`), tokenizer + `generation_config.json` under
`oellm_runtime/configs/Qwen3_config/`, and per-size `.hbm` graphs under `oellm_runtime/model/Qwen3_*/`.

**All four shipped Qwen3 sizes load and generate on this board's raised pool** — in `bpu_first`
(18.00 GiB, applied 2026-10-13) as it stands, and in `balanced` (10.00 GiB) before that; the board boots
`cpu_first`, where the two large ones are refused. See [the memory mode](#the-4b-8b-ceiling-applied):

| Size | `.hbm` | Precision | File size | Loads? |
|---|---|---|---|---|
| Qwen3-0.6B | `Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm` | w8 | 1.02 GiB | **yes** |
| Qwen3-1.7B | `Qwen3-1.7B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 1.70 GiB | **yes** |
| Qwen3-4B | `Qwen3-4B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 3.10 GiB | **yes, in `balanced` / `bpu_first`** |
| Qwen3-8B | `Qwen3-8B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 5.31 GiB | **yes, in `balanced` / `bpu_first`** |

The naming is the compiler's: `chunk_512` is the prefill chunk, `cache_4096` the max context,
`corenum_4_4` the BPU-core assignment. Every large model in the SDK ships as `corenum_4_4`; **there
is no smaller-corenum 4B/8B variant to fall back to.**

## Running it: the exact contract

Two environment variables must be set **before `pocketllm run` starts**, and the CLI requires them
rather than setting them:

```bash
SDK=~/llm_sdk/D-Robotics_LLM_S600_1.0.2_SDK

export LD_LIBRARY_PATH=$SDK/oellm_runtime/lib:$LD_LIBRARY_PATH   # read by dlopen at process start
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6          # the L2m split the .hbm was compiled for

PYTHONPATH=python python -m pocketllm run \
    --model $SDK/oellm_runtime/examples/llm_demo/qwen3_1.7b_config.json \
    --device horizon \
    --prompt "The capital of France is"
```

`--model` accepts either an `.hbm` path or the SDK's demo-style JSON config (the `*_config.json`
above); the config names the `.hbm`, the tokenizer directory, `bpu_core` and `model_type`
(Qwen3 = `9`). `--tokenizer-path` / `--config-path` override what the JSON says. `run` and `serve`
resolve `--model` through the **same** function (`_resolve_model`), so both accept the same spelling.

**Both variables are required, and the failure is only loud if we make it so.**
`LD_LIBRARY_PATH` must point at **`$SDK/oellm_runtime/lib`** — the SDK's libraries (both
`libxlm.so` and `libopencv_world.so.409`) live under `oellm_runtime/`, and there is **no** `$SDK/lib`
directory. A path guessed as `$SDK/lib` is not an error on its own; it fails later as the confusing
missing-`libopencv_world.so.409` this paragraph used to blame, which is why the correct directory is
named here and in the block above. `LD_LIBRARY_PATH` is read by `dlopen` **before the process
starts** — exporting it from inside Python is a no-op, so "set it for the user" cannot fix it.
`HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` is the L2m split the graph was compiled with, and the SDK's
own `run_llm.sh` sets exactly `6:6:6:6` for every model. `_require_delegate_env()` fails with a
message naming the missing variable(s) before touching a 1 GiB `.hbm`; it does not silently default.

### Reading it in `pocketllm devices`

`pocketllm devices` lists the **Python** backends and their stubs, so on the board it prints

```text
horizon    horizon   missing the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device  aot_compile       14 ops
```

and that line says nothing about whether the delegate runs — the delegate is reached through `xlm`,
not through the ABI backend of the same name. The `horizon` kind is what `--device horizon` selects;
`aot_compile` there is the *unimplemented* AOT path, not the `.hbm` delegate.

## Measured ladder

**This is the canonical ladder: all four shipped Qwen3 sizes plus our own compiled graphs — the 0.6B and
4B `cache_1024` builds, and the `w8` rebuilds of the 1.7B, 4B and 8B.** Every row but the last was
measured on the `balanced` pool in one session on 2026-10-10 (the compiled 0.6B was added the same day,
once its build landed); the `w8` rebuild rows were added later, measured **2026-10-13** by the same
method and the same scripts, and the 8B `w8` row — the largest footprint here — is the one row measured
on the `bpu_first` pool applied that day ([why the pool moved](#the-4b-8b-ceiling-applied)). The `pool`
column names the pool each row's `of pool` percentage is against. Fixed prompt **"The capital of France
is"**, greedy `generation_config.json`, canonical-prompt three decode runs per graph, one process per
graph. Throughput and load time from the delegate's `last_performance()` (`xlm_model_performance_t`);
memory held from the kernel's own ION accounting (`/sys/kernel/debug/ion/heaps/all_heap_info`), read
**after the load completes and before the first decode** — the load-time peak, the same instant
[the margin section](#the-8b-sits-comfortably-in-balanced-the-numbers) measures, so the two tables
agree. (A decode adds ~152 MiB of KV/scratch on top; that is per-request work-in-flight, not what the
model costs in the pool, and it is excluded here deliberately.) The memory column is what each model
costs in its row's `ion_carveout`.

| Size | `.hbm` | Load | Prefill | Decode | Memory held at load (carveout) | pool | of pool |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (w8) | 1.02 GiB | 5.26 s | 6169–7111 t/s | **88.3 t/s** (88.09 / 88.73 / 88.28) | 2,995,585,024 B (2.79 GiB) | `balanced` | 27.9% |
| Qwen3-1.7B (w4) | 1.70 GiB | 5.70 s | 5172–5818 t/s | **70.2 t/s** (70.18 / 70.13 / 70.16) | 3,734,503,424 B (3.48 GiB) | `balanced` | 34.8% |
| Qwen3-4B (w4) | 3.10 GiB | 6.44 s | 2415–2573 t/s | **42.8 t/s** (42.77 / 42.72 / 42.78) | 6,686,113,792 B (6.23 GiB) | `balanced` | 62.3% |
| Qwen3-8B (w4) | 5.31 GiB | 7.66 s | 1939–2040 t/s | **29.7 t/s** (29.65 / 29.66 / 29.70) | 9,417,129,984 B (8.77 GiB) | `balanced` | **87.7%** |
| Qwen3-0.6B (w8), ours `cache_1024` | 0.98 GiB | 5.31 s | 11636–13128 t/s | **121.6 t/s** (122.18 / 121.72 / 120.82) | 1,702,756,352 B (1.59 GiB) | `balanced` | 15.9% |
| Qwen3-1.7B (w8), ours `cache_1024` | 2.31 GiB | 6.05 s | 8127–9143 t/s | **65.4 t/s** (65.49 / 65.31 / 65.31) | 3,141,009,408 B (2.93 GiB) | `balanced` | 29.3% |
| Qwen3-4B (w4), ours `cache_1024` | 3.00 GiB | 6.35 s | 4339–4571 t/s | **49.9 t/s** (49.99 / 49.96 / 49.86) | 4,254,269,440 B (3.96 GiB) | `balanced` | 39.6% |
| Qwen3-4B (w8), ours `cache_1024` | 4.69 GiB | 7.34 s | 4163–4491 t/s | **33.1 t/s** (33.07 / 33.10 / 33.04) | 6,073,548,800 B (5.66 GiB) | `balanced` | 56.6% |
| Qwen3-8B (w8), ours `cache_1024` | 8.46 GiB | 9.41 s | 2909–3030 t/s | **20.5 t/s** (20.45–20.49) | 10,472,783,872 B (9.75 GiB) | **`bpu_first`** | **54.2%** |

Decode falls with size — 88.3 → 70.2 → 42.8 → 29.7 t/s on the shipped graphs — and the memory column
rises with it, which is the whole of the ladder's shape: the BPU's per-token cost and the pool's
footprint both track the parameter count. Our own `w4` `cache_1024` builds sit *above* their shipped
twins on both axes (faster, and smaller in the pool), which is the subject of the note on them below —
the `w8` rebuilds keep the pool edge but not the rate one, and their own note below covers the bill.
Every graph's three runs produced **byte-identical text**, so each number is a rate and not a draw —
the 8B `w8` row included, **nine runs across three loads with a single digest**
(sha256 `76025885…`). The 8B is where that determinism has measured exceptions, which is why the new
row is quoted as nine runs: our 8B `w4` `cache_4096` build, re-run under `bpu_first` on this same
canonical prompt, gave three **non**-identical runs (2026-10-13), and the survey above records the
shipped 8B diverging once in four `1..200` runs
([the caveat](#the-corrupted-token-is-the-build-not-the-cache)).

**The switch did not regress 0.6B or 1.7B.** The board changed memory *mode*
(`cpu_first` → `balanced`), and the two small graphs were re-measured against the numbers this page
carried from before it: both produce **the same greedy output, byte for byte** (0.6B sha256
`25a32998bfab7e7e`, 1.7B `6bcca969667a395a`, banner stripped and the `run` prompt echo removed before
hashing, so the comparison is text against text). The decode figures moved slightly *upward* (0.6B
87.0 → 88.3, 1.7B 69.4 → 70.2 t/s), which is run-to-run spread and a warmer machine rather than a
mode effect — the pool is bigger, not different, and nothing about a graph's own allocation changed.
That expectation is now a measurement instead of an assumption.

**TTFT is not available on this SDK build.** The `ttft`, `tpot` and `end_to_end_cost` fields of
`xlm_model_performance_t` come back `0.0`, so a time-to-first-token is not something this page can
quote from the runtime; wall-clock load time above is quoted instead. The spread across three runs is
under 0.8% at every size, so decode is stable rather than a lucky run.

**A few rows deserve their own note.**

*The 8B costs 87.7% of the pool at load.* That is the largest model this board runs, and it leaves
**1,320,288,256 B (1.23 GiB) free — 12.3%** — the figure
[the margin section](#the-8b-sits-comfortably-in-balanced-the-numbers) derives; the two tables are the
same measurement at the same instant, and their numbers agree to the byte. Roughly 6.5 GiB of the pool
is available to a `.hbm` file once the load's fixed overhead is accounted for, which is an 8B–9B class
model and not a 14B.

*A `w4` graph we compile ourselves is faster.* Our own `Qwen3-1.7B` build at `cache_1024` (1.66 GiB, md5
`0b41e627f2227c029b14ed63928fd33f`) also loads and runs: decode **88 t/s** (88.84 / 87.90 / 87.59) and
prefill 8982 t/s, against the shipped `cache_4096` graph's ~70 t/s / ~5.2k t/s in the same session —
which is why the 1.7B row above is a *shipped-graph* number and not a 1.7B number. It is the first
artifact out of [the native compile chain](../architecture/s600_native_chain.md#the-first-build-result),
it answers the same prompts with the same final answers, and its **reasoning text differs** from the
shipped graph's — two build-time quantizations of one checkpoint, not two caches of one graph. Our
`Qwen3-0.6B` and `Qwen3-4B` at the same `cache_1024` do likewise, each with a row in the ladder above;
the smallest size carries the biggest decode edge of the three and has
[a section of its own](#our-compiled-06b-vs-the-shipped-06b). Whether a smaller graph gets **4B** under
the `cpu_first` ceiling is answered — it does not, and the pool had to move instead; see
[the large models after the switch](#the-large-models-after-the-switch).

**The `w8` rows are the accuracy rebuilds, and the rate is what they pay.** The 1.7B, 4B and 8B `w8`
graphs exist to move agreement against truth
([the 1.7B's row](#the-17b-at-w8-above-the-floor-against-truth)), and this table is their bill: decode
**65.4 t/s** at 1.7B against the `w4` `cache_1024` graph's **89.1**, **33.1** at 4B against **49.9**,
and **20.5** at 8B against the 8B `w4` `cache_1024` build's **33.2** — 27%, 34% and 38% slower — while
the pool grows **+0.66 GiB** (2.27 → 2.93), **+1.70 GiB** (3.96 → 5.66) and **+3.23 GiB** (6.52 →
9.75). Prefill moves the same way but less (8127–9143 against 8828–9846; 4163–4491 against 4339–4571;
2909–3030 against ~3.06k at 8B), load is 6.05 s, 7.34 s and 9.41 s against 5.62, 6.35 and 7.87, and the
1.7B and 4B builds sit well inside `balanced` — the 4B `w8` leaves 4.34 GiB of the pool free. **The 8B
does not: its 9.75 GiB held at load is 97.5% of `balanced`'s 10.00 GiB, so it is the one row measured
under the raised `bpu_first` pool** — 18.00 GiB, where it holds 54.2% and leaves 8.25 GiB to spare; the
pool was moved for this build ([the ceiling section](#the-4b-8b-ceiling-applied)). Its row is the band
of nine runs across three loads, and the three loads agree to the byte: loads **9.88 / 9.41 / 9.37 s**,
decode **20.45–20.49 t/s**, prefill **2909–3030 t/s**, carveout **10,472,783,872 B in all three**. The
width is also the second single-variable escape from the 1.7B loop: it closes the block at the loop's
own cache ([the loop at `w8`](#the-loop-at-w8-the-same-cache-closes-it-on-wider-weights)).

**The 8B compiles too, and it is the lighter and faster of the pair.** Our own `Qwen3-8B` build at
`cache_1024` (5.61 GB, md5 `84a312d26361d21b0d8daf398a48651b`) loads in **7.87 s** and costs the pool
**6.52 GiB (65.2%)** — 2.25 GiB below the shipped `cache_4096` 8B's 8.77 GiB, leaving **3.48 GiB free**
where the shipped leaves 1.23. It decodes **33.2 t/s** and prefills ~3.06k t/s against the shipped's
29.7 / ~2.0k, three byte-identical runs, and holds that rate start-to-end with **no swapping**
(`MemAvailable` flat at 42.6 GiB through a cap-length generation — the earlier 8B memory ceiling was the
cache, not the weights). It answers the canonical prompt `Paris` with the usual early reasoning
divergence. **One caveat is real: a reproducible corrupted-token failure on the dominant path, plus a rarer divergence
that never answers.** Driving enumerations, the dominant `1..400` run writes 1…55 cleanly, emits
`.bunifuFlatButton` for the 56th number and resumes 57… — reproducible (sha256 `79b0efae…` in 6 of 8 runs; the other 2 diverge mid-thinking
and never answer), and the shipped graph does not make it — though neither the slip index nor the garbage
token is fixed: the token is `个人` at `1..50` and `.bunifuFlatButton` at `1..200`/`1..400`, and the index is
the **37th** number at `1..50` but the 56th at `1..400`. It is **not** position- or item-indexed either —
under one fixed prompt, `1..50` slips at 37 with `个人` while `1..100` runs **clean** (all three runs), so the
near-tie does not sit at a fixed token position; and the garbage token is **prompt-dependent** too, not only
length-dependent (`1..100` is clean under the long prompt above, yet under the shorter `List every integer
from 1 to 100, one per line.` the same graph slips at 37 with `.bunifuFlatButton`). Both facts point the same
way: it is a **state-dependent numerical near-tie
resolved wrongly** (a digit's logit and a garbage token's within a hair), the same *root* as the 1.7B loop
but a sharper expression: one token, no repetition build-up, and it resumes. `cache_1024 w8` is the build
that targets it (width is the natural lever for a numerical near-tie), but the ~25% run-to-run variance and
the prompt-varying token made it a **bet, not a guarantee** — the intermittent path points at the decode
side too, so a single clean `w8` run would not prove the slip gone. **The bet has now been drawn, and it
is settled on the enumeration as well**: the `w8` build exists, runs at **20.5 t/s** under `bpu_first`
([its ladder row](#measured-ladder)), its 50-prompt capture carries the glitch **zero** times
([the 8B at `w8`](#the-8b-at-w8-the-width-lever-falls-short)), and the `1..N` battery run at `w8`
emits the corrupted token **zero** times in twenty-seven runs across the 1.7B, 4B and 8B
([the battery at `w8`](#the-1n-battery-at-w8-the-slip-is-gone-and-the-8b-fails-a-different-way)).
The battery does find a different 8B failure at the same cache — that section carries it.

The raw per-length run is the evidence for "state-dependent, not index-fixed":

| prompt | run(s) | result | sha256 | garbage token |
|---|---|---|---|---|
| `1..50` | 3/3 identical | slip at the **37th** number | `c2f028c98846…` | `个人` |
| `1..100` | 3/3 identical | **clean** | `8405f496009c…` | — |
| `1..200` | 1, 2 of 3 | TIMEOUT (delegate hung) | — | — |
| `1..200` | 3 of 3 | slip at the **37th** number | `d678e3cf09ac…` | `.bunifuFlatButton` |
| `1..400` | 2, 3, 4, 5, 6, 8 of 8 | slip at the **56th** number | `79b0efae9a08…` | `.bunifuFlatButton` |
| `1..400` | 1, 7 of 8 | no answer (diverged mid-` thinking`, never closed it) | `a1105a5c86aa…`, `ba1257518e1d…` | — |
| `1..100` short prompt | 1 | slip at the **37th** number | `f95507349243…` | `.bunifuFlatButton` |

Prompt, unless the row says otherwise: `List every integer from 1 to N, one per line, in order. Do not
abbreviate, do not use an ellipsis, and do not add any other text.` The last row uses the shorter
`List every integer from 1 to 100, one per line.` It is the one case where a single length is **clean** under
one prompt and slips under another. Each run is its own process, greedy, and the sha256 is over the full
generated text.

#### Where the corrupted-token slip lives: a survey across sizes and both cache lengths

The `1..200` and `1..400` enumerations were then driven through **every graph on the board** — our
compiled 0.6B / 1.7B / 4B / 8B at `cache_1024`, our rebuilt 1.7B at `cache_4096`, and the shipped 1.7B
and 8B at `cache_4096` — three runs each, one process per run, the same greedy prompt and
`bpu_core [0,1,2,3]`, classified from the streamed output as **clean** (the whole `1..N` listed),
**slip** (a wrong token inside the list, which then continues), **self-stop** (EOS before reaching `N`),
**truncate** (cut at the cache cap, clean to the last emitted token), **abbreviate** (the model chose an
ellipsis), or **hang** (the lost wake-up of the section below, caught by the 25-second stall detector).
The point of the survey is to ask whether the corrupted-token signature is an 8B property or a
`cache_1024` property; the 8B's own `cache_4096` build is still compiling, so the shipped 8B at
`cache_4096` stands in for it — it no longer has to; the decisive cell is measured
[below](#the-decisive-cell-our-8b-at-the-vendors-cache-still-corrupts).

| graph | cache | `1..200` | `1..400` | outcome class |
|---|---|---|---|---|
| ours 0.6B (w8) | 1024 | truncate, clean to 152 | truncate, clean to 177 | the ~1035-token cap, no corruption |
| ours 1.7B (w4) | 1024 | abbreviates (`… 200`) | abbreviates, then cut mid-` thinking` at the cap | no corruption |
| ours 1.7B (w4) | 4096 | **clean** | **clean** | lists fully, 1..200 and 1..400 |
| ours 4B (w4) | 1024 | **slip at #24** (`2一`) | **slip at #24** (`2一`) | corrupts |
| ours 8B (w4) | 1024 | **slip at #37** (`.bunifuFlatButton`); 1 of 3 hung | **slip at #56** (`.bunifuFlatButton`); 1 of 3 self-stopped at 22 | corrupts, and hangs |
| ours 8B (w4) | 4096 | **slip, 3 of 3** (runs 1 & 3 identical: a repeated 22…24 block *and* `1.bunifuFlatButton` at #105; run 2 `2.bunifuFlatButton` at #25) | **slip at #45** (`4.bunifuFlatButton`), 3 of 3 identical | corrupts, deterministically — and earlier |
| shipped 1.7B (w4) | 4096 | clean | clean | — |
| shipped 8B (w4) | 4096 | clean | clean | — |

All three runs per cell unless the cell says otherwise; every cell that completed was byte-identical
across its runs, so each is a mode, not a draw. Our 8B at `cache_4096` keeps that rule: its `1..400`
runs are identical three of three, and its `1..200` runs slip three of three (two identical, the third
a different index) — the larger cache moved the position, not the fact.

**The corruption is a `cache_1024` property, and it is not an 8B one.** Two of the four `cache_1024`
graphs corrupt the list — the 8B, as before, and the **4B**, which does it for the first time here: at
both `1..200` and `1..400` it writes 1…23 cleanly and then `2一` for the 24th number before resuming at
23/24, deterministically (3 of 3 identical, sha256 `bb7df4fb3543…` at `1..200`). The 8B keeps its 37th-
and 56th-number `.bunifuFlatButton` slips; of its six runs one was the lost-wake-up hang
and one self-stopped after 22 numbers behind a long reasoning block, so its cells rest on two
observations, not three. Against that, the 0.6B and the 1.7B at the same `cache_1024` do **not** corrupt:
the 0.6B reaches its ~1035-token cap and truncates mid-number (already documented above), and the 1.7B
abbreviates the list outright (`1 2 3 … 200`) rather than corrupting within it. So the corrupted-token
signature tracks neither size nor the `w4` width alone — the 1.7B is `w4` too and does not show it.

**At the larger cache the corruption is gone — on every graph that has one.** Our rebuilt 1.7B at
`cache_4096` lists `1..200` and `1..400` **completely and cleanly**, three runs identical in each — the
same graph that abbreviates and then runs away at `cache_1024`. The shipped 1.7B and the shipped 8B, both
`cache_4096`, are clean at both lengths as well; the shipped 8B in particular is the sum-of-one extra
observation that the 8B, at `our build`'s width and the vendor's cache, does not corrupt. This is the
second data point for the cache as the lever the 1.7B loop was already shown to be, and the reason the
8B's own `cache_4096` build is the right next test: if it comes back clean on these enumerations, the
slip is a `cache_1024` artefact like the loop; if it slips, the lever is width or the build, and the
cache is not the whole story. **The survey cannot settle that on its own — the two `cache_4096` graphs
that are running clean are neither ours nor the corrupting sizes, and the one genuinely decisive cell
(our 8B at `cache_4096`) is the one still compiling.** What it does settle is narrower and worth stating
plainly: the corruption is not peculiar to the 8B — the 4B makes it too — and it is absent wherever the
cache is large.

#### The `1..N` battery at `w8`: the slip is gone, and the 8B fails a different way

The survey's `cache_1024` rows are all `w4`, so it says nothing about the rebuilds the width argument
predicts should clear the near-tie — and every `w8` graph on this page is `cache_1024`. The same battery
has now been run on them (2026-10-13, `bpu_first` pool, one process per run, the 25-second stall
watchdog, greedy, three runs a cell):

| graph | `1..200` | `1..400` | `France?` | byte-identical |
|---|---|---|---|---|
| ours 1.7B `w8` | truncate at **167**, ordered 1…167 | truncate at **36** | closes, 1631 chars, recurs 2× | 3/3 each cell |
| ours 4B `w8` | truncate at **42** | truncate at **83** | closes, 3030 chars, recurs 1× | 3/3 each cell |
| ours 8B `w8` | 1 truncate (a repeated `35, 36` at the 36th, then clean to 135) / 1 hang / 1 never closes | 2 never close (**recurs 5–6×**) / 1 hang | closes, 989 chars, recurs 1× | **0/3** — all three differ |

**The corrupted token does not appear at `w8` — not once, in twenty-seven runs.** No
`.bunifuFlatButton`, no `个人`, no token `95234`; and every list that reached its enumeration is strictly
ordered up to the point the window cut it — the 1.7B to 167, the 4B to 42 and to 83, the 8B to 135. The
width argument's prediction holds at both sizes where the `w4` slipped: the **4B** `w8` writes 1…42 and
1…83 with no `2一` where its `w4` sibling slipped at the 24th number, and the **8B** `w8` never emits
the garbage token its `w4` sibling emitted in six of six `cache_4096` runs. The single anomaly in the
whole battery is benign and of another kind: one 8B `1..200` run repeats the pair `35, 36` at the 36th
entry — a **duplication**, not a substitution — and the list resumes and runs on to 135.

**Width paid at the 1.7B and the 4B, and the 8B is where it stops paying.** Those two graphs are clean
*and byte-identical three of three* in every cell, the determinism the `w4` cells show. The 8B `w8` is
neither: **five of its nine runs fail** — two killed by the stall watchdog (the
[#650 lost wake-up](#the-intermittent-hang-a-lost-wake-up-inside-the-delegates-decode-path)) and three
that never close the reasoning block at all.

**The 8B's failure is a rumination, not the slip.** In those three runs the model argues with its own
instruction *inside* ` thinking` and never emits a number — *"the user's instruction says not to add any
other text, so the assistant must list all numbers"*, carried as an argument rather than a token cycle
(the most-repeated 40-character window recurs 4–6×, against 1–3× in every run that closed) — and the
window ends it mid-sentence. It is the shape the 1.7B `w4` showed at `cache_1024`
([the loop table](#the-loop-at-w8-the-same-cache-closes-it-on-wider-weights)), and unlike that one it is
**not** closed by width at this size. It is also not a mode: the three `1..200` runs are three different
outcomes, where the page's other `cache_1024` failures were byte-identical draws off one another.

**What that means for the graph the accuracy table prefers.** The 8B `w8` — the one build at this size
that [beats its `w4` siblings against truth](#the-8b-at-w8-the-width-lever-falls-short) — is **not
deployable for long generations behind a `cache_1024` window**: five of nine attempts either hang or
never answer, and its three `France?` runs are not byte-identical (one inserts a flag emoji into an
otherwise identical answer). The failure lives in the *reasoning block*, so a short greedy prompt cannot
expose it — the ladder's nine identical runs on `The capital of France is` are real and say nothing about
this. The lever the page's own data points at is the **cache**, not the width: our 8B `w4` at
`cache_4096` completes both enumerations
([the decisive cell](#the-decisive-cell-our-8b-at-the-vendors-cache-still-corrupts)) and the 1.7B sweep
closes the loop already at `2048`. One rebuild — an 8B `w8` at `cache_2048` or `4096` — would buy it;
the caveat that comes with the price is that `cache_4096` is where our 8B `w4` corrupted *most* (six of
six runs, and a shorter `1..400` prefix than at `1024`), so the cache answers the rumination while
leaving the surface it fixes untested at `w8`/`4096` in the other direction.

#### The decisive cell: our 8B at the vendor's cache still corrupts

The cell the survey could not reach — **our own** 8B graph at `cache_4096`, the exact compile that
produced the `cache_1024` corruption, rebuilt at the cache the shipped graph uses — has since landed
(`Qwen3-8B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm`, 5.71 GB, md5
`004e3470f61a84ed1640483628d24ea7`). It is the clean separation of **cache** from **build**: same
width, same model, same source recipe, the cache the only variable against our own `cache_1024` graph,
and the build the only variable against the shipped `cache_4096` graph. The same enumeration battery
was run on it, three runs a cell, one process a run, classified the same way:

| prompt | runs | result | glitch token |
|---|---|---|---|
| `1..200` long prompt | 1, 3 of 3 | slip — an exact `22,23,24` block repeated **inside** the list and `1.bunifuFlatButton` at the 105th number; the list then **completes to 200** | `.bunifuFlatButton` (id 95234) |
| `1..200` long prompt | 2 of 3 | slip at the **25th** number (`2.bunifuFlatButton`) | `.bunifuFlatButton` |
| `1..400` long prompt | 3 of 3 identical | slip at the **45th** number (`4.bunifuFlatButton`); completes to 400 | `.bunifuFlatButton` |
| `1..100` short prompt | 3 of 3 identical | **clean** | — |

**The answer is unambiguous: the corrupted token is not a `cache_1024` artefact.** Our 8B at
`cache_4096` corrupts **six of six** long-format runs — every one emits the same `bunifuFlatButton`
glitch (`4.bunifuFlatButton` for the 45th number, `2.bunifuFlatButton` for the 25th, `1.bunifuFlatButton`
for the 105th) — and the `1..400` cell is **byte-identical three of three**, the same determinism the
`cache_1024` graph shows. The cache did not remove the corruption; it moved its position. So the lever
the 1.7B loop answered to is *not* the lever for this slip, and the cause is the **build**, not the cache.

Two things sharpen the "moved, not removed" reading, and both cut against the cache:

- **The slip moved *earlier*, not later.** `1..200` slips at the 37th number at `cache_1024`, the 25th
  at `cache_4096`; `1..400` slips at the 56th at `cache_1024`, the **45th** at `cache_4096` — a
  *shorter* shared prefix on the enumeration that prompted the whole investigation (the `1..400` shared
  prefix is 45 against 56). A larger cache that made things worse is not a cache that fixed anything.
- **The failure mode is state-driven, not cache-driven.** `1..400` slips deterministically at 45 while
  `1..100` is clean at `cache_4096` — the same "length and prompt move the near-tie, the index is not
  fixed" structure documented at `cache_1024`, reproduced at the larger cache. What survives the cache
  change is the near-tie itself.

**At the larger cache our build also corrupts *differently* — and it corrupts more.** At `cache_1024`
the glitch aborts the run: the sequence stalls at 37 (or 56) and the generation degenerates into a
repeated `6` for a few hundred characters before it ends. At `cache_4096` the glitch is a **local point
substitution** — one number replaced, the model recovers, and the list **finishes** (`1..200` reaches
200; `1..400` reaches 400). That is the same one-token near-tie, now recovered from instead of fatal.
But the deeper cache also raised the count: the `1..400` shared prefix against the oracle is **45** at
`cache_4096` against **56** at `cache_1024` — the larger cache cost this graph eleven tokens of
oracle agreement on the one prompt the corruption first named.

##### Cache versus build on the accuracy table

Running `parity.py` on the graph (`--label 8B-ours-c4096`) drops our 8B at the vendor's cache into the
provenance table, and it separates the two axes for the 8B exactly as the 1.7B sweep did:

| graph | cache | shared prefix min/p25/med/p75/max | ≥5-logit splits | n |
|---|---|---|---|---|
| our 8B `w4` | 1024 | 2 / 7 / **11** / 20 / 61 | 2 | 46 |
| **our 8B `w4`** | **4096** | 2 / 9 / **13** / 20 / 61 | **4** | **48** |
| shipped 8B `w4` | 4096 | 4 / 14 / **23** / 33 / 88 | 2 | 50 |

The row moves our 8B's median **up** two tokens (11 → 13), confirming the cache is not the accuracy
lever — but the **≥5-logit splits double, 2 → 4**, matching the enumeration: the deeper cache made this
build *worse* on the exact axis that separates build from cache. Paired per prompt, against the shipped
graph at the **same cache** our build is the shorter-prefix side on **31 of 48** prompts (9 equal, 8
longer, median Δ **−7**, range −69…+38); the `88`-token shared prefix the shipped graph reaches is
still one ours never approaches (max 61 at both caches). Same build, cache `1024 → 4096`: **30 of 44
identical** to the token, median Δ **0** — the cache changes the cache, not the build.

#### The corrupted token is the build, not the cache

The survey above guessed the corruption was "a `cache_1024` property" on the strength of the 1.7B loop.
The decisive cell falsifies that guess for the 8B: the corruption survives the vendor cache because it
was never the cache that produced it. Restated with the compile chain in view: our 8B's `w4` graph is a
different quantization of the same weights than the vendor's, and that difference — not the cache — is
what puts a digit's logit and `bunifuFlatButton`'s within a hair. The 1.7B loop was a genuine cache
effect; the 8B slip is not one, and the two must not be read as the same failure. It is the build axis
the provenance table measures, showing up as a single token in the text.

**And it is confirmed for the full corpus, not just the one slip token.** The 8B cell separates cache
from build on a single prompt, too narrow to carry a statistic; the matched-cache 1.7B row above carries
it over all fifty. At the identical `cache_4096` ours trails the vendor **377 vs 724** against truth,
and lifting *our* cache from 1024 to 4096 moved nothing (Σ **394** → **377**). So the cache is not the
accuracy lever for the corpus either, and "the build, not the cache" is not a fact about one 8B
enumeration — it is the matched-cache agreement result at the size where the confound is controlled out.

**The answer to the enumeration question for the 1.7B is yes.** The `cache_4096` rebuild that cured the
1.7B's runaway loop also lists all of `1..200` and `1..400` without corruption; the enumeration stays
clean at the larger cache, so the loop and the slip do not trade places when the cache grows.

One determinism caveat the survey turned up, since it bears on how much weight a `3 of 3` carries: the
**8B's reasoning block is not strictly byte-deterministic.** Of four completed shipped-8B `1..200` runs,
three were identical and one diverged deep inside ` thinking` (char 4270 of the generation) before
recovering to the same correct list — a same-prompt, same-graph, greedy divergence at the token level,
the kind the smaller graphs did not show here. It did not change a single answer in the table above, but
it is a second axis on which the 8B is the most fragile graph in the set, alongside the slip and the hang.

### Our compiled 0.6B vs the shipped 0.6B

The recompile was extended to the size that matters most for latency, and it is the cleanest result of
the set. Our own `Qwen3-0.6B` at `cache_1024` — built with the same recipe as the 1.7B and 4B
(`--model_name qwen3 --march nash-p --w_bits 8 --chunk_size 512 --cache_len 1024 --prefill_core_num 4
--decode_core_num 4`) — is **1,052,244,472 B (0.98 GiB)**, md5 `0d9d7e87e4f658eced34f04e17f17b2d`,
sha256 prefix `75ae275557571145`. Measured against the shipped `cache_4096` graph 2026-10-10 with
[the same harness](../architecture/s600_native_chain.md#the-first-build-result) every compiled row here
uses — same prompt, greedy tokenizer directory, `bpu_core [0,1,2,3]`, three runs per side, one process
per side:

| | shipped `cache_4096` (w8) | ours `cache_1024` (w8) | ratio |
|---|---|---|---|
| file | 1,096,189,432 B (1.02 GiB) | 1,052,244,472 B (0.98 GiB) | 0.96× |
| load | 5.29 s | 5.31 s | — |
| prefill | 6169–7111 t/s | **11636–13128 t/s** | **~1.8×** |
| decode | 88.09 / 88.53 / 88.28 t/s | **122.18 / 121.72 / 120.82 t/s** | **1.38×** |
| memory held at load | 2,995,585,024 B (27.9%) | **1,702,756,352 B (15.9%)** | 0.57× |
| greedy output sha256 | `25a32998bfab7e7e…` | `59d31f4401d710e5…` | differ |

**It loads and runs** — the 0.98 GiB artifact is well inside the pool, no `RESOURCE_EXHAUSTED` — and
the answer is coherent and correct:

```text
# shipped 0.6B, cache_4096
…  </think>
The capital of France is **Paris**. It serves as the political, cultural, and economic center of the country.

# ours 0.6B, cache_1024
…  </think>
The capital of France is **Paris**.
```

**1.38× decode and ~1.8× prefill — the largest decode edge of any graph we have built.** Against the
shipped twin's 88.3 t/s, ours decodes at 121.6 t/s, and prefill nearly doubles. That is the `cache_1024`
move paying off most where the graph is smallest: the KV cache is attention work, and at 0.6B it is a
larger share of an already-short per-token step, so halving it buys more here than the 1.17× it bought
at the 4B or the ~1.27× at the 1.7B. The smaller context is the trade — 1024 tokens of maximum context
against the shipped graph's 4096 — and it is also why the file is 4% smaller and the pool footprint is
**1.20 GiB lighter** (1.59 GiB against 2.79 GiB), the same three-way shrink the 4B shows.

**It is not token-identical to the shipped graph, and — unlike the 1.7B and 4B pairs — the two are the
same width and scheme.** Both sides are `w8`, symmetric per-output-channel int8, so unlike the 1.7B
(w4, scheme mismatched against its oracle) or the mixed cases, there is no quantizer difference to
point at here. The divergence is nonetheless real and starts in the **reasoning block**: shipped opens
"I know France's capital is Paris…" and closes with an added sentence after the answer; ours opens "I
need to make sure I provide the correct information…" and stops at **Paris**. Both converge on the
right answer, so the honest statement is that the `cache_1024` build is a **different graph, not a
faster copy** — the shorter context and the compiler's own graph change the early logits enough to flip
the greedy branch, exactly the chaotic-decode effect the 1.7B section describes. It buys speed and pool
footprint at a shorter maximum context, and it costs token-level identity with the vendor graph; which
of those a caller wants is the choice, and it is the same choice the compiled 1.7B and 4B present.

#### Long generation: the answer to "does 121 t/s hold, and what happens at the cache"

Every number above uses the ~8-token canonical answer, which cannot show either property a served graph
lives on: whether the text stays coherent over a long generation, and whether the decode rate holds as
the context fills. Both graphs were measured on 2026-10-10 with a prompt inviting a long answer
(`Write a detailed essay about the capital of France, at least 250 words.`), greedy, driven directly
through **`pocketllm run --device horizon`**, three runs each.

**The text is coherent to the end, at both sizes.** Ours produced an **881-token, 4200-character**
essay — a six-paragraph structure (introduction → history → architecture → economy → government →
conclusion) with no repetition loop, no mid-sentence break and the fact intact throughout. It even
closes with its own self-check, `(Word count: 250)`. The shipped graph's essay (a shorter 791 tokens, on
a different build) is coherent in exactly the same way. The excerpts:

```text
# ours, opening
…  </think>
**The Capital of France: Paris**  

Paris, the capital of France, stands as the heart of Europe, a city that seamlessly blends history,
culture, and modernity. …

# ours, 881 tokens later
… In conclusion, Paris is more than a city—it is the soul of France. Its history, culture, and economic
vitality make it a defining location for the nation. As the capital, it continues to inspire and unite
people across the world.  

(Word count: 250)
```

**It is deterministic**, three runs byte-identical: ours sha256 `47abb370a7108cf6338e0b8f4ac070016271bfc50fb1074488b5b5b43dc5bab7`,
the shipped graph's `e57999e2b4d0b954e322511841745fc4a417c560060256c1d09b9a35607126d3`.

**The rate does not degrade with position — and this is the thing the 8-token prompt could not show.**
Sampling the decode rate from per-token callback timestamps:

| graph | tokens 1–20 | tokens 180–200 | last quarter | delegate's own `decode_tps` |
|---|---|---|---|---|
| ours `cache_1024` | 121.5 | 121.4 | 121.4 | 121.35 / 121.89 / 121.82 |
| shipped `cache_4096` | 88.5 | 88.5 | 88.5 | 88.72 / 88.72 / 88.6 |

Flat at both sizes, and the delegate's own per-run figure agrees with the callback measurement to
within 0.5% — the 121.6 t/s of the ladder is a rate that holds for the whole generation, not a
first-token artifact. The smaller cache does not make decode *cheaper with position*; it makes every
step cheaper, uniformly.

**What the shorter cache actually costs is reach, and it is a hard cap, not a quality cliff.** Asked to
`Count from 1 to 400, one number per line, with no other text.`, ours counts correctly and coherently to
**168 and then truncates mid-number at `1`** — 986 generated tokens, three runs byte-identical (sha256
`17e717b690532b440232e43c66a14cf715eef84bd8fb05f589bd52dd477c20e3`). It is a **cutoff, not
degeneration**: the numbers are clean right up to the stop, with no repetition or garbage before it. The
exact limit is measurable: the tokenized prompt is **49 tokens** and the generation **986**, so
prompt + generation = **1035**, against the compiled `cache_1024`. Lengthening the prompt to **184
tokens** drops the generation to **851** — the same total, **1035**. So the graph caps chat at ~1035
tokens (prompt + generated) and truncates anything longer, mid-token, exactly where the cache ends.

**The shipped graph does the same thing at its own cap, 2 GiB further out.** A handful of prompts ended
on their own before the shipped graph's ~4096-token context (`Count from 1 to 5000` stopped at EOS after
2367 tokens, `1..500` complete). Pushed past it with `Count from 1 to 8000`, it also truncates
mid-token — **4057 generated, ending `…939 9`** — at its own limit, sha256
`0bf00626dafb062d8d84b0adaeb9c8f435550722d28255e7ca0e2124a9c528d7`. So both truncate; the shipped
graph's reach is simply **about four times ours**.

| graph | cache | measured reach (prompt + generated) | on hitting it |
|---|---|---|---|
| ours `cache_1024` | 1024 | **~1035 tokens**, mid-token | truncates; output clean to the stop |
| shipped `cache_4096` | 4096 | **~4100 tokens**, mid-token | truncates; output clean to the stop |

That is the honest trade the `cache_1024` build makes, stated plainly: **the same 1.38× decode, at a
quarter of the context.** A caller whose prompt + answer stays under ~1000 tokens — every chat turn in
these measurements, including an 881-token essay — gets the speed for free and never sees the cap; a
caller that needs a 4k window does not. The cap is the graph, not the HTTP shell: it shows up the same
way through `run` and through [`serve`](#serving-on-the-board).

**The reach is uniform across our compiled graphs — and one of them does not copy the 0.6B's clean long
generation.** Our 1.7B and 4B are `cache_1024` builds too, so the same ~1035-token cap should hold.
Measured the same way (greedy, per-token callback timestamps, `pocketllm run --device horizon` and the
harness), it does: the counting prompt that reaches **986 generated** on the 0.6B reaches the same
**986** on the 4B, and the padded 184-token prompt drops both to **851** — the same total, **1035**,
twice. The 1.7B's short-prompt count stops on its own earlier (it abbreviates to `1 2 3 … 400` instead
of listing), but its padded count reaches the same **851 + 184 = 1035**, so all three share one cap.
The cap is the graph shape, and it is the same shape for all three.

| graph | cache | reach (prompt+generated) | rate, tokens 1–20 → last quarter | long-essay text |
|---|---|---|---|---|
| ours 0.6B | 1024 | ~1035 | 121.5 → 121.4 | coherent (881 tok, ends on its own) |
| ours 1.7B | 1024 | ~1035 | 89.1 → 89.0 | **repetition loop** |
| ours 4B | 1024 | ~1035 | 49.9 → 49.9 | coherent, clean truncation |

**The 4B is the clean mirror.** Its essay runs coherently and — unlike the 0.6B's, which ends on its own
at 881 tokens — reaches the cap and **truncates mid-sentence**, the same clean cutoff the 0.6B's counting
prompt shows, with no repetition before it. Three runs byte-identical (sha256 `0189cacc…`), rate flat at
49.9 t/s from the first token to the last.

**The 1.7B is where it breaks: a repetition loop in the reasoning block.** On the same essay prompt, our
compiled 1.7B never closes its ` thinking` block. Its reasoning degenerates into a verbatim loop
(`… the 12th-century Notre-Dame de la Porte is a different building. The 13th-century Sainte-Trinité is
a different building. …`), the loop consumes the rest of the generation — the last ~60% of the text —
and the cache cap truncates it **mid-loop** at 988 generated tokens. There is no `</think>`, no answer
and no EOS: a caller sees the raw, looping reasoning. It is deterministic (three runs identical, sha256
`bdc40f49…`) and reproduces on a second, unrelated topic (`… The 19th century, the 19th century. …`,
sha256 `268c4ac1…`) and through `pocketllm run --device horizon` as well as the harness. The rate stays
flat *through* the loop (89.1 → 89.0 t/s), so this is the model's own output, not a throughput limit.

**It is our compiled 1.7B, not the 1.7B.** The shipped 1.7B — same weights, `cache_4096` — on the same
essay prompt emits an empty reasoning block and a **coherent essay** that ends on its own conclusion at
405 tokens, no loop, 70.2 t/s flat (sha256 `972dfe4b…`). So the loop is a property of **our compiled
1.7B graph**, not of the 1.7B weights and not of long generations in general — our 0.6B and 4B do not
show it, and the 1.7B's own short generations are fine (it answers all ten faithfulness prompts below).
What is *not* established is the cause inside the compile: this is one measured artifact behaving badly,
not a diagnosis of which pass in the build produced it.

**Tested board-side, without a rebuild — the obvious triggers are refuted.** *Cache length and position*
are not it: the shipped 1.7B (`cache_4096`) was driven to **2022 generated tokens** listing every integer
1–400 completely and correctly with **no loop**, well past our 1024-token window, so a window of ~1024 is
not a threshold past which this graph loops; and our 1.7B loops at *different* positions on different
prompts (piece 143 on the essay, 107 on the list), so there is no fixed onset. *Thinking mode* is not it:
with `enable_thinking` off our 1.7B loops anyway (`the Eiffel Tower, the Eiffel Tower …`, ×141). The
repeated piece is ordinary text (`I`, ` the Eiffel Tower`), not a degenerate special id. There *is* an
early numeric divergence — on the looping prompt our 1.7B and the shipped 1.7B agree on only **two
pieces** (` thinking\n`) and diverge at the **third token** (ours reasons, the shipped emits `</think>`) —
but that immediate divergence is a property of **every** rebuild, not of the loop: our **0.6B** diverges
from its own shipped twin at the **fourth token** and stays coherent, and the same 1.7B pair still answers
all ten faithfulness prompts identically. So the trigger is none of position, template or token mapping; it
is the graph's own generation on the 1.7B `w4` weights. The one parameter that *does* separate our graph
from the shipped one — the cache size — is where the next subsection points the rebuild.

#### The 1.7B loop: a 2-token reproducer, its onset, and the one build parameter that differs

The loop localizes to a **minimal reproducer**, and the two graphs it contrasts differ in exactly one
build parameter.

**Minimal reproducer: the prompt `France?`.** Two tokens, greedy, on our compiled 1.7B (`cache_1024`,
`w4`), it generates **1004 pieces / 4467 chars** to the cache cap with **no `</think>`, no answer and no
EOS**. The repetition starts at generated **piece 31** (char 120, 2.7% in) and holds to the end (a 40-char
window recurs 4× in the first half, 7× in the second); the recurring text is meta-commentary (`… a general
answer. Let me think of the possible answers.`) with immediate word repeats (`fashion and fashion`), and it
is **byte-identical across runs** (sha256 `f3f87ee4…`). It is not *any* prompt — generations that stop
early stay clean (`Write` → 142 pieces ending `How can I assist you?`, `The capital of France is` →
`**Paris**`) — so the trigger is a generation that runs **long**, not the prompt.

**Divergence is early; the loop is late.** Our 1.7B and the shipped 1.7B first disagree at generated
**piece 9** on `France?` and **piece 2** on the essay, but the loop starts at **piece 31** and **143** — a
divergence→onset gap of **22** and **141** pieces. Across eleven prompts the onset lands anywhere from
piece 31 to 810 (0.03–0.81 of the generation): not at the divergence, not at a fixed index, only after the
generation has run long.

**Not the content.** The integer-list prompt — a long, non-repeating enumeration the **shipped** 1.7B drives
cleanly through 1–400 — **loops in ours anyway** (`…from 1 to 400…`, never finishing), while **short**
enumerations (`1–100`, `1–200`) end on their last number. A non-repeating answer loops; a short one does not.

**The commanded build parameters differ only in the cache.** Our 1.7B and the shipped 1.7B are both
`Qwen3-1.7B_language_chunk_512_…_w4_nash-p_corenum_4_4` — same `w_bits` (4), `chunk`, `march`, `corenum`
— and their command lines differ in **exactly one parameter: `cache_len` (1024 vs 4096)**. So the "cache
length is not it" above narrows to: that tested *generation length* (2022 tokens on the shipped graph), not
the *cache size*, and the shipped graph is `cache_4096` and simply never loops. `cache_len` is the leading
suspect **because it is the only commanded difference**.

**But the two artifacts differ in more than the command line.** `model_info` shows the **KV-cache**
quantization scales are *not* the same on the two graphs (`layer_0_cache_key`: shipped `0.0120087`, ours
`0.0120163`; 110 of 112 such tensors differ — [the variance is in the native chain
page](../architecture/s600_native_chain.md#the-first-build-result)). A
quantization scale is set by the **calibration activation range**, not by the tensor's length, so the
shipped graph was **not** produced by an identical build with only the cache changed: the two are separate
calibrations of the same weights, and which of `cache_len` and the calibration causes the loop is not yet
separated. The decisive rebuild is still **ours at the vendor's own cache** — `cache_len 4096`, all else
identical ([the command](../architecture/s600_native_chain.md#the-concrete-command-sequence)): a **clean**
result pins `cache_len 1024` as the cause; a **still-looping** one says the difference is calibration or
provenance — our build pipeline or our calibration — not the cache. (`cache_len 2048` is the cheaper first
step; `w_bits 8, cache_len 1024` the `w4` control. Both have since been run, and both close the block —
[the `w8` rebuild at the loop's own cache below](#the-loop-at-w8-the-same-cache-closes-it-on-wider-weights).)

**The loop is a precision trajectory, not a graph defect.** Four measurements separate the two. One,
the onset is **content-determined and position-independent**: in **absolute context position** it lands
at 52, 165, 180 and 201 tokens on `France?`, the list, the essay and `Tell me about France.` — no
clustering at 256/512/1024, so no cache boundary triggers it — and padding `France?` with a repeated
filler moves the onset in **generated-step count** (piece 31 → 49 → 105) without holding *either* that
count or the absolute position fixed; the onset follows the continuation, not an address. Two, the
degeneration is **gradual, not sudden**: the fraction of never-before-seen 8-char shingles erodes over
roughly **450 largely-coherent pieces** before it collapses, and revives (88% novel at piece 350 of
`France?`) — a fixed wrong op cannot turn correct again. Three, the attractor **changes with the task**:
on one prompt it is a verbatim trap (`. The population is 67 million.` ×103), on the others it is a
self-referential template (`… Let me check the requirements again …`) — a defect would show the same
failure on every prompt. Four, the strongest, is functional: on the **same** list prompt the shipped
graph completes 1–400 while ours never writes the list at all, looping in planning prose — same weights,
same tokenizer, same runtime, different cache.

The verdict is trajectory; what the board cannot yet split is `cache_len` versus the calibration (both
sit behind the same rebuild, and both are precision). The rebuild above **is** the separating
measurement: if the loops vanish, cache-length sensitivity; if they persist, provenance/calibration. **It
has now been run** — our 1.7B at `cache_len 4096` — and the runaway loop does not appear at the larger
cache; [the next subsection](#the-rebuilt-17b-at-the-vendors-cache-the-runaway-loop-is-the-cache) has the
table and the verdict. The **calibration corpus itself has since been tested and ruled out** as the
lever — the matched corpus moved the KV scales *away* from the vendor and left agreement no better — on
[the native compile chain](../architecture/s600_native_chain.md#the-calibration-corpus-tested-and-ruled-out).

**The compiled 0.6B and 4B do not reach it — the 1.7B crosses the threshold first.** The "only the 1.7B
loops" line was originally drawn on short generations that ended cleanly, which the 1.7B's own trigger
(sustained generation) never tests; driven to the cache cap on the same three prompts, the small graphs
stay clean. The comparison is not marginal: across every small-model run the most-repeated 40-character
window recurs **at most twice**, while the 1.7B repeats its attractor **8–18×**, and the small graphs
*answer* — the 0.6B writes the list through 177 and the 4B through 59 (both truncated by the cap), while
the 1.7B never writes the list at all. The 4B shows the mechanism approaching — a repetitive-reasoning
stretch in its thinking block on two prompts, and it reaches the cap on two of three — but nothing
fatal: the threshold sits beyond ~450–650 steps at `w4`, and the 1.7B is the first of our compiled
graphs past it.

#### The rebuilt 1.7B at the vendor's cache: the runaway loop is the cache

#634's decisive build — **our** 1.7B at `cache_len 4096`, the same pipeline and the same `w4`, `chunk 512`,
`march` and `corenum`, with only the cache changed — has been built (`Qwen3-1.7B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm`,
1.83 GB, md5 `af63d83d…`, on the 2080ti with `--device cpu`) and driven the same way as the pair it joins:
the `1..400` list and the `France?` reproducer, greedy, three runs, one graph per process, `bpu_core [0,1,2,3]`.

| 1.7B, `_w4` | decode t/s | prefill t/s | load | carveout after load | peak (long generation) | free |
|---|---|---|---|---|---|---|
| ours `cache_1024` | **89.1** | 8 828–9 846 | 5.62 s | 2.27 GiB (22.7%) | 2.42 GiB (24.2%) | 7.58 GiB |
| ours `cache_4096` (new) | **70.2** | 5 172–5 818 | 5.78 s | 3.48 GiB (34.8%) | 3.63 GiB (36.3%) | 6.37 GiB |
| shipped `cache_4096` | **70.1** | 5 172–5 818 | 5.70 s | 3.48 GiB (34.8%) | 3.63 GiB (36.3%) | 6.37 GiB |

**The runaway loop does not appear at 4096.** The same pipeline that runs away at `cache_1024` terminates
with a complete answer at `cache_4096`:

| prompt | ours `cache_1024` | ours `cache_4096` (new) | shipped `cache_4096` |
|---|---|---|---|
| `France?` | 1004 pieces / 4467 chars, **no `</think>`**, 40-char window recurs **11×** | 229 pieces / 989 chars, closes the block, recurs **2×** | 407 pieces / 1986 chars, closes, recurs 2× |
| `1..400` list | 3605 chars, **no `</think>` and no list** (recurs 8×) | 3876 chars, **closes and writes 1…400** (recurs 4×) | 4276 chars, closes and writes 1…400 (recurs 1×) |

The new build's `France?` ends on its own at 229 pieces against the `cache_1024` graph's 1004, and its list
run writes all of 1–400 where `cache_1024` never closes the block and writes none of it. Since the cache is
the only commanded change, `cache_len 1024` is a **cause**, not a bystander. It is not *identical* to the
shipped graph: at 4096 our plan block still echoes its own task line (**4×** on the list, against the
shipped graph's 1×), a residue of the repetitive trajectory #634 localized — but the block now **closes**
and the answer follows, which at `cache_1024` it never did.

**The new build lands exactly on the `cache_4096` rate, and the #637 lever model holds.** Decode is
**70.2 t/s**, the shipped graph's **70.1** to within a run; prefill is 5 172–5 818, also the shipped
graph's. The price is the **1.27×** smaller-cache gain #637 measured (89.1 → 70.2 t/s, 11.2 → 14.2 µs/step)
and **1.21 GiB** of extra carveout (2.27 → 3.48 GiB, 22.7% → 34.8% of the 10 GiB pool), leaving 6.37 GiB
free. The lever is real for the 1.7B (27%), smaller than the 0.6B's 38% — #630's fixed per-step KV cost is
a smaller share of a heavier model — so 4096 buys the 1.7B **no rate**, only reach and coherence.

**Verdict on #637.** #637 set `cache_len 1024` for 0.6B/1.7B and `2048` for 4B/8B on *rate* grounds, and
deferred the 1.7B loop to a calibration or `w8` rebuild. The **rate model is confirmed** — the new 4096
build reproduces the shipped 70 t/s and the 1.27× is exactly #637's number. But the 1.7B part of the
recommendation is **revised**: the cache is not a pure rate knob, it is the difference between looping and
answering, so the 1.7B wants a window large enough to close the block — `4096` on the shipped graph, and
`2048`, the cheaper step this paragraph deferred as *untested*, is measured next and closes the loop too.
The `w8` rebuild is a third closure — same `cache_1024`, wider weights, block closes — so the runaway
tracks the `w4`/`1024` pairing and not the window alone
([the loop at `w8`](#the-loop-at-w8-the-same-cache-closes-it-on-wider-weights)).
The 4B/8B `2048` call stands and is reinforced — a window too small to close a long thinking block is what
breaks the 1.7B, and those sizes run their thinking longer still.

#### The 1.7B at `cache_2048`: the loop closes, the list does not finish

The one cache between ours (`1024`, loops) and the vendor's (`4096`, clean) — `2048`, the step #637 called
"the cheaper step, untested" — has now been built and driven the same way as the pair it joins
(`Qwen3-1.7B_language_chunk_512_cache_2048_w4_nash-p_corenum_4_4.hbm`, 1.79 GB, md5
`41abcbeec8afa3ec4ec47a610eb12905`): the same `w4`, `chunk 512`, `march` and `corenum`, only the cache
changed, three runs per cell, one graph per process, `bpu_core [0,1,2,3]`.

| 1.7B, `_w4` | decode t/s | load | carveout after load | peak (long generation) | free |
|---|---|---|---|---|---|
| ours `cache_1024` | **89.1** | 5.62 s | 2.27 GiB (22.7%) | 2.42 GiB (24.2%) | 7.58 GiB |
| **ours `cache_2048`** | **80.8** | 5.75 s | 2.86 GiB (28.6%) | **3.01 GiB (30.1%)** | 6.99 GiB |
| ours `cache_4096` | **70.2** | 5.78 s | 3.48 GiB (34.8%) | 3.63 GiB (36.3%) | 6.37 GiB |
| shipped `cache_4096` | **70.1** | 5.70 s | 3.48 GiB (34.8%) | 3.63 GiB (36.3%) | 6.37 GiB |

`2048` lands where a midpoint should: **80.8 t/s** and **2.86 GiB**, between the `1024` build's
89.1 / 2.27 and the `4096` build's 70.2 / 3.48. Decode is **9% below** `cache_1024` and **15% above**
`cache_4096`; the carveout is 0.59 GiB more than `1024` and **0.62 GiB less** than `4096`.

**The runaway loop is gone at `2048`.** The `France?` reproducer that never closes its block at
`cache_1024` terminates cleanly at `2048`, exactly as it does at `4096`:

| prompt | ours `cache_1024` | ours `cache_2048` | ours `cache_4096` |
|---|---|---|---|
| `France?` | 1004 pieces / 4467 chars, **no `</think>`**, 40-char window recurs | 300 pieces / 1313 chars, **closes**, 586-char coherent answer | 229 pieces / 989 chars, closes |
| `1..200` | abbreviates (`… 200`) | **complete, 1…200** (2590 chars) | complete, 1…200 |
| `1..400` | **no `</think>`, no list** (recurs 8×) | **closes**, but abbreviates (`1…10 … 400`, 987 chars) | **complete, 1…400** (3876 chars) |

So `cache_2048` is **enough to close the reasoning block** — the 1.7B loop is not a `4096`-only fix — but
it is **not enough to write the long enumeration in full**: `1..200` completes at `2048` where `1024`
abbreviates it away, yet `1..400` still collapses to an ellipsis (`1 2 … 10 … 400`) where `4096` lists
every number. The completion threshold tracks the cache, and monotonically: `1024` finishes a list of
~100, `2048` of ~200, `4096` of ~400. A larger window buys strictly more of a long deterministic answer.

**No corruption at `2048` — the 1.7B does not slip at either cache.** The corrupted-token signature the
survey localized is a `4B`/`8B`/`cache_1024` property; the 1.7B never showed it, at `1024` or here. At
`2048` the `1..200` list is complete and **correct**, and `1..400` abbreviates *cleanly* — an ellipsis, not
a wrong number that then continues. So the enumeration answer is **clean at `2048`, as at `4096`**: the
larger cache neither introduces a slip nor trades the loop for one; it closes the block and lengthens how
far the list runs.

**The `France?` loop onset, for the record.** At `2048` the answer is a fragment — *"France is a beautiful
country in southern Europe, known for its beautiful landscapes, ri…"* — and it ends on its own at 300
pieces, no loop window recurring three times. The `cache_1024` graph, on the same two tokens, never
closes the block and runs to the cap. Same weights, same prompt, same greedy sampler; the cache is the
only commanded difference, and it is the difference between an answer and a runaway.

#### The loop at `w8`: the same cache closes it on wider weights

The other single-variable escape from the `cache_1024` loop was already built: the 1.7B `w8` rebuild
[below](#the-17b-at-w8-above-the-floor-against-truth) varies **exactly one commanded flag** against the
looping `w4` `cache_1024` graph — `--w_bits 4 → 8`, same `cache_len 1024`, `chunk`, `march`, `corenum`
and pipeline — and driven through the same reproducer (`France?`, greedy, one graph per process, measured
2026-10-13) it says the loop is **gone**:

| prompt | ours `w4` `cache_1024` | **ours `w8` `cache_1024`** |
|---|---|---|
| `France?` | 1004 pieces / 4467 chars, **no `</think>`**, 40-char window recurs **11×** | **353 pieces / 1631 chars, closes the block, recurs 2×** |
| `1..400` list | 3605 chars, **no `</think>` and no list** (recurs 8×) | 3536 chars, **closes**, writes **1…36** before the 1024-token window ends it (recurs 2×) |

The runs are modes, not draws: two `France?` generations are byte-identical (`sha256 8098aa71…`), and
the list is byte-identical three of three (`sha256 22e640d7…`). The block closes and an answer follows —
*"France is a country in Europe, known for its rich history, culture, and iconic landmarks. Its capital,
Paris, …"* — where the `w4` graph never closes at all, and the **353** pieces land between
`cache_2048`'s **300** and the shipped graph's **407**. The recurrence level falls to **2×**, the same
the two clean `cache_4096` builds show. The one honest qualifier is the list: the `w8` graph **does**
start the enumeration — 1…36, cut when the window fills — where the `w4` graph wrote none of it; that is
the 1024-token budget, not a runaway.

So the runaway needs **both** the `w4` weights and the 1024 window: widening either one ends it. That is
the same reading [the `4096` rebuild](#the-rebuilt-17b-at-the-vendors-cache-the-runaway-loop-is-the-cache)
and the onset analysis already give — a precision trajectory, not a fixed graph defect — with the `w8`
rebuild now the second single-variable confirmation, from the width side.

#### A request whose prompt does not fit the cache was an abort, and is now refused

The cap above is about *generation*: a prompt that fits still truncates its answer cleanly at the window.
The other half — what a request does when its **prompt** does not fit — is worse than a truncation, and a
deployer will meet it the first time it forgets a length check.

**A prompt longer than the cache kills the process with `SIGABRT` (glibc `corrupted size vs. prev_size`).**
It is not an error status and not a clean truncation: the delegate walks past its KV and the allocator
aborts. Measured on the compiled **0.6B** (`cache_1024`), greedy, one prompt per process, effective
prompt = prompt + the same 29-token chat wrapper #616 counted:

| effective prompt | 1001 | 1021 | **1041** |
|---|---|---|---|
| result | OK, 152 chars | OK, 50 chars | **abort, rc 134** |

The whole window is usable — 1021 still answers — and **one token past 1024 aborts**, on content that is
just filler text followed by a question. It is the length, not the topic: a `What is the capital of
France?` tail on the same preamble (effective 1037) aborts identically.

**It is the SDK's failure, not our build's — at whatever cache the graph was built with.** The **shipped**
0.6B (`cache_4096`) on the same prompt family answers at effective **4090** and aborts at **4130**. Our
compiled graphs each crash at their own `cache_1024`; the shipped builds would crash at `cache_4096`. The
compiled cache does not *add* the failure, it moves the cliff 4× closer.

**On `serve` the crash is the request and every request after it.** An over-cap completion gets **no HTTP
response at all** — the connection closes after ~2.6 s (curl `HTTP 000`, size 0) — and the server is gone:
the next request is `connection refused` with nothing listening. `/v1/chat/completions` does the same. So a
single over-long prompt **takes the server down for every subsequent client** — silent in the sense that
the first caller sees a dropped connection rather than a 200, but fatal for the process.

**What is clean.** Inside the window the cap is honest: `finish_reason: "length"`, HTTP 200, text coherent
up to where it stops, and the KV is not left dirty — a canonical `The capital of France is` issued in the
same process *after* a request whose **generation** hit the cap (a prompt that fits) still returns `Paris`.
It is the over-cap **prompt** that is fatal, not the truncation: there is no "next request" after that one,
because the process is gone. Two smaller quirks sit beside it and are not
corruption: `max_tokens` is not applied on this path (a request for `max_tokens: 3` returns the full
answer, because the delegate owns its own stop), and a generation consumed entirely by an unterminated
reasoning block surfaces as **empty content at 200** once the chat splitter strips it. Neither is the
prompt-does-not-fit case, which is a crash.

**The crash is now refused on the host, at both entry points, before any tensor is fed.** The cache is the
graph's compiled window and the delegate cannot count a prompt, so the guard is a *host* one: `XlmEngine`
takes an injected `count_tokens: Callable[[str], int]` and the `.hbm`'s own `cache<n>` build shape, and
refuses in `infer` when the effective count is past the window — `pocketllm.xlm` stays stdlib-only, since a
`Callable` is stdlib and the tokenizer is not. The host supplies the real counter from the SDK's own
`configs/Qwen3_config/tokenizer.json` (via the optional `tokenizers` package) plus the measured 29-token chat
wrapper; the window is read from the `.hbm` name (`cache_1024`, `cache_4096`). `run` now exits **rc 1** with a
message naming both numbers, and `serve` answers **HTTP 400 `invalid_request_error`** — the server stays up and
the next valid request is still served. When either half cannot be read (no `tokenizer.json`, a `.hbm` without
a `cache<n>` shape) the guard is **disarmed and the serving path warns once at startup**, rather than running
silently unguarded. The 0.6B `The capital of France is` regression is unchanged: in-window prompts still
answer `Paris`.

**The refusal holds under `serve` concurrency — the case the one-request-at-a-time design exists for.** Six
distinct prompts fired at once all returned 200 with their own correct answers (6/6), the same serialization
the lock already guaranteed. Then four valid distinct prompts and two over-cap ones fired **together**: every
valid request came back **200** with its own text (4/4, no garbling from the 400s interleaving), both over-cap
requests came back **400 `invalid_request_error`** naming `1024` (2/2 — not a 500, not a dropped connection),
and the server was **still up** afterward, serving the next valid request correctly. Repeated four times, same
result, with no `corrupted size`/`SIGABRT` in the log. So the guard turns the crash into a *per-request* error
without the serialization turning one over-cap request into a server-wide failure — the interaction the guard
was most likely to get wrong.

#### Where the compiled 0.6B's 1.38× actually comes from: the BPU is the whole step

The 121.6 t/s above is an end-to-end rate; the question the next lever needs is *where that time goes*. The
delegate reports a `Performance` struct but **zeroes its own token counts and `end_to_end_cost` on this SDK
build** — only `prefill_tps`/`decode_tps` are live — so the split comes from the delegate's rates plus the
SDK's own per-node profiler (`hrt_model_exec perf`, which needs our `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`).

**Load is the same either way.** `xlm_init`→ready is **5.22–5.33 s** for ours and **5.28 s** for the shipped
0.6B; the cache size does not move it.

**Prefill and decode, ours vs the shipped twin** (same weights `w8`, `chunk_512`, `corenum_4_4`; only the
cache differs):

| | cache | prefill tok/s | decode tok/s |
|---|---|---|---|
| ours | `1024` | **11 636–13 128** | **121.3** |
| shipped | `4096` | **6 169–7 014** | **88.2** |

Both halves gain — prefill **1.77–1.89×**, decode **1.38×** — so the win is not decode-only.

**The decode step, decomposed** (`hrt_model_exec perf`, ms; `decode_tps` disagrees with the host because the
profiler runs the raw graph without the delegate's in-place KV handling):

| stage | ours `cache_1024` | shipped `cache_4096` | ratio |
|---|---|---|---|
| Stage2 BPU-core compute | **6.93** | **9.97** | 1.44× |
| Stage1 launch / sync | 0.53 | 0.53 | 1.00× |
| CPU nodes (GatherND + reshape) | 0.013 | 0.014 | 1.08× |
| **total** | **7.46** | **10.51** | **1.41×** |

**It is BPU-compute-bound and spends nothing on the host.** ~93% of the step is BPU-core compute, ~7% is op
launch, ~0.2% is CPU; the host's own per-step cost is **immeasurable** — bracketing `infer`'s callback, the
host-observed step median matches `1/decode_tps` to within noise (host share ≈ −0.7%), so there is no
host-side overhead beyond the BPU. **The CPU nodes are identical across both graphs**, so the entire 1.38× is
BPU compute.

**The KV size explains all of it, and the fitted slope — not the ratio — is what pins the mechanism.** The
graph holds KV for a *fixed* cache even in a one-token step, and for GQA-8 attention that is linear in the
cache: `0.6B · 28 layers · 2 · 8 · 128 · 2 B ≈ 0.115 MB` per token. The BPU-core time fits it exactly:

```
9.970 ms (cache_4096) − 6.927 ms (cache_1024) = 3.043 ms over 3072 tokens
                                             = 0.991 µs/token
0.115 MB/token ÷ 0.991 µs = 116 GB/s effective
```

That is a textbook **LPDDR5 effective bandwidth — and it is the whole step's rate, not a KV-only one.** Reading
the same way at both caches, the step is weights (~700 MB at `w8`, a fixed read present in both graphs) plus KV
(`0.115 MB` per cached token, read in full even for a one-token step): `700 + 117 MB` over 6.93 ms at
`cache_1024`, `700 + 470 MB` over 9.97 ms at `cache_4096` — **~116 GB/s either way**. (The single weight size
that makes the two rates equal is **≈688 MB**, which is the `w8` footprint of 0.6B params plus scales — the
one-parameter fit reproduces the model's own weight size, which is the check that it is the right model.) So
the 3.04 ms is simply the **352 MB of extra KV** the larger cache reads, at the same bandwidth, and it rules
out the alternatives:
quadratic attention is ≪0.01% of a step at ≤1035 tokens (one query against a linear key axis), and a fixed
per-step overhead would show in launch — which is identical at 0.53 ms in both graphs. The step is **pure DRAM
streaming, and the cache sets how many KV bytes it streams**; the ~0.53 ms of launch on top is the only part
not streaming. So the levers left are **a smaller cache for decode** (fewer bytes) and a **batch** to amortize
launch across several steps — not anything on the host, and not the weights, which are already `w8` and a fixed
cost. It is a *decode-only* win: prefill is compute-bound and pays little for the cache (its gain comes from the
build, not the KV).

**Prefill, decomposed the same way, is at the roofline — and has no lever.** The complement of the decode
story is the prefill step, and it is the opposite shape. Profiled at the node level, the compiled prefill is
**94–97% BPU-core** (ours 0.6B: 33.25 of 35.40 ms; ours 1.7B: 46.17 of 49.10), with an almost constant
**~1 ms of launch** (two BPU segments) and ~1–2 ms of CPU nodes. The **1.7–1.9×** edge over the shipped
prefill is **entirely Stage2**: the launch and CPU ratios across the pair are 1.00–1.02, so our build does
about **half the BPU compute** for the same prompt — less work, not fewer launches. It is also **not** near
the memory roofline: 33.25 ms for the ~688 MB of `w8` weights is **~21 GB/s**, ~5.6× below the decode step's
116. So prefill is **BPU-arithmetic-bound, at its ceiling**, and its rate is **flat** with prompt length
(13 128 t/s from 69 to 909 tokens on ours 0.6B — no quadratic-attention falloff, so the cost is linear, a
constant per-token matmul). The lever decode has (a smaller cache) does not exist here: launch is 1–3% and
halving it moves nothing, and the only way under the BPU ceiling is less arithmetic. **No prefill lever.**

**The lever, measured across sizes — the win is shallow and it is a small-model win.** Decode on the long
list prompt, ours `cache_1024` vs the shipped `cache_4096`, same weights otherwise: the gain is **1.38×**
on the 0.6B (121.5 vs 88.4 t/s), **1.27×** on the 1.7B (89.2 vs 70.2) and **1.17×** on the 4B (49.9 vs
42.8) — falling with size because the weights are a fixed read the cache does not touch, so the KV is a
large share of a small model's step and a small share of a large one's. `cache_512` would add only ~25%
more decode over `1024` on the 0.6B and less elsewhere. (These pairs also differ in *calibration* — #634 —
but calibration changes the scale values, not the tensor shapes, so the KV byte count and the *rate* are
still a clean cache comparison.)

**And it trades against context and against the loop threshold.** The usable window is `cache_len − 29`
chat wrapper `− prompt`: `cache_1024` leaves **~940–975 tokens**, enough for the essay (~330 output + 37
prompt) with ~630 to spare, while truncating a long *thinking* run (the 4B reached only 59 in its list
because its thinking used ~800 of the 966 before the answer began). Shrinking below `1024` collides with
#634/#636: the loop threshold sits beyond **~450–650 steps**, so `cache_512` (a ~425–460-token window)
would *truncate a generation before the loop can appear* — faster and loop-proof for the 1.7B, but it caps
an ordinary answer at ~450 tokens, which is not a fix for a precision-trajectory problem.

**Recommendation for the next build round (not built — the queue is ours to sequence):** **`--cache_len
1024` for the 0.6B and 1.7B** (already the sweet spot — fastest, tested; the 1.7B's next build should
spend its budget on the loop, not the cache) and **`--cache_len 2048` for the 4B and 8B** (their thinking
truncates at 1024 and decode is weight-bound, so doubling the window costs almost nothing). **Never build
below `1024`** — the floor is the loop threshold plus the wrapper.

#### The decode step from outside: the host contributes nothing, and there is no lever out here

The 310B got two of its wins from changes *inside* the decode path — a KV-projection dtype fix (615 → 309
ms/step) and a greedy fast path (266 → 137). S600 had never been given the same treatment, and the honest
question is whether an equivalent lever exists on the *outside* of the delegate: the CLI shell
(tokenize, sampling, detokenize, print), the `ctypes` boundary, the BPU-core allocation, the cache. The
delegate is a black box — the compiled `.hbm` is text in, text out — so what can be measured from outside
is exactly the shell and the graph's own rates, and nothing between.

**Method: bracket every decoded piece with `perf_counter` at the callback.** The delegate makes a callback
per piece; the host timestamp at each one gives the *host-observed* per-step time, to compare against the
delegate's own `decode_tps` (whose reciprocal is its per-step claim). If the shell were a real share of the
step, the host-observed step would exceed `1/decode_tps`; the difference is the host's share. Same
enumeration prompt as the rest of this page, greedy, one process per graph, three runs each, median:

| graph | delegate step (`1/decode_tps`) | host-observed step | host share |
|---|---|---|---|
| ours 0.6B `cache_1024` | 8.26 ms | 8.30 ms | **0.5%** |
| ours 1.7B `cache_1024` | 11.26 ms | 11.36 ms | **0.9%** |
| ours 1.7B `cache_4096` | 14.34 ms | 14.44 ms | **0.7%** |
| shipped 1.7B `cache_4096` | 14.30 ms | 14.39 ms | **0.6%** |
| ours 4B `cache_1024` | 20.07 ms | 20.17 ms | **0.5%** |
| shipped 8B `cache_4096` | 33.77 ms | 33.92 ms | **0.4%** |

**The shell is 0.4–0.9% of the step — 70 µs to 150 µs — and it does not grow with the work.** The gap is
roughly constant in absolute terms (0.04–0.15 ms) while the step quadruples from the 0.6B to the 8B, so it
is fixed per-piece cost (the Python callback boundary), not a share of anything the model does. Timing the
shell's own phases separately confirms the same shape: `xlm_init`→ready is **5.3 s cold, 1.3–1.8 s warm**
(page cache), and the over-cap token counter is built **once** and cached — neither is a per-step cost.

**The callback is on the critical path, one-for-one — and that is what proves the host side has no room.**
Injecting real host work into the callback is additive to the step, millisecond for millisecond:

| injected per piece | wall step | host-observed step | delegate-reported step |
|---|---|---|---|
| 0 ms | 14.36 ms | 14.35 ms | 14.24 ms |
| 2 ms | 16.37 ms | 16.37 ms | 16.27 ms |
| 5 ms | 19.41 ms | 19.41 ms | 19.31 ms |
| 10 ms | 24.32 ms | 24.41 ms | 24.31 ms |

Two things fall out. First, the delegate **hands control to the host between tokens and waits for it** —
the callback is not overlapped with the next step's BPU work, so host work and BPU work serialize (this is
why a host-side per-piece hook is a *cost*, never a lever). Second, when 10 ms is injected, the delegate's
own `decode_tps` drops to match the wall time (41 t/s, 24.3 ms reported) — so `decode_tps` is a **wall-time
measurement including the callback**, not a BPU-only figure, and the small sub-1% gap between it and the
host is pure timestamping skew. Nothing hides between them.

**The watchdog's per-piece write is free.** The one piece of per-piece work this tree does add is the
streaming `os.write` the generation watchdog emits per decoded piece. Bracketed the same way, a run that
`os.write`s each piece and one that does not differ by **0.01 ms/step** (14.345 vs 14.355) — inside the
run-to-run spread, i.e. unmeasurable. So the watchdog's live stream costs no decode rate, which is the
check that it is safe to leave on.

**`bpu_core` is fixed at compile time and is not a lever.** Every graph here is `corenum_4_4`, and a config
whose `bpu_core` names a subset of the compiled cores does not simply run fewer of them — the delegate
refuses it (`XLM_STATE_ERROR`, the graph did not run). The core count is a property of the `.hbm`, so
"more cores = faster decode" is not something a caller can dial from the host; it would be a rebuild.

**The cache already told the same story, and the new numbers corroborate it.** The 1.7B step is **11.26 ms
at `cache_1024` and 14.34 ms at `cache_4096`** — same weights, same host, only the cache differs — so the
**3.08 ms difference is the KV alone** (3072 extra cached tokens at ~0.115 MB/token ≈ 353 MB, ≈ 114 GB/s,
the LPDDR5 rate the [step decomposition](#where-the-compiled-06bs-138-actually-comes-from-the-bpu-is-the-whole-step)
fitted). The delegate reports the larger cache at 69.7 t/s against the smaller at 88.8 — a **1.27×** decode
win, reproducing the ladder's figure on a fresh measurement, and the whole of it is DRAM bytes the host
never sees.

**What this cannot attribute, stated plainly.** The split *inside* `xlm_infer` — BPU-core compute vs the
delegate's own op-launch/host glue (its tokenization, sampling bookkeeping, detokenization) — is invisible
from outside the black box. The evidence that the step is DNN-bound is external and indirect: the shell is
<1%, the cache moves the step by exactly the KV's bytes at the DRAM rate (which a host-bound step would
not), and `hrt_model_exec`'s node profiler independently puts ~93% of the 0.6B step in BPU-core. But a
host-glue share of a few percent inside the delegate could hide in that residue and this page would not see
it. What is *established* is the shape of the answer: **the lever is not outside the delegate.** The shell
is 0.4–0.9%, the callback is a cost rather than a saving, and the core count is fixed at build — so a
310B-style win on S600 is a **rebuild** (a dtype / fast-path change inside the `.hbm`, which the compile
host owns), not a host change. The one lever a caller does hold is the cache, and it is already spend
against context, not free.

#### Answer faithfulness: 9/10 on the 0.6B and 10/10 on the larger graphs

A different build of the same weights can be faster and still reach different answers, and the whole
point of the compiled graph is that it is *useful*, not just fast. So ours and the shipped 0.6B were
put through a battery of **ten prompts with a single checkable short answer**, greedy, the final answer
extracted from each (everything after the reasoning block). Both sides are `w8`, so this is the
width-and-scheme-matched pair and the cleanest signal of the set.

| prompt | ours `cache_1024` | shipped `cache_4096` | final answer matches? |
|---|---|---|---|
| `The capital of France is` | **Paris** | **Paris** | yes |
| `2 + 2 =` | **4** | **4** | yes |
| `The largest planet in the solar system is` | **Jupiter** | **Jupiter** | yes |
| `The chemical symbol for gold is` | **Au** | **Au** | yes |
| `How many continents are there?` | **14** | **7** | **no** |
| `The first month of the year is` | **January** | **January** | yes |
| `The capital of Japan is` | **Tokyo** | **Tokyo** | yes |
| `The boiling point of water in Celsius at sea level is` | **100 °C** | **100 °C** | yes |
| `The largest ocean on Earth is` | **Pacific Ocean** | **Pacific Ocean** | yes |
| `The chemical symbol for water is` | **H₂O** | **H₂O** | yes |

**9 / 10 final answers are identical**, and on those nine both graphs give the same correct fact —
whether the answer is one word or the shipped graph adds an elaboration sentence ours omits.

**The tenth is the finding, and it is a wrong answer, not a reworded one.** Asked
`How many continents are there?`, the shipped graph answers **7** (correct) and ours answers **14**,
with a fluent, confident elaboration ("These include regions such as Africa, Asia, Europe, North
America, South America, Oceania, and others. The continents are divided into 14 distinct landmasses, with
no overlap between them."). It is deterministic — three runs of each graph, byte-identical within a
graph — so it is a property of the compiled artifact, not a flaky draw.

This is a **stronger** statement than the reasoning-block divergence recorded above, and it is the one
that matters for the fleet's accuracy bar. There, the two builds *phrased* the same fact differently;
here, on one prompt in ten, the compiled build reaches a **different final answer**, and the wrong one.
The cause is the same — the `cache_len` change and the compiler's own graph, not a quantizer width,
since both sides are `w8` — with a worse outcome on this prompt. The honest read is that a 0.6B model is
weak to begin with, and a different build of a weak model can flip a fact: the compiled graph's 1.38×
decode is real, and so is this failure mode. A caller who needs answer fidelity on a fact-sensitive task
should measure the specific prompts they care about, not assume the two builds agree.

**The same battery on the 1.7B and 4B.** Both of our larger compiled graphs are `cache_1024` builds
too — the 1.7B from the #595 native chain (md5 `0b41e627…`) and the 4B, 3.00 GiB — and each was run
through the identical ten prompts against its **shipped `w4` twin**, so each pair is width- and
scheme-matched exactly as the 0.6B pair was. The comparison is on the **answer fact** — the single
checkable short answer the prompt asks for — not on the whole sentence: the larger models elaborate,
and an exact-sentence match counts a bare "**4**" against "2 + 2 = **4**" as a miss when both reach
the same answer. Every row below reaches the one correct fact; the two builds differ only in how much
they say around it.

| prompt | ours `cache_1024` | shipped `cache_4096` | final answer matches? |
|---|---|---|---|
| `The capital of France is` | **Paris** | **Paris** | yes |
| `2 + 2 =` | **4** | **4** | yes |
| `The largest planet in the solar system is` | **Jupiter** | **Jupiter** | yes |
| `The chemical symbol for gold is` | **Au** | **Au** | yes |
| `How many continents are there?` | **7** | **seven** | yes |
| `The first month of the year is` | **January** | **January** | yes |
| `The capital of Japan is` | **Tokyo** | **Tokyo** | yes |
| `The boiling point of water in Celsius at sea level is` | **100 °C** | **100 °C** | yes |
| `The largest ocean on Earth is` | **Pacific Ocean** | **Pacific Ocean** | yes |
| `The chemical symbol for water is` | **H₂O** | **H₂O** | yes |

**10 / 10 on the 1.7B.** The only differences are surface: ours is usually the terser of the two
(shipped answers `How many continents` as "**seven continents**… " followed by the list, ours as
"**7 continents**…" followed by the same list), and on `The first month of the year is` ours emits a
stray leading `</think>` token before the answer — cosmetic, the answer that follows is `January`.
No prompt reaches a different fact.

| prompt | ours `cache_1024` | shipped `cache_4096` | final answer matches? |
|---|---|---|---|
| `The capital of France is` | **Paris** | **Paris** | yes |
| `2 + 2 =` | **4** | **4** | yes |
| `The largest planet in the solar system is` | **Jupiter** | **Jupiter** | yes |
| `The chemical symbol for gold is` | **Au** | **Au** | yes |
| `How many continents are there?` | **seven** | **seven** | yes |
| `The first month of the year is` | **January** | **January** | yes |
| `The capital of Japan is` | **Tokyo** | **Tokyo** | yes |
| `The boiling point of water in Celsius at sea level is` | **100 °C** | **100 °C** | yes |
| `The largest ocean on Earth is` | **Pacific Ocean** | **Pacific Ocean** | yes |
| `The chemical symbol for water is` | **H₂O** | **H₂O** | yes |

**10 / 10 on the 4B**, on the same terms — every difference is phrasing (ours spells `100 °C` out as
"100 degrees Celsius (°C)"; both write the 4B's answers at length), and no prompt lands on a
different fact.

**The synthesis, over the three compiled graphs.** On this battery the wrong-final-answer failure is
the **0.6B's alone**: 9/10, with the one flip (`How many continents` → **14**) that reaches a
different *and incorrect* answer; the 1.7B and the 4B each reproduce their shipped twin's final
answer on all ten. That is the shape a small-model artifact would have, and it is consistent with
capacity rather than the build: if the `cache_len` change and the compiler were flipping facts on
their own, the 1.7B and 4B — which are also `cache_1024` builds — would flip them too, and on these
prompts they do not. So the honest read is that the compiled graphs preserve the shipped graph's
answers on ten easy recall prompts, and the one failure we hold is on the weakest model, where a
different build of a model that is already shaky on a fact can tip it.

**What this does and does not establish.** It does establish that the compiled 1.7B and 4B match
their shipped twins on these ten single-fact prompts, and that the 0.6B's wrong-answer flip is not
reproduced at 1.7B or 4B. It does **not** establish general answer fidelity for the larger builds:
ten prompts is a probe, not a benchmark, and every one of them is short-fact recall (nothing harder
than `2 + 2` in the way of reasoning or multi-step math), so a divergence that needs a longer chain
to surface would not show up here. The row that matters for a deployment decision is the 0.6B's, and
the rule it implies stands for all three: measure the specific prompts a task depends on rather than
assuming a rebuild is faithful.

### Running the ladder in one process does not work, and that is a finding

Every number above was taken with **one process per graph**, and it has to be. Reusing one interpreter
to walk the ladder in order — open 0.6B, close, open 1.7B, close, … — **cannot load the 8B**, and the
reason is not what the disposition suggests.

Measured: after a graph has been **loaded and decoded on** and then closed, `xlm_destroy` releases the
`.hbm`'s own block but leaves a **residue in the carveout** — 1,059,782,656 B in 60 blocks after a
0.6B or 1.7B, and 1,361,772,544 B in 76 blocks after a 4B or 8B (the residue does not grow from 0.6B
to 1.7B, and grows in one step at the 4B). A graph that was only *opened and closed* with no decode
leaves nothing — the residue needs an inference. The residue is stable: it is still there 20 s later,
so it is not a release that has not happened yet.

That residue is enough to fail an 8B, and the failure is **fragmentation, not capacity**:

```text
[Model] Can not open .../Qwen3-8B_..._w4_..._corenum_4_4.hbm    (makes it look like a bad path)
  hbrt4_mem/src/unified.rs:156: Cannot malloc bpu memory with length 5703561320 bytes
    { len: 5704908800 }
  Fail to do ION_IOC_ALLOC(ret=Cannot allocate memory)!
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
```

The `hbDNNInitializeFromFiles failed` line above it is a red herring the SDK prints when an open fails
for a *missing* file too — the real cause is two lines further down. What the pool actually looks like
after a 4B decode and close:

| | value |
|---|---|
| carveout total | 10,737,418,240 B |
| free (total) | **9,375,645,696 B** — *more* than the 5,704,908,800 B an 8B needs |
| largest **contiguous** free run | **5,166,792,704 B** — *less* than an 8B needs |
| other free runs | 4,048,945,152 B, 159,907,840 B |

An 8B's `.hbm` is one contiguous allocation, and no single free run is big enough for it, so the load
fails while 9.38 GB sits free. The control that makes this unambiguous: into that **same** fragmented
pool, a 0.6B and then a 1.7B both **load fine** (they fit in the 5.17 GB run), and only the 8B fails.
So it is the largest graph that the residue's fragmentation excludes, not capacity in general — and a
fresh process, which starts from an empty pool, loads the 8B without complaint.

This is why the ladder's method is a process per graph. It is the same method the pre-switch ladder
used (`batch.sh` spawned `one.py` per run), so the two are comparable — but the reason is now known
rather than incidental, and it is worth knowing: **a long-lived process that serves several models in
turn accumulates pool residue, and a later large model can be refused by it.** A single-graph process
— what `serve` and `run` actually are — is unaffected.

## The runtime knob space

The `.hbm` is AOT-compiled, but the **runtime** has knobs — and it is worth knowing, before tuning,
that none of them buys decode speed on this board. The space is what the SDK reads at load, found by
`strings` over `oellm_runtime/lib` plus the four demo `run_*.sh` scripts:

- **`HB_DNN_USER_DEFINED_L2M_SIZES`** — the L2m working-memory split, one size per BPU core (read by
  `libdnn.so`; its `[Plan]` log prints the per-node requirement and any allocation failure). Every
  demo script sets `6:6:6:6`.
- **`HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM`** — a cap on the BPU cores the backend may use (`libhbucp.so`).
- **`HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM`**, **`HB_UCP_SCHEDULE_THREAD_SET_AFFINITY`**,
  **`HB_UCP_SCHEDULE_PRIORITY`**, **`HB_UCP_CPU_PROCESS_THREAD_SET_AFFINITY`** — UCP scheduler threads
  and their CPU affinity and priority.
- **`bpu_core`** in the config JSON — which cores the graph binds to (see the
  [deployment recipe](../architecture/s600_native_chain.md#running-this-artifact-on-the-board)).

There is no config key that trades accuracy for speed, either. The keys `libxlm.so` parses are
`hbm_path`, `tokenizer_dir`, `bpu_core`, `model_type`, `enable_thinking`, `enable_multi_turn` and
`use_sequence` (which reorders the sampler chain); nothing sets a token cap, a temperature or a warm-up,
because the sampler is read from the tokenizer directory and the graph is fixed. Memory and cores are
the whole of it.

### The sweep

One knob at a time from the default (`6:6:6:6`, `bpu_core [0,1,2,3]`), three greedy runs each,
canonical prompt, on three graphs. Cells are decode t/s (run 2 of 3); **refused** means the graph did not
run (the SDK's own refusal, not a crash); **every accepted cell produced the byte-identical greedy text
of its graph's default row**, so no setting changed the answer.

| runtime setting | 0.6B shipped | 1.7B shipped | 1.7B ours |
|---|---|---|---|
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` (default) | 82.9 | 67.7 | 87.6 |
| `HB_DNN_USER_DEFINED_L2M_SIZES=8:8:8:8` | refused | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=4:4:4:4` | refused | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=7:7:7:7` | refused | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:4` | refused | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:2:2` | refused | refused | refused |
| `bpu_core [0,1,2,3]` (default) | 82.9 | 67.7 | 88.2 |
| `bpu_core [0,1]` | refused | refused | refused |
| `bpu_core [0]` | refused | refused | refused |
| `bpu_core [0,1,2,3,0,1,2,3]` | 82.9 | 67.7 | 88.2 |
| `bpu_core [3,2,1,0]` | 82.9 | 67.7 | 87.6 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM` unset (default) | 82.9 | 67.7 | 87.6 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=4` | 82.9 | 67.5 | 87.6 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=2` | refused | refused | refused |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=1` | refused | refused | refused |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=8` | 82.9 | 67.7 | 88.2 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=1` | 82.9 | 67.7 | 87.6 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=8` | 83.8 | 67.7 | 87.6 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=14` | 82.9 | 67.7 | 87.6 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=28` | 82.9 | 67.5 | 87.6 |
| `HB_UCP_SCHEDULE_THREAD_SET_AFFINITY=1` | 83.8 | 67.7 | 87.6 |
| `HB_UCP_SCHEDULE_PRIORITY=1` | 83.8 | 67.7 | 88.2 |
| `HB_UCP_CPU_PROCESS_THREAD_SET_AFFINITY=1` | 82.9 | 67.7 | 87.6 |
| `HB_UCP_SCHEDULE_PRIORITY=-1` | 82.9 | 67.7 | 88.2 |

The three columns are three different artifacts, so the number to read is the *within-column* delta,
and it is zero. Every accepted setting sits inside the run-to-run spread: 1.7B
ours 87.6–88.2 t/s, 1.7B shipped 67.5–67.7, 0.6B 82.9–83.8 (the widest, 1.1%). **No runtime setting
on this board beats the vendor default, and the default is what the docs and the deployment recipe
already use.** That is the honest result: the knobs move validation, not throughput, and there is
nothing here to micro-optimize.

Two refusals are the useful part, because they are walls that a caller will otherwise discover badly:

- **`bpu_core` and the core cap are the same wall.** The graph was compiled for four cores, so a config
  bound to `[0]` or `[0,1]` — or `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=1|2` — fails the prefill with the
  core-count mismatch documented in
  [the chain page](../architecture/s600_native_chain.md#the-first-build-result). The only accepted
  spellings are four cores' worth: `[0,1,2,3]`, a reversed `[3,2,1,0]`, the redundant
  `[0,1,2,3,0,1,2,3]`, and a cap of 4 or 8 (both meaning "no fewer than four").
- **The L2m split is a hard window, not a dial.** Below it, the prefill node's requirement is unmet —
  the SDK prints `required l2 memspace info: [6263808, 6259712, 6161408, 6263808]` (per core, bytes)
  and refuses: 5.97 MiB/core is still short, 6.00 accepts. Above it, the allocation fails —
  `Allocate l2M memory failed, size: 7340032` at 7 MiB/core and up. So the accepted band is **about
  5.974 MiB to just under 7 MiB per core**, and `6:6:6:6` is the vendor's round value just
  above the floor (27 KiB of headroom) and comfortably under the ceiling. It is not a tuned number, but
  it is also not a wrong one.

### The same sweep on the 4B and the 8B

The table above was measured before the `balanced` switch, on the three graphs that ran in `cpu_first`;
the large graphs it could not reach were never swept. They were on 2026-10-10. Same method — one knob
at a time from the default (`HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`, `bpu_core [0,1,2,3]`), canonical
prompt, greedy, three runs per cell, one process per cell (the ladder's
[process-per-graph rule](#running-the-ladder-in-one-process-does-not-work-and-that-is-a-finding) —
env knobs are read at process start, so a cell has to be a fresh process to take effect anyway). Cells
are decode t/s of run 2; the spread is over all three runs.

| runtime setting | 4B | 8B |
|---|---|---|
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` (default) | 42.7 | 29.7 |
| `HB_DNN_USER_DEFINED_L2M_SIZES=7:7:7:7` | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=8:8:8:8` | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=5:5:5:5` | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:4` | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:4:4` | refused | refused |
| `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:8` | refused | refused |
| `bpu_core [0,1,2,3]` (default) | 42.7 | 29.7 |
| `bpu_core [0,1,2,3,0,1,2,3]` | 42.7 | 29.7 |
| `bpu_core [3,2,1,0]` | 42.8 | 29.7 |
| `bpu_core [0,1,2]` | refused | refused |
| `bpu_core [0,1]` | refused | refused |
| `bpu_core [0]` | refused | refused |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM` unset (default) | 42.8 | 29.7 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=4` | 42.9 | 29.7 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=8` | 42.8 | 29.7 |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=2` | refused | refused |
| `HB_UCP_ENABLE_BPU_BACKEND_CORE_NUM=1` | refused | refused |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=1` | 42.8 | 29.7 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=8` | 42.8 | 29.7 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=14` | 42.8 | 29.7 |
| `HB_UCP_TASK_SCHEDULE_COMMON_PROCESS_THREAD_NUM=28` | 42.9 | 29.7 |
| `HB_UCP_SCHEDULE_THREAD_SET_AFFINITY=1` | 42.8 | 29.7 |
| `HB_UCP_SCHEDULE_PRIORITY=1` | 42.8 | 29.7 |
| `HB_UCP_SCHEDULE_PRIORITY=-1` | 42.8 | 29.7 |
| `HB_UCP_CPU_PROCESS_THREAD_SET_AFFINITY=1` | 42.7 | 29.7 |

**The result is the small graphs' result again, and this closes the S600 perf question for the large
models: no runtime setting moves them either.** The fourteen accepted cells per graph span **42.67–42.88
t/s** (4B, 0.5%) and **29.65–29.75 t/s** (8B, 0.3%) — one band apiece, with no cell standing apart, so
there is nothing to pick. The `=4` cap's 42.83 mean is the nominal high on the 4B and 29.66 the low on
the 8B; both are inside the noise of the default's own three runs, and reporting either as a win would
be reporting the spread. Every accepted cell produced the **same greedy text as its graph's default
row** — 4B sha256 `63ef9253ce67b4d7`, 8B `e229830a3fb396a2`, the same ids the ladder records — so, as
on the small graphs, no accepted setting changed the answer.

The two walls are the same two walls, and the large graphs sharpen both:

- **The core count must be exactly four.** `[0]`, `[0,1]`, `[0,1,2]` and a backend cap of 1 or 2 all
  fail `hbUCPSubmitTask` for the prefill with the same message the small graphs hit, here verbatim:
  `The number of BPU cores set in the backend should be the same as the number of cores in the actual
  model compilation, given: …1, compiled model bpu core num: 4`. Reversing the list (`[3,2,1,0]`) or
  repeating it (`[0,1,2,3,0,1,2,3]`) is accepted and changes nothing — it is an *equality* check on the
  count, not a set or an order.
- **The L2m split is a hard window, and on the 4B it has no headroom at all.** The prefill node's
  requirement is `required l2 memspace info: [6291456, 6291456, 6291456, 6291456]` for the 4B and
  `[6164480, 6164480, 6164480, 6164480]` for the 8B — **exactly 6 MiB and 5.88 MiB per core**. So below
  it the prefill is refused (`L2 memory not enough` on any core short of 6 MiB), and above it the
  allocation itself fails (`Allocate l2M memory failed, size: 7340032` at 7 MiB and up, `8388608` at
  8 MiB). The accepted band is therefore **[6.0, 7.0) MiB per core** — and `6:6:6:6`, the vendor
  default, is the **floor** of that band, not a round value inside it: the 4B needs all six megabytes
  on every core, so unlike the small graphs (which had 27 KiB of slack below 6 MiB) there is no setting
  below the default that could have worked, and nothing above it fits. It is both the only value and
  the right one.

A note on what this does *not* cover: the large graphs appear to want a **larger** L2m split than the
small ones — 6 MiB/core against the small 0.6B's 5.97 MiB — but the hardware's allocation ceiling (just
under 7 MiB/core) is the same for both, so the window `[6.0, 7.0)` is narrower on the 4B than anything
the small sweep saw. There is no split between 6 and 7 to test at this granularity, and the SDK's
allocator only takes whole megabytes; the conclusion stands as a negative rather than an untested gap.

## The 4B / 8B ceiling — APPLIED

**The board's pool is raised, and 4B and 8B load.** The refusal below is what the board did in the
`cpu_first` mode it booted in, where `ion_carveout` — the pool the whole `.hbm` is loaded into — is
**2.00 GiB**. The SDK ships a supported tool, `hb_switch_ion.sh`, that moves the board to `balanced`
(**10.00 GiB**) and on to `bpu_first` (**18.00 GiB**); `balanced` was applied and the board rebooted
**2026-10-10**, `bpu_first` followed on **2026-10-13**
([the second switch](#a-second-switch-bpu_first-applied-2026-10-13)), and the large graphs initialize
and generate (see [the ladder](#measured-ladder) and
[the large models after the switch](#the-large-models-after-the-switch)). So the ceiling was a *mode*, not a vendor limit on
the model size, and it is now gone on this board.

**Before → after, measured.** The reboot moved the pool, and the cost is general RAM — the reservation
grows by exactly the pool's growth:

| | `cpu_first` (before) | `balanced` (applied) |
|---|---|---|
| `ion_carveout` (DT `reg`) | `41 40000000 0 80000000` = 2.00 GiB | `41 40000000 2 80000000` = **10.00 GiB** |
| dmesg `Memory:` | reserved 8.3 GiB | `48703360K/66871168K available`, **18167808K reserved** |
| `MemTotal` | 55.48 GiB | **46.48 GiB** |
| dmesg reserved | 8.3 GiB | **17.32 GiB** |
| 4B (3.10 GiB) | `RESOURCE_EXHAUSTED` | **loads, 42.6 t/s** |
| 8B (5.31 GiB) | `RESOURCE_EXHAUSTED` | **loads, 29.6 t/s** |

The board is healthy: it came up, reached userspace, and `MemAvailable` is 43 GiB with the two large
graphs still unloaded. The 9 GiB the pool took is the `+8 GiB` of `ion_carveout` plus `+1 GiB` of
`ion_cma` the switch also moves; nothing else changed.

**The switch and its recovery are kept below**, because the mode is a boot-time device-tree setting and
a future board will boot `cpu_first` again. The backups are on disk —
`/boot/hobot/rdk-s600-mcb-v1p0.dtb.bak` and `~/rdk-s600-mcb-v1p0.original.dtb` (285763 B each) — and
`hb_switch_ion.sh default` restores, with the A/B `boot_b` slot and the serial console as the other
recoveries.

### What the refusal was

In `cpu_first`, 4B and 8B **refused to load**, and the refusal was the board's, not this tree's:

```text
Cannot malloc bpu memory with length 3326941192 bytes   # 4B
Cannot malloc bpu memory with length 5703561320 bytes   # 8B
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
  -> ion_alloc ret=-12 (ENOMEM)
  -> hbDNNInitializeFromFiles error code -400001
```

`hbDNNInitializeFromFiles` (**HBRT 4.7.5**, `hbrt4_mem::unified`) reads the whole `.hbm` into **one
contiguous ION buffer** and hands it to the BPU. Measured on the board with `hb_mem`'s own
`/sys/kernel/debug/ion/heaps/` accounting, holding 1.7B open: the `.hbm` lands as a **single
1,829,109,760-byte (`label hbm`) allocation in the `ion_carveout` pool**, which is **2.00 GiB**. The
4B graph needs a **3.10 GiB** contiguous allocation and the 8B **5.31 GiB**, both larger than the 2 GiB
pool — which is the whole of the refusal.

> **Correction to an earlier reading of this page.** 384 MiB was attributed here to `bpu_region` and
> called the pool that overflows. That was wrong: the `.hbm` is not `bpu_region`, and enlarging it
> would not have helped — see [what the pools are](#what-the-pools-are) below.

**This is not our Python.** Three independent checks:

- The SDK's **own** demo binary (`oellm_runtime/examples/llm_demo/llm`) on the same 4B/8B configs
  fails identically, while it initializes 0.6B/1.7B.
- `hrt_model_exec model_info` — the SDK's own tool — reports the same `HBRT4_STATUS_RESOURCE_EXHAUSTED`.
- The 4B/8B `.hbm` md5s match the SDK's published `md5sum.txt`, so the artifacts are not corrupt.

`cli.py` / `xlm.py` surface the delegate's refusal as a clean message and `rc 1`, with no traceback.

**No *runtime* knob moved it.** Every env var and config key below was tried and none changed the
`cpu_first` refusal — because the refusal is the pool's *size*, a device-tree setting, not a runtime
one. The size has its own supported lever — the `hb_switch_ion.sh` mode switch
([below](#how-to-resize-it-the-sdk-ships-the-switch)) — which is a *different* kind of thing from these
knobs; the table is what does **not** work. Measured, all through the SDK's own binary so none of our
code is in the path:

| Knob tried | Values | Effect on 4B |
|---|---|---|
| `HB_DNN_USER_DEFINED_L2M_SIZES` | `6:6:6:6`, `0:0:0:0`, `2:2:2:2`, `1:1:1:1`, `12:12:12:12` | none — identical `ion_alloc -12` |
| `bpu_core` (corenum) | `[0]`, `[0,1]`, `[0,1,2]`, `[0,1,2,3]` | none — identical |
| HBRT memory-mode env var | — | no such variable exists (`strings` over `libxlm.so` / `libhbrt4` / `libhbipm` show only `HBTL_*` diagnostics) |

Corenum cannot help because the refusal happens at `hbDNNInitializeFromFiles`, **before** any core is
assigned. The L2m split cannot help because the failure is one contiguous allocation's size, not its
L2m partition — and the SDK's docs set `6:6:6:6` for every model regardless of size, so it is not a
size-dependent lever. None of these is the pool size, which is why none of them could have worked: the
lever that *does* change the pool size is the mode switch, and it is not a runtime flag.

The **Qwen3-VL-4B-Instruct language graph** was the one 4B-shaped lead that looked like it might
duck this, because the vendor ships it with `cache_1024` rather than `cache_4096` — the same
smaller-footprint shape our own recompile targets. It does not. Measured on the board 2026-10-09: the
file is **2,463,911,336 bytes (2.29 GiB / 2349.8 MiB)**, and `hrt_model_exec model_info` on it reports

```text
Cannot malloc bpu memory with length 2463911336 bytes: AllocError { len: 2465267712 }
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
  -> ion_alloc ret=-12 (ENOMEM)
  -> hbDNNInitializeFromFiles error code -400001
```

the same refusal as the plain 4B, over the same 2.00 GiB `ion_carveout`, by **301.8 MiB** (the loader
pads its request to 2351.1 MiB). The file's md5 is `c54dcf7686c0339a2307de10319a813c`, matching the
SDK's published `md5sum.txt`. So the `cache_1024` shrink is not enough — the weights alone exceed the
pool before any vision or embed artifact is touched — and this lead is closed: no 4B language graph,
plain or VL, loads on this board *in `cpu_first`*.

**Our own `cache_1024` 4B build refused in `cpu_first` too — and runs in `balanced`.** Built on the
x86 host and landed 2026-10-09, the file is **3,221,638,408 bytes (3.002 GiB)**, md5
`539775a7c9b596aa470895ad9fc6cf8e`. In `cpu_first`, `hrt_model_exec model_info` on it reported

```text
Cannot malloc bpu memory with length 3221638408 bytes: AllocError { len: 3222994944 }
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
  -> ion_alloc ret=-12 (ENOMEM)
  -> hbDNNInitializeFromFiles error code -400001
```

so the smaller cache *does* shrink the graph — 3.3269 GB shipped → 3.2216 GB ours, about **99 MiB**
less — but not enough for a 2.00 GiB pool: the weights alone need **3.00 GiB**. The number the bigger
pool had to exceed was therefore **3,222,994,944 B (3.002 GiB)** for 4B and **5,703,561,320 B
(5.31 GiB)** for 8B — both well inside `balanced`'s 10.00 GiB. That is the whole point of the mode
switch: the graph did not have to get smaller, the *pool* had to get bigger, and the SDK's own tool
does exactly that. It is applied, and all three large graphs load; see
[the large models after the switch](#the-large-models-after-the-switch).

### What the pools are

The board's DRAM carve-outs are fixed device-tree `reserved-memory` nodes (**not** kernel cmdline —
`/proc/cmdline` has no memory argument at all). Read from the live DT and confirmed against the boot
blob `/boot/hobot/rdk-s600-mcb-v1p0.dtb` (source `/boot/rdk-s600-mcb-v1p0.dts`), the model draws on
three 2 GiB ION heaps and ignores the one node this page used to blame. The sizes below are the
`cpu_first` values the refusal was measured under; in `balanced` `ion_carveout` is 10.00 GiB and
`ion_cma`/`ion_uncache` move up to make room, as the switch table above shows:

| DT node | `compatible` | Address | Size (`cpu_first`) | Role (measured) |
|---|---|---|---|---|
| `bpu_region@4300000000` | *(none)* | `0x408c000000` | **384 MiB** | `no-map`; **not an ION heap**, and a 1.7B load allocates nothing in it — **not involved in the .hbm load** |
| `ion_reserved@40C0000000` | `ion-pool` | `0x40c0000000` | 2.00 GiB | HBRT workspace — 1.7B uses **1.48 GiB** (activations + KV) |
| `ion_carveout@4140000000` | `ion-carveout` | `0x4140000000` | **2.00 GiB** (10.00 in `balanced`) | **the `.hbm` weights** — 1.7B = one 1.70 GiB `hbm` buffer; **the binding limit** |
| `ion_uncache@400000000` | `ion-uncache` | `0x4200000000` | 2.00 GiB | per-core BPU scratch — 1.7B uses ~0.17 GiB |

A full Qwen3 load therefore consumes **≈ 1.70 GiB (carveout) + 1.48 GiB (pool) ≈ 3.2 GiB**; the 2 GiB
`ion_carveout` is what a 3.10 GiB 4B `.hbm` could not fit into. `bpu_region` is a legacy `no-map`
reserve with no ION personality and no consumer we could observe.

### How to resize it: the SDK ships the switch

The pool sizes are device-tree `reserved-memory` nodes — but they are **not** a manual, unsupported
edit. The SDK **ships a supported tool** for exactly this:

```text
/usr/hobot/bin/hb_switch_ion.sh <bpu_first | cpu_first | balanced | default>
```

and the SDK's on-device setup page (`en/guide/env_install/arm_env.html`) documents `balanced` as the
setting for large models, verbatim:

> The device provides a `hb_switch_ion.sh` script to allocate memory space available for models. It is
> recommended to use the following commands to set the memory allocation to balanced mode.
> `# Apply balanced mode` / `hb_switch_ion.sh balanced` … `# Reboot for changes to take effect` /
> `reboot` … **Failure to execute this command may result in the inability to load Large Language
> Models on the edge side.**

For the S600 the three modes set `ion_carveout` (the `.hbm` pool) to:

| Mode | `ion_carveout` | Fits |
|---|---|---|
| `cpu_first` (the board's boot default) | **2.00 GiB** | 0.6B, 1.7B — not 4B/8B |
| `balanced` (applied 2026-10-10, superseded 2026-10-13) | **10.00 GiB** | 4B (3.00 GiB) **and** 8B (5.31 GiB) |
| `bpu_first` (**applied on this board 2026-10-13**) | **18.00 GiB** (19,327,352,832 B) | every graph on this page, including the 8B `w8` (9.75 GiB), at the cost of general RAM |

**The `bpu_first` row used to read 17.93 GiB; it is 18.00 GiB.** The switch script's own comment calls
the pool "18360MiB" — the same transposed-digit typo as `balanced`'s "10480MiB", corrected on this page
[below](#the-balanced-map-was-proved-without-booting-first) — while the reg cells it writes are exact:
`42 0 4 80000000` is 4·2³² + 2³¹ = **19,327,352,832 B = 18,432 MiB = 18.00 GiB**, and the live pool the
kernel actually reserves measured exactly that ([the applied record](#a-second-switch-bpu_first-applied-2026-10-13)).

So the S600's 4B/8B support was **one supported command plus a reboot** away. The board boots
`cpu_first`, whose 2.00 GiB pool is what refused the graphs — nothing about the model size, the SDK
build, or our code. This is *not* the manual DTB edit this page used to describe; `hb_switch_ion.sh` is
the vendor's own tool, it auto-detects the board, backs the DTB up, and reverts on any `fdtput`
failure.

**The switch is APPLIED, and the reboot succeeded.** `hb_switch_ion.sh balanced` was run and the board
rebooted 2026-10-10; it came up healthy with `ion_carveout` at 10.00 GiB and `MemTotal` at 46.48 GiB
(the before/after table is [above](#the-4b-8b-ceiling-applied)), and 4B, 8B and our compiled 4B all
load and generate. The offline well-formedness proof that made that reboot a one-shot is kept
[below](#the-balanced-map-was-proved-without-booting-first), because the mode is a boot-time setting a
future board will have to move again.

### A second switch: `bpu_first`, applied 2026-10-13

**On 2026-10-13 the board moved again, `balanced` → `bpu_first`, for the 8B `w8` build** — at 9.75 GiB
held at load it would sit at 97.5% of `balanced`'s 10.00 GiB, so the raise is what gives it working
headroom (its row is [in the ladder](#measured-ladder)). `sudo /usr/hobot/bin/hb_switch_ion.sh
bpu_first` ran clean (`rc 0`), the DTB went md5 `6ce6e801c7dd5103553a3a9db36e9224` →
`d599b1a1be952ba947e0545f88259c15`, and the reboot came back healthy. Measured:

| | `balanced` (2026-10-10) | `bpu_first` (applied 2026-10-13) |
|---|---|---|
| `ion_carveout` (DT `reg`) | `41 40000000 2 80000000` = 10.00 GiB | `42 0 4 80000000` = **18.00 GiB** |
| live carve-out (kernel reserve log) | 10,737,418,240 B | **19,327,352,832 B = 18,432 MiB = 18.00 GiB** |
| `free -m` total | 47,599 MB | **39,407 MB** — a drop of **exactly 8192 MB**, the pool's own +8 GiB |
| `MemTotal` | 46.48 GiB | **38.48 GiB** |

The `free -m` delta is the second check on the size, and it reads to the byte: a 17.93 GiB pool would
have moved 7.93 GiB of general RAM, and the board moved **8.00**. The live start addresses are *not*
the DTB's `bpu_first` start addresses — the boot chain packs the regions sequentially in node order,
honoring the DTB **sizes** exactly (the same rule reproduces the `cpu_first` and `balanced` maps) — so
the number to trust is the size, and it is 18.00 GiB.

**Nothing regressed with the mode.** The 1.7B `w8` control, re-run under `bpu_first`, decodes
**65.38 / 65.38 / 65.40 t/s** — its ladder row's 65.4 — with byte-identical text, and the 8B `w4`
`cache_4096` build loads clean in **7.81 s** holding 9,427,156,992 B (8.78 GiB, 48.8% of the 18.00 GiB
pool), decoding 29.77–29.81 t/s. Only the pool *fractions* moved, because the pool is what changed:
the 1.7B costs 16.3% of 18.00 GiB, where its ladder row's 29.3% is against 10.00.

**Rollback, updated.** The script's backup file on disk is now the factory `cpu_first` table
(`/boot/hobot/rdk-s600-mcb-v1p0.dtb.bak`, md5 `35ae26dac46500abe140d1d59b669ce8`), so
`hb_switch_ion.sh default` restores **`cpu_first`** and consumes the backup — it is *not* a way back to
`balanced`. The way back to `balanced` is `hb_switch_ion.sh balanced`; the four independent recoveries
below all still apply.

### The balanced map was proved without booting first

Everything that did **not** need a reboot was run first, on a **copy** of the DTB, so the actual boot
was a one-shot with a proven artifact. **No file under `/boot` was written for the proof, and the live
device tree was not touched** — the write and the reboot came later, once the user authorised them.

Rather than hand-copy the values, the **real** `/usr/hobot/bin/hb_switch_ion.sh` was exercised against a
sandbox: the live DTB was copied to `/tmp/s600_ion_dryrun/boot/hobot/`, and a copy of the script had its
one hard-coded prefix (`/boot/hobot/`) repointed into that sandbox — the only line changed, because the
script has no input-path override, and the sandbox is what keeps it from ever seeing `/boot`. Running it
there produced the `balanced` regs the script itself writes; `fdtget -t x` reads them back:

| Node | `cpu_first` (live) | `balanced` (produced) | Decoded |
|---|---|---|---|
| `ion_reserved` | `40 c0000000 0 80000000` | `40 c0000000 0 80000000` | 2.00 GiB — unchanged |
| `ion_carveout` | `41 40000000 0 80000000` | **`41 40000000 2 80000000`** | **10.00 GiB** |
| `ion_cma` | `41 c0000000 0 40000000` | `43 c0000000 0 80000000` | 2.00 GiB — moved up |
| `ion_uncache` | `42 0 0 80000000` | `44 40000000 0 80000000` | 2.00 GiB — moved up |

The reg cells are `[addr_hi addr_lo size_hi size_lo]`, so `ion_carveout`'s `0x2 0x80000000` is
2·2³² + 2³¹ = **10,737,418,240 B = 10,240 MiB = 10.00 GiB**. (The script's own comment calls it
"10480MiB" — a transposed-digit typo, the same family as `bpu_first`'s "18360MiB"
[above](#a-second-switch-bpu_first-applied-2026-10-13); the cells are exact.) Only **three** of the
four nodes actually change: `ion_reserved` is already at its `balanced` value in `cpu_first`.

**Well-formedness, checked against the *measured* map.** The boot DTB's own `/memory` node lists only
1.77 GiB — U-Boot patches the real map in at boot — so the proof uses the live map instead: **63.77 GiB
present** (three windows; the large one is `0x4080000000..0x4ffffff000`) and **55.48 GiB `MemTotal`**
usable. Against that:

- **In RAM.** All four `balanced` regions fall inside the window `[0x4080000000, 0x4ffffff000)`. The
  four-region pool run is `[0x40c0000000, 0x44c0000000)` — **16.00 GiB** — well inside a 63.77 GiB
  window.
- **No overlap.** The four ION regions are contiguous with **zero gap** between them, and clash with
  **none** of the other `reserved-memory` nodes (`bpu_region`, the `vpu*_ddr` block, `optee`, the
  `pcie*` ranges, … — all 36 checked).

So the change is **well-formed with zero risk in the artifact** — the numbers cannot be wrong. What was
left *unproven* at that point was only whether the kernel boots with the larger reservation. **It does:**
the reboot was done, and the board came up with the reservation applied and `MemTotal` at 46.48 GiB
([before/after](#the-4b-8b-ceiling-applied)).

A smaller variant of the same idea is also on disk — `/tmp/s600_carveout_prep/` holds a 4 GiB candidate
with a `dtc` round-trip that differs in exactly three `reg` lines (md5 `35ae26dac46500abe140d1d59b669ce8`
for its original). It is superseded by the vendor's `balanced`: 4 GiB reaches 4B but not 8B, while
`balanced`'s 10 GiB reaches both.

### Where the bootloader gets the DTB

This was the open question; it is now answered. The S600 does **not** use the plain S100 `/boot`
files. The boot chain is a signed `miniboot` firmware (`SBL`/`spl`, packaged under
`/lib/firmware/rdk/miniboot/stable/debug/img_packages/`) that runs U-Boot, which:

1. runs A/B slot selection (`ab_select_cmd = btype ab_select bootslot`), reading the active slot from
   the SoC's always-on (`aon`) area — the live `hobotboot.slot_suffix=_a` in `/proc/cmdline` is the
   result;
2. runs `sysboot` against **`extlinux/extlinux.conf`** on the **current-slot boot partition**, which
   `/etc/hb-fstab` mounts at `/boot` (`/dev/block/platform/by-name/boot_cur` → `boot_a` = `/dev/sda12`).

So the file to edit **is** `/boot/hobot/rdk-s600-mcb-v1p0.dtb`. The board-specific label is
`drobot-s600-rdk-v1p0-kernel` — built from the board flags `hobotboot.socname=S600`,
`board.hwname=rdk`, `board.ver=V1P0` — whose lines are `lantin /lantinhv` and
`domufdt /hobot/rdk-s600-mcb-v1p0.dtb` (the `domu*` = "domain U", a guest OS under the `lantinhv`
seL4 hypervisor). `spl.img`'s U-Boot env carries exactly these label names
(`drobot-s600-rdk-v1p0-kernel`, `drobot-s600-rdk-v0p1/v0p2-kernel`, …), so the loader selects the
label from the same flags. **The reversion is a file swap**: back up the DTB, replace it, and to
revert put the backup back — but a wrong map can prevent the board from reaching userspace, so the
swap must be done **with the serial console (`ttyS0`) attached and a copy of the original in hand**,
because that is the only way back without a vendor reflash.

**The one remaining unknown for the eventual boot:** `extlinux.conf`'s `default` line is a *leftover*
S100 label (`drobot-s100-rdk-v0p5-kernel`), which is inconsistent with an S600 booting the
`drobot-s600-rdk-v1p0-kernel` label, and none of the `domu*`/`lantin` keywords is standard syslinux.
The `pxe_label`/`fdt_feat` board flags and the label list compiled into `spl.img` strongly imply the
firmware constructs the label rather than reading the `default` literally — but that is an **inference
from strings**, not something observed at boot. It does not change *which file* holds the DTB (both
the S600 label and this page name `/hobot/rdk-s600-mcb-v1p0.dtb`), but a future session should
confirm at the U-Boot prompt which label is selected before flashing anything.

**Where the second DTB is.** The board boots slot `_a` (`boot_a` = `/dev/sda12`); the A/B scheme also
carries **`boot_b` = `/dev/sda13`**, an identical 120 MiB partition with its own `extlinux.conf` and DTB
set. That is a built-in second copy of the map, and the reason a bad edit to the live DTB is not a
one-way door — the recovery section below leans on it.

### The procedure that was run

**This is the sequence that was applied on 2026-10-10**, with the serial console (`ttyS0`) open. It is
kept as the record for the next board, which will boot `cpu_first` again:

```bash
# 1. the tool backs the DTB up itself (to rdk-s600-mcb-v1p0.dtb.bak) and reverts on any fdtput
#    failure; the explicit copy is belt-and-braces.
cp /boot/hobot/rdk-s600-mcb-v1p0.dtb /boot/hobot/rdk-s600-mcb-v1p0.dtb.pre-balanced
# 2. apply the supported mode change -- ion_carveout 2.00 -> 10.00 GiB (values proven above)
/usr/hobot/bin/hb_switch_ion.sh balanced
# 3. flush and reboot, watching ttyS0
sync && reboot
# 4. verify the kernel came up with the larger carve-out and less general RAM:
dmesg | grep -i "Memory:"     # measured: reserved 8.3 -> 17.32 GiB, MemTotal 55.48 -> 46.48 GiB
# 5. load a 4B/8B .hbm; success is hbDNNInitializeFromFiles returning 0, not RESOURCE_EXHAUSTED
```

Step 4 verified on the rebooted board: `Memory: 48703360K/66871168K available (… 18167808K reserved)`
and `MemTotal: 48741632 kB`, exactly the ~9 GiB fall predicted — and step 5 verified three times over
(4B, 8B, and our compiled 4B, all loading).

**Recovery, if it does not come up** — four independent ways back, which is what makes a reboot-needing
change low-risk here:

1. **`hb_switch_ion.sh default`** — the tool's own restore: it copies the `.bak` it made back over the
   live DTB, then `sync && reboot`, and the board is back in `cpu_first`. This needs the board to reach
   a shell, which is why the serial console matters.
2. **The A/B `boot_b` slot** (`/dev/sda13`) holds an identical `extlinux.conf` + DTB set; if `boot_a`
   will not reach userspace, `ab_select` can be pointed at `_b` from the serial U-Boot prompt, and
   `boot_b` is untouched by the switch.
3. **Serial console** (`console=ttyS0,921600n8`) — a stopped boot is diagnosable and recoverable here,
   which is what makes the `.bak` swap in (1) reachable at all.
4. **The on-board USB-DFU miniboot toolchain** — `/lib/firmware/rdk/miniboot/stable/{debug,release}/`
   carries a `xmodem`/USB-DFU loader (SoC USB VID `3652`); a full firmware reflash from the host is the
   vendor's own factory path if all else fails.

With all four in hand, the switch is a file swap under `/boot` plus a reboot, reversible from the serial
console — not a one-way door.

**The other path — a smaller graph — is measured not to be enough, and it was not needed.** A `.hbm`
recompiled with a shorter context (`cache_1024`, as the SDK's VLM 7B graph uses) *does* shrink the graph
— ours came out 3.2216 GB against the shipped 3.3269 GB — but it still needs 3.00 GiB of weights, so it
still refused in `cpu_first`. That recompile chain is scoped in
[the S600 native compile chain](../architecture/s600_native_chain.md); the mode switch was the shorter
road to the same place, and the compiled 4B runs now that the pool fits it.

## The large models after the switch

Measured 2026-10-10 on the rebooted board, same plank as the ladder: canonical prompt **"The capital of
France is"**, greedy `generation_config.json`, `pocketllm`'s `XlmEngine` with `bpu_core [0,1,2,3]` and
`HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`, three runs per graph. Each graph was driven through the
**SDK's own tool first** (`hrt_model_exec model_info`, step 1 of the load test) and then through
**`pocketllm run --device horizon`** on the shipped `qwen3_4b_config.json` / `qwen3_8b_config.json`,
which answers the prompt with `rc 0`. (The numbers below are the first measurement of the large graphs
and agree with [the canonical ladder](#measured-ladder) to within run-to-run spread; where they differ
in the last digit, the ladder's row is the one to quote, because it was measured in one session beside
the small models rather than in a session of its own.)

**All three initialize.** The SDK's own `hrt_model_exec model_info` — the tool that refused them before
— reports `Load model to DDR` for each, with no `Cannot malloc bpu memory` and no
`HBRT4_STATUS_RESOURCE_EXHAUSTED`:

| Graph | Size | `model_info` | Load | Prefill | Decode | 3 runs identical |
|---|---|---|---|---|---|---|
| shipped 4B | 3,326,941,192 B | **initializes** (5646 ms) | 6.46–7.34 s | 2381–2415 t/s | **42.6 t/s** | yes (`63ef9253ce67b4d7`) |
| shipped 8B | 5,703,561,320 B | **initializes** (6876 ms) | 7.66–7.70 s | 1932–1939 t/s | **29.6 t/s** | yes (`e229830a3fb396a2`) |
| ours `cache_1024` 4B | 3,221,638,408 B | **initializes** (5738 ms) | 6.32–6.38 s | 4339–4376 t/s | **50.0 t/s** | yes (`1b615e0a550e345a`) |

**The output is coherent, not garbage.** All three answer the canonical prompt with the fact asked for,
and all three answer a second, third and fourth prompt correctly too — so the coherence is not a
one-prompt artifact. Asked `The largest planet in the solar system is`, all three say **Jupiter**; asked
`2 + 2 =`, all three say **4**; asked for a Python string-reverse function, all three produce a working
slicing one.

```text
# shipped 4B, "The capital of France is"
…  </think>
The capital of France is **Paris**.

# shipped 8B, "The capital of France is"
…  </think>
The capital of France is **Paris**.

# our compiled 4B, "The capital of France is"
…  </think>
The capital of France is **Paris**. It is the largest city in the country and serves as the
political, cultural, and administrative center of France. …
```

Each answer is preceded by the model's own ` thinking…</think>` reasoning block, which is intact and
on-topic rather than repeated or truncated — the delegate's `xlm_result_t` hands back the whole text
and `pocketllm run` prints it as-is. The final answers are the fact asked for in every case.

**Throughput falls as the model grows, and that is the whole result:** 87.0 (0.6B) → 69.4 (1.7B) →
42.6 (4B) → 29.6 (8B) t/s decode on the shipped graphs. The board's per-token cost is the BPU work,
which scales with the parameter count, so a 4B decodes at roughly 0.6× the 1.7B and an 8B at roughly
0.7× the 4B. Nothing here is a surprise, and nothing here needed the recompile: the **pool** was the
blocker, and moving it restored all four sizes at once.

**Our compiled 4B is the fastest of the three large results** — 50.0 t/s decode and 4339–4376 t/s
prefill against the shipped 4B's 42.6 / 2381–2415, a **~17% decode / ~1.8× prefill** edge over its
shipped twin — the same direction as, though a smaller decode margin than, the compiled 1.7B's ~27%
over its own shipped graph (88 vs 69.4 t/s). But this is a *bonus*, not the reason to compile: the recompile was pursued as
the path to 4B and it was not one (3.00 GiB of weights would not fit 2.00 GiB). Its edge is the smaller
`cache_1024` graph, and it comes with the same caveat as the compiled 1.7B — it is a **different
build-time quantization** of the same `Qwen3-4B` weights, so its reasoning text differs from the
shipped graph's even where both converge on the right answer.

The artifacts are **not shipped**: the compiled 4B lives on the x86 build host
(`/mnt/data1/oellm_models/build_out_4b/`) and was copied to `/tmp/` on the board for this measurement,
so a reboot clears it. Its byte size and md5 in the table above are the identity to rebuild it from —
`docs/architecture/s600_native_chain.md` is the build recipe.

## Determinism

Reproducibility on this path comes entirely from the **tokenizer directory's `generation_config.json`**,
not from a flag: the delegate builds its sampler from that file when the model is loaded.

- **Greedy** (`temperature 0`, `do_sample false`) is byte-identical across runs. Measured over three
  runs each: 0.6B sha256 `a154da66b9307a50`, 1.7B sha256 `bc75540e622e9551` (CLI runs), and — added
  2026-10-10 through the same delegate — 4B sha256 `63ef9253ce67b4d7`, 8B sha256 `e229830a3fb396a2`,
  our compiled 4B sha256 `1b615e0a550e345a`. The three runs per graph agreed byte for byte, so the
  determinism contract holds at every size, not only the two small ones.
- **The SDK's default is not deterministic.** The shipped `generation_config.json` uses
  `temperature 0.6`, `top_k 20`; same-prompt runs differ in the reasoning block and in surface form.
  A page that claimed "runs are reproducible" without naming the file would be wrong.

So a deterministic deployment ships a greedy `generation_config.json` beside the tokenizer. That is a
file the caller edits, which is why `--seed` is refused rather than honoured — see below.

### The intermittent hang: a lost wake-up inside the delegate's decode path

**One run in ten never returns.** It is not a BPU fault and not our graph — the process is alive,
sleeping, holding the model, and the board is idle. **A long generation occasionally deadlocks inside
the vendor's `libxlm.so`, and the process must be killed** (there is no timeout and no cancellation in
the delegate's `infer`). Measured on the enumeration prompt `List every integer from 1 to 200 …`,
greedy, one process per run, `bpu_core [0,1,2,3]`, a 25-second no-token stall as the hang detector:

| graph | runs | hangs | rate |
|---|---|---|---|
| our compiled 8B (`cache_1024` w4) | 60 | **6** | 1 in 10 |
| shipped 8B (`cache_4096` w4) | 20 | **1** | 1 in 20 |
| our compiled 1.7B (`cache_1024` w4) | 20 | 0 | — |
| our compiled 8B, `The capital of France is` (short) | 15 | 0 | — |

The two 8B graphs are the same weights recompiled by us and shipped by the vendor, and **both hang**,
so this is a property of the delegate, not of our build (the 1.7B's 0-of-20 is too few to distinguish
1-in-20 from never). The short-prompt batch is clean, which is the expected shape — a hang needs a long
decode. **It is not at a fixed step:** the four captures died after 90, 107, 538 and ~317 generated
pieces, i.e. anywhere in the generation.

**What the process looks like while hung** (captured live, before the kill, with `gdb` and `/proc`):

- `State: S (sleeping)`, `wchan=futex_wait_queue` — **not** a `D`-state I/O wait, and not spinning
  (CPU idle). The BPU core-utilization `ratio` for all four cores is **0**: the accelerator is idle.
- 35 threads, all parked in futex waits. The main thread's stack is
  `python3 → ctypes → xlm_infer → XlmImpl::XlmInfer → XlmImpl::XlmInferInner → std::thread::join`
  — the delegate spawned an internal worker for the decode and is now **joining it forever**.
- That worker's stack is `XlmImpl::XlmInferInnerThread → XlmInferDecode → LlmModel::DoDecode →
  DoBatchDecodeImpl → DnnModelInfer → std::__atomic_futex_unsigned_base::_M_futex_wait_until` with a
  **null timeout** (`__val=2147483648`, the sign bit set — the classic `std::atomic::wait` encoding).
- **No syscall is in flight on any thread** (`strace` attaches and shows every thread already blocked in
  `futex(...)`). So it is not blocked in a work-loop — it is parked on a condition that is never
  signalled. That is a **lost wake-up / missed notification** between the delegate's decode worker and
  whatever the BPU/DNN completion path is supposed to signal.

So the failure is a **delegate-internal condition-variable deadlock in the DNN inference path**, reached
roughly once per ten long decodes, that leaves the process unschedulable-to-completion with the hardware
idle. Everything above is reproducible from the board; the fix is inside the vendor's `.so` (its BPU
completion signalling), which this tree does not own. **The mitigation a caller can apply today is a
generation timeout** — the delegate offers none (`supports_cancellation` is false for the same reason),
so a wrapper that gives up and restarts the process is the only defence until the vendor fixes it.

## Accuracy against the 2080ti oracle

The project's accuracy target is agreement with the 2080ti C-engine oracle. This path is **text-in /
text-out and takes no token-id input**, so the comparison is made by feeding the oracle the exact ids
the delegate's chat template produced and diffing the greedy continuations.

**A single prompt flattered this path; a batch of ten is not yet a distribution.** The first
measurement used one prompt ("The capital of France is") and read as "identical for 14 tokens, then a
near-tie". Ten diverse prompts × both sizes said **0/10 fully identical at either size**, with the
agreement a **prefix** and not the whole answer. That was more honest than one prompt, but ten is
still too thin to carry a divergence *statistic* — a category could disagree completely and never
appear in a ten-prompt draw, and a single large-margin prompt read as an outlier instead of a tail.
The corpus is now **50 prompts across 7 categories**, measured by the same method, and reported as a
distribution rather than a min/max/median:

| Size | Fully identical | Shared prefix min / p25 / median / p75 / max | Splits with a ≥5-logit gap |
|---|---|---|---|
| 0.6B (w8 vs oracle `q4_k_m`) | **0 / 50** | 4 / 8 / **18** / 26 / 59 | **1 / 50** |
| 1.7B (w4 vs oracle `Qwen3-1.7B-Q4_K_M`) | **0 / 50** | 2 / 7 / **11** / 17 / 38 | **9 / 50** |

(Both rows are the **shipped `cache_4096`** graphs. The cache was then swept with the pipeline held
fixed — our own `1024`/`2048`/`4096` builds — and it turns out **not** to move agreement; the pipeline
does. That row sits in [its own subsection](#a-cache-row-on-the-accuracy-table-the-cache-is-not-what-moves-agreement)
below, because it changes how the 1.7B row above should be read: it is the vendor's graph. The same
paired comparison at **0.6B and 8B** — the two other sizes with both a shipped and an ours-built graph —
shows the gap is real at every size but shaped differently, in
[the subsection after it](#the-provenance-gap-holds-at-the-other-sizes-and-it-is-not-uniform).)

The distribution is itself the finding, and it **changes the headline on the 1.7B row.** Ten prompts
said "most divergences are near-ties, but not all"; fifty say that is true of the **0.6B** — 39 of 50
splits sit under one logit, and only one (`multi_02`, the EN→FR translation) reaches past five, at
**7.91** — and **not of the 1.7B**, where **9 of 50** splits are five logits or wider, reaching
**20.49** on `code_06` (a Python word-count function). The one large-margin case the ten-prompt batch
found (row 05, 14.06) was therefore not an outlier to explain away; it was a visible point on a tail
the ten-prompt draw was too small to show the shape of.

The corpus and every capture live in `~/scratch/ref/harness/` on the board: the two shipped rows
(`artifacts/0.6B-w8.*`, `artifacts/1.7B-w4.*`), the four cache-sweep captures of the 1.7B
(`artifacts/1.7B-w4-{ours-c1024,c2048,ours-c4096}.*` plus the shipped `1.7B-w4.*` at its default
`cache_4096` as the sweep's fourth point), the seven provenance-pair captures
(`artifacts/0.6B-{ours,shipped}-w8.*`, `artifacts/0.6B-w4.*`, `artifacts/4B-{ours-c1024,shipped-c4096}.*`,
`artifacts/8B-{ours,shipped}-w4.*`), the `w8` builds (`artifacts/1.7B-w8-ours-c1024.*`,
`artifacts/4B-w8-ours-c1024.*`), the calibration-corpus build (`artifacts/1.7B-ours-c1024-calib.*`)
and the 8B `cache_4096` survey (`artifacts/8B-ours-c4096.*`). The `f16`-oracle re-scoring is **not**
on the board: it lives on the oracle host under `~/s600_f16/artifacts/` (0.6B:
`0.6B-{ours,shipped}-w8-f16`, `0.6B-w4-f16` and `q4_k_m-vs-f16`) and `~/s600_f16b/artifacts/`
(`{1.7B,4B,8B}-{ours,shipped}-w4-f16` with `{1.7B,4B,8B}-q4km-vs-f16`, the whole-table run) — the same
registry, scored against a second oracle. The whole measurement is now
**one command** — see [the harness](#the-harness-one-command) below, which is what makes the vendor's
`w4` 0.6B a single invocation when it lands. The single-prompt artifacts remain at
`~/scratch/ref/parity/s600_parity.json`.

### The distribution, by category

Every category diverges on **every** prompt — there is no category the board reproduces fully at
either size — so the useful question is *where* it parts and *how confidently*, not *whether*. The
shared-prefix histogram (50 prompts, both sizes) and the per-category breakdown:

| shared prefix | 0–4 | 5–9 | 10–19 | 20–29 | 30–39 | 40+ |
|---|---|---|---|---|---|---|
| 0.6B `w8` | 3 | **15** | 12 | 11 | 5 | 4 |
| 1.7B `w4` | 10 | 12 | **17** | 8 | 3 | 0 |

| category | n | 0.6B prefix min/med/max | 0.6B gap med/max | 1.7B prefix min/med/max | 1.7B gap med/max |
|---|---|---|---|---|---|
| factual | 8 | 8 / 12 / 36 | 0.54 / 2.21 | 2 / 20 / 24 | 1.85 / 5.05 |
| arithmetic | 7 | 4 / 29 / 59 | 0.70 / 1.28 | 4 / 9 / 26 | 1.05 / 2.41 |
| code | 6 | 12 / 31 / 56 | 0.26 / 0.63 | 7 / 29 / 38 | **7.14 / 20.49** |
| multilingual | 7 | 9 / 18 / 45 | 0.82 / **7.91** | 3 / 14 / 21 | 0.82 / 5.30 |
| longform | 6 | 4 / 9 / 30 | 0.44 / 2.24 | 4 / 5 / 13 | 2.69 / 5.09 |
| instruction | 6 | 6 / 7 / 27 | 0.32 / 1.54 | 2 / 11 / 15 | 4.37 / 6.85 |
| ambiguous near-tie | 10 | 6 / 18 / 23 | 0.58 / 2.70 | 7 / 15 / 30 | 1.55 / 4.56 |

Two readings fall out and neither is the old one:

- **The 1.7B's widest disagreements are on structured output** — `code` and `instruction` are where
  its gaps live (median 7.14 and 4.37 against 0.4–1.5 everywhere else), while its *prefixes* there
  are long. It agrees on the scaffold (def/return, a numbered list, JSON braces) and then splits
  hard on a token the two quantizations rank very differently. `code_06` diverges after **38** shared
  tokens with a **20.49**-logit gap — the largest in either batch.
- **The 0.6B's one wide split is multilingual**, and it is the mirror image: a short prefix (45 on
  `multi_02`) and a 7.91 gap. Otherwise the width-confounded 0.6B is the *more* near-tie-dominated
  of the two rows — the opposite of what "w8 vs q4_k_m is doubly mismatched" would predict, which is
  itself a caution against reading the width confound as a severity knob.
- **The deliberately ambiguous prompts are the most benign**, as expected — a near-tie the two sides
  resolve differently is what an open-ended first token *is* — and the `ambiguoustie` median prefix
  (0.6B 18, 1.7B 15) is above the overall median at both sizes.

### A cache row on the accuracy table: the cache is not what moves agreement

The 1.7B row above is the **shipped `cache_4096`** graph — its capture names the SDK `.hbm`, not ours —
the same clean graph the loop discussion contrasts against. So a `cache_2048` row does not, by itself,
isolate the cache: it would set our `2048` build against the vendor's `4096` and change the pipeline at
the same time. The cache was therefore measured with the pipeline held fixed — our own `cache_1024`,
`cache_2048` and `cache_4096` builds, same corpus, same config, same oracle (`libpocketllm.so` over
`Qwen3-1.7B-Q4_K_M`), with the prompt ids verified identical across all four captures:

| 1.7B `w4` graph | shared prefix min/p25/med/p75/max | histogram (0–4 / 5–9 / 10–19 / 20–29 / 30–39 / 40+) | splits ≥5 logits | gap max |
|---|---|---|---|---|
| ours `cache_1024` | 0 / 2 / 6 / 13 / 21 | 20 / 17 / 10 / 3 / 0 / 0 | 23 / 50 | 15.31 |
| **ours `cache_2048`** | 0 / 2 / 6 / 9 / 25 | 20 / 19 / 9 / 2 / 0 / 0 | 22 / 50 | 24.34 |
| ours `cache_4096` | 0 / 2 / 6 / 11 / 26 | 19 / 17 / 11 / 3 / 0 / 0 | 26 / 50 | 24.34 |
| shipped `cache_4096` | 2 / 7 / 11 / 17 / 38 | 10 / 12 / 17 / 8 / 3 / 0 | 9 / 50 | 20.49 |

**The cache is not the axis; the pipeline is.** Our three builds — which differ *only* in `cache_len` —
are nearly indistinguishable from one another (median 6 at all three, ~20 of 50 prompts under five shared
tokens, per-prompt divergence points coinciding on 24/50 for `1024` against `2048`). The vendor's `4096`
graph is a different distribution entirely: median 11, eleven of fifty past twenty tokens against our
nine or ten, and **nine** wide-margin splits against our **twenty-two to twenty-six**. So growing the
cache from 1024 to 4096 — the change that closes the loop and lengthens the list — does **not** move the
agreement with the oracle. The variable that does is provenance: on the same weights, the same width, the
same cache and the same prompt ids, **our build agrees with the oracle markedly less than the vendor's**.

This is a new axis on the table, and it does not contradict the loop result — the two measure different
things. The cache governs whether the reasoning block *closes* and how far a list *runs*; it does not
govern how closely the next token matches the oracle. And the accuracy row the page has quoted all along
was the vendor's graph, which is why it read as the better one — not because `4096` is more accurate, but
because that graph was not built by us. The self-built graph carries a quieter, second cost beside the
loop: on this corpus it is the *less* oracle-like of the two, at every cache.

### The provenance gap holds at the other sizes, and it is not uniform

A single size cannot say whether "our build agrees less" is a compile-pipeline fact or a 1.7B one. The
same paired comparison was run at the three other sizes where **both** a shipped and an ours-built graph
exist on the board, each pair at a **matched width** so the pipeline is the only axis (the cache differs,
`1024` vs `4096`, but the 1.7B sweep above shows the cache does not move agreement):

| size | ours (build) | shipped (build) | ours prefix min/p25/med/p75/max | ours hist | ours ≥5 | ship prefix | ship hist | ship ≥5 |
|---|---|---|---|---|---|---|---|---|
| 0.6B `w8` | ours `c1024` | shipped `c4096` | 2 / 7 / **13** / 23 / 64 | 6 / 13 / 12 / 12 / 3 / 4 | **0** | 4 / 8 / **18** / 26 / 59 | 3 / 15 / 12 / 11 / 5 / 4 | 1 |
| 0.6B `w4` | ours `c1024` | — (no vendor `w4`) | 0 / 0 / **4** / 6 / 22 | 32 / 10 / 4 / 4 / 0 / 0 | **14** | — | — | — |
| 1.7B `w4` | ours `c1024` | shipped `c4096` | 0 / 2 / **6** / 13 / 21 | 20 / 17 / 10 / 3 / 0 / 0 | **23** | 2 / 7 / **11** / 17 / 38 | 10 / 12 / 17 / 8 / 3 / 0 | 9 |
| 1.7B `w8` | ours `c1024` | — (no vendor `w8`) | 2 / 9 / **17** / 25 / 79 | 6 / 8 / 18 / 13 / 2 / 3 | 7 | — | — | — |
| **4B `w4`** | ours `c1024` | shipped `c4096` | 2 / 6 / **8** / 16 / 35 | 8 / 21 / 15 / 4 / 2 / 0 | **16** | 4 / 11 / **18** / 24 / 42 | 3 / 6 / 19 / 15 / 5 / 2 | 2 |
| 8B `w4` | ours `c1024` | shipped `c4096` | 2 / 7 / **11** / 20 / 61 | 5 / 14 / 15 / 8 / 3 / 1 | 2 | 4 / 14 / **23** / 33 / 88 | 2 / 5 / 15 / 12 / 8 / 8 | 2 |

**The gap holds at every size — our build is the less oracle-like one in all four vendor pairs — but its
magnitude and its shape are not the same.** The table carries two further our-own rows that have no
vendor counterpart to pair against: the `0.6B w4` build (the SDK publishes no `w4` 0.6B), whose own
comparisons are within 0.6B and are read below, and the `1.7B w8` build (no vendor `w8` 1.7B exists
either), whose result is recorded in full [below](#the-17b-at-w8-above-the-floor-against-truth). Paired per prompt (ours minus shipped on the *same*
prompt, so a shared easy prompt cannot flatter either side):

| size | prompts | ours shorter | equal | ours longer | median Δ | Δ range |
|---|---|---|---|---|---|---|
| 0.6B `w8` | 50 | 15 | **24** | 11 | **0** | −38 … +35 |
| 0.6B `w4` | 50 | **46** | 2 | 2 | **−12** | −59 … +8 |
| 1.7B `w4` | 50 | **30** | 12 | 8 | **−2** | −36 … +12 |
| **4B `w4`** | 50 | **33** | 8 | 9 | **−4** | −38 … +27 |
| 8B `w4` | 46 | **32** | 8 | 6 | **−8** | −69 … +36 |

- **0.6B is the weakest case, and it nearly reverses on individual prompts** — the paired difference is
  a coin-flip (24 of 50 identical to the token, only 15 shorter to 11 longer, median Δ exactly 0). The
  distribution is only modestly worse for us: median 13 against 18, and **zero** wide-margin splits
  against the shipped graph's single 7.91 (`multi_02`). So at the smallest size "our build is less
  oracle-like" is *true on the aggregate and near-invisible on any one prompt*.
- **A second 0.6B row, and it is the widest deficit on the table**: our own `w4` 0.6B build, matched in
  *width* to the `q4_k_m` oracle where the `w8` row above was not, agrees **less** than our `w8` does,
  not more — the opposite of what "match the oracle's width" predicts. Its median prefix is **4** (against
  the `w8` row's 13 and the shipped graph's 18), **32 of 50** prompts share under five tokens, and it
  carries **14** wide-margin splits against the `w8` row's **0** — a new, heavy tail on a row that had
  none. Paired within 0.6B it is the shorter side on **41 of 50** prompts against our `w8` (median Δ
  **−9**) and **46 of 50** against the shipped graph (median Δ **−12**). It is a distinct shape from the
  other four — not a coin-flip, a step-margin drift, or a widened tail on a drifted median, but a
  **near-collapse**: the whole distribution slides toward zero *and* grows a wide-margin tail. Whether
  this is a `w4` effect, a scheme effect (the same `leap_llm` int4 the 1.7B and 4B use), or something
  about this particular build is **not established**; the plain fact on the table is that our-w4 agrees
  less with the `q4_k_m` oracle than our own w8 does.
- **8B is the widest median gap and the most lopsided**: 32 of 46 prompts shorter, median Δ **−8**
  (11 against 23), with the shipped graph reaching a **88**-token shared prefix that ours never approaches
  (max 61). But — unlike the 1.7B — the **wide-margin split count is equal** (2 and 2); the 8B's
  disagreement is a *systematic step earlier*, not a different distribution of confidence.
- **4B is the middle case, and the first one that is both wide and drifted**: 33 of 50 prompts shorter
  at a moderate median Δ **−4** (8 against 18, so the median itself is only a four-token step), but
  the tail is heavy — **16** wide-margin splits against the shipped graph's **2**, eight times the rate
  and the largest *asymmetry* of the four pairs. The wide splits are not one model's noise: the shipped
  graph keeps `code_01`'s 9.69 exactly and adds a single 9.90 (`code_04`), while ours spreads 16 splits
  from 5.47 to 14.25 across code, instruction, factual, multilingual and long-form — every category but
  the two smallest. So the 4B sits between the two extremes: like the 1.7B it *widens the tail*, and
  unlike the 8B its drift is not merely a uniform step earlier.
- **1.7B is the outlier in the tail, not the mechanism**: its median gap (−2) is smaller than the 8B's,
  but it carries **23** wide-margin splits against the shipped graph's **9** — two and a half times the
  rate. That is the 1.7B's signature from the row above, and it is the one place ours does not merely
  drift earlier but *disagrees more confidently*.

So the answer to the question is **neither "1.7B-only" nor "uniform"**: our build is the less oracle-like
graph at all four sizes (a **systematic** compile-pipeline effect), while the *shape* of the deficit is
size-specific — a coin-flip at 0.6B `w8`, a step-margin drift at 8B, a widened tail at 1.7B and 4B, and
a fourth shape on the new `0.6B w4` row: a near-collapse toward zero shared prefix carrying its own
wide-margin tail. The honest statement is that provenance is a real, every-size axis on this table,
weakest where the model is smallest.

**And a narrower precision did not buy agreement here — it cost it. This was an OPEN question; the
`f16` oracle has now been run and it is answered below.** The `0.6B w4` row was built to test whether
matching the oracle's width narrows the gap; it does the opposite, and by a wide margin (our-w4 agrees
less than our own `w8`, at every summary statistic). The tempting inference — "so agreement with a
quantized oracle is not monotonic in *our* precision" — cannot be settled against a `q4_k_m` oracle,
because that oracle is itself a quantization and the two sides read different numeric scales.

#### The `f16` oracle: the gap is real against truth, and a floor separates w8 from w4

The same 50-prompt captures were re-scored against a **near-lossless `f16`** checkpoint, so `f16`
stands in for truth and a `q4_k_m`-oracle row can be read against it. The 0.6B `f16`
(`qwen3-0.6b-f16.gguf`, 1,509,347,584 B) already existed; for the three larger sizes there was none on
the host, so genuine `f16` GGUFs were **built** from the checkpoints' own HF safetensors
(`convert_hf_to_gguf.py --outtype f16`): `qwen3-1.7b-f16.gguf` (4,069,679,360 B), `qwen3-4b-f16.gguf`
(8,051,285,280 B) and `qwen3-8b-f16.gguf` (16,388,044,128 B). Each is `general.file_type = 1` with no
quant tensors (1.7B 198 f16 + 113 f32; 4B 253 + 145; 8B 254 + 145) — genuinely F16, not a re-labelled
quant. The captures are oracle-independent — they hold the prompts' token ids and the board's
continuation ids — so this re-scores the existing captures without re-running the board; each size's
`q4_k_m`-vs-`f16` floor is a direct C-engine-vs-C-engine pass on the same prompt ids. **"≥5-logit
splits" counts prompts whose first divergence had a top-2 logit margin of at least 5 logits on the
truth (`f16`) side** — the same "wide-margin split" the tables above use. It is a *margin* count, not
a shared-prefix count, so it does **not** agree with the histogram's 5+ bucket and must not be read as
one.

**0.6B:**

| graph | vs `f16` (truth) min/p25/med/p75/max | hist (0–4/5–9/10–19/20–29/30–39/40+) | ≥5-logit splits | Σ prefix |
|---|---|---|---|---|
| shipped `w8` | 4 / 10 / **20** / 36 / 133 | 1 / 10 / 13 / 7 / 9 / 10 | 1 | **1327** |
| ours `w8` | 2 / 8 / **16** / 28 / 70 | 3 / 11 / 14 / 12 / 6 / 4 | 0 | **973** |
| **`q4_k_m` vs `f16` (the floor)** | 4 / 8 / **20** / 27 / 56 | 4 / 10 / 9 / 19 / 5 / 3 | 0 | **995** |
| ours `w4` | 0 / 0 / **4** / 6 / 22 | 33 / 11 / 2 / 4 / 0 / 0 | 15 | **240** |

Two readings fall out, and the first one **truth-anchors** the provenance rows rather than retracting
them:

- **The gap holds against truth.** Shipped `w8` reaches a **1327** total shared prefix against `f16`
  where ours reaches **973**; the shipped graph is the closer-to-truth one, exactly as the `q4_k_m`
  table said. What the `f16` run corrects is the *reading*, not the finding: the **`q4_k_m` oracle sits
  ~20 shared tokens from truth itself** (the floor row: median 20, Σ 995) — a GGML k-quant is not a
  near-truth reference — so the `q4_k_m` table **flattened our `w8`**, pinning it at the oracle's own
  error (ours-w8 Σ 973 ≈ floor Σ 995). Against truth our `w8` is closer to the oracle than the
  `q4_k_m` table could show; the `q4_k_m`-measured gap understates how far apart ours and shipped
  actually are.
- **The w4/w8 asymmetry is the load-bearing fact.** Our `w8` sits **at the floor** — its
  `q4_k_m`-oracle "gap" is largely the oracle's own error, not our build's. Our `w4` sits **well below**
  the floor (Σ **240** against the floor's **995**): it is *further from truth than the GGML 4-bit
  k-quant itself is*, a real and severe deficit, not an artefact of the quantized reference. So
  "our build is less oracle-like" is **precision-dependent**: true and severe for our `w4`, mostly the
  oracle for our `w8`.

The consequence is a rule for reading this whole section: **the `q4_k_m`-vs-`f16` floor (~20 median
shared tokens) is the yardstick, and any `q4_k_m`-oracle row must be read against it.** A row at the
floor is measuring the oracle's quantization error, not ours; a row below it is a genuine deficit.

#### The whole table, truth-anchored at every size

The same re-scoring was carried out at the three larger sizes, so the provenance table is now read
against truth everywhere rather than at the 0.6B alone. Each is the same shape: the shipped graph,
our graph (both `w4` here, the width the larger checkpoints ship at) and the size's own
`q4_k_m`-vs-`f16` floor, over the same 50-prompt corpus. The 1.7B, 4B and 8B tables also carry our own
`w8` builds — a separate axis, with the 1.7B's recorded
[below](#the-17b-at-w8-above-the-floor-against-truth) and the 8B's
[in its own subsection](#the-8b-at-w8-the-width-lever-falls-short).

**1.7B** (`n = 50`):

| graph | vs `f16` (truth) min/p25/med/p75/max | hist (0–4/5–9/10–19/20–29/30–39/40+) | ≥5-logit splits | Σ prefix |
|---|---|---|---|---|
| shipped `w4` | 2 / 8 / **14** / 19 / 38 | 6 / 11 / 21 / 10 / 2 / 0 | 6 | **724** |
| ours `w4` | 0 / 2 / **6** / 12 / 21 | 17 / 18 / 12 / 3 / 0 / 0 | **23** | **394** |
| **ours `w8`** | 2 / 12 / **19** / 27 / 89 | 2 / 5 / 19 / 13 / 5 / 6 | 2 | **1174** |
| **`q4_k_m` vs `f16` (the floor)** | 2 / 9 / **19** / 26 / 79 | 7 / 8 / 11 / 18 / 3 / 3 | 3 | **956** |

**4B** (`n = 50`):

| graph | vs `f16` (truth) min/p25/med/p75/max | hist (0–4/5–9/10–19/20–29/30–39/40+) | ≥5-logit splits | Σ prefix |
|---|---|---|---|---|
| shipped `w4` | 4 / 11 / **19** / 23 / 53 | 2 / 7 / 17 / 16 / 3 / 5 | 2 | **990** |
| ours `w4` | 2 / 6 / **8** / 17 / 32 | 9 / 21 / 15 / 4 / 1 / 0 | 12 | **536** |
| **ours `w8`** | 4 / 20 / **31** / 39 / 124 | 1 / 0 / 11 / 12 / 14 / 12 | **0** | **1656** |
| **`q4_k_m` vs `f16` (the floor)** | 4 / 17 / **25** / 47 / 96 | 1 / 4 / 12 / 12 / 7 / 14 | 2 | **1552** |

**8B** (each row on the prompt set it has, because a stall costs a row and the three sets differ: the
shipped graph and the floor are `n = 50`, our `w4` is `n = 46` — its four stalls `fact_07`, `long_04`,
`long_05`, `ambi_01` — and our `w8` is `n = 49` — one stall, `ambi_04`):

| graph | vs `f16` (truth) min/p25/med/p75/max | hist (0–4/5–9/10–19/20–29/30–39/40+) | ≥5-logit splits | Σ prefix |
|---|---|---|---|---|
| shipped `w4` | 4 / 16 / **21** / 33 / 67 | 1 / 3 / 18 / 14 / 9 / 5 | 2 | **1196** |
| ours `w4` | 2 / 7 / **12** / 20 / 71 | 6 / 11 / 17 / 6 / 5 / 1 | 4 | **729** |
| **ours `w8`** | 6 / 17 / **28** / 44 / 109 | 0 / 3 / 13 / 10 / 8 / 15 | **1** | **1619** |
| **`q4_k_m` vs `f16` (the floor)** | 4 / 19 / **34** / 53 / 119 | 1 / 4 / 9 / 6 / 12 / 18 | 0 | **1901** |

Three readings hold across all four sizes:

- **The gap holds against truth everywhere.** Shipped beats ours on total shared prefix against `f16`
  at every size — Σ **1327** vs **973** (0.6B), **724** vs **394** (1.7B), **990** vs **536** (4B),
  **1196** vs **729** (8B) — a gap of 354/330/454/467 tokens. The provenance finding is not an artefact
  of comparing against a quantized oracle; it is the same against near-lossless weights.
- **The floor grows with the model, and at the larger sizes it overtakes *both* graphs.** The
  `q4_k_m`-vs-`f16` floor is Σ **995** (0.6B), **956** (1.7B), **1552** (4B), **1901** (8B): a GGML
  k-quant is a near-truth reference only for the smallest checkpoint. Read as a ratio to the floor,
  shipped/floor is **1.33 / 0.76 / 0.64 / 0.63** and ours/floor is **0.98 / 0.41 / 0.35 / 0.38** at
  `w4` — so from 1.7B up **both the shipped graph and ours sit below the floor** (every `w4` row here
  does, and "below the floor" is *not* a property of our build alone). The `w8` rebuilds are the
  counter-current, and they thin out as the model grows: ours/floor reads **1.23** at 1.7B
  ([recorded below](#the-17b-at-w8-above-the-floor-against-truth)), **1.07** at 4B — the 4B's Σ
  **1656** sits +104 (+6.7%) above its floor — and only **0.85** at 8B, where the lever lifts the row
  **+890** (Σ 729 → 1619) and the floor still wins
  ([the 8B at `w8`](#the-8b-at-w8-the-width-lever-falls-short)). This is the strongest form of the
  0.6B point: the `q4_k_m` reference is good only where the model is small, and above it every graph
  on this table — the vendor's included — is closer to truth than the oracle is.
- **The 1.7B's wide-margin tail survives against truth.** Its `q4_k_m` row carried **23** ≥5-logit
  splits, the worst on the table; re-scored against `f16` it carries **23** again — the same count —
  while the size's own floor carries only **3** and the shipped graph **6**. The tail is therefore a
  real property of our 1.7B build, not the oracle's error: at this size ours does not merely drift
  earlier, it disagrees more confidently than either the reference quant or the shipped graph, and
  truth agrees. The 4B's 12 splits and the 8B's 4 are likewise above their floors' 2 and 0, and the
  8B `w8`'s single split sits between its `w4` sibling's 4 and the floor's 0. The 0.6B
  stays the exception in direction: there our `w4`'s 15 splits are the *only* thing above the floor,
  on a graph otherwise pinned to the floor itself, while the `w8` row sits at it. At the 1.7B the
  same test now reads in reverse: our `w8` build carries **2** wide-margin splits where our `w4`
  carries **23** — so the tail is a property of our 1.7B **at `w4`**, not of the 1.7B pipeline as such
  ([below](#the-17b-at-w8-above-the-floor-against-truth)).

#### The 8B at `w8`: the width lever falls short

The 8B is the third size we have rebuilt at `w8` (after the 1.7B and 4B), and the one where the lever
is not enough. Build provenance: the same compile chain as the other `w8` rebuilds
([the native compile chain](../architecture/s600_native_chain.md)) — `compile_hbo` 9604.9 s +
`link_models` 553.5 s, log clean. Output
`Qwen3-8B_language_chunk_512_cache_1024_w8_nash-p_corenum_4_4.hbm`, **9,078,659,624 B (8.46 GiB)**,
md5 `fec5669a018d9723eccef05491ba3bb2`, relayed 2080ti→Mac→board and md5-verified at all three ends
(re-verified on the board immediately before the capture). Captured 2026-10-13 as `8B-w8-ours-c1024` —
50 prompts, **one stall** (`ambi_04`, the [#650 hang](#the-intermittent-hang-a-lost-wake-up-inside-the-delegates-decode-path)
the harness watches for, on a prompt none of the `w4` captures stalled on) — and scored on the 2080ti
against `qwen3-8b-f16.gguf` (truth). Its rate row is in [the ladder](#measured-ladder): **20.5 t/s**
decode at **9.75 GiB** of pool, the one `bpu_first` row.

| 8B graph (vs `f16` truth) | Σ prefix | n |
|---|---|---|
| shipped `w4` | 1196 | 50 |
| ours `w4` | 729 | 46 |
| **ours `w8`** | **1619** | **49** |
| **`q4_k_m` vs `f16` (the floor)** | **1901** | **50** |

**The row is real progress — the highest Σ of any graph row on this table — and it still lands below
the floor.** Against truth it reads min/p25/med/p75/max **6 / 17 / 28 / 44 / 109**, histogram
**0 / 3 / 13 / 10 / 8 / 15**, **1** wide-margin split (`multi_07`, 5.15) and **0 / 49** fully
identical: **Σ 1619** — **−282 against the floor's 1901** and **+890 over our `w4`'s 729**. On the 45
prompts both share (the apples-to-apples cut, since a stall costs a row) the lift and the gap hold:
**719 → 1530** against the floor's **1751** there — a matched gap of **−221**, and the closest any
8B-family build on this page has come to this size's floor. The shape of the remaining gap, from both
directions:

- **Against its own `w4` sibling, a broad lift.** On the 45 shared prompts the `w8` is the
  longer-prefix side on **34** of them (equal 6, shorter 5), median Δ **+14**, range −19…+105 — the
  improvement is the row, not a handful of prompts.
- **Against the floor, short but close.** Paired per prompt the `w8` is shorter on **28 of 49**
  (equal 6, longer 15), median Δ **−3**, range −76…+79 — a step-earlier deficit that has almost
  closed, and a two-sided tail at that. At every percentile the distribution sits inside the
  floor's (median 28 vs 34, p75 44 vs 53, max 109 vs 119), which is the shape of "closer, not
  there".
- **The size keeps the page's verdict: below the floor.** At 8B every one of our builds — `w4` and
  now `w8` — agrees with truth less than the GGUF `q4_k_m` quantizer does. Note what that does not
  mean: at this size the floor is mostly the *oracle's own* error, so "below the floor" is not
  "worse than `q4_k_m`" — it is that the deficit is still visible *above* the oracle's error, which
  it no longer is at 1.7B (+218) or 4B (+104).
- **The corrupted token is gone from the corpus.** The `bunifuFlatButton` glitch — 12 occurrences in
  the `w4` `cache_1024` capture, 13 in the `cache_4096` one — appears **zero** times here (no token
  `95234`, no text hit, across all 49 completed rows), the first our-build 8B capture clean on the
  corpus (the shipped 8B's capture was already clean; the paragraph below carries the same count).
  The `1..N` enumeration battery has since been re-run at `w8` and the slip is **absent** there too —
  0 occurrences in 27 runs across the 1.7B, 4B and 8B
  ([the battery at `w8`](#the-1n-battery-at-w8-the-slip-is-gone-and-the-8b-fails-a-different-way)) —
  so the corrupted token is gone from the enumeration as well as the corpus. What that battery does
  find is the 8B's *other* long-generation failure, five runs of nine: the graph is not deployable
  behind a `cache_1024` window on long prompts even though its slip is gone.

**The 4B closes the table, and it is the one row still paired the confounded way.** The 4B is the last
size with a shipped-and-ours pair, so the provenance gap now rests on **all four** sizes rather than
three — but it is paired the *same* way as the others, ours at `cache_1024` against the vendor's
`cache_4096`, so the cache is a second axis in the row. Separating cache from build at 4B the way the
1.7B's and 8B's own `cache_4096` builds separate it would need **our** 4B rebuilt at `cache_4096`, and
**no such build exists**. Whether the 4B's 16-split tail would survive a cache-matched own build is
therefore **not measured for the 4B**, and not claimed.

**At the 1.7B the cache *is* now controlled out, and the gap survives it.** Our own 1.7B `cache_4096`
build — the rebuild that closes the `France?` loop — was captured on the same 50-prompt corpus and
scored against the vendor graph *at the same cache*, so cache length is the one variable this cell
pins. It is the matched-cache row the table did not have:

| 1.7B graph (all `w4`) | cache | vs `f16` (truth) min/p25/med/p75/max | ≥5-logit splits | Σ prefix |
|---|---|---|---|---|
| ours `w4` | 1024 | 0 / 2 / **6** / 12 / 21 | 23 | **394** |
| ours `w4` | 4096 | 0 / 2 / **6** / 11 / 26 | 23 | **377** |
| shipped `w4` | 4096 | 2 / 8 / **14** / 19 / 38 | 6 | **724** |
| **`q4_k_m` vs `f16` (the floor)** | — | 2 / 9 / **19** / 26 / 79 | 3 | **956** |

At **matched** cache (both `cache_4096`) our graph trails the vendor **377 vs 724** against truth —
**less than half** — so the provenance gap is the **build**, not the cache: the cache-mismatch confound
is removed and the gap stays. The sweep makes the point twice over. Our own `cache_4096` (Σ **377**) is
a hair *worse* than our `cache_1024` (Σ **394**) — a wash, not an improvement — so the deeper cache buys
the loop and not the corpus. Scored directly against each other at the same cache, the two graphs share
**379** tokens of leading continuation across the fifty prompts, with **13 of 50** diverging after
agreeing on only two tokens — structural from the start of the continuation, not a late drift.

The two axes stay separate, and this result does not overturn the loop finding. The cache **is** still
the lever for the `France?` runaway loop — our `cache_4096` closes it (229 pieces, a 40-char window
recursing twice) where `cache_1024` runs away — but that is the *loop*, a degenerate 2-token prompt. The
new result is narrower and different: the cache is **not** what closes the **corpus-agreement** gap, and
at matched cache our graph is still less than half the vendor's agreement with truth.

**The 8B's corrupted token is present in the capture and invisible in the agreement data.** The
`bunifuFlatButton` glitch — a single Qwen vocabulary token, id `95234`, that substitutes for a *digit*
(`1.5` → `1.bunifuFlatButton`, `75%` → `75.bunifuFlatButton`, `age > 18` → `age > .bunifuFlatButton`) —
appears **12 times across the 8B-ours capture** (the arithmetic, code, instruction and long-form rows);
it appears **zero** times in the shipped 8B's 50 prompts and zero times at 0.6B and 1.7B. It is the exact
signature the enumeration survey localized (`#24`/`#37`/`#56` in the `1..N` lists), now seen on a
different prompt family. **It does not show up in the agreement table** because in all twelve rows the
glitch token sits **after** the first divergence — 130 to 924 tokens after it, e.g. `code_05` diverges at
token 4 but slips at 603 — so it never *causes* the split the harness scores. It is a downstream artifact
of our 8B having already left the oracle's trajectory, which is why a divergence-only comparison misses
it: the corruption is real, it is ours, and it lives past the point the table stops looking.

**The `w8` rebuild's capture carries it zero times.** No token `95234` and no `bunifuFlatButton` in any
of its 49 completed rows, against the `w4` `cache_1024` row's 12 — the first our-build 8B capture clean
on the corpus, and the strongest single sign so far that width is the right lever for this near-tie
([the row and its scope](#the-8b-at-w8-the-width-lever-falls-short)). The shipped 8B's capture was
already clean, so this closes a gap between our `w4` builds and the vendor's graph; the `1..N`
enumeration was not re-run, so it does not close the question.

**The 8B's own `cache_4096` build now joins this table** and confirms the two axes are separate: run on
the same corpus, our 8B at the vendor's cache has the *same* wide 8B-style deficit (median Δ **−7**
against the shipped graph at the same cache; median Δ **0** against our own `cache_1024` graph), so the
cache is not an accuracy lever for the 8B either. The full row and the enumeration that came with it are
[in the survey below](#cache-versus-build-on-the-accuracy-table).

#### The 1.7B at `w8`: above the floor against truth

Every shipped-and-ours pair in the tables above is the width the checkpoints ship at (`w4` at 1.7B and up). Our pipeline
can also compile the same checkpoint at `w8` — the **1.7B `w8`** build varies exactly one flag against
our existing 1.7B `w4` `cache_1024` graph (`--w_bits 4 → 8`) — so the width is a variable the tables
could not see until it was built and run. It was: built clean, relayed, captured on the same 50-prompt
corpus, and scored against both oracles. **Against truth it is the first cell at this size at or above
the floor, and it sits 450 tokens above the vendor's `w4` `cache_4096` graph.**

Build provenance: `oellm_build --model_name qwen3 --march nash-p --w_bits 8 --chunk_size 512
--cache_len 1024 --prefill_core_num 4 --decode_core_num 4 --device cpu`, exited clean
(`Exit status: 0`, wall **6:25:33**). Output
`Qwen3-1.7B_language_chunk_512_cache_1024_w8_nash-p_corenum_4_4.hbm`, **2,484,199,624 B**, md5
`14c18c05975c4684474faacb9110397a`, relayed 2080ti→Mac→board and md5-verified at all three ends
(re-verified on the board immediately before and after the capture). Captured 2026-10-12 as
`1.7B-w8-ours-c1024` — **50/50 prompts, 0 stalls** — and scored on the 2080ti against
`qwen3-1.7b-f16.gguf` (truth) and `Qwen3-1.7B-Q4_K_M.gguf`.

| 1.7B graph (vs `f16` truth; our graphs `cache_1024`, the vendor's `cache_4096`) | Σ prefix |
|---|---|
| **ours `w8`** | **1174** |
| ours `w4` | 394 |
| shipped `w4` | 724 |
| **`q4_k_m` vs `f16` (the floor)** | **956** |

Against `f16` the `w8` row reads min/p25/med/p75/max **2 / 12 / 19 / 27 / 89**, histogram
**2 / 5 / 19 / 13 / 5 / 6**, **2** wide-margin splits (ours `w4`: 23; the floor: 3) and **0 / 50**
fully identical — **Σ 1174, +218 above the floor (+22.8%)**. Against `q4_k_m` the same capture
reads min/p25/med/p75/max **2 / 9 / 17 / 25 / 79**, histogram **6 / 8 / 18 / 13 / 2 / 3**, **7**
wide-margin splits, **Σ 906** — the row added to the `q4_k_m` table
[above](#the-provenance-gap-holds-at-the-other-sizes-and-it-is-not-uniform).

Three scope notes, because this row is easy to over-read:

- **The metric is token agreement against a near-lossless `f16` checkpoint taken as truth** — the
  agreed proxy for this board. The board exposes **no logits** (`xlm` returns text), so the project's
  **1e-3** bound has **no numeric form on this side**: this is not a 1e-3 logit match and must not be
  read as one.
- **The floor is the `q4_k_m` checkpoint's *own* agreement with truth** (Σ 956), not a clean-model
  ceiling: a graph *at* the floor is measuring the quantizer's error, not its own. `w8` at **1174**
  sits **above** it — the first cell at 1.7B to do so.
- **`w8` roughly doubles the packed weights, and at 8B it needs the raised pool.** The 8B `w8` build
  holds **9.75 GiB** at load — 97.5% of `balanced`'s 10.00 GiB, so it is deployed under the `bpu_first`
  pool (18.00 GiB, 54.2%) since the 2026-10-13 switch, not `balanced`. It exists, it runs, and it
  moves the 8B's agreement **+890** — while still sitting below the size's floor, unlike the 1.7B and
  4B rebuilds ([the 8B at `w8`](#the-8b-at-w8-the-width-lever-falls-short)).

Two readings, one of which revises a reading above:

- **The below-floor collapse is a property of `w4`, not of the size.** At 1.7B our `w4` sits below
  the floor (394 vs 956) exactly as it does at 4B and 8B, while our `w8` sits above it — and at 0.6B
  our `w8` sat at the floor. The w4/w8 asymmetry the 0.6B first showed holds at the 1.7B too.
- **The 1.7B's confident-wrong tail is a `w4` property.** The 23 wide-margin splits that made the
  1.7B the worst row on the truth-anchored table fall to **2** at `w8` — below the floor's own 3 —
  and the hardest category follows: `code`'s median shared prefix against truth goes from **2** at
  `w4` to **42** at `w8`.

A statistics convention, since Σ and "median" can disagree about which median: **Σ is the decision
axis on this page** (the floor comparison is what decides the goal), and the medians in the tables
are the harness's positional `prefs[n // 2]` over the 50 prompts. The textbook even-`n` mean-of-two
would read **18.5** on the `f16` row where the tables say **19**; both are correct under their own
definitions, and the tables keep the positional one.

### The width confound, and why only one pair can remove it

The two rows above do **not** confound width equally, and the weaker row is the 0.6B one:

| Pair | Board `.hbm` | Oracle GGUF | Width | Scheme |
|---|---|---|---|---|
| 0.6B | `w8` (int8) | `q4_k_m` | **mismatched** (8 vs 4) | mismatched |
| 1.7B | `w4` (int4) | `Qwen3-1.7B-Q4_K_M` | matched (4 vs 4) | mismatched |

So the 0.6B divergence could be **width or scheme**, while the 1.7B row — where the wide-margin
disagreements live (`code_06`'s 20.49; the ten-prompt batch's row 05 at 14.06) — isolates **scheme**,
because both sides are 4-bit. A same-width 0.6B pair would settle it, but **the vendor does not
publish one**, and neither does it publish the mirror-image lever (a `w8` 1.7B):

- **The manifest lists exactly one language `.hbm` per LLM size**, and the widths are not
  interchangeable — `resolve_model_nash-p.md` (in the SDK at
  `oellm_runtime/model/`) gives `Qwen3-0.6B → w8` and `Qwen3-1.7B → w4`, with no second row for
  either. Only the 4B lists both `w4` and `w8`.
- **The published `md5sum.txt` agrees**, listing exactly one `.hbm` under each of
  `Qwen3-0.6B/` and `Qwen3-1.7B/`.
- **The bucket itself agrees.** Probing
  `.../llm_s600/{1.0.0,1.0.2}/models/Qwen3-{0.6B,1.7B}/{w4,w8}/` across every plausible
  `chunk`/`cache` name, exactly two language files exist — the ones already on the board — and
  every other candidate returns **404** while the two shipped paths return **200**. There is no
  hidden `w4` 0.6B and no hidden `w8` 1.7B to fetch, so **no same-width pair can be built on the
  board**, and the width confound on the 0.6B row stands.

That is a finding, not a gap awaiting a retry: the removal of the confound is **vendor-gated**, and
the only path that would produce a `w4` 0.6B or a `w8` 1.7B `.hbm` is the same
[compile chain](../architecture/s600_native_chain.md) that a smaller 4B would need — built by us,
on x86-64, not fetched. Until then the honest reading is: **the 0.6B row mixes width and scheme; the
1.7B row isolates scheme; and the wide-margin divergences we have measured — all nine of them, and
the ten-prompt batch's row 05 — are on the row that isolates it.**

**A `w4` 0.6B now exists — but it is ours, not the vendor's, so it does not close the confound.** The
row added above (`artifacts/0.6B-w4.*`, md5 `935c299b9a3efa66318ce5b522793ebd`) is our own `leap_llm`
`w4` build of the same checkpoint, matched to the oracle in width. It is therefore the same-width
pair the paragraph above says the vendor does not publish — but pairing the *pipeline* this way does
not remove the *scheme* confound, because `leap_llm`'s per-channel affine int4 is still not
`q4_k_m`'s super-block k-quant. The width axis is now measured and it does not help (see the
`0.6B w4` bullet above); the scheme axis is untouched by this build and still open.

**Near-ties dominate the 0.6B row; they do not dominate the 1.7B.** The fifty-prompt gap
distribution says so directly: at 0.6B, **39 of 50** splits sit under one logit and only `multi_02`
reaches past five; at 1.7B, **9 of 50** are five logits or wider, the largest `code_06` at **20.49**
after 38 shared tokens. A twenty-logit margin is not rounding noise against the *same* weights — the
oracle scores its own next token twenty logits above the board's, so the two sides are reading
**different numbers**, which is the scheme mismatch below and not a tie the board lost. The ten-prompt
batch's single 14.06 case (the old row 05) reproduces here **exactly** — this engine puts 57.33 on its
choice against the board's 43.26 at the same context — and the fifty-prompt corpus shows it was a
point on a tail, not an outlier.

### What quantization the `.hbm` actually is

The `w4`/`w8` in the filename is the **`leap_llm` weight-bit count**, and the scheme is in the
toolchain we hold (the `oellm_build` wheels shipped in the SDK — see
[the native compile chain](../architecture/s600_native_chain.md)):

- **1.7B = `w4` → symmetric, per-output-channel, weight-only affine int4.** `FakeQuantLinear`'s
  4-bit path (`leap_llm/nn/modules/linear.py:66-78`) is `q_weight = clip(round(w / scales), -7, 7)`
  with **one fp scale per output channel**, `zeros = 0` — i.e. GPTQ-**style** per-channel
  symmetric int4, but with *no* GPTQ error compensation and *no* AWQ search (neither appears in the
  shipped `llm_compression` tree).
- **0.6B = `w8` → symmetric per-channel int8** via the `const_fake_quant(..., axis=0)` path.

**This is a different scheme from `q4_k_m`, on both axes that matter.** `q4_k_m` is asymmetric
6-bit sub-scales over 256-weight *super-blocks* (a mixed `q4_K`/`q6_K` file, so `attn_v`/`ffn_down`
are 6-bit); the `.hbm` is one scale per output *row*, uniformly 4-bit (or 8-bit). There is no
`q4_k`/`q6_k`/super-block code path anywhere in `leap_llm` or `hbdk4` — the scheme is not reachable
from `leap_llm` at all. So the board `.hbm` and the oracle's GGUF are **two different quantizers**
that happen to both be "4-bit", and a 14-logit disagreement is the expected consequence, not a bug
in either engine.

**The honest statement is therefore:** over fifty prompts the output is **coherent and on-topic** at
both sizes, but it is **token-identical to the oracle for no prompt at all** (0/50 at each size) —
only for a **prefix**, whose median is 18 tokens at 0.6B and 11 at 1.7B — and the 1.7B's splits are
**not** all near-ties: nine of its fifty carry a ≥5-logit margin, up to 20.49, which is the two sides
being **different quantizations** and not a coin-flip the board lost. This path cannot be compared to
1e-3 at the logit level — the delegate exposes no logits — so the token agreement above is the whole
of the evidence, and the page says so rather than implying more.

### The harness: one command

The measurement is a single invocation now, not a re-derivation, so a new graph (above all the
vendor's `w4` 0.6B that would remove the width confound) is one command to score:

```bash
python3 parity.py <graph.hbm> --config <config.json> --label 0.6B-w4 \
    --oracle-ssh lvyufeng --gguf /mnt/data1/models/qwen3-0.6b-q4_k_m.gguf
```

It captures the 50-prompt corpus on the board, ships the artifact to the oracle host, replays each
prompt on the **C engine**, and prints the table — identical / shared-prefix / divergence token per
side / the oracle's top-2 gap at that step. `parity.py`, `corpus.py`, `capture_agreement.py`,
`compare_agreement.py` and the artifacts are under `~/scratch/ref/harness/` (its `README.md` is the
method). Two conventions are pinned there and matter: the oracle is **`libpocketllm.so`, not
llama.cpp** — the C engine reproduces the page's `57.33 / 43.26 / 14.06` exactly while llama.cpp on
the same GGUF shares zero tokens and, worse, its own greedy output moves with the context window it
is asked for — and the oracle is handed the ids of the delegate's **ChatML-templated** text, both
sides greedy, so agreement is measured on ids.

### Answer quality over the fifty (S600 side)

The corpus is not scored for correctness — the oracle is a quantization of the same weights, not a
truth source — but it does surface the model-quality defects a small quantized checkpoint has:

- **The 1.7B closes its reasoning block on all 50** and answers on-topic at every length; its longest
  answers are the `code` prompts, which diverge widest (above) but stay coherent.
- **The 0.6B closes its block on 49 of 50.** The exception is `code_02` ("reverse a string without
  slicing"): it generates **4065 tokens** — the longest in either batch, and longer than every other
  0.6B answer by 4× — **never closes ` thinking`**, and falls into a repetition loop on the phrase
  *"must not use slicing. So, the solution…"* (that 40-character window recurs seven times in the
  tail). It is the same failure mode the old ten-prompt set put on the 1.7B's `1/7` prompt and it is
  kept in the record rather than dropped: a degenerate loop is a strong oracle test precisely because
  it is hard to reproduce by accident, and here the board produced it where the oracle might not.

The earlier ten-prompt answer-quality table and that 1.7B `1/7` loop remain in
`~/scratch/ref/parity_batch/`.

## Sampling and caps the CLI enforces

The delegate's sampler is **fixed when the `.hbm` is loaded**, so the sampling flags cannot be applied
on this path. The CLI's rule ([#571](https://github.com/lvyufeng/PocketLLM/pull/571),
[#572](https://github.com/lvyufeng/PocketLLM/pull/572)) is to refuse a value that would have changed
the answer, and accept one that merely spells out the default:

| Flag | On `--device horizon` |
|---|---|
| `--temperature 0`, `--top-k 0`, `--top-p 1` | **accepted** — naming the delegate's own behaviour |
| `--temperature > 0`, `--top-k > 0`, `--top-p < 1` | **refused by flag name** |
| `--min-p` (non-default) | **refused by name** — not a delegate field at all |
| `--seed` | **refused by name** — the delegate has no RNG of its own to seed |
| `--max-tokens` (≠ default) | **reported, not applied** — the delegate decodes until its own stop condition |

The refusal is the *same shared function* `server/xlm_backend.py` uses per request
(`_refuse_unsupported_sampling`), so `run` and `serve` cannot disagree about which *sampling*
values are refused. The `--max-tokens` note goes to **stderr** and the answer stays on stdout:
truncating the text *after* the delegate produced it would be the CLI inventing a cap the model never
saw.

The delegate library writes its own runtime banner to file descriptor **1**; `quiet_delegate_stdout`
(`python/pocketllm/xlm.py`) redirects fd 1 to stderr around the load and infer calls, so
`pocketllm run … > out.txt` yields a clean answer and the banner is preserved for debugging on fd 2.
See [`tests/native/test_delegate_stdout.py`](https://github.com/lvyufeng/PocketLLM/blob/main/tests/native/test_delegate_stdout.py).

## Serving on the board

`serve --device horizon` is the second entry point over this delegate
([#571](https://github.com/lvyufeng/PocketLLM/pull/571)/[#572](https://github.com/lvyufeng/PocketLLM/pull/572)),
and it is exercised end to end on the board — measured 2026-10-09 with the **0.6B (`w8`)** `.hbm`
(`Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm`, the same checkpoint
`tests/serving/test_xlm_serve_horizon.py` hardcodes) and a greedy tokenizer directory:

- **A single request is clean.** `POST /v1/chat/completions` returns `200`,
  `Content-Type: application/json; charset=utf-8`, a body any JSON client parses, the answer in
  `message.content` and the reasoning block split into `message.reasoning_content`. No SDK-monitor
  token appears in the body.
- **The threaded server does not leak the fd-1 banner.** 6 concurrent requests × 3 rounds, all 200,
  all valid JSON, **the same answer**, and no `[UCP]`/`[DNN]`/`BPU_MONITOR`/`mod_mgr` fragment in any
  body. The banner (`[UCP]`, `[DNN]`, `[VP]`, `[HPL]`, `BPU_MONITOR`, and the loader's `mod_mgr`/
  `XlmImpl` lines) is present on the server's **stderr** and **absent from its stdout** — routed,
  not dropped, which is the property `quiet_delegate_stdout` promises under `run` and this page now
  confirms under `serve`.
- **Requests serialize.** The six concurrent requests completed in steps of ~1.37 s (a 0.6B decode:
  1.37 / 2.74 / 4.10 / 5.47 / 6.84 / 8.21 s), i.e. one at a time behind the adapter's lock, not an
  error and not an interleave — the answer was identical across the round, so no two requests shared
  the single delegate session. (The 1.7B `.hbm` was also exercised by hand and behaves the same way,
  at a larger ~2.67 s step; the committed test uses 0.6B because a smaller `.hbm` makes the 6×3 round
  cheap enough to run in CI's time budget.)
- **Per-request sampling is refused with a clean `400`.** `temperature: 0.9`, `top_p: 0.5`,
  `top_k: 20` and `min_p: 0.1` each return `400` naming the field; `temperature: 0`, `top_p: 1`,
  `top_k: 0` are accepted. No 500, no silently-ignored field.
- **`stream: true` is well-formed and terminates.** `Content-Type: text/event-stream`, every frame a
  `data: <json>` object or the single terminal `data: [DONE]`, no banner token between two frames, and
  the socket closes (the server sends `Connection: close`) rather than hanging. The content deltas
  reassemble to exactly the non-streamed `message.content`, and the reasoning block arrives once, under
  `reasoning_content`. This is the surface most OpenAI clients use by default, and until this it had
  never been set on the board.
- **The streamed `finish_reason` is `length`, and that is the delegate's honesty, not a stuck cap.**
  The SDK's `xlm_result_t` carries text and a performance block and **no stop signal at all** — no
  end-of-text flag, no reason, and this build leaves even `prefill_token_num`/`decode_token_num` at
  zero — so the adapter has nothing to turn into `stop` and reports the value it defaults to. It is
  exactly one `finish_reason` per stream, as OpenAI requires; a client reading `length` as "ask for
  more" would ask in vain, which is why it is written down here rather than papered over with a
  `stop` the delegate never observed.

One correction this measurement forced, and it is the reason the sentence above says *sampling*: the
two entry points did **not** fully agree on the sampling fields, because `top_k: 0` — the value the
CLI's own help and the adapter's refusal both instruct a caller to use for "no limit" — was accepted
by `run` but rejected by `serve` with `400 top_k must be >= 1` (a `SamplingParams` validator, not the
delegate refusing it). The validator now accepts `0` as "no limit" and refuses only a negative `top_k`;
`tests/serving/test_xlm_serve_horizon.py::test_naming_the_defaults_is_accepted` and
`tests/serving/test_protocol_requests.py::test_top_k_zero_means_no_limit_and_is_accepted` pin it.

A second defect the streaming measurement forced, and it was a real one. The delegate's demo config
sets `enable_thinking: true`, so the `.hbm` always emits a full ` thinking… response` block — but the
adapter read every request as a *chat* model. `split_reasoning` cannot tell the two modes apart once
the `</think>` marker has arrived, so the collected answer was always right; a **stream**, which has
to classify the text before the marker lands, was not. Read as `chat`, the splitter sends the
pre-marker text as `content`, and when the marker finally arrives the block is re-sent as
`reasoning_content` — so the same sentences reached a client twice under two keys and the streamed
`content` was the raw text (` thinking` tag, reasoning block and answer together) rather than the
answer. Measured on the board: streamed content
`3c 74 68 69 6e 6b 3e 0a 4f 6b 61 …` versus the non-streamed `0a 0a 4f 4b`. The fix reads the mode
from the checkpoint's own `enable_thinking` — the same rule the adapter already applies to sampling,
because the delegate builds its template and its sampler at load and a request cannot un-think it —
and the streamed content now equals the collected one byte for byte.

Two further differences are **not** bugs, and are worth stating so the two entry points are not read
as interchangeable:

- **`serve` frames a chat prompt differently from `run`.** With no chat template available to the
  adapter, a `messages` request is rendered by the protocol layer's fallback to `"user: <text>"`,
  while `run --prompt` sends the bare `<text>`. The delegate then applies its own ChatML template to
  each, so the same user text yields a **different (both coherent) continuation**. This is the
  `render_fallback_prompt` path the adapter documents, not a serving defect — a bare-text comparison
  between `run` and `serve` is comparing two different prompts.
- **`seed` is asymmetric, and only one side changes what the path claims.** `run --seed N` is refused;
  a `serve` request with `"seed": N` is **accepted and ignored** (the same class of bug the `top_k: 0`
  fix closed, and noted here rather than fixed: it spans `native_backend` and any backend that does
  sample, so it is not this adapter's alone to correct).

### 4B and 8B over `serve`

The block above is the *surface* behavior, measured on 0.6B because a small `.hbm` keeps the
concurrency rounds cheap. The **large** models had never been through the serving path at all before
this, so they were measured on 2026-10-10, once the `balanced` pool made them loadable — the shipped
4B and 8B, each opened with `pocketllm serve --device horizon`, canonical prompt, greedy tokenizer
directory:

| Graph | `/v1/completions` | `/v1/chat/completions` | Answer |
|---|---|---|---|
| shipped 4B | **200**, 4.46 s, `application/json; charset=utf-8` | **200**, 4.94 s | `The capital of France is **Paris**.` |
| shipped 8B | **200**, 5.14 s | **200**, 5.16 s | `The capital of France is **Paris**.` |

**Both endpoints work on both graphs**, and the answers are coherent, not garbage. On
`/v1/chat/completions` the reasoning block is split into `message.reasoning_content` as on 0.6B (809
characters for the 4B, 602 for the 8B), and `message.content` is the answer alone. No body carries an
SDK-monitor token — `[UCP]`/`[DNN]`/`BPU_MONITOR`/`mod_mgr` and the rest stayed on the server's fd 2 —
the `usage` object is present-and-zero exactly as on the small models, and `finish_reason` is `length`
for the same reason recorded above (the delegate reports no stop signal). Wall time is the BPU decode:
4.5 s at the 4B's 42.6 t/s and 5.1 s at the 8B's 29.6 t/s, which is the *same* generation the ladder
measures — nothing about the HTTP shell changes what the graph does.

### Our compiled 0.6B over `serve`

Our own graphs had only ever been run directly, never through the serving path, and "the compiled
graphs are faster" is only useful if they deploy the way the vendor's do. So our `cache_1024`
`Qwen3-0.6B` was served on 2026-10-10 exactly as the shipped graphs are — `pocketllm serve --device
horizon --model /tmp/ours/ours_06b_cache1024_w8.json`, the same two environment variables, a config
naming our `.hbm` and the greedy tokenizer directory:

| Endpoint | Status | Body |
|---|---|---|
| `GET /v1/models` | **200** | `{"object":"list","data":[{"id":"ours_06b_cache1024_w8.json",…}]}` |
| `POST /v1/completions` | **200** | `"text":"\n\nThe capital of France is **Paris**."` |
| `POST /v1/chat/completions` | **200** | `content` = the answer; `reasoning_content` = the thinking block |
| `POST /v1/chat/completions` (`temperature: 0.9`) | **400** | refused by field name, same `unsupported_feature` message |

**It starts and answers through the same path as the vendor graphs, with no config field, no missing
tensor and no env the shipped path sets that this one does not** — the integration proof the compiled
graphs needed. Both endpoints return `200` with coherent text (`The capital of France is **Paris**.`);
the `finish_reason` is `length`, the `usage` object is present-and-zero, and the reasoning block lands
in `message.reasoning_content` with `message.content` the answer alone — every one of the #608
behaviours, unchanged. No body carries an SDK-monitor token: the loader banner is on the server's stderr
(the same `[UCP]`/`mod_mgr`/`BPU_MONITOR` chatter the shipped 0.6B produces), not in a reply. The one
thing worth noting is that it is **fast**: the whole request is 0.58 s on `/v1/completions` (0.99 s on
`/v1/chat/completions`, whose answer carries the longer reasoning block), against the shipped 4B's 4.46
and 8B's 5.14 — the 121.6 t/s decode of
[the compiled-0.6B section](#our-compiled-06b-vs-the-shipped-06b) showing through the HTTP shell, which
changes nothing about what the graph does.

### The 8B sits comfortably in `balanced` — the numbers

The question the mode switch leaves open is how much room a large model actually has, because it
decides whether `balanced` covers the road ahead or only today's models. Measured while an 8B was held
open by `serve`, from the kernel's own `ion` accounting
(`/sys/kernel/debug/ion/heaps/all_heap_info`), which is readable on this board as root:

| | pool total | used with **4B** held | used with **8B** held | idle |
|---|---|---|---|---|
| `carveout` (the `.hbm`'s pool) | **10,737,418,240 B (10.00 GiB)** | 6,686,113,792 B (**62.3%**) | **9,417,129,984 B (87.7%)** | 0 |
| `ion_uncache` (per-core scratch) | 2,147,483,648 B | 316,735,488 B (14.7%) | 316,735,488 B (14.7%) | 8.4% |
| `MemAvailable` | — | 44.87 GiB | 44.88 GiB | 45.10 GiB |

**An 8B leaves 1,320,288,256 B (1.23 GiB) of the carve-out free — 12.3% — and 44.88 GiB of the
board's 46.48 GiB `MemTotal` still available.** Both graphs release the pool completely on close
(`carveout` returns to 0), so this is a load-time peak and not a leak. The verdict is that 8B is
**near the top of what `balanced` holds, not against the wall**: it fits with room to spare, but the
room is about a fifth of the pool, not multiples of the graph.

**The honest "largest model that fits" number** is then a budget, not a guess. The carve-out's
10.00 GiB is split between the `.hbm` itself and a fixed per-load overhead — the HBRT workspace plus
per-core scratch that land in the same pool. Measured: 5,704,908,800 B (`.hbm`) + **3,712,221,184 B
(3.46 GiB overhead)** = 9,417,129,984 B at 8B, and 3,328,311,296 B + **3,357,802,496 B (3.13 GiB)**
= 6,686,113,792 B at 4B. So the **overhead is ~3.1–3.5 GiB and grows a little with the model**
(+0.33 GiB from 4B to 8B), which leaves roughly **6.5 GiB for a `.hbm` file** once it is accounted
for. Since a w4 graph's file runs a bit above its weight size, that is about an **8B–9B class model**
— an 8B fits with ~1.2 GiB to spare, and there is no headroom for a 14B (which at w4 would be roughly
8 GiB of weights alone). **Anything past that is `bpu_first` — 18.00 GiB**
([the applied value](#a-second-switch-bpu_first-applied-2026-10-13); this page used to print 17.93),
**not `balanced`** — which is the one number that answers the "what else can this board run" follow-up,
and it is why the switch was worth making for 8B specifically rather than as a general 14B door.

**The board has since moved there (2026-10-13) for the 8B `w8` build**, and the budget above shows
why: at **9.75 GiB** held at load it would leave `balanced` 97.5% full — **0.25 GiB free against the
1.23 GiB (12.3%) this section calls comfortable** — so the raised pool is where it runs, at 54.2% of
18.00 GiB ([its ladder row](#measured-ladder)). Its `cache_1024` shape also carries a much smaller
pool overhead than the `cache_4096` budget above: **9,078,659,624 B of file + 1,394,124,248 B
(1.30 GiB) = the 10,472,783,872 B measured**, against the ~3.5 GiB overhead the shipped
`cache_4096` 8B pays.

### The host shell's cost is below the noise floor

The serving shell is a `ThreadingHTTPServer` that serializes one request at a time behind the
adapter's lock. That lock is a *throughput* limit — it stops two requests overlapping — but the
question worth answering is whether the shell also adds *per-request* cost on top of the delegate.
Measured 2026-10-09, shipped 1.7B graph (the ladder's `.hbm`), canonical prompt, greedy tokenizer
directory, the same request driven two ways:

| path | median wall time |
|---|---|
| `XlmEngine.infer` — the raw delegate, what the ladder and `run` use | 2684 ms |
| `POST /v1/completions` through `pocketllm serve` — HTTP in, response bytes out | 2677 ms |

Two runs of that comparison (5 and 15 iterations each) put the difference at **+2.5 ms** and
**−7.1 ms** — opposite signs, both well under the ±10 ms run-to-run spread of a 2.68 s request. So
the shell's per-request cost (HTTP parse, request build, the lock acquire, response serialization,
the socket) is **not resolvable above the noise: under ~0.4% of the request, and negligible against
the delegate's ~14.3 ms/token decode.** There is no serving-perf lever in this shell — the BPU is the
whole cost, which is the shape you would expect from a shell that moves a few hundred bytes of text
and holds a lock while the delegate runs the graph.

Both paths did the same generation work (both ran ~2.68 s for the same 188-token answer); they differ
only in how the answer is presented. `XlmEngine.infer` returns the delegate's raw text with its
` thinking… response` block included, while the HTTP path splits that block out —
`/v1/chat/completions` returns the answer in `message.content` and the reasoning in
`message.reasoning_content`, and `/v1/completions` returns the answer alone. That is the same
`split_reasoning` presentation the [streaming](#serving-on-the-board) path uses, not a second
generation.

## Recovering from the delegate's hang: `--generation-timeout`

`XlmEngine.infer` is **one blocking `ctypes` call** — `xlm_infer` returns nothing until the
whole answer is decoded — and the delegate offers no timeout and no way to stop it once it
has started (`supports_cancellation` is false for exactly that reason). A process that takes
the lost-wake-up hang described above is therefore in a call that never returns, and **no
timeout on the call site can reach it**: the call site is inside the call. The only place a
watchdog can live is *outside the process*.

Three shapes were considered, and two of them do not work:

- **A watchdog thread in the same process cannot interrupt a blocked `ctypes` call.** Python
  delivers an exception into a thread only at a bytecode boundary, and a thread parked in a
  C call never reaches one; `xlm_infer` reads no cancellation flag either. The thread detects
  the hang and recovers nothing — the process is still stuck.
- **A thread running `infer` with a join-timeout recovers the caller but leaks the
  accelerator.** The host can abandon the join and carry on, but the abandoned thread is
  still inside `xlm_infer`, holding the delegate's one session and its BPU context. The
  session cannot be `xlm_destroy`-ed while a call is in flight, so that BPU context is held
  for the life of the process. That is not a mitigated failure; it is a leaked BPU.
- **A subprocess per generation is the only shape that can be forcibly torn down**, and the
  forced teardown is the whole recovery: a hung child holds the BPU only through its live
  context, and `SIGKILL` takes that with it. Measured: after the gate below killed a hung 8B
  run, the **next** run on the same `bpu_core` list produced a full answer. That is the fact
  the design rests on — killing the process *is* the recovery, not a metaphor for one.

`--generation-timeout SECONDS` selects it, and it is **off by default**, so an ordinary `run`
is byte-for-byte what it was. With it set, `run` execs `pocketllm run` in a child with the
same flags (the child is the one that opens the delegate) and watches that child's stdout: as
long as decoded pieces keep arriving the child is making progress; when the stream goes
**silent for the window** the child's **process group** is killed with `SIGKILL` — the group,
because the hung decode worker is a thread of the child and a bare kill of the top pid would
leave it holding the BPU. Detection is the same 25 s no-token stall #650 used; the flag's
value is that window. To make the stream watchable the inner process writes each piece to a
saved duplicate of the real stdout as the delegate produces it, because `infer` runs the call
inside `quiet_delegate_stdout`, which points fd 1 at stderr for the duration.

The output contract is preserved exactly. The inner process echoes the prompt *before* the
decode and writes the pieces as they come, then only a terminating newline — so the child's
stdout is `prompt + pieces + "\n"`, character-for-character the in-process `prompt + text +
"\n"`. Verified on the 0.6B: `pocketllm run … --generation-timeout 25` and the plain run
produce **byte-identical stdout** (`cmp` clean), same exit code, and the delegate's own banner
still lands on stderr. The stderr of the child goes to a file, never a pipe drained at the
end — the delegate's fd-2 banner can exceed a pipe buffer, and an unread pipe would block the
child on a stderr write and be killed as a stall it did not cause.

**Gate (board, 8B `cache_1024`, the `1..200` enumeration, 40 runs under
`--generation-timeout 25`):** 38 runs clean, **2 hangs detected and killed** — run 32 after
40 s (276 bytes of partial output) and run 37 after 55 s (2219 bytes) — both exiting **124**
with the vendor-cause message on stderr and the partial text on stdout, and **0 leftover
processes** across every run. The happy path is unchanged: 38 runs produced the full answer in
~41 s, byte-identical to the unwatched command, and no clean run was falsely killed. The
mechanism itself is pinned hardware-free in `tests/native/test_watchdog.py` (a stand-in child
that stalls is detected and killed promptly; a healthy child is untouched; the parent is
reusable after a kill), because the real hang is a ~1-in-10 race CI has no board to reproduce.

**It is a mitigation, not a repair.** The deadlock is inside the vendor `libxlm.so` and is
still there; a hung generation is *lost* — its partial text is all the caller gets and the
request must be re-run — and the vendor fix is the only thing that removes it. What the
watchdog buys is that the loss is one request instead of the whole process. It is a `run`
flag, not a `serve` flag: the serving adapter serializes behind a lock it must hold for the
request's life, so a deadlock there wedges the whole server, and recovering *that* needs a
process boundary around the session (and a way for HTTP clients to learn the request died)
that this change deliberately does not build. On `serve` today the only recovery from a hang
is restarting the server, and that is stated rather than papered over.

## What is not on this path

- **No Python backend implements the delegate.** `--device horizon` selects the `xlm` serving adapter,
  not a [`pocketllm.backends`](../architecture/backend_model.md) implementation; the underlying
  `.hbm` is a black box that takes text and returns text.
- **No token ids, no logits.** A caller that needs per-token probabilities, a custom sampler, or the
  raw token stream cannot get them from this delegate. That needs the C engine's Qwen3 path on a host
  where it runs, or the vendor compile chain.
- **No user-defined context length.** Context is `cache_4096`, baked into the `.hbm` at compile time;
  it is not a runtime flag on this path.