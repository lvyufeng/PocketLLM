# Qwen3 on the RDK S600 (Horizon Nash BPU)

This page is the durable record of what this tree can actually run on the **RDK S600** board and how
to run it: the command, the two environment variables it cannot start without, the measured
performance ladder, the memory mode the board has to boot in to reach its large models, and the
determinism contract. Every
number and every failure below was measured on the board — first on 2026-10-09, then again on
2026-10-10 **after the `balanced` memory switch and its reboot**, when all four Qwen3 sizes became
runnable; whether either transfers to another S600 is a claim to re-check in place, and the SDK
version is part of the result.

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

**All four shipped Qwen3 sizes load and generate in `balanced` mode** — the board boots `cpu_first`,
where the two large ones are refused; see [the memory mode](#the-4b-8b-ceiling-applied):

| Size | `.hbm` | Precision | File size | Loads? |
|---|---|---|---|---|
| Qwen3-0.6B | `Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm` | w8 | 1.02 GiB | **yes** |
| Qwen3-1.7B | `Qwen3-1.7B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 1.70 GiB | **yes** |
| Qwen3-4B | `Qwen3-4B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 3.10 GiB | **yes, in `balanced`** |
| Qwen3-8B | `Qwen3-8B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 5.31 GiB | **yes, in `balanced`** |

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

**This is the canonical ladder: all four shipped Qwen3 sizes plus our own compiled 0.6B and 4B, on the
`balanced` pool, measured in one session on 2026-10-10 (the compiled 0.6B was added the same day, once
its build landed).** Fixed prompt **"The capital of France is"**,
greedy `generation_config.json`, canonical-prompt three decode runs per graph, one process per graph.
Throughput and load time from the delegate's `last_performance()` (`xlm_model_performance_t`); memory
held from the kernel's own ION accounting (`/sys/kernel/debug/ion/heaps/all_heap_info`), read **after
the load completes and before the first decode** — the load-time peak, the same instant
[the margin section](#the-8b-sits-comfortably-in-balanced-the-numbers) measures, so the two tables
agree. (A decode adds ~152 MiB of KV/scratch on top; that is per-request work-in-flight, not what the
model costs in the pool, and it is excluded here deliberately.) The memory column is what each model
costs in the 10.00 GiB `ion_carveout`.

| Size | `.hbm` | Load | Prefill | Decode | Memory held at load (carveout) | of pool |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (w8) | 1.02 GiB | 5.26 s | 6169–7111 t/s | **88.3 t/s** (88.09 / 88.73 / 88.28) | 2,995,585,024 B (2.79 GiB) | 27.9% |
| Qwen3-1.7B (w4) | 1.70 GiB | 5.70 s | 5172–5818 t/s | **70.2 t/s** (70.18 / 70.13 / 70.16) | 3,734,503,424 B (3.48 GiB) | 34.8% |
| Qwen3-4B (w4) | 3.10 GiB | 6.44 s | 2415–2573 t/s | **42.8 t/s** (42.77 / 42.72 / 42.78) | 6,686,113,792 B (6.23 GiB) | 62.3% |
| Qwen3-8B (w4) | 5.31 GiB | 7.66 s | 1939–2040 t/s | **29.7 t/s** (29.65 / 29.66 / 29.70) | 9,417,129,984 B (8.77 GiB) | **87.7%** |
| Qwen3-0.6B (w8), ours `cache_1024` | 0.98 GiB | 5.31 s | 11636–13128 t/s | **121.6 t/s** (122.18 / 121.72 / 120.82) | 1,702,756,352 B (1.59 GiB) | 15.9% |
| Qwen3-4B (w4), ours `cache_1024` | 3.00 GiB | 6.35 s | 4339–4571 t/s | **49.9 t/s** (49.99 / 49.96 / 49.86) | 4,254,269,440 B (3.96 GiB) | 39.6% |

Decode falls with size — 88.3 → 70.2 → 42.8 → 29.7 t/s on the shipped graphs — and the memory column
rises with it, which is the whole of the ladder's shape: the BPU's per-token cost and the pool's
footprint both track the parameter count. Our own `cache_1024` builds sit *above* their shipped twins
on both axes (faster, and smaller in the pool), which is the subject of the last note below. Every
graph's three runs produced **byte-identical text**, so each number is a rate and not a draw.

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

*A graph we compile ourselves is faster.* Our own `Qwen3-1.7B` build at `cache_1024` (1.66 GiB, md5
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
all ten faithfulness prompts identically. So the trigger is none of position, cache, template or token
mapping; it is the graph's own generation on the 1.7B `w4` weights, and localizing it to a layer or a pass
needs a rebuild this board cannot run.

#### A request whose prompt does not fit the cache aborts the process

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
up to where it stops, and the KV is not left dirty — a canonical `The capital of France is` issued *after*
an over-cap one in the same process still returns `Paris`. Two smaller quirks sit beside it and are not
corruption: `max_tokens` is not applied on this path (a request for `max_tokens: 3` returns the full
answer, because the delegate owns its own stop), and a generation consumed entirely by an unterminated
reasoning block surfaces as **empty content at 200** once the chat splitter strips it. Neither is the
prompt-does-not-fit case, which is a crash.

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

**The board runs `balanced`, and 4B and 8B load.** The refusal below is what the board did in the
`cpu_first` mode it booted in, where `ion_carveout` — the pool the whole `.hbm` is loaded into — is
**2.00 GiB**. The SDK ships a supported tool, `hb_switch_ion.sh`, that moves the board to `balanced`,
where that same pool is **10.00 GiB**; the switch was applied and the board rebooted **2026-10-10**, and
both large graphs now initialize and generate (see [the ladder](#measured-ladder) and
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
| `balanced` (**applied on this board 2026-10-10**) | **10.00 GiB** | 4B (3.00 GiB) **and** 8B (5.31 GiB) |
| `bpu_first` | 17.93 GiB | 4B/8B, at the cost of general RAM |

So the S600's 4B/8B support was **one supported command plus a reboot** away. The board boots
`cpu_first`, whose 2.00 GiB pool is what refused the graphs — nothing about the model size, the SDK
build, or our code. This is *not* the manual DTB edit this page used to describe; `hb_switch_ion.sh` is
the vendor's own tool, it auto-detects the board, backs the DTB up, and reverts on any `fdtput`
failure.

**The switch is APPLIED, and the reboot succeeded.** `hb_switch_ion.sh balanced` was run and the board
rebooted 2026-10-10; it came up healthy with `ion_carveout` at 10.00 GiB and `MemTotal` at 46.48 GiB
(the before/after table is [above](#the-4b-8b-ceiling-applied)), and 4B, 8B and our compiled 4B all
load and generate. What follows is the offline well-formedness proof that made the reboot a one-shot,
kept because the mode is a boot-time setting a future board will have to move again.

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
"10480MiB" — a transposed-digit typo; the cells are exact.) Only **three** of the four nodes actually
change: `ion_reserved` is already at its `balanced` value in `cpu_first`.

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

## Accuracy against the 2080ti oracle

The project's accuracy target is agreement with the 2080ti C-engine oracle. This path is **text-in /
text-out and takes no token-id input**, so the comparison is made by feeding the oracle the exact ids
the delegate's chat template produced and diffing the greedy continuations.

**A single prompt flattered this path; a batch does not.** The first measurement used one prompt
("The capital of France is") and read as "identical for 14 tokens, then a near-tie". Widening it to
ten diverse prompts × both sizes tells a more honest story — **0/10 fully identical at either size**,
and the agreement is a **prefix**, not the whole answer:

| Size | Fully identical | Shared exact prefix (tokens) | Median |
|---|---|---|---|
| 0.6B (w8 vs oracle `q4_k_m`) | **0 / 10** | 8 … 59 | ~26 |
| 1.7B (w4 vs oracle `Qwen3-1.7B-Q4_K_M`) | **0 / 10** | 4 … 52 | ~12 |

The batch (10 prompts, their prompt ids, greedy continuation ids and text, both sizes, plus the
tokenizer and the greedy `generation_config.json`) is `~/scratch/ref/s600_parity_batch.tgz` on the
board; the single-prompt artifacts are `~/scratch/ref/parity/s600_parity.json`.

### The width confound, and why only one pair can remove it

The two rows above do **not** confound width equally, and the weaker row is the 0.6B one:

| Pair | Board `.hbm` | Oracle GGUF | Width | Scheme |
|---|---|---|---|---|
| 0.6B | `w8` (int8) | `q4_k_m` | **mismatched** (8 vs 4) | mismatched |
| 1.7B | `w4` (int4) | `Qwen3-1.7B-Q4_K_M` | matched (4 vs 4) | mismatched |

So the 0.6B divergence could be **width or scheme**, while the 1.7B row — where the 14.06-logit
Row 05 disagreement lives — isolates **scheme**, because both sides are 4-bit. A same-width 0.6B
pair would settle it, but **the vendor does not publish one**, and neither does it publish the
mirror-image lever (a `w8` 1.7B):

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
1.7B row isolates scheme; and the one large-margin divergence we have measured (Row 05) is on the
row that isolates it.**

**Most divergences are near-ties — but not all.** Row 01 (1.7B) splits on a **1.33**-logit gap, the
"two coherent phrasings of the same fact" case the single prompt showed. **Row 05 (1.7B) does not**:
the two engines diverge at the same 12-token context with the oracle scoring its own choice
**57.33** against the S600's choice **43.26** — a **14.06** gap. A gap that large is not 4-bit
rounding noise against the *same* weights; the S600's `.hbm` is therefore **not the oracle's
`q4_k_m` numerically**, whatever the bucket's `w4` label says.

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

**The honest statement is therefore:** the output is **coherent and on-topic** (all ten prompts
produced correct, well-formed answers — see the table below), but it is **token-identical to the
oracle only for a prefix** (median ~12 tokens at 1.7B, ~26 at 0.6B), and at least one divergence
(Row 05) is a large-margin disagreement that reflects the two sides being **different
quantizations**, not a near-tie. This path cannot be compared to 1e-3 at the logit level — the
delegate exposes no logits — so the token agreement above is the whole of the evidence, and the page
says so rather than implying more.

### The per-prompt batch (S600 side)

| Prompt | 0.6B ids | 1.7B ids | Note |
|---|---|---|---|
| factual short-answer (largest planet) | 126 | 125 | both correct |
| multi-step arithmetic (train distance) | 919 | 651 | both correct |
| code snippet (iterative factorial) | 782 | 1330 | both correct |
| list three primes > 20 | 429 | 509 | both correct |
| two-part (boiling + freezing point) | 325 | 229 | **1.7B = Row 05, the 14.06 divergence** |
| translate EN→FR | 225 | 271 | both correct |
| summarize a paragraph | 232 | 442 | both correct |
| yes/no with reasoning (is 0 even) | 251 | 401 | both correct |
| long division (100th digit of 1/7) | 789 | **4055** | 1.7B loops in ` thinking`, never closes it |
| long prefill (~225-token passage) | 120 | 122 | both correct |

Every answer is correct to a human read, including the intended-to-be-hard 1/7 prompt (the 0.6B
computes `100 mod 6 = 4` → the 4th digit of `142857` = **8**, right). The one quality defect is
1.7B's repetition loop on that prompt — a model artifact, kept in the record rather than dropped,
and a strong oracle test precisely because a degenerate loop is hard to reproduce by accident.

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
8 GiB of weights alone). **Anything past that is `bpu_first` (17.93 GiB), not `balanced`** — which is
the one number that answers the "what else can this board run" follow-up, and it is why the switch was
worth making for 8B specifically rather than as a general 14B door.

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

## What is not on this path

- **No Python backend implements the delegate.** `--device horizon` selects the `xlm` serving adapter,
  not a [`pocketllm.backends`](../architecture/backend_model.md) implementation; the underlying
  `.hbm` is a black box that takes text and returns text.
- **No token ids, no logits.** A caller that needs per-token probabilities, a custom sampler, or the
  raw token stream cannot get them from this delegate. That needs the C engine's Qwen3 path on a host
  where it runs, or the vendor compile chain.
- **No user-defined context length.** Context is `cache_4096`, baked into the `.hbm` at compile time;
  it is not a runtime flag on this path.