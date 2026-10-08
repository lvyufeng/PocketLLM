# Qwen3 on the RDK S600 (Horizon Nash BPU)

This page is the durable record of what this tree can actually run on the **RDK S600** board and how
to run it: the command, the two environment variables it cannot start without, the measured
performance ladder, the memory ceiling that stops it at 1.7B, and the determinism contract. Every
number and every failure below was measured on the board on 2026-10-09 — whether it transfers to
another S600 is a claim to re-check in place, and the SDK version is part of the result.

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

Two of the four shipped Qwen3 sizes load and generate:

| Size | `.hbm` | Precision | File size | Loads? |
|---|---|---|---|---|
| Qwen3-0.6B | `Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm` | w8 | 1.02 GiB | **yes** |
| Qwen3-1.7B | `Qwen3-1.7B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 1.70 GiB | **yes** |
| Qwen3-4B | `Qwen3-4B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 3.10 GiB | **no** — see [the ceiling](#the-4b-8b-ceiling) |
| Qwen3-8B | `Qwen3-8B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm` | w4 | 5.31 GiB | **no** |

The naming is the compiler's: `chunk_512` is the prefill chunk, `cache_4096` the max context,
`corenum_4_4` the BPU-core assignment. Every large model in the SDK ships as `corenum_4_4`; **there
is no smaller-corenum 4B/8B variant to fall back to.**

## Running it: the exact contract

Two environment variables must be set **before `pocketllm run` starts**, and the CLI requires them
rather than setting them:

```bash
SDK=~/llm_sdk/D-Robotics_LLM_S600_1.0.2_SDK

export LD_LIBRARY_PATH=$SDK/lib:$LD_LIBRARY_PATH      # read by dlopen at process start
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
`LD_LIBRARY_PATH` is read by `dlopen` **before the process starts** — exporting it from inside Python
is a no-op, so "set it for the user" moves the error to a confusing missing-`libopencv_world.so.409`
later. `HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6` is the L2m split the graph was compiled with, and the
SDK's own `run_llm.sh` sets exactly `6:6:6:6` for every model. `_require_delegate_env()` fails with a
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

Fixed prompt **"The capital of France is"**, greedy `generation_config.json`, through
`pocketllm run --device horizon`. Throughput and load time from the delegate's `last_performance()`
(`xlm_model_performance_t`); three runs per size.

| Size | Loads? | Load | Prefill | Decode | TTFT |
|---|---|---|---|---|---|
| Qwen3-0.6B (w8) | yes | 5.25–5.30 s | 6169 t/s | **87.0 t/s** (86.27 / 87.17 / 87.39) | not available |
| Qwen3-1.7B (w4) | yes | 5.62–5.69 s | 5172 t/s | **69.4 t/s** (69.35 / 69.55 / 69.31) | not available |
| Qwen3-4B (w4) | no | — | — | — | — |
| Qwen3-8B (w4) | no | — | — | — | — |

**TTFT is not available on this SDK build.** The `ttft`, `tpot` and `end_to_end_cost` fields of
`xlm_model_performance_t` come back `0.0`, so a time-to-first-token is not something this page can
quote from the runtime; wall-clock load time above is quoted instead. Decode at 87 t/s (0.6B) and
69 t/s (1.7B) is the load-bearing number: it is what a caller feels, and the spread across three runs
is under 1.4% at 0.6B and under 0.4% at 1.7B, so it is stable rather than a lucky run.

## The 4B / 8B ceiling

4B and 8B **refuse to load**, and the refusal is the board's, not this tree's:

```text
Cannot malloc bpu memory with length 3326941192 bytes   # 4B
Cannot malloc bpu memory with length 5703561320 bytes   # 8B
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED
  -> ion_alloc ret=-12 (ENOMEM)
  -> hbDNNInitializeFromFiles error code -400001
```

`hbDNNInitializeFromFiles` mallocs the whole graph into one contiguous "unified BPU memory" pool
through ION. The board reserves `bpu_region@4300000000` = **384 MiB** for that pool; the 4B graph
needs 3.33 GB and the 8B graph 5.70 GB, and both exceed the budget the SDK can allocate. 1.7B
(1.83 GB) fits, so **the ceiling on this board as configured is 1.7B** — measured to lie between
1.83 GB and 3.33 GB.

**This is not our Python.** Three independent checks:

- The SDK's **own** demo binary (`oellm_runtime/examples/llm_demo/llm`) on the same 4B/8B configs
  fails identically, while it initializes 0.6B/1.7B.
- `hrt_model_exec model_info` — the SDK's own tool — reports the same `HBRT4_STATUS_RESOURCE_EXHAUSTED`.
- The 4B/8B `.hbm` md5s match the SDK's published `md5sum.txt`, so the artifacts are not corrupt.

`cli.py` / `xlm.py` surface the delegate's refusal as a clean message and `rc 1`, with no traceback.

**No board or SDK knob moves it.** Measured, all through the SDK's own binary so none of our code is
in the path:

| Knob tried | Values | Effect on 4B |
|---|---|---|
| `HB_DNN_USER_DEFINED_L2M_SIZES` | `6:6:6:6`, `0:0:0:0`, `2:2:2:2`, `1:1:1:1`, `12:12:12:12` | none — identical `ion_alloc -12` |
| `bpu_core` (corenum) | `[0]`, `[0,1]`, `[0,1,2]`, `[0,1,2,3]` | none — identical |
| HBRT memory-mode env var | — | no such variable exists (`strings` over `libxlm.so` / `libhbrt4` / `libhbipm` show only `HBTL_*` diagnostics) |

Corenum cannot help because the refusal happens at `hbDNNInitializeFromFiles`, **before** any core is
assigned. The L2m split cannot help because the failure is the graph's total footprint, not its L2m
partition — and the SDK's docs set `6:6:6:6` for every model regardless of size, so it is not a
size-dependent lever. `bpu_region` / ION heap sizes are board memory settings, not model knobs, and
are out of scope to change.

**The only path to 4B/8B is a `.hbm` recompiled with a smaller footprint** — a shorter context
(`cache_1024` instead of `4096`, as the SDK's VLM 7B graph uses) or a smaller prefill chunk. The
compiler chain that produces one (`GGUF → HF safetensors → leap_llm`/`oellm_build` → `hbdk4` → `.hbm`)
runs on **x86-64 / cp310 only** and is gated on vendor-supplied components, so it is not a board-side
knob. Until such a graph exists, 1.7B is the ceiling.

## Determinism

Reproducibility on this path comes entirely from the **tokenizer directory's `generation_config.json`**,
not from a flag: the delegate builds its sampler from that file when the model is loaded.

- **Greedy** (`temperature 0`, `do_sample false`) is byte-identical across runs. Measured over three
  CLI runs each: 0.6B sha256 `a154da66b9307a50`, 1.7B sha256 `bc75540e622e9551`.
- **The SDK's default is not deterministic.** The shipped `generation_config.json` uses
  `temperature 0.6`, `top_k 20`; same-prompt runs differ in the reasoning block and in surface form.
  A page that claimed "runs are reproducible" without naming the file would be wrong.

So a deterministic deployment ships a greedy `generation_config.json` beside the tokenizer. That is a
file the caller edits, which is why `--seed` is refused rather than honoured — see below.

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
(`_refuse_unsupported_sampling`), so `run` and `serve` cannot disagree about which values are refused.
The `--max-tokens` note goes to **stderr** and the answer stays on stdout: truncating the text *after*
the delegate produced it would be the CLI inventing a cap the model never saw.

The delegate library writes its own runtime banner to file descriptor **1**; `quiet_delegate_stdout`
(`python/pocketllm/xlm.py`) redirects fd 1 to stderr around the load and infer calls, so
`pocketllm run … > out.txt` yields a clean answer and the banner is preserved for debugging on fd 2.
See [`tests/native/test_delegate_stdout.py`](https://github.com/lvyufeng/PocketLLM/blob/main/tests/native/test_delegate_stdout.py).

## What is not on this path

- **No Python backend implements the delegate.** `--device horizon` selects the `xlm` serving adapter,
  not a [`pocketllm.backends`](../architecture/backend_model.md) implementation; the underlying
  `.hbm` is a black box that takes text and returns text.
- **No token ids, no logits.** A caller that needs per-token probabilities, a custom sampler, or the
  raw token stream cannot get them from this delegate. That needs the C engine's Qwen3 path on a host
  where it runs, or the vendor compile chain.
- **No user-defined context length.** Context is `cache_4096`, baked into the `.hbm` at compile time;
  it is not a runtime flag on this path.