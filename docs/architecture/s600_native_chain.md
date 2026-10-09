# The S600 native compile chain

[Qwen3 on the RDK S600](../models/s600_qwen3.md) once stopped at 1.7B because the board's BPU memory
pool refused the shipped 4B and 8B graphs. **This page is the answer to that refusal**: what the
D-Robotics SDK actually ships for building a `.hbm`, where the build runs, what it takes as input, and —
plainly — whether the chain is in our hands or the vendor's.

**The refusal had two independent answers, and this page is only one of them — and it is the one that
turned out not to be needed.** The first is the vendor's own, and it is one command:
`hb_switch_ion.sh balanced` moves the board's model pool (`ion_carveout`) from the 2.00 GiB it boots
with to **10.00 GiB**, which fits both 4B (3.00 GiB) and 8B (5.31 GiB) — the SDK documents `balanced`
as *the* setting for large models, and lists all four Qwen3 sizes as supported. That switch was applied
and the board rebooted **2026-10-10**; 4B and 8B load and generate now. The second is the compile chain
this page scopes: shrink the graph instead of growing the pool. The recompile *does* shrink a 4B
(3.3269 GB shipped → 3.2216 GB ours, measured), but it is **not enough on its own** — 3.00 GiB of
weights still exceed 2.00 GiB — so the two answers are not equivalent: the mode switch is the one that
reached 4B/8B, and the recompile is a lever that turned out to be too small. Both are recorded here;
the mode switch and its measured result are on
[the model page](../models/s600_qwen3.md#the-4b-8b-ceiling-applied).

It is a **scoping document, not a build record** in its body. The scoping sections below were written
before anything was compiled, installed, or run, so every claim there is either a **measured** fact
from this board (labelled) or something **read off an artifact** — a wheel, the vendor doc set, or the
vendor's own manifest in the SDK tree — and each is marked as such. The **results** are appended at the
end and are labelled measured: the chain's own 1.7B and 4B builds, and the pool switch's outcome.
Whether a cache-1024 4B graph lands under the **2.00 GiB `cpu_first` ceiling** is closed: it does
**not** (3.2216 GB measured, 3.00 GiB of weights). Under the **10.00 GiB `balanced`** carve-out the
vendor's switch installs — now the mode this board runs — it does, and so do the shipped 4B and 8B.

**The scoping has since been acted on.** The chain has produced its artifacts — our own `Qwen3-1.7B`
and `Qwen3-4B` graphs at `cache_1024` — and both **load and run on the board**; see
[the first build result](#the-first-build-result). The page keeps its scoping framing below
because that is the state the rest of it was written in and still describes; the build results
are appended, not woven in.

## The finding that reorders the question

A 4B graph exists, is public, and is already on this board. The vendor publishes `.hbm` files from an
OSS bucket with no login:

- manifest: `oellm_runtime/model/resolve_model_nash-p.md` (in the SDK)
- integrity file: `https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600/1.0.0/models/md5sum.txt`
- URL shape: `.../llm_s600/1.0.0/models/Qwen3-4B/w4/Qwen3-4B_language_chunk_512_cache_4096_w4_nash-p_corenum_4_4.hbm`

and they are on disk now, md5-verified against that file:

| Artifact on this board | Size | md5 vs published |
|---|---|---|
| `oellm_runtime/model/Qwen3_4B/…_w4_…_corenum_4_4.hbm` | 3,326,941,192 B (3.10 GiB) | matches |
| `oellm_runtime/model/Qwen3_8B/…_w4_…_corenum_4_4.hbm` | 5,703,561,320 B (5.31 GiB) | matches |

So "is a 4B `.hbm` obtainable?" is already answered: yes. The band also ships an on-device config for
each (`examples/llm_demo/qwen3_4b_config.json`, `qwen3_8b_config.json`, both `bpu_core [0,1,2,3]`,
`model_type 9`).

The binding constraint is therefore **not** "no 4B graph exists". It is the load ceiling
[the model page](../models/s600_qwen3.md#the-4b-8b-ceiling-applied) already measures: `hrt_model_exec
model_info` gives, verbatim,

```text
Cannot malloc bpu memory with length 3326941192 bytes   # 4B -> HBRT4_STATUS_RESOURCE_EXHAUSTED
Cannot malloc bpu memory with length 5703561320 bytes   # 8B -> same
```

while 1.7B (1,827,743,336 B) loads. So the ceiling is between **1.83 GB and 3.33 GB**. The chain
matters only for a 4B graph *smaller than the shipped one* — the shipped 4B is 3.10 GiB, above the
ceiling.

## What the SDK ships for building a `.hbm`

The SDK splits into `oellm_build/` (the vendor's own words: "quantization tools for the x86
development machine") and `oellm_runtime/` (the board side). `oellm_build/` contains:

| Artifact | Version | Arch / Python |
|---|---|---|
| `hbdk4_compiler-…-cp310-cp310-manylinux_2_17_x86_64.whl` | `4.10.1a2.dev202601220400+388766e.develop` | **x86_64, cp310 only** (607 MB) |
| `hbdk4_runtime_aarch64_unknown_linux_gnu_nash-…-py3-none-any.whl` | same | aarch64 **runtime** only (21 MB) |
| `leap_llm-1.0.0-py310-none-any.whl` | `1.0.0` | pure Python, cp310 (15 MB) |
| `arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz` | 13.2.rel1 | cross-toolchain x86_64→aarch64 (140 MB) |

**The compiler is x86_64-only.** `hbdk4_runtime_…_aarch64` is a *runtime* wheel — it lets the board
load a compiled graph, not build one. There is no aarch64 compiler and no `hb_compile`/`hbdk4` binary
anywhere on the board. **The build runs on the x86_64 GPU host.**

The wheels register two entry points:

```text
oellm_build     = leap_llm.apis.oellm_build:main
oellm_verifier  = leap_llm.apis.verifier_cli:main
```

`hbdk4` is an MLIR-based compiler (the wheel carries `hbdk4/compiler/_mlir_libs/libHBDKPythonCAPI.so`
plus `quant`/`linalg`/`gpu` dialects and `hbdk4/compiler/{apis,leap,hbm,march}.py`). `march.py` lists
the Nash family; **`nash-p` has `num_cores == 4`**, matching the S600's four BPU cores — and
`leap_llm` registers qwen3 against exactly `["nash-p"]` in `apis/model/model_factory.py`.

**Obtainability.** These wheels are in our SDK tarball, so we hold them. They are **not** on public
PyPI: `pip index versions hbdk4-compiler`, `hbdk4`, and `leap-llm` each report no distribution, and a
`pip download` fails the same way. So the toolchain is obtainable **only through the D-Robotics SDK**
— which we have — and no login was needed to get it.

## Input format, and what converts a checkpoint to it

`leap_llm/apis/oellm_build.py` takes:

```text
--input_model_format  {hf, llmc, github}   (default: hf)
--input_model_path    <dir>                (required)
--model_name          qwen3                (required; qwen3 is a registered model)
--march               nash-p               (required)
--output_model_path   <dir>                (required)
--chunk_size          256                  (128-2048, multiple of 64)
--cache_len           4096                 (256-4096, multiple of 64)
--w_bits              {4, 8}               (default 8)
--device              cpu | cuda:0         (cuda accelerates calibration)
--verifier / --remote_ip                   (optional on-device check over SSH)
```

The loader (`leap_llm/models/qwen3/model.py:load_model`) reads a **HuggingFace directory**:
`config.json` + `*.safetensors` (via `load_safetensors_state_dict`) + tokenizer. It does **not** read
GGUF — the `choices` list has no `gguf` and there is no GGUF reader in the wheel.

So the `GGUF → HF` leg sometimes drawn before this step is **unnecessary here**: the compiler wants
HF, and the natural source of a 4B/8B Qwen3 is the HF checkpoint directly. Nothing in this path is
NDA-gated for the artifact — the wheels and the doc are in the public SDK. The only vendor gate is on
*what you do with* the toolchain, in the license (below), and a doc line telling custom builders to
contact support.

## The concrete command sequence

On the x86_64 host, in a **fresh Python 3.10 environment**:

```bash
# 1. environment  (doc: en/guide/env_install/x86_env.html)
conda create -n oellm python=3.10 && conda activate oellm
pip install -r oellm_build/requirements.txt
pip install oellm_build/hbdk4_compiler-4.10.1a2.dev202601220400+388766e.develop-cp310-cp310-manylinux_2_17_x86_64.whl
pip install oellm_build/hbdk4_runtime_aarch64_unknown_linux_gnu_nash-4.10.1a2.dev202601220400+388766e.develop-py3-none-any.whl
pip install oellm_build/leap_llm-1.0.0-py310-none-any.whl
```

```bash
# 2. build -- Qwen3-4B HF -> .hbm, cache shrunk to try to fit the board's ceiling
oellm_build \
  --model_name         qwen3 \
  --march              nash-p \
  --input_model_path   /path/to/Qwen3-4B          # HF dir: config.json + *.safetensors + tokenizer
  --output_model_path  /path/to/out \
  --input_model_format hf \
  --w_bits             4 \
  --chunk_size         512 \
  --cache_len          1024 \
  --prefill_core_num   4 \
  --decode_core_num    4 \
  --device             cuda:0
```

The output name is fixed by `nn/utils.py:standard_lm_name` —
`{basename(input)}_language_chunk_{chunk}_cache_{cache}_w{w}_{march}_corenum_{P}_{D}.hbm` — so the
command above yields `Qwen3-4B_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm`, the same
suffix shape as the vendor's shipped graphs.

## The caveats that decide whether it works

These are the reasons the verdict below is "half", not "yes":

- **`corenum` defaults to `1_1`, not `4_4`.** Every shipped LLM `.hbm` is `corenum_4_4`; the
  registry defaults are `[1]`/`[1]`. `--prefill_core_num`/`--decode_core_num` must be passed
  explicitly to match, or the graph is compiled for the wrong core shape.
- **The shrink may not be enough.** The shipped 4B at `cache_len 4096` is 3.10 GiB and the ceiling is
  under that. Dropping cache 4096→1024 and chunk 512 recovers on the order of a few hundred MB of KV.
  That *may* land below the ceiling — the vendor's own VLM 7B language graph is a cache-1024 precedent
  — but it is **not demonstrated**, and no one has built one to check.
- **Host conflict.** `requirements.txt` pins `torch==2.6.0`, while the x86 host this repository's
  notes describe runs a much newer CUDA torch. This is exactly the CUDA-version-mismatch trap the
  repository's host notes record. A **fresh conda env is mandatory**; do not install into the host's
  existing torch.
- **Calibration data.** `--calib_json_path` is optional to argparse but the API calls
  `load_message_data(...)`; quantization quality depends on it, and the calibration text is not part
  of the SDK slice read here. Budget for supplying it.
- **Vendor hardware ask.** The doc recommends an i9-14900K + **RTX 3090** + 128 GB. A smaller card is
  below spec, and the vendor notes that the compile may be slow or fail.
- **License.** The toolchain license bars reverse-engineering it, **transferring or disclosing it to a
  third party**, and using it to build competing products. Building our own model on our own machine
  is within the grant; redistributing the wheels is not.

## The verdict

- **Is the chain obtainable?** **Yes.** Every compiler artifact is already in the SDK tarball on this
  board, with no login. The one leg that does *not* exist in the SDK — a GGUF reader — is not needed,
  because the tool takes HF directly.
- **Does the chain work end to end?** **Yes, for 1.7B, measured.** Our own `cache_1024` `Qwen3-1.7B`
  `.hbm` (1.66 GiB) loads on the board and generates — see
  [the first build result](#the-first-build-result). The mechanism the page scoped is now a build
  record.
- **What runs where?** Compile on **x86_64 / cp310 with an NVIDIA GPU** (the board cannot compile).
  Deploy and run on the **aarch64 S600**, which already carries the matching `hbdk4_runtime_nash`.
- **Is 4B reachable by us, by compiling a smaller graph?** **No — measured.** A **4B `.hbm` from
  D-Robotics is already on the board** (public, login-free, md5-verified) but does not load in
  `cpu_first`, because of the BPU pool ceiling. A **smaller-cache 4B `.hbm` we build ourselves** was
  built — **3,221,638,408 B (3.002 GiB)**, md5 `539775a7c9b596aa470895ad9fc6cf8e`, against the
  shipped 3,326,941,192 B — and it **also refused** with `RESOURCE_EXHAUSTED`: the cache shrink saves
  ~99 MiB, but the 3.00 GiB of weights alone exceed the 2.00 GiB pool. So the recompile is a real
  lever that is simply too small, not a path to 4B/8B. (It runs fine once the pool fits it — see
  [the model page](../models/s600_qwen3.md#the-large-models-after-the-switch) — but by then the
  shipped graph loads too, so the recompile bought speed, not reach.)
- **What *does* reach 4B/8B?** **The vendor's own mode switch, and it did.** `/usr/hobot/bin/hb_switch_ion.sh
  balanced` grows `ion_carveout` from 2.00 GiB to **10.00 GiB**, which fits 4B (3.00 GiB) and 8B
  (5.31 GiB) with no recompile. The SDK ships the tool, documents `balanced` as the setting for large
  models, and lists all four sizes as supported — the board simply booted in `cpu_first`. It was
  **applied and the board rebooted 2026-10-10**; measured result on
  [the model page](../models/s600_qwen3.md#the-large-models-after-the-switch): 4B **42.6 t/s** decode
  and 8B **29.6 t/s**, both coherent and both byte-identical over three greedy runs.
- **Is 1.7B the ceiling by vendor limitation, not ours?** **No** — it is neither, and 1.7B is not the
  ceiling any more. The ceiling was the *mode* the board booted in, and the vendor ships the switch out
  of it. The honest framing is: **1.7B was the ceiling in `cpu_first`; `balanced` was a one-command,
  vendor-supported mode change, it has been made, and all four Qwen3 sizes now run.**

**The numbers in the scoping sections above are not build results** — they are the SDK's stated tool
versions, the vendor's published artifacts, and this board's measured refusal. The first build result is
at [the first build result](#the-first-build-result): a 1.7B `cache_1024` graph out of this chain that
loads and runs. The chain's **4B** build is now also measured
([result](#the-first-build-result)): it loaded nothing under the 2 GiB `cpu_first` pool — the smaller
graph is still 3.00 GiB of weights — and what reached 4B/8B was the vendor's `balanced` switch, not this
recompile. Once that pool was in place the 4B build ran too, at 50.0 t/s decode
([model page](../models/s600_qwen3.md#the-large-models-after-the-switch)); the chain's contribution to
4B was therefore speed on a graph that already loaded, not reach.

## The first build result

The chain's first artifact is our own **Qwen3-1.7B** language graph at **`cache_1024`**, built on the
x86 host (an RTX 2080 Ti, not the doc's recommended 3090) with the SDK's `leap_llm`/`hbdk4` toolchain,
and measured on the board 2026-10-09.

| | shipped (vendor) | ours |
|---|---|---|
| file | `…1.7B_language…_cache_4096_w4_…` | `…1.7B_language…_cache_1024_w4_…` |
| bytes | 1,827,743,336 | **1,779,556,552** (1.66 GiB) |
| md5 | vendor-published | `0b41e627f2227c029b14ed63928fd33f` |
| `hrt_model_exec model_info` | loads | **loads** — prefill `1x512` + decode `1x1`, 28 layers, `logits (1,1,151936)` |
| `pocketllm run --device horizon` | runs | **runs**, coherent text |

**It loads.** At 1.66 GiB the artifact is under the 2.00 GiB `ion_carveout` (the shipped 1.70 GiB
graph fits too; the ceiling only bites at 4B), and `hrt_model_exec model_info` initializes it with no
`RESOURCE_EXHAUSTED`. That was the prediction, and it holds.

**It runs — with one configuration caveat.** The graph was compiled for 4 BPU cores and the `.hbm`
records that; a config whose `bpu_core` does not match fails prefill with *"The number of BPU cores set
in the backend should be the same as … compiled model bpu core num: 4"*. The shipped `llm_demo`
configs carry `"bpu_core": [0,1,2,3]` for exactly this reason, and a config for this artifact must too.
Ours does, and `pocketllm run` then produces coherent text. The exact invocation — the two environment variables, the config, and the command — is [the deployment recipe](#running-this-artifact-on-the-board).

**It is faster.** Decode is **87.6–88.8 t/s** (three runs: 88.84 / 87.90 / 87.59) and prefill
**8982 t/s**, against the shipped `cache_4096` graph's **67.7–69.8 t/s** decode / ~5172 t/s prefill
measured in the same session. The smaller cache is not a cost — it is the reason the graph is smaller
*and* the decode step is cheaper.

**But it is not token-identical to the shipped graph, and that is the interesting part.** The
expectation was that only the KV cache length differs, so a short greedy generation should match. It
does not. Three prompts, both graphs greedy and each verified deterministic (re-running a graph gives
the same text):

| Prompt | first divergence | shipped / ours |
|---|---|---|
| `The capital of France is` | char 56 | both answer **Paris** |
| `The largest planet in the solar system is` | char 138 | both answer **Jupiter** |
| `2 + 2 =` | char 21 | both answer **4** |

The divergence is early — at the first generated token for `2 + 2 =`, within about ten tokens for the
others — and is confined to the **reasoning block**: every prompt converged on the same final answer.
The cause is visible in the artifact itself. `model_info` prints the baked scale of every quantized
tensor, and the **KV-cache** tensors carry *different* scales on the two graphs (`layer_0_cache_key`:
shipped `0.0120087`, ours `0.0120163`; 110 of 112 such tensors differ). A quantization scale is set by
the activation range the compiler measured, not by the tensor's length — so this is a difference in the
two builds' quantization, not in the cache size. The graphs are two build-time quantizations of the same
`Qwen/Qwen3-1.7B` weights, and the cache length is not the only difference. Greedy decode is chaotic — a tiny logit difference at the first token flips the
branch and the rest diverges — which is exactly the shape seen. Token-for-token equality would require
the vendor's exact calibration statistics, which are not published.

So the honest result: the **compile chain is validated end to end** — our own artifact loads, runs and
is faster — while the behavioral match to the vendor graph is at the answer level, not token-for-token.

**The `cache_1024` 4B came out, and under `cpu_first` it did not fit either.** The build that was in
progress when this section was first written has landed:
`Qwen3-4B_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm`, **3,221,638,408 bytes (3.002 GiB)**,
md5 `539775a7c9b596aa470895ad9fc6cf8e`, i.e. **99 MiB smaller** than the shipped 3.10 GiB 4B. In the
`cpu_first` pool that was then live, `hrt_model_exec model_info` refused it just as it refused the
shipped graph:

```text
Cannot malloc bpu memory with length 3221638408 bytes: AllocError { len: 3222994944 }
  -> HBRT4_STATUS_RESOURCE_EXHAUSTED   -> ion_alloc ret=-12 (ENOMEM)
```

So the recompile path **did not** reach 4B on its own: shrinking the cache recovered ~99 MiB against the
~1.0 GiB the graph would have to lose to fit 2.00 GiB, because the weights — not the KV cache — are the
binding term. The path that reaches 4B/8B is the vendor's `balanced` mode switch, which lifts the pool
to 10.00 GiB and fits both. **This is the chain's honest endpoint: the toolchain works, a 4B builds
cleanly, and the pool — not the graph — is what had to move.**

**And the 4B build runs now that it did.** After the switch was applied and the board rebooted
(2026-10-10), the same file **initializes**: `hrt_model_exec model_info` reports `Load model to DDR`,
and through `XlmEngine` it decodes at **50.0 t/s** (49.98 / 49.91 / 49.96 over three byte-identical
greedy runs) against the shipped 4B's 42.6 t/s — the same ~17% decode edge the compiled 1.7B had over
its shipped twin. So the chain's 4B artifact is real and is the fastest 4B on this board; it is simply
not *what made 4B reachable*, which the pool switch had already done. Full numbers on
[the model page](../models/s600_qwen3.md#the-large-models-after-the-switch). Like the compiled 1.7B, it
is a different build-time quantization of the same `Qwen3-4B` weights, so its reasoning text differs
from the shipped graph's where both converge on the same answer.

### Running this artifact on the board

The graph is not shipped; it lives where the build put it. The two environment variables the SDK's
own `run_llm.sh` sets are read by `dlopen`/`libhbrt4` *before* the process starts, and the config is
the shipped `llm_demo` shape with this artifact's two load-bearing keys — `bpu_core` matching the
four cores it was compiled for, and `model_type 9` (Qwen3):

```bash
SDK=~/llm_sdk/D-Robotics_LLM_S600_1.0.2_SDK
export LD_LIBRARY_PATH=$SDK/oellm_runtime/lib   # the SDK's libs live under oellm_runtime/, not $SDK/lib
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6   # the accepted band is ~5.974-<7 MiB/core, so this
                                              # vendor default is also the recommended value

python -m pocketllm run --device horizon \
  --model Qwen3-1.7B_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.json \
  --prompt "The capital of France is"
```

```json
{
  "hbm_path": "Qwen3-1.7B_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm",
  "bpu_core": [0, 1, 2, 3],
  "tokenizer_dir": "<sdk>/oellm_runtime/configs/Qwen3_config",
  "model_type": 9,
  "enable_multi_turn": false,
  "enable_thinking": true
}
```

Drop `bpu_core` and the prefill is refused, as in the caveat above — and since the fix in
`pocketllm.xlm.XlmInferenceError` that refusal is a non-zero exit with a message rather than an empty
answer printed as if it were one. `tests/native/test_compiled_artifact_horizon.py` is this recipe as a
test: it skips when the artifact is absent (naming the path it looked at), and otherwise asserts the
graph answers the canonical prompt, that three greedy runs are byte-identical, and that the `bpu_core`
mismatch is refused rather than silently wrong.

## Where each claim comes from

- Wheels, versions, arch tags: `ls -l` over `oellm_build/` and the extracted wheel trees.
- `hbdk4/compiler/march.py` — `nash_p` ⇒ `num_cores == 4`.
- `leap_llm/apis/model/model_factory.py` — `@register_model("qwen3", ["nash-p"])`.
- `leap_llm/apis/oellm_build.py` — the argparse block (lines 163–400) and `input_model_format`
  choices.
- `leap_llm/apis/model/qwen3.py` — `Qwen3Api.compile()`, `get_hbm_path()`.
- `leap_llm/models/qwen3/model.py` — `config.json` + `load_safetensors_state_dict` (HF input, no
  GGUF).
- `leap_llm/nn/utils.py:standard_lm_name` — output filename format.
- Public model manifest and md5: `oellm_runtime/model/resolve_model_nash-p.md`.
- License terms: `doc/…/en/guide/license_agreement.html` §3.2/3.3/3.6.
- x86 build procedure: `doc/…/en/guide/env_install/x86_env.html`.
- The load-ceiling errors: measured on this board (see
  [the model page](../models/s600_qwen3.md#the-4b-8b-ceiling-applied)).
- The build result: the artifact and its md5 from the x86 host's `build_out/`, `hrt_model_exec
  model_info` over both the shipped and our `.hbm`, and `pocketllm run --device horizon` on the board —
  all measured 2026-10-09. The differing KV-cache scales are the `scale data:` lines `model_info`
  prints for the `layer_N_cache_key`/`layer_N_cache_value` inputs of each graph.
- Not on PyPI: `pip index versions hbdk4-compiler | hbdk4 | leap-llm` ⇒ no distribution.