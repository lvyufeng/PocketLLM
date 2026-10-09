# The S600 native compile chain

[Qwen3 on the RDK S600](../models/s600_qwen3.md) stops at 1.7B because the board's BPU memory pool
refuses the shipped 4B and 8B graphs. Its ceiling section says the only way past that is a `.hbm`
recompiled with a smaller footprint, and calls that path vendor-gated. **This page is the answer to
that sentence**: what the D-Robotics SDK actually ships for building a `.hbm`, where the build runs,
what it takes as input, and — plainly — whether the chain is in our hands or the vendor's.

It is a **scoping document, not a build record**. Nothing here was compiled, installed, or run. Every
claim is either a **measured** fact from this board (labelled) or something **read off an artifact** —
a wheel, the vendor doc set, or the vendor's own manifest in the SDK tree. Each is marked as such.
Whether a cache-1024 4B graph would actually land under the ceiling is still **unproven**.

**The scoping has since been acted on.** The chain has produced its first artifact — our own
`Qwen3-1.7B` graph at `cache_1024` — and it **loads and runs on the board**; see
[the first build result](#the-first-build-result). The page keeps its scoping framing below
because that is the state the rest of it was written in and still describes; the build result
is appended, not woven in.

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
[the model page](../models/s600_qwen3.md#the-4b-8b-ceiling) already measures: `hrt_model_exec
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
- **Is 4B reachable by us?** Half:
  - A **4B `.hbm` from D-Robotics is already on the board** (public, login-free, md5-verified) but
    **does not load**, because of the BPU pool ceiling at the shipped `cache_len 4096`. That is a
    vendor *packaging* limit — every large model ships `cache 4096, corenum 4_4`, with no smaller
    variant — not a gate on us.
  - A **smaller-cache 4B `.hbm` we build ourselves** is mechanically possible with the tools in hand,
    but it is real work (fresh cp310 env, HF checkpoint, calibration data, a 3090-class compile), and
    **whether a cache-1024 4B lands under the ceiling is unproven**.
- **Is 1.7B the ceiling by vendor limitation, not ours?** **Half.** The *prebuilt* 4B/8B are unusable
  on this board by a memory ceiling that is the vendor's packaging choice. But it is **not** a hard
  vendor gate: the toolchain to produce a smaller graph is in our hands. The honest framing is:
  **1.7B is the ceiling for what the vendor ships and this board can load; 4B becomes reachable only
  if we compile our own smaller graph — that build has now been done at 1.7B ([result](#the-first-build-result))
  and a 4B one is compiling.** (The one other path —
  enlarging the 2 GiB `ion_carveout` pool the `.hbm` is loaded through, from the device tree — is
  examined and left untested on [the model page](../models/s600_qwen3.md#is-it-resizable-in-principle-yes-not-by-us-safely),
  because it needs a reboot the board cannot safely undo.)

**The numbers in the scoping sections above are not build results** — they are the SDK's stated tool
versions, the vendor's published artifacts, and this board's measured refusal. The first build result is
at [the first build result](#the-first-build-result): a 1.7B `cache_1024` graph out of this chain that
loads and runs. Whether the chain carries a **4B** graph under the 2 GiB ceiling is still to be shown,
and that build is in progress.

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
Ours does, and `pocketllm run` then produces coherent text.

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
Whether a `cache_1024` **4B** graph lands under the ceiling is still the open question; that build is
compiling.

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
  [the model page](../models/s600_qwen3.md#the-4b-8b-ceiling)).
- The build result: the artifact and its md5 from the x86 host's `build_out/`, `hrt_model_exec
  model_info` over both the shipped and our `.hbm`, and `pocketllm run --device horizon` on the board —
  all measured 2026-10-09. The differing KV-cache scales are the `scale data:` lines `model_info`
  prints for the `layer_N_cache_key`/`layer_N_cache_value` inputs of each graph.
- Not on PyPI: `pip index versions hbdk4-compiler | hbdk4 | leap-llm` ⇒ no distribution.