# Multi-backend and multi-model: PocketLLM against vLLM 0.30 and SGLang 0.5.20

**Date**: 2026-09-26
**PocketLLM baseline**: master `8162937`
**Compared against**: vLLM **v0.30.0** (released 2026-09-22), SGLang **v0.5.20** (released 2026-09-18), and
the local 2080 Ti fork `vLLM-2080Ti-Definitive-v0.1.15`, whose upstream base is vLLM **0.21.0**.

This document exists because the two comparisons already in this directory are no longer against
current engines. [PocketLLM vs vLLM vs SGLang](vllm_sglang_architecture_analysis.md) was written
against the local fork, which is nine upstream releases behind; [the pre-Phase-1 comparison](vllm_sglang_comparison.md)
predates the `pocketllm` control plane altogether. And both of them catalogue *features* — batching,
prefix caching, speculation — while the two questions this repository keeps running into are
architectural:

1. **How does a second hardware backend get added, and what does it cost?**
2. **How does a second model get added, and what does it cost?**

So this page compares on those two axes, against what vLLM and SGLang actually are in September 2026.

**Evidence standard.** Every claim about this repository is a source citation against `8162937` and is
reproducible with the grep or test named beside it. Every claim about vLLM or SGLang is a read of the
`v0.30.0` / `v0.5.20` tags or their release notes, cited by path or issue number, and **not** measured
on this hardware — the numbers in those engines' own blogs are theirs, taken on hardware this machine
does not have. Where a claim about performance is a *difference*, this page says which two figures were
subtracted and how; where it is a single reading, it says so.

---

## 1. The difference in one paragraph

vLLM and SGLang are **one generic data plane with a per-hardware plugin seam**: a common scheduler,
KV-cache manager and model-runner framework, plus an out-of-tree `Platform`/plugin interface that lets
a vendor supply kernels, attention backends and communicators for a device the core does not know.
PocketLLM is the **inverse**: N per-model data planes — five Python runtimes and two C++ engines, each
hand-written against its own checkpoint — sharing one Python control plane, and a *build-time* backend
choice that makes CUDA and Ascend mutually exclusive in a single binary.

Both designs pay for their choice on exactly the axis the other one optimizes:

- vLLM's platform plugins leak. The vLLM RFCs say so in their own words: **#51212** ("Model Runner V2
  Pluggable Design") records that "each backend ends up copying the entire GPU model runner and
  maintaining it independently… ~8000 lines", and **#45133** records that vLLM Ascend "now has
  rewritten over 55 Triton kernels… the current approach uses monkey-patching". A vendor cannot be a
  backend without forking a runner and patching kernels.
- PocketLLM's per-model planes duplicate. `mimo_backend.py` and `xing4_backend.py` share a
  byte-identical `_decode` (`difflib` ratio 1.00), three worker scripts in
  `pocketllm/backends/factory.py` are the same ~45-line program three times, and there are four
  independent prefix-cache implementations in the tree. Adding a model means writing a sixth stack.

The 2026 counter-current matters here: vLLM **#42770** ("Changes in vLLM Model Development") is vLLM
arguing that "we pursued the unrealistic ideal of a single model definition that works well on every
hardware. In practice, each hardware backend benefits from its own model implementation" — and the
newest frontier models have moved to `vllm/models/<model>/{common,nvidia,amd,xpu,cpu}/` with a
platform-dispatching `__init__.py`. The generic-model camp is splitting its own position into "generic
for the tail, per-vendor for the head". That is, in effect, a statement that PocketLLM's core bet is
defensible; what PocketLLM lacks is not the bet but the **seams**.

**§8 is the decision this page records, and it splits the two axes.** Model computation stays
per-model — that is the part both competitors are moving back toward. The request lifecycle and the
scheduler become one implementation over the narrow `InferenceEngine` contract that already exists in
`cpp_engine/include/inference_engine.hpp`, driving every runtime, C++ or Python. The short version is:
**N architecture families over one lifecycle**, not one runtime for twenty checkpoints.

---

## 2. Layer by layer

| Layer | PocketLLM `8162937` | vLLM v0.30.0 | SGLang v0.5.20 |
|---|---|---|---|
| Control plane | `pocketllm/` (8.9k lines): `EngineArgs`, `LLM`/`AsyncLLM`, HTTP server, supervisor, 5 backend adapters | `vllm/v1/engine` + `EngineCoreClient` (`Inproc`/`MP`/`DP`), API server and engine core in **separate processes over ZMQ** | `TokenizerManager` → `Scheduler` → `TpModelWorker` → `DetokenizerManager`, all over ZMQ; optional Rust front end |
| Scheduler | one `BatchScheduler` for the native server; `pocketllm serve --backend cpp` bypasses it by default (see §6.1) | token-budget scheduler, no prefill/decode phase machine, chunked prefill always on, recompute-only preemption, `--watermark` | `Scheduler` + `PrefillAdder`, overlap scheduler **on by default**, retract-then-recompute preemption, LPM/HRRN/priority policies |
| KV | 16-token blocks in a pool; cross-request sharing only in `QwenEngine`, block-aligned; per-model layouts differ (paged GQA vs ring+compressed vs recurrent state) | `KVCacheManager` → coordinator → per-type manager → `BlockPool`; several cache groups with one page size; Mamba/SWA/CrossAttention/Sink managers; pluggable `KVCacheSpecRegistry` | one **unified radix tree** for all models since 0.5.19, per-component tombstones and cascading eviction (`Full > SWA > Mamba`) |
| Prefix cache | **four** implementations: `src/models/prefix_cache.py` (v41+MiMo), `src/models/xing4_0/prefix_cache.py`, `src/runtime/prefix_snapshot.py`, and `QwenEngine`'s global store | one, in the block pool: chained `sha256`, `--prefix-match-unit`, deterministic `NONE_HASH` since 0.29, on by default incl. Mamba since 0.28 | one radix tree, with 0.5.20's SWA branching-point caching |
| Execution | per-model hand-written loops in C++ and Python; no IR | Model Runner V2 (default since 0.29; MRV1 removal targeted v0.32): persistent rows decoupled from input tensors, no CPU sync in the loop, Triton-native input prep, UVA | `ModelRunner` with `eager` / `decode_cuda_graph` / `prefill_cuda_graph` runners |
| CUDA graphs | **zero in `cpp_engine`** (`grep -rn cudaGraph cpp_engine` → 0 hits); one Python path (`src/models/xing4_0/graphs.py`), measured 3.84× | `FULL_AND_PIECEWISE` by default at `-O2`, `BatchExecutionDescriptor` bucketing, per-backend `AttentionCGSupport`, GC frozen during capture | per-phase config; breakable graphs are the CUDA prefill default; decode stays full |
| Collectives | 2 all-reduces/layer, serial with compute except Qwen prefill; CUDA = stock NCCL only; Ascend has a hand-written IPC all-reduce **and** a device-side arrival wait, both default | NCCL symm-mem → FlashInfer → CustomAllreduce (IPC) → PyNCCL → torch; FlashInfer AR default for TP but **gated to sm90+** | `custom_all_reduce_v2` (one-shot push/pull, two-shot, multicall) with NCCL fallback; MSCCL++ |
| Op surface | 86 + 148 CUDA ops declared in headers, **371 launch sites, all of them under `backends/cuda/kernels/`**; engine/ and core/ contain zero launches; Ascend mirrors 45 | `CustomOp` / `PluggableLayer` / `direct_register_custom_op`; `csrc/` ~107k lines | `BaseFusedOp` with `forward_<kernel backend>` × `forward_<platform>`; AOT kernels in `python/sglang/kernels/aot` |
| Model layer | 7 hand-written runtimes under `src/models/` + 2 in C++, registered by architecture string | registry (368 archs) for the tail, `vllm/models/<model>/<vendor>/` for the newest, plus a `transformers` backend that reached native speed (#47187) | registry + a ~200-line `if/elif` in `configs/model_config.py`; ~240 model modules |
| Second hardware backend | `POCKET_BACKEND=cuda\|ascend` at configure time; four CUDA-only TUs excluded on Ascend behind throwing stubs | out-of-tree plugin package (5 entry-point groups, one platform per process); Ascend, Gaudi, TPU, Metal, OpenVINO all OOT | out-of-tree platform plugin with an explicit [Active]/[Planned] contract; **NPU in-tree** with Codeowners and a separate kernel library |
| Second model | a new run-time stack per checkpoint family | a model file + one registry line (dense), or a per-vendor subtree (frontier) | a model file + two registry edits; a dense PR was measured at +881/−6 lines |

---

## 3. Multi-backend, in detail

### 3.1 What PocketLLM has that the others do not

The `core/` / `engine/` / `backends/` split is enforced, not aspirational: `check_layering` in
`cpp_engine/CMakeLists.txt` fails the build if `include/` or `core/` pulls in a vendor SDK header, and
the invariant holds empirically — **zero `<<<` launch sites in `engine/`, `core/` and `include/`**, all
371 of them under `backends/cuda/kernels/` (22 files). The op contract is coarse-grained and
vendor-neutral by construction
(`void* stream` on all but two of 93 declarations in `cuda_ops.hpp`). That is a cleaner kernel seam
than either competitor has: no vendor in the vLLM or SGLang trees gets that guarantee, which is why
both of them end up monkey-patching (vLLM #45133) or accepting a per-vendor model subtree.

Two consequences worth stating plainly, because they are the *reason* the Ascend work was affordable:

- `core/` (loaders, tokenizer, HTTP server, sampler, metrics, block pool) is shared verbatim across
  both backends — ~7.2k lines that did not have to be written twice.
- Parity is testable per operator: `test_*_parity.cpp` holds the Ascend implementation to the CUDA
  one, which is how the KDA/GDN divergence class of bug was caught before serving.

### 3.2 Where the seam stops

Build-time backend selection is the documented decision (route A: "a single engine plus a device
runtime abstraction, not per-backend engine forks") and this page does not argue against it. The
problem is that the abstraction's *coverage* stops well short of "a second backend is a port":

| | CUDA | Ascend | Gap |
|---|---|---|---|
| Ops declared for the Qwen FP16 path | 148 | 45 | **103 ops missing** |
| Engines buildable | 2 (`QwenEngine`, `PersistentEngineAdapter`) + 2 drafters | 1 (`QwenEngine`) | `deepseek_v4_engine.cpp`, `dspark_engine.cpp`, `qwen_dspark.cpp`, `qwen_dflash2.cpp` excluded |
| Tenants of those engines | the whole model set | Qwen FP16 only, and only in the C++ engine | the Python plane (`--backend v41/mimo/xing4/torch`) has **no** NPU path at all |
| Collective | stock NCCL (`ncclAllReduce`) | hand-written IPC all-reduce + device-side arrival wait | the CUDA side has no equivalent optimization |
| Layering check | `core/` and `include/` enforced | same | `engine/` is **not** covered — Phase 1 of the plan is still open |

The last row is the one with leverage. `engine/deepseek_v4_engine.cpp` names CUDA kernels directly in
~120 places, which is why `engine/backend_unimplemented_ascend.cpp` (317 lines of throwing stubs) has
to exist at all: a CUDA symbol referenced from any object in the link must resolve even when its branch
is unreachable. Every model added to `engine/` without going through `backends/api/device_runtime.hpp`
makes the second backend more expensive, and the plan already identified this — extending
`check_layering` to `engine/` is the mechanical version.

### 3.3 What the competitors' plugin seams cost, and what they buy

vLLM's plugin surface is five entry-point groups; a platform plugin returns a class qualname and must
set `worker_cls`, `get_attn_backend_cls`, `get_device_communicator_cls`. What it *buys*: an accelerator
vendor never touches the core, and the ecosystem shows it (Ascend, Gaudi, TPU, Metal, OpenVINO, plus
vendor forks). What it *costs*, in the maintainers' own accounting: the runner is copied (#51212), the
attention enum is closed so a plugin can only override a member or use `CUSTOM`, quant configs are
registered by an import the user must perform, and registration failures used to be swallowed silently
(#48277). Two further facts are worth internalizing: vLLM's **Ascend plugin's stable line is still
vLLM 0.23.0** — the plugin ecosystem lags the core by design, because each release of vLLM is a
compatibility event — and **`--model-impl transformers` now reaches native speed** (#47187), which is
the strongest existing evidence that a well-specified generic interface can avoid a runtime tax.

SGLang's contract is the most honest document in the field: its plugin guide marks each method
**[Active] or [Planned]**, states that planned ones "will NOT take effect until the core is migrated",
and says outright that "device-family branches remain outside the interface" and that vendors should
"report that gap". Its NPU path is in-tree with named Codeowners and an external kernel library
(`sgl-kernel-npu`), which is the opposite trade from vLLM: less plugin purity, more likely to work.

The contrast worth copying is **llama.cpp**, which is the only engine here with a *fully* pluggable
backend: `ggml_backend_reg_t` / `ggml_backend_dev_t`, a runtime `.so` search honouring
`GGML_BACKEND_PATH`, and a score function that lets a device bid for placement. Sixteen backends exist
under `ggml/src/`. What it costs is visible too: ops must be expressible in the ggml IR, the quant type
set is a compile-time enum with a static traits array (`ggml_type_traits[GGML_TYPE_COUNT]`), so a new
quant format is a core edit — which is precisely the position PocketLLM is in with GGUF, except
PocketLLM chose that position deliberately and gets the whole low-bit family with it.

**Reading for PocketLLM.** The kernel seam is the one that pays. PocketLLM already has it and should
extend it rather than replace it: the missing coverage is op count on Ascend (a kernel-task problem,
not an abstraction problem) and the four CUDA-only TUs in `engine/` (a layering problem the plan
already scoped). A runtime plugin mechanism is not needed for a two-backend, build-time-selected
engine — but a *capability report* per backend is, because the Python layer currently recomputes it by
hand three times (see §4.3).

---

## 4. Multi-model, in detail

### 4.1 How a model arrives in each engine

| | Mechanism | Measured cost of a new dense model | Cost of a new frontier model |
|---|---|---|---|
| PocketLLM | new runtime stack under `src/models/<name>/` + a registry key + an adapter | not measured; the *engine* is reused when the architecture matches (Bonsai reuses Qwen's) | `src/models/` per-model directories run 1.7k–9.4k lines |
| vLLM | registry line + model file | `llama.py` 601, `qwen3.py` 340 lines | per-vendor subtree; Kimi-K3/GLM-5.3 class models are a project |
| SGLang | model file + `EntryClass` + a config `if/elif` arm | +881/−6 across 9 files for one recent dense model | GLM-5.3-Flash was 103 files / +7,740 (−558) |

PocketLLM's genuinely efficient case is the one the new-model roadmap already exploits: **Bonsai
reuses Qwen3.8-27B's runtime field for field** because the GGUF declares `qwen35`. That is per-model
runtime *reuse by architecture*, which is the right unit — and it is exactly what a `ModelRuntime`
spec would make explicit rather than incidental.

### 4.2 The five-stack problem, quantified

Serving entry points end to end as of `8162937`:

| # | Entry | Processes | Who runs the forward | HTTP server |
|---|---|---|---|---|
| 1 | `pocketllm serve --backend torch` | rank 0 + N−1 | `DeepSeekServingEngine` queue → `src/models/deepseek_v4` | Python (`pocketllm/server/openai.py`) |
| 2 | `pocketllm serve --backend cpp` | 1 or rank0+N−1 | `pocketllm_cpp.QwenEngine`, token by token from Python | Python |
| 3 | `pocketllm serve --backend v41` | rank0 + N−1 | `src.models.deepseek_v4_1.generate` | Python |
| 4 | `pocketllm serve --backend mimo` | rank0 + N−1 | `src.models.mimo_v2.generate` | Python |
| 5 | `pocketllm serve --backend xing4` | **1** (no TP path) | `src/models/xing4_0/generate.py` | Python |
| 6 | `pocketllm_engine --serve --tp-world 4` | 4 shell-launched processes | `QwenEngine` or `PersistentEngineAdapter` | **C++** |
| 7 | `torchrun -m src.server.openai` | N | `src/models/deepseek_v4.generation` | Python (**its own** server) |

Five request-lifecycle implementations, three schedulers (`BatchScheduler` in C++,
`DeepSeekServingEngine`'s bounded queue with `max_running_requests` default **1**,
`PDScheduler`'s phase-resource switching), four HTTP servers, and five IPC mechanisms (NCCL, the C++
`CmdChannel` unix socket, the per-worker `work_bell` doorbell, `broadcast_object_list`, and the
Python sidecar pipepair for chat templating).

The duplication is measurable at file granularity:

- `mimo_backend.py` (1,006 lines) vs `xing4_backend.py` (776): a method-level diff finds ~288 lines of
  same-named methods at ≥0.5 similarity, and **`_decode` is byte-identical** (`difflib` ratio 1.00,
  verified). `stream` is a ~65-line clone differing in a thread name.
- `_publish_cache_metrics` exists three times (v41, mimo, xing4) with the same nine keys.
- The three `run_worker` scripts in `factory.py` (lines 571, 620, 675) are the same program three times.
- `_IGNORED_OPTIONS` is written out three times.
- OpenAI request parsing/validation exists twice, in two languages: `pocketllm/protocol/` (575 lines)
  and `cpp_engine/core/openai_request_fields.cpp` + `openai_stop_strings.cpp` + `json_constraint.cpp`
  (1,296 lines).

### 4.3 Three copies of "what does this backend support"

`BackendCapabilities` is declared once but computed by hand per adapter, and the *rejection* logic is
duplicated: `_reject_unsupported_*` appears four times in `factory.py` (lines 189/225/262/291), each
adapter keeps its own `_IGNORED_OPTIONS` frozenset, and `supports_prefix_caching` is set to "the store
exists" rather than "the flag is on" (xing4:495, v41:653). vLLM's equivalent is the `AttentionBackend`
capability set (`supports_head_size/dtype/kv_cache_dtype/block_size`, `is_mla/is_sparse/is_ssm`,
`supports_sink/sliding_window/batch_invariance`, and — new in 0.30 — `supports_dcp()/supports_pcp()`),
where an implementation that does not declare a capability **fails at backend selection instead of
after weight load**. That is the shape worth borrowing: one declaration per runtime, consumed by the
dispatcher, instead of a validator per adapter.

### 4.4 Verification, which is where this bites hardest

`docs/models/README.md` defines a status ladder and requires evidence for each rung. The test suite
does not enforce it:

- **CI runs no tests.** The two workflows build a wheel and a docs site. Everything below is a local
  convention only.
- `tests/test_cpp_backend_batching.py` is not a pytest test — it inserts a path, takes `sys.argv`, and
  prints `SKIP` and returns when under-specified. **Under pytest it is a silent false pass.**
- The only real concurrency test (`tests/test_cpp_scheduler_streaming.py`) needs a GPU *and* a fixture
  binary written by a C++ test into `/tmp`; without it, it skips.
- There is **no golden end-to-end output for any served model**. Parity coverage exists per op
  (`test_*_parity.cpp`, 87 targets) and per feature (78 prefix-cache tests, two of which run on CPU),
  but nothing pins "this checkpoint served through this entry point produces these tokens".
- The suite's baseline failure set (9 failures + 5 errors, with the collection-time failure in
  `tests/test_gguf_q2_precision.py`) is a known quantity — a *set diff*, not a count, is what detects
  a regression.

For a repository whose model pages make per-model claims at four levels of evidence, the absence of a
served-path fixture is the largest process gap in this document. vLLM and SGLang both gate on
generation tests registered into CI suites; PocketLLM's equivalent is a `bench_*` script that a human
reads.

---

## 5. Where the divergence is right, and should be defended

These are the places where "just make it like vLLM" would destroy the thing that makes this engine
worth having. The repository's own rules say not to unify vendor kernels; this section is the same
argument applied to the newer evidence.

1. **Per-model kernel-level specialization.** TensorRT-LLM's own DeepSeek-V4 blog measures +64.5%
   end-to-end on one model generation from model-specific work alone (984 → 1,618 tok/s/GPU on GB300),
   with operator-level gains of 1.2–2.3×. vLLM's #42770 is the same conclusion arrived at from the
   other direction. PocketLLM's GGUF/DP4A/MMA prefill kernels, its MLA and hyper-connection ports, and
   its IQ4_NL codebook are this camp's work.
2. **GGUF and low-bit first.** vLLM **removed GGUF from the tree** (`vllm-gguf-plugin`, described by
   its own docs as "highly experimental and under-optimized"); bitsandbytes left in 0.28; PagedAttention
   was deleted in 0.25. SGLang went the other way (in-tree GGUF loaders for specific 2026 models, plus
   `ExpertPack`). PocketLLM runs `Q2_K`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`, `IQ1_M`, `PTQ1_0` and GGUF
   tokenizers. That is a capability the generic engines do not have, and it is the capability this
   hardware needs: a 22 GiB sm_75 card with no bf16, no FP8 and no FP4 can only run some of these
   checkpoints through low-bit GGUF at all.
3. **The C++ engine as a latency product.** Two engines behind one Python face is unusual, but the
   measured head-to-head against the local vLLM fork (1.019–1.093× decode, 0.957–1.023× prefill at
   TG=128, TP4) and the concurrency scaling (2.14×/3.61×/4.68× at 2/4/8 against vLLM's
   1.52×/2.64×/3.82×) are the *reason* the C++ path exists.
4. **Build-time backend selection.** A runtime vtable in the decode hot path on a card whose decode is
   already close to bandwidth-bound is a cost with no benefit at two backends. Keep route A.
5. **Resident-vs-offload decisions per checkpoint.** The 2026 SGLang SSD Expert Pack work (Q2_K/Q3_K
   GGML expert-major packs, O_DIRECT, VRAM LFU/LRU cache; 6.92× decode over Ollama on
   DeepSeek-V4-Flash) is the same problem this repository has been solving by hand in
   `deepseek_v4_1/device_experts.py` and `components/moe/placement.py`. Independent arrival at the same
   design is confirmation, not redundancy.

---

## 6. Where PocketLLM is behind, ranked by what it costs

### 6.1 The validated fast path is not the default path

`pocketllm serve --backend cpp` drives `QwenEngine` **token by token from Python** and is serial:
`BackendBase` holds a request lock, `supports_batch` is `False` unless `enable_batching` is set, and
`enable_batching` is reachable **only** as `--backend-option`, not as a CLI flag — `--max-batch-size`
alone does nothing because it is read only when batching is already on. The native `--serve` binary, by
contrast, owns a `BatchScheduler` with a waiting queue, chunked prefill and per-choice requests, and it
is the path the concurrency validation measured.

The same shape appears on DeepSeek: `PersistentEngineAdapter`'s `continuous_batching` reads
`POCKETLLM_CPP_BATCHED_DECODE`, which **defaults to 0**, so the scheduler is clamped to width 1
regardless of `--max-batch-size 8`; `paged_kv` and `chunked_prefill` are also false on that engine.

So the repository contains a validated, faster serving path that its own CLI does not select. Three
concrete fixes, in order of value: expose `enable_batching` as a flag and default it on for the cpp
backend; make `--max-batch-size` imply batching; and make the DeepSeek default width match the
documented batch width, or make the clamp an explicit error instead of a silent one.

### 6.2 No captured decode step in the C++ engine

This is the largest single measured headroom in the repository. The evidence is entirely local:

- Xing4.0, **one card**, eager decode step 148–178 ms against a replayed graph at **38–38.6 ms — 3.84×**
  ([the decode-graph record](../performance/xing4_0_decode_graph.md)), with the bucket itself costing
  nothing (147.5 vs 148.1 ms) and capture exact to the bit against its own width.
- The launch-count probe found the reason: **22,155 ATen dispatches a step, 10,508 of which launch no
  kernel**, against 11,536 launches and 46.5 ms of device work.
- In C++, the analogous blocker list is concrete and different: DeepSeek's decode names ~53 ops per
  layer plus ~17 in MoE (~2,000–3,000 launches a step), and the hard blockers are the **per-layer
  blocking D2H** (`memcpy_d2h` on route indices at `deepseek_v4_engine.cpp:5042`, then a host loop that
  names expert tensors by string, then 6 async H2D per staged expert), host-built attention index
  vectors H2D'd per row per layer (`:4022-4073`), 46 `device_synchronize()` calls, 203 pageable H2D
  copies, and `device_malloc`/`device_free` per call.

Both competitor designs name the fixes: vLLM solves position-dependence by padding batches onto
capture-size buckets (`round_up(num_tokens, 1+num_spec_tokens)`) and marks padding rows so MoE kernels
skip them, and solves non-capturable input preparation with `StagedWriteTensor` + UVA + Triton metadata
kernels; SGLang solves the phase problem by capturing decode and prefill separately and letting
breakable graphs handle prefill. Xing4's bucket ladder is the local instance of the first idea. **Qwen
decode in the C++ engine is the plausible first target** — 4 syncs, no per-layer D2H, and
`reserve_paged_slot` is a no-op except at block boundaries — while DeepSeek needs the per-layer D2H
removed before capture is even meaningful.

### 6.3 Scheduler features that matter at a stated goal, and those that do not

| Feature | PocketLLM | vLLM 0.30 | SGLang 0.5.20 | Verdict for this repository |
|---|---|---|---|---|
| Token budget | none; block budget charging `worst_case_blocks` = prompt **+ max_new_tokens** | `max_num_scheduled_tokens`, defaults 8192 offline / 2048 server | `PrefillAdder` with `rem_total_tokens`, new-token ratio | **Worth it.** Charging the full future footprint at admission is why head-of-line blocking hurts. |
| Preemption / priority | none by design ("head-of-line blocking is deliberate", `batch_scheduler.cpp:298-304`) | recompute-only, victim by priority or FCFS-tail | retract-then-recompute, `--retraction-policy length\|priority` | **Low value here.** A 22 GiB card that admits on the full ISL rarely needs to preempt; the single-request goal dominates. |
| Watermark / reserve | none | `--watermark` (fraction of blocks kept free) | SWA/Mamba admission gates | **Cheap.** One knob, and it directly reduces the eviction churn the prefix cache causes. |
| Chunk size | fixed 4096 | per-step from the budget | derived from VRAM (20–35 GiB → 4096) | Already tuned; leave it. |
| Prefix sharing | Qwen only, block-aligned; 4 implementations | in the block pool, `--prefix-match-unit` | one radix tree, default everywhere | **Worth it**, as unification rather than as new machinery. |
| Batched prefill | `batch_prefill` is **serial per request** (`qwen_engine.cpp:5414`) | one batch mixes prefill and decode by construction | chunked prefill with mixed chunk opt-in | **The largest unclaimed measured lever here** — see §6.4. |

### 6.4 One measured throughput lever this repository has already priced

Independent of any comparison, the serving record leaves one large number on the table: **merging the
admission wave into a single prefill forward, worth ~1.6× rather than the ~1.06× quoted in the engine's
own comment** ([serving throughput scaling](../performance/serving_throughput_scaling.md), §"next
steps" item 2). The arithmetic is local and it is tight: a 325-token prompt costs 331 ms of prefill, of
which **131 ms — 40% — is paid before the first token-dependent FLOP**, because a forward pass issues
129 collectives whatever the prompt's width. At `L16` that is 5,385 ms of wave prefill against 3,414 ms
merged, and 49 against ~78 TFLOP/s on four cards. The merge also moves TTFT and prefill TFLOPS
*together*, which almost nothing else in this tree does.

It is unclaimed for a stated reason, not an unknown one: the 48 linear-attention layers carry
per-sequence state, so a merged forward needs a **segmented recurrence**, and the 16 full-attention
layers need a **block-diagonal mask** — without which a 5,162-row forward spends 16× on the dense
attention matrix what sixteen 322-row forwards spend in total. That is a kernel task with a measured
ceiling, which is the best kind of task this repository has. (The half that was a handover rather than a
forward is done — #353 gave each row its token when produced, worth −42% mean TTFT at `L16`, touching no
kernel.)

### 6.5 Collectives and overlap, where the Ascend lesson has not been ported

vLLM 0.30's `--async-scheduling` is auto-on, and MRV2's step loop has **no CPU sync point** and no
async barrier. SGLang's overlap scheduler is on by default. PocketLLM's only overlap is Qwen's prefill
slice overlap (`projection_all_reduce_overlapped`, slice ceiling 4, floor 1024 rows/slice), and its
comments record that decode takes the serial path at every setting; DeepSeek has no overlap at all.

The interesting thing is that this repository has already *measured* the same lever and shipped it —
on Ascend. The hand-written IPC all-reduce took rows=1 from 106.33 ms / 9.41 TPS to 77.32 ms /
12.94 TPS (**+37.7%**), and the device-side arrival wait took it further to 39.4–39.8 ms /
25.15–25.37 TPS, both now defaults. The mechanism was not the bytes; it was deleting the host round
trip around each collective (the first integration paid 12.9 ms of host ordering it did not need). **The
CUDA path has no such optimization — every collective there is stock `ncclAllReduce`.** A CUDA IPC
all-reduce is therefore a well-motivated hypothesis with a local precedent, and it is exactly the kind
of lever this repository's rules require to be measured in one process with interleaved arms before it
is believed.

### 6.6 Structured output and the OpenAI surface

PocketLLM has JSON mode and JSON-Schema constraints in the C++ engine (`core/json_constraint.cpp`,
743 lines) but **no grammar or regex (GBNF) constraints**, and request-level token constraints are
explicitly rejected on the TP>1 path. Both competitors converged on **xgrammar** as the default
(one backend per engine, not per request) and both keep the FSM work integrated with speculative
verification — vLLM applies a grammar bitmask inside `sample_tokens()`, SGLang has spent 2026 on
"grammar × spec verify" specifically. If the roadmap's goal of dropping into Cursor/Continue/Open WebUI
is taken seriously, grammar-constrained decoding is the missing piece, and it is bounded work because
`json_constraint.cpp` is already an FSM-shaped constraint over the sampler.

Two smaller protocol gaps with named evidence: the native and Python paths disagree about semantics
(`cached_tokens` behaviour and TTFT definitions differ for the same checkpoint depending on which
server you started), and two servers can be run against the same checkpoint in one namespace with
different `supports_batch` answers.

### 6.7 The version problem in the existing comparison

[The current comparison](vllm_sglang_architecture_analysis.md) is a good document against the wrong
baseline. Its vLLM column is the local 2080 Ti fork: upstream base **0.21.0**, fork version 0.1.15,
squashed to a single commit. Upstream has since shipped **0.22 → 0.30** (nine releases, ~5,000 commits).
Things in that document that no longer describe upstream:

- "PagedAttention" — **deleted in 0.25** (`#47361`); V0 is gone (`docs/usage/v1_guide.md`: "We have
  fully deprecated V0").
- The scheduler — V1's design is unchanged in shape (token budget, no phase machine, recompute-only
  preemption) but now has `--watermark`, `--max-num-queued-reqs/tokens`, `scheduler_reserve_full_isl`,
  DP prefill balancing, and an `AsyncScheduler`.
- The worker — **Model Runner V2 is the default since 0.29 and MRV1's removal is targeted for v0.32**.
  Every statement about `gpu_model_runner.py` now describes a deprecated path.
- Prefix caching — on by default including Mamba since 0.28, deterministic `NONE_HASH` since 0.29,
  `--prefix-match-unit` for the hash block size.
- Batched-token defaults — 22 GiB-class cards now default to 8192 offline / 2048 with an API server,
  not 2048/128.
- GGUF and bitsandbytes — **out of tree**.
- All-reduce — FlashInfer's fused all-reduce is the default for TP CUDA groups but **capability-gated
  to sm90/100/103/107**, so on sm_75 the fork's `CustomAllreduce` remains the fast path. The fork is
  not simply "behind"; parts of it are the only working path for this hardware.

The honest framing for the head-to-head table is therefore "**against the 2080 Ti fork of vLLM 0.21.0**",
and it should stay that way until someone ports the comparison forward. Also stale on that page: it says
PocketLLM has no cross-request prefix caching, which stopped being true when V4.1 (#342), MiMo (#379)
and the C++ `QwenEngine` global cache landed.

---

## 7. Recommended sequence

The order is by (value × confidence) ÷ risk, and every item is scoped so it can be measured the way
`docs/guides/benchmarking.md` requires: one configuration per process, arms interleaved, a null arm
where the difference is small.

**Tiers 1 and 2 are the implementation of the decision in §8** (scheduler unification, stages R1 and R2
of the refactor project); Tier 3 is the per-model and per-backend work that follows it. Read §8 first if
the split between "one lifecycle" and "per-model implementations" is the question.

### Tier 1 — make what exists reachable

1. **Default `pocketllm serve --backend cpp` to the batch path.** Expose `enable_batching` as a CLI
   flag, default it on, and let `--max-batch-size` imply it. Reuse the existing concurrency harness
   (`scripts/bench_cpp_openai_concurrency.py`) and record the ladder. Same for
   `POCKETLLM_CPP_BATCHED_DECODE=1` on the DeepSeek adapter, or turn the clamp into a startup error.
2. **Fix the false-passing test** and add one served-path golden fixture per entry point: prompt,
   environment, and the token ids that come out. `docs/models/README.md`'s status ladder currently
   rests on run records a human read; a fixture is what makes it checkable.
3. **Report capabilities from one place.** One `RuntimeCapabilities` per model runtime — attention
   kinds, cache kinds, prefix-cache support, graph-ability, TP rule, batch support — consumed by
   `factory.py`, replacing the four `_reject_unsupported_*` bodies and three `_IGNORED_OPTIONS` sets.

### Tier 2 — collapse the per-model duplication without collapsing the per-model code

4. **One adapter parameterised by a runtime spec.** `mimo` and `xing4` already share a byte-identical
   `_decode`, `TokenStreamer`, `byte_size` and `settled_text`; the divergence that matters (rank
   collectives, single-process, prefix-cache shape) belongs in the model module, not the adapter. The
   1,006/776/1,333-line adapters should become one adapter plus three specs. Keep the per-model
   *runtimes* exactly as they are — that is the part the evidence says is right.
5. **One prefix-cache interface, three implementations behind it.** The shared store already has the
   right shape (block-chained hash over tokens, geometry mixed into the seed, whole-prefix anchors
   rather than a radix tree, LRU under a byte budget). Give the C++ `QwenEngine` store and the Xing4
   latent store the same interface and metrics names, and delete `runtime/prefix_snapshot.py` or
   port its one consumer. Note that a radix tree is *not* the goal — vLLM itself uses a chained block
   hash, and SGLang's tree is a response to multi-tenant routing that a single-user box does not have.
6. **Extend `check_layering` to `engine/`.** This is Phase 1 of the existing multi-backend plan, and
   it is the only item that makes the *next* model cheaper on both backends instead of just on CUDA.
   The four CUDA-only TUs are the work; the verification is the existing CUDA regression suite plus
   the Ascend parity targets.

### Tier 3 — close the headline gaps

7. **Merge the admission wave into one prefill forward** (§6.4). Segmented recurrence for the 48
   linear-attention layers, block-diagonal mask for the 16 full-attention layers, expected ~1.6× on
   wave prefill and TTFT together at the serving ladder's prompt length. This is the only item on this
   list whose ceiling was already measured before the work started, and it is CUDA-and-Ascend neutral
   (the measurement is Ascend's; the constraint — per-sequence state — is Qwen's architecture, not the
   backend's).
8. **Capture the Qwen decode step in C++.** Start from the Xing4 measurement: the goal is a replay near
   38 ms rather than an eager step in the 150 ms class, and the prerequisites are the ones vLLM and
   SGLang both document — no host-dependent H2D inside the captured region, no position read from
   Python, padding onto a bucket ladder. Qwen decode is the first target because it has 4 syncs and no
   per-layer D2H. Before touching DeepSeek, remove its per-layer route D2H; nothing else will matter
   until that is gone.
9. **Port the Ascend collective work to CUDA — as a measured hypothesis.** The Ascend numbers (+37.7%
   from the IPC all-reduce, then 39.4–39.8 ms with the device-side wait) make this the best-motivated
   unported optimization in the tree. On this box NVLink exists only between GPUs 2 and 3, so the
   claim to test is specifically a **PCIe-topology decode** one, TP4 across 0-3, arms interleaved in one
   process.
10. **Grammar-constrained decoding (GBNF/regex) on the existing constraint machinery**, integrating with
    speculative verification the way both competitors do it. Bounded because `json_constraint.cpp` is
    already an FSM over the sampler.
11. **Admission tuning rather than a new scheduler**: a token budget and a `--watermark`-style reserve,
    keeping FCFS and head-of-line blocking. Preemption and priority are explicitly *not* recommended —
    no measured need at the stated goal, and they add the state that makes `BatchScheduler` simple.

### Non-goals, with reasons

- **Do not build a generic graph IR or a hw-agnostic model definition.** llama.cpp's IR costs a closed
  quant-type enum and a `supports_op` false-positive class of bug; vLLM is currently unwinding
  abstractions at the model boundary (#42770). PocketLLM's head models are all per-vendor by nature.
- **Do not port CUDA kernels to Ascend or vice versa.** Already the rule; the 45-vs-148 op gap is a
  kernel backlog, not evidence for sharing code.
- **Do not add preemption, KV swap, or PD disaggregation for a single-user box.** Both competitors need
  them for multi-tenant routing; this engine's own head-to-head result is a single-request
  latency claim.
- **Do not add multi-model-in-one-process.** vLLM closed that request as not-planned (#21481), SGLang
  answers it with a router, and the 2026 infrastructure answer is below the engine (kvcached's KV
  virtualisation, SGLang's `ExpertPack`). PocketLLM's supervisor + host expert bank already is that
  answer for this hardware.

---

## 8. Decision: model implementations stay separate; the request lifecycle and scheduler become one

This section is the decision this page exists to record. It is stated as a boundary rather than a
direction, because the failure mode on both sides is real: a generic data plane costs the per-model
kernels that are this engine's measured advantage (§5), and five per-model serving stacks cost five
copies of the same correctness risk (§4.2).

**The decision has two halves, and they point the same way.**

- **Model computation stays per-model.** Attention, KV layout, quantization format, expert placement,
  the forward's op sequence and its tuning knobs are written per checkpoint family and are not to be
  abstracted into a generic layer set. This is §5, and the evidence for it now includes vLLM's own
  position (#42770) and TensorRT-LLM's measured +64.5% from model-specific work.
- **The request lifecycle and the scheduler become one.** A prompt is admitted, chunked, KV-accounted,
  sampled per row, cancelled, streamed and reported to the client by **one** implementation, and that
  implementation drives every runtime — C++ or Python — through the same interface.

### 8.1 The interface already exists, and its design intent is already written down

`cpp_engine/include/inference_engine.hpp:200` is the contract, and it argues for itself:

> This is deliberately the smallest set that `BatchScheduler` actually calls, **not a general model
> API**: slot lifecycle, paged-KV accounting for admission, the two batched forward entry points, and a
> capability declaration. Everything model-specific — weight layout, attention kind, speculative
> decoding, prefix reuse, TP worker protocol — **stays on the concrete engine, where it can keep its own
> types**.
>
> Every method here is called once per scheduler iteration, never per token and never per layer, so
> dispatching them virtually cannot show up against a batched forward pass. Layer-level components are
> concrete types for the opposite reason.

That is the boundary this decision asks for, already implemented, with the cost question already
answered: five virtual calls per scheduler iteration is not a hot path. `Capabilities`
(`inference_engine.hpp:9`) is the other half and records a mistake the project has already made and
fixed:

> What an engine can actually do, declared rather than inferred. The scheduler used to read
> `kv_total_blocks() == 0` as "this engine uses a contiguous arena", which happens to be true for the
> one engine that existed but is an inference from an accounting field, not a statement of capability.

So the scheduler is already **one implementation over a narrow contract with explicit capability
declarations**, and it already drives two engines with very different internals. What is missing is not
the design. What is missing is that it drives only the *native* path.

### 8.2 What actually blocks it

**(a) The Python runtimes do not implement the contract.** They are separate processes with separate
stacks, and each of the three has its own documented reason:

| Runtime | What the record says stands in front of the scheduler |
|---|---|
| v41 | "What it still has none of is **batching, continuous batching** and an MTP layer" ([design record](deepseek_v4_1_flash_design.md)) |
| MiMo | "Batching, a scheduler — **Not implemented — one request at a time**" ([design record](mimo_v2_6_flash_design.md)) |
| Xing4 | "`supports_batch` is **False** — the trunk's forward flattens its input to one token axis, so two sequences given to it together would **attend to each other**" ([design record](xing4_0_29b_a4b_design.md)) |

Xing4's is the structural one: its forward has a token axis but no independent value-batch axis. All
three need per-row sampling and a prefill entry that accepts a budget and can resume.

**(b) The Python control plane has no scheduler — it has a lock.** `pocketllm/backends/base.py:75`:

> The lock is intentionally at the backend boundary. Current native engines own one mutable KV-cache
> transaction, so concurrent calls must **serialize until a request-aware cache scheduler is
> implemented**.

The result is that **the same checkpoint is served by two stacks with different semantics**: the native
binary owns `BatchScheduler`, while `pocketllm serve --backend cpp` drives the engine token by token
from Python under a lock. They disagree about `supports_batch`, about `prompt_tokens_details.cached_tokens`,
and about TTFT — for the same model, on the same host, on the same port number.

### 8.3 The migration ladder already exists, and needs no big-bang

`Capabilities` makes the migration incremental: `continuous_batching = false` with `max_slots = 1` *is*
"this engine can only run one at a time", and the scheduler already knows how to drive such an engine.
The precedent is `PersistentEngineAdapter` (`engine/persistent_engine_adapter.cpp:55`):

```cpp
c.continuous_batching = max_slots_ > 1 && engine_->batched_decode_enabled();
```

with the comment that `Capabilities` should not have, because both directions of error were already
observed:

> a hardcoded false silently discarded `--max-batch-size`, and a hardcoded true would advertise
> concurrency the engine does not deliver.

**So step one is not "make v41/mimo/xing4 batchable". Step one is "register them under the one
scheduler, declaring honestly that they are width 1".** Each runtime's width then improves on its own
schedule, and the scheduler does not change for it.

### 8.4 Where the scheduler lives: the C++ library, driven from either host — recommended

Three reasons, in order:

1. **It is the only real scheduler in the tree.** Waiting queue, block-budget admission, chunked
   prefill, one request per choice, and a measured 4.68× at 8 concurrent requests against vLLM's 3.82×.
2. **It is already exposed to Python.** `cpp_engine/python/bindings.cpp:591` binds it as
   `QwenBatchScheduler` with `submit_request` / `poll_result`, and `pocketllm/backends/cpp_backend.py:800`
   is already its client. The path is open; it is bound to `QwenEngine*` and needs to be bound to
   `InferenceEngine*` instead.
3. **Moving it to Python would rewrite the `Capabilities` lessons and add a layer to the C++ decode
   path** — and that path is the entire reason the C++ engine exists.

The resulting shape: **the scheduler is a library with two hosts** (the Python control plane and the
native binary) rather than two schedulers. The Python runtimes become out-of-process workers on the
same control channel the C++ TP ranks already use (`core/cmd_channel.cpp` is a unix-domain socket and
carries no language-specific assumptions). Whether a model's forward is C++ or Python becomes invisible
to the scheduler.

One consequence has to be taken with it: **the C++ HTTP front end** (`engine/openai_server.cpp`, 2,117
lines, with its own tokenizer, prefix cache, cancellation, SSE and metrics) cannot stay a second
implementation. The fix is not to delete the native binary — its latency path is the product — but to
make the front end a library too, so the native binary is a **host of the same implementation** rather
than a copy of it. In the same commit the two capability declarations (`pocket::Capabilities` and
`pocketllm.api.types.BackendCapabilities`) become one, because they are the same fact in two languages.

### 8.5 The boundary

| One implementation | Written per model, deliberately |
|---|---|
| `BatchScheduler`: admission, chunking, fairness, per-choice requests | attention implementation (MLA / DSA / GDN / SWA / hyper-connection) |
| `InferenceEngine`: slot lifecycle, KV accounting, the two batched forwards, capabilities | KV layout and its kernels |
| the worker control protocol (five IPC mechanisms today) | quantization format and its kernels |
| HTTP / OpenAI / SSE / cancellation / metrics | expert placement decisions |
| the prefix-cache **interface** and its metric names | prefix-cache implementations (they are geometry-specific) |
| adapter: one, plus a per-runtime spec | the forward's op sequence and tuning knobs |
| test harness and served-path acceptance | the single-request latency path |

The per-model side of that table is exactly what §5 defends. The left column is what the audit in §4.2
found duplicated — five lifecycles, three schedulers, four HTTP servers, five IPCs, `_decode` byte-identical
between two adapters.

### 8.6 Where this reaches its limit

This is **not** a claim that 20 checkpoints become one runtime. vLLM's own layout is the honest
precedent: frontier models under `vllm/models/<model>/{nvidia,amd,xpu,cpu}/` with per-vendor
implementations, the registry for the tail. The shape to plan for is **N architecture families over
one lifecycle**, with a small number of bespoke runtimes that genuinely do not fit. A family boundary
is real here and is already measurable: `attention.py` is 0.85-similar between `qwen4_exp` and
`xing4_0`, and the checkpoint roadmap itself is ordered by reuse — Bonsai reuses the Qwen3.8-27B runtime
field for field, and GLM-5.3-Flash is last because it joins two half-built paths.

### 8.7 Sequence

The order matters and the first item is not negotiable.

1. **Repair the acceptance surface first.** `tests/test_cpp_backend_batching.py` is not a pytest test —
   it inserts a path, takes `sys.argv`, and prints `SKIP` and returns, so **under pytest it is a silent
   false pass**. CI runs no tests, and the suite's baseline is 9 failures + 5 errors with a
   collection-time failure. Unifying the lifecycle while regressions are invisible means trading the
   only working acceptance mechanism for an abstraction with no evidence behind it. Fix the test, add
   one served-path golden fixture per entry point, and record the baseline as a *set*, because a falling
   count hides a new failure.
2. **Make the existing scheduler reachable** (§6.1): `enable_batching` becomes a CLI flag and defaults
   on for the cpp backend, `--max-batch-size` implies it, and the DeepSeek clamp either defaults on or
   fails loudly. No runtime changes.
3. **One capability declaration**, consumed by `pocketllm/backends/factory.py`, replacing four
   `_reject_unsupported_*` bodies and three `_IGNORED_OPTIONS` sets.
4. **Bind the scheduler to `InferenceEngine*`** instead of `QwenEngine*`; the Python server drives it.
5. **Register the Python runtimes under it, width 1.** v41, then mimo, then xing4 (cheapest first: it is
   single-card, no TP, experts resident).
6. **Collapse the adapters** to one plus a per-runtime spec, and unify the prefix-cache interface behind
   the four implementations.
7. **Raise each runtime's width** as its own per-model work: xing4's value-batch axis, then per-row
   sampling and resumable prefill for mimo and v41.
8. **Merge the two HTTP front ends**, and extend `check_layering` to `engine/` — the last one matters
   because steps 4–7 move engine code, and `engine/` is the one directory the layering check does not
   cover today.

Items 1–3 are scheduled as Stage R1, 4–6 as R2, and 7–8 as R3 in the refactor project.

---

## 9. What this page does not establish

- No number measured on vLLM 0.30.0 or SGLang 0.5.20 is reproduced here; this machine cannot run
  either on these cards (SGLang has no sm_75 AOT artifacts and retired its CUDA 12 lane; vLLM 0.30
  does still build sm_75, which is worth knowing). The competitor numbers quoted are theirs, on their
  hardware, and are labelled as such.
- The claim that a CUDA IPC all-reduce would help here is a **hypothesis transferred from Ascend**, not
  a measurement on CUDA. It is stated with the arm design that would settle it.
- The claim that C++ graph capture is worth ~3.8× on Qwen is **transferred from Xing4 in PyTorch**,
  which is a different runtime with a different step composition. The local fact is that capture is
  worth 3.84× *there*, and that 22,155 dispatches a step is what it removes.
- Line-count comparisons between engines measure different things (vLLM counts a framework that hosts
  368 architectures; PocketLLM counts seven hand-written runtimes). They are used here only to compare
  *within* an engine — cost per added model, duplication between siblings — never to rank engines.
- vLLM #42770, #44219 and #45470 are **open RFCs**, not policy. They are cited as evidence of direction,
  not as statements of what vLLM does today.

## Evidence

Repository reads at `8162937`:

- `cpp_engine/CMakeLists.txt` (backend selection, `POCKET_CORE_SOURCES`, `POCKET_CUDA_ONLY_ENGINE_SOURCES`,
  `check_layering`), `cpp_engine/engine/backend_unimplemented_ascend.cpp`,
  `cpp_engine/include/batch_scheduler.hpp`, `engine/batch_scheduler.cpp`,
  `engine/qwen_engine.cpp`, `engine/deepseek_v4_engine.cpp`, `include/{cuda_ops,qwen_cuda_ops,qwen_ascend_ops}.hpp`
- `pocketllm/backends/{base,factory,cpp_backend,torch_backend,v41_backend,mimo_backend,xing4_backend}.py`,
  `pocketllm/api/{types,backend}.py`, `pocketllm/server/openai.py`, `pocketllm/supervisor.py`
- `src/models/prefix_cache.py`, `src/models/xing4_0/prefix_cache.py`, `src/runtime/prefix_snapshot.py`,
  `src/server/engine.py`, `src/runtime/pd_scheduler.py`, `src/models/xing4_0/graphs.py`,
  `src/models/xing4_0/decode_pos.py`
- Counts reproduced for this page: `grep -rn '<<<' cpp_engine` → 371 under `backends/`, 0 under
  `engine/` and `core/`; `grep -rn cudaGraph cpp_engine` → 0; distinct `*_cuda(` declarations → 86
  (`cuda_ops.hpp`) + 148 (`qwen_cuda_ops.hpp`), 45 `_ascend` symbols; `difflib` on the `_decode` body of
  `mimo_backend.py` vs `xing4_backend.py` → ratio 1.00.

Repository measurements cited: [native C++ concurrency validation](../performance/cpp_openai_concurrency_validation.md),
[Xing4 decode launch gap](../performance/xing4_0_decode_launch_gap.md),
[Xing4 decode graph buckets](../performance/xing4_0_decode_graph.md),
[Ascend single-request decode](../performance/ascend_single_request_tps.md),
[Ascend TP collective overlap](../performance/ascend_tp_collective_overlap.md),
[Ascend performance roadmap](ascend_performance_roadmap.md).

Upstream reads: vLLM `v0.30.0` tag and release notes, RFCs #42770, #44219, #45470, #51212, #45133,
#11162, #21481, #47187, #48277, #47361; SGLang `v0.5.20` tag, release notes and the platform-plugin and
attention-backend guides; llama.cpp `ggml-backend.h` and `HOWTO-add-model.md`; ONNX Runtime execution
provider docs; TensorRT-LLM DeepSeek-V4 blog.

## Related

- [Refactor project: one request lifecycle, one scheduler](https://github.com/users/lvyufeng/projects/7) —
  §8's decision as a tracked issue tree, tracked from
  [#432](https://github.com/lvyufeng/PocketLLM/issues/432)
- [PocketLLM vs vLLM vs SGLang](vllm_sglang_architecture_analysis.md) — the earlier comparison, against
  the local 0.21.0 fork
- [cpp_engine multi-backend refactor plan](cpp_engine_multi_backend_plan.md) — the layering plan item 6
  of §7 completes
- [Backend unification design](backend_unification_design.md) and
  [PocketLLM refactor analysis (2026-09)](pocketllm_refactor_analysis_2026_09.md) — the two earlier drafts
- [Feature roadmap for old hardware](pocketllm_roadmap_old_hardware.md) — the capability axis
- [New model support on the 2080 Ti](pocketllm_new_model_roadmap.md) — the checkpoint axis
- [Cross-request prefix caching on V4.1](v41_prefix_cache.md) and
  [on MiMo-V2.6-Flash](mimo_v2_6_flash_prefix_cache.md) — the shared store item 5 of §7 generalises
