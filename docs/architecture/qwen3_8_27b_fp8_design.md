# Qwen3.8-27B-FP8: design and measurements

This is the engineering record behind [the Qwen3.8-27B-FP8 model guide](../models/qwen3.8-27b-fp8.md):
the kernels the C++ runtime dispatches, the three speculative-decoding paths it carries, the
prefix-reuse protocol, the boundary case at the model's context limit, and every measurement behind
those choices. Read the model guide first if you want to *run* the model — this document is for
changing it.

Two conventions hold throughout. Prefill and decode are reported separately, and decode throughput is
comparable only between runs that generate the same number of tokens — the sweep below generates 128,
so any figure taken at four or 24 tokens is a different measurement. And a row labelled "one-shot
wall" includes prompt processing that a long-lived prefix-reusing engine does not repeat, which is why
several speculative rows carry a decode speedup above 1× and a wall speedup below it.


## Runtime status

**Validated native C++/CUDA TP4 text runtime.** PocketLLM detects the nested Qwen3.5 text configuration, maps rank-local Safetensors weights, converts BF16 scales/non-FP8 tensors to FP16 for Turing where required, and keeps local FP8 weights resident on each GPU.

The current integration supports text prompt/token-ID smoke, timed greedy generation, and native OpenAI-compatible text serving through `pocketllm_engine`. It does not execute the checkpoint's vision tower or accept image/video inputs.

## Model specification

The validated checkpoint reports:

| Field | Value |
| --- | ---: |
| HF architecture | `Qwen3_5ForConditionalGeneration` |
| Text model type | `qwen3_5_text` |
| Text layers | 64 |
| Gated DeltaNet layers | 48 |
| Full GQA layers | 16 |
| Hidden size | 5120 |
| Dense MLP intermediate | 17,408 |
| Vocabulary | 248,320 |
| Maximum positions | 262,144 |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Partial RoPE | 64 dimensions (factor 0.25) |
| Linear-attention key heads | 16 × 128 |
| Linear-attention value heads | 48 × 128 |
| Convolution kernel | 4 |
| Quantization | FP8 E4M3, dynamic activation scheme |
| Weight scale block | 128×128 |

The root config also contains a vision tower, but PocketLLM deliberately dispatches only the text model tensors.

## Implemented execution path

- Nested Qwen config detection and strict tensor/scale shape validation.
- TP4 rank-local embedding, head, attention, and dense MLP sharding.
- FP8 E4M3 weights stored as bytes with FP16 block scales on RTX 2080 Ti.
- Online FP8 unpacking in CUDA tiles/registers; no full FP16/FP32 weight expansion.
- Separate multi-row prefill and single-token decode projection kernels.
- 48-layer Gated DeltaNet sequence/recurrent kernels with persistent state and convolution tails.
- 16-layer GQA prefill and KV-cache decode with local K/V heads.
- FP16 activation storage with FP32 local accumulation/state where required; no prompt-length FP32 activation expansion.
- Chunked prefill (default 8,192 tokens, `--prefill-chunk-tokens`) that retains only recurrent state, convolution tails, and full-attention KV cache between chunks.
- Exact single-request prefix reuse: the position-indexed GQA KV cache and the DeltaNet recurrent state are retained across sequential `prefill()` calls. Appended prompts execute only their uncached suffix; diverging or compressed prompts restore a device-resident recurrent snapshot at the longest safe common prefix.
- FP16 KV cache by default, plus explicit opt-in FP8 E4M3 cache with per-token/KV-head FP16 scales over 64-channel blocks.
- Decode-only fused FP8 gate/up projection plus SwiGLU.
- Exact FP16 GQA kernels, on by default: tiled prefill and split-context fused decode with compact online-softmax partials. On SM75 the split/merge decode path is dispatched from context 4,096 onward, and the reference score/value decode below that, where it is faster. `POCKETLLM_QWEN_GQA_OPTIMIZED=0` restores the reference path.
- Opt-in FP16 sink-plus-sliding-window attention through `--qwen-attention-window N` and optional `--qwen-attention-sink-tokens N`. This changes full-attention semantics and is not part of exact parity or default performance claims; FP8 cache is intentionally rejected for this mode.
- TP4 NCCL reductions and global greedy top-1 selection.
- Opt-in native one-layer MTP loading and greedy speculative generation through `--qwen-mtp-tokens K`. The MTP layer reuses the target embedding/LM head, recursively proposes drafts, and verifies `[current_token, draft_1, ..., draft_K]` in one multi-row target forward. Partial rejection restores DeltaNet state/convolution tails and replays only the committed input prefix. MTP remains disabled by default.
- Opt-in external Qwen DSpark loading through `--qwen-dspark PATH`. The real five-layer BF16 drafter is replicated on every TP rank, consumes target post-layer taps `4,16,28,40,52`, proposes the checkpoint's fixed seven-token block, and verifies `[anchor,draft_1,...,draft_7]` in one eight-row target forward. Its position-indexed context K/V follows the target prefix cache across append, shorter-prefix, branch, and compressed-context restores.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4 (GPU 0–3), single request, real Qwen3.8-27B-FP8 checkpoint and
deterministic real-tokenizer prompts. One serial sweep of the engine defaults — complete 64 layers,
`prefill_chunk_tokens=8192`, FP16 KV cache, greedy sampling, 128 generated tokens — on master
`cfad866` (2026-09-15). Run record: `.tmp/qwen_fp8_page_20260915/`.

| Prompt | Prefill | Decode | KV data/rank | Peak activation workspace | Rank parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 115.91 tok/s (0.55 s) | 45.05 tok/s | 3.0 MiB | 7.9 MiB | PASS |
| 512 | 864.54 tok/s (0.59 s) | 43.22 tok/s | 10.0 MiB | 63.0 MiB | PASS |
| 4,096 | 1,729.05 tok/s (2.37 s) | 43.82 tok/s | 66.0 MiB | 504.2 MiB | PASS |
| 8,192 | 1,818.65 tok/s (4.50 s) | 43.99 tok/s | 130.0 MiB | 1,008.4 MiB | PASS |
| 32,768 | 1,673.79 tok/s (19.58 s) | 41.77 tok/s | 514.0 MiB | 1,008.4 MiB | PASS |
| 65,536 | 1,453.51 tok/s (45.09 s) | 39.11 tok/s | 1,026.0 MiB | 1,008.4 MiB | PASS |

Decode holds near 44 tok/s out to 8,192 tokens and reaches 39.11 tok/s at 65,536. Prefill peaks at
8,192 tokens and is still at 80% of that peak at 65,536.

Read the table with the two reporting rules from `../guides/benchmarking.md`:

- **Prefill and decode are reported separately**; neither is a wall-clock figure, and the prefill
  seconds above exclude model load and process startup.
- **Decode throughput is only comparable between runs that generate the same number of tokens.**
  These runs generate 128, so the timed decode window excludes the first token (produced by prefill)
  and is long enough to average out per-step overhead. Any measurement generating four or 24 tokens
  is not comparable to this table's decode column, including the earlier figures on this page.

Prefill is close to linear in prompt length above 4,096 tokens: the 4,096→32,768 segment runs at a
marginal 1,670 tok/s (a 600 µs/token slope) and the 32,768→65,536 segment at 1,285 tok/s, the
difference being the growing quadratic attention term. **The 64- and 512-token rows measure
short-prompt latency, not steady-state throughput**: both complete in 0.55–0.59 s because a fixed
per-process cost dominates at that size.

Per rank, 6.86 GiB of resident weights and 0.71 MiB of scales are fixed; only the KV data and the
activation workspace grow with context, which is why the memory curve is nearly flat. The activation
workspace is the peak capacity of the reusable chunk workspace and does not grow with prompt length
past one chunk.

Earlier revisions of this section published 416.48, 453.08 and 295.46 tok/s for a 512-token fixture
in three different places. All three predate the GQA tensor-core prefill and cuBLAS FP8 prefill
defaults, and the long-context tables used 512-token prefill chunks; this sweep supersedes them.

### Current memory-safe FP16-activation kernels

The current reference runtime now uses FP16-input, FP32-accumulation FP8 projection kernels without expanding prompt activations or weights. The prefill path uses a 128-token x 64-output N64 tile when alignment and batch size permit; decode uses vectorized single-row FP8 matvec, while the original scalar kernel remains the fallback. Two-row and four-row decode variants remain explicit experiments because their register pressure reduced end-to-end decode throughput. These kernels preserve the default exact full-attention semantics and FP16 KV cache.

These kernels are the default projection path. Their end-to-end result is the sweep in **Validated
performance** above, and the resident weight and scale bytes they report are `7,367,270,656` and
`742,400` per rank.

The direct FP16-activation FP8 projection gate covers aligned and padded strides, masked rows, tail K tiles, vectorized-versus-scalar decode dispatch, and the wide prefill tile. It reports decode max absolute error `1.459e-2` against the FP32 host reference, with vectorized-versus-scalar output difference `0`; the 4-row experimental path differs by at most `3.906e-3`. The focused FP8 online operator suite and full TP4 rank parity checks also pass.

### Native MTP speculative decoding

The checkpoint declares `mtp_num_hidden_layers=1` and ships the predictor in `mtp.safetensors`. PocketLLM maps this full-attention layer under TP4, uses the required `[normalized_embedding, normalized_hidden]` fusion order, consumes the target's final-normalized hidden state, and uses the MTP layer's normalized output as the recursive hidden. Target verification computes all candidate-row logits together and performs one batched TP global top-1 collective.

Enable it explicitly with:

```bash
--qwen-mtp-tokens 4
```

`--qwen-mtp` is equivalent to enabling the default `K=1`. MTP requires all 64 target layers; combining it with a nonzero partial `--smoke-layers` value is rejected. It is also context-safe: the prompt plus requested output count must fit in `max_context`; each speculative block is capped by the remaining output count, so its temporary verify suffix stays within that bound. The runtime reports `mtp_accept_rate`, proposed/correct draft counts, rollback/replay counts, and separate prefill/draft/verify/replay seconds.

This MTP implementation follows the standard Qwen3.5 shifted-hidden predictor path used by vLLM: prompt target rows are paired with their next-token inputs to prime an independent absolute-position MTP KV cache, after which the predictor advances recursively. SGLang's newer `frozen_kv_mtp` worker is a separate Gemma-oriented optimization that requires a model-specific mapping from assistant layers to target KV-owner layers; Qwen3.5 does not expose such a mapping, so target and MTP KV are not aliased.

The optimized target verifier reuses each FP8 weight row across all 2–8 candidate rows, fuses small-batch gate/up/SwiGLU, reuses transaction buffers, and batches exact GQA while retaining the reference score/softmax/value reduction order. An even faster split online-softmax verifier is available through `QWEN_GQA_VERIFY_SPLIT=1`, but stays experimental: its small numerical drift can change greedy output at near-tie logits even though direct attention error is below `8e-6`.

Real TP4, full 64-layer, 64-token generation results below cover several tokenizer-real 512-token prompts. Every TP rank agreed and each MTP sequence matched its serial plain run. The final adaptive policy starts at `K=1`, doubles K after full acceptance, and backs off after rejection:

| Prompt | Acceptance | Plain TPS | Adaptive K<=4 TPS | Decode speedup | Wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repeated natural language | 100.0% | 37.19 | 79.39 | **2.135x** | **1.331x** |
| Config/model text | 94.0% | 36.75 | 69.61 | **1.894x** | **1.273x** |
| Source code | 70.0% | 36.91 | 50.66 | 1.372x | 1.106x |
| README prose | 70.9% | 36.83 | 48.10 | 1.306x | 1.079x |
| Model documentation | 64.8% | 36.93 | 43.53 | 1.179x | 1.026x |

The optimized path therefore clears 1.5x only when draft acceptance is high (94% or better in the measured cases); it cannot honestly guarantee 1.5x for arbitrary prompts. Starting adaptive mode at K=1 protects mixed prompts better than starting at K=4, but 65–71% acceptance still yields only 1.18–1.37x decode speedup. A target top1-top2 logit-margin gate was also tested and rejected: thresholds 0.5 and 1.0 reduced performance on the difficult prompts, so no margin-gating code or CLI option is retained. The one-shot wall result includes independent-MTP prompt priming; persistent single-concurrency requests with exact prefix reuse remain the intended workload.

The parity-safe exact-GQA verifier now uses a warp-tiled value pass: one warp owns one candidate row, three query heads, and 32 value channels while preserving each output element's left-to-right FP32 accumulation order. Fresh 100%-acceptance measurements show the long-context gain increasing as plain decode becomes attention-bound:

| Prompt | Plain TPS | MTP K=4 TPS | Decode speedup | One-shot wall speedup | Parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4,096 | 30.10 | 50.17 | **1.667x** | not reported | PASS |
| 8,192 | 24.78 | 61.29 | **2.474x** | 0.961x | PASS |
| 32,768 | 11.30 | 36.26 | **3.208x** | 0.929x | PASS |

The 8K/32K one-shot wall numbers remain below 1x because each separately launched MTP process primes an additional full-prompt predictor KV cache; this is not repeated for an exact-prefix hit in the long-lived workload. An opt-in split-GQA experiment measured 2.38–4.57x at 512/4K/8K/32K, but it is not included in the parity-safe claim because a separate near-tie prompt exposed sequence drift.

Reproduce serial plain/K=1/K=2/K=4 A/B cases with real tokenizer IDs and automatic TP-rank/plain parity checks:

```bash
python scripts/bench_qwen_mtp.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,32768 \
  --mtp-tokens 1,2,4 \
  --max-new-tokens 32 \
  --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

For the recommended adaptive policy, pass one maximum K and `--adaptive`:

```bash
python scripts/bench_qwen_mtp.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,8192,32768 \
  --mtp-tokens 4 --adaptive \
  --max-new-tokens 64 --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

### External DSpark speculative decoding

The external checkpoint `RadixArk/Qwen3.8-27B-DSpark` (`epoch_2_step_4166`) is supported as an explicit opt-in:

```bash
--qwen-dspark /path/to/Qwen3.8-27B-DSpark
```

The directory must contain its `config.json` and single `model.safetensors`. The implementation validates all 62 expected BF16 tensors, materializes them as FP16 on SM75, and adds `2,623,214,594` resident weight bytes per rank. The five-layer draft backbone is replicated rather than tensor-parallel; only target embedding/head operations and the vocabulary-sharded Markov `w2` use TP collectives. Native MTP and external DSpark are mutually exclusive, and DSpark requires all 64 target layers.

The checkpoint fixes `block_size=7`, so each transaction proposes seven drafts and target-verifies eight rows: `[anchor,draft_1,...,draft_7]`. The target post-layer taps are `4,16,28,40,52`. Partial rejection restores DeltaNet recurrent state and convolution tails, crops logical target/DSpark K/V to the committed position, then replays only `[anchor,accepted drafts]`. A remaining output tail shorter than eight tokens uses ordinary exact decode rather than changing the checkpoint's block semantics.

The confidence head is evaluated for every draft and reported as `dspark_confidence_count/mean/min/max`, but it does not currently change the static width-8 schedule: neither the checkpoint nor its published static deployment supplies a validated confidence threshold. Speculative accounting retains the existing `mtp_*` field names for CLI compatibility (`mtp_accept_rate`, proposed/correct drafts, rollback/replay, and stage seconds).

Real TP4, full-model, FP16-target-KV A/B results below use the same tokenizer-real deterministic language fixture, 64 generated tokens, plain then DSpark serial execution, and exact DSpark-versus-plain plus all-rank token checks. The table's `draft match rate` is `correct_drafts / proposed_drafts` and excludes the target bonus token; it is not the model-card `spec_accept_length`, whose numerator includes one bonus token per verification step.

| Prompt | Draft match rate | Plain TPS | DSpark TPS | Decode speedup | Plain wall | DSpark wall | Wall speedup | Highest DSpark rank memory | Parity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 94.64% | 37.20 | 60.02 | **1.614x** | 2.944 s | 2.385 s | **1.235x** | 11.19 GB | PASS |
| 8,192 | 96.43% | 24.80 | 43.79 | **1.766x** | 31.505 s | 31.551 s | 0.999x | 11.61 GB | PASS |
| 32,768 | 96.43% | 11.31 | 21.70 | **1.919x** | 295.812 s | 295.773 s | 1.000x | 12.52 GB | PASS |

Long-context one-shot wall time is prefill-bound. DSpark's target-feature projector and five layers of context K/V projection make prefill slightly slower even though decode gets progressively faster. This is why the intended deployment remains a long-lived, single-request prefix-reusing engine rather than repeatedly paying cold prefill.

Acceptance is workload-sensitive. The earlier 17% figure came from a deliberately bare 16-token prompt, not the DSpark model-card benchmark protocol: it omitted the Qwen chat template, used greedy decoding, and counted only draft matches. Re-running the same wording with the real chat template still produced a difficult greedy case (15.34% draft match without thinking, 23.81% with thinking), so it is a valid stress case but not evidence that the published DSpark acceptance is 17%. The model card instead reports `spec_accept_length` including the bonus token, with a request-weighted mean of 3.39 accepted tokens per verification step over 1,164 sampled requests (macro-average 3.35) at temperature 0.6, top-k 20, top-p 0.95, thinking enabled, and 2,048 generated tokens. Our native path is currently greedy and should not be compared to those stochastic benchmark numbers as if they were the same metric. DSpark remains default-off pending benchmark-protocol-matched measurements.

For reference, the bare greedy stress prompt measured 31 correct drafts out of 182 proposals (`31/182 = 17.03%`), 17.42 tok/s DSpark decode, and exact plain-token parity. The chat-template, thinking-enabled rerun measured 35/147 (`23.81%`).

Reproduce the protocol-sensitive chat-template fixture with `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` before passing token IDs to the C++ runtime; do not use the raw user sentence as the benchmark prompt.

DSpark therefore remains default-off; enable it only after measuring representative traffic with the intended chat template, sampling policy, and generation length.

Reproduce the short/long serial A/B with:

```bash
/path/to/deepseek/bin/python scripts/bench_qwen_dspark.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dspark /path/to/Qwen3.8-27B-DSpark \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,8192,32768 \
  --max-new-tokens 64
```

The focused prefix/cold-parity suite covers exact repeat, monotonic append, shorter prefix, interior branch, and compressed context under both target KV dtypes. With the default early 256-token snapshot spacing, a shorter/branched prompt may reuse the deepest safe 256-token boundary and recompute the remainder rather than claiming the entire matched prefix:

```bash
for dtype in fp16 fp8; do
  /path/to/deepseek/bin/python scripts/bench_qwen_dspark_prefix_cache.py \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --dspark /path/to/Qwen3.8-27B-DSpark \
    --kv-cache-dtype "$dtype" \
    --max-context 1024 --max-new-tokens 16
done
```

### External DFlash2 speculative decoding

The external draft checkpoint is supported as an explicit opt-in:

```bash
--qwen-dflash2 /path/to/Qwen3.8-27B-DFlash2
```

The directory must contain its `config.json` and single `model.safetensors`. The runtime validates all 81 expected tensors and adds `1,450,191,360` sharded plus `3,848,808,960` replicated device bytes per rank. `--qwen-dspark` and `--qwen-dflash2` are mutually exclusive, and neither can be combined with native MTP.

The checkpoint fixes `block_size=8` with `target_layer_ids = [5,19,33,47,61]`, a five-layer sliding-attention backbone (`sliding_window=2048`), `selector_rank=256`, and `selector_top_k=16`. Each transaction drafts up to seven tokens and verifies eight target rows. Partial rejection restores DeltaNet recurrent state and convolution tails, crops logical K/V to the committed position, and replays only the accepted prefix.

Unlike DSpark, DFlash2's residual and MLP `down` outputs exceed the FP16 range: a pure FP16 residual overflows in the first layer and produces NaN. The working SM75 mixed precision keeps the residual, `down` output, and finish convolution in FP32 while `gate`/`up`/SwiGLU stay in cuBLAS FP16, converting back to FP16 after each layer norm for attention and dynamic projection. All DFlash2 RMSNorms are standard direct-gamma, not the target's `(1 + gamma)`.

Real TP4, full-model, FP16-target-KV A/B results with serial plain-then-DFlash2 execution and exact cross-mode plus all-rank token checks:

| Fixture | Plain wall | DFlash2 wall | Wall speedup | Decode speedup | Accept length | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic 512, 512 new | 14.510 s | 5.228 s | **2.776x** | **3.017x** | 8 / 8 | PASS |
| Synthetic 4,096, 512 new | 21.019 s | 9.307 s | 2.259x | 3.104x | 8 / 8 | PASS |
| Synthetic 8,192, 512 new | 30.539 s | 15.677 s | 1.951x | 3.400x | 8 / 8 | PASS |
| GSM8K, 8 prompts, 256 new | 57.595 s | 43.356 s | **1.328x** | 1.300x | 2.87 – 4.40 | PASS 8/8 |

Decode-phase speedup lands inside upstream's published 2.67–3.43x band on all three synthetic lengths. Upstream defines speedup as a per-token decoding latency ratio, so a full-request wall ratio is not the same metric: prefill is shared identically by both modes and caps the 8,192 case at 1.95x regardless of drafter quality.

GSM8K acceptance is far lower (0.55–0.71 per-draft rate) and per-prompt wall speedup tracks it directly, from 1.13x to 1.57x. Verify cost is roughly linear in block width because the 48 gated-delta layers recur sequentially over rows, so a full-width block pays for and discards the rejected tail. `POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1` tracks an EWMA of accepted count and verifies `ewma + 1.5` rows with a floor of two, which keeps the fully-accepted synthetic cases at width 8 while lifting the low-acceptance GSM8K case. A fixed width cannot serve both: at width 7 GSM8K regresses to 0.86x, and at width 2 the synthetic gains are discarded.

Four flags are opt-in and all four were enabled for the results above:

| Flag | Effect |
| --- | --- |
| `POCKETLLM_DFLASH2_CUBLAS_FP32=1` | Routes the target LM head through cuBLAS FP32; the head is 55% of draft cost and this makes it ~10x faster |
| `POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1` | EWMA-driven verify width, as above |
| `POCKETLLM_DFLASH2_SPLIT_TOPK=1` | Partitions each row's local top-16 shard and merges with the identical comparator |
| `POCKETLLM_QWEN_GQA_OPTIMIZED=1` | Selects the tiled GQA prefill kernel; long prefill is 64% full-attention, not FP8 GEMM |

`POCKETLLM_DFLASH2_VITERBI_SELECTOR=1` computes an exact MAP over the selector chain instead of greedy argmax. It scores higher but accepts worse (3.02 to 2.95 accept length, 0.555 to 0.279 rate), so draft search is not an acceptance lever.

Reproduce the synthetic serial A/B with:

```bash
POCKETLLM_DFLASH2_CUBLAS_FP32=1 POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1 \
POCKETLLM_DFLASH2_SPLIT_TOPK=1 POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
/path/to/deepseek/bin/python scripts/bench_qwen_dflash2.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dflash2 /path/to/Qwen3.8-27B-DFlash2 \
  --lengths 512,4096,8192 \
  --max-new-tokens 512 --prefill-chunk-tokens 4096 --snapshot-interval 0
```

And the dataset-shaped single-request workload, which matches upstream's aggregate-completion protocol rather than a fixed-token microbenchmark:

```bash
POCKETLLM_DFLASH2_CUBLAS_FP32=1 POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1 \
POCKETLLM_DFLASH2_SPLIT_TOPK=1 POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
/path/to/deepseek/bin/python scripts/bench_qwen_dflash2_upstream.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dflash2 /path/to/Qwen3.8-27B-DFlash2 \
  --dataset gsm8k --num-prompts 8 --max-new-tokens 256
```

### Exact prefix reuse

`QwenEngineOptions::prefix_cache` is enabled by default for a long-lived engine. The full-attention KV cache is already indexed by absolute position, while the 48 DeltaNet layers carry a small recurrent state (`state` plus convolution tail). The engine retains both across sequential prefill requests and reports `prefix_reused_tokens`, `prefix_computed_tokens`, `prefix_matched_tokens`, and `prefix_resume_source` in the persistent stdin result.

The runtime stores device-resident recurrent snapshots at dense 256-token boundaries through the first 4K, then at the configured 4K interval through the 262K limit. This keeps a compressed or diverging request exact while limiting recomputation to the suffix after the selected snapshot. The default 82 snapshots use about 3.0 GiB per rank on the full 48-layer model; this is additional KV/state working memory and is included in the reported GPU memory. `--qwen-no-prefix-cache`, `--qwen-snapshot-interval N`, and `--qwen-max-snapshots N` provide explicit A/B controls.

A long-lived TP4 stdin worker can be started with `--qwen-persistent-stdin --max-context N`; rank 0 reads lines of the form `<max_new_tokens> token0 token1 ...`, and rank 1..3 receive the same requests over the Qwen command socket. This mode is intended for single-concurrency clients and preserves the cache between lines. Each request returns exact greedy tokens and prefix accounting. The ordinary one-shot CLI creates a fresh engine, so it cannot reuse a cache across processes and therefore keeps snapshots disabled; the reported one-shot long-context TPS and memory are unaffected by this feature. `--qwen-no-prefix-cache` fully disables snapshots, prompt history, and cached results even in persistent mode.

A real TP4 continuous-request test on the Qwen3.8-27B-FP8 checkpoint produced:

| Request | Prompt | Reused | Computed | Resume | Request prefill TPS |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 512 | 0 | 512 | empty | 409.9 |
| 2 | 1,028 | 515 | 513 | live | 642.8 |
| 3 | 1,544 | 1,031 | 513 | live | 908.9 |
| 4 | 768 (256 common prefix + compressed suffix) | 256 | 512 | snapshot | 627.5 |

The cache-on and cold A/B runs generated identical tokens on all four TP ranks. In the cold run, requests 2/3/4 computed 1,028/1,544/768 tokens respectively, and every cold request reported `prefix_snapshots=0` with `prefix_snapshot_bytes=0`. Prefix reuse is exact: it does not claim that newly compressed content is cached; only the unchanged token prefix is reused.

Native MTP was also exercised through this same long-lived protocol with adaptive `K<=4`, 64 generated tokens, two appends, and an interior compression branch. A separate serial plain run used the identical four requests. All generated sequences matched, every request had TP-rank parity, and both modes reported identical prefix accounting:

| Request | Prompt | Reused | Computed | Resume | MTP acceptance | Plain decode TPS | MTP decode TPS | Decode speedup | Plain wall | MTP wall | Wall speedup |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0 | 512 | empty | 100.0% | 37.13 | 79.67 | **2.146x** | 2.959 s | 2.219 s | **1.334x** |
| 2 | 1,088 | 575 | 513 | live | 77.8% | 35.75 | 50.42 | 1.410x | 3.426 s | 3.120 s | 1.098x |
| 3 | 1,664 | 1,151 | 513 | live | 70.2% | 34.55 | 48.65 | 1.408x | 3.484 s | 3.169 s | 1.099x |
| 4 | 768 (256 common prefix + compressed suffix) | 256 | 512 | snapshot | 63.2% | 36.30 | 42.45 | 1.169x | 2.955 s | 2.897 s | 1.020x |

This validates the intended single-concurrency cache behavior and shows a wall win on all four requests, but it also confirms that the 1.5x requirement remains acceptance-dependent: only the 100%-acceptance request clears 1.5x decode and wall throughput. Appends reuse the target recurrent/KV state and the MTP shifted boundary; compression restores a target-hidden snapshot and rewrites the MTP boundary before priming the new suffix. The persistent harness reports wall, prefill, and decode timing separately; prefill includes MTP predictor priming when enabled.

The same harness was then run from a 32,768-token cold prompt with 512-token appends and a 4,096-token compression boundary. The run predates the current kernel defaults, so its absolute prefill figures are superseded by **Validated performance**; request 1 is that run's own cold baseline, so the reuse comparison holds within the table:

| Request | Prompt | Reused | Computed | Resume | Request prefill TPS | Snapshot bytes/rank |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 32,768 | 0 | 32,768 | empty | 113.7 | 885,178,368 |
| 2 | 33,284 | 32,771 | 513 | live | 4,083.9 | 923,664,384 |
| 3 | 33,800 | 33,287 | 513 | live | 4,105.8 | 962,150,400 |
| 4 | 4,608 (4,096 common prefix + compressed suffix) | 4,096 | 512 | snapshot | 2,361.3 | 654,262,272 |

Request 1 matches the cold 32K prefill baseline. The two appends each execute only the 513 uncached tokens, and the compressed request recomputes only its 512-token suffix after restoring the 4,096-token snapshot. Snapshot memory shrinks when a shorter prompt invalidates later rollback points.

### KV cache dtype and the 262,144-token boundary

FP16 KV cache is the default and the precision/performance baseline. FP8 E4M3 cache is an explicit
opt-in over 64-channel blocks that halves KV data per rank — 2,048 MiB instead of 4,096 MiB at the
262,140-token boundary — and is not the faster configuration at any length. The quantized-cache
comparison has its own document: [`../performance/qwen_kv_cache_65k_tg512.md`](../performance/qwen_kv_cache_65k_tg512.md)
measures FP16, FP8, TurboQuant K8V4 and INT8 per-token-head at 65,536 tokens with 512 generated
tokens and finds that dequant-once gives the quantized formats **prefill parity** with FP16 (within
0.3%) while decode falls to 0.124x (FP8), 0.370x (TurboQuant K8V4) and 0.052x (INT8 per-token-head).
Earlier revisions of this page said FP8 cache reduced prefill throughput as well; that measurement
predates dequant-once and is corrected here. The two cache dtypes also agree token for token: the
131,072-token FP16 and FP8 runs generated the same 128 tokens. FP16 greedy output is itself stable
across builds — ten runs of the 32,768-token case spanning five revisions and six runs of the
65,536-token case all produced the same sequence.

Chunked prefill is what makes the boundary reachable. It removes the previous prompt-length FP32
activation allocation, so the longest prompt the model accepts — 262,140 tokens, four positions short
of `max_context=262144` — completes without OOM inside the 22 GiB/rank budget: 787.32 tok/s prefill
over 332.95 s, 4,096 MiB of KV data and no scale bytes per rank, a 15.24 GiB highest-rank
`nvidia-smi` peak, TP-rank parity, and `[321, 5979, 13914, 13]` as the four generated tokens. Of that
peak, 11.85 GiB is the engine's own accounting and the remaining 3.4 GiB is the same fixed
per-process overhead seen at every other prompt length.

The boundary case carries no decode column, and cannot: `max_context` has to cover the prompt plus
the generated positions, and the model is limited to 262,144 of them, so 262,140 prompt tokens leave
exactly four. `--max-new-tokens 128` is refused at this length — all four ranks report `Qwen max
context exceeds model configuration`, with either cache dtype — so read the row as a
prefill-and-memory result only. An earlier revision of this page published 1.92 tok/s (FP16) and
0.64 tok/s (FP8) decode for it; those figures came from generations too short to measure decode and
are retracted. Decode is measured at 64, 512, 4,096, 8,192, 32,768 and 65,536 tokens in
**Validated performance**.

### Decode Context Parallelism feasibility on four GPUs

**Rejected for the current four-GPU topology.** DCP can directly shard only the 16 full-GQA layers. The 48 Gated DeltaNet layers retain complete recurrent state on every replica and do not benefit from ordinary KV position sharding.

Keeping four GPUs constrains the proposed topology to TP2xDCP2. For a context length `C`, the per-GPU full-attention decode work is unchanged: TP4 performs `6` local Q heads over `C` positions, while TP2xDCP2 performs `12` local Q heads over `C/2` positions. Both equal `6C` head-position evaluations per device; the DCP topology then adds two DCP all-reduces per full-attention layer and doubles the local TP2 weights.

This was tested with a deliberately favorable upper bound: plain TP2 at half the TP4 context length, without the two DCP collectives or cache compaction. It was already slower and used substantially more memory. Both columns come from the same run, so the comparison stands, but the TP4 decode column was measured with 512-token prefill chunks on the kernels of the time and is superseded by **Validated performance**:

| TP4 context / result | TP2 half-context, no DCP communication | Result |
| --- | --- | --- |
| 512 / 35.84 decode tok/s | 256 / 22.56 decode tok/s | 37% slower upper bound |
| 4,096 / 29.43 decode tok/s | 2,048 / 20.87 decode tok/s | 29% slower upper bound |
| 8,192 / 24.24 decode tok/s | 4,096 / 19.07 decode tok/s | 21% slower upper bound |
| 32,768 / 11.57 decode tok/s | 16,384 / OOM before prefill | infeasible |

TP2 local resident weights measured 13.72 GiB per rank, versus 6.86 GiB under TP4. Therefore an actual TP2xDCP2 implementation would be slower than these already-negative upper bounds and would introduce numerical/communicator complexity without reducing per-device attention work. The default TP4 path remains unchanged; no DCP code is enabled.

A useful context-parallel experiment requires at least eight ranks/GPUs for TP4xDCP2, which preserves the TP4 weight shard while halving per-device full-attention context. Even there, it would apply only to the 16 full-GQA layers and must beat the added two DCP collectives per such layer.

Per rank, the engine reported:

```text
resident_weight_bytes=7367270656
resident_scale_bytes=742400
gpu_memory_total_bytes=23068868608
```

The stable pre-optimization decode baseline was approximately 22.4 tok/s. Rank-local full-attention K/V projection, grouped GQA value aggregation, and fused decode SwiGLU raised the measured result into the 35–37 tok/s range while preserving the separate prefill path.

## Correctness and precision

- All four TP ranks generated identical token sequences on both the 64-token and 512-token fixtures.
- GQA decode matched the CPU reference with worst absolute error `1.192e-7`.
- Fused FP8 SwiGLU matched the separate projection path with `max_abs=0` and `max_rel=0` in its test fixture.
- Qwen RMSNorm, gated RMSNorm, L2 normalization, online FP8 matvec/matmul, DeltaNet, convolution tail, GQA, and TP weight-sharding tests pass.
- Parallel GQA softmax changes reduction association. CPU-reference error remains near `1e-7`, and real generated token sequences were unchanged in the validated runs.
- The existing DeepSeek FP8 matvec/matmul and minimum-layer smoke tests were also run to protect the older path.

## Reproduction

Build the C++ engine, then start four ranks with one shared NCCL ID file:

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --generate-token 123 --max-new-tokens 24 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

The numeric value passed to `--generate-token` is ignored once `--prompt` supplies the prompt IDs; it currently activates the generation mode in the compatibility CLI parser. Rank 0 prints the timed result and all ranks print their local runtime/accounting lines. The CLI defaults to one smoke layer; use `--smoke-layers 0` for a complete 64-layer performance claim.

For reproducible serial long-context TP4 measurements:

```bash
python scripts/bench_qwen_long_context.py \
  --binary cpp_engine/build/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 64,512,4096,8192,32768,65536 \
  --max-new-tokens 128 \
  --prefill-chunk-tokens 8192 \
  --kv-cache-dtype fp16 \
  --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

This is the command behind **Validated performance**, including its generated-token budget and prefill chunk size; changing either makes the output incomparable to that table. The harness persists one log per rank, records rank-local timing and memory fields, checks greedy-token parity across TP ranks, and writes `results.json` after every successful context length. FP16-versus-FP8 cache parity is a separate comparison of the generated sequences from two serial runs.

The exact FP16 GQA path is on by default and is what those measurements exercise: it keeps full attention, uses a tiled prefill kernel, and uses compact split-context fused decode partials from context 4,096 onward on SM75 while shorter contexts retain the reference score/value decode path because it is faster there. On an earlier build, when the path was still opt-in behind `POCKETLLM_QWEN_GQA_OPTIMIZED=1`, it improved a 24-generated-token 32,768-token run from 97.75 to 159.52 tok/s prefill and from 10.66 to 17.36 tok/s decode, with token parity at every length. Those absolute values are superseded by **Validated performance**; the gate itself is unchanged: the direct CUDA test covers causal offsets through 333 tokens, head dimensions 64/256, contexts 4,096/8,192/32,768, and a 262,144-token compact-partial boundary check.

Sparse experiments require an explicit `--qwen-attention-window N` and may add `--qwen-attention-sink-tokens N`; `N=0` is exact full attention. The sparse kernel attends to the leading sink prefix plus the newest window positions without changing KV-cache storage. This is an experimental semantic change, not an exact full-attention optimization claim. Window values that cover the complete context are directly checked against exact output; long-context quality and throughput are not reported here until measured on clean GPUs.

Audit only the rank-local weight mapping:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --tp-rank 0 \
  --qwen-audit
```

## Known limitations

- Text-only: no image/video preprocessing or vision-tower execution.
- Text-only serving: the native OpenAI-compatible server is validated for Qwen text requests, while the checkpoint's vision tower and multimodal request formats are not implemented.
- Stochastic sampling is available through `--temperature`, `--top-p`, and `--top-k`. The default remains greedy (temperature 0) so existing benchmarks stay reproducible. Per-request sampling overrides are exposed in the batched API and the OpenAI server.
- The model limit is 262,144 positions; with four generated tokens, the longest valid benchmark prompt is 262,140 tokens. That boundary is a prefill-and-memory result, not a throughput one — the position limit leaves no room for a decode measurement there. FP8 KV cache halves the boundary's KV footprint per rank by construction but is slower everywhere it has been measured; see [`../performance/qwen_kv_cache_65k_tg512.md`](../performance/qwen_kv_cache_65k_tg512.md).
- CUDA Graph and a decode megakernel remain future work; neither is included in the reported TPS.
- Native MTP is opt-in. Parity-safe high-acceptance cases accelerate decode by 1.67x at 4K, 2.47x at 8K, and 3.21x at 32K; 65–71% acceptance gives only 1.18–1.37x on 512-token varied prompts. Persistent exact-prefix workloads are the intended use case; `--qwen-mtp-adaptive` starts at K=1 and limits but does not eliminate low-acceptance overhead.
- External Qwen DSpark is opt-in and always uses its fixed seven-draft/eight-row transaction. It accelerates high-draft-match decode by 1.61–1.92x in measured 512/8K/32K cases, while a bare greedy stress prompt achieved only 31/182 draft matches and regressed to about 17.4 tok/s. This draft-match ratio excludes bonus tokens and is not comparable to the model card's bonus-inclusive `spec_accept_length=3.39` sampled-workload mean. Confidence is telemetry only; no unvalidated threshold is used to gate transactions.
- Split exact-GQA verification is experimental and enabled only with `QWEN_GQA_VERIFY_SPLIT=1`; direct numerical checks pass, but near-tie greedy output can drift. General long-prefill/decode optimized GQA and sparse attention remain separate opt-in paths; sparse attention changes model semantics.

## Evidence and related notes

- `cpp_engine/include/qwen_config.hpp`
- `cpp_engine/core/qwen_config.cpp`
- `cpp_engine/engine/qwen_weights.cpp`
- `cpp_engine/engine/qwen_engine.cpp`
- `cpp_engine/backends/cuda/kernels/qwen_fp8_ops.cu`
- `cpp_engine/backends/cuda/kernels/qwen_half_ops.cu`
- `cpp_engine/backends/cuda/kernels/qwen_attention_ops.cu`
- `cpp_engine/tests/test_qwen_config.cpp`
- `cpp_engine/tests/test_qwen_fp8_online.cpp`
- `cpp_engine/tests/test_qwen_gqa_attention.cpp`
- `cpp_engine/tests/test_qwen_half_ops.cpp`
- `cpp_engine/tests/test_qwen_weights.cpp`
- `cpp_engine/tests/test_qwen_engine.cpp`
- `cpp_engine/include/qwen_dspark.hpp`
- `cpp_engine/engine/qwen_dspark.cpp`
- `cpp_engine/backends/cuda/kernels/qwen_dspark_ops.cu`
- `cpp_engine/tests/test_qwen_dspark.cpp`
- `cpp_engine/tests/test_qwen_dspark_ops.cpp`
- `scripts/bench_qwen_mtp.py`
- `scripts/bench_qwen_dspark.py`
- `scripts/bench_qwen_dspark_prefix_cache.py`
- [Qwen3.8-27B-NVFP4](../models/qwen3.8-27b-nvfp4.md) for the mixed NVFP4/FP8 checkpoint on the same text runtime
- [Benchmark reporting rules](../guides/benchmarking.md)
