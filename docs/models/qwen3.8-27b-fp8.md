# Qwen3.8-27B-FP8

A 27B text model with a hybrid attention stack — 48 Gated DeltaNet (linear-attention) layers and 16
full GQA layers — shipped as FP8 E4M3 Safetensors. PocketLLM runs it on the native C++/CUDA engine
over four cards, with the weights sharded rank-locally and kept resident on each GPU, and serves it
through the OpenAI-compatible API.

- **Backend**: `--backend cpp` (native C++/CUDA, OpenAI-compatible server)
- **Parallelism**: TP4, one process a card
- **Context**: up to 262,144 positions
- **Validated on**: 4×RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink

## Overview

| Field | Value |
| --- | ---: |
| Text layers | 64 (48 Gated DeltaNet + 16 full GQA) |
| Hidden size | 5,120 |
| Dense MLP intermediate | 17,408 |
| Vocabulary | 248,320 |
| Query heads / KV heads / head dim | 24 / 4 / 256 |
| Partial RoPE | 64 dimensions (factor 0.25) |
| Quantization | FP8 E4M3, 128×128 block scales, dynamic activations |

The checkpoint's root config also carries a vision tower. PocketLLM deliberately dispatches only the
text-model tensors, so this is a text-only runtime: no image or video preprocessing, and no
multimodal request formats.

Three things shape the runtime:

- **The FP8 weights are never expanded.** They stay bytes with FP16 block scales and are unpacked
  online inside the CUDA tiles and registers, which is what keeps a 27B model inside 22 GiB a card at
  6.86 GiB of resident weights and 0.71 MiB of scales a rank.
- **Prefill and decode are separate kernels.** Multi-row prefill uses a tiled FP8 path with a
  128-token × 64-output tile and cuBLAS where it is faster; decode uses a single-row FP8 matvec with
  fused gate/up/SwiGLU.
- **State is retained across requests.** The full-attention KV cache is position-indexed and the 48
  DeltaNet layers carry a small recurrent state; a long-lived engine keeps both, so an appended
  prompt executes only its uncached suffix and a diverging one resumes from a device-resident
  snapshot at the longest safe common prefix.

## Run it

### Serve it

```bash
pocketllm serve \
  --model /path/to/Qwen3.8-27B-FP8 \
  --backend cpp \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pocketllm", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The server covers health and model discovery, streaming and non-streaming chat and completions,
per-token log probabilities, stop-sequence truncation, request-field refusals and concurrent
scheduler admission.

### Serving options

| Option | Default | What it does |
| --- | ---: | --- |
| `--kv-cache-dtype` | `auto` (FP16) | FP16 is the default and the precision baseline. `fp8` halves KV data per rank but is not the faster configuration at any measured length. |
| `--prefill-chunk-tokens` | engine default | Tokens a prefill chunk takes. Chunked prefill is what makes the 262,140-token boundary reachable inside 22 GiB a rank. |
| `--max-model-len` | engine default | The context the engine reserves. It has to cover the prompt plus the generated positions. |
| `--enable-prefix-caching` | on | Exact cross-request prefix reuse, including restores from a device-resident snapshot. `--no-enable-prefix-caching` turns it off. |
| `--backend-option enable_batching=true` | off | Admits concurrent requests through the native scheduler. The width comes from `--backend-option max_batch_size` (8 when batching is on). |
| `--speculative-method` | off | `mtp` for the native one-layer predictor, `dspark` or `dflash2` for an external drafter — the latter two need `--backend-option dspark_checkpoint=PATH` / `dflash2_checkpoint=PATH`, and the three are mutually exclusive. |
| `--speculative-tokens` | 1 | Drafts per speculative step. |
| `--attention-window` / `--attention-sink-tokens` | `0` (exact) | Sink-plus-sliding-window attention. **Changes full-attention semantics**; not part of the exact-parity claim, and FP8 cache is rejected for it. |

Sampling is per request rather than per launch: `temperature`, `top_p` and `top_k` are OpenAI request
fields, and a `temperature` of 0 — the default — is greedy.

### Without a server

The standalone `pocketllm_engine` binary carries the flags the server does not surface, including
`--qwen-mtp-tokens K`, `--qwen-dspark PATH`, `--qwen-dflash2 PATH`, `--qwen-persistent-stdin`,
`--qwen-snapshot-interval N` and `--qwen-no-prefix-cache`.

```bash
# four ranks, one shared NCCL id file, one timed generation
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --max-new-tokens 24 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait

# the rank-local weight mapping, without running the model
build/cpp_engine/pocketllm_engine --ckpt /path/to/Qwen3.8-27B-FP8 --tp-world 4 --tp-rank 0 --qwen-audit
```

`--smoke-layers 0` means all 64 layers; the CLI defaults to one, and a one-layer run is not a
performance claim.

## What is supported

| Capability | State |
| --- | --- |
| OpenAI-compatible serving (chat, completions, streaming, logprobs, stop sequences, batching) | Supported |
| TP4 text generation, complete 64 layers | Supported |
| FP8 E4M3 weights resident on each GPU, unpacked online | Supported |
| Exact full-attention GQA kernels (tiled prefill, split-context fused decode) | Supported, on by default |
| Chunked prefill and a 262,144-position context | Supported |
| Exact cross-request prefix reuse, including compressions and branches | Supported |
| Sampling (per-request `temperature`, `top_p`, `top_k`) | Supported; greedy by default |
| Native MTP speculative decoding | Supported, opt-in |
| External DSpark / DFlash2 drafters | Supported, opt-in, mutually exclusive with each other and with MTP |
| Sink-plus-sliding-window attention | Experimental opt-in, changes semantics |
| FP8 KV cache | Supported, opt-in — smaller, never faster |
| Vision tower, image and video inputs | **Not implemented** |
| Decode Context Parallelism (TP2×DCP2) | **Rejected** — measured slower than TP4 at half the context, before any DCP communication |

## Performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4, one request, the real checkpoint and real tokenizer prompts.
One serial sweep of the engine defaults — 64 layers, `prefill_chunk_tokens=8192`, FP16 KV cache,
greedy, **128 generated tokens** — on master `cfad866`.

| Prompt | Prefill | Decode | KV data/rank | Peak workspace |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 115.91 tok/s | 45.05 tok/s | 3.0 MiB | 7.9 MiB |
| 512 | 864.54 tok/s | 43.22 tok/s | 10.0 MiB | 63.0 MiB |
| 4,096 | 1,729.05 tok/s | 43.82 tok/s | 66.0 MiB | 504.2 MiB |
| 8,192 | **1,818.65 tok/s** | **43.99 tok/s** | 130.0 MiB | 1,008.4 MiB |
| 32,768 | 1,673.79 tok/s | 41.77 tok/s | 514.0 MiB | 1,008.4 MiB |
| 65,536 | 1,453.51 tok/s | 39.11 tok/s | 1,026.0 MiB | 1,008.4 MiB |

Decode holds near 44 tok/s out to 8,192 tokens and is still at 39.11 tok/s at 65,536. Prefill peaks at
8,192 and is at 80% of that peak at 65,536; it is close to linear in prompt length above 4,096, the
gap being the growing quadratic attention term.

Three things to read with the table:

- **The 64- and 512-token rows measure short-prompt latency, not throughput.** Both complete in
  0.55–0.59 s because a fixed per-process cost dominates at that size.
- **Decode throughput is only comparable between runs that generate the same number of tokens.** This
  sweep generates 128, so the timed window excludes the first token and is long enough to average out
  per-step overhead.
- **The memory curve is nearly flat.** 6.86 GiB of weights and 0.71 MiB of scales are fixed per rank;
  only the KV data and the reusable chunk workspace grow with context, and the workspace stops growing
  after one chunk.

At the model's limit, a 262,140-token prompt — four positions short of `max_context` — prefills at
**787.32 tok/s in 332.95 s**, with 4,096 MiB of KV data a rank and a 15.24 GiB highest-rank
`nvidia-smi` peak. That row has no decode column and cannot have one: the position limit leaves
exactly four tokens, and a 128-token request there is refused by all four ranks.

### Optional speculative decoding

All three paths are default-off, because their gains track draft acceptance rather than the model.
Every case below was checked token-for-token against its plain serial run, on all four ranks.

- **Native MTP**, parity-safe, at 100% acceptance: **1.67× decode at 4K, 2.47× at 8K, 3.21× at 32K**.
  At 65–71% acceptance on varied 512-token prompts it is only 1.18–1.37×, so the intended workload is
  a long-lived, single-concurrency, prefix-reusing request stream rather than one-shot cold prompts.
- **DSpark** (five layers, seven drafts an eight-row verify): **1.61× / 1.77× / 1.92× decode** at
  512 / 8K / 32K with 94.6–96.4% draft match. On a deliberately bare greedy prompt it dropped to
  31/182 draft matches, which is why it stays off pending measurements on representative traffic.
- **DFlash2** (five layers, seven drafts an eight-row verify, adaptive width): **2.78× full-request
  and 3.02× decode** on a 512-token fixture, 1.95× at 8,192, and 1.33× aggregate over eight GSM8K
  prompts, where acceptance falls to 0.55–0.71 per draft. Its residual and `down` outputs exceed the
  FP16 range, so the working precision keeps those in FP32.

## Hardware and memory

| | |
| --- | --- |
| Cards | 4, one process each, TP4. Nothing here needs NVLink. |
| Resident weights | **6.86 GiB a rank**, plus 0.71 MiB of FP8 block scales |
| KV cache | 4,096 MiB a rank at the 262,140-token boundary with the default FP16 dtype |
| Prefix snapshots | The default 82 snapshots use about 3.0 GiB a rank on the full model |
| Context | 262,144 positions; with four generated tokens the longest benchmark prompt is 262,140 |

## Known limitations

- **Text only.** The vision tower is not executed and multimodal request formats are not implemented.
- **The position limit leaves no room for a decode measurement at the boundary.** 262,140 prompt
  tokens plus the model's 262,144 positions leaves four; a longer generated-token request is refused.
- **No CUDA Graph and no decode megakernel.** Neither is included in the TPS figures above.
- **Speculative decoding is acceptance-dependent.** Nothing here guarantees a speedup on arbitrary
  prompts; DSpark and DFlash2 in particular were measured on fixtures whose acceptance is high, and
  upstream's published 2.67–3.43× for DFlash2 is a per-token decode-latency ratio, not a full-request
  one — prefill is shared by both modes and caps the wall gain.
- **Split exact-GQA verification is experimental** (`QWEN_GQA_VERIFY_SPLIT=1`): direct numerical
  checks pass, but near-tie greedy output can drift. The default verifier is the parity-safe one.
- **Sparse attention changes model semantics** and is not covered by the exact-parity claim.
- **FP8 KV cache is smaller, not faster.** It halves KV bytes at the boundary but decode falls to
  0.124× of FP16 once the cache is dequantized once.
- **Decode Context Parallelism is not implemented and is not planned for four GPUs.** It would need
  at least eight ranks to keep the TP4 weight shard while halving the per-device context.

## Where the detail is

- [Design and measurements](../architecture/qwen3_8_27b_fp8_design.md) — the kernels, the three
  speculative paths, the prefix-reuse protocol, the DCP feasibility study, and the commands each
  number comes from.
- [Qwen KV cache at 65K](../performance/qwen_kv_cache_65k_tg512.md) — FP16 versus FP8, TurboQuant
  K8V4 and INT8 per-token-head, all at 512 generated tokens.
- [Qwen drafter acceptance](../performance/qwen_drafter_acceptance.md) — the acceptance study behind
  the default-off decision.
- [Benchmarking rules](../guides/benchmarking.md) — required before any Qwen number is quoted.
- The support matrix in [models/README.md](README.md).
