# MiniMax-M2.7

A 62-layer MoE text model with GQA and 256 routed experts, shipped as a `UD-IQ1_M` sharded GGUF
bundle. PocketLLM validates the bundle's schema, keeps each rank's routed-expert partition on device,
and runs raw-block CUDA greedy generation at TP4.

- **Backend**: the shared GGUF CLI (`src.cli.generate_gguf`); no OpenAI-compatible adapter
- **Parallelism**: TP4 expert parallelism
- **Context**: 196,608 tokens in the config — a full-length request does not fit the 4×22 GiB baseline
  with an FP16 KV cache
- **Validated on**: 4×RTX 2080 Ti 22 GiB, real `UD-IQ1_M` checkpoint

## Overview

| Field | Value |
| --- | ---: |
| Architecture | `minimax-m2` |
| Layers | 62 |
| Hidden size | 3,072 |
| Query heads / KV heads / head dim | 48 / 8 / 128 |
| Routed experts / active | 256 / top-8 |
| Expert intermediate size | 1,536 |
| Vocabulary | 200,064 |
| GGUF tensors | 809 |

The quantization is per-tensor: `iq2_xxs` for all three routed-expert matrices, `q5_k` for the
attention projections, `q4_k` for the embedding and output, and floating-point router, norm and bias
tensors.

Two dispatches matter, because prefill and decode take different paths:

- **Prefill** uses Q4_K/Q5_K INT8 MMA kernels, based on the vendored llama.cpp MMQ machinery. This is
  the change that moved full-model prefill from about 49.7 to about 105 tok/s.
- **Decode** stays on the IQ2_XXS DP4A grouped MoE paths for w1/w3 and w2, with a fused CUDA RMSNorm
  and a fused half-split RoPE. The MMA hook only runs for more than one row, so decode never sees it.

## Run it

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_gguf \
  --gguf-path /path/to/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --seed-file /path/to/prompt_tokens.bin \
  --max-new-tokens 32 \
  --prewarm
```

The shared CLI is token-ID oriented; MiniMax tokenizer and chat-framing helpers exist and are tested,
but the generation entrypoint takes prompt token IDs rather than a text prompt.

Inspect and validate the bundle:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --architecture auto \
  --spec-summary --validate-spec \
  --capability-report --placement-report
```

## What is supported

| Capability | State |
| --- | --- |
| Bundle schema validation and rank-local expert ranges | Supported |
| TP4 expert parallelism with NCCL reduction | Supported |
| IQ2_XXS DP4A grouped MoE (w1/w3 and w2) | Supported |
| Q4_K/Q5_K INT8 MMA prefill | Supported, on by default |
| Separate decode path (MMA hook only for `rows > 1`) | Supported |
| Fused decode RMSNorm, half-split RoPE, Turing GQA handling | Supported |
| Text prompts through the shared CLI | **Not wired** — the entrypoint takes token IDs |
| OpenAI-compatible serving | **Not wired** |
| 196,608-token context | **Not claimed** — the KV cache for a full-length request does not fit the baseline |

## Performance

4×RTX 2080 Ti 22 GiB, TP4, real `UD-IQ1_M` checkpoint.

| Measurement | Before | Current | Notes |
| --- | ---: | ---: | --- |
| Full-model 256-token prefill | 49.7 tok/s | **~104.9–107 tok/s** | Q4_K/Q5_K MMA enabled, all 62 layers |
| 43-layer decode benchmark | 5.76 tok/s | **10.32 tok/s** | Fused-RMSNorm milestone; **not** a 62-layer TPS claim |
| Per-layer decode in that 43-layer run | 3.84 ms | 2.33 ms | Same profile point as 10.32 tok/s |
| GPU memory | ~16 GiB/card | ~16 GiB/card | The MMA path did not materially change memory |

Read the decode column with its scope: the 10.32 tok/s figure was measured with a 43-layer debug
limit, while the RoPE milestone has a separate 62-layer result (7.55 versus 5.38 tok/s baseline).
These are sequential optimization milestones at different layer scopes, not one comparable suite.
Earlier MoE work moved 256-token prefill from 12.24 tok/s on the float path to about 49 tok/s through
IQ2 DP4A, which is the "before" column above.

## Known limitations

- **The advertised 196,608-token context does not imply a full-length request fits.** An FP16 KV cache
  at that depth exceeds the 4×22 GiB baseline.
- **The generation CLI is token-ID oriented.** Chat framing helpers exist but the text-in path through
  the shared CLI is not wired.
- **OpenAI-compatible serving is not wired**, so this model is CLI-only.
- **INT8 activation quantization in the MMA prefill path can alter later greedy tokens** through
  KV-cache state. The first generated token matched the float baseline in the measured run; later
  tokens can diverge, and that is disclosed behaviour rather than an exact-sequence claim.
- **The isolated kernel benchmark is not a model-level number.** Take TPS from the generation command.

## Where the detail is

- [Design and measurements](../architecture/minimax_m2_7_design.md) — the kernels, the decode profile,
  the scope each milestone was measured at, and the parity evidence.
- [MiniMax decode bottleneck analysis](../performance/minimax_decode_bottleneck_analysis.md).
- [Benchmarking rules](../guides/benchmarking.md) — required before any MiniMax number is quoted.
- The support matrix in [models/README.md](README.md).
