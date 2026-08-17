# MiniMax-M2.7

## Runtime status

**Validated TP4 generation.** PocketLLM can load the real `UD-IQ1_M` sharded GGUF bundle, validate its model schema, keep each rank's routed-expert partition on device, and run raw-block CUDA greedy generation.

The shared `src.cli.generate_gguf` entrypoint accepts token IDs. MiniMax tokenizer helpers are implemented and tested, but there is no dedicated OpenAI-compatible server adapter.

## Model specification

The validated checkpoint reports:

| Field | Value |
| --- | ---: |
| Architecture | `minimax-m2` |
| Layers | 62 |
| Hidden size | 3072 |
| Vocabulary | 200,064 |
| Context length | 196,608 |
| Query heads | 48 |
| KV heads | 8 |
| Head dimension | 128 |
| Routed experts | 256 |
| Active experts | top-8 |
| Expert intermediate size | 1536 |
| GGUF tensors | 809 |

`UD-IQ1_M` uses `iq2_xxs` for all three routed-expert matrices, `q5_k` for attention projections, `q4_k` for embedding/output, and floating-point router/norm/bias tensors.

## Implemented execution path

- Raw GGUF block loading with rank-local expert ranges.
- TP4 expert parallelism and NCCL reduction.
- IQ2_XXS DP4A grouped MoE paths for w1/w3 and w2.
- Q4_K/Q5_K INT8 MMA prefill path based on the vendored llama.cpp MMQ machinery.
- Separate decode path: the Q4/Q5 MMA hook only runs for `rows > 1`.
- Fused CUDA RMSNorm for decode.
- Fused half-split RoPE and MiniMax-specific GQA handling on Turing.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4, real `UD-IQ1_M` checkpoint.

| Measurement | Before | Current measured result | Notes |
| --- | ---: | ---: | --- |
| Full-model 256-token prefill | 49.7 tok/s | ~104.9–107 tok/s | Q4_K/Q5_K MMA enabled, all 62 layers |
| 43-layer decode benchmark | 5.76 tok/s | 10.32 tok/s | Fused RMSNorm milestone; not a 62-layer TPS claim |
| Per-layer decode in that 43-layer run | 3.84 ms | 2.33 ms | Same profile point as 10.32 tok/s |
| GPU memory | ~16 GiB/card | ~16 GiB/card | MMA path did not materially change memory |

Earlier MoE work moved 256-token prefill from 12.24 tok/s on the float path to approximately 49 tok/s through IQ2 DP4A; Q4/Q5 MMA then produced the approximately 2.1× full-model improvement above. The 10.32 tok/s decode number was measured with a 43-layer debug limit, whereas the RoPE milestone also has a separate 62-layer result (7.55 tok/s versus 5.38 baseline). These are sequential optimization milestones with different layer scopes, not one directly comparable benchmark suite.

## Correctness and precision

- IQ2 DP4A MoE paths maintained exact token parity with their measured float baseline.
- Fused RMSNorm maintained the same generated sequence as its baseline.
- The Q4_K/Q5_K MMA path quantizes activations to INT8. Against the float dequant path, representative projection error was `max_abs≈0.06`, `mean_abs≈0.01`, and `p99_rel≈0.15` for `|y|>0.1`.
- The first generated token matched the float baseline in the measured run. Later tokens can diverge through KV-cache rounding; this is disclosed behavior, not an exact-sequence claim.

## Reproduction

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_gguf \
  --gguf-path /path/to/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --seed-file /path/to/prompt_tokens.bin \
  --max-new-tokens 32 \
  --prewarm
```

Inspect and validate the bundle:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/MiniMax-M2.7-GGUF/UD-IQ1_M \
  --architecture auto \
  --spec-summary --validate-spec \
  --capability-report --placement-report
```

Isolated Q4/Q5 prefill kernel benchmark:

```bash
PYTHONPATH=$PWD python tests/bench_q4k_q5k_mma_prefill.py
```

The isolated benchmark is useful for kernel A/B work, but model-level TPS should come from the generation command.

## Known limitations

- The advertised 196,608-token context would require a large FP16 KV cache; the checkpoint's context metadata does not imply that every full-length request fits the 4×22 GiB baseline.
- Text-in/text-out MiniMax chat framing is implemented in encoding helpers, but the shared generation CLI is token-ID oriented.
- OpenAI-compatible serving is not wired.
- INT8 activation quantization in the MMA prefill path can alter later greedy tokens through KV-cache state.

## Evidence and related notes

- [MiniMax decode bottleneck analysis](../minimax_decode_bottleneck_analysis.md)
- `src/models/minimax_m2/spec.py`
- `src/models/minimax_m2/architecture.py`
- `tests/test_minimax_m2_spec.py`
- `tests/test_encoding_minimax_m2.py`
- `tests/test_q4k_q5k_mma.py`
- `tests/test_minimax_iq2xxs_w2_dp4a.py`
- [Benchmark reporting rules](../benchmarking.md)
