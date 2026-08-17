# GLM-5.2

## Runtime status

**Validated text generation.** PocketLLM supports GLM-5.2 `glm-dsa` GGUF text-in/text-out greedy generation, including GLM chat framing, the dense prefix, the full MoE trunk, rank-local expert execution, and generated-token decoding.

The default routed-expert layout is expert parallelism (EP). Several alternatives remain available as opt-in experiments but are disabled because they did not improve real end-to-end performance.

## Model specification

The validated `UD-Q2_K_XL` bundle reports:

| Field | Value |
| --- | ---: |
| Architecture | `glm-dsa` |
| Physical blocks | 79 |
| Main transformer trunk | 78 layers |
| Leading dense layers | 3 |
| Trailing speculative/NextN blocks | 1 |
| Hidden size | 6144 |
| Vocabulary | 154,880 |
| Context length | 1,048,576 |
| Query heads | 64 |
| KV heads | 1 |
| Head dimension | 576 |
| RoPE dimension | 64 |
| Routed experts | 256 |
| Active experts | top-8 |
| Expert intermediate size | 2048 |
| Shared experts | 1 |
| GGUF tensors | 1,809 |

The runnable trunk excludes the final NextN/MTP block. PocketLLM follows the main residual stream rather than treating all 79 physical blocks as ordinary Transformer layers.

## Implemented execution path

- GLM-specific tokenizer and `[gMASK]<sop>` chat framing.
- Dense-prefix and MoE-layer raw-block GGUF loading.
- IQ2_XS w1/w3 and IQ3_XXS w2 DP4A grouped MoE kernels.
- Q8_0 attention/shared-expert dispatch and rank-sharded vocabulary head.
- Default EP expert partition across TP4 ranks.
- Optional full-file page-cache prewarm for HDD/SMR-backed checkpoints.
- Per-stage GLM decode profiler for staging, attention, MoE, and all-reduce diagnosis.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, real `UD-Q2_K_XL`, warm page cache, full model unless noted.

| Runtime state | Prefill | Decode | Status |
| --- | ---: | ---: | --- |
| Float/general MoE baseline | ~0.66 tok/s | ~0.54 tok/s | Historical baseline |
| IQ2_XS/IQ3_XXS DP4A default | ~0.79 tok/s | ~0.66 tok/s | Current validated fast path |
| Resident CPU expert cache | ~0.17 tok/s | ~0.44 tok/s | Regression; opt-in, default off |
| Routed tensor parallelism | not a win | ~0.54 tok/s vs ~0.60 EP A/B | Regression; opt-in, default off |
| Fused RMSNorm | ~0.78 tok/s | ~0.64 tok/s | Neutral/noisy; opt-in, default off |

These GLM numbers come from different targeted A/B sessions and are presented as runtime milestones, not as one statistically controlled leaderboard. The current default path is the DP4A EP configuration.

The detailed decode profile found two hard floors on the measured system: active-expert staging and per-layer NCCL synchronization. A later routed-TP experiment removed rank skew but staged all eight active experts on every rank; the extra per-expert staging calls outweighed the communication improvement.

## Correctness and precision

- Real-bundle schema validation covers all 1,809 mapped tensors.
- Tokenizer tests cover Chinese/English round trips and GLM chat special-token ordering.
- IQ2_XS/IQ3_XXS DP4A kernels are checked against the FP32 reference. A representative real-layer result was `max_abs=0.033`, `mean_abs=0.005`, `p99_rel=0.30`.
- Routed TP's four-way intermediate sum matched full computation to approximately `max_abs=6e-7`, but the path remained slower and is not the default.
- Fused RMSNorm changes the greedy stream and has no measured e2e win, so the default keeps the previous path.

## Reproduction

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_glm \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --prompt "请用一句话介绍你自己。" \
  --chat \
  --max-new-tokens 32 \
  --prewarm
```

The command reads the real block count and runs the 78-layer main trunk. Set `GLM_PROFILE=1` to print the detailed phase breakdown; profiler overhead means those runs are diagnostic, not headline benchmarks.

Inspect the bundle:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --architecture auto \
  --spec-summary --validate-spec \
  --capability-report --placement-report
```

## Known limitations

- The checkpoint advertises a 1M-token context, but PocketLLM has not established a full-context performance or memory claim for this runtime.
- Decode is slow on the baseline system and is highly sensitive to page-cache, storage, NUMA, expert staging, and NCCL behavior.
- The default path is EP. `GLM_ROUTED_TP=1`, `GLM_ENABLE_RESIDENT_EXPERTS=1`, and `GLM_FUSED_RMSNORM=1` are experiments with documented regressions or neutral results.
- OpenAI-compatible serving is not wired.

## Evidence and related notes

- `src/models/glm_dsa/spec.py`
- `src/models/glm_dsa/architecture.py`
- `src/cli/generate_glm.py`
- `tests/test_glm_dsa_spec.py`
- `tests/test_encoding_glm_dsa.py`
- `tests/test_glm_dsa_iq2xs_iq3xxs_dp4a.py`
- `tests/test_glm_dsa_tp_routed.py`
- [Benchmark reporting rules](../benchmarking.md)
