# Qwen4-Exp Heterogeneous TP4 Performance

**Date:** 2026-08-31

**Hardware:** 4 x RTX 2080 Ti, TP4 over PCIe Gen3
**Model:** Qwen3.8-Flash-Next, real checkpoint at
`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`

## Runtime architecture

The real model has 48 layers, hidden size 2560, 512 routed experts, and top-10
routing. One BF16 routed expert is 9.375 MiB, and the full routed-expert set is
about 225 GiB.

The routed experts are copied explicitly into host RAM during model startup:

- expert ownership is round-robin: `expert_id % world_size == rank`;
- every rank owns 128 experts per layer, or 56.25 GiB across 48 layers;
- the four disjoint rank shards together hold exactly one copy of the expert
  set;
- the shard is pinned by default, with an all-or-nothing pageable fallback;
- only active experts are copied to the rank's GPU and all MoE arithmetic runs
  on the GPU;
- the 95 GiB PLE table remains host-side.

This document uses *host-resident* only for the explicit 56.25 GiB/rank copy.
A warm mmap or page-cache hit is not treated as host residency. The latest warm
preload took about 53 seconds per rank; it is a one-time load cost and is not
included in steady-state throughput.

## Authoritative 8K TP4 prefill result

Measurement conditions:

- 8192 real tokenizer tokens;
- one 8192-token chunk;
- TP4;
- expert cache capacity 1024;
- pinned 56.25 GiB host expert shard per rank;
- first pass discarded as warmup;
- no synchronization-heavy profiler in the timed region;
- all rows below produced token 97792;
- preload excluded.

### Accuracy-first path

The accuracy-first candidate uses transient FP16 tensor-core dense projections
and the CUDA hyper-connection RMSNorm/injection path. Decode rows stay BF16
because all prefill precision gates require at least 128 rows.

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 788.07 / 784.47 tok/s | 10.419 s | **786.27 tok/s** | 14.108 GiB |
| FP16 dense + CUDA HC | 951.04 / 947.77 tok/s | 8.629 s | **949.41 tok/s** | 13.839 GiB |
| BF16 baseline recheck | 770.43 / 767.59 tok/s | 10.653 s | **769.01 tok/s** | 14.108 GiB |

The candidate is a 20.7% gain over the first baseline mean and reaches a stable
about 950 tok/s, just below the 1000 tok/s target.

A 512-token prompt plus 32 forced-token decisions was checked on Chinese,
English, and code inputs. Chinese and code matched all greedy decisions. The
English arm changed decisions 4 and 23; the baseline margins were 0.7500 and
0.1875 respectively. Therefore this path is still opt-in rather than a strict
parity default.

The prefill precision switches do not affect single-token decode. With cache
1024 and a 512-token prefix, a separate 32-step forced-token A/B measured:

| Prompt | BF16 decode | Candidate decode |
|---|---:|---:|
| Chinese | 4.365 tok/s | 4.363 tok/s |
| English | 4.378 tok/s | 4.388 tok/s |
| Code | 4.367 tok/s | 4.370 tok/s |

This is within run noise and confirms the rows=1 BF16 fallback protects decode.

### Throughput-first path: target exceeded, not promoted

A more accurate FP16 GEMM form uses FP16 tensor-core inputs, FP32 accumulation,
and one BF16 output cast. Applying it to dense projections and the HC mix-down
and mix-up projections gives:

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 784.68 / 780.69 / 777.29 tok/s | 10.491 s | **780.89 tok/s** | 14.108 GiB |
| FP32-output dense + HC down/up | 1063.42 / 1061.49 / 1059.48 tok/s | 7.718 s | **1061.46 tok/s** | 13.961 GiB |
| BF16 baseline recheck | 765.46 / 762.26 / 760.08 tok/s | 10.742 s | **762.60 tok/s** | 14.108 GiB |

This exceeds the 1000 tok/s goal by 6.1% and is 35.9% faster than the first
baseline mean. It is not promoted as a correctness-safe default:

- 512-token Chinese and code prompts matched all 33 decisions;
- the English prompt changed decisions 19 and 23;
- decision 19 was an exact baseline tie, but decision 23 had a baseline margin
  of 0.1875 and the candidate selected the other token with a 0.9375 margin;
- a 2048-token plus 8-decision matrix matched all decisions on all three
  prompts, so the mismatch is prompt- and margin-sensitive rather than a
  general runtime failure.

The correct status is therefore: **the performance target is reached by an
experimental precision mode, while the accuracy-first mode remains about
950 tok/s**.

The 8192-token chunk is also the correct operating point. A clean strict-BF16
sweep measured 154.70, 257.88, 409.14, 580.68, and 753.25 tok/s for chunks 512,
1024, 2048, 4096, and 8192 respectively. Smaller chunks do not improve
correctness and sharply increase repeated per-layer dispatch, communication,
and active-expert staging overhead.

## Remaining prefill attribution

A synchronization-heavy real TP4 profile is for attribution only. Under the
accuracy-first configuration, the main 8K phases were:

| Phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Routed MoE | 48 | 7.777 s | 162.03 ms |
| Attention | 48 | 1.484 s | 30.92 ms |
| MLP hyper-connection | 48 | 0.863 s | 17.97 ms |
| Attention hyper-connection | 48 | 0.844 s | 17.58 ms |
| MLP all-reduce | 48 | 0.666 s | 13.88 ms |
| Attention all-reduce | 48 | 0.356 s | 7.41 ms |
| Router | 48 | 0.207 s | 4.31 ms |
| PLE | 1 | 0.149 s | 148.98 ms |

MoE attribution on rank 0 reported 4.189 seconds staging, 3.090 seconds
compute, 4652 active expert calls, and 42.59 GiB H2D. Turning off the existing
copy-stream prefetch reduced the no-profile path from about 950 tok/s to about
727 tok/s, so the overlap must remain enabled.

Attention sub-phases were:

| Sub-phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Core | 48 | 0.808 s | 16.84 ms |
| Projection | 48 | 0.294 s | 6.12 ms |
| QSA indexer | 12 | 0.165 s | 13.73 ms |
| Output | 48 | 0.139 s | 2.89 ms |
| GatedDeltaNet convolution | 36 | 0.076 s | 2.12 ms |

The isolated 8K GatedDeltaNet layer measured about 20.2 ms with transient FP16
projections. Its recurrent core was 8.66 ms. Sweeping 1/2/4/8 work groups per
CTA did not improve the full layer, so CTA grouping is not the next lever.

A synchronization-free 512-token PyTorch trace also showed that all 96 NCCL
collectives ran on the default compute stream: NCCL occupied 327.46 ms, BF16
GEMMs 1.353 s, and pinned H2D copies 2.651 s, with zero timeline overlap between
NCCL and either GEMM or H2D. Moving NCCL to a separate stream reduced an
isolated 8192 x 2560 all-reduce from 7.54 to 7.28 ms, but an independent 0.58 ms
GEMM still did not overlap because both kernels saturate SM resources. The real
8K path gained only about 0.8%, so simple stream reassignment is not a path to
strict-parity 1000 tok/s. Projection slicing or a reduce-scatter formulation is
needed to create useful pipeline granularity.

## Numerical gates and rejected candidates

The parity harness generates one BF16 greedy chain per prompt, feeds that same
chain to every candidate, and reports the BF16 top-1/top-2 margin for every
mismatch. Exact ties are separated from real flips.

### Rejected or diagnostic precision paths

| Candidate | Kernel or 8K result | Parity verdict |
|---|---:|---|
| FP16 dense projections alone | fast | English decision 1 flips at BF16 margin 3.4375 |
| FP32-output dense projections alone | 885.48 tok/s | Only an exact-tie English decision changes; no non-tie flip in the 512+32 matrix |
| FP32-output dense + CUDA HC | 956.80 tok/s | English decision 23 flips at margin 0.1875; grouped RMSNorm is not bit-exact |
| FP32-output dense + HC mix-down only | 1013.28 tok/s | English decisions 0 and 1 flip at margins 4.3750 and 3.4375 |
| FP32-output dense + HC mix-up only | 1005.89 tok/s | English decision 0 flips at margin 4.3750 |
| FP32-output dense + HC injection only | 940.27 tok/s | English decision 23 flips at margin 0.1875 |
| FP16 dense + FP16 HC down/up | earlier 1043.35 tok/s was not reproducible | Chinese decision 5 flips at margin 0.9375 |
| FP32-output dense + HC down/up | 1061.46 tok/s | English decision 23 flips at margin 0.1875 |
| Full FP16 routed MoE | 2.65x kernel speedup | English decision 0 flips at margin 4.375 |
| FP16 gate / BF16 down MoE | 1.75x kernel speedup | real Chinese and English flips |
| BF16 gate / FP16 down MoE | 1.23x kernel speedup | English decision 23 flips at margin 0.1875 |

The routed-MoE FP16 experiments are kept opt-in and disabled by default. Their
small average activation error is not sufficient to preserve real generation.

### Exact-BF16 hyper-connection limits

Two scale-then-activation pairs now use dedicated CUDA kernels when
`DSV4_QWEN4_HC_CUDA=1`: HC mix-down divide plus SiLU, and block-injection divide
plus `2 * sigmoid`. Direct tests at 1, 7, and 8192 rows are bit-identical to the
PyTorch BF16 expressions. At 8192 rows the isolated pairs improved from about
0.069 to 0.054 ms and from 0.083 to 0.049 ms respectively. Their aggregate
end-to-end ceiling is only about 5 ms per 8192-token prefill because each pair is
small relative to the model wall.

The larger CUDA grouped RMSNorm used by the same HC switch is not strictly
bit-exact to PyTorch: on real layer-0 weights at 8192 rows, 99.999541% of values
were equal and the maximum absolute difference was 0.0625. Therefore references
to the CUDA HC configuration mean an accuracy-tested near-parity path, not a
bit-exact implementation. The activation kernels themselves remain exact.

A steady-state TP4 run with two global 8K warmups is the usable end-to-end
measurement for the two activation fusions. With prompt=8192, chunk=8192,
TP4, and expert cache=1024, strict BF16 measured 769.12 / 765.31 tok/s
(767.21 mean), CUDA HC measured 801.09 / 799.17 tok/s (800.13 mean), and the
BF16 recheck measured 752.99 / 750.73 tok/s (751.86 mean). All arms produced
token 97792. This comparison includes the pre-existing non-exact CUDA grouped
RMSNorm and injection kernels; it does not isolate the roughly 5 ms activation
fusion gain. A later confirmatory run suffered an external machine-wide
slowdown from about 786 to 428 tok/s and is discarded rather than averaged into
these numbers. A 512-token Chinese prompt plus eight forced decode decisions
also kept every greedy decision unchanged; the full-logit delta reflects the
pre-existing grouped-RMSNorm approximation rather than rows=1 activation use.

An isolated strict-BF16 HC call at 8192 rows measured 25.363 ms. The two BF16
mix GEMMs consumed 7.071 and 7.610 ms; a kernel trace attributed 14.134 of
16.950 ms of CUDA time to `aten::mm`. On SM75 these GEMMs use
`magma_sgemmEx_kernel` at about 50% occupancy.

A separate optimistic compute-floor probe replaced every staged expert pair by
one GPU-resident pair, removing all 42.76 GiB of expert H2D while preserving the
same dispatch and GEMM shapes. At prompt=8192/chunk=8192/TP4, the real path
measured 779.93 tok/s (10.504 s) and the no-H2D floor measured 909.07 tok/s
(9.011 s). Perfect staging overlap therefore still cannot reach 1000 tok/s,
which requires 8.192 s; at least 0.819 s of compute and/or launch overhead must
also be removed. Elementwise or reduction fusion cannot supply that margin. The
remaining strict-path work must target the HC/attention/MoE compute structure,
not H2D overlap alone.

The weighted mean is already a 100%-occupancy BF16 multiply followed by a
100%-occupancy `MeanOps` reduction. Replacing `.mean(-2)` with
`.sum(-2) / hc_count` is bit-identical but slower. Hand-written left-to-right,
pairwise, and `einsum` reductions do not reproduce PyTorch's reduction order;
`einsum` is also much slower. These variants were rejected.

### Split-K and cached FP16 weight experiments

Splitting the K dimension into FP16 tensor-core partial GEMMs does not reproduce
the SM75 BF16 linear result. Equal-element fractions remained roughly
0.9984-0.9997 for every tested chunk size, while BF16 itself was not bit-equal
to an FP32 reference. Chunk sizes below 256 were also slower than BF16.

Caching FP16 mirrors of dense weights did not materially improve isolated GEMMs:
for example, the 8K GatedDeltaNet QKV projection was 2.360 ms with a transient
weight conversion and 2.365 ms with a cached FP16 weight. Permanent mirrors
would consume substantial VRAM without a measurable benefit, so weights remain
BF16 and conversion is transient.

## Confirmed architecture decisions

1. **Keep routed experts explicitly resident in host RAM.** Mechanical-disk mmap
   faults must never be part of steady-state inference.
2. **Keep per-rank round-robin ownership.** TP4 stages one disjoint 56.25 GiB
   shard per rank and never duplicates routed-expert work.
3. **Keep active-only H2D.** The full-local-layer grouped BF16 path increases
   H2D bytes by about 31.6% and regressed real 8K TP4 prefill to 577.13 tok/s.
   `DSV4_QWEN4_MOE_GROUPED` stays disabled by default.
4. **Keep MoE copy-stream prefetch enabled.** It is worth about 950 versus
   727 tok/s in the current accuracy-first path.
5. **Keep prefill and decode precision paths separate.** Large prefill rows may
   use the experimental FP16 tensor-core modes; rows=1 remain BF16.
6. **Do not promote a candidate from token 0 alone.** Real forced-token matrices
   and baseline margins are required.

## Reproduction commands

Accuracy-first 8K A/B:

```bash
PYTHONPATH=. \
DSV4_QWEN4_DENSE_FP16=1 \
DSV4_QWEN4_HC_CUDA=1 \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 3 --global-warmup 2 \
  --arms baseline,dense_hc_cuda,baseline_recheck
```

Throughput-first 8K A/B:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 4 --global-warmup 2 \
  --arms baseline,dense_fp32_hc_down_up_fp32,baseline_recheck
```

Forced-token margin matrix:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/check_qwen4_exp_tp_long_matrix.py \
  --prompt-tokens 512 --decode-tokens 32 --expert-cache 0 \
  --prompts chinese,english,code \
  --arms dense_fp32_hc_down_up_fp32
```
