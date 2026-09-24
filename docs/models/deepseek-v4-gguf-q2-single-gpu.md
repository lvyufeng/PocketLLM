# DeepSeek-V4 GGUF Q2 on one GPU

The single-card path for the DeepSeek-V4 GGUF IQ2_XXS/Q2_K checkpoint, with the routed experts kept in
host memory. It works, and it is not a serving configuration: this page is the measurement record
behind that conclusion.

- **Use it for**: smoke tests, demos, and very short prompts on one RTX 2080 Ti
- **Do not use it for**: anything interactive beyond a few hundred prompt tokens
- **Prefer instead**: the four-GPU GGUF Q2 resident path or the FP4 resident path — see
  [the DeepSeek-V4 guide](deepseek-v4.md)

## Run it

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
CASE=short_short \
REPEAT=1 \
MAX_MODEL_LEN=1024 \
bash scripts/run_gguf_q2_tp_resident.sh
```

With `NPROC_PER_NODE=1` the script automatically switches to `PARTITION_POLICY=legacy` and disables
GPU prefill MoE (`DEEPSEEK_GGUF_GPU_PREFILL_MOE=0`, `DEEPSEEK_GGUF_GPU_GROUPED_MOE=0`), while keeping
the active-expert decode accelerations on (`..._DECODE_ACTIVE_EXPERT`, `..._DECODE_GROUPED`,
`..._DECODE_SINGLE_TOKEN`, `..._DECODE_SLOT_CACHE` with a cache size of 16).

Full-layer grouped prefill staging is off on one GPU for a memory reason: staging all 256 local routed
experts for a single layer is about **1.7 GiB per layer**. This mode therefore keeps the routed GGUF
experts in host memory and stages only the active decode experts to the GPU — which is exactly what
makes long prompts slow.

## What to expect

1 × RTX 2080 Ti 22 GiB, the same dual-Xeon 1 TiB host as the four-GPU numbers, `NPROC_PER_NODE=1`,
routed experts on CPU, measured through the OpenAI-compatible resident benchmark with no explicit
warmup unless noted.

| Case | Prompt tokens | Decode tokens | Prefill | Decode | Wall | Host PSS peak | GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short prompt, cold | 5 | 7 | 4.50 s (1.11 tok/s) | 2.11 tok/s | 8.09 s | 21.33 GiB | 20.34 GiB |
| Short prompt, repeat | 5 | 7 | 4.87 s (1.03 tok/s) | 3.12 tok/s | 7.24 s | 21.33 GiB | 20.34 GiB |
| Forced 64-token decode, run 1 | 24 | 63 | 8.57 s (2.80 tok/s) | 2.51 tok/s | 34.21 s | 38.85 GiB | 20.29 GiB |
| Forced 64-token decode, run 2 | 24 | 63 | 13.28 s (1.81 tok/s) | 2.29 tok/s | 41.27 s | 38.85 GiB | 20.29 GiB |
| ~128-token prompt | 149 | 7 | 26.02 s (5.73 tok/s) | 1.76 tok/s | 30.17 s | 50.44 GiB | 20.87 GiB |
| ~256-token prompt | 290 | 7 | 41.93 s (6.92 tok/s) | 1.58 tok/s | 46.55 s | 55.38 GiB | 20.87 GiB |
| ~512-token prompt | 557 | 7 | 78.06 s (7.14 tok/s) | 1.51 tok/s | 84.28 s | 61.21 GiB | 20.87 GiB |
| ~1024-token prompt | 1,045 | 7 | 139.98 s (7.47 tok/s) | 1.54 tok/s | 144.73 s | 64.49 GiB | 20.88 GiB |
| ~2048-token prompt | 2,101 | n/a | did not finish in a useful window | n/a | stopped | 71.58 GiB | 21.55 GiB |

A 5-token prompt also completes with `MAX_MODEL_LEN=131072`, so the configuration can allocate a large
maximum sequence for tiny requests. That does not make long prompts usable: what dominates is
routed-expert prefill, not the nominal `MAX_MODEL_LEN`.

## Practical conclusion

Useful as a smoke/demo path, and it produces short responses on one card: for short inputs, decode is
around 2.3–2.5 tok/s on longer forced output, with occasional short-cache runs near 3 tok/s.

It is not a practical long-prompt serving path. By ~128–256 prompt tokens the time to first token is
already tens of seconds; at 512–1024 tokens it is over a minute. The mechanism is the one above —
long prefill falls back to the CPU/host GGUF routed-expert path and touches large parts of the routed
expert mmap.

For production-like use on this repository, prefer the four-GPU GGUF Q2 TP resident path or the FP4
resident path. Treat single-GPU Q2 as functional validation and very short-prompt experimentation.

Back to [the current DeepSeek-V4 guide](deepseek-v4.md).
