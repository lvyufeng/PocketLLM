"""Benchmark Q4_K / Q5_K MMA prefill GEMM vs the float baseline.

Measures raw GEMM throughput for a real MiniMax attention projection
(Q5_K, [3072]x[6144]) at several batch sizes, MMA off vs on. This isolates
the projection speedup from the rest of the model.
"""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch

from pocketllm.loader.gguf.bundle import read_gguf_bundle
from pocketllm.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def load_q5k_proj(name: str, rows: int):
    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    for sh in bundle.shards:
        f = sh.file
        if hasattr(f, "tensors") and name in {t.name for t in f.tensors}:
            reader = get_cached_gguf_tensor_reader(sh.path)
            break
    blocks, type_name, K = reader.read_quantized_matrix_block_rows(name, 0, rows)
    type_id = 3 if type_name.lower() == "q4_k" else 4
    return blocks.contiguous().to("cuda"), int(K), type_id


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000  # ms


def main():
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    device = torch.device("cuda:0")

    # Q5_K attention q_proj: 3072 -> 6144, load first 6144 rows (full output dim).
    blocks, K, type_id = load_q5k_proj("blk.0.attn_q.weight", rows=6144)
    N = blocks.size(0)
    print(f"Q5_K q_proj: K={K}, N={N}, type_id={type_id}")
    grid = torch.empty(0, dtype=torch.int8, device=device)

    for batch in [16, 64, 128, 256, 512, 1024]:
        x = torch.randn(batch, K, device=device, dtype=torch.float16)

        os.environ.pop("GGUF_Q4K_Q5K_MMA", None)
        ms_float = bench(lambda: cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid))

        os.environ["GGUF_Q4K_Q5K_MMA"] = "1"
        ms_mma = bench(lambda: cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid))
        os.environ.pop("GGUF_Q4K_Q5K_MMA", None)

        speedup = ms_float / ms_mma if ms_mma > 0 else 0.0
        tflops_float = 2 * batch * K * N / (ms_float * 1e-3) / 1e12
        tflops_mma = 2 * batch * K * N / (ms_mma * 1e-3) / 1e12
        print(
            f"  batch={batch:5d}  float={ms_float:7.3f}ms ({tflops_float:5.1f} TFLOPS)  "
            f"mma={ms_mma:7.3f}ms ({tflops_mma:5.1f} TFLOPS)  speedup={speedup:5.2f}x"
        )


if __name__ == "__main__":
    main()
