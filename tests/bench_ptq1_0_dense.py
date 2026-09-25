"""Throughput of the sm_75 ternary dense GEMM over Ternary-Bonsai-2-27B's real weights.

The issue's acceptance asks for a tok/s number at the same prompt and context as the
FP8 Qwen3.8-27B path.  That path cannot be timed per kernel -- its projections live in
the C++ runtime -- so this measures the arm that is comparable and leaves the model
number to the serving stage:

* **ternary** -- `gguf_quant_gemm_prefill_forward` / `gguf_quant_gemm_forward` over the
  checkpoint's packed weights, 1.75 bits apiece;
* **cublas fp16** -- the same weights dequantized to fp16 and handed to cuBLAS, which
  is what the FP8 Qwen3.8-27B path does after its online unpack (fp8 -> fp16 -> cuBLAS
  FP16, the only route to a GEMM with no tensor-core fp8 on Turing).  Ten times the
  bytes, so this arm also says how much of the win is the format and how much is the
  kernel.

One representative tensor per distinct shape is read and timed, then multiplied by how
many tensors the checkpoint has with that shape, because the cost depends on `[N, K]`
and on nothing else.  The sum is the dense-projection cost of one forward pass, and
`rows / total` is the token rate it affords with the attention stack and the launch
overhead taken out -- so it is a ceiling, not a model measurement.

    python -m tests.bench_ptq1_0_dense                 # decode, then prefill at 512
    python -m tests.bench_ptq1_0_dense --rows 128 512 2048

The card is at its 260 W power cap and the first run of a series reads high, so the two
arms are interleaved within each shape rather than run in two blocks.
"""

from __future__ import annotations

import argparse
import collections
import os
import statistics
import time
from pathlib import Path

import torch

PTQ1_0_TYPE_ID = 143
CHECKPOINT_DEFAULT = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"


def inventory(path: str) -> list[tuple[tuple[int, int], list[str]]]:
    """Distinct `(N, K)` shapes among the `ptq1_0` tensors, with the names that share one.

    The GGUF dimensions are `(ne0, ne1) = (K, N)`, so a tensor is `[N, K]` row-major
    with K fastest -- which is the layout both kernels read.
    """
    from src.loader.gguf.tensor_reader import GGUFTensorDataReader

    groups: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    with GGUFTensorDataReader(path) as reader:
        for tensor in reader.gguf.tensors:
            if tensor.type_name != "ptq1_0":
                continue
            k, n = (int(d) for d in tensor.dimensions)
            groups[(n, k)].append(tensor.name)
    return sorted(groups.items(), key=lambda item: -item[0][0] * item[0][1])


def time_it(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3  # ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_BONSAI_GGUF", CHECKPOINT_DEFAULT))
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 512])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    from src.kernels.cuda_loader import load_cuda_kernel
    from src.loader.gguf import ptq1_0
    from src.loader.gguf.tensor_reader import GGUFTensorDataReader

    cuda = load_cuda_kernel()
    if cuda is None or not hasattr(cuda, "gguf_ptq1_0_dp4a_decode_forward"):
        raise SystemExit("the cuda_kernel extension is not built, or predates the PTQ1_0 kernels")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(386)

    shapes = inventory(args.checkpoint)

    print(f"checkpoint: {args.checkpoint}")
    print(f"{len(shapes)} distinct shapes, {sum(len(n) for _, n in shapes)} ptq1_0 tensors\n")

    grid = torch.empty(0, dtype=torch.int8, device=device)
    #: shape -> (count, packed bytes, ms at rows=1, ms at rows=512, cublas ms at rows=1, ...)
    timings: dict[tuple[int, int], dict[str, float]] = {}

    # Every figure below is a synchronised round trip rather than a stream of launches,
    # and this host's round trip is 19 us -- a fifth of the small shapes.  The floor is
    # measured with an empty kernel and subtracted per call from the sums at the end;
    # the smallest call either entry point accepts is reported next to it as a check
    # that the fixed cost of the PTQ1_0 API is in the same range.
    scratch = torch.zeros(1, device=device)
    launch_floor = time_it(lambda: scratch.fill_(0), args.iters, args.warmup)
    smallest_call = time_it(
        lambda: cuda.gguf_quant_gemm_forward(
            torch.zeros(1, 128, device=device), torch.zeros(1, 1, 28, dtype=torch.uint8, device=device),
            128, PTQ1_0_TYPE_ID, grid),
        args.iters, args.warmup)
    print(f"empty-kernel round trip: {launch_floor * 1e3:.1f} us; "
          f"smallest PTQ1_0 decode call: {smallest_call * 1e3:.1f} us\n")

    for (n, k), names in shapes:
        reader = GGUFTensorDataReader(args.checkpoint)
        blocks_host, type_name, k_read = reader.read_quantized_matrix_block_rows(names[0], 0, n)
        assert type_name == "ptq1_0" and int(k_read) == k
        blocks_host = blocks_host.numpy()
        blocks = torch.from_numpy(blocks_host).to(device).contiguous()
        row = timings.setdefault((n, k), {"count": float(len(names)), "bytes": float(blocks.nbytes)})

        weight_fp16 = None
        # Interleave the arms rather than blocking them: the card drifts downward
        # within a series, so a block of ternary followed by a block of cuBLAS reads
        # the drift as a speedup.
        for rows in args.rows:
            x = torch.randn(rows, k, device=device, dtype=torch.float32)
            f16 = weight_fp16
            if f16 is None:
                f16 = torch.from_numpy(ptq1_0.dequantize_blocks(blocks_host)).to(device, torch.float16)
                f16 = f16.reshape(n, k).contiguous()
                weight_fp16 = f16
            a16 = x.to(torch.float16)

            def ternary() -> None:
                if rows == 1:
                    cuda.gguf_quant_gemm_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid)
                else:
                    cuda.gguf_quant_gemm_prefill_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid)

            def cublas() -> None:
                torch.mm(a16, f16.t())

            ms_ternary = time_it(ternary, args.iters, args.warmup)
            ms_cublas = time_it(cublas, args.iters, args.warmup)
            row[f"ternary@{rows}"] = ms_ternary
            row[f"cublas@{rows}"] = ms_cublas
            gib = blocks.nbytes / 2**30
            print(
                f"  N={n:6d} K={k:5d} x{len(names):3d}  rows={rows:4d}  "
                f"ternary {ms_ternary:7.3f} ms ({gib / (ms_ternary * 1e-3):6.1f} GiB/s, "
                f"{2 * rows * n * k / (ms_ternary * 1e-3) / 1e12:5.1f} TFLOP/s)   "
                f"cublas {ms_cublas:7.3f} ms ({ms_ternary / ms_cublas:4.2f}x)"
            )
        del weight_fp16
        reader.close()
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("dense projections of one forward pass, summed over the whole checkpoint\n")
    for rows in args.rows:
        t_ternary = sum(c["count"] * c[f"ternary@{rows}"] for c in timings.values())
        t_cublas = sum(c["count"] * c[f"cublas@{rows}"] for c in timings.values())
        gib = sum(c["count"] * c["bytes"] for c in timings.values()) / 2**30
        k_ternary = t_ternary - launch_floor
        k_cublas = t_cublas - launch_floor
        print(
            f"  rows={rows:5d}   ternary {t_ternary:9.2f} ms  -> {rows * 1e3 / t_ternary:7.1f} tok/s   "
            f"({gib / (t_ternary * 1e-3):6.1f} GiB/s over {gib:.3f} GiB)"
            f"   |   cublas-fp16 {t_cublas:9.2f} ms -> {rows * 1e3 / t_cublas:7.1f} tok/s"
            f"   ({t_ternary / t_cublas:4.2f}x)"
        )
        print(
            f"  {'':16s}  minus one {launch_floor * 1e3:.0f} us launch: "
            f"{k_ternary:9.2f} ms -> {rows * 1e3 / k_ternary:7.1f} tok/s ({gib / (k_ternary * 1e-3):6.1f} GiB/s)"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
