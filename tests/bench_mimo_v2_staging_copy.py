#!/usr/bin/env python
"""What a 25.5 MiB expert staging copy costs when it is cut into pieces.

`MimoV2DeviceExperts._stage` copies one expert as six `copy_` calls -- one per (projection, kind)
tensor -- because the arena holds `w1, w2, w3` as separate tensors while the bank holds an expert as
one contiguous record. A draw of two experts is therefore twelve small copies of about 2 MiB each.
This measures whether that costs anything, by moving the same 25.5 MiB as twelve, six, three, two or
one copy. If the pieces are free, the copy stall the decode profile reports is bandwidth and the
arena's layout is not worth changing; if they are not, it is the launch count.

    python tests/bench_mimo_v2_staging_copy.py
"""

from __future__ import annotations

import torch

MIB = 2**20

#: An MXFP4 expert at MiMo's routed dimensions: gate and up [2048, 4096] at half a byte a weight,
#: down [4096, 2048] the same, and a scale byte a 32 elements. 12.75 MiB an expert.
GATE_BYTES = 2048 * 4096 // 2
DOWN_BYTES = 4096 * 2048 // 2
GATE_SCALE = (2048 * 4096) // 32
DOWN_SCALE = (4096 * 2048) // 32
EXPERT = 2 * (GATE_BYTES + GATE_SCALE) + DOWN_BYTES + DOWN_SCALE


def segment(weights: int, scales: int, rows: int) -> list[int]:
    """A tensor's own byte pieces, one a row -- the smallest the arena's rows can be written in."""
    return [weights // rows] * rows


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA device; nothing to measure")
        return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    host = torch.empty(4 * EXPERT, dtype=torch.uint8, pin_memory=True)
    print(f"one expert {EXPERT / MIB:.2f} MiB, a draw of two {2 * EXPERT / MIB:.2f} MiB")

    stream = torch.cuda.Stream()

    def measure(label: str, splits: int, chunks: list[int]) -> float:
        device_buf = torch.empty(sum(chunks), dtype=torch.uint8, device=device)
        offsets = [0]
        for size in chunks:
            offsets.append(offsets[-1] + size)
        source_offsets = [0]
        for size in chunks:
            source_offsets.append(source_offsets[-1] + size)

        def issue() -> None:
            for index in range(len(chunks)):
                device_buf[offsets[index] : offsets[index + 1]].copy_(
                    host[source_offsets[index] : source_offsets[index + 1]], non_blocking=True
                )

        for _ in range(3):
            with torch.cuda.stream(stream):
                issue()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize()

        best = 1e9
        for _ in range(20):
            with torch.cuda.stream(stream):
                start = torch.cuda.Event(True)
                start.record(stream)
                issue()
                end = torch.cuda.Event(True)
                end.record(stream)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        total = sum(chunks)
        print(
            f"{label:>34}: {best:6.3f} ms, {len(chunks):>2} copies, "
            f"{total / MIB / (best / 1e3) / 1024:5.2f} GiB/s"
        )
        return best

    def pieces(count: int) -> list[int]:
        """`count` equal pieces of a draw's 25.5 MiB: 12 is the module's, 2 is an expert a copy."""
        size = 2 * EXPERT // count
        return [size] * count

    results = {}
    results[12] = measure("twelve, the module's own", 12, pieces(12))
    results[6] = measure("six, a projection pair a copy", 6, pieces(6))
    results[3] = measure("three, a projection a copy", 3, pieces(3))
    results[2] = measure("two, an expert a copy", 2, pieces(2))
    results[1] = measure("one, the whole draw", 1, pieces(1))
    print()
    print(
        f"twelve copies cost {results[12] / results[1]:.2f}x one copy, "
        f"and {results[12] - results[1]:.3f} ms of the {results[12]:.3f}"
    )
    print(
        "The arena is read as [E, N, K/2] rows, so a per-projection pair is the widest piece the "
        "kernel's own layout allows without moving the bank."
    )


if __name__ == "__main__":
    main()
