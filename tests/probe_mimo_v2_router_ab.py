#!/usr/bin/env python
"""What the router costs the host, and what the C++ transcription of it costs instead.

`probe_mimo_v2_ablate.py` prices the router by deleting it, and that arm is only half honest: the
stub caches the first draw and returns it forty-seven times, so the expert path stages the same two
experts every layer and the resident set hits every layer. The arm's 36.5 ms is the router *and* the
copies a fixed draw does not make. So the router is priced here instead, directly: the two
implementations, on the release's own shapes, alternating in one process, with nothing else running.

Every routed layer in this checkpoint has the same router -- a `[256, 4096]` gate, a `[256]`
correction bias, and one row of hidden during a decode step -- so a micro-benchmark of one call is a
micro-benchmark of all forty-seven, and the print multiplies it out with that stated.

Values are random and it does not matter: the two paths run the same kernels on the same arguments,
which is the claim, and the point here is the clock rather than the arithmetic. The equality is
checked anyway, once, because a kernel that is fast and wrong should not be reported as fast.

    python tests/probe_mimo_v2_router_ab.py --rows 1 --calls 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402
from src.models.mimo_v2.layers import gate_and_route  # noqa: E402

DIM = 4096
N_EXPERTS = 256
TOP_K = 8
N_GROUP = 1
TOPK_GROUP = 1
SCALING = 2.5
ROUTED_LAYERS = 47


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--calls", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    kernel = load_cuda_kernel()
    if kernel is None or not hasattr(kernel, "mimo_noaux_tc_route"):
        print("the `cuda_kernel` extension is not built for this interpreter; nothing to compare")
        return 0
    if not torch.cuda.is_available():
        print("no CUDA device; nothing to measure")
        return 0

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    hidden = torch.randn(args.rows, DIM, device=device, dtype=torch.bfloat16)
    # The gate is float32 here because that is how the device layer holds it; the checkpoint's bf16
    # is upcast once at load, so the router never sees a bf16 weight at run time.
    gate = torch.randn(N_EXPERTS, DIM, device=device, dtype=torch.float32)
    bias = torch.randn(N_EXPERTS, device=device, dtype=torch.float32) * 0.1

    def reference():
        return gate_and_route(
            hidden,
            gate,
            bias,
            top_k=TOP_K,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            norm_topk_prob=True,
            routed_scaling_factor=SCALING,
        )[:2]

    def transcription():
        return kernel.mimo_noaux_tc_route(
            hidden, gate, bias, TOP_K, N_GROUP, TOPK_GROUP, True, float(SCALING)
        )

    want_idx, want_weight = reference()
    got_idx, got_weight = transcription()
    if not (torch.equal(got_idx, want_idx) and torch.equal(got_weight, want_weight)):
        print("the two paths disagree; the timings below are not worth reading")
        return 1

    def timed(call) -> tuple[float, float]:
        """This call's wall time: `calls` of them issued, and the stream's own queue behind them."""
        for _ in range(50):
            call()
        torch.cuda.synchronize()
        host = 0.0
        for _ in range(args.calls):
            start = time.perf_counter()
            call()
            host += time.perf_counter() - start
        before = time.perf_counter()
        torch.cuda.synchronize()
        queue = time.perf_counter() - before
        return host / args.calls * 1e6, queue / args.calls * 1e6

    took: dict[str, list[tuple[float, float]]] = {"gate_and_route": [], "mimo_noaux_tc_route": []}
    # Alternating by round, and the reference first in each: the box has drifted by tens of
    # milliseconds under a configuration held constant, so a pair minutes apart is not a pair.
    for _ in range(args.rounds):
        took["gate_and_route"].append(timed(reference))
        took["mimo_noaux_tc_route"].append(timed(transcription))

    print(f"{args.rows} row(s) of hidden, {N_EXPERTS} experts, top-{TOP_K}, {args.calls} calls a round")
    for name, rounds in took.items():
        host = sum(entry[0] for entry in rounds) / len(rounds)
        queue = sum(entry[1] for entry in rounds) / len(rounds)
        spread = f"{min(e[0] for e in rounds):.1f}-{max(e[0] for e in rounds):.1f}"
        print(
            f"{name:22s} host {host:8.1f} us  queue {queue:6.1f} us  spread {spread:>13s}  "
            f"({host * ROUTED_LAYERS / 1000:6.2f} ms of a token's {ROUTED_LAYERS} routers)"
        )
    faster = took["mimo_noaux_tc_route"]
    slower = took["gate_and_route"]
    saved = (
        sum(e[0] for e in slower) / len(slower) - sum(e[0] for e in faster) / len(faster)
    ) * ROUTED_LAYERS / 1000
    print(f"the transcription saves {saved:.2f} ms of a decode token's host time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
