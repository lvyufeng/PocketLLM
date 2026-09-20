#!/usr/bin/env python3
"""Roofline for the Qwen3.8-27B decode step on first-generation Ascend 910B.

Reads the bench JSONs a sweep produced and prints, per row count, the step time,
the row-step rate (rows x steps/s, which is the work the decoder retires while
the batch is full), the achieved TFLOP/s, and how far that is from the card's
ceiling and from the memory floor of one weight read.

The ceilings are read from the platform, not assumed:

  * `ai_core_cnt=30`, `cube_m/n/k_size=16`, `cube_freq=900` out of CANN's
    `Ascend910B.ini`, giving 2 x 30 x 16^3 x 900 MHz = 221.2 TFLOP/s a card.
  * the HBM read rate is this host's recorded probe, not the card's datasheet
    figure, because the probe is what the step actually has to beat.

The model arithmetic is Qwen3.8-27B's: 64 layers, 48 of them linear attention
and 16 full, hidden 5120, intermediate 17408, vocab 248320, head_dim 256.

usage: analyze_serving_roofline.py <tag.json> [tag.json ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# --- model arithmetic -------------------------------------------------------
HIDDEN = 5120
VOCAB = 248320
INTERMEDIATE = 17408
LAYERS_LINEAR, LAYERS_FULL = 48, 16


def _gemm(out_features: int, in_features: int) -> int:
    """MACs x 2 for one projection."""
    return 2 * out_features * in_features


# Linear-attention layer: q/k/v/g/b projections, the output projection, and the
# three MLP matrices.
_LINEAR = (_gemm(10240, HIDDEN) + _gemm(6144, HIDDEN) + _gemm(48, HIDDEN) + _gemm(48, HIDDEN)
           + _gemm(HIDDEN, 6144) + 3 * _gemm(INTERMEDIATE, HIDDEN))
# Full-attention layer: q/k/v (24/4/4 heads of 256), the output projection, MLP.
_FULL = (_gemm(12288, HIDDEN) + _gemm(1024, HIDDEN) + _gemm(1024, HIDDEN) + _gemm(HIDDEN, 6144)
         + 3 * _gemm(INTERMEDIATE, HIDDEN))
_HEAD = _gemm(VOCAB, HIDDEN)

FLOP_PER_TOKEN = LAYERS_LINEAR * _LINEAR + LAYERS_FULL * _FULL + _HEAD

_PARAMS = (LAYERS_LINEAR * (10240 * HIDDEN + 6144 * HIDDEN + 96 * HIDDEN + HIDDEN * 6144
                            + 3 * INTERMEDIATE * HIDDEN)
           + LAYERS_FULL * (12288 * HIDDEN + 2048 * HIDDEN + HIDDEN * 6144
                            + 3 * INTERMEDIATE * HIDDEN)
           + 2 * VOCAB * HIDDEN)
BYTES_PER_RANK = 2 * _PARAMS / 4          # fp16, TP4

# --- platform ceilings ------------------------------------------------------
CORES, CUBE, FREQ_HZ = 30, 16 * 16 * 16, 900e6      # Ascend910B.ini
PEAK_PER_CARD = CORES * CUBE * FREQ_HZ * 2           # FLOP/s, fp16 in, fp32 accumulate
HBM_PROBE = 1148e9                                   # this host's recorded read probe, per card
ENGINE_RESIDENT_BYTES = 13449011456                  # the engine's own resident-weight counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--cards", type=int, default=4, help="TP world size (default 4).")
    parser.add_argument("--hbm-bytes-per-second", type=float, default=HBM_PROBE,
                        help="Measured HBM read rate a card; the machine balance depends on it.")
    parser.add_argument("--resident-bytes", type=float, default=ENGINE_RESIDENT_BYTES,
                        help="Resident weight bytes a rank, as the engine reports them.")
    parser.add_argument("--prefill-ms", type=float, default=None,
                        help="Measured prefill time for one prompt, to add a prefill row. "
                             "The ladder's TTFT slope is that quantity measured directly.")
    parser.add_argument("--prefill-tokens", type=int, default=None,
                        help="Tokens in that prompt, as the engine counted them. This is NOT "
                             "--random-input-len: the bench builds the prompt from characters and "
                             "the server re-tokenizes it, so the two differ by the tokenizer's "
                             "real characters-per-token ratio. Read it from the engine log's "
                             "`request prompt_tokens=` line.")
    args = parser.parse_args()

    peak_all = args.cards * PEAK_PER_CARD
    balance = PEAK_PER_CARD / args.hbm_bytes_per_second
    intensity = FLOP_PER_TOKEN / args.cards / args.resident_bytes

    print(f"FLOP/token      {FLOP_PER_TOKEN / 1e9:.3f} GFLOP   "
          f"({LAYERS_LINEAR}x{_LINEAR / 1e6:.1f} + {LAYERS_FULL}x{_FULL / 1e6:.1f} "
          f"+ head {_HEAD / 1e6:.1f} MFLOP)")
    print(f"params          {_PARAMS / 1e9:.3f} B, per-rank fp16 {BYTES_PER_RANK / 2**30:.3f} GiB")
    print(f"engine resident {args.resident_bytes / 2**30:.3f} GiB "
          f"(diff {args.resident_bytes - BYTES_PER_RANK:,.0f} B = norms + alignment)")
    print(f"peak/card       {PEAK_PER_CARD / 1e12:.1f} TFLOP/s "
          f"({CORES} cores x 16^3 x {FREQ_HZ / 1e6:.0f} MHz x 2), {peak_all / 1e12:.1f} across {args.cards}")
    print(f"machine balance {balance:.1f} FLOP/byte "
          f"at {args.hbm_bytes_per_second / 1e9:.0f} GB/s a card")
    print(f"intensity/token {intensity:.3f} FLOP/byte  "
          f"({balance / intensity:.0f}x below the ridge)")
    print(f"weight floor    {args.resident_bytes / args.hbm_bytes_per_second * 1000:.1f} ms "
          f"for a row-independent step")

    print()
    header = (f"{'rows':>4} {'step_ms':>8} {'rowstep/s':>9} {'meas tok/s':>10} "
              f"{'TFLOPS':>7} {'%peak':>6} {'GB/s':>7} {'%HBM':>5} "
              f"{'ms/row':>7} {'steps/s':>7}")
    print(header)
    print("-" * len(header))
    previous: tuple[int, float] | None = None
    for path in args.files:
        d = json.loads(pathlib.Path(path).read_text())
        m = d["metrics"]
        # The bench JSON carries no `max_batch_size`, so the batch width of a
        # run is its client-side concurrency, which every sweep row sets equal
        # to the server's slot count.
        rows = d.get("max_concurrency", d.get("max_batch_size", 0))
        step_ms = m["tpot"]["mean"] * 1000
        tflops = rows * FLOP_PER_TOKEN / (step_ms / 1000) / 1e12
        gbs = args.resident_bytes / (step_ms / 1000) / 1e9
        # A marginal ms per row is only meaningful between two different widths;
        # a second run at the same width is a repeat, not a slope.
        if previous is None or rows == previous[0]:
            marginal = "n/a" if previous is None else "-"
        else:
            marginal = f"{(step_ms - previous[1]) / (rows - previous[0]):.2f}"
        print(f"{rows:>4} {step_ms:>8.2f} {rows / (step_ms / 1000):>9.2f} "
              f"{m.get('output_throughput', 0):>10.2f} {tflops:>7.3f} "
              f"{100 * tflops * 1e12 / peak_all:>5.2f}% {gbs:>7.1f} "
              f"{100 * gbs * 1e9 / args.hbm_bytes_per_second:>4.1f}% {marginal:>7} "
              f"{1000 / step_ms:>7.1f}")
        previous = (rows, step_ms)

    if args.prefill_ms is not None:
        if not args.prefill_tokens:
            parser.error("--prefill-ms needs --prefill-tokens; --random-input-len is not it")
        # Prefill is on the other side of the ridge and paid for by the whole
        # model, not by one rank's share of it.
        prompt_flops = args.prefill_tokens * FLOP_PER_TOKEN
        prefill_tflops = prompt_flops / (args.prefill_ms / 1000) / 1e12
        print()
        print(f"prefill, one {args.prefill_tokens}-token prompt "
              f"{args.prefill_ms:.0f} ms -> {args.prefill_tokens / (args.prefill_ms / 1000):.0f} tok/s, "
              f"{prefill_tflops:.1f} TFLOP/s, {100 * prefill_tflops * 1e12 / peak_all:.1f}% of peak")
        print(f"prefill intensity {prompt_flops / args.resident_bytes:.0f} FLOP/byte, "
              f"{prompt_flops / args.resident_bytes / balance:.0f}x above the ridge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
