#!/usr/bin/env bash
# Phase 3.5 performance benchmarks wrapper for TP4 execution
#
# Every measurement gets its own process.  Building a second engine inside one
# process costs the second one ~10% regardless of mode -- four serial engines in
# a row measured 0.672 / 0.735 / 0.675 / 0.674s at identical GPU clocks -- so a
# script that builds serial and batch back to back charges that penalty to
# whichever ran second and reports it as a batching effect.  Run per config, then
# compare the JSON.
set -euo pipefail

MODEL="${1:-/mnt/data2/Qwen3.8-27B-FP8}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${2:-$REPO/.scratch/phase3.5_benchmarks}"
TP="${TP:-4}"
MAX_TOKENS="${MAX_TOKENS:-32}"
PROMPT_LEN="${PROMPT_LEN:-16}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-5}"
CONCURRENCY="${CONCURRENCY:-2 4 8}"

mkdir -p "$OUTPUT_DIR"

echo "=== Phase 3.5 Performance Benchmarks ==="
echo "Model:  $MODEL"
echo "Output: $OUTPUT_DIR"
echo "TP:     $TP"
echo ""

echo "--- Single-request latency (one process per mode) ---"
for mode in serial batch; do
    echo "  measuring $mode..."
    python3 "$REPO/scripts/bench_single_request_latency.py" \
        "$MODEL" \
        --single "$mode" \
        --json-out "$OUTPUT_DIR/single_${mode}.json" \
        --max-tokens "$MAX_TOKENS" \
        --warmup-runs "$WARMUP" \
        --test-runs "$RUNS" \
        --prompt-length "$PROMPT_LEN" \
        --tensor-parallel-size "$TP" \
        > "$OUTPUT_DIR/single_${mode}.log" 2>&1
done

echo ""
echo "--- Concurrent throughput (one process per mode per level) ---"
for n in $CONCURRENCY; do
    for mode in serial batch; do
        echo "  measuring $mode at concurrency $n..."
        python3 "$REPO/scripts/bench_concurrent_throughput.py" \
            "$MODEL" \
            --single "$mode" \
            --json-out "$OUTPUT_DIR/concurrent_${n}_${mode}.json" \
            --num-concurrent "$n" \
            --max-tokens "$MAX_TOKENS" \
            --warmup-runs "$WARMUP" \
            --test-runs "$RUNS" \
            --prompt-length "$PROMPT_LEN" \
            --tensor-parallel-size "$TP" \
            > "$OUTPUT_DIR/concurrent_${n}_${mode}.log" 2>&1
    done
done

echo ""
python3 "$REPO/scripts/summarize_phase35_benchmarks.py" "$OUTPUT_DIR" \
    --concurrency $CONCURRENCY | tee "$OUTPUT_DIR/summary.txt"

echo ""
echo "=== Benchmarks Complete ==="
echo "Results saved to: $OUTPUT_DIR"
