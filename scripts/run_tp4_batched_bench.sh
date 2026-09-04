#!/bin/bash
# Launch batched throughput benchmark in TP4 mode

set -e

CHECKPOINT="${1:-/mnt/data2/Qwen3.8-27B-FP8}"
BATCH_SIZE="${2:-4}"
CONTEXT="${3:-4096}"
DECODE="${4:-16}"
MEASURED="${5:-10}"

NCCL_ID="/tmp/qwen_batched_bench_nccl_$(date +%s)"

echo "Launching TP4 batched throughput benchmark"
echo "Checkpoint: $CHECKPOINT"
echo "Batch size: $BATCH_SIZE"
echo "NCCL ID: $NCCL_ID"
echo ""

# Launch rank 0 in foreground to capture output
CUDA_VISIBLE_DEVICES=0 ./tests/bench_qwen_batched_throughput "$CHECKPOINT" \
    --batch-size "$BATCH_SIZE" \
    --context "$CONTEXT" \
    --decode "$DECODE" \
    --measured "$MEASURED" \
    --tp-world 4 \
    --tp-rank 0 \
    --device 0 \
    --nccl-id "$NCCL_ID" &
PID0=$!

# Launch workers in background
for rank in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank ./tests/bench_qwen_batched_throughput "$CHECKPOINT" \
        --batch-size "$BATCH_SIZE" \
        --context "$CONTEXT" \
        --decode "$DECODE" \
        --measured "$MEASURED" \
        --tp-world 4 \
        --tp-rank $rank \
        --device 0 \
        --nccl-id "$NCCL_ID" >"/tmp/qwen_batched_bench_rank${rank}.log" 2>&1 &
done

# Wait for rank 0 (it prints the results)
wait $PID0
STATUS=$?

# Clean up NCCL ID
rm -f "$NCCL_ID"

exit $STATUS
