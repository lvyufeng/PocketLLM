#!/bin/bash
# Profile a short Ascend run to collect op_summary and TransData attribution.
# Usage: ./profile_ascend_baseline.sh <output_dir>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir>"
    exit 1
fi

OUTPUT_DIR="$1"
CKPT="/mnt/data1/modelscope/Qwen/Qwen3.8-27B"
BINARY="cpp_engine/build-ascend/pocketllm_engine"
DEVICES=(0 1 2 3)
PROMPT_LENGTH=4096
MAX_NEW_TOKENS=8

source scripts/ascend_env.sh

mkdir -p "$OUTPUT_DIR"

# Generate token fixture from benchmark script's existing fixture
BENCH_FIXTURE=$(ls .tmp/qwen_ascend_long/fixture_${PROMPT_LENGTH}_*.json 2>/dev/null | head -1)
if [ -z "$BENCH_FIXTURE" ]; then
    echo "Error: No fixture found for length ${PROMPT_LENGTH}"
    echo "Run bench_qwen_ascend_long.py first to generate fixtures"
    exit 1
fi

FIXTURE_FILE="$OUTPUT_DIR/tokens_${PROMPT_LENGTH}.txt"
if [ ! -f "$FIXTURE_FILE" ]; then
    python3 -c "
import json
with open('$BENCH_FIXTURE') as f:
    fixture = json.load(f)
with open('$FIXTURE_FILE', 'w') as f:
    f.write(','.join(map(str, fixture['token_ids'])))
print(f'Extracted {len(fixture[\"token_ids\"])} tokens')
"
fi

# Run each rank with msprof
PIDS=()
for RANK in "${!DEVICES[@]}"; do
    DEVICE="${DEVICES[$RANK]}"
    RANK_OUTPUT="$OUTPUT_DIR/rank${RANK}"
    mkdir -p "$RANK_OUTPUT"

    LOG_FILE="$RANK_OUTPUT/engine.log"

    # Set up HCCL for multi-rank
    export HCCL_WHITELIST_DISABLE=1
    export POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=12000

    # Create shared HCCL ID file path
    HCCL_ID_PATH="$OUTPUT_DIR/hccl_profile.id"

    # msprof command with per-rank output directory
    # New syntax: msprof [options] <app> [app args]
    msprof --output="$RANK_OUTPUT" \
           --task-time=on \
           --ai-core=on \
           --hccl=on \
           "$BINARY" \
           --ckpt "$CKPT" \
           --smoke-forward \
           --resident-bench \
           --tp-world 4 \
           --tp-rank $RANK \
           --device $DEVICE \
           --nccl-id-path "$HCCL_ID_PATH" \
           --token-ids-file "$FIXTURE_FILE" \
           --max-new-tokens $MAX_NEW_TOKENS \
           --smoke-layers 0 \
           --kv-cache-dtype fp16 \
           > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

# Wait for all ranks
echo "Waiting for ${#PIDS[@]} profiling processes..."
for PID in "${PIDS[@]}"; do
    wait $PID || echo "Warning: Process $PID failed"
done

echo "Profiling complete. Results in $OUTPUT_DIR"
echo "Extract op_summary with: msprof --export=on --output=$OUTPUT_DIR/rank0"
