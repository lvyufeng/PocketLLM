#!/bin/bash
# Launch the engine-agnostic C++ OpenAI-compatible server on four TP ranks.
# Rank 0 listens on $PORT (default 8000); ranks 1-3 are collective workers.
# The checkpoint's config.json selects the registered engine.
set -e

CKPT="${CKPT:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
PORT="${PORT:-8000}"
NCCL_ID="${NCCL_ID:-/tmp/pocketllm_cpp_serve_nccl.id}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
PREFILL_TOKEN_BUDGET="${PREFILL_TOKEN_BUDGET:-4096}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-0}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-900}"
# 0 means the checkpoint's full depth for every registered architecture.
SMOKE_LAYERS="${SMOKE_LAYERS:-0}"

if [ "$PREFILL_TOKEN_BUDGET" -lt 0 ]; then
    echo "PREFILL_TOKEN_BUDGET must not be negative" >&2
    exit 2
fi
if [ "$PREFILL_CHUNK_TOKENS" -lt 0 ]; then
    echo "PREFILL_CHUNK_TOKENS must not be negative" >&2
    exit 2
fi
if [ "$REQUEST_TIMEOUT_SECONDS" -le 0 ]; then
    echo "REQUEST_TIMEOUT_SECONDS must be positive" >&2
    exit 2
fi

if [ "$PREFILL_CHUNK_TOKENS" -gt 0 ]; then
    PREFILL_CHUNK_ARG="--prefill-chunk-tokens $PREFILL_CHUNK_TOKENS"
else
    PREFILL_CHUNK_ARG=""
fi
# Paging is ignored by an engine that does not support it. It is on here so a
# Qwen server shares one block pool across the requested batch width instead of
# reserving max_context separately for every slot.
KV_PAGED="${KV_PAGED:-1}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
PYTHON="${PYTHON:-python}"
SIDECAR="${SIDECAR:-/mnt/data1/dsv4_inference/src/server/cpp_sidecar.py}"
BIN="${BIN:-/mnt/data1/dsv4_inference/build/cpp_engine/pocketllm_engine}"
LOG_DIR="${LOG_DIR:-/tmp}"
EXTRA_ENV="${EXTRA_ENV:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

rm -f "$NCCL_ID"

COMMON="--serve --ckpt $CKPT --tp-world 4 --nccl-id-path $NCCL_ID --smoke-layers $SMOKE_LAYERS --max-context $MAX_CONTEXT --max-batch-size $MAX_BATCH_SIZE --prefill-token-budget $PREFILL_TOKEN_BUDGET --request-timeout-seconds $REQUEST_TIMEOUT_SECONDS $PREFILL_CHUNK_ARG --kv-block-size $KV_BLOCK_SIZE --python $PYTHON --sidecar $SIDECAR --port $PORT $EXTRA_ARGS"
if [ "$KV_PAGED" != "0" ]; then
    COMMON="$COMMON --kv-paged"
fi

pids=""
cleanup() {
    kill $pids 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for rank in 0 1 2 3; do
    eval "$EXTRA_ENV CUDA_VISIBLE_DEVICES=$rank $BIN $COMMON --tp-rank $rank --device 0 > $LOG_DIR/pocketllm_cpp_serve_rank${rank}.log 2>&1 &"
    pids="$pids $!"
done

echo "started 4 ranks; PIDs:$pids" >&2
echo "log dir: $LOG_DIR/pocketllm_cpp_serve_rank{0,1,2,3}.log" >&2
echo "tail logs with: tail -F $LOG_DIR/pocketllm_cpp_serve_rank0.log" >&2
wait
