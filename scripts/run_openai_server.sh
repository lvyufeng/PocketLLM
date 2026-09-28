#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEEPSEEK_GPU_MOE_DECODE_ACTIVE="${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-0}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2}"

SERVING_PROFILE="${SERVING_PROFILE:-safe1}"
case "$SERVING_PROFILE" in
  safe1)
    export DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS="${DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS:-1}"
    export DEEPSEEK_SERVING_BATCH_WAIT_MS="${DEEPSEEK_SERVING_BATCH_WAIT_MS:-0}"
    ;;
  latency2)
    export DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS="${DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS:-2}"
    export DEEPSEEK_SERVING_BATCH_WAIT_MS="${DEEPSEEK_SERVING_BATCH_WAIT_MS:-200}"
    ;;
  throughput4)
    export DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS="${DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS:-4}"
    export DEEPSEEK_SERVING_BATCH_WAIT_MS="${DEEPSEEK_SERVING_BATCH_WAIT_MS:-250}"
    ;;
  *)
    echo "unknown SERVING_PROFILE=$SERVING_PROFILE; expected safe1, latency2, or throughput4" >&2
    exit 1
    ;;
esac
export DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS="${DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS:-256}"
export DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH="${DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH:-$DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS}"

PYTHON="${PYTHON:-python}"
MASTER_PORT="${MASTER_PORT:-29920}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_CKPT_PATH="$ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CKPT_FORMAT="${CKPT_FORMAT:-auto}"
CONFIG="${CONFIG:-$ROOT/configs/config_w8a8.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
PARTITION_POLICY="${PARTITION_POLICY:-legacy}"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
MODEL_ID="${MODEL_ID:-deepseek-v4-flash-w8a8}"
ROUTED_EXPERTS_DEVICE="${ROUTED_EXPERTS_DEVICE:-cpu}"
PD_MODE="${PD_MODE:-scheduler}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# `pocketllm serve` owns the rank fan-out now. `--tensor-parallel-size` starts the
# supervisor, which assigns the rendezvous environment and runs each nonzero rank
# through the torch backend's worker loop. This script used to hand the same job to
# `torchrun --module src.server.openai`, a second front end that has since been
# deleted (issue #447). `--master-port` went with torchrun; the supervisor reads
# MASTER_PORT from the environment instead, so keeping it exported keeps the port
# choice this script made.
export MASTER_PORT
ARGS=(
  --backend torch
  --host "$HOST"
  --port "$PORT"
  --model "$CKPT_PATH"
  --model-format "$CKPT_FORMAT"
  --config-path "$CONFIG"
  --served-model-name "$MODEL_ID"
  --routed-experts-device "$ROUTED_EXPERTS_DEVICE"
  --pd-mode "$PD_MODE"
  --backend-option "partition_policy=$PARTITION_POLICY"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$NPROC_PER_NODE"
)
if [[ -n "$TOKENIZER_PATH" ]]; then
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi

PYTHONPATH="$ROOT" exec "$PYTHON" -m pocketllm serve "${ARGS[@]}"
