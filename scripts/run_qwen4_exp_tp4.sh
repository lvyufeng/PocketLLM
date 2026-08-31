#!/usr/bin/env bash
# TP4 heterogeneous inference for Qwen3.8-Flash-Next.
#
# Dense weights are sharded across the four 2080 Ti cards.  Each rank loads its
# disjoint 56.25 GiB share of the 225 GiB routed experts into host RAM at
# startup (round-robin by expert id, so the four shards are one full copy); the
# 95 GiB PLE table stays host-resident through the mapping.
#
# Usage: scripts/run_qwen4_exp_tp4.sh [prompt] [max_new_tokens]
set -euo pipefail

MODEL="${QWEN4EXP_MODEL:-/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next}"
PROMPT="${1:-用一句话介绍你自己。}"
MAX_NEW="${2:-16}"
CHUNK="${QWEN4EXP_CHUNK:-512}"
PYTHON="${QWEN4EXP_PYTHON:-/home/lvyufeng/miniconda3/envs/deepseek/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# The 2080 Ti has no NVLink between all pairs here; P2P over PCIe is fine for
# the per-layer all-reduce but must not fall back to a broken path silently.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-29571}" \
  -m src.models.qwen4_exp.runtime \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW" \
  --chunk-size "$CHUNK"
