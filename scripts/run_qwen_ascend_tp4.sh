#!/usr/bin/env bash
# Run a complete Qwen3.8-27B greedy generation on four Ascend ranks.
#
# Usage:
#   scripts/run_qwen_ascend_tp4.sh [prompt] [max_new_tokens]
#
# One process owns one NPU. HCCL uses root-info rendezvous through the shared id
# path; HcclCommInitAll is intentionally not used on this CANN installation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/ascend_env.sh

MODEL="${QWEN_MODEL:-/mnt/data1/modelscope/Qwen/Qwen3.8-27B}"
BIN="${QWEN_BIN:-${REPO_ROOT}/cpp_engine/build-ascend/pocketllm_engine}"
PROMPT="${1:-The capital of France is}"
MAX_NEW="${2:-8}"
LAYERS="${QWEN_LAYERS:-0}"
TP_WORLD="${QWEN_TP_WORLD:-4}"
DEVICES_CSV="${QWEN_ASCEND_DEVICES:-0,1,2,3}"
TIMEOUT_SECONDS="${QWEN_ASCEND_TIMEOUT_SECONDS:-1800}"
PROFILE="${QWEN_ASCEND_PROFILE:-0}"

IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"
if [ "${#DEVICES[@]}" -ne "${TP_WORLD}" ]; then
    echo "expected ${TP_WORLD} devices, got ${#DEVICES[@]}: ${DEVICES_CSV}" >&2
    exit 2
fi
if [ ! -x "${BIN}" ]; then
    echo "Qwen Ascend binary is not executable: ${BIN}" >&2
    exit 2
fi

RUN_DIR="$(mktemp -d /tmp/pocketllm_qwen_ascend_tp4.XXXXXX)"
ID_PATH="${RUN_DIR}/hccl.id"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
# Rank startup and disk loading can be skewed by several minutes. Keep the
# collective wait independent from the process timeout, which is a whole-run cap.
export POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS="${POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS:-12000}"

cleanup_id() {
    rm -f "${ID_PATH}" "${ID_PATH}.tmp" "${ID_PATH}.cmdsock"
}
trap cleanup_id EXIT

printf 'qwen_ascend_tp4_run_dir=%s\n' "${RUN_DIR}"
printf 'qwen_ascend_model=%s\n' "${MODEL}"
printf 'qwen_ascend_prompt=%s\n' "${PROMPT}"
printf 'qwen_ascend_max_new_tokens=%s\n' "${MAX_NEW}"
printf 'qwen_ascend_layers=%s\n' "${LAYERS}"
printf 'qwen_ascend_profile=%s\n' "${PROFILE}"

pids=()
for rank in $(seq 0 $((TP_WORLD - 1))); do
    log="${RUN_DIR}/rank${rank}.log"
    cmd=("${BIN}"
         --ckpt "${MODEL}"
         --smoke-forward
         --smoke-layers "${LAYERS}"
         --resident-bench
         --prompt "${PROMPT}"
         --max-new-tokens "${MAX_NEW}"
         --tp-world "${TP_WORLD}"
         --tp-rank "${rank}"
         --device "${DEVICES[$rank]}"
         --nccl-id-path "${ID_PATH}")
    if [ "${PROFILE}" = "1" ]; then
        # msprof output must be separate per process; otherwise concurrent ranks
        # overwrite one another's task database. The application itself remains
        # unchanged, so the reported timings are the production path.
        msprof_cmd=(msprof
                    "--output=${RUN_DIR}/msprof-rank${rank}"
                    --ascendcl=on
                    --runtime-api=on
                    --task-time=on
                    --ai-core=on
                    --hccl=on
                    --msproftx=on
                    "${cmd[@]}")
        timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
            stdbuf -oL -eL "${msprof_cmd[@]}" >"${log}" 2>&1 &
    else
        timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
            stdbuf -oL -eL "${cmd[@]}" >"${log}" 2>&1 &
    fi
    pids+=("$!")
done

status=0
for rank in $(seq 0 $((TP_WORLD - 1))); do
    rc=0
    wait "${pids[$rank]}" || rc=$?
    if [ "${rc}" -ne 0 ]; then
        printf 'qwen_ascend_rank=%s status=failed rc=%s log=%s\n' \
            "${rank}" "${rc}" "${RUN_DIR}/rank${rank}.log" >&2
        status=1
    else
        printf 'qwen_ascend_rank=%s status=ok log=%s\n' \
            "${rank}" "${RUN_DIR}/rank${rank}.log"
    fi
done

# A TP run is not accepted merely because all processes exited. Every rank must
# commit the same greedy token at every step; local logits/checksums may differ
# because each rank owns a vocabulary shard.
if [ "${status}" -eq 0 ]; then
    reference="${RUN_DIR}/rank0.tokens"
    sed -nE 's/^generate_step=([0-9]+) token=([0-9]+).*/\1 \2/p' \
        "${RUN_DIR}/rank0.log" >"${reference}"
    if [ ! -s "${reference}" ]; then
        echo "no generation steps found in rank 0 log" >&2
        status=1
    else
        for rank in $(seq 1 $((TP_WORLD - 1))); do
            tokens="${RUN_DIR}/rank${rank}.tokens"
            sed -nE 's/^generate_step=([0-9]+) token=([0-9]+).*/\1 \2/p' \
                "${RUN_DIR}/rank${rank}.log" >"${tokens}"
            if ! diff -u "${reference}" "${tokens}"; then
                echo "greedy token parity failed for rank ${rank}" >&2
                status=1
            fi
        done
    fi
fi

cat "${RUN_DIR}/rank0.log"
printf 'qwen_ascend_tp4_status=%s\n' "${status}"
printf 'qwen_ascend_tp4_logs=%s\n' "${RUN_DIR}"
exit "${status}"
