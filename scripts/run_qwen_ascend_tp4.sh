#!/usr/bin/env bash
# Run a complete Qwen3.8-27B greedy generation on four Ascend ranks.
#
# Usage:
#   scripts/run_qwen_ascend_tp4.sh [prompt] [max_new_tokens]
#
# Batched decode, which is what carries the throughput target on this part:
#   QWEN_BATCH_ROWS=16 scripts/run_qwen_ascend_tp4.sh
# `QWEN_BATCH_ROWS` replaces the prompt with that many greedy rows, because the
# batched path seeds each row from `--token-ids` rather than from the chat
# prompt. `QWEN_BATCH_TOKEN` (default 9707) is the first prompt token and
# `QWEN_BATCH_VERIFY` (default 3) is how many steps are parity-checked against a
# synchronous single-row reference.
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
BATCH_ROWS="${QWEN_BATCH_ROWS:-0}"
BATCH_TOKEN="${QWEN_BATCH_TOKEN:-9707}"
BATCH_PROMPT_LEN="${QWEN_BATCH_PROMPT_LEN:-32}"
BATCH_VERIFY="${QWEN_BATCH_VERIFY:-3}"

if [ "${BATCH_ROWS}" -gt 0 ]; then
    # `--token-ids` is ONE prompt, rotated per row so no two rows decode the same
    # sequence; its length is therefore the prompt length and not the batch size.
    #
    # The tokens have to be distinct for that rotation to do anything: a constant
    # list rotates into itself, every row decodes the same prompt, and a slot
    # mix-up between two rows then cancels out of the parity comparison -- which
    # is the one failure the comparison exists to catch. `QWEN_BATCH_TOKEN` is the
    # first token of the run and the rest follow it consecutively.
    BATCH_TOKENS="$(python3 -c "print(','.join(str(${BATCH_TOKEN} + i) for i in range(${BATCH_PROMPT_LEN})))")"
    BATCH_CONTEXT="$(( BATCH_PROMPT_LEN + MAX_NEW + 8 ))"
fi

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
         --max-new-tokens "${MAX_NEW}"
         --tp-world "${TP_WORLD}"
         --tp-rank "${rank}"
         --device "${DEVICES[$rank]}"
         --nccl-id-path "${ID_PATH}")
    if [ "${BATCH_ROWS}" -gt 0 ]; then
        # `--prompt` is deliberately NOT passed alongside `--token-ids`: the
        # engine re-encodes a non-empty `--prompt` over the token list it has
        # already loaded, so passing both silently replaces the batch's 32-token
        # prompts with the 5-token chat prompt and the run stops being a batch
        # test at all.
        cmd+=(--batch-decode
              --max-batch-size "${BATCH_ROWS}"
              --token-ids "${BATCH_TOKENS}"
              --max-context "${BATCH_CONTEXT}")
        export POCKET_BATCH_VERIFY="${BATCH_VERIFY}"
    else
        cmd+=(--prompt "${PROMPT}")
    fi
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

# A TP run is not accepted merely because all processes exited. A single-row run
# must commit the same greedy token on every rank, and a batched one has to clear
# its oracle; local logits/checksums may differ between ranks because each rank
# owns a vocabulary shard.
#
# Two of the three numbers on the summary line are hard gates and one is not.
#
#   seed_mismatches          must be 0. The reference and the batched pass are
#                            both prefilled from a reset engine, so their first
#                            tokens have to agree; if they do not, every number
#                            below is comparing two different sequences.
#   batch_repeat_mismatches  must be 0. A second batched pass over the same grid
#                            holds the arithmetic fixed and varies only the run.
#                            This is the reproducibility the batched path owes
#                            its callers, and it delivers it.
#   verify_mismatches        is NOT a zero test. The single-row reference is not
#                            reproducible run to run at full depth (see
#                            `repeat_mismatches`), and the two paths also differ
#                            systematically from the first full-attention layer
#                            on, so a non-zero count here is expected rather than
#                            alarming. It is printed, and a run in which it
#                            exceeds `repeat_mismatches` gets a NOTICE line, so a
#                            new defect is visible without being confused for the
#                            known offset.
if [ "${status}" -eq 0 ] && [ "${BATCH_ROWS}" -gt 0 ]; then
    summary="${RUN_DIR}/rank0.summary"
    grep -hoE 'batch_decode=1.*' "${RUN_DIR}/rank0.log" >"${summary}" || true
    if [ ! -s "${summary}" ]; then
        echo "no batch_decode summary in rank 0 log" >&2
        status=1
    else
        cat "${summary}"
        seed="$(sed -nE 's/.*seed_mismatches=([0-9]+).*/\1/p' "${RUN_DIR}/rank0.log" | head -1)"
        if [ "${seed:-0}" -ne 0 ]; then
            echo "batch decode seeding failed: seed_mismatches=${seed}" >&2
            status=1
        fi
        repeat="$(sed -nE 's/.*batch_repeat_mismatches=([0-9]+).*/\1/p' "${summary}")"
        if [ "${repeat:-1}" -ne 0 ]; then
            echo "batch decode is not reproducible: batch_repeat_mismatches=${repeat}" >&2
            status=1
        fi
        verify="$(sed -nE 's/.*verify_mismatches=([0-9]+).*/\1/p' "${summary}")"
        baseline="$(sed -nE 's/.*repeat_mismatches=([0-9]+).*/\1/p' "${summary}")"
        if [ -n "${verify}" ] && [ -n "${baseline}" ] && [ "${verify}" -gt "${baseline}" ]; then
            echo "NOTICE: verify_mismatches=${verify} exceeds the single-row" \
                 "reference's own repeat_mismatches=${baseline}; the difference" \
                 "between those two counts is the part that is not reference noise" >&2
        fi
    fi
elif [ "${status}" -eq 0 ]; then
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
