#!/bin/bash
# Run cpp_engine/tests/test_multi_slot_decode_parity at tp_world 1 or 4.
#
# The test needs one process per rank and a rendezvous file only rank 0 writes,
# so a stale id file from a previous group makes the next group die inside
# ncclCommInitRank with "remote process exited or there was a network error".
# This script deletes it before every run and kills the whole group on exit --
# a rank that outlives its group sits on a GPU and makes the next run look like
# a hardware fault.
#
# Every rank uses a distinct physical GPU: the test binds device = tp_rank, so
# CUDA_VISIBLE_DEVICES is deliberately not set. Offsets let a run use GPUs 2-3
# (the NVLink pair) without touching 0-1.
set -u

CKPT="${CKPT:-/mnt/data3/DeepSeek-V4-Flash-0731}"
TP_WORLD="${TP_WORLD:-4}"
GPU_OFFSET="${GPU_OFFSET:-0}"
LAYERS="${LAYERS:-4}"
STEPS="${STEPS:-8}"
PROMPT_LEN="${PROMPT_LEN:-6}"
BIN="${BIN:-/mnt/data1/dsv4_inference/cpp_engine/build/tests/test_multi_slot_decode_parity}"
NCCL_ID="${NCCL_ID:-/tmp/multi_slot_decode_parity_nccl.id}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/msp}"
OUT_DIR="${OUT_DIR:-/tmp/msp_out}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
# EXTRAS carries per-run env for every rank, e.g. "POCKETLLM_PARITY_TRACE=1".
EXTRAS="${EXTRAS:-}"

if [ "$TP_WORLD" -ne 1 ] && [ "$TP_WORLD" -ne 4 ]; then
    echo "TP_WORLD must be 1 or 4" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "not executable: $BIN" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
rm -f "$NCCL_ID"

# One out.bin per rank: --compare diffs the tp_world 1 recording against the
# tp_world 4 one, and rank 0 is the only rank whose recording is meaningful.
#
# Killing $! is not enough to stop a rank. Each line is a pipeline
# (eval -> stdbuf -> binary, or -> tee), so $! is the shell or tee at the near
# end and the binary at the far end survives its parent, gets reparented to
# init, and spins on its GPU forever -- which shows up as an unexplained 100%
# utilization and a next run that cannot tell a stale rank from a real fault.
# Match the binary by its own command line instead: the rendezvous path is
# unique per LOG_PREFIX, so this kills this run's ranks and nothing else.
pids=""
kill_ranks() {
    local signal="$1"
    pkill "-$signal" -f "$BIN .*$NCCL_ID" 2>/dev/null || true
}
cleanup() {
    kill_ranks TERM
    sleep 2
    kill_ranks KILL
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# A group left over from an earlier run of the same prefix would hold the GPUs
# and, once the id file below is gone, never rendezvous. Clear it first.
stale=$(pgrep -f "$BIN .*$NCCL_ID" 2>/dev/null || true)
if [ -n "$stale" ]; then
    echo "killing stale ranks from a previous run: $stale" >&2
    kill_ranks TERM
    sleep 2
    kill_ranks KILL
fi

# device = tp_rank, so the visible set has to be the whole machine; GPU_OFFSET
# moves the group without changing which index each rank binds.
if [ "$TP_WORLD" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPU_OFFSET" eval "$EXTRAS stdbuf -oL -eL $BIN $CKPT $LAYERS $STEPS $PROMPT_LEN 1 0 $NCCL_ID $OUT_DIR/tp1.rank0.bin 2>&1 | tee $LOG_PREFIX.tp1.r0.log" &
    pids="$pids $!"
else
    for rank in 0 1 2 3; do
        eval "$EXTRAS stdbuf -oL -eL $BIN $CKPT $LAYERS $STEPS $PROMPT_LEN 4 $rank $NCCL_ID $OUT_DIR/tp4.rank${rank}.bin > $LOG_PREFIX.tp4.r${rank}.log 2>&1" &
        pids="$pids $!"
    done
fi

echo "started ranks:$pids; logs $LOG_PREFIX.*; id file $NCCL_ID" >&2

deadline=$((SECONDS + TIMEOUT_SECONDS))
while kill -0 $pids 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "TIMEOUT after ${TIMEOUT_SECONDS}s" >&2
        cleanup
        exit 124
    fi
    sleep 5
done

echo "all ranks exited" >&2
