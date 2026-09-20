#!/bin/bash
# Serving width sweep driver.
#
# Every point launches the native engine through `scripts/bench_serving.py` with
# the two opt-in IPC collective switches on, and scrapes the engine's /metrics
# for the whole run so the server-side phase split survives the teardown. The
# artifacts of a point are four files under $POCKET_SWEEP_DIR:
#
#   <tag>.json      the bench's own record (the only source for latency figures)
#   <tag>.metrics   the last /metrics scrape (the only source for phase splits)
#   <tag>.out       the bench's console output, kept for provenance only
#   logs-<tag>/     the per-rank engine logs
#
# The scrape is taken by `_serving_metrics_scrape.py`, which polls while the
# bench runs, so the engine has to stay up past the last request for the
# counters to be complete: `point` passes --server-drain-seconds for exactly
# that, and the scrape is short by the requests in flight when it is missing.
#
# The console table has no Std column and truncates when piped, so read `tok/s`
# and the percentiles out of the .json and nothing else. `summarize_serving_sweep.py`
# and `analyze_serving_roofline.py` do exactly that.
#
# usage:
#   run_serving_sweep.sh point <tag> <slots> <concurrency> <prompts> <in> <out> <rate> <ctx> [K=V ...]
#   run_serving_sweep.sh ladder   # slots == prompts, one wave: the width ladder
#   run_serving_sweep.sh limit    # where the KV pool stops fitting a rank
#   run_serving_sweep.sh ab       # interleaved replicate-rows A/B at concurrency 1
#
# `slots` is --max-batch-size, which is the concurrency ceiling: the scheduler
# admits only while the live slot count is below it. `concurrency` is the
# client's own in-flight cap; a concurrency below `slots` is what makes a
# sweep row a sustained batch rather than a single draining wave.
#
# Environment:
#   POCKET_SWEEP_DIR      artifact directory             (default ./sweep-out)
#   POCKET_SWEEP_CKPT     checkpoint                     (required)
#   POCKET_SWEEP_BINARY   engine binary                  (default cpp_engine/build-ascend/pocketllm_engine)
#   POCKET_SWEEP_DEVICES  device list                    (default 0,1,2,3)
#   POCKET_SWEEP_PORT     server port, matches --port    (default 18280)
#   POCKET_SWEEP_GOODPUT  SLO triple                     (default ttft:2000 tpot:200 e2el:30000)
#   PFB                   --prefill-token-budget         (default 4096)
#   POCKET_SWEEP_DRAIN    --server-drain-seconds, kept up so the /metrics scrape
#                         sees the final request          (default 1.0)
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SWEEP_DIR=${POCKET_SWEEP_DIR:-$ROOT/sweep-out}
: "${POCKET_SWEEP_CKPT:?set POCKET_SWEEP_CKPT to the checkpoint directory}"
BINARY=${POCKET_SWEEP_BINARY:-cpp_engine/build-ascend/pocketllm_engine}
DEVICES=${POCKET_SWEEP_DEVICES:-0,1,2,3}
PORT=${POCKET_SWEEP_PORT:-18280}
GOODPUT=${POCKET_SWEEP_GOODPUT:-"ttft:2000 tpot:200 e2el:30000"}

mkdir -p "$SWEEP_DIR"

# ascend_env.sh must be sourced before any ACL binary starts: launched without
# it the engine does not fail, it hangs before aclInit returns.
# shellcheck source=./ascend_env.sh
source "$ROOT/scripts/ascend_env.sh" >/dev/null 2>&1 || true

point() {
    if [ "$#" -lt 8 ]; then
        echo "usage: run_serving_sweep.sh point <tag> <slots> <concurrency> <prompts> <in> <out> <rate> <ctx> [K=V ...]" >&2
        return 2
    fi
    local tag=$1 slots=$2 conc=$3 np=$4 inl=$5 outl=$6 rate=$7 ctx=$8
    shift 8
    # Remaining arguments are exported for the run only, so a lever cannot leak
    # into the next point of a sweep.
    local kv
    for kv in "$@"; do
        export "$kv"
    done
    export POCKET_ASCEND_IPC_ALLREDUCE=1
    export POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=1

    cd "$ROOT" || return 1
    python "$ROOT/scripts/_serving_metrics_scrape.py" "$PORT" "$SWEEP_DIR/$tag.metrics" -- \
        python "$ROOT/scripts/bench_serving.py" \
        --ckpt "$POCKET_SWEEP_CKPT" \
        --binary "$BINARY" \
        --devices "$DEVICES" --device-style ascend --port "$PORT" \
        --endpoint /v1/chat/completions \
        --random-input-len "$inl" --random-output-len "$outl" \
        --num-prompts "$np" --request-rate "$rate" --max-concurrency "$conc" \
        --num-warmups 0 \
        --max-batch-size "$slots" --max-context "$ctx" \
        --prefill-token-budget "${PFB:-4096}" \
        --server-drain-seconds "${POCKET_SWEEP_DRAIN:-1.0}" \
        --goodput $GOODPUT \
        --log-dir "$SWEEP_DIR/logs-$tag" --json-out "$SWEEP_DIR/$tag.json" \
        > "$SWEEP_DIR/$tag.out" 2>&1
    local rc=$?
    echo "exit=$rc tag=$tag"
    for kv in "$@"; do
        unset "${kv%%=*}"
    done
    return $rc
}

# One wave of prompts per width, slots == prompts, so the batch drains as the
# wave finishes and the row count is the only thing that changes. The output cap
# is 512 and every generation stops at EOS well below it, so no row is truncated.
ladder() {
    point L1      1   1   1 512 512 inf 2048
    point L4      4   4   4 512 512 inf 2048
    point L8      8   8   8 512 512 inf 2048
    point L16    16  16  16 512 512 inf 2048
    point L32    32  32  32 512 512 inf 2048
    point L48    48  48  48 512 512 inf 2048
    point L64    64  64  64 512 512 inf 2048
    point L96    96  96  96 512 512 inf 2048
    point L112  112 112 112 512 512 inf 2048
    # More prompts than slots, so the batch is refilled as it drains instead of
    # being allowed to empty. 16x64 lowers TTFT at flat throughput; 48x192 does
    # not, because a prefill is then always in front of the next decode.
    point L16x64   16  16  64 512 512 inf 2048
    point L48x192  48  48 192 512 512 inf 2048
}

# The concurrency ceiling. 120 and 128 kill a rank in device_malloc at ctx 2048
# while rank 0 still reaches "listening" and keeps admitting requests, so the
# client's only symptom is its own request timeout. L128c1024 holds the same
# width at half the context, which is what separates the pool from the flag.
limit() {
    point L120      120 120 120 512 512 inf 2048
    point L128      128 128 128 512 512 inf 2048
    point L128c1024 128 128 128 512 512 inf 1024
}

# Interleaved A/B for QWEN_ASCEND_REPLICATE_ROWS at concurrency 1, alternating
# control and lever so host drift lands in both arms. One pair is not a series.
ab() {
    local i
    for i in 1 2 3; do
        point "ctl1_r$i" 1 1 1 512 512 inf 8192
        point "rep1_r$i" 1 1 1 512 512 inf 8192 QWEN_ASCEND_REPLICATE_ROWS=16
    done
}

case "${1:-}" in
    point)  shift; point "$@" ;;
    ladder) ladder ;;
    limit)  limit ;;
    ab)     ab ;;
    *)      sed -nE 's/^# ?//p' "$0" | sed -n '1,40p'; exit 2 ;;
esac
