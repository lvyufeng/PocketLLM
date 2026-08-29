#!/usr/bin/env bash
# Source this before building or running anything that links the Ascend backend.
#
#   source scripts/ascend_env.sh
#
# Why this exists rather than just setting LD_LIBRARY_PATH: an ACL binary launched
# without CANN's own environment does not fail, it *hangs* before aclInit even
# returns, producing zero output. LD_LIBRARY_PATH alone is not enough because the
# runtime also needs ASCEND_OPP_PATH and friends, which only set_env.sh defines.
#
# Safe to source more than once.

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}"

if [ ! -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
    echo "ascend_env: no set_env.sh under ${ASCEND_TOOLKIT_HOME}" >&2
    echo "ascend_env: set ASCEND_TOOLKIT_HOME to the CANN install root" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

echo "ascend_env: CANN ${ASCEND_TOOLKIT_HOME}"
echo "ascend_env: soc $(npu-smi info -l 2>/dev/null | grep -m1 -i 'Chip Count' || echo 'npu-smi unavailable')"
