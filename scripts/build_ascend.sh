#!/usr/bin/env bash
# Configure, build and test the Ascend backend.
#
#   scripts/build_ascend.sh            # configure + build + test
#   scripts/build_ascend.sh build      # configure + build only
#
# This exists to hide one environment quirk that fails confusingly: without CANN's
# set_env.sh an ACL binary hangs before aclInit returns and prints nothing at all.
# It looks like broken hardware; it is a missing ASCEND_OPP_PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${REPO_ROOT}/cpp_engine"
BUILD_DIR="${ENGINE_DIR}/build-ascend"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/ascend_env.sh"

# The nested AscendC ExternalProject builds select a GCC 12 toolchain while this
# image provides GCC 11's C++ headers. Environment inheritance is the reliable
# way to make every nested compiler invocation see the installed headers.
GCC_CXX_INCLUDE="/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11"
export CPLUS_INCLUDE_PATH="${GCC_CXX_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

# The Python extension is what gives this backend an HTTP path at all: pocketllm's
# cpp runtime loads pocketllm_cpp, and after the C++ binary's own server was removed
# `pocketllm serve` is the only front end left. It needs an interpreter with
# pybind11, which `python3` on PATH is not here -- that is the base conda
# interpreter -- so one is named explicitly with POCKETLLM_PYTHON.
PYTHON_BIN="${POCKETLLM_PYTHON:-python3}"
BUILD_PYTHON=0
if "${PYTHON_BIN}" -c 'import pybind11' >/dev/null 2>&1; then
    PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
    PYTHON_CMAKE_ARGS=(
        -DPOCKET_BUILD_PYTHON=ON
        "-DPython3_EXECUTABLE=$(command -v "${PYTHON_BIN}")"
        "-Dpybind11_DIR=${PYBIND11_CMAKE_DIR}"
    )
    BUILD_PYTHON=1
else
    # Stated rather than omitted: the build directory is reused between runs, so a
    # stale cached ON would otherwise survive an interpreter that cannot build it.
    echo "build_ascend: ${PYTHON_BIN} has no pybind11, skipping pocketllm_cpp" >&2
    echo "build_ascend: name one that has it with POCKETLLM_PYTHON=/path/to/python" >&2
    PYTHON_CMAKE_ARGS=(-DPOCKET_BUILD_PYTHON=OFF)
fi

cmake -S "${ENGINE_DIR}" -B "${BUILD_DIR}" \
    -DPOCKET_BACKEND=ascend \
    -DASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}" \
    "${PYTHON_CMAKE_ARGS[@]}"

CMAKE_TARGETS=(
    check_layering
    test_device_runtime
    test_qwen_config
    test_qwen_bf16_checkpoint
    test_ipc_allreduce_envelope
    test_ptq1_0_decode
    test_qwen_hadamard_unfold
    test_qwen_ascend_norm_gamma
    test_qwen_ascend_ops
    test_qwen_ascend_group_b
    pocketllm_engine
    test_tp_comm_smoke
)
if [ "${BUILD_PYTHON}" -eq 1 ]; then
    CMAKE_TARGETS+=(pocketllm_cpp)
fi

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target "${CMAKE_TARGETS[@]}"

# `import pocketllm_cpp` resolves only if the module is on sys.path, and the build
# tree is not. Copy it where the named interpreter looks, as the manual build in
# docs/guides/pocketllm_api.md does, so `pocketllm serve` finds it afterwards.
if [ "${BUILD_PYTHON}" -eq 1 ]; then
    purelib="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    module="$(find "${BUILD_DIR}/python" -name 'pocketllm_cpp*.so' -type f | head -1)"
    if [ -z "${module}" ]; then
        echo "build_ascend: pocketllm_cpp built but no module under ${BUILD_DIR}/python" >&2
        exit 1
    fi
    cp "${module}" "${purelib}/"
    echo "build_ascend: installed $(basename "${module}") into ${purelib}"
fi

if [ "${1:-test}" = "build" ]; then
    echo "build_ascend: build only, skipping tests"
    exit 0
fi

status=0

# Single-process tests. Binaries land in different directories depending on
# whether the target sets RUNTIME_OUTPUT_DIRECTORY, so search rather than assume.
for name in test_device_runtime test_qwen_config test_qwen_bf16_checkpoint \
            test_ipc_allreduce_envelope test_ptq1_0_decode \
            test_qwen_hadamard_unfold \
            test_qwen_ascend_norm_gamma \
            test_qwen_ascend_ops test_qwen_ascend_group_b; do
    binary="$(find "${BUILD_DIR}" -name "${name}" -type f -perm -u+x | head -1)"
    if [ -z "${binary}" ]; then
        echo "build_ascend: ${name} not built"
        status=1
        continue
    fi
    if ! "${binary}"; then
        echo "build_ascend: ${name} FAILED"
        status=1
    fi
done

# Collectives need one process per rank: HcclCommInitAll (single process, several
# devices) is unusable on this CANN release, so there is no in-process form to
# fall back to. 4 ranks is the tensor-parallel width Qwen3.8-27B requires.
smoke="$(find "${BUILD_DIR}" -name test_tp_comm_smoke -type f -perm -u+x | head -1)"
if [ -n "${smoke}" ]; then
    id_path="$(mktemp -u /tmp/pocket_tp_id.XXXXXX)"
    # Rank 0 publishes the rendezvous id here; a stale file would make every rank
    # wait on peers from a previous run, so start from a path that cannot exist.
    rm -f "${id_path}"
    pids=()
    for rank in 0 1 2 3; do
        "${smoke}" --world 4 --rank "${rank}" --device "${rank}" \
            --id-path "${id_path}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "build_ascend: test_tp_comm_smoke FAILED"
            status=1
        fi
    done
    rm -f "${id_path}"
else
    echo "build_ascend: test_tp_comm_smoke not built (no libhccl?)"
    status=1
fi

if [ "${status}" -eq 0 ]; then
    echo "build_ascend: all tests passed"
fi
exit "${status}"
