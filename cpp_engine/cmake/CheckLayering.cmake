# Enforces the core/engine/backends layering.
#
# The rule: shared layers must not depend on a vendor SDK. Public headers under
# include/ and the device-agnostic sources under core/ have to compile for any
# accelerator, so a <cuda_runtime.h> or <acl/acl.h> reaching them silently ties
# the shared code to one vendor. Backends are allowed, and expected, to include
# whatever their SDK needs.
#
# backends/api/ is checked too, and is the strictest case: it declares the
# contract both vendors implement, so a vendor type reaching it would defeat the
# whole seam.
#
# engine/ is deliberately NOT checked yet. It still calls vendor runtime APIs
# directly; routing that through backends/api is in progress, file by file.
#
# Run with: cmake --build <dir> --target check_layering

if(NOT DEFINED POCKET_ENGINE_DIR)
    message(FATAL_ERROR "POCKET_ENGINE_DIR must be set")
endif()

set(vendor_patterns
    "cuda_runtime"
    "cuda_fp16"
    "cuda_bf16"
    "cuda_fp8"
    "curand"
    "cublas"
    "nccl.h"
    "nvToolsExt"
    "nvtx3"
    "acl/acl"
    "hccl"
    "aclnn"
)

set(violations "")

file(GLOB_RECURSE guarded_files
    "${POCKET_ENGINE_DIR}/include/*.hpp"
    "${POCKET_ENGINE_DIR}/include/*.h"
    "${POCKET_ENGINE_DIR}/core/*.cpp"
    "${POCKET_ENGINE_DIR}/backends/api/*.hpp"
)

foreach(file ${guarded_files})
    file(STRINGS "${file}" include_lines REGEX "^[ \t]*#[ \t]*include")
    foreach(line ${include_lines})
        foreach(pattern ${vendor_patterns})
            string(FIND "${line}" "${pattern}" found)
            if(NOT found EQUAL -1)
                file(RELATIVE_PATH rel "${POCKET_ENGINE_DIR}" "${file}")
                list(APPEND violations "${rel}: ${line}")
            endif()
        endforeach()
    endforeach()
endforeach()

if(violations)
    list(REMOVE_DUPLICATES violations)
    string(REPLACE ";" "\n  " pretty "${violations}")
    message(FATAL_ERROR
        "Vendor SDK headers leaked into a device-agnostic layer:\n  ${pretty}\n"
        "Move the vendor dependency into backends/<vendor>/ and expose an "
        "opaque handle instead (see include/qwen_sampler.hpp for the pattern).")
endif()

list(LENGTH guarded_files checked)
message(STATUS "check_layering: ${checked} files clean of vendor SDK includes")
