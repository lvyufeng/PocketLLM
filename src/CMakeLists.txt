# The PocketLLM engine.
#
# Built out of tree -- `cmake -B build -S src` -- so the Python package stays
# free of any build artifact and `tests/test_package_boundaries.py` keeps
# passing: the engine is a separate native library the host shell loads, not a
# Python extension that `pip install` compiles.
#
# Warnings are errors here.  This is C++ written to be called from another
# language over a hand-kept ABI, which is the setting where an implicit
# conversion or an unused parameter is worth more than usual to catch.

cmake_minimum_required(VERSION 3.16)

project(pocketllm_engine LANGUAGES CXX VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent because the Rust, Go, Swift and Java hosts all want to
# dlopen this, and a shared object built from non-PIC code fails at link time
# only for them.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The CUDA backend is opt-in by *availability*, not by a flag: if a toolkit is
# found it is compiled and `pocketllm_open(..., "cuda")` works, and if it is not
# the name is refused with a message that says so. A build flag would make the
# presence of a kernel a decision somebody has to remember to make on every
# machine, which is how a release ends up without the backend it was tested
# with.
#
# `CUDAToolkit_ROOT` is not set here on purpose. On this project's x86 machine
# `CUDA_HOME` points at 12.4 while `nvcc` on PATH is 13.0, so a hard-coded
# preference would be wrong on the machine whose notes say to be careful -- CMake
# searches the standard locations and finds the 13.0 the toolchain actually is.
# Override it with `-DCUDAToolkit_ROOT=...` when a host needs another.
option(POCKETLLM_CUDA "Build the CUDA backend" ON)
set(POCKETLLM_CUDA_ARCHITECTURES 75 CACHE STRING
    "CUDA architectures to compile for; sm_75 is the project's 2080 Ti target")

# `-march=native` is the right choice for a build on the machine that will run
# it, which is this project's x86 host.  It is the wrong choice for every
# cross-compilation: "native" then means the *build* host, so the flags describe
# a CPU the output will never run on -- at best it compiles something the target
# cannot execute, at worst it fails outright on a feature the target lacks.
#
# The two aarch64 boards this tree targets are not reachable from the x86 build
# host, so both routes exist: a board builds on itself with this ON, and a
# toolchain file cross-compiles with it OFF.  `POCKETLLM_MARCH` overrides both,
# for a cross build that wants a specific level rather than the baseline.
option(POCKETLLM_NATIVE_ARCH
    "Compile for the build host's own instruction set (-march=native)" ON)
set(POCKETLLM_MARCH "" CACHE STRING
    "Explicit -march value; takes precedence over POCKETLLM_NATIVE_ARCH")

set(POCKETLLM_MARCH_FLAG "")
if(POCKETLLM_MARCH)
  set(POCKETLLM_MARCH_FLAG "-march=${POCKETLLM_MARCH}")
  message(STATUS "pocketllm: building with -march=${POCKETLLM_MARCH}")
elseif(POCKETLLM_NATIVE_ARCH)
  set(POCKETLLM_MARCH_FLAG "-march=native")
  message(STATUS "pocketllm: building with -march=native")
else()
  message(STATUS "pocketllm: building for the compiler's baseline architecture")
endif()

set(POCKETLLM_CUDA_SOURCES)
if(POCKETLLM_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${POCKETLLM_CUDA_ARCHITECTURES})
    # C++17 in the .cu files must match the host translation units: the backend
    # implements a virtual interface declared in a header they share.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(POCKETLLM_CUDA_SOURCES kernel/cuda/backend.cu)
    message(STATUS "pocketllm: CUDA backend enabled, architectures ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "pocketllm: no CUDA toolchain found, building the CPU backend only")
  endif()
endif()

# The Ascend backend, opt-in and default OFF.
#
# Unlike CUDA -- which is on by *availability*, because a machine with a toolkit
# should get the backend -- this one is a flag. Two reasons. It needs CANN *and*
# a per-user custom-op package that is not part of CANN, so "the toolkit is
# present" is not the same as "this can build"; and it is 310B-only, so an
# x86 host with a stray CANN install must not pick it up. A flag is honest about
# both. It is additionally guarded on aarch64: the backend file includes acl/acl.h
# and links aarch64 CANN libraries, and the x86 host this tree is developed on
# must not be broken by the option existing.
option(POCKETLLM_ASCEND "Build the Ascend 310B backend (aarch64 + CANN only)" OFF)
set(POCKETLLM_ASCEND_HOME "" CACHE STRING
    "CANN toolkit root; defaults to \$ASCEND_HOME_PATH, \$ASCEND_TOOLKIT_HOME, then /usr/local/Ascend/ascend-toolkit/latest")
set(POCKETLLM_CUSTOM_OPP_DIR "" CACHE STRING
    "Custom-op vendor dir (the one holding op_api/); defaults to \$ASCEND_CUSTOM_OPP_PATH, then \$HOME/Ascend/custom_opp/vendors/customize")

set(POCKETLLM_ASCEND_SOURCES)
if(POCKETLLM_ASCEND)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR
      "pocketllm: POCKETLLM_ASCEND=ON needs an aarch64 host (this is ${CMAKE_SYSTEM_PROCESSOR})")
  endif()

  if(POCKETLLM_ASCEND_HOME)
    set(_ascend_home "${POCKETLLM_ASCEND_HOME}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(_ascend_home "$ENV{ASCEND_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(_ascend_home "$ENV{ASCEND_TOOLKIT_HOME}")
  else()
    set(_ascend_home "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  if(NOT EXISTS "${_ascend_home}/include/acl/acl.h")
    message(FATAL_ERROR
      "pocketllm: CANN headers not found under ${_ascend_home}; set POCKETLLM_ASCEND_HOME")
  endif()

  if(POCKETLLM_CUSTOM_OPP_DIR)
    set(_custom_opp "${POCKETLLM_CUSTOM_OPP_DIR}")
  elseif(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
    set(_custom_opp "$ENV{ASCEND_CUSTOM_OPP_PATH}")
  else()
    set(_custom_opp "$ENV{HOME}/Ascend/custom_opp/vendors/customize")
  endif()
  # `ASCEND_CUSTOM_OPP_PATH` is a colon-separated list, and the installer's
  # generated `set_env.bash` appends its own value to an (often empty) inherited
  # one -- so what arrives is ".../vendors/customize:". Take the first component.
  string(REGEX REPLACE ":.*$" "" _custom_opp "${_custom_opp}")
  if(NOT EXISTS "${_custom_opp}/op_api/include/aclnn_matmul_w4a16_custom.h")
    message(FATAL_ERROR
      "pocketllm: custom ops not found under ${_custom_opp}; install the custom_opp "
      "package and set POCKETLLM_CUSTOM_OPP_DIR (see docs/architecture/ascend_310b.md)")
  endif()

  find_library(POCKETLLM_ASCENDCL_LIB ascendcl HINTS "${_ascend_home}/lib64" REQUIRED)
  find_library(POCKETLLM_NNOPBASE_LIB nnopbase HINTS "${_ascend_home}/lib64" REQUIRED)
  find_library(POCKETLLM_OPAPI_LIB opapi HINTS "${_ascend_home}/lib64" REQUIRED)
  find_library(POCKETLLM_CUST_OPAPI_LIB cust_opapi HINTS "${_custom_opp}/op_api/lib" REQUIRED)

  set(POCKETLLM_ASCEND_SOURCES kernel/ascend/backend.cpp)
  message(STATUS "pocketllm: Ascend backend enabled (CANN ${_ascend_home}, custom ops ${_custom_opp})")

  # The include roots, the four libraries, and the `POCKETLLM_WITH_ASCEND` define,
  # carried as one INTERFACE target so an option added here reaches every kernel
  # target rather than being copied into each -- the same reason
  # `pocketllm-kernel-options` exists.
  add_library(pocketllm-ascend-options INTERFACE)
  target_include_directories(pocketllm-ascend-options INTERFACE
    "${_ascend_home}/include"
    "${_custom_opp}/op_api/include")
  target_link_directories(pocketllm-ascend-options INTERFACE
    "${_ascend_home}/lib64"
    "${_custom_opp}/op_api/lib")
  target_link_libraries(pocketllm-ascend-options INTERFACE
    ${POCKETLLM_ASCENDCL_LIB}
    ${POCKETLLM_NNOPBASE_LIB}
    ${POCKETLLM_OPAPI_LIB}
    ${POCKETLLM_CUST_OPAPI_LIB})
  target_compile_definitions(pocketllm-ascend-options INTERFACE POCKETLLM_WITH_ASCEND=1)

  # Add the backend to a kernel target: its source, the includes, the libraries
  # and the define. A function rather than five copies of the same three lines,
  # which is how a flag gets added to four of the five.
  function(pocketllm_add_ascend target)
    target_sources(${target} PRIVATE ${POCKETLLM_ASCEND_SOURCES})
    target_link_libraries(${target} PRIVATE pocketllm-ascend-options)
  endfunction()
endif()

# Warning-as-errors and the kernel options, stated once.
#
# The three targets that link `kernel/` -- the library, `pocketllm-run` and
# `pocketllm-opcheck` -- each carried their own copy of these options, which is
# how a flag gets added to two of the three.  An INTERFACE library carries them
# instead, so an option added here reaches every kernel target and neither of the
# two (`pocketllm-gguf-dump`, `pocketllm-tokenize`) that compiles no kernel.
#
# `-march=native` is a build-host choice and is deliberately allowed to change
# the last bits of a reduction: it enables AVX2 and FMA, and GCC's default
# `-ffp-contract=fast` then fuses `a*b+c` into `vfmadd` where the SSE2 baseline
# could not.  `-ffast-math` is *not* here -- it would add full reassociation and
# `-ffinite-math-only`, which the shifted softmax depends on not having.  Which
# `-march` reaches the compiler is decided above, by `POCKETLLM_NATIVE_ARCH` and
# `POCKETLLM_MARCH`; the flags below only apply it.
add_library(pocketllm-kernel-options INTERFACE)
find_package(Threads REQUIRED)
target_link_libraries(pocketllm-kernel-options INTERFACE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # CXX only: `-Werror` is spelled differently for nvcc, and the CUDA file is
  # built by the same targets.  The host keeps warnings-as-errors, and the
  # optimisation flags are host flags -- nvcc does its own.
  target_compile_options(pocketllm-kernel-options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -O3 ${POCKETLLM_MARCH_FLAG}>)
endif()
# The include root, so a kernel target can `#include "kernel/parallel.h"` without
# repeating the directory.
target_include_directories(pocketllm-kernel-options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(pocketllm SHARED
  c_api.cpp
  abi/spec.cpp
  gguf/reader.cpp
  runtime/status.cpp
  runtime/session.cpp
  tokenizer/unicode_data.cpp
  tokenizer/unicode.cpp
  tokenizer/pretokenize.cpp
  tokenizer/bpe.cpp
  kernel/kernels.cpp
  kernel/repack_q4k.cpp
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  model/qwen3.cpp
  ${POCKETLLM_CUDA_SOURCES}
)

if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
if(POCKETLLM_ASCEND_SOURCES)
  pocketllm_add_ascend(pocketllm)
endif()

target_include_directories(pocketllm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pocketllm PRIVATE pocketllm-kernel-options)

# An internal tool, not part of the ABI: it prints a checkpoint's directory as
# JSON so the reader test can diff the C parse against the Python reader's on a
# real file.  It links the reader directly rather than libpocketllm.so, which is
# how the ABI stays the eight functions the header declares.
add_executable(pocketllm-gguf-dump
  tools/gguf_dump.cpp
  abi/spec.cpp
  gguf/reader.cpp
  runtime/status.cpp
)
target_include_directories(pocketllm-gguf-dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pocketllm-gguf-dump PRIVATE -Wall -Wextra -Werror)
endif()

# The tokenizer, as a standalone tool.  Same reason as the dump: the tokenizer
# test wants to run *the C tokenizer* over a corpus and diff the ids, and going
# through libpocketllm.so would mean a session per line.  It also gives a
# `--text` mode so the same binary can be diffed against `llama-tokenize`.
add_executable(pocketllm-tokenize
  tools/tokenize.cpp
  abi/spec.cpp
  gguf/reader.cpp
  runtime/status.cpp
  tokenizer/unicode_data.cpp
  tokenizer/unicode.cpp
  tokenizer/pretokenize.cpp
  tokenizer/bpe.cpp
)
target_include_directories(pocketllm-tokenize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pocketllm-tokenize PRIVATE -Wall -Wextra -Werror)
endif()

# Generate text with the engine, for watching it work and for diffing the first
# tokens against `llama-cli`.  Same shape as the two above: the engine's objects
# linked directly, so the tool can print inside the loop and the ABI stays the
# eight functions the header declares.
add_executable(pocketllm-run
  tools/run.cpp
  abi/spec.cpp
  gguf/reader.cpp
  runtime/status.cpp
  tokenizer/unicode_data.cpp
  tokenizer/unicode.cpp
  tokenizer/pretokenize.cpp
  tokenizer/bpe.cpp
  kernel/kernels.cpp
  kernel/repack_q4k.cpp
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  model/qwen3.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-run PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-run PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_link_libraries(pocketllm-run PRIVATE pocketllm-kernel-options)
if(POCKETLLM_ASCEND_SOURCES)
  pocketllm_add_ascend(pocketllm-run)
endif()

# Run one kernel over inputs a caller supplies, for checking it against the
# Python reference.  It links the kernel objects and both backends, and no model:
# an op is checked in isolation, on shapes the caller writes, which is the only
# way to reach a kernel the graph happens not to exercise with a shape the graph
# never produces.  The caller is `tests/native/test_op_conformance.py`, which
# authors the request and owns the oracle.
add_executable(pocketllm-opcheck
  tools/opcheck.cpp
  runtime/status.cpp
  kernel/kernels.cpp
  kernel/repack_q4k.cpp
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-opcheck PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-opcheck PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_link_libraries(pocketllm-opcheck PRIVATE pocketllm-kernel-options)
if(POCKETLLM_ASCEND_SOURCES)
  pocketllm_add_ascend(pocketllm-opcheck)
endif()

# Drive `MatmulCubeCustom` at a fixed weight shape while varying only m, to tell
# a shape limit (throughput rises with m) from a device limit (it does not).
add_executable(pocketllm-mscale
  tools/mscale.cpp
  runtime/status.cpp
  kernel/kernels.cpp
  kernel/repack_q4k.cpp
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-mscale PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-mscale PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_link_libraries(pocketllm-mscale PRIVATE pocketllm-kernel-options)
if(POCKETLLM_ASCEND_SOURCES)
  pocketllm_add_ascend(pocketllm-mscale)
endif()

# Measure the engine's own throughput, as `llama-bench` measures llama.cpp's.
# Same object list as `pocketllm-run` -- it needs the model and both backends --
# and the same CUDA guard, so a CUDA build links here too.
add_executable(pocketllm-bench
  tools/bench.cpp
  abi/spec.cpp
  gguf/reader.cpp
  runtime/status.cpp
  tokenizer/unicode_data.cpp
  tokenizer/unicode.cpp
  tokenizer/pretokenize.cpp
  tokenizer/bpe.cpp
  kernel/kernels.cpp
  kernel/repack_q4k.cpp
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  model/qwen3.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-bench PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_link_libraries(pocketllm-bench PRIVATE pocketllm-kernel-options)
if(POCKETLLM_ASCEND_SOURCES)
  pocketllm_add_ascend(pocketllm-bench)
endif()

# `POCKETLLM_ABI_VERSION_MAJOR` is a C preprocessor macro in the header, not a
# CMake variable, so the soname major is stated here too.  It is the one number
# that must not drift: the header is the handshake a host checks, and the
# soname is what the loader resolves, so both have to move together.
set(POCKETLLM_ABI_SOVERSION 1)

set_target_properties(pocketllm PROPERTIES
  OUTPUT_NAME pocketllm
  VERSION ${PROJECT_VERSION}
  SOVERSION ${POCKETLLM_ABI_SOVERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)