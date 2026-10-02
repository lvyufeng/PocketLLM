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
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  model/qwen3.cpp
  ${POCKETLLM_CUDA_SOURCES}
)

if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_include_directories(pocketllm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # CXX only: `-Werror` is spelled differently for nvcc, and the CUDA file is
  # built by the same target.  The host half keeps warnings-as-errors.
  target_compile_options(pocketllm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>)
endif()

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
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  model/qwen3.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-run PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-run PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_include_directories(pocketllm-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # CXX only for the same reason as the library above: this target carries the
  # CUDA translation unit.
  target_compile_options(pocketllm-run PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>)
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
  kernel/cpu/backend.cpp
  kernel/registry.cpp
  ${POCKETLLM_CUDA_SOURCES}
)
if(CMAKE_CUDA_COMPILER AND POCKETLLM_CUDA_SOURCES)
  target_compile_definitions(pocketllm-opcheck PRIVATE POCKETLLM_WITH_CUDA=1)
  set_target_properties(pocketllm-opcheck PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
target_include_directories(pocketllm-opcheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pocketllm-opcheck PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>)
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