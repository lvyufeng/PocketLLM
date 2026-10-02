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
  kernel/cpu/kernels.cpp
  model/qwen3.cpp
)

target_include_directories(pocketllm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pocketllm PRIVATE -Wall -Wextra -Werror)
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
  kernel/cpu/kernels.cpp
  model/qwen3.cpp
)
target_include_directories(pocketllm-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pocketllm-run PRIVATE -Wall -Wextra -Werror)
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