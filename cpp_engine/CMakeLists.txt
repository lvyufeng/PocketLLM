cmake_minimum_required(VERSION 3.18)

project(dsv4_cpp_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

set(NCCL_ROOT "" CACHE PATH "Optional NCCL root directory")
set(DSV4_HAVE_NCCL OFF)
if(NCCL_ROOT)
    find_path(NCCL_INCLUDE_DIR nccl.h PATHS "${NCCL_ROOT}/include" NO_DEFAULT_PATH)
    find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 PATHS "${NCCL_ROOT}/lib" NO_DEFAULT_PATH)
else()
    find_path(NCCL_INCLUDE_DIR nccl.h)
    find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2)
endif()
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(DSV4_HAVE_NCCL ON)
    message(STATUS "NCCL found: ${NCCL_LIBRARY}")
else()
    message(STATUS "NCCL not found; distributed C++ TP helpers disabled")
endif()

add_library(dsv4_cpp_core STATIC
    src/json_lite.cpp
    src/gguf_reader.cpp
    src/safetensors_reader.cpp
    src/safetensors_model.cpp
    src/weight_source.cpp
    src/model_config.cpp
    src/qwen_config.cpp
    src/qwen_weights.cpp
    src/qwen_dspark.cpp
    src/qwen_dflash2.cpp
    src/qwen_engine.cpp
    src/tokenizer.cpp
    src/tensor.cpp
    src/tp_comm.cpp
    src/dsv4_engine.cpp
    src/dspark_engine.cpp
    src/sampler.cpp
    src/cmd_channel.cpp
    src/python_sidecar.cpp
    src/openai_server.cpp
    third_party/httplib.cpp
    cuda/cuda_ops.cu
    cuda/bf16_ops.cu
    cuda/quant_gemm.cu
    cuda/fp4_ops.cu
    cuda/fp8_ops.cu
    cuda/qwen_fp8_ops.cu
    cuda/qwen_attention_ops.cu
    cuda/qwen_half_ops.cu
    cuda/qwen_gqa_optimized.cu
    cuda/qwen_gdn_flashqla.cu
    cuda/qwen_turboquant_ops.cu
    cuda/qwen_dspark_ops.cu
    cuda/qwen_dflash2_ops.cu
    cuda/qwen_sampler.cu
    cuda/q2_ops.cu
    cuda/iq1_ops.cu
    cuda/rmsnorm_rope.cu
    cuda/flashmemory_ops.cu
)

target_include_directories(dsv4_cpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(dsv4_cpp_core PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
if(TARGET CUDA::nvtx3)
    target_link_libraries(dsv4_cpp_core PUBLIC CUDA::nvtx3)
elseif(TARGET CUDA::nvToolsExt)
    target_link_libraries(dsv4_cpp_core PUBLIC CUDA::nvToolsExt)
else()
    message(FATAL_ERROR "NVTX is required but was not found in the CUDA toolkit")
endif()
if(DSV4_HAVE_NCCL)
    target_include_directories(dsv4_cpp_core PUBLIC ${NCCL_INCLUDE_DIR})
    target_link_libraries(dsv4_cpp_core PUBLIC ${NCCL_LIBRARY})
    target_compile_definitions(dsv4_cpp_core PUBLIC DSV4_HAVE_NCCL=1)
endif()
set_target_properties(dsv4_cpp_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(dsv4_cpp_engine src/main.cpp)
target_link_libraries(dsv4_cpp_engine PRIVATE dsv4_cpp_core)

add_executable(inspect_gguf tools/inspect_gguf.cpp)
target_link_libraries(inspect_gguf PRIVATE dsv4_cpp_core)
set_target_properties(inspect_gguf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(inspect_safetensors tools/inspect_safetensors.cpp)
target_link_libraries(inspect_safetensors PRIVATE dsv4_cpp_core)
set_target_properties(inspect_safetensors PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(minimax_gguf_tokenizer tools/minimax_gguf_tokenizer.cpp)
target_link_libraries(minimax_gguf_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(minimax_gguf_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(test_gguf_reader tests/test_gguf_reader.cpp)
target_link_libraries(test_gguf_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_iq1m_reader tests/test_gguf_iq1m_reader.cpp)
target_link_libraries(test_gguf_iq1m_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_iq1m_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_reader tests/test_safetensors_reader.cpp)
target_link_libraries(test_safetensors_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_flashmemory_scorer tests/test_flashmemory_scorer.cpp)
target_link_libraries(test_flashmemory_scorer PRIVATE dsv4_cpp_core)
set_target_properties(test_flashmemory_scorer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_model tests/test_safetensors_model.cpp)
target_link_libraries(test_safetensors_model PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_weight_source tests/test_weight_source.cpp)
target_link_libraries(test_weight_source PRIVATE dsv4_cpp_core)
set_target_properties(test_weight_source PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_weight_load tests/test_dspark_weight_load.cpp)
target_link_libraries(test_dspark_weight_load PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_weight_load PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_perfect_draft tests/test_perfect_draft.cpp)
target_link_libraries(test_perfect_draft PRIVATE dsv4_cpp_core)
set_target_properties(test_perfect_draft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_fp4_determinism tests/test_moe_fp4_determinism.cpp)
target_link_libraries(test_moe_fp4_determinism PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_fp4_determinism PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_attention_parity tests/test_dspark_attention_parity.cpp)
target_link_libraries(test_dspark_attention_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_attention_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_moe_parity tests/test_dspark_moe_parity.cpp)
target_link_libraries(test_dspark_moe_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_moe_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_draft tests/test_dspark_draft.cpp)
target_link_libraries(test_dspark_draft PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_draft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_verify_overwrite_safety tests/test_verify_overwrite_safety.cpp)
target_link_libraries(test_verify_overwrite_safety PRIVATE dsv4_cpp_core)
set_target_properties(test_verify_overwrite_safety PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_speculative_scheduler tests/test_speculative_scheduler.cpp)
target_link_libraries(test_speculative_scheduler PRIVATE dsv4_cpp_core)
set_target_properties(test_speculative_scheduler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_dspark_decode tests/bench_dspark_decode.cpp)
target_link_libraries(bench_dspark_decode PRIVATE dsv4_cpp_core)
set_target_properties(bench_dspark_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_batch_verify_transaction tests/test_batch_verify_transaction.cpp)
target_link_libraries(test_batch_verify_transaction PRIVATE dsv4_cpp_core)
set_target_properties(test_batch_verify_transaction PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_tp_consistency tests/test_dspark_tp_consistency.cpp)
target_link_libraries(test_dspark_tp_consistency PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_tp_consistency PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_draft_sensitivity tests/test_dspark_draft_sensitivity.cpp)
target_link_libraries(test_dspark_draft_sensitivity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_draft_sensitivity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_hidden_capture tests/test_dspark_hidden_capture.cpp)
target_link_libraries(test_dspark_hidden_capture PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_hidden_capture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_head_parity tests/test_dspark_head_parity.cpp)
target_link_libraries(test_dspark_head_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_head_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_q2_matvec tests/test_q2_matvec.cpp)
target_link_libraries(test_q2_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_q2_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_dense_smoke tests/test_gguf_dense_smoke.cpp)
target_link_libraries(test_gguf_dense_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_dense_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_min_layer_smoke tests/test_gguf_min_layer_smoke.cpp)
target_link_libraries(test_gguf_min_layer_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_min_layer_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_norm_wqa tests/test_gguf_attn_norm_wqa.cpp)
target_link_libraries(test_gguf_attn_norm_wqa PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_norm_wqa PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_q_path tests/test_gguf_attn_q_path.cpp)
target_link_libraries(test_gguf_attn_q_path PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_q_path PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_kv_path tests/test_gguf_attn_kv_path.cpp)
target_link_libraries(test_gguf_attn_kv_path PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_kv_path PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_full tests/test_gguf_attn_full.cpp)
target_link_libraries(test_gguf_attn_full PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_shared_expert tests/test_gguf_shared_expert.cpp)
target_link_libraries(test_gguf_shared_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_shared_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_routed_expert tests/test_gguf_routed_expert.cpp)
target_link_libraries(test_gguf_routed_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_routed_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_routed_moe tests/test_gguf_routed_moe.cpp)
target_link_libraries(test_gguf_routed_moe PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_routed_moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_layer0_full tests/test_gguf_layer0_full.cpp)
target_link_libraries(test_gguf_layer0_full PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_layer0_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_full_forward tests/test_gguf_full_forward.cpp)
target_link_libraries(test_gguf_full_forward PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_full_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_generate tests/test_gguf_generate.cpp)
target_link_libraries(test_gguf_generate PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(probe_host_register tests/probe_host_register.cpp)
target_link_libraries(probe_host_register PRIVATE dsv4_cpp_core)
set_target_properties(probe_host_register PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_quant_gemm tests/test_quant_gemm.cpp)
target_link_libraries(test_quant_gemm PRIVATE dsv4_cpp_core)
set_target_properties(test_quant_gemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp4_matvec tests/test_fp4_matvec.cpp)
target_link_libraries(test_fp4_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp4_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp8_matvec tests/test_fp8_matvec.cpp)
target_link_libraries(test_fp8_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp8_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE dsv4_cpp_core)
set_target_properties(test_rmsnorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_attention_projection tests/test_attention_projection.cpp)
target_link_libraries(test_attention_projection PRIVATE dsv4_cpp_core)
set_target_properties(test_attention_projection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_single_expert tests/test_moe_single_expert.cpp)
target_link_libraries(test_moe_single_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_single_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_fp4 tests/test_moe_grouped_fp4.cpp)
target_link_libraries(test_moe_grouped_fp4 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_fp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_multi_token_fp4 tests/test_moe_multi_token_fp4.cpp)
target_link_libraries(test_moe_multi_token_fp4 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_multi_token_fp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_continuation_attention tests/test_continuation_attention.cpp)
target_link_libraries(test_continuation_attention PRIVATE dsv4_cpp_core)
set_target_properties(test_continuation_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_iq1 tests/test_moe_grouped_iq1.cpp)
target_link_libraries(test_moe_grouped_iq1 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_iq1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_q2 tests/test_moe_grouped_q2.cpp)
target_link_libraries(test_moe_grouped_q2 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_q2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_embedding_head tests/test_embedding_head.cpp)
target_link_libraries(test_embedding_head PRIVATE dsv4_cpp_core)
set_target_properties(test_embedding_head PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(test_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_tokenizer tests/test_gguf_tokenizer.cpp)
target_link_libraries(test_gguf_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_min_layer_smoke tests/test_min_layer_smoke.cpp)
target_link_libraries(test_min_layer_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_min_layer_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_nccl_smoke tests/test_nccl_smoke.cpp)
target_link_libraries(test_nccl_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_nccl_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_config tests/test_qwen_config.cpp)
target_link_libraries(test_qwen_config PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_config PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dspark tests/test_qwen_dspark.cpp)
target_link_libraries(test_qwen_dspark PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dspark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2 tests/test_qwen_dflash2.cpp)
target_link_libraries(test_qwen_dflash2 PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2_parity tests/test_qwen_dflash2_parity.cpp)
target_link_libraries(test_qwen_dflash2_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dspark_ops tests/test_qwen_dspark_ops.cpp)
target_link_libraries(test_qwen_dspark_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dspark_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2_ops tests/test_qwen_dflash2_ops.cpp)
target_link_libraries(test_qwen_dflash2_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_sampler tests/test_qwen_sampler.cpp)
target_link_libraries(test_qwen_sampler PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_sampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_fp8_online tests/test_qwen_fp8_online.cpp)
target_link_libraries(test_qwen_fp8_online PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_fp8_online PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_weights tests/test_qwen_weights.cpp)
target_link_libraries(test_qwen_weights PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_weights PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_engine tests/test_qwen_engine.cpp)
target_link_libraries(test_qwen_engine PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_delta_rule tests/test_qwen_delta_rule.cpp)
target_link_libraries(test_qwen_delta_rule PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_delta_rule PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_gqa_attention tests/test_qwen_gqa_attention.cpp)
target_link_libraries(test_qwen_gqa_attention PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_gqa_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_half_ops tests/test_qwen_half_ops.cpp)
target_link_libraries(test_qwen_half_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_half_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_conv_tail tests/test_qwen_conv_tail.cpp)
target_link_libraries(test_qwen_conv_tail PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_conv_tail PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_kernels tests/bench_qwen_fp8_kernels.cpp)
target_link_libraries(bench_qwen_fp8_kernels PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_draft_lm_head tests/bench_qwen_draft_lm_head.cpp)
target_link_libraries(bench_qwen_draft_lm_head PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_draft_lm_head PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_verify_batch tests/bench_qwen_verify_batch.cpp)
target_link_libraries(bench_qwen_verify_batch PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_verify_batch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_delta_f16 tests/bench_qwen_delta_f16.cpp)
target_link_libraries(bench_qwen_delta_f16 PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_delta_f16 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_gdn_flashqla tests/test_qwen_gdn_flashqla.cpp)
target_link_libraries(test_qwen_gdn_flashqla PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_gdn_flashqla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_turboquant_k8v4 tests/test_qwen_turboquant_k8v4.cpp)
target_link_libraries(test_qwen_turboquant_k8v4 PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_turboquant_k8v4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_transaction_copy tests/bench_qwen_transaction_copy.cpp)
target_link_libraries(bench_qwen_transaction_copy PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_transaction_copy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_tp_allreduce tests/bench_qwen_tp_allreduce.cpp)
target_link_libraries(bench_qwen_tp_allreduce PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_tp_allreduce PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_persistent_engine tests/test_persistent_engine.cpp)
target_link_libraries(test_persistent_engine PRIVATE dsv4_cpp_core)
set_target_properties(test_persistent_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_batch_crossover tests/bench_qwen_fp8_batch_crossover.cpp)
target_link_libraries(bench_qwen_fp8_batch_crossover PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_batch_crossover PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_dequant_overhead tests/bench_qwen_fp8_dequant_overhead.cpp)
target_link_libraries(bench_qwen_fp8_dequant_overhead PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_dequant_overhead PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_gqa_prefill tests/bench_qwen_gqa_prefill.cpp)
target_link_libraries(bench_qwen_gqa_prefill PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_gqa_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_gqa_verify tests/bench_qwen_gqa_verify.cpp)
target_link_libraries(bench_qwen_gqa_verify PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_gqa_verify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_dflash2_topk tests/bench_qwen_dflash2_topk.cpp)
target_link_libraries(bench_qwen_dflash2_topk PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_dflash2_topk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
