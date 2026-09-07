#pragma once

namespace qwen_components {

template <typename Runtime>
void Linear::forward(
    Runtime& runtime, const DeviceLinear& linear, const uint16_t* input,
    uint16_t* output, int rows, const char* site) const {
    typename Runtime::PhaseScope scope(&runtime, std::string(rows == 1 ? "pd." : "pr.") + site);
    if (linear.logical_shape.size() != 2) {
        throw std::runtime_error("Qwen linear has invalid logical shape");
    }
    const int output_rows = static_cast<int>(linear.logical_shape[0]);
    const int columns = static_cast<int>(linear.logical_shape[1]);
#ifdef POCKET_BACKEND_ASCEND
    // Only the dense FP16 kind has an Ascend implementation. The quantized
    // kinds are rejected here rather than left to a runtime branch: a
    // reference to their CUDA kernels in this object would make the whole
    // executable require the CUDA toolchain to link.
    if (linear.kind != QwenLinearKind::DenseF16) {
        throw std::runtime_error(
            std::string("Qwen Ascend path is not implemented for ") +
            qwen_linear_kind_name(linear.kind));
    }
    require_launch(qwen_fp16_matmul_rows_f16(
        input, linear.weight.f16_data(), output, rows, output_rows,
        columns, columns, output_rows, columns),
        "FP16 activation/weight projection");
    return;
#else
    if (linear.kind == QwenLinearKind::Fp8Block128) {
        if (rows == 1) {
            require_launch(qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                input, linear.weight.fp8_data(), linear.scale.f16_data(),
                output, output_rows, columns, columns,
                static_cast<int>(linear.scale.shape[1])),
                "FP8 FP16-activation decode projection");
        } else if (rows <= 8 && linear.verify_weight.data != nullptr) {
            require_launch(qwen_fp16_matmul_rows_f16_cublas_cuda(
                input, linear.verify_weight.f16_data(), output, rows,
                output_rows, columns, columns, output_rows, columns),
                "resident FP16 verify projection");
        } else {
            require_launch(qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                input, linear.weight.fp8_data(), linear.scale.f16_data(),
                output, rows, output_rows, columns, columns, output_rows,
                columns, static_cast<int>(linear.scale.shape[1])),
                "FP8 FP16-activation projection");
        }
    } else if (linear.kind == QwenLinearKind::Fp8Channel) {
        require_launch(rows == 1
            ? qwen_fp8_e4m3_channel_matvec_f16_cuda(
                  input, linear.weight.fp8_data(), linear.scale.f16_data(),
                  output, output_rows, columns, columns)
            : qwen_fp8_e4m3_channel_matmul_rows_f16_cuda(
                  input, linear.weight.fp8_data(), linear.scale.f16_data(),
                  output, rows, output_rows, columns, columns, output_rows,
                  columns),
            "channel FP8 FP16-activation projection");
    } else if (linear.kind == QwenLinearKind::NvFp4Group16) {
        const NvFp4Mode mode = runtime.layer_config.nvfp4_mode;
        if (mode == NvFp4Mode::Invalid) {
            throw std::runtime_error(
                "POCKETLLM_QWEN_NVFP4 must be auto, dp4a, wmma, or reference");
        }
        const bool reference = mode == NvFp4Mode::Reference;
        const int blocks_per_row = columns / 64;
        if (!reference) {
            const bool use_wmma = mode == NvFp4Mode::Wmma ||
                (mode == NvFp4Mode::Auto && rows >= 8);
            const bool use_wide = use_wmma &&
                runtime.layer_config.use_nvfp4_wide_n64(rows);
            require_launch(runtime.nvfp4_integer_projection(
                linear, input, output, rows, use_wmma),
                use_wide ? "NVFP4 group-16 Q8 wide-N64 projection" :
                use_wmma ? "NVFP4 group-16 Q8 WMMA projection" :
                           "NVFP4 group-16 Q8 DP4A projection");
        } else {
            require_launch(rows == 1
                ? qwen_nvfp4_group16_matvec_f16_cuda(
                      input, linear.weight.u8_data(), output, output_rows,
                      columns, blocks_per_row, linear.weight_global_factor)
                : qwen_nvfp4_group16_matmul_rows_f16_cuda(
                      input, linear.weight.u8_data(), output, rows,
                      output_rows, columns, columns, output_rows,
                      blocks_per_row, linear.weight_global_factor),
                "NVFP4 group-16 reference projection");
        }
    } else if (linear.kind == QwenLinearKind::DenseF16) {
        // The fused DeltaNet a/b matrix is only 24 rows per TP4 rank. At
        // verify width 8, tensor cores are ~5x faster than the generic
        // warp-per-output-row reduction. Decode keeps its established
        // reduction order, and wider prefill remains on the existing path.
        const bool verify_cublas = rows >= 4 && rows <= 8 &&
            output_rows <= 32 && columns % 8 == 0 &&
            runtime.layer_config.verify_small_fp16_cublas;
        if (verify_cublas) {
            require_launch(qwen_fp16_matmul_rows_f16_cublas_cuda(
                input, linear.weight.f16_data(), output, rows, output_rows,
                columns, columns, output_rows, columns),
                "small FP16 verify cuBLAS projection");
        } else {
            require_launch(qwen_fp16_matmul_rows_f16(
                input, linear.weight.f16_data(), output, rows, output_rows,
                columns, columns, output_rows, columns),
                "FP16 activation/weight projection");
        }
    } else {
        throw std::runtime_error(
            std::string("Qwen linear CUDA path is not implemented for ") +
            qwen_linear_kind_name(linear.kind));
    }
#endif
}

template <typename Runtime>
void RMSNorm::forward(
    Runtime& runtime, const QwenDeviceTensor& gamma, const uint16_t* input,
    uint16_t* output, int rows, int columns) const {
    typename Runtime::PhaseScope scope(&runtime, "norm");
    require_launch(qwen_rmsnorm_fp16_gamma_rows_f16(
        input, gamma.f16_data(), output, rows, columns,
        static_cast<float>(runtime.config.rms_norm_eps)), "Qwen FP16 RMSNorm");
}

template <typename Runtime>
void GatedDeltaNetAttention::forward(
    Runtime& runtime, DeviceLayer& layer, const uint16_t* hidden,
    uint16_t* output, int rows, int position_offset, int slot_id) const {
    // Phase 3.4: Every kernel below advances this slot's own recurrent
    // state. Resolved once here so the many call sites cannot disagree.
    const size_t state_offset = runtime.recurrent_slot_offset(slot_id, layer.linear.state);
    const size_t tail_offset = runtime.recurrent_slot_offset(slot_id, layer.linear.conv_tail);
    float* const state = layer.linear.state.f32_data() + state_offset;
    uint16_t* const conv_tail = layer.linear.conv_tail.f16_data() + tail_offset;
    const int key_heads = static_cast<int>(
        runtime.config.linear_attention.key_heads / runtime.options.tp_world);
    const int value_heads = static_cast<int>(
        runtime.config.linear_attention.value_heads / runtime.options.tp_world);
    const int key_dim = key_heads *
        static_cast<int>(runtime.config.linear_attention.key_head_dim);
    const int value_dim = value_heads *
        static_cast<int>(runtime.config.linear_attention.value_head_dim);
    const int packed_dim = 2 * key_dim + value_dim;
    const int kernel = static_cast<int>(runtime.config.linear_attention.conv_kernel_dim);
    const int hidden_size = static_cast<int>(runtime.config.hidden_size);
    const size_t packed_elements = static_cast<size_t>(rows) * packed_dim;
    const size_t key_elements = static_cast<size_t>(rows) * key_dim;
    const size_t value_elements = static_cast<size_t>(rows) * value_dim;
    const size_t gate_elements = static_cast<size_t>(rows) * value_heads;
    QwenDeviceTensor& packed = runtime.workspace_half(
        packed_elements, {static_cast<uint64_t>(rows),
                          static_cast<uint64_t>(packed_dim)});
    QwenDeviceTensor& convolved = runtime.workspace_half(packed_elements, packed.shape);
    QwenDeviceTensor& q = runtime.workspace_half(
        key_elements, {static_cast<uint64_t>(rows),
                       static_cast<uint64_t>(key_dim)});
    QwenDeviceTensor& k = runtime.workspace_half(key_elements, q.shape);
    QwenDeviceTensor& v = runtime.workspace_half(
        value_elements, {static_cast<uint64_t>(rows),
                         static_cast<uint64_t>(value_dim)});
    QwenDeviceTensor& a = runtime.workspace_half(
        gate_elements, {static_cast<uint64_t>(rows),
                        static_cast<uint64_t>(value_heads)});
    QwenDeviceTensor& b = runtime.workspace_half(gate_elements, a.shape);
    QwenDeviceTensor& gates = runtime.workspace_half(gate_elements, a.shape);
    QwenDeviceTensor& beta = runtime.workspace_half(gate_elements, a.shape);
    QwenDeviceTensor& core = runtime.workspace_half(value_elements, v.shape);
    QwenDeviceTensor& z = runtime.workspace_half(value_elements, v.shape);
    QwenDeviceTensor& normalized = runtime.workspace_half(value_elements, v.shape);
    // FlashQLA is an SM75 register/subgroup layout of the same recurrence,
    // not a distinct operation, so the Ascend build simply does not have the
    // selector: the normalized sequence kernel below produces the same
    // result from the same FP32 normalized Q/K.
#ifndef POCKET_BACKEND_ASCEND
    QwenDeviceTensor* q_normalized = nullptr;
    QwenDeviceTensor* k_normalized = nullptr;
    // Every sequence-recurrence kernel below reads `rows` as consecutive
    // tokens of one stream. A batched decode step is the opposite: `rows`
    // independent sequences each advancing one token, so none of them apply
    // and the batched step kernel handles it instead.
    const bool flashqla = runtime.batch_rows == nullptr && rows >= 8 &&
        runtime.layer_config.gated_delta_flashqla;
    if (flashqla) {
        q_normalized = &runtime.workspace_float(
            key_elements,
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(key_dim)});
        k_normalized = &runtime.workspace_float(key_elements, q_normalized->shape);
    }
#endif

#ifndef POCKET_BACKEND_ASCEND
    bool dual_qkvz = rows == 1 &&
        layer.linear.qkv.kind == QwenLinearKind::Fp8Block128 &&
        layer.linear.z.kind == QwenLinearKind::Fp8Block128 &&
        runtime.layer_config.fuse_qkvz_decode;
#else
    const bool dual_qkvz = false;
#endif
#ifndef POCKET_BACKEND_ASCEND
    if (dual_qkvz) {
        typename Runtime::PhaseScope dual_scope(&runtime, "pd.lin.qkvz");
        require_launch(qwen_fp8_e4m3_fp16scale_matvec_dual_f16_cuda(
            hidden,
            layer.linear.qkv.weight.fp8_data(),
            layer.linear.qkv.scale.f16_data(), packed.f16_data(), packed_dim,
            hidden_size, static_cast<int>(layer.linear.qkv.scale.shape[1]),
            layer.linear.z.weight.fp8_data(),
            layer.linear.z.scale.f16_data(), z.f16_data(), value_dim,
            hidden_size, static_cast<int>(layer.linear.z.scale.shape[1]),
            hidden_size), "dual FP8 QKV/Z decode projection");
    } else
#endif
    {
        runtime.linear_component.forward(
            runtime, layer.linear.qkv, hidden, packed.f16_data(), rows,
            "lin.qkv");
    }
#ifndef POCKET_BACKEND_ASCEND
    if (runtime.batch_rows != nullptr) {
        // One token per sequence against that sequence's own tail. The
        // pointer is the arena base here, not this slot's rows, because the
        // kernel resolves the slot per row.
        require_launch(qwen_causal_depthwise_conv_silu_f16_batched_cuda(
            packed.f16_data(), layer.linear.conv.f16_data(),
            layer.linear.conv_tail.f16_data(), convolved.f16_data(),
            runtime.batch_rows->slot_ids, rows, packed_dim, kernel,
            runtime.slot_stride_elements(layer.linear.conv_tail)),
            "FP16 batched linear causal convolution");
    } else
#endif
    require_launch(qwen_causal_depthwise_conv_silu_f16(
        packed.f16_data(), layer.linear.conv.f16_data(),
        conv_tail, convolved.f16_data(), rows,
        packed_dim, kernel, true), "FP16 linear causal convolution");
    require_launch(qwen_split_packed_qkv_f16(
        convolved.f16_data(), q.f16_data(), k.f16_data(), v.f16_data(),
        rows, key_dim, value_dim), "FP16 linear QKV split");
    if (layer.linear.ab.weight.data != nullptr &&
        runtime.layer_config.fuse_ab_projection) {
        // One GEMM over the row-concatenated [2*gate, hidden] weight. The
        // output is [rows, 2*gate] interleaved per row, so a and b are
        // strided slices of it rather than contiguous halves.
        QwenDeviceTensor& ab = runtime.workspace_half(
            gate_elements * 2,
            {static_cast<uint64_t>(rows),
             static_cast<uint64_t>(value_heads * 2)});
        runtime.linear_component.forward(
            runtime, layer.linear.ab, hidden, ab.f16_data(), rows, "lin.ab");
        require_launch(qwen_split_rows_pair_f16(
            ab.f16_data(), a.f16_data(), b.f16_data(), rows, value_heads),
            "FP16 fused a/b split");
    } else {
        runtime.linear_component.forward(
            runtime, layer.linear.a, hidden, a.f16_data(), rows, "lin.a");
        runtime.linear_component.forward(
            runtime, layer.linear.b, hidden, b.f16_data(), rows, "lin.b");
    }
    require_launch(qwen_linear_attn_gates_f16(
        a.f16_data(), b.f16_data(), layer.linear.a_log.f16_data(),
        layer.linear.dt_bias.f16_data(), gates.f16_data(), beta.f16_data(),
        rows, value_heads), "FP16 linear attention gates");
    const float q_scale = 1.0f / std::sqrt(
        static_cast<float>(runtime.config.linear_attention.key_head_dim));
    std::optional<typename Runtime::PhaseScope> delta_scope;
    delta_scope.emplace(&runtime, "gated_delta");
    bool sequenced = false;
#ifndef POCKET_BACKEND_ASCEND
    if (runtime.batch_rows != nullptr) {
        sequenced = qwen_gated_delta_step_batched_f16_cuda(
            layer.linear.state.f32_data(), q.f16_data(), k.f16_data(),
            v.f16_data(), gates.f16_data(), beta.f16_data(),
            core.f16_data(), runtime.batch_rows->slot_ids, rows, value_heads,
            key_heads,
            static_cast<int>(runtime.config.linear_attention.key_head_dim),
            static_cast<int>(runtime.config.linear_attention.value_head_dim),
            q_scale, runtime.slot_stride_elements(layer.linear.state));
        require_launch(sequenced, "FP16 batched linear recurrent state");
    } else if (flashqla) {
        // FlashQLA SM75 subgroup-sharded kernel. Still a serial recurrence,
        // but sharding the [128, 128] state over 16-lane subgroups replaces
        // the baseline's 128-element per-thread register vector: 1.85x on the
        // primitive and +5-7% real TP4 prefill at token-exact parity.
        // Consume the established FP32 normalized Q/K tensors so the norm
        // reduction order remains identical to the reference path.
        sequenced = qwen_normalize_gated_delta_qk_f16(
            q.f16_data(), k.f16_data(), q_normalized->f32_data(),
            k_normalized->f32_data(), rows, key_heads,
            static_cast<int>(runtime.config.linear_attention.key_head_dim)) &&
            qwen_gated_delta_flashqla_sm75_f16_cuda(
                state, q_normalized->f32_data(),
                k_normalized->f32_data(), v.f16_data(), gates.f16_data(),
                beta.f16_data(), core.f16_data(), rows, value_heads,
                key_heads,
                static_cast<int>(runtime.config.linear_attention.key_head_dim),
                static_cast<int>(runtime.config.linear_attention.value_head_dim),
                q_scale);
    } else
#endif
    if (rows >= 5 && key_heads < value_heads &&
        runtime.layer_config.gated_delta_prenormalize) {
        QwenDeviceTensor& q_normalized = runtime.workspace_float(
            key_elements, {static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(key_dim)});
        QwenDeviceTensor& k_normalized = runtime.workspace_float(
            key_elements, q_normalized.shape);
        sequenced = qwen_normalize_gated_delta_qk_f16(
            q.f16_data(), k.f16_data(), q_normalized.f32_data(),
            k_normalized.f32_data(), rows, key_heads,
            static_cast<int>(runtime.config.linear_attention.key_head_dim)) &&
            (runtime.layer_config.gated_delta_shared_state
             ? qwen_gated_delta_sequence_normalized_shared_f16(
                    state, q_normalized.f32_data(),
                    k_normalized.f32_data(), v.f16_data(), gates.f16_data(),
                    beta.f16_data(), core.f16_data(), rows, value_heads,
                    key_heads,
                    static_cast<int>(runtime.config.linear_attention.key_head_dim),
                    static_cast<int>(runtime.config.linear_attention.value_head_dim),
                    q_scale)
                : qwen_gated_delta_sequence_normalized_f16(
                    state, q_normalized.f32_data(),
                    k_normalized.f32_data(), v.f16_data(), gates.f16_data(),
                    beta.f16_data(), core.f16_data(), rows, value_heads,
                    key_heads,
                    static_cast<int>(runtime.config.linear_attention.key_head_dim),
                    static_cast<int>(runtime.config.linear_attention.value_head_dim),
                    q_scale));
    } else {
        // Also covers rows == 1. Decode used to fall through to the step
        // loop below, which round-trips the [128, 128] state through global
        // memory twice per token; the sequence kernel holds it in registers
        // and is bit-exact against the step kernel, so there is no reason to
        // reserve it for multi-token chunks. See the step-vs-sequence parity
        // check in test_qwen_half_ops.
        sequenced = qwen_gated_delta_sequence_f16(
            state, q.f16_data(), k.f16_data(),
            v.f16_data(), gates.f16_data(), beta.f16_data(),
            core.f16_data(), rows, value_heads, key_heads,
            static_cast<int>(runtime.config.linear_attention.key_head_dim),
            static_cast<int>(runtime.config.linear_attention.value_head_dim), q_scale);
    }
    for (int token = 0; !sequenced && token < rows; ++token) {
        require_launch(qwen_gated_delta_step_f16(
            state,
            q.f16_data() + static_cast<size_t>(token) * key_dim,
            k.f16_data() + static_cast<size_t>(token) * key_dim,
            v.f16_data() + static_cast<size_t>(token) * value_dim,
            gates.f16_data() + static_cast<size_t>(token) * value_heads,
            beta.f16_data() + static_cast<size_t>(token) * value_heads,
            core.f16_data() + static_cast<size_t>(token) * value_dim,
            value_heads, key_heads,
            static_cast<int>(runtime.config.linear_attention.key_head_dim),
            static_cast<int>(runtime.config.linear_attention.value_head_dim), q_scale),
            "FP16 linear recurrent state");
    }
    delta_scope.reset();
    if (!dual_qkvz) {
        runtime.linear_component.forward(
            runtime, layer.linear.z, hidden, z.f16_data(), rows, "lin.z");
    }
    require_launch(qwen_gated_rmsnorm_fp16_gamma_rows_f16(
        core.f16_data(), layer.linear.norm.f16_data(), z.f16_data(),
        normalized.f16_data(), rows * value_heads,
        static_cast<int>(runtime.config.linear_attention.value_head_dim),
        static_cast<float>(runtime.config.rms_norm_eps)),
        "FP16 linear gated RMSNorm");
    // ar.lin.out is the second-largest collective (2.69 s at 32K) and runs on
    // the 48 DeltaNet layers, so it is the other worthwhile overlap site.
    if (!runtime.projection_all_reduce_overlapped(
            layer.linear.out, normalized.f16_data(), output, rows,
            hidden_size, "lin.out", "lin.out")) {
        runtime.linear_component.forward(
            runtime, layer.linear.out, normalized.f16_data(), output, rows,
            "lin.out");
        runtime.all_reduce_half(output, rows * hidden_size, "lin.out");
    }
    (void)position_offset;
}

template <typename Runtime>
void GqaAttention::forward(
    Runtime& runtime, DeviceLayer& layer, const uint16_t* hidden,
    uint16_t* output, int rows, int position_offset, int slot_id) const {
    typename Runtime::PhaseScope scope(&runtime, "full_attention");
    const int q_heads = static_cast<int>(
        runtime.config.full_attention.num_heads / runtime.options.tp_world);
    const int kv_heads = static_cast<int>(
        runtime.config.full_attention.num_key_value_heads / runtime.options.tp_world);
    const int head_dim = static_cast<int>(runtime.config.full_attention.head_dim);
    const int attention_dim = q_heads * head_dim;
    const int q_projection_dim = attention_dim * 2;
    const size_t q_projection_elements =
        static_cast<size_t>(rows) * q_projection_dim;
    const size_t attention_elements =
        static_cast<size_t>(rows) * attention_dim;
    const size_t kv_elements =
        static_cast<size_t>(rows) * kv_heads * head_dim;
    QwenDeviceTensor& q_projection = runtime.workspace_half(
        q_projection_elements, {static_cast<uint64_t>(rows),
                                static_cast<uint64_t>(q_projection_dim)});
    QwenDeviceTensor& q = runtime.workspace_half(
        attention_elements, {static_cast<uint64_t>(rows),
                             static_cast<uint64_t>(attention_dim)});
    QwenDeviceTensor& gate = runtime.workspace_half(attention_elements, q.shape);
    QwenDeviceTensor& k = runtime.workspace_half(
        kv_elements, {static_cast<uint64_t>(rows),
                      static_cast<uint64_t>(kv_heads),
                      static_cast<uint64_t>(head_dim)});
    QwenDeviceTensor& v = runtime.workspace_half(kv_elements, k.shape);
    QwenDeviceTensor* kv_projection = nullptr;
    // Keep the candidate isolated to the fixed-width target verify batch;
    // prefill and one-row decode retain their established K/V projections.
    const bool fused_kv = rows > 1 && rows <= 8 &&
        layer.full.kv.weight.data != nullptr;
    if (fused_kv) {
        kv_projection = &runtime.workspace_half(
            static_cast<size_t>(rows) * 2 * kv_heads * head_dim,
            {static_cast<uint64_t>(rows),
             static_cast<uint64_t>(2 * kv_heads * head_dim)});
    }
    QwenDeviceTensor& q_norm = runtime.workspace_half(attention_elements, q.shape);
    QwenDeviceTensor& k_norm = runtime.workspace_half(kv_elements, k.shape);
    QwenDeviceTensor& attention = runtime.workspace_half(attention_elements, q.shape);
    QwenDeviceTensor& merged = runtime.workspace_half(attention_elements, q.shape);

    const int hidden_size = static_cast<int>(runtime.config.hidden_size);
#ifndef POCKET_BACKEND_ASCEND
    const bool grouped_qkv = rows == 1 && !fused_kv &&
        layer.full.q.kind == QwenLinearKind::Fp8Block128 &&
        layer.full.k.kind == QwenLinearKind::Fp8Block128 &&
        layer.full.v.kind == QwenLinearKind::Fp8Block128 &&
        runtime.layer_config.fuse_full_qkv_decode;
    if (grouped_qkv) {
        typename Runtime::PhaseScope sub(&runtime, "pd.full.qkv");
        require_launch(qwen_fp8_e4m3_fp16scale_matvec_triple_f16_cuda(
            hidden, layer.full.q.weight.fp8_data(),
            layer.full.q.scale.f16_data(), q_projection.f16_data(),
            q_projection_dim, hidden_size,
            static_cast<int>(layer.full.q.scale.shape[1]),
            layer.full.k.weight.fp8_data(), layer.full.k.scale.f16_data(),
            k.f16_data(), kv_heads * head_dim, hidden_size,
            static_cast<int>(layer.full.k.scale.shape[1]),
            layer.full.v.weight.fp8_data(), layer.full.v.scale.f16_data(),
            v.f16_data(), kv_heads * head_dim, hidden_size,
            static_cast<int>(layer.full.v.scale.shape[1]), hidden_size),
            "triple FP8 full Q/K/V decode projection");
    } else
#endif
    {
        {
            typename Runtime::PhaseScope sub(&runtime, "full.q_proj");
            runtime.linear_component.forward(
                runtime, layer.full.q, hidden, q_projection.f16_data(), rows,
                "full.q");
        }
        if (fused_kv) {
            {
                typename Runtime::PhaseScope sub(&runtime, "full.kv_proj");
                runtime.linear_component.forward(
                    runtime, layer.full.kv, hidden, kv_projection->f16_data(),
                    rows, "full.kv");
            }
            require_launch(qwen_split_rows_pair_f16(
                kv_projection->f16_data(), k.f16_data(), v.f16_data(), rows,
                kv_heads * head_dim), "FP16 fused full K/V split");
        } else {
            {
                typename Runtime::PhaseScope sub(&runtime, "full.k_proj");
                runtime.linear_component.forward(
                    runtime, layer.full.k, hidden, k.f16_data(), rows,
                    "full.k");
            }
            {
                typename Runtime::PhaseScope sub(&runtime, "full.v_proj");
                runtime.linear_component.forward(
                    runtime, layer.full.v, hidden, v.f16_data(), rows,
                    "full.v");
            }
        }
    }
    require_launch(qwen_split_q_gate_f16(
        q_projection.f16_data(), q.f16_data(), gate.f16_data(), rows,
        q_heads, head_dim), "FP16 full Q/gate split");
    runtime.rms_norm_component.forward(
        runtime, layer.full.q_norm, q.f16_data(), q_norm.f16_data(),
        rows * q_heads, head_dim);
    runtime.rms_norm_component.forward(
        runtime, layer.full.k_norm, k.f16_data(), k_norm.f16_data(),
        rows * kv_heads, head_dim);
#ifndef POCKET_BACKEND_ASCEND
    if (runtime.batch_rows != nullptr) {
        // Each row is a different sequence at its own position, so the
        // rotation angle is per row rather than position_offset + row.
        require_launch(qwen_partial_rope_rows_f16_batched_cuda(
            q_norm.f16_data(), k_norm.f16_data(), runtime.batch_rows->positions,
            rows, static_cast<int>(runtime.config.partial_rotary_dim()),
            static_cast<float>(runtime.config.rope_theta), q_heads, kv_heads,
            head_dim), "FP16 batched partial RoPE");
    } else
#endif
    require_launch(qwen_partial_rope_rows_f16(
        q_norm.f16_data(), k_norm.f16_data(), position_offset, rows,
        static_cast<int>(runtime.config.partial_rotary_dim()),
        static_cast<float>(runtime.config.rope_theta), q_heads, kv_heads, head_dim),
        "FP16 partial RoPE");

    const QwenKvCacheDType cache_dtype = runtime.options.kv_cache_dtype;
    const int attention_window = runtime.options.attention_window;
    const int sink_tokens = runtime.options.attention_sink_tokens;
#ifdef POCKET_BACKEND_ASCEND
    // The engine constructor rejects every non-FP16 cache dtype on this
    // backend, so only the FP16 append and the three neutral attention
    // launchers are compiled. The quantized caches are not merely disabled:
    // naming their kernels here would put a CUDA dependency in this object.

    // Phase 3.3: Use slot_id parameter instead of global current_slot_id
    const size_t slot_offset = runtime.kv_slot_offset_elements(slot_id, kv_heads, head_dim);

    require_launch(qwen_append_kv_cache_f16(
        k_norm.f16_data(), v.f16_data(),
        layer.full.k_cache.f16_data() + slot_offset,
        layer.full.v_cache.f16_data() + slot_offset,
        rows, kv_heads, head_dim,
        position_offset, runtime.max_context), "append FP16 full KV cache");

    if (rows == 1) {
        const int context_length = position_offset + 1;
        QwenDeviceTensor& scores = runtime.workspace_float(
            static_cast<size_t>(q_heads) * context_length,
            {static_cast<uint64_t>(q_heads),
             static_cast<uint64_t>(context_length)});
        require_launch(qwen_gqa_decode_attention_f16(
            q_norm.f16_data(),
            layer.full.k_cache.f16_data() + slot_offset,
            layer.full.v_cache.f16_data() + slot_offset,
            attention.f16_data(),
            scores.f32_data(), q_heads, kv_heads, head_dim,
            context_length, runtime.max_context), "decode FP16-cache GQA");
    } else if (rows <= 8) {
        const int context_length = position_offset + rows;
        // Same split geometry the CUDA verify path uses, so a verify block
        // reduces its partials in the same order on both backends.
        const int default_splits = context_length >= 1024 ? 64 : 32;
        int target_splits = runtime.layer_config.gqa_verify_splits > 0
                ? runtime.layer_config.gqa_verify_splits : default_splits;
        if (target_splits <= 0) target_splits = default_splits;
        const int verify_splits = std::max(
            1, std::min(target_splits, context_length));
        QwenDeviceTensor& partials = runtime.workspace_float(
            static_cast<size_t>(rows) * q_heads * verify_splits *
                static_cast<size_t>(head_dim + 2),
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_heads),
             static_cast<uint64_t>(verify_splits),
             static_cast<uint64_t>(head_dim + 2)});
        typename Runtime::PhaseScope sub(&runtime, "full.attn_kernel");
        require_launch(qwen_gqa_verify_attention_f16(
            q_norm.f16_data(),
            layer.full.k_cache.f16_data() + slot_offset,
            layer.full.v_cache.f16_data() + slot_offset,
            attention.f16_data(),
            partials.f32_data(), rows, q_heads, kv_heads, head_dim,
            position_offset, runtime.max_context, verify_splits),
            "verify split FP16-cache GQA");
    } else {
        require_launch(qwen_gqa_prefill_attention_f16(
            q_norm.f16_data(),
            layer.full.k_cache.f16_data() + slot_offset,
            layer.full.v_cache.f16_data() + slot_offset,
            attention.f16_data(), rows,
            q_heads, kv_heads, head_dim, position_offset, runtime.max_context),
            "prefill FP16-cache GQA");
    }
    (void)sink_tokens;
#else
    // Phase 3.3: Use slot_id parameter instead of global current_slot_id
    const size_t slot_offset = runtime.kv_slot_offset_elements(slot_id, kv_heads, head_dim);
    // Distance between slots in the KV arena. kv_slot_offset_elements folds
    // this to zero for a single session, so the stride is recomputed rather
    // than divided out of it.
    const size_t kv_slot_stride = static_cast<size_t>(runtime.max_context) *
        kv_heads * head_dim;

    if (runtime.batch_rows != nullptr) {
        // Each row appends its new K/V at its own position in its own slot.
        // Under paging the slot stride is replaced by the block table, which
        // the caller has already grown and uploaded for these positions.
        require_launch(qwen_append_kv_cache_f16_batched_cuda(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
            rows, kv_heads, head_dim, runtime.batch_rows->positions,
            runtime.batch_rows->slot_ids, runtime.max_context, kv_slot_stride,
            runtime.block_table_data(), runtime.paged_block_size(),
            runtime.paged_blocks_per_seq()),
            "append batched FP16 full KV cache");
    } else if (runtime.kv_paged()) {
        // Single-sequence paged append: consecutive positions of one
        // sequence, scattered across the blocks its row names.
        require_launch(qwen_append_kv_cache_f16_paged_cuda(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
            rows, kv_heads, head_dim, position_offset,
            runtime.block_table_row(slot_id), runtime.paged_block_size()),
            "append paged FP16 full KV cache");
    } else if (cache_dtype == QwenKvCacheDType::Fp8) {
        require_launch(qwen_append_kv_cache_fp8_cuda(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.fp8_data() + slot_offset,
            layer.full.v_cache.fp8_data() + slot_offset,
            layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
            layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
            rows, kv_heads, head_dim,
            kKvScaleBlock, position_offset, runtime.max_context),
            "append FP8 full KV cache");
    } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
        const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
        const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
            runtime.max_context * kv_heads * slot_bytes;
        require_launch(qwen_append_kv_cache_turboquant_k8v4_cuda(
            k_norm.f16_data(), v.f16_data(),
            layer.full.turboquant_cache.byte_data() + turboquant_slot_offset,
            rows, kv_heads, head_dim, position_offset, runtime.max_context),
            "append TurboQuant K8V4 full KV cache");
    } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
        const size_t scale_offset = static_cast<size_t>(slot_id) * runtime.max_context * kv_heads;
        require_launch(qwen_append_kv_cache_int8_per_token_head_cuda(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.int8_data() + slot_offset,
            layer.full.v_cache.int8_data() + slot_offset,
            layer.full.k_scale.f16_data() + scale_offset,
            layer.full.v_scale.f16_data() + scale_offset,
            rows, kv_heads, head_dim,
            position_offset, runtime.max_context),
            "append INT8 per-token-head full KV cache");
    } else {
        require_launch(qwen_append_kv_cache_f16(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.f16_data() + slot_offset,
            layer.full.v_cache.f16_data() + slot_offset,
            rows, kv_heads, head_dim,
            position_offset, runtime.max_context), "append FP16 full KV cache");
    }

    // Where the read kernels find this sequence's history. Contiguous, that
    // is the slot's own span of the arena. Paged and single-sequence, the
    // blocks are gathered once per layer into the dense
    // `[context, kv_heads, head_dim]` layout every read kernel already
    // indexes, so the five tuned prefill and decode variants stay untouched.
    // This is the same dequant-once trade the FP8 and TurboQuant paths make:
    // one O(context) pass instead of translating inside O(rows * context)
    // inner loops. Batched decode does not come through here; it reads the
    // blocks natively, which is the path that has to scale with batch size.
    // Only the FP16 cache stores halves here; the quantized caches keep
    // their own packed buffers and reach for them inside their own branches
    // below, so binding these eagerly would trip f16_data()'s dtype check.
    const bool fp16_cache = cache_dtype == QwenKvCacheDType::Fp16;
    const uint16_t* k_read = fp16_cache
        ? layer.full.k_cache.f16_data() + slot_offset : nullptr;
    const uint16_t* v_read = fp16_cache
        ? layer.full.v_cache.f16_data() + slot_offset : nullptr;
    if (runtime.kv_paged() && runtime.batch_rows == nullptr) {
        const int context_length = position_offset + rows;
        const std::vector<uint64_t> dense_shape = {
            static_cast<uint64_t>(context_length),
            static_cast<uint64_t>(kv_heads),
            static_cast<uint64_t>(head_dim)};
        const size_t dense_elements =
            static_cast<size_t>(context_length) * kv_heads * head_dim;
        QwenDeviceTensor& k_dense =
            runtime.workspace_half(dense_elements, dense_shape);
        QwenDeviceTensor& v_dense =
            runtime.workspace_half(dense_elements, dense_shape);
        typename Runtime::PhaseScope sub(&runtime, "full.kv_gather");
        require_launch(qwen_gather_kv_cache_f16_paged_cuda(
            layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
            k_dense.f16_data(), v_dense.f16_data(), context_length,
            kv_heads, head_dim, runtime.block_table_row(slot_id),
            runtime.paged_block_size()), "gather paged FP16 KV cache");
        k_read = k_dense.f16_data();
        v_read = v_dense.f16_data();
    }

    if (runtime.batch_rows != nullptr) {
        // One launch for the whole batch. The split geometry comes from the
        // longest row, so the scratch is sized from that same query the
        // launcher uses; shorter rows leave their trailing splits empty.
        const int splits = qwen_gqa_decode_batched_split_count(
            runtime.batch_rows->max_context_len, kv_heads, attention_window,
            sink_tokens);
        require_launch(splits > 0, "batched decode split geometry");
        QwenDeviceTensor& partials = runtime.workspace_float(
            static_cast<size_t>(rows) * q_heads * splits *
                static_cast<size_t>(head_dim + 2),
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_heads),
             static_cast<uint64_t>(splits),
             static_cast<uint64_t>(head_dim + 2)});
        typename Runtime::PhaseScope sub(&runtime, "full.attn_kernel");
        require_launch(qwen_gqa_decode_attention_f16_batched_cuda(
            q_norm.f16_data(), layer.full.k_cache.f16_data(),
            layer.full.v_cache.f16_data(), attention.f16_data(),
            partials.f32_data(), runtime.batch_rows->context_lens,
            runtime.batch_rows->slot_ids, rows, runtime.batch_rows->max_context_len,
            kv_slot_stride, q_heads, kv_heads, head_dim, runtime.max_context,
            attention_window, sink_tokens, runtime.block_table_data(),
            runtime.paged_block_size(), runtime.paged_blocks_per_seq()),
            "batched decode FP16-cache GQA");
    } else if (rows == 1) {
        const int context_length = position_offset + 1;
        const bool optimized_attention =
            runtime.layer_config.gqa_optimized ||
            attention_window > 0;
        // The split/merge decode path used to lose below 16384 context
        // because it walked 2048 positions per split serially, leaving the
        // device idle. At 128 positions per split it wins everywhere it is
        // allowed to run: 3.6x the reference kernels at 4096 context and
        // 16.1x at 65536. Its own guard still declines contexts under 4096.
        const bool optimized_decode = cache_dtype == QwenKvCacheDType::Fp16 &&
            optimized_attention &&
            (context_length >= 4096 || attention_window > 0);
        int attended_positions = context_length;
        if (attention_window > 0) {
            const int sink_count = std::min(sink_tokens, context_length);
            const int window_start = std::max(
                context_length - attention_window, sink_count);
            attended_positions = sink_count + (context_length - window_start);
        }
        // Must match the launch exactly; it sizes the partial scratch.
        const bool tensor_core_decode_shape = attention_window <= 0 &&
            sink_tokens == 0 && q_heads == kv_heads * 6 && head_dim == 256;
        const int optimized_splits = qwen_gqa_decode_split_count(
            attended_positions, kv_heads, tensor_core_decode_shape);
        const size_t score_elements = optimized_decode
            ? static_cast<size_t>(q_heads) * optimized_splits *
                  static_cast<size_t>(head_dim + 2)
            : static_cast<size_t>(q_heads) * context_length;
        QwenDeviceTensor& scores = runtime.workspace_float(
            score_elements,
            optimized_decode
                ? std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                        static_cast<uint64_t>(optimized_splits),
                                        static_cast<uint64_t>(head_dim + 2)}
                : std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                         static_cast<uint64_t>(context_length)});
        if (cache_dtype == QwenKvCacheDType::Fp8) {
            require_launch(qwen_gqa_decode_attention_fp8_cuda(
                q_norm.f16_data(),
                layer.full.k_cache.fp8_data() + slot_offset,
                layer.full.v_cache.fp8_data() + slot_offset,
                layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
                layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
                attention.f16_data(),
                scores.f32_data(), q_heads, kv_heads, head_dim,
                kKvScaleBlock, context_length, runtime.max_context),
                "decode FP8-cache GQA");
        } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
            const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
            const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
                runtime.max_context * kv_heads * slot_bytes;
            require_launch(qwen_gqa_decode_attention_turboquant_k8v4_cuda(
                q_norm.f16_data(),
                layer.full.turboquant_cache.byte_data() + turboquant_slot_offset,
                attention.f16_data(), scores.f32_data(), q_heads, kv_heads,
                head_dim, context_length, runtime.max_context, attention_window,
                sink_tokens), "decode TurboQuant K8V4 GQA");
        } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
            const size_t scale_offset = static_cast<size_t>(slot_id) *
                runtime.max_context * kv_heads;
            require_launch(qwen_gqa_decode_attention_int8_per_token_head_cuda(
                q_norm.f16_data(),
                layer.full.k_cache.int8_data() + slot_offset,
                layer.full.v_cache.int8_data() + slot_offset,
                layer.full.k_scale.f16_data() + scale_offset,
                layer.full.v_scale.f16_data() + scale_offset,
                attention.f16_data(),
                scores.f32_data(), q_heads, kv_heads, head_dim,
                context_length, runtime.max_context, attention_window, sink_tokens),
                "decode INT8 per-token-head GQA");
        } else if (optimized_decode) {
            require_launch(qwen_gqa_decode_attention_f16_fused_cuda(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(),
                scores.f32_data(), q_heads, kv_heads, head_dim,
                context_length, runtime.max_context, attention_window, sink_tokens),
                "decode optimized FP16-cache GQA");
        } else {
            require_launch(qwen_gqa_decode_attention_f16(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(),
                scores.f32_data(), q_heads, kv_heads, head_dim,
                context_length, runtime.max_context), "decode FP16-cache GQA");
        }
    } else if (cache_dtype == QwenKvCacheDType::Fp8) {
        const int context_length = position_offset + rows;
        // Dequantize the [0, context_length) range once into dense FP16
        // buffers, then call the tensor-core prefill kernel. O(ctx) dequant
        // + tensor core beats O(rows*ctx) inline dequant by the same factor
        // as TurboQuant (6-13×).
        QwenDeviceTensor& k_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        require_launch(qwen_fp8_dequant_kv_cache_cuda(
            layer.full.k_cache.fp8_data() + slot_offset,
            layer.full.v_cache.fp8_data() + slot_offset,
            layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
            layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
            k_dense.f16_data(), v_dense.f16_data(), context_length,
            kv_heads, head_dim, kKvScaleBlock, runtime.max_context),
            "dequant FP8 cache to dense FP16");
        if (runtime.layer_config.gqa_optimized ||
            attention_window > 0) {
            require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context, attention_window, sink_tokens),
                "prefill FP8 via tensor-core tiled");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context),
                "prefill FP8 via tensor-core exact");
        }
    } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
        const int context_length = position_offset + rows;
        // INT8 per-token-head uses the same dequant-once architecture as FP8
        QwenDeviceTensor& k_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        const size_t scale_offset = static_cast<size_t>(slot_id) * runtime.max_context * kv_heads;
        require_launch(qwen_int8_dequant_kv_cache_cuda(
            layer.full.k_cache.int8_data() + slot_offset,
            layer.full.v_cache.int8_data() + slot_offset,
            layer.full.k_scale.f16_data() + scale_offset,
            layer.full.v_scale.f16_data() + scale_offset,
            k_dense.f16_data(), v_dense.f16_data(), context_length,
            kv_heads, head_dim, runtime.max_context),
            "dequant INT8 cache to dense FP16");
        if (runtime.layer_config.gqa_optimized ||
            attention_window > 0) {
            require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context, attention_window, sink_tokens),
                "prefill INT8 via tensor-core tiled");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context),
                "prefill INT8 via tensor-core exact");
        }
    } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
        const int context_length = position_offset + rows;
        // Dequantize the [0, context_length) range once into dense FP16
        // buffers laid out like the FP16 cache, then call the tensor-core
        // prefill kernel on them. O(ctx) dequant + tensor core beats
        // O(rows*ctx) inline dequant by 6-13× and keeps prefill fast.
        QwenDeviceTensor& k_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v_dense = runtime.workspace_half(
            static_cast<size_t>(context_length) * kv_heads * head_dim,
            {static_cast<uint64_t>(context_length),
             static_cast<uint64_t>(kv_heads),
             static_cast<uint64_t>(head_dim)});
        const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
        const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
            runtime.max_context * kv_heads * slot_bytes;
        require_launch(qwen_turboquant_k8v4_dequant_kv_cuda(
            layer.full.turboquant_cache.byte_data() + turboquant_slot_offset, k_dense.f16_data(),
            v_dense.f16_data(), context_length, kv_heads, head_dim,
            runtime.max_context), "dequant TurboQuant cache to dense FP16");
        if (runtime.layer_config.gqa_optimized ||
            attention_window > 0) {
            require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context, attention_window, sink_tokens),
                "prefill TurboQuant via tensor-core tiled");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16(
                q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context),
                "prefill TurboQuant via tensor-core exact");
        }
    } else if (cache_dtype == QwenKvCacheDType::Fp16 && rows <= 8 && attention_window == 0) {
        const int context_length = position_offset + rows;
        // Tensor-core QK experiment for the real TP4 shape (one KV head per
        // rank). Keeps the existing FP32 softmax/PV order; opt-in until real
        // parity and latency are validated.
        if (runtime.layer_config.gqa_verify_cublas_qk &&
            kv_heads == 1) {
            const size_t score_elements =
                static_cast<size_t>(rows) * q_heads * context_length;
            QwenDeviceTensor& scores = runtime.workspace_float(
                score_elements,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(context_length)});
            typename Runtime::PhaseScope sub(&runtime, "full.attn_kernel");
            require_launch(qwen_gqa_verify_attention_f16_cublas_qk_cuda(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(),
                scores.f32_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context),
                "verify cuBLAS-QK FP16-cache GQA");
        // The exact three-kernel path avoids split partials below 1K context;
        // the split-K path crosses over above that and remains faster at long
        // context. QWEN_GQA_VERIFY_SPLIT explicitly overrides the crossover.
        } else if (runtime.layer_config.gqa_verify_split ==
                       OptionalSwitch::Enabled ||
                   (runtime.layer_config.gqa_verify_split ==
                        OptionalSwitch::Auto && context_length > 1024)) {
            // 32 splits stay slightly faster end-to-end at short context;
            // 64 cross over at 1K and remain best through 32K.
            const int default_splits = context_length >= 1024 ? 64 : 32;
            int target_splits = runtime.layer_config.gqa_verify_splits > 0
                ? runtime.layer_config.gqa_verify_splits : default_splits;
            if (target_splits <= 0) target_splits = default_splits;
            const int verify_splits = std::max(
                1, std::min(target_splits, context_length));
            const size_t partial_elements =
                static_cast<size_t>(rows) * q_heads * verify_splits *
                static_cast<size_t>(head_dim + 2);
            QwenDeviceTensor& partials = runtime.workspace_float(
                partial_elements,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(verify_splits),
                 static_cast<uint64_t>(head_dim + 2)});
            typename Runtime::PhaseScope sub(&runtime, "full.attn_kernel");
            require_launch(qwen_gqa_verify_attention_f16(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(),
                partials.f32_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, runtime.max_context, verify_splits),
                "verify split FP16-cache GQA");
        } else {
            const size_t score_elements =
                static_cast<size_t>(rows) * q_heads * context_length;
            QwenDeviceTensor& scores = runtime.workspace_float(
                score_elements,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(context_length)});
            {
                typename Runtime::PhaseScope sub(
                    &runtime, "full.attn_kernel");
                require_launch(qwen_gqa_verify_attention_f16_exact_cuda(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    scores.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, runtime.max_context),
                    "verify exact FP16-cache GQA");
            }
        }
    // The tiled kernel shares each K/V element across a head group and a
    // pair of query rows, so it is the default for multi-row prefill. The
    // wider TP4 candidate is selected separately through LONG_TILE.
    } else if (runtime.layer_config.gqa_optimized ||
               attention_window > 0) {
        require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
            q_norm.f16_data(),
            k_read,
            v_read,
            attention.f16_data(), rows,
            q_heads, kv_heads, head_dim, position_offset, runtime.max_context,
            attention_window, sink_tokens),
            "prefill optimized FP16-cache GQA");
    } else {
        require_launch(qwen_gqa_prefill_attention_f16(
            q_norm.f16_data(),
            k_read,
            v_read,
            attention.f16_data(), rows,
            q_heads, kv_heads, head_dim, position_offset, runtime.max_context),
            "prefill FP16-cache GQA");
    }
#endif
    require_launch(qwen_sigmoid_mul_f16(
        attention.f16_data(), gate.f16_data(), merged.f16_data(),
        rows * attention_dim), "FP16 full attention output gate");
    // Row-parallel output projection: each slice's rows are complete once its
    // GEMM lands, so the collective for slice i can run under slice i+1's GEMM.
    // This site was left serial when mlp.down and lin.out were pipelined, and
    // at 65K it was the single largest exposed collective in the profile
    // (ar.full.out 2.12 s against tp_all_reduce's 3.16 s total).
    if (!runtime.projection_all_reduce_overlapped(
            layer.full.out, merged.f16_data(), output, rows,
            static_cast<int>(runtime.config.hidden_size), "full.out", "full.out")) {
        {
            typename Runtime::PhaseScope sub(&runtime, "full.out_proj");
            runtime.linear_component.forward(
                runtime, layer.full.out, merged.f16_data(), output, rows,
                "full.out");
        }
        runtime.all_reduce_half(
            output, rows * static_cast<int>(runtime.config.hidden_size),
            "full.out");
    }
}


template <typename Runtime>
typename FusedGateUpSwiGLU::Workspace FusedGateUpSwiGLU::reserve(
    Runtime& runtime, DeviceLayer& layer, int rows, int hidden_size) const {
    Workspace workspace;
#ifdef POCKET_BACKEND_ASCEND
    // Quantized fused kernels are unavailable on this backend. Constants keep
    // the shared workspace and composition path identical after dead-code removal.
    workspace.fused_decode = false;
    workspace.fused_small_batch = false;
    workspace.fused_nvfp4 = false;
    workspace.shared_nvfp4 = false;
#else
    const bool compatible_fp8 =
        layer.gate.kind == QwenLinearKind::Fp8Block128 &&
        layer.up.kind == QwenLinearKind::Fp8Block128 &&
        layer.gate.logical_shape == layer.up.logical_shape &&
        layer.gate.scale.shape == layer.up.scale.shape;
    workspace.fused_decode = rows == 1 && compatible_fp8;
    workspace.fused_small_batch = rows > 1 && rows <= 8 &&
        compatible_fp8 && hidden_size % 4 == 0;
    const bool nvfp4_wmma =
        runtime.layer_config.nvfp4_mode == NvFp4Mode::Auto ||
        runtime.layer_config.nvfp4_mode == NvFp4Mode::Wmma;
    const bool compatible_nvfp4 = rows >= 8 && nvfp4_wmma &&
        layer.gate.kind == QwenLinearKind::NvFp4Group16 &&
        layer.up.kind == QwenLinearKind::NvFp4Group16 &&
        layer.gate.logical_shape == layer.up.logical_shape;
    workspace.fused_nvfp4 = compatible_nvfp4 &&
        runtime.layer_config.nvfp4_fused_swiglu;
    workspace.shared_nvfp4 = compatible_nvfp4 &&
        runtime.layer_config.nvfp4_shared_q8_swiglu &&
        !workspace.fused_nvfp4;
#endif
    const size_t intermediate_elements = static_cast<size_t>(rows) *
        layer.gate.logical_shape[0];
    if (!workspace.fused_decode && !workspace.fused_small_batch &&
        !workspace.fused_nvfp4) {
        workspace.gate = &runtime.workspace_half(
            intermediate_elements,
            {static_cast<uint64_t>(rows), layer.gate.logical_shape[0]});
        workspace.up = &runtime.workspace_half(
            intermediate_elements, workspace.gate->shape);
    }
    workspace.output = &runtime.workspace_half(
        intermediate_elements,
        {static_cast<uint64_t>(rows), layer.gate.logical_shape[0]});
    return workspace;
}

template <typename Runtime>
void FusedGateUpSwiGLU::forward(
    Runtime& runtime, DeviceLayer& layer, const uint16_t* input,
    Workspace& workspace, int rows, int hidden_size) const {
#ifdef POCKET_BACKEND_ASCEND
    runtime.linear_component.forward(
        runtime, layer.gate, input, workspace.gate->f16_data(), rows,
        "mlp.gate");
    runtime.linear_component.forward(
        runtime, layer.up, input, workspace.up->f16_data(), rows, "mlp.up");
    require_launch(qwen_silu_mul_rows_f16(
        workspace.gate->f16_data(), workspace.up->f16_data(),
        workspace.output->f16_data(), rows,
        static_cast<int>(layer.gate.logical_shape[0])), "FP16 SwiGLU");
#else
    if (workspace.fused_decode) {
        typename Runtime::PhaseScope scope(&runtime, "swiglu.d");
        require_launch(qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
            input, layer.gate.weight.fp8_data(),
            layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
            layer.up.scale.f16_data(), workspace.output->f16_data(),
            static_cast<int>(layer.gate.logical_shape[0]), hidden_size,
            hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
            "FP16 fused decode SwiGLU");
    } else if (workspace.fused_small_batch) {
        typename Runtime::PhaseScope scope(&runtime, "swiglu.r");
        require_launch(
            qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
                input, layer.gate.weight.fp8_data(),
                layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                layer.up.scale.f16_data(), workspace.output->f16_data(), rows,
                static_cast<int>(layer.gate.logical_shape[0]), hidden_size,
                hidden_size, static_cast<int>(layer.gate.logical_shape[0]),
                hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
            "FP16 fused small-batch SwiGLU");
    } else if (workspace.fused_nvfp4) {
        typename Runtime::PhaseScope scope(&runtime, "projection_rows_nvfp4");
        require_launch(runtime.nvfp4_fused_swiglu_projection(
            layer.gate, layer.up, input, workspace.output->f16_data(), rows),
            "NVFP4 fused Q8 WMMA SwiGLU projection");
    } else {
        if (workspace.shared_nvfp4) {
            typename Runtime::PhaseScope scope(
                &runtime, "projection_rows_nvfp4");
            require_launch(runtime.nvfp4_shared_q8_swiglu_projection(
                layer.gate, layer.up, input, workspace.gate->f16_data(),
                workspace.up->f16_data(), rows),
                "NVFP4 shared-Q8 WMMA gate/up projection");
        } else {
            runtime.linear_component.forward(
                runtime, layer.gate, input, workspace.gate->f16_data(), rows,
                "mlp.gate");
            runtime.linear_component.forward(
                runtime, layer.up, input, workspace.up->f16_data(), rows,
                "mlp.up");
        }
        require_launch(qwen_silu_mul_rows_f16(
            workspace.gate->f16_data(), workspace.up->f16_data(),
            workspace.output->f16_data(), rows,
            static_cast<int>(layer.gate.logical_shape[0])), "FP16 SwiGLU");
    }
#endif
}

template <typename Runtime>
void DecoderLayer::forward(Runtime& runtime, DeviceLayer& layer,
                           const uint16_t* hidden, uint16_t* output, int rows,
                           int position_offset, int slot_id) const {
    const int hidden_size = static_cast<int>(runtime.config.hidden_size);
    runtime.begin_workspace();
    const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
    QwenDeviceTensor& normalized = runtime.workspace_half(
        hidden_elements, {static_cast<uint64_t>(rows),
                          static_cast<uint64_t>(hidden_size)});
    QwenDeviceTensor& attention = runtime.workspace_half(
        hidden_elements, normalized.shape);
    QwenDeviceTensor& post = runtime.workspace_half(
        hidden_elements, normalized.shape);
    typename FusedGateUpSwiGLU::Workspace swiglu =
        runtime.swiglu_component.reserve(runtime, layer, rows, hidden_size);
    QwenDeviceTensor& mlp = runtime.workspace_half(
        hidden_elements, normalized.shape);

    runtime.rms_norm_component.forward(
        runtime, layer.input_norm, hidden, normalized.f16_data(), rows,
        hidden_size);
    if (layer.linear.qkv.weight.data != nullptr) {
        runtime.gated_delta_attention_component.forward(
            runtime, layer, normalized.f16_data(), attention.f16_data(), rows,
            position_offset, slot_id);
    } else {
        runtime.gqa_attention_component.forward(
            runtime, layer, normalized.f16_data(), attention.f16_data(), rows,
            position_offset, slot_id);
    }
    // One fused pass replaces residual copy, add, and norm. The opt-out remains
    // available for A/B while construction-time policy keeps requests immutable.
    if (runtime.layer_config.fuse_attention_residual_norm) {
        typename Runtime::PhaseScope scope(&runtime, "attn_resid_norm");
        require_launch(qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
            hidden, attention.f16_data(), layer.post_norm.f16_data(), output,
            post.f16_data(), rows, hidden_size,
            static_cast<float>(runtime.config.rms_norm_eps)),
            "Qwen fused attention residual RMSNorm");
    } else {
        {
            typename Runtime::PhaseScope scope(&runtime, "resid_copy");
            check_device(memcpy_d2d(
                output, hidden, hidden_elements * sizeof(uint16_t)),
                "Qwen FP16 residual copy");
        }
        runtime.add(output, attention.f16_data(), rows * hidden_size);
        runtime.rms_norm_component.forward(
            runtime, layer.post_norm, output, post.f16_data(), rows,
            hidden_size);
    }
    runtime.swiglu_component.forward(
        runtime, layer, post.f16_data(), swiglu, rows, hidden_size);
    // The down projection and all-reduce are the largest overlap opportunity.
    if (!runtime.projection_all_reduce_overlapped(
            layer.down, swiglu.output->f16_data(), mlp.f16_data(), rows,
            hidden_size, "mlp.down", "mlp")) {
        runtime.linear_component.forward(
            runtime, layer.down, swiglu.output->f16_data(), mlp.f16_data(),
            rows, "mlp.down");
        runtime.all_reduce_half(mlp.f16_data(), rows * hidden_size, "mlp");
    }
    runtime.add(output, mlp.f16_data(), rows * hidden_size);
}

}  // namespace qwen_components
