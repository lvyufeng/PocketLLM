#include "qwen_engine.hpp"

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"
#include "tp_comm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace dsv4 {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void require_launch(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("Qwen CUDA launch failed: ") + what);
}

float* fp32(QwenDeviceTensor& tensor) {
    return static_cast<float*>(tensor.data);
}

const float* fp32(const QwenDeviceTensor& tensor) {
    return static_cast<const float*>(tensor.data);
}

uint16_t* fp16(QwenDeviceTensor& tensor) {
    return static_cast<uint16_t*>(tensor.data);
}

const uint16_t* fp16(const QwenDeviceTensor& tensor) {
    return static_cast<const uint16_t*>(tensor.data);
}

uint8_t* fp8(QwenDeviceTensor& tensor) {
    return static_cast<uint8_t*>(tensor.data);
}

const uint8_t* fp8(const QwenDeviceTensor& tensor) {
    return static_cast<const uint8_t*>(tensor.data);
}

struct DeviceLinear {
    QwenDeviceTensor weight;
    QwenDeviceTensor scale;
    bool fp8 = false;
};

struct DeviceLinearAttention {
    DeviceLinear qkv;
    DeviceLinear z;
    DeviceLinear out;
    DeviceLinear a;
    DeviceLinear b;
    QwenDeviceTensor conv;
    QwenDeviceTensor a_log;
    QwenDeviceTensor dt_bias;
    QwenDeviceTensor norm;
    QwenDeviceTensor state;
    QwenDeviceTensor conv_tail;
};

struct DeviceFullAttention {
    DeviceLinear q;
    DeviceLinear k;
    DeviceLinear v;
    DeviceLinear out;
    QwenDeviceTensor q_norm;
    QwenDeviceTensor k_norm;
    QwenDeviceTensor k_cache;
    QwenDeviceTensor v_cache;
};

struct DeviceLayer {
    QwenDeviceTensor input_norm;
    QwenDeviceTensor post_norm;
    DeviceLinearAttention linear;
    DeviceFullAttention full;
    DeviceLinear gate;
    DeviceLinear up;
    DeviceLinear down;
};

QwenDeviceTensor upload(const SafeTensorsIndex& index, const QwenTensorRef& ref) {
    return qwen_upload_tensor_cuda(index, ref);
}

DeviceLinear upload_linear(const SafeTensorsIndex& index, const QwenLinearRef& ref) {
    DeviceLinear out;
    out.weight = upload(index, ref.weight);
    out.fp8 = ref.weight.device_dtype == SafeDType::F8_E4M3;
    if (ref.has_scale) out.scale = upload(index, ref.scale);
    return out;
}

void allocate(QwenDeviceTensor& tensor, size_t bytes, const std::vector<uint64_t>& shape,
              SafeDType dtype) {
    if (bytes == 0) throw std::runtime_error("Qwen attempted to allocate an empty tensor");
    if (tensor.data != nullptr && tensor.capacity >= bytes) {
        tensor.device_dtype = dtype;
        tensor.shape = shape;
        tensor.nbytes = bytes;
        return;
    }
    if (tensor.data != nullptr) {
        check_cuda(cudaFree(tensor.data), "cudaFree Qwen runtime tensor");
        tensor.data = nullptr;
    }
    check_cuda(cudaMalloc(&tensor.data, bytes), "cudaMalloc Qwen runtime tensor");
    tensor.device_dtype = dtype;
    tensor.shape = shape;
    tensor.nbytes = bytes;
    tensor.capacity = bytes;
}

void allocate_float(QwenDeviceTensor& tensor, size_t elements, const std::vector<uint64_t>& shape) {
    allocate(tensor, elements * sizeof(float), shape, SafeDType::F32);
}

void zero_tensor(QwenDeviceTensor& tensor) {
    check_cuda(cudaMemset(tensor.data, 0, tensor.nbytes), "cudaMemset Qwen runtime tensor");
}

// One layer's temporaries are live concurrently, but layers themselves run in
// strict sequence. Reusing slots across layers removes synchronous allocator
// calls without changing any operator inputs.
struct QwenWorkspace {
    std::vector<QwenDeviceTensor> slots;
    size_t cursor = 0;

    QwenWorkspace() { slots.reserve(32); }

    void begin() { cursor = 0; }

    QwenDeviceTensor& float_tensor(size_t elements, const std::vector<uint64_t>& shape) {
        if (cursor == slots.size()) slots.emplace_back();
        QwenDeviceTensor& tensor = slots[cursor++];
        allocate_float(tensor, elements, shape);
        return tensor;
    }
};

}  // namespace

struct QwenEngine::Impl {
    SafeTensorsIndex& index;
    QwenConfig& config;
    QwenEngineOptions options;
    int max_context = 0;
    std::vector<DeviceLayer> layers;
    QwenDeviceTensor embed;
    QwenDeviceTensor final_norm;
    QwenDeviceTensor lm_head;
    std::vector<int> local_tokens;
    QwenDeviceTensor d_tokens;
    QwenDeviceTensor hidden_a;
    QwenDeviceTensor hidden_b;
    QwenDeviceTensor work_a;
    QwenDeviceTensor work_b;
    QwenDeviceTensor final_hidden;
    QwenDeviceTensor logits;
    QwenDeviceTensor argmax_token;
    QwenDeviceTensor argmax_logit;
    QwenWorkspace workspace;

    // active_layers bounds how many decoder layers are uploaded. A partial-depth
    // smoke run must not stage the full 64-layer checkpoint: at TP1 that is
    // 27.4 GiB and does not fit on a 22 GiB card.
    // Bytes actually staged on this device, as opposed to what the full map
    // describes; a partial-depth run uploads only a prefix of the layers.
    uint64_t uploaded_weight_bytes = 0;
    uint64_t uploaded_scale_bytes = 0;

    Impl(SafeTensorsIndex& index_, QwenConfig& config_, const QwenWeightMap& map,
         const QwenEngineOptions& options_, int max_context_, int active_layers)
        : index(index_), config(config_), options(options_), max_context(max_context_) {
        embed = upload(index, map.embed_tokens());
        final_norm = upload(index, map.final_norm());
        lm_head = upload(index, map.lm_head());
        uploaded_weight_bytes += map.embed_tokens().device_nbytes +
                                 map.final_norm().device_nbytes + map.lm_head().device_nbytes;
        const size_t layer_limit = active_layers > 0
            ? std::min(static_cast<size_t>(active_layers), map.layers().size())
            : map.layers().size();
        layers.reserve(layer_limit);
        const int world = options.tp_world;
        const int rank = options.tp_rank;
        const int local_key_heads = static_cast<int>(config.linear_attention.key_heads / world);
        const int local_value_heads = static_cast<int>(config.linear_attention.value_heads / world);
        const int local_key_dim = local_key_heads * static_cast<int>(config.linear_attention.key_head_dim);
        const int local_value_dim = local_value_heads * static_cast<int>(config.linear_attention.value_head_dim);
        const int local_qkv_dim = 2 * local_key_dim + local_value_dim;
        const int local_q_heads = static_cast<int>(config.full_attention.num_heads / world);
        const int local_kv_heads = static_cast<int>(config.full_attention.num_key_value_heads / world);
        for (size_t layer_index = 0; layer_index < layer_limit; ++layer_index) {
            const QwenLayerWeights& src = map.layers()[layer_index];
            for (const QwenTensorRef* ref : {
                     &src.input_layernorm, &src.post_attention_layernorm,
                     &src.mlp.gate_proj.weight, &src.mlp.up_proj.weight, &src.mlp.down_proj.weight,
                     &src.linear_attention.in_proj_qkv.weight, &src.linear_attention.in_proj_z.weight,
                     &src.linear_attention.out_proj.weight, &src.linear_attention.in_proj_a.weight,
                     &src.linear_attention.in_proj_b.weight, &src.linear_attention.conv1d,
                     &src.linear_attention.a_log, &src.linear_attention.dt_bias,
                     &src.linear_attention.norm,
                     &src.full_attention.q_proj.weight, &src.full_attention.k_proj.weight,
                     &src.full_attention.v_proj.weight, &src.full_attention.o_proj.weight,
                     &src.full_attention.q_norm, &src.full_attention.k_norm}) {
                if (ref->found) uploaded_weight_bytes += ref->device_nbytes;
            }
            for (const QwenLinearRef* ref : {
                     &src.mlp.gate_proj, &src.mlp.up_proj, &src.mlp.down_proj,
                     &src.linear_attention.in_proj_qkv, &src.linear_attention.in_proj_z,
                     &src.linear_attention.out_proj,
                     &src.full_attention.q_proj, &src.full_attention.k_proj,
                     &src.full_attention.v_proj, &src.full_attention.o_proj}) {
                if (ref->has_scale) uploaded_scale_bytes += ref->scale.device_nbytes;
            }
            DeviceLayer dst;
            dst.input_norm = upload(index, src.input_layernorm);
            dst.post_norm = upload(index, src.post_attention_layernorm);
            dst.gate = upload_linear(index, src.mlp.gate_proj);
            dst.up = upload_linear(index, src.mlp.up_proj);
            dst.down = upload_linear(index, src.mlp.down_proj);
            if (src.linear_attention.in_proj_qkv.weight.found) {
                dst.linear.qkv = upload_linear(index, src.linear_attention.in_proj_qkv);
                dst.linear.z = upload_linear(index, src.linear_attention.in_proj_z);
                dst.linear.out = upload_linear(index, src.linear_attention.out_proj);
                dst.linear.a = upload_linear(index, src.linear_attention.in_proj_a);
                dst.linear.b = upload_linear(index, src.linear_attention.in_proj_b);
                dst.linear.conv = upload(index, src.linear_attention.conv1d);
                dst.linear.a_log = upload(index, src.linear_attention.a_log);
                dst.linear.dt_bias = upload(index, src.linear_attention.dt_bias);
                dst.linear.norm = upload(index, src.linear_attention.norm);
                allocate_float(dst.linear.state,
                               static_cast<size_t>(local_value_heads) * config.linear_attention.key_head_dim *
                                   config.linear_attention.value_head_dim,
                               {static_cast<uint64_t>(local_value_heads), config.linear_attention.key_head_dim,
                                config.linear_attention.value_head_dim});
                allocate_float(dst.linear.conv_tail,
                               static_cast<size_t>(std::max(0, static_cast<int>(config.linear_attention.conv_kernel_dim) - 1)) *
                                   local_qkv_dim,
                               {static_cast<uint64_t>(std::max(0, static_cast<int>(config.linear_attention.conv_kernel_dim) - 1)),
                                static_cast<uint64_t>(local_qkv_dim)});
                zero_tensor(dst.linear.state);
                zero_tensor(dst.linear.conv_tail);
            } else {
                dst.full.q = upload_linear(index, src.full_attention.q_proj);
                dst.full.k = upload_linear(index, src.full_attention.k_proj);
                dst.full.v = upload_linear(index, src.full_attention.v_proj);
                dst.full.out = upload_linear(index, src.full_attention.o_proj);
                dst.full.q_norm = upload(index, src.full_attention.q_norm);
                dst.full.k_norm = upload(index, src.full_attention.k_norm);
                allocate_float(dst.full.k_cache,
                               static_cast<size_t>(max_context) * local_kv_heads * config.full_attention.head_dim,
                               {static_cast<uint64_t>(max_context), static_cast<uint64_t>(local_kv_heads),
                                config.full_attention.head_dim});
                allocate_float(dst.full.v_cache,
                               static_cast<size_t>(max_context) * local_kv_heads * config.full_attention.head_dim,
                               {static_cast<uint64_t>(max_context), static_cast<uint64_t>(local_kv_heads),
                                config.full_attention.head_dim});
                zero_tensor(dst.full.k_cache);
                zero_tensor(dst.full.v_cache);
            }
            layers.push_back(std::move(dst));
        }
        (void)rank;
    }

    ~Impl() = default;

    void all_reduce(float* values, int count) {
        if (options.tp_world == 1) return;
#ifdef DSV4_HAVE_NCCL
        if (options.nccl_id_path.empty()) throw std::runtime_error("Qwen TP requires --nccl-id-path");
        nccl_all_reduce_sum_float_inplace(options.tp_world, options.tp_rank, options.device,
                                          options.nccl_id_path.c_str(), values, count);
#else
        (void)values;
        (void)count;
        throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
#endif
    }

    void projection(const DeviceLinear& linear, const float* x, float* y, int rows) {
        const int out_rows = static_cast<int>(linear.weight.shape[0]);
        const int cols = static_cast<int>(linear.weight.shape[1]);
        if (linear.fp8) {
            if (rows == 1) {
                // Decode: warp-per-row matvec avoids the block reduction.
                require_launch(qwen_fp8_e4m3_fp16scale_matvec_cuda(
                                   x, fp8(const_cast<QwenDeviceTensor&>(linear.weight)),
                                   fp16(const_cast<QwenDeviceTensor&>(linear.scale)), fp32_view(y),
                                   out_rows, cols, cols,
                                   static_cast<int>(linear.scale.shape[1])),
                               "FP8 decode projection");
                return;
            }
            require_launch(qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(
                               x, fp8(const_cast<QwenDeviceTensor&>(linear.weight)),
                               fp16(const_cast<QwenDeviceTensor&>(linear.scale)), fp32_view(y),
                               rows, out_rows, cols, cols, out_rows, cols,
                               static_cast<int>(linear.scale.shape[1])),
                           "FP8 projection");
        } else {
            require_launch(qwen_fp16_matmul_rows_cuda(
                               x, fp16(const_cast<QwenDeviceTensor&>(linear.weight)), fp32_view(y),
                               rows, out_rows, cols, cols, out_rows, cols),
                           "FP16 projection");
        }
    }

    static float* fp32_view(float* p) { return p; }

    void norm(const QwenDeviceTensor& gamma, const float* x, float* y, int rows, int cols) {
        require_launch(qwen_rmsnorm_fp16_gamma_rows_cuda(
                           x, fp16(const_cast<QwenDeviceTensor&>(gamma)), y, rows, cols,
                           static_cast<float>(config.rms_norm_eps)),
                       "Qwen RMSNorm");
    }

    void add(float* y, const float* x, int count) {
        require_launch(qwen_add_inplace_cuda(y, x, count), "Qwen residual add");
    }

    void begin_workspace() { workspace.begin(); }

    QwenDeviceTensor& workspace_float(size_t elements, const std::vector<uint64_t>& shape) {
        return workspace.float_tensor(elements, shape);
    }

    void linear_attention(DeviceLayer& layer, float* hidden, float* output, int rows,
                          int position_offset) {
        const int key_heads = static_cast<int>(config.linear_attention.key_heads / options.tp_world);
        const int value_heads = static_cast<int>(config.linear_attention.value_heads / options.tp_world);
        const int key_dim = key_heads * static_cast<int>(config.linear_attention.key_head_dim);
        const int value_dim = value_heads * static_cast<int>(config.linear_attention.value_head_dim);
        const int packed_dim = 2 * key_dim + value_dim;
        const int kernel = static_cast<int>(config.linear_attention.conv_kernel_dim);
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t packed_elements = static_cast<size_t>(rows) * packed_dim;
        const size_t key_elements = static_cast<size_t>(rows) * key_dim;
        const size_t value_elements = static_cast<size_t>(rows) * value_dim;
        const size_t gate_elements = static_cast<size_t>(rows) * value_heads;
        QwenDeviceTensor& packed = workspace_float(packed_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(packed_dim)});
        QwenDeviceTensor& conv = workspace_float(packed_elements, packed.shape);
        QwenDeviceTensor& q = workspace_float(key_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(key_dim)});
        QwenDeviceTensor& k = workspace_float(key_elements, q.shape);
        QwenDeviceTensor& v = workspace_float(value_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(value_dim)});
        QwenDeviceTensor& a = workspace_float(gate_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(value_heads)});
        QwenDeviceTensor& b = workspace_float(gate_elements, a.shape);
        QwenDeviceTensor& gates = workspace_float(gate_elements, a.shape);
        QwenDeviceTensor& beta = workspace_float(gate_elements, a.shape);
        QwenDeviceTensor& core = workspace_float(value_elements, v.shape);
        QwenDeviceTensor& z = workspace_float(value_elements, core.shape);
        QwenDeviceTensor& normalized = workspace_float(value_elements, core.shape);
        projection(layer.linear.qkv, hidden, fp32(packed), rows);
        require_launch(qwen_causal_depthwise_conv_silu_cuda(
                           fp32(packed), fp16(layer.linear.conv), fp32(layer.linear.conv_tail),
                           fp32(conv), rows, packed_dim, kernel, true),
                       "linear causal convolution");
        require_launch(qwen_split_packed_qkv_cuda(fp32(conv), fp32(q), fp32(k), fp32(v), rows, key_dim, value_dim),
                       "linear QKV split");
        projection(layer.linear.a, hidden, fp32(a), rows);
        projection(layer.linear.b, hidden, fp32(b), rows);
        require_launch(qwen_linear_attn_gates_cuda(
                           fp32(a), fp32(b), fp16(layer.linear.a_log), fp16(layer.linear.dt_bias),
                           fp32(gates), fp32(beta), rows, value_heads),
                       "linear gate projection");
        const float q_scale = 1.0f / std::sqrt(static_cast<float>(config.linear_attention.key_head_dim));
        // Prefill collapses the recurrence into one launch per layer; decode
        // keeps the single-step kernel so its latency path is unchanged. The
        // sequence kernel declines shapes it cannot hold, hence the fallback.
        const bool sequenced =
            rows > 1 &&
            qwen_gated_delta_sequence_cuda(
                fp32(layer.linear.state), fp32(q), fp32(k), fp32(v), fp32(gates), fp32(beta),
                fp32(core), rows, value_heads, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim),
                static_cast<int>(config.linear_attention.value_head_dim), q_scale);
        for (int t = 0; sequenced ? false : t < rows; ++t) {
            require_launch(qwen_gated_delta_step_cuda(
                               fp32(layer.linear.state), fp32(q) + static_cast<size_t>(t) * key_dim,
                               fp32(k) + static_cast<size_t>(t) * key_dim,
                               fp32(v) + static_cast<size_t>(t) * value_dim,
                               fp32(gates) + static_cast<size_t>(t) * value_heads,
                               fp32(beta) + static_cast<size_t>(t) * value_heads,
                               fp32(core) + static_cast<size_t>(t) * value_dim,
                               value_heads, key_heads, static_cast<int>(config.linear_attention.key_head_dim),
                               static_cast<int>(config.linear_attention.value_head_dim), q_scale),
                           "linear recurrent state");
        }
        projection(layer.linear.z, hidden, fp32(z), rows);
        require_launch(qwen_gated_rmsnorm_fp16_gamma_rows_cuda(
                           fp32(core), fp16(layer.linear.norm), fp32(z), fp32(normalized),
                           rows * value_heads, static_cast<int>(config.linear_attention.value_head_dim),
                           static_cast<float>(config.rms_norm_eps)),
                       "linear gated RMSNorm");
        projection(layer.linear.out, fp32(normalized), output, rows);
        all_reduce(output, rows * hidden_size);
        (void)position_offset;
    }

    void full_attention(DeviceLayer& layer, float* hidden, float* output, int rows,
                        int position_offset) {
        const int q_heads = static_cast<int>(config.full_attention.num_heads / options.tp_world);
        const int total_kv_heads = static_cast<int>(config.full_attention.num_key_value_heads);
        const int kv_heads = total_kv_heads / options.tp_world;
        const int head_dim = static_cast<int>(config.full_attention.head_dim);
        const int attention_dim = q_heads * head_dim;
        const int q_proj_dim = attention_dim * 2;
        const int kv_dim = total_kv_heads * head_dim;
        const size_t q_proj_elements = static_cast<size_t>(rows) * q_proj_dim;
        const size_t attention_elements = static_cast<size_t>(rows) * attention_dim;
        const size_t kv_full_elements = static_cast<size_t>(rows) * kv_dim;
        const size_t kv_elements = static_cast<size_t>(rows) * kv_heads * head_dim;
        QwenDeviceTensor& q_proj = workspace_float(q_proj_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_proj_dim)});
        QwenDeviceTensor& q = workspace_float(attention_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(attention_dim)});
        QwenDeviceTensor& gate = workspace_float(attention_elements, q.shape);
        QwenDeviceTensor& k_full = workspace_float(kv_full_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_dim)});
        QwenDeviceTensor& v_full = workspace_float(kv_full_elements, k_full.shape);
        QwenDeviceTensor& k = workspace_float(kv_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_heads), static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v = workspace_float(kv_elements, k.shape);
        QwenDeviceTensor& q_norm = workspace_float(attention_elements, q.shape);
        QwenDeviceTensor& k_norm = workspace_float(kv_elements, k.shape);
        QwenDeviceTensor& attn = workspace_float(attention_elements, q.shape);
        QwenDeviceTensor& merged = workspace_float(attention_elements, q.shape);
        projection(layer.full.q, hidden, fp32(q_proj), rows);
        require_launch(qwen_split_q_gate_cuda(fp32(q_proj), fp32(q), fp32(gate), rows, q_heads, head_dim), "full Q/gate split");
        projection(layer.full.k, hidden, fp32(k_full), rows);
        projection(layer.full.v, hidden, fp32(v_full), rows);
        const int head_offset = (options.tp_rank * kv_heads);
        require_launch(qwen_select_kv_heads_cuda(fp32(k_full), fp32(k), rows, total_kv_heads, kv_heads,
                                                 head_dim, head_offset), "full K head select");
        require_launch(qwen_select_kv_heads_cuda(fp32(v_full), fp32(v), rows, total_kv_heads, kv_heads,
                                                 head_dim, head_offset), "full V head select");
        norm(layer.full.q_norm, fp32(q), fp32(q_norm), rows * q_heads, head_dim);
        norm(layer.full.k_norm, fp32(k), fp32(k_norm), rows * kv_heads, head_dim);
        if (rows == 1) {
            require_launch(qwen_partial_rope_cuda(
                               fp32(q_norm), fp32(k_norm), position_offset,
                               static_cast<int>(config.partial_rotary_dim()),
                               static_cast<float>(config.rope_theta), q_heads, kv_heads, head_dim),
                           "partial RoPE");
        } else {
            require_launch(qwen_partial_rope_rows_cuda(
                               fp32(q_norm), fp32(k_norm), position_offset, rows,
                               static_cast<int>(config.partial_rotary_dim()),
                               static_cast<float>(config.rope_theta), q_heads, kv_heads, head_dim),
                           "prefill partial RoPE");
        }
        require_launch(qwen_append_kv_cache_cuda(
                           fp32(k_norm), fp32(v), fp32(layer.full.k_cache), fp32(layer.full.v_cache),
                           rows, kv_heads, head_dim, position_offset, max_context),
                       "append full KV cache");
        if (rows == 1) {
            require_launch(qwen_gqa_decode_attention_cuda(
                               fp32(q_norm), fp32(layer.full.k_cache), fp32(layer.full.v_cache),
                               fp32(attn), q_heads, kv_heads, head_dim, position_offset + 1, max_context),
                           "decode GQA");
        } else {
            require_launch(qwen_gqa_prefill_attention_cuda(
                               fp32(q_norm), fp32(layer.full.k_cache), fp32(layer.full.v_cache),
                               fp32(attn), rows, q_heads, kv_heads, head_dim, position_offset, max_context),
                           "prefill GQA");
        }
        require_launch(qwen_sigmoid_mul_cuda(fp32(attn), fp32(gate), fp32(merged), rows * attention_dim),
                       "full attention output gate");
        projection(layer.full.out, fp32(merged), output, rows);
        all_reduce(output, rows * static_cast<int>(config.hidden_size));
    }

    void layer_forward(DeviceLayer& layer, float* hidden, float* work, int rows, int position_offset) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        begin_workspace();
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const size_t intermediate_elements = static_cast<size_t>(rows) * layer.gate.weight.shape[0];
        QwenDeviceTensor& normed = workspace_float(hidden_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)});
        QwenDeviceTensor& attn = workspace_float(hidden_elements, normed.shape);
        QwenDeviceTensor& post = workspace_float(hidden_elements, normed.shape);
        QwenDeviceTensor& gate = workspace_float(intermediate_elements, {static_cast<uint64_t>(rows), layer.gate.weight.shape[0]});
        QwenDeviceTensor& up = workspace_float(intermediate_elements, gate.shape);
        QwenDeviceTensor& intermediate = workspace_float(intermediate_elements, gate.shape);
        QwenDeviceTensor& mlp = workspace_float(hidden_elements, normed.shape);
        norm(layer.input_norm, hidden, fp32(normed), rows, hidden_size);
        if (layer.linear.qkv.weight.data != nullptr) {
            linear_attention(layer, fp32(normed), fp32(attn), rows, position_offset);
        } else {
            full_attention(layer, fp32(normed), fp32(attn), rows, position_offset);
        }
        check_cuda(cudaMemcpy(work, hidden, static_cast<size_t>(rows) * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Qwen residual copy");
        add(work, fp32(attn), rows * hidden_size);
        norm(layer.post_norm, work, fp32(post), rows, hidden_size);
        projection(layer.gate, fp32(post), fp32(gate), rows);
        projection(layer.up, fp32(post), fp32(up), rows);
        require_launch(qwen_silu_mul_rows_cuda(fp32(gate), fp32(up), fp32(intermediate), rows,
                                               static_cast<int>(layer.gate.weight.shape[0])),
                       "Qwen SwiGLU");
        projection(layer.down, fp32(intermediate), fp32(mlp), rows);
        all_reduce(fp32(mlp), rows * hidden_size);
        add(work, fp32(mlp), rows * hidden_size);
        check_cuda(cudaMemcpy(hidden, work, static_cast<size_t>(rows) * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Qwen layer output copy");
    }

    void run_layers(float*& hidden, float*& work, int rows, int position_offset, int active_layers) {
        for (int i = 0; i < active_layers; ++i) {
            layer_forward(layers[static_cast<size_t>(i)], hidden, work, rows, position_offset);
            std::swap(hidden, work);
        }
    }

    QwenForwardResult logits_for(float* hidden, int rows, int last_row, int position_after, int active_layers) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        const int local_vocab = static_cast<int>(lm_head.shape[0]);
        begin_workspace();
        QwenDeviceTensor& normed = workspace_float(static_cast<size_t>(rows) * hidden_size,
                                                    {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)});
        norm(final_norm, hidden, fp32(normed), rows, hidden_size);
        QwenDeviceTensor& selected = workspace_float(hidden_size, {static_cast<uint64_t>(hidden_size)});
        check_cuda(cudaMemcpy(selected.data, static_cast<uint8_t*>(normed.data) +
                                  static_cast<size_t>(last_row) * hidden_size * sizeof(float),
                              hidden_size * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Qwen final row copy");
        QwenDeviceTensor& local_logits = workspace_float(local_vocab, {static_cast<uint64_t>(local_vocab)});
        if (lm_head.device_dtype == SafeDType::F16) {
            require_launch(qwen_fp16_matmul_rows_cuda(
                               fp32(selected), fp16(lm_head), fp32(local_logits), 1, local_vocab,
                               hidden_size, hidden_size, local_vocab, hidden_size),
                           "Qwen lm head");
        } else {
            throw std::runtime_error("Qwen lm_head must be FP16 on Turing");
        }
        allocate(argmax_token, sizeof(int), {1}, SafeDType::I64);
        allocate(argmax_logit, sizeof(float), {1}, SafeDType::F32);
        const int vocab_start = static_cast<int>(weights_vocab_start());
        require_launch(argmax_fp32_cuda(fp32(local_logits), static_cast<int*>(argmax_token.data),
                                        fp32(argmax_logit), local_vocab, vocab_start),
                       "Qwen local argmax");
        check_cuda(cudaDeviceSynchronize(), "Qwen logits synchronization");
        int local_token = 0;
        float local_logit = 0.0f;
        check_cuda(cudaMemcpy(&local_token, argmax_token.data, sizeof(local_token), cudaMemcpyDeviceToHost),
                   "Qwen argmax token copy");
        check_cuda(cudaMemcpy(&local_logit, argmax_logit.data, sizeof(local_logit), cudaMemcpyDeviceToHost),
                   "Qwen argmax logit copy");
        int top_token = local_token;
        float top_logit = local_logit;
#ifdef DSV4_HAVE_NCCL
        if (options.tp_world > 1) {
            if (options.nccl_id_path.empty()) throw std::runtime_error("Qwen TP requires --nccl-id-path");
            const TpTopResult global = nccl_global_top1(options.tp_world, options.tp_rank, options.device,
                                                        options.nccl_id_path.c_str(), local_token, local_logit);
            top_token = global.token;
            top_logit = global.logit;
        }
#else
        if (options.tp_world > 1) throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
#endif
        float checksum = local_logit;
        QwenForwardResult result;
        result.token = last_row >= 0 ? 0 : 0;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = top_token;
        result.top_logit = top_logit;
        result.checksum = checksum;
        result.position = position_after;
        return result;
    }

    uint64_t weights_vocab_start() const {
        return static_cast<uint64_t>(options.tp_rank) * config.vocab_size / options.tp_world;
    }

    QwenForwardResult run(const std::vector<int>& token_ids, int position_offset, int active_layers,
                          int last_row) {
        if (token_ids.empty()) throw std::runtime_error("Qwen forward requires at least one token");
        if (position_offset < 0 || position_offset + static_cast<int>(token_ids.size()) > max_context) {
            throw std::runtime_error("Qwen context length exceeded");
        }
        const int rows = static_cast<int>(token_ids.size());
        local_tokens = token_ids;
        allocate(d_tokens, local_tokens.size() * sizeof(int), {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_cuda(cudaMemcpy(d_tokens.data, local_tokens.data(), local_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "Qwen token upload");
        const int hidden_size = static_cast<int>(config.hidden_size);
        allocate_float(hidden_a, static_cast<size_t>(rows) * hidden_size, {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)});
        allocate_float(hidden_b, hidden_a.nbytes / sizeof(float), hidden_a.shape);
        allocate_float(work_a, hidden_a.nbytes / sizeof(float), hidden_a.shape);
        allocate_float(work_b, hidden_a.nbytes / sizeof(float), hidden_a.shape);
        require_launch(qwen_embedding_fp16_gather_cuda(
                           fp16(embed), static_cast<int*>(d_tokens.data), fp32(hidden_a), rows, hidden_size,
                           static_cast<int>(weights_vocab_start()), static_cast<int>(embed.shape[0])),
                       "Qwen embedding lookup");
        all_reduce(fp32(hidden_a), rows * hidden_size);
        float* hidden = fp32(hidden_a);
        float* work = fp32(hidden_b);
        for (int i = 0; i < active_layers; ++i) {
            layer_forward(layers[static_cast<size_t>(i)], hidden, work, rows, position_offset);
            std::swap(hidden, work);
        }
        QwenForwardResult result = logits_for(hidden, rows, last_row, position_offset + rows, active_layers);
        return result;
    }
};

QwenEngine::QwenEngine(const std::string& ckpt_dir, const QwenEngineOptions& options,
                       int layer_count, int max_context)
    : ckpt_dir_(ckpt_dir), options_(options), config_(QwenConfig::from_hf_config(ckpt_dir)),
      index_(ckpt_dir), weights_(index_, config_, options.tp_world, options.tp_rank) {
    if (options_.device < 0) options_.device = options_.tp_rank;
    if (cudaSetDevice(options_.device) != cudaSuccess) throw std::runtime_error("failed to set Qwen CUDA device");
    if (layer_count <= 0) active_layers_ = static_cast<int>(config_.num_hidden_layers);
    else if (layer_count <= static_cast<int>(config_.num_hidden_layers)) active_layers_ = layer_count;
    else throw std::runtime_error("Qwen layer count exceeds model depth");
    max_context_ = max_context > 0 ? max_context : static_cast<int>(config_.max_position_embeddings);
    if (max_context_ > static_cast<int>(config_.max_position_embeddings)) {
        throw std::runtime_error("Qwen max context exceeds model configuration");
    }
    impl_ = new Impl(index_, config_, weights_, options_, max_context_, active_layers_);
    resident_weight_bytes_ = impl_->uploaded_weight_bytes;
    resident_scale_bytes_ = impl_->uploaded_scale_bytes;
}

QwenEngine::~QwenEngine() {
    delete impl_;
    impl_ = nullptr;
}

void QwenEngine::warmup_tp() {
    if (options_.tp_world == 1) return;
    QwenDeviceTensor scratch;
    allocate_float(scratch, 1, {1});
    zero_tensor(scratch);
    impl_->all_reduce(fp32(scratch), 1);
    check_cuda(cudaDeviceSynchronize(), "Qwen TP warmup synchronization");
}

void QwenEngine::reset() {
    position_ = 0;
    for (DeviceLayer& layer : impl_->layers) {
        if (layer.linear.state.data != nullptr) {
            zero_tensor(layer.linear.state);
            zero_tensor(layer.linear.conv_tail);
        }
        if (layer.full.k_cache.data != nullptr) {
            zero_tensor(layer.full.k_cache);
            zero_tensor(layer.full.v_cache);
        }
    }
}

QwenForwardResult QwenEngine::prefill(const std::vector<int>& token_ids) {
    reset();
    QwenForwardResult result = impl_->run(token_ids, 0, active_layers_, static_cast<int>(token_ids.size()) - 1);
    position_ = static_cast<int>(token_ids.size());
    return result;
}

QwenForwardResult QwenEngine::decode_step(int token_id) {
    QwenForwardResult result = impl_->run({token_id}, position_, active_layers_, 0);
    ++position_;
    return result;
}

std::vector<QwenForwardResult> QwenEngine::generate(const std::vector<int>& prompt_ids,
                                                    int max_new_tokens) {
    if (max_new_tokens <= 0) return {};
    QwenForwardResult next = prefill(prompt_ids);
    std::vector<QwenForwardResult> results;
    results.reserve(static_cast<size_t>(max_new_tokens));
    for (int i = 0; i < max_new_tokens; ++i) {
        results.push_back(next);
        if (i + 1 < max_new_tokens) next = decode_step(next.top_token);
    }
    return results;
}

}  // namespace dsv4
