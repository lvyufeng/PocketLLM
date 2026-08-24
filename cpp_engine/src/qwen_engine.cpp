#include "qwen_engine.hpp"

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"
#include "qwen_dspark.hpp"
#include "qwen_dflash2.hpp"
#include "tp_comm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace dsv4 {
namespace {

constexpr int kKvScaleBlock = 64;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require_launch(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("Qwen CUDA launch failed: ") + what);
}

bool qwen_env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return false;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

int qwen_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1'000'000) {
        return fallback;
    }
    return static_cast<int>(parsed);
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
    QwenDeviceTensor k_scale;
    QwenDeviceTensor v_scale;
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
    DeviceLinear output;
    output.weight = upload(index, ref.weight);
    output.fp8 = ref.weight.device_dtype == SafeDType::F8_E4M3;
    if (ref.has_scale) output.scale = upload(index, ref.scale);
    return output;
}

void allocate(QwenDeviceTensor& tensor, size_t bytes,
              const std::vector<uint64_t>& shape, SafeDType dtype) {
    if (bytes == 0 || safe_dtype_size(dtype) == 0) {
        throw std::runtime_error("Qwen attempted to allocate an invalid tensor");
    }
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

void allocate_elements(QwenDeviceTensor& tensor, size_t elements,
                       const std::vector<uint64_t>& shape, SafeDType dtype) {
    const uint64_t item_size = safe_dtype_size(dtype);
    if (elements > static_cast<size_t>(UINT64_MAX / item_size)) {
        throw std::runtime_error("Qwen tensor byte extent overflow");
    }
    allocate(tensor, elements * item_size, shape, dtype);
}

void allocate_float(QwenDeviceTensor& tensor, size_t elements,
                    const std::vector<uint64_t>& shape) {
    allocate_elements(tensor, elements, shape, SafeDType::F32);
}

void allocate_half(QwenDeviceTensor& tensor, size_t elements,
                   const std::vector<uint64_t>& shape) {
    allocate_elements(tensor, elements, shape, SafeDType::F16);
}

void zero_tensor(QwenDeviceTensor& tensor) {
    if (tensor.data != nullptr && tensor.nbytes != 0) {
        check_cuda(cudaMemset(tensor.data, 0, tensor.nbytes),
                   "cudaMemset Qwen runtime tensor");
    }
}

// One rollback point for the recurrent half of the network. The 16 full
// attention layers need nothing here because their KV cache is indexed by
// absolute position and is never invalidated by a longer prompt.
struct QwenRecurrentSnapshot {
    int position = 0;
    // Device-resident contiguous copies avoid a PCIe round trip and thousands
    // of small allocations. The tensors are local to one TP rank.
    QwenDeviceTensor state;
    QwenDeviceTensor conv_tail;
    QwenDeviceTensor target_hidden;
    bool periodic = false;
    // A request-boundary snapshot can answer an exact shorter-prefix prefill
    // without re-running its final token.
    QwenForwardResult result;
    bool has_result = false;

    uint64_t bytes() const {
        return state.capacity + conv_tail.capacity + target_hidden.capacity;
    }
};

struct QwenVerifyBatch {
    std::vector<int> top_tokens;
    std::vector<float> top_logits;
    std::vector<float> local_logits;
    int position_after = 0;
};

struct QwenWorkspace {
    std::vector<QwenDeviceTensor> slots;
    size_t cursor = 0;

    QwenWorkspace() { slots.reserve(32); }

    void begin() { cursor = 0; }

    QwenDeviceTensor& tensor(size_t elements, const std::vector<uint64_t>& shape,
                             SafeDType dtype) {
        if (cursor == slots.size()) slots.emplace_back();
        QwenDeviceTensor& output = slots[cursor++];
        allocate_elements(output, elements, shape, dtype);
        return output;
    }

    QwenDeviceTensor& half_tensor(size_t elements,
                                  const std::vector<uint64_t>& shape) {
        return tensor(elements, shape, SafeDType::F16);
    }

    QwenDeviceTensor& float_tensor(size_t elements,
                                   const std::vector<uint64_t>& shape) {
        return tensor(elements, shape, SafeDType::F32);
    }

    uint64_t capacity_bytes() const {
        uint64_t total = 0;
        for (const QwenDeviceTensor& slot : slots) total += slot.capacity;
        return total;
    }
};

}  // namespace

const char* qwen_kv_cache_dtype_name(QwenKvCacheDType dtype) {
    switch (dtype) {
        case QwenKvCacheDType::Fp16: return "fp16";
        case QwenKvCacheDType::Fp8: return "fp8";
    }
    return "unknown";
}

QwenKvCacheDType parse_qwen_kv_cache_dtype(const std::string& value) {
    if (value == "fp16") return QwenKvCacheDType::Fp16;
    if (value == "fp8") return QwenKvCacheDType::Fp8;
    throw std::runtime_error("Qwen KV cache dtype must be fp16 or fp8");
}

struct QwenEngine::Impl {
    SafeTensorsIndex& index;
    QwenConfig& config;
    QwenEngineOptions options;
    int max_context = 0;
    std::vector<DeviceLayer> layers;
    std::vector<QwenDeviceTensor> transaction_states;
    std::vector<QwenDeviceTensor> transaction_conv_tails;
    QwenDeviceTensor transaction_target_hidden;
    QwenDeviceTensor embed;
    QwenDeviceTensor final_norm;
    QwenDeviceTensor lm_head;
    bool mtp_enabled = false;
    bool dspark_enabled = false;
    bool dflash2_enabled = false;
    // Caps the verified draft block width. 0 verifies the full seven-draft
    // proposal, which is the upstream behaviour and stays the default.
    int dflash2_draft_width = qwen_env_int("DSV4_DFLASH2_DRAFT_WIDTH", 0);
    // A fixed cap cannot serve both high- and low-acceptance workloads, so the
    // width is also allowed to track measured acceptance. The controller keeps an
    // exponentially weighted accepted-draft count and verifies one row past it,
    // which is the cheapest width that can still grow when acceptance recovers.
    bool dflash2_adaptive_width = qwen_env_enabled("DSV4_DFLASH2_ADAPTIVE_WIDTH");
    double dflash2_accept_ewma = 0.0;
    uint64_t dflash2_width_samples = 0;
    std::unique_ptr<QwenDSparkConfig> dspark_config;
    std::unique_ptr<QwenDFlash2Config> dflash2_config;
    std::unique_ptr<SafeTensorsIndex> dflash2_index;
    std::unique_ptr<QwenDFlash2WeightMap> dflash2_weights;
    std::unique_ptr<QwenDFlash2Runtime> dflash2;
    QwenDeviceTensor dflash2_target_taps;
    std::vector<int> dflash2_debug_target_layer_ids;
    QwenDFlash2DebugCallback dflash2_target_debug_callback;
    std::unique_ptr<SafeTensorsIndex> dspark_index;
    std::unique_ptr<QwenDSparkWeightMap> dspark_weights;
    std::unique_ptr<QwenDSparkRuntime> dspark;
    QwenDeviceTensor dspark_target_taps;
    DeviceLinear mtp_fc;
    QwenDeviceTensor mtp_pre_fc_norm_embedding;
    QwenDeviceTensor mtp_pre_fc_norm_hidden;
    QwenDeviceTensor mtp_norm;
    DeviceLayer mtp_layer;
    QwenDeviceTensor target_hidden_rows;
    QwenDeviceTensor target_last_hidden;
    QwenDeviceTensor mtp_seed_hidden;
    QwenDeviceTensor mtp_embedding;
    QwenDeviceTensor mtp_normalized_embedding;
    QwenDeviceTensor mtp_normalized_hidden;
    QwenDeviceTensor mtp_concat;
    QwenDeviceTensor mtp_fused;
    QwenDeviceTensor mtp_next_hidden;
    QwenDeviceTensor mtp_normalized_output;
    int mtp_position = 0;
    int mtp_seed_input_token = 0;
    int mtp_next_token = 0;
    float mtp_next_logit = 0.0f;
    float mtp_next_checksum = 0.0f;
    bool has_target_last_hidden = false;
    bool mtp_seed_ready = false;
    std::vector<int> local_tokens;
    QwenDeviceTensor d_tokens;
    QwenDeviceTensor hidden_a;
    QwenDeviceTensor hidden_b;
    QwenDeviceTensor argmax_token;
    QwenDeviceTensor argmax_logit;
    QwenWorkspace workspace;
    uint64_t uploaded_weight_bytes = 0;
    uint64_t uploaded_scale_bytes = 0;
    uint64_t cache_data_bytes = 0;
    uint64_t cache_scale_bytes = 0;
    // Prompt whose KV cache and recurrent state are currently materialized.
    std::vector<int> cached_prompt;
    // Ordered by ascending position. Index 0 is always the implicit empty
    // state, which is not stored.
    std::vector<QwenRecurrentSnapshot> snapshots;
    int prefix_hits = 0;
    int prefix_misses = 0;

    Impl(SafeTensorsIndex& index_, QwenConfig& config_,
         const QwenWeightMap& map, const QwenEngineOptions& options_,
         int max_context_, int active_layers)
        : index(index_), config(config_), options(options_),
          max_context(max_context_) {
        embed = upload(index, map.embed_tokens());
        final_norm = upload(index, map.final_norm());
        lm_head = upload(index, map.lm_head());
        uploaded_weight_bytes += map.embed_tokens().device_nbytes +
                                 map.final_norm().device_nbytes +
                                 map.lm_head().device_nbytes;
        const size_t layer_limit = active_layers > 0
            ? std::min(static_cast<size_t>(active_layers), map.layers().size())
            : map.layers().size();
        layers.reserve(layer_limit);
        const int world = options.tp_world;
        const int local_key_heads =
            static_cast<int>(config.linear_attention.key_heads / world);
        const int local_value_heads =
            static_cast<int>(config.linear_attention.value_heads / world);
        const int local_key_dim = local_key_heads *
            static_cast<int>(config.linear_attention.key_head_dim);
        const int local_value_dim = local_value_heads *
            static_cast<int>(config.linear_attention.value_head_dim);
        const int local_qkv_dim = 2 * local_key_dim + local_value_dim;
        const int local_kv_heads =
            static_cast<int>(config.full_attention.num_key_value_heads / world);
        const int head_dim = static_cast<int>(config.full_attention.head_dim);

        for (size_t layer_index = 0; layer_index < layer_limit; ++layer_index) {
            const QwenLayerWeights& source = map.layers()[layer_index];
            for (const QwenTensorRef* ref : {
                     &source.input_layernorm, &source.post_attention_layernorm,
                     &source.mlp.gate_proj.weight, &source.mlp.up_proj.weight,
                     &source.mlp.down_proj.weight,
                     &source.linear_attention.in_proj_qkv.weight,
                     &source.linear_attention.in_proj_z.weight,
                     &source.linear_attention.out_proj.weight,
                     &source.linear_attention.in_proj_a.weight,
                     &source.linear_attention.in_proj_b.weight,
                     &source.linear_attention.conv1d,
                     &source.linear_attention.a_log,
                     &source.linear_attention.dt_bias,
                     &source.linear_attention.norm,
                     &source.full_attention.q_proj.weight,
                     &source.full_attention.k_proj.weight,
                     &source.full_attention.v_proj.weight,
                     &source.full_attention.o_proj.weight,
                     &source.full_attention.q_norm,
                     &source.full_attention.k_norm}) {
                if (ref->found) uploaded_weight_bytes += ref->device_nbytes;
            }
            for (const QwenLinearRef* ref : {
                     &source.mlp.gate_proj, &source.mlp.up_proj,
                     &source.mlp.down_proj,
                     &source.linear_attention.in_proj_qkv,
                     &source.linear_attention.in_proj_z,
                     &source.linear_attention.out_proj,
                     &source.full_attention.q_proj,
                     &source.full_attention.k_proj,
                     &source.full_attention.v_proj,
                     &source.full_attention.o_proj}) {
                if (ref->has_scale) {
                    uploaded_scale_bytes += ref->scale.device_nbytes;
                }
            }

            DeviceLayer destination;
            destination.input_norm = upload(index, source.input_layernorm);
            destination.post_norm = upload(index, source.post_attention_layernorm);
            destination.gate = upload_linear(index, source.mlp.gate_proj);
            destination.up = upload_linear(index, source.mlp.up_proj);
            destination.down = upload_linear(index, source.mlp.down_proj);
            if (source.linear_attention.in_proj_qkv.weight.found) {
                destination.linear.qkv =
                    upload_linear(index, source.linear_attention.in_proj_qkv);
                destination.linear.z =
                    upload_linear(index, source.linear_attention.in_proj_z);
                destination.linear.out =
                    upload_linear(index, source.linear_attention.out_proj);
                destination.linear.a =
                    upload_linear(index, source.linear_attention.in_proj_a);
                destination.linear.b =
                    upload_linear(index, source.linear_attention.in_proj_b);
                destination.linear.conv =
                    upload(index, source.linear_attention.conv1d);
                destination.linear.a_log =
                    upload(index, source.linear_attention.a_log);
                destination.linear.dt_bias =
                    upload(index, source.linear_attention.dt_bias);
                destination.linear.norm =
                    upload(index, source.linear_attention.norm);
                allocate_float(destination.linear.state,
                    static_cast<size_t>(local_value_heads) *
                        config.linear_attention.key_head_dim *
                        config.linear_attention.value_head_dim,
                    {static_cast<uint64_t>(local_value_heads),
                     config.linear_attention.key_head_dim,
                     config.linear_attention.value_head_dim});
                allocate_half(destination.linear.conv_tail,
                    static_cast<size_t>(std::max(
                        0, static_cast<int>(config.linear_attention.conv_kernel_dim) - 1)) *
                        local_qkv_dim,
                    {static_cast<uint64_t>(std::max(
                         0, static_cast<int>(config.linear_attention.conv_kernel_dim) - 1)),
                     static_cast<uint64_t>(local_qkv_dim)});
                zero_tensor(destination.linear.state);
                zero_tensor(destination.linear.conv_tail);
            } else {
                destination.full.q = upload_linear(index, source.full_attention.q_proj);
                destination.full.k = upload_linear(index, source.full_attention.k_proj);
                destination.full.v = upload_linear(index, source.full_attention.v_proj);
                destination.full.out = upload_linear(index, source.full_attention.o_proj);
                destination.full.q_norm = upload(index, source.full_attention.q_norm);
                destination.full.k_norm = upload(index, source.full_attention.k_norm);
                const size_t cache_elements = static_cast<size_t>(max_context) *
                    local_kv_heads * head_dim;
                const std::vector<uint64_t> cache_shape = {
                    static_cast<uint64_t>(max_context),
                    static_cast<uint64_t>(local_kv_heads),
                    static_cast<uint64_t>(head_dim)};
                if (options.kv_cache_dtype == QwenKvCacheDType::Fp16) {
                    allocate_half(destination.full.k_cache, cache_elements, cache_shape);
                    allocate_half(destination.full.v_cache, cache_elements, cache_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                } else {
                    allocate_elements(destination.full.k_cache, cache_elements,
                                      cache_shape, SafeDType::F8_E4M3);
                    allocate_elements(destination.full.v_cache, cache_elements,
                                      cache_shape, SafeDType::F8_E4M3);
                    const size_t scale_elements = static_cast<size_t>(max_context) *
                        local_kv_heads * (head_dim / kKvScaleBlock);
                    const std::vector<uint64_t> scale_shape = {
                        static_cast<uint64_t>(max_context),
                        static_cast<uint64_t>(local_kv_heads),
                        static_cast<uint64_t>(head_dim / kKvScaleBlock)};
                    allocate_half(destination.full.k_scale, scale_elements, scale_shape);
                    allocate_half(destination.full.v_scale, scale_elements, scale_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                    cache_scale_bytes += destination.full.k_scale.nbytes +
                                         destination.full.v_scale.nbytes;
                }
            }
            layers.push_back(std::move(destination));
        }

        if (!options.dspark_checkpoint.empty()) {
            if (active_layers > 0 &&
                active_layers != static_cast<int>(config.num_hidden_layers)) {
                throw std::runtime_error(
                    "Qwen DSpark requires the complete target layer stack");
            }
            dspark_config = std::make_unique<QwenDSparkConfig>(
                QwenDSparkConfig::from_directory(options.dspark_checkpoint));
            dspark_config->validate_for_target(
                config.hidden_size, config.vocab_size, config.num_hidden_layers);
            dspark_index = std::make_unique<SafeTensorsIndex>(
                SafeTensorsIndex::from_single_file(options.dspark_checkpoint));
            dspark_weights = std::make_unique<QwenDSparkWeightMap>(
                *dspark_index, *dspark_config, options.tp_world, options.tp_rank);
            // The target embedding is vocab-sharded. QwenDSparkRuntime gathers
            // local rows then performs the same TP all-reduce as target prefill.
            dspark = std::make_unique<QwenDSparkRuntime>(
                options.dspark_checkpoint, *dspark_config, *dspark_weights,
                embed, lm_head,
                static_cast<uint64_t>(options.tp_rank) * config.vocab_size /
                    options.tp_world,
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path, max_context);
            dspark_enabled = true;
            uploaded_weight_bytes += dspark->resident_weight_bytes();
            cache_data_bytes += dspark->context_cache_bytes();
        }
        if (!options.dflash2_checkpoint.empty()) {
            if (active_layers > 0 &&
                active_layers != static_cast<int>(config.num_hidden_layers)) {
                throw std::runtime_error(
                    "Qwen DFlash2 requires the complete target layer stack");
            }
            dflash2_config = std::make_unique<QwenDFlash2Config>(
                QwenDFlash2Config::from_directory(options.dflash2_checkpoint));
            dflash2_config->validate_for_target(
                config.hidden_size, config.vocab_size, config.num_hidden_layers);
            dflash2_index = std::make_unique<SafeTensorsIndex>(
                SafeTensorsIndex::from_single_file(options.dflash2_checkpoint));
            dflash2_weights = std::make_unique<QwenDFlash2WeightMap>(
                *dflash2_index, *dflash2_config, options.tp_world, options.tp_rank);
            dflash2 = std::make_unique<QwenDFlash2Runtime>(
                options.dflash2_checkpoint, *dflash2_config, *dflash2_weights,
                embed, lm_head,
                static_cast<uint64_t>(options.tp_rank) * config.vocab_size /
                    options.tp_world,
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path, max_context);
            dflash2_enabled = true;
            uploaded_weight_bytes += dflash2->resident_weight_bytes();
            cache_data_bytes += dflash2->context_cache_bytes();
        }

        if (options.mtp) {
            if (!map.mtp().found) {
                throw std::runtime_error(
                    "Qwen MTP requested but checkpoint has no native MTP weights");
            }
            if (config.mtp_num_hidden_layers != 1) {
                throw std::runtime_error(
                    "Qwen runtime currently supports exactly one native MTP layer");
            }
            mtp_enabled = true;
            const QwenMtpWeights& source = map.mtp();
            mtp_pre_fc_norm_embedding = upload(index, source.pre_fc_norm_embedding);
            mtp_pre_fc_norm_hidden = upload(index, source.pre_fc_norm_hidden);
            mtp_norm = upload(index, source.norm);
            mtp_fc = upload_linear(index, source.fc);
            const QwenLayerWeights& mtp_source = source.layer;
            mtp_layer.input_norm = upload(index, mtp_source.input_layernorm);
            mtp_layer.post_norm = upload(index, mtp_source.post_attention_layernorm);
            mtp_layer.gate = upload_linear(index, mtp_source.mlp.gate_proj);
            mtp_layer.up = upload_linear(index, mtp_source.mlp.up_proj);
            mtp_layer.down = upload_linear(index, mtp_source.mlp.down_proj);
            mtp_layer.full.q = upload_linear(index, mtp_source.full_attention.q_proj);
            mtp_layer.full.k = upload_linear(index, mtp_source.full_attention.k_proj);
            mtp_layer.full.v = upload_linear(index, mtp_source.full_attention.v_proj);
            mtp_layer.full.out = upload_linear(index, mtp_source.full_attention.o_proj);
            mtp_layer.full.q_norm = upload(index, mtp_source.full_attention.q_norm);
            mtp_layer.full.k_norm = upload(index, mtp_source.full_attention.k_norm);
            uploaded_weight_bytes += source.pre_fc_norm_embedding.device_nbytes +
                source.pre_fc_norm_hidden.device_nbytes + source.norm.device_nbytes +
                mtp_source.input_layernorm.device_nbytes +
                mtp_source.post_attention_layernorm.device_nbytes +
                mtp_source.full_attention.q_norm.device_nbytes +
                mtp_source.full_attention.k_norm.device_nbytes;
            const auto count_mtp_linear = [this](const QwenLinearRef& linear) {
                uploaded_weight_bytes += linear.weight.device_nbytes;
                if (linear.has_scale) {
                    uploaded_scale_bytes += linear.scale.device_nbytes;
                }
            };
            count_mtp_linear(source.fc);
            for (const QwenLinearRef* linear : {
                     &mtp_source.full_attention.q_proj,
                     &mtp_source.full_attention.k_proj,
                     &mtp_source.full_attention.v_proj,
                     &mtp_source.full_attention.o_proj,
                     &mtp_source.mlp.gate_proj,
                     &mtp_source.mlp.up_proj,
                     &mtp_source.mlp.down_proj}) {
                count_mtp_linear(*linear);
            }
            const size_t mtp_cache_elements = static_cast<size_t>(max_context) *
                local_kv_heads * head_dim;
            const std::vector<uint64_t> mtp_cache_shape = {
                static_cast<uint64_t>(max_context),
                static_cast<uint64_t>(local_kv_heads),
                static_cast<uint64_t>(head_dim)};
            if (options.kv_cache_dtype == QwenKvCacheDType::Fp16) {
                allocate_half(mtp_layer.full.k_cache, mtp_cache_elements,
                              mtp_cache_shape);
                allocate_half(mtp_layer.full.v_cache, mtp_cache_elements,
                              mtp_cache_shape);
                cache_data_bytes += mtp_layer.full.k_cache.nbytes +
                                    mtp_layer.full.v_cache.nbytes;
            } else {
                allocate_elements(mtp_layer.full.k_cache, mtp_cache_elements,
                                  mtp_cache_shape, SafeDType::F8_E4M3);
                allocate_elements(mtp_layer.full.v_cache, mtp_cache_elements,
                                  mtp_cache_shape, SafeDType::F8_E4M3);
                const size_t scale_elements = static_cast<size_t>(max_context) *
                    local_kv_heads * (head_dim / kKvScaleBlock);
                const std::vector<uint64_t> scale_shape = {
                    static_cast<uint64_t>(max_context),
                    static_cast<uint64_t>(local_kv_heads),
                    static_cast<uint64_t>(head_dim / kKvScaleBlock)};
                allocate_half(mtp_layer.full.k_scale, scale_elements, scale_shape);
                allocate_half(mtp_layer.full.v_scale, scale_elements, scale_shape);
                cache_data_bytes += mtp_layer.full.k_cache.nbytes +
                                    mtp_layer.full.v_cache.nbytes;
                cache_scale_bytes += mtp_layer.full.k_scale.nbytes +
                                     mtp_layer.full.v_scale.nbytes;
            }
        }
    }

    bool has_recurrent_state() const {
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data != nullptr) return true;
        }
        return false;
    }

    void zero_recurrent_state() {
        for (DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            zero_tensor(layer.linear.state);
            zero_tensor(layer.linear.conv_tail);
        }
        if (dspark_enabled) dspark->reset();
        if (dflash2_enabled) dflash2->reset();
        has_target_last_hidden = false;
        mtp_seed_ready = false;
        mtp_position = 0;
        mtp_seed_input_token = 0;
        mtp_next_token = 0;
        mtp_next_logit = 0.0f;
        mtp_next_checksum = 0.0f;
    }

    // Copies the recurrent half of the network to device-resident snapshots.
    // Only the 48 DeltaNet layers carry order-dependent state; full attention
    // is skipped because its KV cache is addressed by absolute position.
    void prepare_transaction_state() {
        PhaseScope scope(this, "transaction_snapshot");
        if (transaction_states.size() != layers.size()) {
            transaction_states.clear();
            transaction_conv_tails.clear();
            transaction_states.resize(layers.size());
            transaction_conv_tails.resize(layers.size());
            for (size_t index = 0; index < layers.size(); ++index) {
                const DeviceLayer& layer = layers[index];
                if (layer.linear.state.data == nullptr) continue;
                allocate_float(transaction_states[index],
                               layer.linear.state.nbytes / sizeof(float),
                               layer.linear.state.shape);
                allocate_half(transaction_conv_tails[index],
                              layer.linear.conv_tail.nbytes / sizeof(uint16_t),
                              layer.linear.conv_tail.shape);
            }
        }
        if (mtp_enabled) {
            allocate_half(transaction_target_hidden, config.hidden_size,
                          {config.hidden_size});
        }
        for (size_t index = 0; index < layers.size(); ++index) {
            const DeviceLayer& layer = layers[index];
            if (layer.linear.state.data == nullptr) continue;
            check_cuda(cudaMemcpy(transaction_states[index].data,
                                  layer.linear.state.data,
                                  layer.linear.state.nbytes,
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction state copy");
            check_cuda(cudaMemcpy(transaction_conv_tails[index].data,
                                  layer.linear.conv_tail.data,
                                  layer.linear.conv_tail.nbytes,
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction convolution copy");
        }
        if (mtp_enabled) {
            check_cuda(cudaMemcpy(transaction_target_hidden.data,
                                  target_last_hidden.data,
                                  static_cast<size_t>(config.hidden_size) *
                                      sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction hidden copy");
        }
    }

    void restore_transaction_state(int position) {
        PhaseScope scope(this, "transaction_restore");
        if (transaction_states.size() != layers.size()) {
            throw std::runtime_error("Qwen transaction state is unavailable");
        }
        for (size_t index = 0; index < layers.size(); ++index) {
            DeviceLayer& layer = layers[index];
            if (layer.linear.state.data == nullptr) continue;
            check_cuda(cudaMemcpy(layer.linear.state.data,
                                  transaction_states[index].data,
                                  layer.linear.state.nbytes,
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction state restore");
            check_cuda(cudaMemcpy(layer.linear.conv_tail.data,
                                  transaction_conv_tails[index].data,
                                  layer.linear.conv_tail.nbytes,
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction convolution restore");
        }
        if (mtp_enabled) {
            check_cuda(cudaMemcpy(target_last_hidden.data,
                                  transaction_target_hidden.data,
                                  static_cast<size_t>(config.hidden_size) *
                                      sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice),
                       "Qwen transaction hidden restore");
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = position;
        }
        if (dspark_enabled) dspark->crop_context(position);
        if (dflash2_enabled) dflash2->crop_context(position);
    }

    QwenRecurrentSnapshot capture_recurrent_state(
        int position, const QwenForwardResult* result = nullptr,
        bool periodic = false) {
        QwenRecurrentSnapshot snapshot;
        snapshot.position = position;
        snapshot.periodic = periodic;
        if (result != nullptr) {
            snapshot.result = *result;
            snapshot.has_result = true;
        }
        size_t state_bytes = 0;
        size_t tail_bytes = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            state_bytes += layer.linear.state.nbytes;
            tail_bytes += layer.linear.conv_tail.nbytes;
        }
        if (state_bytes != 0) {
            allocate(snapshot.state, state_bytes, {state_bytes / sizeof(float)},
                     SafeDType::F32);
        }
        if (tail_bytes != 0) {
            allocate(snapshot.conv_tail, tail_bytes,
                     {tail_bytes / sizeof(uint16_t)}, SafeDType::F16);
        }
        if (mtp_enabled && position > 0) {
            if (!has_target_last_hidden || target_last_hidden.data == nullptr) {
                throw std::runtime_error(
                    "Qwen MTP snapshot requires the committed target hidden");
            }
            const size_t hidden_elements = static_cast<size_t>(config.hidden_size);
            allocate_half(snapshot.target_hidden, hidden_elements,
                          {config.hidden_size});
            check_cuda(cudaMemcpy(snapshot.target_hidden.data,
                                  target_last_hidden.data,
                                  hidden_elements * sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice),
                       "Qwen target hidden snapshot copy");
        }
        size_t state_offset = 0;
        size_t tail_offset = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            check_cuda(cudaMemcpy(
                           static_cast<uint8_t*>(snapshot.state.data) + state_offset,
                           layer.linear.state.data, layer.linear.state.nbytes,
                           cudaMemcpyDeviceToDevice),
                       "Qwen recurrent state snapshot copy");
            state_offset += layer.linear.state.nbytes;
            if (layer.linear.conv_tail.nbytes != 0) {
                check_cuda(cudaMemcpy(
                               static_cast<uint8_t*>(snapshot.conv_tail.data) + tail_offset,
                               layer.linear.conv_tail.data,
                               layer.linear.conv_tail.nbytes,
                               cudaMemcpyDeviceToDevice),
                           "Qwen convolution tail snapshot copy");
                tail_offset += layer.linear.conv_tail.nbytes;
            }
        }
        return snapshot;
    }

    void restore_recurrent_state(const QwenRecurrentSnapshot& snapshot,
                                 bool restore_target_hidden = true) {
        size_t expected_state_bytes = 0;
        size_t expected_tail_bytes = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            expected_state_bytes += layer.linear.state.nbytes;
            expected_tail_bytes += layer.linear.conv_tail.nbytes;
        }
        if (snapshot.state.nbytes != expected_state_bytes ||
            snapshot.conv_tail.nbytes != expected_tail_bytes) {
            throw std::runtime_error("Qwen recurrent snapshot extent mismatch");
        }
        if (restore_target_hidden && mtp_enabled && snapshot.position > 0) {
            const size_t hidden_bytes = static_cast<size_t>(config.hidden_size) *
                sizeof(uint16_t);
            if (snapshot.target_hidden.nbytes != hidden_bytes) {
                throw std::runtime_error(
                    "Qwen MTP snapshot target hidden extent mismatch");
            }
            allocate_half(target_last_hidden, config.hidden_size,
                          {config.hidden_size});
            check_cuda(cudaMemcpy(target_last_hidden.data,
                                  snapshot.target_hidden.data, hidden_bytes,
                                  cudaMemcpyDeviceToDevice),
                       "Qwen target hidden snapshot restore");
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = snapshot.position;
        } else if (snapshot.position == 0) {
            has_target_last_hidden = false;
            mtp_seed_ready = false;
            mtp_position = 0;
        }
        if (dspark_enabled) {
            // DSpark K/V is position-indexed like target GQA. Cropping only moves
            // the logical committed boundary; replay overwrites the suffix.
            if (snapshot.position <= dspark->committed_position()) {
                dspark->crop_context(snapshot.position);
            } else {
                throw std::runtime_error(
                    "Qwen DSpark snapshot exceeds committed context");
            }
        }
        if (dflash2_enabled) {
            if (snapshot.position <= dflash2->committed_position()) {
                dflash2->crop_context(snapshot.position);
            } else {
                throw std::runtime_error(
                    "Qwen DFlash2 snapshot exceeds committed context");
            }
        }
        size_t state_offset = 0;
        size_t tail_offset = 0;
        for (DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            check_cuda(cudaMemcpy(
                           layer.linear.state.data,
                           static_cast<const uint8_t*>(snapshot.state.data) + state_offset,
                           layer.linear.state.nbytes, cudaMemcpyDeviceToDevice),
                       "Qwen recurrent state snapshot restore");
            state_offset += layer.linear.state.nbytes;
            if (layer.linear.conv_tail.nbytes != 0) {
                check_cuda(cudaMemcpy(
                               layer.linear.conv_tail.data,
                               static_cast<const uint8_t*>(snapshot.conv_tail.data) + tail_offset,
                               layer.linear.conv_tail.nbytes,
                               cudaMemcpyDeviceToDevice),
                           "Qwen convolution tail snapshot restore");
                tail_offset += layer.linear.conv_tail.nbytes;
            }
        }
    }

    // Keeps snapshots ordered and bounded. The newest position wins because a
    // monotonically growing prompt resumes from the deepest available point.
    void record_snapshot(int position, const QwenForwardResult* result = nullptr,
                         bool periodic = false) {
        if (!options.prefix_cache ||
            options.state_snapshot_interval_tokens <= 0 ||
            options.max_state_snapshots <= 0 || position <= 0) {
            return;
        }
        if (!has_recurrent_state() && !mtp_enabled && !dspark_enabled) return;
        for (QwenRecurrentSnapshot& entry : snapshots) {
            if (entry.position != position) continue;
            entry.periodic = entry.periodic || periodic;
            if (result != nullptr && !entry.has_result) {
                entry.result = *result;
                entry.has_result = true;
            }
            return;
        }
        snapshots.push_back(capture_recurrent_state(position, result, periodic));
        std::sort(snapshots.begin(), snapshots.end(),
                  [](const QwenRecurrentSnapshot& a,
                     const QwenRecurrentSnapshot& b) {
                      return a.position < b.position;
                  });
        // Preserve periodic coverage. Request-boundary snapshots improve exact
        // repeats but are expendable; evict their oldest entries first.
        while (static_cast<int>(snapshots.size()) > options.max_state_snapshots) {
            auto evict = std::find_if(
                snapshots.begin(), snapshots.end(),
                [](const QwenRecurrentSnapshot& entry) {
                    return !entry.periodic;
                });
            if (evict == snapshots.end()) evict = snapshots.begin();
            snapshots.erase(evict);
        }
    }

    // Deepest snapshot at or before limit, or nullptr for the empty state.
    const QwenRecurrentSnapshot* snapshot_at_or_before(int limit) const {
        const QwenRecurrentSnapshot* best = nullptr;
        for (const QwenRecurrentSnapshot& entry : snapshots) {
            if (entry.position <= limit &&
                (best == nullptr || entry.position > best->position)) {
                best = &entry;
            }
        }
        return best;
    }

    // A prompt result needs the hidden/logits for its final token. If the
    // chosen state is already after that token, use an earlier snapshot so the
    // final token is executed again rather than returning stale logits.
    const QwenRecurrentSnapshot* snapshot_strictly_before(int limit) const {
        const QwenRecurrentSnapshot* best = nullptr;
        for (const QwenRecurrentSnapshot& entry : snapshots) {
            if (entry.position < limit &&
                (best == nullptr || entry.position > best->position)) {
                best = &entry;
            }
        }
        return best;
    }

    uint64_t snapshot_bytes() const {
        uint64_t total = 0;
        for (const QwenRecurrentSnapshot& entry : snapshots) {
            total += entry.bytes();
        }
        return total;
    }

    int early_snapshot_interval() const {
        if (options.state_snapshot_interval_tokens <= 0) return 0;
        return std::min(256, options.state_snapshot_interval_tokens);
    }

    bool is_periodic_snapshot_position(int position) const {
        const int interval = options.state_snapshot_interval_tokens;
        if (interval <= 0 || position <= 0) return false;
        if (position % interval == 0) return true;
        const int early_interval = early_snapshot_interval();
        return position <= 4096 && position % early_interval == 0;
    }

    int next_periodic_snapshot_after(int position) const {
        const int interval = options.state_snapshot_interval_tokens;
        if (interval <= 0) return 0;
        const int next_regular = ((position / interval) + 1) * interval;
        if (position >= 4096) return next_regular;
        const int early_interval = early_snapshot_interval();
        const int next_early = ((position / early_interval) + 1) * early_interval;
        return std::min(next_regular, next_early);
    }

    void drop_snapshots_after(int position) {
        snapshots.erase(
            std::remove_if(snapshots.begin(), snapshots.end(),
                           [position](const QwenRecurrentSnapshot& entry) {
                               return entry.position > position;
                           }),
            snapshots.end());
    }

    // Opt-in prefill phase attribution. Each scope synchronises the device, so it
    // is only ever enabled for profiling runs; the default path adds no sync.
    bool phase_profile = qwen_env_enabled("QWEN_PHASE_PROFILE");
    std::map<std::string, double> phase_seconds;
    std::map<std::string, uint64_t> phase_calls;

    class PhaseScope {
    public:
        PhaseScope(Impl* owner, const char* name) : owner_(owner), name_(name) {
            if (owner_->phase_profile) {
                cudaDeviceSynchronize();
                started_ = std::chrono::steady_clock::now();
            }
        }
        ~PhaseScope() {
            if (owner_->phase_profile) {
                cudaDeviceSynchronize();
                owner_->phase_seconds[name_] += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_).count();
                ++owner_->phase_calls[name_];
            }
        }
        PhaseScope(const PhaseScope&) = delete;
        PhaseScope& operator=(const PhaseScope&) = delete;

    private:
        Impl* owner_;
        std::string name_;
        std::chrono::steady_clock::time_point started_;
    };

    void report_phase_profile(const char* tag) const {
        if (!phase_profile) return;
        double total = 0.0;
        for (const auto& entry : phase_seconds) total += entry.second;
        for (const auto& entry : phase_seconds) {
            std::cout << "qwen_phase tag=" << tag << " rank=" << options.tp_rank
                      << " phase=" << entry.first
                      << " seconds=" << entry.second
                      << " calls=" << phase_calls.at(entry.first)
                      << " share=" << (total > 0.0 ? entry.second / total : 0.0)
                      << "\n";
        }
        std::cout << "qwen_phase tag=" << tag << " rank=" << options.tp_rank
                  << " phase=TOTAL seconds=" << total << "\n";
        std::cout.flush();
    }

    void all_reduce_half(uint16_t* values, int count) {
        if (options.tp_world == 1) return;
#ifdef DSV4_HAVE_NCCL
        if (options.nccl_id_path.empty()) {
            throw std::runtime_error("Qwen TP requires --nccl-id-path");
        }
        {
            PhaseScope scope(this, "tp_all_reduce");
            nccl_all_reduce_sum_f16_inplace(
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path.c_str(), values, count);
        }
#else
        (void)values;
        (void)count;
        throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
#endif
    }

    void projection(const DeviceLinear& linear, const uint16_t* input,
                    uint16_t* output, int rows) {
        PhaseScope scope(this, rows == 1 ? "projection_decode" : "projection_rows");
        const int output_rows = static_cast<int>(linear.weight.shape[0]);
        const int columns = static_cast<int>(linear.weight.shape[1]);
        if (linear.fp8) {
            if (rows == 1) {
                require_launch(qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    input, linear.weight.fp8_data(), linear.scale.f16_data(),
                    output, output_rows, columns, columns,
                    static_cast<int>(linear.scale.shape[1])),
                    "FP8 FP16-activation decode projection");
            } else {
                require_launch(qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                    input, linear.weight.fp8_data(), linear.scale.f16_data(),
                    output, rows, output_rows, columns, columns, output_rows,
                    columns, static_cast<int>(linear.scale.shape[1])),
                    "FP8 FP16-activation projection");
            }
        } else {
            require_launch(qwen_fp16_matmul_rows_f16_cuda(
                input, linear.weight.f16_data(), output, rows, output_rows,
                columns, columns, output_rows, columns),
                "FP16 activation/weight projection");
        }
    }

    void norm(const QwenDeviceTensor& gamma, const uint16_t* input,
              uint16_t* output, int rows, int columns) {
        require_launch(qwen_rmsnorm_fp16_gamma_rows_f16_cuda(
            input, gamma.f16_data(), output, rows, columns,
            static_cast<float>(config.rms_norm_eps)), "Qwen FP16 RMSNorm");
    }

    void add(uint16_t* output, const uint16_t* input, int count) {
        require_launch(qwen_add_inplace_f16_cuda(output, input, count),
                       "Qwen FP16 residual add");
    }

    void begin_workspace() { workspace.begin(); }

    QwenDeviceTensor& workspace_half(size_t elements,
                                     const std::vector<uint64_t>& shape) {
        return workspace.half_tensor(elements, shape);
    }

    QwenDeviceTensor& workspace_float(size_t elements,
                                      const std::vector<uint64_t>& shape) {
        return workspace.float_tensor(elements, shape);
    }

    void linear_attention(DeviceLayer& layer, const uint16_t* hidden,
                          uint16_t* output, int rows, int position_offset) {
        const int key_heads = static_cast<int>(
            config.linear_attention.key_heads / options.tp_world);
        const int value_heads = static_cast<int>(
            config.linear_attention.value_heads / options.tp_world);
        const int key_dim = key_heads *
            static_cast<int>(config.linear_attention.key_head_dim);
        const int value_dim = value_heads *
            static_cast<int>(config.linear_attention.value_head_dim);
        const int packed_dim = 2 * key_dim + value_dim;
        const int kernel = static_cast<int>(config.linear_attention.conv_kernel_dim);
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t packed_elements = static_cast<size_t>(rows) * packed_dim;
        const size_t key_elements = static_cast<size_t>(rows) * key_dim;
        const size_t value_elements = static_cast<size_t>(rows) * value_dim;
        const size_t gate_elements = static_cast<size_t>(rows) * value_heads;
        QwenDeviceTensor& packed = workspace_half(
            packed_elements, {static_cast<uint64_t>(rows),
                              static_cast<uint64_t>(packed_dim)});
        QwenDeviceTensor& convolved = workspace_half(packed_elements, packed.shape);
        QwenDeviceTensor& q = workspace_half(
            key_elements, {static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(key_dim)});
        QwenDeviceTensor& k = workspace_half(key_elements, q.shape);
        QwenDeviceTensor& v = workspace_half(
            value_elements, {static_cast<uint64_t>(rows),
                             static_cast<uint64_t>(value_dim)});
        QwenDeviceTensor& a = workspace_half(
            gate_elements, {static_cast<uint64_t>(rows),
                            static_cast<uint64_t>(value_heads)});
        QwenDeviceTensor& b = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& gates = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& beta = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& core = workspace_half(value_elements, v.shape);
        QwenDeviceTensor& z = workspace_half(value_elements, v.shape);
        QwenDeviceTensor& normalized = workspace_half(value_elements, v.shape);

        projection(layer.linear.qkv, hidden, packed.f16_data(), rows);
        require_launch(qwen_causal_depthwise_conv_silu_f16_cuda(
            packed.f16_data(), layer.linear.conv.f16_data(),
            layer.linear.conv_tail.f16_data(), convolved.f16_data(), rows,
            packed_dim, kernel, true), "FP16 linear causal convolution");
        require_launch(qwen_split_packed_qkv_f16_cuda(
            convolved.f16_data(), q.f16_data(), k.f16_data(), v.f16_data(),
            rows, key_dim, value_dim), "FP16 linear QKV split");
        projection(layer.linear.a, hidden, a.f16_data(), rows);
        projection(layer.linear.b, hidden, b.f16_data(), rows);
        require_launch(qwen_linear_attn_gates_f16_cuda(
            a.f16_data(), b.f16_data(), layer.linear.a_log.f16_data(),
            layer.linear.dt_bias.f16_data(), gates.f16_data(), beta.f16_data(),
            rows, value_heads), "FP16 linear attention gates");
        const float q_scale = 1.0f / std::sqrt(
            static_cast<float>(config.linear_attention.key_head_dim));
        std::optional<PhaseScope> delta_scope;
        delta_scope.emplace(this, "gated_delta");
        const bool sequenced = rows > 1 && qwen_gated_delta_sequence_f16_cuda(
            layer.linear.state.f32_data(), q.f16_data(), k.f16_data(),
            v.f16_data(), gates.f16_data(), beta.f16_data(),
            core.f16_data(), rows, value_heads, key_heads,
            static_cast<int>(config.linear_attention.key_head_dim),
            static_cast<int>(config.linear_attention.value_head_dim), q_scale);
        for (int token = 0; !sequenced && token < rows; ++token) {
            require_launch(qwen_gated_delta_step_f16_cuda(
                layer.linear.state.f32_data(),
                q.f16_data() + static_cast<size_t>(token) * key_dim,
                k.f16_data() + static_cast<size_t>(token) * key_dim,
                v.f16_data() + static_cast<size_t>(token) * value_dim,
                gates.f16_data() + static_cast<size_t>(token) * value_heads,
                beta.f16_data() + static_cast<size_t>(token) * value_heads,
                core.f16_data() + static_cast<size_t>(token) * value_dim,
                value_heads, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim),
                static_cast<int>(config.linear_attention.value_head_dim), q_scale),
                "FP16 linear recurrent state");
        }
        delta_scope.reset();
        projection(layer.linear.z, hidden, z.f16_data(), rows);
        require_launch(qwen_gated_rmsnorm_fp16_gamma_rows_f16_cuda(
            core.f16_data(), layer.linear.norm.f16_data(), z.f16_data(),
            normalized.f16_data(), rows * value_heads,
            static_cast<int>(config.linear_attention.value_head_dim),
            static_cast<float>(config.rms_norm_eps)),
            "FP16 linear gated RMSNorm");
        projection(layer.linear.out, normalized.f16_data(), output, rows);
        all_reduce_half(output, rows * hidden_size);
        (void)position_offset;
    }

    void full_attention(DeviceLayer& layer, const uint16_t* hidden,
                        uint16_t* output, int rows, int position_offset) {
        PhaseScope scope(this, "full_attention");
        const int q_heads = static_cast<int>(
            config.full_attention.num_heads / options.tp_world);
        const int kv_heads = static_cast<int>(
            config.full_attention.num_key_value_heads / options.tp_world);
        const int head_dim = static_cast<int>(config.full_attention.head_dim);
        const int attention_dim = q_heads * head_dim;
        const int q_projection_dim = attention_dim * 2;
        const size_t q_projection_elements =
            static_cast<size_t>(rows) * q_projection_dim;
        const size_t attention_elements =
            static_cast<size_t>(rows) * attention_dim;
        const size_t kv_elements =
            static_cast<size_t>(rows) * kv_heads * head_dim;
        QwenDeviceTensor& q_projection = workspace_half(
            q_projection_elements, {static_cast<uint64_t>(rows),
                                    static_cast<uint64_t>(q_projection_dim)});
        QwenDeviceTensor& q = workspace_half(
            attention_elements, {static_cast<uint64_t>(rows),
                                 static_cast<uint64_t>(attention_dim)});
        QwenDeviceTensor& gate = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& k = workspace_half(
            kv_elements, {static_cast<uint64_t>(rows),
                          static_cast<uint64_t>(kv_heads),
                          static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v = workspace_half(kv_elements, k.shape);
        QwenDeviceTensor& q_norm = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& k_norm = workspace_half(kv_elements, k.shape);
        QwenDeviceTensor& attention = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& merged = workspace_half(attention_elements, q.shape);

        projection(layer.full.q, hidden, q_projection.f16_data(), rows);
        require_launch(qwen_split_q_gate_f16_cuda(
            q_projection.f16_data(), q.f16_data(), gate.f16_data(), rows,
            q_heads, head_dim), "FP16 full Q/gate split");
        projection(layer.full.k, hidden, k.f16_data(), rows);
        projection(layer.full.v, hidden, v.f16_data(), rows);
        norm(layer.full.q_norm, q.f16_data(), q_norm.f16_data(),
             rows * q_heads, head_dim);
        norm(layer.full.k_norm, k.f16_data(), k_norm.f16_data(),
             rows * kv_heads, head_dim);
        require_launch(qwen_partial_rope_rows_f16_cuda(
            q_norm.f16_data(), k_norm.f16_data(), position_offset, rows,
            static_cast<int>(config.partial_rotary_dim()),
            static_cast<float>(config.rope_theta), q_heads, kv_heads, head_dim),
            "FP16 partial RoPE");

        const bool fp8_cache = options.kv_cache_dtype == QwenKvCacheDType::Fp8;
        const int attention_window = options.attention_window;
        const int sink_tokens = options.attention_sink_tokens;
        if (fp8_cache) {
            require_launch(qwen_append_kv_cache_fp8_cuda(
                k_norm.f16_data(), v.f16_data(), layer.full.k_cache.fp8_data(),
                layer.full.v_cache.fp8_data(), layer.full.k_scale.f16_data(),
                layer.full.v_scale.f16_data(), rows, kv_heads, head_dim,
                kKvScaleBlock, position_offset, max_context),
                "append FP8 full KV cache");
        } else {
            require_launch(qwen_append_kv_cache_f16_cuda(
                k_norm.f16_data(), v.f16_data(), layer.full.k_cache.f16_data(),
                layer.full.v_cache.f16_data(), rows, kv_heads, head_dim,
                position_offset, max_context), "append FP16 full KV cache");
        }

        if (rows == 1) {
            const int context_length = position_offset + 1;
            const bool optimized_attention =
                qwen_env_enabled("DSV4_QWEN_GQA_OPTIMIZED") ||
                attention_window > 0;
            // The compact split/merge decode path crosses over the reference
            // score/value kernels only at long contexts on SM75.
            constexpr int kOptimizedDecodeMinContext = 16384;
            const bool optimized_decode = !fp8_cache && optimized_attention &&
                (context_length >= kOptimizedDecodeMinContext ||
                 attention_window > 0);
            int attended_positions = context_length;
            if (attention_window > 0) {
                const int sink_count = std::min(sink_tokens, context_length);
                const int window_start = std::max(
                    context_length - attention_window, sink_count);
                attended_positions = sink_count + (context_length - window_start);
            }
            const int optimized_splits = std::max(1, std::min(
                64, (attended_positions + 2048 - 1) / 2048));
            const size_t score_elements = optimized_decode
                ? static_cast<size_t>(q_heads) * optimized_splits *
                      static_cast<size_t>(head_dim + 2)
                : static_cast<size_t>(q_heads) * context_length;
            QwenDeviceTensor& scores = workspace_float(
                score_elements,
                optimized_decode
                    ? std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                            static_cast<uint64_t>(optimized_splits),
                                            static_cast<uint64_t>(head_dim + 2)}
                    : std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                             static_cast<uint64_t>(context_length)});
            if (fp8_cache) {
                require_launch(qwen_gqa_decode_attention_fp8_cuda(
                    q_norm.f16_data(), layer.full.k_cache.fp8_data(),
                    layer.full.v_cache.fp8_data(), layer.full.k_scale.f16_data(),
                    layer.full.v_scale.f16_data(), attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    kKvScaleBlock, context_length, max_context),
                    "decode FP8-cache GQA");
            } else if (optimized_decode) {
                require_launch(qwen_gqa_decode_attention_f16_fused_cuda(
                    q_norm.f16_data(), layer.full.k_cache.f16_data(),
                    layer.full.v_cache.f16_data(), attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    context_length, max_context, attention_window, sink_tokens),
                    "decode optimized FP16-cache GQA");
            } else {
                require_launch(qwen_gqa_decode_attention_f16_cuda(
                    q_norm.f16_data(), layer.full.k_cache.f16_data(),
                    layer.full.v_cache.f16_data(), attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    context_length, max_context), "decode FP16-cache GQA");
            }
        } else if (fp8_cache) {
            require_launch(qwen_gqa_prefill_attention_fp8_cuda(
                q_norm.f16_data(), layer.full.k_cache.fp8_data(),
                layer.full.v_cache.fp8_data(), layer.full.k_scale.f16_data(),
                layer.full.v_scale.f16_data(), attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, kKvScaleBlock, position_offset,
                max_context), "prefill FP8-cache GQA");
        } else if (rows <= 8 && attention_window == 0) {
            const int context_length = position_offset + rows;
            if (qwen_env_enabled("QWEN_GQA_VERIFY_SPLIT")) {
                const int verify_target_positions =
                    qwen_env_int("QWEN_GQA_VERIFY_TILE", 64);
                const int verify_splits = std::max(1, std::min(
                    64, (context_length + verify_target_positions - 1) /
                            verify_target_positions));
                const size_t partial_elements =
                    static_cast<size_t>(rows) * q_heads * verify_splits *
                    static_cast<size_t>(head_dim + 2);
                QwenDeviceTensor& partials = workspace_float(
                    partial_elements,
                    {static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(q_heads),
                     static_cast<uint64_t>(verify_splits),
                     static_cast<uint64_t>(head_dim + 2)});
                require_launch(qwen_gqa_verify_attention_f16_cuda(
                    q_norm.f16_data(), layer.full.k_cache.f16_data(),
                    layer.full.v_cache.f16_data(), attention.f16_data(),
                    partials.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context, verify_splits),
                    "verify split FP16-cache GQA");
            } else {
                const size_t score_elements =
                    static_cast<size_t>(rows) * q_heads * context_length;
                QwenDeviceTensor& scores = workspace_float(
                    score_elements,
                    {static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(q_heads),
                     static_cast<uint64_t>(context_length)});
                require_launch(qwen_gqa_verify_attention_f16_exact_cuda(
                    q_norm.f16_data(), layer.full.k_cache.f16_data(),
                    layer.full.v_cache.f16_data(), attention.f16_data(),
                    scores.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "verify exact FP16-cache GQA");
            }
        } else if (qwen_env_enabled("DSV4_QWEN_GQA_OPTIMIZED") ||
                   attention_window > 0) {
            require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                q_norm.f16_data(), layer.full.k_cache.f16_data(),
                layer.full.v_cache.f16_data(), attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, position_offset, max_context,
                attention_window, sink_tokens),
                "prefill optimized FP16-cache GQA");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16_cuda(
                q_norm.f16_data(), layer.full.k_cache.f16_data(),
                layer.full.v_cache.f16_data(), attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, position_offset, max_context),
                "prefill FP16-cache GQA");
        }
        require_launch(qwen_sigmoid_mul_f16_cuda(
            attention.f16_data(), gate.f16_data(), merged.f16_data(),
            rows * attention_dim), "FP16 full attention output gate");
        projection(layer.full.out, merged.f16_data(), output, rows);
        all_reduce_half(output, rows * static_cast<int>(config.hidden_size));
    }

    void layer_forward(DeviceLayer& layer, const uint16_t* hidden,
                       uint16_t* output, int rows, int position_offset) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        begin_workspace();
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const size_t intermediate_elements = static_cast<size_t>(rows) *
            layer.gate.weight.shape[0];
        QwenDeviceTensor& normalized = workspace_half(
            hidden_elements, {static_cast<uint64_t>(rows),
                              static_cast<uint64_t>(hidden_size)});
        QwenDeviceTensor& attention = workspace_half(hidden_elements, normalized.shape);
        QwenDeviceTensor& post = workspace_half(hidden_elements, normalized.shape);
        const bool compatible_swiglu = layer.gate.fp8 && layer.up.fp8 &&
            layer.gate.weight.shape == layer.up.weight.shape &&
            layer.gate.scale.shape == layer.up.scale.shape;
        const bool fused_decode_swiglu = rows == 1 && compatible_swiglu;
        const bool fused_small_batch_swiglu = rows > 1 && rows <= 8 &&
            compatible_swiglu && hidden_size % 4 == 0;
        QwenDeviceTensor* gate = nullptr;
        QwenDeviceTensor* up = nullptr;
        if (!fused_decode_swiglu && !fused_small_batch_swiglu) {
            gate = &workspace_half(intermediate_elements,
                {static_cast<uint64_t>(rows), layer.gate.weight.shape[0]});
            up = &workspace_half(intermediate_elements, gate->shape);
        }
        QwenDeviceTensor& intermediate = workspace_half(
            intermediate_elements,
            {static_cast<uint64_t>(rows), layer.gate.weight.shape[0]});
        QwenDeviceTensor& mlp = workspace_half(hidden_elements, normalized.shape);

        norm(layer.input_norm, hidden, normalized.f16_data(), rows, hidden_size);
        if (layer.linear.qkv.weight.data != nullptr) {
            linear_attention(layer, normalized.f16_data(), attention.f16_data(),
                             rows, position_offset);
        } else {
            full_attention(layer, normalized.f16_data(), attention.f16_data(),
                           rows, position_offset);
        }
        check_cuda(cudaMemcpy(output, hidden,
            hidden_elements * sizeof(uint16_t), cudaMemcpyDeviceToDevice),
            "Qwen FP16 residual copy");
        add(output, attention.f16_data(), rows * hidden_size);
        norm(layer.post_norm, output, post.f16_data(), rows, hidden_size);
        if (fused_decode_swiglu) {
            require_launch(qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                post.f16_data(), layer.gate.weight.fp8_data(),
                layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                layer.up.scale.f16_data(), intermediate.f16_data(),
                static_cast<int>(layer.gate.weight.shape[0]), hidden_size,
                hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
                "FP16 fused decode SwiGLU");
        } else if (fused_small_batch_swiglu) {
            require_launch(
                qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
                    post.f16_data(), layer.gate.weight.fp8_data(),
                    layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                    layer.up.scale.f16_data(), intermediate.f16_data(), rows,
                    static_cast<int>(layer.gate.weight.shape[0]), hidden_size,
                    hidden_size, static_cast<int>(layer.gate.weight.shape[0]),
                    hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
                "FP16 fused small-batch SwiGLU");
        } else {
            projection(layer.gate, post.f16_data(), gate->f16_data(), rows);
            projection(layer.up, post.f16_data(), up->f16_data(), rows);
            require_launch(qwen_silu_mul_rows_f16_cuda(
                gate->f16_data(), up->f16_data(), intermediate.f16_data(), rows,
                static_cast<int>(layer.gate.weight.shape[0])), "FP16 SwiGLU");
        }
        projection(layer.down, intermediate.f16_data(), mlp.f16_data(), rows);
        all_reduce_half(mlp.f16_data(), rows * hidden_size);
        add(output, mlp.f16_data(), rows * hidden_size);
    }

    QwenVerifyBatch top_tokens_for(const uint16_t* hidden, int rows,
                                   const QwenDeviceTensor* output_norm,
                                   int position_after) {
        if (rows <= 0) {
            throw std::runtime_error("Qwen logits require at least one row");
        }
        const int hidden_size = static_cast<int>(config.hidden_size);
        const int local_vocab = static_cast<int>(lm_head.shape[0]);
        begin_workspace();
        QwenDeviceTensor& normalized = workspace_half(
            static_cast<size_t>(rows) * hidden_size,
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)});
        if (output_norm != nullptr) {
            norm(*output_norm, hidden, normalized.f16_data(), rows, hidden_size);
        } else {
            check_cuda(cudaMemcpy(normalized.data, hidden,
                                  static_cast<size_t>(rows) * hidden_size *
                                      sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice),
                       "Qwen normalized hidden copy");
        }
        QwenDeviceTensor& local_logits = workspace_float(
            static_cast<size_t>(rows) * local_vocab,
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(local_vocab)});
        const bool cublas_logits = rows > 1 &&
            qwen_env_enabled("QWEN_FP16_LOGITS_CUBLAS");
        require_launch(cublas_logits
            ? qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                  normalized.f16_data(), lm_head.f16_data(),
                  local_logits.f32_data(), rows, local_vocab, hidden_size,
                  hidden_size, local_vocab, hidden_size)
            : qwen_fp16_matmul_rows_f16_f32_cuda(
                  normalized.f16_data(), lm_head.f16_data(),
                  local_logits.f32_data(), rows, local_vocab, hidden_size,
                  hidden_size, local_vocab, hidden_size),
            "Qwen batched FP32 logits");
        allocate(argmax_token, static_cast<size_t>(rows) * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        allocate_float(argmax_logit, static_cast<size_t>(rows),
                       {static_cast<uint64_t>(rows)});
        const int vocab_start = static_cast<int>(weights_vocab_start());
        require_launch(argmax_fp32_rows_cuda(
            local_logits.f32_data(), static_cast<int*>(argmax_token.data),
            argmax_logit.f32_data(), rows, local_vocab, vocab_start),
            "Qwen batched local argmax");

        QwenVerifyBatch result;
        result.top_tokens.resize(static_cast<size_t>(rows));
        result.top_logits.resize(static_cast<size_t>(rows));
        result.local_logits.resize(static_cast<size_t>(rows));
        result.position_after = position_after;
#ifdef DSV4_HAVE_NCCL
        if (options.tp_world > 1) {
            if (options.nccl_id_path.empty()) {
                throw std::runtime_error("Qwen TP requires --nccl-id-path");
            }
            nccl_global_top1_rows(
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path.c_str(),
                static_cast<const int*>(argmax_token.data),
                argmax_logit.f32_data(), rows, result.top_tokens.data(),
                result.top_logits.data());
            check_cuda(cudaMemcpy(result.local_logits.data(), argmax_logit.data,
                                  static_cast<size_t>(rows) * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "Qwen batched local logit copy");
        } else
#endif
        {
#ifndef DSV4_HAVE_NCCL
            if (options.tp_world > 1) {
                throw std::runtime_error(
                    "Qwen TP requires an NCCL-enabled build");
            }
#endif
            check_cuda(cudaMemcpy(result.top_tokens.data(), argmax_token.data,
                                  static_cast<size_t>(rows) * sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "Qwen batched argmax token copy");
            check_cuda(cudaMemcpy(result.top_logits.data(), argmax_logit.data,
                                  static_cast<size_t>(rows) * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "Qwen batched argmax logit copy");
            result.local_logits = result.top_logits;
        }
        return result;
    }

    QwenForwardResult logits_for(const uint16_t* hidden, int last_row,
                                 int position_after, int active_layers) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        QwenVerifyBatch batch = top_tokens_for(
            hidden + static_cast<size_t>(last_row) * hidden_size, 1,
            &final_norm, position_after);
        QwenForwardResult result;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = batch.top_tokens[0];
        result.top_logit = batch.top_logits[0];
        result.checksum = batch.local_logits[0];
        result.position = position_after;
        return result;
    }

    QwenVerifyBatch target_logits_for(const uint16_t* hidden, int rows,
                                      int position_after) {
        return top_tokens_for(hidden, rows, &final_norm, position_after);
    }

    QwenForwardResult mtp_logits_for(const uint16_t* normalized_hidden,
                                     int position_after) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        QwenVerifyBatch batch = top_tokens_for(normalized_hidden, 1, nullptr,
                                               position_after);
        QwenForwardResult result;
        result.layers = 1;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = batch.top_tokens[0];
        result.top_logit = batch.top_logits[0];
        result.checksum = batch.local_logits[0];
        result.position = position_after;
        return result;
    }

    QwenForwardResult mtp_forward_rows(const std::vector<int>& tokens,
                                       const uint16_t* hidden, int position) {
        if (!mtp_enabled) {
            throw std::runtime_error("Qwen MTP rows requested while disabled");
        }
        const int rows = static_cast<int>(tokens.size());
        if (hidden == nullptr || rows <= 0 || position < 0 ||
            position + rows > max_context) {
            throw std::runtime_error("invalid Qwen MTP row extent");
        }
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const std::vector<uint64_t> hidden_shape = {
            static_cast<uint64_t>(rows), config.hidden_size};
        allocate(d_tokens, tokens.size() * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_cuda(cudaMemcpy(d_tokens.data, tokens.data(),
                              tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "Qwen MTP token upload");
        allocate_half(mtp_embedding, hidden_elements, hidden_shape);
        require_launch(qwen_embedding_fp16_gather_f16_cuda(
            embed.f16_data(), static_cast<int*>(d_tokens.data),
            mtp_embedding.f16_data(), rows, hidden_size,
            static_cast<int>(weights_vocab_start()),
            static_cast<int>(embed.shape[0])), "Qwen MTP embedding lookup");
        all_reduce_half(mtp_embedding.f16_data(), rows * hidden_size);

        allocate_half(mtp_normalized_embedding, hidden_elements, hidden_shape);
        allocate_half(mtp_normalized_hidden, hidden_elements, hidden_shape);
        norm(mtp_pre_fc_norm_embedding, mtp_embedding.f16_data(),
             mtp_normalized_embedding.f16_data(), rows, hidden_size);
        norm(mtp_pre_fc_norm_hidden, hidden,
             mtp_normalized_hidden.f16_data(), rows, hidden_size);
        allocate_half(mtp_concat, hidden_elements * 2,
                      {static_cast<uint64_t>(rows),
                       static_cast<uint64_t>(2 * hidden_size)});
        require_launch(qwen_concat_rows_f16_cuda(
            mtp_normalized_embedding.f16_data(),
            mtp_normalized_hidden.f16_data(), mtp_concat.f16_data(), rows,
            hidden_size), "Qwen MTP fusion concat");
        allocate_half(mtp_fused, hidden_elements, hidden_shape);
        projection(mtp_fc, mtp_concat.f16_data(), mtp_fused.f16_data(), rows);
        allocate_half(mtp_next_hidden, hidden_elements, hidden_shape);
        layer_forward(mtp_layer, mtp_fused.f16_data(),
                      mtp_next_hidden.f16_data(), rows, position);
        allocate_half(mtp_normalized_output, hidden_elements, hidden_shape);
        norm(mtp_norm, mtp_next_hidden.f16_data(),
             mtp_normalized_output.f16_data(), rows, hidden_size);
        const uint16_t* last_hidden = mtp_normalized_output.f16_data() +
            static_cast<size_t>(rows - 1) * hidden_size;
        QwenForwardResult result = mtp_logits_for(last_hidden, position + rows);
        allocate_half(mtp_seed_hidden, hidden_size, {config.hidden_size});
        // Recursive Qwen3.5 MTP consumes the prior predictor's returned hidden,
        // and that return is after mtp.norm in both vLLM and SGLang.
        check_cuda(cudaMemcpy(mtp_seed_hidden.data, last_hidden,
                              static_cast<size_t>(hidden_size) *
                                  sizeof(uint16_t),
                              cudaMemcpyDeviceToDevice),
                   "Qwen MTP seed hidden copy");
        mtp_position = position + rows;
        mtp_seed_input_token = tokens.back();
        mtp_next_token = result.top_token;
        mtp_next_logit = result.top_logit;
        mtp_next_checksum = result.checksum;
        mtp_seed_ready = true;
        return result;
    }

    QwenForwardResult mtp_forward_row(int token, const uint16_t* hidden,
                                      int position) {
        return mtp_forward_rows({token}, hidden, position);
    }

    QwenForwardResult prime_target_mtp(const std::vector<int>& shifted_tokens,
                                       int position) {
        const int rows = static_cast<int>(shifted_tokens.size());
        const int hidden_size = static_cast<int>(config.hidden_size);
        if (rows <= 0 || target_hidden_rows.data == nullptr ||
            target_hidden_rows.nbytes < static_cast<uint64_t>(rows) * hidden_size *
                sizeof(uint16_t)) {
            throw std::runtime_error(
                "Qwen MTP target prime requires matching target hidden rows");
        }
        return mtp_forward_rows(shifted_tokens, target_hidden_rows.f16_data(),
                                position);
    }

    QwenForwardResult seed_mtp(int input_token) {
        if (!has_target_last_hidden) {
            throw std::runtime_error(
                "Qwen MTP seed requires a committed target hidden");
        }
        return mtp_forward_row(input_token, target_last_hidden.f16_data(),
                               mtp_position - 1);
    }

    void rewrite_mtp_boundary(int input_token, int position) {
        if (!has_target_last_hidden || position < 0 || position >= max_context) {
            throw std::runtime_error(
                "Qwen MTP boundary rewrite requires a committed target hidden");
        }
        (void)mtp_forward_row(input_token, target_last_hidden.f16_data(), position);
    }

    std::vector<int> draft_tokens(int count, int input_token,
                                  QwenMtpStats* stats) {
        if (count <= 0) return {};
        const auto started = std::chrono::steady_clock::now();
        QwenForwardResult next;
        if (mtp_seed_ready && mtp_seed_input_token == input_token) {
            next.top_token = mtp_next_token;
            next.top_logit = mtp_next_logit;
            next.checksum = mtp_next_checksum;
        } else {
            next = seed_mtp(input_token);
        }
        std::vector<int> drafts;
        drafts.reserve(static_cast<size_t>(count));
        drafts.push_back(next.top_token);
        while (static_cast<int>(drafts.size()) < count) {
            next = mtp_forward_row(
                drafts.back(), mtp_seed_hidden.f16_data(), mtp_position);
            drafts.push_back(next.top_token);
        }
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->proposed_drafts += static_cast<uint64_t>(drafts.size());
        }
        return drafts;
    }

    QwenForwardResult dflash2_speculative_step(int input_token,
                                               QwenMtpStats* stats) {
        if (!dflash2_enabled) {
            throw std::runtime_error("Qwen DFlash2 speculative step is disabled");
        }
        const int committed_position = dflash2->committed_position();
        prepare_transaction_state();
        const auto draft_started = std::chrono::steady_clock::now();
        const QwenDFlash2Proposal proposal = dflash2->propose(input_token);
        // The 48 gated-delta layers recur sequentially across verify rows, so a
        // verify block costs close to linearly in its width. When acceptance runs
        // well below the full seven drafts the tail rows are paid for and then
        // discarded, so allow the block to be capped. 0 keeps the full proposal.
        std::vector<int> draft_tokens_used = proposal.tokens;
        int width_limit = dflash2_draft_width;
        if (dflash2_adaptive_width) {
            // Verify one row past the accepted-count average, floored at two so a
            // recovering prompt can always re-earn width, and never above the
            // explicit cap when both are set.
            const int adaptive =
                dflash2_width_samples == 0
                    ? static_cast<int>(proposal.tokens.size())
                    : std::max(2, static_cast<int>(dflash2_accept_ewma + 1.5));
            width_limit = width_limit > 0 ? std::min(width_limit, adaptive) : adaptive;
        }
        if (width_limit > 0 &&
            static_cast<int>(draft_tokens_used.size()) > width_limit) {
            draft_tokens_used.resize(static_cast<size_t>(width_limit));
        }
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_started).count();
            stats->proposed_drafts += draft_tokens_used.size();
        }
        std::vector<int> verify_inputs;
        verify_inputs.reserve(draft_tokens_used.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), draft_tokens_used.begin(),
                             draft_tokens_used.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        (void)run_chunk(verify_inputs, committed_position,
                        static_cast<int>(layers.size()), false, &verify);
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
            ++stats->verify_count;
        }
        int correct = 0;
        while (correct < static_cast<int>(draft_tokens_used.size()) &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   draft_tokens_used[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) stats->correct_drafts += static_cast<uint64_t>(correct);
        if (dflash2_adaptive_width) {
            // A block that accepted every row it verified is censored: the true
            // acceptance may be higher, so credit it one extra row to let the
            // width grow back. Otherwise `correct` is the exact observation.
            const double observed =
                correct == static_cast<int>(draft_tokens_used.size())
                    ? static_cast<double>(correct) + 1.0
                    : static_cast<double>(correct);
            constexpr double kAlpha = 0.25;
            dflash2_accept_ewma = dflash2_width_samples == 0
                                      ? observed
                                      : dflash2_accept_ewma +
                                            kAlpha * (observed - dflash2_accept_ewma);
            ++dflash2_width_samples;
        }
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        if (correct != static_cast<int>(draft_tokens_used.size())) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(draft_tokens_used.begin(),
                                    draft_tokens_used.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                      verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = dflash2->committed_position();
        return result;
    }

    QwenForwardResult dspark_speculative_step(int input_token,
                                              QwenMtpStats* stats) {
        if (!dspark_enabled) {
            throw std::runtime_error("Qwen DSpark speculative step is disabled");
        }
        const int committed_position = dspark->committed_position();
        prepare_transaction_state();
        const auto draft_started = std::chrono::steady_clock::now();
        const QwenDSparkProposal proposal = dspark->propose(input_token);
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_started).count();
            stats->proposed_drafts += proposal.tokens.size();
            for (float confidence : proposal.confidences) {
                if (!std::isfinite(confidence)) {
                    throw std::runtime_error(
                        "Qwen DSpark confidence is not finite");
                }
                if (stats->confidence_count == 0) {
                    stats->confidence_min = confidence;
                    stats->confidence_max = confidence;
                } else {
                    stats->confidence_min = std::min(
                        stats->confidence_min, confidence);
                    stats->confidence_max = std::max(
                        stats->confidence_max, confidence);
                }
                stats->confidence_sum += confidence;
                ++stats->confidence_count;
            }
        }
        std::vector<int> verify_inputs;
        verify_inputs.reserve(proposal.tokens.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), proposal.tokens.begin(),
                             proposal.tokens.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        (void)run_chunk(verify_inputs, committed_position,
                        static_cast<int>(layers.size()), false, &verify);
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
            ++stats->verify_count;
        }
        int correct = 0;
        while (correct < static_cast<int>(proposal.tokens.size()) &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   proposal.tokens[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) stats->correct_drafts += correct;
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        if (correct != static_cast<int>(proposal.tokens.size())) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(proposal.tokens.begin(),
                                    proposal.tokens.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                       verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = dspark->committed_position();
        return result;
    }

    QwenForwardResult speculative_step(int input_token, int draft_count,
                                       QwenMtpStats* stats) {
        if (!mtp_enabled || draft_count <= 0) {
            return run_chunk({input_token}, mtp_position,
                             static_cast<int>(layers.size()), true);
        }
        const int committed_position = mtp_position;
        prepare_transaction_state();
        const std::vector<int> drafts = draft_tokens(draft_count, input_token, stats);
        std::vector<int> verify_inputs;
        verify_inputs.reserve(drafts.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), drafts.begin(), drafts.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        (void)run_chunk(verify_inputs, committed_position,
                        static_cast<int>(layers.size()), false, &verify);
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
        }
        int correct = 0;
        while (correct < draft_count &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   drafts[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) {
            ++stats->verify_count;
            stats->correct_drafts += static_cast<uint64_t>(correct);
        }
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        const bool full_accept = correct == draft_count;
        if (!full_accept) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        // Rebind the last committed MTP row to the target bonus. On full accept
        // this fills the one row beyond the recursively proposed drafts; on
        // rejection it overwrites the stale speculative suffix after replay.
        const auto seed_started = std::chrono::steady_clock::now();
        rewrite_mtp_boundary(bonus, committed_position + correct);
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - seed_started).count();
        }
        // The target state now ends after the accepted input sequence. The
        // bonus is the next input and must not be consumed by target yet. The
        // MTP boundary row is already primed with that bonus for the next round.
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(drafts.begin(), drafts.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                       verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = mtp_position;
        return result;
    }

    uint64_t weights_vocab_start() const {
        return static_cast<uint64_t>(options.tp_rank) *
               config.vocab_size / options.tp_world;
    }

    QwenForwardResult run_chunk(const std::vector<int>& token_ids,
                                int position_offset, int active_layers,
                                bool compute_logits,
                                QwenVerifyBatch* verify_batch = nullptr,
                                const std::vector<int>* mtp_shifted_tokens = nullptr) {
        if (token_ids.empty()) {
            throw std::runtime_error("Qwen forward requires at least one token");
        }
        if (position_offset < 0 ||
            position_offset + static_cast<int>(token_ids.size()) > max_context) {
            throw std::runtime_error("Qwen context length exceeded");
        }
        const int rows = static_cast<int>(token_ids.size());
        local_tokens = token_ids;
        allocate(d_tokens, local_tokens.size() * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_cuda(cudaMemcpy(d_tokens.data, local_tokens.data(),
                              local_tokens.size() * sizeof(int),
                              cudaMemcpyHostToDevice), "Qwen token upload");
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const std::vector<uint64_t> hidden_shape = {
            static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)};
        allocate_half(hidden_a, hidden_elements, hidden_shape);
        allocate_half(hidden_b, hidden_elements, hidden_shape);
        require_launch(qwen_embedding_fp16_gather_f16_cuda(
            embed.f16_data(), static_cast<int*>(d_tokens.data),
            hidden_a.f16_data(), rows, hidden_size,
            static_cast<int>(weights_vocab_start()),
            static_cast<int>(embed.shape[0])), "Qwen FP16 embedding lookup");
        all_reduce_half(hidden_a.f16_data(), rows * hidden_size);
        uint16_t* hidden = hidden_a.f16_data();
        uint16_t* output = hidden_b.f16_data();
        const int dspark_tap_count = dspark_enabled
            ? static_cast<int>(dspark_config->target_layer_ids.size()) : 0;
        const std::vector<int>* dflash2_tap_layers = dflash2_enabled
            ? &dflash2_config->target_layer_ids
            : (!dflash2_debug_target_layer_ids.empty()
                   ? &dflash2_debug_target_layer_ids : nullptr);
        const int dflash2_tap_count = dflash2_tap_layers != nullptr
            ? static_cast<int>(dflash2_tap_layers->size()) : 0;
        if (dspark_enabled) {
            allocate_half(
                dspark_target_taps,
                static_cast<size_t>(rows) * hidden_size * dspark_tap_count,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(hidden_size * dspark_tap_count)});
        }
        if (dflash2_tap_count != 0) {
            allocate_half(
                dflash2_target_taps,
                static_cast<size_t>(rows) * hidden_size * dflash2_tap_count,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(hidden_size * dflash2_tap_count)});
        }
        int dspark_tap_index = 0;
        int dflash2_tap_index = 0;
        for (int layer_index = 0; layer_index < active_layers; ++layer_index) {
            layer_forward(layers[static_cast<size_t>(layer_index)], hidden,
                          output, rows, position_offset);
            std::swap(hidden, output);
            if (dspark_enabled && dspark_tap_index < dspark_tap_count &&
                layer_index == dspark_config->target_layer_ids[
                    static_cast<size_t>(dspark_tap_index)]) {
                // Store [row, tap, hidden] as the row-major concat expected by
                // fc.weight [hidden, tap_count * hidden]. One pitched copy keeps
                // the exact layout without issuing one CUDA copy per target row.
                require_launch(qwen_copy_rows_strided_f16_cuda(
                    hidden, hidden_size,
                    dspark_target_taps.f16_data() +
                        static_cast<size_t>(dspark_tap_index) * hidden_size,
                    dspark_tap_count * hidden_size, rows, hidden_size),
                    "Qwen DSpark target tap copy");
                ++dspark_tap_index;
            }
            if (dflash2_tap_layers != nullptr &&
                dflash2_tap_index < dflash2_tap_count &&
                layer_index == (*dflash2_tap_layers)[
                    static_cast<size_t>(dflash2_tap_index)]) {
                require_launch(qwen_copy_rows_strided_f16_cuda(
                    hidden, hidden_size,
                    dflash2_target_taps.f16_data() +
                        static_cast<size_t>(dflash2_tap_index) * hidden_size,
                    dflash2_tap_count * hidden_size, rows, hidden_size),
                    "Qwen DFlash2 target tap copy");
                ++dflash2_tap_index;
            }
        }
        if (dspark_enabled) {
            if (dspark_tap_index != dspark_tap_count) {
                throw std::runtime_error("Qwen DSpark target taps were not captured");
            }
            if (dspark->committed_position() != position_offset) {
                if (position_offset <= dspark->committed_position()) {
                    dspark->crop_context(position_offset);
                } else {
                    throw std::runtime_error(
                        "Qwen DSpark target context has a position gap");
                }
            }
            dspark->append_target_taps(dspark_target_taps.f16_data(), rows,
                                       position_offset);
        }
        if (dflash2_tap_layers != nullptr) {
            if (dflash2_tap_index != dflash2_tap_count) {
                throw std::runtime_error("Qwen DFlash2 target taps were not captured");
            }
            if (dflash2_enabled) {
                if (dflash2->committed_position() != position_offset) {
                    if (position_offset <= dflash2->committed_position()) {
                        dflash2->crop_context(position_offset);
                    } else {
                        throw std::runtime_error(
                            "Qwen DFlash2 target context has a position gap");
                    }
                }
                dflash2->append_target_taps(dflash2_target_taps.f16_data(), rows,
                                            position_offset);
            } else if (dflash2_target_debug_callback) {
                QwenDFlash2DebugTensor tensor;
                tensor.name = "target_taps";
                tensor.dtype = QwenDFlash2DebugDType::F16;
                tensor.shape = dflash2_target_taps.shape;
                tensor.bytes.resize(static_cast<size_t>(dflash2_target_taps.nbytes));
                check_cuda(cudaMemcpy(tensor.bytes.data(), dflash2_target_taps.data,
                                      tensor.bytes.size(), cudaMemcpyDeviceToHost),
                           "Qwen DFlash2 target tap debug copy");
                dflash2_target_debug_callback(tensor);
            }
        }
        if (mtp_enabled) {
            allocate_half(target_hidden_rows, hidden_elements, hidden_shape);
            // Qwen3.5 MTP consumes the target model's returned hidden states,
            // which are after the target final RMSNorm (matching vLLM/SGLang).
            norm(final_norm, hidden, target_hidden_rows.f16_data(), rows,
                 hidden_size);
            allocate_half(target_last_hidden, hidden_size,
                          {config.hidden_size});
            check_cuda(cudaMemcpy(
                           target_last_hidden.data,
                           target_hidden_rows.f16_data() +
                               static_cast<size_t>(rows - 1) * hidden_size,
                           static_cast<size_t>(hidden_size) * sizeof(uint16_t),
                           cudaMemcpyDeviceToDevice),
                       "Qwen target last hidden copy");
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = position_offset + rows;
            if (mtp_shifted_tokens != nullptr) {
                if (mtp_shifted_tokens->size() != token_ids.size()) {
                    throw std::runtime_error(
                        "Qwen MTP shifted-token extent does not match target rows");
                }
                (void)prime_target_mtp(*mtp_shifted_tokens, position_offset);
            }
        }
        if (verify_batch != nullptr) {
            *verify_batch = target_logits_for(
                hidden, rows, position_offset + rows);
        }
        if (compute_logits) {
            if (verify_batch != nullptr) {
                QwenForwardResult result;
                result.layers = active_layers;
                result.dim = hidden_size;
                result.logits = static_cast<int>(config.vocab_size);
                result.top_token = verify_batch->top_tokens.back();
                result.top_logit = verify_batch->top_logits.back();
                result.checksum = verify_batch->local_logits.back();
                result.position = position_offset + rows;
                return result;
            }
            return logits_for(
                hidden, rows - 1, position_offset + rows, active_layers);
        }
        QwenForwardResult result;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.position = position_offset + rows;
        return result;
    }

    uint64_t activation_capacity_bytes() const {
        return hidden_a.capacity + hidden_b.capacity + d_tokens.capacity +
               argmax_token.capacity + argmax_logit.capacity +
               target_hidden_rows.capacity + target_last_hidden.capacity +
               mtp_seed_hidden.capacity + mtp_embedding.capacity +
               mtp_normalized_embedding.capacity +
               mtp_normalized_hidden.capacity + mtp_concat.capacity +
               mtp_fused.capacity + mtp_next_hidden.capacity +
               mtp_normalized_output.capacity + dspark_target_taps.capacity +
               dflash2_target_taps.capacity + workspace.capacity_bytes() +
               (dspark != nullptr ? dspark->activation_workspace_bytes() : 0) +
               (dflash2 != nullptr ? dflash2->activation_workspace_bytes() : 0);
    }
};

QwenEngine::QwenEngine(const std::string& ckpt_dir,
                       const QwenEngineOptions& options, int layer_count,
                       int max_context)
    : ckpt_dir_(ckpt_dir), options_(options),
      config_(QwenConfig::from_hf_config(ckpt_dir)), index_(ckpt_dir),
      weights_(index_, config_, options.tp_world, options.tp_rank) {
    if (options_.device < 0) options_.device = options_.tp_rank;
    const int external_drafter_count =
        (!options_.dspark_checkpoint.empty() ? 1 : 0) +
        (!options_.dflash2_checkpoint.empty() ? 1 : 0);
    if (options_.mtp && external_drafter_count != 0) {
        throw std::runtime_error(
            "Qwen native MTP cannot be combined with an external drafter");
    }
    if (external_drafter_count > 1) {
        throw std::runtime_error(
            "Qwen DSpark and DFlash2 are mutually exclusive");
    }
    if ((options_.mtp || external_drafter_count != 0) && layer_count > 0 &&
        layer_count != static_cast<int>(config_.num_hidden_layers)) {
        throw std::runtime_error(
            "Qwen speculative decoding requires the complete target model; partial "
            "--smoke-layers would verify drafts against a different model");
    }
    if (options_.prefill_chunk_tokens <= 0) {
        throw std::runtime_error("Qwen prefill chunk size must be positive");
    }
    if (options_.attention_window < 0 || options_.attention_sink_tokens < 0) {
        throw std::runtime_error("Qwen attention window and sink must not be negative");
    }
    if (options_.attention_window == 0 && options_.attention_sink_tokens != 0) {
        throw std::runtime_error(
            "Qwen attention sink requires a nonzero attention window");
    }
    if (options_.kv_cache_dtype == QwenKvCacheDType::Fp8 &&
        options_.attention_window > 0) {
        throw std::runtime_error(
            "Qwen sparse attention currently requires an FP16 KV cache");
    }
    if (cudaSetDevice(options_.device) != cudaSuccess) {
        throw std::runtime_error("failed to set Qwen CUDA device");
    }
    if (layer_count <= 0) {
        active_layers_ = static_cast<int>(config_.num_hidden_layers);
    } else if (layer_count <= static_cast<int>(config_.num_hidden_layers)) {
        active_layers_ = layer_count;
    } else {
        throw std::runtime_error("Qwen layer count exceeds model depth");
    }
    max_context_ = max_context > 0
        ? max_context : static_cast<int>(config_.max_position_embeddings);
    if (max_context_ <= 0 ||
        max_context_ > static_cast<int>(config_.max_position_embeddings)) {
        throw std::runtime_error("Qwen max context exceeds model configuration");
    }
    impl_ = new Impl(index_, config_, weights_, options_, max_context_,
                     active_layers_);
    resident_weight_bytes_ = impl_->uploaded_weight_bytes;
    resident_scale_bytes_ = impl_->uploaded_scale_bytes;
}

QwenEngine::~QwenEngine() {
    delete impl_;
    impl_ = nullptr;
}

uint64_t QwenEngine::activation_workspace_peak_bytes() const {
    return impl_->activation_capacity_bytes();
}

uint64_t QwenEngine::kv_cache_bytes() const {
    return impl_->cache_data_bytes;
}

uint64_t QwenEngine::kv_cache_scale_bytes() const {
    return impl_->cache_scale_bytes;
}

void QwenEngine::set_dflash2_debug_callback(
    QwenDFlash2DebugCallback callback) {
    if (!impl_->dflash2_enabled || impl_->dflash2 == nullptr) {
        throw std::runtime_error("Qwen DFlash2 debug callback requires --qwen-dflash2");
    }
    impl_->dflash2->set_debug_callback(std::move(callback));
}

QwenForwardResult QwenEngine::debug_prefill_dflash2(
    const std::vector<int>& token_ids,
    const std::vector<int>& target_layer_ids,
    QwenDFlash2DebugCallback callback) {
    if (!callback || target_layer_ids.empty()) {
        throw std::runtime_error("Qwen DFlash2 debug prefill requires target taps");
    }
    int previous = -1;
    for (int layer : target_layer_ids) {
        if (layer <= previous || layer < 0 || layer >= active_layers_) {
            throw std::runtime_error("invalid Qwen DFlash2 debug target layer IDs");
        }
        previous = layer;
    }
    clear_prefix_cache();
    impl_->dflash2_debug_target_layer_ids = target_layer_ids;
    impl_->dflash2_target_debug_callback = std::move(callback);
    try {
        QwenForwardResult result = prefill(token_ids);
        impl_->dflash2_debug_target_layer_ids.clear();
        impl_->dflash2_target_debug_callback = {};
        return result;
    } catch (...) {
        impl_->dflash2_debug_target_layer_ids.clear();
        impl_->dflash2_target_debug_callback = {};
        throw;
    }
}

void QwenEngine::warmup_tp() {
    if (options_.tp_world == 1) return;
    QwenDeviceTensor scratch;
    allocate_half(scratch, 1, {1});
    zero_tensor(scratch);
    impl_->all_reduce_half(scratch.f16_data(), 1);
    check_cuda(cudaDeviceSynchronize(), "Qwen TP warmup synchronization");
}

void QwenEngine::reset() {
    clear_prefix_cache();
}

void QwenEngine::clear_prefix_cache() {
    position_ = 0;
    has_cached_result_ = false;
    cached_result_ = QwenForwardResult{};
    prefix_stats_ = QwenPrefixCacheStats{};
    impl_->cached_prompt.clear();
    impl_->snapshots.clear();
    impl_->prefix_hits = 0;
    impl_->prefix_misses = 0;
    impl_->zero_recurrent_state();
    // KV storage is overwritten before it is read. Avoid clearing several GiB
    // on every request; position_ bounds all attention reads.
}

QwenForwardResult QwenEngine::prefill(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        throw std::runtime_error("Qwen prefill requires at least one token");
    }
    if (token_ids.size() > static_cast<size_t>(max_context_)) {
        throw std::runtime_error("Qwen context length exceeded");
    }

    prefix_stats_ = QwenPrefixCacheStats{};
    prefix_stats_.prompt_tokens = static_cast<int>(token_ids.size());
    impl_->phase_seconds.clear();
    impl_->phase_calls.clear();
    const bool can_reuse = options_.prefix_cache &&
        !impl_->cached_prompt.empty() && position_ ==
            static_cast<int>(impl_->cached_prompt.size());
    size_t common = 0;
    if (can_reuse) {
        const size_t limit = std::min(token_ids.size(), impl_->cached_prompt.size());
        while (common < limit && token_ids[common] == impl_->cached_prompt[common]) {
            ++common;
        }
    }
    prefix_stats_.matched_tokens = static_cast<int>(common);

    // Exact repeat: the cached final logits are already the requested result.
    if (can_reuse && common == token_ids.size() &&
        token_ids.size() == impl_->cached_prompt.size() && has_cached_result_) {
        ++impl_->prefix_hits;
        prefix_stats_.hits = impl_->prefix_hits;
        prefix_stats_.misses = impl_->prefix_misses;
        prefix_stats_.reused_tokens = static_cast<int>(token_ids.size());
        prefix_stats_.resume_source = "live";
        prefix_stats_.snapshots = static_cast<int>(impl_->snapshots.size());
        prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
        return cached_result_;
    }

    int start_position = 0;
    const QwenRecurrentSnapshot* resume_snapshot = nullptr;
    if (can_reuse && common == impl_->cached_prompt.size() &&
        common <= token_ids.size()) {
        // The current recurrent state is exactly the state after the common
        // prefix. This is the hot path for monotonically growing prompts.
        start_position = static_cast<int>(common);
        prefix_stats_.resume_source = "live";
        ++impl_->prefix_hits;
    } else if (options_.prefix_cache && common > 0) {
        // For a branch or a shorter prompt, restore the deepest safe snapshot.
        // A request-boundary snapshot also carries the final-token result, so
        // an exact shorter-prefix request can finish without recomputation.
        resume_snapshot = impl_->snapshot_at_or_before(
            static_cast<int>(common));
        if (resume_snapshot != nullptr &&
            resume_snapshot->position == static_cast<int>(common) &&
            resume_snapshot->has_result && token_ids.size() == common) {
            impl_->restore_recurrent_state(*resume_snapshot);
            position_ = static_cast<int>(common);
            impl_->cached_prompt = token_ids;
            cached_result_ = resume_snapshot->result;
            has_cached_result_ = true;
            impl_->drop_snapshots_after(position_);
            prefix_stats_.resume_source = "snapshot";
            prefix_stats_.reused_tokens = static_cast<int>(common);
            prefix_stats_.snapshots = static_cast<int>(impl_->snapshots.size());
            prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
            ++impl_->prefix_hits;
            prefix_stats_.hits = impl_->prefix_hits;
            prefix_stats_.misses = impl_->prefix_misses;
            return cached_result_;
        }
        if (resume_snapshot != nullptr &&
            resume_snapshot->position == static_cast<int>(token_ids.size())) {
            resume_snapshot = impl_->snapshot_strictly_before(
                static_cast<int>(token_ids.size()));
        }
        if (resume_snapshot != nullptr) {
            impl_->restore_recurrent_state(*resume_snapshot);
            start_position = resume_snapshot->position;
            prefix_stats_.resume_source = "snapshot";
            ++impl_->prefix_hits;
        } else {
            impl_->zero_recurrent_state();
            prefix_stats_.resume_source = "empty";
            ++impl_->prefix_misses;
        }
        impl_->drop_snapshots_after(start_position);
    } else {
        impl_->zero_recurrent_state();
        prefix_stats_.resume_source = "empty";
        if (can_reuse) ++impl_->prefix_misses;
        impl_->drop_snapshots_after(0);
    }

    const int target_position = static_cast<int>(token_ids.size());
    if (options_.mtp && start_position > 0 && start_position < target_position) {
        // The cached MTP row at S-1 was paired with the prior request's token
        // x[S]. An append may reuse that token, while a branch or compression
        // may replace it. Rebind the shifted-token boundary before priming the
        // new suffix so every predictor row observes target h[S-1] + new x[S].
        impl_->rewrite_mtp_boundary(
            token_ids[static_cast<size_t>(start_position)],
            start_position - 1);
    }
    QwenForwardResult result;
    const int chunk_size = std::max(1, options_.prefill_chunk_tokens);
    for (int offset = start_position; offset < target_position;) {
        int end = std::min(target_position, offset + chunk_size);
        // Split at exact snapshot boundaries even when a request begins at an
        // arbitrary position. Otherwise a 512-token append starting at 100 can
        // step over 4096 forever and never create the rollback point.
        if (options_.state_snapshot_interval_tokens > 0) {
            const int next_snapshot = impl_->next_periodic_snapshot_after(offset);
            if (next_snapshot > offset && next_snapshot < end) {
                end = next_snapshot;
            }
        }
        const bool periodic_snapshot =
            impl_->is_periodic_snapshot_position(end);
        // Periodic snapshots may later serve an exact shorter-prefix request.
        // Keep the final-row logits with those snapshots so restoring one does
        // not have to recompute the whole prefix just to recover its result.
        const bool snapshot_result = end == target_position || periodic_snapshot;
        std::vector<int> chunk(
            token_ids.begin() + static_cast<ptrdiff_t>(offset),
            token_ids.begin() + static_cast<ptrdiff_t>(end));
        if (options_.mtp) {

            std::vector<int> shifted;
            shifted.reserve(chunk.size());
            for (int position = offset; position < end; ++position) {
                shifted.push_back(
                    position + 1 < target_position
                        ? token_ids[static_cast<size_t>(position + 1)]
                        : -1);
            }
            if (shifted.back() < 0) {
                // The last shifted input is the target token predicted by this
                // final prompt row, so obtain its logits before priming MTP.
                result = impl_->run_chunk(chunk, offset, active_layers_, true);
                shifted.back() = result.top_token;
                (void)impl_->prime_target_mtp(shifted, offset);
            } else {
                result = impl_->run_chunk(chunk, offset, active_layers_,
                                          snapshot_result, nullptr, &shifted);
            }
        } else {
            result = impl_->run_chunk(chunk, offset, active_layers_,
                                      snapshot_result);
        }
        if (periodic_snapshot || end == target_position) {
            impl_->record_snapshot(end, &result, periodic_snapshot);
        }

        offset = end;
    }

    // A same-length prompt resumed from an interior snapshot always computes
    // at least one token; the only zero-work case returned above was exact hit.
    if (start_position == target_position) {
        throw std::runtime_error("Qwen prefix cache failed to produce logits");
    }
    position_ = target_position;
    if (options_.prefix_cache) {
        impl_->cached_prompt = token_ids;
        cached_result_ = result;
        has_cached_result_ = true;
    } else {
        impl_->cached_prompt.clear();
        has_cached_result_ = false;
    }
    prefix_stats_.reused_tokens = start_position;
    prefix_stats_.computed_tokens = target_position - start_position;
    prefix_stats_.snapshots = static_cast<int>(impl_->snapshots.size());
    prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
    prefix_stats_.hits = impl_->prefix_hits;
    prefix_stats_.misses = impl_->prefix_misses;
    impl_->report_phase_profile("prefill");
    return result;
}

QwenForwardResult QwenEngine::decode_step(int token_id) {
    if (position_ >= max_context_) {
        throw std::runtime_error("Qwen context length exceeded");
    }
    QwenForwardResult result = impl_->run_chunk(
        {token_id}, position_, active_layers_, true);
    ++position_;
    if (options_.prefix_cache) {
        impl_->cached_prompt.push_back(token_id);
        cached_result_ = result;
        has_cached_result_ = true;
        if (impl_->is_periodic_snapshot_position(position_)) {
            impl_->record_snapshot(position_, &result, true);
        }
    }
    return result;
}

std::vector<QwenForwardResult> QwenEngine::generate(
    const std::vector<int>& prompt_ids, int max_new_tokens) {
    if (max_new_tokens <= 0) return {};
    if (prompt_ids.empty()) {
        throw std::runtime_error("Qwen generation requires a non-empty prompt");
    }
    if (prompt_ids.size() + static_cast<size_t>(max_new_tokens) >
        static_cast<size_t>(max_context_)) {
        throw std::runtime_error("Qwen prompt plus generation exceeds context");
    }
    mtp_stats_ = QwenMtpStats{};
    const auto prefill_started = std::chrono::steady_clock::now();
    QwenForwardResult next = prefill(prompt_ids);
    mtp_stats_.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    // Prefill already reported; reset so the decode/verify phases are attributed
    // on their own rather than buried under the prompt pass.
    impl_->phase_seconds.clear();
    impl_->phase_calls.clear();
    std::vector<QwenForwardResult> results;
    results.reserve(static_cast<size_t>(max_new_tokens));
    if (!options_.mtp && options_.dspark_checkpoint.empty() &&
        options_.dflash2_checkpoint.empty()) {
        for (int index = 0; index < max_new_tokens; ++index) {
            results.push_back(next);
            if (index + 1 < max_new_tokens) next = decode_step(next.top_token);
        }
        impl_->report_phase_profile("decode");
        return results;
    }

    // `prefill()` predicts the first output token without consuming it, just as
    // the plain path does. A speculative transaction consumes that token and
    // any correct drafts, then returns a bonus token which remains unconsumed
    // until the next transaction.
    results.push_back(next);
    int current_token = next.top_token;
    const bool use_dspark = !options_.dspark_checkpoint.empty();
    const bool use_dflash2 = !options_.dflash2_checkpoint.empty();
    const bool use_external_drafter = use_dspark || use_dflash2;
    const int max_draft_tokens = use_external_drafter
        ? 7 : std::max(1, options_.mtp_speculative_tokens);
    int adaptive_draft_tokens = use_external_drafter
        ? 7 : (options_.mtp_adaptive ? 1 : max_draft_tokens);
    int full_accept_streak = 0;
    while (static_cast<int>(results.size()) < max_new_tokens) {
        const int remaining = max_new_tokens - static_cast<int>(results.size());
        if (remaining == 1) {
            next = decode_step(current_token);
            results.push_back(next);
            break;
        }
        if (use_external_drafter && remaining < 8) {
            // The published checkpoint is trained for a fixed seven-row block;
            // use exact plain decode for a short output tail rather than changing
            // its attention semantics or returning unnecessary tokens.
            while (static_cast<int>(results.size()) < max_new_tokens) {
                next = decode_step(current_token);
                results.push_back(next);
                current_token = next.top_token;
            }
            break;
        }
        const int proposed = std::min(adaptive_draft_tokens, remaining - 1);
        if (use_dspark) {
            next = impl_->dspark_speculative_step(current_token, &mtp_stats_);
        } else if (use_dflash2) {
            next = impl_->dflash2_speculative_step(current_token, &mtp_stats_);
        } else {
            next = impl_->speculative_step(current_token, proposed, &mtp_stats_);
        }
        const int correct = next.correct_drafts;
        if (!use_external_drafter && options_.mtp_adaptive) {
            if (correct == proposed) {
                ++full_accept_streak;
                if (full_accept_streak >= 1 &&
                    adaptive_draft_tokens < max_draft_tokens) {
                    adaptive_draft_tokens = std::min(
                        max_draft_tokens, adaptive_draft_tokens * 2);
                    full_accept_streak = 0;
                }
            } else {
                adaptive_draft_tokens = std::max(
                    1, std::min(adaptive_draft_tokens / 2, correct + 1));
                full_accept_streak = 0;
            }
        }
        const int consumed = 1 + correct;
        position_ += consumed;
        if (options_.prefix_cache) {
            impl_->cached_prompt.push_back(current_token);
            impl_->cached_prompt.insert(impl_->cached_prompt.end(),
                                        next.accept_tokens.begin(),
                                        next.accept_tokens.end());
            cached_result_ = next;
            has_cached_result_ = true;
        }

        for (size_t index = 0; index < next.accept_tokens.size(); ++index) {
            if (static_cast<int>(results.size()) == max_new_tokens) break;
            QwenForwardResult token_result = next;
            token_result.top_token = next.accept_tokens[index];
            token_result.top_logit = next.accept_logits[index];
            token_result.checksum = next.accept_checksums[index];
            token_result.position = position_ - correct + static_cast<int>(index);
            results.push_back(std::move(token_result));
        }
        if (static_cast<int>(results.size()) < max_new_tokens) {
            results.push_back(next);
        }
        current_token = next.bonus_token;
    }
    impl_->report_phase_profile("spec_decode");
    return results;
}

}  // namespace dsv4
