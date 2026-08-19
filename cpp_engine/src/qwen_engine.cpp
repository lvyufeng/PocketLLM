#include "qwen_engine.hpp"

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"
#include "tp_comm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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
    bool periodic = false;
    // A request-boundary snapshot can answer an exact shorter-prefix prefill
    // without re-running its final token.
    QwenForwardResult result;
    bool has_result = false;

    uint64_t bytes() const {
        return state.capacity + conv_tail.capacity;
    }
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
    QwenDeviceTensor embed;
    QwenDeviceTensor final_norm;
    QwenDeviceTensor lm_head;
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
    }

    // Copies the recurrent half of the network to device-resident snapshots.
    // Only the 48 DeltaNet layers carry order-dependent state; full attention
    // is skipped because its KV cache is addressed by absolute position.
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

    void restore_recurrent_state(const QwenRecurrentSnapshot& snapshot) {
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
        if (!has_recurrent_state()) return;
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

    void all_reduce_half(uint16_t* values, int count) {
        if (options.tp_world == 1) return;
#ifdef DSV4_HAVE_NCCL
        if (options.nccl_id_path.empty()) {
            throw std::runtime_error("Qwen TP requires --nccl-id-path");
        }
        nccl_all_reduce_sum_f16_inplace(
            options.tp_world, options.tp_rank, options.device,
            options.nccl_id_path.c_str(), values, count);
#else
        (void)values;
        (void)count;
        throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
#endif
    }

    void projection(const DeviceLinear& linear, const uint16_t* input,
                    uint16_t* output, int rows) {
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
        const bool sequenced = rows > 1 && qwen_gated_delta_sequence_f16_cuda(
            layer.linear.state.f32_data(), q.f16_data(), k.f16_data(),
            v.f16_data(), gates.f16_data(), beta.f16_data(), core.f16_data(),
            rows, value_heads, key_heads,
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
        const bool fused_swiglu = rows == 1 && layer.gate.fp8 && layer.up.fp8 &&
            layer.gate.weight.shape == layer.up.weight.shape &&
            layer.gate.scale.shape == layer.up.scale.shape;
        QwenDeviceTensor* gate = nullptr;
        QwenDeviceTensor* up = nullptr;
        if (!fused_swiglu) {
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
        if (fused_swiglu) {
            require_launch(qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                post.f16_data(), layer.gate.weight.fp8_data(),
                layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                layer.up.scale.f16_data(), intermediate.f16_data(),
                static_cast<int>(layer.gate.weight.shape[0]), hidden_size,
                hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
                "FP16 fused decode SwiGLU");
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

    QwenForwardResult logits_for(const uint16_t* hidden, int last_row,
                                 int position_after, int active_layers) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        const int local_vocab = static_cast<int>(lm_head.shape[0]);
        begin_workspace();
        QwenDeviceTensor& normalized = workspace_half(
            hidden_size, {static_cast<uint64_t>(hidden_size)});
        norm(final_norm, hidden + static_cast<size_t>(last_row) * hidden_size,
             normalized.f16_data(), 1, hidden_size);
        QwenDeviceTensor& local_logits = workspace_float(
            local_vocab, {static_cast<uint64_t>(local_vocab)});
        require_launch(qwen_fp16_matmul_rows_f16_f32_cuda(
            normalized.f16_data(), lm_head.f16_data(), local_logits.f32_data(),
            1, local_vocab, hidden_size, hidden_size, local_vocab, hidden_size),
            "Qwen final FP32 logits");
        allocate(argmax_token, sizeof(int), {1}, SafeDType::I64);
        allocate_float(argmax_logit, 1, {1});
        const int vocab_start = static_cast<int>(weights_vocab_start());
        require_launch(argmax_fp32_cuda(
            local_logits.f32_data(), static_cast<int*>(argmax_token.data),
            argmax_logit.f32_data(), local_vocab, vocab_start),
            "Qwen local argmax");
        check_cuda(cudaDeviceSynchronize(), "Qwen logits synchronization");
        int local_token = 0;
        float local_logit = 0.0f;
        check_cuda(cudaMemcpy(&local_token, argmax_token.data, sizeof(local_token),
                              cudaMemcpyDeviceToHost),
                   "Qwen argmax token copy");
        check_cuda(cudaMemcpy(&local_logit, argmax_logit.data, sizeof(local_logit),
                              cudaMemcpyDeviceToHost),
                   "Qwen argmax logit copy");
        int top_token = local_token;
        float top_logit = local_logit;
#ifdef DSV4_HAVE_NCCL
        if (options.tp_world > 1) {
            if (options.nccl_id_path.empty()) {
                throw std::runtime_error("Qwen TP requires --nccl-id-path");
            }
            const TpTopResult global = nccl_global_top1(
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path.c_str(), local_token, local_logit);
            top_token = global.token;
            top_logit = global.logit;
        }
#else
        if (options.tp_world > 1) {
            throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
        }
#endif
        QwenForwardResult result;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = top_token;
        result.top_logit = top_logit;
        result.checksum = local_logit;
        result.position = position_after;
        return result;
    }

    uint64_t weights_vocab_start() const {
        return static_cast<uint64_t>(options.tp_rank) *
               config.vocab_size / options.tp_world;
    }

    QwenForwardResult run_chunk(const std::vector<int>& token_ids,
                                int position_offset, int active_layers,
                                bool compute_logits) {
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
        for (int layer_index = 0; layer_index < active_layers; ++layer_index) {
            layer_forward(layers[static_cast<size_t>(layer_index)], hidden,
                          output, rows, position_offset);
            std::swap(hidden, output);
        }
        if (compute_logits) {
            return logits_for(hidden, rows - 1, position_offset + rows,
                              active_layers);
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
               workspace.capacity_bytes();
    }
};

QwenEngine::QwenEngine(const std::string& ckpt_dir,
                       const QwenEngineOptions& options, int layer_count,
                       int max_context)
    : ckpt_dir_(ckpt_dir), options_(options),
      config_(QwenConfig::from_hf_config(ckpt_dir)), index_(ckpt_dir),
      weights_(index_, config_, options.tp_world, options.tp_rank) {
    if (options_.device < 0) options_.device = options_.tp_rank;
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
        std::vector<int> chunk(
            token_ids.begin() + static_cast<ptrdiff_t>(offset),
            token_ids.begin() + static_cast<ptrdiff_t>(end));
        result = impl_->run_chunk(chunk, offset, active_layers_,
                                  end == target_position);
        const bool periodic_snapshot =
            impl_->is_periodic_snapshot_position(end);
        if (periodic_snapshot || end == target_position) {
            impl_->record_snapshot(end, end == target_position ? &result : nullptr,
                                   periodic_snapshot);
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
    QwenForwardResult next = prefill(prompt_ids);
    std::vector<QwenForwardResult> results;
    results.reserve(static_cast<size_t>(max_new_tokens));
    for (int index = 0; index < max_new_tokens; ++index) {
        results.push_back(next);
        if (index + 1 < max_new_tokens) next = decode_step(next.top_token);
    }
    return results;
}

}  // namespace dsv4
