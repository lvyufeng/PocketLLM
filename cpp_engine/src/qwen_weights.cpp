#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace dsv4 {
namespace {

uint64_t ceil_div(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::string shape_string(const std::vector<uint64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

void require_tp(int world, int rank) {
    if (world <= 0) throw std::runtime_error("Qwen TP world must be positive");
    if (rank < 0 || rank >= world) throw std::runtime_error("Qwen TP rank is out of range");
}

void shard_range(uint64_t total, int world, int rank, uint64_t* start, uint64_t* size) {
    if (total == 0 || total % static_cast<uint64_t>(world) != 0) {
        throw std::runtime_error("Qwen tensor dimension is not divisible by TP world");
    }
    *size = total / static_cast<uint64_t>(world);
    *start = *size * static_cast<uint64_t>(rank);
}

}  // namespace

SafeDType qwen_device_dtype(SafeDType storage_dtype) {
    // RTX 2080 Ti has no native BF16 arithmetic/storage path. Keep FP8 codes
    // compressed for online unpack, and materialize every checkpoint BF16 tensor
    // as FP16 before uploading it to the device.
    return storage_dtype == SafeDType::BF16 ? SafeDType::F16 : storage_dtype;
}

uint16_t qwen_bf16_to_fp16_bits(uint16_t bits) {
    // Widen the BF16 value to an IEEE FP32 bit pattern and use the standard
    // round-to-nearest-even FP32 -> FP16 conversion. This handles subnormals,
    // infinities, NaNs, and overflow without relying on host compiler half types.
    const uint32_t value = static_cast<uint32_t>(bits) << 16;
    const uint32_t sign = (value >> 16) & 0x8000u;
    const int exponent = static_cast<int>((value >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = value & 0x007fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift = 14 - exponent;
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<uint16_t>(sign | half_mantissa);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}

void qwen_convert_bf16_to_fp16(const uint16_t* src, uint16_t* dst, size_t count) {
    if (src == nullptr || dst == nullptr) throw std::invalid_argument("null BF16/FP16 conversion buffer");
    for (size_t i = 0; i < count; ++i) dst[i] = qwen_bf16_to_fp16_bits(src[i]);
}

QwenDeviceTensor::~QwenDeviceTensor() {
    if (data != nullptr) cudaFree(data);
}

QwenDeviceTensor::QwenDeviceTensor(QwenDeviceTensor&& other) noexcept
    : data(other.data), device_dtype(other.device_dtype), shape(std::move(other.shape)),
      nbytes(other.nbytes), capacity(other.capacity) {
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
}

QwenDeviceTensor& QwenDeviceTensor::operator=(QwenDeviceTensor&& other) noexcept {
    if (this == &other) return *this;
    if (data != nullptr) cudaFree(data);
    data = other.data;
    device_dtype = other.device_dtype;
    shape = std::move(other.shape);
    nbytes = other.nbytes;
    capacity = other.capacity;
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
    return *this;
}

float* QwenDeviceTensor::f32_data() {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<float*>(data);
}

const float* QwenDeviceTensor::f32_data() const {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<const float*>(data);
}

uint16_t* QwenDeviceTensor::f16_data() {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<uint16_t*>(data);
}

const uint16_t* QwenDeviceTensor::f16_data() const {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<const uint16_t*>(data);
}

uint8_t* QwenDeviceTensor::fp8_data() {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::fp8_data() const {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<const uint8_t*>(data);
}

QwenHostTensor qwen_materialize_host_tensor(const SafeTensorsIndex& index,
                                            const QwenTensorRef& ref) {
    if (!ref.found) throw std::runtime_error("cannot materialize an absent Qwen tensor: " + ref.name);
    if (ref.full_shape.empty() || ref.local_shape.empty()) {
        throw std::runtime_error("cannot materialize empty Qwen tensor shape: " + ref.name);
    }
    SafeTensorsShard shard(index.shard_path(ref.shard_name));
    const SafeTensorInfo* info = shard.find_tensor(ref.name);
    if (info == nullptr) throw std::runtime_error("Qwen tensor missing while materializing: " + ref.name);
    if (info->shape != ref.full_shape || info->dtype != ref.dtype) {
        throw std::runtime_error("Qwen tensor metadata changed while materializing: " + ref.name);
    }

    QwenHostTensor out;
    out.storage_dtype = ref.dtype;
    out.device_dtype = ref.device_dtype;
    out.shape = ref.local_shape;
    const uint64_t output_numel = safe_tensor_numel(ref.local_shape);
    const uint64_t device_item_size = safe_dtype_size(ref.device_dtype);
    out.bytes.resize(static_cast<size_t>(output_numel * device_item_size));
    const uint64_t source_item_size = safe_dtype_size(ref.dtype);
    if (source_item_size == 0 || device_item_size == 0) {
        throw std::runtime_error("unsupported Qwen materialization dtype: " + ref.name);
    }

    const uint8_t* source = shard.tensor_data(*info);
    uint8_t* destination = out.bytes.data();
    const uint64_t source_row_numel = safe_tensor_numel(
        std::vector<uint64_t>(ref.full_shape.begin() + 1, ref.full_shape.end()));
    const uint64_t local_row_numel = safe_tensor_numel(
        std::vector<uint64_t>(ref.local_shape.begin() + 1, ref.local_shape.end()));
    const uint64_t source_row_bytes = source_row_numel * source_item_size;
    const uint64_t local_row_bytes = local_row_numel * device_item_size;

    auto copy_bytes = [&](const uint8_t* src, uint8_t* dst, uint64_t elements) {
        if (ref.dtype == SafeDType::BF16) {
            if (ref.device_dtype != SafeDType::F16) {
                throw std::runtime_error("Qwen BF16 tensor has non-FP16 device dtype: " + ref.name);
            }
            qwen_convert_bf16_to_fp16(reinterpret_cast<const uint16_t*>(src),
                                      reinterpret_cast<uint16_t*>(dst),
                                      static_cast<size_t>(elements));
        } else {
            std::memcpy(dst, src, static_cast<size_t>(elements * source_item_size));
        }
    };

    const bool packed = ref.rule == QwenShardRule::PackedQkvColumnParallel ||
                        ref.rule == QwenShardRule::PackedConvChannelParallel;
    if (packed) {
        if (ref.segments.empty() || ref.full_shape[0] == 0 || ref.local_shape[0] == 0) {
            throw std::runtime_error("Qwen packed tensor has no segments: " + ref.name);
        }
        uint64_t output_row = 0;
        for (const auto& segment : ref.segments) {
            const uint64_t source_row = segment.first;
            const uint64_t rows = segment.second;
            if (source_row + rows > ref.full_shape[0] || output_row + rows > ref.local_shape[0]) {
                throw std::runtime_error("Qwen packed tensor segment is out of bounds: " + ref.name);
            }
            copy_bytes(source + source_row * source_row_bytes,
                       destination + output_row * local_row_bytes,
                       rows * source_row_numel);
            output_row += rows;
        }
        if (output_row != ref.local_shape[0]) {
            throw std::runtime_error("Qwen packed tensor segments do not fill local shape: " + ref.name);
        }
        return out;
    }

    const int shard_dim = ref.shard_dim;
    if (ref.rule == QwenShardRule::Replicated || ref.shard_size == 0) {
        copy_bytes(source, destination, safe_tensor_numel(ref.full_shape));
        return out;
    }
    if (shard_dim < 0 || static_cast<size_t>(shard_dim) >= ref.full_shape.size()) {
        throw std::runtime_error("Qwen materialization has invalid shard dimension: " + ref.name);
    }
    if (shard_dim == 0) {
        const uint64_t rows = ref.shard_size;
        if (ref.shard_start + rows > ref.full_shape[0] || ref.local_shape[0] != rows) {
            throw std::runtime_error("Qwen dim-0 shard shape mismatch: " + ref.name);
        }
        copy_bytes(source + ref.shard_start * source_row_bytes, destination,
                   rows * source_row_numel);
        return out;
    }
    if (shard_dim == 1 && ref.full_shape.size() == 2) {
        const uint64_t rows = ref.full_shape[0];
        const uint64_t source_cols = ref.full_shape[1];
        const uint64_t local_cols = ref.local_shape[1];
        if (ref.shard_start + ref.shard_size > source_cols || local_cols != ref.shard_size) {
            throw std::runtime_error("Qwen dim-1 shard shape mismatch: " + ref.name);
        }
        for (uint64_t row = 0; row < rows; ++row) {
            const uint8_t* src = source + row * source_cols * source_item_size +
                                 ref.shard_start * source_item_size;
            uint8_t* dst = destination + row * local_cols * device_item_size;
            copy_bytes(src, dst, local_cols);
        }
        return out;
    }
    throw std::runtime_error("unsupported Qwen materialization rank/dimension: " + ref.name);
}

QwenDeviceTensor qwen_upload_tensor_cuda(const SafeTensorsIndex& index,
                                         const QwenTensorRef& ref,
                                         void* stream) {
    QwenHostTensor host = qwen_materialize_host_tensor(index, ref);
    QwenDeviceTensor device;
    device.device_dtype = host.device_dtype;
    device.shape = host.shape;
    device.nbytes = host.bytes.size();
    device.capacity = device.nbytes;
    if (device.nbytes == 0 || cudaMalloc(&device.data, device.nbytes) != cudaSuccess) {
        throw std::runtime_error("failed to allocate Qwen device tensor: " + ref.name);
    }
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const cudaError_t status = cudaMemcpyAsync(device.data, host.bytes.data(), device.nbytes,
                                               cudaMemcpyHostToDevice, cuda_stream);
    if (status != cudaSuccess) {
        cudaFree(device.data);
        device.data = nullptr;
        device.nbytes = 0;
        device.capacity = 0;
        throw std::runtime_error("failed to upload Qwen device tensor: " + ref.name);
    }
    return device;
}

namespace {

QwenTensorRef make_ref(const std::string& name, const std::string& shard_name,
                       const SafeTensorInfo& info, QwenShardRule rule,
                       int shard_dim, uint64_t start, uint64_t size,
                       const std::vector<uint64_t>& local_shape) {
    QwenTensorRef ref;
    ref.name = name;
    ref.shard_name = shard_name;
    ref.dtype = info.dtype;
    ref.device_dtype = qwen_device_dtype(info.dtype);
    ref.full_shape = info.shape;
    ref.local_shape = local_shape;
    ref.rule = rule;
    ref.shard_dim = shard_dim;
    ref.shard_start = start;
    ref.shard_size = size;
    ref.nbytes = safe_tensor_numel(local_shape) * safe_dtype_size(info.dtype);
    ref.device_nbytes = safe_tensor_numel(local_shape) * safe_dtype_size(ref.device_dtype);
    ref.found = true;
    return ref;
}

}  // namespace

const char* qwen_shard_rule_name(QwenShardRule rule) {
    switch (rule) {
        case QwenShardRule::Replicated: return "replicated";
        case QwenShardRule::ColumnParallel: return "column_parallel";
        case QwenShardRule::RowParallel: return "row_parallel";
        case QwenShardRule::ParallelEmbedding: return "parallel_embedding";
        case QwenShardRule::ParallelHead: return "parallel_head";
        case QwenShardRule::PackedQkvColumnParallel: return "packed_qkv_column_parallel";
        case QwenShardRule::PackedConvChannelParallel: return "packed_conv_channel_parallel";
    }
    return "unknown";
}

QwenWeightMap::QwenWeightMap(const SafeTensorsIndex& index, const QwenConfig& config,
                             int tp_world, int tp_rank)
    : index_(index), config_(config), tp_world_(tp_world), tp_rank_(tp_rank) {
    require_tp(tp_world_, tp_rank_);
    embed_tokens_ = require_tensor(
        "model.language_model.embed_tokens.weight", SafeDType::BF16,
        {config_.vocab_size, config_.hidden_size}, QwenShardRule::ParallelEmbedding, 0);
    final_norm_ = require_tensor(
        "model.language_model.norm.weight", SafeDType::BF16,
        {config_.hidden_size}, QwenShardRule::Replicated, -1);
    lm_head_ = require_tensor(
        "lm_head.weight", SafeDType::BF16,
        {config_.vocab_size, config_.hidden_size}, QwenShardRule::ParallelHead, 0);

    layers_.reserve(config_.num_hidden_layers);
    for (uint64_t layer_id = 0; layer_id < config_.num_hidden_layers; ++layer_id) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_id) + ".";
        QwenLayerWeights layer;
        layer.input_layernorm = require_tensor(prefix + "input_layernorm.weight", SafeDType::BF16,
                                               {config_.hidden_size});
        layer.post_attention_layernorm = require_tensor(prefix + "post_attention_layernorm.weight", SafeDType::BF16,
                                                        {config_.hidden_size});
        if (config_.layer_types[layer_id] == QwenLayerType::LinearAttention) {
            const auto& linear = config_.linear_attention;
            const uint64_t key_dim = linear.key_heads * linear.key_head_dim;
            const uint64_t value_dim = linear.value_heads * linear.value_head_dim;
            layer.linear_attention.in_proj_qkv = require_linear(
                prefix + "linear_attn.in_proj_qkv.weight",
                {2 * key_dim + value_dim, config_.hidden_size},
                QwenShardRule::PackedQkvColumnParallel, 0);
            layer.linear_attention.in_proj_z = require_linear(
                prefix + "linear_attn.in_proj_z.weight", {value_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.out_proj = require_linear(
                prefix + "linear_attn.out_proj.weight", {config_.hidden_size, value_dim},
                QwenShardRule::RowParallel, 1);
            layer.linear_attention.in_proj_a = require_linear(
                prefix + "linear_attn.in_proj_a.weight", {linear.value_heads, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0, SafeDType::BF16);
            layer.linear_attention.in_proj_b = require_linear(
                prefix + "linear_attn.in_proj_b.weight", {linear.value_heads, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0, SafeDType::BF16);
            layer.linear_attention.conv1d = require_tensor(
                prefix + "linear_attn.conv1d.weight", SafeDType::BF16,
                {2 * key_dim + value_dim, 1, linear.conv_kernel_dim},
                QwenShardRule::PackedConvChannelParallel, 0);
            layer.linear_attention.a_log = require_tensor(
                prefix + "linear_attn.A_log", SafeDType::BF16, {linear.value_heads},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.dt_bias = require_tensor(
                prefix + "linear_attn.dt_bias", SafeDType::BF16, {linear.value_heads},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.norm = require_tensor(
                prefix + "linear_attn.norm.weight", SafeDType::BF16, {linear.value_head_dim});
        } else {
            const auto& full = config_.full_attention;
            const uint64_t attention_dim = full.num_heads * full.head_dim;
            const uint64_t q_dim = attention_dim * (full.output_gate ? 2 : 1);
            const uint64_t kv_dim = full.num_key_value_heads * full.head_dim;
            layer.full_attention.q_proj = require_linear(
                prefix + "self_attn.q_proj.weight", {q_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.k_proj = require_linear(
                prefix + "self_attn.k_proj.weight", {kv_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.v_proj = require_linear(
                prefix + "self_attn.v_proj.weight", {kv_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.o_proj = require_linear(
                prefix + "self_attn.o_proj.weight", {config_.hidden_size, attention_dim},
                QwenShardRule::RowParallel, 1);
            layer.full_attention.q_norm = require_tensor(
                prefix + "self_attn.q_norm.weight", SafeDType::BF16, {full.head_dim});
            layer.full_attention.k_norm = require_tensor(
                prefix + "self_attn.k_norm.weight", SafeDType::BF16, {full.head_dim});
        }

        const std::string mlp_prefix = prefix + "mlp.";
        layer.mlp.gate_proj = require_linear(
            mlp_prefix + "gate_proj.weight", {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        layer.mlp.up_proj = require_linear(
            mlp_prefix + "up_proj.weight", {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        layer.mlp.down_proj = require_linear(
            mlp_prefix + "down_proj.weight", {config_.hidden_size, config_.mlp.intermediate_size},
            QwenShardRule::RowParallel, 1);
        layers_.push_back(std::move(layer));
    }

    if (config_.mtp_num_hidden_layers != 0) {
        if (config_.mtp_num_hidden_layers != 1) {
            throw std::runtime_error(
                "Qwen native MTP currently supports exactly one MTP layer");
        }
        if (config_.mtp_use_dedicated_embeddings) {
            throw std::runtime_error(
                "Qwen native MTP dedicated embeddings are not supported");
        }
        mtp_.found = true;
        mtp_.pre_fc_norm_embedding = require_tensor(
            "mtp.pre_fc_norm_embedding.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.pre_fc_norm_hidden = require_tensor(
            "mtp.pre_fc_norm_hidden.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.norm = require_tensor(
            "mtp.norm.weight", SafeDType::BF16, {config_.hidden_size});
        mtp_.fc = require_linear(
            "mtp.fc.weight", {config_.hidden_size, 2 * config_.hidden_size},
            QwenShardRule::Replicated, -1, SafeDType::BF16);

        const std::string prefix = "mtp.layers.0.";
        mtp_.layer.input_layernorm = require_tensor(
            prefix + "input_layernorm.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.layer.post_attention_layernorm = require_tensor(
            prefix + "post_attention_layernorm.weight", SafeDType::BF16,
            {config_.hidden_size});
        const auto& full = config_.full_attention;
        const uint64_t attention_dim = full.num_heads * full.head_dim;
        const uint64_t q_dim = attention_dim * 2;
        const uint64_t kv_dim = full.num_key_value_heads * full.head_dim;
        mtp_.layer.full_attention.q_proj = require_linear(
            prefix + "self_attn.q_proj.weight", {q_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.k_proj = require_linear(
            prefix + "self_attn.k_proj.weight", {kv_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.v_proj = require_linear(
            prefix + "self_attn.v_proj.weight", {kv_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.o_proj = require_linear(
            prefix + "self_attn.o_proj.weight", {config_.hidden_size, attention_dim},
            QwenShardRule::RowParallel, 1);
        mtp_.layer.full_attention.q_norm = require_tensor(
            prefix + "self_attn.q_norm.weight", SafeDType::BF16,
            {full.head_dim});
        mtp_.layer.full_attention.k_norm = require_tensor(
            prefix + "self_attn.k_norm.weight", SafeDType::BF16,
            {full.head_dim});
        const std::string mlp_prefix = prefix + "mlp.";
        mtp_.layer.mlp.gate_proj = require_linear(
            mlp_prefix + "gate_proj.weight",
            {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.mlp.up_proj = require_linear(
            mlp_prefix + "up_proj.weight",
            {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.mlp.down_proj = require_linear(
            mlp_prefix + "down_proj.weight",
            {config_.hidden_size, config_.mlp.intermediate_size},
            QwenShardRule::RowParallel, 1);
    }

    auto count_ref = [this](const QwenTensorRef& ref, bool scale) { record(ref, scale); };
    const auto count_linear = [&count_ref](const QwenLinearRef& linear) {
        if (linear.weight.found) count_ref(linear.weight, false);
        if (linear.has_scale) count_ref(linear.scale, true);
    };
    count_ref(embed_tokens_, false);
    count_ref(final_norm_, false);
    count_ref(lm_head_, false);
    for (const auto& layer : layers_) {
        count_ref(layer.input_layernorm, false);
        count_ref(layer.post_attention_layernorm, false);
        if (layer.linear_attention.in_proj_qkv.weight.found) {
            count_linear(layer.linear_attention.in_proj_qkv);
            count_linear(layer.linear_attention.in_proj_z);
            count_linear(layer.linear_attention.out_proj);
            count_linear(layer.linear_attention.in_proj_a);
            count_linear(layer.linear_attention.in_proj_b);
            count_ref(layer.linear_attention.conv1d, false);
            count_ref(layer.linear_attention.a_log, false);
            count_ref(layer.linear_attention.dt_bias, false);
            count_ref(layer.linear_attention.norm, false);
        } else {
            count_linear(layer.full_attention.q_proj);
            count_linear(layer.full_attention.k_proj);
            count_linear(layer.full_attention.v_proj);
            count_linear(layer.full_attention.o_proj);
            count_ref(layer.full_attention.q_norm, false);
            count_ref(layer.full_attention.k_norm, false);
        }
        count_linear(layer.mlp.gate_proj);
        count_linear(layer.mlp.up_proj);
        count_linear(layer.mlp.down_proj);
    }
    if (mtp_.found) {
        count_ref(mtp_.pre_fc_norm_embedding, false);
        count_ref(mtp_.pre_fc_norm_hidden, false);
        count_linear(mtp_.fc);
        count_ref(mtp_.norm, false);
        const QwenLayerWeights& layer = mtp_.layer;
        count_ref(layer.input_layernorm, false);
        count_ref(layer.post_attention_layernorm, false);
        count_linear(layer.full_attention.q_proj);
        count_linear(layer.full_attention.k_proj);
        count_linear(layer.full_attention.v_proj);
        count_linear(layer.full_attention.o_proj);
        count_ref(layer.full_attention.q_norm, false);
        count_ref(layer.full_attention.k_norm, false);
        count_linear(layer.mlp.gate_proj);
        count_linear(layer.mlp.up_proj);
        count_linear(layer.mlp.down_proj);
    }
}
QwenTensorRef QwenWeightMap::require_tensor(const std::string& name, SafeDType dtype,
                                            const std::vector<uint64_t>& shape,
                                            QwenShardRule rule, int shard_dim) const {
    const std::string* shard_name = index_.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("Qwen tensor not in checkpoint index: " + name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("Qwen tensor missing from shard header: " + name);
    if (info->dtype != dtype) {
        throw std::runtime_error("unexpected dtype for " + name + " expected=" + safe_dtype_name(dtype) +
                                 " actual=" + safe_dtype_name(info->dtype));
    }
    if (info->shape != shape) {
        throw std::runtime_error("unexpected shape for " + name + " expected=" + shape_string(shape) +
                                 " actual=" + shape_string(info->shape));
    }

    std::vector<uint64_t> local_shape = shape;
    uint64_t start = 0;
    uint64_t size = 0;
    if (rule == QwenShardRule::Replicated) {
        size = shard_dim < 0 ? 0 : shape[static_cast<size_t>(shard_dim)];
    } else if (rule == QwenShardRule::PackedQkvColumnParallel ||
               rule == QwenShardRule::PackedConvChannelParallel) {
        const uint64_t key_dim = config_.linear_attention.key_heads * config_.linear_attention.key_head_dim;
        const uint64_t value_dim = config_.linear_attention.value_heads * config_.linear_attention.value_head_dim;
        const uint64_t segments[] = {key_dim, key_dim, value_dim};
        uint64_t source_offset = 0;
        uint64_t local_total = 0;
        QwenTensorRef ref;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start, &segment_size);
            ref.segments.emplace_back(source_offset + segment_start, segment_size);
            source_offset += segment;
            local_total += segment_size;
        }
        local_shape[0] = local_total;
        size = local_total;
        ref = make_ref(name, *shard_name, *info, rule, shard_dim, 0, size, local_shape);
        ref.segments.clear();
        source_offset = 0;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start, &segment_size);
            ref.segments.emplace_back(source_offset + segment_start, segment_size);
            source_offset += segment;
        }
        return ref;
    } else if (shard_dim >= 0) {
        if (static_cast<size_t>(shard_dim) >= shape.size()) throw std::runtime_error("invalid Qwen shard dimension");
        shard_range(shape[static_cast<size_t>(shard_dim)], tp_world_, tp_rank_, &start, &size);
        local_shape[static_cast<size_t>(shard_dim)] = size;
    } else {
        throw std::runtime_error("Qwen non-replicated tensor has no shard dimension: " + name);
    }
    return make_ref(name, *shard_name, *info, rule, shard_dim, start, size, local_shape);
}

QwenLinearRef QwenWeightMap::require_linear(const std::string& name,
                                            const std::vector<uint64_t>& shape,
                                            QwenShardRule rule, int shard_dim,
                                            SafeDType weight_dtype) const {
    QwenLinearRef result;
    result.weight = require_tensor(name, weight_dtype, shape, rule, shard_dim);
    if (weight_dtype != SafeDType::F8_E4M3) return result;

    const std::string suffix = ".weight";
    const std::string scale_name =
        name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0
            ? name.substr(0, name.size() - suffix.size()) + ".weight_scale_inv"
            : name + ".weight_scale_inv";
    const std::string* shard_name = index_.shard_for_tensor(scale_name);
    if (shard_name == nullptr) throw std::runtime_error("Qwen FP8 scale not in checkpoint index: " + scale_name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* scale_info = shard.find_tensor(scale_name);
    if (scale_info == nullptr) throw std::runtime_error("Qwen FP8 scale missing from shard header: " + scale_name);
    if (scale_info->dtype != SafeDType::BF16) throw std::runtime_error("Qwen FP8 scale must be BF16: " + scale_name);
    const std::vector<uint64_t> full_scale_shape = {ceil_div(shape[0], config_.fp8_block_size),
                                                    ceil_div(shape[1], config_.fp8_block_size)};
    if (scale_info->shape != full_scale_shape) {
        throw std::runtime_error("unexpected Qwen FP8 scale shape for " + scale_name +
                                 " expected=" + shape_string(full_scale_shape) +
                                 " actual=" + shape_string(scale_info->shape));
    }

    std::vector<uint64_t> local_scale_shape = full_scale_shape;
    uint64_t scale_start = 0;
    uint64_t scale_size = 0;
    QwenTensorRef scale_ref;
    if (rule == QwenShardRule::Replicated) {
        // No dimension is sharded, so every rank consumes the complete scale.
        scale_ref = make_ref(scale_name, *shard_name, *scale_info, rule, shard_dim,
                             0, 0, local_scale_shape);
    } else if (rule == QwenShardRule::RowParallel) {
        shard_range(shape[1], tp_world_, tp_rank_, &scale_start, &scale_size);
        if (scale_start % config_.fp8_block_size != 0 || scale_size % config_.fp8_block_size != 0) {
            throw std::runtime_error("Qwen row-parallel FP8 shard is not scale-block aligned: " + name);
        }
        local_scale_shape[1] = scale_size / config_.fp8_block_size;
        scale_ref = make_ref(scale_name, *shard_name, *scale_info, rule, shard_dim,
                             scale_start / config_.fp8_block_size,
                             scale_size / config_.fp8_block_size, local_scale_shape);
    } else if (rule == QwenShardRule::PackedQkvColumnParallel) {
        const uint64_t key_dim = config_.linear_attention.key_heads * config_.linear_attention.key_head_dim;
        const uint64_t value_dim = config_.linear_attention.value_heads * config_.linear_attention.value_head_dim;
        const uint64_t segments[] = {key_dim, key_dim, value_dim};
        uint64_t source_offset = 0;
        uint64_t local_rows = 0;
        std::vector<std::pair<uint64_t, uint64_t>> scale_segments;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start, &segment_size);
            if (segment_start % config_.fp8_block_size != 0 || segment_size % config_.fp8_block_size != 0) {
                throw std::runtime_error("Qwen packed QKV shard is not scale-block aligned: " + name);
            }
            scale_segments.emplace_back(source_offset / config_.fp8_block_size +
                                            segment_start / config_.fp8_block_size,
                                        segment_size / config_.fp8_block_size);
            source_offset += segment;
            local_rows += segment_size / config_.fp8_block_size;
        }
        local_scale_shape[0] = local_rows;
        scale_ref = make_ref(scale_name, *shard_name, *scale_info, rule, shard_dim,
                             0, local_rows, local_scale_shape);
        scale_ref.segments = std::move(scale_segments);
    } else {
        shard_range(shape[0], tp_world_, tp_rank_, &scale_start, &scale_size);
        if (scale_start % config_.fp8_block_size != 0 || scale_size % config_.fp8_block_size != 0) {
            throw std::runtime_error("Qwen column-parallel FP8 shard is not scale-block aligned: " + name);
        }
        local_scale_shape[0] = scale_size / config_.fp8_block_size;
        scale_ref = make_ref(scale_name, *shard_name, *scale_info, rule, shard_dim,
                             scale_start / config_.fp8_block_size,
                             scale_size / config_.fp8_block_size, local_scale_shape);
    }
    result.scale = std::move(scale_ref);
    result.has_scale = true;
    return result;
}

void QwenWeightMap::record(const QwenTensorRef& ref, bool scale) {
    ++tensor_count_;
    if (scale) local_scale_bytes_ += ref.nbytes;
    else local_weight_bytes_ += ref.nbytes;
}

}  // namespace dsv4
