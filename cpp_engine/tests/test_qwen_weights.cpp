// Qwen Safetensors mapping test with a small synthetic TP4 checkpoint.
//
// The fixture preserves the real 128-row FP8 scale block contract while using
// one linear-attention layer, so it can verify packed QKV/conv segment offsets
// without requiring a model download.

#include "cuda_ops.hpp"
#include "qwen_config.hpp"
#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
};

uint64_t numel(const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (uint64_t dim : shape) total *= dim;
    return total;
}

uint64_t dtype_size(const std::string& dtype) {
    if (dtype == "F8_E4M3") return 1;
    if (dtype == "BF16") return 2;
    throw std::runtime_error("unsupported fixture dtype: " + dtype);
}

std::string shape_json(const std::vector<uint64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::string fixture_dir() {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    const std::string root = base != nullptr ? std::string(base) + "/tmp" : std::string("/tmp");
    return root + "/qwen_weights_fixture";
}

std::string config_json() {
    return R"JSON({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 512,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attn_output_gate": true,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    "layer_types": ["linear_attention"]
  }
})JSON";
}

std::vector<TensorSpec> tensor_specs() {
    const std::vector<uint64_t> hidden = {128};
    const std::vector<uint64_t> vocab_weight = {512, 128};
    const std::vector<uint64_t> qkv = {1536, 128};
    const std::vector<uint64_t> qkv_scale = {12, 1};
    const std::vector<uint64_t> value_proj = {512, 128};
    const std::vector<uint64_t> value_proj_scale = {4, 1};
    const std::vector<uint64_t> out_proj = {128, 512};
    const std::vector<uint64_t> out_proj_scale = {1, 4};
    const std::vector<uint64_t> vector4 = {4};
    const std::vector<uint64_t> value_heads_weight = {4, 128};
    const std::vector<uint64_t> mlp = {512, 128};
    const std::vector<uint64_t> mlp_scale = {4, 1};
    const std::vector<uint64_t> down = {128, 512};
    const std::vector<uint64_t> down_scale = {1, 4};

    std::vector<TensorSpec> out;
    auto add = [&out](const std::string& name, const std::string& dtype,
                      const std::vector<uint64_t>& shape) {
        out.push_back({name, dtype, shape});
    };
    add("model.language_model.embed_tokens.weight", "BF16", vocab_weight);
    add("model.language_model.norm.weight", "BF16", hidden);
    add("lm_head.weight", "BF16", vocab_weight);

    const std::string prefix = "model.language_model.layers.0.";
    add(prefix + "input_layernorm.weight", "BF16", hidden);
    add(prefix + "post_attention_layernorm.weight", "BF16", hidden);
    add(prefix + "linear_attn.in_proj_qkv.weight", "F8_E4M3", qkv);
    add(prefix + "linear_attn.in_proj_qkv.weight_scale_inv", "BF16", qkv_scale);
    add(prefix + "linear_attn.in_proj_z.weight", "F8_E4M3", value_proj);
    add(prefix + "linear_attn.in_proj_z.weight_scale_inv", "BF16", value_proj_scale);
    add(prefix + "linear_attn.out_proj.weight", "F8_E4M3", out_proj);
    add(prefix + "linear_attn.out_proj.weight_scale_inv", "BF16", out_proj_scale);
    add(prefix + "linear_attn.in_proj_a.weight", "BF16", value_heads_weight);
    add(prefix + "linear_attn.in_proj_b.weight", "BF16", value_heads_weight);
    add(prefix + "linear_attn.conv1d.weight", "BF16", {1536, 1, 4});
    add(prefix + "linear_attn.A_log", "BF16", vector4);
    add(prefix + "linear_attn.dt_bias", "BF16", vector4);
    add(prefix + "linear_attn.norm.weight", "BF16", {128});
    add(prefix + "mlp.gate_proj.weight", "F8_E4M3", mlp);
    add(prefix + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale);
    add(prefix + "mlp.up_proj.weight", "F8_E4M3", mlp);
    add(prefix + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale);
    add(prefix + "mlp.down_proj.weight", "F8_E4M3", down);
    add(prefix + "mlp.down_proj.weight_scale_inv", "BF16", down_scale);
    return out;
}

bool write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(out);
}

bool write_fixture(const std::string& dir) {
    const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) return false;
    if (!write_file(dir + "/config.json", config_json())) return false;

    const std::vector<TensorSpec> specs = tensor_specs();
    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (i) header << ',';
        const TensorSpec& spec = specs[i];
        const uint64_t bytes = numel(spec.shape) * dtype_size(spec.dtype);
        header << '"' << spec.name << "\":{\"dtype\":\"" << spec.dtype
               << "\",\"shape\":" << shape_json(spec.shape)
               << ",\"data_offsets\":[" << offset << ',' << (offset + bytes) << "]}";
        offset += bytes;
    }
    header << '}';

    std::ofstream shard(dir + "/model.safetensors", std::ios::binary | std::ios::trunc);
    if (!shard) return false;
    const uint64_t header_len = header.str().size();
    shard.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    shard.write(header.str().data(), static_cast<std::streamsize>(header.str().size()));
    std::vector<uint8_t> zeros(static_cast<size_t>(offset), 0);
    shard.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size()));
    if (!shard) return false;

    std::ostringstream index;
    index << "{\"metadata\":{\"total_size\":" << offset
          << "},\"weight_map\":{";
    for (size_t i = 0; i < specs.size(); ++i) {
        if (i) index << ',';
        index << '"' << specs[i].name << "\":\"model.safetensors\"";
    }
    index << "}}";
    return write_file(dir + "/model.safetensors.index.json", index.str());
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_segment(const dsv4::QwenTensorRef& ref, size_t index,
                     uint64_t start, uint64_t size, const std::string& label) {
    require(index < ref.segments.size(), label + " missing segment");
    require(ref.segments[index].first == start && ref.segments[index].second == size,
            label + " wrong segment");
}

void check_rank(const dsv4::SafeTensorsIndex& index, const dsv4::QwenConfig& config, int rank) {
    dsv4::QwenWeightMap map(index, config, 4, rank);
    require(map.layers().size() == 1, "expected one mapped layer");
    const auto& linear = map.layers()[0].linear_attention;

    require(linear.in_proj_a.weight.dtype == dsv4::SafeDType::BF16,
            "in_proj_a must remain BF16 in storage");
    require(linear.in_proj_a.weight.local_shape == std::vector<uint64_t>({1, 128}),
            "in_proj_a local shape");
    require(linear.in_proj_a.weight.device_dtype == dsv4::SafeDType::F16,
            "in_proj_a must upload as FP16 on Turing");
    require(linear.in_proj_b.weight.dtype == dsv4::SafeDType::BF16,
            "in_proj_b must remain BF16 in storage");
    require(linear.in_proj_b.weight.device_dtype == dsv4::SafeDType::F16,
            "in_proj_b must upload as FP16 on Turing");
    require(linear.in_proj_qkv.weight.dtype == dsv4::SafeDType::F8_E4M3 &&
                linear.in_proj_qkv.weight.device_dtype == dsv4::SafeDType::F8_E4M3,
            "QKV FP8 must remain compressed on device");
    require(linear.in_proj_qkv.scale.dtype == dsv4::SafeDType::BF16 &&
                linear.in_proj_qkv.scale.device_dtype == dsv4::SafeDType::F16,
            "FP8 scale must upload as FP16 on Turing");
    require(!linear.in_proj_a.has_scale && !linear.in_proj_b.has_scale,
            "BF16 in_proj_a/b must not have FP8 scales");

    require(linear.in_proj_qkv.weight.local_shape == std::vector<uint64_t>({384, 128}),
            "QKV local shape");
    require(linear.in_proj_qkv.weight.segments.size() == 3, "QKV weight segment count");
    require(linear.in_proj_qkv.scale.local_shape == std::vector<uint64_t>({3, 1}),
            "QKV scale local shape");
    require(linear.in_proj_qkv.scale.segments.size() == 3, "QKV scale segment count");

    const uint64_t row_start = static_cast<uint64_t>(rank) * 128;
    require_segment(linear.in_proj_qkv.weight, 0, row_start, 128, "QKV weight Q");
    require_segment(linear.in_proj_qkv.weight, 1, 512 + row_start, 128, "QKV weight K");
    require_segment(linear.in_proj_qkv.weight, 2, 1024 + row_start, 128, "QKV weight V");
    const uint64_t block_start = static_cast<uint64_t>(rank);
    require_segment(linear.in_proj_qkv.scale, 0, block_start, 1, "QKV scale Q");
    require_segment(linear.in_proj_qkv.scale, 1, 4 + block_start, 1, "QKV scale K");
    require_segment(linear.in_proj_qkv.scale, 2, 8 + block_start, 1, "QKV scale V");

    require(linear.conv1d.local_shape == std::vector<uint64_t>({384, 1, 4}),
            "conv local shape");
    require(linear.conv1d.segments.size() == 3, "conv segment count");
    require_segment(linear.conv1d, 0, row_start, 128, "conv Q");
    require_segment(linear.conv1d, 1, 512 + row_start, 128, "conv K");
    require_segment(linear.conv1d, 2, 1024 + row_start, 128, "conv V");

    require(linear.in_proj_z.weight.local_shape == std::vector<uint64_t>({128, 128}),
            "in_proj_z local shape");
    require(linear.in_proj_z.scale.local_shape == std::vector<uint64_t>({1, 1}),
            "in_proj_z scale local shape");
    require(linear.in_proj_z.weight.shard_start == row_start &&
                linear.in_proj_z.scale.shard_start == static_cast<uint64_t>(rank),
            "in_proj_z shard offsets");

    require(linear.out_proj.weight.local_shape == std::vector<uint64_t>({128, 128}),
            "out_proj local shape");
    require(linear.out_proj.scale.local_shape == std::vector<uint64_t>({1, 1}),
            "out_proj scale local shape");
    require(linear.out_proj.weight.shard_start == row_start &&
                linear.out_proj.scale.shard_start == static_cast<uint64_t>(rank),
            "out_proj shard offsets");

    require(map.embed_tokens().local_shape == std::vector<uint64_t>({128, 128}),
            "embedding local shape");
    require(map.embed_tokens().device_dtype == dsv4::SafeDType::F16,
            "embedding must upload as FP16");
    require(map.lm_head().local_shape == std::vector<uint64_t>({128, 128}),
            "head local shape");
    require(map.lm_head().device_dtype == dsv4::SafeDType::F16,
            "head must upload as FP16");

    const dsv4::QwenHostTensor qkv_host =
        dsv4::qwen_materialize_host_tensor(index, linear.in_proj_qkv.weight);
    require(qkv_host.storage_dtype == dsv4::SafeDType::F8_E4M3 &&
                qkv_host.device_dtype == dsv4::SafeDType::F8_E4M3,
            "QKV host materializer must preserve FP8");
    require(qkv_host.bytes.size() == 384u * 128u, "QKV host bytes");
    const dsv4::QwenHostTensor scale_host =
        dsv4::qwen_materialize_host_tensor(index, linear.in_proj_qkv.scale);
    require(scale_host.storage_dtype == dsv4::SafeDType::BF16 &&
                scale_host.device_dtype == dsv4::SafeDType::F16,
            "scale host materializer must convert to FP16");
    require(scale_host.bytes.size() == 3u * sizeof(uint16_t), "scale host bytes");
    const dsv4::QwenHostTensor a_host =
        dsv4::qwen_materialize_host_tensor(index, linear.in_proj_a.weight);
    require(a_host.device_dtype == dsv4::SafeDType::F16 &&
                a_host.bytes.size() == 128u * sizeof(uint16_t),
            "in_proj_a host materializer must convert local BF16 shard");

    require(dsv4::qwen_bf16_to_fp16_bits(0x3f80u) == 0x3c00u,
            "BF16 1.0 converts to FP16 1.0");
    require(dsv4::qwen_bf16_to_fp16_bits(0xbf80u) == 0xbc00u,
            "BF16 -1.0 converts to FP16 -1.0");
    if (dsv4::cuda_runtime_available()) {
        dsv4::QwenDeviceTensor device_scale =
            dsv4::qwen_upload_tensor_cuda(index, linear.in_proj_qkv.scale);
        require(device_scale.device_dtype == dsv4::SafeDType::F16,
                "uploaded scale dtype");
        std::vector<uint8_t> device_bytes(scale_host.bytes.size(), 0);
        require(cudaDeviceSynchronize() == cudaSuccess, "scale upload sync");
        require(cudaMemcpy(device_bytes.data(), device_scale.data, device_bytes.size(),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "scale device readback");
        require(device_bytes == scale_host.bytes,
                "scale device bytes are FP16 materialized bytes");
    }
    require(map.local_weight_bytes() > 0 && map.local_scale_bytes() > 0,
            "weight byte accounting");
}

}  // namespace

int main() {
    try {
        const std::string dir = fixture_dir();
        if (!write_fixture(dir)) {
            std::cout << "[SKIP] could not create Qwen weight fixture\n";
            return 0;
        }
        const dsv4::QwenConfig config = dsv4::QwenConfig::from_hf_config(dir);
        dsv4::SafeTensorsIndex index(dir);
        for (int rank = 0; rank < 4; ++rank) check_rank(index, config, rank);
        std::cout << "[PASS] test_qwen_weights tp=4 layers=" << config.num_hidden_layers << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_weights " << ex.what() << "\n";
        return 1;
    }
}
