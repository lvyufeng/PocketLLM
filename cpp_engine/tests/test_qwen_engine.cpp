// Minimal Qwen engine lifecycle test with a one-layer linear-attention fixture.
// All weights are zero, so the greedy result is deterministic while the test
// still exercises FP8 online projection, FP16 materialization, conv tail,
// recurrent state, residual/MLP wiring, and the decode cache continuation.

#include "cuda_ops.hpp"
#include "qwen_config.hpp"
#include "qwen_engine.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
};

uint64_t numel(const std::vector<uint64_t>& shape) {
    uint64_t out = 1;
    for (uint64_t dim : shape) out *= dim;
    return out;
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
    return root + "/qwen_engine_fixture";
}

std::string config_json() {
    return R"JSON({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 128,
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
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    "layer_types": ["linear_attention"]
  }
})JSON";
}

std::vector<TensorSpec> specs() {
    const std::vector<uint64_t> hidden = {128};
    const std::vector<uint64_t> vocab = {64, 128};
    const std::vector<uint64_t> qkv = {1536, 128};
    const std::vector<uint64_t> qkv_scale = {12, 1};
    const std::vector<uint64_t> value_proj = {512, 128};
    const std::vector<uint64_t> value_scale = {4, 1};
    const std::vector<uint64_t> out_proj = {128, 512};
    const std::vector<uint64_t> out_scale = {1, 4};
    const std::vector<uint64_t> heads = {4, 128};
    const std::vector<uint64_t> vector4 = {4};
    const std::vector<uint64_t> mlp = {128, 128};
    const std::vector<uint64_t> mlp_scale = {1, 1};
    const std::string p = "model.language_model.layers.0.";
    return {
        {"model.language_model.embed_tokens.weight", "BF16", vocab},
        {"model.language_model.norm.weight", "BF16", hidden},
        {"lm_head.weight", "BF16", vocab},
        {p + "input_layernorm.weight", "BF16", hidden},
        {p + "post_attention_layernorm.weight", "BF16", hidden},
        {p + "linear_attn.in_proj_qkv.weight", "F8_E4M3", qkv},
        {p + "linear_attn.in_proj_qkv.weight_scale_inv", "BF16", qkv_scale},
        {p + "linear_attn.in_proj_z.weight", "F8_E4M3", value_proj},
        {p + "linear_attn.in_proj_z.weight_scale_inv", "BF16", value_scale},
        {p + "linear_attn.out_proj.weight", "F8_E4M3", out_proj},
        {p + "linear_attn.out_proj.weight_scale_inv", "BF16", out_scale},
        {p + "linear_attn.in_proj_a.weight", "BF16", heads},
        {p + "linear_attn.in_proj_b.weight", "BF16", heads},
        {p + "linear_attn.conv1d.weight", "BF16", {1536, 1, 4}},
        {p + "linear_attn.A_log", "BF16", vector4},
        {p + "linear_attn.dt_bias", "BF16", vector4},
        {p + "linear_attn.norm.weight", "BF16", {128}},
        {p + "mlp.gate_proj.weight", "F8_E4M3", mlp},
        {p + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale},
        {p + "mlp.up_proj.weight", "F8_E4M3", mlp},
        {p + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale},
        {p + "mlp.down_proj.weight", "F8_E4M3", mlp},
        {p + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale},
    };
}

bool write_fixture(const std::string& dir) {
    const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) return false;
    std::ofstream config(dir + "/config.json", std::ios::binary | std::ios::trunc);
    if (!config) return false;
    config << config_json();
    if (!config) return false;

    const auto tensors = specs();
    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) header << ',';
        const auto& tensor = tensors[i];
        const uint64_t bytes = numel(tensor.shape) * dtype_size(tensor.dtype);
        header << '"' << tensor.name << "\":{\"dtype\":\"" << tensor.dtype
               << "\",\"shape\":" << shape_json(tensor.shape)
               << ",\"data_offsets\":[" << offset << ',' << offset + bytes << "]}";
        offset += bytes;
    }
    header << '}';
    const std::string header_text = header.str();
    std::ofstream shard(dir + "/model.safetensors", std::ios::binary | std::ios::trunc);
    if (!shard) return false;
    const uint64_t header_len = header_text.size();
    shard.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    shard.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    std::vector<uint8_t> zeros(static_cast<size_t>(offset), 0);
    shard.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size()));
    if (!shard) return false;

    std::ofstream index(dir + "/model.safetensors.index.json", std::ios::binary | std::ios::trunc);
    if (!index) return false;
    index << "{\"metadata\":{\"total_size\":" << offset << "},\"weight_map\":{";
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) index << ',';
        index << '"' << tensors[i].name << "\":\"model.safetensors\"";
    }
    index << "}}";
    return static_cast<bool>(index);
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    if (!dsv4::cuda_runtime_available()) {
        std::cout << "[SKIP] test_qwen_engine requires a CUDA device\n";
        return 0;
    }
    try {
        const std::string dir = fixture_dir();
        require(write_fixture(dir), "could not create Qwen engine fixture");
        dsv4::QwenEngineOptions options;
        options.tp_world = 1;
        options.tp_rank = 0;
        options.device = 0;
        dsv4::QwenEngine engine(dir, options, 1, 8);
        require(engine.resident_weight_bytes() > 0, "resident Qwen weights missing");
        require(engine.resident_scale_bytes() > 0, "resident Qwen scales missing");

        const dsv4::QwenForwardResult prefill = engine.prefill({1, 2});
        require(prefill.layers == 1 && prefill.dim == 128, "prefill metadata");
        require(prefill.position == 2, "prefill position");
        require(prefill.top_token == 0, "zero fixture prefill greedy token");

        const dsv4::QwenForwardResult decoded = engine.decode_step(3);
        require(decoded.position == 3, "decode position");
        require(decoded.top_token == 0, "zero fixture decode greedy token");

        engine.reset();
        require(engine.position() == 0, "reset position");
        const dsv4::QwenForwardResult second = engine.prefill({4});
        require(second.position == 1 && second.top_token == 0, "reset and second prefill");
        std::cout << "[PASS] test_qwen_engine layers=" << second.layers
                  << " resident_weight_bytes=" << engine.resident_weight_bytes()
                  << " resident_scale_bytes=" << engine.resident_scale_bytes() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_engine " << ex.what() << "\n";
        return 1;
    }
}
