// Minimal Qwen engine lifecycle test with linear-attention and full-GQA layers.
// All weights are zero, so the greedy result is deterministic while the test
// still exercises FP8 online projection, FP16 materialization, conv tail,
// recurrent state, residual/MLP wiring, and FP16/FP8 cache continuation.

#include "cuda_ops.hpp"
#include "qwen_config.hpp"
#include "qwen_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
    "num_hidden_layers": 2,
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
    "layer_types": ["linear_attention", "full_attention"],
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false
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
    const std::vector<uint64_t> q_proj = {1024, 128};
    const std::vector<uint64_t> q_proj_scale = {8, 1};
    const std::string linear = "model.language_model.layers.0.";
    const std::string full = "model.language_model.layers.1.";
    std::vector<TensorSpec> out = {
        {"model.language_model.embed_tokens.weight", "BF16", vocab},
        {"model.language_model.norm.weight", "BF16", hidden},
        {"lm_head.weight", "BF16", vocab},
        {linear + "input_layernorm.weight", "BF16", hidden},
        {linear + "post_attention_layernorm.weight", "BF16", hidden},
        {linear + "linear_attn.in_proj_qkv.weight", "F8_E4M3", qkv},
        {linear + "linear_attn.in_proj_qkv.weight_scale_inv", "BF16", qkv_scale},
        {linear + "linear_attn.in_proj_z.weight", "F8_E4M3", value_proj},
        {linear + "linear_attn.in_proj_z.weight_scale_inv", "BF16", value_scale},
        {linear + "linear_attn.out_proj.weight", "F8_E4M3", out_proj},
        {linear + "linear_attn.out_proj.weight_scale_inv", "BF16", out_scale},
        {linear + "linear_attn.in_proj_a.weight", "BF16", heads},
        {linear + "linear_attn.in_proj_b.weight", "BF16", heads},
        {linear + "linear_attn.conv1d.weight", "BF16", {1536, 1, 4}},
        {linear + "linear_attn.A_log", "BF16", vector4},
        {linear + "linear_attn.dt_bias", "BF16", vector4},
        {linear + "linear_attn.norm.weight", "BF16", hidden},
        {linear + "mlp.gate_proj.weight", "F8_E4M3", mlp},
        {linear + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale},
        {linear + "mlp.up_proj.weight", "F8_E4M3", mlp},
        {linear + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale},
        {linear + "mlp.down_proj.weight", "F8_E4M3", mlp},
        {linear + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale},
    };
    out.insert(out.end(), {
        {full + "input_layernorm.weight", "BF16", hidden},
        {full + "post_attention_layernorm.weight", "BF16", hidden},
        {full + "self_attn.q_proj.weight", "F8_E4M3", q_proj},
        {full + "self_attn.q_proj.weight_scale_inv", "BF16", q_proj_scale},
        {full + "self_attn.k_proj.weight", "F8_E4M3", value_proj},
        {full + "self_attn.k_proj.weight_scale_inv", "BF16", value_scale},
        {full + "self_attn.v_proj.weight", "F8_E4M3", value_proj},
        {full + "self_attn.v_proj.weight_scale_inv", "BF16", value_scale},
        {full + "self_attn.o_proj.weight", "F8_E4M3", out_proj},
        {full + "self_attn.o_proj.weight_scale_inv", "BF16", out_scale},
        {full + "self_attn.q_norm.weight", "BF16", hidden},
        {full + "self_attn.k_norm.weight", "BF16", hidden},
        {full + "mlp.gate_proj.weight", "F8_E4M3", mlp},
        {full + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale},
        {full + "mlp.up_proj.weight", "F8_E4M3", mlp},
        {full + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale},
        {full + "mlp.down_proj.weight", "F8_E4M3", mlp},
        {full + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale},
    });
    const std::string mtp = "mtp.";
    const std::string mtp_layer = "mtp.layers.0.";
    out.insert(out.end(), {
        {mtp + "pre_fc_norm_embedding.weight", "BF16", hidden},
        {mtp + "pre_fc_norm_hidden.weight", "BF16", hidden},
        {mtp + "norm.weight", "BF16", hidden},
        {mtp + "fc.weight", "BF16", {128, 256}},
        {mtp_layer + "input_layernorm.weight", "BF16", hidden},
        {mtp_layer + "post_attention_layernorm.weight", "BF16", hidden},
        {mtp_layer + "self_attn.q_proj.weight", "F8_E4M3", q_proj},
        {mtp_layer + "self_attn.q_proj.weight_scale_inv", "BF16", q_proj_scale},
        {mtp_layer + "self_attn.k_proj.weight", "F8_E4M3", value_proj},
        {mtp_layer + "self_attn.k_proj.weight_scale_inv", "BF16", value_scale},
        {mtp_layer + "self_attn.v_proj.weight", "F8_E4M3", value_proj},
        {mtp_layer + "self_attn.v_proj.weight_scale_inv", "BF16", value_scale},
        {mtp_layer + "self_attn.o_proj.weight", "F8_E4M3", out_proj},
        {mtp_layer + "self_attn.o_proj.weight_scale_inv", "BF16", out_scale},
        {mtp_layer + "self_attn.q_norm.weight", "BF16", hidden},
        {mtp_layer + "self_attn.k_norm.weight", "BF16", hidden},
        {mtp_layer + "mlp.gate_proj.weight", "F8_E4M3", mlp},
        {mtp_layer + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale},
        {mtp_layer + "mlp.up_proj.weight", "F8_E4M3", mlp},
        {mtp_layer + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale},
        {mtp_layer + "mlp.down_proj.weight", "F8_E4M3", mlp},
        {mtp_layer + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale},
    });
    return out;
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
    std::vector<uint8_t> data(static_cast<size_t>(offset), 0);
    uint64_t data_offset = 0;
    for (const TensorSpec& tensor : tensors) {
        const uint64_t bytes = numel(tensor.shape) * dtype_size(tensor.dtype);
        if (tensor.dtype == "F8_E4M3") {
            // E4M3 1.0, with a nonzero scale, exercises online FP8 paths.
            std::fill(data.begin() + static_cast<ptrdiff_t>(data_offset),
                      data.begin() + static_cast<ptrdiff_t>(data_offset + bytes),
                      static_cast<uint8_t>(0x38));
        } else {
            // BF16 1.0. Make embedding rows token-dependent so a branch at a
            // changed token actually changes the recurrent state.
            const bool embedding = tensor.name.find("embed_tokens.weight") !=
                std::string::npos;
            const uint64_t elements = bytes / 2;
            for (uint64_t element = 0; element < elements; ++element) {
                uint16_t bits = 0x3f80;
                if (embedding && !tensor.shape.empty()) {
                    const uint64_t row = element / tensor.shape.back();
                    bits = static_cast<uint16_t>(0x3f80u + (row & 0x1fu));
                }
                const uint64_t at = data_offset + element * 2;
                data[static_cast<size_t>(at)] = static_cast<uint8_t>(bits & 0xffu);
                data[static_cast<size_t>(at + 1)] = static_cast<uint8_t>(bits >> 8);
            }
        }
        data_offset += bytes;
    }
    shard.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
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

void require_same_generation(
    const std::vector<dsv4::QwenForwardResult>& actual,
    const std::vector<dsv4::QwenForwardResult>& expected,
    const std::string& label) {
    require(actual.size() == expected.size(), label + " output count");
    for (size_t index = 0; index < actual.size(); ++index) {
        require(actual[index].top_token == expected[index].top_token,
                label + " greedy token parity at " + std::to_string(index) +
                    ": got " + std::to_string(actual[index].top_token) +
                    ", expected " + std::to_string(expected[index].top_token));
    }
}

void require_same_mtp_decisions(const dsv4::QwenMtpStats& actual,
                                const dsv4::QwenMtpStats& expected,
                                const std::string& label) {
    require(actual.verify_count == expected.verify_count &&
                actual.proposed_drafts == expected.proposed_drafts &&
                actual.correct_drafts == expected.correct_drafts &&
                actual.rollback_count == expected.rollback_count &&
                actual.replay_tokens == expected.replay_tokens,
            label + " MTP decision parity");
}

std::vector<dsv4::QwenForwardResult> require_cached_mtp_matches_cold(
    const std::string& dir, dsv4::QwenEngine& cached,
    const dsv4::QwenEngineOptions& cached_options,
    const std::vector<int>& prompt, int max_new_tokens,
    const std::string& resume_source, int reused_tokens, int computed_tokens,
    const std::string& label) {
    const auto actual = cached.generate(prompt, max_new_tokens);
    const dsv4::QwenMtpStats actual_stats = cached.mtp_stats();
    const dsv4::QwenPrefixCacheStats prefix = cached.prefix_cache_stats();
    require(prefix.resume_source == resume_source &&
                prefix.reused_tokens == reused_tokens &&
                prefix.computed_tokens == computed_tokens,
            label + " prefix accounting");

    dsv4::QwenEngineOptions cold_options = cached_options;
    cold_options.prefix_cache = false;
    dsv4::QwenEngine cold(dir, cold_options, 2, 32);
    const auto expected = cold.generate(prompt, max_new_tokens);
    require_same_generation(actual, expected, label);
    require_same_mtp_decisions(actual_stats, cold.mtp_stats(), label);
    require(cached.position() == cold.position(), label + " committed position");
    return actual;
}

void exercise_prefix_cache(const std::string& dir,
                           dsv4::QwenKvCacheDType cache_dtype) {
    dsv4::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 2;
    options.kv_cache_dtype = cache_dtype;
    options.state_snapshot_interval_tokens = 2;
    options.max_state_snapshots = 8;
    dsv4::QwenEngine engine(dir, options, 2, 8);

    const dsv4::QwenForwardResult baseline = engine.prefill({1, 2, 3});
    const dsv4::QwenPrefixCacheStats first = engine.prefix_cache_stats();
    require(first.reused_tokens == 0 && first.computed_tokens == 3,
            "initial prefix cache accounting");

    (void)engine.prefill({1, 2, 3, 4, 5});
    const dsv4::QwenPrefixCacheStats appended = engine.prefix_cache_stats();
    require(appended.resume_source == "live" && appended.reused_tokens == 3 &&
                appended.computed_tokens == 2,
            "appended prompt must reuse live recurrent state");

    const dsv4::QwenForwardResult shortened_result = engine.prefill({1, 2, 3});
    const dsv4::QwenPrefixCacheStats shortened = engine.prefix_cache_stats();
    require(shortened.resume_source == "snapshot" &&
                shortened.reused_tokens == 3 && shortened.computed_tokens == 0,
            "shorter prompt must reuse request-boundary snapshot");
    require(shortened_result.top_token == baseline.top_token &&
                shortened_result.top_logit == baseline.top_logit &&
                shortened_result.checksum == baseline.checksum,
            "snapshot result must match cold prefill");

    const dsv4::QwenForwardResult branched_result = engine.prefill({1, 2, 9});
    const dsv4::QwenPrefixCacheStats branched = engine.prefix_cache_stats();
    require(branched.resume_source == "snapshot" && branched.reused_tokens == 2 &&
                branched.computed_tokens == 1,
            "branched prompt must resume from interior snapshot");

    dsv4::QwenEngine cold_engine(dir, options, 2, 8);
    const dsv4::QwenForwardResult cold_branch = cold_engine.prefill({1, 2, 9});
    require(branched_result.top_token == cold_branch.top_token &&
                branched_result.top_logit == cold_branch.top_logit &&
                branched_result.checksum == cold_branch.checksum,
            "snapshot branch must match cold prefill");

    dsv4::QwenEngineOptions no_cache_options = options;
    no_cache_options.prefix_cache = false;
    dsv4::QwenEngine no_cache_engine(dir, no_cache_options, 2, 8);
    (void)no_cache_engine.prefill({1, 2, 3});
    const dsv4::QwenPrefixCacheStats no_cache = no_cache_engine.prefix_cache_stats();
    require(no_cache.reused_tokens == 0 && no_cache.computed_tokens == 3 &&
                no_cache.snapshots == 0 && no_cache.snapshot_bytes == 0,
            "disabled prefix cache must not retain snapshots");

    engine.reset();
    (void)engine.prefill({1, 2, 3});
    const dsv4::QwenPrefixCacheStats reset = engine.prefix_cache_stats();
    require(reset.reused_tokens == 0 && reset.computed_tokens == 3 &&
                reset.resume_source == "empty",
            "reset must invalidate prefix cache");
}

void exercise_mtp(const std::string& dir,
                  dsv4::QwenKvCacheDType cache_dtype) {
    dsv4::QwenEngineOptions plain_options;
    plain_options.tp_world = 1;
    plain_options.tp_rank = 0;
    plain_options.device = 0;
    plain_options.prefill_chunk_tokens = 2;
    plain_options.kv_cache_dtype = cache_dtype;
    plain_options.prefix_cache = false;

    dsv4::QwenEngineOptions mtp_options = plain_options;
    mtp_options.mtp = true;
    mtp_options.mtp_speculative_tokens = 4;
    dsv4::QwenEngine plain(dir, plain_options, 2, 16);
    dsv4::QwenEngine mtp(dir, mtp_options, 2, 16);
    const auto plain_outputs = plain.generate({1, 2, 3}, 8);
    const auto mtp_outputs = mtp.generate({1, 2, 3}, 8);
    require_same_generation(mtp_outputs, plain_outputs, "MTP");
    dsv4::QwenEngineOptions adaptive_options = mtp_options;
    adaptive_options.mtp_adaptive = true;
    dsv4::QwenEngine adaptive(dir, adaptive_options, 2, 16);
    const auto adaptive_outputs = adaptive.generate({1, 2, 3}, 8);
    require_same_generation(adaptive_outputs, plain_outputs, "adaptive MTP");

    require(mtp.mtp_stats().verify_count > 0 &&
                mtp.mtp_stats().proposed_drafts > 0 &&
                mtp.mtp_stats().prefill_seconds > 0.0,
            "MTP runtime statistics");
    require(mtp.position() <= mtp.max_context(), "MTP committed position");

    bool partial_depth_rejected = false;
    try {
        dsv4::QwenEngine invalid(dir, mtp_options, 1, 16);
    } catch (const std::runtime_error&) {
        partial_depth_rejected = true;
    }
    require(partial_depth_rejected, "MTP must reject a partial target model");
}

void exercise_mtp_prefix_cache(const std::string& dir,
                               dsv4::QwenKvCacheDType cache_dtype) {
    dsv4::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 2;
    options.kv_cache_dtype = cache_dtype;
    options.mtp = true;
    options.mtp_speculative_tokens = 2;
    options.state_snapshot_interval_tokens = 2;
    options.max_state_snapshots = 16;

    dsv4::QwenEngine cached(dir, options, 2, 32);
    const std::vector<int> base = {1, 2, 3};
    const auto base_outputs = cached.generate(base, 4);
    const dsv4::QwenPrefixCacheStats initial = cached.prefix_cache_stats();
    require(initial.resume_source == "empty" && initial.reused_tokens == 0 &&
                initial.computed_tokens == 3,
            "MTP initial chunk-boundary prompt prefix accounting");

    // A generated request leaves committed output tokens in cached_prompt. An
    // exact repeat of the original prompt therefore restores its request-boundary
    // snapshot rather than reusing the later live generation state.
    const auto repeated = cached.generate(base, 4);
    const dsv4::QwenPrefixCacheStats repeated_prefix = cached.prefix_cache_stats();
    require(repeated_prefix.resume_source == "snapshot" &&
                repeated_prefix.reused_tokens == 3 &&
                repeated_prefix.computed_tokens == 0,
            "MTP exact repeat prefix accounting");
    require_same_generation(repeated, base_outputs, "MTP exact repeat");

    std::vector<int> committed_after_repeat = base;
    committed_after_repeat.reserve(base.size() + repeated.size() - 1);
    for (size_t index = 0; index + 1 < repeated.size(); ++index) {
        committed_after_repeat.push_back(repeated[index].top_token);
    }
    std::vector<int> appended = committed_after_repeat;
    appended.push_back(repeated.back().top_token);
    require_cached_mtp_matches_cold(
        dir, cached, options, appended, 4, "live",
        static_cast<int>(committed_after_repeat.size()), 1,
        "MTP live continuation");
    require_cached_mtp_matches_cold(
        dir, cached, options, base, 4, "snapshot", 3, 0,
        "MTP shorter request-boundary snapshot");
    require_cached_mtp_matches_cold(
        dir, cached, options, {1, 2, 9, 10}, 4, "snapshot", 2, 2,
        "MTP interior branch");

    // Compression is a branch whose replacement suffix starts at an interior
    // snapshot. It specifically checks that predictor row S-1 is rebound from
    // the old token x[S] to the compressed request's new token.
    require_cached_mtp_matches_cold(
        dir, cached, options, {1, 2, 11}, 4, "snapshot", 2, 1,
        "MTP compressed suffix");

    cached.reset();
    require_cached_mtp_matches_cold(
        dir, cached, options, {7, 8, 9, 10, 11}, 4, "empty", 0, 5,
        "MTP reset after prefix reuse");
}

void exercise_cache_lifecycle(const std::string& dir,
                              dsv4::QwenKvCacheDType cache_dtype) {
    dsv4::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 2;
    options.kv_cache_dtype = cache_dtype;
    dsv4::QwenEngine engine(dir, options, 2, 8);
    require(engine.resident_weight_bytes() > 0, "resident Qwen weights missing");
    require(engine.resident_scale_bytes() > 0, "resident Qwen scales missing");

    const uint64_t fp16_cache_bytes = 2ULL * 8 * 4 * 128 * sizeof(uint16_t);
    const uint64_t fp8_cache_bytes = 2ULL * 8 * 4 * 128;
    const uint64_t fp8_scale_bytes =
        2ULL * 8 * 4 * (128 / 64) * sizeof(uint16_t);
    if (cache_dtype == dsv4::QwenKvCacheDType::Fp16) {
        require(engine.kv_cache_bytes() == fp16_cache_bytes,
                "FP16 KV cache accounting");
        require(engine.kv_cache_scale_bytes() == 0,
                "FP16 KV cache must not allocate scales");
    } else {
        require(engine.kv_cache_bytes() == fp8_cache_bytes,
                "FP8 KV cache accounting");
        require(engine.kv_cache_scale_bytes() == fp8_scale_bytes,
                "FP8 KV scale accounting");
    }

    const dsv4::QwenForwardResult prefill = engine.prefill({1, 2, 3});
    require(prefill.layers == 2 && prefill.dim == 128, "prefill metadata");
    require(prefill.position == 3, "chunked prefill position");
    require(prefill.top_token == 0, "zero fixture prefill greedy token");
    require(engine.activation_workspace_peak_bytes() > 0, "activation accounting");

    const dsv4::QwenForwardResult decoded = engine.decode_step(4);
    require(decoded.position == 4, "decode position");
    require(decoded.top_token == 0, "zero fixture decode greedy token");

    engine.reset();
    require(engine.position() == 0, "reset position");
    const dsv4::QwenForwardResult second = engine.prefill({4, 5});
    require(second.position == 2 && second.top_token == 0,
            "reset and second chunked prefill");
    std::cout << "  cache_dtype=" << dsv4::qwen_kv_cache_dtype_name(cache_dtype)
              << " kv_cache_bytes=" << engine.kv_cache_bytes()
              << " kv_cache_scale_bytes=" << engine.kv_cache_scale_bytes()
              << " activation_workspace_peak_bytes="
              << engine.activation_workspace_peak_bytes() << "\n";
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
        exercise_cache_lifecycle(dir, dsv4::QwenKvCacheDType::Fp16);
        exercise_cache_lifecycle(dir, dsv4::QwenKvCacheDType::Fp8);
        exercise_prefix_cache(dir, dsv4::QwenKvCacheDType::Fp16);
        exercise_prefix_cache(dir, dsv4::QwenKvCacheDType::Fp8);
        exercise_mtp(dir, dsv4::QwenKvCacheDType::Fp16);
        exercise_mtp(dir, dsv4::QwenKvCacheDType::Fp8);
        exercise_mtp_prefix_cache(dir, dsv4::QwenKvCacheDType::Fp16);
        exercise_mtp_prefix_cache(dir, dsv4::QwenKvCacheDType::Fp8);
        std::cout << "[PASS] test_qwen_engine layers=2 mtp=1\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_engine " << ex.what() << "\n";
        return 1;
    }
}
