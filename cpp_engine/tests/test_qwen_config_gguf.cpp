// Qwen3.5 config parsing, from a GGUF header instead of a config.json.
//
// The point of this test is that the two readers describe *one* model, so the
// same fields have to come out of both. The strong form of that check is at the
// bottom: the released ternary checkpoint's header is parsed and compared field
// for field against the values its own HF sibling (`/mnt/data2/Qwen3.8-27B-FP8`,
// the same architecture from the same authors) states in config.json. That check
// needs the checkpoints on disk and is skipped without them.
//
// What always runs is a scaled-down GGUF written here, carrying the same keys in
// the same shape, plus the failure paths -- a header whose SSM sizes disagree
// with each other, and one whose query projection is ungated. Both are silent
// bugs otherwise: the first would quietly pick the wrong value head count, the
// second would read a gate out of the middle of a query projection.

#include "qwen_config.hpp"
#include "model_registry.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <sys/stat.h>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

template <typename A, typename B>
void check_eq(const A& actual, const B& expected, const std::string& what) {
    if (!(actual == static_cast<A>(expected))) {
        std::cout << "[FAIL] " << what << " actual=" << actual << " expected=" << expected << "\n";
        ++failures;
    }
}

// Numbers the two files carry at different widths. A GGUF stores a float32 and
// config.json a JSON literal parsed to double, so the norm epsilon -- 1e-6 in
// both -- is 9.99999997e-07 on one side and 1.00000000e-06 on the other. They
// are the same number to every bit float32 has, which is all the runtime uses,
// so these are compared to a tolerance and not for equality.
void check_close(double actual, double expected, double tolerance, const std::string& what) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::cout << "[FAIL] " << what << " actual=" << actual << " expected=" << expected << "\n";
        ++failures;
    }
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

constexpr uint32_t kPtq1_0 = 143;
constexpr uint32_t kF32 = 0;

class Writer {
public:
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void string(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void zeros(size_t count) {
        std::vector<char> padding(count, '\0');
        raw(padding.data(), count);
    }
    size_t size() const { return bytes_.size(); }
    const std::vector<char>& bytes() const { return bytes_; }

private:
    void raw(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        bytes_.insert(bytes_.end(), c, c + n);
    }
    std::vector<char> bytes_;
};

struct TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = kPtq1_0;
};

// PTQ1_0 rounds a row up to 128-weight blocks; F32 is dense.
uint64_t tensor_bytes(const TensorSpec& spec) {
    uint64_t elems = 1;
    for (uint64_t d : spec.shape) elems *= d;
    if (spec.type == kPtq1_0) return ((elems + 127) / 128) * 28;
    return elems * 4;
}

// One GGUF holding the given metadata and its tensors' bytes, aligned to 32.
std::string write_gguf(const std::string& path,
                       const std::vector<std::pair<std::string, std::string>>& metadata,
                       const std::vector<TensorSpec>& tensors) {
    // Metadata is written from a small tagged list rather than a generic value
    // type: every key this config reader reads is an integer, a float, a string,
    // or the token table, and spelling those four out keeps the writer honest
    // about which GGUF value type each one really is.
    Writer body;
    uint64_t metadata_count = 0;
    for (const auto& entry : metadata) {
        const std::string& key = entry.first;
        const std::string& value = entry.second;
        body.string(key);
        if (key == "general.architecture" || key == "prism.hadamard.transform") {
            body.u32(8);  // string
            body.string(value);
        } else if (ends_with(key, "attention.layer_norm_rms_epsilon") ||
                   ends_with(key, "rope.freq_base")) {
            body.u32(6);  // float32
            body.f32(std::stof(value));
        } else if (key == "tokenizer.ggml.tokens") {
            // A comma-separated list of token texts, written as GGUF's string
            // array. The reader takes the vocabulary size from its length.
            std::vector<std::string> items;
            size_t start = 0;
            while (start <= value.size()) {
                const size_t comma = value.find(',', start);
                const size_t end = comma == std::string::npos ? value.size() : comma;
                items.push_back(value.substr(start, end - start));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            body.u32(9);  // array
            body.u32(8);  // of string
            body.u64(items.size());
            for (const std::string& item : items) body.string(item);
        } else {
            body.u32(10);  // uint64
            body.u64(std::stoull(value));
        }
        ++metadata_count;
    }

    Writer header;
    header.u32(0x46554747);  // "GGUF" little endian
    header.u32(3);
    header.u64(tensors.size());
    header.u64(metadata_count);
    const std::vector<char>& meta_bytes = body.bytes();

    uint64_t data_bytes = 0;
    for (const TensorSpec& spec : tensors) {
        data_bytes += tensor_bytes(spec);
    }

    std::vector<char> buffer(header.bytes());
    const auto push = [&buffer](const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        buffer.insert(buffer.end(), c, c + n);
    };
    const auto push_u32 = [&push](uint32_t v) { push(&v, 4); };
    const auto push_u64 = [&push](uint64_t v) { push(&v, 8); };
    const auto push_string = [&push, &push_u64](const std::string& s) {
        push_u64(s.size());
        push(s.data(), s.size());
    };

    push(meta_bytes.data(), meta_bytes.size());
    uint64_t offset = 0;
    for (const TensorSpec& spec : tensors) {
        push_string(spec.name);
        push_u32(static_cast<uint32_t>(spec.shape.size()));
        for (uint64_t d : spec.shape) push_u64(d);
        push_u32(spec.type);
        push_u64(offset);
        offset += tensor_bytes(spec);
    }
    while (buffer.size() % 32 != 0) buffer.push_back('\0');
    buffer.resize(buffer.size() + data_bytes, '\0');

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        throw std::runtime_error("cannot write " + path);
    }
    out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    out.close();
    return path;
}

// The released checkpoint's own values, scaled down to a size a test can write.
// Every ratio is the real one: 8 blocks with a full-attention layer every 4th,
// 2 key heads over 1 value head, head length 8 with 2 of them rotated, 4 value
// heads at state 4 (so inner_size is their product), and a gated query.
//
// The keys are prefixed by the architecture the header declares, exactly as the
// real file prefixes its own with `qwen35.` -- so the same fixture can declare a
// different architecture and still be a well-formed header.
std::vector<std::pair<std::string, std::string>> scaled_metadata(const std::string& arch) {
    const auto key = [&arch](const char* suffix) { return arch + "." + suffix; };
    return {
        {"general.architecture", arch},
        {key("block_count"), "8"},
        {key("embedding_length"), "16"},
        {key("feed_forward_length"), "32"},
        {key("context_length"), "262144"},
        {key("attention.head_count"), "2"},
        {key("attention.head_count_kv"), "1"},
        {key("attention.key_length"), "8"},
        {key("attention.value_length"), "8"},
        {key("attention.layer_norm_rms_epsilon"), "0.000001"},
        {key("rope.dimension_count"), "2"},
        {key("rope.freq_base"), "10000000"},
        {key("full_attention_interval"), "4"},
        {key("ssm.conv_kernel"), "4"},
        {key("ssm.group_count"), "2"},
        {key("ssm.inner_size"), "16"},
        {key("ssm.state_size"), "4"},
        {key("ssm.time_step_rank"), "4"},
        {"tokenizer.ggml.tokens", "a,b,c,d,e,f,g"},
        {"tokenizer.ggml.bos_token_id", "248044"},
        {"tokenizer.ggml.eos_token_id", "248046"},
    };
}

std::vector<TensorSpec> scaled_tensors(uint64_t q_out) {
    return {
        {"blk.0.attn_qkv.weight", {16, 24}, kPtq1_0},
        {"blk.3.attn_q.weight", {16, q_out}, kPtq1_0},
        {"blk.3.attn_k.weight", {16, 8}, kPtq1_0},
        {"blk.3.attn_v.weight", {16, 8}, kPtq1_0},
        {"blk.3.attn_output.weight", {8, 16}, kPtq1_0},
        {"blk.0.ffn_gate.weight", {16, 32}, kPtq1_0},
        {"blk.0.ffn_down.weight", {32, 16}, kPtq1_0},
        {"blk.0.ffn_up.weight", {16, 32}, kPtq1_0},
        {"token_embd.weight", {16, 7}, kPtq1_0},
        {"output.weight", {16, 7}, kPtq1_0},
        {"blk.3.attn_q_norm.weight", {8}, kF32},
        {"blk.3.attn_k_norm.weight", {8}, kF32},
    };
}

void check_scaled_config(const std::string& dir) {
    // attention_dim is head_count * key_length = 16, so a gated query is 32 wide.
    const std::string file =
        write_gguf(dir + "/scaled.gguf", scaled_metadata("qwen35"), scaled_tensors(32));

    const pocket::QwenConfig cfg = pocket::QwenConfig::from_gguf(file);

    // The declared name is llama.cpp's; the runtime keys on the canonical one.
    check_eq(cfg.architecture, std::string("qwen3_5"), "qwen35 folds onto the canonical name");
    check_eq(cfg.model_type, std::string("qwen3_5"), "model_type follows the architecture");
    check_eq(cfg.vocab_size, uint64_t{7}, "the vocabulary size is the token table's length");
    check_eq(cfg.hidden_size, uint64_t{16}, "embedding_length is the hidden size");
    check_eq(cfg.num_hidden_layers, uint64_t{8}, "block_count is the layer count");
    check_eq(cfg.max_position_embeddings, uint64_t{262144}, "context_length is the position limit");
    check_eq(cfg.mlp.intermediate_size, uint64_t{32}, "feed_forward_length is the MLP width");
    check_close(cfg.rms_norm_eps, 1e-6, 1e-12, "layer_norm_rms_epsilon is the norm epsilon");
    check_eq(cfg.rope_theta, 1e7, "rope.freq_base is the rope theta");
    // 2 of 8 head dimensions is the fraction config.json would state directly.
    check_eq(cfg.partial_rotary_factor, 0.25, "the rotary fraction is dimension over head length");
    check_eq(cfg.partial_rotary_dim(), uint64_t{2}, "rotary dim is the stated dimension count");

    check_eq(cfg.linear_attention.key_heads, uint64_t{2}, "ssm.group_count is the key head count");
    check_eq(cfg.linear_attention.value_heads, uint64_t{4}, "ssm.time_step_rank is the value head count");
    check_eq(cfg.linear_attention.key_head_dim, uint64_t{4}, "ssm.state_size is the key head dim");
    check_eq(cfg.linear_attention.value_head_dim, uint64_t{4}, "ssm.state_size is the value head dim");
    check_eq(cfg.linear_attention.conv_kernel_dim, uint64_t{4}, "ssm.conv_kernel is the conv width");
    check_eq(cfg.linear_attention.qkv_dim(), uint64_t{32}, "the fused qkv width follows");
    check_eq(cfg.linear_attention.value_state_dim(), uint64_t{16}, "the value state width follows");

    check_eq(cfg.full_attention.num_heads, uint64_t{2}, "head_count is the query head count");
    check_eq(cfg.full_attention.num_key_value_heads, uint64_t{1}, "head_count_kv is the KV head count");
    check_eq(cfg.full_attention.head_dim, uint64_t{8}, "key_length is the head dimension");
    check(cfg.full_attention.output_gate, "a query projection of twice the attention width is gated");

    check_eq(cfg.layer_types.size(), size_t{8}, "one layer type per block");
    check_eq(cfg.linear_attention_layers(), uint64_t{6}, "six of eight blocks are linear attention");
    check_eq(cfg.full_attention_layers(), uint64_t{2}, "two of eight blocks are full attention");
    check_eq(cfg.layer_type_name(0), std::string("linear_attention"), "block 0 is linear attention");
    check_eq(cfg.layer_type_name(3), std::string("full_attention"), "block 3 is full attention");
    check_eq(cfg.layer_type_name(7), std::string("full_attention"), "the last block is full attention");

    check_eq(cfg.eos_token_ids.size(), size_t{2}, "both declared stop tokens are kept");
    if (cfg.eos_token_ids.size() == 2) {
        check_eq(cfg.eos_token_ids[0], 248046, "the end-of-turn token comes first");
        check_eq(cfg.eos_token_ids[1], 248044, "then the end-of-text token");
    }

    check(pocket::canonical_qwen_architecture("qwen35") == "qwen3_5",
          "qwen35 canonicalises to qwen3_5");
    check(pocket::canonical_qwen_architecture("qwen3_5_text") == "qwen3_5",
          "qwen3_5_text canonicalises to qwen3_5");
    check(pocket::gguf_declared_architecture(file) == "qwen35",
          "the declared architecture is readable without building a config");
    check(pocket::is_qwen3_5_checkpoint(file), "a qwen35 GGUF is dispatchable as Qwen3.5");
    // The engine registry folds a checkpoint name onto the key an engine
    // registers, and this is that fold: `qwen35` has to reach the same factory
    // `qwen3_5` does, or dispatch finds no engine for a checkpoint whose config
    // it just parsed.
    check_eq(pocket::detect_architecture(file), std::string("qwen3_5"),
             "the registry resolves qwen35 to the Qwen3.5 engine key");

    // A header that names another architecture must not be folded onto this key.
    const std::string other =
        write_gguf(dir + "/other_arch.gguf", scaled_metadata("llama"), scaled_tensors(16));
    check(pocket::detect_architecture(other) != "qwen3_5",
          "a header naming another architecture resolves elsewhere");
}

// A scaled header with one key wrong, or one tensor ungated. Each has to throw
// or flip the derived field rather than be absorbed.
void check_failure_paths(const std::string& dir) {
    {
        auto metadata = scaled_metadata("qwen35");
        for (auto& entry : metadata) {
            if (entry.first == "qwen35.ssm.inner_size") entry.second = "12";  // not 4 * 4
        }
        const std::string file = write_gguf(dir + "/bad_inner.gguf", metadata, scaled_tensors(16));
        bool threw = false;
        try {
            (void)pocket::QwenConfig::from_gguf(file);
        } catch (const std::exception& ex) {
            threw = std::string(ex.what()).find("ssm.time_step_rank") != std::string::npos;
        }
        check(threw, "SSM sizes that disagree with each other are refused by name");
    }
    {
        // A query projection that is 1.5x the attention width: neither the
        // attention width nor twice it, so the gate cannot be read off it.
        const std::string file =
            write_gguf(dir + "/bad_width.gguf", scaled_metadata("qwen35"), scaled_tensors(24));
        bool threw = false;
        try {
            (void)pocket::QwenConfig::from_gguf(file);
        } catch (const std::exception& ex) {
            threw = std::string(ex.what()).find("tile") != std::string::npos;
        }
        check(threw, "a query projection that is neither attention width is refused");
    }
    {
        // The same header with an ungated query projection: the width is the
        // attention width rather than twice it, and the gate has to come out off.
        const std::string file =
            write_gguf(dir + "/ungated.gguf", scaled_metadata("qwen35"), scaled_tensors(16));
        const pocket::QwenConfig cfg = pocket::QwenConfig::from_gguf(file);
        check(!cfg.full_attention.output_gate,
              "a query projection of the attention width exactly is ungated");
    }
    {
        // A header declaring a different architecture must not be answered as
        // Qwen3.5 just because it has the keys.
        auto metadata = scaled_metadata("llama");
        const std::string file = write_gguf(dir + "/other.gguf", metadata, scaled_tensors(16));
        bool threw = false;
        try {
            (void)pocket::QwenConfig::from_gguf(file);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "an architecture this reader does not implement is refused");
    }
}

// The real check, when the checkpoints are on the box: both readers, one model.
void check_against_the_released_checkpoint() {
    const std::string gguf = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
    const std::string hf = "/mnt/data2/Qwen3.8-27B-FP8";
    if (!file_exists(gguf) || !file_exists(hf + "/config.json")) {
        std::cout << "[SKIP] the released checkpoint or its HF sibling is not on disk\n";
        return;
    }

    const pocket::QwenConfig from_gguf = pocket::QwenConfig::from_gguf(gguf);
    const pocket::QwenConfig from_hf = pocket::QwenConfig::from_hf_config(hf);

    check_eq(from_gguf.hidden_size, from_hf.hidden_size, "hidden size agrees across readers");
    check_eq(from_gguf.num_hidden_layers, from_hf.num_hidden_layers, "layer count agrees");
    check_eq(from_gguf.vocab_size, from_hf.vocab_size, "vocabulary size agrees");
    check_eq(from_gguf.max_position_embeddings, from_hf.max_position_embeddings, "context agrees");
    check_eq(from_gguf.mlp.intermediate_size, from_hf.mlp.intermediate_size, "MLP width agrees");
    check_close(from_gguf.rms_norm_eps, from_hf.rms_norm_eps, 1e-12, "norm epsilon agrees");
    check_eq(from_gguf.rope_theta, from_hf.rope_theta, "rope theta agrees");
    check_eq(from_gguf.partial_rotary_factor, from_hf.partial_rotary_factor,
             "the derived rotary fraction equals the stated one");
    check_eq(from_gguf.partial_rotary_dim(), from_hf.partial_rotary_dim(), "rotary dim agrees");
    check_eq(from_gguf.linear_attention.key_heads, from_hf.linear_attention.key_heads,
             "linear key heads agree");
    check_eq(from_gguf.linear_attention.value_heads, from_hf.linear_attention.value_heads,
             "linear value heads agree");
    check_eq(from_gguf.linear_attention.key_head_dim, from_hf.linear_attention.key_head_dim,
             "linear key head dim agrees");
    check_eq(from_gguf.linear_attention.value_head_dim, from_hf.linear_attention.value_head_dim,
             "linear value head dim agrees");
    check_eq(from_gguf.linear_attention.conv_kernel_dim, from_hf.linear_attention.conv_kernel_dim,
             "conv kernel width agrees");
    check_eq(from_gguf.full_attention.num_heads, from_hf.full_attention.num_heads, "query heads agree");
    check_eq(from_gguf.full_attention.num_key_value_heads, from_hf.full_attention.num_key_value_heads,
             "KV heads agree");
    check_eq(from_gguf.full_attention.head_dim, from_hf.full_attention.head_dim, "head dim agrees");
    check_eq(from_gguf.full_attention.output_gate, from_hf.full_attention.output_gate,
             "the query gate derived from the tensor matches the declared one");
    check_eq(from_gguf.full_attention_layers(), from_hf.full_attention_layers(),
             "the interval reproduces the declared layer list");
    check_eq(from_gguf.linear_attention_layers(), from_hf.linear_attention_layers(),
             "the linear layer count matches too");
    check_eq(from_gguf.layer_types.size(), from_hf.layer_types.size(), "one layer type per block");
    for (size_t i = 0; i < from_gguf.layer_types.size() && i < from_hf.layer_types.size(); ++i) {
        if (from_gguf.layer_types[i] != from_hf.layer_types[i]) {
            check(false, "layer " + std::to_string(i) + " has the wrong type");
            break;
        }
    }
    check_eq(from_gguf.eos_token_ids.size(), from_hf.eos_token_ids.size(), "stop token count agrees");
    for (size_t i = 0; i < from_gguf.eos_token_ids.size() && i < from_hf.eos_token_ids.size(); ++i) {
        check_eq(from_gguf.eos_token_ids[i], from_hf.eos_token_ids[i],
                 "stop token " + std::to_string(i) + " agrees");
    }

    std::cout << "[INFO] released checkpoint: " << from_gguf.to_string();
}

}  // namespace

int main() {
    const std::string dir = "/tmp/pocketllm_qwen_config_gguf";
    const std::string mkdir = "mkdir -p '" + dir + "'";
    if (std::system(mkdir.c_str()) != 0) {
        std::cout << "[FAIL] cannot create " << dir << "\n";
        return 1;
    }

    check_scaled_config(dir);
    check_failure_paths(dir);
    check_against_the_released_checkpoint();

    if (failures != 0) {
        std::cout << "test_qwen_config_gguf failures=" << failures << "\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_config_gguf\n";
    return 0;
}
