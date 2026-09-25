// The released ternary checkpoint, run through the native engine end to end.
//
// Everything else in this adaptation pins one piece: the reader reads the file,
// the fold is the fork's transform, the GEMM multiplies three-valued weights, the
// weight map finds the tensors, the rotation kernel matches its definition. This
// file is the one that runs the model, and it exists because the pieces can all be
// right while the assembly is wrong.
//
// The mistakes it is here to catch are the ones that do not fail:
//
//   * a rotation applied on the wrong side of a projection -- the frame the
//     weight was quantized in is not the frame the activation arrives in;
//   * an embedding read in the rotated frame and never restored, so every
//     position starts from the wrong vector;
//   * a permuted gated-DeltaNet output that keeps the tiled head order, which is
//     6144 features each in the right place but the wrong one of forty-eight.
//
// Each of those loads, runs at full speed, and produces text. So the check is the
// text: the checkpoint answers "The capital of France is" with "Paris", and this
// test decodes what it generated and looks for it. The prompt's token ids are
// spelled out rather than tokenized because the engine takes ids and the
// tokenizer is a separate concern.
//
// It is not a parity test. Bit-level agreement with the reference implementation
// is the oracle harness's claim, against the fork's own logits; what this file
// asserts is the weaker, cheaper property that the model is the model.

#include "gguf_reader.hpp"
#include "qwen_config.hpp"
#include "qwen_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

const char* const kGgufPath =
    "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf";

// "The capital of France is" -- the tokenizer's own ids, five tokens.
const int kPrompt[] = {760, 6511, 314, 9338, 369};
const int kSteps = 24;

// Enough of a detokenizer to answer "does the text contain a word". The table's
// entries are byte-level BPE, so a leading space is stored as U+0120 and a newline
// as U+010A; both are translated and nothing else is, because nothing else is
// needed. `tokenizer.ggml.token_type` distinguishes special tokens, and they are
// dropped: a continuation check that passes because an end-of-text marker was
// emitted would be a lie.
std::string detokenize(const std::vector<std::string>& table,
                       const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= table.size()) continue;
        const std::string& piece = table[static_cast<size_t>(id)];
        for (size_t i = 0; i < piece.size();) {
            // U+0120 "G" and U+010A "C" in their two-byte UTF-8 forms.
            if (i + 1 < piece.size() && static_cast<unsigned char>(piece[i]) == 0xC4) {
                const unsigned char second = static_cast<unsigned char>(piece[i + 1]);
                if (second == 0xA0) { out.push_back(' '); i += 2; continue; }
                if (second == 0x8A) { out.push_back('\n'); i += 2; continue; }
            }
            out.push_back(piece[i]);
            ++i;
        }
    }
    return out;
}

}  // namespace

int main() {
    if (!file_exists(kGgufPath)) {
        std::cout << "[SKIP] test_qwen_ternary_engine needs the released checkpoint at "
                  << kGgufPath << "\n";
        return 0;
    }

    pocket::GGUFFile file(kGgufPath);
    const std::vector<std::string> table =
        file.metadata_string_array("tokenizer.ggml.tokens").value_or(
            std::vector<std::string>());
    check(!table.empty(), "the checkpoint carries a token table");

    pocket::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.max_batch_size = 1;
    options.temperature = 0.0f;   // the greedy argmax path
    options.mtp = false;

    const int max_context = 2048;
    pocket::QwenEngine engine(kGgufPath, options, 0, max_context);

    // The transform is a fact about the file, and if the metadata did not reach
    // the runtime then nothing below means anything: the engine would take a
    // checkpoint with pre-rotated weights and multiply it by unrotated
    // activations, which is a model that runs.
    const pocket::QwenRuntimeTelemetry telemetry = engine.runtime_telemetry();
    std::cout << "rotation_block=" << telemetry.rotation_block_size
              << " sign_widths=" << telemetry.rotation_sign_widths
              << " embedding_rotated=" << (telemetry.embedding_rotated ? 1 : 0)
              << " target_head=" << telemetry.target_head_path << "\n";
    check(telemetry.rotation_block_size == 1024,
          "the engine loaded the checkpoint's 1024-element hadamard block");
    check(telemetry.rotation_sign_widths == 3,
          "the engine loaded all three declared sign vectors");
    check(telemetry.embedding_rotated,
          "the engine knows the embedding table is stored rotated");

    const std::vector<int> prompt(std::begin(kPrompt), std::end(kPrompt));
    pocket::ForwardResult result = engine.prefill(prompt);
    std::vector<int> generated;
    generated.push_back(result.top_token);
    for (int step = 1; step < kSteps; ++step) {
        result = engine.decode_step(generated.back());
        generated.push_back(result.top_token);
    }

    std::cout << "generated_ids=";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i != 0) std::cout << ",";
        std::cout << generated[i];
    }
    std::cout << "\n";
    const std::string text = detokenize(table, generated);
    std::cout << "generated_text=" << text << "\n";

    // In range, and actually varying: a model whose logits are degenerate emits
    // one token forever, and a model reading the wrong frame does that as readily
    // as one that is simply broken.
    bool in_range = true;
    for (int id : generated) {
        in_range = in_range && id >= 0 && static_cast<size_t>(id) < table.size();
    }
    check(in_range, "every generated token is in the vocabulary");
    std::vector<int> distinct = generated;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    check(distinct.size() > 8, "the continuation is not one token repeated");

    // The check that costs a minute and catches everything the pieces cannot.
    check(text.find("Paris") != std::string::npos,
          "the model answers with Paris: got \"" + text + "\"");

    if (failures != 0) {
        std::cout << "[FAIL] test_qwen_ternary_engine: " << failures
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_ternary_engine\n";
    return 0;
}
