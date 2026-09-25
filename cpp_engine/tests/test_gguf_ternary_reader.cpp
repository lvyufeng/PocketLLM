// The fork-private ternary types, at the reader level.
//
// Two things are under test and both are about arithmetic rather than about
// kernels. A 1.75-bit packing does not divide a row, so a shape does not imply a
// byte count the way it does for every dense type -- the reader has to carry the
// block geometry or it cannot address the tensor at all. And an *unrecognised*
// type must be refused by name: recording it as a zero-byte tensor leaves the
// bounds check passing trivially and every downstream reader walking a tensor it
// has no geometry for.
//
// The synthetic file is written here rather than read from a checkpoint so the
// test runs anywhere; the shape it carries is one the released
// Ternary-Bonsai-2-27B GGUF really has, so the expected byte count below is a
// figure the file itself confirms.

#include "gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr uint32_t kPtq1_0 = 143;
constexpr uint32_t kPq2_0 = 142;

class Writer {
public:
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void string(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    void zeros(size_t count) {
        std::vector<char> padding(count, '\0');
        raw(padding.data(), count);
    }
    const std::vector<char>& bytes() const { return bytes_; }

private:
    void raw(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        bytes_.insert(bytes_.end(), c, c + n);
    }
    std::vector<char> bytes_;
};

// A GGUF with exactly one tensor, aligned to 32 bytes.
std::string write_gguf(const std::string& path, uint32_t ggml_type,
                       const std::vector<uint64_t>& shape, uint64_t data_bytes) {
    Writer w;
    w.u32(0x46554747);  // "GGUF" little endian
    w.u32(3);           // version
    w.u64(1);           // tensor count
    w.u64(1);           // metadata count
    w.string("general.alignment");
    w.u32(4);  // GGUF_METADATA_VALUE_TYPE_UINT32
    w.u32(32);
    w.string("blk.0.ffn_gate.weight");
    w.u32(static_cast<uint32_t>(shape.size()));
    for (uint64_t dim : shape) w.u64(dim);
    w.u32(ggml_type);
    w.u64(0);  // tensor offset within the data section
    while (w.bytes().size() % 32 != 0) w.zeros(1);
    w.zeros(data_bytes);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    check(out.good(), "cannot open " + path + " for writing");
    out.write(w.bytes().data(), static_cast<std::streamsize>(w.bytes().size()));
    check(out.good(), "cannot write " + path);
    return path;
}

void check_reads(const std::string& path, uint32_t ggml_type, const std::vector<uint64_t>& shape,
                 uint64_t expected_bytes) {
    const std::string file = write_gguf(path, ggml_type, shape, expected_bytes);
    pocket::GGUFFile gguf(file);
    const pocket::GGUFTensorInfo* info = gguf.find_tensor("blk.0.ffn_gate.weight");
    check(info != nullptr, "the synthetic tensor should be found");
    check(info->ggml_type == ggml_type, "the tensor's ggml type should survive the round trip");
    check(info->nbytes == expected_bytes,
          "byte count should be " + std::to_string(expected_bytes) + " but is " +
              std::to_string(info->nbytes));
    std::remove(file.c_str());
}

void check_refuses(const std::string& path) {
    // 999 is not a type any model ships, and the point of the case is that the
    // reader must not quietly accept it as zero bytes.
    const std::string file = write_gguf(path, 999, {256, 256}, 0);
    bool threw = false;
    try {
        pocket::GGUFFile gguf(file);
    } catch (const std::exception& ex) {
        const std::string message = ex.what();
        threw = message.find("999") != std::string::npos &&
                message.find("blk.0.ffn_gate.weight") != std::string::npos;
        check(threw, "the refusal should name the type and the tensor, got: " + message);
    }
    std::remove(file.c_str());
    check(threw, "an unsupported GGML type must be refused, not recorded as zero bytes");
}

}  // namespace

int main() {
    try {
        // Names and dtype mapping, matched to `src/loader/gguf/quant_types.py`.
        check(pocket::ggml_type_name(kPtq1_0) == "ptq1_0", "type 143 should be named ptq1_0");
        check(pocket::ggml_type_name(kPq2_0) == "pq2_0", "type 142 should be named pq2_0");
        // The ternary checkpoint's other half: 96 `ssm_alpha`/`ssm_beta` tensors
        // at GGML type 30. This table used to spell BF16 as 32, which no GGUF
        // this engine reads had ever contradicted.
        check(pocket::ggml_type_name(30) == "bf16", "type 30 should be named bf16");
        check(pocket::ggml_type_to_dtype(30) == pocket::DType::BF16,
              "type 30 should map to DType::BF16");
        check(pocket::ggml_tensor_nbytes(30, {5120, 48}) == 5120ULL * 48 * 2,
              "a BF16 tensor should be two bytes per element");
        check(pocket::ggml_type_name(32) == "ggml_type_32",
              "32 is Q4_0_4_8 upstream and must not answer to bf16");
        check(pocket::ggml_type_to_dtype(kPtq1_0) == pocket::DType::PTQ1_0,
              "type 143 should map to DType::PTQ1_0");
        check(pocket::ggml_type_to_dtype(kPq2_0) == pocket::DType::PQ2_0,
              "type 142 should map to DType::PQ2_0");
        check(pocket::dtype_name(pocket::DType::PTQ1_0) == "ptq1_0",
              "DType::PTQ1_0 should stringify as ptq1_0");
        check(pocket::dtype_name(pocket::DType::PQ2_0) == "pq2_0",
              "DType::PQ2_0 should stringify as pq2_0");

        // Geometry: 128 weights per block, 28 bytes for PTQ1_0 and 34 for PQ2_0,
        // rounding up rather than truncating.
        check(pocket::ggml_tensor_nbytes(kPtq1_0, {128}) == 28, "one PTQ1_0 block should be 28 bytes");
        check(pocket::ggml_tensor_nbytes(kPtq1_0, {129}) == 56, "PTQ1_0 should round up to blocks");
        check(pocket::ggml_tensor_nbytes(kPq2_0, {128}) == 34, "one PQ2_0 block should be 34 bytes");
        check(pocket::ggml_tensor_nbytes(kPq2_0, {129}) == 68, "PQ2_0 should round up to blocks");

        // `output.weight` of the released checkpoint, which reports exactly this
        // byte count in its own header.
        check(pocket::ggml_tensor_nbytes(kPtq1_0, {5120, 248320}) == 278118400ULL,
              "the 5120x248320 embedding/head tensor should tile the file at 265.2 MiB");

        // The same count through a real parse, and the refusal for a type the
        // reader does not know.
        check_reads("/tmp/pocketllm_gguf_ternary_143.gguf", kPtq1_0, {5120, 248320}, 278118400ULL);
        check_reads("/tmp/pocketllm_gguf_ternary_142.gguf", kPq2_0, {128, 128}, 34ULL * 128 * 128 / 128);
        check_refuses("/tmp/pocketllm_gguf_ternary_unknown.gguf");

        std::cout << "[PASS] gguf_ternary_reader geometry and refusal\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
