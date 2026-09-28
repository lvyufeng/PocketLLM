// The Hadamard weight unfold, at the shard boundaries that make it run.
//
// A rank whose column shard takes part of a rotation block has no activation to
// rotate -- a block's outputs read the whole block and part of it lives on
// another rank -- so the loader undoes the rotation on the checkpoint's own
// matrix at load time instead. See `qwen_rotation_needs_weight_unfold`. The
// checkpoint's block is 1024 wide and a TP4 shard is 1280 columns, so every rank
// of that split takes part of a block and every rank unfolds.
//
// Two things about that loop are not arithmetic, and they are what this test
// pins. It transforms only the blocks the shard reads, and it spreads the rows
// over the host's threads. Each is argued to be bit-exact rather than
// approximately equal -- the butterfly has no cross-block term, and rows write
// disjoint slices of the destination -- and bit-exactness is the whole claim, so
// the reference below is the *untrimmed* transform taken over the whole row and
// sliced afterwards: the shipped behaviour, reached through a different path.
// The same reference is then reached with the worker count pinned to one and to
// more than the row count, which is what ties the parallel path to the serial
// one.
//
// The reference transform is written out-of-place, with a fresh buffer per
// level, which is a different data flow from the in-place butterfly under test
// and rounds in the same order -- the pairs within a level are disjoint, so
// where the results are written cannot change them.
//
// The third case drops the block to 64, below the 128-weight packing block. The
// span then has to be rounded to a packing boundary as well as to a transform
// one, which is the only reason the two are not the same number: a span that
// started inside a pack would decode that pack from the wrong byte offset.
//
// Synthetic file, synthetic blocks: no checkpoint, no device and no vendor SDK.
// On a backend whose kernels read the ternary pack in place the unfold never
// runs at all, and each case says so rather than passing vacuously.

#include "qwen_gguf.hpp"
#include "qwen_weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

constexpr uint64_t kPackWeights = 128;
constexpr uint64_t kPackBytes = 28;
constexpr uint32_t kPtq1_0 = 143;

// The GGUF metadata value type codes, and the array element codes, exactly as
// `gguf_reader.cpp` reads them back.
constexpr uint32_t kTypeInt64 = 11;
constexpr uint32_t kTypeString = 8;
constexpr uint32_t kTypeArray = 9;

class GgufWriter {
public:
    void u32(uint32_t value) { raw(&value, sizeof(value)); }
    void u64(uint64_t value) { raw(&value, sizeof(value)); }
    void i64(int64_t value) { raw(&value, sizeof(value)); }
    void i8(int8_t value) { raw(&value, sizeof(value)); }
    void string(const std::string& value) {
        u64(value.size());
        raw(value.data(), value.size());
    }
    void raw(const void* data, size_t bytes) {
        const char* chars = static_cast<const char*>(data);
        bytes_.insert(bytes_.end(), chars, chars + bytes);
    }

    void meta_u32(const std::string& key, uint32_t value) {
        string(key);
        u32(4);  // GGUF_METADATA_VALUE_TYPE_UINT32
        u32(value);
    }
    void meta_int(const std::string& key, int64_t value) {
        string(key);
        u32(kTypeInt64);
        i64(value);
    }
    void meta_string(const std::string& key, const std::string& value) {
        string(key);
        u32(kTypeString);
        string(value);
    }
    void meta_int_array(const std::string& key, const std::vector<int64_t>& values) {
        string(key);
        u32(kTypeArray);
        u32(kTypeInt64);
        u64(values.size());
        for (int64_t value : values) i64(value);
    }
    // The block declares its signs as an array of int8 -- the values are +1 and
    // -1 and the file uses the smallest width that carries them -- while the
    // reader hands every integer array back as int64. The narrow write and the
    // wide read are both part of what is being reproduced here.
    void meta_i8_array(const std::string& key, const std::vector<int64_t>& values) {
        std::vector<int8_t> narrowed;
        narrowed.reserve(values.size());
        for (int64_t value : values) narrowed.push_back(static_cast<int8_t>(value));
        string(key);
        u32(kTypeArray);
        u32(1);  // GGUF_METADATA_VALUE_TYPE_INT8
        u64(narrowed.size());
        raw(narrowed.data(), narrowed.size());
    }
    void meta_string_array(const std::string& key,
                           const std::vector<std::string>& values) {
        string(key);
        u32(kTypeArray);
        u32(kTypeString);
        u64(values.size());
        for (const std::string& value : values) string(value);
    }

    const std::vector<char>& bytes() const { return bytes_; }

private:
    std::vector<char> bytes_;
};

// A one-tensor GGUF carrying the checkpoint's own `prism.hadamard.*` block. The
// tensor's GGML dims are written input-first, as a GGUF states them, so the
// canonical shape the reader reports is [rows, cols].
std::string write_hadamard_gguf(const std::string& path, uint64_t block,
                                uint64_t rows, uint64_t cols,
                                const std::vector<int64_t>& signs,
                                const std::vector<uint8_t>& payload) {
    GgufWriter w;
    w.u32(0x46554747);  // "GGUF", little endian
    w.u32(3);           // version
    w.u64(1);           // tensor count
    w.u64(10);          // metadata count

    w.meta_u32("general.alignment", 32);
    w.meta_int("prism.hadamard.version", 1);
    w.meta_string("prism.hadamard.transform",
                  "normalized-sylvester-walsh-hadamard");
    w.meta_string("prism.hadamard.axis", "input-last-dimension");
    w.meta_string("prism.hadamard.sign_mode", "explicit");
    w.meta_int("prism.hadamard.block_size", static_cast<int64_t>(block));
    w.meta_int_array("prism.hadamard.sign_widths", {static_cast<int64_t>(cols)});
    w.meta_i8_array("prism.hadamard.sign_values", signs);
    w.meta_string_array("prism.hadamard.weight_names", {"blk.0.ffn_down.weight"});
    w.meta_string_array("prism.hadamard.inverse_weight_names", {});
    // The value-head geometry the loader reorders against. Absent is fine for
    // every tensor but `ssm_out`, which this file does not hold; the block's own
    // `gdn_v_grouped` is absent too, so nothing reads them.

    w.string("blk.0.ffn_down.weight");
    w.u32(2);  // dims
    w.u64(cols);
    w.u64(rows);
    w.u32(kPtq1_0);
    w.u64(0);  // offset within the data section
    while (w.bytes().size() % 32 != 0) w.i8(0);
    w.raw(payload.data(), payload.size());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    check(out.good(), "cannot open " + path + " for writing");
    out.write(w.bytes().data(), static_cast<std::streamsize>(w.bytes().size()));
    check(out.good(), "cannot write " + path);
    out.close();
    return path;
}

// Deterministic, and the same sequence on every host.
struct Random {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    uint8_t byte() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint8_t>(state >> 33);
    }
};

// The block scales, all exactly representable in fp16 and all finite, which is
// what lets the reference below convert them with the converter under this one
// rather than with a second one of its own.
const float kScales[] = {
    0.0625f, -0.0625f, 1.0f, -1.0f, 0.5f, -2.0f, 0.001953125f, 65504.0f,
};

// fp16 to fp32, exactly, for the values a decoded ternary row can hold: the
// block scales above and the signed zeros they produce. Written from the format
// rather than copied from the loader, so a disagreement about subnormals or
// about the sign of a zero would be a disagreement between two readings.
float half_to_float(uint16_t bits) {
    const float sign = (bits & 0x8000u) != 0 ? -1.0f : 1.0f;
    const int exponent = static_cast<int>((bits >> 10) & 0x1Fu);
    const int mantissa = static_cast<int>(bits & 0x3FFu);
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    return sign * std::ldexp(static_cast<float>(1024 + mantissa), exponent - 25);
}

// The inverse transform over a whole row, block by block, transcribed from the
// definition: `W' = W . R^-1` with `R = (1/sqrt(N)) H_N diag(s)`, `H` in the
// natural ordering the butterfly lands on.
//
// Level-outer and out of place, one fresh buffer per level. The pairs within a
// level are disjoint, so writing a level into a new buffer computes the same
// sums in the same order the in-place loop does -- which is what makes this a
// reference for a bit-exact claim rather than for a tolerant one.
void reference_inverse(float* data, uint64_t width, uint64_t block,
                       const std::vector<int64_t>& signs) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));
    std::vector<float> tile(static_cast<size_t>(block));
    std::vector<float> next(static_cast<size_t>(block));
    for (uint64_t base = 0; base < width; base += block) {
        for (uint64_t i = 0; i < block; ++i) {
            tile[i] = data[base + i] * scale;
        }
        for (uint64_t step = 1; step < block; step <<= 1) {
            for (uint64_t i = 0; i < block; i += 2 * step) {
                for (uint64_t j = 0; j < step; ++j) {
                    const float low = tile[i + j];
                    const float high = tile[i + step + j];
                    next[i + j] = low + high;
                    next[i + step + j] = low - high;
                }
            }
            tile.swap(next);
        }
        for (uint64_t i = 0; i < block; ++i) {
            data[base + i] = tile[i] * static_cast<float>(signs[base + i]);
        }
    }
}

// The shipped decode, the whole-row transform and the slice -- what the loader
// did before this change, and the bytes it has to keep producing.
std::vector<uint8_t> reference_row(const std::vector<uint8_t>& payload,
                                   uint64_t row, uint64_t cols, uint64_t block,
                                   const std::vector<int64_t>& signs, uint64_t shard_start,
                                   uint64_t shard_size) {
    const uint64_t row_bytes = cols / kPackWeights * kPackBytes;
    const uint8_t* packed = payload.data() + row * row_bytes;
    std::vector<uint8_t> decoded(static_cast<size_t>(cols) * 2, 0);
    pocket::qwen_decode_ptq1_0(packed, decoded.data(), cols, "test_qwen_hadamard_unfold");

    std::vector<float> line(static_cast<size_t>(cols));
    const uint16_t* halves = reinterpret_cast<const uint16_t*>(decoded.data());
    for (uint64_t i = 0; i < cols; ++i) line[i] = half_to_float(halves[i]);
    reference_inverse(line.data(), cols, block, signs);

    std::vector<uint8_t> out(static_cast<size_t>(shard_size) * 2, 0);
    uint16_t* narrowed = reinterpret_cast<uint16_t*>(out.data());
    for (uint64_t i = 0; i < shard_size; ++i) {
        narrowed[i] = pocket::qwen_float_to_fp16_bits(line[shard_start + i]);
    }
    return out;
}

std::vector<uint8_t> reference_rows(const std::vector<uint8_t>& payload, uint64_t rows,
                                    uint64_t cols, uint64_t block,
                                    const std::vector<int64_t>& signs,
                                    uint64_t shard_start, uint64_t shard_size) {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(rows * shard_size * 2));
    for (uint64_t row = 0; row < rows; ++row) {
        const std::vector<uint8_t> line =
            reference_row(payload, row, cols, block, signs, shard_start, shard_size);
        out.insert(out.end(), line.begin(), line.end());
    }
    return out;
}

pocket::QwenTensorRef make_ref(const std::string& canonical, uint64_t rows, uint64_t cols,
                               uint64_t shard_start, uint64_t shard_size) {
    pocket::QwenTensorRef ref;
    ref.name = canonical;
    ref.dtype = pocket::SafeDType::U8;
    // What the map would set for a ternary tensor on a backend that has no
    // block reader: the decoded matrix is what the dense kernels take.
    ref.device_dtype = pocket::SafeDType::F16;
    ref.full_shape = {rows, cols};
    ref.local_shape = {rows, shard_size};
    ref.rule = pocket::QwenShardRule::ColumnParallel;
    ref.shard_dim = 1;
    ref.shard_start = shard_start;
    ref.shard_size = shard_size;
    ref.ternary_blocks = true;
    ref.found = true;
    return ref;
}

std::vector<uint8_t> materialize(const pocket::QwenCheckpointSource& source,
                                 const pocket::QwenTensorRef& ref) {
    return pocket::qwen_materialize_host_tensor(source, ref).bytes;
}

std::string describe(uint64_t block, uint64_t shard_start, uint64_t shard_size) {
    return "block=" + std::to_string(block) + " shard=" + std::to_string(shard_start) +
           "+" + std::to_string(shard_size);
}

// One shard, three ways: against the whole-row reference, and against itself
// with the worker count pinned to one and to more workers than there are rows.
void check_shard(const pocket::QwenCheckpointSource& source,
                 const std::vector<uint8_t>& payload, uint64_t rows, uint64_t cols,
                 uint64_t block, const std::vector<int64_t>& signs, uint64_t shard_start,
                 uint64_t shard_size) {
    const std::string what = describe(block, shard_start, shard_size);
    check(shard_start % block != 0 || shard_size % block != 0,
          what + ": the case must not be block aligned, or the loader never unfolds it");
    const pocket::QwenTensorRef ref =
        make_ref("model.language_model.layers.0.mlp.down_proj.weight", rows, cols,
                 shard_start, shard_size);
    const std::vector<uint8_t> want =
        reference_rows(payload, rows, cols, block, signs, shard_start, shard_size);

    ::setenv("QWEN_LOAD_THREADS", "1", 1);
    const std::vector<uint8_t> serial = materialize(source, ref);
    ::setenv("QWEN_LOAD_THREADS", std::to_string(rows + 5).c_str(), 1);
    const std::vector<uint8_t> parallel = materialize(source, ref);
    ::unsetenv("QWEN_LOAD_THREADS");
    const std::vector<uint8_t> by_default = materialize(source, ref);

    check(serial == want, what + ": the trimmed unfold must equal the whole-row one");
    check(parallel == serial, what + ": rows split across workers must not move a bit");
    check(by_default == serial, what + ": the default worker count must not move a bit");

    if (serial != want) {
        uint64_t differing = 0;
        for (size_t i = 0; i < want.size() && i < serial.size(); ++i) {
            if (want[i] != serial[i]) ++differing;
        }
        std::cout << "[INFO] " << what << " bytes=" << want.size()
                  << " differing=" << differing << "\n";
    } else {
        std::cout << "[INFO] " << what << " bytes=" << want.size() << " identical\n";
    }
}

void run_cases() {
    const uint64_t rows = 5;
    const uint64_t cols = 5120;
    const std::string canonical =
        pocket::qwen_canonical_tensor_name("blk.0.ffn_down.weight");
    check(canonical != "blk.0.ffn_down.weight",
          "the test tensor must be one the GGUF name table knows both ways");
    check(pocket::qwen_gguf_tensor_name(canonical) == "blk.0.ffn_down.weight",
          "the GGUF spelling must round trip through the canonical one");

    Random random;
    const uint64_t packs = cols / kPackWeights;
    std::vector<uint8_t> payload(static_cast<size_t>(rows * packs * kPackBytes), 0);
    for (uint64_t pack = 0; pack < rows * packs; ++pack) {
        uint8_t* block = payload.data() + pack * kPackBytes;
        for (uint64_t byte = 0; byte < 26; ++byte) block[byte] = random.byte();
        const float scale = kScales[pack % (sizeof(kScales) / sizeof(kScales[0]))];
        const uint16_t bits = pocket::qwen_float_to_fp16_bits(scale);
        std::memcpy(block + 26, &bits, sizeof(bits));
    }

    // The checkpoint's own geometry: a 1024-wide block, a TP4 split of a 5120
    // column axis. Every rank takes part of a block, and the four spans the
    // ranks transform are disjoint but for the block each shares with its
    // neighbour -- 5 of `down_proj`'s 17 blocks at the full width.
    const std::vector<int64_t> wide = [&]() {
        std::vector<int64_t> signs(cols);
        for (uint64_t i = 0; i < cols; ++i) signs[i] = (random.byte() & 1) ? 1 : -1;
        return signs;
    }();
    // The sign vector is per width and shared by every case below, so the
    // narrower block sizes are given their own file rather than another vector.

    {
        const std::string path = "/tmp/pocket_qwen_hadamard_unfold_1024.gguf";
        const std::string file = write_hadamard_gguf(path, 1024, rows, cols, wide, payload);
        pocket::QwenGgufSource source(file);
        for (uint64_t rank = 0; rank < 4; ++rank) {
            check_shard(source, payload, rows, cols, 1024, wide, rank * 1280, 1280);
        }
        // A shard that starts on a block boundary and ends inside one: the span
        // is then the two blocks it touches, and the shard's offset within it is
        // zero, which is the one case where the two offsets differ.
        check_shard(source, payload, rows, cols, 1024, wide, 1024, 1536);
        std::remove(file.c_str());
    }

    // A block below the packing width, so the span has to be rounded to 128 as
    // well as to the block. `96` is not a multiple of the 64-wide block, which
    // is what puts the shard inside one.
    {
        std::vector<int64_t> narrow(cols);
        for (uint64_t i = 0; i < cols; ++i) narrow[i] = (random.byte() & 1) ? 1 : -1;
        const std::string path = "/tmp/pocket_qwen_hadamard_unfold_64.gguf";
        const std::string file = write_hadamard_gguf(path, 64, rows, cols, narrow, payload);
        pocket::QwenGgufSource source(file);
        check_shard(source, payload, rows, cols, 64, narrow, 96, 1024);
        std::remove(file.c_str());
    }
}

}  // namespace

int main() {
    if (pocket::qwen_backend_reads_packed_ternary()) {
        std::cout << "[SKIP] test_qwen_hadamard_unfold: this backend reads the ternary "
                     "pack in place, so it never unfolds a weight\n";
        return 0;
    }
    run_cases();
    if (failures != 0) {
        std::cout << "[FAIL] test_qwen_hadamard_unfold (" << failures << " failures)\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_hadamard_unfold\n";
    return 0;
}
