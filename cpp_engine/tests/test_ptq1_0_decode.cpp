// The PTQ1_0 decode, against a second transcription of the packing.
//
// The Ascend backend has no sub-byte weight kernel, so a ternary weight is decoded
// to fp16 on the host at materialization and the dense kernels see an ordinary
// matrix. For this checkpoint that is 6.7 billion weights a rank and it is the
// whole of the Ascend load time, so it is worth making cheap -- but only as long
// as it stays a decode. A trit is -1, 0 or 1 and a block scale is a finite fp16,
// so every block holds exactly three distinct values; the shipped decoder narrows
// those three once and looks them up per weight. What that buys is bit-identity
// with the per-weight form, and bit-identity is the entire claim. This pins it.
//
// The reference below walks a block one weight at a time, by index, the way the
// format is defined; the shipped decoder runs position-outer over the same three
// runs. A mistake in either one's idea of which byte answers which weight is then
// a disagreement between two structures rather than a shared assumption, which is
// the only thing that makes a test of a decoder worth having.
//
// Synthetic blocks, so there is no checkpoint, no device and no vendor SDK: this
// builds and runs on any host.

#include "qwen_weights.hpp"

#include <cstdint>
#include <cstring>
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

constexpr uint64_t kBlockWeights = 128;
constexpr uint64_t kBlockBytes = 28;

// One weight of a block, from the format rather than from the shipped decoder.
//
// Five trits share a byte, most significant first, so the sixth through the
// twentieth weight of a block come from the same bytes as the first five with the
// byte multiplied by a power of three. The multiplication wraps in `uint8_t` and
// that wrap is part of the format: the stored value is a rounded-up division, and
// only reading it back modulo 256 recovers the trit. The twenty-four bytes of `qs`
// are read in runs of sixteen and eight rather than as one run of thirty-two, and
// the last eight weights live in `qh`, interleaved by parity.
uint16_t reference_weight(const uint8_t* block, uint64_t index, float scale) {
    static const uint8_t kPow3[5] = {1, 3, 9, 27, 81};
    uint64_t byte_index = 0;
    int position = 0;
    if (index < 80) {
        byte_index = index % 16;
        position = static_cast<int>(index / 16);
    } else if (index < 120) {
        byte_index = 16 + (index - 80) % 8;
        position = static_cast<int>((index - 80) / 8);
    } else {
        byte_index = 24 + (index - 120) % 2;
        position = static_cast<int>((index - 120) / 2);
    }
    const uint8_t q = static_cast<uint8_t>(block[byte_index] * kPow3[position]);
    const int trit = static_cast<int>((q * 3) >> 8) - 1;
    return pocket::qwen_float_to_fp16_bits(static_cast<float>(trit) * scale);
}

// A block's worth of scales. All of them are exactly representable in fp16, which
// is what lets the test write them back with the narrowing it already has instead
// of carrying an fp16 -> float converter of its own.
//
// The zero and the negative zero are the pair that a table built once per
// checkpoint rather than once per block gets wrong: `0 * scale` carries the sign
// of the scale, and the block reader's kernels store that sign.
const float kScales[] = {
    0.0625f,      // an ordinary positive scale
    -0.0625f,     // and an ordinary negative one
    1.0f,
    -1.0f,
    0.0f,         // `0 * 0` is +0
    -0.0f,        // `0 * -0` is -0, and so is `1 * -0`
    65504.0f,     // the largest finite fp16
    -65504.0f,
    5.9604645e-08f,   // the smallest positive subnormal
    -5.9604645e-08f,
    6.1035156e-05f,   // the smallest positive normal
};

// Deterministic, and the same sequence on every host.
struct Random {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    uint8_t byte() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint8_t>(state >> 33);
    }
    float scale() {
        return kScales[byte() % (sizeof(kScales) / sizeof(kScales[0]))];
    }
};

void write_scale(uint8_t* block, float scale) {
    const uint16_t bits = pocket::qwen_float_to_fp16_bits(scale);
    std::memcpy(block + 26, &bits, sizeof(bits));
}

// One run of `blocks` blocks through the shipped decoder and through the
// reference, compared as fp16 bit patterns rather than as numbers: a decode that
// is off by one ulp is a different matrix, not a rounding difference.
void compare_run(const std::vector<uint8_t>& src, const std::vector<float>& scales,
                 const std::string& what) {
    const uint64_t blocks = src.size() / kBlockBytes;
    std::vector<uint8_t> decoded(src.size() / kBlockBytes * kBlockWeights * 2, 0);
    pocket::qwen_decode_ptq1_0(src.data(), decoded.data(), blocks * kBlockWeights,
                               "test_ptq1_0_decode");

    const uint16_t* got = reinterpret_cast<const uint16_t*>(decoded.data());
    uint64_t compared = 0;
    for (uint64_t block = 0; block < blocks; ++block) {
        const uint8_t* packed = src.data() + block * kBlockBytes;
        for (uint64_t index = 0; index < kBlockWeights; ++index) {
            const uint16_t want = reference_weight(packed, index, scales[block]);
            if (got[block * kBlockWeights + index] != want) {
                if (failures < 12) {
                    std::cout << "[FAIL] " << what << " block=" << block
                              << " weight=" << index << " got=0x" << std::hex
                              << got[block * kBlockWeights + index] << " want=0x" << want
                              << std::dec << "\n";
                }
                ++failures;
            }
            ++compared;
        }
    }
    std::cout << "[INFO] " << what << " weights=" << compared
              << " blocks=" << blocks << "\n";
}

// Every byte value, in every byte position of the block at once. A block filled
// with one value exercises each of the packing's three runs against all 256
// inputs, which is the smallest set that covers the trit arithmetic completely.
void check_exhaustive_bytes() {
    std::vector<uint8_t> src(kBlockBytes * 256, 0);
    std::vector<float> scales(256, 0.0f);
    for (int value = 0; value < 256; ++value) {
        uint8_t* block = src.data() + static_cast<size_t>(value) * kBlockBytes;
        std::memset(block, value, kBlockBytes);
        scales[value] = kScales[value % (sizeof(kScales) / sizeof(kScales[0]))];
        write_scale(block, scales[value]);
    }
    compare_run(src, scales, "every byte value");
}

// Random trits under every scale that can change the answer's sign or magnitude
// in a way a table could get wrong.
void check_scales() {
    Random random;
    std::vector<uint8_t> src;
    std::vector<float> scales;
    for (size_t i = 0; i < sizeof(kScales) / sizeof(kScales[0]); ++i) {
        std::vector<uint8_t> block(kBlockBytes, 0);
        for (uint64_t byte = 0; byte < 26; ++byte) block[byte] = random.byte();
        write_scale(block.data(), kScales[i]);
        src.insert(src.end(), block.begin(), block.end());
        scales.push_back(kScales[i]);
    }
    compare_run(src, scales, "one block a scale");
}

// Several blocks in one call, so the block stride and the run arithmetic are
// covered rather than assumed.
void check_multi_block_run() {
    Random random;
    const uint64_t blocks = 37;
    std::vector<uint8_t> src(blocks * kBlockBytes, 0);
    std::vector<float> scales(blocks, 0.0f);
    for (uint64_t block = 0; block < blocks; ++block) {
        uint8_t* packed = src.data() + block * kBlockBytes;
        for (uint64_t byte = 0; byte < 26; ++byte) packed[byte] = random.byte();
        scales[block] = random.scale();
        write_scale(packed, scales[block]);
    }
    compare_run(src, scales, "a run of blocks");
}

// The one input the decoder has to refuse rather than round.
void check_partial_block_is_refused() {
    std::vector<uint8_t> src(kBlockBytes * 2, 0);
    std::vector<uint8_t> dst(kBlockWeights * 2 * 2, 0);
    bool threw = false;
    try {
        pocket::qwen_decode_ptq1_0(src.data(), dst.data(), kBlockWeights + 1, "partial");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a run that stops inside a block is refused");
}

}  // namespace

int main() {
    check_exhaustive_bytes();
    check_scales();
    check_multi_block_run();
    check_partial_block_is_refused();
    if (failures != 0) {
        std::cout << "[FAIL] test_ptq1_0_decode (" << failures << " failures)\n";
        return 1;
    }
    std::cout << "[PASS] test_ptq1_0_decode\n";
    return 0;
}
