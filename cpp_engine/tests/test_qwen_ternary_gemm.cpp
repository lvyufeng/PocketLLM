// PTQ1_0 dense GEMM in the native engine, against a CPU reference.
//
// The test is built so that its tolerances can be tight rather than comfortable.
// The kernel quantizes the activation to int8 with a per-32-block scale of
// amax/127, which is the only inexact step in the whole path: the weights are
// three-valued and their block scale is fp16. So the activations in most of the
// cases here are *integers* in [-127, 127] with one element of every block forced
// to 127 -- the amax is then exactly 127 and the scale exactly 1.0, which makes
// that quantization lossless. What is left to compare is fp32 accumulation and the
// fp16 output: a few ulps, not a percent.
//
// The packing itself is not taken on faith either. The blocks are built by an
// encoder that is the exact inverse of the decoder the kernel implements, and the
// encoder is brute-forced byte by byte, so a byte only ever exists here if the
// kernel's own walk reads the trits back out of it.
//
// Two cases then exercise what the exact ones cannot. Gaussian activations make
// the int8 quantization visible, and are held to the size of the error it is
// allowed to make rather than to a percentage. And when the released checkpoint is
// on disk, a 512-row slice of a real tensor is compared against a decoder written
// from the packing directly, so the bytes are the file's and not this file's.

#include "gguf_reader.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("[FAIL] %s\n", what.c_str());
    ++failures;
}

void check(bool condition, const std::string& what) {
    if (!condition) fail(what);
}

constexpr int kQK = 128;
constexpr int kBlockBytes = 28;
constexpr int kQhOffset = 24;
constexpr int kScaleOffset = 26;

// ---------------------------------------------------------------------------
// The packing, both directions.
// ---------------------------------------------------------------------------

// One base-3 digit: the byte is tripled, the digit is the byte that carries out
// of the low eight bits, and the low byte carries into the next stage. The
// kernel does this to four bytes at once in 16-bit lanes -- the multiply and the
// mask are the same two steps -- and both the encoder and the decoder below walk
// it one byte at a time. Anything else is a different packing.
inline int next_digit(uint8_t& value) {
    const uint32_t product = static_cast<uint32_t>(value) * 3u;
    value = static_cast<uint8_t>(product & 0xFFu);
    return static_cast<int>(product >> 8);
}

// A byte whose five digits are `digits`. Found by search rather than by inverting
// the walk, which is the point: a byte is usable only if the decode below reads
// the intended trits back out of it.
uint8_t encode_digits(const int digits[5]) {
    for (int candidate = 0; candidate < 256; ++candidate) {
        uint8_t value = static_cast<uint8_t>(candidate);
        bool matches = true;
        for (int t = 0; t < 5 && matches; ++t) {
            matches = next_digit(value) == digits[t];
        }
        if (matches) return static_cast<uint8_t>(candidate);
    }
    throw std::runtime_error("no byte encodes those five digits");
}

void encode_trits(const std::vector<int8_t>& trits, float scale,
                  std::vector<uint8_t>& out) {
    for (size_t block = 0; block < trits.size() / kQK; ++block) {
        const int8_t* w = trits.data() + block * kQK;
        uint8_t bytes[kBlockBytes] = {0};
        // Weights 0..79: bytes 0..15, byte (4g + j) carrying weights
        // {t*16 + 4g + j : t = 0..4}.
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < 4; ++j) {
                int digits[5];
                for (int t = 0; t < 5; ++t) digits[t] = w[t * 16 + 4 * g + j] + 1;
                bytes[4 * g + j] = encode_digits(digits);
            }
        }
        // Weights 80..119: bytes 16..23, byte (16 + 4g + j) carrying
        // {80 + t*8 + 4g + j : t = 0..4}.
        for (int g = 0; g < 2; ++g) {
            for (int j = 0; j < 4; ++j) {
                int digits[5];
                for (int t = 0; t < 5; ++t) digits[t] = w[80 + t * 8 + 4 * g + j] + 1;
                bytes[16 + 4 * g + j] = encode_digits(digits);
            }
        }
        // Weights 120..127: qh, four digits per byte, parity interleaved. The
        // fifth digit is never read, so it is free -- the search below only has
        // to match the four.
        for (int parity = 0; parity < 2; ++parity) {
            int digits[4];
            for (int t = 0; t < 4; ++t) digits[t] = w[120 + 2 * t + parity] + 1;
            bool found = false;
            for (int candidate = 0; candidate < 256 && !found; ++candidate) {
                uint8_t value = static_cast<uint8_t>(candidate);
                bool matches = true;
                for (int t = 0; t < 4 && matches; ++t) {
                    matches = next_digit(value) == digits[t];
                }
                if (matches) {
                    bytes[kQhOffset + parity] = static_cast<uint8_t>(candidate);
                    found = true;
                }
            }
            if (!found) throw std::runtime_error("no byte encodes those four digits");
        }

        const uint16_t scale_bits = __half_as_ushort(__float2half(scale));
        bytes[kScaleOffset + 0] = static_cast<uint8_t>(scale_bits & 0xFF);
        bytes[kScaleOffset + 1] = static_cast<uint8_t>(scale_bits >> 8);
        out.insert(out.end(), bytes, bytes + kBlockBytes);
    }
}

// The decoder, written from the same walk the kernel's dot uses. This is the
// reference for the real-weights case, where the bytes are the file's.
void decode_block(const uint8_t* bytes, float* weights_out) {
    // Weights 0..79.
    for (int g = 0; g < 4; ++g) {
        for (int j = 0; j < 4; ++j) {
            uint8_t value = bytes[4 * g + j];
            for (int t = 0; t < 5; ++t) {
                weights_out[t * 16 + 4 * g + j] = static_cast<float>(next_digit(value) - 1);
            }
        }
    }
    // Weights 80..119.
    for (int g = 0; g < 2; ++g) {
        for (int j = 0; j < 4; ++j) {
            uint8_t value = bytes[16 + 4 * g + j];
            for (int t = 0; t < 5; ++t) {
                weights_out[80 + t * 8 + 4 * g + j] =
                    static_cast<float>(next_digit(value) - 1);
            }
        }
    }
    // Weights 120..127.
    for (int parity = 0; parity < 2; ++parity) {
        uint8_t value = bytes[kQhOffset + parity];
        for (int t = 0; t < 4; ++t) {
            weights_out[120 + 2 * t + parity] = static_cast<float>(next_digit(value) - 1);
        }
    }
    const uint16_t scale_bits = static_cast<uint16_t>(bytes[kScaleOffset]) |
                                (static_cast<uint16_t>(bytes[kScaleOffset + 1]) << 8);
    const float scale = __half2float(__ushort_as_half(scale_bits));
    for (int i = 0; i < kQK; ++i) weights_out[i] *= scale;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;
    ~DeviceBuffer() {
        if (data != nullptr) cudaFree(data);
    }
    bool resize(size_t n) {
        if (n <= count) return true;
        if (data != nullptr) cudaFree(data);
        data = nullptr;
        count = 0;
        if (cudaMalloc(&data, n * sizeof(T)) != cudaSuccess) {
            data = nullptr;
            return false;
        }
        count = n;
        return true;
    }
};

bool to_device(const std::vector<uint16_t>& host, DeviceBuffer<uint16_t>& device) {
    if (!device.resize(host.size())) return false;
    return cudaMemcpy(device.data, host.data(), host.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool to_device(const std::vector<uint8_t>& host, DeviceBuffer<uint8_t>& device) {
    if (!device.resize(host.size())) return false;
    return cudaMemcpy(device.data, host.data(), host.size() * sizeof(uint8_t),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool from_device(const DeviceBuffer<uint16_t>& device, std::vector<uint16_t>& host) {
    host.resize(device.count);
    return cudaMemcpy(host.data(), device.data, host.size() * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

float from_half(uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

uint16_t to_half(float value) { return __half_as_ushort(__float2half(value)); }

// The ops report the launch error, but a kernel that faults *inside* the stream
// surfaces on the next CUDA call, which would be another case's. Draining the
// stream after each case keeps the report attached to what caused it.
void report_stream_error(const char* stage) {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fail(std::string("the ") + stage + " case left the device in an error state: " +
             cudaGetErrorString(status));
    }
}

// Integer activations in [-127, 127] whose *amax is exactly 127 in every 32-wide
// quantization block*, which takes forcing: the largest of 32 draws from that
// range is usually well short of 127, and the kernel's per-block scale would then
// be amax/127 rather than 1.0 and it would round the activations off. With the
// scale pinned at exactly 1.0 the int8 step is lossless and the comparison above
// is down to fp32 accumulation and the fp16 output.
void fill_exact_activation(std::vector<uint16_t>& x, int rows, int k,
                           std::uniform_int_distribution<int>& draw, std::mt19937& rng) {
    for (uint16_t& value : x) value = to_half(static_cast<float>(draw(rng)));
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < k / 32; ++block) {
            x[static_cast<size_t>(row) * k + static_cast<size_t>(block) * 32] = to_half(127.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// Ternary weights on a power-of-two scale, activations that are small integers:
// the kernel's activation quantization is then exact and only fp32 accumulation
// and the fp16 output stand between the two results.
void check_exact_case(int rows, int n, int k, uint32_t seed, bool decode_path) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::uniform_int_distribution<int> activation(-127, 127);

    const int bpr = k / kQK;
    std::vector<int8_t> trits(static_cast<size_t>(n) * k);
    for (int8_t& value : trits) value = static_cast<int8_t>(trit(rng));

    // A power-of-two scale is exact in fp16, so the reference below sees exactly
    // the weights the kernel does.
    std::vector<std::vector<float>> block_scales(n);
    std::vector<uint8_t> blocks;
    blocks.reserve(static_cast<size_t>(n) * bpr * kBlockBytes);
    for (int row = 0; row < n; ++row) {
        block_scales[row].resize(bpr);
        for (int b = 0; b < bpr; ++b) {
            const float scale = std::ldexp(1.0f, -(b % 5) - 3);  // 2^-3 .. 2^-7
            block_scales[row][b] = scale;
            std::vector<int8_t> block_trits(trits.begin() + (size_t) row * k + (size_t) b * kQK,
                                            trits.begin() + (size_t) row * k + (size_t) (b + 1) * kQK);
            encode_trits(block_trits, scale, blocks);
        }
    }

    std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
    fill_exact_activation(x, rows, k, activation, rng);

    DeviceBuffer<uint8_t> d_blocks;
    DeviceBuffer<uint16_t> d_x;
    DeviceBuffer<uint16_t> d_y;
    if (!to_device(blocks, d_blocks) || !to_device(x, d_x) ||
        !d_y.resize(static_cast<size_t>(rows) * n)) {
        fail("device allocation failed");
        return;
    }

    const bool ok = decode_path
        ? pocket::qwen_ptq1_0_matvec_f16_cuda(d_x.data, d_blocks.data, d_y.data, n, k, nullptr)
        : pocket::qwen_ptq1_0_matmul_rows_f16_cuda(d_x.data, d_blocks.data, d_y.data, rows, n, k, k, n,
                                           nullptr);
    if (!ok) {
        fail("the ternary GEMM returned false for rows=" + std::to_string(rows) +
             " n=" + std::to_string(n) + " k=" + std::to_string(k));
        return;
    }

    std::vector<uint16_t> y;
    if (!from_device(d_y, y)) {
        fail("download failed");
        return;
    }

    // Reference: the dequantized weights against the int8-exact activation, in
    // double. Weight values are the decoded trits times the block scale, decoded
    // here rather than reused, so the encoder is checked through the decoder.
    double worst = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < n; ++col) {
            double reference = 0.0;
            for (int b = 0; b < bpr; ++b) {
                float decoded[kQK];
                decode_block(blocks.data() + ((size_t) col * bpr + b) * kBlockBytes, decoded);
                for (int i = 0; i < kQK; ++i) {
                    reference += static_cast<double>(decoded[i]) *
                                 from_half(x[(size_t) row * k + (size_t) b * kQK + i]);
                }
            }
            const double actual = from_half(y[(size_t) row * n + col]);
            const double scale = std::max(1.0, std::fabs(reference));
            worst = std::max(worst, std::fabs(actual - reference) / scale);
        }
    }
    // fp16 has ~10 bits of mantissa; summing k/128 block partials in fp32 and
    // rounding once at the end stays within a few of them.
    check(worst < 2e-3, "rows=" + std::to_string(rows) + " n=" + std::to_string(n) +
                            " k=" + std::to_string(k) + " worst relative error " +
                            std::to_string(worst));
}

// The realistic case: Gaussian activations, so the int8 quantization is the only
// error the kernel introduces and it is now visible.
//
// A relative tolerance would be the wrong instrument here. The output cells are
// sums of k signed terms, so some of them land near zero by cancellation, and a
// fixed relative bound over a matrix of them is a statement about the smallest
// cell rather than about the arithmetic. What is measurable instead is the *size*
// of the error the kernel is allowed to make: every activation is rounded to an
// int8 with a per-32-block step of amax/127, so each one is off by at most half a
// step, and the whole cell is then off by at most
//
//     sum over 32-blocks of half the step times the sum of |w| in that block.
//
// That bound is computed here from the same amax the kernel uses, and the kernel
// has to come in under it. It is a real property rather than a fitted number:
// scaling the activation by the wrong step, or pairing it with the wrong block,
// lands well outside it.
void check_gaussian_case(int rows, int n, int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> trit(-1, 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const int bpr = k / kQK;
    const int blocks32 = k / 32;
    std::vector<int8_t> trits(static_cast<size_t>(n) * k);
    for (int8_t& value : trits) value = static_cast<int8_t>(trit(rng));

    std::vector<uint8_t> blocks;
    for (int row = 0; row < n; ++row) {
        for (int b = 0; b < bpr; ++b) {
            std::vector<int8_t> block_trits(trits.begin() + (size_t) row * k + (size_t) b * kQK,
                                            trits.begin() + (size_t) row * k + (size_t) (b + 1) * kQK);
            encode_trits(block_trits, 0.03f, blocks);
        }
    }

    std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
    std::vector<float> x_f32(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_f32[i] = normal(rng);
        x[i] = to_half(x_f32[i]);
        x_f32[i] = from_half(x[i]);  // the reference sees what the kernel sees
    }

    // The quantization step the kernel will derive, per 32-value block of each
    // activation row: amax/127, with the same amax over the same rounded values.
    std::vector<float> step(static_cast<size_t>(rows) * blocks32);
    for (int row = 0; row < rows; ++row) {
        for (int s = 0; s < blocks32; ++s) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(x_f32[(size_t) row * k + (size_t) s * 32 + i]));
            }
            step[(size_t) row * blocks32 + s] = amax / 127.0f;
        }
    }

    DeviceBuffer<uint8_t> d_blocks;
    DeviceBuffer<uint16_t> d_x;
    DeviceBuffer<uint16_t> d_y;
    if (!to_device(blocks, d_blocks) || !to_device(x, d_x) ||
        !d_y.resize(static_cast<size_t>(rows) * n)) {
        fail("device allocation failed");
        return;
    }
    if (!pocket::qwen_ptq1_0_matmul_rows_f16_cuda(d_x.data, d_blocks.data, d_y.data, rows, n, k, k,
                                                  n, nullptr)) {
        fail("the ternary GEMM returned false for the Gaussian case");
        return;
    }
    std::vector<uint16_t> y;
    if (!from_device(d_y, y)) {
        fail("download failed");
        return;
    }

    double worst_absolute = 0.0;
    double worst_ratio = 0.0;
    double worst_excess = 0.0;
    double largest = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < n; ++col) {
            double reference = 0.0;
            double bound = 0.0;
            for (int b = 0; b < bpr; ++b) {
                float decoded[kQK];
                decode_block(blocks.data() + ((size_t) col * bpr + b) * kBlockBytes, decoded);
                for (int i = 0; i < kQK; ++i) {
                    reference += static_cast<double>(decoded[i]) *
                                 static_cast<double>(x_f32[(size_t) row * k +
                                                           (size_t) b * kQK + i]);
                }
            }
            for (int s = 0; s < blocks32; ++s) {
                double weight_sum = 0.0;
                const int block = s / 4;
                const int within = s % 4;
                float decoded[kQK];
                decode_block(blocks.data() + ((size_t) col * bpr + block) * kBlockBytes, decoded);
                for (int i = 0; i < 32; ++i) {
                    weight_sum += std::fabs(decoded[within * 32 + i]);
                }
                bound += 0.5 * static_cast<double>(step[(size_t) row * blocks32 + s]) * weight_sum;
            }
            const double actual = from_half(y[(size_t) row * n + col]);
            const double error = std::fabs(actual - reference);
            largest = std::max(largest, std::fabs(reference));
            worst_absolute = std::max(worst_absolute, error);
            worst_excess = std::max(worst_excess, error - bound);
            if (bound > 0.0) worst_ratio = std::max(worst_ratio, error / bound);
        }
    }

    // Past the quantization bound the only error left is the fp32 accumulation
    // and the one rounding to fp16 at the end, which are a few ulps of the
    // largest cell -- so that, and not a percentage, is the slack allowed here.
    const double slack = 2e-3 * std::max(1.0, largest);
    check(worst_excess <= slack, "Gaussian rows=" + std::to_string(rows) + " overshoots the " +
                                     "quantization bound by " + std::to_string(worst_excess) +
                                     " against a slack of " + std::to_string(slack));
    std::printf("[INFO] Gaussian rows=%d n=%d k=%d error/bound=%g (absolute %g, largest cell %g)\n",
                rows, n, k, worst_ratio, worst_absolute, largest);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// 512 rows of the real tensor, read out of the checkpoint's own 28-byte blocks.
void check_real_weights(const std::string& path) {
    pocket::GGUFFile gguf(path);
    const pocket::GGUFTensorInfo* info = gguf.find_tensor("blk.0.ffn_gate.weight");
    if (info == nullptr) {
        fail("the checkpoint has no blk.0.ffn_gate.weight");
        return;
    }
    if (info->ggml_type != 143) {
        fail("blk.0.ffn_gate.weight is not PTQ1_0");
        return;
    }
    const int k = static_cast<int>(info->shape[0]);
    const int n = 512;  // rows of the 17,408 the tensor has
    const int bpr = k / kQK;
    if (k % kQK != 0 || info->shape[1] < static_cast<uint64_t>(n)) {
        fail("unexpected ffn_gate.weight shape");
        return;
    }
    const pocket::TensorView view = gguf.tensor_view(*info);
    std::vector<uint8_t> blocks(view.data, view.data + (size_t) n * bpr * kBlockBytes);

    // 64 rows of activations take the 64-wide tile, and 512 weight rows is a
    // multiple of the tile height, so this case runs the widest weight tile with
    // no clamping at all -- the synthetic cases above cover the clamped ones.
    const int rows = 64;
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> activation(-127, 127);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
    fill_exact_activation(x, rows, k, activation, rng);

    DeviceBuffer<uint8_t> d_blocks;
    DeviceBuffer<uint16_t> d_x;
    DeviceBuffer<uint16_t> d_y;
    if (!to_device(blocks, d_blocks) || !to_device(x, d_x) ||
        !d_y.resize(static_cast<size_t>(rows) * n)) {
        fail("device allocation failed");
        return;
    }
    if (!pocket::qwen_ptq1_0_matmul_rows_f16_cuda(d_x.data, d_blocks.data, d_y.data, rows, n, k, k,
                                                  n, nullptr)) {
        fail("the ternary GEMM returned false on real weights");
        return;
    }
    std::vector<uint16_t> y;
    if (!from_device(d_y, y)) {
        fail("download failed");
        return;
    }

    double worst = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < n; ++col) {
            double reference = 0.0;
            for (int b = 0; b < bpr; ++b) {
                float decoded[kQK];
                decode_block(blocks.data() + ((size_t) col * bpr + b) * kBlockBytes, decoded);
                for (int i = 0; i < kQK; ++i) {
                    reference += static_cast<double>(decoded[i]) *
                                 from_half(x[(size_t) row * k + (size_t) b * kQK + i]);
                }
            }
            const double actual = from_half(y[(size_t) row * n + col]);
            worst = std::max(worst, std::fabs(actual - reference) / std::max(1.0, std::fabs(reference)));
        }
    }
    check(worst < 2e-3, "real weights worst relative error " + std::to_string(worst));
    std::printf("[INFO] blk.0.ffn_gate.weight rows=%d n=%d k=%d worst=%g\n", rows, n, k, worst);
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("[SKIP] no CUDA device\n");
        return 0;
    }

    try {
        // The tile widths the prefill launcher may pick, plus the non-divisible
        // rows that fall back to the narrowest one. 48 and 96 are in the list on
        // purpose: they are the row counts an activation-tile width chosen from
        // divisibility alone gets wrong.
        for (int rows : {8, 16, 32, 48, 64, 96, 112, 128, 256}) {
            check_exact_case(rows, 37, 256, 4000u + (uint32_t) rows, false);
        }
        report_stream_error("prefill");
        // Decode, over a row count wide enough to take the K split (the split is
        // taken when n * split < 65536) and one narrow enough to skip it.
        check_exact_case(1, 64, 256, 4100u, true);
        check_exact_case(1, 4096, 512, 4101u, true);
        check_exact_case(1, 17408, 5120, 4102u, true);
        // The ops check the launch error and clear it, but a kernel that faults
        // inside the stream reports on the *next* call -- which would be somebody
        // else's case. This makes that report belong to the case that caused it.
        report_stream_error("decode");

        check_gaussian_case(16, 64, 512, 4200u);
        report_stream_error("gaussian");

        const std::string checkpoint =
            "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
        if (file_exists(checkpoint)) {
            check_real_weights(checkpoint);
            report_stream_error("real weights");
        } else {
            std::printf("[SKIP] the released checkpoint is not on disk\n");
        }
    } catch (const std::exception& ex) {
        fail(std::string("exception: ") + ex.what());
    }

    if (failures != 0) {
        std::printf("test_qwen_ternary_gemm failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_ternary_gemm\n");
    return 0;
}
