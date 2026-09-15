// Microbenchmark for the GQA attention operators at the real TP4 shard shape.
//
// The full model loop costs a 44 s weight load before it reaches attention, which
// makes it unusable as a tuning loop. This runs the same operator through the same
// neutral launcher at the same shape and reports throughput directly.
//
//   ./tests/bench_qwen_ascend_attention [--device N] [--kind prefill|decode]
//                                       [--lengths 1024,4096] [--iters 5]
//
// The Qwen3.8 TP4 shard of full attention is 6 query heads over 1 KV head at
// head_dim 256: the model has 24 query heads over 4 KV heads and TP4 splits the
// heads without splitting head_dim.

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count == 0) return;
        if (!pocket::device_malloc_into(ptr_, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_memset failed");
        }
    }
    explicit DeviceBuffer(const std::vector<uint16_t>& host) : DeviceBuffer(host.size()) {
        if (!pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(uint16_t))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint16_t* get() const { return ptr_; }

private:
    uint16_t* ptr_ = nullptr;
    size_t count_ = 0;
};

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

// One timed launch: the launch plus a device synchronize, which is the only thing
// that makes the measurement mean anything on a queued device.
double time_prefill(uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* out, int rows,
                    int position_offset, int max_context, int iters) {
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_gqa_prefill_attention_f16(q, k, v, out, rows, kQHeads,
                                                    kKvHeads, kHeadDim,
                                                    position_offset, max_context)) {
            throw std::runtime_error("prefill launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count() / iters;
}

double time_decode(uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* out, float* scratch,
                   int context_len, int max_context, int iters) {
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_gqa_decode_attention_f16(q, k, v, out, scratch, kQHeads,
                                                   kKvHeads, kHeadDim, context_len,
                                                   max_context)) {
            throw std::runtime_error("decode launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count() / iters;
}

double time_decode_flashdec(uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* out,
                            float* partials, int context_len, int max_context,
                            int partitions, int iters) {
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_gqa_decode_attention_flashdec_f16(
                q, k, v, out, partials, kQHeads, kKvHeads, kHeadDim, context_len,
                max_context, partitions)) {
            throw std::runtime_error("flashdec decode launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 5;
    std::string kind = "prefill";
    std::vector<int> lengths = {1024, 2048, 4096};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--kind" && i + 1 < argc) {
            kind = argv[++i];
        } else if (arg == "--lengths" && i + 1 < argc) {
            lengths.clear();
            std::string rest = argv[++i];
            size_t at = 0;
            while (at <= rest.size()) {
                const size_t comma = rest.find(',', at);
                const std::string piece =
                    rest.substr(at, comma == std::string::npos ? std::string::npos
                                                               : comma - at);
                if (!piece.empty()) lengths.push_back(std::stoi(piece));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    std::printf("kind=%s q_heads=%d kv_heads=%d head_dim=%d iters=%d device=%d\n",
                kind.c_str(), kQHeads, kKvHeads, kHeadDim, iters, device);

    for (const int length : lengths) {
        std::mt19937 rng(97531u + static_cast<unsigned>(length));
        if (kind == "prefill") {
            const int rows = length;
            const size_t q_elements =
                static_cast<size_t>(rows) * kQHeads * kHeadDim;
            const size_t kv_elements =
                static_cast<size_t>(length) * kKvHeads * kHeadDim;
            const std::vector<uint16_t> host_q = random_halves(q_elements, rng, 0.5f);
            const std::vector<uint16_t> host_k = random_halves(kv_elements, rng, 0.5f);
            const std::vector<uint16_t> host_v = random_halves(kv_elements, rng, 0.5f);
            DeviceBuffer q(host_q), k(host_k), v(host_v);
            DeviceBuffer out(q_elements);
            const double seconds =
                time_prefill(q.get(), k.get(), v.get(), out.get(), rows, 0, length, iters);
            std::printf(
                "prefill rows=%d seconds=%.6f tokens_per_s=%.3f\n", rows, seconds,
                static_cast<double>(rows) / seconds);
        } else {
            const size_t kv_elements =
                static_cast<size_t>(length) * kKvHeads * kHeadDim;
            const size_t q_elements =
                static_cast<size_t>(1) * kQHeads * kHeadDim;
            const std::vector<uint16_t> host_q = random_halves(q_elements, rng, 0.5f);
            const std::vector<uint16_t> host_k = random_halves(kv_elements, rng, 0.5f);
            const std::vector<uint16_t> host_v = random_halves(kv_elements, rng, 0.5f);
            DeviceBuffer q(host_q), k(host_k), v(host_v);
            DeviceBuffer out(q_elements);
            if (kind == "decode-flashdec") {
                // The partition count the engine picks at this context, so the
                // microbenchmark measures the same launch shape the model does.
                const int partitions = std::min(30, (length + 255) / 256);
                float* partials = nullptr;
                if (!pocket::device_malloc_into(
                        partials, static_cast<size_t>(kQHeads) * partitions *
                                      (kHeadDim + 2) * sizeof(float))) {
                    throw std::runtime_error("partials malloc failed");
                }
                const double seconds =
                    time_decode_flashdec(q.get(), k.get(), v.get(), out.get(), partials,
                                         length, length, partitions, iters);
                pocket::device_free(partials);
                std::printf("flashdec context=%d partitions=%d seconds=%.6f tokens_per_s=%.3f\n",
                            length, partitions, seconds, 1.0 / seconds);
                std::fflush(stdout);
                continue;
            }
            float* scratch = nullptr;
            if (!pocket::device_malloc_into(
                    scratch,
                    static_cast<size_t>(kQHeads) * length * sizeof(float))) {
                throw std::runtime_error("scratch malloc failed");
            }
            const double seconds =
                time_decode(q.get(), k.get(), v.get(), out.get(), scratch, length,
                            length, iters);
            pocket::device_free(scratch);
            std::printf(
                "decode context=%d seconds=%.6f tokens_per_s=%.3f\n", length, seconds,
                1.0 / seconds);
        }
        std::fflush(stdout);
    }
    return 0;
}
