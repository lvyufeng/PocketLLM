// Where decode time goes, split into host and device.
//
// Decode on this backend is not bandwidth bound: the TP4 shard reads 13.4 GB of
// resident weights per token, which is 11 ms at the measured HBM rate, and the
// model runs at 114 ms per token. The gap is per-op cost, and per-op cost has two
// halves -- the kernel and the host work that precedes it. This measures both at
// the real decode shapes by timing the same op twice:
//
//   enqueue  N launches back to back, one synchronize at the end. If the host is
//            the bottleneck this is the host cost per op; otherwise it is the
//            kernel cost.
//   sync     one launch and one synchronize per op. Always kernel + host.
//
// The two numbers together say which half to attack. They are reported separately
// rather than as a ratio because a host-bound op and a device-bound op can have
// the same `sync` number and need opposite fixes.
//
//   ./tests/bench_qwen_ascend_decode_ops [--device N] [--iters 200]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The TP4 shard of Qwen3.8-27B, read off the checkpoint's config: hidden 5120,
// intermediate 17408 split four ways, and this rank's slice of each projection.
constexpr int kHidden = 5120;
constexpr int kMlp = 17408 / 4;
constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;
constexpr int kLinearKeyHeads = 16 / 4;
constexpr int kLinearValueHeads = 48 / 4;
constexpr int kLinearHeadDim = 128;

// The engine's resident cache for the benchmark prompt. The attention kernels take
// it as a pitch, and the neutral entry rejects contexts that round past it.
constexpr int kMaxContext = 8192;

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

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

struct Timing {
    double enqueue_seconds = 0.0;
    double sync_seconds = 0.0;
};

// The op is a lambda so the two loops below are written once. `enqueue` runs the
// body `iters` times with no synchronize between them.
template <typename Op>
Timing time_op(Op op, int iters) {
    Timing timing;
    if (!op()) throw std::runtime_error("warmup launch failed");
    if (!pocket::device_synchronize()) throw std::runtime_error("warmup sync failed");

    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.enqueue_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
        if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
    }
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
            if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.sync_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
    }
    return timing;
}

void report(const char* name, const Timing& timing) {
    std::printf("%-28s enqueue=%8.1f us  sync=%8.1f us\n", name,
                timing.enqueue_seconds * 1.0e6, timing.sync_seconds * 1.0e6);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
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

    std::mt19937 rng(20260914u);
    std::printf("iters=%d device=%d\n", iters, device);

    // MLP gate and up: [1, 5120] x [5120, 8704]. The two together are the largest
    // pair of weight reads in the layer.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kMlp));
        report("matmul 1x5120x8704",
               time_op([&] {
                   return pocket::qwen_fp16_matmul_rows_f16(x.get(), w.get(), y.get(),
                                                            1, kMlp, kHidden, kHidden,
                                                            kMlp, kHidden);
               }, iters));
    }
    // MLP down: [1, 8704] x [8704, 5120].
    {
        DeviceBuffer x(random_halves(kMlp, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kHidden) * kMlp, rng));
        DeviceBuffer y(static_cast<size_t>(kHidden));
        report("matmul 1x8704x5120",
               time_op([&] {
                   return pocket::qwen_fp16_matmul_rows_f16(x.get(), w.get(), y.get(),
                                                            1, kHidden, kMlp, kMlp,
                                                            kHidden, kMlp);
               }, iters));
    }
    // SwiGLU: two matmuls plus a SiLU and a multiply, all issued from here.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer gate(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer up(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kMlp));
        report("swiglu 1x5120x8704",
               time_op([&] {
                   return pocket::qwen_fp16_swiglu_matmul_rows_f16(
                       x.get(), gate.get(), up.get(), y.get(), 1, kMlp, kHidden,
                       kHidden, kMlp, kHidden);
               }, iters));
    }
    // RMSNorm is small but runs twice per layer, and its host cost does not scale
    // with the row count.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer gamma(random_halves(kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kHidden));
        report("rmsnorm 1x5120",
               time_op([&] {
                   return pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                       x.get(), gamma.get(), y.get(), 1, kHidden, 1e-6f);
               }, iters));
    }
    // The linear-attention projections and the gated-delta step, which the profile
    // shows is the second largest single item in the linear layers.
    {
        const int heads = kLinearValueHeads;
        const int key_heads = kLinearKeyHeads;
        const int dim = kLinearHeadDim;
        const int value_dim = heads * dim;
        const int key_dim = key_heads * dim;
        DeviceBuffer state(static_cast<size_t>(heads) * dim * dim);
        DeviceBuffer q(random_halves(static_cast<size_t>(key_dim), rng));
        DeviceBuffer k(random_halves(static_cast<size_t>(key_dim), rng));
        DeviceBuffer v(random_halves(static_cast<size_t>(value_dim), rng));
        DeviceBuffer g(random_halves(static_cast<size_t>(heads), rng));
        DeviceBuffer beta(random_halves(static_cast<size_t>(heads), rng));
        DeviceBuffer out(static_cast<size_t>(value_dim));
        report("gated_delta_step",
               time_op([&] {
                   return pocket::qwen_gated_delta_step_f16(
                       reinterpret_cast<float*>(const_cast<uint16_t*>(state.get())),
                       q.get(), k.get(), v.get(), g.get(), beta.get(), out.get(),
                       heads, key_heads, dim, dim, 1.0f);
               }, iters));
    }
    // Decode attention at the context the engine actually runs it at. Two
    // candidate kernels are timed side by side because they parallelize on
    // different axes: the neutral entry uses the Cube unit but, at one query row,
    // has only kv_heads items to spread over cores, while FlashDecoding splits the
    // context and gives every core a piece. Which one wins is a question about
    // this part's core count, not about arithmetic.
    for (int context : {2048, 4097, 8192}) {
        DeviceBuffer q(random_halves(static_cast<size_t>(kQHeads) * kHeadDim, rng));
        DeviceBuffer k(random_halves(static_cast<size_t>(context) * kKvHeads * kHeadDim, rng));
        DeviceBuffer v(random_halves(static_cast<size_t>(context) * kKvHeads * kHeadDim, rng));
        DeviceBuffer out(static_cast<size_t>(kQHeads) * kHeadDim);
        float* scores = nullptr;
        if (!pocket::device_malloc_into(
                scores, static_cast<size_t>(kQHeads) * context * sizeof(float))) {
            throw std::runtime_error("scratch malloc failed");
        }
        char label[64];
        std::snprintf(label, sizeof(label), "decode attn %d cube", context);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_f16(
                       q.get(), k.get(), v.get(), out.get(), scores, kQHeads,
                       kKvHeads, kHeadDim, context, kMaxContext);
               }, iters));

        // Same split geometry the engine's long-context branch uses: one
        // partition per 256 tokens, capped at the core count.
        const int partitions = std::min(30, (context + 255) / 256);
        float* partials = nullptr;
        if (!pocket::device_malloc_into(
                partials, static_cast<size_t>(kQHeads) * partitions *
                              (kHeadDim + 2) * sizeof(float))) {
            throw std::runtime_error("partials malloc failed");
        }
        std::snprintf(label, sizeof(label), "decode attn %d flashdec(%d)", context,
                      partitions);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_flashdec_f16(
                       q.get(), k.get(), v.get(), out.get(), partials, kQHeads,
                       kKvHeads, kHeadDim, context, kMaxContext, partitions);
               }, iters));

        // The Cube decode kernel with the context split across cores. `partitions
        // = 0` asks for the widest split the context allows, which is what the
        // engine would use in production; the third timing pins the count so the
        // scaling against core count is visible rather than inferred.
        std::snprintf(label, sizeof(label), "decode attn %d cube-split(auto)", context);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_cube_split_f16(
                       q.get(), k.get(), v.get(), out.get(), kQHeads, kKvHeads,
                       kHeadDim, context, kMaxContext, 0);
               }, iters));
        if (partitions > 1) {
            std::snprintf(label, sizeof(label), "decode attn %d cube-split(%d)",
                          context, partitions);
            report(label,
                   time_op([&] {
                       return pocket::qwen_gqa_decode_attention_cube_split_f16(
                           q.get(), k.get(), v.get(), out.get(), kQHeads, kKvHeads,
                           kHeadDim, context, kMaxContext, partitions);
                   }, iters));
        }
        pocket::device_free(partials);
        pocket::device_free(scores);
    }

    return 0;
}
