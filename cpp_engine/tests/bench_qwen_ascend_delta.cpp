// The gated-delta recurrence on its own, at the TP4 prefill shape.
//
// This phase is 54% of prefill wall time (gated_delta 2.02 s of 3.75 s at 4433
// tokens, 1181 TPS), and it is the only phase whose cost is a serial chain of
// vector instructions rather than a Cube matmul. Measuring it through a full TP4
// engine run means a rebuild plus a four-minute prefill per data point, which is
// too slow to iterate on. This benchmark calls the same entry point the engine
// calls, with the same shapes, and reports the per-layer cost directly.
//
//   ./tests/bench_qwen_ascend_delta [--device N] [--tokens 4433] [--heads 12]
//                                   [--key-heads 4] [--iters 3]
//
// The engine runs this kernel once per linear-attention layer, over the whole
// chunk of `tokens` rows, with `heads` value heads split across the cores this
// rank owns. Per-layer milliseconds is therefore what the engine adds per layer;
// multiply by the number of linear layers for the prefill contribution. The
// scaling arm (`--heads`) is the interesting one: at TP4 `heads` is 12, while the
// part has 30 cores, so 18 of them are idle and this prints the saturation
// evidence for whether that matters.

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;

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

// Byte-typed device buffers. The recurrence mixes fp32 (state, normalized q/k)
// and fp16 (v, gates, out), so the element width is a property of the call site
// rather than of the buffer.
class DeviceBuffer {
public:
    DeviceBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes == 0) return;
        if (!pocket::device_malloc_into(ptr_, bytes)) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, bytes)) {
            throw std::runtime_error("device_memset failed");
        }
    }
    DeviceBuffer(size_t bytes, const void* host) : DeviceBuffer(bytes) {
        if (bytes != 0 && !pocket::memcpy_h2d(ptr_, host, bytes)) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* raw() const { return ptr_; }
    float* f32() const { return reinterpret_cast<float*>(ptr_); }
    uint16_t* f16() const { return reinterpret_cast<uint16_t*>(ptr_); }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

std::vector<float> random_floats(size_t count, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(count);
    for (float& value : out) value = dist(rng);
    return out;
}

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng, float lo, float hi) {
    const std::vector<float> source = random_floats(count, rng, lo, hi);
    std::vector<uint16_t> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = float_to_half(source[i]);
    return out;
}

struct Args {
    int device = 0;
    int tokens = 4433;
    int heads = 12;
    int key_heads = 4;
    int iters = 3;
    // Per-head key and value widths. The engine only ever uses 128 and 128; the
    // flags exist so the body of the recurrence can be repriced at other widths,
    // which is how the per-instruction and per-repeat costs are separated.
    int key_dim = kKeyDim;
    int value_dim = kValueDim;
};

// Time one launch shape. The state is loaded once and the kernel is run `iters`
// times against it; that is deliberate. The engine's own cost is one layer's
// call, and the recurrence is numerically contractive (`g` is negative, so the
// decay is < 1), so repeated calls neither overflow nor drift into a denormal
// regime that would price the kernel differently from the real one.
double time_kernel(const DeviceBuffer& state, const DeviceBuffer& q,
                   const DeviceBuffer& k, const DeviceBuffer& v,
                   const DeviceBuffer& g, const DeviceBuffer& beta,
                   DeviceBuffer& out, const Args& args) {
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(args.key_dim));
    auto launch = [&]() {
        return pocket::qwen_gated_delta_sequence_normalized_f16(
            state.f32(), q.f32(), k.f32(), v.f16(), g.f16(), beta.f16(),
            out.f16(), args.tokens, args.heads, args.key_heads, args.key_dim,
            args.value_dim, q_scale);
    };
    if (!launch()) throw std::runtime_error("warmup launch failed");
    if (!pocket::device_synchronize()) throw std::runtime_error("warmup sync failed");

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < args.iters; ++i) {
        if (!launch()) throw std::runtime_error("launch failed");
        if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count() / args.iters;
}

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(2);
            }
            return std::stoi(argv[++i]);
        };
        if (arg == "--device") args.device = next();
        else if (arg == "--tokens") args.tokens = next();
        else if (arg == "--heads") args.heads = next();
        else if (arg == "--key-heads") args.key_heads = next();
        else if (arg == "--iters") args.iters = next();
        else if (arg == "--key-dim") args.key_dim = next();
        else if (arg == "--value-dim") args.value_dim = next();
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            std::exit(2);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(args.device)) {
        std::printf("[SKIP] device_set failed for device %d\n", args.device);
        return 0;
    }

    const size_t tokens = static_cast<size_t>(args.tokens);
    const size_t heads = static_cast<size_t>(args.heads);
    const size_t key_heads = static_cast<size_t>(args.key_heads);
    const size_t key_dim = static_cast<size_t>(key_heads) * args.key_dim;
    const size_t value_dim = heads * args.value_dim;

    std::mt19937 rng(20260916u);
    // Normalized q/k are unit-norm rows, which is what the engine hands the
    // kernel after `qwen_normalize_gated_delta_qk_f16`; v is fp16 with the
    // magnitude a projection output has.
    const std::vector<float> q_host = random_floats(tokens * key_dim, rng, -1.0f, 1.0f);
    const std::vector<float> k_host = random_floats(tokens * key_dim, rng, -1.0f, 1.0f);
    const std::vector<uint16_t> v_host = random_halves(tokens * value_dim, rng, -1.0f, 1.0f);
    // Negative gates only: decay = exp(g) < 1 keeps the recurrence contractive,
    // so the repeated iterations below do not diverge.
    const std::vector<uint16_t> g_host = random_halves(tokens * heads, rng, -2.0f, -0.01f);
    const std::vector<uint16_t> beta_host = random_halves(tokens * heads, rng, 0.0f, 1.0f);

    const DeviceBuffer state(heads * args.key_dim * args.value_dim * sizeof(float));
    const DeviceBuffer q(tokens * key_dim * sizeof(float), q_host.data());
    const DeviceBuffer k(tokens * key_dim * sizeof(float), k_host.data());
    const DeviceBuffer v(tokens * value_dim * sizeof(uint16_t), v_host.data());
    const DeviceBuffer g(tokens * heads * sizeof(uint16_t), g_host.data());
    const DeviceBuffer beta(tokens * heads * sizeof(uint16_t), beta_host.data());
    DeviceBuffer out(tokens * value_dim * sizeof(uint16_t));

    std::printf("device=%d tokens=%d heads=%d key_heads=%d key_dim=%d value_dim=%d iters=%d\n",
                args.device, args.tokens, args.heads, args.key_heads, args.key_dim,
                args.value_dim, args.iters);

    const double seconds = time_kernel(state, q, k, v, g, beta, out, args);
    // Per-token-per-head vector repeats, counted from Step() in
    // qwen_gated_delta_f16.cpp at the 128x128 geometry. Reported so the measured
    // milliseconds can be read against the part's issue rate instead of against
    // a feeling about whether the number looks big.
    const double key_dim_d = args.key_dim;
    const double value_dim_d = args.value_dim;
    // decay + kv_mem accumulate + rank-1 + out accumulate, each a full pass over
    // the [key_dim, value_dim] tile, plus the five value-wide row ops (the two
    // zeroing Duplicates, the three delta ops and the final q_scale Muls).
    const double repeats_per_step =
        4.0 * (key_dim_d * value_dim_d / 64.0) + 5.0 * (value_dim_d / 64.0);
    const double steps = static_cast<double>(args.tokens) * static_cast<double>(args.heads);
    const double repeats = steps * repeats_per_step;
    std::printf("gated_delta_sequence  %.3f ms/layer  (%.2f us per token-head)\n",
                seconds * 1.0e3,
                seconds * 1.0e6 / steps);
    std::printf("  %.3e vector repeats, %.2f G repeats/s\n", repeats,
                repeats / seconds / 1.0e9);
    return 0;
}
