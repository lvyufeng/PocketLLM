// Times the gated-delta recurrence on its own, at the shape the model actually
// runs it.
//
// The full-model profile reports this op as one lump per layer, and the layer is
// 64% of the 4096-token prefill's device time, so a tuning loop that needs a 44 s
// weight load before it can print a number is not usable. This runs the operator
// through the same launcher at the same shape and prints milliseconds directly.
//
// The Qwen3.8 TP4 shard is 48 value heads over 16 key heads at key_dim =
// value_dim = 128, i.e. three value heads per key head. The launcher caps the grid
// at 30 blocks (first-generation 910 has 30 AI cores) and the kernel hands each
// block one (head, value slice) pair, so at heads=12 the 24 pairs are one round and
// the wall is set by a single pair -- the reported per-head-token figure divides by
// the heads' worth of work the busiest block walked. Treat it as "the cost of the
// block doing the most work", not an average.
//
//   ./tests/bench_qwen_ascend_gated_delta [--device N] [--rows 4096]
//                                         [--iters 20] [--heads 48]
//                                         [--key-heads 16] [--mode seq|step]
//                                         [--check]
//
// --check runs the double-precision host reference from test_qwen_ascend_group_b
// over the same buffers and reports the worst relative error. It is off by default
// because at 4096 rows the reference is ~10 GFLOP of scalar double arithmetic on
// the host.

#include "device_runtime.hpp"
#include "qwen_gated_delta_geometry.hpp"
#include "qwen_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa == 0) return 0.0f;
        // Subnormal: renormalize into an FP32 exponent.
        uint32_t m = mantissa;
        int shift = 0;
        while ((m & 0x400u) == 0) {
            m <<= 1;
            ++shift;
        }
        bits |= static_cast<uint32_t>(127 - 15 - shift + 1) << 23;
        bits |= (m & 0x3ffu) << 13;
    } else if (exponent == 31) {
        bits |= 0x7f800000u | (mantissa << 13);
    } else {
        bits |= (exponent + 127 - 15) << 23;
        bits |= mantissa << 13;
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count == 0) return;
        if (!pocket::device_malloc_into(ptr_, count * sizeof(T))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, count * sizeof(T))) {
            throw std::runtime_error("device_memset failed");
        }
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        if (count_ != 0 &&
            !pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ != 0 &&
            !pocket::memcpy_d2h(host.data(), ptr_, count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_d2h failed");
        }
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

std::vector<float> random_floats(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> out(count);
    for (float& value : out) value = dist(rng);
    return out;
}

// L2-normalize every row of a [rows, heads, kKeyDim] plane, matching the
// rsqrt(sum + 1e-6) the GPU path uses.
std::vector<float> normalize_rows(const std::vector<float>& source, int rows,
                                  int heads) {
    std::vector<float> out(source.size());
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < heads; ++h) {
            const size_t base = (static_cast<size_t>(r) * heads + h) * kKeyDim;
            double sum = 0.0;
            for (int i = 0; i < kKeyDim; ++i) {
                sum += static_cast<double>(source[base + i]) * source[base + i];
            }
            const float inv = static_cast<float>(1.0 / std::sqrt(sum + 1.0e-6));
            for (int i = 0; i < kKeyDim; ++i) out[base + i] = source[base + i] * inv;
        }
    }
    return out;
}

// The reference from test_qwen_ascend_group_b.cpp, restricted to q/k that are
// already normalized, so the bench can check the path the engine actually takes.
struct Reference {
    std::vector<double> output;
    std::vector<double> state;
};

Reference recurrence_reference(const std::vector<float>& initial_state,
                               const std::vector<float>& q,
                               const std::vector<float>& k,
                               const std::vector<uint16_t>& v,
                               const std::vector<uint16_t>& g,
                               const std::vector<uint16_t>& beta, int rows,
                               int heads, int key_heads, double q_scale) {
    const size_t state_count =
        static_cast<size_t>(heads) * kKeyDim * kValueDim;
    std::vector<double> state(initial_state.begin(), initial_state.end());
    std::vector<double> out(static_cast<size_t>(rows) * heads * kValueDim, 0.0);
    std::vector<double> memory(kValueDim);
    std::vector<double> delta(kValueDim);

    const int repeat = heads / key_heads;
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / repeat;
        const size_t state_base = static_cast<size_t>(head) * kKeyDim * kValueDim;
        for (int token = 0; token < rows; ++token) {
            const size_t key_base =
                (static_cast<size_t>(token) * key_heads + key_head) * kKeyDim;
            const size_t value_base =
                (static_cast<size_t>(token) * heads + head) * kValueDim;
            const double decay =
                std::exp(static_cast<double>(half_to_float(g[token * heads + head])));
            const double gain = half_to_float(beta[token * heads + head]);
            for (int i = 0; i < kKeyDim; ++i) {
                for (int d = 0; d < kValueDim; ++d) {
                    state[state_base + i * kValueDim + d] *= decay;
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + i * kValueDim + d] *
                           static_cast<double>(k[key_base + i]);
                }
                memory[d] = sum;
                delta[d] = (half_to_float(v[value_base + d]) - memory[d]) * gain;
            }
            for (int i = 0; i < kKeyDim; ++i) {
                for (int d = 0; d < kValueDim; ++d) {
                    state[state_base + i * kValueDim + d] +=
                        static_cast<double>(k[key_base + i]) * delta[d];
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + i * kValueDim + d] *
                           static_cast<double>(q[key_base + i]);
                }
                out[value_base + d] = sum * q_scale;
            }
        }
    }
    return {out, state};
}

struct Buffers {
    std::vector<float> q_source;
    std::vector<float> k_source;
    std::vector<float> state_host;
    std::vector<uint16_t> v;
    std::vector<uint16_t> g;
    std::vector<uint16_t> beta;
    std::vector<float> q_normalized;
    std::vector<float> k_normalized;
};

Buffers make_buffers(int rows, int heads, int key_heads, std::mt19937& rng) {
    Buffers buffers;
    const size_t key_count = static_cast<size_t>(rows) * key_heads * kKeyDim;
    const size_t value_count = static_cast<size_t>(rows) * heads * kValueDim;
    const size_t state_count = static_cast<size_t>(heads) * kKeyDim * kValueDim;
    buffers.q_source = random_floats(key_count, rng, 0.6f);
    buffers.k_source = random_floats(key_count, rng, 0.6f);
    buffers.v = random_halves(value_count, rng, 0.35f);
    buffers.g = random_halves(static_cast<size_t>(rows) * heads, rng, 0.06f);
    buffers.beta.resize(static_cast<size_t>(rows) * heads);
    std::uniform_real_distribution<float> beta_dist(0.15f, 0.75f);
    for (uint16_t& value : buffers.beta) value = float_to_half(beta_dist(rng));
    buffers.state_host = random_floats(state_count, rng, 0.002f);
    buffers.q_normalized = normalize_rows(buffers.q_source, rows, key_heads);
    buffers.k_normalized = normalize_rows(buffers.k_source, rows, key_heads);
    return buffers;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The state is carried across tokens and is part of the op's contract, so a timed
// loop has to start every iteration from the same state or the numbers are not
// comparable between runs. Resetting in place would not work: aclrtMemcpy is a
// blocking copy that is not stream-ordered, so it would race the kernel still
// reading the previous state. Instead each iteration owns a state buffer, all
// initialized from the same host image, and the loop cycles through them.
double time_sequence(std::vector<DeviceBuffer<float>*>& states,
                     float* q, float* k, uint16_t* v, uint16_t* g, uint16_t* beta,
                     uint16_t* out, int rows, int heads, int key_heads,
                     float q_scale, int iters) {
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const double started = now_ms();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_gated_delta_sequence_normalized_shared_f16(
                states[i]->get(), q, k, v, g, beta, out, rows, heads, key_heads,
                kKeyDim, kValueDim, q_scale)) {
            throw std::runtime_error("gated delta launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    return (now_ms() - started) / iters;
}

double time_step(std::vector<DeviceBuffer<float>*>& states, uint16_t* q,
                 uint16_t* k, uint16_t* v, uint16_t* g, uint16_t* beta,
                 uint16_t* out, int heads, int key_heads, float q_scale,
                 int iters) {
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const double started = now_ms();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_gated_delta_step_f16(states[i]->get(), q, k, v, g, beta,
                                               out, heads, key_heads, kKeyDim,
                                               kValueDim, q_scale)) {
            throw std::runtime_error("gated delta step launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    return (now_ms() - started) / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int rows = 4096;
    int iters = 20;
    int heads = 48;
    int key_heads = 16;
    float q_scale = 0.125f;
    bool check = false;
    std::string mode = "seq";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--heads" && i + 1 < argc) heads = std::atoi(argv[++i]);
        else if (arg == "--key-heads" && i + 1 < argc) key_heads = std::atoi(argv[++i]);
        else if (arg == "--q-scale" && i + 1 < argc) q_scale = std::atof(argv[++i]);
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--check") check = true;
        else {
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
    if (heads <= 0 || key_heads <= 0 || heads % key_heads != 0 || rows <= 0 ||
        iters <= 0) {
        std::fprintf(stderr, "invalid geometry\n");
        return 2;
    }

    std::printf(
        "gated_delta mode=%s rows=%d heads=%d key_heads=%d key_dim=%d value_dim=%d "
        "iters=%d device=%d\n",
        mode.c_str(), rows, heads, key_heads, kKeyDim, kValueDim, iters, device);

    if (mode != "seq" && mode != "step") {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }
    const int tokens = (mode == "seq") ? rows : 1;
    const size_t out_count =
        static_cast<size_t>(tokens) * heads * kValueDim;

    std::mt19937 rng(20260914u + static_cast<unsigned>(rows));
    const Buffers buffers = make_buffers(rows, heads, key_heads, rng);

    DeviceBuffer<uint16_t> d_out(out_count);
    // One state per timed iteration so the loop never resets a buffer the device
    // may still be reading.
    std::vector<DeviceBuffer<float>*> states;
    for (int i = 0; i < iters; ++i) {
        states.push_back(new DeviceBuffer<float>(buffers.state_host));
    }
    double ms = 0.0;
    if (mode == "seq") {
        DeviceBuffer<uint16_t> d_v(buffers.v), d_g(buffers.g),
            d_beta(buffers.beta);
        DeviceBuffer<float> d_qn(buffers.q_normalized), d_kn(buffers.k_normalized);
        ms = time_sequence(states, d_qn.get(), d_kn.get(), d_v.get(), d_g.get(),
                           d_beta.get(), d_out.get(), rows, heads, key_heads,
                           q_scale, iters);
    } else {
        const size_t one_key = static_cast<size_t>(key_heads) * kKeyDim;
        const size_t one_value = static_cast<size_t>(heads) * kValueDim;
        const std::vector<uint16_t> hq = random_halves(one_key, rng, 0.6f);
        const std::vector<uint16_t> hk = random_halves(one_key, rng, 0.6f);
        const std::vector<uint16_t> hv(buffers.v.begin(),
                                       buffers.v.begin() + one_value);
        const std::vector<uint16_t> hg(buffers.g.begin(),
                                       buffers.g.begin() + heads);
        const std::vector<uint16_t> hb(buffers.beta.begin(),
                                       buffers.beta.begin() + heads);
        DeviceBuffer<uint16_t> d_q(hq), d_k(hk), d_v1(hv), d_g1(hg), d_b1(hb);
        ms = time_step(states, d_q.get(), d_k.get(), d_v1.get(), d_g1.get(),
                       d_b1.get(), d_out.get(), heads, key_heads, q_scale, iters);
    }

    const double per_token_us = ms * 1000.0 / tokens;
    // The kernel distributes one (head, value slice) pair per core iteration and the
    // launcher caps the grid at 30 cores, so two things set the wall: how many items
    // each core walks, and how wide one item is. `us_per_token` is the whole grid's
    // wall for one token; `us_per_head_token` normalises it by how many heads' worth
    // of work the busiest core did, which is what makes the number comparable across
    // geometries (and across the unsliced kernel, where one core walked one head).
    const int slices = static_cast<int>(pocket::gated_delta::kSlices);
    const int items = heads * slices;
    const int blocks = std::min<int>(pocket::gated_delta::kMaxBlocks, items);
    const int items_per_block = (items + blocks - 1) / blocks;
    const int heads_per_block = (items_per_block + slices - 1) / slices;
    std::printf(
        "seconds=%.6f  tokens_per_s=%.3f  us_per_token=%.3f  "
        "slices=%d  items_per_block=%d  heads_per_block=%d  "
        "us_per_head_token=%.3f\n",
        ms / 1000.0, tokens / (ms / 1000.0), per_token_us, slices,
        items_per_block, heads_per_block, per_token_us / heads_per_block);
    std::fflush(stdout);

    if (check && mode != "seq") {
        std::printf("check needs --mode seq; skipped\n");
    } else if (check) {
        std::vector<float> host_state = states[0]->download();
        const std::vector<uint16_t> host_out = d_out.download();
        const Reference reference =
            recurrence_reference(buffers.state_host, buffers.q_normalized,
                                 buffers.k_normalized, buffers.v, buffers.g,
                                 buffers.beta, tokens, heads, key_heads, q_scale);
        double worst = 0.0;
        int mismatches = 0;
        for (size_t i = 0; i < reference.output.size(); ++i) {
            const double actual = half_to_float(host_out[i]);
            const double error = std::fabs(actual - reference.output[i]);
            const double scale = std::max(std::fabs(reference.output[i]), 2.5e-3);
            worst = std::max(worst, error / scale);
            if (error > 2.5e-3 + 7.0e-3 * std::fabs(reference.output[i])) ++mismatches;
        }
        double worst_state = 0.0;
        int state_mismatches = 0;
        for (size_t i = 0; i < reference.state.size(); ++i) {
            const double error =
                std::fabs(static_cast<double>(host_state[i]) - reference.state[i]);
            const double scale = std::max(std::fabs(reference.state[i]), 8.0e-6);
            worst_state = std::max(worst_state, error / scale);
            if (error > 8.0e-6 + 8.0e-4 * std::fabs(reference.state[i])) {
                ++state_mismatches;
            }
        }
        std::printf(
            "check output: worst_relative=%.6e mismatches=%d/%zu  "
            "state: worst_relative=%.6e mismatches=%d/%zu\n",
            worst, mismatches, reference.output.size(), worst_state,
            state_mismatches, reference.state.size());
        std::fflush(stdout);
    }
    return 0;
}
