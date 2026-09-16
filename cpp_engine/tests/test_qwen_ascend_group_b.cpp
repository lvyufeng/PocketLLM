// Correctness tests for the hand-written Group B Qwen operators.
//
// Host references follow the operator definitions in double precision rather than
// copying either vendor implementation. The calls use qwen_ops.hpp's neutral names,
// so this also verifies that the engine-facing dispatch seam reaches the Ascend
// launchers.
//
//   ./tests/test_qwen_ascend_group_b [--device N] [--only a,b,c]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;

int failures = 0;
int checks = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "  FAIL " << what << "\n";
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t shifted = mantissa;
            int shift = 0;
            while ((shifted & 0x400u) == 0) {
                shifted <<= 1;
                ++shift;
            }
            bits = sign |
                   ((127u - 15u - static_cast<uint32_t>(shift) + 1u) << 23) |
                   ((shifted & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

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
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(
                sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | half);
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
        upload(host);
    }

    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) {
            throw std::runtime_error("DeviceBuffer upload size mismatch");
        }
        if (count_ != 0 &&
            !pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }

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

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng,
                                    float scale) {
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

void sync_or_throw(const std::string& what) {
    if (!pocket::device_synchronize()) {
        throw std::runtime_error("device_synchronize failed after " + what);
    }
}

struct ErrorStats {
    double worst = 0.0;
    int mismatches = 0;
};

ErrorStats half_error(const std::vector<uint16_t>& got,
                      const std::vector<double>& want, double relative,
                      double absolute) {
    if (got.size() != want.size()) {
        return {1.0e30, static_cast<int>(std::max(got.size(), want.size()))};
    }
    ErrorStats stats;
    for (size_t i = 0; i < want.size(); ++i) {
        const double actual = half_to_float(got[i]);
        if (!std::isfinite(actual) || !std::isfinite(want[i])) {
            stats.worst = std::numeric_limits<double>::infinity();
            ++stats.mismatches;
            continue;
        }
        const double error = std::fabs(actual - want[i]);
        const double scale = std::max(std::fabs(want[i]), absolute);
        stats.worst = std::max(stats.worst, error / scale);
        if (error > absolute + relative * std::fabs(want[i])) ++stats.mismatches;
    }
    return stats;
}

ErrorStats float_error(const std::vector<float>& got,
                       const std::vector<double>& want, double relative,
                       double absolute) {
    if (got.size() != want.size()) {
        return {1.0e30, static_cast<int>(std::max(got.size(), want.size()))};
    }
    ErrorStats stats;
    for (size_t i = 0; i < want.size(); ++i) {
        const double actual = static_cast<double>(got[i]);
        if (!std::isfinite(actual) || !std::isfinite(want[i])) {
            stats.worst = std::numeric_limits<double>::infinity();
            ++stats.mismatches;
            continue;
        }
        const double error = std::fabs(actual - want[i]);
        const double scale = std::max(std::fabs(want[i]), absolute);
        stats.worst = std::max(stats.worst, error / scale);
        if (error > absolute + relative * std::fabs(want[i])) ++stats.mismatches;
    }
    return stats;
}

void expect_half_close(const std::vector<uint16_t>& got,
                       const std::vector<double>& want, double relative,
                       double absolute, const std::string& what) {
    const ErrorStats stats = half_error(got, want, relative, absolute);
    std::string detail;
    if (stats.mismatches != 0 && got.size() == want.size()) {
        for (size_t i = 0; i < want.size(); ++i) {
            const double actual = half_to_float(got[i]);
            const double error = std::fabs(actual - want[i]);
            if (error > absolute + relative * std::fabs(want[i])) {
                detail = " first=" + std::to_string(i) +
                         " got=" + std::to_string(actual) +
                         " want=" + std::to_string(want[i]);
                break;
            }
        }
    }
    expect(stats.mismatches == 0,
           what + " mismatches=" + std::to_string(stats.mismatches) +
               " worst_relative=" + std::to_string(stats.worst) + detail);
}

void expect_half_exact(const std::vector<uint16_t>& got,
                       const std::vector<uint16_t>& want,
                       const std::string& what) {
    int mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (got[i] != want[i]) {
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    if (got.size() != want.size()) {
        mismatches += static_cast<int>(std::max(got.size(), want.size()) -
                                       std::min(got.size(), want.size()));
    }
    std::string detail;
    if (mismatches != 0 && first < got.size() && first < want.size()) {
        detail = " first=" + std::to_string(first) +
                 " got=" + std::to_string(half_to_float(got[first])) +
                 " want=" + std::to_string(half_to_float(want[first]));
    }
    expect(mismatches == 0,
           what + " mismatches=" + std::to_string(mismatches) + detail);
}

void expect_float_close(const std::vector<float>& got,
                        const std::vector<double>& want, double relative,
                        double absolute, const std::string& what) {
    const ErrorStats stats = float_error(got, want, relative, absolute);
    std::string detail;
    if (stats.mismatches != 0 && got.size() == want.size()) {
        for (size_t i = 0; i < want.size(); ++i) {
            const double actual = static_cast<double>(got[i]);
            const double error = std::fabs(actual - want[i]);
            if (!std::isfinite(actual) ||
                error > absolute + relative * std::fabs(want[i])) {
                detail = " first=" + std::to_string(i) +
                         " got=" + std::to_string(actual) +
                         " want=" + std::to_string(want[i]);
                break;
            }
        }
    }
    expect(stats.mismatches == 0,
           what + " mismatches=" + std::to_string(stats.mismatches) +
               " worst_relative=" + std::to_string(stats.worst) + detail);
}

std::vector<double> normalize_reference(const std::vector<uint16_t>& source,
                                        int rows, int heads, int dim) {
    std::vector<double> out(source.size());
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            const size_t base = (static_cast<size_t>(row) * heads + head) * dim;
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double value = half_to_float(source[base + d]);
                sum += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sum + 1.0e-6);
            for (int d = 0; d < dim; ++d) {
                out[base + d] = half_to_float(source[base + d]) * inverse;
            }
        }
    }
    return out;
}

void test_normalize_qk(std::mt19937& rng) {
    const int rows = 3;
    const int key_heads = 7;
    const size_t count = static_cast<size_t>(rows) * key_heads * kKeyDim;
    const std::vector<uint16_t> q = random_halves(count, rng, 0.7f);
    const std::vector<uint16_t> k = random_halves(count, rng, 0.7f);
    DeviceBuffer<uint16_t> d_q(q);
    DeviceBuffer<uint16_t> d_k(k);
    DeviceBuffer<float> d_qn(count);
    DeviceBuffer<float> d_kn(count);

    expect(pocket::qwen_normalize_gated_delta_qk_f16(
               d_q.get(), d_k.get(), d_qn.get(), d_kn.get(), rows, key_heads,
               kKeyDim),
           "normalize qk launch");
    sync_or_throw("normalize qk");

    expect_float_close(d_qn.download(), normalize_reference(q, rows, key_heads,
                                                             kKeyDim),
                       3.0e-5, 2.0e-6, "normalized q");
    expect_float_close(d_kn.download(), normalize_reference(k, rows, key_heads,
                                                             kKeyDim),
                       3.0e-5, 2.0e-6, "normalized k");
}

struct RecurrenceReference {
    std::vector<double> output;
    std::vector<double> state;
};

RecurrenceReference recurrence_reference(
    const std::vector<float>& initial_state, const std::vector<double>& q,
    const std::vector<double>& k, const std::vector<uint16_t>& v,
    const std::vector<uint16_t>& g, const std::vector<uint16_t>& beta, int rows,
    int heads, int key_heads, double q_scale) {
    std::vector<double> state(initial_state.begin(), initial_state.end());
    std::vector<double> output(static_cast<size_t>(rows) * heads * kValueDim);
    const int repeat = heads / key_heads;
    std::vector<double> memory(kValueDim);
    std::vector<double> delta(kValueDim);
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
                    state[state_base + static_cast<size_t>(i) * kValueDim + d] *= decay;
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + static_cast<size_t>(i) * kValueDim + d] *
                           k[key_base + i];
                }
                memory[d] = sum;
                delta[d] =
                    (half_to_float(v[value_base + d]) - memory[d]) * gain;
            }
            for (int i = 0; i < kKeyDim; ++i) {
                for (int d = 0; d < kValueDim; ++d) {
                    state[state_base + static_cast<size_t>(i) * kValueDim + d] +=
                        k[key_base + i] * delta[d];
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + static_cast<size_t>(i) * kValueDim + d] *
                           q[key_base + i];
                }
                output[value_base + d] = sum * q_scale;
            }
        }
    }
    return {std::move(output), std::move(state)};
}

std::vector<double> normalize_as_double(const std::vector<uint16_t>& source,
                                        int rows, int heads) {
    return normalize_reference(source, rows, heads, kKeyDim);
}

void test_gated_delta_recurrence(std::mt19937& rng) {
    const int rows = 2;
    const int heads = 4;
    const int key_heads = 2;
    const float q_scale = 0.125f;
    const size_t key_count = static_cast<size_t>(rows) * key_heads * kKeyDim;
    const size_t value_count = static_cast<size_t>(rows) * heads * kValueDim;
    const size_t state_count = static_cast<size_t>(heads) * kKeyDim * kValueDim;
    const std::vector<uint16_t> q = random_halves(key_count, rng, 0.6f);
    const std::vector<uint16_t> k = random_halves(key_count, rng, 0.6f);
    const std::vector<uint16_t> v = random_halves(value_count, rng, 0.35f);
    const std::vector<uint16_t> g = random_halves(static_cast<size_t>(rows) * heads,
                                                  rng, 0.06f);
    std::vector<uint16_t> beta(static_cast<size_t>(rows) * heads);
    std::uniform_real_distribution<float> beta_dist(0.15f, 0.75f);
    for (uint16_t& value : beta) value = float_to_half(beta_dist(rng));
    const std::vector<float> initial_state = random_floats(state_count, rng, 0.002f);
    const std::vector<double> qn = normalize_as_double(q, rows, key_heads);
    const std::vector<double> kn = normalize_as_double(k, rows, key_heads);
    const RecurrenceReference reference = recurrence_reference(
        initial_state, qn, kn, v, g, beta, rows, heads, key_heads, q_scale);

    DeviceBuffer<float> d_state(initial_state);
    DeviceBuffer<uint16_t> d_q(q);
    DeviceBuffer<uint16_t> d_k(k);
    DeviceBuffer<uint16_t> d_v(v);
    DeviceBuffer<uint16_t> d_g(g);
    DeviceBuffer<uint16_t> d_beta(beta);
    DeviceBuffer<uint16_t> d_out(value_count);
    expect(pocket::qwen_gated_delta_sequence_f16(
               d_state.get(), d_q.get(), d_k.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "gated delta sequence launch");
    sync_or_throw("gated delta sequence");
    expect_half_close(d_out.download(), reference.output, 7.0e-3, 2.5e-3,
                      "gated delta sequence output");
    expect_float_close(d_state.download(), reference.state, 8.0e-4, 8.0e-6,
                       "gated delta sequence state");

    // Pre-normalized and shared entry points must implement the same recurrence.
    std::vector<float> qn32(qn.begin(), qn.end());
    std::vector<float> kn32(kn.begin(), kn.end());
    DeviceBuffer<float> d_qn(qn32);
    DeviceBuffer<float> d_kn(kn32);
    DeviceBuffer<float> d_norm_state(initial_state);
    DeviceBuffer<float> d_shared_state(initial_state);
    DeviceBuffer<uint16_t> d_norm_out(value_count);
    DeviceBuffer<uint16_t> d_shared_out(value_count);
    expect(pocket::qwen_gated_delta_sequence_normalized_f16(
               d_norm_state.get(), d_qn.get(), d_kn.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_norm_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "normalized gated delta launch");
    expect(pocket::qwen_gated_delta_sequence_normalized_shared_f16(
               d_shared_state.get(), d_qn.get(), d_kn.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_shared_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "shared normalized gated delta launch");
    sync_or_throw("normalized gated delta");
    const std::vector<uint16_t> norm_out = d_norm_out.download();
    const std::vector<uint16_t> shared_out = d_shared_out.download();
    expect(norm_out == shared_out, "normalized and shared outputs are bit-identical");
    expect(d_norm_state.download() == d_shared_state.download(),
           "normalized and shared states are bit-identical");
    expect_half_close(norm_out, reference.output, 7.0e-3, 2.5e-3,
                      "normalized gated delta output");

    // A one-token sequence and the step entry point start from the same state and
    // must finish identically. This is the actual decode geometry.
    const size_t one_key_count = static_cast<size_t>(key_heads) * kKeyDim;
    const size_t one_value_count = static_cast<size_t>(heads) * kValueDim;
    std::vector<uint16_t> q1(q.begin(), q.begin() + one_key_count);
    std::vector<uint16_t> k1(k.begin(), k.begin() + one_key_count);
    std::vector<uint16_t> v1(v.begin(), v.begin() + one_value_count);
    std::vector<uint16_t> g1(g.begin(), g.begin() + heads);
    std::vector<uint16_t> beta1(beta.begin(), beta.begin() + heads);
    DeviceBuffer<uint16_t> d_q1(q1), d_k1(k1), d_v1(v1), d_g1(g1), d_beta1(beta1);
    DeviceBuffer<float> d_step_state(initial_state), d_one_state(initial_state);
    DeviceBuffer<uint16_t> d_step_out(one_value_count), d_one_out(one_value_count);
    expect(pocket::qwen_gated_delta_step_f16(
               d_step_state.get(), d_q1.get(), d_k1.get(), d_v1.get(), d_g1.get(),
               d_beta1.get(), d_step_out.get(), heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "gated delta step launch");
    expect(pocket::qwen_gated_delta_sequence_f16(
               d_one_state.get(), d_q1.get(), d_k1.get(), d_v1.get(), d_g1.get(),
               d_beta1.get(), d_one_out.get(), 1, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "one-row gated delta sequence launch");
    sync_or_throw("gated delta step parity");
    expect(d_step_out.download() == d_one_out.download(),
           "step and one-row sequence outputs are bit-identical");
    expect(d_step_state.download() == d_one_state.download(),
           "step and one-row sequence states are bit-identical");
}

void test_linear_attn_gates(std::mt19937& rng) {
    const int rows = 9;
    const int heads = 7;
    const size_t count = static_cast<size_t>(rows) * heads;
    std::vector<uint16_t> a = random_halves(count, rng, 4.0f);
    const std::vector<uint16_t> b = random_halves(count, rng, 7.0f);
    const std::vector<uint16_t> a_log = random_halves(heads, rng, 1.5f);
    const std::vector<uint16_t> dt_bias = random_halves(heads, rng, 2.0f);
    // Exercise the overflow-safe softplus branch explicitly.
    a[3] = float_to_half(100.0f);
    a[19] = float_to_half(40.0f);
    DeviceBuffer<uint16_t> d_a(a), d_b(b), d_a_log(a_log), d_dt(dt_bias);
    DeviceBuffer<uint16_t> d_g(count), d_beta(count);
    expect(pocket::qwen_linear_attn_gates_f16(
               d_a.get(), d_b.get(), d_a_log.get(), d_dt.get(), d_g.get(),
               d_beta.get(), rows, heads),
           "linear attention gates launch");
    sync_or_throw("linear attention gates");

    std::vector<double> want_g(count), want_beta(count);
    for (size_t i = 0; i < count; ++i) {
        const int head = static_cast<int>(i % heads);
        const double x = static_cast<double>(half_to_float(a[i])) +
                         half_to_float(dt_bias[head]);
        const double softplus = x > 0.0 ? x + std::log1p(std::exp(-x))
                                        : std::log1p(std::exp(x));
        want_g[i] = -std::exp(static_cast<double>(half_to_float(a_log[head]))) *
                    softplus;
        const double bv = half_to_float(b[i]);
        want_beta[i] = bv >= 0.0 ? 1.0 / (1.0 + std::exp(-bv))
                                 : std::exp(bv) / (1.0 + std::exp(bv));
    }
    expect_half_close(d_g.download(), want_g, 2.5e-3, 2.0e-3,
                      "linear attention g");
    // Hardware Reciprocal is intentionally FP16-grade here.
    expect_half_close(d_beta.download(), want_beta, 4.0e-3, 1.5e-3,
                      "linear attention beta");
}

std::vector<double> conv_reference(const std::vector<uint16_t>& x,
                                   const std::vector<uint16_t>& weight,
                                   const std::vector<uint16_t>* tail, int seq_len,
                                   int channels, int kernel) {
    const int tail_len = kernel - 1;
    std::vector<double> out(static_cast<size_t>(seq_len) * channels);
    for (int token = 0; token < seq_len; ++token) {
        for (int channel = 0; channel < channels; ++channel) {
            double sum = 0.0;
            for (int tap = 0; tap < kernel; ++tap) {
                const int source = token - tail_len + tap;
                const double value = source >= 0
                    ? half_to_float(x[static_cast<size_t>(source) * channels + channel])
                    : tail == nullptr
                        ? 0.0
                        : half_to_float((*tail)[static_cast<size_t>(source + tail_len) *
                                                      channels + channel]);
                sum += value *
                       half_to_float(weight[static_cast<size_t>(channel) * kernel + tap]);
            }
            out[static_cast<size_t>(token) * channels + channel] =
                sum / (1.0 + std::exp(-sum));
        }
    }
    return out;
}

std::vector<uint16_t> expected_tail(const std::vector<uint16_t>& old_tail,
                                    const std::vector<uint16_t>& x, int seq_len,
                                    int channels, int kernel) {
    const int tail_len = kernel - 1;
    std::vector<uint16_t> history = old_tail;
    history.insert(history.end(), x.begin(), x.end());
    std::vector<uint16_t> out(static_cast<size_t>(tail_len) * channels);
    const int available_rows = tail_len + seq_len;
    const int first = available_rows - tail_len;
    std::copy(history.begin() + static_cast<size_t>(first) * channels,
              history.begin() + static_cast<size_t>(first + tail_len) * channels,
              out.begin());
    return out;
}

// The Ascend upload stores the convolution weight tap-major so the kernel can load
// one tap's row over the channels as a single contiguous run. Every reference below
// reads the checkpoint's channel-major layout, so a test that builds its own weight
// has to transpose it the same way before handing it to the op.
std::vector<uint16_t> transpose_conv_weight(const std::vector<uint16_t>& weight,
                                            int channels, int kernel) {
    std::vector<uint16_t> out(weight.size());
    for (int channel = 0; channel < channels; ++channel) {
        for (int tap = 0; tap < kernel; ++tap) {
            out[static_cast<size_t>(tap) * channels + channel] =
                weight[static_cast<size_t>(channel) * kernel + tap];
        }
    }
    return out;
}

void test_causal_conv(std::mt19937& rng) {
    const int seq_len = 5;
    const int channels = 517;
    const int kernel = 4;
    const int tail_len = kernel - 1;
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(seq_len) * channels, rng, 0.5f);
    const std::vector<uint16_t> weight =
        random_halves(static_cast<size_t>(channels) * kernel, rng, 0.35f);
    const std::vector<uint16_t> tail =
        random_halves(static_cast<size_t>(tail_len) * channels, rng, 0.4f);
    DeviceBuffer<uint16_t> d_x(x), d_tail(tail);
    DeviceBuffer<uint16_t> d_weight(transpose_conv_weight(weight, channels, kernel));
    DeviceBuffer<uint16_t> d_y(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), d_tail.get(), d_y.get(), seq_len,
               channels, kernel, true),
           "causal conv launch");
    sync_or_throw("causal conv");
    expect_half_close(d_y.download(),
                      conv_reference(x, weight, &tail, seq_len, channels, kernel),
                      6.0e-3, 2.5e-3, "causal conv output");
    expect(d_tail.download() ==
               expected_tail(tail, x, seq_len, channels, kernel),
           "causal conv updates carried tail");

    // update_tail=false must leave history untouched.
    DeviceBuffer<uint16_t> d_tail_unchanged(tail);
    DeviceBuffer<uint16_t> d_y2(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), d_tail_unchanged.get(), d_y2.get(),
               seq_len, channels, kernel, false),
           "causal conv no-tail-update launch");
    sync_or_throw("causal conv no-tail-update");
    expect(d_tail_unchanged.download() == tail,
           "causal conv leaves tail unchanged when requested");

    // Prefill followed by one decode token must equal one fused convolution.
    const int prefix = seq_len - 1;
    std::vector<uint16_t> prefix_x(x.begin(), x.begin() +
                                             static_cast<size_t>(prefix) * channels);
    std::vector<uint16_t> last_x(x.begin() + static_cast<size_t>(prefix) * channels,
                                 x.end());
    DeviceBuffer<uint16_t> d_prefix(prefix_x), d_last(last_x), d_chain_tail(tail);
    DeviceBuffer<uint16_t> d_prefix_y(static_cast<size_t>(prefix) * channels);
    DeviceBuffer<uint16_t> d_last_y(channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_prefix.get(), d_weight.get(), d_chain_tail.get(), d_prefix_y.get(),
               prefix, channels, kernel, true),
           "causal conv prefix launch");
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_last.get(), d_weight.get(), d_chain_tail.get(), d_last_y.get(), 1,
               channels, kernel, true),
           "causal conv decode launch");
    sync_or_throw("causal conv continuity");
    std::vector<uint16_t> chained = d_prefix_y.download();
    const std::vector<uint16_t> last = d_last_y.download();
    chained.insert(chained.end(), last.begin(), last.end());
    expect(chained == d_y.download(), "causal conv prefill/decode continuity");

    DeviceBuffer<uint16_t> d_zero_y(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), nullptr, d_zero_y.get(), seq_len,
               channels, kernel, false),
           "causal conv null-tail launch");
    sync_or_throw("causal conv null-tail");
    expect_half_close(d_zero_y.download(),
                      conv_reference(x, weight, nullptr, seq_len, channels, kernel),
                      6.0e-3, 2.5e-3, "causal conv null-tail output");
}

void rope_reference(std::vector<uint16_t>& values, int rows, int heads,
                    int head_dim, int rotary_dim, int start_position,
                    double theta) {
    const int half = rotary_dim / 2;
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < half; ++index) {
            const double frequency =
                std::pow(theta, -2.0 * static_cast<double>(index) / rotary_dim);
            const double angle = (start_position + row) * frequency;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            for (int head = 0; head < heads; ++head) {
                const size_t base =
                    (static_cast<size_t>(row) * heads + head) * head_dim;
                const double a = half_to_float(values[base + index]);
                const double b = half_to_float(values[base + index + half]);
                values[base + index] =
                    float_to_half(static_cast<float>(a * c - b * s));
                values[base + index + half] =
                    float_to_half(static_cast<float>(b * c + a * s));
            }
        }
    }
}

void test_partial_rope(std::mt19937& rng) {
    const int rows = 3;
    const int q_heads = 5;
    const int kv_heads = 2;
    const int head_dim = 80;
    const int rotary_dim = 64;
    const int start_position = 37;
    const float theta = 1000000.0f;
    const std::vector<uint16_t> q = random_halves(
        static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.7f);
    const std::vector<uint16_t> k = random_halves(
        static_cast<size_t>(rows) * kv_heads * head_dim, rng, 0.7f);
    std::vector<uint16_t> want_q = q;
    std::vector<uint16_t> want_k = k;
    rope_reference(want_q, rows, q_heads, head_dim, rotary_dim, start_position,
                   theta);
    rope_reference(want_k, rows, kv_heads, head_dim, rotary_dim, start_position,
                   theta);
    DeviceBuffer<uint16_t> d_q(q), d_k(k);
    expect(pocket::qwen_partial_rope_rows_f16(
               d_q.get(), d_k.get(), start_position, rows, rotary_dim, theta,
               q_heads, kv_heads, head_dim),
           "partial rope launch");
    sync_or_throw("partial rope");
    const std::vector<uint16_t> got_q = d_q.download();
    const std::vector<uint16_t> got_k = d_k.download();
    expect_half_exact(got_q, want_q, "partial rope q matches rounded reference");
    expect_half_exact(got_k, want_k, "partial rope k matches rounded reference");
    int untouched_mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < q_heads; ++head) {
            const size_t base =
                (static_cast<size_t>(row) * q_heads + head) * head_dim;
            for (int d = rotary_dim; d < head_dim; ++d) {
                if (got_q[base + d] != q[base + d]) ++untouched_mismatches;
            }
        }
        for (int head = 0; head < kv_heads; ++head) {
            const size_t base =
                (static_cast<size_t>(row) * kv_heads + head) * head_dim;
            for (int d = rotary_dim; d < head_dim; ++d) {
                if (got_k[base + d] != k[base + d]) ++untouched_mismatches;
            }
        }
    }
    expect(untouched_mismatches == 0,
           "partial rope preserves unrotated dimensions");
}

void test_append_kv(std::mt19937& rng) {
    const int seq_len = 3;
    const int kv_heads = 3;
    const int head_dim = 17;
    const int start_pos = 2;
    const int max_context = 8;
    const size_t rows_count = static_cast<size_t>(seq_len) * kv_heads * head_dim;
    const size_t cache_count =
        static_cast<size_t>(max_context) * kv_heads * head_dim;
    const std::vector<uint16_t> k_rows = random_halves(rows_count, rng, 0.8f);
    const std::vector<uint16_t> v_rows = random_halves(rows_count, rng, 0.8f);
    std::vector<uint16_t> k_cache(cache_count, float_to_half(7.0f));
    std::vector<uint16_t> v_cache(cache_count, float_to_half(-7.0f));
    std::vector<uint16_t> want_k = k_cache;
    std::vector<uint16_t> want_v = v_cache;
    for (int token = 0; token < seq_len; ++token) {
        const size_t src = static_cast<size_t>(token) * kv_heads * head_dim;
        const size_t dst =
            static_cast<size_t>(start_pos + token) * kv_heads * head_dim;
        std::copy(k_rows.begin() + src,
                  k_rows.begin() + src + kv_heads * head_dim,
                  want_k.begin() + dst);
        std::copy(v_rows.begin() + src,
                  v_rows.begin() + src + kv_heads * head_dim,
                  want_v.begin() + dst);
    }
    DeviceBuffer<uint16_t> d_k_rows(k_rows), d_v_rows(v_rows);
    DeviceBuffer<uint16_t> d_k_cache(k_cache), d_v_cache(v_cache);
    expect(pocket::qwen_append_kv_cache_f16(
               d_k_rows.get(), d_v_rows.get(), d_k_cache.get(), d_v_cache.get(),
               seq_len, kv_heads, head_dim, start_pos, max_context),
           "append kv launch");
    sync_or_throw("append kv");
    expect_half_exact(d_k_cache.download(), want_k,
                      "append kv writes K at the offset");
    expect_half_exact(d_v_cache.download(), want_v,
                      "append kv writes V at the offset");
}

std::vector<double> attention_reference(const std::vector<uint16_t>& q,
                                        const std::vector<uint16_t>& k_cache,
                                        const std::vector<uint16_t>& v_cache,
                                        int rows, int q_heads, int kv_heads,
                                        int head_dim, int position_offset) {
    std::vector<double> output(static_cast<size_t>(rows) * q_heads * head_dim);
    const int repeat = q_heads / kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> scores;
    for (int row = 0; row < rows; ++row) {
        const int context_len = position_offset + row + 1;
        scores.resize(context_len);
        for (int head = 0; head < q_heads; ++head) {
            const int kv_head = head / repeat;
            const size_t q_base =
                (static_cast<size_t>(row) * q_heads + head) * head_dim;
            double maximum = -INFINITY;
            for (int pos = 0; pos < context_len; ++pos) {
                const size_t cache_base =
                    (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                double score = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    score += static_cast<double>(half_to_float(q[q_base + d])) *
                             half_to_float(k_cache[cache_base + d]);
                }
                scores[pos] = score * scale;
                maximum = std::max(maximum, scores[pos]);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int d = 0; d < head_dim; ++d) {
                double value = 0.0;
                for (int pos = 0; pos < context_len; ++pos) {
                    const size_t cache_base =
                        (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                    value += scores[pos] * half_to_float(v_cache[cache_base + d]);
                }
                output[q_base + d] = value / denominator;
            }
        }
    }
    return output;
}

void test_gqa_decode_case(std::mt19937& rng, int head_dim, int context_len,
                          int kv_heads = 2) {
    const int q_heads = 6;
    const int max_context = context_len + 4;
    const std::vector<uint16_t> q =
        random_halves(static_cast<size_t>(q_heads) * head_dim, rng, 0.45f);
    std::vector<uint16_t> k_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    std::vector<uint16_t> v_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    const size_t poison_start =
        static_cast<size_t>(context_len) * kv_heads * head_dim;
    std::fill(k_cache.begin() + poison_start, k_cache.end(), float_to_half(40.0f));
    std::fill(v_cache.begin() + poison_start, v_cache.end(), float_to_half(-40.0f));
    DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(q_heads) * head_dim);
    DeviceBuffer<float> d_scores(static_cast<size_t>(q_heads) * context_len);
    expect(pocket::qwen_gqa_decode_attention_f16(
               d_q.get(), d_k.get(), d_v.get(), d_out.get(), d_scores.get(),
               q_heads, kv_heads, head_dim, context_len, max_context),
           "GQA decode launch");
    sync_or_throw("GQA decode");
    expect_half_close(d_out.download(),
                      attention_reference(q, k_cache, v_cache, 1, q_heads,
                                          kv_heads, head_dim, context_len - 1),
                      5.0e-3, 2.0e-3, "GQA decode output");
    const std::vector<float> scores = d_scores.download();
    int bad_probability_rows = 0;
    for (int head = 0; head < q_heads; ++head) {
        double sum = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            const float value = scores[static_cast<size_t>(head) * context_len + pos];
            if (!(value >= 0.0f)) ++bad_probability_rows;
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 2.0e-4) ++bad_probability_rows;
    }
    std::string score_detail;
    if (bad_probability_rows != 0) {
        score_detail = " rows=" + std::to_string(bad_probability_rows) + " sums=";
        for (int head = 0; head < q_heads; ++head) {
            double sum = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                sum += scores[static_cast<size_t>(head) * context_len + pos];
            }
            score_detail += std::to_string(sum) + (head + 1 == q_heads ? "" : ",");
        }
    }
    expect(bad_probability_rows == 0,
           "GQA decode score scratch holds probabilities" + score_detail);
    std::vector<double> want_scores(scores.size());
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int head = 0; head < q_heads; ++head) {
        const int kv_head = head / (q_heads / kv_heads);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int pos = 0; pos < context_len; ++pos) {
            double score = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                score += static_cast<double>(half_to_float(q[head * head_dim + d])) *
                         half_to_float(k_cache[(pos * kv_heads + kv_head) * head_dim + d]);
            }
            want_scores[head * context_len + pos] = score * scale;
            maximum = std::max(maximum, score * scale);
        }
        double denominator = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            double& value = want_scores[head * context_len + pos];
            value = std::exp(value - maximum);
            denominator += value;
        }
        for (int pos = 0; pos < context_len; ++pos) {
            want_scores[head * context_len + pos] /= denominator;
        }
    }
    expect_float_close(scores, want_scores, 5.0e-5, 2.0e-6,
                       "GQA decode normalized probability values");
}

void test_gqa_decode(std::mt19937& rng) {
    test_gqa_decode_case(rng, 32, 5);
    test_gqa_decode_case(rng, 256, 64);
    test_gqa_decode_case(rng, 17, 65);
    test_gqa_decode_case(rng, 64, 133);
    test_gqa_decode_case(rng, 256, 64, 1);
}

// The batched decode wrappers against the single-row operators they loop over.
//
// Each batched entry point is a host loop around the same single-row launcher,
// so with identical per-row inputs it has to reproduce it byte for byte. That
// is what this compares -- the single-row operator, not a host reference. A
// host reference would only re-test the kernel; comparing the two entry points
// pins down the per-row wiring the batched path adds on top of it: position,
// slot and stride arithmetic. A batched step that still disagrees with a
// single-row reference after passing here disagrees because of what the caller
// fed in, not because of how the rows were addressed.
//
// Every comparison is exact. There is no tolerance to tune: both sides launch
// the same kernel over the same bytes, so a difference means the addressing
// differs, which is the failure this test exists to name.
void test_batched_rows(std::mt19937& rng) {
    {
        const int rows = 4;
        const int q_heads = 5;
        const int kv_heads = 2;
        const int head_dim = 80;
        const int rotary_dim = 64;
        const float theta = 1000000.0f;
        // Independent sequences share no position and need not arrive in order,
        // so the batch covers an out-of-order spread and a repeated position.
        const std::vector<int> positions = {37, 5, 37, 12};
        const std::vector<uint16_t> q = random_halves(
            static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.7f);
        const std::vector<uint16_t> k = random_halves(
            static_cast<size_t>(rows) * kv_heads * head_dim, rng, 0.7f);

        DeviceBuffer<uint16_t> d_q(q), d_k(k);
        expect(pocket::qwen_partial_rope_rows_f16_batched_ascend(
                   d_q.get(), d_k.get(), positions.data(), rows, rotary_dim,
                   theta, q_heads, kv_heads, head_dim, nullptr),
               "batched partial rope launch");
        sync_or_throw("batched partial rope");

        std::vector<uint16_t> want_q(q.size());
        std::vector<uint16_t> want_k(k.size());
        for (int row = 0; row < rows; ++row) {
            const size_t q_row = static_cast<size_t>(row) * q_heads * head_dim;
            const size_t k_row = static_cast<size_t>(row) * kv_heads * head_dim;
            DeviceBuffer<uint16_t> s_q(
                std::vector<uint16_t>(q.begin() + q_row,
                                      q.begin() + q_row + q_heads * head_dim));
            DeviceBuffer<uint16_t> s_k(
                std::vector<uint16_t>(k.begin() + k_row,
                                      k.begin() + k_row + kv_heads * head_dim));
            expect(pocket::qwen_partial_rope_rows_f16(
                       s_q.get(), s_k.get(), positions[row], 1, rotary_dim, theta,
                       q_heads, kv_heads, head_dim),
                   "single-row partial rope launch");
            sync_or_throw("single-row partial rope");
            const std::vector<uint16_t> got_q = s_q.download();
            const std::vector<uint16_t> got_k = s_k.download();
            std::copy(got_q.begin(), got_q.end(), want_q.begin() + q_row);
            std::copy(got_k.begin(), got_k.end(), want_k.begin() + k_row);
        }
        expect_half_exact(d_q.download(), want_q,
                          "batched partial rope matches single-row q");
        expect_half_exact(d_k.download(), want_k,
                          "batched partial rope matches single-row k");
    }

    {
        const int rows = 4;
        const int kv_heads = 3;
        const int head_dim = 17;
        const int max_context = 8;
        const size_t stride =
            static_cast<size_t>(max_context) * kv_heads * head_dim;
        const std::vector<int> positions = {2, 5, 0, 7};
        // Slots are not the row index on purpose: a wrapper that appends to the
        // row's slot instead of the row's *assigned* slot still passes when the
        // two coincide.
        const std::vector<int> slots = {0, 2, 1, 3};
        const std::vector<uint16_t> k_rows = random_halves(
            static_cast<size_t>(rows) * kv_heads * head_dim, rng, 0.8f);
        const std::vector<uint16_t> v_rows = random_halves(
            static_cast<size_t>(rows) * kv_heads * head_dim, rng, 0.8f);

        // A distinct poison per slot turns a slot mix-up into a visible number
        // instead of two slots that happen to hold the same bytes.
        std::vector<uint16_t> k_cache(stride * rows);
        std::vector<uint16_t> v_cache(stride * rows);
        for (int slot = 0; slot < rows; ++slot) {
            std::fill(k_cache.begin() + slot * stride,
                      k_cache.begin() + (slot + 1) * stride,
                      float_to_half(1.0f + static_cast<float>(slot)));
            std::fill(v_cache.begin() + slot * stride,
                      v_cache.begin() + (slot + 1) * stride,
                      float_to_half(-1.0f - static_cast<float>(slot)));
        }

        DeviceBuffer<uint16_t> d_k_rows(k_rows), d_v_rows(v_rows);
        DeviceBuffer<uint16_t> d_batched_k(k_cache), d_batched_v(v_cache);
        expect(pocket::qwen_append_kv_cache_f16_batched_ascend(
                   d_k_rows.get(), d_v_rows.get(), d_batched_k.get(),
                   d_batched_v.get(), positions.data(), slots.data(), rows,
                   kv_heads, head_dim, max_context, stride, nullptr),
               "batched KV append launch");
        sync_or_throw("batched KV append");

        DeviceBuffer<uint16_t> d_single_k(k_cache), d_single_v(v_cache);
        for (int row = 0; row < rows; ++row) {
            const size_t source =
                static_cast<size_t>(row) * kv_heads * head_dim;
            const size_t target = static_cast<size_t>(slots[row]) * stride;
            expect(pocket::qwen_append_kv_cache_f16(
                       d_k_rows.get() + source, d_v_rows.get() + source,
                       d_single_k.get() + target, d_single_v.get() + target, 1,
                       kv_heads, head_dim, positions[row], max_context),
                   "single-row KV append launch");
            sync_or_throw("single-row KV append");
        }
        expect_half_exact(d_batched_k.download(), d_single_k.download(),
                          "batched KV append matches single-row K");
        expect_half_exact(d_batched_v.download(), d_single_v.download(),
                          "batched KV append matches single-row V");
    }

    {
        const int rows = 4;
        const int q_heads = 6;
        const int kv_heads = 2;
        const int head_dim = 64;
        const int max_context = 40;
        const size_t stride =
            static_cast<size_t>(max_context) * kv_heads * head_dim;
        const int scratch_row_stride = q_heads * max_context;
        // Rows in one batch are at different points in their sequences, which is
        // the case the per-row context length argument exists for.
        const std::vector<int> context_lens = {3, 17, 40, 1};
        const std::vector<int> slots = {1, 0, 3, 2};
        std::vector<int> slot_context(rows, 0);
        for (int row = 0; row < rows; ++row) {
            slot_context[slots[row]] = context_lens[row];
        }

        const std::vector<uint16_t> q = random_halves(
            static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.45f);
        // Random inside each slot's live context and poisoned past it, so a row
        // that reads too far -- or reads the wrong slot -- produces an output
        // that has nothing to do with the single-row call it is compared to.
        std::vector<uint16_t> k_cache(stride * rows);
        std::vector<uint16_t> v_cache(stride * rows);
        for (int slot = 0; slot < rows; ++slot) {
            std::vector<uint16_t> k_slot = random_halves(stride, rng, 0.5f);
            std::vector<uint16_t> v_slot = random_halves(stride, rng, 0.5f);
            const size_t live =
                static_cast<size_t>(slot_context[slot]) * kv_heads * head_dim;
            std::fill(k_slot.begin() + live, k_slot.end(), float_to_half(40.0f));
            std::fill(v_slot.begin() + live, v_slot.end(), float_to_half(-40.0f));
            std::copy(k_slot.begin(), k_slot.end(),
                      k_cache.begin() + slot * stride);
            std::copy(v_slot.begin(), v_slot.end(),
                      v_cache.begin() + slot * stride);
        }

        DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
        DeviceBuffer<uint16_t> d_batched_out(
            static_cast<size_t>(rows) * q_heads * head_dim);
        DeviceBuffer<float> d_batched_scores(
            static_cast<size_t>(rows) * scratch_row_stride);
        expect(pocket::qwen_gqa_decode_attention_f16_batched_ascend(
                   d_q.get(), d_k.get(), d_v.get(), d_batched_out.get(),
                   d_batched_scores.get(), scratch_row_stride,
                   context_lens.data(), slots.data(), rows, q_heads, kv_heads,
                   head_dim, max_context, stride, nullptr),
               "batched GQA decode launch");
        sync_or_throw("batched GQA decode");

        DeviceBuffer<uint16_t> d_single_out(
            static_cast<size_t>(rows) * q_heads * head_dim);
        DeviceBuffer<float> d_single_scores(
            static_cast<size_t>(rows) * scratch_row_stride);
        for (int row = 0; row < rows; ++row) {
            expect(pocket::qwen_gqa_decode_attention_f16(
                       d_q.get() + static_cast<size_t>(row) * q_heads * head_dim,
                       d_k.get() + static_cast<size_t>(slots[row]) * stride,
                       d_v.get() + static_cast<size_t>(slots[row]) * stride,
                       d_single_out.get() +
                           static_cast<size_t>(row) * q_heads * head_dim,
                       d_single_scores.get() +
                           static_cast<size_t>(row) * scratch_row_stride,
                       q_heads, kv_heads, head_dim, context_lens[row],
                       max_context),
                   "single-row GQA decode launch");
            sync_or_throw("single-row GQA decode");
        }
        expect_half_exact(d_batched_out.download(), d_single_out.download(),
                          "batched GQA decode matches single-row output");

        // The scratch is compared path against path rather than against a host
        // reference: this test asks whether the two entry points address the
        // rows the same way, and a host reference would fold in the kernel's own
        // precision, which the group A suite already covers.
        const std::vector<float> batched_scores = d_batched_scores.download();
        const std::vector<float> single_scores = d_single_scores.download();
        int score_mismatches = 0;
        size_t first_mismatch = 0;
        for (size_t i = 0; i < single_scores.size(); ++i) {
            if (batched_scores[i] != single_scores[i]) {
                if (score_mismatches == 0) first_mismatch = i;
                ++score_mismatches;
            }
        }
        std::string score_detail;
        if (score_mismatches != 0) {
            score_detail =
                " first=" + std::to_string(first_mismatch) +
                " got=" + std::to_string(batched_scores[first_mismatch]) +
                " want=" + std::to_string(single_scores[first_mismatch]);
        }
        expect(score_mismatches == 0,
               "batched GQA decode matches single-row score scratch mismatches=" +
                   std::to_string(score_mismatches) + score_detail);
    }

    {
        // The linear-attention layers carry the other two batched wrappers.
        // They matter here because a logit difference that is a fraction of an
        // ULP at one layer and half a logit at five is a difference that was
        // seeded early, and these layers are the early ones.
        const int rows = 4;
        const int channels = 24;
        const int kernel = 4;
        const size_t stride = static_cast<size_t>(channels) * kernel;
        const std::vector<int> slots = {2, 0, 3, 1};
        const std::vector<uint16_t> x =
            random_halves(static_cast<size_t>(rows) * channels, rng, 0.9f);
        const std::vector<uint16_t> weight =
            random_halves(static_cast<size_t>(channels) * kernel, rng, 0.6f);
        std::vector<uint16_t> tail(stride * rows);
        for (int slot = 0; slot < rows; ++slot) {
            const std::vector<uint16_t> slice = random_halves(stride, rng, 0.9f);
            std::copy(slice.begin(), slice.end(), tail.begin() + slot * stride);
        }

        DeviceBuffer<uint16_t> d_x(x), d_w(weight);
        DeviceBuffer<uint16_t> d_batched_tail(tail);
        DeviceBuffer<uint16_t> d_batched_y(
            static_cast<size_t>(rows) * channels);
        expect(pocket::qwen_causal_depthwise_conv_silu_f16_batched_ascend(
                   d_x.get(), d_w.get(), d_batched_tail.get(),
                   d_batched_y.get(), slots.data(), rows, channels, kernel,
                   stride, nullptr),
               "batched conv launch");
        sync_or_throw("batched conv");

        DeviceBuffer<uint16_t> d_single_tail(tail);
        DeviceBuffer<uint16_t> d_single_y(static_cast<size_t>(rows) * channels);
        for (int row = 0; row < rows; ++row) {
            expect(pocket::qwen_causal_depthwise_conv_silu_f16(
                       d_x.get() + static_cast<size_t>(row) * channels,
                       d_w.get(),
                       d_single_tail.get() +
                           static_cast<size_t>(slots[row]) * stride,
                       d_single_y.get() + static_cast<size_t>(row) * channels, 1,
                       channels, kernel, true),
                   "single-row conv launch");
            sync_or_throw("single-row conv");
        }
        expect_half_exact(d_batched_y.download(), d_single_y.download(),
                          "batched conv matches single-row output");
        // The tail is written as well as read, so a wrapper that updates the
        // wrong slot leaves the next step's history in a different place even
        // when this step's output happens to agree.
        expect_half_exact(d_batched_tail.download(), d_single_tail.download(),
                          "batched conv matches single-row tail");
    }

    {
        const int rows = 4;
        const int heads = 2;
        const int key_heads = 1;
        const int key_dim = kKeyDim;
        const int value_dim = kValueDim;
        const float q_scale = 0.1f;
        const size_t stride = static_cast<size_t>(heads) * key_dim * value_dim;
        const std::vector<int> slots = {1, 3, 0, 2};
        const std::vector<uint16_t> q = random_halves(
            static_cast<size_t>(rows) * key_heads * key_dim, rng, 0.5f);
        const std::vector<uint16_t> k = random_halves(
            static_cast<size_t>(rows) * key_heads * key_dim, rng, 0.5f);
        const std::vector<uint16_t> v = random_halves(
            static_cast<size_t>(rows) * heads * value_dim, rng, 0.5f);
        const std::vector<uint16_t> g =
            random_halves(static_cast<size_t>(rows) * heads, rng, 0.5f);
        const std::vector<uint16_t> beta =
            random_halves(static_cast<size_t>(rows) * heads, rng, 0.5f);
        const std::vector<float> state = random_floats(stride * rows, rng, 0.5f);

        DeviceBuffer<uint16_t> d_q(q), d_k(k), d_v(v), d_g(g), d_beta(beta);
        DeviceBuffer<float> d_batched_state(state);
        DeviceBuffer<uint16_t> d_batched_out(
            static_cast<size_t>(rows) * heads * value_dim);
        expect(pocket::qwen_gated_delta_step_batched_f16_ascend(
                   d_batched_state.get(), d_q.get(), d_k.get(), d_v.get(),
                   d_g.get(), d_beta.get(), d_batched_out.get(), slots.data(),
                   rows, heads, key_heads, key_dim, value_dim, q_scale, stride,
                   nullptr),
               "batched gated delta launch");
        sync_or_throw("batched gated delta");

        DeviceBuffer<float> d_single_state(state);
        DeviceBuffer<uint16_t> d_single_out(
            static_cast<size_t>(rows) * heads * value_dim);
        for (int row = 0; row < rows; ++row) {
            const size_t q_row = static_cast<size_t>(row) * key_heads * key_dim;
            const size_t v_row = static_cast<size_t>(row) * heads * value_dim;
            expect(pocket::qwen_gated_delta_step_f16(
                       d_single_state.get() +
                           static_cast<size_t>(slots[row]) * stride,
                       d_q.get() + q_row, d_k.get() + q_row, d_v.get() + v_row,
                       d_g.get() + static_cast<size_t>(row) * heads,
                       d_beta.get() + static_cast<size_t>(row) * heads,
                       d_single_out.get() + v_row, heads, key_heads, key_dim,
                       value_dim, q_scale),
                   "single-row gated delta launch");
            sync_or_throw("single-row gated delta");
        }
        expect_half_exact(d_batched_out.download(), d_single_out.download(),
                          "batched gated delta matches single-row output");
        // The state is the whole point of this operator: the next decode step
        // reads it, so a difference here is a difference that outlives the call.
        const std::vector<float> batched_state = d_batched_state.download();
        const std::vector<float> single_state = d_single_state.download();
        int state_mismatches = 0;
        double worst_state = 0.0;
        for (size_t i = 0; i < single_state.size(); ++i) {
            if (batched_state[i] == single_state[i]) continue;
            ++state_mismatches;
            const double want = static_cast<double>(single_state[i]);
            worst_state = std::max(
                worst_state,
                std::fabs(static_cast<double>(batched_state[i]) - want) /
                    std::max(std::fabs(want), 1.0e-3));
        }
        std::cout << "  batched gated delta state: bit_mismatches="
                  << state_mismatches << " of " << single_state.size()
                  << " worst_relative=" << worst_state << "\n";
        expect(state_mismatches == 0,
               "batched gated delta matches single-row state");
    }

    {
        // The one thing batching does that a single-row reference structurally
        // cannot: it hands the rows to one aclnn call instead of one aclnn call
        // per row. aclnn owns its own tiling, and a tiling that changes with M
        // is free to accumulate in a different order, so the two are not
        // guaranteed to be bit-identical. This block puts the size of that
        // difference on record at the shapes the TP4 model actually uses --
        // hidden 5120 and a per-rank MLP shard of 4352 -- because a per-token
        // logit offset that starts at a fraction of an ULP and grows with depth
        // looks the same in a logit diff as a defect does.
        //
        // The answer, measured: of the ten operators the decoder layer puts
        // through aclnn, every fp16 one -- both matmuls, rmsnorm, residual-add
        // rmsnorm, residual add, silu-mul, swiglu matmul, gated rmsnorm -- is
        // bit-identical at M=16 to 16 single-row calls, and only the fp32
        // logits projection is not. That is consistent with the batched path
        // reproducing the single-row path exactly everywhere the layer's own
        // arithmetic is involved, which localises the observed batched-vs-
        // single-row logit offset to this operator's accumulation order.
        const int batch = 16;
        const int in_dim = 5120;
        const int out_dim = 4352;
        int worst_exact_mismatches = 0;
        double worst_exact_relative = 0.0;
        // The fp32 matmul is the only operator in this block that is allowed to
        // differ at all, and what bounds it is absolute error, not relative
        // error -- see `report_bounded`.
        double worst_bounded_absolute = 0.0;
        auto report = [&](const char* name, int mismatches, size_t total,
                          double worst) {
            worst_exact_mismatches = std::max(worst_exact_mismatches, mismatches);
            worst_exact_relative = std::max(worst_exact_relative, worst);
            std::cout << "  " << name << " M=" << batch << " vs " << batch
                      << "x M=1: bit_mismatches=" << mismatches << " of "
                      << total << " worst_relative=" << worst << "\n";
        };
        auto report_bounded = [&](const char* name, int mismatches, size_t total,
                                  double worst, double absolute) {
            worst_bounded_absolute = std::max(worst_bounded_absolute, absolute);
            std::cout << "  " << name << " M=" << batch << " vs " << batch
                      << "x M=1: bit_mismatches=" << mismatches << " of "
                      << total << " worst_relative=" << worst
                      << " (relative is reported, not gated)\n";
        };
        auto compare_halves = [](const std::vector<uint16_t>& got,
                                 const std::vector<uint16_t>& want,
                                 int& mismatches, double& worst) {
            for (size_t i = 0; i < want.size(); ++i) {
                if (got[i] == want[i]) continue;
                ++mismatches;
                const double w = half_to_float(want[i]);
                worst = std::max(worst,
                                 std::fabs(half_to_float(got[i]) - w) /
                                     std::max(std::fabs(w), 1.0e-3));
            }
        };
        auto compare_floats = [](const std::vector<float>& got,
                                 const std::vector<float>& want,
                                 int& mismatches, double& worst,
                                 double& worst_absolute) {
            double sum_absolute = 0.0;
            for (size_t i = 0; i < want.size(); ++i) {
                sum_absolute += std::fabs(static_cast<double>(want[i]));
                if (got[i] == want[i]) continue;
                ++mismatches;
                const double w = static_cast<double>(want[i]);
                worst_absolute = std::max(
                    worst_absolute,
                    std::fabs(static_cast<double>(got[i]) - w));
                // Reported for context only. The reference is a logit-like
                // distribution with a mean magnitude near 0.19, so elements
                // that sit next to zero dominate the ratio and make it swing
                // run to run; it is not a bound anything can be gated on.
                worst = std::max(worst,
                                 std::fabs(static_cast<double>(got[i]) - w) /
                                     std::max(std::fabs(w), 1.0e-6));
            }
            std::cout << "    fp32 detail: worst_abs=" << worst_absolute
                      << " mean_abs_want="
                      << sum_absolute / static_cast<double>(want.size())
                      << "\n";
        };

        const size_t rows_elems = static_cast<size_t>(batch) * in_dim;
        const std::vector<uint16_t> x = random_halves(rows_elems, rng, 0.5f);
        const std::vector<uint16_t> weight =
            random_halves(static_cast<size_t>(out_dim) * in_dim, rng, 0.02f);
        const std::vector<uint16_t> gamma = random_halves(in_dim, rng, 0.4f);
        const float eps = 1.0e-6f;
        DeviceBuffer<uint16_t> d_x(x), d_w(weight), d_gamma(gamma);

        {
            DeviceBuffer<uint16_t> d_batched(
                static_cast<size_t>(batch) * out_dim);
            expect(pocket::qwen_fp16_matmul_rows_f16(
                       d_x.get(), d_w.get(), d_batched.get(), batch, out_dim,
                       in_dim, in_dim, out_dim, in_dim),
                   "batched matmul launch");
            sync_or_throw("batched matmul");
            DeviceBuffer<uint16_t> d_single(
                static_cast<size_t>(batch) * out_dim);
            for (int row = 0; row < batch; ++row) {
                expect(pocket::qwen_fp16_matmul_rows_f16(
                           d_x.get() + static_cast<size_t>(row) * in_dim,
                           d_w.get(),
                           d_single.get() + static_cast<size_t>(row) * out_dim,
                           1, out_dim, in_dim, in_dim, out_dim, in_dim),
                       "single-row matmul launch");
                sync_or_throw("single-row matmul");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched.download(), d_single.download(), mismatches,
                           worst);
            report("matmul fp16", mismatches,
                   static_cast<size_t>(batch) * out_dim, worst);
        }

        {
            // The logits pass is the one whose output the parity oracle reads
            // directly, so it is checked at both the per-rank MLP shard and the
            // real per-rank vocabulary shard (248320 / 4).
            for (const int fp32_out_dim : {out_dim, 62080}) {
                const std::vector<uint16_t> fp32_weight =
                    fp32_out_dim == out_dim
                        ? weight
                        : random_halves(
                              static_cast<size_t>(fp32_out_dim) * in_dim, rng,
                              0.02f);
                DeviceBuffer<uint16_t> d_fp32_w(fp32_weight);
                DeviceBuffer<float> d_batched(
                    static_cast<size_t>(batch) * fp32_out_dim);
                expect(pocket::qwen_fp16_matmul_rows_f16_f32(
                           d_x.get(), d_fp32_w.get(), d_batched.get(), batch,
                           fp32_out_dim, in_dim, in_dim, fp32_out_dim, in_dim),
                       "batched matmul fp32 launch");
                sync_or_throw("batched matmul fp32");
                DeviceBuffer<float> d_single(
                    static_cast<size_t>(batch) * fp32_out_dim);
                for (int row = 0; row < batch; ++row) {
                    expect(pocket::qwen_fp16_matmul_rows_f16_f32(
                               d_x.get() + static_cast<size_t>(row) * in_dim,
                               d_fp32_w.get(),
                               d_single.get() +
                                   static_cast<size_t>(row) * fp32_out_dim,
                               1, fp32_out_dim, in_dim, in_dim, fp32_out_dim,
                               in_dim),
                           "single-row matmul fp32 launch");
                    sync_or_throw("single-row matmul fp32");
                }
                int mismatches = 0;
                double worst = 0.0;
                double worst_absolute = 0.0;
                compare_floats(d_batched.download(), d_single.download(),
                               mismatches, worst, worst_absolute);
                const std::string name =
                    "matmul fp32 N=" + std::to_string(fp32_out_dim);
                report_bounded(name.c_str(), mismatches,
                               static_cast<size_t>(batch) * fp32_out_dim, worst,
                               worst_absolute);
            }
        }

        {
            DeviceBuffer<uint16_t> d_batched(rows_elems);
            expect(pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                       d_x.get(), d_gamma.get(), d_batched.get(), batch, in_dim,
                       eps),
                   "batched rmsnorm launch");
            sync_or_throw("batched rmsnorm");
            DeviceBuffer<uint16_t> d_single(rows_elems);
            for (int row = 0; row < batch; ++row) {
                expect(pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                           d_x.get() + static_cast<size_t>(row) * in_dim,
                           d_gamma.get(),
                           d_single.get() + static_cast<size_t>(row) * in_dim, 1,
                           in_dim, eps),
                       "single-row rmsnorm launch");
                sync_or_throw("single-row rmsnorm");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched.download(), d_single.download(), mismatches,
                           worst);
            report("rmsnorm", mismatches, rows_elems, worst);
        }

        {
            // Both inputs and the residual output are read-modify-written, so
            // each path needs its own copy rather than the same buffer replayed.
            const std::vector<uint16_t> delta = random_halves(rows_elems, rng, 0.5f);
            DeviceBuffer<uint16_t> d_delta(delta);
            DeviceBuffer<uint16_t> d_batched_residual(x);
            DeviceBuffer<uint16_t> d_batched_norm(rows_elems);
            expect(pocket::qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
                       d_x.get(), d_delta.get(), d_gamma.get(),
                       d_batched_residual.get(), d_batched_norm.get(), batch,
                       in_dim, eps),
                   "batched residual add rmsnorm launch");
            sync_or_throw("batched residual add rmsnorm");
            DeviceBuffer<uint16_t> d_single_residual(x);
            DeviceBuffer<uint16_t> d_single_norm(rows_elems);
            for (int row = 0; row < batch; ++row) {
                const size_t offset = static_cast<size_t>(row) * in_dim;
                expect(pocket::qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
                           d_x.get() + offset, d_delta.get() + offset,
                           d_gamma.get(), d_single_residual.get() + offset,
                           d_single_norm.get() + offset, 1, in_dim, eps),
                       "single-row residual add rmsnorm launch");
                sync_or_throw("single-row residual add rmsnorm");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched_norm.download(), d_single_norm.download(),
                           mismatches, worst);
            report("residual add rmsnorm", mismatches, rows_elems, worst);
            mismatches = 0;
            worst = 0.0;
            compare_halves(d_batched_residual.download(),
                           d_single_residual.download(), mismatches, worst);
            report("residual add", mismatches, rows_elems, worst);
        }

        {
            const std::vector<uint16_t> up = random_halves(rows_elems, rng, 0.5f);
            DeviceBuffer<uint16_t> d_up(up);
            DeviceBuffer<uint16_t> d_batched(rows_elems);
            expect(pocket::qwen_silu_mul_rows_f16(d_x.get(), d_up.get(),
                                                  d_batched.get(), batch, in_dim),
                   "batched silu mul launch");
            sync_or_throw("batched silu mul");
            DeviceBuffer<uint16_t> d_single(rows_elems);
            for (int row = 0; row < batch; ++row) {
                const size_t offset = static_cast<size_t>(row) * in_dim;
                expect(pocket::qwen_silu_mul_rows_f16(
                           d_x.get() + offset, d_up.get() + offset,
                           d_single.get() + offset, 1, in_dim),
                       "single-row silu mul launch");
                sync_or_throw("single-row silu mul");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched.download(), d_single.download(), mismatches,
                           worst);
            report("silu mul", mismatches, rows_elems, worst);
        }

        {
            // The two remaining fused operators that feed forward rather than
            // terminate a layer. Both are aclnn calls whose tiling can depend on
            // M, and both sit inside every decoder layer, so a ULP here is a
            // difference the next layer inherits.
            const std::vector<uint16_t> gate_weight =
                random_halves(static_cast<size_t>(out_dim) * in_dim, rng, 0.02f);
            const std::vector<uint16_t> up_weight =
                random_halves(static_cast<size_t>(out_dim) * in_dim, rng, 0.02f);
            DeviceBuffer<uint16_t> d_gate_w(gate_weight), d_up_w(up_weight);
            DeviceBuffer<uint16_t> d_batched(static_cast<size_t>(batch) * out_dim);
            expect(pocket::qwen_fp16_swiglu_matmul_rows_f16(
                       d_x.get(), d_gate_w.get(), d_up_w.get(), d_batched.get(),
                       batch, out_dim, in_dim, in_dim, out_dim, in_dim),
                   "batched swiglu matmul launch");
            sync_or_throw("batched swiglu matmul");
            DeviceBuffer<uint16_t> d_single(static_cast<size_t>(batch) * out_dim);
            for (int row = 0; row < batch; ++row) {
                expect(pocket::qwen_fp16_swiglu_matmul_rows_f16(
                           d_x.get() + static_cast<size_t>(row) * in_dim,
                           d_gate_w.get(), d_up_w.get(),
                           d_single.get() + static_cast<size_t>(row) * out_dim, 1,
                           out_dim, in_dim, in_dim, out_dim, in_dim),
                       "single-row swiglu matmul launch");
                sync_or_throw("single-row swiglu matmul");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched.download(), d_single.download(), mismatches,
                           worst);
            report("swiglu matmul", mismatches,
                   static_cast<size_t>(batch) * out_dim, worst);
        }

        {
            const std::vector<uint16_t> gate = random_halves(rows_elems, rng, 0.4f);
            DeviceBuffer<uint16_t> d_gate(gate);
            DeviceBuffer<uint16_t> d_batched(rows_elems);
            expect(pocket::qwen_gated_rmsnorm_fp16_gamma_rows_f16(
                       d_x.get(), d_gamma.get(), d_gate.get(), d_batched.get(),
                       batch, in_dim, eps),
                   "batched gated rmsnorm launch");
            sync_or_throw("batched gated rmsnorm");
            DeviceBuffer<uint16_t> d_single(rows_elems);
            for (int row = 0; row < batch; ++row) {
                const size_t offset = static_cast<size_t>(row) * in_dim;
                expect(pocket::qwen_gated_rmsnorm_fp16_gamma_rows_f16(
                           d_x.get() + offset, d_gamma.get(),
                           d_gate.get() + offset, d_single.get() + offset, 1,
                           in_dim, eps),
                       "single-row gated rmsnorm launch");
                sync_or_throw("single-row gated rmsnorm");
            }
            int mismatches = 0;
            double worst = 0.0;
            compare_halves(d_batched.download(), d_single.download(), mismatches,
                           worst);
            report("gated rmsnorm", mismatches, rows_elems, worst);
        }

        // Two different claims, held to two different standards on purpose.
        //
        // Every fp16 entry point above has to be bit-identical: the same
        // kernel reads the same bytes on both sides, so any difference at all
        // is the addressing changing with M, and that is a defect rather than
        // a tolerance question.
        //
        // The fp32 matmul cannot meet that standard, because it is not the
        // same computation at the two sizes -- one aclnn call over 16 rows
        // tiles and accumulates differently from 16 calls over one row each.
        // What it can be held to is the size of the difference, and absolute
        // error is the meaningful scale for it: the reference is a logit-like
        // vector of mean magnitude ~0.19, the measured worst case is a handful
        // of fp32 ULPs, and the parity oracle's own offsets at the first
        // full-attention layer are of order 1. A bound of 1e-5 sits roughly an
        // order of magnitude above the measurement and three orders below
        // anything that could reorder an argmax, so it leaves room for a
        // different accumulation order without leaving room for a wrong one.
        std::cout << "  worst over the exact operators: bit_mismatches="
                  << worst_exact_mismatches
                  << " worst_relative=" << worst_exact_relative << "\n";
        std::cout << "  worst fp32 absolute error: " << worst_bounded_absolute
                  << "\n";
        expect(worst_exact_mismatches == 0,
               "batched fp16 aclnn calls are bit-identical to single-row calls");
        expect(worst_bounded_absolute < 1.0e-5,
               "batched fp32 logits matmul stays inside its accumulation bound");
    }
}

void test_argmax(std::mt19937& rng) {
    (void)rng;
    const int rows = 5;
    const int count = 263;
    const int token_offset = -17;
    std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
    logits[7] = 3.0f;
    logits[2] = 3.0f;
    logits[static_cast<size_t>(count) + 5] = 4.0f;
    logits[static_cast<size_t>(count) + 9] = 4.0f;
    logits[static_cast<size_t>(2) * count + 262] = 9.0f;
    logits[static_cast<size_t>(2) * count + 3] = 8.0f;
    logits[static_cast<size_t>(3) * count] = -1.0f;
    logits[static_cast<size_t>(3) * count + 11] = -1.0f;
    logits[static_cast<size_t>(4) * count + 129] = 0.5f;
    logits[static_cast<size_t>(4) * count + 130] = 0.5f;

    DeviceBuffer<float> d_logits(logits);
    DeviceBuffer<int> d_tokens(rows);
    DeviceBuffer<float> d_values(rows);
    expect(pocket::qwen_argmax_fp32_rows(
               d_logits.get(), d_tokens.get(), d_values.get(), rows, count,
               token_offset),
           "argmax launch");
    sync_or_throw("argmax");

    const std::vector<int> expected_tokens = {-15, -12, 245, -17, 112};
    const std::vector<float> expected_values = {3.0f, 4.0f, 9.0f, -1.0f, 0.5f};
    expect(d_tokens.download() == expected_tokens,
           "argmax selects lowest global id on ties");
    expect(d_values.download() == expected_values,
           "argmax returns winning logits");

    // The decode path calls this with rows == 1 and a vocabulary-wide row, and it
    // allocates the 4-byte token and logit outputs from one workspace arena, so
    // they can land in a single 32-byte cache line. Cover that shape explicitly:
    // the multi-row case above would not catch a single-row store defect.
    const int wide = 248320;
    std::vector<float> row(static_cast<size_t>(wide), -3.0f);
    row[wide - 1] = 11.5f;
    row[4096] = 11.0f;
    DeviceBuffer<float> d_row(row);
    DeviceBuffer<int> d_token(1);
    DeviceBuffer<float> d_value(1);
    expect(pocket::qwen_argmax_fp32_rows(d_row.get(), d_token.get(),
                                       d_value.get(), 1, wide, 0),
           "argmax single-row launch");
    sync_or_throw("argmax single row");
    expect(d_token.download() == std::vector<int>{wide - 1},
           "argmax single row selects the last index");
    expect(d_value.download() == std::vector<float>{11.5f},
           "argmax single row returns its logit");
}

void test_gqa_prefill(std::mt19937& rng) {
    struct Shape {
        int rows;
        int q_heads;
        int kv_heads;
        int head_dim;
        int position_offset;
        int max_context;
        const char* name;
    };
    const Shape shapes[] = {
        {3, 6, 2, 32, 2, 9, "general"},
        {3, 6, 1, 256, 2, 9, "vector_aligned"},
    };
    for (const Shape& shape : shapes) {
        const std::vector<uint16_t> q = random_halves(
            static_cast<size_t>(shape.rows) * shape.q_heads * shape.head_dim,
            rng, 0.45f);
        std::vector<uint16_t> k_cache = random_halves(
            static_cast<size_t>(shape.max_context) * shape.kv_heads * shape.head_dim,
            rng, 0.5f);
        std::vector<uint16_t> v_cache = random_halves(
            static_cast<size_t>(shape.max_context) * shape.kv_heads * shape.head_dim,
            rng, 0.5f);
        const size_t poison_start = static_cast<size_t>(shape.position_offset +
                                                         shape.rows) *
                                    shape.kv_heads * shape.head_dim;
        std::fill(k_cache.begin() + poison_start, k_cache.end(),
                  float_to_half(40.0f));
        std::fill(v_cache.begin() + poison_start, v_cache.end(),
                  float_to_half(-40.0f));
        DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
        DeviceBuffer<uint16_t> d_out(static_cast<size_t>(shape.rows) *
                                     shape.q_heads * shape.head_dim);
        expect(pocket::qwen_gqa_prefill_attention_f16(
                   d_q.get(), d_k.get(), d_v.get(), d_out.get(), shape.rows,
                   shape.q_heads, shape.kv_heads, shape.head_dim,
                   shape.position_offset, shape.max_context),
               std::string("GQA prefill ") + shape.name + " launch");
        sync_or_throw(std::string("GQA prefill ") + shape.name);
        expect_half_close(
            d_out.download(),
            attention_reference(q, k_cache, v_cache, shape.rows, shape.q_heads,
                                shape.kv_heads, shape.head_dim,
                                shape.position_offset),
            5.0e-3, 2.0e-3,
            std::string("GQA prefill ") + shape.name + " output");
    }
}

std::vector<double> gqa_verify_partial_reference(
    const std::vector<uint16_t>& q, const std::vector<uint16_t>& k_cache,
    const std::vector<uint16_t>& v_cache, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int splits) {
    const int context_len = position_offset + rows;
    const int positions_per_split = (context_len + splits - 1) / splits;
    const int stride = head_dim + 2;
    std::vector<double> out(static_cast<size_t>(rows) * q_heads * splits * stride,
                            0.0);
    const int repeat = q_heads / kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int row = 0; row < rows; ++row) {
        const int limit = position_offset + row + 1;
        for (int head = 0; head < q_heads; ++head) {
            const int kv_head = head / repeat;
            const size_t q_base =
                (static_cast<size_t>(row) * q_heads + head) * head_dim;
            const size_t output_base =
                (static_cast<size_t>(row) * q_heads + head) * splits * stride;
            for (int split = 0; split < splits; ++split) {
                const int begin = split * positions_per_split;
                const int end = std::min(context_len, std::min(
                    begin + positions_per_split, limit));
                const size_t base = output_base + static_cast<size_t>(split) * stride;
                if (begin >= end) {
                    out[base] = -static_cast<double>(std::numeric_limits<float>::max());
                    continue;
                }
                std::vector<double> probabilities(static_cast<size_t>(end - begin));
                double maximum = -std::numeric_limits<double>::infinity();
                for (int pos = begin; pos < end; ++pos) {
                    const size_t cache_base =
                        (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                    double score = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        score += static_cast<double>(half_to_float(q[q_base + d])) *
                                 half_to_float(k_cache[cache_base + d]);
                    }
                    probabilities[static_cast<size_t>(pos - begin)] = score * scale;
                    maximum = std::max(maximum, score * scale);
                }
                out[base] = maximum;
                double denominator = 0.0;
                for (double& score : probabilities) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                out[base + 1] = denominator;
                for (int d = 0; d < head_dim; ++d) {
                    double value = 0.0;
                    for (int pos = begin; pos < end; ++pos) {
                        const size_t cache_base =
                            (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                        value += probabilities[static_cast<size_t>(pos - begin)] *
                                 half_to_float(v_cache[cache_base + d]);
                    }
                    out[base + 2 + d] = value;
                }
            }
        }
    }
    return out;
}

void test_gqa_verify(std::mt19937& rng) {
    struct Shape {
        int rows;
        int q_heads;
        int kv_heads;
        int head_dim;
        int position_offset;
        int max_context;
        int splits;
        const char* name;
    };
    const Shape shapes[] = {
        {4, 6, 2, 32, 3, 10, 5, "short"},
        {2, 4, 2, 64, 131, 140, 3, "multi_tile"},
        {3, 6, 2, 17, 65, 72, 2, "unaligned"},
        {2, 2, 1, 17, 0, 8, 8, "empty_splits"},
        {4, 6, 1, 256, 3, 10, 5, "vector_aligned"},
    };
    for (const Shape& shape : shapes) {
        const size_t q_count = static_cast<size_t>(shape.rows) * shape.q_heads *
                               shape.head_dim;
        const size_t kv_count = static_cast<size_t>(shape.max_context) *
                                shape.kv_heads * shape.head_dim;
        const size_t out_count = q_count;
        const size_t partial_count = static_cast<size_t>(shape.rows) *
            shape.q_heads * shape.splits * (shape.head_dim + 2);
        const std::vector<uint16_t> q = random_halves(q_count, rng, 0.45f);
        std::vector<uint16_t> k_cache = random_halves(kv_count, rng, 0.5f);
        std::vector<uint16_t> v_cache = random_halves(kv_count, rng, 0.5f);
        const size_t poison_start = static_cast<size_t>(shape.position_offset +
                                                         shape.rows) *
                                    shape.kv_heads * shape.head_dim;
        std::fill(k_cache.begin() + poison_start, k_cache.end(),
                  float_to_half(40.0f));
        std::fill(v_cache.begin() + poison_start, v_cache.end(),
                  float_to_half(-40.0f));
        const std::vector<double> want_partial = gqa_verify_partial_reference(
            q, k_cache, v_cache, shape.rows, shape.q_heads, shape.kv_heads,
            shape.head_dim, shape.position_offset, shape.splits);
        const std::vector<double> want_output = attention_reference(
            q, k_cache, v_cache, shape.rows, shape.q_heads, shape.kv_heads,
            shape.head_dim, shape.position_offset);
        DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
        DeviceBuffer<uint16_t> d_out(out_count + 16);
        DeviceBuffer<float> d_partial(partial_count + 16);
        std::vector<uint16_t> out_init(out_count + 16, float_to_half(9.0f));
        std::vector<float> partial_init(partial_count + 16, 12345.0f);
        d_out.upload(out_init);
        d_partial.upload(partial_init);
        const size_t guard = std::string(shape.name) == "vector_aligned" ? 0 : 8;

        for (int pass = 0; pass < 2; ++pass) {
            expect(pocket::qwen_gqa_verify_attention_f16(
                       d_q.get(), d_k.get(), d_v.get(), d_out.get() + guard,
                       d_partial.get() + guard, shape.rows, shape.q_heads,
                       shape.kv_heads, shape.head_dim, shape.position_offset,
                       shape.max_context, shape.splits),
                   std::string("GQA verify ") + shape.name + " launch");
            sync_or_throw(std::string("GQA verify ") + shape.name);
            const std::vector<uint16_t> got_out = d_out.download();
            const std::vector<float> got_partial = d_partial.download();
            std::vector<uint16_t> out_payload(got_out.begin() + guard,
                                              got_out.begin() + guard + out_count);
            expect_half_close(out_payload, want_output, 5.0e-3, 2.0e-3,
                              std::string("GQA verify ") + shape.name +
                                  " output");
            std::vector<float> partial_payload(got_partial.begin() + guard,
                                                got_partial.begin() + guard + partial_count);
            expect_float_close(partial_payload, want_partial, 5.0e-4, 2.0e-4,
                               std::string("GQA verify ") + shape.name +
                                   " partials");
            expect(std::equal(got_out.begin(), got_out.begin() + guard,
                              out_init.begin()),
                   std::string("GQA verify ") + shape.name + " output prefix guard");
            expect(std::equal(got_out.begin() + guard + out_count, got_out.end(),
                              out_init.begin() + guard + out_count),
                   std::string("GQA verify ") + shape.name + " output suffix guard");
            expect(std::equal(got_partial.begin(), got_partial.begin() + guard,
                              partial_init.begin()),
                   std::string("GQA verify ") + shape.name + " partial prefix guard");
            expect(std::equal(got_partial.begin() + guard + partial_count,
                              got_partial.end(), partial_init.begin() + guard + partial_count),
                   std::string("GQA verify ") + shape.name + " partial suffix guard");
        }
    }
}

void test_argument_rejection() {
    DeviceBuffer<uint16_t> half(1024);
    DeviceBuffer<float> real(65536);
    expect(!pocket::qwen_normalize_gated_delta_qk_f16(
               nullptr, half.get(), real.get(), real.get(), 1, 1, kKeyDim),
           "normalize rejects null input");
    expect(!pocket::qwen_gated_delta_sequence_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 1, 3, 2, kKeyDim, kValueDim, 0.1f),
           "recurrence rejects a bad head ratio");
    expect(!pocket::qwen_gated_delta_step_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 2, 1, 64, kValueDim, 0.1f),
           "recurrence rejects unsupported key width");
    expect(!pocket::qwen_gated_delta_step_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 2, 1, kKeyDim, kValueDim, 0.0f),
           "recurrence rejects nonpositive q scale");
    expect(!pocket::qwen_linear_attn_gates_f16(
               half.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), 0, 4),
           "gates reject zero rows");
    expect(!pocket::qwen_causal_depthwise_conv_silu_f16(
               half.get(), half.get(), half.get(), half.get(), 1, 8, 9, true),
           "convolution rejects a kernel above eight");
    expect(!pocket::qwen_partial_rope_rows_f16(
               half.get(), half.get(), 0, 1, 63, 10000.0f, 2, 1, 64),
           "rope rejects an odd rotary dimension");
    expect(!pocket::qwen_append_kv_cache_f16(
               half.get(), half.get(), half.get(), half.get(), 2, 1, 16, 3, 4),
           "KV append rejects cache overflow");
    expect(!pocket::qwen_gqa_decode_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 3, 2,
               16, 2, 4),
           "GQA decode rejects a bad head ratio");
    expect(!pocket::qwen_gqa_decode_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 2, 1,
               257, 2, 4),
           "GQA decode rejects a head dimension above 256");
    expect(!pocket::qwen_gqa_prefill_attention_f16(
               half.get(), half.get(), half.get(), half.get(), 3, 2, 1, 16, 2,
               4),
           "GQA prefill rejects context overflow");
    expect(!pocket::qwen_gqa_verify_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 2, 2,
               1, 16, 1, 4, 0),
           "GQA verify rejects zero splits");
    expect(!pocket::qwen_gqa_verify_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 1, 2,
               1, 16, 1, 4, 1),
           "GQA verify rejects a single row");
    expect(!pocket::qwen_gqa_verify_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 2, 2,
               1, 16, 1, 4, 257),
           "GQA verify rejects more than 256 splits");
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--only" && i + 1 < argc) {
            only = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (!pocket::device_runtime_available()) {
        std::cout << "[SKIP] no device runtime available\n";
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::cout << "[SKIP] device_set failed for device " << device << "\n";
        return 0;
    }
    std::cout << "backend=" << pocket::device_backend_name() << " device=" << device
              << "\n";

    std::mt19937 rng(20260830u);
    const std::vector<std::pair<const char*, void (*)(std::mt19937&)>> suite = {
        {"normalize_qk", test_normalize_qk},
        {"gated_delta", test_gated_delta_recurrence},
        {"linear_attn_gates", test_linear_attn_gates},
        {"causal_conv", test_causal_conv},
        {"partial_rope", test_partial_rope},
        {"append_kv", test_append_kv},
        {"gqa_decode", test_gqa_decode},
        {"gqa_prefill", test_gqa_prefill},
        {"gqa_verify", test_gqa_verify},
        {"batched_rows", test_batched_rows},
        {"argmax", test_argmax},
    };
    for (const auto& entry : suite) {
        if (!only.empty() &&
            ("," + only + ",").find("," + std::string(entry.first) + ",") ==
                std::string::npos) {
            continue;
        }
        std::cout << "[run] " << entry.first << std::endl;
        try {
            entry.second(rng);
        } catch (const std::exception& error) {
            ++failures;
            std::cout << "  FAIL " << entry.first << " threw: " << error.what()
                      << "\n";
        }
    }
    if (only.empty() ||
        ("," + only + ",").find(",argument_rejection,") != std::string::npos) {
        std::cout << "[run] argument_rejection" << std::endl;
        try {
            test_argument_rejection();
        } catch (const std::exception& error) {
            ++failures;
            std::cout << "  FAIL argument_rejection threw: " << error.what() << "\n";
        }
    }

    if (failures != 0) {
        std::cout << "FAIL " << failures << " of " << checks << " checks\n";
        return 1;
    }
    std::cout << "PASS (" << checks << " checks)\n";
    return 0;
}
