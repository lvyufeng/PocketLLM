#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void fail(const char* message) {
    std::printf("[FAIL] %s\n", message);
    ++failures;
}

uint16_t to_half(float value) {
    return __half_as_ushort(__float2half(value));
}

float from_half(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

float from_fp8_e4m3(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
    const uint32_t exponent = (code >> 3) & 0xfu;
    const uint32_t mantissa = code & 0x7u;
    uint32_t bits;
    if (exponent != 0) {
        bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    } else if (mantissa >= 4) {
        bits = sign | (120u << 23) | ((mantissa - 4u) << 21);
    } else if (mantissa >= 2) {
        bits = sign | (119u << 23) | ((mantissa - 2u) << 22);
    } else {
        bits = sign | (mantissa == 0 ? 0u : (118u << 23));
    }
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { cudaFree(data); }
    template <typename T>
    T* allocate(size_t count) {
        if (cudaMalloc(&data, count * sizeof(T)) != cudaSuccess) return nullptr;
        return static_cast<T*>(data);
    }
};

bool check_fp16_cache(int context_len) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    const int max_context = context_len + 3;
    std::mt19937 rng(1234 + context_len);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(q_heads * head_dim);
    std::vector<uint16_t> k(context_len * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));
    DeviceBuffer dq, dk_rows, dv_rows, dk_cache, dv_cache, dout, dscores;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k_rows = dk_rows.allocate<uint16_t>(k.size());
    uint16_t* d_v_rows = dv_rows.allocate<uint16_t>(v.size());
    uint16_t* d_k_cache = dk_cache.allocate<uint16_t>(max_context * kv_heads * head_dim);
    uint16_t* d_v_cache = dv_cache.allocate<uint16_t>(max_context * kv_heads * head_dim);
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    if (!d_q || !d_k_rows || !d_v_rows || !d_k_cache || !d_v_cache || !d_out || !d_scores) return false;
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_rows, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_rows, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (!dsv4::qwen_append_kv_cache_f16_cuda(d_k_rows, d_v_rows, d_k_cache, d_v_cache,
                                              context_len, kv_heads, head_dim, 0, max_context) ||
        !dsv4::qwen_gqa_decode_attention_f16_cuda(d_q, d_k_cache, d_v_cache, d_out, d_scores,
                                                   q_heads, kv_heads, head_dim, context_len,
                                                   max_context) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 cache launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    double worst = 0.0;
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores(context_len);
        double maximum = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       from_half(k[pos * head_dim + d]);
            }
            scores[pos] = dot * scale;
            maximum = std::max(maximum, scores[pos]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < head_dim; ++d) {
            double value = 0.0;
            for (int pos = 0; pos < context_len; ++pos) value += scores[pos] * from_half(v[pos * head_dim + d]);
            const float expected = static_cast<float>(value / denominator);
            worst = std::max(worst, std::fabs(static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    if (worst > 3.0e-3) {
        fail("FP16 cache numerical check");
    } else {
        std::printf("  FP16 cache context=%d worst=%.3e\n", context_len, worst);
    }
    return true;
}

double max_half_difference(const std::vector<uint16_t>& lhs,
                          const std::vector<uint16_t>& rhs,
                          bool* finite) {
    if (lhs.size() != rhs.size()) {
        *finite = false;
        return INFINITY;
    }
    double worst = 0.0;
    *finite = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float left = from_half(lhs[i]);
        const float right = from_half(rhs[i]);
        if (!std::isfinite(left) || !std::isfinite(right)) *finite = false;
        worst = std::max(worst, std::fabs(static_cast<double>(left) - right));
    }
    return worst;
}

bool check_prefill_tiled(int seq_len, int head_dim, int position_offset) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    const int max_context = position_offset + seq_len + 3;
    std::mt19937 rng(9000 + seq_len * 17 + head_dim + position_offset);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(seq_len) * q_heads * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(max_context) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dold, dnew, dsparse;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_old = dold.allocate<uint16_t>(q.size());
    uint16_t* d_new = dnew.allocate<uint16_t>(q.size());
    uint16_t* d_sparse = dsparse.allocate<uint16_t>(q.size());
    if (!d_q || !d_k || !d_v || !d_old || !d_new || !d_sparse) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !dsv4::qwen_gqa_prefill_attention_f16_cuda(
            d_q, d_k, d_v, d_old, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !dsv4::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_new, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !dsv4::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_sparse, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context, seq_len + position_offset + 8, 0) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 tiled prefill launch");
        return true;
    }
    std::vector<uint16_t> old_output(q.size());
    std::vector<uint16_t> new_output(q.size());
    std::vector<uint16_t> sparse_output(q.size());
    if (cudaMemcpy(old_output.data(), d_old, old_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(new_output.data(), d_new, new_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sparse_output.data(), d_sparse, sparse_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 tiled prefill copy");
        return true;
    }
    bool finite = true;
    const double worst = max_half_difference(old_output, new_output, &finite);
    bool sparse_finite = true;
    const double sparse_worst = max_half_difference(old_output, sparse_output, &sparse_finite);
    if (!finite || worst > 4.0e-3 || !sparse_finite || sparse_worst > 4.0e-3) {
        fail("FP16 tiled prefill numerical check");
    } else {
        std::printf("  FP16 tiled prefill seq=%d dim=%d offset=%d worst=%.3e sparse_exact=%.3e\n",
                    seq_len, head_dim, position_offset, worst, sparse_worst);
    }
    return true;
}

bool check_verify_split(int seq_len, int head_dim, int position_offset) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    const int max_context = position_offset + seq_len + 3;
    const int splits = std::min(64, (position_offset + seq_len + 255) / 256);
    std::mt19937 rng(11000 + seq_len * 17 + head_dim + position_offset);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(seq_len) * q_heads * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(max_context) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dref, dexact, dverify, dscores, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_ref = dref.allocate<uint16_t>(q.size());
    uint16_t* d_exact = dexact.allocate<uint16_t>(q.size());
    uint16_t* d_verify = dverify.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(
        static_cast<size_t>(seq_len) * q_heads * (position_offset + seq_len));
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(seq_len) * q_heads * splits * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_ref || !d_exact || !d_verify ||
        !d_scores || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !dsv4::qwen_gqa_prefill_attention_f16_cuda(
            d_q, d_k, d_v, d_ref, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !dsv4::qwen_gqa_verify_attention_f16_exact_cuda(
            d_q, d_k, d_v, d_exact, d_scores, seq_len, q_heads, kv_heads,
            head_dim, position_offset, max_context) ||
        !dsv4::qwen_gqa_verify_attention_f16_cuda(
            d_q, d_k, d_v, d_verify, d_partial, seq_len, q_heads, kv_heads,
            head_dim, position_offset, max_context, splits) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 split verify launch");
        return true;
    }
    std::vector<uint16_t> reference(q.size());
    std::vector<uint16_t> exact(q.size());
    std::vector<uint16_t> verify(q.size());
    if (cudaMemcpy(reference.data(), d_ref, reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(exact.data(), d_exact, exact.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(verify.data(), d_verify, verify.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 split verify copy");
        return true;
    }
    bool exact_finite = true;
    const double exact_worst = max_half_difference(reference, exact, &exact_finite);
    bool finite = true;
    const double worst = max_half_difference(reference, verify, &finite);
    if (!exact_finite || exact_worst > 4.0e-3) {
        fail("FP16 exact verify numerical check");
    } else if (!finite || worst > 4.0e-3) {
        fail("FP16 split verify numerical check");
    } else {
        std::printf("  FP16 verify seq=%d dim=%d offset=%d splits=%d exact=%.3e split=%.3e\n",
                    seq_len, head_dim, position_offset, splits, exact_worst, worst);
    }
    return true;
}

bool check_decode_fused(int context_len, int head_dim) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    std::mt19937 rng(12000 + context_len + head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(context_len) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dold, dnew, dscores, dpartial, dsparse;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(k.size());
    uint16_t* d_old = dold.allocate<uint16_t>(q.size());
    uint16_t* d_new = dnew.allocate<uint16_t>(q.size());
    uint16_t* d_sparse = dsparse.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(static_cast<size_t>(q_heads) * context_len);
    const int splits = std::min(64, (context_len + 2048 - 1) / 2048);
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * splits * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_old || !d_new || !d_sparse ||
        !d_scores || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !dsv4::qwen_gqa_decode_attention_f16_cuda(
            d_q, d_k, d_v, d_old, d_scores, q_heads, kv_heads, head_dim,
            context_len, context_len) ||
        !dsv4::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_new, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len) ||
        !dsv4::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_sparse, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len, context_len + 8, 0) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 fused decode launch");
        return true;
    }
    std::vector<uint16_t> old_output(q.size());
    std::vector<uint16_t> new_output(q.size());
    std::vector<uint16_t> sparse_output(q.size());
    if (cudaMemcpy(old_output.data(), d_old, old_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(new_output.data(), d_new, new_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sparse_output.data(), d_sparse, sparse_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 fused decode copy");
        return true;
    }
    bool finite = true;
    const double worst = max_half_difference(old_output, new_output, &finite);
    bool sparse_finite = true;
    const double sparse_worst = max_half_difference(old_output, sparse_output, &sparse_finite);
    if (!finite || worst > 4.0e-3 || !sparse_finite || sparse_worst > 4.0e-3) {
        std::printf("  FP16 fused decode context=%d dim=%d worst=%.3e finite=%d\n",
                    context_len, head_dim, worst, finite ? 1 : 0);
        fail("FP16 fused decode numerical check");
    } else {
        std::printf("  FP16 fused decode context=%d dim=%d worst=%.3e sparse_exact=%.3e\n",
                    context_len, head_dim, worst, sparse_worst);
    }
    return true;
}

bool check_decode_window_reference() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 8192;
    constexpr int window = 257;
    constexpr int sink = 3;
    const int window_start = context_len - window;
    std::mt19937 rng(18000);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(context_len) * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    const int attended = sink + (context_len - window_start);
    DeviceBuffer dq, dk, dv, dout, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_out || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !dsv4::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_out, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len, window, sink) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 sparse decode launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    if (cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 sparse decode copy");
        return true;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    double worst = 0.0;
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores;
        std::vector<int> positions;
        for (int pos = 0; pos < context_len; ++pos) {
            if (pos < sink || pos >= window_start) positions.push_back(pos);
        }
        scores.reserve(positions.size());
        double maximum = -1.0e300;
        for (int pos : positions) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       from_half(k[pos * head_dim + d]);
            }
            scores.push_back(dot * scale);
            maximum = std::max(maximum, scores.back());
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < head_dim; ++d) {
            double value = 0.0;
            for (size_t i = 0; i < positions.size(); ++i) {
                value += scores[i] * from_half(v[positions[i] * head_dim + d]);
            }
            const double expected = value / denominator;
            worst = std::max(worst, std::fabs(
                static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    if (worst > 4.0e-3) fail("FP16 sparse decode reference check");
    else std::printf("  FP16 sparse decode context=%d sink=%d window=%d attended=%d worst=%.3e\n",
                     context_len, sink, window, attended, worst);
    return true;
}

bool check_decode_grid_256k() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 262144;
    DeviceBuffer dq, dk_cache, dv_cache, dout, dfused, dscores, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q_heads * head_dim);
    uint16_t* d_k_cache = dk_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_v_cache = dv_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_out = dout.allocate<uint16_t>(q_heads * head_dim);
    uint16_t* d_fused = dfused.allocate<uint16_t>(q_heads * head_dim);
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    const int splits = std::min(64, (context_len + 2048 - 1) / 2048);
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * splits * (head_dim + 2));
    if (!d_q || !d_k_cache || !d_v_cache || !d_out || !d_fused ||
        !d_scores || !d_partial) return false;
    if (cudaMemset(d_q, 0, q_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_k_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_v_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        !dsv4::qwen_gqa_decode_attention_f16_cuda(
            d_q, d_k_cache, d_v_cache, d_out, d_scores, q_heads, kv_heads,
            head_dim, context_len, context_len) ||
        !dsv4::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k_cache, d_v_cache, d_fused, d_partial, q_heads, kv_heads,
            head_dim, context_len, context_len) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 256K decode grid launch");
        return true;
    }
    std::vector<uint16_t> got(q_heads * head_dim);
    std::vector<uint16_t> fused(q_heads * head_dim);
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(fused.data(), d_fused, fused.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    for (uint16_t value : got) {
        if (value != 0) {
            fail("FP16 256K decode zero output");
            break;
        }
    }
    bool finite = true;
    const double worst = max_half_difference(got, fused, &finite);
    if (!finite || worst > 4.0e-3) fail("FP16 256K fused decode numerical check");
    std::printf("  FP16 decode context=%d grid=flattened fused worst=%.3e\n",
                context_len, worst);
    return true;
}

bool check_fp8_f16_projection() {
    std::mt19937 rng(24000);
    std::uniform_real_distribution<float> x_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);
    auto random_code = [&]() {
        // Keep the direct gate finite and away from the E4M3 NaN encodings.
        return static_cast<uint8_t>(32 + (rng() % 192));
    };
    auto max_error = [](const std::vector<uint16_t>& got,
                        const std::vector<float>& expected, int stride,
                        int rows, int batch) {
        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            for (int r = 0; r < rows; ++r) {
                worst = std::max(worst, std::fabs(
                    static_cast<double>(from_half(got[static_cast<size_t>(b) * stride + r])) -
                    expected[static_cast<size_t>(b) * rows + r]));
            }
        }
        return worst;
    };

    // Decode shape crosses the multi-row occupancy threshold (4352 rows ->
    // 136 blocks at four rows per warp) and uses a non-power-of-two block count.
    constexpr int decode_rows = 4352;
    constexpr int decode_cols = 512;
    constexpr int decode_weight_stride = 512;
    constexpr int decode_scale_stride = 4;
    std::vector<uint16_t> decode_x(decode_cols);
    std::vector<uint8_t> decode_weight(static_cast<size_t>(decode_rows) * decode_weight_stride);
    std::vector<uint16_t> decode_scale(static_cast<size_t>(decode_rows / 128) * decode_scale_stride);
    for (uint16_t& value : decode_x) value = to_half(x_dist(rng));
    for (uint8_t& value : decode_weight) value = random_code();
    for (uint16_t& value : decode_scale) value = to_half(scale_dist(rng));
    std::vector<float> decode_expected(decode_rows);
    for (int r = 0; r < decode_rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < decode_cols; ++c) {
            sum += from_half(decode_x[c]) * from_fp8_e4m3(decode_weight[
                static_cast<size_t>(r) * decode_weight_stride + c]) *
                   from_half(decode_scale[static_cast<size_t>(r / 128) * decode_scale_stride + c / 128]);
        }
        decode_expected[r] = sum;
    }
    DeviceBuffer d_decode_x, d_decode_weight, d_decode_scale, d_decode_multi,
        d_decode_vector, d_decode_fallback;
    uint16_t* dx = d_decode_x.allocate<uint16_t>(decode_x.size());
    uint8_t* dw = d_decode_weight.allocate<uint8_t>(decode_weight.size());
    uint16_t* ds = d_decode_scale.allocate<uint16_t>(decode_scale.size());
    uint16_t* dy_multi = d_decode_multi.allocate<uint16_t>(decode_rows);
    uint16_t* dy_vector = d_decode_vector.allocate<uint16_t>(decode_rows);
    uint16_t* dy_fallback = d_decode_fallback.allocate<uint16_t>(decode_rows);
    if (!dx || !dw || !ds || !dy_multi || !dy_vector || !dy_fallback ||
        cudaMemcpy(dx, decode_x.data(), decode_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dw, decode_weight.data(), decode_weight.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ds, decode_scale.data(), decode_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 decode gate allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "1", 1);
    setenv("QWEN_FP8_F16_MULTIROW_ROWS", "4", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_multi, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 multi-row decode launch");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "0", 1);
    unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
    setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_vector, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 vectorized decode launch");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "0", 1);
    setenv("QWEN_FP8_F16_VECTORIZE", "0", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_fallback, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 scalar decode launch");
        return false;
    }
    std::vector<uint16_t> decode_multi(decode_rows), decode_vector(decode_rows),
        decode_fallback(decode_rows);
    cudaMemcpy(decode_multi.data(), dy_multi, decode_multi.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(decode_vector.data(), dy_vector, decode_vector.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(decode_fallback.data(), dy_fallback, decode_fallback.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double decode_vector_dispatch_error = 0.0;
    double decode_multi_dispatch_error = 0.0;
    for (int r = 0; r < decode_rows; ++r) {
        decode_vector_dispatch_error = std::max(decode_vector_dispatch_error,
            std::fabs(static_cast<double>(from_half(decode_vector[r])) - from_half(decode_fallback[r])));
        decode_multi_dispatch_error = std::max(decode_multi_dispatch_error,
            std::fabs(static_cast<double>(from_half(decode_multi[r])) - from_half(decode_fallback[r])));
    }
    const double decode_error = max_error(decode_vector, decode_expected, 1, decode_rows, 1);
    const double decode_multi_error = max_error(decode_multi, decode_expected, 1, decode_rows, 1);
    if (decode_error > 2.0e-2 || decode_multi_error > 2.0e-2 ||
        decode_vector_dispatch_error > 2.0e-2 || decode_multi_dispatch_error > 2.0e-2) {
        fail("FP16 FP8 decode dispatch numerical check");
    } else {
        std::printf("  FP16 FP8 decode rows=%d cols=%d vector=%.3e multi=%.3e "
                    "vector_vs_scalar=%.3e multi_vs_scalar=%.3e\n",
                    decode_rows, decode_cols, decode_error, decode_multi_error,
                    decode_vector_dispatch_error, decode_multi_dispatch_error);
    }

    // Small speculative batches reuse each FP8 weight across all input rows.
    constexpr int small_batch = 5;
    constexpr int small_rows = 130;
    constexpr int small_cols = 300;
    constexpr int small_x_stride = 304;
    constexpr int small_y_stride = 136;
    constexpr int small_weight_stride = 304;
    constexpr int small_scale_stride = 3;
    std::vector<uint16_t> small_x(
        static_cast<size_t>(small_batch) * small_x_stride);
    std::vector<uint8_t> small_weight(
        static_cast<size_t>(small_rows) * small_weight_stride);
    std::vector<uint16_t> small_scale(
        static_cast<size_t>((small_rows + 127) / 128) * small_scale_stride);
    for (uint16_t& value : small_x) value = to_half(x_dist(rng));
    for (uint8_t& value : small_weight) value = random_code();
    for (uint16_t& value : small_scale) value = to_half(scale_dist(rng));
    DeviceBuffer d_small_x, d_small_weight, d_small_scale, d_small_reuse,
        d_small_tiled;
    uint16_t* sx = d_small_x.allocate<uint16_t>(small_x.size());
    uint8_t* sw = d_small_weight.allocate<uint8_t>(small_weight.size());
    uint16_t* ss = d_small_scale.allocate<uint16_t>(small_scale.size());
    uint16_t* sy_reuse = d_small_reuse.allocate<uint16_t>(
        static_cast<size_t>(small_batch) * small_y_stride);
    uint16_t* sy_tiled = d_small_tiled.allocate<uint16_t>(
        static_cast<size_t>(small_batch) * small_y_stride);
    if (!sx || !sw || !ss || !sy_reuse || !sy_tiled ||
        cudaMemcpy(sx, small_x.data(), small_x.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(sw, small_weight.data(), small_weight.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ss, small_scale.data(), small_scale.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 small-batch allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_SMALL_BATCH", "1", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            sx, sw, ss, sy_reuse, small_batch, small_rows, small_cols,
            small_x_stride, small_y_stride, small_weight_stride,
            small_scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 small-batch reuse launch");
        return false;
    }
    setenv("QWEN_FP8_F16_SMALL_BATCH", "0", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            sx, sw, ss, sy_tiled, small_batch, small_rows, small_cols,
            small_x_stride, small_y_stride, small_weight_stride,
            small_scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 small-batch tiled launch");
        return false;
    }
    std::vector<uint16_t> small_reuse(
        static_cast<size_t>(small_batch) * small_y_stride);
    std::vector<uint16_t> small_tiled(small_reuse.size());
    cudaMemcpy(small_reuse.data(), sy_reuse,
               small_reuse.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(small_tiled.data(), sy_tiled,
               small_tiled.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double small_dispatch_error = 0.0;
    for (int sample = 0; sample < small_batch; ++sample) {
        for (int row = 0; row < small_rows; ++row) {
            const size_t at = static_cast<size_t>(sample) * small_y_stride + row;
            small_dispatch_error = std::max(
                small_dispatch_error,
                std::fabs(static_cast<double>(from_half(small_reuse[at])) -
                          from_half(small_tiled[at])));
        }
    }
    unsetenv("QWEN_FP8_F16_SMALL_BATCH");
    if (small_dispatch_error > 3.0e-2) {
        fail("FP16 FP8 small-batch numerical check");
    } else {
        std::printf("  FP16 FP8 small batch=%d rows=%d cols=%d "
                    "reuse_vs_tiled=%.3e\n",
                    small_batch, small_rows, small_cols,
                    small_dispatch_error);
    }

    // Prefill shape triggers the 128-token x 64-row N64 tile and has padded
    // strides so both aligned vector loads and output indexing are exercised.
    constexpr int batch = 128;
    constexpr int rows = 130;
    constexpr int cols = 300;
    constexpr int x_stride = 304;
    constexpr int y_stride = 136;
    constexpr int weight_stride = 304;
    constexpr int scale_stride = 3;
    std::vector<uint16_t> prefill_x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint8_t> prefill_weight(static_cast<size_t>(rows) * weight_stride);
    std::vector<uint16_t> prefill_scale(static_cast<size_t>((rows + 127) / 128) * scale_stride);
    for (uint16_t& value : prefill_x) value = to_half(x_dist(rng));
    for (uint8_t& value : prefill_weight) value = random_code();
    for (uint16_t& value : prefill_scale) value = to_half(scale_dist(rng));
    std::vector<float> prefill_expected(static_cast<size_t>(batch) * rows);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                sum += from_half(prefill_x[static_cast<size_t>(b) * x_stride + c]) *
                       from_fp8_e4m3(prefill_weight[static_cast<size_t>(r) * weight_stride + c]) *
                       from_half(prefill_scale[static_cast<size_t>(r / 128) * scale_stride + c / 128]);
            }
            prefill_expected[static_cast<size_t>(b) * rows + r] = sum;
        }
    }
    DeviceBuffer d_prefill_x, d_prefill_weight, d_prefill_scale, d_prefill_wide, d_prefill_fallback;
    uint16_t* px = d_prefill_x.allocate<uint16_t>(prefill_x.size());
    uint8_t* pw = d_prefill_weight.allocate<uint8_t>(prefill_weight.size());
    uint16_t* ps = d_prefill_scale.allocate<uint16_t>(prefill_scale.size());
    uint16_t* py_wide = d_prefill_wide.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    uint16_t* py_fallback = d_prefill_fallback.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    if (!px || !pw || !ps || !py_wide || !py_fallback ||
        cudaMemcpy(px, prefill_x.data(), prefill_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(pw, prefill_weight.data(), prefill_weight.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ps, prefill_scale.data(), prefill_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 wide prefill allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_PREFILL_WIDE_N64", "1", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            px, pw, ps, py_wide, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 wide prefill launch");
        return false;
    }
    setenv("QWEN_FP8_F16_PREFILL_WIDE_N64", "0", 1);
    if (!dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            px, pw, ps, py_fallback, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 fallback prefill launch");
        return false;
    }
    std::vector<uint16_t> prefill_wide(static_cast<size_t>(batch) * y_stride);
    std::vector<uint16_t> prefill_fallback(prefill_wide.size());
    cudaMemcpy(prefill_wide.data(), py_wide, prefill_wide.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(prefill_fallback.data(), py_fallback, prefill_fallback.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double prefill_dispatch_error = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            prefill_dispatch_error = std::max(prefill_dispatch_error,
                std::fabs(static_cast<double>(from_half(prefill_wide[static_cast<size_t>(b) * y_stride + r])) -
                          from_half(prefill_fallback[static_cast<size_t>(b) * y_stride + r])));
        }
    }
    const double prefill_error = max_error(prefill_wide, prefill_expected, y_stride, rows, batch);
    unsetenv("QWEN_FP8_F16_MULTIROW");
    unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
    unsetenv("QWEN_FP8_F16_PREFILL_WIDE_N64");
    if (prefill_error > 3.0e-2 || prefill_dispatch_error > 3.0e-2) {
        fail("FP16 FP8 wide prefill numerical check");
    } else {
        std::printf("  FP16 FP8 prefill batch=%d rows=%d cols=%d error=%.3e dispatch=%.3e\n",
                    batch, rows, cols, prefill_error, prefill_dispatch_error);
    }
    return true;
}

bool check_batched_argmax() {
    constexpr int rows = 5;
    constexpr int count = 263;
    constexpr int token_offset = 1000;
    std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
    logits[7] = 3.0f;
    logits[2] = 2.5f;
    logits[static_cast<size_t>(count) + 5] = 4.0f;
    logits[static_cast<size_t>(count) + 9] = 4.0f;
    logits[static_cast<size_t>(2) * count + 262] = 9.0f;
    logits[static_cast<size_t>(2) * count + 3] = 8.0f;
    logits[static_cast<size_t>(3) * count] = -1.0f;
    logits[static_cast<size_t>(3) * count + 11] = -2.0f;
    logits[static_cast<size_t>(4) * count + 129] = 0.5f;
    logits[static_cast<size_t>(4) * count + 130] = 0.25f;
    DeviceBuffer d_logits, d_tokens, d_values;
    float* device_logits = d_logits.allocate<float>(logits.size());
    int* device_tokens = d_tokens.allocate<int>(rows);
    float* device_values = d_values.allocate<float>(rows);
    if (!device_logits || !device_tokens || !device_values ||
        cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !dsv4::argmax_fp32_rows_cuda(device_logits, device_tokens, device_values,
                                     rows, count, token_offset) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched argmax launch");
        return false;
    }
    std::vector<int> tokens(rows);
    std::vector<float> values(rows);
    if (cudaMemcpy(tokens.data(), device_tokens, rows * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(values.data(), device_values, rows * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("batched argmax copy");
        return false;
    }
    const std::vector<int> expected_tokens = {1007, 1005, 1262, 1000, 1129};
    const std::vector<float> expected_values = {3.0f, 4.0f, 9.0f, -1.0f, 0.5f};
    if (tokens != expected_tokens || values != expected_values) {
        fail("batched argmax numerical/tie check");
    } else {
        std::printf("  batched argmax rows=%d count=%d tie=lower-token\n",
                    rows, count);
    }
    return true;
}

bool check_fp8_cache() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 17;
    constexpr int max_context = 32;
    constexpr int scale_block = 64;
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<uint16_t> q(q_heads * head_dim);
    std::vector<uint16_t> k(context_len * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));
    DeviceBuffer dq, dk_rows, dv_rows, dk_cache, dv_cache, dks, dvs, dout, dscores;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k_rows = dk_rows.allocate<uint16_t>(k.size());
    uint16_t* d_v_rows = dv_rows.allocate<uint16_t>(v.size());
    uint8_t* d_k_cache = dk_cache.allocate<uint8_t>(max_context * head_dim);
    uint8_t* d_v_cache = dv_cache.allocate<uint8_t>(max_context * head_dim);
    uint16_t* d_k_scale = dks.allocate<uint16_t>(max_context);
    uint16_t* d_v_scale = dvs.allocate<uint16_t>(max_context);
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    if (!d_q || !d_k_rows || !d_v_rows || !d_k_cache || !d_v_cache || !d_k_scale || !d_v_scale || !d_out || !d_scores) return false;
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_rows, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_rows, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (!dsv4::qwen_append_kv_cache_fp8_cuda(d_k_rows, d_v_rows, d_k_cache, d_v_cache,
                                               d_k_scale, d_v_scale, context_len,
                                               kv_heads, head_dim, scale_block, 0,
                                               max_context) ||
        !dsv4::qwen_gqa_decode_attention_fp8_cuda(d_q, d_k_cache, d_v_cache,
                                                   d_k_scale, d_v_scale, d_out,
                                                   d_scores, q_heads, kv_heads,
                                                   head_dim, scale_block,
                                                   context_len, max_context) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP8 cache launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    std::vector<uint8_t> kq(k.size()), vq(v.size());
    std::vector<uint16_t> ks(context_len), vs(context_len);
    cudaMemcpy(ks.data(), d_k_scale, ks.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(vs.data(), d_v_scale, vs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(kq.data(), d_k_cache, kq.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(vq.data(), d_v_cache, vq.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores(context_len);
        double maximum = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            double dot = 0.0;
            const float key_scale = from_half(ks[pos]);
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       (from_fp8_e4m3(kq[pos * head_dim + d]) * key_scale);
            }
            scores[pos] = dot * query_scale;
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
                value += scores[pos] * (from_fp8_e4m3(vq[pos * head_dim + d]) * from_half(vs[pos]));
            }
            const float expected = static_cast<float>(value / denominator);
            worst = std::max(worst, std::fabs(static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    // The quantization itself is checked against the exact max-abs scale.
    for (int pos = 0; pos < context_len; ++pos) {
        float max_k = 0.0f;
        float max_v = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            max_k = std::max(max_k, std::fabs(from_half(k[pos * head_dim + d])));
            max_v = std::max(max_v, std::fabs(from_half(v[pos * head_dim + d])));
        }
        if (std::fabs(from_half(ks[pos]) - (max_k > 0.0f ? max_k / 448.0f : 1.0f)) > 2.0e-3f ||
            std::fabs(from_half(vs[pos]) - (max_v > 0.0f ? max_v / 448.0f : 1.0f)) > 2.0e-3f) {
            fail("FP8 cache scale check");
            break;
        }
    }
    if (worst > 0.35) fail("FP8 cache numerical check");
    else std::printf("  FP8 cache context=%d worst=%.3e\n", context_len, worst);
    return true;
}

}  // namespace

int main() {
    if (!dsv4::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_half_ops requires a CUDA device\n");
        return 0;
    }
    check_fp16_cache(1);
    check_fp16_cache(17);
    check_fp16_cache(333);
    check_prefill_tiled(1, 64, 0);
    check_prefill_tiled(2, 256, 0);
    check_prefill_tiled(7, 64, 3);
    check_prefill_tiled(17, 256, 5);
    check_prefill_tiled(128, 64, 7);
    check_prefill_tiled(333, 256, 11);
    check_verify_split(2, 64, 0);
    check_verify_split(3, 256, 37);
    check_verify_split(5, 64, 4093);
    check_verify_split(8, 256, 8191);
    check_decode_fused(4096, 64);
    check_decode_fused(8192, 256);
    check_decode_fused(32768, 64);
    check_decode_window_reference();
    check_decode_grid_256k();
    check_fp8_f16_projection();
    check_batched_argmax();
    check_fp8_cache();
    if (failures != 0) {
        std::printf("test_qwen_half_ops failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_half_ops\n");
    return 0;
}
