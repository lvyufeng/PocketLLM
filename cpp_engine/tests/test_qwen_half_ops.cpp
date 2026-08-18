#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

bool check_decode_grid_256k() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 262144;
    DeviceBuffer dq, dk_cache, dv_cache, dout, dscores;
    uint16_t* d_q = dq.allocate<uint16_t>(q_heads * head_dim);
    uint16_t* d_k_cache = dk_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_v_cache = dv_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_out = dout.allocate<uint16_t>(q_heads * head_dim);
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    if (!d_q || !d_k_cache || !d_v_cache || !d_out || !d_scores) return false;
    if (cudaMemset(d_q, 0, q_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_k_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_v_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        !dsv4::qwen_gqa_decode_attention_f16_cuda(
            d_q, d_k_cache, d_v_cache, d_out, d_scores, q_heads, kv_heads,
            head_dim, context_len, context_len) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 256K decode grid launch");
        return true;
    }
    std::vector<uint16_t> got(q_heads * head_dim);
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    for (uint16_t value : got) {
        if (value != 0) {
            fail("FP16 256K decode zero output");
            break;
        }
    }
    std::printf("  FP16 decode context=%d grid=flattened PASS\n", context_len);
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
    check_decode_grid_256k();
    check_fp8_cache();
    if (failures != 0) {
        std::printf("test_qwen_half_ops failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_half_ops\n");
    return 0;
}
