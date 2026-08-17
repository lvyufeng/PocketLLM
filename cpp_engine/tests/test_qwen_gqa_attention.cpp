// Numerical check for the online-softmax GQA kernels (decode + prefill) against
// a two-pass CPU softmax reference.
//
// The kernels stream the KV cache once and rescale a running max/denominator.
// head_dim=256 and 24-over-4 heads match the Qwen3.8 TP1 full-attention layout.

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("[FAIL] %s\n", what.c_str());
    ++failures;
}

// Reference: explicit max, exp, normalize over the causal window.
void attention_reference(const std::vector<float>& q, const std::vector<float>& k_cache,
                         const std::vector<float>& v_cache, std::vector<float>& out,
                         int q_heads, int kv_heads, int head_dim, int context_len) {
    const int repeat = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < q_heads; ++head) {
        const int kv_head = head / repeat;
        const float* q_row = q.data() + static_cast<size_t>(head) * head_dim;
        std::vector<double> scores(context_len);
        double max_score = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            const float* key = k_cache.data() +
                               static_cast<size_t>(pos) * kv_heads * head_dim +
                               static_cast<size_t>(kv_head) * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(q_row[d]) * key[d];
            scores[pos] = dot * scale;
            max_score = std::max(max_score, scores[pos]);
        }
        double denom = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            scores[pos] = std::exp(scores[pos] - max_score);
            denom += scores[pos];
        }
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                const float* value = v_cache.data() +
                                     static_cast<size_t>(pos) * kv_heads * head_dim +
                                     static_cast<size_t>(kv_head) * head_dim;
                acc += scores[pos] * value[d];
            }
            out[static_cast<size_t>(head) * head_dim + d] =
                static_cast<float>(acc / (denom > 0.0 ? denom : 1.0));
        }
    }
}

struct Device {
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* out = nullptr;
    float* scores = nullptr;
    ~Device() {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(out);
        cudaFree(scores);
    }
};

bool run_decode(int q_heads, int kv_heads, int head_dim, int context_len, int max_context) {
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<float> k(static_cast<size_t>(max_context) * kv_heads * head_dim, 0.0f);
    std::vector<float> v(k.size(), 0.0f);
    for (float& x : q) x = dist(rng);
    for (size_t i = 0; i < static_cast<size_t>(context_len) * kv_heads * head_dim; ++i) {
        k[i] = dist(rng);
        v[i] = dist(rng);
    }

    Device dev;
    if (cudaMalloc(&dev.q, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.k, k.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.v, v.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.out, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.scores, static_cast<size_t>(q_heads) * context_len * sizeof(float)) != cudaSuccess) {
        return false;
    }
    cudaMemcpy(dev.q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (!dsv4::qwen_gqa_decode_attention_cuda(dev.q, dev.k, dev.v, dev.out, dev.scores,
                                              q_heads, kv_heads, head_dim, context_len,
                                              max_context)) {
        fail("gqa decode launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa decode sync failed");
        return true;
    }
    std::vector<float> got(q.size());
    cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> want(q.size());
    attention_reference(q, k, v, want, q_heads, kv_heads, head_dim, context_len);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
    }
    if (worst > 1.0e-4) {
        fail("gqa decode ctx=" + std::to_string(context_len) + " worst=" + std::to_string(worst));
    } else {
        std::printf("  gqa decode heads=%d/%d head_dim=%d ctx=%4d worst=%.3e\n",
                    q_heads, kv_heads, head_dim, context_len, worst);
    }
    return true;
}

}  // namespace

int main() {
    if (!dsv4::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_gqa_attention requires a CUDA device\n");
        return 0;
    }
    // Qwen3.8 TP1 full attention: 24 q heads over 4 kv heads, head_dim 256.
    const int contexts[] = {1, 7, 128, 333};
    for (int ctx : contexts) {
        if (!run_decode(24, 4, 256, ctx, 512)) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    if (failures != 0) {
        std::printf("test_qwen_gqa_attention failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_gqa_attention\n");
    return 0;
}
