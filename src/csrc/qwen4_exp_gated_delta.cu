#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kDim = 128;
constexpr int kCols = 4;
constexpr int kWidth = 16;
constexpr int kRowsPerLane = kDim / kWidth;
constexpr int kSubgroupsPerWarp = 32 / kWidth;

__device__ __forceinline__ float subgroup_sum(float value) {
#pragma unroll
    for (int offset = kWidth / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset, kWidth);
    }
    return value;
}

__device__ __forceinline__ float subgroup_broadcast(float value) {
    return __shfl_sync(0xffffffffU, value, 0, kWidth);
}

__global__ void normalize_qk_bf16_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    float* __restrict__ query_normalized,
    float* __restrict__ key_normalized,
    int rows,
    int key_heads,
    float eps) {
    const int vector = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (vector >= rows * key_heads) {
        return;
    }

    const int64_t base = static_cast<int64_t>(vector) * kDim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
#pragma unroll
    for (int col = lane; col < kDim; col += 32) {
        const float q = static_cast<float>(query[base + col]);
        const float k = static_cast<float>(key[base + col]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffU, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffU, k_sum, offset);
    }
    const float q_inv = rsqrtf(__shfl_sync(0xffffffffU, q_sum, 0) + eps);
    const float k_inv = rsqrtf(__shfl_sync(0xffffffffU, k_sum, 0) + eps);
#pragma unroll
    for (int col = lane; col < kDim; col += 32) {
        query_normalized[base + col] = static_cast<float>(query[base + col]) * q_inv;
        key_normalized[base + col] = static_cast<float>(key[base + col]) * k_inv;
    }
}

template <int GroupsPerBlock>
__global__ void gated_delta_bf16_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const float* __restrict__ gate,
    const c10::BFloat16* __restrict__ beta,
    float* __restrict__ state,
    c10::BFloat16* __restrict__ output,
    int rows,
    int key_heads,
    int value_heads,
    float query_scale) {
    const int value_head = static_cast<int>(blockIdx.x);
    const int subgroup = static_cast<int>(threadIdx.x) / kWidth;
    const int lane = static_cast<int>(threadIdx.x) % kWidth;
    const int group =
        (static_cast<int>(blockIdx.z) * GroupsPerBlock +
         static_cast<int>(threadIdx.y)) *
            kSubgroupsPerWarp +
        subgroup;
    const int col_base = group * kCols;
    const int key_head = value_head / (value_heads / key_heads);

    float state_shard[kCols][kRowsPerLane];
#pragma unroll
    for (int col_slot = 0; col_slot < kCols; ++col_slot) {
        const int col = col_base + col_slot;
#pragma unroll
        for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
            const int row = row_slot * kWidth + lane;
            float state_value = 0.0f;
            if (col < kDim) {
                const int64_t index =
                    (static_cast<int64_t>(value_head) * kDim + row) * kDim + col;
                state_value = state[index];
            }
            state_shard[col_slot][row_slot] = state_value;
        }
    }

    float query_next[kRowsPerLane];
    float key_next[kRowsPerLane];
    float value_next[kCols];
    float decay_next = 0.0f;
    float beta_next = 0.0f;

    auto load_qk = [&](int token, float* query_out, float* key_out) {
#pragma unroll
        for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
            const int row = row_slot * kWidth + lane;
            const int64_t index =
                (static_cast<int64_t>(token) * key_heads + key_head) * kDim + row;
            query_out[row_slot] = query[index];
            key_out[row_slot] = key[index];
        }
    };
    auto load_value = [&](int token, float* value_out) {
#pragma unroll
        for (int col_slot = 0; col_slot < kCols; ++col_slot) {
            float value_scalar = 0.0f;
            if (lane == 0) {
                const int col = col_base + col_slot;
                if (col < kDim) {
                    const int64_t index =
                        (static_cast<int64_t>(token) * value_heads + value_head) *
                            kDim +
                        col;
                    value_scalar = static_cast<float>(value[index]);
                }
            }
            value_out[col_slot] = value_scalar;
        }
    };
    auto load_scalars = [&](int token, float* decay_out, float* beta_out) {
        float decay = 0.0f;
        float beta_scalar = 0.0f;
        if (threadIdx.x == 0) {
            const int64_t index =
                static_cast<int64_t>(token) * value_heads + value_head;
            decay = expf(gate[index]);
            beta_scalar = static_cast<float>(beta[index]);
        }
        *decay_out = decay;
        *beta_out = beta_scalar;
    };

    load_qk(0, query_next, key_next);
    load_value(0, value_next);
    load_scalars(0, &decay_next, &beta_next);

    for (int token = 0; token < rows; ++token) {
        float query_row[kRowsPerLane];
        float key_row[kRowsPerLane];
        float value_row[kCols];
#pragma unroll
        for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
            query_row[row_slot] = query_next[row_slot];
            key_row[row_slot] = key_next[row_slot];
        }
#pragma unroll
        for (int col_slot = 0; col_slot < kCols; ++col_slot) {
            value_row[col_slot] = value_next[col_slot];
        }
        float decay = decay_next;
        float beta_scalar = beta_next;

        if (token + 1 < rows) {
            load_qk(token + 1, query_next, key_next);
            load_value(token + 1, value_next);
            load_scalars(token + 1, &decay_next, &beta_next);
        }

        decay = __shfl_sync(0xffffffffU, decay, 0);
        beta_scalar = __shfl_sync(0xffffffffU, beta_scalar, 0);

        float memory[kCols];
#pragma unroll
        for (int col_slot = 0; col_slot < kCols; ++col_slot) {
            float sum = 0.0f;
#pragma unroll
            for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
                sum = fmaf(
                    state_shard[col_slot][row_slot], key_row[row_slot], sum);
            }
            memory[col_slot] = subgroup_sum(sum);
        }

        float delta[kCols];
#pragma unroll
        for (int col_slot = 0; col_slot < kCols; ++col_slot) {
            float delta_scalar = 0.0f;
            if (lane == 0 && col_base + col_slot < kDim) {
                delta_scalar =
                    (value_row[col_slot] - decay * memory[col_slot]) * beta_scalar;
            }
            delta[col_slot] = subgroup_broadcast(delta_scalar);
        }

        float attention[kCols];
#pragma unroll
        for (int col_slot = 0; col_slot < kCols; ++col_slot) {
            float sum = 0.0f;
#pragma unroll
            for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
                const float new_state = fmaf(
                    key_row[row_slot], delta[col_slot],
                    decay * state_shard[col_slot][row_slot]);
                state_shard[col_slot][row_slot] = new_state;
                sum = fmaf(new_state, query_row[row_slot], sum);
            }
            attention[col_slot] = subgroup_sum(sum);
        }

        if (lane == 0) {
            const int64_t base =
                (static_cast<int64_t>(token) * value_heads + value_head) * kDim;
#pragma unroll
            for (int col_slot = 0; col_slot < kCols; ++col_slot) {
                const int col = col_base + col_slot;
                if (col < kDim) {
                    output[base + col] =
                        c10::BFloat16(attention[col_slot] * query_scale);
                }
            }
        }
    }

#pragma unroll
    for (int col_slot = 0; col_slot < kCols; ++col_slot) {
        const int col = col_base + col_slot;
#pragma unroll
        for (int row_slot = 0; row_slot < kRowsPerLane; ++row_slot) {
            const int row = row_slot * kWidth + lane;
            if (col < kDim) {
                const int64_t index =
                    (static_cast<int64_t>(value_head) * kDim + row) * kDim + col;
                state[index] = state_shard[col_slot][row_slot];
            }
        }
    }
}

template <int GroupsPerBlock>
void launch_gated_delta(
    const float* query,
    const float* key,
    const c10::BFloat16* value,
    const float* gate,
    const c10::BFloat16* beta,
    float* state,
    c10::BFloat16* output,
    int rows,
    int key_heads,
    int value_heads,
    float query_scale,
    cudaStream_t stream) {
    constexpr int groups = kDim / kCols;
    constexpr int groups_per_cta = GroupsPerBlock * kSubgroupsPerWarp;
    const dim3 block(32, GroupsPerBlock);
    const dim3 grid(
        value_heads,
        1,
        (groups + groups_per_cta - 1) / groups_per_cta);
    gated_delta_bf16_kernel<GroupsPerBlock><<<grid, block, 0, stream>>>(
        query,
        key,
        value,
        gate,
        beta,
        state,
        output,
        rows,
        key_heads,
        value_heads,
        query_scale);
}

}  // namespace

std::vector<torch::Tensor> qwen4_exp_gated_delta_bf16_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    double norm_eps,
    int64_t groups_per_block) {
    c10::cuda::CUDAGuard device_guard(query.device());

    const int rows = static_cast<int>(query.size(1));
    const int key_heads = static_cast<int>(query.size(2));
    const int value_heads = static_cast<int>(value.size(2));
    auto query_normalized = torch::empty(query.sizes(), query.options().dtype(torch::kFloat32));
    auto key_normalized = torch::empty(key.sizes(), key.options().dtype(torch::kFloat32));
    auto output = torch::empty(value.sizes(), value.options());
    auto state = initial_state.clone();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    normalize_qk_bf16_kernel<<<rows * key_heads, 32, 0, stream>>>(
        query.data_ptr<c10::BFloat16>(),
        key.data_ptr<c10::BFloat16>(),
        query_normalized.data_ptr<float>(),
        key_normalized.data_ptr<float>(),
        rows,
        key_heads,
        static_cast<float>(norm_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float query_scale = 1.0f / std::sqrt(static_cast<float>(kDim));
    switch (groups_per_block) {
        case 2:
            launch_gated_delta<2>(
                query_normalized.data_ptr<float>(),
                key_normalized.data_ptr<float>(),
                value.data_ptr<c10::BFloat16>(),
                gate.data_ptr<float>(),
                beta.data_ptr<c10::BFloat16>(),
                state.data_ptr<float>(),
                output.data_ptr<c10::BFloat16>(),
                rows,
                key_heads,
                value_heads,
                query_scale,
                stream);
            break;
        case 4:
            launch_gated_delta<4>(
                query_normalized.data_ptr<float>(),
                key_normalized.data_ptr<float>(),
                value.data_ptr<c10::BFloat16>(),
                gate.data_ptr<float>(),
                beta.data_ptr<c10::BFloat16>(),
                state.data_ptr<float>(),
                output.data_ptr<c10::BFloat16>(),
                rows,
                key_heads,
                value_heads,
                query_scale,
                stream);
            break;
        case 8:
            launch_gated_delta<8>(
                query_normalized.data_ptr<float>(),
                key_normalized.data_ptr<float>(),
                value.data_ptr<c10::BFloat16>(),
                gate.data_ptr<float>(),
                beta.data_ptr<c10::BFloat16>(),
                state.data_ptr<float>(),
                output.data_ptr<c10::BFloat16>(),
                rows,
                key_heads,
                value_heads,
                query_scale,
                stream);
            break;
        default:
            launch_gated_delta<1>(
                query_normalized.data_ptr<float>(),
                key_normalized.data_ptr<float>(),
                value.data_ptr<c10::BFloat16>(),
                gate.data_ptr<float>(),
                beta.data_ptr<c10::BFloat16>(),
                state.data_ptr<float>(),
                output.data_ptr<c10::BFloat16>(),
                rows,
                key_heads,
                value_heads,
                query_scale,
                stream);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, state};
}
