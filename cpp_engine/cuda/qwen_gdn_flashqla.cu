// FlashQLA SM75 Gated DeltaNet kernel — subgroup-sharded state for D=128.
// Ported from vLLM-2080Ti-Definitive v0.1.15 tools/flashqla_sm75_patches/gdn_forward.cu
// for comparison against the baseline serial recurrence in qwen_attention_ops.cu.
//
// Key differences from the baseline:
// - State sharding: each subgroup (WIDTH=16 lanes) holds COLS=4 columns of the
//   [D, D] state matrix as `state_shard[COLS][rows_per_lane]`, rather than one
//   thread per value dimension holding the full key_dim-length state vector.
// - Launch geometry: grid.z dimension fans out when D/COLS exceeds the CTA count,
//   and blockDim.y gives multiple column-groups per CTA.
// - Still a serial `for (int t = 0; t < tokens; ++t)` recurrence — not chunked.
//   Occupancy and serialization remain, but memory access is better.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dsv4 {

__device__ __forceinline__ float warp_reduce_sum(float value, int width) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset, width);
    }
    return value;
}

__device__ __forceinline__ float warp_broadcast_lane0(float value, int width) {
    return __shfl_sync(0xffffffffU, value, 0, width);
}

// D: key_dim and value_dim (compile-time 128)
// COLS: state columns per subgroup (4 for D=128)
// WIDTH: subgroup size (16 for D=128)
template <int D, int COLS, int WIDTH>
__global__ void gdn_flashqla_kernel(
    const float* __restrict__ q_normalized,
    const float* __restrict__ k_normalized,
    const __half* __restrict__ v,
    const __half* __restrict__ gate,
    const __half* __restrict__ beta,
    const float* __restrict__ initial_state,
    __half* __restrict__ output,
    float* __restrict__ final_state,
    int tokens,
    int q_heads,
    int v_heads,
    float q_scale
) {
    static_assert(D == 128 && COLS == 4 && WIDTH == 16, "D=128 COLS=4 WIDTH=16 only");
    static_assert(D % (COLS * (32 / WIDTH)) == 0, "D must divide evenly");

    constexpr int subgroups_per_warp = 32 / WIDTH;  // 2
    constexpr int rows_per_lane = (D + WIDTH - 1) / WIDTH;  // 8

    const int hv = static_cast<int>(blockIdx.x);
    const int subgroup = threadIdx.x / WIDTH;
    const int lane = threadIdx.x % WIDTH;
    const int group_base =
        (static_cast<int>(blockIdx.z) * static_cast<int>(blockDim.y) +
         static_cast<int>(threadIdx.y)) * subgroups_per_warp + subgroup;
    const int col_base = group_base * COLS;
    const int hq = hv / (v_heads / q_heads);

    // Load initial state for this subgroup's column shard.
    float state_shard[COLS][rows_per_lane];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            float value = 0.0f;
            if (row < D && col < D) {
                const size_t state_index =
                    static_cast<size_t>(hv) * D * D +
                    static_cast<size_t>(row) * D +
                    static_cast<size_t>(col);
                value = initial_state == nullptr ? 0.0f : initial_state[state_index];
            }
            state_shard[c][r] = value;
        }
    }

    // Serial recurrence over tokens.
    for (int t = 0; t < tokens; ++t) {
        // Broadcast gate and beta to all threads.
        float gate_value = 0.0f;
        float beta_value = 0.0f;
        // Each y-row is an independent warp processing a different state-column
        // group, so lane 0 of every warp must load the per-head gate and beta.
        if (threadIdx.x == 0) {
            gate_value = expf(__half2float(gate[static_cast<size_t>(t) * v_heads + hv]));
            beta_value = __half2float(beta[static_cast<size_t>(t) * v_heads + hv]);
        }
        gate_value = warp_broadcast_lane0(gate_value, 32);
        beta_value = warp_broadcast_lane0(beta_value, 32);

        // Load pre-normalized Q and K for this position. Q and K share the key
        // head hq under GQA; the key row stride is q_heads * D.
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            float q_value = 0.0f;
            float k_value = 0.0f;
            if (row < D) {
                const size_t qk_index =
                    static_cast<size_t>(t) * q_heads * D +
                    static_cast<size_t>(hq) * D +
                    static_cast<size_t>(row);
                q_value = q_normalized[qk_index];
                k_value = k_normalized[qk_index];
            }
            q_reg[r] = q_value;
            k_reg[r] = k_value;
        }

        // Compute K · state for each column: sum over rows (subgroup reduction).
        float kv_partial[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                sum += state_shard[c][r] * k_reg[r];
            }
            kv_partial[c] = warp_reduce_sum(sum, WIDTH);
        }

        // Compute delta = (v - gate * K·state) * beta for each column.
        // Lane 0 loads v[col], broadcasts delta.
        float delta[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float delta_value = 0.0f;
            if (lane == 0) {
                const int col = col_base + c;
                if (col < D) {
                    const size_t v_index =
                        static_cast<size_t>(t) * v_heads * D +
                        static_cast<size_t>(hv) * D +
                        static_cast<size_t>(col);
                    delta_value = (__half2float(v[v_index]) - gate_value * kv_partial[c]) *
                                  beta_value;
                }
            }
            delta[c] = warp_broadcast_lane0(delta_value, WIDTH);
        }

        // Update state: state = gate * state + K ⊗ delta, and accumulate Q · state.
        float attn_partial[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const float new_state =
                    fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
                state_shard[c][r] = new_state;
                sum += new_state * q_reg[r];
            }
            attn_partial[c] = warp_reduce_sum(sum, WIDTH);
        }

        // Lane 0 writes output for its columns.
        if (lane == 0) {
            const size_t out_base = static_cast<size_t>(t) * v_heads * D +
                                    static_cast<size_t>(hv) * D;
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = col_base + c;
                if (col < D) {
                    output[out_base + col] = __float2half(attn_partial[c] * q_scale);
                }
            }
        }
    }

    // Store final state for this subgroup's shard.
    if (final_state != nullptr) {
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int col = col_base + c;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int row = r * WIDTH + lane;
                if (row < D && col < D) {
                    const size_t state_index =
                        static_cast<size_t>(hv) * D * D +
                        static_cast<size_t>(row) * D +
                        static_cast<size_t>(col);
                    final_state[state_index] = state_shard[c][r];
                }
            }
        }
    }
}

bool qwen_gated_delta_flashqla_sm75_f16_cuda(
    float* d_state,
    const float* d_q_normalized,
    const float* d_k_normalized,
    const uint16_t* d_v,
    const uint16_t* d_g,
    const uint16_t* d_beta,
    uint16_t* d_out,
    int rows,
    int heads,
    int key_heads,
    int key_dim,
    int value_dim,
    float q_scale,
    void* stream
) {
    if (!d_state || !d_q_normalized || !d_k_normalized || !d_v || !d_g ||
        !d_beta || !d_out || rows <= 0 || heads <= 0 || key_heads <= 0 ||
        heads % key_heads != 0 || key_dim != 128 || value_dim != 128 ||
        q_scale <= 0.0f) {
        return false;
    }

    constexpr int D = 128;
    constexpr int COLS = 4;
    constexpr int WIDTH = 16;
    constexpr int subgroups_per_warp = 32 / WIDTH;  // 2
    constexpr int column_groups_per_block = 8;

    const int groups = D / COLS;  // 32
    const int z = (groups + column_groups_per_block * subgroups_per_warp - 1) /
                  (column_groups_per_block * subgroups_per_warp);

    const dim3 block(32, column_groups_per_block);  // 32 threads (1 warp), 8 y-groups
    const dim3 grid(heads, 1, z);

    const cudaStream_t s = stream == nullptr ? nullptr : static_cast<cudaStream_t>(stream);

    gdn_flashqla_kernel<D, COLS, WIDTH><<<grid, block, 0, s>>>(
        d_q_normalized,
        d_k_normalized,
        reinterpret_cast<const __half*>(d_v),
        reinterpret_cast<const __half*>(d_g),
        reinterpret_cast<const __half*>(d_beta),
        d_state,
        reinterpret_cast<__half*>(d_out),
        d_state,
        rows,
        key_heads,
        heads,
        q_scale
    );

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
