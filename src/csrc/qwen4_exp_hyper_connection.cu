#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kMaxWarps = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void qwen4_exp_grouped_rms_norm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t rows,
    int groups,
    int group_size,
    float eps) {
    const int64_t row_group = blockIdx.x;
    if (row_group >= rows * groups) {
        return;
    }
    const int group = static_cast<int>(row_group % groups);
    const int64_t row = row_group / groups;
    const int64_t base = (row * groups + group) * group_size;

    float square_sum = 0.0f;
    for (int col = threadIdx.x; col < group_size; col += blockDim.x) {
        const float value = static_cast<float>(input[base + col]);
        square_sum += value * value;
    }
    square_sum = warp_sum(square_sum);

    __shared__ float warp_sums[kMaxWarps];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    float total = threadIdx.x < kMaxWarps ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        total = warp_sum(total);
        if (lane == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(group_size) + eps);

    for (int col = threadIdx.x; col < group_size; col += blockDim.x) {
        const int64_t index = base + col;
        const float gain = 1.0f + static_cast<float>(weight[group * group_size + col]);
        output[index] = scalar_t(static_cast<float>(input[index]) * inv_rms * gain);
    }
}

__global__ void qwen4_exp_hc_silu_bf16_kernel(
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output,
    int64_t elements,
    float scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float value = static_cast<float>(input[index]) * scale;
    output[index] = c10::BFloat16(value / (1.0f + expf(-value)));
}

__global__ void qwen4_exp_hc_inject_gate_bf16_kernel(
    const c10::BFloat16* __restrict__ input,
    c10::BFloat16* __restrict__ output,
    int64_t elements,
    float scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float value = static_cast<float>(input[index]) * scale;
    output[index] = c10::BFloat16(2.0f / (1.0f + expf(-value)));
}

template <typename scalar_t>
__global__ void qwen4_exp_inject_kernel(
    const scalar_t* __restrict__ block_output,
    const scalar_t* __restrict__ hyper_input,
    const scalar_t* __restrict__ injection_weights,
    scalar_t* __restrict__ output,
    int64_t elements,
    int hidden,
    int groups) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int col = static_cast<int>(index % hidden);
    const int group = static_cast<int>((index / hidden) % groups);
    const int64_t row = index / (static_cast<int64_t>(groups) * hidden);
    const float block = static_cast<float>(block_output[row * hidden + col]);
    const float residual = static_cast<float>(hyper_input[index]);
    const float scale = static_cast<float>(injection_weights[row * groups + group]);
    const scalar_t injection = scalar_t(block * scale);
    output[index] = scalar_t(residual + static_cast<float>(injection));
}

}  // namespace

torch::Tensor qwen4_exp_grouped_rms_norm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t group_size,
    double eps) {
    c10::cuda::CUDAGuard device_guard(input.device());
    const int64_t rows = input.numel() / input.size(-1);
    const int groups = static_cast<int>(input.size(-1) / group_size);
    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "qwen4_exp_grouped_rms_norm",
        [&] {
            qwen4_exp_grouped_rms_norm_kernel<scalar_t><<<
                rows * groups,
                kThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                groups,
                static_cast<int>(group_size),
                static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen4_exp_hc_silu_bf16_cuda(
    const torch::Tensor& input,
    int64_t groups) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    const int64_t elements = input.numel();
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);

    qwen4_exp_hc_silu_bf16_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<c10::BFloat16>(),
        output.data_ptr<c10::BFloat16>(),
        elements,
        1.0f / static_cast<float>(groups));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen4_exp_hc_inject_gate_bf16_cuda(
    const torch::Tensor& input,
    int64_t groups) {
    c10::cuda::CUDAGuard device_guard(input.device());
    auto output = torch::empty_like(input);
    const int64_t elements = input.numel();
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);

    qwen4_exp_hc_inject_gate_bf16_kernel<<<
        blocks,
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<c10::BFloat16>(),
        output.data_ptr<c10::BFloat16>(),
        elements,
        1.0f / static_cast<float>(groups));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen4_exp_inject_cuda(
    const torch::Tensor& block_output,
    const torch::Tensor& hyper_input,
    const torch::Tensor& injection_weights,
    int64_t groups) {
    c10::cuda::CUDAGuard device_guard(hyper_input.device());
    auto output = torch::empty_like(hyper_input);
    const int hidden = static_cast<int>(block_output.size(-1));
    const int64_t elements = hyper_input.numel();
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        hyper_input.scalar_type(),
        "qwen4_exp_inject",
        [&] {
            qwen4_exp_inject_kernel<scalar_t><<<
                blocks,
                kThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                block_output.data_ptr<scalar_t>(),
                hyper_input.data_ptr<scalar_t>(),
                injection_weights.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                elements,
                hidden,
                static_cast<int>(groups));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
