#include <torch/extension.h>

#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

__global__ void gather_routes_padded_kernel(
    const c10::BFloat16* __restrict__ x,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    c10::BFloat16* __restrict__ x_pad,
    float* __restrict__ weight_pad,
    int experts,
    int max_count,
    int hidden) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = experts * max_count * hidden;
    if (index >= total) {
        return;
    }

    const int col = index % hidden;
    const int row = (index / hidden) % max_count;
    const int expert = index / (max_count * hidden);
    const int begin = seg_starts[expert];
    const int end = seg_starts[expert + 1];
    const int route = begin + row;
    const bool valid = route < end;
    const int64_t dst = static_cast<int64_t>(index);
    if (!valid) {
        x_pad[dst] = c10::BFloat16(0.0f);
        if (col == 0) {
            weight_pad[static_cast<int64_t>(expert) * max_count + row] = 0.0f;
        }
        return;
    }

    const int64_t token = route_tokens[route];
    x_pad[dst] = x[token * hidden + col];
    if (col == 0) {
        weight_pad[static_cast<int64_t>(expert) * max_count + row] = route_weights[route];
    }
}

__global__ void swiglu_routes_kernel(
    const c10::BFloat16* __restrict__ gate_up,
    const float* __restrict__ route_weights,
    c10::BFloat16* __restrict__ hidden,
    int experts,
    int max_count,
    int inter,
    float swiglu_limit) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = experts * max_count * inter;
    if (index >= total) {
        return;
    }

    const int col = index % inter;
    const int row = (index / inter) % max_count;
    const int expert = index / (max_count * inter);
    const int64_t base =
        (static_cast<int64_t>(expert) * max_count + row) * (2 * inter);
    float gate = static_cast<float>(gate_up[base + col]);
    float up = static_cast<float>(gate_up[base + inter + col]);
    if (swiglu_limit > 0.0f) {
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        gate = fminf(gate, swiglu_limit);
    }
    const float value = (gate / (1.0f + expf(-gate))) * up *
        route_weights[static_cast<int64_t>(expert) * max_count + row];
    hidden[(static_cast<int64_t>(expert) * max_count + row) * inter + col] =
        c10::BFloat16(value);
}

__global__ void scatter_routes_kernel(
    const c10::BFloat16* __restrict__ out_routes,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    float* __restrict__ output,
    int experts,
    int max_count,
    int hidden) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = experts * max_count * hidden;
    if (index >= total) {
        return;
    }

    const int col = index % hidden;
    const int row = (index / hidden) % max_count;
    const int expert = index / (max_count * hidden);
    const int begin = seg_starts[expert];
    const int end = seg_starts[expert + 1];
    const int route = begin + row;
    if (route >= end) {
        return;
    }
    const int64_t token = route_tokens[route];
    const float value = static_cast<float>(
        out_routes[(static_cast<int64_t>(expert) * max_count + row) * hidden + col]);
    atomicAdd(output + token * hidden + col, value);
}

cublasStatus_t strided_batched_bf16_gemm(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const c10::BFloat16* a,
    long long stride_a,
    const c10::BFloat16* b,
    long long stride_b,
    c10::BFloat16* c,
    long long stride_c,
    int batch_count) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        a,
        CUDA_R_16BF,
        k,
        stride_a,
        b,
        CUDA_R_16BF,
        k,
        stride_b,
        &beta,
        c,
        CUDA_R_16BF,
        m,
        stride_c,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

torch::Tensor qwen4_exp_moe_prefill_bf16_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& route_tokens,
    const torch::Tensor& route_weights,
    const torch::Tensor& seg_starts,
    const torch::Tensor& gate_up,
    const torch::Tensor& down,
    double swiglu_limit) {
    c10::cuda::CUDAGuard device_guard(x.device());

    const int tokens = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int experts = static_cast<int>(gate_up.size(0));
    const int inter = static_cast<int>(gate_up.size(1) / 2);
    const int routes = static_cast<int>(route_tokens.size(0));

    auto seg_i32 = seg_starts.scalar_type() == torch::kInt32
        ? seg_starts.contiguous()
        : seg_starts.to(torch::kInt32);
    auto x_contig = x.contiguous();
    auto route_tokens_contig = route_tokens.contiguous();
    auto route_weights_contig = route_weights.contiguous();
    auto gate_up_contig = gate_up.contiguous();
    auto down_contig = down.contiguous();
    auto output = torch::zeros({tokens, hidden}, x.options().dtype(torch::kFloat32));

    if (routes == 0 || experts == 0) {
        return output.to(x.scalar_type());
    }

    auto counts = seg_i32.slice(0, 1, experts + 1) -
        seg_i32.slice(0, 0, experts);
    const int max_count = counts.max().item<int>();
    if (max_count <= 0) {
        return output.to(x.scalar_type());
    }

    auto x_pad = torch::empty(
        {experts, max_count, hidden}, x.options().dtype(torch::kBFloat16));
    auto weight_pad = torch::empty(
        {experts, max_count}, x.options().dtype(torch::kFloat32));
    auto gate_pad = torch::empty(
        {experts, max_count, 2 * inter}, x.options().dtype(torch::kBFloat16));
    auto hidden_pad = torch::empty(
        {experts, max_count, inter}, x.options().dtype(torch::kBFloat16));
    auto out_pad = torch::empty(
        {experts, max_count, hidden}, x.options().dtype(torch::kBFloat16));

    const int gather_total = experts * max_count * hidden;
    gather_routes_padded_kernel<<<
        ceil_div(gather_total, kThreads), kThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        x_contig.data_ptr<c10::BFloat16>(),
        route_tokens_contig.data_ptr<int64_t>(),
        seg_i32.data_ptr<int32_t>(),
        route_weights_contig.data_ptr<float>(),
        x_pad.data_ptr<c10::BFloat16>(),
        weight_pad.data_ptr<float>(),
        experts,
        max_count,
        hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(
        cublasSetStream(handle, at::cuda::getCurrentCUDAStream()) == CUBLAS_STATUS_SUCCESS,
        "failed to set cuBLAS stream for Qwen4-Exp BF16 MoE");

    cublasStatus_t status = strided_batched_bf16_gemm(
        handle,
        2 * inter,
        max_count,
        hidden,
        gate_up_contig.data_ptr<c10::BFloat16>(),
        static_cast<long long>(2 * inter) * hidden,
        x_pad.data_ptr<c10::BFloat16>(),
        static_cast<long long>(max_count) * hidden,
        gate_pad.data_ptr<c10::BFloat16>(),
        static_cast<long long>(max_count) * (2 * inter),
        experts);
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "Qwen4-Exp BF16 w13 grouped GEMM failed with cuBLAS status ",
        static_cast<int>(status));

    const int hidden_total = experts * max_count * inter;
    swiglu_routes_kernel<<<
        ceil_div(hidden_total, kThreads), kThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        gate_pad.data_ptr<c10::BFloat16>(),
        weight_pad.data_ptr<float>(),
        hidden_pad.data_ptr<c10::BFloat16>(),
        experts,
        max_count,
        inter,
        static_cast<float>(swiglu_limit));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    status = strided_batched_bf16_gemm(
        handle,
        hidden,
        max_count,
        inter,
        down_contig.data_ptr<c10::BFloat16>(),
        static_cast<long long>(hidden) * inter,
        hidden_pad.data_ptr<c10::BFloat16>(),
        static_cast<long long>(max_count) * inter,
        out_pad.data_ptr<c10::BFloat16>(),
        static_cast<long long>(max_count) * hidden,
        experts);
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "Qwen4-Exp BF16 w2 grouped GEMM failed with cuBLAS status ",
        static_cast<int>(status));

    const int output_total = experts * max_count * hidden;
    scatter_routes_kernel<<<
        ceil_div(output_total, kThreads), kThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        out_pad.data_ptr<c10::BFloat16>(),
        route_tokens_contig.data_ptr<int64_t>(),
        seg_i32.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        experts,
        max_count,
        hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output.to(x.scalar_type());
}
