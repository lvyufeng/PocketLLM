// Host wrapper for the vendored llama.cpp MMQ device code.
//
// Computes  out[rows, N]  =  act[rows, K]  @  W[N, K]^T
// where W is stored as GGUF Q4_K / Q5_K / PTQ1_0 weight blocks and act is fp16/bf16/f32.
//
// Pipeline:
//   1. Cast activation to fp32 contiguous [rows, K].
//   2. quantize_mmq_q8_1 : fp32 act -> block_q8_1_mmq  (DS_LAYOUT_DS4 for Q4_K/Q5_K,
//      DS_LAYOUT_D4 for PTQ1_0, whose weights are already signed int8 by then).
//   3. tiled mul_mat_q   : each block calls mul_mat_q_process_tile (MMA tensor core).
//   4. Cast fp32 dst [rows, N] -> bf16.
//
// PTQ1_0's prefill path is the same tile walk with its own loader; its decode path
// is not MMQ at all (see the GEMV further down), because a one-token step is a GEMV
// and the tensor core does nothing for it.
//
// We deliberately omit llama.cpp's stream-K, MoE/ids, multi-channel and fastdiv
// machinery: this is a plain GEMM. The tile function and all weight/activation
// load + MMA primitives come from the vendored mmq.cuh / vecdotq.cuh / mma.cuh.

// PyTorch's CUDAExtension injects -D__CUDA_NO_HALF_CONVERSIONS__ etc. globally,
// which disables the implicit float<->half conversions the vendored llama.cpp
// MMQ code relies on. These macros gate the conversion operators in
// cuda_fp16.hpp the FIRST time it is included, so we must undef them BEFORE
// any header (torch/ATen) pulls in cuda_fp16.hpp.
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "common_shim.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"
#include "mmq.cuh"

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// Programmatic launch dependency sync is Hopper+; no-op on Turing.
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}

namespace {

// ---------------------------------------------------------------------------
// 1. Activation quantization: fp32 -> block_q8_1_mmq.
//    Verbatim from llama.cpp quantize.cu:275, restricted to DS_LAYOUT_DS4
//    (the layout Q4_K / Q5_K consume). ids == nullptr (plain GEMM).
// ---------------------------------------------------------------------------
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
        // Exchange partial sums between vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d = amax / ((1 << 7) - 1);
    float d_inv = d > 0 ? 1.0f / d : 0.0f;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32].x = __float2half(d);
        y[ib].ds4[iqs/32].y = __float2half(sum);
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        // D4 stores the scale alone, as fp32: the weight side carries its own offset,
        // so the partial sum the DS4 layout needs has no consumer here.
        y[ib].d4[iqs/32] = d;
    }

    // Quantize the 4 floats into 4 int8s:
    int8_t * ys = y[ib].qs;
    ys[iqs + 0] = __float2int_rn(xi.x*d_inv);
    ys[iqs + 1] = __float2int_rn(xi.y*d_inv);
    ys[iqs + 2] = __float2int_rn(xi.z*d_inv);
    ys[iqs + 3] = __float2int_rn(xi.w*d_inv);
}

// Host launcher for activation quantization (single 2D batch).  The layout is the
// one the weight side's MMA dot reads: DS4 for Q4_K / Q5_K, D4 for PTQ1_0.
template <mmq_q8_1_ds_layout ds_layout>
void quantize_act_q8_1_mmq_cuda(const float * x, void * vy,
        int64_t ne00, int ne1, int ne2, cudaStream_t stream) {
    // ne00 = K, ne1 = rows (batch), ne2 = 1. Pad ne0 to a multiple of 4*QK8_1.
    const int64_t ne0 = ne00;  // caller guarantees K % (4*QK8_1) == 0 after padding
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    quantize_mmq_q8_1<ds_layout>
        <<<num_blocks, block_size, 0, stream>>>(x, nullptr, vy, ne00, ne00, 0, 0, ne0, ne1, ne2);
}

// ---------------------------------------------------------------------------
// 2. Tiled mul_mat_q (non-stream-K). One block per (weight-tile, act-tile).
//    Replicates llama.cpp mmq.cuh:3582-3635 but stripped of stream-k / ids / channels.
//
//    x = weight blocks  [N, blocks_per_row] of `type`
//    y = activation     [rows, blocks_per_row_q8] of block_q8_1_mmq
//    dst= fp32 output   [rows, N]  (batch-major, i.e. dst[token][feature])
//
//    In llama.cpp's naming, weight -> mmq_y tiles, activation -> mmq_x tiles,
//    and write_back does dst[act_idx * N + weight_idx].
// ---------------------------------------------------------------------------
template <ggml_type type, int mmq_x, bool need_check>
__global__ void gguf_mma_prefill_kernel(
        const char * __restrict__ x,        // weight blocks, as const char*
        const int   * __restrict__ y,        // activation block_q8_1_mmq, as int*
        float       * __restrict__ dst,      // [rows, N] fp32
        const int     stride_row_x,          // blocks_per_row (weight)
        const int     ncols_y,               // rows * blocks_per_row_q8 * (sizeof(block_q8_1_mmq)/sizeof(int))
        const int     nrows_x,               // N (weight rows)
        const int     ncols_dst) {           // rows (activation)

    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int mmq_y     = get_mmq_y_device();

    // blockIdx.x -> weight tile (mmq_y, over N), blockIdx.y -> activation tile (mmq_x, over rows).
    const int it = blockIdx.x;
    const int jt = blockIdx.y;

    extern __shared__ int ids_dst_shared[]; // identity init: ids_dst_shared[j] = j
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
            break;
        }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int offset_x   = it*mmq_y*stride_row_x;                 // weight offset (block index)
    const int offset_y   = (jt*mmq_x)*(sizeof(block_q8_1_mmq)/sizeof(int)); // act offset (int index)
    // dst is [rows, N] row-major: stride_col_dst = N. The weight tile `it` owns
    // output features [it*mmq_y, it*mmq_y+mmq_y), so its base pointer is offset
    // by it*mmq_y; the activation tile `jt` owns tokens [jt*mmq_x, jt*mmq_x+mmq_x),
    // offset by jt*mmq_x*N. write_back does dst[base + token*N + feat_local].
    const int offset_dst = jt*mmq_x*nrows_x + it*mmq_y;           // [rows,N] batch-major

    const int tile_x_max_i = nrows_x   - it*mmq_y - 1;            // weight bound
    const int tile_y_max_j = ncols_dst - jt*mmq_x - 1;            // activation bound

    constexpr bool fixup = false;
    mul_mat_q_process_tile<type, mmq_x, need_check, fixup>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, nrows_x,
        tile_x_max_i, tile_y_max_j, 0, stride_row_x);
}

// Pick the largest mmq_x (activation tile) that divides `rows` and is <= 128
// for the current arch. Mirrors llama.cpp's granularity logic.
template <ggml_type type>
void launch_prefill(const char * x_w, const int * y_q, float * dst,
        int N, int rows, int blocks_per_row, int blocks_per_row_q8,
        int nbytes_shared, cudaStream_t stream) {
    constexpr int mmq_y = 128;  // get_mmq_y_host(Turing) == 128

    const int nty = (N    + mmq_y - 1) / mmq_y;  // weight tiles
    // Granularity on Turing with mmq_x>=48 is 16.
    const int granularity = 16;
    int mmq_x = 128;
    while (mmq_x > 0 && (rows % mmq_x != 0 || mmq_x % granularity != 0)) {
        mmq_x -= 8;
    }
    if (mmq_x <= 0) mmq_x = 8;  // fallback (need_check handles bounds)

    const dim3 block_dims(32, mmq_get_nwarps_host(GGML_CUDA_CC_TURING, 32), 1);
    const dim3 grid(nty, (rows + mmq_x - 1) / mmq_x, 1);

    const int ncols_y = rows;  // ncols_y = number of activation tokens (rows).

    auto launch = [&](auto mmq_x_const) {
        constexpr int MX = decltype(mmq_x_const)::value;
        // need_check true handles N/rows not divisible by tile (the common case).
        constexpr bool need_check = true;
        using Kern = decltype(&gguf_mma_prefill_kernel<type, MX, need_check>);
        // The MMA tile needs >48 KB of dynamic shared memory; raise the per-block
        // limit (2080Ti allows up to 64 KB opt-in). Idempotent per kernel pointer.
        static bool raised = false;
        if (!raised) {
            cudaFuncSetAttribute(
                (const void*)gguf_mma_prefill_kernel<type, MX, need_check>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
            raised = true;
        }
        (void)(Kern*)nullptr;
        gguf_mma_prefill_kernel<type, MX, need_check><<<grid, block_dims, nbytes_shared, stream>>>(
            x_w, y_q, dst, blocks_per_row, ncols_y, N, rows);
    };

    // Instantiate the handful of mmq_x values we may select.
    switch (mmq_x) {
        case 128: launch(std::integral_constant<int, 128>{}); break;
        case  64: launch(std::integral_constant<int,  64>{}); break;
        case  32: launch(std::integral_constant<int,  32>{}); break;
        case  16: launch(std::integral_constant<int,  16>{}); break;
        default:  launch(std::integral_constant<int,   8>{}); break;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point: rows x N bf16 output.
// ---------------------------------------------------------------------------
torch::Tensor gguf_q4k_q5k_mma_prefill_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems,
        int64_t type_id) {  // 3 = Q4_K, 4 = Q5_K
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();

    const int k   = static_cast<int>(row_elems);              // K
    const int N   = static_cast<int>(blocks_contig.size(0)); // weight rows
    const int bpr = static_cast<int>(blocks_contig.size(1)); // blocks per weight row
    const int rows = static_cast<int>(x_contig.numel() / k); // activation batch

    TORCH_CHECK(type_id == 3 || type_id == 4, "MMA path supports Q4_K (3) or Q5_K (4) only");
    TORCH_CHECK(k % QK_K == 0, "K must be a multiple of 256 for Q4_K/Q5_K MMA");
    // Activation quantizer pads to 4*QK8_1; K must already be a multiple of QK8_1 (==32).
    TORCH_CHECK(k % QK8_1 == 0, "K must be a multiple of 32");

    // --- 1. activation -> fp32 contiguous [rows, K] ---
    auto x_f32 = x_contig.to(torch::kFloat32);

    // --- 2. quantize activation -> block_q8_1_mmq ---
    // One block_q8_1_mmq per (row, group of 128 K elements) => blocks_per_row_q8 = K/128.
    const int blocks_per_row_q8 = k / (4 * QK8_1);  // == K/128
    auto y_q = torch::empty({rows, blocks_per_row_q8, (int)sizeof(block_q8_1_mmq)},
                            x.options().dtype(torch::kUInt8));

    quantize_act_q8_1_mmq_cuda<MMQ_Q8_1_DS_LAYOUT_DS4>(
        x_f32.data_ptr<float>(), y_q.data_ptr(),
        k, rows, /*ne2=*/1, stream);

    // --- 3. tiled MMA GEMM -> fp32 [rows, N] ---
    auto dst_f32 = torch::empty({rows, N}, x.options().dtype(torch::kFloat32));

    const int warp_size = 32;
    const int nwarps    = mmq_get_nwarps_host(GGML_CUDA_CC_TURING, warp_size);
    const int mmq_y     = 128;  // Turing
    // Shared memory size (matches mmq_get_nbytes_shared): ids + weight tile + padded activation tile.
    const int mmq_tile_x_k = (type_id == 3)
        ? MMQ_MMA_TILE_X_K_Q8_1  // Q4_K MMA layout == Q8_1 layout in llama.cpp
        : MMQ_MMA_TILE_X_K_Q8_1; // Q5_K also uses Q8_1-style MMA tile
    const size_t nbs_ids = 128 * sizeof(int);  // upper bound for mmq_x ids
    const size_t nbs_x   = (size_t)mmq_y * mmq_tile_x_k * sizeof(int);
    const size_t nbs_y   = (size_t)128 * sizeof(block_q8_1_mmq);  // upper bound for mmq_x activation
    const int nbytes_shared = (int)(nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int)));

    const char * x_w = reinterpret_cast<const char*>(blocks_contig.data_ptr<uint8_t>());
    const int  * y_qp = reinterpret_cast<const int*>(y_q.data_ptr());
    float * dst_p = dst_f32.data_ptr<float>();

    if (type_id == 3) {
        launch_prefill<GGML_TYPE_Q4_K>(x_w, y_qp, dst_p, N, rows, bpr, blocks_per_row_q8, nbytes_shared, stream);
    } else {
        launch_prefill<GGML_TYPE_Q5_K>(x_w, y_qp, dst_p, N, rows, bpr, blocks_per_row_q8, nbytes_shared, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // --- 4. fp32 -> bf16, restore leading batch dims ---
    auto out = dst_f32.to(torch::kBFloat16);
    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    return out.view(out_shape);
}

// ===========================================================================
// PTQ1_0 prefill: the same tile walk, with the ternary loader in place of the
// Q4_K/Q5_K one.  The loader expands each trit to a signed byte, so the tile is
// a Q8_0 tile and the MMA dot is shared -- which is why this entry point is a
// dozen lines rather than a second kernel.
// ===========================================================================
torch::Tensor gguf_ptq1_0_mma_prefill_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();

    const int k    = static_cast<int>(row_elems);             // K
    const int N    = static_cast<int>(blocks_contig.size(0)); // weight rows
    const int bpr  = static_cast<int>(blocks_contig.size(1)); // blocks per weight row
    const int rows = static_cast<int>(x_contig.numel() / k);  // activation batch

    TORCH_CHECK(blocks_contig.size(2) == (int)sizeof(block_ptq1_0), "last dim must be one PTQ1_0 block");
    TORCH_CHECK(k % QK_PTQ1_0 == 0, "K must be a multiple of 128 for PTQ1_0");
    TORCH_CHECK(bpr == k / QK_PTQ1_0, "blocks per row must be K/128");

    auto x_f32 = x_contig.to(torch::kFloat32);

    // One block_q8_1_mmq per (row, 128 K elements), which is also one weight block.
    const int blocks_per_row_q8 = k / (4 * QK8_1);  // == K/128
    auto y_q = torch::empty({rows, blocks_per_row_q8, (int)sizeof(block_q8_1_mmq)},
                            x.options().dtype(torch::kUInt8));
    quantize_act_q8_1_mmq_cuda<MMQ_Q8_1_DS_LAYOUT_D4>(
        x_f32.data_ptr<float>(), y_q.data_ptr(), k, rows, /*ne2=*/1, stream);

    auto dst_f32 = torch::empty({rows, N}, x.options().dtype(torch::kFloat32));

    const int warp_size = 32;
    const int nwarps    = mmq_get_nwarps_host(GGML_CUDA_CC_TURING, warp_size);
    const int mmq_y     = 128;  // Turing
    const size_t nbs_ids = 128 * sizeof(int);
    const size_t nbs_x   = (size_t)mmq_y * MMQ_MMA_TILE_X_K_Q8_0 * sizeof(int);
    const size_t nbs_y   = (size_t)128 * sizeof(block_q8_1_mmq);
    const int nbytes_shared = (int)(nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int)));

    const char * x_w  = reinterpret_cast<const char*>(blocks_contig.data_ptr<uint8_t>());
    const int  * y_qp = reinterpret_cast<const int*>(y_q.data_ptr());
    launch_prefill<GGML_TYPE_PTQ1_0>(
        x_w, y_qp, dst_f32.data_ptr<float>(), N, rows, bpr, blocks_per_row_q8, nbytes_shared, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = dst_f32.to(torch::kBFloat16);
    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    return out.view(out_shape);
}

// ===========================================================================
// DECODE path (rows == 1): DP4A on CUDA cores. For [1,K] x [K,N] a GEMV, the
// MMA tensor core does not help; llama.cpp's MMVQ uses __dp4a per output, which
// is the latency-optimal path. We reuse the vendored vec_dot_q4_K_q8_1 /
// vec_dot_q5_K_q8_1 (which call vec_dot_*_impl_vmmq -> __dp4a).
//
// Activation is quantized to standard block_q8_1 (32 values/block, 34 bytes),
// NOT block_q8_1_mmq.
// ===========================================================================

// Quantize fp32 activation to block_q8_1. block_q8_1 = { half d; half s; int8_t qs[32]; }.
//
// One warp quantizes 32 blocks, one per `threadIdx.y` -- so a launch must give this
// kernel a full 32-deep y dimension or only the first block of each x index gets
// written and the rest of the activation buffer is uninitialized garbage.
struct block_q8_1_decode {
    half   d;       // scale (== amax/127)
    half   s;       // sum of the 32 values
    int8_t qs[32];  // quantized values
};

constexpr int kQ8_1DecodeBlocksPerWarp = 32;

// 1 warp (32 threads) quantizes one 32-element block.
__global__ void quantize_q8_1_decode_kernel(const float * __restrict__ x, void * __restrict__ vy,
        int k, int n_blocks) {
    // Requires blockDim.x == 32 (the shuffles below are warp-wide) and blockDim.y ==
    // kQ8_1DecodeBlocksPerWarp; x is the block index, y the block within it.
    const int block_id = blockIdx.x * kQ8_1DecodeBlocksPerWarp + threadIdx.y;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x;
    const int base = block_id * 32;

    float xi = (base + lane < k) ? x[base + lane] : 0.0f;
    // amax across the 32 lanes
    float amax = fabsf(xi);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    }
    const float d     = amax / 127.0f;
    const float d_inv = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q = (int8_t)__float2int_rn(xi * d_inv);
    float sum = xi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    }

    block_q8_1_decode * y = (block_q8_1_decode *) vy;
    if (lane == 0) {
        y[block_id].d = __float2half(d);
        y[block_id].s = __float2half(sum);
    }
    y[block_id].qs[lane] = q;
}

// Local copies of the Q4_K/Q5_K scale helpers (also defined in cuda_kernel_impl.cu,
// but that is a separate TU). Kept byte-identical to guarantee the same dequant.
static __device__ __forceinline__ float gguf_block_scale_f16(const uint8_t* ptr) {
    const uint16_t bits = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}
static __device__ __forceinline__ void gguf_get_scale_min_k4(
        const uint8_t* __restrict__ scales, int idx, int& d, int& m) {
    if (idx < 4) {
        d = static_cast<int>(scales[idx] & 63);
        m = static_cast<int>(scales[idx + 4] & 63);
    } else {
        d = static_cast<int>((scales[idx + 4] & 0x0F) | ((scales[idx - 4] >> 6) << 4));
        m = static_cast<int>((scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4));
    }
}

// Custom DP4A decode GEMV. One CUDA block per output feature; one warp (32 lanes)
// strides over weight blocks (each lane handles weight blocks {lane, lane+32, ...}).
//
// Q4_K/Q5_K dequant (verified against q4k_block_dot_256 / q5k_block_dot_256):
//   w_i = d*sc_g*q_i - dmin*mn_g,   q_i in {0..15} (Q4) or {0..31} (Q5)
// For Q8_1 activation x_i (signed int8, scale d8, sum stored in a->s):
//   sum_i x_i*w_i = d8 * [ d*sc_g*dp4a(q_g, x_g) - dmin*mn_g * (a->s / d8) ]
//                 = d8*d*sc_g*dp4a - dmin*mn_g*a->s        (per group g)
// We sum over the 8 groups in the weight block.
template <ggml_type type>
__global__ void gguf_dp4a_decode_kernel(
        const char   * __restrict__ x_w,          // weight blocks [N, blocks_per_row, block_bytes]
        const block_q8_1_decode * __restrict__ y, // activation [n_q8_blocks]
        float        * __restrict__ dst,          // [N]
        const int      blocks_per_row,
        const int      block_bytes) {
    const int out_col = blockIdx.x;
    const int lane = threadIdx.x;

    float acc = 0.0f;
    for (int kb = lane; kb < blocks_per_row; kb += 32) {
        const uint8_t * wb = reinterpret_cast<const uint8_t*>(x_w)
                             + (size_t)out_col * blocks_per_row * block_bytes
                             + (size_t)kb * block_bytes;

        const float d    = gguf_block_scale_f16(wb);
        const float dmin = gguf_block_scale_f16(wb + 2);

        float block_acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            int sc = 0, mn = 0;
            gguf_get_scale_min_k4(wb + 4, g, sc, mn);
            const block_q8_1_decode * a = y + kb * 8 + g;  // activation block for this 32-element group
            const float d8 = __half2float(a->d);

            int sumi = 0;
            const int * aq32 = (const int *) a->qs;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {  // 8 int32 packs x 4 int8 = 32 elements
                int8_t wp[4];
                if constexpr (type == GGML_TYPE_Q4_K) {
                    // qs[128]; group g uses qs[g*16 .. g*16+15]; 2 nibbles/byte.
                    const uint8_t * qs = wb + 16 + g * 16;
                    const uint8_t b0 = qs[p*2 + 0], b1 = qs[p*2 + 1];
                    wp[0] = (int8_t)( b0        & 0x0F);
                    wp[1] = (int8_t)((b0 >> 4)  & 0x0F);
                    wp[2] = (int8_t)( b1        & 0x0F);
                    wp[3] = (int8_t)((b1 >> 4)  & 0x0F);
                } else {  // Q5_K: qh[32] at +16, qs[128] at +48; 5th bit from qh
                    const uint8_t * qs = wb + 48 + g * 16;
                    const uint8_t * qh = wb + 16 + g * 4;
                    const uint8_t b0 = qs[p*2 + 0], b1 = qs[p*2 + 1];
                    const uint8_t hb = qh[p];  // 8 bits: 4 for b0's 4 nibbles, 4 for b1's
                    wp[0] = (int8_t)(( b0       & 0x0F) | (((hb >> 0) & 1) << 4));
                    wp[1] = (int8_t)(((b0 >> 4) & 0x0F) | (((hb >> 1) & 1) << 4));
                    wp[2] = (int8_t)(( b1       & 0x0F) | (((hb >> 2) & 1) << 4));
                    wp[3] = (int8_t)(((b1 >> 4) & 0x0F) | (((hb >> 3) & 1) << 4));
                }
                sumi = __dp4a(*((const int*)wp), aq32[p], sumi);
            }
            const float sum_a = __half2float(a->s);  // == d8 * sum(x_int8)
            block_acc += d8 * d * (float)sc * (float)sumi - dmin * (float)mn * sum_a;
        }
        acc += block_acc;
    }

    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    }
    if (lane == 0) {
        dst[out_col] = acc;
    }
}

torch::Tensor gguf_q4k_q5k_dp4a_decode_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems,
        int64_t type_id) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();

    const int k    = static_cast<int>(row_elems);
    const int N    = static_cast<int>(blocks_contig.size(0));
    const int bpr  = static_cast<int>(blocks_contig.size(1));
    const int rows = static_cast<int>(x_contig.numel() / k);
    TORCH_CHECK(rows == 1, "DP4A decode path is for rows == 1");
    TORCH_CHECK(type_id == 3 || type_id == 4, "DP4A decode supports Q4_K (3) or Q5_K (4)");
    TORCH_CHECK(k % QK_K == 0, "K must be a multiple of 256");

    // --- 1. activation -> fp32 contiguous [1, K] ---
    auto x_f32 = x_contig.to(torch::kFloat32);

    // --- 2. quantize activation -> block_q8_1 (32 values per block) ---
    const int n_q8_blocks = k / 32;
    auto y_q = torch::empty({n_q8_blocks, (int)sizeof(block_q8_1_decode)},
                            x.options().dtype(torch::kUInt8));
    {
        const int nblocks = (n_q8_blocks + kQ8_1DecodeBlocksPerWarp - 1) / kQ8_1DecodeBlocksPerWarp;
        quantize_q8_1_decode_kernel<<<nblocks, dim3(32, kQ8_1DecodeBlocksPerWarp), 0, stream>>>(
            x_f32.data_ptr<float>(), y_q.data_ptr(), k, n_q8_blocks);
    }

    // --- 3. DP4A decode GEMV -> fp32 [1, N] ---
    auto dst_f32 = torch::empty({1, N}, x.options().dtype(torch::kFloat32));
    const int block_bytes = static_cast<int>(blocks_contig.size(2));
    const int grid = N;
    const dim3 block(32, 1, 1);

    const char * x_w = reinterpret_cast<const char*>(blocks_contig.data_ptr<uint8_t>());
    if (type_id == 3) {
        gguf_dp4a_decode_kernel<GGML_TYPE_Q4_K><<<grid, block, 0, stream>>>(
            x_w, reinterpret_cast<const block_q8_1_decode*>(y_q.data_ptr()),
            dst_f32.data_ptr<float>(), bpr, block_bytes);
    } else {
        gguf_dp4a_decode_kernel<GGML_TYPE_Q5_K><<<grid, block, 0, stream>>>(
            x_w, reinterpret_cast<const block_q8_1_decode*>(y_q.data_ptr()),
            dst_f32.data_ptr<float>(), bpr, block_bytes);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = dst_f32.to(torch::kBFloat16);
    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    return out.view(out_shape);
}

// ---------------------------------------------------------------------------
// PTQ1_0 decode GEMV.
//
// A one-token step reads all N*K weights once and does one MAC each, so it is a
// bandwidth problem and nothing else.  The card's practical read ceiling is 540 GiB/s
// (measured with a torch reduction over the same shape), and what a kernel has to buy
// with that is memory-level parallelism: at ~600 ns of DRAM latency, 540 GiB/s needs
// 320 KiB of requests in flight, and a warp that reads four bytes per lane per
// iteration never gets close.  So one *thread* owns one output row and reads the whole
// 28-byte block per iteration -- 28 bytes of traffic per lane per iteration instead of
// four -- which is the structure the fork's `vec_dot_ptq1_0_q8_1` uses.
//
// The other half of that structure is where the scale goes.  A 128-weight block spans
// exactly four 32-element activation blocks, so the block's dot product is four
// integer sums rather than one, each with its own activation scale applied once at the
// end.  That keeps both the half load and the float multiply out of the 32-step inner
// loop: `sumi[4]` is the fork's shape and it is worth about a fifth of the ALU work
// per weight.
//
// The activation is staged in shared memory, which is what makes the inner loop's
// 32 activation reads LDS rather than a gather.  Only the running chunk is staged, not
// the whole tensor: at K = 17,408 the full activation is 19.6 KiB, which caps the
// resident blocks per SM at two and leaves the latency uncovered again.
//
// K is split whenever one thread per row would not fill the card.  The card is slow
// enough to launch and fast enough to run that this is not a small-N special case:
// 17,408 rows is 136 blocks, two per SM, and the kernel needs eight.  Split chunks
// land in separate accumulators and a second kernel reduces them in a fixed order, so
// the result does not depend on the schedule.
// ---------------------------------------------------------------------------
#define PTQ1_0_DECODE_THREADS 128

__device__ __forceinline__ void ptq1_0_block_dot(
        const block_ptq1_0 * __restrict__ b,
        const block_q8_1_decode * __restrict__ a,
        float & acc) {
    int sumi[4] = {0, 0, 0, 0};

    // Weights 0..79: bytes 0..15, four trit streams per 32-bit word, one 16 apart.
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = get_int_b4(b->qs, g);
        uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
        uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo                = w_lo & 0x00FF00FF;
            v_hi                = w_hi & 0x00FF00FF;
            const int e = t * 16 + 4 * g;
            const int q = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
            const int u = get_int_b4(a[e >> 5].qs, (e & 31) >> 2);
            sumi[e >> 5] = __dp4a(q, u, sumi[e >> 5]);
        }
    }

    // Weights 80..119: bytes 16..23, the same walk eight apart.
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t packed = get_int_b4(b->qs + 16, g);
        uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
        uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo                = w_lo & 0x00FF00FF;
            v_hi                = w_hi & 0x00FF00FF;
            const int e = 80 + t * 8 + 4 * g;
            const int q = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
            const int u = get_int_b4(a[e >> 5].qs, (e & 31) >> 2);
            sumi[e >> 5] = __dp4a(q, u, sumi[e >> 5]);
        }
    }

    // qh carries weights 120..127, four trits per byte with the parity interleaved:
    // qh[0] holds 120,122,124,126 and qh[1] holds 121,123,125,127.
    {
        uint32_t v = (uint32_t) b->qh[0] | ((uint32_t) b->qh[1] << 16);
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
            const uint32_t w0 = v * 3;
            v                 = w0 & 0x00FF00FF;
            const uint32_t w1 = v * 3;
            v                 = w1 & 0x00FF00FF;
            const int e = 120 + 2 * t;
            const int q = __vsub4(__byte_perm(w0, w1, 0x7531), 0x01010101);
            const int u = get_int_b4(a[e >> 5].qs, (e & 31) >> 2);
            sumi[e >> 5] = __dp4a(q, u, sumi[e >> 5]);
        }
    }

    float local = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        local += __half2float(a[k].d) * (float) sumi[k];
    }
    acc += __half2float(b->d) * local;
}

__global__ void ptq1_0_decode_kernel(
        const block_q8_1_decode * __restrict__ y,   // activation: K/32 blocks
        const uint8_t           * __restrict__ x_w, // [n, bpr] of 28-byte blocks
        float                   * __restrict__ part, // [split, n]
        const int bpr,
        const int n,
        const int split) {
    extern __shared__ block_q8_1_decode s_y[];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk = blockIdx.y;
    const int kb_begin = (int) ((long long) chunk * bpr / split);
    const int kb_end   = (int) ((long long) (chunk + 1) * bpr / split);

    // The chunk's own slice of the activation.  Every thread in the block reads the
    // same address at the same time, so the staging loop and the inner loop's reads
    // are both broadcast.
    const block_q8_1_decode * y_chunk = y + (size_t) kb_begin * (QK_PTQ1_0 / QK8_1);
    const int y_blocks = (kb_end - kb_begin) * (QK_PTQ1_0 / QK8_1);
    for (int i = threadIdx.x; i < y_blocks; i += blockDim.x) {
        s_y[i] = y_chunk[i];
    }
    __syncthreads();

    if (row >= n || chunk >= split) return;

    const uint8_t * row_w = x_w + (size_t) row * bpr * sizeof(block_ptq1_0);
    float acc = 0.0f;
#pragma unroll 2
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        ptq1_0_block_dot((const block_ptq1_0 *) (row_w + (size_t) kb * sizeof(block_ptq1_0)),
                         s_y + (kb - kb_begin) * (QK_PTQ1_0 / QK8_1), acc);
    }
    part[(size_t) chunk * n + row] = acc;
}

// Fixes the split chunks onto the row in chunk order, which is the order the
// single-chunk path would have accumulated them in.
__global__ void ptq1_0_decode_reduce_kernel(const float * __restrict__ part, float * __restrict__ dst,
        const int n, const int split) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.0f;
    for (int c = 0; c < split; ++c) {
        acc += part[(size_t) c * n + row];
    }
    dst[row] = acc;
}

torch::Tensor gguf_ptq1_0_dp4a_decode_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& blocks,
        int64_t row_elems) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_contig = x.contiguous();
    auto blocks_contig = blocks.contiguous();

    const int k    = static_cast<int>(row_elems);
    const int N    = static_cast<int>(blocks_contig.size(0));
    const int bpr  = static_cast<int>(blocks_contig.size(1));
    const int rows = static_cast<int>(x_contig.numel() / k);

    TORCH_CHECK(rows == 1, "PTQ1_0 decode takes one activation row");
    TORCH_CHECK(blocks_contig.size(2) == (int)sizeof(block_ptq1_0), "last dim must be one PTQ1_0 block");
    TORCH_CHECK(k % QK8_1 == 0, "K must be a multiple of 32");

    auto x_f32 = x_contig.to(torch::kFloat32);

    const int n_q8_blocks = k / QK8_1;
    auto y_q = torch::empty({n_q8_blocks, (int)sizeof(block_q8_1_decode)},
                            x.options().dtype(torch::kUInt8));
    {
        const int nblocks = (n_q8_blocks + kQ8_1DecodeBlocksPerWarp - 1) / kQ8_1DecodeBlocksPerWarp;
        quantize_q8_1_decode_kernel<<<nblocks, dim3(32, kQ8_1DecodeBlocksPerWarp), 0, stream>>>(
            x_f32.data_ptr<float>(), y_q.data_ptr(), k, n_q8_blocks);
    }

    // Split K until the grid fills the card: one thread per row is 128 blocks for a
    // 16,384-wide output against the ~544 the SMs will hold, and a chunk shorter than
    // four blocks costs more in the partials than it buys back.
    int split = 1;
    while (split < 8 && ((int64_t) N * split < 65536 || split < (bpr + 39) / 40)
           && bpr / (split * 2) >= 4) {
        split *= 2;
    }

    // The widest chunk, which is `ceil(bpr/split)` blocks and their four q8_1 blocks each.
    const int smem = ((bpr + split - 1) / split * (QK_PTQ1_0 / QK8_1)) * (int)sizeof(block_q8_1_decode);
    const dim3 block(PTQ1_0_DECODE_THREADS, 1, 1);
    const dim3 grid((N + PTQ1_0_DECODE_THREADS - 1) / PTQ1_0_DECODE_THREADS, split, 1);

    auto dst_f32 = torch::empty({1, N}, x.options().dtype(torch::kFloat32));
    if (split == 1) {
        ptq1_0_decode_kernel<<<grid, block, smem, stream>>>(
            reinterpret_cast<const block_q8_1_decode*>(y_q.data_ptr()),
            blocks_contig.data_ptr<uint8_t>(),
            dst_f32.data_ptr<float>(), bpr, N, 1);
    } else {
        auto part = torch::empty({split, N}, x.options().dtype(torch::kFloat32));
        ptq1_0_decode_kernel<<<grid, block, smem, stream>>>(
            reinterpret_cast<const block_q8_1_decode*>(y_q.data_ptr()),
            blocks_contig.data_ptr<uint8_t>(),
            part.data_ptr<float>(), bpr, N, split);
        ptq1_0_decode_reduce_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
            part.data_ptr<float>(), dst_f32.data_ptr<float>(), N, split);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out = dst_f32.to(torch::kBFloat16);
    auto out_shape = x.sizes().vec();
    out_shape.back() = N;
    return out.view(out_shape);
}

