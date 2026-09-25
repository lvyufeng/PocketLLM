// PTQ1_0 dense GEMM for the native engine: the fork-private ternary packing
// (GGML type 143, 128 weights in 28 bytes) at both batch widths.
//
// Provenance, because this is the one place the native engine compiles vendored
// llama.cpp device code:
//
//   * `llama_mmq/` beside this file is a byte-identical copy of the vendored tree
//     the Python extension builds (`src/csrc/llama_mmq/`), which is where the
//     PTQ1_0 tile loader was written and measured. The two copies exist because
//     the two engines are separate builds with separate ABIs, and a shared
//     include across the `src/` / `cpp_engine/` boundary would make one engine's
//     build depend on the other's source tree. They are pinned to the same fork
//     revision and must be updated together.
//   * The kernels below are the torch-free half of
//     `src/csrc/llama_mmq/gguf_mma_wrapper.cu`. That file owns the same two
//     entry points with `torch::Tensor` arguments and allocates its scratch with
//     `torch::empty`; here the arguments are raw device pointers and the scratch
//     is a cached thread-local buffer, matching how `qwen_half_ops.cu` holds its
//     cuBLAS workspace. The device code is otherwise unchanged, so a fix on one
//     side belongs on both.
//
// Why the batch widths are different kernels, and stay different:
//
//   * Prefill is a real GEMM and goes through llama.cpp's tiled MMA walk, whose
//     PTQ1_0 loader unpacks each 128-weight block into one signed byte per
//     weight, turning it into an ordinary Q8_0 tile that the shared MMA dot
//     consumes unchanged. That loader lives in `llama_mmq/mmq.cuh` and is the
//     only place the packing's stage walk is spelled out.
//   * Decode is a GEMV. The tensor core does nothing for one row, so it is a
//     DP4A kernel: one thread per output feature, the trit walk from the fork's
//     own loader, and a K split to fill the card.

#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "llama_mmq/common_shim.cuh"
#include "llama_mmq/mma.cuh"
#include "llama_mmq/vecdotq.cuh"
#include "llama_mmq/mmq.cuh"

namespace pocket {
namespace {

constexpr int kQuantizeBlockSize = 128;

// Programmatic launch dependency sync is Hopper+; no-op on Turing.
__device__ __forceinline__ void pdl_sync_noop() {}

// ---------------------------------------------------------------------------
// Activation quantization, prefill: fp32 -> block_q8_1_mmq in the D4 layout,
// which stores one fp32 scale per 32 values and no partial sum, because the
// weight side carries its own offset. Verbatim from llama.cpp quantize.cu with
// only the ids / multi-channel machinery dropped.
// ---------------------------------------------------------------------------
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_mmq_q8_1_kernel(
        const float* __restrict__ x, const int32_t* __restrict__ ids, void* __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    pdl_sync_noop();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4* x4 = (const float4*) x;
    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;

    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / QK8_1);
    const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4 * QK8_1);

    const float4 xi = i0 < ne00
        ? x4[(i03 * s03 + i02 * s02 + i01 * s01 + i00) / 4]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    // The partial sum is only consumed by the DS4 layout; a D4 weight block
    // carries its own offset, so there is nothing to subtract.
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d = amax / ((1 << 7) - 1);
    const float d_inv = d > 0 ? 1.0f / d : 0.0f;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs / 32].x = __float2half(d);
        y[ib].ds4[iqs / 32].y = __float2half(sum);
    } else if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        y[ib].d4[iqs / 32] = d;
    }

    int8_t* ys = y[ib].qs;
    ys[iqs + 0] = __float2int_rn(xi.x * d_inv);
    ys[iqs + 1] = __float2int_rn(xi.y * d_inv);
    ys[iqs + 2] = __float2int_rn(xi.z * d_inv);
    ys[iqs + 3] = __float2int_rn(xi.w * d_inv);
}

void quantize_act_q8_1_mmq(const float* x, void* vy, int64_t ne00, int ne1, cudaStream_t stream) {
    const int64_t ne0 = ne00;  // the caller guarantees K % (4*QK8_1) == 0
    const int64_t block_num_y =
        (ne0 + 4 * kQuantizeBlockSize - 1) / (4 * kQuantizeBlockSize);
    const dim3 num_blocks(ne1, block_num_y, 1);
    const dim3 block_size(kQuantizeBlockSize, 1, 1);
    quantize_mmq_q8_1_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
        <<<num_blocks, block_size, 0, stream>>>(x, nullptr, vy, ne00, ne00, 0, 0, ne0, ne1, 1);
}

// ---------------------------------------------------------------------------
// Prefill: the tiled mul_mat_q walk, stripped of stream-K, ids and channels.
//
//   weight blocks [N, blocks_per_row] of block_ptq1_0
//   activation    [rows, blocks_per_row] of block_q8_1_mmq
//   dst           [rows, N] fp32
// ---------------------------------------------------------------------------
template <int mmq_x>
__global__ void ptq1_0_prefill_kernel(
        const char* __restrict__ x,        // weight blocks
        const int* __restrict__ y,         // activation blocks, as int*
        float* __restrict__ dst,           // [rows, N]
        const int stride_row_x,            // blocks per weight row
        const int ncols_y,                 // rows
        const int nrows_x,                 // N
        const int ncols_dst) {             // rows
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int mmq_y = get_mmq_y_device();

    const int it = blockIdx.x;  // weight tile
    const int jt = blockIdx.y;  // activation tile

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > mmq_x && j >= mmq_x) {
            break;
        }
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int offset_x = it * mmq_y * stride_row_x;
    const int offset_y = (jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
    // dst is [rows, N] row-major, so the weight tile fixes the column and the
    // activation tile fixes the row.
    const int offset_dst = jt * mmq_x * nrows_x + it * mmq_y;

    const int tile_x_max_i = nrows_x - it * mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;

    // Bounds are always checked: a Qwen3.5 shape's row count divides a tile only
    // by accident, and N is 5,120 or 17,408 rather than a multiple of 128.
    constexpr bool need_check = true;
    constexpr bool fixup = false;
    mul_mat_q_process_tile<GGML_TYPE_PTQ1_0, mmq_x, need_check, fixup>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, nrows_x,
        tile_x_max_i, tile_y_max_j, 0, stride_row_x);
}

// Shared memory the tile walk needs, exactly as `mul_mat_q_process_tile` lays it
// out: the identity ids, then the activation tile, then the weight tile at a
// stride of MMQ_MMA_TILE_X_K_Q8_0 ints per weight row.
//
// This is computed rather than quoted because the sum sits close to the 64 KiB
// opt-in limit and because the activation tile is the term that is easy to get
// wrong: it is `mmq_x * MMQ_TILE_Y_K` ints, not one packed activation block per
// row. Getting it wrong by a few hundred ints is not a compile error or a launch
// failure -- it is a silent read past the end of dynamic shared memory, which on
// this card is only rewarded with the right answer while one block happens to be
// resident per SM.
int prefill_shared_bytes(int mmq_x) {
    constexpr int mmq_y = 128;  // get_mmq_y_host(Turing)
    const int nwarps = mmq_get_nwarps_host(GGML_CUDA_CC_TURING, 32);
    const int ids = mmq_x;
    const int activation = GGML_PAD(mmq_x * MMQ_TILE_Y_K, nwarps * 32);
    const int weights = mmq_y * MMQ_MMA_TILE_X_K_Q8_0;
    return (ids + activation + weights) * (int) sizeof(int);
}

// The activation-tile widths this file instantiates. The choice has to come from
// this list and not from arithmetic alone: the kernel's tile stride, the ids
// array it fills, and the grid's y extent all have to agree on one number, so a
// width that tiles `rows` evenly is not enough on its own -- it also has to be a
// width that exists. Picking, say, 48 because it divides the row count would
// launch the 8-wide kernel over a grid sized for 48 and quietly compute the
// first eight rows.
template <int mmq_x>
bool launch_prefill_tile(const char* x_w, const int* y_q, float* dst, int n, int rows,
                         int blocks_per_row, cudaStream_t stream) {
    constexpr int mmq_y = 128;
    const int nbytes_shared = prefill_shared_bytes(mmq_x);
    const int nty = (n + mmq_y - 1) / mmq_y;
    const dim3 block_dims(32, mmq_get_nwarps_host(GGML_CUDA_CC_TURING, 32), 1);
    const dim3 grid(nty, (rows + mmq_x - 1) / mmq_x, 1);

    // The MMA tile needs more than the default 48 KiB per block, so the limit has
    // to be raised once per instantiation (2080 Ti allows 64 KiB opt-in).
    static bool raised = false;
    if (!raised) {
        if (cudaFuncSetAttribute((const void*) ptq1_0_prefill_kernel<mmq_x>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 nbytes_shared) != cudaSuccess) {
            cudaGetLastError();  // clear: the caller falls back to a narrower tile
            return false;
        }
        raised = true;
    }
    ptq1_0_prefill_kernel<mmq_x><<<grid, block_dims, nbytes_shared, stream>>>(
        x_w, y_q, dst, blocks_per_row, rows, n, rows);
    return cudaGetLastError() == cudaSuccess;
}

// The widest tile that divides `rows`, then narrower ones if the shared-memory
// request cannot be granted. `need_check` covers a `rows` that divides nothing,
// so the 8-wide tile is always a valid answer.
bool launch_prefill(const char* x_w, const int* y_q, float* dst, int n, int rows,
                    int blocks_per_row, cudaStream_t stream) {
    bool launched = false;
    if (rows % 128 == 0) {
        launched = launch_prefill_tile<128>(x_w, y_q, dst, n, rows, blocks_per_row, stream);
    }
    if (!launched && rows % 64 == 0) {
        launched = launch_prefill_tile<64>(x_w, y_q, dst, n, rows, blocks_per_row, stream);
    }
    if (!launched && rows % 32 == 0) {
        launched = launch_prefill_tile<32>(x_w, y_q, dst, n, rows, blocks_per_row, stream);
    }
    if (!launched && rows % 16 == 0) {
        launched = launch_prefill_tile<16>(x_w, y_q, dst, n, rows, blocks_per_row, stream);
    }
    if (!launched) {
        launched = launch_prefill_tile<8>(x_w, y_q, dst, n, rows, blocks_per_row, stream);
    }
    return launched;
}

// ---------------------------------------------------------------------------
// Decode: activation quantized to block_q8_1 (32 values, 36 bytes), then a DP4A
// GEMV over the ternary blocks.
// ---------------------------------------------------------------------------
struct block_q8_1_decode {
    __half d;
    __half s;
    int8_t qs[32];
};

// One warp quantizes 32 blocks, one per threadIdx.y. The launch must give this
// kernel a full 32-deep y dimension, or only the first block of each x index is
// written and the rest of the activation buffer is uninitialized.
constexpr int kQ8_1DecodeBlocksPerWarp = 32;

__global__ void quantize_q8_1_decode_kernel(const float* __restrict__ x, void* __restrict__ vy,
                                            int k, int n_blocks) {
    const int block_id = blockIdx.x * kQ8_1DecodeBlocksPerWarp + threadIdx.y;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x;
    const int base = block_id * 32;

    const float xi = (base + lane < k) ? x[base + lane] : 0.0f;
    float amax = fabsf(xi);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    }
    const float d = amax / 127.0f;
    const float d_inv = d > 0.0f ? 1.0f / d : 0.0f;
    const int8_t q = (int8_t) __float2int_rn(xi * d_inv);
    float sum = xi;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    }

    block_q8_1_decode* y = (block_q8_1_decode*) vy;
    if (lane == 0) {
        y[block_id].d = __float2half(d);
        y[block_id].s = __float2half(sum);
    }
    y[block_id].qs[lane] = q;
}

// One 128-weight block dotted against its four 32-value activation blocks. The
// block spans exactly four activation blocks, so there are four integer sums and
// one scale multiply each at the end -- which keeps the half load and the float
// multiply out of the 32-step inner loop.
__device__ __forceinline__ void ptq1_0_block_dot(
        const block_ptq1_0* __restrict__ b,
        const block_q8_1_decode* __restrict__ a,
        float& acc) {
    int sumi[4] = {0, 0, 0, 0};

    // Weights 0..79: bytes 0..15, four trit streams per 32-bit word, one 16 apart.
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint32_t packed = get_int_b4(b->qs, g);
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF;
            v_hi = w_hi & 0x00FF00FF;
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
        uint32_t v_lo = __byte_perm(packed, 0, 0x4140);
        uint32_t v_hi = __byte_perm(packed, 0, 0x4342);
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const uint32_t w_lo = v_lo * 3;
            const uint32_t w_hi = v_hi * 3;
            v_lo = w_lo & 0x00FF00FF;
            v_hi = w_hi & 0x00FF00FF;
            const int e = 80 + t * 8 + 4 * g;
            const int q = __vsub4(__byte_perm(w_lo, w_hi, 0x7531), 0x01010101);
            const int u = get_int_b4(a[e >> 5].qs, (e & 31) >> 2);
            sumi[e >> 5] = __dp4a(q, u, sumi[e >> 5]);
        }
    }

    // qh carries weights 120..127, four trits per byte with the parity
    // interleaved: qh[0] holds 120,122,124,126 and qh[1] holds 121,123,125,127.
    {
        uint32_t v = (uint32_t) b->qh[0] | ((uint32_t) b->qh[1] << 16);
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
            const uint32_t w0 = v * 3;
            v = w0 & 0x00FF00FF;
            const uint32_t w1 = v * 3;
            v = w1 & 0x00FF00FF;
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

constexpr int kDecodeThreads = 128;

__global__ void ptq1_0_decode_kernel(
        const block_q8_1_decode* __restrict__ y,
        const uint8_t* __restrict__ x_w,
        float* __restrict__ part,
        const int bpr,
        const int n,
        const int split) {
    extern __shared__ block_q8_1_decode s_y[];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk = blockIdx.y;
    const int kb_begin = (int) ((long long) chunk * bpr / split);
    const int kb_end = (int) ((long long) (chunk + 1) * bpr / split);

    // The chunk's own slice of the activation: every thread in the block reads
    // the same address at the same time, so both the staging loop and the inner
    // loop's reads are broadcasts.
    const block_q8_1_decode* y_chunk = y + (size_t) kb_begin * (QK_PTQ1_0 / QK8_1);
    const int y_blocks = (kb_end - kb_begin) * (QK_PTQ1_0 / QK8_1);
    for (int i = threadIdx.x; i < y_blocks; i += blockDim.x) {
        s_y[i] = y_chunk[i];
    }
    __syncthreads();

    if (row >= n || chunk >= split) return;

    const uint8_t* row_w = x_w + (size_t) row * bpr * sizeof(block_ptq1_0);
    float acc = 0.0f;
#pragma unroll 2
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        ptq1_0_block_dot((const block_ptq1_0*) (row_w + (size_t) kb * sizeof(block_ptq1_0)),
                         s_y + (kb - kb_begin) * (QK_PTQ1_0 / QK8_1), acc);
    }
    part[(size_t) chunk * n + row] = acc;
}

// Fixes the split chunks onto the row in chunk order, which is the order the
// single-chunk path would have accumulated them in.
__global__ void ptq1_0_decode_reduce_kernel(const float* __restrict__ part,
                                            float* __restrict__ dst, const int n,
                                            const int split) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.0f;
    for (int c = 0; c < split; ++c) {
        acc += part[(size_t) c * n + row];
    }
    dst[row] = acc;
}

// K is split whenever one thread per row would not fill the card: a chunk shorter
// than four weight blocks costs more in partials than it buys back.
int decode_split_for(int64_t n, int bpr) {
    int split = 1;
    while (split < 8 && (n * split < 65536 || split < (bpr + 39) / 40) &&
           bpr / (split * 2) >= 4) {
        split *= 2;
    }
    return split;
}

// ---------------------------------------------------------------------------
// Element conversions. The engine has fp16->fp32 already but nothing the other
// way, and both are one line.
// ---------------------------------------------------------------------------
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __half2float(*reinterpret_cast<const __half*>(x + i));
}

__global__ void f32_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __half_as_ushort(__float2half(x[i]));
}

// ---------------------------------------------------------------------------
// Scratch. One cached buffer per thread, grown to the high-water mark, in the
// style qwen_half_ops.cu uses for its cuBLAS workspace.
// ---------------------------------------------------------------------------
struct TernaryWorkspace {
    int device = -1;
    void* buffer = nullptr;
    size_t capacity = 0;
};

TernaryWorkspace& ternary_workspace() {
    static thread_local TernaryWorkspace workspace;
    return workspace;
}

bool ensure_ternary_workspace(TernaryWorkspace& workspace, size_t bytes) {
    int current_device = 0;
    if (cudaGetDevice(&current_device) != cudaSuccess) return false;
    if (workspace.device != -1 && workspace.device != current_device) {
        cudaFree(workspace.buffer);
        workspace = {};
    }
    workspace.device = current_device;
    if (workspace.capacity < bytes) {
        cudaFree(workspace.buffer);
        workspace.buffer = nullptr;
        workspace.capacity = 0;
        if (cudaMalloc(&workspace.buffer, bytes) != cudaSuccess) return false;
        workspace.capacity = bytes;
    }
    return true;
}

size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int grid_blocks(int64_t count, int threads) {
    const int64_t blocks = (count + threads - 1) / threads;
    return (int) (blocks > 65535 ? 65535 : blocks);
}

bool convert_f16_to_f32(const uint16_t* x, float* y, int64_t count, cudaStream_t stream) {
    constexpr int kThreads = 256;
    f16_to_f32_kernel<<<grid_blocks(count, kThreads), kThreads, 0, stream>>>(x, y, (int) count);
    return cudaGetLastError() == cudaSuccess;
}

bool convert_f32_to_f16(const float* x, uint16_t* y, int64_t count, cudaStream_t stream) {
    constexpr int kThreads = 256;
    f32_to_f16_kernel<<<grid_blocks(count, kThreads), kThreads, 0, stream>>>(x, y, (int) count);
    return cudaGetLastError() == cudaSuccess;
}

// The tail the two output widths share. Both kernels leave their result in fp32
// in the workspace, so the fp16 entry point is the fp32 one plus a narrowing and
// not a second kernel: exactly one of the two destination pointers is set. The
// target head reads fp32 logits -- 152064 of them for this vocabulary -- and
// rounding them to fp16 on the way out would put the sampler's own arithmetic on
// a coarser grid than the accumulator it came from.
bool store_result(const float* dst, float* d_y_f32, uint16_t* d_y_f16, int batch,
                  int rows, int y_stride, cudaStream_t stream) {
    if (y_stride == rows) {
        if (d_y_f32 != nullptr) {
            return cudaMemcpyAsync(d_y_f32, dst, (size_t) batch * rows * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
        }
        return convert_f32_to_f16(dst, d_y_f16, (int64_t) batch * rows, stream);
    }
    for (int t = 0; t < batch; ++t) {
        if (d_y_f32 != nullptr) {
            if (cudaMemcpyAsync(d_y_f32 + (size_t) t * y_stride, dst + (size_t) t * rows,
                                (size_t) rows * sizeof(float), cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return false;
            }
        } else if (!convert_f32_to_f16(dst + (size_t) t * rows, d_y_f16 + (size_t) t * y_stride,
                                       rows, stream)) {
            return false;
        }
    }
    return true;
}

bool check_shape(int rows, int cols, int batch) {
    if (rows <= 0 || cols <= 0 || batch <= 0) {
        return false;
    }
    // Every packing assumption below -- one activation block per 128 weights,
    // four of them per weight block -- is a division by 128.
    return cols % QK_PTQ1_0 == 0;
}

}  // namespace

bool ptq1_0_matmul_rows(const uint16_t* d_x_fp16, const uint8_t* d_blocks,
                        float* d_y_f32, uint16_t* d_y_f16, int batch, int rows,
                        int cols, int x_stride, int y_stride, cudaStream_t stream) {
    if (!check_shape(rows, cols, batch)) return false;
    if (x_stride < cols || y_stride < rows) return false;

    const int64_t x_elements = (int64_t) batch * cols;
    const int64_t y_elements = (int64_t) batch * rows;
    const int blocks_per_row = cols / QK_PTQ1_0;

    const size_t x_f32_bytes = align_up((size_t) x_elements * sizeof(float), 256);
    const size_t act_bytes = align_up(
        (size_t) batch * blocks_per_row * sizeof(block_q8_1_mmq), 256);
    const size_t dst_bytes = align_up((size_t) y_elements * sizeof(float), 256);

    TernaryWorkspace& workspace = ternary_workspace();
    if (!ensure_ternary_workspace(workspace, x_f32_bytes + act_bytes + dst_bytes)) return false;
    uint8_t* base = static_cast<uint8_t*>(workspace.buffer);
    float* x_f32 = reinterpret_cast<float*>(base);
    void* act = base + x_f32_bytes;
    float* dst = reinterpret_cast<float*>(base + x_f32_bytes + act_bytes);

    // The activation arrives fp16 because that is what the engine's op surface
    // carries; the quantizer is llama.cpp's and takes fp32, so it is widened here
    // rather than duplicated in fp16.
    if (x_stride == cols) {
        if (!convert_f16_to_f32(d_x_fp16, x_f32, x_elements, stream)) return false;
    } else {
        for (int t = 0; t < batch; ++t) {
            if (!convert_f16_to_f32(d_x_fp16 + (size_t) t * x_stride, x_f32 + (size_t) t * cols,
                                    cols, stream)) {
                return false;
            }
        }
    }

    quantize_act_q8_1_mmq(x_f32, act, cols, batch, stream);

    if (!launch_prefill(reinterpret_cast<const char*>(d_blocks),
                        reinterpret_cast<const int*>(act), dst, rows, batch, blocks_per_row,
                        stream)) {
        return false;
    }

    return store_result(dst, d_y_f32, d_y_f16, batch, rows, y_stride, stream);
}

bool qwen_ptq1_0_matmul_rows_f16_cuda(const uint16_t* d_x_fp16, const uint8_t* d_blocks,
                                      uint16_t* d_y_fp16, int batch, int rows, int cols,
                                      int x_stride, int y_stride, void* stream_ptr) {
    return ptq1_0_matmul_rows(d_x_fp16, d_blocks, nullptr, d_y_fp16, batch, rows, cols,
                              x_stride, y_stride,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
}

bool qwen_ptq1_0_matmul_rows_f16_f32_cuda(const uint16_t* d_x_fp16, const uint8_t* d_blocks,
                                          float* d_y_f32, int batch, int rows, int cols,
                                          int x_stride, int y_stride, void* stream_ptr) {
    return ptq1_0_matmul_rows(d_x_fp16, d_blocks, d_y_f32, nullptr, batch, rows, cols,
                              x_stride, y_stride,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
}

bool ptq1_0_matvec(const uint16_t* d_x_fp16, const uint8_t* d_blocks, float* d_y_f32,
                   uint16_t* d_y_f16, int rows, int cols, cudaStream_t stream) {
    if (!check_shape(rows, cols, 1)) return false;

    const int blocks_per_row = cols / QK_PTQ1_0;
    const int n_q8_blocks = cols / QK8_1;
    const int split = decode_split_for(rows, blocks_per_row);

    const size_t x_f32_bytes = align_up((size_t) cols * sizeof(float), 256);
    const size_t act_bytes = align_up((size_t) n_q8_blocks * sizeof(block_q8_1_decode), 256);
    // The partials are split*rows and the result is rows, and both are needed at
    // once when the split is taken, so they are two buffers rather than one reused.
    const size_t part_bytes = align_up((size_t) split * rows * sizeof(float), 256);
    const size_t dst_bytes = align_up((size_t) rows * sizeof(float), 256);

    TernaryWorkspace& workspace = ternary_workspace();
    if (!ensure_ternary_workspace(
            workspace, x_f32_bytes + act_bytes + part_bytes + dst_bytes)) {
        return false;
    }
    uint8_t* base = static_cast<uint8_t*>(workspace.buffer);
    float* x_f32 = reinterpret_cast<float*>(base);
    void* act = base + x_f32_bytes;
    float* part = reinterpret_cast<float*>(base + x_f32_bytes + act_bytes);
    float* dst = reinterpret_cast<float*>(base + x_f32_bytes + act_bytes + part_bytes);

    if (!convert_f16_to_f32(d_x_fp16, x_f32, cols, stream)) return false;

    const int nblocks = (n_q8_blocks + kQ8_1DecodeBlocksPerWarp - 1) / kQ8_1DecodeBlocksPerWarp;
    quantize_q8_1_decode_kernel<<<nblocks, dim3(32, kQ8_1DecodeBlocksPerWarp), 0, stream>>>(
        x_f32, act, cols, n_q8_blocks);
    if (cudaGetLastError() != cudaSuccess) return false;

    // The widest chunk is ceil(bpr/split) blocks and their four activation blocks each.
    const int smem = ((blocks_per_row + split - 1) / split * (QK_PTQ1_0 / QK8_1)) *
                     (int) sizeof(block_q8_1_decode);
    const dim3 block(kDecodeThreads, 1, 1);
    const dim3 grid((rows + kDecodeThreads - 1) / kDecodeThreads, split, 1);

    const block_q8_1_decode* y_q = reinterpret_cast<const block_q8_1_decode*>(act);
    if (split == 1) {
        ptq1_0_decode_kernel<<<grid, block, smem, stream>>>(y_q, d_blocks, dst, blocks_per_row,
                                                           rows, 1);
        if (cudaGetLastError() != cudaSuccess) return false;
        return store_result(dst, d_y_f32, d_y_f16, 1, rows, rows, stream);
    }

    ptq1_0_decode_kernel<<<grid, block, smem, stream>>>(y_q, d_blocks, part, blocks_per_row, rows,
                                                       split);
    if (cudaGetLastError() != cudaSuccess) return false;
    ptq1_0_decode_reduce_kernel<<<grid_blocks(rows, 256), 256, 0, stream>>>(part, dst, rows, split);
    if (cudaGetLastError() != cudaSuccess) return false;
    return store_result(dst, d_y_f32, d_y_f16, 1, rows, rows, stream);
}

bool qwen_ptq1_0_matvec_f16_cuda(const uint16_t* d_x_fp16, const uint8_t* d_blocks,
                                 uint16_t* d_y_fp16, int rows, int cols, void* stream_ptr) {
    return ptq1_0_matvec(d_x_fp16, d_blocks, nullptr, d_y_fp16, rows, cols,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
}

bool qwen_ptq1_0_matvec_f16_f32_cuda(const uint16_t* d_x_fp16, const uint8_t* d_blocks,
                                     float* d_y_f32, int rows, int cols, void* stream_ptr) {
    return ptq1_0_matvec(d_x_fp16, d_blocks, d_y_f32, nullptr, rows, cols,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
}

// ---------------------------------------------------------------------------
// The embedding table, read where it lies.
//
// A row lookup against a ternary table is the same problem as a row against an
// fp16 one, and the difference decides the memory budget: expanding the table at
// load costs 5120 fp16 per row -- 2.4 GiB for a 248,320-token vocabulary, on a
// card where the whole 27B model is 5.5 GiB -- while reading it as blocks costs
// 1120 bytes per row actually looked up. The expanded table does not fit beside a
// KV cache on one card; this is what lets it not have to.
//
// The value is exact: a trit is -1, 0 or 1 and the block scale is a finite fp16,
// so the product is representable in fp16 without rounding, and the lookup
// reproduces the reference bit for bit rather than nearly.
// ---------------------------------------------------------------------------

// The stage walk that decodes one weight of a block, from the format's own
// definition: two qs stages of 16 and 8 bytes over weights 0..79 and 80..119,
// then qh's eight weights with its two bytes interleaved by parity. The
// arithmetic is uint8_t in the reference implementation, so the wrap below is
// masked in rather than avoided.
__device__ __forceinline__ int ptq1_0_trit_at(const block_ptq1_0* __restrict__ b,
                                              int index) {
    const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    uint8_t byte = 0;
    int shift = 1;
    if (index < 80) {
        byte = b->qs[index % 16];
        shift = pow3[index / 16];
    } else if (index < 120) {
        const int within = index - 80;
        byte = b->qs[16 + within % 8];
        shift = pow3[within / 8];
    } else {
        const int within = index - 120;
        byte = b->qh[within % 2];
        shift = pow3[within / 2];
    }
    const uint8_t q = static_cast<uint8_t>(byte * shift);
    return ((q * 3) >> 8) - 1;
}

// One thread per weight: 128 threads for a block's 128 weights, one block per
// (row, 128-weight run). Every thread reads the whole 28-byte block, which is one
// cache line and the same line for all of them. A token this rank does not hold
// gathers zeros, because the caller sums the ranks' rows.
__global__ void ptq1_0_embedding_gather_kernel(
        const uint8_t* __restrict__ table, const int* __restrict__ tokens,
        uint16_t* __restrict__ output, int cols, int row_start, int row_count) {
    const int row = blockIdx.y;
    const int group = blockIdx.x;
    const int index = threadIdx.x;
    const long destination =
        static_cast<long>(row) * cols + static_cast<long>(group) * QK_PTQ1_0 + index;
    const int token = tokens[row];
    if (token < row_start || token >= row_start + row_count) {
        output[destination] = 0;
        return;
    }
    const int groups_per_row = cols / QK_PTQ1_0;
    const block_ptq1_0* block = reinterpret_cast<const block_ptq1_0*>(
        table + (static_cast<size_t>(token - row_start) * groups_per_row + group) *
                    sizeof(block_ptq1_0));
    const float value = static_cast<float>(ptq1_0_trit_at(block, index)) *
                        __half2float(block->d);
    output[destination] = __half_as_ushort(__float2half(value));
}

bool qwen_embedding_ptq1_0_gather_f16_cuda(const uint8_t* d_table_blocks,
                                           const int* d_tokens, uint16_t* d_out_fp16,
                                           int count, int cols, int row_start,
                                           int row_count, void* stream_ptr) {
    if (d_table_blocks == nullptr || d_tokens == nullptr || d_out_fp16 == nullptr) {
        return false;
    }
    if (count <= 0 || cols <= 0 || cols % QK_PTQ1_0 != 0 || row_start < 0 ||
        row_count <= 0) {
        return false;
    }
    const dim3 grid(cols / QK_PTQ1_0, count, 1);
    ptq1_0_embedding_gather_kernel<<<grid, QK_PTQ1_0, 0,
                                     reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        d_table_blocks, d_tokens, d_out_fp16, cols, row_start, row_count);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
