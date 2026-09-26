// Xing4.0-29B-A4B's matrix hyper-connection, as one kernel.
//
// The eager port in `src/models/xing4_0/hyper_connection.py` is 300 launches a
// call, and the trunk makes two calls a layer over 40 layers -- 24000 dispatches
// a decode token to do 0.09 ms of arithmetic, which stage 4 measured at 30.6 ms
// of GPU and 183 ms of wall a token.  Nothing here is new arithmetic: it is the
// same norm, the same 24-wide gate, the same 20-iteration Sinkhorn and the same
// rebuild, arranged so that all of it happens once, with the gate's warp
// reduction and the two block-wide ones around the Sinkhorn being the only
// synchronizations in the kernel.
//
// Cost.  Measured on the 2080 Ti, back to back inside a CUDA graph so that no
// host submission is counted: ~62 us a call at the released shape on one row and
// ~79 us on 64, of which the Sinkhorn is ~24 us (its comment says why) and the
// width-proportional part -- the norm over 14336 and the 24-wide gate -- is the
// remaining ~27 us at 14336 against 5 us at 64.  A call is therefore mostly
// latency: one block of eight warps has little to overlap, and the two loops
// that read the 672 KiB projection are ~7 GB/s rather than the card's 616.
// Splitting the row across blocks would fix that and cost a second launch, which
// at 80 calls a token is the wrong trade while the step is launch-bound.
//
// Shape.  `hidden` is [rows, hc, hidden] and its four streams are flattened to
// `wide = hc * hidden` for the norm and the gate, because the checkpoint's own
// `Xing4_0UnweightedRMSNorm` runs over the flattened 14336 -- the streams share
// one scale, and a per-stream norm would be a different model.
//
// Threading.  One block a row.  Eight warps split the gate's 24 outputs three
// ways (outputs w, w+8, w+16), which is also the widest this gets for
// `hc_mult = 4`; the gate's dot is the only real work and it is a GEMV whose
// weight side is 672 KiB.  The Sinkhorn runs on one thread, because a 4x4 matrix
// with a data dependency between its two normalizations has no parallelism to
// find, and 20 iterations of it is about a microsecond.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kThreads / kWarpSize;
//: `hc_mult` is 4 in the released checkpoint.  The arrays below are sized for
//: half a warp of streams, which is where 24 outputs still divide evenly into
//: eight warps three ways; the kernel checks the bound rather than trusting it.
constexpr int kMaxHc = 8;
constexpr int kMaxMix = (2 + kMaxHc) * kMaxHc;

template <typename scalar_t>
__global__ void xing4_hyper_connection_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ fn,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    scalar_t* __restrict__ post_out,
    scalar_t* __restrict__ comb_out,
    scalar_t* __restrict__ collapsed_out,
    int rows,
    int hc,
    int hidden_size,
    int mix,
    int iters,
    float eps,
    float clamp_min,
    float clamp_max) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int wide = hc * hidden_size;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const scalar_t* row_x = hidden + static_cast<int64_t>(row) * wide;

    __shared__ float shared[kMaxMix + 2 * kMaxHc + kMaxHc * kMaxHc];

    // -- the unweighted norm, over the flattened streams ---------------------
    float square_sum = 0.0f;
    for (int k = threadIdx.x; k < wide; k += kThreads) {
        const float value = static_cast<float>(row_x[k]);
        square_sum = fmaf(value, value, square_sum);
    }
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
    }
    if (lane == 0) shared[warp] = square_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < kMaxWarps; ++w) total += shared[w];
        shared[0] = rsqrtf(total / static_cast<float>(wide) + eps);
    }
    __syncthreads();
    const float inv_rms = shared[0];

    // -- the gate: `mix` outputs, each a dot over `wide` ---------------------
    {
        float acc[3] = {0.0f, 0.0f, 0.0f};
        const int out0 = warp;
        const int out1 = warp + kMaxWarps;
        const int out2 = warp + 2 * kMaxWarps;
        const scalar_t* f0 = fn + static_cast<int64_t>(out0) * wide;
        const scalar_t* f1 = fn + static_cast<int64_t>(out1) * wide;
        const scalar_t* f2 = fn + static_cast<int64_t>(out2) * wide;
        #pragma unroll 8
        for (int k = lane; k < wide; k += kWarpSize) {
            // The reference runs this projection in the model dtype and casts to
            // fp32 only for the gates, so the normalized value is rounded to
            // `scalar_t` before it is multiplied -- not after, and not at all in
            // the default fp32 port.
            const float value = static_cast<float>(
                static_cast<scalar_t>(static_cast<float>(row_x[k]) * inv_rms));
            acc[0] = fmaf(static_cast<float>(f0[k]), value, acc[0]);
            acc[1] = fmaf(static_cast<float>(f1[k]), value, acc[1]);
            acc[2] = fmaf(static_cast<float>(f2[k]), value, acc[2]);
        }
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc[0] += __shfl_down_sync(0xffffffff, acc[0], offset);
            acc[1] += __shfl_down_sync(0xffffffff, acc[1], offset);
            acc[2] += __shfl_down_sync(0xffffffff, acc[2], offset);
        }
        if (lane == 0) {
            shared[out0] = acc[0];
            shared[out1] = acc[1];
            shared[out2] = acc[2];
        }
    }
    __syncthreads();

    // -- the three gates and the Sinkhorn ------------------------------------
    const int gate_pre = mix;
    const int gate_post = mix + kMaxHc;
    const int gate_comb = mix + 2 * kMaxHc;
    if (threadIdx.x == 0) {
        for (int s = 0; s < hc; ++s) {
            shared[gate_pre + s] = 1.0f / (1.0f + expf(-(shared[s] * scale[0] + base[s])));
        }
        for (int s = 0; s < hc; ++s) {
            const int idx = hc + s;
            shared[gate_post + s] = 2.0f / (1.0f + expf(-(shared[idx] * scale[1] + base[idx])));
        }
        // The iterate lives in shared memory rather than in a local `comb[8][8]`.
        // Every index below is a runtime `d`/`s`, so a register array would be
        // placed in local memory and its ~1300 accesses would be dynamic round
        // trips.  This one thread is also the kernel's serial section: at the
        // released `iters=20` and `hc=4` it is ~24 us of the kernel's ~62 us, at
        // ~60 cycles a division, because each iteration's 32 divisions are a
        // dependent chain of row-pass then column-pass.  Parallelizing it across
        // `hc` threads would take that to ~6 us; it is left alone because the
        // decode step it sits in is bound by 5829 launches, not by this kernel.
        for (int d = 0; d < hc; ++d) {
            for (int s = 0; s < hc; ++s) {
                const int idx = 2 * hc + d * hc + s;
                // The clamp is on the logits, before the exponential: +-30 is far
                // outside fp32's ability to represent exp, so it is a guard.
                shared[gate_comb + d * hc + s] =
                    fminf(fmaxf(shared[idx] * scale[2] + base[idx], clamp_min), clamp_max);
            }
            // The row max is over the clamped logits, and it is taken in place so
            // that no array is indexed by `d`.
            float row_max = -INFINITY;
            for (int s = 0; s < hc; ++s) row_max = fmaxf(row_max, shared[gate_comb + d * hc + s]);
            for (int s = 0; s < hc; ++s) {
                shared[gate_comb + d * hc + s] = expf(shared[gate_comb + d * hc + s] - row_max);
            }
        }
        // Rows then columns, `iters` times, with `eps` added to each denominator.
        // The order is the reference's and the iteration count is a config value.
        for (int step = 0; step < iters; ++step) {
            for (int d = 0; d < hc; ++d) {
                float sum = eps;
                for (int s = 0; s < hc; ++s) sum += shared[gate_comb + d * hc + s];
                for (int s = 0; s < hc; ++s) shared[gate_comb + d * hc + s] /= sum;
            }
            for (int s = 0; s < hc; ++s) {
                float sum = eps;
                for (int d = 0; d < hc; ++d) sum += shared[gate_comb + d * hc + s];
                for (int d = 0; d < hc; ++d) shared[gate_comb + d * hc + s] /= sum;
            }
        }
    }
    __syncthreads();

    scalar_t* row_post = post_out + static_cast<int64_t>(row) * hc;
    scalar_t* row_comb = comb_out + static_cast<int64_t>(row) * hc * hc;
    if (threadIdx.x < hc) {
        row_post[threadIdx.x] = static_cast<scalar_t>(shared[gate_post + threadIdx.x]);
    }
    if (threadIdx.x < hc * hc) {
        row_comb[threadIdx.x] = static_cast<scalar_t>(shared[gate_comb + threadIdx.x]);
    }

    // -- the collapsed input: one stream, `pre`-weighted ----------------------
    scalar_t* row_out = collapsed_out + static_cast<int64_t>(row) * hidden_size;
    for (int d = threadIdx.x; d < hidden_size; d += kThreads) {
        float acc = 0.0f;
        for (int s = 0; s < hc; ++s) {
            const float pre = static_cast<float>(static_cast<scalar_t>(shared[gate_pre + s]));
            acc = fmaf(pre, static_cast<float>(row_x[s * hidden_size + d]), acc);
        }
        row_out[d] = static_cast<scalar_t>(acc);
    }
}

}  // namespace

std::vector<torch::Tensor> xing4_hyper_connection_forward_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& fn,
    const torch::Tensor& base,
    const torch::Tensor& scale,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps,
    double clamp_min,
    double clamp_max) {
    const int64_t hc = hc_mult;
    TORCH_CHECK(hidden.dim() == 3, "hidden must have shape [rows, hc, hidden]");
    TORCH_CHECK(hidden.size(1) == hc, "hidden's stream axis is not hc_mult");
    TORCH_CHECK(hc >= 1 && hc <= kMaxHc, "hc_mult outside the kernel's range: ", hc);
    const int64_t mix = (2 + hc) * hc;
    TORCH_CHECK(fn.size(0) == mix, "fn must have (2 + hc) * hc rows");
    TORCH_CHECK(mix <= 3 * kMaxWarps, "the gate's outputs do not divide into the block's warps");
    TORCH_CHECK(base.numel() == mix && scale.numel() == 3, "hc_base/hc_scale have the wrong sizes");
    TORCH_CHECK(fn.size(1) == hc * hidden.size(2), "fn's width is not hc * hidden");
    TORCH_CHECK(hidden.is_cuda() && fn.is_cuda(), "hidden and fn must be CUDA tensors");
    TORCH_CHECK(hidden.is_contiguous() && fn.is_contiguous(), "hidden and fn must be contiguous");
    TORCH_CHECK(base.is_contiguous() && scale.is_contiguous(), "hc_base/hc_scale must be contiguous");
    TORCH_CHECK(base.scalar_type() == torch::kFloat32 && scale.scalar_type() == torch::kFloat32,
                "hc_base and hc_scale are fp32 in both releases");
    TORCH_CHECK(fn.scalar_type() == hidden.scalar_type(), "fn and hidden must share a dtype");

    c10::cuda::CUDAGuard device_guard(hidden.device());
    const auto options = hidden.options();
    auto post = torch::empty({hidden.size(0), hc}, options);
    auto comb = torch::empty({hidden.size(0), hc, hc}, options);
    auto collapsed = torch::empty({hidden.size(0), hidden.size(2)}, options);

    const int rows = static_cast<int>(hidden.size(0));
    const int hidden_size = static_cast<int>(hidden.size(2));
    if (rows > 0) {
        const dim3 grid(rows);
        const dim3 block(kThreads);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, hidden.scalar_type(), "xing4_hyper_connection", [&] {
                xing4_hyper_connection_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    hidden.data_ptr<scalar_t>(),
                    fn.data_ptr<scalar_t>(),
                    base.data_ptr<float>(),
                    scale.data_ptr<float>(),
                    post.data_ptr<scalar_t>(),
                    comb.data_ptr<scalar_t>(),
                    collapsed.data_ptr<scalar_t>(),
                    rows,
                    static_cast<int>(hc),
                    hidden_size,
                    static_cast<int>(mix),
                    static_cast<int>(sinkhorn_iters),
                    static_cast<float>(eps),
                    static_cast<float>(clamp_min),
                    static_cast<float>(clamp_max));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {post, comb, collapsed};
}

std::vector<torch::Tensor> xing4_hyper_connection_forward(
    const torch::Tensor& hidden,
    const torch::Tensor& fn,
    const torch::Tensor& base,
    const torch::Tensor& scale,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps,
    double clamp_min,
    double clamp_max) {
    return xing4_hyper_connection_forward_cuda(
        hidden, fn, base, scale, hc_mult, sinkhorn_iters, eps, clamp_min, clamp_max);
}
