// MiMo-V2.6's decode-step rotation, as one kernel instead of ten dispatches.
//
// `device_attention.rope_rows` is the reference's `apply_partial_rope` with the head and sequence
// axes folded away: a split, a half swap, two multiplies and an add, thirty-two times a token (the
// query and the key of every one of the forty-eight layers). It is elementwise, so it should be
// nothing -- and it is 135.4 us of host time a call on this box, which is what ten eager
// dispatches cost, so it is 13.0 ms of a decode token's host before anything has been computed.
// This kernel is 10.7 us of the same, and what those 12.3 ms are worth to the *token* is a
// question for the measurement rather than for this file: at twenty resident rows it is 17.2 ms
// and at sixteen it is 9.2, because the smaller the resident set the further ahead the host runs.
//
// The arithmetic is reproduced exactly rather than approximated, and the two places that matter
// are stated here because they are the whole reason this file is safe to drop in:
//
//   * The products round *separately*. `rope` is bfloat16 and the cosine is float32, so
//     `rope * cos` is a float32 multiply of an exactly-widened bfloat16 and it rounds once; the
//     reference then adds the two rounded products in float32. `__fmul_rn` and `__fadd_rn` below
//     keep that order, and `--use_fast_math` would otherwise be free to contract the pair into an
//     fma and produce a different last bit.
//   * The half swap is a sign flip and not a subtraction. `rotated` is `cat(-x2, x1)` and the
//     reference's sum is `(x * cos) + (rotated * sin)`, so the first half is `p1 + (-p2)`, which is
//     exact and is not the same expression as `fma(-x2, sin, p1)`.
//
// Everything past `rope_dim` is the pass-through half of the row, and it is written as the float32
// widening of the input because the reference's `torch.cat` promotes it: the returned tensor is
// float32 either way, and `attention_output` casts it back itself.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

// One block a head. `head_dim` is 192 for this checkpoint and `rope_dim` is 64 of it, so the block
// walks the row once and the grid is as wide as the layer has heads.
constexpr int kThreads = 128;

// The widening the reference does implicitly: `rope * cos` promotes a bfloat16 row to float32, and
// that promotion is exact for every dtype this path carries.
template <typename scalar_t>
__device__ __forceinline__ float widen(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__global__ void mimo_rope_rows_kernel(
    const scalar_t* __restrict__ states,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ out,
    const int head_dim,
    const int rope_dim) {
  const int head = blockIdx.x;
  const int half = rope_dim >> 1;
  const scalar_t* source = states + static_cast<long>(head) * head_dim;
  float* target = out + static_cast<long>(head) * head_dim;

  for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
    float value;
    if (column < rope_dim) {
      const float first = widen<scalar_t>(source[column]);
      if (column < half) {
        const float second = widen<scalar_t>(source[column + half]);
        value = __fadd_rn(__fmul_rn(first, cos[column]), -__fmul_rn(second, sin[column]));
      } else {
        const float second = widen<scalar_t>(source[column - half]);
        value = __fadd_rn(__fmul_rn(first, cos[column]), __fmul_rn(second, sin[column]));
      }
    } else {
      value = widen<scalar_t>(source[column]);
    }
    target[column] = value;
  }
}

}  // namespace

// `rope_rows(states, cos, sin, dim)`, the reference's signature and its returned dtype.
//
// `states` is `[heads, head_dim]` and must be contiguous: a caller that has the fused qkv output
// slices it and asks for a view, which is contiguous, and one that does not is asked to fix it
// before the call rather than silently reading the wrong stride.
torch::Tensor mimo_rope_rows(
    const torch::Tensor& states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t rope_dim) {
  TORCH_CHECK(states.dim() == 2, "rope states are [heads, head_dim], got ", states.dim(), " dims");
  TORCH_CHECK(states.is_contiguous(), "rope states must be contiguous");
  TORCH_CHECK(states.is_cuda(), "rope states are on the card");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "the rope tables are on the same card as the row");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "the rope tables are float32; a narrower one would be a different rounding");
  TORCH_CHECK(cos.numel() == rope_dim && sin.numel() == rope_dim,
              "the rope tables hold one entry a rotated coordinate");

  const int64_t heads = states.size(0);
  const int64_t head_dim = states.size(1);
  TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim && rope_dim % 2 == 0,
              "rope_dim ", rope_dim, " does not halve into ", head_dim);
  TORCH_CHECK(heads > 0, "an empty rope row is not a rotation");

  const torch::Tensor flat_cos = cos.contiguous().reshape({-1});
  const torch::Tensor flat_sin = sin.contiguous().reshape({-1});
  torch::Tensor out = torch::empty({heads, head_dim}, states.options().dtype(at::kFloat));

  const dim3 grid(static_cast<unsigned>(heads));
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      states.scalar_type(),
      "mimo_rope_rows",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        mimo_rope_rows_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            states.data_ptr<at::BFloat16>(),
            flat_cos.data_ptr<float>(),
            flat_sin.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(head_dim),
            static_cast<int>(rope_dim));
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        mimo_rope_rows_kernel<at::Half><<<grid, block, 0, stream>>>(
            states.data_ptr<at::Half>(),
            flat_cos.data_ptr<float>(),
            flat_sin.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(head_dim),
            static_cast<int>(rope_dim));
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        mimo_rope_rows_kernel<float><<<grid, block, 0, stream>>>(
            states.data_ptr<float>(),
            flat_cos.data_ptr<float>(),
            flat_sin.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(head_dim),
            static_cast<int>(rope_dim));
      }));

  return out;
}
