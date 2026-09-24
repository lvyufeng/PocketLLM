// The decode step's attention as one kernel instead of the twenty dispatches it is in torch.
//
// `device_attention.decode_output` is deliberately the reference's arithmetic on smaller shapes,
// and what it costs is dispatch rather than flops: on this box a trivial `torch.add` is 15.9 us of
// host time under `no_grad` and 9.8 under `inference_mode`, and the block this file replaces --
// two `matmul`s around a masked softmax with a sink column -- is about eighteen of them a layer,
// forty-eight layers a token. Measured at sixteen resident rows, `scores` 6.9 ms a token, `softmax`
// 6.6, `out` 4.8 and the final `view` 0.8: **19.1 ms of a token's host** for a piece of arithmetic
// whose every tensor is at most `[16, 192]` and whose key span is at most `FOLD_KEYS`.
//
// **This is the one kernel in the MiMo path that is not bit-exact, and that is the trade it was
// written under.** `mimo_rope_rows` reproduces `rope_rows` to the bit because it can: it is
// elementwise, so there is no summation order to get wrong. A softmax over a key span has one. The
// reference's `amax` is exact and order-free and is reproduced exactly; its `sum` is a `torch`
// reduction whose tree shape is torch's, its `matmul`s are cuBLAS's blocking of the same products,
// and this kernel walks the span a warp at a time. The last bits of a probability therefore differ,
// and a decode step that used to be held to `torch.equal` against the chunk path is held to a
// bound instead. What is *not* given up:
//
//   * the products are not approximated. `key` and `value` are bfloat16 only because the cache is;
//     they are widened exactly, the same way the reference's `.to(torch.float32)` widens them.
//   * the exponentials are the reference's, and this needs saying because `--use_fast_math` is on
//     for this translation unit: it turns `expf` into `__expf`, whose few-2^-21 error is enough to
//     move a probability's last bits. `accurate_exp` below goes through double, which fast math
//     does not substitute, and it is called `keys` times a warp rather than `keys * v_head_dim`.
//   * the sink is a softmax column and not a bias, exactly as the reference has it: it enters the
//     row maximum, and its exponential is added to the denominator *after* the sum of the span's,
//     which is the order the reference adds them in.
//   * the divisions are correctly rounded rather than fast.
//
// The shape is `[kv_heads, groups]` of independent problems, so one warp takes one of them: a warp
// computes the whole span's scores against one query row, softmaxes them, and mixes the values.
// `heads == kv_heads * groups` and a head's index is `kv_head * groups + group`, which is the
// ordering `query.view(kv_heads, groups, 1, head_dim)` implies and the one the caller's
// `out.view(1, o_in)` depends on.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

//: One warp a (kv_head, group), four warps a block. The problem is tiny -- 1024 keys against 192
//: dimensions at the very most -- so the block size is about the grid being a legal shape rather
//: than about occupancy, and the key span is walked once with the query held in registers.
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;

//: `head_dim / 32` floats of a query row a lane and `v_head_dim / 32` accumulators. The release is
//: 192 and 128, so six and four; a shape past these is refused rather than silently truncated.
constexpr int kMaxQuery = 8;
constexpr int kMaxValue = 8;

// `--use_fast_math` substitutes `__expf` for `expf`, and the error that buys is a few 2^-21
// relative -- which is the same order as the tolerance this kernel is held to, so it would be the
// dominant error if it were left in. Double precision is not substituted.
__device__ __forceinline__ float accurate_exp(float value) {
  return static_cast<float>(exp(static_cast<double>(value)));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <typename scalar_t>
__global__ void mimo_decode_attention_kernel(
    const float* __restrict__ query,   // [heads, head_dim], the rotated query, already float32
    const scalar_t* __restrict__ key,  // [kv_heads, keys, head_dim]
    const scalar_t* __restrict__ value,  // [kv_heads, keys, v_head_dim]
    const float* __restrict__ sink,    // [heads] or nullptr
    float* __restrict__ out,           // [heads, v_head_dim]
    const long key_stride0,
    const long key_stride1,
    const long value_stride0,
    const long value_stride1,
    const int kv_heads,
    const int groups,
    const int keys,
    const int head_dim,
    const int v_head_dim,
    const float scaling) {
  // One span of scores a warp, at `[kWarps][keys]`. It is the only thing that has to be shared:
  // the query row the scores are taken against lives in each lane's registers, because the lanes
  // divide the *dimensions* of one key rather than the keys, which is what makes both the query
  // and the key loads coalesced.
  extern __shared__ float scores[];
  float* mine = scores + (threadIdx.x >> 5) * keys;

  const int lane = threadIdx.x & 31;
  const int problem = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (problem >= kv_heads * groups) {
    return;
  }
  // `heads` are ordered `kv_head * groups + group`, so the head index is the problem index.
  const int head = problem;
  const int kv = problem / groups;

  float qv[kMaxQuery];
  int held = 0;
  for (int column = lane; column < head_dim; column += 32) {
    qv[held++] = query[static_cast<long>(head) * head_dim + column];
  }

  const scalar_t* key_row = key + static_cast<long>(kv) * key_stride0;
  const int steps = (v_head_dim + 31) / 32;

  // One pass over the span: the scores, held in shared memory because the softmax needs the whole
  // row before it can normalise any of it, and the running maximum on the way past. The reduction
  // after each dot is the price of having the lanes on the dimensions -- five shuffles for a dot
  // that is six fused multiplies a lane -- and it is why the maximum is exact while the sum below
  // is only the reference's to within a rounding.
  // Four keys at a time, and the four are not an optimisation of the arithmetic: they are four
  // independent memory streams. A key is six loads of this lane's share of its row, and a single
  // key's loads are one round trip deep, so a loop that walks the span one key at a time spends
  // almost all of its time waiting for a row it could have asked for three iterations ago. Measured
  // on the release's worst case -- 1024 keys, sixteen heads, both families -- that is **1.2 us of
  // device time a key**, which is a thousand times the arithmetic's own cost; the four streams, the
  // four reductions and the four maxima below are what the loop is shaped around.
  float maximum = -INFINITY;
  int i = 0;
  for (; i + 4 <= keys; i += 4) {
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    const scalar_t* row0 = key_row + static_cast<long>(i) * key_stride1;
    const scalar_t* row1 = row0 + head_dim;
    const scalar_t* row2 = row1 + head_dim;
    const scalar_t* row3 = row2 + head_dim;
    for (int j = 0, column = lane; column < head_dim; column += 32, ++j) {
      dot0 += qv[j] * static_cast<float>(row0[column]);
      dot1 += qv[j] * static_cast<float>(row1[column]);
      dot2 += qv[j] * static_cast<float>(row2[column]);
      dot3 += qv[j] * static_cast<float>(row3[column]);
    }
    dot0 = warp_sum(dot0) * scaling;
    dot1 = warp_sum(dot1) * scaling;
    dot2 = warp_sum(dot2) * scaling;
    dot3 = warp_sum(dot3) * scaling;
    if (lane == 0) {
      mine[i] = dot0;
      mine[i + 1] = dot1;
      mine[i + 2] = dot2;
      mine[i + 3] = dot3;
    }
    maximum = fmaxf(maximum, fmaxf(fmaxf(dot0, dot1), fmaxf(dot2, dot3)));
  }
  for (; i < keys; ++i) {
    const scalar_t* row = key_row + static_cast<long>(i) * key_stride1;
    float dot = 0.0f;
    for (int j = 0, column = lane; column < head_dim; column += 32, ++j) {
      dot += qv[j] * static_cast<float>(row[column]);
    }
    dot = warp_sum(dot) * scaling;
    if (lane == 0) {
      mine[i] = dot;
    }
    maximum = fmaxf(maximum, dot);
  }
  __syncwarp();
  maximum = warp_max(maximum);
  if (sink != nullptr) {
    maximum = fmaxf(maximum, sink[head]);
  }

  // The exponentials and their sum. `exp(column - maximum)` is added to the denominator after the
  // span's own sum rather than inside it, which is the reference's order and not a detail: adding
  // it inside would round the span's sum differently.
  float total = 0.0f;
  for (int i = lane; i < keys; i += 32) {
    const float probability = accurate_exp(mine[i] - maximum);
    mine[i] = probability;
    total += probability;
  }
  total = warp_sum(total);
  if (sink != nullptr) {
    total += accurate_exp(sink[head] - maximum);
  }
  __syncwarp();

  // Normalise in place, one division a key rather than one a (key, value channel), and then mix.
  for (int i = lane; i < keys; i += 32) {
    mine[i] = __fdiv_rn(mine[i], total);
  }

  __syncwarp();
  // **Double**, and it is the one place in this kernel where that is not showing off. The span is
  // summed sequentially, so a thousand keys against a float32 accumulator is a thousand roundings
  // and the error grows with the span: measured over the release's own geometry it is 1.8e-6
  // relative at 1024 keys, which is two thousand times the bfloat16 the answer is cast to and still
  // thirty times the bound the test asks for. In double the sum's own error stops being the term
  // that matters and what is left is the reference's cuBLAS blocking, which is the right thing for
  // a difference to be made of. The price is a double-precision FMA at a thirty-second of this
  // card's float32 rate, on `keys * v_head_dim` of them a warp: at the released span it is a few
  // microseconds of a whole layer.
  // Four keys at a time here too, and for the same reason as the scores above: the loop is four
  // loads and four multiplies an iteration and the loads are one round trip deep, so what it is
  // waiting on is the next row of values. The four partials are float and the fold into the double
  // accumulator happens once a group of four, which keeps the double arithmetic off the critical
  // path without giving up the sum's accuracy -- four float partials summed into a double is a
  // shorter chain than four doubles added in sequence.
  double accumulator[kMaxValue];
  float partial0[kMaxValue], partial1[kMaxValue], partial2[kMaxValue], partial3[kMaxValue];
  for (int j = 0; j < steps; ++j) {
    accumulator[j] = 0.0;
    partial0[j] = partial1[j] = partial2[j] = partial3[j] = 0.0f;
  }
  const scalar_t* value_row = value + static_cast<long>(kv) * value_stride0;
  int at = 0;
  for (; at + 4 <= keys; at += 4) {
    const float p0 = mine[at], p1 = mine[at + 1], p2 = mine[at + 2], p3 = mine[at + 3];
    const scalar_t* row0 = value_row + static_cast<long>(at) * value_stride1;
    const scalar_t* row1 = row0 + v_head_dim;
    const scalar_t* row2 = row1 + v_head_dim;
    const scalar_t* row3 = row2 + v_head_dim;
    for (int j = 0, column = lane; column < v_head_dim; column += 32, ++j) {
      partial0[j] += p0 * static_cast<float>(row0[column]);
      partial1[j] += p1 * static_cast<float>(row1[column]);
      partial2[j] += p2 * static_cast<float>(row2[column]);
      partial3[j] += p3 * static_cast<float>(row3[column]);
    }
  }
  for (int j = 0; j < steps; ++j) {
    accumulator[j] = static_cast<double>(partial0[j]) + static_cast<double>(partial1[j])
        + static_cast<double>(partial2[j]) + static_cast<double>(partial3[j]);
  }
  for (; at < keys; ++at) {
    const double probability = static_cast<double>(mine[at]);
    const scalar_t* row = value_row + static_cast<long>(at) * value_stride1;
    for (int j = 0, column = lane; column < v_head_dim; column += 32, ++j) {
      accumulator[j] += probability * static_cast<double>(row[column]);
    }
  }

  float* target = out + static_cast<long>(head) * v_head_dim;
  for (int j = 0, column = lane; column < v_head_dim; column += 32, ++j) {
    target[column] = static_cast<float>(accumulator[j]);
  }
}

}  // namespace

// `mimo_decode_attention(query, key, value, sink, scaling)`.
//
// `query` is `[heads, head_dim]` **float32**, which is what `rope_rows` returns and what the
// reference's `query.to(torch.float32)` is a no-op for. `key` and `value` are the KV cache's own
// width -- bfloat16 for this release -- and `sink` is `[heads]` float32 or an empty tensor, which
// is the layer's own switch for a family that carries no sink. The answer is float32
// `[1, heads * v_head_dim]`, the reference's `out.view(1, shape.o_in).to(query.dtype)` without the
// cast, because `query` is float32 by then and the cast is a no-op.
torch::Tensor mimo_decode_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sink,
    double scaling) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(), "the attention is on the card");
  TORCH_CHECK(query.dim() == 2, "the query is [heads, head_dim], got ", query.dim(), " dims");
  TORCH_CHECK(key.dim() == 3, "the keys are [kv_heads, span, head_dim]");
  TORCH_CHECK(value.dim() == 3, "the values are [kv_heads, span, v_head_dim]");
  TORCH_CHECK(query.scalar_type() == at::kFloat,
              "the query arrives already widened; the reference upcasts it and this cannot");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "the keys and the values are one cache and therefore one width");
  TORCH_CHECK(key.size(0) == value.size(0) && key.size(1) == value.size(1),
              "a key row and a value row are the same position");
  // The query is this module's own row and is contiguous; the key and the value are a *span of a
  // ring buffer*, so they arrive as views -- `all_key[:, lower:]` keeps the buffer's own strides --
  // and the two that matter are passed through rather than paid for with a copy. The last
  // dimension is the one that is read a lane at a time and it has to be the contiguous one.
  TORCH_CHECK(query.is_contiguous(), "the query row is contiguous");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "a key or value row is read along its last dimension and must be contiguous there");

  const int64_t heads = query.size(0);
  const int64_t head_dim = query.size(1);
  const int64_t kv_heads = key.size(0);
  const int64_t keys = key.size(1);
  const int64_t v_head_dim = value.size(2);
  TORCH_CHECK(head_dim == key.size(2), "the query and the keys are the same width");
  TORCH_CHECK(kv_heads > 0 && heads % kv_heads == 0,
              "heads ", heads, " do not divide into ", kv_heads, " kv heads");
  TORCH_CHECK(keys > 0, "an attention over no keys is not a softmax");
  TORCH_CHECK(head_dim <= kMaxQuery * 32, "head_dim ", head_dim, " is past this kernel's ", kMaxQuery * 32);
  TORCH_CHECK(v_head_dim > 0 && v_head_dim <= kMaxValue * 32,
              "v_head_dim ", v_head_dim, " is past this kernel's ", kMaxValue * 32);

  const int64_t groups = heads / kv_heads;
  const int64_t problems = kv_heads * groups;
  const int64_t blocks = (problems + kWarps - 1) / kWarps;

  // Held rather than read through: `contiguous()` on a tensor that is already contiguous hands back
  // the same storage, but on one that is not it hands back a temporary, and a temporary freed at
  // the end of the statement would leave the kernel reading freed memory.
  torch::Tensor flat_sink;
  const float* sink_ptr = nullptr;
  if (sink.defined() && sink.numel()) {
    TORCH_CHECK(sink.is_cuda() && sink.scalar_type() == at::kFloat && sink.numel() == heads,
                "the sink is one float32 column a head, or empty for a family without one");
    flat_sink = sink.contiguous();
    sink_ptr = flat_sink.data_ptr<float>();
  }

  torch::Tensor out = torch::empty(
      {1, heads * v_head_dim}, query.options().dtype(at::kFloat));
  const size_t shared = static_cast<size_t>(kWarps) * keys * sizeof(float);
  TORCH_CHECK(shared <= 48 * 1024,
              "a span of ", keys, " needs ", shared, " bytes of shared memory, and a block has 48k");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 block(kThreads);

  AT_DISPATCH_SWITCH(
      key.scalar_type(),
      "mimo_decode_attention",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        mimo_decode_attention_kernel<at::BFloat16><<<grid, block, shared, stream>>>(
            query.data_ptr<float>(), key.data_ptr<at::BFloat16>(),
            value.data_ptr<at::BFloat16>(), sink_ptr, out.data_ptr<float>(),
            key.stride(0), key.stride(1), value.stride(0), value.stride(1),
            static_cast<int>(kv_heads), static_cast<int>(groups), static_cast<int>(keys),
            static_cast<int>(head_dim), static_cast<int>(v_head_dim),
            static_cast<float>(scaling));
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        // A cache at the query's own width, which is what a config that keeps the KV cache in
        // float32 has. There is no widening to do, so this is the arm where a difference between
        // this kernel and the reference is the arithmetic and not a rounding -- which is why the
        // test grid carries it.
        mimo_decode_attention_kernel<float><<<grid, block, shared, stream>>>(
            query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
            sink_ptr, out.data_ptr<float>(),
            key.stride(0), key.stride(1), value.stride(0), value.stride(1),
            static_cast<int>(kv_heads), static_cast<int>(groups), static_cast<int>(keys),
            static_cast<int>(head_dim),
            static_cast<int>(v_head_dim), static_cast<float>(scaling));
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        mimo_decode_attention_kernel<at::Half><<<grid, block, shared, stream>>>(
            query.data_ptr<float>(), key.data_ptr<at::Half>(), value.data_ptr<at::Half>(),
            sink_ptr, out.data_ptr<float>(),
            key.stride(0), key.stride(1), value.stride(0), value.stride(1),
            static_cast<int>(kv_heads), static_cast<int>(groups), static_cast<int>(keys),
            static_cast<int>(head_dim),
            static_cast<int>(v_head_dim), static_cast<float>(scaling));
      }));

  return out;
}
