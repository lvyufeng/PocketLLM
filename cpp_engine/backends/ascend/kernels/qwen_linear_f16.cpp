// AscendC linear-attention support kernels for first-generation 910: the gate
// projection and the causal depthwise convolution with SiLU.

#include "qwen_ascend_kernel_common.hpp"

namespace {

using namespace pocket;

// Tile width for the elementwise gate math. 512 floats keeps every vector op at
// full width while leaving room for the six live buffers.
constexpr uint32_t kGateTile = 512;

// Largest convolution kernel the tail buffer supports, matching kMaxTail on the
// CUDA side.
constexpr uint32_t kMaxKernel = 8;

// Channel tile for the convolution. Each of the kernel's taps needs its own
// float row, so kMaxKernel * kConvTile floats is the working set: 8 * 512 * 4 =
// 16 KiB.
constexpr uint32_t kConvTile = 512;

// Above this input, softplus(x) and x agree to better than one FP32 ulp, because
// exp(-x) < 2^-24. It is also far below where exp(x) overflows, which is what makes
// the clamp below sufficient rather than merely helpful.
constexpr float kSoftplusLinear = 20.0f;

// softplus(x) = log1p(exp(x)), built from the two vector transcendentals that do
// exist. Both Exp and Ln are near-exact here (~6e-8), so this is as accurate as the
// CUDA log1pf for the magnitudes the gates see; log1p's advantage only shows for
// exp(x) very close to zero, where g is dominated by -exp(a_log) anyway.
//
// The clamp is not optional. exp(x) overflows to inf above x = 88, and Ln(inf) is
// inf, so an unclamped version returns inf for large positive x where the right
// answer is x itself. Real dt_bias + a values do reach that range. The clamp caps
// the exp argument and Max restores the identity branch afterwards, so both halves
// of the domain come out of the same vector ops with no divergence. `work` is
// clobbered.
//
// The clamp is a Duplicate plus a tensor-tensor Min rather than the one-instruction
// `Mins`. First-generation 910 has no vmins/vmaxs: `Ascend910B.ini` lists
// `Intrinsic_vmin` and `Intrinsic_vmax` but no scalar-operand form, and dav_c100
// implements Mins/Maxs as ASCENDC_REPORT_NOT_SUPPORT stubs. Those stubs compile and
// launch; they simply do not write `dst`, so the clamped value silently stays
// whatever the buffer held before.
__aicore__ inline void softplus(const AscendC::LocalTensor<float>& dst,
                                const AscendC::LocalTensor<float>& src,
                                const AscendC::LocalTensor<float>& work,
                                uint32_t count) {
    AscendC::Duplicate(work, kSoftplusLinear, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Min(work, src, work, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(dst, work, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(dst, dst, 1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Ln(dst, dst, count);
    AscendC::PipeBarrier<PIPE_V>();
    // For x <= kSoftplusLinear the Ln result already dominates x; above it the Ln
    // result is the saturated constant and x is the answer.
    AscendC::Max(dst, dst, src, count);
    AscendC::PipeBarrier<PIPE_V>();
}

// sigmoid(x) = 1/(1 + exp(-x)).
//
// The reciprocal is a vector Reciprocal, which is only ~2e-3 accurate on this part.
// That is acceptable here and nowhere else in this file: beta is immediately rounded
// to fp16, whose own spacing near 1.0 is 4.9e-4, so a Newton step would buy less
// than half a bit of the stored result.
__aicore__ inline void sigmoid(const AscendC::LocalTensor<float>& dst,
                               const AscendC::LocalTensor<float>& src,
                               uint32_t count) {
    AscendC::Muls(dst, src, -1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(dst, dst, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(dst, dst, 1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Reciprocal(dst, dst, count);
    AscendC::PipeBarrier<PIPE_V>();
}

// SiLU(x) = x * sigmoid(x). Same reciprocal argument as above: the result is stored
// as fp16.
__aicore__ inline void silu(const AscendC::LocalTensor<float>& dst,
                            const AscendC::LocalTensor<float>& src,
                            const AscendC::LocalTensor<float>& work,
                            uint32_t count) {
    sigmoid(work, src, count);
    AscendC::Mul(dst, src, work, count);
    AscendC::PipeBarrier<PIPE_V>();
}

}  // namespace

// Gate projection: g = -exp(a_log[head]) * softplus(a + dt_bias[head]),
// beta = sigmoid(b), over an [rows, heads] pair of fp16 planes.
//
// a_log and dt_bias are per-head, so a head's whole column shares two scalars. But
// the natural tiling is the other way round: a and b are row-major [rows, heads], so
// a contiguous tile spans heads, not rows. Rather than gather a strided column, this
// walks contiguous tiles of the flat [rows*heads] plane and builds a per-lane
// broadcast of the two head parameters once, since head = index % heads is periodic
// with period `heads` and every tile is a whole number of head groups when
// kGateTile is a multiple of heads. It is not in general, so the broadcast is
// rebuilt per tile from the head index of each lane; `heads` is 32 or 64 in
// practice, so that is at most kGateTile/heads Duplicates.
extern "C" __global__ __aicore__ void qwen_linear_attn_gates_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR g,
    GM_ADDR beta, uint32_t rows, uint32_t heads) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> half_buf, val_buf, par_buf, work_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> head_half_buf, head_float_buf;
    pipe.InitBuffer(half_buf, kGateTile * sizeof(half));
    pipe.InitBuffer(val_buf, kGateTile * sizeof(float));
    pipe.InitBuffer(par_buf, kGateTile * sizeof(float));
    pipe.InitBuffer(work_buf, kGateTile * sizeof(float));
    pipe.InitBuffer(head_half_buf, kGateTile * sizeof(half));
    pipe.InitBuffer(head_float_buf, kGateTile * sizeof(float));

    const uint32_t total = rows * heads;
    AscendC::GlobalTensor<half> a_gm, b_gm, a_log_gm, dt_gm, g_gm, beta_gm;
    a_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a), total);
    b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), total);
    a_log_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a_log), heads);
    dt_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dt_bias), heads);
    g_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(g), total);
    beta_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta), total);

    AscendC::LocalTensor<half> staging = half_buf.Get<half>();
    AscendC::LocalTensor<float> value = val_buf.Get<float>();
    AscendC::LocalTensor<float> param = par_buf.Get<float>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();
    AscendC::LocalTensor<half> head_staging = head_half_buf.Get<half>();
    AscendC::LocalTensor<float> head_values = head_float_buf.Get<float>();

    const uint32_t tiles = (total + kGateTile - 1) / kGateTile;
    for (uint32_t tile = AscendC::GetBlockIdx(); tile < tiles;
         tile += AscendC::GetBlockNum()) {
        const uint32_t offset = tile * kGateTile;
        const uint32_t count = min_u32(kGateTile, total - offset);
        const uint32_t vector_count = pocket_align_up(count, kAlignFloat);

        // beta = sigmoid(b): no per-head parameter, so it goes first and frees the
        // parameter buffer for the g pass. Vector instructions operate on a complete
        // eight-float block; zero the padding so a partial final tile is still safe.
        load_half_exact(staging, b_gm, offset, count);
        for (uint32_t i = count; i < vector_count; ++i) staging.SetValue(i, 0);
        wait_scalar_before_compute();
        AscendC::Cast(value, staging, AscendC::RoundMode::CAST_NONE, vector_count);
        AscendC::PipeBarrier<PIPE_V>();
        sigmoid(work, value, vector_count);
        AscendC::Cast(staging, work, AscendC::RoundMode::CAST_NONE, vector_count);
        store_half_exact(beta_gm, offset, staging, count);
        // `staging` is overwritten by the following MTE2 load. `work` is not the
        // source of the MTE3 transfer, so only the staging-buffer lifetime matters.
        wait_store_before_load();

        // g = -exp(a_log) * softplus(a + dt_bias). Lay dt_bias out per lane, add,
        // softplus, then scale by the per-lane -exp(a_log).
        load_half_exact(staging, a_gm, offset, count);
        for (uint32_t i = count; i < vector_count; ++i) staging.SetValue(i, 0);
        wait_scalar_before_compute();
        AscendC::Cast(value, staging, AscendC::RoundMode::CAST_NONE, vector_count);
        AscendC::PipeBarrier<PIPE_V>();

        load_half_exact(head_staging, dt_gm, 0, heads);
        AscendC::Cast(head_values, head_staging, AscendC::RoundMode::CAST_NONE, heads);
        AscendC::PipeBarrier<PIPE_V>();
        wait_compute_before_scalar();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint32_t head = (offset + lane) % heads;
            param.SetValue(lane, head_values.GetValue(head));
        }
        for (uint32_t lane = count; lane < vector_count; ++lane) {
            param.SetValue(lane, 0.0f);
        }
        wait_scalar_before_compute();
        AscendC::Add(value, value, param, vector_count);
        AscendC::PipeBarrier<PIPE_V>();
        // param is free again once the sum is in `value`, so it doubles as the
        // softplus clamp scratch.
        softplus(work, value, param, vector_count);

        load_half_exact(head_staging, a_log_gm, 0, heads);
        AscendC::Cast(head_values, head_staging, AscendC::RoundMode::CAST_NONE, heads);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(head_values, head_values, heads);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(head_values, head_values, -1.0f, heads);
        AscendC::PipeBarrier<PIPE_V>();
        wait_compute_before_scalar();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint32_t head = (offset + lane) % heads;
            param.SetValue(lane, head_values.GetValue(head));
        }
        for (uint32_t lane = count; lane < vector_count; ++lane) {
            param.SetValue(lane, 0.0f);
        }
        wait_scalar_before_compute();
        AscendC::Mul(work, work, param, vector_count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(staging, work, AscendC::RoundMode::CAST_NONE, vector_count);
        store_half_exact(g_gm, offset, staging, count);
        wait_store_before_load();
    }
}

// Causal depthwise convolution followed by SiLU, over an [seq_len, channels] fp16
// plane with a per-channel [channels, kernel] weight and a [kernel-1, channels]
// carried tail.
//
// The convolution is depthwise, so channels are independent and the tiling is over
// channel tiles: each core owns kConvTile channels for the whole sequence. That
// keeps the tap rows contiguous in the channel axis, which is the axis the data is
// contiguous in, so every load is a straight DataCopy and the accumulation is a
// plain Axpy against a broadcast weight.
//
// The weight is [channels, kernel], i.e. the taps of one channel are contiguous and
// the channels of one tap are strided. A tile therefore needs a per-lane gather of
// `kernel` values, done on the scalar unit once per tile rather than once per token.
extern "C" __global__ __aicore__ void qwen_causal_depthwise_conv_silu_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR tail, GM_ADDR y, uint32_t seq_len,
    uint32_t channels, uint32_t kernel, uint32_t update_tail) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> half_buf, acc_buf, work_buf, tap_buf, w_buf;
    pipe.InitBuffer(half_buf, kConvTile * sizeof(half));
    pipe.InitBuffer(acc_buf, kConvTile * sizeof(float));
    pipe.InitBuffer(work_buf, kConvTile * sizeof(float));
    pipe.InitBuffer(tap_buf, kMaxKernel * kConvTile * sizeof(float));
    pipe.InitBuffer(w_buf, kMaxKernel * kConvTile * sizeof(float));

    const uint32_t tail_len = kernel - 1;
    AscendC::GlobalTensor<half> x_gm, w_gm, tail_gm, y_gm;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), seq_len * channels);
    w_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), channels * kernel);
    y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), seq_len * channels);
    if (tail != nullptr) {
        tail_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(tail),
                                tail_len * channels);
    }

    AscendC::LocalTensor<half> staging = half_buf.Get<half>();
    AscendC::LocalTensor<float> acc = acc_buf.Get<float>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();
    AscendC::LocalTensor<float> taps = tap_buf.Get<float>();
    AscendC::LocalTensor<float> weights = w_buf.Get<float>();

    const uint32_t tiles = (channels + kConvTile - 1) / kConvTile;
    for (uint32_t tile = AscendC::GetBlockIdx(); tile < tiles;
         tile += AscendC::GetBlockNum()) {
        const uint32_t base = tile * kConvTile;
        const uint32_t count = min_u32(kConvTile, channels - base);

        // Gather this tile's weights: weights[tap * kConvTile + lane] is channel
        // (base + lane)'s tap `tap`.
        for (uint32_t lane = 0; lane < count; ++lane) {
            for (uint32_t tap = 0; tap < kernel; ++tap) {
                weights.SetValue(tap * kConvTile + lane,
                                 static_cast<float>(w_gm.GetValue((base + lane) * kernel + tap)));
            }
        }
        wait_scalar_before_compute();

        // Prime the tap window with the carried tail: taps[j] holds the input at
        // relative position j - tail_len, so the first token's window is exactly
        // the tail followed by x[0].
        //
        // Every iteration reloads the same `staging` tile, so the Cast that drains it
        // has to be fenced against the next load. load_half_exact only supplies the
        // forward MTE2->V hand-off; without the V->MTE2 anti-dependency here, MTE2
        // issues copy j+1 while Vector is still reading copy j, and every tap slot
        // ends up holding the following row. That shifts the whole primed window by
        // one and is wrong only for the first output token, which is the one token
        // whose window is entirely made of primed slots.
        for (uint32_t j = 0; j < tail_len; ++j) {
            if (tail != nullptr) {
                load_half_exact(staging, tail_gm, j * channels + base, count);
                AscendC::Cast(taps[j * kConvTile], staging,
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                wait_compute_before_load();
            } else {
                AscendC::Duplicate(taps[j * kConvTile], 0.0f, count);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        for (uint32_t t = 0; t < seq_len; ++t) {
            // Slide the window: the newest sample takes the last tap slot.
            load_half_exact(staging, x_gm, t * channels + base, count);
            AscendC::Cast(taps[tail_len * kConvTile], staging,
                          AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();

            // Accumulate the taps oldest-first, matching the CUDA loop order so the
            // FP32 rounding sequence is the same.
            AscendC::Mul(acc, taps, weights, count);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tap = 1; tap < kernel; ++tap) {
                AscendC::Mul(work, taps[tap * kConvTile], weights[tap * kConvTile],
                             count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(acc, acc, work, count);
                AscendC::PipeBarrier<PIPE_V>();
            }

            silu(acc, acc, work, count);
            AscendC::Cast(staging, acc, AscendC::RoundMode::CAST_NONE, count);
            store_half_exact(y_gm, t * channels + base, staging, count);
            // The output transfer reads only `staging`, which the next iteration
            // refills. The FP32 window and accumulation buffers are independent.
            wait_store_before_load();

            // Shift the window down one slot for the next token. The source and
            // destination overlap, so this must use memmove semantics. DataCopy does
            // not guarantee that: `Adds(dst, src, 0.0f)` calls MTE, whose direction
            // is undefined when the tensors overlap. Per-element copying permits a
            // newest-to-oldest ordering that avoids clobbering, but staging a whole
            // row is simpler than three Adds calls.
            for (uint32_t j = 0; j + 1 < kernel; ++j) {
                AscendC::Adds(work, taps[(j + 1) * kConvTile], 0.0f, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(taps[j * kConvTile], work, 0.0f, count);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // The carried tail is the last tail_len inputs, which the window already
        // holds in slots 1..kernel-1 after the final shift.
        if (update_tail != 0 && tail != nullptr && tail_len > 0) {
            for (uint32_t j = 0; j < tail_len; ++j) {
                AscendC::Cast(staging, taps[j * kConvTile],
                              AscendC::RoundMode::CAST_NONE, count);
                store_half_exact(tail_gm, j * channels + base, staging, count);
                // Next writer of `staging` is the following Cast, not a load, so the
                // store has to drain against Vector rather than against MTE2.
                wait_store_before_compute();
            }
        }
    }
}
