// AscendC gated-delta recurrence for first-generation 910.
//
// The recurrence, per head, per token, over a [key_dim, value_dim] FP32 state S,
// a normalized key row k, a normalized query row q, a value row v, a decay scalar
// and a beta scalar:
//
//     S       <- S * decay                      (elementwise, whole matrix)
//     kv_mem  <- k^T S                          (row vector, value_dim wide)
//     delta   <- (v - kv_mem) * beta            (row vector)
//     S       <- S + k delta^T                  (rank-1 update)
//     out     <- q^T S * q_scale                (row vector)
//
// Why this maps onto the vector unit the way it does:
//
//   - A head's state is 128*128 FP32 = 64 KiB, which fits UB (256 KiB) alongside
//     the row buffers, so the whole recurrence runs out of UB with two GM touches
//     per unit: load the state once, store it once.
//   - `k^T S` and `q^T S` are reductions down the key axis, i.e. across rows of the
//     state tile. Both are written as one `Axpy` per key row accumulating into a
//     value-wide row, with the key element as the instruction's scalar operand, so
//     nothing has to splat it across a row first. The alternative, one dot product
//     per value column, is 128 narrow reductions.
//   - The rank-1 update is the same instruction again: `Axpy` per key row,
//     S[i,:] += delta * k[i], which is exactly dst = src*scalar + dst.
//
// So each of the three passes over the state tile costs 128 issues and no more.
// Writing `k^T S` as broadcast-multiply-fold instead costs 2*rows + log2(rows)
// issues -- 264 rather than 128 -- and this part turns out to charge for issues
// rather than for bytes, so the accumulation form is the cheaper one by roughly
// the ratio of those two numbers.
//
// Work is spread over `heads * value_split` units, not over heads: see the comment
// on kMaxValueSplit for why the value axis and only the value axis can be cut, and
// what the cut is worth. Which split to run is a launch argument taken from the
// head count, because whether it pays depends on the head count -- but the width it
// implies stays a compile-time constant, for the reason recorded there.
//
// The accumulation order is also the more defensible one: it sums sequentially
// down the key axis, which is what the CUDA kernel and the tests' double-precision
// host reference both do, where the fold was a tree with a different rounding.
//
// The state is fp32 and stays fp32 the whole way through; only q/k/v/out are fp16.

#include "qwen_ascend_kernel_common.hpp"
#include "qwen_gated_delta_geometry.hpp"

namespace {

using namespace pocket;

// The head geometry and the value split live in a header the launcher and the
// benchmark read too, because the kernel's item count and the launcher's grid size
// are two derivations of the same number: see qwen_gated_delta_geometry.hpp.
using pocket::gated_delta::kKeyDim;
using pocket::gated_delta::kMaxValueSplit;
using pocket::gated_delta::kStateElems;
using pocket::gated_delta::kValueDim;

// Accumulators the two `k^T S` / `q^T S` reductions are spread over. Rows go to
// them round-robin, so this is also the distance between two instructions that
// depend on each other.
//
// 1, and not by omission. Spreading the rows was tried at 2, 4 and 8 groups, on
// the theory that one accumulator makes the whole loop a chain of `rows` dependent
// read-modify-writes that the vector pipe cannot overlap. Every group count was
// slower than one -- 41.6, 42.4, 44.7, 44.7 ms/layer for 1, 2, 4, 8 -- and the
// degradation is monotone in the group count, which is the signature of paying for
// the extra zeroing Duplicates and partial adds and getting nothing back. So the
// loop is not waiting on the dependence; it is simply issuing instructions at a
// fixed cost each, and the way to make it faster is to issue fewer of them.
constexpr uint32_t kAccGroups = 1;

// Which split to run is a runtime decision and why is argued on kMaxValueSplit in
// the geometry header. What belongs here is how the kernel takes it.
//
// By an `if` on a template parameter, rather than by carrying the width in a
// variable, and that is not a stylistic choice. The width is the repeat count of
// every instruction in the step, and a runtime variable stopped the vector
// intrinsics from constant-folding theirs: 41.3 -> 45.4 ms/layer on the full-width
// arm and 36.0 -> 44.4 on the split one, both measurably worse than the slower of
// the two compile-time arms. Both specializations are instantiated and the untaken
// one is dead-code eliminated, so the dispatch costs nothing.
static_assert(kMaxValueSplit >= 1 &&
                  (kMaxValueSplit & (kMaxValueSplit - 1)) == 0,
              "value split must be a power of two so the width stays aligned");
static_assert(kValueDim % kMaxValueSplit == 0,
              "value split must divide the value dimension");

// One value range of one head's state in UB. The buffers are always allocated at
// the full value width so that either split fits:
//   state      64 KiB   [kKeyDim, kValueDim] fp32
//   scratch    64 KiB   split workspace, same shape
//   rows        3 KiB   q, k (fp32, kKeyDim) and v, out (fp32, kValueDim)
//   halves      1 KiB   fp16 staging for v and out
// which leaves better than half of UB free. A split leaves the spare columns dead
// rather than resizing the buffers, which costs UB that is not scarce and keeps the
// split out of the buffer arithmetic.
template <uint32_t kValueSplit>
class GatedDeltaHead {
public:
    static constexpr uint32_t kValueWidth = kValueDim / kValueSplit;

    __aicore__ inline void Init(AscendC::TPipe& pipe) {
        pipe.InitBuffer(state_buf_, kKeyDim * kValueDim * sizeof(float));
        pipe.InitBuffer(scratch_buf_, kKeyDim * kValueDim * sizeof(float));
        pipe.InitBuffer(q_buf_, kKeyDim * sizeof(float));
        pipe.InitBuffer(k_buf_, kKeyDim * sizeof(float));
        pipe.InitBuffer(v_buf_, kValueDim * sizeof(float));
        pipe.InitBuffer(acc_buf_, kAccGroups * kValueDim * sizeof(float));
        pipe.InitBuffer(half_buf_, kValueDim * sizeof(half));
        pipe.InitBuffer(aux_buf_, kAlignFloat * sizeof(float));
    }

    // Load S for this head and value range. State layout is
    // [heads, key_dim, value_dim], so the range is a column window of the head's
    // slice: contiguous inside each row, `kValueDim` apart between rows.
    __aicore__ inline void LoadState(const AscendC::GlobalTensor<float>& state,
                                     uint32_t head, uint32_t half) {
        AscendC::LocalTensor<float> st = state_buf_.Get<float>();
        const uint32_t base = head * kKeyDim * kValueDim + half * kValueWidth;
        if (kValueSplit == 1) {
            AscendC::DataCopy(st, state[base], kKeyDim * kValueWidth);
        } else {
            AscendC::DataCopy(st, state[base],
                              window_load_params<float>(kKeyDim, kValueWidth, kValueDim));
        }
        wait_load_before_compute();
    }

    __aicore__ inline void StoreState(const AscendC::GlobalTensor<float>& state,
                                      uint32_t head, uint32_t half) {
        AscendC::LocalTensor<float> st = state_buf_.Get<float>();
        const uint32_t base = head * kKeyDim * kValueDim + half * kValueWidth;
        wait_compute_before_store();
        if (kValueSplit == 1) {
            AscendC::DataCopy(state[base], st, kKeyDim * kValueWidth);
        } else {
            AscendC::DataCopy(state[base], st,
                              window_store_params<float>(kKeyDim, kValueWidth, kValueDim));
        }
    }

    // Pull an fp32 key or query row straight out of a normalized GM buffer.
    __aicore__ inline void LoadKeyRow(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::GlobalTensor<float>& src,
                                      uint32_t offset) {
        // The row buffer is reused on the next token. MTE2->V only makes the
        // current copy visible; the reverse hand-off prevents that next copy from
        // overwriting a row that the previous Step still reads.
        wait_compute_before_load();
        AscendC::DataCopy(dst, src[offset], kKeyDim);
        wait_load_before_compute();
        // Step broadcasts individual q/k elements through the scalar unit as
        // well as consuming the completed row through Vector.
        wait_load_before_scalar();
    }

    // Pull an fp16 row and widen it. Used for v, and for q/k on the unnormalized
    // entry point.
    __aicore__ inline void LoadHalfRow(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::GlobalTensor<half>& src,
                                       uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<half> staging = half_buf_.Get<half>();
        // Cast reads staging on Vector. Do not refill it until that read has
        // retired; the forward MTE2->V event alone does not provide this
        // anti-dependency.
        wait_compute_before_load();
        AscendC::DataCopy(staging, src[offset], count);
        wait_load_before_compute();
        AscendC::Cast(dst, staging, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // L2-normalize a row in place, matching the CUDA kernel's rsqrt(sum + 1e-6).
    //
    // The reciprocal square root is taken as a scalar `1/sqrt(x)`: scalar sqrt and
    // scalar division are both exact here, whereas vector Rsqrt is a ~3e-3
    // approximation that would show up directly in the output.
    __aicore__ inline void NormalizeRow(const AscendC::LocalTensor<float>& row) {
        AscendC::LocalTensor<float> work = scratch_buf_.Get<float>();
        AscendC::Mul(work, row, row, kKeyDim);
        const float sum = fold_sum(work, kKeyDim);
        const float inv = 1.0f / sqrt(sum + 1.0e-6f);
        wait_scalar_before_compute();
        AscendC::Muls(row, row, inv, kKeyDim);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // One token of the recurrence. `q` and `k` are already normalized fp32 rows,
    // `v` an fp32 row, and the result lands back in `v`'s buffer scaled by q_scale.
    __aicore__ inline void Step(const AscendC::LocalTensor<float>& q,
                                const AscendC::LocalTensor<float>& k,
                                const AscendC::LocalTensor<float>& v,
                                float decay, float beta, float q_scale) {
        AscendC::LocalTensor<float> st = state_buf_.Get<float>();
        AscendC::LocalTensor<float> work = scratch_buf_.Get<float>();
        AscendC::LocalTensor<float> acc = acc_buf_.Get<float>();

        // S *= decay
        AscendC::Muls(st, st, decay, kKeyDim * kValueWidth);
        AscendC::PipeBarrier<PIPE_V>();

        // kv_mem = k^T S. One issue per key row, accumulated into `acc`; the old
        // form broadcast k across the state, multiplied elementwise and folded the
        // rows, which is 2*rows + log2(rows) issues for the same sum.
        accumulate_weighted_rows(acc, k, st, kKeyDim, kValueWidth, kAccGroups);

        // delta = (v - kv_mem) * beta, held in work.
        AscendC::Muls(work, acc, -1.0f, kValueWidth);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(work, work, v, kValueWidth);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(work, work, beta, kValueWidth);
        AscendC::PipeBarrier<PIPE_V>();

        // S += k delta^T, one Axpy per key row. Distinct dst rows, so the only
        // barrier needed is the one before the next read of S. The row count is the
        // key dimension and a split does not touch it, which is why this loop costs
        // the same either way and bounds what a split can buy.
        for (uint32_t i = 0; i < kKeyDim; ++i) {
            AscendC::Axpy(st[i * kValueWidth], work, k.GetValue(i), kValueWidth);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // out = q^T S * q_scale. Same accumulation over the updated state, and it
        // reuses `acc`: delta was already formed out of it above and the barrier
        // after the rank-1 update is what orders this read of S behind that write.
        accumulate_weighted_rows(acc, q, st, kKeyDim, kValueWidth, kAccGroups);
        AscendC::Muls(v, acc, q_scale, kValueWidth);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Narrow an fp32 result row to fp16 and store it.
    __aicore__ inline void StoreHalfRow(const AscendC::GlobalTensor<half>& dst,
                                        const AscendC::LocalTensor<float>& src,
                                        uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<half> staging = half_buf_.Get<half>();
        AscendC::Cast(staging, src, AscendC::RoundMode::CAST_NONE, count);
        wait_compute_before_store();
        AscendC::DataCopy(dst[offset], staging, count);
        wait_store_before_load();
    }

    __aicore__ inline AscendC::LocalTensor<float> Query() { return q_buf_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Key() { return k_buf_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Value() { return v_buf_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Aux() { return aux_buf_.Get<float>(); }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> state_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> q_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> k_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> v_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> acc_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> half_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aux_buf_;
};

// Per-token gate scalars. g and beta are [rows, heads] fp16; a head walks a column
// of stride `heads`, so reading them as a scalar per token beats staging a tile.
// decay = exp(g), and exp is vector-only on this part, so it goes through an 8-lane
// tile once per token.
template <uint32_t kValueSplit>
struct Gates {
    __aicore__ inline void Read(GatedDeltaHead<kValueSplit>& head,
                                const AscendC::GlobalTensor<half>& g,
                                const AscendC::GlobalTensor<half>& beta,
                                uint32_t index) {
        const float g_value = static_cast<float>(g.GetValue(index));
        beta_value = static_cast<float>(beta.GetValue(index));
        decay = scalar_exp(head.Aux(), g_value);
    }

    float decay;
    float beta_value;
};

// Whole-sequence recurrence over pre-normalized fp32 q/k.
//
// One head, or one value range of a head, per core iteration, round-robin over the
// 30 cores. Units are independent, so there is no cross-core communication at all;
// tokens inside a unit are strictly sequential, which is what forces the loop to
// live on one core.
template <uint32_t kValueSplit>
__aicore__ inline void sequence_normalized_body(
    GM_ADDR state, GM_ADDR q_normalized, GM_ADDR k_normalized, GM_ADDR v,
    GM_ADDR g, GM_ADDR beta, GM_ADDR out, uint32_t rows, uint32_t heads,
    uint32_t key_heads, float q_scale) {
    constexpr uint32_t width = kValueDim / kValueSplit;
    AscendC::TPipe pipe;
    GatedDeltaHead<kValueSplit> worker;
    worker.Init(pipe);

    AscendC::GlobalTensor<float> state_gm;
    AscendC::GlobalTensor<float> q_gm;
    AscendC::GlobalTensor<float> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> g_gm;
    AscendC::GlobalTensor<half> beta_gm;
    AscendC::GlobalTensor<half> out_gm;
    state_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state), heads * kStateElems);
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q_normalized), rows * key_heads * kKeyDim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_normalized), rows * key_heads * kKeyDim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), rows * heads * kValueDim);
    g_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(g), rows * heads);
    beta_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta), rows * heads);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), rows * heads * kValueDim);

    const uint32_t repeat = heads / key_heads;
    const uint32_t key_stride = key_heads * kKeyDim;
    const uint32_t units = heads * kValueSplit;

    for (uint32_t unit = AscendC::GetBlockIdx(); unit < units;
         unit += AscendC::GetBlockNum()) {
        const uint32_t head = unit / kValueSplit;
        const uint32_t half = unit % kValueSplit;
        const uint32_t key_head = head / repeat;
        worker.LoadState(state_gm, head, half);
        for (uint32_t token = 0; token < rows; ++token) {
            const uint32_t key_offset = token * key_stride + key_head * kKeyDim;
            const uint32_t value_offset =
                (token * heads + head) * kValueDim + half * width;
            AscendC::LocalTensor<float> q = worker.Query();
            AscendC::LocalTensor<float> k = worker.Key();
            AscendC::LocalTensor<float> v = worker.Value();
            worker.LoadKeyRow(q, q_gm, key_offset);
            worker.LoadKeyRow(k, k_gm, key_offset);
            worker.LoadHalfRow(v, v_gm, value_offset, width);
            Gates<kValueSplit> gates;
            gates.Read(worker, g_gm, beta_gm, token * heads + head);
            worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
            worker.StoreHalfRow(out_gm, v, value_offset, width);
        }
        worker.StoreState(state_gm, head, half);
    }
}

// Clamp a caller's split to one this kernel implements. The two specializations are
// both instantiated and the untaken branch is dead-code eliminated. Clamping rather
// than trusting the launcher keeps a host bug from launching a grid that covers only
// part of the state.
__aicore__ inline uint32_t clamp_split(uint32_t value_split) {
    return value_split >= kMaxValueSplit ? kMaxValueSplit : 1;
}

// Whole-sequence recurrence over raw fp16 q/k, normalizing each row on the fly.
//
// Same kernel as above with the normalization folded in. It exists because the
// engine's default path (and every rows == 1 decode step) takes this entry point
// without a separate normalization pass.
template <uint32_t kValueSplit>
__aicore__ inline void sequence_body(
    GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR out, uint32_t rows, uint32_t heads, uint32_t key_heads,
    float q_scale) {
    constexpr uint32_t width = kValueDim / kValueSplit;
    AscendC::TPipe pipe;
    GatedDeltaHead<kValueSplit> worker;
    worker.Init(pipe);

    AscendC::GlobalTensor<float> state_gm;
    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> g_gm;
    AscendC::GlobalTensor<half> beta_gm;
    AscendC::GlobalTensor<half> out_gm;
    state_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state), heads * kStateElems);
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), rows * key_heads * kKeyDim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), rows * key_heads * kKeyDim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), rows * heads * kValueDim);
    g_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(g), rows * heads);
    beta_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta), rows * heads);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), rows * heads * kValueDim);

    const uint32_t repeat = heads / key_heads;
    const uint32_t key_stride = key_heads * kKeyDim;
    const uint32_t units = heads * kValueSplit;

    for (uint32_t unit = AscendC::GetBlockIdx(); unit < units;
         unit += AscendC::GetBlockNum()) {
        const uint32_t head = unit / kValueSplit;
        const uint32_t half = unit % kValueSplit;
        const uint32_t key_head = head / repeat;
        worker.LoadState(state_gm, head, half);
        for (uint32_t token = 0; token < rows; ++token) {
            const uint32_t key_offset = token * key_stride + key_head * kKeyDim;
            const uint32_t value_offset =
                (token * heads + head) * kValueDim + half * width;
            AscendC::LocalTensor<float> q = worker.Query();
            AscendC::LocalTensor<float> k = worker.Key();
            AscendC::LocalTensor<float> v = worker.Value();
            worker.LoadHalfRow(q, q_gm, key_offset, kKeyDim);
            worker.NormalizeRow(q);
            worker.LoadHalfRow(k, k_gm, key_offset, kKeyDim);
            worker.NormalizeRow(k);
            worker.LoadHalfRow(v, v_gm, value_offset, width);
            Gates<kValueSplit> gates;
            gates.Read(worker, g_gm, beta_gm, token * heads + head);
            worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
            worker.StoreHalfRow(out_gm, v, value_offset, width);
        }
        worker.StoreState(state_gm, head, half);
    }
}

}  // namespace

// The entry points. Each one is a thin `if` over the two value splits, so the body
// it dispatches to is a compile-time specialization and the untaken one is dropped
// before codegen. `extern "C"` and outside the anonymous namespace, because the
// launch stub resolves these by name.

extern "C" __global__ __aicore__ void qwen_gated_delta_sequence_kernel(
    GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR out, uint32_t rows, uint32_t heads, uint32_t key_heads,
    float q_scale, uint32_t value_split) {
    if (clamp_split(value_split) >= kMaxValueSplit) {
        sequence_body<kMaxValueSplit>(state, q, k, v, g, beta, out, rows, heads,
                                      key_heads, q_scale);
    } else {
        sequence_body<1>(state, q, k, v, g, beta, out, rows, heads, key_heads,
                         q_scale);
    }
}

extern "C" __global__ __aicore__ void qwen_gated_delta_sequence_normalized_kernel(
    GM_ADDR state, GM_ADDR q_normalized, GM_ADDR k_normalized, GM_ADDR v,
    GM_ADDR g, GM_ADDR beta, GM_ADDR out, uint32_t rows, uint32_t heads,
    uint32_t key_heads, float q_scale, uint32_t value_split) {
    if (clamp_split(value_split) >= kMaxValueSplit) {
        sequence_normalized_body<kMaxValueSplit>(state, q_normalized, k_normalized,
                                                 v, g, beta, out, rows, heads,
                                                 key_heads, q_scale);
    } else {
        sequence_normalized_body<1>(state, q_normalized, k_normalized, v, g, beta,
                                    out, rows, heads, key_heads, q_scale);
    }
}

// Single-token step. The gate, q, k and v buffers hold exactly one row, so the
// indexing loses its token term; the recurrence itself is identical.
template <uint32_t kValueSplit>
__aicore__ inline void step_body(GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                 GM_ADDR g, GM_ADDR beta, GM_ADDR out,
                                 uint32_t heads, uint32_t key_heads,
                                 float q_scale) {
    constexpr uint32_t width = kValueDim / kValueSplit;
    AscendC::TPipe pipe;
    GatedDeltaHead<kValueSplit> worker;
    worker.Init(pipe);

    AscendC::GlobalTensor<float> state_gm;
    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> g_gm;
    AscendC::GlobalTensor<half> beta_gm;
    AscendC::GlobalTensor<half> out_gm;
    state_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state), heads * kStateElems);
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), key_heads * kKeyDim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), key_heads * kKeyDim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), heads * kValueDim);
    g_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(g), heads);
    beta_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta), heads);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), heads * kValueDim);

    const uint32_t repeat = heads / key_heads;
    const uint32_t units = heads * kValueSplit;

    for (uint32_t unit = AscendC::GetBlockIdx(); unit < units;
         unit += AscendC::GetBlockNum()) {
        const uint32_t head = unit / kValueSplit;
        const uint32_t half = unit % kValueSplit;
        const uint32_t key_head = head / repeat;
        AscendC::LocalTensor<float> q = worker.Query();
        AscendC::LocalTensor<float> k = worker.Key();
        AscendC::LocalTensor<float> v = worker.Value();
        worker.LoadState(state_gm, head, half);
        worker.LoadHalfRow(q, q_gm, key_head * kKeyDim, kKeyDim);
        worker.NormalizeRow(q);
        worker.LoadHalfRow(k, k_gm, key_head * kKeyDim, kKeyDim);
        worker.NormalizeRow(k);
        worker.LoadHalfRow(v, v_gm, head * kValueDim + half * width, width);
        Gates<kValueSplit> gates;
        gates.Read(worker, g_gm, beta_gm, head);
        worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
        worker.StoreHalfRow(out_gm, v, head * kValueDim + half * width, width);
        worker.StoreState(state_gm, head, half);
    }
}

extern "C" __global__ __aicore__ void qwen_gated_delta_step_kernel(
    GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR out, uint32_t heads, uint32_t key_heads, float q_scale,
    uint32_t value_split) {
    if (clamp_split(value_split) >= kMaxValueSplit) {
        step_body<kMaxValueSplit>(state, q, k, v, g, beta, out, heads, key_heads,
                                  q_scale);
    } else {
        step_body<1>(state, q, k, v, g, beta, out, heads, key_heads, q_scale);
    }
}

// Q/K L2 normalization on its own, fp16 in and fp32 out.
//
// Work is distributed over (token, key_head) pairs rather than tokens, so a decode
// step with rows == 1 still spreads across cores.
extern "C" __global__ __aicore__ void qwen_normalize_gated_delta_qk_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR q_normalized, GM_ADDR k_normalized,
    uint32_t rows, uint32_t key_heads) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> half_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> row_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    pipe.InitBuffer(half_buf, kKeyDim * sizeof(half));
    pipe.InitBuffer(row_buf, kKeyDim * sizeof(float));
    pipe.InitBuffer(work_buf, kKeyDim * sizeof(float));

    const uint32_t total = rows * key_heads * kKeyDim;
    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<float> q_out;
    AscendC::GlobalTensor<float> k_out;
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), total);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), total);
    q_out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(q_normalized), total);
    k_out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k_normalized), total);

    AscendC::LocalTensor<half> staging = half_buf.Get<half>();
    AscendC::LocalTensor<float> row = row_buf.Get<float>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();

    const uint32_t pairs = rows * key_heads;
    for (uint32_t pair = AscendC::GetBlockIdx(); pair < pairs;
         pair += AscendC::GetBlockNum()) {
        const uint32_t offset = pair * kKeyDim;
        // Q then K through the same buffers: the loads are independent but the
        // buffers are not, so each pass ends with its store drained.
        for (uint32_t which = 0; which < 2; ++which) {
            // The previous Cast used this staging buffer as its Vector source.
            // Fence that read before the next MTE2 transfer, including between
            // the Q and K passes for one pair.
            wait_compute_before_load();
            if (which == 0) {
                AscendC::DataCopy(staging, q_gm[offset], kKeyDim);
            } else {
                AscendC::DataCopy(staging, k_gm[offset], kKeyDim);
            }
            wait_load_before_compute();
            AscendC::Cast(row, staging, AscendC::RoundMode::CAST_NONE, kKeyDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(work, row, row, kKeyDim);
            const float sum = fold_sum(work, kKeyDim);
            const float inv = 1.0f / sqrt(sum + 1.0e-6f);
            wait_scalar_before_compute();
            AscendC::Muls(row, row, inv, kKeyDim);
            wait_compute_before_store();
            if (which == 0) {
                AscendC::DataCopy(q_out[offset], row, kKeyDim);
            } else {
                AscendC::DataCopy(k_out[offset], row, kKeyDim);
            }
            wait_store_before_load();
        }
    }
}
