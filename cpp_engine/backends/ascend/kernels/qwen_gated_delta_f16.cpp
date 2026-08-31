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
//   - One head per core iteration, not one value column per core. A head's state is
//     128*128 FP32 = 64 KiB, which fits UB (256 KiB) alongside the row buffers, so
//     the whole recurrence runs out of UB with two GM touches per head: load the
//     state once, store it once. Splitting a head across cores would mean either
//     re-reading the state per token or a cross-core reduction for kv_mem, and this
//     part has no separate vector cores to win that back.
//   - `k^T S` and `q^T S` are reductions down the key axis, i.e. across rows of the
//     state tile. Broadcasting the key scalar over a value-wide row and folding rows
//     pairwise keeps every instruction a full-width vadd/vmul. The alternative,
//     one dot product per value column, is 128 narrow reductions.
//   - The rank-1 update is `Axpy` per key row: S[i,:] += delta * k[i], which is
//     exactly dst = src*scalar + dst.
//
// Numerical order differs from the CUDA kernel: CUDA accumulates kv_mem as a scalar
// sequential sum down the key axis, this folds pairwise in a tree. Both are FP32,
// neither is "the" reference, so the tests compare against a double-precision host
// reference with a tolerance that covers both.
//
// The state is fp32 and stays fp32 the whole way through; only q/k/v/out are fp16.

#include "qwen_ascend_kernel_common.hpp"

namespace {

using namespace pocket;

// Qwen3.5's linear-attention head geometry. Both are 128, and the host rejects
// anything else, so they are compile-time constants: the state tile size, the
// number of Axpy rows and the fold depth all depend on them.
constexpr uint32_t kKeyDim = 128;
constexpr uint32_t kValueDim = 128;
constexpr uint32_t kStateElems = kKeyDim * kValueDim;

// Rows are folded pairwise, so the fold works on a power-of-two row count. 128 is
// already one.
constexpr uint32_t kFoldRows = kKeyDim;

// One head's working set in UB:
//   state      64 KiB   [kKeyDim, kValueDim] fp32
//   scratch    64 KiB   broadcast/fold workspace, same shape
//   rows        3 KiB   q, k (fp32, kKeyDim) and v, out, kv_mem, delta (fp32, kValueDim)
//   halves      1 KiB   fp16 staging for v and out
// which leaves better than half of UB free.
class GatedDeltaHead {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe) {
        pipe.InitBuffer(state_buf_, kStateElems * sizeof(float));
        pipe.InitBuffer(scratch_buf_, kStateElems * sizeof(float));
        pipe.InitBuffer(q_buf_, kKeyDim * sizeof(float));
        pipe.InitBuffer(k_buf_, kKeyDim * sizeof(float));
        pipe.InitBuffer(v_buf_, kValueDim * sizeof(float));
        pipe.InitBuffer(acc_buf_, kValueDim * sizeof(float));
        pipe.InitBuffer(half_buf_, kValueDim * sizeof(half));
        pipe.InitBuffer(aux_buf_, kAlignFloat * sizeof(float));
    }

    // Load S for this head. State layout is [heads, key_dim, value_dim], so a head's
    // slice is contiguous.
    __aicore__ inline void LoadState(const AscendC::GlobalTensor<float>& state,
                                     uint32_t head) {
        AscendC::LocalTensor<float> st = state_buf_.Get<float>();
        AscendC::DataCopy(st, state[head * kStateElems], kStateElems);
        wait_load_before_compute();
    }

    __aicore__ inline void StoreState(const AscendC::GlobalTensor<float>& state,
                                      uint32_t head) {
        AscendC::LocalTensor<float> st = state_buf_.Get<float>();
        wait_compute_before_store();
        AscendC::DataCopy(state[head * kStateElems], st, kStateElems);
    }

    // Pull an fp32 key or query row straight out of a normalized GM buffer.
    __aicore__ inline void LoadKeyRow(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::GlobalTensor<float>& src,
                                      uint32_t offset) {
        AscendC::DataCopy(dst, src[offset], kKeyDim);
        wait_load_before_compute();
    }

    // Pull an fp16 row and widen it. Used for v, and for q/k on the unnormalized
    // entry point.
    __aicore__ inline void LoadHalfRow(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::GlobalTensor<half>& src,
                                       uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<half> staging = half_buf_.Get<half>();
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
        AscendC::Muls(st, st, decay, kStateElems);
        AscendC::PipeBarrier<PIPE_V>();

        // kv_mem = k^T S, as a row-wise fold of S scaled by broadcast k.
        broadcast_rows(work, k, kKeyDim, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work, work, st, kStateElems);
        AscendC::PipeBarrier<PIPE_V>();
        fold_rows(work, kFoldRows, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();

        // delta = (v - kv_mem) * beta, held in acc.
        AscendC::Muls(acc, work, -1.0f, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc, acc, v, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(acc, acc, beta, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();

        // S += k delta^T, one Axpy per key row. Distinct dst rows, so the only
        // barrier needed is the one before the next read of S.
        for (uint32_t i = 0; i < kKeyDim; ++i) {
            AscendC::Axpy(st[i * kValueDim], acc, k.GetValue(i), kValueDim);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // out = q^T S * q_scale, same fold as kv_mem.
        broadcast_rows(work, q, kKeyDim, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work, work, st, kStateElems);
        AscendC::PipeBarrier<PIPE_V>();
        fold_rows(work, kFoldRows, kValueDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(v, work, q_scale, kValueDim);
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
struct Gates {
    __aicore__ inline void Read(GatedDeltaHead& head,
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

}  // namespace

// Whole-sequence recurrence over pre-normalized fp32 q/k.
//
// One head per core iteration, round-robin over the 30 cores. Heads are independent,
// so there is no cross-core communication at all; tokens inside a head are strictly
// sequential, which is what forces the loop to live on one core.
extern "C" __global__ __aicore__ void qwen_gated_delta_sequence_normalized_kernel(
    GM_ADDR state, GM_ADDR q_normalized, GM_ADDR k_normalized, GM_ADDR v,
    GM_ADDR g, GM_ADDR beta, GM_ADDR out, uint32_t rows, uint32_t heads,
    uint32_t key_heads, float q_scale) {
    AscendC::TPipe pipe;
    GatedDeltaHead worker;
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

    for (uint32_t head = AscendC::GetBlockIdx(); head < heads;
         head += AscendC::GetBlockNum()) {
        const uint32_t key_head = head / repeat;
        worker.LoadState(state_gm, head);
        for (uint32_t token = 0; token < rows; ++token) {
            const uint32_t key_offset = token * key_stride + key_head * kKeyDim;
            const uint32_t value_offset = (token * heads + head) * kValueDim;
            AscendC::LocalTensor<float> q = worker.Query();
            AscendC::LocalTensor<float> k = worker.Key();
            AscendC::LocalTensor<float> v = worker.Value();
            worker.LoadKeyRow(q, q_gm, key_offset);
            worker.LoadKeyRow(k, k_gm, key_offset);
            worker.LoadHalfRow(v, v_gm, value_offset, kValueDim);
            Gates gates;
            gates.Read(worker, g_gm, beta_gm, token * heads + head);
            worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
            worker.StoreHalfRow(out_gm, v, value_offset, kValueDim);
        }
        worker.StoreState(state_gm, head);
    }
}

// Whole-sequence recurrence over raw fp16 q/k, normalizing each row on the fly.
//
// Same kernel as above with the normalization folded in. It exists because the
// engine's default path (and every rows == 1 decode step) takes this entry point
// without a separate normalization pass.
extern "C" __global__ __aicore__ void qwen_gated_delta_sequence_kernel(
    GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR out, uint32_t rows, uint32_t heads, uint32_t key_heads,
    float q_scale) {
    AscendC::TPipe pipe;
    GatedDeltaHead worker;
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

    for (uint32_t head = AscendC::GetBlockIdx(); head < heads;
         head += AscendC::GetBlockNum()) {
        const uint32_t key_head = head / repeat;
        worker.LoadState(state_gm, head);
        for (uint32_t token = 0; token < rows; ++token) {
            const uint32_t key_offset = token * key_stride + key_head * kKeyDim;
            const uint32_t value_offset = (token * heads + head) * kValueDim;
            AscendC::LocalTensor<float> q = worker.Query();
            AscendC::LocalTensor<float> k = worker.Key();
            AscendC::LocalTensor<float> v = worker.Value();
            worker.LoadHalfRow(q, q_gm, key_offset, kKeyDim);
            worker.NormalizeRow(q);
            worker.LoadHalfRow(k, k_gm, key_offset, kKeyDim);
            worker.NormalizeRow(k);
            worker.LoadHalfRow(v, v_gm, value_offset, kValueDim);
            Gates gates;
            gates.Read(worker, g_gm, beta_gm, token * heads + head);
            worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
            worker.StoreHalfRow(out_gm, v, value_offset, kValueDim);
        }
        worker.StoreState(state_gm, head);
    }
}

// Single-token step. The gate, q, k and v buffers hold exactly one row, so the
// indexing loses its token term; the recurrence itself is identical.
extern "C" __global__ __aicore__ void qwen_gated_delta_step_kernel(
    GM_ADDR state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR out, uint32_t heads, uint32_t key_heads, float q_scale) {
    AscendC::TPipe pipe;
    GatedDeltaHead worker;
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

    for (uint32_t head = AscendC::GetBlockIdx(); head < heads;
         head += AscendC::GetBlockNum()) {
        const uint32_t key_head = head / repeat;
        AscendC::LocalTensor<float> q = worker.Query();
        AscendC::LocalTensor<float> k = worker.Key();
        AscendC::LocalTensor<float> v = worker.Value();
        worker.LoadState(state_gm, head);
        worker.LoadHalfRow(q, q_gm, key_head * kKeyDim, kKeyDim);
        worker.NormalizeRow(q);
        worker.LoadHalfRow(k, k_gm, key_head * kKeyDim, kKeyDim);
        worker.NormalizeRow(k);
        worker.LoadHalfRow(v, v_gm, head * kValueDim, kValueDim);
        Gates gates;
        gates.Read(worker, g_gm, beta_gm, head);
        worker.Step(q, k, v, gates.decay, gates.beta_value, q_scale);
        worker.StoreHalfRow(out_gm, v, head * kValueDim, kValueDim);
        worker.StoreState(state_gm, head);
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
