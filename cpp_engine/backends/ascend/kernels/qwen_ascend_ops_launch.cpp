// Host launchers for the hand-written AscendC kernels.
//
// These are the twelve operators with no aclnn equivalent on this install: the
// gated-delta recurrence and its gates/normalization, the causal depthwise
// convolution, partial RoPE, the KV-cache append and the three GQA attention
// forms. The device code lives beside this file; everything here is argument
// validation, block-count selection and the launch itself.
//
// ascendc_library() generates one host stub per `extern "C" __global__ __aicore__`
// entry point:
//
//     extern "C" uint32_t aclrtlaunch_<kernel>(uint32_t numBlocks,
//                                              aclrtStream stream, <args...>);
//
// It returns 0 on success and allocates/frees its own 8-byte overflow-status
// buffer per launch. The generated headers land in
// ${CMAKE_BINARY_DIR}/include/pocket_ascend_kernels, which is on this target's
// include path.
//
// Two things are worth knowing before changing anything here:
//
//   1. Validation is not defensive decoration. The kernels index GM with
//      unsigned arithmetic and no bounds checks, so a bad `max_context` or a
//      `heads % key_heads != 0` is an out-of-bounds device write, not a wrong
//      number. Every contract the device code assumes is checked here, and the
//      checks mirror the CUDA launchers so a caller cannot tell the backends
//      apart by which arguments they reject.
//   2. There is no device sin/cos reachable from a classic __aicore__ kernel, so
//      partial RoPE takes precomputed FP32 tables. The first correctness path builds
//      the requested tables per call on the host and uploads them synchronously; it
//      deliberately makes no retained per-device or per-geometry cache claim.

#include "aclnn_common.hpp"

#include "qwen_ascend_ops.hpp"
#include "qwen_gated_delta_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

#include "aclrtlaunch_qwen_append_kv_cache_kernel.h"
#include "aclrtlaunch_qwen_argmax_f32_rows_kernel.h"
#include "aclrtlaunch_qwen_causal_depthwise_conv_silu_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_sequence_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_sequence_normalized_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_step_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_vector_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_flashdec_partial_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_flashdec_reduce_kernel.h"
// #include "aclrtlaunch_qwen_hbm_read_probe_kernel.h"  // Kernel not built yet
#include "aclrtlaunch_qwen_gqa_prefill_attention_kernel.h"
#include "aclrtlaunch_qwen_gqa_prefill_attention_vector_kernel.h"
#include "aclrtlaunch_qwen_gqa_verify_attention_kernel.h"
#include "aclrtlaunch_qwen_gqa_verify_attention_vector_kernel.h"
#include "aclrtlaunch_qwen_linear_attn_gates_kernel.h"
#include "aclrtlaunch_qwen_normalize_gated_delta_qk_kernel.h"
#include "aclrtlaunch_qwen_partial_rope_rows_kernel.h"
#include "aclrtlaunch_qwen_cube_gemm_probe_kernel.h"
#include "aclrtlaunch_qwen_cube_transpose_probe_kernel.h"
#include "aclrtlaunch_qwen_transpose_f16_kernel.h"
#include "aclrtlaunch_qwen_gqa_attention_cube_kernel.h"
#include "aclrtlaunch_qwen_gqa_attention_cube_partial_kernel.h"
#include "aclrtlaunch_qwen_gqa_attention_cube_reduce_kernel.h"

namespace pocket {
namespace {

using ascend::resolve;

// The generated stubs return 0 for success.
constexpr uint32_t kLaunchOk = 0;

// The recurrence kernels hard-code a [128, 128] state tile, so the host has to
// reject any other head geometry rather than launch a kernel that would read the
// wrong stride. The CUDA launchers reject the same values.
constexpr int kRecurrentKeyDim = pocket::gated_delta::kKeyDim;
constexpr int kRecurrentValueDim = pocket::gated_delta::kValueDim;

constexpr int kMaxAttentionHeadDim = 256;
constexpr uint32_t kAttentionAlignmentBytes = 32;
constexpr uint32_t kAttentionAlignmentHalfs = 16;

// Convolution tail capacity, matching kMaxKernel in the device code and kMaxTail
// on the CUDA side.
constexpr int kMaxConvKernel = 8;

// First-generation 910 has 30 AI cores. Every kernel here is a grid-stride loop
// over independent work items, so more blocks than cores would only add launch
// overhead; fewer than the work available would leave cores idle.
//
// Defined in the gated-delta geometry header because the value-axis split there is
// the one place a kernel's item count and the launcher's grid size are two
// derivations of the same number, and a mismatch between them would be silent.
constexpr uint32_t kMaxBlocks = pocket::gated_delta::kMaxBlocks;

// Query the AI core count once per device and fall back to the known
// first-generation value. Reading it rather than hard-coding it means a run on
// second-generation silicon (20 or 24 cores) still launches a legal grid, even
// though the kernels are not tuned for it.
uint32_t core_count() {
    static std::mutex mutex;
    static std::map<int32_t, uint32_t> cache;
    int32_t device = 0;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) return kMaxBlocks;
    std::lock_guard<std::mutex> guard(mutex);
    auto found = cache.find(device);
    if (found != cache.end()) return found->second;
    int64_t cores = 0;
    uint32_t resolved = kMaxBlocks;
    if (aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                           ACL_DEV_ATTR_AICORE_CORE_NUM,
                           &cores) == ACL_SUCCESS &&
        cores > 0) {
        // Clamped rather than trusted. A reported count above 30 means this is
        // not the SoC these kernels were written for, and launching that grid
        // would size the loop for hardware whose UB budget has not been checked.
        resolved = static_cast<uint32_t>(cores);
        if (resolved > kMaxBlocks) resolved = kMaxBlocks;
    }
    cache.emplace(device, resolved);
    return resolved;
}

// Blocks for `work` independent items: never more than there is work for, never
// more than there are cores, never zero.
uint32_t blocks_for(uint64_t work) {
    const uint32_t cores = core_count();
    if (work == 0) return 1;
    if (work < static_cast<uint64_t>(cores)) return static_cast<uint32_t>(work);
    return cores;
}

// The unit of work in the recurrence kernels is one head times one value range, not
// one head: the state is separable down the value axis, so a split multiplies the
// number of independent items. The rule that picks the split, and the argument for
// it, live in the geometry header next to the constant the kernel derives its own
// item count from.
uint64_t recurrent_units(uint64_t heads, uint32_t split) {
    return heads * static_cast<uint64_t>(split);
}

// Shared attention preconditions, mirroring valid_attention() in the CUDA
// launcher so both backends reject the same shapes.
bool valid_attention(int q_heads, int kv_heads, int head_dim, int context_len,
                     int max_context) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
           head_dim > 0 && head_dim <= kMaxAttentionHeadDim &&
           context_len > 0 && context_len <= max_context;
}

// The first vector path is intentionally limited to contiguous KV rows. The real
// Qwen TP4 full-attention shape is kv_heads=1; other valid neutral shapes stay on
// the scalar kernel until a strided UB loader is separately validated. The fast
// path is enabled by default after hardware validation; set the switch to 0 for
// an apples-to-apples scalar baseline.
bool vector_gqa_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("QWEN_ASCEND_GQA_VECTOR");
        return value == nullptr || value[0] != '0' || value[1] != '\0';
    }();
    return enabled;
}

bool vector_attention_geometry(const void* q, const void* k, const void* v,
                               const void* out, int q_heads, int kv_heads,
                               int head_dim) {
    const auto aligned = [](const void* pointer) {
        return pointer != nullptr &&
               (reinterpret_cast<uintptr_t>(pointer) & (kAttentionAlignmentBytes - 1)) == 0;
    };
    return vector_gqa_enabled() && q_heads > 0 && kv_heads == 1 &&
           head_dim > 0 && head_dim <= kMaxAttentionHeadDim &&
           head_dim % static_cast<int>(kAttentionAlignmentHalfs) == 0 && aligned(q) &&
           aligned(k) && aligned(v) && aligned(out);
}

// Return the smallest group count that makes `group * elements` an integral
// number of alignment units. This keeps adjacent block-owned records on distinct
// 32-byte GM cache lines.
uint32_t aligned_work_group(uint32_t elements, uint32_t alignment) {
    for (uint32_t group = 1; group <= alignment; ++group) {
        if ((group * elements) % alignment == 0) return group;
    }
    return alignment;
}

// The softmax scale. Passed to the kernel as a float because aicore cannot cast
// an unsigned integer to floating point, and its integer sqrt overload truncates
// (head_dim 128 would give 11 instead of 11.3137).
float attention_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// The kernels take GM_ADDR, which the generated stubs expose as void*. Const is
// dropped at the boundary because the launch ABI has no const form; the device
// code only reads these.
template <typename T>
void* gm(const T* pointer) {
    return const_cast<void*>(static_cast<const void*>(pointer));
}

// ---- Cube (Mmad) attention sizing ----

// Context columns per Cube chunk, matching kChunk in qwen_attention_cube_f16.cpp.
// The kernel refuses a score pitch that is not a multiple of it.
constexpr uint32_t kCubeChunk = 512;

// Largest stacked row tile (positions * head group) the kernel accepts. One work
// item covers a whole query head group stacked as rows, and that tile is what has
// to fit L0C at four bytes per element.
constexpr uint32_t kCubeMaxStackedRows = 96;

// Beyond this the kernel's vector row buffers, which are sized from the score
// pitch rounded up to a power of two, stop fitting its UB budget.
constexpr uint32_t kCubeMaxScoreStride = 8192;

// Cube attention is on by default; `QWEN_ASCEND_GQA_CUBE=0` selects the vector
// kernel instead, which is how the numeric test obtains a reference on the same
// binary. Read per call rather than cached for that reason -- one getenv beside a
// kernel launch is not measurable.
bool cube_gqa_enabled() {
    const char* value = std::getenv("QWEN_ASCEND_GQA_CUBE");
    return value == nullptr || value[0] != '0' || value[1] != '\0';
}

// The Cube path's preconditions that depend only on the shape. Split out from
// cube_attention_geometry because the engine-side availability query has no
// pointers to check, only a shape.
bool cube_attention_shape_ok(int q_heads, int kv_heads, int head_dim) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 && head_dim > 0 &&
           head_dim <= kMaxAttentionHeadDim &&
           head_dim % static_cast<int>(kAttentionAlignmentHalfs) == 0;
}

// The Cube path's preconditions that do not depend on the context length. The Cube
// contracts along 16-element fractals, so head_dim has to be a multiple of 16, and
// because a work item covers a whole query head group the group has to divide
// evenly.
bool cube_attention_geometry(const void* q, const void* k, const void* v,
                             const void* out, int q_heads, int kv_heads,
                             int head_dim) {
    const auto aligned = [](const void* pointer) {
        return pointer != nullptr &&
               (reinterpret_cast<uintptr_t>(pointer) & (kAttentionAlignmentBytes - 1)) == 0;
    };
    return cube_gqa_enabled() && cube_attention_shape_ok(q_heads, kv_heads, head_dim) &&
           aligned(q) && aligned(k) && aligned(v) && aligned(out);
}

// Prefill attention on the Cube. Returns false for any shape it cannot express,
// which is the caller's signal to fall back to the vector kernel.
//
// Sizing, and why each number is what it is:
//
//   score_stride   the attention extent rounded up to a whole chunk. It is the row
//                  pitch of both the score scratch and the transposed V image,
//                  because the kernel's chunk grid has no partial case.
//   rows_per_tile  positions per work item. One item covers the whole head group
//                  (repeat heads) stacked as rows -- K and V do not depend on which
//                  head asks for them, so an item per head would re-read both
//                  caches `repeat` times more than the arithmetic needs. The stacked
//                  tile is therefore what has to fit L0C, and L0C is the only thing
//                  bounding this. Kept a power of two so the scratch row count is
//                  stable across context lengths.
//   score_rows     the scratch row count per core: the rounded stacked tile.
//
// The score scratch and the transposed V are one allocation with the V image
// appended, not two. WorkspacePool hands out one buffer per (device, stream,
// purpose), so a second Intermediate request would return the first buffer
// whenever it happened to be large enough -- and here it always is.
bool cube_prefill_attention(const uint16_t* d_q_rows_fp16,
                            const uint16_t* d_k_cache_fp16,
                            const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
                            int seq_len, int q_heads, int kv_heads, int head_dim,
                            int position_offset, int max_context, void* stream) {
    const uint32_t limit = static_cast<uint32_t>(position_offset + seq_len);
    const uint32_t repeat = static_cast<uint32_t>(q_heads / kv_heads);
    const uint32_t score_stride = (limit + kCubeChunk - 1) / kCubeChunk * kCubeChunk;
    const uint32_t head = static_cast<uint32_t>(head_dim);
    if (score_stride == 0 || score_stride > kCubeMaxScoreStride) return false;

    uint32_t rows_per_tile = 1;
    while (rows_per_tile * 2 * repeat <= kCubeMaxStackedRows) rows_per_tile *= 2;
    const uint32_t stacked = rows_per_tile * repeat;
    const uint32_t score_rows =
        (stacked + kAttentionAlignmentHalfs - 1) / kAttentionAlignmentHalfs *
        kAttentionAlignmentHalfs;

    const uint32_t row_blocks =
        (static_cast<uint32_t>(seq_len) + rows_per_tile - 1) / rows_per_tile;
    const uint32_t items = row_blocks * static_cast<uint32_t>(kv_heads);
    const uint32_t blocks = items < core_count() ? items : core_count();
    if (blocks == 0) return false;

    const uint64_t score_bytes =
        static_cast<uint64_t>(blocks) * score_rows * score_stride * sizeof(uint16_t);
    const uint64_t value_bytes =
        static_cast<uint64_t>(kv_heads) * head * score_stride * sizeof(uint16_t);
    bool ok = false;
    uint8_t* pool = static_cast<uint8_t*>(ascend::WorkspacePool::acquire(
        score_bytes + value_bytes, ascend::resolve(stream), ok,
        ascend::WorkspacePool::Purpose::Intermediate));
    if (!ok || pool == nullptr) return false;
    uint16_t* scratch = reinterpret_cast<uint16_t*>(pool);
    uint16_t* vt = reinterpret_cast<uint16_t*>(pool + score_bytes);

    // One transpose per KV head: the cache interleaves the heads, so each call
    // walks a head stride rather than a contiguous block. `score_stride` columns
    // are produced for an extent of `limit`, and the columns past `limit` come out
    // zero so the kernel can read whole chunks without a bounds check.
    for (int k_head = 0; k_head < kv_heads; ++k_head) {
        if (!qwen_transpose_f16_padded_ascend(
                d_v_cache_fp16 + static_cast<size_t>(k_head) * head_dim,
                vt + static_cast<size_t>(k_head) * head * score_stride,
                static_cast<int>(score_stride), head_dim, static_cast<int>(limit),
                kv_heads * head_dim, static_cast<int>(score_stride), stream)) {
            return false;
        }
    }

    return aclrtlaunch_qwen_gqa_attention_cube_kernel(
               blocks, ascend::resolve(stream), gm(d_q_rows_fp16),
               gm(d_k_cache_fp16), gm(vt), gm(d_out_rows_fp16), gm(scratch),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(q_heads),
               static_cast<uint32_t>(kv_heads), head,
               static_cast<uint32_t>(position_offset),
               static_cast<uint32_t>(max_context), rows_per_tile, 1u, score_stride,
               score_rows, attention_scale(head_dim)) == kLaunchOk;
}

// Decode attention on the Cube: one query position, the whole head group stacked
// as rows, every row sharing the same causal limit.
//
// This is the same kernel with `causal = 0`, and the reason to route decode here
// is starker than for prefill. The vector decode kernel hands each query head to
// its own block, so the real TP4 shape -- six heads over one KV head -- runs on
// six of the thirty AI cores, and each of those walks the whole context alone.
// The Cube entry stacks those six rows into one Mmad and reads K and V once.
//
// One work item, one core: the kernel indexes decode items by KV head, and this
// model has a single KV head per rank. That is a real ceiling on this shape and
// not an oversight -- what it buys is that the K and V caches are each read
// exactly once from GM, which is what the operator is bound by at long context.
// Splitting the context across cores would need a partial-softmax reduction to
// put the pieces back together.
bool cube_decode_attention(const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
                           const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
                           int q_heads, int kv_heads, int head_dim, int context_len,
                           int max_context, void* stream) {
    // The kernel indexes decode work items by KV head and exits without writing
    // anything when a rank holds more than one, so this is a guard and not a
    // tuning choice: without it a TP1 or TP2 run would return silence.
    if (kv_heads != 1) return false;
    if (context_len <= 0 || context_len > max_context) return false;
    const uint32_t limit = static_cast<uint32_t>(context_len);
    const uint32_t repeat = static_cast<uint32_t>(q_heads / kv_heads);
    const uint32_t score_stride = (limit + kCubeChunk - 1) / kCubeChunk * kCubeChunk;
    const uint32_t head = static_cast<uint32_t>(head_dim);
    if (score_stride == 0 || score_stride > kCubeMaxScoreStride) return false;

    // One position, so the stacked tile is the head group itself and the scratch
    // only has to be tall enough for the fractal-rounded version of it.
    const uint32_t score_rows =
        (repeat + kAttentionAlignmentHalfs - 1) / kAttentionAlignmentHalfs *
        kAttentionAlignmentHalfs;

    const uint64_t score_bytes =
        static_cast<uint64_t>(score_rows) * score_stride * sizeof(uint16_t);
    const uint64_t value_bytes =
        static_cast<uint64_t>(kv_heads) * head * score_stride * sizeof(uint16_t);
    bool ok = false;
    uint8_t* pool = static_cast<uint8_t*>(ascend::WorkspacePool::acquire(
        score_bytes + value_bytes, ascend::resolve(stream), ok,
        ascend::WorkspacePool::Purpose::Intermediate));
    if (!ok || pool == nullptr) return false;
    uint16_t* scratch = reinterpret_cast<uint16_t*>(pool);
    uint16_t* vt = reinterpret_cast<uint16_t*>(pool + score_bytes);

    for (int k_head = 0; k_head < kv_heads; ++k_head) {
        if (!qwen_transpose_f16_padded_ascend(
                d_v_cache_fp16 + static_cast<size_t>(k_head) * head_dim,
                vt + static_cast<size_t>(k_head) * head * score_stride,
                static_cast<int>(score_stride), head_dim, static_cast<int>(limit),
                kv_heads * head_dim, static_cast<int>(score_stride), stream)) {
            return false;
        }
    }

    return aclrtlaunch_qwen_gqa_attention_cube_kernel(
               1u, ascend::resolve(stream), gm(d_q_fp16), gm(d_k_cache_fp16),
               gm(vt), gm(d_out_fp16), gm(scratch), 1u,
               static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads), head,
               limit - 1, static_cast<uint32_t>(max_context), 1u, 0u, score_stride,
               score_rows, attention_scale(head_dim)) == kLaunchOk;
}

// Partition count the reduce kernel's weight vector is sized for, matching
// kMaxCubePartitions in the device code.
constexpr uint32_t kCubeMaxPartitions = 64;

// Decode attention on the Cube with the context split across cores.
//
// The single-core entry above is correct and reads K and V exactly once, but at
// the real TP4 shape it has one work item, so one AI core walks the whole context
// while thirty sit idle. The measured cost is linear in the context -- 585 us at
// 2048, 1220 us at 4097, 2098 us at 8192 -- which is what a serial pipeline with
// no overlap against anything looks like, not what an arithmetic or a bandwidth
// bound looks like.
//
// This splits the chunk axis: every block takes a contiguous run of 512-column
// chunks, computes an online softmax over just those, and writes an unnormalized
// partial. The reduce kernel then merges with exp(m_p - M) weights. Total K and V
// traffic is unchanged -- each chunk is still read once -- so the only thing that
// scales is the number of cores doing it.
//
// Chunks are the granularity, so the useful partition count is bounded by the
// number of chunks in the context; past that a partition would own no columns and
// the kernel returns without writing.
bool cube_decode_attention_partitioned(const uint16_t* d_q_fp16,
                                       const uint16_t* d_k_cache_fp16,
                                       const uint16_t* d_v_cache_fp16,
                                       uint16_t* d_out_fp16, int q_heads,
                                       int kv_heads, int head_dim, int context_len,
                                       int max_context, int partitions,
                                       void* stream) {
    if (kv_heads != 1) return false;
    if (context_len <= 0 || context_len > max_context) return false;
    const uint32_t limit = static_cast<uint32_t>(context_len);
    const uint32_t repeat = static_cast<uint32_t>(q_heads / kv_heads);
    const uint32_t head = static_cast<uint32_t>(head_dim);
    const uint32_t chunks_total = (limit + kCubeChunk - 1) / kCubeChunk;
    // One chunk per partition is the finest split the kernel can express, and the
    // reduce kernel's weight vector caps the other end.
    uint32_t parts = chunks_total;
    const uint32_t cores = core_count();
    if (parts > cores) parts = cores;
    if (parts > kCubeMaxPartitions) parts = kCubeMaxPartitions;
    if (partitions > 0) {
        const uint32_t requested = static_cast<uint32_t>(partitions);
        parts = requested < parts ? requested : parts;
    }
    if (parts < 2 || parts > chunks_total) return false;

    const uint32_t chunks_per_part = (chunks_total + parts - 1) / parts;
    const uint32_t part_stride = chunks_per_part * kCubeChunk;
    const uint32_t score_stride = chunks_total * kCubeChunk;
    if (score_stride > kCubeMaxScoreStride) return false;

    const uint32_t score_rows =
        (repeat + kAttentionAlignmentHalfs - 1) / kAttentionAlignmentHalfs *
        kAttentionAlignmentHalfs;

    // score scratch per partition, then the transposed V image, then the two
    // partial arrays. The V image holds `score_stride` columns, which is wider than
    // any single partition's slice; the kernel indexes it by absolute column, so
    // every block shares the one image.
    const uint64_t score_bytes = static_cast<uint64_t>(parts) * kv_heads *
                                 score_rows * part_stride * sizeof(uint16_t);
    const uint64_t value_bytes =
        static_cast<uint64_t>(kv_heads) * head * score_stride * sizeof(uint16_t);
    const uint64_t out_bytes =
        static_cast<uint64_t>(parts) * q_heads * head * sizeof(uint16_t);
    const uint64_t stats_bytes =
        static_cast<uint64_t>(parts) * q_heads * 2 * sizeof(float);
    bool ok = false;
    uint8_t* pool = static_cast<uint8_t*>(ascend::WorkspacePool::acquire(
        score_bytes + value_bytes + out_bytes + stats_bytes, ascend::resolve(stream),
        ok, ascend::WorkspacePool::Purpose::Intermediate));
    if (!ok || pool == nullptr) return false;
    uint16_t* scratch = reinterpret_cast<uint16_t*>(pool);
    uint16_t* vt = reinterpret_cast<uint16_t*>(pool + score_bytes);
    uint16_t* partials = reinterpret_cast<uint16_t*>(pool + score_bytes + value_bytes);
    float* stats = reinterpret_cast<float*>(pool + score_bytes + value_bytes +
                                            out_bytes);

    for (int k_head = 0; k_head < kv_heads; ++k_head) {
        if (!qwen_transpose_f16_padded_ascend(
                d_v_cache_fp16 + static_cast<size_t>(k_head) * head_dim,
                vt + static_cast<size_t>(k_head) * head * score_stride,
                static_cast<int>(score_stride), head_dim, static_cast<int>(limit),
                kv_heads * head_dim, static_cast<int>(score_stride), stream)) {
            return false;
        }
    }

    const uint32_t blocks = parts * static_cast<uint32_t>(kv_heads);
    if (aclrtlaunch_qwen_gqa_attention_cube_partial_kernel(
            blocks, ascend::resolve(stream), gm(d_q_fp16), gm(d_k_cache_fp16),
            gm(vt), gm(partials), gm(stats), gm(scratch),
            static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads), head,
            limit, static_cast<uint32_t>(max_context), chunks_per_part, part_stride,
            score_stride, parts, attention_scale(head_dim)) != kLaunchOk) {
        return false;
    }

    // One block per query head; there are six of them at the real shape and the
    // kernel grid-strides, so the grid is capped at the core count rather than at
    // q_heads.
    const uint32_t reduce_blocks =
        static_cast<uint32_t>(q_heads) < cores ? static_cast<uint32_t>(q_heads) : cores;
    return aclrtlaunch_qwen_gqa_attention_cube_reduce_kernel(
               reduce_blocks, ascend::resolve(stream), gm(partials), gm(stats),
               gm(d_out_fp16), static_cast<uint32_t>(q_heads), head,
               parts) == kLaunchOk;
}

bool valid_recurrent(int heads, int key_heads, int key_dim, int value_dim,
                     float q_scale) {
    return heads > 0 && key_heads > 0 && heads % key_heads == 0 &&
           key_dim == kRecurrentKeyDim && value_dim == kRecurrentValueDim &&
           q_scale > 0.0f;
}

// Per-call FP32 cos/sin tables for partial RoPE.
//
// The table is [rows, rotary_dim / 2] and contains the absolute positions starting
// at `start_position`. Angles are computed in double and rounded once to float;
// the CUDA device implementation's float powf/cosf path is therefore not copied
// into this first-generation kernel, which has no classic-aicore trig primitive.
//
// The device allocation comes from WorkspacePool's Intermediate slot rather than
// aclrtMalloc. The copy is queued on the same stream as the kernel, so the table
// remains live until the launch consumes it and can be reused by the next
// serialized operator without a stream-wide synchronization.
class RopeTables {
public:
    static bool acquire(int rotary_dim, float theta, int start_position, int rows,
                        aclrtStream stream, const float** cos_rows,
                        const float** sin_rows) {
        const size_t half = static_cast<size_t>(rotary_dim / 2);
        const size_t elements = static_cast<size_t>(rows) * half;
        const size_t bytes = elements * sizeof(float);
        std::vector<float> host_cos(elements);
        std::vector<float> host_sin(elements);
        std::vector<double> inv_freq(half);
        for (size_t i = 0; i < half; ++i) {
            inv_freq[i] = std::pow(
                static_cast<double>(theta),
                -2.0 * static_cast<double>(i) /
                    static_cast<double>(rotary_dim));
        }
        for (int row = 0; row < rows; ++row) {
            const double position = static_cast<double>(start_position + row);
            const size_t base = static_cast<size_t>(row) * half;
            for (size_t i = 0; i < half; ++i) {
                const double angle = position * inv_freq[i];
                host_cos[base + i] = static_cast<float>(std::cos(angle));
                host_sin[base + i] = static_cast<float>(std::sin(angle));
            }
        }

        bool pool_ok = true;
        void* storage = ascend::WorkspacePool::acquire(
            bytes * 2, stream, pool_ok,
            ascend::WorkspacePool::Purpose::Intermediate);
        if (!pool_ok || storage == nullptr) return false;
        auto* device = static_cast<float*>(storage);
        // The host vectors are temporary. A synchronous copy keeps their lifetime
        // explicit; enqueueing an async H2D and destroying them at return races the
        // DMA engine and produces position-dependent RoPE corruption.
        if (aclrtMemcpy(device, bytes, host_cos.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(device + elements, bytes, host_sin.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return false;
        }
        *cos_rows = device;
        *sin_rows = device + elements;
        return true;
    }
};

bool valid_rope(int start_position, int rows, int rotary_dim, float theta,
                int q_heads, int kv_heads, int head_dim) {
    return start_position >= 0 && rows > 0 && rotary_dim > 0 &&
           (rotary_dim & 1) == 0 && rotary_dim <= head_dim &&
           std::isfinite(theta) && theta > 0.0f && q_heads > 0 &&
           kv_heads > 0 && head_dim > 0;
}

}  // namespace

// Many-core Cube decode attention. `partitions <= 0` asks for the widest split the
// context allows; the context has to span at least two chunks or there is nothing
// to split and the single-core entry is the right one.
bool qwen_gqa_decode_attention_cube_split_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16, int q_heads,
    int kv_heads, int head_dim, int context_len, int max_context, int partitions,
    void* stream) {
    if (d_q_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_fp16 == nullptr ||
        !valid_attention(q_heads, kv_heads, head_dim, context_len, max_context) ||
        !vector_attention_geometry(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                   d_out_fp16, q_heads, kv_heads, head_dim)) {
        return false;
    }
    return cube_decode_attention_partitioned(
        d_q_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_fp16, q_heads, kv_heads,
        head_dim, context_len, max_context, partitions, stream);
}

bool qwen_normalize_gated_delta_qk_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, float* d_q_normalized,
    float* d_k_normalized, int rows, int key_heads, int key_dim, void* stream) {
    if (d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_q_normalized == nullptr || d_k_normalized == nullptr || rows <= 0 ||
        key_heads <= 0 || key_dim != kRecurrentKeyDim) {
        return false;
    }
    const uint64_t pairs = static_cast<uint64_t>(rows) * key_heads;
    return aclrtlaunch_qwen_normalize_gated_delta_qk_kernel(
               blocks_for(pairs), resolve(stream), gm(d_q_fp16), gm(d_k_fp16),
               gm(d_q_normalized), gm(d_k_normalized),
               static_cast<uint32_t>(rows),
               static_cast<uint32_t>(key_heads)) == kLaunchOk;
}

bool qwen_gated_delta_sequence_f16_ascend(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_v_fp16 == nullptr || d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr || rows <= 0 ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    const uint32_t split = pocket::gated_delta::value_split_for(
        static_cast<uint32_t>(heads), core_count());
    return aclrtlaunch_qwen_gated_delta_sequence_kernel(
               blocks_for(recurrent_units(heads, split)), resolve(stream),
               gm(d_state), gm(d_q_fp16), gm(d_k_fp16), gm(d_v_fp16),
               gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(heads),
               static_cast<uint32_t>(key_heads), q_scale,
               split) == kLaunchOk;
}

bool qwen_gated_delta_sequence_normalized_f16_ascend(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_normalized == nullptr ||
        d_k_normalized == nullptr || d_v_fp16 == nullptr ||
        d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr || rows <= 0 ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    const uint32_t split = pocket::gated_delta::value_split_for(
        static_cast<uint32_t>(heads), core_count());
    return aclrtlaunch_qwen_gated_delta_sequence_normalized_kernel(
               blocks_for(recurrent_units(heads, split)), resolve(stream),
               gm(d_state), gm(d_q_normalized), gm(d_k_normalized),
               gm(d_v_fp16), gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(heads),
               static_cast<uint32_t>(key_heads), q_scale,
               split) == kLaunchOk;
}

// The CUDA `_shared` variant shards one head's state across a block so several
// value columns progress together in shared memory. That is a launch-geometry
// choice, not a different recurrence: it is bit-comparable to the normalized
// kernel by construction. This part has no equivalent to shard onto (a head's
// state already lives entirely in UB and tokens are strictly sequential), so the
// contract is satisfied by the normalized kernel itself rather than by a second
// device implementation.
bool qwen_gated_delta_sequence_normalized_shared_f16_ascend(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    return qwen_gated_delta_sequence_normalized_f16_ascend(
        d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16,
        d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim,
        q_scale, stream);
}

bool qwen_gated_delta_step_f16_ascend(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_v_fp16 == nullptr || d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    const uint32_t split = pocket::gated_delta::value_split_for(
        static_cast<uint32_t>(heads), core_count());
    return aclrtlaunch_qwen_gated_delta_step_kernel(
               blocks_for(recurrent_units(heads, split)), resolve(stream),
               gm(d_state), gm(d_q_fp16), gm(d_k_fp16), gm(d_v_fp16),
               gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(heads), static_cast<uint32_t>(key_heads),
               q_scale, split) == kLaunchOk;
}

bool qwen_linear_attn_gates_f16_ascend(
    const uint16_t* d_a_fp16, const uint16_t* d_b_fp16,
    const uint16_t* d_a_log_fp16, const uint16_t* d_dt_bias_fp16,
    uint16_t* d_g_fp16, uint16_t* d_beta_fp16, int rows, int heads,
    void* stream) {
    if (d_a_fp16 == nullptr || d_b_fp16 == nullptr ||
        d_a_log_fp16 == nullptr || d_dt_bias_fp16 == nullptr ||
        d_g_fp16 == nullptr || d_beta_fp16 == nullptr || rows <= 0 ||
        heads <= 0) {
        return false;
    }
    // The kernel tiles the flat [rows*heads] plane in 512-lane chunks.
    const uint64_t total = static_cast<uint64_t>(rows) * heads;
    const uint64_t tiles = (total + 511) / 512;
    return aclrtlaunch_qwen_linear_attn_gates_kernel(
               blocks_for(tiles), resolve(stream), gm(d_a_fp16), gm(d_b_fp16),
               gm(d_a_log_fp16), gm(d_dt_bias_fp16), gm(d_g_fp16),
               gm(d_beta_fp16), static_cast<uint32_t>(rows),
               static_cast<uint32_t>(heads)) == kLaunchOk;
}

bool qwen_causal_depthwise_conv_silu_f16_ascend(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_tail_fp16, uint16_t* d_y_fp16, int seq_len, int channels,
    int kernel, bool update_tail, void* stream) {
    // d_tail_fp16 is allowed to be null: that signals zero convolution history,
    // and the device code branches on it. Everything else is required.
    if (d_x_fp16 == nullptr || d_weight_fp16 == nullptr || d_y_fp16 == nullptr ||
        seq_len <= 0 || channels <= 0 || kernel <= 0 ||
        kernel > kMaxConvKernel) {
        return false;
    }
    // The kernel walks 512-channel tiles for the whole sequence.
    const uint64_t tiles = (static_cast<uint64_t>(channels) + 511) / 512;
    return aclrtlaunch_qwen_causal_depthwise_conv_silu_kernel(
               blocks_for(tiles), resolve(stream), gm(d_x_fp16),
               gm(d_weight_fp16), gm(d_tail_fp16), gm(d_y_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(channels),
               static_cast<uint32_t>(kernel),
               update_tail ? 1u : 0u) == kLaunchOk;
}

bool qwen_partial_rope_rows_f16_ascend(
    uint16_t* d_q_fp16, uint16_t* d_k_fp16, int start_position, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim,
    void* stream) {
    if (d_q_fp16 == nullptr || d_k_fp16 == nullptr || start_position < 0 ||
        rows <= 0 || rotary_dim <= 0 || (rotary_dim & 1) != 0 ||
        rotary_dim > head_dim || theta <= 0.0f || q_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const float* cos_rows = nullptr;
    const float* sin_rows = nullptr;
    if (!RopeTables::acquire(rotary_dim, theta, start_position, rows, s,
                             &cos_rows, &sin_rows)) {
        return false;
    }
    const uint64_t pairs = static_cast<uint64_t>(rows) * (rotary_dim / 2);
    return aclrtlaunch_qwen_partial_rope_rows_kernel(
               blocks_for(pairs), s, gm(d_q_fp16), gm(d_k_fp16), gm(cos_rows),
               gm(sin_rows), static_cast<uint32_t>(rows),
               static_cast<uint32_t>(rotary_dim),
               static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim)) == kLaunchOk;
}

bool qwen_append_kv_cache_f16_ascend(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len,
    int kv_heads, int head_dim, int start_pos, int max_context, void* stream) {
    if (d_k_rows_fp16 == nullptr || d_v_rows_fp16 == nullptr ||
        d_k_cache_fp16 == nullptr || d_v_cache_fp16 == nullptr ||
        seq_len <= 0 || kv_heads <= 0 || head_dim <= 0 || start_pos < 0 ||
        max_context <= 0 || start_pos + seq_len > max_context) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(seq_len) * kv_heads * head_dim;
    const uint64_t tiles = (elements + 511) / 512;
    // Scalar handling of an unaligned tail is serialized to avoid two cores
    // updating different halfs in the same cache line. Qwen's 128-wide rows take
    // the aligned parallel path.
    const bool aligned = head_dim % 16 == 0;
    return aclrtlaunch_qwen_append_kv_cache_kernel(
               aligned ? blocks_for(tiles) : 1u, resolve(stream), gm(d_k_rows_fp16),
               gm(d_v_rows_fp16), gm(d_k_cache_fp16), gm(d_v_cache_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(start_pos),
               static_cast<uint32_t>(max_context)) == kLaunchOk;
}

// Whether a decode call of this shape will reach the Cube path. The engine asks
// before choosing between splitting the context across cores (FlashDecoding) and
// the neutral decode entry, because the two are not interchangeable: FlashDecoding
// exists to buy parallelism the Cube path does not need, and it pays for that with
// a partial-softmax reduction. Above the Cube path's score pitch limit the split
// is the only option, so the answer has to be a real query rather than a threshold
// copied into the engine, where it would drift from the launcher's constants.
bool qwen_gqa_decode_attention_cube_available_ascend(int q_heads, int kv_heads,
                                                     int head_dim, int context_len,
                                                     int max_context) {
    if (!cube_gqa_enabled()) return false;
    if (!cube_attention_shape_ok(q_heads, kv_heads, head_dim)) return false;
    if (kv_heads != 1) return false;
    if (context_len <= 0 || context_len > max_context) return false;
    const uint32_t score_stride =
        (static_cast<uint32_t>(context_len) + kCubeChunk - 1) / kCubeChunk * kCubeChunk;
    return score_stride != 0 && score_stride <= kCubeMaxScoreStride;
}

bool qwen_gqa_decode_attention_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_score_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, void* stream) {
    if (d_q_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_fp16 == nullptr ||
        d_score_scratch == nullptr ||
        !valid_attention(q_heads, kv_heads, head_dim, context_len,
                         max_context)) {
        return false;
    }

    // Cube first, and for decode it is the larger of the two wins: the vector
    // kernel below can only reach six of the thirty cores at this head count.
    if (cube_attention_geometry(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_fp16,
                                q_heads, kv_heads, head_dim) &&
        cube_decode_attention(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_fp16,
                              q_heads, kv_heads, head_dim, context_len, max_context,
                              stream)) {
        return true;
    }

    if (vector_attention_geometry(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                  d_out_fp16, q_heads, kv_heads, head_dim) &&
        (reinterpret_cast<uintptr_t>(d_score_scratch) & (kAttentionAlignmentBytes - 1)) == 0) {
        const uint32_t works_per_block = aligned_work_group(
            static_cast<uint32_t>(context_len), 8);
        const uint32_t blocks =
            (static_cast<uint32_t>(q_heads) + works_per_block - 1) /
            works_per_block;
        return aclrtlaunch_qwen_gqa_decode_attention_vector_kernel(
                   blocks, resolve(stream), gm(d_q_fp16), gm(d_k_cache_fp16),
                   gm(d_v_cache_fp16), gm(d_out_fp16), gm(d_score_scratch),
                   static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
                   static_cast<uint32_t>(head_dim),
                   static_cast<uint32_t>(context_len),
                   static_cast<uint32_t>(max_context), attention_scale(head_dim),
                   works_per_block) == kLaunchOk;
    }

    // The scalar baseline remains the safe path for unaligned and strided KV
    // geometries. Its single block also protects irregular scratch records.
    return aclrtlaunch_qwen_gqa_decode_attention_kernel(
               1u, resolve(stream), gm(d_q_fp16), gm(d_k_cache_fp16),
               gm(d_v_cache_fp16), gm(d_out_fp16), gm(d_score_scratch),
               static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(context_len),
               static_cast<uint32_t>(max_context), attention_scale(head_dim)) ==
           kLaunchOk;
}

bool qwen_gqa_decode_attention_flashdec_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_partials_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, int num_partitions, void* stream) {
    if (d_q_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_fp16 == nullptr ||
        d_partials_scratch == nullptr || num_partitions <= 0 ||
        !valid_attention(q_heads, kv_heads, head_dim, context_len,
                         max_context)) {
        return false;
    }

    if (!vector_attention_geometry(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                   d_out_fp16, q_heads, kv_heads, head_dim)) {
        return false;
    }

    // Launch partial computation kernel (one block per partition)
    const int partial_result = aclrtlaunch_qwen_gqa_decode_attention_flashdec_partial_kernel(
        static_cast<uint32_t>(num_partitions), resolve(stream),
        gm(d_q_fp16), gm(d_k_cache_fp16), gm(d_v_cache_fp16),
        gm(d_partials_scratch),
        static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(context_len),
        static_cast<uint32_t>(max_context), attention_scale(head_dim),
        static_cast<uint32_t>(num_partitions));

    if (partial_result != kLaunchOk) {
        return false;
    }

    // Launch reduction kernel (single block)
    const int reduce_result = aclrtlaunch_qwen_gqa_decode_attention_flashdec_reduce_kernel(
        1u, resolve(stream),
        gm(d_partials_scratch), gm(d_out_fp16),
        static_cast<uint32_t>(q_heads), static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(num_partitions));

    return reduce_result == kLaunchOk;
}

bool qwen_gqa_prefill_attention_f16_ascend(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, int seq_len,
    int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, void* stream) {
    if (d_q_rows_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_rows_fp16 == nullptr ||
        seq_len <= 0 || position_offset < 0 ||
        position_offset + seq_len > max_context ||
        !valid_attention(q_heads, kv_heads, head_dim, position_offset + seq_len,
                         max_context)) {
        return false;
    }
    const uint64_t work = static_cast<uint64_t>(seq_len) * q_heads;

    // Cube first: where the geometry fits it is the faster of the two, and any
    // shape it cannot express -- or a failed launch -- falls through to the vector
    // kernel below rather than failing the operator.
    if (cube_attention_geometry(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                d_out_rows_fp16, q_heads, kv_heads, head_dim) &&
        cube_prefill_attention(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16,
                               d_out_rows_fp16, seq_len, q_heads, kv_heads,
                               head_dim, position_offset, max_context, stream)) {
        return true;
    }

    if (vector_attention_geometry(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                  d_out_rows_fp16, q_heads, kv_heads, head_dim)) {
        // Not overridable on purpose: this value is what keeps adjacent blocks'
        // records off a shared GM cache line, so a smaller one is a correctness
        // bug rather than a tuning choice.
        const uint32_t works_per_block = aligned_work_group(
            static_cast<uint32_t>(head_dim), 16);
        const uint32_t blocks =
            (static_cast<uint32_t>(work) + works_per_block - 1) /
            works_per_block;
        return aclrtlaunch_qwen_gqa_prefill_attention_vector_kernel(
                   blocks, resolve(stream), gm(d_q_rows_fp16),
                   gm(d_k_cache_fp16), gm(d_v_cache_fp16), gm(d_out_rows_fp16),
                   static_cast<uint32_t>(seq_len), static_cast<uint32_t>(q_heads),
                   static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim),
                   static_cast<uint32_t>(position_offset),
                   static_cast<uint32_t>(max_context), attention_scale(head_dim),
                   works_per_block) == kLaunchOk;
    }
    return aclrtlaunch_qwen_gqa_prefill_attention_kernel(
               blocks_for(work), resolve(stream), gm(d_q_rows_fp16),
               gm(d_k_cache_fp16), gm(d_v_cache_fp16), gm(d_out_rows_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(q_heads),
               static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(position_offset),
               static_cast<uint32_t>(max_context),
               attention_scale(head_dim)) == kLaunchOk;
}

bool qwen_gqa_verify_attention_f16_ascend(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    float* d_partial_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context, int splits,
    void* stream) {
    // splits is also the scratch geometry the caller sized
    // [rows, q_heads, splits, head_dim + 2] against, so a mismatch here is an
    // out-of-bounds write rather than a wrong answer.
    if (d_q_rows_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_rows_fp16 == nullptr ||
        d_partial_scratch == nullptr || rows < 2 || rows > 8 || head_dim > 256 ||
        position_offset < 0 || max_context < rows ||
        position_offset > max_context - rows || splits <= 0 || splits > 256 ||
        !valid_attention(q_heads, kv_heads, head_dim, position_offset + rows,
                         max_context)) {
        return false;
    }
    if (vector_attention_geometry(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16,
                                  d_out_rows_fp16, q_heads, kv_heads, head_dim) &&
        (reinterpret_cast<uintptr_t>(d_partial_scratch) & (kAttentionAlignmentBytes - 1)) == 0) {
        const uint32_t record_elements =
            static_cast<uint32_t>(splits * (head_dim + 2));
        const uint32_t works_per_block = aligned_work_group(record_elements, 8);
        const uint32_t total = static_cast<uint32_t>(rows * q_heads);
        const uint32_t blocks = (total + works_per_block - 1) / works_per_block;
        return aclrtlaunch_qwen_gqa_verify_attention_vector_kernel(
                   blocks, resolve(stream), gm(d_q_rows_fp16),
                   gm(d_k_cache_fp16), gm(d_v_cache_fp16), gm(d_out_rows_fp16),
                   gm(d_partial_scratch), static_cast<uint32_t>(rows),
                   static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
                   static_cast<uint32_t>(head_dim),
                   static_cast<uint32_t>(position_offset),
                   static_cast<uint32_t>(max_context), static_cast<uint32_t>(splits),
                   attention_scale(head_dim), works_per_block) == kLaunchOk;
    }

    // Irregular records remain serialized so adjacent blocks cannot race in a
    // shared 32-byte GM cache line.
    return aclrtlaunch_qwen_gqa_verify_attention_kernel(
               1u, resolve(stream), gm(d_q_rows_fp16), gm(d_k_cache_fp16),
               gm(d_v_cache_fp16), gm(d_out_rows_fp16), gm(d_partial_scratch),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(q_heads),
               static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(position_offset),
               static_cast<uint32_t>(max_context),
               static_cast<uint32_t>(splits), attention_scale(head_dim)) ==
           kLaunchOk;
}

// Greedy top-1 per logits row. aclnn has ArgMax, but it returns only the index
// and the caller needs the winning logit as well to merge vocabulary shards
// across TP ranks; running an aclnn Max beside it would scan the row twice and
// would not guarantee the two agree on which duplicate maximum won.
bool qwen_argmax_fp32_rows_ascend(const float* d_logits, int* d_tokens,
                                  float* d_logits_out, int rows, int count,
                                  int token_offset, void* stream) {
    // Exactly the CUDA launcher's preconditions, including requiring
    // d_logits_out and accepting a negative token_offset, so a caller cannot
    // tell the two backends apart by what they reject.
    if (d_logits == nullptr || d_tokens == nullptr || d_logits_out == nullptr ||
        rows <= 0 || count <= 0) {
        return false;
    }
    // The kernel gives each block eight consecutive rows so its two 4-byte-per-row
    // output spans are whole 32-byte cache lines; block count follows that grouping,
    // not the row count, or the extra blocks would idle.
    const uint64_t groups = (static_cast<uint64_t>(rows) + 7) / 8;
    return aclrtlaunch_qwen_argmax_f32_rows_kernel(
               blocks_for(groups), resolve(stream),
               gm(d_logits), gm(d_tokens), gm(d_logits_out),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(count),
               static_cast<int32_t>(token_offset)) == kLaunchOk;
}

// Measurement-only: streams tile_count 64x256 FP16 tiles and touches every
// element once, to establish the achievable HBM read bandwidth that the
// bandwidth-bound decode attention kernels are judged against. Fixed at the 30
// cores those kernels use so the result is an upper bound they could reach.
bool qwen_hbm_read_probe_ascend(const uint16_t* d_source, uint16_t* d_sink,
                                int tile_count, void* stream) {
    // Not implemented yet - stub for measurement
    (void)d_source;
    (void)d_sink;
    (void)tile_count;
    (void)stream;
    return false;
}

// One-core Cube tile, used to validate the L1 fractal layout that AscendC::Gemm
// expects against a CPU reference. Not part of any inference path.
//
// `mode == 0` writes the logical [m, n] product. `mode == 1` hands back the
// untouched L0C image, whose size is the tile rounded up to Gemm's blockSize
// (16, not 16 * 16 -- the comment inside GetGemmTiling is wrong), so the caller
// must size `d_c` for round-up(m) * round-up(n) in that mode. `mode == 2` is the
// same readout with `readout_len` overriding the copy length the host sweep uses
// to pin down the unit.
bool qwen_cube_gemm_probe(const uint16_t* d_a, const uint16_t* d_b, uint16_t* d_c,
                          int m, int n, int k, uint32_t mode, uint32_t readout_len,
                          void* stream) {
    if (d_a == nullptr || d_b == nullptr || d_c == nullptr || m <= 0 || n <= 0 ||
        k <= 0 || m > 4096 || n > 4096 || k > 4096 || (k % 16) != 0 ||
        (mode == 0 && (n % 16) != 0) || readout_len > 0xffffu) {
        return false;
    }
    return aclrtlaunch_qwen_cube_gemm_probe_kernel(
               1, resolve(stream), gm(d_a), gm(d_b), gm(d_c),
               static_cast<uint32_t>(m), static_cast<uint32_t>(n),
               static_cast<uint32_t>(k), mode, readout_len) == kLaunchOk;
}

// Same one-core, one-tile cube oracle, but for the transposed operand: `d_b` is
// the [k, n] matrix and the answer is A * B. `transpose` picks between the two
// candidate semantics of LoadData2DParams::ifTranspose -- see the device kernel.
bool qwen_cube_transpose_probe(const uint16_t* d_a, const uint16_t* d_b, uint16_t* d_c,
                               int m, int n, int k, uint32_t transpose, void* stream) {
    if (d_a == nullptr || d_b == nullptr || d_c == nullptr || m <= 0 || n <= 0 ||
        k <= 0 || (m % 16) != 0 || (n % 16) != 0 || (k % 16) != 0 ||
        m * k * 2 > 1024 * 1024 || n * k * 2 > 1024 * 1024) {
        return false;
    }
    return aclrtlaunch_qwen_cube_transpose_probe_kernel(
               1, resolve(stream), gm(d_a), gm(d_b), gm(d_c),
               static_cast<uint32_t>(m), static_cast<uint32_t>(n),
               static_cast<uint32_t>(k), transpose, 0) == kLaunchOk;
}

// fp16 transpose: d_dst[cols, rows] = transpose(d_src[rows, cols]).
//
// The whole grid is used rather than one core, because this is not an oracle but
// the transpose step the Cube attention kernel needs for V -- and that step runs
// once per layer on a matrix that can be megabytes. Both extents must be
// multiples of 16: the tiling has no partial-fractal case.
bool qwen_transpose_f16_ascend(const uint16_t* d_src, uint16_t* d_dst, int rows, int cols,
                               void* stream) {
    if (d_src == nullptr || d_dst == nullptr || rows <= 0 || cols <= 0 ||
        (rows % 16) != 0 || (cols % 16) != 0) {
        return false;
    }
    return aclrtlaunch_qwen_transpose_f16_kernel(
               core_count(), resolve(stream), gm(d_src), gm(d_dst),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
               static_cast<uint32_t>(rows)) == kLaunchOk;
}

// The Cube attention variant of the above, with the extents that kernel needs.
//
// `src_rows` source rows are transposed into a destination of `cols` rows whose
// column extent is `rows` (>= src_rows) and whose row stride is `dst_pitch`
// (>= rows); the destination columns past `src_rows` come out zero. `src_pitch` is
// the source row stride, so one call transposes one head's slice out of a cache
// where the heads are interleaved.
//
// The padding is what lets the attention kernel read V in whole 16-row chunks
// without a bounds check: the padded columns meet probabilities that masking has
// already zeroed, so zeros contribute nothing where uninitialised memory would
// contribute a NaN. Every extent except `src_rows` must be a multiple of 16.
bool qwen_transpose_f16_padded_ascend(const uint16_t* d_src, uint16_t* d_dst, int rows,
                                      int cols, int src_rows, int src_pitch, int dst_pitch,
                                      void* stream) {
    if (d_src == nullptr || d_dst == nullptr || rows <= 0 || cols <= 0 ||
        src_rows <= 0 || src_rows > rows || src_pitch < cols || dst_pitch < rows ||
        (rows % 16) != 0 || (cols % 16) != 0 || (src_pitch % 16) != 0 ||
        (dst_pitch % 16) != 0) {
        return false;
    }
    return aclrtlaunch_qwen_transpose_f16_kernel(
               core_count(), resolve(stream), gm(d_src), gm(d_dst),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
               static_cast<uint32_t>(dst_pitch), static_cast<uint32_t>(src_pitch),
               static_cast<uint32_t>(src_rows)) == kLaunchOk;
}

}  // namespace pocket
