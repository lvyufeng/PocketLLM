// AscendC kernel for attention score post-processing: scale + causal mask + softmax
// This kernel bridges the gap between QK^T matmul and P·V matmul in batched attention
//
// Input: scores [batch, seq_len, ctx_len] from QK^T matmul
// Output: probabilities [batch, seq_len, ctx_len] ready for P·V matmul
//
// Operations:
//   1. Scale by 1/sqrt(head_dim)
//   2. Apply causal mask (set scores[i,j] = -inf where j > position_offset + i)
//   3. Softmax over ctx_len dimension (numerically stable)

#include "qwen_ascend_kernel_common.hpp"
#include <limits>

namespace AscendC {

constexpr int TILE_SIZE = 256;  // Process 256 context positions at a time

template<typename T>
class ScaleMaskSoftmaxKernel {
public:
    __aicore__ inline ScaleMaskSoftmaxKernel() {}

    __aicore__ inline void Init(
        GM_ADDR scores_gm,       // [batch, seq_len, ctx_len]
        GM_ADDR probs_gm,        // [batch, seq_len, ctx_len]
        int batch,
        int seq_len,
        int ctx_len,
        int position_offset,
        float scale_factor) {

        scores_gm_ = reinterpret_cast<__gm__ T*>(scores_gm);
        probs_gm_ = reinterpret_cast<__gm__ T*>(probs_gm);
        batch_ = batch;
        seq_len_ = seq_len;
        ctx_len_ = ctx_len;
        position_offset_ = position_offset;
        scale_factor_ = scale_factor;

        // Allocate local buffers
        pipe_.InitBuffer(input_queue_, 1, TILE_SIZE * sizeof(T));
        pipe_.InitBuffer(output_queue_, 1, TILE_SIZE * sizeof(T));
        pipe_.InitBuffer(max_queue_, 1, sizeof(float));
        pipe_.InitBuffer(sum_queue_, 1, sizeof(float));
    }

    __aicore__ inline void Process() {
        // Process each batch and sequence position independently
        for (int b = 0; b < batch_; ++b) {
            for (int s = 0; s < seq_len_; ++s) {
                ProcessRow(b, s);
            }
        }
    }

private:
    __aicore__ inline void ProcessRow(int batch_idx, int seq_idx) {
        const int row_offset = (batch_idx * seq_len_ + seq_idx) * ctx_len_;
        const int max_valid_pos = position_offset_ + seq_idx;

        // Two-pass algorithm for numerically stable softmax:
        // Pass 1: Find max value (ignoring masked positions)
        float max_val = -INFINITY;

        for (int tile_start = 0; tile_start < ctx_len_; tile_start += TILE_SIZE) {
            const int tile_len = (tile_start + TILE_SIZE <= ctx_len_) ?
                                 TILE_SIZE : (ctx_len_ - tile_start);

            // Load tile
            LocalTensor<T> input_local = input_queue_.AllocTensor<T>();
            DataCopy(input_local, scores_gm_[row_offset + tile_start], tile_len);
            input_queue_.EnQue(input_local);

            LocalTensor<T> input_work = input_queue_.DeQue<T>();

            // Find max in tile (only for non-masked positions)
            for (int i = 0; i < tile_len; ++i) {
                const int pos = tile_start + i;
                if (pos <= max_valid_pos) {
                    float val = static_cast<float>(input_work.GetValue(i)) * scale_factor_;
                    max_val = (val > max_val) ? val : max_val;
                }
            }

            input_queue_.FreeTensor(input_work);
        }

        // Pass 2: Compute exp(x - max) and sum
        float sum_exp = 0.0f;

        for (int tile_start = 0; tile_start < ctx_len_; tile_start += TILE_SIZE) {
            const int tile_len = (tile_start + TILE_SIZE <= ctx_len_) ?
                                 TILE_SIZE : (ctx_len_ - tile_start);

            // Load tile
            LocalTensor<T> input_local = input_queue_.AllocTensor<T>();
            DataCopy(input_local, scores_gm_[row_offset + tile_start], tile_len);
            input_queue_.EnQue(input_local);

            LocalTensor<T> input_work = input_queue_.DeQue<T>();
            LocalTensor<T> output_local = output_queue_.AllocTensor<T>();

            // Compute exp(scale * x - max) or 0 for masked positions
            for (int i = 0; i < tile_len; ++i) {
                const int pos = tile_start + i;
                float val;
                if (pos <= max_valid_pos) {
                    val = exp(static_cast<float>(input_work.GetValue(i)) * scale_factor_ - max_val);
                    sum_exp += val;
                } else {
                    val = 0.0f;  // Masked position
                }
                output_local.SetValue(i, static_cast<T>(val));
            }

            output_queue_.EnQue(output_local);
            input_queue_.FreeTensor(input_work);
        }

        // Pass 3: Normalize by sum
        const float inv_sum = 1.0f / sum_exp;

        for (int tile_start = 0; tile_start < ctx_len_; tile_start += TILE_SIZE) {
            const int tile_len = (tile_start + TILE_SIZE <= ctx_len_) ?
                                 TILE_SIZE : (ctx_len_ - tile_start);

            LocalTensor<T> output_work = output_queue_.DeQue<T>();

            // Normalize
            for (int i = 0; i < tile_len; ++i) {
                float val = static_cast<float>(output_work.GetValue(i)) * inv_sum;
                output_work.SetValue(i, static_cast<T>(val));
            }

            // Write back
            DataCopy(probs_gm_[row_offset + tile_start], output_work, tile_len);
            output_queue_.FreeTensor(output_work);
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> input_queue_;
    TQue<QuePosition::VECOUT, 1> output_queue_;
    TQue<QuePosition::VECIN, 1> max_queue_;
    TQue<QuePosition::VECIN, 1> sum_queue_;

    __gm__ T* scores_gm_;
    __gm__ T* probs_gm_;
    int batch_;
    int seq_len_;
    int ctx_len_;
    int position_offset_;
    float scale_factor_;
};

} // namespace AscendC

extern "C" __global__ __aicore__ void qwen_attention_scale_mask_softmax_f16_kernel(
    GM_ADDR scores_gm,
    GM_ADDR probs_gm,
    GM_ADDR args_gm) {

    // Unpack arguments
    __gm__ int* args = reinterpret_cast<__gm__ int*>(args_gm);
    int batch = args[0];
    int seq_len = args[1];
    int ctx_len = args[2];
    int position_offset = args[3];
    float scale_factor = *reinterpret_cast<__gm__ float*>(&args[4]);

    AscendC::ScaleMaskSoftmaxKernel<half> kernel;
    kernel.Init(scores_gm, probs_gm, batch, seq_len, ctx_len, position_offset, scale_factor);
    kernel.Process();
}

// Host-side wrapper
namespace pocket {

bool qwen_attention_scale_mask_softmax_f16_ascend(
    const uint16_t* d_scores_fp16,      // [batch, seq_len, ctx_len]
    uint16_t* d_probs_fp16,             // [batch, seq_len, ctx_len]
    int batch, int seq_len, int ctx_len,
    int position_offset, float scale_factor,
    void* stream) {

    // Pack kernel arguments
    int args[5];
    args[0] = batch;
    args[1] = seq_len;
    args[2] = ctx_len;
    args[3] = position_offset;
    *reinterpret_cast<float*>(&args[4]) = scale_factor;

    // TODO: Allocate device memory for args, copy, launch kernel
    // This requires integration with the Ascend runtime infrastructure

    return true;
}

} // namespace pocket
