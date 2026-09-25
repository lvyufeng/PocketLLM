#include "qwen_target_head.hpp"

#include "qwen_ops.hpp"

namespace pocket {

bool QwenTargetHeadAdapter::valid() const {
    if (weight == nullptr || weight->data == nullptr || local_vocab <= 0 ||
        hidden_size <= 0) {
        return false;
    }
    // A folded weight with no transform to reach its frame is not a usable head:
    // the projection would return numbers, from the wrong basis.
    if (input_rotated && transform.forward == nullptr) return false;
    if (kind == QwenLinearKind::DenseF16) {
        return weight->device_dtype == SafeDType::F16;
    }
    if (kind == QwenLinearKind::Fp8Channel) {
        return weight->device_dtype == SafeDType::F8_E4M3 && scale != nullptr &&
               scale->data != nullptr && scale->device_dtype == SafeDType::F16;
    }
    if (kind == QwenLinearKind::Ptq1_0) {
        // The blocks are the whole weight: the scale lives in them, one fp16 per
        // 128 weights, so there is no second tensor to require here.
        return weight->device_dtype == SafeDType::U8;
    }
    return false;
}

bool QwenTargetHeadAdapter::project_f16_to_f32(
    const uint16_t* hidden, float* logits, int rows, void* stream) const {
    if (!valid() || hidden == nullptr || logits == nullptr || rows <= 0) {
        return false;
    }
    if (input_rotated) {
        // A folded weight reads the rotated activation. Without the transform
        // there is nothing to fall back to: the unrotated projection is a
        // well-formed matrix multiply of the wrong numbers.
        if (transform.forward == nullptr) return false;
        hidden = transform.forward(transform.context, hidden, rows, hidden_size);
        if (hidden == nullptr) return false;
    }
    if (kind == QwenLinearKind::DenseF16) {
#ifdef POCKET_BACKEND_ASCEND
        // The portable Ascend path is the only dense implementation available
        // here; cublas_fp32 is a CUDA tuning hint and must not affect it.
        (void)cublas_fp32;
        return qwen_fp16_matmul_rows_f16_f32(
            hidden, weight->f16_data(), logits, rows, local_vocab,
            hidden_size, hidden_size, local_vocab, hidden_size, stream);
#else
        return cublas_fp32
            ? qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                  hidden, weight->f16_data(), logits, rows, local_vocab,
                  hidden_size, hidden_size, local_vocab, hidden_size, stream)
            : qwen_fp16_matmul_rows_f16_f32(
                  hidden, weight->f16_data(), logits, rows, local_vocab,
                  hidden_size, hidden_size, local_vocab, hidden_size, stream);
#endif
    }
#ifdef POCKET_BACKEND_ASCEND
    (void)stream;
    // First-generation Ascend bring-up supports dense FP16 target heads only.
    return false;
#else
    if (kind == QwenLinearKind::Ptq1_0) {
        return rows == 1
            ? qwen_ptq1_0_matvec_f16_f32_cuda(hidden, weight->u8_data(), logits,
                                              local_vocab, hidden_size, stream)
            : qwen_ptq1_0_matmul_rows_f16_f32_cuda(
                  hidden, weight->u8_data(), logits, rows, local_vocab,
                  hidden_size, hidden_size, local_vocab, stream);
    }
    return qwen_fp8_e4m3_channel_matmul_rows_f16_f32_cuda(
        hidden, weight->fp8_data(), scale->f16_data(), logits, rows,
        local_vocab, hidden_size, hidden_size, local_vocab, hidden_size,
        stream);
#endif
}

}  // namespace pocket
