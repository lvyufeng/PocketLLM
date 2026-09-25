#pragma once

#include "qwen_weights.hpp"

#include <cstdint>

namespace pocket {

// Non-owning view of the target model's vocab-parallel head. External drafters
// use this instead of assuming the checkpoint keeps a dense FP16 lm_head.
// The incoherence transform a folded head weight needs applied to its input.
//
// The adapter cannot own the transform: the sign vectors and the scratch buffer
// belong to the engine, which is also what knows the block size. It is therefore
// a callback rather than a copy of the state, and a folded head with no callback
// is refused rather than run unrotated -- that mistake does not fail, it
// generates.
struct QwenTargetHeadTransform {
    const uint16_t* (*forward)(void* context, const uint16_t* input, int rows,
                               int width) = nullptr;
    void* context = nullptr;
};

struct QwenTargetHeadAdapter {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    const QwenDeviceTensor* weight = nullptr;
    const QwenDeviceTensor* scale = nullptr;
    int local_vocab = 0;
    int hidden_size = 0;
    uint64_t vocab_start = 0;
    bool cublas_fp32 = false;
    // Set when the head weight is folded, together with the transform that
    // rotates into its frame.
    bool input_rotated = false;
    QwenTargetHeadTransform transform;

    bool valid() const;
    bool project_f16_to_f32(const uint16_t* hidden, float* logits, int rows,
                            void* stream = nullptr) const;
};

}  // namespace pocket
