#pragma once

#include "qwen_config.hpp"
#include "safetensors_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dsv4 {

enum class QwenShardRule {
    Replicated,
    ColumnParallel,
    RowParallel,
    ParallelEmbedding,
    ParallelHead,
    PackedQkvColumnParallel,
    PackedConvChannelParallel,
};

struct QwenTensorRef {
    std::string name;
    std::string shard_name;
    // dtype is the checkpoint/storage dtype. device_dtype is the dtype the
    // Qwen loader must materialize on Turing GPUs.
    SafeDType dtype = SafeDType::Unknown;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> full_shape;
    std::vector<uint64_t> local_shape;
    QwenShardRule rule = QwenShardRule::Replicated;
    int shard_dim = -1;
    uint64_t shard_start = 0;
    uint64_t shard_size = 0;
    uint64_t nbytes = 0;
    uint64_t device_nbytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> segments;
    bool found = false;
};

struct QwenHostTensor {
    SafeDType storage_dtype = SafeDType::Unknown;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

struct QwenDeviceTensor {
    void* data = nullptr;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    // nbytes is the logical extent currently exposed to an operator. capacity
    // is the allocation size, so workspaces can reuse a larger buffer.
    uint64_t nbytes = 0;
    uint64_t capacity = 0;

    ~QwenDeviceTensor();
    QwenDeviceTensor() = default;
    QwenDeviceTensor(const QwenDeviceTensor&) = delete;
    QwenDeviceTensor& operator=(const QwenDeviceTensor&) = delete;
    QwenDeviceTensor(QwenDeviceTensor&& other) noexcept;
    QwenDeviceTensor& operator=(QwenDeviceTensor&& other) noexcept;
};

struct QwenLinearRef {
    QwenTensorRef weight;
    QwenTensorRef scale;
    bool has_scale = false;
};

struct QwenLinearAttentionWeights {
    QwenLinearRef in_proj_qkv;
    QwenLinearRef in_proj_z;
    QwenLinearRef out_proj;
    QwenLinearRef in_proj_a;
    QwenLinearRef in_proj_b;
    QwenTensorRef conv1d;
    QwenTensorRef a_log;
    QwenTensorRef dt_bias;
    QwenTensorRef norm;
};

struct QwenFullAttentionWeights {
    QwenLinearRef q_proj;
    QwenLinearRef k_proj;
    QwenLinearRef v_proj;
    QwenLinearRef o_proj;
    QwenTensorRef q_norm;
    QwenTensorRef k_norm;
};

struct QwenMlpWeights {
    QwenLinearRef gate_proj;
    QwenLinearRef up_proj;
    QwenLinearRef down_proj;
};

struct QwenLayerWeights {
    QwenTensorRef input_layernorm;
    QwenTensorRef post_attention_layernorm;
    QwenLinearAttentionWeights linear_attention;
    QwenFullAttentionWeights full_attention;
    QwenMlpWeights mlp;
};

class QwenWeightMap {
public:
    QwenWeightMap(const SafeTensorsIndex& index, const QwenConfig& config,
                  int tp_world = 1, int tp_rank = 0);

    const QwenTensorRef& embed_tokens() const { return embed_tokens_; }
    const QwenTensorRef& final_norm() const { return final_norm_; }
    const QwenTensorRef& lm_head() const { return lm_head_; }
    const std::vector<QwenLayerWeights>& layers() const { return layers_; }
    const QwenConfig& config() const { return config_; }
    int tp_world() const { return tp_world_; }
    int tp_rank() const { return tp_rank_; }

    uint64_t local_weight_bytes() const { return local_weight_bytes_; }
    uint64_t local_scale_bytes() const { return local_scale_bytes_; }
    size_t tensor_count() const { return tensor_count_; }

private:
    QwenTensorRef require_tensor(const std::string& name,
                                 SafeDType dtype,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule = QwenShardRule::Replicated,
                                 int shard_dim = -1) const;
    QwenLinearRef require_linear(const std::string& name,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule,
                                 int shard_dim,
                                 SafeDType weight_dtype = SafeDType::F8_E4M3) const;
    void record(const QwenTensorRef& ref, bool scale);

    const SafeTensorsIndex& index_;
    QwenConfig config_;
    int tp_world_ = 1;
    int tp_rank_ = 0;
    QwenTensorRef embed_tokens_;
    QwenTensorRef final_norm_;
    QwenTensorRef lm_head_;
    std::vector<QwenLayerWeights> layers_;
    uint64_t local_weight_bytes_ = 0;
    uint64_t local_scale_bytes_ = 0;
    size_t tensor_count_ = 0;
};

const char* qwen_shard_rule_name(QwenShardRule rule);

// Qwen checkpoint tensors retain their source dtype for validation. On Turing,
// BF16 tensors must be materialized as IEEE FP16 before device upload.
SafeDType qwen_device_dtype(SafeDType storage_dtype);
uint16_t qwen_bf16_to_fp16_bits(uint16_t bits);
void qwen_convert_bf16_to_fp16(const uint16_t* src, uint16_t* dst, size_t count);

// Materialize a local tensor from its mmap'd source shard. The output is ready
// for a Turing GPU upload: BF16 storage is converted to FP16 and packed/row
// slices are copied without expanding FP8 weights.
QwenHostTensor qwen_materialize_host_tensor(const SafeTensorsIndex& index,
                                            const QwenTensorRef& ref);
QwenDeviceTensor qwen_upload_tensor_cuda(const SafeTensorsIndex& index,
                                         const QwenTensorRef& ref,
                                         void* stream = nullptr);

}  // namespace dsv4
