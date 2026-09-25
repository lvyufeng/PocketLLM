#pragma once

#include "qwen_config.hpp"
#include "safetensors_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace pocket {

enum class QwenShardRule {
    Replicated,
    ColumnParallel,
    RowParallel,
    ParallelEmbedding,
    ParallelHead,
    PackedQkvColumnParallel,
    PackedConvChannelParallel,
};

enum class QwenLinearKind {
    DenseF16,
    Fp8Block128,
    Fp8Channel,
    NvFp4Group16,
    // The fork-private ternary pack (GGML type 143): 128 three-valued weights in
    // 28 bytes of packed trits plus an fp16 block scale, 1.75 bits per weight.
    // The kernel reads the blocks; nothing expands them to a float matrix.
    Ptq1_0,
};

// The incoherence transform a ternary container declares. Forward-declared
// because the source interface answers with it and only the containers that
// declare one have to know what it is.
class QwenHadamardSpec;

// One tensor as a checkpoint holds it, resolved from its canonical name.
//
// `shape` is spelled the way this file and the engine spell shapes -- [out, in],
// or [n] -- rather than the way any one container stores them, so a reader whose
// format orders dimensions the other way says so in its own lookup instead of
// making every caller remember. `dtype` is the storage dtype the map validates
// against; the device dtype follows from it and the backend policy, as it does
// for every other tensor here.
struct QwenSourceTensor {
    // The canonical name the lookup was asked for. Kept so that `data_of` can
    // reach the bytes without a second argument: the two sources each have to
    // re-find the tensor in whatever structure their container keeps, and one of
    // them has only the name to find it by.
    std::string name;
    SafeDType dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    // Which file the bytes are in. A single-file checkpoint (GGUF) names itself
    // here; a sharded one names the shard.
    std::string shard_name;
    // The storage is a ternary block pack rather than an element array, so a row
    // is not `in * item_size` bytes wide and only the type's own geometry can
    // count it.
    bool ternary_blocks = false;
    bool present = false;
};

// A checkpoint as the weight map needs it: one name lookup, the bytes behind a
// hit, the full name list for coverage, and a format name for the errors.
//
// The map is written against this rather than against either container, which is
// what lets one shape table, one TP sharding rule set and one coverage check
// serve an HF safetensors directory and a single-file GGUF. The two readers
// differ in exactly one more place -- whether the norm gamma arrives with the
// (1 + gamma) convention already folded in -- and that is a question the source
// answers about itself.
class QwenCheckpointSource {
public:
    virtual ~QwenCheckpointSource() = default;

    // `canonical_name` is the HF spelling; each source translates it itself.
    virtual QwenSourceTensor lookup(const std::string& canonical_name) const = 0;

    // Base of the tensor's storage, valid for as long as the source is.
    virtual const uint8_t* data_of(const QwenSourceTensor& tensor) const = 0;

    // Every tensor the checkpoint holds, in canonical spelling. A name the
    // translation does not know comes back unchanged, so coverage reports it as
    // unmapped rather than silently dropping it.
    virtual std::vector<std::string> tensor_names() const = 0;
    virtual size_t tensor_count() const = 0;
    virtual const char* format_name() const = 0;

    // What the backend keeps resident for a storage dtype. The default is the
    // CUDA/SM75 policy; a source overrides it where its own container states the
    // dtype differently from the model it holds.
    virtual SafeDType device_dtype(SafeDType storage_dtype) const;

    // Whether a tensor the map declared with `declared` may be stored as
    // `actual` here. An HF directory stores exactly what a model's tensors are,
    // so its answer is equality. A GGUF chooses a GGML type per tensor, and its
    // declaration is about the *model* rather than about the file -- the widths
    // it picks among are still a short list, which is what keeps this a check.
    virtual bool accepts_storage_dtype(SafeDType declared, SafeDType actual,
                                       bool ternary) const {
        (void)ternary;
        return declared == actual;
    }

    // The storage row each canonical row's bytes are in, or empty for the
    // identity.
    //
    // A container can hold a matrix's rows in a different order than the model
    // does -- a graph that broadcasts one axis wants that axis tiled where
    // training grouped it, and the conversion is a place to do the swap. This is
    // where a container says so, and the materializer reads through it: an entry
    // is the row of the *file* that a row of the *model* is stored in. Empty for
    // everything whose storage is already the model's order.
    virtual std::vector<uint64_t> row_order(const QwenSourceTensor& tensor) const {
        (void)tensor;
        return {};
    }

    // True when the stored norm gamma is already `1 + gamma`. The released GGUF
    // bakes the convention in because its kernels apply the weight directly; an
    // HF export stores gamma and lets the runtime add the one.
    virtual bool folds_one_plus_norm_gamma() const { return false; }

    // True when the stored decay is already `-exp(A_log)`. Same reason as the
    // norm fold: the fork's converter applies the exponential for a GGML graph,
    // which multiplies by the decay, while the gates kernel here takes the log
    // and applies `-exp` itself. An HF export stores `A_log` and needs no fold.
    virtual bool folds_negative_exp_a_log() const { return false; }

    // The incoherence transform this container declares, or null when it
    // declares none -- which is every checkpoint that is not ternary, since a
    // 1.75-bit weight is the reason to have one at all. The map reads it to give
    // each linear the input transform its weight was folded against, and the
    // engine reads it to rotate the activations that meet those weights.
    virtual const QwenHadamardSpec* hadamard() const { return nullptr; }
};

// The HF safetensors directory, as a source. A directory is many files, so the
// lookup opens the one the tensor is in and keeps it open.
class SafeTensorsCheckpointSource : public QwenCheckpointSource {
public:
    // An index the caller owns and keeps alive. This is the spelling every
    // caller written before the engine had a source uses.
    explicit SafeTensorsCheckpointSource(const SafeTensorsIndex& index)
        : owned_index_(), index_(index) {}
    // A directory this source owns. The engine opens a checkpoint by path and
    // has nowhere else to keep an index alive, so the reading is the source's.
    explicit SafeTensorsCheckpointSource(const std::string& directory);

    QwenSourceTensor lookup(const std::string& canonical_name) const override;
    const uint8_t* data_of(const QwenSourceTensor& tensor) const override;
    std::vector<std::string> tensor_names() const override;
    size_t tensor_count() const override;
    const char* format_name() const override { return "safetensors"; }

    const SafeTensorsIndex& index() const { return index_; }

private:
    // Set only by the directory constructor; `index_` refers to it.
    std::unique_ptr<SafeTensorsIndex> owned_index_;
    const SafeTensorsIndex& index_;
    // Shards stay open for the source's lifetime. The materializer reads through
    // a pointer into the mmap, and a shard closed at the end of the lookup would
    // leave it dangling -- so the open handle is held here rather than by the
    // caller, which is the same thing SafetensorsWeightSource does.
    mutable std::map<std::string, std::unique_ptr<SafeTensorsShard>> shards_;
};

// The checkpoint a path names, as a source: a single-file GGUF, or an HF
// safetensors directory. The extension is the whole of the choice -- a GGUF is
// one file and a directory cannot be one -- and both containers answer the same
// interface, which is what lets every caller above this line stay blind to
// which of them it was handed.
std::unique_ptr<QwenCheckpointSource> open_qwen_checkpoint(const std::string& path);

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
    // The whole tensor's bytes in the container, as against `nbytes`, which is
    // this rank's share. The distinction matters for the coverage audit: it
    // reports a checkpoint's size, and for a block-packed tensor that is the
    // packing's, not the logical shape multiplied out.
    uint64_t full_nbytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> segments;
    // full_shape and local_shape are the *logical* shape for a block-packed
    // tensor -- its rows count weights -- so bytes are not the shape multiplied
    // out. A byte count that did that would overstate a ternary tensor by 128/28,
    // which is the difference between reporting a checkpoint's size and reporting
    // what it would have been as an element array.
    bool ternary_blocks = false;
    bool found = false;
};

struct QwenHostTensor {
    SafeDType storage_dtype = SafeDType::Unknown;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

// Lossless device record for 64 logical compressed-tensors NVFP4 weights.
// Four E4M3 group-16 scales precede the original low-nibble-first E2M1 codes.
struct QwenNvfp4Block64 {
    uint8_t d[4];
    uint8_t qs[32];
};
static_assert(sizeof(QwenNvfp4Block64) == 36,
              "Qwen NVFP4 block64 layout changed");

struct QwenNvfp4HostLinear {
    std::vector<uint64_t> logical_shape;
    std::vector<QwenNvfp4Block64> blocks;
    float weight_global_factor = 0.0f;
    float input_global_scale = 0.0f;
};

// Which checkpoint tensors the text runtime actually reads. Official Qwen3.8
// checkpoints bundle a vision tower in the same shards, so a complete audit has
// to account for every index entry as either mapped, deliberately ignored
// vision weights, or unexpected.
struct QwenCoverage {
    size_t index_tensors = 0;
    size_t mapped_tensors = 0;
    size_t visual_tensors = 0;
    size_t unexpected_tensors = 0;
    // Full (unsharded) bytes of every mapped tensor, i.e. the text share of the
    // checkpoint. Independent of TP world size.
    uint64_t checkpoint_text_bytes = 0;
    // Local resident bytes split by whether a rank holds a full copy or a shard.
    // Replicated bytes are present on every rank, so per-rank totals must not be
    // expected to sum to checkpoint_text_bytes.
    uint64_t replicated_local_bytes = 0;
    uint64_t sharded_local_bytes = 0;
    std::vector<std::string> unexpected_examples;
};

struct QwenLinearKindCounts {
    uint64_t dense_f16 = 0;
    uint64_t fp8_block128 = 0;
    uint64_t fp8_channel = 0;
    uint64_t nvfp4_group16 = 0;
    uint64_t ptq1_0 = 0;
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

    float* f32_data();
    const float* f32_data() const;
    uint16_t* f16_data();
    const uint16_t* f16_data() const;
    uint8_t* fp8_data();
    const uint8_t* fp8_data() const;
    int8_t* int8_data();
    const int8_t* int8_data() const;
    // Raw byte storage for packed caches whose slot mixes several element types,
    // so they cannot claim a single arithmetic dtype. Accepts I8 only.
    uint8_t* byte_data();
    const uint8_t* byte_data() const;
    // Packed NVFP4 block records, which are U8-typed rather than I8.
    uint8_t* u8_data();
    const uint8_t* u8_data() const;
};

struct QwenLinearRef {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    // The canonical name, kept so that a consumer can ask the checkpoint's own
    // declarations about this weight rather than matching on its place in the
    // model.
    std::string name;
    std::vector<uint64_t> logical_full_shape;
    std::vector<uint64_t> logical_local_shape;
    QwenShardRule rule = QwenShardRule::Replicated;
    int shard_dim = -1;
    QwenTensorRef weight;
    QwenTensorRef scale;
    QwenTensorRef weight_global_scale;
    QwenTensorRef input_global_scale;
    // The activation transform this weight's input goes through, as the
    // container declared it. A weight that was folded is in a rotated frame, so
    // its input has to be rotated into that frame; a weight that was not takes
    // the activation as it is. Both false for every checkpoint that declares no
    // transform, which is all of them but the ternary one.
    //
    // The transform is the whole of the activation-side story. A checkpoint
    // whose value axis arrives in a different head order is normalized on the
    // way in instead, by `QwenCheckpointSource::row_order`, so no reorder is
    // declared here.
    bool input_rotated = false;
    bool has_scale = false;
    bool has_weight_global_scale = false;
    bool has_input_global_scale = false;
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

struct QwenMtpWeights {
    QwenTensorRef pre_fc_norm_embedding;
    QwenTensorRef pre_fc_norm_hidden;
    QwenLinearRef fc;
    QwenLayerWeights layer;
    QwenTensorRef norm;
    bool found = false;
};

class QwenWeightMap {
public:
    // The checkpoint as a source: an HF directory, or a single-file GGUF. The
    // shape table, the sharding rules and the coverage accounting below are the
    // same for both, because they are statements about the architecture rather
    // than about a container.
    QwenWeightMap(const QwenCheckpointSource& source, const QwenConfig& config,
                  int tp_world = 1, int tp_rank = 0);
    // The HF directory, the spelling every caller written before the source
    // existed uses. Wraps the index rather than duplicating the table.
    QwenWeightMap(const SafeTensorsIndex& index, const QwenConfig& config,
                  int tp_world = 1, int tp_rank = 0);

    const QwenTensorRef& embed_tokens() const { return embed_tokens_; }
    const QwenTensorRef& final_norm() const { return final_norm_; }
    const QwenLinearRef& lm_head() const { return lm_head_; }
    const std::vector<QwenLayerWeights>& layers() const { return layers_; }
    const QwenMtpWeights& mtp() const { return mtp_; }
    const QwenConfig& config() const { return config_; }
    int tp_world() const { return tp_world_; }
    int tp_rank() const { return tp_rank_; }

    uint64_t local_weight_bytes() const { return local_weight_bytes_; }
    uint64_t local_scale_bytes() const { return local_scale_bytes_; }
    uint64_t host_global_metadata_bytes() const {
        return host_global_metadata_bytes_;
    }
    // Counts every linear descriptor present in the checkpoint map, including
    // optional MTP weights whether or not the runtime enables MTP residency.
    const QwenLinearKindCounts& checkpoint_linear_kind_counts() const {
        return checkpoint_linear_kind_counts_;
    }
    size_t tensor_count() const { return tensor_count_; }

    // Classify every checkpoint index entry against what this map claims.
    QwenCoverage coverage() const;
    // Throw unless every index entry is either mapped or a vision tensor.
    void require_full_coverage() const;

private:
    // The shape table both readers share, run once from whichever constructor
    // was used. A member function rather than a delegating constructor because
    // the safetensors spelling has to allocate its wrapper first, and a
    // constructor cannot both fill a member and delegate.
    void build();

    QwenTensorRef require_tensor(const std::string& name,
                                 SafeDType dtype,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule = QwenShardRule::Replicated,
                                 int shard_dim = -1) const;
    QwenLinearRef require_linear(const std::string& name,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule,
                                 int shard_dim) const;
    void record(const QwenTensorRef& ref, bool scale);
    void record_linear(const QwenLinearRef& ref);
    void claim(const QwenTensorRef& ref);

    // Set only by the safetensors constructor, which has to wrap an index the
    // caller owns; `source_` refers to it.
    std::unique_ptr<QwenCheckpointSource> owned_source_;
    const QwenCheckpointSource& source_;
    QwenConfig config_;
    int tp_world_ = 1;
    int tp_rank_ = 0;
    QwenTensorRef embed_tokens_;
    QwenTensorRef final_norm_;
    QwenLinearRef lm_head_;
    std::vector<QwenLayerWeights> layers_;
    QwenMtpWeights mtp_;
    uint64_t local_weight_bytes_ = 0;
    uint64_t local_scale_bytes_ = 0;
    uint64_t host_global_metadata_bytes_ = 0;
    QwenLinearKindCounts checkpoint_linear_kind_counts_;
    size_t tensor_count_ = 0;
    std::set<std::string> claimed_tensors_;
    uint64_t checkpoint_text_bytes_ = 0;
    uint64_t replicated_local_bytes_ = 0;
    uint64_t sharded_local_bytes_ = 0;
};

const char* qwen_shard_rule_name(QwenShardRule rule);
const char* qwen_linear_kind_name(QwenLinearKind kind);
// True for vision-tower tensors bundled into an official multimodal checkpoint.
bool qwen_is_visual_tensor(const std::string& name);

// Qwen checkpoint tensors retain their source dtype for validation. Storage
// dtype is what the checkpoint holds; device dtype is what the backend keeps
// resident. This is the CUDA/SM75 policy: Turing has no native BF16 arithmetic,
// so every BF16 tensor -- whether it comes from the official BF16 checkpoint or
// from the BF16 scale metadata of an FP8 checkpoint -- is converted to IEEE FP16
// at the upload boundary, while FP8 and NVFP4 codes stay compressed. A native
// BF16 backend must supply its own policy here rather than inherit this one.
SafeDType qwen_device_dtype(SafeDType storage_dtype);
uint16_t qwen_bf16_to_fp16_bits(uint16_t bits);
// Round-to-nearest-even FP32 -> FP16, for the norms a GGUF stores in fp32 where
// the kernels want fp16 gamma.
uint16_t qwen_float_to_fp16_bits(float value);
void qwen_convert_bf16_to_fp16(const uint16_t* src, uint16_t* dst, size_t count);

// Materialize a local tensor from its mmap'd source shard. The output is ready
// for upload: BF16 storage is converted to FP16 and packed/row slices are copied
// without expanding FP8 weights. Both supported backends want FP16 here, for
// unrelated reasons; see qwen_device_dtype in core/qwen_weight_map.cpp.
QwenHostTensor qwen_materialize_host_tensor(const SafeTensorsIndex& index,
                                            const QwenTensorRef& ref);
// The same, out of a source rather than a directory. A GGUF stores its norms in
// fp32 with the (1 + gamma) convention folded in and its linears as ternary
// blocks, so the source decides the device dtype and the fold; see
// QwenCheckpointSource and qwen_device_dtype.
QwenHostTensor qwen_materialize_host_tensor(const QwenCheckpointSource& source,
                                            const QwenTensorRef& ref);

// True for the RMSNorm affine weights that the Qwen3.5 runtime applies as
// (1 + weight): the layer input/post norms, the final norm, and the per-head
// q/k norms. False for linear_attn.norm.weight, which the gated norm applies
// directly, and for everything that is not a norm weight.
bool qwen_is_one_plus_norm_gamma(const std::string& name);

// Backend policy hook applied between materialization and upload.
//
// On CUDA this is a no-op: the kernels carry the (1 + weight) convention in the
// arithmetic. On Ascend the normalization comes from aclnnRmsNorm, which applies
// gamma directly and rejects an FP32 gamma against FP16 activations, so the +1 is
// folded into the FP16 weight here instead. Doing it once at load time rather than
// as a per-layer fixup op keeps 64 layers x 3 norms off the hot path.
//
// Folding in FP16 loses precision relative to the CUDA path, which adds 1.0 in
// FP32 at use time. Gamma values sit near zero where FP16 has ~2^-24 resolution
// but 1+gamma sits near one where it has 2^-11, so the folded value is the FP16
// neighbour of the exact sum. See test_qwen_ascend_norm_gamma for the bound.
void qwen_apply_norm_gamma_policy(const QwenTensorRef& ref, QwenHostTensor& host);
// Rewrites the linear attention's depthwise convolution weight, which the
// checkpoint stores [channels, 1, kernel], into the tap-major [kernel, channels]
// layout the Ascend kernel loads contiguously. Applies to that one tensor only and
// is a no-op on CUDA, where the kernel reads the checkpoint layout directly.
void qwen_apply_conv_weight_layout_policy(const QwenTensorRef& ref,
                                         QwenHostTensor& host);
QwenNvfp4HostLinear qwen_materialize_nvfp4_host_linear(
    const QwenCheckpointSource& source, const QwenLinearRef& ref);
QwenDeviceTensor qwen_upload_tensor(const SafeTensorsIndex& index,
                                         const QwenTensorRef& ref,
                                         void* stream = nullptr);
QwenDeviceTensor qwen_upload_tensor(const QwenCheckpointSource& source,
                                         const QwenTensorRef& ref,
                                         void* stream = nullptr);
QwenDeviceTensor qwen_upload_nvfp4_linear_cuda(
    const QwenCheckpointSource& source, const QwenLinearRef& ref,
    float* weight_global_factor, float* input_global_scale,
    void* stream = nullptr);

}  // namespace pocket
