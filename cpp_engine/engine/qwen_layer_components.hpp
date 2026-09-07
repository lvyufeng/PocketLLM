#pragma once

#include "qwen_engine.hpp"

#include <cstdint>
#include <vector>

namespace pocket::qwen_components {

// Device-resident weights grouped by the operation that consumes them. These
// are runtime layer objects, not checkpoint schema: loading stays in
// qwen_engine.cpp, while the forward components below own execution.
struct DeviceLinear {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    std::vector<uint64_t> logical_shape;
    QwenDeviceTensor weight;
    QwenDeviceTensor scale;
    // Optional dense FP16 expansion used only by batched target verification.
    // The raw FP8 matrix remains canonical and serves decode/prefill.
    QwenDeviceTensor verify_weight;
    // NVFP4 tensor-level scaling. The factor is reciprocal(weight_global_scale);
    // input_global_scale is calibration metadata the SM75 path does not apply.
    float weight_global_factor = 1.0f;
    float input_global_scale = 1.0f;
};

struct DeviceLinearAttention {
    DeviceLinear qkv;
    DeviceLinear z;
    DeviceLinear out;
    DeviceLinear a;
    DeviceLinear b;
    // Row-concat of a and b; empty when the two cannot be fused.
    DeviceLinear ab;
    QwenDeviceTensor conv;
    QwenDeviceTensor a_log;
    QwenDeviceTensor dt_bias;
    QwenDeviceTensor norm;
    QwenDeviceTensor state;
    QwenDeviceTensor conv_tail;
};

struct DeviceFullAttention {
    DeviceLinear q;
    DeviceLinear k;
    DeviceLinear v;
    // Row-concat of K and V for batched verify/prefill. Decode keeps the
    // individual projections so its established one-row path is unchanged.
    DeviceLinear kv;
    DeviceLinear out;
    QwenDeviceTensor q_norm;
    QwenDeviceTensor k_norm;
    QwenDeviceTensor k_cache;
    QwenDeviceTensor v_cache;
    QwenDeviceTensor k_scale;
    QwenDeviceTensor v_scale;
    // TurboQuant K8V4 combined cache: 196-byte slots with FP8 E5M2 key + 4-bit
    // value + FP16 metadata. Only allocated for TurboQuantK8V4.
    QwenDeviceTensor turboquant_cache;
};

struct DeviceLayer {
    QwenDeviceTensor input_norm;
    QwenDeviceTensor post_norm;
    DeviceLinearAttention linear;
    DeviceFullAttention full;
    DeviceLinear gate;
    DeviceLinear up;
    DeviceLinear down;
};

enum class NvFp4Mode {
    Auto,
    Dp4a,
    Wmma,
    Reference,
    Invalid,
};

enum class OptionalSwitch {
    Auto,
    Disabled,
    Enabled,
};

// Environment-selected layer policy, captured once when the engine is built.
// Forward used to call getenv() in every layer; besides the avoidable host work,
// that let one process mutate the kernel sequence midway through a request.
struct LayerExecutionConfig {
    NvFp4Mode nvfp4_mode = NvFp4Mode::Auto;
    bool nvfp4_wide_n64 = true;
    int nvfp4_wide_n64_min_rows = 128;
    bool nvfp4_fused_swiglu = false;
    bool nvfp4_shared_q8_swiglu = true;
    bool verify_small_fp16_cublas = true;
    bool gated_delta_flashqla = true;
    bool fuse_qkvz_decode = false;
    bool fuse_ab_projection = true;
    bool gated_delta_prenormalize = true;
    bool gated_delta_shared_state = false;
    bool fuse_full_qkv_decode = false;
    bool gqa_optimized = true;
    bool gqa_verify_cublas_qk = false;
    OptionalSwitch gqa_verify_split = OptionalSwitch::Auto;
    int gqa_verify_splits = 0;
    bool fuse_attention_residual_norm = true;
    int comm_overlap_slices = 4;

    bool use_nvfp4_wide_n64(int rows) const {
        return nvfp4_wide_n64 && rows >= nvfp4_wide_n64_min_rows;
    }
};

// Concrete components rather than a layer-level vtable. Their templated runtime
// parameter keeps the engine's workspace/communication services inlineable in
// the 64-layer loop while giving each model operation an explicit owner.
struct Linear {
    template <typename Runtime>
    void forward(Runtime& runtime, const DeviceLinear& linear,
                 const uint16_t* input, uint16_t* output, int rows,
                 const char* site = "other") const;
};

struct RMSNorm {
    template <typename Runtime>
    void forward(Runtime& runtime, const QwenDeviceTensor& gamma,
                 const uint16_t* input, uint16_t* output, int rows,
                 int columns) const;
};

struct GatedDeltaNetAttention {
    template <typename Runtime>
    void forward(Runtime& runtime, DeviceLayer& layer, const uint16_t* hidden,
                 uint16_t* output, int rows, int position_offset,
                 int slot_id) const;
};

struct GqaAttention {
    template <typename Runtime>
    void forward(Runtime& runtime, DeviceLayer& layer, const uint16_t* hidden,
                 uint16_t* output, int rows, int position_offset,
                 int slot_id) const;
};

struct FusedGateUpSwiGLU {
    struct Workspace {
        QwenDeviceTensor* gate = nullptr;
        QwenDeviceTensor* up = nullptr;
        QwenDeviceTensor* output = nullptr;
        bool fused_decode = false;
        bool fused_small_batch = false;
        bool fused_nvfp4 = false;
        bool shared_nvfp4 = false;
    };

    // Reserve before attention runs so workspace slot order stays identical to
    // the monolithic layer implementation. Forward then owns kernel selection.
    template <typename Runtime>
    Workspace reserve(Runtime& runtime, DeviceLayer& layer, int rows,
                      int hidden_size) const;

    template <typename Runtime>
    void forward(Runtime& runtime, DeviceLayer& layer, const uint16_t* input,
                 Workspace& workspace, int rows, int hidden_size) const;
};

struct DecoderLayer {
    template <typename Runtime>
    void forward(Runtime& runtime, DeviceLayer& layer, const uint16_t* hidden,
                 uint16_t* output, int rows, int position_offset,
                 int slot_id) const;
};

}  // namespace pocket::qwen_components
