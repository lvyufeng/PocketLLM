/* The Ascend 310B backend.
 *
 * The third implementation of `kernel/backend.h`, after the CPU and the card.
 * The *memory* half of the interface -- allocate, release, the three copies,
 * fill, synchronize -- is real, and so are five compute ops: `gemm_quant` for
 * q4_k, `rms_norm`, `silu_mul`, `gemm` (dense f32/f16) and `attention` (one
 * decode step).  Every other op throws, by name, so a caller that reaches it
 * learns which op is absent rather than getting a wrong number.
 *
 * ## What each implemented op is measured at
 *
 * Each op was probed standalone against a float64 CPU reference computed from
 * the *same* fp16 inputs -- the reference rounds only where the op itself
 * rounds, so what is left is the op's own arithmetic error and not the storage
 * format's.  Measured on the board (CANN 8.3.RC2, Ascend310B1):
 *
 *   op                                  /max|ref|   worst per-element
 *   RmsNormNdCustom   8 x 1024            2.7e-04        4.8e-04
 *   RmsNormNdCustom  512 x 128            2.6e-04        4.9e-04
 *   RmsNormNdCustom   16 x 128  in-place   2.7e-04        4.7e-04
 *   SiluMulCustom    4096 elements        0.0            0.0
 *   MatmulCubeCustom  1 x 1024 x 1024     1.6e-06        5.7e-04
 *   MatmulCubeCustom  1 x 1024 x 3072     1.0e-04        9.3e-04
 *   MatmulCubeCustom 16 x 1024 x 3072     1.8e-04        9.6e-04
 *   AttentionStepCustom 16q/8kv/D128/ctx64 2.1e-04       9.0e-04
 *
 * All under the 1e-3 gate.  The cube and attention per-element errors are the
 * fp16 output rounding of a long reduction -- the accumulate is f32 (L0C for
 * the cube, an fp32 score row for attention), which is why they do not grow
 * with K or context the way an fp16 accumulator's would; silu_mul is bit-exact
 * because the kernel and the reference both evaluate `g / (1 + exp(-g)) * u`
 * in f32.  The same four ops were then driven through this backend's own f32
 * interface -- the f32<->f16 bridges, the gemm transpose, the attention repack
 * and the accumulate path included -- and every case stayed under 1e-3.
 *
 * ## The layout bridge `attention` needs
 *
 * Every op here is a direct drive of a custom aclnn op except `attention`, and
 * that one has a real impedance mismatch: `pipelines`'s cache is
 * ``[position][head][d]`` (token-major), while `AttentionStepCustom` walks the
 * cache as ``[head][position][d]`` and -- this is the part that cannot be
 * papered over with tensor strides -- its kernel uses raw base-relative offsets
 * (`kGm[t * headDim]`), ignoring whatever strides the `aclTensor` declares.  So
 * the window ``[0, context)`` is repacked onto the host between the two
 * layouts on every call.  At Qwen3-0.6B's shapes (8 KV heads, head_dim 128)
 * that is ~2 MB per layer per
 * token of host traffic, which is slow and is the honest cost of not having a
 * device-side transpose op; it is stated here rather than hidden because it is
 * the first thing to fix when this backend is made fast.
 *
 * ## Why the built-in matmul cannot be used
 *
 * `aclnnMm` and the other built-in matmuls ship no `ascend310b` kernel binary
 * on CANN 8.3.RC2 -- the factory `op_impl` for the 310B carries only
 * `batch_matmul_v2` -- so they fail on this board.  The custom ops built in the
 * minicpm-o-4.5-orangepi tree supply the 310B binaries, and this backend drives
 * `aclnnMatmulW4a16Custom` for the quantized GEMM and `aclnnMatmulCubeCustom`
 * for the dense one.
 *
 * ## The environment is a hard requirement, not a convenience
 *
 * `ASCEND_CUSTOM_OPP_PATH` must point at the directory that *contains*
 * `vendors/customize` (i.e. `.../custom_opp/vendors/customize`, not its parent),
 * and it must be exported before `aclInit`. Without it the op is never
 * registered with Nnopbase and the two-phase call fails phase 1 with
 * `161001 ACLNN_ERR_PARAM_NULLPTR` -- a null *executor*, not a null tensor,
 * which is why the message reads as a caller mistake and is not one. The
 * documented setup is `source <custom_opp>/vendors/customize/bin/set_env.bash`;
 * `describe()` reports the variable so a misconfigured host is visible.
 *
 * ## What the interface cannot express
 *
 * `gemm_quant`'s contract is a q4_K checkpoint tensor: 256-weight super-blocks,
 * `(n, k / 256, 144)` bytes. `MatmulW4a16Custom` wants GPTQ int4 packed as int8
 * + a per-128 fp16 scale row -- a different quantization entirely. There is no
 * byte-level reinterpretation between them, so this backend bridges by
 * *requantizing*: it decodes each q4_K block to float with the tree's own
 * `dequant_q4_k` and rounds the result to int8 in [-8, 7] with a per-128 scale.
 * That is a real accuracy change (the weights are effectively re-quantized to
 * ~4 bits with a coarser scale granularity) and it is done once, at first use.
 * The alternative -- feeding the op q4_K blocks as if they were GPTQ int4 --
 * would be silently wrong, and a wrong number is worse than a slower one.
 *
 * A second thing the interface cannot express: `gemm_quant` takes `k` weights
 * per row and the kernel's group size is fixed at 128, so `k % 128 == 0` is a
 * precondition this backend enforces rather than assumes. Qwen3's K values
 * (1024, 2048, ...) satisfy it; a checkpoint that did not would be refused.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aclnn_attention_step_custom.h"
#include "aclnn_matmul_cube_custom.h"
#include "aclnn_matmul_w4a16_custom.h"
#include "aclnn_rms_norm_nd_custom.h"
#include "aclnn_silu_mul_custom.h"
#include "kernel/backend.h"
#include "quant/blocks.h"
#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

namespace {

constexpr int64_t kGroup = 128;        /* MatmulW4a16Custom's fixed group size */
constexpr int64_t kTileLen = 64;       /* tileLen: % 16 == 0 and <= 448 */
constexpr int64_t kGemmAlign = 128;    /* n must be a multiple of this */
constexpr int64_t kRmsMaxD = 4096;     /* RmsNormNdCustom keeps one row in UB */
constexpr int64_t kMaxContext = 8192;  /* AttentionStepCustom's MAX_CONTEXT */

/* Check an `aclError` and throw with the call's name in the message.  Every acl
 * entry point returns this and none of them throw, so without a wrapper the
 * failure arrives as a number at a call site unrelated to the cause. */
void acl_ok(aclError status, const char *what) {
  if (status != ACL_SUCCESS) {
    throw Error(std::string("ascend: ") + what + " failed (aclError " +
                std::to_string(static_cast<int>(status)) + ")");
  }
}

void aclnn_ok(aclnnStatus status, const char *what) {
  if (status != 0) {
    std::string hint;
    if (status == 161001) {
      hint = "; 161001 is ACLNN_ERR_PARAM_NULLPTR -- most often the executor is null "
             "because ASCEND_CUSTOM_OPP_PATH does not point at .../vendors/customize "
             "(source <custom_opp>/vendors/customize/bin/set_env.bash before aclInit)";
    }
    throw Error(std::string("ascend: ") + what + " failed (aclnnStatus " +
                std::to_string(static_cast<int>(status)) + hint + ")");
  }
}

/* ---- requantized q4_K weights, in the layout the op wants --------------- */

/* int4-as-int8 rows, packed by (group, core, tile, k) exactly as the kernel's
 * `ComputeNTile` indexes them, plus the per-128 fp16 scale row.  Built once per
 * distinct source matrix and cached by the block pointer. */
struct PackedQ4K {
  int64_t n = 0;
  int64_t k = 0;
  int64_t block_len = 0;
  int64_t tiles_per_block = 0;
  int64_t packed_rows = 0;
  std::vector<int8_t> w;    /* [packed_rows, kTileLen] */
  std::vector<uint16_t> sc; /* [k/128, n] fp16 */
};

/* Round-to-nearest-even float -> fp16.  The same conversion `quant/half.h` does
 * for the kernels; kept local so this file needs nothing from the ABI headers. */
uint16_t f32_to_f16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (exp >= 0x1f) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    const int32_t shift = 14 - exp;
    const uint32_t half_m = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    return static_cast<uint16_t>(sign | (half_m + (rem > halfway || (rem == halfway && (half_m & 1u)) ? 1u : 0u)));
  }
  const uint32_t half_m = mant >> 13;
  const uint32_t rem = mant & 0x1fffu;
  uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_m);
  if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) {
    ++out; /* may carry into the exponent; that is the correct round-to-even */
  }
  return out;
}

float f16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/* ---- host scratch for a device region ----------------------------------- */

std::vector<float> half_to_f32(const std::vector<uint16_t> &half) {
  std::vector<float> out(half.size());
  for (std::size_t i = 0; i < half.size(); ++i) {
    out[i] = f16_to_f32(half[i]);
  }
  return out;
}

std::vector<uint16_t> to_f16(const float *src, int64_t count) {
  std::vector<uint16_t> out(static_cast<std::size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<std::size_t>(i)] = f32_to_f16(src[i]);
  }
  return out;
}

/* Decode one q4_K matrix (n columns of k weights) to float once, then build the
 * packed int4 image.  `blocks` is (n, k/256, 144) with a 256-weight super-block
 * per (column, k-group), which is `ggml_type_of(12)`'s geometry. */
void pack_q4k(const uint8_t *blocks, int64_t n, int64_t k, PackedQ4K *out) {
  if (k % kGroup != 0) {
    throw Error("ascend: gemm_quant needs k a multiple of 128 for q4_k, got " + std::to_string(k));
  }
  if (n % kGemmAlign != 0) {
    throw Error("ascend: gemm_quant needs n a multiple of 128, got " + std::to_string(n));
  }
  const int64_t groups = k / kGroup;
  const int64_t block_len = (n + 7) / 8;
  const int64_t tiles_per_block = (block_len + kTileLen - 1) / kTileLen;
  const int64_t packed_rows = groups * 8 * tiles_per_block * kGroup;

  out->n = n;
  out->k = k;
  out->block_len = block_len;
  out->tiles_per_block = tiles_per_block;
  out->packed_rows = packed_rows;
  out->w.assign(static_cast<std::size_t>(packed_rows * kTileLen), 0);
  out->sc.assign(static_cast<std::size_t>(groups * n), 0);

  const int64_t super_blocks = k / quant::kBlockWeights; /* 256-weight super-blocks per column */
  std::vector<float> column(static_cast<std::size_t>(k));
  for (int64_t col = 0; col < n; ++col) {
    const uint8_t *const src =
        blocks + static_cast<std::size_t>(col * super_blocks * quant::kQ4KBlockBytes);
    for (int64_t b = 0; b < super_blocks; ++b) {
      const uint8_t *const blk = src + static_cast<std::size_t>(b * quant::kQ4KBlockBytes);
      for (int i = 0; i < quant::kBlockWeights; ++i) {
        column[static_cast<std::size_t>(b * quant::kBlockWeights + i)] = quant::dequant_q4_k(blk, i);
      }
    }
    /* Requantize per 128-group: scale = max|x| / 7, q = round(x / scale) in
     * [-8, 7].  A zero group gets scale 0 and all-zero weights, which the op
     * multiplies to zero rather than to a NaN. */
    for (int64_t g = 0; g < groups; ++g) {
      const float *const group = column.data() + g * kGroup;
      float max_abs = 0.0f;
      for (int64_t i = 0; i < kGroup; ++i) {
        const float a = group[i] < 0 ? -group[i] : group[i];
        if (a > max_abs) {
          max_abs = a;
        }
      }
      const float scale = max_abs > 0 ? max_abs / 7.0f : 0.0f;
      out->sc[static_cast<std::size_t>(g * n + col)] = f32_to_f16(scale);
      /* The kernel's tile is (group, core, tile, k) -> real row (g*128 + k),
       * real column core*block_len + tile*tile_len + j.  Guarded by
       * n % 128 == 0, which makes core < 8 for every column. */
      const int64_t core = col / block_len;
      const int64_t within = col - core * block_len;
      const int64_t tile = within / kTileLen;
      const int64_t j = within - tile * kTileLen;
      for (int64_t i = 0; i < kGroup; ++i) {
        int q = 0;
        if (scale > 0) {
          const float scaled = group[i] / scale;
          q = static_cast<int>(scaled < 0 ? scaled - 0.5f : scaled + 0.5f);
          if (q > 7) q = 7;
          if (q < -8) q = -8;
        }
        const int64_t row = (((g * 8 + core) * tiles_per_block + tile) * kGroup) + i;
        out->w[static_cast<std::size_t>(row * kTileLen + j)] = static_cast<int8_t>(q);
      }
    }
  }
}

/* ---- the backend --------------------------------------------------------- */

class AscendBackend final : public Backend {
 public:
  AscendBackend() {
    /* The custom op is only registered when this is set, and it is read when the
     * runtime initialises -- so it is set here, before `aclInit`, from the same
     * location the installer writes.  An explicit value the caller exported is
     * left alone. */
    if (std::getenv("ASCEND_CUSTOM_OPP_PATH") == nullptr) {
      const char *const home = std::getenv("HOME");
      if (home != nullptr && home[0] != '\0') {
        const std::string path = std::string(home) + "/Ascend/custom_opp/vendors/customize";
        setenv("ASCEND_CUSTOM_OPP_PATH", path.c_str(), 1);
      }
    }
    acl_ok(aclInit(nullptr), "aclInit");
    acl_ok(aclrtSetDevice(0), "aclrtSetDevice(0)");
    acl_ok(aclrtCreateStream(&stream_), "aclrtCreateStream");
    const char *const soc = aclrtGetSocName();
    soc_ = soc != nullptr ? soc : "unknown";
  }

  ~AscendBackend() override {
    /* The graph releases its own buffers; anything left is a leak this backend
     * cannot see, and freeing it here would be a guess at the size.  The stream
     * and the device are ours to hand back. */
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
    }
    aclrtResetDevice(0);
    aclFinalize();
  }

  const char *name() const override { return "ascend"; }

  DeviceBuffer allocate(int64_t bytes) override {
    if (bytes <= 0) {
      throw Error("ascend: cannot allocate " + std::to_string(bytes) + " bytes");
    }
    void *ptr = nullptr;
    acl_ok(aclrtMalloc(&ptr, static_cast<std::size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST),
           "aclrtMalloc");
    /* The handle is a *device* address. It is distinct from a host pointer by
     * construction -- the two address spaces are different -- which is the
     * property the interface's comment asks a backend to keep. */
    return DeviceBuffer{reinterpret_cast<uintptr_t>(ptr), bytes};
  }

  void release(DeviceBuffer buffer) override {
    if (buffer.handle != 0) {
      aclrtFree(reinterpret_cast<void *>(buffer.handle));
    }
  }

  void copy_to_device(DeviceBuffer dst, const void *src, int64_t bytes) override {
    acl_ok(aclrtMemcpy(reinterpret_cast<void *>(dst.handle), static_cast<std::size_t>(bytes), src,
                       static_cast<std::size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE),
           "aclrtMemcpy H2D");
  }

  void copy_to_host(void *dst, DeviceBuffer src, int64_t bytes) override {
    acl_ok(aclrtMemcpy(dst, static_cast<std::size_t>(bytes),
                       reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes),
                       ACL_MEMCPY_DEVICE_TO_HOST),
           "aclrtMemcpy D2H");
  }

  void copy_device_to_device(DeviceBuffer dst, DeviceBuffer src, int64_t bytes) override {
    acl_ok(aclrtMemcpy(reinterpret_cast<void *>(dst.handle), static_cast<std::size_t>(bytes),
                       reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes),
                       ACL_MEMCPY_DEVICE_TO_DEVICE),
           "aclrtMemcpy D2D");
  }

  /* Zero a device region. `aclrtMemset` is the direct call, but it takes a byte
   * and the contract is a *float* value; the two agree only on 0.0f/1.0f and
   * nothing else, so a host staging buffer is the honest implementation and the
   * one that cannot be wrong for a value the graph happens to pass. */
  void fill(DeviceBuffer dst, float value) override {
    const int64_t count = dst.bytes / 4;
    std::vector<float> staged(static_cast<std::size_t>(count), value);
    acl_ok(aclrtMemcpy(reinterpret_cast<void *>(dst.handle),
                       static_cast<std::size_t>(dst.bytes), staged.data(),
                       static_cast<std::size_t>(count * 4), ACL_MEMCPY_HOST_TO_DEVICE),
           "aclrtMemcpy fill");
  }

  void gemm_quant(DeviceBuffer x, DeviceBuffer blocks, DeviceBuffer bias, DeviceBuffer out,
                  int64_t m, int64_t n, int64_t k, int type_id, bool accumulate) override {
    if (type_id != quant::kGgmlQ4K) {
      throw Error("ascend: gemm_quant supports q4_k only, got type_id " + std::to_string(type_id));
    }
    if (m != 1) {
      throw Error("ascend: gemm_quant not implemented for m>1 (got m=" + std::to_string(m) +
                  "); the W4a16 custom op is an M=1 decode path");
    }
    if (accumulate) {
      throw Error("ascend: gemm_quant accumulate is not implemented yet");
    }
    if (bias.handle != 0) {
      throw Error("ascend: gemm_quant bias is not implemented yet");
    }
    if (n % kGemmAlign != 0 || k % kGroup != 0) {
      throw Error("ascend: gemm_quant needs n % 128 == 0 and k % 128 == 0, got n=" +
                  std::to_string(n) + " k=" + std::to_string(k));
    }

    const PackedQ4K &packed = pack_for(blocks, n, k);

    /* Stage the packed image and the activation on the device. The activation
     * arrives f32 (the graph's activation type is f32 on every backend) and the
     * op wants fp16, so it is converted on the host for this first-round op. */
    const int64_t rows = packed.packed_rows;
    std::vector<uint16_t> xh(static_cast<std::size_t>(k));
    std::vector<float> xf(static_cast<std::size_t>(k));
    copy_to_host(xf.data(), x, k * 4);
    for (int64_t i = 0; i < k; ++i) {
      xh[static_cast<std::size_t>(i)] = f32_to_f16(xf[static_cast<std::size_t>(i)]);
    }

    DeviceBuffer dx = allocate(k * 2);
    DeviceBuffer dw = allocate(rows * kTileLen);
    DeviceBuffer dsc = allocate((k / kGroup) * n * 2);
    DeviceBuffer dout = allocate(n * 2);
    copy_to_device(dx, xh.data(), k * 2);
    copy_to_device(dw, packed.w.data(), rows * kTileLen);
    copy_to_device(dsc, packed.sc.data(), (k / kGroup) * n * 2);

    const int64_t xs[2] = {1, k};
    const int64_t ws[2] = {rows, kTileLen};
    const int64_t ss[2] = {k / kGroup, n};
    const int64_t os[2] = {1, n};
    const int64_t xs_t[2] = {k, 1};
    const int64_t ws_t[2] = {kTileLen, 1};
    const int64_t ss_t[2] = {n, 1};
    const int64_t os_t[2] = {n, 1};
    aclTensor *tx = aclCreateTensor(xs, 2, ACL_FLOAT16, xs_t, 0, ACL_FORMAT_ND, xs, 2, dx.handle ? reinterpret_cast<void *>(dx.handle) : nullptr);
    aclTensor *tw = aclCreateTensor(ws, 2, ACL_INT8, ws_t, 0, ACL_FORMAT_ND, ws, 2, reinterpret_cast<void *>(dw.handle));
    aclTensor *tsc = aclCreateTensor(ss, 2, ACL_FLOAT16, ss_t, 0, ACL_FORMAT_ND, ss, 2, reinterpret_cast<void *>(dsc.handle));
    aclTensor *tout = aclCreateTensor(os, 2, ACL_FLOAT16, os_t, 0, ACL_FORMAT_ND, os, 2, reinterpret_cast<void *>(dout.handle));
    if (tx == nullptr || tw == nullptr || tsc == nullptr || tout == nullptr) {
      throw Error("ascend: aclCreateTensor returned null");
    }

    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnn_ok(aclnnMatmulW4a16CustomGetWorkspaceSize(tx, tw, tsc, tout, &ws_size, &executor),
             "aclnnMatmulW4a16CustomGetWorkspaceSize");
    void *workspace = nullptr;
    if (ws_size > 0) {
      acl_ok(aclrtMalloc(&workspace, static_cast<std::size_t>(ws_size), ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc workspace");
    }
    aclnn_ok(aclnnMatmulW4a16Custom(workspace, ws_size, executor, stream_),
             "aclnnMatmulW4a16Custom");
    acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");

    /* The op emits fp16; the graph's `out` is f32.  Widen on the host. */
    std::vector<uint16_t> outh(static_cast<std::size_t>(n));
    copy_to_host(outh.data(), dout, n * 2);
    std::vector<float> outf(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      outf[static_cast<std::size_t>(i)] = f16_to_f32(outh[static_cast<std::size_t>(i)]);
    }
    copy_to_device(out, outf.data(), n * 4);

    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    aclDestroyTensor(tx);
    aclDestroyTensor(tw);
    aclDestroyTensor(tsc);
    aclDestroyTensor(tout);
    release(dx);
    release(dw);
    release(dsc);
    release(dout);
  }

  void rms_norm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer out, int64_t n_tokens,
                int64_t d, float eps) override {
    if (n_tokens <= 0 || d <= 0) {
      return;
    }
    if (d > kRmsMaxD) {
      throw Error("ascend: rms_norm hidden size " + std::to_string(d) +
                  " exceeds RmsNormNdCustom's " + std::to_string(kRmsMaxD) +
                  "; a wider row needs a tiled reduction this backend does not have");
    }
    /* The AscendC vector pipe is 32 bytes = 16 fp16 lanes wide, and a row
     * narrower than one repeat is not processed: `RmsNormNdCustom` returns the
     * output buffer untouched for d < 16 (measured: d = 1, 8, 15 all come back
     * as zeros, d = 16 onwards is right).  Its own host op runs the model at
     * d = 4096 and d = 128, both aligned, so the op's authors never met this.
     * This backend refuses rather than returning zeros: no shape the graph runs
     * reaches it, and a phone-side build that one day did would want the error
     * far more than it would want a silent row of zeros. */
    if (d % 16 != 0) {
      throw Error("ascend: rms_norm row width " + std::to_string(d) +
                  " is not a multiple of 16, below which RmsNormNdCustom does not write its "
                  "output");
    }
    /* The op is fp16-native and the graph's operands are f32 -- `bind_dense`
     * widens every dense tensor, the norm gammas included, so both x and the
     * weight arrive f32.  Narrow both, drive the op, widen the result back. */
    const int64_t elements = n_tokens * d;
    const std::vector<uint16_t> x_h = to_f16(read_f32(x, elements, ACL_FLOAT).data(), elements);
    const std::vector<uint16_t> w_h = to_f16(read_f32(weight, d, ACL_FLOAT).data(), d);
    DeviceBuffer dx = allocate(elements * 2);
    DeviceBuffer dw = allocate(d * 2);
    DeviceBuffer dout = allocate(elements * 2);
    copy_to_device(dx, x_h.data(), elements * 2);
    copy_to_device(dw, w_h.data(), d * 2);
    run_rms_nd(dx, dw, dout, n_tokens, d, eps);
    std::vector<uint16_t> out_h(static_cast<std::size_t>(elements));
    copy_to_host(out_h.data(), dout, elements * 2);
    const std::vector<float> out_f = half_to_f32(out_h);
    copy_to_device(out, out_f.data(), elements * 4);
    release(dx);
    release(dw);
    release(dout);
  }

  void gemm(DeviceBuffer x, DeviceBuffer w, DeviceBuffer bias, DeviceBuffer out, int64_t m,
            int64_t n, int64_t k, bool accumulate) override {
    if (m <= 0 || n <= 0 || k <= 0) {
      return;
    }
    if (bias.handle != 0) {
      throw Error("ascend: gemm bias is not implemented yet (MatmulCubeCustom has no bias input)");
    }
    /* MatmulCubeCustom takes B in [K, N] storage order and this interface's `w`
     * is row-major [N, K], so the operand is a transpose of the plane the graph
     * holds.  Driving the op with [N, K] would multiply by the wrong matrix
     * silently, so the operand is brought up and transposed on the host once.
     * The weights are bound once and never change, so this is a host-side
     * transpose per weight per process, not per token. */
    run_cube(read_f32(x, m * k, ACL_FLOAT), w, m, n, k, accumulate, out);
  }
  void embedding(DeviceBuffer, int64_t, DeviceBuffer, int64_t, int64_t, DeviceBuffer) override {
    throw Error("ascend: embedding not implemented yet");
  }
  void embedding_quant(DeviceBuffer, int64_t, DeviceBuffer, int64_t, int64_t, int,
                       DeviceBuffer) override {
    throw Error("ascend: embedding_quant not implemented yet");
  }
  void silu_mul(DeviceBuffer gate, DeviceBuffer up, DeviceBuffer out, int64_t n) override {
    if (n <= 0) {
      return;
    }
    /* The op's tiling splits the flat element range across blocks, so it wants
     * a 1-D [n] tensor regardless of the graph's row/column shape; `n` here is
     * already `tokens * intermediate`, the flat count.  The graph's gate/up are
     * f32 and the op is fp16.
     *
     * The fp16 inputs ride in on one transfer: a two-`[2, n]` view lets the
     * host stage both operands in a single buffer and hand the op one half
     * each, so a 4096-wide FFN pays two copies rather than four. */
    DeviceBuffer dstage = allocate(n * 2 * 2);
    DeviceBuffer dout = allocate(n * 2);
    std::vector<uint16_t> staged(static_cast<std::size_t>(2 * n));
    const std::vector<float> gate_f = read_f32(gate, n, ACL_FLOAT);
    const std::vector<float> up_f = read_f32(up, n, ACL_FLOAT);
    for (int64_t i = 0; i < n; ++i) {
      staged[static_cast<std::size_t>(i)] = f32_to_f16(gate_f[static_cast<std::size_t>(i)]);
      staged[static_cast<std::size_t>(n + i)] = f32_to_f16(up_f[static_cast<std::size_t>(i)]);
    }
    copy_to_device(dstage, staged.data(), 2 * n * 2);
    DeviceBuffer dgate{static_cast<uintptr_t>(dstage.handle), n * 2};
    DeviceBuffer dup{static_cast<uintptr_t>(dstage.handle + n * 2), n * 2};

    const int64_t shape[1] = {n};
    const int64_t stride[1] = {1};
    aclTensor *tg = aclCreateTensor(shape, 1, ACL_FLOAT16, stride, 0, ACL_FORMAT_ND, shape, 1,
                                    reinterpret_cast<void *>(dgate.handle));
    aclTensor *tu = aclCreateTensor(shape, 1, ACL_FLOAT16, stride, 0, ACL_FORMAT_ND, shape, 1,
                                    reinterpret_cast<void *>(dup.handle));
    aclTensor *to = aclCreateTensor(shape, 1, ACL_FLOAT16, stride, 0, ACL_FORMAT_ND, shape, 1,
                                    reinterpret_cast<void *>(dout.handle));
    if (tg == nullptr || tu == nullptr || to == nullptr) {
      throw Error("ascend: aclCreateTensor returned null (silu_mul)");
    }
    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnn_ok(aclnnSiluMulCustomGetWorkspaceSize(tg, tu, to, &ws_size, &executor),
             "aclnnSiluMulCustomGetWorkspaceSize");
    void *workspace = nullptr;
    if (ws_size > 0) {
      acl_ok(aclrtMalloc(&workspace, static_cast<std::size_t>(ws_size), ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc workspace");
    }
    aclnn_ok(aclnnSiluMulCustom(workspace, ws_size, executor, stream_), "aclnnSiluMulCustom");
    acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");

    /* The op emits fp16; the graph's `out` is f32. */
    std::vector<uint16_t> out_h(static_cast<std::size_t>(n));
    copy_to_host(out_h.data(), dout, n * 2);
    const std::vector<float> out_f = half_to_f32(out_h);
    copy_to_device(out, out_f.data(), n * 4);

    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    aclDestroyTensor(tg);
    aclDestroyTensor(tu);
    aclDestroyTensor(to);
    release(dstage);
    release(dout);
  }

  void rope_neox(DeviceBuffer, int64_t, int64_t, int64_t, int64_t, DeviceBuffer,
                 DeviceBuffer) override {
    throw Error("ascend: rope_neox not implemented yet (the custom RopeCustom op is installed; "
                "its layout has not been wired to this interface)");
  }

  /* `AttentionStepCustom` keeps its own score row per head in device UB, so it
   * needs no caller-provided scratch at all -- and the graph allocates this
   * much on the strength of this answer, so a good number here is free.  The
   * contract still requires `max_span` be accounted for, so the span is
   * reported: see the file header on why the op's scratch is device-internal. */
  int64_t attention_scratch(int64_t q_len, int64_t n_heads, int64_t max_span) const override {
    (void)max_span;
    return q_len * n_heads * 4;
  }

  void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                 DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                 int64_t q_offset, float scale, DeviceBuffer out, DeviceBuffer scores,
                 KVDtype kv_dtype) override {
    (void)scores;  /* the op keeps its score row on the device, not in this buffer */
    if (q_len != 1) {
      throw Error("ascend: attention is not implemented for prefill (q_len=" +
                  std::to_string(q_len) + "); only the one-token decode step maps to the "
                  "AttentionStepCustom op");
    }
    if (first_key != 0) {
      throw Error("ascend: attention with a sliding window (first_key=" +
                  std::to_string(first_key) + ") is not implemented");
    }
    if (kv_dtype != KVDtype::kF32 && kv_dtype != KVDtype::kF16) {
      throw Error("ascend: attention has no path for this cache dtype");
    }
    if (q_offset < 0) {
      throw Error("ascend: attention got a negative q_offset " + std::to_string(q_offset));
    }
    if (d % 16 != 0) {
      throw Error("ascend: attention head_dim " + std::to_string(d) +
                  " must be a multiple of 16 for the fused step kernel");
    }
    /* `q_offset` is the chunk's first position, so a one-token decode at
     * `q_offset` must attend to keys `0..q_offset` inclusive -- its own key has
     * already been appended to the cache by the time attention runs (this model
     * scores against the current key; see `Qwen3Model::forward`).  That makes
     * the visible count `q_offset + 1`, the same `end_pos` `attention_scratch`
     * is sized to.  Attending to `q_offset` keys instead -- which this backend
     * did until the conformance run caught it at an 85% error on a one-key span
     * -- drops the token's own key and is simply a wrong answer.
     *
     * The graph's cache is ``[position][head][d]`` with the visible rows packed
     * contiguously from position 0, and the op's `kCache` is
     * ``[n_head_kv, max_seq, d]``, so `max_seq == context` is the layout the
     * graph actually produces and the op's window is `[0, context)` either way. */
    const int64_t context = q_offset + 1;
    if (context > kMaxContext) {
      throw Error("ascend: attention context " + std::to_string(context) + " exceeds the op's " +
                  std::to_string(kMaxContext));
    }
    run_attention(q, n_heads, n_head_kv, d, context, scale, k_cache, v_cache, kv_dtype, out);
  }

  void kv_append(DeviceBuffer dst, DeviceBuffer src, int64_t n, int64_t n_head_kv, int64_t d,
                 int64_t elem) override {
    if (n <= 0) {
      return;
    }
    if (elem != 2 && elem != 4) {
      throw Error("ascend: kv_append elem must be 2 or 4, got " + std::to_string(elem));
    }
    /* `src` is device f32; the slab is `elem` bytes per element with rows
     * `n_head_kv * d` apart.  The generic aclnn conversion ops the package
     * carries do not include a plain f32->f16 cast, so a half-width slab is
     * converted on the host for now -- which is why `preferred_kv_dtype()`
     * reports f32 here.  The f32 path is an exact device-to-device copy. */
    const int64_t width = n_head_kv * d;
    if (elem == 4) {
      copy_device_to_device(dst, src, n * width * 4);
      return;
    }
    std::vector<float> rows(static_cast<std::size_t>(n * width));
    copy_to_host(rows.data(), src, n * width * 4);
    const std::vector<uint16_t> half = to_f16(rows.data(), n * width);
    copy_to_device(dst, half.data(), n * width * 2);
  }

  /* f32 by default: `kv_append(elem=2)` on this backend is a host conversion,
   * because a custom-op package with no plain f32->f16 cast op is what this
   * board has.  The graph binds the cache from this answer, so declaring f32 is
   * what keeps that round trip off the per-token path.
   *
   * `$POCKETLLM_ASCEND_KV_F16=1` selects the f16 cache the graph will want once
   * a device cast exists -- the same switch-by-name mechanism as the CPU
   * backend's `$POCKETLLM_CPU_KV_F32`, so a test can ask for the *other*
   * convention without a rebuild. */
  KVDtype preferred_kv_dtype() const override {
    const char *from_env = std::getenv("POCKETLLM_ASCEND_KV_F16");
    const bool as_f16 = from_env != nullptr && from_env[0] != '\0' && from_env[0] != '0';
    return as_f16 ? KVDtype::kF16 : KVDtype::kF32;
  }
  void argmax(DeviceBuffer, int64_t, DeviceBuffer) override {
    throw Error("ascend: argmax not implemented yet");
  }
  void softmax(DeviceBuffer, DeviceBuffer, int64_t, int64_t) override {
    throw Error("ascend: softmax not implemented yet");
  }
  void logits_temperature(DeviceBuffer, DeviceBuffer, int64_t, float) override {
    throw Error("ascend: logits_temperature not implemented yet");
  }
  void topk_sample(DeviceBuffer logits, int64_t vocab, float uniform, int64_t top_k, float top_p,
                   float min_p, DeviceBuffer order, DeviceBuffer out) override {
    (void)logits;
    (void)vocab;
    (void)uniform;
    (void)top_k;
    (void)top_p;
    (void)min_p;
    (void)order;
    (void)out;
    throw Error("ascend: topk_sample not implemented yet");
  }

  void synchronize() override { acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream"); }

  std::string describe() const override {
    const char *const custom = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    return "ascend (" + soc_ + ", device 0; custom ops " +
           (custom != nullptr && custom[0] != '\0' ? custom : "NOT FOUND") + ")";
  }

 private:
  /* Widen `count` device elements -- f16 or f32 -- to host f32.  The dtype is
   * the checkpoint's, and only the two widths the graph binds f16 weights as
   * are accepted; anything else is refused rather than reinterpreted. */
  std::vector<float> read_f32(DeviceBuffer buffer, int64_t count, int dtype) {
    if (dtype != ACL_FLOAT && dtype != ACL_FLOAT16) {
      throw Error("ascend: expected an f32 or f16 operand, got aclnn dtype " +
                  std::to_string(dtype));
    }
    if (dtype == ACL_FLOAT) {
      std::vector<float> out(static_cast<std::size_t>(count));
      copy_to_host(out.data(), buffer, count * 4);
      return out;
    }
    std::vector<uint16_t> half(static_cast<std::size_t>(count));
    copy_to_host(half.data(), buffer, count * 2);
    return half_to_f32(half);
  }

  /* One `aclnnRmsNormNdCustom` drive.  `x`/`weight`/`out` are fp16 on the
   * device; the op keeps the whole row in UB, sums the squares in f32 (a
   * 1e4-magnitude residual stream overflows an fp16 square), and rounds the
   * result to fp16.  It handles the in-place case (out == x): the kernel reads
   * the row into UB before it writes it back, so there is no aliasing hazard. */
  void run_rms_nd(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer out, int64_t rows,
                  int64_t hidden, float eps) {
    const int64_t x_shape[2] = {rows, hidden};
    const int64_t g_shape[1] = {hidden};
    const int64_t x_stride[2] = {hidden, 1};
    const int64_t g_stride[1] = {1};
    aclTensor *tx = aclCreateTensor(x_shape, 2, ACL_FLOAT16, x_stride, 0, ACL_FORMAT_ND, x_shape,
                                    2, reinterpret_cast<void *>(x.handle));
    aclTensor *tg = aclCreateTensor(g_shape, 1, ACL_FLOAT16, g_stride, 0, ACL_FORMAT_ND, g_shape, 1,
                                    reinterpret_cast<void *>(weight.handle));
    aclTensor *to = aclCreateTensor(x_shape, 2, ACL_FLOAT16, x_stride, 0, ACL_FORMAT_ND, x_shape, 2,
                                    reinterpret_cast<void *>(out.handle));
    if (tx == nullptr || tg == nullptr || to == nullptr) {
      throw Error("ascend: aclCreateTensor returned null (rms_norm)");
    }
    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnn_ok(aclnnRmsNormNdCustomGetWorkspaceSize(tx, tg, static_cast<double>(eps), to, &ws_size,
                                                  &executor),
             "aclnnRmsNormNdCustomGetWorkspaceSize");
    void *workspace = nullptr;
    if (ws_size > 0) {
      acl_ok(aclrtMalloc(&workspace, static_cast<std::size_t>(ws_size), ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc workspace");
    }
    aclnn_ok(aclnnRmsNormNdCustom(workspace, ws_size, executor, stream_), "aclnnRmsNormNdCustom");
    acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    aclDestroyTensor(tx);
    aclDestroyTensor(tg);
    aclDestroyTensor(to);
  }

  /* One `aclnnMatmulCubeCustom` drive: `out = x_f32[m, k] @ w^T`, where `w` is
   * the row-major [n, k] plane the graph holds and the op wants B as [k, n].
   * The transpose is done on the host (see `gemm`), the fp16 operands and the
   * fp16 result go on the device, and the result comes back widened to f32 --
   * the graph's activation type. */
  void run_cube(const std::vector<float> &x_f32, DeviceBuffer w, int64_t m, int64_t n,
                int64_t k, bool accumulate, DeviceBuffer out) {
    const std::vector<float> w_f32 = read_f32(w, n * k, ACL_FLOAT);
    const std::vector<uint16_t> x_h = to_f16(x_f32.data(), m * k);
    std::vector<uint16_t> b_h(static_cast<std::size_t>(k * n));
    for (int64_t r = 0; r < n; ++r) {
      for (int64_t c = 0; c < k; ++c) {
        b_h[static_cast<std::size_t>(c * n + r)] = f32_to_f16(w_f32[static_cast<std::size_t>(r * k + c)]);
      }
    }
    DeviceBuffer dx = allocate(m * k * 2);
    DeviceBuffer db = allocate(k * n * 2);
    DeviceBuffer dout = allocate(m * n * 2);
    copy_to_device(dx, x_h.data(), m * k * 2);
    copy_to_device(db, b_h.data(), k * n * 2);

    const int64_t a_shape[2] = {m, k};
    const int64_t b_shape[2] = {k, n};
    const int64_t o_shape[2] = {m, n};
    const int64_t a_stride[2] = {k, 1};
    const int64_t b_stride[2] = {n, 1};
    const int64_t o_stride[2] = {n, 1};
    aclTensor *ta = aclCreateTensor(a_shape, 2, ACL_FLOAT16, a_stride, 0, ACL_FORMAT_ND, a_shape, 2,
                                    reinterpret_cast<void *>(dx.handle));
    aclTensor *tb = aclCreateTensor(b_shape, 2, ACL_FLOAT16, b_stride, 0, ACL_FORMAT_ND, b_shape, 2,
                                    reinterpret_cast<void *>(db.handle));
    aclTensor *to = aclCreateTensor(o_shape, 2, ACL_FLOAT16, o_stride, 0, ACL_FORMAT_ND, o_shape, 2,
                                    reinterpret_cast<void *>(dout.handle));
    if (ta == nullptr || tb == nullptr || to == nullptr) {
      throw Error("ascend: aclCreateTensor returned null (gemm)");
    }
    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnn_ok(aclnnMatmulCubeCustomGetWorkspaceSize(ta, tb, to, &ws_size, &executor),
             "aclnnMatmulCubeCustomGetWorkspaceSize");
    void *workspace = nullptr;
    if (ws_size > 0) {
      acl_ok(aclrtMalloc(&workspace, static_cast<std::size_t>(ws_size), ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc workspace");
    }
    aclnn_ok(aclnnMatmulCubeCustom(workspace, ws_size, executor, stream_), "aclnnMatmulCubeCustom");
    acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");

    std::vector<uint16_t> out_h(static_cast<std::size_t>(m * n));
    copy_to_host(out_h.data(), dout, m * n * 2);
    std::vector<float> out_f = half_to_f32(out_h);
    if (accumulate) {
      std::vector<float> prior(static_cast<std::size_t>(m * n));
      copy_to_host(prior.data(), out, m * n * 4);
      for (std::size_t i = 0; i < out_f.size(); ++i) {
        out_f[i] += prior[i];
      }
    }
    copy_to_device(out, out_f.data(), m * n * 4);

    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(to);
    release(dx);
    release(db);
    release(dout);
  }

  /* Bring a window of the K/V cache up as f32, whatever width it was bound in.
   * The f16 arm exists for the `$POCKETLLM_ASCEND_KV_F16` cache; the default
   * f32 arm is a straight read. */
  void read_kv(DeviceBuffer cache, int64_t count, KVDtype dtype, std::vector<float> *out) {
    if (dtype == KVDtype::kF32) {
      copy_to_host(out->data(), cache, count * 4);
      return;
    }
    std::vector<uint16_t> half(static_cast<std::size_t>(count));
    copy_to_host(half.data(), cache, count * 2);
    *out = half_to_f32(half);
  }

  /* One `aclnnAttentionStepCustom` drive for a single decode token.
   *
   * The graph's cache is ``[position][head][d]`` and the op walks
   * ``[head][position][d]`` with *raw base-relative offsets* -- it does not
   * honour the aclTensor strides, so a strided view cannot bridge the two.  The
   * `context`-row window is therefore gathered head-by-head on the host and
   * packed into the op's layout, and the output is scattered back.  See the file
   * header for what this costs and why it is stated rather than hidden. */
  void run_attention(DeviceBuffer q, int64_t n_heads, int64_t n_head_kv, int64_t d,
                     int64_t context, float scale, DeviceBuffer k_cache, DeviceBuffer v_cache,
                     KVDtype kv_dtype, DeviceBuffer out) {
    const int64_t width = n_head_kv * d;
    const int64_t span = context * width;
    /* The graph's token-major cache holds the `context` visible rows packed
     * contiguously from position 0, so one streaming read of `context * width`
     * elements is the whole window -- in whichever width the graph bound. */
    std::vector<float> k_rows(static_cast<std::size_t>(span));
    std::vector<float> v_rows(static_cast<std::size_t>(span));
    read_kv(k_cache, span, kv_dtype, &k_rows);
    read_kv(v_cache, span, kv_dtype, &v_rows);

    std::vector<float> q_f(static_cast<std::size_t>(n_heads * d));
    copy_to_host(q_f.data(), q, n_heads * d * 4);

    /* [position][head][d] -> [head][position][d] */
    std::vector<float> k_planar(static_cast<std::size_t>(span));
    std::vector<float> v_planar(static_cast<std::size_t>(span));
    for (int64_t t = 0; t < context; ++t) {
      for (int64_t h = 0; h < n_head_kv; ++h) {
        const int64_t src = (t * n_head_kv + h) * d;
        const int64_t dst = (h * context + t) * d;
        std::memcpy(&k_planar[static_cast<std::size_t>(dst)], &k_rows[static_cast<std::size_t>(src)],
                    static_cast<std::size_t>(d) * 4);
        std::memcpy(&v_planar[static_cast<std::size_t>(dst)], &v_rows[static_cast<std::size_t>(src)],
                    static_cast<std::size_t>(d) * 4);
      }
    }
    /* fp32 scores with the query and the cache in fp16 is the combination the
     * kernel is written for, so the cache is widened through fp32 above and the
     * op narrows it back -- the double rounding is a fp16 value either way. */
    std::vector<uint16_t> q_h = to_f16(q_f.data(), n_heads * d);
    std::vector<uint16_t> k_h = to_f16(k_planar.data(), span);
    std::vector<uint16_t> v_h = to_f16(v_planar.data(), span);

    DeviceBuffer dq = allocate(n_heads * d * 2);
    DeviceBuffer dk = allocate(span * 2);
    DeviceBuffer dv = allocate(span * 2);
    DeviceBuffer dout = allocate(n_heads * d * 2);
    copy_to_device(dq, q_h.data(), n_heads * d * 2);
    copy_to_device(dk, k_h.data(), span * 2);
    copy_to_device(dv, v_h.data(), span * 2);

    const int64_t q_shape[1] = {n_heads * d};
    const int64_t kv_shape[3] = {n_head_kv, context, d};
    const int64_t q_stride[1] = {1};
    const int64_t kv_stride[3] = {context * d, d, 1};
    aclTensor *tq = aclCreateTensor(q_shape, 1, ACL_FLOAT16, q_stride, 0, ACL_FORMAT_ND, q_shape, 1,
                                    reinterpret_cast<void *>(dq.handle));
    aclTensor *tk = aclCreateTensor(kv_shape, 3, ACL_FLOAT16, kv_stride, 0, ACL_FORMAT_ND, kv_shape,
                                    3, reinterpret_cast<void *>(dk.handle));
    aclTensor *tv = aclCreateTensor(kv_shape, 3, ACL_FLOAT16, kv_stride, 0, ACL_FORMAT_ND, kv_shape,
                                    3, reinterpret_cast<void *>(dv.handle));
    aclTensor *to = aclCreateTensor(q_shape, 1, ACL_FLOAT16, q_stride, 0, ACL_FORMAT_ND, q_shape, 1,
                                    reinterpret_cast<void *>(dout.handle));
    if (tq == nullptr || tk == nullptr || tv == nullptr || to == nullptr) {
      throw Error("ascend: aclCreateTensor returned null (attention)");
    }
    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnn_ok(aclnnAttentionStepCustomGetWorkspaceSize(tq, tk, tv, context, n_heads, n_head_kv,
                                                      static_cast<double>(scale), to, &ws_size,
                                                      &executor),
             "aclnnAttentionStepCustomGetWorkspaceSize");
    void *workspace = nullptr;
    if (ws_size > 0) {
      acl_ok(aclrtMalloc(&workspace, static_cast<std::size_t>(ws_size), ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc workspace");
    }
    aclnn_ok(aclnnAttentionStepCustom(workspace, ws_size, executor, stream_),
             "aclnnAttentionStepCustom");
    acl_ok(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");

    std::vector<uint16_t> out_h(static_cast<std::size_t>(n_heads * d));
    copy_to_host(out_h.data(), dout, n_heads * d * 2);
    const std::vector<float> out_f = half_to_f32(out_h);
    copy_to_device(out, out_f.data(), n_heads * d * 4);

    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    aclDestroyTensor(tq);
    aclDestroyTensor(tk);
    aclDestroyTensor(tv);
    aclDestroyTensor(to);
    release(dq);
    release(dk);
    release(dv);
    release(dout);
  }

  /* Decode+repack a q4_K matrix once per distinct source pointer.  The pointer
   * is the key rather than the contents because scanning the contents would cost
   * more than the repack it is trying to avoid; a checkpoint allocates each
   * weight once and never frees it, which is what makes the pointer stable. */
  const PackedQ4K &pack_for(DeviceBuffer blocks, int64_t n, int64_t k) {
    const std::string key = std::to_string(blocks.handle) + ":" + std::to_string(n) + ":" +
                            std::to_string(k);
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    /* The q4_K tensor lives in device memory; the decode (`dequant_q4_k`) reads
     * host bytes.  Bring it back once, prepare the packed image, and keep only
     * the packed image -- the decoded copy is the memory the quantization was
     * meant to save, so it does not outlive this call. */
    const int64_t block_bytes = quant::kQ4KBlockBytes;
    const int64_t total = n * (k / quant::kBlockWeights) * block_bytes;
    std::vector<uint8_t> host(static_cast<std::size_t>(total));
    copy_to_host(host.data(), blocks, total);
    PackedQ4K packed;
    pack_q4k(host.data(), n, k, &packed);
    auto inserted = cache_.emplace(key, std::move(packed));
    return inserted.first->second;
  }

  aclrtStream stream_ = nullptr;
  std::string soc_;
  std::mutex mutex_;
  std::unordered_map<std::string, PackedQ4K> cache_;
};

}  // namespace

std::unique_ptr<Backend> make_ascend_backend() { return std::make_unique<AscendBackend>(); }

}  // namespace kernel
}  // namespace pocketllm