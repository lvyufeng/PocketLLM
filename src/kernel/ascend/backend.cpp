/* The Ascend 310B backend.
 *
 * The third implementation of `kernel/backend.h`, after the CPU and the card.
 * This one is a skeleton by design and by scope: the *memory* half of the
 * interface -- allocate, release, the three copies, fill, synchronize -- is
 * real, and exactly one compute op (`gemm_quant` for q4_k) is real, because
 * that is the op the 310B's missing built-in matmul makes the blocking one.
 * Every other op throws, by name, so a caller that reaches it learns which op
 * is absent rather than getting a wrong number.
 *
 * ## Why the built-in matmul cannot be used
 *
 * `aclnnMm` and the other built-in matmuls ship no `ascend310b` kernel binary
 * on CANN 8.3.RC2, so they fail on this board. The custom ops built in the
 * minicpm-o-4.5-orangepi tree supply the 310B binaries; this backend drives
 * `aclnnMatmulW4a16Custom` for the quantized GEMM. The other custom ops the
 * backend will eventually want -- `RmsNorm1024Custom`, `SiluMulCustom`,
 * `MatmulCubeCustom`, `AttentionStepCustom` -- are built and installed but not
 * wired here yet, which is why the ops that would use them throw.
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

#include "aclnn_matmul_w4a16_custom.h"
#include "kernel/backend.h"
#include "quant/blocks.h"
#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

namespace {

constexpr int64_t kGroup = 128;        /* MatmulW4a16Custom's fixed group size */
constexpr int64_t kTileLen = 64;       /* tileLen: % 16 == 0 and <= 448 */
constexpr int64_t kGemmAlign = 128;    /* n must be a multiple of this */

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

  void rms_norm(DeviceBuffer, DeviceBuffer, DeviceBuffer, int64_t, int64_t, float) override {
    throw Error("ascend: rms_norm not implemented yet");
  }
  void gemm(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, int64_t, int64_t, int64_t,
            bool) override {
    throw Error("ascend: gemm not implemented yet");
  }
  void embedding(DeviceBuffer, int64_t, DeviceBuffer, int64_t, int64_t, DeviceBuffer) override {
    throw Error("ascend: embedding not implemented yet");
  }
  void embedding_quant(DeviceBuffer, int64_t, DeviceBuffer, int64_t, int64_t, int,
                       DeviceBuffer) override {
    throw Error("ascend: embedding_quant not implemented yet");
  }
  void silu_mul(DeviceBuffer, DeviceBuffer, DeviceBuffer, int64_t) override {
    throw Error("ascend: silu_mul not implemented yet");
  }
  void rope_neox(DeviceBuffer, int64_t, int64_t, int64_t, int64_t, DeviceBuffer,
                 DeviceBuffer) override {
    throw Error("ascend: rope_neox not implemented yet");
  }
  int64_t attention_scratch(int64_t, int64_t, int64_t) const override {
    throw Error("ascend: attention_scratch not implemented yet");
  }
  void attention(DeviceBuffer, int64_t, int64_t, DeviceBuffer, DeviceBuffer, int64_t, int64_t,
                 int64_t, int64_t, float, DeviceBuffer, DeviceBuffer, KVDtype) override {
    throw Error("ascend: attention not implemented yet");
  }
  void kv_append(DeviceBuffer, DeviceBuffer, int64_t, int64_t, int64_t, int64_t) override {
    throw Error("ascend: kv_append not implemented yet");
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