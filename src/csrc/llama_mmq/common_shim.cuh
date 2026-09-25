// Minimal shim providing the symbols that llama.cpp's vendored mma.cuh,
// vecdotq.cuh, and mmq.cuh expect from common.cuh. We only need the
// device-side helpers and type traits; the host-side launcher (mul_mat_q
// stream-k, pool, device info) is replaced by our own wrapper in
// gguf_mma_wrapper.cu.

#pragma once

// PyTorch's CUDAExtension injects -D__CUDA_NO_HALF_CONVERSIONS__ and
// -D__CUDA_NO_BFLOAT16_CONVERSIONS__ globally, which disables the implicit
// float<->half conversions that the vendored llama.cpp MMQ code relies on
// heavily (e.g. `const float d = bxi->dm;`). Undefine them so the vendored
// headers compile; this only affects translation units that include this shim.
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif

#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ggml-common.h provides block structs, K-quant constants, ggml_half.
// Its declarations are gated behind GGML_COMMON_DECL_CUDA (selected under __CUDACC__).
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

// ---------------------------------------------------------------------------
// Misc macros / stubs (from common.cuh). Many grids/tables below are only
// referenced by IQ/MXFP4 paths we never instantiate, but the non-template
// device functions in vecdotq.cuh reference them, so they must resolve.
// ---------------------------------------------------------------------------
#define GGML_PAD(x, y) (((x) + (y) - 1) / (y) * (y))

#define STRINGIZE(x) #x
#define STRINGIZE2(x) STRINGIZE(x)

#define GGML_UNUSED(...) (void)sizeof(0 __VA_OPT__(,) __VA_ARGS__)
#define GGML_ABORT(...) do { if (false) { (void)sizeof(0 __VA_OPT__(,) __VA_ARGS__); } } while (0)

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE do { /* device no-op for unsupported arch path */ } while (0)
#else
#define NO_DEVICE_CODE
#endif

#define GGML_UNUSED_VARS(...) (void)sizeof(0 __VA_OPT__(,) __VA_ARGS__)

static constexpr int WARP_SIZE = 32;
static constexpr int WARP_SIZE_GGUF = 32;

// IQ-quant grid stubs (never dereferenced on Q4_K/Q5_K paths).
// llama.cpp declares these as int8_t arrays (used by get_int_from_table_16);
// keep the element type matching so the vendored functions compile.
static const int8_t iq2xxs_grid[1]   = {0};
static const int8_t iq2xs_grid[1]    = {0};
static const int8_t iq2s_grid[1]     = {0};
static const int8_t iq3xxs_grid[1]   = {0};
static const int8_t iq3s_grid[1]     = {0};
static const int8_t iq1s_grid_gpu[1] = {0};
static const int8_t kvalues_iq4nl[1] = {0};
static const int8_t kvalues_mxfp4[1] = {0};
#define IQ1M_DELTA 0.0f
#define IQ1S_DELTA 0.0f

// ---------------------------------------------------------------------------
// Compute capability constants (we target sm_75 = Turing).
// ---------------------------------------------------------------------------
#define GGML_CUDA_CC_VOLTA   700
#define GGML_CUDA_CC_TURING  750
#define GGML_CUDA_CC_AMPERE  800

// ---------------------------------------------------------------------------
// Platform detection (mirrors common.cuh)
// ---------------------------------------------------------------------------
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif

static inline bool turing_mma_available(int cc) {
    return cc >= GGML_CUDA_CC_TURING;
}

// CC classifiers (we only run NVIDIA; AMD macros map to false).
static inline bool GGML_CUDA_CC_IS_NVIDIA(int cc) { (void)cc; return true; }
static inline bool GGML_CUDA_CC_IS_AMD(int cc) { (void)cc; return false; }
static inline bool GGML_CUDA_CC_IS_RDNA1(int cc) { (void)cc; return false; }

// We compile for sm_75, so the highest compiled arch is Turing.
static inline int ggml_cuda_highest_compiled_arch(int cc) {
    (void)cc;
    return GGML_CUDA_CC_TURING;
}

// Max contiguous shared-mem copy bytes. llama.cpp returns 16 (a single ldmatrix
// lane's worth); ggml_cuda_memcpy_1 asserts n ∈ {4,8,16}. Must be constexpr so
// it can size static_assert / loop bounds inside load_tiles_*.
static constexpr __device__ __host__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}

// AMD matrix-core host predicates — we never run AMD, so always false.
static inline bool amd_mfma_available(int cc) { (void)cc; return false; }
static inline bool amd_wmma_available(int cc) { (void)cc; return false; }

// ---------------------------------------------------------------------------
// ggml_type enum — values must match llama.cpp's ggml.h exactly (the vendored
// headers use these as switch-case labels and template non-type args).
// ---------------------------------------------------------------------------
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_NVFP4   = 40,
    GGML_TYPE_Q1_0    = 41,
    // Fork-private ids from PrismML-Eng/llama.cpp's `prism` branch, read out of the
    // released Ternary-Bonsai-2-27B GGUF header.  Upstream GGML assigns nothing near
    // 142, so the enumeration has to be extended rather than aliased.  GGML_TYPE_COUNT
    // is not referenced anywhere in the vendored tree; it is moved to keep the
    // enumerators ordered, not because anything sizes an array by it.
    GGML_TYPE_PQ2_0   = 142,
    GGML_TYPE_PTQ1_0  = 143,
    GGML_TYPE_COUNT   = 144,
};

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return __dp4a(a, b, c);
#else
    const int8_t * aa = (const int8_t *) &a;
    const int8_t * bb = (const int8_t *) &b;
    return c + aa[0]*bb[0] + aa[1]*bb[1] + aa[2]*bb[2] + aa[3]*bb[3];
#endif
}

static constexpr __device__ __forceinline__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

// FP4 scale decoders — only referenced by MXFP4/NVFP4 paths we never instantiate,
// but the symbols must resolve when the headers compile. Trivial stubs are fine.
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t e) {
    return (e == 0) ? 0.0f : __exp2f((int)e - 127);
}
static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t d) {
    return (float)d;  // stub: unused for Q4_K/Q5_K
}

template <int n>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * dst, const void * src) {
    static_assert(n == 4 || n == 8 || n == 16, "unsupported memcpy size");
    if constexpr (n == 4) {
        *((int *) dst) = *((const int *) src);
    } else if constexpr (n == 8) {
        int2 * d8 = (int2 *) dst; const int2 * s8 = (const int2 *) src; *d8 = *s8;
    } else {
        int4 * d16 = (int4 *) dst; const int4 * s16 = (const int4 *) src; *d16 = *s16;
    }
}

// ---------------------------------------------------------------------------
// type_traits: qk/qr/qi per type (from common.cuh)
// ---------------------------------------------------------------------------
template <ggml_type>
struct ggml_cuda_type_traits;

template <> struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> { static constexpr int qk = QK_K; static constexpr int qr = QR4_K; static constexpr int qi = QI4_K; };
template <> struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> { static constexpr int qk = QK_K; static constexpr int qr = QR5_K; static constexpr int qi = QI5_K; };
// qk is what the tile walk reads; qr/qi only have to stay consistent with it (the
// block is 32 words of expanded trits, so 4 words per eight weights).
template <> struct ggml_cuda_type_traits<GGML_TYPE_PTQ1_0> {
    static constexpr int qk = QK_PTQ1_0;
    static constexpr int qr = QK_PTQ1_0 / 32;
    static constexpr int qi = QK_PTQ1_0 / (4 * qr);
};
