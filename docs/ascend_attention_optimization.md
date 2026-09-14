# Ascend Attention Optimization Analysis

## Current State (2026-09-14)

### Profiling Results (Qwen 27B, 4K context, TP=4, 910A first-gen)
- **Prefill total**: 21.9s
- **Attention phase**: ~18.4s (84%)
- **Decode**: ~200ms/token

### Root Cause: No Cube Unit Usage

Grep results from `qwen_attention_f16.cpp` (1194 lines):
```bash
$ grep -i "Matmul\|matmul\|Mmad\|mmad" qwen_attention_f16.cpp
325:extern "C" __global__ __aicore__ void qwen_gqa_prefill_attention_kernel(
932:extern "C" __global__ __aicore__ void qwen_gqa_prefill_attention_vector_kernel(
```

**All attention kernels use Vector-only operations**. The Cube (matrix/Mmad) unit sits idle during the most expensive phase.

### Current Implementation Bottlenecks

1. **QK^T computation** (lines 710-724 in `vector_online_softmax_pass`):
   ```cpp
   for (uint32_t i = 0; i < count; ++i) {
       wait_scalar_before_compute();
       const float score = vector_dot(query, row_float[i * head_dim], product, head_dim) * scale;
       // vector_dot itself is a Mul + fold_sum loop over head_dim
   }
   ```
   - Scalar loop over positions
   - Inner vector_dot is element-wise multiply + reduction
   - No Cube matmul

2. **P·V accumulation** (lines 756-763):
   ```cpp
   for (uint32_t i = 0; i < count; ++i) {
       const float probability = scores.GetValue(i);
       wait_scalar_before_compute();
       AscendC::Muls(product, row_float[i * head_dim], probability, head_dim);
       AscendC::PipeBarrier<PIPE_V>();
       AscendC::Add(accum, accum, product, head_dim);
       AscendC::PipeBarrier<PIPE_V>();
   }
   ```
   - Scalar loop over positions
   - Each position: scale + add (Vector ops)
   - No Cube matmul

3. **Small tile size**:
   ```cpp
   constexpr uint32_t kVectorPositionTile = 16;
   ```
   - 4K context = 256 tile iterations
   - Poor memory locality

## Optimization Strategy

### Phase 1: Increase Tile Size (IMPLEMENTED, NOT TESTED)

**Change**: `kVectorPositionTile: 16 → 64`

**Expected impact**:
- 4× fewer outer loop iterations (256 → 64 for 4K context)
- Better memory access patterns (64 pos × 128 dim = 8KB K/V tiles fit in UB)
- Reduced loop overhead
- **Estimated speedup**: 1.3-1.5× on prefill

**Status**: Code changed in `cpp_engine/backends/ascend/kernels/qwen_attention_f16.cpp:553` but AscendC compilation is blocked.

**Blocker**: AscendC compiler fails with:
```
/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/utils/sys_macros.h:18:10: 
fatal error: 'cstdint' file not found
```

Despite `ASCENDC_COMPILE_OPTIONS` being set to include `/usr/include/c++/11`, the compiler can't find standard headers. This suggests a deeper toolchain configuration issue.

### Phase 2: Use aclnnBatchMatMul (PARTIAL PROTOTYPE)

**Approach**: Replace custom AscendC kernels with `aclnnBatchMatMul` calls for QK^T and P·V.

**Advantages**:
- Uses Cube unit automatically
- No AscendC compilation required
- Proven operator

**Challenges**:
- Requires separate passes for QK^T → softmax → P·V
- Causal masking and scaling need additional kernels
- GQA broadcast (q_heads/kv_heads repeat) adds complexity
- May require more memory for intermediate scores tensor

**Status**: Skeleton implementation in `cpp_engine/backends/ascend/kernels/qwen_attention_batched.cpp` (not integrated).

### Phase 3: Custom Cube Kernel (FUTURE)

**Approach**: Write AscendC kernel using `Mmad` intrinsic with fractal (C0) layout.

**Requirements**:
- Convert Q/K/V from row-major to C0 (16×16 tiles) using `LoadData2D`
- Size L0A/L0B/L0C buffers properly (~512KB total on 910A)
- Interleave Mmad QK^T with online softmax state updates
- Mmad P·V with running accumulator
- Convert output from C0 back to row-major

**Complexity**: High - fractal layout conversion is non-trivial and poorly documented.

**Expected impact**: 3-5× speedup if done correctly (Cube is ~2× faster than Vector at matmul, plus better pipelining).

## Hardware Context

**Ascend 910A (first generation)**:
- AI Core: 32 cores
- L2: 32 MB
- Cube frequency: 1000 MHz
- **No BF16 support** (FP16 only)
- Cube operates on 16×16 tiles in C0 fractal format
- `Short_SoC_version=Ascend910` (not `Ascend910B`)

## Next Steps

1. **Fix AscendC compilation toolchain**
   - Debug why `-isystem /usr/include/c++/11` isn't working
   - Or use precompiled kernel workflow if source builds are not supported

2. **Test tile size optimization** (Phase 1)
   - Rebuild with `kVectorPositionTile=64`
   - Benchmark 4K prefill with `QWEN_PHASE_PROFILE=1`
   - Expect ~18.4s → ~12-14s

3. **Profile decode FlashDecoding reduce**
   - Currently ~68ms/token (see PR #184 description)
   - May be next bottleneck after prefill

4. **Investigate aclnnBatchMatMul path** (Phase 2)
   - Prototype full attention with batched matmul
   - Measure vs. tile-optimized Vector kernel
   - Decide if complexity is worth it

5. **Long-term: Custom Cube kernel** (Phase 3)
   - Only pursue if Phase 2 shows Cube delivers significant wins
   - Requires CANN SDK documentation on C0 layout

## Files Modified

- `cpp_engine/backends/ascend/kernels/qwen_attention_f16.cpp:553` - tile size 16→64
- `cpp_engine/backends/ascend/kernels/qwen_attention_batched.cpp` - partial aclnnBatchMatMul prototype (not integrated)

## References

- Previous optimization: PR #184 (HCCL comm-stream overlap, +1.21× decode)
- Profiling data: 4K prefill = 21.9s total, 18.4s attention
- Current branch: `perf/attention-tile-optimization`
