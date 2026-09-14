# Ascend 910A Performance Optimization Roadmap
## Target: Qwen2.5-27B on 4×Ascend910A

### Performance Targets
- **Prefill**: ≥2000 tokens/second (TPS)
- **Decode**: ≥100 tokens/second (TPS)

### Current Baseline (from profiling)
- **Prefill (4K context)**: 21.9s → ~183 TPS (4096 tokens / 21.9s)
  - Attention: 18.4s (84%)
  - Other: 3.5s (16%)
- **Decode**: ~200ms/token → ~5 TPS per request × 4 devices = ~20 TPS total

### Performance Gap
- **Prefill**: Need 10.9× speedup (183 → 2000 TPS)
- **Decode**: Need 5× speedup (20 → 100 TPS)

---

## Phase 1: Critical Attention Optimization (DONE ✅)

**Status**: Tile size 16→64 implemented and tested

**Expected gain**: 1.3-1.5× prefill
- Before: ~183 TPS
- After: ~238-275 TPS

**Remaining gap**: 7.3-8.4× to reach 2000 TPS

---

## Phase 2: Cube-Accelerated Attention (HIGH PRIORITY)

**Approach**: Use aclnnBatchMatMul + custom scale/mask/softmax kernel

**Expected gain**: 3-4× on attention (which is 84% of prefill)
- Attention: 18.4s → ~5s
- Total prefill: 21.9s → ~8.5s → ~480 TPS

**Combined with Phase 1**: ~640-720 TPS

**Remaining gap**: 2.8-3.1× to reach 2000 TPS

**Implementation steps**:
1. Complete `qwen_attention_scale_mask_softmax.cpp` kernel
2. Integrate with `qwen_attention_batched.cpp`
3. Add to build system
4. Benchmark

**Estimated effort**: 2-3 days

---

## Phase 3: Operator Fusion (MEDIUM PRIORITY)

**Target**: Fuse small kernels to reduce launch overhead

**Candidates**:
1. RMSNorm + Residual Add → single kernel
2. RoPE + QKV split → single kernel
3. Gated activation (SwiGLU) fusion

**Expected gain**: 1.3-1.5× overall
- Reduced kernel launch overhead
- Better memory access patterns

**Combined with Phase 1+2**: ~960-1080 TPS

**Remaining gap**: 1.85-2.1× to reach 2000 TPS

---

## Phase 4: Memory & Communication Optimization (HIGH PRIORITY)

### 4.1 Tensor Parallelism Communication Overlap

**Current**: TP AllReduce blocks computation
**Target**: Overlap communication with next layer computation

**Expected gain**: 1.2-1.3× prefill, 1.5-2× decode
- Prefill: ~1200-1400 TPS
- Decode: ~30-40 TPS (per-request latency improvement)

**Implementation**:
- Use double buffering for AllReduce
- Pipeline layer N+1 computation with layer N communication

### 4.2 KV Cache Layout Optimization

**Current**: Potentially fragmented memory access
**Target**: Contiguous layout for better memory bandwidth

**Expected gain**: 1.1-1.2× overall

---

## Phase 5: Quantization (OPTIONAL)

**Approach**: W8A16 or W4A16 quantization

**Expected gain**: 
- 1.5-2× throughput (less memory bandwidth, faster matmul)
- But: Accuracy trade-off

**Decision point**: Only if other optimizations insufficient

---

## Phase 6: Decode Batch Optimization (CRITICAL FOR DECODE)

### 6.1 FlashDecoding Optimization

**Current**: ~68ms/token in reduce stage (from PR #182 description)
**Target**: Optimize reduce to <20ms

**Expected gain**: 3.4× decode latency improvement
- 200ms → ~60ms per token
- ~16 TPS → ~54 TPS (single request)

### 6.2 Continuous Batching

**Current**: One request at a time
**Target**: Batch multiple decode requests

**Expected gain**: 2-3× decode throughput
- With 4-8 concurrent requests: 54 TPS → ~100-160 TPS

**This is CRITICAL for decode target!**

---

## Aggressive Optimization Plan (3-4 weeks)

### Week 1: Attention Optimization
- [ ] Day 1-2: Benchmark Phase 1 (tile size)
- [ ] Day 3-5: Complete Phase 2 (Cube matmul)
- [ ] Day 6-7: Operator fusion pass 1

**Target after Week 1**: 800-1000 TPS prefill

### Week 2: Communication & Memory
- [ ] Day 1-3: TP communication overlap
- [ ] Day 4-5: KV cache layout optimization
- [ ] Day 6-7: Profile and fix bottlenecks

**Target after Week 2**: 1400-1600 TPS prefill

### Week 3: Decode Optimization
- [ ] Day 1-3: FlashDecoding reduce optimization
- [ ] Day 4-7: Continuous batching implementation

**Target after Week 3**: 80-100 TPS decode

### Week 4: Final Push
- [ ] Day 1-3: Profiling and micro-optimizations
- [ ] Day 4-5: Additional operator fusion
- [ ] Day 6-7: End-to-end testing and tuning

**Target after Week 4**: 2000+ TPS prefill, 100+ TPS decode

---

## Critical Path Analysis

**Must have for prefill target**:
1. ✅ Tile optimization (1.3-1.5×) - DONE
2. ⚠️ Cube attention (3-4×) - CRITICAL
3. ⚠️ TP communication overlap (1.2-1.3×) - CRITICAL
4. ⚠️ Operator fusion (1.3-1.5×) - IMPORTANT

**Must have for decode target**:
1. ⚠️ FlashDecoding reduce (3.4×) - CRITICAL
2. ⚠️ Continuous batching (2-3×) - CRITICAL

---

## Risk Assessment

### High Risk
- **Cube attention complexity**: Causal masking integration is non-trivial
  - Mitigation: Use prototype as starting point, iterate quickly

### Medium Risk
- **TP communication overlap**: Requires careful synchronization
  - Mitigation: Start with simple double-buffering, add complexity incrementally

### Low Risk
- **Operator fusion**: Well-understood optimization
- **Tile size**: Already implemented and tested

---

## Immediate Next Steps (Today)

1. **Benchmark Phase 1** to validate 1.3-1.5× gain
   ```bash
   cd /mnt/data/pocketllm/cpp_engine/build-ascend
   export QWEN_PHASE_PROFILE=1
   ./pocketllm_engine --model <path> --prefill-length 4096
   ```

2. **Complete Cube attention kernel** (Phase 2)
   - Fix `qwen_attention_scale_mask_softmax.cpp`
   - Integrate with batched matmul
   - Build and test

3. **Profile decode** to identify exact bottleneck
   ```bash
   export QWEN_PHASE_PROFILE=1
   # Run decode workload
   ```

---

## Success Metrics

### Prefill (target: 2000 TPS)
- Phase 1: ~240 TPS (baseline × 1.3)
- Phase 1+2: ~720 TPS (baseline × 3.9)
- Phase 1+2+3: ~1080 TPS (baseline × 5.9)
- Phase 1+2+3+4: ~1400 TPS (baseline × 7.6)
- **Final push to 2000 TPS**: Additional micro-optimizations

### Decode (target: 100 TPS)
- Current: ~20 TPS (4 devices, single request each)
- FlashDecoding: ~54 TPS
- Continuous batching: ~100-160 TPS ✅

---

**Created**: 2026-09-14
**Status**: Phase 1 complete, Phase 2 in progress
**Owner**: Performance optimization team
