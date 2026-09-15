# Phase 3.2 Implementation Plan: Multi-slot KV Cache

## 目标

实现 KV cache 的 multi-slot 布局，支持 `max_batch_size=2/4/8` 的并发请求。

## 当前架构分析

### 单 Session KV Cache（现有）

**分配位置**: `qwen_engine.cpp:778-833`

**当前布局**:
```cpp
const size_t cache_elements = max_context * local_kv_heads * head_dim;
const std::vector<uint64_t> cache_shape = {
    max_context,        // [seq_len, kv_heads, head_dim]
    local_kv_heads,
    head_dim
};
allocate_half(destination.full.k_cache, cache_elements, cache_shape);
```

**问题**:
- 只能服务单个请求
- `allocate_batch_slots(N>1)` 抛出异常

## 实施策略

### 方案 A: 构造时确定 max_batch_size（推荐）

**优点**:
- 一次性分配，无需后期重新分配
- 避免运行时重新分配的复杂性和风险
- 符合现有代码结构（构造函数中分配所有资源）

**实施步骤**:
1. 从环境变量读取 `QWEN_MAX_BATCH_SIZE`（默认 1）
2. 在构造函数中使用 `max_batch_size` 分配 KV cache
3. 修改所有 attention kernel 调用添加 slot offset

### 方案 B: 延迟分配（更复杂）

**缺点**:
- 需要在 `allocate_batch_slots()` 中释放旧 tensor 并重新分配
- 风险高：可能破坏已初始化的状态
- 需要处理 TP 同步问题

**结论**: 采用方案 A

## 详细实施计划

### Step 1: 添加 max_batch_size 参数到 QwenEngineOptions

```cpp
// qwen_engine.hpp
struct QwenEngineOptions {
    // ... existing fields ...
    
    // Maximum concurrent requests for batched execution.
    // 1 = single-session mode (default, backward compatible)
    // 2-8 = multi-slot mode (enables continuous batching)
    int max_batch_size = 1;
};
```

### Step 2: 修改 KV cache 分配逻辑

**位置**: `qwen_engine.cpp:778-833`

**修改前**:
```cpp
const size_t cache_elements = max_context * local_kv_heads * head_dim;
const std::vector<uint64_t> cache_shape = {max_context, local_kv_heads, head_dim};
```

**修改后**:
```cpp
const int slot_count = impl_->max_batch_size;
const size_t cache_elements = slot_count * max_context * local_kv_heads * head_dim;
const std::vector<uint64_t> cache_shape = {
    static_cast<uint64_t>(slot_count),
    static_cast<uint64_t>(max_context),
    static_cast<uint64_t>(local_kv_heads),
    static_cast<uint64_t>(head_dim)
};
```

**适用于所有 KV dtype**:
- FP16: k_cache/v_cache
- FP8: k_cache/v_cache + k_scale/v_scale
- TurboQuantK8V4: turboquant_cache
- Int8PerTokenHead: k_cache/v_cache + k_scale/v_scale

### Step 3: 添加 slot offset 计算辅助函数

```cpp
// qwen_engine.cpp Impl
struct QwenEngine::Impl {
    // ... existing fields ...
    
    // Calculate byte offset for a given slot_id in KV cache
    size_t kv_slot_offset_bytes(int slot_id, int local_kv_heads, int head_dim) const {
        if (max_batch_size == 1) return 0;  // Fast path for single session
        return slot_id * max_context * local_kv_heads * head_dim * sizeof(uint16_t);
    }
    
    size_t kv_slot_offset_elements(int slot_id, int local_kv_heads, int head_dim) const {
        if (max_batch_size == 1) return 0;
        return slot_id * max_context * local_kv_heads * head_dim;
    }
};
```

### Step 4: 修改 attention kernel 调用添加 slot offset

**示例位置**: `qwen_engine.cpp:2220` (FP16 KV cache append)

**修改前**:
```cpp
check_device(qwen_append_kv_cache_f16(
    k_norm.f16_data(), v_norm.f16_data(),
    layer.full.k_cache.f16_data(),
    layer.full.v_cache.f16_data(),
    rows, kv_heads, head_dim,
    position_offset, max_context), "append FP16 full KV cache");
```

**修改后**:
```cpp
const int slot_id = 0;  // TODO: get from request context
const size_t slot_offset_elems = impl_->kv_slot_offset_elements(slot_id, kv_heads, head_dim);
check_device(qwen_append_kv_cache_f16(
    k_norm.f16_data(), v_norm.f16_data(),
    layer.full.k_cache.f16_data() + slot_offset_elems,
    layer.full.v_cache.f16_data() + slot_offset_elems,
    rows, kv_heads, head_dim,
    position_offset, max_context), "append FP16 full KV cache");
```

**需要修改的所有位置**:
```bash
grep -n "k_cache.f16_data()\|k_cache.f8_e4m3_data()\|k_cache.i8_data()\|turboquant_cache" \
  cpp_engine/engine/qwen_engine.cpp | grep -v "allocate"
```

### Step 5: 修改 batch API 传递 slot_id

**batch_prefill**:
```cpp
QwenBatchPrefillResult QwenEngine::batch_prefill(
    const std::vector<QwenBatchedRequest*>& requests) {
    
    for (QwenBatchedRequest* req : requests) {
        // Save current position and restore
        const int saved_position = position_;
        const int slot_id = req->slot_id;
        
        // TODO: Set current slot_id in Impl for kernel access
        impl_->current_slot_id = slot_id;
        
        QwenForwardResult fwd_result = prefill(req->prompt_tokens);
        req->last_result = fwd_result;
        
        // Restore position (for multi-slot isolation)
        position_ = saved_position;
    }
}
```

### Step 6: 传递 slot_id 到 kernel 调用

**方案 A: Thread-local current_slot_id**（简单）
```cpp
struct QwenEngine::Impl {
    int current_slot_id = 0;  // Used by kernel calls
};
```

**方案 B: 显式参数传递**（更安全，推荐）
- 将 `slot_id` 作为参数传递给所有内部函数
- 修改 `run_chunk()` 等函数签名

**Phase 3.2 采用方案 A**（快速验证），Phase 3.3 重构为方案 B。

### Step 7: 测试计划

1. **单 session 零回归**:
   ```bash
   # max_batch_size=1（默认）
   ./test_qwen_single_session
   # 验证：token parity, 性能 ≤1.05× baseline
   ```

2. **2-slot 基础测试**:
   ```bash
   # QWEN_MAX_BATCH_SIZE=2
   ./test_qwen_2slot_basic
   # 验证：slot 0 和 slot 1 独立生成正确 tokens
   ```

3. **内存泄漏检查**:
   ```bash
   valgrind --leak-check=full ./test_qwen_2slot_alloc_free
   ```

## 内存开销分析

### 单 Layer FP16 KV Cache

**公式**: `slot_count × max_seq_len × kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16)`

**TP2 示例** (Qwen3.8-27B-FP8, 64 layers):
- `kv_heads = 32 / 2 = 16` (TP分片)
- `head_dim = 128`
- `max_seq_len = 65536`

**单 layer**:
- 1-slot: `1 × 65536 × 16 × 128 × 2 × 2 = 1.07 GB`
- 2-slot: `2 × 65536 × 16 × 128 × 2 × 2 = 2.14 GB`
- 8-slot: `8 × 65536 × 16 × 128 × 2 × 2 = 8.59 GB`

**64 layers**:
- 1-slot: 1.07 GB × 64 = 68.7 GB (分片后每 rank: 68.7/2 = 34.4 GB) ❌ 超预算
- 实际: 只有 16 full-attention layers，其余是 linear-attention
- 估算: ~8 GB/rank (1-slot) → 16 GB/rank (2-slot) → 64 GB/rank (8-slot) ❌

**结论**: 8-slot 对 64K context 不可行，Phase 3.2 目标 **max_batch_size=2**。

## 实施顺序

1. ✅ **Step 1**: 添加 `max_batch_size` 到 `QwenEngineOptions`
2. ✅ **Step 2**: 修改 KV cache 分配（所有 dtype）
3. ✅ **Step 3**: 添加 slot offset 辅助函数
4. ✅ **Step 4**: 修改所有 attention kernel 调用（~30 处）
5. ✅ **Step 5**: 修改 batch API 使用 slot_id
6. ✅ **Step 6**: 实现 slot_id 传递机制
7. ⏳ **Step 7**: 测试和验证

**预计时间**: 1 天编码 + 1 天测试 = 2 天

## 风险和缓解

### 风险 1: 遗漏某些 kernel 调用点

**缓解**: 
- 用 `grep` 系统性查找所有 `k_cache.*data()` 调用
- 编译器会捕获类型不匹配

### 风险 2: 破坏单 session 性能

**缓解**:
- 快速路径: `if (max_batch_size == 1) return 0`
- 测试: 单 session baseline ≤1.05×

### 风险 3: 内存超预算

**缓解**:
- Phase 3.2 只实现 2-slot
- 8-slot 需要 sliding window 或 paged attention

## 验收标准

- [ ] 编译通过
- [ ] 单 session token parity (max_batch_size=1)
- [ ] 单 session 性能 ≤1.05× baseline
- [ ] 2-slot 独立正确性（两个请求生成不同 tokens）
- [ ] 2-slot token parity（每个请求与单独运行一致）
- [ ] 无内存泄漏
- [ ] `allocate_batch_slots(2)` 不再抛出异常

---

**开始实施**: Phase 3.2 Step 1
