# cpp_engine Batching Implementation (Phase 3.1)

## 目标

为 `QwenEngine` 添加 continuous batching 支持，实现 2-8 并发请求的高效处理，同时保持单请求路径的所有优化。

## 已完成工作

### 1. 设计 per-request state (Task #2 ✅)

**文件修改**：
- `cpp_engine/include/qwen_engine.hpp`

**新增类型**：
```cpp
struct QwenBatchSamplingParams {
    float temperature, top_p;
    int top_k, max_new_tokens;
    unsigned long long seed;
};

struct QwenBatchedRequest {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    int seq_len;
    int slot_id;  // KV cache slot (-1 = not allocated)
    int cached_prefix_len;
    QwenBatchSamplingParams sampling;
    bool finished;
    int last_token;
    std::vector<int> generated_tokens;
    QwenForwardResult last_result;
};

struct QwenBatchPrefillResult {
    std::vector<QwenForwardResult> results;
    int total_tokens;
    double seconds;
};

struct QwenBatchDecodeResult {
    std::vector<int> next_tokens;
    std::vector<bool> finished;
    double seconds;
};
```

**新增 API**：
```cpp
class QwenEngine {
    // Batch slot management
    void allocate_batch_slots(int max_batch_size);
    int allocate_slot(uint64_t request_id);
    void free_slot(uint64_t request_id);
    
    // Batch execution
    QwenBatchPrefillResult batch_prefill(const std::vector<QwenBatchedRequest*>&);
    QwenBatchDecodeResult batch_decode_step(const std::vector<QwenBatchedRequest*>&);
    bool supports_batching() const;
};
```

## 当前任务：重构 KV cache 布局 (Task #3 进行中)

### 当前 KV cache 结构

**单 session 布局**（现有）：
```cpp
struct DeviceFullAttention {
    QwenDeviceTensor k_cache;  // [max_seq_len, kv_heads, head_dim]
    QwenDeviceTensor v_cache;  // [max_seq_len, kv_heads, head_dim]
    QwenDeviceTensor k_scale;  // FP8/INT8 量化时使用
    QwenDeviceTensor v_scale;
    // ...
};
```

**分配代码位置**：`qwen_engine.cpp:775-833`
```cpp
const size_t cache_elements = max_context * local_kv_heads * head_dim;
const std::vector<uint64_t> cache_shape = {max_context, local_kv_heads, head_dim};
allocate_half(destination.full.k_cache, cache_elements, cache_shape);
allocate_half(destination.full.v_cache, cache_elements, cache_shape);
```

### 目标 multi-slot 布局

**批量 session 布局**（目标）：
```cpp
struct DeviceFullAttention {
    // 单 session 模式（slot_count == 1）：
    //   k_cache: [max_seq_len, kv_heads, head_dim]
    // 批量模式（slot_count > 1）：
    //   k_cache: [slot_count, max_seq_len, kv_heads, head_dim]
    QwenDeviceTensor k_cache;
    QwenDeviceTensor v_cache;
    QwenDeviceTensor k_scale;
    QwenDeviceTensor v_scale;
    
    // Slot 管理（仅批量模式使用）
    int slot_count = 1;  // 默认 1 = 单 session
    std::unordered_map<uint64_t, int> request_to_slot;
    std::vector<int> free_slots;
};
```

### 实施步骤

#### Step 3.1: 添加 slot 管理字段到 Impl
```cpp
// qwen_engine.cpp Impl 结构
struct QwenEngine::Impl {
    // ... 现有字段 ...
    
    // Batch slot management
    int max_batch_size = 1;  // 默认单 session
    bool batch_mode_enabled = false;
    std::unordered_map<uint64_t, int> request_to_slot;
    std::vector<int> free_slots;
};
```

#### Step 3.2: 实现 `allocate_batch_slots()`
```cpp
void QwenEngine::allocate_batch_slots(int max_batch_size) {
    if (max_batch_size < 1) {
        throw std::runtime_error("max_batch_size must be >= 1");
    }
    if (impl_->batch_mode_enabled) {
        throw std::runtime_error("batch slots already allocated");
    }
    
    impl_->max_batch_size = max_batch_size;
    impl_->batch_mode_enabled = (max_batch_size > 1);
    
    // 如果是单 session，不需要重新分配
    if (max_batch_size == 1) {
        return;
    }
    
    // 重新分配所有层的 KV cache 为 [max_batch_size, max_seq_len, ...]
    // TODO: 实现重新分配逻辑
    
    // 初始化 free_slots
    impl_->free_slots.clear();
    for (int i = 0; i < max_batch_size; ++i) {
        impl_->free_slots.push_back(i);
    }
}
```

#### Step 3.3: 修改 KV cache 分配逻辑
```cpp
// 在 QwenEngine 构造函数中
const int slot_count = impl_->max_batch_size;
const size_t cache_elements = slot_count * max_context * local_kv_heads * head_dim;
const std::vector<uint64_t> cache_shape = {
    static_cast<uint64_t>(slot_count),
    static_cast<uint64_t>(max_context),
    static_cast<uint64_t>(local_kv_heads),
    static_cast<uint64_t>(head_dim)
};
```

#### Step 3.4: 修改 attention kernel 调用
```cpp
// 单 session 模式（slot_id = 0）
qwen_append_kv_cache_f16(
    k_norm.f16_data(),
    v.f16_data(),
    layer.full.k_cache.f16_data() + slot_offset,  // 添加 slot offset
    layer.full.v_cache.f16_data() + slot_offset,
    rows, kv_heads, head_dim,
    position_offset, max_context
);

// slot_offset = slot_id * (max_context * kv_heads * head_dim)
```

### 向后兼容保证

1. **默认行为不变**：
   - 构造函数不调用 `allocate_batch_slots()` → `max_batch_size = 1`
   - 单 session 路径零开销（slot_id = 0, 无 offset 计算）

2. **现有 API 不变**：
   - `prefill()` / `decode_step()` 继续使用 slot 0
   - 单请求性能 ≤1.05× baseline

3. **渐进式启用**：
   - 用户必须显式调用 `allocate_batch_slots(N)` 启用批量模式
   - Python bindings 可选暴露批量 API

## 待办任务

### Task #3: 重构 KV cache 布局（进行中）
- [ ] Step 3.1: 添加 slot 管理字段
- [ ] Step 3.2: 实现 `allocate_batch_slots()`
- [ ] Step 3.3: 修改 KV cache 分配
- [ ] Step 3.4: 修改 attention kernel 调用（添加 slot offset）
- [ ] Step 3.5: 测试单 session 零回归

### Task #4: 添加 batch API
- [ ] 实现 `batch_prefill()`（FCFS 串行处理每个请求）
- [ ] 实现 `batch_decode_step()`（真正的批量 decode）
- [ ] 实现 `allocate_slot()` / `free_slot()`
- [ ] 实现 `supports_batching()`

### Task #5: 实现调度器
- [ ] 创建 `cpp_engine/core/simple_scheduler.hpp`
- [ ] 实现 FCFS 队列
- [ ] 实现调度循环 `step()`
- [ ] 完成时回调支持

### Task #6: 集成到 CppBackend
- [ ] 检测 `supports_batching()`
- [ ] 启动调度器后台线程
- [ ] 非阻塞提交 + 等待完成
- [ ] 更新 `BackendCapabilities.supports_batch = True`

### Task #7: 测试
- [ ] 单请求 parity
- [ ] 2/4/8 并发正确性
- [ ] Token-by-token 一致性

### Task #8: 性能验证
- [ ] 1/2/4/8 并发吞吐测量
- [ ] 目标：2 并发 ≥1.8×, 8 并发 ≥4×
- [ ] 单请求延迟 ≤1.05× baseline

## 关键设计决策

### 1. 为什么不做 paged attention？

**原因**：
- 固定 slot 更简单，kernel 改动最小
- PocketLLM 目标场景（2-8 并发）不需要 paged 的内存效率
- 可以后续按需添加（接口已预留）

**内存开销**：
- 8-slot TP2 FP16 KV: 8 × 2 × 64K × 32 heads × 128 dim × 2B = 8 GiB/rank
- 在 22 GiB 预算内可接受

### 2. 为什么 `batch_prefill()` 串行处理？

**原因**：
- 不同长度请求的混批需要 padding 或复杂的 ragged tensor
- 串行处理每个 prefill 简单且正确
- Decode 阶段才是吞吐瓶颈（占 70%+ 时间）

**未来优化**：
- Phase 4 可以添加 chunked mixed-length prefill

### 3. 单请求路径零开销如何保证？

**实现**：
- `max_batch_size = 1` 时不重新分配 KV cache
- `slot_id = 0` 时 `slot_offset = 0`（编译器优化掉）
- 现有 `prefill()` / `decode_step()` 内部调用批量 API with single request

## 风险和缓解

### 风险 1: KV cache 重分配破坏现有功能

**缓解**：
- 单 session 模式（默认）不触发重分配
- 逐步测试：先单 slot，再多 slot
- Token parity 验证：batched vs sequential

### 风险 2: 性能回归

**缓解**：
- 单请求路径保持不变（slot 0）
- Decode kernel 已接近带宽顶（477 GB/s），batching 不会更慢
- 实测 baseline 作为门控

### 风险 3: Prefix cache 与 batching 冲突

**缓解**：
- Phase 3.1 每个请求独立的 prefix cache（per-slot）
- Phase 4 再考虑跨请求共享（RadixAttention-like）

## 验收标准

### Correctness
- [ ] 单请求 token parity（batched vs original）
- [ ] 8 并发 token parity（每个请求独立正确）
- [ ] Prefix cache 命中率不降低

### Performance
- [ ] 单请求延迟 ≤ 1.05× baseline
- [ ] 2 并发吞吐 ≥ 1.8× 单请求
- [ ] 8 并发吞吐 ≥ 4.0× 单请求
- [ ] 内存占用 ≤ max_batch_size × 单请求

### Integration
- [ ] Python bindings 可用
- [ ] CppBackend 自动检测并启用
- [ ] HTTP server 并发请求自动合批

## 参考

- **架构文档**: `docs/vllm_sglang_comparison.md`
- **性能基线**: `qwen_tp2_64k_final_gap.md` (TP2 8K: 1434/30.9 tok/s)
- **内存预算**: 22 GB/rank (TP2/TP4)
- **现有 KV cache**: `qwen_engine.cpp:775-833`
