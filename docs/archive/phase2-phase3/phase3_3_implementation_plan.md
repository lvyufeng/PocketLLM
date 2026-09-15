# Phase 3.3 Implementation Plan: Slot ID Threading and Batch API Integration

## 目标

将 `slot_id` 从全局状态（`current_slot_id`）改为显式参数传递，实现线程安全的多 slot 管理，并集成到 batch API 中。

## 当前问题

Phase 3.2 使用全局 `current_slot_id`：

```cpp
struct QwenEngine::Impl {
    int current_slot_id = 0;  // Global state - not thread-safe!
};

void full_attention(...) {
    const int slot_id = current_slot_id;  // Read from global
    const size_t slot_offset = kv_slot_offset_elements(slot_id, ...);
    // Use slot_offset in kernel calls
}
```

**问题**：
- ❌ 不是线程安全的
- ❌ 难以追踪哪个请求使用哪个 slot
- ❌ 测试和调试困难

## 解决方案：显式参数传递

### 策略

采用自底向上的修改策略：
1. 从最底层的 `full_attention()` 开始添加 `slot_id` 参数
2. 逐层向上传递到 `run_chunk()` → `prefill()` / `decode_step()`
3. 最后在 batch API 中调用时传递 `req->slot_id`

### 实施步骤

## Step 1: 修改 full_attention() 添加 slot_id 参数

```cpp
// Before
void full_attention(DeviceLayer& layer, const uint16_t* gate,
                   uint16_t* merged, int rows, int position_offset);

// After
void full_attention(DeviceLayer& layer, const uint16_t* gate,
                   uint16_t* merged, int rows, int position_offset,
                   int slot_id);  // New parameter
```

**修改内容**：
```cpp
void full_attention(..., int slot_id) {
    // Remove: const int slot_id = current_slot_id;
    const size_t slot_offset = kv_slot_offset_elements(slot_id, kv_heads, head_dim);
    // Rest remains the same
}
```

## Step 2: 修改 run_chunk() 添加 slot_id 参数

```cpp
// Before
void run_chunk(const int* tokens, int start_row, int row_count,
               uint16_t* logits_out);

// After
void run_chunk(const int* tokens, int start_row, int row_count,
               uint16_t* logits_out, int slot_id);  // New parameter
```

**调用 full_attention()**：
```cpp
void run_chunk(..., int slot_id) {
    // ... existing code ...
    
    if (has_full_attention) {
        full_attention(layer, gate.data(), merged.data(), 
                      row_count, position, slot_id);  // Pass slot_id
    }
    
    // ... rest of code ...
}
```

## Step 3: 修改 prefill() 和 decode_step() 添加 slot_id 参数

```cpp
// Before
QwenForwardResult prefill(const std::vector<int>& tokens);
QwenForwardResult decode_step(int token);

// After
QwenForwardResult prefill(const std::vector<int>& tokens, int slot_id = 0);
QwenForwardResult decode_step(int token, int slot_id = 0);
```

**使用默认参数保持向后兼容**：
- 单 session 模式：`prefill(tokens)` → slot_id = 0
- Multi-slot 模式：`prefill(tokens, req->slot_id)` → 显式传递

**实现**：
```cpp
QwenForwardResult QwenEngine::prefill(const std::vector<int>& tokens, int slot_id) {
    // ... existing validation ...
    
    for (size_t chunk_start = 0; chunk_start < tokens.size(); chunk_start += chunk_size) {
        int chunk_tokens = std::min(chunk_size, 
                                   static_cast<int>(tokens.size() - chunk_start));
        impl_->run_chunk(tokens.data() + chunk_start, position_, 
                        chunk_tokens, logits.data(), slot_id);  // Pass slot_id
        position_ += chunk_tokens;
    }
    
    // ... rest of code ...
}

QwenForwardResult QwenEngine::decode_step(int token, int slot_id) {
    // ... existing code ...
    impl_->run_chunk(&token, position_, 1, logits.data(), slot_id);  // Pass slot_id
    position_++;
    // ... rest of code ...
}
```

## Step 4: 更新头文件声明

```cpp
// qwen_engine.hpp
class QwenEngine {
public:
    // ... existing methods ...
    
    // Single-session API (backward compatible)
    QwenForwardResult prefill(const std::vector<int>& tokens, int slot_id = 0);
    QwenForwardResult decode_step(int token, int slot_id = 0);
    
    // Batch API (Phase 3.1)
    QwenBatchPrefillResult batch_prefill(const std::vector<QwenBatchedRequest*>& requests);
    QwenBatchDecodeResult batch_decode_step(const std::vector<QwenBatchedRequest*>& requests);
};
```

## Step 5: 修改 batch API 使用 slot_id

```cpp
QwenBatchPrefillResult QwenEngine::batch_prefill(
    const std::vector<QwenBatchedRequest*>& requests) {
    
    QwenBatchPrefillResult result;
    
    for (QwenBatchedRequest* req : requests) {
        if (req->finished) continue;
        
        // Save and restore position for isolation
        const int saved_position = position_;
        position_ = req->seq_len;
        
        // Use req->slot_id for this request
        QwenForwardResult fwd_result = prefill(req->prompt_tokens, req->slot_id);
        
        req->seq_len = position_;
        req->last_token = fwd_result.next_token;
        req->last_result = fwd_result;
        
        result.last_tokens.push_back(fwd_result.next_token);
        result.prompt_lengths.push_back(req->prompt_tokens.size());
        
        // Restore position
        position_ = saved_position;
    }
    
    return result;
}

QwenBatchDecodeResult QwenEngine::batch_decode_step(
    const std::vector<QwenBatchedRequest*>& requests) {
    
    QwenBatchDecodeResult result;
    
    for (QwenBatchedRequest* req : requests) {
        if (req->finished) {
            result.next_tokens.push_back(req->last_token);
            result.finished.push_back(true);
            continue;
        }
        
        // Save and restore position
        const int saved_position = position_;
        position_ = req->seq_len;
        
        // Use req->slot_id for this request
        QwenForwardResult fwd_result = decode_step(req->last_token, req->slot_id);
        
        req->seq_len = position_;
        req->last_token = fwd_result.next_token;
        req->generated_tokens.push_back(fwd_result.next_token);
        
        // Check completion
        bool is_finished = (req->generated_tokens.size() >= 
                           static_cast<size_t>(req->sampling.max_new_tokens));
        req->finished = is_finished;
        
        result.next_tokens.push_back(fwd_result.next_token);
        result.finished.push_back(is_finished);
        
        // Restore position
        position_ = saved_position;
    }
    
    return result;
}
```

## Step 6: 移除 current_slot_id

```cpp
// qwen_engine.cpp Impl
struct QwenEngine::Impl {
    // ... existing fields ...
    
    // REMOVED: int current_slot_id = 0;  // No longer needed
    
    // ... rest of code ...
};
```

## 验证清单

### 编译验证
- [ ] 所有修改的函数签名匹配
- [ ] 默认参数正确应用
- [ ] 无编译错误或警告

### 功能验证
- [ ] 单 session 模式：`prefill(tokens)` 使用 slot_id=0
- [ ] 单 session parity：与 Phase 3.2 结果一致
- [ ] Batch API：每个请求使用自己的 slot_id

### 代码质量
- [ ] 无全局状态（`current_slot_id` 已移除）
- [ ] 线程安全（每个调用显式传递 slot_id）
- [ ] 向后兼容（默认参数 `slot_id=0`）

## 测试计划

### Test 1: 单 Session Parity

```python
def test_single_session_parity():
    """验证 Phase 3.3 与 Phase 3.2 结果一致"""
    engine = QwenEngine(config, options)
    
    prompt = [1, 2, 3, 4, 5]
    result1 = engine.prefill(prompt)       # slot_id=0 (default)
    result2 = engine.prefill(prompt, 0)    # slot_id=0 (explicit)
    
    assert result1.next_token == result2.next_token
```

### Test 2: Slot 隔离性

```python
def test_slot_isolation():
    """验证不同 slot 互不干扰"""
    engine = QwenEngine(config, options)
    engine.allocate_batch_slots(2)
    
    prompt_a = [1, 2, 3]
    prompt_b = [4, 5, 6]
    
    # 在 slot 0 中生成
    result_a = engine.prefill(prompt_a, slot_id=0)
    token_a1 = engine.decode_step(result_a.next_token, slot_id=0).next_token
    
    # 在 slot 1 中生成
    result_b = engine.prefill(prompt_b, slot_id=1)
    token_b1 = engine.decode_step(result_b.next_token, slot_id=1).next_token
    
    # 再次在 slot 0 中生成（应该延续之前的状态）
    token_a2 = engine.decode_step(token_a1, slot_id=0).next_token
    
    # slot 0 和 slot 1 应该生成不同的 tokens
    assert token_a1 != token_b1
```

### Test 3: Batch API 正确性

```python
def test_batch_api_with_slots():
    """验证 batch API 使用正确的 slot_id"""
    engine = QwenEngine(config, options)
    engine.allocate_batch_slots(2)
    
    req1 = QwenBatchedRequest(
        request_id=1,
        prompt_tokens=[1, 2, 3],
        slot_id=0,
        sampling=default_sampling
    )
    
    req2 = QwenBatchedRequest(
        request_id=2,
        prompt_tokens=[4, 5, 6],
        slot_id=1,
        sampling=default_sampling
    )
    
    # Batch prefill
    result = engine.batch_prefill([req1, req2])
    assert len(result.last_tokens) == 2
    
    # Batch decode
    for _ in range(10):
        result = engine.batch_decode_step([req1, req2])
        assert len(result.next_tokens) == 2
```

## 向后兼容性

**单 session 代码无需修改**：
```cpp
// Phase 3.2 code
QwenForwardResult result = engine.prefill(tokens);
int next = engine.decode_step(result.next_token).next_token;

// Phase 3.3 code (same!)
QwenForwardResult result = engine.prefill(tokens);  // slot_id=0 (default)
int next = engine.decode_step(result.next_token).next_token;  // slot_id=0 (default)
```

**Batch API 使用显式 slot_id**：
```cpp
// Phase 3.3
req->slot_id = engine.allocate_slot(req->request_id);
QwenForwardResult result = engine.prefill(req->prompt_tokens, req->slot_id);
```

## 实施顺序

1. ✅ Step 1: 修改 `full_attention()` 添加 `slot_id` 参数
2. ✅ Step 2: 修改 `run_chunk()` 添加 `slot_id` 参数
3. ✅ Step 3: 修改 `prefill()` 和 `decode_step()` 添加 `slot_id` 参数
4. ✅ Step 4: 更新头文件声明
5. ✅ Step 5: 修改 batch API 使用 `req->slot_id`
6. ✅ Step 6: 移除 `current_slot_id` 全局状态
7. ⏳ Step 7: 编译测试
8. ⏳ Step 8: 功能测试

**预计时间**: 2-3 小时

---

**开始实施**: Phase 3.3 Step 1
