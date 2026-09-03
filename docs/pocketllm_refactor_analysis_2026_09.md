# PocketLLM 重构分析：与 vLLM/SGLang 对比及双后端统一设计

**日期**: 2026-09-03  
**基线**: PocketLLM master @ e59d5d3 (Phase 3.1–3.3 已合并)  
**作者**: 系统架构分析

---

## 执行摘要

PocketLLM 在 Phase 1–3 重构中已经建立了 **控制平面统一**（`pocketllm` 包）和 **cpp_engine 批处理 API 框架**（Phase 3.1–3.3）。当前状态：

- ✅ **控制平面统一完成**：`pocketllm.LLM`/`AsyncLLM` 提供统一的 Python API
- ✅ **双后端适配器就绪**：`CppBackend` 和 `TorchBackend` 实现 `EngineBackend` 协议
- ✅ **cpp_engine 批处理基础设施**：slot 化 KV cache、`batch_prefill`/`batch_decode_step` API
- ❌ **连续批处理调度器缺失**：两个后端都是串行执行（`supports_batch=False`）
- ❌ **请求级内存管理缺失**：无 PagedAttention/RadixCache 式的灵活分配

**核心差距**（vs vLLM/SGLang）：调度器层。物理后端已解耦，但缺乏统一的请求感知调度器来驱动批处理能力。

---

## 1. PocketLLM 当前架构（Phase 3 后）

### 1.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Control Plane: pocketllm package (4046 lines Python)       │
│  - LLM / AsyncLLM 公共 API                                  │
│  - OpenAI HTTP server (/v1/chat/completions, /metrics)      │
│  - EngineArgs 统一配置                                       │
│  - TP 进程监督器（自动启动 rank 0-N）                        │
└─────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  CppBackend     │            │  TorchBackend   │
│  (677 lines)    │            │  (397 lines)    │
│                 │            │                 │
│  - 序列化单会话  │            │  - 复用 src/    │
│  - 边界取消     │            │    server 队列  │
│  - TP worker    │            │  - 边界取消     │
│    入口         │            │                 │
└────────┬────────┘            └────────┬────────┘
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  cpp_engine     │            │  src/models     │
│  (212k lines    │            │  (30k lines     │
│   C++/CUDA)     │            │   PyTorch)      │
│                 │            │                 │
│  Phase 3.1-3.3: │            │  - MoE routing  │
│  - batch API    │            │  - Triton       │
│  - slot KV      │            │    kernels      │
│  - 未启用批处理  │            │  - GGUF loader  │
└─────────────────┘            └─────────────────┘
```

### 1.2 cpp_engine Phase 3 批处理基础设施

**已实现**（PR #121–#124）：

```cpp
// qwen_engine.hpp
struct QwenBatchedRequest {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    int slot_id;  // KV cache 插槽 ID
    int seq_len;
    bool finished;
    std::vector<int> generated_tokens;
};

class QwenEngine {
    // Phase 3.1: API 框架
    void allocate_batch_slots(int max_batch_size);
    int allocate_slot(uint64_t request_id);
    void free_slot(uint64_t request_id);
    
    // Phase 3.2: 批量前向
    QwenBatchPrefillResult batch_prefill(
        const std::vector<QwenBatchedRequest*>& requests);
    QwenBatchDecodeResult batch_decode_step(
        const std::vector<QwenBatchedRequest*>& requests);
    
    // Phase 3.3: slot 化位置追踪
    // 每个 slot 独立维护 position_/prefix_len_ 等状态
};
```

**KV cache 布局**（Phase 3.2）：

```cpp
// 旧单会话: [max_seq_len, n_layers, n_kv_heads, head_dim]
// 新 slot 化: [max_batch_size, max_seq_len, n_layers, ...]
//             ↑ slot_id        ↑ 序列内偏移
```

**关键限制**：

- `batch_prefill` 当前**串行处理**每个 request（逐一调用 `prefill()`）
- `batch_decode_step` 能并行 decode（所有 slot 在一次 kernel 调用中），但 `CppBackend` 适配器**未启用**
- `CppBackend.capabilities.supports_batch = False`：Python 层仍串行派发请求

---

## 2. 与 vLLM/SGLang 的功能差距矩阵

| 维度 | PocketLLM (Phase 3) | vLLM | SGLang |
|-----|---------------------|------|--------|
| **调度** | 串行 FCFS（适配器层） | 连续批处理 + 抢占 | 连续批处理 + prefix-aware |
| **内存管理** | 固定 slot KV cache | PagedAttention (block pool) | RadixAttention (trie) |
| **Prefix 共享** | 单会话 snapshot/restore | prefix_caching (有限) | 自动 prefix 树复用 |
| **请求迁移** | ❌ | ✅ Chunked prefill | ✅ |
| **Backend 抽象** | ✅ `EngineBackend` 协议 | ✅ Executor 接口 | ✅ ModelRunner 接口 |
| **多后端** | C++ (CUDA/Ascend) + PyTorch | CUDA/ROCm/CPU/TPU | CUDA/ROCm |
| **Tensor Parallel** | ✅ (自有 NCCL) | ✅ (Ray/torchrun) | ✅ (Ray) |
| **投机解码** | ✅ MTP/DSpark/DFlash2 | ✅ SpecDecodeWorker | ✅ 原生 |
| **量化** | **✅✅ FP8/FP4/Q2/IQ1M/TurboQuant** | INT8/FP8 | FP8/INT4 |
| **Observability** | `/metrics` (Prometheus) | Prometheus + Ray dashboard | Prometheus + custom |
| **多模态** | ❌ | ✅ LLaVA/Qwen-VL | ✅ 原生 vision |
| **Embeddings** | ❌ | ✅ | ✅ |
| **LoRA** | ❌ | ✅ | ✅ |
| **结构化输出** | ❌ | ✅ (outlines) | ✅ (FSM) |
| **性能** | **单请求延迟领先**（自定义 CUDA kernel） | 高吞吐（调度优势） | 高吞吐 + prefix 效率 |

**优势保持**：

- 量化支持最全（FP4/IQ1M/TurboQuant 是独有的）
- 单请求延迟最优（专用 kernel：GQA tensor core、DFlash2、Q2 MMQ）
- 模型特化优化（DeepSeek-V4/Qwen/MiniMax 专用路径）

**核心短板**：

1. **调度器**：无连续批处理，无法在高并发下保持吞吐
2. **多模态/LoRA/结构化输出**：全部缺失
3. **社区生态**：vLLM/SGLang 有大量第三方集成

---

## 3. 双后端统一的设计原则

### 3.1 已验证的分层原则

PocketLLM Phase 1–3 的核心设计决策：**控制平面统一，数据平面独立**。

```
统一层（pocketllm）:
  - 请求 ID、生命周期、取消、流式、OpenAI 协议
  - 配置校验（EngineArgs）
  - 指标暴露（/metrics）

独立层（cpp_engine / src.models）:
  - 物理 KV cache 布局
  - 量化/attention kernel
  - TP 通信协议（NCCL/Gloo/HCCL）
  - 调度器实现
```

**关键洞察**：不要强制两个后端共享 KV cache 物理格式或调度器实现。vLLM 的 `GPUExecutor` 和 `CPUExecutor` 也是各自实现调度的。

### 3.2 当前 `EngineBackend` 协议

```python
# pocketllm/api/backend.py
class EngineBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...
    
    def health(self) -> HealthStatus: ...
    
    def generate(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]: ...
    
    def stream(
        self, request: GenerationRequest
    ) -> Iterator[TokenEvent]: ...
    
    def cancel(self, request_id: str) -> bool: ...
    
    def prepare(self) -> None: ...  # TP rank 初始化
    def run_worker(self, on_ready) -> None: ...  # TP worker 循环
    def close(self) -> None: ...
```

**现状**：

- `generate(requests)` 接受 `Sequence[GenerationRequest]`，但两个后端都**串行处理**
- `BackendBase._request_lock` 序列化所有调用（保护单会话 mutable state）

### 3.3 缺失的调度器层

```
当前:
  HTTP request → pocketllm.engine.UnifiedEngine.generate()
                 → backend.generate([request])  # 单个请求
                 → 串行执行

理想:
  HTTP requests → ContinuousBatchScheduler.add_request()
                  ↓ (后台调度循环)
                  → prefill_batch = [req1, req2]
                  → backend.generate(prefill_batch)  # 真实批量
                  → decode_batch = [req3, req4, req5]
                  → backend.generate(decode_batch)   # 真实批量
```

**问题**：当前 `pocketllm/engine.py` 的 `UnifiedEngine` 只是一个薄包装器，没有调度逻辑。

---

## 4. 统一架构设计方案

### 4.1 目标

1. **最小侵入**：复用 Phase 1–3 的 `EngineBackend` 协议和适配器
2. **后端自治**：cpp_engine 和 PyTorch 各自实现调度（如果需要）
3. **渐进启用**：通过 `capabilities.supports_batch` 控制是否启用批处理
4. **性能无回归**：单请求路径不增加开销

### 4.2 三种部署模式

#### 模式 A：传统串行（当前默认）

```python
# capabilities.supports_batch = False
backend = CppBackend(...)  # 内部 _request_lock 串行
engine = UnifiedEngine(backend)
# 请求逐个处理，无调度器开销
```

**适用场景**：单用户、低并发、延迟敏感

#### 模式 B：后端内建调度器（推荐 cpp_engine）

```python
# capabilities.supports_batch = True
backend = CppBackend(..., enable_batching=True)
# 内部启动调度线程，自己管理 slot/queue
engine = UnifiedEngine(backend)
# 外部只负责 add_request，内部自动合批
```

**cpp_engine 实现** (Phase 3.4)：

```cpp
class QwenBatchScheduler {
    std::queue<QwenBatchedRequest*> waiting_queue_;
    std::vector<QwenBatchedRequest*> running_batch_;
    std::thread schedule_thread_;
    
    void schedule_loop() {
        while (running_) {
            // 1. Admit new requests from waiting_queue_
            // 2. Call batch_prefill for new admits
            // 3. Call batch_decode_step for running batch
            // 4. Handle completions
        }
    }
};
```

**优势**：

- 调度器在 C++ 内，微秒级延迟
- 直接访问 slot state，无 Python/C++ 跨境开销
- `CppBackend` 适配器变薄：只转发 `add_request` 到内部调度器

#### 模式 C：Python 统一调度器（适合 PyTorch 后端）

```python
# capabilities.supports_batch = True (backend reports)
scheduler = ContinuousBatchScheduler(backend)
engine = UnifiedEngine(backend, scheduler=scheduler)
# Python 调度器管理队列，定期调用 backend.generate(batch)
```

**实现**（参考 vLLM）：

```python
class ContinuousBatchScheduler:
    def __init__(self, backend: EngineBackend):
        self.backend = backend
        self.waiting: deque[ScheduledRequest] = deque()
        self.running: dict[str, ScheduledRequest] = {}
        self.max_batch_size = 8  # from backend
        
    def add_request(self, req: GenerationRequest) -> None:
        self.waiting.append(ScheduledRequest(req))
    
    def step(self) -> None:
        # 1. Admit waiting → running (if slots available)
        # 2. Separate prefill_batch / decode_batch
        # 3. backend.generate(prefill_batch)
        # 4. backend.generate(decode_batch)
        # 5. Check finish, free slots
```

**优势**：

- PyTorch 后端无需重写调度器（复用此 Python 实现）
- 调度策略易迭代（Python）
- 支持 prefix cache 查找、抢占等高级特性

**劣势**：

- Python 调度延迟 ~0.5–1ms（对 cpp_engine 的快速 decode 有影响）

### 4.3 推荐路线图

| Phase | 工作 | 时间 | 输出 |
|-------|------|------|------|
| **Phase 3.4** | cpp_engine 内建调度器 | 1 周 | `QwenBatchScheduler` 类，C++ 内批处理 |
| **Phase 3.5** | CppBackend 批处理适配 | 3 天 | `supports_batch=True`，性能验证 |
| **Phase 4.1** | Python 统一调度器 | 1 周 | `ContinuousBatchScheduler`，prefix cache |
| **Phase 4.2** | TorchBackend 批处理 | 5 天 | PyTorch 后端启用批处理 |
| **Phase 4.3** | PagedAttention/RadixCache | 2 周 | 灵活内存管理（可选） |
| **Phase 5** | 高级特性 | 持续 | 抢占、LoRA、多模态、结构化输出 |

---

## 5. 关键设计决策

### 5.1 调度器在 C++ 还是 Python？

**推荐**：**cpp_engine 用 C++，PyTorch 后端用 Python**。

**理由**：

- cpp_engine decode 单步 ~20–40 ms，Python 调度开销 0.5–1 ms = **2.5–5% 开销**
- PyTorch 后端 decode 单步 ~100–200 ms，Python 调度开销 <1% 无所谓
- C++ 调度器可以零拷贝访问 slot state，Python 需要跨 FFI 边界

**证据**：vLLM 的 C++ 引擎（如 TensorRT-LLM）也是自带调度器。

### 5.2 PagedAttention 还是固定 slot？

**推荐**：**短期保持固定 slot，长期可选 PagedAttention**。

**理由**：

- 固定 slot 简单，cpp_engine Phase 3.2 已实现
- PagedAttention 复杂（block 管理、碎片化、kernel 修改），收益主要在**高并发 + 长序列**场景
- PocketLLM 的优势在**单请求低延迟**，不是高并发吞吐

**证据**：

- vLLM 的 PagedAttention 有 ~10–20% 内存开销（block metadata）
- 对于 batch_size=8、max_seq_len=8192 的场景，固定 slot 无内存浪费

**可选升级路径**：

- Phase 4.3 实现 PagedAttention 作为 **opt-in 特性**（`enable_paged_attention=True`）
- 默认保持固定 slot 以保护性能

### 5.3 Prefix 共享：Snapshot 还是 RadixCache？

**当前 cpp_engine**：每个 slot 独立维护 prefix snapshot（PR #122）。

**推荐**：**保持现状，长期添加 RadixCache 作为可选项**。

**理由**：

- SGLang 的 RadixCache 复杂（trie 结构、LRU 驱逐、跨请求匹配）
- cpp_engine 的 snapshot 机制已经在单会话内高效工作（Qwen 65K context 验证）
- 跨请求 prefix 共享收益取决于**工作负载**（chat 历史、few-shot prompt 是否重复）

**证据**：

- vLLM 的 `prefix_caching` 也是 opt-in，不是默认（因为有元数据开销）

### 5.4 `EngineBackend` 协议需要修改吗？

**推荐**：**不修改**。

当前 `generate(requests: Sequence)` 已经支持批量输入。只需：

1. 后端内部判断 `len(requests) > 1` 时走批处理路径
2. `capabilities.supports_batch` 控制外部调度器是否合并请求

**无需增加新方法**，保持向后兼容。

---

## 6. 实施细节：cpp_engine 批处理调度器

### 6.1 核心类设计

```cpp
// qwen_batch_scheduler.hpp
class QwenBatchScheduler {
public:
    explicit QwenBatchScheduler(QwenEngine* engine, int max_batch_size);
    ~QwenBatchScheduler();
    
    // Thread-safe request submission
    uint64_t submit_request(
        const std::vector<int>& prompt_tokens,
        const QwenBatchSamplingParams& sampling);
    
    // Non-blocking result poll (or blocking with timeout)
    bool poll_result(uint64_t request_id, GenerationResult* out);
    
    // Cancel a pending/running request
    void cancel_request(uint64_t request_id);
    
private:
    QwenEngine* engine_;
    int max_batch_size_;
    
    // Request queues
    std::mutex queue_mutex_;
    std::queue<QwenBatchedRequest*> waiting_queue_;
    std::unordered_map<int, QwenBatchedRequest*> slot_to_request_;
    
    // Scheduling thread
    std::thread schedule_thread_;
    std::atomic<bool> running_{true};
    
    void schedule_loop();
    void admit_requests();
    void run_prefill_batch();
    void run_decode_batch();
    void handle_completions();
};
```

### 6.2 调度循环伪代码

```cpp
void QwenBatchScheduler::schedule_loop() {
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            // 1. Admit new requests if slots available
            while (!waiting_queue_.empty() && 
                   slot_to_request_.size() < max_batch_size_) {
                auto* req = waiting_queue_.front();
                waiting_queue_.pop();
                
                int slot_id = engine_->allocate_slot(req->request_id);
                if (slot_id < 0) break;  // no slots
                
                req->slot_id = slot_id;
                slot_to_request_[slot_id] = req;
            }
            
            // 2. Separate prefill vs decode
            std::vector<QwenBatchedRequest*> prefill_batch;
            std::vector<QwenBatchedRequest*> decode_batch;
            for (auto& [slot_id, req] : slot_to_request_) {
                if (req->seq_len == 0) {
                    prefill_batch.push_back(req);
                } else {
                    decode_batch.push_back(req);
                }
            }
            
            // 3. Run prefill (currently serial in engine)
            if (!prefill_batch.empty()) {
                auto result = engine_->batch_prefill(prefill_batch);
                for (size_t i = 0; i < prefill_batch.size(); ++i) {
                    prefill_batch[i]->last_token = result.results[i].top_token;
                    prefill_batch[i]->seq_len = prefill_batch[i]->prompt_tokens.size();
                }
            }
            
            // 4. Run decode (parallel in engine)
            if (!decode_batch.empty()) {
                auto result = engine_->batch_decode_step(decode_batch);
                for (size_t i = 0; i < decode_batch.size(); ++i) {
                    int token = result.next_tokens[i];
                    decode_batch[i]->generated_tokens.push_back(token);
                    decode_batch[i]->last_token = token;
                    decode_batch[i]->finished = result.finished[i];
                }
            }
            
            // 5. Handle completions
            std::vector<int> to_free;
            for (auto& [slot_id, req] : slot_to_request_) {
                if (req->finished || 
                    req->generated_tokens.size() >= req->sampling.max_new_tokens) {
                    // Notify result (via callback or result map)
                    notify_completion(req);
                    to_free.push_back(slot_id);
                }
            }
            for (int slot_id : to_free) {
                engine_->free_slot(slot_to_request_[slot_id]->request_id);
                delete slot_to_request_[slot_id];
                slot_to_request_.erase(slot_id);
            }
        }
        
        // Small sleep to avoid busy loop
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
```

### 6.3 Python 绑定（pybind11）

```cpp
// cpp_engine/python/qwen_bindings.cpp
PYBIND11_MODULE(pocketllm_cpp, m) {
    // ... existing bindings ...
    
    py::class_<QwenBatchScheduler>(m, "QwenBatchScheduler")
        .def(py::init<QwenEngine*, int>())
        .def("submit_request", &QwenBatchScheduler::submit_request,
             py::call_guard<py::gil_scoped_release>())
        .def("poll_result", &QwenBatchScheduler::poll_result,
             py::call_guard<py::gil_scoped_release>())
        .def("cancel_request", &QwenBatchScheduler::cancel_request);
}
```

### 6.4 CppBackend 适配器修改

```python
# pocketllm/backends/cpp_backend.py
class CppBackend(BackendBase):
    def __init__(self, args: EngineArgs, ...):
        super().__init__()
        self._engine = self._construct_engine()
        
        # Phase 3.4: 启用批处理调度器
        if args.backend_options.get("enable_batching", False):
            max_batch_size = args.backend_options.get("max_batch_size", 8)
            self._scheduler = self._native.QwenBatchScheduler(
                self._engine, max_batch_size)
            self._batching_enabled = True
        else:
            self._scheduler = None
            self._batching_enabled = False
        
        self._ready = True
    
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="cpp",
            supports_batch=self._batching_enabled,  # 动态启用
            # ... 其他不变 ...
        )
    
    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if not self._batching_enabled:
            # 旧路径：串行执行
            return self._generate_serial(requests)
        
        # 新路径：提交到内部调度器
        request_ids = []
        for req in requests:
            prompt_ids = self._prompt_ids(req)
            sampling = self._native.QwenBatchSamplingParams()
            sampling.max_new_tokens = req.sampling_params.max_tokens
            
            req_id = self._scheduler.submit_request(prompt_ids, sampling)
            request_ids.append(req_id)
        
        # Poll for results (blocking with timeout)
        results = []
        for req_id in request_ids:
            result = self._native.GenerationResult()
            if self._scheduler.poll_result(req_id, result, timeout_ms=30000):
                results.append(self._convert_result(result))
            else:
                raise TimeoutError(f"request {req_id} timed out")
        
        return results
```

---

## 7. 性能预测与验证

### 7.1 预期收益（cpp_engine 批处理启用后）

**单请求**（batch_size=1）：

- 延迟无变化（调度器快速路径 ~0.1 ms）
- 吞吐无变化

**2 并发**：

- 理想吞吐：1.9× (decode kernel 复用)
- 实际预期：1.7–1.8× (调度器开销 + prefill 串行)

**4 并发**：

- 理想吞吐：3.5×
- 实际预期：3.0–3.2×

**8 并发**：

- 理想吞吐：6.0×
- 实际预期：4.5–5.5× (受限于 GPU 利用率上限)

### 7.2 验证基准

```bash
# 单请求延迟（无回归）
python scripts/bench_single_request_latency.py \
  --backend cpp --enable-batching

# 并发吞吐
python scripts/bench_concurrent_throughput.py \
  --backend cpp --enable-batching \
  --num-requests 2,4,8 --max-tokens 128

# 对比 PyTorch 后端
python scripts/bench_concurrent_throughput.py \
  --backend torch --num-requests 2,4,8
```

### 7.3 性能目标

| Metric | 目标 | 门槛 |
|--------|------|------|
| 单请求延迟 | ≤1.05× 基线 | ≤1.10× |
| 2 并发吞吐 | ≥1.7× 基线 | ≥1.5× |
| 4 并发吞吐 | ≥3.0× 基线 | ≥2.5× |
| 8 并发吞吐 | ≥4.5× 基线 | ≥3.5× |
| 队列延迟 (p99) | ≤50 ms | ≤100 ms |

---

## 8. 与 vLLM/SGLang 的长期对标路线

### 8.1 短期目标（Q4 2026）

- ✅ Phase 3.4–3.5：cpp_engine 批处理调度器
- ✅ Phase 4.1–4.2：Python 统一调度器 + PyTorch 批处理
- 🎯 **达成**：2–8 并发吞吐与 vLLM 同档次
- 🎯 **保持**：单请求延迟领先（自定义 kernel）

### 8.2 中期目标（Q1–Q2 2027）

- PagedAttention（可选，高并发场景）
- RadixCache（可选，高 prefix 重复场景）
- 基础多模态（LLaVA/Qwen-VL）
- 结构化输出（FSM-based）

### 8.3 长期目标（2027 下半年）

- LoRA 多路复用
- Pipeline Parallel
- Ascend backend 功能对等（当前缺 DFlash2/DSpark）
- 企业级部署特性（加密推理、审计日志）

### 8.4 永久差异化（不追求）

- **社区生态**：vLLM/SGLang 有 HuggingFace/LangChain 深度集成，PocketLLM 专注性能
- **模型覆盖广度**：vLLM 支持 100+ 模型，PocketLLM 只优化核心生产模型（DeepSeek-V4/Qwen/MiniMax）
- **分布式调度**：vLLM 用 Ray，PocketLLM 用自有 TP 监督器（更轻量）

---

## 9. 关键风险与缓解

### 9.1 风险：批处理破坏单请求性能

**缓解**：

- 快速路径：batch_size=1 时跳过调度器，直接调用 `engine->prefill/decode`
- 性能门禁：CI 中加单请求延迟回归测试（≤1.05× 基线）
- 默认关闭：`enable_batching=False` 直到验证完毕

### 9.2 风险：C++ 调度器复杂度

**缓解**：

- MVP 先实现 FCFS，无抢占、无 prefix 匹配
- 充分测试：单元测试（slot 分配/释放）+ 集成测试（并发正确性）
- 参考 vLLM：复用已验证的调度逻辑（用 C++ 重写）

### 9.3 风险：PyTorch 后端性能跟不上

**缓解**：

- 接受现实：PyTorch 后端定位是**兼容性后备**，不是性能主力
- 优先保证 cpp_engine 性能
- 长期：PyTorch 后端可以走 Triton 优化路径（但仍慢于自定义 CUDA）

---

## 10. 行动计划

### 10.1 立即开始（本周）

1. **Review Phase 3.1–3.3 代码**
   - 验证 `batch_decode_step` 确实能并行（不是伪批处理）
   - 确认 slot KV cache 布局正确

2. **设计 `QwenBatchScheduler` 接口**
   - 确定 `submit_request` / `poll_result` 语义
   - 确定结果通知机制（callback vs polling）

3. **编写性能基准脚本**
   - `bench_single_request_latency.py`
   - `bench_concurrent_throughput.py`

### 10.2 Phase 3.4（第 1 周）

- [ ] 实现 `QwenBatchScheduler` 类（FCFS 调度循环）
- [ ] 添加 pybind11 绑定
- [ ] 单元测试（slot 分配/并发请求/取消）

### 10.3 Phase 3.5（第 2 周）

- [ ] `CppBackend` 适配器支持批处理模式
- [ ] 集成测试（2/4/8 并发正确性）
- [ ] 性能验证（对比基线，确认无回归）
- [ ] 文档更新（用户指南、性能数字）

### 10.4 Phase 4.1–4.2（第 3–4 周）

- [ ] 实现 Python `ContinuousBatchScheduler`
- [ ] `TorchBackend` 启用批处理
- [ ] 跨后端一致性测试

### 10.5 长期（Q4 2026 – Q1 2027）

- [ ] PagedAttention（可选）
- [ ] RadixCache（可选）
- [ ] 多模态支持
- [ ] 结构化输出

---

## 11. 总结

### 11.1 当前状态（Phase 3 完成）

✅ **已完成的基础设施**：

- 控制平面统一（`pocketllm` 包）
- 双后端适配器（`CppBackend` / `TorchBackend`）
- cpp_engine slot 化 KV cache 和批处理 API

❌ **缺失的关键组件**：

- 连续批处理调度器（C++ 或 Python）
- 请求感知内存管理（可选 PagedAttention）

### 11.2 核心差距（vs vLLM/SGLang）

PocketLLM 的短板是**调度器**，不是后端抽象。物理后端已经解耦良好，只需在控制平面添加调度层。

### 11.3 推荐方案

**双调度器架构**：

- **cpp_engine**：C++ 内建调度器（低延迟，适合高性能场景）
- **PyTorch**：Python 统一调度器（易迭代，适合兼容性场景）

通过 `capabilities.supports_batch` 和 `enable_batching` 配置灵活切换。

### 11.4 最终目标

- **2–8 并发吞吐**：达到 vLLM 同档次
- **单请求延迟**：保持领先（自定义 kernel 优势）
- **量化支持**：继续领先（FP4/IQ1M/TurboQuant）
- **生产就绪**：metrics、监控、文档齐全

---

**文档版本**: v1.0  
**作者**: PocketLLM 架构分析  
**最后更新**: 2026-09-03  
**下一步**: Phase 3.4 实施计划
