# PocketLLM 针对老硬件的功能规划

**目标硬件**: 2080Ti (SM75, 22GB, PCIe), Ascend 910A (1st gen, 32GB HBM)  
**分析时间**: 2026-09-11  
**当前状态**: Phase 3 (paged KV 已默认开启)

---

## 一、基础能力补齐（必需项）

### 1.1 动态调度器与真正的 Continuous Batching

**现状**（2026-09-12 复核源码后更正）：
- ✅ Slot 化基础设施已就绪（KV cache/position/prefix cache/snapshots 已 per-slot）
- ✅ Batching kernel 已实现
- ✅ **真批处理 forward 已就位**：`batch_decode_step` → `run_batched_decode`
  一次 forward 覆盖整批（`qwen_engine.cpp:3829,3790`），不是逐请求循环
- ✅ **waiting 队列与准入控制已存在**：`BatchScheduler::admit_requests()` +
  基于 `worst_case_blocks` 的 block 预算（`batch_scheduler.hpp:174,214,217`）
- ✅ per-request 采样参数已按行生效；`openai_server.hpp` 默认 `max_batch_size=8`
- ⚠️ **并发吞吐从未实测**——本节验收标准（4 并发 ≥ 3×）是否已达成属未知
- ❌ 与投机解码（MTP/DSpark/DFlash2）及量化 KV 互斥，两者都被显式拒绝
- ❌ 没有抢占与优先级（FCFS，block 分配后不回收）

**因此本项的首要任务是实测而非实现**：先跑 `scripts/run_qwen_concurrent_tp4.py`
拿到 serial/2/4/8 的数字，再决定剩余工作量。

**为什么重要（老硬件视角）**：
- **2080Ti PCIe 带宽受限**: 22GB 显存装不下大 batch 的激活，但可以装 4-8 个请求的 KV cache
- **提高 GPU 利用率**: 单请求 decode 只有 ~45 tok/s，GPU 利用率已 96%，但 prefill 有大量空档
- **多租户场景**: 个人部署的 2080Ti 往往服务多个本地客户端（IDE/浏览器插件/命令行）

**实现优先级**: **P0 (Phase 4 核心任务)**

**设计要点**：
- 借鉴 vLLM 的三队列架构（waiting/running/finished）
- 调度约束基于 **KV cache 容量 + token 预算**，不是固定 batch size
- Prefill 优先（短请求快速响应），decode 填充空闲 slot
- 初期不做 preemption（简化实现），队列满时直接拒绝新请求
- **老硬件特化**: 动态调整 `max_num_running_reqs`（2080Ti 建议 4-8，910A 建议 8-16）

**收益预估**：
- 吞吐量 2-4× (多请求并发)
- 平均延迟下降（短请求不再排队等长请求）

---

### 1.2 Prefix Caching（跨请求 KV 共享）

**现状**：
- ✅ Paged KV 已默认开启
- ✅ Block 池化已实现
- ❌ 没有 block hash 和跨请求共享机制
- ❌ Prefix cache 只在单请求内复用（session 内）

**为什么重要（老硬件视角）**：
- **节省 prefill 时间**: 2080Ti prefill 1500-1800 tok/s，长 system prompt (2K tokens) 需要 ~1.1s
- **多轮对话场景**: 个人助手每轮都重复发整个历史，没有 prefix cache 每轮都重算
- **Few-shot prompting**: 相同 demonstrations 在多个请求间共享，只计算一次

**实现优先级**: **P0 (Phase 4)**

**设计要点**：
- 为每个 KV block 计算 hash（prompt tokens 的内容哈希）
- 在调度器中维护全局 `hash -> block_id` 映射
- 新请求 prefill 时先查表，命中的 block 直接复用
- LRU 淘汰策略（KV cache 满时优先淘汰未命中的 block）
- **老硬件特化**: 
  - 2080Ti 22GB 显存紧张，prefix cache 占比不宜超过 30%（~6.6GB）
  - 910A 32GB 可以更激进，50% 给 prefix cache（~16GB）

**收益预估**：
- 多轮对话：首轮后的 prefill 时间降至 ~0（只计算新增部分）
- 共享 system prompt 的批量推理：prefill 加速 2-10×（取决于前缀长度）

---

### 1.3 OpenAI API 语义完整性

**现状**：
- ✅ `/v1/chat/completions` 基础功能可用
- ✅ Streaming (SSE) 已支持
- ❌ 缺失 tools/function calling
- ✅ logprobs 已实现（2026-09-15）：sampler 在 decode 时对整份词表做 log-softmax，逐位置返回
  采样 token 自身的 logprob 与 top-N 候选（chat 用 `top_logprobs`，`/v1/completions` 用
  `logprobs` 计数）；流式请求拒绝，投机解码与 Ascend 后端声明为不支持
- ✅ multiple choices (n>1) 已实现（2026-09-15）：native C++ server 为每个 choice 提交一个独立的
  scheduler 请求，逐 choice 的 seed / grammar / KV，流式响应交错输出；上限 128
- ❌ 缺失 JSON mode / structured output

**为什么重要（老硬件视角）**：
- **生态兼容性**: 个人部署的 2080Ti 往往对接现有工具链（LangChain/Cursor/Continue）
- **无需改客户端代码**: 直接替换 OpenAI endpoint，零成本切换

**实现优先级**: **P1 (Phase 4-5)**

**设计要点**：
- **Tools/function calling**: 在 Python 控制平面实现（调度器触发函数调用，返回 tool_calls）
- **Logprobs**: ~~在 C++ decode 时记录 top-k logits~~ 已实现：不记录 top-k，而是对整份词表算
  log-softmax（`vocab_logsumexp_rows` + 逐行采样 kernel），因此概率是真实归一化概率；代价是
  每个 decode step 多一次全词表归约，只在请求要求时开启
- **Multiple choices (n>1)**: 在调度器中为同一 prompt 创建 n 个独立 slot，并行采样
- **JSON mode**: 简化版可以用 post-filter（生成后校验），完整版需要 FSM token filtering

**收益预估**：
- 生态兼容性大幅提升
- 可以对接 Cursor/Continue/Open WebUI 等主流客户端

---

### 1.4 多模态支持（Qwen-VL）

**现状**：
- ❌ 当前完全不支持
- ✅ Qwen-VL 模型架构与 Qwen 文本模型类似（主要是多了 vision encoder）

**为什么重要（老硬件视角）**：
- **2080Ti 边缘场景**: 本地 OCR/图像理解，不依赖云服务
- **Ascend 910A 有 32GB 显存**: 可以跑 Qwen2-VL-7B 的 vision encoder

**实现优先级**: **P2 (Phase 5-6)**

**设计要点**：
- 先支持 Qwen2-VL-7B（较小，老硬件能跑）
- Vision encoder 可以在 CPU 或单独一张 GPU 上跑（与 LLM 分离）
- 图像 token 作为额外的 prompt 序列输入
- **老硬件特化**: 
  - 2080Ti: 图像压缩到 256 tokens 以内（减少 prefill 开销）
  - 910A: 可以更激进，512-1024 tokens

**收益预估**：
- 打开本地多模态应用场景（截图问答/文档理解）

---

## 二、老硬件针对性优化（特色功能）

### 2.1 CPU Offloading with Prefetch Pipeline

**动机**：
- **2080Ti 只有 22GB**: 跑 27B FP8 勉强够，70B 完全装不下
- **Ascend 910A 32GB**: 跑 70B FP4 也装不下
- **PCIe/HBM 带宽**: 2080Ti PCIe 3.0 x16 ~16 GB/s，910A HBM2 ~1.2 TB/s（片上）

**现有方案的问题**：
- vLLM/SGLang 都假设权重常驻 GPU，不支持 CPU offloading
- llama.cpp 有 CPU offloading 但没有 prefetch，每层都同步等待

**PocketLLM 优势**：
- ✅ 已有 expert CPU-GPU 混合路径（GLM/Qwen4-Exp）
- ✅ H2D 拷贝已优化（pinned staging）

**设计方案**：
- **权重分层**: 
  - 前 N 层常驻 GPU（prefill/decode 都快）
  - 后 M 层放 CPU，按需 H2D
- **Prefetch 流水线**:
  - 第 i 层计算时，异步启动第 i+1 层权重 H2D
  - 用两个 CUDA stream：`compute_stream` 和 `h2d_stream`
  - 计算和拷贝并行，hide H2D latency
- **自适应边界**:
  - 根据显存占用动态调整 N（多少层常驻 GPU）
  - Prefill 时可以多放几层（激活小），decode 时少放（batch 大）

**实现优先级**: **P1 (Phase 4-5)**

**老硬件收益**：
- 2080Ti 跑 70B FP4: 前 16 层 GPU（~10GB）+ 后 64 层 CPU offload
  - Prefill: ~200 tok/s（受 H2D 带宽限制）
  - Decode: ~5 tok/s（每层 ~200ms，其中 H2D ~100ms + 计算 ~100ms）
- 910A 跑 70B Q2: 前 24 层 GPU（~15GB）+ 后 56 层 CPU
  - HBM2 带宽更高，decode ~8-10 tok/s

---

### 2.2 Aggressive Quantization（FP4/Q2/Q3）

**现状**：
- ✅ PocketLLM 已支持 FP4/NVFP4/GGUF Q2/Q3
- ✅ Token parity 已验证
- ⚠️ 但在 SM75 上 FP4 慢于 FP8（0.45-0.65×）

**为什么仍然重要（老硬件视角）**：
- **显存容量 > 速度**: 22GB 的 2080Ti 跑 70B 模型，只能靠 FP4/Q2
- **Ascend 910A 没有 FP8**: CANN 原生不支持 FP8，FP4 是最接近的选项

**设计改进**：
- **Hybrid quantization**: 
  - Prefill 瓶颈层用 FP8/INT8（速度优先）
  - Decode 瓶颈层用 FP4/Q2（显存优先）
  - 动态切换（根据 batch size）
- **Per-layer dtype 配置**:
  - Attention 用 FP8（compute-bound）
  - MLP 用 FP4（memory-bound）
- **INT4 Tensor Core on SM75**:
  - 研究 llama.cpp 的 MMQ kernel（已证明在 SM75 上比 DP4A 快）
  - 移植到 PocketLLM Q2/Q4 路径

**实现优先级**: **P1 (Phase 4-5)**

**老硬件收益**：
- 2080Ti 跑 70B FP4: 显存占用 ~18GB（vs FP8 的 36GB 装不下）
- 如果 INT4 Tensor Core 实现成功，FP4/Q2 速度可能接近 FP8 的 0.8-1.0×

---

### 2.3 FlashAttention Fallback for SM75

**现状**：
- ✅ PocketLLM 已有 exact GQA kernel（用 m16n8k8 tensor core）
- ✅ 32K prefill 已破 1K tok/s
- ⚠️ 但没有 memory-efficient attention（O(N) 显存）

**为什么重要（老硬件视角）**：
- **2080Ti 22GB 显存紧**: 长上下文 prefill 的激活占用巨大
  - 65K context, TP4: 激活 ~2GB/rank（batch_size=1 已经这样）
  - 如果 batch_size=4，激活 ~8GB/rank，挤压 KV cache 空间
- **SM75 不支持 FlashAttention-2**: FA2 需要 SM80+
- **但可以用 FlashAttention-1**: FA1 支持 SM75，PyTorch 2.x 已内置

**设计方案**：
- 在 `cpp_engine` 中增加 **FA1 kernel 调用**（通过 cuDNN 或 PyTorch C++ API）
- 优先级：
  1. Prefill batch_size >= 2 时强制用 FA1（节省激活显存）
  2. 单请求长上下文（>32K）时可选 FA1（减少 HBM 读写）
- Decode 仍用现有 split kernel（FA1 对 decode 无优势）

**实现优先级**: **P2 (Phase 5)**

**老硬件收益**：
- Prefill 激活显存降至 O(sqrt(N))（FA1 的复杂度）
- 2080Ti batch_size=4 时，65K prefill 激活从 ~8GB 降至 ~1.5GB
- 为 continuous batching 腾出显存空间

---

### 2.4 Speculative Decoding with Tiny Draft Models

**现状**：
- ✅ PocketLLM 已有 Qwen MTP/DSpark/DFlash2
- ❌ 但这些都是 Qwen 原生方法，不支持通用 draft model

**为什么重要（老硬件视角）**：
- **2080Ti 的典型用例**: 跑一个 27B target + 一个 1.5B draft
  - 27B FP8 ~14GB，1.5B FP16 ~3GB，总共 ~17GB < 22GB
- **Decode 加速**: draft model 可以在剩余显存/CPU 上跑，target 只验证

**设计方案**：
- **Draft model 放 CPU**: 
  - 1.5B 模型 decode ~20 tok/s（CPU）
  - 生成 k=4 个 draft 需要 ~200ms
  - Target model verify 4 个 token 只需 ~100ms（比单独 decode 4 次快 4×）
- **Hybrid placement**:
  - 2080Ti 剩余显存 >5GB: draft 放 GPU
  - 显存紧张: draft 放 CPU，target 独占 GPU
- **Tree attention for verify**:
  - Verify 阶段用 tree attention 并行验证多个 draft
  - 借鉴 vLLM 的 batched verify 实现

**实现优先级**: **P2 (Phase 5-6)**

**老硬件收益**：
- 2080Ti 27B decode: 45 tok/s → ~80-100 tok/s（1.8-2.2×）
- CPU draft 的延迟可以被 target verify 隐藏

---

### 2.5 KV Cache Quantization 默认开启

**现状**：
- ✅ FP8 KV / TurboQuant K8V4 / INT8 per-token-head 都已实现
- ✅ Token parity 已验证
- ⚠️ 但 **paged KV 只支持 FP16**（这是个矛盾）

**为什么重要（老硬件视角）**：
- **KV cache 是长上下文的显存瓶颈**:
  - 65K context, FP16 KV: ~2GB/rank (TP4)
  - 65K context, INT8 KV: ~1GB/rank（省 50%）
  - 65K context, K8V4: ~0.76GB/rank（省 62%）
- **2080Ti 跑 4 个并发请求**: 
  - FP16 KV: 4 × 2GB = 8GB（太大）
  - INT8 KV: 4 × 1GB = 4GB（可接受）

**设计改进**：
- **修复 paged KV + quantized KV 的冲突**:
  - 当前 paged KV 假设每个 block 是连续的 FP16 数组
  - 量化 KV 需要 scale/zero-point，不能简单分块
- **方案 1**: Per-block quantization
  - 每个 block（16 tokens）有独立的 scale/zp
  - Block metadata 增大，但仍可接受
- **方案 2**: 放弃 paged，用 contiguous + quantized
  - 回到 contiguous KV arena，但默认量化
  - 对老硬件来说，省显存 > paged 的灵活性

**实现优先级**: **P1 (Phase 4-5)**

**老硬件收益**：
- 2080Ti continuous batching (4 并发): KV 显存从 8GB 降至 4GB
- 可以支持更长的 context 或更多并发

---

### 2.6 Ascend 910A 专项优化

**现状**：
- ✅ 多后端架构已就绪
- ⚠️ 但 Ascend 后端尚未实现（只有 CUDA 后端）

**为什么重要**：
- **国产化需求**: 很多场景必须用国产芯片
- **910A 是老硬件**: 与 2080Ti 同时代（2019），面临类似问题
- **PocketLLM 的差异化**: vLLM/SGLang 都不支持 Ascend

**Ascend 910A 特点**：
- 32 AI Core (vs 910B 的 24)
- 32MB L2 (vs 910B 的 192MB) ← 这是最大区别
- Cube freq 1000 MHz (vs 910B 的 1850 MHz)
- **No cube_vector_combine=split** (1st gen 限制)

**设计要点**：
- **L2 cache blocking**:
  - 910A 只有 32MB L2，权重/KV 分块必须更细
  - MoE expert 需要按 8 个一组 stage（vs 910B 的 24 个）
- **Cube + Vector 串行流水**:
  - 1st gen 的 Cube 和 Vector 不能并行，必须手动流水
- **CANN 9.0 的 GQA kernel**:
  - 参考 `fastllm` 的 Ascend GQA 实现
  - 或者等 CANN 9.x 官方 FlashAttention

**实现优先级**: **P2 (Phase 5-6，取决于硬件可用性）**

**Ascend 910A 收益**：
- 填补 vLLM/SGLang 的空白
- 国产化场景的唯一高性能选择

---

## 三、功能优先级矩阵

| 功能 | 优先级 | 实现难度 | 老硬件收益 | 备注 |
|------|--------|---------|-----------|------|
| **Continuous Batching** | P0 | 中 | ⭐⭐⭐⭐⭐ | 吞吐量 2-4×，必需 |
| **Prefix Caching** | P0 | 中 | ⭐⭐⭐⭐⭐ | 多轮对话必需 |
| **KV Quant + Paged** | P1 | 高 | ⭐⭐⭐⭐⭐ | 显存省 50-62% |
| **CPU Offload + Prefetch** | P1 | 高 | ⭐⭐⭐⭐ | 70B 可跑 |
| **OpenAI API 完整** | P1 | 低-中 | ⭐⭐⭐ | 生态兼容 |
| **Hybrid Quantization** | P1 | 中 | ⭐⭐⭐⭐ | 速度 vs 显存平衡 |
| **FlashAttention-1 (SM75)** | P2 | 中 | ⭐⭐⭐ | Batch prefill |
| **Tiny Draft Model** | P2 | 中-高 | ⭐⭐⭐ | Decode 1.8-2.2× |
| **Qwen-VL** | P2 | 高 | ⭐⭐⭐ | 多模态场景 |
| **Ascend 910A Backend** | P2 | 高 | ⭐⭐⭐⭐ | 国产化 |

---

## 四、Phase 4-6 Roadmap 建议

### Phase 4: 基础能力补齐（3-4 个月）
1. ✅ **Continuous Batching**: 实现动态调度器，默认 `max_num_seqs=4`
2. ✅ **Prefix Caching**: Block hash + 全局共享
3. ✅ **KV Quantization + Paged**: 修复冲突，INT8 KV 默认开启
4. ✅ **OpenAI API**: logprobs / n>1 / stop 序列（tools 仍缺失，见 1.3）
5. ✅ **CPU Offload**: 前 N 层 GPU + 后 M 层 CPU + prefetch 流水线

**验收标准**：
- 4 并发请求下，吞吐量 ≥ 单请求的 3×
- 多轮对话第 2 轮起，prefill 时间 <100ms（前缀全命中）
- 70B FP4 在 2080Ti 上可跑，decode ≥5 tok/s

### Phase 5: 生态与性能（2-3 个月）
1. ✅ **Qwen-VL**: 支持 Qwen2-VL-7B
2. ✅ **Tiny Draft Model**: 通用 draft-verify 框架
3. ✅ **FlashAttention-1**: Batch prefill 用 FA1
4. ✅ **Hybrid Quantization**: Per-layer dtype 配置

**验收标准**：
- Qwen-VL 图像问答可用
- 27B + 1.5B draft，decode 加速 ≥1.8×
- Batch_size=4 的 prefill 激活显存 <2GB/rank

### Phase 6: 多后端与高级功能（3-4 个月）
1. ✅ **Ascend 910A Backend**: CANN 后端完整实现
2. ✅ **结构化输出**: JSON mode + FSM filtering
3. ✅ **LoRA Adapter**: 动态加载和切换
4. ✅ **INT4 Tensor Core**: SM75 MMQ kernel 移植

**验收标准**：
- Ascend 910A 性能达到 vLLM CUDA 的 60-80%
- JSON mode 可用，与 SGLang 对比
- LoRA 切换延迟 <100ms

---

## 五、与 vLLM/SGLang 的差异化定位

**PocketLLM 的核心竞争力（针对老硬件）**：

1. **极致量化**: FP4/Q2 让 2080Ti 跑 70B
2. **CPU Offloading**: vLLM/SGLang 都不支持
3. **多后端**: Ascend 910A 是独家优势
4. **Qwen 推测解码**: MTP/DSpark/DFlash2 比通用 draft model 更高效
5. **单请求延迟优化**: vLLM 为吞吐优化，PocketLLM 为个人/边缘场景优化

**不追求的方向**（让给 vLLM/SGLang）：

1. **企业级部署**: Ray serving/KServe/multi-tenancy → vLLM 更成熟
2. **广泛模型支持**: Llama/Mistral/GPT-J → vLLM 覆盖更全
3. **结构化输出性能极致**: → SGLang 是专家
4. **最新研究落地**: → SGLang 更激进

**目标用户**：

- 个人开发者（2080Ti/3090/4090 在家跑大模型）
- 边缘设备（Jetson/嵌入式 GPU）
- 国产化场景（Ascend 910A）
- 研究者（需要深度定制 kernel）

---

## 六、总结

**必做（P0-P1）**：
- Continuous Batching + Prefix Caching（吞吐量和多轮对话）
- KV Quantization + Paged（显存优化）
- CPU Offloading（70B 可跑）
- OpenAI API 完整（生态兼容）

**选做（P2）**：
- Qwen-VL / Tiny Draft / FlashAttention-1（锦上添花）
- Ascend 910A（取决于硬件可用性和市场需求）

**长期**：
- 与 vLLM/SGLang 错位竞争，聚焦老硬件和国产化
- 保持单请求延迟优势
- 打造 Qwen 系列的最佳推理引擎
