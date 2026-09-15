# PocketLLM vs vLLM vs SGLang 架构对比分析

**分析时间**: 2026-09-11  
**PocketLLM 状态**: Phase 3 (paged KV 已默认开启; 真批处理 forward 与 waiting 队列已就位, 并发吞吐待实测)  
**vLLM 版本**: v0.1.15 (2080Ti fork)  
**SGLang**: 基于公开文档和设计论文

---

## 1. 请求调度与批处理

### vLLM

**核心机制**：
- **Continuous Batching**: 在 decode 阶段动态合并多个请求，每个 step 可以同时处理不同阶段的请求
- **调度器架构** (v1/core/sched/scheduler.py):
  - `Scheduler` 类管理三个队列：waiting、running、finished
  - 每个 step 根据 KV cache 容量和 token 预算动态决定调度哪些请求
  - 支持 preemption：KV cache 不足时可以暂停低优先级请求
- **调度约束**:
  ```python
  max_num_running_reqs = scheduler_config.max_num_seqs
  max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens
  ```
- **请求状态机**: WAITING → RUNNING → FINISHED/PREEMPTED
- **公平性策略**: 支持 FCFS (First-Come-First-Serve) 和 priority-based 调度

**优势**：
- 吞吐量最大化：GPU 利用率高，不同长度的请求可以同时运行
- 支持动态抢占和恢复
- 对延迟敏感的在线服务友好

### SGLang

**核心机制**：
- **RadixAttention**: 基于 Radix Tree 的 KV cache 共享和复用
- **调度器特点**:
  - 重点优化 prompt 前缀复用场景（如多轮对话、few-shot prompting）
  - 请求之间可以共享相同的 prompt 前缀，避免重复计算
  - Tree-based 调度：利用 prefix matching 减少 prefill 开销
- **批处理策略**:
  - 支持 continuous batching
  - 特别优化了结构化输出场景（JSON/grammar-constrained decoding）
  - 对推测解码（speculative decoding）的调度器集成更深

**优势**：
- 前缀复用效率极高（多轮对话、agent 场景）
- 结构化输出性能领先
- RadixAttention 可以跨请求共享 KV cache

### PocketLLM (当前状态)

**Phase 3 实现**（2026-09-12 复核源码后更新）：
- **真批处理 forward 已就位**: `QwenEngine::batch_decode_step`
  (`cpp_engine/engine/qwen_engine.cpp:3829`) 把整批 token/slot 交给
  `batch_decode_tokens` → `run_batched_decode`，**一次 forward 覆盖全批**，不是逐请求
  循环。批内 slot 必须互不重复（同 slot 两行会读到彼此刚写的 KV），引擎显式校验。
- **调度器有 waiting 队列与准入控制**: `BatchScheduler`
  (`cpp_engine/include/batch_scheduler.hpp`) 持有 `waiting_queue_`、
  `admit_requests()`，并按各请求 `worst_case_blocks` 之和对照 block 预算做准入。
- **per-request 采样**: 每行携带自己的 `BatchSamplingParams`（temperature/top_k/top_p），
  由 batched sampler 分别处理，而不是共用引擎默认值。
- **KV cache 已 slot 化**: paged KV per-slot 分配；position/prefix cache/snapshots
  已 slot 化；workspace 仍然单份共享。
- **服务端默认已开并发**: `openai_server.hpp` 的 `max_batch_size` 默认 **8**（不是 1）。
- **验证测试**: `tests/test_batch_*` 已通过交替驱动两个 slot 的正确性验证。

**差距**：
- ⚠️ **并发吞吐尚未实测**：批处理路径存在且正确，但「4 并发 ≥ 单请求 3×」这条验收标准
  从未跑过数字。缺口大小未知，须先测再谈实现。
- ❌ **批处理与投机解码/量化 KV 互斥**：`run_batched_decode` 显式拒绝
  MTP/DSpark/DFlash2（它们各自持单序列上下文），且要求 FP16 KV
  （`qwen_engine.cpp:3243`）。
- ❌ 无抢占与优先级：block 一旦分配不会被回收，准入是 FCFS。
- ❌ 无跨请求 KV 共享：`block_hash` 在 `cpp_engine/` 中零引用，前缀复用仅限单 slot 内。
- ✅ 单请求延迟优化到位（这是当前设计目标）

---

## 2. KV Cache 管理

### vLLM: PagedAttention

**核心设计** (v1/core/kv_cache_manager.py, v1/core/block_pool.py):
```python
class KVCacheManager:
    - coordinator: KVCacheCoordinator  # 管理 block 分配
    - block_pool: BlockPool            # 物理 block 池
    - enable_caching: prefix caching 开关
    - hash_block_size: prefix 匹配粒度
```

**特点**：
- **Paged Memory**: KV cache 按 block 分配（block_size 通常 16-32 tokens）
- **动态分配**: 请求只占用实际需要的 blocks，释放后可被其他请求复用
- **Prefix Caching**: 
  - 通过 hash 匹配共享前缀 blocks
  - 跨请求复用相同 prompt 的 KV cache
- **内存效率**: 
  - 避免每个请求预留 `max_context` 的连续内存
  - 实际测试中 vLLM 预留 `gpu_memory_utilization * 总显存`（默认 0.9）
  - 运行时峰值显存 ~19.78 GB/卡 (TP4, 65K context)

**数据结构**：
```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # 每个 KV head group 的 block 列表
    
class KVCacheBlock:
    block_id: int
    block_hash: Optional[int]  # prefix caching 用
    is_null: bool              # padding block 标记
```

### SGLang: RadixAttention

**核心设计**：
- **Radix Tree**: 所有请求的 KV cache 组织成一棵前缀树
- **自动共享**: 
  - 相同 prompt 前缀自动共享同一条树路径
  - 分支发生在 prompt 首次分叉的位置
  - LRU 策略淘汰不活跃的树节点
- **动态增长**: 树节点按需分配，不需要预先估计每个请求的长度
- **跨请求复用**: 
  - 多轮对话的历史轮次可以完全复用
  - Few-shot prompting 的 demonstrations 只计算一次

**优势场景**：
- Agent 多轮交互（每轮复用历史）
- Batch inference with shared system prompt
- Self-consistency sampling (同一个 prompt 采样多次)

### PocketLLM: Paged KV (FP16 only)

**当前实现** (cpp_engine/engine/qwen_engine.cpp:3388-3405):
```cpp
if (options_.kv_paged) {
    if (options_.kv_cache_dtype != QwenKvCacheDType::Fp16) {
        throw std::runtime_error("Qwen paged KV cache requires FP16; got ...");
    }
}
```

**特点**：
- ✅ **已默认开启**: `--kv-paged` 默认 true (PR #149)
- ✅ **Block 池化**: 共享一个 block pool，按需分配
- ✅ **Per-slot 隔离**: 每个 slot 有独立的 block 列表
- ⚠️ **限制**: 
  - FP16 only（FP8/INT8/TurboQuant 需要 contiguous arena）
  - 没有 prefix caching（无 hash 和跨请求共享）
  - 没有 preemption（block 一旦分配就不会被抢占）

**内存对比**（实测，TP4）：
- PocketLLM (paged, 待本轮测试完成确认)
- vLLM: 19.78 GB 峰值（预留 90% 显存，22 GB * 0.9 = 19.8 GB）

---

## 3. Tensor Parallel 通信

### vLLM

**TP 实现**：
- **NCCL all-reduce**: 每层 MLP/Attention 输出做 all-reduce
- **Custom all-reduce**: 
  - `--disable-custom-all-reduce=False` 时使用 IPC-based 优化
  - 跨 TP rank 共享显存，避免 NCCL 协议开销
  - 仅适用于单机 NVLink 拓扑
- **通信模式**: 
  - TP-split 权重（列切或行切）
  - Decode 阶段每个 token 都需要 all-reduce
  - Prefill 阶段 chunked，减少通信频率

**性能** (实测 TP4, 2080Ti PCIe):
- Decode: all-reduce 约占 36% 时间（内存: glm_dsa_decode_profile）
- 8K decode: 43.3 tok/s (已追平 PocketLLM)

### SGLang

**TP 特点**：
- 继承 vLLM 的 TP 实现
- RadixAttention 的 prefix 共享在 TP 上需要额外同步树结构
- 对推测解码的 TP 通信有额外优化（draft model 可以不参与 TP）

### PocketLLM

**TP 实现** (cpp_engine TP4):
- **NCCL all-reduce**: 每层 MLP/Attention 输出
- **通信开销**: 
  - Decode: all-reduce 占 ~47 ms/token (内存: cpp_engine_decode_reduce_profile)
  - TP4 是实测最优配置（内存: cpp_engine_tp_world_sweep）
  - TP=1/2 prefill 慢 1.8-3×
- **优化**: 
  - Prefill: comm overlap (内存: qwen_q64_attention_tile_and_comm_overlap)
  - Decode: split 256 + FP8 wide loads (内存: qwen_decode_split_and_fp8_wide_loads)
  - 当前 decode 29→37 tok/s (TP4)

**差距**：
- NCCL 通信时间已是硬底（47 ms/token），进一步提速需要 overlap
- 没有 custom all-reduce (PCIe 拓扑收益有限)
- ✅ TP rank 间已做到 token parity (内存: cpp_engine_batching_slot_state_scope)

---

## 4. 量化支持

### vLLM

**支持的量化方案**：
- **Weight-only**: 
  - FP8 (Marlin kernel on SM75+)
  - INT8/INT4 (GPTQ, AWQ)
- **Weight+Activation**:
  - W8A8 (SmoothQuant)
- **KV Cache 量化**:
  - FP8 KV cache
  - INT8 per-token-head quantization

**Marlin FP8** (实测):
- SM75 (2080 Ti) 使用 Marlin FP8 weight-only
- Prefill: cuBLAS 已达 tensor core 峰值 37-63 TFLOP/s
- Token 与 PocketLLM 逐位一致

### SGLang

**量化支持**：
- 继承 vLLM 的量化后端
- 额外优化了 FP8 + RadixAttention 的组合

### PocketLLM

**支持的量化方案**：
- **Weight**:
  - ✅ FP8 (cuBLAS prefill, DP4A decode 已验证 token parity)
  - ✅ FP4/NVFP4 (自研 kernel)
  - ✅ GGUF Q2 (IQ2_XXS/IQ3_XXS, DP4A grouped kernel)
  - ✅ GGUF Q4/Q5 (Q4_K/Q5_K, MMA prefill 2.1×)
- **KV Cache**:
  - ✅ FP8 (dequant-once 修复完成，prefill 追上 FP16)
  - ✅ TurboQuant K8V4 (省 62% KV 显存，token 与 FP16 一致)
  - ✅ INT8 per-token-head (Triton kernel 已写，未接入 cpp_engine)

**差异**：
- ✅ **GGUF 原生支持**: PocketLLM 独有，vLLM/SGLang 不支持 GGUF
- ✅ **FP4**: PocketLLM 独有 (但在 SM75 慢于 FP8)
- ❌ **GPTQ/AWQ**: vLLM 有，PocketLLM 无
- ⚠️ **Paged KV 限制**: PocketLLM 的 paged KV 只支持 FP16，量化 KV 需要 contiguous arena

---

## 5. 推测解码 (Speculative Decoding)

### vLLM

**支持的方法**：
- **Draft model**: 小模型生成 draft，大模型验证
- **Medusa**: 多头并行预测
- **EAGLE**: 动态树搜索
- **集成度**: 调度器原生支持，draft 和 verify 在同一个 batch 里

**实现**：
- Draft model 可以与 target model 共享 TP
- 验证阶段使用 batched verify (tree attention)

### SGLang

**支持的方法**：
- 继承 vLLM 的 draft model
- **额外优化**: 
  - 结构化输出 + speculative decoding 结合
  - RadixAttention 可以复用 draft 的 KV cache

### PocketLLM

**支持的方法** (Qwen3.5/Qwen4 only):
- ✅ **MTP (Multi-Token Prediction)**: Qwen 原生 MTP heads
  - Stage A accept rate 78% (内存: mtp_stage_a_accept_rate)
  - 实测 FP8 1.15-1.22× on 512-token fixture
- ✅ **DSpark**: 共享 hidden states 的 draft-verify
  - C++ batched verify 已修复 (内存: cpp_batched_verify_stale_seed_fix)
  - Real prompt 接受率 <30%，draft 成本≈主模型 decode
  - 当前 0.85× (sequential verify 瓶颈)
- ✅ **DFlash2**: FlashMemory plugin scoring
  - 合成 512: 2.78×，GSM8K: 1.33× (内存: dflash2_current_verified_state)
  - Drafter 瓶颈是 LM head (55% 成本)
  - cuBLAS FP32 让 decode 加速比进入上游 2.67-3.43× 区间

**差距**：
- ❌ 无通用 draft model 支持（只有 Qwen 原生方法）
- ❌ 无 Medusa/EAGLE
- ✅ Qwen MTP/DSpark/DFlash2 是 PocketLLM 独有优势

---

## 6. 多模态与结构化输出

### vLLM

**多模态**：
- 支持 LLaVA、Qwen-VL、InternVL 等主流视觉模型
- 图像/视频作为额外 token 序列输入

**结构化输出**：
- JSON schema 约束
- 正则表达式约束
- 基于 FSM 的 token filtering

### SGLang

**多模态**：
- 支持主流视觉模型
- 特别优化了多轮多模态对话（RadixAttention 复用图像 embeddings）

**结构化输出**：
- ✅ **最强优势**: 结构化输出性能业界领先
- JSON/Regex/Grammar-constrained generation
- 与 continuous batching 深度集成

### PocketLLM

**多模态**：
- ❌ 当前不支持（文档明确标注 text-only）

**结构化输出**：
- 部分支持（native C++ engine，PR #183）
- JSON mode 与 JSON Schema 约束已实现（`cpp_engine/core/json_constraint.cpp` + `token_constraint.cpp`，HTTP `response_format` 解析）
- ❌ 不支持 grammar / regex（GBNF）约束
- ❌ TP > 1 下不可用：请求级 token 约束在张量并行路径被显式拒绝
- Python 控制平面仍无 `StructuredOutputManager`，结构化输出只走 C++ engine

---

## 7. 扩展性与生态

### vLLM

**优势**：
- ✅ 成熟的 plugin 生态
- ✅ LoRA adapter 动态加载和多路复用
- ✅ 完整的 OpenAI API 兼容（tools/function calling）
- ✅ 企业级部署支持（Ray serving, KServe）
- ✅ 活跃社区和快速迭代

**限制**：
- ⚠️ 主要面向 NVIDIA GPU（CUDA）
- ⚠️ AMD ROCm 支持较弱
- ❌ 不支持 Ascend NPU

### SGLang

**优势**：
- ✅ RadixAttention 对 agent/多轮对话场景的极致优化
- ✅ 结构化输出性能领先
- ✅ 对推理框架集成友好（LangChain/CrewAI）
- ✅ 前沿研究快速落地（如 compressed attention）

**限制**：
- ⚠️ 社区规模小于 vLLM
- ⚠️ 模型支持覆盖面不如 vLLM 广
- ❌ 不支持 Ascend NPU

### PocketLLM

**优势**：
- ✅ **多后端架构**: 同时支持 CUDA 和 Ascend NPU
- ✅ **GGUF 原生支持**: 可以直接加载 llama.cpp 格式权重
- ✅ **极致单请求延迟优化**: 针对 2080Ti 等消费级卡优化到极限
- ✅ **Qwen 系列深度优化**: MTP/DSpark/DFlash2 推测解码
- ✅ **FP4/Q2 量化**: 在 22GB 显存跑 27B 模型

**限制**：
- ❌ 模型支持少（仅 Qwen/DeepSeek-V3/MiniMax）
- ❌ 生态不成熟（无 LoRA、无多模态；结构化输出仅部分支持，见第 6 节）
- ⚠️ Continuous batching 路径已存在（批处理 forward + waiting 队列 + 准入预算），但并发
  吞吐未实测，且与投机解码/量化 KV 互斥
- ❌ 社区规模小
- ⚠️ 文档和易用性需加强

---

## 8. 性能对比总结 (2080Ti TP4)

**基准**: Qwen3.8-27B-FP8, TP4 (GPU 0-3), TG=128, `prefill_chunk_tokens=8192`, KV FP16

**测量条件**（两侧一致，2026-09-11 实测）:
- 同一份 token fixture，sha256 逐个校验一致（PocketLLM 生成，vLLM 复用）
- 显存由外部 `nvidia-smi` 以 0.25s 间隔采样，记录 before / peak / after
- 两侧空载基线均为 1 MiB/卡，因此 peak 即净占用
- vLLM: `gpu_memory_utilization=0.90`, `max_num_seqs=1`, `gdn_prefill_backend=triton`
- 运行记录: `.tmp/ab3_pocket_20260911_071901/`, `.tmp/ab3_vllm_20260911_072632/`

### Prefill (tok/s)

| Context | PocketLLM | vLLM   | 比值      |
|---------|-----------|--------|-----------|
| 4096    | 1735.1    | 1812.3 | 0.957×    |
| 8192    | 1821.0    | 1779.6 | **1.023×** |
| 32768   | 1677.8    | 1655.1 | **1.014×** |
| 65536   | 1453.5    | 1503.6 | 0.967×    |

### Decode (tok/s)

| Context | PocketLLM | vLLM  | 比值      |
|---------|-----------|-------|-----------|
| 4096    | 45.84     | 43.33 | **1.058×** |
| 8192    | 45.44     | 41.56 | **1.093×** |
| 32768   | 43.34     | 39.91 | **1.086×** |
| 65536   | 40.00     | 39.24 | **1.019×** |

*decode 在全部四个长度上领先，8K/32K 领先 8-9%；这是 TG=128 的测法，短 TG 的 decode 数字不可引用。*

### 整请求墙钟 (prefill + 128 token decode)

| Context | PocketLLM | vLLM   | vLLM/PocketLLM |
|---------|-----------|--------|----------------|
| 4096    | 5.13 s    | 5.19 s | 1.012×         |
| 8192    | 7.29 s    | 7.66 s | 1.050×         |
| 32768   | 22.46 s   | 22.98 s| 1.023×         |
| 65536   | 48.26 s   | 46.82 s| 0.970×         |

### GPU 显存峰值 (GiB/卡)

| Context | PocketLLM | vLLM  | 比值   |
|---------|-----------|-------|--------|
| 4096    | 8.13      | 19.19 | 0.424× |
| 8192    | 8.69      | 19.80 | 0.439× |
| 32768   | 9.06      | 19.78 | 0.458× |
| 65536   | 9.56      | 19.80 | 0.483× |

**这一列不能读作"显存效率 2.1×"**。vLLM 的 ~19.8 GiB 是 `gpu_memory_utilization=0.90`
预先划走的 KV 池，不是它完成该请求所必需的量；把该参数调低它同样能跑。这张表说明的是
两者的**默认显存策略不同**：vLLM 预占换取后续并发不再分配，PocketLLM 按需增长。

PocketLLM 侧的实际构成（来自引擎自身计数器，65536 场景）:

| 组成 | 字节/rank |
|------|-----------|
| 常驻权重 | 6.86 GiB |
| 权重 scale | 0.71 MiB |
| KV cache | 1.00 GiB |
| Activation workspace 峰值 | 1.01 GiB |

权重是固定项，随 context 增长的只有 KV 与 activation workspace。4096 场景下 KV 仅
66 MiB、workspace 504 MiB，这解释了显存曲线为何几乎是平的。

**PocketLLM 这一列在当前 master 上会更高，读表时须知。** 上表取自 2026-09-11 的实测
（rev `d57584a`，运行记录 `.tmp/ab3_pocket_20260911_071901/`）。当前 master（`cfad866`）在
相同配置（TG=128、chunk 8192、FP16 KV）下，引擎自身计数逐字节一致——65536 场景仍是权重
6.86 GiB + KV 1.00 GiB + workspace 1.01 GiB——但 `nvidia-smi` 峰值高 2.75 GiB：引擎之外每进程
固定多占用 3.46 GiB 而不是 0.71 GiB（CUDA context、cuBLAS workspace、NCCL buffer 一类），且该
差值不随 context 变化。因此当前 master 在 65536 上峰值是 12.31 GiB，比值 0.62× 而非 0.483×。
**结论不变**：差异来自默认显存策略而不是效率倍数。这 2.75 GiB 由哪次提交引入尚未定位。

同一原因使 decode 列也有小幅漂移：当前 master 重跑同一 sweep，prefill 与上表相差 0.4% 以内，
decode 则低 2–4%（8192 场景 43.99 vs 45.44 tok/s）。模型页的
[Validated performance](../models/qwen3.8-27b-fp8.md) 以当前 master 为准。

### 怎么解读这组数字

- **prefill 基本持平**（0.957×-1.023×）。差距的根因已单独记录：65K 上 attention 的 N²
  项慢 1.94×（25% occupancy），线性非 GEMM 项慢 1.18×，而 GEMM 本身已到 95-102% 峰值、
  NCCL 是双方共享地板。
- **decode 全面领先**（1.019×-1.093×）。gqa_split / gdn_step / decode MMA 几轮优化的
  累积结果。
- **单请求场景已无实质差距**。这张表是 `max_num_seqs=1` 的单流对比，两侧都没有走并发
  路径，所以它衡量的是 kernel 与调度开销，不是服务吞吐。
- **并发是唯一未量化的维度**。PocketLLM 的批处理 forward 与 waiting 队列都已存在（见
  第 1 节），但「4 并发 ≥ 单请求 3×」从未实测。这决定 Phase 4 的 continuous batching
  到底是「已完成待验证」还是「真有缺口」，因此并发实测排在任何实现工作之前。

---

## 9. 架构选型建议

### 选择 vLLM 的场景
- ✅ 高并发在线服务（QPS > 10）
- ✅ 需要完整的 OpenAI API 兼容
- ✅ 需要 LoRA adapter 动态切换
- ✅ 多模态推理
- ✅ 企业级部署和监控
- ✅ 成熟生态和社区支持

### 选择 SGLang 的场景
- ✅ Agent 和多轮对话（前缀复用效率最高）
- ✅ 结构化输出（JSON/grammar-constrained）
- ✅ Few-shot prompting (demonstrations 复用)
- ✅ Self-consistency sampling
- ✅ 需要最新推理优化算法

### 选择 PocketLLM 的场景
- ✅ **消费级 GPU 部署**（2080Ti/3090/4090）
- ✅ **Ascend NPU 部署**（华为昇腾）
- ✅ **单请求低延迟优先**（个人助手/边缘推理）
- ✅ **GGUF 格式模型**（llama.cpp 生态）
- ✅ **极致量化**（FP4/Q2 在 22GB 跑 27B）
- ✅ **Qwen 系列推测解码**（MTP/DSpark/DFlash2）
- ✅ **研究和实验**（kernel 优化、硬件适配）

---

## 10. PocketLLM 下一步建议

基于与 vLLM/SGLang 的对比，PocketLLM 的关键改进方向：

### 短期 (Phase 4)
1. **启用 continuous batching**: 
   - Slot 化基础设施已就绪
   - 需要实现动态调度器（waiting/running queue）
   - 目标：batch_size=4 下保持单请求延迟不回归
2. **补全 OpenAI API 语义**:
   - Tools/function calling
   - ~~Logprobs~~ ✅（2026-09-15）
   - ~~Multiple choices (n>1)~~ ✅（2026-09-15）
3. **Prefix caching**:
   - 实现 block hash 和跨请求共享
   - 优先支持多轮对话场景

### 中期 (Phase 5)
1. **多模态支持**: Qwen-VL/Qwen2-VL
2. **结构化输出**: JSON schema + FSM token filtering
3. **LoRA adapter**: 动态加载和切换
4. **扩展模型支持**: Llama3/Mistral/GLM

### 长期
1. **RadixAttention 类似机制**: 基于 prefix tree 的 KV cache 共享
2. **通用 draft model**: 支持任意小模型做 speculative decoding
3. **分布式推理**: 跨节点 TP/PP
4. **企业级特性**: Metrics/tracing/multi-tenancy

---

## 参考资料

- vLLM 源码: /mnt/data1/vLLM-2080Ti-Definitive-v015
- SGLang 论文: "Efficiently Programming Large Language Models using SGLang"
- PocketLLM 文档: docs/architecture/vllm_sglang_comparison.md
- 性能内存记录: MEMORY.md (qwen_*, cpp_engine_*)
