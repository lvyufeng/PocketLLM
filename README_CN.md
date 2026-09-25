# PocketLLM

[English](README.md) | 中文

PocketLLM 是一个面向消费级多卡系统的大模型推理工程栈，包含 C++/CUDA 与 PyTorch runtime。它结合模型专用 kernel、低 bit 格式、tensor/expert parallel、CPU/GPU placement，以及面向单请求的可复现实测 benchmark。

项目最初来自在 4×RTX 2080 Ti 上运行 DeepSeek-V4 的工程实践，目前已经包含 DeepSeek-V4、MiniMax-M2.7、GLM-5.2、Qwen3.8-27B、DeepSeek-V4.1-Flash、MiMo-V2.6-Flash 和 Ternary-Bonsai-2-27B 的已验证 runtime。PocketLLM 不是一个“所有模型共用同一后端”的框架：不同模型使用与其架构和 checkpoint 格式匹配的执行路径。

其中四个模型已经通过 OpenAI 兼容 API 端到端服务：**Qwen3.8-27B-FP8** 走原生 C++ runtime，**Ternary-Bonsai-2-27B** 走同一个 runtime 且只用**一张**卡，**DeepSeek-V4.1-Flash** 走 `pocketllm serve --backend v41`，**MiMo-V2.6-Flash** 走 `pocketllm serve --backend mimo`。四条路径都在真实 checkpoint 上做过验证。

> **项目状态：** 研究和工程软件。下面的数字来自特定 checkpoint、硬件和测试口径，不代表通用性能保证。

## News

- **[2026/09] Ternary-Bonsai-2-27B 单卡端到端可服务。** 一个 27B 混合注意力模型 —— 48 层 Gated
  DeltaNet + 16 层 GQA，dense MLP —— 发布成权重只有 **1.75 bit** 的 GGUF（GGML type 143，
  5.53 GiB），文件里还声明了一个 Hadamard 旋转。`pocketllm serve` 直接从容器自带的
  `general.architecture` 选中原生引擎，不需要任何 flag：4,096 token prompt 下 **prefill
  636.0 tok/s**、**decode 25.9 tok/s**，同卡上游参考是 642.5 与 30.7；5.53 GiB 权重留出的余地
  够 **245,760 token 上下文**（配 fp8 KV cache 可到 262,144）。
  [模型页](docs/models/ternary-bonsai-2-27b.md)
- **[2026/09] MiMo-V2.6-Flash 端到端可服务。** `pocketllm serve --backend mimo` 把发布版跑成四个
  进程、四张卡：149.81 GiB routed expert 放在 host bank 里，48 层 backbone 在 GPU 上执行。九个
  global 层的 attention 按 checkpoint 自带的四路 `qkv_proj` 划分切开、用 all-gather 拼回，于是
  **262,144-token prompt 的 prefill 达到 104.04 tok/s**，该深度下 decode 一步
  **197.2 ms —— 5.07 tok/s**，浅上下文 5.63，四个 rank 逐位一致。
  [模型页](docs/models/mimo-v2.6-flash.md)
- **[2026/09] DeepSeek-V4.1-Flash 端到端可服务。** `pocketllm serve --backend v41` 把发布的
  475 GiB checkpoint 跑成四个进程、四张 22 GiB 卡，其中 457.8 GiB routed expert 是 pinned 在
  host 内存而不是常驻卡上。runtime 最长接受 262,144 token 上下文；260,244-token prompt 实测
  prefill 150.3–152.0 tok/s、decode 3.48–3.54 tok/s。跨请求 prefix caching 也在同一批落地：
  已经服务过的前缀，重发时只 forward 尾部。
  [模型页](docs/models/deepseek-v4.1-flash.md) ·
  [Run record](docs/performance/deepseek_v4_1_flash_served_gate.md)
- **[2026/09] Qwen3.8-27B-FP8 有了原生 OpenAI 兼容 server** —— health 与 model discovery、
  流式与非流式 chat/completions、逐 token logprobs、stop sequence 截断、非法字段拒绝，以及并发
  scheduler 准入，全部在真实 checkpoint 上验证过。[模型页](docs/models/qwen3.8-27b-fp8.md)
- **[2026/08] Qwen3.8-27B 接上两个外部投机 drafter。**
  [DSpark](docs/architecture/qwen3_8_27b_fp8_design.md#external-dspark-speculative-decoding) 在前，
  [DFlash2](docs/architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding) 在后：512-token
  fixture 上全请求 2.78×、decode 3.02×，逐 token 完全一致。两者都是 opt-in，因为收益取决于
  acceptance rate，而上游公布的 2.67–3.43× 是 decode 延迟比而非全请求比。
- **[2026/08] Qwen3.8-27B-FP8 上 C++/CUDA runtime** —— TP4 下的 FP8 E4M3 Safetensors 文本生成，
  512-token prompt 上 prefill 864.54 tok/s、decode 43.22 tok/s，另有 256K 上下文路径，以及一个
  跨请求保持 prefix state 的常驻 TP4 worker。
- **[2026/07] GLM-5.2 文本生成** —— 走共用的 GGUF raw-block 路径，入口与其它 GGUF 模型相同的
  `src.cli.generate_glm`。
- **[2026/06] MiniMax-M2.7 on GGUF `UD-IQ1_M`** —— full-model 256-token prefill 约 104.9–107
  tok/s，融合 RMSNorm 后 43 层 decode benchmark 10.32 tok/s。
- **[2026/05] DeepSeek-V4-Flash** —— 项目起步时的那个 checkpoint：FP4/FP8 Safetensors 与
  GGUF Q2/IQ2/IQ1 generation，32K–64K 下 C++ FP4 prefill 约 401 tok/s。
  [模型页](docs/models/deepseek-v4.md)

## PocketLLM 提供什么

- **模型专用推理路径：** 支持 hybrid attention、MLA、GQA、Gated DeltaNet、dense MLP 和 routed MoE 层。
- **避免不必要的低 bit 展开：** 在支持的热路径中直接消费 FP4、FP8 E4M3、GGUF Q4/Q5/Q8、IQ1/IQ2/IQ3、Q2 等量化 block。
- **消费级 GPU 并行：** 支持 PCIe 多卡上的 TP4/NCCL；对放不进显存的 checkpoint，支持 CPU/NUMA expert placement。
- **Prefill/decode 分离：** 大 batch kernel 与单 token latency 路径独立调度、独立优化。
- **原生 C++/CUDA runtime：** `cpp_engine/` 当前支持 DeepSeek-V4 GGUF/Safetensors 路径、Qwen3.8 FP8 Safetensors 文本生成、端到端按 ternary 消费的 1.75 bit GGUF，以及已验证的 OpenAI 兼容文本 server。
- **为“放不进显存的 checkpoint”准备的 host-PyTorch adapter：** `--backend v41` 用四个进程（每卡一个）在 memory-mapped checkpoint 上运行 DeepSeek-V4.1-Flash —— dense tree 和 packed FP4 expert 在 GPU 上执行，routed expert 从 pinned host bank 读取。
- **四个 rank 共享的 host 常驻 expert bank：** `--backend mimo` 把 MiMo-V2.6-Flash 的 149.81 GiB MXFP4 expert 放在一个 `/dev/shm` 段里，每个 rank 都 attach 到同一份，并逐层把 expert 分出去 —— decode 一步按“抽取”分，prefill chunk 按“expert”分 —— 于是一个 rank 只需要 stage 一个 token 抽到的 8 个 expert 中的 2 个，而不是全部 8 个。
- **检查和验证工具：** GGUF 架构/spec 报告、Safetensors audit、tensor shape 检查、数值 parity 测试和真实 checkpoint benchmark。

## 支持模型一览

| 模型 | Checkpoint / 格式 | Runtime 状态 | 已验证路径 | 4×RTX 2080 Ti 代表结果 |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md) | Safetensors FP8 E4M3 dense + FP4 E2M1 expert | **已验证 OpenAI server 后的 TP4 文本生成** | `pocketllm serve --backend v41`：host PyTorch 跑 mapped checkpoint，dense tree 与 packed FP4 expert 在卡上，一 rank 一进程 | Served TP4：260,244 token prompt 下 **prefill 150.3–152.0 tok/s**（1,364 token 时 137–141），**decode 3.48–3.54 tok/s**，同时只跑一个请求 |
| [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md) | Safetensors FP8 E4M3 dense + MXFP4 expert，attention 输出 BF16 | **已验证 OpenAI server 后的 TP4 文本生成** | `pocketllm serve --backend mimo`：48 层 backbone 在卡上，routed expert 走 149.81 GiB host bank，attention 按 checkpoint 自带的四路 `qkv_proj` 划分切开 | Served TP4：262,144 token prompt 下 **prefill 104.04 tok/s**（attention 复制时 48.37），该深度 **decode 5.07 tok/s**，浅上下文 5.63，同时只跑一个请求 |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | **已验证 C++ 文本 runtime 与 OpenAI server** | C++/CUDA TP4、GPU-resident FP8 | Served TP4：512-token prompt 下 prefill 864.54 tok/s、decode 43.22 tok/s（生成 128 token） |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0`（GGML type 143），**每个权重 1.75 bit**，5.53 GiB，Hadamard 写在文件里 | **已验证 C++ 文本 runtime 与 OpenAI server，单卡** | `pocketllm serve` 直接从文件自带的 `general.architecture` 选中原生引擎，无需 flag | **1×**RTX 2080 Ti：4,096-token prompt 下 **prefill 636.0 tok/s**、**decode 25.9 tok/s**（同卡上游参考为 642.5 / 30.7），245,760 token 上下文 |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8；GGUF Q2/IQ2/IQ1 | **已验证 generation** | PyTorch 异构、C++/CUDA、GGUF TP4 | C++ FP4：32K–64K prefill 约 401 tok/s；decode 约 3.7 tok/s |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | **已验证 TP4 generation** | Raw-block CUDA、GGUF TP4 | Full-model 256-token prefill 约 104.9–107 tok/s；43-layer decode benchmark 10.32 tok/s |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | **已验证文本生成** | Raw-block CUDA、GGUF TP4 | prefill 约 0.79 tok/s；decode 约 0.66 tok/s |

Qwen3.8-27B 一行覆盖同一文本架构下的三个 checkpoint：上面已验证的 FP8 runtime、[NVFP4](docs/models/qwen3.8-27b-nvfp4.md) 变体，以及[官方 BF16](docs/models/qwen3.8-27b-bf16.md) 发布版（已审计但未运行）。每个的确切状态见[支持矩阵](docs/models/README.md)。

模型页面会把“模型架构规格”和“PocketLLM 当前实际实现能力”分开。`inspect`、`smoke` 和 benchmark 也不自动等于 production serving 保证。

## 性能摘要

本节数字除非特别说明，都来自同一台基线机器上的真实 checkpoint：4× NVIDIA RTX 2080 Ti、每卡 22 GiB、PCIe Gen3、无 NVLink、单请求执行、适用时使用 TP4。Qwen3.8 下的那组并发阶梯是例外 —— 它跑的是原生 batch scheduler。比较前请先阅读 [Benchmark 口径](docs/guides/benchmarking.md)。

### DeepSeek-V4.1-Flash v41 runtime（served）

发布的 475.24 GiB checkpoint 跑在四个 rank 上，一卡一进程，457.8 GiB routed expert 是 pinned 在 host 内存里而不是常驻卡上；dense tree 和 packed FP4 expert 都在 GPU 上执行。下面是 runtime 能接受的最长一条腿 —— 260,244 token prompt，走 `pocketllm serve --backend v41`，连续三个请求，每个生成 64 个 greedy token：

- **prefill 150.3、152.0、152.0 tok/s**，而 1,364-token prompt 上是 137–141 tok/s。长 prompt 反而*更快*，因为短 prompt 由固定的 per-process 开销主导。
- **decode 3.53、3.48、3.54 tok/s**，每步 253–262 ms。

对照参考 launcher 自己的 262,144-token 那一行（103.54 tok/s、3.88 tok/s），这是 prefill 1.45–1.47×、decode 0.90–0.91×，wall 1,730.0 s 对 2,555.4 s。有两点限定必须一起看：两个 prompt 不是同一段文本 —— served 那条腿是重复的填充文本，参考行是章节模板 —— 所以 prefill 的差距里有一部分可能来自 prompt 而非 runtime；而 decode 那一半是不成立的，因为同一个服务在 1,364 prompt token 上读到的是 4.45–4.53 tok/s。

在短上下文配置上（`--max-model-len 2048`、288 个 expert pool row）、1,364-token prompt 下，prefill 为 137.5、140.8、138.3 tok/s，decode 为 4.45–4.53 tok/s；第一个请求读到 108.0，因为它要付 capture pass 的代价。这些是冷 prompt 数字，取自跨请求 prefix caching 落地之前 —— 现在重发一个已经服务过的 prompt 只会 forward 它的尾部。

这个 runtime 没有的东西是 batching、continuous batching 和 MTP 层。三个 DSpark draft 层共 7.39 GiB，loader 有意把它们留在 shard 里，所以这里没有 speculative decoding，请求是串行而非批量执行。也没有数值 oracle：参考栈需要 `torch>=2.10.0` 和 `tilelang==0.1.8`，本机两者都没有，因此验收证据是生成的文本而不是 logit 对比。

### MiMo-V2.6-Flash 异构 runtime（served）

发布版跑成四个进程、一卡一个：149.81 GiB routed MXFP4 expert 放在一个 `/dev/shm` 段里，每个 rank attach 到同一份；48 层 backbone —— 九个 global attention 层，加三十九个跑在 128-slot ring 上的滑窗层 —— 在卡上执行，每层 expert 再分给四个 rank。用哪个 deal 取决于这次调用的行数：decode 一步每 rank 抽 `top_k / world` 个 expert，prefill chunk 则需要把 expert 本身切开。

- **262,144 token prompt：prefill 104.04 tok/s**，2048-token chunk，卡上 10.21 GiB，四个 rank 末行逐位一致。
- **该深度 decode：一步 197.2 ms，5.07 tok/s**；浅上下文同一步是 177.6 ms、5.63 tok/s。作为对照，单卡是 610 ms 一步、1.64 tok/s。
- prefill 这个数是 attention 切分挣来的：同一 prompt 在每 rank 复制 attention 时是 **48.37 tok/s**、卡上 18.71 GiB。64k prompt 配 4096-token chunk 跑到 104.4 tok/s。

这一步剩下的东西是一次拷贝和一个调度。**其中 99.9 ms 是 kernel 真正等到的 expert H2D** —— 1198.5 MiB，12.0 GiB/s，PCIe 3.0 x16 基本跑到链路速率 —— 其余是 attention 38.8 ms、expert kernel 12.5 ms（47 次调用），以及约 10 ms 的集合通信。用上一个 token 的抽取来预取这次拷贝的方案已经实测证伪：一个 rank 的两行里，行内容与上一步相同的情况只有 9–13.5%，整集重复只有 1.9–2.4%。

这里没有 batching、没有 speculative decoding，attention 和 dense linear 也还是 torch 而非 kernel。

### Qwen3.8-27B-FP8 C++ runtime

master `cfad866` 上按引擎默认值做的一轮串行 sweep，每次生成 128 token：

- 64-token prompt：prefill 115.91 tok/s（0.55 s），decode 45.05 tok/s。
- 512-token prompt：prefill 864.54 tok/s（0.59 s），decode 43.22 tok/s。
- 8,192-token prompt：prefill 1,818.65 tok/s（4.50 s），decode 43.99 tok/s。
- 65,536-token prompt：prefill 1,453.51 tok/s（45.09 s），decode 39.11 tok/s。

每 rank 的引擎计数为 6.86 GiB 常驻 FP8 权重与 scale，加上 65,536 token 时的 1.00 GiB KV 数据和 1.01 GiB chunk workspace；`nvidia-smi` 在此之上还要多出 3.4–3.5 GiB（CUDA context、cuBLAS workspace、NCCL buffer），且该差值不随 prompt 长度变化。四个 TP rank 的生成 token 序列一致。原生 OpenAI 兼容 server 已验证 text 请求；图像和视频输入仍不支持。

64 和 512 token 两行的 prefill 反映的是短 prompt 延迟而非稳态吞吐：两者都在 0.55–0.59 s 内完成，因为该规模下固定进程开销占主导。4,096 token 以上，prefill 到 32,768 的边际吞吐为 1,670 tok/s，之后再为 1,285 tok/s。

这条 runtime 也做 batching，而且是这里唯一带完整 scheduler 的那个。请求由一个 scheduler 准入：从 paged KV block pool 里分配 block、在 token 预算下推进 prefill，整个活跃集合走一次 batched decode step。8 个并发的 128 词请求 1.561 s 跑完，串行跑同样 8 个要 7.300 s —— **4.68×**，聚合输出 163.95 tok/s，而串行平在 35.07；2 并发和 4 并发分别是 2.14× 和 3.61×。同样四张卡上与 vLLM 0.1.15 的 batch mode 相比，1/2/4/8 并发下 vLLM 的 wall time 分别是 PocketLLM 的 1.22×/1.34×/1.32×/1.17×。batching 在引擎自带的 OpenAI server 上默认开启，在 Python `--backend cpp` adapter 上要用 `--backend-option enable_batching=true` 打开；完整阶梯表、复现命令和注意事项（单请求那一档的 17% 里含有 slot 间 prompt prefix 复用的成分，只能当 smoke 上界看）见[并发验证记录](docs/performance/cpp_openai_concurrency_validation.md)。

通过 OpenAI API 服务 Qwen3.8 的正是同一个 runtime，它也是这里唯一一个带有两个可选外部 speculative drafter 的模型：[DSpark](docs/architecture/qwen3_8_27b_fp8_design.md#external-dspark-speculative-decoding) 和 [DFlash2](docs/architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding)，两者互斥，且都与原生 MTP 路径互斥。DFlash2 在其 opt-in 开关全开时，512-token fixture 上实测 full-request 2.78×、decode 3.02×，八个 GSM8K prompt 上聚合 1.33×，且每种情况下 token 完全一致。两个 drafter 都默认关闭，因为收益依赖接受率，而上游公布的 2.67–3.43× 是 decode 延迟比而非 full-request 比。另外还有一个常驻 TP4 worker 会在请求之间保留 prefix state，因此下一个 prompt 是上一个的追加或压缩的客户端只需为增量付费。

### Ternary-Bonsai-2-27B ternary GGUF runtime（served，单卡）

服务 Qwen3.8 的同一个引擎也服务这个 checkpoint，因为它就是同一架构、同一形状 —— 区别在容器。一张
RTX 2080 Ti，发布的 `PTQ1_0` 文件，同时只跑一个请求，greedy：

- 4,096-token prompt：**prefill 636.0 tok/s**（6.44 s），decode 25.89 tok/s。
- 8,192-token prompt：prefill 639.1 tok/s（12.82 s），decode 25.44 tok/s。
- 同卡同 artifact 的上游 `llama-bench`：4,096 时 prefill 642.5 tok/s（8,192 时 615.2），
  decode 30.7 tok/s —— 也就是 **prefill 与上游持平**，decode 是其 84%。

**prompt 的 token 数不是 64 的整数倍时，最后一个残缺 tile 会一次性多花最多 12 秒**，这正是这个
checkpoint 早先那次测量被记成“内核慢”的原因：4,097 token 用 18.11 s，而 4,096 token 用 6.44 s。
除此之外每个 token 的开销从 2,048 到 22,378 token 都平在 1.55 ms。这条代价可复现、但机制尚未查明。
卡上内存是 5.53 GiB 权重、含 runtime 共 6,566 MiB，另有 **每 token 64 KiB 的 KV**。

### DeepSeek-V4 C++ FP4 runtime

- 32K prompt：prefill 约 402 tok/s，约 11.2 GiB/rank。
- 64K prompt：prefill 约 401 tok/s，约 14.5 GiB/rank。
- Decode：该 4×RTX 2080 Ti 配置下约 3.7 tok/s。

### MiniMax-M2.7 与 GLM-5.2 GGUF runtime

- MiniMax-M2.7 在 Q4/Q5 MMA 和 IQ2 DP4A 路径后，full-model 256-token prefill 约 104.9–107 tok/s；另一个 fused RMSNorm 的 43-layer decode benchmark 达到 10.32 tok/s。
- GLM-5.2 已通过 raw-block GGUF 路径实现 generation。由于模型规模、expert staging 和逐层同步，其当前 decode floor 明显更低；resident-cache、routed-TP 和 fused-RMSNorm 等实验开关默认不启用。

这些是模型专属结果，不能合并成一个 PocketLLM 总分。

## 架构概览

PocketLLM 包含两类互补执行方式：

1. **GPU-resident 与低 bit 执行：** 在总显存预算允许时，让本地权重或 expert block 常驻 GPU。
2. **异构执行：** 将 routed experts 放在 CPU/NUMA 内存，只把当前 token 或 prefill chunk 激活的量化 block 搬到 GPU。

Runtime 是模型专用的：DeepSeek-V4 使用 MLA/indexing 和 routed-expert 调度；DeepSeek-V4.1-Flash 使用 causal encoder-decoder 与 CSA2 shared-KV attention，其 checkpoint 的 268.95 GiB routed expert 和 189.13 GiB Engram 表留在 host 内存或磁盘上；MiMo-V2.6-Flash 使用 global attention 与带 per-head sink 的滑窗 attention 的混合结构，149.81 GiB MXFP4 expert 放在共享 host bank 里，attention 按 checkpoint 自带的四路划分切开；MiniMax-M2.7、GLM-5.2 使用 GGUF raw-block 路径；Qwen3.8 使用 Safetensors FP8 online unpacking 加 hybrid linear/full attention；Ternary-Bonsai-2-27B 是同一种 hybrid attention 装在一个 1.75 bit 的 GGUF 里，tensor 端到端按 ternary 消费，文件声明的 incoherence 旋转作用在激活上。设计上的热路径不会将完整量化权重展开成 FP32 副本。

## 快速开始

### 构建 Python extensions

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
```

Python package metadata 现在使用 `pocketllm`；为了兼容性，现有 `src.*` Python import namespace 不变。

### 构建 C++/CUDA engine

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

可执行文件为 `pocketllm_engine`：

```text
build/cpp_engine/pocketllm_engine
```

它原名 `dsv4_cpp_engine`。该改名与 `pocket::` 命名空间、`POCKETLLM_*` 环境变量一起构成破坏性变更，
详见[迁移说明](docs/migration/dsv4-to-pocket-rename.md)。

### 运行 DeepSeek-V4 C++ TP4 serving

```bash
CKPT=/path/to/DeepSeek-V4-Flash \
PORT=8000 \
MAX_CONTEXT=8192 \
PYTHON=python \
bash scripts/run_cpp_serve_tp4.sh
```

该命令让 rank 0 运行 OpenAI 兼容服务，rank 1–3 运行 NCCL worker。

### 运行 DeepSeek-V4.1-Flash OpenAI serving

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python -m pocketllm serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --backend v41 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000 \
  --backend-option expert_pool_rows=148 \
  --backend-option prefill_chunk=4096 \
  --backend-option decode_graphs=true \
  --backend-option threads=22
```

CLI 自带的 supervisor 会一 rank 起一个进程 —— rank 0 绑定 listener，rank 1–3 是 NCCL worker —— 四个都报告 ready 之后 server 才会应答。启动不算快：开了 `DEEPSEEK_V41_RESIDENT_EXPERTS=1` 后，每个 rank 要 pin 自己在 457.8 GiB expert bank 中的份额，约 100 s 一个 rank，48 个 shard 再花约 130 s 加载。同样的 flag 配 `--max-model-len 262144` 就是 runtime 能接受的最长上下文，也是上面长上下文数字的测量配置。

### 运行 Ternary-Bonsai-2-27B OpenAI serving

```bash
python -m pocketllm serve \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --served-model-name bonsai \
  --max-model-len 245760 \
  --port 8000
```

不需要 backend flag，也不需要 tensor parallel flag：checkpoint 就是一个 `.gguf` 文件，adapter 从
它的 header 读出 `general.architecture=qwen35`，选中声明了该名字的原生引擎；tokenizer、special
token id 和 chat template 也都来自同一个 header。一张卡装得下，而 `--max-model-len` 是内存决策
而不只是上下文决策 —— KV 是**每 token 64 KiB**：245,760 token 是 22 GiB 卡在 5.53 GiB 权重旁边
能放下的最大 FP16-KV 上下文，`--kv-cache-dtype fp8` 把 KV 减半，于是 checkpoint 自带的 262,144
也放得下。

### 运行 MiMo-V2.6-Flash OpenAI serving

```bash
python -m pocketllm serve \
  --backend mimo \
  --model /path/to/MiMo-V2.6-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --port 8000 \
  --backend-option prefill_chunk=2048 \
  --backend-option chunk_rows=16
```

同一套 supervisor、同样四个进程。第一次启动会把 149.81 GiB expert bank 从发布版灌进 `/dev/shm`，213 MiB/s 约需 12 分钟；之后再启动是 attach 已有的段，0.07 s，而且四个 rank attach 的是同一份。routed expert 从 bank 里读，所以卡上只有 dense 权重、attention 和两个 expert arena —— **262144 上下文时每卡 10.21 GiB**。`rm -rf /dev/shm/pocketllm_mimo_experts_*` 可以把这段内存还回去。

### 通过共享 raw-block CLI 运行 GGUF 模型

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_gguf \
  --gguf-path /path/to/model.gguf \
  --seed-file /path/to/prompt_tokens.bin \
  --max-new-tokens 32 \
  --prewarm
```

GLM-5.2 文本 prompt：

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_glm \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --prompt "你好" \
  --chat \
  --max-new-tokens 32 \
  --prewarm
```

### 检查 GGUF checkpoint

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/model.gguf \
  --architecture auto \
  --spec-summary \
  --validate-spec \
  --capability-report \
  --placement-report
```

### 运行 Qwen3.8-27B-FP8 C++ smoke/benchmark

Qwen 路径支持 text prompt 或 token IDs，并通过 NCCL ID 文件启动 TP4 rank：

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "请用一段话解释 tensor parallelism。" \
    --generate-token 123 --max-new-tokens 32 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

正常运行时使用相同的 Qwen smoke 参数；rank 0 会输出 `prefill_tokens_per_s`、`decode_tokens_per_s`、resident weight bytes 和 GPU memory。可以使用真实 checkpoint 验证原生 Qwen 文本 server：

```bash
python scripts/verify_cpp_qwen_openai.py \\
  --ckpt /path/to/Qwen3.8-27B-FP8 \\
  --binary build/cpp_engine/pocketllm_engine \\
  --python /path/to/python-with-transformers \\
  --sidecar src/server/cpp_sidecar.py \\
  --devices 0,1,2,3
```

该 harness 会检查 health、model discovery、非流式和流式的 chat 与 text completion、固定 sampling 校验、请求字段拒绝、stop 序列截断，以及并发 scheduler admission。

对于单并发客户端，如果后续请求会追加或压缩上一次请求，使用长期存活的 TP4 token-ID worker。rank 0 读取 `<max_new_tokens> token0 token1 ...`，并输出 exact prefix 统计；追加请求复用 live state，分叉请求从 GPU snapshot 恢复：

```bash
python scripts/bench_qwen_prefix_cache.py \\
  --ckpt /path/to/Qwen3.8-27B-FP8 \\
  --token-ids-file /path/to/prompt_ids.csv \\
  --max-context 32768 \\
  --max-new-tokens 4 \\
  --compression-prefix-tokens 4096
```

benchmark 会启动 rank 1–3 command worker，让 rank 0 在多轮请求间保持 engine。使用 `--disable-prefix-cache` 做 cold parity A/B。一次性 Qwen 命令只处理单个请求，因此默认不创建 prefix snapshot；`--qwen-persistent-stdin` 开启缓存，`--qwen-no-prefix-cache` 可显式关闭。

## 文档

文档站点已发布在 **<https://lvyufeng.github.io/PocketLLM/>**，由本仓库的 `docs/` 目录构建，
支持全文搜索和分主题导航，内容与下面的文件一致。

- [文档总览](docs/README.md)
- [快速开始](docs/getting-started.md)
- [模型支持矩阵](docs/models/README.md)
- [Benchmark 口径](docs/guides/benchmarking.md)
- [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md)
- [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md)
- [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md)
- [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md)
- [DeepSeek-V4](docs/models/deepseek-v4.md)
- [MiniMax-M2.7](docs/models/minimax-m2.7.md)
- [GLM-5.2](docs/models/glm-5.2.md)
- [在 OpenAI server 后服务 V4.1](docs/performance/deepseek_v4_1_flash_served_gate.md)
- [一个 V4.1 请求的代价](docs/performance/deepseek_v4_1_flash_single_request_capability.md)
- [V4.1 的跨请求 prefix caching](docs/architecture/v41_prefix_cache.md)
- [DSpark speculative decoding](docs/performance/dspark.md)
- [FlashMemory 1M context](docs/performance/flashmemory_1m_context.md)
- [MiniMax decode bottleneck 分析](docs/performance/minimax_decode_bottleneck_analysis.md)
- [历史 2080 Ti 报告](docs/reports/dsv4_2080ti_report.pdf)

## Roadmap

- [x] DeepSeek-V4 FP4/FP8 与 GGUF Q2/IQ2/IQ1 generation 路径。
- [x] MiniMax-M2.7 与 GLM-5.2 GGUF raw-block generation 路径。
- [x] Qwen3.8-27B-FP8 C++ TP4 文本 runtime。
- [ ] 在不破坏现有脚本的前提下，统一 C++ model dispatch 和 binary 命名。
- [x] Qwen OpenAI 兼容文本 serving adapter。
- [x] OpenAI server 后的 DeepSeek-V4.1-Flash TP4 文本生成，以及跨请求 prefix caching。
- [x] OpenAI server 后的 MiMo-V2.6-Flash TP4 文本生成：host 常驻 expert bank、按 checkpoint 自带划分切开的 attention、256k 上下文。
- [x] Ternary-Bonsai-2-27B：1.75 bit ternary GGUF 在**单卡**上服务，5.53 GiB 权重换来 636 tok/s prefill 和 245,760 token 上下文。
- [ ] 在实测有收益时接入 CUDA Graph 和 persistent decode dispatch。
- [ ] 增加更多模型 benchmark fixture 和自动化 regression dashboard。

## 已知限制

- 性能高度依赖 GPU 型号、PCIe 拓扑、NUMA placement、驱动/runtime 版本和 checkpoint 变体。
- PCIe 系统上的 GGUF expert staging 可能主导 decode；prefill TPS 高不代表 decode TPS 高。
- DeepSeek-V4.1-Flash 一次只服务一个请求：`--backend v41` 持有一把请求锁并报告 `supports_batch=False`，因此没有 continuous batching、没有 chunked prefill、也没有 paged KV pool。后续请求如果与已服务过的请求共享 prefix，确实会复用它，但这改变的是一个请求的代价，而不是同时能跑几个。
- MiMo-V2.6-Flash 同样一次只服务一个请求，但原因更硬：每个 routed 层都在每个 rank 程序的同一点以 all-reduce 收尾，因此一个 rank 若没在跑同伴们跑的那个请求，它并不是空闲，而是停在另一个 collective 上，NCCL 遇到不匹配只会挂住。所以 rank 0 必须先广播整个请求再开始生成，cancel 和 stop 字符串也必须由四个 rank 达成一致，而不能由某一个自行处理。
- MiMo-V2.6-Flash 的 attention 和 dense linear 还是 torch 而非 kernel，且 decode 一步的下界由 expert 拷贝决定：256k 下 197.2 ms 的一步里有 99.9 ms 是 kernel 等到的 H2D，已经贴着 PCIe 3.0 链路速率。唯一能把它藏起来的调度需要一个 router 给不出的预测 —— 用上一步抽取来预取，实测行命中率只有 9–13.5%。
- DeepSeek-V4.1-Flash 在这条路径上没有 speculative decoding。checkpoint 带有三个 MTP 层和一个 DSpark draft head，loader 有意把它们全部留在 shard 里。
- DeepSeek-V4.1-Flash 的验证依据是生成的文本，而不是 logit 对比。参考栈需要 `torch>=2.10.0` 和 `tilelang==0.1.8`，本机都没有，因此对 V4.1 的 forward pass 不存在数值 oracle。
- DSpark 当前 C++ verify path 是 sequential，不应宣称为加速路径；multi-token verify 有独立的数值漂移策略。
- Qwen runtime 当前只支持 text checkpoint 路径，视觉输入和多模态 serving 尚未实现。
- Ternary-Bonsai-2-27B 有一条实测但未解释的 prefill 代价：prompt 的 token 数不是 64 的整数倍时，最后一个残缺 tile 会一次性多花最多 12 秒 —— 4,097 token 要 18.11 s，而 4,096 token 只要 6.44 s。对齐后 prefill 平在 1.55 ms/token，与上游参考持平。它的 batch scheduler 默认关闭，因为换来的是聚合吞吐而非单请求延迟；投机解码在该 artifact 上未验证；它的 prefix 复用是“复用上一个请求”，不是存储。
- 部分实验优化在真实端到端测试出现回归后被保留为 opt-in 或关闭；具体见模型页和历史分析文档。

## License

PocketLLM 采用 [MIT License](LICENSE) 发布。你可以自由使用、修改和分发本代码，包括商业用途，只需保留版权声明和许可声明。

模型权重、tokenizer、CUDA、PyTorch、GGUF 资源和其他第三方组件分别受其自身许可证约束。PocketLLM 代码许可证不授予任何第三方模型资产的额外权利。

## 致谢

PocketLLM 基于 CUDA、PyTorch、safetensors、GGUF、Transformers、NCCL 和 llama.cpp 量化研究。仓库中的模型专用 runtime 与 benchmark，是面向消费级硬件可复现本地推理的工程实践。
