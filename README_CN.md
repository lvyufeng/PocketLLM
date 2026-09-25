# PocketLLM

[English](README.md) | 中文

PocketLLM 是一个面向消费级多卡系统的大模型推理工程栈，包含 C++/CUDA 与 PyTorch runtime。它结合模型专用 kernel、低 bit 格式、tensor/expert parallel、CPU/GPU placement，以及面向单请求的可复现实测 benchmark。

项目最初来自在 4×RTX 2080 Ti 上运行 DeepSeek-V4 的工程实践，目前已经包含 DeepSeek-V4、MiniMax-M2.7、GLM-5.2、Qwen3.8-27B、DeepSeek-V4.1-Flash、MiMo-V2.6-Flash 和 Ternary-Bonsai-2-27B 的已验证 runtime。PocketLLM 不是一个“所有模型共用同一后端”的框架：不同模型使用与其架构和 checkpoint 格式匹配的执行路径。

其中四个模型已经通过 OpenAI 兼容 API 端到端服务：**Qwen3.8-27B-FP8** 走原生 C++ runtime，**Ternary-Bonsai-2-27B** 走同一个 runtime 且只用**一张**卡，**DeepSeek-V4.1-Flash** 走 `pocketllm serve --backend v41`，**MiMo-V2.6-Flash** 走 `pocketllm serve --backend mimo`。四条路径都在真实 checkpoint 上做过验证。

> **项目状态：** 研究和工程软件。下面的数字来自特定 checkpoint、硬件和测试口径，不代表通用性能保证。

## News

- [2026/09] [Ternary-Bonsai-2-27B 单卡端到端可服务](docs/models/ternary-bonsai-2-27b.md)
- [2026/09] [MiMo-V2.6-Flash 四卡端到端可服务](docs/models/mimo-v2.6-flash.md)
- [2026/09] [DeepSeek-V4.1-Flash 端到端可服务](docs/models/deepseek-v4.1-flash.md)
- [2026/09] [Qwen3.8-27B-FP8 有了原生 OpenAI 兼容 server](docs/models/qwen3.8-27b-fp8.md)
- [2026/08] [Qwen3.8-27B 接上两个外部投机 drafter](docs/models/qwen3.8-27b-fp8.md#optional-speculative-decoding)

[更早的条目与每条的实测数字 →](https://lvyufeng.github.io/PocketLLM/#news)

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

每个模型都有与其架构和 checkpoint 格式匹配的 runtime，每个名字都链到它的模型页。表里的数字是该页
的 headline，不是这里的独立 benchmark —— 记录是那一页，连同它的测量条件和 `## Known limitations`。

| 模型 | 格式 | Runtime | 状态 | Headline |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md) | Safetensors FP8 + FP4 | `--backend v41`，host PyTorch，TP4 | 文本 + OpenAI server | 260k prompt prefill 150–152 tok/s |
| [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md) | Safetensors FP8 + MXFP4 | `--backend mimo`，host expert bank，TP4 | 文本 + OpenAI server | 262k prompt prefill 104 tok/s |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | C++/CUDA，TP4 | 文本 + OpenAI server | prefill 865 tok/s，decode 43 tok/s |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF ternary，1.75 bit/权重 | C++/CUDA，**单卡**，无需 flag | 文本 + OpenAI server | prefill 636 tok/s，decode 26 tok/s |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8，GGUF Q2 | PyTorch 与 C++/CUDA，TP4 | 文本 + server（C++/PyTorch） | C++ FP4 prefill 约 401 tok/s |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | Raw-block CUDA，TP4 | 文本，仅 CLI | 256-token prefill 约 105 tok/s |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | Raw-block CUDA，TP4 | 文本，仅 CLI | prefill 约 0.79 tok/s |

[Qwen3.8-27B](docs/models/qwen3.8-27b-fp8.md) 一行还覆盖同一文本架构下的
[NVFP4](docs/models/qwen3.8-27b-nvfp4.md) 和[官方 BF16](docs/models/qwen3.8-27b-bf16.md) 两个
checkpoint；上面这张表的完整八列版本、带格式与验证细节，是[支持矩阵](docs/models/README.md)。

模型页面会把“模型架构规格”和“PocketLLM 当前实际实现能力”分开。`inspect`、`smoke` 和 benchmark 也不自动等于 production serving 保证。

## 性能

这里发布的每一个数字都是一次测量 —— 一个 checkpoint、一套硬件、一种测法 —— 从来不是性能承诺。
结果紧挨着产生它的 runtime 存放：每个模型页都有自己的 Performance 段，连同测量条件和不能做的对比。
更长的记录独立放在[性能记录](docs/performance/index.md)下，其中包括
[DeepSeek-V4.1-Flash served 运行记录](docs/performance/deepseek_v4_1_flash_served_gate.md)和
[Qwen 并发验证](docs/performance/cpp_openai_concurrency_validation.md)。

比较这个 repository 里任意两个结果之前，先读 [Benchmark 口径](docs/guides/benchmarking.md)。

## 架构概览

PocketLLM 包含两类互补执行方式：

1. **GPU-resident 与低 bit 执行：** 在总显存预算允许时，让本地权重或 expert block 常驻 GPU。
2. **异构执行：** 将 routed experts 放在 CPU/NUMA 内存，只把当前 token 或 prefill chunk 激活的量化 block 搬到 GPU。

Runtime 是模型专用的：DeepSeek-V4 使用 MLA/indexing 和 routed-expert 调度；DeepSeek-V4.1-Flash 使用 causal encoder-decoder 与 CSA2 shared-KV attention，其 checkpoint 的 268.95 GiB routed expert 和 189.13 GiB Engram 表留在 host 内存或磁盘上；MiMo-V2.6-Flash 使用 global attention 与带 per-head sink 的滑窗 attention 的混合结构，149.81 GiB MXFP4 expert 放在共享 host bank 里，attention 按 checkpoint 自带的四路划分切开；MiniMax-M2.7、GLM-5.2 使用 GGUF raw-block 路径；Qwen3.8 使用 Safetensors FP8 online unpacking 加 hybrid linear/full attention；Ternary-Bonsai-2-27B 是同一种 hybrid attention 装在一个 1.75 bit 的 GGUF 里，tensor 端到端按 ternary 消费，文件声明的 incoherence 旋转作用在激活上。设计上的热路径不会将完整量化权重展开成 FP32 副本。

## 快速开始

安装方式和坑见 [Getting started](docs/getting-started.md)。从源码构建：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation .
```

### Python API

```python
from pocketllm import LLM

llm = LLM(
    model="/path/to/checkpoint",
    backend="auto",  # 或 "torch"、"cpp"
    tensor_parallel_size=1,
)

print(llm.generate("What is artificial intelligence?").text)

for token in llm.stream("Explain quantum computing"):
    print(token.text, end="", flush=True)
```

### OpenAI 兼容 server

```bash
pocketllm serve \
    --model /path/to/checkpoint \
    --backend auto \
    --tensor-parallel-size 4
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pocketllm", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### Tensor Parallel（多卡）

```bash
pocketllm serve \
    --model /path/to/qwen-27b-fp8 \
    --backend cpp \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

各模型的启动命令、可调项和 standalone engine 的用法在[模型页](docs/models/README.md)和
[Getting started](docs/getting-started.md)；C++ 引擎的独立构建见
[cpp_engine/README.md](cpp_engine/README.md)。

## 文档

文档站点已发布在 **<https://lvyufeng.github.io/PocketLLM/>**，由本仓库的 `docs/` 目录构建，
支持全文搜索和分主题导航，内容与下面的文件一致。模型页已在上面的模型表里逐行链接，这里列的是
不是模型页的入口。

- [文档总览](docs/README.md)
- [快速开始](docs/getting-started.md)
- [模型支持矩阵](docs/models/README.md)
- [Benchmark 口径](docs/guides/benchmarking.md)
- [原生引擎 API 与 backend](docs/guides/pocketllm_api.md)
- [架构总览](docs/architecture/index.md)
- [性能记录](docs/performance/index.md)
- [在 OpenAI server 后服务 V4.1](docs/performance/deepseek_v4_1_flash_served_gate.md)
- [OpenAI 并发验证](docs/performance/cpp_openai_concurrency_validation.md)
- [DSpark speculative decoding](docs/performance/dspark.md)
- [FlashMemory 1M context](docs/performance/flashmemory_1m_context.md)
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

这三条会改变一个数字的含义，而不只是给它加限定。每个模型页末尾都有各自的
`## Known limitations` 列出其余部分。

- **prefill 的速率不代表 decode 的速率，任何数字都不能跨配置搬用。** PCIe 拓扑、NUMA placement、
  驱动与 toolkit 版本、checkpoint 变体和 warm state 都会改变结果；GGUF expert staging 尤其可能
  压住 decode 而 prefill 看起来很健康。见 [Benchmark 口径](docs/guides/benchmarking.md)。
- **Ternary-Bonsai-2-27B 在 prompt token 数不是 64 的整数倍时会付一次性代价**：4,097 token 要
  18.11 s，而 4,096 token 只要 6.44 s。多一个 token 造成 3 倍误差，可复现但机制尚未查明 ——
  [模型页](docs/models/ternary-bonsai-2-27b.md#known-limitations)。
- **不是每个 backend 都做 batching。** 原生 C++ runtime 有 request scheduler、paged KV pool 和一次
  覆盖整批的 batched decode step；`--backend v41` 和 `--backend mimo` 各持一把请求锁，一次服务一个。
  见[并发验证记录](docs/performance/cpp_openai_concurrency_validation.md)。

部分实验性优化在真实端到端出现回归后被保留为 opt-in 或关闭；具体哪个见各模型页。

## License

PocketLLM 采用 [MIT License](LICENSE) 发布。你可以自由使用、修改和分发本代码，包括商业用途，只需保留版权声明和许可声明。

模型权重、tokenizer、CUDA、PyTorch、GGUF 资源和其他第三方组件分别受其自身许可证约束。PocketLLM 代码许可证不授予任何第三方模型资产的额外权利。

## 致谢

PocketLLM 基于 CUDA、PyTorch、safetensors、GGUF、Transformers、NCCL 和 llama.cpp 量化研究。仓库中的模型专用 runtime 与 benchmark，是面向消费级硬件可复现本地推理的工程实践。
