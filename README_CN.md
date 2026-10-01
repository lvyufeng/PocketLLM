# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[English](README.md) | 中文

PocketLLM 把大模型跑在**单个加速器**上——一张消费级显卡，或者端侧、手机这类目标。模型装得下就整卡常驻；装不下就继续降位宽，而不是加第二张卡。

> **本仓的定位已经收紧。** 它以前描述的是整个多卡栈。那个栈——tensor parallel、expert parallel、host offload、多卡 serving 路径——现在在 **[RelicLLM](https://github.com/lvyufeng/RelicLLM)**，原生 kernel 在 **[relic-core](https://github.com/lvyufeng/relic-core)**，退役的 C++ engine 归档在 **[relic-engine](https://github.com/lvyufeng/relic-engine)**。代码还没有切；本 README 和文档说明的是边界将落在哪里。

> **项目状态：** 研究和工程软件。本仓的每个数字都来自特定 checkpoint 和硬件配置的实测，不代表通用性能保证。

## 规则

**装不下就降位宽——不 offload，也不拆到多卡。**

位宽阶梯是 Q4 → Q2 → IQ2 → IQ1 → ternary，按这个顺序往下走，停在模型还能正确作答的最低一档。host offload 和多卡 tensor parallel 明确不在范围内：实测 hybrid GPU/CPU expert 路径比 experts 常驻卡上慢 **2.3×**，而一个需要四张卡才能回答问题的 checkpoint 是另一个产品。

有三类 checkpoint 能整卡装下：

| 模型 | 格式 | 占用 |
| --- | --- | --- |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0` —— 每权重 1.75 bit | **5.53 GiB** |
| [Xing4.0-29B-A4B](docs/models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` —— 每权重 4.5 bit | **17.94 GiB** |
| [DeepSeek-V4 GGUF Q2](docs/models/deepseek-v4-gguf-q2-single-gpu.md) | GGUF Q2 / IQ2 / IQ1 | 一张 22 GiB 卡 |

## News

- [2026/09] [Ternary-Bonsai-2-27B 单卡端到端服务](docs/models/ternary-bonsai-2-27b.md)
- [2026/09] [Xing4.0-29B-A4B 单卡端到端服务](docs/models/xing4.0-29b-a4b.md)
- [2026/09] [DeepSeek-V4.1-Flash 四卡端到端服务](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.1-flash.md)
- [2026/09] [Qwen3.8-27B-FP8 有了原生 OpenAI 兼容服务端](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/qwen3.8-27b-fp8.md)
- [2026/05] [DeepSeek-V4 GGUF Q2 单卡运行](docs/models/deepseek-v4-gguf-q2-single-gpu.md)

多卡条目只作衔接列出，它们的记录在 [RelicLLM 的文档](https://lvyufeng.github.io/RelicLLM/)里。

## 安装

### 完整安装

```bash
# 先装构建前置依赖：构建过程是从环境里 import 它们，不会自己去拉。
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11

pip install pocketllm --no-build-isolation
```

会编译 CUDA 扩展和原生 C++ engine，耗时 5–15 分钟。

**环境要求：**
- Python >= 3.10
- PyTorch >= 2.0, < 2.7（先装：`pip install "torch>=2.0,<2.7"`）
- CUDA toolkit 11.8+（GPU 加速）
- CMake >= 3.18、pybind11 >= 2.10、Ninja >= 1.11
- `setuptools >= 68` 和 `wheel`
- 16 GB+ 内存（编译用）

**注意：** `--no-build-isolation` 是必须的，它让构建用你环境里的 PyTorch——后者必须与你的 CUDA toolkit 版本匹配。代价是 pip 不会去拉上面那些构建前置依赖，你必须在安装前把它们装进环境。新建的 virtualenv 一个都没有：`python -m venv` 只带解释器自带的 `setuptools`，Python 3.10 上它比提供 `bdist_wheel` 命令的版本旧，Python 3.12+ 上则根本不装 setuptools——所以安装会先在元数据生成阶段以 `invalid command 'bdist_wheel'` 失败，再在原生 engine 构建阶段因为缺 `pybind11` 或 `cmake` 失败。上面那行 `pip install "torch…"` 会把它们都装上。

### 仅 PyTorch 安装（跳过 C++ engine）

只需要 PyTorch 后端，或者缺 C++ 构建依赖时：

```bash
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

跳过 C++ engine，但仍然会编译 PyTorch CUDA 扩展，所以依然需要 `torch` 和一个够新的 `setuptools`。

### 开发安装

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11
pip install -e . --no-build-isolation
```

## 快速开始

### Python API

```python
from pocketllm import LLM

llm = LLM(
    model="/path/to/checkpoint",
    backend="auto",  # 或 "torch"、"cpp"
    tensor_parallel_size=1,
)

result = llm.generate("What is artificial intelligence?")
print(result.text)

for token in llm.stream("Explain quantum computing"):
    print(token.text, end="", flush=True)
```

### OpenAI 兼容服务端

```bash
# 单卡，默认路径
pocketllm serve --model /path/to/checkpoint --backend auto

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pocketllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

`--tensor-parallel-size` 在 CLI 里还在，取值大于 1 今天也还能跑——但那是多卡路径，属于 [RelicLLM](https://github.com/lvyufeng/RelicLLM)，不是本库的目标。

## 什么时候用 PocketLLM

**PocketLLM 面向：**
- ✅ 一张消费级显卡（RTX 2080 Ti、3090、4090）跑一个**整卡常驻**的 checkpoint，没有 host bank，没有第二个进程
- ✅ 把量化当作"装得下"的手段——GGUF Q4/Q2/IQ2/IQ1、FP4，以及一份 1.75-bit 的 ternary GGUF，它的权重始终按 ternary 消费、从不 upcast 成 fp32
- ✅ 低延迟单请求推理：prefill 与 decode 分派路径独立，优化其中一个不会伤到另一个
- ✅ 端侧与手机目标——那里"装得下"是硬约束，不是一个可调参数

**如果你需要以下能力，请看 [RelicLLM](https://github.com/lvyufeng/RelicLLM)：**
- ❌ **多卡。** tensor parallel、expert parallel、host expert bank、CPU/NUMA placement 都是 RelicLLM 的，这是刻意的——本仓不往那边走。
- ❌ **超过一张卡的 checkpoint。** 这里的答案是继续降位宽，不是加卡。

**如果你需要**广泛的模型覆盖、multi-LoRA、多模态输入或生产级调度特性，请看 vLLM 或 SGLang。PocketLLM 是一份很短的 checkpoint 清单配上很深的模型专用优化，不是一个通用后端。

## PocketLLM 提供什么

- **单卡执行。** checkpoint 常驻一张卡。没有 host bank，没有第二个进程，没有需要同步的集合通信。
- **低 bit 执行，不做无谓展开。** GGUF Q4/Q2/IQ2/IQ1、FP4、FP8 E4M3 与 1.75-bit ternary 格式都在热路径上直接消费量化块；该省的地方不会把原始权重展开成完整 FP32 拷贝。
- **一条低到 1.75 bit 的量化阶梯。** Ternary-Bonsai-2-27B 把 27B 模型放进 **5.53 GiB**，于是 22 GiB 的卡上还剩下 245,760 token 的上下文空间。
- **一个 experts 全常驻的 29B MoE。** Xing4.0-29B-A4B 的 **17.94 GiB** `IQ4_NL` 权重——38 个 MoE 层全部 64 个 expert——整卡装下，decode 时没有 expert 目录要查。
- **prefill / decode 分派。** 大行 kernel 与单 token 延迟路径各自优化。
- **检查与验证工具：** GGUF 架构/spec 报告、Safetensors 审计、张量形状检查、数值 parity 测试、真实 checkpoint benchmark。

## 支持的模型

每个名字链到它的模型页，那上面写着数字的测量条件和一个 `## Known limitations` 小节。四类 checkpoint 是单卡的，也是本仓的目标：

| 模型 | 格式 | Runtime | 数字 |
| --- | --- | --- | --- |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF ternary，1.75 bit | 原生 C++/CUDA，**单卡**，无需 flag | prefill 636 tok/s，decode 26 tok/s，常驻 5.53 GiB |
| [Xing4.0-29B-A4B](docs/models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` | **单卡**，64 个 expert 全常驻 | prefill 75.22 tok/s，decode 6.72 tok/s，常驻 17.94 GiB |
| [DeepSeek-V4 GGUF Q2](docs/models/deepseek-v4-gguf-q2-single-gpu.md) | GGUF Q2 / IQ2 / IQ1 | PyTorch 与 C++/CUDA，一张 22 GiB 卡 | 32K–64K 上约 401 tok/s prefill |
| [DeepSeek-V4-Flash](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.md) | Safetensors FP4/FP8 | 其中的 GGUF Q2/IQ2/IQ1 单卡路径 | 见 [RelicLLM 页面](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.md) |

多卡的 runtime——DeepSeek-V4.1-Flash、MiMo-V2.6-Flash、Qwen3.8-27B、MiniMax-M2.7——模型页在 [RelicLLM](https://lvyufeng.github.io/RelicLLM/)。本仓的[支持矩阵](docs/models/README.md)只覆盖单卡部分。

`inspect`、`smoke` 和一次 benchmark 并不自动等于生产级 serving 保证。

## 架构

一张卡，checkpoint 整卡常驻。设计围绕的是**装得下**：这个 checkpoint 还能正确作答的最低格式是哪一档，以及它给 KV cache 留下多少空间。

- **Ternary-Bonsai-2-27B** 是一个 27B 混合注意力模型——48 层 Gated DeltaNet 加 16 层 GQA，上面是 dense MLP——装在一份每权重 1.75 bit 的 GGUF 里，文件声明的 Hadamard 旋转作用在激活上。22 GiB 的卡上，5.53 GiB 权重换来的几乎是完整的 262,144 token 上下文。
- **Xing4.0-29B-A4B** 是一个 29B MoE——MLA 注意力、64 个 routed expert 取 top-4 加一个 shared，每个 block 还有四条 residual stream 由一个矩阵 hyper-connection 混合——官方 `IQ4_NL` GGUF 整卡装下。它的 residual stream 比 sublayer 带得更宽，因为这个 checkpoint 的激活超出 fp16 范围。
- **DeepSeek-V4** 的 GGUF Q2/IQ2/IQ1 把全精度路径的 MLA/indexing 与 routed-expert 调度塞进一张卡的内存预算里。

热路径上不会把量化权重展开成完整 FP32 拷贝。

## 文档

发布在 **<https://lvyufeng.github.io/PocketLLM/>**，由本仓 `docs/` 构建，支持全文检索和按主题导航。

- [文档索引](docs/README.md)
- [快速上手](docs/getting-started.md)
- [模型支持矩阵](docs/models/README.md)
- [架构与老硬件 roadmap](docs/architecture/pocketllm_roadmap_old_hardware.md)
- [2080 Ti 上的新模型支持](docs/architecture/pocketllm_new_model_roadmap.md)
- [PyPI 发布](docs/guides/pypi_release.md)
- [2080 Ti 历史报告](docs/reports/dsv4_2080ti_report.pdf)

主题属于多卡 runtime、算子层或退役 engine 的页面不在这里——它们待在所描述代码的旁边，上面的链接和站内各处链接都指向那里。

## Roadmap

- [x] Ternary-Bonsai-2-27B：1.75-bit ternary GGUF 在**单卡**上服务，5.53 GiB 权重换来 prefill 636 tok/s 与 245,760 token 上下文。
- [x] Xing4.0-29B-A4B：38 个 MoE 层全部 64 个 expert 在**单卡**常驻。
- [x] DeepSeek-V4 GGUF Q2/IQ2/IQ1 单卡生成。
- [ ] 把多卡代码从本仓切出去，交给 RelicLLM。
- [ ] 端侧与手机后端——"单卡"所代表的那个真正的目标。
- [ ] 在实测有收益的地方引入 CUDA Graph 与 persistent decode dispatch。
- [ ] 更多单卡 benchmark fixture 与自动化回归看板。

## 已知限制

下面三条会改变一个数字的含义，而不只是给它加个前提。每个模型页末尾各自还有一份 `## Known limitations`。

- **prefill 速率不蕴含 decode 速率，任何数字都不能跨配置迁移。** PCIe 拓扑、NUMA 位置、driver 与 toolkit 版本、checkpoint 变体和 warm state 都会改变结果。见[benchmark 与报告规则](https://github.com/lvyufeng/RelicLLM/blob/master/docs/guides/benchmarking.md)。
- **Ternary-Bonsai-2-27B 在 prompt 不是 64 token 整数倍时要付一次性代价**：4,097 token 的 prefill 花 18.11 s，而 4,096 token 只要 6.44 s。多一个 token 带来 3× 误差，可测量、可复现，机理尚未定位——见[模型页](docs/models/ternary-bonsai-2-27b.md#known-limitations)。
- **多卡路径仍然会构建。** `--tensor-parallel-size 4`、`--backend v41`、`--backend mimo` 在代码切完之前都还在这棵树里。按定位它们属于 RelicLLM，本仓维护的是上面那些单卡路径。

## 许可证

PocketLLM 以 [MIT License](LICENSE) 发布。你可以自由使用、修改和分发代码，包括商业用途，但需保留版权声明和许可声明。

模型权重、tokenizer 文件、CUDA、PyTorch、GGUF 资产及其他第三方组件受各自许可证约束。PocketLLM 的代码许可证不授予对第三方模型资产的额外权利。

## 致谢

PocketLLM 构建在 CUDA、PyTorch、safetensors、GGUF、Transformers、NCCL 以及 llama.cpp 的量化研究之上。各模型专用 runtime 与 benchmark 是为了在消费级硬件上可复现本地推理而做的工程工作。