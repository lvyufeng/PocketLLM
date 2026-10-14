# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[English](README.md) | **中文**

在**单一加速器**上运行大语言模型 —— 一张显卡、一块边缘板卡，或者口袋里的手机。一个进程独占一个设备；
如果模型放不下，就继续量化，而不是卸载到主机、也不是拆到第二张卡上。

## 能跑什么

Qwen3 通过同一个引擎跑在四类加速器上。真正跑模型的是 C 核（`src/`，`libpocketllm.so`）—— 它读
GGUF、用 checkpoint 自带的分词器切词、走完计算图并解码；权重可以是 `f16`，也可以是打包的
`q4_k`/`q6_k`（在 kernel 内解码，绝不展开）。

| 设备 | 路径 | Qwen3 尺寸 | 权重格式 | 相对参考实现的精度 |
|---|---|---|---|---|
| **CPU**（AVX2） | C 核 | 0.6B、1.7B | `f16`、`q4_k_m` | **与 llama.cpp 逐 token 一致** |
| **1× NVIDIA**（sm_75+） | C 核，eager kernel | 0.6B、1.7B | `f16`、`q4_k_m` | **与 llama.cpp 逐 token 一致** |
| **RDK S600**（Horizon Nash BPU） | 预编译 `.hbm`，经 `xlm` delegate | 0.6B → 8B | `leap_llm` 对称 int4（`w4`/`w8`） | token 前缀一致；处于量化器自身 floor |
| **Orange Pi AIpro**（Ascend 310B1） | C 核，CANN 后端 | 0.6B → 8B | `q4_k_m` | token 一致；**每个尺寸都贴住量化器 floor** |

有两条诚实的界限贯穿整表。C 核那两行是**与 llama.cpp 逐 token 对齐**的；两块板卡**没有 logits 输出面**，
所以「对齐到 1e-3」在它们上只能表达为 **token 一致** —— 而且它们的权重方案与参考实现不同，因此跨板卡的
一致度数字是**量化器溯源**比较，不是硬件排名。S600 不是 C 核的后端：它跑的是主机**预先编译好的图**，
所以按「它执行的产物」列出，而不是按「它实现的算子」列出。

**参考性能 —— Qwen3-1.7B，`q4_k_m`，对比 llama.cpp：** CPU 路径 **prefill 1.04×、decode 1.37×**（两半都
领先）；CUDA 路径是 0.11× / 0.40×，且 kernel 刻意保持朴素。每个数字、测量方法及出处都在
[舰队支持矩阵](https://lvyufeng.github.io/PocketLLM/architecture/fleet_support_matrix/)（英文）。

## 规则

**放不下就量化 —— 不卸载到主机，也不跨卡切分。**

位宽阶梯是 Q4 → Q2 → IQ2 → IQ1 → 三值，按此顺序下降，停在模型仍能正确作答的最低格式。主机卸载与
多卡并行在结构上就被排除：ABI 里没有 rank、没有集合通信、没有第二个设备，`EngineArgs` 里也没有
`tensor_parallel_size` 可设。

## 当前状态

| 部件 | 状态 |
|---|---|
| [`src/`](https://lvyufeng.github.io/PocketLLM/architecture/c_engine/) —— C++ 引擎（`libpocketllm.so`） | **能跑 Qwen3-0.6B 与 1.7B**（`f16` 与 `q4_k_m`），在 `cpu` 和一张 `cuda` 卡上。贪心解码与 llama.cpp 逐 token 对齐；CPU 路径两半都领先 llama.cpp |
| [`src/kernel/ascend/`](https://lvyufeng.github.io/PocketLLM/architecture/ascend_310b/) —— Ascend 310B 后端 | **能跑 Qwen3-0.6B … 8B**（Orange Pi AIpro），贴住量化器自身 floor |
| [`pocketllm.kernels`](https://lvyufeng.github.io/PocketLLM/architecture/kernel_abi_v1/) —— 内核 ABI：描述符、算子 schema、dispatch、图 IR | **完成。** 声明 17 个算子；仅依赖标准库 |
| `pocketllm.backends.reference` —— numpy 参考实现 | **完成。** 规范性实现，覆盖每个已声明算子 |
| [`pocketllm.quant`](https://lvyufeng.github.io/PocketLLM/architecture/execution/) —— GGML 块解码器 | **完成。** 码本表随树内置 |
| `pocketllm.loader.gguf` —— GGUF 读取器 | **完成。** numpy 进，描述符出 |
| [`pocketllm.engine`](https://lvyufeng.github.io/PocketLLM/architecture/execution/) —— 执行器、planner、内存、session | **完成**，基于参考后端 |
| `pocketllm.protocol` / `pocketllm.server` —— OpenAI 兼容 HTTP | **完成。** 由 `server/native_backend.py` 驱动 C 核 |
| `pocketllm.backends.cpu` | **桩。** 仅声明与选择 |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **桩**，各自标明所等待的运行时 |
| [`pocketllm.architectures`](https://lvyufeng.github.io/PocketLLM/models/architectures/) —— 模型 IR 与构建器 | **脚手架。** 只有 `toy` |
| `pocketllm.tokenizer` —— GGUF 词表 BPE | **骨架。** 空白切分可用；BPE 会抛异常 |

**Python 包本身不能跑模型。** 它是**主机侧** —— 内核 ABI 作为规范、numpy 参考实现、加载器、解码器、
执行层 —— 且**没有任何 Python 后端实现了 kernel**：除 `reference` 外每个后端都只是一份声明。两半在
一个地方接上：`python/pocketllm/native.py`，也就是 `ctypes` 桥，`pocketllm run` 与 `pocketllm serve`
都经由它进入 C 核。两者都不是 *Python* 后端，所以上表 `pocketllm.backends.*` 各行仍是桩。

## 安装

基础安装**只有一个依赖**。ABI 只依赖标准库，参考后端依赖 numpy，设备运行时是可选 extra ——
这正是同一个 wheel 既能装到 CUDA 机器上、也能装到手机上的原因。

```bash
pip install pocketllm

# 带设备运行时
pip install "pocketllm[cuda]"      # NVIDIA，经由 torch
pip install "pocketllm[mps]"       # Apple Silicon，经由 torch
```

**要求：** Python >= 3.10。基础安装不需要编译器、不需要 CUDA toolkit。本树没有 `ext_modules`，
没有构建步骤：C 核是一个**独立的原生库**（`src/`，CMake），运行时通过 `ctypes` 加载，不是 Python 扩展。

## 试试看

只读命令在任意主机上都能跑，包括没有任何加速器、也没装 torch 的机器：

```bash
pocketllm devices        # 本机可以打开哪些后端，其余各自缺什么
pocketllm backends       # 每个后端声明了什么，无论本机能否打开
pocketllm architectures  # 本树能够构建的模型结构
pocketllm ops --op gemm_quant --device cuda
```

`devices` 是出问题时才会去跑的命令，所以它也是唯一一条不能依赖任何东西已安装的命令：它背后每一步
检查都是文件系统探测，不导入任何运行时。注意 `available` 与 `implemented` 是两个不同的问题 ——
而且这条命令回答的都是关于 **Python** 后端的问题：某个 `available` 行仍然没有 *Python* kernel，
这里也不描述 `src/`：`pocketllm devices` 并不知道 C 核的存在。

`run` 进入 C 引擎，所以它会真的生成 —— 默认贪心，且只支持 C 核实现的架构（目前是 Qwen3）。它需要
先构建 `libpocketllm.so`；没有它命令会明说，而不是假装：

```bash
cmake -B build -S src && cmake --build build -j8
pocketllm run --model /path/to/qwen3-0.6b-f16.gguf --prompt "The capital of France is" --max-tokens 8
```

`run` 默认贪心，`--temperature` 大于零时采样，旁边还有 `--top-k`、`--top-p`、`--min-p`、`--seed`。
随机数由主机侧提供（`random.Random(seed)`），因为引擎只接受一个均匀分布采样值、自身不持有 RNG，
所以同一个 seed 复现同一段文本。

`serve` 把同一个引擎放到 OpenAI 兼容的 HTTP 之后，用的还是同一个 checkpoint：

```bash
pocketllm serve --model ckpt.gguf --host 0.0.0.0 --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

提示词走 checkpoint **自带的 chat template**（从 GGUF 的 `tokenizer.chat_template` 读出、用 Jinja
渲染）—— 这需要 `jinja2`，而它刻意不是本包的依赖，所以没有它时聊天请求会退回朴素渲染，模型并未为此
训练。采样是逐请求的（`"temperature"`、`"top_p"`、`"top_k"`、`"min_p"`、`"seed"`），`"stop"` 与 `"n"` 同理。

**它一次只服务一个请求。** C `Session` 持有单个 position 和单个 KV cache、无锁，而服务端是
thread-per-request 的 `ThreadingHTTPServer`，所以适配层在锁后排成串行，并声明 `supports_batch = False`，
而不是承诺引擎给不了的并发。实测六个并发请求：加锁 6/6 正确，不加锁 0/6 —— 且两种情况都不报错，这正是
这把锁不是「可优化项」的原因。取消出于同类原因被拒绝。没有算子支撑的字段 —— `logprobs`、惩罚项、
`logit_bias`、`response_format`、`echo`、`suffix`、`best_of` —— 会以 `400` 指名拒绝，而不是静默忽略。

## 为什么再造一个推理引擎

多数运行时把 checkpoint 铺到多张卡上；PocketLLM 做的是相反的事 —— 让**一张**卡，或者一部手机，就足够。
这一个约束改变的是设计，而不是调参：

- **可移植的内核 ABI，而不是框架绑定。** `pocketllm.kernels` 只有描述符和声明，**零依赖** —— 没有
  numpy、没有 torch。后端实现它，引擎驱动它。正因如此，同一个 C 核可以原样坐落在 CUDA 卡、Ascend NPU
  和 CPU 之后。
- **阶梯就是适配策略。** 放不下的 checkpoint 会被一直量化到放得下。没有卸载路径可走，也没有第二块
  设备可伸手。
- **Torch 是可选的，且是后端自己的事。** 加载器、解码器和参考后端都是 numpy，所以手机端安装无需训练栈
  即可读取 `.gguf`。

## 路线图

- [x] **C 核能跑 Qwen3-0.6B 与 1.7B**：GGUF 读取、BPE 切词、稠密前向与贪心解码，在 `cpu` 和一张
      `cuda` 卡上与 llama.cpp 对齐。
- [x] **C 核内的量化权重**：`q4_k` 与 `q6_k` 在 kernel 内解码，保持打包、绝不展开。
- [x] **`pocketllm run` 与 `pocketllm serve`**：两者都经 `native.py` 进入 C 核。
- [x] **CPU 路径在 prefill 与 decode 上领先 llama.cpp**（匹配 KV 深度）。
- [ ] **Python 后端。** 先做 `cuda`；目前每个都只是声明。
- [ ] **`python/pocketllm/tokenizer/`** —— BPE 还是桩。C 核切词正确；Python 骨架落后于它。
- [ ] `architectures/xing4_0/`，按 ABI 重建，并补上 golden fixture。
- [ ] QNN 后端与 Android 交付路径 —— 整个仓库名字所指的目标。

## 许可

PocketLLM 以 [Apache License 2.0](LICENSE) 发布。

模型权重、分词器文件、CUDA、PyTorch、GGUF 资产及其他第三方组件受各自许可约束；PocketLLM 的代码
许可不授予这些第三方模型资产的额外权利。内置的 GGML 表头来自 llama.cpp，为 MIT 许可 ——
见[出处说明](python/pocketllm/loader/gguf/vendor/README.md)。