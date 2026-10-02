# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[English](README.md) | **中文**

在**单一加速器**上运行大语言模型 —— 一张显卡、一块边缘板卡，或者口袋里的手机。一个进程独占一个设备；
如果模型放不下，就继续量化，而不是拆到第二张卡上。

> **状态：C 引擎能跑 Qwen3-0.6B（f16 或 `q4_k_m`），Python 包还不能跑模型。** `src/`
> （`libpocketllm.so`）读 GGUF、用 checkpoint 自带的分词器切词、走完 Qwen3 图并贪心解码 —— 权重可以是
> `f32`/`f16`，也可以是打包的 `q4_k`/`q6_k`（在 kernel 内解码，绝不展开成 f32）。1.4 GB 的 f16 与由它
> 量化出的 456 MB `q4_k_m`，都已在 `cpu` 和一张 `cuda` 卡上与 llama.cpp 逐 token 对齐。Python 包是
> **主机侧** —— 内核 ABI 作为规范、numpy 参考实现、GGUF 加载器、量化解码器、执行层，以及 OpenAI 兼容的
> HTTP 服务面 —— 它**没有任何后端实现了 kernel**：除 `reference` 外每个后端都只是一份声明，其 session 会抛
> `BackendNotImplementedError`。
>
> 这两半在一个地方接上了：`python/pocketllm/native.py`，也就是 `ctypes` 桥，现在由 `pocketllm run` 驱动
> —— `pocketllm run --model ckpt.gguf --prompt "…"` 通过 C 核贪心生成。`pocketllm serve` 还没有接上，
> 任何 *Python* 后端也没有。安装前请先读 [English README 的 Current status](README.md#current-status)。

## 规则

**放不下就量化 —— 不卸载到主机，也不跨卡切分。**

位宽阶梯是 Q4 → Q2 → IQ2 → IQ1 → 三值，按此顺序下降，停在模型仍能正确作答的最低格式。主机卸载与
多卡并行在结构上就被排除：ABI 里没有 rank、没有集合通信、没有第二个设备，`EngineArgs` 里也没有
`tensor_parallel_size` 可设。

## 安装

基础安装**只有一个依赖**。ABI 只依赖标准库，参考后端依赖 numpy，设备运行时是可选 extra ——
这正是同一个 wheel 既能装到 CUDA 机器上、也能装到手机上的原因。

```bash
pip install pocketllm

# 带设备运行时
pip install "pocketllm[cuda]"      # NVIDIA，经由 torch
pip install "pocketllm[mps]"       # Apple Silicon，经由 torch
```

**要求：** Python >= 3.10。基础安装不需要编译器、不需要 CUDA toolkit、也不需要 `relic-core`。
本树没有 `ext_modules`，没有构建步骤：所有原生内核都归属另一个仓库。

## 试试看

只读命令在任意主机上都能跑，包括没有任何加速器、也没装 torch 的机器：

```bash
pocketllm devices        # 本机可以打开哪些后端，其余各自缺什么
pocketllm backends       # 每个后端声明了什么，无论本机能否打开
pocketllm architectures  # 本树能够构建的模型结构
pocketllm ops --op gemm_quant --device cuda
```

`devices` 是出问题时才会去跑的命令，所以它也是唯一一条不能依赖任何东西已安装的命令：它背后每一步
检查都是文件系统探测，不导入任何运行时。`--backend` 与 `--device` 的候选项直接来自后端注册表，
因此第三方后端新增的设备类型无需改核心代码即可出现在 `--help` 里。

需要加载 checkpoint 的两个命令目前还没有可用的后端：

```bash
pocketllm run   --model /path/to/model.gguf --prompt "hello"
pocketllm serve --model /path/to/model.gguf
```

两者都会先完整校验参数，再带着「具体缺什么」退出，而不是假装开始服务。

## 代码现在在哪里

本仓库过去承载整套多卡栈，如今按硬件切分，PocketLLM 是其中一端：

| 仓库 | 是什么 |
|---|---|
| **PocketLLM**（本树） | 单卡 / 边缘 / 移动。一个进程独占一个设备 |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | 多卡 PyTorch 运行时与服务外壳 |
| [relic-core](https://github.com/lvyufeng/relic-core) | 共享的 torch 算子库（CUDA sm_75 + CPU host ops） |
| [relic-engine](https://github.com/lvyufeng/relic-engine) | 已退役的 `cpp_engine` 树，冻结归档 |

**`relic-core` 不是本包的依赖。** 本树过去从它读取的唯一内容 —— GGML 码本表头 —— 现在
[已随树内置](python/pocketllm/loader/gguf/vendor/README.md)并附出处说明：一个离开内核库就读不了 checkpoint
的加载器，就是一个上不了手机的加载器。CUDA 后端可以把 `relic_core` 作为可选 extra 包起来，
但核心代码不导入它。

## 路线图

- [x] 切出多卡代码交给 RelicLLM，并在无 torch 的 ABI 上重建。
- [x] 移植并去 torch 化 GGUF 加载器与量化解码器。
- [x] 移植服务层与协议层。
- [ ] **第一个真实后端。** 自然是 CUDA —— 硬件在手，内核也已在 relic-core 里。
- [ ] `architectures/xing4_0/`，按 ABI 重建，并补上 golden fixture。
- [ ] GGUF 词表的 BPE 分词器。
- [ ] QNN 后端与 Android 交付路径 —— 整个仓库名字所指的目标。

## 许可

PocketLLM 以 [Apache License 2.0](LICENSE) 发布。

模型权重、分词器文件、CUDA、PyTorch、GGUF 资产及其他第三方组件受各自许可约束；PocketLLM 的代码
许可不授予这些第三方模型资产的额外权利。内置的 GGML 表头来自 llama.cpp，为 MIT 许可 ——
见[出处说明](python/pocketllm/loader/gguf/vendor/README.md)。