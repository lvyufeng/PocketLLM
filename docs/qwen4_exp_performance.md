# Qwen4-Exp 异构 TP4 性能分析

**日期**：2026-08-28
**硬件**：4×RTX 2080 Ti（TP4，PCIe Gen3）
**模型**：Qwen3.8-Flash-Next，真实 checkpoint 位于
`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`

## 模型和测量方法

真实配置为 48 层、hidden size 2560、512 个 routed experts、top-10，单个
BF16 expert 为 9.38 MiB，全 expert 集合约 225 GiB。routed experts 和 PLE
表通过 mmap 保留在 host，active experts 按需复制到各 GPU。

运行时通过 `--profile` 开启层级 profiler；加入
`DSV4_QWEN4_MOE_PHASE_PROFILE=1` 后，额外统计 expert staging 和计算时间。
完整的 staging 微基准可以运行：

```bash
PYTHONPATH=. python tests/bench_qwen4_exp_staging.py
```

## 实测基线

以下是 clean run（不启用 profiler）的结果：

| 场景 | chunk | expert cache | prefill | decode |
|---|---:|---:|---:|---:|
| 60-token prompt，32-token generation | 512 | 64 | 6.78 tok/s | 1.05 tok/s |
| 60-token prompt，32-token generation | 512 | 1024 | 6.73 tok/s | **2.75 tok/s** |
| 570-token prompt，1-token generation | 512 | 64 | 32.6 tok/s | — |
| 4162-token prompt，1-token generation | 512 | 64 | 38.1 tok/s | — |
| 4162-token prompt，1-token generation | 4096 | 64 | **151.4 tok/s** | — |
| 8272-token prompt，1-token generation | 8192 | 64 | **244.1 tok/s** | — |

因此，早先的 0.42/0.95 tok/s 组合不应作为模型能力基线：它对应很短
生成、chunk=512 和较小 expert cache。长 prefill 的 chunk 大小和 decode
的 cache 容量必须一起记录。

## 瓶颈结论

### Prefill

60-token profile 的 48 层累计时间中，MoE 占 68.8%，NCCL MLP reduce 占
10.4%，attention 占 9.6%。MoE 的分相测量为：

```text
stage 5.091 s   compute 0.601 s   1874 expert calls / 48 MoE calls
17.16 GiB H2D，cache hit 0%
```

在 8K prefill 中，单 rank 约搬运 48 GiB；4 个 rank 聚合有效吞吐约
5.5 GiB/s，与单进程纯 RAM→GPU 微基准 5.4–5.7 GiB/s 一致。因此主要地板
是共享 PCIe 带宽，而不是 checkpoint 所在机械盘。cold/warm page-cache
测量只有 1.16–1.37× 差异。

### Decode

32-token decode、cache=1024 的层内累计时间为：MoE 46.6%、attention
18.6%、MLP NCCL reduce 16.2%、attention NCCL reduce 3.8%。层外约 18%
花在每一步对 248320 词表做 `all_gather`，以及随后由 `.item()` 引入的
同步。

cache=64→1024 将 decode 从 1.05 提到 2.75 tok/s；cache=2048 在当前
BF16 staging 下 OOM。输出在两种 cache 容量下逐字一致。

## 已验证和已否定的方向

1. **增大 prefill chunk 是当前最大的免费收益**：8K prompt 上
   chunk=512→8192 达到约 4×（38.1→244.1 tok/s），因为 active expert
   集合很快接近每 rank 的 128 个上限，搬运成本被更多 token 摊薄。
2. **增大 decode cache 有效**：容量 1024 能保留跨步复用的
   `(layer, expert)`，但需要 O(1) LRU 实现，不能继续使用
   `list.pop(0)`。
3. **合并 active expert H2D 没有收益**：108 expert 的实测为逐 expert
   loop 155 ms、host gather 后的 batched 方案 215 ms（0.72×）；host
   `index_select` 本身约 76 ms。不要仅凭减少 DMA call 数就合并 staging。
4. **TP 分片没有重复搬运**：`ShardedMoE` 的每个 rank 只 fetch 自己的
   expert，实测 duplication=1.00×。
5. **prefetch 不是带宽优化**：PCIe 已饱和时，cross-layer prefetch 最多
   改变时序，不能降低搬运字节数；应在量化后再重新 A/B。

## 朝 500 / 10 的工作分解

当前最佳实测点是长 prefill 244 tok/s、短 decode 2.75 tok/s。

### P0：默认值和低风险运行时修正

- 默认 chunk 调到 4096–8192，并依据可用显存夹取；
- expert cache 默认 1024；把 LRU 替换为 `OrderedDict` 或 ring；
- decode 不再 all-gather 全词表，只在各 rank 做 local argmax，然后
  all-gather `(value, index)`。

### P1：消除 PCIe 地板

- 对 routed expert 使用 FP8/INT4 storage，并在 GPU 侧使用对应的
  dequant/GEMM kernel；
- FP8 将 active bytes 约减半，长 prefill 的带宽上限接近 500 tok/s；
- decode 若要从约 2.75 达到 10 tok/s，预计需要 INT4 或等效的 4× bytes
  reduction，再叠加 local-vocab reduction 和 batched decode kernel；
- 所有量化路线必须先做真实权重 logit parity，再做性能 A/B。

### P2：计算和通信收尾

- prefill 中 MoE compute 只占 staging 的约 1/8.5，优先级低于量化；
  staging 地板松动后再做 grouped/batched expert GEMM；
- 尝试将两次每层 NCCL reduce 与下一段计算 overlap；
- 重新评估 cross-layer prefetch，不能把它当作增加 PCIe 带宽的方案。

在当前 PCIe Gen3 机器上，**prefill 500 的必要条件是 expert 量化**；
仅优化 Python dispatch 或 attention 不可能越过 H2D 下限。**decode 10**
则需要量化、cache、词表归约和 decode 专用 MoE kernel 的组合。
