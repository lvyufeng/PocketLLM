# Qwen4-Exp 异构 TP4 性能分析

**日期**：2026-08-28
**硬件**：4×RTX 2080 Ti（TP4，PCIe Gen3）
**模型**：Qwen3.8-Flash-Next，真实 checkpoint 位于
`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`

## 模型和测量方法

真实配置为 48 层、hidden size 2560、512 个 routed experts、top-10，单个
BF16 expert 为 9.38 MiB，全 expert 集合约 225 GiB。routed experts 在启动时
按 rank 装入 host 内存（每 rank 56.25 GiB，4 份互不相交），PLE 表继续通过
mmap 留在 host，active experts 按需复制到各 GPU。运行时默认启用该预加载；
`--mmap-experts` 仅用于诊断旧的按需 fault 路径。市面上说的“host resident”
在本文特指这份显式拷贝，不把 page cache 命中当作实现保证。

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

### host 常驻 vs 按需 mmap 的 A/B（2026-08-29，同机连续两跑）

60-token prompt、8-token generation、chunk 512、expert cache 1024，TP4：

| 模式 | 加载 | prefill | decode | 每 rank H2D 速率 | 生成期磁盘读 |
|---|---:|---:|---:|---:|---:|
| `resident`（默认） | 732 s 一次性 | **18.96 tok/s** | 1.97 tok/s | 2.741 GiB/s | 0.01 GiB |
| `--mmap-experts` | 1.3 s | 9.16 tok/s | 2.38 tok/s | 2.006 GiB/s | 0.00 GiB |

读法（不要过度解读）：

- prefill 2.07×、H2D 速率 1.37×，来自 pinned host 内存的 DMA 而不是 pageable
  mmap 页；两跑 staging 字节数完全相同（19.89 GiB，2173 miss / 567 hit），
  所以差异纯粹是搬运路径；
- decode 一列两跑都只有 8 个 token，差值 0.4 tok/s 在这个长度上不可引用
  （见 memory `perf_claims_need_tg_length`）；要结论必须跑 ≥32 token；
- **mmap 臂的磁盘读也是 0.00 GiB**：它紧跟在 resident 跑之后，expert 页还在
  page cache 里。这个 A/B 因此只测出了"pinned 拷贝 vs 暖 page cache"，
  没测出"vs 冷盘"。冷盘对比需要 drop_caches（需要 root），当前未做；
- 732 s 的加载在第一次冷读时是 0.10–0.12 GiB/s 的盘瓶颈；暖 page cache 下
  同样的预加载只要约 4 s（实测 1.19–1.24 GiB/s，4 层 4.69 GiB）。

## 瓶颈结论

### Prefill

60-token profile 的 48 层累计时间中，MoE 占 68.8%，NCCL MLP reduce 占
10.4%，attention 占 9.6%。MoE 的分相测量为：

```text
stage 5.091 s   compute 0.601 s   1874 expert calls / 48 MoE calls
17.16 GiB H2D，cache hit 0%
```

在 8K prefill 中，单 rank 约搬运 48 GiB，实测有效吞吐约 1.39 GiB/s/rank。

**修正（2026-08-29）**：此前的"主要地板是共享 PCIe 带宽"结论已被证伪。
原推理把 4 rank 的**聚合**吞吐（5.55 GiB/s）与单进程**单链路**微基准
（5.4–5.7 GiB/s）相比，属于量级错配。重测的 4 进程聚合吞吐为：

```text
mmap  20.86 GiB/s    ram  21.37 GiB/s    pinned  42.54 GiB/s
```

即真实 run 的 1.39 GiB/s/rank 比同一代码路径孤立跑低 3.6×，PCIe 远未饱和。

真正的地板是 checkpoint 所在的机械盘 `sdc`（ST4000VX015）：

```text
裸 O_DIRECT 顺序读        0.109 GiB/s（1/2/4/8 线程均为 0.112–0.125，并行无效）
未触碰层（30/31）冷读     0.06  GiB/s
同层暖 page cache         1.09–1.14 GiB/s   → 冷/暖差 18.8×
暖 mmap → GPU             5.2–6.5 GiB/s
```

此前"cold/warm 只差 1.16–1.37×"的测量无效：它的冷臂重读了已经 fault
过的第 17 层，两臂实际都命中 page cache。

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
5. **grouped expert GEMM 有效**：`_dispatch_experts` 的 per-expert Python
   循环换成 grouped `bmm`，decode 形状 2.41×/2.90×，prefill 形状
   4.45×/5.03×，max_abs_diff 0–3.9e-03。
6. **NUMA pinning 影响很大**：pinned-bounce 路径在 node0 与 node1 之间
   差 3.3×（9.66 vs 2.90 GiB/s）。GPU0/1 挂 node0，GPU2/3 挂 node1
   （两者 NV2 互连，但对 host 都是 SYS）。
7. **vocab reduce 可省**：full-vocab all_gather 换成 local argmax +
   `(value, index)` all_gather，3.112 → 2.497 ms/step，argmax 逐位一致。

## 朝 500 / 10 的工作分解

当前最佳实测点是长 prefill 244 tok/s、短 decode 2.75 tok/s。

### P0：默认值和低风险运行时修正

- 默认 chunk 调到 4096–8192，并依据可用显存夹取；
- expert cache 默认 1024；把 LRU 替换为 `OrderedDict` 或 ring；
- decode 不再 all-gather 全词表，只在各 rank 做 local argmax，然后
  all-gather `(value, index)`。

### P1：routed expert 常驻 host 内存（已实现）

目标是消掉机械盘，而不是换一块更快的盘。本机 MemFree 约 962 GiB，BF16
expert 全集 225 GiB，完全装得下；暖 mmap→GPU 5.2–6.5 GiB/s 也高于本机
NVMe 的 2.2 GiB/s，所以迁 NVMe 严格劣于常驻 host 内存。

实现形态（`weights.HostExpertShard` + `Qwen4ExpCheckpoint.preload_experts`）：
启动时每个 rank 把**自己那份** routed expert 从 mmap 拷进 host 内存，之后
`expert_rows()` 只读这份常驻拷贝，稳态 staging 不再有机会 fault 到盘上。

- 无需 POSIX-shm 第二份拷贝：ownership 是 `expert_id % world_size == rank`
  （`ShardedMoE` 的既有规则），4 个 shard 互不相交、加起来正好是 225 GiB
  的一份全量，每 rank 56.25 GiB。`SharedCPUMoEWeightArena` 只作布局参考，
  它的 `build_specs()` 是写死 int8 的，存不了 BF16；
- pin 是可行的：`ulimit -l` ≈ 126 GiB 是**每进程**上限，56.25 GiB 在限内。
  实测每层 1.172 GiB/rank pin 成功。pin 失败时退回 pageable 并把已装载的
  层一起转成 pageable，不留下半 pin 状态（`--pageable-experts` 可强制）；
- 每层两个连续 host tensor（`[128, 1280, 2560]` / `[128, 2560, 640]`），
  48 层共 96 次分配，而不是 12,288 次 per-expert 分配；
- 装载按 layer 推进并在 rank 间设 barrier：4 个进程读同一批文件区域，让
  盘头留在一个区域内，而不是各自在 225 GiB 上乱跳；
- 已知陷阱（都已避开）：不要对整个 mmap 做 `cudaHostRegister`（会造出巨量
  Dirty 页并拖慢无关 I/O）；routed expert 的**计算**仍在 GPU，没有走 GLM
  上已证伪的 CPU 侧 MoE；
- 首次装载仍要过一次机械盘，这是一次性成本：实测 0.102 GiB/s，
  56.25 GiB/rank ≈ 9.2 min，属于加载期而不是稳态延迟，报性能时必须分开算。

### P2：dispatch、NUMA、词表

- 用 grouped `bmm` 替换 `_dispatch_experts` 的 per-expert 循环（已测
  2.41–5.03×），并把 `HostExpertMoE` 的 `list.pop(0)` LRU 换成 O(1)；
- 按 rank 做 CPU/NUMA 亲和，复用
  `_cpu_affinity_for_rank`（`src/models/deepseek_v4/generation.py:67`），
  让 GPU0/1 的 staging 落在 node0、GPU2/3 落在 node1；
- decode 不再 all-gather 全词表，只做 local argmax + `(value, index)`
  all_gather（3.112 → 2.497 ms/step）。

### P3：量化和通信收尾

- host 常驻之后再评估 FP8/INT4 storage：此时它的作用是压 PCIe bytes 和
  放宽 pin 上限，而不是绕开盘；
- 尝试将两次每层 NCCL reduce 与下一段计算 overlap；
- 重新评估 cross-layer prefetch，只有在 staging 不再是盘 I/O 之后才有
  意义。

结论：**prefill 和 decode 的第一必要条件都是 routed expert 常驻 host
内存并在 4 rank 间共享单份拷贝**。在仍然向机械盘要页的前提下，优化
Python dispatch、attention 或量化都会被 0.06–0.11 GiB/s 的冷读吃掉。
