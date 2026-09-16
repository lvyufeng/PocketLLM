# DeepSeek-V4.1-Flash

## Runtime status

**Inspect only: the config parses and a 37-check safetensors-header audit is validated on the host; no V4.1 execution path exists in this repository.**

Nothing in this page generates tokens. The repository has no V4.1 code: `engram`, `kv_source_layers`, `candidate_source_layer` and `bias_vl` do not appear anywhere under `cpp_engine/`, `src/`, `pocketllm/`, `tests/` or `docs/` except in this page and the audit script it documents. `cpp_engine/engine/deepseek_v4_engine.cpp` and `src/models/deepseek_v4/` target the 43-layer, 4096-hidden DeepSeek-V4-Flash geometry.

What *was* validated is a fact list read out of the checkpoint's own metadata:

- `scripts/audit_dsv41_headers.py` parses the V4.1 config and runs 37 checks over the safetensors headers.
- The audit ran against the first 3,000,001 bytes of each of the 48 published shards — roughly 144 MB in total. **No weight payload was downloaded or read.**
- Upshot: 96,085 tensors, 510,286,023,000 B (475.24 GiB), a fully consistent tensor inventory, and a checkpoint-specific configuration that differs from the validated V4-Flash config in the ways listed below.

Generation, TPS and numerical parity are unmeasured and are not claimed. See [Known limitations](#known-limitations).

## Checkpoint/model specification

The model card (`deepseek-ai/DeepSeek-V4.1-Flash`, "Pushing the Limits of KV Cache Compression") describes a multimodal MoE with **552B backbone parameters** and contexts up to one million tokens, natively consuming images and text and generating text. Its architecture is a **Causal Encoder-Decoder (CED)**: a 40-layer Transformer organized as a **20-layer causal encoder followed by a 20-layer decoder**, where the decoder's global KV cache is projected from the final encoder hidden states rather than derived from each decoder layer's own hidden states. It reports **8B parameters per token during prefill** and **16B during decode**, and states two cache reductions: **SWA Bounded Replay** reconstructs missing sliding-window KV states by replaying only the most recent *n*_win tokens, cutting the persistent KV cache to roughly **1/8** of DeepSeek-V4-Flash's, and **FP4 main KV caching** (E2M1 format, one E4M3 scale per 16 channels) at **890 bytes per token**, roughly **1/4** of V4-Flash's. Attention is **CSA2**, which assigns each attention layer one of three static modes — **Full**, **Reindex** or **Reuse** — to share main KV and indexer K across layers and reuse Top-K sparse-attention indices, and in the decoder a **Hierarchical Sparse Indexer** restricts later indexing layers to a candidate pool built by the first Full Mode layer, bounding deeper indexer cost independently of context length. The remaining named components are **Single-Pass mHC** residual-stream mixing and **Engram conditional memory (196B parameters, sparsely accessed via token-based lookup)**. Multimodal support is a DeepSeek-ViT encoder (2D-RoPE, 3×3 pixel-unshuffle downsampling) plus a two-layer MLP projector. Pre-training used 45T tokens with sparse attention trained at 64K and context extended to 1M.

The geometry below is transcribed from the released `config.json` and cross-checked against the tensor shapes:

| Field | Value |
| --- | ---: |
| Backbone layers | 40 |
| MTP layers | 3 |
| Hidden size | 5120 |
| Attention heads | 64 |
| Head dimension | 512 |
| RoPE head dimension | 64 |
| Q LoRA rank | 1280 |
| Output groups × LoRA rank | 8 × 1024 |
| KV dimension (`wkv` output) | 512 |
| Sliding window | 128 |
| Routed experts | 384 |
| Active experts | top-6 |
| MTP active experts | top-3 |
| Shared experts | 1 |
| Expert intermediate size | 2304 |
| Route scale / score function | 1.5 / `sqrtsoftplus` |
| Hyper-connections | `hc_mult` 4, 20 Sinkhorn iterations |
| Vocabulary | 129,280 |
| Original sequence length | 65,536 |
| Norm epsilon | 1e-20 |
| Compressed-attention RoPE theta | 160,000 |
| Indexer heads × head dim | 32 × 128 |
| Index top-k | 512 |
| Candidate block size / top-k blocks | 8 / 2048 |
| Engram layers | `[1, 14]`, 8 heads, max n-gram 4 |
| Engram embedding rows (declared) | `[384006168, 384016682]` |
| Engram compressed vocabulary | 99,092 |
| Vision | 32 blocks, dim 1024, 16 heads, patch 14, 3×3 downsample |
| Checkpoint dtype / expert dtype | `fp8` / `fp4` |

### Configuration delta against DeepSeek-V4-Flash

`configs/config.json` is the validated V4-Flash config; the V4.1 config is the released `inference_config.json`. Same-named keys that changed:

| Key | V4-Flash | V4.1-Flash |
| --- | ---: | ---: |
| `n_layers` | 43 | 40 |
| `dim` | 4096 | 5120 |
| `q_lora_rank` | 1024 | 1280 |
| `moe_inter_dim` | 2048 | 2304 |
| `n_routed_experts` | 256 | 384 |
| `index_n_heads` | 64 | 32 |
| `compress_ratios` | alternating 4 / 128 | 0,0 then 18×2, 20×1, 0,0,0 |
| `n_hash_layers` | 3 | absent |
| `scale_fmt` | `ue8m0` | absent |

Unchanged between the two: `head_dim` 512, `n_heads` 64, `rope_head_dim` 64, `o_groups` 8, `o_lora_rank` 1024, `window_size` 128, `index_head_dim` 128, `index_topk` 512, `n_shared_experts` 1, `n_activated_experts` 6, `route_scale` 1.5, `score_func` `sqrtsoftplus`, `swiglu_limit` 10.0, `rope_factor` 16, `rope_theta` 10000, `compress_rope_theta` 160000, `original_seq_len` 65536, `vocab_size` 129280, `hc_mult` 4, `hc_sinkhorn_iters` 20.

V4.1 adds keys with no V4-Flash counterpart: `n_mtp_layers`, `dspark_block_size`, `dspark_markov_rank`, `dspark_n_routed_experts`, `dspark_n_activated_experts`, `dspark_noise_token_id`, `dspark_target_layer_ids`, `kv_source_layers`, `index_source_layers`, `candidate_source_layer`, `candidate_topk_blocks`, `candidate_block_size`, the `engram_*` block, the `vision_*` block, `hc_eps`, `norm_eps` and `image_token_id`.

`compress_ratios` changes meaning, not just value. In V4-Flash it alternates 4 and 128 and every layer with a non-zero ratio owns its own compressor. In V4.1 only four layers — `kv_source_layers = [2, 8, 14, 20]` — pool their own KV, and `compress_ratios[l] > 0` merely marks a layer whose attention *reads* compressed positions. That is why the V4.1 checkpoint carries `attn.compressor.*` on 4 layers where V4-Flash carries it on 41.

### Tensor inventory, verified from the shard headers

| Category | Tensors | Bytes | Share |
| --- | ---: | ---: | ---: |
| Routed experts | 94,464 | 275.67 GiB | 58.0% |
| Engram | 12 | 189.13 GiB | 39.8% |
| Attention | 603 | 4.80 GiB | 1.0% |
| Embedding and head | 3 | 2.47 GiB | 0.5% |
| Shared experts | 240 | 1.32 GiB | 0.3% |
| Vision and aligner | 266 | 0.90 GiB | 0.2% |
| MTP / DSpark | 97 | 0.66 GiB | 0.1% |
| Layer norms and hyper-connections | 400 | 0.29 GiB | 0.1% |
| **Total** | **96,085** | **475.24 GiB** | |

Shard layout, from the 48 published headers:

| Shards | Contents |
| --- | --- |
| `h00001` | 259 `vision.*` tensors and 4 `aligner.*` tensors, 0.90 GiB |
| `h00002` | `embed.weight` `[129280, 5120]`, plus `image_start`, `image_newline`, `image_end`, 1.23 GiB |
| `h00003`–`h00042` | one backbone layer each; 2,334 tensors and 6.88 GiB on a plain layer, 2,337 / 6.89 GiB on the four Reindex layers, 2,341 on layer 20 and 2,342 / 6.90 GiB on the three other Full layers |
| `h00043` | `head.weight` and `norm.weight`, 1.23 GiB |
| `h00044`–`h00046` | `mtp.0`, `mtp.1`, `mtp.2`, ~2.5 GiB each |
| `h00047`, `h00048` | Engram tables for layers 1 and 14, 94.56 GiB each |

Two Engram tables dominate the checkpoint. `layers.1.engram.embed.weight` is F8_E4M3 `[384006168, 256]` with an F8_E8M0 `[384006168, 8]` scale, 91.55 GiB; `layers.14` is `[384016682, 256]`, 91.56 GiB. Together, 768,022,850 rows. The two tables hold 196,613,849,600 embedding parameters (row count × 256), which is the card's "196B parameters, sparsely accessed via token-based lookup". Each layer also has `engram.wkv` F8_E4M3 `[25600, 6144]`, an F8_E8M0 `[800, 192]` scale, and BF16 `engram.q_weight` / `engram.k_weight` of shape `[4, 5120]`.

### Quantization, verified from the headers

- Every non-Engram FP8 weight uses a **32×32 block**: for example `attn.wq_a` F8_E4M3 `[1280, 5120]` with scale `[40, 160]`, `attn.wo_a` F8_E4M3 `[8192, 4096]` with scale `[256, 128]`.
- The Engram tables use a **per-row (1, 32)** block instead — `[384006168, 256]` against a `[384006168, 8]` scale.
- Routed experts are stored as `I8` holding two FP4 values per byte. `experts.N.w1.weight` is `I8 [2304, 2560]` with an F8_E8M0 `[2304, 160]` scale, i.e. FP4 with a block of 32 along K; `w2` is `I8 [5120, 1152]` with scale `[5120, 72]`. The reference `inference_convert.py` repacks these to E4M3 via its `cast_e2m1fn_to_e4m3fn` with `fp8_block_size = fp4_block_size = 32` and `MAX_OFFSET_BITS = 6`, because 6.0 × 2⁶ = 384 stays inside the E4M3 range.
- Shared experts are plain F8_E4M3 32×32 (`w1` `[2304, 5120]` scale `[72, 160]`), and `attn_sink`, `q_norm`, `kv_norm`, `attn_norm`, `ffn_norm` stay unquantized as F32/BF16.

## Implemented execution path

**There is no V4.1 execution path.** What exists is a host-only audit, plus a V4-Flash runtime that the layer-level comparison below suggests is a partial starting point.

### The audit script

`scripts/audit_dsv41_headers.py` is pure standard library — `argparse`, `json`, `os`, `re`, `struct`, `sys`, `collections` — so it runs under any interpreter with no `torch`, `safetensors` or `numpy`. It reads the 8-byte little-endian header length at the start of each shard, parses the header JSON, and validates the declared `dtype`/`shape`/`data_offsets` against the config. It never reads a payload, which is why it works identically on complete shards and on 3 MB header prefixes.

### What the V4-Flash runtime already provides

A backbone layer in V4.1 carries the same module set as a V4-Flash layer, in the same names: `attn.{wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink}`, `attn_norm`, `ffn_norm`, `ffn.gate`, `ffn.shared_experts.{w1,w2,w3}`, `ffn.experts.N.{w1,w2,w3}`, and the six `hc_attn_*` / `hc_ffn_*` hyper-connection tensors. Hyper-connections (`hc_mult` 4, 20 Sinkhorn iterations) and the DSpark MTP stack are already implemented for V4-Flash, and the 384-expert / top-6 / `sqrtsoftplus` MoE is a geometry change from 256 experts, not a new algorithm.

The deltas a V4.1 path would actually have to add:

| Delta | Evidence |
| --- | --- |
| Shared CSA2 compression | `attn.compressor.*` on 4 layers instead of 41, plus new `kv_source_layers` / `index_source_layers` / `Full`/`Reindex`/`Reuse` mode logic |
| Hierarchical sparse indexer | New `indexer.wk` + `indexer.k_norm` (absent in V4-Flash) and `candidate_source_layer` / `candidate_topk_blocks` / `candidate_block_size` |
| Engram | 189.13 GiB of n-gram hash tables, a 99,092-entry compressed vocabulary, and the `engram_compressed_vocab_size` assertion in the reference `inference_engram.py` |
| Vision | 259 `vision.*` tensors, a 4-tensor aligner, and `ffn.gate.bias_vl` — a vision-conditioned routing bias present on all 43 V4.1 layers and absent from V4-Flash |
| Hash-routing removal | V4-Flash has `layers.{0,1,2}.ffn.gate.tid2eid`; V4.1 has no `tid2eid` tensor and no `n_hash_layers` key |
| Hyper-connection head removal | V4-Flash has top-level `hc_head_{fn,base,scale}`; V4.1 has none, and the last MTP stage has none either |
| MTP head rename | V4-Flash `mtp.2.markov_head.{markov_w1,markov_w2}`; V4.1 `mtp.2.markov_head.{embed,head}` |
| Compressor APE removal | V4-Flash `attn.compressor.ape` on 41 layers; V4.1 has no `ape` tensor |

Two V4.1 MTP asymmetries are worth recording before any loader is written: only `mtp.0` has `main_norm` and `main_proj` (F8_E4M3 `[5120, 15360]`, with 15,360 = 3 MTP layers × 5120), and only `mtp.2` has `markov_head.embed`, `markov_head.head` (both BF16 `[129280, 256]`), `confidence_head.proj` (BF16 `[1, 5376]`, i.e. 5120 + `dspark_markov_rank`) and `norm`. `mtp.1` has no module beyond attention, FFN and hyper-connections.

The removed `hc_head_*` is not a cosmetic deletion, and the parameter shapes alone do not reveal what replaced it. In the V4-Flash runtime both `src/models/deepseek_v4/runtime.py` and `src/models/deepseek_v4/dspark.py` collapse the hyper-connection copies before the head with a dedicated `hc_head` projection. The V4.1 reference instead collapses with the `pre_mix` carried out of the **last block's FFN** — `h = layer.hc_pre(h, pre_mix)` before `self.norm(h)`, and `self.hc_pre(x, pre_mix)` inside the DSpark head — and passes `hc_eps` into `ParallelHead` without using it. So a V4.1 loader must carry `pre_mix` out of every block and use the final one, rather than looking for a head tensor that no longer exists. This is the card's **Single-Pass mHC**; the parameterization is otherwise unchanged (`hc_mult` 4, `mix_hc` = 24, `hc_dim` = 20480, `hc_sinkhorn_iters` 20), with `hc_eps` the one new key.

## Validated performance

**None.** No prefill or decode throughput, latency or memory figure has been measured for V4.1-Flash, and none is implied by the audit. The reasons are concrete:

- The checkpoint is 475.24 GiB and is not present on this host; only 48 header prefixes were fetched.
- The four RTX 2080 Ti cards in the reference host hold 22 GiB each, so even a TP4 shard of the routed-expert and Engram tensors does not fit.
- The released reference stack (`inference_requirements.txt`) requires `torch>=2.10.0` and `tilelang==0.1.8`. The `deepseek` environment has `torch 2.9.1+cu128` and no `tilelang`, so the reference implementation cannot be run here to produce a comparison point either.

The model card's "8B parameters per token during prefill / 16B during decode" and "890 bytes per token" KV figures are the vendor's numbers. This repository has not reproduced them and this page does not restate them as measurements.

## Correctness and precision

The audit's 37 checks currently pass on the real headers. They are grouped as follows:

- **Config (12 checks).** All 36 required keys present; `len(compress_ratios) == n_layers + n_mtp_layers`; MTP layers compress nothing (`compress_ratios[40:43] == 0`); `kv_source_layers ⊆ index_source_layers`; every source layer reads compressed positions; every index source has a KV source at or below it; `candidate_source_layer` is the first layer after `kv_source_layers[-1]`; `dspark_target_layer_ids` are the last `n_mtp_layers` backbone layers; one Engram table size per Engram layer; `engram_layer_ids` inside the backbone; `engram_compressed_vocab_size` set.
- **Packing and scales (4 checks).** Every tensor's byte extent matches its shape and dtype; every weight/scale pair blocks evenly; all non-Engram FP8 weights use a 32×32 block; the Engram tables use a 1×32 per-row block.
- **Inventory (4 checks).** Every backbone layer has its full tensor set; the known shapes match the config; the F32/BF16 tensors are not quantized; the tensor count and byte total are non-zero.
- **Experts (3 checks).** `w1`/`w2`/`w3` are FP4 packed into `I8` with FP4-block-32 E8M0 scales; every backbone layer has 384 routed experts; every MTP layer has 128.
- **CSA2 (5 checks).** `compressor.wkv` exactly on `kv_source_layers`; `compressor.wgate` exactly on the ratio>1 KV sources; `indexer.wk` exactly on `kv_source_layers`; `indexer.wq_b` exactly on `index_source_layers`; Full/Reindex/Reuse partition the backbone.
- **Engram (4 checks).** The tables sit on exactly `engram_layer_ids`, F8_E4M3 with E8M0 scales; each Engram layer has its gate and value projection; the tables are at least as large as the derived bucket ranges and no larger.
- **Vision (2 checks).** All 32 blocks present; the encoder and projector shapes match the config.
- **DSpark (3 checks).** The Markov and confidence heads match the config; `main_proj` is `n_mtp_layers * dim` wide; every MTP layer carries attention and FFN, but only the last carries the heads.

Two facts from this section deserve emphasis because they were derived rather than read off:

The Engram row counts are *derived*, and the derivation is exact to the row. The reference draws primes in order starting just above `engram_vocab_size` (16,000,000), hands them out across both layers without reuse, taking `(max_ngram_size − 1) × n_heads = 3 × 8 = 24` primes per layer, and `NgramHashState` builds its bucket offsets as a running sum over the layer's *flattened* prime list. The largest id a layer can produce is therefore `sum(primes) − 1` and the table needs exactly `sum(primes)` rows. Both layers match `engram_num_embeddings` with a difference of zero: the declared `[384006168, 384016682]` equals the derived values.

The CSA2 modes are derived from tensor presence, not from a config field. Diffing the three shard shapes against each other gives an exact per-mode tensor delta: a Reuse layer carries nothing extra; a Reindex layer adds `indexer.wq_b.weight`, `indexer.wq_b.scale` and `indexer.weights_proj.weight`; a Full layer adds those three plus `compressor.wkv.weight`, `compressor.norm.weight`, `compressor.wgate.weight`, `indexer.wk.weight` and `indexer.k_norm.weight`. So Full = owns `indexer.wk`, Reindex = owns `indexer.wq_b` only, Reuse = neither, which partitions the backbone as Full `[2, 8, 14, 20]`, Reindex `[24, 28, 32, 36]`, Reuse the other 32 layers.

`compressor.wgate` is the one tensor that separates layer 20 from the other three Full layers, and the reason is in the config value rather than in the tensor set: the reference constructs the gate only `if compress_ratio > 1`, because the gate is the learned softmax that pools `compress_ratio` consecutive tokens into one KV latent and a ratio of 1 has no group to pool. `compress_ratios` is `[0, 0]`, then 18 entries of 2, then 20 entries of 1, then `[0, 0, 0]` — so layer 20 is the single KV source at ratio 1. All four sources store `compressor.wkv` as BF16 `[512, 5120]` on disk; the reference declares that matrix FP32 for the ratio-2 layers and BF16 for ratio 1, so a loader reading the checkpoint has to upcast the three ratio-2 copies itself. (`compress_ratios[0:2]` and `compress_ratios[40:43]` are zero: layers 0 and 1, and all three MTP layers, read no compressed positions at all.)

Negative controls were run to confirm the checks are live rather than vacuous: perturbing a config key the audit does not consume leaves the result at 37/37 with exit 0, while setting `engram_n_heads = 7` drops it to 36/37 with exit 1, reporting `declared=[384006168, 384016682] derived=[336004849, 336012883]`.

The audit establishes nothing about numerics. No tensor value has been read, so no claim about quantization error, activation range or output parity is available.

## Reproduction

The audit needs only the first few megabytes of each shard. Downloading just enough bytes to cover the largest header in this checkpoint — 261,440 bytes — yields 48 files totalling about 144 MB:

```bash
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /path/to/dsv41-header-prefixes \
  --config /path/to/inference_config.json \
  --header-prefix
```

Against a complete checkpoint the same command runs without `--header-prefix`; results are identical because the payload is never read. `--json out.json` writes the report as machine-readable JSON, and `--list-tensors PATTERN` prints individual tensors as `name<TAB>dtype<TAB>shape<TAB>bytes<TAB>shard`. `--expect-fp8-block 32 32` overrides the FP8 block size the scale check assumes.

Getting the prefixes does not need a checkpoint download. Fetch the first 3,000,001 bytes of each of the 48 published shards — a byte range covers the largest header in this checkpoint, 261,440 bytes, many times over — and write them to any 48 local files:

```bash
# One range request per shard. The URL layout is mirror-specific; only the first
# 3,000,001 bytes of each shard matter, and the local name is free-form.
curl -r 0-3000000 -o h00001.bin "<shard-1-url>"
# ... repeated for h00002.bin .. h00048.bin
```

Shard files are detected either by a `.safetensors` suffix or by their first 8 bytes looking like a plausible header length, so the local names above are arbitrary. A shard whose header is byte-identical to another's is skipped as a duplicate — worth knowing because a mirror that serves the same file under two names would otherwise be counted twice. The audit's own inventory (`96,085 tensors`, `475.24 GiB`) is the check that all 48 shards were actually distinct.

To confirm the checks are live, make a copy of the config with `engram_n_heads` changed to 7 and expect 36/37 with exit 1.

## Known limitations

- **No generation of any kind.** No tokenizer is exercised, no embedding is loaded, and no forward pass exists for this architecture.
- **No local checkpoint.** Only 48 header prefixes were downloaded; the 475.24 GiB of weights is not on this host, and it does not fit on the four 22 GiB cards available here.
- **The audit validates metadata consistency, not correctness.** It proves the config and the tensor inventory agree with each other. A checkpoint could satisfy all 37 checks and still be unusable, and a wrong value that is *consistently* wrong in both the config and the shapes would pass.
- **Engram cannot be exercised.** `inference_engram.py` asserts `vocab_size == args.engram_compressed_vocab_size` as a binary gate, and the 99,092-entry compressed token map is built from the n-gram tokenizer rather than from the checkpoint. Until that map exists in this repository, the 189.13 GiB of Engram tables have no consumer.
- **The vision path is unvalidated in both directions.** The 263 vision and aligner tensors are accounted for and their shapes match the config, but no image has been processed and no projector has been run.
- **No reference comparison is possible on this host.** The released stack needs `torch>=2.10.0` and `tilelang==0.1.8`; neither is available in the `deepseek` environment.
- **The prompt format changed and is unimplemented here.** DSML tags gain a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`), reasoning effort becomes a numeric budget in 1–100 rendered only under `thinking_mode="thinking"` at index 0, and mid-conversation `<｜System｜>` messages are supported. None of that is wired into this repository's chat templates.
- **Memory is the structural problem, not the kernels.** 58.0% of the checkpoint is routed experts and 39.8% is two Engram tables. A 4×22 GiB TP4 deployment cannot hold either, so any V4.1 plan has to place experts and Engram in host memory or on disk before kernel work matters.

## Evidence and related notes

- `scripts/audit_dsv41_headers.py` — the host-only header audit and config parser
- `configs/config.json` — the validated V4-Flash config used as the delta baseline
- [DeepSeek-V4-Flash](deepseek-v4.md) — the validated runtime whose layer structure the V4.1 backbone reuses
- [Benchmark reporting rules](../guides/benchmarking.md) — required before any V4.1 number is quoted
- Model card and released config: `https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash`
