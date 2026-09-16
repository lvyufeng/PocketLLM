# DeepSeek-V4.1-Flash

## Runtime status

**Inspect only: both config layouts the release ships are read into one schema, a 38-check safetensors-header audit — 39 against the release's own index — runs on the host and distinguishes a shard that has not downloaded from a shard that is wrong, and the Engram hash front end that addresses the two 189 GiB tables is reproduced and tested; no V4.1 execution path exists in this repository.**

Nothing in this page generates tokens. The repository has no V4.1 layer code: `kv_source_layers` and `candidate_source_layer` appear only as config key names — in the schema and the audit and the tests this page documents, and nowhere a forward pass reads them. `ffn.gate.bias_vl` is the same kind of name: it is a real F32 `[384]` tensor on every backbone layer of the checkpoint, read out of the shards, but nothing here consumes it. `engram` does appear — `src/encoding/engram.py` reproduces the reference's tokenizer-side front end — but that is address arithmetic over a tokenizer, not a layer of the model. `cpp_engine/engine/deepseek_v4_engine.cpp` and `src/models/deepseek_v4/` target the 43-layer, 4096-hidden DeepSeek-V4-Flash geometry.

What *was* validated is a set of facts read out of the checkpoint's own metadata, plus an Engram front end that needs only the config and a tokenizer:

- `src/models/deepseek_v4_1/config.py` reads either config layout into one schema, so the two files that describe this model are checked against each other rather than parsed ad hoc by each consumer.
- `scripts/audit_dsv41_headers.py` reads the config through that schema and runs 38 checks over the safetensors headers — 39 when the checkpoint's own `model.safetensors.index.json` is present, which is what the release ships.
- The audit has run twice: against the first 3,000,001 bytes of each of the 48 published shards (roughly 144 MB in total), where 38 of 38 checks pass, and against the real checkpoint as it downloads, where 20 of 48 shards give 31 passes, 8 undecided and no failures. **No weight payload was downloaded or read** in either run.
- `src/encoding/engram.py` re-derives the Engram bucket layout and hashes token n-grams onto it. The primes it draws add up to the declared `engram_num_embeddings` exactly, so the 189.13 GiB of Engram tables are addressable rather than merely counted.
- Upshot: 96,085 tensors, 510,286,023,000 B (475.24 GiB), a fully consistent tensor inventory, an Engram layout that closes to the row, and a checkpoint-specific configuration that differs from the validated V4-Flash config in the ways listed below. The 8 undecided checks are shape claims waiting on the 28 shards that had not landed — undecided is not a pass, and `--require-complete` turns it into a failure.

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

`configs/config.json` is the validated V4-Flash config; the V4.1 config is the released `inference/config.json`, whose flat key names the table below uses. Same-named keys that changed:

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
| `scale_fmt` | `ue8m0` | `ue8m0`, under `config.json`'s `quantization_config` only |

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

The counts in that table are per *shard*, which is not the same as per *layer*: layer 1's six Engram tensors are shipped in `h00047` while layer 1 itself is `h00004`, so `h00004` holds 2,334 tensors and the layer has 2,340. Counting the index by layer instead gives an exact partition — 2,334 for a Reuse layer, 2,337 for a Reindex layer, 2,342 for a Full layer, 2,340 for layer 1 (+6, Engram), 2,341 for layer 20 (the Full set minus `compressor.wgate`, since `compress_ratios[20] == 1`) and 2,348 for layer 14 (Full + Engram) — with every one of the 40 layers a strict superset of the plain set. The landed shards match the index's assignment shard for shard. [Auditing the shards while the checkpoint arrives](../performance/deepseek_v4_1_shard_audit.md) records that partition in full, together with the tensor shapes read out of the shards that have downloaded; no per-layer count in this paragraph is taken from a model card.

Two Engram tables dominate the checkpoint. `layers.1.engram.embed.weight` is F8_E4M3 `[384006168, 256]` with an F8_E8M0 `[384006168, 8]` scale, 91.55 GiB; `layers.14` is `[384016682, 256]`, 91.56 GiB. Together, 768,022,850 rows. The two tables hold 196,613,849,600 embedding parameters (row count × 256), which is the card's "196B parameters, sparsely accessed via token-based lookup". Each layer also has `engram.wkv` F8_E4M3 `[25600, 6144]`, an F8_E8M0 `[800, 192]` scale, and BF16 `engram.q_weight` / `engram.k_weight` of shape `[4, 5120]`.

### Quantization, verified from the headers

- Every non-Engram FP8 weight uses a **32×32 block**: for example `attn.wq_a` F8_E4M3 `[1280, 5120]` with scale `[40, 160]`, `attn.wo_a` F8_E4M3 `[8192, 4096]` with scale `[256, 128]`.
- The Engram tables use a **per-row (1, 32)** block instead — `[384006168, 256]` against a `[384006168, 8]` scale.
- Routed experts are stored as `I8` holding two FP4 values per byte. `experts.N.w1.weight` is `I8 [2304, 2560]` with an F8_E8M0 `[2304, 160]` scale, i.e. FP4 with a block of 32 along K; `w2` is `I8 [5120, 1152]` with scale `[5120, 72]`. The reference `inference_convert.py` repacks these to E4M3 via its `cast_e2m1fn_to_e4m3fn` with `fp8_block_size = fp4_block_size = 32` and `MAX_OFFSET_BITS = 6`, because 6.0 × 2⁶ = 384 stays inside the E4M3 range.
- Shared experts are plain F8_E4M3 32×32 (`w1` `[2304, 5120]` scale `[72, 160]`), and `attn_sink`, `q_norm`, `kv_norm`, `attn_norm`, `ffn_norm` stay unquantized as F32/BF16.

## Implemented execution path

**There is no V4.1 execution path.** What exists is a config schema, a host-only audit, an Engram front end that needs no checkpoint at all, and a V4-Flash runtime that the layer-level comparison below suggests is a partial starting point.

### The config schema

A V4.1 checkpoint describes the same model twice, in two layouts that are each silent about something the other states, and until this schema existed every reader parsed whichever one it happened to be handed. `src/models/deepseek_v4_1/config.py` reads both into one `V41Config`:

| | `config.json` | `inference/config.json` |
| --- | --- | --- |
| Layout | 11 top-level keys; the text half under `text_config`, the vision tower under `vision_config`, the weight quantization under `quantization_config` | one flat level of 64 keys |
| Backbone names | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `sliding_window`, `scoring_func` | `dim`, `n_layers`, `n_heads`, `window_size`, `score_func` |
| MoE / norm names | `moe_intermediate_size`, `num_experts_per_tok`, `routed_scaling_factor`, `rms_norm_eps` | `moe_inter_dim`, `n_activated_experts`, `route_scale`, `norm_eps` |
| Layer lists | `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id` | `kv_source_layers`, `index_source_layers`, `candidate_source_layer` |
| YaRN | `rope_scaling.{type, factor, original_max_position_embeddings}` | `rope_factor` / `rope_theta` / `original_seq_len`, flat |
| Vision | `vision_config.{num_hidden_layers, hidden_size, …}` | `vision_n_layers`, `vision_dim`, … |
| Top-level `dtype` means | the storage dtype of the unquantized tensors, `bfloat16` | the quantization dtype, `fp8` |

The `dtype` row is the one that bites hardest: the two files use the same key name for different facts, and the Transformers file's value is *not* the quantization format — the dense linears are FP8 and the routed experts FP4, which that file states under `quantization_config` (`quant_method` `fp8`, `expert_dtype` `fp4`, `weight_block_size` `[32, 32]`, `scale_fmt` `ue8m0`) and the flat file states as `dtype`/`expert_dtype`. Reading `dtype` as `bfloat16` and concluding the checkpoint is unquantized is exactly the mistake the schema exists to prevent. `param_dtype` is the canonical name for the Transformers file's storage dtype, and the flat file has no counterpart for it — as it has none for 11 other fields, `model_type`, `architectures`, the three token ids, `hidden_act`, `max_position_embeddings`, `num_key_value_heads`, `tie_word_embeddings`, `topk_method` and `norm_topk_prob`.

What makes the two files checkable against each other is `as_reference_dict()`: it maps a parsed config back into the flat layout, and on the released pair it reproduces `inference/config.json` **exactly** — same 64 keys, same values — from either input file, so the nested file round-trips to the flat one and the flat one round-trips to itself. `differs_from()` then reports every field on which two configs disagree *or* one is silent, and on the released pair it returns exactly the 12 fields above and nothing else. That is the schema's real claim: the checkpoint's two descriptions of itself do not contradict each other anywhere, and the only asymmetries are omissions in the flat file rather than different values.

`verify()` checks a config against itself and is run by the module's command line:

```bash
python -m src.models.deepseek_v4_1.config /path/to/DeepSeek-V4.1-Flash
```

which prints the file it resolved, the shape it read, and the verdict:

```
/path/to/DeepSeek-V4.1-Flash/config.json (nested)
  40 layers + 3 draft, dim 5120, 64 heads of 512, 384 experts (6 active); Engram on layers [1, 14], vision 32 layers, DSpark block 5 over layers [37, 38, 39]
  [ok] the config is self-consistent
```

Passing a directory resolves the config the same way every consumer does, and printing `(nested)` or `(flat)` says which file was read. The checks are the relationships a reader would otherwise have to hold in their head: `compress_ratios` is as long as the backbone plus the MTP layers and is not all zero; every layer that reads compressed positions has a KV source at or before it *with its own ratio* (the source writes the cache and divides the position by its ratio, so a mismatch reads a cache written at a different granularity) and an index source likewise; `candidate_source_layer` is an index source; the Engram tables are on distinct ascending layers inside the backbone with one row count each; `engram_pad_id` is the tokenizer's pad; every embedding id is inside `vocab_size`; and the vision tower's `dim` divides by its `n_heads` with an even `patch_size` (it unshuffles 2×2). The 12 Transformers-only fields are required only when the nested file was read — the flat one is a reference-runtime artifact that never had them, so demanding them there would report the release's own choice as a defect. `--json` writes the normalized config and the problem list, and the exit status is non-zero only when `verify()` finds something.

### The audit script

`scripts/audit_dsv41_headers.py` is standard library only — its own `argparse`, `bisect`, `collections`, `collections.abc`, `json`, `os`, `re`, `struct` and `sys`, plus the config schema above, which is likewise standard library only — so it needs no `torch`, `safetensors` or `numpy` and runs under any interpreter as long as the repository root is importable. It reads the 8-byte little-endian header length at the start of each shard, parses the header JSON, and validates the declared `dtype`/`shape`/`data_offsets` against the config. It never reads a payload, which is why it works identically on complete shards and on 3 MB header prefixes.

Presence and shape are answered from two different places, and the audit keeps them apart. The release ships `model.safetensors.index.json`, which names the shard holding each of the 96,085 tensors, so *is this tensor in the checkpoint* is decidable before a byte of payload arrives; shapes, dtypes and byte extents come from the headers and are decidable only per shard. A check therefore has three outcomes rather than two — **passed**, **failed**, and **undecided** for a check whose evidence sits in a shard that is not on disk. Undecided is reported separately, is never counted as a pass, and is turned into a failure by `--require-complete`. A failure is never deferred: a wrong shape inside a shard that *is* on disk fails while other shards are still missing. That is what makes the audit usable while a 475 GiB download is in progress, and it is the difference between this mode and a run over a complete checkpoint, where the two collapse into one.

### The Engram hash front end

`src/encoding/engram.py` is the consumer the 189.13 GiB of tables were missing, and it needs neither the checkpoint nor a download. Nothing in the checkpoint names a row: the row ids are computed at inference time from the tokenizer, so the tables are unusable without this arithmetic. The module implements, in pure Python:

- `build_compressed_token_map(tokenizer)` — the normalizer chain the reference builds (`NFKC` → `NFD` → strip accents → lowercase → collapse whitespace runs → map a lone space to a `U+E000` sentinel → strip → restore the sentinel), the `"�"` raw-decode fallback for ids the normalizer destroys, and first-seen interning. It returns the per-token-id lookup and the number of distinct keys. Note the sentinel: without it a token that normalizes to nothing and a token that normalizes to `" "` would fold together.
- `EngramLayout.from_config(config)` — takes primes in ascending order starting just above `engram_vocab_size`, hands them out across Engram layers without reuse, and derives the flattened bucket list and its offsets. `verify()` re-derives the row count and compares it against `engram_num_embeddings`, and also checks that the buckets tile the table with no gap and no overlap.
- `compute_hash_multipliers(...)` — the reference's `numpy.random.default_rng(10007 * layer_id)` stream, one multiplier per lookback position, forced odd. The bound divides by the **compressed** vocabulary, so a wrong `engram_compressed_vocab_size` silently rehashes both tables while the weights stay put. `NgramHasher` refuses to construct when the map size disagrees with the config, which is where the reference's `assert` lives.
- `NgramHasher.hash_ids(...)` — the 2-gram/3-gram/4-gram rolling XOR over the masked lookback window, `mod` each bucket, plus that bucket's offset. Dead tokens are sticky: once a position's window crosses a masked token every longer n-gram is padded, not just the one that reached it. The cache carries the lookback across chunk boundaries so a prefill/decode split yields the same ids as a single call.

`is_prime` is a deterministic Miller-Rabin over the first twelve prime bases instead of the reference's `sympy.isprime`, because `sympy` is not a dependency of this repository. numpy is imported lazily inside `compute_hash_multipliers` for the same reason. The module also exposes a CLI:

```bash
python -m src.encoding.engram \
  --config /path/to/config.json \          # or .../inference/config.json: either shape
  --tokenizer /path/to/tokenizer_dir     # optional
```

It prints, per Engram layer, the derived row count against the declared one and the difference, the GiB the two readings imply, the multipliers, the pad id's compressed id, and a sample of hash ids checked to be inside the table, then exits non-zero if anything disagrees. With `--tokenizer` it additionally rebuilds the compressed map and fails when its size does not match `engram_compressed_vocab_size`.

The module reads its eight keys through `V41Config.engram_block()` rather than parsing the file itself. Seven of the eight — the layer ids, the n-gram size, the head count, the head dim, the vocabulary, the compressed vocabulary and the row counts — are spelled identically in both layouts, so the derivation was never shape-dependent; the pad id is the one real rename (`engram_pad_id` against `engram_pad_token_id`), and the schema is where it is paid.

### What the V4-Flash runtime already provides

A backbone layer in V4.1 carries the same module set as a V4-Flash layer, in the same names: `attn.{wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink}`, `attn_norm`, `ffn_norm`, `ffn.gate`, `ffn.shared_experts.{w1,w2,w3}`, `ffn.experts.N.{w1,w2,w3}`, and the six `hc_attn_*` / `hc_ffn_*` hyper-connection tensors. Hyper-connections (`hc_mult` 4, 20 Sinkhorn iterations) and the DSpark MTP stack are already implemented for V4-Flash, and the 384-expert / top-6 / `sqrtsoftplus` MoE is a geometry change from 256 experts, not a new algorithm.

The deltas a V4.1 path would actually have to add:

| Delta | Evidence |
| --- | --- |
| Shared CSA2 compression | `attn.compressor.*` on 4 layers instead of 41, plus new `kv_source_layers` / `index_source_layers` / `Full`/`Reindex`/`Reuse` mode logic |
| Hierarchical sparse indexer | New `indexer.wk` + `indexer.k_norm` (absent in V4-Flash) and `candidate_source_layer` / `candidate_topk_blocks` / `candidate_block_size` |
| Engram | 189.13 GiB of n-gram hash tables and a 99,092-entry compressed vocabulary. The tokenizer-side front end is `src/encoding/engram.py`; the GPU consumer of the rows is not written |
| Vision | 259 `vision.*` tensors, a 4-tensor aligner, and `ffn.gate.bias_vl` — a vision-conditioned routing bias present as F32 `[384]` on all 40 backbone layers and on all 3 MTP layers, and absent from V4-Flash |
| Hash-routing removal | V4-Flash has `layers.{0,1,2}.ffn.gate.tid2eid`; V4.1 has no `tid2eid` tensor and no `n_hash_layers` key |
| Hyper-connection head removal | V4-Flash has top-level `hc_head_{fn,base,scale}`; V4.1 has none, and the last MTP stage has none either |
| MTP head rename | V4-Flash `mtp.2.markov_head.{markov_w1,markov_w2}`; V4.1 `mtp.2.markov_head.{embed,head}` |
| Compressor APE removal | V4-Flash `attn.compressor.ape` on 41 layers; V4.1 has no `ape` tensor |

Two V4.1 MTP asymmetries are worth recording before any loader is written: only `mtp.0` has `main_norm` and `main_proj` (F8_E4M3 `[5120, 15360]`, with 15,360 = 3 MTP layers × 5120), and only `mtp.2` has `markov_head.embed`, `markov_head.head` (both BF16 `[129280, 256]`), `confidence_head.proj` (BF16 `[1, 5376]`, i.e. 5120 + `dspark_markov_rank`) and `norm`. `mtp.1` has no module beyond attention, FFN and hyper-connections.

The removed `hc_head_*` is not a cosmetic deletion, and the parameter shapes alone do not reveal what replaced it. In the V4-Flash runtime both `src/models/deepseek_v4/runtime.py` and `src/models/deepseek_v4/dspark.py` collapse the hyper-connection copies before the head with a dedicated `hc_head` projection. The V4.1 reference instead collapses with the `pre_mix` carried out of the **last block's FFN** — `h = layer.hc_pre(h, pre_mix)` before `self.norm(h)`, and `self.hc_pre(x, pre_mix)` inside the DSpark head — and passes `hc_eps` into `ParallelHead` without using it. So a V4.1 loader must carry `pre_mix` out of every block and use the final one, rather than looking for a head tensor that no longer exists. This is the card's **Single-Pass mHC**; the parameterization is otherwise unchanged (`hc_mult` 4, `mix_hc` = 24, `hc_dim` = 20480, `hc_sinkhorn_iters` 20), with `hc_eps` the one new key.

## Validated performance

**None.** No prefill or decode throughput, latency or memory figure has been measured for V4.1-Flash, and none is implied by the audit. The reasons are concrete:

- The checkpoint's 475.24 GiB of tensor payload is not on this host in full, and no weight byte has been read from the shards that are.
- The four RTX 2080 Ti cards in the reference host hold 22 GiB each, so even a TP4 shard of the routed-expert and Engram tensors does not fit.
- The released reference stack (`inference_requirements.txt`) requires `torch>=2.10.0` and `tilelang==0.1.8`. The `deepseek` environment has `torch 2.9.1+cu128` and no `tilelang`, so the reference implementation cannot be run here to produce a comparison point either.

The model card's "8B parameters per token during prefill / 16B during decode" and "890 bytes per token" KV figures are the vendor's numbers. This repository has not reproduced them and this page does not restate them as measurements.

## Correctness and precision

The audit runs 38 checks against the header prefixes, and 39 against a checkpoint that ships `model.safetensors.index.json` — the release does. They are grouped as follows:

- **Config (12 checks).** All 36 required keys present; `len(compress_ratios) == n_layers + n_mtp_layers`; MTP layers compress nothing (`compress_ratios[40:43] == 0`); `kv_source_layers ⊆ index_source_layers`; every source layer reads compressed positions; every index source has a KV source at or below it; `candidate_source_layer` is the first layer after `kv_source_layers[-1]`; `dspark_target_layer_ids` are the last `n_mtp_layers` backbone layers; one Engram table size per Engram layer; `engram_layer_ids` inside the backbone; `engram_compressed_vocab_size` set.
- **Engram (5 checks).** The tables sit on exactly `engram_layer_ids`; each Engram layer has its gate and value projection; the tables are at least as large as the derived bucket ranges and no larger; the tables are F8_E4M3 with one E8M0 scale per 32 channels; the tables use a 1×32 per-row block.
- **CSA2 (5 checks).** `compressor.wkv` exactly on `kv_source_layers`; `compressor.wgate` exactly on the ratio>1 KV sources; `indexer.wk` exactly on `kv_source_layers`; `indexer.wq_b` exactly on `index_source_layers`; Full/Reindex/Reuse partition the backbone.
- **Inventory (4 checks).** Every backbone layer has its full tensor set; the known shapes match the config; the F32/BF16 tensors are not quantized; the tensor count and byte total are non-zero.
- **Packing and scales (3 checks).** Every tensor's byte extent matches its shape and dtype; every weight/scale pair blocks evenly; all non-Engram FP8 weights use a 32×32 block.
- **Experts (3 checks).** `w1`/`w2`/`w3` are FP4 packed into `I8` with FP4-block-32 E8M0 scales; every backbone layer has 384 routed experts; every MTP layer has 128.
- **DSpark (3 checks).** The Markov and confidence heads match the config; `main_proj` is `n_mtp_layers * dim` wide; every MTP layer carries attention and FFN, but only the last carries the heads.
- **Vision (2 checks).** All 32 blocks present; the encoder and projector shapes match the config.
- **Index (1 check, only when an index is present).** Every local shard holds exactly the tensors the index assigns it — which is what makes the other 38 answerable over a partial checkpoint at all.

That totals 38 checks without an index and 39 with one. A total is not the same as a "pass" count: on the 20 shards downloaded at the time of the run, 31 passed and 8 were undecided, because a check whose evidence is in a shard that has not landed has a third outcome rather than a pass. [Auditing the shards while the checkpoint arrives](../performance/deepseek_v4_1_shard_audit.md) records both runs and the 8 waiting checks with the shards each one waits on. The 8 are the shape-level claims — the FP8 block size, the FP4 expert packing, the Engram table dtypes and row blocks, the `head`/`norm` shapes, and the two DSpark head claims — while every presence claim (the CSA2 partition, the per-layer tensor sets, the expert counts, the Engram placement, the vision block set) resolves from the index and the first shard that lands.

The config schema is checked in both directions against the released pair, not only against a fixture. `as_reference_dict()` on the nested file reproduces the flat file exactly — all 64 keys, every value equal — and on the flat file it round-trips to itself, so the alias table is lossy in neither direction; `differs_from()` reports exactly the 12 Transformers-only fields and nothing else, which is the statement that no field both files state disagrees. All 76 canonical fields (66 text, 10 vision) have an alias row, every row names a key the released files actually carry, and the 64 the flat file is expected to carry are exactly the 64 it has — no key unaccounted for and none named that is absent. `tests/test_models_deepseek_v4_1_config.py` holds all of that: 13 tests that need no checkpoint — a 64-key toy config in the flat layout and the same toy model written out in the nested layout by hand, asserted to read identically, plus the negative direction (a `candidate_source_layer` that is not an index source, a compress ratio with no source at or before it, two configs disagreeing on a field, and an unknown key, each reported or ignored rather than fatal) — and 8 more that skip unless a released pair is on the host. The toy nested file is built by hand rather than derived from the alias table on purpose: deriving it would make the round-trip test agree with whatever the table happens to say.

Two facts from this section deserve emphasis because they were derived rather than read off:

The Engram row counts are *derived*, and the derivation is exact to the row. The reference draws primes in order starting just above `engram_vocab_size` (16,000,000), hands them out across both layers without reuse, taking `(max_ngram_size − 1) × n_heads = 3 × 8 = 24` primes per layer, and `NgramHashState` builds its bucket offsets as a running sum over the layer's *flattened* prime list. The largest id a layer can produce is therefore `sum(primes) − 1` and the table needs exactly `sum(primes)` rows. Both layers match `engram_num_embeddings` with a difference of zero: the declared `[384006168, 384016682]` equals the derived values. That derivation is no longer only a claim on this page: `EngramLayout.from_config` reproduces it, `verify()` returns no problems, and `python -m src.encoding.engram --config …` prints a difference of 0 for both layers and exits 0. Layer 1's 24 primes run from 16,000,057 to 16,000,463 and layer 14's from 16,000,477 to 16,000,889; all 48 are distinct and all sit above `engram_vocab_size`, which is why one shared prime stream can serve both tables without overlap.

The CSA2 modes are derived from tensor presence, not from a config field. Diffing the three shard shapes against each other gives an exact per-mode tensor delta: a Reuse layer carries nothing extra; a Reindex layer adds `indexer.wq_b.weight`, `indexer.wq_b.scale` and `indexer.weights_proj.weight`; a Full layer adds those three plus `compressor.wkv.weight`, `compressor.norm.weight`, `compressor.wgate.weight`, `indexer.wk.weight` and `indexer.k_norm.weight`. So Full = owns `indexer.wk`, Reindex = owns `indexer.wq_b` only, Reuse = neither, which partitions the backbone as Full `[2, 8, 14, 20]`, Reindex `[24, 28, 32, 36]`, Reuse the other 32 layers.

`compressor.wgate` is the one tensor that separates layer 20 from the other three Full layers, and the reason is in the config value rather than in the tensor set: the reference constructs the gate only `if compress_ratio > 1`, because the gate is the learned softmax that pools `compress_ratio` consecutive tokens into one KV latent and a ratio of 1 has no group to pool. `compress_ratios` is `[0, 0]`, then 18 entries of 2, then 20 entries of 1, then `[0, 0, 0]` — so layer 20 is the single KV source at ratio 1. All four sources store `compressor.wkv` as BF16 `[512, 5120]` on disk; the reference declares that matrix FP32 for the ratio-2 layers and BF16 for ratio 1, so a loader reading the checkpoint has to upcast the three ratio-2 copies itself. (`compress_ratios[0:2]` and `compress_ratios[40:43]` are zero: layers 0 and 1, and all three MTP layers, read no compressed positions at all.)

Negative controls were run to confirm the checks are live rather than vacuous: perturbing a config key the audit does not consume leaves the result at 38/38 with exit 0, while setting `engram_n_heads = 7` drops it to 37/38 with exit 1, reporting `declared=[384006168, 384016682] derived=[336004849, 336012883]`.

The audit establishes nothing about numerics. No tensor value has been read, so no claim about quantization error, activation range or output parity is available.

The Engram front end is held to a different and stronger standard than the rest of this page, because it is executable. `tests/test_encoding_engram.py` pins the eight multipliers, both row counts and each layer's prime range as literals, so a change that would silently rehash 189 GiB fails the suite instead of passing quietly; it also asserts the negative direction (a tampered `engram_num_embeddings` is reported, not accepted) and checks `is_prime` against `sympy.isprime` across both bucket windows plus its edge cases, skipping rather than passing where `sympy` is absent. Separately, `NgramHasher.hash_ids` was compared position-for-position against the released reference's `NgramHashState.forward` on the host — plain sequences, masked sequences, a prefill-then-decode split across the cache, and a batch of two — and the outputs agree exactly. That comparison needs the unpacked reference under `/tmp` and the reference's `torch`, so it is a host run rather than a committed test, and it was made with the compressed map stubbed to an identity map over 99,092 entries; it validates the hashing arithmetic, not the tokenizer normalization feeding it.

One limit on the Engram result remains, and it is narrower than it was. The compressed vocabulary has since been rebuilt against **V4.1's own** `tokenizer.json` rather than the V4-Flash one: 6,367,257 bytes, sha256 `c90dfa01…`, carrying `<｜System｜>` and `<｜deepseek_image｜>` where the V4-Flash tokenizer instead carries `<｜image｜>` and `<｜image2｜>`. It collapses to exactly 99,092 distinct keys with the ids filling `[0, 99092)` with no gaps, matching `engram_compressed_vocab_size`, and it produces the same eight hash multipliers and the same sampled hash ids as the V4-Flash tokenizer did — which is the expected outcome, since the multipliers range over the *compressed* vocabulary size and that size is 99,092 either way, but it is now measured rather than argued. The 111-byte difference and the substituted special tokens change which id spells a marker, not how many distinct keys the normalization folds 129,280 ids into. The limit that stays open is downstream of the tokenizer entirely: no row has been read out of either table, so the row count and the id range are consistent with each other while nothing here confirms that the FP8 payload at a given row is the embedding the reference would fetch.

## Reproduction

The audit needs only the first few megabytes of each shard. Downloading just enough bytes to cover the largest header in this checkpoint — 261,440 bytes — yields 48 files totalling about 144 MB:

```bash
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /path/to/dsv41-header-prefixes \
  --config /path/to/config.json \
  --header-prefix
```

`--config` takes either released shape; the schema normalizes it to the flat key set before the first check runs, so `inference/config.json` gives the identical 38/38. Without `--config` the script resolves `<checkpoint-dir>/config.json` first and falls back to `<checkpoint-dir>/inference/config.json`.

Against a checkpoint rather than a prefix tree the same command runs without `--header-prefix`; the tensor inventory is identical because the payload is never read. Two flags exist for the checkpoint case. `--index PATH` reads the presence half from an index other than `<checkpoint-dir>/model.safetensors.index.json` — with the release's index in place the script reports how many of the 96,085 tensors are readable and adds the 39th check, `index: every local shard holds exactly the tensors the index assigns it`. `--require-complete` turns every undecided check into a failure and exits non-zero, for a caller that needs a binary answer and not a partial one.

```bash
# the release as it downloads: 31/39, 8 undecided, exit 0
python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash

# the same run read strictly: undecided counts as failure, exit 1
python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash --require-complete
```

`--json out.json` writes the report as machine-readable JSON — every check carrying a `status` of `pass` / `fail` / `undecided` and a `detail` naming the failing evidence or the shards waited on — and `--list-tensors PATTERN` prints individual tensors as `name<TAB>dtype<TAB>shape<TAB>bytes<TAB>shard`. `--expect-fp8-block 32 32` overrides the FP8 block size the scale check assumes.

Getting the prefixes does not need a checkpoint download. Fetch the first 3,000,001 bytes of each of the 48 published shards — a byte range covers the largest header in this checkpoint, 261,440 bytes, many times over — and write them to any 48 local files:

```bash
# One range request per shard. The URL layout is mirror-specific; only the first
# 3,000,001 bytes of each shard matter, and the local name is free-form.
curl -r 0-3000000 -o h00001.bin "<shard-1-url>"
# ... repeated for h00002.bin .. h00048.bin
```

Shard files are detected either by a `.safetensors` suffix or by their first 8 bytes looking like a plausible header length, so the local names above are arbitrary. A shard whose header is byte-identical to another's is skipped as a duplicate — worth knowing because a mirror that serves the same file under two names would otherwise be counted twice. The audit's own inventory (`96,085 tensors`, `475.24 GiB`) is the check that all 48 shards were actually distinct.

To confirm the checks are live, make a copy of the config with `engram_n_heads` changed to 7 and expect 37/38 with exit 1.

The Engram front end needs only the config, and optionally a tokenizer directory; neither the checkpoint nor any of the header prefixes above are required:

```bash
python -m src.encoding.engram \
  --config /path/to/config.json \
  --tokenizer /path/to/DeepSeek-V4.1-Flash   # optional
```

On this host, against the released `config.json` and V4.1's own tokenizer, that prints:

```
Engram layers [1, 14] | 24 hash columns per position
  layer   1: 24 buckets, primes 16000057..16000463, rows derived 384006168 vs declared 384006168 -> ok
            embedding rows [384006168 x 256] = 91.55 GiB at one byte per element, plus 2.86 GiB of row scales
  layer  14: 24 buckets, primes 16000477..16000889, rows derived 384016682 vs declared 384016682 -> ok
            embedding rows [384016682 x 256] = 91.56 GiB at one byte per element, plus 2.86 GiB of row scales
  [ok] the primes add up to the declared row counts

tokenizer /mnt/data3/DeepSeek-V4.1-Flash: 129280 tokens -> 99092 compressed ids
  [ok] matches engram_compressed_vocab_size (99092)
  pad id 2 -> compressed 2
  layer 1 multipliers: [76632096046245, 4839876093313, 35959672319349, 73987337458391]
  layer 14 multipliers: [67716810739261, 51510806800915, 30921347202721, 82619226485591]
  layer 1: 192 ids from 8 tokens, min 3395123 max 382971602 of 384006168 rows -> all in range
  layer 14: 192 ids from 8 tokens, min 2266587 max 383166700 of 384016682 rows -> all in range

[PASS] Engram layout verification
```

Exit code 0. Both row counts are `ok` — the difference is zero for each layer — and the sampled ids land inside the tables. `--tokenizer` is optional: without it the second block is replaced by `[SKIP] compressed token map: pass --tokenizer to check it against the config`, and the layout checks still run. Passing a tokenizer whose map does not come out at 99,092 fails instead of skipping, because every hash multiplier derives from that size. `--json out.json` writes the same report machine-readably; note that it writes the report *and* the tokenizer leg's findings even when a check fails, so a caller can diff a failure rather than re-run it.

The config schema is its own command, and with the released pair on disk it is the one that checks the checkpoint's two descriptions of itself against each other:

```bash
python -m src.models.deepseek_v4_1.config /path/to/DeepSeek-V4.1-Flash          # config.json (nested)
python -m src.models.deepseek_v4_1.config /path/to/DeepSeek-V4.1-Flash/inference/config.json  # (flat)
```

Both print the same shape line and `[ok] the config is self-consistent` with exit 0. To see the round-trip and the difference list directly:

```python
from src.models.deepseek_v4_1.config import load_config
import json

hf = load_config("…/config.json")
flat = load_config("…/inference/config.json")
assert hf.as_reference_dict() == json.load(open("…/inference/config.json"))  # 64 keys, exact
assert flat.as_reference_dict() == json.load(open("…/inference/config.json"))  # round-trips
for line in hf.differs_from(flat):
    print(line)   # exactly the 12 Transformers-only fields
```

The same two directions are covered by the test suite, which needs no checkpoint either:

```bash
python -m pytest tests/test_encoding_engram.py -q
```

Expect 21 tests collected with nothing failing. `numpy` is not a declared dependency of this repository, so on a host without it the tests that draw the multipliers skip instead of failing, as do the `sympy` and tokenizer legs without their packages; a skip is not a pass, and the layout tests that need neither still run. Checked both ways: 21 passed with `numpy`, 8 passed and 13 skipped with it blocked.

## Known limitations

- **No generation of any kind.** No embedding is loaded and no forward pass exists for this architecture. The tokenizers that have been exercised are V4.1's own and the V4-Flash one, and only to rebuild the Engram compressed vocabulary — not to tokenize a prompt, and not to render a chat template.
- **A complete local checkpoint is neither here nor usable here.** The release is 48 shards and 475.24 GiB; 20 of them had downloaded at the time of the audit, so 42,303 of the 96,085 tensor descriptors are readable by header and the other 28 shards' checks are undecided rather than passing. The tensor facts that do not depend on a landed shard come from 48 header prefixes. No weight payload has been read in any run. That 475.24 GiB does not fit on the four 22 GiB cards available here in any case.
- **The two config layouts are not interchangeable field for field.** A consumer handed only `inference/config.json` has no `model_type`, no `architectures`, no token ids, no `topk_method`, no `norm_topk_prob` and no `param_dtype`, and its top-level `dtype` is the *quantization* dtype rather than the storage one. `resolve_config` prefers the nested file for exactly this reason and treats the flat one as the fallback; the schema makes the missing fields visibly absent rather than silently defaulted, but it cannot supply them.
- **The audit validates metadata consistency, not correctness.** It proves the config and the tensor inventory agree with each other. A checkpoint could satisfy all 38 checks and still be unusable, and a wrong value that is *consistently* wrong in both the config and the shapes would pass.
- **Engram is addressable but not consumed.** The compressed token map, the prime-derived bucket layout and the n-gram hasher now exist in `src/encoding/engram.py`, and the two layers reproduce their declared row counts exactly, so the 189.13 GiB of tables can be indexed. What is still missing is everything downstream: no row has been read, no embedding lookup has been written, and no gate or value projection has been run (see [Correctness and precision](#correctness-and-precision)).
- **The vision path is unvalidated in both directions.** The 263 vision and aligner tensors are accounted for and their shapes match the config, but no image has been processed and no projector has been run.
- **No reference comparison is possible on this host.** The released stack needs `torch>=2.10.0` and `tilelang==0.1.8`; neither is available in the `deepseek` environment.
- **The prompt format changed and is unimplemented here.** DSML tags gain a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`), reasoning effort becomes a numeric budget in 1–100 rendered only under `thinking_mode="thinking"` at index 0, and mid-conversation `<｜System｜>` messages are supported. None of that is wired into this repository's chat templates.
- **Memory is the structural problem, not the kernels.** 58.0% of the checkpoint is routed experts and 39.8% is two Engram tables. A 4×22 GiB TP4 deployment cannot hold either, so any V4.1 plan has to place experts and Engram in host memory or on disk before kernel work matters.

## Evidence and related notes

- `src/models/deepseek_v4_1/config.py` — the config schema that reads both released layouts into one view, with an `as_reference_dict` inverse and a `python -m src.models.deepseek_v4_1.config` CLI
- `scripts/audit_dsv41_headers.py` — the host-only header audit, reading its config through the schema above and distinguishing a shard that is missing from a shard that is wrong
- `tests/test_models_deepseek_v4_1_tensor_audit.py` — 16 tests over the audit's three outcomes, the last of which skips unless the released checkpoint is on the host
- [Auditing the shards while the checkpoint arrives](../performance/deepseek_v4_1_shard_audit.md) — the two audit runs, the per-layer tensor partition and the shapes the landed shards assert
- `src/encoding/engram.py` — the Engram compressed token map, bucket layout, hash multipliers and n-gram hasher, plus a `--config`/`--tokenizer` CLI
- `tests/test_models_deepseek_v4_1_config.py` — 21 tests over the schema: the alias table, the round-trip to the flat layout, the shape-independent reading of a toy model in both layouts, and the released pair when it is on the host
- `tests/test_encoding_engram.py` — 21 tests pinning the primes, multipliers and row counts, with the tokenizer and `sympy` legs skipping when unavailable
- `configs/config.json` — the validated V4-Flash config used as the delta baseline
- [DeepSeek-V4-Flash](deepseek-v4.md) — the validated runtime whose layer structure the V4.1 backbone reuses
- [Benchmark reporting rules](../guides/benchmarking.md) — required before any V4.1 number is quoted
- Model card and released config: `https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash`
