# DeepSeek-V4.1-Flash

## Runtime status

**Runs on the host, and its routed experts also run on the four cards: the complete released checkpoint loads into `src/models/deepseek_v4_1` and generates correct greedy text end to end, on the CPU and, behind `--expert-device cuda --expert-world 4`, on all four RTX 2080 Ti.** All 48 shards and all 96,085 tensors pass the header audit; `load_backbone` fills the 924-parameter text backbone from the shards — 330 of those parameters quantized and dequantized at load, 16.79 GiB retained — and leaves the 268.95 GiB of routed experts and the 189.13 GiB of Engram tables where they are, reading them per miss and per gather. What follows is measured rather than claimed: a generated token costs **15 to 42 s** on this host — 31.15 s and 42.37 s over two runs, and a 15.3 s decode step with the expert pages warm — of which 99.7% is turning the fp4 expert codes into bf16 numbers and 0.3% is reading them. The four cards are the cheaper place to do that work and now do it: `DeviceRoutedExperts` stages the checkpoint's packed fp4 rows and runs the fp4 kernels on all four, a measured **1.06–1.14 s per step** against the host's 15.3 s, and it needs no NCCL and no all-to-all to do it, because the only thing that crosses back is 20 KiB per card per layer. The dense tree was host code too and has since been cut across the same four cards, one process per card — **722–747 ms per step**, of which the staging is the larger half — while the Engram tables are still in the shards or in RAM; the expert-parallel split and its numbers are in [the device page](../performance/deepseek_v4_1_flash_device_experts.md). **What this is not is a serving path**: the runnable surface is `generate.py`, one request at a time, with no batching, no continuous batching and no native engine for this geometry.

The attention half of a V4.1 layer was the first piece of layer-level code here: `src/models/deepseek_v4_1/attention.py` builds an `AttentionStack` over a config and runs a prefill or a decode step through it, and on its own it is layers with no model around them, driven in its tests over random parameters. What that module is missing is supplied by `src/models/deepseek_v4_1/modules.py` — embedding, hyper-connections, the MoE with its shared and routed halves, the Engram consumer, the norm and the head — and by `loader.py`, which fills that tree from the checkpoint and pairs it with the hash front end its forward needs. `kv_source_layers` and `candidate_source_layer` are config key names *and* values a forward pass reads. `ffn.gate.bias_vl` is still only the first kind: it is a real F32 `[384]` tensor on every backbone layer of the checkpoint, read out of the shards, but nothing consumes it. `src/models/deepseek_v4_1/kernels.py` is the op layer those attention layers call into on the host, and `cpp_engine/engine/deepseek_v4_engine.cpp` and `src/models/deepseek_v4/` target the 43-layer, 4096-hidden DeepSeek-V4-Flash geometry — a different model, not a stale copy of this one.

What is validated is that set of facts read out of the checkpoint's own metadata, an Engram front end that needs only the config and a tokenizer, and — since `loader.py` — a text backbone filled from the shards and generating from it:

- `src/models/deepseek_v4_1/config.py` reads either config layout into one schema, so the two files that describe this model are checked against each other rather than parsed ad hoc by each consumer.
- `scripts/audit_dsv41_headers.py` reads the config through that schema and runs 38 checks over the safetensors headers — 39 when the checkpoint's own `model.safetensors.index.json` is present, which is what the release ships.
- The audit has run against the real checkpoint at every stage of its download, and its coverage grew with it. It now runs against the complete checkpoint: **48 local shards, 96,085 of 96,085 tensors readable, 39 of 39 checks passed, none undecided, none failed, 475.24 GiB**, and it exits 0 under `--require-complete`, which is the flag that turns an undecided check into a failure. The earlier partial runs — 31/39 with 8 undecided on the first 20 shards, 35/39 with 4 undecided on 46 — are recorded as such in [the shard audit](../performance/deepseek_v4_1_shard_audit.md); what settled them was the two Engram shards arriving, not a change in the checks.
- `src/models/deepseek_v4_1/loader.py` loads that checkpoint. `V41Checkpoint` maps the 48 shards, `load_backbone` fills `modules.Backbone` with 924 parameters and 330 dequantizations, and the routed experts and Engram tables are read from the mapping on demand rather than allocated — which is what makes a 475 GiB checkpoint load into 16.79 GiB of retained parameters in 65.5 s warm. It is a correctness path and is measured as one: [the host-run record](../performance/deepseek_v4_1_flash_host_run.md) has the load report, the phase table and the token cost.
- `src/encoding/engram.py` re-derives the Engram bucket layout and hashes token n-grams onto it. The primes it draws add up to the declared `engram_num_embeddings` exactly, so the 189.13 GiB of Engram tables are addressable rather than merely counted.
- `src/models/deepseek_v4_1/kernels.py` gathers the reference's six TileLang ops behind one module. All six already exist in `src/kernels/ops.py` and are re-exported from there; the one thing that module adds is `fp4_act_quant_e4m3`, the E4M3-scale branch of `fp4_act_quant` that the compressed-KV path calls and that `src/kernels/ops.py` does not implement. The reference stack needs `torch>=2.10.0` and `tilelang==0.1.8`, so it cannot run here and there is no numeric oracle for a V4.1 forward pass; the tests pin the reference's documented arithmetic instead of its output.
- `src/models/deepseek_v4_1/attention.py` implements the CSA2 attention half of a backbone layer in pure PyTorch, above those ops: the three layer modes, the shared `compress_kv`/`index_k`/`topk_idxs`/`candidates` slots, and the two-level sparse indexer. It runs — `AttentionStack(cfg)(x, 0)` prefills and a `start_pos`-carrying forward decodes — on random parameters, with no checkpoint and no numeric oracle. `tests/test_models_deepseek_v4_1_attention.py` holds the parts that are decidable without one.
- Upshot: 96,085 tensors, 510,286,023,000 B (475.24 GiB), a fully consistent tensor inventory with nothing undecided, an Engram layout that closes to the row, a checkpoint-specific configuration that differs from the validated V4-Flash config in the ways listed below, and a text backbone that loads all of it and answers from it.

Generation is now measured rather than unmeasured: greedy decode returns `' Paris'` and then `'.'` for `"The capital of France is"` on every path, host and cards alike — the next token is a half-logit tie between EOS and `' The'`, and the host path's own prefill and stepwise decodes land on opposite sides of it ([the numbers behind that](../performance/deepseek_v4_1_flash_device_experts.md)). **TPS is still not a useful number for the host** — a generated token is 15 to 42 s on the host CPU, against a measured **1.06–1.14 s per step** with the routed experts on the four cards and **722–747 ms** once the dense tree is cut across them as well ([the device page](../performance/deepseek_v4_1_flash_device_experts.md)) — and numerical parity with the reference is still unclaimed, because the reference stack needs `torch>=2.10.0` and `tilelang==0.1.8` and neither is available here. See [Known limitations](#known-limitations).

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

The counts in that table are per *shard*, which is not the same as per *layer*: layer 1's six Engram tensors are shipped in `h00047` while layer 1 itself is `h00004`, so `h00004` holds 2,334 tensors and the layer has 2,340. Counting the index by layer instead gives an exact partition — 2,334 for a Reuse layer, 2,337 for a Reindex layer, 2,342 for a Full layer, 2,340 for layer 1 (+6, Engram), 2,341 for layer 20 (the Full set minus `compressor.wgate`, since `compress_ratios[20] == 1`) and 2,348 for layer 14 (Full + Engram) — with every one of the 40 layers a strict superset of the plain set. The landed shards match the index's assignment shard for shard. [Auditing the shards from arrival to complete](../performance/deepseek_v4_1_shard_audit.md) records that partition in full, together with the tensor shapes read out of the shards that have downloaded; no per-layer count in this paragraph is taken from a model card.

Two Engram tables dominate the checkpoint. `layers.1.engram.embed.weight` is F8_E4M3 `[384006168, 256]` with an F8_E8M0 `[384006168, 8]` scale, 91.55 GiB; `layers.14` is `[384016682, 256]`, 91.56 GiB. Together, 768,022,850 rows. The two tables hold 196,613,849,600 embedding parameters (row count × 256), which is the card's "196B parameters, sparsely accessed via token-based lookup". Each layer also has `engram.wkv` F8_E4M3 `[25600, 6144]`, an F8_E8M0 `[800, 192]` scale, and BF16 `engram.q_weight` / `engram.k_weight` of shape `[4, 5120]`.

### Quantization, verified from the headers

- Every non-Engram FP8 weight uses a **32×32 block**: for example `attn.wq_a` F8_E4M3 `[1280, 5120]` with scale `[40, 160]`, `attn.wo_a` F8_E4M3 `[8192, 4096]` with scale `[256, 128]`.
- The Engram tables use a **per-row (1, 32)** block instead — `[384006168, 256]` against a `[384006168, 8]` scale.
- Routed experts are stored as `I8` holding two FP4 values per byte. `experts.N.w1.weight` is `I8 [2304, 2560]` with an F8_E8M0 `[2304, 160]` scale, i.e. FP4 with a block of 32 along K; `w2` is `I8 [5120, 1152]` with scale `[5120, 72]`. The reference `inference_convert.py` repacks these to E4M3 via its `cast_e2m1fn_to_e4m3fn` with `fp8_block_size = fp4_block_size = 32` and `MAX_OFFSET_BITS = 6`, because 6.0 × 2⁶ = 384 stays inside the E4M3 range.
- Shared experts are plain F8_E4M3 32×32 (`w1` `[2304, 5120]` scale `[72, 160]`), and `attn_sink`, `q_norm`, `kv_norm`, `attn_norm`, `ffn_norm` stay unquantized as F32/BF16.

## Implemented execution path

**There is a V4.1 model and it runs on the host, and its routed experts also run on the cards.** The path is a config schema, a host-only audit, an Engram front end, an op layer, a CSA2 attention stack, a backbone above it, and a loader that fills that backbone from the released 48 shards — `src/models/deepseek_v4_1/loader.py` and the `generate` entry point above it. It generates correct greedy text; what it does not have is a serving path, and the sections below separate what is measured from what is still missing.

### The op layer

The reference implements six ops in TileLang — `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`, `hc_split_sinkhorn` — and `import tilelang` runs at module scope in its `kernel.py`. This host has neither `tilelang` nor `torch>=2.10.0` and will not get them, and its sm_75 cards have no FP4 tensor core for the reference's own fallbacks, so **no V4.1 forward pass can be run against a reference at all here**.

The first thing that followed from that was to check whether the six ops are already here rather than to write them again. All six are, in `src/kernels/ops.py`. Two of them were then compared against the reference's arithmetic rather than assumed equal — `sparse_attn` and `hc_split_sinkhorn`, the two whose contracts are least legible from their signatures — and both agree to the bit on the fixtures in `tests/test_models_deepseek_v4_1_kernels.py`. That comparison is also what retired the duplicate: `hc_split_sinkhorn` in `src/kernels/ops.py` reshapes through `view(-1, mix_hc)` and so already handles the batched `mixes` a V4.1 block feeds it, where the copy written here looped per batch.

`src/models/deepseek_v4_1/kernels.py` is therefore a **facade, not a second implementation**: it re-exports those six and adds only the one op that is genuinely missing.

That op is `fp4_act_quant_e4m3`. The reference's `fp4_quant_kernel` branches on its *scale dtype*, and `src/kernels/ops.py` implements only the E8M0 branch — the one the indexer takes. The **E4M3** branch is what the compressed-KV path calls (`Attention._compress_kv` asks for `scale_dtype=torch.float8_e4m3fn` at block size 16), and it is not a reformat of the E8M0 one:

- The scale is `e4m3(amax / 6)` — the ratio rounded to E4M3 — where E8M0 stores `2**round(log2(amax / 6))`. The two branches therefore agree on the packed nibbles for most inputs and disagree on every stored scale, by up to a factor of two. Storing `e4m3(amax)` instead is the obvious misreading of "an E4M3 scale" and also round-trips, just with a scale wrong by 6×, so the test asserts the ratio rather than leaving it to a round-trip check.
- The amax floor is `6 * 2**-9` rather than `6 * 2**-126`, which is high enough that an **all-zero block still gets a nonzero scale**. That is the point of the branch: a zero scale cannot be divided out on the way back, and the compressed-KV cache is read back on every decode step.

The shared ops are re-exported rather than imported at each call site because two of the contracts they carry belong to V4.1 rather than to V4-Flash, and a V4-Flash test would not notice them regressing:

- `sparse_attn`'s finite `-1e30` score floor. The reference's kernel comment explains why it is not `-inf`: a query row whose every top-k index is `-1` would compute `exp(-inf - (-inf))` and produce NaN, where the finite floor makes the same row fall out as zeros. A dense per-row softmax written from the formula instead of from that comment gets NaN, so it is tested directly. `src/kernels/ops.py` satisfies this, but nothing in it says a rewrite has to.
- `hc_split_sinkhorn`'s normalization order. The softmax is followed by a *column* normalization before the `sinkhorn_iters - 1` loop begins, and the loop body ends on a column step — the total is not a symmetric round in which the starting axis does not matter. Both orders yield a plausibly doubly-stochastic matrix, so the test pins the resulting values and asserts the other order differs.
- The two gates the identical output shapes hide: `pre` is `sigmoid(...) + eps`, a floor; `post` is `2 * sigmoid(...)`.

`sparse_attn` computes its quantity in one pass over the top-k where the reference runs a block-wise online softmax, so it is exact where the reference accumulates a running rescale and much hungrier in memory. It is a correctness reference and a small-config fixture, not a serving path.

### The CSA2 attention layers

`src/models/deepseek_v4_1/attention.py` is the layer-level half of the model, and the reason it is new code rather than a V4-Flash attention with different constants is that a *mode* has no counterpart in the V4-Flash runtime: the four Full layers, four Reindex layers and 32 Reuse layers derived above differ in which caches they write, not in which weights they hold. It follows the released `inference/model.py`, in pure PyTorch, over random parameters — `AttentionStack(cfg)` builds one `Attention` per backbone layer plus one `SharedAttentionRuntime`, and `forward(x, start_pos)` runs the layers in order. Nothing loads a weight.

A layer that reads compressed positions is not a layer that produces them, and that is the first thing the module has to keep straight. `compress_ratios` is non-zero on 38 of the 40 backbone layers, but only the four `kv_source_layers` — the Full mode, `[2, 8, 14, 20]` — own a `Compressor` and write the shared compressed-KV cache; the other 34 read it. The same split holds one level up: `index_source_layers` is the eight Full and Reindex layers, only the four Full ones own `indexer.wk` and publish index keys, and the eight publish `topk_idxs` that the layers after them reuse instead of re-ranking.

Concretely, per layer:

- **`Compressor`** pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate. Ratio 1 is the degenerate branch and the code has it: one token per group, so `wkv` alone, no gate, no fp32, and the checkpoint's tensor set reflects exactly that (layer 20 is the only ratio-1 Full layer and the only one of the four without `compressor.wgate`). Above ratio 1 the pooling runs in fp32 — the reference's dtype choice here is structural rather than a storage detail — and the partial trailing group waits in `kv_state`/`score_state` between calls, which is what makes a decode step at a non-flush position return `None` and skip the cache write entirely.
- **`Indexer`** is the sparse side attention: FP4 query heads against one shared index key per compressed position, scores rectified with `relu` and combined through `weights_proj` into one score per position, then a top-k. Its keys are `wk` of the *compressor's latent* before RoPE and before quantization, which is why it runs before the cache write rather than after — a layer rotates and quantizes the latent into `compress_kv_cache` only once the indexer has taken what it needs.
- **`Attention`** then attends over two KV sources concatenated into one `sparse_attn` call: a sliding window of raw KV in a ring buffer of `window_size` slots, and the `index_topk` compressed positions the indexer chose. The window is quantized to FP8 over 32 elements, the indexer's queries and keys to FP4 with E8M0 scales over 32, and the compressed KV to FP4 with E4M3 scales over 16 — the last of those being the branch only this repository's `fp4_act_quant_e4m3` implements.

The second level of the hierarchy is `select_candidate_blocks`, and it has one rule that is not obvious from its name. The `candidate_source_layer` — layer 20, a Full layer — narrows the field to `candidate_topk_blocks` blocks of `candidate_block_size` positions before the four Reindex layers rank inside it, so the deeper indexers cost O(candidate pool) rather than O(context). Blocks are scored by their best position, which makes an unreachable position's `-inf` propagate to a whole block and drop it — and makes the block a query is still filling *unreliable*, since a half-filled block's `amax` can be beaten by an older, full one even though it holds the most recent tokens. That block is pinned in explicitly rather than left to the ranking, and it is the kind of rule that is invisible until the decode step where the newest block loses.

RoPE follows the same split as everything else, and the reference's choice is worth naming: the layers that read compressed positions use `compress_rope_theta` (160,000) with YaRN on, the pure sliding-window layers use `rope_theta` (10,000) with YaRN off, and a compressed latent is rotated at the position of the *first* token of its group rather than at the position it was written — group `j` takes position `j * ratio`, because one latent stands for the whole group.

Two things in the reference are deliberately not reproduced. Tensor parallelism is absent: the reference splits heads and groups across ranks and all-reduces the indexer's scores, all of which is the identity at one rank, so what is written here is the one-rank case. And the shared runtime is an explicit object passed to each `forward` rather than the module-level singleton the reference uses, so that two stacks in one process cannot silently share a cache. `reset_state` is the third, smaller divergence: the reference is one conversation per process and never resets, which means a second forward over the same module silently continues the first — a trap for a test rather than a property of the model, so the module has a way out.

### The backbone above attention

`src/models/deepseek_v4_1/modules.py` is what the attention stack was missing to be a model: `Block` (attention, MoE, the two norms, the hyper-connection tensors and the Engram gather, in the reference's order), `Backbone` (embedding, the 40 blocks, the final `hc_pre`, the norm and the head), and the four leaf modules the checkpoint has to be poured into — `EngramTable`, `RoutedExperts`, `SharedExperts` and `Head`.

Three of its behaviours are worth naming because they are the reference's and not the obvious reading:

- **The last block's `pre_mix` collapses the hyper-connection copies.** There is no `hc_head_*` tensor in V4.1, so `Backbone.forward` calls `self.layers[-1].hc_pre(h, pre_mix)` with the `pre_mix` the last block returned and only then norm and head. A loader that looks for a head tensor finds none, and a forward that drops `pre_mix` produces shapes that look right.
- **The MTP heads read the attention *input* of their target layers**, not their output, which is why `forward` collects `main_hiddens` inside the loop rather than after it.
- **`forward` takes `hash_ids` rather than computing them.** They come from the tokenizer through `src/encoding/engram.py`, and the cache that carries them across a prefill is state the model does not otherwise own — so it is an argument, and `LoadedBackbone` is the object that pairs the two.

`Backbone` is also where `sample()` lives: with `temperature == 0` it is an argmax, otherwise it is the reference's Gumbel-max trick, which avoids the host sync `torch.multinomial` would introduce. Nothing in the released config schema sets `temperature`, so it is the 1.0 default — and since the sampling is not the model's job at that point, `src/models/deepseek_v4_1/generate.py` reads the returned logits itself and pins the model to greedy for the duration of a call.

### Loading the released checkpoint

`src/models/deepseek_v4_1/loader.py` is the half that touches disk, and its subject is a size fact: the checkpoint is 475.24 GiB and this host has 88 GiB of VRAM across four cards and about 1 TiB of RAM, so **what loads and what stays in the shards is the central design decision, not a caching detail.**

- `V41Checkpoint` maps the 48 shards through `safetensors` and reads any tensor by name, dequantizing FP8 E4M3 and packed FP4 on the way out. `weight`, `rows` and `nbytes` are the three access shapes: a whole tensor, a row gather with no host-side copy, and a size without a read.
- `checkpoint_weights` fills the 924 dense parameters — attention, shared experts, norms, hyper-connections, embedding and head — which is about 16.8 GiB in bf16 and is what actually becomes resident.
- `CheckpointRoutedExperts` takes the other side. Each of the 40 backbone layers routes to 6 of its 384 experts per token, so a forward needs a window of them and nothing else; the class keeps a bounded LRU of **dequantized** experts per layer (`DEFAULT_EXPERT_CACHE = 16`) and re-reads plus re-expands on a miss. One expert is 17.9 MiB packed and 67.5 MiB expanded, a whole layer is 25.3 GiB expanded, and the 16-expert window is 1.05 GiB per layer, 42 GiB across the backbone. **The window is worth about half a step and no more**: measured over four decode steps, 119 of a token's 240 expert rows missed, warm, and 759 of 1,200 at a five-token prefill.
- `CheckpointEngramTable` is the one component whose access pattern makes the shards a real choice rather than a shrug: a row gather out of a 91.56 GiB table read from a shingled disk. `resident_engram=True` copies both tables into RAM — 189.13 GiB, roughly 750 s of reading, once — and the default gathers from the shards instead.
- `EngramHashIds` adapts `NgramHasher` to the `[b, s, n_engram_layers, n_hash_cols]` tensor `Backbone.forward` expects and carries its lookback across calls; `build_hasher` builds it from the config, the checkpoint's `tokenizer.json` and the derived layout.
- `load_backbone` is the entry point: it builds the backbone, fills it, wires the hash front end and returns a `LoadedBackbone` — the model, a `LoadReport` with the parameter and byte census, the hasher and the hash cache. `LoadedBackbone.__call__` hashes then forwards. There is no `main()` here; the runnable surface is `generate.py`.

The cost of all of that is measured rather than asserted, in [the host run record](../performance/deepseek_v4_1_flash_host_run.md): 15 to 42 s per generated token, 99.7% of it the fp4 → bf16 expansion of the 240 experts a step routes to — 119 of which a warm step still misses — and 0.3% of it the reads, against 1.0 s for a forward whose experts are already expanded and 27.2 s for a first, entirely cold one. One token's routed experts are 4.20 GiB packed and 15.82 GiB expanded. [The device page](../performance/deepseek_v4_1_flash_device_experts.md) is the same subject on the cards, where the expansion never happens: 1.06–1.14 s per step with the tree on the host, 722–747 ms with it cut across the cards.

### Generating from it

The loop above the backbone is small, and it is small because the cost is not in it. `src/models/deepseek_v4_1/generate.py` prefills the prompt in one call, then makes one call per new token, taking each token from the returned logits:

```python
from src.models.deepseek_v4_1.loader import V41Checkpoint, build_hasher, load_backbone
from src.models.deepseek_v4_1.generate import generate

front = load_backbone(cfg, V41Checkpoint(ckpt), hasher=hasher)   # hasher from build_hasher
result = generate(front, prompt_ids, max_new_tokens=8)           # greedy by default
```

Three properties are deliberate rather than incidental:

- **The prefix comes from the logits, not from the model's own sampling.** `Backbone.forward` samples with the config temperature, which the released schema never sets — so it is always the 1.0 default, a softmax and a Gumbel draw over the whole vocabulary each step. `generate` pins the model to greedy for the duration of the call and restores it afterwards, so that the draw neither wastes work nor consumes the global RNG a seeded caller expects to own.
- **The prompt is one forward, not a loop.** Its last row is the distribution the first new token comes from, so the first new token costs no extra pass. The two orderings are not bit-identical past the first position — fp32 reduction order across 40 layers — which the host record bisects, so a caller comparing against a reference has to pick one and say which.
- **Greedy is the default and is reproducible; `temperature > 0` is seeded and reproducible only for a fixed torch build.** Ties break to the lowest id.

There is also a command line, which is the runnable surface for the checkpoint:

```bash
python -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash \
  --prompt "The capital of France is" --max-new-tokens 8
```

It prints the generation to stdout and everything else — the load, the token count, the seconds per token — to stderr, so stdout stays pipeable. `--resident-engram` copies the two tables into RAM instead of gathering from the shards, `--expert-cache` bounds the per-layer expert window, `--expert-device cuda --expert-world 4` moves the routed experts off the host and onto the cards (opt-in, and it falls back to the host path with one line on stderr rather than failing), and `--quiet` drops the progress lines. The dense tree moves with `torchrun --nproc_per_node=4 -m src.cli.generate_v41`, which is one process per card and the same flags. The device path's cost is [its own page](../performance/deepseek_v4_1_flash_device_experts.md).

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

The deltas a V4.1 path had to add, and where each one stands now:

| Delta | Evidence | Status |
| --- | --- | --- |
| Shared CSA2 compression | `attn.compressor.*` on 4 layers instead of 41, plus new `kv_source_layers` / `index_source_layers` / `Full`/`Reindex`/`Reuse` mode logic | implemented, `attention.py` |
| Hierarchical sparse indexer | New `indexer.wk` + `indexer.k_norm` (absent in V4-Flash) and `candidate_source_layer` / `candidate_topk_blocks` / `candidate_block_size` | implemented, `attention.py` |
| Engram | 189.13 GiB of n-gram hash tables and a 99,092-entry compressed vocabulary | implemented on the host — the front end is `src/encoding/engram.py` and the rows are gathered by `CheckpointEngramTable`. There is no GPU consumer of the rows |
| Vision | 259 `vision.*` tensors, a 4-tensor aligner, and `ffn.gate.bias_vl` — a vision-conditioned routing bias present as F32 `[384]` on all 40 backbone layers and on all 3 MTP layers, and absent from V4-Flash | not implemented; the tensors are audited and unloaded, and the text path carries no image mask |
| Hash-routing removal | V4-Flash has `layers.{0,1,2}.ffn.gate.tid2eid`; V4.1 has no `tid2eid` tensor and no `n_hash_layers` key | handled |
| Hyper-connection head removal | V4-Flash has top-level `hc_head_{fn,base,scale}`; V4.1 has none, and the last MTP stage has none either | handled — see the `pre_mix` paragraph below |
| MTP head rename | V4-Flash `mtp.2.markov_head.{markov_w1,markov_w2}`; V4.1 `mtp.2.markov_head.{embed,head}` | handled by name in the inventory, but **no MTP forward runs**: `main_hidden` is collected and unused |
| Compressor APE removal | V4-Flash `attn.compressor.ape` on 41 layers; V4.1 has no `ape` tensor | handled |

Two V4.1 MTP asymmetries were worth recording before the loader was written, and the loader's tensor census reflects both: only `mtp.0` has `main_norm` and `main_proj` (F8_E4M3 `[5120, 15360]`, with 15,360 = 3 MTP layers × 5120), and only `mtp.2` has `markov_head.embed`, `markov_head.head` (both BF16 `[129280, 256]`), `confidence_head.proj` (BF16 `[1, 5376]`, i.e. 5120 + `dspark_markov_rank`) and `norm`. `mtp.1` has no module beyond attention, FFN and hyper-connections.

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

That totals 38 checks without an index and 39 with one. A total is not the same as a "pass" count, and the two came apart while the checkpoint was arriving: on the 20 shards downloaded at the first real run, 31 passed and 8 were undecided; on the 46 shards of the second run, 35 passed, 4 were undecided and none failed, because a check whose evidence is in a shard that has not landed has a third outcome rather than a pass. [Auditing the shards from arrival to complete](../performance/deepseek_v4_1_shard_audit.md) records both partial runs and, for each, the waiting checks with the shards each one waits on. All 4 had the same cause — two Engram shape claims, the table dtype and the tables' 1×32 row block, plus the gate and value projections on each Engram layer and the 32×32 block claim the two `engram.wkv` weights hold open — and all 4 now pass, because the two shards they waited on, `model-00047` and `model-00048` at 94.56 GiB each, are on disk. Everything else had already resolved as its shard arrived: the CSA2 partition, the per-layer tensor sets, the expert counts, the Engram placement, the vision block set, the `head`/`norm` shapes, the FP4 expert packing and the two DSpark head claims.

The config schema is checked in both directions against the released pair, not only against a fixture. `as_reference_dict()` on the nested file reproduces the flat file exactly — all 64 keys, every value equal — and on the flat file it round-trips to itself, so the alias table is lossy in neither direction; `differs_from()` reports exactly the 12 Transformers-only fields and nothing else, which is the statement that no field both files state disagrees. All 76 canonical fields (66 text, 10 vision) have an alias row, every row names a key the released files actually carry, and the 64 the flat file is expected to carry are exactly the 64 it has — no key unaccounted for and none named that is absent. `tests/test_models_deepseek_v4_1_config.py` holds all of that: 13 tests that need no checkpoint — a 64-key toy config in the flat layout and the same toy model written out in the nested layout by hand, asserted to read identically, plus the negative direction (a `candidate_source_layer` that is not an index source, a compress ratio with no source at or before it, two configs disagreeing on a field, and an unknown key, each reported or ignored rather than fatal) — and 8 more that skip unless a released pair is on the host. The toy nested file is built by hand rather than derived from the alias table on purpose: deriving it would make the round-trip test agree with whatever the table happens to say.

Two facts from this section deserve emphasis because they were derived rather than read off:

The Engram row counts are *derived*, and the derivation is exact to the row. The reference draws primes in order starting just above `engram_vocab_size` (16,000,000), hands them out across both layers without reuse, taking `(max_ngram_size − 1) × n_heads = 3 × 8 = 24` primes per layer, and `NgramHashState` builds its bucket offsets as a running sum over the layer's *flattened* prime list. The largest id a layer can produce is therefore `sum(primes) − 1` and the table needs exactly `sum(primes)` rows. Both layers match `engram_num_embeddings` with a difference of zero: the declared `[384006168, 384016682]` equals the derived values. That derivation is no longer only a claim on this page: `EngramLayout.from_config` reproduces it, `verify()` returns no problems, and `python -m src.encoding.engram --config …` prints a difference of 0 for both layers and exits 0. Layer 1's 24 primes run from 16,000,057 to 16,000,463 and layer 14's from 16,000,477 to 16,000,889; all 48 are distinct and all sit above `engram_vocab_size`, which is why one shared prime stream can serve both tables without overlap.

The CSA2 modes are derived from tensor presence, not from a config field. Diffing the three shard shapes against each other gives an exact per-mode tensor delta: a Reuse layer carries nothing extra; a Reindex layer adds `indexer.wq_b.weight`, `indexer.wq_b.scale` and `indexer.weights_proj.weight`; a Full layer adds those three plus `compressor.wkv.weight`, `compressor.norm.weight`, `compressor.wgate.weight`, `indexer.wk.weight` and `indexer.k_norm.weight`. So Full = owns `indexer.wk`, Reindex = owns `indexer.wq_b` only, Reuse = neither, which partitions the backbone as Full `[2, 8, 14, 20]`, Reindex `[24, 28, 32, 36]`, Reuse the other 32 layers.

`compressor.wgate` is the one tensor that separates layer 20 from the other three Full layers, and the reason is in the config value rather than in the tensor set: the reference constructs the gate only `if compress_ratio > 1`, because the gate is the learned softmax that pools `compress_ratio` consecutive tokens into one KV latent and a ratio of 1 has no group to pool. `compress_ratios` is `[0, 0]`, then 18 entries of 2, then 20 entries of 1, then `[0, 0, 0]` — so layer 20 is the single KV source at ratio 1. All four sources store `compressor.wkv` as BF16 `[512, 5120]` on disk; the reference declares that matrix FP32 for the ratio-2 layers and BF16 for ratio 1, so a loader reading the checkpoint has to upcast the three ratio-2 copies itself. (`compress_ratios[0:2]` and `compress_ratios[40:43]` are zero: layers 0 and 1, and all three MTP layers, read no compressed positions at all.)

Negative controls were run to confirm the checks are live rather than vacuous: perturbing a config key the audit does not consume leaves the result at 38/38 with exit 0, while setting `engram_n_heads = 7` drops it to 37/38 with exit 1, reporting `declared=[384006168, 384016682] derived=[336004849, 336012883]`.

The audit establishes nothing about numerics. No tensor value has been read, so no claim about quantization error, activation range or output parity is available.

The attention stack is the one V4.1 component that is executable, so it is held to properties rather than to text. `tests/test_models_deepseek_v4_1_attention.py` drives it on a six-layer toy geometry with two KV sources, three index sources and one candidate source, and asserts five things: that a twelve-token prefill and twelve one-token decode steps produce **bit-identical** output; that both orderings select exactly the reachable compressed positions, with `-1` filling the rest of the top-k width; that the shared slots are wired the way the modes say — index key caches only on the KV sources, a candidate mask written by the candidate source and read only by the index sources after it; that `reset_state` makes two forwards independent, which every other test in the file relies on; and that `select_candidate_blocks` keeps the block still being filled even when an older full block out-scores it, which is the rule in that function that no shape reveals. The parity assertion is equality and not a tolerance, on purpose: with no truncation the two orderings are the same arithmetic in a different grouping, so a tolerance would hide exactly the class of error the test exists to catch.

Getting to that equality took a measurement worth recording, because the first version of the test failed and the reason was not an arithmetic error. With `index_topk` smaller than the number of reachable compressed positions, prefill and decode diverge — and they diverge *only* through ties: over eight seeds and 168 checked query positions, 26 selections differed and every one of them was a position whose k-th and (k+1)-th index scores were equal, where prefill's wider score matrix and decode's narrower one break the tie differently. No position whose k-th score was distinct ever differed. Sweeping the width confirms the boundary is exactly where truncation stops: `index_topk` of 2, 4 and 8 give maximum output differences of 0.091, 0.036 and 0.020, and 12 — the full reachable width at that sequence length — and 16 are both exactly 0. So the toy config sets `index_topk` and `candidate_topk_blocks` wide enough that nothing is truncated, and that is a statement about the test, not about the model: a real V4.1 layer truncates at 512 of a possible 1M positions, and whether two orderings agree there is not decidable without a reference.

What that file does not establish is the thing this page keeps returning to. Its parameters are random, so it says the layer is internally consistent and says nothing about whether any of it matches the released model's tokens.

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
# the release as it downloads (20 of 48 shards at the first run): 31/39, 8 undecided, exit 0
python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash

# a later partial run (46 of 48 shards): 35/39, 4 undecided, exit 0
# -- the shard count is the only thing that changed
python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash

# the complete checkpoint: 39/39 passed, nothing undecided, exit 0
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

The attention stack is the one thing here a forward pass can be run through, and it needs neither a checkpoint nor a GPU. `torch` is the only requirement:

```bash
python -m pytest tests/test_models_deepseek_v4_1_attention.py -q
```

Expect 5 tests collected with nothing failing, in about two seconds, on a six-layer toy geometry small enough to read: `dim` 64, four heads of 32, a window of 4, ratios `(0, 0, 2, 2, 1, 1)`, KV sources `(2, 4)`, index sources `(2, 4, 5)` and a candidate source at layer 4. The same stack can be driven directly, which is how the parity numbers in [Correctness and precision](#correctness-and-precision) were measured:

```python
import torch
from src.models.deepseek_v4_1.attention import AttentionStack
from src.models.deepseek_v4_1.config import V41TextConfig

# the test file's toy geometry: every key the attention layers read has to be stated, because the
# module refuses to guess one (`_required`) rather than defaulting it
cfg = V41TextConfig(dim=64, n_layers=6, n_mtp_layers=0, n_heads=4, head_dim=32, rope_head_dim=8,
                    q_lora_rank=32, o_groups=2, o_lora_rank=16, window_size=4,
                    compress_ratios=(0, 0, 2, 2, 1, 1), kv_source_layers=(2, 4),
                    index_source_layers=(2, 4, 5), index_n_heads=2, index_head_dim=32, index_topk=16,
                    candidate_source_layer=4, candidate_topk_blocks=16, candidate_block_size=2,
                    norm_eps=1e-6, rope_theta=10000.0, compress_rope_theta=160000.0, rope_factor=40.0,
                    beta_fast=32, beta_slow=1, original_seq_len=512, max_position_embeddings=1024)
stack = AttentionStack(cfg, max_batch_size=1, max_seq_len=64)
x = torch.randn(1, 12, cfg.dim, dtype=torch.bfloat16)

stack.reset_state(1)
prefill = stack(x, 0)                                          # one chunk
stack.reset_state(1)
decode = torch.cat([stack(x[:, p:p + 1], p) for p in range(12)], dim=1)   # one token at a time
```

`index_topk` is 16 rather than the released 512 because the test needs the top-k to keep every reachable position; at 12 tokens the widest reachable set is 12, and truncating it is what makes the two orderings disagree.

`reset_state` is not optional there: without it the second pass continues the first, because the reference never resets and the module keeps that behaviour by default.

## Known limitations

- **Generation runs, on the host and with the experts on the cards — and nothing serves it.** `loader.load_backbone` fills the text backbone from the checkpoint and it answers: greedy decode returns `' Paris'` and then `'.'` for `"The capital of France is"` on every path, both with the routed experts on the CPU and with them split over the four cards (`--expert-device cuda --expert-world 4`) — the next token is a half-logit tie between EOS and `' The'` that the host's own prefill and stepwise decodes land on opposite sides of. What does not exist is a serving path, a batched path, or an MTP layer — the three DSpark draft layers are 7.39 GiB the loader deliberately leaves in the shards, so there is no speculative decoding here. The host path is CPU code over a memory mapping at 15 to 42 s per generated token; the device path is **1.06–1.14 s per step** with the dense tree on the host and **722–747 ms** with it cut across the four cards too, which is what `src/cli/generate_v41.py` under `torchrun --nproc_per_node=4` does — and both device figures are warm-page-cache numbers, since the expert staging reads the checkpoint mapping rather than a resident bank. `docs/performance/deepseek_v4_1_flash_host_run.md` and [its device companion](../performance/deepseek_v4_1_flash_device_experts.md) are where both are broken down.
- **The attention layers have no oracle, and now neither does the model.** The parity and causality properties in [Correctness and precision](#correctness-and-precision) say the layer is internally consistent, and the host run adds a text-level check — a coherent top-5 and a correct completion — but neither says the logits match the released model's, and the same TileLang/torch-2.10 wall that blocks the op layer blocks that comparison. One further gap is structural: `attention.py`'s own tests run at one rank, where the reference's cross-rank head split and score all-reduce are the identity, so those two paths are unexercised.
- **The checkpoint is complete and none of the 268.95 GiB of routed experts is resident on a card.** The release is 48 shards and 475.24 GiB, all 48 of them on disk and all 96,085 tensor descriptors readable; the audit reports 39/39 with nothing undecided. The routed experts and the two 189.13 GiB Engram tables are read from the mapping when a forward needs them, and `resident_engram=True` is the one way to stop re-reading the tables. **The experts cannot be treated the same way, and that is the model's central constraint**: a decode step routes to 4.20 GiB of them and misses about half of the rows in every layer, so on the host path they are expanded again every token — 240 experts at 0.122 s each on the CPU, 14.5 s warm and 29 s cold. The device path replaces that arithmetic with the fp4 kernel and moves the same 4.20 GiB over pinned PCIe in 0.30 s of host staging — a page-cache read, so it is 9.91 s on the same row once the cache is emptied — so [the measurements](../performance/deepseek_v4_1_flash_device_experts.md) make the device path the cheaper half of this model and not the more expensive one; what the four cards cannot do, holding 88 GiB between them, is keep the experts.
- **The two config layouts are not interchangeable field for field.** A consumer handed only `inference/config.json` has no `model_type`, no `architectures`, no token ids, no `topk_method`, no `norm_topk_prob` and no `param_dtype`, and its top-level `dtype` is the *quantization* dtype rather than the storage one. `resolve_config` prefers the nested file for exactly this reason and treats the flat one as the fallback; the schema makes the missing fields visibly absent rather than silently defaulted, but it cannot supply them.
- **The audit validates metadata consistency, not correctness.** It proves the config and the tensor inventory agree with each other. A checkpoint could satisfy all 38 checks and still be unusable, and a wrong value that is *consistently* wrong in both the config and the shapes would pass.
- **Engram is addressable, and now gathered, but not measured against the reference.** The compressed token map, the prime-derived bucket layout and the n-gram hasher exist in `src/encoding/engram.py`, the two layers reproduce their declared row counts exactly, and `CheckpointEngramTable` reads the rows a forward asks for — 24 per position, 456 per table over the 19 forwards the acceptance probe makes in one process, 12,288 per table on a 512-token prefill. What is still open is the gate and value projections' own parity with the reference, and the fact that gathering from the shards is not a viable steady state: [the host run](../performance/deepseek_v4_1_flash_host_run.md) measures a cold 512-token prefill's gathers at 253.4 s per table against 0.009 s once the table is resident.
- **The vision path is unvalidated in both directions.** The 263 vision and aligner tensors are accounted for and their shapes match the config, but no image has been processed and no projector has been run.
- **No reference comparison is possible on this host.** The released stack needs `torch>=2.10.0` and `tilelang==0.1.8`; neither is available in the `deepseek` environment. The op layer above is not an oracle either: six of its seven names are `src/kernels/ops.py`'s implementations, and the seventh is this repository's transcription of an E4M3 branch the reference states only as three lines of TileLang. Together with the attention stack above they make a V4.1 forward pass runnable and self-consistent — end to end, on the loaded checkpoint — and they cannot say whether any of it matches the released model's tokens. The acceptance evidence here is the generated text, not a logit comparison.
- **The prompt format changed and is unimplemented here.** DSML tags gain a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`), reasoning effort becomes a numeric budget in 1–100 rendered only under `thinking_mode="thinking"` at index 0, and mid-conversation `<｜System｜>` messages are supported. None of that is wired into this repository's chat templates.
- **Memory is the structural problem, not the kernels.** 58.0% of the checkpoint is routed experts and 39.8% is two Engram tables. A 4×22 GiB TP4 deployment cannot hold either, so any V4.1 plan has to place experts and Engram in host memory or on disk before kernel work matters.

## Evidence and related notes

- `src/models/deepseek_v4_1/loader.py` — the checkpoint reader: `V41Checkpoint` over the 48 shards, `load_backbone` filling `modules.Backbone`, `CheckpointRoutedExperts` and `CheckpointEngramTable` reading from the mapping on demand, the hash front end, and the `LoadReport` that says what was and was not filled
- `src/models/deepseek_v4_1/modules.py` — the backbone the config describes: embedding, hyper-connections, the MoE with its shared and routed halves, the Engram consumer, the norm and the head
- `tests/test_models_deepseek_v4_1_loader.py` — the tensor accounting, the groups a backbone leaves unread, and the quantized-name/scale pairing, over synthetic shards
- [DeepSeek-V4.1-Flash: what the released checkpoint costs to run on one host](../performance/deepseek_v4_1_flash_host_run.md) — the load report, the byte census, the Engram gather's cold and resident costs, the per-phase token table, and the PCIe floor, all measured on the host
- `src/models/deepseek_v4_1/config.py` — the config schema that reads both released layouts into one view, with an `as_reference_dict` inverse and a `python -m src.models.deepseek_v4_1.config` CLI
- `src/models/deepseek_v4_1/kernels.py` — the reference's six TileLang ops gathered behind one module, all six re-exported from `src/kernels/ops.py`, plus the E4M3 FP4 quantizer that module does not have
- `tests/test_models_deepseek_v4_1_kernels.py` — 12 tests pinning the parts those ops are easy to get subtly wrong (the finite score floor, the Sinkhorn order, the asymmetric gates, the round-to-even FP4 tie rule, the `e4m3(amax / 6)` scale); none compares against the reference, which cannot run here
- `src/models/deepseek_v4_1/attention.py` — the CSA2 attention half of a V4.1 backbone layer in pure PyTorch: the three layer modes, the four shared caches `SharedAttentionRuntime` carries down the stack, the two-level indexer, and the prefill/decode state machines `Compressor` and `Attention` need to be equivalent across
- `tests/test_models_deepseek_v4_1_attention.py` — 5 tests over a six-layer toy geometry: prefill and token-by-token decode bit-identical, the index selection causal and `-1`-padded, the shared slots wired to the modes, and `candidate` blocks pinned so the block still filling cannot lose to an older full one; that file's own docstring records why the top-k has to be untruncated for the parity claim to mean anything
- `scripts/audit_dsv41_headers.py` — the host-only header audit, reading its config through the schema above and distinguishing a shard that is missing from a shard that is wrong
- `tests/test_models_deepseek_v4_1_tensor_audit.py` — 16 tests over the audit's three outcomes, the last of which skips unless the released checkpoint is on the host
- [Auditing the shards from arrival to complete](../performance/deepseek_v4_1_shard_audit.md) — the audit runs and how their coverage grew with the download, the per-layer tensor partition, and the shapes the landed shards assert
- `src/encoding/engram.py` — the Engram compressed token map, bucket layout, hash multipliers and n-gram hasher, plus a `--config`/`--tokenizer` CLI
- `tests/test_models_deepseek_v4_1_config.py` — 21 tests over the schema: the alias table, the round-trip to the flat layout, the shape-independent reading of a toy model in both layouts, and the released pair when it is on the host
- `tests/test_encoding_engram.py` — 21 tests pinning the primes, multipliers and row counts, with the tokenizer and `sympy` legs skipping when unavailable
- `configs/config.json` — the validated V4-Flash config used as the delta baseline
- [DeepSeek-V4-Flash](deepseek-v4.md) — the validated runtime whose layer structure the V4.1 backbone reuses
- [Benchmark reporting rules](../guides/benchmarking.md) — required before any V4.1 number is quoted
- Model card and released config: `https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash`
