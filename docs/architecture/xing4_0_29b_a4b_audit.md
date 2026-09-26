# Xing4.0-29B-A4B: the checkpoint audit

Stage 2's first task ([#389](https://github.com/lvyufeng/PocketLLM/issues/389)) is a go/no-go, and the
thing it is a go/no-go *on* is one block. The [roadmap](pocketllm_new_model_roadmap.md) says why this
checkpoint is second: an official IQ4_NL GGUF at 18.7 GiB fits one 2080 Ti whole, so the smallest card
in the fleet gets a frontier-shaped architecture — MLA, a 64-expert MoE, a 262144-token context — and
the card stops being the constraint. It also says what the risk is:

> `hc_*` hyper-connections … is the one part of the architecture with no precedent here, and the
> stage's first task is a go/no-go on it — read out of the checkpoint's own remote code, not guessed.

This is that reading. It is a **pass**: the block is a matrix hyper-connection with a closed-form
forward pass, both released implementations of it agree operation for operation, and it is a port
rather than a research project. What follows is the description a reimplementation can be written
from, the artifact inventory the other four tasks start against, and four corrections to what the
issue tree assumed before the files were read.

Everything here is read out of an artifact — the released `modeling_xing4_0.py`, the official GGUF's
own header and tensor table, the safetensors index, and the open llama.cpp port — not out of a model
card. The last section names the read behind each claim.

---

## 1. What the block is

The reference is `Xing4_0HyperConnection` in
[`modeling_xing4_0.py`](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/modeling_xing4_0.py),
called twice per block by `Xing4_0DecoderLayer.forward`. Every model in this repository before now has
had **one** residual stream of width `hidden`. Xing4.0 has **four**, and the block's sublayers still
see one:

```
hidden_streams  : [tokens, 4, 3584]        the residual state
pre, post, comb : [tokens, 4], [tokens, 4], [tokens, 4, 4]      computed from it
collapsed       : [tokens, 3584]           = Σ_s pre[s] · hidden[s]
```

The sublayer — attention or FFN — runs on `collapsed`, and the four streams are re-formed from its
output:

```
hidden'[d] = post[d] · sublayer_out  +  Σ_s comb[d, s] · hidden[s]        for d in 0..3
```

So `post[d]` is how much of the sublayer's output stream `d` takes, and `comb` is how the old streams
are mixed into the new ones. Nothing here is shaped like the single-stream residual it replaces: the
coefficients are not a scalar, they are produced per token from the hidden state itself.

### The coefficients

One projection produces all 24 of them, from all four streams flattened together. With `hc = 4`:

| Step | What it is |
| --- | --- |
| `flat = rms_norm(hidden_streams.flatten(start_dim=2))` | **unweighted** RMSNorm — no learnable scale — over the *flattened* `4 × 3584 = 14336`, not over 3584 four times |
| `w = hc_fn · flat` | `[24, 14336] · [14336, tokens] → [24, tokens]` |
| `pre_w, post_w, comb_w = w.split([4, 4, 16])` | the 24 are **4 + 4 + 16**, not 4 streams × 6 |
| `pre = sigmoid(pre_w · hc_scale[0] + hc_base[0:4])` | range (0, 1) |
| `post = 2 · sigmoid(post_w · hc_scale[1] + hc_base[4:8])` | range (0, 2), so it can exceed 1 |
| `L = clamp(comb_w.view(4, 4) · hc_scale[2] + hc_base[8:24], -30, +30)` | the clamp is to ±30, and it is on the **logits** |
| `M = exp(L − max_over_src L)` | a softmax numerator, over `src` |
| `M ← M / (Σ_src M + ε)` then `M ← M / (Σ_dst M + ε)`, **20 times** | Sinkhorn; `ε = 1e-6` in every denominator |

`hc_scale` is three separate scalars — one per gate — and `hc_base` is three separate biases, read at
those offsets. They are part of the arithmetic, not tuning knobs: `hc_scale` is initialised to ones
and `hc_base` to zeros at training time, so a reimplementation that drops either diverges from the
released weights immediately.

`comb` ends up approximately doubly stochastic: after 20 rounds each row and each column sums to 1.
That is what makes four parallel streams stable rather than a four-fold gain, and it is why the
iteration count is a semantic value in the config rather than a performance knob.

### Where it sits in the block

```python
# Xing4_0DecoderLayer.forward
post, comb, collapsed = self.attn_hc(hidden_states)
collapsed = self.input_layernorm(collapsed)          # the *pre*-attention norm is inside the block
attn_out, _ = self.self_attn(collapsed)
hidden_states = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(comb, hidden_states)

post, comb, collapsed = self.ffn_hc(hidden_states)
collapsed = self.post_attention_layernorm(collapsed)
mlp_out = self.mlp(collapsed)
return post.unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb, hidden_states)
```

Two consequences worth stating, because both are places a port goes wrong quietly:

- **The residual is not a tensor you carry, it is a pair you rebuild.** `post` and `comb` are produced
  by the *same* call that produces the collapsed input, and they are the only things the sublayer
  output is combined with. An implementation that keeps a conventional `residual` accumulator and adds
  to it will produce plausible, wrong text.
- **The stream shape is not the sublayer shape.** Attention and the MoE take `[tokens, 3584]`; only the
  HC plumbing is `[tokens, 4, 3584]`. On this card that is the difference between a decode step that
  reads 17.8 GiB and one that reads 71 GiB.

At the head, `hidden_states.mean(dim=2)` collapses the four streams to one before the final norm —
**mean, not sum**, and there is no learned head-side gate (unlike DeepSeek-V4's `hc_head_fn`).

### The gate

`hc_attn_fn.weight` is not quantized to IQ4_NL in the released GGUF, and the reason is visible in the
llama.cpp port's quantizer: it is excluded by name as precision-sensitive, alongside the router. Its
`[24, 14336]` shape is small enough that this costs nothing — 240 `hc_*` tensors across 40 blocks are
52.51 MiB in total, 0.27% of the file. Our kernels may exploit that freely: the gate can be FP16 (it
is BF16 on disk) with FP32 accumulation for the sigmoid, the clamp and all 40 Sinkhorn normalizations
with no measurable memory cost.

---

## 2. The two released implementations agree

The block exists twice in public, from different authors and in different frameworks, and this section
is the cross-check. Both were read line by line; the second is
[PR #29012](https://github.com/ggml-org/llama.cpp/pull/29012), still open, which is where `xing4_0.cpp`
and the `XING4_0_HC_{PRE,COMB,POST}` ops come from.

| | `modeling_xing4_0.py` (Transformers) | `src/models/xing4_0.cpp` (llama.cpp) |
| --- | --- | --- |
| flatten | `flatten(start_dim=2)` → `[.,14336]`, stream-major | `reshape_2d(x, hc*n_embd, nt)`, same order |
| norm | `Xing4_0UnweightedRMSNorm(eps=rms_norm_eps)` | `ggml_rms_norm(flat, norm_rms_eps)` |
| split | `split([hc, hc, hc*hc])` | `view_1d`/`view_2d` at offsets `0`, `hc`, `2*hc` |
| pre | `sigmoid(w·scale[0] + base[0:4])` | `sigmoid(w·scale[0] + base[0:4])` |
| post | `2 · sigmoid(w·scale[1] + base[4:8])` | `2 · sigmoid(w·scale[1] + base[4:8])` |
| clamp | `clamp(·, -30, 30)` | `ggml_clamp(·, -30.0f, 30.0f)` |
| comb index | `matmul(comb, h)` reads `comb[dst, src]`; flat `p = dst·4 + src` | `view_2d` at `src·nb[0] + dst·nb[1]`; flat `p = dst·4 + src` |
| sinkhorn | 20 × (normalize over `src` with ε, then over `dst` with ε) | softmax over `src`, then `norm_dst`, then 19 × (`norm_src`, `norm_dst`) |
| collapse | `Σ_s pre[s]·h[s]` | `Σ_s x[s]·pre[s]` |
| head | `mean(dim=2)` | `xing4_0_hc_mean`, sum then scale `1/hc` |

The comb layout is the one that deserves the second look, because the flat 16-vector is reshaped
to a 4×4 matrix by two *different* mechanisms — `view(4,4)` in PyTorch, `ggml_reshape_3d(4,4,nt)` in
ggml — and a transposition between them would be invisible in every other row of this table. Both
resolve to the same element: **the flat index inside the 16 is `dst·4 + src`**, which is what makes
`matmul(comb, h)` in one and the `src`/`dst` offset arithmetic in the other the same computation.
(Only on the first normalization does the llama.cpp *decomposed* fallback differ — it uses
`ggml_soft_max`, which cannot add ε to its denominator, and its own comment quantifies that at ~1e-6.
The fused kernel it replaces matches the reference exactly. We implement the reference.)

The two disagree only on the dtype of the gate GEMM: Transformers runs it in the checkpoint's dtype
(BF16) then widens, llama.cpp accumulates against BF16 weights in F32. That is a tolerance, not a
semantic.

---

## 3. The artifact inventory

Read out of `xing4_0-29b-IQ4_NL.gguf`'s own header and tensor table.

### Metadata

The keys the runtime needs are all present, and two of them matter more than the rest.

| Key | Value | Note |
| --- | --- | --- |
| `general.architecture` | `xing4_0` | the dispatch string |
| `general.file_type` | `25` | `MOSTLY_IQ4_NL` |
| `xing4_0.block_count` | **41** | 40 trunk + 1 NextN; `nextn_predict_layers = 1` |
| `xing4_0.context_length` | `262144` | |
| `xing4_0.vocab_size` | `131072` | |
| `xing4_0.embedding_length` | `3584` | |
| `xing4_0.attention.head_count` | `32` | |
| `xing4_0.attention.head_count_kv` | `1` | MQA — the absorbed MLA form, not the raw one |
| `xing4_0.attention.key_length` | `576` | `kv_lora_rank 512 + qk_rope 64`, the cache width |
| `xing4_0.attention.key_length_mla` | `192` | `qk_nope 128 + qk_rope 64`, the score width |
| `xing4_0.attention.value_length` | `512` | `kv_lora_rank`, the absorbed value width |
| `xing4_0.attention.value_length_mla` | `128` | `v_head_dim` |
| `xing4_0.attention.q_lora_rank` | `768` | |
| `xing4_0.attention.kv_lora_rank` | `512` | |
| `xing4_0.rope.dimension_count` | `64` | |
| `xing4_0.rope.freq_base` | `10000.0` | |
| `xing4_0.rope.scaling.type` | `yarn` | |
| `xing4_0.rope.scaling.factor` | `64.0` | |
| `xing4_0.rope.scaling.original_context_length` | `4096` | |
| `xing4_0.rope.scaling.yarn_beta_fast` / `_slow` | `32.0` / `1.0` | |
| `xing4_0.rope.scaling.yarn_log_multiplier` | `0.1` | `0.1 × mscale_all_dim`, with `mscale_all_dim = 1.0` |
| `xing4_0.expert_count` | `64` | |
| `xing4_0.expert_used_count` | `4` | |
| `xing4_0.expert_shared_count` | `1` | |
| `xing4_0.expert_feed_forward_length` | `1024` | |
| `xing4_0.expert_gating_func` | `2` | sigmoid |
| `xing4_0.expert_weights_norm` / `_scale` | `True` / `2.0` | `norm_topk_prob`, `routed_scaling_factor` |
| `xing4_0.expert_group_count` / `_used_count` | `1` / `1` | no group-limited routing |
| `xing4_0.leading_dense_block_count` | `2` | |
| `xing4_0.hyper_connection.count` | `4` | |
| `xing4_0.hyper_connection.sinkhorn_iterations` | `20` | |
| `xing4_0.hyper_connection.epsilon` | `1e-6` | |
| `tokenizer.ggml.model` | `llama` | SentencePiece, not BPE |

The two that carry risk:

- **`key_length = 576` alongside `key_length_mla = 192`.** The runtime stores the *absorbed* cache —
  512 of compressed KV plus 64 of shared RoPE key — and scores against a 576-wide inner product, while
  the attention scale stays `192^-0.5`. That pairing is the DeepSeek MLA absorption this repository
  already implements for V4.1 and GLM-5.2, and reading only `key_length` would silently change the
  scale.
- **`yarn_log_multiplier = 0.1`.** This is the whole of the YaRN frequency re-base as far as the GGUF
  is concerned, and it is *not* `1.0`: the correct scaling is `mscale = yarn_get_mscale(factor,
  mscale_all_dim)` applied as a square to the logits and once to cos/sin, which needs
  `original_context_length` and `factor` as well. A runtime that reads `freq_base` and stops will
  produce a model that is fluent at short range and incoherent past ~4K.

### Tensors

977 tensors, 31.215 B parameters, 18.7204 GiB, **5.1516 bits per weight** overall.

| Type | Tensors | Parameters | Bytes | bpw | What it covers |
| --- | ---: | ---: | ---: | ---: | --- |
| `iq4_nl` | 243 | 29.074 B | 15.2309 GiB | 4.500 | every MoE expert, both shared experts per MoE layer, the two dense FFNs, the NextN block |
| `bf16` | 327 | 1.662 B | 3.0959 GiB | 16.000 | **all of attention**, `token_embd`, all 80 `hc_*_fn` |
| `f32` | 406 | 0.009 B | 0.0347 GiB | 32.000 | norms, the router, `exp_probs_b`, `hc_*_base`, `hc_*_scale` |
| `q6_k` | 1 | 0.470 B | 0.3589 GiB | 6.5625 | `output.weight` |

Two facts here change the plan rather than describe it:

- **The attention is BF16, not quantized.** llama.cpp's quantizer keeps `attn_k`, `attn_v`, `attn_qkv`
  and `output` off the low-bit path, and this file follows it: `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`,
  `attn_k_b`, `attn_v_b` and `attn_output` are all 16-bit, 2.117 GiB of the file. So the MLA path's
  weights arrive at full width and the only work is BF16 → FP16, which is **exact** for these values
  (BF16 has 8 mantissa bits and a wider exponent than FP16; every normal weight survives the
  narrowing unchanged). There is no IQ4_NL kernel to write for attention.
- **The 243 IQ4_NL tensors are 93.1% of the parameters and 81.4% of the bytes.** IQ4_NL is the whole
  of the model's quantized work, and it is a 4.5-bit format: 32 values in 18 bytes, one FP16 scale per
  32, and indices into a 16-entry non-linear codebook.

Per-tensor shapes, with the MLA names mapped onto the path this repository already runs. The naming is
not merely similar to GLM-5.2's — it is **identical**, because both GGUF conversions go through
llama.cpp's `DeepseekV2Model` converter:

| GGUF tensor | Shape (ggml) | Role | Already in `src/loader/mappings/glm_dsa.py` |
| --- | --- | --- | --- |
| `attn_q_a.weight` | `[3584, 768]` | down-project to the q LoRA rank | `attn_q_a` |
| `attn_q_a_norm.weight` | `[768]` | RMSNorm on the q latent | `attn_q_a_norm` |
| `attn_q_b.weight` | `[768, 6144]` | up-project to 32 heads × 192 | `attn_q_b` |
| `attn_kv_a_mqa.weight` | `[3584, 576]` | to compressed KV + shared RoPE key | `attn_kv_a` |
| `attn_kv_a_norm.weight` | `[512]` | RMSNorm on the compressed KV | `attn_kv_a_norm` |
| `attn_k_b.weight` | `[128, 512, 32]` | absorb: K up-project, folded into the score | `attn_k_b` |
| `attn_v_b.weight` | `[512, 128, 32]` | absorb: V up-project, applied after the weights | `attn_v_b` |
| `attn_output.weight` | `[4096, 3584]` | 32 × 128 → hidden | `attn_o` |
| `attn_norm.weight`, `ffn_norm.weight` | `[3584]` | the two block norms | `attn_norm`, `ffn_norm` |
| `hc_attn_fn.weight` | `[14336, 24]` | **new**: attention hyper-connection gate, BF16 | — |
| `hc_attn_base.weight` / `hc_attn_scale.weight` | `[24]` / `[3]` | **new**: its bias and per-gate scales | — |
| `hc_ffn_fn.weight` / `_base` / `_scale` | same | **new**: the FFN's, structurally identical | — |
| `ffn_gate_inp.weight` | `[3584, 64]` | router, F32 | `gate` |
| `exp_probs_b.bias` | `[64]` | `e_score_correction_bias`, for selection only | `gate_bias` |
| `ffn_gate_exps.weight` | `[3584, 1024, 64]` | routed w1, IQ4_NL | `routed_w1` |
| `ffn_up_exps.weight` | `[3584, 1024, 64]` | routed w3 | `routed_w3` |
| `ffn_down_exps.weight` | `[1024, 3584, 64]` | routed w2 | `routed_w2` |
| `ffn_{gate,up,down}_shexp.weight` | `[3584, 1024]` etc. | the single shared expert | `shared_w1/2/3` |
| `ffn_{gate,up,down}.weight` | `[3584, 9216]` etc. | the two leading dense blocks, IQ4_NL | `dense_w1/2/3` |
| `token_embd.weight` | `[3584, 131072]` | BF16 | `embed_tokens` |
| `output.weight` | `[3584, 131072]` | Q6_K | `lm_head` |
| `output_norm.weight` | `[3584]` | F32 | `final_norm` |

`blk.40.*` is the NextN/MTP block: 24 tensors, 0.9344 GiB, and no `hc_*` — the hyper-connection is
trunk-only, which the converter enforces. **The trunk is 953 tensors and 17.7860 GiB.** That is what
has to be resident, and what the card has to hold.

---

## 4. Where the issue tree's reading does not hold

The issue tree was written from the config and the GGUF header before the remote code was read. One of
this document's own readings is wrong too — the roadmap's Stage 2 table, since it inherited the first
item below — and the four are collected here rather than scattered through the sections above.

1. **"`qk_nope 192`, `qk_rope 128`"** — the halves are the other way round. `qk_nope_head_dim` is 128
   and `qk_rope_head_dim` is 64, summing to the 192 that the GGUF calls `key_length_mla`. The GGUF's
   `rope.dimension_count = 64` is therefore *equal* to `qk_rope_head_dim`, not half of it, and the
   question the issue raised — "confirm how they compose, since that is a place a port silently
   diverges" — has a simple answer: they are the same number, and there is no composition.
2. **"24 = 4 streams × 6 or 4 × (4 + 2)"** — neither. The split is `[hc, hc, hc·hc]` = `[4, 4, 16]`:
   one gate each for the collapse and the output, then the 16 entries of the mixing matrix. Six is not
   a number that appears.
3. **"`context_length = 0` and `vocab_size = 0` in the GGUF metadata — the converter dropped them"** —
   they were not dropped. The GGUF carries `context_length = 262144` and `vocab_size = 131072`, and
   the tokenizer arrays with them. Nothing needs a fallback to the config.
4. **"the Sinkhorn iteration runs 20 times per call … fold it into one kernel rather than 20
   dispatches"** — right conclusion, and the count is confirmed, but the cost estimate behind it is
   off by more than the note implies. A call is not one Sinkhorn: 20 per gate × 2 gates × 40 blocks is
   **1600 iterations per token**, on sixteen-element matrices. At decode that is 1600 serial
   dependent steps inside one kernel launch. It is not a dispatch problem to be optimized later; it is
   the block's decode shape, and it sets the kernel's design.

One claim in the issue tree is *confirmed* and worth repeating because it decides the whole stage:
`ep_size = 1`, and there is no expert-parallel key in the GGUF metadata. There is no sharding the
runtime has to respect — the checkpoint is what a single card runs.

---

## 5. Go/no-go, and what each remaining task now costs

**Go.** The reason is not that the block is small; it is that the block is *decidable*. Every step
above is a pointwise or small-matrix operation with no data-dependent control flow, no search, and no
learned state beyond three tensors per gate — and it exists twice in public from different authors,
which is the strongest correctness evidence available for an architecture this new. The stage is a
port with one genuinely new kernel, and that kernel is a 4×4 iteration, not a new attention family.

What the four remaining tasks inherit:

| Task | What the audit changes about it |
| --- | --- |
| [#390](https://github.com/lvyufeng/PocketLLM/issues/390) IQ4_NL | Smaller than it looked. `GGML_TYPES[20]` already carries `iq4_nl`'s geometry (32 values, 18 bytes) and `reader.py` already addresses its bytes; what is missing is the entry in the runtime dispatch table `GGUF_DENSE_TYPE_IDS`, the 16-entry codebook, and a decode path. The format is upstream llama.cpp, not fork-private like Bonsai's `PTQ1_0`, so the reference is `kvalues_iq4nl` in `ggml-common.h`. |
| [#391](https://github.com/lvyufeng/PocketLLM/issues/391) MLA | The geometry is V4.1's and GLM-5.2's, and so is the vocabulary. The new work is the **YaRN** re-base and reading two rank pairs instead of one. `key_length_mla 192` against a 576-wide absorbed score is the trap the audit names. |
| [#392](https://github.com/lvyufeng/PocketLLM/issues/392) the HC block | Scoped by §1: one gate GEMM per sublayer, then a fused pre/post/comb kernel whose decode shape is 1600 serial 4×4 Sinkhorn iterations per token and whose prefill shape is one 4×4 per (token, sublayer). Prefill and decode are different kernels, as the issue anticipated. |
| [#393](https://github.com/lvyufeng/PocketLLM/issues/393) MoE, serving, docs | Unchanged, with one addition the audit supplies: the routed reduction is over top-4, so it takes the deterministic ordering this repository already settled on for top-k ≥ 3. |

And what the stage is worth, as arithmetic rather than as a promise. One decode step reads the active
set once:

| Per token | GiB | Share |
| --- | ---: | ---: |
| BF16 attention (all 40 blocks) | 2.117 | 56.3% |
| Routed experts, 4 of 64 selected | 0.877 | 23.3% |
| Q6_K `output.weight` | 0.359 | 9.5% |
| Shared expert | 0.219 | 5.8% |
| Dense FFN (2 blocks) | 0.104 | 2.8% |
| `hc_*`, router, norms, embedding | 0.086 | 2.3% |
| **Total** | **3.761** | |

On a 2080 Ti's 616 GB/s that is **6.56 ms, a 152.5 tok/s ceiling**, or ~130 tok/s at a realistic 85% of
peak. For scale, the fastest decode this repository has measured is Qwen3.8-27B-FP8 at 43.22 tok/s.

The shape of that table is the finding, not the total. **Attention is the largest single term and it is
unquantized** — 2.12 GiB against the routed experts' 0.88 GiB, because this GGUF leaves every attention
projection at BF16 while quantizing 93% of the checkpoint's parameters. On a bandwidth-bound decode
step the MLA path is therefore the thing to optimize, which is the opposite of the expectation the
roadmap states ("the MoE is [the bottleneck]") and of what MiMo-V2.6 and V4.1 both measured. A card
that holds 17.79 GiB of 18.72 accordingly has 4.2 GiB left for KV: about 97,000 tokens at 576 values per
layer per token at FP16, or the full 262144 with a quantized cache.

Those are bounds from byte counts, not measurements — the last task is where they become numbers, and
this table is a prediction the stage should be judged against.

---

## 6. What this audit did not do

- **No weights have been run.** Nothing here is a parity result. The gate GEMM, the Sinkhorn and the
  MLA path are described from the reference arithmetic and from two agreeing implementations; whether
  our kernels reproduce them is #391's and #392's acceptance, not this task's.
- **The NextN/MTP block is not read past its shape.** 24 tensors, 0.9344 GiB, no `hc_*`, and it is
  excluded from the trunk. This repository's record on speculative decoding is that it is
  acceptance-dependent and that a length-2 MTP path on a GGUF cost more than it returned, so the
  block is loaded or skipped by measurement later rather than assumed into the plan.
- **The chat template is not evaluated.** It is present, it is 3.7 KB of Jinja, and it has
  `visible_text`/tool-call macros. Whether it renders correctly through this repository's server is a
  serving-time question.
- **No quality claim.** The authors' benchmark table is theirs.

## Evidence

Every fact above is a read, and this is the read:

- **GGUF header, metadata and 977-tensor table** — `XingChen-AGI/Xing4.0-29B-A4B-GGUF`,
  `xing4_0-29b-IQ4_NL.gguf`, 2026-09-20, via this repository's own `GGUFReader`, which is also what
  produced the type histogram, the byte totals and the shapes.
- **`hc_*` semantics** — `modeling_xing4_0.py` in `XingChen-AGI/Xing4.0-29B-A4B`, classes
  `Xing4_0HyperConnection`, `Xing4_0UnweightedRMSNorm` and `Xing4_0DecoderLayer`; the config's
  `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `mhc_h_res_clamp_min/max`, `rope_interleave`.
- **The cross-check** — [ggml-org/llama.cpp PR #29012](https://github.com/ggml-org/llama.cpp/pull/29012)
  (`src/models/xing4_0.cpp`, `ggml/include/ggml.h`, `src/llama-quant.cpp`, `conversion/xing.py`), open
  as of 2026-09-26, and its `llm_arch`/tensor-type registration.
- **The IQ4_NL block** — upstream llama.cpp `ggml/src/ggml-common.h` (`QK4_NL 32`, `block_iq4_nl`,
  `kvalues_iq4nl`) and `ggml/src/ggml-cuda/dequantize.cuh` (`dequantize_iq4_nl`), read from `master`.
- **The safetensors side** — `model.safetensors.index.json` (8307 tensors, one layer per shard, layer
  40 alone in shard 41) and the shard-1 header, which confirms `hc_fn` as BF16 `[24, 14336]` against
  the GGUF's `(14336, 24)`.
