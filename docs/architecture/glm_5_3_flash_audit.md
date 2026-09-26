# GLM-5.3-Flash: the checkpoint audit

Stage 3's first task ([#395](https://github.com/lvyufeng/PocketLLM/issues/395)) is a go/no-go, and
the thing it is a go/no-go *on* is the `linear_attention` layer. The
[roadmap](pocketllm_new_model_roadmap.md) says why this checkpoint is last: its 45 layers alternate
**three `linear_attention` to one `deepseek_sparse_attention`**, and both halves already exist in
this repository — Qwen3.8-27B's Gated DeltaNet, GLM-5.2's DSA — just never in the same trunk. It
also names the risk:

> Everything else is familiar … *if* the linear layer is GDN. The issue tree is written on that
> assumption and task 1 exists to test it.

The assumption does not hold. **The linear layer is not GDN.** It is Kimi Delta Attention (KDA):
same recurrence, four different pieces of gate arithmetic, three gate projections where GDN has
two, and a weight layout neither of this repository's linear-attention paths can read. A port of
`GatedDeltaNet` would run and produce fluent, wrong text.

Two things did come back better than the issue tree expected, and they are the reason this is still
a **go**:

- **The hyper-connection half is not a risk at all.** GLM-5.3's mHC is *bit-for-bit* the mHC
  [DeepSeek-V4.1](deepseek_v4_1_flash_design.md) already runs, including the transposed
  `matmul(comb.T, residual)` — which is the opposite of Xing4.0's. `hc_split_sinkhorn` in
  `src/kernels/ops.py` is reusable verbatim.
- **Three of the five indexer knobs the task asks about do not exist in the released config.**
  `index_topk_pattern`, `index_topk_freq` and `index_skip_topk_offset` are absent, `indexer_types`
  is uniformly `'full'`, and the machinery they were assumed to drive is therefore dormant. What
  *is* new on the sparse side is a k-pool compressor the 5.2 path does not have.

What follows is the description a reimplementation can be written from, the artifact inventory the
other four tasks start against, and the corrections to the issue tree that the files force.

Everything here is read out of an artifact — the released `config.json`, the reference
implementation shipped inside `transformers` (v5.17.0, `models/glm5_next/`, so no
`trust_remote_code` is needed), the official GGUF's own header and tensor table, `unsloth/GLM-5.2-GGUF`'s
header for the diff against 5.2, and this repository's own linear-attention and mHC code. The last
section names the read behind each claim.

---

## 1. The 45 layers

`config.json` carries three parallel per-layer lists, and they are not the same list.

| # | attention | MLP | maps onto |
| ---: | --- | --- | --- |
| 0 | `linear_attention` | dense | **new** — KDA + `ffn_{gate,up,down}` |
| 1 | `linear_attention` | dense | **new** |
| 2 | `linear_attention` | dense | **new** |
| 3 | `deepseek_sparse_attention` | sparse | GLM-5.2 DSA + MoE, with a reworked indexer (§4) |
| 4–6 | `linear_attention` | sparse | **new** ×3 |
| 7 | `deepseek_sparse_attention` | sparse | as 3 |
| … | three linear, one sparse | | the LLLS pattern repeats to layer 43 |
| 43 | `deepseek_sparse_attention` | sparse | as 3 |
| 44 | `linear_attention` | sparse | **new** |
| 45 | `deepseek_sparse_attention` | sparse | **MTP / NextN**, plus the four `nextn.*` tensors |

Read out of the config's own lists rather than off the pattern: `linear_attn_config.kda_layers` is
`[0,1,2,4,5,6,8,…,44]` — 34 entries — and `full_attn_layers` is `[3,7,11,…,43]` — 11. Both match
`layer_types`. `mlp_layer_types` is dense for 0,1,2 and sparse for 3 onwards, which is
`first_k_dense_replace 3` written out.

So **34 KDA + dense-or-MoE**, **11 DSA + MoE**, and a 46th block that is the MTP head. The GGUF's
`head_count_kv` says the same thing a second way, one entry per block:

```
[0, 0, 0, 1, 0, 0, 0, 1, …, 0, 1]
  ^^^^^^^^^^^^                  ^  ^
  linear: no KV cache at all     sparse   NextN
```

— a **zero** for every linear layer, which is the loader's cheapest discriminator and worth keying
off rather than off `layer_types`, since it survives a config the loader never sees.

The two halves share the block, the two RMSNorms and both mHC sites. They differ in everything
below the first norm: what the sublayer projects, what it keeps as state, and what it costs.

---

## 2. The linear layer, written out

`Glm5NextTextLinearAttention` at `hidden_size 4096`, `num_heads 64`, `head_dim 128`,
`conv_kernel_size 4`:

```
x                                    [tokens, 4096]
q, k, v = q_proj(x), k_proj(x), v_proj(x)         three [4096, 8192] projections, no bias
mixed   = cat([q, k, v], dim=-1)                  [tokens, 24576]
mixed   = causal_conv1d(mixed, kernel=4, groups=24576)
mixed   = silu(mixed)[:, -seq_len:]               the conv is depthwise; see below
q, k, v = split(mixed, [8192]*3) -> [tokens, 64, 128] each

f    = f_b_proj(f_a_proj(x))                      4096 -> 128 -> 8192
g    = (-5.0 * sigmoid(exp(A_log) * (f + dt_bias))).view(tokens, 64, 128)
beta = sigmoid(b_proj(x))                         [tokens, 64], one scalar a head

out, state = kimi_delta_attention(q, k, v, g, beta, state)     §3
z    = g_b_proj(g_a_proj(x)).view(tokens, 64, 128)             the output gate
out  = RMSNormGated(out, z, eps=1e-5, activation="sigmoid")    over head_dim 128
out  = o_proj(out)                                [8192, 4096]
```

**The conv splits cleanly into three.** The reference builds one `nn.Conv1d` with
`in_channels = out_channels = 24576` and `groups = 24576`, over the *concatenation* `[q, k, v]` in
that order. A depthwise conv's output channel `c` reads only input channel `c`, so three separate
8192-channel depthwise convs over `q`, `k` and `v` are the same arithmetic *given that
concatenation order* — which is why the GGUF can store them as `ssm_conv1d_q/k/v` and still be a
lossless reshape. Getting the order wrong (or concatenating after the conv) is silent.

**The gate is a low-rank pair, and the GGUF names it `ssm_`.**

| reference | GGUF | file shape | loader shape | what it is |
| --- | --- | --- | --- | --- |
| `q_proj` / `k_proj` / `v_proj` | `attn_q` / `attn_k` / `attn_v` | `[4096, 8192]` | `[8192, 4096]` | the three input projections |
| `conv1d.weight` | `ssm_conv1d_{q,k,v}` | `[4, 1, 8192]` | same | the depthwise conv, split in three |
| `forget_gate.f_a_proj` | `ssm_f_a` | `[4096, 128]` | `[128, 4096]` | forget gate, down |
| `forget_gate.f_b_proj` | `ssm_f_b` | `[128, 8192]` | `[8192, 128]` | forget gate, up |
| `forget_gate.dt_bias` | `ssm_dt.bias` | `[8192]` | `[8192]` | forget gate bias |
| `forget_gate.A_log` | `ssm_a` | `[64]` | `[64]` | **one scalar a head** |
| `b_proj` | `ssm_beta` | `[4096, 64]` | `[64, 4096]` | beta, **one scalar a head** |
| `g_a_proj` / `g_b_proj` | `ssm_g_a` / `ssm_g_b` | `[4096, 128]`, `[128, 8192]` | reversed | the output gate, low-rank |
| `o_norm` | `ssm_norm` | `[128]` | `[128]` | the gated norm, over `head_dim` |
| `o_proj` | `attn_output` | `[8192, 4096]` | `[4096, 8192]` | back to hidden |

Two of those names mislead if read from the 5.2 side: `ssm_a` is `A_log` and `ssm_beta` is the
beta projection, and neither is the `ssm_a`/`ssm_beta` of a Mamba-style SSM. The `ssm_` prefix is
carried over from the Kimi family the layer comes from.

`A_log` and `beta` being per-**head** while the forget gate `g` is per-`(head, dim)` is the shape
that matters: the decay rate is shared across a head's 128 channels, so 64 scalars control the rate
at which a head forgets and 128 numbers control only the per-channel input to it.

---

## 3. KDA against this repository's GDN

`src/models/qwen4_exp/attention.py` implements Gated DeltaNet for Qwen3.8-27B. Its
`recurrent_gated_delta_rule` and `chunk_gated_delta_rule` are **the same recurrence** as
`recurrent_kimi_delta_attention` and `chunk_kimi_delta_attention`:

```
S        = S * exp(g_i)                                  decay
kv_mem   = (S * k_i).sum(-2)
delta    = (v_i - kv_mem) * beta_i
S        = S + k_i ⊗ delta                               rank-1 delta write
out_i    = (S * q_i).sum(-2)
```

So the **scan kernel is a port, not a new kernel** — `chunk_size 64`, fp32 state, `scale =
head_dim**-0.5` after l2norm, all shared. Six things around it are not, and each is a way a
copy-paste comes out wrong:

| | Qwen3.8-27B GDN | GLM-5.3-Flash KDA | why it matters |
| --- | --- | --- | --- |
| **decay** | `g = -exp(A_log) · softplus(a + dt_bias)` | `g = -5.0 · sigmoid(exp(A_log) · (f + dt_bias))` | GDN's is unbounded below; KDA's is **bounded to `(-5, 0)`**, so `exp(g) ≥ e⁻⁵ ≈ 0.0067` and a state can never decay faster than that per step. Not a rescaling — a different function, with `kda.gate_lower_bound = -5.0` as its own metadata key. |
| **beta width** | `in_proj_b` → per `(head, dim)` | `b_proj` → **per head**, `[4096, 64]` | KDA broadcasts `beta[..., None]` inside the rule. A GDN port carries the wrong `beta` shape into the scan without erroring, because the rule broadcasts. |
| **output gate** | `in_proj_z`, full rank `[4096, v_dim]` | `g_a_proj`/`g_b_proj`, `4096→128→8192` | two tensors where GDN has one, and one of the three cases where reading `ssm_g_*` as an SSM parameter goes nowhere. |
| **qk l2norm** | in the **model dtype**, then cast to fp32 | cast to fp32 **first**, then l2norm | the reference says so in a comment ("FLA calculates these in fp32 so we do this after the float casts"). Same formula, different order, different rounding. |
| **conv** | one depthwise conv over the fused qkv, `ssm_conv1d` as one tensor | same conv, stored as `ssm_conv1d_q/k/v` | arithmetic identical (§2); the *tensor layout* is not, and neither loader reads the other's. |
| **norm activation** | `activation=config.output_gate_type or hidden_act` | hardcoded `"sigmoid"` | a config-driven activation on one side and a constant on the other. |

The `l2norm` itself is the one place the two agree more closely than expected: both use
`x / sqrt(sum(x²) + ε)` with ε *inside* the sum, not `max(‖x‖, ε)`. This repository's version is
written `x * rsqrt(…)` rather than the reference's `x / sqrt(…)`, which differs only in rounding.

**The verdict for task 2 is therefore "new wrapper, reused scan".** `src/models/qwen4_exp/` is the
right place to read the recurrence from and the wrong place to call: the projections, the gates,
the norm and the cache shape are all GLM-5.3's.

---

## 4. The sparse layer, and what changed from GLM-5.2

The DSA half is GLM-5.2's MLA — compressed KV in `kv_lora_rank 512`, `q_lora_rank 1536`, 64 heads,
`qk_nope 256`, `v_head_dim 256`, `scaling = 256**-0.5` — with **no RoPE anywhere on the qk path**
(§5), plus an indexer. GLM-5.2's loader contract in `src/models/glm_dsa/spec.py` names five indexer
tensors; GLM-5.3 has those five and two more:

| tensor | 5.2 | 5.3 file shape | what it is |
| --- | :-: | --- | --- |
| `indexer.attn_q_b.weight` | ✓ | `[1536, 4096]` | 32 index heads × 128, **from the same `q_a` latent** |
| `indexer.attn_k.weight` | ✓ | `[4096, 128]` | one index key, 128 |
| `indexer.k_norm.{weight,bias}` | ✓ | `[128]` | a `LayerNorm`, not an RMSNorm |
| `indexer.proj.weight` | ✓ | `[4096, 32]` | per-head score weights, F32 |
| `indexer_compressor_gate.weight` | — | `[4096, 128]` | **new**: pool gate logits |
| `indexer_compressor_ape.weight` | — | `[128, 4]` | **new**: a per-pool-slot embedding on those logits |

And the *metadata* the two releases carry is where the task's question is answered:

| key | GLM-5.2 | GLM-5.3 |
| --- | --- | --- |
| `*.attention.indexer.head_count` | 32 | 32 |
| `*.attention.indexer.key_length` | 128 | 128 |
| `*.attention.indexer.top_k` | 2048 | 2048 |
| `*.attention.indexer.kpool` | — | 4 |
| `*.attention.indexer.kpool_always_select_tail` | — | true |
| `*.rope.dimension_count` | 64 | **0** |

So **the indexer's geometry is unchanged** — 32 heads, 128 wide, top-2048 — and what changed is
*what a candidate is*. GLM-5.2 scores tokens; GLM-5.3 groups the key cache into pools of 4,
compresses each complete pool to one 128-wide key by a softmax-weighted average over the pool
(`softmax(gate_scores + ape)`, masked to valid members), scores **pools**, takes the top
`2048 / 4 = 512` of them, and expands back to 2048 raw token indices. The incomplete tail pool is
appended as raw tokens, which is what makes the selection width `index_topk + index_kpool - 1 =
2051` rather than 2048 — the reference's own docstring says `2*topk - 1` and is stale; the code
adds `index_kpool - 1`.

Three details in that are worth naming because each is a way a reimplementation is wrong without
erroring:

- **Pools are anchored to the first *valid* token, not to cache slot 0.** `get_pooled_states`
  computes `first_key = valid_keys.argmax(-1)` and offsets from it, so a cache whose head is
  padding pools `[A,B,C,D]` from `A` rather than pooling `[P,P,A,B]`. With prefix reuse the head is
  a resume point, and anchoring at slot 0 silently shifts every pool boundary by up to 3.
- **A pool is a candidate only if its *last* token is visible.** The mask is
  `visible_tokens.gather(pool_end) & pool_valid`, not "any member visible" — so a pool straddling
  the causality edge is excluded rather than partly scored. `pool_valid` separately requires all
  four members to exist, which is why the tail needs its own path.
- **The pooled average runs in the key dtype.** `probabilities` is cast back with
  `.to(grouped_keys.dtype)` *before* `(probabilities * grouped_keys).sum(dim=2)`, so the
  compression is bf16 arithmetic even though the softmax is fp32. A parity claim measured against
  an fp32 average will not close.

That is a strictly smaller problem to *select* over (512 candidates instead of 2048, four times
cheaper scoring) at the same retrieval width, and the compressor is 4096×128 + 128×4 of extra
weights — 0.5 MiB a layer, 6 MiB across the 11. This is the one place the new release buys something
the repository can measure, and task 3 is where it should be measured.

**On the five knobs the task names:** `index_topk_pattern`, `index_topk_freq` and
`index_skip_topk_offset` **are not in the released `config.json`**. `index_share_for_mtp_iteration`
is present but `True`, and it scopes to the MTP head. `indexer_types` **is** present and is
`'full'` for all 45 layers. That last one is the load-bearing one: `Glm5NextTextAttention.__init__`
computes `skip_topk = indexer_types[layer_idx] == "shared"`, and `next_skip_topk` from the *next*
layer's entry — with every entry `'full'` there is no shared layer, no layer is constructed without
its own `Indexer`, and the `prev_topk_indices` plumbing is dead code for this release. **Nothing
here has to be implemented**, and a `'shared'` layer would in any case come from a future config,
not from this checkpoint.

---

## 5. The four places a copy-paste from 5.2 goes wrong quietly

**No RoPE on the qk path.** `qk_rope_head_dim 0`, `qk_nope_head_dim 256`, `head_dim 0`,
`mla_use_nope True`, and the GGUF's `rope.dimension_count 0` against 5.2's 64. The consequences
run further than "delete the RoPE call":

- `kv_a_proj_with_mqa` is `[4096, 512]`, not `[4096, 576]` — 5.2's extra 64 is the shared RoPE key.
  A 5.2-shaped loader asking for `kv_lora_rank + rope_dim` reads 512 into a 576-wide buffer.
- `scaling = qk_head_dim ** -0.5` with `qk_head_dim = 256 + 0 = 256`, and **no YaRN mscale factor**
  is applied. GLM-5.2's config carries `rope_scaling`, and this repository's Xing4.0 path applies a
  squared YaRN mscale; carrying either over would scale every score.
- **The indexer's keys are not rotated, because there is nothing to rotate them into.** The qk path
  has no rotary dimension at all, so `indexer_rope_interleave` — which is present in the released
  config — is read by neither the reference's modeling file nor its configuration class. It is a
  key with no consumer.

**The sparse layers' `attn_output` is twice as wide as the linear layers'.** `[16384, 4096]` against
`[8192, 4096]`: 64 heads × 256 for MLA against 64 × 128 for KDA. The two `attn_output`s are not the
same tensor under two names, and a loader keyed on the suffix alone gets a shape error *or*, in a
reshaping path, silence.

**Every SwiGLU is clamped, asymmetrically.** `swiglu_limit 10.0` reaches four separate MLP classes —
the dense one, the routed expert, the shared expert and the NextN block's — and in each of them the
clamp is applied to the two projections *before* the product:

```python
gate = gate.clamp(min=None, max=swiglu_limit)      # +10 only
up   = up.clamp(min=-swiglu_limit, max=swiglu_limit)   # ±10
out  = silu(gate) * up
```

The GGUF agrees, and more loudly than the config does: it carries `swiglu_clamp_exp` **and**
`swiglu_clamp_shexp` as **46-element arrays**, one per block, all `10.0`. 46 is 45 + the NextN
block, so the clamp is present on the MTP path too. The lower bound on `gate` being *absent* is the
part a fused `silu(gate) * up` loses: there is no clamp on the pre-activation at all, so a kernel
that has no clamp is right for the gate and wrong for `up`.

**The indexer's `k_norm` is a LayerNorm, not an RMSNorm, and it has a bias.** This repository's DSA
indexer — `src/models/deepseek_v4_1/attention.py:816`, the one a `glm_dsa` port would be modelled on
— builds `RMSNorm(index_head_dim, norm_eps)`. The reference builds
`nn.LayerNorm(head_dim, eps=1e-6)`: it subtracts the mean, it has a bias, and its epsilon is
hardcoded rather than read from `config.rms_norm_eps`. The GGUF is the tell, and it is easy to read
past — `indexer.k_norm.bias` and `indexer.k_norm.weight` are both present as `[128]` F32, the same
`1e-6` that the header's separate `attention.layer_norm_epsilon` carries against the trunk's
`layer_norm_rms_epsilon 1e-5`. A loader keyed on the `.weight` suffix alone loads half the norm and
raises nothing.

---

## 6. Where the two halves meet

The trunk is one stack with two `attn` implementations and two `mlp` implementations, and the
things that make it one stack are the shared parts:

```
Glm5NextTextDecoderLayer.forward
    residual = hidden_states
    post, comb, hidden_states = attn_hc(hidden_states)     # collapse
    hidden_states = input_layernorm(hidden_states)
    hidden_states = linear_attention(...) | sparse_attention(...)   # <- the only fork
    hidden_states = post ⊙ out + matmul(combᵀ, residual)   # expand
    -- then identically for ffn_hc / post_attention_layernorm / mlp
```

Both sublayers are `[tokens, 4096] -> [tokens, 4096]`, both take the same norm (RMSNorm, eps 1e-5),
both are wrapped by the same mHC pair, and neither is the residual — the residual is rebuilt from
`comb` and the previous streams, so a `GatedDeltaNet` port that returns `x + out` corrupts the
streams from the first layer.

**mHC is already in this repository, exactly.** `Glm5NextTextHyperConnection.forward` is
line-for-line `hc_split_sinkhorn_torch` (`src/kernels/ops.py`):

```python
pre  = sigmoid(w * scale[0] + base[:hc]) + eps
post = 2 * sigmoid(w * scale[1] + base[hc:2hc])
comb = softmax(w * scale[2] + base[2hc:], dim=-1) + eps
comb = comb / (comb.sum(-2) + eps)
for _ in range(iters - 1):
    comb = comb / (comb.sum(-1) + eps)
    comb = comb / (comb.sum(-2) + eps)
```

— the same `+eps` after the sigmoid, the same `softmax` **then** `eps` (not `exp(logits − max)`, and
no clamp), the same **column-first** normalization, and the same `hc_mult 4`, `hc_sinkhorn_iters 20`,
`hc_eps 1e-6` that DeepSeek-V4.1's `modules.py` uses. The block applies it the same way, too:
DeepSeek-V4.1's `hc_post` is `torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`, which
evaluates to `combᵀ @ residual` — the same transposed form GLM-5.3's decoder layer writes literally.

The only deviations are outside the kernel:

- **Scheduling.** DeepSeek-V4.1 pipelines (`hc_mixes` produces the mix the *next* sublayer consumes;
  attention uses the previous layer's FFN's `pre_mix`), GLM-5.3 does not (one call returns `post`,
  `comb` and `collapsed`, and all three are used in the same sublayer). Wiring, not arithmetic.
- **Storage.** DeepSeek-V4.1 keeps `hc_{attn,ffn}_fn` in fp32 from a safetensors checkpoint; the
  GGUF stores `hc_*_fn` at **Q8_0** in the same `[mix, hc·hidden]` orientation Xing4.0's
  `from_gguf` already reverses — the one tensor in this block that has to be dequantized rather than
  read as a float.
- **The 46th block has no mHC.** `hc_*` is 45 tensors, not 46: the NextN block carries its own
  attention, MoE and norms, but no hyper-connection pair.

This is the second half of the task's question, and the answer is that **one of the two halves is
already shipped**. The risk is concentrated entirely in KDA.

---

## 7. What the issue tree assumed that the artifact does not support

**a. "The linear layer, if GDN, and the issue tree is written on that assumption."** It is not GDN
(§3). `GatedDeltaNet` cannot be reused as a wrapper; the scan inside it can.

**b. "`index_topk_pattern` sounds like the indexer is not run identically on every sparse layer."**
The key does not exist in this release, and the list that *does* — `indexer_types` — is uniformly
`'full'` (§4). The indexer is run identically on every sparse layer, including the MTP block. The
cross-layer sharing machinery is real code in the reference and unreachable from this checkpoint.

**c. "Confirm `qk_rope_head_dim = 0`" — confirmed, and it is not only a RoPE deletion.** It changes
`kv_a_proj_with_mqa`'s width from 576 to 512 and leaves `rope.dimension_count 0`, so a 5.2-shaped
loader fails on a shape rather than on a silently wrong rotation (§5).

**d. "MTP is out of scope until the base path works, but the tensor list says whether the NextN
block is even present."** It is present, as a full block: a 46th entry in `head_count_kv` and the
sparse-attention + MoE tensor set, plus `nextn.eh_proj [8192, 4096]`, `nextn.enorm`,
`nextn.hnorm` and `nextn.shared_head_norm` — and, unlike the trunk, no `hc_*`. It is not a
weight-only stub, so #398's scheduler has to decide whether to load it at all rather than
assuming it is absent.

---

## 8. The artifact inventory

`unsloth/GLM-5.3-Flash-GGUF`, `UD-Q2_K_XL`, 1412 tensors across four shards. Computed from the
tensor table and the ggml block sizes the header names: **320.76 B parameters, 101.24 GiB** of
tensor bytes — the *whole* checkpoint, not a subset.

By layer type, which is the split #398 has to plan placement against:

| what | params | GiB | share |
| --- | ---: | ---: | ---: |
| routed experts (`ffn_{gate,up,down}_exps`, 43 layers) | 311.65 B | 94.68 | **93.5%** |
| KDA layers (34): `attn_{q,k,v}`, `attn_output`, `ssm_*` | 5.98 B | 4.34 | 4.3% |
| DSA layers (11): MLA q/kv/v projections | 1.59 B | 1.26 | 1.2% |
| embedding + `output.weight` | 1.27 B | 0.74 | 0.7% |
| shared experts + routers | 0.15 B | 0.12 | 0.1% |
| the 11 indexers | 0.08 B | 0.09 | 0.1% |
| the NextN block, experts excluded | 0.03 B | 0.03 | 0.03% |

And by storage type:

| type | params | GiB | where |
| --- | ---: | ---: | --- |
| IQ2_XS | 198.11 B | 53.33 | `ffn_{gate,up}_exps` |
| IQ3_XXS | 99.05 B | 35.31 | `ffn_down_exps` on most layers |
| IQ4_XS | 7.25 B | 3.59 | `ffn_down_exps` on the last few layers |
| Q2_K / Q3_K | 7.25 B | 2.45 | the **NextN** block's experts only |
| Q5_K | 4.73 B | 3.03 | `attn_q`, `attn_output`, `ffn_{gate,up}_shexp` |
| Q6_K | 2.88 B | 2.20 | `attn_{k,v}`, `ffn_down`, `ffn_down_shexp` |
| Q8_0 | 0.81 B | 0.80 | the KDA gates, the indexer, `hc_*_fn`, `nextn.eh_proj` |
| Q4_K | 0.63 B | 0.33 | `output.weight` |
| F32 | 0.06 B | 0.21 | every norm, bias, `ssm_a`, `ssm_conv1d_*` and `hc_*_base`/`hc_*_scale` |

The mix is what an "unsloth dynamic" quant means: the routed experts — **311.7 B of the 320.8 B, at
2.3 bits a weight** — are the low-bit part, and essentially everything else is Q5_K or better. What
is left unquantized is all the small, precision-sensitive tensors: `ssm_a`, `ssm_dt.bias`,
`ssm_conv1d_{q,k,v}`, `ssm_norm`, every `hc_*_base`/`hc_*_scale`, `ffn_gate_inp`,
`exp_probs_b.bias`, `indexer.proj` and both indexer norm tensors are **F32** in the file. The
hyper-connection projection is the one exception that is neither F32 nor float: `hc_*_fn` is Q8_0,
which is a quantized type but a 1-byte one, so #398 has to dequantize it rather than `read_dense`
it.

Two layout facts #398 has to plan around:

- **The experts are 3-D.** `ffn_gate_exps` is `[4096, 2048, 288]` — one tensor per layer holding all
  288 experts, not 288 tensors, and `ffn_down_exps` is `[2048, 4096, 288]` on the same pattern. In a
  sparse layer the three expert tensors are **2.16 GiB of 2.28 GiB** — 95% of the block, against
  0.12 GiB for the MLA, the shared expert and the router together — so a layer's routed experts
  cannot be resident and must be staged. That is the decision
  [GLM-5.2's record](glm_5_2_design.md) already made once at a larger size, now with the
  compression that made the file loadable at all.
- **The 34 KDA layers are not the memory problem.** A KDA layer's attention is ~101 MiB, but 96 of
  that is the four 8192-wide projections; the `ssm_*` gates this stage is actually about are
  **3.7 MiB**. All 34 KDA layers together are 4.34 GiB against the experts' 94.68 GiB. The linear
  layers' cost is per-token structure — 34 sequential state updates a step — not footprint.

---

## 9. Go/no-go

**Go, with the scope of task 2 rewritten.** The checkpoint is a port throughout, but not the port
the issue tree describes.

| task | as scoped | after this audit |
| --- | --- | --- |
| #396 linear layers | retune the GDN kernel, or scope a new kernel | **reuse `chunk_/recurrent_gated_delta_rule` from `qwen4_exp`, write a new layer wrapper.** The scan is common; the projections, the three gates, the per-head `beta`, the fp32 l2norm order and the gated norm are KDA's. |
| #397 sparse layers | port the 5.2 indexer, implement four new knobs | **port the 5.2 MLA; rewrite the indexer for k-pool compression.** Three of the five knobs do not exist and the other two are inert. The new work is the pool compressor, and it makes selection 4× cheaper. |
| #398 checkpoint | read the type mix, decide placement | **unchanged, and now with the actual mix**: 311.7 B of the 320.8 B is experts, and in a sparse layer they are 95% of the block, so the 3-D expert layout forces staging and the NextN block is a full block to load or skip. |
| #399 serve + write-up | compare the hybrid against GLM-5.2 | unchanged. |
| mHC (unlisted) | the risk the roadmap named | **removed.** It is the same mHC DeepSeek-V4.1 already runs, and the kernel exists. |

The one open question this audit could not close from an artifact is arithmetic, not structural:
**whether KDA's bounded `(-5, 0)` decay changes what a 1048576-token context needs** — the released
`context_length` / `max_position_embeddings`, not a round number chosen here. GDN's softplus
lets a state decay arbitrarily fast, and this repository's Qwen3.8 path relies on that; KDA's floor
of `e⁻⁵` per step caps how fast old information can be dropped at a fixed rate. Whether that is a
precision detail or a capability difference is a question for #396's parity run, and it is named
here because it is the one place the two models' behaviour could diverge rather than their code.

---

## Evidence

| claim | read from |
| --- | --- |
| The 45-layer table, `kda_layers` / `full_attn_layers`, `mlp_layer_types`, `indexer_types` | `config.json` of `zai-org/GLM-5.3-Flash`, top-level `text_config` |
| The KDA forward, the encoder body, `l2norm`, the forget gate's `-5·sigmoid`, `beta[..., None]` | `modeling_glm5_next.py` in `transformers` v5.17.0: `Glm5NextTextLinearAttention` (587), `Glm5NextTextForgetGate` (306), `recurrent_kimi_delta_attention` (430), `l2norm` (418), `Glm5NextTextRMSNormGated` (341) |
| mHC arithmetic and scheduling | same file, `Glm5NextTextHyperConnection` (220) and `Glm5NextTextDecoderLayer` (1262) |
| The indexer, pool compression, tail, `output_width` | same file, `Glm5NextTextIndexer` (739) and `Glm5NextTextAttention` (1067) |
| The 1412-tensor inventory, the type histogram, `head_count_kv`, `kda.gate_lower_bound`, `ssm.conv_kernel`, `hyper_connection.*`, `nextn_predict_layers` | the four UD-Q2_K_XL shard headers, over HTTP range requests. The first shard's metadata sits in its opening ~9.4 MB and it carries no tensors at all; shards 2/3/4 carry all 1412 between them, each within its first 16 MB |
| GLM-5.2's indexer geometry (`head_count 32`, `key_length 128`, `top_k 2048`, `rope.dimension_count 64`) | `unsloth/GLM-5.2-GGUF`'s `UD-Q2_K_XL-00001-of-00007` header |
| GDN's decay, `l2norm` site and norm activation | `src/models/qwen4_exp/attention.py` |
| DSV4.1's mHC, `hc_post`'s axis semantics and `norm_eps` source | `src/models/deepseek_v4_1/modules.py`, `src/models/deepseek_v4_1/config.py`, and a numeric check of `hc_post` against `einsum('ji,jd->id', comb, residual)` |
| `hc_split_sinkhorn`'s arithmetic and the Triton path | `src/kernels/ops.py` (`hc_split_sinkhorn_torch`, `hc_split_sinkhorn`) |
| GLM-5.2's indexer loader contract | `src/models/glm_dsa/spec.py` |

**Environment.** This audit is a read, not a measurement, so it is host-independent — but the
machine it was made on is the aarch64 Ascend 910B host, which has no CUDA toolchain. The
`trust_remote_code` reference the task asks for turned out not to be needed: `transformers` 5.17.0
ships `models/glm5_next/` in-tree, which is the same release the config names
(`transformers_version`).

---
