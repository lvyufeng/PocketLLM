# New model support on the 2080 Ti: an ordered roadmap

This is the engineering record and the plan behind [issue #380](https://github.com/lvyufeng/PocketLLM/issues/380),
the repository's second roadmap. [The old-hardware roadmap](pocketllm_roadmap_old_hardware.md) asks what
*capability* the engine is missing — continuous batching, prefix caching, KV quantization, offload. This
one asks which *checkpoints* to support next, and it is tracked in
[PocketLLM Execution Queue](https://github.com/users/lvyufeng/projects/6).

Three checkpoints, in this order:

| # | Checkpoint | Weights we would run | Issue |
| ---: | --- | --- | --- |
| 1 | Ternary-Bonsai-2-27B | 5.5 GiB, PTQ1_0 ternary GGUF | [#381](https://github.com/lvyufeng/PocketLLM/issues/381) |
| 2 | Xing4.0-29B-A4B | 18.7 GiB, official IQ4_NL GGUF | [#388](https://github.com/lvyufeng/PocketLLM/issues/388) |
| 3 | GLM-5.3-Flash | 101.3 GiB, UD-Q2_K_XL GGUF | [#394](https://github.com/lvyufeng/PocketLLM/issues/394) |

## The lens: what the card can actually execute

An RTX 2080 Ti is compute capability 7.5. It has FP16 tensor cores and INT8/DP4A. It has **no BF16, no
TF32, no FP8 and no FP4** — every modern checkpoint's release format is one of those four. So "which
model does the release ship" is the wrong question here. The question that decides the cost of a new
checkpoint is:

> Is there a low-bit artifact whose packing something can consume, and what does it cost to teach our
> kernels to?

This is why the table above lists *weights we would run* and not the safetensors the authors published.
It is also why the ordering below is not by model quality or by popularity: it is by how much of the
existing engine each one reuses.

Two hardware facts bound all three stages:

- **`/dev/shm` is the scarce resource, not RAM.** The box has ~1 TiB of memory but the resident expert
  banks live in `/dev/shm`, which is 700 GB and currently holds the DeepSeek-V4.1-Flash bank (491 GB)
  and the MiMo-V2.6-Flash bank (161 GB). A new multi-hundred-gigabyte bank means evicting one of those
  first — a real cost, and part of why the first two stages were chosen as single-card or nearly so.
- **A dump cross a PCIe 3.0 link is a floor, not a starting point.** The MiMo-V2.6 record measured the
  expert copy at essentially the link's own rate; any checkpoint whose weights must stream from host
  memory a token inherits that floor. Stages 1 and 2 avoid it by fitting on the card.

## Stage 1 — Ternary-Bonsai-2-27B

`prism-ml/Ternary-Bonsai-2-27B-gguf`, 2026-09-16. The model card's own claim is that it is *"Derived
from Qwen3.8-27B, a 27B hybrid-attention causal language model (architecture unchanged)"*, and the
GGUF header confirms it field for field: `general.architecture = qwen35` — the string our Qwen3.8-27B
runtime already dispatches on — 64 blocks with `full_attention_interval = 4` (48 Gated DeltaNet + 16
GQA), hidden 5120, 24 query heads over 4 KV heads at head_dim 256, and a **dense** MLP of 17408.

That means the attention, the linear layers, the runtime, the server, the batching and the prefix cache
all exist and are validated in TP4 on `/mnt/data2/Qwen3.8-27B-FP8`. The stage's new work is the
artifact:

| | |
| --- | --- |
| `PTQ1_0` | 5.54 GiB, 1.75 bits/weight, trits packed densely — GGML type **143** |
| `PQ2_0` | 6.71 GiB, 2.13 bits/weight — GGML type **142** |
| Tensor mix | 402 ternary, 353 F32 (norms, `ssm_a`, `conv1d`, `dt`), 96 BF16 (`ssm_alpha/beta`) |
| Reference kernels | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), `prism` branch |

Every id here is read out of the file's own header, not from documentation. Two details matter and both
are unusual:

- **The type ids are fork-private.** Upstream GGML assigns nothing near 142; our
  `src/loader/gguf/quant_types.py` lists nine dense types and no ternary one. The failure mode to design
  against is a silent F16 upcast, which costs ten times the memory and makes a wrong kernel look right.
- **`token_embd.weight` and `output.weight` are ternary too**, and `token_embd` carries an *inverse*
  Hadamard. The model card is explicit that there are "no high-precision escape hatches behind a low-bit
  label", and the header bears that out: the packing is end to end.

The Hadamard is the stage's real risk. `prism.hadamard.*` declares a normalized
Sylvester–Walsh–Hadamard transform, `axis = input-last-dimension`, block size 1024, with **explicit sign
vectors** of widths 5120, 6144 and 17408. It is the incoherence transform that makes a 1.75-bit weight
representable, so getting it subtly wrong produces fluent-looking nonsense rather than a crash — hence
the first task in the stage is to *read the fork's source and write down the semantics*, not to
reimplement from the name.

Why it is first: on this card it is the cheapest stage to build and the most differentiated to ship. It
puts a 27B model with a 262144-token context in 5.5 GiB of weights, leaving roughly 16 GiB of a 22 GiB
card for KV. The old-hardware roadmap's first differentiation claim is 极致量化 — "FP4/Q2 lets a 2080 Ti
run a 70B"; this would give it a sharper example. And the quality number is the interesting one: the
card reports 84.78 average over 14 thinking-mode benchmarks, 98.2% of the FP16 parent, against **72.59
for a conventional IQ2_XXS build at a larger size** — and IQ2_XXS is a quant this repository already has
hand-written kernels for.

[The gate measurement](ternary_bonsai_2_reference_gate.md) has run since this was written: the upstream
reference decodes at 30.7 tokens/s and prefills at 665 tokens/s on one card, and 262144 contexts fit in
15,836 MiB with a quantized KV cache. It passes, and it also shows the reference spending only a third of
the card's bandwidth per decode step, which is the number the kernel task has to beat.

The loader task ([#384](https://github.com/lvyufeng/PocketLLM/issues/384)) has run since as well. Both packs
read end to end: 851 tensors whose byte counts tile the file exactly, the type histogram the release ships
(402 quantized / 353 F32 / 96 BF16), and decoded rows equal to the same rows of the checkpoint's own F16
GGUF, bit for bit. The bytes are addressable and nothing runs them yet — the loader refuses a ternary tensor
by name rather than upcasting it to F16, which is the failure mode the task existed to prevent.

The Hadamard transform ([#385](https://github.com/lvyufeng/PocketLLM/issues/385)) has run too, and it is the
half of this stage that cannot fail loudly. `src/loader/gguf/prism_hadamard.py` now computes the rotation as
well as parsing it, and every case in `tests/test_prism_hadamard_transform.py` is checked against
`scripts/prism_hadamard_oracle.cpp` — the fork's own `ggml_permute`, `ggml_mul` and FWHT path — as a digest
of the fork's fp32 bytes, not as a tolerance. What that leaves unclaimed is parity: the activations are
synthetic, because the model does not run yet, so "the transform is the fork's" is proven and "the model
generates" is not.

The ternary dense GEMM ([#386](https://github.com/lvyufeng/PocketLLM/issues/386)) has run as well, and it is the
stage's first uncomfortable number. Both phases are correct — bit-exact in bf16 against their own reference
arithmetic over the released weights, with the packing pinned by the decoder's own round-trip — and the decode
half is where the format pays: the dense projections of one forward pass **decode at 21.8 tok/s against 9.2 for
the same shapes at FP8 width**, at a tenth of the bytes. Prefill is the other way round: **600 tok/s of dense
projections at a 512-token prompt, 1.27× slower** than handing the same weights to cuBLAS as fp16, and both
phases are behind what the gate measured the upstream *whole model* reaching on this card. The decode gap has a
measured mechanism — a lane reads its own 28-byte block with seven 4-byte loads, and a reduction at that stride
reaches 73 GiB/s of useful bytes where a dense one reaches 526 — so the fix is staged loads rather than a
different unpack. [The measurement and both gaps](ternary_bonsai_2_dense_gemm.md) are written up in full.

What remains before the model generates is the runtime wiring
([#387](https://github.com/lvyufeng/PocketLLM/issues/387)), and the kernel work the two gaps above name.

## Stage 2 — Xing4.0-29B-A4B

`XingChen-AGI/Xing4.0-29B-A4B`, 2026-09-16, 58.1 GiB of safetensors with an official IQ4_NL GGUF at
18.7 GiB. This is the stage where the card stops being the constraint: 18.7 GiB fits one 2080 Ti, so the
smallest card in the fleet gets an architecture with the shape of a frontier model.

| | |
| --- | --- |
| Blocks | 40 (+1 NextN), hidden 3584, 32 query heads |
| Attention | **MLA**: `q_lora_rank 768`, `kv_lora_rank 512`, `qk_nope 128`, `qk_rope 64`, `v_head_dim 128` |
| MoE | 64 experts, top-4, 1 shared, 2 leading dense blocks, `expert_ffn 1024` |
| RoPE | **YaRN**, `original_context_length 4096` |
| GGUF types | 243 × `IQ4_NL`, 327 BF16, 406 F32, 1 Q6_K |

MLA is familiar — DeepSeek-V4.1-Flash and GLM-5.2 both use it — but the ranks and head counts differ, so
the mapping is a mapping and not a copy. Two things are genuinely new:

- **`IQ4_NL` is not in our loader.** `IQ4_XS`, its k-quant sibling, is one of the nine types we support;
  the non-linear variant decodes through a 16-entry codebook rather than a scale-and-add, and it is 243
  of this checkpoint's tensors.
- **`hc_*` hyper-connections.** Every block carries `hc_attn_fn.weight` of shape `[14336, 24]`,
  `hc_attn_base` of 24 and `hc_attn_scale` of 3, and the same for the FFN, with
  `hyper_connection.count = 4`, `sinkhorn_iterations = 20`. 14336 is 4 × 3584, so a 24-wide coefficient
  vector is produced per token from a 4×-hidden projection and then Sinkhorn-normalised. That is a
  *matrix* hyper-connection: it widens the residual topology every model in this repository has had as a
  single stream. It is the one part of the architecture with no precedent here, and the stage's first
  task is a go/no-go on it — read out of the checkpoint's own remote code, not guessed.

The stage is second because it is a genuinely new architecture family at a size that fits, and because
it is the only one of the three where the *card* is not the interesting constraint.

The gate ([#389](https://github.com/lvyufeng/PocketLLM/issues/389)) has run:
[the audit](xing4_0_29b_a4b_audit.md) reads the hyper-connection out of the checkpoint's own remote code
and out of the open llama.cpp port, and the two agree operation for operation, so the block is a port
rather than a research project and the stage is a **go**. The same page is where the description a
reimplementation is written from now lives, and it corrects this document's attention ranks — the only
reading of its Stage 2 table that did not survive the artifacts, since 192 is `key_length_mla` and its
halves are 128 and 64. Two of the audit's findings change the remaining tasks rather than annotating
them: **`IQ4_NL` is smaller work than it looked** (`reader.GGML_TYPES` already carries its geometry, so
what is missing is a codebook and a dispatch entry, and the format is upstream rather than
fork-private), and **this GGUF leaves the whole attention path at BF16** — 2.117 GiB of the 3.761 GiB a
decode step reads, against 0.877 GiB for the routed experts. On a bandwidth-bound decode step the MLA
path, not the MoE, is where this checkpoint's time will go, which is the opposite of what MiMo-V2.6 and
V4.1 both measured and is the first prediction the stage can be judged against.

## Stage 3 — GLM-5.3-Flash

`zai-org/GLM-5.3-Flash`, 2026-08-25, 305.8 GiB of FP8 safetensors with upstream GGUFs at
[`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) — `UD-IQ1_S` at
86.7 GiB, `UD-IQ2_XXS` at 94.9, `UD-Q2_K_XL` at 101.3, `UD-Q4_K_XL` at 186.0. UD-Q2_K_XL is the target:
101 GiB fits TP2 on the NVLink pair, or a three-way split of the four cards, on the 2-bit path this
repository already has.

The architecture is why it is last. Forty-five layers alternate **three `linear_attention` to one
`deepseek_sparse_attention`** — 34 linear, 11 sparse, read out of the config's own `layer_types`. Both
halves exist in this repository and have never run together:

- **linear attention** is Qwen3.8-27B's 48 Gated DeltaNet layers, validated in TP4 with a server;
- **sparse indexed attention** is GLM-5.2's `glm-dsa` path, with its indexer, its `index_topk` and its
  own measured record — in which the indexer's prefix tile was half collective traffic, which is why its
  default depth is zero.

Everything else is familiar: hidden 4096, 288 experts activated top-8 with one shared,
`first_k_dense_replace 3`, `kv_lora_rank 512`, `q_lora_rank 1536`. One declared value has to be treated
as a warning rather than a detail: `qk_rope_head_dim = 0`, i.e. **no RoPE on the qk path**, which no
model here has yet and which a copy-paste port from GLM-5.2 would get silently wrong.

Note what this stage is *not*: `zai-org/GLM-5.3`, the 703.7 GiB `glm_moe_dsa` release, is GLM-5.2's
trunk with a newer indexer (`indexer_types`, `index_topk_pattern`, `index_skip_topk_offset`,
`index_share_for_mtp_iteration`) and would be a separate, smaller piece of work.

## What "supported" means here

The bar every model in the support matrix was already held to, restated so this roadmap can be judged
against it:

1. The released checkpoint generates tokens on this hardware from a **real prompt** — through the
   model's own tokenizer, not a list of synthetic ids. A synthetic prompt moves the router's draw and
   invalidates every bytes-moved figure, which is a mistake this repository has made and recorded.
2. A **comparable** number: prefill and decode tokens a second at a stated context, taken as
   [the benchmarking rules](../guides/benchmarking.md) require — a real prompt, warm state, one
   configuration per process, differences taken interleaved inside one process with a null arm where the
   difference is small.
3. Served behind the OpenAI-compatible endpoint, when the checkpoint is text-only.
4. A guide under `docs/models/` and a design record under `docs/architecture/`, registered in the
   directory's `index.md` and in the `mkdocs.yml` nav **in the same commit** — `mkdocs build --strict`
   treats a missing registration as a failure, which is the point.

Out of scope for all three: vision and audio towers, which every one of these checkpoints bundles and
this repository deliberately does not read; and any claim about model quality, which belongs to the
authors' evaluations and not to an inference engine's measurements.

## Ordering, and when to stop

The order is cost-of-reuse, and each stage has an explicit early gate rather than a hope:

| Stage | The gate | What a failure means |
| --- | --- | --- |
| Bonsai | task 1/6 measures the upstream reference on this card | the ternary path is slower than the memory it saves → close the stage on the measurement |
| Xing4.0 | task 1/5 reads the hyper-connection out of the reference code | it is not a port → re-scope or drop before any kernel is written |
| GLM-5.3-Flash | task 1/5 identifies whether the linear layer is GDN | it is not → the stage grows a new attention kernel and the estimate changes |

Stage 1's gate has run: [the reference measures 30.7 tokens/s of decode and 665 tokens/s of prefill on one
card](ternary_bonsai_2_reference_gate.md), which is a pass on both the memory and the speed axis. That page also
pins the block format, the Hadamard and the kernel question, so the tasks after it start from a measured artifact
rather than from a model card.

Stage 2's gate has run too, and it is also a pass: [the audit](xing4_0_29b_a4b_audit.md) writes the
hyper-connection out as a forward pass a reimplementation can be built from and finds it agreed
operation for operation by the two published implementations of it, which is what "it is a port" means
here. It also re-reads this document's Stage 2 table against the artifacts and produces the per-token
byte table that the stage's final task is measured against.

A stage that dies at its gate is a result, and it goes in this document rather than the issue tree being
quietly pruned. That is the same convention the old-hardware roadmap follows: it records what was
measured and closed — the FP4 cross-layer prefetch, the score-split hybrid, the resident expert cache —
and not only what shipped.

## Evidence

Everything in this document is read from the artifacts rather than from documentation: the GGUF
metadata sections, `config.json` files, and the safetensors inventories, all via the Hugging Face API
and HTTP range requests. The specific reads:

- `prism-ml/Ternary-Bonsai-2-27B-gguf` — metadata and tensor table of both `PTQ1_0` (type 143) and
  `PQ2_0` (type 142); architecture string `qwen35`, `full_attention_interval 4`, and the
  `prism.hadamard.*` block.
- `XingChen-AGI/Xing4.0-29B-A4B-GGUF` — metadata and tensor table of `xing4_0-29b-IQ4_NL.gguf`;
  architecture string `xing4_0`, MLA ranks, the `hc_*` tensors, and the `hyper_connection.*` keys.
- `zai-org/GLM-5.3-Flash` — `layer_types` (34 linear / 11 sparse), the indexer keys, and the quant
  inventory of its GGUF repository.
- `zai-org/GLM-5.3` and `XiaomiMiMo/MiMo-V2.6-Pro-RL` — configs read for the comparison in the issue
  tree; the Pro checkpoint is deliberately not a stage here, for the reasons in
  [#380](https://github.com/lvyufeng/PocketLLM/issues/380).

## Related

- [Feature roadmap for old hardware](pocketllm_roadmap_old_hardware.md) — the capability axis, and the
  other half of what this repository is for
- [PocketLLM vs vLLM vs SGLang](vllm_sglang_architecture_analysis.md) — the comparison that motivates
  serving a single card well rather than scaling out
- [Qwen3.8-27B-FP8 design and measurements](qwen3_8_27b_fp8_design.md) — the runtime stage 1 reuses
- [GLM-5.2 design and measurements](glm_5_2_design.md) — the DSA path stage 3 reuses
- [Benchmarking and reporting](../guides/benchmarking.md) — how every number in these stages must be
  taken
