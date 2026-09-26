# Xing4.0-29B-A4B: the design record

What the runtime for `Xing4.0-29B-A4B` does, why it does it that way, and the measurements behind each
choice. It is written for changing the runtime rather than for running it — the commands a user wants
are on [the guide](../models/xing4.0-29b-a4b.md).

**Read the numbers with their kind.** Most of what follows is a *difference* taken in one process,
because two 18 GiB models do not fit on a 22 GiB card and a lever that cannot be isolated cannot be
priced: the hyper-connection's 2.17× is A-B-A-B interleaved on the same weights, and the A-B-A-B
harness is in the repository so the measurement can be repeated rather than taken on trust. The ones
that are *single readings* — a prefill rate, a decode rate, a memory figure — say so where they
appear, with the card, the context and the invocation. Two of them are explicitly conditional on the
allocator's state and both readings are given.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388). The stage's other four tasks each
have their own record: the [audit](xing4_0_29b_a4b_audit.md) for the block itself, and
[#390](https://github.com/lvyufeng/PocketLLM/issues/390) IQ4_NL, [#391](https://github.com/lvyufeng/PocketLLM/issues/391)
MLA and [#392](https://github.com/lvyufeng/PocketLLM/issues/392) the hyper-connection for the pieces
this one composes.

---

## 1. The shape of the thing

Forty trunk blocks, each one an MLA attention and a 64-expert MoE with one shared expert, wrapped in a
**matrix hyper-connection** that gives the block four residual streams instead of one. Thirty-eight of
the forty blocks are MoE (`first_k_dense_replace = 2`), so 14.7 GiB of the file is expert weights —
which is the number the whole design follows from.

| | Value |
| --- | ---: |
| File | 20,104,012,544 bytes — 18.72 GiB, 977 tensors, GGUF type 25 |
| Resident | 17.94 GiB, every expert on the card |
| Trunk | 40 blocks; `blk.40` is the NextN/MTP block and is not loaded |
| Attention | MLA, absorbed: 512 latent + 64 rope key = 576 values a layer a token |
| MoE | 64 routed top-4 + 1 shared, `expert_ffn` 1,024, sigmoid router, `routed_scaling_factor` 2.0 |
| Residual | `hc_mult` 4, Sinkhorn 20 iterations, logits clamped to ±30 |
| KV | 46,080 bytes a token across the 40 layers, fp16 |

### One decode token's bytes

| Per token | GiB | Share |
| --- | ---: | ---: |
| BF16 attention, all 40 blocks | 2.117 | 56.3% |
| Routed experts, 4 of 64 selected | 0.877 | 23.3% |
| Q6_K `output.weight` | 0.359 | 9.5% |
| Shared expert | 0.219 | 5.8% |
| Dense FFN, blocks 0–1 | 0.104 | 2.8% |
| `hc_*`, router, norms, embedding | 0.086 | 2.3% |
| **Total** | **3.761** | |

At 616 GB/s that is 6.56 ms, or 152 tok/s. **The measured rate is 6.7 tok/s**, and the reason is not
in this table — it is in §6.

---

## 2. The residual streams are carried wider than the sublayers

This is the defect a served conversation found, and it is worth the space because it was invisible to
every test that existed at the time.

**Symptom.** A two-turn chat answered in `<_unk>` tokens. A one-turn chat was fine. So was a raw
completion prompt of the right shape.

**It is not the template, the cache or the kernel.** The same degeneration reproduced with the fused
kernel on and off. It reproduced on a *cold single* `forward` of the raw token ids, with no cache
reuse involved at all. What the degenerate prompts had in common was that they contained an
end-of-turn token or an assistant message — i.e. the model's own generated text, fed back.

**What it is.** Walk the trunk one block at a time and watch the residual's magnitude, for a
one-token prompt of `<_end>` (token 2):

```
block  0..25   finite, growing from 0.09 to 8.4
block 26       finite, max 8.37
block 27       MoE output max 2.171e+04      <-- the value that leaves fp16
block 28       finite, 3.07e+04
block 29       finite, 6.55e+04  (fp16 saturates at 65504)
block 30..39   NaN
logits         all 131072 entries NaN
```

`post.unsqueeze(-1) * mlp_out.unsqueeze(-2)` is computed in fp16, so block 29's 6.55e+04 becomes
`inf`, and from block 30 the residual is `inf - inf`.

**The weights are not the bug.** Two independent checks say the checkpoint is what it says it is:

- Dequantize `blk.27.ffn_gate_exps.weight` from IQ4_NL to fp64 and run the block's own MoE against
  the captured activation: expert 0's output is **1.2309e+04**, the same order as the kernel's 2.17e4.
  The magnitudes are in the file.
- They are in the file because that expert is *loud*: its IQ4_NL block scales run 0.0054 at the top
  against a neighbours' 0.0011, i.e. 5×, and the vendor quantized from an imatrix
  (`quantize.imatrix.file` is in the GGUF header). Across the 40 blocks, five have at least one expert
  whose weight bound exceeds 0.3, and `blk.39`'s expert 5 reaches 2.523 against a 0.16 median.

**The reference does not have this problem because it is bf16.** `modeling_xing4_0.py` runs the whole
model in bf16, whose range is fp32's. This port ran fp16 because that is what an sm_75 card's tensor
cores and `torch.matmul` are fast at, and 1e5 is inside bf16 and outside fp16.

**The fix: split the two widths.** The sublayers keep the narrow dtype — what they are handed has just
been through an RMS norm, so their arithmetic is on O(1) numbers whatever the residual has grown to —
and the four streams move to fp32:

```python
attn_out = attend(collapsed.to(self.dtype), ...)          # sublayer: model dtype
hidden = post.unsqueeze(-1) * attn_out.to(residual).unsqueeze(-2) + torch.matmul(comb, hidden)
mlp_out = self.mlp(collapsed.to(self.dtype))
return post.unsqueeze(-1) * mlp_out.to(residual).unsqueeze(-2) + torch.matmul(comb, hidden)
```

Two other places had to move with it, and both are the same mistake:

- **`RoutedMoE` narrowed its own sum.** The grouped kernel returns fp32 and the shared expert is added
  to it; the result was then cast to the model dtype. `routed + shared` is exactly where a 1e4 becomes
  an inf, so both halves are now fp32 (`out_dtype`) and the block makes the narrowing decision once.
- **`silu(gate) * up` is quadratic in the input** and is the quantity that actually crosses 65504, so
  every expert's activation product is handed to its down projection in fp32: the shared expert's
  `QuantizedGGUFLinear`s are built `out_dtype=torch.float32`, and `SwiGLUMLP` for the dense blocks too.

**Cost: nothing measurable.** One card, real prose, 512 tokens, greedy:

| | Prefill | Decode |
| --- | ---: | ---: |
| fp16 residuals (before) | 102.11 tok/s | 3.20 tok/s |
| Mixed, sublayers fp16 (after) | 102.01 tok/s | 3.33 tok/s |

**Correctness: the three degenerate prompts now agree with a fully-fp32 model.** The same
one-token `[2]` prompt that produced a NaN row returns `▁` at 15.0, and a two-turn chat — the prompt
that started this — returns `1` at logit 30.5 where fp16 returned NaNs. Both match the
`dtype=torch.float32` run's top-4 exactly, which is the evidence that the fix is a width and not a
behaviour change.

The regression tests are `test_the_residual_stream_is_wider_than_the_sublayer` (both halves: the
sublayer is handed the narrow dtype *and* its output reaches the residual whole) and
`test_the_moe_does_not_narrow_its_own_sum`.

---

## 3. The hyper-connection in one kernel

The block's plumbing, read out of the reference and cross-checked against the open llama.cpp port in
[the audit](xing4_0_29b_a4b_audit.md#2-the-two-released-implementations-agree), is:

```
flat       = unweighted_rms_norm(hidden_streams.flatten(-2))     # 4 x 3584 = 14336 wide
w          = hc_fn @ flat                                        # [24, 14336]
pre, post, comb = w.split([4, 4, 16])
pre        = sigmoid(pre_w * scale[0] + base[0:4])               # (0, 1)
post       = 2 * sigmoid(post_w * scale[1] + base[4:8])          # (0, 2)
comb       = sinkhorn_20(exp(clamp(comb_w * scale[2] + base[8:24], -30, 30) - rowmax))
collapsed  = Σ_s pre[s] * hidden[s]
hidden[d]  = post[d] * sublayer_out + Σ_s comb[d, s] * hidden[s]
```

**Why it is one kernel.** 20 iterations × 2 gates × 40 blocks is **1,600 serial dependent small-matrix
steps a token**, and the tensors are 4×4: the cost is the dependency chain, not the arithmetic, and a
chain cannot be shortened by giving each step more work. Anything that dispatches per step pays
1,600 launches for it. So `src/csrc/xing4_hyper_connection.cu` runs one block a row: 256 threads in
8 warps, the 24-wide gate split three ways across the warps (the three 24/8-wide dots), thread 0 doing
the sigmoid, the clamp, the exponential and the Sinkhorn in **shared** memory, and the `pre`-weighted
collapse at the end. One launch replaces 300.

The shared memory is not an optimization and the code says so: every index into the 16 is a runtime
`d`/`s`, so a register array would spill to local memory. The `#pragma unroll 8` on the gate loop is
for the same reason the comment gives, not a measured win.

**Cost, measured.** The profiler on the real model at a 4,096-token context attributes **9.07 ms of a
decode step's 46.5 ms device time to this kernel over 80 launches — 113 µs each**, which makes it the
single largest kernel in the step. The isolated figure was 61.7 µs a row when the model ran fp16; the
fp32 residual of §2 is why it is now nearer 113 µs, and that is a real cost paid knowingly.

**What it is worth.** In-process A-B-A-B, one model, the arm switched by rebinding
`HyperConnection.cuda` on the same objects, arms alternating so drift lands on both:

| Arm | ms/token | tok/s |
| --- | ---: | ---: |
| eager (`use_kernel=false`) | 266.1 (median of 3, min 264.9) | 3.76 |
| fused kernel | 122.5 (median of 3, min 122.5) | 8.16 |

**2.17×**, on a 60-token prompt with a 60-token context. The earlier build measured 2.45× when the
residual was fp16; the absolute rates moved on both arms and the ratio moved with them, and 2.17× is
the current build's.

The two arms generate coherent text and part company at token 6 — `A refractor bends light through`
against `A refractor uses a lens to`. That divergence is fp16 rounding accumulating over 80
hyper-connections, and it is why the correctness evidence for this kernel is a parity test against the
reference and not a text comparison: `test_the_kernel_is_the_port` at three dtypes and three row
counts, `test_the_kernel_agrees_with_the_eager_path_on_a_real_activation`, and
`test_the_kernel_survives_a_row_that_the_eager_path_never_sees` (`hc_mult=1`). The tolerances are
`{fp32: 1e-5, fp16: 4e-3, bf16: 4e-2}` — the kernel does its arithmetic in the activation's dtype, so
the tolerance has to move with the dtype.

---

## 4. The MoE, resident, with a route sum that is the same twice

Three decisions, none of them about speed.

**Resident, not a host bank.** 14.7 GiB of expert weights for 38 blocks is 396 MiB a block in IQ4_NL,
and the whole checkpoint is 17.94 GiB against a 22 GiB card. Every other MoE in this repository streams
its active experts over PCIe because its checkpoint is several times a card; here a per-token copy
would be a round trip for weights that are already resident. So the experts are read once at load,
folded into the 144-byte row element the grouped kernel indexes rows by
(`src/loader/gguf/iq4_nl.fold_to_runtime_span`), and held.

**One grouped kernel for prefill and decode.** 64 experts × 3 projections is 192 GEMMs a layer if each
is its own launch, and a 256-token chunk is 1024 routes. `gguf_moe_prefill_grouped_forward` takes a CSR
route table and every route in one call.

**top-4 is where float addition stops being reproducible, and this repository has that measured.** The
grouped kernel's scatter ends in an `atomicAdd` per output element, so its result depends on the order
blocks finish in. Two summands have one order; four do not. So `RoutePlan` sorts the routes by expert
and passes `route_tokens = arange(routes)`, which gives every `atomicAdd` exactly one writer, and the
top-k is summed afterwards in slot order by torch (`index_copy_` + `view(tokens, top_k, dim).sum(1)`).
The kernel is unchanged and the answer is the same on every run —
`test_the_grouped_kernel_is_bit_reproducible_at_top_four` and
`test_a_permuted_route_table_gives_the_same_answer`. The same ordering is what `DenseExpertStack`, the
host reference, produces the slow way, which is what makes the two comparable.

**Where the time goes at prefill**, from the profiler at a 512-token prompt:

| Kernel | Share of prefill |
| --- | ---: |
| `gguf_moe_w13` | 46.5% |
| `gguf_moe_w2_scatter` | 28.0% |
| `gguf_quant_gemm_prefill` | 21.5% |

74.5% of prefill is the routed experts, which is the opposite of decode's shape (§1, §6) and the
reason prefill falls only 24% from 512 to 32,768 tokens while its attention term grows quadratically.

---

## 5. The prefill chunk is a function of the card

`_chunk_for` exists because a first long prompt otherwise OOMs inside the attention several minutes
into a request. It did not work, for two reasons, and both were found by measuring rather than by
reading it.

**It counted 4 bytes a score element where the peak is 8.** The score matrix is fp16 (2 bytes), the
fp32 softmax the reference asks for is a *second copy of the whole thing* (4), and the fp16
probabilities it is cast back into are a third (2). All three are live at once. Measured at an
8192-token context:

| chunk | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| peak, measured | 281.5 MiB | 552.4 MiB | 1095.2 MiB | 2183.3 MiB |
| `8 · heads · chunk · context` | 256.0 | 512.0 | 1024.0 | 2048.0 |

The remainder is the second term, and it is not a fourth copy: it is linear in the chunk at
**126 KiB a chunk token plus 11 MiB**, which is the fp32 residual streams — `[1, chunk, 4, 3584]` is
56 KiB a chunk token before the hyper-connection rebuilds several of them — and the grouped MoE's
per-route buffers. `SCORE_BYTES_PER_ELEMENT`, `PREFILL_WORKSPACE_BYTES_PER_CHUNK` and
`PREFILL_WORKSPACE_FLOOR_BYTES` are those three numbers, and
`test_the_prefill_peak_is_the_two_measured_terms` pins the formula to the table above within 5%.

**It budgeted a constant rather than the card.** With `--max-model-len 32768` the cache takes 1.41 GiB
of the 3.40 GiB that are free and a 128-token chunk needs 1051 MiB of the rest, so the *room* is what
decides the chunk and it is a property of the device. `Xing4Backend._prefill_room` reads
`torch.cuda.mem_get_info` after the model and the cache are on the card and withholds a 768 MiB
reserve; `_chunk_for` inverts the peak formula against it.

**And past some context there is no chunk left**, because the KV cache and the score path want the
same few GiB. That is now a refusal at load rather than an OOM inside a request:

```text
ConfigurationError: --max-model-len 49152 leaves 554 MiB above the 2160 MiB cache, and the
narrowest prefill chunk (128 tokens) needs 1563 MiB of it at that context; lower
--max-model-len, or raise it on a card with more room
```

**The ceiling, measured on one card**, by allocating the cache and running a 128-token chunk at the
*end* of the context, which is the worst case:

| `--max-model-len` | Cache | Room | Chunk | Last chunk |
| ---: | ---: | ---: | ---: | --- |
| 32,768 | 1440 MiB | 1274 MiB | 128 | OK, 722 MiB free after |
| 40,960 | 1874 MiB | 840 MiB | 128 | OK, 2 MiB free after |
| 49,152 | — | — | — | OOM |
| 65,536 | — | — | — | OOM |

So the hardware's edge is between 40,960 and 49,152, and the runtime's own promise — after the reserve
— is about 34,816. Both are stated because they answer different questions.

---

## 6. Where a decode step's time goes

The profiler at a 4,096-token context, one card, greedy, ten steps:

```
decode step: wall 128.3 ms   device 46.5 ms   11536 launches/step
```

| Kernel | ms/step |
| --- | ---: |
| `xing4_hyper_connection_kernel<float>` | 9.07 |
| `gguf_moe_w13_kernel<half>` | 6.84 |
| `gguf_moe_w2_scatter_kernel<float>` | 3.74 |
| `gguf_quant_gemm_kernel<float>` | 3.59 |
| `gguf_quant_gemm_kernel<half>` | 3.41 |
| `gemvx::kernel` (the lm_head) | 3.26 |
| everything else | ~16.6 |

**11,536 launches at ~15 µs of host time each is 173 ms, and the device work is 46.5 ms.** Measured
directly, host submission is 178–187 ms a step against a 176 ms step, so the GPU is idle about
three-quarters of every decode step and the card's 616 GB/s is barely touched. That is the whole
explanation for 6.7 tok/s against a 152 tok/s byte floor, and it is why the entry in **Known
limitations** on the guide is about launch count and not about bandwidth.

*Corrected later:* the 6.5 this section first carried was measured across the prefill/decode seam,
which is a host clock around a boundary the device does not have — the prompt's last chunk drained
inside the first decode step and was charged to it. The measured size of that error, and the corrected
table, are in
[the record](../performance/xing4_0_rate_clock_split.md). Nothing in this section moves: the launch
count, the host time and the device time are all the step's own.

Two things follow, and both are left for a later stage rather than half-done here:

- **CUDA graphs are the obvious lever** and are not implemented. Three sites in the block are hostile
  to capture (pageable host-to-device copies and a `.item()`), and a decode graph would need them
  removed first.
- **Fewer, larger kernels is the other**, and the fused hyper-connection kernel is the evidence for
  what it is worth: 2.17× and 5,800 launches gone at once.

**Both were then measured, and this section's premise was wrong about where the host time goes.**
[Xing4.0-29B-A4B: the decode step's launch count, and what a graph buys](../performance/xing4_0_decode_launch_gap.md)
profiles the step by name and prices the ceiling. The step hands the host **22,155 ATen dispatches**,
of which **10,508 are metadata-only** (`view`, `reshape`, `as_strided` and their kin, 24 ms of host
time for nothing) and a further **4,349 are fp16↔fp32 casts** (21 ms); those submit **5,346
`cudaLaunchKernel` calls for 5,577 kernels that take 46 ms of device time**. A whole-step CUDA graph at
a frozen position then takes the step to **38.0 ms at bit-identical logits**, with the card at 100% of
the step and zero host submissions. Three corrections to what is above:

- **The hostile site is one, not three.** It is `torch.bincount` in `plan_routes`, which sizes its
  output from the data's maximum and so reads it back — **76 device-to-host copies and 78 stream
  drains a step**. Removed there; its own cost was 3.6 ms of the step, so it was never the bottleneck,
  but it is what refused the capture.
- **A `.item()` is not among them.** `generate.sample_token` reads the logits row back, and that is at
  the end of the step rather than inside it, so a graphed step excludes it by construction.
- **`torch.compile` is not a substitute.** It refuses cudagraphs on the step's mutated inputs and
  cannot trace the two pybind MoE ops, so it lands at 1.15× against the hand graph's 4.2× or better.

The fused-kernel lever is unchanged and is the *second* one: once the graph removes the submission
cost the step is device-bound at 38 ms, which is the number the fusions have to be priced against.
This section's own 11,536 launches does not reconcile with the profile above's 5,346 submissions and
5,577 kernels — different instruments, and the newer pair is the one that can be reproduced.

**A conditional number, honestly.** `generate` reports `first_step_seconds` as well as
`step_seconds`, because the first decode step after a prefill is not a steady-state step. In a warm
process it is 141–158 ms and indistinguishable from the rest. In a *fresh* process, where the
allocator has never held a request-sized working set, it is **571 ms after a 4,096-token prefill at a
128-wide chunk and 2,981 ms after the same prefill at a 1024-wide chunk**, against a steady 176 ms.
The mechanism is an allocation, not a kernel, and the consequence is that a 24-step average taken on a
cold process reads 4.02 tok/s where the same run's steady rate is 6.39 — which is exactly the mistake
this record nearly published. Both readings are reported, and the bench prints both.

---

## 7. Serving

### The adapter

`pocketllm/backends/xing4_backend.py`, selected by the factory from the GGUF's own
`general.architecture = xing4_0` or from a `config.json` whose `model_type` is `xing4_0`, and
selectable by hand with `--backend xing4`. Its capabilities:

| | |
| --- | --- |
| `supports_batch` | **False** — the trunk's forward flattens its input to one token axis, so two sequences given to it together would attend to each other. Serialized at the backend boundary. |
| `supports_streaming` | True |
| `supports_cancellation` | True, per decode step, at a token boundary — not inside a prefill chunk |
| `supports_prefix_caching` | True when a store exists, i.e. when the budget is non-zero |
| `supports_logprobs` | False |

**Two things have to be named, not one.** The weights are a `.gguf` and the tokenizer, the chat
template and the ±30 clamp bounds are a directory — the GGUF header carries the vocabulary but not the
merges its BPE needs, and no GGUF key carries `mhc_h_res_clamp_min`/`_max`. `resolve_paths` takes the
file from wherever a `.gguf` is and everything else from a directory, and refuses a launch that names
a file and no directory with the paths it looked at. `_is_xing4_release` tests for
`tokenization_xing4_0.py` or `model_type == "xing4_0"` rather than for a `config.json`, because the
model directory sits beside other checkpoints and an alphabetical scan would pick a sibling and
tokenize the prompt with the wrong vocabulary.

### A served request, from the released checkpoint, on one 2080 Ti

```
$ python -m pocketllm serve --backend xing4 \
    --model .../xing4_0-29b-IQ4_NL.gguf --tokenizer-path .../Xing4.0-29B-A4B \
    --served-model-name xing4 --device cuda:2 --max-model-len 8192 --port 8123

$ curl -s localhost:8123/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"xing4","temperature":0,"max_tokens":48,
  "messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}]}'
{"id":"chatcmpl-86db4034...","choices":[{"message":{"role":"assistant","content":
  "1.  **Analyze the Request:**\n    *   Question: \"What is the capital of France?\"\n    *
   Constraint: \"Answer in one short sentence.\"\n\n2.  **Formulate the Answer:**\n    "}}],
 "usage":{"prompt_tokens":19,"completion_tokens":48,"total_tokens":67}}

$ curl -s localhost:8123/metrics | grep xing4
pocketllm_xing4_resident_bytes 19262579648
pocketllm_xing4_kv_cache_bytes 377487360
pocketllm_xing4_context_positions 8192
```

`/metrics` also carries the store's family — `prefix_cache_hits_total`, `_misses_total`,
`_reused_tokens_total`, `_entries`, `_budget_bytes`, `_bytes` — published once at construction so a
deployment can read the budget it was launched with before it has served anything. SSE streams token
by token, and a client disconnect increments `pocketllm_cancellations_total` (measured: 1 → 2 across
one aborted request).

### The prefix store, and the bug a served run found

`src/models/xing4_0/prefix_cache.py` holds each prompt's cache state on the host keyed by the prompt's
own token tuple, under an LRU byte budget. A later request restores the longest prefix it shares with
one already served and forwards only the rest, at absolute positions.

Reuse works, measured on the served checkpoint: a repeat of the same prompt reported
`prefix_cache_hits_total 1`, `_reused_tokens_total 19`, and answered in **1.15 s** where the first
request took several seconds.

**What the served run found is that the caller's cache is not reset for you.** A service allocates one
`KVLatentCache` and hands it to every request; `append` only ever *raises* `length`. So a request whose
prompt was shorter than its predecessor's inherited the predecessor's trailing rows as context, and
attended to text that had left the conversation — plausibly, and silently. `KVLatentCache.reset()`
drops the length without touching a byte of the buffer (the length is what says how much of it is a
context, and the next prefill overwrites the rows it fills), and `_resume` calls it before anything is
forwarded, so the reset belongs to the loop rather than to a caller's memory. The test is
`test_a_second_request_on_a_shared_cache_does_not_see_the_first_one`.

**And what the store does not get: most of a chat's prefix.** The released chat template renders the
generation prompt as `<_bot><think>\n`, while a turn's assistant message is prefixed with `</think>`:

```
turn 1: <_system><_user>What is the capital of France?<_bot><think>\n
turn 2: <_system><_user>What is the capital of France?<_bot></think>Paris.<_end>\n<_user>And Japan?<_bot><think>\n
        |<---------- 10 shared ids ---------->|
```

Turn 2 shares 10 of its 24 tokens with turn 1 and diverges exactly where the model's own answer began.
Cross-*turn* reuse therefore recovers the system and user prefix and not the conversation — which is
the case a prefix cache is usually deployed for, and it does not hit. It is a template property, not a
store property, and the fix is a template that renders an empty assistant placeholder; none of the
checkpoint's own renderings does.

---

## 8. What was rejected

**Tensor parallelism.** Refused rather than unimplemented: `--tensor-parallel-size 2` raises.
`ep_size = 1`, nothing spills, and a per-layer collective would add to the host submission that
already bounds decode (§6) rather than subtract from it. The measurement that stands in for it is two
*processes*, one a card, on the same prompt: **82.81 and 84.29 tok/s prefill against 83.94 alone, and
7.27 and 7.09 tok/s decode against 7.10 alone** — 2× aggregate at no cost to either. (Both columns
were re-measured after §6's *Corrected later* note; the same seam was in them, and
[the record](../performance/xing4_0_rate_clock_split.md#the-two-card-table) carries the before and
after.) Running one
server a card is the supported multi-card shape here. Whether TP2 would raise a *single* stream's rate
is left open, and would have to be priced against the added collective.

**Batching.** `supports_batch` is False for the reason in §7. The memory budget agrees: 3.4 GiB free
is not a batching headroom on a checkpoint using 85% of the card.

**The NextN/MTP block.** `blk.40` is 24 tensors, 0.93 GiB, no `hc_*`, and is not loaded. This
repository's record on speculative decoding is that it is acceptance-dependent, and a length-2 MTP path
on a GGUF cost more than it returned for DeepSeek-V4. No speculative speedup is claimed here and there
is no flag that would produce one.

**A quantized KV cache.** It would extend the context past ~41k tokens and is not implemented; the
absorbed latent is 576 values a layer a token and quantizing it changes the attention's arithmetic in
a way that would need its own parity evidence.

**Splitting the hyper-connection kernel across blocks.** Considered and rejected on the launch
arithmetic: the kernel is one block a row, and a second launch to parallelise the row would double the
submission cost to save device time that is not the bottleneck.

---

## 9. Evidence

Every number above is a measurement on one x86_64 box with 4×RTX 2080 Ti, on the released
`xing4_0-29b-IQ4_NL.gguf` and the `Xing4.0-29B-A4B` release beside it, 2026-09-26. The commands:

| Claim | Command |
| --- | --- |
| Prefill and decode rates on real prose | `scripts/bench_xing4_0_e2e.py --device cuda:2 --lengths 512,4096,16384,32768 --decode-steps 32 --max-model-len 32768` |
| The hyper-connection's 2.17× | `scripts/bench_xing4_0_hyper_connection.py --device cuda:2 --steps 8 --arms 3 --warmup 2` |
| The per-chunk peak, and the two-term formula | a 128/256/512/1024 sweep at a fixed 8192-token context, `torch.cuda.max_memory_allocated` per chunk |
| The context ceiling | allocate the cache, then run a 128-token chunk at `ctx - 128` |
| The decode step's launch count and per-kernel device time | `torch.profiler` over 10 steps at a 4096-token context |
| The residual-width defect | a per-block magnitude walk on the one-token `[2]` prompt, plus an fp64 dequantized replay of `blk.27`'s MoE |
| The served request, streaming, cancel, metrics | `pocketllm serve --backend xing4 --device cuda:2 --max-model-len 8192 --port 8123` and `curl` |
| Two cards as two processes | the same bench on `cuda:0` alone and then on `cuda:0` and `cuda:1` together |

The tests are `tests/test_xing4_0_hyper_connection.py`, `tests/test_xing4_0_moe.py`,
`tests/test_xing4_0_attention.py`, `tests/test_xing4_0_rope.py` and `tests/test_xing4_0_serving.py` —
102 tests, none of which needs the 18 GiB checkpoint except the two parity modules, which skip
themselves without it. `pocketllm/backends/xing4_backend.py`'s chunk derivation is pinned to the
measurement by `test_the_prefill_peak_is_the_two_measured_terms`.

The artifact inventory, the block's semantics and the two agreeing reference implementations are in
[the audit](xing4_0_29b_a4b_audit.md), which is the read this stage was written against.
