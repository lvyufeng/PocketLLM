# Reference architectures

An architecture is a model's *structure*: it turns a config into a
[`ModelSpec`](../architecture/execution.md#the-model-ir) — a graph, a weight table and a cache plan —
and stops there. It names no device, allocates nothing, and imports no backend, so the same executor
drives every architecture in this tree.

Two ship today. `toy` is deliberately not a model; [`qwen3`](#qwen3) is one, written a second time
against the ABI — the C engine's copy is the first — and checked against a transcription of it.

## `toy`

**Status: scaffold.** Not a checkpoint and not a benchmark. It exists so the scaffold is *runnable*:
a builder that has never produced a graph an executor accepted is a design, not code.

```console
$ pocketllm architectures
toy           A tiny embedding + SwiGLU block; the executor's own smoke test
```

It builds an ordinary graph through the same `GraphBuilder` a real model uses, and runs it through
the same executor:

```
tokens -> embedding -> rms_norm -> gemm(gate) -> silu_mul -> gemm(down) -> y
                                -> gemm(up)   /
```

What that exercises end to end: shape inference, argument marshalling, dispatch, buffer liveness and
the memory planner. If `toy` stops running on the reference backend, something in the
executor-visible contract moved, whatever the unit tests claim.

Two details are deliberate rather than incidental:

- **The gate and up projections are two weights, not one fused `(2*ff, hidden)` projection.**
  `silu_mul` takes two tensors of one shape and the ABI has no slice op; a backend fuses that split
  into its own GEMM rather than paying for a copy. Two weights keeps the graph honest without
  inventing an op the ABI does not have.
- **Its cache plan is empty rather than absent.** A caller that sizes memory for it gets zero, not a
  special case.

## `qwen3`

**Status: scaffold.** The graph builds, verifies, and runs on the reference backend with synthetic
weights; `tests/architectures/test_qwen3.py` checks its logits against an independent numpy
transcription of `Qwen3Model::forward`, and
[`pocketllm.engine.decode`](../architecture/execution.md#decode) drives it token by token —
appending, advancing the position and sampling.

`tests/architectures/test_qwen3_oracle.py` goes further and closes the hole the transcription has.
A transcription written from the same reading of the C source agrees with the graph wherever that
reading is right *and* wherever it is wrong the same way, so on its own it cannot say the model is
Qwen3. That test therefore writes a tiny synthetic checkpoint with
`pocketllm.loader.gguf.writer` — a real GGUF with a real byte-level vocabulary, which both this
tree's readers and llama.cpp load — and compares the graph's logits against the C engine's and
llama.cpp's on the same weights. It is a `.gguf` bound into the graph and a token produced from a
file, so the earlier "nothing here has produced a token from a real file" is no longer true.

What is still not true, and why the status stays Scaffold: **no real checkpoint and no device**. The
checkpoint is two layers with an eight-wide head, written by the test rather than converted from a
model; nothing has bound `Qwen3-0.6B` into this graph, and nothing has run it anywhere but the numpy
reference backend. Calling it "Runnable" would claim both, and neither is claimed.

This is Qwen3 as the C engine implements it — the same model, written a second time as a
`ModelSpec`:

```
tokens -> embedding
       -> per layer:
            rms_norm -> q/k/v gemm
            -> reshape (rows*heads, head_dim) -> rms_norm(q_norm) -> reshape (rows, heads, head_dim)
            -> rope  (split-half)
            -> cache_append(k), cache_append(v)
            -> attention -> reshape -> o gemm -> add
            -> rms_norm -> gate/up gemm -> silu_mul -> down gemm -> add
       -> rms_norm(output_norm) -> lm_head gemm -> logits
```

Three Qwen3-specific details are worth naming, because each produces finite, plausible, wrong output
when it is got wrong:

- **QK-norm is per head, over `head_dim`.** The projection is `heads * head_dim` wide and the norm
  weight is `head_dim` long, so the graph reshapes to `(rows * heads, head_dim)` first — one head per
  row, which is the view the C engine's `n * n_head_` argument produces. Reducing over the projection
  row instead reduces over sixteen times as many values. This is why the
  [`reshape`](../architecture/kernel_abi_v1.md#ops) op exists.
- **Norm before RoPE**, on both q and k.
- **RoPE is split-half (NeoX)**: it pairs index `i` with `i + d/2`, not adjacent elements.

Two consequences of the ABI's shape vocabulary, stated rather than hidden:

- **The graph has a fixed call width, `rows`.** `reshape` takes a literal target shape, so the
  `(rows * heads, head_dim)` split has to know `rows` when the graph is built; there is no dynamic
  reshape and no `slice`. `Qwen3Config(rows=1)` is a decode step, and that is the configuration a
  decode loop uses. Prefilling a longer prompt means calling the graph once per token — which is what
  the C engine does for every position after the first anyway. An AOT backend would want exactly this
  split: a decode artifact at `rows=1` and a prefill artifact at a wider one.
- **The KV cache is one input pair per layer.** Each layer gets `blk.<i>.k_cache` /
  `blk.<i>.v_cache` of shape `(capacity, kv_heads, head_dim)`, matching the C engine's per-layer
  slabs. A single `(layers, capacity, kv_heads, head_dim)` input cannot work: `attention` reads a 3-D
  cache and there is no `slice` to take a layer out of a 4-D one. For Qwen3-0.6B that is 56 graph
  inputs — unwieldy, and the alternative is a new op this tree does not need yet.

```python
from pocketllm.architectures import build
from pocketllm.architectures.qwen3 import Qwen3Config

spec = build("qwen3", Qwen3Config(context=4096))   # 0.6B geometry by default
spec.cache.bytes_for()                              # what a phone budgets for the KV cache
```

The weights are named as GGUF names them — `token_embd.weight`, `blk.<i>.attn_q.weight`,
`blk.<i>.attn_q_norm.weight`, `output_norm.weight` — because that is what the C engine binds, and a
rename here would be a binding failure rather than a rebuild. A checkpoint with no `output.weight`
ties the head to the embedding table, as the C engine does; `tie_embeddings=True` builds that.

```python
from pocketllm.architectures import build
from pocketllm.architectures.toy import ToyConfig

spec = build("toy", ToyConfig(hidden=8, ff=16, vocab=32))
spec.graph.verify()          # every node checked against the ABI schemas
```

## What a real architecture has to decide

An architecture that is not a toy answers four questions, and each one maps to a part of the spec:

| Question | Where it lands |
|---|---|
| Which ops, in which order? | `spec.graph` — nodes over the declared [ABI vocabulary](../architecture/kernel_abi_v1.md#ops) |
| Which weights, and at what precision? | `spec.weights` — descriptors, with `quantizable` marking what the loader may hand over packed |
| How much context, at what dtype? | `spec.cache` — a capacity-parameterized [`CachePlan`](../architecture/execution.md#memory) |
| What is the decode step's shape? | the graph's input descriptors, which the executor uses to size buffers |

**Capacity is a parameter rather than a constant.** A phone and a card run the same model at very
different contexts, and a context limit is a *deployment* choice: the same checkpoint should serve a
4K chat and a 32K summarisation without being rebuilt. That is why `uniform_cache(layers, kv_heads,
head_dim, capacity=…)` takes the sequence length and returns descriptors, and why nothing is
allocated at that point.

**Paging is not a plan.** A paged KV cache is a backend's implementation of `cache_append` and
`attention`, not a different plan. The plan describes the logical shape — `(capacity, kv_heads, d)`
per layer — and a backend that pages maps that shape onto its own blocks. Keeping the plan logical is
what lets the reference backend and a card agree on what a `cache_append` *means* while disagreeing
completely about where the bytes are.

## `xing4_0`

**Status: planned, not written.** The previous tree carried a
runtime for Xing4.0-29B-A4B — MLA attention, 64 routed experts activated top-4 plus one shared, and
four residual streams per block mixed by a matrix hyper-connection — and it was CUDA-shaped in the
wrong places: it called `relic_core.kernels.cuda_loader.load_cuda_kernel()` directly from the model
code, which is the coupling this rebuild exists to remove.

Porting it means rebuilding it as an architecture against the ABI rather than moving the files: a
graph of declared ops, a weight table, a cache plan. Whether the ABI's vocabulary is sufficient for
it is the real test of the vocabulary — a hyper-connection that mixes four residual streams may need
an op the v1 list does not have, and the answer is to declare it *with a reference implementation in
the same commit*, not to reach around the ABI.

No model page for it exists here, and none will until it runs: a page that describes a checkpoint
this tree cannot execute is a promise, not documentation.

## Where the old model pages went

The previous tree's model pages — Xing4.0-29B-A4B, Ternary-Bonsai-2-27B, DeepSeek-V4 on GGUF Q2 —
recorded measurements taken against runtimes that have been rewritten. Their numbers belong with that
code, which is preserved on the `legacy` branch and, for the multi-card half, in
[RelicLLM](https://github.com/lvyufeng/RelicLLM). They are not carried forward: a number with no live
code behind it is still evidence about a checkpoint, but it is not documentation for this tree.