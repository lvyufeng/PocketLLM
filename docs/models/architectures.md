# Reference architectures

An architecture is a model's *structure*: it turns a config into a
[`ModelSpec`](../architecture/execution.md#the-model-ir) — a graph, a weight table and a cache plan —
and stops there. It names no device, allocates nothing, and imports no backend, so the same executor
drives every architecture in this tree.

Only one ships today, and it is deliberately not a model.

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