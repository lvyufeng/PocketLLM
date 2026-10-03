# Model support

PocketLLM runs a model by building it as a **graph** against the
[kernel ABI](../architecture/kernel_abi_v1.md), rather than by treating every checkpoint as the same
Transformer. An *architecture* turns a config into a graph, the weights it reads and the cache it
needs — and stops there. It names no device and allocates nothing, which is what lets one executor
drive a toy model and, later, a 29B MoE.

## Status definitions

A status is a claim about evidence, so each one names the evidence it stands on.

| Status | What it means, and what proves it |
|---|---|
| **Runnable** | A real checkpoint generates tokens from a real checkpoint on the stated hardware, through an entry point in this tree. Nothing in *this* table is — the matrix below is the Python architectures, and the C engine's Qwen3 is not one of them. |
| **Scaffold** | The architecture builds and the executor runs it, but the "checkpoint" is a synthetic model whose purpose is to exercise the ABI. `toy` is the one instance. |
| **Planned** | The architecture is designed against the ABI and not yet written. |

## Support matrix

| Architecture | Structure | Device | Status |
|---|---|---|---|
| [`toy`](architectures.md#toy) | Embedding, one SwiGLU block, LM head | reference (numpy) | **Scaffold** — the executor's own smoke test |
| [`qwen3`](architectures.md#qwen3) | Qwen3 decoder: per-head QK-norm, split-half RoPE, SwiGLU MLP | reference (numpy) | **Scaffold** — the graph runs and its numerics are checked against a transcription of the C forward; no loader binds a checkpoint into it yet |
| `xing4_0` | MLA attention, 64-expert MoE, matrix hyper-connection | not yet | **Planned** |

**This table is the Python architectures, and the `reference` backend is the only implemented one.**
Every other Python backend is a declaration with a session that raises, so a row's Device column names
where its graph *can* run today, not where it is intended to.

Qwen3-0.6B does run — but through the C engine, not through the `qwen3` row above. The two are the
same model written twice against different interfaces: `src/model/qwen3.cpp` over the C engine's
`Backend`, and `python/pocketllm/architectures/qwen3.py` as a `ModelSpec` graph of ABI ops. The row
exists because that second implementation is now real and tested, and because the engine has a
[decode loop](../architecture/execution.md#decode) that drives it a token at a time. It is
**Scaffold** and not **Runnable** on the remaining half: no loader reads a `.gguf` into it, so the
weights that decode are synthetic.

The previous tree's model pages — Xing4.0-29B-A4B, Ternary-Bonsai-2-27B, the DeepSeek-V4 GGUF path —
measured code that has since been rewritten, and their numbers belong with that code; they are not
carried here.

## What an architecture is

```python
spec = build("toy", config)      # -> ModelSpec: a graph, weights, a cache layout
```

A `ModelSpec` carries three things:

- **a graph** — nodes naming [ABI ops](../architecture/kernel_abi_v1.md#ops), with the values they
  read and write. `spec.graph.verify()` checks it against the schemas before anything runs.
- **a weight table** — the names and descriptors of the tensors the caller must bind. The
  architecture says what it needs; the [loader](../architecture/execution.md#reading-a-checkpoint)
  is what supplies it from a `.gguf`.
- **a cache layout** — the KV cache's shape and dtype, which is what makes "does the checkpoint fit,
  and at what context length" an answerable question rather than a feeling.

Nothing in an architecture imports a backend. The dependency rule is `architectures/ → kernels` and
nothing else, and a test enforces it.

## Adding an architecture

An architecture under this tree ships as a package under `python/pocketllm/architectures/<name>/`; a
third-party one is registered through the `pocketllm.architectures` entry-point group, the same way a
[third-party backend](../architecture/devices.md#adding-a-backend) is. Either way it must:

1. Build a graph whose ops are all in the declared vocabulary — `spec.graph.verify()` is the check.
2. Declare its weights as descriptors, so the loader can bind them by name.
3. Declare its cache layout.
4. Import no backend and no device.

An architecture that needs an op the ABI does not declare cannot ship alone: a new op is declared in
`python/pocketllm/kernels/ops/` **with a reference implementation in the same commit**, because an op whose
only implementation is on hardware most people do not have has no way to tell a wrong fast answer
from a right one.