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
| `xing4_0` | MLA attention, 64-expert MoE, matrix hyper-connection | not yet | **Planned** |

One checkpoint **does** run today through an entry point in this tree, and it is not an architecture:
[**Qwen3 on the RDK S600**](s600_qwen3.md) runs on the Horizon Nash BPU through the `xlm`/`.hbm`
delegate (0.6B and 1.7B; the board's 384 MiB BPU region refuses 4B and 8B). It is documented
separately because it is a *delegate over a prebuilt artifact*, not a graph this tree builds — so it
is not a row above, for the same reason the C engine's Qwen3 is not.

**Every architecture here is a Python one, and none of the Python backends implements a kernel yet**,
so a checkpoint could not be run through this table even if an architecture existed for it.

Qwen3-0.6B does run — but not through any of this. It is implemented in the C engine
(`src/model/qwen3.cpp`) over a different backend interface, and it is deliberately not a row above:
an architecture here is a `ModelSpec` graph of ABI ops for `pocketllm.engine` to execute, and Qwen3
in C never builds one. Adding it to this matrix would claim a Python path that does not exist.

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