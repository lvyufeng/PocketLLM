# Execution

This page covers what happens between a graph and a run: choosing and owning a device, planning which
parts a backend takes whole, running the rest op by op, and deciding how much memory that needs. It
also covers the other end of the pipe — reading a GGUF checkpoint — because the loader is what turns
bytes on disk into the descriptors the execution layer names.

## The session

`EngineSession` is where "one process owns one device" stops being a slogan and becomes a
constructor. It is opened with a device and produces exactly one `BackendSession`; it never opens a
second, spawns a worker, or coordinates with another process. A model that does not fit is quantized
further — a decision the caller makes *before* the device is chosen, not something this layer does
behind their back.

Device selection is a policy, and the policy differs by workload:

```python
SessionPolicy.for_run()      # allow_reference_fallback=True
SessionPolicy.for_serve()    # allow_reference_fallback=False
```

The asymmetry is deliberate and is the only place the engine has an opinion about performance. For
`run`, a slow answer beats no answer — a 29B model on numpy is slow but visible. For `serve`, a
request that will not return in time is worse than a refusal naming the device that would have
worked. Stated once here rather than at every call site.

The session imports no device runtime: it asks the registry, which asks each backend's probe, which
is a filesystem question.

## Planning

A backend with a graph path does not want the whole model handed to it. A CUDA graph captures a
decode *step*; a QNN context binary is the whole graph; and both have ops they cannot absorb at all
— a sampler reads a random variate, and a host-side embedding gather has no device kernel.

So before execution the planner splits the node list into **regions** and asks the backend's declared
`GraphCapability` about each one:

```python
ExecutionPlan.regions        # contiguous runs, captured or eager
ExecutionPlan.captured_nodes # how many nodes the graph path took
```

The split is deliberately conservative and deliberately boring. A region is a maximal contiguous run
of nodes whose ops are all in `captures` and which fits `max_nodes`; anything else is a one-node
eager region. Two consequences are worth stating, because they are why it is not cleverer:

- **Adjacency is respected.** A capturable op on either side of a sampling step is not merged across
  it, even though the two could in principle be captured together: replaying a capture does not
  re-run the sampler, so the region's output would silently be stale.
- **Falling back is not a failure.** A backend without a graph path produces exactly one eager
  region, and `captured_nodes == 0` is the normal case for the reference backend and for a CPU, not
  an error to report.

The planner reads declarations only — no device, no session — so a plan for a phone can be computed
and printed on a CUDA host. That is what makes the `ops` style of debugging work for hardware that is
not present.

## Executing

`Executor` is the always-works path: resolve each node through the dispatcher, allocate its output
buffers, call the chosen backend's `run`, and release the buffers of values that have just died. It
contains no device-specific code and no assumption that the backend can capture anything.

Two properties are load-bearing.

**Resolution is per node, not per graph.** A model on a phone does not run entirely on the NPU: the
sampling tail is on the CPU, and an embedding gather may be too. Resolving per node is what lets one
graph span several devices, and `ExecutionTrace.crossed_a_device` reports where it did — a host
round-trip was worth knowing about in the old tree and is worth knowing about here.

**Nothing is copied that did not have to be.** A graph input's tensor is bound by reference; only a
node's *outputs* are allocated. A value nothing reads is freed at its own birth. The one unavoidable
copy is a cross-device one, and it is named in the trace.

## Memory

A decode step's graph is *long and thin*: every value is produced, read once or twice, and never read
again. Allocating one buffer per value would make the high-water mark the sum of every activation in
the model. Measuring when each value is born and when it dies — `plan_memory` — lets a buffer be
handed back the moment its last reader has run, which turns that sum into the width of the widest
single layer.

The arena's policy is one line: a request of *n* bytes is served from the pool of freed *n*-byte
buffers if one is there, and allocated otherwise. **Exact size, not first-fit** — a subview of a
larger buffer would alias a value the plan thinks is dead, and the plan is the thing that has to be
right.

The plan is not a guess. It is produced by *running the arena* against a session that counts instead
of allocating, so the numbers the executor observes and the numbers the plan reports cannot drift:
there is one policy, and the plan is a measurement of it.

## Reading a checkpoint

The loader turns a `.gguf` file into descriptors and buffers with **no device runtime involved**. It
depends on `pocketllm.quant` and numpy, and not on torch — which is what lets a phone install read a
checkpoint without the training stack.

```
pocketllm/loader/gguf/
├── reader.py            the file format: header, metadata KV, tensor directory
├── bundle.py            the parsed file: metadata + tensor directory
├── quant_types.py       which GGML type id a tensor names
├── tensor_reader.py     bytes -> numpy arrays, one decoder per block family
├── quantized_tensor.py  a packed tensor: blocks + geometry + device
├── quantized_loader.py  walk the directory and upload through a session
├── host_array.py        the one numpy <-> Buffer boundary
└── vendor/ggml-common.h the GGML codebooks, vendored with provenance
```

`host_array.upload(session, array, desc)` is the single boundary function that moves data onto a
device. It takes a **session**, not a device: where a tensor ends up is the session's business, and a
host device can hand back a `memoryview` over its own bytes, so the "upload" costs nothing on a CPU
or phone target.

### The quantization decoders

`pocketllm.quant` is a **leaf**: numpy and the vendored table header, and nothing else in the
package — not `kernels`, not `loader`, not `backends`. The placement is load-bearing. Two very
different layers need to turn a GGUF block into weights: the *loader*, which must not have to know
the reference backend exists, and the *reference backend*, which has to dequantize a weight to
compute with it. If the decoders lived in either one the other would import it, and the
`backends/ → kernels` rule would be broken. A shared leaf with no edge back into the package is the
only shape that keeps both honest.

The formats implemented are `iq4_nl`, `iq4_xs`, `iq1_m`, `iq2_xxs` / `iq2_xs` / `iq3_xxs`, `q2_k`
through `q6_k`, and `q8_0`. The ternary packs (`ptq1_0`, `pq2_0`) are decodable — the loader
addresses them — but no kernel consumes them yet, which is `runtime_id=None` in
[the format table](kernel_abi_v1.md#tensors-buffers-devices).

### The vendored header

The loader used to reach into `relic_core.__file__` to find `ggml-common.h`, which meant reading a
checkpoint required the kernel library to be installed. It is vendored now, with a provenance note
recording the upstream revision and the MIT license, and `relic_core` is no longer an import of this
package. Non-`.py` payload needs an explicit line in `pyproject.toml`'s `package-data` and in
`MANIFEST.in`, or an installed wheel's loader dies at the point it reads a codebook — a failure that
does not appear in a source checkout at all.

## The model IR

An architecture turns a config into a `ModelSpec` — a graph, the weights it reads, and the cache it
needs — and stops there. It names no device, allocates nothing, and does not import a backend. That
is what lets the same executor drive `toy` and, later, `xing4_0`. See
[Reference architectures](../models/architectures.md).