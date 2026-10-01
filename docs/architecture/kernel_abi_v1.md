# The kernel ABI

**This page is normative.** It describes what `pocketllm.kernels` declares and what a backend must
implement to be driven by the engine. Where this page and the code disagree, the code is the bug --
but a backend that satisfies this page and not the code is a backend the code has failed.

The ABI is **op-level with an optional graph path**: a backend implements ops one at a time, and may
*additionally* offer to take a whole region as a captured or compiled graph. The op path is the
contract; the graph path is an optimization a backend may decline, and declining is correct rather
than degraded.

## What the ABI is not

It is not a tensor framework and not a thin wrapper over one. It imports **nothing** -- no numpy, no
torch, no `relic_core`, no I/O. A tensor is *described*, not constructed by a library; a device is a
name; an op is a record. `tests/test_package_boundaries.py` enforces this on every module under
`pocketllm/kernels/`, and `pocketllm.kernels` is the only package in the tree with that rule.

The reason is the mission. If the ABI needed numpy to read a shape, then a phone build could not be
trimmed of numpy; if it needed torch, the wheel could not install on a phone at all.

## Tensors, buffers, devices

```python
Device(kind="cuda", index=2)          # a kind and an index; `cuda:2`
TensorDesc(shape=(1, 4096), dtype=DType.F16)      # unpacked f16
TensorDesc(shape=(64, 4096), quant=QUANT_FORMATS["iq4_nl"])   # packed blocks
Tensor(desc, buffer)                  # a desc plus where its bytes live
```

Three things are worth stating precisely.

**`DType` and `QuantFormat` are different kinds of thing.** `DType` is the type of one *unpacked*
element. `QuantFormat` is a *packed block* format, where a run of weights shares a scale and the
values are recovered by a decoder. A tensor carries **exactly one** of the two. That is what lets a
quantized weight stay packed all the way to a kernel instead of being expanded to a dense copy on
the way in -- a correctness statement about the hot path, not an optimization detail.

A `QuantFormat` carries the geometry a kernel needs, not the file layout:

```python
QuantFormat(name="iq4_nl", file_type_id=20, block_elems=32, block_bytes=18, runtime_id=20)
```

`file_type_id` is GGUF's identifier for a tensor. `runtime_id` is the compact number a raw-block
kernel switches on, and it is **not** the same number -- `iq2_xxs` is file id 16 and runtime id 0.
They are separate fields because they answer separate questions, and conflating them has already
cost this project once. `runtime_id=None` means the loader can decode the format and no kernel
consumes it yet, which is the honest state of the ternary packs; the same category `iq4_nl` was in
before it graduated.

**`Device.kind` is an open string, not an enum.** A backend registers whatever kind name it runs on,
so a third-party engine can add `s600` or `rknn` without this ABI changing. `KNOWN_DEVICE_KINDS` is
documentation and default ordering, not a whitelist. A closed enum would mean a new device needed a
change to the ABI, which is precisely the coupling the rebuild exists to remove.

**`Buffer` is a device's memory, and it is a protocol.** `address()` is an opaque device address,
`host_view()` may return `None` -- and `None` is the honest answer for a discrete card, not a
failure. A host device hands out a `memoryview` over its own bytes, which is what lets the loader
upload without a copy on a CPU or phone target where the "upload" is a no-op.

## Ops

An op is declared once, in the ABI, and implemented by backends:

```python
OpSchema(
    name="gemm_quant",
    args=(ArgSpec("x", TENSOR, shape=("*", "K")),
          ArgSpec("w_blocks", TENSOR, role="weight"),
          ArgSpec("bias", TENSOR, optional=True)),
    returns=(ArgSpec("y", TENSOR, shape=("*", "N")),),
    dtypes=frozenset({F32, F16, BF16}),
    quants=frozenset({IQ4_NL, Q4_K, Q5_K, Q6_K, ...}),
    shape_rule=_gemm_quant_shape,
    semantics="...",
)
```

A backend does **not** restate the shape of a GEMM. It declares *that it implements* `gemm_quant`
and over what domain, and `OpSchema.infer` is the single statement of what `gemm_quant` means --
argument names, which are tensors, admissible dtypes and quant formats, and how an output shape is
inferred. Every backend gets shape inference for free, and all of them agree because there is only
one rule.

Shape inference uses a small symbolic vocabulary. In a schema, `shape=("m", "k")` names the first
dimension `m` and the second `k`; two arguments that both say `k` must agree at call time, and a
disagreement raises `ShapeError` naming the dimension rather than merely failing.

**The v1 vocabulary is 19 ops.** They are grouped by family under `pocketllm/kernels/ops/`:

| Family | Ops |
|---|---|
| `gemm` | `gemm`, `gemm_quant` |
| `attention` | `attention` |
| `moe` | `moe_ffn` |
| `rope` | `rope` |
| `norm` | `rms_norm`, `layer_norm` |
| `elementwise` | `add`, `mul`, `silu_mul`, `softmax`, `logits_temperature` |
| `embedding` | `embedding` |
| `sample` | `topk_sample`, `argmax` |
| `cache` | `cache_append`, `cache_truncate` |

`gemm_quant`, `moe_ffn` and `embedding` accept quantized weights; the rest are dense. Every op is
f32/f16/bf16.

## Backends

```python
class Backend(Protocol):
    name: str
    device_kind: str
    version: str

    def available(self) -> bool: ...              # cheap probe; must NOT import the runtime
    def capabilities(self) -> tuple[Capability, ...]: ...
    def graph(self) -> GraphCapability: ...
    def compile_spec(self) -> CompileSpec | None: ...
    def open(self, device, *, options=None) -> BackendSession: ...
```

Two obligations are easy to get wrong and are checked:

- **`available()` must not import the runtime.** It may use `find_spec`, `find_library`, a device
  node or an environment variable. `pocketllm devices` runs *because* something is broken, and a
  probe that raises on a host without the runtime takes the diagnostic down with it. The probe
  lives in `pocketllm.backends.base.RuntimeProbe` for exactly this reason.
- **A backend declares only what it implements.** A capability is a promise the conformance harness
  will keep you to: `tests/backends/test_backend_conformance.py` parameterizes over
  `capabilities()` and runs each declared op.

A `Capability` may carry an `accepts` predicate for a domain the schema cannot express -- "decode
only (one row)", "K a multiple of 256", "no mask". `rank` is a documented relative preference among
equal matches, lower wins; it is a preference, not a measurement.

### The reference backend is normative

`pocketllm.backends.reference` implements **every** declared op in numpy at f32, and a new op may not
be declared without a reference implementation in the same commit.
`tests/abi/test_reference_completeness.py` asserts `OPS.names() ⊆ reference.capabilities()`, which
makes the oracle complete by construction rather than by discipline.

It is always a dispatch candidate, so a wrong device gets a correct-but-slow answer instead of "no
backend" -- but it is never a *preference*. The session policy `allow_reference_fallback` is **on for
`run`** (an answer beats no answer) and **off for `serve`** (silently running a 29B model on numpy is
a ten-minute first token, not graceful degradation).

## The optional graph path

```python
GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE,
                captures=frozenset({"gemm_quant", "rms_norm", ...}),
                granularity=RegionGranularity.STEP, max_nodes=4096)
```

A backend may accelerate a region instead of an op, in one of two ways:

- **`STREAM_CAPTURE`** -- record a stream once and replay it with new inputs of the same shape (CUDA
  Graphs, CANN `aclgraph`).
- **`AOT_COMPILE`** -- compile ahead of time into a device artifact (QNN context binaries, Horizon
  `.hbm`).

`granularity` is `STEP` (one decode step at a time, the host stays in the loop) or `GRAPH` (the whole
model at once, the host is out of the loop). A region containing an op the backend does not list in
`captures`, or one resolved to the reference backend, is **not capturable**, and the engine runs it
eagerly. `capture()` and `compile_graph()` return `None` to mean "fall back", never "error".

The AOT case is why `CompileSpec` exists as a separate record from the run path. An NPU toolchain is
typically host-x86-only while the device is arm64, so the ABI never assumes the compile host equals
the run host: `CompileSpec` says what the offline step needs (host toolchain, target arch, artifact
format), and the produced artifact is loaded by device-side code at `open()` time.

## Resolution

Resolution is a **pure function over declarations**. It reads schemas and capabilities, never a
device, so a given tree resolves identically on two machines and a resolution can be tested on a
host with no accelerator at all.

That is what makes `pocketllm ops` possible from a development host:

```console
$ pocketllm ops --op gemm_quant --device qnn
```

`Dispatcher.explain` returns each candidate's accept/reject reason **as data**. The old tree's
dispatch raised "unsupported" with no trace, and finding out which layer refused a call was the
expensive part of adding a format. Here the trace is the API.

## Where this is going

The ABI is versioned by this page, not by a number in the code. What would change it: a second
device per process (out of scope by rule), a dynamic shape the schema vocabulary cannot express, or
an op whose domain is not a dtype/quant set and a shape rule. None of those is planned.