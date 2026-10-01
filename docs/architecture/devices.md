# Device targets

What each backend is for, what it declares, and what it is waiting for. Every backend here except
`reference` is **declared but not implemented**: it answers `available()` honestly and raises
`BackendNotImplementedError` naming its missing runtime from every session method. That is the v1
milestone — the shape is complete and the pending work is visible.

`pocketllm devices` prints the same table, with the availability of *this* host:

```console
$ pocketllm devices
reference  cpu       available                     eager only        17 ops
cpu        cpu       available                     eager only        16 ops
mps        mps       missing torch>=2.2 with an MPS device  eager only        16 ops
cuda       cuda      available                     stream_capture    17 ops
qnn        qnn       missing the QNN SDK (libQnnHtp*.so) and a Hexagon DSP device node  aot_compile       14 ops
horizon    horizon   missing the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device  aot_compile       14 ops
ascend     ascend    missing CANN (libascendcl.so) and an Ascend NPU (a /dev/davinci node)  stream_capture    14 ops
```

Read `available` as *the runtime is present*, not *the kernels are written*. `cuda` reports
available on the development host because torch is installed; it still has no kernels.

## The table

| Backend | Device kind | Graph path | Runtime it waits for |
|---|---|---|---|
| `reference` | `cpu` | none | **numpy — implemented** |
| `cpu` | `cpu` | AOT optional | numpy; later BLAS, KleidiAI on arm64 |
| `mps` | `mps` | none | `torch>=2.2` with MPS, macOS on Apple Silicon |
| `cuda` | `cuda` | `STREAM_CAPTURE` / `STEP` | `torch`; optionally `relic_core` |
| `qnn` | `qnn` | `AOT_COMPILE` / `GRAPH` | QNN SDK, a Hexagon DSP, `qai_appbuilder` for Android |
| `horizon` | `horizon` | `AOT_COMPILE` / `GRAPH` | OpenExplorer `libhbrt4.so`, a BPU device |
| `ascend` | `ascend` | `STREAM_CAPTURE` / `STEP` | CANN `libascendcl.so`, a `/dev/davinci*` node, aarch64 |

## reference — the normative numpy backend

Every declared op, at f32, on host memory. It is the oracle accelerated backends are measured
against, and it is the completeness rule: **an op may not be declared without an implementation here
in the same commit**, asserted by `tests/abi/test_reference_completeness.py`. Otherwise the op's only
implementation is on hardware most people do not have, and there is no way to tell a wrong fast
answer from a right one.

It is never the *preferred* backend. A 29B model on numpy is a ten-minute first token — not graceful
degradation, a hang with better manners.

## cpu — the first-class target, and the easiest mistake

`available()` is true on any host with numpy, which makes this the one stub that is *loadable*
everywhere — and the one easiest to mistake for done. It is not done: `open()` returns a session
whose every call raises.

The CPU is not the fallback here, it is a **v1 target**. Phone and edge deployments run on the host
CPU more often than on the NPU; a small model on a good CPU beats a large one that does not fit
anywhere. Its declared op set is narrower than the reference backend's on purpose: fast paths for the
ops that pay, dispatch falls back for the rest.

Its kernels must **not** simply wrap the reference ones, or the two become the same backend with two
names and the conformance check becomes circular. GEMM comes first, since it dominates decode — a
BLAS call through `ctypes` on x86, KleidiAI for the quantized formats on arm64.

## mps — Apple Silicon

Dense GEMM, attention, norms, RoPE, elementwise, embedding, cache and sampling, over f32/f16/bf16,
with quantized weights for the formats whose decoder is cheap to run on the way in. The one v1 device
whose runtime dependency is torch — and that is the design working, not a compromise: an install that
never selects `mps` never installs torch.

It declares **no graph path**. Metal exposes no stable graph-capture API through torch, and a capture
path that is not bit-identical to the eager one is worse than no capture path at all.

The probe is gated on `platforms=("darwin",)` because torch is installed on the x86_64 CUDA host,
where MPS can never run; without the gate, `pocketllm devices` would report an Apple GPU on a
machine that has none. The probe deliberately does not import torch, so it can only answer
"loadable", not "usable" — `open()` still verifies `torch.backends.mps.is_available()`.

## cuda — the first backend to write

The whole vocabulary, including formats `relic_core` does not currently consume.
`STREAM_CAPTURE` at `STEP` granularity: a decode step is a fixed-shape sequence and the host stays in
the loop for sampling and the next position, so the capturable set excludes the sampling ops.

**The open question, to answer before any code is written:** does this backend *wrap* `relic_core` or
*grow its own kernels*? It cannot do both. `relic_core.kernels.ops` has a resolver (`_auto_impl` /
`_resolve_impl`) and `pocketllm.kernels.dispatch` has another; two resolvers over the same op set is
exactly the coupling this rebuild exists to remove. Wrapping is fastest to a working card — the sm_75
kernels exist and are tested — at the cost of demoting `relic_core`'s resolver to "give me the kernel
for this op" rather than "decide which implementation to use".

**The sm_75 constraint.** The development cards are RTX 2080 Ti — Turing, compute capability 7.5. No
bf16 tensor cores, no Ampere-form `cp.async`, no FlashAttention-3. A kernel written for sm_80+ fails
to load with `no kernel image is available`, which is why the capability is pinned in `relic-core`
and must stay pinned.

## qnn — the phone NPU, and why the ABI splits compile from run

Qualcomm Hexagon through the QNN SDK / HTP. This is the primary phone target, and it is the reason
the ABI separates **offline compilation** from **device execution**.

An x86 *host* compiles a graph to a serialized QNN context binary; an arm64 *Android device* loads
that binary through `libQnnHtp*.so` and runs it. The compile host and the run host are different
machines of different architectures — an ABI that assumed they were the same would be unusable here.
That is what `CompileSpec` (host toolchain, target arch, artifact format) and
`GraphCapability(mode=AOT_COMPILE, granularity=GRAPH)` describe separately. `granularity=GRAPH`
rather than `STEP` because once the context is loaded the QNN runtime owns the whole forward pass.

**The delivery route is not decided.** `qai_appbuilder` (Python bindings) is easiest but adds a wheel
to ship and may not expose the context-binary path; the raw QNN C API through `ctypes` means no build
step and no extra wheel, keeping the "no native toolchain at install" rule intact, but more C to
wrap. A compiled shim would break that rule, so if neither route works the answer is to rethink the
approach rather than to add a build step.

**The probe imports nothing.** On a host with the SDK installed but no DSP, importing the QNN
bindings can *hang* rather than fail; `available()` looks for libraries and `/dev/fastrpc-cdsp` and
never imports a QNN module.

## horizon — 地瓜 S600

D-Robotics BPU through the Horizon OpenExplorer toolchain. Structurally the same as QNN — a host
compiles ahead of time, an arm64 board loads the artifact — and deliberately the *second* backend of
that shape, because one AOT backend can be a special case and two establish the pattern.

For Horizon, **offline compile is where the quantization happens**: `hb_compile` / `hb_mapper`
produce a `.hbm` model and the int8 calibration is part of that step, not a runtime conversion. That
is why `CompileSpec.options` carries `calibration` and `march` rather than a run-time flag: by the
time the board sees the artifact, the weight decisions are already made.

## ascend — 310B, not 910B

The 310B (Orange Pi AIpro / Atlas 200I) is **not** the 910B that the repository's host notes
describe: different generation, different CANN support, different `aclgraph` behaviour. None of the
recorded 910B facts transfer, and none of them can be verified from the x86_64 CUDA host. Treat
everything about this board as a claim to re-check in place.

`mode=STREAM_CAPTURE`, `granularity=STEP` — the CANN `aclgraph` model, which records a stream and
replays it, the same shape as a CUDA graph. AOT through `atc` to a `.om` artifact is possible and is
often what a 310B deployment wants, so `compile_spec()` describes it; it is the second path, not the
primary one.

`find_library("acl")` is deliberately **not** probed: it resolves to BSD's POSIX ACL library, which
is on essentially every Linux machine, and would report an Ascend NPU on a host that has none. The
real gates are `libascendcl.so` and `/dev/davinci0`.

One note from the 910B host, worth re-checking on the board: an ACL binary launched without CANN's
own `set_env.sh` does not fail — it *hangs* before `aclInit` returns.

## Adding a backend

A third-party backend does not need this tree to know it exists:

```toml
[project.entry-points."pocketllm.backends"]
mydevice = "my_package.backend:BACKEND"
```

The module must export `BACKEND`, an object satisfying `pocketllm.kernels.backend.Backend`. Two
obligations that the harness will hold you to: `available()` may not import your runtime, and
`capabilities()` is a promise — a declared op is one the conformance harness will run. See
[Backends and dispatch](backend_model.md#conformance).