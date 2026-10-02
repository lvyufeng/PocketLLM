# Backends and dispatch

A *backend* is a device implementation: it allocates memory on a device and runs ops there. It is
what `--backend cuda` names and what `--device cuda:1` selects, and it is the object the engine talks
to after the architecture has produced a graph.

## One process, one device

The rule the whole design reduces to: **one process owns one device**. The ABI has no rank, no
collective and no second buffer space, and there is nothing in the tree that launches a second
process. A checkpoint that does not fit is quantized further -- Q4 → Q2 → IQ2 → IQ1 → ternary -- and
never split. The consequence in the code is that `open()` returns a `BackendSession` holding
*one* `Device`, and there is no API by which a caller could ask for two.

`EngineArgs` reflects this directly: it has no `tensor_parallel_size`, no `tensor_parallel_rank` and
no `device_ids` list, and the environment variables that used to carry them
(`TENSOR_PARALLEL_*`, `POCKETLLM_DEVICE_IDS`) are gone. A leftover field is how a deleted feature
stays half-alive.

## Devices are named, not enumerated

```python
Device(kind="cuda", index=1)    # "cuda:1"
Device(kind="qnn")              # "qnn", index 0
```

`kind` is an **open string**. `KNOWN_DEVICE_KINDS` lists the kinds this tree ships backends for, but
it is documentation and default ordering, not a whitelist: a third-party backend registers whatever
kind it runs on and `pocketllm --help` will list it. This is deliberate and load-bearing -- "more
devices later" is a requirement of this project, not an aspiration, and a closed enum would mean
every new device needed a change to the ABI.

`--device`'s argparse choices are read from the backend registry at parser-construction time
(`pocketllm.api.device_kinds()`), so the two cannot disagree: there is no second list of device names
to update.

## Discovery

Two mechanisms, and the split matters.

**In-tree: a static table.** `python/pocketllm/backends/registry.py` maps a name to a module and a factory.
The table holds *strings* -- importing `pocketllm.backends` loads no backend, and `get()` imports
exactly one. A bare `pip install pocketllm` therefore imports no backend module at all.

What a *listing* does import is each backend's declaration module, because there is no way to read an
out-of-tree backend's capabilities without it. That makes the rule that keeps `pocketllm devices`
working on a bare machine narrower and worth stating exactly:

> **A declaration module imports no runtime, and its probe is a filesystem question.**

That is why the probes live in `RuntimeProbe` and why a backend imports numpy (if at all) inside its
session, not at module scope.

**Out-of-tree: entry points.** `pocketllm.backends` is an entry-point group, so a third party ships
a backend as a plugin without this tree knowing it exists:

```toml
[project.entry-points."pocketllm.backends"]
rknn = "my_package.backend:BACKEND"
```

A broken entry point is skipped rather than raised. The listing path is the one an operator runs
*because* something is wrong, and one third-party package with a bad metadata line must not take it
down.

## The availability probe

```python
class RuntimeProbe:
    modules=("torch",)                    # importlib.util.find_spec -- locates, does not execute
    libraries=("libQnnHtp.so",)           # ctypes.util.find_library -- reads the loader's cache
    device_nodes=("/dev/davinci0",)       # a path check
```

`Backend.available()` answers *can this backend be loaded here*, and it must not import the runtime
to find out. Three reasons, all practical:

1. A probe that imports torch to ask whether torch is installed pays torch's import cost on every
   listing -- on the machines that have it, which are exactly the ones where `devices` is fast today.
2. On a host with the QNN SDK installed but no DSP, importing the bindings can **hang** rather than
   fail.
3. The command has to work when everything is broken. That is its purpose.

`available` and `implemented` are different questions and the code keeps them apart. On the
development host, `cuda` reports `available` because torch is present, and it still has no kernels --
which is why `pocketllm run` refuses. Two distinct errors exist for the two states:
`BackendUnavailable` ("the runtime is not here") and `BackendNotImplementedError` ("the runtime is
here; this backend has not been written"). They want different fixes, so they are not the same
exception.

## Resolution

An op is resolved against the backends for a device:

```python
Dispatcher(backends, allow_reference_fallback=True).resolve("gemm_quant", args, Device("cuda"))
```

The rule, in order:

1. A backend whose `device_kind` does not match the device is out -- **except** the reference
   backend, which is a candidate for any kind.
2. A backend that is not `available()` is out.
3. A backend that does not declare the op is out.
4. A declared capability must admit the operands' dtypes and quant formats, and its `accepts`
   predicate (if any) must hold.
5. Among survivors, the lowest `rank` wins; ties are broken by listing order, which runs from "always
   present" to "most specialised".

The result records `is_reference`, which tells the engine a host round-trip is happening. That
matters twice: it is a performance signal, and it disqualifies the region from capture.

`explain()` returns the same computation as **data** -- every candidate with its accept/reject
reason. Resolution reads no bytes and opens no device, so `pocketllm ops --op gemm_quant --device
qnn` answers for a phone's backends from a development host:

```console
$ pocketllm ops --op gemm_quant --device cuda
gemm_quant on cuda -> cuda (100)
  rejected cpu: device kind 'cpu' != 'cuda'
  rejected mps: device kind 'mps' != 'cuda'
  rejected reference: a better candidate was found
```

## The reference backend

`pocketllm.backends.reference` is numpy, host memory, and every declared op at f32. It is the
**normative** implementation: a new op may not be declared without a reference implementation in the
same commit, which `tests/abi/test_reference_completeness.py` asserts by set inclusion. It is the
oracle every accelerated backend is checked against, and it is why the conformance harness can run
on a laptop.

It is always a candidate but never a *preference*. Whether it may actually be used is a session
policy:

| Command | `allow_reference_fallback` | Why |
|---|---|---|
| `run` | **on** | An answer beats no answer; a slow first token is visible and fixable |
| `serve` | **off** | Silently running a 29B model on numpy is a ten-minute first token, not graceful degradation |

## What each backend ships in v1

| Backend | Device kind | Graph path | Runtime |
|---|---|---|---|
| `reference` | `cpu` | none | numpy -- **implemented** |
| `cpu` | `cpu` | AOT optional | numpy; later BLAS/oneDNN, arm64 KleidiAI |
| `mps` | `mps` | none | `torch>=2.2` with MPS |
| `cuda` | `cuda` | `STREAM_CAPTURE` / `STEP` | `torch`; optionally `relic_core` |
| `qnn` | `qnn` | `AOT_COMPILE` / `GRAPH` | QNN SDK, `qai_appbuilder` |
| `horizon` | `horizon` | `AOT_COMPILE` / `GRAPH` | Horizon OpenExplorer, `libhbrt4.so` |
| `ascend` | `ascend` | `STREAM_CAPTURE` / `STEP` | CANN, `libascendcl.so` |

Everything but `reference` is a **stub this milestone**: it declares its ops, answers `available()`
honestly, and raises `BackendNotImplementedError` naming its missing runtime from every session
method. See [Device targets](devices.md) for what each one is waiting for.

## Conformance

A backend is conformant when the harness can show four things:

1. Every op in `capabilities()` runs, parameterized over the declared domain.
2. Its numerics match the reference within tolerance.
3. Its capture path is bit-identical to its eager path -- **skipped** when
   `GraphCapability.supported` is `False`, and a skip is not a pass.
4. It declares nothing outside `capabilities()`, so dispatch never routes an op it cannot run.

Harness location: `tests/backends/conftest.py` and `tests/backends/test_backend_conformance.py`.