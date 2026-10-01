# The QNN backend

## Status: declared, not implemented

Qualcomm Hexagon NPU, through the QNN SDK / HTP. This is the primary phone
target and the reason the ABI splits **offline compilation** from **device
execution**.

## The AOT shape

An x86 *host* compiles a graph to a serialized QNN context binary; an arm64
*Android device* loads that binary through `libQnnHtp*.so` and runs it. The
compile host and the run host are different machines and different
architectures — an ABI that assumed they were the same would be unusable here.
That is what `CompileSpec` (host toolchain, target arch, artifact format) and
`GraphCapability(mode=AOT_COMPILE, granularity=GRAPH)` describe separately.

`granularity=GRAPH` rather than `STEP`: once the context is loaded the QNN
runtime owns the whole forward pass, and the host is out of the loop between
steps.

## The delivery question, not yet decided

Two routes, with different consequences:

- **`qai_appbuilder`** — the Python bindings. Easiest, but it is another wheel
  to ship and it may not expose the context-binary path.
- **The raw QNN C API through `ctypes`** — no build step, no extra wheel, and it
  keeps the "no native toolchain at install" rule intact. More C to wrap.

A compiled shim would break that rule, so if neither route works the answer is
to rethink the approach rather than to add a build step.

## Why the probe does not import anything

On a host with the SDK installed but no DSP, importing the QNN bindings can
*hang* rather than fail. So `available()` looks for libraries (`find_library`,
which reads the loader's cache) and device nodes (`/dev/fastrpc-cdsp`) and never
imports a QNN module.

## Dependencies

The QNN SDK, a `libQnnHtp*.so` for the target HTP version, and a Hexagon DSP.
For the offline step, `qnn-context-binary-generator` on an x86 host.
