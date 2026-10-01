# The CPU backend

## Status: declared, not implemented

`available()` is true on any host with numpy, which makes this the one stub that
is *loadable* everywhere — and the one that is easiest to mistake for done.
It is not done: `open()` returns a session whose every call raises
`BackendNotImplementedError`. Loadable and implemented are different questions.

## What it is for

The CPU is not the fallback here, it is a **first-class v1 target**. Phone and
edge deployments run on the host CPU more often than on the NPU — a small model
on a good CPU beats a large one that does not fit anywhere. That is also why the
declared op set is narrower than the reference backend's: this backend will have
fast paths for the ops that pay, and dispatch falls back for the rest.

## The work

1. **`session.py`** — a real `BackendSession` over numpy: allocation, upload,
   the op call. The reference session is a working template; the difference is
   that this one must not simply wrap the reference kernels, or the two become
   the same backend with two names and the ABI check is circular.
2. **Kernels** — GEMM first, since it dominates decode. On x86 that is a BLAS
   call (openblas/MKL) through `ctypes`; on arm64 it is KleidiAI for the
   quantized formats and a BLAS or SVE2 path for the dense ones.
3. **Quantized GEMM** — the declared formats decode through
   `pocketllm.quant.formats`; a block walk that avoids materialising the weights
   as float32 is the whole point, since a decode step is bandwidth-bound on them.
4. **Threading** — one process owns one device, so the session owns the thread
   pool. Do not spin one up per call.

## Why "no torch"

The ABI core, the loader and the reference backend import numpy and nothing
else. A CPU backend that reached for torch would drag ~2 GB of wheel onto a
phone to do what a BLAS call does better. If a BLAS is absent the backend is
still correct, just slow — that is the bargain.

## Dependencies

`numpy` (base). Optionally a BLAS (`libopenblas.so` / `libblas.so`) for speed,
KleidiAI on arm64. None of them are required for `available()`.