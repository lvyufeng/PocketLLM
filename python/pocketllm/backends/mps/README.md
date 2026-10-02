# The MPS backend

## Status: declared, not implemented

Apple Silicon, through PyTorch's Metal backend. The one v1 device whose runtime
dependency is torch — and that is the design working, not a compromise: the ABI
core, the loader and the reference backend import numpy and nothing else, so an
install that never selects `mps` never installs torch.

## What is declared

Dense GEMM, attention, the norms, RoPE, the elementwise ops, embedding, the
cache, and the sampling ops — over `f32`/`f16`/`bf16`. Quantized weights are
declared for the formats whose decoder is cheap to run on the way in
(`q4_k`, `q5_k`, `q6_k`, `q8_0`, `iq4_nl`, `iq4_xs`).

## No graph path

`GraphCapability()` is the default — `supported=False`. Metal exposes no stable
graph-capture API through torch, and a capture path that is not bit-identical to
the eager one is worse than no capture path: the engine's eager path is correct,
and silently producing a different result would not be.

## Why the probe gates on `darwin`

`RuntimeProbe(modules=("torch",), platforms=("darwin",))`. torch is installed on
the x86_64 CUDA host, where MPS can never run; without the platform gate,
`pocketllm devices` would report an Apple GPU on a machine that has none.

## Dependencies

`torch>=2.2` built with MPS support, on macOS on Apple Silicon. `open()` must
still verify `torch.backends.mps.is_available()` — the probe deliberately does
not import torch, so it can only answer "loadable", not "usable".
