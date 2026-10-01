# The Horizon 地瓜 S600 backend

## Status: declared, not implemented

D-Robotics BPU, through the Horizon OpenExplorer toolchain. Structurally the
same as the QNN backend — a host compiles ahead of time, an arm64 board loads
the artifact — and it is deliberately the *second* backend of that shape, because
one AOT backend can be a special case and two establish the pattern.

## Offline compile is where the quantization happens

`hb_compile` / `hb_mapper` produce a `.hbm` model, and the int8 calibration is
part of that step, not a runtime conversion. That is why `CompileSpec.options`
carries `calibration` and `march` rather than a run-time flag: by the time the
board sees the artifact, the weight decisions are already made.

The board then loads the `.hbm` through `libhbrt4.so` and runs a forward pass.
`granularity=GRAPH`, same as QNN: the runtime owns the pass.

## Dependencies

For the board: `libhbrt4.so`, `libhbipm.so`, and a BPU device node. For the
offline step: the OpenExplorer Docker image (the toolchain is not usually
installed on the target).
