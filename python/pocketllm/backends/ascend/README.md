# The Ascend 310B backend

## Status: declared, not implemented

The 310B — Orange Pi AIpro / Atlas 200I. **Not the 910B** the host notes in
`CLAUDE.md` describe: different generation, different CANN support, different
`aclgraph` behaviour. The recorded 910B facts do not transfer, and none of them
can be verified from the x86_64 CUDA host. Treat everything here as a claim to
re-check on the board.

## Open question

**Which board, and which CANN version?** The declaration names
`Ascend310B1` in its `atc` compile options, which is the second-generation
naming; the board should be confirmed before that is trusted. Whether
`aclgraph` exposes the same capture semantics on 310B as it does on 910B is the
part to verify first, since the graph declaration depends on it.

## The capture path

`mode=STREAM_CAPTURE`, `granularity=STEP` — the CANN `aclgraph` model, which
records a stream and replays it, the same shape as a CUDA graph. AOT is
*possible* through `atc` to a `.om` artifact and is often what a 310B deployment
wants, so `compile_spec()` describes it; it is the second path, not the primary
one.

## Why `acl` is not probed by name

`RuntimeProbe(libraries=("ascendcl", "nnopbase"), ...)`. `find_library("acl")`
resolves to BSD's POSIX ACL library, which is on essentially every Linux
machine, so probing it would report an Ascend NPU on a host that has none. The
real gates are `libascendcl.so` and `/dev/davinci0`.

## Dependencies

CANN (`libascendcl.so`, `nnopbase`), a `/dev/davinci*` node, aarch64. For the
`atc` offline path, the CANN toolkit's compiler on the build host.

Note from the 910B host, worth re-checking: an ACL binary launched without
CANN's own `set_env.sh` does not fail — it *hangs* before `aclInit` returns.
