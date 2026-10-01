# The CUDA backend

## Status: declared, not implemented

The one backend the current tree already half-is. Everything below is the plan,
not a description of what runs today.

## The open question, first

**Does this backend wrap `relic_core`, or grow its own kernels?** It cannot do
both. `relic_core.kernels.ops` has its own resolver (`_auto_impl` /
`_resolve_impl`); `pocketllm.kernels.dispatch` has another. Two resolvers over
the same op set is exactly the coupling this rebuild exists to remove, so the
choice has to be made deliberately and recorded here before any code is written.

- **Wrap `relic_core`.** Fastest to a working card: the sm_75 kernels already
  exist and are tested. The cost is that `relic-core` becomes a dependency of
  this backend, and its resolver must be demoted to "give me the kernel for this
  op" rather than "decide which implementation to use".
- **Grow its own.** Cleaner ABI boundary, and the op-level ABI is a better fit
  than `relic_core`'s dispatch. The cost is re-deriving the quantized GEMMs.

## What is declared

The whole vocabulary, including the formats `relic_core` does not currently
consume. `graph()` declares `STREAM_CAPTURE` at `STEP` granularity: a decode step
is a fixed-shape sequence, and the host stays in the loop for sampling and the
next position, so the capturable set excludes the sampling ops.

## The sm_75 constraint

The development cards are RTX 2080 Ti — Turing, compute capability 7.5. No
bf16 tensor cores, no `cp.async` in the Ampere form, no FlashAttention-3. A
kernel written for sm_80+ will fail to load (`no kernel image is available`),
which is why the capability is pinned in `relic-core` and must stay pinned.

## Dependencies

`torch` built against a CUDA toolkit matching the driver, and optionally
`relic_core`. Note the documented trap: `nvcc` on the x86 host is 13.0 while
`CUDA_HOME` points at 12.4, and a torch built against 12.4 hard-fails in
`cpp_extension` when it sees 13.0. Keep `CUDA_HOME` on a 12.x toolkit.
