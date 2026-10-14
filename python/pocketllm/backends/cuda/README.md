# The CUDA backend

## Status: declared, not implemented

The one backend the current tree already half-is. Everything below is the plan,
not a description of what runs today.

## The open question, first

**Does this backend wrap an external operator library, or grow its own kernels?**
It cannot do both. A wrapped library has its own resolver; `pocketllm.kernels.dispatch`
has another. Two resolvers over the same op set is exactly the coupling this
rebuild exists to remove, so the choice has to be made deliberately and recorded
here before any code is written.

- **Wrap an external library.** Fastest to a working card: the sm_75 kernels may
  already exist and be tested. The cost is that this backend takes a dependency,
  and the library's resolver must be demoted to "give me the kernel for this op"
  rather than "decide which implementation to use".
- **Grow its own.** Cleaner ABI boundary, and the op-level ABI is a better fit
  than an external dispatch. The cost is re-deriving the quantized GEMMs.

## What is declared

The whole vocabulary, including the formats no external consumer currently
handles. `graph()` declares `STREAM_CAPTURE` at `STEP` granularity: a decode step
is a fixed-shape sequence, and the host stays in the loop for sampling and the
next position, so the capturable set excludes the sampling ops.

## The sm_75 constraint

The development cards are RTX 2080 Ti — Turing, compute capability 7.5. No
bf16 tensor cores, no `cp.async` in the Ampere form, no FlashAttention-3. A
kernel written for sm_80+ will fail to load (`no kernel image is available`),
which is why the capability is pinned and must stay pinned.

## Dependencies

`torch` built against a CUDA toolkit matching the driver. Note the documented
trap: `nvcc` on the x86 host is 13.0 while `CUDA_HOME` points at 12.4, and a
torch built against 12.4 hard-fails in `cpp_extension` when it sees 13.0. Keep
`CUDA_HOME` on a 12.x toolkit.