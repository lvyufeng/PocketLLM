# The reference backend

Numpy, host memory, every op. This is the backend that is always present and
always correct, and it is *normative*: the answer it produces is the definition
of the op, and every accelerated backend is measured against it.

## Why it exists

Two jobs, and both are load-bearing.

**It is the oracle.** A kernel that produces a different number is wrong until
shown otherwise, and "different from what" needs an answer. `tests/backends/`
runs every declared op on every backend and compares against this one.

**It is the completeness rule.** An op may not be declared in
`pocketllm/kernels/ops/` without an implementation here, in the same commit.
Otherwise the op's only implementation is on hardware most people do not have,
and there is no way to tell a wrong fast answer from a right one.
`tests/abi/test_reference_completeness.py` fails until the kernel exists.

## Why it is never the chosen backend

Dispatch sorts it last among equal candidates, and the serving path turns it off
entirely. A 29B model on numpy is a ten-minute first token — that is not
graceful degradation, it is a hang with better manners. `run` allows the
fallback because a slow answer beats no answer interactively; `serve` does not.

## What it does not have

- **No graph path.** `compile_graph` and `capture` both return `None`, which the
  engine reads as "run eagerly", never as an error.
- **No device memory.** `Buffer.host_view()` always succeeds, so a backend that
  assumes a discrete card's `None` will not be exercised here.
- **No quantized kernel.** Packed weights are decoded through
  `pocketllm.quant.formats` before the multiply, which is why the reference is
  correct for a format with no fast kernel at all.

## Layout

| File | What it is |
|---|---|
| `session.py` | The ABI translation: tensors to numpy and back, allocation, the op call |
| `kernels.py` | One function per op — the executable form of each schema's `semantics` |
| `dtypes.py` | ABI dtype ↔ numpy, including bfloat16, which numpy does not have |

## Measuring a backend against it

```bash
python -m pytest tests/backends -q
```

A backend that declares an op it cannot run fails; a backend whose numerics
drift fails; a backend with no runtime skips, with the reason it skipped.