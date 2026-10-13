"""The greedy fast path is the same answer as `forward` + `argmax`, on the GPU.

`Engine.next_token` calls `pocketllm_forward_argmax`, which reads the greedy id
from the device's `Backend::argmax` and never brings the vocabulary to the host.
`Engine.forward` + `Engine.argmax` is the other route: all 151,936 floats cross
the bus and the same `Backend::argmax` runs on the host.  The two cannot
disagree by construction -- both resolve a tie to the lowest index -- and the
host shell routes every greedy decode through the fast one (`cli.py`'s `run`
loop, `native_backend.py`'s `_advance`, which both serving loops drive).

**Nothing else in the suite pins that identity.** The token gate and the other
native tests decode through `forward` + `argmax`, which is the route the fast
path replaces, so a change that quietly stopped routing greedy decoding through
`next_token` -- or that changed what `next_token` returns -- would leave every
existing test green and every existing answer unchanged on the slow route while
the fast route moved underneath it.  The whole value of the fast path is that it
is *the same answer* for four bytes of readback instead of 607 KB, so the
identity is the thing to hold, and it is held here on the checkpoint whose
greedy output is already pinned against llama.cpp elsewhere.

**The saving is host-side, and deliberately not asserted here.**  Measured with
a preallocated-buffer decomposition (no Python list in either arm): a decode
step is 8.60 ms through `pocketllm_forward` and 8.61 ms through
`pocketllm_forward_argmax`, so the transfer and the device argmax together are
under 0.04 ms/step.  What the wiring removes is the 151,936-element Python list
that `Engine.forward` builds and the second ctypes array `Engine.argmax` builds
from it, which the two routes measure as a ~37 ms/step difference on a quiet
window (1.7B 45.714 ms -> 8.682 ms; 0.6B 44.744 ms -> 6.472 ms, interleaved,
start_load 6.05 end_load 11.08).  A timing assertion on a shared host would be
a flaky test, so the number lives in this docstring and the PR body and the
test asserts the contract: identical tokens.

The larger checkpoint is the one whose decode runs the k-split kernel as well
(`test_decode_split.py`), so its identity holds across the decode GEMM the fast
path shares with the rest of the tree; the smaller one is the checkpoint the
token gates compare against llama.cpp.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm import native

#: Both q4_k_m checkpoints, keyed for the parametrization id.  The 1.7B is the
#: checkpoint `test_decode_split.py` decodes on; the 0.6B is the one the token
#: gates pin against llama.cpp.
CHECKPOINTS = {
    "1_7b": pathlib.Path("/mnt/data1/models/Qwen3-1.7B-Q4_K_M.gguf"),
    "0_6b": pathlib.Path("/mnt/data1/models/qwen3-0.6b-q4_k_m.gguf"),
}

#: `"The capital of France is"`, the prompt the other native tests use, as ids
#: so a tokenizer regression cannot read as a routing regression.
PROMPT = [785, 6722, 315, 9625, 374]

#: Greedy steps per route.  Sixteen matches `test_decode_split.py`, and is well
#: past the point where a broken route would visibly drift.
STEPS = 16


def _cuda_reason() -> str | None:
    """Why `cuda` cannot be used here, or None if it can.

    The probe `test_quantized_forward.py` and `test_decode_split.py` use: open a
    path that does not exist and read which refusal comes back, so a build
    without the CUDA backend skips instead of failing inside `Engine.open`.
    """
    if not native.is_available():
        return "libpocketllm.so is not built"
    probe = next(iter(CHECKPOINTS.values())).with_suffix(".does-not-exist.gguf")
    try:
        native.Engine.open(str(probe), "cuda")
    except native.EngineUnavailable as exc:
        message = str(exc)
        return message if "cuda" in message and "not in this build" in message else None
    return "the cuda probe unexpectedly succeeded"


needs_cuda = pytest.mark.skipif(_cuda_reason() is not None, reason=_cuda_reason() or "")

pytestmark = [needs_cuda]


def _route_fast(checkpoint: pathlib.Path) -> list[int]:
    """Prefill and `STEPS` tokens through the device argmax (`next_token`)."""
    tokens = []
    with native.Engine.open(str(checkpoint), "cuda") as engine:
        token = engine.next_token(PROMPT)
        for step in range(STEPS):
            tokens.append(token)
            if step + 1 < STEPS:
                token = engine.next_token([token])
    return tokens


def _route_gate(checkpoint: pathlib.Path) -> list[int]:
    """The same, through `forward` -- the route the token gates decode on."""
    tokens = []
    with native.Engine.open(str(checkpoint), "cuda") as engine:
        token = engine.argmax(engine.forward(PROMPT))
        for step in range(STEPS):
            tokens.append(token)
            if step + 1 < STEPS:
                token = engine.argmax(engine.forward([token]))
    return tokens


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_next_token_is_the_forward_argmax_answer(name: str) -> None:
    """Sixteen greedy tokens, both routes, one checkpoint, identical ids.

    The two routes are two different sets of `ctypes` calls into two different
    ABI symbols (`pocketllm_forward_argmax` versus `pocketllm_forward` +
    `pocketllm_argmax`), so this is a real comparison of two code paths and not
    a tautology -- and it is the comparison a caller relies on when it routes
    greedy decoding to the cheap path and keeps sampling on the expensive one.

    The skip is per-checkpoint, so a host with only one of the two files still
    runs the other's arm.
    """
    checkpoint = CHECKPOINTS[name]
    if not checkpoint.is_file():
        pytest.skip(f"no checkpoint at {checkpoint}")

    fast = _route_fast(checkpoint)
    gate = _route_gate(checkpoint)

    assert fast == gate, (
        f"next_token and forward+argmax disagree at step "
        f"{next(i for i, (a, b) in enumerate(zip(fast, gate)) if a != b)}: "
        f"{fast} != {gate}"
    )
    assert len(fast) == STEPS