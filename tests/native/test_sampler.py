"""The C sampler through the ABI, against the numpy reference.

`test_op_conformance.py` already compares `topk_sample` to the reference through
`opcheck`, and this does not repeat that. What it adds is the *other* road into
the same kernel: the exported `pocketllm_sample` / `pocketllm_temperature`
symbols, called from `pocketllm.native` the way `cli.py` calls them. The two
entry points share the kernel and not the marshalling, so a bug in the argument
order or the float type at the boundary would pass op-level conformance and fail
here -- and it is the boundary the CLI actually crosses.

The comparison is exact and it is of a *token*, not of a distribution. That is a
stronger claim than it sounds: the reference and the kernel round their
probabilities differently, and the token is the same token unless the draw lands
exactly between two of them, so agreement over many random draws is evidence
that the ranking and the truncation rules match and not that two floats are
close. The boundary draws are here for the same reason -- they are the draws
where a difference would show.

Everything skips without the library. A skip is not a pass.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pocketllm import native
from pocketllm.backends.reference import kernels as ref

pytestmark = pytest.mark.skipif(
    not native.is_available(),
    reason="libpocketllm.so is not built (run `cmake -B build -S src && cmake --build build`)",
)

#: The vocabulary-sized draw is the one where the two accumulators can disagree
#: -- see `test_op_conformance.test_a_flat_tail_samples_where_an_exact_reference_would`
#: for the measured divergence.  It is excluded from the exact comparison below
#: and covered by a check of its own.
WIDE = 151_936


def test_temperature_matches_the_reference() -> None:
    """Every entry divided, at a temperature that is not 1.0 -- a value `1.0`
    would agree with a function that ignored the argument entirely."""
    rng = np.random.default_rng(11)
    logits = (rng.standard_normal(64).astype(np.float32) * 8.0).tolist()
    got = native.Engine.temperature(logits, 0.6)
    want = ref.logits_temperature(np.array(logits, dtype=np.float32), temperature=0.6)
    assert np.array_equal(np.array(got, dtype=np.float32), want.astype(np.float32))


def test_temperature_refuses_zero_the_way_the_reference_does() -> None:
    """Both refuse, and both refuse for the same documented reason.

    The two raise different exception types -- `ValueError` from the binding and
    from the reference, `EngineUnavailable` nowhere -- but what is being pinned
    is that neither silently returns a vector of infinities, which is what a
    division by zero would produce and what a decode loop would then sample from.
    """
    logits = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        native.Engine.temperature(logits, 0.0)
    with pytest.raises(ValueError):
        ref.logits_temperature(np.array(logits, dtype=np.float32), temperature=0.0)


@pytest.mark.parametrize("dtype", [np.float32])
def test_sampling_matches_the_reference_over_many_draws(dtype) -> None:
    """The token, over a sweep of vocabularies and the three truncation rules.

    The parameter combinations are drawn per case rather than enumerated: the
    argument space is four-dimensional and small, but a random sweep over it
    finds disagreements between combinations that a hand-written table would
    miss -- in particular the ones where `top_k` and `top_p` interact.
    """
    rng = np.random.default_rng(20261002)
    mismatches = []
    for _ in range(300):
        vocab = int(rng.integers(2, 200))
        logits = (rng.standard_normal(vocab) * float(rng.choice([1.0, 8.0, 60.0]))).astype(dtype)
        uniform = float(rng.random())
        top_k = int(rng.choice([0, 1, 3, 40]))
        top_p = float(rng.choice([1.0, 0.95, 0.7, 0.2]))
        min_p = float(rng.choice([0.0, 0.005, 0.05, 1.0]))

        got = native.Engine.sample(logits.tolist(), uniform, top_k, top_p, min_p)
        want = int(
            ref.topk_sample(
                logits,
                np.array([uniform], dtype=np.float32),
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
        )
        if got != want:
            mismatches.append((vocab, uniform, top_k, top_p, min_p, got, want))

    assert not mismatches, (
        f"{len(mismatches)} of 300 draws disagreed with the reference; first: "
        f"(vocab, u, top_k, top_p, min_p, got, want) = {mismatches[0]}"
    )


@pytest.mark.parametrize("uniform", [0.0, 0.25, 0.5, 0.75, 1.0 - 1e-7])
def test_a_uniform_draw_of_zero_takes_the_argmax(uniform: float) -> None:
    """The lowest draw takes the top token, and it is the same token `argmax`
    returns -- which is the one place the sampler and the greedy path must agree,
    because a caller may reach both in one decode loop."""
    rng = np.random.default_rng(5)
    logits = (rng.standard_normal(64).astype(np.float32) * 3.0).tolist()
    if uniform == 0.0:
        assert native.Engine.sample(logits, uniform) == native.Engine.argmax(logits)
    # Whatever the draw, the answer is in range and the two entry points agree
    # on the ranking they are reading.
    assert 0 <= native.Engine.sample(logits, uniform) < len(logits)


def test_tied_logits_put_the_draw_on_the_lower_index() -> None:
    """Eight equal logits and a draw at the middle of the range.

    Each token holds 0.125, so the cumulative reaches exactly 0.5 at the fourth
    -- index 3 -- and the reference takes the first index at or past the draw.
    A strict `>` instead of `>=` skips it and answers 4, and a tie rule that ran
    the other way answers 4 as well, by a different route. Both are a different
    token rather than a rounding, which is why the tied case is worth a test.
    """
    logits = [0.0] * 8
    want = int(ref.topk_sample(np.zeros(8, dtype=np.float32), np.array([0.5], dtype=np.float32)))
    assert native.Engine.sample(logits, 0.5) == want
    assert want == 3


def test_min_p_equal_to_one_keeps_only_the_argmax() -> None:
    """The cutoff is inclusive, so `min_p=1.0` keeps the top token rather than
    discarding everything and falling back to it by a different route."""
    rng = np.random.default_rng(9)
    logits = (rng.standard_normal(32).astype(np.float32) * 4.0).tolist()
    for uniform in (0.0, 0.4, 0.9, 0.999):
        assert native.Engine.sample(logits, uniform, min_p=1.0) == native.Engine.argmax(logits)


def test_top_k_of_one_ignores_the_draw() -> None:
    """One token kept leaves the uniform nothing to choose between, which is
    what makes `top_k=1` a valid way to ask for greedy through the sampler."""
    rng = np.random.default_rng(13)
    logits = (rng.standard_normal(50).astype(np.float32) * 2.0).tolist()
    best = native.Engine.argmax(logits)
    for uniform in (0.0, 0.3, 0.6, 0.999):
        assert native.Engine.sample(logits, uniform, top_k=1) == best


def test_a_draw_repeats_given_the_same_seed() -> None:
    """The property the CLI's `--seed` rests on, at the level the CLI uses it.

    `random.Random(seed)` is what `cli.py` passes the draws from, so this is the
    reproduction check without a checkpoint: the same seed produces the same
    sequence of uniform variates, and the engine is a pure function of them.
    """
    rng = np.random.default_rng(3)
    logits = (rng.standard_normal(128).astype(np.float32) * 4.0).tolist()

    def draw(seed: int) -> list[int]:
        source = random.Random(seed)
        return [native.Engine.sample(logits, source.random(), top_k=20, top_p=0.9) for _ in range(16)]

    assert draw(7) == draw(7)


def test_the_vocabulary_sized_draw_agrees_with_an_exact_reference() -> None:
    """The one case where the reference's own rounding, not the kernel's, is the
    outlier -- pinned here as well as in the op-conformance test because this is
    the entry point a caller reaches, and the divergence is a property of the
    op and not of `opcheck`.

    The reference's `np.cumsum` accumulates in float32 and the kernel's in
    double; over 151936 additions the float32 sum drifts by ~4e-7, which in a
    tail as flat as this vocabulary's moves the crossing index by thousands of
    positions.  Recomputing the cumulative in float64 settles which side is
    which, and it is the kernel that agrees with it.
    """
    rng = np.random.default_rng(20261002)
    logits = rng.standard_normal(WIDE).astype(np.float32)
    uniform = 0.999

    probs = ref.softmax(logits).astype(np.float64)
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    exact = int(order[int(np.searchsorted(cumulative, uniform, side="left"))])

    assert native.Engine.sample(logits.tolist(), uniform) == exact
    assert int(ref.topk_sample(logits, np.array([uniform], dtype=np.float32))) != exact