"""The Qwen3 architecture: that the graph is the model, and not merely runnable.

The toy's test asserts a built graph executes.  That is the wrong bar for a real
architecture, because a graph that runs and a graph that computes Qwen3 are two
different claims and only the second one matters.  So the tests here come in
three layers, each one a stronger statement than the last:

* **the geometry** -- weight names, shapes and the cache plan, pinned against an
  explicit expected set, so a rename or an off-by-one in a projection width is a
  test failure rather than a graph that silently reads the wrong tensor;
* **the structure** -- `verify()` passes, which checks every node against the ABI
  schemas, and the per-head norm really is per head;
* **the numerics** -- the graph's logits against an independent numpy
  transcription of ``Qwen3Model::forward``, written out here rather than shared
  with the builder.  This is the layer that catches a wrong reduction axis or a
  rope applied after the cache write, which is exactly the class of bug that
  produces finite, plausible, wrong output.

The transcription is deliberately a *second* implementation rather than a call
into the builder: comparing a thing with itself proves nothing, and the point of
this file is the comparison.  It is not the C engine -- that comparison needs a
real checkpoint and lives in ``tests/native/`` -- but it is the same forward pass
written from the same source, and it is checkable on a host with nothing
installed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pocketllm.architectures import ARCHITECTURES, build, names
from pocketllm.architectures.qwen3 import Qwen3Config, build as build_qwen3
from pocketllm.backends.reference import BACKEND
from pocketllm.engine import Executor
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType

#: A config small enough to run in a test and shaped so every geometry question
#: is non-trivial: heads != kv_heads (so the grouping is exercised), head_dim
#: larger than 2 (so the rotary split has interior elements), and two rows (so a
#: reshape to heads cannot pass by accident on a singleton).
TINY = Qwen3Config(hidden=32, layers=3, heads=4, kv_heads=2, head_dim=8, ff=48, vocab=64, context=16, rows=2)


def _expected_weight_names(cfg: Qwen3Config) -> set[str]:
    """The GGUF names ``src/model/qwen3.cpp`` binds, spelled the same way."""
    out = {"token_embd.weight", "output_norm.weight"}
    if not cfg.tie_embeddings:
        out.add("output.weight")
    for il in range(cfg.layers):
        base = f"blk.{il}."
        out.update(
            base + name
            for name in (
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        )
    return out


# -- geometry -----------------------------------------------------------------


def test_the_graph_verifies() -> None:
    """Every node checked against the ABI schemas, before any device is opened."""
    spec = build_qwen3(TINY)
    assert spec.verified
    spec.graph.verify()


def test_the_weights_are_exactly_the_gguf_names() -> None:
    """A rename here is a binding failure, not a rebuild -- so pin the names."""
    spec = build_qwen3(TINY)
    assert set(spec.weights.names()) == _expected_weight_names(TINY)


def test_the_projection_widths_are_the_head_geometry() -> None:
    spec = build_qwen3(TINY)
    weights = spec.weights
    hidden, q_width, kv_width, ff = TINY.hidden, TINY.q_width, TINY.kv_width, TINY.ff
    assert weights.get("token_embd.weight").desc.shape == (TINY.vocab, hidden)
    assert weights.get("blk.0.attn_q.weight").desc.shape == (q_width, hidden)
    assert weights.get("blk.0.attn_k.weight").desc.shape == (kv_width, hidden)
    assert weights.get("blk.0.attn_v.weight").desc.shape == (kv_width, hidden)
    assert weights.get("blk.0.attn_output.weight").desc.shape == (hidden, q_width)
    assert weights.get("blk.0.ffn_gate.weight").desc.shape == (ff, hidden)
    assert weights.get("blk.0.ffn_down.weight").desc.shape == (hidden, ff)
    # The QK norms are per head, so they are `head_dim` long and not `q_width`.
    assert weights.get("blk.0.attn_q_norm.weight").desc.shape == (TINY.head_dim,)
    assert weights.get("blk.0.attn_k_norm.weight").desc.shape == (TINY.head_dim,)


def test_the_norms_are_not_quantizable_and_the_projections_are() -> None:
    spec = build_qwen3(TINY)
    assert not spec.weights.get("blk.0.attn_norm.weight").quantizable
    assert not spec.weights.get("output_norm.weight").quantizable
    assert spec.weights.get("blk.0.attn_q.weight").quantizable
    assert spec.weights.get("token_embd.weight").quantizable


def test_tied_embeddings_drop_the_separate_head() -> None:
    """A checkpoint with no ``output.weight`` shares the table, as the C engine does."""
    spec = build_qwen3(replace(TINY, tie_embeddings=True))
    assert "output.weight" not in spec.weights.names()
    assert "output.weight" not in spec.graph.input_names


def test_the_cache_is_one_pair_per_layer() -> None:
    """Unrolled over the layer axis: the shape `attention` can actually read."""
    spec = build_qwen3(TINY)
    expected = tuple(
        name for il in range(TINY.layers) for name in (f"blk.{il}.k_cache", f"blk.{il}.v_cache")
    )
    assert spec.cache.names() == expected
    assert spec.cache_values == expected
    for layout in spec.cache.layouts:
        assert layout.shape(TINY.context) == (1, TINY.context, TINY.kv_heads, TINY.head_dim)
    assert spec.cache.default_capacity == TINY.context


def test_the_config_refuses_geometry_that_cannot_run() -> None:
    with pytest.raises(ValueError, match="multiple of kv_heads"):
        Qwen3Config(heads=8, kv_heads=3)
    with pytest.raises(ValueError, match="head_dim must be even"):
        Qwen3Config(head_dim=7)
    with pytest.raises(ValueError, match="must be positive"):
        Qwen3Config(layers=0)
    with pytest.raises(ValueError, match="cannot exceed the context"):
        Qwen3Config(context=8, rows=16)


def test_the_registry_reaches_qwen3_by_name() -> None:
    spec = build("qwen3", TINY)
    assert spec.name == "qwen3"
    assert "qwen3" in names()
    assert ARCHITECTURES["qwen3"].summary


# -- the numerics -------------------------------------------------------------


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _rope_neox(x: np.ndarray, positions: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Split-half rotation: pair ``i`` with ``i + d/2``.  ``x`` is ``(rows, heads, d)``."""
    half = x.shape[-1] // 2
    out = np.empty_like(x)
    for t, p in enumerate(positions):
        a, b = x[t, :, :half], x[t, :, half:]
        out[t, :, :half] = a * cos[p] - b * sin[p]
        out[t, :, half:] = a * sin[p] + b * cos[p]
    return out


def _config_weights(spec, env, rng) -> dict[str, np.ndarray]:
    """Deterministic weights for every graph weight, at the shape it declares."""
    out: dict[str, np.ndarray] = {}
    for name in spec.weight_values:
        shape = env[name].shape
        if name.endswith("_norm.weight"):
            out[name] = (1.0 + rng.standard_normal(shape) * 0.1).astype(np.float32)
        else:
            out[name] = (rng.standard_normal(shape) * 0.08).astype(np.float32)
    # Embeddings and the head are shared-scale in a real checkpoint; give them
    # their own draw anyway, so a graph that accidentally reads one for the other
    # disagrees rather than coincidentally matching.
    out["token_embd.weight"] = (rng.standard_normal(env["token_embd.weight"].shape) * 0.3).astype(np.float32)
    if "output.weight" in out:
        out["output.weight"] = (rng.standard_normal(env["output.weight"].shape) * 0.08).astype(np.float32)
    return out


def _transcribe(cfg: Qwen3Config, weights, tokens, positions, cos, sin) -> tuple:
    """``Qwen3Model::forward``, written out in numpy from ``src/model/qwen3.cpp``.

    A second implementation on purpose: the whole value of the comparison is that
    the two were written separately from the same source, so a shared helper
    between them would destroy what it measures.  The order is the C engine's --
    norm, project, per-head norm, rope, *then* the cache write, then attention,
    the output projection, and the two residual adds around a SwiGLU MLP.

    Returns ``(logits, caches)`` where ``caches`` maps the same ``blk.<i>.{k,v}_cache``
    names the graph uses to the state the forward should have left behind.  That
    second value is what makes "the cache holds the rotated key" checkable rather
    than merely "the cache is not zero".
    """
    rows, layers, heads, kv_heads = cfg.rows, cfg.layers, cfg.heads, cfg.kv_heads
    q_width = cfg.q_width
    ck = np.zeros((layers, cfg.context, kv_heads, cfg.head_dim), np.float32)
    cv = np.zeros_like(ck)
    x = weights["token_embd.weight"][tokens]
    group = heads // kv_heads
    scale = 1.0 / np.sqrt(cfg.head_dim)
    for il in range(layers):
        base = f"blk.{il}."
        xn = _rms(x, weights[base + "attn_norm.weight"], cfg.rms_eps)
        q = xn @ weights[base + "attn_q.weight"].T
        k = xn @ weights[base + "attn_k.weight"].T
        v = xn @ weights[base + "attn_v.weight"].T
        # One row per head -- the reduction the C engine's `n * n_head_` argument
        # produces, and the reason the norm weight is `head_dim` long.
        q = _rms(q.reshape(rows * heads, cfg.head_dim), weights[base + "attn_q_norm.weight"], cfg.rms_eps).reshape(rows, heads, cfg.head_dim)
        k = _rms(k.reshape(rows * kv_heads, cfg.head_dim), weights[base + "attn_k_norm.weight"], cfg.rms_eps).reshape(rows, kv_heads, cfg.head_dim)
        v = v.reshape(rows, kv_heads, cfg.head_dim)
        q = _rope_neox(q, positions, cos, sin)
        k = _rope_neox(k, positions, cos, sin)
        ck[il][positions] = k  # the cache holds the *rotated* key
        cv[il][positions] = v
        attn = np.zeros((rows, heads, cfg.head_dim), np.float32)
        for t, p in enumerate(positions):
            for h in range(heads):
                kv = h // group
                scores = (ck[il][: p + 1, kv, :] @ q[t, h, :]) * scale
                scores = scores - scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                attn[t, h, :] = probs @ cv[il][: p + 1, kv, :]
        x = x + attn.reshape(rows, q_width) @ weights[base + "attn_output.weight"].T
        xn = _rms(x, weights[base + "ffn_norm.weight"], cfg.rms_eps)
        gate = xn @ weights[base + "ffn_gate.weight"].T
        up = xn @ weights[base + "ffn_up.weight"].T
        x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ weights[base + "ffn_down.weight"].T
    logits = _rms(x, weights["output_norm.weight"], cfg.rms_eps) @ weights["output.weight"].T
    caches = {
        f"blk.{il}.k_cache": ck[il] for il in range(layers)
    } | {
        f"blk.{il}.v_cache": cv[il] for il in range(layers)
    }
    return logits, caches


def _run_graph(spec, binding):
    """Run the graph on the reference backend and read its logits back.

    A session is opened per call and always closed -- the reference session owns
    a host arena, and leaking one per test is how a suite that used to finish
    starts swapping.  Reading a value *copies* it into numpy, so the returned
    arrays stay valid after the session goes away.
    """
    session = BACKEND.open(Device("cpu"))
    try:
        executor = Executor(session)
        inputs = {
            name: session.tensor(np.ascontiguousarray(binding[name])) for name in spec.graph.input_names
        }
        out = executor.run(spec.graph, inputs)
        return np.asarray(session.array(out["logits"])).copy()
    finally:
        session.close()


def _rope_tables(cfg: Qwen3Config) -> tuple[np.ndarray, np.ndarray]:
    half = cfg.head_dim // 2
    angle = np.arange(cfg.context)[:, None] * cfg.rope_theta ** (-2.0 * np.arange(half) / cfg.head_dim)
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _zero_caches(cfg: Qwen3Config) -> dict[str, np.ndarray]:
    return {
        f"blk.{il}.{kind}_cache": np.zeros((cfg.context, cfg.kv_heads, cfg.head_dim), np.float32)
        for il in range(cfg.layers)
        for kind in ("k", "v")
    }


def _bind(cfg: Qwen3Config, weights, tokens, positions) -> dict[str, np.ndarray]:
    """Everything a graph input needs: the weights, this call's tokens, and an empty cache."""
    cos, sin = _rope_tables(cfg)
    binding = dict(weights)
    binding.update(tokens=tokens, positions=positions, rope_cos=cos, rope_sin=sin, **_zero_caches(cfg))
    return binding


@pytest.fixture(scope="module")
def case():
    """One seeded tiny model, its weights, and its graph's logits.

    The caches are read back *before* the session closes, and the logits are
    copied: a numpy array read out of a host buffer is a view, and the session
    owns that buffer's lifetime.
    """
    rng = np.random.default_rng(11)
    spec = build_qwen3(TINY)
    env = spec.graph.verify()
    weights = _config_weights(spec, env, rng)
    tokens = np.array([5, 17], np.int32)
    positions = np.array([0, 1], np.int32)
    binding = _bind(TINY, weights, tokens, positions)

    session = BACKEND.open(Device("cpu"))
    executor = Executor(session)
    inputs = {name: session.tensor(np.ascontiguousarray(binding[name])) for name in spec.graph.input_names}
    try:
        graph_logits = np.asarray(session.array(executor.run(spec.graph, inputs)["logits"])).copy()
        caches = {
            name: np.asarray(session.array(inputs[name])).copy() for name in spec.cache_values
        }
    finally:
        session.close()

    expected, expected_caches = _transcribe(TINY, weights, tokens, positions, *_rope_tables(TINY))
    return {
        "spec": spec,
        "weights": weights,
        "logits": graph_logits,
        "expected": expected,
        "caches": caches,
        "expected_caches": expected_caches,
        "tokens": tokens,
        "positions": positions,
    }


def test_the_graph_computes_what_the_c_engine_computes(case) -> None:
    """The acceptance claim: the graph's logits equal the transcribed forward.

    A wrong QK-norm axis, a rope before the norm, a cache written before the
    rotation -- each produces finite logits of roughly the right scale, and the
    only thing that tells them apart from the right answer is this comparison.
    """
    got, want = case["logits"], case["expected"]
    assert got.shape == want.shape
    assert np.all(np.isfinite(got))
    assert not np.allclose(want, 0.0), "an all-zero reference means a weight was not read"
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_the_rotation_does_not_change_the_row_norm(case) -> None:
    """RoPE preserves each ``(head, d)`` row's norm; the stored key must match.

    The structural fact to test is not "the cache is non-zero" -- a graph that
    wrote the *unrotated* key would pass that -- but that the write happened
    *after* the rotation.  Rather than reconstruct the orthonormality argument
    here, this pins the cache against the transcribed forward's own cache: two
    independent readings of ``src/model/qwen3.cpp`` agreeing on the stored
    bytes is what makes "rotated, at the token's position" a checked claim.
    """
    positions = case["positions"]
    for il in range(TINY.layers):
        for kind in ("k", "v"):
            got = case["caches"][f"blk.{il}.{kind}_cache"]
            want = case["expected_caches"][f"blk.{il}.{kind}_cache"]
            np.testing.assert_allclose(
                got[positions], want[positions], rtol=1e-5, atol=1e-6,
                err_msg=f"blk.{il}.{kind}_cache: the graph's write differs from the transcribed forward",
            )
            # Every other slot is untouched: a per-layer slab, not a shared one.
            assert np.allclose(got[TINY.rows :], 0.0), f"blk.{il}.{kind}_cache wrote outside its positions"


def test_the_rotation_is_what_makes_the_cache_non_trivial(case) -> None:
    """The stronger half: the stored key is *not* the pre-rotation normalized key.

    Computed here rather than transcribed, so this fails if the rope is applied
    to nothing or applied in the wrong order.  ``_rope_neox`` at position 0 is
    the identity (angle zero), so a bug that skipped the rope entirely would
    still match at position 0 -- the check uses position 1, where the angle is
    non-zero.
    """
    weights, positions = case["weights"], case["positions"]
    embed = weights["token_embd.weight"][case["tokens"]]
    xn = _rms(embed, weights["blk.0.attn_norm.weight"], TINY.rms_eps)
    k_pre = _rms(
        (xn @ weights["blk.0.attn_k.weight"].T).reshape(TINY.rows * TINY.kv_heads, TINY.head_dim),
        weights["blk.0.attn_k_norm.weight"],
        TINY.rms_eps,
    ).reshape(TINY.rows, TINY.kv_heads, TINY.head_dim)
    stored = case["caches"]["blk.0.k_cache"][positions]
    assert not np.allclose(stored[1], k_pre[1], atol=1e-6), (
        "the cache holds the pre-rotation key: rope ran after the write, or not at all"
    )
    # ...and the rotation is orthonormal, so the row norms are unchanged.
    np.testing.assert_allclose(
        np.linalg.norm(stored[1], axis=-1), np.linalg.norm(k_pre[1], axis=-1), rtol=1e-5, atol=1e-6
    )


def test_the_graph_is_deterministic() -> None:
    first, second = build_qwen3(TINY), build_qwen3(TINY)
    assert [n.label() for n in first.graph.nodes] == [n.label() for n in second.graph.nodes]
    assert first.graph.nodes == second.graph.nodes


def test_a_decode_width_of_one_also_runs() -> None:
    """The width a decode loop uses, exercised separately: `rows` is a build-time constant."""
    cfg = replace(TINY, rows=1)
    spec = build_qwen3(cfg)
    env = spec.graph.verify()
    rng = np.random.default_rng(3)
    weights = _config_weights(spec, env, rng)
    tokens, positions = np.array([9], np.int32), np.array([0], np.int32)
    got = _run_graph(spec, _bind(cfg, weights, tokens, positions))
    expected, _caches = _transcribe(cfg, weights, tokens, positions, *_rope_tables(cfg))
    assert got.shape == (1, cfg.vocab)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)