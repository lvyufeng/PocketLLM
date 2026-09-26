"""Xing4.0-29B-A4B's rotary embedding: the YaRN re-base, and the layout fork.

Two things are pinned here that a port could get wrong without failing anything.

**YaRN.**  The checkpoint's own code does not compute YaRN; it calls
`ROPE_INIT_FUNCTIONS["yarn"]`, which is `transformers`'
`_compute_yarn_parameters`.  So the first test holds `rope.inv_freq` to that
function directly, at the released config's own numbers -- not to a copy of the
numbers, and not to a tolerance built from the same formula.

**The layout.**  `rope_interleave = True` and the reference reads *adjacent*
pairs, but it writes the rotated halves back *contiguously*.  This repository
already had the other two conventions in it: `glm_dsa` reads adjacent pairs and
interleaves the result back, and the ordinary `rotate_half` formulation reads a
first-half/second-half split.  All three leave a 64-wide vector 64 wide, so the
output shape cannot tell them apart; `test_the_three_layouts_disagree` asserts
they really are three different results, and the tests after it say which one
Xing4.0 is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from src.models.xing4_0 import rope
from src.models.xing4_0.config import Xing4_0Params, yarn_get_mscale

CONFIG = Path("/mnt/data2/Xing4.0-29B-A4B/config.json")


def _params() -> Xing4_0Params:
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    return Xing4_0Params.from_config(json.loads(CONFIG.read_text(encoding="utf-8")))


def _small(**overrides) -> Xing4_0Params:
    """A fixture-scale copy of the released config: same rope, smaller widths."""
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw.update(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=6,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        vocab_size=32,
        max_position_embeddings=128,
        num_hidden_layers=2,
    )
    raw["rope_scaling"] = dict(raw["rope_scaling"], original_max_position_embeddings=32)
    raw.update(overrides)
    return Xing4_0Params.from_config(raw)


# --------------------------------------------------------------------------- #
# The config's own derived numbers
# --------------------------------------------------------------------------- #


def test_head_dim_is_the_rotary_width_and_not_hidden_over_heads() -> None:
    params = _params()
    assert params.qk_head_dim == 192
    assert params.head_dim == 64
    # The trap: 3584 / 32 is 112, and a port that derives head_dim the usual way
    # builds a 112-wide frequency table for a 64-wide rotary slice.
    assert params.hidden_size // params.n_heads == 112


def test_the_attention_scale_is_the_yarn_mscale_squared() -> None:
    params = _params()
    assert params.qk_head_dim == 192
    mscale = yarn_get_mscale(params.yarn.factor, params.yarn.mscale_all_dim)
    assert mscale == pytest.approx(0.1 * math.log(64.0) + 1.0)
    assert mscale == pytest.approx(1.4158883, rel=1e-6)
    # Not `1/sqrt(192)`, and not `1/sqrt(192) * mscale`: the checkpoint squares it,
    # which is a 2.0x change to every logit and is the difference between
    # attending and not.
    assert params.attention_scale == pytest.approx(192 ** -0.5 * mscale * mscale, rel=1e-12)
    assert params.attention_scale == pytest.approx(0.14467963, rel=1e-6)
    assert params.attention_scale / (192 ** -0.5) == pytest.approx(mscale * mscale, rel=1e-12)


def test_cos_and_sin_are_not_scaled_as_well() -> None:
    """The other `mscale` use, which cancels: two calls, as a ratio."""
    params = _params()
    assert rope.yarn_attention_factor(params) == 1.0
    cos, sin = rope.cos_sin(params, torch.arange(4))
    # `head_dim` wide and not `head_dim / 2`, because the reference duplicates the
    # angles before taking the cosine; only the first half is ever read.
    half = params.head_dim // 2
    angles = torch.arange(4, dtype=torch.float32)[:, None] * rope.inv_freq(params)[None, :]
    assert torch.equal(cos[:, :half], torch.cos(angles))
    assert torch.equal(cos[:, half:], torch.cos(angles))
    assert torch.equal(sin[:, :half], torch.sin(angles))
    assert torch.equal(sin[:, half:], torch.sin(angles))


# --------------------------------------------------------------------------- #
# YaRN, against the function the checkpoint delegates to
# --------------------------------------------------------------------------- #


def test_the_frequencies_are_transformers_yarn() -> None:
    """`ROPE_INIT_FUNCTIONS["yarn"]` is what the reference actually calls."""
    transformers = pytest.importorskip("transformers")
    from transformers import PretrainedConfig
    from transformers.modeling_rope_utils import _compute_yarn_parameters

    params = _params()
    config = PretrainedConfig()
    config.rope_theta = params.rope_theta
    config.head_dim = params.head_dim
    config.hidden_size = params.hidden_size
    config.num_attention_heads = params.n_heads
    config.max_position_embeddings = params.context_length
    config.rope_scaling = {
        "factor": params.yarn.factor,
        "original_max_position_embeddings": params.yarn.original_max_position_embeddings,
        "beta_fast": params.yarn.beta_fast,
        "beta_slow": params.yarn.beta_slow,
        "mscale": params.yarn.mscale,
        "mscale_all_dim": params.yarn.mscale_all_dim,
        "type": "yarn",
    }

    expected, attention_factor = _compute_yarn_parameters(config, device=torch.device("cpu"))
    got = rope.inv_freq(params)
    assert got.shape == expected.shape == (params.head_dim // 2,)
    assert torch.allclose(got, expected.to(got.dtype), rtol=0, atol=0)
    # `attention_factor` is the cos/sin post-scale, and it is 1.0 for this config.
    assert attention_factor == pytest.approx(rope.yarn_attention_factor(params))
    assert attention_factor == 1.0
    assert transformers  # imported, and the version is whatever the environment has


def test_the_yarn_ramp_spans_the_frequencies_it_should() -> None:
    """The correction range, as a number rather than as a curve.

    `beta_fast 32` and `beta_slow 1` over `original_max_position_embeddings 4096`
    put the ramp at frequency indices 10 through 23 out of 32, so the six fastest
    frequencies extrapolate, the nine slowest interpolate, and fourteen are
    between.  A port with the range in *channel* indices instead of pair indices
    would place it at 20..47 and ramp over the whole table.
    """
    params = _params()
    low, high = rope._correction_range(
        params.yarn.beta_fast,
        params.yarn.beta_slow,
        params.head_dim,
        params.rope_theta,
        params.yarn.original_max_position_embeddings,
        params.yarn.truncate,
    )
    assert (low, high) == (10, 23)

    frequencies = rope.inv_freq(params)
    plain = 1.0 / (params.rope_theta ** (torch.arange(0, params.head_dim, 2, dtype=torch.float32) / params.head_dim))
    # At and below the ramp's foot the frequency is untouched, i.e. extrapolated.
    assert torch.allclose(frequencies[: low + 1], plain[: low + 1], rtol=0, atol=0)
    # At and above its head it is interpolated by the factor, i.e. 64x smaller.
    assert torch.allclose(frequencies[high:], plain[high:] / params.yarn.factor, rtol=1e-6, atol=0)
    # Strictly between the two across the twelve indices in between, and monotone
    # the whole way -- a ramp read at channel indices instead of pair indices
    # would leave most of the table on one side.
    assert bool((frequencies[low + 1 : high] < plain[low + 1 : high]).all())
    assert bool((frequencies[low + 1 : high] > plain[low + 1 : high] / params.yarn.factor).all())
    assert bool((frequencies[1:] < frequencies[:-1]).all())


def test_the_factor_is_what_makes_the_context_reachable() -> None:
    """262144 positions at a base of 10000 needs the re-base, not just a bigger table."""
    params = _params()
    assert params.context_length == 262144
    assert params.yarn.original_max_position_embeddings == 4096
    assert params.yarn.factor == 64
    assert params.context_length == params.yarn.original_max_position_embeddings * params.yarn.factor
    # The slowest frequency is the one that decides whether the far end of the
    # context still has phases to distinguish: unscaled it completes a whole
    # rotation well inside the context.
    plain_slowest = 1.0 / params.rope_theta
    yarn_slowest = float(rope.inv_freq(params)[-1])
    assert plain_slowest * params.context_length > 2 * math.pi
    assert yarn_slowest * params.context_length < 2 * math.pi


# --------------------------------------------------------------------------- #
# The layout fork
# --------------------------------------------------------------------------- #


def _interleaved_back(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`glm_dsa`'s convention: read adjacent pairs, write them back interleaved."""
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


def _rotate_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The ordinary formulation: a first-half/second-half split, not adjacent pairs."""
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def test_the_three_layouts_disagree() -> None:
    """They all return a 64-wide vector, and they are not the same vector."""
    torch.manual_seed(0)
    x = torch.randn(2, 2, 5, 8)  # (batch, heads, seq, dim)
    angles = torch.randn(5, 4)
    # Full width, with the angle duplicated, exactly as `cos_sin` returns it and
    # as the reference builds it; each convention takes the half it needs.  The
    # angle rows sit on the seq axis, which for this layout is dim 2.
    cos = torch.cos(angles).repeat(1, 2).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).repeat(1, 2).unsqueeze(0).unsqueeze(0)

    xing = rope.rotate_interleaved(x, cos, sin)
    glm = _interleaved_back(x, cos, sin)
    halves = _rotate_half_split(x, cos, sin)
    assert xing.shape == glm.shape == halves.shape == x.shape
    assert not torch.allclose(xing, glm)
    assert not torch.allclose(xing, halves)
    assert not torch.allclose(glm, halves)
    # The Xing4.0 and glm_dsa results are permutations of one another, which is
    # exactly why the shape cannot tell them apart and why this test has to: weave
    # the contiguous halves back into pairs and the two agree.
    woven = xing.reshape(*xing.shape[:-1], 2, -1).transpose(-2, -1).reshape(xing.shape)
    assert torch.equal(woven, glm)


def test_xing4_reads_adjacent_pairs_and_writes_contiguous_halves() -> None:
    """The convention, stated as the reference's `apply_rotary_pos_emb_interleave` states it.

    Restricting to the first frequency makes the assertion readable: `q1` is the
    even elements, `q2` the odd ones, and the result is `[q1 cos - q2 sin,
    q2 cos + q1 sin]` with the two halves laid end to end.
    """
    width = 8
    x = torch.arange(width, dtype=torch.float32).reshape(1, 1, 1, width)
    ones = torch.ones(1, 1, width)
    zeros = torch.zeros(1, 1, width)
    q1, q2 = x[..., 0::2], x[..., 1::2]
    assert torch.equal(rope.rotate_interleaved(x, ones, zeros), torch.cat((q1, q2), dim=-1))
    rotated = rope.rotate_interleaved(x, zeros, ones)
    # A zero cosine and a unit sine is the pair rotation by 90 degrees, written
    # into contiguous halves rather than back into the pairs.
    assert torch.equal(rotated, torch.cat((-q2, q1), dim=-1))


def test_a_zero_angle_is_the_de_interleave() -> None:
    """At angle zero the operation is still not the identity -- it moves channels.

    cos = 1, sin = 0 gives `cat((x1, x2))`, so the even elements come first and the
    odd ones follow.  A 64-wide vector comes out 64 wide and every value is one of
    the inputs, so nothing downstream fails; the q vector simply holds the rotary
    channels in a different order than it did.  This is the property that makes
    the layout worth a test rather than a comment.
    """
    torch.manual_seed(2)
    params = _small()
    x = torch.randn(1, 3, 2, params.head_dim)  # (batch, seq, heads, dim)
    cos, sin = rope.cos_sin(params, torch.zeros(3))
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (seq, 1, dim)
    out = rope.rotate_interleaved(x, cos, sin)
    assert not torch.equal(out, x)
    assert torch.equal(out, torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1))
    assert sorted(out.flatten().tolist()) == sorted(x.flatten().tolist())


def test_a_quarter_turn_is_the_pair_rotation() -> None:
    """`theta = pi/2` on one frequency rotates every pair by 90 degrees."""
    params = _small()
    frequencies = torch.zeros(params.head_dim // 2)
    frequencies[0] = 1.0
    cos, sin = rope.cos_sin(params, torch.tensor([math.pi / 2]), frequencies=frequencies)
    width = params.head_dim
    x = torch.zeros(1, 1, 1, width)
    x[..., 0], x[..., 1] = 3.0, 4.0
    out = rope.rotate_interleaved(x, cos, sin)
    # The pair (3, 4) rotated by 90 degrees is (-4, 3).  The two halves are laid
    # out contiguously, so the second member of the pair lands at `width / 2`.
    assert out[..., 0].item() == pytest.approx(-4.0)
    assert out[..., width // 2].item() == pytest.approx(3.0)
    # Every other pair is untouched, because only the first frequency is non-zero.
    others = [i for i in range(width) if i not in (0, width // 2)]
    assert bool((out[..., others] == 0).all())


def test_an_odd_slice_is_refused() -> None:
    with pytest.raises(ValueError, match="even"):
        rope.rotate_interleaved(torch.zeros(1, 1, 1, 5), torch.zeros(1, 2), torch.zeros(1, 2))
