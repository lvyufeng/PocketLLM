"""Xing4.0-29B-A4B's rotary embedding: YaRN over a 64-wide interleaved slice.

Two things here are new to this repository, and both are places a port from
`deepseek_v4_1` or `glm_dsa` diverges without failing.

**YaRN.**  Every other checkpoint here uses plain RoPE or a partial one.  This
one re-bases the frequencies: `factor 64`, an `original_max_position_embeddings`
of 4096 against a 262144 context, and a linear ramp over frequency indices
10..23 that extrapolates the fast end and interpolates the slow end.  The
reference does not implement YaRN itself -- `Xing4_0RotaryEmbedding` calls
`ROPE_INIT_FUNCTIONS["yarn"]`, which is `transformers`'
`_compute_yarn_parameters`.  So the authority for the arithmetic is that
function, and `tests/test_xing4_0_rope.py` holds this port to it directly rather
than to a transcribed copy of the numbers.

**The interleaved layout, with a half-split output.**  `rope_interleave = True`,
and the reference's `apply_rotary_pos_emb_interleave` reads the 64-wide slice as
adjacent pairs -- `(x0, x1), (x2, x3), ...`, one frequency each -- but writes the
two rotated halves *contiguously*::

    q1, q2 = q[..., 0::2], q[..., 1::2]
    out = cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)

That is neither of the two conventions this repository already had.  `glm_dsa`
also reads interleaved pairs, but it interleaves the result back
(``stack((y1, y2), -1).flatten(-2)``); the ordinary `rotate_half` formulation
applies the same formula to a *first-half/second-half* split instead of an
even/odd one.  Xing4.0 reads even/odd and writes first-half/second-half, so a
decode kernel has to reproduce both halves of that choice.  It is invisible in
the output *shape* either way -- the q vector is 192 wide whichever layout the
64 rotary slots hold -- which is what makes it worth pinning in a test.

`attention_scaling` is 1.0 for this config and is still carried through, because
it is not always: it is `get_mscale(factor, mscale) / get_mscale(factor,
mscale_all_dim)`, which is 1 for this checkpoint and would not be for a
`mscale_all_dim` of 0.
"""

from __future__ import annotations

import math

import torch

from src.models.xing4_0.config import Xing4_0Params

__all__ = [
    "cos_sin",
    "inv_freq",
    "rotate_interleaved",
    "yarn_attention_factor",
]


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """The frequency index that rotates `num_rotations` times over `max_position_embeddings`."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool
) -> tuple[float, float]:
    low = _find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(low: float, high: float, dim: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity, as the reference does.
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return ramp.clamp(0, 1)


def inv_freq(params: Xing4_0Params, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 32 inverse frequencies, YaRN-corrected.

    A port of `transformers.modeling_rope_utils._compute_yarn_parameters`, which
    is what the checkpoint's own `Xing4_0RotaryEmbedding` calls.  `dim` is
    `head_dim` -- 64, the rotary width -- and the ramp runs over `dim // 2 = 32`
    frequencies, so the correction range is in *pair* indices and not in channel
    indices.
    """
    dim = params.head_dim
    base = float(params.rope_theta)
    yarn = params.yarn
    pos_freqs = base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (yarn.factor * pos_freqs)
    low, high = _correction_range(
        yarn.beta_fast,
        yarn.beta_slow,
        dim,
        base,
        yarn.original_max_position_embeddings,
        yarn.truncate,
    )
    # ramp is 0 at the fast end (extrapolate) and 1 at the slow end (interpolate),
    # so `1 - ramp` weights the extrapolated frequencies.
    extrapolation_factor = 1 - _linear_ramp_factor(low, high, dim // 2, device=device)
    out = interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor
    return out.to(dtype)


def yarn_attention_factor(params: Xing4_0Params) -> float:
    """The cos/sin post-scale, which is 1.0 for this checkpoint.

    The same `mscale` machinery as the attention scale, but as a ratio of two
    calls rather than a square -- see `Xing4_0Params.attention_scale` for the
    other use, which does not cancel.
    """
    yarn = params.yarn

    def mscale(scale: float, m: float) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * m * math.log(scale) + 1.0

    if yarn.mscale and yarn.mscale_all_dim:
        return float(mscale(yarn.factor, yarn.mscale) / mscale(yarn.factor, yarn.mscale_all_dim))
    return float(mscale(yarn.factor, 1.0))


def cos_sin(
    params: Xing4_0Params,
    positions: torch.Tensor,
    *,
    frequencies: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`cos`/`sin` of shape `(positions.numel(), head_dim)` for a flat position list.

    `head_dim` wide and not `head_dim / 2`, because the reference concatenates
    the angles with themselves (`emb = cat((freqs, freqs), -1)`) before taking
    the cosine.  The interleaved rotation then reads only the first half, so the
    second half is a duplicate the caller may index into either way; keeping the
    duplication is what makes this comparable with the reference line for line.
    """
    if frequencies is None:
        frequencies = inv_freq(params, device=device)
    positions = positions.to(device=device, dtype=torch.float32).reshape(-1)
    angles = positions[:, None] * frequencies[None, :].to(torch.float32)
    emb = torch.cat((angles, angles), dim=-1)
    scaling = yarn_attention_factor(params)
    return (emb.cos() * scaling).to(dtype), (emb.sin() * scaling).to(dtype)


def rotate_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The reference's `apply_rotary_pos_emb_interleave`, in fp32.

    `x` is `(..., 2k)` -- the rotary slice of a query or key -- and `cos`/`sin`
    are `(..., 2k)` with the angle duplicated, so the first half carries it.
    They must already be broadcastable to `x`: the reference inserts the head
    axis itself (`cos.unsqueeze(1)`) and so does the caller here, because the
    axis it belongs on depends on whether `x` is laid out head-major or
    row-major.

    The pairs are adjacent (`x[..., 0::2]`, `x[..., 1::2]`) and the two rotated
    halves are written back contiguously, not interleaved.
    """
    if x.shape[-1] % 2:
        raise ValueError(f"the rotary slice must be even, got {x.shape[-1]}")
    half = cos.shape[-1] // 2
    cos = cos[..., :half].float()
    sin = sin[..., :half].float()
    x = x.float()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
