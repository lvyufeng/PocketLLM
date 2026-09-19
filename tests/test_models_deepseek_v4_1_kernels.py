"""Semantic-contract tests for the ops `src/models/deepseek_v4_1/kernels.py` exposes.

These ops have no oracle on this host. The released V4.1 runtime implements them in TileLang, which
is not installed and cannot be (it needs `torch>=2.10.0`; this environment is on 2.9.1), and the
sm_75 cards here have no FP4 tensor core. So nothing below compares against the reference's own
output, and no test here claims to.

What is checked is that each op behaves the way the V4.1 reference says it must:

* `sparse_attn` against a brute-force softmax written out longhand in the test, plus the empty-index
  case the reference's finite-floor comment exists for. That comment is the reason this file is not
  merely re-testing `src/kernels/ops.py`: the reference changed its score fill from `-inf` to `-1e30`
  for V4.1, explicitly so that a row whose every index is `-1` yields zeros instead of NaN. The
  implementation here satisfies that, but nothing in `src/kernels/ops.py` says it has to, and a
  rewrite to a plain `-inf` fill would still pass every V4-Flash test.
* `hc_split_sinkhorn` against the reference's *stated* sequence of normalizations, traced step by
  step rather than read off the implementation, so a reordering of the row and column steps fails.
  It also pins the two asymmetric gates (`pre` has an epsilon floor, `post` is doubled), which the
  identical output shapes hide.
* `fp4_act_quant_e4m3`'s scale, which is the one op here that `src/kernels/ops.py` does not have.
  The scale is `e4m3(amax / 6)` and not `e4m3(amax)`, and the amax floor is high enough that an
  all-zero block still gets a nonzero scale.

Every assertion is on values computed here, so a pass means these ops agree with the reference's
documented arithmetic and nothing more. Whether a V4.1 forward pass built on them produces the
right tokens is not measurable on this host; see the V4.1 model page for that limitation.
"""

from __future__ import annotations

import pytest
import torch

from src.kernels import ops as shared_kernels
from src.models.deepseek_v4_1 import kernels as v41_kernels
from src.models.deepseek_v4_1.kernels import (
    _fp4_codes,
    _fp4_values,
    fp4_act_quant_e4m3,
    hc_split_sinkhorn,
    sparse_attn,
)

HC_MULT = 4
SINKHORN_ITERS = 20
EPS = 1e-6
FP4_MAX = 6.0


def _brute_force_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """The same quantity, written out per (batch, position, head) with no tensor tricks."""
    b, m, h, d = q.shape
    out = torch.empty(b, m, h, d)
    for bi in range(b):
        for mi in range(m):
            keys = kv[bi][topk_idxs[bi, mi].clamp_min(0)].float()
            scores = (q[bi, mi].float() @ keys.T) * scale
            scores = scores.masked_fill(~(topk_idxs[bi, mi] >= 0).unsqueeze(0), float("-inf"))
            # Any constant works as the softmax's stabilizer, so one max over the whole row is fine
            # and a row with every score at -inf still stabilizes against the sink rather than
            # against -inf. That is the V4.1 contract; the -inf fill above is what would break it.
            row_max = max(scores.max().item(), attn_sink.max().item())
            weights = torch.exp(scores - row_max)
            denom = weights.sum(-1) + torch.exp(attn_sink.float() - row_max)
            out[bi, mi] = (weights @ keys) / denom.unsqueeze(-1)
    return out


def _reference_sinkhorn(comb, iters, eps):
    """The reference's normalization sequence, transcribed from its kernel comment order.

    `comb` arrives already scaled and biased. The kernel then: softmax over the last axis,
    add eps, one *column* normalization, and only then `iters - 1` rounds of row-then-column.
    """
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def test_shared_ops_are_reexported_not_reimplemented() -> None:
    """Six names are `src/kernels/ops.py`'s, four are additions that module does not have."""
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
        assert getattr(v41_kernels, name) is getattr(shared_kernels, name), name
    assert not hasattr(shared_kernels, "fp4_act_quant_e4m3")
    # The two weight expanders are this module's own; the packed layout and the two block-scale
    # dequantizers they stand on are `src/kernels/ops.py`'s and are re-exported with them.
    assert v41_kernels.Packed4BitWeightAlongK is shared_kernels.Packed4BitWeightAlongK
    assert v41_kernels.dequant_fp8_weight is not getattr(shared_kernels, "dequant_fp8_weight", None)
    assert v41_kernels.dequant_fp4_weight is not getattr(shared_kernels, "dequant_fp4_weight", None)
    assert set(v41_kernels.__all__) == {
        "Packed4BitWeightAlongK",
        "act_quant",
        "dequant_fp4_weight",
        "dequant_fp8_weight",
        "fp4_act_quant",
        "fp4_act_quant_e4m3",
        "fp4_gemm",
        "fp8_gemm",
        "hc_split_sinkhorn",
        "sparse_attn",
    }


def test_sparse_attn_matches_a_longhand_softmax() -> None:
    torch.manual_seed(0)
    b, m, h, d, n, topk = 2, 3, 4, 8, 16, 5
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.randint(0, n, (b, m, topk))
    scale = (1.0 / d) ** 0.5

    got = sparse_attn(q, kv, sink, idx, scale)
    want = _brute_force_sparse_attn(q, kv, sink, idx, scale)
    # bf16 storage, so the agreement is at bf16 resolution, not fp32.
    assert torch.allclose(got.float(), want, atol=2e-2, rtol=2e-2)


def test_sparse_attn_empty_row_is_zero_not_nan() -> None:
    """The case the reference's finite floor exists for: every index in the row is -1.

    The reference's kernel fills `scores_max` with -1e30 rather than -inf precisely so this row comes
    out as zeros. Written from the formula with a -inf fill it produces exp(-inf - (-inf)) = NaN
    instead, which is a plausible way to write it and the reason this is asserted.
    """
    torch.manual_seed(2)
    b, m, h, d, n, topk = 2, 3, 4, 8, 16, 5
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.randint(0, n, (b, m, topk))
    idx[0, 0] = -1
    idx[1, 2] = -1

    out = sparse_attn(q, kv, sink, idx, (1.0 / d) ** 0.5)
    assert torch.isfinite(out).all()
    assert (out[0, 0] == 0).all()
    assert (out[1, 2] == 0).all()
    # A row that did have positions is unaffected by an empty row beside it: recompute the fixture
    # with those two rows restored and the surviving rows must be bit-identical.
    restored = idx.clone()
    restored[0, 0] = torch.randint(0, n, (topk,))
    restored[1, 2] = torch.randint(0, n, (topk,))
    filled = sparse_attn(q, kv, sink, restored, (1.0 / d) ** 0.5)
    keep = torch.ones(b, m, dtype=torch.bool)
    keep[0, 0] = keep[1, 2] = False
    assert torch.equal(filled[keep], out[keep])


def test_hc_split_sinkhorn_matches_the_stated_normalization_order() -> None:
    torch.manual_seed(3)
    mix_hc = (2 + HC_MULT) * HC_MULT
    mixes = torch.randn(2, 3, mix_hc)
    hc_scale = torch.randn(3)
    hc_base = torch.randn(mix_hc)

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS)

    raw = mixes[..., 2 * HC_MULT :].reshape(2, 3, HC_MULT, HC_MULT) * hc_scale[2] + hc_base[
        2 * HC_MULT :
    ].reshape(HC_MULT, HC_MULT)
    assert torch.allclose(comb[0, 0], _reference_sinkhorn(raw, SINKHORN_ITERS, EPS)[0, 0], atol=1e-6)

    # Reordering the two axes inside the loop does not raise and does not look wrong -- the matrix
    # is still doubly stochastic to a glance. It just is a different matrix.
    other = torch.softmax(raw, dim=-1) + EPS
    other = other / (other.sum(dim=-1, keepdim=True) + EPS)
    for _ in range(SINKHORN_ITERS - 1):
        other = other / (other.sum(dim=-2, keepdim=True) + EPS)
        other = other / (other.sum(dim=-1, keepdim=True) + EPS)
    assert not torch.allclose(comb[0, 0], other, atol=1e-6)


def test_hc_split_sinkhorn_gates_are_asymmetric() -> None:
    """`pre` is sigmoid plus eps and `post` is twice sigmoid; identical shapes hide that."""
    torch.manual_seed(4)
    mix_hc = (2 + HC_MULT) * HC_MULT
    mixes = torch.randn(5, mix_hc)
    hc_scale = torch.ones(3)
    hc_base = torch.zeros(mix_hc)

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS)

    logits = mixes[:, :HC_MULT]
    assert torch.allclose(pre, torch.sigmoid(logits) + EPS, atol=1e-6)
    assert torch.allclose(post, 2 * torch.sigmoid(mixes[:, HC_MULT : 2 * HC_MULT]), atol=1e-6)

    # An all-zero mixing vector is the sharpest case: the pre-mix floor and the post-gate midpoint
    # are constants, and they are not the same constant.
    zero = torch.zeros(2, mix_hc)
    pre0, post0, comb0 = hc_split_sinkhorn(zero, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS)
    assert torch.allclose(pre0, torch.full_like(pre0, 0.5 + EPS))
    assert torch.allclose(post0, torch.ones_like(post0))

    # Sinkhorn output is doubly stochastic, which is what the alternating rescale buys.
    assert torch.allclose(comb.sum(-1), torch.ones_like(comb.sum(-1)), atol=1e-5)
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-5)
    assert torch.allclose(comb0.sum(-1), torch.ones_like(comb0.sum(-1)), atol=1e-5)


def test_hc_split_sinkhorn_iteration_count_changes_the_result() -> None:
    """A `iters` that is read but ignored would pass the doubly-stochastic check; this does not."""
    torch.manual_seed(5)
    mix_hc = (2 + HC_MULT) * HC_MULT
    mixes = torch.randn(3, mix_hc) * 3
    hc_scale, hc_base = torch.ones(3), torch.zeros(mix_hc)
    one = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, 1, EPS)[2]
    twenty = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS)[2]
    assert not torch.allclose(one, twenty, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 96, 129])
def test_hc_split_sinkhorn_kernel_matches_the_loop(rows: int) -> None:
    """The fused kernel is the loop, to fp32 rounding, including across its own block boundary.

    Row counts on both sides of 32 and just past a second block are the point: a kernel that reads or
    writes a whole block either way is exactly right at 32 rows and wrong at 33.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.kernels import ops

    if not ops._USE_TRITON:
        pytest.skip("triton is unavailable")

    torch.manual_seed(11 + rows)
    mix_hc = (2 + HC_MULT) * HC_MULT
    mixes = torch.randn(1, rows, mix_hc, device="cuda") * 2
    hc_scale = torch.randn(3, device="cuda")
    hc_base = torch.randn(mix_hc, device="cuda")

    want = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS, impl="torch")
    got = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS, impl="triton")

    for a, b in zip(want, got):
        assert b.shape == a.shape
        # fp32 rounding, not bit equality: the kernel sums a row's four columns in a different order
        # than ATen does, so the two agree to the epsilon of the dtype and no further.
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_hc_split_sinkhorn_keeps_the_original_shape_front() -> None:
    """`[b, s, mix_hc]` and `[b, s, h, mix_hc]` both flatten to rows and both come back as they went in."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(12)
    mix_hc = (2 + HC_MULT) * HC_MULT
    for shape in ((1, 5, mix_hc), (2, 3, mix_hc), (1, 2, 3, mix_hc)):
        mixes = torch.randn(*shape, device="cuda")
        hc_scale = torch.randn(3, device="cuda")
        hc_base = torch.randn(mix_hc, device="cuda")
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS)
        assert pre.shape == (*shape[:-1], HC_MULT)
        assert post.shape == (*shape[:-1], HC_MULT)
        assert comb.shape == (*shape[:-1], HC_MULT, HC_MULT)


def test_hc_split_sinkhorn_falls_back_when_the_kernel_cannot_run() -> None:
    """`impl="triton"` on a host tensor is a request, not a promise: it must still be correct."""
    torch.manual_seed(13)
    mix_hc = (2 + HC_MULT) * HC_MULT
    mixes = torch.randn(4, mix_hc)
    hc_scale = torch.randn(3)
    hc_base = torch.randn(mix_hc)
    want = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS, impl="torch")
    got = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC_MULT, SINKHORN_ITERS, EPS, impl="triton")
    for a, b in zip(want, got):
        torch.testing.assert_close(b, a, rtol=0, atol=0)

    # `hc_mult` has to be a power of two for `tl.arange`; three is not, and the loop handles it.
    three = torch.randn(4, (2 + 3) * 3)
    scale, base = torch.randn(3), torch.randn(5 * 3)
    a = hc_split_sinkhorn(three, scale, base, 3, SINKHORN_ITERS, EPS, impl="torch")
    b = hc_split_sinkhorn(three, scale, base, 3, SINKHORN_ITERS, EPS, impl="triton")
    for x, y in zip(a, b):
        torch.testing.assert_close(y, x, rtol=0, atol=0)


def test_fp4_codes_round_half_to_even() -> None:
    """The reference gets its tie rule from a hardware cast it never states; this pins ours."""
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    # Ties are 0.25 (0 vs 0.5 -> 0), 0.75 (0.5 vs 1.0 -> 1.0), 1.25 (1.0 vs 1.5 -> 1.0),
    # 1.75 (1.5 vs 2.0 -> 2.0), 2.5 (2.0 vs 3.0 -> 2.0), 3.5 (3.0 vs 4.0 -> 4.0),
    # 5.0 (4.0 vs 6.0 -> 4.0). Every tie lands on the even code.
    assert _fp4_codes(values).tolist() == [0, 0, 1, 2, 2, 2, 4, 4, 6, 6, 7]
    assert _fp4_values(_fp4_codes(values)).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    # the sign is the code's high bit, and -0 is reachable
    assert _fp4_codes(torch.tensor([-0.25, -6.0, -0.0])).tolist() == [8, 15, 0]


def test_fp4_act_quant_e4m3_scale_is_the_ratio_not_the_amax() -> None:
    """`e4m3(amax / 6)`, which is what makes `x / scale` fill [-6, 6].

    Storing `e4m3(amax)` -- the obvious reading of "an E4M3 scale" -- also round-trips, just with six
    times less of the codebook in use and a scale that is wrong by that factor. The reference's
    `s_local = T.Cast(FP8, amax_local[i] / fp4_max)` is unambiguous, so this is asserted rather than
    left to the round-trip test to catch.
    """
    torch.manual_seed(6)
    x = torch.randn(6, 128, dtype=torch.bfloat16) * 3
    packed, scales = fp4_act_quant_e4m3(x, block_size=32)
    assert packed.shape == (6, 64) and packed.dtype == torch.uint8
    assert scales.shape == (6, 4) and scales.dtype == torch.float8_e4m3fn

    block_amax = x.float().abs().view(6, 4, 32).amax(-1)
    want = (block_amax / FP4_MAX).to(torch.float8_e4m3fn)
    assert torch.equal(scales, want)
    # and the two conventions are not accidentally the same for this input
    assert not torch.equal(scales, block_amax.to(torch.float8_e4m3fn))


def test_fp4_act_quant_e4m3_round_trip_is_within_one_scale() -> None:
    torch.manual_seed(7)
    x = torch.randn(6, 128, dtype=torch.bfloat16) * 3
    packed, scales = fp4_act_quant_e4m3(x, block_size=32)

    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(6, 128)
    scale_full = scales.float().repeat_interleave(32, dim=-1)
    dequant = _fp4_values(codes) * scale_full
    # Normalized values live in [-6, 6]; the widest gap in the codebook is 4 -> 6, so rounding to
    # nearest lands within half of it, i.e. within one scale after multiplying back out.
    assert ((dequant - x.float()).abs() <= scale_full + 1e-6).all()


def test_fp4_act_quant_e4m3_keeps_a_nonzero_scale_for_an_all_zero_block() -> None:
    """The other behavioural difference from the repo's E8M0 quantizer, and the reason for it."""
    zero = torch.zeros(2, 64)
    packed, scales = fp4_act_quant_e4m3(zero, block_size=32)
    assert (scales.float() > 0).all()
    # the amax floor is 6 * 2**-9, so the scale is exactly E4M3's smallest subnormal
    assert torch.allclose(scales.float(), torch.full_like(scales.float(), 2.0**-9))
    assert (packed == 0).all()

    # inplace returns the dequantized tensor rather than the packed codes, as the reference's
    # `inplace=True` does, and an all-zero input survives that round trip unchanged
    out = fp4_act_quant_e4m3(zero.clone(), block_size=32, inplace=True)
    assert out.shape == zero.shape and torch.equal(out, zero)


def test_fp4_act_quant_e4m3_accepts_the_block_size_the_compressed_kv_path_uses() -> None:
    """`Attention._compress_kv` calls this at 16; the indexer's variant is 32."""
    torch.manual_seed(8)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    packed, scales = fp4_act_quant_e4m3(x, block_size=16)
    assert packed.shape == (2, 32) and scales.shape == (2, 4)
    assert torch.equal(fp4_act_quant_e4m3(x, block_size=16)[1], scales)


def test_fp4_act_quant_e4m3_rejects_a_ragged_block() -> None:
    for bad in [torch.randn(4, 48), torch.randn(4, 33), torch.randn(4, 0)]:
        try:
            fp4_act_quant_e4m3(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted a last dim of {bad.size(-1)}")


def test_fp4_codebooks_are_cached_per_device_not_rebuilt_per_call() -> None:
    """The two level tables are the same tensor across calls, which is what removes the H2D.

    Both were `torch.tensor(<host tuple>, device=...)` inside the function, so every quantize paid
    a pageable host-to-device copy of eight or sixteen floats -- a launch-time cost on the decode
    path, and a hard failure inside a CUDA graph capture, where a pageable copy is not recordable.
    The cache is keyed by device, so the observable contract is object identity across calls and a
    separate entry per device.
    """
    v41_kernels._FP4_LEVEL_CACHE.clear()
    v41_kernels._FP4_SIGNED_LEVEL_CACHE.clear()
    x = torch.tensor([0.0, 0.5, 1.5, 4.0, -2.0])

    first = v41_kernels._levels_for(x.device, v41_kernels._FP4_LEVEL_CACHE, v41_kernels._FP4_MAGNITUDES)
    second = v41_kernels._levels_for(x.device, v41_kernels._FP4_LEVEL_CACHE, v41_kernels._FP4_MAGNITUDES)
    assert first is second, "the codebook was rebuilt, so the copy came back"

    signed = [*v41_kernels._FP4_MAGNITUDES, *(-m for m in v41_kernels._FP4_MAGNITUDES)]
    a = v41_kernels._levels_for(x.device, v41_kernels._FP4_SIGNED_LEVEL_CACHE, signed)
    b = v41_kernels._levels_for(x.device, v41_kernels._FP4_SIGNED_LEVEL_CACHE, signed)
    assert a is b and a is not first
    assert v41_kernels._FP4_SIGNED_LEVEL_CACHE is not v41_kernels._FP4_LEVEL_CACHE


def test_fp4_codes_and_values_are_unchanged_by_the_cache() -> None:
    """The values have to be what the uncached construction produced, bit for bit.

    The cache is an optimization on a quantize the whole compressor runs through, so the only
    interesting failure is a stale or wrong-width table -- which would show up as a value, not as a
    crash. `_fp4_codes` and `_fp4_values` are therefore also run against the literal construction
    they replaced, and the two are required to be equal exactly.
    """
    torch.manual_seed(4)
    values = torch.randn(64, 128) * 7.0

    def uncached_codes(normalized: torch.Tensor) -> torch.Tensor:
        """The body of `_fp4_codes` with the level table built the old way, per call."""
        levels = torch.tensor(v41_kernels._FP4_MAGNITUDES, dtype=torch.float32,
                              device=normalized.device)
        magnitude = normalized.abs()
        upper = torch.searchsorted(levels, magnitude, right=False).clamp(
            1, len(v41_kernels._FP4_MAGNITUDES) - 1)
        lower = upper - 1
        to_upper = (magnitude - levels[lower]) > (levels[upper] - magnitude)
        to_upper |= ((magnitude - levels[lower]) == (levels[upper] - magnitude)) & (upper % 2 == 0)
        index = torch.where(to_upper, upper, lower)
        return (index + torch.where(normalized < 0, 8, 0)).to(torch.uint8)

    def uncached_values(codes: torch.Tensor) -> torch.Tensor:
        levels = torch.tensor(
            [*v41_kernels._FP4_MAGNITUDES, *(-m for m in v41_kernels._FP4_MAGNITUDES)],
            dtype=torch.float32, device=codes.device)
        return levels[codes.long()]

    for _ in range(2):  # the second pass reads the cache the first one filled
        codes = _fp4_codes(values)
        assert torch.equal(codes, uncached_codes(values))
        assert torch.equal(_fp4_values(codes), uncached_values(codes))
    assert v41_kernels._FP4_LEVEL_CACHE, "nothing was cached"
