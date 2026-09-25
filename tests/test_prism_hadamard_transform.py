"""The ``prism.hadamard`` transform itself: the activation side of the weight fold.

This is the half of the stage that cannot fail loudly.  The packing in
``ptq1_0.py`` is checked against the file's own bytes and the F16 release; the
rotation is checked against ``PrismML-Eng/llama.cpp``'s own code, driven by
``scripts/prism_hadamard_oracle.cpp`` and dumped into
``tests/data/ternary_bonsai_hadamard_transform.json``.  Every digest there is the
fork's fp32 output for one of the three ops it composes -- the gated-DeltaNet
permute, the sign multiply, the Walsh-Hadamard rotation -- so a sign consumed in the
wrong order, a block boundary moved, or the permute transposed shows up as a
mismatch rather than as fluent nonsense.

The sign vectors are the checkpoint's; the activations are synthetic, and that is the
honest limit of this file: what it proves is that the transform *is* the fork's,
not that the model generates.  The weights enter at the kernel task, and the parity
claim against the fork's logits belongs there.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.loader.gguf.prism_hadamard import (
    DEFAULT_GDN_GEOMETRY,
    GDN_GROUPED_WIDTH,
    GdnGeometry,
    HadamardRotation,
    HadamardSpec,
    HadamardSpecError,
    _hadamard_last,
    gdn_group_permute,
    walsh_hadamard_blocks,
)
from tests.hadamard_test_utils import hadamard_spec

ORACLE_PATH = Path(__file__).parent / "data" / "ternary_bonsai_hadamard_transform.json"

#: The modes the fixture uses, and the method each one is the fork's counterpart of.
#: ``gdn`` is not a method: it is the per-tensor pipeline, which needs the tensor name.
FORWARD_MODES = {"signs": "forward", "inverse": "inverse"}


def _oracle() -> dict:
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def rotation() -> HadamardRotation:
    return HadamardRotation(hadamard_spec())


@pytest.fixture(scope="module")
def spec() -> HadamardSpec:
    return hadamard_spec()


def _digest(values: torch.Tensor) -> str:
    array = np.ascontiguousarray(values.detach().to(torch.float32).numpy(), dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _explicit_hadamard(n: int) -> np.ndarray:
    """``H[i][j] = (-1)^popcount(i AND j)``, materialised, for the small sizes.

    The matrix the butterfly is supposed to be, written out from the definition rather
    than from any fast transform -- so this is a check on the butterfly, not a second
    copy of it.
    """
    index = np.arange(n, dtype=np.uint32)
    parity = np.bitwise_and(index[:, None], index[None, :])
    popcount = np.zeros_like(parity)
    while parity.any():
        popcount += parity & 1
        parity >>= 1
    return ((-1.0) ** popcount).astype(np.float32)


# --------------------------------------------------------------------------- #
# The butterfly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 4, 8, 16, 256])
def test_the_butterfly_is_the_sylvester_matrix(n: int) -> None:
    x = torch.arange(1.0, n + 1, dtype=torch.float32)
    assert torch.equal(_hadamard_last(x), torch.from_numpy(_explicit_hadamard(n) @ x.numpy()))


def test_the_butterfly_matches_the_matrix_at_the_real_block_size() -> None:
    """1024 is the checkpoint's block size, and the smallest size the model ever uses."""
    x = torch.arange(1.0, 1025, dtype=torch.float32)
    expected = torch.from_numpy(_explicit_hadamard(1024) @ x.numpy())
    assert torch.equal(_hadamard_last(x), expected)


def test_the_normalized_transform_is_orthogonal() -> None:
    """``(1/sqrt(N)) H`` preserves the norm, which is what makes the fold a rotation.

    A missing or doubled normalization would leave the transform invertible and the
    model running, with every folded weight's scale wrong by sqrt(1024).
    """
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(3, 2048, generator=generator)
    y = walsh_hadamard_blocks(x, 1024)
    assert torch.allclose(y.reshape(3, 2, 1024).norm(dim=-1), x.reshape(3, 2, 1024).norm(dim=-1), rtol=1e-5)


def test_the_scale_is_exact_so_both_orders_agree_bit_for_bit() -> None:
    """Why the fixture can pin bytes instead of a tolerance.

    ``1/sqrt(1024)`` is ``2^-5``, exactly representable, so scaling before the
    butterflies and scaling after them round identically -- every intermediate is the
    same number times a power of two.  A block size that is not a power of four would
    not have that property and the digest would have to become a tolerance.
    """
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(2, 1024, generator=generator)
    scale = torch.tensor(1.0 / 1024 ** 0.5)
    assert scale.item() == 2.0 ** -5
    normalized = walsh_hadamard_blocks(x, 1024)
    assert torch.equal(normalized, _hadamard_last(x * scale))
    assert torch.equal(normalized, _hadamard_last(x) * scale)
    # And the normalization is present: without it the result is 32 times larger.
    assert torch.allclose(_hadamard_last(x), normalized * 32.0, rtol=1e-6)


def test_blocks_are_independent() -> None:
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(2, 3072, generator=generator)
    moved = x.clone()
    moved[:, 1024:2048] += 5.0
    before, after = walsh_hadamard_blocks(x, 1024), walsh_hadamard_blocks(moved, 1024)
    assert torch.equal(before[:, :1024], after[:, :1024])
    assert torch.equal(before[:, 2048:], after[:, 2048:])
    assert not torch.equal(before[:, 1024:2048], after[:, 1024:2048])


def test_the_leading_dimensions_are_untouched() -> None:
    """The transform is along the last axis only, at any rank the caller passes."""
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(2, 3, 1024, generator=generator)
    y = walsh_hadamard_blocks(x, 1024)
    assert y.shape == x.shape
    for i in range(2):
        for j in range(3):
            assert torch.equal(y[i, j], walsh_hadamard_blocks(x[i, j], 1024))


@pytest.mark.parametrize(
    "block_size, width, message",
    [
        (1000, 2000, "not a power of two"),
        (0, 1024, "not a power of two"),
        # 5120 is five blocks of 1024 and two and a half of 2048.
        (2048, 5120, "not a multiple"),
    ],
)
def test_a_shape_the_butterfly_cannot_take_raises(block_size: int, width: int, message: str) -> None:
    with pytest.raises(HadamardSpecError, match=message):
        walsh_hadamard_blocks(torch.zeros(1, width), block_size)


# --------------------------------------------------------------------------- #
# The two directions
# --------------------------------------------------------------------------- #


def test_forward_is_the_signs_then_the_rotation(rotation: HadamardRotation) -> None:
    generator = torch.Generator().manual_seed(15)
    x = torch.randn(2, 5120, generator=generator)
    signs = rotation.signs_for(5120)
    assert torch.equal(rotation.forward(x), walsh_hadamard_blocks(x * signs, 1024))


def test_the_inverse_is_the_rotation_then_the_signs(rotation: HadamardRotation) -> None:
    """``token_embd``'s order, and not the forward one with the signs moved."""
    generator = torch.Generator().manual_seed(16)
    x = torch.randn(2, 5120, generator=generator)
    signs = rotation.signs_for(5120)
    assert torch.equal(rotation.inverse(x), walsh_hadamard_blocks(x, 1024) * signs)
    assert not torch.equal(rotation.inverse(x), rotation.forward(x))


def test_forward_then_inverse_is_the_identity(rotation: HadamardRotation) -> None:
    """The fold is a rotation, so undoing it has to return the input.

    Not bit-exact -- two fp32 passes over 5120-wide rows is about 1.3e-7 relative --
    which is why the fixture below is a digest and this is a tolerance.
    """
    generator = torch.Generator().manual_seed(17)
    for width in (5120, 6144, 17408):
        x = torch.randn(2, width, generator=generator)
        back = rotation.inverse(rotation.forward(x))
        assert torch.allclose(back, x, rtol=1e-5, atol=1e-6), width
        assert (back - x).abs().max() < 1e-5 * x.abs().max()


def test_the_signs_are_per_width_and_not_shared(rotation: HadamardRotation) -> None:
    vectors = {width: rotation.signs_for(width) for width in (5120, 6144, 17408)}
    assert {width: len(vector) for width, vector in vectors.items()} == {5120: 5120, 6144: 6144, 17408: 17408}
    assert not torch.equal(vectors[5120], vectors[6144][:5120])
    for vector in vectors.values():
        assert set(vector.unique().tolist()) <= {-1.0, 1.0}


def test_an_undeclared_width_raises(rotation: HadamardRotation) -> None:
    with pytest.raises(HadamardSpecError, match="no sign vector declared for width 4096"):
        rotation.signs_for(4096)


# --------------------------------------------------------------------------- #
# The gated-DeltaNet permute
# --------------------------------------------------------------------------- #


def test_the_permute_is_the_fork_s_two_orders() -> None:
    """``[head_dim, groups, rep] -> [head_dim, rep, groups]`` in the fork's axis order.

    Written out on a tensor that labels every position, so the assertion is on where
    each feature goes rather than on a shape.  The fork's two orders are ``ne`` orders,
    ``ne[0]`` fastest, which is why the flat feature vector reads as ``(rep, groups,
    head_dim)`` here and as ``(groups, rep, head_dim)`` after the swap.
    """
    width = GDN_GROUPED_WIDTH
    geometry = DEFAULT_GDN_GEOMETRY
    assert (geometry.value_heads, geometry.groups, geometry.rep) == (48, 16, 3)
    head_dim = geometry.head_dim(width)
    assert head_dim == 128

    x = torch.arange(width, dtype=torch.float32).reshape(1, width)
    tiled = x.reshape(geometry.rep, geometry.groups, head_dim)
    grouped = gdn_group_permute(x, geometry).reshape(geometry.groups, geometry.rep, head_dim)
    for rep in range(geometry.rep):
        for group in range(geometry.groups):
            assert torch.equal(grouped[group, rep], tiled[rep, group])
    # A no-op or a head-local shuffle would pass a shape check and fail this.
    assert not torch.equal(grouped.reshape(-1), tiled.reshape(-1))


def _undo_group_permute(y: torch.Tensor, geometry: GdnGeometry) -> torch.Tensor:
    """The swap read the other way round: ``(groups, rep, head_dim) -> (rep, groups, ...)``.

    The permute is a transposition of two views of the same bytes, so undoing it is the
    same swap applied to the output's own labelling rather than the same function
    applied twice.
    """
    width = int(y.shape[-1])
    head_dim = geometry.head_dim(width)
    lead = y.shape[:-1]
    return (
        y.reshape(*lead, geometry.groups, geometry.rep, head_dim)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, width)
    )


def test_the_documented_inverse_undoes_the_permute() -> None:
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(2, GDN_GROUPED_WIDTH, generator=generator)
    assert torch.equal(_undo_group_permute(gdn_group_permute(x), DEFAULT_GDN_GEOMETRY), x)


def test_a_permute_with_the_wrong_geometry_still_has_the_right_width() -> None:
    """Why the geometry has to be right rather than merely consistent with the width.

    Every grouping of 6144 that the geometry accepts produces a full-width vector, so
    nothing downstream can tell a wrong permute from a right one by shape or by norm.
    """
    x = torch.arange(GDN_GROUPED_WIDTH, dtype=torch.float32).reshape(1, GDN_GROUPED_WIDTH)
    right = gdn_group_permute(x, GdnGeometry(value_heads=48, groups=16))
    wrong = gdn_group_permute(x, GdnGeometry(value_heads=48, groups=3))
    assert right.shape == wrong.shape
    assert torch.allclose(right.norm(), wrong.norm())
    assert not torch.equal(right, wrong)


@pytest.mark.parametrize(
    "geometry, width, message",
    [
        # 7 heads does not divide 6144, so there is no head width to permute within.
        (GdnGeometry(value_heads=7, groups=7), GDN_GROUPED_WIDTH, "not divisible"),
        (GdnGeometry(value_heads=48, groups=16), 5120, "not divisible"),
    ],
)
def test_a_geometry_that_does_not_fit_the_width_raises(geometry: GdnGeometry, width: int, message: str) -> None:
    with pytest.raises(HadamardSpecError, match=message):
        gdn_group_permute(torch.zeros(1, width), geometry)


@pytest.mark.parametrize("value_heads, groups", [(0, 16), (48, 0), (48, 7)])
def test_an_impossible_geometry_is_refused_at_construction(value_heads: int, groups: int) -> None:
    with pytest.raises(HadamardSpecError):
        GdnGeometry(value_heads=value_heads, groups=groups)


# --------------------------------------------------------------------------- #
# The per-tensor pipeline
# --------------------------------------------------------------------------- #


def test_apply_dispatches_the_three_kinds_of_declared_tensor(rotation: HadamardRotation) -> None:
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(2, 5120, generator=generator)
    gdn = torch.randn(2, GDN_GROUPED_WIDTH, generator=generator)

    # A folded matrix weight: signs then rotation, no permute.
    assert torch.equal(
        rotation.apply("blk.0.ffn_gate.weight", hidden), rotation.forward(hidden)
    )
    # output.weight reads the embedding table the other way round but is a matmul.
    assert torch.equal(rotation.apply("output.weight", hidden), rotation.forward(hidden))
    # The gated-DeltaNet output: permute first, then signs and rotation.
    assert torch.equal(
        rotation.apply("blk.0.ssm_out.weight", gdn),
        rotation.forward(gdn_group_permute(gdn)),
    )
    assert not torch.equal(rotation.apply("blk.0.ssm_out.weight", gdn), rotation.forward(gdn))
    # The embedding table alone takes the inverse, with no permute.
    assert torch.equal(rotation.apply("token_embd.weight", hidden), rotation.inverse(hidden))


def test_a_tensor_the_block_does_not_declare_is_refused(rotation: HadamardRotation) -> None:
    """A folded weight is a fact about the file, not something to infer from a name."""
    with pytest.raises(HadamardSpecError, match="is not declared"):
        rotation.apply("blk.0.ssm_conv1d.weight", torch.zeros(1, 5120))
    with pytest.raises(HadamardSpecError, match="is not declared"):
        rotation.apply("blk.0.attn_q_norm.weight", torch.zeros(1, 5120))


def test_the_gdn_permute_only_touches_the_tensor_it_is_declared_for(rotation: HadamardRotation) -> None:
    """``gdn_v_grouped`` is about one tensor role, so the check is on the role."""
    generator = torch.Generator().manual_seed(20)
    x = torch.randn(1, GDN_GROUPED_WIDTH, generator=generator)
    assert torch.equal(rotation.apply("blk.0.ffn_down.weight", x), rotation.forward(x))
    assert not torch.equal(rotation.apply("blk.0.ssm_out.weight", x), rotation.forward(x))


def test_a_rotation_without_the_gdn_flag_leaves_the_width_alone(spec: HadamardSpec) -> None:
    """The flag is the file's; without it the permute must not be applied at all."""
    bare = HadamardRotation(
        HadamardSpec(**{**spec.__dict__, "gdn_v_grouped": False})
    )
    generator = torch.Generator().manual_seed(21)
    x = torch.randn(1, GDN_GROUPED_WIDTH, generator=generator)
    assert torch.equal(bare.apply("blk.0.ssm_out.weight", x), bare.forward(x))


# --------------------------------------------------------------------------- #
# Against the fork
# --------------------------------------------------------------------------- #


def test_every_case_matches_the_fork_bit_for_bit(rotation: HadamardRotation) -> None:
    fixture = _oracle()
    assert fixture["block_size"] == rotation.block_size == 1024
    assert fixture["gdn_geometry"] == {"value_heads": rotation.gdn.value_heads, "groups": rotation.gdn.groups}
    assert {case["mode"] for case in fixture["cases"]} == {"signs", "gdn", "permute", "inverse"}
    for case in fixture["cases"]:
        width, mode = int(case["width"]), case["mode"]
        x = torch.from_numpy(
            np.random.default_rng(fixture["input"]["seed"] + width).standard_normal(
                (fixture["input"]["rows"], width), dtype=np.float32
            )
        )
        assert _digest(x) == case["x_sha256"], case["name"]
        if mode == "permute":
            got = gdn_group_permute(x, rotation.gdn)
        elif mode == "gdn":
            got = rotation.apply(case["tensor"], x)
        else:
            got = getattr(rotation, FORWARD_MODES[mode])(x)
        # The digest is the whole comparison: the fork's own fp32 bytes, not a
        # rounding-tolerant restatement of the same arithmetic.
        assert _digest(got) == case["out_sha256"], case["name"]
        assert float(got.norm()) == pytest.approx(
            float(case["out_rms"]) * np.sqrt(got.numel()), rel=1e-6
        ), case["name"]
        assert float(got.abs().max()) == pytest.approx(case["out_max_abs"], rel=1e-6), case["name"]


def test_the_fixture_covers_the_three_folded_widths(rotation: HadamardRotation) -> None:
    """A width the block declares and the fixture never exercises is an untested one."""
    covered = {int(case["width"]) for case in _oracle()["cases"]}
    assert covered == set(rotation.spec.sign_widths)


def test_the_oracle_is_not_a_restatement_of_our_own_arithmetic() -> None:
    """The fixture has to be able to fail, and this is the check that it can.

    Two of the five cases differ by the permute alone and two more share a width, so a
    fixture whose digests all agreed would not be pinning anything.  Asserting that the
    five are distinct is what says the file discriminates the ops it names.
    """
    digests = [case["out_sha256"] for case in _oracle()["cases"]]
    assert len(set(digests)) == len(digests)
    by_name = {case["name"]: case for case in _oracle()["cases"]}
    assert by_name["forward_gdn"]["out_sha256"] != by_name["permute_gdn"]["out_sha256"]
    assert by_name["forward_gdn"]["x_sha256"] == by_name["permute_gdn"]["x_sha256"]
