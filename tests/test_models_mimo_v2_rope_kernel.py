"""The decode step's rotation in one kernel: the reference's bits, and where it is reached.

`device_attention.rope_rows` is `apply_partial_rope` with the head and sequence axes folded away,
and a decode step calls it thirty-two times a token -- the query and the key of every one of the
forty-eight layers. It is eight tensor operations, so it is cheap in flops and expensive on the
host: 123.8 us a call on this box against a dispatch at about five, which is 11.9 ms of a token's
host time for a split, a half swap, two multiplies and an add.

`mimo_rope_rows` is those operations as one kernel, and the point of the file is that it is the
*same* operations. Two things about the reference make that easy to get wrong and are the reason
this test is `torch.equal` rather than `allclose`:

* The products round separately. The row is bfloat16 and the cosine is float32, so `rope * cos`
  rounds once in float32 and the addition rounds again. A kernel that contracts the pair into an
  fma -- which `--use_fast_math` is free to do and this translation unit is compiled with -- lands
  on a different last bit for a large fraction of rows.
* The half swap is a sign flip, not a subtraction. `rotated` is `cat(-x2, x1)`, so the first half
  of the sum is `p1 + (-p2)` and not `fma(-x2, sin, p1)`.

Neither is visible from a tolerance. So the suite runs the released geometry, both dtypes the
checkpoint's families actually carry, and a set of shapes the release does not have, and asks for
equality on every one of them.

The second half of the file is the dispatch: `MimoV2DeviceAttention.rotate` has to reach the
kernel when the extension is built and `rope_rows` when it is not, and the full decode step has to
be unchanged between the two.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402
from src.models.mimo_v2.device_attention import (  # noqa: E402
    MimoV2DeviceAttention,
    MimoV2KVCache,
    rope_rows,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the decode rotation needs both the release and a CUDA device",
)

#: A windowed layer and a global one: 2 carries a sink, 5 does not.
SWA_LAYER, GA_LAYER = 2, 5

OPS = load_cuda_kernel()
HAS_KERNEL = OPS is not None and hasattr(OPS, "mimo_rope_rows")
needs_kernel = pytest.mark.skipif(not HAS_KERNEL, reason="the cuda_kernel extension is not built")


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


def pivot(
    heads: int,
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A row set and its cosine, on the card, in the shapes the two callers produce.

    `cos` and `sin` are `[1, rope_dim]` because that is what `decode_output` slices out of the
    shared table; the values are a real rotation's rather than noise so that a sign error in the
    half swap has something to be wrong about.
    """
    card = torch.device("cuda")
    generator = torch.Generator(device=card).manual_seed(seed)
    states = torch.randn(heads, head_dim, generator=generator, device=card, dtype=dtype)
    angles = torch.rand(1, rope_dim, generator=generator, device=card, dtype=torch.float32)
    return states, torch.cos(angles), torch.sin(angles)


#: The released geometry, and shapes around it. The release is 192 wide with 64 rotated of it and
#: 64 or 8 heads a layer, which is 16 or 2 heads a rank at the four-way split; the rest are here
#: because the kernel's geometry is `head_dim`, `rope_dim` and the grid, and a shape the release
#: does not have is where an index is wrong rather than merely retuned.
GEOMETRY = (
    (16, 192, 64),
    (8, 192, 64),
    (2, 192, 64),
    (1, 192, 64),
    (64, 192, 64),
    (8, 192, 128),
    (8, 192, 2),
    (5, 130, 66),
    (3, 64, 64),
    (7, 256, 128),
)


@needs_kernel
@pytest.mark.parametrize("heads,head_dim,rope_dim", GEOMETRY)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_the_kernel_is_the_reference_row_by_row(heads, head_dim, rope_dim, dtype):
    """Equality, over the geometry the checkpoint has and the shapes around it.

    The bfloat16 case is the released one and the float32 case is the one where the widening is
    the identity, so a mismatch there is an arithmetic error and not a rounding; the float16 case
    is a dtype the checkpoint's config admits even though this release does not use it.
    """
    for seed in range(3):
        states, cos, sin = pivot(heads, head_dim, rope_dim, dtype, seed=seed)
        want = rope_rows(states, cos, sin, rope_dim)
        got = OPS.mimo_rope_rows(states, cos, sin, rope_dim)
        assert got.dtype == want.dtype == torch.float32, "the reference returns float32"
        assert got.shape == want.shape == (heads, head_dim)
        assert torch.equal(got, want), (
            f"{heads}x{head_dim} rope {rope_dim} {dtype} seed {seed}: "
            f"{(got - want).abs().max().item()} off"
        )


@needs_kernel
def test_the_widening_is_exact_and_not_a_recomputation():
    """A bfloat16 row's rotation is the rotation of its widened row, which is the whole claim.

    `rope * cos` promotes the row to float32 because the cosine is float32, so the reference reads
    the bfloat16 value *exactly* and then rounds; a kernel that read the row back at another width
    would differ in the last bits of every coordinate. Stating it separately is what makes the
    bfloat16 case above a statement about rounding rather than about the cast.
    """
    states, cos, sin = pivot(8, 192, 64, torch.bfloat16, seed=11)
    wide = rope_rows(states.to(torch.float32), cos, sin, 64)
    assert torch.equal(OPS.mimo_rope_rows(states, cos, sin, 64), wide)


@needs_kernel
def test_the_tail_past_the_rotated_width_is_carried_across_not_rotated():
    """`rope_dim` of 64 in a 192-wide row leaves 128 coordinates the rotation does not touch.

    They are not a copy in the reference: `torch.cat` promotes them to float32 alongside the
    rotated half, so the kernel widens them too. A kernel that wrote them at the input's width, or
    that rotated the whole row against a zero-padded table, would still return the right shape.
    """
    states, cos, sin = pivot(4, 192, 64, torch.bfloat16, seed=5)
    got = OPS.mimo_rope_rows(states, cos, sin, 64)
    assert torch.equal(got[:, 64:], states[:, 64:].to(torch.float32))
    assert not torch.equal(got[:, :64], states[:, :64].to(torch.float32))


@needs_kernel
def test_a_row_whose_cosine_is_the_identity_is_the_row():
    """`cos = 1, sin = 0` is the arithmetic's fixed point and not the kernel's shortcut.

    It costs one kernel launch to say, and it is the cheap way to catch a half swap that has been
    turned into a subtraction: `x - 0` and `x + (-0)` agree, but a swap that reads the *rotated*
    row's own coordinate returns `x1 * 1 + x1 * 0` for the second half, which is wrong for every
    row that is not symmetric.
    """
    states, _, _ = pivot(3, 192, 64, torch.bfloat16, seed=9)
    cos = torch.ones(1, 64, device="cuda", dtype=torch.float32)
    sin = torch.zeros(1, 64, device="cuda", dtype=torch.float32)
    assert torch.equal(
        OPS.mimo_rope_rows(states, cos, sin, 64), states.to(torch.float32)
    )


@needs_kernel
def test_the_kernel_refuses_a_call_it_cannot_make_right():
    """A wrong shape is refused rather than read at the wrong stride.

    `decode_output` always hands in a contiguous `view`, so none of these is reachable from the
    model; they are here because the op is public and a silent stride error returns numbers.
    """
    states, cos, sin = pivot(8, 192, 64, torch.bfloat16)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states.view(1, 8, 192), cos, sin, 64)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states.t().contiguous().t(), cos, sin, 64)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states, cos.to(torch.bfloat16), sin, 64)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states, cos[:, :32].contiguous(), sin[:, :32].contiguous(), 64)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states, cos, sin, 65)
    with pytest.raises(RuntimeError):
        OPS.mimo_rope_rows(states, cos, sin, 256)


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_the_layer_reaches_the_kernel_when_it_is_built(release, layer_idx):
    """The dispatch, on a released layer: the kernel is what `rotate` picks, and it is pickable.

    This is the half the equality tests above cannot state, because an op that exists and is never
    called passes them all. It is skipped rather than faked when the extension is absent -- a
    fallback is a legitimate build and asserting on it here would fail one.
    """
    if not HAS_KERNEL:
        pytest.skip("the cuda_kernel extension is not built")
    layer = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    assert layer._rope_ops is not None, "the loader built the extension and `rotate` did not take it"
    states, cos, sin = pivot(layer.shape.num_q_heads, layer.shape.head_dim, layer.shape.rope_dim,
                             torch.bfloat16, seed=3)
    assert torch.equal(
        layer.rotate(states, cos, sin, layer.shape.rope_dim),
        rope_rows(states, cos, sin, layer.shape.rope_dim),
    )


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_the_fallback_is_the_reference_when_the_extension_is_not_there(release, layer_idx):
    """With `_rope_ops` taken away the call is `rope_rows` itself, to the bit.

    `MimoV2DeviceLayer.route`'s fallback has the same test for the same reason: a build without the
    extension has to be a supported build, and the only way to know it is to run it.
    """
    layer = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    layer._rope_ops = None
    states, cos, sin = pivot(layer.shape.num_q_heads, layer.shape.head_dim, layer.shape.rope_dim,
                             torch.bfloat16, seed=4)
    assert torch.equal(
        layer.rotate(states, cos, sin, layer.shape.rope_dim),
        rope_rows(states, cos, sin, layer.shape.rope_dim),
    )


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_a_decode_step_is_the_same_step_whether_the_kernel_carries_the_rotation(
    release, layer_idx
):
    """The whole decode step, twice, one cache each: the kernel is not allowed to move a number.

    One rotation being exact does not make a step exact -- the query and the key are rotated and
    then multiplied together, and an error of one ulp in the query is an error in the scores. So
    the two arms walk the same positions from the same hidden rows and the outputs are compared
    with `torch.equal`, which is the standard `decode_output` is held to against the chunk path.
    """
    steps = 136
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    device.share_rope_table(device.build_rope_table(steps + 8))

    torch.manual_seed(23)
    hidden = torch.randn(release.layer.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.25

    def walk(reach_the_kernel: bool):
        was = device._rope_ops
        device._rope_ops = was if reach_the_kernel else None
        try:
            cache = MimoV2KVCache(
                release.layer, steps + 8, [layer_idx], device="cuda", dtype=torch.bfloat16
            )
            rows = []
            for position in range(steps):
                out = device.forward(hidden.unsqueeze(0), start_pos=position, cache=cache)
                rows.append(out["attn_out_post_o"].clone())
            return torch.cat(rows, dim=0)
        finally:
            device._rope_ops = was

    slow = walk(False)
    if not HAS_KERNEL:
        pytest.skip("the cuda_kernel extension is not built; both arms would be the reference")
    fast = walk(True)
    assert fast.shape == slow.shape == (steps, release.layer.hidden_size)
    assert torch.equal(fast, slow), (fast.float() - slow.float()).abs().max().item()
