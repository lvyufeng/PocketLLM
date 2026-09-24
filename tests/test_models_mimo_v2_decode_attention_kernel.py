"""The decode step's attention in one kernel: a bound instead of an equality, and why.

`device_attention.decode_output`'s softmax block is deliberately the reference's arithmetic on
smaller shapes -- two `matmul`s around a masked softmax with a sink column -- and what it costs is
dispatch rather than flops. Every tensor in it is at most `[16, 192]`, and at sixteen resident rows
it is `scores` 6.9 ms of a token's host, `softmax` 6.6, `out` 4.8 and the final `view` 0.8: 19.1 ms
for eighteen eager dispatches a layer, forty-eight layers a token.

`mimo_decode_attention` is that block as one kernel, and it is **the one kernel in the MiMo path
that is not bit-exact.** `mimo_rope_rows` could be exact because it is elementwise -- there is no
summation order to get wrong -- and this one has three: the reference's `sum` is a `torch`
reduction whose tree shape is torch's, its two matmuls are cuBLAS's blocking of the same products,
and the kernel walks the span a warp at a time. So the file is organised around what is *not* given
up and how far the answer moves, in that order:

* **The products are not approximated.** `key` and `value` are bfloat16 because the cache is, they
  are widened exactly the way the reference's `.to(torch.float32)` widens them, and a span of one
  key -- where the softmax is the identity and every non-widening step of the kernel cancels -- is
  `torch.equal` to the widened value row. That test is the whole claim in one line.
* **The exponentials are the reference's.** `--use_fast_math` is on for this translation unit and
  turns `expf` into `__expf`, whose few-2^-21 of relative error is the same order as the tolerance,
  so the kernel goes through `double`, which fast math does not substitute.
* **The sum is accumulated in double**, and for the same reason: a thousand keys against a float32
  accumulator is a thousand roundings, and measured over the release's own span that is 1.8e-6
  relative -- two thousand times the bfloat16 the answer is cast to.

What is left is the reference's own reassociation, and the last test in this file is what it costs
*on the token*: a float32 agreement of about 3e-7 relative, which is a hundred-and-twentieth of a
bfloat16 step, and a bfloat16 answer that moves only where that lands on a rounding boundary.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402
from src.models.mimo_v2.device_attention import (  # noqa: E402
    MimoV2DeviceAttention,
    MimoV2KVCache,
    single_pass_attention,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the decode attention needs both the release and a CUDA device",
)

#: A windowed layer and a global one: 2 carries a sink and a 128-slot ring, 5 does neither.
SWA_LAYER, GA_LAYER = 2, 5

OPS = load_cuda_kernel()
HAS_KERNEL = OPS is not None and hasattr(OPS, "mimo_decode_attention")
needs_kernel = pytest.mark.skipif(not HAS_KERNEL, reason="the cuda_kernel extension is not built")


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


def problem(
    kv_heads: int,
    groups: int,
    head_dim: int,
    v_head_dim: int,
    keys: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    seed: int = 0,
    sink: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """A decode step's four tensors at a shape the release has, or one near it.

    The query is float32 because `rotate` returns float32 and the reference's `.to(torch.float32)`
    is a no-op by then; the key and the value are the cache's own width. The values are scaled so
    that a softmax over them has a peak rather than a plateau -- a uniform row would hide a
    denominator that is an approximation.
    """
    card = torch.device("cuda")
    generator = torch.Generator(device=card).manual_seed(seed)
    heads = kv_heads * groups
    query = torch.randn(heads, head_dim, generator=generator, device=card, dtype=torch.float32)
    scale = 0.5 if dtype == torch.float32 else 1.0
    key = torch.randn(
        kv_heads, keys, head_dim, generator=generator, device=card, dtype=torch.float32
    ).to(dtype) * scale
    value = torch.randn(
        kv_heads, keys, v_head_dim, generator=generator, device=card, dtype=torch.float32
    ).to(dtype) * scale
    column = (
        torch.randn(heads, generator=generator, device=card, dtype=torch.float32) * 2.0
        if sink
        else torch.empty(0, device=card, dtype=torch.float32)
    )
    return query, key, value, column, head_dim**-0.5


def reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """`single_pass_attention` with the mask a decode step does not need, `[heads, v_head_dim]`.

    It is the oracle here rather than a second copy of the kernel's arithmetic: the visibility
    bounds of one query over its whole span are `lower = 0, upper = keys - 1`, which is the case
    `single_pass_attention` and `decode_output` agree on to the bit, and the file that says so is
    `test_models_mimo_v2_device_attention.py`.
    """
    heads, keys = query.shape[0], key.shape[1]
    lower = torch.zeros(1, dtype=torch.int64, device=query.device)
    upper = torch.full((1,), keys - 1, dtype=torch.int64, device=query.device)
    out, _ = single_pass_attention(
        query.unsqueeze(1),
        key,
        value,
        lower,
        upper,
        scaling=scaling,
        sink=None if sink.numel() == 0 else sink,
    )
    return out[:, 0, :]


#: `kv_heads, groups, head_dim, v_head_dim, keys`. The first two are the release's own two
#: families at the four-way attention split -- sixteen query heads over two kv heads for a
#: windowed layer, sixty-four over one for a global one, both 192 wide with 128 value channels --
#: and the rest are the shapes around them: one head, a head count that does not fill the block's
#: four warps, a span of one, and the widest head the kernel admits.
GEOMETRY = (
    (2, 8, 192, 128, 128),
    (1, 16, 192, 128, 1),
    (2, 8, 192, 128, 33),
    (1, 1, 32, 32, 7),
    (3, 5, 96, 96, 200),
    (1, 1, 256, 256, 1024),
)

DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@needs_kernel
@pytest.mark.parametrize("kv_heads,groups,head_dim,v_head_dim,keys", GEOMETRY)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sink", (False, True))
def test_the_kernel_is_the_reference_softmax_to_the_rounding(
    kv_heads, groups, head_dim, v_head_dim, keys, dtype, sink
):
    """The bound, at the op, over both families' geometry and the shapes around them.

    Not equality, and the file's docstring says why that is the honest bound rather than a
    convenience: the reference's sum is a `torch` reduction and this walks the span a warp at a
    time, so the last bits of a probability are the reference's *order* and not the reference's
    value. What the number has to be small enough to say is that nothing else moved -- a wrong
    sink, a misread stride or a transcendantal evaluated at the wrong width all land orders of
    magnitude above it. Measured worst case over this grid is 6.5e-7 of the output's peak.

    The float32 arm is the one where the widening is the identity, so a mismatch there is an
    arithmetic error rather than a rounding; float16 is a dtype the config admits even though this
    release does not use it, and it is here because `AT_DISPATCH` has a second case to get wrong.
    """
    query, key, value, column, scaling = problem(
        kv_heads, groups, head_dim, v_head_dim, keys, dtype, seed=kv_heads * 7 + keys, sink=sink
    )
    got = OPS.mimo_decode_attention(query, key, value, column, scaling)
    want = reference(query, key, value, column, scaling)
    assert got.dtype == torch.float32 and got.shape == (1, want.numel())
    assert got.shape == (1, kv_heads * groups * v_head_dim)
    off = (got.view_as(want) - want).abs().max().item()
    peak = want.abs().max().item()
    assert off <= peak * 1e-5, (off, peak, off / peak)


@needs_kernel
def test_a_span_of_one_key_is_the_value_row_widened_and_nothing_else():
    """The strong claim in one line: with one key there is nothing to approximate.

    A span of one makes the score row a single number, so `exp(x - max)` is `exp(0)` and the
    normalised probability is exactly one however the reduction is ordered; what is left of the
    kernel is `1 * widen(value)`, accumulated in double and cast to float32. That is exactly the
    reference's `.to(torch.float32)`, so this is `torch.equal` and not a tolerance -- and a kernel
    that read the bfloat16 row at the wrong width, or contracted the product with an fma, or
    divided by a denominator it had approximated, fails here while passing the bound above.
    """
    for dtype in DTYPES:
        _, key, value, _, scaling = problem(2, 4, 192, 128, 1, dtype, seed=3)
        query = torch.zeros(8, 192, device="cuda", dtype=torch.float32)
        got = OPS.mimo_decode_attention(
            query, key, value, torch.empty(0, device="cuda", dtype=torch.float32), scaling
        )
        want = value[:, 0, :].repeat_interleave(4, dim=0).to(torch.float32)
        assert torch.equal(got.view_as(want), want)


@needs_kernel
def test_the_sink_is_a_column_of_the_softmax_and_not_a_bias():
    """The sink is the reference's, which is the difference between two plausible readings.

    As a column it enters the row maximum and its exponential joins the denominator; as a bias it
    would be added to the output after the attention and would leave the probabilities alone. The
    two agree whenever the sink is not the maximum, so the tests that separate them are the
    extremes: a sink far above the span takes all of the mass and the output goes to zero, and a
    sink at `-inf` contributes `exp(-inf - max) = 0` and is therefore the no-sink row bit for bit.
    """
    query, key, value, _, scaling = problem(2, 8, 192, 128, 64, seed=5)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)

    without = OPS.mimo_decode_attention(query, key, value, empty, scaling)
    assert torch.equal(
        OPS.mimo_decode_attention(
            query, key, value, torch.full((16,), float("-inf"), device="cuda"), scaling
        ),
        without,
    ), "a sink at -inf is a column whose exponential is zero, which is no column at all"

    high = torch.full((16,), 1e4, device="cuda", dtype=torch.float32)
    starved = OPS.mimo_decode_attention(query, key, value, high, scaling)
    assert starved.abs().max().item() < 1e-7, "the sink owns all of the mass"
    assert starved.shape == without.shape


@needs_kernel
def test_a_span_that_came_from_a_wrapped_ring_is_read_at_its_strides():
    """The caller does not copy, so the kernel has to read a view.

    A windowed layer's span is `all_key[:, lower:]` of a ring that has wrapped, which keeps the
    buffer's strides and is not contiguous. The kernel takes the two strides rather than asking for
    a copy, and this is the test that says the two paths are the same read: the view against a
    contiguous copy of it, `torch.equal` because it is the same bytes in the same order.
    """
    slots, kv_heads, head_dim, v_head_dim = 128, 2, 192, 128
    generator = torch.Generator(device="cuda").manual_seed(13)
    ring_key = torch.randn(kv_heads, slots, head_dim, generator=generator, device="cuda")
    ring_value = torch.randn(kv_heads, slots, v_head_dim, generator=generator, device="cuda")
    ring_key = ring_key.to(torch.bfloat16)
    ring_value = ring_value.to(torch.bfloat16)
    query = torch.randn(16, head_dim, generator=generator, device="cuda", dtype=torch.float32)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)

    for lower in (1, 37, 127):
        span_key, span_value = ring_key[:, lower:], ring_value[:, lower:]
        if lower < slots:
            assert not span_key.is_contiguous(), "the slice is the point of the test"
        assert torch.equal(
            OPS.mimo_decode_attention(query, span_key, span_value, empty, head_dim**-0.5),
            OPS.mimo_decode_attention(
                query, span_key.contiguous(), span_value.contiguous(), empty, head_dim**-0.5
            ),
        )


@needs_kernel
def test_the_shared_memory_bound_is_a_key_count_and_not_a_guess():
    """A span has to fit in the block's 48 KiB, and the refusal is a count rather than a shape.

    Four warps of float32 scores is sixteen bytes a key, so the largest span that fits is 3072 and
    3073 is one over. `FOLD_KEYS` is 1024, so nothing in the model is close to it -- the point of
    the test is that the boundary is where the arithmetic says it is, and that a caller past it
    gets an error rather than a launch that corrupts another block's scores.
    """
    query, key, value, _, scaling = problem(1, 1, 192, 128, 3073, seed=17)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    assert OPS.mimo_decode_attention(
        query, key[:, :3072].contiguous(), value[:, :3072].contiguous(), empty, scaling
    ).shape == (1, 128)
    with pytest.raises(RuntimeError):
        OPS.mimo_decode_attention(query, key, value, empty, scaling)


@needs_kernel
def test_the_kernel_refuses_a_call_it_cannot_make_right():
    """A wrong shape or width is refused rather than read at the wrong stride.

    `decode_output` builds all of these correctly, so none is reachable from the model; they are
    here because the op is public and every one of them would otherwise return numbers.
    """
    query, key, value, _, scaling = problem(2, 8, 192, 128, 64, seed=19)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    wide = torch.randn(16, 288, device="cuda", dtype=torch.float32)
    cases = [
        ("a widened query", query.to(torch.bfloat16), key, value, empty),
        ("a three-dimensional query", query.unsqueeze(0), key, value, empty),
        ("a query that is not contiguous", query[:, ::2], key, value, empty),
        ("a key and a value at different widths", query, key, value.float(), empty),
        (
            "a key head count that does not divide the query heads",
            query,
            torch.randn(3, 64, 192, device="cuda", dtype=torch.bfloat16),
            torch.randn(3, 64, 128, device="cuda", dtype=torch.bfloat16),
            empty,
        ),
        ("a span of no keys", query, key[:, :0], value[:, :0], empty),
        ("a transposed key", query, key.transpose(1, 2), value, empty),
        ("a sink that is not a column a head", query, key, value, torch.zeros(3, device="cuda")),
        (
            "a head wider than a warp's share",
            wide,
            torch.randn(2, 64, 288, device="cuda", dtype=torch.bfloat16),
            value,
            empty,
        ),
    ]
    for what, q, k, v, s in cases:
        with pytest.raises(RuntimeError):
            OPS.mimo_decode_attention(q, k, v, s, scaling)


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_the_layer_reaches_the_kernel_when_it_is_built(release, layer_idx):
    """The dispatch, on a released layer: the kernel is what `decode_output` picks.

    An op that exists and is never called passes every equality test above, so the reach is a
    separate claim. It is skipped rather than faked when the extension is absent -- a build without
    it is a legitimate build, and the fallback has a test of its own.
    """
    if not HAS_KERNEL:
        pytest.skip("the cuda_kernel extension is not built")
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    assert device._decode_ops is not None, "the loader built the extension and the layer missed it"
    assert device._no_sink.numel() == 0


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_the_fallback_is_the_reference_when_the_extension_is_not_there(release, layer_idx):
    """With `_decode_ops` taken away the block is the torch one, and the step is the chunk path's.

    `MimoV2DeviceLayer.route` and `MimoV2DeviceAttention.rotate` have the same test for the same
    reason: a build without the extension has to be a supported build, and the only way to know it
    is to run it. The arm is the one `test_models_mimo_v2_device_attention.py` holds to
    `torch.equal`, so a fallback that drifted is caught there and not only here.
    """
    steps = 136
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    device.share_rope_table(device.build_rope_table(steps + 8))
    torch.manual_seed(29)
    hidden = torch.randn(release.layer.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.25

    def walk():
        cache = MimoV2KVCache(
            release.layer, steps + 8, [layer_idx], device="cuda", dtype=torch.bfloat16
        )
        rows, paths = [], set()
        for position in range(steps):
            out = device.forward(hidden.unsqueeze(0), start_pos=position, cache=cache)
            rows.append(out["attn_out_post_o"].clone())
            paths.add(device.last_stats.path)
        return torch.cat(rows, dim=0), paths

    was = device._decode_ops
    assert was is not None or not HAS_KERNEL
    try:
        device._decode_ops = None
        plain, paths = walk()
    finally:
        device._decode_ops = was
    assert paths == {"decode"}, "the one-row path carried every step, in both arms"
    if not HAS_KERNEL:
        pytest.skip("the cuda_kernel extension is not built; both arms would be the reference")

    # **The bound this branch ships under**, and the numbers behind it. At the op the kernel's
    # float32 agrees with the reference's to 3.1e-7 relative over these 136 steps of both families
    # -- a hundred-and-twentieth of a bfloat16 step -- and what survives into the bfloat16 answer
    # is that fraction of the cases where a rounding boundary sat between the two: on the windowed
    # layer, 11 of 136 steps differ, by at most 0.061% of a step's elements, and by at most one
    # bfloat16 step of the peak's magnitude. The global layer does not differ at all, over any
    # step. So the promise is not "nothing moves" but "almost nothing moves", and the two bounds
    # below are the size of that: four bfloat16 steps at the peak, and one element in a hundred.
    shipped, _ = walk()
    assert shipped.shape == plain.shape == (steps, release.layer.hidden_size)
    delta = (shipped.float() - plain.float()).abs()
    peak = plain.float().abs().max().item()
    assert delta.max().item() <= peak * 2.0**-6, (delta.max().item(), peak)
    assert (shipped != plain).float().mean().item() <= 0.01
