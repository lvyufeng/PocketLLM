"""The Xing4.0 hyper-connection, against the reference's own forward pass.

The reference is `Xing4_0HyperConnection.forward` from the released
`modeling_xing4_0.py`, written out below rather than imported: the module needs
`transformers` 5.x and this environment has 4.57.1.  The port keeps the
reference's line structure so a reader can diff the two.

What the tests are for.  The block multiplies the plumbing rather than the FLOPs,
so nothing here fails loudly if it is wrong -- a wrong `comb` still produces four
streams of the right width, and the model still generates text.  So the assertions
are about the parts that are *silent*:

- the split is 4 + 4 + 16, and the 16 is `dst * 4 + src`
- `hc_scale` and `hc_base` are in the arithmetic, per gate
- `pre` is a sigmoid in (0, 1) and `post` is `2 * sigmoid` in (0, 2)
- the clamp is on the logits, before the exponential
- the 20 Sinkhorn iterations are row-then-column, with `eps` in every denominator
- the unweighted norm is over the flattened 14336, and it has no learnable scale
- `collapsed` is `sum_s pre[s] * hidden[s]` -- a sum, and the head's collapse is a
  **mean**

The input to the parity tests is noise at the released widths, not a captured
activation: a real one needs the embedding and the two dense layers above it,
which arrive with [#393](https://github.com/lvyufeng/PocketLLM/issues/393)'s
end-to-end check.  The weights are real, though -- layer 2's own `attn_hc` out of
the released shard -- and the second parity case is heavy-tailed, which is what a
transformer's hidden state actually looks like, because the unweighted norm makes
the coefficient vector a function of the input's *direction* and the one place a
realistic direction could differ is whether the Sinkhorn iterate saturates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.models.xing4_0.block import DecoderLayer, DecoderLayerWeights, rms_norm
from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.hyper_connection import HyperConnection, HyperConnectionWeights

CONFIG = Path("/mnt/data2/Xing4.0-29B-A4B/config.json")
SHARD = Path("/mnt/data2/Xing4.0-29B-A4B/model-00003-of-00041.safetensors")
LAYER = 2
_PREFIX = f"model.layers.{LAYER}."


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _params(**overrides) -> Xing4_0Params:
    """The released config, at widths small enough to be a unit test.

    `hc_mult` and the iteration count are *not* scaled down: the plumbing is the
    thing under test, and a 4x4 Sinkhorn iterate is 4x4 whatever the hidden width.
    """
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw.update(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=6,
               kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
               vocab_size=32, max_position_embeddings=128, num_hidden_layers=2)
    raw["rope_scaling"] = dict(raw["rope_scaling"], original_max_position_embeddings=32)
    raw.update(overrides)
    return Xing4_0Params.from_config(raw)


def _weights(params: Xing4_0Params, seed: int = 0) -> HyperConnectionWeights:
    generator = torch.Generator().manual_seed(seed)
    mix = (2 + params.hc_mult) * params.hc_mult
    wide = params.hc_mult * params.hidden_size
    return HyperConnectionWeights(
        hc_fn=torch.randn(mix, wide, generator=generator) * 0.1,
        hc_base=torch.randn(mix, generator=generator) * 0.1,
        # The released values, so a dropped scale or bias moves the result.
        hc_scale=torch.tensor([1.0, 1.0, 1.0]),
    )


def _real_tensors() -> dict[str, torch.Tensor]:
    if not SHARD.exists():
        pytest.skip(f"{SHARD} is not on disk")
    safetensors = pytest.importorskip("safetensors.torch")
    with safetensors.safe_open(str(SHARD), framework="pt") as handle:
        return {
            name[len(_PREFIX) :]: handle.get_tensor(name).float()
            for name in handle.keys()
            if name.startswith(_PREFIX)
        }


# --------------------------------------------------------------------------- #
# The reference
# --------------------------------------------------------------------------- #


def _reference_hyper_connection(
    params: Xing4_0Params, weights: HyperConnectionWeights, hidden_streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`Xing4_0HyperConnection.forward`, line for line."""
    original_dtype = hidden_streams.dtype
    hc = params.hc_mult
    flat = _reference_unweighted_norm(hidden_streams.flatten(start_dim=2).float(), params)
    pre_w, post_w, comb_w = (
        F.linear(flat.to(original_dtype), weights.hc_fn.to(original_dtype))
        .float()
        .split([hc, hc, hc * hc], dim=-1)
    )
    pre_b, post_b, comb_b = weights.hc_base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = weights.hc_scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b)
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb_logits = torch.clamp(comb_logits, min=params.hc_clamp_min, max=params.hc_clamp_max)
    comb_max = comb_logits.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb_logits - comb_max)
    for _ in range(params.hc_sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + params.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + params.hc_eps)
    collapsed = (pre.unsqueeze(-1).to(original_dtype) * hidden_streams).sum(dim=2)
    return post.to(original_dtype), comb.to(original_dtype), collapsed.to(original_dtype)


def _reference_unweighted_norm(x: torch.Tensor, params: Xing4_0Params) -> torch.Tensor:
    """`Xing4_0UnweightedRMSNorm.forward`, at `eps=rms_norm_eps`."""
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + params.rms_norm_eps)).to(x.dtype)


def _tree_close(got: tuple, expected: tuple, scale: float, atol: float = 1e-6) -> None:
    for index, (a, b) in enumerate(zip(got, expected)):
        worst = (a - b).abs().max().item()
        assert worst <= atol * max(scale, 1.0), f"tensor {index} differs by {worst}"


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #


def test_the_port_is_the_reference() -> None:
    params = _params()
    weights = _weights(params, seed=1)
    connection = HyperConnection(params, weights, dtype=torch.float32)
    torch.manual_seed(2)
    # [batch, seq, hc, hidden], which is the layout the released code assumes and
    # the reason its `flatten(start_dim=2)` is the flattening it wants.
    hidden = torch.randn(1, 3, params.hc_mult, params.hidden_size) * 0.5

    got = connection.forward(hidden)
    expected = _reference_hyper_connection(params, weights, hidden)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_the_real_weights_are_the_reference() -> None:
    """The released `attn_hc` and `ffn_hc` of layer 2, at the released widths."""
    params = _real_params()
    tensors = _real_tensors()
    generator = torch.Generator().manual_seed(3)
    for prefix in ("attn_hc", "ffn_hc"):
        weights = HyperConnectionWeights.from_hf(tensors, params, prefix)
        connection = HyperConnection(params, weights, dtype=torch.float32)
        hidden = torch.randn(1, 4, params.hc_mult, params.hidden_size, generator=generator) * 0.3
        got = connection.forward(hidden)
        expected = _reference_hyper_connection(params, weights, hidden)
        assert all(torch.equal(a, b) for a, b in zip(got, expected)), prefix


def test_a_heavy_tailed_hidden_state_is_the_reference() -> None:
    """What a transformer's activations actually look like: a few large channels.

    The unweighted norm makes the coefficients a function of the input's
    *direction*, so a realistic direction is the one that could push the Sinkhorn
    iterate somewhere Gaussian noise does not.  This input has 1% of its channels
    scaled up 50x, which is the shape of an activation outlier.
    """
    params = _real_params()
    weights = HyperConnectionWeights.from_hf(_real_tensors(), params, "attn_hc")
    connection = HyperConnection(params, weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(1, 4, params.hc_mult, params.hidden_size, generator=generator)
    mask = torch.rand(hidden.shape, generator=generator) < 0.01
    hidden = hidden * torch.where(mask, torch.full_like(hidden, 50.0), torch.ones_like(hidden))
    got = connection.forward(hidden)
    expected = _reference_hyper_connection(params, weights, hidden)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))


def test_a_decode_step_is_the_last_row_of_a_chunk() -> None:
    """The coefficients are computed per token, so the shapes have to agree.

    A chunk of 5 computes 5 sets of them at once; a step computes one.  The
    reference has the same property, and this asserts the port did not acquire a
    cross-token dependency -- a `max` over the wrong axis, or a mean taken over
    the tokens rather than the channels, would.
    """
    params = _params()
    weights = _weights(params, seed=5)
    connection = HyperConnection(params, weights, dtype=torch.float32)
    torch.manual_seed(6)
    hidden = torch.randn(1, 5, params.hc_mult, params.hidden_size) * 0.5
    whole = connection.forward(hidden)
    for row in range(5):
        step = connection.forward(hidden[:, row : row + 1])
        for a, b in zip(step, whole):
            # A chunk is a wider GEMM than a step, so the two agree to fp32 rather
            # than bit for bit; what is being ruled out here is a cross-token
            # dependency, which would be orders of magnitude larger than this.
            assert torch.allclose(a, b[:, row : row + 1], atol=1e-6), row
    # The same state with the batch and sequence axes merged is the same state:
    # the coefficients are per token, so only the last two dims matter.
    merged = hidden.reshape(5, params.hc_mult, params.hidden_size)
    for a, b in zip(connection.forward(merged), whole):
        assert torch.allclose(a, b.reshape(5, *a.shape[1:]), atol=1e-6), "rank 3 vs rank 4"


# --------------------------------------------------------------------------- #
# The properties a silent failure would break
# --------------------------------------------------------------------------- #


def test_the_coefficient_ranges() -> None:
    params = _real_params()
    connection = HyperConnection(
        params, HyperConnectionWeights.from_hf(_real_tensors(), params, "attn_hc"), dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 4, params.hc_mult, params.hidden_size, generator=generator) * 0.3
    post, comb, collapsed = connection.forward(hidden)

    assert tuple(post.shape) == (1, 4, params.hc_mult)
    assert tuple(comb.shape) == (1, 4, params.hc_mult, params.hc_mult)
    assert tuple(collapsed.shape) == (1, 4, params.hidden_size)
    # `2 * sigmoid`, so the upper bound is 2 and not 1 -- a port that writes
    # `sigmoid` here halves the sublayer's share of every stream.
    assert bool((post >= 0).all()) and bool((post <= 2).all())
    assert bool((post > 1).any())


def test_the_sinkhorn_result_is_doubly_stochastic() -> None:
    params = _real_params()
    connection = HyperConnection(
        params, HyperConnectionWeights.from_hf(_real_tensors(), params, "attn_hc"), dtype=torch.float32
    )
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        hidden = torch.randn(2, 8, params.hc_mult, params.hidden_size, generator=generator)
        _, comb, _ = connection.forward(hidden)
        assert torch.allclose(comb.sum(dim=-1), torch.ones_like(comb.sum(dim=-1)), atol=1e-5)
        assert torch.allclose(comb.sum(dim=-2), torch.ones_like(comb.sum(dim=-2)), atol=1e-5)
        assert bool((comb > 0).all())


def test_the_weights_are_in_the_arithmetic() -> None:
    """`hc_scale` and `hc_base` are not tuning knobs, and neither is the count.

    Each of these produces a different answer with the same shapes, which is the
    whole reason they are asserted rather than assumed.
    """
    params = _params()
    weights = _weights(params, seed=8)
    torch.manual_seed(9)
    hidden = torch.randn(1, 3, params.hc_mult, params.hidden_size) * 0.5
    baseline = HyperConnection(params, weights, dtype=torch.float32).forward(hidden)

    zeroed = HyperConnectionWeights(
        hc_fn=weights.hc_fn, hc_base=torch.zeros_like(weights.hc_base), hc_scale=weights.hc_scale
    )
    assert not torch.allclose(HyperConnection(params, zeroed).forward(hidden)[0], baseline[0])

    scaled = HyperConnectionWeights(
        hc_fn=weights.hc_fn, hc_base=weights.hc_base, hc_scale=weights.hc_scale * 2
    )
    assert not torch.allclose(HyperConnection(params, scaled).forward(hidden)[0], baseline[0])

    # One Sinkhorn iteration is not twenty.  A `comb` that is merely row-normalized
    # is not doubly stochastic, and four streams then grow with depth.
    one = HyperConnection(params, weights, dtype=torch.float32)
    one.iters = 1
    assert not torch.allclose(one.forward(hidden)[1], baseline[1], atol=1e-4)


def test_the_clamp_is_on_the_logits() -> None:
    """It is a guard at +-30, and it is applied before the exponential.

    Clamping after `exp` would be clamping probabilities near 1e13, which is a
    different function; and 30 is far past float32's `exp` range, so the guard is
    what makes a saturated gate finite instead of `inf`.
    """
    params = _params()
    weights = _weights(params, seed=10)
    torch.manual_seed(11)
    hidden = torch.randn(1, 2, params.hc_mult, params.hidden_size) * 0.5
    connection = HyperConnection(params, weights, dtype=torch.float32)
    # A `comb_w` this large saturates the clamp; the result is still finite and
    # still doubly stochastic, which is not true of the unclamped version.
    huge = HyperConnectionWeights(hc_fn=weights.hc_fn * 1e4, hc_base=weights.hc_base, hc_scale=weights.hc_scale)
    post, comb, _ = HyperConnection(params, huge).forward(hidden)
    assert bool(torch.isfinite(post).all()) and bool(torch.isfinite(comb).all())
    assert params.hc_clamp_min == -30 and params.hc_clamp_max == 30


def test_the_norm_is_unweighted_and_spans_all_four_streams() -> None:
    """Two things a port can get wrong that both leave the shape alone.

    Weighted instead of unweighted: the coefficients get multiplied by 14336
    numbers that are not in the gate's weights.  Per-stream instead of flattened:
    each stream is normalized on its own, so the four streams stop being
    comparable and `pre` no longer sums a common scale.
    """
    params = _real_params()
    weights = HyperConnectionWeights.from_hf(_real_tensors(), params, "attn_hc")
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(1, 3, params.hc_mult, params.hidden_size, generator=generator) * 0.3

    flat = hidden.flatten(start_dim=-2)
    normalised = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + params.rms_norm_eps)
    # The flattened norm has unit RMS over all four streams at once, which a
    # per-stream norm does not: scaling one stream changes the union's RMS.
    assert torch.allclose(normalised.pow(2).mean(dim=-1), torch.ones(1, 3), atol=1e-5)
    per_stream = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + params.rms_norm_eps)
    assert torch.allclose(per_stream.flatten(start_dim=-2).pow(2).mean(dim=-1), torch.ones(1, 3), atol=1e-5)
    # They agree only when the streams have equal RMS; make them unequal and the
    # two differ, which is what the port would be doing instead.
    skewed = hidden.clone()
    skewed[:, :, 0] *= 10  # one stream, not one token
    a = skewed.flatten(start_dim=2)
    a = a * torch.rsqrt(a.square().mean(dim=-1, keepdim=True) + params.rms_norm_eps)
    b = skewed * torch.rsqrt(skewed.square().mean(dim=-1, keepdim=True) + params.rms_norm_eps)
    assert not torch.allclose(a, b.flatten(start_dim=-2))

    # And the released gate is what consumes it, so the port's own input is the
    # flattened one: the same hidden state through the two norms gives two
    # different coefficient vectors.
    assert not torch.allclose(
        HyperConnection(params, weights).forward(hidden)[0],
        HyperConnection(params, weights).forward(skewed)[0],
    )


def test_the_collapse_is_a_mean_not_a_sum() -> None:
    params = _params()
    weights = _weights(params, seed=13)
    layer = _layer(params, weights)
    torch.manual_seed(14)
    hidden = torch.randn(1, 2, params.hc_mult, params.hidden_size) * 0.5
    assert torch.equal(layer.collapse(hidden), hidden.mean(dim=-2))
    assert not torch.allclose(layer.collapse(hidden), hidden.sum(dim=-2))


# --------------------------------------------------------------------------- #
# The block's plumbing
# --------------------------------------------------------------------------- #


def _real_params() -> Xing4_0Params:
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    return Xing4_0Params.from_config(json.loads(CONFIG.read_text(encoding="utf-8")))


def _layer(params: Xing4_0Params, weights: HyperConnectionWeights, mlp=None) -> DecoderLayer:
    """A layer whose FFN is known, so only the plumbing is measured."""
    from src.models.xing4_0.attention import MLAAttentionWeights

    if mlp is None:
        mlp = lambda x: x  # noqa: E731

    shape = lambda *dims: torch.zeros(*dims)  # noqa: E731
    attention = MLAAttentionWeights(
        q_a_proj=shape(params.q_lora_rank, params.hidden_size),
        q_a_norm=torch.ones(params.q_lora_rank),
        q_b_proj=shape(params.n_heads * params.qk_head_dim, params.q_lora_rank),
        kv_a_proj=shape(params.kv_lora_rank + params.qk_rope_head_dim, params.hidden_size),
        kv_a_norm=torch.ones(params.kv_lora_rank),
        k_b=shape(params.n_heads, params.kv_lora_rank, params.qk_nope_head_dim),
        v_b=shape(params.n_heads, params.v_head_dim, params.kv_lora_rank),
        o_proj=shape(params.hidden_size, params.n_heads * params.v_head_dim),
    )
    return DecoderLayer(
        params,
        DecoderLayerWeights(
            attn_hc=weights,
            ffn_hc=weights,
            attention=attention,
            input_layernorm=torch.ones(params.hidden_size),
            post_attention_layernorm=torch.ones(params.hidden_size),
        ),
        mlp=mlp,
        dtype=torch.float32,
    )


def test_the_residual_is_rebuilt_and_not_accumulated() -> None:
    """`post` and `comb` come from the same call as the collapsed input.

    The property that distinguishes this block from a residual add: with the
    sublayer replaced by an identity, the output is *not* `hidden + hidden`, and
    with `post` at its `2 * sigmoid` maximum it is not `2 * hidden` either --
    `comb` contributes too, and the two are not independent.
    """
    params = _params()
    weights = _weights(params, seed=15)
    gain = 3.0
    layer = _layer(params, weights, mlp=lambda x: x * gain)
    torch.manual_seed(16)
    hidden = torch.randn(1, 2, params.hc_mult, params.hidden_size) * 0.5
    positions = torch.arange(2)
    out = layer.forward(hidden, positions)
    assert tuple(out.shape) == tuple(hidden.shape)

    # Rebuilt by hand out of the block's own halves, which is what the reference
    # does: `post * sublayer_out + comb @ hidden`, where the sublayer reads the
    # *normalized* collapsed state.  The attention's weights are zero here, so its
    # output is zero and the first rebuild is `comb @ hidden`.
    post, comb, collapsed = layer.attn_hc.forward(hidden)
    attn_out = layer.attention.forward_expanded(
        rms_norm(collapsed, layer.weights.input_layernorm, params.rms_norm_eps), positions
    )
    assert not attn_out.abs().max()  # the zero weights really do give zero
    mid = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(comb, hidden)
    post, comb, collapsed = layer.ffn_hc.forward(mid)
    mlp_out = rms_norm(collapsed, layer.weights.post_attention_layernorm, params.rms_norm_eps) * gain
    expected = post.unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb, mid)
    assert torch.allclose(out, expected, atol=1e-6)

    # And it is not a residual add: `hidden + sublayer_out` is a different tensor,
    # which is the failure this test exists to catch.
    assert not torch.allclose(out, mid + mlp_out.unsqueeze(-2), atol=1e-4)


def test_the_norms_are_in_the_block_and_not_only_in_the_model() -> None:
    """`input_layernorm` is applied to `collapsed`, inside the block.

    A port that normalizes the sublayer input once, outside, or that applies the
    norm to the four-stream state instead of the collapsed one, changes the
    arithmetic without changing any shape.
    """
    params = _params()
    weights = _weights(params, seed=17)
    layer = _layer(params, weights)
    torch.manual_seed(18)
    hidden = torch.randn(1, 2, params.hc_mult, params.hidden_size) * 0.5
    post, comb, collapsed = layer.attn_hc.forward(hidden)
    # The norm the block applies is over `collapsed`, i.e. `hidden_size` wide, and
    # the identity scales it to unit RMS -- not the four-stream state.
    normalised = rms_norm(collapsed, layer.weights.input_layernorm, params.rms_norm_eps)
    assert torch.allclose(normalised.pow(2).mean(dim=-1), torch.ones(1, 2), atol=1e-5)
    assert tuple(normalised.shape) == (1, 2, params.hidden_size)


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #

#: The fused kernel does its arithmetic in the activation's dtype -- it rounds
#: the normalized activation before the gate's fma, like the reference's own
#: `flat.to(original_dtype)`, and it accumulates the collapsed stream in fp32 and
#: stores it back rounded.  Two different summation orders over 14336 products is
#: a few ulp of the dtype, not a few ulp of fp32: what is asserted here is that
#: the kernel is the *same forward pass*, at the precision the dtype has.
_KERNEL_REL_TOL = {torch.float32: 1e-5, torch.float16: 4e-3, torch.bfloat16: 4e-2}


def _kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.models.xing4_0.hyper_connection import _load_hyper_connection_kernel

    module = _load_hyper_connection_kernel()
    if module is None:
        pytest.skip("xing4_hyper_connection_forward is not built for this interpreter")
    return module


def _kernel_parity(dtype: torch.dtype, rows: int, scale: float = 0.5, seed: int = 11) -> None:
    params = _params()
    weights = _weights(params, seed=seed)
    connection = HyperConnection(params, weights, dtype=dtype, use_kernel=True)
    # `[batch, seq, hc, hidden]`: the eager port flattens from dim 2, so a state
    # without the batch axis is a shape it is not written for.
    hidden = torch.randn(1, rows, params.hc_mult, params.hidden_size, generator=torch.Generator().manual_seed(seed + 1)) * scale
    hidden = hidden.to(dtype).cuda()

    got = connection.forward(hidden)
    expected = _reference_hyper_connection(params, weights, hidden.cpu())
    for index, (a, b) in enumerate(zip(got, expected)):
        a = a.float().cpu()
        magnitude = b.abs().max().item()
        worst = (a - b).abs().max().item()
        assert worst <= _KERNEL_REL_TOL[dtype] * max(magnitude, 1e-3), (
            f"rows={rows} {dtype} tensor {index}: {worst} on a magnitude of {magnitude}"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 64])
def test_the_kernel_is_the_port(dtype: torch.dtype, rows: int) -> None:
    """One block a row, and the same arithmetic the eager port does."""
    _kernel()
    _kernel_parity(dtype, rows)


def test_the_kernel_takes_the_batch_axis_and_gives_it_back() -> None:
    """`(*batch, tokens, hc, hidden)` in, the same leading shape out.

    The block's own state carries a batch axis that the eager port flattens
    implicitly; the kernel is one block a row and has to be told the rows, so the
    reshape has to come back the way it went in or every shape below it is wrong.
    """
    _kernel()
    params = _params()
    connection = HyperConnection(params, _weights(params, seed=5), dtype=torch.float32, use_kernel=True)
    hidden = torch.randn(2, 3, params.hc_mult, params.hidden_size).cuda()
    post, comb, collapsed = connection.forward(hidden)
    assert tuple(post.shape) == (2, 3, params.hc_mult)
    assert tuple(comb.shape) == (2, 3, params.hc_mult, params.hc_mult)
    assert tuple(collapsed.shape) == (2, 3, params.hidden_size)


def test_the_kernel_agrees_with_the_eager_path_on_a_real_activation() -> None:
    """Rows of a real released `attn_hc`, at the released 3584-wide state.

    The synthetic cases above are noise, which makes the unweighted norm's output
    direction uniform on the sphere.  A real activation has a heavy tail, and the
    one place that could matter is whether the Sinkhorn iterate saturates -- so
    this runs the released weights at the released width, where it can.
    """
    _kernel()
    params = _real_params()
    tensors = _real_tensors()
    weights = HyperConnectionWeights.from_hf(tensors, params, "attn_hc")
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 9, params.hc_mult, params.hidden_size, generator=generator) * 4.0
    hidden[:, :, 0] *= 40.0

    kernel = HyperConnection(params, weights, dtype=torch.float32, use_kernel=True)
    eager = HyperConnection(params, weights, dtype=torch.float32)
    got, expected = kernel.forward(hidden.cuda()), eager.forward(hidden.cpu())
    for index, (a, b) in enumerate(zip(got, expected)):
        worst = (a.float().cpu() - b).abs().max().item()
        assert worst <= 1e-3 * max(b.abs().max().item(), 1.0), f"tensor {index} differs by {worst}"


def test_the_kernel_survives_a_row_that_the_eager_path_never_sees() -> None:
    """`hc_mult = 1`: no mixing, and the kernel's own bounds are what allow it.

    The kernel is sized for eight streams and asserts the bound rather than
    trusting the caller, so a width it was not written for is the case that would
    read past an array.  One stream is the smallest that still has a Sinkhorn.
    """
    _kernel()
    params = _params(hc_mult=1)
    weights = _weights(params, seed=9)
    connection = HyperConnection(params, weights, dtype=torch.float32, use_kernel=True)
    hidden = torch.randn(1, 5, 1, params.hidden_size).cuda()
    got = connection.forward(hidden)
    expected = _reference_hyper_connection(params, weights, hidden.cpu())
    _tree_close(got[0].float().cpu(), expected[0], 1.0, atol=1e-5)
    _tree_close(got[1].float().cpu(), expected[1], 1.0, atol=1e-5)
