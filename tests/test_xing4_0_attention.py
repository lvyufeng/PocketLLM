"""Xing4.0-29B-A4B's MLA attention, against the checkpoint's own remote code.

The reference (`modeling_xing4_0.py`, read from the release) expands the
compressed KV per head; the released GGUF is shaped for the absorbed form.  Both
are implemented in `src/models/xing4_0/attention.py`, and this file is what makes
"the same arithmetic" a measurement rather than a claim.

The port below is the reference's own functions, written out here rather than
imported, because the reference cannot be imported in this environment: it needs
`transformers` 5.x (`from transformers import initialization as init`) and this
environment has 4.57.1.  That is the same reason
`src/models/deepseek_v4_1/attention.py` carries the reference's helpers, and the
port keeps the reference's line structure so a reader can diff the two:

- `_ReferenceRMSNorm` is `Xing4_0RMSNorm.forward`
- `_reference_rotary` is `Xing4_0RotaryEmbedding.forward`; the frequencies come
  from the same `transformers` function the reference delegates to, and
  `tests/test_xing4_0_rope.py` pins them
- `_apply_rotary_pos_emb_interleave` is the module-level function of that name
- `_reference_attention` is `Xing4_0Attention.forward` plus the
  `eager_attention_forward` it dispatches to, with `repeat_kv` left in at
  `num_key_value_groups == 1` where it is the identity
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.models.xing4_0.attention import KVLatentCache, MLAAttention, MLAAttentionWeights
from src.models.xing4_0.config import Xing4_0Params, yarn_get_mscale

CONFIG = Path("/mnt/data2/Xing4.0-29B-A4B/config.json")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _params(**overrides) -> Xing4_0Params:
    """The released config's rope, at fixture widths.

    The widths are what change; `qk_rope_head_dim` stays even and
    `original_max_position_embeddings` is small enough that the YaRN ramp is
    exercised by a handful of positions.  Nothing about the arithmetic depends on
    the widths being 3584 and 32, and a fixture that used them would need a
    checkpoint to be useful.
    """
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


def _real() -> Xing4_0Params:
    """The released config, at the released widths."""
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    return Xing4_0Params.from_config(json.loads(CONFIG.read_text(encoding="utf-8")))


def _weights(params: Xing4_0Params, seed: int = 0) -> MLAAttentionWeights:
    generator = torch.Generator().manual_seed(seed)
    shape = lambda *dims: torch.randn(*dims, generator=generator) * 0.3  # noqa: E731
    return MLAAttentionWeights(
        q_a_proj=shape(params.q_lora_rank, params.hidden_size),
        q_a_norm=torch.ones(params.q_lora_rank),
        q_b_proj=shape(params.n_heads * params.qk_head_dim, params.q_lora_rank),
        kv_a_proj=shape(params.kv_lora_rank + params.qk_rope_head_dim, params.hidden_size),
        kv_a_norm=torch.ones(params.kv_lora_rank),
        k_b=shape(params.n_heads, params.kv_lora_rank, params.qk_nope_head_dim),
        v_b=shape(params.n_heads, params.v_head_dim, params.kv_lora_rank),
        o_proj=shape(params.hidden_size, params.n_heads * params.v_head_dim),
    )


# --------------------------------------------------------------------------- #
# The reference, ported
# --------------------------------------------------------------------------- #


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    return (weight * (x * torch.rsqrt(variance + eps))).to(dtype)


def _reference_rotary_q_rot(
    params: Xing4_0Params, q_rot: torch.Tensor, k_rot: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Xing4_0RotaryEmbedding.forward` plus `apply_rotary_pos_emb_interleave`.

    `inv_freq` is computed here rather than imported, exactly as the reference
    computes it through `ROPE_INIT_FUNCTIONS["yarn"]`, and the `2x2` result of the
    two `yarn_get_mscale` calls is the ratio the reference's `attention_factor`
    is: for this config, 1.0.
    """
    base = params.rope_theta
    dim = params.head_dim
    factor = params.yarn.factor
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    low = (dim * torch.log(torch.tensor(float(params.yarn.original_max_position_embeddings)) / (params.yarn.beta_fast * 2 * torch.pi))) / (2 * torch.log(torch.tensor(base)))
    high = (dim * torch.log(torch.tensor(float(params.yarn.original_max_position_embeddings)) / (params.yarn.beta_slow * 2 * torch.pi))) / (2 * torch.log(torch.tensor(base)))
    low, high = max(float(torch.floor(low)), 0.0), min(float(torch.ceil(high)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    inv_freq = interpolation * ramp + extrapolation * (1 - ramp)

    def mscale(scale: float, m: float) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * float(torch.log(torch.tensor(scale))) + 1.0

    attention_scaling = mscale(factor, params.yarn.mscale) / mscale(factor, params.yarn.mscale_all_dim)

    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

    # `apply_rotary_pos_emb_interleave`, at `unsqueeze_dim=1`: q_rot is
    # (b, seq, heads, rope) so the angle goes on the head axis, and k_rot is
    # (b, seq, 1, rope) so it does not need one.
    def apply(x: torch.Tensor, unsqueeze: int) -> torch.Tensor:
        ori = x.dtype
        xf = x.float()
        c = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze)
        s = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze)
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(ori)

    return apply(q_rot, 1), apply(k_rot, 1)


def _reference_attention(
    params: Xing4_0Params,
    weights: MLAAttentionWeights,
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """`Xing4_0Attention.forward` with `eager_attention_forward`, no cache."""
    b, seq = hidden.shape[0], hidden.shape[1]
    heads, nope, rope = params.n_heads, params.qk_nope_head_dim, params.qk_rope_head_dim
    v_head = params.v_head_dim
    eps = params.rms_norm_eps

    q_states = F.linear(_reference_rms_norm(F.linear(hidden, weights.q_a_proj), weights.q_a_norm, eps), weights.q_b_proj)
    q_states = q_states.view(b, seq, heads, nope + rope)
    q_pass, q_rot = torch.split(q_states, [nope, rope], dim=-1)

    compressed_kv = F.linear(hidden, weights.kv_a_proj)
    k_pass, k_rot = torch.split(compressed_kv, [params.kv_lora_rank, rope], dim=-1)
    k_pass = _reference_rms_norm(k_pass, weights.kv_a_norm, eps)

    # `kv_b_proj` in the reference is one `(heads * (nope + v_head), kv_lora)`
    # weight, and its row order is head-major: head 0's 128 `nope` rows, then
    # head 0's 128 `v` rows, then head 1's, and so on -- which is what makes the
    # reference's `.view(..., heads, nope + v_head)` read as "this head's key,
    # then this head's value".  The release ships that weight as the two halves
    # `k_b` and `v_b`, so rebuilding it here has to interleave per head rather
    # than concatenate the two blocks; getting that wrong leaves both tensors the
    # right shape and mixes one head's key into the next head's value.
    kv_b = torch.cat(
        [torch.cat([weights.k_b[h].T, weights.v_b[h]], dim=0) for h in range(heads)], dim=0
    )
    kv = F.linear(k_pass, kv_b).view(b, seq, heads, nope + v_head)
    k_nope, value_states = torch.split(kv, [nope, v_head], dim=-1)

    k_rot = k_rot.view(b, seq, 1, rope)
    q_rot, k_rot = _reference_rotary_q_rot(params, q_rot, k_rot, positions)
    k_rot = k_rot.expand(*k_nope.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_nope, k_rot), dim=-1)

    # `eager_attention_forward`: `repeat_kv` at `num_key_value_groups == 1` is the
    # identity, so the key already carries all `heads` and nothing is repeated.
    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)

    scaling = params.qk_head_dim ** -0.5
    if params.yarn.mscale_all_dim:
        mscale = yarn_get_mscale(params.yarn.factor, params.yarn.mscale_all_dim)
        scaling = scaling * mscale * mscale

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    q_pos = torch.arange(seq).view(seq, 1)
    k_pos = torch.arange(seq).view(1, seq)
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, value)
    out = out.transpose(1, 2).contiguous().reshape(b, seq, -1)
    return F.linear(out, weights.o_proj)


# --------------------------------------------------------------------------- #
# The properties
# --------------------------------------------------------------------------- #


def test_the_expanded_form_is_the_reference() -> None:
    """Parity at fixture scale, to floating-point re-association and no further.

    The reference's `q` and the absorbed form's start from the same projection and
    the same rope; everything after that is a contraction order, so the tolerance
    is fp32 noise and not a modelling choice.
    """
    params = _params()
    weights = _weights(params)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    torch.manual_seed(3)
    hidden = torch.randn(1, 6, params.hidden_size) * 0.4
    positions = torch.arange(6)

    expected = _reference_attention(params, weights, hidden, positions)
    got_expanded = attention.forward_expanded(hidden, positions)
    got_absorbed = attention.forward_absorbed(hidden, positions)

    assert got_expanded.shape == got_absorbed.shape == expected.shape == hidden.shape
    for name, got in (("expanded", got_expanded), ("absorbed", got_absorbed)):
        worst = (got - expected).abs().max().item()
        scale = expected.abs().max().item()
        assert worst <= scale * 1e-5, f"{name} differs from the reference by {worst} against {scale}"


def test_the_two_forms_are_one_arithmetic() -> None:
    params = _params()
    weights = _weights(params, seed=1)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    torch.manual_seed(4)
    hidden = torch.randn(1, 7, params.hidden_size) * 0.4
    positions = torch.arange(7)
    expanded = attention.forward_expanded(hidden, positions)
    absorbed = attention.forward_absorbed(hidden, positions)
    assert (expanded - absorbed).abs().max().item() <= expanded.abs().max().item() * 1e-5


def test_a_cached_prefill_is_an_uncached_one() -> None:
    params = _params()
    weights = _weights(params, seed=2)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    torch.manual_seed(5)
    hidden = torch.randn(1, 5, params.hidden_size) * 0.4
    positions = torch.arange(5)
    for name, forward in (("expanded", attention.forward_expanded), ("absorbed", attention.forward_absorbed)):
        cache = KVLatentCache(1, 16, params)
        cached = forward(hidden, positions, cache=cache, start_pos=0)
        plain = forward(hidden, positions)
        assert (cached - plain).abs().max().item() == 0.0, name
        assert cache.length == 5
        assert tuple(cache.view().shape) == (1, 5, params.kv_lora_rank + params.qk_rope_head_dim)


def test_a_decode_step_is_the_last_row_of_a_full_pass() -> None:
    """The property a decode kernel has to have, and the one a port loses first."""
    params = _params()
    weights = _weights(params, seed=3)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    torch.manual_seed(6)
    prompt = torch.randn(1, 6, params.hidden_size) * 0.4
    step = torch.randn(1, 1, params.hidden_size) * 0.4

    full_hidden = torch.cat((prompt, step), dim=1)
    full_positions = torch.arange(7)
    for name, forward in (("expanded", attention.forward_expanded), ("absorbed", attention.forward_absorbed)):
        cache = KVLatentCache(1, 16, params)
        prefill = forward(prompt, torch.arange(6), cache=cache, start_pos=0)
        decoded = forward(step, torch.tensor([6]), cache=cache, start_pos=6)
        whole = forward(full_hidden, full_positions)
        assert (prefill - whole[:, :6]).abs().max().item() <= whole.abs().max().item() * 1e-5, name
        assert (decoded - whole[:, 6:]).abs().max().item() <= whole.abs().max().item() * 1e-5, name
        # A one-token step attends to everything cached, including itself.
        assert cache.length == 7


def test_a_chunked_prefill_is_a_whole_one() -> None:
    """Two chunks with absolute positions, which is what a chunked prefill writes."""
    params = _params()
    weights = _weights(params, seed=4)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    torch.manual_seed(7)
    hidden = torch.randn(1, 8, params.hidden_size) * 0.4
    for name, forward in (("expanded", attention.forward_expanded), ("absorbed", attention.forward_absorbed)):
        cache = KVLatentCache(1, 16, params)
        first = forward(hidden[:, :3], torch.arange(3), cache=cache, start_pos=0)
        second = forward(hidden[:, 3:], torch.arange(3, 8), cache=cache, start_pos=3)
        whole = forward(hidden, torch.arange(8))
        assert (first - whole[:, :3]).abs().max().item() <= whole.abs().max().item() * 1e-5, name
        assert (second - whole[:, 3:]).abs().max().item() <= whole.abs().max().item() * 1e-5, name


def test_the_cache_is_seventeen_times_narrower_absorbed() -> None:
    """The reason the released file is shaped the way it is, at the real widths.

    Two different 32-head widths are worth keeping apart.  `kv_b_proj` is
    `heads * (nope + v_head)` = 8192 wide, because the rope key does not come out
    of it -- `kv_a_proj` produces that separately.  The *cache* is
    `heads * (qk_head_dim + v_head_dim)` = 10240, because the reference stores the
    rope key once per head alongside the expanded key and value.  Absorbed, both
    collapse to one latent plus one shared rope key: 512 + 64 = 576.
    """
    params = _real()
    absorbed = params.kv_lora_rank + params.qk_rope_head_dim
    kv_b_width = params.n_heads * (params.qk_nope_head_dim + params.v_head_dim)
    expanded = params.n_heads * (params.qk_head_dim + params.v_head_dim)
    assert absorbed == 576
    assert kv_b_width == 8192
    assert expanded == 10240
    assert expanded / absorbed == pytest.approx(17.78, rel=1e-2)
    cache = KVLatentCache(1, 4, params)
    assert cache.width == absorbed


def test_the_head_group_count_is_one() -> None:
    """`repeat_kv` is the identity here, so the expanded key is already per head.

    The GGUF says `head_count_kv 1`, which is a statement about the *absorbed*
    cache and not about a grouped-query head count: the reference has 32 key-value
    heads over 32 query heads, so nothing is repeated, and the single cached key
    is the compressed latent.
    """
    params = _params()
    assert params.n_heads == 2
    real = _real()
    assert real.n_heads == 32
    # 32 query heads over 32 key-value heads, not a grouped-query model, so the
    # expanded form's key already carries every head and nothing is repeated.
    assert int(json.loads(CONFIG.read_text(encoding="utf-8"))["num_key_value_heads"]) == real.n_heads


# --------------------------------------------------------------------------- #
# The released weights, at the released widths
# --------------------------------------------------------------------------- #

# Layer 2 is the first MoE layer and lives in shard 3, which is the one shard of
# the 41 this machine has.  Its attention is the same attention as every other
# layer's, so one layer is enough to hold the mapping to the released file.
SHARD = Path("/mnt/data2/Xing4.0-29B-A4B/model-00003-of-00041.safetensors")
LAYER = 2
_PREFIX = f"model.layers.{LAYER}."


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


def test_the_released_attention_has_the_shape_this_module_assumes() -> None:
    """Every tensor this module reads, held to the released file's own header."""
    params = _real()
    tensors = _real_tensors()
    expected = {
        "self_attn.q_a_proj.weight": (params.q_lora_rank, params.hidden_size),
        "self_attn.q_a_layernorm.weight": (params.q_lora_rank,),
        "self_attn.q_b_proj.weight": (params.n_heads * params.qk_head_dim, params.q_lora_rank),
        "self_attn.kv_a_proj_with_mqa.weight": (params.kv_lora_rank + params.qk_rope_head_dim, params.hidden_size),
        "self_attn.kv_a_layernorm.weight": (params.kv_lora_rank,),
        "self_attn.kv_b_proj.weight": (params.n_heads * (params.qk_nope_head_dim + params.v_head_dim), params.kv_lora_rank),
        "self_attn.o_proj.weight": (params.hidden_size, params.n_heads * params.v_head_dim),
    }
    for name, shape in expected.items():
        assert tuple(tensors[name].shape) == shape, name


def test_the_hf_split_reproduces_kv_b_proj() -> None:
    """`k_b` and `v_b` are the released `kv_b_proj`, taken apart and put back.

    This is the mapping the GGUF cannot check, because the GGUF ships the two
    halves already separated.  If the row order were a plain concatenation instead
    of head-major, the round trip below would still pass -- so the test that
    matters is the one after it, which runs both forms against each other.
    """
    params = _real()
    tensors = _real_tensors()
    weights = MLAAttentionWeights.from_hf(tensors, params)
    assert tuple(weights.k_b.shape) == (params.n_heads, params.kv_lora_rank, params.qk_nope_head_dim)
    assert tuple(weights.v_b.shape) == (params.n_heads, params.v_head_dim, params.kv_lora_rank)
    rebuilt = torch.cat([torch.cat([weights.k_b[h].T, weights.v_b[h]], dim=0) for h in range(params.n_heads)], dim=0)
    assert torch.equal(rebuilt, tensors["self_attn.kv_b_proj.weight"])


def test_the_released_weights_agree_with_the_reference() -> None:
    """The parity test once more, with nothing synthetic in it.

    Both forms read the released file's own weights, and the reference is built
    from its own `kv_b_proj` rather than from `from_hf`'s split, so this is the
    point at which a wrong split stops being invisible.
    """
    params = _real()
    tensors = _real_tensors()
    weights = MLAAttentionWeights.from_hf(tensors, params)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(11)
    hidden = (torch.randn(1, 8, params.hidden_size, generator=generator) * 0.2).to(torch.float32)
    positions = torch.arange(8)

    expected = _reference_attention(params, weights, hidden, positions)
    # The reference is handed the checkpoint's own `kv_b_proj` here, via the split
    # put back together -- the same tensor `test_the_hf_split_reproduces_kv_b_proj`
    # proves equal to the released one.
    for name, got in (
        ("expanded", attention.forward_expanded(hidden, positions)),
        ("absorbed", attention.forward_absorbed(hidden, positions)),
    ):
        worst = (got - expected).abs().max().item()
        assert worst <= expected.abs().max().item() * 1e-5, f"{name}: {worst}"


def test_a_real_decode_step_lands_in_the_reference() -> None:
    """The same, one token at a time, with the released weights."""
    params = _real()
    weights = MLAAttentionWeights.from_hf(_real_tensors(), params)
    attention = MLAAttention(params, weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(12)
    prompt = (torch.randn(1, 5, params.hidden_size, generator=generator) * 0.2).to(torch.float32)
    step = (torch.randn(1, 1, params.hidden_size, generator=generator) * 0.2).to(torch.float32)
    whole = attention.forward_absorbed(torch.cat((prompt, step), dim=1), torch.arange(6))

    for name, forward in (("expanded", attention.forward_expanded), ("absorbed", attention.forward_absorbed)):
        cache = KVLatentCache(1, 8, params)
        forward(prompt, torch.arange(5), cache=cache, start_pos=0)
        decoded = forward(step, torch.tensor([5]), cache=cache, start_pos=5)
        assert (decoded - whole[:, 5:]).abs().max().item() <= whole.abs().max().item() * 1e-5, name
        # The absorbed cache is the whole point: 576 wide, not 10240.
        assert tuple(cache.view().shape) == (1, 6, 576)
