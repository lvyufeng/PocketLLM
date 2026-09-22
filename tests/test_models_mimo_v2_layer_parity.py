"""Stage-1 CPU parity for the MiMo-V2 text decoder.

Compares `src/models/mimo_v2/layers.py` against the golden fixture captured from
the checkpoint's own remote code (`/mnt/data1/mimo_v2_oracle/out/golden_tiny.pt`).
The golden carries the fixture's parameters, so the reference implementation is
not imported here at all: weights in, the reference's own intermediates out.

The fixture is 4 layers of hidden 64 -- small enough to run in a second, and built
so that every semantic under test is *active*: two attention families with
different head counts and RoPE bases, a sink on one family only, V narrower than
QK, a sliding window shorter than the sequence, a dense FFN on layer 0 and routed
experts on the rest, with the top-k selection genuinely flipped by the router's
correction bias.

What this pins is shape and semantic correctness at fixture scale. It says nothing
about the quantized layouts (MXFP4 experts, FP8 E4M3 linears), about bf16 storage,
or about numeric agreement on the real 309B checkpoint -- `test_models_mimo_v2_config.py`
covers the shape contract, and the quantized paths are their own stages.

Point `POCKETLLM_MIMO_ORACLE` at the harness directory to run against a different
copy; the tests skip when the golden is not there, so a checkout without it stays
green rather than silently passing on a stub.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.config import MimoV2TextConfig  # noqa: E402
from src.models.mimo_v2.layers import (  # noqa: E402
    MimoV2HostModel,
    build_attention_masks,
    build_rope_inv_freq,
    gate_and_route,
    rms_norm,
    rope_dim,
)

ORACLE_DIR = os.environ.get("POCKETLLM_MIMO_ORACLE", "/mnt/data1/mimo_v2_oracle")
GOLDEN = os.path.join(ORACLE_DIR, "out", "golden_tiny.pt")

#: The reference runs in fp32 at this size and the port reproduces it step for
#: step, so the only slack needed is reassociation in the attention matmuls.
TOL = 1e-6

pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN),
    reason=f"MiMo-V2 oracle golden not present at {GOLDEN}",
)


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def config(golden):
    """The port's config, built from the same kwargs the reference was built from."""
    return MimoV2TextConfig.from_dict(golden["config_kwargs"])


@pytest.fixture(scope="module")
def model(config, golden):
    return MimoV2HostModel(config, golden["params"])


@pytest.fixture(scope="module")
def run(model, golden):
    """One forward, plus the per-expert outputs the fixtures' hooks recorded."""
    capture: dict[int, dict[int, torch.Tensor]] = {}
    with torch.no_grad():
        logits = model(golden["input_ids"], capture_experts=capture)
    return {"logits": logits, "experts": capture}


def delta(actual, expected) -> float:
    return float((actual - expected).abs().max())


def normed_input(layer, layer_in):
    """The stream the attention block actually sees: `qkv_proj` is fed the normed one."""
    return rms_norm(layer_in, layer.weights.input_layernorm, layer.config.layernorm_epsilon)


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_is_the_hybrid_it_claims(config):
    """If the fixture is not the shape it advertises, every check below is vacuous."""
    assert config.total_layers == 4
    assert config.swa_layer_indices == (1, 3)
    assert config.global_layer_indices == (0, 2)
    assert config.dense_layer_indices == (0,)
    assert config.moe_layer_indices == (1, 2, 3)

    ga, swa = config.attention(0), config.attention(1)
    assert (ga.family, swa.family) == ("ga", "swa")
    assert ga.num_q_heads == swa.num_q_heads == 4
    assert (ga.num_kv_heads, swa.num_kv_heads) == (1, 2)
    assert (ga.head_dim, swa.head_dim) == (16, 16)
    assert (ga.v_head_dim, swa.v_head_dim) == (8, 8)
    assert (ga.qkv_out, swa.qkv_out) == (88, 112)
    assert ga.o_in == swa.o_in == 32
    assert (ga.has_sink, swa.has_sink) == (False, True)
    assert (ga.sliding_window, swa.sliding_window) == (None, 4)
    assert ga.rope_theta == 1e7 and swa.rope_theta == 1e4
    assert ga.rope_dim == swa.rope_dim == 8
    assert ga.value_scale == swa.value_scale == pytest.approx(0.707)


def test_every_fixture_parameter_is_claimed(config, golden):
    """`MimoV2HostModel` refuses an unused tensor; assert that it is refusing nothing."""
    MimoV2HostModel(config, golden["params"], strict=True)


# ---------------------------------------------------------------------------
# Boundaries, outermost first
# ---------------------------------------------------------------------------


def test_logits_match_the_reference(run, golden):
    assert delta(run["logits"], golden["logits"]) <= TOL


def test_every_layer_output_matches(run, model, golden):
    """Each layer's output, fed the reference's own input for that layer.

    Running layer by layer from the captured input isolates a single layer's error
    from the drift accumulated above it, which a single logits comparison cannot.
    """
    for idx in sorted(golden["layer_in"]):
        layer = model.layers[idx]
        with torch.no_grad():
            out = layer(
                golden["layer_in"][idx],
                attention_mask=golden["layer_mask"][idx],
                position_embeddings=golden["layer_rope"][idx],
            )
        assert delta(out, golden["layer_out"][idx]) <= TOL, f"layer {idx}"


def test_attention_boundaries_match(run, model, golden):
    """The qkv projection, the o_proj input and the o_proj output."""
    for idx in sorted(golden["layer_in"]):
        layer = model.layers[idx]
        with torch.no_grad():
            detail = layer.attend(
                normed_input(layer, golden["layer_in"][idx]),
                attention_mask=golden["layer_mask"][idx],
                position_embeddings=golden["layer_rope"][idx],
            )
        assert delta(detail["qkv_in"], golden["qkv_in"][idx]) <= TOL, f"layer {idx} qkv input"
        assert delta(detail["qkv_raw"], golden["qkv_raw"][idx]) <= TOL, f"layer {idx} qkv output"
        assert delta(detail["attn_out_pre_o"], golden["attn_out_pre_o"][idx]) <= TOL, (
            f"layer {idx} o_proj input"
        )
        assert delta(detail["attn_out_post_o"], golden["attn_out_post_o"][idx]) <= TOL, (
            f"layer {idx} o_proj output"
        )


def test_the_router_matches_including_the_ties_it_breaks(run, model, golden):
    """Indices and weights, not just the mixed output.

    The indices are compared exactly: the fixture's correction bias flips the
    selection on a number of token-layer pairs, so an implementation that routes on
    the raw scores lands on a different expert set and cannot pass by accident.
    """
    flipped = 0
    for idx in sorted(golden["gate"]):
        expected = golden["gate"][idx]
        layer = model.layers[idx]
        with torch.no_grad():
            topk_idx, topk_weight, logits, scores, choice = layer.route(expected["input"])

        assert delta(logits, expected["logits"]) <= TOL, f"layer {idx} router logits"
        assert delta(scores, expected["scores"]) <= TOL, f"layer {idx} scores"
        assert delta(choice, expected["scores_for_choice"]) <= TOL, f"layer {idx} choice scores"
        assert torch.equal(topk_idx, expected["topk_idx"]), f"layer {idx} expert selection"
        assert delta(topk_weight, expected["topk_weight"]) <= TOL, f"layer {idx} weights"

        # How much work the correction bias is actually doing in this fixture.
        raw = scores.topk(expected["topk_idx"].shape[-1], dim=-1)[1]
        flipped += int((raw != expected["topk_idx"]).any(dim=-1).sum())
    assert flipped > 0, "the fixture no longer exercises noaux_tc: raw and corrected agree"


def test_weights_are_a_partition_and_the_scaling_factor_is_applied(run, model, golden):
    """`norm_topk_prob` renormalises, then `routed_scaling_factor` scales."""
    for idx in sorted(golden["gate"]):
        layer = model.layers[idx]
        with torch.no_grad():
            _, topk_weight, _, _, _ = layer.route(golden["gate"][idx]["input"])
        total = topk_weight.sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total) * layer.config.resolved_routed_scaling_factor)


def test_single_expert_outputs_match(run, golden):
    """Every expert's own output, on exactly the tokens routed to it.

    The summed layer output can agree while an individual expert is wrong, or while
    two experts' errors cancel, so the experts are compared one at a time. This is
    also the boundary the MXFP4 port replaces.
    """
    assert run["experts"], "no expert calls captured"
    real_experts = 0
    for layer_idx in sorted(golden["expert_out"]):
        assert layer_idx in run["experts"], f"layer {layer_idx} ran no experts"
        for expert_idx in sorted(golden["expert_out"][layer_idx]):
            assert expert_idx in run["experts"][layer_idx], (
                f"layer {layer_idx} expert {expert_idx} did not run"
            )
            got = run["experts"][layer_idx][expert_idx]
            want = golden["expert_out"][layer_idx][expert_idx]
            assert delta(got, want) <= TOL, f"layer {layer_idx} expert {expert_idx}"
            real_experts += 1
    assert real_experts >= 20, f"the fixture exercised only {real_experts} experts"


def test_the_experts_that_did_not_run_are_the_ones_the_router_dropped(run, model, golden):
    """A layer that ran 6 of 8 experts is evidence the routing is selective."""
    partial = [
        (idx, len(golden["expert_out"][idx]))
        for idx in sorted(golden["expert_out"])
        if len(golden["expert_out"][idx]) < model.config.n_routed_experts
    ]
    assert partial, "every expert ran in every layer; routing is no longer selective here"


# ---------------------------------------------------------------------------
# The pieces a kernel port replaces
# ---------------------------------------------------------------------------


def test_masks_reproduced_match_the_reference(golden, config):
    """Our own mask construction, against the ones the reference's layers received."""
    ours = build_attention_masks(
        golden["config"]["seq"], config.resolved_window, dtype=torch.float32
    )
    for idx, expected in sorted(golden["layer_mask"].items()):
        key = "sliding_window_attention" if config.attention(idx).is_swa else "full_attention"
        assert torch.equal(ours[key], expected), f"layer {idx} mask ({key})"


def test_the_window_counts_the_query_itself(golden):
    """`i - j < window`, not `<=`: the diagonal stays visible."""
    mask = golden["layer_mask"][1][0, 0]
    allowed = mask == 0
    for query in range(allowed.shape[0]):
        visible = [
            key for key in range(allowed.shape[1]) if bool(allowed[query, key])
        ]
        lo = max(0, query - golden["config"]["window"] + 1)
        assert visible == list(range(lo, query + 1)), f"query {query}: {visible}"


def test_rope_tables_match_the_reference_tables(golden, config):
    """Both families, from the port's own frequency construction."""
    for idx, expected in sorted(golden["layer_rope"].items()):
        shape = config.attention(idx)
        inv_freq = build_rope_inv_freq(shape.rope_dim, shape.rope_theta)
        from src.models.mimo_v2.layers import build_rope_cos_sin

        cos, sin = build_rope_cos_sin(
            inv_freq, torch.arange(golden["config"]["seq"]).unsqueeze(0)
        )
        assert cos.shape == expected[0].shape
        assert delta(cos, expected[0]) <= TOL, f"layer {idx} cos"
        assert delta(sin, expected[1]) <= TOL, f"layer {idx} sin"


def test_the_two_families_use_different_tables(golden):
    """A single shared table would make the dual-RoPE checks vacuous."""
    ga_cos = golden["layer_rope"][0][0]
    swa_cos = golden["layer_rope"][1][0]
    assert not torch.allclose(ga_cos, swa_cos)


def test_the_tail_past_rope_dim_does_not_rotate(config, golden):
    """Only the leading `rope_dim` of each head carries position; the rest is passed through."""
    from src.models.mimo_v2.layers import apply_partial_rope

    shape = config.attention(0)
    states = torch.randn(1, shape.num_q_heads, 10, shape.head_dim)
    cos, sin = golden["layer_rope"][0]
    rotated = apply_partial_rope(states, cos, sin, shape.rope_dim)
    assert torch.equal(rotated[..., shape.rope_dim :], states[..., shape.rope_dim :])


def test_the_sink_is_a_softmax_column_not_an_additive_bias(model, golden):
    """Dropping the sink column changes the normalisation of every real entry."""
    from src.models.mimo_v2.layers import attention as attention_fn

    layer = model.layers[1]
    shape = layer.shape
    assert shape.has_sink and layer.weights.sink is not None

    detail = layer.attend(
        normed_input(layer, golden["layer_in"][1]),
        attention_mask=golden["layer_mask"][1],
        position_embeddings=golden["layer_rope"][1],
    )
    # The probabilities are over real keys only, so they must still be a proper
    # sub-distribution after the sink column is dropped: mass was absorbed, not
    # added. The fixture's sinks are drawn N(0, 1.5) precisely so they take a large
    # share -- the retained fraction bottoms out near 0.37 here -- which is what
    # makes an additive-bias reading measurably different.
    row_sums = detail["probs"].sum(dim=-1)
    assert (row_sums < 1.0).all(), "the sink absorbed no probability mass"
    assert (row_sums > 0.25).all(), "the sink absorbed implausibly much mass"

    qkv = detail["qkv_raw"]
    hidden = layer.weights
    query, key, value = qkv.split([shape.q_size, shape.k_size, shape.v_size], dim=-1)
    query = query.view(1, -1, shape.num_q_heads, shape.head_dim).transpose(1, 2)
    key = key.view(1, -1, shape.num_kv_heads, shape.head_dim).transpose(1, 2)
    value = value.view(1, -1, shape.num_kv_heads, shape.v_head_dim).transpose(1, 2)

    from src.models.mimo_v2.layers import apply_partial_rope

    cos, sin = golden["layer_rope"][1]
    query = apply_partial_rope(query, cos, sin, shape.rope_dim)
    key = apply_partial_rope(key, cos, sin, shape.rope_dim)
    value = value * shape.value_scale

    no_sink, _ = attention_fn(
        query, key, value, golden["layer_mask"][1], shape.scaling,
        shape.num_key_value_groups, sink=None,
    )
    with_sink, _ = attention_fn(
        query, key, value, golden["layer_mask"][1], shape.scaling,
        shape.num_key_value_groups, sink=hidden.sink,
    )
    assert delta(with_sink, no_sink) > 1e-4, "the sink made no difference"


def test_the_value_scale_is_applied_inside_attention(model, golden):
    """Scaling the o_proj output instead is a different, plausible-looking answer."""
    layer = model.layers[1]
    detail = layer.attend(
        normed_input(layer, golden["layer_in"][1]),
        attention_mask=golden["layer_mask"][1],
        position_embeddings=golden["layer_rope"][1],
    )
    scaled = detail["attn_out_pre_o"]
    unscaled_o = torch.nn.functional.linear(
        scaled / layer.shape.value_scale, layer.weights.o_proj
    )
    assert delta(unscaled_o, detail["attn_out_post_o"]) > 1e-4


def test_rms_norm_casts_before_applying_the_weight():
    """The reference multiplies by `weight` in the input dtype, after the cast back."""
    x = torch.randn(2, 8, dtype=torch.float64)
    weight = torch.full((8,), 2.0, dtype=torch.float64)
    got = rms_norm(x, weight, 1e-6)
    upcast = x.to(torch.float32)
    want = weight * (upcast * torch.rsqrt(upcast.pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        torch.float64
    )
    assert torch.equal(got, want)


def test_an_odd_rotary_dimension_is_rejected():
    """`int(192 * 0.34) = 65`; the reference raises rather than rounding down."""
    with pytest.raises(ValueError, match="must be even"):
        rope_dim(192, 0.34)


def test_the_router_rejects_an_unknown_scoring_function():
    with pytest.raises(NotImplementedError, match="scoring function"):
        gate_and_route(
            torch.randn(1, 2, 4),
            torch.randn(8, 4),
            torch.randn(8),
            top_k=2,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
        )


def test_the_router_refuses_noaux_tc_without_the_bias():
    with pytest.raises(ValueError, match="correction bias"):
        gate_and_route(
            torch.randn(1, 2, 4),
            torch.randn(8, 4),
            None,
            top_k=2,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


def test_each_plausible_wrong_reading_misses_the_golden(model, golden):
    """Compute every alternative and require it to be measurably off the golden.

    The parity tests above are only worth their runtime if the readings they rule
    out actually fail them. Each case here is an implementation someone would
    plausibly write -- so each is computed the wrong way and diffed against the
    same captured boundary the correct one matches.
    """
    import torch.nn.functional as F

    from src.models.mimo_v2.layers import (
        apply_partial_rope,
        attention as attention_fn,
        build_rope_cos_sin,
        route_and_mix,
    )

    layer = model.layers[1]
    shape = layer.shape
    normed = normed_input(layer, golden["layer_in"][1])
    want_pre_o = golden["attn_out_pre_o"][1]

    def qkv_of(states):
        qkv = F.linear(states, layer.weights.qkv_proj)
        q, k, v = qkv.split([shape.q_size, shape.k_size, shape.v_size], dim=-1)
        q = q.view(1, -1, shape.num_q_heads, shape.head_dim).transpose(1, 2)
        k = k.view(1, -1, shape.num_kv_heads, shape.head_dim).transpose(1, 2)
        v = v.view(1, -1, shape.num_kv_heads, shape.v_head_dim).transpose(1, 2)
        return q, k, v

    query, key, value = qkv_of(normed)
    cos, sin = golden["layer_rope"][1]
    rotated_q = apply_partial_rope(query, cos, sin, shape.rope_dim)
    rotated_k = apply_partial_rope(key, cos, sin, shape.rope_dim)
    scaled_v = value * shape.value_scale

    # (1) The sink as an additive logit bias rather than an extra softmax column.
    from src.models.mimo_v2.layers import repeat_kv

    k_rep = repeat_kv(rotated_k, shape.num_key_value_groups)
    v_rep = repeat_kv(scaled_v, shape.num_key_value_groups)
    logits = torch.matmul(rotated_q, k_rep.transpose(2, 3)) * shape.scaling
    logits = logits + golden["layer_mask"][1][:, :, :, : k_rep.shape[-2]]
    additive = logits + layer.weights.sink.reshape(1, -1, 1, 1)
    additive = F.softmax(additive - additive.max(-1, keepdim=True).values, dim=-1)
    additive = torch.matmul(additive, v_rep).transpose(1, 2).reshape(1, -1, shape.o_in)
    assert delta(additive, want_pre_o) > 1e-4, "an additive sink passes as the column form"

    # (2) One RoPE table for both families: the global layer's theta on the SWA layer.
    ga_shape = model.config.attention(0)
    wrong_inv = build_rope_inv_freq(ga_shape.rope_dim, ga_shape.rope_theta)
    wrong_cos, wrong_sin = build_rope_cos_sin(
        wrong_inv, torch.arange(golden["config"]["seq"]).unsqueeze(0)
    )
    wrong, _ = attention_fn(
        apply_partial_rope(query, wrong_cos, wrong_sin, shape.rope_dim),
        apply_partial_rope(key, wrong_cos, wrong_sin, shape.rope_dim),
        scaled_v,
        golden["layer_mask"][1],
        shape.scaling,
        shape.num_key_value_groups,
        layer.weights.sink,
    )
    assert delta(wrong.reshape(1, -1, shape.o_in), want_pre_o) > 1e-4, (
        "a single shared RoPE table passes"
    )

    # (3) Weighting the chosen experts by the corrected scores, and (4) skipping the
    # renormalisation. Both feed the same expert call, so they only differ downstream.
    want_out = golden["layer_out"][1]
    moe_input = rms_norm(
        golden["attn_out_post_o"][1] + golden["layer_in"][1],
        layer.weights.post_attention_layernorm,
        layer.config.layernorm_epsilon,
    )
    idx, weight, _, scores, choice = layer.route(moe_input)
    run_expert = layer.weights.expert_fn(layer.config)

    corrected = F.softmax(choice.gather(1, idx), dim=-1)
    for label, wrong_weight in (
        ("corrected-score weighting", corrected),
        ("unnormalised top-k", scores.gather(1, idx)),
    ):
        mixed = route_and_mix(
            moe_input, idx, wrong_weight, run_expert, model.config.n_routed_experts
        )
        got = golden["layer_in"][1] + golden["attn_out_post_o"][1] + mixed
        assert delta(got, want_out) > 1e-5, f"{label} passes as the reference's arithmetic"

    # And the correct arithmetic does reproduce that boundary, so the threshold
    # above is not just a large number in a no-op.
    good = route_and_mix(moe_input, idx, weight, run_expert, model.config.n_routed_experts)
    assert delta(golden["layer_in"][1] + golden["attn_out_post_o"][1] + good, want_out) <= TOL
