"""The bridge from the released shards into `layers.py`, on the release.

`test_models_mimo_v2_layer_parity.py` pins the arithmetic against the oracle
fixture, and `test_models_mimo_v2_loader.py` pins where the tensors are. Neither
answers the question this file asks: does the reference run the *released*
weights, end to end, through the same code path a kernel port will be diffed
against?

That is the point of the bridge. Without it there is no host-side reference for
the device path at all -- the fixture is 4 layers of hidden 64 and the release is
48 of hidden 4096, and a port needs something in between to be checked against.
So these tests build a real layer (or two), run it, and check the three things
that would make a reference useless if they were wrong:

* the routed experts stay **packed**. A layer's 256 MXFP4 experts are 3.2 GiB
  dense, and the model's are 4.7 TB; a bridge that expands them is not a reference
  anybody can run. The tests assert the packed source is what answers, and that
  asking for an expert outside the loaded set raises instead of returning zeros.
* the sparse path is **deterministic**. Dequantize-on-selection and caching a
  dequantized expert have to give the same numbers, or a port cannot be diffed.
* restricting the expert set is **transparent**. This is the seam a
  pre-computed-routing test uses to turn a 256-expert layer into an 8-expert one,
  and if it changed the output it would be measuring itself.

Cost is real -- this reads tens of megabytes of a 172 GiB checkpoint per test --
so the layers are module-scoped and each test does one forward over four tokens.
Everything skips when the checkpoint is not on this host.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.layers import (  # noqa: E402
    build_attention_masks,
    gate_and_route,
    swiglu_mlp,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.quant import dequant_mxfp4  # noqa: E402
from src.models.mimo_v2.weights import (  # noqa: E402
    MimoV2Mxfp4Experts,
    host_model_from_checkpoint,
    layer_weights_from_checkpoint,
    router_weights,
)

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

pytestmark = pytest.mark.skipif(
    not HAS_RELEASE, reason=f"MiMo-V2.6 checkpoint not present at {RELEASE}"
)

#: Layer 1 of the release is the first routed layer and a sliding-window one, so
#: one build covers the sink, the windowed mask, the router and the MXFP4 experts.
MOE_LAYER = 1
TOKENS = 4
DTYPE = torch.float32


@pytest.fixture(scope="module")
def checkpoint() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


@pytest.fixture(scope="module")
def config(checkpoint):
    return checkpoint.layer


@pytest.fixture(scope="module")
def model(checkpoint):
    """Layers 0 and 1 of the release: one dense, one routed."""
    return host_model_from_checkpoint(checkpoint, dtype=DTYPE, device="cpu", layers=[0, MOE_LAYER])


@pytest.fixture(scope="module")
def inputs(config):
    torch.manual_seed(7)
    ids = torch.randint(0, config.vocab_size, (1, TOKENS))
    mask = build_attention_masks(TOKENS, config.resolved_window, DTYPE)
    return ids, mask


def run(model, layer_idx: int, inputs, **kwargs):
    ids, mask = inputs
    layer = model.layers[layer_idx]
    with torch.no_grad():
        hidden = torch.nn.functional.embedding(ids, model.embed_tokens)
        return layer(
            hidden,
            attention_mask=mask[
                "sliding_window_attention" if layer.shape.is_swa else "full_attention"
            ],
            position_embeddings=layer.rope(torch.arange(ids.shape[1]).unsqueeze(0)),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
            **kwargs,
        )


def test_a_routed_layer_keeps_its_experts_packed(checkpoint, config):
    """256 experts of 12.75 MiB stay 12.75 MiB each, and the router is dense."""
    weights = layer_weights_from_checkpoint(checkpoint, MOE_LAYER, DTYPE)
    assert isinstance(weights.expert_source, MimoV2Mxfp4Experts)
    assert weights.experts == (), "expanding the dense tuple is what this avoids"
    assert weights.gate is not None and weights.correction_bias is not None
    assert tuple(weights.gate.shape) == (config.n_routed_experts, config.hidden_size)
    assert tuple(weights.correction_bias.shape) == (config.n_routed_experts,)
    assert weights.expert_source.n_experts == config.n_routed_experts


def test_the_dense_layer_has_a_full_ffn_and_no_router(checkpoint):
    weights = layer_weights_from_checkpoint(checkpoint, 0, DTYPE)
    assert weights.expert_source is None
    assert weights.gate is None and weights.correction_bias is None
    assert weights.sink is None, "the release puts a sink on the windowed family only"
    width = checkpoint.layer.ffn_intermediate_size(0)
    assert tuple(weights.mlp_gate_proj.shape) == (width, checkpoint.layer.hidden_size)


def test_the_router_reads_the_released_gate(checkpoint, config):
    """The `noaux_tc` split, on the real gate: select on corrected, weight on raw.

    The released bias is a positive offset of about 1.7 to 2.2, which is close to
    uniform -- so the test is not "the bias reorders everything" but "the bias is
    what is being selected on", which is the claim a port has to reproduce.
    """
    gate, bias = router_weights(checkpoint, MOE_LAYER)
    assert tuple(gate.shape) == (config.n_routed_experts, config.hidden_size)
    assert tuple(bias.shape) == (config.n_routed_experts,)
    assert torch.isfinite(gate).all() and torch.isfinite(bias).all()

    torch.manual_seed(7)
    hidden = torch.randn(16, config.hidden_size) * 0.5
    topk_idx, topk_weight, logits, scores, for_choice = gate_and_route(
        hidden,
        gate,
        bias,
        top_k=config.num_experts_per_tok,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.resolved_norm_topk_prob,
        routed_scaling_factor=config.resolved_routed_scaling_factor,
    )
    assert tuple(topk_idx.shape) == (16, config.num_experts_per_tok)
    assert torch.equal(scores, logits.sigmoid())
    assert torch.equal(for_choice, scores + bias)
    # The weights are the *raw* scores of the chosen experts, renormalised, so they
    # sum to one and are strictly below the score of the best expert overall.
    assert torch.allclose(topk_weight.sum(-1), torch.ones(16), atol=1e-5)
    assert float(topk_weight.max()) <= float(scores.max()) + 1e-6
    assert topk_idx.max() < config.n_routed_experts

    uncorrected = scores.topk(config.num_experts_per_tok, dim=-1)[1]
    changed = (torch.sort(topk_idx, dim=-1).values != torch.sort(uncorrected, dim=-1).values)
    assert changed.any(), "the correction bias is not reaching the selection"


def test_dequantizing_an_expert_reproduces_the_loader_and_the_swiglu(checkpoint, config):
    """The bridge is a thin wrapper, and this is what "thin" means."""
    source = MimoV2Mxfp4Experts(checkpoint, MOE_LAYER, DTYPE)
    dense = source.dense_expert(4)
    arrays = checkpoint.expert_arrays(MOE_LAYER, 4)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert torch.equal(
            dense[proj],
            dequant_mxfp4(arrays[(proj, "weight")], arrays[(proj, "weight_scale")], 32, DTYPE),
        )
    tokens = torch.randn(TOKENS, config.hidden_size)
    run_expert = source.expert_fn(config.hidden_act)
    assert torch.equal(
        run_expert(4, tokens),
        swiglu_mlp(
            tokens, dense["gate_proj"], dense["up_proj"], dense["down_proj"], config.hidden_act
        ),
    )


def test_a_released_layer_runs_and_produces_a_finite_hidden_state(model, config, inputs):
    out = run(model, MOE_LAYER, inputs)
    assert tuple(out.shape) == (1, TOKENS, config.hidden_size)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, torch.zeros_like(out))


def test_caching_a_dequantized_expert_does_not_change_the_arithmetic(checkpoint, config):
    """Dequantize-on-selection and keep-the-expert have to be the same numbers.

    A cache is the obvious optimization for a layer whose tokens keep hitting the
    same expert, and it is invisible until it is not: the two sources have to
    agree bit for bit or a port diffed against the cached one is diffed against
    something else.
    """
    plain = MimoV2Mxfp4Experts(checkpoint, MOE_LAYER, DTYPE)
    cached = MimoV2Mxfp4Experts(checkpoint, MOE_LAYER, DTYPE, cache=True)
    for expert in (0, 4, 4, 255):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert torch.equal(plain.dense_expert(expert)[proj], cached.dense_expert(expert)[proj])
    assert set(cached._dense) == {0, 4, 255}, "the second ask for 4 was a cache hit"
    cached.dropped_cache()
    assert cached._dense == {}

    tokens = torch.randn(TOKENS, config.hidden_size)
    assert torch.equal(plain.expert_fn()(4, tokens), cached.expert_fn()(4, tokens))


def test_restricting_the_experts_to_the_routed_set_changes_nothing(model, inputs):
    """The seam a pre-computed-routing test uses to shrink a 256-expert layer.

    A layer whose experts are restricted to exactly the set its router selected
    must produce the same output as the unrestricted one -- otherwise a test that
    restricts is measuring the restriction. An expert outside the set has to raise
    rather than read a zero, because a router that selects one would be a bug.
    """
    from dataclasses import replace

    layer = model.layers[MOE_LAYER]
    captured: dict[int, torch.Tensor] = {}
    full = run(model, MOE_LAYER, inputs, capture_experts=captured)
    routed = sorted(captured)
    assert routed, "the router selected nothing, so this test proves nothing"
    assert len(routed) < layer.weights.expert_source.n_experts, "restricting must restrict"

    source = layer.weights.expert_source
    restricted = MimoV2Mxfp4Experts(source.checkpoint, MOE_LAYER, DTYPE, experts=routed)
    narrowed = replace(layer, weights=replace(layer.weights, expert_source=restricted))
    with torch.no_grad():
        ids, mask = inputs
        hidden = torch.nn.functional.embedding(ids, model.embed_tokens)
        again = narrowed(
            hidden,
            attention_mask=mask["sliding_window_attention"],
            position_embeddings=layer.rope(torch.arange(ids.shape[1]).unsqueeze(0)),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
        )
    assert torch.equal(full, again)

    outside = next(e for e in range(source.n_experts) if e not in captured)
    with pytest.raises(KeyError, match="was not loaded into this source"):
        restricted.dense_expert(outside)
