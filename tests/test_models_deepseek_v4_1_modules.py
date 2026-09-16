"""The V4.1 module tree's own arithmetic, checked without the released runtime.

`tests/test_models_deepseek_v4_1_reference_parity.py` is the strong check on `modules.py` -- it runs
the released `inference/model.py` beside this tree and compares logits -- but it skips wherever that
tree is not on disk, and it only ever builds one small geometry. What is here is the complement: the
handful of places where the reference's behaviour is decided by something *outside* the arithmetic,
so that reading both files side by side would not catch a disagreement.

* **A field the config does not state.** `V41TextConfig` documents `None` as "this file does not
  say" rather than "the model does not have one", because one of the two released config files omits
  fields the other carries. A consumer that reads such a field with a bare truthiness test therefore
  reads *absence* as `False`. `norm_topk_prob` is exactly that field: the flat
  `inference/config.json` omits it, the reference's `ModelArgs` defaults it `True`, and the flag
  turns the top-k renormalization on. This file pins the resolution.
* **Which number scales the expert.** The reference's `Gate` adds the correction bias to the scores
  to pick experts and then gathers the weights back out of the *unbiased* scores. A port that
  normalized after adding the bias would still pick the same experts and still sum to
  `route_scale`, so it would survive a shape-and-range check.
* **When the clamp happens.** `Expert.forward` clamps the up branch on both sides and the gate branch
  from above only, in fp32 and before the silu. Doing it in bf16, or after the silu, changes values
  that routinely exceed the limit at `swiglu_limit=10`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.encoding.engram import EngramLayout
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.modules import (
    Engram,
    Gate,
    MoE,
    ResidentEngramTable,
    expert_forward,
    sample,
)

DIM = 32
INTER_DIM = 48
N_EXPERTS = 6
TOPK = 2
ENGRAM_HEAD_DIM = 32
ENGRAM_ROWS = 64

# One Engram layer at id 0, so `Engram.layer_hash_index` is 0 and a test can address the table
# directly. The primes this derives are the released arithmetic over a smaller vocabulary.
_ENGRAM_FIELDS = dict(
    engram_layer_ids=(0,),
    engram_num_embeddings=(ENGRAM_ROWS,),
    engram_max_ngram_size=2,
    engram_vocab_size=32,
    engram_n_heads=2,
    engram_head_dim=ENGRAM_HEAD_DIM,
)


def _fill(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Give every parameter a reproducible value.

    Every module here is built out of `torch.empty`, because every one of them expects a checkpoint
    to arrive afterwards; `checkpoint_weights` in `loader.py` is what normally does the filling. A
    test that skips that step is reading uninitialized memory rather than arithmetic -- the first
    draft of this file read NaNs straight out of a freshly built `Gate` and reported the NaN as
    `norm_topk_prob` being ignored. Drawing in fp32 and narrowing keeps the values inside bf16's
    range, which is the one thing a fill has to get right for a bf16 module.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            parameter.copy_(values.to(parameter.dtype))
    return module


def _cfg(**overrides) -> V41TextConfig:
    values = dict(
        dim=DIM,
        moe_inter_dim=INTER_DIM,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=TOPK,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        hc_mult=2,
        norm_eps=1e-6,
    )
    values.update(overrides)
    return V41TextConfig(**values)


def _gate(**overrides) -> Gate:
    """Two calls to this are two gates with the same weights, which is what the first test needs."""
    return _fill(Gate(_cfg(**overrides), N_EXPERTS, TOPK))


def test_an_unstated_norm_topk_prob_normalizes_like_the_reference_default() -> None:
    """`norm_topk_prob=None` means the config file did not say, and the reference's answer is yes.

    The released pair is exactly this shape: `config.json` carries `text_config.norm_topk_prob`, the
    flat `inference/config.json` does not. So this is not a hypothetical config -- it is the one a
    run off the flat file builds.
    """
    torch.manual_seed(0)
    x = torch.randn(4, DIM, dtype=torch.bfloat16)
    stated = _gate(norm_topk_prob=True)(x)
    unstated = _gate(norm_topk_prob=None)(x)
    off = _gate(norm_topk_prob=False)(x)
    assert torch.equal(stated[0], unstated[0]), "an unstated flag must resolve to the stated default"
    assert torch.equal(stated[1], unstated[1])
    # and the flag is not being ignored: the weights sum to route_scale exactly when normalized
    expected_sum = torch.full((4,), _cfg().route_scale)
    assert torch.allclose(stated[0].sum(dim=-1), expected_sum, atol=1e-6)
    assert not torch.allclose(off[0].sum(dim=-1), expected_sum, atol=1e-3)


def test_the_gate_bias_picks_experts_without_scaling_them() -> None:
    """The bias moves selection; the weights stay the unbiased scores, renormalized."""
    gate = _gate()
    torch.manual_seed(1)
    x = torch.randn(4, DIM, dtype=torch.bfloat16)
    gate.bias.data.normal_(0, 1.0)

    weights, indices = gate(x)

    scores = F.softplus(F.linear(x.float(), gate.weight.float())).sqrt()
    expected_indices = (scores + gate.bias).topk(TOPK, dim=-1)[1]
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * _cfg().route_scale

    assert torch.equal(indices, expected_indices)
    assert torch.allclose(weights, expected_weights, atol=1e-6)
    # the bias does not enter the weights: a token routed to a strongly biased expert keeps that
    # expert's own score, which is what makes this different from normalizing the biased scores
    biased = (scores + gate.bias).gather(1, indices)
    biased = biased / biased.sum(dim=-1, keepdim=True) * _cfg().route_scale
    assert not torch.allclose(weights, biased, atol=1e-3)


def test_the_expert_clamp_lands_before_the_silu_and_only_above_on_the_gate_branch() -> None:
    """`swiglu_limit` is small enough here that both branches cross it on every input."""
    limit = 0.5
    w1 = torch.full((INTER_DIM, DIM), 3.0, dtype=torch.bfloat16)
    w2 = torch.eye(DIM, INTER_DIM, dtype=torch.bfloat16)
    w3 = torch.full((INTER_DIM, DIM), -3.0, dtype=torch.bfloat16)
    x = torch.ones(2, DIM, dtype=torch.bfloat16)

    out = expert_forward(x, w1, w2, w3, limit)

    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    # the up branch crosses the floor and the gate branch crosses the ceiling
    assert up.min() < -limit and gate.max() > limit
    expected = F.linear((F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)).to(torch.bfloat16), w2)
    assert torch.allclose(out, expected, atol=1e-5)

    # an up branch clamped only from above is not the same number, so the floor is load-bearing
    one_sided = F.linear((F.silu(gate.clamp(max=limit)) * up.clamp(max=limit)).to(torch.bfloat16), w2)
    assert not torch.allclose(out, one_sided, atol=1e-3)


def test_the_moe_is_the_routed_half_plus_one_shared_expert() -> None:
    """`MoE.forward` is the top-k experts at their routing weights, plus one expert with no gate.

    The routed half is recomputed here slot by slot rather than by calling `moe.routed`, so what is
    being compared is the composition and not the same call twice. Slots are walked in top-k order,
    which is id order because `topk` sorts, and that is the order `ResidentRoutedExperts` accumulates
    in as well.
    """
    moe = _fill(MoE(_cfg(), 0, N_EXPERTS, TOPK))
    torch.manual_seed(0)
    x = torch.randn(2, 3, DIM, dtype=torch.bfloat16)

    flat = x.view(-1, DIM)
    weights, indices = moe.gate(flat)
    routed = torch.zeros_like(flat, dtype=torch.float32)
    for row in range(flat.size(0)):
        for slot in range(TOPK):
            expert = int(indices[row, slot])
            contribution = expert_forward(
                flat[row : row + 1],
                moe.routed.w1[expert],
                moe.routed.w2[expert],
                moe.routed.w3[expert],
                moe.routed.swiglu_limit,
                weights[row, slot : slot + 1, None],
            )
            routed[row] += contribution.reshape(DIM)
    expected = (routed + moe.shared_experts(flat)).to(torch.bfloat16).view(2, 3, DIM)
    assert torch.allclose(moe(x), expected, atol=1e-6)

    # the shared expert is unconditional: with the routing weights zeroed it is all that is left
    moe.gate.route_scale = 0.0
    shared = moe.shared_experts(flat).type_as(flat).view(2, 3, DIM)
    assert torch.allclose(moe(x), shared, atol=1e-6)


def test_a_masked_engram_position_passes_the_stream_through_untouched() -> None:
    """The mask is what keeps an image token, which takes no part in an n-gram, out of the memory."""
    layout = EngramLayout.from_config({**_cfg().__dict__, **_ENGRAM_FIELDS})
    torch.manual_seed(0)
    table = ResidentEngramTable(
        weight=torch.randn(ENGRAM_ROWS, ENGRAM_HEAD_DIM).to(torch.float8_e4m3fn),
        scale=torch.ones(ENGRAM_ROWS, 1).to(torch.float8_e8m0fnu),
    )
    engram = _fill(Engram(_cfg(**_ENGRAM_FIELDS), 0, layout, table))
    x = torch.randn(1, 4, 2, DIM, dtype=torch.bfloat16)
    ids = torch.randint(0, ENGRAM_ROWS, (1, 4, layout.n_hash_columns))

    unmasked = engram(x, ids)
    assert not torch.allclose(unmasked, x), "the memory has to write something for the mask to matter"

    mask = torch.tensor([[True, False, True, False]])
    masked = engram(x, ids, mask)
    assert torch.equal(masked[0, 1], x[0, 1])
    assert torch.equal(masked[0, 3], x[0, 3])
    assert torch.allclose(masked[0, 0], unmasked[0, 0])


def test_temperature_zero_samples_the_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 16)
    assert torch.equal(sample(logits, 0.0), logits.argmax(dim=-1))
    # and a positive temperature is stochastic, so it is not that
    draws = {tuple(sample(logits, 1.0).tolist()) for _ in range(32)}
    assert len(draws) > 1
