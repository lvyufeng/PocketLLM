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
from src.models.deepseek_v4_1 import modules as modules_module
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.decode_pos import Pos
from src.models.deepseek_v4_1.modules import (
    Backbone,
    Block,
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

# The smallest geometry `Block` accepts. The attention fields mirror `TOY` in
# `test_models_deepseek_v4_1_attention.py` and the MoE fields mirror `MINI` in
# `test_models_deepseek_v4_1_loader.py`, because those two files already pin what each half needs to
# be valid and a third spelling would be a third thing to keep in step. Only the Hyper-Connections
# fields are this file's own, and `hc_sinkhorn_iters`/`hc_eps` are stated rather than defaulted:
# `V41TextConfig.validate` rejects a config that leaves them out, and `Block` reads them.
_BLOCK_FIELDS = dict(
    dim=DIM,
    norm_eps=1e-6,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    moe_inter_dim=INTER_DIM,
    n_routed_experts=N_EXPERTS,
    n_shared_experts=1,
    n_activated_experts=TOPK,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=4,
    head_dim=32,
    rope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    window_size=4,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 5),
    index_n_heads=2,
    index_head_dim=32,
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=16,
    candidate_block_size=2,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=512,
    max_position_embeddings=64,
)

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
    # The activation's width is the tree's, not this test's. `expert_forward` multiplies `x` by a
    # weight it is handed and has no way to reconcile the two, so an `x` written at some other width
    # does not exercise the MoE -- it dies inside `F.linear` naming two c10 dtypes and no cause. That
    # was this line until the dense width moved to fp16, which is why it reads the constant rather
    # than spelling bf16: the width is a property the module owns and this test is measuring the
    # module.
    dtype = modules_module.LINEAR_DTYPE
    x = torch.randn(2, 3, DIM, dtype=dtype)

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
    expected = (routed + moe.shared_experts(flat)).to(dtype).view(2, 3, DIM)
    assert torch.allclose(moe(x), expected, atol=1e-6)

    # the shared expert is unconditional: with the routing weights zeroed it is all that is left
    moe.gate.route_scale = 0.0
    shared = moe.shared_experts(flat).type_as(flat).view(2, 3, DIM)
    assert torch.allclose(moe(x), shared, atol=1e-6)


def test_the_token_tile_leaves_the_hyper_connections_unchanged(monkeypatch) -> None:
    """`HC_TOKEN_TILE` walks the token axis of the three Hyper-Connections methods. Does it move them?

    The tile is a memory device and nothing else: `hc_mixes` flattens to fp32 `[b,s,hc*d]` and
    `hc_pre`/`hc_post` build fp32 products of the `hc`-shaped stream, and on a card whose 256K caches
    leave under 2 GiB those allocations are the prefill's wall. Every one of the three reduces over
    `dim` or over the `hc_mult` copies and never over the token axis, so a tile partitions independent
    work and the answer must be the arithmetic the untiled pass states. This pins that.

    A tile is exercised at a width *smaller than the sequence* rather than at the default 1024, so
    that this file's small geometry crosses several boundaries; a tile wider than the sequence takes
    the fall-through and is asserted to be the untiled pass exactly.

    `hc_pre` and `hc_post` are exact at every width, because the reduction they do is over `hc` and
    the tile never splits one. `hc_mixes` is the one that only *stays* exact, and the reason is the
    Sinkhorn normalization rather than the tiling: `hc_split_sinkhorn` assigns rows to buckets with a
    `topk`, which is a discrete pick, so a narrower call can break an exact tie differently and change
    which rows share a bucket -- after which the fp32 that follows is not the same sum. At this
    geometry that never happens and all three come out `torch.equal`. At the released geometry it does:
    8192 tokens, `dim` 5120, `hc_mult` 4, a tile of 1024, and the three coefficients move by
    5.0e-06 / 5.4e-06 / 1.2e-05. So the coefficients are bounded at `1e-4` -- two orders below the
    `O(0.1)` a tiling bug would move them, an order above the measured rounding -- and only `hc_pre`
    and `hc_post` are asserted equal. `test_models_deepseek_v4_1_kernels.py` pins the same discreteness
    for the op itself.
    """
    block = _fill(Block(_cfg(**_BLOCK_FIELDS), 0, 1, 16))
    torch.manual_seed(0)
    tokens, tile = 10, 3
    x = torch.randn(1, tokens, block.hc_mult, DIM, dtype=torch.bfloat16)
    residual = torch.randn(1, tokens, block.hc_mult, DIM, dtype=torch.bfloat16)
    sub = torch.randn(1, tokens, DIM, dtype=torch.bfloat16)
    pre_mix = torch.randn(1, tokens, block.hc_mult).softmax(-1)
    post = torch.randn(1, tokens, block.hc_mult).softmax(-1)
    comb = torch.randn(1, tokens, block.hc_mult, block.hc_mult).softmax(-1)
    gates = (block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)

    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", 0)
    whole = (block.hc_mixes(x, *gates), block.hc_pre(x, pre_mix), block.hc_post(sub, residual, post, comb))
    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", tile)
    tiled = (block.hc_mixes(x, *gates), block.hc_pre(x, pre_mix), block.hc_post(sub, residual, post, comb))

    assert torch.equal(tiled[1], whole[1]), "hc_pre is a sum over hc, which the tile does not touch"
    assert torch.equal(tiled[2], whole[2]), "hc_post is a broadcast over hc, which the tile does not touch"
    assert tiled[0][0].shape == whole[0][0].shape == pre_mix.shape
    for got, want, name in zip(tiled[0], whole[0], ("pre", "post", "comb")):
        assert (got - want).abs().max() <= 1e-4, f"hc_mixes {name} moved more than a fp32 rounding"

    # A tile wider than the sequence must be the untiled pass and nothing else.
    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", tokens + 1)
    wider = (block.hc_mixes(x, *gates), block.hc_pre(x, pre_mix), block.hc_post(sub, residual, post, comb))
    assert torch.equal(wider[1], whole[1]) and torch.equal(wider[2], whole[2])
    assert all(torch.equal(a, b) for a, b in zip(wider[0], whole[0]))


def test_the_hyper_connections_stream_is_carried_wider_than_fp16s_ceiling() -> None:
    """The one constraint on the stream's width, stated as the range the released model reaches.

    `LINEAR_DTYPE` is a single width for the dense stack, and on this card fp16 is the faster of the
    two: sm_75 has no bf16 tensor core, so the same GEMM shapes run at 47-58 TFLOP/s in fp16 against
    6.4-7.5 on the fp32 SIMT path a bf16 operand lands on. The width is still not free, because the
    Hyper-Connections stream is *carried* at it rather than only multiplied at it, and fp16's largest
    finite is 65504.

    The magnitudes below are the released model's, read from a forward over the prompt the chat
    renderer builds, `[0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822]`: a site-by-site trace
    of that forward at bf16 peaks at **2.038e+06** on the stream `hc_post` writes and **4.567e+05** on
    the collapsed copy a layer's norm reads, both by layer 30 of 40. At fp16 the same forward has 415
    non-finite sites of 956 and every logit is NaN -- eight begin-of-sentence tokens out of a model
    that answers `The capital of France is **Paris**.` at bf16. The same sequence *without* its
    leading BOS stays under 200 at every site, which is where the fp16 headroom measurement came from
    and why it read as safe: it metered a chunk of prose, and prose is the no-BOS case.

    So this pins the invariant rather than the two constants: a stream at the magnitude the model
    actually produces must survive the collapse and the expansion. Drawn at `LINEAR_DTYPE`, it fails
    at fp16 on `hc_post`'s expansion -- `tensor([inf, -inf, nan, ...], dtype=torch.float16)`, the
    collapse narrowing to a finite `4.57e+05` and the returned expansion not -- and passes at bf16,
    whatever either constant is set to."""

    block = _fill(Block(_cfg(**_BLOCK_FIELDS), 0, 1, 16))
    x = torch.randn(1, 4, block.hc_mult, DIM, dtype=modules_module.LINEAR_DTYPE)
    residual = (x / x.abs().amax() * 2.038e6).to(modules_module.LINEAR_DTYPE)
    assert float(residual.abs().max()) > 65504, "the stream must start past fp16's ceiling"
    sub = torch.randn(1, 4, DIM, dtype=modules_module.LINEAR_DTYPE)
    pre_mix = torch.randn(1, 4, block.hc_mult).softmax(-1)
    post = torch.randn(1, 4, block.hc_mult).softmax(-1)
    comb = torch.randn(1, 4, block.hc_mult, block.hc_mult).softmax(-1)

    expanded = block.hc_post(sub, residual, post, comb)
    collapsed = block.hc_pre(residual, pre_mix)
    assert expanded.dtype == collapsed.dtype == modules_module.LINEAR_DTYPE
    assert torch.isfinite(expanded).all(), "hc_post's stream is the widest tensor in the block"
    assert torch.isfinite(collapsed).all(), "a collapse of finite copies is bounded by their max"


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


def test_the_token_tile_leaves_the_engram_unchanged(monkeypatch) -> None:
    """The Engram takes the same tile as the Hyper-Connections arithmetic, one module later.

    `Engram.forward` is fp32 `[b,s,hc_mult,dim]` twice over -- `x` and the `key` half of the
    projection -- which is sixteen times the hidden width a token, and the 256K prefill OOMed on
    `key.float()` here once the block's own Hyper-Connections walls were tiled. Its steps are a gather
    by hash id, a reduce over `dim` and a reduce over the `hc_mult` copies, so the token axis is
    again independent work and the tiling is exact rather than approximate.

    This is asserted with `torch.equal` and not a tolerance, and the reason is that the *mask* is
    sliced per tile: a tile that took the wrong span of it would move a position's gate by a large
    amount at one boundary and be invisible at a tile width that happened to land on it. So the mask
    is a block of contiguous False rows in the middle, which every tile boundary below crosses.
    """
    layout = EngramLayout.from_config({**_cfg().__dict__, **_ENGRAM_FIELDS})
    torch.manual_seed(0)
    table = ResidentEngramTable(
        weight=torch.randn(ENGRAM_ROWS, ENGRAM_HEAD_DIM).to(torch.float8_e4m3fn),
        scale=torch.ones(ENGRAM_ROWS, 1).to(torch.float8_e8m0fnu),
    )
    engram = _fill(Engram(_cfg(**_ENGRAM_FIELDS), 0, layout, table))
    tokens = 10
    x = torch.randn(1, tokens, 2, DIM, dtype=torch.bfloat16)
    ids = torch.randint(0, ENGRAM_ROWS, (1, tokens, layout.n_hash_columns))
    mask = torch.ones(1, tokens, dtype=torch.bool)
    mask[:, 3:7] = False

    monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", 0)
    whole = (engram(x, ids), engram(x, ids, mask))
    # A tile narrower than the sequence, so several boundaries fall inside the masked block, and one
    # wider, which is the fall-through and has to be the same pass.
    for tile in (3, 4, tokens + 1):
        monkeypatch.setattr(modules_module, "HC_TOKEN_TILE", tile)
        got = (engram(x, ids), engram(x, ids, mask))
        assert torch.equal(got[0], whole[0]), f"tile {tile} moved the ungated stream"
        assert torch.equal(got[1], whole[1]), f"tile {tile} moved the gated stream"


def test_temperature_zero_samples_the_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 16)
    assert torch.equal(sample(logits, 0.0), logits.argmax(dim=-1))
    # and a positive temperature is stochastic, so it is not that
    draws = {tuple(sample(logits, 1.0).tolist()) for _ in range(32)}
    assert len(draws) > 1


# -- the chunk loop hands each chunk its own position, in the form it was given -------------------


class _PositionRecorder:
    """A `Block` stand-in that keeps the `start_pos` it was handed and passes the stream through.

    `Backbone.forward` asks a block for `(h, pre_mix)` and nothing else, so three lines stand in for
    a layer here. What this section reads is one argument of the call, and building forty real blocks
    to read it would be a different test.
    """

    engram = None

    def __init__(self) -> None:
        self.seen: list = []

    def __call__(self, h, start_pos, pre_mix, image_mask, shared):
        self.seen.append(start_pos)
        return h, pre_mix

    def hc_pre(self, h, pre_mix):
        return h


def _stub_backbone(layers: int = 2) -> tuple[Backbone, list[_PositionRecorder]]:
    """A `Backbone` with `__init__` bypassed: the chunk loop is the subject, not the tree."""
    held = [_PositionRecorder() for _ in range(layers)]
    model = object.__new__(Backbone)
    model.__dict__.update(
        device=None,
        hc_mult=2,
        temperature=0.0,
        target_layer_ids=(),
        layers=held,
        embed=lambda ids: torch.zeros(*ids.shape, DIM),
        norm=lambda h: h,
        head=lambda h: h.reshape(h.size(0), -1),
    )
    return model, held


def test_a_chunked_forward_gives_each_chunk_the_position_its_first_token_sits_at() -> None:
    """Six tokens two at a time from position 10: the layers see 10, 12, 14 -- not 10 three times."""
    model, held = _stub_backbone()
    model.forward(torch.zeros(1, 6, dtype=torch.long), 10, None, None, 2)
    assert held[0].seen == [10, 12, 14]
    assert held[1].seen == [10, 12, 14], "the layers of one chunk disagree about their position"


def test_an_unchunked_forward_hands_the_layer_the_position_it_was_given() -> None:
    model, held = _stub_backbone(layers=1)
    model.forward(torch.zeros(1, 4, dtype=torch.long), 7, None, None, None)
    assert held[0].seen == [7]


def test_a_chunked_forward_keeps_the_position_object_a_graph_hands_it() -> None:
    """The form has to survive the offset, and this is the assertion that failed when it did not.

    A decode step replayed from a graph passes a `Pos` -- its position reaches the card as an index
    tensor, which is the whole reason the object exists -- so `start_pos + c0` is a `TypeError` on
    that path and a capture is where it fires: `--decode-graphs` died in `capture_pass` on
    `unsupported operand type(s) for +: 'Pos' and 'int'`, after the prefill and past every earlier
    check. `int(start_pos) + c0` is the other way to make it stop crashing and it is worse: an `int`
    is a Python value a capture records as a constant, so the replay would decode at the position it
    was recorded at.

    The tensor here is on the host, not on a card: `Pos.__add__` adds to the host counter and rebuilds
    the index from it, so nothing it does is device arithmetic, and
    `tests/test_models_deepseek_v4_1_decode_pos.py` pins the CUDA form.
    """
    model, held = _stub_backbone(layers=1)
    pos = Pos.device(10, torch.device("cpu"))
    model.forward(torch.zeros(1, 6, dtype=torch.long), pos, None, None, 2)
    seen = held[0].seen
    assert all(isinstance(at, Pos) for at in seen), "the offset turned the position into an index"
    assert [at.host for at in seen] == [10, 12, 14]
    assert [at.row().item() for at in seen] == [10, 12, 14]
