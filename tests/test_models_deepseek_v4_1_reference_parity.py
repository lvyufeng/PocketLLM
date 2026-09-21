"""Parity between this repository's V4.1 backbone and the released `inference/model.py`.

`src/models/deepseek_v4_1/` reimplements the released runtime rather than importing it, for the
reason `kernels.py` gives: the released `inference/kernel.py` needs TileLang and a tensor core this
host does not have. A reimplementation is only worth anything if something holds it to the original,
and prose cannot: the module tree has to reproduce the reference's *order of operations*, and the
places that matters -- the hc mix that one sublayer computes for the next, the clamp that happens
before the silu rather than after, the divide by `gate_temp` that neither released config states --
are exactly the ones a reviewer would not catch by reading both files side by side.

So this file runs both. It imports the released `model.py` with three of its dependencies stubbed,
builds the reference and this repository's `Backbone` from one `ModelArgs`, copies every reference
parameter into the matching slot here, and compares the logits of a prefill and the logits of the
decode step that follows it.

What the stubs mean for the result:

* `kernel` is bound to *this repository's* ops, not the reference's TileLang ones. So what the
  comparison measures is the module tree around attention -- `Block`, the hc mixing, `MoE`, `Gate`,
  `Engram` -- and not the attention ops, which `tests/test_models_deepseek_v4_1_kernels.py` and
  `tests/test_models_deepseek_v4_1_attention.py` cover separately. Running the reference's own
  kernel would compare two op sets at once and report a difference in either as a difference here.
* `vision` and `image_processor` are never reached: `vision_n_layers=0` and no `token_types`, and
  the engram hash is fed the reference's own output rather than a tokenizer's normalizer chain (see
  `_load_reference`), so the compressed token map -- `src/encoding/engram.py`'s subject -- is the
  one piece of the reference's front end this file does not run.

The whole file skips when the released inference tree is not on this host, which is the same
condition `tests/test_models_deepseek_v4_1_tensor_audit.py` uses for the checkpoint.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from src.models.deepseek_v4_1.config import from_dict
from src.models.deepseek_v4_1.modules import Backbone, ResidentEngramTable

REFERENCE = Path("/mnt/data3/DeepSeek-V4.1-Flash/inference")
requires_reference = pytest.mark.skipif(
    not (REFERENCE / "model.py").is_file(), reason="no released V4.1 inference tree on this host"
)

# `ffn.experts.{j}.w{k}.weight` in the reference, where this tree keeps one bank per layer.
_EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")

# The vocab the n-gram hash is computed over here, and the model's vocab along with it. The released
# pair is 129280 tokens over a 16M hash vocab; at this size the identity token map the loader
# installs is already a total map, and the primes the reference derives from it -- four per engram
# layer, so the largest bucket id is `sum(primes[layer]) - 1` -- stay small enough for a test table.
# The arithmetic downstream of the size is unchanged.
_ENGRAM_VOCAB = 256


def _stub(name: str, **attributes) -> None:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _kernel_stub() -> dict:
    """The reference's six kernel entry points, bound to this repository's implementations."""
    from src.kernels import ops
    from src.models.deepseek_v4_1.kernels import fp4_act_quant_e4m3

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        # The reference's `fp4_quant_kernel` branches on the scale dtype and `src/kernels/ops.py`
        # implements one branch, so the dispatch has to be reproduced for the other to be reached.
        if scale_dtype == torch.float8_e4m3fn:
            return fp4_act_quant_e4m3(x, block_size, inplace)
        return ops.fp4_act_quant(x, block_size, inplace)

    return {
        "act_quant": ops.act_quant,
        "fp4_act_quant": fp4_act_quant,
        "fp4_gemm": ops.fp4_gemm,
        "fp8_gemm": ops.fp8_gemm,
        "hc_split_sinkhorn": ops.hc_split_sinkhorn,
        "sparse_attn": ops.sparse_attn,
    }


@lru_cache(maxsize=1)
def _load_reference():
    """The released `inference/model.py` as a module, with the three unimportable stubs installed."""
    _stub("kernel", **_kernel_stub())
    # Only four ints are read from image_processor; `vision` is imported but never instantiated.
    _stub("image_processor", IMAGE=1, IMAGE_START=2, IMAGE_END=3, IMAGE_NEW_LINE=4)
    _stub("vision", ViT=object, Aligner=object)
    sys.path.insert(0, str(REFERENCE))
    try:
        engram = importlib.import_module("engram")
        # `NgramHashState.__init__` runs a `tokenizers` normalizer chain over all 129280 released
        # tokens to build its compressed map, and asserts the result against the config. That map is
        # what `src/encoding/engram.py` reimplements and `tests/test_encoding_engram.py` covers, so
        # here it is the identity: the reference's *hashing* -- the primes, the per-layer bucket
        # offsets, the rolling XOR -- still runs, over raw token ids rather than compressed ones,
        # and both models are handed its output.
        engram.build_compressed_token_map = lambda tokenizer: (list(range(_ENGRAM_VOCAB)), _ENGRAM_VOCAB)
        return importlib.import_module("model")
    finally:
        sys.path.remove(str(REFERENCE))


def _tiny_args(reference, **overrides):
    """A `ModelArgs` small enough to run on CPU in a second, with every mechanism switched on.

    Dims are the reference's own defaults shrunk; the scale-independent values (`norm_eps`,
    `score_func`, `hc_*`, `swiglu_limit`, `route_scale`, the n-gram settings) are the released
    ones, because those are the numbers the arithmetic depends on. Two of the reference's defaults
    are deliberately changed: `dtype` is bf16 and `expert_dtype` is None, which is what makes its
    `Linear` a plain `F.linear` instead of an fp8/fp4 GEMM this host cannot run.
    """
    values = dict(
        max_batch_size=2,
        max_seq_len=64,
        temperature=1.0,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=_ENGRAM_VOCAB,
        dim=64,
        moe_inter_dim=96,
        n_layers=5,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        norm_eps=1e-20,
        o_groups=2,
        o_lora_rank=16,
        window_size=16,
        compress_ratios=(0, 2, 2, 1, 1, 0),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3),
        compress_rope_theta=40000.0,
        original_seq_len=0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        # Also a multiple of the fp4 block size, for the same reason: the indexer quantizes its keys
        # 32 columns at a time. 128 in the checkpoint, 32 here.
        index_head_dim=32,
        index_topk=8,
        candidate_source_layer=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        engram_layer_ids=(1, 3),
        # Rows the two layers' tables declare. The largest bucket id these primes can produce is
        # `sum(primes[layer]) - 1` -- under 1200 for both layers -- so one size fits both tables;
        # `_hash` asserts that rather than trusting it.
        engram_num_embeddings=(2048, 2048),
        engram_max_ngram_size=3,
        engram_vocab_size=_ENGRAM_VOCAB,
        engram_n_heads=2,
        # Must be a multiple of the fp8 block size: the table's scales are one per 32 columns, in
        # the reference and in the checkpoint alike. 256 there, 32 here.
        engram_head_dim=32,
        engram_pad_id=2,
        engram_compressed_vocab_size=_ENGRAM_VOCAB,
        vision_n_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=(1, 3),
    )
    values.update(overrides)
    return reference.ModelArgs(**values)


def _add_vision_gate_bias(ref_model) -> None:
    """Put back the `Gate.bias_vl` this test's config drops.

    The released config enables vision, so every checkpoint has `bias_vl` and this tree always
    builds it. `ModelArgs.vision_enabled` is `vision_n_layers > 0`, and this test sets that to 0
    rather than build a ViT tower, so the reference registers no such parameter at all. Adding it
    back as zeros keeps the two trees name-for-name and cannot move a number: the reference reads
    `bias_vl` only when it is handed an `image_mask`, and this file never hands one.
    """
    for block in ref_model.layers:
        if block.ffn.gate.bias_vl is None:
            block.ffn.gate.bias_vl = torch.nn.Parameter(torch.zeros_like(block.ffn.gate.bias))


def _randomize(model: torch.nn.Module, seed: int) -> None:
    """Fill every tensor a checkpoint would supply with reproducible values.

    Both trees are built out of `torch.empty`, because both expect real weights to arrive, so
    without this the comparison would be of uninitialized memory. That is not a hypothetical: the
    first version of this file did exactly that and "passed" on whichever allocation happened to be
    finite.

    Values are drawn in fp32 and cast down, which is what keeps them inside the narrow formats --
    an E8M0 tensor filled with random *bytes* lands on its NaN exponent about one time in 256, and
    E8M0 has no sign, so only the magnitude is drawn.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                values = torch.randn(tensor.shape, generator=generator, dtype=torch.float32) * 0.2
                if tensor.dtype == torch.float8_e8m0fnu:
                    values = values.abs() + 0.5
                tensor.copy_(values.to(tensor.dtype))
            else:
                tensor.copy_(torch.randint(0, 4, tensor.shape, generator=generator))


def _pin_dense_dtype(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Point this tree's dense stack at `dtype`, returning what it was.

    The reference is bf16 by construction and not by argument. `ModelArgs.dtype` names the *weight*
    storage its `Linear` wraps, and the pieces that are not `Linear` -- `Engram.q_weight`,
    `k_weight`, the indexer's projections, `ParallelHead` -- are written `torch.bfloat16` in
    `model.py` itself. So there is no width to build the reference at other than the one it
    declares, and this tree's dense stack is fp16 in production.

    Two widths cannot be compared exactly, which is what this file does. The weights are not why:
    every weight the reference holds is bf16, and bf16's 8 significand bits fit inside fp16's 11, so
    the copy is exact down to fp16's denormal floor of 2**-14. It is the *activations* -- a GEMM's
    inputs are cast to its weight's width, so a bf16 reference rounds them to 8 bits where this tree
    keeps 11, and over forty layers the two streams separate to about a tenth of the logit scale.
    That is arithmetic and not a defect, and it is why the comparison in this file is run at the
    reference's width.

    The shipping width's own evidence is elsewhere and is not a tie-break this file can make: the
    same harness run at fp16 against the same bf16 reference reads a max|logit diff| of at most
    0.116 of the logit scale with the bf16 run as a zero control, recorded in
    `docs/performance/v41_dense_gemm_dtype.md` and reproduced by `probe_v41_dense_dtype_abab.py` on
    the real checkpoint.

    Three modules and not one: `attention.py` and `modules.py` each hold the name, and `loader.py`
    binds it a third time (`from ...attention import LINEAR_DTYPE`), so a build that moved only the
    first two would leave any reader in the third at the old width. `CACHE_DTYPE` moves with it --
    not as a detail, but because the sparse-attention kernels dispatch on the query and read the
    cache, so a linear stack at one width over a cache at another is silent corruption rather than
    an error.

    A build is not the only reader, which is why this is called from a test-wide fixture rather than
    from `_build_pair`: `EngramTable.lookup` narrows to `LINEAR_DTYPE` *per call*, inside
    `Engram.forward`, so a build pinned and unpinned around construction leaves every weight at one
    width and hands the first `Engram.wkv` an activation at the other. The error that produces is
    `expected m1 and m2 to have the same dtype`, raised from a stock `nn.Linear` inside a module
    named nowhere in this file.
    """
    from src.models.deepseek_v4_1 import attention, loader, modules

    previous = {"CACHE_DTYPE": attention.CACHE_DTYPE}
    for name, module in (("attention", attention), ("loader", loader), ("modules", modules)):
        previous[name] = module.LINEAR_DTYPE
        module.LINEAR_DTYPE = dtype
    attention.CACHE_DTYPE = dtype
    return previous


def _restore_dense_dtype(previous: dict[str, torch.dtype]) -> None:
    from src.models.deepseek_v4_1 import attention, loader, modules

    attention.CACHE_DTYPE = previous["CACHE_DTYPE"]
    for name, module in (("attention", attention), ("loader", loader), ("modules", modules)):
        module.LINEAR_DTYPE = previous[name]


@pytest.fixture(autouse=True)
def reference_width():
    """Hold this tree at the reference's width for the whole test, construction and forward alike.

    Autouse because every test here ends in a comparison against the reference, and a test that ran
    at the tree's own width would be comparing two widths -- which is the one thing this file is not
    allowed to do. See `_pin_dense_dtype`.
    """
    previous = _pin_dense_dtype(torch.bfloat16)
    yield
    _restore_dense_dtype(previous)


def _build_pair(seed: int = 0, **overrides):
    """The reference model and this repository's backbone, sharing one config and one set of weights."""
    reference = _load_reference()
    args = _tiny_args(reference, **overrides)
    default_dtype = torch.get_default_dtype()
    # The reference's own entry point opens with `torch.set_default_dtype(torch.bfloat16)`, and it is
    # load-bearing: `ParallelEmbedding.weight`, `Gate.weight` and `Engram.q_weight`/`k_weight` are
    # bare `torch.empty`/`torch.ones` and take their dtype from here. Under the fp32 default they
    # come out wider than the checkpoint holds them, and the copy into this tree -- which is bf16
    # throughout by declaration -- would be lossy.
    torch.set_default_dtype(torch.bfloat16)
    try:
        torch.manual_seed(seed)
        ref_model = reference.Transformer(args)
        _add_vision_gate_bias(ref_model)
        _randomize(ref_model, seed)
        cfg = from_dict(dataclasses.asdict(args)).text
        layout = ref_model.engram_layout
        torch.manual_seed(seed)
        ours = Backbone(
            cfg,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            layout=layout,
            engram_tables={
                layer_id: ResidentEngramTable(
                    weight=torch.empty(layout.num_embeddings[i], layout.head_dim, dtype=torch.float8_e4m3fn),
                    scale=torch.empty(
                        layout.num_embeddings[i], layout.head_dim // 32, dtype=torch.float8_e8m0fnu
                    ),
                )
                for i, layer_id in enumerate(layout.layer_ids)
            },
        )
    finally:
        torch.set_default_dtype(default_dtype)
    # The `reference_width` fixture owns the dense width, and this is the cheap check that it is
    # still in force: a call from outside a test would build a tree the copy below cannot fill.
    from src.models.deepseek_v4_1 import attention, modules

    if (modules.LINEAR_DTYPE, attention.CACHE_DTYPE) != (torch.bfloat16, torch.bfloat16):
        raise RuntimeError(
            "this pair must be built under the `reference_width` fixture, which holds the tree at "
            f"the reference's bf16; the tree is at {modules.LINEAR_DTYPE}"
        )
    mismatched, unfilled = _copy_weights(ref_model, ours)
    assert not mismatched, f"reference tensors this tree cannot hold: {mismatched}"
    assert not unfilled, f"this tree's tensors left at random init: {unfilled}"
    ref_model.eval()
    ours.eval()
    return ref_model, ours, args


def _copy_weights(reference: torch.nn.Module, ours: Backbone) -> tuple[list[str], list[str]]:
    """Copy every reference tensor into the matching slot of ours, returning what did not line up.

    One name differs, and it is a structural one: the reference builds 384 `Expert` modules per
    layer and this tree builds a single bank, so `ffn.experts.{j}.w{k}.weight` fills
    `ffn.routed.w{k}[j]`. Every other name is key-for-key, which is not a coincidence -- the loader
    depends on it -- so reporting the ones that do not line up, rather than skipping them, is what
    keeps it true.

    A name that matches is not yet a tensor that fits: the two trees are allowed to hold the same
    value at different widths (`RMSNorm.weight` is fp32 here and bf16 in the reference, and the
    reference's `ParallelHead` keeps in fp32 what the checkpoint stores bf16), so each copy is
    checked to round-trip. A widening passes; a narrowing that changed a bit does not, and that is
    what a genuine dtype disagreement looks like.
    """
    target = dict(ours.state_dict())
    written: set[str] = set()
    mismatched: list[str] = []
    for key, value in reference.state_dict().items():
        expert = _EXPERT_KEY.match(key)
        owner = f"layers.{expert[1]}.ffn.routed.w{expert[3]}" if expert else key
        slot = target.get(owner)
        if slot is None:
            mismatched.append(f"{key}: no such tensor in this tree")
            continue
        if expert:
            slot = slot[int(expert[2])]
        if tuple(slot.shape) != tuple(value.shape):
            mismatched.append(f"{key}: {tuple(value.shape)} -> {tuple(slot.shape)}")
            continue
        with torch.no_grad():
            slot.copy_(value)
        if slot.dtype != value.dtype and not slot.to(value.dtype).equal(value):
            mismatched.append(f"{key}: {value.dtype} -> {slot.dtype} is lossy")
            continue
        written.add(owner)
    return mismatched, sorted(set(target) - written)


def _hash(ref_model, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """The reference's own n-gram row ids, checked to be addressable in every layer's table.

    Text-only input, so the reference would pass `engram_mask=None` and so does this. Calling
    `engram_hash` outside the model's forward is safe -- it writes positions the forward is about to
    write again with the same values -- and it is what lets one set of ids feed both models.
    """
    ids = ref_model.engram_hash(input_ids, start_pos)
    rows = min(ref_model.engram_layout.num_embeddings)
    assert ids.shape == (
        input_ids.size(0),
        input_ids.size(1),
        len(ref_model.engram_layout.layer_ids),
        (ref_model.engram_layout.max_ngram_size - 1) * ref_model.engram_layout.n_heads,
    ), ids.shape
    assert ids.min() >= 0 and ids.max() < rows, f"row id {ids.max().item()} outside a {rows}-row table"
    return ids


@requires_reference
def test_backbone_reproduces_the_reference_prefill():
    ref_model, ours, args = _build_pair()
    input_ids = torch.randint(0, args.vocab_size, (2, 12))
    ids = _hash(ref_model, input_ids)

    torch.manual_seed(11)
    _, ref_logits, ref_main = ref_model(input_ids)
    # The reference samples with the global RNG, so seed again: the comparison below is of logits,
    # and letting the sampler consume a different number of draws would only hide that.
    torch.manual_seed(11)
    _, our_logits, our_main = ours(input_ids, hash_ids=ids)

    assert torch.allclose(our_logits, ref_logits, atol=1e-4, rtol=1e-4), (
        f"max abs diff {(our_logits - ref_logits).abs().max().item()}"
    )
    assert ref_main is not None and our_main is not None
    assert torch.allclose(our_main, ref_main, atol=1e-4, rtol=1e-4)


@requires_reference
def test_backbone_reproduces_the_reference_decode_step():
    """The step after a prefill, which is the one that reads state the prefill wrote.

    The compressor's partial group, the window ring buffer and the index keys all carry across the
    split, so a tree that got the prefill right and the hand-off wrong passes the first test and
    fails this one.

    Both models are prefilled before either decodes. There is no way to prefill one and not the
    other and still be comparing the same thing: the state a decode step reads *is* what the prefill
    wrote, so a reference whose caches were never filled would be decoding from zeros.
    """
    ref_model, ours, args = _build_pair(seed=1)
    prompt = torch.randint(0, args.vocab_size, (2, 12))
    step = torch.randint(0, args.vocab_size, (2, 1))

    ours(prompt, hash_ids=_hash(ref_model, prompt))
    ref_model(prompt)

    _, our_logits, _ = ours(step, start_pos=12, hash_ids=_hash(ref_model, step, 12))
    _, ref_logits, _ = ref_model(step, 12)

    assert torch.allclose(our_logits, ref_logits, atol=1e-4, rtol=1e-4), (
        f"max abs diff {(our_logits - ref_logits).abs().max().item()}"
    )


@requires_reference
def test_weight_copy_leaves_nothing_random():
    """A guard on the harness itself: a rename that empties `written` would otherwise pass silently."""
    _, ours, _ = _build_pair(seed=2)
    filled = dict(ours.named_parameters())
    assert len(filled) > 100, "the tiny config should still build a full module tree"
    assert all(torch.isfinite(p).all() for p in filled.values())
