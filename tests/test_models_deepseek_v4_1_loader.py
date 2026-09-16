"""Feed a checkpoint into the V4.1 tree, and check the three things a load can get wrong.

`src/models/deepseek_v4_1/loader.py` is the only place that knows which name in a shard is which
parameter in the tree, which of them are quantized, and how an Engram row is addressed. None of that
can be checked against the released 476 GiB checkpoint from a test that has to run anywhere, so what
is here is a *miniature* checkpoint written to `tmp_path`: the same names in the same layouts, over a
toy geometry small enough to enumerate. Loading it exercises the whole path -- the `weight`/`scale`
pairing, the derived block width, the fp4 expert rebanking, the Engram row gather -- and the
assertions are against values computed in this file, not against the loader's own output.

Three of these tests are about the checkpoint's *shape* rather than its numbers, and they correspond
to real properties of the release:

* a KV source layer whose `compress_ratio` is 1 has no `compressor.wgate` -- layer 20 of the released
  model, and layer 4 of the geometry below -- so a per-KV-source key list would be wrong;
* the routed experts are `ffn.experts.{j}.w{k}` in the file and one stacked `ffn.routed.w{k}` in the
  tree, so a name-for-name copy would leave the entire MoE at `torch.empty`;
* the vision tower, the aligner, the image tokens and the DSpark draft layers are wanted by nothing
  the text backbone builds, and the report counts them rather than passing over them in silence.

`EngramHashIds` needs none of that. It is a second *formulation* of `src/encoding/engram.NgramHasher`
-- which is stdlib-only, and therefore a real oracle -- and the tests below hold the two to the same
ids on a token stream split across a prefill and a decode. That comparison is the reason the tensor
version is allowed to exist, and the split is the case a cache exists for: a port that re-reads the
prompt instead of carrying it agrees with the reference on a whole prefill and disagrees on step two.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.encoding.engram import EngramLayout, NgramHasher
from src.loader.safetensors import MmapSafetensors
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.kernels import _fp4_values
from src.models.deepseek_v4_1.loader import (
    CheckpointEngramTable,
    CheckpointRoutedExperts,
    EngramHashIds,
    V41Checkpoint,
    checkpoint_weights,
    load_backbone,
    scale_key,
)
from src.models.deepseek_v4_1.modules import (
    Backbone,
    ResidentEngramTable,
    ResidentRoutedExperts,
    dequantize_rows,
)

# The released geometry's two awkward properties, and nothing else large. Layer 4 is a KV source whose
# compress_ratio is 1, so like the released layer 20 it owns no `wgate`; layers 0 and 3 carry Engram
# tables; and the ratios mix 2 and 1 so both compressor branches run.
DIM = 64
INTER_DIM = 64  # a multiple of BLOCK, because `w2` is `[dim, inter_dim]` and its scale runs along K
N_EXPERTS = 6
N_LAYERS = 6
VOCAB = 64
ENGRAM_HEAD_DIM = 32
ENGRAM_FIELDS = dict(
    engram_layer_ids=(0, 3),
    engram_max_ngram_size=2,
    engram_vocab_size=24,
    engram_n_heads=2,
    engram_head_dim=ENGRAM_HEAD_DIM,
)
# Deliberately neither expander's default: `dequant_fp8_weight` assumes 128 and `dequant_fp4_weight`
# assumes 32, and the release uses a square 32x32 grid. A loader that took either default would pass
# on this checkpoint only by coincidence, so this one is a width neither of them names.
BLOCK = 16
MINI = dict(
    dim=DIM,
    moe_inter_dim=INTER_DIM,
    n_routed_experts=N_EXPERTS,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    vocab_size=VOCAB,
    n_layers=N_LAYERS,
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
    # wide enough that nothing is truncated: a truncated index source makes prefill and decode differ
    # as a tie artifact rather than as an arithmetic error -- see `test_..._attention.py`
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
    engram_compressed_vocab_size=VOCAB,
    engram_pad_id=1,
    **ENGRAM_FIELDS,
)

# The dense projections the release stores quantized, so the mini one does too. This is the whole set
# rather than a sample: eight per layer in the released checkpoint (`attn.wkv`, `attn.wq_a`,
# `attn.wq_b`, `attn.wo_a`, `attn.wo_b`, and the three shared experts), plus `attn.indexer.wq_b` in the
# layers that index, which is why its census per layer reads 2312 and not 2304. The two whose scale
# rows a test has to be able to tell apart -- a 2-D grid on a dense weight, a row per row on a table --
# are both here, because a loader that read either as the other would be wrong on half the shards.
FP8_SUFFIXES = (
    "attn.wkv.weight",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.indexer.wq_b.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
)


def _layout() -> EngramLayout:
    """The layout the config implies. The primes come from `engram_vocab_size` and the head count
    alone, so this is the same layout whatever `engram_num_embeddings` says -- which is why the row
    counts can be written back into the config in a second pass without moving the primes."""
    layout = EngramLayout.from_config(V41TextConfig(**{**MINI, "engram_num_embeddings": (0, 0)}).__dict__)
    assert layout is not None
    return layout


def _cfg() -> V41TextConfig:
    """The toy config, declaring the row counts the layout actually derives.

    `engram_num_embeddings` is not what sizes a table -- `EngramLayout.row_counts` sums the primes --
    but a config claiming 64 rows for a layout whose hasher says 60 is a lie that would eventually be
    load-bearing, and the released pair agree, so this one does too.
    """
    return V41TextConfig(**{**MINI, "engram_num_embeddings": _layout().row_counts()})


def _e8m0(x: torch.Tensor) -> torch.Tensor:
    """The checkpoint's scale dtype: a power of two, rounded up, so no block can saturate."""
    tiny = torch.finfo(torch.float32).tiny
    return torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(x, min=tiny)))).to(torch.float8_e8m0fnu)


def _quantize_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One dense fp8 weight and its scale grid, in the release's layout.

    A dense projection's scale is a two-dimensional grid -- `scale[out // 32, in // 32]`, one value
    for each square block -- and `kernels.dequant_fp8_weight` expands it along *both* axes. That is
    not the layout the Engram tables use, so a test that wrote one spelling for both would pass while
    the loader read the other. The release tells them apart: `layers.0.attn.wq_a.scale` is `(40, 160)`
    of a `(1280, 5120)` weight, while `layers.0.engram.embed.scale` is `(rows, 8)` of a
    `(rows, 256)` one.
    """
    n, k = w.shape
    assert n % BLOCK == 0 and k % BLOCK == 0
    blocks = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK).permute(0, 2, 1, 3)
    scale = _e8m0(blocks.abs().amax(dim=(2, 3)))
    codes = (blocks / scale.float()[..., None, None]).to(torch.float8_e4m3fn)
    return codes.permute(0, 2, 1, 3).reshape(n, k).contiguous(), scale.contiguous()


def _quantize_row_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One fp8 matrix whose scale keeps a row per output row, which is the table layout.

    `engram.embed.scale` is `[rows, head_dim // 32]` in the release, and a packed fp4 expert's is
    `[out, in // 32]`: both put the block structure along K only. `CheckpointEngramTable` gathers
    rows of both tensors and hands them to `dequantize_rows`, so a table written on the dense grid
    would read the wrong scales rather than fail outright.
    """
    n, k = w.shape
    assert k % BLOCK == 0
    blocks = w.float().reshape(-1, k // BLOCK, BLOCK)
    scale = _e8m0(blocks.abs().amax(-1))
    codes = (blocks / scale.float().unsqueeze(-1)).to(torch.float8_e4m3fn)
    return codes.reshape(n, k), scale.reshape(n, k // BLOCK)


def _quantize_fp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a bf16 weight whose values are already *on* the four-bit ladder.

    The ladder's magnitudes doubled are `{0, 1, 2, 3, 4, 6, 8, 12}` and their negatives, every one of
    them exact in bf16, so a bank holding those values is encoded bit for bit and the tests below can
    assert equality rather than a tolerance. A tolerance would pass on a swapped nibble order or a
    scale applied along the wrong axis, which is exactly what a packing convention gets wrong.
    """
    n, k = w.shape
    assert k % BLOCK == 0
    ladder = (_fp4_values(torch.arange(16)) * 2.0).to(torch.bfloat16)
    codes = (w.reshape(-1, 1) == ladder.reshape(1, -1)).to(torch.uint8).argmax(-1).reshape(n, k)
    assert torch.equal(ladder[codes], w), "the value is not on the four-bit ladder"
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    scale = torch.full((n, k // BLOCK), 2.0).to(torch.float8_e8m0fnu)
    return packed, scale


def _on_the_ladder(shape, generator: torch.Generator) -> torch.Tensor:
    """A bf16 tensor of values the four-bit ladder can hold exactly -- see `_quantize_fp4`."""
    ladder = (_fp4_values(torch.arange(16)) * 2.0).to(torch.bfloat16)
    return ladder[torch.randint(0, 16, shape, generator=generator)]


def _reference(cfg: V41TextConfig, layout: EngramLayout, tables: dict[int, tuple]) -> Backbone:
    """A `Backbone` holding every weight the tree has, so the file's names are derived rather than
    typed out. What the tree calls `ffn.routed.w1` is written once per expert below, which is the one
    structural difference between the two spellings; nothing else is renamed.

    The routed experts are given values from the four-bit ladder rather than arbitrary ones, and the
    Engram tables are handed in already quantized, because in the file both are quantized: a bank
    holding values its own encoding cannot reproduce would push every assertion below onto a
    tolerance, which is the one thing a packing convention needs it not to be.
    """
    model = Backbone(
        cfg,
        max_batch_size=1,
        max_seq_len=cfg.max_position_embeddings,
        layout=layout,
        engram_tables={layer_id: ResidentEngramTable(*tables[layer_id], BLOCK) for layer_id in layout.layer_ids},
        routed={i: ResidentRoutedExperts(N_EXPERTS, DIM, INTER_DIM, cfg.swiglu_limit) for i in range(N_LAYERS)},
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            parameter.copy_(values.to(parameter.dtype))
        for block in model.layers:
            for which in ("w1", "w2", "w3"):
                parameter = getattr(block.ffn.routed, which)
                parameter.copy_(_on_the_ladder(parameter.shape, generator))
    return model


def _mini_checkpoint(root: str, cfg: V41TextConfig, layout: EngramLayout) -> tuple[dict, dict]:
    """Write a mini checkpoint holding every name the tree will ask for.

    Returns the tensors as written and the reference `state_dict` they were derived from, because
    that dict is the oracle for the expert rebanking: the fp4 in the file is an exact encoding of the
    bf16 the tree's own stacked bank holds, so the two have to come back equal.
    """
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(11)
    tables = {
        layer_id: _quantize_row_fp8(
            torch.randn(layout.row_counts()[index], ENGRAM_HEAD_DIM, generator=generator) * 0.5
        )
        for index, layer_id in enumerate(layout.layer_ids)
    }
    reference = _reference(cfg, layout, tables)
    state = reference.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        parts = name.split(".")
        if len(parts) > 4 and parts[2] == "ffn" and parts[3] == "routed":
            # ffn.routed.w{k} -> one packed expert per leading index, which is how the release holds
            # 384 experts without 384 module names.
            which = parts[4]
            for expert in range(value.shape[0]):
                packed, scale = _quantize_fp4(value[expert].contiguous())
                tensors[f"layers.{parts[1]}.ffn.experts.{expert}.{which}.weight"] = packed
                tensors[f"layers.{parts[1]}.ffn.experts.{expert}.{which}.scale"] = scale
        elif value.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            # The Engram tables are already in the checkpoint's layout, because they cannot be
            # anywhere else: `ResidentEngramTable` cannot gather fp8 rows on the CPU.
            tensors[name] = value.contiguous()
        elif name.startswith("layers.") and name.endswith(FP8_SUFFIXES):
            codes, scale = _quantize_fp8(value.float())
            tensors[name] = codes
            tensors[scale_key(name)] = scale
        else:
            tensors[name] = value.contiguous()

    # Tensors the text backbone does not build, spelled the way the release spells them.
    tensors["vision.patch_embed.proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    tensors["aligner.w1.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    tensors["image_newline"] = torch.zeros(4, dtype=torch.bfloat16)
    tensors["mtp.0.ffn.gate.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)

    save_file(tensors, f"{root}/model.safetensors", metadata={"format": "pt"})
    with open(f"{root}/model.safetensors.index.json", "w") as handle:
        json.dump({"metadata": {}, "weight_map": {name: "model.safetensors" for name in tensors}}, handle)
    return tensors, state


def _toy_hasher(layout: EngramLayout) -> NgramHasher:
    """A hasher over an identity token map, so a token id is its own compressed id.

    `build_hasher` needs a real tokenizer; nothing in the hash path can tell the difference, since it
    only ever indexes the map. The map has to cover the config's vocabulary, or a token id used below
    would index past its end.
    """
    return NgramHasher(layout, list(range(VOCAB)), pad_id=MINI["engram_pad_id"])


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """One mini checkpoint for the whole file: writing it costs more than every test on it."""
    root = str(tmp_path_factory.mktemp("v41-mini"))
    cfg, layout = _cfg(), _layout()
    stored, reference = _mini_checkpoint(root, cfg, layout)
    checkpoint = V41Checkpoint(root)
    loaded = load_backbone(cfg, checkpoint, layout=layout, hasher=_toy_hasher(layout))
    loaded.model.temperature = 0.0
    yield SimpleNamespace(
        root=root, cfg=cfg, layout=layout, stored=stored, reference=reference, ckpt=checkpoint, loaded=loaded
    )
    checkpoint.close()


# -- the hash bridge --------------------------------------------------------------------------


def test_the_vectorized_hash_ids_agree_with_the_host_hasher_on_a_whole_prefill() -> None:
    """The one oracle in this file. `NgramHasher` is stdlib Python with no tensor library in it."""
    layout = _layout()
    hasher = _toy_hasher(layout)
    tokens = [3, 17, 51, 5, 0, 61, 42, 42, 8, 12, 47]

    got = EngramHashIds(hasher, max_seq_len=64)(torch.tensor([tokens]))
    want = torch.tensor(hasher.hash_ids(tokens))

    assert got.shape == (1, len(tokens), len(layout.layer_ids), layout.n_hash_columns)
    assert torch.equal(got[0], want)
    # and every id lands inside its layer's table, which is what makes the row gather in range
    for layer, rows in enumerate(layout.row_counts()):
        assert int(want[..., layer, :].max()) < rows


def test_the_vectorized_hash_ids_agree_across_a_prefill_then_decode_split() -> None:
    """The split is what the cache exists for, and where a plausible port goes wrong.

    `NgramHasher` carries its own history, so it has to be fed the same prefill; comparing a decode
    step against a hasher that was only ever handed that one token compares a cache against nothing.
    An earlier draft of this probe did exactly that and reported a mismatch at the split.
    """
    layout = _layout()
    hasher = _toy_hasher(layout)
    tokens = [3, 17, 51, 5, 0, 61, 42, 42, 8, 12, 47]
    split = 5

    front = EngramHashIds(hasher, max_seq_len=64)
    front(torch.tensor([tokens[:split]]), 0)
    hasher.hash_ids(tokens[:split], 0)
    for position in range(split, len(tokens)):
        step = front(torch.tensor([[tokens[position]]]), position)
        want = torch.tensor(hasher.hash_ids([tokens[position]], position))
        assert torch.equal(step[0, 0], want[0]), f"position {position}"


def test_a_masked_token_is_dead_for_every_ngram_that_would_span_it() -> None:
    """An image span takes no part in an n-gram, and `token_mask=False` is how that is said.

    `LoadedBackbone` turns the image mask into this mask itself, so the two spellings have to be the
    same statement: a masked position hashes like an unknown token, and the position after it loses
    the n-gram that would have reached back across the mask.
    """
    layout = _layout()
    hasher = _toy_hasher(layout)
    tokens = [3, 17, 51, 5, 0, 61, 42, 42]
    mask = torch.ones(1, len(tokens), dtype=torch.bool)
    mask[0, 4] = False

    got = EngramHashIds(hasher, max_seq_len=64)(torch.tensor([tokens]), 0, mask)
    want = torch.tensor(hasher.hash_ids(tokens, 0, mask[0].tolist()))

    assert torch.equal(got[0], want)
    # and it is not a no-op: masking one token changes the ids there and at the next position
    plain = EngramHashIds(hasher, max_seq_len=64)(torch.tensor([tokens]), 0)
    assert not torch.equal(plain[0, 4], got[0, 4])
    assert not torch.equal(plain[0, 5], got[0, 5])


def test_reset_clears_the_hash_history() -> None:
    """Two forwards from a reset state are the same forward, which is what makes a test repeatable."""
    hasher = _toy_hasher(_layout())
    tokens = torch.tensor([[3, 17, 51, 5, 0, 61]])
    front = EngramHashIds(hasher, max_seq_len=64)
    once = front(tokens, 0)
    again = front(tokens, 0)
    assert torch.equal(once, again), "the first forward left a cache behind"
    front.reset()
    assert torch.equal(front(tokens, 0), once)


# -- the names, the layouts and the scales ----------------------------------------------------


def test_every_parameter_the_tree_names_is_in_the_checkpoint(mini) -> None:
    """The load's own guard, and the census that says what it deliberately left behind.

    A name the file does not have stays `torch.empty`, which is not a NaN until something reads it, and
    by then it is forty layers away from the mistake. The other half is the accounting: what a load
    reports as unread has to be exactly the things a load is allowed to leave unread -- the expert
    projections, the scale companions of the weights it did fill, and the Engram tables -- or a
    projection silently skipped would hide in the same number as a scale nobody needs.
    """
    report = mini.loaded.report
    assert report.ok, report.missing
    loaded = set(report.loaded)
    for layer_id in range(N_LAYERS):
        group = f"layers.{layer_id}."
        unread = [key for key in mini.stored if key.startswith(group) and key not in loaded]
        for key in unread:
            leaf = key[len(group) :]
            assert "ffn.experts." in leaf or leaf.endswith(".scale") or leaf.startswith("engram."), key
        assert report.unloaded_groups[f"layers.{layer_id}"] == len(unread)
        # and every expert projection is among them: N_EXPERTS x 3 matrices x (weight, scale)
        assert len([key for key in unread if ".ffn.experts." in key]) == N_EXPERTS * 3 * 2
    # the names nothing in the text backbone builds are counted too, not dropped
    other = {group: count for group, count in report.unloaded_groups.items() if not group.startswith("layers.")}
    assert other == {"aligner": 1, "image_newline": 1, "mtp": 1, "vision": 1}


def test_the_expert_bank_is_repacked_from_the_per_expert_names(mini) -> None:
    """`ffn.experts.{j}.w{k}` in the file, one stacked `ffn.routed.w{k}` in the tree.

    The oracle is the tree's own bank: the fp4 in the file is an exact encoding of the bf16 the
    resident bank holds, so a name-for-name copy -- which would leave the stack uninitialized -- and a
    transposed nibble order both fail here.
    """
    store = CheckpointRoutedExperts(mini.ckpt, 0, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER_DIM, cache_size=2)
    for expert in (0, 1, N_EXPERTS - 1):
        for which, got in zip(("w1", "w2", "w3"), store.expert(expert)):
            assert got.dtype == torch.bfloat16
            assert torch.equal(got, mini.reference[f"layers.0.ffn.routed.{which}"][expert])


def test_the_expert_cache_evicts_in_order_and_reloads_what_it_dropped(mini) -> None:
    """A miss is the whole cost of `CheckpointRoutedExperts`, so what counts as one is worth pinning.

    First-in-first-out, not least-recently-used: a step touches the experts its own token routed to,
    so recency within a layer carries no information. The read the release makes is expensive enough
    that "the cache asked the disk twice" and "the cache answered" have to be distinguishable.
    """
    store = CheckpointRoutedExperts(mini.ckpt, 1, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER_DIM, cache_size=2)
    store.expert(0)
    store.expert(1)
    store.expert(1)
    assert store.misses == 2, "a hit must not count as a miss"
    store.expert(0)  # still resident: reading 1 again evicted nothing
    assert store.misses == 2
    store.expert(2)  # evicts 0, the oldest
    store.expert(0)
    assert store.misses == 4


def test_an_fp8_weight_expands_on_the_derived_block_grid(mini) -> None:
    """The block width is derived from the stored shape and the scale shape, not assumed.

    `dequant_fp8_weight` defaults to 128 and the release uses 32, so a loader that took the default is
    wrong on every dense projection in the checkpoint; taking `block_size` from the pair of shapes also
    means a checkpoint laid out differently fails with both widths in the message. The scale is a grid
    on both axes, and the product below expands it on both -- an expansion that only repeated along K
    would agree with this one on the first block row and nowhere else.
    """
    name = "layers.0.attn.wq_a.weight"
    assert mini.ckpt.is_quantized(name) and not mini.ckpt.is_packed_fp4(name)
    assert mini.ckpt.block_size(name) == BLOCK

    got = mini.ckpt.weight(name, dtype=torch.float32)
    scale = mini.stored[scale_key(name)].float()
    want = mini.stored[name].float() * scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    assert got.shape == tuple(mini.stored[name].shape)
    assert torch.equal(got, want)
    # the grid is what the width came from, so the product above is not a coincidence
    assert scale.shape == (mini.stored[name].shape[0] // BLOCK, mini.stored[name].shape[1] // BLOCK)
    # and the table layout is the other spelling, so this checkpoint can tell the two apart
    table = scale_key("layers.0.engram.embed.weight")
    assert mini.ckpt.block_size("layers.0.engram.embed.weight") == BLOCK
    assert mini.stored[table].shape == (mini.stored["layers.0.engram.embed.weight"].shape[0], ENGRAM_HEAD_DIM // BLOCK)


def test_a_layer_with_a_ratio_of_one_has_no_compressor_gate(mini) -> None:
    """The release's layer 20, reproduced: `Compressor.__init__` returns before `wgate` exists.

    So the file has no such name and the loader asks only for what the tree built -- which is why a
    per-KV-source key list would be wrong here. The compressor's own `wkv` and `norm` are still
    there, so this is a missing *gate* and not a missing layer.
    """
    names = set(mini.loaded.model.state_dict())
    assert "layers.4.attn.compressor.wgate.weight" not in names
    assert "layers.4.attn.compressor.wkv.weight" in names
    assert "layers.4.attn.compressor.norm.weight" in names
    # layer 2 is a KV source with a ratio of 2, and does own one
    assert "layers.2.attn.compressor.wgate.weight" in names
    assert "layers.4.attn.compressor.wgate.weight" not in mini.stored


def test_the_engram_table_gathers_the_rows_the_hasher_named(mini) -> None:
    """The two released tables are 91 GiB each and are never read whole.

    A position resolves to one row per hash column and there are 24 of those, so the widest a prefill
    ever gathers is `positions x 24` rows. Checked against a plain alias gather on the same stored
    bytes, so a table that read the wrong layer, the wrong offset, or a view where a copy is wanted
    fails here; the indices include a repeat, because two hash columns resolving to the same row is
    the normal case and a view would alias where a copy is required.
    """
    for index, layer_id in enumerate(mini.layout.layer_ids):
        weight_key = f"layers.{layer_id}.engram.embed.weight"
        rows = mini.layout.row_counts()[index]
        indices = torch.tensor([0, 1, rows - 1, 4, 4, 2])

        store = CheckpointEngramTable(mini.ckpt, layer_id)
        got = store.lookup(indices)
        want = dequantize_rows(
            mini.stored[weight_key].view(torch.uint8)[indices].view(torch.float8_e4m3fn),
            mini.stored[scale_key(weight_key)].view(torch.uint8)[indices].view(torch.float8_e8m0fnu),
            BLOCK,
        )
        assert got.shape == (indices.numel(), ENGRAM_HEAD_DIM)
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, want)
        assert store.rows_gathered == indices.numel()

        wider = indices.reshape(2, 3)
        assert CheckpointEngramTable(mini.ckpt, layer_id).lookup(wider).shape == (2, 3, ENGRAM_HEAD_DIM)


def test_a_skipped_name_is_not_a_missing_one(mini) -> None:
    """`skip` is how a caller declares storage the file does not name, and `prefix` how it says where.

    The distinction is the whole reason `LoadReport` has both fields: a resident bank or a truncated
    table is not named after anything in the checkpoint, and calling that a gap would make the one
    field that matters -- `missing` -- useless.
    """
    probe = nn.Linear(DIM, DIM, bias=False)  # its only parameter is named "weight", which the file has not
    assert checkpoint_weights(probe, mini.ckpt).missing == ["weight"]
    report = checkpoint_weights(probe, mini.ckpt, skip=lambda key: key == "weight")
    assert report.skipped == ["weight"] and not report.missing and report.ok

    # and a real name loads into a module whose own name is not the file's
    head = nn.Linear(DIM, VOCAB, bias=False)
    report = checkpoint_weights(head, mini.ckpt, prefix="head.")
    assert report.loaded == ["head.weight"] and not report.missing
    assert torch.equal(head.weight, mini.stored["head.weight"].to(head.weight.dtype))


def test_the_load_goes_through_the_mmap_reader(mini) -> None:
    """Nothing above this line would notice if the loader opened the shards itself and copied them."""
    assert isinstance(mini.ckpt.reader, MmapSafetensors)
    assert len(mini.ckpt) == len(mini.stored)
    assert "layers.0.attn.wq_a.weight" in mini.ckpt


# -- a load that runs -------------------------------------------------------------------------


def test_the_dense_load_leaves_every_store_reading_from_the_shards(mini) -> None:
    """What a load allocates is the dense half. The two Engram tables and every layer's bank of 384
    experts must still be reading from the shards when it is done, or the 476 GiB became an
    allocation and the model never runs on four 22 GiB cards."""
    model = mini.loaded.model
    for index, layer_id in enumerate(mini.layout.layer_ids):
        table = model.layers[layer_id].engram.embed
        assert isinstance(table, CheckpointEngramTable)
        assert table.rows_total == mini.layout.row_counts()[index]
    names = set(model.state_dict())
    for block in model.layers:
        assert isinstance(block.ffn.routed, CheckpointRoutedExperts)
        assert not [name for name in names if name.startswith(f"layers.{block.layer_id}.ffn.routed.")]


def test_the_loaded_backbone_runs_a_prefill_and_a_stepwise_decode_identically(mini) -> None:
    """The integration check: the same tokens through both orderings, from the same weights.

    This is the property the whole loader is for. A load that put the wrong tensor in the wrong place
    would still produce finite logits of the right shape, so what is asserted is that prefill and
    decode agree -- an oracle-free check covering the attention caches, the compressor's group state,
    the Engram hash cache and the routed expert store at once.

    `Head` returns the last position only, as the reference's generation loop wants, so agreement at
    position `k` is read by prefilling `tokens[:k + 1]` and comparing against the `k`-th step: every
    position is compared, by a prefill that never saw the tokens after it.
    """
    tokens = [5, 3, 17, 11]
    front = mini.loaded

    front.reset_state(1)
    _, prefill, main_hidden = front(torch.tensor([tokens]), 0)
    assert main_hidden is None, "no DSpark target layers are configured, so nothing is published"
    assert prefill.shape == (1, VOCAB), "the head publishes the last position only"
    assert torch.isfinite(prefill).all()

    front.reset_state(1)
    steps = []
    for position, token in enumerate(tokens):
        _, logits, _ = front(torch.tensor([[token]]), position)
        steps.append(logits[0].clone())
    assert torch.equal(steps[-1], prefill[0]), "a one-shot prefill disagrees with the stepwise decode"

    for length in range(1, len(tokens)):
        front.reset_state(1)
        _, logits, _ = front(torch.tensor([tokens[:length]]), 0)
        assert torch.equal(logits[0], steps[length - 1]), f"prefill to {length} disagrees at the last position"


def test_the_load_report_names_the_unfilled_and_counts_the_unread(mini) -> None:
    """`summary` is the only thing a load tells a caller, so it has to name unfilled parameters
    rather than count them, and count the deliberately unread groups rather than name them."""
    report = mini.loaded.report
    text = report.summary()
    assert "no parameter left unfilled" in text and "MISSING" not in text
    assert report.bytes_read > 0
    assert report.quantized, "the dense quantized weights have to be reported as such"
    assert f"{len(report.loaded)} tensors" in text
