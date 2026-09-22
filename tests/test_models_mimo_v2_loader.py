"""Reading MiMo-V2.6: the release's layout, and the miniature that stands in for it.

Two halves, deliberately.

The first half writes a *miniature* MiMo checkpoint to `tmp_path`: the same file
names, the same shard convention, the same six-tensor expert runs, the same FP8 /
MXFP4 / BF16 split, over a four-layer geometry small enough to enumerate by hand.
The loader's whole job is to turn headers into addresses, and headers are the one
part of a 172 GiB release a test can reproduce exactly -- so these run everywhere,
and the offsets they assert against are parsed from the written file by this
module rather than read back through the loader.

The second half checks the same properties on the actual release when it is on
this host. Those pin the numbers the miniature only stands in for: 13568 against
14848 columns of fused qkv, 4 experts per shard, 12.75 MiB per expert, 65 shard
files. They skip when the checkpoint is absent, and a skip is a skip.

The two halves exist because they fail differently. A miniature that drifts from
the release proves nothing about the release, and a release-only test that skips
in CI proves nothing at all.

One thing the miniature deliberately does *not* have is an index. The release has
one, and `test_the_released_index_agrees_with_the_headers` checks that against the
release; the miniature's job is the path a reader takes when the only statement of
where a tensor lives is the tensor's own file.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from src.models.mimo_v2.config import MimoV2Config  # noqa: E402
from src.models.mimo_v2.loader import (  # noqa: E402
    EXPERT_PROJECTIONS,
    MimoV2Checkpoint,
    dense_weight_keys,
    layer_weight_keys,
)
from src.models.mimo_v2.quant import (  # noqa: E402
    E2M1_LEVELS,
    QKV_SHARDS,
    dequant_fp8_block,
    dequant_mxfp4,
)

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release = pytest.mark.skipif(
    not HAS_RELEASE, reason=f"MiMo-V2.6 checkpoint not present at {RELEASE}"
)

SAFETENSORS_TO_TORCH = {
    "U8": torch.uint8,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
}
TORCH_TO_SAFETENSORS = {value: key for key, value in SAFETENSORS_TO_TORCH.items()}


# ---------------------------------------------------------------------------
# Reading and writing shards here, so the loader is not its own oracle
# ---------------------------------------------------------------------------


def raw_header(path: str) -> dict[str, tuple[str, tuple[int, ...], int, int]]:
    """Parse a shard's header here, independently of `MmapSafetensors`."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        out[name] = (meta["dtype"], tuple(meta["shape"]), begin, end)
    return out


def read_stored_tensor(path: str, entry: tuple[str, tuple[int, ...], int, int]) -> torch.Tensor:
    """Copy one tensor out of a shard, at its stored dtype."""
    dtype, shape, begin, end = entry
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + length + begin)
        raw = torch.frombuffer(bytearray(handle.read(end - begin)), dtype=torch.uint8)
    if SAFETENSORS_TO_TORCH[dtype] != torch.uint8:
        raw = raw.view(SAFETENSORS_TO_TORCH[dtype])
    return raw.reshape(shape).clone()


def write_shard(path: str, items: list[tuple[str, torch.Tensor]]) -> None:
    """Write a shard with the tensors in exactly this order.

    `safetensors.torch.save_file` orders by name, so it cannot express a shard
    whose tensors sit in some other order -- which is precisely what the malformed
    shards below are. The format is a little-endian header length, the header, and
    then the payload, so writing it here is a dozen lines and buys exact control.
    """
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, tensor in items:
        blob = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": TORCH_TO_SAFETENSORS[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(blob)],
        }
        payload += blob
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def rewrite_shard(root: str, file_name: str, transform) -> None:
    """Rebuild one shard, preserving its own byte order, through `transform`."""
    path = os.path.join(root, file_name)
    header = raw_header(path)
    items = [(name, read_stored_tensor(path, entry)) for name, entry in header.items()]
    write_shard(path, transform(items))


# ---------------------------------------------------------------------------
# The miniature checkpoint
# ---------------------------------------------------------------------------

#: A four-layer geometry with every property the loader cares about *active*: two
#: attention families with different fused widths, one dense layer and three
#: routed ones, and 8 experts over 4 shards so "expert e is in shard e // 2" is a
#: claim the test can break on purpose.
TINY = {
    "model_type": "mimo_v2",
    "architectures": ["MiMoV2ForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "v_head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 16,
    "swa_v_head_dim": 8,
    "rope_theta": 10_000_000.0,
    "swa_rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_value_scale": 0.707,
    "attention_projection_layout": "fused_qkv",
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "sliding_window": 4,
    "hybrid_layer_pattern": [0, 1, 0, 1],
    "moe_layer_freq": [0, 1, 1, 1],
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "n_shared_experts": None,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 1024,
    "layernorm_epsilon": 1e-6,
    "attention_chunk_size": 4,
    "eos_token_id": 127,
    "pad_token_id": 126,
}

EXPERTS_PER_SHARD = 2
#: A tile scale the whole miniature shares, so "was the scale applied" is checkable.
TILE_SCALE = 0.02


def fp8(codes: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float8_e4m3fn)


def tiny_layer_tensors(layer: int, config) -> dict[str, torch.Tensor]:
    """One decoder layer's backbone tensors, in the release's names and dtypes."""
    shape = config.attention(layer)
    root = f"model.layers.{layer}"
    rows, cols = shape.qkv_out, config.hidden_size
    # The released fused projection is quantised one tensor-parallel shard at a
    # time, so its scale carries a tile row per shard rather than one per run of
    # 128 rows of the whole weight. The two agree only when a shard is a whole
    # number of tiles, which at this size no shard is.
    scale_rows = QKV_SHARDS * -(-(rows // QKV_SHARDS) // 128)
    generator = torch.Generator().manual_seed(1000 + layer)
    tensors = {
        f"{root}.input_layernorm.weight": torch.ones(config.hidden_size),
        f"{root}.post_attention_layernorm.weight": torch.ones(config.hidden_size),
        f"{root}.self_attn.qkv_proj.weight": fp8(torch.randn(rows, cols, generator=generator)),
        f"{root}.self_attn.qkv_proj.weight_scale_inv": torch.full(
            (scale_rows, -(-cols // 128)), TILE_SCALE
        ),
        f"{root}.self_attn.o_proj.weight": torch.randn(
            config.hidden_size, shape.o_in, generator=generator
        ).to(torch.bfloat16),
    }
    if shape.has_sink:
        tensors[f"{root}.self_attn.attention_sink_bias"] = torch.zeros(shape.num_q_heads)
    if config.ffn_kind(layer) == "moe":
        tensors[f"{root}.mlp.gate.weight"] = torch.randn(
            config.n_routed_experts, config.hidden_size, generator=generator
        ).to(torch.bfloat16)
        tensors[f"{root}.mlp.gate.e_score_correction_bias"] = torch.zeros(
            config.n_routed_experts
        )
    else:
        width = config.ffn_intermediate_size(layer)
        for proj, out_features, in_features in (
            ("gate_proj", width, config.hidden_size),
            ("up_proj", width, config.hidden_size),
            ("down_proj", config.hidden_size, width),
        ):
            tensors[f"{root}.mlp.{proj}.weight"] = fp8(
                torch.randn(out_features, in_features, generator=generator)
            )
            tensors[f"{root}.mlp.{proj}.weight_scale_inv"] = torch.full(
                (-(-out_features // 128), -(-in_features // 128)), TILE_SCALE
            )
    return tensors


def tiny_expert_tensors(layer: int, expert: int, config) -> dict[str, torch.Tensor]:
    """One routed expert's six tensors, in the order the release stores them.

    Every code is a real E2M1 nibble drawn from the whole byte range and every E8M0
    scale is 127 -- that is, 1.0 -- so a dequantized expert is exactly the codebook
    and nothing else, which is what makes the round trip checkable.
    """
    hidden = config.hidden_size
    width = config.moe_intermediate_size
    generator = torch.Generator().manual_seed(7000 + 100 * layer + expert)
    root = f"model.layers.{layer}.mlp.experts.{expert}"
    shapes = {
        "down_proj": (hidden, width),
        "gate_proj": (width, hidden),
        "up_proj": (width, hidden),
    }
    tensors: dict[str, torch.Tensor] = {}
    for proj in EXPERT_PROJECTIONS:
        rows, cols = shapes[proj]
        tensors[f"{root}.{proj}.weight"] = torch.randint(
            0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8
        )
        tensors[f"{root}.{proj}.weight_scale"] = torch.full(
            (rows, -(-cols // 32)), 127, dtype=torch.uint8
        )
    return tensors


def write_tiny_checkpoint(root: str, **overrides) -> str:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as handle:
        json.dump({**TINY, **overrides}, handle)
    config = MimoV2Config.from_pretrained(root).text

    for index in range(4):
        tensors: dict[str, torch.Tensor] = {}
        if index == 0:
            tensors["model.embed_tokens.weight"] = torch.randn(
                config.vocab_size, config.hidden_size, generator=torch.Generator().manual_seed(3)
            ).to(torch.bfloat16)
            tensors["model.norm.weight"] = torch.ones(config.hidden_size)
            tensors["lm_head.weight"] = torch.randn(
                config.vocab_size, config.hidden_size, generator=torch.Generator().manual_seed(4)
            ).to(torch.bfloat16)
            for layer in range(config.num_hidden_layers):
                tensors.update(tiny_layer_tensors(layer, config))
        lo = index * EXPERTS_PER_SHARD
        for layer in config.moe_layer_indices:
            for expert in range(lo, lo + EXPERTS_PER_SHARD):
                tensors.update(tiny_expert_tensors(layer, expert, config))
        save_file(tensors, os.path.join(root, f"model_pp0_ep{index}_shard0.safetensors"))

    # A file a reader searching recursively would fold into the text model:
    # `dflash/` is a separate model with its own namespace, and this is a name the
    # text backbone wants. Flattening it in would silently shadow layer 0's qkv.
    nested = os.path.join(root, "dflash")
    os.makedirs(nested, exist_ok=True)
    save_file(
        {"model.layers.0.self_attn.qkv_proj.weight": torch.zeros(1, 1, dtype=torch.float8_e4m3fn)},
        os.path.join(nested, "model.safetensors"),
    )
    save_file(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                1, 1, dtype=torch.float8_e4m3fn
            )
        },
        os.path.join(root, "model_mtp.safetensors"),
    )
    return root


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> MimoV2Checkpoint:
    return MimoV2Checkpoint(write_tiny_checkpoint(str(tmp_path_factory.mktemp("mimo_tiny"))))


@pytest.fixture(scope="module")
def tiny_config() -> MimoV2Config:
    return MimoV2Config.from_dict(TINY)


# ---------------------------------------------------------------------------
# The miniature: headers into addresses
# ---------------------------------------------------------------------------


def test_a_release_without_an_index_is_read_from_its_own_headers(tiny):
    assert not os.path.exists(os.path.join(tiny.root, "model.safetensors.index.json"))
    assert tiny.mmap._shard_files() == ["model_mtp.safetensors"] + [
        f"model_pp0_ep{i}_shard0.safetensors" for i in range(4)
    ]


def test_a_subdirectory_is_not_searched_for_tensors(tiny):
    """`dflash/` names a tensor the backbone wants; flattening it in would collide."""
    assert tiny.entry("model.layers.0.self_attn.qkv_proj.weight").shape == (88, 64)


def test_every_backbone_tensor_the_config_names_is_in_the_file(tiny, tiny_config):
    keys = dense_weight_keys(tiny_config.text)
    assert all(key in tiny for key in keys)
    assert len(keys) == len(set(keys)), "the key list names a tensor twice"


def test_the_expert_map_is_contiguous_and_expert_e_lives_in_shard_e_over_two(tiny):
    layout = tiny.layout
    assert layout.experts_per_shard == EXPERTS_PER_SHARD
    assert layout.n_experts == 8
    assert layout.moe_layers == (1, 2, 3)
    assert [layout.shard_of(e) for e in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    for layer in layout.moe_layers:
        for expert in range(8):
            assert layout.run(layer, expert).file_name == (
                f"model_pp0_ep{expert // 2}_shard0.safetensors"
            )


def test_one_expert_is_six_tensors_in_the_released_order(tiny):
    run = tiny.layout.run(1, 5)
    expected = tuple(
        (proj, kind) for proj in EXPERT_PROJECTIONS for kind in ("weight", "weight_scale")
    )
    assert tuple((part[2], part[3]) for part in run.parts) == expected
    assert run.parts[0][2:4] == ("down_proj", "weight")
    # 32x32 and 32x2 for gate and up, 64x16 and 64x1 for down: 3264 bytes.
    assert run.nbytes == 3264


def test_the_expert_region_of_a_shard_is_the_experts_own_byte_range(tiny):
    """The `pread` a bank wants: exactly the shard's share of the layer, no more."""
    file_name, begin, end = tiny.expert_region(2, 3)
    assert file_name == "model_pp0_ep3_shard0.safetensors"
    header = raw_header(os.path.join(tiny.root, file_name))
    first = header["model.layers.2.mlp.experts.6.down_proj.weight"][2]
    last = header["model.layers.2.mlp.experts.7.up_proj.weight_scale"][3]
    assert (begin, end) == (first, last)
    assert end - begin == 2 * tiny.layout.expert_bytes


def test_a_shard_that_holds_the_wrong_experts_is_refused(tmp_path):
    """The `e // experts_per_shard` rule, broken on purpose.

    Shards 1 and 2 have their experts exchanged. Every expert is still somewhere a
    reader would look for *an* expert, which is why the rule has to be checked
    rather than assumed.
    """
    root = str(tmp_path / "misassigned")
    write_tiny_checkpoint(root)
    config = MimoV2Config.from_pretrained(root).text
    for shard, experts in ((1, (4, 5)), (2, (2, 3))):
        tensors = {}
        for layer in config.moe_layer_indices:
            for expert in experts:
                tensors.update(tiny_expert_tensors(layer, expert, config))
        save_file(tensors, os.path.join(root, f"model_pp0_ep{shard}_shard0.safetensors"))
    with pytest.raises(ValueError, match="expert e is expected in shard"):
        MimoV2Checkpoint(root)


def test_an_expert_stored_out_of_order_is_refused(tmp_path):
    """`down, gate, up` is the order the fp4 kernels index one expert by.

    One expert's six tensors are written `up, gate, down` instead. Every tensor is
    well formed and the byte count is unchanged: only the slots moved, which is a
    permutation a kernel would read happily and get wrong.
    """
    root = str(tmp_path / "reordered")
    write_tiny_checkpoint(root)

    def as_up_gate_down(items):
        """Layer 1's expert 0's six tensors, moved to the end as up, gate, down."""
        moved = [item for item in items if ".layers.1.mlp.experts.0." in item[0]]
        rest = [item for item in items if ".layers.1.mlp.experts.0." not in item[0]]
        grouped: dict[str, list] = {"down_proj": [], "gate_proj": [], "up_proj": []}
        for name, tensor in moved:
            grouped[name.split(".")[-2]].append((name, tensor))
        order = ("up_proj", "gate_proj", "down_proj")
        return rest + [item for proj in order for item in grouped[proj]]

    rewrite_shard(root, "model_pp0_ep0_shard0.safetensors", as_up_gate_down)
    with pytest.raises(ValueError, match=r"expected \(\('down_proj', 'weight'\)"):
        MimoV2Checkpoint(root)


def test_an_expert_interrupted_mid_run_is_refused(tmp_path):
    """The bank reads one expert as one run; a foreign tensor inside it breaks that."""
    root = str(tmp_path / "interrupted")
    write_tiny_checkpoint(root)

    def insert_a_tensor(items):
        out = []
        for name, tensor in items:
            out.append((name, tensor))
            if name == "model.layers.1.mlp.experts.0.down_proj.weight_scale":
                out.append(("model.layers.1.input_layernorm.weight", torch.ones(64)))
        return out

    rewrite_shard(root, "model_pp0_ep0_shard0.safetensors", insert_a_tensor)
    with pytest.raises(ValueError, match="not contiguous"):
        MimoV2Checkpoint(root)


def test_dense_tensor_dequantizes_fp8_and_passes_bf16_through(tiny):
    file_name = "model_pp0_ep0_shard0.safetensors"
    key = "model.layers.1.self_attn.qkv_proj.weight"
    entry = tiny.entry(key)
    codes = read_stored_tensor(os.path.join(tiny.root, file_name), (
        entry.dtype, entry.shape, entry.begin, entry.end
    ))
    got = tiny.dense_tensor(key, torch.float32)
    assert tuple(got.shape) == entry.shape
    # The stored byte is an E4M3 *value*, not an integer count: `w = w_fp8 * scale`,
    # and every tile scale in this shard is TILE_SCALE.
    bare = codes.view(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(got, bare * TILE_SCALE)
    assert not torch.allclose(got, bare), "the tile scale was not applied"

    o_proj = tiny.dense_tensor("model.layers.1.self_attn.o_proj.weight", torch.float32)
    assert o_proj.dtype == torch.float32
    assert tiny.entry("model.layers.1.self_attn.o_proj.weight").dtype == "BF16"


def test_a_global_qkv_scale_is_read_one_tile_row_per_shard(tmp_path):
    """The released 108-vs-106 anomaly, in miniature and made fatal.

    Every scale row of a global layer's fused projection is given its own value,
    so a reader that indexes the scale as one run of tiles over the whole weight
    scales each shard's rows by the first shard's tile and lands on a number that
    is none of the rows' own. A reader that indexes it per shard reproduces the
    rows exactly.
    """
    root = str(tmp_path / "per_shard_scale")
    write_tiny_checkpoint(root)
    config = MimoV2Config.from_dict(TINY).text
    rows = config.attention(0).qkv_out
    shard_rows = rows // QKV_SHARDS
    per_shard = torch.tensor([1.0, 2.0, 4.0, 8.0])

    def give_each_scale_row_its_own_value(items):
        out = []
        for name, tensor in items:
            if name == "model.layers.0.self_attn.qkv_proj.weight_scale_inv":
                tensor = per_shard.reshape(QKV_SHARDS, 1).expand(QKV_SHARDS, 1).clone()
            out.append((name, tensor))
        return out

    rewrite_shard(root, "model_pp0_ep0_shard0.safetensors", give_each_scale_row_its_own_value)
    checkpoint = MimoV2Checkpoint(root)
    assert checkpoint.entry(
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv"
    ).shape[0] == QKV_SHARDS

    codes = checkpoint.read("model.layers.0.self_attn.qkv_proj.weight", copy=False)
    scale = checkpoint.read("model.layers.0.self_attn.qkv_proj.weight_scale_inv", copy=False)
    bare = codes.view(torch.float8_e4m3fn).to(torch.float32)
    want = torch.cat(
        [bare[rank * shard_rows : (rank + 1) * shard_rows] * per_shard[rank]
         for rank in range(QKV_SHARDS)]
    )
    got = checkpoint.dense_tensor("model.layers.0.self_attn.qkv_proj.weight", torch.float32)
    assert torch.equal(got, want), "a shard was scaled by another shard's tile"
    assert not torch.equal(got, bare * per_shard[0]), "every shard got the first tile"


def test_a_mxfp4_expert_weight_is_refused_by_the_dense_reader(tiny):
    with pytest.raises(ValueError, match="packed MXFP4 expert weight"):
        tiny.dense_tensor("model.layers.1.mlp.experts.0.gate_proj.weight", torch.float32)


def test_an_absent_tensor_raises_rather_than_returning_nothing(tiny):
    with pytest.raises(KeyError, match="is not in the checkpoint"):
        tiny.entry("model.layers.0.mlp.gate_proj.weight.scale_of_a_tensor_that_never_was")


def test_a_layer_owns_exactly_the_tensors_its_config_says(tiny):
    for layer in range(4):
        keys = layer_weight_keys(layer, tiny.layer)
        assert all(key in tiny for key in keys), f"layer {layer} names a tensor the file lacks"
        assert (f"model.layers.{layer}.mlp.gate.weight" in keys) == (
            layer in tiny.layer.moe_layer_indices
        )
        assert (
            f"model.layers.{layer}.self_attn.attention_sink_bias" in keys
        ) == tiny.layer.attention(layer).has_sink


def test_a_real_expert_dequantizes_to_the_configured_shape(tiny):
    arrays = tiny.expert_arrays(2, 3)
    assert tuple(arrays[("gate_proj", "weight")].shape) == (32, 32)
    assert tuple(arrays[("gate_proj", "weight_scale")].shape) == (32, 2)
    assert tuple(arrays[("down_proj", "weight")].shape) == (64, 16)
    assert arrays[("up_proj", "weight")].dtype == torch.uint8, "the views must stay packed"

    dense = dequant_mxfp4(
        arrays[("gate_proj", "weight")], arrays[("gate_proj", "weight_scale")], 32
    )
    assert tuple(dense.shape) == (32, 64)
    # Every scale in the file is 127, so the expert *is* the codebook.
    assert set(dense.unique().tolist()) <= set(E2M1_LEVELS)
    assert dense.abs().max() == 6.0, "the largest E2M1 magnitude should occur by chance"


# ---------------------------------------------------------------------------
# The release itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    # Only the `needs_release` tests below request this, so a host without the
    # checkpoint never constructs it.
    return MimoV2Checkpoint(RELEASE)


@needs_release
def test_the_released_shards_are_65_files_and_the_index_names_them_all(release):
    files = release.mmap._shard_files()
    assert len(files) == 65
    assert files[0] == "model_mtp.safetensors"
    assert files[1] == "model_pp0_ep0_shard0.safetensors"
    assert files[-1] == "model_pp0_ep9_shard0.safetensors"


@needs_release
def test_the_released_index_agrees_with_the_headers(release):
    """The release ships an index; it is a name map and nothing more.

    It is checked here rather than trusted, because the loader's own reading of the
    layout comes from the headers: if the two ever disagreed, the index would be
    the thing that is wrong and this is where that shows up.
    """
    index_path = os.path.join(release.root, "model.safetensors.index.json")
    assert os.path.exists(index_path)
    with open(index_path) as handle:
        index = json.load(handle)
    assert index["metadata"]["save_format"] == "mxfp4"
    assert index["metadata"]["tp_size"] == 4
    weight_map = index["weight_map"]
    assert len(weight_map) == len(release.mmap)
    for key, file_name in weight_map.items():
        assert release.entry(key).file_name == file_name


@needs_release
def test_every_released_backbone_tensor_lives_in_ep0(release):
    names = {release.entry(key).file_name for key in dense_weight_keys(release.layer)}
    assert names == {"model_pp0_ep0_shard0.safetensors"}


@needs_release
def test_the_released_expert_map_is_four_experts_per_shard(release):
    layout = release.layout
    assert layout.experts_per_shard == 4
    assert layout.n_experts == 256
    assert layout.moe_layers == release.layer.moe_layer_indices
    assert layout.expert_bytes == 12 * 1048576 + 768 * 1024, "12.75 MiB per expert"
    assert layout.run(23, 200).file_name == "model_pp0_ep50_shard0.safetensors"


@needs_release
def test_the_released_fused_widths_differ_between_the_two_families(release):
    config = release.layer
    assert config.attention(0).qkv_out == 13568
    assert config.attention(1).qkv_out == 14848
    for layer in (0, 1):
        width = config.attention(layer).qkv_out
        assert release.entry(f"model.layers.{layer}.self_attn.qkv_proj.weight").shape == (
            width,
            4096,
        )
        assert release.entry(f"model.layers.{layer}.self_attn.o_proj.weight").shape == (4096, 8192)


@needs_release
def test_the_released_global_qkv_scale_is_blocked_one_shard_at_a_time(release):
    """106 tiles of weight, 108 rows of scale, and the extra rows are not spare.

    108 is 4 x 27 and a global layer's projection is four shards of 3392 rows,
    which is 26.5 tiles: the scale restarts at each shard boundary, so the last
    row of each shard's share of it covers a half tile. The sliding-window layers
    are 3712 rows in four shards of 29 whole tiles, so their scale is the same
    either way, which is why only the global layers move.
    """
    ga = release.entry("model.layers.0.self_attn.qkv_proj.weight")
    ga_scale = release.entry("model.layers.0.self_attn.qkv_proj.weight_scale_inv")
    assert ga.shape[0] // 128 == 106
    assert ga_scale.shape == (108, 32)
    assert ga_scale.shape[0] == QKV_SHARDS * -(-(ga.shape[0] // QKV_SHARDS) // 128)
    swa = release.entry("model.layers.1.self_attn.qkv_proj.weight")
    swa_scale = release.entry("model.layers.1.self_attn.qkv_proj.weight_scale_inv")
    assert swa.shape[0] // 128 == swa_scale.shape[0] == 116, "the windowed family is exact"
    assert swa_scale.shape[0] == QKV_SHARDS * -(-(swa.shape[0] // QKV_SHARDS) // 128)

    codes = release.read("model.layers.0.self_attn.qkv_proj.weight", copy=False)
    scale = release.read("model.layers.0.self_attn.qkv_proj.weight_scale_inv", copy=False)
    per_shard = dequant_fp8_block(codes, scale, out_dtype=torch.float32, shards=QKV_SHARDS)
    one_run = dequant_fp8_block(codes, scale, out_dtype=torch.float32)
    shard_rows = codes.shape[0] // QKV_SHARDS
    differs = (per_shard != one_run).any(dim=1).nonzero().flatten()
    assert differs.numel() and int(differs.min()) == shard_rows, (
        "the two blockings first disagree at the second shard's first row, where one "
        "reads that shard's own tile and the other reads the first shard's last one"
    )


@needs_release
def test_the_released_windowed_qkv_scale_reads_the_same_either_way(release):
    """The windowed layers are the control: their shards are whole tiles."""
    codes = release.read("model.layers.1.self_attn.qkv_proj.weight", copy=False)
    scale = release.read("model.layers.1.self_attn.qkv_proj.weight_scale_inv", copy=False)
    assert torch.equal(
        dequant_fp8_block(codes, scale, out_dtype=torch.float32, shards=QKV_SHARDS),
        dequant_fp8_block(codes, scale, out_dtype=torch.float32),
    )


@needs_release
def test_the_released_sink_is_on_the_windowed_family_only(release):
    config = release.layer
    for layer in (0, 1, 5, 47):
        key = f"model.layers.{layer}.self_attn.attention_sink_bias"
        shape = config.attention(layer)
        assert (key in release) == shape.has_sink
        if shape.has_sink:
            assert release.entry(key).shape == (shape.num_q_heads,)


@needs_release
def test_a_released_expert_dequantizes_to_the_configured_shapes(release):
    config = release.layer
    arrays = release.expert_arrays(1, 4)
    width, hidden = config.moe_intermediate_size, config.hidden_size
    assert tuple(arrays[("gate_proj", "weight")].shape) == (width, hidden // 2)
    assert tuple(arrays[("gate_proj", "weight_scale")].shape) == (width, hidden // 32)
    assert tuple(arrays[("down_proj", "weight")].shape) == (hidden, width // 2)
    gate = dequant_mxfp4(arrays[("gate_proj", "weight")], arrays[("gate_proj", "weight_scale")], 32)
    assert tuple(gate.shape) == (width, hidden)
    assert torch.isfinite(gate).all()
    assert 0.001 < float(gate.std()) < 0.1, "a misplaced nibble would move this a lot"


@needs_release
def test_nothing_in_the_dense_key_list_points_at_an_expert(release):
    for key in dense_weight_keys(release.layer):
        assert ".mlp.experts." not in key
