"""The assembled stack: what a token costs, and what it has to agree with.

`device_attention.py` and `device_experts.py` each answer a question about one layer.
This file is about the model: the embedding, forty-eight layers of two adds each, the
final norm and the head -- and the four things the assembly can get wrong while every
layer is right.

Three of them are the reference's own behaviour and are checked against it. The embedding
is not scaled by `sqrt(hidden)`. The head is not tied to it. A layer's residual is added in
the hidden dtype on both sides, and the routed sum is rounded once, where the kernel leaves
it in float32. The fourth is this stage's: the routed layers *share* one expert arena, so
the layer a draw belongs to is a property of the call, and a model that forgot to pass it
would fill the arena with the wrong layer's experts and be wrong in a way that looks like
a slow model rather than a broken one.

The synthetic half runs on the release's own config shape at miniature dimensions, with
every weight drawn from a seed and the experts stored as packed fp4 whose scales are
exactly one -- so the dense experts the host reads and the packed codes the kernel reads
are the same numbers, and a disagreement is a bug rather than a rounding. The router's gate
is built so that the draw is known in advance: with one-hot rows and a separating correction
bias the same two experts are chosen for every token, which is what lets the tests ask
*which* experts were staged instead of only how many.

The release half is the real thing: two layers of the 309B checkpoint -- layer 0, which is
global attention and a dense FFN, and layer 1, which is windowed attention and routed
experts -- against the host reference in float32, on a prompt fed one token at a time so the
cache is exercised. It goes through the mmap source and not the bank, because the bank is
149.81 GiB of this host's `/dev/shm` and a test that needs it is a test that only runs here.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.models.mimo_v2.config import MimoV2TextConfig  # noqa: E402
from src.models.mimo_v2.device_experts import MmapExpertSource  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.layers import (  # noqa: E402
    MimoV2HostModel,
    build_attention_masks,
    rms_norm,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.quant import dequant_mxfp4  # noqa: E402
from src.models.mimo_v2.weights import host_model_from_checkpoint  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the released two-layer comparison needs both the release and a CUDA device",
)

DEVICE = "cuda:0"

#: The miniature's dimensions. Small enough to run in a second, and legal for both paths:
#: `dim` and `inter` are multiples of 32 for the fp4 kernel, and the key and value sections
#: divide by four for the released fused-projection row order.
HIDDEN = 64
INTER = 32
VOCAB = 64
N_EXPERTS = 4
TOP_K = 2

#: What the routed kernel's int8 activation quantization costs at this scale. The measured
#: disagreement is 0.064 to 0.087 on logits whose peak is 4.6, which is *worse* than the
#: released checkpoint's percent because random experts at 32 wide produce outputs that are
#: not small against the stream they are added to -- `test_the_released_two_layers_...`
#: records the real number.
ROUTED_TOLERANCE = 0.15


def tiny_config(
    *,
    layers: int = 2,
    routed: tuple[int, ...] = (0, 1),
    window: int = 4,
    eos: int = 151645,
) -> MimoV2TextConfig:
    """The release's shape at miniature dimensions, with the layer pattern spelled out."""
    raw = {
        "model_type": "mimo_v2",
        "vocab_size": VOCAB,
        "eos_token_id": eos,
        "hidden_size": HIDDEN,
        "num_hidden_layers": layers,
        "hidden_act": "silu",
        "layernorm_epsilon": 1e-6,
        "tie_word_embeddings": False,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "v_head_dim": 8,
        "swa_num_attention_heads": 8,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 16,
        "swa_v_head_dim": 8,
        "swa_rope_theta": 10_000.0,
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.5,
        "attention_projection_layout": "fused_qkv",
        "attention_value_scale": 0.707,
        "add_swa_attention_sink_bias": True,
        "sliding_window": window,
        # 1 is a windowed layer and 0 is a global one, which is the release's encoding and
        # the opposite of the reading a first guess would take.
        "hybrid_layer_pattern": tuple(1 if i % 2 else 0 for i in range(layers)),
        "intermediate_size": INTER,
        "moe_intermediate_size": INTER,
        "n_routed_experts": N_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_layer_freq": tuple(routed),
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
    }
    return MimoV2TextConfig.from_dict(raw)


class SyntheticSource:
    """Packed experts that identify their own layer and id, so neither can be substituted.

    The scales are E8M0 127, which is exactly one, so a dequantized weight is the codebook
    value and the host's dense tensor and the kernel's codes are the same numbers.
    """

    def __init__(self, dim: int = HIDDEN, inter: int = INTER, n_experts: int = N_EXPERTS) -> None:
        self.dim = dim
        self.inter = inter
        self.n_experts = n_experts
        self.draws: list[tuple[int, int]] = []
        self._views: dict[tuple[int, int], dict[tuple[str, str], torch.Tensor]] = {}
        for layer in range(4):
            for expert in range(n_experts):
                self._views[(layer, expert)] = self._pack(layer, expert)

    def _pack(self, layer: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        generator = torch.Generator().manual_seed(9000 + 1000 * layer + expert)
        shapes = {
            ("gate_proj", "weight"): (self.inter, self.dim // 2),
            ("gate_proj", "weight_scale"): (self.inter, self.dim // 32),
            ("down_proj", "weight"): (self.dim, self.inter // 2),
            ("down_proj", "weight_scale"): (self.dim, self.inter // 32),
            ("up_proj", "weight"): (self.inter, self.dim // 2),
            ("up_proj", "weight_scale"): (self.inter, self.dim // 32),
        }
        out: dict[tuple[str, str], torch.Tensor] = {}
        for key, shape in shapes.items():
            if key[1] == "weight_scale":
                out[key] = torch.full(shape, 127, dtype=torch.uint8)
            else:
                codes = torch.randint(0, 256, shape, generator=generator, dtype=torch.int16)
                out[key] = codes.to(torch.uint8)
        return out

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        self.draws.append((int(layer_id), int(expert)))
        return self._views[(int(layer_id), int(expert))]

    def dense(self, layer: int, expert: int) -> dict[str, torch.Tensor]:
        """The same expert as the host reads it: the codes, dequantized, and nothing else."""
        views = self._views[(layer, expert)]
        return {
            proj: dequant_mxfp4(
                views[(proj, "weight")], views[(proj, "weight_scale")], 32, torch.float32
            )
            for proj in ("gate_proj", "up_proj", "down_proj")
        }


class SyntheticCheckpoint:
    """The little of `MimoV2Checkpoint` the device path reads: a config and named tensors."""

    def __init__(self, config: MimoV2TextConfig, tensors: dict[str, torch.Tensor]) -> None:
        self.layer = config
        self.tensors = dict(tensors)

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def dense_tensor(
        self,
        key: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if key not in self.tensors:
            raise KeyError(key)
        return self.tensors[key].to(dtype).to(device)


def synthetic_tensors(
    config: MimoV2TextConfig, source: SyntheticSource, *, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Every tensor one backbone reads, dense, drawn from a seed.

    The routed experts are stored as the *dequantized* form of the same packed codes the
    source hands the kernel, which is what makes the host comparison a comparison.
    """
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.2

    tensors = {
        "model.embed_tokens.weight": normal(config.vocab_size, config.hidden_size),
        "model.norm.weight": torch.ones(config.hidden_size),
        "lm_head.weight": normal(config.vocab_size, config.hidden_size),
    }
    for layer_idx in range(config.num_hidden_layers):
        shape = config.attention(layer_idx)
        root = f"model.layers.{layer_idx}"
        tensors[f"{root}.self_attn.qkv_proj.weight"] = normal(shape.qkv_out, config.hidden_size)
        tensors[f"{root}.self_attn.o_proj.weight"] = normal(
            config.hidden_size, shape.num_q_heads * shape.v_head_dim
        )
        tensors[f"{root}.input_layernorm.weight"] = torch.ones(config.hidden_size)
        tensors[f"{root}.post_attention_layernorm.weight"] = torch.ones(config.hidden_size)
        if shape.has_sink:
            tensors[f"{root}.self_attn.attention_sink_bias"] = torch.full(
                (shape.num_q_heads,), 2.0
            )

        if config.ffn_kind(layer_idx) == "moe":
            # One-hot rows and a separating bias: the top two are experts 0 and 1 whatever
            # the token is, so the draw is a constant the tests can name.
            gate = torch.zeros(config.n_routed_experts, config.hidden_size)
            for expert in range(config.n_routed_experts):
                gate[expert, expert] = 1.0
            tensors[f"{root}.mlp.gate.weight"] = gate
            tensors[f"{root}.mlp.gate.e_score_correction_bias"] = torch.tensor(
                [10.0, 5.0, -10.0, -10.0][: config.n_routed_experts]
            )
            for expert in range(config.n_routed_experts):
                for proj, tensor in source.dense(layer_idx, expert).items():
                    tensors[f"{root}.mlp.experts.{expert}.{proj}.weight"] = tensor
        else:
            tensors[f"{root}.mlp.gate_proj.weight"] = normal(
                config.intermediate_size, config.hidden_size
            )
            tensors[f"{root}.mlp.up_proj.weight"] = normal(
                config.intermediate_size, config.hidden_size
            )
            tensors[f"{root}.mlp.down_proj.weight"] = normal(
                config.hidden_size, config.intermediate_size
            )
    return tensors


class Fixture:
    """One synthetic model on the card and the same weights on the host, side by side."""

    def __init__(
        self,
        config: MimoV2TextConfig,
        *,
        source: SyntheticSource | None = None,
        layers: list[int] | None = None,
        dtype: torch.dtype = torch.float32,
        pin: bool | None = None,
    ) -> None:
        self.config = config
        self.source = source if source is not None else SyntheticSource()
        self.tensors = synthetic_tensors(config, self.source)
        self.checkpoint = SyntheticCheckpoint(config, self.tensors)
        self.host = MimoV2HostModel(config, self.tensors)
        self.model = MimoV2DeviceModel(
            self.checkpoint,
            device=DEVICE,
            dtype=dtype,
            layers=layers,
            expert_source=source,
            pin=pin,
        )

    def rows(self, ids: list[int], *, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """The device's logits row by row through the cache, and the host's in one pass."""
        cache = self.model.cache(len(ids) + 4)
        device = []
        for position, token in enumerate(ids):
            logits = self.model.forward(torch.tensor([token]), start_pos=position, cache=cache)
            device.append(logits[0].to(torch.float32).cpu())
        with torch.no_grad():
            host = self.host.forward(torch.tensor([ids]))[0].to(torch.float32).cpu()
        return torch.stack(device), host


def delta(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max())


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


@needs_cuda
def test_the_device_stack_adds_up_the_way_the_reference_does():
    """A dense two-layer stack, float32 on both sides: the adds, the norms, the head.

    Nothing here is quantized, so the only difference the two paths can have is a
    reassociation inside a matmul. Anything the assembly got wrong -- a missing residual,
    a norm on the wrong stream, an embedding scaled on the way in -- moves a row by more
    than that.
    """
    config = tiny_config(routed=(0, 0))
    fixture = Fixture(config)
    assert config.ffn_kind(0) == "dense" and config.ffn_kind(1) == "dense"

    device, host = fixture.rows([3, 17, 42, 8])
    assert delta(device, host) < 1e-5
    # And not the same row for every position, which would pass the comparison above only
    # if the cache were ignored on both sides.
    assert delta(device[0], device[-1]) > 1e-3
    assert [int(row.argmax()) for row in device] == [int(row.argmax()) for row in host]


@needs_cuda
def test_the_cache_reproduces_the_prefix_it_stands_for():
    """One pass over four tokens against four calls through the cache, on the device.

    The reference has no cache, so this is the property the device path is *for*: a token
    fed at position `p` must see what the same token saw in a single pass over the prefix.
    """
    config = tiny_config(routed=(0, 0))
    fixture = Fixture(config)
    ids = [11, 5, 61, 30]
    one_pass = fixture.model.forward(torch.tensor(ids), start_pos=0)
    cache = fixture.model.cache(16)
    stepwise = torch.cat(
        [
            fixture.model.forward(torch.tensor([token]), start_pos=position, cache=cache)
            for position, token in enumerate(ids)
        ]
    )
    # 1.7e-6 measured: the same numbers through matmuls of different shapes, which is
    # reassociation and not the cache.
    assert delta(stepwise.to(torch.float32), one_pass.to(torch.float32)) < 1e-5


@needs_cuda
def test_a_truncated_stack_says_that_it_is_truncated():
    config = tiny_config(routed=(0, 0))
    fixture = Fixture(config, layers=[0])
    assert fixture.model.is_complete is False
    assert len(fixture.model.layers) == 1
    assert len(fixture.model.cache(8)) == 1


@needs_cuda
def test_a_checkpoint_without_a_head_is_refused():
    """`tie_word_embeddings` is false, so an untied head is the only correct one."""
    config = tiny_config(routed=(0, 0))
    tensors = synthetic_tensors(config, SyntheticSource())
    del tensors["lm_head.weight"]
    with pytest.raises(ValueError, match="untied head"):
        MimoV2DeviceModel(SyntheticCheckpoint(config, tensors), device=DEVICE)


# ---------------------------------------------------------------------------
# The routed path
# ---------------------------------------------------------------------------


@needs_cuda
def test_a_routed_layer_draws_the_experts_the_router_chose():
    """The stage's own question: whose experts are in the arena."""
    config = tiny_config(routed=(0, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source, layers=[1])
    fixture.model.forward(torch.tensor([7]), start_pos=0, cache=fixture.model.cache(8))
    assert source.draws == [(1, 0), (1, 1)]


@needs_cuda
def test_two_routed_layers_share_one_arena_and_still_draw_their_own():
    """The arena is one object handed to every layer, so the layer has to travel with the call.

    A model that staged layer 1's experts while computing layer 2 would produce a number
    that is the wrong layer's and look merely imprecise. The source jitters its codes by
    layer, so the comparison against the host is what fails when the id does not travel.
    """
    config = tiny_config(layers=3, routed=(0, 1, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source)
    assert fixture.model.experts is not None
    assert all(layer.experts is fixture.model.experts for layer in fixture.model.layers if layer.kind == "moe")

    device, host = fixture.rows([7, 21])
    assert sorted({layer for layer, _ in source.draws}) == [1, 2]
    assert delta(device, host) < ROUTED_TOLERANCE


@needs_cuda
def test_the_routed_stack_agrees_with_the_reference_within_its_quantization():
    """The kernel quantizes activations to int8 a row at a time, so this is the tolerance it costs."""
    config = tiny_config(routed=(0, 1))
    source = SyntheticSource()
    fixture = Fixture(config, source=source, dtype=torch.float32)
    device, host = fixture.rows([9, 44, 2])
    assert delta(device, host) < ROUTED_TOLERANCE
    shared = set(device[-1].topk(5).indices.tolist()) & set(host[-1].topk(5).indices.tolist())
    assert len(shared) >= 3


@needs_cuda
def test_a_chunk_through_a_layer_without_a_band_is_refused_rather_than_squeezed_into_a_draw():
    """A chunk is a real path now, and the arena it needs is not the one a draw uses.

    The model does not refuse a chunk: `prefill` is what serves one, and `mlp` sends a multi-row
    call to `forward_chunk`. What refuses is the experts module, when it was built with an arena
    one token's draw wide -- and it says which construction would hold the chunk rather than
    answering it one row at a time, which is the behaviour this pins.
    """
    config = tiny_config(routed=(0, 1))
    fixture = Fixture(config, source=SyntheticSource(), layers=[1])
    with pytest.raises(ValueError, match="without a chunk band"):
        fixture.model.forward(torch.tensor([1, 2]), start_pos=0, cache=fixture.model.cache(8))


@needs_cuda
def test_a_routed_layer_with_no_expert_source_is_refused():
    config = tiny_config(routed=(0, 1))
    with pytest.raises(ValueError, match="no source was handed in"):
        MimoV2DeviceModel(SyntheticCheckpoint(config, synthetic_tensors(config, SyntheticSource())), device=DEVICE)


# ---------------------------------------------------------------------------
# Pinning, and the accounting
# ---------------------------------------------------------------------------


class PinningSource(SyntheticSource):
    """A source that can pin itself, and reports being asked."""

    def __init__(self) -> None:
        super().__init__()
        self.pins = 0

    def pin_if_enabled(self):
        self.pins += 1
        return None


@needs_cuda
def test_a_source_that_can_pin_itself_is_asked_once_and_can_be_told_not_to():
    config = tiny_config(routed=(0, 1))
    asked = PinningSource()
    Fixture(config, source=asked, layers=[1])
    assert asked.pins == 1

    refused = PinningSource()
    Fixture(config, source=refused, layers=[1], pin=False)
    assert refused.pins == 0

    with pytest.raises(ValueError, match="no `pin_if_enabled`"):
        Fixture(config, source=SyntheticSource(), layers=[1], pin=True)


@needs_cuda
def test_the_memory_accounting_covers_the_layers_the_arena_and_the_head():
    config = tiny_config(routed=(0, 1))
    fixture = Fixture(config, source=SyntheticSource())
    weights = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (fixture.model.embed_tokens, fixture.model.lm_head, fixture.model.norm)
    )
    layers = sum(layer.memory_bytes for layer in fixture.model.layers)
    assert fixture.model.memory_bytes == weights + layers + fixture.model.experts.arena_bytes
    # A layer's own bill excludes the arena, which is what makes it a per-layer number.
    assert sum(layer.memory_bytes for layer in fixture.model.layers) == layers


@needs_cuda
def test_a_step_returns_a_row_and_greedy_feeds_that_row_back():
    """The decode loop's one job, and the one way it fails quietly.

    A step has to return `[1, vocab]`, because the loop indexes the row with `[-1]`. If a
    step returned a squeezed `[vocab]`, that index would take the *last logit* of the row
    instead -- and the loop would then append `argmax` of a scalar, which is token zero, and
    keep appending it. Every token after the first would be the same plausible-looking
    number, the model would run at full speed, and the text would be `'!!!!!!'`.
    """
    config = tiny_config(routed=(0, 0))
    fixture = Fixture(config)
    prompt = [3, 17]

    cache = fixture.model.cache(16)
    logits = None
    for position, token in enumerate(prompt):
        logits = fixture.model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]

    manual = []
    for step in range(5):
        manual.append(int(logits.argmax()))
        logits = fixture.model.step(manual[-1], start_pos=len(prompt) + step, cache=cache)
        assert logits.shape == (1, config.vocab_size)
        logits = logits[-1]

    assert manual[1:] != [0] * 4
    assert fixture.model.greedy(prompt, max_tokens=5) == manual


@needs_cuda
def test_greedy_stops_at_the_end_of_turn():
    """A finished turn fed back in continues the template rather than the answer.

    The head is replaced by one that can only answer token 0 and the config is told that 0
    ends the turn, so the loop has to stop on the token it just drew instead of asking the
    model what comes after its own end-of-turn marker.
    """
    config = tiny_config(routed=(0, 0), eos=0)
    fixture = Fixture(config)
    assert config.eos_token_ids == (0,)
    with torch.no_grad():
        # A head with no rows at all: every logit is zero, `argmax` is the lowest tied index,
        # and the draw is token 0 whatever the stream is.
        fixture.model.lm_head.zero_()
    assert fixture.model.greedy([3, 17], max_tokens=5) == [0]


@needs_cuda
def test_a_layer_reports_a_draw_price_of_one_expert():
    """`expert_bytes` is what one draw of one expert costs the link, and it is 12.75 MiB on the release."""
    config = tiny_config(routed=(0, 1))
    fixture = Fixture(config, source=SyntheticSource(), layers=[1])
    experts = fixture.model.experts
    assert experts.expert_bytes == sum(
        shape[0] * shape[1] for shape in experts._shapes().values()
    )
    assert experts.arena_bytes == experts.slots * experts.arena_rows * experts.expert_bytes


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release_layers() -> list[int]:
    return [0, 1]


@needs_release_cuda
def test_the_released_two_layers_agree_with_the_host_reference(release_layers):
    """Layer 0 is global attention and a dense FFN; layer 1 is windowed and routed.

    The device runs bfloat16 and the host runs float32, and the routed kernel quantizes its
    activations to int8, so the two do not agree exactly and are not meant to. What is
    compared is the layer stack's output *before* the final norm -- 2.8e-3 to 7.9e-3 against
    a peak of 0.55, measured -- and then the logits, whose disagreement is around a percent
    of their peak because the head averages the difference over 4096 columns instead of
    keeping it.

    The normed stream is deliberately not the comparison. Normalising divides by the row's
    RMS, and a residual stream whose peak is 68 times its RMS has most of its entries below
    bfloat16's resolution next to the largest one -- so the normed comparison measures the
    rounding of the small components and reports 0.35, which is a fact about bfloat16 and not
    about the port.

    The prompt is fed one token at a time through the cache and the host sees all four in one
    pass, so a cache that dropped or duplicated a key would show up here as well.
    """
    checkpoint = MimoV2Checkpoint(RELEASE)
    config = checkpoint.layer
    host = host_model_from_checkpoint(
        checkpoint, dtype=torch.float32, layers=release_layers, device="cpu"
    )
    model = MimoV2DeviceModel(
        checkpoint,
        device=DEVICE,
        layers=release_layers,
        expert_source=MmapExpertSource(checkpoint),
        pin=False,
    )
    assert model.is_complete is False

    ids = [1024, 2048, 4096, 8192]
    cache = model.cache(16)
    stream = []
    logits = []
    for position, token in enumerate(ids):
        row = model.forward(
            torch.tensor([token]), start_pos=position, cache=cache, final_norm=False
        )
        stream.append(row[0].to(torch.float32).cpu())
        logits.append(
            F.linear(
                rms_norm(row, model.norm, config.layernorm_epsilon).float(), model.lm_head.float()
            )[0].cpu()
        )

    # The host's own layer loop, because `forward` puts the norm and the head on the end and
    # the stream in between is what the device's `final_norm=False` is comparable to.
    positions = torch.arange(len(ids)).unsqueeze(0)
    masks = build_attention_masks(len(ids), config.resolved_window)
    tables = host.rope_tables(positions)
    with torch.no_grad():
        reference = F.embedding(torch.tensor([ids]), host.embed_tokens)
        for layer in host.layers:
            reference = layer(
                reference,
                attention_mask=masks[
                    "sliding_window_attention" if layer.shape.family == "swa" else "full_attention"
                ],
                position_embeddings=tables[layer.shape.family],
                position_ids=positions,
            )
    reference_logits = F.linear(
        rms_norm(reference, host.norm, config.layernorm_epsilon), host.lm_head
    )

    # The host carries a batch axis and the device path does not, so both are indexed to
    # `[sequence, ...]` before they are compared.
    assert delta(torch.stack(stream), reference[0]) < 5e-2
    assert delta(torch.stack(logits), reference_logits[0]) < 1.0
    assert [int(row.argmax()) for row in logits] == [
        int(row.argmax()) for row in reference_logits[0]
    ]
