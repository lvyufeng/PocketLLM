"""Qwen4-Exp configuration parsing.

Reads the HF `config.json` shipped with Qwen3.8-Flash-Next without depending on
a transformers version that knows the architecture.  Only the fields the runtime
actually needs are kept; the vision tower is parsed but unused by the text path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """`count`-th prime strictly greater than `start` (matches upstream)."""
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass
class Qwen4ExpTextConfig:
    # Core transformer geometry
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    # Per-layer attention flavour: "linear_attention" or "qwen_sparse_attention".
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4

    # GatedDeltaNet (linear attention) layers
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    output_gate_type: str | None = "sigmoid"

    # Hyper-connections (4 parallel residual streams)
    hc_count: int = 4
    hc_lowrank: int = 320

    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    # QSA indexer on full-attention layers
    indexer_n_heads: int | None = 4
    indexer_kv_heads: int | None = 1
    indexer_head_dim: int | None = 128
    indexer_budget: int | None = 2048
    indexer_compress_ratio: int | None = 4

    # PLE hashed n-gram embeddings
    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    seed: int = 1234

    # RoPE
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True

    # Tokens
    bos_token_id: int = 248044
    eos_token_id: list[int] = field(default_factory=lambda: [248044])
    pad_token_id: int | None = None

    # MTP draft head (one extra layer); unused until spec decoding lands.
    mtp_num_hidden_layers: int = 1
    mtp_rope_theta: float = 1e7

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                "linear_attention" if (i + 1) % self.full_attention_interval else "qwen_sparse_attention"
                for i in range(self.num_hidden_layers)
            ]
        # The shipped checkpoint labels indexer layers "full_attention"; upstream
        # renames them so the QSA path is explicit.
        self.layer_types = [
            "qwen_sparse_attention" if t == "full_attention" else t for t in self.layer_types
        ]
        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]

    # -- derived geometry -------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def hc_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def primary_eos_token_id(self) -> int:
        return self.eos_token_id[0]

    def is_linear_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    def ple_layer_index(self, layer_idx: int) -> int | None:
        """Position of `layer_idx` in the PLE layer list, or None.

        `ple_layer_ids` is one-indexed in the checkpoint config.
        """
        one_indexed = layer_idx + 1
        if one_indexed in self.ple_layer_ids:
            return self.ple_layer_ids.index(one_indexed)
        return None

    def ngram_head_vocab_sizes(self, ple_layer_index: int) -> tuple[list[int], list[int], int]:
        """Per-head hashed vocab sizes/offsets for one PLE layer.

        Each head gets a distinct prime above `ngram_vocab_size_base`, indexed
        globally across PLE layers so tables never collide.  Returns
        (sizes, offsets, padded_total).
        """
        sizes: list[int] = []
        offsets: list[int] = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        divisor = self.make_ngram_vocab_size_divisible_by
        padded = math.ceil(total / divisor) * divisor
        return sizes, offsets, padded

    @classmethod
    def from_dict(cls, raw: dict) -> "Qwen4ExpTextConfig":
        rope = raw.get("rope_parameters") or {}
        mtp = raw.get("mtp") or {}
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["rope_theta"] = float(rope.get("rope_theta", raw.get("rope_theta", 1e7)))
        kwargs["partial_rotary_factor"] = float(
            rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 0.25))
        )
        if "mrope_section" in rope:
            kwargs["mrope_section"] = list(rope["mrope_section"])
        kwargs["mrope_interleaved"] = bool(rope.get("mrope_interleaved", True))
        if "rope_theta" in mtp:
            kwargs["mtp_rope_theta"] = float(mtp["rope_theta"])
        return cls(**kwargs)


@dataclass
class Qwen4ExpConfig:
    text_config: Qwen4ExpTextConfig
    architecture: str = "qwen4_exp"
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    tie_word_embeddings: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Qwen4ExpConfig":
        text_raw = raw.get("text_config", raw)
        return cls(
            text_config=Qwen4ExpTextConfig.from_dict(text_raw),
            architecture=(raw.get("architectures") or ["qwen4_exp"])[0],
            image_token_id=raw.get("image_token_id"),
            video_token_id=raw.get("video_token_id"),
            vision_start_token_id=raw.get("vision_start_token_id"),
            vision_end_token_id=raw.get("vision_end_token_id"),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            raw=raw,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Qwen4ExpConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))
