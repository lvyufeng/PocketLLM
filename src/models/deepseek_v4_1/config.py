"""The DeepSeek-V4.1-Flash configuration, in both shapes the release ships it in.

The released checkpoint carries the same model described twice, in two different
layouts, and anything that reads a V4.1 config has to handle both:

* ``config.json`` -- the Transformers layout. The text hyper-parameters are
  nested under ``text_config``, the vision tower under ``vision_config`` and the
  weight quantization under ``quantization_config``. Its top-level ``dtype`` is
  *not* the quantization dtype: it is the dtype the unquantized tensors are
  stored in (BF16 -- the header audit confirms the norms, the gate and the
  embedding), while the dense linears are FP8 and the routed experts FP4.
* ``inference/config.json`` -- the reference runtime's layout. One flat level,
  shorter names (``dim`` for ``hidden_size``, ``n_layers`` for
  ``num_hidden_layers``, ``engram_pad_id`` for ``engram_pad_token_id``), and the
  YaRN parameters hoisted out of the ``rope_scaling`` block. It is *not* a
  superset: `topk_method`, `norm_topk_prob` and the token ids appear only in the
  Transformers file.

`load_config` accepts either shape and returns one `V41Config`, whose text half
is named after the reference runtime's ``ModelArgs`` fields -- the names a V4.1
implementation in this repository would use -- so the aliasing is paid once here
rather than at every call site.

Two things are deliberately *not* done here. The Engram row-count derivation
lives in `src/encoding/engram.py`, which owns the primes and the hash, and is
cross-checked against this schema by the tests rather than called from it. And
the tensor inventory is not this module's business: `scripts/audit_dsv41_headers.py`
maps names to tensors, and it reads its config through `from_dict` so that it no
longer depends on which shape it was handed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

__all__ = [
    "V41Config",
    "V41TextConfig",
    "V41VisionConfig",
    "from_dict",
    "load_config",
]

# The nested blocks of the Transformers layout, and the flat file that carries
# the same text and vision fields at the top level instead.
_HF_TEXT_BLOCK = "text_config"
_HF_VISION_BLOCK = "vision_config"
_HF_QUANT_BLOCK = "quantization_config"
_FLAT_INFERENCE_CONFIG = "inference/config.json"

# Where each canonical field comes from, as (path in config.json, path in
# inference/config.json). A `None` means that file does not carry the field --
# the flat one is missing the handful of facts the Transformers one has, all of
# which are listed at the bottom of the table. Reproducing the whole table here
# rather than deriving it keeps the mapping greppable and makes a rename a
# visible diff instead of a silent fallback.
#
# The two halves get a table each because they share field names -- `dim` and
# `n_layers` mean the text backbone here and the vision tower there -- and the
# flat file prefixes only the vision ones (`vision_dim`), so a single table
# would either collide or need the prefix threaded through the dataclasses.
_TEXT_ALIASES: dict[str, tuple[str | None, str | None]] = {
    # identity
    "model_type": ("model_type", None),
    "architectures": ("architectures", None),
    "bos_token_id": ("bos_token_id", None),
    "eos_token_id": ("eos_token_id", None),
    "pad_token_id": ("pad_token_id", None),
    "param_dtype": ("dtype", None),
    # tokenizer and quantization
    "vocab_size": ("text_config.vocab_size", "vocab_size"),
    "image_token_id": ("image_token_id", "image_token_id"),
    "dtype": ("quantization_config.quant_method", "dtype"),
    "expert_dtype": ("quantization_config.expert_dtype", "expert_dtype"),
    # Facts the Transformers file states and the flat one leaves implicit.
    "hidden_act": ("text_config.hidden_act", None),
    "max_position_embeddings": ("text_config.max_position_embeddings", None),
    "num_key_value_heads": ("text_config.num_key_value_heads", None),
    "tie_word_embeddings": ("text_config.tie_word_embeddings", None),
    # backbone shape
    "dim": ("text_config.hidden_size", "dim"),
    "moe_inter_dim": ("text_config.moe_intermediate_size", "moe_inter_dim"),
    "n_layers": ("text_config.num_hidden_layers", "n_layers"),
    "n_mtp_layers": ("text_config.num_nextn_predict_layers", "n_mtp_layers"),
    "n_heads": ("text_config.num_attention_heads", "n_heads"),
    # moe
    "n_routed_experts": ("text_config.n_routed_experts", "n_routed_experts"),
    "n_shared_experts": ("text_config.n_shared_experts", "n_shared_experts"),
    "n_activated_experts": ("text_config.num_experts_per_tok", "n_activated_experts"),
    "score_func": ("text_config.scoring_func", "score_func"),
    "topk_method": ("text_config.topk_method", None),
    "norm_topk_prob": ("text_config.norm_topk_prob", None),
    "route_scale": ("text_config.routed_scaling_factor", "route_scale"),
    "swiglu_limit": ("text_config.swiglu_limit", "swiglu_limit"),
    # attention
    "q_lora_rank": ("text_config.q_lora_rank", "q_lora_rank"),
    "head_dim": ("text_config.head_dim", "head_dim"),
    "rope_head_dim": ("text_config.qk_rope_head_dim", "rope_head_dim"),
    "norm_eps": ("text_config.rms_norm_eps", "norm_eps"),
    "o_groups": ("text_config.o_groups", "o_groups"),
    "o_lora_rank": ("text_config.o_lora_rank", "o_lora_rank"),
    # sparse attention
    "window_size": ("text_config.sliding_window", "window_size"),
    "compress_ratios": ("text_config.compress_ratios", "compress_ratios"),
    "kv_source_layers": ("text_config.kv_source_layer_ids", "kv_source_layers"),
    "index_source_layers": ("text_config.index_source_layer_ids", "index_source_layers"),
    "compress_rope_theta": ("text_config.compress_rope_theta", "compress_rope_theta"),
    # rope: the flat file hoists the four YaRN numbers out of `rope_scaling`
    "original_seq_len": ("text_config.rope_scaling.original_max_position_embeddings", "original_seq_len"),
    "rope_theta": ("text_config.rope_theta", "rope_theta"),
    "rope_factor": ("text_config.rope_scaling.factor", "rope_factor"),
    "beta_fast": ("text_config.rope_scaling.beta_fast", "beta_fast"),
    "beta_slow": ("text_config.rope_scaling.beta_slow", "beta_slow"),
    # indexer
    "index_n_heads": ("text_config.index_n_heads", "index_n_heads"),
    "index_head_dim": ("text_config.index_head_dim", "index_head_dim"),
    "index_topk": ("text_config.index_topk", "index_topk"),
    # candidate pre-filtering
    "candidate_source_layer": ("text_config.candidate_source_layer_id", "candidate_source_layer"),
    "candidate_topk_blocks": ("text_config.candidate_topk_blocks", "candidate_topk_blocks"),
    "candidate_block_size": ("text_config.candidate_block_size", "candidate_block_size"),
    # hyper-connections
    "hc_mult": ("text_config.hc_mult", "hc_mult"),
    "hc_sinkhorn_iters": ("text_config.hc_sinkhorn_iters", "hc_sinkhorn_iters"),
    "hc_eps": ("text_config.hc_eps", "hc_eps"),
    # engram
    "engram_layer_ids": ("text_config.engram_layer_ids", "engram_layer_ids"),
    "engram_num_embeddings": ("text_config.engram_num_embeddings", "engram_num_embeddings"),
    "engram_max_ngram_size": ("text_config.engram_max_ngram_size", "engram_max_ngram_size"),
    "engram_vocab_size": ("text_config.engram_vocab_size", "engram_vocab_size"),
    "engram_n_heads": ("text_config.engram_n_heads", "engram_n_heads"),
    "engram_head_dim": ("text_config.engram_head_dim", "engram_head_dim"),
    "engram_pad_id": ("text_config.engram_pad_token_id", "engram_pad_id"),
    "engram_compressed_vocab_size": (
        "text_config.engram_compressed_vocab_size",
        "engram_compressed_vocab_size",
    ),
    # dspark draft head
    "dspark_block_size": ("text_config.dspark_block_size", "dspark_block_size"),
    "dspark_noise_token_id": ("text_config.dspark_noise_token_id", "dspark_noise_token_id"),
    "dspark_target_layer_ids": ("text_config.dspark_target_layer_ids", "dspark_target_layer_ids"),
    "dspark_markov_rank": ("text_config.dspark_markov_rank", "dspark_markov_rank"),
    "dspark_n_routed_experts": ("text_config.dspark_n_routed_experts", "dspark_n_routed_experts"),
    "dspark_n_activated_experts": (
        "text_config.dspark_num_experts_per_tok",
        "dspark_n_activated_experts",
    ),
}

# The vision tower. The flat file carries these at the top level with a `vision_`
# prefix; the Transformers file nests them. Its `model_type` is the only key
# there with no counterpart, and it is not read.
_VISION_ALIASES: dict[str, tuple[str | None, str | None]] = {
    "n_layers": ("vision_config.num_hidden_layers", "vision_n_layers"),
    "dim": ("vision_config.hidden_size", "vision_dim"),
    "n_heads": ("vision_config.num_attention_heads", "vision_n_heads"),
    "inter_dim": ("vision_config.intermediate_size", "vision_inter_dim"),
    "patch_size": ("vision_config.patch_size", "vision_patch_size"),
    "rope_theta": ("vision_config.rope_theta", "vision_rope_theta"),
    "downsample_ratio": ("vision_config.downsample_ratio", "vision_downsample_ratio"),
    "max_n_token": ("vision_config.max_image_tokens", "vision_max_n_token"),
    "min_pixels": ("vision_config.min_pixels", "vision_min_pixels"),
    "max_wh_ratio": ("vision_config.max_wh_ratio", "vision_max_wh_ratio"),
}

# The canonical field -> alias table for each half, in the order `_build` needs.
_ALIASES: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "text": _TEXT_ALIASES,
    "vision": _VISION_ALIASES,
}

# Fields whose value is a JSON list and becomes a tuple, so the dataclasses stay
# hashable and a caller cannot mutate one through the frozen wrapper.
_SEQUENCE_FIELDS = frozenset(
    {
        "architectures",
        "compress_ratios",
        "kv_source_layers",
        "index_source_layers",
        "engram_layer_ids",
        "engram_num_embeddings",
        "dspark_target_layer_ids",
    }
)

# The three literal-valued fields the reference runtime types as `Literal`. Both
# shipped files agree on all three; the point of checking is that a repacked
# config cannot quietly claim a dtype this repository has no kernel for.
_LITERAL_CHOICES = {
    "dtype": ("bf16", "fp8"),
    "expert_dtype": ("fp4",),
    "score_func": ("softmax", "sigmoid", "sqrtsoftplus"),
    "hidden_act": ("silu",),
}


def _lookup(source: Mapping[str, Any], path: str | None) -> Any:
    """Read a dotted path, or return None if any step of it is absent."""
    if path is None:
        return None
    node: Any = source
    for step in path.split("."):
        if not isinstance(node, Mapping) or step not in node:
            return None
        node = node[step]
    return node


def _is_unstated(value: Any) -> bool:
    """Whether a field is absent from the file it was read from.

    A null and an empty list are the same claim here -- this file does not say --
    because the tuple-typed fields have to default to something iterable. The two
    released files only ever express it as null.
    """
    return value is None or value == ()


@dataclass(frozen=True)
class V41TextConfig:
    """The text backbone, named after the reference runtime's ``ModelArgs``.

    Every field is optional because the two shipped files do not carry the same
    set: ``topk_method``, ``norm_topk_prob`` and the token ids are Transformers-only.
    A ``None`` here therefore means "this file does not say", not "the model does
    not have one".
    """

    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    param_dtype: str | None = None

    vocab_size: int | None = None
    image_token_id: int | None = None
    dtype: str | None = None
    expert_dtype: str | None = None

    hidden_act: str | None = None
    max_position_embeddings: int | None = None
    num_key_value_heads: int | None = None
    tie_word_embeddings: bool | None = None

    dim: int | None = None
    moe_inter_dim: int | None = None
    n_layers: int | None = None
    n_mtp_layers: int | None = None
    n_heads: int | None = None

    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    n_activated_experts: int | None = None
    score_func: str | None = None
    topk_method: str | None = None
    norm_topk_prob: bool | None = None
    route_scale: float | None = None
    swiglu_limit: float | None = None

    q_lora_rank: int | None = None
    head_dim: int | None = None
    rope_head_dim: int | None = None
    norm_eps: float | None = None
    o_groups: int | None = None
    o_lora_rank: int | None = None

    window_size: int | None = None
    compress_ratios: tuple[int, ...] = ()
    kv_source_layers: tuple[int, ...] = ()
    index_source_layers: tuple[int, ...] = ()
    compress_rope_theta: float | None = None

    original_seq_len: int | None = None
    rope_theta: float | None = None
    rope_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None

    index_n_heads: int | None = None
    index_head_dim: int | None = None
    index_topk: int | None = None

    candidate_source_layer: int | None = None
    candidate_topk_blocks: int | None = None
    candidate_block_size: int | None = None

    hc_mult: int | None = None
    hc_sinkhorn_iters: int | None = None
    hc_eps: float | None = None

    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int | None = None
    engram_vocab_size: int | None = None
    engram_n_heads: int | None = None
    engram_head_dim: int | None = None
    engram_pad_id: int | None = None
    engram_compressed_vocab_size: int | None = None

    dspark_block_size: int | None = None
    dspark_noise_token_id: int | None = None
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int | None = None
    dspark_n_routed_experts: int | None = None
    dspark_n_activated_experts: int | None = None

    @property
    def total_layers(self) -> int | None:
        """Backbone plus draft layers: `compress_ratios` is indexed by layer id."""
        if self.n_layers is None or self.n_mtp_layers is None:
            return None
        return self.n_layers + self.n_mtp_layers

    @property
    def compress_ratio_by_layer(self) -> dict[int, int]:
        """Per-layer ``r``, or an empty map when the file does not carry the list."""
        return {layer: ratio for layer, ratio in enumerate(self.compress_ratios)}

    def verify(self) -> list[str]:
        """Self-consistency problems, as strings; an empty list means none.

        These are the identities the released files satisfy and a repacked or
        hand-edited config could break. Each one is grounded in the reference
        runtime rather than in a guess: `Compressor` and `Indexer` read
        ``compress_ratios[layer_id]`` and test membership in the source lists,
        so a compressor at a layer with no ratio, or an index source with
        nothing to index, is a config the reference cannot run.
        """
        problems: list[str] = []

        def want(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        if self.model_type and self.architectures:
            want(
                any("V41" in name for name in self.architectures),
                f"architectures {list(self.architectures)} names no V4.1 class",
            )
            want(
                any("V41" in name for name in self.architectures),
                f"architectures {list(self.architectures)} names no V4.1 class",
            )
        if self.dtype is not None and self.expert_dtype is not None:
            want(
                self.dtype in ("bf16", "fp8"),
                f"the routed experts are {self.expert_dtype} but the dense linears are {self.dtype}",
            )

        for name, choices in _LITERAL_CHOICES.items():
            value = getattr(self, name)
            if value is None:
                # Only `hidden_act` may be absent -- the flat file omits it. The
                # dtype fields are checked for presence just below.
                want(name not in ("dtype", "expert_dtype", "score_func"), f"{name} is absent")
                continue
            want(value in choices, f"{name} is {value!r}, not one of {list(choices)}")

        for name in (
            "vocab_size",
            "dim",
            "moe_inter_dim",
            "n_layers",
            "n_heads",
            "n_routed_experts",
            "q_lora_rank",
            "head_dim",
            "rope_head_dim",
            "o_groups",
            "o_lora_rank",
            "window_size",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "hc_mult",
            "hc_sinkhorn_iters",
        ):
            value = getattr(self, name)
            want(value is not None and value > 0, f"{name} is {value}, which is not positive")

        want(self.n_mtp_layers is not None and self.n_mtp_layers >= 0, f"n_mtp_layers is {self.n_mtp_layers}")
        want(
            self.n_activated_experts is not None
            and self.n_routed_experts is not None
            and 0 < self.n_activated_experts <= self.n_routed_experts,
            f"n_activated_experts {self.n_activated_experts} of n_routed_experts {self.n_routed_experts}",
        )
        want(
            self.n_shared_experts is not None and self.n_shared_experts >= 0,
            f"n_shared_experts is {self.n_shared_experts}",
        )
        want(self.route_scale is not None and self.route_scale > 0, f"route_scale is {self.route_scale}")
        want(self.swiglu_limit is not None and self.swiglu_limit >= 0, f"swiglu_limit is {self.swiglu_limit}")
        want(self.norm_eps is not None and self.norm_eps > 0, f"norm_eps is {self.norm_eps}")
        want(self.hc_eps is not None and self.hc_eps > 0, f"hc_eps is {self.hc_eps}")

        # Attention shape identities, all checkable from the config alone. The
        # header audit checks the tensors against the same identities; this is
        # the half that catches a broken config before a checkpoint is opened.
        if self.rope_head_dim is not None and self.head_dim is not None:
            want(
                self.rope_head_dim <= self.head_dim,
                f"rope_head_dim {self.rope_head_dim} exceeds head_dim {self.head_dim}",
            )
        if self.o_lora_rank is not None and self.o_groups is not None:
            want(
                self.o_groups > 0 and self.o_lora_rank % self.o_groups == 0,
                f"o_lora_rank {self.o_lora_rank} does not divide into o_groups {self.o_groups}",
            )

        # compress_ratios is indexed by layer id and covers the draft layers too.
        total = self.total_layers
        if total is not None:
            want(
                len(self.compress_ratios) == total,
                f"compress_ratios has {len(self.compress_ratios)} entries, not n_layers + n_mtp_layers = {total}",
            )
        ratios = list(self.compress_ratios)

        def check_source_list(name: str, layers: Sequence[int]) -> None:
            if list(layers) != sorted(set(layers)):
                problems.append(f"{name} {list(layers)} is not strictly ascending and unique")
            for layer in layers:
                if total is not None and layer >= total:
                    problems.append(f"{name} contains layer {layer}, outside the layer range")
                elif layer < 0 or layer >= len(ratios):
                    problems.append(f"{name} contains layer {layer}, outside compress_ratios")
                elif ratios[layer] <= 0:
                    problems.append(
                        f"{name} contains layer {layer}, whose compress_ratio is {ratios[layer]}"
                    )

        check_source_list("kv_source_layers", self.kv_source_layers)
        check_source_list("index_source_layers", self.index_source_layers)
        want(
            any(ratio > 0 for ratio in ratios),
            "compress_ratios is all zero, so no layer compresses and the sources are dead",
        )

        # The sharing model, read off `SharedAttentionRuntime`: layers run in
        # order and every source writes before its consumers read, so one slot
        # each is enough. A layer with a non-zero ratio attends over the shared
        # compressed cache, which means some kv source must already have written
        # it, and that source's ratio has to be the consumer's own: `_compress_kv`
        # divides the position by *its* ratio, while the cache it reads is sized
        # and written by the source's. Same argument for `topk_idxs`, which only
        # an index source produces and the layers in between reuse.
        def first_source_at_or_before(layers: Sequence[int], ratio: int, layer: int) -> bool:
            return any(other <= layer and ratios[other] == ratio for other in layers)

        for layer, ratio in enumerate(ratios):
            if ratio <= 0:
                continue
            if not first_source_at_or_before(self.kv_source_layers, ratio, layer):
                problems.append(
                    f"layer {layer} has compress_ratio {ratio} but no kv source at or before it "
                    f"with the same ratio, so the compressed cache it reads is never written"
                )
            if not first_source_at_or_before(self.index_source_layers, ratio, layer):
                problems.append(
                    f"layer {layer} has compress_ratio {ratio} but no index source at or before it "
                    f"with the same ratio, so it has no topk_idxs to attend over"
                )

        # Candidate pre-filtering is off with a negative source layer, and the
        # reference then ignores the other two; otherwise all three must be usable.
        # `select_candidate_blocks` runs inside `Indexer.forward`, so the source
        # must be an index source, and `uses_candidates` starts strictly after it.
        candidate = self.candidate_source_layer
        if candidate is not None and candidate >= 0:
            want(
                candidate in self.index_source_layers,
                f"candidate_source_layer {candidate} is not an index source, so nothing writes the candidates",
            )
            want(
                candidate < self.n_layers if self.n_layers is not None else True,
                f"candidate_source_layer {candidate} is outside the backbone",
            )
            want(
                self.candidate_topk_blocks is not None and self.candidate_topk_blocks > 0,
                "candidate pre-filtering is on but candidate_topk_blocks is not positive",
            )
            want(
                self.candidate_block_size is not None and self.candidate_block_size > 0,
                "candidate pre-filtering is on but candidate_block_size is not positive",
            )

        # Engram: one row count per table, on distinct layers, with a sentinel
        # row count that has to be above the vocabulary the buckets search from.
        want(
            len(self.engram_layer_ids) == len(self.engram_num_embeddings),
            f"engram_layer_ids {list(self.engram_layer_ids)} and engram_num_embeddings "
            f"{list(self.engram_num_embeddings)} are different lengths",
        )
        if list(self.engram_layer_ids) != sorted(set(self.engram_layer_ids)):
            problems.append(f"engram_layer_ids {list(self.engram_layer_ids)} is not strictly ascending")
        if total is not None:
            for layer in self.engram_layer_ids:
                want(0 <= layer < total, f"engram_layer_ids contains layer {layer}, outside the layer range")
        if self.engram_layer_ids:
            for name in ("engram_max_ngram_size", "engram_vocab_size", "engram_n_heads", "engram_head_dim"):
                value = getattr(self, name)
                want(value is not None and value > 0, f"Engram is enabled but {name} is {value}")
            want(
                self.engram_pad_id is not None and self.engram_pad_id >= 0,
                f"Engram is enabled but engram_pad_id is {self.engram_pad_id}",
            )
            want(
                self.engram_max_ngram_size is not None and self.engram_max_ngram_size >= 2,
                f"engram_max_ngram_size is {self.engram_max_ngram_size}, so no n-gram past the unigram exists",
            )

        # DSpark: the draft head projects the target layers back to `dim`, so it
        # needs at least one, and the reference asserts that at construction.
        if self.n_mtp_layers:
            want(bool(self.dspark_target_layer_ids), "n_mtp_layers is non-zero but dspark_target_layer_ids is empty")
        if list(self.dspark_target_layer_ids) != sorted(set(self.dspark_target_layer_ids)):
            problems.append(f"dspark_target_layer_ids {list(self.dspark_target_layer_ids)} is not ascending")
        if self.n_layers is not None:
            for layer in self.dspark_target_layer_ids:
                want(
                    0 <= layer < self.n_layers,
                    f"dspark_target_layer_ids contains layer {layer}, outside the backbone",
                )
        if self.dspark_block_size is not None:
            want(self.dspark_block_size > 0, f"dspark_block_size is {self.dspark_block_size}")
        if self.dspark_markov_rank is not None:
            want(self.dspark_markov_rank > 0, f"dspark_markov_rank is {self.dspark_markov_rank}")

        # Every id that indexes an embedding table has to be inside it.
        for name in ("image_token_id", "dspark_noise_token_id", "engram_pad_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if value is not None and self.vocab_size is not None:
                want(0 <= value < self.vocab_size, f"{name} {value} is outside vocab_size {self.vocab_size}")

        if self.pad_token_id is not None and self.engram_pad_id is not None:
            want(
                self.pad_token_id == self.engram_pad_id,
                f"pad_token_id {self.pad_token_id} and engram_pad_id {self.engram_pad_id} disagree; "
                "the Engram pad must be the tokenizer's pad",
            )

        return problems


@dataclass(frozen=True)
class V41VisionConfig:
    """The bundled ViT tower. ``n_layers == 0`` disables the vision path."""

    n_layers: int | None = None
    dim: int | None = None
    n_heads: int | None = None
    inter_dim: int | None = None
    patch_size: int | None = None
    rope_theta: float | None = None
    downsample_ratio: int | None = None
    max_n_token: int | None = None
    min_pixels: int | None = None
    max_wh_ratio: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.n_layers)

    def verify(self) -> list[str]:
        problems: list[str] = []
        if not self.enabled:
            return problems
        for name in ("dim", "n_heads", "inter_dim", "patch_size", "downsample_ratio", "max_n_token", "min_pixels"):
            value = getattr(self, name)
            if value is None or value <= 0:
                problems.append(f"vision is enabled but {name} is {value}")
        if self.dim and self.n_heads and self.dim % self.n_heads:
            problems.append(f"vision_dim {self.dim} does not divide by vision_n_heads {self.n_heads}")
        if self.patch_size and self.patch_size % 2:
            problems.append(f"vision_patch_size {self.patch_size} is odd, so the tower cannot downsample by half")
        return problems


@dataclass(frozen=True)
class V41Config:
    """One DeepSeek-V4.1-Flash config, whichever shape it was read from."""

    text: V41TextConfig
    vision: V41VisionConfig
    # Which file this came from, and whether it was the nested one. Recorded
    # because the two are not interchangeable: see the module docstring.
    source: str | None = None
    nested: bool = False

    def verify(self) -> list[str]:
        problems = self.text.verify() + self.vision.verify()
        # The two shapes are held to different standards on purpose. `model_type`
        # and `architectures` are how Transformers recognises a checkpoint, and
        # the nested file is the one a loader is handed; the flat file is a
        # reference-runtime artifact that never had them, so requiring them there
        # would be reporting the release's own choice as a defect.
        if self.nested:
            if not self.text.model_type:
                problems.append("config.json carries no model_type")
            if not self.text.architectures:
                problems.append("config.json carries no architectures")
        return problems

    def differs_from(self, other: V41Config) -> list[str]:
        """Every canonical field on which two configs disagree, or one is silent.

        Used to prove that the two shipped files describe the same model, which
        is the claim the schema exists to make checkable. A field absent from one
        of them is reported as a difference rather than skipped, so the list is
        also the honest statement of what the flat file does not say.
        """
        differences: list[str] = []
        for half in ("text", "vision"):
            mine, theirs = getattr(self, half), getattr(other, half)
            for field in fields(mine):
                a, b = getattr(mine, field.name), getattr(theirs, field.name)
                if _is_unstated(a) and _is_unstated(b):
                    continue
                if _is_unstated(a) or _is_unstated(b):
                    differences.append(
                        f"{half}.{field.name}: {a!r} here, {b!r} in {other.source or 'the other config'}"
                    )
                elif a != b:
                    differences.append(f"{half}.{field.name}: {a!r} != {b!r}")
        return differences

    def as_reference_dict(self) -> dict[str, Any]:
        """This config keyed the way the flat reference file keys it.

        The inverse of the alias table: a field is included exactly when the flat
        shape has a name for it. Fields only the Transformers file names -- the
        token ids, `topk_method`, the YaRN-free rope keys -- are left out, and so
        are the vision fields under their nested names. Values are copied as
        lists, so the result is JSON-shaped.

        A null here can mean the source stated null or that it did not mention
        the key at all; the flat file itself only does the former, for
        `vision_max_wh_ratio`, so the distinction does not arise for the two
        released files.
        """
        out: dict[str, Any] = {}
        for half in ("text", "vision"):
            instance = getattr(self, half)
            for field in fields(instance):
                key = _ALIASES[half][field.name][1]
                if key is None:
                    continue
                value = getattr(instance, field.name)
                out[key] = list(value) if field.name in _SEQUENCE_FIELDS else value
        return out

    def engram_block(self) -> dict[str, Any]:
        """The Engram fields under the names `src/encoding/engram.py` reads.

        The flat file already uses these names, so this is the identity there and
        the aliasing everywhere else. Kept explicit rather than making the Engram
        module learn the canonical names: it is a standalone derivation over a
        dict, and this is the one place that knows how the two name sets relate.
        """
        text = self.text
        return {
            "engram_layer_ids": list(text.engram_layer_ids),
            "engram_max_ngram_size": text.engram_max_ngram_size,
            "engram_n_heads": text.engram_n_heads,
            "engram_head_dim": text.engram_head_dim,
            "engram_vocab_size": text.engram_vocab_size,
            "engram_num_embeddings": list(text.engram_num_embeddings),
            "engram_compressed_vocab_size": text.engram_compressed_vocab_size,
            "engram_pad_id": text.engram_pad_id,
        }


def _build(half: str, cls: type, source: Mapping[str, Any], which: int) -> Any:
    """Instantiate a dataclass by reading each canonical field out of `source`.

    `which` selects the column of the alias table: 0 for the nested
    Transformers layout, 1 for the flat reference one.
    """
    values: dict[str, Any] = {}
    for name in (field.name for field in fields(cls)):
        value = _lookup(source, _ALIASES[half][name][which])
        if name in _SEQUENCE_FIELDS:
            # Kept iterable rather than left as None, so a consumer can loop over
            # a config whose file simply does not list any.
            value = tuple(value or ())
        values[name] = value
    return cls(**values)


def from_dict(config: Mapping[str, Any], source: str | None = None) -> V41Config:
    """Build a `V41Config` from either shipped shape.

    The nested Transformers layout is recognised by its ``text_config`` block.
    Anything else is read as the flat reference layout -- including the
    ``{"model": {...}}`` wrapper the model card describes, which the released
    files do not actually use but which costs nothing to accept.
    """
    nested = isinstance(config.get(_HF_TEXT_BLOCK), Mapping)
    flat: Mapping[str, Any] = config
    if not nested and isinstance(config.get("model"), Mapping):
        flat = config["model"]
        nested = isinstance(flat.get(_HF_TEXT_BLOCK), Mapping)

    if nested:
        return V41Config(
            text=_build("text", V41TextConfig, config, 0),
            vision=_build("vision", V41VisionConfig, config, 0),
            source=source,
            nested=True,
        )
    return V41Config(
        text=_build("text", V41TextConfig, flat, 1),
        vision=_build("vision", V41VisionConfig, flat, 1),
        source=source,
        nested=False,
    )


def load_config(path: str) -> V41Config:
    """Read a released ``config.json`` or ``inference/config.json``."""
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return from_dict(config, source=path)


def resolve_config(checkpoint_dir: str, explicit: str | None = None) -> str:
    """The config a checkpoint directory carries, preferring the nested one.

    `config.json` first: it is the file the release ships and the only one that
    carries `topk_method`, `norm_topk_prob` and the token ids. The flat
    ``inference/config.json`` is the fallback for a directory that only has the
    reference tree.
    """
    if explicit:
        return explicit
    for name in ("config.json", _FLAT_INFERENCE_CONFIG):
        candidate = os.path.join(checkpoint_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"no config.json or {_FLAT_INFERENCE_CONFIG} in {checkpoint_dir}")


def describe(config: V41Config) -> str:
    """A one-line summary, for logs and for the module's own command line."""
    text = config.text
    shape = (
        f"{text.n_layers} layers + {text.n_mtp_layers} draft, dim {text.dim}, "
        f"{text.n_heads} heads of {text.head_dim}, {text.n_routed_experts} experts "
        f"({text.n_activated_experts} active)"
    )
    extras = []
    if text.engram_layer_ids:
        extras.append(f"Engram on layers {list(text.engram_layer_ids)}")
    if config.vision.enabled:
        extras.append(f"vision {config.vision.n_layers} layers")
    if text.n_mtp_layers:
        extras.append(f"DSpark block {text.dspark_block_size} over layers {list(text.dspark_target_layer_ids)}")
    return f"{shape}; " + ", ".join(extras) if extras else shape


def main(argv: Sequence[str] | None = None) -> int:
    """Read a config, print the normalized view, and check it against itself.

    Exit status is 0 only when `verify()` finds nothing. Naming a checkpoint
    directory resolves the config the same way the audit script does, so both
    read the same file for the same input.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.models.deepseek_v4_1.config",
        description="Normalize and check a DeepSeek-V4.1-Flash config.",
    )
    parser.add_argument("path", help=f"config.json, {_FLAT_INFERENCE_CONFIG}, or a checkpoint directory")
    parser.add_argument("--json", help="write the normalized config and the verification result here")
    args = parser.parse_args(argv)

    path = args.path
    if os.path.isdir(path):
        try:
            path = resolve_config(path)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 2

    config = load_config(path)
    print(f"{path} ({'nested' if config.nested else 'flat'})")
    print(f"  {describe(config)}")

    problems = config.verify()
    for problem in problems:
        print(f"  [FAIL] {problem}")
    if problems:
        print(f"  {len(problems)} problem(s)")
    else:
        print("  [ok] the config is self-consistent")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "path": path,
                    "nested": config.nested,
                    "text": {field.name: getattr(config.text, field.name) for field in fields(config.text)},
                    "vision": {field.name: getattr(config.vision, field.name) for field in fields(config.vision)},
                    "problems": problems,
                },
                handle,
                indent=2,
                default=list,
            )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
