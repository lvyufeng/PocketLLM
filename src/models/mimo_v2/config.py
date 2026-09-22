"""The MiMo-V2.6 configuration, and the layer shape contract the checkpoint is cut to.

The released checkpoint carries a single flat `config.json`: the text backbone's
hyper-parameters sit at the top level, and only the towers and the quantization
block are nested (`vision_config`, `audio_config`, `processor_config`,
`quantization_config`). That is unlike V4.1, whose text half is nested under
`text_config` and ships twice in two layouts -- see
`src/models/deepseek_v4_1/config.py`. There is one file here, so there is one
reader.

What this module is for is narrower than "parse the JSON". The model is a hybrid
of two attention families whose geometry differs per layer, and the difference is
not cosmetic:

* Global-attention (GA) layers use 4 KV heads, sliding-window (SWA) layers 8.
* SWA layers carry a per-head attention sink bias; GA layers do not.
* The two families use different RoPE bases.

So `qkv_proj`'s output width is **13568 on a GA layer and 14848 on an SWA layer**,
inside one checkpoint. A reader that derives that width once and reuses it is
wrong on 39 of the 48 layers, and wrong in a way that surfaces as a shape error at
load time rather than as a numerics drift. `attention(layer_idx)` is the accessor
that keeps that straight; `main()` prints the whole table.

Two things are deliberately *not* done here. The vision and audio towers are
parsed into their raw dicts and nothing else -- the text runtime does not execute
them, and normalizing them here would suggest otherwise. And the tensor inventory
is not this module's business: the shard-header audit owns the mapping from names
to tensors, and reads its config through `from_dict`.

Field contracts
---------------

The reference implementation is a `PretrainedConfig`, so a key it does not name
in its signature falls through to `**kwargs` and is *stored but never read*,
while a key it names but that is absent from the file takes the signature
default. Those two cases look identical from the JSON and mean opposite things.
Every field below therefore states which it is, and the ones where "absent" is a
trap carry the trap in the comment:

* ``norm_topk_prob`` -- absent means **unsaid**, and the reference's own default
  is ``True``. Absent is not ``False``. This is the contract V4.1 documents too;
  reading it as a bare truthiness test silently drops the top-k renormalization,
  which is a routing change and not a rounding one.
* ``attention_projection_layout`` -- absent means **unsaid**, and the reference's
  default is ``"split"``, which is the *opposite* of what this checkpoint uses
  (``"fused_qkv"``). The field decides both the tensor names and the tensor
  count: one ``qkv_proj`` versus separate ``q_proj``/``k_proj``/``v_proj``. A
  reader that trusts the default finds no ``qkv_proj.weight`` and three tensors
  that do not exist.
* ``hybrid_layer_pattern`` -- absent means **unsaid**, and the reference resolves
  it to *all global attention*: either from ``hybrid_block_size`` when that is
  given (``[0 if (i+1) % block else 1]``, i.e. GA every n-th layer), or to
  ``[0] * n``. Assuming the hybrid pattern is the default inverts the model.
* ``moe_layer_freq`` -- absent means **unsaid**, and the reference resolves it to
  *all dense*, turning a 309B MoE checkpoint into a dense model with no error. An
  int is a period rather than a flag: the reference expands ``freq`` as
  ``i % freq == 0``.
* ``head_dim`` / ``v_head_dim`` -- absent means **unsaid**, and the reference falls
  back to ``hidden_size // num_attention_heads``, which for this architecture is
  64 against the checkpoint's 192. The fallback is self-consistent, so nothing
  raises; the derived qkv width just stops matching the tensors.
* ``attention_value_scale`` -- ``None`` here is a **real value**, not "unsaid": the
  reference guards on ``if self.v_scale is not None`` and skips the multiply. This
  is the one field where the two readings coincide, and it is called out because
  the opposite habit is correct for ``norm_topk_prob``.
* ``routed_scaling_factor`` -- absent means unsaid and the reference substitutes
  ``1.0``, where V4.1 requires the field to be present and positive. Do not carry
  the V4.1 check over.
* ``n_shared_experts`` -- absent means unsaid, and the reference builds **no shared
  expert in either case**: ``MiMoV2MoE`` constructs only the routed experts and
  never reads the field. A non-null value is therefore inert, and a reader who
  honors it adds a computation the checkpoint does not have.
* ``moe_router_dtype`` -- inert in the same way: the reference's gate upcasts both
  operands to fp32 explicitly, so the declared ``"bfloat16"`` describes storage
  and not the inference arithmetic.

  These last two are inert rather than broken, so they are reported by
  ``runtime_notes()`` and not by ``verify()``: the released checkpoint states
  ``moe_router_dtype``, a correct implementation ignores it, and failing a load
  gate over it would be wrong. The line is whether honoring the field changes the
  answer.
* ``attention_chunk_size`` -- not in the reference's signature at all, so it is
  stored and never read. It equals ``sliding_window`` in this checkpoint, which is
  presumably why it survived; it is not a second window.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

__all__ = [
    "MimoV2AttentionShape",
    "MimoV2Config",
    "MimoV2DraftConfig",
    "MimoV2QuantSpec",
    "MimoV2TextConfig",
    "describe",
    "from_dict",
    "from_pretrained",
    "load_config",
]

# The nested blocks. Everything else in the file belongs to the text backbone.
_VISION_BLOCK = "vision_config"
_AUDIO_BLOCK = "audio_config"
_PROCESSOR_BLOCK = "processor_config"
_QUANT_BLOCK = "quantization_config"

# The draft model ships its own config in a subdirectory, with a different
# `model_type`: it reuses the Qwen3 components rather than the MiMo ones.
_DRAFT_CONFIG = "dflash/config.json"

# The reference's own signature defaults, quoted because "absent" resolves to these
# and not to the JSON's apparent zero value.
_DEFAULT_PROJECTION_LAYOUT = "split"
_DEFAULT_NORM_TOP_K_PROB = True
_DEFAULT_ROUTED_SCALING_FACTOR = 1.0
_DEFAULT_PARTIAL_ROTARY_FACTOR = 1.0
_DEFAULT_SCORING_FUNC = "sigmoid"
_DEFAULT_TOPK_METHOD = "noaux_tc"

# The reference's gate implements exactly one scoring function and one top-k
# method; anything else raises at the first MoE layer.
_SCORING_FUNCS = ("sigmoid",)
_TOPK_METHODS = ("noaux_tc",)
_PROJECTION_LAYOUTS = ("split", "fused_qkv")
#: What a fused projection's row order is when the file does not say. The released
#: weights are stored as four tensor-parallel shards of `[q | k | v]`; see
#: `layers.split_fused_qkv` for the measurement that rules the alternative out.
_DEFAULT_FUSED_QKV_ROW_LAYOUT = "tp4_interleaved"
_QKV_ROW_LAYOUTS = ("contiguous", "tp4_interleaved", "tp4_interleaved_vk")
_ACTIVATIONS = ("silu", "gelu", "gelu_pytorch_tanh", "gelu_new", "relu", "sigmoid", "tanh")


def _is_layer_scoped(name: str) -> bool:
    """Whether an `ignored_layers` entry names a specific layer.

    The backbone is `model.layers.<i>.…` and the draft heads are
    `model.mtp.layers.<i>.…`; everything the quantization block can address is one
    of those two, so an entry that is neither addresses nothing.
    """
    for prefix in ("model.layers.", "model.mtp.layers."):
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):].split(".", 1)
        if len(rest) == 2 and rest[0].isdigit():
            return True
    return False


@dataclass(frozen=True)
class MimoV2AttentionShape:
    """One layer's attention geometry, after the reference has resolved the defaults.

    The per-family fields are the ones this exists for: `num_kv_heads`, `rope_theta`
    and `has_sink` are all a function of `is_swa`, and `qkv_out` changes with them.
    """

    layer_idx: int
    is_swa: bool

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    num_key_value_groups: int

    q_size: int
    k_size: int
    v_size: int
    #: The fused projection's output width, and the `o_proj` input width. The first
    #: differs between the two families; the second does not, because the value
    #: width and the query count are the same in both.
    qkv_out: int
    o_in: int

    rope_dim: int
    rope_theta: float
    partial_rotary_factor: float
    scaling: float

    sliding_window: int | None
    has_sink: bool
    #: The reference applies this to the values before attention. `None` means it
    #: applies nothing -- a real value, not an unstated field.
    value_scale: float | None
    projection_layout: str
    #: Row order of the released fused `qkv_proj`. See
    #: `layers.split_fused_qkv`: the projection is one tensor, so a wrong reading
    #: is shape-correct and only shows up in the logits.
    qkv_row_layout: str = "contiguous"

    @property
    def family(self) -> str:
        return "swa" if self.is_swa else "ga"

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class MimoV2TextConfig:
    """The text backbone, named after the reference's `ModelArgs` fields.

    Every field is optional, because "the file does not say" and "the file says no"
    are different claims and the reference distinguishes them by falling back to
    its own defaults. The `resolved_*` accessors perform that fallback so call
    sites do not each re-derive it; the raw fields stay raw so a caller can tell
    which of the two it is looking at.
    """

    # -- identity ---------------------------------------------------------
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    #: Absent from the released file. The chat template opens with an explicit
    #: `<|im_start|>`, so nothing needs a BOS.
    bos_token_id: int | None = None
    eos_token_id: int | tuple[int, ...] = 151645
    pad_token_id: int | None = 151643
    dtype: str | None = None

    # -- backbone ---------------------------------------------------------
    vocab_size: int = 152576
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    hidden_act: str = "silu"
    layernorm_epsilon: float = 1e-6
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    use_cache: bool = True

    # -- attention: the two families --------------------------------------
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    #: Unstated falls back to `hidden_size // num_attention_heads`, not to a
    #: sensible MiMo value. See the module docstring.
    head_dim: int | None = None
    v_head_dim: int | None = None
    swa_num_attention_heads: int | None = None
    swa_num_key_value_heads: int | None = None
    swa_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    swa_rope_theta: float | None = None
    rope_theta: float = 10_000_000.0
    rope_parameters: Mapping[str, Any] | None = None
    partial_rotary_factor: float = _DEFAULT_PARTIAL_ROTARY_FACTOR
    attention_value_scale: float | None = None
    attention_projection_layout: str | None = None
    #: Row order of the fused `qkv_proj`, or absent to take the one the projection
    #: layout implies -- see `resolved_qkv_row_layout`. It is a field only so a
    #: fixture whose fused weight was built in the other order can say so.
    attention_qkv_row_layout: str | None = None
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = False
    sliding_window: int | None = None
    sliding_window_size: int | None = None
    #: Stored by the reference and never read. See the module docstring.
    attention_chunk_size: int | None = None

    # -- layer pattern ----------------------------------------------------
    hybrid_layer_pattern: tuple[int, ...] = ()
    hybrid_block_size: int | None = None

    # -- MoE --------------------------------------------------------------
    intermediate_size: int = 16384
    moe_intermediate_size: int | None = None
    n_routed_experts: int | None = None
    #: Inert in the reference, which builds no shared expert either way.
    n_shared_experts: int | None = None
    num_experts_per_tok: int | None = None
    #: An int is a period, not a flag; see `resolved_moe_layer_freq`.
    moe_layer_freq: tuple[int, ...] | int = ()
    scoring_func: str = _DEFAULT_SCORING_FUNC
    topk_method: str = _DEFAULT_TOPK_METHOD
    #: Required in practice: the reference reshapes by `n_group` and raises on
    #: `None`. Its own signature default is `None`, so an absent key is a config
    #: the reference cannot run rather than a defaulted one.
    n_group: int | None = None
    topk_group: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    #: Inert: the reference upcasts the gate to fp32 regardless.
    moe_router_dtype: str | None = None

    # -- draft head count (the drafter's weights live in their own file) ---
    num_nextn_predict_layers: int = 0

    # -- multimodal token ids: the towers are out of scope, the ids are not,
    # because the tokenizer and the chat template both need them -----------
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    audio_token_id: int | None = None
    audio_start_token_id: int | None = None
    audio_end_token_id: int | None = None
    vision_model_type: str | None = None

    def __post_init__(self) -> None:
        # `__post_init__` on a frozen dataclass may only assign through
        # `object.__setattr__`, and only to normalize a type that the constructor
        # accepts in more than one shape.
        if isinstance(self.eos_token_id, int):
            object.__setattr__(self, "eos_token_id", (self.eos_token_id,))
        elif isinstance(self.eos_token_id, list):
            object.__setattr__(self, "eos_token_id", tuple(self.eos_token_id))

    # =====================================================================
    # Resolution: what the reference computes from the raw fields
    # =====================================================================

    @property
    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        return self.hidden_size // self.num_attention_heads

    @property
    def resolved_v_head_dim(self) -> int:
        return self.resolved_head_dim if self.v_head_dim is None else int(self.v_head_dim)

    @property
    def resolved_swa_head_dim(self) -> int:
        return self.resolved_head_dim if self.swa_head_dim is None else int(self.swa_head_dim)

    @property
    def resolved_swa_v_head_dim(self) -> int:
        """Defaults through the SWA head dim, not straight to the global one."""
        if self.swa_v_head_dim is not None:
            return int(self.swa_v_head_dim)
        return self.resolved_swa_head_dim

    @property
    def resolved_hybrid_layer_pattern(self) -> tuple[int, ...]:
        """Per-layer 1 = SWA, 0 = global attention.

        Reproduces the reference exactly, including the case that matters most:
        with neither field stated the pattern is all-zero, i.e. **every layer is
        global attention**. Reading an absent pattern as "all SWA" inverts the
        model, and it is a plausible mistake because the released checkpoint is 39
        SWA layers out of 48.
        """
        if self.hybrid_layer_pattern:
            return tuple(int(v) for v in self.hybrid_layer_pattern)
        if self.hybrid_block_size is not None:
            return tuple(
                0 if (i + 1) % self.hybrid_block_size == 0 else 1
                for i in range(self.num_hidden_layers)
            )
        return (0,) * self.num_hidden_layers

    @property
    def resolved_moe_layer_freq(self) -> tuple[bool, ...]:
        """Per-layer True = routed MoE FFN, False = dense FFN.

        An int is a period, not a flag: the reference expands `freq` as
        `i % freq == 0`, so `1` means every layer is MoE and `2` means every other
        one starting at layer 0. An absent value is all-dense, which is what makes
        an omitted `moe_layer_freq` dangerous rather than merely wrong.

        `bool` is a subclass of `int` and the reference's own `isinstance` test
        accepts it, so `True` means every layer here as well.
        """
        raw = self.moe_layer_freq
        if isinstance(raw, (tuple, list)):
            return tuple(bool(v) for v in raw) if raw else (False,) * self.num_hidden_layers
        if isinstance(raw, int):
            return tuple(bool(raw > 0 and i % raw == 0) for i in range(self.num_hidden_layers))
        return (False,) * self.num_hidden_layers

    @property
    def resolved_partial_rotary_factor(self) -> float:
        """`rope_parameters` wins over the flat field, as in the reference.

        The released file carries both and they agree (0.334); a repack that moved
        only one of them would otherwise change the rotation silently.
        """
        if self.rope_parameters and "partial_rotary_factor" in self.rope_parameters:
            return float(self.rope_parameters["partial_rotary_factor"])
        return float(self.partial_rotary_factor)

    @property
    def resolved_rope_theta(self) -> float:
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            return float(self.rope_parameters["rope_theta"])
        return float(self.rope_theta)

    @property
    def resolved_swa_rope_theta(self) -> float:
        """The SWA base, defaulting to the global one when unstated.

        The checkpoint states both and they differ by three orders of magnitude
        (`1e4` against `1e7`), so an implementation that built one table and used
        it for both would be wrong on 39 of 48 layers.
        """
        if self.swa_rope_theta is None:
            return self.resolved_rope_theta
        return float(self.swa_rope_theta)

    @property
    def resolved_projection_layout(self) -> str:
        return self.attention_projection_layout or _DEFAULT_PROJECTION_LAYOUT

    @property
    def resolved_qkv_row_layout(self) -> str:
        """Row order of the fused projection, which its layout decides.

        A split projection has separate `q_proj`/`k_proj`/`v_proj` tensors and no
        row order to get wrong. A fused one is stored as four tensor-parallel
        shards of `[q | k | v]` -- the order the serving stack's loader requires,
        and the order the released weights are measured to be in -- so the row
        order is a consequence of the layout rather than a separate field. See
        `layers.split_fused_qkv`.
        """
        if self.attention_qkv_row_layout is not None:
            return self.attention_qkv_row_layout
        if self.resolved_projection_layout == "fused_qkv":
            return _DEFAULT_FUSED_QKV_ROW_LAYOUT
        return "contiguous"

    @property
    def resolved_norm_topk_prob(self) -> bool:
        """Absent means unsaid, and unsaid means the reference's `True`."""
        return _DEFAULT_NORM_TOP_K_PROB if self.norm_topk_prob is None else bool(self.norm_topk_prob)

    @property
    def resolved_routed_scaling_factor(self) -> float:
        return (
            _DEFAULT_ROUTED_SCALING_FACTOR
            if self.routed_scaling_factor is None
            else float(self.routed_scaling_factor)
        )

    @property
    def resolved_moe_intermediate_size(self) -> int:
        return (
            self.intermediate_size if self.moe_intermediate_size is None else int(self.moe_intermediate_size)
        )

    @property
    def resolved_window(self) -> int | None:
        """`sliding_window` first, then `sliding_window_size`, as in the reference."""
        if self.sliding_window is not None:
            return int(self.sliding_window)
        if self.sliding_window_size is not None:
            return int(self.sliding_window_size)
        return None

    @property
    def resolved_swa_num_attention_heads(self) -> int:
        return (
            self.num_attention_heads
            if self.swa_num_attention_heads is None
            else int(self.swa_num_attention_heads)
        )

    @property
    def resolved_swa_num_key_value_heads(self) -> int:
        return (
            self.num_key_value_heads
            if self.swa_num_key_value_heads is None
            else int(self.swa_num_key_value_heads)
        )

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """Every stop id, as a tuple: a checkpoint may state one or several."""
        if isinstance(self.eos_token_id, int):
            return (self.eos_token_id,)
        return tuple(self.eos_token_id)

    # =====================================================================
    # The shape contract
    # =====================================================================

    def attention(self, layer_idx: int) -> MimoV2AttentionShape:
        """The one layer's attention geometry, with every default resolved.

        `qkv_out` is the number that matters: 13568 on a global layer and 14848 on
        a sliding-window layer, in the same checkpoint. It is derived rather than
        tabulated so a config edit cannot leave it stale.
        """
        pattern = self.resolved_hybrid_layer_pattern
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(f"layer {layer_idx} is outside 0..{self.num_hidden_layers - 1}")
        is_swa = bool(pattern[layer_idx])

        if is_swa:
            num_q_heads = self.resolved_swa_num_attention_heads
            num_kv_heads = self.resolved_swa_num_key_value_heads
            head_dim = self.resolved_swa_head_dim
            v_head_dim = self.resolved_swa_v_head_dim
            rope_theta = self.resolved_swa_rope_theta
            window = self.resolved_window
            has_sink = bool(self.add_swa_attention_sink_bias)
        else:
            num_q_heads = self.num_attention_heads
            num_kv_heads = self.num_key_value_heads
            head_dim = self.resolved_head_dim
            v_head_dim = self.resolved_v_head_dim
            rope_theta = self.resolved_rope_theta
            window = None
            has_sink = bool(self.add_full_attention_sink_bias)

        q_size = num_q_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * v_head_dim
        return MimoV2AttentionShape(
            layer_idx=layer_idx,
            is_swa=is_swa,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            num_key_value_groups=num_q_heads // num_kv_heads if num_kv_heads else 0,
            q_size=q_size,
            k_size=k_size,
            v_size=v_size,
            qkv_out=q_size + k_size + v_size,
            o_in=num_q_heads * v_head_dim,
            rope_dim=int(head_dim * self.resolved_partial_rotary_factor),
            rope_theta=rope_theta,
            partial_rotary_factor=self.resolved_partial_rotary_factor,
            scaling=head_dim**-0.5,
            sliding_window=window,
            has_sink=has_sink,
            value_scale=None if self.attention_value_scale is None else float(self.attention_value_scale),
            projection_layout=self.resolved_projection_layout,
            qkv_row_layout=self.resolved_qkv_row_layout,
        )

    def ffn_kind(self, layer_idx: int) -> str:
        """`"dense"` or `"moe"` for one layer."""
        return "moe" if self.resolved_moe_layer_freq[layer_idx] else "dense"

    def ffn_intermediate_size(self, layer_idx: int) -> int:
        """A dense layer's width, or a routed expert's."""
        return (
            self.resolved_moe_intermediate_size
            if self.ffn_kind(layer_idx) == "moe"
            else self.intermediate_size
        )

    @property
    def moe_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, is_moe in enumerate(self.resolved_moe_layer_freq) if is_moe)

    @property
    def dense_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, is_moe in enumerate(self.resolved_moe_layer_freq) if not is_moe)

    @property
    def swa_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, is_swa in enumerate(self.resolved_hybrid_layer_pattern) if is_swa)

    @property
    def global_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, is_swa in enumerate(self.resolved_hybrid_layer_pattern) if not is_swa)

    @property
    def total_layers(self) -> int:
        """Backbone plus draft layers."""
        return self.num_hidden_layers + self.num_nextn_predict_layers

    # =====================================================================
    # Self-consistency
    # =====================================================================

    def verify(self) -> list[str]:
        """Problems with the config itself, as strings; empty means none.

        Each entry is grounded in the reference rather than in a preference: it is
        either a config the reference raises on, or a field the reference silently
        ignores that a reader would honor. Both are worth failing a load over, and
        the second kind is why this method exists at all -- a config the reference
        *cannot* run is loud, while an inert field getting honored is silent.

        PocketLLM-side constraints -- which quantized layouts this repository has
        kernels for -- are deliberately not here. They are in `runtime_notes`.
        """
        problems: list[str] = []

        def want(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        n = self.num_hidden_layers
        want(n > 0, f"num_hidden_layers is {n}")
        if self.model_type is not None:
            want(self.model_type == "mimo_v2", f"model_type is {self.model_type!r}, not 'mimo_v2'")
        if self.architectures:
            want(
                any("MiMoV2" in name for name in self.architectures),
                f"architectures {list(self.architectures)} names no MiMoV2 class",
            )

        # -- the two per-layer patterns must cover the stack ---------------
        if self.hybrid_layer_pattern:
            want(
                len(self.hybrid_layer_pattern) == n,
                f"hybrid_layer_pattern has {len(self.hybrid_layer_pattern)} entries for {n} layers",
            )
            want(
                all(v in (0, 1) for v in self.hybrid_layer_pattern),
                f"hybrid_layer_pattern must be 0/1, got {sorted(set(self.hybrid_layer_pattern))}",
            )
            # The reference ignores `hybrid_block_size` when the pattern is
            # present, so a file carrying both is a file whose author meant one of
            # them and got the other.
            want(
                self.hybrid_block_size is None,
                "both hybrid_layer_pattern and hybrid_block_size are set; the reference "
                "ignores hybrid_block_size whenever the pattern is present",
            )
        if isinstance(self.moe_layer_freq, (tuple, list)) and self.moe_layer_freq:
            want(
                len(self.moe_layer_freq) == n,
                f"moe_layer_freq has {len(self.moe_layer_freq)} entries for {n} layers",
            )
        want(
            len(self.resolved_hybrid_layer_pattern) == n,
            f"the resolved hybrid pattern has {len(self.resolved_hybrid_layer_pattern)} entries "
            f"for {n} layers",
        )
        want(
            len(self.resolved_moe_layer_freq) == n,
            f"the resolved MoE frequency has {len(self.resolved_moe_layer_freq)} entries "
            f"for {n} layers",
        )

        # -- the SWA family needs a window ---------------------------------
        has_swa = bool(self.swa_layer_indices)
        want(
            not has_swa or self.resolved_window is not None,
            "the layer pattern has sliding-window layers but neither sliding_window nor "
            "sliding_window_size is set; the reference raises rather than defaulting",
        )
        if has_swa and self.resolved_window is not None:
            want(self.resolved_window > 0, f"sliding window is {self.resolved_window}")

        # -- head divisibility, per family ---------------------------------
        for q_name, q, kv_name, kv in (
            ("num_attention_heads", self.num_attention_heads,
             "num_key_value_heads", self.num_key_value_heads),
            ("swa_num_attention_heads", self.resolved_swa_num_attention_heads,
             "swa_num_key_value_heads", self.resolved_swa_num_key_value_heads),
        ):
            want(q > 0 and kv > 0, f"{q_name}={q} / {kv_name}={kv} are not both positive")
            if q > 0 and kv > 0:
                want(q % kv == 0, f"{q_name} {q} is not divisible by {kv_name} {kv}")

        for label, head_dim, v_head_dim in (
            ("global", self.resolved_head_dim, self.resolved_v_head_dim),
            ("sliding-window", self.resolved_swa_head_dim, self.resolved_swa_v_head_dim),
        ):
            want(head_dim > 0, f"{label} head_dim is {head_dim}")
            want(v_head_dim > 0, f"{label} v_head_dim is {v_head_dim}")
            rope_dim = int(head_dim * self.resolved_partial_rotary_factor)
            want(
                rope_dim % 2 == 0,
                f"{label} rope_dim is {rope_dim} from head_dim {head_dim} x "
                f"partial_rotary_factor {self.resolved_partial_rotary_factor}; the reference "
                f"requires an even rotary dimension",
            )
            want(rope_dim <= head_dim, f"{label} rope_dim {rope_dim} exceeds head_dim {head_dim}")

        if self.head_dim is None:
            problems.append(
                f"head_dim is unstated, so the reference falls back to "
                f"hidden_size // num_attention_heads = {self.resolved_head_dim}; for this "
                f"architecture the checkpoint's value is unrelated to that ratio"
            )
        if self.v_head_dim is None:
            problems.append(
                "v_head_dim is unstated, so the reference falls back to head_dim; the "
                "checkpoint's value is smaller, which changes the o_proj input width"
            )

        want(
            0.0 < self.resolved_partial_rotary_factor <= 1.0,
            f"partial_rotary_factor is {self.resolved_partial_rotary_factor}",
        )
        want(self.resolved_rope_theta > 0, f"rope_theta is {self.resolved_rope_theta}")
        want(self.resolved_swa_rope_theta > 0, f"swa_rope_theta is {self.resolved_swa_rope_theta}")
        want(self.layernorm_epsilon > 0, f"layernorm_epsilon is {self.layernorm_epsilon}")
        want(self.hidden_act in _ACTIVATIONS, f"hidden_act is {self.hidden_act!r}")
        want(
            self.resolved_projection_layout in _PROJECTION_LAYOUTS,
            f"attention_projection_layout is {self.resolved_projection_layout!r}, not one of "
            f"{list(_PROJECTION_LAYOUTS)}",
        )
        want(
            self.resolved_qkv_row_layout in _QKV_ROW_LAYOUTS,
            f"the fused qkv row layout is {self.resolved_qkv_row_layout!r}, not one of "
            f"{list(_QKV_ROW_LAYOUTS)}",
        )
        if self.attention_value_scale is not None:
            want(self.attention_value_scale > 0, f"attention_value_scale is {self.attention_value_scale}")

        # -- the routed FFN ------------------------------------------------
        if self.moe_layer_indices:
            want(
                self.n_routed_experts is not None and self.n_routed_experts > 0,
                f"n_routed_experts is {self.n_routed_experts} with "
                f"{len(self.moe_layer_indices)} MoE layer(s)",
            )
            want(
                self.num_experts_per_tok is not None
                and self.n_routed_experts is not None
                and 0 < self.num_experts_per_tok <= self.n_routed_experts,
                f"num_experts_per_tok {self.num_experts_per_tok} of n_routed_experts "
                f"{self.n_routed_experts}",
            )
            want(
                self.scoring_func in _SCORING_FUNCS,
                f"scoring_func is {self.scoring_func!r}; the reference's gate implements only "
                f"{list(_SCORING_FUNCS)} and raises otherwise",
            )
            want(
                self.topk_method in _TOPK_METHODS,
                f"topk_method is {self.topk_method!r}; the reference's gate implements only "
                f"{list(_TOPK_METHODS)} and raises otherwise",
            )
            want(
                self.n_group is not None and self.n_group > 0,
                f"n_group is {self.n_group}; the reference reshapes by it and would raise",
            )
            want(
                self.topk_group is not None and self.topk_group > 0,
                f"topk_group is {self.topk_group}; the reference top-k's by it and would raise",
            )
            if self.n_group is not None and self.topk_group is not None and self.n_group > 0:
                want(
                    self.topk_group <= self.n_group,
                    f"topk_group {self.topk_group} exceeds n_group {self.n_group}",
                )
                if self.n_routed_experts:
                    want(
                        self.n_routed_experts % self.n_group == 0,
                        f"n_routed_experts {self.n_routed_experts} does not divide into n_group "
                        f"{self.n_group}",
                    )
                    per_group = self.n_routed_experts // self.n_group
                    # The reference's group score is a `topk(2)` within each group.
                    want(
                        per_group >= 2,
                        f"each of the {self.n_group} group(s) holds {per_group} expert(s); the "
                        f"reference takes the top 2 per group for the group score",
                    )
            want(
                self.resolved_moe_intermediate_size > 0,
                f"moe_intermediate_size is {self.resolved_moe_intermediate_size}",
            )
            # Inert fields a reader is likely to honor. They are reported by
            # `runtime_notes` rather than here: the released checkpoint states
            # `moe_router_dtype` and the reference ignores it, so neither is a
            # defect in the config -- but both change the answer if honored.

        want(self.intermediate_size > 0, f"intermediate_size is {self.intermediate_size}")
        want(self.hidden_size > 0, f"hidden_size is {self.hidden_size}")
        want(self.vocab_size > 0, f"vocab_size is {self.vocab_size}")
        want(
            self.num_nextn_predict_layers >= 0,
            f"num_nextn_predict_layers is {self.num_nextn_predict_layers}",
        )
        want(
            self.max_position_embeddings > 0,
            f"max_position_embeddings is {self.max_position_embeddings}",
        )
        return problems

    def describe(self) -> str:
        parts = [
            f"{self.num_hidden_layers} layers, hidden {self.hidden_size}, vocab {self.vocab_size}"
        ]
        if self.global_layer_indices:
            shape = self.attention(self.global_layer_indices[0])
            parts.append(
                f"GA {len(self.global_layer_indices)} ({self._head_summary(shape)}, "
                f"qkv {shape.qkv_out}, theta {shape.rope_theta:g}"
                f"{', sink' if shape.has_sink else ''})"
            )
        if self.swa_layer_indices:
            shape = self.attention(self.swa_layer_indices[0])
            parts.append(
                f"SWA {len(self.swa_layer_indices)} ({self._head_summary(shape)}, "
                f"qkv {shape.qkv_out}, theta {shape.rope_theta:g}, "
                f"window {shape.sliding_window}{', sink' if shape.has_sink else ''})"
            )
        if self.moe_layer_indices:
            parts.append(
                f"MoE on {len(self.moe_layer_indices)} layer(s): {self.n_routed_experts} experts, "
                f"top-{self.num_experts_per_tok}, {self.resolved_moe_intermediate_size} wide, "
                f"{self.scoring_func}/{self.topk_method}"
            )
        if self.dense_layer_indices:
            parts.append(f"dense FFN on layer(s) {list(self.dense_layer_indices)}")
        if self.num_nextn_predict_layers:
            parts.append(f"{self.num_nextn_predict_layers} MTP layers")
        return "; ".join(parts)

    @staticmethod
    def _head_summary(shape: MimoV2AttentionShape) -> str:
        return (
            f"{shape.num_q_heads}Q/{shape.num_kv_heads}KV, head {shape.head_dim}, "
            f"v {shape.v_head_dim}, rope {shape.rope_dim}"
        )

    # =====================================================================

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MimoV2TextConfig":
        """Build from a flat config mapping, ignoring what this class does not name.

        The nested tower and quantization blocks are dropped here rather than
        carried: `MimoV2Config` owns them, and a copy in two places is a copy that
        can disagree. Only `None` is dropped -- the values this signature does not
        name are the reference's `**kwargs`, which it stores and never reads.
        """
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known or value is None:
                continue
            kwargs[key] = value
        for name in ("architectures", "hybrid_layer_pattern"):
            if name in kwargs and isinstance(kwargs[name], (list, tuple)):
                kwargs[name] = tuple(kwargs[name])
        if "moe_layer_freq" in kwargs:
            freq = kwargs["moe_layer_freq"]
            kwargs["moe_layer_freq"] = tuple(freq) if isinstance(freq, (list, tuple)) else freq
        return cls(**kwargs)


@dataclass(frozen=True)
class MimoV2QuantSpec:
    """The checkpoint's mixed quantization: FP8 dense, MXFP4 experts, BF16 the rest.

    This is not one scheme with exceptions, it is three, and they are selected by
    *where* a tensor lives rather than by a single flag:

    * Dense attention and the dense FFN linears are FP8 E4M3 with a 128x128
      block-scaled inverse (``<name>.weight_scale_inv``). ``fmt`` and
      ``weight_block_size`` describe these.
    * The routed experts are MXFP4: E2M1 pairs in one byte each, with an E8M0
      scale per 32 weights (``<name>.weight_scale``). ``store_dtype`` and
      ``mxfp4_block_size`` describe these.
    * Everything in ``ignored_layers`` -- all 48 attention ``o_proj``, plus the
      norms, the router, the embedding and the head -- is stored unquantized. The
      48 entries are the same tensor set as the attention ``o_proj``, which is why
      the count matters: a reader that quantizes ``o_proj`` reads FP8 where BF16
      is stored.

    Both scale tensors multiply a decoded code, but they differ in more than
    granularity: the FP8 inverse is a float, exact under a multiply, while the
    MXFP4 scale is an E8M0 *exponent byte* whose value is ``2 ** (byte - 127)``.
    Treating the second as a float is the classic way to get a quietly wrong model.
    """

    method: str | None = None
    fmt: str | None = None
    activation_scheme: str | None = None
    weight_block_size: tuple[int, int] = (128, 128)
    mxfp4_block_size: int | None = None
    store_dtype: str | None = None
    ignored_layers: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "MimoV2QuantSpec | None":
        if not raw:
            return None
        block = raw.get("weight_block_size") or [128, 128]
        size = raw.get("mxfp4_block_size")
        return cls(
            method=raw.get("quant_method"),
            fmt=raw.get("fmt"),
            activation_scheme=raw.get("activation_scheme"),
            weight_block_size=(int(block[0]), int(block[1])),
            mxfp4_block_size=None if size is None else int(size),
            store_dtype=raw.get("store_dtype"),
            ignored_layers=tuple(raw.get("ignored_layers") or ()),
        )

    @property
    def experts_are_mxfp4(self) -> bool:
        return self.store_dtype in ("mxfp4", "fp4") or self.mxfp4_block_size is not None

    @property
    def dense_is_fp8(self) -> bool:
        return (self.method or "").lower() == "fp8"

    @property
    def ignored_layer_scoped(self) -> tuple[str, ...]:
        """The `ignored_layers` entries that name a specific backbone layer.

        The released list has 49 entries of which 48 are
        `model.layers.<i>.self_attn.o_proj`; the 49th is `model.decoder.self_attn.o_proj`,
        which uses a prefix the checkpoint never uses and therefore matches no
        tensor. Keeping the two apart is what makes the count meaningful -- the 48
        are the attention `o_proj` of every layer, which is a claim about the
        weight layout, while the 49th is a typo that has survived release.
        """
        return tuple(name for name in self.ignored_layers if _is_layer_scoped(name))

    @property
    def ignored_o_proj_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.ignored_layer_scoped if name.endswith(".self_attn.o_proj"))

    def ignores(self, name: str) -> bool:
        """True when `name` is stored unquantized, per `ignored_layers`.

        Accepts a bare module path or one ending in a weight or scale suffix; the
        released file lists module paths (``model.layers.0.self_attn.o_proj``).
        """
        stripped = name
        for suffix in (".weight_scale_inv", ".weight_scale", ".weight"):
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)]
                break
        return stripped in self.ignored_layers

    def verify(self) -> list[str]:
        problems: list[str] = []
        if self.dense_is_fp8 and self.weight_block_size != (128, 128):
            problems.append(
                f"FP8 weight_block_size is {self.weight_block_size}, not (128, 128); the block "
                f"geometry is the kernel's index arithmetic, not a hint"
            )
        if self.experts_are_mxfp4 and self.mxfp4_block_size is None:
            problems.append("the experts are MXFP4 but mxfp4_block_size is unstated")
        if self.ignored_layers and not self.ignored_layer_scoped:
            problems.append(
                "no entry in ignored_layers is layer-scoped, so the list cannot be describing "
                "the per-layer unquantized tensors"
            )
        return problems

    def describe(self) -> str:
        parts = []
        if self.dense_is_fp8:
            parts.append(
                f"FP8 {self.fmt or 'e4m3'} dense, {self.weight_block_size[0]}x"
                f"{self.weight_block_size[1]} blocks"
            )
        if self.experts_are_mxfp4:
            parts.append(f"MXFP4 experts, block {self.mxfp4_block_size}")
        if self.ignored_layers:
            parts.append(f"{len(self.ignored_layers)} tensor(s) left stored")
        return ", ".join(parts) or "unquantized"


@dataclass(frozen=True)
class MimoV2DraftConfig:
    """The DFlash-style speculative drafter, read from `dflash/config.json`.

    A separate file with a different `model_type` (``qwen3``): the drafter is built
    from the Qwen3 components rather than the MiMo ones, which is why it reads
    ``rms_norm_eps`` where the backbone reads ``layernorm_epsilon``.

    Two facts here are load-bearing and neither is a default. ``is_causal`` is
    **False**: the drafter attends bidirectionally within a block, so it is not a
    causal language model and cannot be run through a causal attention path. And
    ``partial_rotary_factor`` is 0.5 over ``head_dim`` 128, giving ``rope_dim`` 64
    -- the same rotary width as the backbone, by coincidence of two different pairs
    of numbers, so a reader that copies one from the other gets the right answer
    for the wrong reason and breaks on any other checkpoint.
    """

    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    v_head_dim: int = 128
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    vocab_size: int = 152576
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    sliding_window: int | None = None
    use_sliding_window: bool = False
    layer_types: tuple[str, ...] = ()
    is_causal: bool = False
    attention_value_scale: float | None = None
    add_swa_attention_sink_bias: bool = False

    # -- dflash_config ----------------------------------------------------
    block_size: int = 8
    target_layer_ids: tuple[int, ...] = ()
    mask_token_id: int | None = None
    num_anchors: int | None = None
    num_target_layers: int | None = None
    target_hidden_size: int | None = None
    loss_decay_gamma: float | None = None

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads if self.num_key_value_heads else 0

    @property
    def qkv_out(self) -> int:
        """The drafter uses *separate* q/k/v projections, unlike the backbone."""
        return (
            self.num_attention_heads * self.head_dim
            + 2 * self.num_key_value_heads * self.head_dim
        )

    @property
    def context_width(self) -> int:
        """`fc`'s input: one hidden state per target layer, concatenated."""
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def all_layers_are_swa(self) -> bool:
        return bool(self.layer_types) and all(t == "sliding_attention" for t in self.layer_types)

    def verify(self) -> list[str]:
        problems: list[str] = []

        def want(condition: bool, message: str) -> None:
            if not condition:
                problems.append(message)

        want(self.num_hidden_layers > 0, f"num_hidden_layers is {self.num_hidden_layers}")
        want(
            self.num_key_value_heads > 0
            and self.num_attention_heads % self.num_key_value_heads == 0,
            f"num_attention_heads {self.num_attention_heads} is not a multiple of "
            f"num_key_value_heads {self.num_key_value_heads}",
        )
        want(
            self.rope_dim % 2 == 0,
            f"rope_dim is {self.rope_dim} from head_dim {self.head_dim} x "
            f"partial_rotary_factor {self.partial_rotary_factor}",
        )
        want(self.is_causal is False, "is_causal is True; the drafter attends bidirectionally")
        want(
            not self.use_sliding_window or self.sliding_window is not None,
            "use_sliding_window is set but sliding_window is not",
        )
        want(self.block_size > 0, f"block_size is {self.block_size}")
        want(
            self.layer_types == () or len(self.layer_types) == self.num_hidden_layers,
            f"layer_types has {len(self.layer_types)} entries for "
            f"{self.num_hidden_layers} layers",
        )
        if self.target_layer_ids:
            want(
                all(i >= 0 for i in self.target_layer_ids),
                f"target_layer_ids has a negative entry: {list(self.target_layer_ids)}",
            )
            want(
                len(set(self.target_layer_ids)) == len(self.target_layer_ids),
                f"target_layer_ids repeats: {list(self.target_layer_ids)}",
            )
            if self.num_target_layers is not None:
                want(
                    all(i < self.num_target_layers for i in self.target_layer_ids),
                    f"target_layer_ids {list(self.target_layer_ids)} reaches past "
                    f"num_target_layers {self.num_target_layers}",
                )
        else:
            problems.append(
                "target_layer_ids is unstated; the drafter's fc projection width is "
                "len(target_layer_ids) * hidden_size and nothing else determines it"
            )
        return problems

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MimoV2DraftConfig":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        # `dflash_config` is nested one level down and its keys take precedence,
        # because that block is where the drafter's own parameters live while the
        # top level is the Qwen3 component config it was assembled from.
        merged: dict[str, Any] = {**raw, **(raw.get("dflash_config") or {})}
        for key, value in merged.items():
            if key not in known or value is None:
                continue
            kwargs[key] = value
        for name in ("architectures", "layer_types", "target_layer_ids"):
            if name in kwargs and isinstance(kwargs[name], (list, tuple)):
                kwargs[name] = tuple(kwargs[name])
        return cls(**kwargs)


@dataclass(frozen=True)
class MimoV2Config:
    """The whole `config.json`: the text backbone plus what surrounds it."""

    text: MimoV2TextConfig
    quant: MimoV2QuantSpec | None = None
    #: Kept raw on purpose. The towers are out of scope for the text runtime, and
    #: normalizing them here would suggest otherwise.
    vision_config: Mapping[str, Any] = field(default_factory=dict)
    audio_config: Mapping[str, Any] = field(default_factory=dict)
    processor_config: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source: str | None = None) -> "MimoV2Config":
        return cls(
            text=MimoV2TextConfig.from_dict(raw),
            quant=MimoV2QuantSpec.from_dict(raw.get(_QUANT_BLOCK)),
            vision_config=raw.get(_VISION_BLOCK) or {},
            audio_config=raw.get(_AUDIO_BLOCK) or {},
            processor_config=raw.get(_PROCESSOR_BLOCK) or {},
            raw=raw,
            source=source,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "MimoV2Config":
        return load_config(os.path.join(model_dir, "config.json"))

    @classmethod
    def load_draft(cls, model_dir: str) -> "MimoV2DraftConfig | None":
        """The drafter's config, or None when the checkpoint does not ship one."""
        path = os.path.join(model_dir, _DRAFT_CONFIG)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return MimoV2DraftConfig.from_dict(json.load(handle))

    @property
    def has_vision(self) -> bool:
        return bool(self.vision_config)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_config)

    def verify(self) -> list[str]:
        problems = list(self.text.verify())
        if self.quant is None:
            return problems
        problems.extend(self.quant.verify())
        # The layer-scoped entries are the attention o_proj of every layer, so the
        # two counts have to agree or the quantization is being applied to a tensor
        # set the config does not describe. Counting only the layer-scoped ones keeps
        # a dangling entry out of this comparison, where it would otherwise read as
        # a 49-layer model.
        scoped = self.quant.ignored_o_proj_names
        if scoped and len(scoped) != self.text.num_hidden_layers:
            problems.append(
                f"ignored_layers names {len(scoped)} layer-scoped attention o_proj but the model "
                f"has {self.text.num_hidden_layers} layers"
            )
        return problems

    def dangling_ignored_layers(self) -> list[str]:
        """`ignored_layers` entries that name no layer and so match no tensor.

        Reported by `runtime_notes` rather than by `verify`: such an entry is
        inert, the checkpoint loads correctly with it present, and failing a load
        gate over it would be wrong. It is still worth surfacing, because it means
        `ignores()` cannot be trusted to catch anything outside the layer-scoped
        form -- the released file reaches its 49 entries by listing all 48 layers
        plus one prefix the checkpoint never uses.
        """
        if self.quant is None:
            return []
        return [name for name in self.quant.ignored_layers if not _is_layer_scoped(name)]

    def runtime_notes(self) -> list[str]:
        """Notes for whoever implements against this config.

        Kept separate from `verify` on purpose, and for two different reasons.
        Some notes are about this repository: whether its kernels can consume the
        layout, where a failure means a missing kernel rather than a bad
        checkpoint. The rest are about arithmetic the JSON does not state -- a
        field the reference stores and ignores, or a precision it chooses anyway --
        where the config is correct and a naive reader still gets the wrong answer.
        """
        notes: list[str] = []
        if self.quant is not None and self.quant.experts_are_mxfp4:
            block = self.quant.mxfp4_block_size
            if block == 32:
                notes.append(
                    "the MXFP4 expert layout (block 32, E8M0 scales) matches "
                    "`fp4_e2m1_e8m0_matvec_cuda`; the expert weights need no requantization"
                )
            else:
                notes.append(
                    f"the experts are MXFP4 with block {block}; "
                    f"`fp4_e2m1_e8m0_matvec_cuda` indexes scales at a block of 32"
                )
        if self.quant is not None and self.quant.ignored_layer_scoped:
            notes.append(
                f"{len(self.quant.ignored_layer_scoped)} tensor(s) are stored unquantized and are "
                f"read at their stored dtype"
            )
        if self.text.attention_value_scale is not None:
            notes.append(
                f"attention_value_scale is {self.text.attention_value_scale}; the value states "
                f"are scaled before attention, which no attention kernel here does"
            )
        if self.text.num_experts_per_tok and self.text.n_shared_experts is None:
            notes.append(
                "there is no shared expert: the reference's MoE builds only the routed experts and "
                "never reads `n_shared_experts`"
            )
        if self.text.moe_router_dtype is not None and self.text.moe_router_dtype != "float32":
            notes.append(
                f"moe_router_dtype is {self.text.moe_router_dtype!r}, but the reference's gate "
                f"upcasts both operands to fp32 before the linear, so the field describes storage "
                f"and not the routing arithmetic"
            )
        dangling = self.dangling_ignored_layers()
        if dangling:
            notes.append(
                f"ignored_layers has {len(dangling)} entry/entries naming no layer, so they match "
                f"no tensor and are inert: {dangling}"
            )
        if self.text.attention_chunk_size is not None:
            notes.append(
                f"attention_chunk_size is {self.text.attention_chunk_size}; it is not in the "
                f"reference's signature, so it is stored and never read"
            )
        if self.text.swa_layer_indices and self.text.add_swa_attention_sink_bias:
            notes.append(
                f"the {len(self.text.swa_layer_indices)} SWA layer(s) carry a per-head attention "
                f"sink bias and the GA layer(s) do not, so the sink has to be optional per layer"
            )
        if self.text.resolved_projection_layout == "fused_qkv":
            notes.append(
                "the projections are fused (`qkv_proj`), and the GA and SWA layers have "
                "different output widths"
            )
        towers = [
            name
            for name, present in (("vision", self.has_vision), ("audio", self.has_audio))
            if present
        ]
        if towers:
            notes.append(
                f"the checkpoint ships {' and '.join(towers)} weights; the text runtime does not "
                f"execute them"
            )
        return notes

    def describe(self) -> str:
        parts = [self.text.describe()]
        parts.append(self.quant.describe() if self.quant is not None else "unquantized")
        if self.has_vision:
            parts.append("vision tower present")
        if self.has_audio:
            parts.append("audio tower present")
        return "; ".join(parts)


def from_dict(raw: Mapping[str, Any], source: str | None = None) -> MimoV2Config:
    """Build the normalized config from a parsed `config.json`."""
    return MimoV2Config.from_dict(raw, source)


def from_pretrained(model_dir: str) -> MimoV2Config:
    """Read `<model_dir>/config.json`."""
    return MimoV2Config.from_pretrained(model_dir)


def load_config(path: str) -> MimoV2Config:
    """Read a config file, or a checkpoint directory that holds one."""
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path, encoding="utf-8") as handle:
        return MimoV2Config.from_dict(json.load(handle), source=path)


def describe(config: MimoV2Config) -> str:
    return config.describe()


def _shape_table(text: MimoV2TextConfig) -> str:
    """The per-family shape contract, which is the reason this module exists.

    One row per attention family, because the whole point is that the families
    differ; the layer lists are printed separately so a row can be read without
    having to reconstruct which layers it covers.
    """
    lines = [
        f"{'family':<4} {'layers':<22} {'Q':>3} {'KV':>3} {'head':>4} {'v':>4} {'rope':>4} "
        f"{'theta':>9} {'window':>6} {'sink':>4} {'qkv_out':>7} {'o_in':>6}  ffn"
    ]
    for label, indices in (
        ("GA", text.global_layer_indices),
        ("SWA", text.swa_layer_indices),
    ):
        if not indices:
            continue
        shape = text.attention(indices[0])
        span = f"{indices[0]}" if len(indices) == 1 else f"{indices[0]}-{indices[-1]}"
        kinds = "/".join(sorted({text.ffn_kind(i) for i in indices}))
        window = shape.sliding_window if shape.sliding_window is not None else "-"
        lines.append(
            f"{label:<4} {f'{span} ({len(indices)})':<22} {shape.num_q_heads:>3} "
            f"{shape.num_kv_heads:>3} {shape.head_dim:>4} {shape.v_head_dim:>4} "
            f"{shape.rope_dim:>4} {shape.rope_theta:>9.0f} {window:>6} "
            f"{('yes' if shape.has_sink else 'no'):>4} {shape.qkv_out:>7} {shape.o_in:>6}  {kinds}"
        )
    lines.append(f"{'':<4} GA layers: {list(text.global_layer_indices)}")
    lines.append(f"{'':<4} SWA layers: {list(text.swa_layer_indices)}")
    lines.append(f"{'':<4} dense FFN layers: {list(text.dense_layer_indices)}")
    lines.append(
        f"{'':<4} MoE layers: {len(text.moe_layer_indices)} of {text.num_hidden_layers}"
    )
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Read a config, print the shape contract, and check the config against itself.

    Exit status is 0 only when `verify()` finds nothing. `runtime_notes()` is
    printed either way and does not affect the status: a note is a statement about
    which kernels this repository has, not about whether the checkpoint is sound.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.models.mimo_v2.config",
        description="Normalize and check a MiMo-V2.6 config.",
    )
    parser.add_argument("path", help="config.json or a checkpoint directory")
    parser.add_argument("--json", help="write the normalized config and the problems here")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="also read and check dflash/config.json when the path is a directory",
    )
    args = parser.parse_args(argv)

    path = args.path
    if os.path.isdir(path):
        candidate = os.path.join(path, "config.json")
        if not os.path.exists(candidate):
            print(f"no config.json in {path}", file=sys.stderr)
            return 2
        checkpoint_dir, path = path, candidate
    else:
        checkpoint_dir = os.path.dirname(os.path.abspath(path))

    config = load_config(path)
    print(path)
    print(f"  {describe(config)}")
    print()
    print(_shape_table(config.text))
    print()

    draft = None
    if args.draft:
        draft = MimoV2Config.load_draft(checkpoint_dir)
        if draft is None:
            print(f"  no {_DRAFT_CONFIG} under {checkpoint_dir}")
        else:
            print(
                f"drafter: {draft.num_hidden_layers} layers, {draft.num_attention_heads}Q/"
                f"{draft.num_key_value_heads}KV, head {draft.head_dim}, rope {draft.rope_dim}, "
                f"qkv {draft.qkv_out}, block {draft.block_size}, window {draft.sliding_window}, "
                f"causal={draft.is_causal}, targets {list(draft.target_layer_ids)}, "
                f"mask token {draft.mask_token_id}"
            )

    problems = config.verify()
    if draft is not None:
        problems = problems + [f"drafter: {p}" for p in draft.verify()]
    for problem in problems:
        print(f"  [FAIL] {problem}")
    if problems:
        print(f"  {len(problems)} problem(s)")
    else:
        print("  [ok] the config is self-consistent")

    notes = config.runtime_notes()
    for note in notes:
        print(f"  [note] {note}")

    if args.json:
        payload: dict[str, Any] = {
            "path": path,
            "text": {f.name: _jsonable(getattr(config.text, f.name)) for f in fields(config.text)},
            "quant": (
                None
                if config.quant is None
                else {f.name: _jsonable(getattr(config.quant, f.name)) for f in fields(config.quant)}
            ),
            "problems": problems,
            "runtime_notes": notes,
        }
        if draft is not None:
            payload["draft"] = {f.name: _jsonable(getattr(draft, f.name)) for f in fields(draft)}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
