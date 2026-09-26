"""The Xing4.0-29B-A4B text config, as the checkpoint's own `config.json` states it.

Read from the released `configuration_xing4_0.py` rather than inferred, because
three of its defaults are load-bearing and invisible in `config.json` itself:

- ``head_dim = qk_rope_head_dim``, i.e. **64**, not ``hidden_size / num_heads``.
  It is the width the rotary frequencies are computed over, so a port that
  derives head_dim the usual way gets a 112-wide frequency table for a 64-wide
  rotary slice.
- ``rope_interleave = True``, which selects the interleaved rotary layout.  See
  `src/models/xing4_0/rope.py` -- the layout is a real fork in the road and the
  checkpoint takes the less common one.
- ``qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`` is derived, and it is the
  number the attention scale takes the inverse square root of (192, not 128).

The class is deliberately a plain dataclass rather than a `PretrainedConfig`
subclass: nothing in this repository builds a `transformers` model object, and
the fields here are the ones the arithmetic reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["YarnParams", "Xing4_0Params", "yarn_get_mscale"]


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """The YaRN magnitude scale, as the checkpoint's own remote code defines it.

    `modeling_xing4_0.py` carries this function twice over: once as its own
    module-level `yarn_get_mscale`, used for the attention scale, and once
    inline in `transformers`' `_compute_yarn_parameters`, where it is used as a
    *ratio* of two calls and therefore cancels.  This is the first of the two.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass(frozen=True)
class YarnParams:
    """The `rope_scaling` block, with the keys the arithmetic uses and no others."""

    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float
    truncate: bool = True

    @classmethod
    def from_dict(cls, block: dict, *, max_position_embeddings: int) -> "YarnParams":
        return cls(
            factor=float(block["factor"]),
            original_max_position_embeddings=int(
                block.get("original_max_position_embeddings") or max_position_embeddings
            ),
            beta_fast=float(block.get("beta_fast") or 32),
            beta_slow=float(block.get("beta_slow") or 1),
            mscale=float(block.get("mscale") or 0.0),
            mscale_all_dim=float(block.get("mscale_all_dim") or 0.0),
            truncate=bool(block.get("truncate", True)),
        )


@dataclass(frozen=True)
class Xing4_0Params:
    """Everything the attention, the hyper-connection and the MoE read."""

    n_layers: int
    hidden_size: int
    vocab_size: int
    context_length: int
    n_heads: int

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rope_interleave: bool
    yarn: YarnParams

    rms_norm_eps: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    hc_clamp_min: float
    hc_clamp_max: float

    intermediate_size: int
    moe_intermediate_size: int
    first_k_dense_replace: int
    n_routed_experts: int
    n_shared_experts: int
    n_experts_per_tok: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    scoring_func: str
    nextn_layers: int

    @property
    def qk_head_dim(self) -> int:
        """The full per-head query/key width: 128 nope + 64 rope = 192."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def head_dim(self) -> int:
        """The rotary width, which is what the frequency table spans."""
        return self.qk_rope_head_dim

    @property
    def attention_scale(self) -> float:
        """`1/sqrt(qk_head_dim)` times the YaRN magnitude scale, squared.

        From `Xing4_0Attention.__init__`: when `mscale_all_dim` is set, the scale
        is multiplied by `yarn_get_mscale(factor, mscale_all_dim) ** 2`.  The
        square is not a typo in the checkpoint and not one here: with this
        config's `factor 64` and `mscale_all_dim 1.0` it is `1.4159 ** 2`, which
        is a 2x change to every logit.  cos/sin are *not* scaled by it as well --
        that is the other `mscale`, and it cancels in the ratio transformers
        computes.
        """
        scale = self.qk_head_dim ** -0.5
        if self.yarn.mscale_all_dim:
            mscale = yarn_get_mscale(self.yarn.factor, self.yarn.mscale_all_dim)
            scale *= mscale * mscale
        return scale

    @classmethod
    def from_config(cls, raw: dict) -> "Xing4_0Params":
        qk_nope = int(raw["qk_nope_head_dim"])
        qk_rope = int(raw["qk_rope_head_dim"])
        return cls(
            n_layers=int(raw["num_hidden_layers"]),
            hidden_size=int(raw["hidden_size"]),
            vocab_size=int(raw["vocab_size"]),
            context_length=int(raw["max_position_embeddings"]),
            n_heads=int(raw["num_attention_heads"]),
            q_lora_rank=int(raw["q_lora_rank"]),
            kv_lora_rank=int(raw["kv_lora_rank"]),
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=int(raw["v_head_dim"]),
            rope_theta=float(raw["rope_theta"]),
            # The released `config.json` does not carry it; the config class
            # defaults it to True, which is the checkpoint's own default.
            rope_interleave=bool(raw.get("rope_interleave", True)),
            yarn=YarnParams.from_dict(
                raw["rope_scaling"], max_position_embeddings=int(raw["max_position_embeddings"])
            ),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            hc_mult=int(raw["hc_mult"]),
            hc_sinkhorn_iters=int(raw["hc_sinkhorn_iters"]),
            hc_eps=float(raw["hc_eps"]),
            hc_clamp_min=float(raw["mhc_h_res_clamp_min"]),
            hc_clamp_max=float(raw["mhc_h_res_clamp_max"]),
            intermediate_size=int(raw["intermediate_size"]),
            moe_intermediate_size=int(raw["moe_intermediate_size"]),
            first_k_dense_replace=int(raw["first_k_dense_replace"]),
            n_routed_experts=int(raw["n_routed_experts"]),
            n_shared_experts=int(raw["n_shared_experts"]),
            n_experts_per_tok=int(raw["num_experts_per_tok"]),
            routed_scaling_factor=float(raw["routed_scaling_factor"]),
            n_group=int(raw["n_group"]),
            topk_group=int(raw["topk_group"]),
            norm_topk_prob=bool(raw["norm_topk_prob"]),
            scoring_func=str(raw["scoring_func"]),
            nextn_layers=int(raw.get("num_nextn_predict_layers", 0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Xing4_0Params":
        return cls.from_config(json.loads(Path(path).read_text(encoding="utf-8")))
