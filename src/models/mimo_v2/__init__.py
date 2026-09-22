"""MiMo-V2.6 (MiMo-V2.6-Flash-RL) model support.

The checkpoint is a 309B-parameter sparse MoE over a hybrid of two attention
families: 9 global-attention layers and 39 sliding-window ones, with a dense FFN
on layer 0 and 256 routed experts at top-8 everywhere else. The two families differ
in KV head count (4 against 8), in RoPE base (1e7 against 1e4), and in whether the
layer carries a per-head attention sink bias, so `config.attention(layer_idx)` is
the accessor that keeps a caller from deriving the shape once and being wrong on
two thirds of the stack.

The routed experts are MXFP4 at a block of 32 with E8M0 scales, which is the
layout `fp4_e2m1_e8m0_matvec_cuda` already consumes, so the expert weights need no
requantization. The dense linears are FP8 E4M3 under 128x128 block scales and the
attention `o_proj` is stored unquantized.

Scope, stated because the metadata advertises more than the runtime executes: the
vision tower, the audio encoders and the 1M-token context are not implemented, and
the MTP and DFlash drafters are not wired to a serving path.
"""

from src.models.mimo_v2.config import (
    MimoV2AttentionShape,
    MimoV2Config,
    MimoV2DraftConfig,
    MimoV2QuantSpec,
    MimoV2TextConfig,
    load_config,
)
from src.models.mimo_v2.layers import (
    MimoV2DecoderLayer,
    MimoV2ExpertWeights,
    MimoV2HostModel,
    MimoV2LayerWeights,
)

__all__ = [
    "MimoV2AttentionShape",
    "MimoV2Config",
    "MimoV2DecoderLayer",
    "MimoV2DraftConfig",
    "MimoV2ExpertWeights",
    "MimoV2HostModel",
    "MimoV2LayerWeights",
    "MimoV2QuantSpec",
    "MimoV2TextConfig",
    "load_config",
]
