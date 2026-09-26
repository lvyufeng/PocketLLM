"""Xing4.0-29B-A4B, staged.

Stage 3 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388) is the
attention -- `config`, `rope` and the two forms of MLA.  Stage 4 is the
hyper-connection, `hyper_connection` and the `block` that wraps the sublayers in
it.  The MoE, serving and the write-up are [#393](https://github.com/lvyufeng/PocketLLM/issues/393),
so nothing in this package runs the model end to end yet.
"""

from src.models.xing4_0.block import DecoderLayer, DecoderLayerWeights, rms_norm
from src.models.xing4_0.config import Xing4_0Params, YarnParams, yarn_get_mscale
from src.models.xing4_0.attention import KVLatentCache, MLAAttention, MLAAttentionWeights
from src.models.xing4_0.hyper_connection import HyperConnection, HyperConnectionWeights, sinkhorn

__all__ = [
    "DecoderLayer",
    "DecoderLayerWeights",
    "HyperConnection",
    "HyperConnectionWeights",
    "KVLatentCache",
    "MLAAttention",
    "MLAAttentionWeights",
    "Xing4_0Params",
    "YarnParams",
    "rms_norm",
    "sinkhorn",
    "yarn_get_mscale",
]
