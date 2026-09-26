"""Xing4.0-29B-A4B, staged.

Stage 3 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388) is the
attention: `config`, `rope` and the two forms of MLA.  The hyper-connection
(`hc_*`) and the MoE are later tasks and are not here yet, so nothing in this
package runs the model end to end.
"""

from src.models.xing4_0.config import Xing4_0Params, YarnParams, yarn_get_mscale
from src.models.xing4_0.attention import KVLatentCache, MLAAttention, MLAAttentionWeights

__all__ = [
    "KVLatentCache",
    "MLAAttention",
    "MLAAttentionWeights",
    "Xing4_0Params",
    "YarnParams",
    "yarn_get_mscale",
]
