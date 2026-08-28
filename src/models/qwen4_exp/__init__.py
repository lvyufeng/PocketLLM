"""Qwen4-Exp (Qwen3.8-Flash-Next) inference support.

The checkpoint is a 335 GiB BF16 hybrid model: 48 decoder layers alternating
GatedDeltaNet linear attention (36x) with QSA sparse full attention (12x), a
4-stream hyper-connection residual, 512 routed experts at top-10, and a 95 GiB
hashed n-gram Per-Layer-Embedding table on layer 2.

Sizes drive the placement plan: routed experts (225 GiB) and the PLE table
(95 GiB) live in host RAM, everything else (~10 GiB) is tensor-parallel across
the four 2080 Ti cards.  See `placement.py`.
"""

from src.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig

__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig"]
