"""Quantization: the GGML block formats, decoded once and owned here.

This is a **leaf**: it depends on numpy and the GGML table header and on
nothing else in ``pocketllm`` -- not ``kernels``, not ``loader``, not
``backends``.  That placement is deliberate and load-bearing.  Two very
different layers need to turn a GGUF block into weights:

* the **loader**, which reads bytes off disk and must not have to know the
  reference backend exists;
* the **reference backend**, which is the normative implementation of every op
  and therefore has to dequantize a weight to compute with it.

If the decoders lived in either one, the other would import it, and
``backends/ -> kernels`` (the dependency rule) would be broken.  A shared leaf
with no edge back into the package is the only shape that keeps both honest.

The module split mirrors the formats' actual families:

``ggml_tables``
    the codebooks, parsed from the vendored ``ggml-common.h``.  Everything else
    here reads its constants from this module rather than restating them.
``iq4_nl``
    the 32-weight non-linear 4-bit block, and the nibble split ``IQ4_XS``
    shares.
``iq4_xs``
    the 256-weight k-quant built on that shared split.
``iq1``
    the IQ1 codebook and the 256-weight IQ1_M super-block.
``k_quants``
    the pure-arithmetic super-blocks -- ``q2_k`` through ``q6_k`` -- plus
    ``q8_0``.
``iq23``
    the sign-expanded super-blocks: ``iq2_xxs``, ``iq2_xs``, ``iq3_xxs``.
``formats``
    the one table from a GGUF type name to a block's geometry and decoder.

``formats`` is the module most callers want; the family modules exist so that
a decoder can be tested, and a block layout reasoned about, without a GGUF
file in hand.
"""

from __future__ import annotations

from . import formats, ggml_tables, iq1, iq23, iq4_nl, iq4_xs, k_quants

__all__ = ["formats", "ggml_tables", "iq1", "iq23", "iq4_nl", "iq4_xs", "k_quants"]