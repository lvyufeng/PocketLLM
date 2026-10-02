"""The GGML lookup tables, parsed from one header instead of retyped.

Several GGUF formats are not arithmetic -- they are *codebook* formats.  A
nibble selects an entry from a fixed table of small integer weights, and the
only way to decode the block is to have that table byte-exact.  GGML ships
those tables in `ggml-common.h`, so this module reads that header as **text**
and extracts the tables by name.  Nothing here is compiled; the header is data.

The header is resolved in two steps, in order:

1. ``$POCKETLLM_GGML_COMMON`` -- an explicit override, for working against a
   different llama.cpp revision without editing this tree;
2. the copy vendored at ``loader/gguf/vendor/ggml-common.h``, whose sha256 is
   pinned below.  This is what a normal install uses.

There is deliberately no third step.  The header used to fall back to
``relic_core``'s copy, which made the kernel library a hidden dependency of the
loader -- the exact coupling vendoring exists to remove.  A phone or edge
install has no ``relic_core``, and a lookup that silently works on the
developer's machine and fails there is worse than one that fails loudly on both.
``POCKETLLM_GGML_COMMON`` is how a non-default revision is named.

Why a header at all, rather than Python literals: a codebook transcribed into
source is a second statement of a fact that already has an owner, and a
transcription error is invisible -- it produces plausible weights and a model
that is subtly wrong.  Reading the table removes that class of bug.  Every
codebook the tree needs is a ``GGML_TABLE_BEGIN`` block in this header, the IQ1
grid included.

Every accessor is cached and returns a **read-only** array: the tables are
shared by every decoder in the process, and one accidental in-place write would
corrupt all of them.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "HEADER_ENV",
    "header_path",
    "header_text",
    "iq1s_grid",
    "iq2s_grid",
    "iq2xs_grid",
    "iq2xxs_grid",
    "iq3s_grid",
    "iq3xxs_grid",
    "iq2xs_signed_grid",
    "iq2xxs_signed_grid",
    "iq3xxs_signed_grid",
    "kvalues_iq4nl",
    "kvalues_mxfp4",
]

#: Environment variable that overrides which header is read.
HEADER_ENV = "POCKETLLM_GGML_COMMON"

#: The vendored header's identity, as recorded in ``vendor/README.md``.  A local
#: edit to the vendored copy is refused rather than silently changing what every
#: decoder in the process reads.
VENDORED_SHA256 = "d09a7116254352959002c50efd0f0a6008bb6109d342f3285c94ad461f877d9a"

_VENDOR_RELATIVE = ("loader", "gguf", "vendor", "ggml-common.h")

_TABLE_RE = r"GGML_TABLE_BEGIN\([^,]+,\s*{name},\s*{size}\)(.*?)GGML_TABLE_END\(\)"
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_DECIMAL_RE = re.compile(r"-?\d+")
_COMMENT_RE = re.compile(r"//[^\n]*")


def header_path() -> Path:
    """The header this process will read, resolved once per process."""
    override = os.environ.get(HEADER_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise RuntimeError(f"{HEADER_ENV}={override} does not name a readable file")
        return path

    vendored = Path(__file__).resolve().parent.parent.joinpath(*_VENDOR_RELATIVE)
    if vendored.is_file():
        _verify_vendored(vendored)
        return vendored

    raise RuntimeError(
        "no GGML table header found: set "
        f"{HEADER_ENV}, or install the vendored copy at {vendored}"
    )


def _verify_vendored(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != VENDORED_SHA256:
        raise RuntimeError(
            f"{path} has sha256 {digest}, but the vendored header is pinned at "
            f"{VENDORED_SHA256}. Either re-vendor it and update VENDORED_SHA256, "
            f"or set {HEADER_ENV} to point somewhere else."
        )


@lru_cache(maxsize=1)
def header_text() -> str:
    return header_path().read_text(encoding="utf-8")


def _body(name: str, size: str) -> str:
    """The literal entries of one ``GGML_TABLE_BEGIN`` block, comments removed."""
    pattern = _TABLE_RE.format(name=re.escape(name), size=re.escape(size))
    match = re.search(pattern, header_text(), flags=re.S)
    if match is None:
        raise RuntimeError(f"failed to locate {name}[{size}] in {header_path()}")
    return _COMMENT_RE.sub("", match.group(1))


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@lru_cache(maxsize=None)
def _hex_table(name: str, size: str, expected: int, dtype: str) -> np.ndarray:
    values = [int(item, 16) for item in _HEX_RE.findall(_body(name, size))]
    if len(values) != expected:
        raise RuntimeError(f"{name} expected {expected} entries, got {len(values)}")
    return _readonly(np.asarray(values, dtype=np.dtype(dtype)))


@lru_cache(maxsize=None)
def _decimal_table(name: str, size: str, expected: int, dtype: str) -> np.ndarray:
    values = [int(item) for item in _DECIMAL_RE.findall(_body(name, size))]
    if len(values) != expected:
        raise RuntimeError(f"{name} expected {expected} entries, got {len(values)}")
    return _readonly(np.asarray(values, dtype=np.dtype(dtype)))


# -- raw codebooks -----------------------------------------------------------


def iq2xxs_grid() -> np.ndarray:
    """256 entries of eight 2-bit-signed weights, packed into a uint64."""
    return _hex_table("iq2xxs_grid", "256", 256, "<u8")


def iq2xs_grid() -> np.ndarray:
    """512 entries, packed like :func:`iq2xxs_grid`."""
    return _hex_table("iq2xs_grid", "512", 512, "<u8")


def iq2s_grid() -> np.ndarray:
    """1024 entries, packed like :func:`iq2xxs_grid`."""
    return _hex_table("iq2s_grid", "1024", 1024, "<u8")


def iq3xxs_grid() -> np.ndarray:
    """256 entries of eight 3-bit-signed weights, packed into a uint32."""
    return _hex_table("iq3xxs_grid", "256", 256, "<u4")


def iq3s_grid() -> np.ndarray:
    """512 entries, packed like :func:`iq3xxs_grid`."""
    return _hex_table("iq3s_grid", "512", 512, "<u4")


def iq1s_grid() -> np.ndarray:
    """2048 entries of eight ternary weights in {-1, 0, 1}, packed into a uint64."""
    return _hex_table("iq1s_grid", "NGRID_IQ1S", 2048, "<u8")


def kvalues_iq4nl() -> np.ndarray:
    """The 16-entry IQ4_NL codebook, as signed decimal literals.

    Unlike the grids above these are *not* hex, which is why this goes through a
    separate reader.  The values are deliberately not evenly spaced: the table is
    the format.
    """
    return _decimal_table("kvalues_iq4nl", "16", 16, "int8")


def kvalues_mxfp4() -> np.ndarray:
    """The 16-entry MXFP4 codebook: two interleaved sign/magnitude halves."""
    return _decimal_table("kvalues_mxfp4", "16", 16, "int8")


# -- sign-expanded lookups ---------------------------------------------------


@lru_cache(maxsize=1)
def _sign_matrix_128() -> np.ndarray:
    """The 128 sign patterns an i-quant sign index selects, as ``(128, 8)`` +-1.

    Index `i` covers eight signs; bit 7 of the mask is the parity of `i`'s
    low bits, which is how GGML folds one more sign bit into the same byte.
    """
    index = np.arange(128, dtype=np.uint16)
    parity = np.array([i.bit_count() & 1 for i in range(128)], dtype=np.uint16)
    mask = index | (parity << 7)
    bits = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint16)
    return np.where((mask[:, None] & bits[None, :]) != 0, -1, 1).astype(np.int8)


def _unpack_rows(packed: np.ndarray, width: int) -> np.ndarray:
    """``(n,)`` packed integers -> ``(n, width)`` int8, little-endian byte order."""
    as_bytes = packed.astype(f"<u{packed.dtype.itemsize}").tobytes()
    return np.frombuffer(as_bytes, dtype=np.int8).reshape(len(packed), width)


@lru_cache(maxsize=1)
def iq2xxs_signed_grid() -> np.ndarray:
    """``(256, 128, 8)``: every IQ2_XXS codebook entry under every sign pattern."""
    base = _unpack_rows(iq2xxs_grid(), 8)
    return _readonly(_signed(base))


@lru_cache(maxsize=1)
def iq2xs_signed_grid() -> np.ndarray:
    """``(512, 128, 8)``: every IQ2_XS codebook entry under every sign pattern."""
    base = _unpack_rows(iq2xs_grid(), 8)
    return _readonly(_signed(base))


@lru_cache(maxsize=1)
def iq3xxs_signed_grid() -> np.ndarray:
    """``(256, 128, 8)``: IQ3_XXS, whose eight signs come in two half-groups.

    The format spends its sign index on two 4-wide groups with independent sign
    bits, so the expansion splits each 4-byte codebook entry across the two
    halves of the sign pattern rather than repeating one 8-wide apply.
    """
    base = _unpack_rows(iq3xxs_grid(), 4)
    signs = _sign_matrix_128().astype(np.int16)
    expanded = np.empty((256, 128, 8), dtype=np.int8)
    # The 4-wide codebook entry is applied once per sign half, so the same four
    # weights appear under sign bits 0..3 and again under bits 4..7.
    for half in (slice(0, 4), slice(4, 8)):
        product = base[:, None, :].astype(np.int16) * signs[None, :, half]
        expanded[:, :, half] = product.astype(np.int8)
    return _readonly(expanded)


def _signed(base: np.ndarray) -> np.ndarray:
    signs = _sign_matrix_128().astype(np.int16)
    product = base[:, None, :].astype(np.int16) * signs[None, :, :]
    return product.astype(np.int8)