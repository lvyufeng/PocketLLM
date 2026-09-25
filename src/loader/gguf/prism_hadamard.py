"""Decode the ``prism.hadamard.*`` metadata block, the incoherence transform.

The Ternary-Bonsai-2-27B GGUF carries its own description of the Hadamard transform
that makes a 1.75-bit weight representable: which tensors were folded, what the
block size is, and the explicit sign vector each folded width consumes.  Catching a
mistake in it is the difference between a model that runs and a model that emits
fluent nonsense, so this module reads the block and refuses anything it has not been
shown the semantics of, rather than falling back to a default.

The semantics, read out of ``PrismML-Eng/llama.cpp`` at ``842b188`` (branch ``prism``)
and recorded in ``docs/architecture/ternary_bonsai_2_reference_gate.md``:

* the weights in the file are **pre-rotated**, ``W' = W . R^-1`` for
  ``R = (1/sqrt(N)) . H_N . diag(s)``, ``H_N[i][j] = (-1)^popcount(i ^ j)``;
* at run time the *activation* is rotated instead, as ``x |-> R x``: multiply by the
  signs and then apply the Walsh-Hadamard transform, blockwise along the last axis;
* exactly one declared tensor, ``token_embd.weight``, takes the **inverse** instead,
  because an embedding row is indexed rather than multiplied, so the rotation has to
  be undone after the lookup: ``(1/sqrt(N)) . s * (H . z)`` -- the signs and ``H`` in
  the opposite order.

Nothing here computes a transform.  It is the metadata contract the transform and the
kernel are built against, and it is validated on construction: a block whose version,
transform name, axis or sign bookkeeping is not the one that was read out of the fork
raises rather than parses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping

PREFIX = "prism.hadamard."

KEY_VERSION = PREFIX + "version"
KEY_TRANSFORM = PREFIX + "transform"
KEY_AXIS = PREFIX + "axis"
KEY_BLOCK_SIZE = PREFIX + "block_size"
KEY_SIGN_MODE = PREFIX + "sign_mode"
KEY_SIGN_WIDTHS = PREFIX + "sign_widths"
KEY_SIGN_VALUES = PREFIX + "sign_values"
KEY_WEIGHT_NAMES = PREFIX + "weight_names"
KEY_INVERSE_WEIGHT_NAMES = PREFIX + "inverse_weight_names"
KEY_GDN_V_GROUPED = PREFIX + "gdn_v_grouped"

#: The only metadata version whose semantics have been read out of the fork.
SUPPORTED_VERSION = 1
SUPPORTED_TRANSFORM = "normalized-sylvester-walsh-hadamard"
SUPPORTED_AXIS = "input-last-dimension"
SUPPORTED_SIGN_MODE = "explicit"

#: Width of the tensor ``gdn_v_grouped`` permutes into grouped order -- the
#: gated-DeltaNet output, ``value_heads * value_head_dim``.  The permute runs before
#: the signs, so the sign vector has to be declared for the grouped width.
GDN_GROUPED_WIDTH = 6144


class HadamardSpecError(ValueError):
    """The ``prism.hadamard.*`` block is absent, incomplete or not understood."""


@dataclass(frozen=True)
class HadamardSpec:
    """A validated ``prism.hadamard.*`` block.

    ``sign_values`` is the concatenation of one ``+/-1`` vector per entry of
    ``sign_widths``, in that order; use :meth:`signs_for_width` rather than slicing
    it by hand.
    """

    version: int
    transform: str
    axis: str
    block_size: int
    sign_mode: str
    sign_widths: tuple[int, ...]
    sign_values: tuple[int, ...]
    weight_names: tuple[str, ...]
    inverse_weight_names: tuple[str, ...]
    gdn_v_grouped: bool

    @property
    def folded_names(self) -> tuple[str, ...]:
        """Every declared tensor, forward and inverse, in declaration order."""
        return self.weight_names + self.inverse_weight_names

    def is_declared(self, tensor_name: str) -> bool:
        return tensor_name in self.weight_names or tensor_name in self.inverse_weight_names

    def takes_inverse(self, tensor_name: str) -> bool:
        """True when the rotation is undone after the lookup instead of applied to the input."""
        return tensor_name in self.inverse_weight_names

    def signs_for_width(self, width: int) -> tuple[int, ...]:
        """The sign vector declared for a folded width.

        ``width`` is the last dimension of the pre-rotation tensor, which is what
        ``axis = input-last-dimension`` means.  A width the block does not declare is
        an error: the signs are per-width and there is no sensible default.
        """
        offset = 0
        for declared in self.sign_widths:
            if declared == int(width):
                return tuple(self.sign_values[offset : offset + declared])
            offset += declared
        raise HadamardSpecError(
            f"no sign vector declared for width {width}; the block declares {self.sign_widths}"
        )


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise HadamardSpecError(f"missing metadata key {key!r}")
    return mapping[key]


def _as_int_tuple(value: Any, key: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise HadamardSpecError(
            f"{key} must be read with GGUFReader(..., read_arrays=True); got {type(value).__name__}"
        )
    return tuple(int(item) for item in value)


def _as_name_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise HadamardSpecError(
            f"{key} must be read with GGUFReader(..., read_arrays=True); got {type(value).__name__}"
        )
    return tuple(str(item) for item in value)


def has_hadamard_block(metadata: Mapping[str, Any]) -> bool:
    return any(str(key).startswith(PREFIX) for key in metadata)


def parse_hadamard_spec(
    metadata: Mapping[str, Any],
    *,
    known_tensor_names: Collection[str] | None = None,
) -> HadamardSpec | None:
    """Validate and decode ``prism.hadamard.*``.

    Returns ``None`` when the file declares no such block, which is every other
    checkpoint this repository reads -- an absence is not an error, but a *partial*
    block is.  Pass ``known_tensor_names`` to also assert that every declared tensor
    exists in the file, which is the check that catches a name list carried over from
    a different conversion.
    """
    if not has_hadamard_block(metadata):
        return None

    version = int(_require(metadata, KEY_VERSION))
    if version != SUPPORTED_VERSION:
        raise HadamardSpecError(
            f"{KEY_VERSION} is {version}, and only version {SUPPORTED_VERSION} has been read out "
            "of the reference implementation; re-read the fork before accepting it"
        )

    transform = str(_require(metadata, KEY_TRANSFORM))
    if transform != SUPPORTED_TRANSFORM:
        raise HadamardSpecError(f"{KEY_TRANSFORM} is {transform!r}, expected {SUPPORTED_TRANSFORM!r}")

    axis = str(_require(metadata, KEY_AXIS))
    if axis != SUPPORTED_AXIS:
        raise HadamardSpecError(f"{KEY_AXIS} is {axis!r}, expected {SUPPORTED_AXIS!r}")

    sign_mode = str(_require(metadata, KEY_SIGN_MODE))
    if sign_mode != SUPPORTED_SIGN_MODE:
        raise HadamardSpecError(
            f"{KEY_SIGN_MODE} is {sign_mode!r}, expected {SUPPORTED_SIGN_MODE!r}; a derived or "
            "random sign vector is not the same transform"
        )

    block_size = int(_require(metadata, KEY_BLOCK_SIZE))
    if block_size <= 0 or block_size & (block_size - 1):
        raise HadamardSpecError(f"{KEY_BLOCK_SIZE} is {block_size}, which is not a power of two")

    sign_widths = _as_int_tuple(_require(metadata, KEY_SIGN_WIDTHS), KEY_SIGN_WIDTHS)
    sign_values = _as_int_tuple(_require(metadata, KEY_SIGN_VALUES), KEY_SIGN_VALUES)
    if len(set(sign_widths)) != len(sign_widths):
        # signs_for_width would have to guess which vector a repeated width means.
        raise HadamardSpecError(f"{KEY_SIGN_WIDTHS} repeats a width: {sign_widths}")
    if sum(sign_widths) != len(sign_values):
        raise HadamardSpecError(
            f"{KEY_SIGN_VALUES} has {len(sign_values)} entries but {KEY_SIGN_WIDTHS} sums to "
            f"{sum(sign_widths)}"
        )
    unexpected = sorted({value for value in sign_values} - {1, -1})
    if unexpected:
        raise HadamardSpecError(f"{KEY_SIGN_VALUES} carries values other than +/-1: {unexpected[:8]}")

    weight_names = _as_name_tuple(_require(metadata, KEY_WEIGHT_NAMES), KEY_WEIGHT_NAMES)
    inverse_weight_names = _as_name_tuple(_require(metadata, KEY_INVERSE_WEIGHT_NAMES), KEY_INVERSE_WEIGHT_NAMES)
    if len(set(weight_names)) != len(weight_names):
        raise HadamardSpecError(f"{KEY_WEIGHT_NAMES} repeats a tensor name")
    overlap = sorted(set(weight_names) & set(inverse_weight_names))
    if overlap:
        raise HadamardSpecError(
            f"a tensor cannot take both the forward and the inverse transform: {overlap[:8]}"
        )

    gdn_v_grouped = bool(metadata.get(KEY_GDN_V_GROUPED, False))
    if gdn_v_grouped and GDN_GROUPED_WIDTH not in sign_widths:
        raise HadamardSpecError(
            f"{KEY_GDN_V_GROUPED} is set but {KEY_SIGN_WIDTHS} does not declare the grouped width "
            f"{GDN_GROUPED_WIDTH}: {sign_widths}"
        )

    if known_tensor_names is not None:
        known = set(known_tensor_names)
        missing = sorted(name for name in weight_names + inverse_weight_names if name not in known)
        if missing:
            raise HadamardSpecError(
                f"{len(missing)} declared tensors are not in the file, first: {missing[:8]}"
            )

    return HadamardSpec(
        version=version,
        transform=transform,
        axis=axis,
        block_size=block_size,
        sign_mode=sign_mode,
        sign_widths=sign_widths,
        sign_values=sign_values,
        weight_names=weight_names,
        inverse_weight_names=inverse_weight_names,
        gdn_v_grouped=gdn_v_grouped,
    )
