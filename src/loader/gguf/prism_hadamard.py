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
  ``R = (1/sqrt(N)) . H_N . diag(s)``, ``H_N[i][j] = (-1)^popcount(i AND j)``;
* at run time the *activation* is rotated instead, as ``x |-> R x``: multiply by the
  signs and then apply the Walsh-Hadamard transform, blockwise along the last axis;
* exactly one declared tensor, ``token_embd.weight``, takes the **inverse** instead,
  because an embedding row is indexed rather than multiplied, so the rotation has to
  be undone after the lookup: ``(1/sqrt(N)) . s * (H . z)`` -- the signs and ``H`` in
  the opposite order.

This module holds both halves of that contract: the metadata, which is *what* the
transform is and is validated rather than defaulted, and :class:`HadamardRotation`,
which *computes* it.  They are one module because they define each other -- a
sign-order or block-size change that lands in one and not the other produces a model
that still runs and still generates, which is the failure this checkpoint is most
exposed to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping

import torch

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


# --------------------------------------------------------------------------- #
# The transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GdnGeometry:
    """The gated-DeltaNet feature geometry the ``gdn_v_grouped`` permute needs.

    Read out of the model's hyperparameters rather than out of ``prism.hadamard``:
    ``value_heads`` is ``ssm.time_step_rank`` -- 48 for this checkpoint, the width of
    the DeltaNet value output divided by its head -- and ``groups`` is
    ``ssm.group_count``, 16.  The activation arrives with its feature axis in *tiled*
    head order ``[head_dim, groups, rep]`` and the fold was computed in *grouped*
    order ``[head_dim, rep, groups]``, where ``rep = value_heads / groups`` is 3.

    Both orders multiply out to the same width, which is what makes a wrong geometry
    quiet: the result is not short, not scaled, and not obviously wrong -- it has the
    right features in the wrong places inside each head.  The fork's own comment on
    the permute is the source for the two orders; see
    ``docs/architecture/ternary_bonsai_2_reference_gate.md``.
    """

    value_heads: int
    groups: int

    def __post_init__(self) -> None:
        if self.value_heads <= 0 or self.groups <= 0:
            raise HadamardSpecError(f"gdn geometry must be positive, got {self!r}")
        if self.value_heads % self.groups:
            raise HadamardSpecError(
                f"gdn value heads {self.value_heads} are not a multiple of {self.groups} groups"
            )

    @property
    def rep(self) -> int:
        return self.value_heads // self.groups

    def head_dim(self, width: int) -> int:
        if int(width) % self.value_heads:
            raise HadamardSpecError(
                f"gdn width {width} is not divisible by {self.value_heads} value heads"
            )
        return int(width) // self.value_heads


#: This checkpoint's numbers, from ``qwen35.ssm.time_step_rank`` (48) and
#: ``qwen35.ssm.group_count`` (16) in its own header.
DEFAULT_GDN_GEOMETRY = GdnGeometry(value_heads=48, groups=16)


def gdn_group_permute(x: torch.Tensor, geometry: GdnGeometry = DEFAULT_GDN_GEOMETRY) -> torch.Tensor:
    """Reorder a gated-DeltaNet activation from tiled head order to grouped order.

    ``[head_dim, groups, rep] -> [head_dim, rep, groups]``, in the fork's own axis
    order, which makes this a transpose of the last-but-one and last dimensions of
    the flat feature vector read as ``(rep, groups, head_dim)``.  It is its own
    inverse -- the fork applies the same swap to the weight side at load.
    """
    width = int(x.shape[-1])
    head_dim = geometry.head_dim(width)
    lead = x.shape[:-1]
    tiled = x.reshape(*lead, geometry.rep, geometry.groups, head_dim)
    grouped = tiled.transpose(-3, -2).contiguous()
    return grouped.reshape(*lead, width)


def _hadamard_last(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized in-place butterfly along the last dimension, length a power of two.

    The same radix-2 pass order the fork's FWHT kernel uses, so the two agree to the
    last bit on fp32 input and not merely to within rounding.
    """
    n = int(x.shape[-1])
    if n & (n - 1):
        raise HadamardSpecError(f"hadamard block length {n} is not a power of two")
    lead = x.shape[:-1]
    step = 1
    while step < n:
        blocks = x.reshape(*lead, n // (2 * step), 2, step)
        low, high = blocks[..., 0, :], blocks[..., 1, :]
        x = torch.cat((low + high, low - high), dim=-1).reshape(*lead, n)
        step *= 2
    return x


def walsh_hadamard_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """``(1/sqrt(N)) H x``, applied independently to each ``block_size`` run of the last axis.

    ``H[i][j] = (-1)^popcount(i AND j)`` is the natural-order (Sylvester) Hadamard
    matrix, which is what the butterfly computes.  The scale is applied to the input
    before the butterflies rather than to the output after them: that is the fork's
    order (``dst = src * scale`` then the passes) and it is the reason the two agree
    bit for bit instead of nearly.
    """
    width = int(x.shape[-1])
    if int(block_size) <= 0 or int(block_size) & (int(block_size) - 1):
        raise HadamardSpecError(f"hadamard block size {block_size} is not a power of two")
    if width % block_size:
        raise HadamardSpecError(f"width {width} is not a multiple of block size {block_size}")
    lead = x.shape[:-1]
    blocks = x.reshape(*lead, width // int(block_size), int(block_size))
    scale = 1.0 / float(int(block_size)) ** 0.5
    blocks = _hadamard_last(blocks * scale)
    return blocks.reshape(*lead, width)


class HadamardRotation:
    """The activation-side transform for a file that carries a ``prism.hadamard`` block.

    Holds the spec and the sign vectors sliced out of it, per width, so the hot path
    is a multiply, a butterfly and a transpose.  The dtype is a parameter and defaults
    to fp32 because that is what the fork computes in -- the rotation of a 1x5120
    decode row is cheap, and doing it in fp16 would put rounding error into the one
    place the 1.75-bit weights cannot afford it.

    The two directions are not each other's transpose-with-a-scale, and the file says
    which tensors take which:

    * ``forward`` -- ``(1/sqrt(N)) H (s * x)``, signs *then* ``H``.  Every folded
      matrix weight takes this, including ``output.weight``.
    * ``inverse`` -- ``(1/sqrt(N)) s * (H x)``, ``H`` *then* signs.  ``token_embd``'s
      rows are indexed rather than multiplied, so its rotation is undone after the
      lookup, and undoing it means the opposite order.
    """

    def __init__(
        self,
        spec: HadamardSpec,
        *,
        gdn: GdnGeometry | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spec = spec
        self.gdn = DEFAULT_GDN_GEOMETRY if gdn is None else gdn
        self.device = device
        self.dtype = dtype
        self._signs: dict[int, torch.Tensor] = {}

    @property
    def block_size(self) -> int:
        return self.spec.block_size

    def signs_for(self, width: int) -> torch.Tensor:
        """The declared sign vector for a folded width, materialized once."""
        signs = self._signs.get(int(width))
        if signs is None:
            signs = torch.tensor(
                self.spec.signs_for_width(int(width)), dtype=self.dtype, device=self.device
            )
            self._signs[int(width)] = signs
        return signs

    def _signs_for_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.signs_for(int(x.shape[-1])).to(dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x |-> (1/sqrt(N)) H (s * x)``: the folded weights' activation transform."""
        return walsh_hadamard_blocks(x * self._signs_for_activation(x), self.block_size)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """``z |-> (1/sqrt(N)) s * (H z)``: the token embedding's, applied after lookup."""
        return walsh_hadamard_blocks(x, self.block_size) * self._signs_for_activation(x)

    def apply(self, tensor_name: str, x: torch.Tensor) -> torch.Tensor:
        """The pipeline a declared tensor sees, in the fork's order.

        Permute first (only for the gated-DeltaNet output, and only when the block says
        so), then the signs and the rotation -- or the inverse, alone, for a tensor the
        block declares inverse.
        """
        if not self.spec.is_declared(tensor_name):
            raise HadamardSpecError(
                f"{tensor_name} is not declared in {KEY_WEIGHT_NAMES} or {KEY_INVERSE_WEIGHT_NAMES}; "
                "a folded weight is a fact about the file, not something to infer from a name"
            )
        if self.spec.takes_inverse(tensor_name):
            return self.inverse(x)
        if self.spec.gdn_v_grouped and ".ssm_out." in tensor_name:
            x = gdn_group_permute(x, self.gdn)
        return self.forward(x)
