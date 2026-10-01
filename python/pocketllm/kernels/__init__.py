"""The portable kernel ABI.

This package is the one thing every device backend agrees on.  It is deliberately
poor: no numpy, no torch, no I/O, no kernel.  It describes tensors and buffers,
declares the op vocabulary, and defines what a backend must be able to say about
itself.  A backend implements it; the engine drives it.

Importing this package declares the core op vocabulary and freezes it.
"""

from __future__ import annotations

from .backend import (
    Backend,
    BackendSession,
    Capability,
    CapturedGraph,
    CompiledGraph,
    CompileSpec,
    GraphCapability,
    GraphMode,
    RegionGranularity,
)
from .buffer import Buffer, DeviceBuffer, HostBuffer
from .device import Device, KNOWN_DEVICE_KINDS, parse_device
from .dispatch import Dispatcher, Resolution, ResolvedOp
from .dtypes import QUANT_FORMATS, DType, QuantFormat, quant_format, quant_format_by_file_id
from .errors import (
    BackendNotImplementedError,
    KernelError,
    NoBackendError,
    OpNotDeclaredError,
    ShapeError,
)
from .graph import Graph, GraphRegion, Node, Value
from .registry import OPS, OpRegistry
from .schema import ArgSpec, Kind, OpSchema
from .tensor import Tensor, TensorDesc

# Importing the families populates OPS and freezes it.  Kept last so every name
# above exists before any schema is declared.
from . import ops as _ops  # noqa: E402,F401

__all__ = [
    "ArgSpec",
    "Backend",
    "BackendNotImplementedError",
    "BackendSession",
    "Buffer",
    "Capability",
    "CapturedGraph",
    "CompileSpec",
    "CompiledGraph",
    "Device",
    "DeviceBuffer",
    "Dispatcher",
    "DType",
    "Graph",
    "GraphCapability",
    "GraphMode",
    "GraphRegion",
    "HostBuffer",
    "KNOWN_DEVICE_KINDS",
    "KernelError",
    "Kind",
    "NoBackendError",
    "Node",
    "OPS",
    "OpNotDeclaredError",
    "OpRegistry",
    "OpSchema",
    "QUANT_FORMATS",
    "QuantFormat",
    "RegionGranularity",
    "Resolution",
    "ResolvedOp",
    "ShapeError",
    "Tensor",
    "TensorDesc",
    "Value",
    "parse_device",
    "quant_format",
    "quant_format_by_file_id",
]