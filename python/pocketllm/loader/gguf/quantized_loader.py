from __future__ import annotations

from pathlib import Path

import numpy as np

from pocketllm.kernels.backend import BackendSession
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import Tensor
from pocketllm.loader.gguf.bundle import GGUFBundle, GGUFTensorRef, read_gguf_bundle
from pocketllm.loader.gguf.host_array import dtype_for, upload
from pocketllm.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
from pocketllm.loader.gguf.quantized_tensor import QuantizedGGUFTensor
from pocketllm.loader.gguf.tensor_reader import GGUFTensorDataReader
from pocketllm.quant.iq4_nl import fold_to_runtime_span

#: The ABI element type each GGUF dense type decodes to.  ``bf16`` maps to f32
#: because numpy has no bfloat16 dtype: the reader widens it into an f32 carrier
#: and the caller narrows on the way to the device if it wants to.  That keeps
#: "what the file holds" and "what the backend is handed" separately named.
_DENSE_ABI_DTYPE = {
    "f32": DType.F32,
    "f16": DType.F16,
    "i32": DType.I32,
    "bf16": DType.F32,
}


class GGUFQuantizedTensorLoader:
    """Read dense and raw quantized GGUF tensors into device memory.

    The loader owns GGUF file readers and format/type checks only; it hands
    bytes to a :class:`~pocketllm.kernels.backend.BackendSession` and receives
    buffers back.  Kernel invocation lives in the backend, never here.
    """

    def __init__(self, bundle_or_path: GGUFBundle | str | Path, *, session: BackendSession):
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        self.session = session
        self._readers: dict[str, GGUFTensorDataReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "GGUFQuantizedTensorLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def tensor_ref(self, name: str) -> GGUFTensorRef:
        try:
            return self.bundle.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def reader_for(self, tensor: GGUFTensorRef) -> GGUFTensorDataReader:
        reader = self._readers.get(tensor.shard_path)
        if reader is None:
            reader = GGUFTensorDataReader(tensor.shard_path)
            self._readers[tensor.shard_path] = reader
        return reader

    def read_dense(self, name: str, *, dtype: DType = DType.F32) -> Tensor:
        """Read a dense tensor and upload it as ``dtype``.

        The decoder's own type is the carrier; ``dtype`` is what the caller
        wants resident.  Narrowing happens before upload, so a device never has
        to hold a wider copy of a weight than it asked for.
        """
        tensor = self.tensor_ref(name)
        values = self.reader_for(tensor).read_tensor(tensor.name)
        native = _DENSE_ABI_DTYPE[tensor.type_name]
        if dtype is not native:
            values = _convert(values, native, dtype)
        return upload(self.session, values)

    def read_quant(self, name: str, expected_type: str) -> QuantizedGGUFTensor:
        tensor = self.tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self.reader_for(tensor).read_quantized_matrix_blocks(tensor.name)
        return self._to_quantized_tensor(name, blocks, type_name, row_elems, expected_type, row_start=0)

    def read_quant_rows(
        self,
        name: str,
        expected_type: str,
        row_start: int,
        row_count: int,
    ) -> QuantizedGGUFTensor:
        tensor = self.tensor_ref(name)
        if tensor.type_name != expected_type:
            raise ValueError(f"{name} expected {expected_type}, got {tensor.type_name}")
        blocks, type_name, row_elems = self.reader_for(tensor).read_quantized_matrix_block_rows(
            tensor.name,
            int(row_start),
            int(row_count),
        )
        return self._to_quantized_tensor(name, blocks, type_name, row_elems, expected_type, row_start=int(row_start))

    def _to_quantized_tensor(
        self,
        name: str,
        blocks: np.ndarray,
        type_name: str,
        row_elems: int,
        expected_type: str,
        *,
        row_start: int,
    ) -> QuantizedGGUFTensor:
        if type_name != expected_type:
            raise RuntimeError(f"{name} reader type mismatch: expected={expected_type} got={type_name}")
        if type_name == "iq4_nl":
            blocks = fold_to_runtime_span(blocks, row_elems)
        try:
            type_id = GGUF_DENSE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(f"GGUF type {type_name!r} is not supported by the GGUF raw-block runtime") from exc
        device_blocks = upload(self.session, np.ascontiguousarray(blocks))
        return QuantizedGGUFTensor(
            source_name=name,
            blocks=device_blocks.buffer,
            device=device_blocks.buffer.device,
            type_name=type_name,
            type_id=int(type_id),
            row_elems=int(row_elems),
            out_dim=int(blocks.shape[0]),
            row_start=int(row_start),
        )


def _convert(values: np.ndarray, native: DType, target: DType) -> np.ndarray:
    """Narrow or widen a dense array from its decoded type to the requested one.

    Only the conversions a GGUF file actually needs are implemented; anything
    else is refused rather than silently reinterpreted, because a wrong cast on
    weights is invisible until the logits are wrong.
    """
    if native is target:
        return values
    if target is DType.F32:
        return values.astype(np.float32)
    if target is DType.F16:
        return values.astype(np.float16)
    raise TypeError(f"no {native.value} -> {target.value} conversion")