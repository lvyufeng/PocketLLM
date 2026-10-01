from __future__ import annotations

from dataclasses import dataclass

from pocketllm.kernels.buffer import Buffer
from pocketllm.kernels.device import Device


@dataclass(frozen=True)
class QuantizedGGUFTensor:
    """Device-resident raw GGUF quantized matrix blocks.

    The tensor stores raw GGUF blocks, not dequantized weights.  ``row_start`` is
    non-zero for row-sliced tensors such as a sharded vocab/lm_head.

    ``blocks`` is a device :class:`~pocketllm.kernels.buffer.Buffer` and
    ``device`` says where it lives; the loader never holds a torch tensor, so a
    backend that is not torch-based can consume what this describes.  The block
    geometry itself is recovered from ``type_name`` and the shape, which is why
    neither the buffer nor the device has to carry it.
    """

    source_name: str
    blocks: Buffer
    device: Device
    type_name: str
    type_id: int
    row_elems: int
    out_dim: int
    row_start: int = 0

    @property
    def nbytes(self) -> int:
        return int(self.blocks.nbytes)