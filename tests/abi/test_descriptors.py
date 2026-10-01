"""Tensors, buffers and the quantization geometry they carry.

The interesting property here is that a quantized tensor's ``nbytes`` is a fact
about its *block* geometry, not its element count: a 1.75-bit row cannot be
addressed by assuming one element is one element wide.  These tests pin that
arithmetic against the formats' hand-computed sizes.
"""

from __future__ import annotations

import pytest

from pocketllm.kernels import (
    QUANT_FORMATS,
    Device,
    DeviceBuffer,
    DType,
    HostBuffer,
    ShapeError,
    Tensor,
    TensorDesc,
    parse_device,
    quant_format,
    quant_format_by_file_id,
)


def test_dtype_itemsize_and_floatness() -> None:
    assert DType.F32.itemsize == 4
    assert DType.F16.itemsize == 2
    assert DType.I8.itemsize == 1
    assert DType.F32.is_float and DType.BF16.is_float
    assert not DType.I32.is_float


def test_quant_lookup_by_name_and_file_id() -> None:
    iq4 = quant_format("iq4_nl")
    assert iq4.block_elems == 32
    assert iq4.block_bytes == 18
    assert quant_format_by_file_id(20) is iq4


def test_every_named_quant_format_is_in_the_table() -> None:
    for name, fmt in QUANT_FORMATS.items():
        assert fmt.name == name
        assert quant_format(name) is fmt
        assert quant_format_by_file_id(fmt.file_type_id) is fmt


def test_unknown_quant_format_is_refused() -> None:
    with pytest.raises(KeyError):
        quant_format("q99_fantasy")


def test_desc_needs_exactly_one_of_dtype_or_quant() -> None:
    with pytest.raises(ShapeError):
        TensorDesc((4, 4))
    with pytest.raises(ShapeError):
        TensorDesc((4, 4), dtype=DType.F32, quant=quant_format("iq4_nl"))


def test_desc_rejects_negative_extent() -> None:
    with pytest.raises(ShapeError):
        TensorDesc((4, -1), dtype=DType.F32)


def test_dense_nbytes_is_elements_times_itemsize() -> None:
    desc = TensorDesc((3, 5), dtype=DType.F16)
    assert desc.elem_count == 15
    assert desc.nbytes == 30


def test_iq4_nl_row_geometry() -> None:
    # 256 weights, 32 per 18-byte block -> eight blocks -> 144 bytes, matching
    # the runtime span the loader folds a row into.
    desc = TensorDesc((7, 256), quant=quant_format("iq4_nl"))
    assert desc.rows == 7
    assert desc.packed_row_elems == 256
    assert desc.nbytes == 7 * 8 * 18


def test_iq1_m_row_geometry() -> None:
    # 2048 weights, 256 per 56-byte super-block -> eight blocks.
    desc = TensorDesc((2, 2048), quant=quant_format("iq1_m"))
    assert desc.nbytes == 2 * 8 * 56


def test_quantized_row_must_be_packable() -> None:
    # 33 weights is not a whole number of 32-weight IQ4_NL blocks; nbytes rounds
    # up, which is the honest answer for a padded row.
    desc = TensorDesc((1, 33), quant=quant_format("iq4_nl"))
    assert desc.nbytes == 2 * 18


def test_parse_device_forms() -> None:
    assert parse_device("cpu") == Device("cpu", 0)
    assert parse_device("cuda:2") == Device("cuda", 2)
    # A bare integer names an index on the default kind, which is the host.
    assert parse_device(3) == Device("cpu", 3)
    assert parse_device(Device("mps", 1)) == Device("mps", 1)
    with pytest.raises(ValueError):
        parse_device("cuda:-1")


def test_device_kind_is_extensible() -> None:
    """A third-party backend registers its own kind; the core does not enumerate them."""
    assert Device("my_custom_npu", 0).kind == "my_custom_npu"


def test_host_buffer_round_trip() -> None:
    buffer = HostBuffer(bytearray(b"\x01\x02\x03\x04"))
    assert buffer.nbytes == 4
    assert buffer.device == Device("cpu", 0)
    assert bytes(buffer.host_view()) == b"\x01\x02\x03\x04"


def test_device_buffer_subview_stays_in_bounds() -> None:
    buffer = DeviceBuffer(device=Device("cpu", 0), nbytes=64)
    assert buffer.subview(8, 16).nbytes == 16
    with pytest.raises(ValueError):
        buffer.subview(60, 8)


def test_device_buffer_rejects_negative_size() -> None:
    with pytest.raises(ValueError):
        DeviceBuffer(device=Device("cpu", 0), nbytes=-1)


def test_tensor_does_not_own_its_buffer() -> None:
    buffer = HostBuffer(bytearray(16))
    tensor = Tensor(TensorDesc((4,), dtype=DType.F32), buffer)
    assert tensor.buffer is buffer
    assert tensor.desc.nbytes == 16