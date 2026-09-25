"""The sm_75 ternary GEMM: ``PTQ1_0`` prefill and decode, against the decoded weights.

``src/loader/gguf/ptq1_0.py`` decodes the packing and ``tests/test_ptq1_0_layout.py``
pins that decoder against blocks read out of the released checkpoint, so the packing
is not in question here.  What is in question is the two kernels that consume it: a
tensor-core tile walk for prefill and a DP4A GEMV for decode.

Two references.  The first is the *unquantized* dot product -- decode the blocks to
fp32, multiply by the activation -- which is what the issue's acceptance asks for and
is only accurate to a few parts in a thousand, because the kernels quantize the
activation to int8 per 32 elements exactly as llama.cpp does.  The second reference
reproduces that quantization in torch, so it can be compared to within a summation
order and localises a bug to the weight unpack instead of to the activation path.
Both are needed: the loose one alone would pass for a kernel that reads the wrong
trits of the right magnitude, and the tight one alone would not notice a scale
applied in the wrong place if the reference made the same mistake.

The second reference is compared at the output's own precision.  Both entry points
return bf16 -- that is what the rest of the engine carries and what the Q4_K/Q5_K
paths next to them return -- so the reference is rounded to bf16 as well and the
bound is a couple of bf16 ulps rather than a relative tolerance.  Measured, the two
agree bit-for-bit on every shape tested; the ulp of slack is there for a summation
order that a future warp count could change, and an unpack error is orders of
magnitude above it either way.

With no built extension the file skips; with no checkpoint the checkpoint-backed case
skips and the synthetic cases still run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.loader.gguf import ptq1_0
from src.loader.gguf.quant_types import GGUF_TERNARY_FILE_TYPE_IDS
from src.kernels.cuda_loader import load_cuda_kernel

PTQ1_0_TYPE_ID = GGUF_TERNARY_FILE_TYPE_IDS["ptq1_0"]

CHECKPOINT_ENV = "POCKETLLM_BONSAI_GGUF"
CHECKPOINT_DEFAULT = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
CHECKPOINT_TENSOR = "blk.0.ffn_gate.weight"

#: The decoder's own digit extraction, for every byte: ``((b * 3**s) & 0xFF) * 3 >> 8``.
#: The encoder below is the inverse of *this table* rather than of a re-derivation of
#: the format, so a synthetic block is one the decoder round-trips by construction.
_DIGITS_OF_BYTE = np.array(
    [[((b * 3**s) & 0xFF) * 3 >> 8 for s in range(5)] for b in range(256)], dtype=np.int64
)


def _inverse_table(width: int) -> np.ndarray:
    """``(d_0, ..., d_{width-1})`` as a mixed-radix index -> the byte that decodes to it.

    Built by running the decoder over all 256 bytes rather than from the writer's
    ``(q * 256 + 242) / 243``, because the two are only equivalent where the writer
    lands and this asks the decoder which byte it will accept.  -1 is the sentinel
    and not 255: 255 is a byte the writer does emit.
    """
    table = np.full(3**width, -1, dtype=np.int16)
    for byte in range(256):
        index = 0
        for digit in _DIGITS_OF_BYTE[byte][:width]:
            index = index * 3 + int(digit)
        if table[index] < 0:
            table[index] = byte
    # Every digit tuple the format can represent has to be writable, or the encoder
    # would silently produce a block that decodes to something else.
    assert (table >= 0).all(), f"no byte decodes to {np.argmin(table)} at width {width}"
    for index, byte in enumerate(table):
        assert tuple(_DIGITS_OF_BYTE[byte][:width]) == _radix_digits(index, width)
    return table


def _radix_digits(index: int, width: int) -> tuple[int, ...]:
    digits = []
    for _ in range(width):
        digits.append(index % 3)
        index //= 3
    return tuple(reversed(digits))


_BYTE_OF_DIGITS5 = _inverse_table(5)
_BYTE_OF_DIGITS4 = _inverse_table(4)


def _pack(trits: np.ndarray, table: np.ndarray) -> np.ndarray:
    """One stage's worth of bytes: ``trits`` is ``(..., width)`` in ``{-1, 0, 1}``."""
    digits = trits.astype(np.int64) + 1
    index = np.zeros(digits.shape[:-1], dtype=np.int64)
    for s in range(digits.shape[-1]):
        index = index * 3 + digits[..., s]
    return table[index].astype(np.uint8)


def encode_blocks(trits: np.ndarray) -> np.ndarray:
    """Pack ``(n, 128)`` trits into ``(n, 28)`` PTQ1_0 blocks, scale and all.

    The walk is the one ``block_trits`` reads: a 16-wide stage over bytes 0..15
    (weights 0..79), an 8-wide stage over bytes 16..23 (weights 80..119), and qh for
    the last eight with the parity interleaved.
    """
    if trits.shape[-1] != ptq1_0.QK_PTQ1_0:
        raise ValueError(f"trits must be {ptq1_0.QK_PTQ1_0} wide, got {trits.shape[-1]}")
    out = np.zeros((*trits.shape[:-1], ptq1_0.PTQ1_0_BLOCK_BYTES), dtype=np.uint8)

    wide = np.stack([trits[..., s * 16 : s * 16 + 16] for s in range(5)], axis=-1)
    out[..., 0:16] = _pack(wide, _BYTE_OF_DIGITS5)
    narrow = np.stack([trits[..., 80 + s * 8 : 80 + s * 8 + 8] for s in range(5)], axis=-1)
    out[..., 16:24] = _pack(narrow, _BYTE_OF_DIGITS5)

    for parity in range(2):
        stream = np.stack([trits[..., 120 + 2 * s + parity] for s in range(4)], axis=-1)
        out[..., 24 + parity] = _pack(stream, _BYTE_OF_DIGITS4)

    return out


def make_blocks(rng: np.random.Generator, rows: int, k: int, scale: float | None = None) -> np.ndarray:
    """A ``[rows, k]`` ternary weight matrix as ``(rows, k/128, 28)`` blocks."""
    n_blocks = rows * k // ptq1_0.QK_PTQ1_0
    trits = rng.integers(-1, 2, size=(n_blocks, ptq1_0.QK_PTQ1_0)).astype(np.int8)
    blocks = encode_blocks(trits)
    scales = (
        np.full(n_blocks, np.float16(scale))
        if scale is not None
        else rng.uniform(0.25, 1.5, size=n_blocks).astype(np.float16)
    )
    blocks[:, -2:] = scales.view(np.uint8).reshape(-1, 2)
    return blocks.reshape(rows, k // ptq1_0.QK_PTQ1_0, ptq1_0.PTQ1_0_BLOCK_BYTES)


def dequantize(blocks: np.ndarray) -> torch.Tensor:
    """``(rows, k/128, 28)`` blocks -> fp32 ``[rows, k]`` weights, via the loader."""
    weight = torch.from_numpy(ptq1_0.dequantize_blocks(blocks))
    return weight.reshape(blocks.shape[0], -1)


# --------------------------------------------------------------------------- #
# The encoder the synthetic cases are built on
# --------------------------------------------------------------------------- #


def test_the_encoder_round_trips_through_the_decoder() -> None:
    rng = np.random.default_rng(3860)
    trits = rng.integers(-1, 2, size=(64, ptq1_0.QK_PTQ1_0)).astype(np.int8)
    blocks = encode_blocks(trits)
    decoded = np.stack([ptq1_0.block_trits(b.tobytes()) for b in blocks])
    assert np.array_equal(decoded, trits)


def test_the_encoder_covers_every_digit_tuple() -> None:
    """A table with a hole would make the encoder write a *different* block, quietly."""
    assert len(set(_BYTE_OF_DIGITS5.tolist())) == 3**5
    assert len(set(_BYTE_OF_DIGITS4.tolist())) == 3**4


# --------------------------------------------------------------------------- #
# The references
# --------------------------------------------------------------------------- #


def q8_1_reference(x: torch.Tensor, weight: torch.Tensor, *, scale_in_fp16: bool) -> torch.Tensor:
    """The kernels' own arithmetic, in torch: per-32 amax int8 activation, fp32 dot.

    ``scale_in_fp16`` follows the storage of the activation scale on the path being
    tested -- the decode quantizer writes a ``half``, the MMQ quantizer a ``float`` --
    because that rounding is visible at the tolerance this reference is compared at.
    """
    rows, k = x.shape
    groups = x.reshape(rows, k // 32, 32)
    amax = groups.abs().amax(dim=-1)
    d = amax / 127.0
    d_inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.round(groups * d_inv.unsqueeze(-1)).clamp(-127, 127)
    if scale_in_fp16:
        d = d.to(torch.float16).to(torch.float32)
    return ((q * d.unsqueeze(-1)).reshape(rows, k)).to(torch.float32) @ weight.t()


#: The gap to the *unquantized* dot product is the activation quantization and nothing
#: else.  Measured over eight seeds at K in {5120, 6144, 17408} the relative deviation
#: stays under 6.7e-3, so this is the measured bound with slack -- it is still tight
#: enough that an overall scale error, or a stage read at the wrong offset, fails it.
ACTIVATION_TOLERANCE = 1e-2

#: bf16 carries eight explicit mantissa bits, so one ulp of a value is between 2**-9
#: and 2**-8 of it.  See the module docstring for why the tight comparison is stated
#: in ulps of the output rather than as a relative tolerance.
BF16_ULPS = 2


def bf16_ulps(out: torch.Tensor, reference: torch.Tensor) -> float:
    """How far ``out`` is from ``reference``, at worst, in bf16 ulps of the reference."""
    rounded = reference.to(torch.bfloat16).float()
    return ((out - rounded).abs() / (rounded.abs() * 2.0**-8 + 1e-30)).max().item()


def _cuda():
    module = load_cuda_kernel()
    if module is None:
        pytest.skip("the cuda_kernel extension is not built")
    for name in ("gguf_quant_gemm_forward", "gguf_quant_gemm_prefill_forward", "gguf_ptq1_0_dp4a_decode_forward"):
        if not hasattr(module, name):
            pytest.skip(f"{name} is not in the built extension")
    return module


@pytest.fixture(scope="module")
def cuda():
    return _cuda()


def _to_device(blocks: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(blocks)).to(device)


def decode(cuda, x: torch.Tensor, blocks: torch.Tensor, k: int) -> torch.Tensor:
    """Through the dispatcher, so the type id's routing is covered too."""
    grid = torch.empty(0, dtype=torch.int8, device=x.device)
    return cuda.gguf_quant_gemm_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid).float()


def prefill(cuda, x: torch.Tensor, blocks: torch.Tensor, k: int) -> torch.Tensor:
    grid = torch.empty(0, dtype=torch.int8, device=x.device)
    return cuda.gguf_quant_gemm_prefill_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid).float()


# --------------------------------------------------------------------------- #
# Synthetic weights: shapes, and both tolerances
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [5120, 6144, 17408])
def test_decode_shapes_and_the_loose_tolerance(cuda, k: int) -> None:
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3861 + k)
    rows, n = 1, 96
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((rows, k), dtype=np.float32)).to(device)

    out = decode(cuda, x, _to_device(blocks, device), k)
    assert out.shape == (rows, n)

    reference = x @ weight.t()
    # What a per-32 int8 activation costs, and no more: see the tight case below for
    # the part of the error that is not the activation.
    relative = (out - reference).norm() / reference.norm()
    assert relative < ACTIVATION_TOLERANCE, (k, float(relative))


@pytest.mark.parametrize("k", [5120, 17408])
def test_decode_matches_the_kernels_own_arithmetic(cuda, k: int) -> None:
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3862 + k)
    rows, n = 1, 64
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((rows, k), dtype=np.float32)).to(device)

    out = decode(cuda, x, _to_device(blocks, device), k)
    reference = q8_1_reference(x, weight, scale_in_fp16=True)
    # Only the summation order differs here, so this is a real bit-level check of the
    # trit unpack: a wrong stage, a wrong qh parity or a scale on the wrong block all
    # move the answer by far more than a couple of ulps.
    assert bf16_ulps(out, reference) <= BF16_ULPS, (k, bf16_ulps(out, reference))


@pytest.mark.parametrize("rows", [2, 8, 16, 33])
def test_prefill_shapes_and_the_loose_tolerance(cuda, rows: int) -> None:
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3863 + rows)
    k, n = 5120, 160
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((rows, k), dtype=np.float32)).to(device)

    out = prefill(cuda, x, _to_device(blocks, device), k)
    assert out.shape == (rows, n)
    relative = (out - (x @ weight.t())).norm() / (x @ weight.t()).norm()
    assert relative < ACTIVATION_TOLERANCE, (rows, float(relative))


@pytest.mark.parametrize("rows", [2, 8, 16, 33])
def test_prefill_matches_the_kernels_own_arithmetic(cuda, rows: int) -> None:
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3864 + rows)
    k, n = 5120, 160
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((rows, k), dtype=np.float32)).to(device)

    out = prefill(cuda, x, _to_device(blocks, device), k)
    reference = q8_1_reference(x, weight, scale_in_fp16=False)
    assert bf16_ulps(out, reference) <= BF16_ULPS, (rows, bf16_ulps(out, reference))


def test_a_leading_batch_dimension_survives(cuda) -> None:
    """The kernels flatten the activation; the output has to be put back the same way."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3865)
    k, n = 5120, 32
    blocks = make_blocks(rng, n, k)
    x = torch.from_numpy(rng.standard_normal((2, 3, k), dtype=np.float32)).to(device)
    out = prefill(cuda, x, _to_device(blocks, device), k)
    assert out.shape == (2, 3, n)


def test_both_entry_points_hand_back_bf16(cuda) -> None:
    """The engine's carrier, and what the Q4_K/Q5_K paths beside these two return."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3872)
    k, n = 5120, 32
    blocks = _to_device(make_blocks(rng, n, k), device)
    x = torch.from_numpy(rng.standard_normal((1, k), dtype=np.float32)).to(device)
    grid = torch.empty(0, dtype=torch.int8, device=device)

    one = cuda.gguf_quant_gemm_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid)
    assert one.dtype == torch.bfloat16 and one.shape == (1, n)
    many = cuda.gguf_quant_gemm_prefill_forward(x, blocks, k, PTQ1_0_TYPE_ID, grid)
    assert many.dtype == torch.bfloat16 and many.shape == (1, n)


def test_a_matrix_wide_enough_to_need_a_second_warp_still_agrees(cuda) -> None:
    """Eight warps per block: an off-by-one in the feature index shows up as a shift."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3866)
    k, n = 5120, 130
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((1, k), dtype=np.float32)).to(device)
    out = decode(cuda, x, _to_device(blocks, device), k)
    reference = q8_1_reference(x, weight, scale_in_fp16=True)
    assert bf16_ulps(out, reference) <= BF16_ULPS


def test_the_rows_are_not_a_permutation_of_each_other(cuda) -> None:
    """Which guards the feature index: a transposed walk would still have the right norm."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3867)
    k, n = 5120, 64
    blocks = make_blocks(rng, n, k)
    x = torch.from_numpy(rng.standard_normal((1, k), dtype=np.float32)).to(device)
    out = decode(cuda, x, _to_device(blocks, device), k)
    assert torch.unique(out).numel() == n


def test_a_negative_activation_is_not_rectified(cuda) -> None:
    """The trits are signed and so is the activation: the dot has to carry the sign."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3868)
    k, n = 5120, 32
    blocks = make_blocks(rng, n, k, scale=1.0)
    x = -torch.from_numpy(np.abs(rng.standard_normal((1, k), dtype=np.float32))).to(device)
    out = decode(cuda, x, _to_device(blocks, device), k)
    reference = x @ dequantize(blocks).to(device).t()
    assert (out - reference).abs().max() < 1e-2 * reference.abs().max()


def test_a_block_count_the_dp4a_walk_cannot_tile_in_fours_still_agrees(cuda) -> None:
    """K = 384 is three blocks per row, so a warp's last group has nothing to read."""
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3869)
    k, n = 384, 48
    blocks = make_blocks(rng, n, k)
    weight = dequantize(blocks).to(device)
    x = torch.from_numpy(rng.standard_normal((1, k), dtype=np.float32)).to(device)
    out = decode(cuda, x, _to_device(blocks, device), k)
    reference = q8_1_reference(x, weight, scale_in_fp16=True)
    assert bf16_ulps(out, reference) <= BF16_ULPS


def test_the_decode_entry_point_refuses_more_than_one_row(cuda) -> None:
    device = torch.device("cuda", 0)
    rng = np.random.default_rng(3870)
    k, n = 5120, 16
    blocks = make_blocks(rng, n, k)
    x = torch.from_numpy(rng.standard_normal((2, k), dtype=np.float32)).to(device)
    with pytest.raises((RuntimeError, ValueError)):
        cuda.gguf_ptq1_0_dp4a_decode_forward(x, _to_device(blocks, device), k)


# --------------------------------------------------------------------------- #
# The released checkpoint
# --------------------------------------------------------------------------- #


def test_the_released_weights_agree_with_their_own_decode() -> None:
    """Real blocks, not synthetic ones: the packing's own byte patterns.

    The synthetic cases above write blocks the encoder produced; this one reads the
    file's, which is where a stage walk that only agrees with *our* encoder would
    show up.  ``blk.0.ffn_gate.weight`` is 5120 x 17408, and 64 of its rows are
    ~72 KiB of blocks -- the whole file is not needed, but it is skipped when absent.
    """
    path = os.environ.get(CHECKPOINT_ENV, CHECKPOINT_DEFAULT)
    if not os.path.exists(path):
        pytest.skip(f"set {CHECKPOINT_ENV} or place the checkpoint at {CHECKPOINT_DEFAULT}")

    from src.loader.gguf.tensor_reader import GGUFTensorDataReader

    cuda = _cuda()
    device = torch.device("cuda", 0)
    with GGUFTensorDataReader(path) as reader:
        blocks_t, type_name, k = reader.read_quantized_matrix_block_rows(CHECKPOINT_TENSOR, 0, 64)
        assert type_name == "ptq1_0"
    blocks = blocks_t.numpy()
    assert blocks.shape == (64, k // ptq1_0.QK_PTQ1_0, ptq1_0.PTQ1_0_BLOCK_BYTES)
    weight = dequantize(blocks).to(device)

    rng = np.random.default_rng(3871)
    x = torch.from_numpy(rng.standard_normal((1, k), dtype=np.float32)).to(device)
    out = decode(cuda, x, _to_device(blocks, device), k)
    reference = q8_1_reference(x, weight, scale_in_fp16=True)
    assert bf16_ulps(out, reference) <= BF16_ULPS

    rows = 8
    x_batch = torch.from_numpy(rng.standard_normal((rows, k), dtype=np.float32)).to(device)
    batch = prefill(cuda, x_batch, _to_device(blocks, device), k)
    batch_reference = q8_1_reference(x_batch, weight, scale_in_fp16=False)
    assert bf16_ulps(batch, batch_reference) <= BF16_ULPS
