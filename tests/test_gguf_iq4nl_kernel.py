"""The IQ4_NL raw-block GEMM against the format's own decoder.

``iq4_nl`` is the eleventh type in the raw-block runtime and the only one whose
block is 32 weights rather than 256.  ``src/csrc/cuda_kernel_impl.cu`` handles
that by folding: a kernel's 256-wide span is eight consecutive native blocks, so
``iq4nl_block_dot_256`` walks eight sub-blocks where the other ten formats unpack
one header, and no GEMM above it needed an IQ4_NL case.

What that buys is a small kernel and what it costs is a specific class of bug --
a geometry that is right for every other format and wrong for this one -- so the
tests here are aimed at the seams rather than at the arithmetic:

- the codebook and the nibble order, against ``loader/gguf/iq4_nl.py``;
- the decode kernel (one row) and the prefill kernel (four rows at a time),
  which take different code paths through the same dot;
- the grouped-MoE entry point, which is how this format is actually reached;
- a real tensor out of the released checkpoint, because a synthetic block can be
  uniform in ways the format is not.

Every comparison is against a fp32 reference computed from the decoded blocks,
so the tolerance is the kernel's accumulation order and not the format's error.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.loader.gguf import iq4_nl
from src.loader.gguf.bundle import read_gguf_bundle
from src.loader.gguf.quant_types import IQ4_NL_RUNTIME_SPAN
from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader
from src.loader.gguf.tensor_reader import GGUFTensorDataReader


CHECKPOINT_DIR_ENV = "POCKETLLM_XING4_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2"
GGUF_NAME = "Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"

#: Layer 2's routed gate stack: 3-D, one 3584x1024 plane per expert.  This is
#: how the MoE reads the format, and it is the shape `read_routed_expert_blocks`
#: exists for.
REAL_ROUTED_TENSOR = "blk.2.ffn_gate_exps.weight"

#: Layer 0 is a dense block, so its MLP is a plain 2-D matrix: 3584 in, 9216 out,
#: and the shape `GGUFQuantizedTensorLoader.read_quant` accepts.
REAL_DENSE_TENSOR = "blk.0.ffn_gate.weight"

#: The routed-expert stack the MoE entry point wants.  Four experts keeps the
#: read to one tensor and is still more than the kernel's route count.
MOE_EXPERTS = 4

#: The output of this kernel is ``c10::BFloat16``; the caller casts it on.  So the
#: comparison is not "fp32 to within some epsilon" -- it is the bf16 store's own
#: granularity, and a tolerance tighter than this would fail on a correct kernel.
#: Stated against each output row's largest magnitude, because that is the scale
#: both the store's relative error and the fp32 accumulation error are relative to.
BF16_REL_TOL = 2.0**-7


def _assert_matches(got: np.ndarray, want: np.ndarray) -> None:
    """Compare a bf16 kernel output against an fp32 reference, row by row.

    Row-relative rather than element-relative: an element that is near zero in a
    row whose dot product is O(200) carries the row's absolute error, and an
    elementwise relative test would read that as a mismatch on a correct kernel.
    """
    scale = np.maximum(np.abs(np.asarray(want)).max(axis=-1, keepdims=True), 1e-6)
    err = np.abs(np.asarray(got) - np.asarray(want)) / scale
    assert np.all(err <= BF16_REL_TOL), (float(err.max()), BF16_REL_TOL)


def _cuda_mod():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from src.kernels.cuda_loader import load_cuda_kernel

    module = load_cuda_kernel()
    if module is None or not hasattr(module, "gguf_quant_gemm_forward"):
        pytest.skip("cuda_kernel extension is not built for this interpreter")
    return module


def _gguf_path() -> Path:
    root = Path(os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT))
    path = root / GGUF_NAME
    if not path.exists():
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {path}")
    return path


def _marker_blocks(rows: int, row_elems: int, *, seed: int) -> np.ndarray:
    """Native 18-byte blocks with real codebook indices and real fp16 scales.

    Both nibbles of every byte are filled from the codebook, so a kernel that
    reads only the low half or transposes the halves produces a different number
    rather than the same number twice.
    """
    rng = np.random.default_rng(seed)
    n_blocks = rows * iq4_nl.blocks_per_row(row_elems)
    payload = np.empty((n_blocks, iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    # Scales span a decade rather than being uniform: a kernel that reads the
    # scale from the wrong byte gets a different decade, not a different rounding.
    payload[:, :2] = np.frombuffer(
        rng.uniform(0.004, 0.05, size=n_blocks).astype("<f2").tobytes(), dtype=np.uint8
    ).reshape(n_blocks, 2)
    payload[:, 2:] = rng.integers(0, 256, size=(n_blocks, 16), dtype=np.uint8)
    return payload.reshape(rows, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES)


def _sigmoid(a: np.ndarray) -> np.ndarray:
    """Stable logistic, so a large negative activation does not warn `exp` overflow."""
    out = np.empty_like(a, dtype=np.float32)
    positive = a >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-a[positive]))
    exp_a = np.exp(a[~positive])
    out[~positive] = exp_a / (1.0 + exp_a)
    return out


def _reference(blocks: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`x @ W.T` in fp32, with `W` decoded from the native blocks.

    ``dequantize_blocks`` is exact -- a codebook entry is a small integer and the
    scale is a finite fp16 -- so this is the format's arithmetic and not an
    approximation of it.
    """
    rows = np.asarray(blocks).shape[0]
    weights = iq4_nl.dequantize_blocks(np.asarray(blocks)).reshape(rows, -1)
    return np.asarray(x, dtype=np.float32) @ weights.astype(np.float32).T


def _run(cuda_mod, blocks: np.ndarray, x: np.ndarray, row_elems: int, device: str) -> np.ndarray:
    """Drive the raw-block GEMM exactly as `QuantizedGGUFLinear` does.

    `row_elems` is passed rather than derived from the block grid: the two shapes
    this is called with -- `(out_dim, blocks_per_row, 18)` for a dense matrix and
    `(experts, out_dim, blocks_per_row, 18)` for a routed stack -- put the block
    count in different axes, and a derived one would read the expert count as
    blocks in the second case.
    """
    rows_out = np.asarray(blocks).shape[-3]
    grid = torch.empty(0, dtype=torch.int8, device=device)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).to(torch.float16).contiguous()
    x_t = x_t.reshape(-1, row_elems)
    folded = torch.as_tensor(
        iq4_nl.fold_to_runtime_span(np.asarray(blocks), row_elems).copy(), device=device
    )
    entry = cuda_mod.gguf_quant_gemm_prefill_forward if x_t.shape[0] > 1 else cuda_mod.gguf_quant_gemm_forward
    y = entry(x_t, folded, row_elems, 20, grid)
    return y.float().reshape(-1, rows_out).cpu().numpy()


# --------------------------------------------------------------------------- #
# Synthetic blocks: the geometry, and both kernels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("row_elems,rows_out", [(256, 5), (1024, 4), (3584, 3)])
def test_the_decode_kernel_matches_the_decoder(row_elems: int, rows_out: int) -> None:
    """One row in, one row out -- the path `rows == 1` takes."""
    cuda_mod = _cuda_mod()
    blocks = _marker_blocks(rows_out, row_elems, seed=row_elems)
    x = np.random.default_rng(1).standard_normal((1, row_elems)).astype(np.float32)
    got = _run(cuda_mod, blocks, x, row_elems, "cuda")
    want = _reference(blocks, x.astype(np.float16).astype(np.float32))
    _assert_matches(got, want)


@pytest.mark.parametrize("n_rows", [2, 3, 4, 5, 8, 33])
def test_the_prefill_kernel_matches_the_decoder(n_rows: int) -> None:
    """`kGGUFQuantPrefillRows` is four, so 5 and 33 cross a partial tile.

    A tile whose tail is padded with zeros has to reduce to zero and not to a
    stale accumulator, which is what the row-by-row comparison here sees.
    """
    cuda_mod = _cuda_mod()
    row_elems, rows_out = 256, 3
    blocks = _marker_blocks(rows_out, row_elems, seed=n_rows)
    x = np.random.default_rng(n_rows).standard_normal((n_rows, row_elems)).astype(np.float32)
    got = _run(cuda_mod, blocks, x, row_elems, "cuda")
    want = _reference(blocks, x.astype(np.float16).astype(np.float32))
    assert got.shape == want.shape
    _assert_matches(got, want)


def test_a_row_that_is_all_one_codebook_entry_is_not_the_zero_row() -> None:
    """The degenerate block: one codebook index everywhere, and a real scale.

    A kernel that masks the wrong nibble, or reads the scale from the wrong
    offset, lands on a constant -- and a constant times a random row is still
    plausible.  Pinning the exact value is what makes the constant visible.
    """
    cuda_mod = _cuda_mod()
    row_elems, rows_out = 256, 2
    blocks = np.zeros((rows_out, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :, :2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    blocks[:, :, 2:] = 0x77  # index 7 in both nibbles: -10
    x = np.ones((1, row_elems), dtype=np.float32)
    got = _run(cuda_mod, blocks, x, row_elems, "cuda")
    _assert_matches(got, np.full((1, rows_out), -10.0 * row_elems, dtype=np.float32))


def test_the_codebook_entry_zero_is_the_scale_and_only_the_scale() -> None:
    """Index 0 is ``-127``, not zero and not ``1``.

    Reading the codebook as ``index - 8`` or as an unsigned table is the same
    class of mistake as reading the wrong nibble, and both produce a matrix that
    looks like a quantized matrix.  One entry at a time makes the value of that
    entry the whole assertion.
    """
    cuda_mod = _cuda_mod()
    row_elems, rows_out = 256, 16
    blocks = np.zeros((rows_out, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    for row in range(rows_out):
        blocks[row, :, :2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
        blocks[row, :, 2:] = row | (row << 4)
    x = np.ones((1, row_elems), dtype=np.float32)
    got = _run(cuda_mod, blocks, x, row_elems, "cuda")
    table = iq4_nl.kvalues().astype(np.float32)
    # Each row is a different codebook entry, so this is the table itself: index 0
    # must come out at -127 * 32 * 8 * scale and index 15 at 113 * 32 * 8 * scale,
    # in that order.
    got_row_scaled = got / (row_elems + 0) * np.float32(1.0)
    assert np.allclose(got_row_scaled[0], table, rtol=BF16_REL_TOL), (got[0], table)


def test_the_low_and_high_nibbles_are_not_interleaved() -> None:
    """Byte j is weight j in its low nibble and weight j + 16 in its high one.

    An alternating read -- weight 2j, weight 2j+1 -- is IQ4_NL's most plausible
    wrong answer: it is another real packing, it is what several 4-bit formats
    do, and it survives an all-ones row.  A row that counts makes it fail.
    """
    cuda_mod = _cuda_mod()
    row_elems, rows_out = 256, 1
    blocks = np.zeros((rows_out, iq4_nl.blocks_per_row(row_elems), iq4_nl.IQ4_NL_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :, :2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    # Byte j holds index j low and index 0 high: weight j is table[j], weight
    # j + 16 is table[0].
    for j in range(16):
        blocks[:, :, 2 + j] = j
    x = np.arange(1, row_elems + 1, dtype=np.float32)
    got = _run(cuda_mod, blocks, x, row_elems, "cuda")
    table = iq4_nl.kvalues().astype(np.float32)
    per_block = np.concatenate([table[:16], np.full(16, table[0], dtype=np.float32)])
    want = np.array([[np.dot(np.tile(per_block, row_elems // 32), x)]], dtype=np.float32)
    _assert_matches(got, want)


# --------------------------------------------------------------------------- #
# The grouped MoE entry point, which is how the format is reached in the model
# --------------------------------------------------------------------------- #


def test_the_grouped_moe_kernel_runs_iq4_nl_experts() -> None:
    """The generic grouped path, over stacked IQ4_NL expert weights.

    Xing4.0's routed experts are this type, and the MoE kernel reaches the block
    dot through the same dispatch the dense GEMM does -- so this is the check
    that the new type id did not fall through to the Q2_K branch, where it would
    read 18-byte blocks as 132-byte ones and return something.
    """
    cuda_mod = _cuda_mod()
    if not hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward"):
        pytest.skip("grouped MoE kernel is not in this build")
    dim, inter, experts = 256, 512, MOE_EXPERTS
    rng = np.random.default_rng(7)
    w1 = np.stack([_marker_blocks(inter, dim, seed=e) for e in range(experts)])
    w3 = np.stack([_marker_blocks(inter, dim, seed=100 + e) for e in range(experts)])
    w2 = np.stack([_marker_blocks(dim, inter, seed=200 + e) for e in range(experts)])

    tokens, top_k = 3, 2
    # Scaled down because the reference's swiglu is evaluated in fp16, the way the
    # kernel evaluates it: a wide row of full-size activations would overflow the
    # format and make the comparison a statement about fp16's range, not the kernel.
    x = (rng.standard_normal((tokens, dim)) * 0.05).astype(np.float32)
    picks = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]], dtype=np.float32)

    # seg_starts is the CSR offset of each expert's routes, and the routes are
    # ordered by expert, so token order within an expert segment is free.
    seg = np.zeros(experts + 1, dtype=np.int32)
    for e in range(experts):
        seg[e + 1] = seg[e] + int((picks == e).sum())
    route_tokens = np.empty(seg[-1], dtype=np.int64)
    route_weights = np.empty(seg[-1], dtype=np.float32)
    cursor = seg[:-1].copy()
    for token in range(tokens):
        for slot in range(top_k):
            e = int(picks[token, slot])
            route_tokens[cursor[e]] = token
            route_weights[cursor[e]] = weights[token, slot]
            cursor[e] += 1

    device = "cuda"
    grid = torch.empty(0, dtype=torch.int8, device=device)
    y = cuda_mod.gguf_moe_prefill_grouped_forward(
        torch.as_tensor(x, device=device).half(),
        torch.as_tensor(route_tokens, device=device),
        torch.as_tensor(route_weights, device=device),
        torch.as_tensor(seg, device=device),
        torch.as_tensor(iq4_nl.fold_to_runtime_span(w1, dim).copy(), device=device),
        torch.as_tensor(iq4_nl.fold_to_runtime_span(w3, dim).copy(), device=device),
        torch.as_tensor(iq4_nl.fold_to_runtime_span(w2, inter).copy(), device=device),
        dim, 20, dim, 20, inter, 20, grid, 0.0,
    ).float().cpu().numpy()

    # Reference: swiglu(silu(w1 x) * w3 x) per expert, summed by route weight.
    want = np.zeros((tokens, dim), dtype=np.float32)
    for token in range(tokens):
        acc = np.zeros(dim, dtype=np.float32)
        for slot in range(top_k):
            e = int(picks[token, slot])
            # The kernel quantizes its activation to fp16 before the dot, so the
            # reference does too; the tolerance is that and not the format's.
            gate = _reference(w1[e], x[token].astype(np.float16).astype(np.float32))
            up = _reference(w3[e], x[token].astype(np.float16).astype(np.float32))
            hidden = (gate * _sigmoid(gate)) * up
            acc += weights[token, slot] * _reference(w2[e], hidden.astype(np.float16).astype(np.float32))
        want[token] = acc
    assert y.shape == want.shape
    _assert_matches(y, want)


# --------------------------------------------------------------------------- #
# The released checkpoint's own tensors
# --------------------------------------------------------------------------- #


def _real_blocks(dim_rows: int) -> tuple[np.ndarray, int]:
    path = _gguf_path()
    bundle = read_gguf_bundle(path)
    tensor = bundle.tensors_by_name[REAL_ROUTED_TENSOR]
    reader = GGUFTensorDataReader(tensor.shard_path)
    try:
        blocks = np.stack(
            [
                reader.read_routed_expert_blocks(REAL_ROUTED_TENSOR, e)[0].numpy()
                for e in range(dim_rows)
            ]
        )
        _, type_name, row_elems = reader.read_routed_expert_blocks(REAL_ROUTED_TENSOR, 0)
    finally:
        reader.close()
    assert type_name == "iq4_nl"
    return blocks, int(row_elems)


def test_a_real_checkpoint_tensor_matches_its_own_decode() -> None:
    """The release's IQ4_NL experts, whole-block, against the format's decoder.

    A synthetic payload can be uniform in ways a quantizer's output is not --
    every scale the same, every index populated.  The real tensor's scales span
    the range the format actually uses, so a kernel that reads the scale from
    the wrong byte still passes the synthetic tests and fails this one.
    """
    cuda_mod = _cuda_mod()
    blocks, row_elems = _real_blocks(MOE_EXPERTS)
    assert row_elems == 3584
    assert blocks.shape == (MOE_EXPERTS, 1024, 3584 // iq4_nl.QK_IQ4_NL, 18)
    # The stack is four different experts.  A reader that ignored the expert index
    # would hand back the same plane four times and every comparison below would
    # still pass, so this is asserted rather than assumed.
    for expert in range(1, MOE_EXPERTS):
        assert not np.array_equal(blocks[0], blocks[expert])
    x = np.random.default_rng(11).standard_normal((1, row_elems)).astype(np.float32)
    for expert in range(MOE_EXPERTS):
        # The dense GEMM takes one (out_dim, blocks_per_row, 18) matrix, so this
        # walks the stack a plane at a time.
        got = _run(cuda_mod, blocks[expert], x, row_elems, "cuda")
        want = _reference(blocks[expert], x.astype(np.float16).astype(np.float32))
        _assert_matches(got, want)
        # And the output is not degenerate: a kernel that returned zeros would pass
        # a relative tolerance against a reference that was also zeros.
        assert np.abs(want).max() > 1.0


def test_reading_it_through_the_quantized_loader_folds_it() -> None:
    """The one path a model actually takes: `GGUFQuantizedTensorLoader`.

    `test_gguf_iq4nl_reader` pins the fold in isolation; this pins it as the
    loader hands it to a kernel, including the device copy, so a fold applied
    after the copy or not at all fails here.
    """
    path = _gguf_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with GGUFQuantizedTensorLoader(str(path), device=device) as loader:
        quant = loader.read_quant(REAL_DENSE_TENSOR, "iq4_nl")
    assert quant.type_id == 20
    assert quant.row_elems == 3584
    assert quant.out_dim == 9216
    assert tuple(quant.blocks.shape) == (
        quant.out_dim,
        3584 // IQ4_NL_RUNTIME_SPAN,
        IQ4_NL_RUNTIME_SPAN // iq4_nl.QK_IQ4_NL * iq4_nl.IQ4_NL_BLOCK_BYTES,
    )
    assert quant.blocks.shape[-1] == 144


def test_a_real_dense_tensor_matches_its_own_decode() -> None:
    """The loader-built GEMM over a real 2-D IQ4_NL matrix.

    The routed-expert test above reads the same format through a different
    function, so this one is what pins the path `QuantizedGGUFLinear` takes: the
    loader's fold, the device copy, and the dispatch that reads the type id.
    """
    cuda_mod = _cuda_mod()
    path = _gguf_path()
    device = "cuda"
    with GGUFQuantizedTensorLoader(str(path), device=device) as loader:
        quant = loader.read_quant(REAL_DENSE_TENSOR, "iq4_nl")
    row_elems, out_dim = quant.row_elems, quant.out_dim
    x = np.random.default_rng(13).standard_normal((1, row_elems)).astype(np.float32)
    grid = torch.empty(0, dtype=torch.int8, device=device)
    y = cuda_mod.gguf_quant_gemm_forward(
        torch.as_tensor(x, device=device).half(), quant.blocks, row_elems, quant.type_id, grid
    )
    # The reference decodes the *native* geometry, so it never sees the fold.
    b = read_gguf_bundle(path)
    reader = GGUFTensorDataReader(b.tensors_by_name[REAL_DENSE_TENSOR].shard_path)
    try:
        native, type_name, _ = reader.read_quantized_matrix_blocks(REAL_DENSE_TENSOR)
    finally:
        reader.close()
    assert type_name == "iq4_nl"
    want = _reference(native.numpy(), x.astype(np.float16).astype(np.float32))
    assert y.shape == (1, out_dim)
    _assert_matches(y.float().cpu().numpy(), want)
