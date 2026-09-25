"""The Bonsai GGUF weight map, pinned against its FP8 sibling checkpoint.

The ternary file is self-describing about the *transform* (``prism.hadamard.*``
names every folded tensor, the block size and the sign vectors) but not about what
it is a transform *of*.  This module supplies that half: it takes the released
``Ternary-Bonsai-2-27B-PTQ1_0.gguf``, undoes the fold on a real weight, and compares
the result row by row against the same tensor in the FP8 export of the same model.
A row that comes back decorrelated means the fold was undone with the wrong sign
vector, the wrong block size or on the wrong axis; a row that comes back at the
right cosine means the whole chain -- the PTQ1_0 decode, the sign vector for that
width, the Walsh-Hadamard rotation, and where in the row the rotation lands -- is
the fork's.

Three things it settles that names cannot:

* **The signs are chosen by the tensor's last dimension**, i.e. the width of the
  activation the fold was computed against: 5120 for everything reading the residual
  stream, 6144 for the two that read a concatenated head axis (``ssm_out`` and
  ``o_proj``), 17408 for ``ffn_down``.  Using 5120 everywhere matches the residual
  tensors and decorrelates the rest, which is what the negative controls below do.
* **The gated-DeltaNet value axis is stored tiled**, ``[head_dim, groups, rep]`` with
  ``rep`` outermost, while ``ssm_out``'s input axis is stored *grouped*,
  ``[head_dim, rep, groups]``.  That asymmetry is the fork's, not an accident of the
  conversion, and it is the reason ``prism.hadamard.gdn_v_grouped`` exists.
* **The token embedding takes the inverse**, in the opposite sign/H order, because a
  lookup is not a matmul.

The comparisons are cosine similarities rather than equalities: the ternary file
stores 1.75 bits per weight and the sibling is FP8, and neither is the original.  So
the thresholds are coarse and the *shape* of the result is the evidence: folded rows
land at 0.86-0.89 against the ternary quantization, unrotated or mis-permuted rows
land at 0.0.

Both artifacts are local and large, so this module skips unless they are on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from src.loader.gguf.ptq1_0 import dequantize_blocks
from src.loader.gguf.prism_hadamard import parse_hadamard_spec
from src.loader.gguf.reader import GGUFReader

GGUF_PATH = Path("/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
SIBLING_DIR = Path("/mnt/data2/Qwen3.8-27B-FP8")

# Cosine a folded ternary row reaches against its FP8 counterpart.  The fold is exact,
# so what separates the two is 1.75 bits of the row and 8 bits of the sibling: a
# 5120-wide row comes back at 0.86-0.89 measured, and nothing here is near the bound.
TERNARY_COSINE = 0.80
# Tensors that are not folded at all, compared directly.  Both sides are 16-bit, so
# the agreement is much tighter -- 0.97 and up measured.
DIRECT_COSINE = 0.95
# What a wrong rotation or a wrong axis produces.  It is not "a bit worse": an
# unrotated row is a different vector, so this is a decorrelation threshold.
DECORRELATED = 0.20

HD = 128
GROUPS = 16
REP = 3


pytestmark = pytest.mark.skipif(
    not GGUF_PATH.exists() or not (SIBLING_DIR / "model.safetensors.index.json").exists(),
    reason="the released ternary checkpoint and its FP8 sibling are not both on disk",
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean row cosine, on ``[rows, width]`` batches."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean())


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean-removed cosine for one-dimensional vectors.

    A permutation of a vector with a single sign -- ``A_log`` is all negative -- keeps
    the raw cosine near one, so the permutation checks have to remove the mean first
    or they would pass on their own negative control.
    """
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


class _Checkpoint:
    """The ternary file, its metadata, and the sibling's index, opened once."""

    def __init__(self) -> None:
        self.info = GGUFReader(str(GGUF_PATH), read_arrays=True).read()
        self.by_name = self.info.tensors_by_name
        self.spec = parse_hadamard_spec(
            self.info.metadata, known_tensor_names=self.by_name
        )
        self._handle = open(GGUF_PATH, "rb")
        self._sibling = json.load(
            open(SIBLING_DIR / "model.safetensors.index.json")
        )["weight_map"]
        self._rotations: dict[int, torch.Tensor] = {}

    def rotation(self, width: int) -> torch.Tensor:
        """The file's forward rotation for one folded width, as a dense matrix.

        ``R = (1/sqrt(N)) H diag(s)``, so ``W' = W R^-1`` is undone by ``W' R``.
        Block-diagonal by construction, which is what makes it cheap and what makes a
        block-size mistake visible: the wrong block size is not a small perturbation
        of the right one.
        """
        cached = self._rotations.get(width)
        if cached is not None:
            return cached
        n = self.spec.block_size
        index = torch.arange(n)
        H = torch.tensor(
            [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)],
            dtype=torch.float32,
        ) / (n ** 0.5)
        signs = torch.tensor(self.spec.signs_for_width(width), dtype=torch.float32)
        blocks = torch.block_diag(*([H] * (width // n)))
        self._rotations[width] = blocks * signs[None, :]
        return self._rotations[width]

    def gguf_tensor(self, name: str, rows: int | None = None) -> torch.Tensor:
        """One GGUF tensor as ``[out, in]`` float32, decoded or widened as stored."""
        tensor = self.by_name[name]
        if len(tensor.dimensions) == 1:
            # A per-head vector: present it as one row so the callers stay uniform.
            outputs, inputs = 1, int(tensor.dimensions[0])
        else:
            outputs = int(tensor.dimensions[1])
            inputs = int(tensor.dimensions[0])
        take = outputs if rows is None else min(rows, outputs)
        per_row = (
            inputs // 128 * 28
            if tensor.type_id == 143
            else inputs * {0: 4, 1: 2, 30: 2}[tensor.type_id]
        )
        self._handle.seek(tensor.absolute_offset)
        raw = self._handle.read(take * per_row)
        if tensor.type_id == 143:
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(take, inputs // 128, 28)
            return torch.from_numpy(dequantize_blocks(blocks).reshape(take, inputs))
        if tensor.type_id == 30:
            widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            values = widened.view(np.float32)
        else:
            values = np.frombuffer(raw, dtype={0: "<f4", 1: "<f2"}[tensor.type_id])
        return torch.from_numpy(values.reshape(take, inputs).astype(np.float32))

    def sibling_tensor(self, name: str, rows: int | None = None) -> torch.Tensor:
        with safe_open(str(SIBLING_DIR / self._sibling[name]), framework="pt") as handle:
            weight = handle.get_tensor(name).float()
        return weight if rows is None else weight[:rows]

    # -- the gated-DeltaNet value axis --------------------------------------- #

    @staticmethod
    def head_permutation() -> torch.Tensor:
        """``perm[sibling_head] = file_head`` for the gated-DeltaNet value axis.

        Indexing a file-ordered head axis with this gives the sibling's order.  The
        sibling is grouped -- head ``k * rep + r`` -- and the file is tiled -- head
        ``r * groups + k``, which is the swap ``gdn_v_grouped`` names.  As a
        *transpose* the swap is its own inverse, but as an index map it is not, and
        the other direction is one of the negative controls below.
        """
        grouped_of_tiled = [
            (head % GROUPS) * REP + head // GROUPS for head in range(GROUPS * REP)
        ]
        return torch.argsort(torch.tensor(grouped_of_tiled))

    @staticmethod
    def reorder_heads(
        matrix: torch.Tensor, perm: torch.Tensor, head_dim: int = HD
    ) -> torch.Tensor:
        """Reorder a matrix whose rows are ``heads * head_dim``.

        ``head_dim`` is 128 for a weight and 1 for a per-head vector, where the row is
        the head itself rather than its features.
        """
        return matrix.reshape(GROUPS * REP, head_dim, matrix.shape[1])[perm].reshape(
            -1, matrix.shape[1]
        )


@pytest.fixture(scope="module")
def ckpt() -> _Checkpoint:
    return _Checkpoint()


def test_the_metadata_is_the_released_one(ckpt: _Checkpoint) -> None:
    """The block this module trusts is the one the artifact declares."""
    spec = ckpt.spec
    assert spec.version == 1
    assert spec.block_size == 1024
    assert spec.transform == "normalized-sylvester-walsh-hadamard"
    assert spec.axis == "input-last-dimension"
    assert spec.sign_widths == (5120, 6144, 17408)
    assert spec.gdn_v_grouped is True
    assert spec.inverse_weight_names == ("token_embd.weight",)
    # 401 folded matrices and one inverse: 48 linear-attention blocks x 6, 16
    # full-attention blocks x 7, plus the head.
    assert len(spec.weight_names) == 401


@pytest.mark.parametrize(
    "gguf_name,sibling_name",
    [
        ("blk.3.attn_q.weight", "model.language_model.layers.3.self_attn.q_proj.weight"),
        ("blk.3.attn_k.weight", "model.language_model.layers.3.self_attn.k_proj.weight"),
        ("blk.3.attn_v.weight", "model.language_model.layers.3.self_attn.v_proj.weight"),
        ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj.weight"),
        ("blk.0.ffn_up.weight", "model.language_model.layers.0.mlp.up_proj.weight"),
        ("output.weight", "lm_head.weight"),
    ],
)
def test_residual_stream_rows_unfold_at_width_5120(
    ckpt: _Checkpoint, gguf_name: str, sibling_name: str
) -> None:
    """Everything reading the residual stream is folded with the 5120 signs."""
    folded = ckpt.gguf_tensor(gguf_name, rows=256) @ ckpt.rotation(5120)
    sibling = ckpt.sibling_tensor(sibling_name, rows=256)
    assert _cosine(folded, sibling) > TERNARY_COSINE
    # The same rows without the rotation, which is what proves the fold is there.
    assert abs(_cosine(ckpt.gguf_tensor(gguf_name, rows=256), sibling)) < DECORRELATED
    # And with the wrong width's signs, which is the mistake a name cannot catch.
    wrong = ckpt.gguf_tensor(gguf_name, rows=256) @ ckpt.rotation(6144)[:5120, :5120]
    assert abs(_cosine(wrong, sibling)) < DECORRELATED


def test_ffn_down_unfolds_at_width_17408(ckpt: _Checkpoint) -> None:
    """The one folded tensor whose activation is the 17408-wide intermediate."""
    folded = ckpt.gguf_tensor("blk.0.ffn_down.weight", rows=256) @ ckpt.rotation(17408)
    sibling = ckpt.sibling_tensor(
        "model.language_model.layers.0.mlp.down_proj.weight", rows=256
    )
    assert _cosine(folded, sibling) > TERNARY_COSINE


def test_o_proj_unfolds_at_width_6144(ckpt: _Checkpoint) -> None:
    """``o_proj`` reads a concatenated head axis, so it takes the 6144 signs."""
    folded = ckpt.gguf_tensor("blk.3.attn_output.weight") @ ckpt.rotation(6144)
    sibling = ckpt.sibling_tensor(
        "model.language_model.layers.3.self_attn.o_proj.weight"
    )
    assert _cosine(folded, sibling) > TERNARY_COSINE


def test_linear_attention_packed_qkv_is_ordered_q_k_v(ckpt: _Checkpoint) -> None:
    """The packed projection is ``[q 2048][k 2048][v 6144]``, and only v is tiled."""
    unfolded = ckpt.gguf_tensor("blk.0.attn_qkv.weight") @ ckpt.rotation(5120)
    sibling = ckpt.sibling_tensor(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    perm = ckpt.head_permutation()
    for start, end, name in ((0, 2048, "q"), (2048, 4096, "k")):
        assert _cosine(unfolded[start:end], sibling[start:end]) > TERNARY_COSINE, name
    # The value block is stored tiled, so it matches only after the swap.
    v_file = unfolded[4096:]
    v_sibling = sibling[4096:]
    assert abs(_cosine(v_file, v_sibling)) < DECORRELATED
    assert _cosine(_Checkpoint.reorder_heads(v_file, perm), v_sibling) > TERNARY_COSINE
    # ...and the swap is not its own index map, so the other direction must not work.
    assert (
        _cosine(_Checkpoint.reorder_heads(v_file, torch.argsort(perm)), v_sibling)
        < DECORRELATED
    )


def test_ssm_out_reads_a_grouped_value_axis(ckpt: _Checkpoint) -> None:
    """``ssm_out`` is the one value-axis tensor stored in the sibling's order.

    This is the asymmetry ``gdn_v_grouped`` describes: the gated-DeltaNet output
    leaves the scan tiled, and the weight here expects it grouped, so the runtime
    permutes the activation rather than the weight.
    """
    unfolded = ckpt.gguf_tensor("blk.0.ssm_out.weight") @ ckpt.rotation(6144)
    sibling = ckpt.sibling_tensor(
        "model.language_model.layers.0.linear_attn.out_proj.weight"
    )
    columns = unfolded.t()
    assert _cosine(columns, sibling.t()) > TERNARY_COSINE
    permuted = unfolded.reshape(5120, GROUPS * REP, HD)[:, ckpt.head_permutation()]
    assert _cosine(permuted.reshape(5120, -1).t(), sibling.t()) < DECORRELATED


def test_value_head_vectors_are_tiled_and_ungrouped(ckpt: _Checkpoint) -> None:
    """``A_log``, ``dt_bias`` and the two small projections carry the same tiled order.

    These are what tie the scan's head indexing to the weights' row order, so a
    conversion that moved one and not the others would leave the model running with
    each value head reading another head's decay.
    """
    perm = ckpt.head_permutation()

    # A_log is stored as -exp(A_log): its log is the sibling's tensor.
    a_log = torch.log(-ckpt.gguf_tensor("blk.0.ssm_a")).flatten()
    sibling_a_log = ckpt.sibling_tensor(
        "model.language_model.layers.0.linear_attn.A_log"
    ).flatten()
    assert abs(_corr(a_log[perm], sibling_a_log)) > 0.99
    # The permutation is not its own index map, so the file's own order is the
    # negative control rather than a restatement of the same check.
    assert abs(_corr(a_log, sibling_a_log)) < DECORRELATED

    dt_bias = ckpt.gguf_tensor("blk.0.ssm_dt.bias").flatten()
    sibling_dt = ckpt.sibling_tensor(
        "model.language_model.layers.0.linear_attn.dt_bias"
    ).flatten()
    assert abs(_corr(dt_bias[perm], sibling_dt)) > 0.99
    assert abs(_corr(dt_bias, sibling_dt)) < DECORRELATED

    # These two are not folded at all, so they compare directly after the swap.
    for gguf_name, sibling_name in (
        ("blk.0.ssm_alpha.weight", "in_proj_a"),
        ("blk.0.ssm_beta.weight", "in_proj_b"),
    ):
        rows = ckpt.gguf_tensor(gguf_name)
        sibling_rows = ckpt.sibling_tensor(
            f"model.language_model.layers.0.linear_attn.{sibling_name}.weight"
        )
        reordered = _Checkpoint.reorder_heads(rows, perm, head_dim=1)
        assert reordered.shape == sibling_rows.shape
        assert _cosine(reordered, sibling_rows) > DIRECT_COSINE
        assert abs(_cosine(rows, sibling_rows)) < DECORRELATED


def test_the_convolution_channels_carry_the_same_tiled_value_axis(
    ckpt: _Checkpoint,
) -> None:
    """The depthwise conv is tap-major and its value channels are tiled too."""
    conv = ckpt.gguf_tensor("blk.0.ssm_conv1d.weight")
    sibling = ckpt.sibling_tensor(
        "model.language_model.layers.0.linear_attn.conv1d.weight"
    ).squeeze(1)
    assert conv.shape == sibling.shape == (10240, 4)
    # Not folded, and both sides are fp32 here, so the q/k half is nearly exact.
    # Flattened: a four-element row is too short for a per-row cosine to mean much.
    assert _cosine(conv[:4096].flatten()[None], sibling[:4096].flatten()[None]) > 0.99
    v_file = conv[4096:]
    assert abs(_cosine(v_file, sibling[4096:])) < DECORRELATED
    assert (
        _cosine(
            _Checkpoint.reorder_heads(v_file, ckpt.head_permutation()), sibling[4096:]
        )
        > 0.99
    )


def test_the_token_embedding_takes_the_inverse(ckpt: _Checkpoint) -> None:
    """A lookup is not a matmul, so the embedding's fold is undone the other way.

    The stored rows need ``(1/sqrt(N)) H (s * x)`` applied *after* the lookup, which
    as a right-multiplication is the rotation matrix itself rather than its transpose.
    """
    rows = ckpt.gguf_tensor("token_embd.weight", rows=512)
    sibling = ckpt.sibling_tensor(
        "model.language_model.embed_tokens.weight", rows=512
    )
    assert abs(_cosine(rows, sibling)) < DECORRELATED
    assert _cosine(rows @ ckpt.rotation(5120), sibling) > TERNARY_COSINE
    # The wrong order of the signs and H is the transpose, and it decorrelates.
    inverse = ckpt.rotation(5120).t()
    assert abs(_cosine(rows @ inverse, sibling)) < DECORRELATED


def test_the_norms_are_the_sibling_plus_one(ckpt: _Checkpoint) -> None:
    """The GGUF folds the ``(1 + gamma)`` convention into the stored norm weights.

    The runtime applies ``1 + weight`` on CUDA, so a GGUF norm has to be loaded as
    ``weight - 1``.  The gated-DeltaNet norm is the exception: it is applied directly,
    which is what ``qwen_is_one_plus_norm_gamma`` already encodes for the sibling.
    """
    folded = [
        ("blk.0.attn_norm.weight", "layers.0.input_layernorm.weight", True),
        ("blk.0.post_attention_norm.weight", "layers.0.post_attention_layernorm.weight", True),
        ("output_norm.weight", "norm.weight", True),
        ("blk.3.attn_q_norm.weight", "layers.3.self_attn.q_norm.weight", True),
        ("blk.3.attn_k_norm.weight", "layers.3.self_attn.k_norm.weight", True),
        ("blk.0.ssm_norm.weight", "layers.0.linear_attn.norm.weight", False),
    ]
    for gguf_name, suffix, one_plus in folded:
        stored = ckpt.gguf_tensor(gguf_name)
        sibling = ckpt.sibling_tensor(f"model.language_model.{suffix}")
        direct = float(np.abs(stored - sibling).max())
        shifted = float(np.abs(stored - sibling - 1.0).max())
        # Every norm here is large enough that the 1.0 either dominates the residual
        # (folded) or is the entire residual (not folded), so the margin is wide.
        if one_plus:
            assert shifted < direct / 2, gguf_name
        else:
            assert direct < shifted / 2, gguf_name
