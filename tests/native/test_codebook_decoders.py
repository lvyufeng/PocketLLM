"""Every format the C engine decodes, block for block against llama.cpp.

`test_op_conformance.py` compares the C kernels against `backends/reference`,
which decodes with `pocketllm.quant`. That comparison is exact in its arithmetic
and blind in one direction: where the kernel and the reference read the same bit
layout the same *wrong* way, every case passes while both are wrong together.

This file closes that direction by comparing the *Python decoder itself* against
llama.cpp's own `to_float`, on blocks taken out of files llama.cpp quantized. It
covers every format `src/quant/blocks.h` can decode, and that is the point: the
formats that need this most are the codebooks, where the decode is a table lookup
and a table read with the wrong stride, bias or sign layout produces weights that
look like weights. The same comparison, run by hand, is how this tree's `q3_K`
decoder was found to read ggml's six-bit scales as four and lose two thirds of
each block's scale range.

**The blocks are real, and that is what makes it a test of the format.** Two
decoders can agree bit for bit while both read a layout the checkpoint does not
use; taking the bytes from a file llama.cpp wrote and then asking llama.cpp to
decode them is what removes that possibility. Nothing here is authored by this
suite -- that is the other file's job.

Everything skips without llama.cpp or the checkpoints. A skip is not a pass.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pocketllm.loader.gguf.reader import GGUFReader
from pocketllm.quant import formats

from ggml_dequant import compile_tool, decode as ggml_decode, tool_path
from llama_oracle import LLAMA_LIB

#: One checkpoint per format the engine decodes. A llama.cpp K-quant file is not
#: uniform -- it promotes some tensors -- so each of these carries two or three
#: types, and the loop compares whatever types it finds that this tree decodes.
#: Produced from `qwen3-0.6b-f16.gguf` with `llama-quantize` -- one command per
#: K-quant, and for the IQ2 pair an `llama-imatrix` pass first, since llama.cpp
#: refuses those types without an importance matrix; see `test_quant_ladder.py`
#: for the commands. `q4_k_m` is the file the other tests in this directory
#: already use.
CHECKPOINTS = (
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-q4_k_m.gguf"),
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-q2_k.gguf"),
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-q3_k.gguf"),
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-q5_k.gguf"),
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-iq2_xxs.gguf"),
    pathlib.Path("/mnt/data1/models/qwen3-0.6b-iq2_xs.gguf"),
)

#: Blocks per tensor. A block is 256 weights and one tool call decodes a whole
#: run of them, so this bounds the work; a layout read the wrong way is wrong in
#: the first block as surely as in the last.
BLOCKS_PER_TENSOR = 64

#: Tensors per format, taken from the middle of the tensor's block run rather
#: than its start: an off-by-one that still lands inside the tensor is what a
#: first-block-only check would miss, and the middle is as far from both ends as
#: it gets.
TENSORS_PER_FORMAT = 2


def _needs_llama() -> str | None:
    if not LLAMA_LIB.is_file():
        return f"no llama.cpp build at {LLAMA_LIB}"
    return None


needs_llama = pytest.mark.skipif(_needs_llama() is not None, reason=_needs_llama() or "")


@pytest.fixture(scope="session", autouse=True)
def ggml_tool() -> None:
    """Compile `ggml_dequant.c` if it is not already built."""
    if _needs_llama() is None and not tool_path().is_file():
        compile_tool()


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda p: p.stem)
def test_the_python_decoder_matches_llama_cpp(checkpoint: pathlib.Path) -> None:
    """Every decodable type in the file, on real blocks, bit for bit."""
    if not checkpoint.is_file():
        pytest.skip(f"no checkpoint at {checkpoint} (llama-quantize the f16 one)")
    if _needs_llama() is not None:
        pytest.skip(_needs_llama() or "")

    gguf = GGUFReader(str(checkpoint)).read()
    by_type: dict[str, list] = {}
    for tensor in gguf.tensors:
        name = tensor.type_name
        if name in formats.FORMATS and formats.FORMATS[name].decode is not None:
            by_type.setdefault(name, []).append(tensor)
    assert by_type, f"{checkpoint.name}: no tensor of a format this tree decodes"

    with open(checkpoint, "rb") as handle:
        for name, tensors in sorted(by_type.items()):
            fmt = formats.format_for(name)
            for tensor in _spread(tensors, TENSORS_PER_FORMAT):
                total = tensor.elements // fmt.block_elems
                count = min(BLOCKS_PER_TENSOR, total)
                start = (total - count) // 2
                handle.seek(tensor.absolute_offset + start * fmt.block_bytes)
                raw = handle.read(count * fmt.block_bytes)
                blocks = np.frombuffer(raw, dtype=np.uint8).reshape(count, fmt.block_bytes)

                ours = fmt.decode(blocks).astype(np.float32)
                theirs = ggml_decode(tensor.type_id, raw).reshape(count, fmt.block_elems)
                assert count > 0 and theirs.size == ours.size
                # A NaN would agree with itself and make the comparison vacuous.
                assert np.isfinite(theirs).all(), f"{tensor.name}: llama.cpp decoded a non-finite value"

                differing = np.nonzero(
                    ours.view(np.uint32).reshape(-1) != theirs.view(np.uint32).reshape(-1)
                )[0]
                assert differing.size == 0, (
                    f"{tensor.name} ({name}): {differing.size} of {ours.size} weights differ from "
                    f"llama.cpp's decode; first at weight {int(differing[0])}: ours "
                    f"{ours.reshape(-1)[differing[0]]!r} against llama.cpp's "
                    f"{theirs.reshape(-1)[differing[0]]!r}"
                )


def _spread(items: list, count: int) -> list:
    """`count` items spread across `items`, so the samples are not all neighbours."""
    if len(items) <= count:
        return list(items)
    step = len(items) // count
    return [items[i * step] for i in range(count)]
