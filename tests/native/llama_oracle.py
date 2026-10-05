"""llama.cpp as the oracle for the model's arithmetic.

The tokenizer has its own oracle, and for the same reason: there is no Python
Qwen3 in this tree to compare against.  `backends/reference` is a set of numpy
*kernels*, not a graph, and `architectures/` ships only `toy` -- so the only
implementation that can say what the right logits are is the one the checkpoint
was converted for.  A second C++ implementation written from the same reading of
the format would agree with the first exactly where that reading was right and
disagree only where it was wrong in a different way.

What this wraps is `src/tools/oracle.cpp`, which is compiled here rather than by
CMake.  That is deliberate: `libpocketllm.so` is the deliverable and must not
depend on a llama.cpp checkout, so the dependency lives with the test that needs
it.  The first compilation is a few seconds, and only a missing llama.cpp -- not
a missing compiler -- makes it skip.

The ordering of the pieces matters and is the reason this is a module rather
than a fixture in the test: `oracle_tool` is a session-scoped fixture, the
compile happens once, and every test that needs an answer calls the same
subprocess wrapper.

## Flash attention is pinned off

`llama_context_default_params()` leaves `flash_attn_type = AUTO`, and AUTO turns
flash attention *on* for every backend that has a kernel -- the CPU one
included.  That is a different reduction over the attention span than a plain
softmax (a running maximum with a rescaled merge, against one shift), and the
two are far enough apart on a quantized checkpoint to move a near-tie: at the
second generated token of the suite's prompt the top-2 margin is 0.0925 with
flash attention off and 0.0196 with it on, against a logit difference that
reaches 1.16, so the two conventions pick different tokens.  On f16 the same
switch is 0.0004 of the logit spread, which is why the f16 tests never saw it.

The C engine's CPU attention kernel is the full-softmax one, so the oracle has
to be too or it reports llama.cpp's own kernel choice as the engine's error.
:func:`run` takes `flash_attn` for the other side of it -- the CUDA backend's
attention is a block-reduced shift, which lands on the same token as llama.cpp's
flash mode, and `test_quantized_forward.py` names per backend which convention
it compares against.  The default here is the CPU convention, because that is
the backend every test can run.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

LLAMA_CPP = pathlib.Path("/mnt/data1/llama.cpp-latest")
LLAMA_LIB = LLAMA_CPP / "build" / "bin" / "libllama.so"
LLAMA_BUILD = LLAMA_CPP / "build" / "bin"

#: What the oracle asks llama.cpp for unless the caller says otherwise.
#: `"off"` because the engine's CPU attention is the full-softmax reduction;
#: `"on"` agrees with the CUDA kernel, and `"auto"` is llama.cpp's shipped
#: default.  See the module docstring.
FLASH_ATTN = "off"


def repository_root() -> pathlib.Path:
    """`tests/native/llama_oracle.py` -> the checkout root."""
    return pathlib.Path(__file__).resolve().parents[2]


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "llama-logits"


def compile_tool() -> None:
    """Build the oracle against llama.cpp's own headers.

    Compiling against the header, rather than binding through `ctypes`, is the
    whole point: `llama_context_params` has twenty-nine fields and gains them
    without notice, and a hand-written struct that is one field out does not
    fail to compile -- it reads a garbage `n_ctx` and segfaults inside
    `llama_init_from_model`.  An earlier version of this oracle did exactly
    that.  The compiler knows the layout on every version; a transcription knows
    it on one.
    """
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-o",
            str(output),
            str(repository_root() / "src" / "tools" / "oracle.cpp"),
            f"-I{LLAMA_CPP / 'include'}",
            f"-I{LLAMA_CPP / 'ggml' / 'include'}",
            f"-L{LLAMA_BUILD}",
            "-lllama",
            "-lggml",
            f"-Wl,-rpath,{LLAMA_BUILD}",
        ],
        check=True,
        capture_output=True,
    )


def run(model_path: str, tokens: list[int], steps: int = 0, flash_attn: str | None = None) -> dict:
    """One call to the tool, parsed.

    `steps` of 0 asks for the logits of the last token of `tokens`; a positive
    `steps` also greedily generates that many more and returns them under
    ``"generated"``.

    `flash_attn` overrides :data:`FLASH_ATTN` with ``"on"``, ``"off"`` or
    ``"auto"``.  It exists so a caller that wants to see llama.cpp's *shipped*
    behaviour -- AUTO, which enables flash attention -- can ask for it by name
    instead of the module silently changing what every test compares against.
    """
    command = [str(tool_path()), model_path] + [str(t) for t in tokens]
    command += ["--flash-attn", FLASH_ATTN if flash_attn is None else flash_attn]
    if steps:
        command += ["--steps", str(steps)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"the llama.cpp oracle failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def logits(model_path: str, tokens: list[int], flash_attn: str | None = None) -> list[float]:
    return run(model_path, tokens, flash_attn=flash_attn)["logits"]


def generated(
    model_path: str, tokens: list[int], steps: int, flash_attn: str | None = None
) -> list[int]:
    return run(model_path, tokens, steps, flash_attn=flash_attn)["generated"]