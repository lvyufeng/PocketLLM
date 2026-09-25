"""Regenerate ``tests/data/ternary_bonsai_hadamard_transform.json`` from the fork.

The transform in ``src/loader/gguf/prism_hadamard.py`` is the activation side of a
weight fold, and a mistake in it produces a model that still runs and still writes
fluent text -- so it is pinned against the reference implementation rather than
against a second reading of the same comment.  ``scripts/prism_hadamard_oracle.cpp``
is the fork's own code (its ``ggml_permute``, ``ggml_mul`` and its FWHT ``mul_mat``
path) applying the same three ops, in the same order, that
``llama-graph.cpp``'s ``build_lora_mm`` applies; this script drives it and writes the
answers into the fixture the tests read.

The activations are synthetic -- this repository does not run the model yet, and a
random 5120-wide row is as good a probe of a linear transform as a real one.  The
sign vectors are the checkpoint's own, and the widths are the three the block
declares.  What is *not* claimed here is end-to-end parity: that arrives with the
kernel (#386), where the same transform feeds a real weight matrix and the output is
compared against the fork's logits.

    python scripts/dump_prism_hadamard_oracle.py

Environment: ``POCKETLLM_PRISM_LLAMA_CPP`` (the fork's source tree, for headers and
libraries) and ``POCKETLLM_HADAMARD_ORACLE`` (a prebuilt harness).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.loader.gguf.prism_hadamard import HadamardRotation  # noqa: E402
from tests.hadamard_test_utils import hadamard_spec  # noqa: E402

FORK_SOURCE = Path(os.environ.get("POCKETLLM_PRISM_LLAMA_CPP", "/mnt/data1/llama_cpp_prism"))
FORK_REVISION = "842b188"
FORK_BRANCH = "prism"
FORK_BUILD = FORK_SOURCE / "build-sm75" / "bin"
HARNESS_SOURCE = REPO / "scripts" / "prism_hadamard_oracle.cpp"
OUTPUT = REPO / "tests" / "data" / "ternary_bonsai_hadamard_transform.json"

#: The fixture's generator, stated rather than implied: one row per case would be
#: cheaper, but two rows catch a transform that mixes rows.
SEED = 20260925
ROWS = 2

#: name, mode, width, and the tensor whose pipeline the case reproduces.  ``signs``
#: is what a folded matrix weight sees, ``gdn`` is the same plus the gated-DeltaNet
#: permute, ``inverse`` is the token embedding's, and ``permute`` is the permute
#: alone, so a failure localizes to one of the three ops.
CASES = (
    ("forward_hidden", "signs", 5120, "blk.0.ffn_gate.weight"),
    ("forward_mlp", "signs", 17408, "blk.0.ffn_down.weight"),
    ("forward_gdn", "gdn", 6144, "blk.0.ssm_out.weight"),
    ("permute_gdn", "permute", 6144, "blk.0.ssm_out.weight"),
    ("inverse_token_embd", "inverse", 5120, "token_embd.weight"),
)


def _harness() -> Path:
    """The oracle binary: the one named in the environment, or a fresh build."""
    given = os.environ.get("POCKETLLM_HADAMARD_ORACLE")
    if given:
        return Path(given)
    binary = Path(tempfile.gettempdir()) / "prism_hadamard_oracle"
    compiler = shutil.which("g++")
    if compiler is None:
        raise SystemExit("no g++; set POCKETLLM_HADAMARD_ORACLE to a built harness")
    if not FORK_BUILD.is_dir():
        raise SystemExit(f"no fork build at {FORK_BUILD}; set POCKETLLM_PRISM_LLAMA_CPP")
    command = [
        compiler, "-O2", "-std=c++17", f"-I{FORK_SOURCE / 'ggml' / 'include'}",
        str(HARNESS_SOURCE), "-o", str(binary),
        f"-L{FORK_BUILD}", "-lggml", "-lggml-base", "-lggml-cpu",
        f"-Wl,-rpath,{FORK_BUILD}", "-pthread",
    ]
    subprocess.run(command, check=True)
    return binary


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def _run(binary: Path, work: Path, case: str, mode: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + width)
    x = rng.standard_normal((ROWS, width), dtype=np.float32)
    path = work / f"{case}-x.f32"
    x.tofile(path)
    out = work / f"{case}-out.f32"
    command = [str(binary), str(path), str(out), str(width), str(ROWS), mode]
    if mode != "permute":
        # The checkpoint's own sign vector for this width: the point of the fixture is
        # that the two sides consume the real signs in the same order.
        signs = np.asarray(rotation.signs_for(width).numpy(), dtype=np.float32)
        signs_path = work / f"{case}-signs.f32"
        signs.tofile(signs_path)
        command.append(str(signs_path))
    subprocess.run(command, check=True)
    return x, np.fromfile(out, dtype=np.float32).reshape(ROWS, width)


rotation = HadamardRotation(hadamard_spec())
binary = _harness()

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    cases = []
    for name, mode, width, tensor in CASES:
        x, y = _run(binary, work, name, mode, width)
        cases.append({
            "name": name,
            "mode": mode,
            "width": width,
            "tensor": tensor,
            "x_sha256": _digest(x),
            "out_sha256": _digest(y),
            "out_rms": float(np.sqrt((y.astype(np.float64) ** 2).mean())),
            "out_max_abs": float(np.abs(y).max()),
            "out_first8": [float(v) for v in y.reshape(-1)[:8]],
        })
        print(f"{name:20s} width={width:6d} rms={cases[-1]['out_rms']:.6f} sha={cases[-1]['out_sha256'][:16]}")

fixture = {
    "source": (
        f"{FORK_SOURCE} at {FORK_REVISION}, branch {FORK_BRANCH}, via "
        "scripts/prism_hadamard_oracle.cpp and scripts/dump_prism_hadamard_oracle.py"
    ),
    "note": (
        "The fork's own composition of the three ops a folded weight's activation passes "
        "through -- the gated-DeltaNet permute, the sign multiply, the Walsh-Hadamard "
        "rotation -- applied to the checkpoint's real sign vectors at the three folded "
        "widths. The activation rows are synthetic; the signs are not. Each case states "
        "the sha256 of the fp32 little-endian input and output bytes, so the test compares "
        "digests and not a restatement of the arithmetic."
    ),
    "input": {
        "generator": "numpy.random.default_rng(SEED + width).standard_normal((ROWS, width), dtype=float32)",
        "seed": SEED,
        "rows": ROWS,
        "note": "one seed per width, so the cases are independent; two rows catch a transform that mixes them",
    },
    "block_size": rotation.block_size,
    "gdn_geometry": {"value_heads": rotation.gdn.value_heads, "groups": rotation.gdn.groups},
    "cases": cases,
}
OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n")
print(f"wrote {OUTPUT.relative_to(REPO)} ({OUTPUT.stat().st_size} bytes)")
