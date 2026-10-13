"""A row's arithmetic does not depend on the batch it rides in.

`equals_solo` is the serving contract: a stream batched with others is
byte-equal to the same stream run alone (asserted end to end by
`tests/serving/test_native_backend.py::test_batched_streams_match_their_solo_output`).
It can only hold if a row's result is a function of the row and the weights and
nothing else.  It did not hold: `Qwen3Model::matmul` selected the repacked
panel GEMM when a call's total row count was a multiple of eight and the panel
GEMV otherwise, so a 16-token prompt and the same prompt split 16 + one decode
step produced different logits -- a near-tie moved the token, and the batched
stream answered different text from its solo run.  Silent wrong text, not a
perf note.

This module pins the property one layer below the serving test, where a
failure names the call shape that broke it:

* **16 + 1 vs 17** -- a decode step appended to an aligned prefill, the shape
  that diverged;
* **8 + 8 vs 16** -- an aligned split inside the prefill;
* **a ragged two-sequence batch** (16 and 20 tokens, one call at a total that
  is not a multiple of eight) against each sequence's own solo run, greedy ids
  over several steps.

And the same three in a child process with `$POCKETLLM_CPU_REPACK=0`, because
the selector is read once per process and the row kernel is a different
arithmetic: an immutable row is a property of the dispatch, not of one layout.
The cases ran in the child for both arms so the two are symmetric and neither
inherits this process's engine state.

Both logits comparisons are bit-exact -- the logits are compared as packed
bytes, not as numbers -- because that is the contract: `equals_solo`,
not `close_to_solo`.  The batch case compares greedy ids (the batch API
returns ids, not logits), which is the sharpest form the batch path allows.

Everything skips without the library or a Qwen3-0.6B q4_k_m checkpoint -- any
quantization of it will do, because every comparison here is self-referential.
A skip is not a pass.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from pocketllm import native

#: Directories scanned for a checkpoint, in order: the override first (one path
#: per `os.pathsep`), then where the machines of this project keep them.
ENV_DIRS = "POCKETLLM_TEST_MODELS"
DEFAULT_DIRS = ("/mnt/data1/models", str(pathlib.Path.home() / "models"))

#: The child process.  It prints one JSON object: the three cases' verdicts and
#: the digests to show when one of them fails.
PROBE = r'''
import hashlib
import json
import struct
import sys

from pocketllm import native


def md5(values):
    return hashlib.md5(struct.pack(f"<{len(values)}f", *values)).hexdigest()[:12]


def same(a, b):
    return struct.pack(f"<{len(a)}f", *a) == struct.pack(f"<{len(b)}f", *b)


checkpoint = sys.argv[1]
with native.Engine.open(checkpoint, "cpu") as engine:
    ids = engine.encode(
        "The quick brown fox jumps over the lazy dog. "
        "The capital of France is Paris, and the capital of Italy is Rome.",
        add_special=False,
    )
    if len(ids) < 20:
        raise SystemExit(f"the probe sentence tokenizes to {len(ids)} tokens; 20 are needed")
    p16, p17 = ids[:16], ids[:17]

    # (1) One sequence, one call of 17 vs 16 followed by a one-token decode.
    whole = engine.forward(p17)
    engine.reset()
    engine.forward(p16)
    tail = engine.forward(p17[16:])
    case1 = {"equal": same(whole, tail), "whole": md5(whole), "split": md5(tail)}

    # (2) One sequence, one call of 16 vs two aligned calls of 8.
    engine.reset()
    whole = engine.forward(ids[:16])
    engine.reset()
    engine.forward(ids[:8])
    tail = engine.forward(ids[8:16])
    case2 = {"equal": same(whole, tail), "whole": md5(whole), "split": md5(tail)}

    # (3) Two sequences of different lengths in one ragged batch (16 + 20 rows,
    # a total that is not a multiple of eight) against each sequence's solo run.
    steps = 6
    a, b = ids[:16], ids[:20]
    engine.reset()
    engine.reserve(1024, 2)
    batched = [[], []]
    row = engine.next_tokens([a, b])
    for i in range(2):
        batched[i].append(row[i])
    for _ in range(steps - 1):
        row = engine.next_tokens([[batched[0][-1]], [batched[1][-1]]])
        for i in range(2):
            batched[i].append(row[i])

solo = []
for sequence in (a, b):
    with native.Engine.open(checkpoint, "cpu") as single:
        out = [single.next_token(sequence)]
        for _ in range(steps - 1):
            out.append(single.next_token([out[-1]]))
        solo.append(out)

case3 = {
    "equal": batched[0] == solo[0] and batched[1] == solo[1],
    "batched": batched,
    "solo": solo,
}

print(json.dumps({"case1": case1, "case2": case2, "case3": case3}), flush=True)
'''


def _model_dirs() -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    override = os.environ.get(ENV_DIRS)
    if override:
        dirs.extend(pathlib.Path(part) for part in override.split(os.pathsep) if part)
    dirs.extend(pathlib.Path(path) for path in DEFAULT_DIRS)
    return dirs


def _pick_checkpoint() -> pathlib.Path | None:
    """The smallest `qwen3-0.6b` `q4_k_m` checkpoint, whatever its file name's
    case -- every comparison here is self-referential, so any quantization of
    the model will do and the smallest keeps the two probe runs short."""
    best: pathlib.Path | None = None
    for directory in _model_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.gguf")):
            name = path.name.lower()
            if not name.startswith("qwen3-0.6b") or "q4_k_m" not in name:
                continue
            if best is None or path.stat().st_size < best.stat().st_size:
                best = path
    return best


@pytest.fixture(scope="module")
def checkpoint() -> pathlib.Path:
    found = _pick_checkpoint()
    if found is None:
        pytest.skip("no Qwen3-0.6B q4_k_m checkpoint found")
    return found


def _run_probe(checkpoint: pathlib.Path, repack: str) -> dict:
    env = dict(os.environ)
    env["POCKETLLM_CPU_REPACK"] = repack
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(checkpoint)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if done.returncode != 0:
        raise AssertionError(f"the probe failed ({done.returncode}):\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


needs_engine = pytest.mark.skipif(
    not native.is_available(), reason="libpocketllm.so is not built"
)

pytestmark = [needs_engine]


@pytest.mark.parametrize("repack", ["1", "0"])
def test_the_dispatch_is_uniform_at_every_call_shape(
    checkpoint: pathlib.Path, repack: str
) -> None:
    """The three shapes above, bit-equal to their own splits, on both arms.

    `repack=1` is the shipped panel layout and `repack=0` the row kernel; the
    two are different arithmetic, and the property -- a row's result does not
    depend on which rows ride with it -- has to hold for each of them
    independently.  Re-running one arm's cases by hand is how the original
    `m % 8 == 0` defect got in: the 8 + 8 and 16 + 16 comparisons passed, and
    the cross-boundary 16 + 1 case was never made.
    """
    report = _run_probe(checkpoint, repack)
    assert report["case1"]["equal"], (
        f"POCKETLLM_CPU_REPACK={repack}: a 17-token call and 16 + 1 differ "
        f"({report['case1']}) -- the 8-row boundary is deciding arithmetic again"
    )
    assert report["case2"]["equal"], (
        f"POCKETLLM_CPU_REPACK={repack}: a 16-token call and 8 + 8 differ "
        f"({report['case2']})"
    )
    assert report["case3"]["equal"], (
        f"POCKETLLM_CPU_REPACK={repack}: a ragged batch diverged from its solo runs "
        f"({report['case3']}) -- a stream's answer depends on its batch"
    )