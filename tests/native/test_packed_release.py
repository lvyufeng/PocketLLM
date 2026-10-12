"""The Ascend backend lets the checkpoint's packed bytes go once its decode exists.

`Weight::blocks` holds the checkpoint's own q4_K/q6_K bytes on the device, and
on this board those bytes are the *source* of the form the graph actually reads:
the fp16 plane `gemm_quant` caches per weight, and the f32 table
`embedding_quant` caches for the embedding.  On the 8B the two sets are 4.68 GiB
(packed) and 16.4 GiB (decoded) resident at once for the model's whole life, and
nothing reads the packed set again after the first decode.  `Qwen3Model::load`
therefore has the backend build every decode and then release the bytes --
`Backend::materializes_packed_weights()`, which only Ascend answers true to.

A release is the kind of change that goes wrong quietly: a freed handle still
looks like a handle, and a lookup that missed its cache would decode device
memory the allocator has already handed to something else and use the result as
a weight.  So the backend refuses a miss on a released handle instead of
decoding it, and this file pins the two properties that matter:

* **The tokens do not move.**  One greedy sequence, decoded end to end, must be
  the same tokens with the release on as with it off --
  `$POCKETLLM_ASCEND_KEEP_PACKED=1` is the backend's own kill switch, so the
  "before" side is the code as it was rather than a hand-built expectation.
* **The bytes actually go.**  The released run's peak resident set must be lower
  by the packed bytes.  That is the whole point of the change -- the 8B is
  supposed to fit on this board because of it -- and a release that quietly
  stopped happening would pass every numerical test in the tree.

Each side runs in its own child process.  The backend owns the device
(`aclInit`/`aclFinalize`) and its decode caches are never evicted, so two
sessions in one process would measure the first one's residency as much as the
second's.  Everything skips without the library, without `/proc/self/status`, or
without a checkpoint whose weights are packed -- a skip is not a pass.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from pocketllm import native
from pocketllm.kernels.dtypes import quant_format_by_file_id
from pocketllm.loader.gguf.reader import GGUFReader

#: Directories scanned for checkpoints, in order: the override first (one path
#: per `os.pathsep`), then where the machines of this project keep them.  The
#: same set `test_head_release.py` scans.
ENV_DIRS = "POCKETLLM_TEST_MODELS"
DEFAULT_DIRS = ("/mnt/data1/models", str(pathlib.Path.home() / "models"))

#: The backend's kill switch: set, the release is off and the packed bytes stay
#: resident -- exactly the behaviour of the code before the release existed.
KEEP_PACKED = "POCKETLLM_ASCEND_KEEP_PACKED"

#: The prompt, and how many greedy tokens to decode from it.  Long enough that a
#: wrong weight would have moved a token somewhere along the way, short enough
#: that a 0.6B decode at the board's rate stays inside a minute.
PROMPT = "The capital of France is"
STEPS = 8

#: The child process.  It prints one JSON object: the greedy ids and this
#: process's peak resident set.  VmHWM and not VmRSS because the question is
#: what the run needed, not what was left over at the end.
PROBE = r'''
import json
import sys

from pocketllm import native


def peak_rss():
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise SystemExit("no VmHWM line in /proc/self/status")


checkpoint, steps, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with native.Engine.open(checkpoint, "ascend") as engine:
    tokens = engine.encode(prompt, add_special=False, parse_special=True)
    token = engine.next_token(tokens)
    ids = [token]
    for _ in range(steps):
        token = engine.next_token([token])
        ids.append(token)
print(json.dumps({"ids": ids, "peak": peak_rss()}))
'''

#: What a run that could not reach the board's runtime looks like.  These turn a
#: failed probe into a skip -- the device or the CANN runtime is the reason, not
#: the code -- and anything else is a failure with the child's own message.
UNAVAILABLE = ("no backend named", "is not built", "aclInit", "aclrtSetDevice", "aclrtCreateContext")


def _model_dirs() -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    override = os.environ.get(ENV_DIRS)
    if override:
        dirs.extend(pathlib.Path(part) for part in override.split(os.pathsep) if part)
    dirs.extend(pathlib.Path(path) for path in DEFAULT_DIRS)
    return dirs


def _header(path: pathlib.Path):
    """The file's tensor index, or None when the file cannot be read as GGUF."""
    try:
        return GGUFReader(str(path)).read()
    except Exception:
        return None


def _pick() -> pathlib.Path | None:
    """The smallest checkpoint with packed weights -- the release is only
    defined for one, and the smallest keeps the two probe runs short."""
    best: pathlib.Path | None = None
    for directory in _model_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.gguf")):
            parsed = _header(path)
            if parsed is None:
                continue
            if not any(
                quant_format_by_file_id(info.type_id) is not None
                for info in parsed.tensors_by_name.values()
            ):
                continue
            if best is None or path.stat().st_size < best.stat().st_size:
                best = path
    return best


def _packed_bytes(path: pathlib.Path) -> int:
    """How many bytes of the checkpoint are packed blocks: what the release
    hands back.  A tensor whose type has a block format is packed; f32 and f16
    are bound dense and were never `Weight::blocks`."""
    parsed = _header(path)
    assert parsed is not None
    return sum(
        int(info.nbytes)
        for info in parsed.tensors_by_name.values()
        if quant_format_by_file_id(info.type_id) is not None
    )


def _probe(checkpoint: pathlib.Path, keep_packed: bool) -> dict:
    env = dict(os.environ)
    if keep_packed:
        env[KEEP_PACKED] = "1"
    else:
        env.pop(KEEP_PACKED, None)
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(checkpoint), str(STEPS), PROMPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if done.returncode != 0:
        message = (done.stderr or "").strip().splitlines()
        last = message[-1] if message else ""
        if any(marker in last for marker in UNAVAILABLE):
            pytest.skip(f"no Ascend runtime for this build: {last}")
        raise AssertionError(f"the probe failed ({done.returncode}):\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs() -> tuple[pathlib.Path, dict, dict]:
    """The two probe runs: release on, then the kill-switch run."""
    checkpoint = _pick()
    if checkpoint is None:
        pytest.skip("no packed checkpoint found")
    return checkpoint, _probe(checkpoint, keep_packed=False), _probe(checkpoint, keep_packed=True)


needs_engine = pytest.mark.skipif(
    not native.is_available(), reason="libpocketllm.so is not built"
)
needs_linux = pytest.mark.skipif(
    not pathlib.Path("/proc/self/status").is_file(),
    reason="the RSS evidence reads /proc/self/status",
)

pytestmark = [needs_engine, needs_linux]


def test_the_release_does_not_move_a_token(runs: tuple[pathlib.Path, dict, dict]) -> None:
    """Greedy output is identical with the packed bytes released and kept.

    This is the identity gate the release has to pass: the tokens are the
    model's own answer, and the release is supposed to be a residency change
    with no arithmetic in it.  A cache miss on a released handle throws -- the
    backend refuses rather than decodes -- so a sequence that arrives at all is
    also the evidence that every decode the forward path asked for was built
    before the bytes went.
    """
    checkpoint, released, kept = runs
    assert released["ids"] == kept["ids"], (
        f"{checkpoint.name}: the released run decoded {released['ids']} and the "
        f"packed-kept run decoded {kept['ids']}; the release changed the model's answer"
    )


def test_the_release_gives_the_packed_bytes_back(runs: tuple[pathlib.Path, dict, dict]) -> None:
    """The peak resident set drops by the packed bytes.

    Half the packed bytes is the bound: the two runs build the same decoded
    planes and both pay the decode's own staging, so the difference between them
    is the packed set and nothing else -- and half of it is far more than any
    allocator's noise and far less than the bytes the release claims.
    """
    checkpoint, released, kept = runs
    packed = _packed_bytes(checkpoint)
    freed = kept["peak"] - released["peak"]
    assert freed >= packed // 2, (
        f"{checkpoint.name}: the released run peaked at {released['peak']} B and the "
        f"packed-kept run at {kept['peak']} B, a difference of {freed} B against "
        f"{packed} B of packed weights; the release did not give the bytes back"
    )