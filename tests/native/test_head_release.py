"""A tied head must not be freed twice, and a separate head must be freed.

`Qwen3Model`'s destructor releases the embedding table and the output head. When
the checkpoint *ties* them, both members hold the same buffer and the head must
not be released a second time; when the checkpoint carries a separate one, both
must be released. Which of the two a checkpoint is, is a property of the file
-- a tied one stores no `output.weight` tensor and reuses `token_embd.weight`
as the projection, an untied one carries its own -- and the test that decides
must be the buffers' identity, never their shape. A Qwen3 head is
`(n_embd, n_vocab)` whether it is tied or not, so a shape comparison reads
every separate head as a tie and never releases it.

This pins both directions against real files, on the CPU backend, where
`release` is `delete[]` and a buffer that is never released is visible as
resident memory that never comes back:

* on an **untied** checkpoint, open/use/close cycles must not grow the
  process's resident set -- a destructor that reads the separate head as a tie
  leaks it once per cycle (0.24 GiB on the 1.7B, 0.48 GiB on the 8B), and that
  shows up as per-cycle growth of one head's worth;
* on a **tied** checkpoint, the same cycles must not abort -- a destructor
  that releases the head whenever it has rows would `delete[]` the table
  twice, and a double free corrupts the heap.

The checkpoints are discovered by scanning directories (never by name): the
tie verdict is read from each file's own tensor list with a header parse, the
same way the loader binds them. Everything skips without the library or
without a checkpoint of the needed kind; a skip is not a pass.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from pocketllm import native
from pocketllm.loader.gguf.reader import GGUFReader

#: Directories scanned for checkpoints, in order: the override first (one
#: path per `os.pathsep`), then where the machines of this project keep them.
ENV_DIRS = "POCKETLLM_TEST_MODELS"
DEFAULT_DIRS = ("/mnt/data1/models", str(pathlib.Path.home() / "models"))

#: How many open/use/close cycles. The first is warm-up for whatever the
#: allocator and the loader do once, and the *last two* are compared.
CYCLES = 3

#: The growth allowed between the last two cycles. A leaked head is one whole
#: head per cycle, so half of it is a bound no leak can pass and no allocator
#: noise reaches.
NOISE_FRACTION = 0.5

TK = 1024


def _model_dirs() -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    override = os.environ.get(ENV_DIRS)
    if override:
        dirs.extend(pathlib.Path(part) for part in override.split(os.pathsep) if part)
    dirs.extend(pathlib.Path(path) for path in DEFAULT_DIRS)
    return dirs


def _candidates() -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for directory in _model_dirs():
        if directory.is_dir():
            found.extend(sorted(directory.glob("*.gguf")))
    return found


def _header(path: pathlib.Path):
    """The file's tensor index, or None when the file cannot be read as GGUF.

    A half-downloaded file in a model directory is not this test's subject;
    it is skipped here rather than allowed to fail the scan.
    """
    try:
        return GGUFReader(str(path)).read()
    except Exception:
        return None


def _tie_state(path: pathlib.Path) -> bool | None:
    """True when tied, False when the file carries a separate head, None when
    the file is unreadable or is not a checkpoint this test can classify.

    The question is whether the head and the embedding table are one buffer, so
    a file that has no `token_embd.weight` at all -- the draft exports (dflash,
    dspark) on the model hosts -- is neither tied nor untied here: reading it as
    "tied" picked it for the tied test, where `_head_bytes` then had no table to
    measure and the test died with a `KeyError` instead of skipping.  The gate
    is the tensor `_head_bytes` reads, not the file's name or architecture, so a
    checkpoint that has what the test needs is still selected regardless of how
    it is spelled."""
    parsed = _header(path)
    if parsed is None:
        return None
    tensors = parsed.tensors_by_name
    if "token_embd.weight" not in tensors:
        return None
    return "output.weight" not in tensors


def _pick(tied: bool) -> pathlib.Path | None:
    """The smallest checkpoint of the requested kind, so a board that holds
    the whole ladder still runs the 1.7B and not the 8B."""
    best: pathlib.Path | None = None
    for path in _candidates():
        if _tie_state(path) is not tied:
            continue
        if best is None or path.stat().st_size < best.stat().st_size:
            best = path
    return best


def _head_bytes(path: pathlib.Path) -> int:
    """The bytes of the buffer whose release the destructor's tie test
    decides: the separate head on an untied file, the table on a tied one."""
    parsed = _header(path)
    assert parsed is not None
    tensors = parsed.tensors_by_name
    info = tensors["output.weight"] if "output.weight" in tensors else tensors["token_embd.weight"]
    nbytes = info.nbytes
    assert nbytes is not None, f"{path}: no byte count for {info.name}"
    return int(nbytes)


def _rss_bytes() -> int:
    """This process's resident set, from /proc -- the CPU backend's device
    memory is the host heap, so a `delete[]` that never happens is visible
    here as pages that never come back."""
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * TK
    raise RuntimeError("/proc/self/status has no VmRSS line")


def _cycles(checkpoint: pathlib.Path) -> list[int]:
    """`CYCLES` open/use/close cycles on the CPU backend, returning the RSS
    after each close."""
    samples: list[int] = []
    for _ in range(CYCLES):
        with native.Engine.open(str(checkpoint), "cpu") as engine:
            tokens = engine.encode("The capital of France is", add_special=True)
            engine.forward(tokens)
        samples.append(_rss_bytes())
    return samples


UNTIED = _pick(tied=False)
TIED = _pick(tied=True)

needs_engine = pytest.mark.skipif(
    not native.is_available(), reason="libpocketllm.so is not built"
)
needs_linux = pytest.mark.skipif(
    not pathlib.Path("/proc/self/status").is_file(),
    reason="the RSS evidence reads /proc/self/status",
)

pytestmark = [needs_engine, needs_linux]


@pytest.mark.skipif(UNTIED is None, reason="no untied checkpoint found to read a leak from")
def test_a_separate_head_is_released_at_close() -> None:
    """The untied file's head does not survive a close.

    Before the destructor compared handles, every cycle leaked one head: the
    test was `output_.rows != tok_embd_.rows`, an untied Qwen3 head has the
    table's exact shape, and the release was skipped every time. Measured on
    the 1.7B this was the head's 255252480 B of resident set per cycle; after
    the fix the same loop is flat.
    """
    assert UNTIED is not None
    head = _head_bytes(UNTIED)
    samples = _cycles(UNTIED)
    growth = samples[-1] - samples[-2]
    assert growth < NOISE_FRACTION * head, (
        f"{UNTIED.name}: resident set grew by {growth} B over the last cycle "
        f"(head is {head} B); samples after each close: {samples}"
    )


@pytest.mark.skipif(TIED is None, reason="no tied checkpoint found to pin the double free")
def test_a_tied_head_is_not_released_twice() -> None:
    """The tied file's head and table are one buffer, freed once.

    A destructor that released the head by shape would call `delete[]` on the
    table's buffer twice here, and the second call is heap corruption -- the
    process dies inside the close, which this test refuses to do. The growth
    bound repeats the untied test's assertion for the tied layout: whatever
    the destructor releases, it must release the table exactly once.
    """
    assert TIED is not None
    head = _head_bytes(TIED)
    samples = _cycles(TIED)
    growth = samples[-1] - samples[-2]
    assert growth < NOISE_FRACTION * head, (
        f"{TIED.name}: resident set grew by {growth} B over the last cycle "
        f"(head is {head} B); samples after each close: {samples}"
    )