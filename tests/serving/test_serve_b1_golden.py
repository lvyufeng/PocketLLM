"""G1: a width-1 streamed request is byte-identical to what it was before Stage 2.

Stage 2 replaced the per-request `_run` loop in `server/native_backend.py` with one
scheduler thread, on the promise that the width-1 path does not move.  That promise
is only checkable against a *recorded* answer: a test that runs the new backend at
`max_batch_size=1` and compares it to a second new backend compares the path to
itself, and would pass through a regression in it.

So this module replays `tests/fixtures/golden/native_serve_b1.json`, which
`record_b1_golden.py` recorded from the module at the pre-Stage-2 rev, and asserts
the token ids, the text and the finish reason are unchanged.

It skips -- and a skip is not a pass -- when the exact checkpoint it was recorded
against is not on the machine, because a token id from one quantization is not an
answer about another.  The fixture's `checkpoint.sha256` decides, not a filename.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from pocketllm.api import EngineArgs
from pocketllm.server import native_backend

_FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "golden" / "native_serve_b1.json"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixture() -> dict:
    if not _FIXTURE.exists():
        pytest.skip(f"no recorded fixture at {_FIXTURE}")
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


_FIXTURE_DATA = None
_FIXTURE_LOAD_ERROR = None
try:
    _FIXTURE_DATA = _load_fixture()
except Exception as exc:  # noqa: BLE001 -- a broken fixture is a skip, not a crash
    _FIXTURE_LOAD_ERROR = exc


def _resolve_checkpoint(fixture: dict) -> pathlib.Path:
    if _FIXTURE_LOAD_ERROR is not None:
        pytest.skip(f"the fixture could not be read: {_FIXTURE_LOAD_ERROR!r}")
    wanted = fixture["checkpoint"]["sha256"]
    name = fixture["checkpoint"]["name"]
    candidates = [
        pathlib.Path("/mnt/data1/models") / name,
        pathlib.Path.home() / "models" / name,
    ]
    for candidate in candidates:
        if candidate.exists() and _sha256(candidate) == wanted:
            return candidate
    pytest.skip(
        f"{name} with sha256 {wanted[:12]}... is not on this machine, so the recorded "
        "token ids are not an answer about any checkpoint that is"
    )


pytestmark = [
    pytest.mark.skipif(
        _FIXTURE_DATA is None,
        reason="the recorded width-1 fixture is missing or unreadable",
    ),
    pytest.mark.skipif(
        not native_backend.is_available(),
        reason="the C engine is not built on this host, so `serve` has nothing to run",
    ),
]


def test_width_one_stream_is_unchanged_from_before_the_scheduler() -> None:
    """The recorded pre-Stage-2 answer, replayed through the scheduler's width-1 path.

    Token ids are the assertion that matters: identical text can come from different
    ids, and `_decoded_text` is one of the pieces the scheduler had to keep verbatim.
    """
    fixture = _FIXTURE_DATA
    checkpoint = _resolve_checkpoint(fixture)

    from pocketllm.protocol import build_chat_request

    request = build_chat_request(fixture["request"], request_id="g1-replay")
    backend = native_backend.NativeBackend(EngineArgs(model=str(checkpoint), max_batch_size=1), "cpu")
    try:
        events = list(backend.stream(request))
    finally:
        backend.close()

    token_ids = [event.token_id for event in events if event.token_id is not None]
    text = "".join(event.text or "" for event in events)
    finish_reason = events[-1].finish_reason

    expected = fixture["expected"]
    assert token_ids == expected["token_ids"], (
        f"the width-1 path moved: {token_ids} != {expected['token_ids']} "
        f"(recorded at {fixture['recorded_from']['rev']})"
    )
    assert text == expected["text"], f"{text!r} != {expected['text']!r}"
    assert finish_reason == expected["finish_reason"], (
        f"{finish_reason!r} != {expected['finish_reason']!r}"
    )