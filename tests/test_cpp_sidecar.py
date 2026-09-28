"""Which vocabulary the native server's Python sidecar loads, and from where.

The sidecar is the C++ engine's only templating, and it is handed the same
``--ckpt`` the engine was: a directory for a safetensors export, one file for a
GGUF. Those two containers carry the vocabulary in different places, so the
branch that reads it is worth a test that does not need either artifact on disk
-- what is asserted here is which reader a path selects, which is the half that
can be wrong without anyone noticing. The readers themselves are covered where
they live (``tests/test_gguf_tokenizer_pre.py``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="the sidecar imports transformers to load a vocabulary")

from src.server import cpp_sidecar  # noqa: E402  (after the importorskip)


class _FakeAutoTokenizer:
    """Stands in for ``transformers.AutoTokenizer`` and records the path it was given."""

    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str):
        cls.calls.append(path)
        return ("auto", path)


@pytest.fixture()
def auto_tokenizer(monkeypatch):
    _FakeAutoTokenizer.calls = []
    monkeypatch.setattr(cpp_sidecar, "AutoTokenizer", _FakeAutoTokenizer)
    return _FakeAutoTokenizer


def test_a_gguf_checkpoint_is_read_out_of_its_own_header(monkeypatch, auto_tokenizer):
    """A released ternary artifact is one file with nothing beside it.

    ``AutoTokenizer.from_pretrained`` would be asked to open a directory that
    does not exist, and the engine would come up with no vocabulary at all.
    """

    monkeypatch.setattr(
        "pocketllm.backends.cpp_backend.gguf_checkpoint_file",
        lambda path: path if path.endswith(".gguf") else "",
    )
    seen: list[str] = []
    monkeypatch.setattr(
        "src.encoding.gguf_tokenizer.build_gguf_hf_tokenizer",
        # The real builder answers with (tokenizer, metadata), and the sidecar
        # unpacks exactly that.
        lambda path: (seen.append(path), (("gguf", path), {}))[1],
    )

    tokenizer = cpp_sidecar.load_tokenizer("/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")

    assert tokenizer == ("gguf", "/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    assert seen == ["/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"]
    assert auto_tokenizer.calls == []


def test_a_safetensors_checkpoint_still_goes_through_transformers(monkeypatch, auto_tokenizer):
    monkeypatch.setattr(
        "pocketllm.backends.cpp_backend.gguf_checkpoint_file", lambda _path: ""
    )
    monkeypatch.setattr(
        "src.encoding.gguf_tokenizer.build_gguf_hf_tokenizer",
        lambda _path: pytest.fail("a checkpoint with no GGUF must not be read as one"),
    )

    tokenizer = cpp_sidecar.load_tokenizer("/models/Qwen3.8-27B")

    assert tokenizer == ("auto", "/models/Qwen3.8-27B")
    assert auto_tokenizer.calls == ["/models/Qwen3.8-27B"]


def test_an_explicit_tokenizer_path_wins_over_the_checkpoint(monkeypatch, auto_tokenizer):
    """A caller who names a vocabulary wants that vocabulary.

    It is also the only way to serve a GGUF whose companion directory exists,
    and it stays the reason the GGUF branch is second rather than first.
    """

    monkeypatch.setattr(
        "pocketllm.backends.cpp_backend.gguf_checkpoint_file",
        lambda path: path if path.endswith(".gguf") else "",
    )
    monkeypatch.setattr(
        "src.encoding.gguf_tokenizer.build_gguf_hf_tokenizer",
        lambda _path: pytest.fail("the named tokenizer path is the one to read"),
    )

    cpp_sidecar.load_tokenizer("/models/model.gguf", "/models/tokenizer")

    assert auto_tokenizer.calls == ["/models/tokenizer"]
