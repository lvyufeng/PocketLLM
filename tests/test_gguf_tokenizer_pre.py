from __future__ import annotations

from typing import Mapping

import pytest

from src.encoding import gguf_tokenizer
from src.encoding.gguf_tokenizer import _pre_tokenizer_for, build_gguf_bpe_tokenizer


def _synthetic_bpe_metadata(pre: str | None) -> dict:
    """A minimal byte-level BPE vocab/merges for pre-tokenizer branch tests.

    Covers enough of the byte alphabet to encode ASCII digits/letters/space so
    the two pre-tokenizer branches can be distinguished by how they split
    ``"1234"`` (glm4 groups digits 1-3; plain ByteLevel does not).
    """

    # Byte-level tokens for the characters we exercise, using the GPT-2 byte map:
    # space -> 'Ġ', digits and letters map to themselves. Include the merged
    # piece "34" and a single merge rule "3 4" so the two pre-tokenizer branches
    # produce a *different final token count*: the glm4 split cuts "12345" into
    # "123"/"45", which straddles the "3 4" merge and prevents it, while plain
    # ByteLevel keeps "12345" as one pre-token and applies the merge.
    chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
    tokens = ["<unk>", "Ġ"] + chars + ["34"]
    token_type = [3] + [1] * (len(tokens) - 1)  # token 0 is a control/unknown
    return {
        "general.architecture": "test-arch",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["3 4"],
        "tokenizer.ggml.token_type": token_type,
        "tokenizer.ggml.unknown_token_id": 0,
    }


def test_pre_tokenizer_glm4_and_llama3_use_split_sequence() -> None:
    for pre in ("glm4", "chatglm-bpe", "llama3", "llama-bpe", "llama-v3"):
        pt = _pre_tokenizer_for(pre)
        assert type(pt).__name__ == "Sequence", pre
        # 1-3 digit grouping: "12345" -> "123","45"
        pieces = [tok for tok, _span in pt.pre_tokenize_str("12345")]
        assert pieces == ["123", "45"], (pre, pieces)


def test_pre_tokenizer_default_stays_plain_bytelevel() -> None:
    # Anything not in the llama3/glm4 family (incl. MiniMax's pre and None)
    # must keep the historical plain ByteLevel, unchanged.
    for pre in (None, "", "default", "minimax", "gpt2", "smaug-bpe"):
        pt = _pre_tokenizer_for(pre)
        assert type(pt).__name__ == "ByteLevel", pre
        # ByteLevel keeps the whole digit run together (no 1-3 split)
        pieces = [tok for tok, _span in pt.pre_tokenize_str("12345")]
        assert pieces == ["12345"], (pre, pieces)


def test_build_tokenizer_respects_glm4_pre(monkeypatch) -> None:
    monkeypatch.setattr(
        gguf_tokenizer,
        "read_gguf_tokenizer_metadata",
        lambda _path: _synthetic_bpe_metadata("glm4"),
    )
    tokenizer, _md = build_gguf_bpe_tokenizer("ignored")
    # glm4 split cuts "12345" into "123"/"45", so the "3 4" merge cannot apply
    # across the boundary -> five single-digit tokens.
    enc = tokenizer.encode("12345", add_special_tokens=False)
    assert enc.tokens == ["1", "2", "3", "4", "5"], enc.tokens


def test_build_tokenizer_default_pre_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(
        gguf_tokenizer,
        "read_gguf_tokenizer_metadata",
        lambda _path: _synthetic_bpe_metadata("minimax"),
    )
    tokenizer, _md = build_gguf_bpe_tokenizer("ignored")
    # plain ByteLevel keeps "12345" as one pre-token, so the "3 4" merge applies
    # -> "1","2","34","5" (four tokens). This is the historical behavior and
    # must be unchanged for non-glm4/llama3 models like MiniMax.
    enc = tokenizer.encode("12345", add_special_tokens=False)
    assert enc.tokens == ["1", "2", "34", "5"], enc.tokens


def test_pre_tokenizer_qwen35_uses_qwen_split() -> None:
    # Qwen's pattern is not the llama3-family one, and the difference is not
    # cosmetic: digits split one at a time rather than in groups of three, and
    # combining marks are part of a word.
    pt = _pre_tokenizer_for("qwen35")
    assert type(pt).__name__ == "Sequence"
    assert [tok for tok, _span in pt.pre_tokenize_str("12345")] == ["1", "2", "3", "4", "5"]
    # "e" + COMBINING ACUTE ACCENT stays attached to its letter here. The pieces
    # come back byte-level mapped, so an assertion on their text would be about
    # that mapping; this counts them instead. One pre-token for Qwen, two for the
    # llama3 pattern, which cuts the mark off the letter it belongs to.
    assert len(pt.pre_tokenize_str("e\u0301")) == 1
    assert len(_pre_tokenizer_for("llama3").pre_tokenize_str("e\u0301")) == 2


def test_build_tokenizer_respects_qwen35_pre(monkeypatch) -> None:
    monkeypatch.setattr(
        gguf_tokenizer,
        "read_gguf_tokenizer_metadata",
        lambda _path: _synthetic_bpe_metadata("qwen35"),
    )
    tokenizer, _md = build_gguf_bpe_tokenizer("ignored")
    # Single-digit split, so the "3 4" merge can never apply across "3" and "4"
    # and "12345" is five tokens -- the same count the glm4 branch produces for a
    # different reason, which is why the count alone is not the check.
    enc = tokenizer.encode("12345", add_special_tokens=False)
    assert enc.tokens == ["1", "2", "3", "4", "5"], enc.tokens


QWEN35_GGUF = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"


@pytest.mark.skipif(
    not __import__("os").path.exists(QWEN35_GGUF),
    reason="the released ternary checkpoint is not on disk",
)
def test_qwen35_tokenizer_matches_the_reference() -> None:
    """The ids llama.cpp produces for the same text, pinned.

    Taken from the fork's own vocabulary (``llama_tokenize`` with
    ``parse_special``), which is what the released file was quantized for. The
    first case is the one the engine test prefills, so the two files agree about
    the same prompt by construction.
    """

    from src.encoding.gguf_tokenizer import build_gguf_hf_tokenizer

    tokenizer, metadata = build_gguf_hf_tokenizer(QWEN35_GGUF)
    assert tokenizer.eos_token_id == metadata["tokenizer.ggml.eos_token_id"] == 248046
    assert tokenizer.chat_template

    cases = {
        "The capital of France is": [760, 6511, 314, 9338, 369],
        "1 2 33 444 5555": [16, 220, 17, 220, 18, 18, 220, 19, 19, 19, 220, 20, 20, 20, 20],
        "hello, world!": [14556, 11, 1814, 0],
        "café naïve": [895, 56868, 91603, 571],
        "你好，世界。": [109266, 3709, 96748, 1710],
        "def f(x): return x**2  # 12.5%": [
            727, 281, 2007, 1590, 460, 830, 332, 17, 220, 653, 220, 16, 17, 13, 20, 4,
        ],
    }
    for text, expected in cases.items():
        assert tokenizer.encode(text) == expected, text
        # ...and the decode is the same string back, which is what the response
        # text is built from.
        assert tokenizer.decode(expected) == text, text

    # The chat template is the file's own, and it addresses the same tokens.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=True, add_generation_prompt=True
    )
    # A list, or a mapping holding one: transformers returns the mapping for a
    # templated call in some versions and the list in others, which is the same
    # two shapes the protocol's normalizer accepts.
    rendering = (
        rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    )
    ids = [int(token) for token in rendering]
    assert 248045 in ids and 248046 in ids
    assert tokenizer.decode(ids).startswith("<|im_start|>system")
