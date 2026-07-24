from __future__ import annotations

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
