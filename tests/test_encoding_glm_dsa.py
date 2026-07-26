from __future__ import annotations

from pathlib import Path

import pytest

from src.encoding.glm_dsa import (
    build_glm_dsa_tokenizer,
    decode_glm_dsa_ids,
    encode_glm_dsa_prompt,
    glm_dsa_context_info,
    render_glm_chat_prompt,
)

REAL_GLM_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")

# Known special-token ids in the GLM-5.2 vocab (confirmed from bundle metadata).
GMASK_ID = 154822
SOP_ID = 154824
SYSTEM_ID = 154826
USER_ID = 154827
ASSISTANT_ID = 154828
EOS_ID = 154820


def test_glm_chat_prompt_compact_framing() -> None:
    prompt = render_glm_chat_prompt("请用一句话介绍你自己。", system_prompt="You are helpful.")

    assert prompt.startswith("[gMASK]<sop>")
    assert "<|system|>\nYou are helpful." in prompt
    assert "<|user|>\n请用一句话介绍你自己。" in prompt
    assert prompt.endswith("<|assistant|>")
    assert "<think>" not in prompt


def test_glm_chat_prompt_without_system() -> None:
    prompt = render_glm_chat_prompt("hello")

    assert prompt == "[gMASK]<sop><|user|>\nhello<|assistant|>"


def test_glm_chat_prompt_can_start_thinking() -> None:
    prompt = render_glm_chat_prompt("hello", thinking=True)

    assert prompt.endswith("<|assistant|>\n<think>")


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="local GLM-5.2 GGUF bundle not present")
def test_real_glm_context_info() -> None:
    info = glm_dsa_context_info(REAL_GLM_PATH)

    assert info["architecture"] == "glm-dsa"
    # block_count is 79 but the trailing block is a NextN/MTP speculative-decode
    # layer, not part of the main transformer trunk (llama.cpp skips it). The
    # runnable trunk depth is block_count - nextn_predict_layers = 79 - 1 = 78.
    assert info["n_layers"] == 78
    assert info["context_length"] == 1048576
    assert info["eos_token_id"] == EOS_ID
    assert info["bos_token_id"] == GMASK_ID


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="local GLM-5.2 GGUF bundle not present")
def test_real_glm_chat_prompt_leading_special_token_ids() -> None:
    ids, prompt_text, metadata = encode_glm_dsa_prompt(
        REAL_GLM_PATH,
        "请用一句话介绍你自己。",
        chat=True,
        system_prompt="You are a helpful assistant.",
    )

    assert prompt_text.startswith("[gMASK]<sop><|system|>\n")
    assert prompt_text.endswith("<|assistant|>")
    # Framing tokens must map to their single dedicated ids, in order.
    assert ids[:3] == [GMASK_ID, SOP_ID, SYSTEM_ID]
    assert USER_ID in ids
    assert ids[-1] == ASSISTANT_ID
    assert int(metadata["tokenizer.ggml.eos_token_id"]) == EOS_ID


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="local GLM-5.2 GGUF bundle not present")
@pytest.mark.parametrize(
    "text",
    [
        "Hello world",
        "请用一句话介绍你自己。",
        "abc123def 4567",
        "混合 mixed 文本 123 with punctuation!",
    ],
)
def test_real_glm_tokenizer_roundtrip(text: str) -> None:
    ids, _prompt_text, _metadata = encode_glm_dsa_prompt(REAL_GLM_PATH, text, chat=False)
    decoded = decode_glm_dsa_ids(REAL_GLM_PATH, ids)

    assert decoded == text


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="local GLM-5.2 GGUF bundle not present")
def test_real_glm_number_split_matches_glm4_pre() -> None:
    # glm4 pre-tokenizer groups digits in runs of at most 3.
    tokenizer, _md = build_glm_dsa_tokenizer(REAL_GLM_PATH)
    enc = tokenizer.encode("12345", add_special_tokens=False)
    decoded = tokenizer.decode(enc.ids)

    assert decoded == "12345"
    # The digit run must not collapse into a single 5-digit token.
    assert all(len(tok.strip("Ġ")) <= 3 for tok in enc.tokens), enc.tokens


def test_glm_tokenizer_rejects_wrong_architecture(monkeypatch) -> None:
    import src.encoding.glm_dsa as glm_mod

    monkeypatch.setattr(
        glm_mod,
        "build_gguf_bpe_tokenizer",
        lambda _path: (object(), {"general.architecture": "minimax-m2"}),
    )
    with pytest.raises(ValueError, match="glm-dsa"):
        build_glm_dsa_tokenizer("ignored")
