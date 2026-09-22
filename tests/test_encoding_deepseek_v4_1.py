"""Reading a V4.1 completion back: the tool-call cut, and the facts about the checkpoint behind it.

The adapter in :mod:`pocketllm.backends.v41_backend` streams a *text*, and the format's tool-call
block is text too -- V4.1 wraps it in ``<｜DSML｜ calls>``, whose ``｜`` is a fullwidth vertical line
and not the ASCII character a reader expects, so the block is tokenized as ordinary text and reaches
a client that is being sent the running decode. :func:`cut_tool_calls` is what stops the answer where
the checkpoint's own parser stops reading its content, and the second half of this file pins the
facts about the released checkpoint that the cut is written against -- that the marker is the
encoder's own start token, that the block is plain text rather than special tokens, and that the
checkpoint's parser reads it into a call with no ``id``.

Those three are properties of the checkpoint rather than of this package, and none of them can be
had without the checkpoint: they are skipped when it is not on the host, with
``DEEPSEEK_V41_CHECKPOINT`` naming where it is. Read as a whole they are the argument for the
streaming code in ``v41_backend``: the cut is text, but the call is not -- it needs the
end-of-sentence token a client is never shown -- so the call travels on the last event, out of the
loop's own finished tokens.
"""

from __future__ import annotations

import json
import os

import pytest

from src.encoding.deepseek_v4_1 import (
    THINKING_END,
    TOOL_CALLS_START,
    cut_tool_calls,
    load_encoder,
    parse_strict,
    split_completion,
)


CHECKPOINT = os.environ.get("DEEPSEEK_V41_CHECKPOINT", "/mnt/data3/DeepSeek-V4.1-Flash")
HAS_CHECKPOINT = os.path.isdir(CHECKPOINT)


# ---------------------------------------------------------------------------- the cut


#: The block as a generation writes it, spelled out: two newlines, the tag opened and closed, one
#: call with one string parameter. The ``｜`` are U+FF1C, as in the checkpoint's own constants.
BLOCK = (
    "\n\n<｜DSML｜ calls>\n"
    '<｜DSML｜ invoke name="builtin_web_search">\n'
    '<｜DSML｜ parameter name="additionalContext" string="true">短篇温馨小故事</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)


def test_the_answer_stops_where_the_block_opens():
    assert cut_tool_calls("answer" + BLOCK) == "answer"
    assert cut_tool_calls(BLOCK) == ""
    assert cut_tool_calls("answer" + BLOCK + "trailing markup") == "answer"


def test_a_trailing_fragment_of_the_tag_is_held_back_until_it_resolves():
    """The running decode's cost: at the character before the tag the cut cannot know whether the
    next token opens a block, and a stream cannot take a character back."""
    for size in range(1, len(TOOL_CALLS_START)):
        assert cut_tool_calls("answer" + TOOL_CALLS_START[:size]) == "answer"
    # And the fragment goes out once the next character rules the tag out.
    assert cut_tool_calls("answer" + TOOL_CALLS_START[:-1] + "x") == (
        "answer" + TOOL_CALLS_START[:-1] + "x"
    )


def test_a_text_that_is_not_a_prefix_of_the_tag_is_untouched():
    assert cut_tool_calls("answer") == "answer"
    assert cut_tool_calls("") == ""
    assert cut_tool_calls("answer\n") == "answer"  # one newline is still the tag's first character
    assert cut_tool_calls("answer\nx") == "answer\nx"
    # The ASCII marker is a different string, and V4's own block is not this format's.
    assert cut_tool_calls("answer<tool_calls>") == "answer<tool_calls>"


def test_a_held_back_fragment_only_ever_moves_the_cut_forward():
    """What the streaming diff rests on: the text a caller is given grows, never rewrites."""
    seen = ""
    for character in "answer" + TOOL_CALLS_START + "markup":
        current = cut_tool_calls(seen + character)
        assert current.startswith(seen) or seen.startswith(current)
        seen += character


# ---------------------------------------------------------------------------- the checkpoint


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no checkpoint at {CHECKPOINT}")
def test_the_marker_is_the_start_token_the_checkpoint_itself_matches_on():
    """Written out in the module rather than read off the encoder, because the readers run on a text
    that may be truncated at any character -- so the two are compared here instead."""
    encoder = load_encoder(CHECKPOINT)
    assert TOOL_CALLS_START == f"\n\n<{encoder.dsml_token}{encoder.tool_calls_block_name}"


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no checkpoint at {CHECKPOINT}")
def test_the_block_is_plain_text_and_a_decode_flag_does_not_hide_it():
    """Why the block reaches a client at all: the tag is text to the tokenizer, so no cleanup of
    special tokens removes it. This is also why the streaming path cannot be fixed by a decode flag.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    ids = tokenizer.encode(BLOCK, add_special_tokens=False)

    assert not [token for token in ids if token in set(tokenizer.all_special_ids)]
    assert tokenizer.decode(ids, skip_special_tokens=True) == BLOCK
    assert tokenizer.decode(ids, skip_special_tokens=False) == BLOCK


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no checkpoint at {CHECKPOINT}")
@pytest.mark.parametrize(
    "thinking_mode, head",
    [("chat", ""), ("thinking", "我需要搜一下。\n" + THINKING_END)],
)
def test_the_parser_reads_the_block_into_a_call_where_the_cut_stops_the_answer(
    thinking_mode, head
):
    """The cut and the parse are the same boundary. The parse is also the only place a call exists:
    it wants the end-of-sentence token, which is stripped from everything a client sees, and the
    call it returns carries no ``id`` -- which is why the adapter mints one."""
    encoder = load_encoder(CHECKPOINT)
    text = head + BLOCK + encoder.eos_token

    parsed = parse_strict(CHECKPOINT, text, thinking_mode=thinking_mode)

    assert parsed is not None
    assert parsed["content"] == ""
    # The reasoning half is not the answer's, so the cut runs on the content half -- the same
    # order the streaming path applies them in.
    assert cut_tool_calls(split_completion(text, thinking_mode=thinking_mode)["content"]) == ""
    assert parsed["reasoning_content"] == ("" if thinking_mode == "chat" else "我需要搜一下。\n")
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "builtin_web_search"
    assert json.loads(call["function"]["arguments"]) == {
        "additionalContext": "短篇温馨小故事"
    }
    assert "id" not in call


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no checkpoint at {CHECKPOINT}")
def test_an_answer_that_is_only_an_answer_keeps_its_own_trailing_newline():
    """The cut is not a stripper: what precedes the block is the answer, newlines and all."""
    encoder = load_encoder(CHECKPOINT)
    text = "故事是这样的。\n" + BLOCK + encoder.eos_token

    parsed = parse_strict(CHECKPOINT, text, thinking_mode="chat")

    assert parsed is not None
    assert parsed["content"] == "故事是这样的。\n"
    assert cut_tool_calls(text) == "故事是这样的。\n"


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no checkpoint at {CHECKPOINT}")
def test_the_released_tokenizer_splits_the_character_before_the_block_across_two_tokens():
    """The other half of what a running decode costs, measured rather than assumed: the emoji is
    four bytes in one token after a lead-in that is not, and the decode of the tokens up to but not
    including the last is the replacement character a stream must not send."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    ids = tokenizer.encode("你好！😊", add_special_tokens=False)

    assert len(ids) == 4
    assert tokenizer.decode(ids[:3]) == "你好！�"
    assert tokenizer.decode(ids) == "你好！😊"
