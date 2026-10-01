"""Per-token log probabilities, rendered the way OpenAI's chat endpoint states them.

``logprobs`` is a *response* field with no engine behind it: what an engine reports is a ranking per
generated position -- the probability of the token it drew, plus the top ``n`` candidates it
considered -- and OpenAI's shape is a rendering of that ranking into JSON. Ported from the C++
front end's ``render_logprobs`` so a caller cannot tell which host answered, and kept apart from the
adapter because the rendering is the same for every runtime that can rank a position.

Two cuts have to survive the rendering, and both are in this module's reason for existing:

* **The stop token the engine appended is not part of the answer**, so a ranking is reported for a
  position the text does not cover unless the two are cut together. The caller cuts the token vector
  before calling; :func:`complete` is what says it may.
* **A client stop sequence ends the answer inside a token.** The sequence is matched on decoded
  text, so the byte the text ends at can fall in the middle of a token's surface. That token's
  probability describes text the caller never received and cannot be lined up against anything, so
  the ranking is walked to the same byte the text was cut at and stops there. See
  :func:`render`'s ``text_bytes``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["complete", "render"]


def complete(rankings: Sequence[Any], token_count: int) -> bool:
    """Whether the engine ranked every position it was asked about.

    A ranking that arrives short, or with a position the engine left absent, would leave the
    response's ``content`` array describing fewer positions than the text covers -- and a caller
    reading a probability per token has no way to notice the two drifted apart. The native front end
    failed the choice instead of reporting less than was asked for, and so does this: the caller
    turns a ``False`` here into an engine error.

    Only a caller that asked for a ranking has any business calling this, so there is no "asked"
    flag: a request that did not ask gets an empty vector and reads nothing from it.

    ``token_count`` is the length of the *unstripped* token vector the engine generated, because
    that is what the ranking is parallel to. The comparison is against it rather than against the
    text, which the stop cut may have shortened.
    """
    if len(rankings) != token_count:
        return False
    return all(bool(getattr(entry, "present", False)) for entry in rankings)


def render(
    tokens: Sequence[int],
    rankings: Sequence[Any],
    *,
    decode: Callable[[list[int]], str],
    text_start: int,
    text_end: int,
    alternatives: int,
) -> dict[str, Any]:
    """OpenAI's ``logprobs`` object for one choice, as the dict a JSON response carries.

    One entry per token, reporting the token's text, the logarithm of the probability the model
    gave it, the UTF-8 bytes of that text, and -- only when the request named a count -- that
    position's ranked alternatives.

    The bytes are there because the text alone does not say where a token begins and ends: a caller
    reassembling an answer from token boundaries needs them, and a piece whose text decodes to
    nothing (a special token, say) is invisible without them.

    ``text_start`` and ``text_end`` are the byte span the response's own text occupies in the
    decoded token stream. Only tokens lying entirely inside it are reported, which is what makes the
    array describe the same positions as the text it sits beside -- and there are two reasons the
    span is not simply the whole stream. A chat answer is a suffix of what the engine decoded
    (the reasoning block is in front of it and is not part of ``content``), and either kind of answer
    can be cut short from the back, by a client stop sequence landing inside a token or by the
    answer parser dropping a trailing marker. A probability reported for a position outside the span
    describes text the caller never received and cannot be lined up against anything.

    ``alternatives`` is how many ranked candidates the request asked for, which is not the width of
    the ranking: a batch ranks every row at the widest width any row asked for, so an entry can hold
    more candidates than this request wants and is narrowed here.
    """
    kept: list[tuple[int, str]] = []
    covered = 0
    for index in range(min(len(tokens), len(rankings))):
        surface = decode([int(tokens[index])])
        width = len(surface.encode("utf-8"))
        if covered >= text_start and covered + width <= text_end:
            kept.append((index, surface))
        covered += width
        if covered >= text_end:
            break

    content: list[dict[str, Any]] = []
    for index, surface in kept:
        entry = rankings[index]
        item: dict[str, Any] = {
            "token": surface,
            "logprob": _json_number(float(getattr(entry, "logprob", 0.0))),
            "bytes": list(surface.encode("utf-8")),
        }
        top = _alternatives(entry, alternatives, decode)
        if top:
            item["top_logprobs"] = top
        content.append(item)
    return {"content": content}


def _alternatives(
    entry: Any, alternatives: int, decode: Callable[[list[int]], str]
) -> list[dict[str, Any]]:
    """The ranked candidates for one position, or an empty list when the request named none.

    An empty list rather than an absent key would be wrong: OpenAI omits ``top_logprobs`` on a
    request that did not ask for a count, and a client that distinguishes "no alternatives came
    back" from "I did not ask" reads the key's presence.
    """
    if alternatives <= 0:
        return []
    tokens = list(getattr(entry, "top_tokens", ()) or ())
    logprobs = list(getattr(entry, "top_logprobs", ()) or ())
    count = min(len(tokens), len(logprobs), alternatives)
    ranked: list[dict[str, Any]] = []
    for index in range(count):
        surface = decode([int(tokens[index])])
        ranked.append(
            {
                "token": surface,
                "logprob": _json_number(float(logprobs[index])),
                "bytes": list(surface.encode("utf-8")),
            }
        )
    return ranked


def _json_number(value: float) -> float:
    """``value`` as JSON can carry it, which a log probability is not guaranteed to be.

    A candidate the model gave no mass to has a log probability of negative infinity, and that is a
    true answer rather than a missing one, so it is reported at the negative edge of the double
    range instead of being dropped from the array -- dropping it would silently renumber every
    position after it. ``NaN`` is reported the same way: it is not a probability the model could
    have produced, and the only way to reach it is a forward pass that went numerically wrong, which
    this number cannot report any better than the edge can.

    This is a divergence from the native front end on purpose. Its JSON writer streamed the value
    straight out, so a position the model gave no mass to was written as ``-inf`` -- three
    characters that are not a JSON number, in a body every client is promised it can parse.
    """
    if math.isfinite(value):
        return value
    return -1.7976931348623157e308 if math.copysign(1.0, value) < 0 or math.isnan(value) else 1.7976931348623157e308
