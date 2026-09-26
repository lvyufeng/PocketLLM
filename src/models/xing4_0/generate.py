"""Xing4.0-29B-A4B's generation loop: a prompt in, tokens and their cost out.

The trunk's forward is one call that consumes whichever tokens it is handed
(``gguf_model.forward``), so a decode step is that call with one token and a
prefill is the same call with a chunk.  What this module adds is the loop around
it -- the sampler, the end-of-turn set, the chunk boundary and the clocks -- and
it is separate from `pocketllm`'s adapters for the reason the other models'
loops are: the numbers a request reports and the way a cancel reaches it are the
same questions whatever is serving them.

The cache is the caller's.  It is 46 KB a token across the forty layers, so a
262144-position run is 11.8 GiB and a service allocates it once and resets it per
request rather than paying the zeroing every time.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from src.models.xing4_0.prefix_cache import restore, snapshot

__all__ = ["Generation", "generate", "sample_token"]

DEFAULT_PREFILL_CHUNK = 2048
"""Tokens one prefill forward takes, when the caller names no width.

A chunk is the unit the attention's cost is paid in: every token in it attends
to every token before it, so a single 32K-token forward would be a 32K x 32K
score matrix in fp32 -- 4 GiB for one layer -- where eight 4K chunks are eight
smaller ones with the same answer.  It is also the width the rate was measured
at, which is why it is a named default rather than a number inside the loop.
"""


@dataclass(slots=True)
class Generation:
    """What a request produced, and what it cost, in the units a client is billed in.

    `prefill_seconds` is the prompt's own pass and `decode_seconds` is every step
    after it.  `ttft_seconds` is the wall from the loop's start to the first
    token, which the prefill is nearly all of: the last chunk ends holding the
    final row's logits, so the first token costs one distribution rather than one
    more forward.  `stopped` names why the loop ended rather than how.

    `cached_tokens` is a prefix store's number and is zero here, because this
    loop forward-passes whatever it is given.  It is kept as a field rather than
    left out so that a store's arrival changes one loop rather than one interface.
    """

    tokens: list[int] = field(default_factory=list)
    stopped: str = "length"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttft_seconds: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @property
    def step_seconds(self) -> float:
        """One decode step's share of the loop, which is what a per-token rate means."""
        return self.decode_seconds / max(1, len(self.tokens) - 1)


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """One token from one row of logits, greedily unless a temperature says otherwise.

    The row is moved to the host and sampled there.  It is 131072 numbers a step,
    which is not nothing, but the alternative is a device-side sampler whose
    tie-breaking is the kernel's and the reference's is `argmax`'s, and a served
    answer that disagrees with the same request run greedily is a difference
    nobody can attribute afterwards.

    The order is the usual one and the reference's: temperature, then top-k, then
    top-p, then a draw.  `temperature <= 1e-5` short-circuits to `argmax`, which
    is the path every acceptance test in this repository takes.
    """
    row = logits.reshape(-1).to(torch.float32).cpu()
    if temperature <= 1e-5:
        return int(row.argmax())
    if top_k is not None and 0 < int(top_k) < row.numel():
        cutoff = torch.topk(row, int(top_k)).values.min()
        row = torch.where(row < cutoff, torch.full_like(row, float("-inf")), row)
    probabilities = torch.softmax(row / temperature, dim=-1)
    if top_p is not None and float(top_p) < 1.0:
        ordered, order = torch.sort(probabilities, descending=True)
        kept = torch.cumsum(ordered, dim=-1) - ordered <= float(top_p)
        if not bool(kept.any()):
            kept[0] = True
        mask = torch.zeros_like(probabilities, dtype=torch.bool)
        mask[order[kept]] = True
        probabilities = torch.where(mask, probabilities, torch.zeros_like(probabilities))
        probabilities = probabilities / probabilities.sum()
    if generator is None:
        return int(probabilities.argmax())
    return int(torch.multinomial(probabilities, 1, generator=generator))


def generate(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    eos_token_id: Sequence[int] | int | None = None,
    chunk: int = DEFAULT_PREFILL_CHUNK,
    cache: Any = None,
    prefix_cache: Any = None,
    on_token: Callable[[int, torch.Tensor], None] | None = None,
    on_step: Callable[[], bool] | None = None,
) -> Generation:
    """A prompt and its answer, through the chunked prefill and the caller's cache.

    `on_step` is called before every decode step and stops the loop when it
    returns true, which is how a cancel reaches a loop that is otherwise a
    straight line.  It is not called inside the prompt's forward: a chunk of
    2048 tokens is one kernel sequence with no seam to stop at, so a cancel
    during a long prefill is observed at the boundary after it.

    `cache` is the caller's, because the caller is the one that knows how many
    positions the run can hold and that the memory is worth keeping between
    requests.  One is built here for a caller with nothing to share, sized to the
    prompt and the budget and no more.

    `prefix_cache` is the store, or `None` for a run that forward-passes its whole
    prompt.  It is consulted once, before anything is forwarded, and what it hands
    back is the state a cold prefill of the same tokens leaves -- see
    :mod:`src.models.xing4_0.prefix_cache`.
    """
    ids = [int(token) for token in prompt_ids]
    if not ids:
        raise ValueError("a request with no prompt tokens has nothing to prefill")
    budget = int(max_new_tokens)
    if budget < 1:
        raise ValueError(f"a budget of {budget} tokens is not a generation")
    if chunk <= 0:
        raise ValueError(f"a chunk of {chunk} tokens is not a chunk")
    if cache is None:
        cache = model.make_cache(len(ids) + budget + 8, batch=1)

    generator = None
    if temperature > 1e-5:
        # Seeded from the request rather than from the clock, so the same request
        # is the same answer twice.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else int(seed))

    started = time.perf_counter()
    cached_len, logits = _resume(model, cache, prefix_cache, ids, int(chunk))
    if prefix_cache is not None and len(ids) > cached_len:
        # Stored now rather than at the end of the request: this is the state a
        # *prompt* leaves, and by the time the answer is finished the cache holds
        # the answer's rows too.
        prefix_cache.store(ids, snapshot(cache, len(ids)))
    prefill = time.perf_counter() - started
    first = time.perf_counter()

    eos = _eos_set(eos_token_id)
    tokens: list[int] = []
    stopped = "length"
    for index in range(budget):
        if on_step is not None and on_step():
            stopped = "cancel"
            break
        token = sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        tokens.append(token)
        if on_token is not None:
            on_token(token, logits)
        if token in eos:
            stopped = "eos"
            break
        if index + 1 == budget:
            break
        # The token just drawn is the next step's input, at the position after the
        # prompt and the tokens already emitted.  A chunked prefill leaves the
        # cache holding exactly `len(ids)`, which is what makes this arithmetic
        # rather than a counter to keep.
        step = model.forward([token], cache=cache, start_pos=len(ids) + index)
        logits = step[-1]

    decode = time.perf_counter() - first
    return Generation(
        tokens=tokens,
        stopped=stopped,
        prefill_seconds=prefill,
        decode_seconds=decode,
        ttft_seconds=prefill,
        prompt_tokens=len(ids),
        cached_tokens=cached_len,
    )


def _resume(
    model: Any, cache: Any, prefix_cache: Any, ids: list[int], chunk: int
) -> tuple[int, torch.Tensor]:
    """Forward as little of the prompt as the store allows; returns `(reused, logits)`.

    A store that has the whole prompt is the degenerate case a chat loop does not
    reach -- the rendered prompt grows every turn -- and it is handled by the same
    path: the remainder is empty, nothing is forwarded, and the logits come from a
    one-token forward at the boundary, which is what the request would have needed
    anyway to produce its first token.
    """
    # Before anything else: a cache the caller shares between requests still holds
    # the previous one's rows, and `append` only raises `length`.  A prompt shorter
    # than its predecessor's would otherwise attend to text that is no longer in
    # the conversation -- and would do it plausibly, which is why this is here
    # rather than trusted to the caller.
    for layer in cache if isinstance(cache, list) else [cache]:
        reset = getattr(layer, "reset", None)
        if reset is not None:
            reset()
        else:
            layer.length = 0

    reused = 0
    if prefix_cache is not None:
        reused = int(prefix_cache.longest(ids))
        prefix_cache.note(reused)
        if reused:
            restore(cache, prefix_cache.materialise(ids, reused), reused)
    if reused >= len(ids):
        return reused, model.forward([ids[-1]], cache=cache, start_pos=len(ids) - 1)[-1]
    return reused, _prefill(model, cache, ids[reused:], chunk, offset=reused)


def _prefill(
    model: Any, cache: Any, ids: list[int], chunk: int, *, offset: int = 0
) -> torch.Tensor:
    """The prompt, in `chunk`-token forwards; returns the last row's logits.

    Chunking is not an optimization here, it is the difference between a 32K
    prompt fitting and not: the attention's score matrix is `tokens x tokens` in
    fp32, so one forward over the whole prompt is quadratic in the card's memory.
    Every chunk is `[start_pos, start_pos + width)`, which is what the cache
    wants and what the causal mask is expressed against.
    """
    logits = None
    for start in range(0, len(ids), chunk):
        piece = ids[start : start + chunk]
        # `offset` is what a resumed prompt's remainder starts at: positions are
        # absolute, so the first token the model sees here is not position zero.
        logits = model.forward(piece, cache=cache, start_pos=offset + start)
    assert logits is not None
    return logits[-1]


def _eos_set(eos_token_id: Sequence[int] | int | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(token) for token in eos_token_id}
