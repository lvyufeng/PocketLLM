"""A request's prompt and its answer: the loop a serving process runs, and nothing else.

The benches in `tests/` drive the model a call at a time because what they are measuring is the
call. A served request is not a call: it is a prompt through the chunked path, then a token a step
until the model stops, with a first token's timing and a per-token rate that a client is billed by
and a cancel that a client's disconnect has to reach. That loop is here rather than in an adapter
because it is arithmetic over the model's own API -- `prefill`, `step`, the cache -- and two
adapters that each wrote it would disagree about what `max_new_tokens` means.

What this is not: batching, a scheduler, a prefix cache, or a sampler anyone should ship. One
request at a time, one sequence, greedy unless a temperature is given, and a stop that is EOS, a
budget, a string, or a cancel. Those are the four a first deployment needs and the ones the HTTP
layer already has parsed by the time it gets here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class Generation:
    """What a request produced, and what it cost, in the units a client is billed in.

    `prefill_seconds` is the prompt's own pass -- chunks, the expert copy, the attention over
    itself -- and `decode_seconds` is every step after it. `ttft_seconds` is the wall from the
    prompt's first byte to the first token, which the prefill is nearly all of: the chunked path
    ends holding the last row's logits, so the first token costs one distribution and not one more
    forward. `stopped` names why the loop ended rather than how: `eos`, `length`, `stop`, `cancel`.
    """

    tokens: list[int] = field(default_factory=list)
    stopped: str = "length"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttft_seconds: float = 0.0

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

    The row is moved to the host and sampled there: it is 152,576 numbers twice a step, which at
    this runtime's rate is nothing next to the expert copy, and a host-side sampler is the one a
    process can be held to the same answer across four ranks by -- the ranks' logits are
    byte-identical, so a sampler seeded from the request is identical too, and no rank has to be
    told what the others drew.
    """
    row = logits.reshape(-1).to(torch.float32).cpu()
    if temperature <= 1e-5:
        return int(row.argmax())
    if top_k is not None and int(top_k) < row.numel():
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
    chunk: int = 2048,
    cache: Any = None,
    on_token: Callable[[int, torch.Tensor], None] | None = None,
    on_step: Callable[[], bool] | None = None,
) -> Generation:
    """A prompt and its answer, through the chunked prefill and the cache.

    `on_step` is called before every decode step and stops the loop when it returns true. It is how
    a cancel reaches a loop that is otherwise a straight line -- and it is *every* rank's, not rank
    zero's, because a rank that stopped on its own would leave its peers inside a collective that
    nobody else enters. The caller is the one that can make it a collective; see the adapter.

    The cache is the caller's, because it is the caller that knows how many positions this run can
    hold and that the memory is worth keeping between requests. One is built here for a caller with
    nothing to share, sized to the prompt and the budget and no more.
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
        cache = model.cache(len(ids) + budget + 8)
    else:
        cache.reset()

    generator = None
    if temperature > 1e-5:
        # Seeded from the request rather than from the clock, so four ranks sampling the same
        # distribution draw the same token without a message between them.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else int(seed))

    started = time.perf_counter()
    logits = model.prefill(ids, cache=cache, chunk=chunk)
    prefill = time.perf_counter() - started
    first = time.perf_counter()

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
        if eos_token_id is not None and token in _eos_set(eos_token_id):
            stopped = "eos"
            break
        if index + 1 == budget:
            break
        logits = model.step(token, start_pos=len(ids) + index, cache=cache)[-1]

    decode = time.perf_counter() - first
    return Generation(
        tokens=tokens,
        stopped=stopped,
        prefill_seconds=prefill,
        decode_seconds=decode,
        ttft_seconds=prefill,
    )


def _eos_set(eos_token_id: Sequence[int] | int) -> set[int]:
    if isinstance(eos_token_id, int):
        return {int(eos_token_id)}
    return {int(token) for token in eos_token_id}
