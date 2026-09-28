"""One request, several choices: the fan-out both front ends share.

``n`` is a dispatch rule rather than a model rule -- an engine schedules one generation, so ``n``
choices are ``n`` generations -- and this module is where that rule is decided once. The seeds are
checked against a vector taken from the C++ front end's own ``derived_choice_seed``, compiled and
run for these values, because a seed that drifts from the one the front end pinned would change the
text of every choice in a request that named one.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pocketllm.api import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    RequestCancelledError,
    SamplingParams,
    TokenEvent,
    Usage,
)
from pocketllm.backends.base import BackendBase
from pocketllm.choices import (
    CHOICE_MARK,
    SEED_MASK,
    choice_seed,
    expanded,
    folded_usage,
    streamed,
)


def _request(**params) -> GenerationRequest:
    return GenerationRequest(
        prompt="a prompt", request_id="req-1", sampling_params=SamplingParams(**params)
    )


def _result(index: int, *, prompt: int = 7, completion: int = 3, cached: int = 0):
    return GenerationResult(
        request_id=f"req-1{CHOICE_MARK}{index}",
        token_ids=[1, 2, 3],
        text=f"answer {index}",
        usage=Usage(prompt, completion, cached),
    )


#: ``derived_choice_seed(base, choice)``, taken from the C++ implementation compiled for the
#: occasion. The C++ front end is what pinned these numbers, so a change here is a change to the
#: text every ``n``-choice request produces, and this test is the only place that would notice.
_SPLITMIX64 = {
    0: (16294208416658607535, 7960286522194355700, 487617019471545679, 17909611376780542444),
    5: (7134611160154358618, 13877614986023876344, 4292726422858613063, 1832488697174800709),
    42: (13679457532755275413, 2949826092126892291, 5139283748462763858, 6349198060258255764),
    SEED_MASK: (
        16490336266968443936,
        16834447057089888969,
        4048727598324417001,
        7862637804313477842,
    ),
}


def test_the_seed_derivation_is_the_front_ends_own() -> None:
    """Every choice but the first runs at a seed derived from the caller's, bit for bit.

    The vector is the C++ ``derived_choice_seed`` for choices 1..3, and it includes the wraps: the
    widest 64-bit seed, and a product that carries out of the top of the register. Two
    implementations that agree on the small cases can still disagree on those, and the disagreement
    would show only as different text.
    """
    for base, derived in _SPLITMIX64.items():
        assert tuple(choice_seed(base, index, vary=True) for index in range(1, 4)) == derived[1:]


def test_choice_zero_keeps_the_seed_the_caller_named() -> None:
    """A pinned seed still describes the first choice.

    The fan-out must not be observable on choice 0 for the reason it must not be observable at all:
    a caller who pinned a seed and asked for one choice has to get the run that seed names.
    """
    for base in _SPLITMIX64:
        assert choice_seed(base, 0, vary=True) == base


def test_a_greedy_request_gives_every_choice_the_same_seed() -> None:
    """Under greedy decoding the seed is never read, so varying it would only be noise.

    What greedy means is ``n`` copies of one answer, and deriving seeds that no sampler looks at
    would suggest the choices differ when they cannot.
    """
    request = _request(n=3, temperature=0.0, seed=9)
    assert [choice.sampling_params.seed for choice in expanded(request)] == [9, 9, 9]


def test_a_stochastic_request_varies_the_seed() -> None:
    """Sampling is where a seed means something, and different seeds are what makes choices differ."""
    request = _request(n=3, temperature=0.7, seed=9)
    seeds = [choice.sampling_params.seed for choice in expanded(request)]
    assert seeds[0] == 9
    assert len(set(seeds)) == 3


def test_one_choice_is_handed_back_untouched() -> None:
    """The fan-out is not observable on a request that did not ask for it.

    Same object, not an equal copy: the ordinary path has to keep the id and the parameters the
    caller built, and a copy would be a place for them to drift.
    """
    for params in ({}, {"n": 1}, {"n": 1, "seed": 4, "temperature": 0.7}):
        request = _request(**params)
        assert expanded(request) == [request]
        assert expanded(request)[0] is request


def test_choices_are_numbered_and_each_asks_for_one() -> None:
    """A choice is an ordinary request: a marked id, one choice, and nothing else changed."""
    request = _request(n=3, max_tokens=8, stop=["END"])
    choices = expanded(request)
    assert [choice.request_id for choice in choices] == [f"req-1{CHOICE_MARK}{i}" for i in range(3)]
    for choice in choices:
        assert choice.sampling_params.n == 1
        assert choice.sampling_params.max_tokens == 8
        assert choice.sampling_params.stop == ("END",)
        assert choice.prompt == request.prompt
    # `SamplingParams` is shared by dataclass equality rather than by identity, so the copy has to
    # be a real one: a choice that mutated its parameters would mutate its siblings'.
    assert choices[0].sampling_params is not request.sampling_params


def test_a_marked_id_cannot_collide_with_an_unmarked_one() -> None:
    """``req-1#2`` and ``req-1`` are distinct, and so are two requests that differ by a digit.

    The mark is what ``BackendBase.cancel`` matches on, so an id that could be a prefix of an
    unrelated request's would let one cancellation reach another request's choices.
    """
    first = _request(n=1)
    second = replace(first, request_id="req-10")
    assert expanded(first)[0].request_id == "req-1"
    assert expanded(replace(second, sampling_params=SamplingParams(n=2)))[1].request_id == (
        f"req-10{CHOICE_MARK}1"
    )


class _Recorder(BackendBase):
    """A backend that records what it was asked for and echoes it back as events.

    Only the lifecycle half of :class:`BackendBase` is under test here, so the two methods a real
    adapter has to answer are stubs rather than a model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.streamed_ids: list[str] = []

    @property
    def capabilities(self):
        return BackendCapabilities(name="recorder")

    def generate(self, requests):  # pragma: no cover - this backend is streamed, not generated
        raise NotImplementedError

    def stream(self, request):
        self._begin_request(request.request_id)
        self.streamed_ids.append(request.request_id)
        try:
            yield TokenEvent(request.request_id, token_id=1, text="a")
            yield TokenEvent(request.request_id, token_id=2, text="b", finish_reason="stop")
        finally:
            self._clear_request(request.request_id)


def test_a_stream_names_every_choice_and_its_own_chunk_index() -> None:
    """What a client accumulating per index needs: which choice, and in the order of the indices.

    The id on the chunk is the caller's, not the choice's: ``req-1#2`` names an engine request the
    client never made, and the fan-out is the host's business rather than the response's.
    """
    backend = _Recorder()
    try:
        events = list(streamed(backend, _request(n=3)))
    finally:
        backend.close()

    assert backend.streamed_ids == [f"req-1{CHOICE_MARK}{i}" for i in range(3)]
    assert [event.choice_index for event in events] == [0, 0, 1, 1, 2, 2]
    assert {event.request_id for event in events} == {"req-1"}


def test_the_usage_of_a_group_counts_the_prompt_once() -> None:
    """OpenAI's rule, and the one a client billing per token depends on.

    The prompt was asked for once and prefilled ``n`` times; the completion really was generated
    ``n`` times. Counting the prompt per choice would bill the caller ``n`` times for the same
    conversation, and counting the completion once would understate what the engine did.
    """
    usage = folded_usage([_result(index, prompt=7, completion=3) for index in range(3)])
    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 9
    assert usage.cached_tokens == 0
    # `total_tokens` is derived rather than stored, so a folded usage still satisfies the identity a
    # client checks: prompt plus completion.
    assert usage.total_tokens == 16


def test_no_results_is_no_usage() -> None:
    """A group that produced nothing reports zeros rather than raising.

    A backend answering fewer choices than were asked for is reported by there being fewer choices,
    not by an error -- losing the choices that did succeed would be the worse answer.
    """
    assert folded_usage([]) == Usage()


def test_cancelling_a_request_reaches_every_choice_of_it() -> None:
    """The client knows one id, so that id has to be enough.

    ``DELETE`` names the request the client sent. If the choices were only cancelled when they were
    named individually, a client stopping a three-choice request would leave two generations
    running, which is the failure the endpoint exists to prevent.
    """
    backend = _Recorder()
    try:
        choices = expanded(_request(n=3))
        for choice in choices:
            backend._begin_request(choice.request_id)
        assert backend.cancel("req-1") is True
        for choice in choices:
            assert backend._is_cancelled(choice.request_id)
    finally:
        backend.close()


def test_cancelling_does_not_reach_a_request_that_merely_starts_the_same() -> None:
    """``req-1`` must not cancel ``req-10``; only the mark makes an id a choice of another.

    A prefix match without the mark would be a request id that cancels its neighbours, which is a
    bug a client could trigger by choosing ids, and one nothing in the response would show.
    """
    backend = _Recorder()
    try:
        backend._begin_request("req-10")
        assert backend.cancel("req-1") is False
        assert backend._is_cancelled("req-10") is False
    finally:
        backend.close()


def test_cancelling_an_id_nothing_is_running_is_not_success() -> None:
    """The endpoint's answer has to mean something: no match is a ``False``, not an empty success."""
    backend = _Recorder()
    try:
        assert backend.cancel("req-nothing") is False
    finally:
        backend.close()


def test_a_cancelled_choice_raises_the_public_error() -> None:
    """The mechanism the whole group-cancel rests on: a marked id is refused at the next boundary.

    A cancellation that only set a flag would be a no-op the client could not see; the flag is read
    back by the generation loop, and this is that read.
    """
    backend = _Recorder()
    try:
        backend._begin_request("req-1")
        backend.cancel("req-1")
        with pytest.raises(RequestCancelledError):
            backend._check_cancelled("req-1")
    finally:
        backend.close()
