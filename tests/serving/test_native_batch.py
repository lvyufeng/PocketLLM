"""The batched decode and the KV-wipe guard, over the real C engine.

These are the Stage-2 gates -- the scheduler's batched decode must equal the serialized one, and a
ragged group past its KV reservation must not wipe a live slot -- and they live apart from
``test_native_backend`` for one hardware reason: **a board holds one engine at a time.**  Each test
here opens its *own* engine at a width or window the shared service engine cannot be, and the Ascend
runtime refuses a second ``aclInit`` while the first is open (``aclError 100002``).  pytest finalizes
one module's fixtures before starting the next, so keeping the pooled engine in ``test_native_backend``
and the self-managed engines here is what lets both modules run on the board in one session.

The checkpoint and device both resolve through ``conftest`` so the batched and solo runs -- and the
service module -- are always the *same file*, which is the only thing that makes a batched-vs-solo
token comparison a statement about the scheduler rather than about two checkpoints.
"""

from __future__ import annotations

import threading

import pytest

from pocketllm.api import EngineArgs
from pocketllm.protocol import build_chat_request

from .conftest import (
    SERVING_CHECKPOINT as CHECKPOINT,
    SERVING_DEVICE as _DEVICE,
    native_available,
    require_checkpoint,
)


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="libpocketllm.so is not built (cmake -B build -S src && cmake --build build)",
)


@pytest.fixture(scope="module", autouse=True)
def _batch_gates_need_a_checkpoint() -> None:
    """An absent checkpoint is a FAILURE here, not a skip -- see the conftest's rationale."""
    require_checkpoint()


def test_batched_streams_match_their_solo_output() -> None:
    """The acceptance gate for batch serving: a batched request equals a solo one.

    A wrong KV span returns *fluent text with no error*, so equality is the only proof.  Distinct
    prompts of different lengths make the post-prefill positions ragged, and the width is asserted
    against ``_batched_widths`` -- the width ``_run_group`` actually ran at, appended only inside
    it -- so the test fails if the scheduler serialized rather than batched.
    """
    from pocketllm.server.native_backend import NativeBackend

    prompts = [
        "Reply with only the word: alpha",
        "Name one color, one word only.",
        "Reply with exactly: the quick brown fox",
        "Say only: omega",
    ]

    def streamed(instance, text, rid):
        events = list(
            instance.stream(
                build_chat_request(
                    {"messages": [{"role": "user", "content": text}], "max_tokens": 12},
                    request_id=rid,
                )
            )
        )
        return "".join(e.text for e in events if e.text), events[-1].finish_reason

    for width in (2, 4):
        chosen = prompts[:width]

        solo = {}
        solo_backend = NativeBackend(EngineArgs(model=CHECKPOINT, max_batch_size=1), _DEVICE)
        try:
            for i, text in enumerate(chosen):
                solo[i] = streamed(solo_backend, text, f"solo-{width}-{i}")
        finally:
            solo_backend.close()

        batched = {}
        errors = {}
        backend = NativeBackend(
            EngineArgs(model=CHECKPOINT, max_batch_size=width, batch_kv_window=1024), _DEVICE
        )
        try:
            start = threading.Event()

            def run(i, text):
                start.wait()
                try:
                    batched[i] = streamed(backend, text, f"batch-{width}-{i}")
                except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
                    errors[i] = repr(exc)

            threads = [
                threading.Thread(target=run, args=(i, text)) for i, text in enumerate(chosen)
            ]
            for thread in threads:
                thread.start()
            start.set()
            for thread in threads:
                thread.join(timeout=300)
            groups = list(backend._batched_widths)
        finally:
            backend.close()

        assert not errors, errors
        assert max(groups, default=0) >= width, (
            f"the batched decode never reached width {width}: {groups}"
        )
        for i in range(width):
            assert batched[i] == solo[i], f"request {i} at width {width} drifted from its solo output"


#: The batch widths the KV-wipe guard is exercised at.  One is the control: at width 1 the
#: scheduler runs the serialized path, no group forms, and the same prompt a group stops at the
#: group bound runs on to its own ``max_tokens`` -- which is what makes the width-2 and width-4
#: stops attributable to the group's bound rather than to the model or the request.
_GUARD_WIDTHS = (1, 2, 4)

#: Ragged prompts for the guard, as (filler words, filler letter, instruction).  Four token
#: lengths, so a group's slots sit at four different positions and the longest sets the cap; a
#: width takes a prefix of the longest-first order, so widths 2 and 4 share one cap -- and one
#: bound.  The instructions have to keep the model counting past the bound: a prompt that stops on
#: its own ends its stream early and leaves the lockstep untested, which is what "count upward
#: from 1" does (it signs off after ~73 tokens on the 0.6B), so it is the *quitter* -- asked
#: `_QUIT_TOKENS` rather than the group's ask.
_RAGGED_PROMPTS = (
    (15, "w", "Now count upward from 1, one number per line:"),
    (55, "q", "Now count backward from 100, one number per line:"),
    (20, "p", "Now count upward from 1000, one number per line:"),
    (30, "z", "Now count backward from 1000, one number per line:"),
)

#: What a group's last (shortest) job asks for: a small budget parks it early, so the group runs on
#: with its slot padded and fed ``_FINISHED_TOKEN`` -- the case the group bound exists for -- on
#: every run, rather than whenever a prompt happens to stop on its own.
_QUIT_TOKENS = 16


@pytest.mark.parametrize("width", _GUARD_WIDTHS)
def test_a_ragged_group_past_the_reservation_does_not_wipe_a_slot(width: int) -> None:
    """The regression test for the KV-wipe: a ragged group whose longest slot over-runs.

    ``ensure_kv`` grows the slab by release + realloc *without copy* whenever a position passes the
    reservation, so if one slot walks past it while another is still live, the other's KV is erased
    and fluent-wrong tokens come back with rc=0 and no error.  A group decodes in lockstep and a
    finished member keeps its slot, fed ``_FINISHED_TOKEN``, so an unbounded slot climbs one
    position per step the *group* runs.  With a per-job bound -- ``window - len(prompt)`` -- the
    shorter prompt gets the larger budget and keeps decoding while the longest prompt's already-full
    slot keeps advancing, walking it past the reservation.

    The bound has to be the group's, not each job's: every job stops ``window - cap`` steps past its
    own prompt, where ``cap`` is the group's longest prompt, so the longest slot ends *at* the
    reservation (``cap + bound == window``) and no slot -- padded or live -- can pass it.  A window
    of 256 is the smallest power of two the reservation can sit at, which is what lets the longest
    slot actually reach it in a handful of steps; the prompts differ enough that a per-job bound
    would let the shorter ones run well past.

    The width axis is the point of the parametrization.  Widths 2 and 4 assert the guard's own
    arithmetic -- every job but the quitter emits exactly ``window - cap`` tokens, the quitter emits
    its own small ask, and every job's stream is a *prefix of its solo run*, finish reason included,
    because a wiped slot comes back fluent and wrong and a count alone would not see it.  Width 1 is
    the control: one of the group's own prompts, the same ask a width-2 group member submits,
    submitted alone.  The serialized path sizes against the model's context and applies no group
    window, so the control is not bounded by the group's bound -- and the assertion is exactly that:
    a *length* stop short of its own ask would be the group bound clamping it, which would make the
    width-2 and width-4 stops a property of the window rather than the scheduler.  The control is
    robust to an early EOS -- a *stop*, a near-tie a 1e-3 divergence can flip on another host's
    checkpoint -- because a clamp ends the run in a length stop, not an EOS, so the two are told
    apart by the finish reason and the check stays a loud FAIL without a dependence on the model
    reaching its ask.
    """
    from pocketllm.server.native_backend import NativeBackend

    window = 256
    #: Longest first, so a width-w prefix shares the cap -- and the bound -- with every other width.
    #: Width 1 takes index 1 rather than index 0: the control's ask (``bound + 64``) has to stay
    #: inside the window, and only a prompt shorter than the cap leaves room for it.
    ordered = sorted(_RAGGED_PROMPTS, key=lambda entry: entry[0], reverse=True)
    table = [
        " ".join(f"{letter}{j}" for j in range(count)) + " " + instruction
        for count, letter, instruction in ordered
    ]
    picks = [1] if width == 1 else list(range(width))
    prompts = [table[i] for i in picks]

    def one(instance, text, rid, max_tokens):
        return instance.generate(
            [
                build_chat_request(
                    {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens},
                    request_id=rid,
                )
            ]
        )[0]

    #: One serialized backend for the whole leg: it tokenizes the *table* -- the cap is the table's,
    #: whatever this width submits -- and produces the solo run each job's stream is a prefix of.
    #: Both are width-1 work, so they share a session, which is one model load instead of two.
    reference = {}
    solo_backend = NativeBackend(EngineArgs(model=CHECKPOINT, max_batch_size=1), _DEVICE)
    try:
        lengths_all = [
            one(solo_backend, text, f"ragged-len-{width}-{i}", 1).usage.prompt_tokens
            for i, text in enumerate(table)
        ]
        cap = max(lengths_all)
        assert cap < window, f"the longest prompt ({cap}) does not fit under window {window}"
        bound = window - cap
        assert bound >= 4, f"the prompts leave only {bound} steps under window {window}"
        #: The guard is load-bearing only while a *per-job* bound would let a shorter job run past
        #: the group's -- assert the raggedness, so a prompt edit cannot quietly make this vacuous.
        assert window - min(lengths_all) > bound, (
            f"the shortest prompt ({min(lengths_all)}) leaves no slack under window {window}: a "
            "per-job bound would stop it in the same place, and the test would pass without the "
            "guard"
        )

        #: Every *group* job asks for more than the group bound, so a stream that stops at ``bound``
        #: was stopped by the scheduler rather than by its own request.  At wider widths the group's
        #: last, shortest job asks ``_QUIT_TOKENS`` instead: a small budget parks it early so the
        #: padded-slot path runs on every trial.  At width 1 the lone request is the *control*, not a
        #: group member: it submits the ask that a width-2 group member does (``bound + 64``) and has
        #: to run past the group bound, because the serialized path applies no batch window.
        asked = bound + 64
        quit_index = len(prompts) - 1 if width > 1 else None
        asks = [asked if i != quit_index else _QUIT_TOKENS for i in range(len(prompts))]
        for i, text in enumerate(prompts):
            result = one(solo_backend, text, f"ragged-solo-{width}-{i}", asks[i])
            reference[i] = (list(result.token_ids), result.finish_reason)
    finally:
        solo_backend.close()

    lengths = [lengths_all[i] for i in picks]
    #: What each group job's stream is expected to hold: the group bound for the lockstep jobs, the
    #: quitter's own ask for the quitter.  (Unused at width 1, where the lone request is the control
    #: and its expectation is its own ask.)
    expects = [asks[i] if i == quit_index else bound for i in range(len(prompts))]
    #: Setup, asserted before the group runs so a bad table fails as a bad table.  The quitter has to
    #: end at its own ask; every non-quitter has to reach *at least* the group bound, so the group's
    #: stream is long enough to compare and the lockstep this test exists for is really exercised.
    #: It may end there or past it (early EOS), which is why this is a lower bound and not equality.
    for i in range(len(prompts)):
        if i == quit_index:
            assert len(reference[i][0]) == _QUIT_TOKENS, (
                f"the quitter's solo run emitted {len(reference[i][0])} tokens, not its ask "
                f"{_QUIT_TOKENS}"
            )
        else:
            assert len(reference[i][0]) >= bound, (
                f"the solo run of request {i} emitted {len(reference[i][0])} tokens, before the "
                f"group bound {bound}: the prompt table needs an instruction this model does not "
                "stop on before the lockstep is exercised"
            )

    results = {}
    errors = {}
    backend = NativeBackend(
        EngineArgs(model=CHECKPOINT, max_batch_size=width, batch_kv_window=window), _DEVICE
    )
    try:
        start = threading.Event()

        def go(i, text):
            start.wait()
            try:
                result = one(backend, text, f"ragged-{width}-{i}", asks[i])
                results[i] = (list(result.token_ids), result.finish_reason)
            except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
                errors[i] = repr(exc)

        threads = [threading.Thread(target=go, args=(i, text)) for i, text in enumerate(prompts)]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(timeout=600)
        batched_widths = list(backend._batched_widths)
    finally:
        backend.close()

    assert not errors, errors
    if width == 1:
        assert batched_widths == [], (
            f"a width-1 backend ran a group: {batched_widths}; width 1 is the control, and the "
            "serialized path is what it is a control for"
        )
    else:
        assert max(batched_widths, default=0) >= width, (
            f"the batched decode never reached width {width}: {batched_widths}"
        )
    for i in range(len(prompts)):
        emitted, finish_reason = results[i]
        solo = reference[i][0]
        if width == 1:
            #: The control submitted a width-2 group member's ask (``bound + 64``).  The serialized
            #: path applies no group window, so the stream can only end two ways: at its own ask (a
            #: *length* stop) or early at EOS (a *stop*, a property of the model on this host -- the
            #: same near-tie a 1e-3 divergence can flip, and not a guard failure).  A *length* stop
            #: short of the ask is the clamp: the group bound ``bound`` cutting a run that its own
            #: request never capped.  That is the assertion that matters, and it holds whether or not
            #: the model happens to stop early -- the EOS case ends in a *stop*, which clamping does
            #: not produce.  When the run does reach its ask, it is ``bound + 64`` tokens, past the
            #: group window by construction.
            assert finish_reason in ("length", "stop"), (
                f"the width-1 control ended with {finish_reason!r}, neither 'length' nor 'stop'"
            )
            if finish_reason == "length":
                assert len(emitted) == asks[i], (
                    f"the width-1 control stopped for length at {len(emitted)} tokens, not its own "
                    f"ask {asks[i]}: the serialized path applies no group window, so a length stop "
                    f"short of the ask is the group bound {bound} clamping it -- and then the "
                    "width-2 and width-4 stops would not be the group's bound either"
                )
            expected_pos = lengths[i] + len(emitted)
        else:
            assert len(emitted) == expects[i], (
                f"request {i} at width {width} emitted {len(emitted)} tokens, not {expects[i]} "
                f"(window {window} - longest prompt {cap} for the lockstep, the quitter's own ask "
                f"for request {quit_index}); a per-job bound lets the shorter prompts run past the "
                "reservation, which grows the slab under -- and wipes -- the other slots"
            )
            #: Both the lockstep jobs and the quitter decode in lockstep as a group, so a slot's
            #: position advances by the *group's* step count -- ``bound`` -- regardless of how many
            #: tokens that job emitted itself.  A padded finished slot (the quitter's) keeps
            #: advancing on the group's window; this is the position a per-job bound would walk past
            #: the reservation, and the assertion is that it does not.
            expected_pos = lengths[i] + bound
        assert expected_pos <= window, (
            f"request {i} at width {width} reached position {expected_pos}, past the reservation at "
            f"{window}: a slot was not bounded to the group's window"
        )
        if width == 1:
            continue
        #: A wiped slot returns fluent wrong tokens with no error, so the check is the tokens
        #: themselves.  In a group the job is clipped to the group bound, so it equals the solo
        #: run's *prefix* of that length -- a divergence anywhere inside it is the wipe.
        if emitted != solo[: len(emitted)]:
            where = next(
                (k for k, (a, b) in enumerate(zip(emitted, solo)) if a != b),
                min(len(emitted), len(solo)),
            )
            raise AssertionError(
                f"request {i} at width {width} drifted from its solo run at token {where}: "
                f"group {emitted[: where + 1]} vs solo {solo[: where + 1]} -- a wiped slot returns "
                "fluent wrong tokens with no error, so equality is the only proof"
            )
        assert finish_reason == reference[i][1], (
            f"request {i} at width {width} ended with {finish_reason!r}, not the solo run's "
            f"{reference[i][1]!r}"
        )
