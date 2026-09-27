"""The bridge that lets a Python runtime be the engine behind the C++ scheduler.

Two halves, tested apart. [`RuntimeRun`] is the handshake between a whole-generation loop and a
scheduler that wants one token per call, and it is plain Python, so it is tested against a fake
loop here rather than against a model. `RuntimeEngine` is tested against a fake native module,
because what it does with the binding -- which fields it writes back, what it refuses -- is the
part that can go wrong independently of whether an extension was built.

`tests/test_python_engine_binding.py` is the other end: a real `BatchScheduler` driving a real
`RuntimeEngine`.
"""

from __future__ import annotations

import threading
import time

import pytest

from pocketllm.backends.runtime_engine import RuntimeRun, RuntimeSpec, engine_class


class FakeLoop:
    """A runtime's generation entry point: a budget, a gate per step, a token per step.

    Mirrors the shape of `src/models/*/generate.py`: `on_step` is asked before every token, the
    first one included, and the loop ends when the budget is spent or `on_step` says stop.
    """

    def __init__(self, budget: int, *, first: int = 100, eos: int | None = None, fail: str = ""):
        self.budget = budget
        self.first = first
        self.eos = eos
        self.fail = fail
        self.steps = 0
        self.emitted: list[int] = []
        self.stopped = ""
        # What the bridge handed over, so a test can assert the prompt and the request's sampling
        # parameters arrived rather than were reconstructed somewhere.
        self.prompt_ids: list[int] = []
        self.request_id: int | None = None
        self.sampling = None

    def __call__(self, *, request_id=None, prompt_ids=(), sampling=None, on_token=None, on_step=None):
        self.request_id = request_id
        self.prompt_ids = list(prompt_ids)
        self.sampling = sampling
        if self.fail:
            raise RuntimeError(self.fail)
        for index in range(self.budget):
            if on_step is not None and on_step():
                self.stopped = "cancel"
                return
            self.steps += 1
            token = self.first + index
            self.emitted.append(token)
            if on_token is not None:
                on_token(token)
            if self.eos is not None and token == self.eos:
                self.stopped = "eos"
                return
        self.stopped = "length"


def test_a_run_hands_over_one_token_per_step():
    loop = FakeLoop(4, first=100)
    run = RuntimeRun(loop, name="fake", timeout=5.0)
    try:
        assert [run.take() for _ in range(4)] == [100, 101, 102, 103]
        # The loop's budget is spent, so the next step has nothing to give and says so rather than
        # blocking until the timeout.
        assert run.take() is None
    finally:
        run.close()
    assert loop.stopped == "length"


def test_a_run_advances_only_when_it_is_asked_to():
    """The loop is parked between steps, not racing ahead producing tokens nobody asked for.

    This is the property the whole design rests on: the scheduler decides when the next step
    happens, so admission, cancellation and (later) interleaving have somewhere to stand.
    """
    loop = FakeLoop(8)
    run = RuntimeRun(loop, name="fake", timeout=5.0)
    try:
        assert run.take() == 100
        time.sleep(0.2)
        assert loop.steps == 1
        assert run.take() == 101
        assert loop.steps == 2
    finally:
        run.close()


def test_a_cancelled_run_unwinds_at_its_next_step():
    loop = FakeLoop(64)
    run = RuntimeRun(loop, name="fake", timeout=5.0)
    try:
        assert run.take() == 100
        run.cancel()
        assert run.take() is None
        assert loop.stopped == "cancel"
    finally:
        run.close()
    assert not run._thread.is_alive()


def test_a_failing_runtime_is_reported_where_the_scheduler_is_waiting():
    """The failure is raised on the scheduler's thread, with the runtime's own message.

    A loop that dies must not look like a loop that is slow: the scheduler is blocked on a queue,
    and a silent thread exit would leave it there until the timeout.
    """
    run = RuntimeRun(FakeLoop(4, fail="no checkpoint"), name="fake", timeout=5.0)
    try:
        with pytest.raises(RuntimeError, match="no checkpoint"):
            run.take()
    finally:
        run.close()


def test_a_wedged_runtime_times_out_with_a_message_that_names_it():
    def never(*, on_token=None, on_step=None):
        threading.Event().wait()

    run = RuntimeRun(never, name="fake", timeout=0.2)
    try:
        with pytest.raises(RuntimeError, match="produced no token"):
            run.take()
    finally:
        run.close()


# ---------------------------------------------------------------------------------------------
# The engine half, against a fake native module.
# ---------------------------------------------------------------------------------------------


class FakeCapabilities:
    def __init__(self) -> None:
        self.max_slots = 1
        self.continuous_batching = False
        self.chunked_prefill = False
        self.paged_kv = False
        self.per_request_sampling = False
        self.per_request_top_k = False


class FakeForwardResult:
    def __init__(self) -> None:
        self.token = 0
        self.top_token = 0
        self.position = 0


class FakePrefillResult:
    def __init__(self) -> None:
        self.results = ()
        self.incomplete = ()
        self.total_tokens = 0
        self.seconds = 0.0


class FakeDecodeResult:
    def __init__(self) -> None:
        self.next_tokens = ()
        self.finished = ()
        self.hit_stop_token = ()
        self.seconds = 0.0


class FakeEngine:
    """Stands in for the bound `InferenceEngine` a Python engine inherits from."""


class FakeNative:
    InferenceEngine = FakeEngine
    Capabilities = FakeCapabilities
    QwenForwardResult = FakeForwardResult
    BatchPrefillResult = FakePrefillResult
    BatchDecodeResult = FakeDecodeResult


class FakeSampling:
    def __init__(self, *, max_new_tokens: int = 4, stop_token_ids=(), ignore_eos: bool = False):
        self.max_new_tokens = max_new_tokens
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 20
        self.seed = 0
        self.stop_token_ids = list(stop_token_ids)
        self.ignore_eos = ignore_eos


class FakeRequest:
    def __init__(self, request_id: int, prompt_tokens, sampling: FakeSampling):
        self.request_id = request_id
        self.prompt_tokens = list(prompt_tokens)
        self.seq_len = 0
        self.slot_id = -1
        self.sampling = sampling
        self.finished = False
        self.last_token = 0


def _engine(start, *, eos=(999,), **spec_kwargs):
    spec = RuntimeSpec(
        name="fake",
        start=start,
        eos_tokens=lambda: set(eos),
        max_context=4096,
        step_timeout=5.0,
        **spec_kwargs,
    )
    return engine_class(FakeNative)(spec)


def test_the_declaration_is_the_specs_and_not_a_guess():
    engine = _engine(FakeLoop(4), max_slots=1)
    caps = engine.caps()
    assert caps.max_slots == 1
    assert caps.continuous_batching is False
    # Nothing here pages: the runtimes hold their cache themselves and report no block pool, so a
    # scheduler admission decision must not be made against block counts that do not exist.
    assert caps.paged_kv is False
    assert engine.max_context() == 4096
    assert engine.device() == -1


def test_a_width_larger_than_the_runtime_has_is_refused():
    engine = _engine(FakeLoop(4), max_slots=1)
    engine.allocate_batch_slots(1)
    with pytest.raises(ValueError, match="declares 1 slot"):
        engine.allocate_batch_slots(2)


def test_prefill_reports_the_token_and_how_much_prompt_it_consumed():
    engine = _engine(FakeLoop(4, first=100))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2, 3], FakeSampling(max_new_tokens=4))
    assert engine.allocate_slot(7) == 0

    out = engine.batch_prefill([request], 0)

    assert [row.top_token for row in out.results] == [100]
    assert list(out.incomplete) == [False]
    assert out.total_tokens == 3
    # `seq_len` is the write-back the scheduler reads to know the prompt is done: a runtime that
    # does not set it looks like one that has not started.
    assert request.seq_len == 3
    assert request.last_token == 100
    assert request.finished is False
    engine.free_slot(7)


def test_decode_steps_come_back_one_row_at_a_time():
    engine = _engine(FakeLoop(4, first=100))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2, 3], FakeSampling(max_new_tokens=4))
    engine.allocate_slot(7)
    engine.batch_prefill([request], 0)

    for expected in (101, 102, 103):
        out = engine.batch_decode_step([request])
        assert list(out.next_tokens) == [expected]
        assert list(out.finished) == [False]
        assert list(out.hit_stop_token) == [False]
    engine.free_slot(7)


def test_a_stop_token_from_the_prompt_ends_the_request_before_it_decodes():
    engine = _engine(FakeLoop(8, first=999), eos=(999,))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4))
    engine.allocate_slot(7)

    engine.batch_prefill([request], 0)

    # Flagged on the request, so the scheduler retires it without asking for a decode step.
    assert request.finished is True
    engine.free_slot(7)


def test_a_requests_own_stop_ids_are_recognised_here():
    """The runtime's loop only knows its checkpoint's EOS; a request's stop ids are the engine's.

    They are the scheduler's vocabulary, so a runtime that cannot see them would run past the token
    the caller asked it to stop on.
    """
    engine = _engine(FakeLoop(8, first=100))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4, stop_token_ids=[101]))
    engine.allocate_slot(7)
    engine.batch_prefill([request], 0)

    out = engine.batch_decode_step([request])

    assert list(out.next_tokens) == [101]
    assert list(out.hit_stop_token) == [True]
    engine.free_slot(7)


def test_ignore_eos_keeps_a_stop_token_from_ending_the_request():
    engine = _engine(FakeLoop(8, first=999), eos=(999,))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4, ignore_eos=True))
    engine.allocate_slot(7)

    engine.batch_prefill([request], 0)

    assert request.finished is False
    engine.free_slot(7)


def test_a_partially_prefilled_prompt_is_refused_rather_than_replayed():
    engine = _engine(FakeLoop(4))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2, 3], FakeSampling())
    request.seq_len = 2

    with pytest.raises(ValueError, match="chunked_prefill"):
        engine.batch_prefill([request], 0)


def test_a_failed_prefill_does_not_leave_its_run_parked():
    """The run thread is closed on the way out, so a failed request does not leak a wedged loop."""
    engine = _engine(FakeLoop(4, fail="no checkpoint"))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2], FakeSampling())
    engine.allocate_slot(7)

    with pytest.raises(RuntimeError, match="no checkpoint"):
        engine.batch_prefill([request], 0)

    assert engine._runs == {}
    engine.free_slot(7)


def test_freeing_a_slot_cancels_the_run_it_was_holding():
    engine = _engine(FakeLoop(64))
    engine.allocate_batch_slots(1)
    request = FakeRequest(7, [1, 2], FakeSampling())
    engine.allocate_slot(7)
    engine.batch_prefill([request], 0)
    run = engine._runs[7]

    engine.free_slot(7)

    assert not run._thread.is_alive()
    # And the slot is handed back, which is what a later request needs.
    assert engine.allocate_slot(8) == 0
