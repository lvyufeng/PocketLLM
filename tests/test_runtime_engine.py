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

from pocketllm.backends.runtime_engine import (
    RuntimeRun,
    RuntimeSpec,
    SchedulerHost,
    device_index,
    engine_class,
)


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
        #: The adapter's own request for the row, resolved by the bridge. `None` for a spec that did
        #: not ask for it.
        self.context: object = "not called"

    def __call__(
        self,
        *,
        request_id=None,
        prompt_ids=(),
        sampling=None,
        context=None,
        on_token=None,
        on_step=None,
    ):
        self.request_id = request_id
        self.prompt_ids = list(prompt_ids)
        self.sampling = sampling
        self.context = context
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


# -----------------------------------------------------------------------------------
# which card a runtime binds, which is the one mistake that costs a whole run
# -----------------------------------------------------------------------------------


def test_the_card_a_runtime_names_is_the_card_it_gets():
    """Every shape a launch hands this over in, because two of them were read wrong.

    `device_index` is what decides which GPU a run thread binds before it drives the model, and
    both of its live callers hand it something that used to be misread: the v41 route passes
    `"cuda:{rank}"` and the torch route passes an `int`. A `str` went through `getattr(x, "index")`
    and `"cuda:3".index` is `str.index`, a builtin method, so `int()` of it raised and the run died
    before its first forward; an `int` fell through to the suffix parse and answered **0**, so a
    rank that asked for card 3 bound card 0 -- silently, in the direction that does not raise.

    Tested without torch: what is under test is the reading, and a test that imports torch to check
    it would skip on exactly the hosts where the string route is the one that runs.
    """
    assert device_index(None) == -1
    assert device_index(-1) == -1
    assert device_index(0) == 0
    assert device_index(2) == 2
    assert device_index(3) == 3
    assert device_index("cpu") == -1
    assert device_index("cuda") == 0
    assert device_index("cuda:0") == 0
    assert device_index("cuda:3") == 3
    with pytest.raises(TypeError):
        device_index(True)


def test_a_torch_device_is_read_the_same_way():
    """`torch.device` carries the index as an attribute rather than in its text, which is why the
    attribute branch exists at all."""
    torch = pytest.importorskip("torch")
    assert device_index(torch.device("cpu")) == -1
    assert device_index(torch.device("cuda")) == 0
    assert device_index(torch.device("cuda", 3)) == 3


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


# -- the adapter's own request, which the scheduler has no field for ------------------------------


def test_a_spec_that_does_not_ask_for_the_request_is_called_with_none():
    """The default, because resolving it is a wait between two threads and is not free."""
    loop = FakeLoop(4, first=100)
    engine = _engine(loop)
    engine.publish_row(1, "the request")
    request = FakeRequest(1, [1, 2], FakeSampling(max_new_tokens=4))
    engine.batch_prefill([request], 0)
    engine.batch_decode_step([request])
    engine.free_slot(1)

    assert loop.context is None


def test_a_published_row_reaches_the_runtime_as_its_own_request():
    loop = FakeLoop(4, first=100)
    engine = _engine(loop, wants_request=True)
    engine.publish_row(7, "the request")
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4))

    engine.batch_prefill([request], 0)

    assert loop.context == "the request"
    engine.batch_decode_step([request])
    engine.free_slot(7)


def test_a_row_that_is_published_after_the_run_starts_still_reaches_it():
    """The race the wait exists for.

    A scheduler may admit a row the instant `submit_request` returns, which can be before the
    submitting thread gets the GIL back to publish what the row is for. A run thread that read the
    mapping without waiting would find nothing there, and a runtime that renders part of its answer
    from the request would render the default for that request and the requested value for the next
    one -- the kind of wrong that looks like a model.
    """
    loop = FakeLoop(4, first=100)
    engine = _engine(loop, wants_request=True)
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4))

    publisher = threading.Timer(0.05, engine.publish_row, args=(7, "late"))
    publisher.start()
    try:
        engine.batch_prefill([request], 0)
    finally:
        publisher.join()

    assert loop.context == "late"
    engine.batch_decode_step([request])
    engine.free_slot(7)


def test_a_row_that_is_never_published_is_none_rather_than_a_hang():
    """Bounded, because an adapter that does not publish is a mistake and not a slow path."""
    loop = FakeLoop(4, first=100)
    engine = _engine(loop, wants_request=True)
    request = FakeRequest(7, [1, 2], FakeSampling(max_new_tokens=4))

    started = time.monotonic()
    engine.batch_prefill([request], 0)

    assert loop.context is None
    assert time.monotonic() - started < 30.0
    engine.batch_decode_step([request])
    engine.free_slot(7)


def test_forgetting_a_row_takes_it_out_of_the_mapping():
    loop = FakeLoop(4, first=100)
    engine = _engine(loop)
    engine.publish_row(7, "the request")
    assert engine.row_request(7, timeout=0.01) == "the request"

    engine.forget_row(7)

    assert engine.row_request(7, timeout=0.01) is None


# ------------------------------------------------------------- the spec an adapter registers


class _StubHost(SchedulerHost):
    """An adapter's runtime half: the four facts a spec is made of, and nothing else.

    The real hosts answer all four from a checkpoint, a launch and a card. What is under test here
    is not those answers but the spec built out of them -- which is the same body for every runtime
    in the family, and the reason there is one to test.
    """

    def __init__(self) -> None:
        self.name = "stub"
        self._max_seq_len = 4096
        self.bound: list[int] = []

    def _start_runtime(self, *, request_id=None, **kwargs) -> None:
        """A runtime's generation entry point. Never run: the spec carries it and does not call it."""

    def _eos_tokens(self) -> set[int]:
        return {7}

    def _runtime_device(self) -> int:
        self.bound.append(3)
        return 3


def test_the_registered_spec_is_built_from_the_adapters_own_facts():
    """The spec is the runtime's facts, and `device` stays a callable until a run thread asks.

    Every one of these five fields was written out in three adapters before it was written here,
    and three copies of a literal like `wants_request=True` is three chances to leave it out. So
    this asserts the mapping rather than the values: name from the adapter's, `start` the adapter's
    own entry point, context and end-of-sequence from what it already holds.

    `device` is deliberately not read here -- the card is a property of the launch, and this spec is
    built before the weights are loaded, so a spec that resolved it now would bind the run thread to
    whatever card happened to be current at construction.
    """
    host = _StubHost()

    spec = host._runtime_spec()

    assert spec.name == "stub"
    assert spec.start == host._start_runtime
    assert spec.eos_tokens() == {7}
    assert spec.max_context == 4096
    assert spec.wants_request is True
    assert host.bound == []
    assert spec.device() == 3
    assert host.bound == [3]
