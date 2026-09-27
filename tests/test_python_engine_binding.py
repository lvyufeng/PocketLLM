"""A Python object driven by the real C++ `BatchScheduler`.

The scheduler is one library with two kinds of host: the native binary, and whatever a Python
process hands it. This file is about the second kind, and it is the only place the trampoline is
exercised against the scheduler itself rather than against a fake of it -- a fake native module
cannot show that the GIL handoff and the by-value struct conversion work, because it is the
handoff and the conversion that are being tested.

The engine here does no model work. It answers every prompt with one fixed token and every decode
step with another, which is enough for the scheduler's own contract: it has to be handed one result
row and one completion flag per request, a token per decode step, and a `seq_len` that says how far
the prompt got.
"""

from __future__ import annotations

import importlib
import threading

import pytest


@pytest.fixture(scope="module")
def native_module():
    try:
        return importlib.import_module("pocketllm_cpp")
    except ImportError as exc:
        pytest.skip(f"native pocketllm_cpp module is not built: {exc}")


def _toy_engine(native, *, prompt_token: int = 7, answer: int = 42):
    """A width-1 engine of the shape a real Python runtime has.

    Declaring `continuous_batching=False` and `max_slots=1` is not a formality: the scheduler reads
    the declaration and runs one request at a time whatever width it was constructed with, so an
    engine that understates what it can do is driven correctly rather than concurrently.
    """

    class ToyEngine(native.InferenceEngine):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple] = []
            self.slots: dict[int, int] = {}

        def caps(self):
            caps = native.Capabilities()
            caps.max_slots = 1
            caps.continuous_batching = False
            caps.chunked_prefill = False
            caps.paged_kv = False
            caps.per_request_sampling = False
            caps.per_request_top_k = False
            return caps

        def max_context(self):
            return 4096

        def device(self):
            return -1

        def allocate_batch_slots(self, max_batch_size):
            self.calls.append(("allocate_batch_slots", max_batch_size))

        def allocate_slot(self, request_id):
            self.slots[request_id] = 0
            return 0

        def free_slot(self, request_id):
            self.calls.append(("free_slot", request_id))
            self.slots.pop(request_id, None)

        def kv_paged(self):
            return False

        def kv_free_blocks(self):
            return 0

        def kv_total_blocks(self):
            return 0

        def kv_blocks_for_tokens(self, tokens):
            return 0

        def batch_prefill(self, requests, token_budget):
            self.calls.append(("batch_prefill", len(requests), token_budget))
            out = native.BatchPrefillResult()
            rows = []
            flags = []
            for request in requests:
                # `seq_len` is the write-back: how much of the prompt this call consumed. A
                # runtime that chunks advances it by its budget; this one runs the prompt whole.
                request.seq_len = len(request.prompt_tokens)
                row = native.QwenForwardResult()
                row.token = prompt_token
                row.top_token = prompt_token
                rows.append(row)
                flags.append(False)
                out.total_tokens += len(request.prompt_tokens)
            # Assigned, not appended to: these read back as tuples, because a list here would be a
            # copy that looks mutable and silently discards the append.
            out.results = rows
            out.incomplete = flags
            return out

        def batch_decode_step(self, requests):
            self.calls.append(("batch_decode_step", len(requests)))
            out = native.BatchDecodeResult()
            out.next_tokens = [answer for _ in requests]
            out.finished = [False for _ in requests]
            out.hit_stop_token = [False for _ in requests]
            return out

    return ToyEngine()


def _run_one(native, engine, width: int, *, max_new_tokens: int = 4, timeout: float = 30.0):
    """Submit one request to a real scheduler and wait for its completion callback."""
    scheduler = native.QwenBatchScheduler(engine, width)
    done = threading.Event()
    box: dict[str, object] = {}

    def on_completion(result):
        box["result"] = result
        done.set()

    sampling = native.QwenBatchSamplingParams()
    sampling.max_new_tokens = max_new_tokens
    request_id = scheduler.submit_request([1, 2, 3], sampling, on_completion)
    try:
        assert done.wait(timeout), "the completion callback never fired"
    finally:
        scheduler.stop()
    assert request_id > 0
    return box["result"]


def test_a_python_engine_drives_the_real_scheduler(native_module):
    engine = _toy_engine(native_module)
    result = _run_one(native_module, engine, 1)

    assert result.error == ""
    # The prompt's own token first, then one per decode step -- the scheduler's model of a
    # generation, with the engine supplying every one of them.
    assert list(result.generated_tokens) == [7, 42, 42, 42]
    assert result.finish_reason == "length"

    calls = [call[0] for call in engine.calls]
    assert calls == [
        "allocate_batch_slots",
        "batch_prefill",
        "batch_decode_step",
        "batch_decode_step",
        "batch_decode_step",
        "free_slot",
    ]
    # Every call is one row: the engine declared one slot, so the scheduler never asked for more.
    assert all(call[1] == 1 for call in engine.calls if len(call) > 1)


def test_the_engine_declaration_decides_the_width(native_module):
    """An engine that can run one request at a time is asked for one request at a time.

    The width is the caller's request and the declaration is the engine's answer, so the scheduler
    takes the smaller of the two and the engine is sized for what it will actually be handed.
    """
    engine = _toy_engine(native_module)
    result = _run_one(native_module, engine, 8)

    assert list(result.generated_tokens) == [7, 42, 42, 42]
    assert ("allocate_batch_slots", 1) in engine.calls


def test_a_python_engine_reports_its_declaration_through_the_scheduler(native_module):
    scheduler = native_module.QwenBatchScheduler(_toy_engine(native_module), 1)
    try:
        caps = scheduler.engine_caps()
        assert caps.max_slots == 1
        assert caps.continuous_batching is False
    finally:
        scheduler.stop()


def test_result_rows_and_completion_flags_are_assigned_not_appended_to(native_module):
    """The read is a copy, and it says so.

    pybind11's stl caster builds a fresh list from a `std::vector` member on every read, so an
    in-place append would go to a temporary and leave the C++ vector empty -- which surfaces much
    later as the scheduler complaining that `batch_prefill` returned no rows. A tuple cannot be
    appended to, so the mistake is an `AttributeError` on the line that made it.
    """
    out = native_module.BatchPrefillResult()
    assert out.results == ()
    assert out.incomplete == ()

    with pytest.raises(AttributeError):
        out.results.append(native_module.QwenForwardResult())

    row = native_module.QwenForwardResult()
    row.top_token = 11
    out.results = [row]
    out.incomplete = [False]
    assert [r.top_token for r in out.results] == [11]
    assert list(out.incomplete) == [False]

    decode = native_module.BatchDecodeResult()
    decode.next_tokens = [1, 2]
    assert tuple(decode.next_tokens) == (1, 2)
    # Left empty where an engine has no speculative row metadata, which is what the scheduler
    # reads as "a plain one-token-per-row engine".
    assert decode.emitted_tokens == ()
    assert decode.position_advances == ()


def test_a_nested_member_writes_through(native_module):
    """`request.last_result.top_token = x` has to reach the C++ object.

    The same copy-on-read applies one level down: a `def_readwrite` member of a registered type
    hands back a temporary, so the nested write is discarded and the engine's row silently carries
    nothing. These two are bound by reference instead.
    """
    request = native_module.BatchedRequest()
    request.last_result.top_token = 5
    assert request.last_result.top_token == 5

    request.sampling.max_new_tokens = 9
    assert request.sampling.max_new_tokens == 9


def test_the_bridge_carries_a_runtime_through_the_real_scheduler(native_module):
    """The whole path: a runtime's loop, the bridge, and the C++ scheduler in between.

    The two files are tested apart for speed, and this is where they meet -- `RuntimeSpec` wrapping
    an ordinary Python function, `make_runtime_engine` turning it into an engine, and a real
    `BatchScheduler` driving it. What it establishes that neither half can alone is that the tokens
    a runtime emits arrive in the answer, in order and exactly once.
    """
    from pocketllm.backends.runtime_engine import RuntimeSpec, make_runtime_engine

    emitted: list[int] = []

    def generate(*, request_id, prompt_ids, sampling, context, on_token, on_step):
        """A runtime of the shape the real ones have: a gate and a token per step."""
        emitted.append((request_id, list(prompt_ids)))
        for token in range(100, 100 + int(sampling.max_new_tokens)):
            if on_step():
                return
            on_token(token)

    spec = RuntimeSpec(
        name="fake",
        start=generate,
        eos_tokens=lambda: {999},
        max_context=4096,
        step_timeout=30.0,
    )
    engine = make_runtime_engine(spec, native_module)
    scheduler = native_module.QwenBatchScheduler(engine, 4)

    done = threading.Event()
    box = {}

    def on_completion(result):
        box["result"] = result
        done.set()

    sampling = native_module.QwenBatchSamplingParams()
    sampling.max_new_tokens = 4
    try:
        scheduler.submit_request([10, 20, 30], sampling, on_completion)
        assert done.wait(30), "the completion callback never fired"
    finally:
        scheduler.stop()

    result = box["result"]
    assert result.error == ""
    assert list(result.generated_tokens) == [100, 101, 102, 103]
    assert result.finish_reason == "length"
    assert result.prompt_tokens == 3
    # The prompt reached the runtime once, whole, under the id the scheduler will cancel it by:
    # `seq_len` is how the bridge reports that the prompt is consumed, and a runtime that never
    # sees its prompt produces an answer to something else.
    assert [tokens for _, tokens in emitted] == [[10, 20, 30]]
    assert emitted[0][0] == result.request_id



def test_the_engine_is_asked_for_its_device_before_the_loop_thread_starts(native_module):
    """A scheduler has to be stoppable without the GIL, and asking the engine is what stops it.

    `BatchScheduler::schedule_loop` runs on its own thread and asks the engine for its device before
    anything else. Against a Python engine that question crosses the language boundary and needs the
    GIL -- and the thread that holds the GIL is very often the one that is about to drop the
    scheduler, because `~BatchScheduler` runs `stop()` and `stop()` joins. A scheduler created and
    dropped in the same breath therefore deadlocked the process: not a slow shutdown, a wedged one,
    which is how this was found.

    So the constructor asks, on the constructing thread, and hands the loop thread the answer. That
    is asserted by thread identity rather than by stopwatch: a test that simply dropped a scheduler
    would hang rather than fail, and a hang is not a failing test.
    """
    asked: list[int] = []

    class Engine(native_module.InferenceEngine):
        def caps(self):
            caps = native_module.Capabilities()
            caps.max_slots = 1
            caps.continuous_batching = False
            return caps

        def max_context(self) -> int:
            return 1024

        def device(self) -> int:
            asked.append(threading.get_ident())
            return -1

        def allocate_batch_slots(self, max_batch_size: int) -> None:
            return None

        def allocate_slot(self, request_id: int) -> int:
            return 0

        def free_slot(self, request_id: int) -> None:
            return None

    here = threading.get_ident()
    scheduler = native_module.QwenBatchScheduler(Engine(), 1)
    try:
        assert asked == [here]
    finally:
        scheduler.stop()


def test_a_scheduler_that_is_dropped_rather_than_stopped_lets_the_process_go(native_module):
    """The shape the deadlock took, run for real.

    The assertion is the one above -- this is here because the failure mode is a hang and a hang in
    a test suite is indistinguishable from a slow machine. It is bounded by the interpreter's own
    teardown: the `del` runs `~QwenBatchScheduler`, which joins the loop thread, so a regression
    here wedges the suite rather than failing it. That is worse than a failure and better than
    silence.
    """
    scheduler = native_module.QwenBatchScheduler(_toy_engine(native_module), 1)

    del scheduler
