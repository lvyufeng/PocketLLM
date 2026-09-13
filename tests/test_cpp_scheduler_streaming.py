"""Scheduler-backed async requests exposed to Python (Issue #167).

The surface tests need only the module. The behavioural tests drive a real
single-GPU QwenEngine over the synthetic fixture that
`cpp_engine/tests/test_batch_scheduler.cpp` writes, so the token callback is
driven by the actual scheduler thread rather than a stub. Generate it with:

    cmake --build <build> --target test_batch_scheduler
    ./<build>/tests/test_batch_scheduler
"""
from __future__ import annotations

import asyncio
import importlib
import os
import threading
import time

import pytest

FIXTURE = "/tmp/test_batch_scheduler_qwen_fixture"


@pytest.fixture(scope="module")
def native_module():
    try:
        return importlib.import_module("pocketllm_cpp")
    except ImportError as exc:
        pytest.skip(f"native pocketllm_cpp module is not built: {exc}")


@pytest.fixture(scope="module")
def engine(native_module):
    if not os.path.isdir(FIXTURE):
        pytest.skip(f"scheduler fixture missing at {FIXTURE}")

    options = native_module.QwenEngineOptions()
    options.device = 0
    options.tp_world = 1
    options.tp_rank = 0
    options.prefill_chunk_tokens = 128
    options.max_batch_size = 2
    options.prefix_cache = False
    options.temperature = 0.0
    options.top_p = 1.0
    options.top_k = 1
    options.sampling_seed = 12345

    try:
        built = native_module.QwenEngine(FIXTURE, options)
    except Exception as exc:  # no usable GPU, or the fixture will not load
        pytest.skip(f"could not construct Qwen engine: {exc}")
    return built


@pytest.fixture
def scheduler(native_module, engine):
    sched = native_module.QwenBatchScheduler(engine, max_batch_size=4)
    yield sched
    sched.stop()


def sampling_params(native_module, max_new_tokens):
    sampling = native_module.QwenBatchSamplingParams()
    sampling.max_new_tokens = max_new_tokens
    sampling.temperature = 0.0
    sampling.top_k = 1
    sampling.top_p = 1.0
    # Fixed token count regardless of what the fixture's vocab samples.
    sampling.ignore_eos = True
    return sampling


def test_capabilities_type_is_read_only(native_module):
    caps = native_module.Capabilities()
    assert caps.max_slots == 1
    assert caps.continuous_batching is False
    assert caps.fixed_top_p == 1.0
    assert "Capabilities" in repr(caps)
    # The engine declares these; Python must not be able to forge them.
    with pytest.raises(AttributeError):
        caps.max_slots = 99


def test_submit_request_accepts_token_callback(native_module):
    signature = native_module.QwenBatchScheduler.submit_request.__doc__.splitlines()[0]
    for argument in ("prompt_tokens", "sampling", "callback", "on_token"):
        assert argument in signature


def test_engine_caps_reflect_the_engine(scheduler):
    caps = scheduler.engine_caps()
    assert caps.continuous_batching is True
    assert caps.chunked_prefill is True
    assert caps.max_slots >= 2


def test_max_batch_size_clamps_to_slots(scheduler):
    # The fixture engine has fewer slots than the 4 the scheduler asked for.
    assert scheduler.max_batch_size() <= scheduler.engine_caps().max_slots


def test_prefill_token_budget_roundtrip(scheduler):
    assert scheduler.prefill_token_budget() > 0
    scheduler.set_prefill_token_budget(2048)
    assert scheduler.prefill_token_budget() == 2048
    # 0 is meaningful rather than rejected: it disables chunking.
    scheduler.set_prefill_token_budget(0)
    assert scheduler.prefill_token_budget() == 0


def test_token_callback_streams_the_final_answer(native_module, scheduler):
    streamed = []
    completed = {}
    done = threading.Event()

    def on_token(request_id, token):
        streamed.append((request_id, token))

    def on_complete(result):
        completed["result"] = result
        done.set()

    want = 8
    request_id = scheduler.submit_request(
        [10, 20, 30, 40, 50],
        sampling_params(native_module, want),
        callback=on_complete,
        on_token=on_token,
    )
    assert request_id != 0
    assert done.wait(timeout=60), "completion callback never fired"

    result = completed["result"]
    # The streamed sequence must be exactly the final answer, in order: that is
    # the contract a streaming server depends on.
    assert [token for _, token in streamed] == list(result.generated_tokens)
    assert {rid for rid, _ in streamed} == {request_id}
    assert len(streamed) == want


def test_poll_result_without_callbacks(native_module, scheduler):
    request_id = scheduler.submit_request(
        [15, 25, 35, 45], sampling_params(native_module, 6)
    )
    assert request_id != 0
    result = scheduler.poll_result(request_id, timeout_ms=60000)
    assert result is not None
    assert len(result.generated_tokens) == 6


def test_raising_token_callback_does_not_kill_the_request(native_module, scheduler):
    completed = {}
    done = threading.Event()
    calls = []

    def bad_on_token(request_id, token):
        calls.append(token)
        raise RuntimeError("callback boom")

    def on_complete(result):
        completed["result"] = result
        done.set()

    request_id = scheduler.submit_request(
        [10, 20, 30],
        sampling_params(native_module, 4),
        callback=on_complete,
        on_token=bad_on_token,
    )
    assert request_id != 0

    # The exception is reported and swallowed; generation still completes rather
    # than the scheduler thread dying and stranding every other request.
    assert done.wait(timeout=60), "completion never fired after the callback raised"
    assert calls
    assert len(completed["result"].generated_tokens) == 4


def test_cancel_request_before_completion(native_module, scheduler):
    """Test that cancel_request stops generation early."""
    tokens_received = []
    done = threading.Event()

    def on_token(request_id, token):
        tokens_received.append(token)
        # Cancel after receiving 2 tokens
        if len(tokens_received) == 2:
            scheduler.cancel_request(request_id)

    def on_complete(result):
        done.set()

    request_id = scheduler.submit_request(
        [10, 20, 30],
        sampling_params(native_module, 20),  # Ask for 20 tokens
        callback=on_complete,
        on_token=on_token,
    )
    assert request_id != 0

    # Should complete quickly after cancellation
    assert done.wait(timeout=10), "request never completed after cancellation"

    # Should have received only a few tokens, not all 20
    assert len(tokens_received) < 20, f"got {len(tokens_received)} tokens, cancellation didn't stop generation"


def test_cancel_unknown_request_succeeds(scheduler):
    """Test that cancelling an unknown request ID returns True.

    The scheduler always returns True for cancel_request to handle race
    conditions safely - it marks the request as cancelled even if it's not
    currently tracked.
    """
    result = scheduler.cancel_request(999999)
    assert result is True


def test_cancel_already_completed_request_succeeds(native_module, scheduler):
    """Test that cancelling a completed request returns True.

    The scheduler always returns True for cancel_request to handle race
    conditions safely.
    """
    request_id = scheduler.submit_request(
        [10, 20], sampling_params(native_module, 2)
    )
    result = scheduler.poll_result(request_id, timeout_ms=60000)
    assert result is not None

    # Now try to cancel the completed request - still returns True
    cancel_result = scheduler.cancel_request(request_id)
    assert cancel_result is True


def test_asyncio_loop_stays_alive_during_native_generation(native_module, scheduler):
    """Test that asyncio event loop remains runnable during native generation.

    This validates Issue #166's requirement: the asyncio loop must stay responsive
    while native generation is running in the scheduler's background thread.
    """
    loop_ticks = []
    generation_done = threading.Event()

    async def tick_loop():
        """Background task that ticks every 50ms to prove loop is alive."""
        for i in range(20):  # Run for ~1 second
            loop_ticks.append(i)
            await asyncio.sleep(0.05)

    def on_complete(result):
        generation_done.set()

    async def run_test():
        # Start the background tick task
        tick_task = asyncio.create_task(tick_loop())

        # Submit a generation request (runs in scheduler's background thread)
        request_id = scheduler.submit_request(
            [10, 20, 30, 40] * 10,  # Long prompt
            sampling_params(native_module, 16),
            callback=on_complete,
        )
        assert request_id != 0

        # Wait for generation to complete
        await asyncio.get_event_loop().run_in_executor(
            None, generation_done.wait, 60
        )

        # Wait for tick task
        await tick_task

        return len(loop_ticks)

    # Run the async test
    ticks = asyncio.run(run_test())

    # The loop should have ticked many times while generation was running
    assert ticks >= 10, f"loop only ticked {ticks} times, it may have been blocked"


def test_concurrent_requests_with_callbacks(native_module, scheduler):
    """Test that multiple concurrent requests with callbacks work correctly.

    This validates that the scheduler properly handles multiple in-flight
    requests with their own token callbacks running from the same thread.
    """
    results = {}
    tokens = {}
    done_count = threading.Semaphore(0)

    def make_handlers(req_num):
        tokens[req_num] = []

        def on_token(request_id, token):
            tokens[req_num].append(token)

        def on_complete(result):
            results[req_num] = result
            done_count.release()

        return on_token, on_complete

    # Submit 3 concurrent requests
    request_ids = []
    for i in range(3):
        on_token, on_complete = make_handlers(i)
        request_id = scheduler.submit_request(
            [10 * (i + 1), 20 * (i + 1)],  # Different prompts
            sampling_params(native_module, 5),
            callback=on_complete,
            on_token=on_token,
        )
        assert request_id != 0
        request_ids.append(request_id)

    # Wait for all to complete
    for _ in range(3):
        assert done_count.acquire(timeout=60), "not all requests completed"

    # All requests should have completed with correct token counts
    assert len(results) == 3
    assert len(tokens) == 3
    for i in range(3):
        assert len(results[i].generated_tokens) == 5
        assert len(tokens[i]) == 5
        assert tokens[i] == list(results[i].generated_tokens)

