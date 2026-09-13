"""Scheduler-backed async requests exposed to Python (Issue #167).

The surface tests need only the module. The behavioural tests drive a real
single-GPU QwenEngine over the synthetic fixture that
`cpp_engine/tests/test_batch_scheduler.cpp` writes, so the token callback is
driven by the actual scheduler thread rather than a stub. Generate it with:

    cmake --build <build> --target test_batch_scheduler
    ./<build>/tests/test_batch_scheduler
"""
from __future__ import annotations

import importlib
import os
import threading

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
