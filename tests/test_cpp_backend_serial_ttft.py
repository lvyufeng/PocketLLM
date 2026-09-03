"""The serial path must report a real TTFT, not a placeholder.

``_generate_serial`` used to hardcode ``ttft_seconds=0.0``.  Nothing downstream
notices a zero until something divides by it: the Phase 3.5 latency benchmark
compares batch TTFT against serial TTFT and died with ZeroDivisionError, and any
serving metric that averages TTFT silently reports 0 for every serial request.

The stepped path is the one that can measure it -- it drives prefill and decode
itself, so it knows when the first token appeared.
"""

from __future__ import annotations

import time

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.cpp_backend import CppBackend

from test_cpp_backend_tp import SimpleTokenizer, TpFakeNative, make_tp_backend


def _request(max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id="ttft",
        prompt_tokens=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def test_serial_path_reports_nonzero_ttft() -> None:
    backend, _ = make_tp_backend()
    try:
        result = backend.generate([_request()])[0]
        assert result.timings.ttft_seconds > 0.0, "serial TTFT is still a placeholder"
    finally:
        backend.close()


def test_serial_ttft_excludes_decode() -> None:
    """TTFT must cover prefill only, so it stays below the total.

    A TTFT that accidentally measured the whole loop would compare equal to
    total_seconds and hide any prefill regression.  A slow fake decode makes the
    two separable.
    """

    class SlowDecodeNative(TpFakeNative):
        def QwenEngine(self, checkpoint, options, layer_count, max_context):
            engine = super().QwenEngine(checkpoint, options, layer_count, max_context)
            original = engine.decode_step

            def slow_decode_step(token: int):
                time.sleep(0.01)
                return original(token)

            engine.decode_step = slow_decode_step
            return engine

    backend, _ = make_tp_backend(native=SlowDecodeNative())
    try:
        result = backend.generate([_request(max_tokens=8)])[0]
        assert result.timings.ttft_seconds > 0.0
        # Seven slow decode steps sit between the first token and the last one.
        assert result.timings.ttft_seconds < result.timings.total_seconds / 2
    finally:
        backend.close()


def test_ttft_is_measured_from_request_start() -> None:
    """The clock starts when the request starts, not when prefill is entered."""
    backend, _ = make_tp_backend()
    try:
        start = time.perf_counter()
        result = backend.generate([_request()])[0]
        elapsed = time.perf_counter() - start
        assert 0.0 < result.timings.ttft_seconds <= elapsed
    finally:
        backend.close()


def test_single_rank_without_eos_reports_no_ttft() -> None:
    """Native generate() drives its own loop, so the first token is not visible.

    Reporting 0.0 here is honest rather than a placeholder; the alternative would
    be to attribute the whole call to prefill.
    """
    args = EngineArgs(model="model", backend="cpp")
    backend = CppBackend(args, native_module=TpFakeNative(), tokenizer=SimpleTokenizer())
    try:
        result = backend.generate([_request()])[0]
        assert result.timings.ttft_seconds == 0.0
        assert result.timings.total_seconds > 0.0
    finally:
        backend.close()
