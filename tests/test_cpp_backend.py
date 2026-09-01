from __future__ import annotations

import threading
import time
import pytest

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    RequestCancelledError,
    SamplingParams,
    UnsupportedFeatureError,
)
from pocketllm.backends.cpp_backend import CppBackend
from pocketllm.cli import _args, build_parser


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text), 7]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class FakeResult:
    def __init__(self, top_token: int) -> None:
        self.top_token = top_token


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.closed = 0
        self.active = 0
        self.max_active = 0
        self._active_lock = threading.Lock()

    def generate(self, prompt_ids: list[int], max_tokens: int) -> list[FakeResult]:
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(("generate", list(prompt_ids), max_tokens))
            time.sleep(0.02)
            return [FakeResult(10 + index) for index in range(max_tokens)]
        finally:
            with self._active_lock:
                self.active -= 1

    def reset(self) -> None:
        self.calls.append(("reset",))

    def clear_prefix_cache(self) -> None:
        self.calls.append(("clear_prefix_cache",))

    def prefill(self, prompt_ids: list[int]) -> FakeResult:
        self.calls.append(("prefill", list(prompt_ids)))
        return FakeResult(10)

    def decode_step(self, token: int) -> FakeResult:
        self.calls.append(("decode_step", token))
        return FakeResult(token + 1)

    def close(self) -> None:
        self.closed += 1


class FakeOptions:
    def __init__(self) -> None:
        self.tp_world = 1
        self.tp_rank = 0
        self.device = 0
        self.prefill_chunk_tokens = 0
        self.kv_cache_dtype = "unset"
        self.attention_window = 0
        self.attention_sink_tokens = 0
        self.prefix_cache = False
        self.state_snapshot_interval_tokens = 0
        self.max_state_snapshots = 0
        self.mtp = False
        self.mtp_speculative_tokens = 1
        self.mtp_adaptive = False
        self.dspark_checkpoint = ""
        self.dflash2_checkpoint = ""
        self.nccl_id_path = ""
        self.temperature = 1.0
        self.top_p = 0.0
        self.top_k = 0
        self.sampling_seed = -1


class FakeNativeModule:
    QwenEngineOptions = FakeOptions

    def __init__(self) -> None:
        self.constructed: tuple[str, FakeOptions, int, int] | None = None

    @staticmethod
    def parse_qwen_kv_cache_dtype(value: str) -> str:
        return f"native:{value}"

    def QwenEngine(
        self,
        checkpoint: str,
        options: FakeOptions,
        layer_count: int,
        max_context: int,
    ) -> FakeEngine:
        self.constructed = (checkpoint, options, layer_count, max_context)
        return FakeEngine()


def make_backend(engine: FakeEngine | None = None) -> tuple[CppBackend, FakeEngine]:
    fake_engine = engine or FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=fake_engine,
        tokenizer=FakeTokenizer(),
    )
    return backend, fake_engine


def test_generate_converts_native_results_and_usage() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt="hello",
        request_id="req-generate",
        sampling_params=SamplingParams(max_tokens=3),
    )

    result = backend.generate([request])[0]

    assert engine.calls == [("generate", [5, 7], 3)]
    assert result.request_id == "req-generate"
    assert result.token_ids == [10, 11, 12]
    assert result.text == "<10><11><12>"
    assert result.finish_reason == "length"
    assert result.usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


def test_stream_matches_native_prefill_decode_order() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=3),
    )

    events = list(backend.stream(request))

    # No reset()/clear_prefix_cache(): native prefill() owns prefix matching, so
    # resetting per request would disable configured prefix reuse.
    assert engine.calls == [
        ("prefill", [4, 5]),
        ("decode_step", 10),
        ("decode_step", 11),
    ]
    assert [event.token_id for event in events] == [10, 11, 12]
    assert [event.text for event in events] == ["<10>", "<11>", "<12>"]
    assert [event.finish_reason for event in events] == [None, None, "length"]
    assert events[-1].usage is not None
    assert events[-1].usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


def test_stream_cancellation_stops_before_next_decode_step() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt_tokens=[4],
        request_id="req-cancel",
        sampling_params=SamplingParams(max_tokens=3),
    )
    stream = backend.stream(request)

    assert next(stream).token_id == 10
    assert backend.cancel(request.request_id) is True
    with pytest.raises(RequestCancelledError):
        next(stream)

    assert engine.calls == [("prefill", [4])]


def test_requests_are_serialized_around_one_native_session() -> None:
    backend, engine = make_backend()
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def generate(index: int) -> None:
        try:
            barrier.wait()
            backend.generate(
                [
                    GenerationRequest(
                        prompt_tokens=[index + 1],
                        request_id=f"req-{index}",
                        sampling_params=SamplingParams(max_tokens=1),
                    )
                ]
            )
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=generate, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2.0)

    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert engine.max_active == 1


def test_close_releases_native_engine_and_rejects_new_work() -> None:
    backend, engine = make_backend()

    backend.close()
    backend.close()

    assert engine.closed == 1
    assert backend.health().status == "stopped"
    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.generate(
            [GenerationRequest(prompt_tokens=[1], request_id="req-closed")]
        )


def test_close_during_stream_releases_at_safe_boundary() -> None:
    backend, engine = make_backend()
    stream = backend.stream(
        GenerationRequest(
            prompt_tokens=[1],
            request_id="req-close-stream",
            sampling_params=SamplingParams(max_tokens=2),
        )
    )

    assert next(stream).token_id == 10
    backend.close()
    assert engine.closed == 0
    with pytest.raises(RuntimeError, match="backend is closed"):
        next(stream)
    assert engine.closed == 1


def test_native_options_normalize_cli_device_and_auto_kv_dtype() -> None:
    namespace = build_parser().parse_args(
        [
            "serve",
            "--model",
            "checkpoint",
            "--backend",
            "cpp",
            "--device",
            "cuda:2",
            "--max-model-len",
            "4096",
        ]
    )
    native = FakeNativeModule()

    backend = CppBackend(_args(namespace), native_module=native, tokenizer=FakeTokenizer())

    assert native.constructed is not None
    checkpoint, options, layer_count, max_context = native.constructed
    assert checkpoint == "checkpoint"
    assert options.device == 2
    assert options.kv_cache_dtype == "native:fp16"
    assert options.temperature == 0.0
    assert (layer_count, max_context) == (0, 4096)
    backend.close()


@pytest.mark.parametrize("device", ["gpu", "cuda:", -1, True])
def test_invalid_native_device_is_rejected(device: object) -> None:
    native = FakeNativeModule()
    with pytest.raises(ConfigurationError, match="device"):
        CppBackend(
            EngineArgs(model="checkpoint", backend="cpp", device=device),
            native_module=native,
            tokenizer=FakeTokenizer(),
        )


def test_cpp_capabilities_match_phase_one_surface() -> None:
    backend, _ = make_backend()

    capabilities = backend.capabilities
    assert capabilities.models == ("qwen3.5",)
    assert capabilities.model_formats == ("safetensors",)
    assert capabilities.supports_streaming is True
    assert capabilities.supports_batch is False


def test_cpp_capabilities_follow_the_linked_device_backend() -> None:
    class AscendModule(FakeNativeModule):
        backend = "ascend"

    native = AscendModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    capabilities = backend.capabilities
    # The Ascend build rejects DSpark/DFlash2 drafters, so they must not be
    # advertised just because the CUDA build supports them.
    assert capabilities.supports_speculative_decoding == ("mtp",)
    assert capabilities.devices == ("ascend",)
    backend.close()


def test_cancel_only_accepts_active_request_ids() -> None:
    backend, _ = make_backend()

    assert backend.cancel("req-never-submitted") is False

    stream = backend.stream(
        GenerationRequest(
            prompt_tokens=[1],
            request_id="req-active",
            sampling_params=SamplingParams(max_tokens=2),
        )
    )
    assert next(stream).token_id == 10
    assert backend.cancel("req-active") is True
    assert backend.cancel("req-other") is False
    with pytest.raises(RequestCancelledError):
        next(stream)
    assert backend.cancel("req-active") is False
    assert backend.active_request_count() == 0


def test_stream_text_deltas_use_cumulative_tokenizer_decode() -> None:
    class MergingTokenizer:
        """Emulates a tokenizer whose pieces only render as a full sequence."""

        def encode(self, text: str) -> list[int]:
            return [1]

        def decode(self, token_ids: list[int]) -> str:
            return "".join("ab"[index % 2] for index, _ in enumerate(token_ids))

    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=engine,
        tokenizer=MergingTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-delta",
        sampling_params=SamplingParams(max_tokens=3),
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["a", "b", "a"]
    backend.close()


def test_cpp_backend_rejects_unexposed_sampling_controls() -> None:
    backend, _ = make_backend()
    request = GenerationRequest(
        prompt_tokens=[1],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.5),
    )

    with pytest.raises(UnsupportedFeatureError, match="greedy"):
        backend.generate([request])
