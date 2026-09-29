from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
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


class ByteLevelTokenizer:
    """The other kind of tokenizer: pieces are bytes, and a decode renders what it has.

    A prefix that ends inside a multi-byte character is decoded with ``errors="replace"``, which is
    where U+FFFD comes from. This is the tokenizer the held-back tail exists for.
    """

    def __init__(self, pieces: dict[int, bytes]) -> None:
        self._pieces = pieces

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self._pieces[token] for token in token_ids).decode(
            "utf-8", errors="replace"
        )


class TemplateTokenizer(FakeTokenizer):
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
        return [71, 72]



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
        self.detected: list[str] = []

    @staticmethod
    def parse_qwen_kv_cache_dtype(value: str) -> str:
        return f"native:{value}"

    @staticmethod
    def registered_architectures() -> list[str]:
        return ["deepseek_v4", "qwen3_5"]

    def detect_architecture(self, checkpoint: str) -> str:
        self.detected.append(checkpoint)
        return "qwen3_5"

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


def scripted_caps(*, structured_outputs: bool = True, logprobs: bool = True) -> object:
    """The engine's own declaration, as a scheduler double reports it through `engine_caps`.

    One helper rather than one per double because it is one fact: the capability is the engine's, the
    scheduler is only the thing the adapter can reach it through, and a double that disagreed with
    the others about it would be a double whose structured-output tests were about something else.

    The defaults are a CUDA build's answers, which is what most of these tests are describing. Each
    is parameterized rather than fixed because the audit is supposed to *follow* the declaration, and
    the Ascend build is the case where it has to: no per-row sampler, so no ranking, so a request for
    log probabilities is refused by name instead of admitted and answered with a 500.
    """
    return SimpleNamespace(structured_outputs=structured_outputs, logprobs=logprobs)


def test_cpp_chat_prompt_uses_tokenizer_owned_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-template",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_mode": "chat",
        },
    )

    assert backend._prompt_ids(request) == [71, 72]
    assert tokenizer.template_calls[0][0] == [{"role": "user", "content": "hi"}]
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    backend.close()


def test_cpp_pre_tokenized_prompt_bypasses_chat_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt_tokens=[7, 8],
        request_id="req-template-tokens",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": [{"role": "user", "content": "ignored"}]},
    )

    assert backend._prompt_ids(request) == [7, 8]
    assert tokenizer.template_calls == []
    backend.close()


def test_cpp_raw_prompt_bypasses_chat_template() -> None:
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt="raw completion",
        request_id="req-template-raw",
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert backend._prompt_ids(request) == [len("raw completion"), 7]
    assert tokenizer.template_calls == []
    backend.close()


def test_generate_converts_native_results_and_usage() -> None:
    backend, engine = make_backend()
    request = GenerationRequest(
        prompt="hello",
        request_id="req-generate",
        sampling_params=SamplingParams(max_tokens=3),
    )

    result = backend.generate([request])[0]

    # Without a known EOS the adapter delegates to native generate().
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


def test_batched_native_error_is_raised_and_clears_request() -> None:
    class NativeSamplingParams:
        pass

    class NativeModule:
        QwenBatchSamplingParams = NativeSamplingParams

    class NativeResult:
        error = "synthetic failure"

    class Scheduler:
        def __init__(self) -> None:
            self.next_id = 16
            self.polled = []

        def submit_request(self, prompt_ids, sampling, callback, on_token=None, constraint=None):
            self.next_id += 1
            return self.next_id

        def poll_result(self, request_id, timeout_ms):
            self.polled.append(request_id)
            return NativeResult()

        def engine_caps(self):
            return scripted_caps()

    backend, _ = make_backend()
    backend._native = NativeModule()
    scheduler = Scheduler()
    backend._scheduler = scheduler
    backend._batching_enabled = True
    requests = [
        GenerationRequest(
            prompt_tokens=[4, 5],
            request_id=f"req-native-error-{index}",
            sampling_params=SamplingParams(max_tokens=1),
        )
        for index in range(2)
    ]

    with pytest.raises(RuntimeError, match="native C\\+\\+ generation failed: synthetic failure"):
        backend.generate(requests)

    assert scheduler.polled == [17, 18]
    assert backend.active_request_count() == 0
    backend.close()


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
            "--device-ids",
            "2",
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
    # The served models come from the native engine registry rather than a
    # literal here, so a build that links a second engine reports it.
    assert capabilities.models == ("qwen3_5",)
    # Both formats, because the backend serves both: a safetensors directory, and
    # the single-file GGUF export the Qwen3.5 path reads straight out of its own
    # header. A capability list that under-reported formats would make
    # `pocketllm serve` refuse a checkpoint this adapter can actually run.
    assert capabilities.model_formats == ("safetensors", "gguf")
    assert capabilities.supports_streaming is True
    assert capabilities.supports_batch is False


def test_cpp_capabilities_list_the_registered_architectures() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    assert backend.capabilities.models == ("deepseek_v4", "qwen3_5")
    backend.close()


def test_cpp_engine_kind_is_detected_from_the_checkpoint() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(model="checkpoint", backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    # Detection ran against the checkpoint and picked the Qwen runtime, without
    # an engine_kind having been supplied.
    assert native.detected == ["checkpoint"]
    assert native.constructed is not None
    backend.close()


def test_a_directory_holding_a_gguf_reaches_the_engine_as_the_file_it_holds(
    tmp_path, monkeypatch
) -> None:
    """The native reader dispatches on the path's own suffix, so the file has to arrive as one.

    The released ternary artifact is a bare GGUF in a directory and nothing else. Handed the
    directory, the reader opens it as a safetensors index and fails on a ``config.json`` that is not
    there -- which is what ``pocketllm serve`` did on that checkpoint, and it looked like a missing
    model rather than a misnamed path. Detection is asked the same path, so an answer that named the
    directory here would route and construct the checkpoint from two different things.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    bundle = ckpt / "model.gguf"
    bundle.write_bytes(b"GGUF")
    monkeypatch.setattr(
        "pocketllm.backends.cpp_backend.gguf_checkpoint_file", lambda path: str(bundle)
    )

    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(model=str(ckpt), backend="cpp"),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )
    try:
        assert native.detected == [str(bundle)]
        assert native.constructed is not None
        assert native.constructed[0] == str(bundle)
    finally:
        backend.close()


def test_cpp_unknown_architecture_is_reported_not_guessed() -> None:
    class UnknownArchModule(FakeNativeModule):
        def detect_architecture(self, checkpoint: str) -> str:
            return "llama"

    with pytest.raises(UnsupportedFeatureError, match="llama"):
        CppBackend(
            EngineArgs(model="checkpoint", backend="cpp"),
            native_module=UnknownArchModule(),
            tokenizer=FakeTokenizer(),
        )


def test_cpp_engine_kind_override_skips_detection() -> None:
    native = FakeNativeModule()
    backend = CppBackend(
        EngineArgs(
            model="checkpoint",
            backend="cpp",
            backend_options={"engine_kind": "qwen"},
        ),
        native_module=native,
        tokenizer=FakeTokenizer(),
    )

    assert native.detected == []
    backend.close()


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


def test_stream_holds_back_a_character_a_token_ended_inside() -> None:
    """The stream sends the answer, and the replacement character a byte-level decode puts where a
    character is still arriving is not part of it. The tail goes out with the token that finishes
    the character, and the whole stream is the decode of the whole generation."""
    tokenizer = ByteLevelTokenizer(
        pieces={10: b"\xe4\xbd", 11: b"\xa0\xe5", 12: b"\xa5\xbd", 13: b"!"}
    )
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=FakeEngine(),
        tokenizer=tokenizer,
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-half-character",
        sampling_params=SamplingParams(max_tokens=4),
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["", "你", "好", "!"]
    assert "".join(event.text for event in events) == "你好!"
    assert not any("�" in event.text for event in events)
    backend.close()


def test_a_stream_that_ends_on_eos_sends_what_its_decode_was_holding() -> None:
    """No token is coming to settle the tail, so the stream would end one character short of the
    decode ``generate`` reports for the same ids. EOS itself is still not emitted."""
    tokenizer = ByteLevelTokenizer(pieces={10: b"\xe4\xbd", 11: b"\xa0"})
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": 11}),
        engine=FakeEngine(),
        tokenizer=tokenizer,
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-eos-tail",
        sampling_params=SamplingParams(max_tokens=4),
    )

    events = list(backend.stream(request))

    assert [event.token_id for event in events] == [10, None]
    assert [event.text for event in events] == ["", "�"]
    assert events[-1].finish_reason == "stop"
    # The same ids through an unstreamed decode, which is what the stream has to agree with.
    assert "".join(event.text for event in events) == tokenizer.decode([10])
    backend.close()


def test_generate_stops_at_eos_and_reports_stop() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": 11}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-eos-generate",
        sampling_params=SamplingParams(max_tokens=4),
    )

    result = backend.generate([request])[0]

    # With a known EOS the adapter drives prefill/decode itself so the session is
    # never advanced past EOS, and EOS stays out of the visible output.
    assert engine.calls == [("prefill", [4, 5]), ("decode_step", 10)]
    assert result.token_ids == [10]
    assert result.text == "<10>"
    assert result.finish_reason == "stop"
    assert result.usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    backend.close()


def test_stream_stops_at_eos_without_another_decode_step() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": [12]}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-eos-stream",
        sampling_params=SamplingParams(max_tokens=8),
    )

    events = list(backend.stream(request))

    assert engine.calls == [
        ("prefill", [4, 5]),
        ("decode_step", 10),
        ("decode_step", 11),
    ]
    assert [event.token_id for event in events] == [10, 11, None]
    assert [event.text for event in events] == ["<10>", "<11>", ""]
    assert [event.finish_reason for event in events] == [None, None, "stop"]
    assert events[-1].usage is not None
    assert events[-1].usage.as_dict() == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    backend.close()


def test_generate_does_not_advance_the_session_past_eos() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": 11}),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )

    backend.generate([
        GenerationRequest(
            prompt_tokens=[4],
            request_id="req-a",
            sampling_params=SamplingParams(max_tokens=6),
        )
    ])
    backend.generate([
        GenerationRequest(
            prompt_tokens=[5],
            request_id="req-b",
            sampling_params=SamplingParams(max_tokens=6),
        )
    ])

    # Each request stops at the EOS step, so no decode runs on text the caller
    # never received and the next request starts from a clean position.
    assert engine.calls == [
        ("prefill", [4]),
        ("decode_step", 10),
        ("prefill", [5]),
        ("decode_step", 10),
    ]
    backend.close()


def test_eos_comes_from_the_native_engine_when_available() -> None:
    class EosEngine(FakeEngine):
        def eos_id(self) -> int:
            return 11

    backend, _ = make_backend(EosEngine())

    assert backend.eos_token_ids == frozenset({11})
    assert backend.capabilities.details["eos_source"] == "native engine"
    backend.close()


def test_eos_falls_back_to_generation_config(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [7, 9]}', encoding="utf-8")
    (tmp_path / "config.json").write_text('{"eos_token_id": 1}', encoding="utf-8")

    backend = CppBackend(
        EngineArgs(model=str(tmp_path), backend="cpp"),
        engine=FakeEngine(),
        tokenizer=FakeTokenizer(),
    )

    # generation_config.json wins over config.json because it is what the
    # checkpoint declares for generation.
    assert backend.eos_token_ids == frozenset({7, 9})
    assert backend.capabilities.details["eos_source"] == "generation_config.json"
    backend.close()


def test_without_any_eos_generation_ends_on_the_token_budget() -> None:
    backend, _ = make_backend()

    assert backend.eos_token_ids == frozenset()
    assert backend.capabilities.details["eos_source"] == "none"
    result = backend.generate(
        [
            GenerationRequest(
                prompt_tokens=[1],
                request_id="req-no-eos",
                sampling_params=SamplingParams(max_tokens=2),
            )
        ]
    )[0]
    assert result.finish_reason == "length"
    backend.close()


def test_without_a_budget_the_native_engine_gets_all_the_prompt_leaves() -> None:
    """No ``max_tokens`` reaches native generate() as this engine's own context minus the prompt.

    The engine is built with that same number (8192 when nothing was configured), so a budget
    derived from it is one the engine can actually run, and the answer then ends at EOS or when
    the context runs out -- not at a length invented before the context was known.
    """
    backend, engine = make_backend()
    request = GenerationRequest(prompt_tokens=[1, 2, 3], request_id="req-open-ended")

    result = backend.generate([request])[0]

    assert engine.calls == [("generate", [1, 2, 3], 8189)]
    assert result.finish_reason == "length"
    backend.close()


def test_a_stepped_request_without_a_budget_runs_to_the_context() -> None:
    """The stepped path resolves an absent budget against the same configured context."""
    backend = CppBackend(
        EngineArgs(
            model="model",
            backend="cpp",
            max_model_len=8,
            # A known EOS puts the request on the stepped path; the fake never emits it.
            backend_options={"eos_token_id": 99},
        ),
        engine=FakeEngine(),
        tokenizer=FakeTokenizer(),
    )
    try:
        result = backend.generate(
            [GenerationRequest(prompt_tokens=[1, 2, 3], request_id="req-open-ended")]
        )[0]
        # Eight positions, three of them the prompt.
        assert result.token_ids == [10, 11, 12, 13, 14]
        assert result.finish_reason == "length"
    finally:
        backend.close()


def test_a_batched_request_without_a_budget_asks_for_all_the_prompt_leaves() -> None:
    """The batched path derives the same number, so one rule covers all three."""
    submitted: list[object] = []

    class NativeSamplingParams:
        pass

    class NativeModule:
        QwenBatchSamplingParams = NativeSamplingParams

    class NativeResult:
        error = ""
        generated_tokens = [10]
        finish_reason = "length"
        prompt_tokens = 3
        completion_tokens = 1
        total_seconds = 0.1
        ttft_seconds = 0.05

    class Scheduler:
        def submit_request(self, prompt_ids, sampling, callback, on_token=None, constraint=None):
            submitted.append(sampling)
            return 1

        def poll_result(self, request_id, timeout_ms):
            return NativeResult()

        def engine_caps(self):
            return scripted_caps()

    backend, _ = make_backend()
    backend._native = NativeModule()
    backend._scheduler = Scheduler()
    backend._batching_enabled = True
    try:
        backend.generate([GenerationRequest(prompt_tokens=[1, 2, 3], request_id="req-open-ended")])
    finally:
        backend.close()

    assert len(submitted) == 1
    assert submitted[0].max_new_tokens == 8189


def test_invalid_eos_override_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="eos_token_id"):
        CppBackend(
            EngineArgs(model="model", backend="cpp", backend_options={"eos_token_id": "11"}),
            engine=FakeEngine(),
            tokenizer=FakeTokenizer(),
        )


def test_cpp_backend_rejects_unexposed_sampling_controls() -> None:
    """A temperature this engine cannot apply is refused by name, not generated as if greedy.

    The engine built here is the serialized session's, whose sampler is fixed at what the adapter
    constructed it with; asking for 0.5 and getting greedy text back is the failure the refusal
    exists to prevent. The message names the field and the value the engine would use instead, so a
    caller can act on it without reading this test.
    """
    backend, _ = make_backend()
    request = GenerationRequest(
        prompt_tokens=[1],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.5),
    )

    with pytest.raises(UnsupportedFeatureError, match='"temperature" = 0.5'):
        backend.generate([request])


# --------------------------------------------------------------------------------------------------
# What the batch path hands the client
# --------------------------------------------------------------------------------------------------


class ScriptedScheduler:
    """A scheduler that returns one canned native result, so the answer shape is the subject.

    It mirrors the binding's `submit_request` argument for argument, constraint included, because a
    double that took fewer would accept a call the real one refuses -- and one of the arguments is
    the object under test. `engine_caps` is the engine's declaration as the scheduler reports it,
    which is where the structured-output capability is read from.
    """

    def __init__(self, result, *, structured_outputs: bool = True, logprobs: bool = True) -> None:
        self.result = result
        self.submitted: list[object] = []
        self.constraints: list[object | None] = []
        self.structured_outputs = structured_outputs
        self.logprobs = logprobs

    def engine_caps(self) -> object:
        return scripted_caps(structured_outputs=self.structured_outputs, logprobs=self.logprobs)

    def submit_request(
        self, prompt_ids, sampling, callback, on_token=None, constraint=None
    ) -> int:
        self.submitted.append(sampling)
        self.constraints.append(constraint)
        return 1

    def poll_result(self, request_id, timeout_ms):
        return self.result


class ScriptedNativeModule:
    """The native module as far as the batch path reaches: sampling params, and the constraint.

    `Tokenizer` and the two factories are recorded rather than implemented, because what the adapter
    is responsible for is *which* factory a `response_format` selects, over *whose* vocabulary. What
    the factory then builds is the C++ validator's business and is tested there.
    """

    class QwenBatchSamplingParams:
        pass

    def __init__(self) -> None:
        self.tokenizer_paths: list[str] = []
        self.object_calls: list[object] = []
        self.schema_calls: list[tuple[object, str]] = []

    class _Constraint:
        def __init__(self, kind: str, schema: str = "") -> None:
            self.kind = kind
            self.schema = schema

        def __repr__(self) -> str:
            return f"<constraint {self.kind}>"

    def Tokenizer(self, checkpoint: str):  # noqa: N802 - the binding's name
        self.tokenizer_paths.append(checkpoint)
        return f"tokenizer:{checkpoint}"

    def make_json_object_constraint(self, tokenizer):  # noqa: N802 - the binding's name
        self.object_calls.append(tokenizer)
        return ScriptedNativeModule._Constraint("object")

    def make_json_schema_constraint(self, tokenizer, schema_json):  # noqa: N802 - the binding's name
        self.schema_calls.append((tokenizer, schema_json))
        return ScriptedNativeModule._Constraint("schema", schema_json)


def batched_backend_with(
    result, *, structured_outputs: bool = True, logprobs: bool = True
) -> CppBackend:
    backend, _ = make_backend()
    backend._native = ScriptedNativeModule()
    backend._scheduler = ScriptedScheduler(
        result, structured_outputs=structured_outputs, logprobs=logprobs
    )
    backend._batching_enabled = True
    return backend


class ScriptedNativeResult:
    def __init__(self, *, tokens, finish_reason, constraint_completed=False, logprobs=()) -> None:
        self.error = ""
        self.generated_tokens = list(tokens)
        self.finish_reason = finish_reason
        self.constraint_completed = constraint_completed
        # Parallel to `generated_tokens`, as the binding reports it: one entry per token the engine
        # produced, including the terminal stop token the answer does not contain.
        self.logprobs = list(logprobs)
        self.prompt_tokens = 2
        self.completion_tokens = len(tokens)
        self.total_seconds = 0.1
        self.ttft_seconds = 0.05


def test_a_stop_token_is_not_part_of_the_answer() -> None:
    """The engine returns the stop token; the client must not see it.

    The engine keeps it because its KV cache has to agree with what it reports, and every consumer is
    responsible for dropping it: the deleted C++ front end's `strip_stop_token` did it before
    detokenizing, and the scheduler's streaming path never emits it. The non-streaming result was the
    one place it leaked through, so one request answered through `pocketllm serve` came back with a
    visible `<|im_end|>` on the end and the same request through the native binary did not. The `cpp`
    served-path fixture is what found it -- it is recorded from the serial path and compares ids.
    """
    backend = batched_backend_with(
        ScriptedNativeResult(tokens=[271, 93884, 12149, 248046], finish_reason="stop")
    )
    try:
        result = backend.generate(
            [GenerationRequest(prompt_tokens=[1, 2], request_id="req-stop")]
        )[0]
    finally:
        backend.close()

    assert result.token_ids == [271, 93884, 12149]
    assert "<|im_end|>" not in result.text
    assert result.finish_reason == "stop"
    # Usage still counts the stop step, which is the convention the serial path uses as well, so the
    # two paths stay comparable rather than the batch one reporting one token fewer.
    assert result.usage.completion_tokens == 4


def test_a_terminal_constraint_token_is_part_of_the_answer() -> None:
    """"stop" also describes a structured-output close token, which is output and must be kept."""
    backend = batched_backend_with(
        ScriptedNativeResult(tokens=[1, 2, 3], finish_reason="stop", constraint_completed=True)
    )
    try:
        result = backend.generate(
            [GenerationRequest(prompt_tokens=[1, 2], request_id="req-constraint")]
        )[0]
    finally:
        backend.close()

    assert result.token_ids == [1, 2, 3]


def test_a_length_capped_answer_keeps_its_last_token() -> None:
    backend = batched_backend_with(ScriptedNativeResult(tokens=[1, 2, 3], finish_reason="length"))
    try:
        result = backend.generate(
            [GenerationRequest(prompt_tokens=[1, 2], request_id="req-length")]
        )[0]
    finally:
        backend.close()

    assert result.token_ids == [1, 2, 3]


# --------------------------------------------------------------------------------------------------
# How an answer is read back
# --------------------------------------------------------------------------------------------------


class ScriptedTextTokenizer(FakeTokenizer):
    """A tokenizer whose decode is whatever the test scripted.

    What is under test is the reading of a decoded text, so the ids only have to be non-empty and
    the text only has to be the shape the engine would have produced.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, token_ids: list[int]) -> str:
        return self.text


class RunningTextTokenizer(FakeTokenizer):
    """The running decode of N tokens, scripted one entry per token.

    A stream calls ``decode`` with the whole generated prefix on every token, so a tokenizer that
    answered the same text each time would hand the split a marker it had not reached yet.
    """

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def decode(self, token_ids: list[int]) -> str:
        return self.texts[min(len(token_ids), len(self.texts)) - 1]


class SurfaceTokenizer(FakeTokenizer):
    """Decodes each id to its own surface, which is what a per-token ranking is indexed against.

    ``ScriptedTextTokenizer`` answers the same string for every id, which is right for a test about
    how a whole text is read and wrong for one about a ranking: the rendering walks the surfaces to
    find how many bytes of the answer each position covers, and a tokenizer that gave every position
    the same answer would give them all the same width.
    """

    def __init__(self, surfaces: dict[int, str]) -> None:
        self._surfaces = surfaces

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._surfaces.get(int(token), "") for token in token_ids)


class ScriptedLogprob:
    """One position's ranking, in the shape the binding reports it.

    Duck-typed rather than the real ``pocketllm_cpp.TokenLogprob``, for the reason this file's other
    fakes are: the rendering is Python, and it has to be testable on a build with no extension.
    """

    def __init__(self, logprob, *, top_tokens=(), top_logprobs=(), present=True) -> None:
        self.present = present
        self.logprob = logprob
        self.top_tokens = list(top_tokens)
        self.top_logprobs = list(top_logprobs)


def _engine_sampling(**fields):
    """A scheduler that declares what the engine's sampler can vary per request.

    The adapter reads these off the scheduler's capability object, so a scripted scheduler that does
    not answer ``engine_caps`` is read as the serialized session -- fixed at greedy. Tests that need
    the other reading attach one of these; the defaults are that session's.
    """
    declared = dict(
        per_request_sampling=False,
        per_request_top_k=False,
        fixed_temperature=0.0,
        fixed_top_p=1.0,
        fixed_top_k=20,
        fixed_seed=0,
        # Whether the sampler applies a per-row token mask travels with the same declaration and for
        # the same reason: it is the device sampler that applies one, so an engine that samples
        # engine-wide applies no mask either. A double that said otherwise would let a test pass on
        # a request the real engine answers with unconstrained text.
        structured_outputs=True,
    )
    declared.update(fields)
    return lambda: SimpleNamespace(**declared)


def _ranking_backend(surfaces, tokens, rankings, *, finish_reason="length"):
    backend, _ = make_backend()
    backend._tokenizer = SurfaceTokenizer(surfaces)
    backend._native = ScriptedNativeModule()
    backend._scheduler = ScriptedScheduler(
        ScriptedNativeResult(
            tokens=tokens, finish_reason=finish_reason, logprobs=rankings
        )
    )
    backend._batching_enabled = True
    return backend


QWEN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}

QWEN_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)


def _declaring_checkpoint(tmp_path, architecture: str) -> str:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps({"model_type": architecture}), encoding="utf-8"
    )
    return str(ckpt)


def _scripted_backend(checkpoint: str, text: str, *, finish_reason: str = "stop"):
    backend = CppBackend(
        EngineArgs(model=checkpoint, backend="cpp"),
        engine=FakeEngine(),
        tokenizer=ScriptedTextTokenizer(text),
    )
    backend._native = ScriptedNativeModule()
    backend._scheduler = ScriptedScheduler(
        ScriptedNativeResult(tokens=[1, 2, 3], finish_reason=finish_reason)
    )
    backend._batching_enabled = True
    return backend


def test_a_tool_call_is_read_out_of_the_answer(tmp_path) -> None:
    """The engine returns text; which part of it is a call is this checkpoint's architecture.

    `pocketllm serve --backend cpp` returned the call as prose while the same checkpoint through the
    C++ front end returned `tool_calls`, because the reading lived in that front end's sidecar. A
    client branching on the field would have run nothing.
    """
    backend = _scripted_backend(_declaring_checkpoint(tmp_path, "qwen3_5"), QWEN_CALL)
    try:
        result = backend.generate(
            [
                GenerationRequest(
                    prompt_tokens=[1, 2],
                    request_id="req-call",
                    metadata={
                        "messages": [{"role": "user", "content": "weather?"}],
                        "tools": [QWEN_TOOL],
                        "thinking_mode": "chat",
                    },
                )
            ]
        )[0]
    finally:
        backend.close()

    assert result.text == ""
    # A call is *why* the generation ended: under "stop" a client reads it as the final answer.
    assert result.finish_reason == "tool_calls"
    call = result.metadata["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    # A client echoes this back to attribute a tool result to the call it answers, so it has to be
    # named even though Qwen's own reading of a call -- out of a prompt -- has no id to give.
    assert call["id"]
    # The schema is what makes "3" an integer rather than the string the XML spelling alone gives.
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_a_thinking_answer_is_split_into_reasoning_and_content(tmp_path) -> None:
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"), "weighing it up</think>the answer"
    )
    try:
        result = backend.generate(
            [
                GenerationRequest(
                    prompt_tokens=[1, 2],
                    request_id="req-think",
                    metadata={
                        "messages": [{"role": "user", "content": "hi"}],
                        "thinking_mode": "thinking",
                    },
                )
            ]
        )[0]
    finally:
        backend.close()

    assert result.text == "the answer"
    assert result.metadata["reasoning_content"] == "weighing it up"
    assert "tool_calls" not in result.metadata


def test_an_undeclared_architecture_reads_no_tool_calls(tmp_path) -> None:
    """No call syntax is registered, so the XML is what the client sees.

    Inventing a parse for a syntax nobody has read would drop or corrupt calls silently. The
    reasoning split is a different question and is generic -- it is the marker text every
    thinking-mode template writes -- so it still happens.
    """
    undeclared = tmp_path / "plain"
    undeclared.mkdir()
    backend = _scripted_backend(str(undeclared), QWEN_CALL)
    try:
        result = backend.generate(
            [
                GenerationRequest(
                    prompt_tokens=[1, 2],
                    request_id="req-plain",
                    metadata={
                        "messages": [{"role": "user", "content": "weather?"}],
                        "tools": [QWEN_TOOL],
                        "thinking_mode": "chat",
                    },
                )
            ]
        )[0]
    finally:
        backend.close()

    assert result.text == QWEN_CALL
    assert "tool_calls" not in result.metadata


def test_a_completion_is_never_read_back(tmp_path) -> None:
    """Text in, text out: `</think>` in a completion is four characters the model wrote.

    The C++ front end drew the same line -- it parsed the assistant message and handed
    `/v1/completions` the raw decode -- and `messages` in the metadata is what tells the two routes
    apart here, the same field `_prompt_ids` already keys on.
    """
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"), "weighing it up</think>the answer"
    )
    try:
        result = backend.generate(
            [GenerationRequest(prompt="just text", request_id="req-completion")]
        )[0]
    finally:
        backend.close()

    assert result.text == "weighing it up</think>the answer"
    assert result.metadata == {}


def test_the_stream_sends_reasoning_as_its_own_field(tmp_path) -> None:
    """A thinking answer streams its reasoning rather than hiding it until the marker closes."""
    backend = CppBackend(
        EngineArgs(model=_declaring_checkpoint(tmp_path, "qwen3_5"), backend="cpp"),
        engine=FakeEngine(),
        tokenizer=RunningTextTokenizer(
            ["plan", "plan</think>", "plan</think>done"]
        ),
    )
    request = GenerationRequest(
        prompt_tokens=[4],
        request_id="req-stream-think",
        sampling_params=SamplingParams(max_tokens=3),
        metadata={
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_mode": "thinking",
        },
    )

    events = list(backend.stream(request))

    # The prompt ended inside the thinking block, so the first tokens are reasoning and the content
    # does not begin until the marker closes it -- which is the C++ front end's reading too.
    assert [event.text for event in events] == ["", "", "done"]
    assert [event.metadata.get("reasoning_content") for event in events] == [
        "plan",
        None,
        None,
    ]
    # The answer a stream sends equals the answer the unstreamed path returns, which is what makes
    # the two routes comparable at all. The marker itself belongs to neither field.
    assert "".join(event.text for event in events) == "done"
    backend.close()


# --------------------------------------------------------------------------------------------------
# The request surface the native front end answered
# --------------------------------------------------------------------------------------------------


def test_a_stop_sequence_cuts_the_answer_the_engine_returned(tmp_path) -> None:
    """`stop` is matched here, not in the engine: the scheduler cannot see a text-level sequence.

    The native server truncated the decoded text and reported "stop" for it, so a client asking the
    same thing of the same engine gets the same completion out of either front end. The engine's run
    is untouched -- it goes to its budget, because a sequence the sampler never sees cannot end one --
    which is why the budget below is "length" and the answer's reason is still "stop".
    """
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"), "the answer USER: more", finish_reason="length"
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-stop",
                sampling_params=SamplingParams(max_tokens=8, stop=["USER:"]),
                metadata={"messages": [{"role": "user", "content": "hi"}]},
            )
        ])[0]
    finally:
        backend.close()

    assert result.text == "the answer "
    assert result.finish_reason == "stop"
    # The tokens are what the engine executed, so they are not cut with the text: `usage` reports the
    # steps that were really taken, which is the pair the native server reported.
    assert result.token_ids == [1, 2, 3]
    assert result.usage.completion_tokens == 3


def test_a_stop_sequence_is_applied_before_the_answer_is_read(tmp_path) -> None:
    """The cut comes first, so a sequence that stops before a call does not leave a call behind.

    This is the order the deleted C++ front end used, and the difference only shows when the answer
    is both text and markup: reading the whole decode first would find the call, report it, and send
    `finish_reason: "tool_calls"` -- telling a client to run a function it had asked to stop short
    of. Reading the cut instead finds no call, which is what the client asked for.
    """
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"),
        "Paris is nice" + QWEN_CALL,
        finish_reason="length",
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-stop-before-call",
                sampling_params=SamplingParams(max_tokens=8, stop=["<tool_call>"]),
                metadata={
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [QWEN_TOOL],
                    "thinking_mode": "chat",
                },
            )
        ])[0]
    finally:
        backend.close()

    assert result.text == "Paris is nice"
    assert result.finish_reason == "stop"
    assert "tool_calls" not in result.metadata


def test_a_stream_ends_at_a_stop_sequence() -> None:
    """A stop sequence ends the stream, rather than the budget ending it three tokens later.

    The scheduler still runs to `max_tokens` -- it has no way to see a text-level sequence -- so the
    run is abandoned here instead. The tokens already sent are the answer up to the sequence, and
    `usage` counts the steps that produced them rather than the budget.
    """
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=FakeEngine(),
        tokenizer=RunningTextTokenizer(["ans", "answer", "answer USER:", "answer USER: x"]),
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-stop-stream",
        sampling_params=SamplingParams(max_tokens=8, stop=["USER:"]),
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["ans", "wer", " "]
    assert [event.finish_reason for event in events] == [None, None, "stop"]
    assert "".join(event.text for event in events) == "answer "
    assert events[-1].usage is not None
    assert events[-1].usage.completion_tokens == 3
    # The stop token is the one the sequence was found in, and nothing was decoded after it.
    assert backend._engine.calls == [
        ("prefill", [1]),
        ("decode_step", 10),
        ("decode_step", 11),
    ]
    backend.close()


def test_a_stream_withholds_a_tail_that_could_become_a_stop_sequence() -> None:
    """A prefix of a sequence waits for the token that decides it, then goes out whole.

    A stream cannot take a character back, so "U" is not "U" until the next token says whether it
    was the start of "USER:". The withheld tail is not lost: this one turns out not to be a
    sequence at all, and it is sent with the token that settles it.
    """
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=FakeEngine(),
        tokenizer=RunningTextTokenizer(["hello U", "hello US", "hello USR"]),
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-stop-prefix",
        sampling_params=SamplingParams(max_tokens=3, stop=["USER:"]),
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["hello ", "", "USR"]
    assert [event.finish_reason for event in events] == [None, None, "length"]
    assert "".join(event.text for event in events) == "hello USR"
    backend.close()


def test_more_than_one_choice_is_not_this_adapters_to_refuse(tmp_path) -> None:
    """`n` is served, and it is served one level up.

    A request for four choices is four requests to this adapter, which the host's dispatch builds;
    the adapter's part of serving the field is to answer each of them, so an audit that refused `n`
    here would refuse it on every runtime and the fan-out could never happen.
    """
    backend = _scripted_backend(_declaring_checkpoint(tmp_path, "qwen3_5"), "x")
    try:
        assert backend.audit_request({"n": 4}) is None
        assert backend.audit_request({"n": 1}) is None
        assert backend.audit_request({}) is None
    finally:
        backend.close()


def test_several_choices_are_refused_where_they_could_not_differ(tmp_path) -> None:
    """The one `n` this server still will not serve, and it is about the seeds rather than the count.

    The choices of one request differ only in the seed the fan-out hands each of them, and an engine
    that samples at engine-wide values reads no seed it was given. Four runs would then be four
    copies of one answer presented as independent samples -- which the caller has no way to tell
    from the field being honoured. Under greedy decoding the same text `n` times is what was asked
    for, so only the stochastic case is refused; that is the line the native front end drew too.
    """
    backend = _scripted_backend(_declaring_checkpoint(tmp_path, "qwen3_5"), "x")
    # A scheduler that declares the engine's sampler, which is the engine-wide part of the answer.
    backend._scheduler.engine_caps = _engine_sampling(fixed_temperature=0.7)
    try:
        refusal = backend.audit_request({"n": 3})
        assert refusal is not None
        assert refusal.field == "n"
        assert refusal.requested == "3"
        assert "0.7" in refusal.message
        # One choice is what a request asking for one gets, and it is not this field's business.
        assert backend.audit_request({"n": 1}) is None
        # Where the engine does sample per request the seeds differ, so every count is served.
        backend._scheduler.engine_caps = _engine_sampling(per_request_sampling=True)
        assert backend.audit_request({"n": 3}) is None
    finally:
        backend.close()


def test_a_requested_ranking_comes_back_as_openai_logprobs() -> None:
    """The engine's per-token ranking, rendered into the shape a caller asked for.

    Three things have to hold at once and each is checked here: the array covers exactly the tokens
    the answer is made of, in order, so it rejoins into the text beside it; each position reports
    the probability of the token the engine generated there; and the ranked alternatives are the
    ones the request asked for, narrowed from whatever width the batch ranked at.
    """
    backend = _ranking_backend(
        {1: "1", 2: ", ", 3: "2"},
        [1, 2, 3],
        [
            ScriptedLogprob(-0.11, top_tokens=[1, 2], top_logprobs=[-0.11, -1.5]),
            ScriptedLogprob(-0.22, top_tokens=[2, 3], top_logprobs=[-0.22, -2.5]),
            ScriptedLogprob(-0.33, top_tokens=[3, 1], top_logprobs=[-0.33, -3.5]),
        ],
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-logprobs",
                sampling_params=SamplingParams(logprobs=True, top_logprobs=2, max_tokens=3),
            )
        ])[0]
        # The request side: without this the engine ranks nothing and the array comes back empty,
        # so a rendering test that skipped it would pass on a build that never asked.
        assert backend._scheduler.submitted[0].logprobs_n == 2
    finally:
        backend.close()

    assert result.text == "1, 2"
    content = result.logprobs["content"]
    assert [entry["token"] for entry in content] == ["1", ", ", "2"]
    assert "".join(entry["token"] for entry in content) == result.text
    assert [entry["logprob"] for entry in content] == [-0.11, -0.22, -0.33]
    # The bytes are what says where one token ends and the next begins, which the text alone does
    # not: a caller reassembling an answer by token boundary needs them.
    assert [entry["bytes"] for entry in content] == [[0x31], [0x2C, 0x20], [0x32]]
    assert [alt["token"] for alt in content[0]["top_logprobs"]] == ["1", ", "]
    assert [alt["logprob"] for alt in content[0]["top_logprobs"]] == [-0.11, -1.5]


def test_a_ranking_covers_the_answer_and_not_the_reasoning_in_front_of_it() -> None:
    """Chat answers are a suffix of the decode, so a ranking indexed from zero describes the wrong text.

    A thinking model decodes its reasoning first. Reporting a probability for those positions beside
    an answer that does not contain them gives a caller an array it cannot line up against the
    content -- and one that is longer than the answer, which is how a client would notice.
    """
    backend = _ranking_backend(
        {1: "weighing it up", 2: "</think>", 3: "the answer"},
        [1, 2, 3],
        [ScriptedLogprob(-9.0), ScriptedLogprob(-9.1), ScriptedLogprob(-0.5)],
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-logprobs-chat",
                sampling_params=SamplingParams(logprobs=True, max_tokens=3),
                metadata={
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking_mode": "chat",
                },
            )
        ])[0]
    finally:
        backend.close()

    assert result.text == "the answer"
    assert result.metadata["reasoning_content"] == "weighing it up"
    assert [entry["token"] for entry in result.logprobs["content"]] == ["the answer"]
    assert [entry["logprob"] for entry in result.logprobs["content"]] == [-0.5]


def test_a_ranking_stops_where_a_stop_sequence_ended_the_text() -> None:
    """A position outside the answer describes text the caller never received.

    The stop sequence is matched on decoded text and lands after the answer, so the tokens from it
    onwards have probabilities that belong to no part of what was returned. The array is cut at the
    same byte the text was.
    """
    backend = _ranking_backend(
        {1: "1", 2: ", ", 3: "2", 4: "END", 5: "!"},
        [1, 2, 3, 4, 5],
        [ScriptedLogprob(-0.1) for _ in range(5)],
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-logprobs-stop",
                sampling_params=SamplingParams(logprobs=True, stop=["END"], max_tokens=5),
            )
        ])[0]
    finally:
        backend.close()

    assert result.text == "1, 2"
    assert result.finish_reason == "stop"
    assert [entry["token"] for entry in result.logprobs["content"]] == ["1", ", ", "2"]


def test_a_position_the_engine_never_ranked_fails_the_request() -> None:
    """An array shorter than the text is the one outcome not allowed.

    A caller reading a probability per token has no way to notice that the array describes fewer
    positions than the answer covers, so the native front end failed the choice and so does this.
    Speculative decoding is where it happens: the engine describes the first token a row emitted,
    and a row that emitted several leaves the rest absent.
    """
    backend = _ranking_backend(
        {1: "a", 2: "b"},
        [1, 2],
        [ScriptedLogprob(-0.1), ScriptedLogprob(0.0, present=False)],
    )
    try:
        with pytest.raises(RuntimeError, match="no log probabilities for a position"):
            backend.generate([
                GenerationRequest(
                    prompt_tokens=[1],
                    request_id="req-logprobs-short",
                    sampling_params=SamplingParams(logprobs=True, max_tokens=2),
                )
            ])
    finally:
        backend.close()


def test_logprobs_is_refused_where_there_is_no_scheduler_to_rank() -> None:
    """One field, one answer per build: no scheduler, no ranking, and a refusal rather than an empty array.

    ``batching=false`` selects the serialized compatibility session, whose ``generate`` reports
    tokens and nothing about their probabilities. Answering the field with an empty array would be
    the caller's own ranking silently replaced by nothing.
    """
    backend = _ranking_backend({1: "a"}, [1], [])
    backend._batching_enabled = False
    try:
        refusal = backend.audit_request({"logprobs": True})
        assert refusal is not None
        assert refusal.field == "logprobs"
        assert "per-token log probabilities" in refusal.message
        assert backend.audit_request({"logprobs": False}) is None
    finally:
        backend.close()


def test_logprobs_is_refused_where_the_engine_does_not_rank() -> None:
    """A scheduler is necessary and not sufficient: the ranking is produced by the engine's sampler.

    The Ascend build is the case. It has no per-row device sampler, so its ``caps()`` reports
    ``logprobs`` false and its results carry no ranking -- and asking the scheduler alone declared
    the field served anyway, admitted the request, and answered it with a 500 that said the ranking
    described fewer tokens than the text. The refusal belongs at the audit, where every other
    undeclared capability is refused by name.
    """
    backend = _ranking_backend({1: "a"}, [1], [])
    backend._scheduler.logprobs = False
    try:
        refusal = backend.audit_request({"logprobs": True})
        assert refusal is not None
        assert refusal.field == "logprobs"
        assert "per-token log probabilities" in refusal.message
        assert backend.audit_request({"logprobs": False}) is None
    finally:
        backend.close()


def test_a_seed_the_caller_named_reaches_the_sampler() -> None:
    """Until now it did not, which made the fan-out unable to vary anything.

    The choices of an `n`-choice request differ only in the seed the host hands each of them, so a
    field this adapter never assigned turned several choices into one answer written out several
    times -- and silently ignored a seed on a request that asked for one choice.
    """
    backend = _ranking_backend({1: "a"}, [1], [])
    backend._scheduler.engine_caps = _engine_sampling(per_request_sampling=True)
    try:
        backend.generate([
            GenerationRequest(
                prompt_tokens=[1],
                request_id="req-seed",
                sampling_params=SamplingParams(seed=7, temperature=0.7, max_tokens=1),
            )
        ])
        assert backend._scheduler.submitted[0].seed == 7
    finally:
        backend.close()


def test_a_negative_seed_is_masked_into_the_engines_unsigned_field() -> None:
    """The engine's seed is a ``uint64_t`` and a Python int is not, so the wrap has to be explicit.

    A negative seed that reached the binding unmasked would raise there, turning a request the field
    contract accepted into a 500 rather than a run: the shape check has no reason to refuse a
    negative seed, and the native front end reinterpreted one the same way this does.
    """
    backend = _ranking_backend({1: "a"}, [1], [])
    backend._scheduler.engine_caps = _engine_sampling(per_request_sampling=True)
    try:
        backend.generate([
            GenerationRequest(
                prompt_tokens=[1],
                request_id="req-seed-negative",
                sampling_params=SamplingParams(seed=-1, temperature=0.7, max_tokens=1),
            )
        ])
        assert backend._scheduler.submitted[0].seed == 0xFFFFFFFFFFFFFFFF
    finally:
        backend.close()


def test_the_typed_surface_is_audited_by_the_same_contract(tmp_path) -> None:
    """`LLM.chat` never produces a body, and a field set on it is more visible than a JSON key.

    One policy, two spellings: the typed request is mapped back onto the body keys the audit reads,
    so the library surface cannot disagree with the HTTP one about what this backend serves.

    `n` and `logprobs` are absent from the refused list because this backend serves both now -- the
    first through the host's fan-out, the second through the scheduler -- and what holds their
    behaviour is `tests/test_choices.py` and the ranking tests below rather than a refusal.
    """
    backend = _scripted_backend(_declaring_checkpoint(tmp_path, "qwen3_5"), "x")
    try:
        for params, expected in (
            (SamplingParams(min_p=0.1), '"min_p" = 0.1'),
            (SamplingParams(repetition_penalty=1.2), '"repetition_penalty" = 1.2'),
            (SamplingParams(temperature=0.7), '"temperature" = 0.7'),
        ):
            request = GenerationRequest(
                prompt_tokens=[1], request_id="req-typed", sampling_params=params
            )
            with pytest.raises(UnsupportedFeatureError, match=expected):
                backend.generate([request])
        # The defaults and the stop sequences this backend does serve are not refused.
        backend.generate([
            GenerationRequest(
                prompt_tokens=[1],
                request_id="req-typed-ok",
                sampling_params=SamplingParams(
                    max_tokens=1, stop=["USER:"], top_p=1.0, n=1, logprobs=False
                ),
            )
        ])
    finally:
        backend.close()


def test_add_generation_prompt_false_reaches_the_prompt_builder() -> None:
    """The native front end's field, which had no way through this adapter before.

    A request that asked to encode the conversation as it stands -- no assistant header after the
    last message -- was answered as if it had asked for the header, because the builder was called
    with the template's own default and nothing could override it.
    """
    tokenizer = TemplateTokenizer()
    backend, _ = make_backend()
    backend._tokenizer = tokenizer
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-no-assistant-header",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": False,
        },
    )

    assert backend._prompt_ids(request) == [71, 72]
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is False
    backend.close()


def test_a_stop_sequence_inside_the_reasoning_block_does_not_end_the_answer(tmp_path) -> None:
    """`stop` ends the answer, and on a chat request the reasoning block is not part of it.

    The sequences clients actually use are the ones this matters for: `"\\n\\n"` is common and a
    reasoning block is full of blank lines, so matching the whole decode would end the answer before
    the model had written any of it -- and the client would read that as an empty answer rather than
    as a truncated one. The block is a separate field whose text is not the completion.
    """
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"),
        "weighing USER: it up</think>the answer USER: and more",
        finish_reason="length",
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-stop-reasoning",
                sampling_params=SamplingParams(max_tokens=8, stop=["USER:"]),
                metadata={
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking_mode": "thinking",
                },
            )
        ])[0]
    finally:
        backend.close()

    assert result.metadata["reasoning_content"] == "weighing USER: it up"
    assert result.text == "the answer "
    assert result.finish_reason == "stop"


def test_a_stop_sequence_inside_a_call_leaves_no_call_behind(tmp_path) -> None:
    """A call the client asked to stop short of is not a call it should act on.

    The parse is all-or-nothing, so handing it the truncated text is what makes "a reported call is
    complete" hold: half a call read as a whole one would have a client run a function with
    arguments the model never finished writing.
    """
    backend = _scripted_backend(
        _declaring_checkpoint(tmp_path, "qwen3_5"),
        QWEN_CALL + "trailing",
        finish_reason="length",
    )
    try:
        result = backend.generate([
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id="req-stop-inside-call",
                sampling_params=SamplingParams(max_tokens=8, stop=["</parameter>"]),
                metadata={
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [QWEN_TOOL],
                    "thinking_mode": "chat",
                },
            )
        ])[0]
    finally:
        backend.close()

    assert "tool_calls" not in result.metadata
    assert result.finish_reason == "stop"
    assert "</parameter>" not in result.text
    assert result.text.startswith("<tool_call>")


def test_a_stream_does_not_end_on_a_sequence_inside_the_reasoning_block() -> None:
    """The streaming reading of the same rule: the block's text is not the completion.

    A client watching `reasoning_content` sees a sequence there in full and the answer still
    arrives, which is the one place the two fields are visibly independent rather than two views of
    one string.
    """
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        engine=FakeEngine(),
        tokenizer=RunningTextTokenizer(
            [
                "USER: thinking",
                "USER: thinking</think>",
                "USER: thinking</think>the answer",
                "USER: thinking</think>the answer USER:",
            ]
        ),
    )
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-stream-reasoning-stop",
        sampling_params=SamplingParams(max_tokens=4, stop=["USER:"]),
        metadata={
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_mode": "thinking",
        },
    )

    events = list(backend.stream(request))

    assert [event.text for event in events] == ["", "", "the answer", " "]
    assert [event.finish_reason for event in events] == [None, None, None, "stop"]
    assert "".join(event.text for event in events) == "the answer "
    assert [event.metadata.get("reasoning_content") for event in events][0] == "USER: thinking"
    backend.close()


# --------------------------------------------------------------------------------------------------
# Structured outputs: the token constraint the native front end built and this one did not
# --------------------------------------------------------------------------------------------------


def _structured_backend(checkpoint: str, *, structured_outputs: bool = True):
    """A batch-path backend whose engine declares what it can do about a schema.

    The vocabulary is a real directory here rather than a stub, because *which* path the native
    `Tokenizer` is handed is one of the two things this adapter decides: a directory holding a GGUF
    is not a directory holding `tokenizer.json`, and the C++ reader takes whichever one the engine
    itself would have opened.
    """
    backend = CppBackend(
        EngineArgs(model=checkpoint, backend="cpp"),
        engine=FakeEngine(),
        tokenizer=FakeTokenizer(),
    )
    backend._native = ScriptedNativeModule()
    backend._scheduler = ScriptedScheduler(
        ScriptedNativeResult(tokens=[1, 2, 3], finish_reason="stop"),
        structured_outputs=structured_outputs,
    )
    backend._batching_enabled = True
    return backend


def _constrained(backend, response_format, *, request_id: str = "req-schema"):
    return backend.generate(
        [
            GenerationRequest(
                prompt_tokens=[1, 2],
                request_id=request_id,
                sampling_params=SamplingParams(max_tokens=4, response_format=response_format),
            )
        ]
    )


CONSTRAINED_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def test_a_schema_selects_the_schema_factory_over_the_checkpoints_own_vocabulary(tmp_path) -> None:
    """The field reaches the scheduler as a token constraint, which is the only way it is applied.

    Both halves matter and neither is guessable from the other: the *schema* factory is what holds
    the answer to the schema where the json-object factory would accept any JSON, and the vocabulary
    has to be the checkpoint's own because the mask is indexed by the engine's token ids. A run that
    picked the wrong factory would answer a valid JSON object the client cannot use, and one that
    masked over a different vocabulary would refuse the tokens it meant to allow.
    """
    backend = _structured_backend(str(tmp_path))
    native, scheduler = backend._native, backend._scheduler
    try:
        _constrained(
            backend,
            {"type": "json_schema", "json_schema": {"name": "City", "schema": CONSTRAINED_SCHEMA}},
        )
    finally:
        backend.close()

    assert native.tokenizer_paths == [str(tmp_path)]
    assert [call[0] for call in native.schema_calls] == [f"tokenizer:{tmp_path}"]
    assert json.loads(native.schema_calls[0][1]) == CONSTRAINED_SCHEMA
    assert native.object_calls == []
    (constraint,) = scheduler.constraints
    assert constraint.kind == "schema"


def test_a_json_object_response_format_needs_no_schema_and_gets_its_own_factory(tmp_path) -> None:
    """`{"type": "json_object"}` is a shape the client asked for, not a schema it supplied.

    The engine derives the grammar for "any object" itself, so the adapter's whole part is picking
    the factory -- and picking the schema one here would leave the request waiting for a schema that
    was never sent.
    """
    backend = _structured_backend(str(tmp_path))
    native, scheduler = backend._native, backend._scheduler
    try:
        _constrained(backend, {"type": "json_object"})
    finally:
        backend.close()

    assert native.schema_calls == []
    assert native.object_calls == [f"tokenizer:{tmp_path}"]
    assert scheduler.constraints[0].kind == "object"


def test_text_is_the_shape_that_asks_for_nothing(tmp_path) -> None:
    """`{"type": "text"}` is what an OpenAI client sends by default.

    Constraining it would generate JSON for a caller who asked for prose, and refusing it would
    break every client that spells the default out -- which is most of them.
    """
    backend = _structured_backend(str(tmp_path))
    native, scheduler = backend._native, backend._scheduler
    try:
        _constrained(backend, {"type": "text"})
    finally:
        backend.close()

    assert scheduler.constraints == [None]
    assert native.tokenizer_paths == []


def test_a_request_that_names_no_response_format_builds_no_constraint(tmp_path) -> None:
    """The common case, and the one the vocabulary load must not be paid for."""
    backend = _structured_backend(str(tmp_path))
    native, scheduler = backend._native, backend._scheduler
    try:
        backend.generate(
            [GenerationRequest(prompt_tokens=[1, 2], request_id="req-plain",
                               sampling_params=SamplingParams(max_tokens=4))]
        )
    finally:
        backend.close()

    assert scheduler.constraints == [None]
    assert native.tokenizer_paths == []


def test_the_vocabulary_is_read_once_however_many_requests_ask_for_a_schema(tmp_path) -> None:
    """Megabytes, and a decode step never touches them: read on first use and kept.

    The cost of getting this wrong is not correctness, which is why it is worth a test of its own:
    it is a per-request read of a vocabulary file that never changes.
    """
    backend = _structured_backend(str(tmp_path))
    native = backend._native
    try:
        _constrained(backend, {"type": "json_object"}, request_id="req-1")
        _constrained(backend, {"type": "json_object"}, request_id="req-2")
    finally:
        backend.close()

    assert native.tokenizer_paths == [str(tmp_path)]
    assert len(native.object_calls) == 2
    # Both requests get their own: a constraint carries the state of the answer so far, and two
    # requests sharing one would accept each other's tokens.
    assert native.object_calls[0] is native.object_calls[1]


def test_a_directory_holding_a_gguf_is_read_through_the_file_it_holds(tmp_path, monkeypatch) -> None:
    """The container decides where the vocabulary lives, and only one of them is a directory read.

    The released ternary artifact is a bare GGUF, so `Tokenizer(<dir>)` would fail looking for a
    `tokenizer.json` that does not exist -- while the engine, opening the same path, reads the
    header happily. Naming the file the engine would open is what keeps the two agreeing.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    bundle = ckpt / "model.gguf"
    bundle.write_bytes(b"GGUF")
    monkeypatch.setattr("pocketllm.backends.cpp_backend.gguf_checkpoint_file", lambda path: str(bundle))

    backend = _structured_backend(str(ckpt))
    native = backend._native
    try:
        _constrained(backend, {"type": "json_object"})
    finally:
        backend.close()

    assert native.tokenizer_paths == [str(bundle)]


def test_an_engine_that_applies_no_mask_refuses_the_field_rather_than_ignoring_it(tmp_path) -> None:
    """The constraint lives in the per-row device sampler, so engine-wide sampling has no mask.

    Under tensor parallelism every rank has to enter the same sampler collectives, so an engine that
    samples engine-wide applies no per-row mask -- and answering a schema there would return
    unconstrained text with a 200, which is the silent drop this contract exists to prevent.
    """
    backend = _structured_backend(str(tmp_path), structured_outputs=False)
    scheduler = backend._scheduler
    try:
        with pytest.raises(UnsupportedFeatureError, match="constrained decoding"):
            _constrained(backend, {"type": "json_object"})
    finally:
        backend.close()

    assert scheduler.constraints == []


def test_the_serial_path_refuses_a_schema_because_nothing_carries_a_constraint(tmp_path) -> None:
    """`batching=false` selects the serialized session, which has no scheduler and no mask.

    The same rule as `logprobs`, and refused for the same reason: the field would be answered by
    text that does not apply it.
    """
    backend = _structured_backend(str(tmp_path))
    backend._scheduler = None
    backend._batching_enabled = False
    try:
        with pytest.raises(UnsupportedFeatureError, match="constrained decoding"):
            _constrained(backend, {"type": "json_object"})
    finally:
        backend.close()


def test_a_response_format_that_is_not_one_is_refused_by_shape(tmp_path) -> None:
    """Well formed JSON, and not one of the three shapes -- so the refusal names the field.

    A client sending `{"type": "json_schema"}` without the schema has made a mistake the engine
    cannot act on, and the answer has to be the one that says which key is missing rather than a
    generation that runs unconstrained.
    """
    backend = _structured_backend(str(tmp_path))
    scheduler = backend._scheduler
    try:
        with pytest.raises(UnsupportedFeatureError, match="json_schema.schema"):
            _constrained(backend, {"type": "json_schema", "json_schema": {"name": "City"}})
        with pytest.raises(UnsupportedFeatureError, match="must be 'text'"):
            _constrained(backend, {"type": "csv"})
    finally:
        backend.close()

    assert scheduler.constraints == []


def test_the_host_refuses_a_schema_before_the_backend_ever_sees_it(tmp_path) -> None:
    """The two halves of the audit, in the order the server runs them: shape, then capability.

    A malformed value is refused by the *shape* pass, which knows nothing about the runtime, so the
    same refusal is what every runtime answers. That separation is what stops a runtime from being
    blamed for a client's typo.
    """
    from pocketllm.protocol.contract import audit_shape

    refusal = audit_shape({"response_format": {"type": "json_schema", "json_schema": {}}})
    assert refusal is not None
    assert refusal.field == "response_format"
    assert "response_format" in refusal.message

    assert audit_shape({"response_format": {"type": "text"}}) is None
    assert audit_shape({}) is None
