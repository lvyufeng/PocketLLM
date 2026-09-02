"""C++ backend tensor-parallel command fan-out and engine selection."""

from __future__ import annotations

import pytest

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    SamplingParams,
)
from pocketllm.backends.cpp_backend import CppBackend


class TpFakeResult:
    def __init__(self, top_token: int) -> None:
        self.top_token = top_token


class TpFakeEngine:
    """Records the order of local ops and broadcast commands.

    A worker rank has to enter every collective in the same order as rank 0, so
    the interleaving of ``worker_command_*`` and the local op is what these tests
    assert on.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.warmups = 0
        self.closed = 0

    def warmup_tp(self) -> None:
        self.warmups += 1
        self.calls.append(("warmup_tp",))

    def prefill(self, prompt_ids: list[int]) -> TpFakeResult:
        self.calls.append(("prefill", list(prompt_ids)))
        return TpFakeResult(10)

    def decode_step(self, token: int) -> TpFakeResult:
        self.calls.append(("decode_step", token))
        return TpFakeResult(token + 1)

    def generate(self, prompt_ids: list[int], max_tokens: int) -> list[TpFakeResult]:
        self.calls.append(("generate", list(prompt_ids), max_tokens))
        return [TpFakeResult(10 + index) for index in range(max_tokens)]

    def reset(self) -> None:
        self.calls.append(("reset",))

    def clear_prefix_cache(self) -> None:
        self.calls.append(("clear_prefix_cache",))

    def worker_command_prefill(self, token_ids: list[int]) -> None:
        self.calls.append(("cmd_prefill", list(token_ids)))

    def worker_command_decode(self, last_token: int) -> None:
        self.calls.append(("cmd_decode", last_token))

    def worker_command_reset(self) -> None:
        self.calls.append(("cmd_reset",))

    def worker_command_shutdown(self) -> None:
        self.calls.append(("cmd_shutdown",))

    def run_worker_loop(self) -> None:
        self.calls.append(("run_worker_loop",))

    def close(self) -> None:
        self.closed += 1


class TpFakeOptions:
    def __init__(self) -> None:
        self.tp_world = 1
        self.tp_rank = 0
        self.device = -1
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


class TpFakeNative:
    QwenEngineOptions = TpFakeOptions

    def __init__(self) -> None:
        self.qwen_options: TpFakeOptions | None = None
        self.persistent_calls = 0

    @staticmethod
    def parse_qwen_kv_cache_dtype(value: str) -> str:
        return f"native:{value}"

    def QwenEngine(
        self,
        checkpoint: str,
        options: TpFakeOptions,
        layer_count: int,
        max_context: int,
    ) -> TpFakeEngine:
        self.qwen_options = options
        return TpFakeEngine()

    # A DeepSeek-V4 engine that must not be reached by a default TP launch.
    ForwardSmokeOptions = TpFakeOptions

    def PersistentEngine(self, *args: object, **kwargs: object) -> TpFakeEngine:
        self.persistent_calls += 1
        return TpFakeEngine()


class SimpleTokenizer:
    def encode(self, text: str) -> list[int]:
        return [5, 6]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


def make_tp_backend(
    *,
    world: int = 4,
    rank: int = 0,
    eos: object | None = None,
    native: TpFakeNative | None = None,
) -> tuple[CppBackend, TpFakeNative]:
    module = native or TpFakeNative()
    options: dict[str, object] = {"nccl_id_path": "/tmp/pocketllm-test-nccl"}
    if eos is not None:
        options["eos_token_id"] = eos
    args = EngineArgs(
        model="model",
        backend="cpp",
        tensor_parallel_size=world,
        tensor_parallel_rank=rank,
        backend_options=options,
    )
    backend = CppBackend(args, native_module=module, tokenizer=SimpleTokenizer())
    return backend, module


def test_tp_default_engine_is_qwen_not_persistent() -> None:
    """TP must not fall back to the DeepSeek-V4 loader.

    That fallback made every Qwen TP launch fail on `embed.weight`.
    """
    backend, native = make_tp_backend()
    try:
        assert native.persistent_calls == 0
        assert native.qwen_options is not None
        assert native.qwen_options.tp_world == 4
    finally:
        backend.close()


def test_tp_engine_is_warmed_up_before_use() -> None:
    backend, _ = make_tp_backend()
    try:
        assert backend._engine.warmups == 1
    finally:
        backend.close()


def test_single_rank_skips_warmup() -> None:
    args = EngineArgs(model="model", backend="cpp")
    native = TpFakeNative()
    backend = CppBackend(args, native_module=native, tokenizer=SimpleTokenizer())
    try:
        assert backend._engine.warmups == 0
    finally:
        backend.close()


def test_tp_requires_nccl_id_path() -> None:
    args = EngineArgs(
        model="model",
        backend="cpp",
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
    )
    with pytest.raises(ConfigurationError, match="nccl_id_path"):
        CppBackend(args, native_module=TpFakeNative(), tokenizer=SimpleTokenizer())


def test_tp_ranks_claim_distinct_devices() -> None:
    for rank in range(4):
        backend, native = make_tp_backend(rank=rank)
        try:
            assert native.qwen_options is not None
            assert native.qwen_options.device == rank
        finally:
            backend.close()


def test_explicit_device_overrides_rank_offset() -> None:
    args = EngineArgs(
        model="model",
        backend="cpp",
        tensor_parallel_size=4,
        tensor_parallel_rank=2,
        device="cuda:7",
        backend_options={"nccl_id_path": "/tmp/pocketllm-test-nccl"},
    )
    native = TpFakeNative()
    backend = CppBackend(args, native_module=native, tokenizer=SimpleTokenizer())
    try:
        assert native.qwen_options is not None
        assert native.qwen_options.device == 7
    finally:
        backend.close()


def test_narrowed_visible_devices_keeps_device_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher that pins one GPU per rank already renumbered it to 0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    backend, native = make_tp_backend(rank=2)
    try:
        assert native.qwen_options is not None
        assert native.qwen_options.device == 0
    finally:
        backend.close()


def test_tp_generate_announces_every_step() -> None:
    backend, _ = make_tp_backend()
    engine = backend._engine
    request = GenerationRequest(
        prompt_tokens=[1, 2, 3],
        request_id="req-tp",
        sampling_params=SamplingParams(max_tokens=3),
    )
    try:
        backend.generate([request])
        steps = [call for call in engine.calls if call[0] != "warmup_tp"]
        assert steps[:2] == [("cmd_prefill", [1, 2, 3]), ("prefill", [1, 2, 3])]
        # Each command precedes its local op so all ranks stay in step.
        assert steps[2] == ("cmd_decode", 10)
        assert steps[3] == ("decode_step", 10)
        # Native generate() drives its own loop and cannot announce steps.
        assert not any(call[0] == "generate" for call in steps)
    finally:
        backend.close()


def test_tp_stream_announces_every_step() -> None:
    backend, _ = make_tp_backend()
    engine = backend._engine
    request = GenerationRequest(
        prompt_tokens=[4, 5],
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=2),
    )
    try:
        list(backend.stream(request))
        steps = [call for call in engine.calls if call[0] != "warmup_tp"]
        assert steps[0] == ("cmd_prefill", [4, 5])
        assert steps[1] == ("prefill", [4, 5])
        assert steps[2] == ("cmd_decode", 10)
        assert steps[3] == ("decode_step", 10)
    finally:
        backend.close()


def test_single_rank_still_uses_native_generate() -> None:
    args = EngineArgs(model="model", backend="cpp")
    native = TpFakeNative()
    backend = CppBackend(args, native_module=native, tokenizer=SimpleTokenizer())
    engine = backend._engine
    request = GenerationRequest(
        prompt_tokens=[1, 2],
        request_id="req-single",
        sampling_params=SamplingParams(max_tokens=2),
    )
    try:
        backend.generate([request])
        assert ("generate", [1, 2], 2) in engine.calls
        assert not any(call[0].startswith("cmd_") for call in engine.calls)
    finally:
        backend.close()


def test_rank_zero_shuts_workers_down_on_close() -> None:
    backend, _ = make_tp_backend()
    engine = backend._engine
    backend.close()
    assert ("cmd_shutdown",) in engine.calls
    assert engine.closed == 1


def test_worker_rank_does_not_broadcast_shutdown() -> None:
    backend, _ = make_tp_backend(rank=1)
    engine = backend._engine
    backend.close()
    assert not any(call[0] == "cmd_shutdown" for call in engine.calls)


def test_tp_stops_at_eos_without_extra_commands() -> None:
    backend, _ = make_tp_backend(eos=11)
    engine = backend._engine
    request = GenerationRequest(
        prompt_tokens=[1],
        request_id="req-eos",
        sampling_params=SamplingParams(max_tokens=5),
    )
    try:
        results = backend.generate([request])
        assert results[0].finish_reason == "stop"
        # prefill yields 10, decode yields 11 == EOS, so the loop must stop
        # there and not announce a further decode.
        assert [call for call in engine.calls if call[0] == "cmd_decode"] == [("cmd_decode", 10)]
    finally:
        backend.close()
