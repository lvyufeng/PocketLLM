"""What `--enable-batching` and `--max-batch-size` mean on the cpp backend.

The two flags are the same decision seen twice: a width above 1 is a request for a scheduler, and
the scheduler is what makes the width mean anything. Before this, batching existed only as
`--backend-option enable_batching=true` -- a spelling no operator types -- while `--max-batch-size`
was accepted by the CLI, put into `EngineArgs`, and read by nothing on this backend. A flag that is
parsed and ignored is the failure mode these tests are about, so each one names the observable
consequence rather than the field.

These are hermetic: the native module is a fake that records how it was constructed, so the width the
engine would have been given is asserted directly and no checkpoint, card or extension is needed.
The engine sizes its KV cache from that width, so it is not a cosmetic number.
"""

from __future__ import annotations

import json

import pytest

from typing import Any

from pocketllm.api import ConfigurationError, EngineArgs, GenerationRequest, UnsupportedFeatureError
from pocketllm.backends.cpp_backend import DEFAULT_BATCH_SLOTS, CppBackend
from pocketllm.cli import _args, build_parser


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text), 7]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class FakeEngine:
    def close(self) -> None:
        pass


class FakeScheduler:
    def __init__(self, engine: object, width: int) -> None:
        self.engine = engine
        self.width = width


class FakeNativeWithScheduler:
    """A build that links the scheduler, which is the one this backend now assumes."""

    QwenBatchScheduler = FakeScheduler

    def registered_architectures(self) -> list[str]:
        return ["qwen3_5"]


class FakeNativeWithoutScheduler:
    """A build that does not, e.g. an older extension or the Ascend one."""

    def registered_architectures(self) -> list[str]:
        return ["qwen3_5"]


class FakeNativeRejectingEngine:
    """A build that has the scheduler, over an engine it will not take.

    The real case is a DeepSeek-V4 checkpoint: `PersistentEngine` is not an
    `InferenceEngine` in the binding, so `QwenBatchScheduler(engine, width)` is a
    `TypeError` -- pybind rejects the argument before the constructor body runs.
    `FakeEngine` below stands in for it exactly as well as any other object that
    is not a scheduler-acceptable engine, which is the property that decides this.
    """

    def registered_architectures(self) -> list[str]:
        return ["deepseek_v4", "qwen3_5"]

    @staticmethod
    def QwenBatchScheduler(engine: object, width: int) -> object:
        raise TypeError(
            "__init__(): incompatible constructor arguments. The following argument types are "
            "supported:\n    1. pocketllm_cpp.QwenBatchScheduler(engine: "
            "pocketllm_cpp.InferenceEngine, max_batch_size: typing.SupportsInt)"
        )


class FakeNativeWithEngineClasses:
    """A build whose class hierarchy is the real one's, for the pre-load check.

    `PersistentEngine` is not an `InferenceEngine` in this module, which is true of the real
    binding: `PersistentEngine.__bases__` is `(pybind11_object,)` and `QwenEngine.__bases__` is
    `(InferenceEngine,)`. That difference is the whole of the pre-load check's input.
    """

    class InferenceEngine:  # noqa: D106 -- a stand-in for the pybind trampoline base
        pass

    class QwenEngine(InferenceEngine):
        pass

    class PersistentEngine:
        pass

    QwenBatchScheduler = FakeScheduler

    def registered_architectures(self) -> list[str]:
        return ["deepseek_v4", "qwen3_5"]

    @staticmethod
    def detect_architecture(checkpoint: str) -> str:
        return "deepseek_v4"


_CONSTRUCT = object()
"""Sentinel for `make_backend(engine=...)`: `None` is a real value and means "let the backend
construct its engine", which is the path the pre-load check lives on."""


def make_backend(*, native=None, engine=_CONSTRUCT, **engine_args) -> CppBackend:
    return CppBackend(
        EngineArgs(model="model", backend="cpp", **engine_args),
        native_module=native if native is not None else FakeNativeWithScheduler(),
        engine=FakeEngine() if engine is _CONSTRUCT else engine,
        tokenizer=FakeTokenizer(),
    )


def cli_args(argv: list[str]) -> EngineArgs:
    return _args(build_parser().parse_args(argv))


# --------------------------------------------------------------------------------------------------
# the CLI spelling
# --------------------------------------------------------------------------------------------------


def test_the_flag_is_absent_unless_it_is_given() -> None:
    """`None` and `False` have to be different: one is "nobody asked", the other is a refusal."""
    assert cli_args(["serve", "--model", "m"]).enable_batching is None
    assert cli_args(["serve", "--model", "m", "--enable-batching"]).enable_batching is True
    assert cli_args(["serve", "--model", "m", "--no-enable-batching"]).enable_batching is False


def test_the_cli_width_reaches_the_engine_because_that_is_where_the_kv_cache_is_sized() -> None:
    """`--max-batch-size 4` has to arrive as 4 rows of KV, and the scheduler is built for the same
    number. The engine cannot grow its cache afterwards, so a width that stops at `EngineArgs` is a
    width the server will refuse at the first concurrent request."""
    args = cli_args(["serve", "--model", "m", "--max-batch-size", "4"])
    backend = make_backend(max_batch_size=args.max_batch_size, enable_batching=args.enable_batching)

    assert args.max_batch_size == 4
    assert backend._batching_enabled is True
    assert backend._scheduler.width == 4
    assert backend.capabilities.details["max_batch_size"] == 4


def test_a_width_without_the_flag_asks_for_batching_by_itself() -> None:
    """`--max-batch-size 4` alone has to do something; that is the whole complaint."""
    backend = make_backend(max_batch_size=4)

    assert backend._batching_enabled is True
    assert backend._scheduler.width == 4
    assert backend.capabilities.supports_batch is True


# --------------------------------------------------------------------------------------------------
# the default
# --------------------------------------------------------------------------------------------------


def test_batching_is_on_by_default_and_not_at_a_width_of_one() -> None:
    """A width of 1 is not a batch, so a default of 1 would make the default path serial."""
    backend = make_backend()

    assert backend._batching_enabled is True
    assert backend._scheduler.width == DEFAULT_BATCH_SLOTS
    assert backend.capabilities.supports_batch is True
    assert backend.capabilities.details["scheduler"] == "batch scheduler"
    assert backend.capabilities.details["max_batch_size"] == DEFAULT_BATCH_SLOTS


def test_no_enable_batching_selects_the_serialized_session() -> None:
    backend = make_backend(enable_batching=False)

    assert backend._batching_enabled is False
    assert backend._configured_max_batch_size() == 1
    assert backend.capabilities.supports_batch is False
    assert backend.capabilities.details["scheduler"] == "serialized compatibility session"


def test_a_width_next_to_no_enable_batching_is_refused_rather_than_resolved() -> None:
    """Two flags that contradict each other, and a third state is what this refactor is removing."""
    with pytest.raises(ConfigurationError, match="serialized session runs one request"):
        EngineArgs(model="model", backend="cpp", max_batch_size=8, enable_batching=False)


def test_the_scheduler_is_built_around_the_engine_that_was() -> None:
    engine = FakeEngine()
    backend = CppBackend(
        EngineArgs(model="model", backend="cpp"),
        native_module=FakeNativeWithScheduler(),
        engine=engine,
        tokenizer=FakeTokenizer(),
    )

    assert backend._scheduler.engine is engine


# --------------------------------------------------------------------------------------------------
# the backend option, which is the spelling the benchmarks use
# --------------------------------------------------------------------------------------------------


def test_the_backend_option_still_sets_the_width_on_its_own() -> None:
    """Every benchmark in `scripts/` passes the option and no CLI flag; reading the flag first would
    silently reset them to its default of 1."""
    backend = make_backend(backend_options={"enable_batching": True, "max_batch_size": 3})

    assert backend._batching_enabled is True
    assert backend._scheduler.width == 3


def test_the_backend_option_can_still_turn_batching_off() -> None:
    backend = make_backend(backend_options={"enable_batching": False})

    assert backend._batching_enabled is False
    assert backend._configured_max_batch_size() == 1


def test_the_backend_option_wins_over_the_flag_when_both_are_set() -> None:
    backend = make_backend(
        max_batch_size=4, backend_options={"max_batch_size": 2}
    )

    assert backend._scheduler.width == 2


def test_a_width_that_is_not_a_width_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="is not a width"):
        make_backend(backend_options={"max_batch_size": 0})


def test_a_width_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="is not an integer"):
        make_backend(backend_options={"max_batch_size": "wide"})


# --------------------------------------------------------------------------------------------------
# a scheduler that cannot be built
#
# Two ways for that to happen, and both are the same answer to a caller who asked for a width:
# refusing. A width reaches the engine through the scheduler's slot allocation, so "the scheduler
# could not be built" and "the width cannot be honoured" are one fact. Accepting the flag and
# serving the serialized session anyway is the third state these tests exist to remove.
# --------------------------------------------------------------------------------------------------


def test_a_build_without_a_scheduler_refuses_a_width_it_was_asked_for() -> None:
    with pytest.raises(UnsupportedFeatureError, match="does not expose QwenBatchScheduler"):
        make_backend(native=FakeNativeWithoutScheduler(), enable_batching=True)


def test_a_build_without_a_scheduler_refuses_a_width_as_a_width() -> None:
    """`--max-batch-size 8` is the spelling an operator types, and it is a request like any other.

    Before this it produced a warning and a serialized session, which is what #437 records as the
    third state: the flag was parsed, reached `EngineArgs`, and changed nothing.
    """
    with pytest.raises(UnsupportedFeatureError, match="from --max-batch-size 8"):
        make_backend(native=FakeNativeWithoutScheduler(), max_batch_size=8)


def test_the_refusal_names_the_flag_the_build_and_the_way_out() -> None:
    with pytest.raises(UnsupportedFeatureError) as caught:
        make_backend(native=FakeNativeWithoutScheduler(), max_batch_size=4)

    message = str(caught.value)
    assert "from --max-batch-size 4" in message
    assert "POCKET_BUILD_PYTHON" in message


def test_a_build_without_a_scheduler_is_quiet_when_nobody_asked() -> None:
    """The default is this backend's assumption about the build, and `details["scheduler"]` reports
    the answer on every construction. A warning per process for an assumption nobody made is noise;
    a refusal for an explicit request that cannot be honoured is the answer."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = make_backend(native=FakeNativeWithoutScheduler())

    assert backend._batching_enabled is False
    assert backend.capabilities.details["scheduler"] == "serialized compatibility session"


def test_an_engine_the_scheduler_rejects_refuses_a_width_it_was_asked_for() -> None:
    """The DeepSeek-V4 case, and the one #437 was opened for.

    `PersistentEngine` is not an `InferenceEngine` in the binding, so the scheduler's constructor
    rejects it. The capability declaration this clamps on was never reached: there is no scheduler
    to read it, because the engine is offered to the scheduler first.
    """
    with pytest.raises(UnsupportedFeatureError, match="could not be built over this engine"):
        make_backend(native=FakeNativeRejectingEngine(), max_batch_size=8)


def test_an_engine_the_scheduler_rejects_still_warns_when_nobody_asked() -> None:
    """Silence is only for the build fact. An engine that actively rejected the scheduler is new
    information -- no capability field says it -- so the default path reports it once."""
    with pytest.warns(UserWarning, match="could not be built over this engine"):
        backend = make_backend(native=FakeNativeRejectingEngine())

    assert backend._batching_enabled is False
    assert backend.capabilities.supports_batch is False


# --------------------------------------------------------------------------------------------------
# the same refusal, moved ahead of the model load
# --------------------------------------------------------------------------------------------------


def test_the_engine_class_the_checkpoint_selects_decides_the_pre_load_refusal() -> None:
    """Both directions, so the test fails if the predicate stops reading the hierarchy.

    A false negative here would refuse a build that works, which is the direction this must not err
    in; the `persistent` arm is the one #437 is about.
    """
    backend = make_backend(native=FakeNativeWithEngineClasses(), max_batch_size=2)

    backend._refuse_unschedulable_engine_before_loading("qwen")  # does not raise

    with pytest.raises(UnsupportedFeatureError, match="is not an InferenceEngine"):
        backend._refuse_unschedulable_engine_before_loading("persistent")


def test_a_deepseek_checkpoint_refuses_a_width_before_it_is_loaded() -> None:
    """The end-to-end shape of it: construction is refused, so no checkpoint was read.

    A real DeepSeek-V4 checkpoint takes minutes to open, and the whole reason this check exists
    ahead of `_init_batch_scheduler` is that learning it afterwards costs that load. `engine=None`
    is what makes the backend take the construction path at all.
    """
    with pytest.raises(UnsupportedFeatureError, match="is not an InferenceEngine"):
        make_backend(native=FakeNativeWithEngineClasses(), engine=None, max_batch_size=8)


def test_the_pre_load_check_is_silent_when_nobody_asked() -> None:
    """Nobody asked, so the load proceeds and the late check reports it once -- not twice.

    The early check is refusal-only by construction: it has nothing to say on the path that does
    not refuse, which is what keeps one unbuildable scheduler from producing two warnings.
    """
    import warnings

    backend = make_backend(native=FakeNativeWithEngineClasses())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend._refuse_unschedulable_engine_before_loading("persistent")


# --------------------------------------------------------------------------------------------------
# the other ranks
# --------------------------------------------------------------------------------------------------


def test_the_batch_decision_reaches_every_rank() -> None:
    """Only rank 0 ever runs a scheduler, but every rank *builds an engine*, and the batch decision
    is what that engine sizes its KV cache from -- `max_batch_size` at construction. A worker rank
    that resolved the default instead of the operator's `--no-enable-batching` would therefore
    allocate eight slots where rank 0 allocated one, under the same process group. The size of the
    engine is one of the fields `_WORKER_SHARED_ARGS` exists to keep in step; the scheduler flag
    looks per-rank and is not, which is why it belongs in the list too."""
    from pocketllm.backends import factory

    args = EngineArgs(model="model", backend="cpp", tensor_parallel_size=4, enable_batching=False)

    assert factory._worker_arg_overrides(args)["enable_batching"] is False
    env = factory._worker_env(args, "cpp")
    assert json.loads(env["POCKETLLM_WORKER_ARGS"])["enable_batching"] is False

    # And the worker that rebuilds from those overrides lands on the same width rank 0 did.
    worker_args = EngineArgs(
        model="model", backend="cpp", tensor_parallel_size=4, tensor_parallel_rank=1,
        **factory._worker_arg_overrides(args),
    )
    worker = CppBackend(
        worker_args,
        native_module=FakeNativeWithScheduler(),
        engine=FakeEngine(),
        tokenizer=FakeTokenizer(),
    )
    assert worker._configured_max_batch_size() == 1


# --------------------------------------------------------------------------------------------------
# what the scheduler reports about itself
# --------------------------------------------------------------------------------------------------


class FakeStats:
    """The fields `BatchScheduler::Stats` carries, and no more."""

    def __init__(self, **values: int) -> None:
        self.waiting_requests = values.get("waiting_requests", 0)
        self.running_requests = values.get("running_requests", 0)
        self.completed_requests = values.get("completed_requests", 0)
        self.cancelled_requests = values.get("cancelled_requests", 0)
        self.free_slots = values.get("free_slots", 0)
        self.reserved_blocks = values.get("reserved_blocks", 0)
        self.total_blocks = values.get("total_blocks", 0)
        self.free_blocks = values.get("free_blocks", 0)
        self.cache_pinned_blocks = values.get("cache_pinned_blocks", 0)


class FakeSchedulerWithStats(FakeScheduler):
    def __init__(self, engine: object, width: int, *, stats=None, paged_kv: bool = False) -> None:
        super().__init__(engine, width)
        self._stats = stats or FakeStats()
        self._paged_kv = paged_kv

    def get_stats(self) -> FakeStats:
        return self._stats

    def engine_caps(self) -> Any:
        class _Caps:
            paged_kv = self._paged_kv

        return _Caps()


class FakeNativeWithStats(FakeNativeWithScheduler):
    """A build whose scheduler can be asked what it is doing."""

    def __init__(self, **scheduler_kwargs: Any) -> None:
        self.QwenBatchScheduler = lambda engine, width: FakeSchedulerWithStats(
            engine, width, **scheduler_kwargs
        )


def test_the_python_server_publishes_the_schedulers_own_gauges() -> None:
    """`/metrics` is where "one scheduler, several requests" is observable, and it is the same
    reading on both hosts: the Python server prefixes what it exports, so the native host's
    `pocket_requests_running` is this server's `pocketllm_requests_running` and the suffixes match.
    A gauge named anything else would make the two hosts' numbers incomparable without a table."""
    backend = make_backend(
        native=FakeNativeWithStats(stats=FakeStats(running_requests=2, waiting_requests=1,
                                                   free_slots=6))
    )

    published = backend.metrics()

    assert published["requests_running"] == 2.0
    assert published["requests_waiting"] == 1.0
    assert published["slots_free"] == 6.0


def test_the_serialized_path_publishes_no_scheduler_gauges() -> None:
    """There is no scheduler on that path, so publishing zeros would say "the scheduler exists and
    is idle" about a process that does not have one -- which is worse than an absent series, because
    an absent series cannot be mistaken for a measurement."""
    assert make_backend(enable_batching=False).metrics() == {}


def test_the_block_gauges_appear_only_on_an_engine_that_pages() -> None:
    """An unpaged engine's block counts are zeros that a scraper reads as a pool of no blocks
    rather than as no pool, and `pocket_kv_blocks` is a family a paged deployment reads."""
    unpaged = make_backend(native=FakeNativeWithStats(stats=FakeStats(total_blocks=64)))
    paged = make_backend(
        native=FakeNativeWithStats(stats=FakeStats(total_blocks=64, free_blocks=41), paged_kv=True)
    )

    assert not [name for name in unpaged.metrics() if name.startswith("kv_blocks")]
    assert paged.metrics()['kv_blocks{state="total"}'] == 64.0
    assert paged.metrics()['kv_blocks{state="free"}'] == 41.0


def test_a_scheduler_that_cannot_report_does_not_fail_the_scrape() -> None:
    """A metrics scrape must not fail a request path or a scrape itself. An engine that cannot
    report is an engine whose series are absent, which a scraper reads as no data."""
    class _Broken(FakeScheduler):
        def get_stats(self) -> Any:
            raise RuntimeError("no stats")

    class _Native(FakeNativeWithScheduler):
        QwenBatchScheduler = _Broken

    assert make_backend(native=_Native()).metrics() == {}
