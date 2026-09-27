"""One program runs every supervised worker rank; what differs is a registry entry.

The program was generated as source text once per runtime and the three copies were the same
forty lines apart from three fields. These tests are about the replacement: a constant program,
a registry, and the fields that are still genuinely per-runtime.

The construction is faked rather than really run -- a real one loads a checkpoint -- so what is
asserted is the *contract*: what the child rebuilds from its environment, and when it announces
itself. A test that loaded a model would be measuring something else.
"""

from __future__ import annotations

import os

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.backends import worker


class FakeBackend:
    """Records what the program built and where in the rank's life it was told it was up."""

    instances: list["FakeBackend"] = []

    def __init__(self, args) -> None:
        self.args = args
        self.on_ready = None
        FakeBackend.instances.append(self)
        EVENTS.append("constructed")

    def run_worker(self, on_ready=None) -> None:
        self.on_ready = on_ready
        EVENTS.append("loop")
        if on_ready is not None:
            on_ready()


#: What the rank did, in order. Readiness is a moment, not a flag: a rank that announced before
#: it loaded is the failure this exists to catch, and a boolean cannot tell the two apart.
EVENTS: list[str] = []


@pytest.fixture
def worker_env(monkeypatch):
    """The environment the supervisor gives a child, as `factory._worker_env` writes it."""
    FakeBackend.instances = []
    EVENTS.clear()
    monkeypatch.setattr(worker, "_construct", lambda spec, args: FakeBackend(args))
    monkeypatch.setattr(
        "builtins.print", lambda *a, **k: EVENTS.append("ready " + " ".join(map(str, a)))
    )
    values = {
        "POCKETLLM_WORKER_BACKEND": "mimo",
        "POCKETLLM_CHECKPOINT": "/nonexistent/checkpoint",
        "POCKETLLM_TP_SIZE": "4",
        "POCKETLLM_MAX_MODEL_LEN": "4096",
        "POCKETLLM_KV_CACHE_DTYPE": "auto",
        "POCKETLLM_NCCL_ID_PATH": "/tmp/nccl-id",
        "POCKETLLM_BACKEND_OPTIONS": '{"threads": 22}',
        "POCKETLLM_WORKER_ARGS": '{"enable_batching": true, "max_batch_size": 4}',
        "POCKETLLM_TOKENIZER_PATH": "/nonexistent/tokenizer",
        "TP_RANK": "2",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("POCKETLLM_CONFIG_PATH", raising=False)
    return values


def test_the_program_is_one_string_for_every_runtime():
    """Constant, because the name travels in the environment and always did.

    Asserted as an identity rather than as a shape: a program that grew a per-runtime branch
    would still be "a string", and that is exactly the regression these tests exist for.
    """
    assert worker.program() is worker.program()
    assert "pocketllm.backends.worker" in worker.program()


def test_every_registered_runtime_names_a_class_that_exists():
    """The entry point is a `module:Class` pair, and a typo in one is not a startup failure."""
    from importlib import import_module

    for name, spec in worker.WORKERS.items():
        module_name, _, class_name = spec.entry_point.partition(":")
        assert class_name, f"{name} has no class in {spec.entry_point!r}"
        assert hasattr(import_module(module_name), class_name), (
            f"{name} names {spec.entry_point!r}, which does not exist"
        )


def test_a_worker_rebuilds_the_args_rank_zero_resolved(worker_env):
    worker.main()

    args = FakeBackend.instances[0].args
    assert args.model == "/nonexistent/checkpoint"
    assert args.backend == "mimo"
    assert args.tensor_parallel_size == 4
    assert args.tensor_parallel_rank == 2
    assert args.max_model_len == 4096
    # The rendezvous is carried in backend options for the native adapter and in the environment
    # for the supervision check; a child that only had the environment would not be able to build
    # the engine, and one that only had the options would spawn a second group.
    assert args.backend_options["nccl_id_path"] == "/tmp/nccl-id"
    assert args.backend_options["threads"] == 22
    # The shared fields arrive as one object precisely so a rank cannot default them separately.
    assert args.enable_batching is True
    assert args.max_batch_size == 4
    assert args.tokenizer_path == "/nonexistent/tokenizer"
    assert args.config_path is None


def test_the_rank_comes_from_the_supervisor_and_not_from_the_group(worker_env, monkeypatch):
    """`RANK` is the group's, `TP_RANK` is what the supervisor assigned, and rank 0 is not a child.

    A child that derived its rank from the group rather than reading it would be one ahead of
    where the process group puts it, and the mismatch shows up as a hang inside a collective.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    worker.main()

    assert FakeBackend.instances[0].args.tensor_parallel_rank == 2


def test_a_runtime_that_loads_lazily_announces_from_the_load(worker_env):
    """mimo and v41 join the group and load inside `run_worker`, so "constructed" is not ready.

    The marker goes out from the load's closing barrier. Announcing at construction would tell
    rank 0 a rank is up while it is still inside a collective, which no supervisor recovers from.
    """
    worker.main()

    assert EVENTS == ["constructed", "loop", "ready POCKETLLM_RANK_READY rank=2"]
    assert FakeBackend.instances[0].on_ready is not None


def test_a_runtime_that_loads_eagerly_announces_before_the_loop(worker_env, monkeypatch):
    """The native engine loads in the constructor, so there is no later moment to announce."""
    monkeypatch.setenv("POCKETLLM_WORKER_BACKEND", "cpp")

    worker.main()

    assert EVENTS == ["constructed", "ready POCKETLLM_RANK_READY rank=2", "loop"]
    assert FakeBackend.instances[0].on_ready is None


def test_an_unknown_runtime_names_the_ones_that_exist(worker_env, monkeypatch):
    monkeypatch.setenv("POCKETLLM_WORKER_BACKEND", "not-a-runtime")

    with pytest.raises(ConfigurationError, match="known runtimes are"):
        worker.main()


def test_a_missing_rendezvous_says_it_is_not_a_hand_launch(worker_env, monkeypatch):
    """Nothing else in the child's environment distinguishes "started by the supervisor"."""
    monkeypatch.delenv("POCKETLLM_NCCL_ID_PATH")

    with pytest.raises(ConfigurationError, match="POCKETLLM_NCCL_ID_PATH"):
        worker.main()


def test_the_two_paths_are_read_for_every_runtime(worker_env, monkeypatch, tmp_path):
    """A runtime that reads neither is unaffected by the read; one that needs one does not
    need a per-runtime branch in the program to get it."""
    monkeypatch.setenv("POCKETLLM_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setenv("POCKETLLM_WORKER_BACKEND", "v41")

    worker.main()

    args = FakeBackend.instances[0].args
    assert args.config_path == str(tmp_path / "config.json")
    assert args.tokenizer_path == "/nonexistent/tokenizer"
