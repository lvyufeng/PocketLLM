"""Supervised TP setup must not leak rendezvous state between backends.

``create_backend`` spawns worker ranks and hands rank 0 the NCCL rendezvous path.
Anything it records process-wide survives the backend, so a second engine built
in the same process sees leftover state from the first.  These tests build two
backends in a row, which is what any benchmark comparing two configurations does.
"""

from __future__ import annotations

import os

import pytest

from pocketllm.api import EngineArgs
from pocketllm.backends import factory


class FakeSupervisor:
    """Stands in for TensorParallelSupervisor; records start/stop only."""

    instances: list["FakeSupervisor"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.started = 0
        self.stopped = 0
        self.nccl_id_path = f"/tmp/fake-nccl-id-{len(FakeSupervisor.instances)}"
        FakeSupervisor.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class FakeBackend:
    """Captures what create_backend resolved for this rank."""

    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.seen_nccl_id_path = args.backend_options.get("nccl_id_path")
        self.seen_env = os.environ.get("POCKETLLM_NCCL_ID_PATH")


@pytest.fixture
def supervised(monkeypatch):
    """Route create_backend at fakes and keep the env var change contained."""
    FakeSupervisor.instances = []

    import pocketllm.supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "TensorParallelSupervisor", FakeSupervisor)
    monkeypatch.setattr(factory, "CppBackend", FakeBackend)
    monkeypatch.setattr(factory, "select_backend", lambda args: "cpp")
    monkeypatch.delenv("POCKETLLM_NCCL_ID_PATH", raising=False)
    return FakeSupervisor


def _args(**overrides) -> EngineArgs:
    base = dict(
        model="/nonexistent/checkpoint",
        backend="cpp",
        tensor_parallel_size=4,
    )
    base.update(overrides)
    return EngineArgs(**base)


def test_second_backend_still_spawns_workers(supervised):
    """The env var must not make the second backend think it is supervised.

    A leaked POCKETLLM_NCCL_ID_PATH satisfies the ``not os.environ.get(...)``
    arm of the needs_supervision check, so the second engine skips spawning and
    then blocks in the NCCL rendezvous waiting for ranks nobody started.
    """
    first = factory.create_backend(_args())
    second = factory.create_backend(_args())

    assert len(supervised.instances) == 2, "second backend did not spawn workers"
    assert first.seen_nccl_id_path == supervised.instances[0].nccl_id_path
    assert second.seen_nccl_id_path == supervised.instances[1].nccl_id_path
    assert first.seen_nccl_id_path != second.seen_nccl_id_path


def test_env_restored_after_construction(supervised):
    """Rank 0 needs the path during construction, not afterwards."""
    assert "POCKETLLM_NCCL_ID_PATH" not in os.environ

    backend = factory.create_backend(_args())

    # Visible to the backend while it builds ...
    assert backend.seen_env == supervised.instances[0].nccl_id_path
    # ... and gone once it is built.
    assert "POCKETLLM_NCCL_ID_PATH" not in os.environ


def test_preexisting_env_value_is_preserved(supervised, monkeypatch):
    """An externally supervised path belongs to the caller; restore it verbatim."""
    monkeypatch.setenv("POCKETLLM_NCCL_ID_PATH", "/tmp/external-nccl-id")

    # A non-empty env var means external supervision, so no workers are spawned
    # and the caller's value must survive untouched.
    factory.create_backend(_args())

    assert os.environ["POCKETLLM_NCCL_ID_PATH"] == "/tmp/external-nccl-id"
    assert supervised.instances == []


def test_caller_engine_args_not_mutated(supervised):
    """backend_options is caller-owned and may be reused for another engine."""
    args = _args()
    factory.create_backend(args)

    assert "nccl_id_path" not in args.backend_options


def test_workers_stopped_when_rank0_fails(supervised, monkeypatch):
    """Workers are useless without rank 0 and would hold device memory."""

    class Boom(FakeBackend):
        def __init__(self, args, **kwargs) -> None:
            raise RuntimeError("rank 0 failed to allocate")

    monkeypatch.setattr(factory, "CppBackend", Boom)

    with pytest.raises(RuntimeError, match="rank 0 failed"):
        factory.create_backend(_args())

    assert len(supervised.instances) == 1
    assert supervised.instances[0].stopped == 1
    assert "POCKETLLM_NCCL_ID_PATH" not in os.environ
