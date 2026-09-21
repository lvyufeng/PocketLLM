"""The V4.1 adapter joins the factory's supervised tensor-parallel path.

The adapter loads its checkpoint from rank 0's own process, so the supervisor cannot
hand rank 0 a command the way it hands one to ranks 1..N-1: rank 0 is built here, and
the rendezvous it needs -- the file path for the NCCL id and the TCP address for the
process group -- has to be in *this* process's environment for exactly as long as the
load takes.  These tests pin that contract, and pin that it is not left behind
afterwards, which is what makes a second engine in the same process hang.
"""

from __future__ import annotations

import json
import os

import pytest

from pocketllm.api import EngineArgs
from pocketllm.backends import factory
from pocketllm.backends.base import BackendBase


class FakeSupervisor:
    """Stands in for TensorParallelSupervisor; records the spawn and the teardown."""

    instances: list["FakeSupervisor"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.started = 0
        self.cleaned = 0
        self.master_port = 29517
        self.nccl_id_path = f"/tmp/fake-v41-nccl-id-{len(FakeSupervisor.instances)}"
        FakeSupervisor.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def cleanup(self) -> None:
        self.cleaned += 1


class FakeV41Backend(BackendBase):
    """Captures what create_backend resolved for rank 0, at load time.

    Extends the real base class because the teardown under test -- releasing the ranks
    this construction spawned -- lives there, and a stand-in with its own ``close`` would
    test nothing.
    """

    def __init__(self, args, **_kwargs) -> None:
        super().__init__()
        self.args = args
        self.prepared = 0
        self.seen: dict[str, str | None] = {}

    def prepare(self) -> None:
        # The rendezvous is a construction-time window, so this is where it has to be
        # readable -- not before, and not after.
        self.prepared += 1
        self.seen = {
            name: os.environ.get(name)
            for name in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "POCKETLLM_NCCL_ID_PATH",
            )
        }


@pytest.fixture
def supervised(monkeypatch):
    """Route create_backend at fakes along the V4.1 path."""
    FakeSupervisor.instances = []

    import pocketllm.supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "TensorParallelSupervisor", FakeSupervisor)
    monkeypatch.setattr(factory, "V41Backend", FakeV41Backend)
    monkeypatch.setattr(factory, "select_backend", lambda args: "v41")
    for name in ("POCKETLLM_NCCL_ID_PATH", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    return FakeSupervisor


def _args(**overrides) -> EngineArgs:
    base = dict(
        model="/nonexistent/checkpoint",
        backend="v41",
        tensor_parallel_size=4,
    )
    base.update(overrides)
    return EngineArgs(**base)


def test_rank_zero_is_supervised_with_a_v41_worker_command(supervised):
    backend = factory.create_backend(_args())

    assert len(supervised.instances) == 1
    config = supervised.instances[0].config
    assert config.world_size == 4
    assert config.child_ranks == (1, 2, 3)
    # The children run the V4.1 worker, not the native one: the two build different
    # adapters and a mismatch is a group that never forms.
    assert config.command[2] == factory._v41_worker_script()
    assert "V41Backend" in config.command[2]


def test_rank_zero_loads_inside_the_rendezvous_it_was_given(supervised):
    backend = factory.create_backend(_args())
    supervisor = supervised.instances[0]

    # Rank 0 is returned loaded: its adapter reads the process group at load time, so a
    # backend handed back unloaded would be looking for a rendezvous that has been
    # restored away.
    assert backend.prepared == 1
    assert backend.seen["POCKETLLM_NCCL_ID_PATH"] == supervisor.nccl_id_path
    assert backend.seen["MASTER_PORT"] == str(supervisor.master_port)
    assert backend.seen["MASTER_ADDR"] == "127.0.0.1"
    assert backend.seen["RANK"] == "0"
    assert backend.seen["WORLD_SIZE"] == "4"
    assert backend.seen["LOCAL_RANK"] == "0"
    assert backend.args.backend_options["nccl_id_path"] == supervisor.nccl_id_path


def test_rendezvous_environment_does_not_outlive_construction(supervised):
    factory.create_backend(_args())

    for name in ("POCKETLLM_NCCL_ID_PATH", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        assert name not in os.environ, f"{name} leaked out of create_backend"


def test_rank_zero_backend_releases_the_group_on_close(supervised):
    backend = factory.create_backend(_args())

    backend.close()
    assert supervised.instances[0].cleaned == 1

    # A second close must not tear the same group down twice.
    backend.close()
    assert supervised.instances[0].cleaned == 1


def test_single_rank_v41_is_not_supervised(supervised):
    backend = factory.create_backend(_args(tensor_parallel_size=1))

    assert supervised.instances == []
    assert backend.prepared == 0


def test_v41_worker_script_rebuilds_the_same_engine():
    script = factory._v41_worker_script()

    # Rank comes from the supervisor's assignment, matching the native worker.
    assert 'actual_rank = int(os.environ.get("TP_RANK", "0"))' in script
    assert 'backend="v41"' in script
    # Readiness is announced from the worker loop, after the group is joined and the
    # checkpoint is loaded -- not from a bare print at import time.
    assert "run_worker(" in script
    assert "on_ready=lambda: print(f\"POCKETLLM_RANK_READY rank={actual_rank}\"" in script


def test_v41_worker_env_carries_the_paths_rank_zero_resolved(tmp_path):
    """A rank that resolves its own tokenizer or config loads a different model."""
    args = _args(config_path=str(tmp_path / "config.json"), tokenizer_path=str(tmp_path / "tok"))
    env = factory._worker_env(args)

    assert env["POCKETLLM_CONFIG_PATH"] == str(tmp_path / "config.json")
    assert env["POCKETLLM_TOKENIZER_PATH"] == str(tmp_path / "tok")
    assert env["POCKETLLM_CHECKPOINT"] == "/nonexistent/checkpoint"
    assert env["POCKETLLM_TP_SIZE"] == "4"
    assert json.loads(env["POCKETLLM_WORKER_ARGS"])["model_format"] == args.model_format


def test_v41_worker_env_blanks_a_path_rank_zero_never_set():
    env = factory._worker_env(_args())

    assert env["POCKETLLM_CONFIG_PATH"] == ""
    assert env["POCKETLLM_TOKENIZER_PATH"] == ""
