"""Tests for local tensor-parallel process supervision."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from pocketllm.api import ConfigurationError
from pocketllm.supervisor import (
    TensorParallelConfig,
    TensorParallelSupervisor,
    TensorParallelSupervisorError,
)


_CHILD_READY = """
import os
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
import time
while True:
    time.sleep(0.1)
"""

_CHILD_EXIT = """
import os
import sys
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
sys.exit(0)
"""

_CHILD_FAIL = """
import os
import sys
rank = int(os.environ["RANK"])
print(f"child rank {rank} failing before readiness", file=sys.stderr, flush=True)
sys.exit(42)
"""

_CHILD_NO_READY = """
import time
time.sleep(600)
"""


def test_supervisor_config_validation() -> None:
    with pytest.raises(ConfigurationError, match="command"):
        TensorParallelConfig(command=(), world_size=2)
    with pytest.raises(ConfigurationError, match="world_size"):
        TensorParallelConfig(command=("echo",), world_size=1)
    with pytest.raises(ConfigurationError, match="startup_timeout"):
        TensorParallelConfig(command=("echo",), world_size=2, startup_timeout=0.0)
    with pytest.raises(ConfigurationError, match="shutdown_timeout"):
        TensorParallelConfig(command=("echo",), world_size=2, shutdown_timeout=-1.0)
    with pytest.raises(ConfigurationError, match="master_port"):
        TensorParallelConfig(command=("echo",), world_size=2, master_port=99999)
    with pytest.raises(ConfigurationError, match="rank_flag"):
        TensorParallelConfig(command=("echo",), world_size=2, rank_flag="")


def test_supervisor_requires_config_or_kwargs() -> None:
    config = TensorParallelConfig(command=("true",), world_size=2)
    TensorParallelSupervisor(config=config)
    TensorParallelSupervisor(command=("true",), world_size=2)
    with pytest.raises(TypeError, match="config"):
        TensorParallelSupervisor(config=config, command=("false",))
    with pytest.raises(TypeError, match="command"):
        TensorParallelSupervisor()


def test_supervisor_child_environment_and_command() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", "pass", "--tensor-parallel-rank", "99"],
        world_size=2,
        startup_timeout=1.0,
        shutdown_timeout=1.0,
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})

    env0 = supervisor._child_env(0)
    assert env0["RANK"] == "0"
    assert env0["LOCAL_RANK"] == "0"
    assert env0["WORLD_SIZE"] == "2"
    assert env0["TP_RANK"] == "0"
    assert env0["TP_WORLD"] == "2"
    assert env0["POCKETLLM_TP_SUPERVISED"] == "1"
    assert "MASTER_ADDR" in env0
    assert "MASTER_PORT" in env0
    assert "POCKETLLM_TP_RENDEZVOUS_DIR" in env0
    assert "POCKETLLM_NCCL_ID_PATH" in env0
    assert env0["NCCL_ID_PATH"] == env0["POCKETLLM_NCCL_ID_PATH"]

    env1 = supervisor._child_env(1)
    assert env1["RANK"] == "1"
    assert env1["TP_RANK"] == "1"
    assert env1["MASTER_PORT"] == env0["MASTER_PORT"]

    cmd = supervisor._rank_command(1)
    assert cmd == [sys.executable, "-c", "pass", "--tensor-parallel-rank", "1"]


def test_supervisor_child_command_with_builder() -> None:
    def builder(rank: int, env: dict[str, str]) -> list[str]:
        return [sys.executable, "-c", "pass", "--rank", str(rank), "--nccl", env["NCCL_ID_PATH"]]

    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", "pass"],
        world_size=2,
        command_builder=builder,
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})

    env = supervisor._child_env(0)
    cmd = supervisor._command_for_rank(0, env)
    assert cmd[4] == "0"
    assert cmd[6] == env["NCCL_ID_PATH"]


def test_supervisor_rank_flag_variants() -> None:
    supervisor = TensorParallelSupervisor(
        command=[
            "cmd",
            "a",
            "--tensor-parallel-rank",
            "99",
            "b",
            "--tensor-parallel-rank=88",
            "c",
        ],
        world_size=2,
        rank_flag="--tensor-parallel-rank",
    )
    cmd = supervisor._rank_command(7)
    assert cmd == ["cmd", "a", "b", "c", "--tensor-parallel-rank", "7"]


def test_supervisor_successful_startup_and_cleanup() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_READY],
        world_size=2,
        startup_timeout=5.0,
        shutdown_timeout=2.0,
        forward_output=False,
    )
    supervisor.start()
    try:
        assert len(supervisor.processes) == 2
        assert supervisor.rendezvous_dir is not None
        assert supervisor.nccl_id_path is not None
        assert supervisor.master_port is not None
        assert 1 <= supervisor.master_port <= 65535

        supervisor.wait_ready(timeout=5.0)
        assert all(record.ready for record in supervisor.processes)
        assert all(record.returncode is None for record in supervisor.processes)
    finally:
        supervisor.cleanup()

    assert all(record.returncode is not None for record in supervisor.processes)


def test_supervisor_removes_only_owned_rendezvous_dir() -> None:
    with tempfile.TemporaryDirectory() as caller_dir:
        caller_path = Path(caller_dir)
        supervisor = TensorParallelSupervisor(
            command=[sys.executable, "-c", _CHILD_EXIT],
            world_size=2,
            rendezvous_dir=caller_path,
            startup_timeout=3.0,
            shutdown_timeout=1.0,
            forward_output=False,
        )
        supervisor.start()
        supervisor.wait_ready()
        supervisor.cleanup()
        assert caller_path.exists(), "supervisor must not delete caller-owned rendezvous"

    supervisor2 = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_EXIT],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor2.start()
    generated = supervisor2.rendezvous_dir
    assert generated is not None
    supervisor2.wait_ready()
    supervisor2.cleanup()
    assert not generated.exists(), "supervisor must remove generated rendezvous"


def test_supervisor_startup_timeout() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_NO_READY],
        world_size=2,
        startup_timeout=0.5,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor.start()
    try:
        with pytest.raises(TensorParallelSupervisorError, match="timed out.*missing ranks"):
            supervisor.wait_ready()
    finally:
        supervisor.cleanup()


def test_supervisor_early_rank_failure() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_FAIL],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor.start()
    try:
        with pytest.raises(TensorParallelSupervisorError, match="exited during startup.*42"):
            supervisor.wait_ready()
    finally:
        supervisor.cleanup()


def test_supervisor_rank_exit_after_readiness() -> None:
    child_script = """
import os
import sys
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
import time
time.sleep(0.2)
if rank == 1:
    sys.exit(17)
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor.start()
    supervisor.wait_ready()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if supervisor.processes[1].returncode is not None:
                break
            time.sleep(0.05)
        assert supervisor.processes[1].returncode == 17
    finally:
        supervisor.cleanup()


def test_supervisor_run_propagates_worker_failure_and_reaps_rank_zero() -> None:
    child_script = """
import os
import sys
import time
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
if rank == 1:
    time.sleep(0.2)
    print("worker failed after readiness", file=sys.stderr, flush=True)
    sys.exit(23)
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    with pytest.raises(TensorParallelSupervisorError, match="rank 1.*runtime.*23"):
        supervisor.run()
    assert all(record.returncode is not None for record in supervisor.processes)


def test_supervisor_run_returns_after_rank_zero_exits_cleanly() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_EXIT],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    assert supervisor.run() == 0
    assert all(record.returncode is not None for record in supervisor.processes)


def test_supervisor_normalizes_duplicate_rank_flags() -> None:
    supervisor = TensorParallelSupervisor(
        command=["cmd", "--tensor-parallel-rank", "1", "--tensor-parallel-rank=2"],
        world_size=2,
    )
    # Repeated flags are normalized to the rank selected by the supervisor.
    assert supervisor._rank_command(0) == ["cmd", "--tensor-parallel-rank", "0"]

def test_supervisor_rejects_incomplete_rank_flag() -> None:
    supervisor = TensorParallelSupervisor(
        command=["cmd", "--tensor-parallel-rank"],
        world_size=2,
    )
    with pytest.raises(ConfigurationError, match="incomplete"):
        supervisor._rank_command(0)


def test_supervisor_rejects_duplicate_rank_flags_with_builder() -> None:
    def builder(rank: int, env: dict[str, str]) -> list[str]:
        return ["cmd", "--tensor-parallel-rank", "1", "--tensor-parallel-rank", str(rank)]

    supervisor = TensorParallelSupervisor(
        command=["cmd"],
        world_size=2,
        command_builder=builder,
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})
    # Builders own their complete argv and are not rewritten by _rank_command.
    assert supervisor._command_for_rank(1, supervisor._child_env(1)) == [
        "cmd", "--tensor-parallel-rank", "1", "--tensor-parallel-rank", "1"
    ]


def test_supervisor_passes_config_env_to_builder() -> None:
    seen: list[tuple[int, str]] = []

    def builder(rank: int, env: dict[str, str]) -> list[str]:
        seen.append((rank, env["POCKETLLM_NCCL_ID_PATH"]))
        return [sys.executable, "-c", _CHILD_READY]

    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_READY],
        world_size=2,
        env={"BUILDER_VALUE": "yes"},
        command_builder=builder,
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})
    env = supervisor._child_env(0)
    assert env["BUILDER_VALUE"] == "yes"
    assert supervisor._command_for_rank(0, env)[0] == sys.executable
    assert seen == [(0, env["POCKETLLM_NCCL_ID_PATH"])]


def test_supervisor_kills_signal_ignoring_children() -> None:
    child_script = """
import os
import signal
import time
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=0.15,
        forward_output=False,
    )
    supervisor.start()
    supervisor.wait_ready()
    supervisor._shutdown_children(signal.SIGTERM)
    assert all(record.returncode is not None for record in supervisor.processes)
    supervisor.cleanup()


def test_supervisor_forwards_signal_to_child_process_group() -> None:
    child_script = """
import os
import signal
import subprocess
import sys
import time
rank = int(os.environ["RANK"])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=False)
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=0.15,
        forward_output=False,
    )
    supervisor.start()
    supervisor.wait_ready()
    supervisor._shutdown_children(signal.SIGTERM)
    assert all(record.returncode is not None for record in supervisor.processes)
    supervisor.cleanup()


def test_supervisor_signal_handler_forwards_and_restores() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_READY],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    previous = signal.getsignal(signal.SIGTERM)
    supervisor._install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert supervisor.signal_received == signal.SIGTERM
    finally:
        supervisor._restore_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_supervisor_signal_handling() -> None:
    # This test can only install handlers if it runs on the main thread.
    # ThreadPoolExecutor-driven tests skip handler installation harmlessly.
    child_script = """
import os
rank = int(os.environ["RANK"])
print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)
import signal
import time
caught = []
def handle(signum, frame):
    caught.append(signum)
signal.signal(signal.SIGTERM, handle)
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=2.0,
        forward_output=False,
    )
    supervisor.start()
    supervisor.wait_ready()
    supervisor._shutdown_children(signal.SIGTERM)
    assert all(record.returncode is not None for record in supervisor.processes)
    supervisor.cleanup()


def test_supervisor_idempotent_cleanup() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", _CHILD_EXIT],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor.start()
    supervisor.wait_ready()
    supervisor.cleanup()
    supervisor.cleanup()
    supervisor.cleanup()


def test_supervisor_rank_announcement_mismatch() -> None:
    child_script = """
import os
rank = int(os.environ["RANK"])
announced = 1 - rank
print(f"POCKETLLM_RANK_READY rank={announced}", flush=True)
import time
time.sleep(600)
"""
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", child_script],
        world_size=2,
        startup_timeout=3.0,
        shutdown_timeout=1.0,
        forward_output=False,
    )
    supervisor.start()
    try:
        with pytest.raises(TensorParallelSupervisorError, match="announced readiness for rank"):
            supervisor.wait_ready()
    finally:
        supervisor.cleanup()


def test_supervisor_master_port_selection() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", "pass"],
        world_size=2,
        master_port=0,
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})
    assert 1 <= supervisor.master_port <= 65535

    supervisor2 = TensorParallelSupervisor(
        command=[sys.executable, "-c", "pass"],
        world_size=2,
        master_port=12345,
    )
    supervisor2._prepare_rendezvous()
    supervisor2._prepare_master_port({})
    assert supervisor2.master_port == 12345


def test_supervisor_preserves_base_env() -> None:
    os.environ["POCKETLLM_TEST_VAR"] = "from_parent"
    try:
        supervisor = TensorParallelSupervisor(
            command=[sys.executable, "-c", "pass"],
            world_size=2,
        )
        supervisor._prepare_rendezvous()
        supervisor._prepare_master_port({})
        env = supervisor._child_env(0)
        assert env["POCKETLLM_TEST_VAR"] == "from_parent"
        assert env["RANK"] == "0"
    finally:
        os.environ.pop("POCKETLLM_TEST_VAR", None)


def test_supervisor_config_env_override() -> None:
    supervisor = TensorParallelSupervisor(
        command=[sys.executable, "-c", "pass"],
        world_size=2,
        env={"CUSTOM_KEY": "custom_value"},
    )
    supervisor._prepare_rendezvous()
    supervisor._prepare_master_port({})
    env = supervisor._child_env(0)
    assert env["CUSTOM_KEY"] == "custom_value"
    assert env["RANK"] == "0"
