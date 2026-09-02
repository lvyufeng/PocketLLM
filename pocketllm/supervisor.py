"""Local tensor-parallel process supervision.

This module owns process lifecycle and rendezvous metadata only.  It deliberately
knows nothing about model weights, device buffers, collectives, or schedulers;
those remain responsibilities of the selected backend and its rank entry point.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api.errors import ConfigurationError, TensorParallelSupervisorError


_READY_RE = re.compile(r"^POCKETLLM_RANK_READY\s+rank=(?P<rank>[0-9]+)\s*$")


@dataclass(frozen=True, slots=True)
class TensorParallelConfig:
    """Immutable configuration for one local tensor-parallel process group."""

    command: tuple[str, ...]
    world_size: int
    env: Mapping[str, str] | None = None
    cwd: str | os.PathLike[str] | None = None
    startup_timeout: float = 300.0
    shutdown_timeout: float = 30.0
    master_addr: str | None = None
    master_port: int | None = None
    rendezvous_dir: str | os.PathLike[str] | None = None
    rank_flag: str = "--tensor-parallel-rank"
    forward_output: bool = True

    def __post_init__(self) -> None:
        command = tuple(str(item) for item in self.command)
        object.__setattr__(self, "command", command)
        if not command or any(not item for item in command):
            raise ConfigurationError("supervisor command must not be empty")
        if self.world_size < 2:
            raise ConfigurationError("supervisor world_size must be >= 2")
        if self.startup_timeout <= 0:
            raise ConfigurationError("supervisor startup_timeout must be positive")
        if self.shutdown_timeout <= 0:
            raise ConfigurationError("supervisor shutdown_timeout must be positive")
        if self.master_port is not None and not 0 <= int(self.master_port) <= 65535:
            raise ConfigurationError("supervisor master_port must be in [0, 65535]")
        if not self.rank_flag:
            raise ConfigurationError("supervisor rank_flag must not be empty")


@dataclass(slots=True)
class RankProcess:
    """Runtime record for one supervised child."""

    rank: int
    process: subprocess.Popen[str]
    process_group_id: int | None = None
    ready: bool = False
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    reader_threads: list[threading.Thread] = field(default_factory=list)

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    def output(self, limit: int = 40) -> str:
        """Return a bounded diagnostic tail suitable for an exception message."""
        lines: list[str] = []
        if self.stdout:
            lines.append("stdout:")
            lines.extend(self.stdout[-limit:])
        if self.stderr:
            lines.append("stderr:")
            lines.extend(self.stderr[-limit:])
        return "\n".join(lines)


class TensorParallelSupervisor:
    """Launch and supervise one local rank process per TP rank.

    ``command`` is the common command prefix for every rank.  The supervisor
    removes an existing ``rank_flag`` pair and appends the rank-specific value,
    so callers can safely pass a command assembled from user arguments.  A
    ``command_builder`` can be supplied when a backend needs generated
    rendezvous data in argv; it receives ``(rank, child_env)`` after the private
    rendezvous directory has been created.
    """

    def __init__(
        self,
        config: TensorParallelConfig | None = None,
        *,
        command: Sequence[str] | None = None,
        world_size: int | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        startup_timeout: float = 300.0,
        shutdown_timeout: float = 30.0,
        master_addr: str | None = None,
        master_port: int | None = None,
        rendezvous_dir: str | os.PathLike[str] | None = None,
        rank_flag: str = "--tensor-parallel-rank",
        forward_output: bool = True,
        command_builder: Callable[[int, Mapping[str, str]], Sequence[str]] | None = None,
    ) -> None:
        if config is not None and any(
            value is not None
            for value in (command, world_size, env, cwd, master_addr, master_port, rendezvous_dir)
        ):
            raise TypeError("pass either TensorParallelConfig or supervisor keyword configuration")
        if config is None:
            if command is None or world_size is None:
                raise TypeError("command and world_size are required")
            config = TensorParallelConfig(
                command=tuple(command),
                world_size=int(world_size),
                env=env,
                cwd=cwd,
                startup_timeout=float(startup_timeout),
                shutdown_timeout=float(shutdown_timeout),
                master_addr=master_addr,
                master_port=master_port,
                rendezvous_dir=rendezvous_dir,
                rank_flag=rank_flag,
                forward_output=forward_output,
            )
        elif command_builder is not None:
            # A builder is an execution detail and cannot be represented by the
            # immutable value object.  It is still valid with an explicit config.
            pass
        self.config = config
        self.command_builder = command_builder
        self._records: dict[int, RankProcess] = {}
        self._events: queue.Queue[tuple[int, str, str]] = queue.Queue()
        self._rendezvous_dir: Path | None = None
        self._owns_rendezvous_dir = False
        self._nccl_id_path: Path | None = None
        self._master_port: int | None = None
        self._started = False
        self._cleaned = False
        self._shutdown_started = False
        self._signal_received: int | None = None
        self._old_signal_handlers: dict[int, Any] = {}
        self._lock = threading.RLock()

    @property
    def processes(self) -> tuple[RankProcess, ...]:
        """Return rank records in deterministic rank order."""
        return tuple(self._records[rank] for rank in sorted(self._records))

    @property
    def rendezvous_dir(self) -> Path | None:
        return self._rendezvous_dir

    @property
    def nccl_id_path(self) -> Path | None:
        return self._nccl_id_path

    @property
    def master_port(self) -> int | None:
        return self._master_port

    @property
    def signal_received(self) -> int | None:
        return self._signal_received

    def _base_env(self) -> dict[str, str]:
        result = dict(os.environ)
        if self.config.env is not None:
            result.update({str(key): str(value) for key, value in self.config.env.items()})
        return result

    @staticmethod
    def _free_port(addr: str) -> int:
        # The socket is intentionally held only during selection.  The child
        # process group owns the actual TCPStore listener after it starts.
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        bind_addr: tuple[str, int] | tuple[str, int, int, int]
        bind_addr = (addr, 0, 0, 0) if family == socket.AF_INET6 else (addr, 0)
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(bind_addr)
            return int(sock.getsockname()[1])

    def _prepare_rendezvous(self) -> None:
        configured = self.config.rendezvous_dir
        if configured is None:
            path = Path(tempfile.mkdtemp(prefix="pocketllm-tp-"))
        else:
            parent = Path(configured).expanduser()
            parent.mkdir(parents=True, exist_ok=True)
            # Treat a caller-provided path as a parent, not as a reusable ID
            # directory. Each run gets a fresh child, while the caller's path
            # and any unrelated contents are never removed by cleanup.
            path = Path(tempfile.mkdtemp(prefix="pocketllm-tp-", dir=parent))
        self._owns_rendezvous_dir = True
        try:
            os.chmod(path, 0o700)
        except OSError:
            # A restrictive umask still protects a newly-created directory; a
            # caller-owned filesystem may not allow chmod and must not block the
            # process lifecycle solely for that reason.
            pass
        self._rendezvous_dir = path
        self._nccl_id_path = path / "nccl_id"

    def _prepare_master_port(self, env: Mapping[str, str]) -> None:
        requested = self.config.master_port
        if requested is None:
            raw = env.get("MASTER_PORT", "")
            try:
                requested = int(raw) if raw.strip() else 0
            except ValueError as exc:
                raise ConfigurationError("MASTER_PORT must be an integer") from exc
        try:
            requested = int(requested)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("supervisor master_port must be an integer") from exc
        if requested == 0:
            addr = self.config.master_addr or env.get("MASTER_ADDR") or "127.0.0.1"
            # A hostname may resolve to IPv6; localhost is deliberately mapped
            # to IPv4 because torchrun and the local worker set both support it.
            if addr in {"localhost", "::1"}:
                addr = "127.0.0.1"
            requested = self._free_port(addr)
        if not 1 <= requested <= 65535:
            raise ConfigurationError("supervisor master_port must resolve to [1, 65535]")
        self._master_port = requested

    def _child_env(self, rank: int) -> dict[str, str]:
        if self._rendezvous_dir is None or self._nccl_id_path is None or self._master_port is None:
            raise RuntimeError("supervisor rendezvous is not prepared")
        env = self._base_env()
        master_addr = self.config.master_addr or env.get("MASTER_ADDR") or "127.0.0.1"
        env.update(
            {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(self._master_port),
                "WORLD_SIZE": str(self.config.world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "TP_WORLD": str(self.config.world_size),
                "TP_RANK": str(rank),
                "POCKETLLM_TP_SUPERVISED": "1",
                "POCKETLLM_TP_RENDEZVOUS_DIR": str(self._rendezvous_dir),
                "POCKETLLM_NCCL_ID_PATH": str(self._nccl_id_path),
                # Native backends that use a file rendezvous can consume this
                # without importing the Python control plane.
                "NCCL_ID_PATH": str(self._nccl_id_path),
            }
        )
        return env

    def _rank_command(self, rank: int) -> list[str]:
        command = list(self.config.command)
        result: list[str] = []
        index = 0
        while index < len(command):
            item = command[index]
            if item == self.config.rank_flag:
                if index + 1 >= len(command):
                    raise ConfigurationError(
                        f"supervisor command has incomplete {self.config.rank_flag} option"
                    )
                index += 2
                continue
            prefix = self.config.rank_flag + "="
            if item.startswith(prefix):
                index += 1
                continue
            result.append(item)
            index += 1
        result.extend((self.config.rank_flag, str(rank)))
        return result

    def _command_for_rank(self, rank: int, env: Mapping[str, str]) -> list[str]:
        command = (
            list(self.command_builder(rank, env))
            if self.command_builder is not None
            else self._rank_command(rank)
        )
        if not command or any(not str(item) for item in command):
            raise ConfigurationError(f"supervisor command for rank {rank} is empty")
        return [str(item) for item in command]

    def _reader(self, record: RankProcess, stream_name: str, stream: Any) -> None:
        target = record.stdout if stream_name == "stdout" else record.stderr
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                target.append(line)
                self._events.put((record.rank, stream_name, line))
                if self.config.forward_output:
                    destination = sys.stdout if stream_name == "stdout" else sys.stderr
                    print(f"[tp rank {record.rank}] {line}", file=destination, flush=True)
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _spawn(self, rank: int) -> RankProcess:
        env = self._child_env(rank)
        command = self._command_for_rank(rank, env)
        try:
            process = subprocess.Popen(
                command,
                cwd=self.config.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise TensorParallelSupervisorError(
                f"failed to launch TP rank {rank}: {' '.join(command)}: {exc}"
            ) from exc
        try:
            process_group_id = os.getpgid(process.pid)
        except OSError:
            process_group_id = None
        record = RankProcess(rank=rank, process=process, process_group_id=process_group_id)
        assert process.stdout is not None and process.stderr is not None
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            thread = threading.Thread(
                target=self._reader,
                args=(record, name, stream),
                name=f"pocketllm-tp-rank{rank}-{name}",
                daemon=True,
            )
            record.reader_threads.append(thread)
            thread.start()
        self._records[rank] = record
        return record

    def start(self) -> "TensorParallelSupervisor":
        """Create rendezvous state and launch every rank exactly once."""
        with self._lock:
            if self._started and not self._cleaned:
                raise RuntimeError("tensor-parallel supervisor is already started")
            self._records.clear()
            self._cleaned = False
            self._shutdown_started = False
            self._signal_received = None
            try:
                self._prepare_rendezvous()
                self._prepare_master_port(self._base_env())
                for rank in range(self.config.world_size):
                    self._spawn(rank)
            except BaseException:
                self.cleanup()
                raise
            self._started = True
            return self

    def _drain_events(self) -> None:
        while True:
            try:
                rank, stream, line = self._events.get_nowait()
            except queue.Empty:
                return
            match = _READY_RE.match(line)
            if match is None:
                continue
            announced_rank = int(match.group("rank"))
            if announced_rank != rank:
                raise TensorParallelSupervisorError(
                    f"TP rank process {rank} announced readiness for rank {announced_rank}"
                )
            record = self._records.get(rank)
            if record is not None:
                record.ready = True

    def _failure(self, record: RankProcess, phase: str) -> TensorParallelSupervisorError:
        # Give the pipe readers a brief chance to publish the final traceback
        # before constructing diagnostics for a just-exited child.
        for thread in record.reader_threads:
            thread.join(timeout=0.05)
        returncode = record.returncode
        details = record.output()
        suffix = f"\n{details}" if details else ""
        return TensorParallelSupervisorError(
            f"TP rank {record.rank} exited during {phase} with return code {returncode}{suffix}"
        )

    def wait_ready(self, timeout: float | None = None) -> None:
        """Wait until every rank emits the exact readiness marker."""
        if not self._started:
            raise RuntimeError("tensor-parallel supervisor has not been started")
        effective_timeout = self.config.startup_timeout if timeout is None else float(timeout)
        if effective_timeout <= 0:
            raise ConfigurationError("supervisor readiness timeout must be positive")
        deadline = time.monotonic() + effective_timeout
        try:
            while True:
                self._drain_events()
                if self._signal_received is not None:
                    raise TensorParallelSupervisorError(
                        f"TP supervisor interrupted by signal {self._signal_received} during startup"
                    )
                missing = [record for record in self.processes if not record.ready]
                if not missing:
                    return
                for record in self.processes:
                    returncode = record.returncode
                    if returncode is not None and not record.ready:
                        raise self._failure(record, "startup")
                if time.monotonic() >= deadline:
                    ranks = ", ".join(str(record.rank) for record in missing)
                    raise TensorParallelSupervisorError(
                        f"timed out waiting for TP rank readiness after "
                        f"{effective_timeout:.3f}s; "
                        f"missing ranks: {ranks}"
                    )
                time.sleep(0.01)
        except BaseException:
            # A failed rank can leave other ranks blocked in NCCL/TCPStore. Do
            # not require callers to remember a second cleanup call before the
            # startup failure is contained.
            self._shutdown_children(signal.SIGTERM)
            raise

    def _install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: Any) -> None:
            self._signal_received = int(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, handle)
            except (OSError, RuntimeError, ValueError):
                # Signal handlers are process-global and can only be installed
                # by the main thread. Unit tests may drive the supervisor from a
                # helper thread, where child cleanup remains fully functional.
                continue

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._old_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                pass
        self._old_signal_handlers.clear()

    @staticmethod
    def _send_process_signal(record: RankProcess, signum: int) -> None:
        process = record.process
        if process.poll() is not None and record.process_group_id is None:
            return
        try:
            # Every child starts a private session, so this also reaches any
            # backend helper descendants after the leader has exited.
            if record.process_group_id is not None:
                os.killpg(record.process_group_id, signum)
            else:
                process.send_signal(signum)
        except (OSError, ProcessLookupError):
            try:
                process.send_signal(signum)
            except (OSError, ProcessLookupError):
                pass

    def _wait_for_exit(self, timeout: float) -> list[RankProcess]:
        deadline = time.monotonic() + timeout
        while True:
            survivors = [record for record in self.processes if record.returncode is None]
            if not survivors or time.monotonic() >= deadline:
                return survivors
            time.sleep(0.01)

    def _shutdown_children(self, signum: int = signal.SIGTERM) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        for record in self.processes:
            self._send_process_signal(record, signum)
        survivors = self._wait_for_exit(self.config.shutdown_timeout)
        for record in survivors:
            self._send_process_signal(record, signal.SIGKILL)
        # Reap all children even when they were already observed as exited.
        for record in self.processes:
            try:
                record.process.wait(timeout=max(self.config.shutdown_timeout, 0.1))
            except (subprocess.TimeoutExpired, OSError):
                pass

    def cleanup(self) -> None:
        """Terminate/reap children and remove only supervisor-owned artifacts."""
        with self._lock:
            if self._cleaned:
                return
            self._shutdown_children(signal.SIGTERM)
            for record in self.processes:
                for thread in record.reader_threads:
                    thread.join(timeout=1.0)
            if self._owns_rendezvous_dir and self._rendezvous_dir is not None:
                shutil.rmtree(self._rendezvous_dir, ignore_errors=True)
            self._cleaned = True
            self._started = False
            self._rendezvous_dir = None
            self._nccl_id_path = None

    def __enter__(self) -> "TensorParallelSupervisor":
        try:
            self.start()
            self.wait_ready()
        except BaseException:
            self.cleanup()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()

    def run(self) -> int:
        """Launch, await readiness, and monitor the rank-0 serving process."""
        self._install_signal_handlers()
        try:
            self.start()
            self.wait_ready()
            while True:
                self._drain_events()
                if self._signal_received is not None:
                    self._shutdown_children(self._signal_received)
                    return 0
                # A worker exiting while rank 0 is serving is always an error;
                # collectives would otherwise hang the surviving ranks.
                for record in self.processes:
                    returncode = record.returncode
                    if returncode is None:
                        continue
                    if record.rank != 0:
                        raise self._failure(record, "runtime")
                    if returncode != 0:
                        raise self._failure(record, "runtime")
                    self._shutdown_children(signal.SIGTERM)
                    return 0
                time.sleep(0.05)
        finally:
            try:
                self.cleanup()
            finally:
                self._restore_signal_handlers()


# Friendly aliases for callers that prefer the shorter process-oriented name.
TPProcessSupervisor = TensorParallelSupervisor
TPProcessConfig = TensorParallelConfig


__all__ = [
    "RankProcess",
    "TensorParallelConfig",
    "TensorParallelSupervisor",
    "TensorParallelSupervisorError",
    "TPProcessConfig",
    "TPProcessSupervisor",
]
