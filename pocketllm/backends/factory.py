"""Backend selection and construction."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from pocketllm.api import BackendUnavailableError, EngineArgs, UnsupportedFeatureError

from .cpp_backend import CppBackend
from .torch_backend import TorchBackend


_QWEN35_TYPES = {"qwen3_5", "qwen3_5_text"}


def _config_path(path: str, explicit: str | None = None) -> Path:
    candidate = Path(explicit) if explicit else Path(path) / "config.json"
    if candidate.is_dir():
        candidate = candidate / "config.json"
    return candidate


def _read_config(path: str, explicit: str | None = None) -> dict[str, Any] | None:
    try:
        config = _config_path(path, explicit)
        if not config.is_file():
            return None
        with config.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _is_qwen35_config(config: dict[str, Any]) -> bool:
    def is_qwen35(value: Any) -> bool:
        return str(value or "").lower() in _QWEN35_TYPES

    if is_qwen35(config.get("model_type")):
        return True
    architectures = config.get("architectures", ())
    if isinstance(architectures, (list, tuple)):
        if any("qwen3_5" in str(item).lower() for item in architectures):
            return True
    nested = config.get("text_config")
    return isinstance(nested, dict) and _is_qwen35_config(nested)


def _looks_like_qwen35(path: str, config_path: str | None = None) -> bool:
    config = _read_config(path, config_path)
    return config is not None and _is_qwen35_config(config)


def _checkpoint_has_gguf(path: str) -> bool:
    """Return whether the requested checkpoint is visibly a GGUF model."""
    try:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.suffix.lower() == ".gguf"
        if not candidate.is_dir():
            return False
        return any(candidate.glob("*.gguf"))
    except OSError:
        return False


def _cpp_model_supported(args: EngineArgs) -> bool:
    if args.model_format == "gguf":
        return False
    if args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir):
        return False
    return _looks_like_qwen35(args.checkpoint_dir, args.config_path)


def _reject_unsupported_cpp_checkpoint(args: EngineArgs) -> None:
    """Fail fast for checkpoints the native adapter provably cannot serve.

    A missing or unreadable path is left alone so injected engines and unusual
    layouts still reach the native loader, which reports the precise error.
    """
    if args.model_format == "gguf" or (
        args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir)
    ):
        raise UnsupportedFeatureError(
            "the native C++ adapter supports Qwen3.5 safetensors only; "
            "GGUF checkpoints must use backend='torch'"
        )
    config = _read_config(args.checkpoint_dir, args.config_path)
    if config is not None and not _is_qwen35_config(config):
        raise UnsupportedFeatureError(
            "the native C++ adapter supports Qwen3.5 checkpoints only"
        )


def select_backend(args: EngineArgs) -> str:
    """Select a backend without silently changing an explicit user choice."""
    if args.backend != "auto":
        if args.backend == "cpp":
            _reject_unsupported_cpp_checkpoint(args)
        return args.backend
    if CppBackend.native_available() and _cpp_model_supported(args):
        return "cpp"
    return "torch"


def create_backend(args: EngineArgs, **injected: Any):
    """Construct the selected adapter.

    ``injected`` is intentionally useful for tests and embedding applications;
    production callers normally only pass ``EngineArgs``.
    """
    selected = select_backend(args)
    if selected == "torch":
        return TorchBackend(
            args,
            runtime=injected.get("runtime"),
            serving_engine=injected.get("serving_engine"),
            runtime_loader=injected.get("runtime_loader"),
        )
    if selected == "cpp":
        # Check if we need automatic tensor-parallel supervision
        needs_supervision = (
            args.tensor_parallel_size > 1
            and args.tensor_parallel_rank == 0
            and not args.backend_options.get("nccl_id_path")
            and not os.environ.get("POCKETLLM_NCCL_ID_PATH")
            and not injected.get("_supervised_child")
        )

        if needs_supervision:
            # Auto-launch supervisor for multi-GPU setup
            from ..supervisor import TensorParallelSupervisor, TensorParallelConfig

            # Build command for worker processes (rank 1, 2, 3, ...)
            worker_command = [
                sys.executable,
                "-c",
                _worker_script(),
            ]

            # Build environment for worker processes
            # Note: NCCL ID path will be set by supervisor after rendezvous setup
            # Every field that changes how a rank executes has to be forwarded.
            # A worker that falls back to an EngineArgs default takes a different
            # branch from rank 0 and enters a different number of collectives,
            # which hangs the whole group instead of failing.  enable_prefix_caching
            # and max_batch_size are the ones that bite: the first decides whether
            # prefill resumes from a snapshot, the second sizes the KV arena.
            worker_env = {
                "POCKETLLM_CHECKPOINT": args.checkpoint_dir,
                "POCKETLLM_TP_SIZE": str(args.tensor_parallel_size),
                "POCKETLLM_MAX_MODEL_LEN": str(args.max_model_len or 8192),
                "POCKETLLM_KV_CACHE_DTYPE": str(args.kv_cache_dtype or "auto"),
                "POCKETLLM_BACKEND_OPTIONS": json.dumps(args.backend_options),
                "POCKETLLM_WORKER_ARGS": json.dumps(_worker_arg_overrides(args)),
            }

            # Create supervisor configuration
            # Supervisor will spawn workers for ranks 1, 2, 3
            # (it starts from rank 0 in its own numbering, which we map to our rank 1)
            config = TensorParallelConfig(
                command=tuple(worker_command),
                world_size=args.tensor_parallel_size - 1,  # Spawn N-1 workers (ranks 1..N-1)
                env=worker_env,
            )

            # Launch worker processes - supervisor creates NCCL ID file during start()
            supervisor = TensorParallelSupervisor(config)
            supervisor.start()

            # Get NCCL ID path created by supervisor
            nccl_id_path = str(supervisor.nccl_id_path)

            # Rank 0 needs the rendezvous path too, but it belongs to this
            # supervisor only.  Leaking it into os.environ makes the *next*
            # create_backend() in the same process see a non-empty
            # POCKETLLM_NCCL_ID_PATH, conclude it is externally supervised, skip
            # spawning workers, and then block forever in the NCCL rendezvous
            # waiting for ranks nobody started.  Pass it through backend_options,
            # which is scoped to this backend, and restore the environment.
            previous_nccl_id_path = os.environ.get("POCKETLLM_NCCL_ID_PATH")
            os.environ["POCKETLLM_NCCL_ID_PATH"] = nccl_id_path
            # backend_options is caller-owned; a copy keeps the mutation from
            # persisting in the EngineArgs a caller may reuse for another engine.
            args = replace(
                args,
                backend_options={**args.backend_options, "nccl_id_path": nccl_id_path},
            )

            try:
                # Main process creates rank-0 backend
                backend = CppBackend(
                    args,
                    native_module=injected.get("native_module"),
                    engine=injected.get("engine"),
                    tokenizer=injected.get("tokenizer"),
                )
            except BaseException:
                # The workers are useless without rank 0, and leaving them alive
                # would hold device memory and block a retry.
                try:
                    supervisor.stop()
                finally:
                    _restore_env("POCKETLLM_NCCL_ID_PATH", previous_nccl_id_path)
                raise

            _restore_env("POCKETLLM_NCCL_ID_PATH", previous_nccl_id_path)

            # Store supervisor reference so it can be cleaned up
            backend._supervisor = supervisor

            return backend
        else:
            # Single GPU or externally supervised
            return CppBackend(
                args,
                native_module=injected.get("native_module"),
                engine=injected.get("engine"),
                tokenizer=injected.get("tokenizer"),
            )
    raise BackendUnavailableError(f"unsupported backend {selected!r}")


# EngineArgs fields that change control flow or memory layout inside the native
# engine.  Ranks must agree on all of them; see the comment at the worker_env
# construction site.  Fields that are per-rank (tensor_parallel_rank, device) or
# already forwarded explicitly are deliberately absent.
_WORKER_SHARED_ARGS = (
    "prefill_chunk_tokens",
    "enable_prefix_caching",
    "attention_window",
    "attention_sink_tokens",
    "speculative_method",
    "speculative_tokens",
    "max_batch_size",
    "model_format",
    "dtype",
)


def _worker_arg_overrides(args: EngineArgs) -> dict[str, Any]:
    """Collect the EngineArgs values a worker rank must match."""
    return {name: getattr(args, name) for name in _WORKER_SHARED_ARGS}


def _restore_env(name: str, previous: str | None) -> None:
    """Put ``name`` back the way it was, distinguishing unset from empty."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _worker_script() -> str:
    """Generate Python code for worker processes (ranks 1, 2, 3, ...).

    The supervisor spawns world_size-1 processes with TP_RANK 0, 1, 2, ...
    We map them to actual tensor-parallel ranks 1, 2, 3, ...
    """
    return """
import os
import json
from pocketllm import EngineArgs
from pocketllm.backends.cpp_backend import CppBackend

# Get configuration from environment
# TP_RANK is set by supervisor (0, 1, 2, ... for world_size-1 workers)
# Map to actual TP ranks 1, 2, 3, ... (rank 0 is the main process)
supervisor_rank = int(os.environ.get("TP_RANK", "0"))
actual_rank = supervisor_rank + 1

tp_size = int(os.environ["POCKETLLM_TP_SIZE"])
checkpoint = os.environ["POCKETLLM_CHECKPOINT"]
nccl_id_path = os.environ["POCKETLLM_NCCL_ID_PATH"]
max_model_len = int(os.environ.get("POCKETLLM_MAX_MODEL_LEN", "8192"))
kv_cache_dtype = os.environ.get("POCKETLLM_KV_CACHE_DTYPE", "auto")
backend_options = json.loads(os.environ.get("POCKETLLM_BACKEND_OPTIONS", "{}"))
# Fields rank 0 resolved; a default here would desynchronize the collectives.
shared_args = json.loads(os.environ.get("POCKETLLM_WORKER_ARGS", "{}"))

# Add NCCL ID to backend options
backend_options["nccl_id_path"] = nccl_id_path

# Create EngineArgs for this worker rank
args = EngineArgs(
    model=checkpoint,
    backend="cpp",
    tensor_parallel_size=tp_size,
    tensor_parallel_rank=actual_rank,
    max_model_len=max_model_len,
    kv_cache_dtype=kv_cache_dtype,
    backend_options=backend_options,
    **shared_args,
)

# Create backend and enter worker loop
backend = CppBackend(args)
print(f"POCKETLLM_RANK_READY rank={actual_rank}", flush=True)
backend.run_worker(on_ready=None)
"""

