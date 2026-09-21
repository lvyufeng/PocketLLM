"""Backend selection and construction."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from pocketllm.api import BackendUnavailableError, EngineArgs, UnsupportedFeatureError

from .cpp_backend import CppBackend
from .torch_backend import TorchBackend
from .v41_backend import V41Backend


_QWEN35_TYPES = {"qwen3_5", "qwen3_5_text"}
_V41_TYPES = {"deepseek_v41", "deepseek_v41_text"}


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


def _is_v41_config(config: dict[str, Any]) -> bool:
    """Whether a config describes DeepSeek-V4.1-Flash.

    The released checkpoint nests its text stack, so ``model_type`` is
    ``deepseek_v41`` at the root and ``deepseek_v41_text`` inside
    ``text_config`` -- the same shape ``_is_qwen35_config`` walks, and for the
    same reason: a wrapper config is still the architecture.
    """

    def is_v41(value: Any) -> bool:
        return str(value or "").lower() in _V41_TYPES

    if is_v41(config.get("model_type")):
        return True
    architectures = config.get("architectures", ())
    if isinstance(architectures, (list, tuple)):
        if any("deepseekv41" in str(item).lower() for item in architectures):
            return True
    nested = config.get("text_config")
    return isinstance(nested, dict) and _is_v41_config(nested)


def _looks_like_v41(path: str, config_path: str | None = None) -> bool:
    config = _read_config(path, config_path)
    return config is not None and _is_v41_config(config)


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


def _v41_model_supported(args: EngineArgs) -> bool:
    """Whether the requested checkpoint is one the V4.1 adapter can read."""
    if args.model_format == "gguf":
        return False
    if args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir):
        return False
    return _looks_like_v41(args.checkpoint_dir, args.config_path)


def _reject_unsupported_v41_checkpoint(args: EngineArgs) -> None:
    """Fail fast for checkpoints the V4.1 adapter provably cannot serve.

    A missing or unreadable path is left alone: the adapter reports the
    unreadable checkpoint itself, and with the encoder it needs named.
    """
    if args.model_format == "gguf" or (
        args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir)
    ):
        raise UnsupportedFeatureError(
            "backend='v41' reads the checkpoint's safetensors shards only; "
            "a GGUF checkpoint must use backend='torch'"
        )
    config = _read_config(args.checkpoint_dir, args.config_path)
    if config is not None and not _is_v41_config(config):
        raise UnsupportedFeatureError(
            "backend='v41' serves DeepSeek-V4.1-Flash checkpoints only"
        )


def select_backend(args: EngineArgs) -> str:
    """Select a backend without silently changing an explicit user choice."""
    if args.backend != "auto":
        if args.backend == "cpp":
            _reject_unsupported_cpp_checkpoint(args)
        elif args.backend == "v41":
            _reject_unsupported_v41_checkpoint(args)
        return args.backend
    # V4.1 before the native adapter: the two read different architectures, and
    # the adapter's own runtime is the only thing that can serve a V4.1
    # checkpoint at all -- the native one has no factory for it and rejects it.
    if _v41_model_supported(args):
        return "v41"
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
    if selected == "v41":
        def construct(resolved: EngineArgs) -> V41Backend:
            return V41Backend(
                resolved,
                front=injected.get("front"),
                loader=injected.get("loader"),
                tokenizer=injected.get("tokenizer"),
            )

        if _needs_supervision(args, injected):
            def build(resolved: EngineArgs) -> V41Backend:
                # Rank 0 loads inside the rendezvous window.  The environment that names the
                # group belongs to this process only while this call is on the stack, and this
                # adapter reads the group at load time rather than at construction time, so a
                # rank-0 backend handed back unloaded would find the rendezvous gone.
                backend = construct(resolved)
                backend.prepare()
                return backend

            return _supervise_rank_zero(
                args,
                worker_script=_v41_worker_script(),
                build=build,
                torch_rendezvous=True,
            )
        return construct(args)
    if selected == "cpp":
        def build(resolved: EngineArgs) -> CppBackend:
            return CppBackend(
                resolved,
                native_module=injected.get("native_module"),
                engine=injected.get("engine"),
                tokenizer=injected.get("tokenizer"),
            )

        if _needs_supervision(args, injected):
            return _supervise_rank_zero(args, worker_script=_worker_script(), build=build)
        return build(args)
    raise BackendUnavailableError(f"unsupported backend {selected!r}")


def _needs_supervision(args: EngineArgs, injected: Mapping[str, Any]) -> bool:
    """Whether rank 0 should launch the rest of its tensor-parallel group.

    A non-empty ``POCKETLLM_NCCL_ID_PATH`` or a supplied ``nccl_id_path`` means someone
    else owns the ranks -- the supervisor that started this process, or an embedding
    application -- and spawning here would start a second group at the same rendezvous.
    """
    return (
        args.tensor_parallel_size > 1
        and args.tensor_parallel_rank == 0
        and not args.backend_options.get("nccl_id_path")
        and not os.environ.get("POCKETLLM_NCCL_ID_PATH")
        and not injected.get("_supervised_child")
    )


def _supervise_rank_zero(
    args: EngineArgs,
    *,
    worker_script: str,
    build: Callable[[EngineArgs], Any],
    torch_rendezvous: bool = False,
) -> Any:
    """Launch ranks 1..N-1 and build rank 0 here, under the same rendezvous.

    Rank 0 stays in this process, so anything the supervisor puts in a child's
    environment has to be assembled here instead.  The rendezvous is published two
    ways for exactly that reason: ``backend_options['nccl_id_path']`` is scoped to the
    backend that needs it, and the process environment -- which is the only channel the
    workers have -- is set for the length of ``build`` and then put back.  Leaking the
    environment into the next ``create_backend`` call is what makes a second engine
    conclude it is already supervised, skip spawning, and then block forever in a
    rendezvous waiting for ranks nobody started.
    """
    from ..supervisor import TensorParallelConfig, TensorParallelSupervisor

    supervisor = TensorParallelSupervisor(
        TensorParallelConfig(
            # Rank 0 remains in the parent process, while the supervisor owns the remaining
            # actual TP ranks.  Keep the full world size in child environments so NCCL sees
            # the same group as rank 0, including the TP2 case with one child.
            command=(sys.executable, "-c", worker_script),
            world_size=args.tensor_parallel_size,
            child_ranks=tuple(range(1, args.tensor_parallel_size)),
            env=_worker_env(args),
        )
    )
    supervisor.start()

    nccl_id_path = str(supervisor.nccl_id_path)
    environment = {"POCKETLLM_NCCL_ID_PATH": nccl_id_path}
    if torch_rendezvous:
        # The same variables the supervisor gave the children, resolved the way it
        # resolves them: a caller-supplied address wins, and everything else is a local
        # group.  A rank 0 that met the others anywhere else would be a different group.
        environment.update({
            "MASTER_ADDR": os.environ.get("MASTER_ADDR") or "127.0.0.1",
            "MASTER_PORT": str(supervisor.master_port),
            "RANK": "0",
            "WORLD_SIZE": str(args.tensor_parallel_size),
            "LOCAL_RANK": "0",
        })

    resolved = replace(
        args, backend_options={**args.backend_options, "nccl_id_path": nccl_id_path}
    )
    with _environment(environment):
        try:
            backend = build(resolved)
        except BaseException:
            # The workers are useless without rank 0, and leaving them alive would hold
            # device memory and block a retry.
            supervisor.cleanup()
            raise

    # Store supervisor reference so it can be cleaned up
    backend._supervisor = supervisor

    return backend


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set ``values`` in the process environment, then put every name back."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            _restore_env(name, value)


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


def _worker_env(args: EngineArgs) -> dict[str, str]:
    """The environment a worker rank needs to rebuild rank 0's engine.

    The two trailing path variables are read by the V4.1 worker only: its adapter
    resolves the tokenizer and the config by path, and a rank that resolved a
    different one would load a different model under the same process group.  The
    native worker ignores them because its adapter reads neither.
    """
    return {
        "POCKETLLM_CHECKPOINT": args.checkpoint_dir,
        "POCKETLLM_TP_SIZE": str(args.tensor_parallel_size),
        "POCKETLLM_MAX_MODEL_LEN": str(args.max_model_len or 8192),
        "POCKETLLM_KV_CACHE_DTYPE": str(args.kv_cache_dtype or "auto"),
        "POCKETLLM_BACKEND_OPTIONS": json.dumps(args.backend_options),
        "POCKETLLM_WORKER_ARGS": json.dumps(_worker_arg_overrides(args)),
        "POCKETLLM_CONFIG_PATH": str(args.config_path or ""),
        "POCKETLLM_TOKENIZER_PATH": str(args.tokenizer_path or ""),
    }


def _restore_env(name: str, previous: str | None) -> None:
    """Put ``name`` back the way it was, distinguishing unset from empty."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _worker_script() -> str:
    """Generate Python code for worker processes (actual TP ranks 1, 2, ...).

    Rank 0 stays in the parent process. The supervisor assigns each child its
    actual TP rank, so the worker environment already matches the NCCL group.
    """
    return """
import os
import json
from pocketllm import EngineArgs
from pocketllm.backends.cpp_backend import CppBackend

# Get the actual rank assigned by the supervisor. Rank 0 belongs to the
# parent process; child TP ranks start at 1.
actual_rank = int(os.environ.get("TP_RANK", "0"))

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


def _v41_worker_script() -> str:
    """Generate Python code for a V4.1 worker rank (actual TP ranks 1, 2, ...).

    Rank 0 stays in the parent process.  Unlike the native worker, this one does not
    print its readiness marker before entering the loop: ``run_worker`` joins the
    process group first and loads the checkpoint second, and the load ends in a
    barrier rank 0 is already waiting at, so "constructed" is not a state worth
    announcing.  The marker is emitted from ``on_ready``, which runs once the rank
    has actually loaded and can serve.
    """
    return """
import os
import json
from pocketllm import EngineArgs
from pocketllm.backends.v41_backend import V41Backend

# Get the actual rank assigned by the supervisor. Rank 0 belongs to the
# parent process; child TP ranks start at 1.
actual_rank = int(os.environ.get("TP_RANK", "0"))

tp_size = int(os.environ["POCKETLLM_TP_SIZE"])
checkpoint = os.environ["POCKETLLM_CHECKPOINT"]
nccl_id_path = os.environ["POCKETLLM_NCCL_ID_PATH"]
max_model_len = int(os.environ.get("POCKETLLM_MAX_MODEL_LEN", "8192"))
kv_cache_dtype = os.environ.get("POCKETLLM_KV_CACHE_DTYPE", "auto")
backend_options = json.loads(os.environ.get("POCKETLLM_BACKEND_OPTIONS", "{}"))
# Fields rank 0 resolved; a default here would desynchronize the collectives.
shared_args = json.loads(os.environ.get("POCKETLLM_WORKER_ARGS", "{}"))
config_path = os.environ.get("POCKETLLM_CONFIG_PATH") or None
tokenizer_path = os.environ.get("POCKETLLM_TOKENIZER_PATH") or None

# Add NCCL ID to backend options
backend_options["nccl_id_path"] = nccl_id_path

# Create EngineArgs for this worker rank
args = EngineArgs(
    model=checkpoint,
    backend="v41",
    tensor_parallel_size=tp_size,
    tensor_parallel_rank=actual_rank,
    max_model_len=max_model_len,
    kv_cache_dtype=kv_cache_dtype,
    backend_options=backend_options,
    config_path=config_path,
    tokenizer_path=tokenizer_path,
    **shared_args,
)

# Construct and enter the worker loop.  The group is joined and the checkpoint
# loaded inside run_worker, so readiness is announced from there.
backend = V41Backend(args)
backend.run_worker(
    on_ready=lambda: print(f"POCKETLLM_RANK_READY rank={actual_rank}", flush=True)
)
"""

