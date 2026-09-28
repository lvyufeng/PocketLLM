"""The program a supervised tensor-parallel worker rank runs.

Rank 0 stays in the parent process; the supervisor starts ranks 1..N-1 as children, each with
the environment that names the process group and the checkpoint. What the child has to do is the
same whatever runtime it is serving: rebuild the `EngineArgs` rank 0 resolved, construct the
adapter named by the environment, tell the parent it is up, and enter `run_worker`. Only three
things about that differ per runtime, and they are the three fields on [`WorkerSpec`].

The program used to be *generated as source text*, once per runtime, and the three copies were
the same forty lines apart from their import line, their `backend=` value, and whether the load
had already happened by the time the adapter was constructed. Source built at runtime is code no
test can import, no type checker reads and no traceback shows a useful line for, and the
per-runtime difference it encoded was three fields -- so it is a registry entry now and the
program is constant. `program()` is what the supervisor is handed; `main()` is what it runs.

The child is a real module rather than a string for one more reason: a worker that fails to
start is the hardest thing in this tree to debug. The parent sees a child exit and a rendezvous
that never completes, and the exception that would explain it is in a process whose only output
is a `-c` script nobody can open.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from pocketllm.api import ConfigurationError, EngineArgs

#: The variable that says which of [`WORKERS`] this child is. It is set by the supervisor from
#: the same `WorkerSpec` that produced the program, so the two cannot name different runtimes.
BACKEND_ENV = "POCKETLLM_WORKER_BACKEND"


@dataclass(frozen=True)
class WorkerSpec:
    """What a worker rank of one runtime differs from the others in.

    Everything else about being a worker -- the environment it is handed, the engine it rebuilds,
    the marker it prints, the loop it enters -- is the same, and lives in `main`.
    """

    #: The adapter's class, as ``module:Class``. Imported inside the child and not here, so that
    #: listing the registry does not import every runtime's torch graph into the parent, and so
    #: that a child's import error is the child's.
    entry_point: str
    #: Whether the checkpoint is loaded by the time the adapter is constructed.
    #:
    #: True for the native adapter: its engine loads in the constructor, so the rank can announce
    #: itself before entering the loop.
    #:
    #: False for the Python runtimes, which join the process group and load *inside* `run_worker`.
    #: "Constructed" is not a state worth announcing there, and the marker is emitted from
    #: `on_ready` once the load's closing barrier has been passed. Announcing early would tell the
    #: parent a rank is up while it is still inside a barrier, and no supervisor can recover from
    #: a readiness marker that is a lie.
    ready_at_construction: bool
    #: `EngineArgs` fields whose value is a *path* rank 0 resolved and the child has to reach the
    #: same one by, as ``(environment variable, field)``. A rank that resolved a different
    #: tokenizer or config would load a different model under the same process group, and the
    #: mismatch would show up as a desynchronized collective rather than as a wrong path.
    path_fields: tuple[tuple[str, str], ...] = ()
    #: Anything else the child's `EngineArgs` needs, for a runtime with a field that is neither a
    #: path nor one of the shared ones. Empty for every runtime today.
    extra_args: Mapping[str, Any] = field(default_factory=dict)


WORKERS: dict[str, WorkerSpec] = {
    "cpp": WorkerSpec(
        entry_point="pocketllm.backends.cpp_backend:CppBackend",
        ready_at_construction=True,
    ),
    "mimo": WorkerSpec(
        entry_point="pocketllm.backends.mimo_backend:MimoBackend",
        ready_at_construction=False,
        path_fields=(("POCKETLLM_TOKENIZER_PATH", "tokenizer_path"),),
    ),
    "v41": WorkerSpec(
        entry_point="pocketllm.backends.v41_backend:V41Backend",
        ready_at_construction=False,
        path_fields=(
            ("POCKETLLM_CONFIG_PATH", "config_path"),
            ("POCKETLLM_TOKENIZER_PATH", "tokenizer_path"),
        ),
    ),
}

#: The child's program. Constant: which runtime it is arrives in the environment, because that is
#: the only channel the supervisor has to the child anyway. `-c` is kept rather than `-m` so the
#: command has no importable name to resolve against whatever the child's working directory is.
_PROGRAM = "from pocketllm.backends.worker import main; main()"


def program() -> str:
    """The ``-c`` argument the supervisor gives every worker rank."""
    return _PROGRAM


def _required(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError as exc:
        raise ConfigurationError(
            f"worker rank is missing {name}; it is started by the supervisor, not by hand"
        ) from exc


def _args_from_environment(name: str, spec: WorkerSpec) -> EngineArgs:
    """Rebuild the `EngineArgs` rank 0 resolved, from the environment the supervisor set.

    The shared fields come through as one JSON object rather than as one variable each, because
    they are exactly the ones that must not be defaulted independently: a rank that guessed a
    different `max_batch_size` would size its KV cache differently from rank 0 under the same
    process group. See `factory._WORKER_SHARED_ARGS`.

    The two optional paths are read unconditionally, for every runtime. They are set for every
    worker that starts through `factory`, so a runtime that ignores one is not affected by the
    read, and a runtime that needs one does not need a per-runtime branch here to get it.

    ``POCKETLLM_RESOLVED_OPTIONS`` is the other option tier and travels for the same reason the
    first one does: a rank that defaulted a prefix budget or a prefill width while rank 0 honoured
    the flag would evict a different prefix at a different time. Both are read as JSON objects and
    neither is defaulted here -- an absent variable is an empty tier, not a set of defaults, which
    is the only reading that cannot invent a value rank 0 never chose.
    """
    backend_options = json.loads(os.environ.get("POCKETLLM_BACKEND_OPTIONS", "{}"))
    backend_options["nccl_id_path"] = _required("POCKETLLM_NCCL_ID_PATH")
    resolved_options = json.loads(os.environ.get("POCKETLLM_RESOLVED_OPTIONS", "{}"))
    paths = {
        name: os.environ.get(variable) or None for variable, name in spec.path_fields
    }
    return EngineArgs(
        model=_required("POCKETLLM_CHECKPOINT"),
        backend=name,
        tensor_parallel_size=int(_required("POCKETLLM_TP_SIZE")),
        # The supervisor assigns the actual rank, and it is the one the process group is built
        # on. Rank 0 is not started as a child at all, so a child that computed its rank from
        # the group rather than reading it here would be one ahead of where it really is.
        tensor_parallel_rank=int(os.environ.get("TP_RANK", "0")),
        max_model_len=int(os.environ.get("POCKETLLM_MAX_MODEL_LEN", "8192")),
        kv_cache_dtype=os.environ.get("POCKETLLM_KV_CACHE_DTYPE", "auto"),
        backend_options=backend_options,
        resolved_options=resolved_options,
        **paths,
        **dict(spec.extra_args),
        **json.loads(os.environ.get("POCKETLLM_WORKER_ARGS", "{}")),
    )


def _construct(spec: WorkerSpec, args: EngineArgs) -> Any:
    module_name, _, class_name = spec.entry_point.partition(":")
    if not class_name:
        raise ConfigurationError(
            f"worker entry point {spec.entry_point!r} is not a module:Class pair"
        )
    return getattr(import_module(module_name), class_name)(args)


def main() -> None:
    """Run one worker rank: construct, announce, then serve rank 0's requests."""
    name = os.environ.get(BACKEND_ENV, "")
    if name not in WORKERS:
        raise ConfigurationError(
            f"no worker program for backend {name!r}; known runtimes are "
            f"{', '.join(sorted(WORKERS))}"
        )
    spec = WORKERS[name]

    rank = int(os.environ.get("TP_RANK", "0"))

    def announce() -> None:
        print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)

    backend = _construct(spec, _args_from_environment(name, spec))
    if spec.ready_at_construction:
        # The native engine loaded in the constructor, so there is no later moment to announce
        # from and `run_worker` is entered ready.
        announce()
        backend.run_worker()
        return
    backend.run_worker(on_ready=announce)
