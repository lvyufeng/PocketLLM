"""What each runtime is, and which checkpoints are its, declared once.

Before this the answer lived in several places that could disagree. Capability was computed by hand
in six ``capabilities()`` methods; the *rejection* of a checkpoint was written twice per adapter --
once as a predicate for ``auto`` and once as a raise for an explicit ``--backend``; three adapters
carried their own copy of the same ``_IGNORED_OPTIONS`` set; and ``supports_prefix_caching`` meant
"the store exists" on two runtimes and "the byte budget is positive" on a third, so the same field
answered two different questions depending on which backend you asked.

This module is the one place that answers "can this runtime do X" and "is this checkpoint its". The
adapters read it, the dispatcher reads it, and a runtime that has more to say than its declaration
says it by naming the field (see :func:`declared_capabilities`).

Two things are deliberately *not* here:

* **Whether a runtime is available** -- whether the native module imports, whether a card is
  present. That is a property of the machine, not of the runtime, and it changes between two
  processes on the same box.
* **Anything a configuration decides.** A declaration is static; the one capability that genuinely
  depends on the configuration is prefix reuse, and the value is resolved per instance because the
  runtime is the only thing that knows whether its gate is open. See
  :attr:`RuntimeCapabilities.reads_prefix_cache` for the single question that field asks.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from pocketllm.api import BackendCapabilities, EngineArgs


# ---------------------------------------------------------------------------------------------
# Recognising a checkpoint
# ---------------------------------------------------------------------------------------------
#
# These read a checkpoint's own headers. They are here rather than in the dispatcher because the
# declaration is what uses them, and a second copy of "does this config say qwen3_5" is how the
# routing question and the refusal question drift apart -- which is the defect this module exists
# to remove.

_XING4_TYPES = {"xing4_0", "xing4_0_text"}


def config_path_for(path: str, explicit: str | None = None) -> Path:
    candidate = Path(explicit) if explicit else Path(path) / "config.json"
    if candidate.is_dir():
        candidate = candidate / "config.json"
    return candidate


def read_config(path: str, explicit: str | None = None) -> dict[str, Any] | None:
    """A checkpoint's ``config.json``, or ``None`` when there is not a readable one.

    ``None`` means *no evidence*, and every caller has to treat it that way: a checkpoint whose
    config cannot be read is not one that has been disproved.
    """
    try:
        config = config_path_for(path, explicit)
        if not config.is_file():
            return None
        with config.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _names_model(config: Mapping[str, Any], types: set[str], architectures: str) -> bool:
    """Whether a config names one of ``types``, at the root or nested.

    The nesting walk is shared because the reason for it is: the released DeepSeek-V4.1 and
    Qwen3.5 checkpoints nest their text stack, so ``model_type`` is the architecture at the root
    and again inside ``text_config``. A wrapper config is still the architecture.
    """
    value = str(config.get("model_type") or "").lower()
    if value in types:
        return True
    declared = config.get("architectures", ())
    if isinstance(declared, (list, tuple)):
        if any(architectures in str(item).lower() for item in declared):
            return True
    nested = config.get("text_config")
    return isinstance(nested, dict) and _names_model(nested, types, architectures)


def is_xing4_config(config: Mapping[str, Any]) -> bool:
    """Whether a config describes Xing4.0-29B-A4B.

    The released checkpoint is not nested and its ``model_type`` is the name the GGUF's own
    ``general.architecture`` carries, so the two artifacts the release ships are recognized by the
    same string.
    """
    return _names_model(config, _XING4_TYPES, "xing4")


def requested_gguf(args: EngineArgs) -> bool:
    """Whether the caller is pointing at a GGUF, by declaration or by shape."""
    return args.model_format == "gguf" or (
        args.model_format == "auto" and checkpoint_has_gguf(args.checkpoint_dir)
    )


def checkpoint_has_gguf(path: str) -> bool:
    """Whether the requested checkpoint is visibly a GGUF model."""
    try:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.suffix.lower() == ".gguf"
        if not candidate.is_dir():
            return False
        return any(candidate.glob("*.gguf"))
    except OSError:
        return False


def gguf_architecture(path: str) -> str | None:
    """``general.architecture`` out of a GGUF, or ``None`` when there is no answer.

    Read rather than assumed: the file the launcher points at may be any GGUF, so this is what
    separates "a Xing4 export" from "a file ending in .gguf". The two failures are kept apart --
    ``None`` is *no evidence* (not a GGUF, more than one of them, a header that will not parse) and
    an empty string is impossible, so a caller can refuse on a header it read rather than on one it
    could not. Where it is ``None`` the caller falls through to the adapter, and the native reader
    reports the precise error when it is the one that ends up with the file.
    """
    try:
        candidate = Path(path)
        if candidate.is_dir():
            found = sorted(candidate.glob("*.gguf"))
            if len(found) != 1:
                return None
            candidate = found[0]
        if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
            return None
        from src.loader.gguf.bundle import read_gguf_bundle

        metadata = read_gguf_bundle(candidate).metadata
        return str(metadata.get("general.architecture") or "") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------------------------


class Verdict(Enum):
    """What a checkpoint says about which runtime should serve it."""

    #: This runtime is the one for it.
    READ = "read"
    #: No evidence either way. Not enough to be routed here, and not enough to refuse either --
    #: a caller who named this backend by hand gets the backend's own loader error, which names
    #: the file it could not read.
    UNKNOWN = "unknown"
    #: Evidence *against*. A refusal, and the sentence says what the evidence was.
    REFUSED = "refused"


@dataclass(frozen=True)
class Identification:
    verdict: Verdict
    reason: str = ""

    @property
    def routes_here(self) -> bool:
        return self.verdict is Verdict.READ

    @property
    def refuses(self) -> bool:
        return self.verdict is Verdict.REFUSED


READ = Identification(Verdict.READ)
UNKNOWN = Identification(Verdict.UNKNOWN)


def refused(reason: str) -> Identification:
    return Identification(Verdict.REFUSED, reason)


#: The ``backend_options`` keys every launch carries and no *model* options parser has a use for.
#:
#: The CLI fills in ``engine_kind``, ``routed_experts_device`` and ``pd_mode`` on every serve
#: command and the supervisor adds ``nccl_id_path`` for a sharded one, so all four arrive whether or
#: not the selected adapter reads them. Accepting them is what makes one launch command line work
#: for every backend. Every *other* unknown key is a refusal, because a tuning option that silently
#: does nothing is how a run ends up measured on the wrong lever.
#:
#: ``enable_batching`` and ``scheduler_timeout_ms`` are carried for the same reason and are *not*
#: ignored: `SchedulerHost` reads both, and they decide whether this runtime joins the shared
#: scheduler and how long it waits on it. They are here so a launch can name them without the model
#: option parser refusing them as unknown, which is the treatment every other key still gets.
IGNORED_OPTIONS = frozenset(
    {
        "engine_kind",
        "routed_experts_device",
        "pd_mode",
        "nccl_id_path",
        "enable_batching",
        "scheduler_timeout_ms",
    }
)


@dataclass(frozen=True)
class RuntimeCapabilities:
    """One runtime, declared. See the module docstring for what belongs here.

    The fields mirror :class:`pocketllm.api.BackendCapabilities`, which is the shape a client
    reads; :func:`declared_capabilities` is the one function that turns one into the other.
    """

    name: str
    models: tuple[str, ...] = ()
    model_formats: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    #: Whether this runtime can ever hold more than one request at a time. False is a promise that
    #: a width above 1 is refused rather than accepted and ignored.
    supports_batch: bool = False
    supports_streaming: bool = True
    supports_cancellation: bool = False
    supports_logprobs: bool = False
    supports_structured_outputs: bool = False
    supports_speculative_decoding: tuple[str, ...] = ()
    #: Whether this runtime has a prefix-resumption path *at all*. Whether it is open in a given
    #: configuration is the instance's answer, not this one's -- see
    #: :func:`declared_capabilities`'s ``reads_prefix_cache``.
    reads_prefix_cache: bool = False
    #: ``backend_options`` keys this runtime accepts and ignores.
    ignores_options: frozenset[str] = IGNORED_OPTIONS
    #: What ``auto`` needs to route here, and what an explicit ``--backend`` refuses on if the
    #: answer is :attr:`Verdict.REFUSED`. One predicate answers both, which is the point: the two
    #: questions used to be two functions and could disagree.
    identifies: Callable[[EngineArgs], Identification] = lambda _args: UNKNOWN
    #: Prose the dispatcher merges into ``details``: what this runtime executes on, what its
    #: scheduler is, how it cancels. A declaration's details are static; an instance adds its own.
    details: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------


_XING4_ONLY = (
    "backend='xing4' serves Xing4.0-29B-A4B checkpoints only: a directory whose "
    "config.json says model_type=xing4_0, or a .gguf whose general.architecture is xing4_0"
)

_XING4_GGUF_ONLY = (
    "backend='xing4' reads its weights from the GGUF; the safetensors directory beside it "
    "supplies the tokenizer and the config only, so --model-format safetensors is not a "
    "checkpoint this backend can serve"
)


def _identify_xing4(args: EngineArgs) -> Identification:
    """Whether this checkpoint is one ``xing4`` can serve, asked architecture first.

    The two causes are told apart because the caller acts on them differently: "this is not a
    Xing4 checkpoint" says stop, and "this is one, but you asked for the wrong ``--model-format``"
    says change a flag. Asking the format first would report the second for a checkpoint the first
    is true of, which is why the readability of the config comes before either.

    The format rule itself is not a formality: this runtime always reads its weights from the GGUF,
    so ``--model-format safetensors`` is never something it can serve. The auto path had always
    refused that pair; the explicit path used to let it through selection and fail at the loader
    with "no .gguf file at or under ...", which is the failure this moves earlier.
    """
    config = read_config(args.checkpoint_dir, args.config_path)
    if config is not None and not is_xing4_config(config):
        return refused(_XING4_ONLY)
    if str(args.model_format).lower() == "safetensors":
        return refused(_XING4_GGUF_ONLY)
    if config is not None:
        return READ
    # No config to read, so the GGUF's own header is the only evidence left -- and its absence is
    # no evidence at all rather than evidence against.
    architecture = gguf_architecture(args.checkpoint_dir)
    if architecture is None:
        return UNKNOWN
    if architecture == "xing4_0":
        return READ
    return refused(f"{_XING4_ONLY} (this one declares general.architecture={architecture!r})")


RUNTIMES: dict[str, RuntimeCapabilities] = {
    "xing4": RuntimeCapabilities(
        name="xing4",
        models=("xing4_0",),
        model_formats=("gguf",),
        devices=("cuda",),
        supports_cancellation=True,
        reads_prefix_cache=True,
        identifies=_identify_xing4,
        # No static prose: this runtime's description is entirely about the run, so the adapter
        # supplies all of it.
        details={},
    ),
}


#: The order ``auto`` asks the runtimes in. With one runtime this is the whole list; it stays a
#: tuple rather than a scalar because the loop below is what "auto" means, and a second checkpoint
#: family is an entry here plus a declaration.
AUTO_ORDER: tuple[str, ...] = ("xing4",)


def runtime_capabilities(name: str) -> RuntimeCapabilities:
    try:
        return RUNTIMES[name]
    except KeyError as exc:
        raise KeyError(
            f"no capability declaration for backend {name!r}; declared: {sorted(RUNTIMES)}"
        ) from exc


def declared_capabilities(
    name: str,
    *,
    details: Mapping[str, Any] | None = None,
    reads_prefix_cache: bool = True,
    **overrides: Any,
) -> BackendCapabilities:
    """The wire declaration for one runtime, from its declaration plus what only it knows.

    ``reads_prefix_cache`` is the gate, and it is a parameter rather than a field a caller can set
    directly because the question is fixed: **will a repeated prefix be resumed in this
    configuration, instead of being forwarded again?** That is the only reading a client can act
    on, and it is the reading the field used to lack -- the two runtimes that reported "the store
    object exists" and the one that reported "the byte budget is positive" were answering about
    their configuration while the field is documented as being about reuse.

    A runtime whose gate is open passes ``True`` (the default). One that can resume but is not
    configured to passes ``False``, and so does one on a code path that does not resume -- the
    ``cpp`` adapter's batch scheduler, which does not consult the prefix cache at all.

    ``overrides`` names any other field the instance knows better than the declaration does -- the
    architectures a build actually registered, the device backend it linked, whether it built a
    scheduler. Naming the field is deliberate: it is a short, greppable list of the places a
    runtime disagrees with its own declaration, which is what keeps the declaration honest.
    """
    declaration = replace(runtime_capabilities(name), **overrides)
    return BackendCapabilities(
        name=declaration.name,
        models=declaration.models,
        model_formats=declaration.model_formats,
        devices=declaration.devices,
        supports_batch=declaration.supports_batch,
        supports_streaming=declaration.supports_streaming,
        supports_cancellation=declaration.supports_cancellation,
        supports_embeddings=False,
        supports_logprobs=declaration.supports_logprobs,
        supports_structured_outputs=declaration.supports_structured_outputs,
        supports_prefix_caching=declaration.reads_prefix_cache and reads_prefix_cache,
        supports_speculative_decoding=declaration.supports_speculative_decoding,
        details={**declaration.details, **(details or {})},
    )


def identify(name: str, args: EngineArgs) -> Identification:
    """What the checkpoint in hand says about ``name``."""
    return runtime_capabilities(name).identifies(args)


def route(args: EngineArgs) -> str:
    """The runtime ``auto`` selects, or the empty string when nothing identifies the checkpoint."""
    for name in AUTO_ORDER:
        if runtime_capabilities(name).identifies(args).routes_here:
            return name
    return ""


def refusal(name: str, args: EngineArgs) -> str:
    """The sentence refusing ``args`` for an explicitly requested ``name``, or the empty string."""
    verdict = runtime_capabilities(name).identifies(args)
    return verdict.reason if verdict.refuses else ""
