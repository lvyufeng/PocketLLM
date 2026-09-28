"""The ``serve`` flags the runtimes' own option declarations generate.

U2b-2 of #447. Before this, a runtime's levers had two surfaces that knew nothing about each other:
``--backend-option expert_pool_rows=148``, which the adapter decodes, and ``--help``, which listed
``--backend-option KEY=VALUE`` once and left the operator to find the keys in the source. The
declarations from U2b-1 (:mod:`pocketllm.backends.options`) are the one statement of what those
levers are, so this module reads them into flags: one flag per concept, one ``--help`` section per
:class:`~pocketllm.backends.options.Group`, and the key names still accepted as the escape hatch.

**Every runtime's flags go in one namespace.** vLLM and SGLang both do this -- their parsers are
built from one config struct or one flat field list -- and the alternative, a namespace per runtime,
is what makes an operator guess which of ``--v41-prefill-chunk`` and ``--mimo-prefill-chunk`` is
theirs before they know which adapter their checkpoint will select. ``--backend`` defaults to
``auto``, so at parse time nobody knows. What makes one namespace safe is that a name two runtimes
both declare is *one* declaration (:mod:`pocketllm.backends.shared_options`), and that naming a
flag the selected runtime does not read is refused rather than ignored -- see
:func:`unread_options` and its caller in :func:`pocketllm.backends.factory.select_backend`.

**Two declarations are answered by a flag that already exists**, and they are the only two:

* ``prefill_chunk`` is spelled ``--prefill-chunk-tokens``, the native engine's name for the same
  quantity, which predates the declarations and is read by paths that have no options of their own.
* ``device`` has no flag at all, because ``--device`` already exists and means the *vendor* on one
  path and the card on another; splitting it into ``--device auto|cuda|ascend`` plus
  ``--device-ids`` is the whole of U3, and a generated ``--device`` in the meantime would be a
  second meaning for a name that has one. The card stays reachable as ``--backend-option
  device=cuda:1``, as it is today.

**A flag nobody named is absent**, not set to a sentinel: every action here is registered with
``argparse.SUPPRESS``, so the mapping this module fills holds exactly the options the launch named.
That is what lets the refusal ask "did the operator name this?" without SGLang's per-field
``---x-explicitly-set`` marks, which exist because its resolution passes write into a record the
parse has already populated. Ours has no second pass to distinguish itself from.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from . import mimo_backend, v41_backend, xing4_backend
from .options import BackendOption, Group, Kind

#: The runtimes that declare options, in ``capabilities.AUTO_ORDER``. Not every runtime does: the
#: native and generic adapters take their tuning from ``EngineArgs`` fields and the CLI's own flags,
#: so this is also the list of runtimes a generated flag can belong to.
DECLARING: dict[str, Any] = {
    "v41": v41_backend,
    "mimo": mimo_backend,
    "xing4": xing4_backend,
}

#: The namespace attribute every generated flag writes into, as ``{option name: value}``. One
#: attribute rather than one per flag, so the mapping's keys are the set of options this launch
#: named -- which is what :func:`unread_options` reads and what keeps "named" and "defaulted" from
#: being the same thing.
DECLARED_DEST = "declared_options"

#: The declarations the host's own flags already spell, as ``{option name: namespace attribute}``.
#:
#: Generating a flag for one of these would be a second spelling of a name that already has one,
#: which is the collision the whole merge exists to remove -- so the host's flag stays, and this is
#: where the two are tied together. The attribute may hold ``0`` or the empty string for "the
#: launch did not name it", which is how the host has always spelled unset on those flags; see
#: :func:`resolved_options`.
HOST_FLAGS: dict[str, str] = {
    "prefill_chunk": "prefill_chunk_tokens",
}

#: The declarations that deliberately have no flag yet. See the module docstring: ``device``'s name
#: is taken by a host flag that means two things, and replacing it is U3.
NO_FLAG: frozenset[str] = frozenset({"device"})

_HOST_ONLY = frozenset({*HOST_FLAGS, *NO_FLAG})


# ------------------------------------------------------------------------------------- registration


class _DeclaredOption(argparse.Action):
    """One generated flag, recording into the launch's declared-options mapping."""

    #: The declaration this flag answers for, passed as ``option=`` at registration.
    option: BackendOption

    def __init__(
        self, option_strings: Sequence[str], dest: str, *, option: BackendOption, **kwargs: Any
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.option = option

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        _record(namespace, self.dest, self.option.name, values)


class _DeclaredFlag(argparse.BooleanOptionalAction):
    """The same, for a flag declaration, which is one name and its ``--no-`` other half.

    The test is on the option string rather than on the value argparse passes, because a flag has
    no value: ``nargs=0`` means the *spelling* is the whole of what the operator said.
    """

    option: BackendOption

    def __init__(
        self, option_strings: Sequence[str], dest: str, *, option: BackendOption, **kwargs: Any
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.option = option

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        _record(namespace, self.dest, self.option.name, not str(option_string).startswith("--no-"))


def _record(namespace: argparse.Namespace, dest: str, name: str, value: Any) -> None:
    if getattr(namespace, dest, None) is None:
        setattr(namespace, dest, {})
    getattr(namespace, dest)[name] = value


def cli_name(name: str) -> str:
    """The flag an option name is spelled with, dashes and all."""
    return "--" + name.replace("_", "-")


def declarations() -> dict[str, dict[str, BackendOption]]:
    """``{option name: {runtime: the declaration that runtime reads}}`` over every runtime.

    A name two runtimes report is one shared declaration referenced twice, which is the point of the
    merge: the values here are the same object, and :func:`unread_options` can therefore name the
    runtimes that read a flag without a second table to keep in step.
    """
    found: dict[str, dict[str, BackendOption]] = {}
    for runtime, module in DECLARING.items():
        for option in module.OPTIONS:
            found.setdefault(option.name, {})[runtime] = option
    return found


def generated() -> dict[str, dict[str, BackendOption]]:
    """The declarations that get a flag, in the shape :data:`declarations` returns."""
    return {
        name: readers
        for name, readers in declarations().items()
        if name not in _HOST_ONLY
    }


def sections() -> dict[Group, list[str]]:
    """The generated option names, by the ``--help`` section they are listed under.

    The section comes from the declaration, not from the runtime that reads it, which is the whole
    reason :class:`~pocketllm.backends.options.Group` is a closed set: ``chunk_rows`` is MiMo's and
    ``expert_pool_rows`` is V4.1's, and both are the expert arena's, so they belong on one page.
    """
    found: dict[Group, list[str]] = {}
    for name, readers in generated().items():
        found.setdefault(_shape(readers).group, []).append(name)
    return found


def add_declared_options(parser: argparse.ArgumentParser) -> None:
    """Add every generated flag to ``parser``, under its declaration's ``--help`` section."""
    declared = generated()
    for group in Group:
        names = [name for name in declared if _shape(declared[name]).group is group]
        if not names:
            continue
        section = parser.add_argument_group(group.value)
        for name in names:
            readers = declared[name]
            shape = _shape(readers)
            section.add_argument(
                cli_name(name),
                dest=DECLARED_DEST,
                **_argument(shape, _help(shape, readers)),
            )


def _argument(shape: BackendOption, help_text: str) -> dict[str, Any]:
    """The ``add_argument`` keywords one declaration turns into.

    Two decisions worth stating. The value's *type* is argparse's, so a mistyped number is refused
    by the parser with the flag in the message, and the declaration's own decoder still runs
    afterwards on whatever arrives -- it is the tier that reads ``--backend-option`` as strings, and
    one decoder for both is what keeps a byte count's ``4g`` meaning one thing. And the default is
    ``SUPPRESS``, so an unnamed flag leaves no attribute behind (see the module docstring).
    """
    if shape.kind is Kind.FLAG:
        return {
            "action": _DeclaredFlag,
            "option": shape,
            "default": argparse.SUPPRESS,
            "help": help_text,
        }
    argument: dict[str, Any] = {
        "action": _DeclaredOption,
        "option": shape,
        "default": argparse.SUPPRESS,
        "type": int if shape.kind is Kind.INTEGER else str,
        "help": help_text,
    }
    if shape.choices:
        # No metavar: argparse renders the accepted values as `{sorted,id}` when it has them, which
        # is a better answer to "what goes here" than the word `VALUE`.
        argument["choices"] = shape.choices
    else:
        argument["metavar"] = _METAVARS[shape.kind]
    return argument


_METAVARS: dict[Kind, str] = {
    Kind.STRING: "VALUE",
    Kind.INTEGER: "N",
    Kind.BYTES: "BYTES",
    Kind.FLAG: "",
}


def _flag_default(shape: BackendOption) -> str:
    """What ``--flag`` means against ``--no-flag``, which a reader cannot guess from the name."""
    if shape.default is True:
        return f"on by default; {_negation(shape.name)} turns it off"
    return "off by default"


def _negation(name: str) -> str:
    return "--no-" + name.replace("_", "-")


def _help(shape: BackendOption, readers: Mapping[str, BackendOption]) -> str:
    """The shared sentence, then who reads the flag and what each of them answers when it is unset.

    The parenthetical is the part a flat namespace owes a reader: ``--expert-pool-rows`` is V4.1's
    and ``--chunk-rows`` is MiMo's, and neither the section nor the name says so. Upstream answers it
    with the family prefix, and we do too where the concept *is* a family's (``--expert-*``), but a
    runtime prefix is not what this is -- the concept is the same one on both sides of it.

    It is one line and not two because argparse collapses a newline in a help string into a space;
    the parentheses are what keep the boundary visible once it has.
    """
    details = [_readers_sentence(readers)]
    if shape.kind is Kind.FLAG:
        details.append(_flag_default(shape))
    else:
        unset = _unset(readers)
        if unset:
            details.append(unset)
    return f"{shape.help} ({'; '.join(details)})"


def _readers_sentence(readers: Mapping[str, BackendOption]) -> str:
    return f"{next(iter(readers))} only" if len(readers) == 1 else _and(readers)


def _unset(readers: Mapping[str, BackendOption]) -> str:
    """Where the flag lands when nobody names it, or the empty string when there is nothing to say.

    A runtime's answer is its ``resolution`` -- how it computes the value -- or the default it takes;
    an option whose default is ``None`` and which resolves nothing has no answer, and the sentence
    is dropped rather than filled with ``None``, which is not a value anybody can type and which the
    declaration's own prose already covers ("the host when unset", "the network's own width").

    When the readers agree, this is the ordinary ``--help`` line. When they do not, each answer names
    the runtimes it holds for, because that difference is exactly what the merge turned into a
    resolved default rather than a flag per runtime.
    """
    by_answer: dict[str, list[str]] = {}
    answered = 0
    for runtime, option in readers.items():
        answer = _answer(option)
        if answer is None:
            continue
        by_answer.setdefault(answer, []).append(runtime)
        answered += 1
    if not by_answer:
        return ""
    if len(by_answer) == 1 and answered == len(readers):
        return f"default {next(iter(by_answer))}"
    clauses = [f"{answer} on {_and(runtimes)}" for answer, runtimes in by_answer.items()]
    return "when unset: " + "; ".join(clauses)


def _answer(option: BackendOption) -> str | None:
    """One runtime's answer for an unset option: how it resolves, or the default it takes."""
    if option.resolution:
        return option.resolution
    if option.default is None:
        return None
    return option.rendered_default()


def _and(names: Iterable[str]) -> str:
    items = list(names)
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _shape(readers: Mapping[str, BackendOption]) -> BackendOption:
    """The declaration a flag's *shape* comes from -- name, kind, group, choices, bounds.

    Which of the readers that is does not matter, and a test holds them to it: the fields a runtime
    may answer differently are its default, its resolution and its own sentence appended to the
    shared help, and everything else is the flag rather than the runtime's version of it.
    """
    return next(iter(readers.values()))


# ------------------------------------------------------------------------------------ the launch


def resolved_options(namespace: argparse.Namespace) -> dict[str, Any]:
    """What the host's own command line decided, keyed the way a runtime declares it.

    Two sources, and both are the host's: a generated flag the launch named, and the host flags in
    :data:`HOST_FLAGS`. This is the function that ties the second kind to the declarations, and it
    is the only place that knows both names.

    An unset host flag is left out rather than recorded as ``0``/``""``: the tier this feeds is "the
    host decided", and a decision to name nothing is not one.
    """
    values = dict(getattr(namespace, DECLARED_DEST, None) or {})
    for name, attribute in HOST_FLAGS.items():
        value = getattr(namespace, attribute, None)
        if value:
            values[name] = value
    return values


def unread_options(runtime: str, args: Any) -> list[str]:
    """The options a launch named that ``runtime`` does not read, by their declared names.

    Names rather than flag spellings, because the caller's message names the flag (``cli_name``) and
    the declaration is the thing that decides whether it is read. The list is empty for a launch
    that named nothing this runtime does not cover -- including every launch of a runtime that
    declares no options, where a generated flag has no reader at all. A host flag is exempt:
    ``--prefill-chunk-tokens`` is the native engine's own field and predates the declarations, so a
    launch that names it is not naming something the selected runtime lacks.

    Read by ``select_backend``, because that is the first place the runtime is known: at parse time
    ``--backend`` may still be ``auto``, and the checkpoint that decides it has not been looked at.
    """
    module = DECLARING.get(runtime)
    declared = {option.name for option in module.OPTIONS} if module is not None else set()
    named = set(getattr(args, "resolved_options", None) or {})
    return sorted(named - declared - set(HOST_FLAGS))


def readers_of(name: str) -> tuple[str, ...]:
    """The runtimes that declare ``name``, for a refusal that can say who does read it."""
    return tuple(declarations().get(name, ()))
