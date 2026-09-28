"""One declaration per backend option, so a runtime's levers are named once.

Before this, every runtime that took options carried the same answer in three places: a
``_KNOWN_OPTIONS`` frozenset with the names, an ``_Options`` dataclass with the types and defaults,
and a ``from_args`` that hand-wrote an ``int(...)``, a range check and a "must be 'id' or 'sorted'"
per key. Adding one runtime option meant editing all three plus the refusal message, and nothing
made the three agree -- the frozenset could list a name the dataclass had no field for, and the
refusal could tell an operator the name was known while ``from_args`` raised ``TypeError`` on it.

A :class:`BackendOption` is the one statement of a key's name, type, default, bounds, accepted
values and one-line meaning; :func:`decode_options` is the one reader. A runtime declares its
options beside its capabilities and gets the refusal, the coercion and the ``--help`` text from
them, so the three cannot drift.

What stays in the adapter is what a declaration cannot express:

* **A default that is a function of the run** -- Xing4's chunk width is derived from the card's
  free memory, so the declaration carries the *parameter* (``prefill_chunk``, default ``None``) and
  the adapter resolves it after decoding.
* **A rule that spans keys** -- ``--enable-prefix-caching=false`` zeroes the budget rather than
  being a second switch, which is a statement about two options and not about either one.

That split is what keeps this module from having to know what an option *does*. It knows how to
read one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pocketllm.api import ConfigurationError


#: What a byte count's suffix multiplies by. Binary multiples, because the constants that describe
#: these budgets are written as shifts.
_UNITS = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}


def byte_size(value: Any, name: str) -> int:
    """A byte count, as an integer or as a ``<n>[kmg]`` string.

    ``--backend-option`` carries strings either way, and a byte budget is the one option whose plain
    value cannot be read at a glance in a launch script: ``4294967296`` against ``4g``. It lives in
    this module rather than in one adapter because two of them take a byte budget under the same
    option name and a launch that parsed differently on one path than on the other would be a run
    measured under a budget its launcher did not choose.
    """
    text = str(value).strip().lower()
    factor = _UNITS.get(text[-1:], 1) if text else 1
    if factor > 1:
        text = text[:-1]
    try:
        count = int(text)
    except ValueError as exc:
        raise ConfigurationError(
            f"backend option {name!r} must be a byte count, optionally suffixed k/m/g "
            f"(got {value!r})"
        ) from exc
    if count < 0:
        raise ConfigurationError(f"backend option {name!r} must not be negative (got {value!r})")
    return count * factor


#: The spellings a launch has for "yes" and "no". A flag arrives as a JSON ``true``, as the string
#: ``"true"`` out of a shell, or as ``1``; all three are the same request, and so are the negatives.
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})


class Kind(str, Enum):
    """What a value has to be for one option.

    Deliberately small. A kind is a *shape*, and the shapes a launch has are the ones a shell can
    write: a string, a whole number, a truth value, a byte count with an optional suffix. Anything
    narrower than that -- "one of these two words", "at least one" -- is a property of the
    particular option and lives on :class:`BackendOption` beside it, because encoding it as a kind
    would mean one kind per constraint and a decoder to go with each.
    """

    STRING = "string"
    INTEGER = "integer"
    FLAG = "flag"
    BYTES = "bytes"


class Group(str, Enum):
    """Which part of a run an option belongs to, for ``--help``.

    vLLM registers every flag inside ``parser.add_argument_group(title="CacheConfig")`` and SGLang
    groups by the file a field is declared in; neither puts the group in the flag's *name*. This is
    the same thing: the section a generated flag is listed under, and the answer to "*whose* lever
    is this" -- which is the one question a runtime prefix would otherwise be answering badly.

    A closed set rather than a free string, because two spellings of one section is a section split
    in half, and that is not a mistake a test can see in a help page nobody diffs.
    """

    DEVICE = "Device"
    EXPERT = "Expert arena"
    PREFILL = "Prefill"
    PREFIX_CACHE = "Prefix cache"
    DECODE = "Decode"
    KERNELS = "Kernels"
    EXECUTION = "Execution"
    MODEL = "Model and tokenizer"


@dataclass(frozen=True)
class BackendOption:
    """One ``backend_options`` key, as the runtime that reads it declares it.

    ``default`` is stated here *and* on the adapter's own ``_Options`` dataclass. That is not two
    sources of truth when both name the same constant -- the convention this module's callers
    follow, and what ``tests/test_declared_options.py`` checks -- and it is the only way a
    declaration can render ``--help`` without importing the adapter it describes.

    A ``None`` is *unset* rather than a value: it passes through :meth:`decode` untouched, which is
    what every adapter already does with a key it popped and found empty. It is also how an option
    whose real default is derived at load time says so.
    """

    name: str
    kind: Kind
    default: Any
    help: str
    #: The ``--help`` section this option is listed under. See :class:`Group`.
    group: Group = Group.EXECUTION
    #: How this runtime gets a value when nothing names one, for the options whose answer is not a
    #: constant -- a measurement, the card's free memory, another flag of the launch's. Only
    #: meaningful while ``default`` is ``None``, and there so that ``--help`` can say what an unset
    #: option becomes instead of printing ``None``, which is not a value anybody can type.
    resolution: str | None = None
    #: The runtimes that read this option, for a declaration in
    #: :mod:`pocketllm.backends.shared_options`. Empty for a runtime-private option, whose reader is
    #: the module it is declared in. It is what lets a test ask whether a shared concept's readers
    #: agree, and what U2b-2 asks when a launch names an option the selected runtime does not read.
    readers: tuple[str, ...] = ()
    #: Older spellings of the same key, still accepted. A launch script that uses one keeps working.
    #: Naming both the canonical key and an alias is refused rather than silently resolved by order.
    aliases: tuple[str, ...] = ()
    #: The values this option accepts, or empty for any. Case-sensitive: these are the values the
    #: adapter compares against, and accepting a different case here would only move the failure.
    #: Checked after the kind, and not reached at all for an unset value -- see :meth:`decode`.
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Every spelling this option answers to, canonical first."""
        return (self.name, *self.aliases)

    @property
    def bounded(self) -> bool:
        """Whether :attr:`minimum` / :attr:`maximum` apply to this option's kind."""
        return self.kind in {Kind.INTEGER, Kind.BYTES}

    def decode(self, value: Any) -> Any:
        """``value`` as the adapter wants it, or a :class:`ConfigurationError` naming this key."""
        if value is None:
            return None
        if self.kind is Kind.FLAG:
            decoded: Any = _flag(value, self.name)
        elif self.kind is Kind.INTEGER:
            decoded = _integer(value, self.name)
        elif self.kind is Kind.BYTES:
            decoded = byte_size(value, self.name)
        else:
            decoded = str(value)
        if self.choices and decoded not in self.choices:
            raise ConfigurationError(
                f"backend option {self.name!r} is one of {_quoted(self.choices)}, got {decoded!r}"
            )
        if self.bounded:
            self._check_bounds(decoded, value)
        return decoded

    def _check_bounds(self, decoded: float, original: Any) -> None:
        if self.minimum is not None and decoded < self.minimum:
            raise ConfigurationError(
                f"backend option {self.name!r} {lower_bound(self.minimum)}, got {original!r}"
            )
        if self.maximum is not None and decoded > self.maximum:
            raise ConfigurationError(
                f"backend option {self.name!r} must be <= {_number(self.maximum)}, "
                f"got {original!r}"
            )

    def describe(self) -> str:
        """One ``--help`` block: the key, its type and default, and what it does."""
        detail = [self.kind.value]
        if self.choices:
            detail.append("one of " + ", ".join(self.choices))
        if self.bounded and self.minimum is not None:
            detail.append(lower_bound_detail(self.minimum))
        if self.bounded and self.maximum is not None:
            detail.append(f"<= {_number(self.maximum)}")
        rendered = f"  {self.name} ({', '.join(detail)}; default {self.rendered_default()})"
        if self.aliases:
            rendered += f"\n      also spelled {', '.join(self.aliases)}"
        return f"{rendered}\n      {self.help}"

    def rendered_default(self) -> str:
        """The default as ``--help`` shows it.

        A byte budget is rendered the way a launch writes it -- ``4g`` rather than ``4294967296`` --
        because the option takes the suffix and the number is the part nobody reads. An option whose
        runtime resolves it says so instead of printing ``None``, which is not a value anybody can
        type: the sentence is what a reader needs, and the code is not.
        """
        if self.resolution and self.default is None:
            return f"unset (resolved: {self.resolution})"
        if self.kind is Kind.BYTES and isinstance(self.default, int) and self.default > 0:
            for suffix, factor in sorted(_UNITS.items(), key=lambda item: -item[1]):
                if self.default % factor == 0:
                    return f"{self.default // factor}{suffix}"
        return repr(self.default)

    def __post_init__(self) -> None:
        if not isinstance(self.group, Group):
            raise ConfigurationError(
                f"backend option {self.name!r} names the --help group {self.group!r}; groups are "
                f"{', '.join(member.value for member in Group)}"
            )
        if self.resolution is not None and self.default is not None:
            raise ConfigurationError(
                f"backend option {self.name!r} declares both the default {self.default!r} and a "
                f"resolution ({self.resolution!r}); a value the runtime computes is not a default"
            )


def lower_bound(minimum: float) -> str:
    """A lower bound as the predicate a refusal states: ``must be >= 1``.

    A bound of zero is the one with a name in English -- ``must not be negative`` -- and it is what
    the messages this replaced said, so a launch that trips it reads exactly as it did before.
    """
    if minimum == 0:
        return "must not be negative"
    return f"must be >= {_number(minimum)}"


def lower_bound_detail(minimum: float) -> str:
    """The same bound in a ``--help`` column, where the line has no room for the ``must``."""
    return "not negative" if minimum == 0 else f">= {_number(minimum)}"


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        # `True` is an `int` in Python, and "true" for a count is a typo rather than a width.
        raise ConfigurationError(
            f"backend option {name!r} is a whole number, got the flag {value!r}"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"backend option {name!r} is a whole number, got {value!r}") from exc


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    raise ConfigurationError(f"backend option {name!r} is a flag and {value!r} is not one")


def _number(value: float) -> str:
    """A bound as it would be written, without the ``.0`` an integer-valued float carries."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _quoted(values: Iterable[str]) -> str:
    return " or ".join(repr(value) for value in values)


def canonical_names(declarations: Iterable[BackendOption]) -> dict[str, str]:
    """Every spelling in ``declarations`` mapped to its canonical name."""
    return {spelling: option.name for option in declarations for spelling in option.names}


def decode_options(
    declarations: Iterable[BackendOption],
    values: Mapping[str, Any] | None,
    *,
    runtime: str,
    ignored: Iterable[str] = (),
    resolved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``values`` decoded against ``declarations``, with the defaults filled in.

    Three sources decide an option's value, and they are ordered because a launch can name more
    than one of them:

    1. ``values`` -- the ``--backend-option KEY=VALUE`` spellings this call is handed.
    2. ``resolved`` -- what the *host* already decided: a flag of the command line's own, which is
       the concept by a second name. The flag lives above the runtimes, so it is passed in rather
       than read here, and this is where a runtime says which of its options that flag answers.
    3. the declaration's ``default``.

    The order is the same rule the whole surface follows -- the more specific spelling wins -- and
    it is why ``--backend-option`` is an escape hatch rather than a second opinion: naming an option
    there beats naming its flag, exactly as it did before the flags existed.

    The returned mapping covers every declared option, so a caller can hand it straight to its
    ``_Options`` dataclass and read the result the way it reads any other instance. The refusals
    this makes are the ways a launch can name an option wrongly:

    * a value the declared kind cannot read -- the whole of the type checking;
    * a key that is one option's alias and its canonical name at once -- which of the two is meant
      is not knowable from the values, so resolving it by iteration order would be a silent one;
    * a key no runtime declares, named with the ones this runtime knows listed, because a tuning
      option that silently does nothing is how a run gets measured on the wrong lever.

    ``ignored`` names the keys every launch carries and no *model* options parser has a use for --
    see :data:`pocketllm.backends.capabilities.IGNORED_OPTIONS` for why they are accepted rather
    than refused. They are dropped before the check and do not appear in the result.
    """
    supplied = dict(values or {})
    for name in ignored:
        supplied.pop(name, None)
    spellings = canonical_names(declarations)
    unknown = sorted(set(supplied) - set(spellings))
    if unknown:
        raise ConfigurationError(
            f"backend={runtime!r} has no option {unknown[0]!r}; it knows "
            f"{', '.join(sorted(option.name for option in declarations))}"
        )
    resolved = dict(resolved or {})
    undeclared = sorted(set(resolved) - {option.name for option in declarations})
    if undeclared:
        # Not a launch's mistake, so not the launch's message: the host resolved a concept this
        # runtime does not read, and silently dropping it would make the host's answer a no-op.
        raise ConfigurationError(
            f"backend={runtime!r} was handed a resolved value for {undeclared[0]!r}, which it does "
            "not declare; the host and the runtime disagree about what this option is"
        )
    resolved_set: dict[str, tuple[str, Any]] = {}
    for spelling, value in supplied.items():
        canonical = spellings[spelling]
        if canonical in resolved_set:
            raise ConfigurationError(
                f"backend={runtime!r} was given {canonical!r} twice, as "
                f"{resolved_set[canonical][0]!r} and {spelling!r}; they are the same option and only "
                "one can win"
            )
        resolved_set[canonical] = (spelling, value)
    decoded: dict[str, Any] = {}
    for option in declarations:
        if option.name in resolved_set:
            source = resolved_set[option.name][1]
        elif option.name in resolved and resolved[option.name] is not None:
            source = resolved[option.name]
        else:
            source = option.default
        decoded[option.name] = option.decode(source)
    return decoded
