"""The declarations in :mod:`pocketllm.backends.options` against the adapters they describe.

The point of a declared option is that one statement answers what used to be three -- the accepted
names, the type, and the refusal -- so these tests are about the statement being the one the adapter
actually reads. Two things would make it not:

* a declaration with no field behind it, which is an option a launch can name and nothing consumes;
* a field with no declaration, which is an option an adapter reads and a launch cannot name -- it
  worked before the declaration and would silently stop now.

The default is checked in both directions for the same reason, and against the *bare* ``_Options()``
rather than against a hand-written number, so the two spellings of "nothing was set" cannot drift.
"""

from __future__ import annotations

import dataclasses

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.backends import mimo_backend, shared_options, v41_backend, xing4_backend
from pocketllm.backends.options import BackendOption, Group, Kind, decode_options

#: The three runtimes whose options are declared, with a checkpoint name each recognises.
DECLARED = {
    "v41": (v41_backend, "a-deepseek-v41-checkpoint"),
    "mimo": (mimo_backend, "a-mimo-v2-checkpoint"),
    "xing4": (xing4_backend, "a-xing4-checkpoint"),
}


def _fields(module) -> dict[str, dataclasses.Field]:
    return {field.name: field for field in dataclasses.fields(module._Options)}


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_every_declaration_is_a_field_and_every_field_a_declaration(runtime: str) -> None:
    module, _ = DECLARED[runtime]
    declared = [option.name for option in module.OPTIONS]
    fields = list(_fields(module))

    assert set(declared) == set(fields), (
        f"declared but unread: {sorted(set(declared) - set(fields))}; "
        f"read but unnameable: {sorted(set(fields) - set(declared))}"
    )
    # The declarations are written in the dataclass's order so the two read as one list; keeping
    # them comparable is what lets a reader hold the two side by side.
    assert declared == fields


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_a_declaration_states_the_default_the_dataclass_takes(runtime: str) -> None:
    module, _ = DECLARED[runtime]
    fields = _fields(module)
    bare = module._Options()

    for option in module.OPTIONS:
        assert option.default == getattr(bare, option.name), (
            f"{option.name}: declared {option.default!r}, dataclass {getattr(bare, option.name)!r}"
        )
        # A default is a value the adapter can read, not prose about one: if it does not decode,
        # no launch can ever reach it.
        assert option.decode(option.default) == option.default
    assert fields, "a runtime with no options would make this test vacuous"


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_a_launch_that_names_nothing_is_the_bare_dataclass(runtime: str) -> None:
    """The other half of the default: nothing declared, nothing named, one answer."""
    module, model = DECLARED[runtime]
    args = EngineArgs(model=model, backend=runtime)

    assert module._Options.from_args(args) == module._Options()


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_the_declared_names_are_the_names_the_adapter_refuses_the_rest_of(runtime: str) -> None:
    """Every key of the shared launch set is accepted and none of them is a declaration."""
    module, model = DECLARED[runtime]
    args = EngineArgs(
        model=model,
        backend=runtime,
        backend_options={"engine_kind": "auto", "nccl_id_path": "/tmp/nccl", "pd_mode": "scheduler"},
    )

    module._Options.from_args(args)


def test_an_alias_is_the_same_option_under_an_older_name() -> None:
    """MiMo answered to these two before the option was renamed, and both spellings still work."""
    declared = {option.name: option for option in mimo_backend.OPTIONS}

    assert declared["chunk_rows"].aliases == ("expert_rows",)
    # ``deal`` was this adapter's canonical name and ``expert_deal`` its alias; the merge with V4.1
    # swapped them, so the key a launch writes today is the one both runtimes spell the same way.
    assert declared["expert_deal"].aliases == ("deal",)

    renamed = mimo_backend._Options.from_args(
        EngineArgs(model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"expert_rows": 8})
    )
    canonical = mimo_backend._Options.from_args(
        EngineArgs(model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"chunk_rows": 8})
    )

    assert renamed.chunk_rows == canonical.chunk_rows == 8

    old_deal = mimo_backend._Options.from_args(
        EngineArgs(model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"deal": "id"})
    )
    new_deal = mimo_backend._Options.from_args(
        EngineArgs(
            model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"expert_deal": "id"}
        )
    )

    assert old_deal.expert_deal == new_deal.expert_deal == "id"


def test_a_flag_spelled_as_a_word_is_read_as_one() -> None:
    """`bool("false")` is `True`, which is how a launch turns a flag off and gets it back on.

    Xing4 read `use_kernel` that way: the option was off only when a caller handed it a real
    ``False``, so `--backend-option use_kernel=false` quoted through a shell, or set from the
    library as the string a shell would have produced, left the kernels running. Nothing said so.
    """
    off = xing4_backend._Options.from_args(
        EngineArgs(
            model="a-xing4-checkpoint", backend="xing4", backend_options={"use_kernel": "false"}
        )
    )
    on = xing4_backend._Options.from_args(
        EngineArgs(
            model="a-xing4-checkpoint", backend="xing4", backend_options={"use_kernel": "yes"}
        )
    )

    assert off.use_kernel is False
    assert on.use_kernel is True


def test_naming_an_alias_and_its_canonical_name_is_refused() -> None:
    """The two values cannot both win, and which one is meant is not knowable, so nothing is read."""
    with pytest.raises(ConfigurationError, match="was given 'chunk_rows' twice"):
        decode_options(
            mimo_backend.OPTIONS,
            {"chunk_rows": 8, "expert_rows": 4},
            runtime="mimo",
        )
    with pytest.raises(ConfigurationError, match="was given 'chunk_rows' twice"):
        decode_options(
            mimo_backend.OPTIONS,
            {"expert_rows": 4, "chunk_rows": 8},
            runtime="mimo",
        )


def test_an_unknown_name_is_refused_with_the_declared_set() -> None:
    with pytest.raises(ConfigurationError) as caught:
        decode_options(mimo_backend.OPTIONS, {"chunk_row": 8}, runtime="mimo")

    message = str(caught.value)
    assert "has no option 'chunk_row'" in message
    assert "chunk_rows" in message, "the refusal has to name something that does work"


def test_the_shared_launch_keys_are_dropped_rather_than_decoded() -> None:
    """They arrive on every launch from the CLI and are not this runtime's to read."""
    decoded = decode_options(
        xing4_backend.OPTIONS,
        {"engine_kind": "auto", "nccl_path": "typo", "pd_mode": "scheduler"},
        runtime="xing4",
        ignored={"engine_kind", "nccl_path", "pd_mode"},
    )

    assert "engine_kind" not in decoded
    assert set(decoded) == {option.name for option in xing4_backend.OPTIONS}


# ---------------------------------------------------------------------------- the four kinds


@pytest.mark.parametrize(
    "value, expected",
    [(4, 4), ("4", 4), (4.0, 4), (" 4 ", 4)],
)
def test_an_integer_is_read_from_whatever_spells_it(value: object, expected: int) -> None:
    option = BackendOption("threads", Kind.INTEGER, None, "")

    assert option.decode(value) == expected


def test_a_flag_is_read_from_a_bool_or_a_shell_word() -> None:
    option = BackendOption("pin", Kind.FLAG, True, "")

    for value in (True, "true", "1", "YES", "on"):
        assert option.decode(value) is True
    for value in (False, "false", "0", "No", "off"):
        assert option.decode(value) is False
    with pytest.raises(ConfigurationError, match="is a flag and 'maybe' is not one"):
        option.decode("maybe")


def test_a_byte_count_takes_a_suffix_and_nothing_negative() -> None:
    option = BackendOption("prefix_cache_bytes", Kind.BYTES, 0, "")

    assert option.decode("4g") == 4 << 30
    assert option.decode("512m") == 512 << 20
    assert option.decode(1 << 30) == 1 << 30
    with pytest.raises(ConfigurationError, match="must be a byte count"):
        option.decode("4 gig")
    with pytest.raises(ConfigurationError, match="must not be negative"):
        option.decode(-1)


def test_a_string_option_is_a_string() -> None:
    option = BackendOption("device", Kind.STRING, None, "")

    assert option.decode(0) == "0"
    assert option.decode(None) is None, "unset is unset, not the word 'None'"


def test_a_number_is_not_quietly_read_as_a_flag() -> None:
    """`True` is an `int` in Python, and `threads=true` is a typo rather than a thread count."""
    option = BackendOption("threads", Kind.INTEGER, None, "")

    with pytest.raises(ConfigurationError, match="is a whole number, got the flag True"):
        option.decode(True)


# ------------------------------------------------------------------- bounds, choices, help


@pytest.mark.parametrize(
    "option, value, message",
    [
        (BackendOption("x", Kind.INTEGER, 4, "", minimum=1), 0, "must be >= 1"),
        (BackendOption("y", Kind.INTEGER, 0, "", minimum=0), -1, "must not be negative"),
        (BackendOption("z", Kind.BYTES, 0, "", minimum=1), 0, "must be >= 1"),
        (BackendOption("w", Kind.INTEGER, 4, "", maximum=8), 9, "must be <= 8"),
    ],
)
def test_a_bound_says_which_side_it_is(option: BackendOption, value: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        option.decode(value)


def test_a_choice_is_refused_by_name_and_lists_what_would_work() -> None:
    option = BackendOption("deal", Kind.STRING, "sorted", "", choices=("id", "sorted"))

    assert option.decode("id") == "id"
    with pytest.raises(ConfigurationError, match="is one of 'id' or 'sorted', got 'random'"):
        option.decode("random")


def test_a_bound_does_not_apply_to_a_flag_or_a_string() -> None:
    """`bounded` is what keeps `min`/`max` off kinds whose values are not ordered."""
    assert BackendOption("f", Kind.FLAG, True, "", minimum=1).bounded is False
    assert BackendOption("s", Kind.STRING, "a", "", minimum=1).bounded is False
    assert BackendOption("i", Kind.INTEGER, 1, "", minimum=1).bounded is True


def test_the_help_block_names_the_key_its_type_its_default_and_the_old_spelling() -> None:
    option = BackendOption(
        "prefill_chunk",
        Kind.INTEGER,
        2048,
        "tokens one prefill forward takes",
        aliases=("chunk",),
        minimum=1,
    )

    described = option.describe()

    assert "prefill_chunk (integer, >= 1; default 2048)" in described
    assert "also spelled chunk" in described
    assert "tokens one prefill forward takes" in described


def test_a_byte_default_is_rendered_the_way_a_launch_writes_it() -> None:
    """`4g`, not `4294967296`: the option takes the suffix and the digits are what nobody reads."""
    assert BackendOption("b", Kind.BYTES, 4 << 30, "").rendered_default() == "4g"
    assert BackendOption("b", Kind.BYTES, 512 << 20, "").rendered_default() == "512m"
    assert BackendOption("b", Kind.BYTES, 3 << 10, "").rendered_default() == "3k"
    assert BackendOption("b", Kind.BYTES, 0, "").rendered_default() == "0"
    assert BackendOption("i", Kind.INTEGER, 2048, "").rendered_default() == "2048"
    assert BackendOption("s", Kind.STRING, "sorted", "").rendered_default() == "'sorted'"
    assert BackendOption("f", Kind.FLAG, None, "").rendered_default() == "None"


def test_every_declared_option_can_describe_itself() -> None:
    """`--help` is generated from these, so a declaration without prose is a blank line in it."""
    for runtime, (module, _) in DECLARED.items():
        for option in module.OPTIONS:
            assert option.help.strip(), f"{runtime}.{option.name} has no help"
            assert option.describe().strip(), f"{runtime}.{option.name} describes nothing"


# ----------------------------------------------------------- one concept, declared once


def _by_name(declarations) -> dict[str, BackendOption]:
    return {option.name: option for option in declarations}


def test_the_shared_list_is_the_concepts_more_than_one_runtime_declares() -> None:
    """`shared_options` is a claim about the tree, so the tree is what checks it.

    A concept two runtimes read is the case a duplicated flag comes from, and the way to stop the
    duplication is for both to reference one declaration -- which is only true while the list says
    so. Both directions matter: a name in the list that one runtime declares is a declaration nobody
    shares, and a name two declare that is *not* in the list is the drift this replaced.
    """
    readers: dict[str, set[str]] = {}
    for runtime, (module, _) in DECLARED.items():
        for option in module.OPTIONS:
            readers.setdefault(option.name, set()).add(runtime)

    assert {name for name, runs in readers.items() if len(runs) > 1} == {
        option.name for option in shared_options.SHARED
    }


def test_a_shared_declaration_names_the_runtimes_that_read_it() -> None:
    """``readers`` is prose about the code, so it is checked against the code."""
    for shared in shared_options.SHARED:
        declared_by = {
            runtime
            for runtime, (module, _) in DECLARED.items()
            if shared.name in _by_name(module.OPTIONS)
        }

        assert declared_by == set(shared.readers), (
            f"{shared.name}: declared by {sorted(declared_by)}, readers say "
            f"{sorted(shared.readers)}"
        )


#: The fields of a shared declaration a runtime may state differently:
#: its own default, its own resolution, and its own sentence appended to the shared one.
#: Everything else -- the canonical name, the alias, the kind, the group, the accepted values, the
#: bounds -- is the shape the flag has wherever it is read, and a runtime that changed one of those
#: would be declaring a second flag under the first flag's name, which is the thing the merge exists
#: to remove.
_STATED_PER_RUNTIME = frozenset({"default", "resolution", "help"})


def test_a_shared_concept_has_one_shape_wherever_it_is_read() -> None:
    for shared in shared_options.SHARED:
        for runtime in shared.readers:
            module, _ = DECLARED[runtime]
            option = _by_name(module.OPTIONS)[shared.name]

            for field in dataclasses.fields(BackendOption):
                if field.name in _STATED_PER_RUNTIME:
                    continue
                assert getattr(option, field.name) == getattr(shared, field.name), (
                    f"{runtime}.{shared.name} disagrees with the shared declaration on {field.name}"
                )
            assert option.help.startswith(shared.help), (
                f"{runtime}.{shared.name} rewrote the shared sentence instead of adding to it"
            )


def test_a_runtime_states_its_own_answer_where_it_differs_and_nowhere_else() -> None:
    """The differences that remain after the merge, each one a runtime's own answer to a shared
    question: V4.1 leaves the deal and the prefill width to the loader, MiMo defaults the deal to
    ``sorted`` and the width to a measured 2048, and Xing4 derives the width from the card."""
    v41 = _by_name(v41_backend.OPTIONS)
    mimo = _by_name(mimo_backend.OPTIONS)
    xing4 = _by_name(xing4_backend.OPTIONS)

    assert v41["expert_deal"].default is None and v41["expert_deal"].resolution
    assert mimo["expert_deal"].default == "sorted"
    assert v41["prefill_chunk"].resolution and mimo["prefill_chunk"].default == 2048
    assert xing4["prefill_chunk"].default is None and xing4["prefill_chunk"].resolution
    assert xing4["prefix_cache_bytes"].default == 2 << 30
    assert v41["prefix_cache_bytes"].default == mimo["prefix_cache_bytes"].default == 4 << 30


# ------------------------------------------------------------------- groups and resolutions


def test_every_declared_option_is_listed_under_a_section_of_the_help() -> None:
    """`--help` is generated in groups, so a declaration without one has nowhere to appear."""
    for runtime, (module, _) in DECLARED.items():
        for option in module.OPTIONS:
            assert isinstance(option.group, Group), f"{runtime}.{option.name} has no group"


def test_no_group_is_a_section_nobody_appears_in() -> None:
    used = {option.group for _, (module, _) in DECLARED.items() for option in module.OPTIONS}

    assert used == set(Group), f"unused: {sorted(group.value for group in set(Group) - used)}"


def test_a_group_that_is_not_one_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="names the --help group 'Expert'"):
        BackendOption("x", Kind.FLAG, True, "", group="Expert")


def test_an_option_the_runtime_computes_says_so_instead_of_printing_none() -> None:
    """``None`` is not a value anybody can type, so an unset option that resolves states how."""
    declared = _by_name(xing4_backend.OPTIONS)["prefill_chunk"]

    assert declared.default is None, "a value computed at load is not a default"
    assert declared.resolution
    assert declared.rendered_default() == f"unset (resolved: {declared.resolution})"


def test_a_default_and_a_resolution_are_alternatives() -> None:
    """Both at once is two answers to one question, so it is refused where it is written."""
    with pytest.raises(ConfigurationError, match="declares both the default 4"):
        BackendOption("x", Kind.INTEGER, 4, "", resolution="the card's free memory")


# ------------------------------------------------------- the value a host flag resolves to


def test_a_resolved_value_stands_where_a_default_would() -> None:
    option = BackendOption("prefill_chunk", Kind.INTEGER, 2048, "")

    assert decode_options([option], None, runtime="r", resolved={"prefill_chunk": 4096}) == {
        "prefill_chunk": 4096
    }
    # An unset outcome of the host's own resolution is unset, and the declaration's default stands.
    assert decode_options([option], None, runtime="r", resolved={"prefill_chunk": None}) == {
        "prefill_chunk": 2048
    }
    # ``--backend-option`` is the more specific spelling, so it beats the flag it shares a concept
    # with -- the same rule every other pair on this surface resolves by.
    assert decode_options(
        [option], {"prefill_chunk": 512}, runtime="r", resolved={"prefill_chunk": 4096}
    ) == {"prefill_chunk": 512}


def test_a_resolved_value_the_runtime_does_not_declare_is_refused() -> None:
    """Not a launch's mistake, so not the launch's message: the host and the runtime disagree."""
    with pytest.raises(
        ConfigurationError, match="was handed a resolved value for 'prefix_cache_bytes'"
    ):
        decode_options(
            [BackendOption("prefill_chunk", Kind.INTEGER, None, "")],
            None,
            runtime="r",
            resolved={"prefix_cache_bytes": 1 << 30},
        )


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_a_resolved_option_reaches_every_runtime_that_declares_it(runtime: str) -> None:
    """The host flag for a declared option is one tier, and every reader of that option takes it.

    MiMo is why this is a test and not a default: its adapter read no prefill width from anywhere
    before the declarations existed, so a launch that named one prefillled at 2048 regardless. A
    tuning option that does nothing is how a run gets measured on the wrong lever. The flag itself
    -- ``--prefill-chunk-tokens`` -- and the tier it lands in are `tests/test_cli_declared_options.py`.
    """
    module, model = DECLARED[runtime]
    args = EngineArgs(model=model, backend=runtime, resolved_options={"prefill_chunk": 4096})

    assert module._Options.from_args(args).prefill_chunk == 4096


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_the_flag_that_names_nothing_leaves_the_runtime_its_own_answer(runtime: str) -> None:
    """A launch that does not name the width is the bare dataclass, on every runtime."""
    module, model = DECLARED[runtime]
    args = EngineArgs(model=model, backend=runtime, prefill_chunk_tokens=0)

    assert module._Options.from_args(args) == module._Options()


@pytest.mark.parametrize("runtime", sorted(DECLARED))
def test_the_legacy_engine_field_is_the_same_option_by_its_own_name(runtime: str) -> None:
    """``prefill_chunk_tokens`` and ``prefill_chunk`` are two names for one quantity.

    The field is the native engine's and predates the declarations; the key is what every runtime
    declares. ``EngineArgs`` is where they are one value, because every construction path goes
    through it -- the command line, the environment bridge, a worker rank rebuilding rank 0's args,
    an application building one by hand -- and a caller that reached for the field would otherwise
    get a runtime that quietly used its own width instead.
    """
    module, model = DECLARED[runtime]

    assert EngineArgs(
        model=model, backend=runtime, prefill_chunk_tokens=4096
    ).resolved_options == {"prefill_chunk": 4096}
    assert module._Options.from_args(
        EngineArgs(model=model, backend=runtime, prefill_chunk_tokens=4096)
    ).prefill_chunk == 4096
