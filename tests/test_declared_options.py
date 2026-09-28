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
from pocketllm.backends import mimo_backend, v41_backend, xing4_backend
from pocketllm.backends.options import BackendOption, Kind, decode_options

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
    assert declared["deal"].aliases == ("expert_deal",)

    renamed = mimo_backend._Options.from_args(
        EngineArgs(model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"expert_rows": 8})
    )
    canonical = mimo_backend._Options.from_args(
        EngineArgs(model="a-mimo-v2-checkpoint", backend="mimo", backend_options={"chunk_rows": 8})
    )

    assert renamed.chunk_rows == canonical.chunk_rows == 8


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
