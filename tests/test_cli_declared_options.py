"""The flags the option declarations generate, and the refusal for one nothing reads.

U2b-2 of #447. `tests/test_declared_options.py` holds each runtime's declarations to its ``_Options``
dataclass; this is the other end of the same statement -- the command line built from those
declarations -- and the two failures it exists for are the ones a flat namespace brings:

* **a flag that is not the declaration's name**, which is how a merge turns back into two flags;
* **a flag the selected runtime does not read**, which is a tuning option that silently does
  nothing, and the reason one namespace is safe at all.

The parser is the real one. There is no second list of flags here to keep in step with it: the tests
read ``build_parser()``'s own actions, so a flag added by hand shows up as a declaration nothing
generated and a generated flag shows up against the declaration it came from.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.backends import cli_surface, factory
from pocketllm.backends.options import Group

#: The `serve` parser and its subparser, read the way a caller gets them.
PARSER = None


def _serve() -> argparse.ArgumentParser:
    """The ``serve`` subparser, which is where the generated flags are registered."""
    global PARSER
    if PARSER is None:
        PARSER = _build()
    return PARSER


def _build() -> argparse.ArgumentParser:
    from pocketllm.cli import build_parser

    return build_parser()._subparsers._group_actions[0].choices["serve"]


def _parse(*extra: str) -> argparse.Namespace:
    from pocketllm.cli import _args, build_parser

    namespace = build_parser().parse_args(["serve", "--model", "checkpoint", *extra])
    return _args(namespace)


def _grouped() -> dict[str, str]:
    """``{flag: the --help section it is listed under}`` for every generated flag.

    One entry per declaration: a flag declaration registers two spellings (``--pin`` and
    ``--no-pin``) and the first is the one the option is named by.
    """
    found: dict[str, str] = {}
    for section in _serve()._action_groups:
        for action in section._group_actions:
            if getattr(action, "option_strings", None) and action.dest == cli_surface.DECLARED_DEST:
                found[action.option_strings[0]] = section.title
    return found


def _helps() -> dict[str, str]:
    """``{flag: the text argparse prints beside it}`` for every generated flag."""
    return {
        action.option_strings[0]: action.help or ""
        for section in _serve()._action_groups
        for action in section._group_actions
        if getattr(action, "option_strings", None) and action.dest == cli_surface.DECLARED_DEST
    }


def _checkpoint(directory: pathlib.Path, model_type: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(directory)


# --------------------------------------------------------------- one flag per declared concept


def test_every_generated_flag_is_a_declaration_s_own_name():
    """``{flag} == cli_surface.cli_name(name)``: `--expert-pool-rows`, not `--v41-expert-pool-rows`.

    The name is the concept's, and the runtime that reads it is what the help section and the
    parenthetical say. A runtime prefix -- the shape this asserts against -- is what a second flag
    per runtime would be called, which is the collision the merge removed.
    """
    generated = cli_surface.generated()

    assert set(_grouped()) == {cli_surface.cli_name(name) for name in generated}
    assert len(_grouped()) == len(generated), "two declarations generated the same flag"


def test_a_declaration_the_host_already_spells_gets_no_second_flag():
    """The one place a flag's spelling is not its name, and it is a tie rather than a rename.

    ``--prefill-chunk-tokens`` is the native engine's field for the same quantity and predates the
    declarations; ``device`` is left alone for U3, because today's ``--device`` means the vendor on
    one path and the card on another. Both are one flag, and neither is generated.
    """
    flags = _grouped()
    host = _host_flags()

    for name in cli_surface.HOST_FLAGS:
        assert cli_surface.cli_name(name) not in flags
    for name in cli_surface.NO_FLAG:
        assert cli_surface.cli_name(name) not in flags
        assert cli_surface.cli_name(name) in host, "the name has to be reachable by *something*"


def test_no_generated_flag_collides_with_one_the_host_declares():
    """One namespace is only safe while every name in it means one thing.

    The generated surface is added last and could not overwrite an existing option string silently
    -- argparse refuses a duplicate -- but a *differently spelled* host flag for the same concept
    would sail through, and that is the shape of the four collisions this whole exercise is about.
    """
    overlap = set(_grouped()) & _host_flags()

    assert not overlap, f"generated flags that already exist: {sorted(overlap)}"


def test_every_group_a_declaration_names_is_a_section_of_the_help():
    """The section is the declaration's, so the set of sections is the set of groups in use."""
    sections = {section.title for section in _serve()._action_groups} - {"options", "positional arguments"}

    assert sections == {group.value for group in Group if _in_use(group)}


def _in_use(group: Group) -> bool:
    return any(
        cli_surface._shape(readers).group is group
        for readers in cli_surface.generated().values()
    )


def test_a_generated_flag_says_which_runtimes_read_it():
    """Neither the name nor the section answers it, so the help text has to.

    ``--expert-pool-rows`` and ``--chunk-rows`` are both the expert arena's and both unprefixed --
    the concept is the same one on either side of the runtime boundary -- so the readers are the
    only thing that tells an operator which is theirs.
    """
    for name, readers in cli_surface.generated().items():
        flag = cli_surface.cli_name(name)
        for runtime in readers:
            assert runtime in _helps()[flag], f"{flag} does not name {runtime}"


def test_a_shared_flag_says_what_each_runtime_answers_when_it_is_unset():
    """The difference the merge turned into a resolved default, which is only useful if it is stated.

    ``--prefix-cache-bytes`` is 4 GiB on two runtimes and 2 GiB on the third, and the operators of
    each are reading the same page.
    """
    helps = _helps()

    assert "4g on v41 and mimo; 2g on xing4" in helps["--prefix-cache-bytes"]
    assert "the loader's own deal on v41; 'sorted' on mimo" in helps["--expert-deal"]
    # A reader that answers the same thing as its neighbours says it once.
    assert "default 1024" in helps["--prefix-cache-head-tokens"]


def test_a_bound_choice_and_metavar_come_from_the_declaration():
    """The parser is given the declaration's shape, so a typo is caught before any runtime is built.

    ``choices`` in particular: refusing ``--expert-deal random`` here costs an argparse exit rather
    than a model load, and the decoder refuses it too, which is the tier a ``--backend-option``
    value takes.
    """
    choices = {action.option_strings[0]: action for action in _serve()._actions}

    assert choices["--expert-deal"].choices == ("sorted", "id")
    assert choices["--prefix-cache-bytes"].metavar == "BYTES"
    assert choices["--prefix-cache-bytes"].type is str
    assert choices["--expert-pool-rows"].type is int


# ------------------------------------------------------------------------ what a launch named


def test_a_flag_nobody_named_leaves_nothing_behind():
    """`None` is a value and ``SUPPRESS`` is not, and only the second can be told from "unset"."""
    args = _parse()

    assert args.resolved_options == {}


def test_the_mapping_holds_exactly_what_the_launch_named():
    args = _parse("--expert-pool-rows", "148", "--no-decode-graphs")

    assert args.resolved_options == {"expert_pool_rows": 148, "decode_graphs": False}


def test_the_host_flag_lands_in_the_same_mapping():
    """It is the same tier: the host decided it, and `--backend-option` still beats both."""
    args = _parse("--prefill-chunk-tokens", "4096")

    assert args.resolved_options == {"prefill_chunk": 4096}


def test_the_backend_option_still_wins_over_the_flag_it_shares_a_concept_with():
    args = _parse(
        "--prefix-cache-bytes", "2g",
        "--backend-option", "prefix_cache_bytes=1g",
        "--backend", "torch",
    )

    assert args.backend_options["prefix_cache_bytes"] == "1g"


def test_a_flag_reaches_every_runtime_that_declares_it():
    """One flag, one value, and each runtime's own answer for everything it did not name."""
    from pocketllm.backends import mimo_backend, v41_backend, xing4_backend

    args = _parse("--prefix-cache-bytes", "2g", "--prefill-chunk-tokens", "4096")

    for module in (v41_backend, mimo_backend, xing4_backend):
        options = module._Options.from_args(args)
        assert options.prefix_cache_bytes == 2 << 30
        assert options.prefill_chunk == 4096


def test_a_flag_two_runtimes_share_reaches_both():
    from pocketllm.backends import mimo_backend, v41_backend

    args = _parse("--prefix-cache-head-tokens", "512", "--expert-deal", "id")

    for module in (v41_backend, mimo_backend):
        options = module._Options.from_args(args)
        assert options.prefix_cache_head_tokens == 512
        assert options.expert_deal == "id"


def test_a_byte_flag_takes_the_suffix_the_declaration_takes():
    """One decoder for the flag and for ``--backend-option``, so ``4g`` means one thing."""
    from pocketllm.backends import v41_backend

    args = _parse("--prefix-cache-bytes", "4g")

    assert v41_backend._Options.from_args(args).prefix_cache_bytes == 4 << 30


# --------------------------------------------------------------- a flag nothing reads


def test_a_flag_the_selected_runtime_does_not_read_is_refused(tmp_path):
    """The failure this whole surface exists to prevent: a lever that is accepted and ignored.

    ``--expert-pool-rows`` is on the same command line as ``--chunk-rows`` because both runtimes
    share one parser, so the check is what makes that safe. It names the flag, and it names who
    does read it.
    """
    model = _checkpoint(tmp_path / "mimo", "mimo_v2")

    with pytest.raises(ConfigurationError, match="does not read --expert-pool-rows"):
        factory.select_backend(
            EngineArgs(model=model, backend="mimo", resolved_options={"expert_pool_rows": 148})
        )


def test_a_runtime_that_declares_no_options_reads_no_generated_flag():
    """`cpp` and `torch` take their tuning from fields and from flags of their own."""
    with pytest.raises(ConfigurationError, match="does not read --prefix-cache-bytes"):
        factory.select_backend(
            EngineArgs(
                model="a-qwen-checkpoint",
                backend="torch",
                resolved_options={"prefix_cache_bytes": "2g"},
            )
        )


def test_the_host_flag_is_exempt_because_a_runtime_of_its_own_reads_it():
    """``--prefill-chunk-tokens`` is the native engine's field; a native launch that names it is
    naming something that exists, and the declarations have nothing to do with the question."""
    assert cli_surface.unread_options(
        "cpp", EngineArgs(model="m", backend="cpp", resolved_options={"prefill_chunk": 4096})
    ) == []
    assert cli_surface.unread_options(
        "v41", EngineArgs(model="m", backend="v41", resolved_options={"prefill_chunk": 4096})
    ) == []


def test_the_refusal_names_the_runtime_that_does_read_the_flag(tmp_path):
    """``--pin`` is MiMo's, so it is a V4.1 launch that gets the refusal -- and the sentence says who
    does read it, which is the one thing an operator with the wrong flag cannot look up."""
    model = _checkpoint(tmp_path / "v41", "deepseek_v41")

    with pytest.raises(ConfigurationError) as caught:
        factory.select_backend(
            EngineArgs(model=model, backend="v41", resolved_options={"pin": False})
        )

    message = str(caught.value)
    assert "does not read --pin" in message
    assert "it is mimo's" in message


def test_a_backend_option_key_is_left_to_the_runtime_s_own_decoder(tmp_path):
    """An explicit key is not this check's business -- the adapter refuses it with its own message.

    That matters for a key that will never have a flag: the refusal has to keep naming the key the
    launch wrote, in the vocabulary the runtime declares.
    """
    from pocketllm.backends import mimo_backend

    args = EngineArgs(
        model=_checkpoint(tmp_path / "mimo", "mimo_v2"),
        backend="mimo",
        backend_options={"expert_pool_rows": 148},
    )

    factory.select_backend(args)  # not this check's refusal: nothing was resolved
    with pytest.raises(ConfigurationError, match="has no option 'expert_pool_rows'"):
        mimo_backend._Options.from_args(args)


def test_the_refusal_reaches_a_launch_through_the_parser(tmp_path):
    """End to end, the way an operator meets it: a flag, a checkpoint, and a selection."""
    from pocketllm.cli import _args, build_parser

    model = _checkpoint(tmp_path / "mimo", "mimo_v2")
    namespace = build_parser().parse_args(
        ["serve", "--model", model, "--backend", "mimo", "--expert-pool-rows", "148"]
    )

    with pytest.raises(ConfigurationError, match="does not read --expert-pool-rows"):
        factory.select_backend(_args(namespace))


def _host_flags() -> set[str]:
    """The option strings the ``serve`` parser declares outside the generated sections."""
    generated = set(_grouped())
    return {
        option
        for action in _serve()._actions
        for option in action.option_strings
        if option.startswith("--") and option not in generated
    }
