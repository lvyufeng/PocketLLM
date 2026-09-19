"""The V4.1 launcher's arena default: what it is, and where it turns into the off state.

`--expert-pool-rows` is the one flag in this launcher whose default is a size, and two things follow
from that which no measurement on the cards can hold still: the number itself, and the rule that it
is a size *on the device path only* -- a host run has no arena, so the flag has to resolve to the off
state there rather than to a number the loader warns about and then ignores. Both are asserted here.

The numbers behind the default are in `src/cli/generate_v41.py`'s module docstring and
`docs/performance/deepseek_v4_1_flash_device_experts.md`; what these tests pin is that the launcher
still asks for them, and that `--expert-pool-rows 0` is still reachable, since it is the control
column every pooled number on that page was read against.

`--expert-buffers` is the second size, and it is pinned for a different reason: its default is not a
tuning choice but the *decode* path's own rotation, the one every graphed-decode number was taken
with. A default that drifted upward to make a prefill faster would move the decode step under the
numbers that validated it, so 2 is asserted as a literal here and the layer's ring is the only place
allowed to consume a different one.

`--prompt-file` is neither a size nor a default, and it is here because of what the flag is *for*:
the prompt a chunked prefill exists to run. A single argv entry cannot exceed `MAX_ARG_STRLEN`, 128
KiB, so a 256K-token prompt cannot be passed as an argument at all and the file is the only route
in. The tests below pin the read and the mutual exclusion; the token count is the caller's business
and is measured on the cards.
"""

from __future__ import annotations

import inspect

import pytest

from src.cli.generate_v41 import build_arg_parser, resolve_pool_rows, resolve_prompt
from src.models.deepseek_v4_1.loader import load_backbone

# The width the default, the batched path's end-to-end number and the 512-token pool sweeps were all
# taken at. Pinned as a literal rather than imported so that changing it means changing this line
# too, which is the point: it decides both mechanisms at once.
DEFAULT_POOL_ROWS = 288

# The pinned staging ring the graphed decode was validated at. See the module docstring.
DEFAULT_BUFFERS = 2


def _defaults(*extra: str):
    return build_arg_parser().parse_args(["--checkpoint", "/checkpoint", *extra])


def test_the_pool_defaults_to_a_width_and_not_to_off() -> None:
    """0 is not a neutral default here: the batched prefill cannot run without a pool."""
    args = _defaults()

    assert args.expert_pool_rows == DEFAULT_POOL_ROWS, (
        "the pool is off by default again, which drops the batched path with it -- the batched call "
        "reads each arena row it is handed as one expert's weights for a whole chunk, so "
        "`DeviceRoutedExperts` refuses it at `pool_rows=0` and a default run silently falls back to "
        "a row a call"
    )
    assert args.expert_batched is True, (
        "the batched flag is on but its gate is the pool's width; if that width went back to 0 the "
        "flag would read as on and run off"
    )


def test_the_default_width_is_the_one_the_device_path_gets() -> None:
    assert resolve_pool_rows(DEFAULT_POOL_ROWS, "cuda:0") == DEFAULT_POOL_ROWS


def test_the_host_path_pools_nothing_however_wide_the_flag_is() -> None:
    """The control column: no `--expert-device` means no arena, and no arena means no pool rows."""
    assert resolve_pool_rows(DEFAULT_POOL_ROWS, None) == 0


def test_zero_survives_the_resolution_so_the_control_column_stays_reachable() -> None:
    assert resolve_pool_rows(0, "cuda:0") == 0


def test_the_staging_ring_defaults_to_the_rotation_decode_was_validated_at() -> None:
    """2, and the flag can only widen it: a ring of one is a serial pipeline, not a control."""
    assert _defaults().expert_buffers == DEFAULT_BUFFERS, (
        "the staging ring's default moved, and it is the width the graphed decode step was measured "
        "at -- a wider ring is a prefill question and has to be asked with the flag, not assumed"
    )
    assert _defaults("--expert-buffers", "4").expert_buffers == 4
    assert _defaults("--expert-buffers", "1").expert_buffers == 1


def test_the_ring_width_reaches_the_layer_that_builds_it() -> None:
    """The flag is only real if `load_backbone` hands it on; the ring itself is sized in the layer.

    Nothing here drives a layer: `DeviceRoutedExperts` allocates page-locked memory and CUDA events
    in its constructor, so which slot `_take_buffer` hands out on the third working row is measured
    on the cards (`/tmp/probe_v41_prefill_phases.py`) and not asserted from a CPU test. What can be
    held still anywhere is that the two ends of the plumbing agree on the name and the default.
    """
    default = inspect.signature(load_backbone).parameters["expert_buffers"].default
    assert default == DEFAULT_BUFFERS
    assert "pinned_buffers=expert_buffers" in inspect.getsource(load_backbone), (
        "`load_backbone` takes the ring width and does not pass it to `DeviceRoutedExperts`, so "
        "`--expert-buffers` parses and is then silently ignored"
    )


def test_the_prompt_defaults_to_the_short_one_and_can_be_given_as_an_argument() -> None:
    assert resolve_prompt(None, None) == "The capital of France is"
    assert resolve_prompt("The capital of France is", None) == "The capital of France is"
    assert resolve_prompt("hello", None) == "hello"


def test_the_prompt_file_carries_what_an_argument_cannot(tmp_path) -> None:
    """The flag's whole reason: a prompt past `MAX_ARG_STRLEN` has no other way in.

    A 256K-token prompt is several megabytes of text, so the file is not a convenience -- writing the
    prompt into an argument is refused by the kernel at 128 KiB (`MAX_ARG_STRLEN`, 32 pages) with an
    `E2BIG` that names `execve` and not the flag. The length asserted here is past that page count on
    purpose, since a file route that worked only up to an argument's size would be no route at all.
    """
    assert len(resolve_prompt(None, _prompt_file(tmp_path, "x" * (32 * 4096 + 1)))) == 32 * 4096 + 1
    assert resolve_prompt(None, _prompt_file(tmp_path, "line one\nline two\n")) == "line one\nline two\n"


def _prompt_file(tmp_path, text: str) -> str:
    path = tmp_path / "prompt.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_two_ways_of_giving_a_prompt_are_mutually_exclusive() -> None:
    """Both at once is a run that silently used one of them, which is the failure to refuse.

    `main` reads the file when it is set, so `--prompt` beside it would parse, be dropped, and leave
    the caller thinking they had overridden a file they had not. `argparse` refuses the pair instead.
    """
    with pytest.raises(SystemExit):
        _defaults("--prompt", "a", "--prompt-file", "/tmp/prompt.txt")


def test_a_prompt_file_reaches_the_parser_under_its_own_name() -> None:
    assert _defaults("--prompt-file", "/tmp/prompt.txt").prompt_file == "/tmp/prompt.txt"
    assert _defaults("--prompt-file", "/tmp/prompt.txt").prompt is None
    assert _defaults().prompt is None, (
        "a parser default here would be indistinguishable from a caller's `--prompt` in "
        "`resolve_prompt`, which is what makes the two flags mutually exclusive rather than ordered"
    )
