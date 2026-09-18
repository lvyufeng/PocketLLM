"""The V4.1 launcher's arena default: what it is, and where it turns into the off state.

`--expert-pool-rows` is the one flag in this launcher whose default is a size, and two things follow
from that which no measurement on the cards can hold still: the number itself, and the rule that it
is a size *on the device path only* -- a host run has no arena, so the flag has to resolve to the off
state there rather than to a number the loader warns about and then ignores. Both are asserted here.

The numbers behind the default are in `src/cli/generate_v41.py`'s module docstring and
`docs/performance/deepseek_v4_1_flash_device_experts.md`; what these tests pin is that the launcher
still asks for them, and that `--expert-pool-rows 0` is still reachable, since it is the control
column every pooled number on that page was read against.
"""

from __future__ import annotations

from src.cli.generate_v41 import build_arg_parser, resolve_pool_rows

# The width the default, the batched path's end-to-end number and the 512-token pool sweeps were all
# taken at. Pinned as a literal rather than imported so that changing it means changing this line
# too, which is the point: it decides both mechanisms at once.
DEFAULT_POOL_ROWS = 288


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
