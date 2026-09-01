from __future__ import annotations

import pytest

from pocketllm.cli import _args, build_parser


def _parse(*extra: str):
    return build_parser().parse_args(["serve", "--model", "checkpoint", *extra])


def test_cli_maps_common_engine_fields() -> None:
    args = _args(_parse(
        "--backend", "torch",
        "--tensor-parallel-size", "2",
        "--max-model-len", "16384",
        "--kv-cache-dtype", "fp8",
        "--prefill-chunk-tokens", "4096",
        "--no-enable-prefix-caching",
    ))

    assert args.backend == "torch"
    assert args.tensor_parallel_size == 2
    assert args.max_model_len == 16384
    assert args.kv_cache_dtype == "fp8"
    assert args.prefill_chunk_tokens == 4096
    assert args.enable_prefix_caching is False


def test_cli_exposes_attention_window_and_speculation() -> None:
    args = _args(_parse(
        "--attention-window", "4096",
        "--attention-sink-tokens", "128",
        "--speculative-method", "mtp",
        "--speculative-tokens", "3",
    ))

    assert args.attention_window == 4096
    assert args.attention_sink_tokens == 128
    assert args.speculative_method == "mtp"
    assert args.speculative_tokens == 3


def test_backend_option_values_are_json_typed() -> None:
    args = _args(_parse(
        "--backend-option", "max_state_snapshots=16",
        "--backend-option", "mtp_adaptive=true",
        "--backend-option", "dspark_checkpoint=/models/drafter",
    ))

    assert args.backend_options["max_state_snapshots"] == 16
    assert args.backend_options["mtp_adaptive"] is True
    # A bare path must survive as a string rather than failing JSON parsing.
    assert args.backend_options["dspark_checkpoint"] == "/models/drafter"
    # Defaults stay present alongside explicit overrides.
    assert args.backend_options["engine_kind"] == "qwen"


def test_backend_option_requires_key_value_form() -> None:
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _args(_parse("--backend-option", "bare-flag"))


def test_served_model_name_defaults_to_model_path() -> None:
    namespace = _parse()
    assert namespace.served_model_name is None
    named = _parse("--served-model-name", "qwen-local")
    assert named.served_model_name == "qwen-local"
