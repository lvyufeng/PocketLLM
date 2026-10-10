"""The serving adapter over the S600 delegate.

Two kinds of test, and the split is deliberate.  The *refusal* and *resolution*
logic is pure host code — it decides what to reject and where the delegate's
three inputs are — so it runs anywhere and is checked on every host.  The
*behaviour* needs the delegate, the SDK and a `.hbm`, and skips without them; a
skip is not a pass.

The refusal tests carry the most weight, because getting them wrong is silent in
the direction that matters: a sampling field accepted but not applied returns a
200 whose text does not match the request, and no client can tell.  The accept
side is tested just as hard as the refuse side, because refusing a client that
spelled out the delegate's own behaviour would break every caller who sends
`temperature: 0` for the default.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.server import xlm_backend
from pocketllm.server.xlm_backend import _refuse_unsupported_sampling, _resolve_model
from pocketllm.xlm import is_available

_SDK = pathlib.Path.home() / "llm_sdk" / "D-Robotics_LLM_S600_1.0.2_SDK" / "oellm_runtime"
_HBM = _SDK / "model/Qwen3_0.6B/Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm"
_DEMO = _SDK / "examples/llm_demo"

needs_delegate = pytest.mark.skipif(
    not is_available(), reason="libxlm.so is not installed on this host"
)
needs_checkpoint = pytest.mark.skipif(
    not (_HBM.is_file() and (_DEMO / "qwen3_0.6b_config.json").is_file()),
    reason="the S600 SDK's Qwen3-0.6B .hbm and config are not on this host",
)

_NOTE = "the delegate samples greedily."


# -- the sampling refusal, checkable on any host ----------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},                                   # nothing asked
        {"temperature": 0},                   # greedy, what the delegate does
        {"temperature": 0.0},
        {"top_p": 1.0},                        # the value that disables it
        {"top_k": 0},                          # likewise
        {"top_k": -1},
        {"temperature": 0, "top_p": 1.0, "top_k": 0},   # defaults spelled out
        {"prompt": "hi", "max_tokens": 16},    # unrelated fields are not our business
    ],
)
def test_a_request_that_names_the_delegates_behaviour_is_accepted(body: dict) -> None:
    """A client that spells out the defaults must not be punished for it.

    This is the half that would break real callers if it were wrong: an OpenAI
    client very often sends `temperature: 0` or `top_p: 1.0` explicitly, and a
    refusal of either would be a server that rejects ordinary requests.
    """
    assert _refuse_unsupported_sampling(body, _NOTE) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 1.0),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 40),
    ],
)
def test_a_sampling_value_the_delegate_cannot_apply_is_refused_by_name(
    field: str, value: float
) -> None:
    """The value that would have changed the answer is refused, naming the field.

    `temperature: 1.0` is included on purpose: it is the OpenAI default and the
    temptation is to let it through, but the delegate is greedy, so serving it
    silently would answer a question the client did not ask.
    """
    refusal = _refuse_unsupported_sampling({field: value}, _NOTE)
    assert refusal is not None
    assert refusal.field == field
    assert field in refusal.message
    # The remedy has to be actionable without reading the source.
    assert "Omit" in refusal.message


def test_refusals_do_not_mistake_a_bool_for_a_number() -> None:
    """`true` is a bool and `isinstance(True, int)` is true; it is still not a temperature."""
    assert _refuse_unsupported_sampling({"temperature": True}, _NOTE) is None
    assert _refuse_unsupported_sampling({"top_k": False}, _NOTE) is None


# -- whether a session can be deterministic, checkable on any host -----------


@pytest.mark.parametrize(
    ("config", "deterministic"),
    [
        ({"temperature": 0.0, "do_sample": False, "top_k": 1, "top_p": 1.0}, True),
        ({"do_sample": False}, True),
        ({"temperature": 0}, True),
        # The SDK's own shipped file: sampling, and not reproducible run to run.
        ({"temperature": 0.6, "do_sample": True, "top_k": 20, "top_p": 0.95}, False),
        ({"temperature": 0.6}, False),
        # A bool is not a temperature, and a missing file is not a promise.
        ({"temperature": False}, False),
        ({}, False),
    ],
)
def test_the_determinism_flag_reads_the_generation_config(
    tmp_path: pathlib.Path, config: dict, deterministic: bool
) -> None:
    """The sampler is built from `generation_config.json`, so that file decides.

    The delegate ignores the request's `Sampling` block, so the only lever on
    reproducibility is the file beside the tokenizer.  Reading it here is what
    tells the refusal message (and a diagnostic) whether this session can
    promise the same text twice; a spine that guessed "greedy" would be lying
    on the SDK's own default file.
    """
    (tmp_path / "generation_config.json").write_text(json.dumps(config))
    assert xlm_backend._tokenizer_dir_is_deterministic(tmp_path) is deterministic


def test_a_directory_without_a_generation_config_is_not_called_deterministic(
    tmp_path: pathlib.Path,
) -> None:
    """An absent file is the delegate's own default, which samples."""
    assert xlm_backend._tokenizer_dir_is_deterministic(tmp_path) is False


# -- where the delegate's inputs come from, checkable on any host -----------


def test_a_demo_config_json_resolves_the_hbm_and_tokenizer(tmp_path: pathlib.Path) -> None:
    """`--model config.json` is enough, because that is the SDK demo's own shape."""
    (tmp_path / "tok").mkdir()
    (tmp_path / "m.hbm").write_bytes(b"")
    config = tmp_path / "qwen3_0.6b_config.json"
    config.write_text(
        json.dumps(
            {
                "hbm_path": "m.hbm",
                "tokenizer_dir": "tok",
                "model_type": 9,
                "context_size": 4096,
            }
        )
    )
    resolved = _resolve_model(EngineArgs(model=str(config)))
    assert resolved.hbm == (tmp_path / "m.hbm").resolve()
    assert resolved.tokenizer_dir == (tmp_path / "tok").resolve()
    assert resolved.config == config
    assert resolved.model_type == 9


def test_the_configs_enable_thinking_sets_the_reasoning_mode(tmp_path: pathlib.Path) -> None:
    """The demo config's ``enable_thinking`` is a property of the model, not the request.

    The delegate builds its chat template from this at load, so it is read here
    and carried on the resolved paths.  It matters for streaming: a ``.hbm``
    built for thinking always emits a `` thinking`` block, and reading it as a
    plain chat model streams the block into ``content``.  Absent is the SDK's
    own default of a non-thinking model.
    """
    (tmp_path / "tok").mkdir()
    (tmp_path / "m.hbm").write_bytes(b"")
    config = tmp_path / "c.json"

    def mode_for(spec: dict) -> str:
        config.write_text(json.dumps({"hbm_path": "m.hbm", "tokenizer_dir": "tok", **spec}))
        return _resolve_model(EngineArgs(model=str(config))).thinking_mode

    assert mode_for({"enable_thinking": True}) == "thinking"
    assert mode_for({"enable_thinking": False}) == "chat"
    assert mode_for({}) == "chat"


def test_a_missing_config_is_reported_before_anything_is_loaded() -> None:
    """An `.hbm` with no config is refused with the reason, at construction.

    The config is not optional even though it looks it: it carries the BPU core
    list and the context size the graph was compiled for, so a session without it
    runs a different model than the caller named.
    """
    with pytest.raises(ConfigurationError) as raised:
        _resolve_model(EngineArgs(model="/tmp/does-not-exist.hbm"))
    assert "config" in str(raised.value).lower()


def test_a_config_that_names_a_missing_hbm_is_refused(tmp_path: pathlib.Path) -> None:
    (tmp_path / "tok").mkdir()
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"hbm_path": "absent.hbm", "tokenizer_dir": "tok"}))
    with pytest.raises(ConfigurationError) as raised:
        _resolve_model(EngineArgs(model=str(config)))
    assert ".hbm" in str(raised.value)


def test_the_module_imports_without_the_delegate() -> None:
    """The boundary the whole design rests on: this module never needs the SDK to import."""
    assert hasattr(xlm_backend, "XlmBackend")


# -- the over-cap guard's host half, checkable on any host ------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3-0.6B_language_chunk_512_cache_1024_w8_nash-p_corenum_4_4.hbm", 1024),
        ("Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm", 4096),
        ("ours_06b_cache1024_w8.hbm", 1024),  # the shape is spelled without the underscore too
        ("a_graph_with_no_build_shape.hbm", None),  # nothing to read -> the guard stays off
        ("something_cache_0.hbm", None),  # zero is not a window
    ],
)
def test_the_cache_window_is_read_from_the_hbm_shape(name: str, expected: int | None) -> None:
    """The window comes from the compiled name, and a name without one reads as "no guard".

    The SDK exposes the cache nowhere else — no ``xlm.h`` call reports it and
    ``context_size`` does not move it — so the build shape in the filename is the
    only honest source.  The ``None`` cases matter as much as the numbers: a shape
    that cannot be read must *disarm* the guard, never default to a guess, because
    guarding against the wrong window is the failure the guard exists to prevent.
    """
    assert xlm_backend._hbm_cache_tokens(name) == expected


def test_a_missing_tokenizer_yields_no_counter(tmp_path: pathlib.Path) -> None:
    """No ``tokenizer.json`` means no counter, so the caller can warn rather than run blind.

    The guard is a pair, and half a pair guards nothing.  Returning ``None`` here
    is what lets :class:`XlmBackend` raise its loud startup warning instead of
    arming a guard against an unknown length.
    """
    assert xlm_backend._build_token_counter(tmp_path / "absent.json") is None


def test_the_counter_counts_ids_plus_the_chat_wrapper() -> None:
    """The count is the tokenizer's ids **plus** the wrapper the delegate prepends.

    The guard must compare against what the delegate's *prefill* sees, not what the
    raw tokenizer returns: the chat scaffold is ~29 tokens the caller never typed.
    An empty prompt is the clean read of the constant — the ids are zero, so the
    count is the wrapper alone.
    """
    pytest.importorskip("tokenizers")
    tokenizer_file = _SDK / "configs/Qwen3_config/tokenizer.json"
    if not tokenizer_file.is_file():
        pytest.skip("the SDK tokenizer.json is not on this host")

    counter = xlm_backend._build_token_counter(tokenizer_file)
    assert counter is not None
    assert counter("") == xlm_backend._OVER_CAP_WRAPPER_TOKENS
    # Same text, same count — the counter is a function of the prompt alone.
    assert counter("The capital of France is") == counter("The capital of France is")


def test_the_over_cap_error_becomes_a_client_facing_request_error() -> None:
    """The seam's stdlib exception is translated to the API's ``ConfigurationError``.

    ``xlm.py`` is stdlib-only and cannot inherit the package's error, so its
    :class:`~pocketllm.xlm.XlmOverCapError` would otherwise reach
    :func:`~pocketllm.server.openai._error_status` as a bare ``RuntimeError`` and
    be answered **500 server_error** — the server blamed for a prompt the caller
    can shorten.  ``ConfigurationError`` is what that function maps to **400
    invalid_request_error**, so this asserts the type it reads.  Every other
    exception must pass through untouched.
    """
    from pocketllm.xlm import XlmOverCapError

    translated = xlm_backend._as_request_error(XlmOverCapError("too long"))
    assert isinstance(translated, ConfigurationError)

    unrelated = ValueError("something else")
    assert xlm_backend._as_request_error(unrelated) is unrelated


# -- the behaviour, needs the board -----------------------------------------


@needs_delegate
@needs_checkpoint
def test_a_backend_answers_through_the_serving_contract() -> None:
    """Open, generate, and check the two honest gaps this path has.

    The answer is the easy half.  The half worth testing is what the adapter
    *does not* claim: no token ids, and — because this SDK build leaves the
    delegate's token counters at zero — no token counts either.  Both are
    asserted as absent rather than merely unchecked, because the failure this
    guards is an adapter that fills them with something plausible.
    """
    from pocketllm.api import GenerationRequest, SamplingParams

    backend = xlm_backend.XlmBackend(
        EngineArgs(model=str(_DEMO / "qwen3_0.6b_config.json"))
    )
    try:
        assert backend.health().ready
        assert backend.audit_request({"temperature": 1.0}) is not None
        assert backend.audit_request({"temperature": 0}) is None

        result = backend.generate(
            [GenerationRequest(prompt="Name the largest planet.", sampling_params=SamplingParams())]
        )[0]
        assert "jupiter" in result.text.lower()
        # No token ids on this path, and the result says so by being empty.
        assert result.token_ids == []
        # The throughput the delegate did report is carried; the token counts it
        # did not are zero, not estimated from the two.
        assert result.metadata.get("decode_tps", 0) > 0
        assert result.usage.completion_tokens == 0
    finally:
        backend.close()