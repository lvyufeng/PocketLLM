"""`serve --device horizon` end to end: the delegate behind the HTTP surface.

Every S600 measurement before this file went through ``pocketllm run``.  The
serving path is a different surface around the same delegate -- a
``ThreadingHTTPServer``, one backend lock, per-request sampling refusals -- and
the part worth a test is the one that fails *silently*: the delegate writes its
runtime banner to file descriptor **1**, and
:func:`pocketllm.xlm.quiet_delegate_stdout` moves fd 1 to stderr for the life of
each load/infer.  Under ``run`` that is one process doing one thing; under a
threaded server the same descriptor is shared by every request handler.

**A leaked banner is the failure mode this file exists to catch**, because it is
the one that returns HTTP 200 with the answer buried under (or replaced by) SDK
noise -- a reply a client parses into garbage, the same class of silent 200 we
hit on the Ascend board.  So the assertions are: every body is valid JSON, the
answer is exactly what the same prompt produced, no SDK-monitor token appears in
any body, and concurrent requests serialize to the *same* text rather than
interleaving into a wrong one.

The behaviour needs the delegate, the SDK, the `.hbm`, **and** the two
environment variables the SDK's own ``run_llm.sh`` sets.  The skip names them,
because a skip is not a pass: without ``LD_LIBRARY_PATH`` pointing at
``<sdk>/oellm_runtime/lib`` (the directory the SDK's libraries actually live in --
there is no ``<sdk>/lib``) the ``dlopen`` dies on ``libopencv_world.so.409``, and
without ``HB_DNN_USER_DEFINED_L2M_SIZES`` the BPU allocation fails -- both before
any test body runs, and neither is a property this host can fake.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import re
import threading
import urllib.error
import urllib.request

import pytest

from pocketllm.api import EngineArgs
from pocketllm.server import xlm_backend
from pocketllm.server.openai import OpenAIHandler, PocketLLMHTTPServer
from pocketllm.xlm import is_available

_SDK = pathlib.Path.home() / "llm_sdk" / "D-Robotics_LLM_S600_1.0.2_SDK" / "oellm_runtime"
_HBM = _SDK / "model/Qwen3_0.6B/Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm"
_DEMO = _SDK / "examples/llm_demo"
_TOKENIZER = _SDK / "configs/Qwen3_config"

#: A short prompt, so a concurrent round costs a few seconds rather than a
#: minute.  The banner-leak property does not depend on the answer's length.
_PROMPT = "Reply with the single word: OK"

#: Tokens that appear in the delegate's fd-1 banner (and the SDK's own loader
#: chatter on fd 2) and never in a legitimate answer.  A body containing one is
#: the leak this file is for.
_BANNER = re.compile(
    r"\[UCP\]|\[DNN\]|\[VP\]|\[HPL\]|\[UCPT\]|BPU_MONITOR|BPULib|HBRT|mod_mgr|XlmImpl"
)

needs_delegate = pytest.mark.skipif(
    not is_available(), reason="libxlm.so is not installed on this host"
)
needs_checkpoint = pytest.mark.skipif(
    not (_HBM.is_file() and (_DEMO / "qwen3_0.6b_config.json").is_file()),
    reason="the S600 SDK's Qwen3-0.6B .hbm and config are not on this host",
)
needs_board_env = pytest.mark.skipif(
    not os.environ.get("LD_LIBRARY_PATH") or not os.environ.get("HB_DNN_USER_DEFINED_L2M_SIZES"),
    reason="set LD_LIBRARY_PATH=<sdk>/oellm_runtime/lib and HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6 "
    "(they are read by dlopen/libhbrt4 before this process starts)",
)

#: Greedy, so a concurrent round compares answers by equality and not by a
#: sampler that drifted between two runs.  The delegate builds its sampler from
#: this file, not from the request's ``Sampling`` block.
_GREEDY = {
    "bos_token_id": 151643,
    "do_sample": False,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "transformers_version": "4.51.0",
}


def _greedy_tokenizer_dir(root: pathlib.Path) -> pathlib.Path:
    """A tokenizer directory whose ``generation_config.json`` forces greedy decode."""
    directory = root / "tokenizer"
    directory.mkdir()
    for source in _TOKENIZER.iterdir():
        if source.is_file() and source.name != "generation_config.json":
            (directory / source.name).symlink_to(source)
    (directory / "generation_config.json").write_text(json.dumps(_GREEDY), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory):
    """One 0.6B backend behind a live HTTP server, torn down at module end.

    Module-scoped on purpose: opening the delegate maps a 1.02 GiB ``.hbm`` onto
    the BPU, and a fresh open per test would spend the whole run loading.  The
    server is the same ``PocketLLMHTTPServer`` the CLI builds, bound to port 0.
    """
    root = tmp_path_factory.mktemp("s600_serve")
    tokenizer = _greedy_tokenizer_dir(root)
    backend = xlm_backend.XlmBackend(
        EngineArgs(
            model=str(_DEMO / "qwen3_0.6b_config.json"),
            tokenizer_path=str(tokenizer),
        )
    )
    server = PocketLLMHTTPServer(("127.0.0.1", 0), OpenAIHandler, backend, "qwen3-0.6b")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        backend.close()


def _chat(base: str, content: str, **sampling):
    """POST one chat completion and return ``(status, content_type, raw_bytes)``."""
    body = json.dumps(
        {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": content}],
            **sampling,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), exc.read()


@needs_delegate
@needs_checkpoint
@needs_board_env
def test_a_single_request_is_clean_json_with_no_banner(served: str) -> None:
    """The body is the answer, parseable, and free of any SDK-monitor token.

    This is the contract a caller depends on and the one the fd-1 redirect
    exists to keep: ``serve`` is not ``run`` -- there is no "echo the prompt,
    then the completion" line to hide a banner behind, so a stray fd-1 write
    lands *inside* the JSON.
    """
    status, content_type, raw = _chat(served, _PROMPT)
    assert status == 200
    assert content_type is not None and "application/json" in content_type
    text = raw.decode("utf-8")
    reply = json.loads(text)  # a client that parses JSON must not choke
    assert reply["choices"][0]["message"]["content"].strip()
    assert not _BANNER.search(text), f"the delegate's banner leaked into the body: {text[:200]!r}"


@needs_delegate
@needs_checkpoint
@needs_board_env
def test_concurrent_requests_do_not_leak_the_banner(served: str) -> None:
    """Several rounds of overlapping requests, every body clean and correct.

    The leak, if it exists, is a race on the shared fd 1, so one round proves
    nothing -- this is several.  The answer is also asserted to be *identical*
    across the round: the adapter holds one delegate behind a lock, so an
    overlap that actually overlapped would interleave two prompts into one
    answer and return a fluent, wrong 200.  That is a correctness failure, not a
    perf note.
    """
    rounds, concurrency = 3, 6
    for _ in range(rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: _chat(served, _PROMPT), range(concurrency)))
        contents = set()
        for status, _content_type, raw in results:
            text = raw.decode("utf-8")
            assert status == 200, text[:200]
            assert not _BANNER.search(text), f"banner leaked: {text[:200]!r}"
            contents.add(json.loads(text)["choices"][0]["message"]["content"])
        assert len(contents) == 1, (
            f"{concurrency} concurrent greedy requests returned {len(contents)} distinct answers; "
            "the single-session delegate is not being serialized"
        )


@needs_delegate
@needs_checkpoint
@needs_board_env
def test_the_sampling_refusal_is_a_clean_400(served: str) -> None:
    """A per-request value the fixed delegate cannot apply is refused by name.

    Not a 500 traceback and not a silently-ignored field: a client that asked for
    ``temperature: 0.9`` and got a greedy answer with a 200 could not tell the
    reply from the one it wanted.  The refusal names the field it refused.
    """
    for field, value in (("temperature", 0.9), ("top_p", 0.5), ("top_k", 20)):
        status, _content_type, raw = _chat(served, _PROMPT, **{field: value})
        assert status == 400, f"{field}={value} was not refused: {raw[:120]!r}"
        message = json.loads(raw.decode())["error"]["message"]
        assert field in message


@needs_delegate
@needs_checkpoint
@needs_board_env
def test_naming_the_defaults_is_accepted(served: str) -> None:
    """A client that spells out the greedy behavior is served, not punished.

    ``top_k: 0`` is the load-bearing case and a regression guard: it is the "no
    limit" value the CLI help and the adapter's own refusal both hand out, so a
    validator that rejected it would contradict the remedy the server prints.
    """
    for sampling in (
        {"temperature": 0, "top_p": 1.0},
        {"top_k": 0},
        {"temperature": 0, "top_p": 1.0, "top_k": 0},
    ):
        status, _content_type, raw = _chat(served, _PROMPT, **sampling)
        assert status == 200, f"{sampling} was refused: {raw[:120]!r}"