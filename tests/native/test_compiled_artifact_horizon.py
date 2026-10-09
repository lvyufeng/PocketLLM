"""Our own compiled S600 artifact, run on the board.

The S600 compile chain produced its first artifact — a `Qwen3-1.7B` language
graph at `cache_1024`, built with the SDK's `leap_llm`/`hbdk4` toolchain on the
x86 host (see the build result in
[the native chain page](https://github.com/lvyufeng/PocketLLM/blob/main/docs/architecture/s600_native_chain.md)).
Every other test in this directory drives a `.hbm` the **vendor** built; this is
the only place the tree asserts that a graph **we** compiled loads and answers,
which is the claim the whole chain-scoping page rests on. It is regression
protection for the chain, not for the delegate.

It also pins the one trap that already bit us. The `.hbm` records the BPU core
count it was compiled for, and a config whose ``"bpu_core"`` disagrees fails
*inside* the decode: the SDK logs the mismatch to stderr, the callback sees
``XLM_STATE_ERROR`` — and ``xlm_infer`` still returns **0**, so before the fix
this test guards, ``pocketllm run`` printed the prompt and an empty answer and
exited 0. A silent wrong answer is the failure class this tree has been bitten by
before, so the mismatch is asserted to be a *refusal*, not a blank success.

**The artifact is not shipped.** It is a ~1.7 GiB build output produced on the
x86 host and copied to the board; when it is absent the tests **skip**, and the
skip names the path it looked at (``POCKETLLM_COMPILED_HBM`` overrides it) so the
next person knows where it goes. A skip is not a pass.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from pocketllm.xlm import is_available

_SDK = pathlib.Path.home() / "llm_sdk" / "D-Robotics_LLM_S600_1.0.2_SDK" / "oellm_runtime"

#: The build output's name, and the directory the deployment recipe puts it in.
_HBM_NAME = "Qwen3-1.7B_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm"
_ARTIFACT_ENV = "POCKETLLM_COMPILED_HBM"
_ARTIFACT = pathlib.Path(
    os.environ.get(_ARTIFACT_ENV)
    or _SDK / "model" / "Qwen3_1.7B_compiled" / _HBM_NAME
)

#: The same canonical prompt the ladder is measured on, so the answer is a known
#: one and the assertion is about the graph, not about fluency.
_PROMPT = "The capital of France is"

#: The BPU core list the artifact was compiled for (``corenum_4_4`` in its name).
#: A config that omits it makes the delegate fall back to a single core and the
#: prefill is refused — which is the trap `test_a_bpu_core_mismatch_is_refused`
#: drives on purpose.
_BPU_CORE = [0, 1, 2, 3]

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

needs_delegate = pytest.mark.skipif(
    not is_available(), reason="libxlm.so is not installed on this host"
)
needs_artifact = pytest.mark.skipif(
    not _ARTIFACT.is_file(),
    reason=f"no self-compiled .hbm at {_ARTIFACT} (set {_ARTIFACT_ENV}, or copy the "
    "x86 host's build_out/ output there)",
)
needs_board_env = pytest.mark.skipif(
    not os.environ.get("LD_LIBRARY_PATH") or not os.environ.get("HB_DNN_USER_DEFINED_L2M_SIZES"),
    reason="set LD_LIBRARY_PATH=<sdk>/oellm_runtime/lib and HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6 "
    "(they are read by dlopen/libhbrt4 before this process starts)",
)


def _greedy_tokenizer_dir(root: pathlib.Path) -> pathlib.Path:
    """A tokenizer directory whose ``generation_config.json`` forces greedy decode.

    The delegate builds its sampler from this file, so it is also what makes the
    determinism assertion below meaningful: the same prompt through the same graph
    is a fixed text rather than a draw.
    """
    directory = root / "tokenizer"
    directory.mkdir()
    for source in (_SDK / "configs/Qwen3_config").iterdir():
        if source.is_file() and source.name != "generation_config.json":
            (directory / source.name).symlink_to(source)
    (directory / "generation_config.json").write_text(json.dumps(_GREEDY), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def configs(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, pathlib.Path]:
    """``(good, bad)`` config files — the pair differing only in ``bpu_core``.

    The two differ in exactly one key on purpose: the "good" config is the
    deployment recipe, and the "bad" one is the same recipe with the core list
    dropped, so a failure cannot be attributed to anything else.
    """
    root = tmp_path_factory.mktemp("s600_compiled")
    tokenizer = _greedy_tokenizer_dir(root)
    common = {
        "hbm_path": str(_ARTIFACT),
        "tokenizer_dir": str(tokenizer),
        "model_type": 9,
        "enable_multi_turn": False,
        "enable_thinking": True,
    }
    good = root / "compiled_config.json"
    good.write_text(json.dumps({**common, "bpu_core": _BPU_CORE}), encoding="utf-8")
    bad = root / "compiled_config_no_core.json"
    bad.write_text(json.dumps(common), encoding="utf-8")
    return good, bad


def _run(config: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """``pocketllm run --device horizon`` on ``config``, in a fresh interpreter.

    A subprocess rather than an in-process call to ``_cmd_run``: the deployment
    recipe is a command line, and the delegate's two environment variables are
    read by ``dlopen``/``libhbrt4`` *before* this process started — running the
    child is the only way the test exercises what a caller actually types.  The
    child inherits ``LD_LIBRARY_PATH``/``HB_DNN_USER_DEFINED_L2M_SIZES`` and the
    ``PYTHONPATH`` `tests/conftest.py` exports.
    """
    return subprocess.run(
        [
            sys.executable, "-m", "pocketllm", "run",
            "--device", "horizon", "--model", str(config), "--prompt", _PROMPT,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )


def _answer(stdout: str) -> str:
    """The completion, with the ``run`` path's echoed prompt stripped off.

    ``run`` mirrors the native contract: it prints the prompt, then the answer.
    The prompt is a prefix of stdout, so what follows it is the answer.
    """
    assert stdout.startswith(_PROMPT), f"the prompt was not echoed: {stdout[:80]!r}"
    return stdout[len(_PROMPT) :]


@needs_delegate
@needs_artifact
@needs_board_env
def test_the_compiled_graph_loads_and_answers(configs) -> None:
    """Our own artifact loads through ``run --device horizon`` and answers the prompt.

    This is the property nothing else in the tree asserts: a ``.hbm`` *this*
    project compiled, driven end to end on the board.  The answer is checked
    against the canonical prompt's known text, so a graph that loaded but
    produced garbage fails here rather than passing on "non-empty".
    """
    good, _bad = configs
    result = _run(good)
    assert result.returncode == 0, f"rc {result.returncode}\nstderr:\n{result.stderr[-2000:]}"
    answer = _answer(result.stdout)
    assert "Paris" in answer, f"the compiled graph did not answer the prompt: {answer[:200]!r}"


@needs_delegate
@needs_artifact
@needs_board_env
def test_the_compiled_graph_is_deterministic(configs) -> None:
    """Three greedy runs of one prompt produce byte-identical text.

    The delegate's sampler is fixed by the tokenizer directory's
    ``generation_config.json``, so a graph that is deterministic gives one text
    and a graph that is not gives three — this is the same property the vendor
    graphs were held to, applied to ours.
    """
    good, _bad = configs
    runs = [_run(good) for _ in range(3)]
    for result in runs:
        assert result.returncode == 0, f"rc {result.returncode}\nstderr:\n{result.stderr[-2000:]}"
    texts = {result.stdout for result in runs}
    assert len(texts) == 1, (
        "three greedy runs of our compiled graph produced "
        f"{len(texts)} distinct texts; the graph is not deterministic"
    )


@needs_delegate
@needs_artifact
@needs_board_env
def test_a_bpu_core_mismatch_is_refused(configs) -> None:
    """A config that disagrees with the graph's core count fails *loudly*.

    The trap this test exists for: the ``.hbm`` is compiled for four BPU cores,
    and a config with no ``bpu_core`` makes the delegate run on one.  The prefill
    is refused inside the decode, and the failure reaches the caller as a non-zero
    exit with a refusal on stderr -- **not** as a zero exit with an empty answer,
    which is how it behaved before ``XlmEngine.infer`` learned to raise on
    ``XLM_STATE_ERROR``.  A regression to the silent form fails here.
    """
    _good, bad = configs
    result = _run(bad)
    assert result.returncode != 0, (
        "a bpu_core mismatch exited 0 -- the empty answer is being reported as a success: "
        f"{result.stdout[:200]!r}"
    )
    assert "inference error" in result.stderr or "bpu_core" in result.stderr, (
        f"the refusal does not say what went wrong: {result.stderr[-2000:]!r}"
    )