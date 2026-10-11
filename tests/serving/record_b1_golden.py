"""Record the G1 golden fixture for the width-1 `serve` path.

G1 says a single streamed request's output does not move when the per-request
`_run` loop is replaced by the scheduler.  A test can only enforce that against a
*recorded* answer, because the substitution happened inside one module: comparing
the post-change backend against a second post-change backend (`max_batch_size=1`)
compares a path to itself and passes through any regression in it.

This script records the answer from the module as it was *before* Stage 2 -- the
tree at the rev named by `--rev`, on the python3.10 interpreter the package
needs -- for one greedy
streamed request, and writes `tests/fixtures/golden/native_serve_b1.json` tagged
with the checkpoint's sha256 and the pristine module's sha256.  The replay test
uses the checkpoint tag to decide whether it can run at all, so the fixture only
claims a machine that holds the exact checkpoint it was recorded against.

Re-record only when the width-1 path is *deliberately* changed, and look at the
new answer before committing it: re-recording is how a real regression gets
quietly blessed into "correct".

The interpreter and the library path: this package needs Python 3.10, which is
not necessarily the `python3` on the path -- on the Ascend board numpy lives in a
python3.9 site-packages while `/usr/bin/python3` is 3.10, and the conda env that
has both is the one to use.  Run the script from a directory that does not hold a
`types.py`: the interpreter prepends its cwd, and a shadowed stdlib module breaks
start-up before anything here runs.  The command that recorded the committed
fixture, on the Ascend board::

    cd /tmp && PY=/path/to/python3.10-with-numpy/bin/python \
      source /usr/local/Ascend/ascend-toolkit/set_env.sh \
      export LD_LIBRARY_PATH=$REPO/src/build:$LD_LIBRARY_PATH \
      export POCKETLLM_CORE_LIB=$REPO/src/build/libpocketllm.so \
      $PY $REPO/tests/serving/record_b1_golden.py --rev ac36842 \
      --checkpoint /path/to/Qwen3-0.6B-Q4_K_M.gguf

`set_env.sh` is what puts `libmsprofiler.so` on the path, which the engine's
shared object needs to load at all; `$REPO/src/build` goes ahead of it so the
locally built core wins over any installed one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PRISTINE_REL = "python/pocketllm/server/native_backend.py"

PROMPT = "Reply with the single line: golden fixture."
MAX_TOKENS = 8


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pristine(rev: str) -> tuple[object, str]:
    """The module as it was at `rev`, loaded under a private name.

    Imported by path rather than by package, so it cannot shadow or be shadowed by
    the current `pocketllm.server.native_backend`.  It still imports the tracked
    package for `EngineArgs` and friends, which is what makes it a *path* fixture
    rather than a copy of the whole world.
    """
    source = subprocess.run(
        ["git", "show", f"{rev}:{PRISTINE_REL}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.encode()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "pristine_native_backend.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("pristine_native_backend", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, _sha256_bytes(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rev", required=True, help="the pre-Stage-2 rev to record from")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "tests" / "fixtures" / "golden" / "native_serve_b1.json",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "python"))
    from pocketllm.api import EngineArgs
    from pocketllm.protocol import build_chat_request

    module, module_sha = _load_pristine(args.rev)
    checkpoint = pathlib.Path(args.checkpoint)

    backend = module.NativeBackend(EngineArgs(model=str(checkpoint), max_batch_size=1), "cpu")
    try:
        request = build_chat_request(
            {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": MAX_TOKENS},
            request_id="g1-golden",
        )
        events = list(backend.stream(request))
    finally:
        backend.close()

    token_ids = [event.token_id for event in events if event.token_id is not None]
    text = "".join(event.text or "" for event in events)
    finish_reason = events[-1].finish_reason
    usage = events[-1].usage
    if not token_ids:
        print("the entry point produced no tokens; nothing to record", file=sys.stderr)
        return 1

    fixture = {
        "entry": "serve/native-b1",
        "note": (
            "Recorded from the module before the Stage 2 scheduler replaced its per-request "
            "`_run` loop, on the python3.10 interpreter the package needs, greedy.  G1: a "
            "width-1 streamed request must still produce exactly this."
        ),
        "recorded_from": {"rev": args.rev, "module": PRISTINE_REL, "sha256": module_sha},
        "checkpoint": {"name": checkpoint.name, "sha256": _sha256_file(checkpoint)},
        "request": {
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        "expected": {
            "token_ids": token_ids,
            "text": text,
            "finish_reason": finish_reason,
            "prompt_tokens": None if usage is None else usage.prompt_tokens,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  token_ids={token_ids} text={text!r} finish_reason={finish_reason!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())