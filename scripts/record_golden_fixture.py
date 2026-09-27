#!/usr/bin/env python3
"""Record a served-path golden fixture for one entry point.

Run this once per entry point, when the entry point's answer is believed correct. It runs the entry
point for real, records the command line, the environment and the answer, and writes
`tests/fixtures/golden/<entry>.json`. `tests/test_served_path_golden.py` then re-runs the same thing
on every suite run, and skips when the checkpoint is not on the machine reading it.

Recording is a deliberate act rather than a side effect: a fixture is only worth having if a human
looked at the answer and agreed it was right, because re-recording is how a real regression gets
quietly blessed into the baseline of "correct".

Examples::

    # a Python entry point
    python scripts/record_golden_fixture.py --entry xing4 \\
        --checkpoint /mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf \\
        --max-tokens 16 -- --max-model-len 4096

    # the native binary: the flag after `--` is the binary's own
    python scripts/record_golden_fixture.py --entry native \\
        --checkpoint /mnt/data2/Qwen3.8-27B-FP8 \\
        -- --binary cpp_engine/build-python/pocketllm_engine --devices 0,1,2,3 \\
           --max-context 4096 --max-batch-size 1 --port 18291

Everything after `--` goes to the entry point verbatim, and the parser needs the separator because
an unknown `--flag` is otherwise an error rather than a pass-through. The prompt and the token budget
are this script's own `--prompt` and `--max-tokens`, because they are what a fixture records; greedy
is not overridable, because a fixture that samples is a fixture that fails one run in ten for no
reason.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import pathlib
import subprocess
import sys
import traceback

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The default prompt. Short, deterministic, and it asks for a specific string so a truncated or
#: restarted answer is visible in the recorded text rather than looking plausible.
DEFAULT_PROMPT = "Reply with the single line: golden fixture."

_GOLDEN = REPO_ROOT / "tests" / "golden_fixtures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("golden_fixtures", _GOLDEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _current_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=list(_load_module().ENTRY_POINTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment the run needs; repeatable")
    parser.add_argument(
        "--requires-dev-shm-gib",
        type=float,
        default=0.0,
        help=(
            "free /dev/shm the run needs, for the entries that pin an expert bank there. Recorded "
            "rather than guessed at run time, because only the operator knows what the entry point "
            "allocates; the two bank-using entries need 458 and 150 GiB and do not fit together"
        ),
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=pathlib.Path, help="defaults to tests/fixtures/golden/<entry>.json")
    parser.add_argument("extra", nargs="*", help="flags passed through to the entry point")
    args = parser.parse_args(argv)

    golden = _load_module()
    env: dict[str, str] = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value

    sampling = {"temperature": 0.0, "max_tokens": args.max_tokens}
    requires = (
        {"dev_shm_bytes": int(args.requires_dev_shm_gib * 2**30)}
        if args.requires_dev_shm_gib > 0
        else {}
    )
    if args.entry == "native":
        argv_list = ["--ckpt", args.checkpoint, *extra_flags(args.extra)]
    else:
        argv_list = ["serve", "--backend", args.entry, "--model", args.checkpoint, *extra_flags(args.extra)]

    fixture = golden.GoldenFixture(
        entry=args.entry,
        checkpoint=args.checkpoint,
        argv=tuple(argv_list),
        prompt=args.prompt,
        expected={},
        env=env,
        sampling=sampling,
        requires=requires,
        commit=_current_commit(),
        taken_at=datetime.date.today().isoformat(),
        notes=args.notes,
    )

    print(f"recording {args.entry}: {' '.join(fixture.argv)}", flush=True)
    try:
        # The same path the suite takes, child process included, so what is recorded is what will be
        # checked. Recording through a different launcher than the one that verifies it is how a
        # fixture ends up pinning something nobody runs.
        outcome = golden.run_isolated(fixture, verbose=True)
    except Exception:
        traceback.print_exc()
        return 1

    expected: dict[str, object] = {}
    if outcome.prompt_tokens is not None:
        expected["prompt_tokens"] = int(outcome.prompt_tokens)
    if outcome.token_ids is not None:
        expected["token_ids"] = list(outcome.token_ids)
    if outcome.text is not None:
        expected["text"] = outcome.text
    if "token_ids" not in expected and "text" not in expected:
        print("the entry point produced neither token ids nor text; nothing to record", file=sys.stderr)
        return 1

    fixture = golden.GoldenFixture(**{**fixture.__dict__, "expected": expected})
    path = golden.write_fixture(fixture) if args.out is None else _write(args.out, fixture)
    print(f"wrote {path} in {outcome.elapsed_seconds:.1f}s")
    print(f"  expected = {expected}")
    return 0


def extra_flags(items: list[str]) -> list[str]:
    """Pass the flags after `--` straight through, in the `--flag value` spelling operators use.

    Kept as typed rather than re-encoded, so the recorded `argv` is a command line a person could
    paste into a shell. A bare word is refused rather than ignored: it is almost always a prompt or
    a budget that was meant for `--prompt` or `--max-tokens`.
    """
    flags: list[str] = []
    index = 0
    while index < len(items):
        item = items[index]
        if item.startswith("--"):
            flags.append(item)
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                flags.append(items[index + 1])
                index += 2
                continue
        else:
            raise SystemExit(
                f"pass-through flags have to start with `--`, got {item!r} "
                f"(the prompt goes in --prompt and the budget in --max-tokens)"
            )
        index += 1
    return flags


def _write(path: pathlib.Path, fixture) -> pathlib.Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture.to_json(), indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
