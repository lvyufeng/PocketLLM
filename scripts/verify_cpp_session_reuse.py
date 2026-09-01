#!/usr/bin/env python3
"""Check that a request stopping at EOS leaves the C++ adapter session clean.

``QwenEngine.generate`` takes no EOS argument and keeps mutating its session for
the whole token budget, so an adapter that ran the full budget and truncated the
result afterwards would leave the recurrent state and prefix cache positioned
after text the caller never saw.  The next request on that backend would then
resume from corrupted state.

This runs the same sequence of greedy requests two ways on a real checkpoint:

* ``shared``   - one ``CppBackend``, every request in order (the real serving path)
* ``isolated`` - a fresh ``CppBackend`` per request (nothing to carry over)

Parity between the two modes is the property under test.  The sequence puts a
request that hits EOS in the middle, which is exactly the case that used to
poison its successor.

Usage:
  python scripts/verify_cpp_session_reuse.py \
    --ckpt /mnt/data2/Qwen3.8-27B-FP8 --tp-world 2 --devices 0,1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.verify_cpp_adapter_parity import launch_ranks  # noqa: E402

# The middle prompt terminates in a couple of tokens, so it exercises the EOS
# stop; the ones around it run to the length limit.
DEFAULT_PROMPTS = [
    "Name three prime numbers.",
    "What is the capital of France? Answer in one word.",
    "用一句话介绍你自己。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tp-world", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp16")
    parser.add_argument("--work-dir", default="session_reuse_work")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def run_mode(args, cases_path: Path, work_dir: Path, devices, isolated: bool) -> list[dict]:
    mode = "isolated" if isolated else "shared"
    mode_dir = work_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [mode_dir / f"rank{rank}.json" for rank in range(args.tp_world)]

    def command_for_rank(rank: int, id_path: Path) -> list[str]:
        command = [
            sys.executable,
            str(REPO / "scripts/_session_reuse_rank.py"),
            "--ckpt", args.ckpt,
            "--tp-world", str(args.tp_world),
            "--tp-rank", str(rank),
            "--nccl-id-path", str(id_path),
            "--cases-file", str(cases_path),
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-context", str(args.max_context),
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--out", str(out_paths[rank]),
        ]
        if isolated:
            command.append("--isolated")
        return command

    launch_ranks(command_for_rank, args.tp_world, devices, mode_dir, args.timeout, mode)
    return [json.loads(path.read_text(encoding="utf-8")) for path in out_paths]


def main() -> int:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) < args.tp_world:
        raise SystemExit(f"--devices needs at least {args.tp_world} entries")

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    prompt_ids = [[int(token) for token in tokenizer.encode(prompt)] for prompt in prompts]
    cases_path = work_dir / "cases.txt"
    cases_path.write_text(
        "\n".join(",".join(str(token) for token in ids) for ids in prompt_ids) + "\n",
        encoding="utf-8",
    )

    shared = run_mode(args, cases_path, work_dir, devices, isolated=False)
    isolated = run_mode(args, cases_path, work_dir, devices, isolated=True)

    eos_ids = sorted(shared[0]["eos_token_ids"])
    print(f"eos={eos_ids} source={shared[0]['eos_source']}", flush=True)

    all_ok = True
    report: list[dict] = []
    for index, prompt in enumerate(prompts):
        shared_case = shared[0]["cases"][index]
        isolated_case = isolated[0]["cases"][index]
        tokens_match = shared_case["token_ids"] == isolated_case["token_ids"]
        reason_match = shared_case["finish_reason"] == isolated_case["finish_reason"]
        usage_match = shared_case["usage"] == isolated_case["usage"]
        ranks_agree = all(
            item["cases"][index]["token_ids"] == shared_case["token_ids"] for item in shared
        )
        passed = tokens_match and reason_match and usage_match and ranks_agree
        all_ok = all_ok and passed
        checks = {
            "tokens_match_isolated": tokens_match,
            "finish_reason_match": reason_match,
            "usage_match": usage_match,
            "shared_ranks_agree": ranks_agree,
        }
        report.append(
            {
                "index": index,
                "prompt": prompt,
                "prompt_tokens": shared_case["prompt_tokens"],
                "shared": shared_case,
                "isolated": isolated_case,
                "checks": checks,
                "pass": passed,
            }
        )
        print(f"[case {index}] prompt={prompt!r} prompt_tokens={shared_case['prompt_tokens']}", flush=True)
        print(f"  shared   ={shared_case['token_ids']} reason={shared_case['finish_reason']} usage={shared_case['usage']}", flush=True)
        print(f"  isolated ={isolated_case['token_ids']} reason={isolated_case['finish_reason']} usage={isolated_case['usage']}", flush=True)
        print(f"  {'PASS' if passed else 'FAIL'} {checks}", flush=True)

    payload = {
        "ckpt": args.ckpt,
        "tp_world": args.tp_world,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_ids": eos_ids,
        "eos_source": shared[0]["eos_source"],
        "cases": report,
        "pass": all_ok,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)

    failures = sum(1 for case in report if not case["pass"])
    print("", flush=True)
    if all_ok:
        print(f"ALL PASS ({len(report)} cases)", flush=True)
        return 0
    print(f"{failures} CASE(S) FAILED ({len(report)} cases)", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
