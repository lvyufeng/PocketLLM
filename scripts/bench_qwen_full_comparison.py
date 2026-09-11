#!/usr/bin/env python3
"""Aggregate PocketLLM and vLLM benchmark artifacts into one comparison table.

This script does not run models. It reads the raw JSON summaries produced by
  * scripts/bench_qwen_long_context.py   (PocketLLM plain long context)
  * scripts/bench_qwen_mtp.py            (PocketLLM plain/MTP K sweep)
  * scripts/bench_qwen_drafters.py       (PocketLLM MTP/DSpark/DFlash2)
  * scripts/bench_qwen_vllm_long_context.py (vLLM plain/MTP)
and emits comparison.json, comparison.csv and comparison.md.

Keeping aggregation separate from execution means a long matrix can be resumed
and re-summarized without re-timing anything, and failed cases stay visible
with their reason instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "engine", "mode", "mtp_k", "drafter", "workload", "prompt_tokens",
    "generated_tokens", "prefill_seconds", "prefill_tps", "ttft_seconds",
    "decode_seconds", "decode_tokens", "decode_tps", "e2e_wall_seconds",
    "gpu_memory_bytes", "gpu_memory_instrument", "accept_length", "draft_match_rate",
    "rank_token_parity", "cross_engine_token_parity", "first_divergence",
    "status", "failure_reason", "provenance",
]


def blank_row() -> dict[str, Any]:
    return {field: None for field in FIELDS}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sampled_peak_bytes(case: dict[str, Any]) -> Any:
    """Externally sampled peak, or None when the run predates the sampler.

    Both runners now record `gpu_memory` from the shared nvidia-smi sampler.
    Older result files carry only engine self-reports (PocketLLM) or a single
    post-run reading (vLLM); those are not comparable across engines, so they
    are reported as absent rather than silently mixed into the same column.
    """
    memory = case.get("gpu_memory")
    if isinstance(memory, dict):
        return memory.get("max_peak_bytes")
    return None


def memory_instrument(case: dict[str, Any]) -> str:
    memory = case.get("gpu_memory")
    if isinstance(memory, dict):
        return str(memory.get("instrument", "unknown"))
    return "self-report"


def rows_from_pocketllm_long_context(path: Path) -> list[dict[str, Any]]:
    payload = load(path)
    provenance = {
        "git_revision": None,
        "binary_sha256": None,
        "source": str(path),
    }
    rows: list[dict[str, Any]] = []
    for case in payload.get("results", payload if isinstance(payload, list) else []):
        row = blank_row()
        runtime = case.get("runtime", {})
        prov = dict(provenance)
        prov["git_revision"] = case.get("git_revision")
        prov["binary_sha256"] = (case.get("binary_metadata") or {}).get("binary_sha256")
        prov["token_fixture_sha256"] = case.get("token_fixture_sha256")
        row.update({
            "engine": "pocketllm",
            "mode": "plain",
            "mtp_k": 0,
            "workload": f"ctx{case.get('prompt_tokens')}",
            "prompt_tokens": case.get("prompt_tokens"),
            "generated_tokens": case.get("generated_tokens"),
            "prefill_seconds": runtime.get("prefill_seconds"),
            "prefill_tps": runtime.get("prefill_tokens_per_s"),
            "ttft_seconds": runtime.get("prefill_seconds"),
            "decode_seconds": runtime.get("decode_seconds"),
            "decode_tokens": runtime.get("decode_token_count"),
            "decode_tps": runtime.get("decode_tokens_per_s"),
            "e2e_wall_seconds": runtime.get("wall"),
            "gpu_memory_bytes": sampled_peak_bytes(case),
            "gpu_memory_instrument": memory_instrument(case),
            "rank_token_parity": case.get("rank_token_parity"),
            "status": "ok",
            "provenance": prov,
            "_tokens": case.get("tokens"),
        })
        rows.append(row)
    return rows


def rows_from_vllm(path: Path) -> list[dict[str, Any]]:
    payload = load(path)
    rows: list[dict[str, Any]] = []
    for case in payload.get("results", []):
        row = blank_row()
        row.update({
            "engine": "vllm",
            "mode": case.get("mode", payload.get("mode", "plain")),
            "mtp_k": case.get("mtp_k", payload.get("mtp_tokens", 0)),
            "workload": f"ctx{case.get('prompt_tokens')}",
            "prompt_tokens": case.get("prompt_tokens"),
            "generated_tokens": case.get("generated_tokens"),
            "prefill_seconds": case.get("prefill_seconds"),
            "prefill_tps": case.get("prefill_tps"),
            "ttft_seconds": case.get("ttft_seconds"),
            "decode_seconds": case.get("decode_seconds"),
            "decode_tokens": case.get("decode_tokens"),
            "decode_tps": case.get("decode_tps"),
            "e2e_wall_seconds": case.get("e2e_wall_seconds"),
            "gpu_memory_bytes": sampled_peak_bytes(case),
            "gpu_memory_instrument": memory_instrument(case),
            "status": case.get("status", "ok"),
            "failure_reason": case.get("failure_reason"),
            "provenance": {
                "source": str(path),
                "vllm_python": payload.get("python"),
                "token_fixture_sha256": case.get("token_fixture_sha256"),
                "max_num_seqs": payload.get("max_num_seqs"),
                "gpu_memory_utilization": payload.get("gpu_memory_utilization"),
            },
            "_tokens": case.get("tokens"),
        })
        rows.append(row)
    return rows


def rows_from_pocketllm_mtp(path: Path) -> list[dict[str, Any]]:
    payload = load(path)
    rows: list[dict[str, Any]] = []
    for group in payload.get("results", []):
        for case in group.get("cases", []):
            runtime = case.get("runtime", {})
            mode = case.get("mode", "plain")
            row = blank_row()
            row.update({
                "engine": "pocketllm",
                "mode": "plain" if mode == "plain" else "mtp",
                "mtp_k": int(runtime.get("mtp_tokens") or 0),
                "workload": f"ctx{group.get('prompt_tokens')}",
                "prompt_tokens": group.get("prompt_tokens"),
                "generated_tokens": runtime.get("generated_tokens"),
                "prefill_seconds": runtime.get("prefill_seconds"),
                "prefill_tps": runtime.get("prefill_tokens_per_s"),
                "ttft_seconds": runtime.get("prefill_seconds"),
                "decode_seconds": runtime.get("decode_seconds"),
                "decode_tokens": runtime.get("decode_token_count"),
                "decode_tps": runtime.get("decode_tokens_per_s"),
                "e2e_wall_seconds": runtime.get("wall"),
                # This runner reports engine self-accounting, not the shared
                # sampler, so it is labeled rather than mixed into the
                # cross-engine memory comparison.
                "gpu_memory_bytes": case.get("max_gpu_memory_used_bytes"),
                "gpu_memory_instrument": "self-report",
                "accept_length": runtime.get("spec_accept_length"),
                "draft_match_rate": runtime.get("mtp_accept_rate"),
                "rank_token_parity": case.get("rank_token_parity"),
                "status": "ok",
                "provenance": {
                    "source": str(path),
                    "binary": payload.get("binary"),
                    "kv_cache_dtype": payload.get("kv_cache_dtype"),
                    "prefill_chunk_tokens": payload.get("prefill_chunk_tokens"),
                    "draft_seconds": runtime.get("mtp_draft_seconds"),
                    "verify_seconds": runtime.get("mtp_verify_seconds"),
                    "replay_seconds": runtime.get("mtp_replay_seconds"),
                    "proposed_drafts": runtime.get("mtp_proposed_drafts"),
                    "correct_drafts": runtime.get("mtp_correct_drafts"),
                    "speedup_vs_plain_wall": case.get("speedup_vs_plain_wall"),
                    "speedup_vs_plain_decode": case.get("speedup_vs_plain_decode"),
                },
                "_tokens": case.get("tokens"),
            })
            rows.append(row)
    return rows


def rows_from_pocketllm_drafters(path: Path) -> list[dict[str, Any]]:
    payload = load(path)
    rows: list[dict[str, Any]] = []
    for case in payload.get("results", []):
        row = blank_row()
        row.update({
            "engine": "pocketllm",
            "mode": case.get("mode", "plain"),
            "drafter": case.get("drafter"),
            "mtp_k": case.get("mtp_tokens", 0),
            "workload": case.get("dataset") or case.get("workload"),
            "prompt_tokens": case.get("mean_prompt_tokens") or case.get("prompt_tokens"),
            "generated_tokens": case.get("generated_tokens"),
            "decode_tps": case.get("decode_tps"),
            "e2e_wall_seconds": case.get("wall") or case.get("e2e_wall_seconds"),
            "accept_length": case.get("accept_length"),
            "draft_match_rate": case.get("draft_match_rate"),
            "rank_token_parity": case.get("rank_token_parity"),
            "cross_engine_token_parity": case.get("token_parity"),
            "first_divergence": case.get("first_divergence"),
            "status": case.get("status", "ok"),
            "failure_reason": case.get("failure_reason"),
            "provenance": {"source": str(path)},
        })
        rows.append(row)
    return rows


LOADERS = {
    "pocketllm-long-context": rows_from_pocketllm_long_context,
    "pocketllm-mtp": rows_from_pocketllm_mtp,
    "pocketllm-drafters": rows_from_pocketllm_drafters,
    "vllm": rows_from_vllm,
}


def cross_engine_parity(rows: list[dict[str, Any]]) -> None:
    """Compare each vLLM row's tokens against the PocketLLM row at the same shape."""
    baseline: dict[tuple[Any, Any, Any], list[int]] = {}
    for row in rows:
        if row["engine"] != "pocketllm":
            continue
        tokens = row.get("_tokens")
        if tokens:
            baseline[(row["workload"], row["mode"], row["mtp_k"])] = tokens
    # PocketLLM plain is the reference for every engine at the same prompt.
    plain = {
        key[0]: value for key, value in baseline.items() if key[1] == "plain"
    }
    for row in rows:
        tokens = row.get("_tokens")
        reference = plain.get(row["workload"])
        if not tokens or not reference:
            continue
        limit = min(len(tokens), len(reference))
        divergence = next(
            (i for i in range(limit) if tokens[i] != reference[i]), None
        )
        if row["engine"] == "pocketllm" and row["mode"] == "plain":
            row["cross_engine_token_parity"] = True
            continue
        row["cross_engine_token_parity"] = divergence is None
        row["first_divergence"] = divergence


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], title: str) -> str:
    header = ("| engine | mode | K | drafter | workload | prompt | gen | "
              "prefill s | prefill tok/s | decode s | decode tok/s | wall s | "
              "accept | parity | status |")
    sep = "|" + "---|" * 15
    lines = [f"### {title}", "", header, sep]
    for row in rows:
        lines.append(
            f"| {row['engine']} | {row['mode']} | {fmt(row['mtp_k'], 0)} | "
            f"{row['drafter'] or '-'} | {row['workload']} | "
            f"{fmt(row['prompt_tokens'], 0)} | {fmt(row['generated_tokens'], 0)} | "
            f"{fmt(row['prefill_seconds'], 3)} | {fmt(row['prefill_tps'])} | "
            f"{fmt(row['decode_seconds'], 3)} | {fmt(row['decode_tps'], 3)} | "
            f"{fmt(row['e2e_wall_seconds'], 3)} | {fmt(row['accept_length'], 3)} | "
            f"{fmt(row['cross_engine_token_parity'])} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"


def ratio_table(rows: list[dict[str, Any]]) -> str:
    by_key: dict[tuple[Any, Any, Any], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["workload"], row["mode"], row["mtp_k"])
        by_key.setdefault(key, {})[row["engine"]] = row
    lines = ["### PocketLLM / vLLM ratios", "",
             "| workload | mode | K | prefill ratio | decode ratio | wall ratio |",
             "|---|---|---|---|---|---|"]
    for (workload, mode, k), engines in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        pocket, vllm = engines.get("pocketllm"), engines.get("vllm")
        if not pocket or not vllm:
            continue

        def div(a: Any, b: Any, invert: bool = False) -> str:
            if not a or not b:
                return "n/a"
            return f"{(b / a) if invert else (a / b):.3f}"

        lines.append(
            f"| {workload} | {mode} | {fmt(k, 0)} | "
            f"{div(pocket['prefill_tps'], vllm['prefill_tps'])} | "
            f"{div(pocket['decode_tps'], vllm['decode_tps'])} | "
            f"{div(pocket['e2e_wall_seconds'], vllm['e2e_wall_seconds'], invert=True)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], metavar="KIND=PATH",
                        help=f"one of {sorted(LOADERS)} plus a JSON path")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(".scratch/qwen_full_comparison"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.input:
        kind, _, path = spec.partition("=")
        if kind not in LOADERS:
            raise SystemExit(f"unknown input kind: {kind}")
        target = Path(path)
        if not target.exists():
            print(f"[skip] missing {target}")
            continue
        loaded = LOADERS[kind](target)
        print(f"[load] {kind} {target} -> {len(loaded)} rows")
        rows.extend(loaded)

    cross_engine_parity(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    (args.out_dir / "comparison.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8")

    with (args.out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in clean:
            out = dict(row)
            out["provenance"] = json.dumps(row.get("provenance"))
            writer.writerow(out)

    plain = [r for r in rows if r["mode"] == "plain" and not r["drafter"]]
    mtp = [r for r in rows if r["mode"] in ("mtp", "native_mtp") or (r["mtp_k"] or 0) > 0]
    drafters = [r for r in rows if r["drafter"]]

    doc = ["# PocketLLM vs vLLM TP4 comparison", "",
           "All rows: TP4 on 4x RTX 2080 Ti, concurrency=1, serial execution,",
           "same real tokenizer fixture, full 64-layer target, greedy, TG128.", ""]
    if plain:
        doc.append(markdown_table(plain, "Table 1 - plain (no speculation)"))
        doc.append(ratio_table(plain))
    if mtp:
        doc.append(markdown_table(mtp, "Table 2 - native MTP K=1/2/3"))
        doc.append(ratio_table(mtp))
    if drafters:
        doc.append(markdown_table(drafters, "Table 3 - MTP / DSpark / DFlash2"))
    (args.out_dir / "comparison.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"[done] wrote {args.out_dir}/comparison.{{json,csv,md}} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
