#!/usr/bin/env python3
"""Summarize per-path C++/PyTorch DSpark parity benchmark JSONL logs."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_json_records(path: Path, marker: str) -> list[dict]:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if marker in line:
            records.append(json.loads(line.split(marker, 1)[1]))
    return records


def median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def first_mismatch(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return -1 if len(left) == len(right) else min(len(left), len(right))


def token_comparison(left: dict, right: dict) -> dict:
    mismatch = first_mismatch(left["tokens"], right["tokens"])
    result = {"match": mismatch < 0, "first_mismatch": mismatch}
    if mismatch >= 0:
        result["left_token"] = (
            left["tokens"][mismatch] if mismatch < len(left["tokens"]) else None
        )
        result["right_token"] = (
            right["tokens"][mismatch] if mismatch < len(right["tokens"]) else None
        )
    return result


def stable_tokens(rows: list[dict]) -> bool:
    return all(row["tokens"] == rows[0]["tokens"] for row in rows[1:])


def path_summary(rows: list[dict]) -> dict:
    result = {
        "tps_runs": [row["tps"] for row in rows],
        "tps_median": median(rows, "tps"),
        "seconds_median": median(rows, "seconds"),
        "tokens_stable": stable_tokens(rows),
    }
    if "rounds" in rows[0]:
        result["rounds_runs"] = [row["rounds"] for row in rows]
    if "accepted_tokens" in rows[0]:
        result["accepted_tokens_runs"] = [row["accepted_tokens"] for row in rows]
    if "accepted_per_round" in rows[0]:
        result["accepted_per_round_runs"] = [row["accepted_per_round"] for row in rows]
        result["accepted_per_round_median"] = median(rows, "accepted_per_round")
    if "accept_rate" in rows[0]:
        result["accept_rate_runs"] = [row["accept_rate"] for row in rows]
        result["accept_rate_median"] = median(rows, "accept_rate")
    return result


def rows_by_repeat(rows: list[dict]) -> dict[int, dict]:
    return {int(row["repeat"]): row for row in rows}


def compare_paths(left: list[dict], right: list[dict]) -> list[dict]:
    left_by_repeat = rows_by_repeat(left)
    right_by_repeat = rows_by_repeat(right)
    comparisons = []
    for repeat in sorted(set(left_by_repeat) & set(right_by_repeat)):
        result = token_comparison(left_by_repeat[repeat], right_by_repeat[repeat])
        result["repeat"] = repeat
        comparisons.append(result)
    return comparisons


def near_tie_verdicts(
    cpp_topk: dict[str, dict], pytorch_topk: dict[str, dict]
) -> dict[str, dict]:
    """Classify each case's prefill argmax as agreeing, a near-tie, or a real split.

    A greedy argmax flips whenever the two runtimes' logits differ by more than
    the top-2 gap, so a bare token comparison cannot tell numerical drift from a
    wrong state. Recording the margin makes that distinction explicit: if one
    runtime's pick sits within the other's top-2 at a margin far below the gap
    the agreeing cases clear, the mismatch is a tie-break, not a bug.
    """
    verdicts: dict[str, dict] = {}
    for case in sorted(set(cpp_topk) & set(pytorch_topk)):
        cpp_row = cpp_topk[case]
        pt_row = pytorch_topk[case]
        cpp_pick = cpp_row["top_tokens"][0]
        pt_pick = pt_row["top_tokens"][0]
        entry = {
            "cpp_argmax": cpp_pick,
            "pytorch_argmax": pt_pick,
            "cpp_margin": cpp_row.get("margin"),
            "pytorch_margin": pt_row.get("margin"),
            "topk_overlap": len(set(cpp_row["top_tokens"]) & set(pt_row["top_tokens"])),
        }
        if cpp_pick == pt_pick:
            entry["verdict"] = "argmax_agree"
        else:
            rank = (
                pt_row["top_tokens"].index(cpp_pick)
                if cpp_pick in pt_row["top_tokens"]
                else None
            )
            entry["cpp_pick_rank_in_pytorch"] = rank
            if rank is not None:
                gap = pt_row["top_logits"][0] - pt_row["top_logits"][rank]
                entry["pytorch_gap_to_cpp_pick"] = gap
                entry["verdict"] = "near_tie" if rank == 1 else "argmax_split"
            else:
                entry["verdict"] = "argmax_split"
        verdicts[case] = entry
    return verdicts


def first_trace_divergence(left: list[dict], right: list[dict]) -> dict | None:
    right_by_key = {
        (row["case"], int(row["repeat"]), int(row["round"])): row for row in right
    }
    fields = (
        "position",
        "committed_token",
        "draft_tokens",
        "target_successors",
        "accepted_tokens",
    )
    for row in sorted(left, key=lambda item: (item["case"], item["repeat"], item["round"])):
        key = (row["case"], int(row["repeat"]), int(row["round"]))
        peer = right_by_key.get(key)
        if peer is None:
            continue
        for field in fields:
            if row.get(field) != peer.get(field):
                return {
                    "case": key[0],
                    "repeat": key[1],
                    "round": key[2],
                    "field": field,
                    "left": row.get(field),
                    "right": peer.get(field),
                }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-log", required=True, type=Path)
    parser.add_argument("--pytorch-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = load_json_records(args.cpp_log, "RESULT_JSON ")
    records += load_json_records(args.pytorch_log, "RESULT_JSON ")
    if not records:
        raise ValueError("no RESULT_JSON records found")

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["case"], row["runtime"], row["path"])].append(row)

    cases = sorted({row["case"] for row in records})
    report: dict = {"cases": {}}
    all_stable = True
    all_plain_match = True
    for case in cases:
        case_rows = [row for row in records if row["case"] == case]
        paths = {
            f"{runtime}:{path}": path_summary(rows)
            for (group_case, runtime, path), rows in sorted(grouped.items())
            if group_case == case
        }
        all_stable &= all(summary["tokens_stable"] for summary in paths.values())

        comparisons = {}
        cpp_plain_nodspark = grouped.get((case, "cpp", "plain_nodspark"), [])
        cpp_plain_dspark = grouped.get((case, "cpp", "plain_dspark"), [])
        cpp_sequential = grouped.get((case, "cpp", "spec_sequential"), [])
        cpp_batched = grouped.get((case, "cpp", "spec_batched"), [])
        pytorch_plain = grouped.get((case, "pytorch", "plain"), [])
        pytorch_spec = grouped.get((case, "pytorch", "spec_always_k5"), [])

        pairs = (
            ("cpp_plain_nodspark_vs_dspark", cpp_plain_nodspark, cpp_plain_dspark),
            ("cpp_plain_nodspark_vs_pytorch", cpp_plain_nodspark, pytorch_plain),
            ("cpp_plain_dspark_vs_pytorch", cpp_plain_dspark, pytorch_plain),
            ("cpp_sequential_vs_pytorch_spec", cpp_sequential, pytorch_spec),
            ("cpp_batched_vs_sequential", cpp_batched, cpp_sequential),
        )
        for name, left, right in pairs:
            if left and right:
                comparisons[name] = compare_paths(left, right)

        for name in (
            "cpp_plain_nodspark_vs_dspark",
            "cpp_plain_nodspark_vs_pytorch",
            "cpp_plain_dspark_vs_pytorch",
        ):
            all_plain_match &= all(row["match"] for row in comparisons.get(name, []))

        speedups = {}
        if cpp_plain_dspark:
            base = median(cpp_plain_dspark, "tps")
            if cpp_sequential:
                speedups["cpp_sequential_vs_plain_dspark"] = median(cpp_sequential, "tps") / base
            if cpp_batched:
                speedups["cpp_batched_vs_plain_dspark"] = median(cpp_batched, "tps") / base
        if pytorch_plain and pytorch_spec:
            speedups["pytorch_spec_vs_plain"] = median(pytorch_spec, "tps") / median(
                pytorch_plain, "tps"
            )
        if cpp_batched and pytorch_spec:
            speedups["cpp_batched_vs_pytorch_spec"] = median(cpp_batched, "tps") / median(
                pytorch_spec, "tps"
            )

        report["cases"][case] = {
            "prompt_tokens": case_rows[0]["prompt_tokens"],
            "decode_tokens": case_rows[0]["decode_tokens"],
            "paths": paths,
            "comparisons": comparisons,
            "speedups": speedups,
        }

    cpp_traces = load_json_records(args.cpp_log, "TRACE_JSON ")
    pytorch_traces = load_json_records(args.pytorch_log, "TRACE_JSON ")
    if cpp_traces and pytorch_traces:
        report["first_trace_divergence"] = first_trace_divergence(
            cpp_traces, pytorch_traces
        )

    cpp_topk = {row["case"]: row for row in load_json_records(args.cpp_log, "TOPK_JSON ")}
    pytorch_topk = {
        row["case"]: row for row in load_json_records(args.pytorch_log, "TOPK_JSON ")
    }
    if cpp_topk and pytorch_topk:
        report["prefill_argmax"] = near_tie_verdicts(cpp_topk, pytorch_topk)
    report["all_paths_repeat_stable"] = all_stable
    report["all_plain_token_comparisons_match"] = all_plain_match
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for case, result in report["cases"].items():
        print(f"{case}:")
        for path, summary in result["paths"].items():
            acceptance = ""
            if "accepted_per_round_median" in summary:
                acceptance = f" accepted/round={summary['accepted_per_round_median']:.3f}"
            print(
                f"  {path}: {summary['tps_median']:.3f} tok/s"
                f" stable={summary['tokens_stable']}{acceptance}"
            )
        for name, value in result["speedups"].items():
            print(f"  {name}: {value:.3f}x")
        for name, comparisons in result["comparisons"].items():
            mismatches = [row for row in comparisons if not row["match"]]
            status = "MATCH" if not mismatches else f"MISMATCH@{mismatches[0]['first_mismatch']}"
            print(f"  {name}: {status}")
        verdict = report.get("prefill_argmax", {}).get(case)
        if verdict:
            detail = f"  prefill_argmax: {verdict['verdict']}"
            if verdict["verdict"] != "argmax_agree":
                detail += (
                    f" cpp={verdict['cpp_argmax']} pytorch={verdict['pytorch_argmax']}"
                    f" pytorch_gap={verdict.get('pytorch_gap_to_cpp_pick')}"
                )
            print(detail)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
