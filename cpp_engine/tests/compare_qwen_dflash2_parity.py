#!/usr/bin/env python3
"""Compare named DFlash2 parity stages and stop at the first failed stage."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from qwen_dflash2_tensor_file import read_tensor_file


@dataclass(frozen=True)
class Threshold:
    atol: float
    rtol: float
    cosine: float


FP16_DEFAULT = Threshold(atol=3.0e-2, rtol=3.0e-2, cosine=0.9995)
FP32_DEFAULT = Threshold(atol=1.0e-1, rtol=3.0e-2, cosine=0.9990)
FP32_RESIDUAL = Threshold(atol=16.0, rtol=3.0e-2, cosine=0.9995)
FP16_STRICT = Threshold(atol=4.0e-3, rtol=4.0e-3, cosine=0.99999)


def threshold_for(name: str, dtype: np.dtype) -> Threshold:
    if name in {
        "target_taps",
        "noise.embedding",
        "noise.tokens",
        "topk.local.tokens",
        "topk.global.tokens",
        "selector.path",
    }:
        return FP16_STRICT
    if dtype == np.dtype("<f2"):
        # One ULP grows with FP16 magnitude. Relative error and cosine carry the
        # semantic gate; use an absolute floor large enough for values near 8K.
        return Threshold(atol=8.0, rtol=FP16_DEFAULT.rtol,
                         cosine=FP16_DEFAULT.cosine)
    if name.endswith(".residual") or name.endswith(".branch_f32") or \
            name.endswith(".down") or name.endswith(".finish"):
        return FP32_RESIDUAL
    return FP32_DEFAULT


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    if denominator == 0.0:
        return 1.0 if np.array_equal(left64, right64) else 0.0
    return float(np.dot(left64, right64) / denominator)


def describe(name: str, actual: np.ndarray, expected: np.ndarray) -> tuple[bool, str]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, (
            f"shape/dtype actual={actual.shape}/{actual.dtype} "
            f"expected={expected.shape}/{expected.dtype}"
        )
    if actual.dtype.kind in "iu":
        mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
        if mismatch.size == 0:
            return True, "exact"
        flat = int(mismatch[0])
        index = np.unravel_index(flat, actual.shape)
        return False, (
            f"mismatches={mismatch.size} first={index} "
            f"actual={actual[index]} expected={expected[index]}"
        )

    actual32 = actual.astype(np.float32)
    expected32 = expected.astype(np.float32)
    finite_actual = np.isfinite(actual32)
    finite_expected = np.isfinite(expected32)
    nonfinite_actual = int((~finite_actual).sum())
    nonfinite_expected = int((~finite_expected).sum())
    if nonfinite_actual or nonfinite_expected:
        return False, (
            f"nonfinite actual={nonfinite_actual} expected={nonfinite_expected}"
        )
    difference = np.abs(actual32 - expected32)
    scale = np.maximum(np.abs(expected32), 1.0e-8)
    relative = difference / scale
    flat = int(np.argmax(difference))
    index = np.unravel_index(flat, actual.shape)
    max_abs = float(difference[index])
    mean_abs = float(difference.mean())
    max_rel = float(relative.max())
    cosine = cosine_similarity(actual32, expected32)
    threshold = threshold_for(name, actual.dtype)
    close = np.isclose(
        actual32, expected32, atol=threshold.atol, rtol=threshold.rtol
    )
    passed = bool(close.all()) and cosine >= threshold.cosine
    detail = (
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"max_rel={max_rel:.6g} cosine={cosine:.9f} worst={index} "
        f"actual={float(actual32[index]):.6g} "
        f"expected={float(expected32[index]):.6g} "
        f"outside={int((~close).sum())}/{actual.size}"
    )
    return passed, detail


def top2_margin(values: np.ndarray) -> str:
    if values.ndim != 2 or values.shape[1] < 2:
        return ""
    sorted_values = np.sort(values.astype(np.float32), axis=1)
    margins = sorted_values[:, -1] - sorted_values[:, -2]
    return " margins=" + ",".join(f"{value:.6g}" for value in margins)


def topk_cutoff_equivalent(
    actual_tokens: np.ndarray,
    expected_tokens: np.ndarray,
    actual_file,
    expected_file,
    stage: str,
    tolerance: float = 1.0e-2,
) -> tuple[bool, str] | None:
    if stage not in {"topk.local.tokens", "topk.global.tokens"}:
        return None
    prefix = stage.removesuffix(".tokens")
    actual_logits = actual_file.tensors.get(prefix + ".logits")
    expected_logits = expected_file.tensors.get(prefix + ".logits")
    if actual_logits is None or expected_logits is None:
        return None
    details = []
    for row in range(actual_tokens.shape[0]):
        actual_set = set(map(int, actual_tokens[row]))
        expected_set = set(map(int, expected_tokens[row]))
        if actual_set == expected_set:
            continue
        actual_cutoff = float(np.min(actual_logits[row]))
        expected_cutoff = float(np.min(expected_logits[row]))
        missing = expected_set - actual_set
        added = actual_set - expected_set
        if not missing or not added:
            return False, f"row={row} non-exchange top-k mismatch"
        for token in missing:
            if abs(float(actual_file.tensors["logits.local"][row, token]) - actual_cutoff) > tolerance:
                return False, f"row={row} missing token={token} is outside cutoff band"
        for token in added:
            local_token = token
            if local_token >= actual_file.tensors["logits.local"].shape[1]:
                return False, f"row={row} added token={token} is outside local shard"
            if abs(float(expected_file.tensors["logits.local"][row, local_token]) - expected_cutoff) > tolerance:
                return False, f"row={row} added token={token} is outside cutoff band"
        details.append(
            f"row={row} cutoff_swap missing={sorted(missing)} added={sorted(added)}"
        )
    return True, "; ".join(details) if details else "same candidate sets"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--allow-extra", action="store_true")
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--skip-prefix", action="append", default=[])
    args = parser.parse_args()

    reference = read_tensor_file(args.reference)
    actual = read_tensor_file(args.actual)
    if (
        reference.position_offset != actual.position_offset
        or reference.anchor_token != actual.anchor_token
    ):
        print(
            "[FAIL] metadata "
            f"reference=({reference.position_offset},{reference.anchor_token}) "
            f"actual=({actual.position_offset},{actual.anchor_token})"
        )
        return 1

    missing = [name for name in reference.tensors if name not in actual.tensors]
    extra = [name for name in actual.tensors if name not in reference.tensors]
    if missing:
        print(f"[FAIL] missing stages: {', '.join(missing)}")
        return 1
    if extra and not args.allow_extra:
        print(f"[FAIL] unexpected stages: {', '.join(extra)}")
        return 1

    compared = 0
    for index, (name, expected) in enumerate(reference.tensors.items()):
        if name in args.skip or any(name.startswith(prefix) for prefix in args.skip_prefix):
            print(f"[SKIP] {index:03d} {name}")
            continue
        compared += 1
        passed, detail = describe(name, actual.tensors[name], expected)
        if not passed:
            equivalent = topk_cutoff_equivalent(
                actual.tensors[name], expected, actual, reference, name
            )
            if equivalent is not None and equivalent[0]:
                passed = True
                detail = "near-tie " + equivalent[1]
        suffix = ""
        if name.endswith("topk.global.logits") or name.endswith("topk.local.logits"):
            suffix = top2_margin(expected)
        print(f"[{'PASS' if passed else 'FAIL'}] {index:03d} {name}: {detail}{suffix}")
        if not passed:
            print(f"first_divergent_stage={name}")
            return 1

    print(f"[PASS] compared {compared} DFlash2 stages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
