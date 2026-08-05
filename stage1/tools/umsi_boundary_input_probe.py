#!/usr/bin/env python3
"""Lowcontrast-only production-boundary input probe for UMSIPlus.predict_saliency.

This is a one-shot diagnostic tool. It does not modify production behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np


@dataclass
class PredictCapture:
    call_count: int
    input_array: np.ndarray
    restored: bool


@dataclass
class DualCapture:
    padding_call_count: int
    predict_call_count: int
    padded_uint8: np.ndarray
    predict_input: np.ndarray
    padding_restored: bool
    predict_restored: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_native_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def compare_arrays(left: np.ndarray, right: np.ndarray) -> Dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    same_shape = tuple(left_arr.shape) == tuple(right_arr.shape)
    same_dtype = str(left_arr.dtype) == str(right_arr.dtype)
    left_finite = bool(np.isfinite(left_arr).all())
    right_finite = bool(np.isfinite(right_arr).all())
    array_equal = bool(np.array_equal(left_arr, right_arr))

    if same_shape:
        max_abs_diff_float64 = float(
            np.max(np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64)))
        )
    else:
        max_abs_diff_float64 = float("inf")

    left_sha256 = sha256_native_array(left_arr)
    right_sha256 = sha256_native_array(right_arr)

    return {
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "left_shape": list(left_arr.shape),
        "right_shape": list(right_arr.shape),
        "left_dtype": str(left_arr.dtype),
        "right_dtype": str(right_arr.dtype),
        "left_finite": left_finite,
        "right_finite": right_finite,
        "array_equal": array_equal,
        "max_abs_diff_float64": max_abs_diff_float64,
        "left_sha256": left_sha256,
        "right_sha256": right_sha256,
        "hash_equal": left_sha256 == right_sha256,
    }


def build_candidates_from_padded(padded_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    padded = np.asarray(padded_uint8)
    if padded.shape != (256, 256, 3) or padded.dtype != np.uint8:
        raise ValueError("Expected padded uint8 array with shape (256, 256, 3)")

    current_float32 = padded.astype(np.float32)
    current_float32[..., 0] -= 103.939
    current_float32[..., 1] -= 116.779
    current_float32[..., 2] -= 123.68
    current_float32 = np.expand_dims(current_float32, axis=0)

    legacy_float64 = np.zeros((1, 256, 256, 3))
    legacy_float64[0] = padded
    legacy_float64[..., 0] -= 103.939
    legacy_float64[..., 1] -= 116.779
    legacy_float64[..., 2] -= 123.68
    legacy_then_float32 = legacy_float64.astype(np.float32)

    return current_float32, legacy_then_float32


def channel_delta_stats(production_boundary: np.ndarray, legacy_preproc: np.ndarray) -> Dict[str, Any]:
    left = np.asarray(production_boundary)
    right = np.asarray(legacy_preproc)
    if left.shape != (1, 256, 256, 3) or right.shape != (1, 256, 256, 3):
        raise ValueError("Expected both arrays to have shape (1, 256, 256, 3)")

    out: Dict[str, Any] = {}
    channel_names = ["B", "G", "R"]
    for idx, name in enumerate(channel_names):
        lch = left[0, :, :, idx].astype(np.float64)
        rch = right[0, :, :, idx].astype(np.float64)
        delta = lch - rch
        abs_delta = np.abs(delta)
        unequal = abs_delta > 0.0
        uniq = np.unique(delta)
        uniq_sorted = np.sort(uniq)
        uniq_limited = uniq_sorted[:20]
        out[name] = {
            "unequal_count": int(np.count_nonzero(unequal)),
            "total_count": int(delta.size),
            "max_abs_diff_float64": float(abs_delta.max()),
            "mean_abs_diff_float64": float(abs_delta.mean()),
            "sorted_unique_deltas_capped_20": [float(v) for v in uniq_limited],
        }
    return out


def capture_predict_once(
    predict_owner: Any,
    invoke: Callable[[], Any],
) -> Tuple[PredictCapture, Any]:
    """Capture exactly one predict() input and restore the original method."""
    original_predict = predict_owner.predict
    call_count = 0
    captured_input: np.ndarray | None = None

    def proxy(x: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count, captured_input
        call_count += 1
        captured_input = np.array(x, copy=True)
        return original_predict(x, *args, **kwargs)

    predict_owner.predict = proxy
    try:
        invoke_result = invoke()
    finally:
        predict_owner.predict = original_predict

    restored = predict_owner.predict is original_predict
    if call_count != 1:
        raise RuntimeError(f"Expected exactly one predict() call, got {call_count}")
    if captured_input is None:
        raise RuntimeError("Predict input was not captured")

    return (
        PredictCapture(
            call_count=call_count,
            input_array=captured_input,
            restored=restored,
        ),
        invoke_result,
    )


def capture_padding_and_predict_once(
    umsi_module: Any,
    predict_owner: Any,
    invoke: Callable[[], Any],
) -> Tuple[DualCapture, Any]:
    """Capture exactly one _padding output and one model.predict input."""
    original_padding = umsi_module._padding
    original_predict = predict_owner.predict

    padding_call_count = 0
    predict_call_count = 0
    padded_uint8: np.ndarray | None = None
    predict_input: np.ndarray | None = None

    def padding_proxy(*args: Any, **kwargs: Any) -> Any:
        nonlocal padding_call_count, padded_uint8
        padding_call_count += 1
        out = original_padding(*args, **kwargs)
        padded_uint8 = np.array(out, copy=True)
        return out

    def predict_proxy(x: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal predict_call_count, predict_input
        predict_call_count += 1
        predict_input = np.array(x, copy=True)
        return original_predict(x, *args, **kwargs)

    umsi_module._padding = padding_proxy
    predict_owner.predict = predict_proxy
    try:
        invoke_result = invoke()
    finally:
        umsi_module._padding = original_padding
        predict_owner.predict = original_predict

    padding_restored = umsi_module._padding is original_padding
    predict_restored = predict_owner.predict is original_predict

    if padding_call_count != 1:
        raise RuntimeError(f"Expected exactly one _padding() call, got {padding_call_count}")
    if predict_call_count != 1:
        raise RuntimeError(f"Expected exactly one predict() call, got {predict_call_count}")
    if padded_uint8 is None:
        raise RuntimeError("Padding output was not captured")
    if predict_input is None:
        raise RuntimeError("Predict input was not captured")

    return (
        DualCapture(
            padding_call_count=padding_call_count,
            predict_call_count=predict_call_count,
            padded_uint8=padded_uint8,
            predict_input=predict_input,
            padding_restored=padding_restored,
            predict_restored=predict_restored,
        ),
        invoke_result,
    )


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    target_repo = Path(args.target_repo).resolve()
    fixture_path = Path(args.fixture).resolve()
    evidence_npz = Path(args.evidence_npz).resolve()
    weights_path = Path(args.weights).resolve()

    expected_fixture_sha256 = args.fixture_sha256
    expected_evidence_sha256 = args.evidence_sha256
    expected_weights_sha256 = args.weights_sha256

    fixture_sha256 = sha256_file(fixture_path)
    evidence_sha256 = sha256_file(evidence_npz)
    weights_sha256 = sha256_file(weights_path)

    fixture_hash_ok = fixture_sha256 == expected_fixture_sha256
    evidence_hash_ok = evidence_sha256 == expected_evidence_sha256
    weights_hash_ok = weights_sha256 == expected_weights_sha256

    with np.load(evidence_npz, allow_pickle=False) as npz:
        legacy_preproc = np.array(npz["legacy_raw/lowcontrast_preproc"], copy=True)
        legctrl_input = np.array(npz["modern_raw/legctrl_lowcontrast_input"], copy=True)

    if str(target_repo) not in sys.path:
        sys.path.insert(0, str(target_repo))

    umsi_mod = importlib.import_module("saliency.umsi_model")
    module_file = Path(inspect.getfile(umsi_mod)).resolve()
    expected_module_file = (target_repo / "saliency" / "umsi_model.py").resolve()
    import_inside_target = str(module_file).startswith(str(target_repo) + "/")
    import_exact_file = module_file == expected_module_file

    model = umsi_mod.UMSIPlus(str(weights_path))
    predict_owner = model.model

    capture, invoke_result = capture_padding_and_predict_once(
        umsi_mod,
        predict_owner,
        lambda: model.predict_saliency(str(fixture_path), return_classif=True),
    )

    prod_heatmap, prod_classif = invoke_result
    padded_uint8 = capture.padded_uint8
    captured = capture.predict_input

    current_float32, legacy_then_float32 = build_candidates_from_padded(padded_uint8)

    captured_shape = list(captured.shape)
    captured_dtype = str(captured.dtype)
    captured_finite = bool(np.isfinite(captured).all())
    captured_sha256 = sha256_native_array(captured)
    padded_shape = list(np.asarray(padded_uint8).shape)
    padded_dtype = str(np.asarray(padded_uint8).dtype)
    padded_finite = bool(np.isfinite(np.asarray(padded_uint8)).all())
    padded_sha256 = sha256_native_array(padded_uint8)

    cmp_prod_vs_current = compare_arrays(captured, current_float32)
    cmp_prod_vs_legacy_then_f32 = compare_arrays(captured, legacy_then_float32)
    cmp_ref_legacy_vs_current = compare_arrays(legacy_preproc, current_float32)
    cmp_ref_legacy_vs_legacy_then_f32 = compare_arrays(legacy_preproc, legacy_then_float32)
    cmp_ref_legctrl_vs_current = compare_arrays(legctrl_input, current_float32)
    cmp_ref_legctrl_vs_legacy_then_f32 = compare_arrays(legctrl_input, legacy_then_float32)
    cmp_current_vs_legacy_then = compare_arrays(current_float32, legacy_then_float32)
    per_channel_delta = channel_delta_stats(captured, legacy_preproc)

    expected_boundary_sha256 = "6486898bc81a5666f407f0a277138ffe19358b82be942dfe0bf8c99546e3d393"

    result = {
        "status": "ok",
        "hash_checks": {
            "fixture": {
                "expected": expected_fixture_sha256,
                "observed": fixture_sha256,
                "ok": fixture_hash_ok,
            },
            "evidence_npz": {
                "expected": expected_evidence_sha256,
                "observed": evidence_sha256,
                "ok": evidence_hash_ok,
            },
            "weights": {
                "expected": expected_weights_sha256,
                "observed": weights_sha256,
                "ok": weights_hash_ok,
            },
        },
        "import_origin": {
            "module": "saliency.umsi_model",
            "observed_file": str(module_file),
            "expected_file": str(expected_module_file),
            "inside_target_repo": import_inside_target,
            "exact_file_match": import_exact_file,
        },
        "proxy": {
            "padding_call_count": capture.padding_call_count,
            "predict_call_count": capture.predict_call_count,
            "padding_restored": capture.padding_restored,
            "predict_restored": capture.predict_restored,
        },
        "captured_padding": {
            "shape": padded_shape,
            "dtype": padded_dtype,
            "finite": padded_finite,
            "sha256": padded_sha256,
        },
        "captured_input": {
            "shape": captured_shape,
            "dtype": captured_dtype,
            "finite": captured_finite,
            "sha256": captured_sha256,
        },
        "candidate_inputs": {
            "current_float32": {
                "shape": list(current_float32.shape),
                "dtype": str(current_float32.dtype),
                "finite": bool(np.isfinite(current_float32).all()),
                "sha256": sha256_native_array(current_float32),
            },
            "legacy_then_float32": {
                "shape": list(legacy_then_float32.shape),
                "dtype": str(legacy_then_float32.dtype),
                "finite": bool(np.isfinite(legacy_then_float32).all()),
                "sha256": sha256_native_array(legacy_then_float32),
            },
        },
        "comparisons": {
            "production_boundary_vs_current_float32": cmp_prod_vs_current,
            "production_boundary_vs_legacy_then_float32": cmp_prod_vs_legacy_then_f32,
            "legacy_raw/lowcontrast_preproc_vs_current_float32": cmp_ref_legacy_vs_current,
            "legacy_raw/lowcontrast_preproc_vs_legacy_then_float32": cmp_ref_legacy_vs_legacy_then_f32,
            "modern_raw/legctrl_lowcontrast_input_vs_current_float32": cmp_ref_legctrl_vs_current,
            "modern_raw/legctrl_lowcontrast_input_vs_legacy_then_float32": cmp_ref_legctrl_vs_legacy_then_f32,
            "current_float32_vs_legacy_then_float32": cmp_current_vs_legacy_then,
        },
        "production_vs_legacy_preproc_channel_deltas": per_channel_delta,
        "acceptance": {
            "captured_padding_uint8_256x256x3_finite": (
                padded_shape == [256, 256, 3]
                and padded_dtype == "uint8"
                and padded_finite
            ),
            "prod_equals_legacy_then_float32": (
                cmp_prod_vs_legacy_then_f32["array_equal"]
                and cmp_prod_vs_legacy_then_f32["max_abs_diff_float64"] == 0.0
                and cmp_prod_vs_legacy_then_f32["hash_equal"]
            ),
            "prod_equals_both_npz_references": (
                compare_arrays(captured, legacy_preproc)["array_equal"]
                and compare_arrays(captured, legacy_preproc)["max_abs_diff_float64"] == 0.0
                and compare_arrays(captured, legacy_preproc)["hash_equal"]
                and compare_arrays(captured, legctrl_input)["array_equal"]
                and compare_arrays(captured, legctrl_input)["max_abs_diff_float64"] == 0.0
                and compare_arrays(captured, legctrl_input)["hash_equal"]
            ),
            "refs_equal_legacy_then_float32": (
                cmp_ref_legacy_vs_legacy_then_f32["array_equal"]
                and cmp_ref_legacy_vs_legacy_then_f32["max_abs_diff_float64"] == 0.0
                and cmp_ref_legacy_vs_legacy_then_f32["hash_equal"]
                and cmp_ref_legctrl_vs_legacy_then_f32["array_equal"]
                and cmp_ref_legctrl_vs_legacy_then_f32["max_abs_diff_float64"] == 0.0
                and cmp_ref_legctrl_vs_legacy_then_f32["hash_equal"]
            ),
            "prod_not_equal_current_float32": (
                not cmp_prod_vs_current["array_equal"]
                and cmp_prod_vs_current["max_abs_diff_float64"] > 0.0
            ),
            "captured_boundary_sha256_matches_expected": (
                captured_sha256 == expected_boundary_sha256
            ),
            "proxy_calls_once_and_restored": (
                capture.padding_call_count == 1
                and capture.predict_call_count == 1
                and capture.padding_restored
                and capture.predict_restored
            ),
            "current_vs_legacy_then_rounding_scale_only": (
                cmp_current_vs_legacy_then["max_abs_diff_float64"] <= 1e-5
            ),
            "return_shapes_unchanged": (
                list(np.asarray(prod_heatmap).shape) == [800, 1280]
                and list(np.asarray(prod_classif).shape) == [6]
            ),
        },
        "predict_saliency_return": {
            "heatmap_shape": list(np.asarray(prod_heatmap).shape),
            "heatmap_dtype": str(np.asarray(prod_heatmap).dtype),
            "classif_shape": list(np.asarray(prod_classif).shape),
            "classif_dtype": str(np.asarray(prod_classif).dtype),
        },
    }

    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe UMSI production-boundary input")
    p.add_argument(
        "--target-repo",
        default="/Users/Q682780/Thesis_G_layout",
    )
    p.add_argument(
        "--fixture",
        default="/Users/Q682780/Thesis_G_umsi_legacy_resize_fix/golden/fixtures/lowcontrast.png",
    )
    p.add_argument(
        "--fixture-sha256",
        default="4a467bdd72d41b9fd490573468db171a254e7cbf4aa8727ac8fb86bef05a3562",
    )
    p.add_argument(
        "--evidence-npz",
        default="/Users/Q682780/Thesis_G/golden_evidence/umsi_resize_causality_arrays_6a17288_consolidated.npz",
    )
    p.add_argument(
        "--evidence-sha256",
        default="9e933c1071924f180c32cc6327363ca5760b5968d497414a9f7939919b630079",
    )
    p.add_argument(
        "--weights",
        default="/Users/Q682780/Thesis_G/saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5",
    )
    p.add_argument(
        "--weights-sha256",
        default="f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = run_probe(args)
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
