#!/usr/bin/env python3
"""P2 / AG-05 production-postprocess evidence gate for five UMSI fixtures.

The historical P8/P9/P10 gates remain immutable. This successor gate closes
their affine-scale masking gap: the captured production E2E map is never
renormalized before value comparison. The production postprocessor is invoked
directly on both captured production raw output and frozen legacy raw output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from saliency.postprocessing import normalize_saliency_map, postprocess_saliency
from saliency.saliency_features import (
    _detect_peak_mask,
    extract_saliency_features,
)


FEATURE_NAMES = (
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
)
REQUIRED_THRESHOLDS = frozenset(
    {"endpoint_self_abs_max", "e2e_abs_max", "feature_abs_max"}
)


class ContractError(ValueError):
    """Raised when the P2 successor contract is incomplete or inconsistent."""


@dataclass
class GateResult:
    """Machine-readable result for one fixture."""

    passed: bool
    reason: Optional[str] = None
    observed: Dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_and_validate_contract(path: Path) -> Dict[str, Any]:
    """Load the P2 contract and fail closed on any missing acceptance rule."""
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    if contract.get("contract_name") != "umsi_production_postprocess_gate_contract":
        raise ContractError("unexpected P2 contract_name")
    if contract.get("scope") != "P2_AG_05_ONLY":
        raise ContractError("P2 contract scope must be P2_AG_05_ONLY")
    if contract.get("historical_artifacts_immutable") is not True:
        raise ContractError("historical_artifacts_immutable must be true")

    thresholds = contract.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ContractError("thresholds must be an object")
    missing = REQUIRED_THRESHOLDS - set(thresholds)
    if missing:
        raise ContractError(f"missing P2 thresholds: {sorted(missing)}")
    for key in REQUIRED_THRESHOLDS:
        value = thresholds[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"threshold {key!r} must be numeric")
        if not np.isfinite(value) or float(value) < 0.0:
            raise ContractError(f"threshold {key!r} must be finite and non-negative")
    if float(thresholds["endpoint_self_abs_max"]) != 0.0:
        raise ContractError("endpoint_self_abs_max must remain exact (0.0)")
    if thresholds.get("peak_positions_exact") is not True:
        raise ContractError("peak_positions_exact must be true")
    if thresholds.get("peak_count_exact") is not True:
        raise ContractError("peak_count_exact must be true")

    reference = contract.get("historical_reference_contract", {})
    for key in ("filename", "sha256"):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise ContractError(f"historical_reference_contract.{key} is required")

    sources = contract.get("pinned_sources")
    if not isinstance(sources, dict) or not sources:
        raise ContractError("pinned_sources must be a non-empty object")
    for label, definition in sources.items():
        if not isinstance(definition, dict):
            raise ContractError(f"pinned_sources.{label} must be an object")
        for key in ("path", "sha256"):
            if not isinstance(definition.get(key), str) or not definition[key]:
                raise ContractError(f"pinned_sources.{label}.{key} is required")

    return contract


def verify_contract_identities(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    reference_contract_path: Path,
) -> Dict[str, str]:
    """Verify the historical contract and every P2 production source hash."""
    reference = contract["historical_reference_contract"]
    if reference_contract_path.name != reference["filename"]:
        raise ContractError("historical reference contract basename mismatch")
    observed_reference_sha = sha256_file(reference_contract_path)
    if observed_reference_sha != reference["sha256"]:
        raise ContractError("historical reference contract SHA-256 mismatch")

    observed = {"historical_reference_contract": observed_reference_sha}
    for label, definition in contract["pinned_sources"].items():
        source_path = repo_root / definition["path"]
        if not source_path.is_file():
            raise ContractError(f"pinned source missing: {source_path}")
        source_sha = sha256_file(source_path)
        if source_sha != definition["sha256"]:
            raise ContractError(f"pinned source SHA-256 mismatch: {label}")
        observed[label] = source_sha
    return observed


def _map_contract(arr: np.ndarray, expected_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Check the exact public production-map contract without normalization."""
    value = np.asarray(arr)
    observed: Dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "expected_shape": list(expected_shape),
    }
    if value.shape != expected_shape:
        observed.update({"finite": False, "range_passed": False, "passed": False})
        return observed
    try:
        finite = bool(np.isfinite(value).all())
    except TypeError:
        finite = False
    observed["finite"] = finite
    if not finite:
        observed.update({"range_passed": False, "passed": False})
        return observed

    work = value.astype(np.float64)
    vmin = float(work.min())
    vmax = float(work.max())
    constant = vmin == vmax
    range_passed = vmin >= 0.0 and vmax <= 1.0
    endpoint_passed = (vmin == 0.0 and vmax == 1.0) or (
        constant and vmin == 0.0
    )
    observed.update(
        {
            "min": vmin,
            "max": vmax,
            "constant": constant,
            "range_passed": range_passed,
            "endpoint_passed": endpoint_passed,
            "passed": bool(
                value.dtype == np.float32 and range_passed and endpoint_passed
            ),
        }
    )
    return observed


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if np.asarray(left).shape != np.asarray(right).shape:
        return float("inf")
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def evaluate_fixture(
    *,
    production_raw: np.ndarray,
    production_e2e: np.ndarray,
    legacy_raw: np.ndarray,
    original_h: int,
    original_w: int,
    thresholds: Mapping[str, Any],
) -> GateResult:
    """Evaluate AG-05 for one fixture using the real production postprocessor.

    ``production_e2e`` is never normalized by this gate. Any affine scale or
    offset mutation therefore remains visible to the range, self-consistency,
    absolute-value, and dependent feature gates.
    """
    expected_shape = (int(original_h), int(original_w))
    observed: Dict[str, Any] = {}

    try:
        recomputed_production = postprocess_saliency(
            production_raw, original_h, original_w
        )
        reference_e2e = postprocess_saliency(legacy_raw, original_h, original_w)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"PRODUCTION_POSTPROCESS_ERROR: {type(exc).__name__}: {exc}",
            observed=observed,
        )

    output_contract = _map_contract(production_e2e, expected_shape)
    observed["output_contract"] = output_contract
    observed["production_e2e_sha256"] = sha256_array(np.asarray(production_e2e))
    observed["recomputed_production_sha256"] = sha256_array(recomputed_production)
    observed["reference_e2e_sha256"] = sha256_array(reference_e2e)

    endpoint_self_diff = _max_abs_diff(production_e2e, recomputed_production)
    endpoint_self_passed = (
        endpoint_self_diff <= float(thresholds["endpoint_self_abs_max"])
    )
    observed["production_endpoint_self_check"] = {
        "max_abs_diff_float64": endpoint_self_diff,
        "threshold": float(thresholds["endpoint_self_abs_max"]),
        "passed": endpoint_self_passed,
    }

    reference_diff = _max_abs_diff(production_e2e, reference_e2e)
    reference_value_passed = reference_diff <= float(thresholds["e2e_abs_max"])
    observed["reference_value_check"] = {
        "max_abs_diff_float64": reference_diff,
        "threshold": float(thresholds["e2e_abs_max"]),
        "passed": reference_value_passed,
    }

    try:
        production_features = extract_saliency_features(production_e2e)
        reference_features = extract_saliency_features(reference_e2e)
        deviations = {
            name: abs(float(production_features[name]) - float(reference_features[name]))
            for name in FEATURE_NAMES
        }
        feature_max = max(deviations.values())
        peak_count_exact = (
            int(production_features["saliency_peak_count"])
            == int(reference_features["saliency_peak_count"])
        )

        production_normalized = normalize_saliency_map(production_e2e).astype(
            np.float64, copy=False
        )
        reference_normalized = normalize_saliency_map(reference_e2e).astype(
            np.float64, copy=False
        )
        production_peaks = _detect_peak_mask(production_normalized)
        reference_peaks = _detect_peak_mask(reference_normalized)
        peak_positions_exact = bool(np.array_equal(production_peaks, reference_peaks))
        peak_mismatch_pixels = int(np.count_nonzero(production_peaks != reference_peaks))
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"FEATURE_OR_PEAK_ERROR: {type(exc).__name__}: {exc}",
            observed=observed,
        )

    feature_threshold_passed = feature_max <= float(thresholds["feature_abs_max"])
    feature_gate_passed = bool(
        output_contract["passed"]
        and feature_threshold_passed
        and peak_count_exact
    )
    peak_gate_passed = bool(output_contract["passed"] and peak_positions_exact)
    observed["feature_check"] = {
        "production": {name: float(production_features[name]) for name in FEATURE_NAMES},
        "reference": {name: float(reference_features[name]) for name in FEATURE_NAMES},
        "absolute_deviations": deviations,
        "max_abs_deviation": feature_max,
        "threshold": float(thresholds["feature_abs_max"]),
        "threshold_passed": feature_threshold_passed,
        "peak_count_exact": peak_count_exact,
        "passed": feature_gate_passed,
    }
    observed["peak_position_check"] = {
        "positions_exact": peak_positions_exact,
        "mismatch_pixels": peak_mismatch_pixels,
        "production_peak_mask_sha256": sha256_array(production_peaks),
        "reference_peak_mask_sha256": sha256_array(reference_peaks),
        "passed": peak_gate_passed,
    }

    checks = (
        (output_contract["passed"], "PRODUCTION_OUTPUT_CONTRACT_FAIL"),
        (endpoint_self_passed, "PRODUCTION_ENDPOINT_SELF_MISMATCH"),
        (reference_value_passed, "REFERENCE_E2E_VALUE_MISMATCH"),
        (feature_gate_passed, "FIVE_FEATURE_DEVIATION_FAIL"),
        (peak_gate_passed, "EXACT_PEAK_POSITION_FAIL"),
    )
    for passed, reason in checks:
        if not passed:
            return GateResult(passed=False, reason=reason, observed=observed)
    return GateResult(passed=True, observed=observed)


def run_five_fixture_gate(
    *,
    p2_contract: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    fixtures_root: Path,
    npz: Any,
    inference_fn: Callable[[str, Path], Any],
) -> Dict[str, Any]:
    """Run the successor gate over the exact five historical fixtures."""
    from stage1.tools.umsi_five_fixture_gate_harness import (
        _EXPECTED_FIXTURE_NAMES,
        preflight_fixture,
    )
    from stage1.tools.umsi_step2b_gate_runner import PreflightError

    per_fixture: Dict[str, Any] = {}
    for fixture_name in _EXPECTED_FIXTURE_NAMES:
        entry: Dict[str, Any] = {"fixture": fixture_name}
        try:
            preflight = preflight_fixture(
                fixture_name, reference_contract, fixtures_root, npz
            )
            capture = inference_fn(
                fixture_name,
                fixtures_root / reference_contract["fixtures"][fixture_name]["filename"],
            )
        except PreflightError as exc:
            entry.update({"status": "BLOCKED", "blocked_reason": str(exc)})
            per_fixture[fixture_name] = entry
            continue
        except Exception as exc:
            entry.update(
                {
                    "status": "BLOCKED",
                    "blocked_reason": f"INFERENCE_EXCEPTION: {type(exc).__name__}: {exc}",
                }
            )
            per_fixture[fixture_name] = entry
            continue

        reference_member = reference_contract["reference_members"][fixture_name][
            "legacy_raw_saliency"
        ]
        expected_shape = reference_contract["expected_reference_shapes"][
            "legacy_e2e_saliency"
        ]["per_fixture_shape"][fixture_name]
        result = evaluate_fixture(
            production_raw=capture.raw_saliency,
            production_e2e=capture.e2e_saliency,
            legacy_raw=np.asarray(npz[reference_member]),
            original_h=int(expected_shape[0]),
            original_w=int(expected_shape[1]),
            thresholds=p2_contract["thresholds"],
        )
        entry.update(
            {
                "status": "PASS" if result.passed else "FAILED",
                "preflight": preflight,
                "passed": result.passed,
                "reason": result.reason,
                "observed": result.observed,
            }
        )
        per_fixture[fixture_name] = entry

    statuses = [per_fixture[name]["status"] for name in _EXPECTED_FIXTURE_NAMES]
    if "BLOCKED" in statuses:
        overall = "BLOCKED"
    elif all(status == "PASS" for status in statuses):
        overall = "PASS"
    else:
        overall = "FAILED"
    return {
        "overall_verdict": overall,
        "scope": "P2_AG_05_ONLY",
        "fixture_order": list(_EXPECTED_FIXTURE_NAMES),
        "per_fixture": per_fixture,
        "thresholds": dict(p2_contract["thresholds"]),
    }


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--reference-contract", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--evidence-npz", required=True, type=Path)
    parser.add_argument("--fixtures-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    p2_contract = load_and_validate_contract(args.contract)
    identities = verify_contract_identities(
        p2_contract,
        repo_root=repo_root,
        reference_contract_path=args.reference_contract,
    )

    from stage1.tools.umsi_five_fixture_gate_harness import (
        capture_full_inference,
        load_and_validate_five_fixture_contract as load_reference_contract,
    )

    reference_contract = load_reference_contract(args.reference_contract)
    for path, expected_name, expected_sha in (
        (args.weights, reference_contract["weights_filename"], reference_contract["weights_sha256"]),
        (args.evidence_npz, reference_contract["evidence_npz_filename"], reference_contract["evidence_npz_sha256"]),
    ):
        if path.name != expected_name or not path.is_file():
            raise ContractError(f"artifact identity mismatch or missing: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ContractError(f"artifact SHA-256 mismatch: {path.name}")
        identities[path.name] = observed_sha

    from saliency.umsi_model import UMSIPlus

    model = UMSIPlus(str(args.weights))
    with np.load(str(args.evidence_npz)) as npz:
        report = run_five_fixture_gate(
            p2_contract=p2_contract,
            reference_contract=reference_contract,
            fixtures_root=args.fixtures_root,
            npz=npz,
            inference_fn=lambda _name, path: capture_full_inference(model, str(path)),
        )
    report["identities"] = identities
    report["p2_contract_sha256"] = sha256_file(args.contract)
    report["repository_head"] = _git_head(repo_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"overall_verdict": report["overall_verdict"], "output": str(args.output)}))
    return 0 if report["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
