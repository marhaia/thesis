#!/usr/bin/env python3
"""STAGE 1 STEP 2B P9 CORRIGENDUM — Zero-inference re-evaluator.

Assessment type: POST_HOC_ZERO_INFERENCE_PROTOCOL_CORRIGENDUM

This script re-evaluates the immutable P8 production outputs against
a corrected, source-faithful contract.  It performs:

  - Full integrity preflight of all P8 artifacts and the P9 contract.
  - Gate 1: production saliency vs legacy reference (Pearson/Spearman/SSIM).
  - Gate 2: production classification vs legacy reference (max_abs_diff).
  - Diag-1: negative-control check — production vs M_LC reference.
  - Diag-2: classification drift record — production vs M_LC classification.

No TensorFlow, Keras, or saliency model is imported.
No inference is performed.
The P8 bundle is opened read-only; it is never modified or copied.
A new output bundle is written only when this script is explicitly invoked.

Exit codes:
  0  STEP_2B_P9_CORRIGENDUM_PASS
  2  STEP_2B_P9_CORRIGENDUM_FAIL_* or INCONCLUSIVE_NEGATIVE_CONTROL
  1  STEP_2B_P9_CORRIGENDUM_INVALID_INTEGRITY or internal error
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

# ---------------------------------------------------------------------------
# Guard: no TF/Keras/UMSI at import time
# ---------------------------------------------------------------------------
# (These names must not appear as active imports anywhere in this file.)
_FORBIDDEN_MODULES = frozenset(
    {"tensorflow", "keras", "saliency", "umsi_model", "torch"}
)
for _m in _FORBIDDEN_MODULES:
    if _m in sys.modules:  # pragma: no cover
        raise ImportError(
            f"Forbidden module already imported: {_m!r}. "
            "This evaluator must not share a process with TF/Keras/UMSI."
        )

# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


# ---------------------------------------------------------------------------
# Conclusion strings
# ---------------------------------------------------------------------------

_PREFIX = "STEP_2B_P9_CORRIGENDUM_"

CONCLUSION_PASS = _PREFIX + "PASS"
CONCLUSION_FAIL_GATE_1 = _PREFIX + "FAIL_GATE_1"
CONCLUSION_FAIL_GATE_2 = _PREFIX + "FAIL_GATE_2"
CONCLUSION_FAIL_GATE_1_GATE_2 = _PREFIX + "FAIL_GATE_1_GATE_2"
CONCLUSION_INCONCLUSIVE = _PREFIX + "INCONCLUSIVE_NEGATIVE_CONTROL"
CONCLUSION_INVALID = _PREFIX + "INVALID_INTEGRITY"

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_PREFLIGHT_KEYS = frozenset({
    "p8_gate_report_sha256",
    "p8_production_outputs_npz_sha256",
    "p8_contract_sha256",
    "p8_runner_sha256",
    "p8_production_source_sha256",
    "evidence_npz_sha256",
    "weights_sha256",
    "fixture_sha256_lowcontrast",
    "prod_raw_saliency_sha256",
    "prod_classif_sha256",
})
_REQUIRED_REFERENCE_MEMBERS = frozenset({
    "legacy_raw_saliency",
    "legacy_classification",
    "m_lc_raw_saliency_neg_control",
    "m_lc_classification_diag",
})
_REQUIRED_GATE_KEYS = frozenset({"gate_1", "gate_2"})
_REQUIRED_THRESHOLD_KEYS = frozenset({
    "pearson_min", "spearman_min", "ssim_min", "cli_abs_max",
})
_REQUIRED_ARRAY_SPECS = {
    "prod_raw": {"shape": (512, 512, 1), "dtype": "float32"},
    "prod_clf": {"shape": (6,), "dtype": "float32"},
    "ref_leg_raw": {"shape": (512, 512), "dtype": "float32"},
    "ref_leg_clf": {"shape": (6,), "dtype": "float64"},
    "ref_mlc_raw": {"shape": (512, 512), "dtype": "float32"},
    "ref_mlc_clf": {"shape": (6,), "dtype": "float64"},
}

# ---------------------------------------------------------------------------
# Contract loading and validation
# ---------------------------------------------------------------------------

class ContractError(Exception):
    pass


def load_and_validate_contract(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        c = json.load(fh)

    for section in (
        "assessment_metadata",
        "pinned_p9_evaluator",
        "pinned_p8_identities",
        "reference_members",
        "acceptance_gates",
        "diagnostic_gates",
        "frozen_thresholds",
        "verdict_logic",
    ):
        if section not in c:
            raise ContractError(f"Missing required contract section: {section!r}")

    meta = c["assessment_metadata"]
    if meta.get("assessment_type") != "POST_HOC_ZERO_INFERENCE_PROTOCOL_CORRIGENDUM":
        raise ContractError(
            "assessment_metadata.assessment_type must be "
            "POST_HOC_ZERO_INFERENCE_PROTOCOL_CORRIGENDUM"
        )
    if meta.get("independent_inference_performed") is not False:
        raise ContractError(
            "assessment_metadata.independent_inference_performed must be false"
        )

    ids = c["pinned_p8_identities"]
    missing = _REQUIRED_PREFLIGHT_KEYS - set(ids.keys())
    if missing:
        raise ContractError(f"pinned_p8_identities missing keys: {sorted(missing)}")
    for k in _REQUIRED_PREFLIGHT_KEYS:
        v = ids[k]
        if not isinstance(v, str) or not _SHA256_RE.match(v):
            raise ContractError(
                f"pinned_p8_identities[{k!r}] must be 64 lowercase hex chars; "
                f"got {v!r}"
            )

    missing = _REQUIRED_REFERENCE_MEMBERS - set(c["reference_members"].keys())
    if missing:
        raise ContractError(f"reference_members missing keys: {sorted(missing)}")

    missing = _REQUIRED_GATE_KEYS - set(c["acceptance_gates"].keys())
    if missing:
        raise ContractError(f"acceptance_gates missing keys: {sorted(missing)}")

    thr = c["frozen_thresholds"]
    missing = _REQUIRED_THRESHOLD_KEYS - set(thr.keys())
    if missing:
        raise ContractError(f"frozen_thresholds missing keys: {sorted(missing)}")

    g1 = c["acceptance_gates"]["gate_1"]
    if "authoritative_shared_postprocess" not in g1:
        raise ContractError("gate_1 missing authoritative_shared_postprocess")
    order = g1["authoritative_shared_postprocess"].get("order")
    if order != ["squeeze", "identical_resize", "identical_min_max_to_0_1"]:
        raise ContractError(f"gate_1 postprocess order unexpected: {order!r}")

    g2 = c["acceptance_gates"]["gate_2"]
    if g2.get("comparison") != "cast_both_to_float64":
        raise ContractError(
            "gate_2.comparison must be cast_both_to_float64"
        )

    return c


# ---------------------------------------------------------------------------
# Integrity preflight
# ---------------------------------------------------------------------------

class IntegrityError(Exception):
    pass


def run_integrity_preflight(
    *,
    p8_gate_report_path: Path,
    p8_npz_path: Path,
    p8_contract_path: Path,
    contract_path: Path,
    expected_contract_sha: str,
    p8_runner_path: Path,
    p8_source_path: Path,
    weights_path: Path,
    evidence_npz_path: Path,
    fixture_path: Path,
    evaluator_path: Path,
    expected_evaluator_sha: str,
    contract_pinned_evaluator_sha: str,
    pinned: Dict[str, str],
) -> Dict[str, str]:
    """Verify all P8 artifact and evaluator identities.

    Raises IntegrityError on any mismatch.
    Returns a dict of observed SHAs for inclusion in the report.

    Evaluator SHA check is performed FIRST to prevent a modified evaluator
    from passing all other checks.  The actual on-disk evaluator SHA, the
    CLI --expected-evaluator-sha, and the contract-pinned
    pinned_p9_evaluator.evaluator_sha256 must all agree.

    Repository HEAD is NOT checked; adding the P9 files changes HEAD and
    must not invalidate P9.  The production source and model artifact SHAs
    are verified instead.
    """
    observed: Dict[str, str] = {}

    # 1. Evaluator self-check — first
    if not _SHA256_RE.match(expected_evaluator_sha):
        raise IntegrityError(
            f"--expected-evaluator-sha is not 64 lowercase hex: "
            f"{expected_evaluator_sha!r}"
        )
    obs_eval_sha = sha256_file(evaluator_path)
    if obs_eval_sha != expected_evaluator_sha:
        raise IntegrityError(
            f"Evaluator SHA mismatch: expected {expected_evaluator_sha!r}, "
            f"got {obs_eval_sha!r}"
        )
    observed["evaluator_sha256"] = obs_eval_sha

    # 1b. Contract-pinned evaluator SHA must agree with the actual/CLI SHA.
    if not _SHA256_RE.match(contract_pinned_evaluator_sha):
        raise IntegrityError(
            "contract pinned_p9_evaluator.evaluator_sha256 is not 64 "
            f"lowercase hex: {contract_pinned_evaluator_sha!r}"
        )
    if contract_pinned_evaluator_sha != obs_eval_sha:
        raise IntegrityError(
            "Evaluator identity mismatch: actual evaluator SHA "
            f"{obs_eval_sha!r} does not match contract-pinned "
            f"pinned_p9_evaluator.evaluator_sha256 "
            f"{contract_pinned_evaluator_sha!r}"
        )

    # 2. Contract
    if not _SHA256_RE.match(expected_contract_sha):
        raise IntegrityError(
            f"--contract-sha is not 64 lowercase hex: {expected_contract_sha!r}"
        )
    obs_contract_sha = sha256_file(contract_path)
    if obs_contract_sha != expected_contract_sha:
        raise IntegrityError(
            f"Contract SHA mismatch: expected {expected_contract_sha!r}, "
            f"got {obs_contract_sha!r}"
        )
    observed["contract_sha256"] = obs_contract_sha

    # 3. File identity checks against pinned values
    file_checks: List[Tuple[str, Path, str]] = [
        ("p8_gate_report", p8_gate_report_path, pinned["p8_gate_report_sha256"]),
        ("p8_production_outputs_npz", p8_npz_path,
         pinned["p8_production_outputs_npz_sha256"]),
        ("p8_contract", p8_contract_path, pinned["p8_contract_sha256"]),
        ("p8_runner", p8_runner_path, pinned["p8_runner_sha256"]),
        ("p8_source", p8_source_path, pinned["p8_production_source_sha256"]),
        ("evidence_npz", evidence_npz_path, pinned["evidence_npz_sha256"]),
        ("weights", weights_path, pinned["weights_sha256"]),
        ("fixture", fixture_path, pinned["fixture_sha256_lowcontrast"]),
    ]
    for label, path, expected in file_checks:
        if not path.is_file():
            raise IntegrityError(f"Required file missing: {path}")
        obs = sha256_file(path)
        if obs != expected:
            raise IntegrityError(
                f"{label} SHA mismatch: expected {expected!r}, got {obs!r}"
            )
        observed[label + "_sha256"] = obs

    # 4. P8 gate_report content: confirm historical conclusion on record
    with open(p8_gate_report_path, "r", encoding="utf-8") as fh:
        p8_report = json.load(fh)
    historical_conclusion = p8_report.get("conclusion", "")
    if historical_conclusion != "STEP_2B_FAIL_GATE_A_GATE_C":
        raise IntegrityError(
            f"P8 gate_report conclusion is unexpected: {historical_conclusion!r}. "
            "Expected STEP_2B_FAIL_GATE_A_GATE_C."
        )
    observed["p8_historical_conclusion"] = historical_conclusion

    return observed


# ---------------------------------------------------------------------------
# Array loading and structural validation
# ---------------------------------------------------------------------------

def load_and_validate_arrays(
    *,
    p8_npz_path: Path,
    evidence_npz_path: Path,
    reference_members: Dict[str, str],
    pinned: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Load and validate production and reference arrays.

    Raises IntegrityError if any pinned SHA, shape, dtype or finite check fails.
    Returns a dict keyed by internal label.
    """
    try:
        with np.load(p8_npz_path, allow_pickle=False) as p8:
            prod_raw = np.array(p8["production_raw_saliency"], copy=True)
            prod_clf = np.array(p8["production_classif"], copy=True)
    except KeyError as exc:
        raise IntegrityError(
            f"Missing required key in P8 production_outputs.npz: {exc}"
        ) from exc

    try:
        with np.load(evidence_npz_path, allow_pickle=False) as ref:
            ref_leg_raw = np.array(ref[reference_members["legacy_raw_saliency"]], copy=True)
            ref_leg_clf = np.array(ref[reference_members["legacy_classification"]], copy=True)
            ref_mlc_raw = np.array(ref[reference_members["m_lc_raw_saliency_neg_control"]], copy=True)
            ref_mlc_clf = np.array(ref[reference_members["m_lc_classification_diag"]], copy=True)
    except KeyError as exc:
        raise IntegrityError(
            f"Missing required key in evidence NPZ: {exc}"
        ) from exc

    arrays: Dict[str, np.ndarray] = {
        "prod_raw": prod_raw,
        "prod_clf": prod_clf,
        "ref_leg_raw": ref_leg_raw,
        "ref_leg_clf": ref_leg_clf,
        "ref_mlc_raw": ref_mlc_raw,
        "ref_mlc_clf": ref_mlc_clf,
    }

    # Validate production array SHAs against pinned values
    obs_raw_sha = sha256_array(prod_raw)
    if obs_raw_sha != pinned["prod_raw_saliency_sha256"]:
        raise IntegrityError(
            f"prod_raw_saliency SHA mismatch: expected "
            f"{pinned['prod_raw_saliency_sha256']!r}, got {obs_raw_sha!r}"
        )
    obs_clf_sha = sha256_array(prod_clf)
    if obs_clf_sha != pinned["prod_classif_sha256"]:
        raise IntegrityError(
            f"prod_classif SHA mismatch: expected "
            f"{pinned['prod_classif_sha256']!r}, got {obs_clf_sha!r}"
        )

    # Structural checks: shape, dtype, finite
    specs = {
        "prod_raw":     {"shape": (512, 512, 1), "dtype": "float32"},
        "prod_clf":     {"shape": (6,),           "dtype": "float32"},
        "ref_leg_raw":  {"shape": (512, 512),     "dtype": "float32"},
        "ref_leg_clf":  {"shape": (6,),           "dtype": "float64"},
        "ref_mlc_raw":  {"shape": (512, 512),     "dtype": "float32"},
        "ref_mlc_clf":  {"shape": (6,),           "dtype": "float64"},
    }
    for label, arr in arrays.items():
        spec = specs[label]
        if tuple(arr.shape) != spec["shape"]:
            raise IntegrityError(
                f"{label} shape {arr.shape} != expected {spec['shape']}"
            )
        if str(arr.dtype) != spec["dtype"]:
            raise IntegrityError(
                f"{label} dtype {arr.dtype} != expected {spec['dtype']}"
            )
        if not np.isfinite(arr).all():
            raise IntegrityError(f"{label} contains non-finite values")

    return arrays


# ---------------------------------------------------------------------------
# Gate B / P8-equivalent postprocessing helpers
# ---------------------------------------------------------------------------

def _squeeze_to_2d(arr: np.ndarray, label: str) -> np.ndarray:
    a = np.squeeze(arr)
    if a.ndim != 2:
        raise ValueError(
            f"_squeeze_to_2d({label!r}): expected 2D, got {arr.shape} -> {a.shape}"
        )
    return a


def _apply_constant_map_policy(
    arr: np.ndarray, label: str
) -> Tuple[np.ndarray, bool]:
    """Min-max normalize to [0,1] with constant-map policy (identical to P8)."""
    a = arr.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"_apply_constant_map_policy({label!r}): non-finite values")
    vmin = float(a.min())
    vmax = float(a.max())
    span = vmax - vmin
    if span == 0.0:
        return np.zeros_like(a, dtype=np.float64), True
    return (a - vmin) / span, False


def _windowed_ssim(x: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
    """Windowed SSIM using scipy.ndimage.uniform_filter (identical to P8)."""
    from scipy.ndimage import uniform_filter

    K1 = float(params["K1"])
    K2 = float(params["K2"])
    win_size = int(params["win_size"])
    mode = str(params["boundary"])
    data_range = float(params["data_range"])

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    mu_x = uniform_filter(xf, size=win_size, mode=mode)
    mu_y = uniform_filter(yf, size=win_size, mode=mode)
    mu_xsq = mu_x * mu_x
    mu_ysq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = uniform_filter(xf * xf, size=win_size, mode=mode) - mu_xsq
    sy2 = uniform_filter(yf * yf, size=win_size, mode=mode) - mu_ysq
    sxy = uniform_filter(xf * yf, size=win_size, mode=mode) - mu_xy
    num = (2.0 * mu_xy + C1) * (2.0 * sxy + C2)
    den = (mu_xsq + mu_ysq + C1) * (sx2 + sy2 + C2)
    return float((num / den).mean())


def _compute_saliency_parity_metrics(
    prod_raw: np.ndarray,
    ref_raw: np.ndarray,
    ssim_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Compute normalized saliency parity metrics (Gate 1 and Diag-1 shared path).

    Returns (obs_dict, constant_map_abort) where constant_map_abort=True means
    both metrics and verdict are undefined.
    """
    from scipy.stats import pearsonr, spearmanr

    prod_2d = _squeeze_to_2d(prod_raw, "prod_saliency")
    ref_2d = _squeeze_to_2d(ref_raw, "ref_saliency")

    obs: Dict[str, Any] = {
        "prod_shape": list(prod_2d.shape),
        "ref_shape": list(ref_2d.shape),
        "raw_max_abs_diff_float64": float(
            np.max(np.abs(prod_2d.astype(np.float64) - ref_2d.astype(np.float64)))
        ),
        "raw_mean_abs_diff_float64": float(
            np.mean(np.abs(prod_2d.astype(np.float64) - ref_2d.astype(np.float64)))
        ),
        "prod_sha256": sha256_array(prod_2d),
        "ref_sha256": sha256_array(ref_2d),
    }

    if prod_2d.shape != ref_2d.shape:
        obs["error"] = f"shape mismatch: {prod_2d.shape} vs {ref_2d.shape}"
        return obs, False

    prod_norm, prod_const = _apply_constant_map_policy(prod_2d, "prod_norm")
    ref_norm, ref_const = _apply_constant_map_policy(ref_2d, "ref_norm")
    obs["prod_constant_map"] = prod_const
    obs["ref_constant_map"] = ref_const

    if prod_const or ref_const:
        obs["pearson"] = None
        obs["spearman"] = None
        obs["windowed_ssim"] = None
        return obs, True

    prod_flat = prod_norm.ravel()
    ref_flat = ref_norm.ravel()
    pearson_r, _ = pearsonr(prod_flat, ref_flat)
    spearman_r, _ = spearmanr(prod_flat, ref_flat)
    ssim_val = _windowed_ssim(prod_norm, ref_norm, ssim_params)

    obs["pearson"] = float(pearson_r)
    obs["spearman"] = float(spearman_r)
    obs["windowed_ssim"] = float(ssim_val)
    return obs, False


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    passed: Optional[bool]
    reason: Optional[str] = None
    observed: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate evaluators
# ---------------------------------------------------------------------------

def evaluate_gate_1(
    prod_raw: np.ndarray,
    ref_leg_raw: np.ndarray,
    gate_def: Dict[str, Any],
    frozen_thresholds: Dict[str, Any],
) -> GateResult:
    """Gate 1: production saliency vs legacy reference.

    Identical calculation to P8 Gate B.
    Selected because it tests source-faithful parity, not because its P8
    outcome was known in advance.
    """
    name = "gate_1"
    ssim_params = gate_def["authoritative_shared_postprocess"].get(
        "ssim_params",
        {"K1": 0.01, "K2": 0.03, "win_size": 7,
         "boundary": "reflect", "data_range": 1.0},
    )

    try:
        obs, constant_abort = _compute_saliency_parity_metrics(
            prod_raw, ref_leg_raw, ssim_params
        )
    except Exception as exc:
        return GateResult(name=name, passed=False,
                          reason=f"GATE_1_COMPUTE_ERROR: {exc}")

    if "error" in obs:
        return GateResult(name=name, passed=False,
                          reason=f"GATE_1_ERROR: {obs['error']}", observed=obs)

    if constant_abort:
        return GateResult(name=name, passed=False,
                          reason="GATE_1_UNDEFINED_CORRELATION_CONSTANT_MAP",
                          observed=obs)

    pearson_min = float(frozen_thresholds["pearson_min"])
    spearman_min = float(frozen_thresholds["spearman_min"])
    ssim_min = float(frozen_thresholds["ssim_min"])

    r_p = obs.get("pearson", 0.0)
    r_s = obs.get("spearman", 0.0)
    r_ssim = obs.get("windowed_ssim", 0.0)

    if not (np.isfinite(r_p) and np.isfinite(r_s) and np.isfinite(r_ssim)):
        return GateResult(name=name, passed=False,
                          reason="GATE_1_NON_FINITE_METRIC", observed=obs)
    if r_p < pearson_min:
        return GateResult(name=name, passed=False,
                          reason=f"GATE_1_PEARSON_FAIL: {r_p:.10f} < {pearson_min}",
                          observed=obs)
    if r_s < spearman_min:
        return GateResult(name=name, passed=False,
                          reason=f"GATE_1_SPEARMAN_FAIL: {r_s:.10f} < {spearman_min}",
                          observed=obs)
    if r_ssim < ssim_min:
        return GateResult(name=name, passed=False,
                          reason=f"GATE_1_SSIM_FAIL: {r_ssim:.10f} < {ssim_min}",
                          observed=obs)
    return GateResult(name=name, passed=True, observed=obs)


def evaluate_gate_2(
    prod_clf: np.ndarray,
    ref_leg_clf: np.ndarray,
    gate_def: Dict[str, Any],
    frozen_thresholds: Dict[str, Any],
) -> GateResult:
    """Gate 2: production classification vs legacy reference.

    Identical calculation to P8 Gate D.
    Both arrays cast to float64 before comparison; native dtypes recorded.
    Selected because it tests source-faithful parity, not because its P8
    outcome was known in advance.
    """
    name = "gate_2"
    prod = np.asarray(prod_clf)
    ref = np.asarray(ref_leg_clf)

    obs: Dict[str, Any] = {
        "prod_shape": list(prod.shape),
        "ref_shape": list(ref.shape),
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
        "prod_sha256": sha256_array(prod),
        "ref_sha256": sha256_array(ref),
        "comparison": "cast_both_to_float64",
    }

    if tuple(prod.shape) != tuple(ref.shape):
        return GateResult(name=name, passed=False,
                          reason=f"GATE_2_SHAPE_MISMATCH: {prod.shape} vs {ref.shape}",
                          observed=obs)

    max_diff = float(
        np.max(np.abs(prod.astype(np.float64) - ref.astype(np.float64)))
    )
    obs["max_abs_diff_float64"] = max_diff
    cli_abs_max = float(frozen_thresholds["cli_abs_max"])
    obs["cli_abs_max_threshold"] = cli_abs_max

    if max_diff > cli_abs_max:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_2_CLI_ABS_MAX_FAIL: {max_diff:.8e} > {cli_abs_max}",
            observed=obs,
        )
    return GateResult(name=name, passed=True, observed=obs)


def evaluate_diag_1(
    prod_raw: np.ndarray,
    ref_mlc_raw: np.ndarray,
    frozen_thresholds: Dict[str, Any],
    ssim_params: Dict[str, Any],
) -> GateResult:
    """Diag-1: M_LC negative-control diagnostic.

    Records how production saliency compares to the M_LC (broken-resize)
    reference.  Expected outcome: production does NOT match M_LC thresholds.
    If production unexpectedly satisfies all positive-parity thresholds
    against M_LC, the result is INCONCLUSIVE_NEGATIVE_CONTROL (blocking).
    """
    name = "diag_1"
    try:
        obs, constant_abort = _compute_saliency_parity_metrics(
            prod_raw, ref_mlc_raw, ssim_params
        )
    except Exception as exc:
        return GateResult(name=name, passed=None,
                          reason=f"DIAG_1_COMPUTE_ERROR: {exc}")

    obs["negative_control_purpose"] = (
        "Production must NOT reproduce the M_LC broken-resize output. "
        "Expected: metrics below positive-parity thresholds."
    )

    if constant_abort:
        obs["alarm"] = False
        obs["alarm_reason"] = "constant_map_abort; negative control indeterminate"
        return GateResult(name=name, passed=None, observed=obs,
                          reason="DIAG_1_CONSTANT_MAP_ABORT")

    pearson_min = float(frozen_thresholds["pearson_min"])
    spearman_min = float(frozen_thresholds["spearman_min"])
    ssim_min = float(frozen_thresholds["ssim_min"])

    r_p = obs.get("pearson")
    r_s = obs.get("spearman")
    r_ssim = obs.get("windowed_ssim")

    unexpected_pass = (
        r_p is not None and np.isfinite(r_p) and r_p >= pearson_min
        and r_s is not None and np.isfinite(r_s) and r_s >= spearman_min
        and r_ssim is not None and np.isfinite(r_ssim) and r_ssim >= ssim_min
    )

    obs["alarm"] = unexpected_pass
    if unexpected_pass:
        obs["alarm_reason"] = (
            "Production unexpectedly satisfies ALL positive-parity thresholds "
            "against the M_LC broken-resize reference. This indicates the "
            "production model may be using the wrong resize semantics."
        )
        return GateResult(
            name=name,
            passed=False,
            reason="DIAG_1_UNEXPECTED_M_LC_PASS",
            observed=obs,
        )

    obs["alarm_reason"] = "none — negative control is discriminative as expected"
    return GateResult(name=name, passed=True, observed=obs)


def evaluate_diag_2(
    prod_clf: np.ndarray,
    ref_mlc_clf: np.ndarray,
) -> GateResult:
    """Diag-2: classification M_LC proximity, record-only.

    Compares production classification to M_LC classification after float64
    conversion.  Does not require dtype identity.  Does not contribute to
    the PASS/FAIL verdict.
    """
    name = "diag_2"
    prod = np.asarray(prod_clf)
    ref = np.asarray(ref_mlc_clf)
    obs: Dict[str, Any] = {
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
        "comparison": "cast_both_to_float64",
        "prod_sha256": sha256_array(prod),
        "ref_sha256": sha256_array(ref),
        "records_only": True,
        "no_dtype_requirement": True,
    }
    if tuple(prod.shape) != tuple(ref.shape):
        obs["max_abs_diff_float64"] = None
        obs["note"] = f"shape mismatch {prod.shape} vs {ref.shape}"
        return GateResult(name=name, passed=None, observed=obs,
                          reason="DIAG_2_SHAPE_MISMATCH")
    max_diff = float(
        np.max(np.abs(prod.astype(np.float64) - ref.astype(np.float64)))
    )
    obs["max_abs_diff_float64"] = max_diff
    obs["note"] = (
        "Records cross-environment drift between production float32 output "
        "and M_LC float64 reference. No pass/fail verdict."
    )
    return GateResult(name=name, passed=None, observed=obs)


# ---------------------------------------------------------------------------
# Safe evaluator wrapper
# ---------------------------------------------------------------------------

def _safe_evaluate(name: str, fn: Any, *args: Any) -> GateResult:
    try:
        result = fn(*args)
    except Exception as exc:
        return GateResult(
            name=name, passed=False,
            reason="EVALUATOR_EXCEPTION",
            observed={
                "evaluator_exception_type": type(exc).__name__,
                "evaluator_exception_message": str(exc),
            },
        )
    if not isinstance(result, GateResult) or result.name != name:
        return GateResult(
            name=name, passed=False,
            reason="EVALUATOR_INVALID_RETURN",
            observed={"returned": repr(result)},
        )
    return result


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------

def assemble_verdict(
    gate_1: GateResult,
    gate_2: GateResult,
    diag_1: GateResult,
    p9_contract_sha: str,
    p8_source_sha: str,
    p8_weights_sha: str,
    pinned: Dict[str, str],
) -> Tuple[str, str, Dict[str, Any]]:
    """Determine the conclusion string, norm-release status, and verdict metadata.

    Returns (conclusion, norm_release_status, verdict_meta_dict).

    Decision logic, in strict priority order (highest first):
      - INVALID:       production source SHA or weights SHA does not match
                       the P8-pinned identity.  Checked FIRST, before any
                       gate or negative-control outcome is consulted.  This
                       makes PASS + BLOCKED structurally impossible: norms
                       release is only even considered once this identity
                       check has already passed.
      - INCONCLUSIVE:  diag_1 fired an unexpected M_LC pass, OR diag_1 could
                       not be evaluated to a valid, discriminative
                       negative-control result for any other reason
                       (constant-map abort, compute error, evaluator
                       exception, invalid return, or a malformed/missing
                       alarm state).  Always blocking.
      - FAIL:          gate_1 or gate_2 failed.
      - PASS:          gate_1 passed, gate_2 passed, diag_1 is valid and
                       discriminative (passed is True, reason is None,
                       observed['alarm'] is False).  Since the identity
                       check above already passed to reach this branch,
                       PASS always implies UNBLOCKED and never BLOCKED.
    """
    sources_match = (p8_source_sha == pinned["p8_production_source_sha256"])
    weights_match = (p8_weights_sha == pinned["weights_sha256"])

    if not sources_match or not weights_match:
        conclusion = CONCLUSION_INVALID
        norms = "BLOCKED"
        mismatches: List[str] = []
        if not sources_match:
            mismatches.append(
                "production source SHA mismatch (pinned="
                f"{pinned['p8_production_source_sha256']!r}, "
                f"observed={p8_source_sha!r})"
            )
        if not weights_match:
            mismatches.append(
                "weights SHA mismatch (pinned="
                f"{pinned['weights_sha256']!r}, observed={p8_weights_sha!r})"
            )
        norms_rationale = (
            "Identity invalidation: " + "; ".join(mismatches) + ". "
            "Norm computation must not proceed against an unverified "
            "production artifact set. Identities do not match; this "
            "conclusion is never combined with a PASS or UNBLOCKED state."
        )
        verdict_meta: Dict[str, Any] = {
            "conclusion": conclusion,
            "p8_historical_result": "STEP_2B_FAIL_GATE_A_GATE_C",
            "p8_historical_result_preserved": True,
            "independent_inference_performed": False,
            "saliency_norms_release_status": norms,
            "saliency_norms_rationale": norms_rationale,
            "source_sha_match": sources_match,
            "weights_sha_match": weights_match,
            "direct_activation_capture_status": "NOT_PROVEN",
        }
        return conclusion, norms, verdict_meta

    # From this point, production source and weights identities are
    # confirmed to match the P8-pinned values.
    diag_1_discriminative = (
        diag_1.passed is True
        and diag_1.reason is None
        and diag_1.observed.get("alarm") is False
    )

    if diag_1.reason == "DIAG_1_UNEXPECTED_M_LC_PASS":
        conclusion = CONCLUSION_INCONCLUSIVE
        norms = "BLOCKED"
        negative_control_status = "ALARM_TRIGGERED"
    elif not diag_1_discriminative:
        conclusion = CONCLUSION_INCONCLUSIVE
        norms = "BLOCKED"
        negative_control_status = "UNEVALUABLE"
    elif not gate_1.passed and not gate_2.passed:
        conclusion = CONCLUSION_FAIL_GATE_1_GATE_2
        norms = "BLOCKED"
        negative_control_status = "DISCRIMINATIVE"
    elif not gate_1.passed:
        conclusion = CONCLUSION_FAIL_GATE_1
        norms = "BLOCKED"
        negative_control_status = "DISCRIMINATIVE"
    elif not gate_2.passed:
        conclusion = CONCLUSION_FAIL_GATE_2
        norms = "BLOCKED"
        negative_control_status = "DISCRIMINATIVE"
    else:
        conclusion = CONCLUSION_PASS
        norms = "UNBLOCKED"
        negative_control_status = "DISCRIMINATIVE"

    norms_rationale: str
    if conclusion == CONCLUSION_PASS:
        norms_rationale = (
            "P9 acceptance gates passed. Production source and weights "
            "match P8-pinned identities. Saliency norm computation may proceed."
        )
    elif conclusion == CONCLUSION_INCONCLUSIVE:
        if negative_control_status == "ALARM_TRIGGERED":
            norms_rationale = (
                "M_LC negative control was unexpectedly matched. "
                "Production resize semantics require investigation."
            )
        else:
            norms_rationale = (
                "The M_LC negative control could not be evaluated to a "
                f"discriminative result (diag_1 reason: {diag_1.reason!r}). "
                "A non-discriminative negative control cannot support a "
                "PASS conclusion."
            )
    else:
        norms_rationale = "One or more acceptance gates failed."

    activation_capture_status = "NOT_PROVEN"
    activation_capture_note = (
        "Direct runtime activation capture was not performed in this "
        "zero-inference re-evaluation. The combination of: (1) pinned and "
        "verified production source SHA, (2) static code-path trace confirming "
        "LegacyBilinearUpSampling2D uses tf.raw_ops.ResizeBilinear, "
        "(3) near-perfect functional parity with the legacy reference "
        "(Gate 1 Pearson > 0.999), and (4) M_LC negative-control discrimination, "
        "constitutes sufficient evidence for thesis-level source-faithful "
        "validation without direct activation capture. A future P10 run with "
        "activation capture would upgrade this from NOT_PROVEN to PROVEN."
    )

    verdict_meta = {
        "conclusion": conclusion,
        "p8_historical_result": "STEP_2B_FAIL_GATE_A_GATE_C",
        "p8_historical_result_preserved": True,
        "independent_inference_performed": False,
        "saliency_norms_release_status": norms,
        "saliency_norms_rationale": norms_rationale,
        "source_sha_match": sources_match,
        "weights_sha_match": weights_match,
        "negative_control_status": negative_control_status,
        "direct_activation_capture_status": activation_capture_status,
        "direct_activation_capture_note": activation_capture_note,
    }
    if conclusion == CONCLUSION_PASS:
        verdict_meta["pass_meaning"] = (
            "The immutable P8 production outputs satisfy the corrected, "
            "source-faithful acceptance semantics (Gate 1: saliency parity "
            "with legacy reference; Gate 2: classification parity with legacy "
            "reference; Diag-1: M_LC negative control valid and "
            "discriminative). P8's historical FAIL conclusion is not "
            "overwritten. No independent new inference or direct activation "
            "capture was performed in this re-evaluation."
        )
    return conclusion, norms, verdict_meta


# ---------------------------------------------------------------------------
# Evidence bundle writer
# ---------------------------------------------------------------------------

def write_evidence_bundle(
    *,
    output_dir: Path,
    bundle_name: str,
    report: Dict[str, Any],
) -> Path:
    """Write gate_report_p9.json atomically.

    Does NOT write production_outputs.npz — the P8 bundle is the authoritative
    source of production data and must not be duplicated or modified.
    Refuses to overwrite an existing bundle.
    """
    bundle_path = output_dir / bundle_name
    if bundle_path.exists():
        raise FileExistsError(
            f"P9 bundle already exists: {bundle_path}. "
            "Refusing to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = bundle_path.parent / (bundle_name + f".tmp_{os.getpid()}")
    if tmp_path.exists():
        raise FileExistsError(f"Temp dir already exists: {tmp_path}")
    tmp_path.mkdir(parents=False, exist_ok=False)
    try:
        with open(tmp_path / "gate_report_p9.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
        os.rename(str(tmp_path), str(bundle_path))
    except Exception:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(str(tmp_path))
        raise
    return bundle_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_p9_corrigendum(args: argparse.Namespace) -> int:
    """Execute the P9 corrigendum re-evaluation.

    No TF, Keras or UMSI module is imported.
    No inference is performed.
    """
    import platform

    # ── Load and validate contract ──────────────────────────────────────────
    contract_path = Path(args.contract).resolve()
    try:
        contract = load_and_validate_contract(contract_path)
    except (ContractError, Exception) as exc:
        print(f"BLOCKED: Contract validation failed: {exc}", file=sys.stderr)
        return 1

    pinned = contract["pinned_p8_identities"]
    contract_pinned_evaluator_sha = contract["pinned_p9_evaluator"]["evaluator_sha256"]
    frozen_thresholds = contract["frozen_thresholds"]
    reference_members = contract["reference_members"]
    ssim_params = contract["acceptance_gates"]["gate_1"][
        "authoritative_shared_postprocess"
    ].get(
        "ssim_params",
        {"K1": 0.01, "K2": 0.03, "win_size": 7,
         "boundary": "reflect", "data_range": 1.0},
    )

    # ── Resolve paths ───────────────────────────────────────────────────────
    p8_bundle_path = Path(args.p8_bundle).resolve()
    p8_gate_report_path = p8_bundle_path / "gate_report.json"
    p8_npz_path = p8_bundle_path / "production_outputs.npz"
    p8_contract_path = Path(args.p8_contract).resolve()
    p8_runner_path = Path(args.p8_runner).resolve()
    p8_source_path = Path(args.p8_source).resolve()
    evidence_npz_path = Path(args.evidence_npz).resolve()
    weights_path = Path(args.weights).resolve()
    fixture_path = Path(args.fixture).resolve()
    evaluator_path = Path(__file__).resolve()
    output_dir = Path(args.output_dir).resolve()

    # ── Integrity preflight — stop before metric interpretation on failure ──
    try:
        observed_shas = run_integrity_preflight(
            p8_gate_report_path=p8_gate_report_path,
            p8_npz_path=p8_npz_path,
            p8_contract_path=p8_contract_path,
            contract_path=contract_path,
            expected_contract_sha=args.contract_sha,
            p8_runner_path=p8_runner_path,
            p8_source_path=p8_source_path,
            weights_path=weights_path,
            evidence_npz_path=evidence_npz_path,
            fixture_path=fixture_path,
            evaluator_path=evaluator_path,
            expected_evaluator_sha=args.expected_evaluator_sha,
            contract_pinned_evaluator_sha=contract_pinned_evaluator_sha,
            pinned=pinned,
        )
    except IntegrityError as exc:
        print(f"BLOCKED: Integrity preflight failed: {exc}", file=sys.stderr)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        invalid_report: Dict[str, Any] = {
            "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "conclusion": CONCLUSION_INVALID,
            "assessment_metadata": contract.get("assessment_metadata", {}),
            "integrity_error": str(exc),
            "saliency_norms_release_status": "BLOCKED",
        }
        _try_write_invalid_report(output_dir, ts, invalid_report)
        return 1

    # ── Load and validate arrays ────────────────────────────────────────────
    try:
        arrays = load_and_validate_arrays(
            p8_npz_path=p8_npz_path,
            evidence_npz_path=evidence_npz_path,
            reference_members=reference_members,
            pinned=pinned,
        )
    except IntegrityError as exc:
        print(f"BLOCKED: Array validation failed: {exc}", file=sys.stderr)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        invalid_report: Dict[str, Any] = {
            "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "conclusion": CONCLUSION_INVALID,
            "assessment_metadata": contract.get("assessment_metadata", {}),
            "integrity_error": str(exc),
            "saliency_norms_release_status": "BLOCKED",
        }
        _try_write_invalid_report(output_dir, ts, invalid_report)
        return 1

    prod_raw = arrays["prod_raw"]
    prod_clf = arrays["prod_clf"]
    ref_leg_raw = arrays["ref_leg_raw"]
    ref_leg_clf = arrays["ref_leg_clf"]
    ref_mlc_raw = arrays["ref_mlc_raw"]
    ref_mlc_clf = arrays["ref_mlc_clf"]

    # ── Evaluate gates ──────────────────────────────────────────────────────
    gate_1_result = _safe_evaluate(
        "gate_1",
        evaluate_gate_1,
        prod_raw, ref_leg_raw,
        contract["acceptance_gates"]["gate_1"],
        frozen_thresholds,
    )
    gate_2_result = _safe_evaluate(
        "gate_2",
        evaluate_gate_2,
        prod_clf, ref_leg_clf,
        contract["acceptance_gates"]["gate_2"],
        frozen_thresholds,
    )
    diag_1_result = _safe_evaluate(
        "diag_1",
        evaluate_diag_1,
        prod_raw, ref_mlc_raw, frozen_thresholds, ssim_params,
    )
    diag_2_result = _safe_evaluate(
        "diag_2",
        evaluate_diag_2,
        prod_clf, ref_mlc_clf,
    )

    # ── Assemble verdict ────────────────────────────────────────────────────
    conclusion, norms_status, verdict_meta = assemble_verdict(
        gate_1=gate_1_result,
        gate_2=gate_2_result,
        diag_1=diag_1_result,
        p9_contract_sha=args.contract_sha,
        p8_source_sha=observed_shas.get("p8_source_sha256", ""),
        p8_weights_sha=observed_shas.get("weights_sha256", ""),
        pinned=pinned,
    )

    # ── Build report ────────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "conclusion": conclusion,
        "assessment_metadata": {
            **contract["assessment_metadata"],
            "saliency_norms_release_status": norms_status,
        },
        "verdict": verdict_meta,
        "versions": {
            "python": sys.version,
            "numpy": str(getattr(np, "__version__", "unknown")),
            "scipy": str(getattr(scipy, "__version__", "unknown")),
            "platform": platform.platform(),
        },
        "identities": {
            **observed_shas,
            "p8_bundle_path": str(p8_bundle_path),
            "evidence_npz_path": str(evidence_npz_path),
        },
        "frozen_thresholds": frozen_thresholds,
        "acceptance_gate_results": [
            {
                "name": g.name,
                "passed": g.passed,
                "reason": g.reason,
                "observed": g.observed,
            }
            for g in [gate_1_result, gate_2_result]
        ],
        "diagnostic_gate_results": [
            {
                "name": g.name,
                "passed": g.passed,
                "reason": g.reason,
                "observed": g.observed,
            }
            for g in [diag_1_result, diag_2_result]
        ],
    }

    # ── Write evidence bundle ───────────────────────────────────────────────
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"umsi_step2b_p9_corrigendum_{ts}"
    try:
        bundle_path = write_evidence_bundle(
            output_dir=output_dir,
            bundle_name=bundle_name,
            report=report,
        )
    except Exception as exc:
        print(f"EVIDENCE WRITE ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "conclusion": conclusion,
                "saliency_norms_release_status": norms_status,
                "bundle": str(bundle_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    if conclusion == CONCLUSION_INVALID:
        return 1
    if conclusion == CONCLUSION_INCONCLUSIVE:
        return 2
    return 0 if conclusion == CONCLUSION_PASS else 2


def _try_write_invalid_report(
    output_dir: Path, ts: str, report: Dict[str, Any]
) -> None:
    try:
        bundle_name = f"umsi_step2b_p9_corrigendum_{ts}_INVALID"
        bundle_path = output_dir / bundle_name
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_path.mkdir(parents=False, exist_ok=True)
        with open(bundle_path / "gate_report_p9.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "STAGE 1 STEP 2B P9 CORRIGENDUM — Zero-inference re-evaluator.\n"
            "Reads immutable P8 production outputs; applies corrected P9 contract.\n"
            "No TensorFlow, Keras or UMSI module is imported."
        )
    )
    p.add_argument(
        "--expected-evaluator-sha",
        required=True,
        help="Expected SHA-256 (64 hex) of this evaluator script.",
    )
    p.add_argument(
        "--contract",
        required=True,
        help="Path to umsi_raw_output_gate_contract_p9_corrigendum.json.",
    )
    p.add_argument(
        "--contract-sha",
        required=True,
        help="Expected SHA-256 (64 hex) of the P9 contract.",
    )
    p.add_argument(
        "--p8-bundle",
        required=True,
        help="Path to the P8 evidence bundle directory "
             "(umsi_step2b_20260803T130415Z).",
    )
    p.add_argument(
        "--p8-contract",
        required=True,
        help="Path to the original P8 contract JSON "
             "(stage1/evidence/umsi_raw_output_gate_contract.json).",
    )
    p.add_argument(
        "--p8-runner",
        required=True,
        help="Path to the original P8 frozen runner "
             "(stage1/tools/umsi_step2b_gate_runner.py).",
    )
    p.add_argument(
        "--p8-source",
        required=True,
        help="Path to saliency/umsi_model.py used for P8.",
    )
    p.add_argument(
        "--evidence-npz",
        required=True,
        help="Path to umsi_resize_causality_arrays_6a17288_consolidated.npz.",
    )
    p.add_argument(
        "--weights",
        required=True,
        help="Path to umsi++.hdf5.",
    )
    p.add_argument(
        "--fixture",
        required=True,
        help="Path to lowcontrast.png fixture used in P8.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to write the P9 evidence bundle.",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_p9_corrigendum(args)


if __name__ == "__main__":
    raise SystemExit(main())
