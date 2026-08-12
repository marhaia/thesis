#!/usr/bin/env python3
"""STAGE 1 STEP 2B — Contract-driven one-shot four-gate experiment harness.

Evaluates the four frozen gates from the prospective contract at
stage1/evidence/umsi_raw_output_gate_contract.json.

IMPORTANT: Importing this module does NOT import TensorFlow, Keras or
saliency.umsi_model. All production model imports are deferred exclusively to
the real-execution path inside run_frozen_experiment(), which requires the CLI
flag --execute-frozen-experiment.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── NO top-level TensorFlow, Keras or saliency imports. ───────────────────────
# scipy and subprocess are stdlib/safe but also deferred where possible.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_REQUIRED_GATES = frozenset({"gate_a", "gate_b", "gate_c", "gate_d"})
_REQUIRED_FROZEN_THRESHOLD_KEYS = frozenset(
    {"pearson_min", "spearman_min", "ssim_min", "cli_abs_max", "repeatability_abs_max"}
)
_REQUIRED_MEMBERS = frozenset(
    {
        "legacy_raw_saliency",
        "legacy_classification",
        "controlled_modern_raw_saliency",
        "controlled_modern_classification",
    }
)
_REQUIRED_POSTPROCESS_ORDER = ["squeeze", "identical_resize", "identical_min_max_to_0_1"]
_CONSTANT_MAP_FAIL_REASON = "UNDEFINED_CORRELATION_CONSTANT_MAP"
_SSIM_REQUIRED_PARAMS = frozenset({"K1", "K2", "win_size", "boundary", "data_range", "filter"})


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 hex digest of array bytes in C-contiguous order."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


# ---------------------------------------------------------------------------
# Contract loading and schema validation
# ---------------------------------------------------------------------------

class ContractError(Exception):
    """Raised when the contract is missing or inconsistent with required schema."""


def load_and_validate_contract(contract_path: Path) -> Dict[str, Any]:
    """Load the frozen contract JSON and validate its schema.

    Raises ContractError if any required field is missing or inconsistent.
    Does not load or inspect any numerical evidence.
    """
    with open(contract_path, "r", encoding="utf-8") as fh:
        c = json.load(fh)

    # Required top-level sections
    for section in (
        "pinned_identities",
        "reference_members",
        "gates",
        "prospective_new_rule_before_output_observation",
        "historical_reused_authority",
    ):
        if section not in c:
            raise ContractError(f"Missing required contract section: {section!r}")

    # All four gates
    missing = _REQUIRED_GATES - set(c["gates"].keys())
    if missing:
        raise ContractError(f"Missing contract gates: {sorted(missing)}")

    # Reference member keys
    missing = _REQUIRED_MEMBERS - set(c["reference_members"].keys())
    if missing:
        raise ContractError(f"Missing reference_members: {sorted(missing)}")

    # Gate B: authoritative shared postprocess
    if "authoritative_shared_postprocess" not in c["gates"]["gate_b"]:
        raise ContractError("gate_b missing authoritative_shared_postprocess")
    order = c["gates"]["gate_b"]["authoritative_shared_postprocess"].get("order")
    if order != _REQUIRED_POSTPROCESS_ORDER:
        raise ContractError(f"gate_b postprocess order unexpected: {order!r}")

    # Gate B requirements
    b_req = c["gates"]["gate_b"].get("requirements", {})
    for metric in ("pearson", "spearman", "windowed_ssim"):
        if metric not in b_req:
            raise ContractError(f"gate_b requirements missing: {metric!r}")

    # SSIM params
    ssim_params = b_req["windowed_ssim"].get("params", {})
    missing = _SSIM_REQUIRED_PARAMS - set(ssim_params.keys())
    if missing:
        raise ContractError(f"windowed_ssim params missing: {sorted(missing)}")

    # Constant-map policy
    policy = (
        c["prospective_new_rule_before_output_observation"]
        .get("policy", {})
    )
    steps = policy.get("steps", [])
    if len(steps) < 6:
        raise ContractError("Constant-map policy steps are incomplete (need ≥ 6 entries)")

    # Gate D cli_abs_max binding
    d_req = c["gates"]["gate_d"].get("requirements", {})
    d_max = d_req.get("max_abs_diff_float64", {})
    if d_max.get("bound_to_threshold") != "cli_abs_max":
        raise ContractError("gate_d max_abs_diff_float64 not bound to cli_abs_max")

    # Validate frozen_thresholds nested structure before any inference.
    # This guarantees post-inference lookups into this section cannot fail.
    ha = c["historical_reused_authority"]
    if not isinstance(ha, dict):
        raise ContractError(
            f"historical_reused_authority must be a dict; got {type(ha).__name__!r}"
        )
    if "frozen_thresholds" not in ha:
        raise ContractError(
            "historical_reused_authority missing required key: 'frozen_thresholds'"
        )
    thr = ha["frozen_thresholds"]
    if not isinstance(thr, dict):
        raise ContractError(
            f"historical_reused_authority.frozen_thresholds must be a dict; "
            f"got {type(thr).__name__!r}"
        )
    missing_thr = _REQUIRED_FROZEN_THRESHOLD_KEYS - set(thr.keys())
    if missing_thr:
        raise ContractError(
            f"frozen_thresholds missing required keys: {sorted(missing_thr)}"
        )
    # Validate each threshold value: must be int or float (not bool), finite,
    # and within the semantically expected range.
    _CORR_SSIM_KEYS = frozenset({"pearson_min", "spearman_min", "ssim_min"})
    _ABS_MAX_KEYS = frozenset({"cli_abs_max", "repeatability_abs_max"})
    import math as _math
    for _k in _REQUIRED_FROZEN_THRESHOLD_KEYS:
        _v = thr[_k]
        if isinstance(_v, bool) or not isinstance(_v, (int, float)):
            raise ContractError(
                f"frozen_thresholds[{_k!r}] must be int or float (not bool); "
                f"got {type(_v).__name__!r}"
            )
        if not _math.isfinite(float(_v)):
            raise ContractError(
                f"frozen_thresholds[{_k!r}] must be finite; got {_v!r}"
            )
        if _k in _CORR_SSIM_KEYS and not (-1.0 <= float(_v) <= 1.0):
            raise ContractError(
                f"frozen_thresholds[{_k!r}] = {_v!r} is outside [-1.0, 1.0]"
            )
        if _k in _ABS_MAX_KEYS and float(_v) < 0.0:
            raise ContractError(
                f"frozen_thresholds[{_k!r}] = {_v!r} must be non-negative"
            )

    return c


# ---------------------------------------------------------------------------
# Identity pre-flight
# ---------------------------------------------------------------------------

class PreflightError(Exception):
    """Raised when a required identity check fails."""


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256_hex(value: str, label: str) -> None:
    """Raise PreflightError if value is not exactly 64 lowercase hex chars."""
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise PreflightError(
            f"{label} must be exactly 64 lowercase hexadecimal characters; "
            f"got {value!r}"
        )


def run_preflight(
    *,
    repo_path: Path,
    expected_head: str,
    contract_path: Path,
    expected_contract_sha: str,
    source_path: Path,
    expected_source_sha: str,
    expected_runner_sha: str,
    weights_path: Path,
    expected_weights_sha: str,
    npz_path: Path,
    expected_npz_sha: str,
    fixture_path: Path,
    expected_fixture_sha: str,
) -> Dict[str, str]:
    """Verify all required identities. Raises PreflightError on any mismatch.

    Returns a dict of {label: observed_sha256} for inclusion in the evidence
    report.

    The runner hash check is performed FIRST, before any git or file SHA
    validation, so that a modified runner cannot pass all other checks.
    """
    import subprocess  # deferred; stdlib only

    # ── Runner hash — first check, before any other validation ──────────────
    _validate_sha256_hex(expected_runner_sha, "--expected-runner-sha")
    observed_runner_sha = sha256_file(Path(__file__).resolve())
    if observed_runner_sha != expected_runner_sha:
        raise PreflightError(
            f"Runner SHA-256 mismatch: "
            f"expected {expected_runner_sha!r}, got {observed_runner_sha!r}. "
            f"The executing runner does not match the audited hash."
        )

    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise PreflightError(
            f"git rev-parse HEAD failed (rc={result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    observed_head = result.stdout.decode("utf-8", "replace").strip()
    if observed_head != expected_head:
        raise PreflightError(
            f"HEAD mismatch: expected {expected_head!r}, got {observed_head!r}"
        )

    file_checks: Dict[str, Tuple[Path, str]] = {
        "contract": (contract_path, expected_contract_sha),
        "source": (source_path, expected_source_sha),
        "weights": (weights_path, expected_weights_sha),
        "npz": (npz_path, expected_npz_sha),
        "fixture": (fixture_path, expected_fixture_sha),
    }
    observed: Dict[str, str] = {}
    for label, (path, expected) in file_checks.items():
        if not path.is_file():
            raise PreflightError(f"Required file missing: {path}")
        obs = sha256_file(path)
        if obs != expected:
            raise PreflightError(
                f"{label} SHA-256 mismatch: expected {expected!r}, got {obs!r}"
            )
        observed[label] = obs

    observed["runner"] = observed_runner_sha
    return observed


# ---------------------------------------------------------------------------
# Output-capturing, call-counting inference proxy
# ---------------------------------------------------------------------------

@dataclass
class CapturedOutputs:
    """Native raw outputs from the single real inference call."""

    raw_saliency: np.ndarray   # preds[0][0] — before any postprocessing
    classif: np.ndarray        # preds[1][0] — native six-value vector
    call_count: int
    restored: bool


class InferenceCallError(Exception):
    """Raised when the proxy detects a violation of the one-call contract."""


def capture_and_invoke(
    model_instance: Any,
    fixture_path: str,
) -> Tuple[CapturedOutputs, Any]:
    """Wrap model.model.predict and invoke predict_saliency exactly once.

    The proxy:
    - Increments the call count BEFORE delegation.
    - Rejects a second attempted call WITHOUT executing it.
    - Delegates to the original real predict exactly once.
    - Validates the returned output structure immediately.
    - Copies both native raw outputs (preds[0][0], preds[1][0]) before any
      postprocessing performed by predict_saliency.
    - Restores the original predict in a finally block regardless of exceptions.
    - Does not retry on exception.
    - Requires the final call count to be exactly one.
    """
    predict_owner = model_instance.model
    original_predict = predict_owner.predict

    call_count: int = 0
    captured_raw_saliency: Optional[np.ndarray] = None
    captured_classif: Optional[np.ndarray] = None

    def _proxy(x: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count, captured_raw_saliency, captured_classif

        call_count += 1
        if call_count > 1:
            raise InferenceCallError(
                f"_proxy: attempted call #{call_count}; only one call is permitted. "
                "Aborting before executing."
            )

        result = original_predict(x, *args, **kwargs)

        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise InferenceCallError(
                f"_proxy: unexpected output structure — type={type(result).__name__}, "
                f"len={len(result) if hasattr(result, '__len__') else '?'}"
            )

        # Copy native outputs immediately, before predict_saliency postprocesses them.
        captured_raw_saliency = np.array(result[0][0], copy=True)
        captured_classif = np.array(result[1][0], copy=True)

        return result

    predict_owner.predict = _proxy
    try:
        predict_result = model_instance.predict_saliency(
            fixture_path, return_classif=True
        )
    finally:
        predict_owner.predict = original_predict

    restored = predict_owner.predict is original_predict

    if call_count != 1:
        raise InferenceCallError(
            f"capture_and_invoke: expected exactly 1 predict call, got {call_count}"
        )
    if captured_raw_saliency is None or captured_classif is None:
        raise InferenceCallError(
            "capture_and_invoke: raw outputs were not captured (proxy never ran)"
        )

    return (
        CapturedOutputs(
            raw_saliency=captured_raw_saliency,
            classif=captured_classif,
            call_count=call_count,
            restored=restored,
        ),
        predict_result,
    )


# ---------------------------------------------------------------------------
# Gate B shared postprocessing helpers
# ---------------------------------------------------------------------------

def _squeeze_to_2d(arr: np.ndarray, label: str) -> np.ndarray:
    """Remove trailing single-element dimensions; require result to be 2D."""
    a = np.squeeze(arr)
    if a.ndim != 2:
        raise ValueError(
            f"_squeeze_to_2d({label!r}): expected 2D after squeeze, "
            f"got {arr.shape!r} → {a.shape!r}"
        )
    return a


def _apply_constant_map_policy(
    arr: np.ndarray,
    label: str,
) -> Tuple[np.ndarray, bool]:
    """Apply the contract's prospective constant-map policy (Gate B, step 3).

    Contract rule (prospective_new_rule_before_output_observation):
      - Convert to float64; require all values finite.
      - span = max - min (no epsilon, no observed-data tolerance).
      - If span > 0.0: normalize as (arr - min) / span → [0, 1].
      - If span == 0.0: return all-zero float64 array; set constant_map_detected=True.

    Returns (normalized_float64_array, constant_map_detected).
    """
    a = arr.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(
            f"_apply_constant_map_policy({label!r}): non-finite values present"
        )
    vmin = float(a.min())
    vmax = float(a.max())
    span = vmax - vmin
    if span == 0.0:
        return np.zeros_like(a, dtype=np.float64), True
    return (a - vmin) / span, False


def _windowed_ssim(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
) -> float:
    """Windowed SSIM using scipy.ndimage.uniform_filter.

    Implements the contract's frozen SSIM definition:
      filter=scipy.ndimage.uniform_filter, K1, K2, win_size, boundary, data_range.
    """
    from scipy.ndimage import uniform_filter  # deferred; no TF/Keras

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

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = uniform_filter(xf * xf, size=win_size, mode=mode) - mu_x_sq
    sigma_y_sq = uniform_filter(yf * yf, size=win_size, mode=mode) - mu_y_sq
    sigma_xy = uniform_filter(xf * yf, size=win_size, mode=mode) - mu_xy

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Gate evaluators
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    passed: bool
    reason: Optional[str] = None
    observed: Dict[str, Any] = field(default_factory=dict)


def evaluate_gate_a(
    prod_raw: np.ndarray,
    ref_m_lc_raw: np.ndarray,
    gate_def: Dict[str, Any],
) -> GateResult:
    """Gate A: production raw saliency vs controlled-modern M_LC raw saliency.

    Requirements (from contract): same_shape, same_dtype, finite_values,
    array_equal, max_abs_diff_float64 == 0.0 (repeatability_abs_max).
    """
    name = "gate_a"
    req = gate_def["requirements"]

    try:
        prod = _squeeze_to_2d(np.asarray(prod_raw), "prod_raw_gate_a")
        ref = _squeeze_to_2d(np.asarray(ref_m_lc_raw), "ref_m_lc_raw_gate_a")
    except Exception as exc:
        return GateResult(name=name, passed=False, reason=f"GATE_A_PREP_ERROR: {exc}")

    obs: Dict[str, Any] = {
        "prod_shape": list(prod.shape),
        "ref_shape": list(ref.shape),
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
    }

    if req.get("same_shape") and tuple(prod.shape) != tuple(ref.shape):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_A_SHAPE_MISMATCH: {prod.shape} != {ref.shape}",
            observed=obs,
        )
    if req.get("same_dtype") and str(prod.dtype) != str(ref.dtype):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_A_DTYPE_MISMATCH: {prod.dtype} != {ref.dtype}",
            observed=obs,
        )
    if req.get("finite_values"):
        if not np.isfinite(prod).all():
            return GateResult(name=name, passed=False, reason="GATE_A_NON_FINITE_PROD", observed=obs)
        if not np.isfinite(ref).all():
            return GateResult(name=name, passed=False, reason="GATE_A_NON_FINITE_REF", observed=obs)

    array_equal = bool(np.array_equal(prod, ref))
    max_abs_diff = float(
        np.max(np.abs(prod.astype(np.float64) - ref.astype(np.float64)))
    )
    obs["array_equal"] = array_equal
    obs["max_abs_diff_float64"] = max_abs_diff
    obs["prod_sha256"] = sha256_array(prod)
    obs["ref_sha256"] = sha256_array(ref)

    if req.get("array_equal") and not array_equal:
        return GateResult(name=name, passed=False, reason="GATE_A_NOT_ARRAY_EQUAL", observed=obs)

    max_req = req.get("max_abs_diff_float64", {})
    if max_req.get("operator") == "==" and max_abs_diff != float(max_req["value"]):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_A_MAX_ABS_DIFF_FAIL: {max_abs_diff} != {max_req['value']}",
            observed=obs,
        )

    return GateResult(name=name, passed=True, observed=obs)


def evaluate_gate_b(
    prod_raw: np.ndarray,
    ref_legacy_raw: np.ndarray,
    gate_def: Dict[str, Any],
) -> GateResult:
    """Gate B: production raw saliency vs legacy raw saliency with shared postprocess.

    Contract postprocess: squeeze → identical_resize → identical_min_max_to_0_1.
    Then: Pearson ≥ 0.999, Spearman ≥ 0.99, windowed-SSIM ≥ 0.99.
    Constant-map policy applied before min-max normalization.
    """
    name = "gate_b"
    req = gate_def["requirements"]
    ssim_params = req["windowed_ssim"]["params"]
    obs: Dict[str, Any] = {}

    try:
        prod_2d = _squeeze_to_2d(np.asarray(prod_raw), "prod_raw_gate_b")
        ref_2d = _squeeze_to_2d(np.asarray(ref_legacy_raw), "ref_legacy_gate_b")
    except Exception as exc:
        return GateResult(name=name, passed=False, reason=f"GATE_B_SQUEEZE_ERROR: {exc}")

    obs["prod_shape_before_postprocess"] = list(prod_2d.shape)
    obs["ref_shape_before_postprocess"] = list(ref_2d.shape)

    # identical_resize: both must already be at the same shape; a bilinear resize
    # from (H,W) to (H,W) is an identity operation. Unequal shapes are fail-closed
    # because the contract's "identical" resize is only well-defined at a shared
    # native resolution.
    if prod_2d.shape != ref_2d.shape:
        return GateResult(
            name=name, passed=False,
            reason=(
                f"GATE_B_SHAPE_MISMATCH_BEFORE_POSTPROCESS: "
                f"{prod_2d.shape} != {ref_2d.shape}"
            ),
            observed=obs,
        )

    # Raw diagnostics (diagnostics_only per contract)
    obs["raw_max_abs_diff_float64"] = float(
        np.max(np.abs(prod_2d.astype(np.float64) - ref_2d.astype(np.float64)))
    )
    obs["prod_sha256_raw"] = sha256_array(prod_2d)
    obs["ref_sha256_raw"] = sha256_array(ref_2d)

    # Step 3: identical_min_max_to_0_1 with constant-map policy
    try:
        prod_norm, prod_const = _apply_constant_map_policy(prod_2d, "prod_gate_b")
        ref_norm, ref_const = _apply_constant_map_policy(ref_2d, "ref_gate_b")
    except ValueError as exc:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_POSTPROCESS_ERROR: {exc}",
            observed=obs,
        )

    obs["prod_constant_map"] = prod_const
    obs["ref_constant_map"] = ref_const

    if prod_const or ref_const:
        obs["pearson"] = None
        obs["spearman"] = None
        obs["windowed_ssim"] = None
        return GateResult(name=name, passed=False, reason=_CONSTANT_MAP_FAIL_REASON, observed=obs)

    # Pearson and Spearman
    try:
        from scipy.stats import pearsonr, spearmanr  # deferred
        prod_flat = prod_norm.ravel()
        ref_flat = ref_norm.ravel()
        pearson_r, _ = pearsonr(prod_flat, ref_flat)
        spearman_r, _ = spearmanr(prod_flat, ref_flat)
    except Exception as exc:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_CORRELATION_ERROR: {exc}",
            observed=obs,
        )

    # Windowed SSIM
    try:
        ssim_val = _windowed_ssim(prod_norm, ref_norm, ssim_params)
    except Exception as exc:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_SSIM_ERROR: {exc}",
            observed=obs,
        )

    obs["pearson"] = float(pearson_r)
    obs["spearman"] = float(spearman_r)
    obs["windowed_ssim"] = float(ssim_val)

    pearson_min = float(req["pearson"]["value"])
    spearman_min = float(req["spearman"]["value"])
    ssim_min = float(req["windowed_ssim"]["value"])

    if not np.isfinite(pearson_r):
        return GateResult(name=name, passed=False, reason="GATE_B_PEARSON_NON_FINITE", observed=obs)
    if not np.isfinite(spearman_r):
        return GateResult(name=name, passed=False, reason="GATE_B_SPEARMAN_NON_FINITE", observed=obs)
    if not np.isfinite(ssim_val):
        return GateResult(name=name, passed=False, reason="GATE_B_SSIM_NON_FINITE", observed=obs)

    if pearson_r < pearson_min:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_PEARSON_FAIL: {pearson_r:.8f} < {pearson_min}",
            observed=obs,
        )
    if spearman_r < spearman_min:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_SPEARMAN_FAIL: {spearman_r:.8f} < {spearman_min}",
            observed=obs,
        )
    if ssim_val < ssim_min:
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_B_SSIM_FAIL: {ssim_val:.8f} < {ssim_min}",
            observed=obs,
        )

    return GateResult(name=name, passed=True, observed=obs)


def evaluate_gate_c(
    prod_classif: np.ndarray,
    ref_m_lc_classif: np.ndarray,
    gate_def: Dict[str, Any],
) -> GateResult:
    """Gate C: production classification vs controlled-modern M_LC classification.

    Requirements: same_shape, same_dtype, finite_values, array_equal,
    max_abs_diff_float64 == 0.0 (repeatability_abs_max).
    """
    name = "gate_c"
    req = gate_def["requirements"]
    prod = np.asarray(prod_classif)
    ref = np.asarray(ref_m_lc_classif)

    obs: Dict[str, Any] = {
        "prod_shape": list(prod.shape),
        "ref_shape": list(ref.shape),
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
    }

    if req.get("same_shape") and tuple(prod.shape) != tuple(ref.shape):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_C_SHAPE_MISMATCH: {prod.shape} != {ref.shape}",
            observed=obs,
        )
    if req.get("same_dtype") and str(prod.dtype) != str(ref.dtype):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_C_DTYPE_MISMATCH: {prod.dtype} != {ref.dtype}",
            observed=obs,
        )
    if req.get("finite_values"):
        if not np.isfinite(prod).all():
            return GateResult(name=name, passed=False, reason="GATE_C_NON_FINITE_PROD", observed=obs)
        if not np.isfinite(ref).all():
            return GateResult(name=name, passed=False, reason="GATE_C_NON_FINITE_REF", observed=obs)

    array_equal = bool(np.array_equal(prod, ref))
    max_abs_diff = float(
        np.max(np.abs(prod.astype(np.float64) - ref.astype(np.float64)))
    )
    obs["array_equal"] = array_equal
    obs["max_abs_diff_float64"] = max_abs_diff

    if req.get("array_equal") and not array_equal:
        return GateResult(name=name, passed=False, reason="GATE_C_NOT_ARRAY_EQUAL", observed=obs)

    max_req = req.get("max_abs_diff_float64", {})
    if max_req.get("operator") == "==" and max_abs_diff != float(max_req["value"]):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_C_MAX_ABS_DIFF_FAIL: {max_abs_diff} != {max_req['value']}",
            observed=obs,
        )

    return GateResult(name=name, passed=True, observed=obs)


def evaluate_gate_d(
    prod_classif: np.ndarray,
    ref_legacy_classif: np.ndarray,
    gate_def: Dict[str, Any],
) -> GateResult:
    """Gate D: production classification vs legacy classification.

    cli_abs_max=0.01 is the Step 2B classification-output tolerance.
    This threshold is NOT connected to any Stage-2 score or threshold.
    """
    name = "gate_d"
    req = gate_def["requirements"]
    prod = np.asarray(prod_classif)
    ref = np.asarray(ref_legacy_classif)

    obs: Dict[str, Any] = {
        "prod_shape": list(prod.shape),
        "ref_shape": list(ref.shape),
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
        "threshold_label": (
            "cli_abs_max — Step 2B classification-output tolerance; "
            "NOT a Stage-2 score or downstream threshold"
        ),
    }

    if req.get("same_shape") and tuple(prod.shape) != tuple(ref.shape):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_D_SHAPE_MISMATCH: {prod.shape} != {ref.shape}",
            observed=obs,
        )
    if req.get("finite_values"):
        if not np.isfinite(prod).all():
            return GateResult(name=name, passed=False, reason="GATE_D_NON_FINITE_PROD", observed=obs)
        if not np.isfinite(ref).all():
            return GateResult(name=name, passed=False, reason="GATE_D_NON_FINITE_REF", observed=obs)

    max_abs_diff = float(
        np.max(np.abs(prod.astype(np.float64) - ref.astype(np.float64)))
    )
    max_req = req["max_abs_diff_float64"]
    obs["max_abs_diff_float64"] = max_abs_diff
    obs["cli_abs_max"] = float(max_req["value"])

    if max_req.get("operator") == "<=" and max_abs_diff > float(max_req["value"]):
        return GateResult(
            name=name, passed=False,
            reason=(
                f"GATE_D_CLI_ABS_MAX_FAIL: {max_abs_diff:.8f} > {max_req['value']} "
                f"(cli_abs_max — Step 2B classification-output tolerance)"
            ),
            observed=obs,
        )

    return GateResult(name=name, passed=True, observed=obs)



# ---------------------------------------------------------------------------
# Safe evaluator boundary (D2)
# ---------------------------------------------------------------------------

def safe_evaluate_gate(
    gate_name: str,
    evaluator_fn: Any,
    argument_factory: Any,
) -> GateResult:
    """Invoke a gate evaluator inside a full exception boundary.

    argument_factory() is called INSIDE the try block so that any KeyError,
    TypeError or other Exception raised during argument preparation is also
    captured. This prevents fallible dict lookups such as contract["gates"][key]
    from escaping the safe boundary when they appear in the argument list.

    Catches Exception only — not KeyboardInterrupt, SystemExit or other
    BaseException subclasses.

    On any exception (including during argument preparation):
      - name = gate_name
      - passed = False
      - reason = EVALUATOR_EXCEPTION
      - observed contains evaluator_exception_type and evaluator_exception_message

    Also validates that a successful call returns a GateResult for the expected
    gate name. An invalid return type or wrong gate name becomes a failed result.
    """
    try:
        args = argument_factory()
        result = evaluator_fn(*args)
    except Exception as exc:
        return GateResult(
            name=gate_name,
            passed=False,
            reason="EVALUATOR_EXCEPTION",
            observed={
                "evaluator_exception_type": type(exc).__name__,
                "evaluator_exception_message": str(exc),
            },
        )

    if not isinstance(result, GateResult):
        return GateResult(
            name=gate_name,
            passed=False,
            reason="EVALUATOR_INVALID_RETURN",
            observed={
                "evaluator_return_type": type(result).__name__,
            },
        )
    if result.name != gate_name:
        return GateResult(
            name=gate_name,
            passed=False,
            reason="EVALUATOR_WRONG_GATE_NAME",
            observed={
                "expected_gate_name": gate_name,
                "returned_gate_name": result.name,
            },
        )
    return result


# ---------------------------------------------------------------------------
# Evidence bundle writer
# ---------------------------------------------------------------------------

def write_evidence_bundle(
    *,
    output_dir: Path,
    bundle_name: str,
    report: Dict[str, Any],
    raw_saliency: np.ndarray,
    classif: np.ndarray,
) -> Path:
    """Write production_outputs.npz and gate_report.json atomically.

    Uses a temporary sibling directory + a single os.rename() for atomicity.
    Aborts if the target bundle path already exists (no overwrite).

    Returns the final bundle directory path.
    """
    bundle_path = output_dir / bundle_name
    if bundle_path.exists():
        raise FileExistsError(
            f"Evidence bundle already exists: {bundle_path}. "
            "Refusing to overwrite. Use a unique bundle name."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = bundle_path.parent / (bundle_name + f".tmp_{os.getpid()}")
    if tmp_path.exists():
        raise FileExistsError(f"Temporary directory already exists: {tmp_path}")

    tmp_path.mkdir(parents=False, exist_ok=False)
    try:
        np.savez(
            str(tmp_path / "production_outputs.npz"),
            production_raw_saliency=raw_saliency,
            production_classif=classif,
        )
        with open(tmp_path / "gate_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
        os.rename(str(tmp_path), str(bundle_path))
    except Exception:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(str(tmp_path))
        raise

    return bundle_path


# ---------------------------------------------------------------------------
# Library version helper
# ---------------------------------------------------------------------------

def _lib_version(name: str) -> str:
    try:
        import importlib
        m = importlib.import_module(name)
        return str(getattr(m, "__version__", "unknown"))
    except ImportError:
        return "not_installed"


# ---------------------------------------------------------------------------
# Static report-payload builder (production-independent)
# ---------------------------------------------------------------------------

def _build_report_statics(
    *,
    frozen_thresholds: Dict[str, Any],
    pinned: Dict[str, Any],
    observed_shas: Dict[str, str],
    module_file_str: str,
    expected_runner_sha: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the complete static report payload before model construction.

    Extracts all report-only values from frozen_thresholds, pinned and
    observed_shas into two plain dicts:
      - report_thresholds: the contract_thresholds section of the report
      - report_identities_static: the identities section of the report

    After this function returns, frozen_thresholds, pinned and observed_shas
    are no longer needed by the post-capture helper.  The gate-definition
    mapping (contract["gates"][...]) remains available for gate argument
    factories and is protected by safe_evaluate_gate's try/except boundary.

    Does not import or instantiate the production model.
    """
    report_thresholds: Dict[str, Any] = {
        "pearson_min": frozen_thresholds["pearson_min"],
        "spearman_min": frozen_thresholds["spearman_min"],
        "ssim_min": frozen_thresholds["ssim_min"],
        "cli_abs_max_step2b_classif_only__not_stage2": frozen_thresholds["cli_abs_max"],
        "repeatability_abs_max": frozen_thresholds["repeatability_abs_max"],
    }
    report_identities_static: Dict[str, Any] = {
        "head": pinned["expected_repository_head"],
        "expected_runner_sha256": expected_runner_sha,
        "observed_runner_sha256": observed_shas["runner"],
        "runner_sha256_match": observed_shas["runner"] == expected_runner_sha,
        "source_sha256": observed_shas["source"],
        "contract_sha256": observed_shas["contract"],
        "weights_sha256": observed_shas["weights"],
        "npz_sha256": observed_shas["npz"],
        "fixture_sha256": observed_shas["fixture"],
        "production_import_origin": module_file_str,
    }
    return report_thresholds, report_identities_static


# ---------------------------------------------------------------------------
# Factored post-capture sequence (production-independent)
# ---------------------------------------------------------------------------

def _run_post_capture_sequence(
    *,
    captured: "CapturedOutputs",
    ref_legacy_raw: "np.ndarray",
    ref_legacy_classif: "np.ndarray",
    ref_m_lc_raw: "np.ndarray",
    ref_m_lc_classif: "np.ndarray",
    contract: Dict[str, Any],
    report_thresholds: Dict[str, Any],
    report_identities_static: Dict[str, Any],
    output_dir: Path,
) -> int:
    """Post-capture orchestration: safe gate evaluation → report → evidence write.

    This is the single post-inference path used by run_frozen_experiment().
    It accepts already-captured native outputs and the pre-built static report
    payload (built before model construction by _build_report_statics) so
    that it can also be exercised with synthetic data in production-independent
    tests.

    Gate argument factories are lambdas evaluated INSIDE safe_evaluate_gate's
    try block, so any KeyError or other preparation exception is captured and
    becomes a failed GateResult for that gate only — the other gates continue.

    Does not import or instantiate the production model.
    Does not index frozen_thresholds, pinned or observed_shas.
    """
    gate_results: List[GateResult] = [
        safe_evaluate_gate(
            "gate_a",
            evaluate_gate_a,
            lambda: (captured.raw_saliency, ref_m_lc_raw,
                     contract["gates"]["gate_a"]),
        ),
        safe_evaluate_gate(
            "gate_b",
            evaluate_gate_b,
            lambda: (captured.raw_saliency, ref_legacy_raw,
                     contract["gates"]["gate_b"]),
        ),
        safe_evaluate_gate(
            "gate_c",
            evaluate_gate_c,
            lambda: (captured.classif, ref_m_lc_classif,
                     contract["gates"]["gate_c"]),
        ),
        safe_evaluate_gate(
            "gate_d",
            evaluate_gate_d,
            lambda: (captured.classif, ref_legacy_classif,
                     contract["gates"]["gate_d"]),
        ),
    ]

    all_passed = all(g.passed for g in gate_results)
    failed_gates = [g.name.upper() for g in gate_results if not g.passed]
    conclusion = (
        "STEP_2B_PASS_ALL_GATES"
        if all_passed
        else "STEP_2B_FAIL_" + "_".join(failed_gates)
    )

    report: Dict[str, Any] = {
        "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "conclusion": conclusion,
        "versions": {
            "python": sys.version,
            "numpy": _lib_version("numpy"),
            "scipy": _lib_version("scipy"),
            "platform": platform.platform(),
        },
        "identities": report_identities_static,
        "inference": {
            "real_inference_call_count": captured.call_count,
            "proxy_restored": captured.restored,
            "raw_saliency_shape": list(captured.raw_saliency.shape),
            "raw_saliency_dtype": str(captured.raw_saliency.dtype),
            "raw_saliency_sha256": sha256_array(captured.raw_saliency),
            "classif_shape": list(captured.classif.shape),
            "classif_dtype": str(captured.classif.dtype),
            "classif_sha256": sha256_array(captured.classif),
        },
        "contract_thresholds": report_thresholds,
        "gate_results": [
            {
                "name": g.name,
                "passed": g.passed,
                "reason": g.reason,
                "observed": g.observed,
            }
            for g in gate_results
        ],
    }

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"umsi_step2b_{ts}"
    try:
        bundle_path = write_evidence_bundle(
            output_dir=output_dir,
            bundle_name=bundle_name,
            report=report,
            raw_saliency=captured.raw_saliency,
            classif=captured.classif,
        )
    except Exception as exc:
        print(f"EVIDENCE WRITE ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {"conclusion": conclusion, "bundle": str(bundle_path)},
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if all_passed else 2


# ---------------------------------------------------------------------------
# Main execution path (production imports happen ONLY here)
# ---------------------------------------------------------------------------

def run_frozen_experiment(args: argparse.Namespace) -> int:
    """Execute the frozen four-gate experiment with exactly one real inference.

    ALL imports of TensorFlow, Keras and saliency.umsi_model are deferred to
    this function. This function must not be called except via the CLI with the
    --execute-frozen-experiment flag, or from the real Step 2B invocation.
    """
    import importlib
    import inspect

    # ── Load and validate contract ──────────────────────────────────────────
    contract_path = Path(args.contract).resolve()
    try:
        contract = load_and_validate_contract(contract_path)
    except (ContractError, Exception) as exc:
        print(f"BLOCKED: Contract validation failed: {exc}", file=sys.stderr)
        return 1

    # Extract validated threshold data before production import or inference.
    # load_and_validate_contract guarantees the nested structure and required keys.
    frozen_thresholds = contract["historical_reused_authority"]["frozen_thresholds"]

    pinned = contract["pinned_identities"]
    repo_path = Path(args.repo).resolve()
    source_path = repo_path / pinned["production_source_path"]

    # ── Pre-flight identity checks ──────────────────────────────────────────
    try:
        observed_shas = run_preflight(
            repo_path=repo_path,
            expected_head=pinned["expected_repository_head"],
            contract_path=contract_path,
            expected_contract_sha=args.contract_sha,
            source_path=source_path,
            expected_source_sha=pinned["production_source_sha256_before_inference"],
            expected_runner_sha=args.expected_runner_sha,
            weights_path=Path(args.weights),
            expected_weights_sha=pinned["weights_sha256"],
            npz_path=Path(args.evidence_npz),
            expected_npz_sha=pinned["evidence_npz_sha256"],
            fixture_path=Path(args.fixture),
            expected_fixture_sha=pinned["fixture_sha256_lowcontrast"],
        )
    except PreflightError as exc:
        print(f"BLOCKED: Pre-flight failed: {exc}", file=sys.stderr)
        return 1

    # ── Load reference arrays from NPZ ─────────────────────────────────────
    ref_members = contract["reference_members"]
    try:
        with np.load(args.evidence_npz, allow_pickle=False) as npz_data:
            ref_legacy_raw = np.array(
                npz_data[ref_members["legacy_raw_saliency"]], copy=True
            )
            ref_legacy_classif = np.array(
                npz_data[ref_members["legacy_classification"]], copy=True
            )
            ref_m_lc_raw = np.array(
                npz_data[ref_members["controlled_modern_raw_saliency"]], copy=True
            )
            ref_m_lc_classif = np.array(
                npz_data[ref_members["controlled_modern_classification"]], copy=True
            )
    except Exception as exc:
        print(f"BLOCKED: NPZ load failed: {exc}", file=sys.stderr)
        return 1

    # Validate reference arrays
    ref_checks = [
        ("ref_legacy_raw", ref_legacy_raw, (512, 512), "float32"),
        ("ref_m_lc_raw", ref_m_lc_raw, (512, 512), "float32"),
        ("ref_legacy_classif", ref_legacy_classif, (6,), "float64"),
        ("ref_m_lc_classif", ref_m_lc_classif, (6,), "float64"),
    ]
    for label, arr, expected_shape, expected_dtype in ref_checks:
        if tuple(arr.shape) != expected_shape:
            print(
                f"BLOCKED: {label} shape {arr.shape} != {expected_shape}",
                file=sys.stderr,
            )
            return 1
        if str(arr.dtype) != expected_dtype:
            print(
                f"BLOCKED: {label} dtype {arr.dtype} != {expected_dtype}",
                file=sys.stderr,
            )
            return 1
        if not np.isfinite(arr).all():
            print(f"BLOCKED: {label} has non-finite values", file=sys.stderr)
            return 1

    # ── Lazy import of production model — ONLY HERE ─────────────────────────
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        umsi_mod = importlib.import_module("saliency.umsi_model")
    except ImportError as exc:
        print(f"BLOCKED: Cannot import saliency.umsi_model: {exc}", file=sys.stderr)
        return 1

    # Verify exact import origin against pinned source path
    module_file = Path(inspect.getfile(umsi_mod)).resolve()
    expected_module_file = source_path.resolve()
    if module_file != expected_module_file:
        print(
            f"BLOCKED: Import origin mismatch: {module_file} != {expected_module_file}",
            file=sys.stderr,
        )
        return 1

    # Pre-resolve output directory and capture all helper metadata BEFORE
    # model construction and inference so that _run_post_capture_sequence
    # receives only already-prepared plain local values.
    output_dir = Path(args.output_dir).resolve()
    module_file_str = str(module_file)
    expected_runner_sha_val = args.expected_runner_sha

    # Build complete static report payload before model construction.
    # After this call, frozen_thresholds, pinned and observed_shas are no
    # longer needed by the post-capture helper.
    report_thresholds, report_identities_static = _build_report_statics(
        frozen_thresholds=frozen_thresholds,
        pinned=pinned,
        observed_shas=observed_shas,
        module_file_str=module_file_str,
        expected_runner_sha=expected_runner_sha_val,
    )

    # ── Initialise production model ─────────────────────────────────────────
    try:
        model = umsi_mod.UMSIPlus(args.weights)
    except Exception as exc:
        print(f"BLOCKED: UMSIPlus init failed: {exc}", file=sys.stderr)
        return 1

    # ── Single real inference ────────────────────────────────────────────────
    try:
        captured, _ = capture_and_invoke(model, args.fixture)
    except (InferenceCallError, Exception) as exc:
        print(f"INFERENCE ERROR: {exc}", file=sys.stderr)
        return 1

    return _run_post_capture_sequence(
        captured=captured,
        ref_legacy_raw=ref_legacy_raw,
        ref_legacy_classif=ref_legacy_classif,
        ref_m_lc_raw=ref_m_lc_raw,
        ref_m_lc_classif=ref_m_lc_classif,
        contract=contract,
        report_thresholds=report_thresholds,
        report_identities_static=report_identities_static,
        output_dir=output_dir,
    )



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "STAGE 1 STEP 2B: Contract-driven one-shot four-gate experiment harness.\n"
            "The --execute-frozen-experiment flag is required. Without it the runner "
            "terminates before importing any production model."
        )
    )
    p.add_argument(
        "--expected-runner-sha",
        default=None,
        help=(
            "Required: exact SHA-256 (64 lowercase hex chars) of the audited "
            "runner. Must be supplied explicitly; no default. The runner will "
            "verify its own hash against this value before any production import."
        ),
    )
    p.add_argument(
        "--execute-frozen-experiment",
        action="store_true",
        default=False,
        help="Required authorisation flag. Absent → exit without importing model.",
    )
    p.add_argument(
        "--repo",
        default=str(_REPO_ROOT),
        help="Production repository root (defaults to this checkout).",
    )
    p.add_argument(
        "--contract",
        default=str(_REPO_ROOT / "stage1/evidence/umsi_raw_output_gate_contract.json"),
        help="Path to the frozen gate contract JSON.",
    )
    p.add_argument(
        "--contract-sha",
        default="4a786d8d82570cef4787a9f8d350376a753cbf66e200ee94529225b9607c7542",
        help="Expected SHA-256 of the frozen contract.",
    )
    p.add_argument(
        "--weights",
        required=True,
        help="Required path to the UMSI++ checkpoint.",
    )
    p.add_argument(
        "--evidence-npz",
        required=True,
        help="Required path to the frozen reference NPZ.",
    )
    p.add_argument(
        "--fixture",
        required=True,
        help="Required path to the hash-pinned low-contrast fixture image.",
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "stage1/evidence"),
        help="Directory in which to write the evidence bundle.",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.execute_frozen_experiment:
        print(
            "Refusing to proceed: --execute-frozen-experiment flag not provided.\n"
            "No production model was imported. This flag is required to authorise\n"
            "the real single-inference experiment.",
            file=sys.stderr,
        )
        return 1

    # Validate --expected-runner-sha before any production import.
    # _validate_sha256_hex raises PreflightError, which is not caught here —
    # it propagates as an unhandled exception. Wrap it for a clean error message.
    try:
        _validate_sha256_hex(
            args.expected_runner_sha or "",
            "--expected-runner-sha",
        )
    except PreflightError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1

    return run_frozen_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
