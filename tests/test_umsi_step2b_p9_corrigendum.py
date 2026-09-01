"""Tests for umsi_step2b_p9_corrigendum_reeval.py.

All tests use synthetic arrays only.  No real P8 bundle, no real NPZ,
no TensorFlow, no Keras, no UMSI model.

Test matrix:
  - test_pass_all_gates: Gate 1 PASS, Gate 2 PASS, Diag-1 discriminative
  - test_fail_gate_1: Gate 1 FAIL (low Pearson), Gate 2 PASS
  - test_fail_gate_2: Gate 1 PASS, Gate 2 FAIL (large classif diff)
  - test_fail_both_gates: Gate 1 FAIL, Gate 2 FAIL
  - test_inconclusive_negative_control: Diag-1 unexpected M_LC pass
  - test_invalid_integrity_evaluator_sha: wrong evaluator SHA
  - test_invalid_integrity_contract_missing_key: contract validation failure
  - test_no_forbidden_imports: static check that tf/keras/umsi are not imported
  - test_constant_map_policy: edge case — constant production map
  - test_gate_2_shape_mismatch: Gate 2 shape mismatch
  - test_conclusion_strings_have_correct_prefix: all conclusion constants
  - test_contract_validation_rejects_wrong_assessment_type
  - test_contract_validation_rejects_wrong_postprocess_order
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import copy

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Add repo root to path so we can import the evaluator without installing.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from stage1.tools.umsi_step2b_p9_corrigendum_reeval import (
    CONCLUSION_FAIL_GATE_1,
    CONCLUSION_FAIL_GATE_1_GATE_2,
    CONCLUSION_FAIL_GATE_2,
    CONCLUSION_INCONCLUSIVE,
    CONCLUSION_INVALID,
    CONCLUSION_PASS,
    ContractError,
    IntegrityError,
    GateResult,
    _apply_constant_map_policy,
    _PREFIX,
    assemble_verdict,
    evaluate_diag_1,
    evaluate_diag_2,
    evaluate_gate_1,
    evaluate_gate_2,
    load_and_validate_arrays,
    load_and_validate_contract,
    sha256_array,
    sha256_file,
    run_p9_corrigendum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SSIM_PARAMS = {
    "K1": 0.01, "K2": 0.03, "win_size": 7,
    "boundary": "reflect", "data_range": 1.0,
    "filter": "scipy.ndimage.uniform_filter",
}

_FROZEN_THRESHOLDS = {
    "pearson_min": 0.999,
    "spearman_min": 0.99,
    "ssim_min": 0.99,
    "cli_abs_max": 0.01,
    "repeatability_abs_max": 0.0,
}

_GATE_1_DEF = {
    "authoritative_shared_postprocess": {
        "order": ["squeeze", "identical_resize", "identical_min_max_to_0_1"],
        "ssim_params": _SSIM_PARAMS,
    },
    "requirements": {
        "pearson": {"operator": ">=", "value": 0.999},
        "spearman": {"operator": ">=", "value": 0.99},
        "windowed_ssim": {"operator": ">=", "value": 0.99},
    },
}

_GATE_2_DEF = {
    "comparison": "cast_both_to_float64",
    "requirements": {
        "same_shape": True,
        "finite_values": True,
        "max_abs_diff_float64": {"operator": "<=", "value": 0.01},
    },
}


def _make_smooth_map(seed: int = 0, shape: tuple = (512, 512)) -> np.ndarray:
    """Return a smooth float32 array with clear structure (not constant)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    xs = np.linspace(0, 4 * np.pi, w, dtype=np.float32)
    ys = np.linspace(0, 4 * np.pi, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    base = (np.sin(xx) * np.cos(yy)).astype(np.float32)
    noise = rng.standard_normal((h, w)).astype(np.float32) * 0.01
    return base + noise


def _make_classif(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(6).astype(np.float32)
    return (v / v.sum()).astype(np.float32)


def _make_valid_contract(
    *,
    evaluator_sha: str = "a" * 64,
    p8_gate_report_sha: str = "b" * 64,
    p8_npz_sha: str = "c" * 64,
    p8_contract_sha: str = "d" * 64,
    p8_runner_sha: str = "e" * 64,
    p8_source_sha: str = "f" * 64,
    evidence_npz_sha: str = "0" * 64,
    weights_sha: str = "1" * 64,
    fixture_sha: str = "2" * 64,
    prod_raw_sha: str = "3" * 64,
    prod_clf_sha: str = "4" * 64,
) -> Dict[str, Any]:
    # Use deepcopy so that tests mutating the returned dict cannot corrupt
    # the module-level constant dicts (_GATE_1_DEF, _GATE_2_DEF, etc.).
    return {
        "contract_name": "test_contract",
        "assessment_metadata": {
            "assessment_type": "POST_HOC_ZERO_INFERENCE_PROTOCOL_CORRIGENDUM",
            "independent_inference_performed": False,
        },
        "pinned_p9_evaluator": {
            "evaluator_sha256": evaluator_sha,
        },
        "pinned_p8_identities": {
            "p8_gate_report_sha256": p8_gate_report_sha,
            "p8_production_outputs_npz_sha256": p8_npz_sha,
            "p8_contract_sha256": p8_contract_sha,
            "p8_runner_sha256": p8_runner_sha,
            "p8_production_source_sha256": p8_source_sha,
            "evidence_npz_sha256": evidence_npz_sha,
            "weights_sha256": weights_sha,
            "fixture_sha256_lowcontrast": fixture_sha,
            "prod_raw_saliency_sha256": prod_raw_sha,
            "prod_classif_sha256": prod_clf_sha,
        },
        "reference_members": {
            "legacy_raw_saliency": "legacy_raw/lowcontrast_raw",
            "legacy_classification": "legacy_raw/lowcontrast_classif",
            "m_lc_raw_saliency_neg_control": "modern_raw/M_LC_lowcontrast_raw",
            "m_lc_classification_diag": "modern_raw/M_LC_lowcontrast_classif",
        },
        "acceptance_gates": {
            "gate_1": copy.deepcopy(_GATE_1_DEF),
            "gate_2": copy.deepcopy(_GATE_2_DEF),
        },
        "diagnostic_gates": {
            "diag_1": {"name": "saliency_negative_control_m_lc"},
            "diag_2": {"name": "classification_m_lc_proximity_diagnostic"},
        },
        "frozen_thresholds": copy.deepcopy(_FROZEN_THRESHOLDS),
        "verdict_logic": {},
    }


# ---------------------------------------------------------------------------
# Tests: conclusion prefix
# ---------------------------------------------------------------------------

def test_conclusion_strings_have_correct_prefix():
    for c in [CONCLUSION_PASS, CONCLUSION_FAIL_GATE_1, CONCLUSION_FAIL_GATE_2,
              CONCLUSION_FAIL_GATE_1_GATE_2, CONCLUSION_INCONCLUSIVE,
              CONCLUSION_INVALID]:
        assert c.startswith(_PREFIX), f"{c!r} does not start with {_PREFIX!r}"


def test_pass_conclusion_is_not_bare_pass():
    assert CONCLUSION_PASS == "STEP_2B_P9_CORRIGENDUM_PASS"
    assert CONCLUSION_PASS != "STEP_2B_P9_PASS"


# ---------------------------------------------------------------------------
# Tests: constant-map policy
# ---------------------------------------------------------------------------

def test_apply_constant_map_policy_normalizes():
    arr = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    norm, const = _apply_constant_map_policy(arr, "test")
    assert not const
    np.testing.assert_allclose(norm.min(), 0.0, atol=1e-12)
    np.testing.assert_allclose(norm.max(), 1.0, atol=1e-12)


def test_apply_constant_map_policy_constant_map():
    arr = np.full((4, 4), 5.0, dtype=np.float32)
    norm, const = _apply_constant_map_policy(arr, "test")
    assert const is True
    np.testing.assert_array_equal(norm, 0.0)


# ---------------------------------------------------------------------------
# Tests: contract validation
# ---------------------------------------------------------------------------

def test_contract_validation_rejects_wrong_assessment_type():
    c = _make_valid_contract()
    c["assessment_metadata"]["assessment_type"] = "WRONG_TYPE"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(c, fh)
        tmp = Path(fh.name)
    try:
        with pytest.raises(ContractError, match="assessment_type"):
            load_and_validate_contract(tmp)
    finally:
        tmp.unlink()


def test_contract_validation_rejects_wrong_postprocess_order():
    c = _make_valid_contract()
    c["acceptance_gates"]["gate_1"]["authoritative_shared_postprocess"][
        "order"
    ] = ["squeeze", "wrong_step"]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(c, fh)
        tmp = Path(fh.name)
    try:
        with pytest.raises(ContractError, match="postprocess order"):
            load_and_validate_contract(tmp)
    finally:
        tmp.unlink()


def test_contract_validation_rejects_missing_section():
    c = _make_valid_contract()
    del c["frozen_thresholds"]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(c, fh)
        tmp = Path(fh.name)
    try:
        with pytest.raises(ContractError):
            load_and_validate_contract(tmp)
    finally:
        tmp.unlink()


def test_contract_validation_rejects_wrong_gate2_comparison():
    c = _make_valid_contract()
    c["acceptance_gates"]["gate_2"]["comparison"] = "wrong"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(c, fh)
        tmp = Path(fh.name)
    try:
        # The validator raises ContractError mentioning cast_both_to_float64
        # when gate_2.comparison is not the required value.
        with pytest.raises(ContractError):
            load_and_validate_contract(tmp)
    finally:
        tmp.unlink()


# ---------------------------------------------------------------------------
# Tests: Gate 1
# ---------------------------------------------------------------------------

def test_gate_1_pass_near_identical_maps():
    """Gate 1 passes when production and legacy maps are nearly identical."""
    base = _make_smooth_map(seed=0)
    prod_raw = (base + np.random.default_rng(42).standard_normal(base.shape
                ).astype(np.float32) * 0.0001).reshape(512, 512, 1)
    result = evaluate_gate_1(
        prod_raw, base, _GATE_1_DEF, _FROZEN_THRESHOLDS
    )
    assert result.passed is True, f"Expected PASS, got {result.reason}"
    assert result.observed["pearson"] >= 0.999
    assert result.observed["spearman"] >= 0.99
    assert result.observed["windowed_ssim"] >= 0.99


def test_gate_1_fail_low_correlation():
    """Gate 1 fails when maps are uncorrelated."""
    rng = np.random.default_rng(7)
    prod_raw = rng.standard_normal((512, 512, 1)).astype(np.float32)
    ref_raw = rng.standard_normal((512, 512)).astype(np.float32)
    result = evaluate_gate_1(
        prod_raw, ref_raw, _GATE_1_DEF, _FROZEN_THRESHOLDS
    )
    assert result.passed is False
    assert "GATE_1" in (result.reason or "")


# ---------------------------------------------------------------------------
# Tests: Gate 2
# ---------------------------------------------------------------------------

def test_gate_2_pass_small_difference():
    prod_clf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    ref_clf = (prod_clf + 1e-7).astype(np.float64)
    result = evaluate_gate_2(prod_clf, ref_clf, _GATE_2_DEF, _FROZEN_THRESHOLDS)
    assert result.passed is True
    assert result.observed["max_abs_diff_float64"] < 0.01


def test_gate_2_fail_large_difference():
    prod_clf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    ref_clf = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    result = evaluate_gate_2(prod_clf, ref_clf, _GATE_2_DEF, _FROZEN_THRESHOLDS)
    assert result.passed is False
    assert "GATE_2" in (result.reason or "")


def test_gate_2_shape_mismatch():
    prod_clf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    ref_clf = np.array([0.5, 0.5], dtype=np.float64)
    result = evaluate_gate_2(prod_clf, ref_clf, _GATE_2_DEF, _FROZEN_THRESHOLDS)
    assert result.passed is False
    assert "MISMATCH" in (result.reason or "")


def test_gate_2_does_not_check_dtype():
    """Gate 2 must not fail due to dtype difference (float32 prod, float64 ref)."""
    prod_clf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    ref_clf = prod_clf.astype(np.float64)
    result = evaluate_gate_2(prod_clf, ref_clf, _GATE_2_DEF, _FROZEN_THRESHOLDS)
    assert result.passed is True
    assert result.observed["prod_dtype"] == "float32"
    assert result.observed["ref_dtype"] == "float64"


# ---------------------------------------------------------------------------
# Tests: Diag-1 (negative control)
# ---------------------------------------------------------------------------

def test_diag_1_no_alarm_when_maps_differ():
    """Diag-1 does not alarm when production differs from M_LC."""
    base = _make_smooth_map(seed=0)
    prod_raw = _make_smooth_map(seed=99).reshape(512, 512, 1)
    result = evaluate_diag_1(
        prod_raw, base, _FROZEN_THRESHOLDS, _SSIM_PARAMS
    )
    # Alarm must not be set; result is discriminative (passed=True or at least not False via alarm)
    assert result.reason != "DIAG_1_UNEXPECTED_M_LC_PASS"
    assert result.observed.get("alarm") is False


def test_diag_1_inconclusive_when_maps_identical():
    """Diag-1 fires INCONCLUSIVE_NEGATIVE_CONTROL when prod matches M_LC perfectly."""
    base = _make_smooth_map(seed=0)
    # Use the SAME map for both prod and M_LC ref (unexpected pass scenario)
    prod_raw = base.reshape(512, 512, 1)
    result = evaluate_diag_1(
        prod_raw, base, _FROZEN_THRESHOLDS, _SSIM_PARAMS
    )
    assert result.passed is False
    assert result.reason == "DIAG_1_UNEXPECTED_M_LC_PASS"
    assert result.observed.get("alarm") is True


# ---------------------------------------------------------------------------
# Tests: Diag-2 (record-only, no dtype check)
# ---------------------------------------------------------------------------

def test_diag_2_records_without_pass_fail():
    prod_clf = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    ref_clf = (prod_clf + 1e-8).astype(np.float64)
    result = evaluate_diag_2(prod_clf, ref_clf)
    assert result.passed is None  # record-only
    assert result.observed["prod_dtype"] == "float32"
    assert result.observed["ref_dtype"] == "float64"
    assert result.observed["no_dtype_requirement"] is True
    assert "max_abs_diff_float64" in result.observed


def test_diag_2_float64_reference_does_not_fail():
    """Diag-2 must not fail on float32 vs float64 dtype mismatch."""
    prod = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1], dtype=np.float32)
    ref = prod.astype(np.float64)
    result = evaluate_diag_2(prod, ref)
    assert result.passed is None
    assert result.reason != "GATE_C_DTYPE_MISMATCH"


# ---------------------------------------------------------------------------
# Tests: assemble_verdict
# ---------------------------------------------------------------------------

def _make_gate_result(name: str, passed: bool) -> GateResult:
    return GateResult(name=name, passed=passed, observed={})


def _make_diag_result_normal() -> GateResult:
    r = GateResult(name="diag_1", passed=True, observed={"alarm": False})
    return r


def _make_diag_result_alarm() -> GateResult:
    r = GateResult(
        name="diag_1", passed=False,
        reason="DIAG_1_UNEXPECTED_M_LC_PASS",
        observed={"alarm": True},
    )
    return r


_DUMMY_PINNED = {
    "p8_production_source_sha256": "f" * 64,
    "weights_sha256": "1" * 64,
}


def test_verdict_pass():
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_PASS
    assert norms == "UNBLOCKED"
    assert meta["p8_historical_result_preserved"] is True
    assert meta["independent_inference_performed"] is False
    assert "pass_meaning" in meta


def test_verdict_fail_gate_1():
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", False),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_FAIL_GATE_1
    assert norms == "BLOCKED"


def test_verdict_fail_gate_2():
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", False),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_FAIL_GATE_2
    assert norms == "BLOCKED"


def test_verdict_fail_both():
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", False),
        gate_2=_make_gate_result("gate_2", False),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_FAIL_GATE_1_GATE_2
    assert norms == "BLOCKED"


def test_verdict_inconclusive_overrides_pass():
    """INCONCLUSIVE must override a PASS when diag_1 alarm fires, even if both gates pass."""
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_alarm(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"


def test_verdict_inconclusive_overrides_fail():
    """INCONCLUSIVE must override FAIL (priority: INCONCLUSIVE > FAIL > PASS)."""
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", False),
        gate_2=_make_gate_result("gate_2", False),
        diag_1=_make_diag_result_alarm(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"


def test_verdict_norms_blocked_if_source_sha_mismatch():
    """Source SHA mismatch must be INVALID + BLOCKED, never PASS + BLOCKED."""
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="different_sha" + "0" * 51,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INVALID
    assert norms == "BLOCKED"
    assert meta["source_sha_match"] is False
    assert meta["weights_sha_match"] is True


def test_verdict_norms_blocked_if_weights_sha_mismatch():
    """Weights SHA mismatch must be INVALID + BLOCKED, never PASS + BLOCKED."""
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="different_weights_sha" + "0" * 44,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INVALID
    assert norms == "BLOCKED"
    assert meta["source_sha_match"] is True
    assert meta["weights_sha_match"] is False


def test_verdict_norms_blocked_if_both_shas_mismatch():
    """Both SHAs mismatching must be INVALID + BLOCKED, never PASS + BLOCKED."""
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="bad_source" + "0" * 54,
        p8_weights_sha="bad_weights" + "0" * 53,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INVALID
    assert norms == "BLOCKED"
    assert meta["source_sha_match"] is False
    assert meta["weights_sha_match"] is False


def test_diag_1_is_blocking_negative_control_not_records_only():
    """Diag-1 is a blocking negative control, not a passive records_only diagnostic.

    When production unexpectedly matches M_LC, the conclusion must be
    INCONCLUSIVE_NEGATIVE_CONTROL, not a PASS or a record-only outcome.
    """
    # Near-identical maps trigger the alarm
    base = _make_smooth_map(seed=0)
    prod_raw = base.reshape(512, 512, 1)
    result = evaluate_diag_1(
        prod_raw, base, _FROZEN_THRESHOLDS, _SSIM_PARAMS
    )
    assert result.passed is False, (
        "Diag-1 must return passed=False when M_LC alarm fires"
    )
    assert result.reason == "DIAG_1_UNEXPECTED_M_LC_PASS", (
        f"Expected DIAG_1_UNEXPECTED_M_LC_PASS, got {result.reason!r}"
    )
    # Confirm that assemble_verdict then produces INCONCLUSIVE (blocking)
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=result,
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"


def test_diag_1_constant_map_abort_cannot_pass():
    """A constant-map abort in diag_1 must never allow a PASS conclusion."""
    diag_1_abort = GateResult(
        name="diag_1", passed=None,
        reason="DIAG_1_CONSTANT_MAP_ABORT",
        observed={
            "alarm": False,
            "alarm_reason": "constant_map_abort; negative control indeterminate",
        },
    )
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=diag_1_abort,
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c != CONCLUSION_PASS
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"
    assert meta["negative_control_status"] == "UNEVALUABLE"


def test_diag_1_compute_error_cannot_pass():
    """A diag_1 compute error must never allow a PASS conclusion."""
    diag_1_error = GateResult(
        name="diag_1", passed=None,
        reason="DIAG_1_COMPUTE_ERROR: boom",
        observed={},
    )
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=diag_1_error,
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c != CONCLUSION_PASS
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"
    assert meta["negative_control_status"] == "UNEVALUABLE"


def test_diag_1_evaluator_exception_cannot_pass():
    """A _safe_evaluate-wrapped diag_1 exception must never allow a PASS conclusion."""
    diag_1_exc = GateResult(
        name="diag_1", passed=False,
        reason="EVALUATOR_EXCEPTION",
        observed={
            "evaluator_exception_type": "ValueError",
            "evaluator_exception_message": "boom",
        },
    )
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=diag_1_exc,
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c != CONCLUSION_PASS
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"
    assert meta["negative_control_status"] == "UNEVALUABLE"


def test_diag_1_malformed_alarm_state_cannot_pass():
    """diag_1 passed=True with a missing/malformed alarm field must never PASS."""
    diag_1_malformed = GateResult(
        name="diag_1", passed=True,
        reason=None,
        observed={},  # missing 'alarm' key entirely
    )
    c, norms, meta = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=diag_1_malformed,
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c != CONCLUSION_PASS
    assert c == CONCLUSION_INCONCLUSIVE
    assert norms == "BLOCKED"
    assert meta["negative_control_status"] == "UNEVALUABLE"


def test_pass_never_coexists_with_blocked_norms():
    """Invariant: PASS implies UNBLOCKED, and UNBLOCKED implies PASS."""
    c, norms, _ = assemble_verdict(
        gate_1=_make_gate_result("gate_1", True),
        gate_2=_make_gate_result("gate_2", True),
        diag_1=_make_diag_result_normal(),
        p9_contract_sha="c" * 64,
        p8_source_sha="f" * 64,
        p8_weights_sha="1" * 64,
        pinned=_DUMMY_PINNED,
    )
    assert c == CONCLUSION_PASS
    assert norms == "UNBLOCKED"
    assert not (c == CONCLUSION_PASS and norms == "BLOCKED")
    assert not (norms == "UNBLOCKED" and c != CONCLUSION_PASS)


# ---------------------------------------------------------------------------
# Tests: full run_p9_corrigendum via temp filesystem
# ---------------------------------------------------------------------------

def _write_contract_file(path: Path, contract: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(contract, fh)


def _setup_temp_run(
    tmp_path: Path,
    *,
    prod_raw: np.ndarray,
    prod_clf: np.ndarray,
    ref_leg_raw: np.ndarray,
    ref_leg_clf: np.ndarray,
    ref_mlc_raw: np.ndarray,
    ref_mlc_clf: np.ndarray,
    overwrite_prod_raw_sha: bool = False,
    p8_conclusion: str = "STEP_2B_FAIL_GATE_A_GATE_C",
) -> tuple:
    """
    Write all synthetic artifacts to tmp_path and return (args_namespace, contract).
    Returns (args, contract_path, output_dir).
    """
    # Production NPZ
    p8_npz_path = tmp_path / "production_outputs.npz"
    np.savez(str(p8_npz_path),
             production_raw_saliency=prod_raw,
             production_classif=prod_clf)

    # P8 gate_report.json
    p8_report = {
        "conclusion": p8_conclusion,
        "utc_timestamp": "2026-08-03T13:04:15.700906Z",
    }
    p8_report_path = tmp_path / "gate_report.json"
    with open(p8_report_path, "w") as fh:
        json.dump(p8_report, fh)

    # Reference NPZ
    ref_npz_path = tmp_path / "ref.npz"
    np.savez(
        str(ref_npz_path),
        **{
            "legacy_raw/lowcontrast_raw": ref_leg_raw,
            "legacy_raw/lowcontrast_classif": ref_leg_clf,
            "modern_raw/M_LC_lowcontrast_raw": ref_mlc_raw,
            "modern_raw/M_LC_lowcontrast_classif": ref_mlc_clf,
        }
    )

    # Dummy weight/fixture/runner/source files
    weights_path = tmp_path / "weights.hdf5"
    fixture_path = tmp_path / "lowcontrast.png"
    runner_path = tmp_path / "runner.py"
    source_path = tmp_path / "umsi_model.py"
    p8_contract_path = tmp_path / "p8_contract.json"
    for f in [weights_path, fixture_path, runner_path, source_path]:
        f.write_bytes(b"dummy")
    p8_contract_path.write_text('{"dummy": true}')

    # Compute real SHAs
    prod_raw_sha = sha256_array(prod_raw.squeeze())
    if overwrite_prod_raw_sha:
        prod_raw_sha = "a" * 64  # deliberately wrong
    prod_clf_sha = sha256_array(prod_clf)
    p8_report_sha = sha256_file(p8_report_path)
    p8_npz_sha = sha256_file(p8_npz_path)
    p8_contract_file_sha = sha256_file(p8_contract_path)
    runner_sha = sha256_file(runner_path)
    source_sha = sha256_file(source_path)
    evidence_sha = sha256_file(ref_npz_path)
    weights_sha = sha256_file(weights_path)
    fixture_sha = sha256_file(fixture_path)

    contract = _make_valid_contract(
        p8_gate_report_sha=p8_report_sha,
        p8_npz_sha=p8_npz_sha,
        p8_contract_sha=p8_contract_file_sha,
        p8_runner_sha=runner_sha,
        p8_source_sha=source_sha,
        evidence_npz_sha=evidence_sha,
        weights_sha=weights_sha,
        fixture_sha=fixture_sha,
        prod_raw_sha=prod_raw_sha,
        prod_clf_sha=prod_clf_sha,
    )

    # Evaluator path (real file)
    evaluator_path = (
        Path(__file__).resolve().parent.parent
        / "stage1" / "tools" / "umsi_step2b_p9_corrigendum_reeval.py"
    )
    evaluator_sha = sha256_file(evaluator_path)
    contract["pinned_p9_evaluator"] = {
        "evaluator_sha256": evaluator_sha
    }

    contract_path = tmp_path / "contract_p9.json"
    _write_contract_file(contract_path, contract)
    contract_sha = sha256_file(contract_path)

    output_dir = tmp_path / "output"

    import argparse
    args = argparse.Namespace(
        expected_evaluator_sha=evaluator_sha,
        contract=str(contract_path),
        contract_sha=contract_sha,
        p8_bundle=str(tmp_path),
        p8_contract=str(p8_contract_path),
        p8_runner=str(runner_path),
        p8_source=str(source_path),
        evidence_npz=str(ref_npz_path),
        weights=str(weights_path),
        fixture=str(fixture_path),
        output_dir=str(output_dir),
    )
    return args, contract_path, output_dir


def _make_near_identical_prod(base_map: np.ndarray) -> np.ndarray:
    noise = np.random.default_rng(1).standard_normal(base_map.shape
                                                     ).astype(np.float32) * 0.0001
    return (base_map + noise).reshape(512, 512, 1)


def _make_uncorrelated_map() -> np.ndarray:
    return np.random.default_rng(42).standard_normal(
        (512, 512, 1)).astype(np.float32)


def test_full_run_pass(tmp_path):
    """Full run: Gate 1 PASS, Gate 2 PASS, Diag-1 discriminative -> PASS."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)    # different from prod -> no alarm
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )
    rc = run_p9_corrigendum(args)
    assert rc == 0
    bundles = list(output_dir.iterdir())
    assert len(bundles) == 1
    report_path = bundles[0] / "gate_report_p9.json"
    assert report_path.exists()
    with open(report_path) as fh:
        report = json.load(fh)
    assert report["conclusion"] == CONCLUSION_PASS
    assert report["verdict"]["saliency_norms_release_status"] == "UNBLOCKED"
    assert report["verdict"]["p8_historical_result"] == "STEP_2B_FAIL_GATE_A_GATE_C"
    assert report["verdict"]["independent_inference_performed"] is False
    # Confirm no production_outputs.npz was written (P8 bundle is not duplicated)
    assert not (output_dir / "production_outputs.npz").exists()


def test_full_run_fail_gate_1(tmp_path):
    """Full run: Gate 1 FAIL (uncorrelated prod map)."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_uncorrelated_map()
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )
    rc = run_p9_corrigendum(args)
    assert rc == 2
    bundles = list(output_dir.iterdir())
    report = json.load(open(bundles[0] / "gate_report_p9.json"))
    assert report["conclusion"] == CONCLUSION_FAIL_GATE_1


def test_full_run_fail_gate_2(tmp_path):
    """Full run: Gate 2 FAIL (large classif difference)."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    # Make legacy classif very different
    leg_clf = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )
    rc = run_p9_corrigendum(args)
    assert rc == 2
    bundles = list(output_dir.iterdir())
    report = json.load(open(bundles[0] / "gate_report_p9.json"))
    assert report["conclusion"] == CONCLUSION_FAIL_GATE_2


def test_full_run_inconclusive_negative_control(tmp_path):
    """Full run: Diag-1 alarm fires when prod matches M_LC perfectly."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    # Make M_LC identical to production to trigger INCONCLUSIVE
    mlc_raw = prod_raw.squeeze()
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )
    rc = run_p9_corrigendum(args)
    assert rc == 2
    bundles = [b for b in output_dir.iterdir() if "INVALID" not in b.name]
    assert len(bundles) == 1
    report = json.load(open(bundles[0] / "gate_report_p9.json"))
    assert report["conclusion"] == CONCLUSION_INCONCLUSIVE


def test_full_run_invalid_integrity_wrong_prod_sha(tmp_path):
    """Full run: integrity fails when prod_raw SHA is wrong."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
        overwrite_prod_raw_sha=True,  # intentionally wrong
    )
    rc = run_p9_corrigendum(args)
    assert rc == 1  # INVALID


def test_full_run_invalid_integrity_wrong_p8_conclusion(tmp_path):
    """Full run: integrity fails if P8 historical conclusion is not the expected one."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
        p8_conclusion="STEP_2B_PASS_ALL_GATES",  # wrong
    )
    rc = run_p9_corrigendum(args)
    assert rc == 1  # INVALID


def test_full_run_invalid_integrity_contract_pinned_evaluator_sha_mismatch(tmp_path):
    """Full run: INVALID when the contract-pinned evaluator SHA differs from
    the actual on-disk/CLI-expected evaluator SHA, even though the CLI SHA
    itself is correct.
    """
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, contract_path, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )

    # Tamper the on-disk contract's pinned evaluator SHA to a well-formed
    # but incorrect value, then recompute the contract SHA the CLI expects
    # so that the contract-file-integrity check itself still passes and the
    # mismatch is isolated to the evaluator-identity cross-check.
    with open(contract_path) as fh:
        contract = json.load(fh)
    contract["pinned_p9_evaluator"]["evaluator_sha256"] = "9" * 64
    with open(contract_path, "w") as fh:
        json.dump(contract, fh)
    args.contract_sha = sha256_file(contract_path)

    rc = run_p9_corrigendum(args)
    assert rc == 1  # INVALID


def test_full_run_no_production_outputs_written(tmp_path):
    """P8 production_outputs.npz must never be created in the output bundle."""
    leg_raw = _make_smooth_map(seed=0)
    prod_raw = _make_near_identical_prod(leg_raw)
    prod_clf = _make_classif(seed=0)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=99)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)

    args, _, output_dir = _setup_temp_run(
        tmp_path,
        prod_raw=prod_raw, prod_clf=prod_clf,
        ref_leg_raw=leg_raw, ref_leg_clf=leg_clf,
        ref_mlc_raw=mlc_raw, ref_mlc_clf=mlc_clf,
    )
    run_p9_corrigendum(args)
    # Output dir may not exist if the run exited before writing (INVALID).
    # When it does exist, no bundle should contain production_outputs.npz.
    if output_dir.exists():
        for bundle in output_dir.iterdir():
            assert not (bundle / "production_outputs.npz").exists(), (
                "production_outputs.npz must not be written to the P9 bundle"
            )


# ---------------------------------------------------------------------------
# Array-integrity unit tests (load_and_validate_arrays)
# ---------------------------------------------------------------------------

def _write_valid_reference_npz(path: Path, prod_clf: np.ndarray) -> Dict[str, str]:
    leg_raw = _make_smooth_map(seed=1)
    leg_clf = (prod_clf + 1e-8).astype(np.float64)
    mlc_raw = _make_smooth_map(seed=2)
    mlc_clf = (prod_clf + 2e-8).astype(np.float64)
    np.savez(
        str(path),
        **{
            "legacy_raw/lowcontrast_raw": leg_raw,
            "legacy_raw/lowcontrast_classif": leg_clf,
            "modern_raw/M_LC_lowcontrast_raw": mlc_raw,
            "modern_raw/M_LC_lowcontrast_classif": mlc_clf,
        },
    )
    return {
        "legacy_raw_saliency": "legacy_raw/lowcontrast_raw",
        "legacy_classification": "legacy_raw/lowcontrast_classif",
        "m_lc_raw_saliency_neg_control": "modern_raw/M_LC_lowcontrast_raw",
        "m_lc_classification_diag": "modern_raw/M_LC_lowcontrast_classif",
    }


def test_array_validation_missing_npz_key_raises_integrity_error(tmp_path):
    """A missing required NPZ key must raise IntegrityError, not KeyError."""
    prod_clf = _make_classif(seed=0)
    p8_npz_path = tmp_path / "production_outputs.npz"
    # 'production_classif' key is missing entirely
    np.savez(str(p8_npz_path), production_raw_saliency=np.zeros((512, 512, 1), dtype=np.float32))

    ref_npz_path = tmp_path / "ref.npz"
    reference_members = _write_valid_reference_npz(ref_npz_path, prod_clf)
    pinned = {"prod_raw_saliency_sha256": "0" * 64, "prod_classif_sha256": "0" * 64}

    with pytest.raises(IntegrityError, match="Missing required key"):
        load_and_validate_arrays(
            p8_npz_path=p8_npz_path,
            evidence_npz_path=ref_npz_path,
            reference_members=reference_members,
            pinned=pinned,
        )


def test_array_validation_wrong_shape_raises_integrity_error(tmp_path):
    """A wrong production array shape must raise IntegrityError."""
    bad_raw = np.zeros((256, 256, 1), dtype=np.float32)  # wrong shape
    prod_clf = _make_classif(seed=0)
    p8_npz_path = tmp_path / "production_outputs.npz"
    np.savez(str(p8_npz_path), production_raw_saliency=bad_raw, production_classif=prod_clf)

    ref_npz_path = tmp_path / "ref.npz"
    reference_members = _write_valid_reference_npz(ref_npz_path, prod_clf)
    pinned = {
        "prod_raw_saliency_sha256": sha256_array(bad_raw),
        "prod_classif_sha256": sha256_array(prod_clf),
    }

    with pytest.raises(IntegrityError, match="shape"):
        load_and_validate_arrays(
            p8_npz_path=p8_npz_path,
            evidence_npz_path=ref_npz_path,
            reference_members=reference_members,
            pinned=pinned,
        )


def test_array_validation_wrong_dtype_raises_integrity_error(tmp_path):
    """A wrong production array dtype must raise IntegrityError."""
    bad_raw = np.zeros((512, 512, 1), dtype=np.float64)  # wrong dtype
    prod_clf = _make_classif(seed=0)
    p8_npz_path = tmp_path / "production_outputs.npz"
    np.savez(str(p8_npz_path), production_raw_saliency=bad_raw, production_classif=prod_clf)

    ref_npz_path = tmp_path / "ref.npz"
    reference_members = _write_valid_reference_npz(ref_npz_path, prod_clf)
    pinned = {
        "prod_raw_saliency_sha256": sha256_array(bad_raw),
        "prod_classif_sha256": sha256_array(prod_clf),
    }

    with pytest.raises(IntegrityError, match="dtype"):
        load_and_validate_arrays(
            p8_npz_path=p8_npz_path,
            evidence_npz_path=ref_npz_path,
            reference_members=reference_members,
            pinned=pinned,
        )


def test_array_validation_non_finite_raises_integrity_error(tmp_path):
    """A production array containing NaN/Inf must raise IntegrityError."""
    bad_raw = np.zeros((512, 512, 1), dtype=np.float32)
    bad_raw[0, 0, 0] = np.nan
    prod_clf = _make_classif(seed=0)
    p8_npz_path = tmp_path / "production_outputs.npz"
    np.savez(str(p8_npz_path), production_raw_saliency=bad_raw, production_classif=prod_clf)

    ref_npz_path = tmp_path / "ref.npz"
    reference_members = _write_valid_reference_npz(ref_npz_path, prod_clf)
    pinned = {
        "prod_raw_saliency_sha256": sha256_array(bad_raw),
        "prod_classif_sha256": sha256_array(prod_clf),
    }

    with pytest.raises(IntegrityError, match="non-finite"):
        load_and_validate_arrays(
            p8_npz_path=p8_npz_path,
            evidence_npz_path=ref_npz_path,
            reference_members=reference_members,
            pinned=pinned,
        )


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def test_no_forbidden_imports_in_evaluator_source():
    """The evaluator source must contain no tf/keras/umsi/torch import statements."""
    evaluator_path = (
        Path(__file__).resolve().parent.parent
        / "stage1" / "tools" / "umsi_step2b_p9_corrigendum_reeval.py"
    )
    source = evaluator_path.read_text(encoding="utf-8")
    import ast
    tree = ast.parse(source, filename=str(evaluator_path))

    forbidden: list = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in {"tensorflow", "keras", "torch", "saliency", "umsi_model"}:
                    forbidden.append(name)

    assert not forbidden, (
        f"Evaluator contains forbidden import(s): {forbidden}"
    )


# Environment-variable marker identifying the fresh evaluator-import child
# process. Its presence is what makes the ``if __name__ == "__main__"``
# bootstrap below run the check; its absence during normal pytest
# collection/import means that bootstrap is always a no-op then. The
# wrapper test also refuses to proceed if it finds this marker already set
# in ITS OWN environment, which would indicate an (unexpected) attempt to
# recurse into another child.
_P9_CHILD_MARKER = "UMSI_P9_CHILD_MARKER"


def _p9_child_main() -> int:
    """Fresh-process entry point for the evaluator-import forbidden-module check.

    Imports the SAME evaluator module previously verified
    (``stage1.tools.umsi_step2b_p9_corrigendum_reeval``) into a brand-new
    process with a pristine ``sys.modules``, then asserts that doing so does
    not activate TensorFlow, Keras, or ``saliency.umsi_model``. Returns 0 on
    success; any failed check raises, producing a nonzero exit code with a
    traceback on stderr for the parent process to report.
    """
    import importlib

    importlib.import_module("stage1.tools.umsi_step2b_p9_corrigendum_reeval")

    # Compact module-NAME checks only -- never stringify the whole
    # sys.modules mapping (that stringification is what makes an equivalent
    # in-process check pathologically slow once another test file has
    # already loaded TensorFlow's large dependency tree into a shared
    # pytest process).
    for forbidden in ["tensorflow", "keras", "torch"]:
        assert forbidden not in sys.modules, (
            f"Forbidden module {forbidden!r} found in sys.modules after "
            "importing the evaluator"
        )
    assert "saliency.umsi_model" not in sys.modules, (
        "saliency.umsi_model must not be imported after importing the evaluator"
    )
    return 0


if __name__ == "__main__" and os.environ.get(_P9_CHILD_MARKER) == "1":
    # Child-process bootstrap only. This branch is never reached during
    # normal pytest collection/import, and it is only reached in this
    # file's own child invocations spawned by the wrapper test below -- so
    # there is no path by which this can recurse into spawning a further
    # child.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    sys.exit(_p9_child_main())


def test_evaluator_module_does_not_activate_forbidden_imports():
    """Importing the evaluator module must not cause tf/keras to appear in sys.modules.

    This check runs in a FRESH subprocess (see ``_p9_child_main`` below),
    because checking ``sys.modules`` in the shared pytest process is
    order-dependent: an earlier test file in the same session
    (``test_umsi_boundary_input_probe.py::test_preprocess_image_matches_legacy_float64_order``)
    legitimately loads real TensorFlow via ``saliency.umsi_model``. A fresh
    child process starts with none of that pre-existing state, so this
    verifies the evaluator module's OWN import graph rather than leftover
    state from unrelated tests in the same session.
    """
    if os.environ.get(_P9_CHILD_MARKER):
        raise RuntimeError(
            "test_evaluator_module_does_not_activate_forbidden_imports must "
            "not be invoked from within its own child process (recursion guard)."
        )

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env[_P9_CHILD_MARKER] = "1"

    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        out_tail = str(exc.stdout or "")[-2000:]
        err_tail = str(exc.stderr or "")[-2000:]
        raise AssertionError(
            f"P9 evaluator-import child process timed out after {exc.timeout}s.\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )

    if result.returncode != 0:
        out_tail = result.stdout[-2000:]
        err_tail = result.stderr[-2000:]
        raise AssertionError(
            f"P9 evaluator-import child process failed (exit={result.returncode}).\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )


# ---------------------------------------------------------------------------
# On-disk frozen-pair test
# ---------------------------------------------------------------------------

def test_real_on_disk_frozen_pair():
    """Verify the final on-disk evaluator/contract pair is internally consistent.

    Loads the real production contract through the evaluator's actual
    load_and_validate_contract function.  Recomputes both SHAs from disk and
    verifies them against the declared final values and against the SHA
    stored inside the contract.  Reads no P8 production or reference array
    and performs no inference.
    """
    _DECLARED_EVALUATOR_SHA = (
        "6986e7921661dfe1d53b02c032841e47b465789d1164be979fe1bfb98bbc2900"
    )
    _DECLARED_CONTRACT_SHA = (
        "fd9105e6e88bb85578c06a20026f98f8e886dd4f0f90ccc3699f1e90ece80b58"
    )

    repo_root = Path(__file__).resolve().parent.parent
    evaluator_path = (
        repo_root / "stage1" / "tools" / "umsi_step2b_p9_corrigendum_reeval.py"
    )
    contract_path = (
        repo_root
        / "stage1"
        / "evidence"
        / "umsi_raw_output_gate_contract_p9_corrigendum.json"
    )

    # Both files must exist
    assert evaluator_path.is_file(), f"Evaluator not found: {evaluator_path}"
    assert contract_path.is_file(), f"Contract not found: {contract_path}"

    # 1. Recompute SHA-256 of the real on-disk evaluator
    actual_evaluator_sha = sha256_file(evaluator_path)
    assert actual_evaluator_sha == _DECLARED_EVALUATOR_SHA, (
        f"Evaluator SHA mismatch.\n"
        f"  computed: {actual_evaluator_sha}\n"
        f"  declared: {_DECLARED_EVALUATOR_SHA}"
    )

    # 2. Recompute SHA-256 of the real on-disk contract
    actual_contract_sha = sha256_file(contract_path)
    assert actual_contract_sha == _DECLARED_CONTRACT_SHA, (
        f"Contract SHA mismatch.\n"
        f"  computed: {actual_contract_sha}\n"
        f"  declared: {_DECLARED_CONTRACT_SHA}"
    )

    # 3. Load and validate the real contract through the evaluator's own validator.
    #    This is exactly the call made during a real P9 execution.
    #    ContractError here is a hard failure — do not catch it.
    contract = load_and_validate_contract(contract_path)

    # 4. The evaluator SHA pinned inside the contract must equal the actual SHA.
    sha_in_contract = contract["pinned_p9_evaluator"]["evaluator_sha256"]
    assert sha_in_contract == actual_evaluator_sha, (
        f"pinned_p9_evaluator.evaluator_sha256 in contract does not match "
        f"the actual on-disk evaluator SHA.\n"
        f"  in contract: {sha_in_contract}\n"
        f"  on disk:     {actual_evaluator_sha}"
    )
    assert sha_in_contract == _DECLARED_EVALUATOR_SHA, (
        f"pinned_p9_evaluator.evaluator_sha256 in contract does not match "
        f"the declared final SHA.\n"
        f"  in contract: {sha_in_contract}\n"
        f"  declared:    {_DECLARED_EVALUATOR_SHA}"
    )

    # 5. Verify binding metadata
    meta = contract["assessment_metadata"]
    assert meta["assessment_type"] == "POST_HOC_ZERO_INFERENCE_PROTOCOL_CORRIGENDUM", (
        f"assessment_type mismatch: {meta['assessment_type']!r}"
    )
    assert meta["independent_inference_performed"] is False, (
        "independent_inference_performed must be false"
    )
    assert meta.get("p8_historical_result_preserved") is True, (
        "p8_historical_result_preserved must be true"
    )

    # 6. Verify the P8 historical result on record in the contract successor block
    successor = contract.get("successor_of", {})
    assert successor.get("p8_conclusion_on_record") == "STEP_2B_FAIL_GATE_A_GATE_C", (
        f"P8 historical conclusion in contract: "
        f"{successor.get('p8_conclusion_on_record')!r}"
    )

    # 7. Verify Diag-1 role: must be a blocking negative control, not records_only.
    diag_1 = contract["diagnostic_gates"]["diag_1"]
    assert diag_1.get("positive_acceptance_gate") is False, (
        "diag_1.positive_acceptance_gate must be false"
    )
    assert diag_1.get("verdict_role") == "BLOCKING_NEGATIVE_CONTROL", (
        f"diag_1.verdict_role must be BLOCKING_NEGATIVE_CONTROL, "
        f"got {diag_1.get('verdict_role')!r}"
    )
    assert "records_only" not in diag_1, (
        "diag_1 must not contain the ambiguous 'records_only' field"
    )
    assert diag_1["unexpected_pass_alarm"]["is_blocking"] is True, (
        "diag_1.unexpected_pass_alarm.is_blocking must be true"
    )

    # 8. Verify norms release condition explicitly requires both source and weights.
    norms_release = contract["verdict_logic"]["saliency_norms_release"]
    unblocked_condition = norms_release.get("UNBLOCKED_if", "")
    assert "weights_sha256" in unblocked_condition or "weights" in unblocked_condition, (
        "UNBLOCKED_if must reference weights identity: "
        f"{unblocked_condition!r}"
    )
