"""P2 / AG-05 regression tests for the successor production evidence gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from saliency.postprocessing import postprocess_saliency
from stage1.tools import umsi_production_postprocess_gate as gate


THRESHOLDS = {
    "endpoint_self_abs_max": 0.0,
    "e2e_abs_max": 0.02,
    "feature_abs_max": 0.02,
    "peak_count_exact": True,
    "peak_positions_exact": True,
}


def _evaluate(
    production_raw: np.ndarray,
    legacy_raw: np.ndarray | None = None,
    production_e2e: np.ndarray | None = None,
) -> gate.GateResult:
    legacy_raw = production_raw if legacy_raw is None else legacy_raw
    if production_e2e is None:
        production_e2e = postprocess_saliency(production_raw, 64, 64)
    return gate.evaluate_fixture(
        production_raw=production_raw,
        production_e2e=production_e2e,
        legacy_raw=legacy_raw,
        original_h=64,
        original_w=64,
        thresholds=THRESHOLDS,
    )


def _gaussian(center_x: float, center_y: float, size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 6.0 ** 2)
    ).astype(np.float32)


def test_identical_production_and_reference_pass_all_p2_checks():
    raw = _gaussian(31.0, 29.0)

    result = _evaluate(raw)

    assert result.passed is True
    assert result.reason is None
    assert result.observed["output_contract"]["passed"] is True
    assert result.observed["production_endpoint_self_check"]["max_abs_diff_float64"] == 0.0
    assert result.observed["reference_value_check"]["max_abs_diff_float64"] == 0.0
    assert result.observed["feature_check"]["max_abs_deviation"] == 0.0
    assert result.observed["peak_position_check"]["positions_exact"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda arr: arr * np.float32(2.0),
        lambda arr: arr + np.float32(0.25),
    ],
    ids=["scaled", "positive-offset"],
)
def test_affine_mutation_fails_even_when_correlation_would_be_one(mutator):
    raw = _gaussian(31.0, 29.0)
    correct = postprocess_saliency(raw, 64, 64)
    mutated = mutator(correct)
    assert np.corrcoef(correct.ravel(), mutated.ravel())[0, 1] == pytest.approx(1.0)

    result = _evaluate(raw, production_e2e=mutated)

    assert result.passed is False
    assert result.observed["output_contract"]["passed"] is False
    assert result.observed["production_endpoint_self_check"]["passed"] is False
    assert result.observed["reference_value_check"]["passed"] is False
    assert result.observed["feature_check"]["passed"] is False


def test_gate_fails_if_production_postprocessor_regresses_to_max_only(monkeypatch):
    raw = np.array([[-8.0, -4.0], [-2.0, 4.0]], dtype=np.float32)
    correct = postprocess_saliency(raw, 64, 64)

    def max_only(pred, original_h, original_w):
        resized = np.resize(np.squeeze(pred), (original_h, original_w)).astype(np.float32)
        vmax = resized.max()
        return resized / vmax if vmax > 0 else resized

    monkeypatch.setattr(gate, "postprocess_saliency", max_only)

    result = _evaluate(raw, production_e2e=correct)

    assert result.passed is False
    assert result.reason == "PRODUCTION_ENDPOINT_SELF_MISMATCH"


def test_feature_deviation_above_frozen_limit_is_binding():
    production_raw = _gaussian(18.0, 32.0)
    legacy_raw = _gaussian(46.0, 32.0)

    result = _evaluate(production_raw, legacy_raw=legacy_raw)

    assert result.passed is False
    assert result.observed["feature_check"]["max_abs_deviation"] > 0.02
    assert result.observed["feature_check"]["passed"] is False


def test_peak_positions_are_exact_not_just_peak_count():
    production_raw = _gaussian(31.0, 31.0)
    legacy_raw = _gaussian(32.0, 31.0)

    result = _evaluate(production_raw, legacy_raw=legacy_raw)

    feature = result.observed["feature_check"]
    peaks = result.observed["peak_position_check"]
    assert feature["peak_count_exact"] is True
    assert peaks["positions_exact"] is False
    assert peaks["mismatch_pixels"] > 0
    assert peaks["passed"] is False


def test_non_finite_captured_endpoint_fails_closed():
    raw = _gaussian(31.0, 29.0)
    invalid = postprocess_saliency(raw, 64, 64)
    invalid[0, 0] = np.nan

    result = _evaluate(raw, production_e2e=invalid)

    assert result.passed is False
    assert result.reason.startswith("FEATURE_OR_PEAK_ERROR")
    assert result.observed["output_contract"]["finite"] is False


def test_production_postprocessor_is_exercised_for_both_raw_maps(monkeypatch):
    raw = _gaussian(31.0, 29.0)
    correct = postprocess_saliency(raw, 64, 64)
    original = gate.postprocess_saliency
    calls = []

    def spy(pred, original_h, original_w):
        calls.append(np.asarray(pred).shape)
        return original(pred, original_h, original_w)

    monkeypatch.setattr(gate, "postprocess_saliency", spy)

    result = _evaluate(raw, production_e2e=correct)

    assert result.passed is True
    assert calls == [(64, 64), (64, 64)]


def test_contract_rejects_relaxed_or_missing_acceptance_rules(tmp_path):
    contract = {
        "contract_name": "umsi_production_postprocess_gate_contract",
        "scope": "P2_AG_05_ONLY",
        "historical_artifacts_immutable": True,
        "historical_reference_contract": {"filename": "p10.json", "sha256": "a" * 64},
        "pinned_sources": {"postprocess": {"path": "x.py", "sha256": "b" * 64}},
        "thresholds": dict(THRESHOLDS),
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    assert gate.load_and_validate_contract(path)["scope"] == "P2_AG_05_ONLY"

    contract["thresholds"]["endpoint_self_abs_max"] = 0.001
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(gate.ContractError, match="must remain exact"):
        gate.load_and_validate_contract(path)


def test_constant_zero_endpoint_is_a_defined_output_contract():
    raw = np.full((4, 4), 7.5, dtype=np.float32)

    result = _evaluate(raw)

    assert result.passed is True
    assert result.observed["output_contract"]["constant"] is True
    assert result.observed["output_contract"]["min"] == 0.0
    assert result.observed["output_contract"]["max"] == 0.0
