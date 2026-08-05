"""Synthetic tests for umsi_step2b_gate_runner.

All tests use fake models and synthetic arrays only.
No TensorFlow, Keras, saliency.umsi_model, real weights, real fixture image,
or real reference NPZ arrays are imported or accessed anywhere in this file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: minimal contract factory
# ---------------------------------------------------------------------------

def _make_minimal_contract() -> Dict[str, Any]:
    """Return a fully valid minimal contract dict matching the real schema."""
    return {
        "pinned_identities": {
            "expected_repository_head": "abc123",
            "production_source_path": "saliency/umsi_model.py",
            "production_source_sha256_before_inference": "deadbeef",
            "evidence_npz_sha256": "cafebabe",
            "fixture_sha256_lowcontrast": "f00d",
            "weights_sha256": "feedfeed",
        },
        "reference_members": {
            "legacy_raw_saliency": "legacy_raw/lowcontrast_raw",
            "legacy_classification": "legacy_raw/lowcontrast_classif",
            "controlled_modern_raw_saliency": "modern_raw/M_LC_lowcontrast_raw",
            "controlled_modern_classification": "modern_raw/M_LC_lowcontrast_classif",
        },
        "historical_reused_authority": {
            "frozen_thresholds": {
                "pearson_min": 0.999,
                "spearman_min": 0.99,
                "ssim_min": 0.99,
                "cli_abs_max": 0.01,
                "repeatability_abs_max": 0.0,
                "feature_abs_max": 0.02,
                "peak_count_exact": True,
                "preproc_abs_max": 0.0,
            }
        },
        "prospective_new_rule_before_output_observation": {
            "policy": {
                "steps": [
                    "Convert resized numeric map to float64.",
                    "Require all values finite.",
                    "Compute exact minimum, maximum, and span = maximum - minimum.",
                    "Do not use epsilon or observed-data-dependent tolerance.",
                    "If span > 0.0: normalize as (map - minimum) / span.",
                    "If span == 0.0: return all-zero float64 normalized map of identical shape; "
                    "set constant_map_detected=true; fail Gate B with reason "
                    "UNDEFINED_CORRELATION_CONSTANT_MAP; store Pearson, Spearman, SSIM as null.",
                ]
            }
        },
        "gates": {
            "gate_a": {
                "requirements": {
                    "same_shape": True,
                    "same_dtype": True,
                    "finite_values": True,
                    "array_equal": True,
                    "max_abs_diff_float64": {
                        "operator": "==",
                        "value": 0.0,
                        "bound_to_threshold": "repeatability_abs_max",
                    },
                }
            },
            "gate_b": {
                "authoritative_shared_postprocess": {
                    "order": [
                        "squeeze",
                        "identical_resize",
                        "identical_min_max_to_0_1",
                    ],
                    "resize_details": {
                        "legacy_native_operator": "tf.raw_ops.ResizeBilinear(...)",
                        "modern_operator": "keras UpSampling2D(...)",
                        "interpolation": "bilinear",
                    },
                    "normalization_target_range": [0.0, 1.0],
                    "constant_map_policy_ref": "prospective_new_rule_before_output_observation",
                },
                "requirements": {
                    "pearson": {"operator": ">=", "value": 0.999},
                    "spearman": {"operator": ">=", "value": 0.99},
                    "windowed_ssim": {
                        "operator": ">=",
                        "value": 0.99,
                        "params": {
                            "K1": 0.01,
                            "K2": 0.03,
                            "win_size": 7,
                            "boundary": "reflect",
                            "data_range": 1.0,
                            "filter": "scipy.ndimage.uniform_filter",
                        },
                    },
                },
            },
            "gate_c": {
                "requirements": {
                    "same_shape": True,
                    "same_dtype": True,
                    "finite_values": True,
                    "array_equal": True,
                    "max_abs_diff_float64": {
                        "operator": "==",
                        "value": 0.0,
                        "bound_to_threshold": "repeatability_abs_max",
                    },
                }
            },
            "gate_d": {
                "requirements": {
                    "same_shape": True,
                    "finite_values": True,
                    "max_abs_diff_float64": {
                        "operator": "<=",
                        "value": 0.01,
                        "bound_to_threshold": "cli_abs_max",
                    },
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers: fake model classes
# ---------------------------------------------------------------------------

class _FakeInnerModel:
    """Fake Keras-style inner model with a .predict method."""

    def __init__(
        self,
        saliency_batch: np.ndarray,  # shape (1, H, W, 1) — what predict returns
        classif_batch: np.ndarray,   # shape (1, 6)
    ) -> None:
        self._saliency = saliency_batch
        self._classif = classif_batch
        self.call_count = 0

    def predict(self, x: Any, **kwargs: Any) -> list:
        self.call_count += 1
        return [self._saliency, self._classif]


class _FakeUMSIPlus:
    """Fake UMSIPlus that mimics the production class interface.

    The public predict_saliency squeezes/reduces the raw saliency so that
    its return value is visibly different from the raw captured tensor.
    """

    def __init__(self, inner: _FakeInnerModel) -> None:
        self.model = inner  # .model.predict is the intercept point

    def predict_saliency(
        self, image_path: Any, return_classif: bool = False
    ) -> Any:
        # Call self.model.predict (may be proxied by capture_and_invoke)
        preds = self.model.predict(None, verbose=0)
        raw_heatmap = preds[0][0]       # (H, W, 1) — raw
        classif = preds[1][0]           # (6,)
        # "Postprocess": squeeze to (H, W) — different from the (H, W, 1) raw
        heatmap = raw_heatmap[:, :, 0]  # shape (H, W), values may differ after norm
        if return_classif:
            return heatmap, classif
        return heatmap


def _make_fake_model(
    h: int = 8,
    w: int = 8,
    saliency_dtype: str = "float32",
    classif_dtype: str = "float32",
    saliency_fill: float = 0.5,
) -> Tuple[_FakeUMSIPlus, _FakeInnerModel]:
    rng = np.random.default_rng(42)
    sal = rng.random((1, h, w, 1)).astype(saliency_dtype)
    sal = sal * 0.5 + saliency_fill * 0.5  # ensure non-constant, non-zero
    clf = rng.random((1, 6)).astype(classif_dtype)
    inner = _FakeInnerModel(sal, clf)
    return _FakeUMSIPlus(inner), inner


# ---------------------------------------------------------------------------
# Test 1: Importing the runner has no production-model side effect
# ---------------------------------------------------------------------------

def test_import_no_production_model_side_effect() -> None:
    """Importing the runner must not import TF, Keras or saliency.umsi_model.

    This check runs in a fresh subprocess. Checking ``sys.modules`` of the
    running pytest process would give a false failure (or a false pass) if
    another test file collected/executed earlier in the same session --
    e.g. ``test_umsi_boundary_input_probe.py`` -- already imported
    TensorFlow/Keras for unrelated reasons; the shared process would then
    already contain those modules regardless of what this import actually
    does. A fresh child process has none of that pre-existing state: it
    snapshots ``sys.modules`` before the import and inspects only the
    modules newly added by importing
    ``stage1.tools.umsi_step2b_gate_runner`` itself.
    """
    repo_root = Path(__file__).parent.parent
    child_script = (
        "import sys\n"
        "_before = set(sys.modules)\n"
        "import stage1.tools.umsi_step2b_gate_runner\n"
        "_new = set(sys.modules) - _before\n"
        "_forbidden = sorted(\n"
        "    m for m in _new\n"
        "    if 'tensorflow' in m.lower()\n"
        "    or (m.startswith('keras') and 'saliency' not in m)\n"
        "    or m == 'saliency.umsi_model'\n"
        ")\n"
        "if _forbidden:\n"
        "    sys.stderr.write('count=%d first=%s' % (len(_forbidden), _forbidden[:5]))\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Compact failure message only (count + first few names) so pytest never
    # has to difflib-render a potentially huge sys.modules-derived list.
    assert result.returncode == 0, (
        f"Forbidden modules imported by stage1.tools.umsi_step2b_gate_runner "
        f"in a fresh subprocess: {result.stderr.strip()!r}"
    )


# ---------------------------------------------------------------------------
# Test 2a: Real frozen contract schema is accepted
# ---------------------------------------------------------------------------

def test_frozen_contract_schema_accepted() -> None:
    """The actual frozen contract at its known path must pass schema validation."""
    from stage1.tools.umsi_step2b_gate_runner import (
        ContractError,
        load_and_validate_contract,
    )

    contract_path = (
        Path(__file__).parent.parent
        / "stage1"
        / "evidence"
        / "umsi_raw_output_gate_contract.json"
    )
    assert contract_path.is_file(), f"Frozen contract not found at {contract_path}"

    # Must not raise and must return a dict with all four gates
    c = load_and_validate_contract(contract_path)
    assert "gates" in c
    assert {"gate_a", "gate_b", "gate_c", "gate_d"} <= set(c["gates"].keys())


# ---------------------------------------------------------------------------
# Test 2b: Missing required section raises ContractError
# ---------------------------------------------------------------------------

def test_contract_missing_section_raises(tmp_path: Path) -> None:
    from stage1.tools.umsi_step2b_gate_runner import (
        ContractError,
        load_and_validate_contract,
    )

    c = _make_minimal_contract()
    del c["gates"]
    p = tmp_path / "bad_contract.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="gates"):
        load_and_validate_contract(p)


# ---------------------------------------------------------------------------
# Test 3: One fake underlying prediction is captured exactly once
# ---------------------------------------------------------------------------

def test_fake_predict_captured_exactly_once() -> None:
    from stage1.tools.umsi_step2b_gate_runner import capture_and_invoke

    fake_model, inner = _make_fake_model()
    captured, _ = capture_and_invoke(fake_model, "ignored_path.png")

    assert inner.call_count == 1, f"Expected 1 call, got {inner.call_count}"
    assert captured.call_count == 1


# ---------------------------------------------------------------------------
# Test 4: Raw outputs captured before postprocessing (different from public return)
# ---------------------------------------------------------------------------

def test_raw_outputs_captured_before_postprocess() -> None:
    """captured.raw_saliency must be shape (H,W,1); public return is (H,W)."""
    from stage1.tools.umsi_step2b_gate_runner import capture_and_invoke

    fake_model, inner = _make_fake_model(h=8, w=8)
    captured, predict_result = capture_and_invoke(fake_model, "ignored.png")

    heatmap, classif = predict_result

    # raw_saliency is (H,W,1) — captured from preds[0][0] before predict_saliency squeezes
    assert captured.raw_saliency.shape == (8, 8, 1), (
        f"Expected (8,8,1), got {captured.raw_saliency.shape}"
    )
    # public heatmap is (H,W) — different shape from raw
    assert heatmap.shape == (8, 8), f"Expected (8,8), got {heatmap.shape}"
    # values must come from the same underlying data
    np.testing.assert_array_equal(
        captured.raw_saliency[:, :, 0], heatmap
    )


# ---------------------------------------------------------------------------
# Test 5: Original fake predict method is restored after successful call
# ---------------------------------------------------------------------------

def test_predict_restored_after_success() -> None:
    """After capture_and_invoke the runner reports restored=True and the method
    is no longer the proxy (verified by checking the underlying call_count
    increments on a direct call after the harness returns)."""
    from stage1.tools.umsi_step2b_gate_runner import capture_and_invoke

    fake_model, inner = _make_fake_model()

    captured, _ = capture_and_invoke(fake_model, "ignored.png")

    assert captured.restored is True, "runner must report restored=True"
    assert inner.call_count == 1, "exactly one call during capture"

    # Call the method directly; it must reach the real predict (not the proxy)
    count_before = inner.call_count
    inner.predict(None, verbose=0)
    assert inner.call_count == count_before + 1, (
        "after restoration a direct call must reach the original predict"
    )


# ---------------------------------------------------------------------------
# Test 6: A second attempted call is rejected BEFORE executing
# ---------------------------------------------------------------------------

def test_second_call_rejected_before_executing() -> None:
    """Proxy must raise InferenceCallError on a second call without delegating."""
    from stage1.tools.umsi_step2b_gate_runner import (
        InferenceCallError,
        capture_and_invoke,
    )

    class _TwoCallFakeInner:
        def __init__(self) -> None:
            self.call_count = 0
            sal = np.zeros((1, 4, 4, 1), dtype=np.float32)
            clf = np.zeros((1, 6), dtype=np.float32)
            self._sal = sal
            self._clf = clf

        def predict(self, x: Any, **kwargs: Any) -> list:
            self.call_count += 1
            return [self._sal, self._clf]

    class _TwoCallFakeModel:
        def __init__(self) -> None:
            self.model = _TwoCallFakeInner()

        def predict_saliency(self, path: Any, return_classif: bool = False) -> Any:
            # First call
            preds1 = self.model.predict(None, verbose=0)
            # Second call — proxy must reject this
            preds2 = self.model.predict(None, verbose=0)
            heatmap = preds1[0][0][:, :, 0]
            classif = preds2[1][0]
            if return_classif:
                return heatmap, classif
            return heatmap

    m = _TwoCallFakeModel()
    inner = m.model

    with pytest.raises(InferenceCallError, match="attempted call #2"):
        capture_and_invoke(m, "ignored.png")

    # The proxy must be gone: a direct call must reach the original predict
    count_before = inner.call_count
    inner.predict(None, verbose=0)
    assert inner.call_count == count_before + 1, (
        "after rejection the proxy must be removed; direct call must reach original"
    )
    # The second underlying call must NOT have been delegated
    assert inner.call_count == 2, (
        f"Total underlying calls: 1 (captured) + 1 (post-restore check); got {inner.call_count}"
    )


# ---------------------------------------------------------------------------
# Test 7: Exception in predict causes no retry and still restores method
# ---------------------------------------------------------------------------

def test_exception_restores_method_no_retry() -> None:
    from stage1.tools.umsi_step2b_gate_runner import capture_and_invoke

    class _RaisingInner:
        def __init__(self) -> None:
            self.call_count = 0

        def predict(self, x: Any, **kwargs: Any) -> Any:
            self.call_count += 1
            raise RuntimeError("simulated model failure")

    class _RaisingModel:
        def __init__(self) -> None:
            self.model = _RaisingInner()

        def predict_saliency(self, path: Any, return_classif: bool = False) -> Any:
            return self.model.predict(None, verbose=0)

    m = _RaisingModel()

    with pytest.raises(RuntimeError, match="simulated model failure"):
        capture_and_invoke(m, "ignored.png")

    assert m.model.call_count == 1, "no retry: underlying predict must be called exactly once"
    # Verify the proxy is gone: calling again raises the real RuntimeError (not InferenceCallError)
    with pytest.raises(RuntimeError, match="simulated model failure"):
        m.model.predict(None, verbose=0)
    assert m.model.call_count == 2, "second direct call must reach original predict"


# ---------------------------------------------------------------------------
# Test 8: Identical nonconstant synthetic arrays pass appropriate gates
# ---------------------------------------------------------------------------

def test_identical_nonconstant_arrays_pass_gates() -> None:
    from stage1.tools.umsi_step2b_gate_runner import (
        evaluate_gate_a,
        evaluate_gate_b,
        evaluate_gate_c,
        evaluate_gate_d,
    )

    c = _make_minimal_contract()
    rng = np.random.default_rng(7)

    # Gate A: (8,8) float32 identical arrays, passed as (8,8,1) prod and (8,8) ref
    sal = rng.random((8, 8)).astype(np.float32) * 0.8 + 0.1
    prod_raw = sal[:, :, np.newaxis]  # (8,8,1)
    ref_sal = sal.copy()              # (8,8)

    ga = evaluate_gate_a(prod_raw, ref_sal, c["gates"]["gate_a"])
    assert ga.passed, f"Gate A should pass: {ga.reason}"

    # Gate B: identical non-constant arrays
    gb = evaluate_gate_b(prod_raw, ref_sal, c["gates"]["gate_b"])
    assert gb.passed, f"Gate B should pass: {gb.reason}"
    assert gb.observed.get("pearson", 0.0) > 0.999
    assert gb.observed.get("spearman", 0.0) > 0.99
    assert gb.observed.get("windowed_ssim", 0.0) > 0.99

    # Gate C: (6,) float32 identical arrays
    clf = rng.random(6).astype(np.float32)
    gc = evaluate_gate_c(clf, clf.copy(), c["gates"]["gate_c"])
    assert gc.passed, f"Gate C should pass: {gc.reason}"

    # Gate D: identical classif → max_abs_diff=0.0 ≤ 0.01
    gd = evaluate_gate_d(clf, clf.copy(), c["gates"]["gate_d"])
    assert gd.passed, f"Gate D should pass: {gd.reason}"


# ---------------------------------------------------------------------------
# Test 9: Gate B constant map fails with contract-defined reason
# ---------------------------------------------------------------------------

def test_gate_b_constant_map_fails() -> None:
    from stage1.tools.umsi_step2b_gate_runner import (
        _CONSTANT_MAP_FAIL_REASON,
        evaluate_gate_b,
    )

    c = _make_minimal_contract()

    # Production array is constant; reference is not
    constant_prod = np.full((8, 8, 1), 0.5, dtype=np.float32)
    nonconstant_ref = np.linspace(0.1, 0.9, 64).reshape(8, 8).astype(np.float32)

    result = evaluate_gate_b(constant_prod, nonconstant_ref, c["gates"]["gate_b"])

    assert not result.passed
    assert result.reason == _CONSTANT_MAP_FAIL_REASON, (
        f"Expected reason {_CONSTANT_MAP_FAIL_REASON!r}, got {result.reason!r}"
    )
    assert result.observed.get("pearson") is None
    assert result.observed.get("spearman") is None
    assert result.observed.get("windowed_ssim") is None
    assert result.observed.get("prod_constant_map") is True

    # Also check reference-constant case
    nonconstant_prod = np.linspace(0.1, 0.9, 64).reshape(8, 8, 1).astype(np.float32)
    constant_ref = np.full((8, 8), 0.3, dtype=np.float32)

    result2 = evaluate_gate_b(nonconstant_prod, constant_ref, c["gates"]["gate_b"])
    assert not result2.passed
    assert result2.reason == _CONSTANT_MAP_FAIL_REASON
    assert result2.observed.get("ref_constant_map") is True


# ---------------------------------------------------------------------------
# Test 10: Missing required contract key fails closed
# ---------------------------------------------------------------------------

def test_missing_contract_key_fails_closed(tmp_path: Path) -> None:
    from stage1.tools.umsi_step2b_gate_runner import (
        ContractError,
        load_and_validate_contract,
    )

    # Remove a required member key
    c = _make_minimal_contract()
    del c["reference_members"]["legacy_raw_saliency"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="reference_members"):
        load_and_validate_contract(p)

    # Remove SSIM params
    c2 = _make_minimal_contract()
    del c2["gates"]["gate_b"]["requirements"]["windowed_ssim"]["params"]["K1"]
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps(c2))
    with pytest.raises(ContractError, match="K1"):
        load_and_validate_contract(p2)


# ---------------------------------------------------------------------------
# Test 11: Non-finite values fail closed
# ---------------------------------------------------------------------------

def test_non_finite_values_fail_closed() -> None:
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_b

    c = _make_minimal_contract()

    normal = np.linspace(0.1, 0.9, 64).reshape(8, 8).astype(np.float32)
    nan_arr = normal.copy()
    nan_arr[0, 0] = float("nan")

    # Non-finite in production array
    result = evaluate_gate_b(nan_arr[:, :, np.newaxis], normal, c["gates"]["gate_b"])
    assert not result.passed
    assert "NON_FINITE" in result.reason or "POSTPROCESS_ERROR" in result.reason

    # Non-finite in reference array
    result2 = evaluate_gate_b(
        normal[:, :, np.newaxis], nan_arr, c["gates"]["gate_b"]
    )
    assert not result2.passed
    assert "NON_FINITE" in result2.reason or "POSTPROCESS_ERROR" in result2.reason


# ---------------------------------------------------------------------------
# Test 12: Shape mismatches fail closed (Gate A)
# ---------------------------------------------------------------------------

def test_shape_mismatch_fails_gate_a() -> None:
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_a

    c = _make_minimal_contract()
    prod = np.ones((8, 8, 1), dtype=np.float32)
    ref_wrong_shape = np.ones((6, 8), dtype=np.float32)

    result = evaluate_gate_a(prod, ref_wrong_shape, c["gates"]["gate_a"])
    assert not result.passed
    assert "SHAPE_MISMATCH" in result.reason


# ---------------------------------------------------------------------------
# Test 13: Dtype mismatches fail closed (Gate C)
# ---------------------------------------------------------------------------

def test_dtype_mismatch_fails_gate_c() -> None:
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_c

    c = _make_minimal_contract()
    clf_f32 = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1], dtype=np.float32)
    clf_f64 = clf_f32.astype(np.float64)

    result = evaluate_gate_c(clf_f32, clf_f64, c["gates"]["gate_c"])
    assert not result.passed
    assert "DTYPE_MISMATCH" in result.reason


# ---------------------------------------------------------------------------
# Test 14a: Gate D passes exactly at the frozen boundary (cli_abs_max = 0.01)
# ---------------------------------------------------------------------------

def test_gate_d_passes_at_exact_boundary() -> None:
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_d

    c = _make_minimal_contract()
    base = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1], dtype=np.float64)
    # Exactly at limit: one element differs by exactly 0.01
    delta = np.zeros(6, dtype=np.float64)
    delta[0] = 0.01
    prod = base + delta

    result = evaluate_gate_d(prod, base, c["gates"]["gate_d"])
    assert result.passed, f"Gate D should pass at boundary: {result.reason}"
    assert abs(result.observed["max_abs_diff_float64"] - 0.01) < 1e-14


# ---------------------------------------------------------------------------
# Test 14b: Gate D fails immediately above the frozen boundary
# ---------------------------------------------------------------------------

def test_gate_d_fails_above_boundary() -> None:
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_d

    c = _make_minimal_contract()
    base = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1], dtype=np.float64)
    delta = np.zeros(6, dtype=np.float64)
    delta[0] = 0.010001  # just above 0.01
    prod = base + delta

    result = evaluate_gate_d(prod, base, c["gates"]["gate_d"])
    assert not result.passed
    assert "GATE_D_CLI_ABS_MAX_FAIL" in result.reason
    # Confirm the label distinguishes Step 2B classif tolerance from Stage-2
    assert "Stage-2" not in result.reason or "NOT" in result.reason
    assert "cli_abs_max" in result.observed["threshold_label"]
    assert "Stage-2" not in result.observed["threshold_label"].split("cli_abs_max")[0]


# ---------------------------------------------------------------------------
# Test 15: Evidence bundle contains both required files
# ---------------------------------------------------------------------------

def test_evidence_bundle_contains_both_files(tmp_path: Path) -> None:
    from stage1.tools.umsi_step2b_gate_runner import write_evidence_bundle

    rng = np.random.default_rng(0)
    raw_sal = rng.random((4, 4, 1)).astype(np.float32)
    classif = rng.random(6).astype(np.float32)
    report = {"conclusion": "TEST", "utc_timestamp": "2026-08-03T00:00:00Z"}

    bundle_path = write_evidence_bundle(
        output_dir=tmp_path,
        bundle_name="test_bundle_001",
        report=report,
        raw_saliency=raw_sal,
        classif=classif,
    )

    assert bundle_path.is_dir(), "Bundle must be a directory"
    npz_file = bundle_path / "production_outputs.npz"
    report_file = bundle_path / "gate_report.json"

    assert npz_file.is_file(), "production_outputs.npz must exist"
    assert report_file.is_file(), "gate_report.json must exist"

    # Verify NPZ contents
    with np.load(str(npz_file)) as npz_data:
        np.testing.assert_array_equal(
            npz_data["production_raw_saliency"], raw_sal
        )
        np.testing.assert_array_equal(
            npz_data["production_classif"], classif
        )

    # Verify report JSON
    loaded = json.loads(report_file.read_text())
    assert loaded["conclusion"] == "TEST"


# ---------------------------------------------------------------------------
# Test 16: Existing target bundle is never overwritten
# ---------------------------------------------------------------------------

def test_evidence_bundle_not_overwritten(tmp_path: Path) -> None:
    from stage1.tools.umsi_step2b_gate_runner import write_evidence_bundle

    rng = np.random.default_rng(1)
    raw_sal = rng.random((4, 4, 1)).astype(np.float32)
    classif = rng.random(6).astype(np.float32)
    report = {"conclusion": "FIRST"}

    # Create the bundle once
    write_evidence_bundle(
        output_dir=tmp_path,
        bundle_name="no_overwrite_test",
        report=report,
        raw_saliency=raw_sal,
        classif=classif,
    )

    # Second write with the same name must raise FileExistsError
    with pytest.raises(FileExistsError, match="no_overwrite_test"):
        write_evidence_bundle(
            output_dir=tmp_path,
            bundle_name="no_overwrite_test",
            report={"conclusion": "SECOND"},
            raw_saliency=raw_sal,
            classif=classif,
        )

    # Original report must be intact
    report_file = tmp_path / "no_overwrite_test" / "gate_report.json"
    loaded = json.loads(report_file.read_text())
    assert loaded["conclusion"] == "FIRST", "Original report must not be overwritten"


# ---------------------------------------------------------------------------
# Test 17: Post-inference gate failure still produces a fail-closed bundle
# ---------------------------------------------------------------------------

def test_post_inference_gate_failure_produces_fail_closed_bundle(tmp_path: Path) -> None:
    """If gate evaluation fails after capture, the bundle is still written and
    reports the failure explicitly."""
    from stage1.tools.umsi_step2b_gate_runner import (
        CapturedOutputs,
        GateResult,
        evaluate_gate_d,
        write_evidence_bundle,
    )

    rng = np.random.default_rng(2)
    raw_sal = rng.random((4, 4, 1)).astype(np.float32)
    classif = rng.random(6).astype(np.float64)

    c = _make_minimal_contract()

    # Deliberately cause Gate D to fail: delta > 0.01
    ref_classif = classif.copy()
    ref_classif[0] += 0.1  # 0.1 > 0.01 → Gate D fails

    gate_d_result = evaluate_gate_d(classif, ref_classif, c["gates"]["gate_d"])
    assert not gate_d_result.passed, "Precondition: Gate D must fail in this test"

    # Simulate what run_frozen_experiment does: write bundle even on gate failure
    all_passed = False
    conclusion = "STEP_2B_FAIL_GATE_D"
    report: Dict[str, Any] = {
        "conclusion": conclusion,
        "utc_timestamp": "2026-08-03T00:00:00Z",
        "gate_results": [
            {
                "name": gate_d_result.name,
                "passed": gate_d_result.passed,
                "reason": gate_d_result.reason,
                "observed": gate_d_result.observed,
            }
        ],
    }

    bundle_path = write_evidence_bundle(
        output_dir=tmp_path,
        bundle_name="fail_closed_bundle",
        report=report,
        raw_saliency=raw_sal,
        classif=classif,
    )

    assert bundle_path.is_dir()
    loaded = json.loads((bundle_path / "gate_report.json").read_text())
    assert loaded["conclusion"] == "STEP_2B_FAIL_GATE_D"
    assert not loaded["gate_results"][0]["passed"]
    assert loaded["gate_results"][0]["reason"] is not None
    assert (bundle_path / "production_outputs.npz").is_file()


# ===========================================================================
# D1 — Runner hash preflight tests
# ===========================================================================

# The SHA-256 of the *current* repaired runner (computed at test-authoring time).
# This value is used ONLY to test that the preflight accepts a correct hash;
# it is not hardcoded as a normative frozen value for production runs.
_CURRENT_RUNNER_SHA = "df356064b4d20f5ba652f16348cd3ef295c66528b2539afafd8309a8f77302fa"

# A syntactically valid SHA-256 that does NOT match the runner on disk.
_WRONG_RUNNER_SHA = "a" * 64


def _fake_preflight_deps(tmp_path: Path) -> Dict[str, Any]:
    """Build minimal fake filesystem objects to satisfy run_preflight.

    Returns a dict of keyword args for run_preflight that use tiny stub files
    so all SHA checks except the runner hash can pass.  The stub SHAs are
    pre-computed from the stub content and passed as expected values.
    """
    import hashlib as _hl
    from stage1.tools.umsi_step2b_gate_runner import run_preflight  # noqa

    def make_file(name: str, content: bytes) -> Tuple[Path, str]:
        p = tmp_path / name
        p.write_bytes(content)
        sha = _hl.sha256(content).hexdigest()
        return p, sha

    contract_p, contract_sha = make_file("contract.json", b'{"stub":1}')
    source_p, source_sha = make_file("source.py", b"# stub source")
    weights_p, weights_sha = make_file("weights.hdf5", b"\x00weights")
    npz_p, npz_sha = make_file("ref.npz", b"\x00npz")
    fixture_p, fixture_sha = make_file("fixture.png", b"\x00png")

    return dict(
        repo_path=tmp_path,
        expected_head="FAKEHEAD",
        contract_path=contract_p,
        expected_contract_sha=contract_sha,
        source_path=source_p,
        expected_source_sha=source_sha,
        weights_path=weights_p,
        expected_weights_sha=weights_sha,
        npz_path=npz_p,
        expected_npz_sha=npz_sha,
        fixture_path=fixture_p,
        expected_fixture_sha=fixture_sha,
    )


# ---------------------------------------------------------------------------
# Test D1-1: Correct expected runner SHA passes the runner-hash preflight check
# ---------------------------------------------------------------------------

def test_d1_correct_runner_sha_passes(tmp_path: Path) -> None:
    """run_preflight accepts the correct SHA-256 of the executing runner."""
    from stage1.tools.umsi_step2b_gate_runner import PreflightError, run_preflight

    deps = _fake_preflight_deps(tmp_path)

    # All file checks will pass with fake stubs; HEAD check will fail because
    # no real git repo is in tmp_path — that is expected here.  We only care
    # that the runner hash check itself does NOT raise.
    # Strategy: patch git HEAD to return the expected value by using a real
    # git repo path (the runner's own repo) and matching HEAD.
    import subprocess
    repo = Path(__file__).parent.parent
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode().strip()

    deps["repo_path"] = repo
    deps["expected_head"] = head

    # The runner-hash check runs FIRST — it must not raise for the correct SHA.
    # The subsequent file-SHA checks will fail (stubs don't match real files),
    # so we expect a PreflightError about something other than the runner hash.
    try:
        run_preflight(**deps, expected_runner_sha=_CURRENT_RUNNER_SHA)
    except PreflightError as exc:
        assert "Runner SHA" not in str(exc), (
            f"Runner SHA check should pass but got: {exc}"
        )
    # No InferenceCallError or other exception must have been raised.


# ---------------------------------------------------------------------------
# Test D1-2: Incorrect runner SHA raises PreflightError before other checks
# ---------------------------------------------------------------------------

def test_d1_wrong_runner_sha_fails(tmp_path: Path) -> None:
    """run_preflight raises PreflightError immediately for a wrong runner SHA."""
    from stage1.tools.umsi_step2b_gate_runner import PreflightError, run_preflight

    deps = _fake_preflight_deps(tmp_path)
    # Even with valid fake stubs, the wrong hash must fail before git HEAD check.
    with pytest.raises(PreflightError, match="Runner SHA-256 mismatch"):
        run_preflight(**deps, expected_runner_sha=_WRONG_RUNNER_SHA)


# ---------------------------------------------------------------------------
# Test D1-3: Malformed expected runner SHA is rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sha", [
    None,
    "",
    "abc",
    "G" * 64,    # uppercase hex
    "a" * 63,   # too short
    "a" * 65,   # too long
    "  " + "a" * 62,  # leading spaces
])
def test_d1_malformed_runner_sha_rejected(tmp_path: Path, bad_sha: Any) -> None:
    """run_preflight rejects malformed or None expected_runner_sha."""
    from stage1.tools.umsi_step2b_gate_runner import PreflightError, run_preflight

    deps = _fake_preflight_deps(tmp_path)
    with pytest.raises(PreflightError, match="--expected-runner-sha"):
        run_preflight(**deps, expected_runner_sha=bad_sha)


# ---------------------------------------------------------------------------
# Test D1-4: Runner-hash rejection happens before any production import
# ---------------------------------------------------------------------------

def test_d1_rejection_before_production_import(tmp_path: Path) -> None:
    """Wrong runner SHA causes PreflightError; saliency.umsi_model must stay absent."""
    from stage1.tools.umsi_step2b_gate_runner import PreflightError, run_preflight
    import sys

    deps = _fake_preflight_deps(tmp_path)

    modules_before = set(sys.modules.keys())
    with pytest.raises(PreflightError):
        run_preflight(**deps, expected_runner_sha=_WRONG_RUNNER_SHA)
    modules_after = set(sys.modules.keys())

    new_modules = modules_after - modules_before
    forbidden = [m for m in new_modules if "tensorflow" in m.lower()
                 or "keras" in m.lower() or "saliency" in m.lower()]
    assert forbidden == [], f"Forbidden modules loaded: {forbidden}"


# ---------------------------------------------------------------------------
# Test D1-5: Observed runner hash cannot be the caller's own supplied value
# ---------------------------------------------------------------------------

def test_d1_observed_hash_is_independent(tmp_path: Path) -> None:
    """The observed runner SHA is calculated from __file__, not echoed from the arg.

    Verify that supplying a WRONG expected SHA causes a mismatch error that
    includes BOTH the expected AND observed values in the message, proving
    the implementation calculated its own hash independently.
    """
    from stage1.tools.umsi_step2b_gate_runner import PreflightError, run_preflight

    deps = _fake_preflight_deps(tmp_path)
    with pytest.raises(PreflightError) as exc_info:
        run_preflight(**deps, expected_runner_sha=_WRONG_RUNNER_SHA)

    msg = str(exc_info.value)
    # Message must contain BOTH values
    assert _WRONG_RUNNER_SHA in msg, "Expected hash should appear in mismatch message"
    assert _CURRENT_RUNNER_SHA in msg, (
        "Observed hash (independently calculated) should appear in mismatch message"
    )


# ---------------------------------------------------------------------------
# Test D1-6: Report distinguishes expected and observed runner hashes
# ---------------------------------------------------------------------------

def test_d1_report_distinguishes_hashes(tmp_path: Path) -> None:
    """write_evidence_bundle can store expected/observed/match runner hash fields."""
    from stage1.tools.umsi_step2b_gate_runner import write_evidence_bundle

    rng = np.random.default_rng(99)
    raw_sal = rng.random((4, 4, 1)).astype(np.float32)
    classif = rng.random(6).astype(np.float32)

    report = {
        "conclusion": "TEST",
        "utc_timestamp": "2026-08-03T00:00:00Z",
        "identities": {
            "expected_runner_sha256": _WRONG_RUNNER_SHA,
            "observed_runner_sha256": _CURRENT_RUNNER_SHA,
            "runner_sha256_match": (_WRONG_RUNNER_SHA == _CURRENT_RUNNER_SHA),
        },
    }

    bundle_path = write_evidence_bundle(
        output_dir=tmp_path,
        bundle_name="runner_hash_report_test",
        report=report,
        raw_saliency=raw_sal,
        classif=classif,
    )

    loaded = json.loads((bundle_path / "gate_report.json").read_text())
    ids = loaded["identities"]
    assert ids["expected_runner_sha256"] == _WRONG_RUNNER_SHA
    assert ids["observed_runner_sha256"] == _CURRENT_RUNNER_SHA
    assert ids["runner_sha256_match"] is False


# ===========================================================================
# D2 — Safe evaluator boundary tests
# ===========================================================================

def _make_raising_evaluator(exc_type: type, msg: str) -> Any:
    """Return a callable that raises exc_type(msg) unconditionally."""
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise exc_type(msg)
    return _raise


# ---------------------------------------------------------------------------
# Test D2-1: Exception from Gate C becomes a failed GateResult
# ---------------------------------------------------------------------------

def test_d2_gate_c_exception_becomes_failed_result() -> None:
    from stage1.tools.umsi_step2b_gate_runner import safe_evaluate_gate

    result = safe_evaluate_gate(
        "gate_c",
        _make_raising_evaluator(RuntimeError, "gate_c boom"),
        lambda: (np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32), {}),
    )
    assert not result.passed
    assert result.reason == "EVALUATOR_EXCEPTION"
    assert result.observed["evaluator_exception_type"] == "RuntimeError"
    assert "gate_c boom" in result.observed["evaluator_exception_message"]
    assert result.name == "gate_c"


# ---------------------------------------------------------------------------
# Test D2-2: Exception from Gate D becomes a failed GateResult
# ---------------------------------------------------------------------------

def test_d2_gate_d_exception_becomes_failed_result() -> None:
    from stage1.tools.umsi_step2b_gate_runner import safe_evaluate_gate

    result = safe_evaluate_gate(
        "gate_d",
        _make_raising_evaluator(KeyError, "missing_key"),
        lambda: (np.zeros(6, dtype=np.float64), np.zeros(6, dtype=np.float64), {}),
    )
    assert not result.passed
    assert result.reason == "EVALUATOR_EXCEPTION"
    assert result.observed["evaluator_exception_type"] == "KeyError"
    assert result.name == "gate_d"


# ---------------------------------------------------------------------------
# Test D2-3: Other gates remain represented (complete four-gate result set)
# ---------------------------------------------------------------------------

def test_d2_all_four_gates_represented_on_partial_failure() -> None:
    """When one gate evaluator raises, the other three still produce results."""
    from stage1.tools.umsi_step2b_gate_runner import (
        GateResult,
        evaluate_gate_a,
        evaluate_gate_b,
        evaluate_gate_c,
        evaluate_gate_d,
        safe_evaluate_gate,
    )

    c = _make_minimal_contract()
    rng = np.random.default_rng(5)
    sal = rng.random((8, 8)).astype(np.float32) * 0.8 + 0.1
    prod_raw = sal[:, :, np.newaxis]
    ref_sal = sal.copy()
    clf = rng.random(6).astype(np.float32)

    results = [
        safe_evaluate_gate("gate_a", evaluate_gate_a,
                           lambda: (prod_raw, ref_sal, c["gates"]["gate_a"])),
        safe_evaluate_gate("gate_b", evaluate_gate_b,
                           lambda: (prod_raw, ref_sal, c["gates"]["gate_b"])),
        safe_evaluate_gate("gate_c",
                           _make_raising_evaluator(ValueError, "synth_failure"),
                           lambda: (clf, clf.copy(), c["gates"]["gate_c"])),
        safe_evaluate_gate("gate_d", evaluate_gate_d,
                           lambda: (clf, clf.copy(), c["gates"]["gate_d"])),
    ]

    assert len(results) == 4
    names = [r.name for r in results]
    assert names == ["gate_a", "gate_b", "gate_c", "gate_d"]
    assert results[0].passed, f"gate_a: {results[0].reason}"
    assert results[1].passed, f"gate_b: {results[1].reason}"
    assert not results[2].passed  # raised
    assert results[2].reason == "EVALUATOR_EXCEPTION"
    assert results[3].passed, f"gate_d: {results[3].reason}"


# ---------------------------------------------------------------------------
# Test D2-4: Evaluator exception forces overall result to fail
# ---------------------------------------------------------------------------

def test_d2_evaluator_exception_forces_overall_fail() -> None:
    from stage1.tools.umsi_step2b_gate_runner import safe_evaluate_gate

    results = [
        safe_evaluate_gate("gate_a",
                           _make_raising_evaluator(Exception, "exc"),
                           lambda: (None, None, {})),
        safe_evaluate_gate("gate_b",
                           _make_raising_evaluator(Exception, "exc"),
                           lambda: (None, None, {})),
        safe_evaluate_gate("gate_c",
                           _make_raising_evaluator(Exception, "exc"),
                           lambda: (None, None, {})),
        safe_evaluate_gate("gate_d",
                           _make_raising_evaluator(Exception, "exc"),
                           lambda: (None, None, {})),
    ]
    all_passed = all(r.passed for r in results)
    assert not all_passed, "Overall must fail if any evaluator raised"


# ---------------------------------------------------------------------------
# Test D2-5 to D2-9: After fake inference, evaluator exception → both evidence files
# ---------------------------------------------------------------------------

def test_d2_evaluator_exception_still_produces_bundle(tmp_path: Path) -> None:
    """After one fake inference, a gate evaluator exception still produces both
    evidence files and the NPZ contains the captured native arrays unchanged."""
    from stage1.tools.umsi_step2b_gate_runner import (
        CapturedOutputs,
        GateResult,
        safe_evaluate_gate,
        write_evidence_bundle,
    )

    c = _make_minimal_contract()

    # Simulate captured outputs from exactly one inference
    rng = np.random.default_rng(77)
    raw_sal = rng.random((4, 4, 1)).astype(np.float32)
    classif = rng.random(6).astype(np.float32)
    inner_call_count = 1  # recorded value; no real model was used

    # Gate C raises, Gate D is normal
    gate_d_result = GateResult(name="gate_d", passed=True, observed={"max_abs_diff_float64": 0.0})

    gate_results = [
        safe_evaluate_gate("gate_a",
                           _make_raising_evaluator(RuntimeError, "gate_a exc"),
                           lambda: (None, None, {})),
        safe_evaluate_gate("gate_b",
                           _make_raising_evaluator(ValueError, "gate_b exc"),
                           lambda: (None, None, {})),
        safe_evaluate_gate("gate_c",
                           _make_raising_evaluator(TypeError, "gate_c exc"),
                           lambda: (None, None, {})),
        gate_d_result,
    ]

    all_passed = all(r.passed for r in gate_results)
    assert not all_passed

    failed_names = [r.name.upper() for r in gate_results if not r.passed]
    conclusion = "STEP_2B_FAIL_" + "_".join(failed_names)

    report: Dict[str, Any] = {
        "conclusion": conclusion,
        "utc_timestamp": "2026-08-03T12:00:00Z",
        "inference": {
            "real_inference_call_count": inner_call_count,
        },
        "gate_results": [
            {
                "name": r.name,
                "passed": r.passed,
                "reason": r.reason,
                "observed": r.observed,
            }
            for r in gate_results
        ],
    }

    # write_evidence_bundle must succeed and produce both files
    bundle_path = write_evidence_bundle(
        output_dir=tmp_path,
        bundle_name="d2_exc_bundle",
        report=report,
        raw_saliency=raw_sal,
        classif=classif,
    )

    # Both files present
    assert (bundle_path / "production_outputs.npz").is_file(), "NPZ must be present"
    assert (bundle_path / "gate_report.json").is_file(), "Report must be present"

    # NPZ contains captured arrays unchanged
    with np.load(str(bundle_path / "production_outputs.npz")) as npz_data:
        np.testing.assert_array_equal(npz_data["production_raw_saliency"], raw_sal)
        np.testing.assert_array_equal(npz_data["production_classif"], classif)

    # Report records evaluator exception type and message
    loaded = json.loads((bundle_path / "gate_report.json").read_text())
    assert loaded["conclusion"].startswith("STEP_2B_FAIL_")
    gate_c_report = next(g for g in loaded["gate_results"] if g["name"] == "gate_c")
    assert gate_c_report["reason"] == "EVALUATOR_EXCEPTION"
    assert gate_c_report["observed"]["evaluator_exception_type"] == "TypeError"
    assert "gate_c exc" in gate_c_report["observed"]["evaluator_exception_message"]

    # Inference count is recorded as exactly one
    assert loaded["inference"]["real_inference_call_count"] == 1


def test_d2_evaluator_exception_no_retry(tmp_path: Path) -> None:
    """The underlying fake predict count must remain exactly 1 even when a gate
    evaluator raises, confirming no retry occurs."""
    from stage1.tools.umsi_step2b_gate_runner import capture_and_invoke, safe_evaluate_gate

    fake_model, inner = _make_fake_model(h=4, w=4)
    captured, _ = capture_and_invoke(fake_model, "ignored.png")

    assert inner.call_count == 1, "Inference must happen exactly once"

    # Now trigger evaluator exceptions
    for gate_name in ("gate_a", "gate_b", "gate_c", "gate_d"):
        safe_evaluate_gate(gate_name,
                           _make_raising_evaluator(RuntimeError, "no retry"),
                           lambda: (None, None, {}))

    # Still exactly one inference — safe_evaluate_gate must not call the model
    assert inner.call_count == 1, (
        f"Evaluator exceptions must not trigger re-inference; "
        f"inner.call_count={inner.call_count}"
    )


# ===========================================================================
# P6 — Deferred argument factory + frozen_thresholds + shared helper tests
# ===========================================================================

def _fake_observed_shas() -> Dict[str, str]:
    return {k: "a" * 64 for k in ("runner", "source", "contract", "weights", "npz", "fixture")}


# ---------------------------------------------------------------------------
# P6-1 / P6-2: Missing gate_a key caught inside safe boundary
# ---------------------------------------------------------------------------

def test_p6_missing_gate_def_caught_inside_boundary() -> None:
    """KeyError from a missing gate definition is caught inside safe_evaluate_gate.

    With the deferred argument_factory interface, contract["gates"]["gate_a"]
    is evaluated INSIDE the try block and cannot escape as an unhandled exception.
    """
    from stage1.tools.umsi_step2b_gate_runner import evaluate_gate_a, safe_evaluate_gate

    contract = _make_minimal_contract()
    del contract["gates"]["gate_a"]   # force KeyError in lambda

    rng = np.random.default_rng(1)
    sal = rng.random((4, 4, 1)).astype(np.float32)
    ref = rng.random((4, 4)).astype(np.float32)

    result = safe_evaluate_gate(
        "gate_a",
        evaluate_gate_a,
        lambda: (sal, ref, contract["gates"]["gate_a"]),
    )

    # Must produce a failed GateResult — not propagate the KeyError
    assert not result.passed
    assert result.reason == "EVALUATOR_EXCEPTION"
    assert result.name == "gate_a"
    assert result.observed["evaluator_exception_type"] == "KeyError"
    assert result.observed["evaluator_exception_message"] != ""


# ---------------------------------------------------------------------------
# P6-3 / P6-4 / P6-5: Missing gate_a leaves other gates intact via shared helper
# ---------------------------------------------------------------------------

def test_p6_missing_gate_def_leaves_other_gates_intact(tmp_path: Path) -> None:
    """When gate_a is absent from contract["gates"], the shared helper still
    evaluates gates B, C and D and writes evidence — four results always present."""
    from stage1.tools.umsi_step2b_gate_runner import (
        CapturedOutputs,
        _run_post_capture_sequence,
    )

    contract = _make_minimal_contract()
    del contract["gates"]["gate_a"]   # gate_a will raise KeyError inside boundary

    rng = np.random.default_rng(2)
    sal_3d = rng.random((4, 4, 1)).astype(np.float32)
    ref_sal = rng.random((4, 4)).astype(np.float32)
    ref_clf = rng.random(6).astype(np.float32)

    captured = CapturedOutputs(
        raw_saliency=sal_3d,
        classif=ref_clf.copy(),
        call_count=1,
        restored=True,
    )

    from stage1.tools.umsi_step2b_gate_runner import _build_report_statics
    _rt, _ris = _build_report_statics(
        frozen_thresholds=contract["historical_reused_authority"]["frozen_thresholds"],
        pinned=contract["pinned_identities"],
        observed_shas=_fake_observed_shas(),
        module_file_str="fake_module.py",
        expected_runner_sha="a" * 64,
    )
    result = _run_post_capture_sequence(
        captured=captured,
        ref_legacy_raw=ref_sal,
        ref_legacy_classif=ref_clf,
        ref_m_lc_raw=ref_sal.copy(),
        ref_m_lc_classif=ref_clf.copy(),
        contract=contract,
        report_thresholds=_rt,
        report_identities_static=_ris,
        output_dir=tmp_path,
    )

    # Must fail (non-zero) because gate_a raised
    assert result != 0

    # Evidence bundle must be written
    bundles = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert (bundle / "gate_report.json").is_file()
    assert (bundle / "production_outputs.npz").is_file()

    loaded = json.loads((bundle / "gate_report.json").read_text())
    gate_names = [g["name"] for g in loaded["gate_results"]]

    # All four gates must be present
    assert gate_names == ["gate_a", "gate_b", "gate_c", "gate_d"], gate_names

    # gate_a must have exception result
    ga = next(g for g in loaded["gate_results"] if g["name"] == "gate_a")
    assert not ga["passed"]
    assert ga["reason"] == "EVALUATOR_EXCEPTION"
    assert ga["observed"]["evaluator_exception_type"] == "KeyError"

    # Gates B, C, D must also be present (even if their values fail the thresholds)
    for name in ("gate_b", "gate_c", "gate_d"):
        gr = next(g for g in loaded["gate_results"] if g["name"] == name)
        assert gr["name"] == name

    # Overall must be fail because gate_a raised
    assert not loaded["conclusion"].startswith("STEP_2B_PASS")


# ---------------------------------------------------------------------------
# P6-8 supplement: invalid return type and wrong gate name remain fail-closed
# ---------------------------------------------------------------------------

def test_p6_invalid_return_type_and_wrong_gate_name_fail_closed() -> None:
    """safe_evaluate_gate is fail-closed for invalid return types and wrong names."""
    from stage1.tools.umsi_step2b_gate_runner import GateResult, safe_evaluate_gate

    # Invalid return type (returns an integer instead of GateResult)
    def _returns_int(*a: Any, **kw: Any) -> int:
        return 42

    r_invalid = safe_evaluate_gate("gate_a", _returns_int, lambda: ())
    assert not r_invalid.passed
    assert r_invalid.reason == "EVALUATOR_INVALID_RETURN"
    assert r_invalid.name == "gate_a"

    # Wrong gate name (returns gate_b when gate_a expected)
    def _returns_wrong_name(*a: Any, **kw: Any) -> GateResult:
        return GateResult(name="gate_b", passed=True)

    r_wrong = safe_evaluate_gate("gate_a", _returns_wrong_name, lambda: ())
    assert not r_wrong.passed
    assert r_wrong.reason == "EVALUATOR_WRONG_GATE_NAME"
    assert r_wrong.name == "gate_a"
    assert r_wrong.observed["expected_gate_name"] == "gate_a"
    assert r_wrong.observed["returned_gate_name"] == "gate_b"


# ---------------------------------------------------------------------------
# P6-9: run_frozen_experiment delegates to _run_post_capture_sequence
# ---------------------------------------------------------------------------

def test_p6_run_frozen_experiment_delegates_to_post_capture_helper() -> None:
    """Verify by source inspection that run_frozen_experiment calls the shared helper."""
    import inspect
    from stage1.tools.umsi_step2b_gate_runner import (
        _run_post_capture_sequence,
        run_frozen_experiment,
    )

    src = inspect.getsource(run_frozen_experiment)
    assert "_run_post_capture_sequence" in src, (
        "run_frozen_experiment must delegate to _run_post_capture_sequence"
    )
    # The inline gate evaluation list must NOT appear in run_frozen_experiment
    assert "gate_results: List[GateResult]" not in src, (
        "run_frozen_experiment must not contain an inline gate_results list"
    )


# ---------------------------------------------------------------------------
# P6-10 to P6-15: Shared helper + one fake capture_and_invoke + raising gate
# ---------------------------------------------------------------------------

def test_p6_shared_helper_fake_inference_raising_gate_c(tmp_path: Path) -> None:
    """After exactly one fake capture_and_invoke, the shared helper processes
    a raising Gate C evaluator and still writes both evidence files.

    This test exercises the same _run_post_capture_sequence path used by
    run_frozen_experiment, not a manually assembled sequence.
    """
    from stage1.tools.umsi_step2b_gate_runner import (
        _run_post_capture_sequence,
        capture_and_invoke,
    )

    # Step 1: exactly one fake inference via the production capture mechanism
    fake_model, inner = _make_fake_model(h=4, w=4)
    captured, _ = capture_and_invoke(fake_model, "ignored.png")
    assert inner.call_count == 1, "exactly one inference before helper call"

    # Step 2: remove gate_c from contract to force KeyError inside safe boundary
    contract = _make_minimal_contract()
    del contract["gates"]["gate_c"]

    rng = np.random.default_rng(7)
    ref_sal = rng.random((4, 4)).astype(np.float32)
    ref_clf = rng.random(6).astype(np.float32)

    # Step 3: call the shared helper — same path as run_frozen_experiment
    from stage1.tools.umsi_step2b_gate_runner import _build_report_statics
    _rt, _ris = _build_report_statics(
        frozen_thresholds=contract["historical_reused_authority"]["frozen_thresholds"],
        pinned=contract["pinned_identities"],
        observed_shas=_fake_observed_shas(),
        module_file_str="fake_module.py",
        expected_runner_sha="a" * 64,
    )
    rc = _run_post_capture_sequence(
        captured=captured,
        ref_legacy_raw=ref_sal,
        ref_legacy_classif=ref_clf,
        ref_m_lc_raw=ref_sal.copy(),
        ref_m_lc_classif=ref_clf.copy(),
        contract=contract,
        report_thresholds=_rt,
        report_identities_static=_ris,
        output_dir=tmp_path,
    )

    # P6-10: helper was called (non-zero because gate_c raised + possibly others)
    assert rc != 0

    # P6-11: both evidence files written
    bundles = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(bundles) == 1, f"expected 1 bundle dir, got {[p.name for p in bundles]}"
    bundle = bundles[0]
    assert (bundle / "production_outputs.npz").is_file(), "NPZ must be written"
    assert (bundle / "gate_report.json").is_file(), "JSON must be written"

    # P6-12: NPZ arrays exactly equal the captured native arrays
    with np.load(str(bundle / "production_outputs.npz")) as npz_data:
        np.testing.assert_array_equal(
            npz_data["production_raw_saliency"], captured.raw_saliency
        )
        np.testing.assert_array_equal(
            npz_data["production_classif"], captured.classif
        )

    loaded = json.loads((bundle / "gate_report.json").read_text())

    # P6-13: all four gates in JSON + gate_c has exception type and message
    gate_names = [g["name"] for g in loaded["gate_results"]]
    assert gate_names == ["gate_a", "gate_b", "gate_c", "gate_d"], gate_names

    gc = next(g for g in loaded["gate_results"] if g["name"] == "gate_c")
    assert gc["reason"] == "EVALUATOR_EXCEPTION"
    assert gc["observed"]["evaluator_exception_type"] == "KeyError"
    assert gc["observed"]["evaluator_exception_message"] != ""

    # P6-14 / P6-15: inference count exactly one; no retry
    assert loaded["inference"]["real_inference_call_count"] == 1
    assert inner.call_count == 1, (
        f"model called {inner.call_count} times; must remain at 1 after helper"
    )


# ---------------------------------------------------------------------------
# P6-16: Missing or malformed frozen_thresholds → ContractError
# ---------------------------------------------------------------------------

def test_p6_missing_frozen_thresholds_raises_contract_error(tmp_path: Path) -> None:
    """load_and_validate_contract raises ContractError when frozen_thresholds absent."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    del c["historical_reused_authority"]["frozen_thresholds"]
    contract_path = tmp_path / "no_thr.json"
    contract_path.write_text(json.dumps(c))

    with pytest.raises(ContractError, match="frozen_thresholds"):
        load_and_validate_contract(contract_path)


def test_p6_malformed_frozen_thresholds_not_dict_raises(tmp_path: Path) -> None:
    """load_and_validate_contract raises ContractError when frozen_thresholds is not a dict."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    c["historical_reused_authority"]["frozen_thresholds"] = "not_a_dict"
    contract_path = tmp_path / "bad_thr.json"
    contract_path.write_text(json.dumps(c))

    with pytest.raises(ContractError, match="frozen_thresholds"):
        load_and_validate_contract(contract_path)


@pytest.mark.parametrize("missing_key", [
    "pearson_min", "spearman_min", "ssim_min", "cli_abs_max", "repeatability_abs_max"
])
def test_p6_missing_threshold_key_raises_contract_error(
    tmp_path: Path, missing_key: str
) -> None:
    """load_and_validate_contract raises ContractError for each missing threshold key."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    del c["historical_reused_authority"]["frozen_thresholds"][missing_key]
    contract_path = tmp_path / f"missing_{missing_key}.json"
    contract_path.write_text(json.dumps(c))

    with pytest.raises(ContractError, match="frozen_thresholds"):
        load_and_validate_contract(contract_path)


# ===========================================================================
# P6R — Targeted post-capture, schema and orchestration test repairs
# ===========================================================================

# ---------------------------------------------------------------------------
# P6R-3: Genuine Gate-C evaluator exception through the shared helper
# (monkeypatches module-level evaluate_gate_c so it raises RuntimeError)
# ---------------------------------------------------------------------------

def test_p6r_genuine_gate_c_evaluator_exception_via_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four module-level evaluators are wrapped with counters and a shared
    ordered call log.  Gate C raises; Gates A, B and D delegate to genuine
    evaluators.  Asserts exact counts, exact call order, Gate D ran after
    Gate C, evaluator-specific output fields, exception details, evidence
    files, and no retry.
    """
    import stage1.tools.umsi_step2b_gate_runner as runner_mod
    from stage1.tools.umsi_step2b_gate_runner import (
        _build_report_statics,
        _run_post_capture_sequence,
        capture_and_invoke,
        evaluate_gate_a,
        evaluate_gate_b,
        evaluate_gate_d,
    )

    # Step 1: one fake inference (h=8, w=8 for better SSIM computation)
    fake_model, inner = _make_fake_model(h=8, w=8)
    captured, _ = capture_and_invoke(fake_model, "ignored.png")
    assert inner.call_count == 1, "exactly one inference before helper"

    # Step 2: shared state for all four wrappers
    invoke_counts: Dict[str, int] = {"gate_a": 0, "gate_b": 0, "gate_c": 0, "gate_d": 0}
    gate_call_order: list = []

    # Save genuine evaluators for A, B, D
    _real_gate_a = evaluate_gate_a
    _real_gate_b = evaluate_gate_b
    _real_gate_d = evaluate_gate_d

    def _counting_gate_a(*args: Any) -> Any:
        invoke_counts["gate_a"] += 1
        gate_call_order.append("gate_a")
        return _real_gate_a(*args)

    def _counting_gate_b(*args: Any) -> Any:
        invoke_counts["gate_b"] += 1
        gate_call_order.append("gate_b")
        return _real_gate_b(*args)

    def _counting_gate_c(prod_classif: Any, ref_m_lc_classif: Any, gate_def: Any) -> Any:
        invoke_counts["gate_c"] += 1
        gate_call_order.append("gate_c")
        raise RuntimeError("synthetic gate-c evaluator failure")

    def _counting_gate_d(*args: Any) -> Any:
        invoke_counts["gate_d"] += 1
        gate_call_order.append("gate_d")
        return _real_gate_d(*args)

    # Step 3: patch all four module-level names used by _run_post_capture_sequence
    monkeypatch.setattr(runner_mod, "evaluate_gate_a", _counting_gate_a)
    monkeypatch.setattr(runner_mod, "evaluate_gate_b", _counting_gate_b)
    monkeypatch.setattr(runner_mod, "evaluate_gate_c", _counting_gate_c)
    monkeypatch.setattr(runner_mod, "evaluate_gate_d", _counting_gate_d)

    # Step 4: valid contract; use identical arrays so A/B/D pass cleanly
    contract = _make_minimal_contract()
    ref_sal_2d = captured.raw_saliency[:, :, 0]               # (8,8) float32
    ref_clf_identical = np.array(captured.classif, copy=True)  # (6,) float32

    _rt, _ris = _build_report_statics(
        frozen_thresholds=contract["historical_reused_authority"]["frozen_thresholds"],
        pinned=contract["pinned_identities"],
        observed_shas=_fake_observed_shas(),
        module_file_str="fake_module.py",
        expected_runner_sha="a" * 64,
    )

    # Step 5: call the actual shared helper used by run_frozen_experiment
    rc = _run_post_capture_sequence(
        captured=captured,
        ref_legacy_raw=ref_sal_2d,
        ref_legacy_classif=ref_clf_identical,
        ref_m_lc_raw=ref_sal_2d.copy(),
        ref_m_lc_classif=ref_clf_identical.copy(),
        contract=contract,
        report_thresholds=_rt,
        report_identities_static=_ris,
        output_dir=tmp_path,
    )

    # Step 6: overall must fail (Gate C exception)
    assert rc != 0, "Gate C raised → rc must be non-zero"

    # Step 7: exact invocation counts for all four gates
    assert invoke_counts == {"gate_a": 1, "gate_b": 1, "gate_c": 1, "gate_d": 1}, (
        f"Expected all counts == 1, got: {invoke_counts}"
    )

    # Step 8: exact gate call order
    assert gate_call_order == ["gate_a", "gate_b", "gate_c", "gate_d"], (
        f"Expected ['gate_a','gate_b','gate_c','gate_d'], got: {gate_call_order}"
    )

    # Step 9: Gate D ran after Gate C (explicit order proof)
    assert gate_call_order.index("gate_d") > gate_call_order.index("gate_c"), (
        "Gate D must be invoked after Gate C raises"
    )

    # Step 10: both evidence files written
    bundles = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(bundles) == 1, f"expected 1 bundle, got {[p.name for p in bundles]}"
    bundle = bundles[0]
    assert (bundle / "production_outputs.npz").is_file(), "NPZ must be written"
    assert (bundle / "gate_report.json").is_file(), "JSON must be written"

    # Step 11: NPZ arrays exactly equal captured native arrays
    with np.load(str(bundle / "production_outputs.npz")) as npz_data:
        np.testing.assert_array_equal(
            npz_data["production_raw_saliency"], captured.raw_saliency
        )
        np.testing.assert_array_equal(
            npz_data["production_classif"], captured.classif
        )

    loaded = json.loads((bundle / "gate_report.json").read_text())

    # Step 12: exactly Gates A–D in JSON
    gate_names = [g["name"] for g in loaded["gate_results"]]
    assert gate_names == ["gate_a", "gate_b", "gate_c", "gate_d"], gate_names

    # Gate C: exact exception type and message
    gc = next(g for g in loaded["gate_results"] if g["name"] == "gate_c")
    assert not gc["passed"]
    assert gc["reason"] == "EVALUATOR_EXCEPTION"
    assert gc["observed"]["evaluator_exception_type"] == "RuntimeError"
    assert gc["observed"]["evaluator_exception_message"] == "synthetic gate-c evaluator failure"

    # Gate A: evaluator-specific fields prove genuine evaluator reached comparison step
    ga = next(g for g in loaded["gate_results"] if g["name"] == "gate_a")
    assert "array_equal" in ga["observed"], (
        "Gate A genuine evaluator must produce 'array_equal'"
    )
    assert "max_abs_diff_float64" in ga["observed"], (
        "Gate A genuine evaluator must produce 'max_abs_diff_float64'"
    )

    # Gate B: correlation/SSIM fields prove genuine evaluator reached metric step
    gb = next(g for g in loaded["gate_results"] if g["name"] == "gate_b")
    assert "pearson" in gb["observed"], (
        "Gate B genuine evaluator must produce 'pearson'"
    )
    assert "spearman" in gb["observed"], (
        "Gate B genuine evaluator must produce 'spearman'"
    )
    assert "windowed_ssim" in gb["observed"], (
        "Gate B genuine evaluator must produce 'windowed_ssim'"
    )

    # Gate D: threshold fields prove genuine evaluator ran
    gd = next(g for g in loaded["gate_results"] if g["name"] == "gate_d")
    assert "max_abs_diff_float64" in gd["observed"], (
        "Gate D genuine evaluator must produce 'max_abs_diff_float64'"
    )
    assert "cli_abs_max" in gd["observed"], (
        "Gate D genuine evaluator must record 'cli_abs_max' threshold"
    )

    # Step 13: overall conclusion reflects failure
    assert not loaded["conclusion"].startswith("STEP_2B_PASS")

    # Step 14: prediction count exactly one, no retry
    assert loaded["inference"]["real_inference_call_count"] == 1
    assert inner.call_count == 1, (
        f"model called {inner.call_count} times; must stay at 1 after helper"
    )


# ---------------------------------------------------------------------------
# P6R-4: Behavioral delegation — run_frozen_experiment monkeypatched end-to-end
# ---------------------------------------------------------------------------

def test_p6r_run_frozen_experiment_behavioral_delegation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioral temporal-ordering test for run_frozen_experiment.

    Instruments four boundaries via monkeypatching and asserts the exact
    call order:  prepare < model_construction < capture < post_capture.

    Also verifies:
    - Each phase occurs exactly once.
    - The identical CapturedOutputs object reaches the helper.
    - The helper's sentinel return code is propagated.
    - No raw frozen_thresholds / pinned / observed_shas mapping reaches the helper.
    - The pre-built report_thresholds and report_identities_static dicts reach it.
    - output_dir is an absolute Path; module_file is encoded in report_identities_static.
    """
    import argparse
    import importlib as _importlib
    import inspect as _inspect
    import types
    import stage1.tools.umsi_step2b_gate_runner as runner_mod
    from stage1.tools.umsi_step2b_gate_runner import (
        CapturedOutputs,
        run_frozen_experiment,
    )

    SENTINEL_RC = 42
    call_log: list = []
    post_capture_kwargs_received: list = []

    # ── Fake captured outputs ──
    rng = np.random.default_rng(99)
    fake_captured = CapturedOutputs(
        raw_saliency=rng.random((4, 4, 1)).astype(np.float32),
        classif=rng.random(6).astype(np.float32),
        call_count=1,
        restored=True,
    )

    # ── Fake contract (valid schema) ──
    contract = _make_minimal_contract()
    contract["pinned_identities"]["expected_repository_head"] = "HEAD_FAKE"
    contract["pinned_identities"]["production_source_sha256_before_inference"] = "b" * 64
    contract["pinned_identities"]["evidence_npz_sha256"] = "c" * 64
    contract["pinned_identities"]["fixture_sha256_lowcontrast"] = "d" * 64
    contract["pinned_identities"]["weights_sha256"] = "e" * 64
    contract["pinned_identities"]["production_source_path"] = "saliency/umsi_model.py"
    contract_file = tmp_path / "fake_contract.json"
    contract_file.write_text(json.dumps(contract))

    # ── Monkeypatch load_and_validate_contract ──
    monkeypatch.setattr(runner_mod, "load_and_validate_contract", lambda _: contract)

    # ── Monkeypatch run_preflight ──
    monkeypatch.setattr(runner_mod, "run_preflight", lambda **_kw: _fake_observed_shas())

    # ── Monkeypatch np.load ──
    rng2 = np.random.default_rng(7)
    ref_sal_512 = rng2.random((512, 512)).astype(np.float32)
    ref_clf_6 = rng2.random(6).astype(np.float64)

    class _FakeNpz512:
        def __init__(self) -> None:
            self._d = {
                contract["reference_members"]["legacy_raw_saliency"]: ref_sal_512,
                contract["reference_members"]["legacy_classification"]: ref_clf_6,
                contract["reference_members"]["controlled_modern_raw_saliency"]: ref_sal_512.copy(),
                contract["reference_members"]["controlled_modern_classification"]: ref_clf_6.copy(),
            }
        def __getitem__(self, k: str) -> Any: return self._d[k]
        def __enter__(self) -> "_FakeNpz512": return self
        def __exit__(self, *_: Any) -> None: pass

    monkeypatch.setattr(np, "load", lambda *a, **kw: _FakeNpz512())

    # ── Monkeypatch importlib.import_module ──
    fake_umsi_mod = types.ModuleType("saliency.umsi_model")
    source_path = tmp_path / "saliency" / "umsi_model.py"
    source_path.parent.mkdir(exist_ok=True)
    source_path.write_text("# fake")
    fake_umsi_mod.__file__ = str(source_path)

    original_import = _importlib.import_module
    def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
        if name == "saliency.umsi_model":
            return fake_umsi_mod
        return original_import(name, *a, **kw)
    monkeypatch.setattr(_importlib, "import_module", _fake_import)

    # ── Monkeypatch inspect.getfile ──
    monkeypatch.setattr(_inspect, "getfile", lambda m: str(source_path))

    # ── Instrument boundary 1: _build_report_statics records "prepare" ──
    original_build = runner_mod._build_report_statics
    def _recording_build(**kwargs: Any) -> Any:
        call_log.append("prepare")
        return original_build(**kwargs)
    monkeypatch.setattr(runner_mod, "_build_report_statics", _recording_build)

    # ── Instrument boundary 2: UMSIPlus records "model_construction" ──
    fake_inner = _FakeInnerModel(
        np.zeros((1, 4, 4, 1), dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
    )
    def _recording_umsi_plus(weights_path: Any) -> Any:
        call_log.append("model_construction")
        return _FakeUMSIPlus(fake_inner)
    fake_umsi_mod.UMSIPlus = _recording_umsi_plus

    # ── Instrument boundary 3: capture_and_invoke records "capture" ──
    def _recording_capture(model: Any, fixture: Any) -> Any:
        call_log.append("capture")
        return fake_captured, None
    monkeypatch.setattr(runner_mod, "capture_and_invoke", _recording_capture)

    # ── Instrument boundary 4: _run_post_capture_sequence records "post_capture" ──
    def _recording_post_capture(**kwargs: Any) -> int:
        call_log.append("post_capture")
        post_capture_kwargs_received.append(kwargs)
        return SENTINEL_RC
    monkeypatch.setattr(runner_mod, "_run_post_capture_sequence", _recording_post_capture)

    # ── Build args ──
    args = argparse.Namespace(
        execute_frozen_experiment=True,
        repo=str(tmp_path),
        contract=str(contract_file),
        contract_sha="a" * 64,
        weights=str(tmp_path / "weights.hdf5"),
        evidence_npz=str(tmp_path / "ref.npz"),
        fixture=str(tmp_path / "fixture.png"),
        output_dir=str(tmp_path / "evidence"),
        expected_runner_sha="a" * 64,
    )
    for p in [tmp_path / "weights.hdf5", tmp_path / "ref.npz", tmp_path / "fixture.png"]:
        p.write_bytes(b"\x00stub")

    # ── Execute ──
    rc = run_frozen_experiment(args)

    # === Exact call order: prepare < model_construction < capture < post_capture ===
    assert call_log == ["prepare", "model_construction", "capture", "post_capture"], (
        f"Expected ordered sequence, got: {call_log}"
    )
    idx = call_log.index
    assert idx("prepare") < idx("model_construction"), "prepare must precede model_construction"
    assert idx("model_construction") < idx("capture"), "model_construction must precede capture"
    assert idx("capture") < idx("post_capture"), "capture must precede post_capture"

    # Each phase exactly once
    assert call_log.count("prepare") == 1, "prepare must occur exactly once"
    assert call_log.count("capture") == 1, "capture must occur exactly once"
    assert call_log.count("post_capture") == 1, "post_capture must occur exactly once"

    # Sentinel RC propagated
    assert rc == SENTINEL_RC, f"Expected sentinel {SENTINEL_RC}, got {rc}"

    # Identical CapturedOutputs reaches helper
    assert len(post_capture_kwargs_received) == 1
    helper_kwargs = post_capture_kwargs_received[0]
    assert helper_kwargs["captured"] is fake_captured, (
        "Helper must receive the exact same CapturedOutputs object from capture"
    )

    # output_dir is absolute Path (resolved before fake capture event in call_log)
    assert isinstance(helper_kwargs["output_dir"], Path), (
        "output_dir must be a Path"
    )
    assert helper_kwargs["output_dir"].is_absolute(), (
        "output_dir must be absolute (resolved before model construction)"
    )

    # No raw source mappings reach the helper
    for forbidden in ("frozen_thresholds", "pinned", "observed_shas",
                      "module_file", "expected_runner_sha"):
        assert forbidden not in helper_kwargs, (
            f"Raw mapping {forbidden!r} must not be passed to the helper"
        )

    # Pre-built report dicts are present and have correct structure
    assert "report_thresholds" in helper_kwargs, "report_thresholds must reach helper"
    assert "report_identities_static" in helper_kwargs, "report_identities_static must reach helper"

    rt = helper_kwargs["report_thresholds"]
    assert isinstance(rt, dict)
    expected_thr_keys = {
        "pearson_min", "spearman_min", "ssim_min",
        "cli_abs_max_step2b_classif_only__not_stage2", "repeatability_abs_max",
    }
    assert set(rt.keys()) == expected_thr_keys, (
        f"report_thresholds keys mismatch: {set(rt.keys())}"
    )

    ris = helper_kwargs["report_identities_static"]
    assert isinstance(ris, dict)
    for required_key in ("head", "expected_runner_sha256", "observed_runner_sha256",
                         "runner_sha256_match", "production_import_origin"):
        assert required_key in ris, f"report_identities_static missing {required_key!r}"


# ---------------------------------------------------------------------------
# P7R-4: Post-capture guarded-mapping regression test
# ---------------------------------------------------------------------------

# Environment-variable marker identifying the fresh P7 child process. Its
# presence is what makes the ``if __name__ == "__main__"`` bootstrap below
# run the check; its absence during normal pytest collection/import means
# that bootstrap is always a no-op then. The wrapper test also refuses to
# proceed if it finds this marker already set in ITS OWN environment, which
# would indicate an (unexpected) attempt to recurse into another child.
_P7R_CHILD_MARKER = "UMSI_P7R_CHILD_MARKER"


def _p7r_child_main() -> int:
    """Fresh-process entry point for the full P7 guarded-mapping check.

    Runs the ENTIRE original check (guarded-dict sealing, exactly-one fake
    inference, post-capture-sequence delegation, evidence-bundle contents,
    and the final import-isolation assertion) in a brand-new process with a
    pristine ``sys.modules``. This process has no pytest assertion-rewriting
    machinery involved at all (it runs as a plain script, not under pytest),
    so even a failing module-name check stays cheap and fast.

    Returns 0 on success. Any failed check raises (AssertionError or other
    exception), which becomes a nonzero exit code with a traceback on stderr
    for the parent process to report.
    """
    from stage1.tools.umsi_step2b_gate_runner import (
        _build_report_statics,
        _run_post_capture_sequence,
        capture_and_invoke,
    )

    pre_seal_accesses: list = []
    post_seal_accesses: list = []

    class _GuardedDict(dict):
        def __init__(self, name: str, data: dict) -> None:
            super().__init__(data)
            self._name = name
            self._sealed = False

        def seal(self) -> None:
            self._sealed = True

        def __getitem__(self, key: Any) -> Any:
            if self._sealed:
                post_seal_accesses.append((self._name, key))
                raise AssertionError(
                    f"Post-capture access to sealed mapping {self._name!r}[{key!r}]"
                )
            pre_seal_accesses.append((self._name, key))
            return super().__getitem__(key)

    contract = _make_minimal_contract()
    thr_data = dict(contract["historical_reused_authority"]["frozen_thresholds"])
    pinned_data = dict(contract["pinned_identities"])
    obs_data = dict(_fake_observed_shas())

    guarded_thr = _GuardedDict("frozen_thresholds", thr_data)
    guarded_pinned = _GuardedDict("pinned", pinned_data)
    guarded_obs = _GuardedDict("observed_shas", obs_data)

    # PRE-CAPTURE: extract all report-only values via _build_report_statics
    report_thresholds, report_identities_static = _build_report_statics(
        frozen_thresholds=guarded_thr,
        pinned=guarded_pinned,
        observed_shas=guarded_obs,
        module_file_str="fake.py",
        expected_runner_sha="a" * 64,
    )

    # Verify all five required threshold keys were accessed before sealing
    accessed_thr_keys = {k for (n, k) in pre_seal_accesses if n == "frozen_thresholds"}
    for required_key in ("pearson_min", "spearman_min", "ssim_min",
                         "cli_abs_max", "repeatability_abs_max"):
        assert required_key in accessed_thr_keys, (
            f"frozen_thresholds[{required_key!r}] must be accessed before seal"
        )

    # Seal the guarded dicts — any further access raises AssertionError
    guarded_thr.seal()
    guarded_pinned.seal()
    guarded_obs.seal()

    # CAPTURE BOUNDARY: exactly one fake inference
    fake_model, inner = _make_fake_model(h=4, w=4)
    captured, _ = capture_and_invoke(fake_model, "ignored.png")
    assert inner.call_count == 1, "exactly one inference"

    rng = np.random.default_rng(77)
    ref_sal = rng.random((4, 4)).astype(np.float32)
    ref_clf = rng.random(6).astype(np.float32)

    # No pytest tmp_path fixture is available in a plain-script child
    # process, so a dedicated temporary directory is created and cleaned up
    # explicitly.
    tmp_dir = Path(tempfile.mkdtemp(prefix="p7r_child_"))
    try:
        # POST-CAPTURE: call shared helper with only pre-built statics
        # If any sealed dict is accessed, AssertionError propagates immediately
        rc = _run_post_capture_sequence(
            captured=captured,
            ref_legacy_raw=ref_sal,
            ref_legacy_classif=ref_clf,
            ref_m_lc_raw=ref_sal.copy(),
            ref_m_lc_classif=ref_clf.copy(),
            contract=contract,      # gate defs — accessed only inside safe boundary
            report_thresholds=report_thresholds,
            report_identities_static=report_identities_static,
            output_dir=tmp_dir,
        )

        # No post-capture access to any sealed dict
        assert post_seal_accesses == [], (
            f"Report-only mappings accessed after capture: {post_seal_accesses}"
        )

        # Evidence bundle must be written (rc = 0 pass or 2 gate-fail; 1 = infra error)
        assert rc in (0, 2), f"Expected 0 or 2 (not infra-error 1), got {rc}"
        bundles = [p for p in tmp_dir.iterdir() if p.is_dir()]
        assert len(bundles) == 1, "Evidence bundle must be written"

        loaded = json.loads((bundles[0] / "gate_report.json").read_text())
        # Verify pre-built threshold values appear correctly in the report
        thr_section = loaded["contract_thresholds"]
        assert thr_section["pearson_min"] == 0.999
        assert thr_section["repeatability_abs_max"] == 0.0

        # Gate results are present (gate-def lookups used unguarded contract["gates"])
        assert len(loaded["gate_results"]) == 4

        # No production module was imported. Compact module-NAME check only —
        # never stringify the whole sys.modules mapping (that stringification
        # is exactly what made the equivalent in-process assertion
        # pathologically expensive once another test file had already loaded
        # TensorFlow into a shared pytest process).
        forbidden = sorted(
            m for m in sys.modules
            if "tensorflow" in m.lower()
            or (m.startswith("keras") and "saliency" not in m)
        )
        assert not forbidden, f"Forbidden modules imported: {forbidden[:5]!r}"
        assert "saliency.umsi_model" not in sys.modules, (
            "saliency.umsi_model must not be imported"
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return 0


if __name__ == "__main__" and os.environ.get(_P7R_CHILD_MARKER) == "1":
    # Child-process bootstrap only. This branch is never reached during
    # normal pytest collection/import (that never sets __name__ to
    # "__main__"), and it is only reached in this file's own child
    # invocations spawned by the wrapper test below -- so there is no path
    # by which this can recurse into spawning a further child.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    sys.exit(_p7r_child_main())


def test_p7r_guarded_mapping_no_post_capture_access() -> None:
    """Prove all report-only mapping accesses occur before capture.

    Guarded dicts record every access.  After _build_report_statics extracts
    all required values (pre-capture), the dicts are sealed.  Any further
    access raises AssertionError.  Calling _run_post_capture_sequence with
    only the pre-built statics must not trigger any sealed-dict access.

    Also verifies:
    1. All required threshold keys were accessed before sealing.
    2. Gate-definition lookups occur only inside safe_evaluate_gate boundaries
       (via the unguarded contract["gates"] mapping).
    3. Evidence writing succeeds (rc != 1 infrastructure error).
    4. No production module or real artifact is accessed.

    This full check runs in a FRESH subprocess (see ``_p7r_child_main``
    above). Checking ``sys.modules`` in the shared pytest process would be
    order-dependent: if another test file earlier in the same session
    already imported the real ``saliency.umsi_model`` (e.g.
    ``test_umsi_boundary_input_probe.py::test_preprocess_image_matches_legacy_float64_order``),
    TensorFlow would already be present in the shared process regardless of
    what this test's own code does -- and pytest's assertion-rewriting
    introspection over a TensorFlow-sized ``sys.modules`` becomes extremely
    slow (it looks like a hang, though it is not an infinite loop). A fresh
    child process starts with none of that pre-existing state.
    """
    if os.environ.get(_P7R_CHILD_MARKER):
        raise RuntimeError(
            "test_p7r_guarded_mapping_no_post_capture_access must not be "
            "invoked from within its own child process (recursion guard)."
        )

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env[_P7R_CHILD_MARKER] = "1"

    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        out_tail = str(exc.stdout or "")[-2000:]
        err_tail = str(exc.stderr or "")[-2000:]
        raise AssertionError(
            f"P7 child process timed out after {exc.timeout}s.\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )

    if result.returncode != 0:
        out_tail = result.stdout[-2000:]
        err_tail = result.stderr[-2000:]
        raise AssertionError(
            f"P7 child process failed (exit={result.returncode}).\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )


# ---------------------------------------------------------------------------
# P7R2-3: Guarded mappings exercised through run_frozen_experiment
# ---------------------------------------------------------------------------

# Environment-variable marker identifying the fresh P7R2 child process.
# Mirrors the _P7R_CHILD_MARKER pattern above: only set by the wrapper test
# below when spawning its own child, never set during normal pytest
# collection/import, so there is no recursion path.
_P7R2_CHILD_MARKER = "UMSI_P7R2_CHILD_MARKER"


def _p7r2_child_main() -> int:
    """Fresh-process entry point for the full P7R2 run_frozen_experiment check.

    Runs the ENTIRE original check (guarded mappings injected into the
    contract, all monkeypatched boundaries, direct run_frozen_experiment
    call, orchestration-order assertions, evidence-bundle assertions, and
    the final import-isolation assertion) in a brand-new process with a
    pristine ``sys.modules``. Returns 0 on success; any failed check raises,
    which becomes a nonzero exit code with a traceback on stderr for the
    parent process to report.

    No pytest tmp_path/monkeypatch fixtures are available in a plain-script
    child process. A dedicated temp directory and a standalone MonkeyPatch
    instance (public pytest API, usable without the fixture machinery) are
    used instead, with explicit cleanup/undo in the finally block below.
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="p7r2_child_"))
    mp = pytest.MonkeyPatch()
    try:
        _p7r2_child_body(tmp_path, mp)
    finally:
        mp.undo()
        shutil.rmtree(str(tmp_path), ignore_errors=True)
    return 0


def _p7r2_child_body(tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
    """Original P7R2 check body, parameterized over an explicit temp dir and
    MonkeyPatch instance instead of pytest's ``tmp_path``/``monkeypatch``
    fixtures (unavailable outside a running pytest session).
    """
    import argparse
    import importlib as _importlib
    import inspect as _inspect
    import types
    import stage1.tools.umsi_step2b_gate_runner as runner_mod
    from stage1.tools.umsi_step2b_gate_runner import run_frozen_experiment

    # ── Guarded mapping implementation ──────────────────────────────────────
    pre_seal_accesses: list = []
    post_seal_accesses: list = []

    class _GuardedDict(dict):
        def __init__(self, name: str, data: dict) -> None:
            super().__init__(data)
            self._name = name
            self._sealed = False

        def seal(self) -> None:
            self._sealed = True

        def _record(self, op: str, key: Any) -> None:
            if self._sealed:
                post_seal_accesses.append((self._name, op, key))
                raise AssertionError(
                    f"Post-capture access to sealed {self._name!r}.{op}({key!r})"
                )
            pre_seal_accesses.append((self._name, op, key))

        def __getitem__(self, key: Any) -> Any:
            self._record("__getitem__", key)
            return super().__getitem__(key)

        def get(self, key: Any, default: Any = None) -> Any:
            self._record("get", key)
            return super().get(key, default)

        def __contains__(self, key: Any) -> bool:
            self._record("__contains__", key)
            return super().__contains__(key)

        def keys(self) -> Any:
            self._record("keys", None)
            return super().keys()

        def values(self) -> Any:
            self._record("values", None)
            return super().values()

        def items(self) -> Any:
            self._record("items", None)
            return super().items()

        def __iter__(self) -> Any:
            self._record("__iter__", None)
            return super().__iter__()

    # ── Build contract with guarded source mappings ─────────────────────────
    base_contract = _make_minimal_contract()

    # We will inject guarded dicts into the contract so that run_frozen_experiment
    # receives them via load_and_validate_contract and run_preflight results.
    thr_data = dict(base_contract["historical_reused_authority"]["frozen_thresholds"])
    pinned_data = dict(base_contract["pinned_identities"])
    obs_data = dict(_fake_observed_shas())

    guarded_thr = _GuardedDict("frozen_thresholds", thr_data)
    guarded_pinned = _GuardedDict("pinned", pinned_data)
    guarded_obs = _GuardedDict("observed_shas", obs_data)

    # Inject guarded_thr and guarded_pinned into the returned contract
    injected_contract = dict(base_contract)
    injected_ha = dict(base_contract["historical_reused_authority"])
    injected_ha["frozen_thresholds"] = guarded_thr
    injected_contract["historical_reused_authority"] = injected_ha
    injected_contract["pinned_identities"] = guarded_pinned

    # ── Orchestration call log ───────────────────────────────────────────────
    orchestration_log: list = []

    # ── Monkeypatch load_and_validate_contract to return injected contract ──
    mp.setattr(
        runner_mod, "load_and_validate_contract",
        lambda _path: injected_contract,
    )

    # ── Monkeypatch run_preflight to return guarded observed SHAs ───────────
    mp.setattr(
        runner_mod, "run_preflight",
        lambda **_kw: guarded_obs,
    )

    # ── Monkeypatch np.load ──────────────────────────────────────────────────
    rng = np.random.default_rng(3)
    ref_sal_512 = rng.random((512, 512)).astype(np.float32)
    ref_clf_6 = rng.random(6).astype(np.float64)

    class _FakeNpz:
        def __init__(self) -> None:
            self._d = {
                base_contract["reference_members"]["legacy_raw_saliency"]: ref_sal_512,
                base_contract["reference_members"]["legacy_classification"]: ref_clf_6,
                base_contract["reference_members"]["controlled_modern_raw_saliency"]: ref_sal_512.copy(),
                base_contract["reference_members"]["controlled_modern_classification"]: ref_clf_6.copy(),
            }
        def __getitem__(self, k: str) -> Any: return self._d[k]
        def __enter__(self) -> "_FakeNpz": return self
        def __exit__(self, *_: Any) -> None: pass

    mp.setattr(np, "load", lambda *a, **kw: _FakeNpz())

    # ── Monkeypatch importlib.import_module ──────────────────────────────────
    fake_umsi_mod = types.ModuleType("saliency.umsi_model")
    source_path = tmp_path / "saliency" / "umsi_model.py"
    source_path.parent.mkdir(exist_ok=True)
    source_path.write_text("# fake")
    fake_umsi_mod.__file__ = str(source_path)

    original_import = _importlib.import_module
    def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
        if name == "saliency.umsi_model":
            return fake_umsi_mod
        return original_import(name, *a, **kw)
    mp.setattr(_importlib, "import_module", _fake_import)
    mp.setattr(_inspect, "getfile", lambda m: str(source_path))

    # ── Instrument _build_report_statics to record "prepare" ────────────────
    original_build = runner_mod._build_report_statics
    def _recording_build(**kwargs: Any) -> Any:
        orchestration_log.append("prepare")
        return original_build(**kwargs)
    mp.setattr(runner_mod, "_build_report_statics", _recording_build)

    # ── Fake UMSIPlus: records "model_construction" ─────────────────────────
    fake_inner = _FakeInnerModel(
        np.zeros((1, 4, 4, 1), dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
    )
    def _recording_umsi_plus(weights_path: Any) -> Any:
        orchestration_log.append("model_construction")
        return _FakeUMSIPlus(fake_inner)
    fake_umsi_mod.UMSIPlus = _recording_umsi_plus

    # ── Fake capture_and_invoke: records "capture", seals guarded mappings ──
    fake_captured_g = runner_mod.CapturedOutputs(
        raw_saliency=rng.random((4, 4, 1)).astype(np.float32),
        classif=rng.random(6).astype(np.float32),
        call_count=1,
        restored=True,
    )

    def _recording_capture(model: Any, fixture: Any) -> Any:
        orchestration_log.append("capture_and_seal")
        # Seal all guarded mappings at the capture boundary
        guarded_thr.seal()
        guarded_pinned.seal()
        guarded_obs.seal()
        return fake_captured_g, None
    mp.setattr(runner_mod, "capture_and_invoke", _recording_capture)

    # ── Let _run_post_capture_sequence run for real (not monkeypatched) ─────
    # We record its entry via a thin wrapper that delegates immediately
    original_post_capture = runner_mod._run_post_capture_sequence
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    def _recording_post_capture(**kwargs: Any) -> int:
        orchestration_log.append("post_capture")
        return original_post_capture(**kwargs)
    mp.setattr(runner_mod, "_run_post_capture_sequence", _recording_post_capture)

    # ── Build args ───────────────────────────────────────────────────────────
    contract_file = tmp_path / "fake_contract.json"
    contract_file.write_text(json.dumps(base_contract))

    args = argparse.Namespace(
        execute_frozen_experiment=True,
        repo=str(tmp_path),
        contract=str(contract_file),
        contract_sha="a" * 64,
        weights=str(tmp_path / "weights.hdf5"),
        evidence_npz=str(tmp_path / "ref.npz"),
        fixture=str(tmp_path / "fixture.png"),
        output_dir=str(evidence_dir),
        expected_runner_sha="a" * 64,
    )
    for p in [tmp_path / "weights.hdf5", tmp_path / "ref.npz", tmp_path / "fixture.png"]:
        p.write_bytes(b"\x00stub")
    base_contract["pinned_identities"]["production_source_path"] = "saliency/umsi_model.py"

    # ── Execute ─────────────────────────────────────────────────────────────
    rc = run_frozen_experiment(args)

    # === Orchestration call order ============================================
    assert orchestration_log == [
        "prepare", "model_construction", "capture_and_seal", "post_capture"
    ], f"Unexpected orchestration order: {orchestration_log}"
    assert orchestration_log.index("prepare") < orchestration_log.index("model_construction")
    assert orchestration_log.index("model_construction") < orchestration_log.index("capture_and_seal")
    assert orchestration_log.index("capture_and_seal") < orchestration_log.index("post_capture")

    # === No guarded-mapping access after capture =============================
    assert post_seal_accesses == [], (
        f"Guarded mapping(s) accessed after capture: {post_seal_accesses}"
    )

    # === All required threshold keys accessed before seal ====================
    EXPECTED_THR_KEYS = frozenset({
        "pearson_min", "spearman_min", "ssim_min",
        "cli_abs_max", "repeatability_abs_max",
    })
    pre_thr_keys = {k for (n, _op, k) in pre_seal_accesses if n == "frozen_thresholds"}
    assert EXPECTED_THR_KEYS <= pre_thr_keys, (
        f"threshold keys not fully consumed before seal: "
        f"missing {EXPECTED_THR_KEYS - pre_thr_keys}"
    )

    # === All required pinned-identity keys accessed before seal ==============
    EXPECTED_PINNED_KEYS = frozenset({
        "expected_repository_head",
    })
    pre_pinned_keys = {k for (n, _op, k) in pre_seal_accesses if n == "pinned"}
    assert EXPECTED_PINNED_KEYS <= pre_pinned_keys, (
        f"pinned-identity keys not consumed before seal: "
        f"missing {EXPECTED_PINNED_KEYS - pre_pinned_keys}"
    )

    # === All required observed-SHA keys accessed before seal =================
    EXPECTED_OBS_KEYS = frozenset({
        "runner", "source", "contract", "weights", "npz", "fixture",
    })
    pre_obs_keys = {k for (n, _op, k) in pre_seal_accesses if n == "observed_shas"}
    assert EXPECTED_OBS_KEYS <= pre_obs_keys, (
        f"observed-SHA keys not consumed before seal: "
        f"missing {EXPECTED_OBS_KEYS - pre_obs_keys}"
    )

    # === run_frozen_experiment used directly (not the helper alone) ==========
    # Confirmed by the orchestration_log containing all 4 events including
    # model_construction and capture_and_seal, which are only reachable through
    # run_frozen_experiment's production flow.

    # === Both synthetic evidence files written ================================
    bundles = [p for p in evidence_dir.iterdir() if p.is_dir()]
    assert len(bundles) == 1, (
        f"Expected exactly 1 evidence bundle, got: {[p.name for p in bundles]}"
    )
    bundle = bundles[0]
    assert (bundle / "production_outputs.npz").is_file(), "NPZ must be written"
    assert (bundle / "gate_report.json").is_file(), "JSON must be written"

    loaded = json.loads((bundle / "gate_report.json").read_text())

    # === Exactly Gates A–D in report =========================================
    gate_names = [g["name"] for g in loaded["gate_results"]]
    assert set(gate_names) == {"gate_a", "gate_b", "gate_c", "gate_d"}, (
        f"Unexpected gate names: {gate_names}"
    )

    # === No raw mapping reached the post-capture helper ======================
    # (Confirmed by sealed-dict mechanism: any such access would have raised
    # AssertionError and been recorded in post_seal_accesses, which is asserted
    # empty above.)

    # === No production TF/Keras/saliency import ==============================
    # Compact module-NAME check only — never stringify the whole sys.modules
    # mapping (see _p7r_child_main above for why that matters).
    forbidden = sorted(
        m for m in sys.modules
        if "tensorflow" in m.lower()
        or (m.startswith("keras") and "saliency" not in m)
    )
    assert not forbidden, f"Forbidden modules imported: {forbidden[:5]!r}"
    assert "saliency.umsi_model" not in sys.modules, (
        "saliency.umsi_model must not be imported"
    )


if __name__ == "__main__" and os.environ.get(_P7R2_CHILD_MARKER) == "1":
    # Child-process bootstrap only. This branch is never reached during
    # normal pytest collection/import, and it is only reached in this
    # file's own child invocations spawned by the wrapper test below -- so
    # there is no path by which this can recurse into spawning a further
    # child.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    sys.exit(_p7r2_child_main())


def test_p7r2_guarded_mapping_via_run_frozen_experiment() -> None:
    """Exercises run_frozen_experiment() directly with all real boundaries
    monkeypatched.  Guarded mappings for frozen_thresholds, pinned identities
    and observed SHAs detect any access after the capture boundary.

    Expected orchestration order:
      prepare (via _build_report_statics) < model_construction < capture + seal
      < shared post-capture helper

    This full check runs in a FRESH subprocess (see ``_p7r2_child_main`` /
    ``_p7r2_child_body`` above), for the same reason as
    ``test_p7r_guarded_mapping_no_post_capture_access``: checking
    ``sys.modules`` in the shared pytest process is order-dependent, since
    an earlier test file in the same session
    (``test_umsi_boundary_input_probe.py::test_preprocess_image_matches_legacy_float64_order``)
    legitimately loads real TensorFlow via ``saliency.umsi_model``. A fresh
    child process starts with none of that pre-existing state.
    """
    if os.environ.get(_P7R2_CHILD_MARKER):
        raise RuntimeError(
            "test_p7r2_guarded_mapping_via_run_frozen_experiment must not be "
            "invoked from within its own child process (recursion guard)."
        )

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env[_P7R2_CHILD_MARKER] = "1"

    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        out_tail = str(exc.stdout or "")[-2000:]
        err_tail = str(exc.stderr or "")[-2000:]
        raise AssertionError(
            f"P7R2 child process timed out after {exc.timeout}s.\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )

    if result.returncode != 0:
        out_tail = result.stdout[-2000:]
        err_tail = result.stderr[-2000:]
        raise AssertionError(
            f"P7R2 child process failed (exit={result.returncode}).\n"
            f"--- child stdout (tail) ---\n{out_tail}\n"
            f"--- child stderr (tail) ---\n{err_tail}"
        )


# ---------------------------------------------------------------------------
# P6R-5: Malformed threshold tests (all varieties)
# ---------------------------------------------------------------------------

def test_p6r_historical_reused_authority_not_dict(tmp_path: Path) -> None:
    """ContractError when historical_reused_authority is not a dict."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    c["historical_reused_authority"] = "not_a_dict"
    p = tmp_path / "c.json"; p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="historical_reused_authority"):
        load_and_validate_contract(p)


@pytest.mark.parametrize("bad_value,label", [
    ("a string", "string"),
    (None, "null"),
    (True, "boolean True"),
    (False, "boolean False"),
    (float("nan"), "NaN"),
    (float("inf"), "+inf"),
    (float("-inf"), "-inf"),
])
def test_p6r_invalid_threshold_value_type_or_finite(
    tmp_path: Path, bad_value: Any, label: str
) -> None:
    """ContractError for non-numeric, boolean, NaN or infinity threshold values."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    # Use pearson_min as the test key (a correlation/SSIM threshold)
    c = _make_minimal_contract()
    c["historical_reused_authority"]["frozen_thresholds"]["pearson_min"] = bad_value
    p = tmp_path / f"bad_{label.replace(' ', '_')}.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="frozen_thresholds|pearson_min"):
        load_and_validate_contract(p)


def test_p6r_correlation_minimum_above_1(tmp_path: Path) -> None:
    """ContractError when a correlation/SSIM minimum is > 1.0."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    c["historical_reused_authority"]["frozen_thresholds"]["pearson_min"] = 1.5
    p = tmp_path / "pearson_above_1.json"; p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="pearson_min"):
        load_and_validate_contract(p)


def test_p6r_correlation_minimum_below_minus_1(tmp_path: Path) -> None:
    """ContractError when a correlation/SSIM minimum is < -1.0."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    c["historical_reused_authority"]["frozen_thresholds"]["ssim_min"] = -1.5
    p = tmp_path / "ssim_below_minus1.json"; p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="ssim_min"):
        load_and_validate_contract(p)


def test_p6r_abs_max_negative(tmp_path: Path) -> None:
    """ContractError when an absolute maximum threshold is negative."""
    from stage1.tools.umsi_step2b_gate_runner import ContractError, load_and_validate_contract

    c = _make_minimal_contract()
    c["historical_reused_authority"]["frozen_thresholds"]["cli_abs_max"] = -0.01
    p = tmp_path / "cli_neg.json"; p.write_text(json.dumps(c))
    with pytest.raises(ContractError, match="cli_abs_max"):
        load_and_validate_contract(p)


def test_p6r_malformed_threshold_blocks_before_production_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed threshold causes run_frozen_experiment to return 1 before
    reaching the production import, model constructor, or prediction boundary.
    """
    import argparse
    import stage1.tools.umsi_step2b_gate_runner as runner_mod
    from stage1.tools.umsi_step2b_gate_runner import run_frozen_experiment

    import_calls: list = []
    model_calls: list = []
    predict_calls: list = []

    # Contract with a NaN threshold — deliberately invalid
    bad_contract = _make_minimal_contract()
    bad_contract["historical_reused_authority"]["frozen_thresholds"]["pearson_min"] = float("nan")

    contract_file = tmp_path / "bad_contract.json"
    contract_file.write_text(json.dumps(bad_contract))

    # Monkeypatch load_and_validate_contract to run the real function
    # (it should raise ContractError before returning)
    # Do NOT monkeypatch it — let the real validation run.

    # But monkeypatch importlib.import_module to detect if it's called
    import importlib as _importlib
    original_import = _importlib.import_module
    def _spy_import(name: str, *a: Any, **kw: Any) -> Any:
        if name == "saliency.umsi_model":
            import_calls.append(name)
        return original_import(name, *a, **kw)
    monkeypatch.setattr(_importlib, "import_module", _spy_import)

    args = argparse.Namespace(
        execute_frozen_experiment=True,
        repo=str(tmp_path),
        contract=str(contract_file),
        contract_sha="a" * 64,
        weights=str(tmp_path / "w.hdf5"),
        evidence_npz=str(tmp_path / "ref.npz"),
        fixture=str(tmp_path / "fix.png"),
        output_dir=str(tmp_path / "out"),
        expected_runner_sha="a" * 64,
    )

    rc = run_frozen_experiment(args)

    # Must block (return 1)
    assert rc == 1, f"Expected return 1, got {rc}"

    # Production import, model, and prediction must NOT have been reached
    assert import_calls == [], (
        f"saliency.umsi_model must not be imported when contract validation fails; "
        f"got: {import_calls}"
    )
