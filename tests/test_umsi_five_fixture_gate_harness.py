"""Synthetic, no-inference tests for umsi_five_fixture_gate_harness.

All tests use small synthetic in-memory arrays and a fake ``inference_fn``.
No TensorFlow, Keras, saliency.umsi_model, real weights, real fixture images,
or the real evidence NPZ are imported, loaded, or accessed anywhere in this
file. No absolute, user-specific filesystem path appears anywhere in this
file: all fixture/weights/evidence-NPZ files used by tests are created under
pytest's ``tmp_path``. The real fixture SHA-256 hashes are asserted as
literal expected constants (matching the frozen contract) without reading
any real fixture file from disk, so no dependency on a specific machine's
fixture directory exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import stage1.tools.umsi_five_fixture_gate_harness as _harness_module  # noqa: E402
from stage1.tools.umsi_five_fixture_gate_harness import (  # noqa: E402
    CapturedFullInference,
    ContractError,
    ProvenanceError,
    _EXPECTED_FIXTURE_NAMES,
    _LAYER_ORDER,
    _PROTECTED_OUTPUT_DIRECTORY_NAMES,
    _aggregate_overall_verdict,
    _bootstrap_sha256_file,
    _check_git_status_against_allowlist,
    _find_unsafe_run_directory_name_reason,
    _get_runner_module,
    _git_rev_parse_head,
    _git_status_porcelain,
    _validate_run_directory_name,
    _verify_launcher_preflight,
    _verify_output_directory_safety,
    _verify_runtime_versions,
    _verify_worktree_and_blob_identity,
    evaluate_determinism_repeat,
    evaluate_gate_0_model_input_identity,
    load_and_validate_five_fixture_contract,
    preflight_fixture,
    run_five_fixtures,
    run_frozen_five_fixture_experiment,
    run_provenance_only_preflight,
)
from stage1.tools.umsi_step2b_gate_runner import (  # noqa: E402
    PreflightError,
    evaluate_gate_b,
    evaluate_gate_d,
)

_HARNESS_FILE_PATH = Path(_harness_module.__file__).resolve()


def _actual_harness_sha256() -> str:
    """Compute the real harness file's current SHA-256 by reading it fresh,
    never hardcoded as a literal, so tests remain correct across any future
    edit to the harness file."""
    return hashlib.sha256(_HARNESS_FILE_PATH.read_bytes()).hexdigest()



_REAL_CONTRACT_PATH = Path(_repo_root) / "stage1" / "evidence" / "umsi_five_fixture_gate_contract.json"

# Expected fixture SHA-256 values, pinned as literal constants (matching the
# frozen contract). Asserted directly against the contract file, never read
# from a real fixture image on disk.
_EXPECTED_FIXTURE_SHA256 = {
    "lowcontrast": "4a467bdd72d41b9fd490573468db171a254e7cbf4aa8727ac8fb86bef05a3562",
    "ui1": "b0741b0660e3cede9a8f11db92c3973e4336e4c8c859315ba69c8677963ece3b",
    "ui2": "1e693f150402276b819aaa133c17ef5e9540e66789518a47856b17892d7266a6",
    "ui3": "38c4348cbf93da4e95f07d5e2bf7b7a315eec57b9841a8dfc174b8a3bab6cab0",
    "uniform": "2987ab872f4a798686ce3f975c7a8a3ad41bbf619fae113678b5afc53a634db6",
}
_EXPECTED_EVIDENCE_NPZ_FILENAME = "umsi_resize_causality_arrays_6a17288_consolidated.npz"
_EXPECTED_EVIDENCE_NPZ_SHA256 = "9e933c1071924f180c32cc6327363ca5760b5968d497414a9f7939919b630079"
_EXPECTED_WEIGHTS_FILENAME = "umsi++.hdf5"
_EXPECTED_WEIGHTS_SHA256 = "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
_EXPECTED_WEIGHTS_SIZE_BYTES = 120093896


# ---------------------------------------------------------------------------
# Helpers: minimal synthetic contract + fake in-memory "NPZ" + fake capture
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Small stand-in e2e shape (real contract's e2e shape varies per fixture and
# is not schema-hardcoded, so any fixed shape is a valid synthetic contract).
_SYNTHETIC_E2E_SHAPE = (10, 10)


def _make_synthetic_contract(fixtures_dir: Path) -> Dict[str, Any]:
    """Build a minimal, schema-valid five-fixture contract plus matching
    fixture bytes on disk. Model-input/raw/classif shapes match the real,
    schema-hardcoded model boundaries exactly; e2e uses a small stand-in
    shape shared by all fixtures (the real contract's e2e shape varies per
    fixture but that is not enforced by the schema validator).
    """
    names = list(_EXPECTED_FIXTURE_NAMES)
    fixtures: Dict[str, Any] = {}
    reference_members: Dict[str, Any] = {}
    for i, name in enumerate(names):
        blob = f"synthetic-fixture-bytes-{name}-{i}".encode("utf-8")
        (fixtures_dir / f"{name}.png").write_bytes(blob)
        fixtures[name] = {"filename": f"{name}.png", "sha256": _sha256_bytes(blob)}
        reference_members[name] = {
            "legacy_model_input": f"legacy_raw/{name}_preproc",
            "legacy_raw_saliency": f"legacy_raw/{name}_raw",
            "legacy_classification": f"legacy_raw/{name}_classif",
            "legacy_e2e_saliency": f"legacy_raw/{name}_e2e",
        }

    contract = {
        "contract_name": "synthetic_five_fixture_contract",
        "fixtures": fixtures,
        "reference_members": reference_members,
        "evidence_npz_filename": "synthetic_evidence.npz",
        "evidence_npz_sha256": "a" * 64,
        "weights_filename": "synthetic_weights.hdf5",
        "weights_sha256": "b" * 64,
        "weights_size_bytes": _EXPECTED_WEIGHTS_SIZE_BYTES,
        "model_boundaries": {
            "input_tensor_shape": [1, 256, 256, 3],
            "raw_decoder_output_shape": [512, 512, 1],
        },
        "gate_definitions": {
            "gate_0_model_input_identity": {
                "requirements": {"same_shape": True, "same_dtype": True, "array_equal": True},
            },
            "gate_1_saliency": {
                "requirements": {
                    "pearson": {"operator": ">=", "value": 0.999},
                    "spearman": {"operator": ">=", "value": 0.99},
                    "windowed_ssim": {
                        "operator": ">=",
                        "value": 0.99,
                        "params": {
                            "K1": 0.01, "K2": 0.03, "win_size": 7,
                            "boundary": "reflect", "data_range": 1.0,
                            "filter": "scipy.ndimage.uniform_filter",
                        },
                    },
                }
            },
            "gate_2_classification": {
                "requirements": {
                    "same_shape": True,
                    "finite_values": True,
                    "max_abs_diff_float64": {"operator": "<=", "value": 0.01},
                }
            },
            "gate_3_e2e_saliency": {
                "requirements": {
                    "pearson": {"operator": ">=", "value": 0.999},
                    "spearman": {"operator": ">=", "value": 0.99},
                    "windowed_ssim": {
                        "operator": ">=",
                        "value": 0.99,
                        "params": {
                            "K1": 0.01, "K2": 0.03, "win_size": 7,
                            "boundary": "reflect", "data_range": 1.0,
                            "filter": "scipy.ndimage.uniform_filter",
                        },
                    },
                }
            },
            "determinism_repeat": {
                "requirements": {
                    "max_abs_diff_float64": {
                        "operator": "==", "value": 0.0,
                        "bound_to_threshold": "repeatability_abs_max",
                    }
                }
            },
        },
        "frozen_thresholds": {
            "pearson_min": 0.999,
            "spearman_min": 0.99,
            "ssim_min": 0.99,
            "repeatability_abs_max": 0.0,
        },
        "expected_reference_shapes": {
            "legacy_model_input": {"shape": [1, 256, 256, 3], "dtype": "float32"},
            "legacy_raw_saliency": {"shape": [512, 512], "dtype": "float32"},
            "legacy_classification": {"shape": [6], "dtype": "float64"},
            "legacy_e2e_saliency": {
                "dtype": "float32",
                "per_fixture_shape": {name: list(_SYNTHETIC_E2E_SHAPE) for name in names},
            },
        },
        "expected_repository_head": "0" * 40,
        "production_source_identity": {
            "path": "saliency/umsi_model.py",
            "sha256": "c" * 64,
        },
        "reused_legacy_runner_identity": {
            "path": "stage1/tools/umsi_step2b_gate_runner.py",
            "sha256": "d" * 64,
        },
        "candidate_harness_identity": {
            "path": "stage1/tools/umsi_five_fixture_gate_harness.py",
            "sha256": "e" * 64,
        },
        "allowed_untracked_paths": [
            "stage1/evidence/umsi_five_fixture_gate_contract.json",
            "stage1/tools/umsi_five_fixture_gate_harness.py",
            "tests/test_umsi_five_fixture_gate_harness.py",
        ],
        "runtime_identity": {
            "python_implementation": "CPython",
            "python_version": "0.0.0",
            "python_executable": "/synthetic/python",
            "package_versions": {"numpy": "0.0.0"},
        },
        "execution": {
            "run_directory_name": "five_fixture_run_synthetic_00000000",
            "run_directory_name_prefix_requirement": "five_fixture_run_",
            "launcher": {
                "mode": "module",
                "python_executable": "/synthetic/python",
                "interpreter_flags": ["-B"],
                "module": "stage1.tools.umsi_five_fixture_gate_harness",
                "working_directory": "/synthetic/repo",
                "required_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "pythonpath_required_state": "unset",
                "action": "execute_frozen_experiment",
                "contract_path": "/synthetic/repo/stage1/evidence/umsi_five_fixture_gate_contract.json",
                "output_root": "/synthetic/output_root",
                "weights_path": "/synthetic/synthetic_weights.hdf5",
                "evidence_npz_path": "/synthetic/synthetic_evidence.npz",
                "fixtures_root": "/synthetic/fixtures",
            },
            "predecessor_incident": {
                "consumed_run_directory_name": "five_fixture_run_synthetic_consumed_00000000",
                "failure_classification": (
                    "infrastructure_bootstrap_import_path_failure_before_production_model_import"
                ),
                "execution_attempted": True,
                "identity_permanently_consumed": True,
                "run_directory_created": False,
                "production_model_imported": False,
                "weights_loaded": False,
                "inference_executed": False,
                "scientific_verdict_reached": False,
            },
        },
    }
    return contract


class _FakeNPZ:
    """Minimal stand-in for np.load(...)'s NpzFile: dict-like with .keys()."""

    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


def _ref_arrays_for(fixture_name: str) -> Dict[str, np.ndarray]:
    """Deterministic, non-constant reference arrays for one fixture, for all
    four layers, matching the synthetic contract's expected shapes/dtypes."""
    rng = np.random.default_rng(abs(hash(fixture_name)) % (2**32))
    return {
        "legacy_model_input": rng.random((1, 256, 256, 3)).astype(np.float32),
        "legacy_raw_saliency": rng.random((512, 512)).astype(np.float32),
        "legacy_classification": np.array(
            [0.1, 0.2, 0.3, 0.15, 0.15, 0.1], dtype=np.float64
        ),
        "legacy_e2e_saliency": rng.random(_SYNTHETIC_E2E_SHAPE).astype(np.float32),
    }


def _make_fake_npz_for_contract(contract: Dict[str, Any]) -> _FakeNPZ:
    data: Dict[str, np.ndarray] = {}
    for name in _EXPECTED_FIXTURE_NAMES:
        members = contract["reference_members"][name]
        refs = _ref_arrays_for(name)
        for layer, arr in refs.items():
            data[members[layer]] = arr
    return _FakeNPZ(data)


def _perfect_inference_fn_factory(contract: Dict[str, Any], npz: _FakeNPZ):
    """An inference_fn that returns an exact copy of the reference arrays for
    every fixture and every layer — trivially passes all four gates and the
    determinism repeat."""

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        members = contract["reference_members"][fixture_name]
        return CapturedFullInference(
            model_input=np.array(npz[members["legacy_model_input"]], copy=True),
            raw_saliency=np.array(npz[members["legacy_raw_saliency"]], copy=True),
            classif=np.array(npz[members["legacy_classification"]], copy=True),
            e2e_saliency=np.array(npz[members["legacy_e2e_saliency"]], copy=True),
            call_count=1,
            restored=True,
        )

    return _inference_fn


# ---------------------------------------------------------------------------
# 1-9: Real contract structural / provenance checks (no disk fixture reads)
# ---------------------------------------------------------------------------

def test_real_contract_loads_and_validates():
    """1. The actual frozen contract file must load and validate completely."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert set(contract["fixtures"].keys()) == set(_EXPECTED_FIXTURE_NAMES)
    assert len(contract["fixtures"]) == 5
    for name in _EXPECTED_FIXTURE_NAMES:
        assert set(contract["reference_members"][name].keys()) == set(_LAYER_ORDER)


def test_real_contract_exact_fixture_set_and_order():
    """2. Exactly five fixtures; run_five_fixtures uses a fixed, deterministic
    processing order regardless of contract JSON key order."""
    assert _EXPECTED_FIXTURE_NAMES == ("lowcontrast", "ui1", "ui2", "ui3", "uniform")
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert set(contract["fixtures"].keys()) == set(_EXPECTED_FIXTURE_NAMES)
    assert len(set(contract["fixtures"].keys())) == 5


def test_real_contract_all_five_fixture_hashes_match_expected_constants():
    """3. All five fixture hashes, asserted against literal expected
    constants (no real fixture file read from disk)."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    for name, expected_sha in _EXPECTED_FIXTURE_SHA256.items():
        assert contract["fixtures"][name]["sha256"] == expected_sha


def test_real_contract_evidence_provenance():
    """4. Evidence basename and SHA-256."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert contract["evidence_npz_filename"] == _EXPECTED_EVIDENCE_NPZ_FILENAME
    assert contract["evidence_npz_sha256"] == _EXPECTED_EVIDENCE_NPZ_SHA256


def test_real_contract_weights_provenance():
    """5. Checkpoint basename, size, and SHA-256."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert contract["weights_filename"] == _EXPECTED_WEIGHTS_FILENAME
    assert contract["weights_size_bytes"] == _EXPECTED_WEIGHTS_SIZE_BYTES
    assert contract["weights_sha256"] == _EXPECTED_WEIGHTS_SHA256


def test_real_contract_reference_members_all_four_layers():
    """8. All four reference keys present for every fixture."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    for name in _EXPECTED_FIXTURE_NAMES:
        members = contract["reference_members"][name]
        assert members["legacy_model_input"] == f"legacy_raw/{name}_preproc"
        assert members["legacy_raw_saliency"] == f"legacy_raw/{name}_raw"
        assert members["legacy_classification"] == f"legacy_raw/{name}_classif"
        assert members["legacy_e2e_saliency"] == f"legacy_raw/{name}_e2e"


def test_real_contract_expected_reference_shapes_and_dtypes():
    """9. Expected shapes and dtypes for all four layers."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    ers = contract["expected_reference_shapes"]
    assert ers["legacy_model_input"]["shape"] == [1, 256, 256, 3]
    assert ers["legacy_model_input"]["dtype"] == "float32"
    assert ers["legacy_raw_saliency"]["shape"] == [512, 512]
    assert ers["legacy_raw_saliency"]["dtype"] == "float32"
    assert ers["legacy_classification"]["shape"] == [6]
    assert ers["legacy_classification"]["dtype"] == "float64"
    e2e = ers["legacy_e2e_saliency"]
    assert e2e["dtype"] == "float32"
    assert e2e["per_fixture_shape"] == {
        "lowcontrast": [800, 1280],
        "ui1": [800, 1280],
        "ui2": [768, 1024],
        "ui3": [600, 800],
        "uniform": [800, 1280],
    }


def test_real_contract_preserves_unmodified_thresholds():
    """Frozen thresholds must be exactly the historical values, unchanged."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    thr = contract["frozen_thresholds"]
    assert thr["pearson_min"] == 0.999
    assert thr["spearman_min"] == 0.99
    assert thr["ssim_min"] == 0.99
    assert thr["repeatability_abs_max"] == 0.0


# ---------------------------------------------------------------------------
# 6-7: Wrong evidence/checkpoint basename/size/hash -> BLOCKED
# (via run_provenance_only_preflight, using a hand-built minimal contract so
# arbitrarily small dummy files can be used without needing a real 120 MB
# checkpoint or 594 MB evidence NPZ on disk.)
# ---------------------------------------------------------------------------

def _minimal_provenance_contract(tmp_path: Path) -> Tuple[Dict[str, Any], Path, Path]:
    """A hand-built contract dict (bypassing full schema validation, since
    run_provenance_only_preflight only reads specific keys) plus small dummy
    weights/evidence files whose bytes match the pinned hashes exactly."""
    weights_bytes = b"dummy-weights-bytes"
    evidence_bytes = b"dummy-evidence-bytes"
    weights_path = tmp_path / "weights_dir" / "correct_weights.bin"
    evidence_path = tmp_path / "evidence_dir" / "correct_evidence.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(weights_bytes)
    evidence_path.write_bytes(evidence_bytes)

    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    contract = _make_synthetic_contract(fixtures_dir)
    contract["weights_filename"] = "correct_weights.bin"
    contract["weights_size_bytes"] = len(weights_bytes)
    contract["weights_sha256"] = _sha256_bytes(weights_bytes)
    contract["evidence_npz_filename"] = "correct_evidence.npz"
    contract["evidence_npz_sha256"] = _sha256_bytes(evidence_bytes)
    return contract, weights_path, evidence_path


def test_provenance_only_wrong_weights_basename_blocked(tmp_path):
    """6/7 (weights half): wrong checkpoint basename -> BLOCKED."""
    contract, weights_path, evidence_path = _minimal_provenance_contract(tmp_path)
    wrong_name_path = weights_path.with_name("wrong_name.bin")
    weights_path.rename(wrong_name_path)

    report = run_provenance_only_preflight(
        contract=contract, fixtures_root=tmp_path / "fixtures",
        weights_path=wrong_name_path, evidence_npz_path=evidence_path,
    )
    assert report["overall_verdict"] == "BLOCKED"
    assert any("basename" in r for r in report["blocked_reasons"])


def test_provenance_only_wrong_weights_hash_blocked(tmp_path):
    """7: wrong checkpoint SHA-256 (correct basename/size) -> BLOCKED."""
    contract, weights_path, evidence_path = _minimal_provenance_contract(tmp_path)
    weights_path.write_bytes(b"tampered-weights-bytes-of-diff-len!")

    report = run_provenance_only_preflight(
        contract=contract, fixtures_root=tmp_path / "fixtures",
        weights_path=weights_path, evidence_npz_path=evidence_path,
    )
    assert report["overall_verdict"] == "BLOCKED"
    assert any(
        "SHA-256" in r or "size" in r for r in report["blocked_reasons"]
    )


def test_provenance_only_wrong_evidence_basename_blocked(tmp_path):
    """6 (evidence half): wrong evidence NPZ basename -> BLOCKED."""
    contract, weights_path, evidence_path = _minimal_provenance_contract(tmp_path)
    wrong_name_path = evidence_path.with_name("wrong_evidence.npz")
    evidence_path.rename(wrong_name_path)

    report = run_provenance_only_preflight(
        contract=contract, fixtures_root=tmp_path / "fixtures",
        weights_path=weights_path, evidence_npz_path=wrong_name_path,
    )
    assert report["overall_verdict"] == "BLOCKED"
    assert any("basename" in r for r in report["blocked_reasons"])


def test_provenance_only_wrong_evidence_hash_blocked(tmp_path):
    """6 (evidence half): wrong evidence NPZ SHA-256 -> BLOCKED."""
    contract, weights_path, evidence_path = _minimal_provenance_contract(tmp_path)
    evidence_path.write_bytes(b"tampered-evidence-bytes")

    report = run_provenance_only_preflight(
        contract=contract, fixtures_root=tmp_path / "fixtures",
        weights_path=weights_path, evidence_npz_path=evidence_path,
    )
    assert report["overall_verdict"] == "BLOCKED"
    assert any("SHA-256" in r for r in report["blocked_reasons"])


# ---------------------------------------------------------------------------
# 10-13: Missing reference key per layer -> BLOCKED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", list(_LAYER_ORDER))
def test_preflight_missing_reference_key_per_layer_blocked(tmp_path, layer):
    """10-13: a missing key for any one of the four layers -> BLOCKED."""
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    key_to_remove = contract["reference_members"]["ui3"][layer]
    data = dict(npz._data)
    del data[key_to_remove]
    npz_missing = _FakeNPZ(data)
    with pytest.raises(PreflightError, match="not present in evidence NPZ"):
        preflight_fixture("ui3", contract, tmp_path, npz_missing)


# ---------------------------------------------------------------------------
# 14: Wrong shape per layer -> BLOCKED in preflight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", list(_LAYER_ORDER))
def test_preflight_wrong_shape_per_layer_blocked(tmp_path, layer):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    key = contract["reference_members"]["uniform"][layer]
    data = dict(npz._data)
    # Any shape different from what the layer expects.
    data[key] = np.zeros((3, 3, 3), dtype=data[key].dtype)
    npz_bad = _FakeNPZ(data)
    with pytest.raises(PreflightError, match="has shape"):
        preflight_fixture("uniform", contract, tmp_path, npz_bad)


# ---------------------------------------------------------------------------
# 15: Wrong dtype per layer -> BLOCKED in preflight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", list(_LAYER_ORDER))
def test_preflight_wrong_dtype_per_layer_blocked(tmp_path, layer):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    key = contract["reference_members"]["ui1"][layer]
    data = dict(npz._data)
    # Cast to a dtype different from the expected one, keeping shape intact.
    original = data[key]
    new_dtype = np.int32 if str(original.dtype) != "int32" else np.int64
    data[key] = original.astype(new_dtype)
    npz_bad = _FakeNPZ(data)
    with pytest.raises(PreflightError, match="has dtype"):
        preflight_fixture("ui1", contract, tmp_path, npz_bad)


def test_preflight_all_five_pass_with_matching_bytes_shapes_and_dtypes(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    for name in _EXPECTED_FIXTURE_NAMES:
        obs = preflight_fixture(name, contract, tmp_path, npz)
        assert obs["layers"]["legacy_model_input"]["ref_shape"] == [1, 256, 256, 3]
        assert obs["layers"]["legacy_raw_saliency"]["ref_shape"] == [512, 512]
        assert obs["layers"]["legacy_classification"]["ref_shape"] == [6]
        assert obs["layers"]["legacy_e2e_saliency"]["ref_shape"] == list(_SYNTHETIC_E2E_SHAPE)


def test_preflight_missing_fixture_file_blocked(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    os.remove(tmp_path / "ui2.png")
    with pytest.raises(PreflightError, match="Fixture file missing"):
        preflight_fixture("ui2", contract, tmp_path, npz)


def test_preflight_hash_mismatch_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    (tmp_path / "ui1.png").write_bytes(b"corrupted-bytes")
    with pytest.raises(PreflightError, match="SHA-256 mismatch"):
        preflight_fixture("ui1", contract, tmp_path, npz)


# ---------------------------------------------------------------------------
# 16: Model input not exactly identical -> FAILED (not BLOCKED)
# ---------------------------------------------------------------------------

def test_run_five_fixtures_model_input_mismatch_is_failed_not_blocked(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "ui1":
            tampered_input = result.model_input.copy()
            tampered_input[0, 0, 0, 0] += 1.0
            result = CapturedFullInference(
                model_input=tampered_input, raw_saliency=result.raw_saliency,
                classif=result.classif, e2e_saliency=result.e2e_saliency,
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["ui1"]["status"] == "FAILED"
    assert report["per_fixture"]["ui1"]["gate_0_model_input_identity"]["passed"] is False
    for name in ("lowcontrast", "ui2", "ui3", "uniform"):
        assert report["per_fixture"][name]["status"] == "PASS"


def test_gate_0_model_input_identity_exact_match_passes():
    arr = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    result = evaluate_gate_0_model_input_identity(arr, arr.copy(), {})
    assert result.passed is True


def test_gate_0_model_input_identity_mismatch_fails():
    arr = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    other = arr.copy()
    other[0, 0, 0, 0] += 1.0
    result = evaluate_gate_0_model_input_identity(arr, other, {})
    assert result.passed is False
    assert "GATE_0" in result.reason


# ---------------------------------------------------------------------------
# 17-19: Metric below threshold (raw / classification / e2e) -> FAILED
# ---------------------------------------------------------------------------

def test_run_five_fixtures_raw_metric_below_threshold_is_failed(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "ui2":
            rng = np.random.default_rng(999)
            result = CapturedFullInference(
                model_input=result.model_input,
                raw_saliency=rng.random((512, 512)).astype(np.float32),
                classif=result.classif, e2e_saliency=result.e2e_saliency,
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["ui2"]["status"] == "FAILED"
    assert report["per_fixture"]["ui2"]["gate_1_saliency"]["passed"] is False
    for name in ("lowcontrast", "ui1", "ui3", "uniform"):
        assert report["per_fixture"][name]["status"] == "PASS"


def test_run_five_fixtures_classification_out_of_bound_is_failed(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "ui3":
            bad_classif = result.classif.copy()
            bad_classif[0] += 1.0  # exceeds cli_abs_max=0.01
            result = CapturedFullInference(
                model_input=result.model_input, raw_saliency=result.raw_saliency,
                classif=bad_classif, e2e_saliency=result.e2e_saliency,
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["ui3"]["status"] == "FAILED"
    assert report["per_fixture"]["ui3"]["gate_2_classification"]["passed"] is False


def test_run_five_fixtures_e2e_metric_below_threshold_is_failed(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "uniform":
            rng = np.random.default_rng(4242)
            result = CapturedFullInference(
                model_input=result.model_input, raw_saliency=result.raw_saliency,
                classif=result.classif,
                e2e_saliency=rng.random(_SYNTHETIC_E2E_SHAPE).astype(np.float32),
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["uniform"]["status"] == "FAILED"
    assert report["per_fixture"]["uniform"]["gate_3_e2e_saliency"]["passed"] is False


# ---------------------------------------------------------------------------
# 20: E2E shape wrong -> FAILED (evaluated at gate_3, not BLOCKED at preflight,
# since preflight only validates the *reference* array's shape).
# ---------------------------------------------------------------------------

def test_run_five_fixtures_e2e_shape_wrong_is_failed_not_blocked(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "lowcontrast":
            result = CapturedFullInference(
                model_input=result.model_input, raw_saliency=result.raw_saliency,
                classif=result.classif,
                e2e_saliency=np.zeros((5, 5), dtype=np.float32),  # wrong shape
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["lowcontrast"]["status"] == "FAILED"
    assert report["per_fixture"]["lowcontrast"]["gate_3_e2e_saliency"]["passed"] is False
    assert "SHAPE_MISMATCH" in report["per_fixture"]["lowcontrast"]["gate_3_e2e_saliency"]["reason"]


# ---------------------------------------------------------------------------
# 21: Determinism deviation in each of the four layers -> FAILED
# ---------------------------------------------------------------------------

def _make_captured(rng_seed: int) -> CapturedFullInference:
    rng = np.random.default_rng(rng_seed)
    return CapturedFullInference(
        model_input=rng.random((1, 4, 4, 3)).astype(np.float32),
        raw_saliency=rng.random((8, 8)).astype(np.float32),
        classif=rng.random((6,)).astype(np.float64),
        e2e_saliency=rng.random((5, 5)).astype(np.float32),
        call_count=1, restored=True,
    )


@pytest.mark.parametrize("layer", list(_LAYER_ORDER))
def test_determinism_repeat_deviation_per_layer_fails(layer):
    first = _make_captured(1)
    second = _make_captured(1)
    field_name = {
        "legacy_model_input": "model_input",
        "legacy_raw_saliency": "raw_saliency",
        "legacy_classification": "classif",
        "legacy_e2e_saliency": "e2e_saliency",
    }[layer]
    tampered = getattr(second, field_name).copy()
    tampered.flat[0] += 1.0
    setattr(second, field_name, tampered)

    result = evaluate_determinism_repeat("fixture", first, second, 0.0)
    assert result.passed is False
    assert layer.upper() in result.reason


def test_determinism_repeat_identical_captures_pass():
    first = _make_captured(7)
    second = _make_captured(7)
    result = evaluate_determinism_repeat("fixture", first, second, 0.0)
    assert result.passed is True


def test_run_five_fixtures_determinism_failure_is_failed_not_blocked(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)
    call_counter = {"lowcontrast": 0}

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "lowcontrast":
            call_counter["lowcontrast"] += 1
            if call_counter["lowcontrast"] == 2:
                tampered_raw = result.raw_saliency.copy()
                tampered_raw[0, 0] += 0.001
                result = CapturedFullInference(
                    model_input=result.model_input, raw_saliency=tampered_raw,
                    classif=result.classif, e2e_saliency=result.e2e_saliency,
                    call_count=1, restored=True,
                )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["overall_verdict"] == "FAILED"
    assert report["per_fixture"]["lowcontrast"]["status"] == "FAILED"
    assert report["per_fixture"]["lowcontrast"]["determinism_repeat"]["passed"] is False


# ---------------------------------------------------------------------------
# 22: Full synthetic PASS
# ---------------------------------------------------------------------------

def test_run_five_fixtures_all_pass(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    inference_fn = _perfect_inference_fn_factory(contract, npz)

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=inference_fn
    )
    assert report["overall_verdict"] == "PASS"
    assert report["fixture_order"] == list(_EXPECTED_FIXTURE_NAMES)
    assert len(report["per_fixture"]) == 5
    for name in _EXPECTED_FIXTURE_NAMES:
        entry = report["per_fixture"][name]
        assert entry["status"] == "PASS"
        assert entry["gate_0_model_input_identity"]["passed"] is True
        assert entry["gate_1_saliency"]["passed"] is True
        assert entry["gate_2_classification"]["passed"] is True
        assert entry["gate_3_e2e_saliency"]["passed"] is True
        assert entry["determinism_repeat"]["passed"] is True


def test_run_five_fixtures_execution_order_is_deterministic(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    inference_fn = _perfect_inference_fn_factory(contract, npz)
    call_order = []

    def _wrapped(fixture_name, fixture_path):
        call_order.append(fixture_name)
        return inference_fn(fixture_name, fixture_path)

    run_five_fixtures(contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_wrapped)
    expected = []
    for name in _EXPECTED_FIXTURE_NAMES:
        expected.extend([name, name])
    assert call_order == expected


# ---------------------------------------------------------------------------
# 23: Combined BLOCKED + FAILED -> overall BLOCKED
# ---------------------------------------------------------------------------

def test_run_five_fixtures_combined_blocked_and_failed_yields_overall_blocked(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    npz = _make_fake_npz_for_contract(contract)
    perfect_fn = _perfect_inference_fn_factory(contract, npz)
    os.remove(tmp_path / "ui2.png")  # forces ui2 -> BLOCKED at preflight

    def _inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        result = perfect_fn(fixture_name, fixture_path)
        if fixture_name == "ui3":
            rng = np.random.default_rng(13)
            result = CapturedFullInference(
                model_input=result.model_input,
                raw_saliency=rng.random((512, 512)).astype(np.float32),  # forces FAILED
                classif=result.classif, e2e_saliency=result.e2e_saliency,
                call_count=1, restored=True,
            )
        return result

    report = run_five_fixtures(
        contract=contract, fixtures_root=tmp_path, npz=npz, inference_fn=_inference_fn
    )
    assert report["per_fixture"]["ui2"]["status"] == "BLOCKED"
    assert report["per_fixture"]["ui3"]["status"] == "FAILED"
    assert report["overall_verdict"] == "BLOCKED"


# ---------------------------------------------------------------------------
# 24: A missing/incomplete per-fixture result set can never produce PASS.
# ---------------------------------------------------------------------------

def test_aggregate_overall_verdict_incomplete_per_fixture_cannot_pass():
    per_fixture = {
        "lowcontrast": {"status": "PASS"},
        "ui1": {"status": "PASS"},
        "ui2": {"status": "PASS"},
        "ui3": {"status": "PASS"},
        # "uniform" entry missing entirely.
    }
    verdict = _aggregate_overall_verdict(per_fixture, _EXPECTED_FIXTURE_NAMES)
    assert verdict != "PASS"
    assert verdict == "BLOCKED"


def test_aggregate_overall_verdict_empty_per_fixture_cannot_pass():
    verdict = _aggregate_overall_verdict({}, _EXPECTED_FIXTURE_NAMES)
    assert verdict != "PASS"
    assert verdict == "BLOCKED"


def test_aggregate_overall_verdict_all_present_and_passed_is_pass():
    per_fixture = {name: {"status": "PASS"} for name in _EXPECTED_FIXTURE_NAMES}
    assert _aggregate_overall_verdict(per_fixture, _EXPECTED_FIXTURE_NAMES) == "PASS"


def test_aggregate_overall_verdict_blocked_takes_precedence_over_failed():
    per_fixture = {name: {"status": "PASS"} for name in _EXPECTED_FIXTURE_NAMES}
    per_fixture["ui1"] = {"status": "FAILED"}
    per_fixture["ui2"] = {"status": "BLOCKED"}
    assert _aggregate_overall_verdict(per_fixture, _EXPECTED_FIXTURE_NAMES) == "BLOCKED"


def test_run_five_fixtures_rejects_contract_with_wrong_fixture_count():
    contract = {
        "fixtures": {"lowcontrast": {"filename": "lowcontrast.png", "sha256": "a" * 64}},
        "reference_members": {},
        "frozen_thresholds": {
            "pearson_min": 0.999, "spearman_min": 0.99,
            "ssim_min": 0.99, "repeatability_abs_max": 0.0,
        },
        "gate_definitions": {
            "gate_0_model_input_identity": {"requirements": {}},
            "gate_1_saliency": {"requirements": {}},
            "gate_2_classification": {"requirements": {}},
            "gate_3_e2e_saliency": {"requirements": {}},
            "determinism_repeat": {"requirements": {}},
        },
        "expected_reference_shapes": {
            "legacy_model_input": {"shape": [1, 256, 256, 3], "dtype": "float32"},
            "legacy_raw_saliency": {"shape": [512, 512], "dtype": "float32"},
            "legacy_classification": {"shape": [6], "dtype": "float64"},
            "legacy_e2e_saliency": {"dtype": "float32", "per_fixture_shape": {}},
        },
    }
    with pytest.raises(ContractError, match="exactly 5"):
        run_five_fixtures(
            contract=contract, fixtures_root=Path("."), npz=_FakeNPZ({}),
            inference_fn=lambda *a: (None, None),
        )


# ---------------------------------------------------------------------------
# 25: Package- and script-style import share the same PreflightError class.
# ---------------------------------------------------------------------------

def test_package_and_script_import_share_same_preflight_error_class():
    """25: the harness module's own lazily-imported runner
    (_get_runner_module()) must yield the identical PreflightError class
    object regardless of whether the HARNESS itself was imported
    package-style or via a fresh script-style sys.path insertion, so
    exceptions raised via one harness import style are catchable via the
    other. Both harness instances' _get_runner_module() prefer the
    package-style `import stage1.tools.umsi_step2b_gate_runner` first,
    which is cached once in sys.modules regardless of which harness
    instance requested it (importing umsi_step2b_gate_runner directly via
    two different qualified names, e.g. package-style AND flat-style in
    the same process, would instead create two distinct module objects
    with two distinct exception classes -- a separate, well-known Python
    duplicate-module hazard that this harness design avoids by always
    preferring the package-style import path)."""
    tools_dir = str(Path(_repo_root) / "stage1" / "tools")
    proc = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            f"sys.path.insert(0, {tools_dir!r}); "
            f"sys.path.insert(0, {_repo_root!r}); "
            "from stage1.tools.umsi_five_fixture_gate_harness import "
            "_get_runner_module as pkg_get; "
            "import umsi_five_fixture_gate_harness as script_mod; "
            "pkg_runner = pkg_get(); "
            "script_runner = script_mod._get_runner_module(); "
            "assert script_runner.PreflightError is pkg_runner.PreflightError, "
            "'PreflightError identity mismatch'; "
            "print('IDENTICAL')"
        )],
        capture_output=True, text=True, cwd=str(tools_dir),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IDENTICAL" in proc.stdout


def test_harness_lazy_runner_module_shares_preflight_error_identity():
    """The harness's lazily-imported runner module (_get_runner_module())
    must expose the SAME PreflightError class object as a direct
    `from stage1.tools.umsi_step2b_gate_runner import PreflightError`
    import, so that pytest.raises(PreflightError) elsewhere in this file
    correctly catches exceptions raised internally by the harness (e.g. in
    preflight_fixture(), which obtains PreflightError via
    _get_runner_module() rather than a module-level import)."""
    runner_mod = _get_runner_module()
    assert runner_mod.PreflightError is PreflightError
    assert runner_mod.evaluate_gate_b is evaluate_gate_b
    assert runner_mod.evaluate_gate_d is evaluate_gate_d


# ---------------------------------------------------------------------------
# 26-27: Pure import / preflight-only mode never load TensorFlow/Keras/tf_keras
# ---------------------------------------------------------------------------

def test_module_import_does_not_load_tensorflow_or_keras():
    """26: a fresh process's plain import of the harness module must not
    import tensorflow, keras, or tf_keras."""
    proc = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            f"sys.path.insert(0, {_repo_root!r}); "
            "import stage1.tools.umsi_five_fixture_gate_harness; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('tensorflow', 'keras', 'tf_keras')]; "
            "print('BAD_MODULES=' + repr(bad))"
        )],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BAD_MODULES=[]" in proc.stdout


def test_provenance_only_mode_does_not_load_tensorflow_or_keras(tmp_path):
    """27: running --provenance-only end to end (with a deliberately wrong
    weights basename, so it short-circuits to BLOCKED before ever calling
    np.load on a real NPZ) must not import tensorflow, keras, or tf_keras."""
    contract, weights_path, evidence_path = _minimal_provenance_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    wrong_weights = weights_path.with_name("definitely_wrong_name.bin")
    weights_path.rename(wrong_weights)

    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {_repo_root!r})\n"
        "from pathlib import Path\n"
        "from stage1.tools.umsi_five_fixture_gate_harness import run_provenance_only_preflight\n"
        f"contract = json.load(open({str(contract_path)!r}))\n"
        "report = run_provenance_only_preflight(\n"
        f"    contract=contract, fixtures_root=Path({str(tmp_path / 'fixtures')!r}),\n"
        f"    weights_path=Path({str(wrong_weights)!r}),\n"
        f"    evidence_npz_path=Path({str(evidence_path)!r}),\n"
        ")\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras', 'tf_keras')]\n"
        "print('VERDICT=' + report['overall_verdict'])\n"
        "print('BAD_MODULES=' + repr(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERDICT=BLOCKED" in proc.stdout
    assert "BAD_MODULES=[]" in proc.stdout


# ---------------------------------------------------------------------------
# 28: Historical lowcontrast-only runner remains importable and functional.
# ---------------------------------------------------------------------------

def test_existing_lowcontrast_only_runner_remains_functionally_usable():
    """28: evaluate_gate_b / evaluate_gate_d from the historical single-
    fixture runner remain importable AND functionally correct (not just
    hasattr) on tiny synthetic identical arrays."""
    identical = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    gate_b_def = {
        "requirements": {
            "pearson": {"value": 0.999}, "spearman": {"value": 0.99},
            "windowed_ssim": {
                "value": 0.99,
                "params": {
                    "K1": 0.01, "K2": 0.03, "win_size": 3,
                    "boundary": "reflect", "data_range": 1.0,
                },
            },
        }
    }
    result_b = evaluate_gate_b(identical, identical.copy(), gate_b_def)
    assert result_b.passed is True

    classif = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.1], dtype=np.float64)
    gate_d_def = {
        "requirements": {
            "same_shape": True, "finite_values": True,
            "max_abs_diff_float64": {"operator": "<=", "value": 0.01},
        }
    }
    result_d = evaluate_gate_d(classif, classif.copy(), gate_d_def)
    assert result_d.passed is True


# ---------------------------------------------------------------------------
# 29: Every individual frozen threshold is rejected when tampered.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "threshold_key,tampered_value",
    [
        ("pearson_min", 0.9),
        ("spearman_min", 0.5),
        ("ssim_min", 0.5),
        ("repeatability_abs_max", 0.01),
    ],
)
def test_contract_tampered_threshold_rejected(tmp_path, threshold_key, tampered_value):
    contract = _make_synthetic_contract(tmp_path)
    contract["frozen_thresholds"][threshold_key] = tampered_value
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match=threshold_key):
        load_and_validate_five_fixture_contract(bad_path)


# ---------------------------------------------------------------------------
# 30: Manipulation of the input or raw boundary shape is rejected.
# ---------------------------------------------------------------------------

def test_contract_wrong_input_tensor_shape_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    contract["model_boundaries"]["input_tensor_shape"] = [1, 128, 128, 3]
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="input_tensor_shape"):
        load_and_validate_five_fixture_contract(bad_path)


def test_contract_wrong_raw_decoder_output_shape_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    contract["model_boundaries"]["raw_decoder_output_shape"] = [256, 256, 1]
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="raw_decoder_output_shape"):
        load_and_validate_five_fixture_contract(bad_path)


# ---------------------------------------------------------------------------
# Additional contract schema guards (missing fixture / duplicate / count)
# ---------------------------------------------------------------------------

def test_contract_missing_fixture_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    del contract["fixtures"]["ui3"]
    del contract["reference_members"]["ui3"]
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError):
        load_and_validate_five_fixture_contract(bad_path)


def test_contract_duplicate_fixture_name_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    contract["fixtures"]["ui1_dup"] = contract["fixtures"]["ui1"]
    del contract["fixtures"]["uniform"]
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="do not match the expected set"):
        load_and_validate_five_fixture_contract(bad_path)


def test_contract_wrong_fixture_count_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    del contract["fixtures"]["ui2"]
    del contract["reference_members"]["ui2"]
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="exactly 5 fixtures"):
        load_and_validate_five_fixture_contract(bad_path)


# ---------------------------------------------------------------------------
# Provenance / execution-safety corrections (audit findings #3, #4, #6, #7,
# #10): repository HEAD, production-source and legacy-runner identity,
# candidate-harness self-hash, git-status allowlist, runtime versions, and
# output-directory safety. All scenarios below use a throwaway git
# repository created under pytest's tmp_path (never the real thesis
# repository) and/or direct unit calls to the new provenance helpers. None
# of these tests imports or executes the real UMSI model.
# ---------------------------------------------------------------------------

def _init_fake_repo(tmp_path: Path, suffix: str = "") -> Path:
    """Initialize a throwaway git repository under tmp_path (never the real
    thesis repository) containing a fake production source and a fake
    legacy runner file, then commit them. Returns the repo root."""
    repo = tmp_path / f"fake_repo{suffix}"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "saliency").mkdir()
    (repo / "stage1" / "tools").mkdir(parents=True)
    (repo / "saliency" / "umsi_model.py").write_bytes(b"# fake production source\n")
    (repo / "stage1" / "tools" / "umsi_step2b_gate_runner.py").write_bytes(
        b"# fake legacy runner\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _write_real_npz_for_contract(contract: Dict[str, Any], path: Path) -> None:
    """Write a real, loadable .npz file on disk containing matching
    reference arrays for every fixture/layer in the given synthetic
    contract (using the same deterministic arrays as _ref_arrays_for)."""
    data: Dict[str, np.ndarray] = {}
    for name in _EXPECTED_FIXTURE_NAMES:
        members = contract["reference_members"][name]
        refs = _ref_arrays_for(name)
        for layer, arr in refs.items():
            data[members[layer]] = arr
    np.savez(str(path), **data)


# load_and_validate_five_fixture_contract hardcodes the real pinned
# checkpoint size (120093896 bytes) as a schema requirement (retained
# unchanged; not weakened by this task). Any contract used with the full
# schema loader must declare exactly this weights_size_bytes, and any
# weights file used with run_frozen_five_fixture_experiment's own size
# check must be exactly this many bytes on disk. Written as a sparse
# (all-zero-content) file via truncate() so no real 120 MB of disk I/O is
# required per test; its SHA-256 is precomputed once at module import time.
_ZERO_WEIGHTS_SHA256 = hashlib.sha256(b"\x00" * _EXPECTED_WEIGHTS_SIZE_BYTES).hexdigest()


def _write_correctly_sized_weights_file(weights_path: Path) -> None:
    with open(weights_path, "wb") as fh:
        fh.truncate(_EXPECTED_WEIGHTS_SIZE_BYTES)


def _build_frozen_scenario(tmp_path: Path) -> Dict[str, Any]:
    """Build a complete, valid scenario for run_frozen_five_fixture_experiment
    tests: a throwaway git repo (with fake production-source and legacy-
    runner files), a schema-valid contract matching that repo's real HEAD
    and real file hashes, a real loadable evidence NPZ, dummy weights, real
    fixture files, and a fresh output root. Every negative test starts from
    this valid baseline and corrupts exactly one aspect."""
    repo = _init_fake_repo(tmp_path)
    head = _git_head(repo)
    harness_sha = _actual_harness_sha256()

    fixtures_dir = tmp_path / "frozen_fixtures"
    fixtures_dir.mkdir()
    contract = _make_synthetic_contract(fixtures_dir)

    # load_and_validate_five_fixture_contract hardcodes the real pinned
    # checkpoint size (120093896 bytes); a dummy weights file must match
    # that size exactly to pass contract validation and the harness's own
    # weights-size check. Written as a sparse (all-zero-content) file so no
    # real 120 MB of disk I/O is required per test.
    weights_path = tmp_path / "frozen_weights.bin"
    _write_correctly_sized_weights_file(weights_path)
    contract["weights_filename"] = weights_path.name
    contract["weights_size_bytes"] = _EXPECTED_WEIGHTS_SIZE_BYTES
    contract["weights_sha256"] = _ZERO_WEIGHTS_SHA256

    evidence_path = tmp_path / "frozen_evidence.npz"
    _write_real_npz_for_contract(contract, evidence_path)
    contract["evidence_npz_filename"] = evidence_path.name
    contract["evidence_npz_sha256"] = _sha256_bytes(evidence_path.read_bytes())

    prod_source_bytes = (repo / "saliency" / "umsi_model.py").read_bytes()
    runner_bytes = (repo / "stage1" / "tools" / "umsi_step2b_gate_runner.py").read_bytes()

    contract["expected_repository_head"] = head
    contract["production_source_identity"] = {
        "path": "saliency/umsi_model.py",
        "sha256": _sha256_bytes(prod_source_bytes),
    }
    contract["reused_legacy_runner_identity"] = {
        "path": "stage1/tools/umsi_step2b_gate_runner.py",
        "sha256": _sha256_bytes(runner_bytes),
    }
    contract["candidate_harness_identity"] = {
        "path": "stage1/tools/umsi_five_fixture_gate_harness.py",
        "sha256": harness_sha,
    }
    contract["allowed_untracked_paths"] = [
        "stage1/evidence/umsi_five_fixture_gate_contract.json",
        "stage1/tools/umsi_five_fixture_gate_harness.py",
        "tests/test_umsi_five_fixture_gate_harness.py",
    ]
    contract["runtime_identity"] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": {"numpy": importlib.metadata.version("numpy")},
    }

    output_root = tmp_path / "output_root"
    output_root.mkdir()
    contract_path = tmp_path / "frozen_contract.json"

    # execution.launcher/predecessor_incident: built to genuinely match how
    # _run_frozen_experiment_subprocess actually launches this scenario
    # (cwd=repo, PYTHONPATH stripped, PYTHONDONTWRITEBYTECODE=1 forced, the
    # harness imported normally as a real package so its own __spec__.name
    # is set correctly for module-form), so the new launcher preflight
    # (added ahead of every other provenance check) passes cleanly by
    # default and every negative test below still reaches exactly the one
    # deviation it introduces.
    contract["execution"] = {
        "run_directory_name": f"five_fixture_run_test_exec_{head[:8]}",
        "run_directory_name_prefix_requirement": "five_fixture_run_",
        "launcher": {
            "mode": "module",
            "python_executable": sys.executable,
            "interpreter_flags": ["-B"],
            "module": "stage1.tools.umsi_five_fixture_gate_harness",
            "working_directory": str(repo),
            "required_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "pythonpath_required_state": "unset",
            "action": "execute_frozen_experiment",
            "contract_path": str(contract_path),
            "output_root": str(output_root),
            "weights_path": str(weights_path),
            "evidence_npz_path": str(evidence_path),
            "fixtures_root": str(fixtures_dir),
        },
        "predecessor_incident": {
            "consumed_run_directory_name": f"five_fixture_run_test_consumed_{head[:8]}",
            "failure_classification": (
                "infrastructure_bootstrap_import_path_failure_before_production_model_import"
            ),
            "execution_attempted": True,
            "identity_permanently_consumed": True,
            "run_directory_created": False,
            "production_model_imported": False,
            "weights_loaded": False,
            "inference_executed": False,
            "scientific_verdict_reached": False,
        },
    }

    contract_path.write_text(json.dumps(contract))

    return {
        "repo": repo,
        "head": head,
        "harness_sha": harness_sha,
        "contract": contract,
        "contract_path": contract_path,
        "weights_path": weights_path,
        "evidence_path": evidence_path,
        "fixtures_dir": fixtures_dir,
        "output_root": output_root,
    }


_FAKE_SALIENCY_STUB_ROOT = Path(tempfile.mkdtemp(prefix="fake_saliency_stub_"))
_fake_saliency_pkg_dir = _FAKE_SALIENCY_STUB_ROOT / "saliency"
_fake_saliency_pkg_dir.mkdir(parents=True, exist_ok=True)
(_fake_saliency_pkg_dir / "__init__.py").write_text("")
(_fake_saliency_pkg_dir / "umsi_model.py").write_text(
    "class UMSIPlus:\n"
    "    def __init__(self, *a, **kw):\n"
    "        raise RuntimeError(\n"
    "            'SAFETY STUB: the real UMSIPlus model must never be "
    "constructed by this test suite'\n"
    "        )\n"
)
# Defense-in-depth safety net: this stub directory is placed FIRST on
# sys.path (ahead of the real repository root) in every subprocess launched
# by _run_frozen_experiment_subprocess, below. This guarantees that even if
# a provenance/preflight check has a latent bug and fails to block a
# corrupted scenario, `from saliency.umsi_model import UMSIPlus` resolves to
# this harmless stub (which immediately raises RuntimeError) instead of the
# real production saliency.umsi_model module, which IS otherwise importable
# in this environment (the real project root is also on sys.path) and would
# otherwise import TensorFlow/Keras and construct the real model -- both
# forbidden by this task's hard safety constraints.


def _default_frozen_args(scenario: Dict[str, Any]) -> Dict[str, str]:
    return {
        "contract": str(scenario["contract_path"]),
        "contract_sha": _sha256_bytes(scenario["contract_path"].read_bytes()),
        "expected_harness_sha": scenario["harness_sha"],
        "repo": str(scenario["repo"]),
        "weights": str(scenario["weights_path"]),
        "evidence_npz": str(scenario["evidence_path"]),
        "fixtures_root": str(scenario["fixtures_dir"]),
        "output_root": str(scenario["output_root"]),
    }


def _run_frozen_experiment_subprocess(args_dict: Dict[str, str]) -> Tuple[int, str, str]:
    """Run run_frozen_five_fixture_experiment(...) in a fresh subprocess
    (so no in-process module cache leaks between test cases) and return
    (returncode, stdout, stderr). stdout additionally reports, as an import
    sentinel, whether tensorflow/keras/tf_keras/saliency and the legacy
    runner module were present in sys.modules at exit. The subprocess's own
    exit code is set to exactly the function's own return value via
    sys.exit(rc), printed only after the sentinel info so nothing is lost.

    cwd is pinned to the scenario's own fake repository root and PYTHONPATH
    is stripped (with PYTHONDONTWRITEBYTECODE forced to "1"), matching
    exactly the execution.launcher.working_directory / required_environment
    / pythonpath_required_state values _build_frozen_scenario pins in the
    contract, so the new launcher preflight passes cleanly for the shared
    valid baseline and every test below still reaches exactly the one
    deviation it introduces."""
    ns_lines = ",\n".join(f"    {k}={v!r}" for k, v in args_dict.items())
    code = (
        "import sys, argparse\n"
        f"sys.path.insert(0, {_repo_root!r})\n"
        f"sys.path.insert(0, {str(_FAKE_SALIENCY_STUB_ROOT)!r})\n"
        "from stage1.tools.umsi_five_fixture_gate_harness import "
        "run_frozen_five_fixture_experiment\n"
        f"args = argparse.Namespace(\n{ns_lines}\n)\n"
        "rc = run_frozen_five_fixture_experiment(args)\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('tensorflow', 'keras', 'tf_keras', 'saliency')]\n"
        "runner_present = any(\n"
        "    m in sys.modules\n"
        "    for m in ('stage1.tools.umsi_step2b_gate_runner', 'umsi_step2b_gate_runner')\n"
        ")\n"
        "print('RC=' + str(rc))\n"
        "print('BAD_MODULES=' + repr(bad))\n"
        "print('RUNNER_IMPORTED=' + str(runner_present))\n"
        "sys.exit(rc)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=args_dict["repo"],
        env=env,
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_frozen_scenario_baseline_passes_all_individual_provenance_checks(tmp_path):
    """Sanity check: the shared valid baseline scenario's repository HEAD,
    git status, production-source identity, legacy-runner identity,
    runtime versions, output-directory safety, contract schema, and
    per-fixture preflight all pass. Verified via DIRECT calls to the
    individual provenance helpers -- never through
    run_frozen_five_fixture_experiment() itself, which would proceed all
    the way to importing and constructing the real production model once
    every check passes (forbidden by this task's hard safety constraints,
    since the real saliency package IS importable in this environment).
    This proves the baseline itself is not accidentally already broken, so
    every negative test below is known to be testing exactly the one
    deviation it introduces."""
    scenario = _build_frozen_scenario(tmp_path)
    contract = scenario["contract"]
    repo = scenario["repo"]

    assert _git_rev_parse_head(repo) == contract["expected_repository_head"]
    assert _check_git_status_against_allowlist(
        _git_status_porcelain(repo), set(contract["allowed_untracked_paths"])
    ) == []
    _verify_worktree_and_blob_identity(
        repo_dir=repo,
        rel_path=contract["production_source_identity"]["path"],
        worktree_path=repo / contract["production_source_identity"]["path"],
        expected_sha256=contract["production_source_identity"]["sha256"],
    )  # must not raise
    _verify_worktree_and_blob_identity(
        repo_dir=repo,
        rel_path=contract["reused_legacy_runner_identity"]["path"],
        worktree_path=repo / contract["reused_legacy_runner_identity"]["path"],
        expected_sha256=contract["reused_legacy_runner_identity"]["sha256"],
    )  # must not raise
    _verify_runtime_versions(contract["runtime_identity"])  # must not raise
    validated_run_dir = _verify_output_directory_safety(
        output_root=scenario["output_root"],
        run_dir_name=contract["execution"]["run_directory_name"],
        required_prefix=contract["execution"]["run_directory_name_prefix_requirement"],
        protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
    )  # must not raise
    # Positive proof: the unchanged, valid, contract-pinned run directory
    # name resolves to exactly one direct child of the canonical
    # (resolved) output root -- never a nested path, never outside it.
    assert validated_run_dir == (
        scenario["output_root"].resolve() / contract["execution"]["run_directory_name"]
    )
    assert validated_run_dir.parent == scenario["output_root"].resolve()

    loaded = load_and_validate_five_fixture_contract(scenario["contract_path"])
    npz = np.load(str(scenario["evidence_path"]), allow_pickle=True)
    for name in _EXPECTED_FIXTURE_NAMES:
        preflight_fixture(name, loaded, scenario["fixtures_dir"], npz)  # must not raise


def test_frozen_experiment_wrong_head_blocked(tmp_path):
    """Corrupts expected_repository_head only, while keeping
    run_directory_name / consumed_run_directory_name's own HEAD-derived
    naming-convention suffix internally consistent with the (bogus) new
    head, so contract SCHEMA validation still passes and the failure is
    correctly attributed to the RUNTIME git HEAD check instead."""
    scenario = _build_frozen_scenario(tmp_path)
    bogus_head = "0" * 40
    scenario["contract"]["expected_repository_head"] = bogus_head
    scenario["contract"]["execution"]["run_directory_name"] = (
        f"five_fixture_run_test_exec_{bogus_head[:8]}"
    )
    scenario["contract"]["execution"]["predecessor_incident"]["consumed_run_directory_name"] = (
        f"five_fixture_run_test_consumed_{bogus_head[:8]}"
    )
    scenario["contract_path"].write_text(json.dumps(scenario["contract"]))
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "HEAD mismatch" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_harness_hash_cli_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    args = _default_frozen_args(scenario)
    args["expected_harness_sha"] = "1" * 64
    rc, stdout, stderr = _run_frozen_experiment_subprocess(args)
    assert rc == 1
    assert "BLOCKED" in stderr and "harness SHA-256 mismatch" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_contract_hash_cli_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    args = _default_frozen_args(scenario)
    args["contract_sha"] = "2" * 64
    rc, stdout, stderr = _run_frozen_experiment_subprocess(args)
    assert rc == 1
    assert "BLOCKED" in stderr and "contract SHA-256 mismatch" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_production_source_worktree_hash_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    scenario["contract"]["production_source_identity"]["sha256"] = "3" * 64
    scenario["contract_path"].write_text(json.dumps(scenario["contract"]))
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "production source identity" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_production_source_differs_from_blob_at_head_blocked(tmp_path):
    """A tracked production-source file whose worktree content diverges
    from the committed blob at HEAD (without being re-committed) is
    rejected by the earlier git-status-vs-allowlist check (any tracked
    worktree change is fail-closed there, before the three-way identity
    check is ever reached) -- it never reaches model import either way."""
    scenario = _build_frozen_scenario(tmp_path)
    prod_path = scenario["repo"] / "saliency" / "umsi_model.py"
    prod_path.write_bytes(b"# modified after commit, never re-committed\n")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "git status violation" in stderr
    assert "saliency/umsi_model.py" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_production_source_hash_blocked(tmp_path):
    """The pinned production-source SHA-256 not matching the actual
    (untouched, git-clean) file content is rejected by the three-way
    worktree==blob==contract identity check."""
    scenario = _build_frozen_scenario(tmp_path)
    scenario["contract"]["production_source_identity"]["sha256"] = "5" * 64
    scenario["contract_path"].write_text(json.dumps(scenario["contract"]))
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "production source identity" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_legacy_runner_hash_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    scenario["contract"]["reused_legacy_runner_identity"]["sha256"] = "4" * 64
    scenario["contract_path"].write_text(json.dumps(scenario["contract"]))
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "legacy runner identity" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_legacy_runner_differs_from_blob_at_head_blocked(tmp_path):
    """A tracked legacy-runner file whose worktree content diverges from
    the committed blob at HEAD (without being re-committed) is rejected by
    the earlier git-status-vs-allowlist check (any tracked worktree change
    is fail-closed there, before the three-way identity check is ever
    reached) -- it never reaches model import either way."""
    scenario = _build_frozen_scenario(tmp_path)
    runner_path = scenario["repo"] / "stage1" / "tools" / "umsi_step2b_gate_runner.py"
    runner_path.write_bytes(b"# modified after commit, never re-committed\n")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "git status violation" in stderr
    assert "stage1/tools/umsi_step2b_gate_runner.py" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_staged_change_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    repo = scenario["repo"]
    (repo / "new_staged_file.txt").write_bytes(b"staged but not committed\n")
    subprocess.run(["git", "add", "new_staged_file.txt"], cwd=repo, check=True)
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "git status violation" in stderr
    assert "staged change" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_tracked_unstaged_change_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    repo = scenario["repo"]
    (repo / "saliency" / "umsi_model.py").write_bytes(b"# unstaged tracked modification\n")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "git status violation" in stderr
    assert "tracked worktree change" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_unexpected_untracked_file_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    repo = scenario["repo"]
    (repo / "stray_untracked_file.txt").write_bytes(b"not in the allowlist\n")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "git status violation" in stderr
    assert "unexpected untracked path" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_runtime_version_mismatch_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    scenario["contract"]["runtime_identity"]["python_version"] = "0.0.0"
    scenario["contract_path"].write_text(json.dumps(scenario["contract"]))
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "runtime version check failed" in stderr
    # The legacy runner is imported before the runtime-version check, but
    # the production model must still never be imported.
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_weights_hash_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    # Corrupt content while preserving the exact pinned file size, so this
    # specifically exercises the SHA-256 mismatch check rather than the
    # (separate) file-size check.
    with open(scenario["weights_path"], "r+b") as fh:
        fh.write(b"\x01" * 1024)
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "weights file SHA-256 mismatch" in stderr
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_evidence_npz_hash_blocked(tmp_path):
    scenario = _build_frozen_scenario(tmp_path)
    scenario["evidence_path"].write_bytes(b"tampered-evidence-bytes")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "evidence NPZ SHA-256 mismatch" in stderr
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_wrong_fixture_hash_blocked(tmp_path):
    """Wrong fixture (reference) hash must block before the production
    model is imported, even though it is only detected during per-fixture
    preflight, which runs after the legacy runner has already been
    imported (its own identity was already verified in an earlier step)."""
    scenario = _build_frozen_scenario(tmp_path)
    (scenario["fixtures_dir"] / "ui1.png").write_bytes(b"corrupted-fixture-bytes")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "fixture preflight failed for 'ui1'" in stderr
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_pre_existing_run_directory_blocked(tmp_path):
    """A pre-existing run directory (whether from a genuinely earlier
    attempt or any other cause) blocks before the legacy runner or the
    production model is imported; this is also the mechanism that makes a
    second execution attempting to reuse the same execution ID fail."""
    scenario = _build_frozen_scenario(tmp_path)
    run_dir = scenario["output_root"] / scenario["contract"]["execution"]["run_directory_name"]
    run_dir.mkdir()
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "output directory check failed" in stderr
    assert "already exists" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_pre_existing_file_at_run_directory_path_blocked(tmp_path):
    """Phase 2: a pre-existing regular file (not a directory) at the final
    run-directory path blocks before the legacy runner or the production
    model is imported."""
    scenario = _build_frozen_scenario(tmp_path)
    run_dir_path = scenario["output_root"] / scenario["contract"]["execution"]["run_directory_name"]
    run_dir_path.write_bytes(b"a plain file, not a directory")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "output directory check failed" in stderr
    assert "already exists" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_symlink_at_run_directory_path_blocked(tmp_path):
    """Phase 2: a valid symlink at the final run-directory path, resolving
    to a location outside the output root, blocks before the legacy
    runner or the production model is imported."""
    scenario = _build_frozen_scenario(tmp_path)
    outside_target = tmp_path / "outside_of_output_root"
    outside_target.mkdir()
    run_dir_path = scenario["output_root"] / scenario["contract"]["execution"]["run_directory_name"]
    run_dir_path.symlink_to(outside_target, target_is_directory=True)
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "output directory check failed" in stderr
    assert "already exists" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_broken_symlink_at_run_directory_path_blocked(tmp_path):
    """Phase 2: a broken (dangling) symlink at the final run-directory
    path blocks before the legacy runner or the production model is
    imported."""
    scenario = _build_frozen_scenario(tmp_path)
    run_dir_path = scenario["output_root"] / scenario["contract"]["execution"]["run_directory_name"]
    run_dir_path.symlink_to(tmp_path / "does_not_exist_target")
    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))
    assert rc == 1
    assert "BLOCKED" in stderr and "output directory check failed" in stderr
    assert "already exists" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout


def test_frozen_experiment_second_use_of_same_execution_id_blocked(tmp_path):
    """Simulates a first execution having already created (and partially
    populated) the run directory for a given execution ID; a second
    execution with the identical contract-pinned run_directory_name must
    fail before any model import, and the pre-existing partial directory
    and its contents must be left untouched (never deleted or overwritten)."""
    scenario = _build_frozen_scenario(tmp_path)
    run_dir = scenario["output_root"] / scenario["contract"]["execution"]["run_directory_name"]
    run_dir.mkdir()
    sentinel_path = run_dir / "partial_from_first_attempt.txt"
    sentinel_path.write_text("partial state from a previous execution attempt")

    rc, stdout, stderr = _run_frozen_experiment_subprocess(_default_frozen_args(scenario))

    assert rc == 1
    assert "BLOCKED" in stderr and "already exists" in stderr
    assert "RUNNER_IMPORTED=False" in stdout
    assert "BAD_MODULES=[]" in stdout
    # The pre-existing partial state must be preserved untouched.
    assert sentinel_path.is_file()
    assert sentinel_path.read_text() == "partial state from a previous execution attempt"


def test_contract_protected_run_directory_name_rejected_at_schema_time(tmp_path):
    """Item #10 / #15: a run_directory_name colliding with a protected
    P8/P9 evidence location name is rejected at contract-schema-validation
    time — an even earlier and stronger guarantee than the runtime
    output-directory check in run_frozen_five_fixture_experiment."""
    contract = _make_synthetic_contract(tmp_path)
    contract["execution"]["run_directory_name"] = "step2b_results_p8"
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="protected"):
        load_and_validate_five_fixture_contract(bad_path)


def test_contract_run_directory_name_wrong_prefix_rejected(tmp_path):
    contract = _make_synthetic_contract(tmp_path)
    contract["execution"]["run_directory_name"] = "not_the_required_prefix_run"
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="does not start with"):
        load_and_validate_five_fixture_contract(bad_path)


# ---------------------------------------------------------------------------
# Direct unit tests for the individual provenance helper functions (faster
# and more precise than the end-to-end subprocess scenarios above; both
# kinds of tests are retained for defense in depth).
# ---------------------------------------------------------------------------

def test_git_status_allowlist_positive_case_exactly_three_allowed_paths():
    """Positive preflight case: exactly the three allowed untracked paths,
    nothing else -> no violations."""
    allowed = {
        "stage1/evidence/umsi_five_fixture_gate_contract.json",
        "stage1/tools/umsi_five_fixture_gate_harness.py",
        "tests/test_umsi_five_fixture_gate_harness.py",
    }
    lines = [f"?? {p}" for p in allowed]
    assert _check_git_status_against_allowlist(lines, allowed) == []


def test_git_status_allowlist_rejects_unexpected_untracked_path():
    allowed = {"stage1/evidence/umsi_five_fixture_gate_contract.json"}
    lines = ["?? some/other/file.py"]
    violations = _check_git_status_against_allowlist(lines, allowed)
    assert violations
    assert any("unexpected untracked path" in v for v in violations)


def test_git_status_allowlist_rejects_staged_change():
    violations = _check_git_status_against_allowlist(["M  staged_file.py"], set())
    assert violations
    assert any("staged change" in v for v in violations)


def test_git_status_allowlist_rejects_tracked_unstaged_change():
    violations = _check_git_status_against_allowlist([" M tracked_file.py"], set())
    assert violations
    assert any("tracked worktree change" in v for v in violations)


def test_git_status_allowlist_rejects_renamed_entry():
    violations = _check_git_status_against_allowlist(["R  old.py -> new.py"], set())
    assert violations
    assert any("renamed/copied" in v for v in violations)


def test_git_status_allowlist_rejects_conflicted_entry():
    violations = _check_git_status_against_allowlist(["UU conflicted.py"], set())
    assert violations
    assert any("conflicted" in v for v in violations)


@pytest.mark.parametrize("malformed_line", ["XY", "XYZ path_without_space_at_index_2"])
def test_git_status_allowlist_rejects_malformed_record(malformed_line):
    violations = _check_git_status_against_allowlist([malformed_line], set())
    assert violations
    assert any("malformed" in v for v in violations)


def test_verify_worktree_and_blob_identity_accepts_matching_file(tmp_path):
    repo = _init_fake_repo(tmp_path)
    path = repo / "saliency" / "umsi_model.py"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _verify_worktree_and_blob_identity(
        repo_dir=repo, rel_path="saliency/umsi_model.py",
        worktree_path=path, expected_sha256=sha,
    )  # must not raise


def test_verify_worktree_and_blob_identity_rejects_wrong_sha(tmp_path):
    repo = _init_fake_repo(tmp_path)
    with pytest.raises(ProvenanceError, match="worktree SHA-256"):
        _verify_worktree_and_blob_identity(
            repo_dir=repo, rel_path="saliency/umsi_model.py",
            worktree_path=repo / "saliency" / "umsi_model.py", expected_sha256="f" * 64,
        )


def test_verify_worktree_and_blob_identity_rejects_worktree_blob_divergence(tmp_path):
    repo = _init_fake_repo(tmp_path)
    path = repo / "saliency" / "umsi_model.py"
    path.write_bytes(b"# modified after commit, not re-committed\n")
    new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ProvenanceError, match="blob-at-HEAD"):
        _verify_worktree_and_blob_identity(
            repo_dir=repo, rel_path="saliency/umsi_model.py",
            worktree_path=path, expected_sha256=new_sha,
        )


def test_verify_worktree_and_blob_identity_rejects_missing_file(tmp_path):
    repo = _init_fake_repo(tmp_path)
    missing = repo / "saliency" / "does_not_exist.py"
    with pytest.raises(ProvenanceError, match="worktree file missing"):
        _verify_worktree_and_blob_identity(
            repo_dir=repo, rel_path="saliency/does_not_exist.py",
            worktree_path=missing, expected_sha256="a" * 64,
        )


def test_verify_runtime_versions_accepts_matching_values():
    expected = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": {"numpy": importlib.metadata.version("numpy")},
    }
    _verify_runtime_versions(expected)  # must not raise


def test_verify_runtime_versions_rejects_wrong_python_version():
    expected = {
        "python_implementation": platform.python_implementation(),
        "python_version": "0.0.0",
        "python_executable": sys.executable,
        "package_versions": {"numpy": importlib.metadata.version("numpy")},
    }
    with pytest.raises(ProvenanceError, match="python_version"):
        _verify_runtime_versions(expected)


def test_verify_runtime_versions_rejects_wrong_package_version():
    expected = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": {"numpy": "0.0.0"},
    }
    with pytest.raises(ProvenanceError, match="numpy"):
        _verify_runtime_versions(expected)


def test_verify_runtime_versions_rejects_missing_package_no_fallback():
    """No automatic package-name fallback: a pinned package that cannot be
    found via importlib.metadata is a hard failure, never silently
    skipped or substituted with an alternate name."""
    expected = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": {"this_package_does_not_exist_xyz": "1.0.0"},
    }
    with pytest.raises(ProvenanceError, match="not installed"):
        _verify_runtime_versions(expected)


def test_verify_output_directory_safety_rejects_existing_directory(tmp_path):
    run_dir_name = "five_fixture_run_existing"
    (tmp_path / run_dir_name).mkdir()
    with pytest.raises(ProvenanceError, match="already exists"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name=run_dir_name,
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_rejects_existing_file(tmp_path):
    """Phase 2: a pre-existing regular file at the final path is rejected
    exactly like a pre-existing directory."""
    run_dir_name = "five_fixture_run_existing_file"
    (tmp_path / run_dir_name).write_bytes(b"not a directory")
    with pytest.raises(ProvenanceError, match="already exists"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name=run_dir_name,
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_rejects_symlink_pointing_outside_output_root(tmp_path):
    """Phase 2: a symlink at the final path -- even a valid one resolving
    outside the output root -- is rejected via os.path.lexists(), which
    does not follow the final symlink component."""
    outside_target = tmp_path / "outside_target"
    outside_target.mkdir()
    run_dir_name = "five_fixture_run_symlinked"
    (tmp_path / run_dir_name).symlink_to(outside_target, target_is_directory=True)
    with pytest.raises(ProvenanceError, match="already exists"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name=run_dir_name,
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_rejects_broken_symlink(tmp_path):
    """Phase 2: a broken (dangling) symlink at the final path is rejected;
    os.path.lexists() reports True for it even though Path.exists() would
    report False."""
    run_dir_name = "five_fixture_run_broken_symlink"
    (tmp_path / run_dir_name).symlink_to(tmp_path / "does_not_exist_target")
    with pytest.raises(ProvenanceError, match="already exists"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name=run_dir_name,
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_resolves_symlinked_output_root_correctly(tmp_path):
    """Phase 2 (direct-parent property): the final path's direct parent
    must be the *resolved* output root, even when the caller-supplied
    output root itself is a symlink to another directory -- proving the
    containment check cannot be defeated by a symlinked output-root
    alias."""
    real_output_root = tmp_path / "real_output_root"
    real_output_root.mkdir()
    symlinked_output_root = tmp_path / "output_root_symlink"
    symlinked_output_root.symlink_to(real_output_root, target_is_directory=True)

    result = _verify_output_directory_safety(
        output_root=symlinked_output_root, run_dir_name="five_fixture_run_via_symlink",
        required_prefix="five_fixture_run_",
        protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
    )
    assert result == real_output_root.resolve() / "five_fixture_run_via_symlink"
    assert result.parent == real_output_root.resolve()


@pytest.mark.parametrize(
    "bad_name",
    [
        "five_fixture_run_x/../../../escape",
        "five_fixture_run_x\\..\\..\\escape",
        "/etc/five_fixture_run_absolute",
        "C:\\Windows\\five_fixture_run_absolute",
        "\\\\server\\share\\five_fixture_run_x",
        "five_fixture_run_outer/five_fixture_run_inner",
    ],
)
def test_verify_output_directory_safety_rejects_unsafe_run_directory_names(tmp_path, bad_name):
    """Phase 2: POSIX traversal, Windows-style traversal, absolute POSIX
    path, absolute Windows-style path (drive-letter and UNC), and multiple
    path components are all rejected before any filesystem entry is even
    checked."""
    with pytest.raises(ProvenanceError, match="single path component"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name=bad_name,
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_rejects_wrong_prefix(tmp_path):
    with pytest.raises(ProvenanceError, match="required prefix"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name="wrong_prefix_dir",
            required_prefix="five_fixture_run_",
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_rejects_protected_name(tmp_path):
    with pytest.raises(ProvenanceError, match="protected"):
        _verify_output_directory_safety(
            output_root=tmp_path, run_dir_name="step2b_results_p8",
            required_prefix="", protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


def test_verify_output_directory_safety_accepts_fresh_directory(tmp_path):
    run_dir_name = "five_fixture_run_fresh"
    result = _verify_output_directory_safety(
        output_root=tmp_path, run_dir_name=run_dir_name,
        required_prefix="five_fixture_run_",
        protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
    )
    # Positive proof: a valid, single-component name resolves to exactly
    # one direct child of the canonical (resolved) output root.
    assert result == tmp_path.resolve() / run_dir_name
    assert result.parent == tmp_path.resolve()


@pytest.mark.parametrize(
    "bad_name,expected_match",
    [
        ("five_fixture_run_x/../../../escape", "single path component"),
        ("five_fixture_run_x\\..\\..\\escape", "single path component"),
        ("/etc/five_fixture_run_absolute", "single path component"),
        ("C:\\Windows\\five_fixture_run_absolute", "single path component"),
        ("\\\\server\\share\\five_fixture_run_x", "single path component"),
        ("five_fixture_run_outer/five_fixture_run_inner", "single path component"),
        ("five_fixture_run_C:drive_relative", "drive-letter marker"),
        (".", "must not be '.' or '..'"),
        ("..", "must not be '.' or '..'"),
        ("", "non-empty string"),
    ],
)
def test_find_unsafe_run_directory_name_reason_rejects_unsafe_names(bad_name, expected_match):
    reason = _find_unsafe_run_directory_name_reason(bad_name)
    assert reason is not None
    assert expected_match in reason


def test_find_unsafe_run_directory_name_reason_accepts_safe_name():
    assert _find_unsafe_run_directory_name_reason("five_fixture_run_ok") is None


def test_validate_run_directory_name_accepts_valid_name():
    _validate_run_directory_name(
        "five_fixture_run_ok", "five_fixture_run_", _PROTECTED_OUTPUT_DIRECTORY_NAMES,
    )  # must not raise


def test_validate_run_directory_name_rejects_traversal():
    with pytest.raises(ProvenanceError, match="single path component"):
        _validate_run_directory_name(
            "five_fixture_run_x/../escape", "five_fixture_run_",
            _PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )


@pytest.mark.parametrize(
    "bad_name",
    [
        "five_fixture_run_x/../../../escape",
        "five_fixture_run_x\\..\\..\\escape",
        "/etc/five_fixture_run_absolute",
        "C:\\Windows\\five_fixture_run_absolute",
        "five_fixture_run_outer/five_fixture_run_inner",
    ],
)
def test_contract_unsafe_run_directory_name_rejected_at_schema_time(tmp_path, bad_name):
    """Item #10 correction: a traversal, absolute, or multi-component
    run_directory_name is rejected at contract-schema-validation time --
    even earlier than the runtime output-directory-safety check in
    run_frozen_five_fixture_experiment."""
    contract = _make_synthetic_contract(tmp_path)
    contract["execution"]["run_directory_name"] = bad_name
    bad_path = tmp_path / "bad_contract.json"
    bad_path.write_text(json.dumps(contract))
    with pytest.raises(ContractError, match="single path component"):
        load_and_validate_five_fixture_contract(bad_path)


def test_harness_module_import_does_not_load_legacy_runner():
    """A fresh process's plain import of the harness module must not
    import stage1.tools.umsi_step2b_gate_runner (module-level import of
    the reused legacy runner has been removed; it is imported lazily, on
    first actual need, via _get_runner_module())."""
    proc = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            f"sys.path.insert(0, {_repo_root!r}); "
            "import stage1.tools.umsi_five_fixture_gate_harness; "
            "present = any(\n"
            "    m in sys.modules\n"
            "    for m in ('stage1.tools.umsi_step2b_gate_runner', 'umsi_step2b_gate_runner')\n"
            ")\n"
            "print('RUNNER_IMPORTED=' + str(present))"
        )],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RUNNER_IMPORTED=False" in proc.stdout


def test_real_contract_provenance_and_execution_fields_present():
    """The real, frozen five-fixture contract must carry every provenance/
    execution-safety field required by the acceptance-audit corrections."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert contract["expected_repository_head"] == "878e3e9fcfac7828913d67cf32c70920c80d3e1c"
    assert contract["production_source_identity"]["path"] == "saliency/umsi_model.py"
    assert (
        contract["production_source_identity"]["sha256"]
        == "38b7503626a0ec3c176d05aaed0ad6c9461085a7f771f96d3f9a62a0884ab554"
    )
    assert (
        contract["reused_legacy_runner_identity"]["path"]
        == "stage1/tools/umsi_step2b_gate_runner.py"
    )
    assert (
        contract["reused_legacy_runner_identity"]["sha256"]
        == "df356064b4d20f5ba652f16348cd3ef295c66528b2539afafd8309a8f77302fa"
    )
    assert (
        contract["candidate_harness_identity"]["path"]
        == "stage1/tools/umsi_five_fixture_gate_harness.py"
    )
    assert set(contract["allowed_untracked_paths"]) == {
        "stage1/evidence/umsi_five_fixture_gate_contract.json",
        "stage1/tools/umsi_five_fixture_gate_harness.py",
        "tests/test_umsi_five_fixture_gate_harness.py",
    }
    assert contract["execution"]["run_directory_name"].startswith("five_fixture_run_")
    assert contract["execution"]["run_directory_name_prefix_requirement"] == "five_fixture_run_"
    assert "launcher" in contract["execution"]
    assert "predecessor_incident" in contract["execution"]
    assert "numpy" in contract["runtime_identity"]["package_versions"]


def test_real_contract_candidate_harness_hash_matches_actual_file():
    """The real contract's pinned candidate-harness SHA-256 must match the
    real harness file's actual current content."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    assert contract["candidate_harness_identity"]["sha256"] == _actual_harness_sha256()


# ---------------------------------------------------------------------------
# Successor-freeze correction: module-form launcher regression test.
#
# Background (incident): the one authorized real execution under the
# consumed identity "five_fixture_run_20260806T120245Z_878e3e9f" was
# launched as a direct script (python -B stage1/tools/umsi_five_fixture_gate_harness.py
# ...) and failed with ModuleNotFoundError: No module named 'saliency',
# because direct-script invocation does not place the repository root on
# sys.path. Module-form invocation (python -B -m
# stage1.tools.umsi_five_fixture_gate_harness ..., run from the exact
# repository root) places that root on sys.path instead, so
# `from saliency.umsi_model import UMSIPlus` resolves correctly. This is now
# the only sanctioned launcher form.
#
# The helper functions below are a genuinely different subprocess-invocation
# path from _run_frozen_experiment_subprocess above: they launch the CLI as
# a real argv list (never a `python -c` code string), from an isolated fake
# repository root that carries its own untracked copy of the real,
# unmodified harness file plus fully self-contained fake/stub `saliency`
# and legacy-runner packages placed directly in that fake repo's own
# directory tree (never via sys.path.insert), with PYTHONPATH explicitly
# stripped from the subprocess environment and no sys.path manipulation
# anywhere.
# ---------------------------------------------------------------------------

_MODULE_FORM_SENTINEL_MESSAGE = "MODULE_FORM_LAUNCHER_SENTINEL_REACHED"

_FAKE_SALIENCY_STUB_SOURCE_FOR_MODULE_FORM_TEST = (
    "import sys\n"
    "\n"
    "\n"
    "class UMSIPlus:\n"
    "    \"\"\"Fake stand-in for the real UMSIPlus model, used only inside an\n"
    "    isolated fake repository for the module-form launcher regression\n"
    "    test. Never performs real inference and never loads real weights;\n"
    "    its constructor immediately raises a distinctive sentinel exception\n"
    "    that proves execution reached this point only via a successfully-\n"
    "    resolved repository-root 'saliency' package.\n"
    "    \"\"\"\n"
    "\n"
    "    def __init__(self, *a, **kw):\n"
    "        bad = [\n"
    "            m for m in sys.modules\n"
    "            if m.split('.')[0] in ('tensorflow', 'keras', 'tf_keras')\n"
    "        ]\n"
    "        print('BAD_MODULES=' + repr(bad))\n"
    "        print('" + _MODULE_FORM_SENTINEL_MESSAGE + "')\n"
    "        raise RuntimeError(\n"
    "            '" + _MODULE_FORM_SENTINEL_MESSAGE + ": fake UMSIPlus '\n"
    "            'constructed via a successfully-resolved repository-root '\n"
    "            'saliency package; no real weights were loaded and no '\n"
    "            'inference was performed'\n"
    "        )\n"
)

_FAKE_FUNCTIONAL_LEGACY_RUNNER_STUB_SOURCE = (
    "import hashlib\n"
    "\n"
    "\n"
    "class PreflightError(Exception):\n"
    "    pass\n"
    "\n"
    "\n"
    "def sha256_file(path):\n"
    "    h = hashlib.sha256()\n"
    "    with open(path, 'rb') as fh:\n"
    "        for chunk in iter(lambda: fh.read(1024 * 1024), b''):\n"
    "            h.update(chunk)\n"
    "    return h.hexdigest()\n"
)


def _init_fake_repo_for_module_form_test(tmp_path: Path) -> Path:
    """Initialize a throwaway git repository under tmp_path (never the real
    thesis repository), with the same top-level package layout as
    production (saliency/ and stage1/tools/ at the repo root, resolved as
    implicit namespace packages, matching the real repository), containing
    ONLY fully self-contained fake/stub production-source and legacy-
    runner modules -- never the real ones. A placeholder file is also
    committed inside stage1/evidence/ so that directory is already tracked
    (otherwise git would collapse an entirely-untracked new directory into
    a single directory-level status line, instead of reporting the
    untracked contract file within it individually, which the allowlist
    check below requires). The candidate harness copy and the contract
    file are added afterwards by the caller as untracked entries,
    mirroring their real, never-committed status in the actual thesis
    repository."""
    repo = tmp_path / "fake_repo_module_form_launcher"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True)
    (repo / "saliency").mkdir()
    (repo / "stage1" / "tools").mkdir(parents=True)
    (repo / "stage1" / "evidence").mkdir(parents=True)
    (repo / "saliency" / "umsi_model.py").write_text(
        _FAKE_SALIENCY_STUB_SOURCE_FOR_MODULE_FORM_TEST
    )
    (repo / "stage1" / "tools" / "umsi_step2b_gate_runner.py").write_text(
        _FAKE_FUNCTIONAL_LEGACY_RUNNER_STUB_SOURCE
    )
    (repo / "stage1" / "evidence" / ".gitkeep").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _build_module_form_launcher_scenario(tmp_path: Path) -> Dict[str, Any]:
    """Build a complete, valid scenario for the module-form launcher
    regression test: an isolated fake repository (its own saliency/ and
    stage1/tools/ stub packages, matching production's package layout),
    containing an untracked COPY of the real, unmodified harness file
    (never the harness file being edited or executed in place -- a copy,
    so this test suite never writes into the real repository's own
    tracked/untracked files), an untracked contract pinned to this fake
    repository's own HEAD and fake production-source/legacy-runner
    hashes, a real loadable evidence NPZ, correctly-sized weights, and
    real fixture files, all placed so that per-fixture preflight (the
    last check before model import) succeeds cleanly and execution reaches
    the fake UMSIPlus sentinel."""
    repo = _init_fake_repo_for_module_form_test(tmp_path)
    head = _git_head(repo)

    real_harness_bytes = _HARNESS_FILE_PATH.read_bytes()
    harness_copy_path = repo / "stage1" / "tools" / "umsi_five_fixture_gate_harness.py"
    harness_copy_path.write_bytes(real_harness_bytes)
    harness_sha = hashlib.sha256(real_harness_bytes).hexdigest()
    assert harness_sha == _actual_harness_sha256()

    fixtures_dir = tmp_path / "module_form_fixtures"
    fixtures_dir.mkdir()
    contract = _make_synthetic_contract(fixtures_dir)

    weights_path = tmp_path / "module_form_weights.bin"
    _write_correctly_sized_weights_file(weights_path)
    contract["weights_filename"] = weights_path.name
    contract["weights_size_bytes"] = _EXPECTED_WEIGHTS_SIZE_BYTES
    contract["weights_sha256"] = _ZERO_WEIGHTS_SHA256

    evidence_path = tmp_path / "module_form_evidence.npz"
    _write_real_npz_for_contract(contract, evidence_path)
    contract["evidence_npz_filename"] = evidence_path.name
    contract["evidence_npz_sha256"] = _sha256_bytes(evidence_path.read_bytes())

    prod_source_bytes = (repo / "saliency" / "umsi_model.py").read_bytes()
    runner_bytes = (repo / "stage1" / "tools" / "umsi_step2b_gate_runner.py").read_bytes()

    contract["expected_repository_head"] = head
    contract["production_source_identity"] = {
        "path": "saliency/umsi_model.py",
        "sha256": _sha256_bytes(prod_source_bytes),
    }
    contract["reused_legacy_runner_identity"] = {
        "path": "stage1/tools/umsi_step2b_gate_runner.py",
        "sha256": _sha256_bytes(runner_bytes),
    }
    contract["candidate_harness_identity"] = {
        "path": "stage1/tools/umsi_five_fixture_gate_harness.py",
        "sha256": harness_sha,
    }
    contract["allowed_untracked_paths"] = [
        "stage1/evidence/umsi_five_fixture_gate_contract.json",
        "stage1/tools/umsi_five_fixture_gate_harness.py",
        "tests/test_umsi_five_fixture_gate_harness.py",
    ]
    contract["runtime_identity"] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": {"numpy": importlib.metadata.version("numpy")},
    }

    output_root = tmp_path / "module_form_output_root"
    output_root.mkdir()
    contract_path = repo / "stage1" / "evidence" / "umsi_five_fixture_gate_contract.json"

    # execution.launcher/predecessor_incident: built to genuinely match how
    # _run_module_form_launcher_subprocess actually launches this scenario
    # (module-form argv, cwd=repo, PYTHONPATH stripped,
    # PYTHONDONTWRITEBYTECODE=1 forced, sys.executable, and every CLI path
    # argument passed as an absolute path equal to the corresponding
    # launcher field), so the new launcher preflight is exercised for real
    # -- not merely schema-validated -- by this end-to-end subprocess test.
    contract["execution"] = {
        "run_directory_name": f"five_fixture_run_test_module_form_launcher_{head[:8]}",
        "run_directory_name_prefix_requirement": "five_fixture_run_",
        "launcher": {
            "mode": "module",
            "python_executable": sys.executable,
            "interpreter_flags": ["-B"],
            "module": "stage1.tools.umsi_five_fixture_gate_harness",
            "working_directory": str(repo),
            "required_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "pythonpath_required_state": "unset",
            "action": "execute_frozen_experiment",
            "contract_path": str(contract_path),
            "output_root": str(output_root),
            "weights_path": str(weights_path),
            "evidence_npz_path": str(evidence_path),
            "fixtures_root": str(fixtures_dir),
        },
        "predecessor_incident": {
            "consumed_run_directory_name": (
                f"five_fixture_run_test_module_form_consumed_{head[:8]}"
            ),
            "failure_classification": (
                "infrastructure_bootstrap_import_path_failure_before_production_model_import"
            ),
            "execution_attempted": True,
            "identity_permanently_consumed": True,
            "run_directory_created": False,
            "production_model_imported": False,
            "weights_loaded": False,
            "inference_executed": False,
            "scientific_verdict_reached": False,
        },
    }

    contract_path.write_text(json.dumps(contract))

    return {
        "repo": repo,
        "head": head,
        "harness_sha": harness_sha,
        "contract": contract,
        "contract_path": contract_path,
        "weights_path": weights_path,
        "evidence_path": evidence_path,
        "fixtures_dir": fixtures_dir,
        "output_root": output_root,
    }


def _run_module_form_launcher_subprocess(
    scenario: Dict[str, Any], *, script_form: bool = False
) -> Tuple[int, str, str]:
    """Launch the harness as a real subprocess argv list -- never a
    `python -c` code string -- from the isolated fake repository root as
    cwd, with PYTHONPATH stripped from the subprocess environment and no
    sys.path manipulation anywhere (neither here nor inside any launched
    code). By default uses the frozen, sanctioned module-form launcher
    (python -B -m stage1.tools.umsi_five_fixture_gate_harness); if
    script_form is True, uses the now-forbidden direct-script form
    instead, to prove the incident's root cause remains real and
    reproducible (i.e. that reverting the launcher convention would
    immediately regress)."""
    repo = scenario["repo"]
    if script_form:
        target = [str(repo / "stage1" / "tools" / "umsi_five_fixture_gate_harness.py")]
    else:
        target = ["-m", "stage1.tools.umsi_five_fixture_gate_harness"]
    argv = [
        sys.executable, "-B", *target,
        "--execute-frozen-experiment",
        "--contract", str(scenario["contract_path"]),
        "--contract-sha", _sha256_bytes(scenario["contract_path"].read_bytes()),
        "--expected-harness-sha", scenario["harness_sha"],
        "--repo", ".",
        "--weights", str(scenario["weights_path"]),
        "--evidence-npz", str(scenario["evidence_path"]),
        "--fixtures-root", str(scenario["fixtures_dir"]),
        "--output-root", str(scenario["output_root"]),
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(argv, cwd=str(repo), env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_module_form_launcher_resolves_saliency_and_reaches_fake_umsiplus_sentinel(tmp_path):
    """Successor-freeze correction regression test.

    Proves that the frozen, sanctioned launcher form -- python -B -m
    stage1.tools.umsi_five_fixture_gate_harness, invoked from the exact
    repository root as cwd, with no PYTHONPATH and no sys.path
    manipulation anywhere -- successfully resolves the repository-root
    'saliency' package and reaches model construction (a deliberate fake
    UMSIPlus sentinel), whereas the now-forbidden direct-script form,
    invoked identically from the same cwd, still fails with
    ModuleNotFoundError for 'saliency' -- exactly the incident that
    consumed execution identity
    'five_fixture_run_20260806T120245Z_878e3e9f'. This locks in why
    module-form is required and would fail if a future change ever
    silently regressed the sanctioned launcher back to direct-script
    form.
    """
    scenario = _build_module_form_launcher_scenario(tmp_path)

    # 1. Frozen, sanctioned module-form launcher: must resolve 'saliency'
    #    and reach the fake UMSIPlus sentinel.
    rc, stdout, stderr = _run_module_form_launcher_subprocess(scenario, script_form=False)
    assert "BAD_MODULES=[]" in stdout, stdout + stderr
    assert _MODULE_FORM_SENTINEL_MESSAGE in stdout, stdout + stderr
    assert _MODULE_FORM_SENTINEL_MESSAGE in stderr, stdout + stderr
    assert "RuntimeError" in stderr, stdout + stderr
    assert rc == 1
    # No output artifact created: UMSIPlus() raises before run_five_fixtures,
    # before mkdir(), before the report is written.
    assert list(scenario["output_root"].iterdir()) == []
    assert "PASS" not in stdout and "FAILED" not in stdout and "overall_verdict" not in stdout

    # 2. Now-forbidden direct-script form, invoked identically from the same
    #    isolated fake repository root: the new launcher preflight (added
    #    ahead of every other provenance check) must now BLOCK this
    #    cleanly, with a precise BLOCKED: message referencing module-form
    #    and the permanently consumed predecessor identity, strictly
    #    before the deferred `saliency` import -- rather than relying on an
    #    uncaught ModuleNotFoundError (the pre-correction behavior). This
    #    still proves the incident's root cause remains real and
    #    reproducible (i.e. that reverting the launcher convention would
    #    immediately be caught, not silently regress into another
    #    uncaught crash).
    rc2, stdout2, stderr2 = _run_module_form_launcher_subprocess(scenario, script_form=True)
    assert rc2 == 1
    assert "BLOCKED" in stderr2, stdout2 + stderr2
    assert "launcher preflight failed" in stderr2, stdout2 + stderr2
    assert "module form" in stderr2, stdout2 + stderr2
    assert scenario["contract"]["execution"]["predecessor_incident"][
        "consumed_run_directory_name"
    ] in stderr2, stdout2 + stderr2
    assert "ModuleNotFoundError" not in stderr2, stdout2 + stderr2
    assert _MODULE_FORM_SENTINEL_MESSAGE not in stdout2
    assert list(scenario["output_root"].iterdir()) == []


def test_real_contract_successor_execution_identity_replaces_consumed_identity():
    """Successor-freeze correction: the real contract's execution identity
    must be a brand-new, never-before-used value, never the consumed
    (permanently failed) 'five_fixture_run_20260806T120245Z_878e3e9f'
    identity, and the consumed identity's incident facts must be recorded
    as structured, typed, exact-key-set-validated fields within
    execution.predecessor_incident (not as free-text
    consumed_run_directory_name / consumed_run_directory_incident_note
    keys tolerated by a permissive parent object)."""
    contract = load_and_validate_five_fixture_contract(_REAL_CONTRACT_PATH)
    execution = contract["execution"]
    consumed_id = "five_fixture_run_20260806T120245Z_878e3e9f"

    assert execution["run_directory_name"] != consumed_id
    assert execution["run_directory_name"].startswith("five_fixture_run_")
    assert execution["run_directory_name"].endswith("_878e3e9f")
    assert execution["run_directory_name_prefix_requirement"] == "five_fixture_run_"

    # The old, unvalidated, permissively-tolerated free-text keys must be
    # gone entirely -- not merely superseded while still present.
    assert "consumed_run_directory_name" not in execution
    assert "consumed_run_directory_incident_classification" not in execution
    assert "consumed_run_directory_incident_note" not in execution

    incident = execution["predecessor_incident"]
    assert incident["consumed_run_directory_name"] == consumed_id
    assert incident["failure_classification"] == (
        "infrastructure_bootstrap_import_path_failure_before_production_model_import"
    )
    assert incident["execution_attempted"] is True
    assert incident["identity_permanently_consumed"] is True
    assert incident["run_directory_created"] is False
    assert incident["production_model_imported"] is False
    assert incident["weights_loaded"] is False
    assert incident["inference_executed"] is False
    assert incident["scientific_verdict_reached"] is False

    launcher = execution["launcher"]
    assert launcher["mode"] == "module"
    assert launcher["module"] == "stage1.tools.umsi_five_fixture_gate_harness"
    assert launcher["action"] == "execute_frozen_experiment"
    assert launcher["pythonpath_required_state"] == "unset"
    assert launcher["interpreter_flags"] == ["-B"]
    assert launcher["required_environment"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    for path_key in (
        "python_executable", "working_directory", "contract_path",
        "output_root", "weights_path", "evidence_npz_path", "fixtures_root",
    ):
        assert Path(launcher[path_key]).is_absolute(), path_key


# ---------------------------------------------------------------------------
# Successor-freeze correction: focused schema-validator rejection tests for
# execution.launcher / execution.predecessor_incident (replacing the old
# permissive, never-enforced consumed_run_directory_* free-text keys).
# ---------------------------------------------------------------------------

def _synthetic_contract_with_execution(tmp_path: Path) -> Dict[str, Any]:
    """A schema-valid synthetic contract (via _make_synthetic_contract),
    used as the shared starting point for every launcher/predecessor_
    incident rejection test below: each test mutates exactly one aspect of
    contract["execution"] and expects ContractError."""
    fixtures_dir = tmp_path / "schema_fixtures"
    fixtures_dir.mkdir()
    return _make_synthetic_contract(fixtures_dir)


def _write_and_load(tmp_path: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    path = tmp_path / "contract_under_test.json"
    path.write_text(json.dumps(contract))
    return load_and_validate_five_fixture_contract(path)


def test_contract_execution_missing_launcher_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    del contract["execution"]["launcher"]
    with pytest.raises(ContractError, match=r"execution key set mismatch.*missing.*launcher"):
        _write_and_load(tmp_path, contract)


def test_contract_execution_missing_predecessor_incident_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    del contract["execution"]["predecessor_incident"]
    with pytest.raises(
        ContractError, match=r"execution key set mismatch.*missing.*predecessor_incident"
    ):
        _write_and_load(tmp_path, contract)


def test_contract_execution_unexpected_key_rejected(tmp_path):
    """A permissive parent object that merely tolerates additional keys is
    exactly the gap this correction closes: an unrecognized key in
    execution itself must now be rejected, not silently ignored."""
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["consumed_run_directory_incident_note"] = "free text, no longer allowed"
    with pytest.raises(ContractError, match=r"execution key set mismatch.*unexpected"):
        _write_and_load(tmp_path, contract)


def test_contract_launcher_unexpected_key_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["launcher"]["extra_undeclared_key"] = "x"
    with pytest.raises(ContractError, match=r"execution\.launcher key set mismatch.*unexpected"):
        _write_and_load(tmp_path, contract)


def test_contract_launcher_wrong_mode_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["launcher"]["mode"] = "script"
    with pytest.raises(ContractError, match=r"execution\.launcher\.mode"):
        _write_and_load(tmp_path, contract)


def test_contract_launcher_wrong_module_rejected(tmp_path):
    """The launcher's module name must agree with the module name derived
    from candidate_harness_identity.path; an arbitrary/incorrect module
    string is rejected even though it is a well-formed non-empty string."""
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["launcher"]["module"] = "not_the_real_module_path"
    with pytest.raises(ContractError, match=r"execution\.launcher\.module"):
        _write_and_load(tmp_path, contract)


def test_contract_launcher_relative_working_directory_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["launcher"]["working_directory"] = "relative/not/absolute"
    with pytest.raises(ContractError, match=r"execution\.launcher\.working_directory.*absolute"):
        _write_and_load(tmp_path, contract)


def test_contract_launcher_relative_output_root_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["launcher"]["output_root"] = "relative/output/root"
    with pytest.raises(ContractError, match=r"execution\.launcher\.output_root.*absolute"):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_unexpected_key_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["extra_undeclared_key"] = "x"
    with pytest.raises(
        ContractError, match=r"execution\.predecessor_incident key set mismatch.*unexpected"
    ):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_wrong_type_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["execution_attempted"] = "true"
    with pytest.raises(
        ContractError, match=r"execution\.predecessor_incident\.execution_attempted.*boolean"
    ):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_wrong_enum_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["failure_classification"] = (
        "some_other_unrecognized_classification"
    )
    with pytest.raises(
        ContractError, match=r"execution\.predecessor_incident\.failure_classification"
    ):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_reuse_of_consumed_identity_rejected(tmp_path):
    """The permanently consumed identity can never be reselected as the
    current run directory: setting both fields to the identical value must
    be rejected at schema-validation time."""
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["consumed_run_directory_name"] = (
        contract["execution"]["run_directory_name"]
    )
    with pytest.raises(ContractError, match=r"must never equal execution\.run_directory_name"):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_must_be_true_key_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["identity_permanently_consumed"] = False
    with pytest.raises(
        ContractError, match=r"execution\.predecessor_incident\.identity_permanently_consumed"
    ):
        _write_and_load(tmp_path, contract)


def test_contract_predecessor_incident_must_be_false_key_rejected(tmp_path):
    contract = _synthetic_contract_with_execution(tmp_path)
    contract["execution"]["predecessor_incident"]["scientific_verdict_reached"] = True
    with pytest.raises(
        ContractError, match=r"execution\.predecessor_incident\.scientific_verdict_reached"
    ):
        _write_and_load(tmp_path, contract)


# ---------------------------------------------------------------------------
# Successor-freeze correction: direct (non-subprocess) unit tests for
# _verify_launcher_preflight -- fast, precise complements to the end-to-end
# module-form subprocess regression test above.
# ---------------------------------------------------------------------------

def _matching_launcher_args(tmp_path: Path, contract: Dict[str, Any]) -> argparse.Namespace:
    launcher = contract["execution"]["launcher"]
    return argparse.Namespace(
        contract=launcher["contract_path"],
        output_root=launcher["output_root"],
        weights=launcher["weights_path"],
        evidence_npz=launcher["evidence_npz_path"],
        fixtures_root=launcher["fixtures_root"],
    )


def _contract_with_real_working_launcher(tmp_path: Path) -> Dict[str, Any]:
    """A synthetic contract whose execution.launcher is rewritten to
    genuinely match this test process's own real, currently-running
    interpreter and a tmp_path-based working directory/paths, so
    _verify_launcher_preflight can be called directly (no subprocess) and
    be expected to pass cleanly as a positive baseline."""
    contract = _synthetic_contract_with_execution(tmp_path)
    launcher = contract["execution"]["launcher"]
    launcher["python_executable"] = sys.executable
    launcher["working_directory"] = str(tmp_path)
    launcher["contract_path"] = str(tmp_path / "somewhere" / "contract.json")
    launcher["output_root"] = str(tmp_path / "somewhere" / "output_root")
    launcher["weights_path"] = str(tmp_path / "somewhere" / contract["weights_filename"])
    launcher["evidence_npz_path"] = str(
        tmp_path / "somewhere" / contract["evidence_npz_filename"]
    )
    launcher["fixtures_root"] = str(tmp_path / "somewhere" / "fixtures")
    return contract


def test_verify_launcher_preflight_accepts_matching_launcher(tmp_path, monkeypatch):
    contract = _contract_with_real_working_launcher(tmp_path)
    validated = _write_and_load(tmp_path, contract)  # proves the contract is schema-valid too
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    args = _matching_launcher_args(tmp_path, validated)
    _verify_launcher_preflight(args, validated)  # must not raise


def test_verify_launcher_preflight_rejects_pythonpath_set(tmp_path, monkeypatch):
    contract = _contract_with_real_working_launcher(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/some/unwanted/path")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    args = _matching_launcher_args(tmp_path, contract)
    with pytest.raises(ProvenanceError, match="PYTHONPATH is set"):
        _verify_launcher_preflight(args, contract)


def test_verify_launcher_preflight_rejects_direct_script_invocation(tmp_path, monkeypatch):
    """Simulates direct-script execution by temporarily clearing the
    harness module's own __spec__ (set to None by the interpreter itself
    for a real direct-script invocation; a real dotted-name ModuleSpec for
    real `-m` invocation). Complements the end-to-end subprocess proof in
    test_module_form_launcher_resolves_saliency_and_reaches_fake_umsiplus_sentinel
    with a fast, precise, single-property check."""
    contract = _contract_with_real_working_launcher(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    args = _matching_launcher_args(tmp_path, contract)
    monkeypatch.setattr(_harness_module, "__spec__", None)
    with pytest.raises(ProvenanceError, match="not launched in module form"):
        _verify_launcher_preflight(args, contract)


def test_verify_launcher_preflight_rejects_cli_contract_path_mismatch(tmp_path, monkeypatch):
    """A CLI --contract value that resolves to a different path than the
    contract's own pinned execution.launcher.contract_path must be
    rejected before the deferred production import, proving structured
    launcher paths and actual CLI arguments are cross-checked, not merely
    independently schema-valid."""
    contract = _contract_with_real_working_launcher(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    args = _matching_launcher_args(tmp_path, contract)
    args.contract = str(tmp_path / "somewhere" / "a_completely_different_contract.json")
    with pytest.raises(ProvenanceError, match=r"--contract resolved to"):
        _verify_launcher_preflight(args, contract)


def test_verify_launcher_preflight_rejects_relative_cli_path(tmp_path, monkeypatch):
    contract = _contract_with_real_working_launcher(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    args = _matching_launcher_args(tmp_path, contract)
    args.output_root = "relative/output/root"
    with pytest.raises(ProvenanceError, match=r"--output-root value .* is not an absolute path"):
        _verify_launcher_preflight(args, contract)
