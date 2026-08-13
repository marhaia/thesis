"""XR-04 regressions for the explicit current P2/AG-05 evidence contract."""

from __future__ import annotations

import json
import copy
from pathlib import Path

from stage1.tools import umsi_production_postprocess_gate as gate
from stage1.tools import umsi_production_postprocess_current_gate as current_gate


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "stage1" / "evidence"
CURRENT_CONTRACT = (
    EVIDENCE_ROOT / "umsi_production_postprocess_gate_current_contract.json"
)
CURRENT_REPORT = EVIDENCE_ROOT / "p2_ag05_current_report_v2.json"
HISTORICAL_CONTRACT = (
    EVIDENCE_ROOT / "umsi_production_postprocess_gate_contract.json"
)
HISTORICAL_REPORT = EVIDENCE_ROOT / "p2_ag05_real_weight_report_a133653.json"

HISTORICAL_CONTRACT_SHA256 = (
    "a9207c5b64b976e35ad336d86ff5401712881f7cd8962431b444e12fad6cdab8"
)
HISTORICAL_REPORT_SHA256 = (
    "a310c787c996630f53e39cef98a677dd76849eec82528b95f2c1b27b7ac66d85"
)
CURRENT_PRODUCTION_E2E_SHA256 = {
    "lowcontrast": "65d9dfbb7cf2fc83958b72da041e3ec8e3bc84f18b886ad120d985aaeb03e8d8",
    "ui1": "d3df5b6a8c8231f30afaa37bdb31bb4da77dc0f456184b60599bc53cb79bb6e4",
    "ui2": "3314504f002d73aaf016cece1cd3c37e0c985d04fe9bb5b447bfe80bda83bca6",
    "ui3": "4bb9f15ba8887d0d4daa741925d80d8d5f532cd2b9928467a44238577e46654b",
    "uniform": "0dc22d79e05cf6ef63242602c165f4f45b10e7566d2254b70cce55ff64c71779",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_historical_p2_contract_and_report_remain_byte_identical():
    assert gate.sha256_file(HISTORICAL_CONTRACT) == HISTORICAL_CONTRACT_SHA256
    assert gate.sha256_file(HISTORICAL_REPORT) == HISTORICAL_REPORT_SHA256


def test_current_contract_is_explicit_successor_with_unchanged_thresholds():
    contract = gate.load_and_validate_contract(CURRENT_CONTRACT)

    assert contract["contract_version"] == "p2_ag05_current_v2"
    assert contract["lifecycle"] == "CURRENT_SUCCESSOR"
    assert contract["remediation_finding"] == "XR-04"
    assert contract["supersedes_execution_contract"] == {
        "filename": HISTORICAL_CONTRACT.name,
        "sha256": HISTORICAL_CONTRACT_SHA256,
        "report_filename": HISTORICAL_REPORT.name,
        "report_sha256": HISTORICAL_REPORT_SHA256,
        "mutation_policy": "immutable_historical_evidence",
    }
    assert contract["thresholds"] == {
        "endpoint_self_abs_max": 0.0,
        "e2e_abs_max": 0.02,
        "feature_abs_max": 0.02,
        "peak_count_exact": True,
        "peak_positions_exact": True,
    }
    assert contract["expected_production_e2e_sha256"] == (
        CURRENT_PRODUCTION_E2E_SHA256
    )


def test_current_contract_pins_and_verifies_current_source_bytes():
    contract = gate.load_and_validate_contract(CURRENT_CONTRACT)
    observed = gate.verify_contract_identities(
        contract,
        repo_root=ROOT,
        reference_contract_path=EVIDENCE_ROOT
        / contract["historical_reference_contract"]["filename"],
    )

    assert observed == {
        "historical_reference_contract": (
            "da9c968b5ee0fdc95b0f8d2d7dfbf4462e8513836e8b658ed0b1838c8619e1fc"
        ),
        **{
            label: source["sha256"]
            for label, source in contract["pinned_sources"].items()
        },
    }
    assert contract["pinned_sources"]["production_saliency_features"][
        "sha256"
    ] == "f425375767739c5a668e810d14b25bdb85f523b038aea3d38f1475479f678f45"


def test_current_real_weight_report_passes_all_five_exact_value_gates():
    contract = gate.load_and_validate_contract(CURRENT_CONTRACT)
    report = _load(CURRENT_REPORT)

    assert report["overall_verdict"] == "PASS"
    assert report["scope"] == "P2_AG_05_ONLY"
    assert report["fixture_order"] == list(CURRENT_PRODUCTION_E2E_SHA256)
    assert report["p2_contract_sha256"] == gate.sha256_file(CURRENT_CONTRACT)
    assert report["thresholds"] == contract["thresholds"]
    for fixture, expected_sha in contract[
        "expected_production_e2e_sha256"
    ].items():
        result = report["per_fixture"][fixture]
        observed = result["observed"]
        assert result["status"] == "PASS"
        assert result["passed"] is True
        assert observed["production_e2e_sha256"] == expected_sha
        assert observed["output_contract"]["passed"] is True
        assert observed["production_endpoint_self_check"][
            "max_abs_diff_float64"
        ] == 0.0
        assert observed["reference_value_check"]["passed"] is True
        assert observed["feature_check"]["passed"] is True
        assert observed["peak_position_check"]["positions_exact"] is True
    checks = current_gate.validate_current_report(contract, report)
    assert checks["passed"] is True
    assert checks["exact_production_e2e_sha256"] == (
        contract["expected_production_e2e_sha256"]
    )
    assert checks["historical_artifacts_mutated"] is False


def test_current_launcher_rejects_one_same_shape_output_identity_mutation():
    contract = gate.load_and_validate_contract(CURRENT_CONTRACT)
    report = copy.deepcopy(_load(CURRENT_REPORT))
    report["per_fixture"]["ui2"]["observed"][
        "production_e2e_sha256"
    ] = "0" * 64

    try:
        current_gate.validate_current_report(contract, report)
    except gate.ContractError as exc:
        assert "production E2E SHA-256 mismatch: ui2" in str(exc)
    else:
        raise AssertionError("mutated current output identity must fail closed")


def test_current_report_identity_block_matches_current_contract():
    contract = gate.load_and_validate_contract(CURRENT_CONTRACT)
    report = _load(CURRENT_REPORT)

    assert report["identities"]["historical_reference_contract"] == (
        contract["historical_reference_contract"]["sha256"]
    )
    for label, source in contract["pinned_sources"].items():
        assert report["identities"][label] == source["sha256"]
    assert report["identities"]["umsi++.hdf5"] == (
        "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
    )
    assert report["identities"][
        "umsi_resize_causality_arrays_6a17288_consolidated.npz"
    ] == "9e933c1071924f180c32cc6327363ca5760b5968d497414a9f7939919b630079"
