"""P5 / AG-08 regression tests for HCEye coefficient provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hceye.hceye_features import HCEYE_COEFFICIENTS  # noqa: E402
from scripts.hceye_coefficient_provenance import (  # noqa: E402
    DEFAULT_MANIFEST,
    HCEyeProvenanceError,
    verify_csv,
)


def _manifest() -> dict:
    with DEFAULT_MANIFEST.open(encoding="utf-8") as source:
        return json.load(source)


def test_manifest_hash_pins_external_primary_source():
    manifest = _manifest()
    source = manifest["source"]

    assert source == {
        "dataset": "HCEye",
        "artifact_filename": "fixation_AOI_metrics_final.csv",
        "dataset_record_url": "https://osf.io/x8p9b/",
        "paper_doi_url": "https://doi.org/10.1145/3655610",
        "sha256": (
            "e6251a354afd97cabce1253e5c345ae8d90c25acc8ee5062b4ffa11249003fee"
        ),
        "bytes": 12475528,
        "repository_policy": "external_hash_pinned_source_not_committed",
    }
    assert manifest["dataset_identity"] == {
        "rows": 4049,
        "participants": 27,
        "images": 150,
        "cognitive_load_conditions": ["Absent", "High", "Low"],
        "highlight_conditions": ["Absent", "Dynamic", "Static"],
    }


def test_coefficient_matrix_exactly_covers_production_constants():
    manifest = _manifest()
    matrix = manifest["coefficients"]

    assert set(matrix) == set(HCEYE_COEFFICIENTS)
    assert len(matrix) == 17
    for name, spec in matrix.items():
        assert round(spec["source_value"], spec["round_digits"]) == spec["stored_value"]
        assert HCEYE_COEFFICIENTS[name] == spec["stored_value"]
        assert spec["aggregation"]
        assert isinstance(spec["used_in_production"], bool)


def test_unreproducible_unused_frequency_std_is_not_retained():
    manifest = _manifest()
    removed = manifest["removed_legacy_metadata"]["frequency_reduction_std"]

    assert "frequency_reduction_std" not in HCEYE_COEFFICIENTS
    assert removed["legacy_value"] == 0.12
    assert removed["per_image_ratio_sample_std"] == pytest.approx(
        0.12655052552042195
    )
    assert "not reproducible" in removed["reason"]


def test_verifier_fails_closed_before_parsing_wrong_source_bytes(tmp_path):
    bogus_csv = tmp_path / "fixation_AOI_metrics_final.csv"
    bogus_csv.write_bytes(b"not-the-pinned-hceye-source")

    with pytest.raises(HCEyeProvenanceError, match="byte-size mismatch"):
        verify_csv(bogus_csv)
