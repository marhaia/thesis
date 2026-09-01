"""Regression tests for the UEyes development/evaluation split correction."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stage1.tools import build_ueyes_development_reference as m


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "stage1" / "data" / "results"


def _split_records():
    records = {}
    for category in m.AUTHORIZED_CATEGORIES:
        for index in range(m.TRAIN_PER_CATEGORY):
            filename = f"{category}_train_{index:03d}.png"
            records[filename] = {
                "filename": filename,
                "image_id": Path(filename).stem,
                "category": category,
                "block": index // 9,
                "partition": "Train",
            }
        for block in m.TEST_BLOCKS:
            for index in range(m.TEST_PER_CATEGORY_BLOCK):
                filename = f"{category}_test_{block}_{index}.png"
                records[filename] = {
                    "filename": filename,
                    "image_id": Path(filename).stem,
                    "category": category,
                    "block": block,
                    "partition": "Test",
                }
    return records


def _visual_rows(split):
    return {
        filename: {
            "filename": filename,
            "category": record["category"],
            "image_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
        }
        for filename, record in split.items()
    }


def test_official_split_contract_is_exactly_1404_development_81_test():
    split = _split_records()
    m.validate_official_split(split)

    counts = collections.Counter(
        (row["category"], row["partition"]) for row in split.values()
    )
    assert counts == collections.Counter(
        {
            ("desktop", "Train"): 468,
            ("desktop", "Test"): 27,
            ("mobile", "Train"): 468,
            ("mobile", "Test"): 27,
            ("web", "Train"): 468,
            ("web", "Test"): 27,
        }
    )


def test_split_contract_rejects_test_image_mislabeled_as_train():
    split = _split_records()
    target = next(
        row for row in split.values()
        if row["category"] == "web" and row["partition"] == "Test"
    )
    target["partition"] = "Train"

    with pytest.raises(m.DevelopmentReferenceError, match="web Train rows"):
        m.validate_official_split(split)


def test_comma_decimal_block_values_are_parsed_as_official_integer_blocks():
    assert m._parse_block("5,30E+01") == 53
    assert m._parse_block("54") == 54
    with pytest.raises(m.DevelopmentReferenceError):
        m._parse_block("53.5")


def test_candidate_selection_is_deterministic_category_and_block_balanced():
    split = _split_records()
    visual = _visual_rows(split)

    first = m.build_selection(split, visual, seed=m.DEFAULT_SELECTION_SEED)
    second = m.build_selection(split, visual, seed=m.DEFAULT_SELECTION_SEED)

    assert first == second
    selected = [row for row in first if row["status"] == "selected_candidate"]
    reserves = [row for row in first if row["status"] == "reserve_candidate"]
    assert collections.Counter(row["category"] for row in selected) == {
        "desktop": 20,
        "mobile": 20,
        "web": 20,
    }
    assert collections.Counter(row["category"] for row in reserves) == {
        "desktop": 7,
        "mobile": 7,
        "web": 7,
    }
    assert collections.Counter(row["block"] for row in selected) == {
        53: 20,
        54: 20,
        55: 20,
    }
    assert all(row["partition"] == "Test" for row in selected + reserves)


def test_candidate_selection_seed_changes_membership_but_not_contract():
    split = _split_records()
    visual = _visual_rows(split)
    first = m.build_selection(split, visual, seed="seed-a")
    second = m.build_selection(split, visual, seed="seed-b")

    first_selected = {
        row["filename"] for row in first if row["status"] == "selected_candidate"
    }
    second_selected = {
        row["filename"] for row in second if row["status"] == "selected_candidate"
    }
    assert first_selected != second_selected
    assert len(first_selected) == len(second_selected) == 60


def test_manual_exclusion_cannot_silently_reference_non_test_image():
    split = _split_records()
    visual = _visual_rows(split)
    train_name = next(
        name for name, row in split.items() if row["partition"] == "Train"
    )

    with pytest.raises(m.DevelopmentReferenceError, match="not official GUI Test"):
        m.build_selection(
            split,
            visual,
            seed=m.DEFAULT_SELECTION_SEED,
            exclusions={train_name: "not eligible"},
        )


def test_aggregate_uses_sample_standard_deviation_and_linear_percentiles():
    rows = [{"feature": value} for value in (1.0, 2.0, 3.0, 4.0)]
    result = m.aggregate_rows(rows, ("feature",))["feature"]

    assert result["n"] == 4
    assert result["mean"] == 2.5
    assert result["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert result["p50"] == 2.5
    assert result["p5"] == pytest.approx(np.percentile([1, 2, 3, 4], 5))


def test_production_reference_and_selection_artifacts_freeze_official_boundary():
    feature_norms = json.loads((RESULTS / "feature_norms.json").read_text())
    visual_norms = json.loads(
        (RESULTS / "canonical_visual_feature_norms.json").read_text()
    )
    saliency_norms = json.loads(
        (RESULTS / "development_saliency_feature_norms.json").read_text()
    )
    manifest = json.loads(
        (RESULTS / "development_reference_manifest.json").read_text()
    )

    assert feature_norms["meta"]["reference_partition"] == "Train"
    assert feature_norms["meta"]["evaluation_partition_excluded"] == "Test"
    assert feature_norms["meta"]["category_counts"] == {
        "desktop": 468,
        "mobile": 468,
        "web": 468,
    }
    assert feature_norms["meta"]["num_images"] == 1404
    assert all(block["n"] == 1404 for block in feature_norms["features"].values())
    assert visual_norms["num_images"] == 1404
    assert saliency_norms["num_images"] == 1404
    assert manifest["reference_total"] == 1404
    assert manifest["evaluation_total"] == 81

    with (RESULTS / "ueyes_stimulus_selection_v1.csv").open(newline="") as source:
        selection = list(csv.DictReader(source))
    selected = [row for row in selection if row["status"] == "selected_candidate"]
    reserves = [row for row in selection if row["status"] == "reserve_candidate"]
    assert len(selected) == 60
    assert len(reserves) == 21
    assert all(row["official_split"] == "Test" for row in selection)
    assert all(
        row["eligibility_status"] == "technical_pass_manual_review_pending"
        for row in selection
    )


def test_development_manifest_hashes_every_generated_artifact():
    manifest = json.loads(
        (RESULTS / "development_reference_manifest.json").read_text()
    )
    for filename, identity in manifest["outputs"].items():
        path = RESULTS / filename
        assert path.stat().st_size == identity["bytes"]
        assert m.sha256_file(path) == identity["sha256"]


def test_reference_change_does_not_modify_raw_v8_or_s5_extraction_code():
    expected = {
        "saliency/umsi_model.py": (
            "c8d0c85454c95732e44be91caab1c083bd870e160c67f76cd1b04e3d737ca228"
        ),
        "saliency/postprocessing.py": (
            "5380cd66cc2c41221255accd227bf1fe701a01c0b19b66da785c1b1703e499e2"
        ),
        "saliency/saliency_features.py": (
            "f899f6ecf32155d1da4b7c71b817e6f385f36f6051abfcef7638f99e7bb36d7f"
        ),
        "stage1/visual_complexity.py": (
            "6a1cdc1291436347a7433215b7a2a0cd85e85eb700d614e1599a59df6a2a708d"
        ),
        "stage1/canonical_layout.py": (
            "a0cec3f4ccf6d6ac98e17bcf8a0f61018ad74a9478906d5a7b70f4c8acc50e2b"
        ),
    }
    for relative, expected_sha in expected.items():
        assert m.sha256_file(ROOT / relative) == expected_sha
