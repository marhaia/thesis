"""P6 runtime/reference identity, cache, and study-export regressions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1"))

import app as app_module  # noqa: E402
import reproducibility as repro  # noqa: E402


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(autouse=True)
def _clear_identity_caches():
    repro.load_runtime_environment_manifest.cache_clear()
    repro.load_reference_pack_manifest.cache_clear()
    repro.verify_runtime_environment.cache_clear()
    repro.layout_ocr_identity.cache_clear()
    repro.saliency_cache_identity.cache_clear()
    repro.visual_cache_identity.cache_clear()
    repro.resolve_source_state.cache_clear()
    app_module._saliency_cache.clear()
    app_module._visual_cache.clear()
    yield
    repro.load_runtime_environment_manifest.cache_clear()
    repro.load_reference_pack_manifest.cache_clear()
    repro.verify_runtime_environment.cache_clear()
    repro.layout_ocr_identity.cache_clear()
    repro.saliency_cache_identity.cache_clear()
    repro.visual_cache_identity.cache_clear()
    repro.resolve_source_state.cache_clear()
    app_module._saliency_cache.clear()
    app_module._visual_cache.clear()


def test_pinned_runtime_and_reference_manifests_validate_exact_artifact_bytes():
    runtime, runtime_sha = repro.load_runtime_environment_manifest()
    reference, reference_sha = repro.load_reference_pack_manifest()

    assert runtime_sha == (
        "915cb746edf82da2e86840645815441e26cd590d5aae1e894e2dc9e729e72fde"
    )
    assert reference_sha == (
        "2aa7d45a92158a3a26bfd4fc38ae79b33bd2fc81827eace55ad9198b7aba6987"
    )
    assert runtime["platform"] == {
        "operating_system": "macOS 26.3",
        "machine": "arm64",
        "python": "3.9.6",
    }
    assert runtime["dependency_freeze"]["lock_file_sha256"] == (
        "319201eb160b44ec3d47b2e931e58f4d7ba657b2607bb2a91ea82f54eec57bf4"
    )
    feature_norms = reference["reference_pack"]["feature_norms"]
    assert feature_norms["num_images"] == 1404
    assert feature_norms["reference_partition"] == "Train"
    assert feature_norms["evaluation_partition_excluded"] == "Test"


def test_reference_pack_freezes_corpus_and_statistical_methods():
    manifest, _ = repro.load_reference_pack_manifest()

    provenance = manifest["ueyes_corpus_provenance"]
    assert provenance["dataset"] == "UEyes"
    assert provenance["image_types_csv_sha256"] == (
        "fc22d679849862c0f0476b1d9c270a50383c62304c7635b87cb18a2559a25c68"
    )
    assert provenance["full_authorized_counts"] == {
        "desktop": 495,
        "mobile": 495,
        "web": 495,
    }
    assert provenance["development_reference"] == {
        "official_partition": "Train",
        "counts": {"desktop": 468, "mobile": 468, "web": 468},
        "total": 1404,
        "population_sha256": (
            "4c90612e1d228d8ea8d0e2d6ad1dccb6db8a67ff9180569545e11584071b1801"
        ),
    }
    assert provenance["heldout_evaluation"] == {
        "official_partition": "Test",
        "counts": {"desktop": 27, "mobile": 27, "web": 27},
        "total": 81,
        "population_sha256": (
            "1d98dd2072f99e2bfbdcc5baa46517d5d1bb80fdd113956b30139dc86607f6a8"
        ),
    }
    assert provenance["excluded_categories"] == ["poster"]
    assert provenance["is_subsample"] is False
    stats = manifest["statistical_conventions"]
    assert stats["standard_deviation"] == {
        "convention": "sample standard deviation",
        "ddof": 1,
    }
    assert stats["stored_quantiles"]["method"] == "linear"
    assert stats["stored_quantiles"]["numpy_version"] == "1.26.4"
    assert stats["runtime_percentile_estimate"]["method"] == (
        "piecewise_linear_interpolation"
    )


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "stage1-reference-pack-v2"}),
        encoding="utf-8",
    )
    hash_path = tmp_path / "manifest.sha256"
    hash_path.write_text(f"{'0' * 64}  manifest.json\n", encoding="utf-8")

    with pytest.raises(repro.ReproducibilityError, match="SHA-256 mismatch"):
        repro._load_pinned_manifest(
            manifest, hash_path, "stage1-reference-pack-v2"
        )


def test_study_metadata_exports_commit_checkpoint_norm_and_schema_ids(
    monkeypatch,
):
    commit = "1" * 40
    monkeypatch.setenv("STAGE1_SOURCE_COMMIT", commit)
    monkeypatch.setenv("STAGE1_SOURCE_TREE_STATE", "clean")
    monkeypatch.setenv("STAGE1_RUNTIME_VERIFICATION", "metadata_only")

    metadata = repro.study_reproducibility_metadata()

    assert metadata["source"] == {
        "commit": commit,
        "tree_state": "clean",
        "source": "environment",
    }
    assert metadata["umsi_checkpoint_sha256"] == (
        "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
    )
    assert metadata["easyocr_detector_sha256"] == (
        "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"
    )
    assert metadata["easyocr_recognizer_sha256"] == (
        "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8"
    )
    assert metadata["feature_norms_sha256"] == (
        "4aa326d920cbf936b57da2356dffff7a885d859ed1e8a1422377e0a7a98ed116"
    )
    assert metadata["schema_id"] == "stage1-study-export-v1"
    assert metadata["stage1_vector_schema"] == "stage1-x19-float32-v1"
    assert metadata["runtime_verification"] == {
        "status": "not_verified",
        "mode": "metadata_only",
        "reason": "explicit lightweight-CI bypass",
    }
    for key in (
        "runtime_environment_manifest_sha256",
        "reference_pack_manifest_sha256",
        "easyocr_model_identity_sha256",
        "layout_ocr_identity",
        "saliency_cache_identity",
        "visual_cache_identity",
    ):
        assert SHA256_RE.fullmatch(metadata[key])


def test_exported_source_override_defaults_to_explicit_exported_state(
    monkeypatch,
):
    commit = "2" * 40
    monkeypatch.setenv("STAGE1_SOURCE_COMMIT", commit)
    monkeypatch.delenv("STAGE1_SOURCE_TREE_STATE", raising=False)

    assert repro.resolve_source_state() == {
        "commit": commit,
        "tree_state": "exported",
        "source": "environment",
    }


def test_strict_runtime_verification_checks_all_locked_distributions(monkeypatch):
    lock_path = ROOT / "stage1/environment/requirements-macos-arm64-python3.9.lock.txt"
    locked = repro._locked_distributions(lock_path)
    monkeypatch.setenv("STAGE1_RUNTIME_VERIFICATION", "strict")
    monkeypatch.setattr(repro.platform, "mac_ver", lambda: ("26.3", (), ""))
    monkeypatch.setattr(repro.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(repro.sys, "version_info", (3, 9, 6))
    monkeypatch.setattr(
        repro.importlib.metadata,
        "version",
        lambda name: locked[name],
    )

    result = repro.verify_runtime_environment()

    assert result["status"] == "verified"
    assert result["mode"] == "strict"
    assert result["locked_distributions"] == 97


def test_lightweight_ci_runtime_bypass_is_explicit_in_export(monkeypatch):
    monkeypatch.setenv("STAGE1_RUNTIME_VERIFICATION", "metadata_only")

    assert repro.verify_runtime_environment() == {
        "status": "not_verified",
        "mode": "metadata_only",
        "reason": "explicit lightweight-CI bypass",
    }


def test_saliency_cache_invalidates_when_analysis_identity_changes(monkeypatch):
    calls = []

    class _Model:
        def predict_saliency(self, image_path, return_classif):
            calls.append((str(image_path), return_classif))
            return np.zeros((2, 2), dtype=np.float32), np.zeros(6, dtype=np.float32)

    identity = {"value": "a" * 64}
    monkeypatch.setattr(app_module, "saliency_cache_identity", lambda: identity["value"])
    monkeypatch.setattr(app_module, "_get_saliency_model", lambda: _Model())

    _, _, first_hit = app_module._predict_saliency_cached("image-sha", "image.png")
    _, _, second_hit = app_module._predict_saliency_cached("image-sha", "image.png")
    identity["value"] = "b" * 64
    _, _, changed_identity_hit = app_module._predict_saliency_cached(
        "image-sha", "image.png"
    )

    assert (first_hit, second_hit, changed_identity_hit) == (False, True, False)
    assert len(calls) == 2


def test_visual_cache_invalidates_when_analysis_identity_changes(monkeypatch):
    calls = []
    identity = {"value": "a" * 64}

    def _compute(path):
        calls.append(str(path))
        return {"shannon_entropy": float(len(calls))}

    monkeypatch.setattr(app_module, "visual_cache_identity", lambda: identity["value"])
    monkeypatch.setattr(app_module, "compute_complexity_vector", _compute)

    first, first_hit = app_module._compute_visual_cached("image-sha", "image.png")
    second, second_hit = app_module._compute_visual_cached("image-sha", "image.png")
    identity["value"] = "b" * 64
    changed, changed_identity_hit = app_module._compute_visual_cached(
        "image-sha", "image.png"
    )

    assert (first_hit, second_hit, changed_identity_hit) == (False, True, False)
    assert first == second == {"shannon_entropy": 1.0}
    assert changed == {"shannon_entropy": 2.0}
    assert len(calls) == 2


def test_visual_cache_identity_binds_layout_ocr_identity(monkeypatch):
    monkeypatch.setattr(repro, "verify_runtime_environment", lambda: {})
    monkeypatch.setattr(
        repro,
        "load_runtime_environment_manifest",
        lambda: ({}, "1" * 64),
    )
    monkeypatch.setattr(
        repro,
        "load_reference_pack_manifest",
        lambda: (
            {"reference_pack": {"feature_norms": {"sha256": "2" * 64}}},
            "3" * 64,
        ),
    )
    identity = {"value": "4" * 64}
    monkeypatch.setattr(repro, "layout_ocr_identity", lambda: identity["value"])

    first = repro.visual_cache_identity()
    repro.visual_cache_identity.cache_clear()
    identity["value"] = "5" * 64
    second = repro.visual_cache_identity()

    assert first != second


def test_active_csv_export_carries_required_reproducibility_columns():
    ui = (ROOT / "stage1" / "ui" / "index.html").read_text(encoding="utf-8")

    assert "reproducibility: data.reproducibility" in ui
    for column in (
        "source_commit",
        "source_tree_state",
        "umsi_checkpoint_sha256",
        "easyocr_model_identity_sha256",
        "easyocr_detector_sha256",
        "easyocr_recognizer_sha256",
        "layout_ocr_identity",
        "feature_norms_sha256",
        "reference_pack_manifest_sha256",
        "runtime_environment_manifest_sha256",
        "study_export_schema",
        "stage1_vector_schema",
        "visual_norms_schema",
        "saliency_norms_schema",
    ):
        assert f'["{column}",' in ui
