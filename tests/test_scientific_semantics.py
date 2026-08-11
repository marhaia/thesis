"""P5 / AG-07..09 scientific-semantics regression tests."""

from __future__ import annotations

import io
import os
import re
import sys
from typing import Any

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for endpoint tests")

import app as app_module  # noqa: E402
from app import app  # noqa: E402


EXPECTED_PROXY_NAMES = [
    "hceye_fixation_ratio_proxy",
    "hceye_duration_ratio_proxy",
    "hceye_exploration_ratio_proxy",
    "hceye_aoi_sensitivity_proxy",
    "hceye_highlight_effectiveness_proxy",
    "experimental_layout_complexity_index",
]

EXPECTED_SCIENTIFIC_SEMANTICS = {
    "construct": "exploratory_project_specific_layout_proxy",
    "validated_cognitive_load_measurement": False,
    "hceye_derivation_reproducible_from_repository": False,
    "hceye_coefficient_provenance": (
        "hash_pinned_external_csv_with_repository_verifier"
    ),
    "umsi_class_label_mapping_verified": False,
}

FORBIDDEN_PUBLIC_KEYS = {
    "classification",
    "predicted_class",
    "design_classification",
    "out_of_domain",
    "probabilities",
    "confidence",
    "cognitive_load_features",
    "cognitive_load_index",
    "cognitive_load_score",
    "base_prediction",
    "adjusted_prediction",
}


def _png_bytes() -> bytes:
    image = np.full((96, 128, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (70, 45), (40, 100, 210), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _CanonicalMeasurement:
    whitespace_ratio = 0.65
    text_density = 0.12
    text_density_source = "p5_contract_fixture"

    @staticmethod
    def readability_report_native():
        return None

    @staticmethod
    def as_dict():
        return {
            "whitespace_ratio": 0.65,
            "text_density": 0.12,
            "text_density_source": "p5_contract_fixture",
        }


@pytest.fixture()
def client(monkeypatch):
    visual = {
        "shannon_entropy": 0.11,
        "edge_density": 0.22,
        "feature_congestion": 0.33,
        "subband_entropy": 0.44,
        "layout_symmetry": 0.55,
        "chromatic_coherence": 0.66,
        "visual_hierarchy": 0.77,
        "interactive_element_density": 0.08,
    }
    yy, xx = np.mgrid[0:64, 0:64]
    heatmap = np.exp(-((xx - 31.0) ** 2 + (yy - 29.0) ** 2) / (2 * 8.0**2))
    heatmap = (heatmap / heatmap.max()).astype(np.float32)
    unverified_head = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.10, 0.80], dtype=np.float32
    )

    monkeypatch.setattr(
        app_module,
        "_compute_visual_cached",
        lambda image_hash, image_path: (dict(visual), False),
    )
    monkeypatch.setattr(
        app_module,
        "_predict_saliency_cached",
        lambda image_hash, image_path: (
            heatmap.copy(), unverified_head.copy(), False
        ),
    )

    def _must_not_resolve_class_labels():
        raise AssertionError("unverified UMSI class labels must not be resolved")

    monkeypatch.setattr(
        app_module, "_get_saliency_model", _must_not_resolve_class_labels
    )

    import canonical_layout
    import cognitive.element_detector
    import cognitive.jokinen_model

    monkeypatch.setattr(
        canonical_layout,
        "measure_canonical_layout",
        lambda image: _CanonicalMeasurement(),
    )
    monkeypatch.setattr(
        cognitive.element_detector, "detect_elements", lambda image: []
    )
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: {"mean_search_time_s": 0.0, "per_element": []},
    )

    app.config.update(TESTING=True)
    return app.test_client()


def _post(client, route: str):
    response = client.post(
        route,
        data={"image": (io.BytesIO(_png_bytes()), "p5.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def _assert_no_unverified_class_semantics(response) -> None:
    body = response.get_json()
    assert not (FORBIDDEN_PUBLIC_KEYS & _all_keys(body))


def test_saliency_route_exposes_numeric_saliency_without_class_semantics(client):
    response = _post(client, "/api/saliency")

    _assert_no_unverified_class_semantics(response)
    assert set(response.get_json()) == {
        "filename",
        "features",
        "heatmap_png_base64",
        "saliency_cache_hit",
    }


def test_score_route_exposes_only_bounded_proxy_semantics(client):
    response = _post(client, "/api/cognitive-load")

    _assert_no_unverified_class_semantics(response)
    body = response.get_json()
    assert body["scientific_semantics"] == EXPECTED_SCIENTIFIC_SEMANTICS
    reproducibility = body["reproducibility"]
    assert reproducibility["schema_id"] == "stage1-study-export-v1"
    assert re.fullmatch(r"[0-9a-f]{40}", reproducibility["source"]["commit"])
    assert reproducibility["source"]["tree_state"] in {"clean", "dirty"}
    for key in (
        "umsi_checkpoint_sha256",
        "feature_norms_sha256",
        "reference_pack_manifest_sha256",
        "runtime_environment_manifest_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", reproducibility[key])
    assert set(body["hceye_proxy_features"]) == set(EXPECTED_PROXY_NAMES)
    assert body["stage1_feature_names"][-6:] == EXPECTED_PROXY_NAMES
    assert set(body["base_experimental_outputs"]) == {
        "layout_complexity_score",
        "search_efficiency_proxy",
        "attention_demand_proxy",
    }
    assert set(body["context_adjusted_experimental_outputs"]) == {
        "layout_complexity_score",
        "search_efficiency_proxy",
        "attention_demand_proxy",
    }
    assert body["prediction_source"] == "project_specific_hceye_heuristic"
    assert body["layout"]["experimental_complexity_index"] == pytest.approx(
        100.0
        * body["hceye_proxy_features"]["experimental_layout_complexity_index"]
    )


def test_p5_semantics_policy_explicitly_records_missing_validation_evidence():
    policy = app_module.SCIENTIFIC_SEMANTICS

    assert policy["construct"] == "exploratory_project_specific_layout_proxy"
    assert policy["validated_cognitive_load_measurement"] is False
    assert policy["hceye_derivation_reproducible_from_repository"] is False
    assert policy["hceye_coefficient_provenance"] == (
        "hash_pinned_external_csv_with_repository_verifier"
    )
    assert policy["umsi_class_label_mapping_verified"] is False
