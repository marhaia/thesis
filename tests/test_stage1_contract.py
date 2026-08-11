"""P3 / AG-01 and AG-02 regression tests for the public Stage-1 x19 contract."""

from __future__ import annotations

import io
import os
import struct
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for endpoint tests")

import app as app_module  # noqa: E402
from app import app  # noqa: E402
from stage2.task_descriptor import (  # noqa: E402
    SEARCH_MODE_WEIGHTS,
    SPECIFICITY_WEIGHTS,
    TASK_TYPE_WEIGHTS,
    TIME_PRESSURE_WEIGHTS,
)
from stage2.user_profile import PROFILE_PRESETS  # noqa: E402


EXPECTED_STAGE1_NAMES = [
    "shannon_entropy",
    "edge_density",
    "feature_congestion",
    "subband_entropy",
    "layout_symmetry",
    "chromatic_coherence",
    "visual_hierarchy",
    "interactive_element_density",
    "saliency_dispersion",
    "saliency_entropy",
    "saliency_coverage",
    "saliency_peak_count",
    "saliency_center_bias",
    "hceye_fixation_ratio_proxy",
    "hceye_duration_ratio_proxy",
    "hceye_exploration_ratio_proxy",
    "hceye_aoi_sensitivity_proxy",
    "hceye_highlight_effectiveness_proxy",
    "experimental_layout_complexity_index",
]


def _png_bytes() -> bytes:
    image = np.full((96, 128, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (70, 45), (40, 100, 210), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _CanonicalMeasurement:
    whitespace_ratio = 0.65
    text_density = 0.12
    text_density_source = "p3_contract_fixture"

    @staticmethod
    def readability_report_native():
        return None

    @staticmethod
    def as_dict():
        return {
            "whitespace_ratio": 0.65,
            "text_density": 0.12,
            "text_density_source": "p3_contract_fixture",
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
    classification = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        app_module,
        "_compute_visual_cached",
        lambda image_hash, image_path: (dict(visual), False),
    )
    monkeypatch.setattr(
        app_module,
        "_predict_saliency_cached",
        lambda image_hash, image_path: (heatmap.copy(), classification.copy(), False),
    )

    class _SaliencyModel:
        pass

    monkeypatch.setattr(app_module, "_get_saliency_model", lambda: _SaliencyModel())

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


def _post(client, **context):
    payload = {
        "task_type": "search",
        "target_specificity": "medium",
        "time_pressure": "medium",
        "search_mode": "known_item",
        "profile_preset": "neutral",
    }
    payload.update(context)
    payload["image"] = (io.BytesIO(_png_bytes()), "p3.png")
    response = client.post(
        "/api/cognitive-load",
        data=payload,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def _vector_bytes(response):
    return np.asarray(response["stage1_feature_vector"], dtype=np.float32).tobytes()


def _score_bytes(response):
    score = response["layout"]["experimental_complexity_index"]
    return struct.pack("!d", score)


def test_public_stage1_boundary_is_exact_named_float32_x19(client):
    response = _post(client)

    assert "full_feature_vector" not in response
    assert response["vector_dimensions"] == 19
    assert response["stage1_vector_dtype"] == "float32"
    assert response["stage1_feature_names"] == EXPECTED_STAGE1_NAMES
    assert len(response["stage1_feature_vector"]) == 19
    assert all(isinstance(value, float) for value in response["stage1_feature_vector"])

    expected_values = (
        [response["visual_features"][name] for name in EXPECTED_STAGE1_NAMES[:8]]
        + [response["saliency_features"][name] for name in EXPECTED_STAGE1_NAMES[8:13]]
        + [response["hceye_proxy_features"][name] for name in EXPECTED_STAGE1_NAMES[13:]]
    )
    expected_bytes = np.asarray(expected_values, dtype=np.float32).tobytes()
    assert _vector_bytes(response) == expected_bytes


def test_stage1_boundary_is_bit_identical_across_every_task_profile_selection(client):
    baseline = _post(client)
    baseline_vector = _vector_bytes(baseline)
    baseline_score = _score_bytes(baseline)

    selections = {
        "task_type": tuple(TASK_TYPE_WEIGHTS),
        "target_specificity": tuple(SPECIFICITY_WEIGHTS),
        "time_pressure": tuple(TIME_PRESSURE_WEIGHTS),
        "search_mode": tuple(SEARCH_MODE_WEIGHTS),
        "profile_preset": tuple(PROFILE_PRESETS),
    }
    for field, values in selections.items():
        for value in values:
            response = _post(client, **{field: value})
            assert _vector_bytes(response) == baseline_vector, (field, value)
            assert _score_bytes(response) == baseline_score, (field, value)
            assert response["stage1_feature_names"] == EXPECTED_STAGE1_NAMES
            assert response["vector_dimensions"] == 19


def test_stage2_modifiers_remain_separate_from_stage1_layout(client):
    neutral = _post(client)
    maximal = _post(
        client,
        task_type="decision",
        target_specificity="low",
        time_pressure="high",
        search_mode="exploratory",
        profile_preset="stress_sensitive",
    )

    assert neutral["layout"] == maximal["layout"]
    assert set(neutral["layout"]) == {"experimental_complexity_index"}
    assert (
        neutral["context_adjusted_experimental_outputs"]
        != maximal["context_adjusted_experimental_outputs"]
    )
    assert neutral["task_descriptor"]["modifier"] != maximal["task_descriptor"]["modifier"]
    assert neutral["big_five_profile"]["modifier"] != maximal["big_five_profile"]["modifier"]
