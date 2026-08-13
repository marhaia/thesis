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
from app import app, Stage1VectorUnavailableError  # noqa: E402
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

EXPECTED_VECTOR_FAILURE = {
    "analysis_complete": False,
    "error": {
        "code": "stage1_vector_invalid",
        "message": (
            "A required Stage-1 feature was non-finite; no vector or score "
            "was produced."
        ),
    },
}

FORBIDDEN_VECTOR_FAILURE_FIELDS = {
    "layout",
    "hceye_proxy_features",
    "base_experimental_outputs",
    "context_adjusted_experimental_outputs",
    "stage1_feature_vector",
    "stage1_feature_names",
    "stage1_vector_dtype",
    "vector_dimensions",
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
def client(monkeypatch, tmp_path):
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
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)

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


@pytest.mark.parametrize(
    "nonfinite",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
@pytest.mark.parametrize("position", range(19))
def test_stage1_assembler_rejects_every_nonfinite_position(position, nonfinite):
    values = np.linspace(0.1, 1.9, 19, dtype=np.float32)
    values[position] = nonfinite

    with pytest.raises(Stage1VectorUnavailableError) as exc_info:
        app_module._assemble_stage1_vector(
            values[:8], values[8:13], values[13:], EXPECTED_STAGE1_NAMES[13:]
        )

    assert exc_info.value.code == "stage1_vector_invalid"
    assert exc_info.value.message == EXPECTED_VECTOR_FAILURE["error"]["message"]


@pytest.mark.parametrize(
    "nonfinite",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
@pytest.mark.parametrize("position", range(19))
def test_score_route_rejects_every_nonfinite_x19_position(
    client, monkeypatch, position, nonfinite
):
    visual = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(EXPECTED_STAGE1_NAMES[:8])
    }
    saliency = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(EXPECTED_STAGE1_NAMES[8:13])
    }
    hceye = np.array([0.9, 1.1, 0.95, 0.25, 0.6, 0.52], dtype=np.float32)

    if position < 8:
        visual[EXPECTED_STAGE1_NAMES[position]] = nonfinite
    elif position < 13:
        saliency[EXPECTED_STAGE1_NAMES[position]] = nonfinite
    else:
        hceye[position - 13] = nonfinite

    monkeypatch.setattr(
        app_module,
        "_compute_visual_cached",
        lambda image_hash, image_path: (dict(visual), False),
    )

    import saliency.saliency_features as saliency_features_module
    import hceye.hceye_features as hceye_features_module

    monkeypatch.setattr(
        saliency_features_module,
        "extract_saliency_features",
        lambda _heatmap: dict(saliency),
    )

    class _InjectedHCEyeExtractor:
        def extract_features(self, *_args, **_kwargs):
            return hceye.copy()

        def get_feature_names(self):
            return EXPECTED_STAGE1_NAMES[13:]

    monkeypatch.setattr(
        hceye_features_module, "HCEyeFeatureExtractor", _InjectedHCEyeExtractor
    )

    response = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_png_bytes()), "nonfinite.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    body = response.get_json()
    assert body == EXPECTED_VECTOR_FAILURE
    assert FORBIDDEN_VECTOR_FAILURE_FIELDS.isdisjoint(body)
    raw = response.get_data(as_text=True)
    assert "NaN" not in raw
    assert "Infinity" not in raw


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_json_provider_rejects_nonstandard_nonfinite_tokens(nonfinite):
    with pytest.raises(ValueError, match="Out of range float values"):
        app.json.dumps({"value": float(nonfinite)})


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
