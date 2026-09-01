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
from stage2.scenario_proxy import (  # noqa: E402
    TASK_CATEGORIES,
    TIME_PRESSURE_DIRECTIONS,
)


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
    "stage2_scenario_proxy",
    "cross_signal_review",
    "jokinen_diagnostic",
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
        lambda self, **kwargs: _valid_jokinen_result(),
    )

    app.config.update(TESTING=True)
    return app.test_client()


def _post(client, **context):
    payload = {
        "task_type": "search",
        "time_pressure": "medium",
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


def _valid_jokinen_element():
    return {
        "id": 0,
        "search_time_s": 1.5,
        "search_time_std_s": 0.2,
        "fixation_count": 3.0,
        "bbox": [10.0, 12.0, 30.0, 20.0],
        "center": [25.0, 22.0],
        "color_category": "blue",
    }


def _valid_jokinen_result():
    element = _valid_jokinen_element()
    return {
        "per_element": [element],
        "mean_search_time_s": 1.5,
        "max_search_time_s": 1.5,
        "min_search_time_s": 1.5,
        "search_time_std_s": 0.0,
        "predicted_difficulty": "moderate",
        "n_elements": 1,
        "n_simulations": 100,
    }


def _valid_native_element():
    return {
        "id": 0,
        "bbox": [10.0, 12.0, 30.0, 20.0],
        "center": [25.0, 22.0],
        "area": 600.0,
        "dominant_color_hsv": [210.0, 0.8, 0.7],
        "color_category": "blue",
        "angular_size": 2.5,
        "contrast_ratio": 4.5,
        "wcag_aa_pass": True,
    }


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
        def __init__(self, *_args, **_kwargs):
            pass

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


def test_stage1_boundary_is_bit_identical_across_every_stage2_v1_scenario(client):
    baseline = _post(client)
    baseline_vector = _vector_bytes(baseline)
    baseline_score = _score_bytes(baseline)

    for task_type in TASK_CATEGORIES:
        for time_pressure in TIME_PRESSURE_DIRECTIONS:
            response = _post(
                client,
                task_type=task_type,
                time_pressure=time_pressure,
            )
            scenario = (task_type, time_pressure)
            assert _vector_bytes(response) == baseline_vector, scenario
            assert _score_bytes(response) == baseline_score, scenario
            assert response["stage1_feature_names"] == EXPECTED_STAGE1_NAMES
            assert response["vector_dimensions"] == 19


def test_stage2_v1_is_qualitative_non_score_bearing_and_has_no_legacy_outputs(client):
    low = _post(client, task_type="navigation", time_pressure="low")
    high = _post(client, task_type="decision", time_pressure="high")

    assert low["layout"] == high["layout"]
    assert low["stage2_scenario_proxy"]["direction"] == "lower"
    assert high["stage2_scenario_proxy"]["direction"] == "higher"
    for response in (low, high):
        proxy = response["stage2_scenario_proxy"]
        assert proxy["score_bearing"] is False
        assert proxy["numeric_modifier"] is None
        assert response["cross_signal_review"]["score_bearing"] is False
        assert response["jokinen_diagnostic"]["requested"] is False
        for legacy in (
            "task_descriptor",
            "big_five_profile",
            "base_experimental_outputs",
            "context_adjusted_experimental_outputs",
            "prediction_source",
            "trained_model_requested",
            "trained_model_available",
        ):
            assert legacy not in response


@pytest.mark.parametrize(
    "unsupported",
    ["target_specificity", "search_mode", "profile_preset", "use_trained_model"],
)
def test_stage2_v1_rejects_legacy_context_fields(client, unsupported):
    payload = {
        "image": (io.BytesIO(_png_bytes()), "legacy.png"),
        "task_type": "search",
        "time_pressure": "medium",
        unsupported: "legacy-value",
    }
    response = client.post(
        "/api/cognitive-load", data=payload, content_type="multipart/form-data"
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"


@pytest.mark.parametrize("empty_field", ["task_type", "time_pressure"])
def test_stage2_v1_rejects_explicit_empty_scenario_fields(client, empty_field):
    payload = {
        "image": (io.BytesIO(_png_bytes()), "empty-scenario.png"),
        "task_type": "search",
        "time_pressure": "medium",
    }
    payload[empty_field] = ""

    response = client.post(
        "/api/cognitive-load", data=payload, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"


@pytest.mark.parametrize(
    "context",
    [
        {"include_jokinen_diagnostic": "true", "display_preset": ""},
        {"include_jokinen_diagnostic": "true", "display_preset": "   "},
        {"include_jokinen_diagnostic": "true", "display_preset": "PHONE"},
        {"include_jokinen_diagnostic": "TRUE"},
    ],
)
def test_stage2_v1_rejects_empty_or_case_variant_optional_values(client, context):
    payload = {
        "image": (io.BytesIO(_png_bytes()), "invalid-optional.png"),
        "task_type": "search",
        "time_pressure": "medium",
        **context,
    }

    response = client.post(
        "/api/cognitive-load", data=payload, content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"


@pytest.mark.parametrize(
    "field,form_value,query_value",
    [
        ("task_type", "search", ""),
        ("task_type", "search", "SEARCH"),
        ("time_pressure", "medium", ""),
        ("include_jokinen_diagnostic", "true", "false"),
        ("display_preset", "phone", "PHONE"),
    ],
)
def test_stage2_v1_rejects_duplicate_form_and_query_values(
    client, field, form_value, query_value
):
    payload = {
        "image": (io.BytesIO(_png_bytes()), "duplicate-context.png"),
        "task_type": "search",
        "time_pressure": "medium",
        "include_jokinen_diagnostic": "true",
    }
    payload[field] = form_value

    response = client.post(
        "/api/cognitive-load",
        query_string=[(field, query_value)],
        data=payload,
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"


def test_stage2_v1_rejects_duplicate_values_within_one_request_source(client):
    response = client.post(
        "/api/cognitive-load",
        query_string=[("task_type", "search"), ("task_type", "SEARCH")],
        data={
            "image": (io.BytesIO(_png_bytes()), "duplicate-query.png"),
            "time_pressure": "medium",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"


def test_optional_jokinen_is_off_by_default_and_explicit_when_requested(client):
    default = _post(client)
    requested = _post(client, include_jokinen_diagnostic="true")

    assert default["jokinen_diagnostic"]["status"] == "not_requested"
    assert requested["jokinen_diagnostic"]["requested"] is True
    assert requested["jokinen_diagnostic"]["status"] == "complete"
    assert requested["jokinen_diagnostic"]["score_bearing"] is False
    assert default["layout"] == requested["layout"]
    assert _vector_bytes(default) == _vector_bytes(requested)


@pytest.mark.parametrize(
    "nonfinite",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "mean_search_time_s",
        "search_time_s",
        "search_time_std_s",
        "fixation_count",
        "bbox",
        "center",
    ],
)
def test_optional_jokinen_contains_nonfinite_results(
    client, monkeypatch, field, nonfinite
):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    element = result["per_element"][0]
    if field == "mean_search_time_s":
        result[field] = nonfinite
    elif field == "bbox":
        element[field] = np.array([10.0, nonfinite, 30.0, 20.0])
    elif field == "center":
        element[field] = [25.0, nonfinite]
    else:
        element[field] = nonfinite

    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    response = client.post(
        "/api/cognitive-load",
        data={
            "image": (io.BytesIO(_png_bytes()), "nonfinite-jokinen.png"),
            "task_type": "search",
            "time_pressure": "medium",
            "include_jokinen_diagnostic": "true",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["jokinen_diagnostic"]["requested"] is True
    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert body["jokinen_diagnostic"]["score_bearing"] is False
    assert body["cross_signal_review"]["score_bearing"] is False
    assert len(body["stage1_feature_vector"]) == 19
    raw = response.get_data(as_text=True)
    assert "NaN" not in raw
    assert "Infinity" not in raw


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {},
        {"mean_search_time_s": 1.0},
        {"mean_search_time_s": 1.0, "per_element": "not-a-list"},
        {"mean_search_time_s": "1.0", "per_element": []},
        {"mean_search_time_s": True, "per_element": []},
        {"mean_search_time_s": 1.0, "per_element": [{}]},
        {"mean_search_time_s": 1.0, "per_element": ["not-an-object"]},
        {
            "mean_search_time_s": 1.0,
            "per_element": [{**_valid_jokinen_element(), "id": 1.5}],
        },
    ],
)
def test_optional_jokinen_contains_malformed_results(client, monkeypatch, result):
    import cognitive.jokinen_model

    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize(
    "field,bad_value",
    [
        pytest.param("bbox", [-1.0, 12.0, 30.0, 20.0], id="bbox-negative-x"),
        pytest.param("bbox", [10.0, -1.0, 30.0, 20.0], id="bbox-negative-y"),
        pytest.param("bbox", [10.0, 12.0, -1.0, 20.0], id="bbox-negative-width"),
        pytest.param("bbox", [10.0, 12.0, 30.0, -1.0], id="bbox-negative-height"),
        pytest.param("bbox", [10.0, 12.0, 0.0, 20.0], id="bbox-zero-width"),
        pytest.param("bbox", [10.0, 12.0, 30.0, 0.0], id="bbox-zero-height"),
        pytest.param("center", [-1.0, 22.0], id="center-negative-x"),
        pytest.param("center", [25.0, -1.0], id="center-negative-y"),
        pytest.param("center", [25.0, 23.0], id="center-bbox-inconsistent"),
    ],
)
def test_optional_jokinen_contains_invalid_geometry_without_changing_scores(
    client, monkeypatch, field, bad_value
):
    import cognitive.jokinen_model

    baseline = _post(client, include_jokinen_diagnostic="true")
    assert baseline["jokinen_diagnostic"]["status"] == "complete"

    result = _valid_jokinen_result()
    result["per_element"][0][field] = bad_value
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    observed = _post(client, include_jokinen_diagnostic="true")

    assert observed["jokinen_diagnostic"]["status"] == "unavailable"
    assert observed["jokinen_diagnostic"]["result"] is None
    assert observed["jokinen_diagnostic"]["score_bearing"] is False
    assert _vector_bytes(observed) == _vector_bytes(baseline)
    assert _score_bytes(observed) == _score_bytes(baseline)
    assert observed["layout"] == baseline["layout"]


def test_optional_jokinen_contains_nonfinite_derived_arithmetic(client, monkeypatch):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    first = result["per_element"][0]
    first["fixation_count"] = 1e308
    second = dict(first)
    second["id"] = 1
    second["bbox"] = [50.0, 12.0, 30.0, 20.0]
    second["center"] = [65.0, 22.0]
    result["per_element"] = [first, second]
    result["n_elements"] = 2
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize("container_position", ["bbox", "per_element"])
def test_optional_jokinen_rejects_finite_numpy_containers(
    client, monkeypatch, container_position
):
    import cognitive.jokinen_model

    element = _valid_jokinen_element()
    if container_position == "bbox":
        element["bbox"] = np.array([10.0, 12.0, 30.0, 20.0])
        per_element = [element]
    else:
        per_element = np.array([element], dtype=object)
    result = _valid_jokinen_result()
    result["per_element"] = per_element
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("max_search_time_s", "bad"),
        ("max_search_time_s", True),
        ("min_search_time_s", "bad"),
        ("min_search_time_s", False),
        ("search_time_std_s", "bad"),
        ("search_time_std_s", True),
        ("predicted_difficulty", 7),
        ("n_elements", "one"),
        ("n_elements", True),
        ("n_simulations", "many"),
        ("n_simulations", False),
    ],
)
def test_optional_jokinen_contains_malformed_declared_aggregate_fields(
    client, monkeypatch, field, bad_value
):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    result[field] = bad_value
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize(
    "field",
    [
        "max_search_time_s",
        "min_search_time_s",
        "search_time_std_s",
        "predicted_difficulty",
        "n_elements",
        "n_simulations",
    ],
)
def test_optional_jokinen_contains_missing_declared_aggregate_fields(
    client, monkeypatch, field
):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    result.pop(field)
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize("field", ["search_time_std_s", "color_category"])
def test_optional_jokinen_contains_missing_declared_per_element_fields(
    client, monkeypatch, field
):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    result["per_element"][0].pop(field)
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize(
    "mutation",
    [
        "unexpected_top_level",
        "unexpected_element",
        "count_mismatch",
        "aggregate_mismatch",
        "difficulty_mismatch",
    ],
)
def test_optional_jokinen_requires_one_exact_result_schema(
    client, monkeypatch, mutation
):
    import cognitive.jokinen_model

    result = _valid_jokinen_result()
    if mutation == "unexpected_top_level":
        result["extra"] = 1
    elif mutation == "unexpected_element":
        result["per_element"][0]["extra"] = 1
    elif mutation == "count_mismatch":
        result["n_elements"] = 2
    elif mutation == "aggregate_mismatch":
        result["max_search_time_s"] = 2.0
    else:
        result["predicted_difficulty"] = "easy"
    monkeypatch.setattr(
        cognitive.jokinen_model.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: result,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


def test_empty_jokinen_producer_matches_the_complete_public_schema():
    from cognitive.jokinen_model import JokinenParams, JokinenSearchModel

    result = JokinenSearchModel(
        JokinenParams(n_simulations=7)
    ).predict_search_times(elements=[], image_shape=(64, 64))

    assert set(result) == app_module.JOKINEN_RESULT_FIELDS
    assert result["per_element"] == []
    assert result["n_elements"] == 0
    assert result["n_simulations"] == 7
    assert result["predicted_difficulty"] == "trivial"
    assert app_module._validated_jokinen_result(result) == result


@pytest.mark.parametrize("container_position", ["center", "elements"])
def test_native_detection_rejects_finite_numpy_containers(
    client, monkeypatch, container_position
):
    import cognitive.element_detector

    element = _valid_native_element()
    if container_position == "center":
        element["center"] = np.array([25.0, 22.0])
        elements = [element]
    else:
        elements = np.array([element], dtype=object)
    monkeypatch.setattr(
        cognitive.element_detector,
        "detect_elements",
        lambda image: elements,
    )

    body = _post(client, include_jokinen_diagnostic="true")

    assert body["detected_elements"] == []
    assert body["jokinen_diagnostic"]["status"] == "unavailable"
    assert body["jokinen_diagnostic"]["result"] is None
    assert len(body["stage1_feature_vector"]) == 19


@pytest.mark.parametrize("include_jokinen", [False, True])
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("contrast_ratio", np.nan),
        ("contrast_ratio", np.inf),
        ("contrast_ratio", -np.inf),
        ("bbox", [10.0, np.nan, 30.0, 20.0]),
        ("bbox", np.array([10.0, np.inf, 30.0, 20.0])),
        ("bbox", "not-a-box"),
        ("center", [25.0, -np.inf]),
        ("center", [True, 22.0]),
    ],
)
def test_invalid_native_elements_never_break_the_stage1_response(
    client, monkeypatch, include_jokinen, field, bad_value
):
    import cognitive.element_detector

    element = _valid_native_element()
    element[field] = bad_value
    monkeypatch.setattr(
        cognitive.element_detector,
        "detect_elements",
        lambda image: [element],
    )

    context = {
        "include_jokinen_diagnostic": "true" if include_jokinen else "false"
    }
    body = _post(client, **context)

    assert len(body["stage1_feature_vector"]) == 19
    assert body["detected_elements"] == []
    expected_status = "unavailable" if include_jokinen else "not_requested"
    assert body["jokinen_diagnostic"]["status"] == expected_status
    assert body["jokinen_diagnostic"]["result"] is None


@pytest.mark.parametrize(
    "extra",
    [
        {"undeclared_context": "value"},
        {"display_preset": "desktop"},
        {
            "include_jokinen_diagnostic": "true",
            "display_preset": "automotive_legacy",
        },
    ],
)
def test_stage2_v1_rejects_undeclared_or_misplaced_optional_fields(client, extra):
    payload = {
        "image": (io.BytesIO(_png_bytes()), "invalid-context.png"),
        "task_type": "search",
        "time_pressure": "medium",
        **extra,
    }
    response = client.post(
        "/api/cognitive-load", data=payload, content_type="multipart/form-data"
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "stage2_scenario_invalid"
