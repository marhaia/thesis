"""P4 / AG-06 regression tests for score-driving layout/OCR failures."""

from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for endpoint tests")

import app as app_module  # noqa: E402
import canonical_layout as canonical_layout_module  # noqa: E402
import cognitive.element_detector as element_detector_module  # noqa: E402
import cognitive.jokinen_model as jokinen_module  # noqa: E402
import cognitive.text_reader as text_reader_module  # noqa: E402
from app import app, ScoreInputUnavailableError  # noqa: E402


FORBIDDEN_SUCCESS_FIELDS = (
    "cognitive_load_index",
    "base_prediction",
    "adjusted_prediction",
    "layout",
    "cognitive_load_features",
    "stage1_feature_vector",
    "stage1_feature_names",
    "stage1_vector_dtype",
    "vector_dimensions",
)


def _png_bytes() -> bytes:
    image = np.full((96, 128, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (15, 18), (80, 50), (40, 100, 210), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


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
        DESIGN_CLASSES = tuple(f"class_{index}" for index in range(6))

    monkeypatch.setattr(app_module, "_get_saliency_model", lambda: _SaliencyModel())
    app.config.update(TESTING=True)
    return app.test_client()


def _post(client):
    return client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_png_bytes()), "p4.png")},
        content_type="multipart/form-data",
    )


def _assert_p4_failure(response, forbidden_text: str | None = None):
    assert response.status_code == 503
    body = response.get_json()
    assert body == {
        "analysis_complete": False,
        "error": {
            "code": "layout_ocr_unavailable",
            "message": (
                "Required layout/OCR analysis is unavailable; no score was "
                "produced. Check the server's layout/OCR setup and retry."
            ),
        },
    }
    for key in FORBIDDEN_SUCCESS_FIELDS:
        assert key not in body
    raw = response.get_data(as_text=True)
    assert "Traceback" not in raw
    if forbidden_text is not None:
        assert forbidden_text not in raw


def test_layout_exception_returns_structured_non_success_without_score_or_x19(
    client, monkeypatch
):
    sentinel = "/Users/private-layout-path/layout-model.bin"

    def _broken_layout(_image):
        raise RuntimeError(f"layout failed at {sentinel}")

    monkeypatch.setattr(
        canonical_layout_module, "measure_canonical_layout", _broken_layout
    )

    _assert_p4_failure(_post(client), forbidden_text=sentinel)


@pytest.mark.parametrize("mode", ["raises", "returns-none"])
def test_ocr_failure_or_unavailability_never_uses_neutral_substitute(
    client, monkeypatch, mode
):
    sentinel = "/Users/private-ocr-path/reader.bin"
    canonical_elements = [{"id": 0, "bbox": (10, 10, 30, 20)}]
    monkeypatch.setattr(
        canonical_layout_module,
        "detect_elements",
        lambda _image: [dict(element) for element in canonical_elements],
    )

    if mode == "raises":
        def _broken_ocr(_image, _elements):
            raise RuntimeError(f"OCR failed at {sentinel}")

        monkeypatch.setattr(text_reader_module, "compute_readability", _broken_ocr)
    else:
        monkeypatch.setattr(
            text_reader_module, "compute_readability", lambda _image, _elements: None
        )

    _assert_p4_failure(_post(client), forbidden_text=sentinel)


@pytest.mark.parametrize(
    "measurement",
    [
        None,
        SimpleNamespace(
            whitespace_ratio=np.nan,
            text_density=0.2,
            text_density_source="ocr",
        ),
        SimpleNamespace(
            whitespace_ratio=0.5,
            text_density=None,
            text_density_source="ocr",
        ),
        SimpleNamespace(
            whitespace_ratio=0.5,
            text_density=0.5,
            text_density_source="fallback_neutral",
        ),
    ],
    ids=["missing", "nonfinite", "missing-ocr-value", "neutral-fallback"],
)
def test_invalid_or_substituted_score_inputs_are_rejected(measurement):
    with pytest.raises(ScoreInputUnavailableError) as exc_info:
        app_module._validated_layout_score_inputs(measurement)
    assert exc_info.value.code == "layout_ocr_unavailable"


def test_no_elements_is_defined_zero_measurement_not_an_ocr_failure(monkeypatch):
    monkeypatch.setattr(canonical_layout_module, "detect_elements", lambda _image: [])

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("OCR must not run without canonical elements")

    monkeypatch.setattr(text_reader_module, "compute_readability", _must_not_run)
    image = np.full((64, 96, 3), 240, dtype=np.uint8)

    measurement = canonical_layout_module.measure_canonical_layout(image)

    assert measurement.whitespace_ratio == 1.0
    assert measurement.text_density == 0.0
    assert measurement.text_density_source == "no_elements"


def test_defined_no_element_measurement_can_complete_the_endpoint(client, monkeypatch):
    monkeypatch.setattr(canonical_layout_module, "detect_elements", lambda _image: [])
    monkeypatch.setattr(element_detector_module, "detect_elements", lambda _image: [])
    monkeypatch.setattr(
        text_reader_module,
        "compute_readability",
        lambda *_args, **_kwargs: pytest.fail("OCR must not run without elements"),
    )
    monkeypatch.setattr(
        jokinen_module.JokinenSearchModel,
        "predict_search_times",
        lambda self, **kwargs: {"mean_search_time_s": 0.0, "per_element": []},
    )

    response = _post(client)

    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    assert body["hceye_inputs"]["text_density"] == 0.0
    assert body["hceye_inputs"]["text_density_source"] == "no_elements"
    assert len(body["stage1_feature_vector"]) == 19
