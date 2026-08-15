"""Regression tests for requested-saliency failure on auxiliary diagnostics."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from stage1 import app as app_module


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "stage1" / "data" / "screenshots" / "bmw_route.png"
ROUTES = (
    "/api/search-time?n_simulations=1",
    "/api/scanpath-to-target?target_id=0&n_simulations=1",
    "/api/learning-curve?exposures=1,2&n_simulations=1",
)
RESULT_FIELDS = {
    "mean_search_time_s",
    "per_element",
    "scanpath",
    "selected_target",
    "target_load",
    "curve",
    "novice_time_s",
    "expert_time_s",
}


def _upload():
    return {"image": (io.BytesIO(IMAGE.read_bytes()), IMAGE.name)}


def _use_scratch_uploads(monkeypatch, tmp_path):
    def persist(extension, image_bytes):
        path = tmp_path / f"upload{extension}"
        path.write_bytes(image_bytes)
        return path

    monkeypatch.setattr(app_module, "_persist_uploaded_image_bytes", persist)


@pytest.mark.parametrize("route", ROUTES)
def test_requested_saliency_failure_returns_structured_non_success_without_result(
    route, monkeypatch, tmp_path
):
    def fail_saliency(*_args, **_kwargs):
        raise RuntimeError("injected saliency failure")

    monkeypatch.setattr(app_module, "_predict_saliency_cached", fail_saliency)
    _use_scratch_uploads(monkeypatch, tmp_path)
    response = app_module.app.test_client().post(
        route, data=_upload(), content_type="multipart/form-data"
    )

    assert response.status_code == 503
    body = response.get_json()
    assert body == {
        "analysis_complete": False,
        "error": {
            "code": "saliency_unavailable",
            "message": app_module.AUXILIARY_SALIENCY_ERROR_MESSAGE,
        },
    }
    assert RESULT_FIELDS.isdisjoint(body)


@pytest.mark.parametrize("route", ROUTES)
def test_explicit_feature_only_mode_is_separate_and_identified(
    route, monkeypatch, tmp_path
):
    def forbidden_saliency(*_args, **_kwargs):
        pytest.fail("explicit feature-only mode must not call saliency")

    monkeypatch.setattr(app_module, "_predict_saliency_cached", forbidden_saliency)
    _use_scratch_uploads(monkeypatch, tmp_path)
    separator = "&" if "?" in route else "?"
    response = app_module.app.test_client().post(
        f"{route}{separator}use_saliency=false",
        data=_upload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["analysis_complete"] is True
    assert body["analysis_mode"] == "feature_only_explicit"
    assert body["saliency_requested"] is False
    assert body["saliency_used"] is False
    assert "error" not in body
