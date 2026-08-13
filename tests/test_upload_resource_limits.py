"""Regression guards for decoded-image and screen-set resource bounds."""

from __future__ import annotations

import io
import os
import sys

import pytest
from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for upload tests")

import app as app_module  # noqa: E402
from app import app, InvalidImageUploadError  # noqa: E402


SINGLE_IMAGE_ROUTES = (
    "/api/analyze",
    "/api/saliency",
    "/api/search-time",
    "/api/scanpath-to-target?target_id=0",
    "/api/cognitive-load",
    "/api/learning-curve",
)
SCREEN_SET_ROUTES = (
    "/api/screen-consistency",
    "/api/product-learning",
)


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def _png_bytes(size=(96, 72), mode="RGB") -> bytes:
    image = Image.new(mode, size, 1 if mode == "1" else (220, 230, 240))
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _large_compressed_png() -> bytes:
    # Mode "1" keeps the test allocation small while producing a valid PNG
    # whose 4096x4096 header would expand beyond the production pixel/byte cap.
    return _png_bytes((4096, 4096), mode="1")


def _assert_resource_failure(response):
    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {
        "analysis_complete": False,
        "error": {
            "code": "image_resource_limit",
            "message": app_module.IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE,
        },
    }


def test_large_compressed_header_is_rejected_before_opencv_decode(monkeypatch):
    def _must_not_decode(*_args, **_kwargs):
        raise AssertionError("oversized image reached full OpenCV decode")

    monkeypatch.setattr(cv2, "imdecode", _must_not_decode)
    with pytest.raises(InvalidImageUploadError) as exc_info:
        app_module._validate_uploaded_image_bytes(_large_compressed_png())

    assert exc_info.value.code == "image_resource_limit"


@pytest.mark.parametrize("route", SINGLE_IMAGE_ROUTES)
def test_every_single_image_route_rejects_decoded_resource_expansion(
    client, monkeypatch, tmp_path, route
):
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)
    response = client.post(
        route,
        data={"image": (io.BytesIO(_large_compressed_png()), "large.png")},
        content_type="multipart/form-data",
    )

    _assert_resource_failure(response)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("route", ("/api/analyze", "/api/saliency"))
def test_visual_and_saliency_routes_map_corrupt_bytes_to_structured_400(
    client, monkeypatch, tmp_path, route
):
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path)
    response = client.post(
        route,
        data={"image": (io.BytesIO(b"not-an-image"), "corrupt.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json()["error"]["code"] == "invalid_image"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("route", SCREEN_SET_ROUTES)
def test_screen_set_single_file_decoder_failure_is_structured_json(client, route):
    response = client.post(
        route,
        data={"image": (io.BytesIO(b"not-an-image"), "corrupt.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.is_json
    assert "Cannot read image" in response.get_json()["error"]


@pytest.mark.parametrize("route", SCREEN_SET_ROUTES)
def test_screen_set_rejects_single_frame_over_per_image_limit(client, route):
    response = client.post(
        route,
        data={"image": (io.BytesIO(_large_compressed_png()), "large.png")},
        content_type="multipart/form-data",
    )

    _assert_resource_failure(response)


@pytest.mark.parametrize("route", SCREEN_SET_ROUTES)
def test_screen_set_enforces_cumulative_pixel_budget(client, monkeypatch, route):
    monkeypatch.setattr(app_module, "MAX_SCREEN_SET_PIXELS", 1000)
    monkeypatch.setattr(app_module, "MAX_SCREEN_SET_DECODED_BYTES", 100000)
    first = _png_bytes((30, 30))
    second = _png_bytes((30, 30))
    response = client.post(
        route,
        data={
            "images": [
                (io.BytesIO(first), "first.png"),
                (io.BytesIO(second), "second.png"),
            ]
        },
        content_type="multipart/form-data",
    )

    _assert_resource_failure(response)


@pytest.mark.parametrize("route", SCREEN_SET_ROUTES)
def test_screen_set_enforces_cumulative_decoded_byte_budget(
    client, monkeypatch, route
):
    monkeypatch.setattr(app_module, "MAX_SCREEN_SET_PIXELS", 100000)
    monkeypatch.setattr(app_module, "MAX_SCREEN_SET_DECODED_BYTES", 5000)
    first = _png_bytes((30, 30))
    second = _png_bytes((30, 30))
    response = client.post(
        route,
        data={
            "images": [
                (io.BytesIO(first), "first.png"),
                (io.BytesIO(second), "second.png"),
            ]
        },
        content_type="multipart/form-data",
    )

    _assert_resource_failure(response)


def test_extreme_aspect_ratio_is_rejected_after_minimum_size_boundary():
    with pytest.raises(InvalidImageUploadError) as exc_info:
        app_module._validate_uploaded_image_bytes(
            _png_bytes((2100, 100), mode="1")
        )

    assert exc_info.value.code == "image_resource_limit"
