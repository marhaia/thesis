"""Regression tests for the UMSI++ / saliency integrity fix.

These guard the specific corrections made on ``fix/umsi-saliency-integrity``:

  A. The unvalidated six-way softmax head is never exposed as a semantic
     classification by either saliency endpoint.
  B. Saliency maps are min-max normalised exactly once, and feature extraction
     validates its input instead of silently re-normalising or returning
     plausible numbers for a broken map.
  C. ``/api/cognitive-load`` and ``/api/saliency`` fail closed (HTTP 503) when
     the required saliency stage is unavailable, without leaking filesystem
     paths or tracebacks.

The ``extract_saliency_features`` and endpoint tests are lightweight (no
TensorFlow, no model weights). The ``postprocess_saliency`` tests import
``saliency.umsi_model`` (which pulls TensorFlow) and are skipped when TF is not
installed. The final real-checkpoint test is skipped unless the authoritative
``umsi++.hdf5`` weights are present.

Run: python -m pytest tests/test_saliency_integrity.py -q
"""
import hashlib
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for these tests")

from saliency.saliency_features import (  # noqa: E402
    extract_saliency_features,
    InvalidSaliencyMapError,
)

# Authoritative checkpoint provenance (verified against Aalto + GitHub release).
WEIGHTS_PATH = os.path.join(
    ROOT, "saliency", "weights", "model_weights",
    "saliency_models", "UMSI++", "umsi++.hdf5",
)
EXPECTED_SIZE = 120093896
EXPECTED_SHA256 = (
    "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
)

_FEATURE_KEYS = {
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
}


def _valid_map(seed: int = 0) -> np.ndarray:
    """A finite, [0, 1] saliency map with a couple of Gaussian peaks."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:120, 0:160]
    m = (
        np.exp(-(((xx - 40) ** 2 + (yy - 30) ** 2) / (2 * 15.0 ** 2)))
        + 0.6 * np.exp(-(((xx - 120) ** 2 + (yy - 90) ** 2) / (2 * 10.0 ** 2)))
        + 0.02 * rng.random((120, 160))
    )
    return ((m - m.min()) / (m.max() - m.min())).astype(np.float32)


# ---------------------------------------------------------------------------
# B. extract_saliency_features: input validation, no hidden re-normalisation
# ---------------------------------------------------------------------------

def test_extract_accepts_valid_map_and_returns_all_finite_features():
    feats = extract_saliency_features(_valid_map())
    assert set(feats.keys()) == _FEATURE_KEYS
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_extract_rejects_nan_map():
    m = _valid_map()
    m[0, 0] = np.nan
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_rejects_out_of_range_map():
    # Values above 1.0 must be rejected: the caller must min-max normalise first.
    m = _valid_map() * 3.0
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_rejects_non_2d_map():
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(np.linspace(0, 1, 100).astype(np.float32))


def test_extract_does_not_hidden_renormalise():
    # A correctly [0, 1] map and the SAME map scaled into [0, 0.5] must produce
    # DIFFERENT features: the old code divided by max() internally, which would
    # make these identical. Their difference proves the hidden max-only
    # normalisation was removed.
    base = _valid_map()
    half = (base * 0.5).astype(np.float32)
    f_base = extract_saliency_features(base)
    f_half = extract_saliency_features(half)
    assert f_base["saliency_entropy"] != pytest.approx(f_half["saliency_entropy"])


def test_extract_rejects_negative_map():
    m = _valid_map()
    m[5, 5] = -0.5
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


# ---------------------------------------------------------------------------
# B. postprocess_saliency: min-max once, reject non-finite / constant output
# ---------------------------------------------------------------------------

def _import_postprocess():
    pytest.importorskip("tensorflow", reason="TensorFlow is required for umsi_model")
    from saliency.umsi_model import (  # noqa: E402
        postprocess_saliency,
        InvalidSaliencyOutputError,
    )
    return postprocess_saliency, InvalidSaliencyOutputError


def test_postprocess_minmax_gives_exact_unit_range():
    postprocess_saliency, _ = _import_postprocess()
    raw = np.random.default_rng(1).random((512, 512, 1)).astype(np.float32)
    out = postprocess_saliency(raw, 240, 320)
    assert out.shape == (240, 320)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_postprocess_removes_negatives():
    # The linear decoder head can emit negatives; min-max must map them to >= 0.
    postprocess_saliency, _ = _import_postprocess()
    raw = (np.random.default_rng(2).random((512, 512, 1)) - 0.5).astype(np.float32)
    out = postprocess_saliency(raw, 200, 200)
    assert float(out.min()) >= 0.0
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_postprocess_rejects_nonfinite_output():
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.random.default_rng(3).random((512, 512, 1)).astype(np.float32)
    raw[0, 0, 0] = np.nan
    with pytest.raises(InvalidSaliencyOutputError):
        postprocess_saliency(raw, 200, 200)


def test_postprocess_rejects_constant_output():
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.full((512, 512, 1), 0.7, dtype=np.float32)
    with pytest.raises(InvalidSaliencyOutputError):
        postprocess_saliency(raw, 200, 200)


# ---------------------------------------------------------------------------
# A + C. Endpoint behaviour (saliency mocked; no weights, no TensorFlow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app import app  # noqa: E402  (import after sys.path setup)
    app.config.update(TESTING=True)
    return app.test_client()


def _png_bytes() -> bytes:
    im = np.full((300, 400, 3), 240, np.uint8)
    cv2.rectangle(im, (40, 40), (160, 90), (30, 30, 200), -1)
    cv2.rectangle(im, (200, 200), (320, 250), (30, 180, 30), -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


def _fixed_valid_saliency(*args, **kwargs):
    yy, xx = np.mgrid[0:64, 0:64]
    hm = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 12.0 ** 2)))
    hm = (hm - hm.min()) / (hm.max() - hm.min())
    return hm.astype(np.float32), np.zeros(6, dtype=np.float32), False


def _raise_saliency(*args, **kwargs):
    raise RuntimeError("simulated saliency failure")


def test_saliency_endpoint_exposes_no_semantic_classification(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_predict_saliency_cached",
                        _fixed_valid_saliency)
    r = client.post(
        "/api/saliency",
        data={"image": (io.BytesIO(_png_bytes()), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # No semantic-class prediction of any kind.
    for banned in ("classification", "predicted_class", "design_classification",
                   "probabilities", "out_of_domain"):
        assert banned not in body, f"endpoint exposed banned key {banned!r}"
    # Only a disabled/unvalidated status is allowed.
    assert body.get("classification_used") is False
    assert body.get("classification_status") == "disabled_unvalidated"
    assert set(body["features"].keys()) == _FEATURE_KEYS


def test_saliency_endpoint_fails_closed_with_503(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_predict_saliency_cached", _raise_saliency)
    r = client.post(
        "/api/saliency",
        data={"image": (io.BytesIO(_png_bytes()), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 503, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("error") == "saliency_unavailable"
    assert body.get("saliency_used") is False
    # No filesystem paths or tracebacks leaked.
    text = r.get_data(as_text=True)
    assert "Traceback" not in text
    assert ROOT not in text


def test_cognitive_load_fails_closed_with_503(client, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_predict_saliency_cached", _raise_saliency)
    r = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_png_bytes()), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 503, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("error") == "saliency_unavailable"
    assert body.get("saliency_used") is False
    # None of the study-critical outputs may be present on failure.
    for banned in ("cognitive_load_index", "layout", "full_feature_vector",
                   "design_classification"):
        assert banned not in body, f"503 body still exposed {banned!r}"
    text = r.get_data(as_text=True)
    assert "Traceback" not in text
    assert ROOT not in text


# ---------------------------------------------------------------------------
# E. Real-checkpoint regression test (marked; opt-in via weight presence).
# ---------------------------------------------------------------------------

@pytest.mark.real_checkpoint
@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH),
                    reason="authoritative umsi++.hdf5 checkpoint not present")
def test_real_checkpoint_loads_strict_and_predicts(tmp_path):
    pytest.importorskip("tensorflow", reason="TensorFlow is required for umsi_model")

    # (1) Provenance: exact size + SHA-256 of the authoritative checkpoint.
    assert os.path.getsize(WEIGHTS_PATH) == EXPECTED_SIZE
    h = hashlib.sha256()
    with open(WEIGHTS_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == EXPECTED_SHA256

    # (2) Strict load of the two-output graph.
    from saliency.umsi_model import UMSIPlus
    model = UMSIPlus(WEIGHTS_PATH)

    # (3) Deterministic synthetic fixture.
    im = np.full((256, 384, 3), 235, np.uint8)
    cv2.rectangle(im, (30, 30), (150, 110), (40, 40, 210), -1)
    cv2.rectangle(im, (220, 150), (330, 210), (40, 190, 60), -1)
    fixture = tmp_path / "fixture.png"
    assert cv2.imwrite(str(fixture), im)

    heatmap, aux = model.predict_saliency(str(fixture), return_classif=True)

    # (4) Shape / finiteness / range of the corrected map.
    assert heatmap.shape == (256, 384)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(heatmap.max()) == pytest.approx(1.0, abs=1e-6)
    assert aux.shape == (6,)

    # (5) Repeatability within the same process.
    heatmap2, _ = model.predict_saliency(str(fixture), return_classif=True)
    assert np.array_equal(heatmap, heatmap2)

    # (6) Feature extraction succeeds on the real, corrected map.
    feats = extract_saliency_features(heatmap)
    assert set(feats.keys()) == _FEATURE_KEYS
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} not finite"
