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


def test_extract_rejects_empty_map():
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(np.zeros((0, 0), dtype=np.float32))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_extract_rejects_constant_feature_map(value):
    # Constant 0, constant 0.5 and constant 1 all have zero dynamic range and
    # must be rejected (mirrors postprocess_saliency's constant guard).
    m = np.full((120, 160), value, dtype=np.float32)
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_features_within_documented_ranges():
    # Every returned feature must be finite and within its documented range:
    # dispersion/center_bias/entropy/coverage in [0, 1]; peak_count a
    # non-negative integer-valued count.
    for seed in range(4):
        feats = extract_saliency_features(_valid_map(seed))
        for key in ("saliency_dispersion", "saliency_center_bias",
                    "saliency_entropy", "saliency_coverage"):
            v = feats[key]
            assert np.isfinite(v), f"{key} not finite"
            assert -1e-9 <= v <= 1.0 + 1e-9, f"{key}={v} out of [0, 1]"
        pk = feats["saliency_peak_count"]
        assert np.isfinite(pk) and pk >= 0
        assert float(pk) == float(int(pk)), "peak_count must be integer-valued"


def test_extract_rejects_subtly_negative_map():
    # A single -5e-7 pixel in an otherwise valid map must be REJECTED, not
    # tolerated by a 1e-6 band and then clipped to constant zero territory.
    m = _valid_map().astype(np.float64)
    m[10, 10] = -5e-7
    assert float(m.min()) < 0.0
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_rejects_subtly_above_one_map():
    # A single 1.0 + 5e-7 pixel must be REJECTED (strict upper bound).
    m = _valid_map().astype(np.float64)
    m[10, 10] = 1.0 + 5e-7
    assert float(m.max()) > 1.0
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_rejects_clip_to_zero_example():
    # The exact example the independent review reproduced: a map whose only
    # out-of-range content is a tiny negative that np.clip would flatten to 0.
    # It must be rejected up front, never clipped into plausible features.
    m = np.zeros((120, 160), dtype=np.float64)
    m[:] = -5e-7  # would clip to constant 0.0
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_rejects_clip_to_one_example():
    # Mirror case: a map slightly above 1.0 everywhere would clip to constant 1.
    m = np.full((120, 160), 1.0 + 5e-7, dtype=np.float64)
    with pytest.raises(InvalidSaliencyMapError):
        extract_saliency_features(m)


def test_extract_accepts_valid_very_low_dynamic_range_map():
    # A legitimate non-constant map fully inside [0, 1] with a TINY dynamic
    # range must be accepted exactly (no tolerance band may reject it).
    m = np.full((120, 160), 0.5, dtype=np.float64)
    m[60, 80] = 0.5 + 1e-4  # non-constant, well inside [0, 1]
    feats = extract_saliency_features(m)
    assert set(feats.keys()) == _FEATURE_KEYS
    for v in feats.values():
        assert np.isfinite(v)


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


def test_postprocess_all_negative_nonconstant_ok():
    # An all-negative but non-constant raw map is valid: min-max maps it to
    # exact [0, 1] (the largest, i.e. least-negative, pixel becomes 1).
    postprocess_saliency, _ = _import_postprocess()
    raw = (np.random.default_rng(7).random((512, 512, 1)) - 1.5).astype(np.float32)
    assert float(raw.max()) < 0.0
    out = postprocess_saliency(raw, 200, 200)
    assert np.isfinite(out).all()
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_postprocess_rejects_nonfinite_output():
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.random.default_rng(3).random((512, 512, 1)).astype(np.float32)
    raw[0, 0, 0] = np.nan
    with pytest.raises(InvalidSaliencyOutputError):
        postprocess_saliency(raw, 200, 200)


def test_postprocess_rejects_pos_inf_output():
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.random.default_rng(4).random((512, 512, 1)).astype(np.float32)
    raw[1, 1, 0] = np.inf
    with pytest.raises(InvalidSaliencyOutputError):
        postprocess_saliency(raw, 200, 200)


def test_postprocess_rejects_neg_inf_output():
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.random.default_rng(5).random((512, 512, 1)).astype(np.float32)
    raw[2, 2, 0] = -np.inf
    with pytest.raises(InvalidSaliencyOutputError):
        postprocess_saliency(raw, 200, 200)


@pytest.mark.parametrize("value", [0.0, 0.7, -0.7])
def test_postprocess_rejects_constant_output(value):
    # Constant positive, zero and negative raw outputs all have zero dynamic
    # range and must be rejected before resize can inject float noise.
    postprocess_saliency, InvalidSaliencyOutputError = _import_postprocess()
    raw = np.full((512, 512, 1), value, dtype=np.float32)
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


def _constant_map_saliency(*args, **kwargs):
    # Simulates a degenerate map that slips past postprocess_saliency and only
    # fails inside extract_saliency_features (constant / zero dynamic range).
    return (np.full((64, 64), 0.5, dtype=np.float32),
            np.zeros(6, dtype=np.float32), False)


def _subtly_negative_saliency(*args, **kwargs):
    # A non-constant map whose single out-of-range pixel is -5e-7. This must be
    # rejected by the strict [0, 1] check (not tolerated then clipped).
    yy, xx = np.mgrid[0:64, 0:64]
    hm = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 12.0 ** 2)))
    hm = (hm - hm.min()) / (hm.max() - hm.min())
    hm = hm.astype(np.float32)
    hm[0, 0] = np.float32(-5e-7)
    return hm, np.zeros(6, dtype=np.float32), False


def _subtly_above_one_saliency(*args, **kwargs):
    yy, xx = np.mgrid[0:64, 0:64]
    hm = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 12.0 ** 2)))
    hm = (hm - hm.min()) / (hm.max() - hm.min())
    hm = hm.astype(np.float64)
    hm[0, 0] = 1.0 + 5e-7
    return hm, np.zeros(6, dtype=np.float32), False


@pytest.mark.parametrize("url", ["/api/saliency", "/api/cognitive-load"])
def test_constant_map_at_cache_boundary_fails_closed(client, monkeypatch, url):
    # A constant heatmap returned at the _predict_saliency_cached boundary must
    # be rejected by extract_saliency_features and surfaced as a controlled 503
    # (this is the feature-extraction layer, distinct from postprocess_saliency
    # which rejects a constant RAW model output earlier).
    import app as app_module
    monkeypatch.setattr(app_module, "_predict_saliency_cached",
                        _constant_map_saliency)
    r = client.post(
        url, data={"image": (io.BytesIO(_png_bytes()), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 503, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("error") == "saliency_unavailable"
    assert body.get("saliency_used") is False
    for banned in ("cognitive_load_index", "layout", "full_feature_vector"):
        assert banned not in body


@pytest.mark.parametrize("boundary_fn", [_subtly_negative_saliency,
                                         _subtly_above_one_saliency])
@pytest.mark.parametrize("url", ["/api/saliency", "/api/cognitive-load"])
def test_subtly_out_of_range_map_at_boundary_fails_closed(
        client, monkeypatch, url, boundary_fn):
    # A subtly out-of-range map (-5e-7 or 1.0 + 5e-7) supplied at the cache
    # boundary must NOT be clipped into plausible features: both endpoints must
    # return a sanitized 503.
    import app as app_module
    monkeypatch.setattr(app_module, "_predict_saliency_cached", boundary_fn)
    r = client.post(
        url, data={"image": (io.BytesIO(_png_bytes()), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 503, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("error") == "saliency_unavailable"
    assert body.get("saliency_used") is False
    for banned in ("cognitive_load_index", "layout", "full_feature_vector"):
        assert banned not in body
    text = r.get_data(as_text=True)
    assert "Traceback" not in text
    assert ROOT not in text


@pytest.mark.parametrize("url", ["/api/saliency", "/api/cognitive-load"])
def test_failure_cause_is_logged_but_response_sanitized(client, monkeypatch, url):
    # The full cause must be logged server-side (app.logger.exception) while the
    # client response stays generic (no exception string, path or traceback).
    import logging
    import app as app_module

    secret = "SENTINEL_secret_cause_marker_12345"

    def _raise_with_marker(*a, **k):
        raise RuntimeError(secret)

    monkeypatch.setattr(app_module, "_predict_saliency_cached", _raise_with_marker)

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    app_module.app.logger.addHandler(handler)
    try:
        r = client.post(
            url, data={"image": (io.BytesIO(_png_bytes()), "s.png")},
            content_type="multipart/form-data",
        )
    finally:
        app_module.app.logger.removeHandler(handler)

    assert r.status_code == 503
    text = r.get_data(as_text=True)
    # Client response must NOT contain the underlying cause / traceback / path.
    assert secret not in text
    assert "Traceback" not in text
    assert ROOT not in text
    # Server-side log MUST contain the cause with traceback info.
    logged = "\n".join(
        (rec.getMessage() + "\n" +
         (logging.Formatter().formatException(rec.exc_info) if rec.exc_info else ""))
        for rec in records
    )
    assert secret in logged, "underlying cause was not logged server-side"


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


# ---------------------------------------------------------------------------
# E. Real-checkpoint ENDPOINT coverage (marked; real UMSI++, not a mock).
#
# Saliency itself is REAL here (the whole point). The only thing mocked is OCR
# (cognitive.text_reader.compute_readability -> None), which is an unrelated,
# very heavy easyocr/torch dependency; text_density then uses its neutral
# fallback. Jokinen, visual complexity, canonical layout and HCEye all run for
# real. No other mocks are applied.
# ---------------------------------------------------------------------------

@pytest.mark.real_checkpoint
@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH),
                    reason="authoritative umsi++.hdf5 checkpoint not present")
def test_real_saliency_endpoint_success(client):
    pytest.importorskip("tensorflow", reason="TensorFlow is required for umsi_model")
    r = client.post(
        "/api/saliency",
        data={"image": (io.BytesIO(_png_bytes()), "real.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert set(body["features"].keys()) == _FEATURE_KEYS
    assert body.get("classification_used") is False
    assert body.get("classification_status") == "disabled_unvalidated"
    for banned in ("classification", "predicted_class", "design_classification",
                   "probabilities", "out_of_domain"):
        assert banned not in body


@pytest.mark.real_checkpoint
@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH),
                    reason="authoritative umsi++.hdf5 checkpoint not present")
def test_real_cognitive_load_endpoint_success(client, monkeypatch):
    pytest.importorskip("tensorflow", reason="TensorFlow is required for umsi_model")
    # ONLY OCR is mocked (heavy, unrelated). Saliency stays real.
    import cognitive.text_reader as tr
    monkeypatch.setattr(tr, "compute_readability", lambda *a, **k: None)

    r = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_png_bytes()), "real.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("saliency_used") is True
    assert "cognitive_load_index" in body
    assert np.isfinite(body["cognitive_load_index"])
    assert set(body["saliency_features"].keys()) == _FEATURE_KEYS
    for banned in ("design_classification", "predicted_class", "probabilities",
                   "out_of_domain"):
        assert banned not in body


# ---------------------------------------------------------------------------
# C (cleanup). Repository-hygiene regression: the previously-missed diagnostic
# scripts and documentation must not resurrect the semantic-classification
# vocabulary, and the diagnostic scripts must be import-safe (main()-guarded).
# ---------------------------------------------------------------------------

def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


def test_diagnostic_scripts_have_no_semantic_classification():
    for rel in ("cognitive/test_jokinen.py", "saliency/test_full_pipeline.py"):
        src = _read(rel)
        assert "DESIGN_CLASSES" not in src, f"{rel} still references DESIGN_CLASSES"
        assert "Predicted class" not in src, f"{rel} still prints a predicted class"
        assert "aux_classif" in src, f"{rel} should use the aux_classif label"
        assert "UNVALIDATED" in src, f"{rel} must label the aux tensor unvalidated"


def test_diagnostic_scripts_are_main_guarded():
    # Heavy imports/execution must live inside main() so pytest collection never
    # triggers inference or writes files.
    for rel in ("cognitive/test_jokinen.py", "saliency/test_full_pipeline.py"):
        src = _read(rel)
        assert "def main(" in src, f"{rel} must define main()"
        assert 'if __name__ == "__main__":' in src, f"{rel} needs a main guard"


def test_full_pipeline_has_no_warning_suppression_and_fails_hard():
    src = _read("saliency/test_full_pipeline.py")
    assert "TF_CPP_MIN_LOG_LEVEL" not in src
    assert "filterwarnings('ignore')" not in src
    assert 'filterwarnings("ignore")' not in src
    # No unqualified "fully operational" / parity claim after a partial run.
    assert "fully operational" not in src
    assert "SUCCESS - UMSI++ saliency pipeline fully operational" not in src


def test_jokinen_script_no_unconditional_success():
    src = _read("cognitive/test_jokinen.py")
    assert "SUCCESS — Jokinen 2020 model operational!" not in src


def test_docs_have_no_semantic_classification_labels():
    html = _read("planning/status/dev_status_presentation.html")
    assert "Design-Klassifikation (6-class Softmax)" not in html
    doc = _read("stage1/DOCUMENTATION.md")
    assert "out_classif (6-Klassen)" not in doc
    heavy = _read(".github/workflows/saliency-heavy.yml")
    # Stale intro that claimed it runs only the marked test must be gone.
    assert "runs the marked real-checkpoint regression test" not in heavy


def test_diagnostic_scripts_collectable_no_inference():
    # A pure import of each diagnostic module must succeed quickly and define a
    # main() without executing it. (If module-level code ran a model, importing
    # here would attempt to load TF/weights and fail or hang.) We also assert no
    # saliency/output artifacts are written as a side effect of import.
    import importlib
    out_dir = os.path.join(ROOT, "saliency", "output")
    before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    for mod in ("cognitive.test_jokinen", "saliency.test_full_pipeline"):
        m = importlib.import_module(mod)
        assert hasattr(m, "main"), f"{mod}.main missing"
    after = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    assert before == after, "importing a diagnostic script wrote output files"
