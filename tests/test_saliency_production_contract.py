"""
Fail-closed saliency production-contract tests.
=================================================

Proves the NEW fail-closed contract for the mandatory saliency stage and the
score-relevant saliency norms in the real ``/api/cognitive-load`` production
route and in the underlying ``HCEyeFeatureExtractor`` norms-loading /
percentile-normalisation code.

Scope (see the implementation run report for the full rationale):
  1. A saliency model-init / weights / inference / feature-extraction failure
     must surface as a structured, non-200 response with no full score field
     -- never a silent degrade to a partial-but-200 result.
  2. A missing, unreadable, or syntactically invalid production feature-norms
     file must raise / surface visibly -- never silently substitute an empty
     reference structure.
  3. A missing or invalid (non-numeric / non-finite) score-relevant saliency
     norm block must raise / surface visibly -- never silently substitute a
     neutral 0.5.
  4. The consumer-side percentile normalisation of a valid raw value against
     the real production norms is cross-checked against an independently
     computed application of the same (unchanged) formula.

IMPORTANT: tests here that monkeypatch a failure (e.g. forcing
``_predict_saliency_cached`` to raise, or a real UMSI weights file being
absent) prove the FAIL-CLOSED CONTRACT, not real weighted-model success-path
parity. Real weighted end-to-end parity is a separate, still-open concern
(see the implementation run report) and must not be inferred from these
tests.

Temporary norms files are created exclusively under pytest's ``tmp_path``;
the production file at ``stage1/data/results/feature_norms.json`` is never
written to by any test in this module.
"""
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for these tests")

import app as app_module  # noqa: E402  (import after sys.path setup)
from app import app, SaliencyUnavailableError  # noqa: E402
from hceye.hceye_features import HCEyeFeatureExtractor, FeatureNormsError  # noqa: E402

PRODUCTION_NORMS_PATH = os.path.join(
    ROOT, "stage1", "data", "results", "feature_norms.json"
)
EXPECTED_PRODUCTION_NORMS_SHA256 = (
    "d17b3698c2e4b0016a091955e374203f3d6f2eb258c5de6a47a3dc0b6ba9736f"
)
REQUIRED_SALIENCY_KEYS = (
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture(autouse=True)
def _reset_norms_cache(monkeypatch):
    """Ensure each test starts from a clean, unpatched norms cache/path, and
    restores the real production path/cache afterwards regardless of what an
    individual test patches."""
    monkeypatch.setattr(app_module, "_feature_norms", None)
    yield
    app_module._feature_norms = None


@pytest.fixture(autouse=True)
def _valid_score_driving_ocr(monkeypatch):
    """Keep these saliency/norm tests on a valid P4 layout/OCR success path."""
    import cognitive.text_reader as text_reader

    monkeypatch.setattr(
        text_reader,
        "compute_readability",
        lambda _image, elements: {
            "n_elements": len(elements),
            "n_text_elements": 0,
            "text_elements": [],
        },
    )


def _small_png_bytes() -> bytes:
    im = np.full((300, 400, 3), 240, np.uint8)
    cv2.rectangle(im, (40, 40), (160, 90), (30, 30, 200), -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


def _valid_production_norms_dict() -> dict:
    """A full, valid copy of the real production norms structure, for tests
    that need to mutate a single value/block without touching the real file
    on disk."""
    with open(PRODUCTION_NORMS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Real Flask route: saliency failure must not yield a 200 + full score.
# ---------------------------------------------------------------------------

def test_saliency_inference_failure_yields_structured_non_200(client, monkeypatch):
    def _broken_predict(image_hash, image_path):
        raise RuntimeError("simulated UMSI++ inference failure")

    monkeypatch.setattr(app_module, "_predict_saliency_cached", _broken_predict)

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code != 200
    data = resp.get_json()
    assert data["analysis_complete"] is False
    assert data["error"]["code"] == "saliency_unavailable"
    assert isinstance(data["error"]["message"], str) and data["error"]["message"]

    # No field that could be mistaken for a successfully computed full score.
    for forbidden_key in (
        "cognitive_load_index", "adjusted_prediction", "base_prediction",
        "layout", "cognitive_load_features", "full_feature_vector",
        "stage1_feature_vector", "stage1_feature_names",
        "stage1_vector_dtype", "vector_dimensions",
    ):
        assert forbidden_key not in data, (
            f"fail-closed response must not contain '{forbidden_key}'"
        )


def test_saliency_model_init_failure_yields_structured_non_200(client, monkeypatch):
    """Simulates the exact real-world blocker found in this worktree: the
    UMSI++ weights file is absent (gitignored, not present in this checkout).
    """
    def _broken_get_model():
        raise FileNotFoundError(
            "UMSI++ weights not found (simulated: gitignored binary absent "
            "in this worktree)"
        )

    monkeypatch.setattr(app_module, "_get_saliency_model", _broken_get_model)

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code != 200
    data = resp.get_json()
    assert data["analysis_complete"] is False
    assert data["error"]["code"] == "saliency_unavailable"


def test_saliency_failure_response_never_leaks_artificial_path(client, monkeypatch):
    """An underlying saliency-stage exception whose own text embeds an
    absolute weights/user path must never let that path (or a stacktrace)
    reach the client response. The route must always emit a fixed, generic
    ``saliency_unavailable`` message regardless of the inner exception text.
    """
    sentinel_path = (
        "/Users/definitely_not_a_real_user_zzz42/secret_home/"
        "model_weights/saliency_models/UMSI++/umsi++.hdf5"
    )

    def _broken_predict(image_hash, image_path):
        raise RuntimeError(f"failed to load weights from {sentinel_path}")

    monkeypatch.setattr(app_module, "_predict_saliency_cached", _broken_predict)

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code != 200
    raw_text = resp.get_data(as_text=True)
    data = resp.get_json()

    assert data["analysis_complete"] is False
    assert data["error"]["code"] == "saliency_unavailable"
    assert data["error"]["message"] == (
        "Saliency computation failed; a complete analysis could not "
        "be produced."
    )

    # No fragment of the artificial path, nor a stacktrace, anywhere in the
    # raw response body.
    for forbidden_fragment in (
        sentinel_path, "definitely_not_a_real_user_zzz42", "secret_home",
        "umsi++.hdf5", "Traceback", "RuntimeError",
    ):
        assert forbidden_fragment not in raw_text, (
            f"response must not contain '{forbidden_fragment}'"
        )

    for forbidden_key in (
        "cognitive_load_index", "adjusted_prediction", "base_prediction",
        "layout", "cognitive_load_features", "full_feature_vector",
        "stage1_feature_vector", "stage1_feature_names",
        "stage1_vector_dtype", "vector_dimensions",
    ):
        assert forbidden_key not in data


# ---------------------------------------------------------------------------
# 2 & 3. Missing / invalid production norms file (unit-level, tmp_path only).
# ---------------------------------------------------------------------------

def test_missing_norms_file_rejected_at_construction(tmp_path):
    missing_path = tmp_path / "does_not_exist_feature_norms.json"
    with pytest.raises(FeatureNormsError):
        HCEyeFeatureExtractor(feature_norms_path=str(missing_path))


def test_syntactically_invalid_norms_json_rejected(tmp_path):
    bad_path = tmp_path / "invalid_feature_norms.json"
    bad_path.write_text("{not valid json,,,")
    with pytest.raises(FeatureNormsError):
        HCEyeFeatureExtractor(feature_norms_path=str(bad_path))


def test_missing_norms_file_rejected_through_real_route(client, monkeypatch, tmp_path):
    """Full-route proof: an unreadable production norms file must surface as
    a structured, non-200 failure through the real /api/cognitive-load
    endpoint, not a silently degraded 200 response.

    The saliency model stage itself is faked to succeed (real weights are
    unavailable in this worktree), so this isolates the norms-file failure
    specifically rather than reporting the (also-true) saliency_unavailable
    condition.
    """
    size = 64
    heat = np.ones((size, size), dtype=np.float32) * 0.5
    classif = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        app_module, "_predict_saliency_cached",
        lambda image_hash, image_path: (heat, classif, False),
    )
    monkeypatch.setattr(
        app_module, "_FEATURE_NORMS_PATH", tmp_path / "missing_feature_norms.json"
    )

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code != 200
    data = resp.get_json()
    assert data["analysis_complete"] is False
    assert data["error"]["code"] == "saliency_norms_invalid"


def test_missing_norms_file_response_never_leaks_sentinel_path(client, monkeypatch, tmp_path):
    """Full-route proof that the local absolute path of the (missing)
    production norms file never reaches the client: the response must use a
    fixed, generic message, and neither the sentinel path nor any of its
    distinctive path segments may appear anywhere in the raw response body.
    """
    size = 64
    heat = np.ones((size, size), dtype=np.float32) * 0.5
    classif = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        app_module, "_predict_saliency_cached",
        lambda image_hash, image_path: (heat, classif, False),
    )
    sentinel_dir = tmp_path / "definitely_not_a_real_dir_zzz9182" / "secret_user_someone"
    sentinel_path = sentinel_dir / "missing_feature_norms.json"
    monkeypatch.setattr(app_module, "_FEATURE_NORMS_PATH", sentinel_path)

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code != 200
    raw_text = resp.get_data(as_text=True)
    data = resp.get_json()

    assert data["analysis_complete"] is False
    assert data["error"]["code"] == "saliency_norms_invalid"
    assert data["error"]["message"] == (
        "Production feature norms reference is unavailable or invalid; "
        "a complete analysis could not be produced."
    )

    for forbidden_fragment in (
        str(sentinel_path), "definitely_not_a_real_dir_zzz9182",
        "secret_user_someone", "missing_feature_norms.json", "Traceback",
    ):
        assert forbidden_fragment not in raw_text, (
            f"response must not contain '{forbidden_fragment}'"
        )

    for forbidden_key in (
        "cognitive_load_index", "adjusted_prediction", "base_prediction",
        "layout", "cognitive_load_features", "full_feature_vector",
        "stage1_feature_vector", "stage1_feature_names",
        "stage1_vector_dtype", "vector_dimensions",
    ):
        assert forbidden_key not in data


# ---------------------------------------------------------------------------
# 4. Missing required saliency norm block.
# ---------------------------------------------------------------------------

def test_missing_required_saliency_norm_block_rejected(tmp_path):
    norms = _valid_production_norms_dict()
    del norms["features"]["saliency_coverage"]
    bad_path = tmp_path / "norms_missing_block.json"
    bad_path.write_text(json.dumps(norms))

    with pytest.raises(FeatureNormsError, match="saliency_coverage"):
        HCEyeFeatureExtractor(feature_norms_path=str(bad_path))


@pytest.mark.parametrize("key", REQUIRED_SALIENCY_KEYS)
def test_each_required_saliency_norm_block_is_enforced(tmp_path, key):
    """All five promoted saliency norm blocks are individually required, not
    just the two the current highlight-effectiveness formula reads."""
    norms = _valid_production_norms_dict()
    del norms["features"][key]
    bad_path = tmp_path / f"norms_missing_{key}.json"
    bad_path.write_text(json.dumps(norms))

    with pytest.raises(FeatureNormsError):
        HCEyeFeatureExtractor(feature_norms_path=str(bad_path))


# ---------------------------------------------------------------------------
# 5. Invalid (non-numeric / non-finite) required norm value.
# ---------------------------------------------------------------------------

def test_non_numeric_required_norm_value_rejected(tmp_path):
    norms = _valid_production_norms_dict()
    # Mutate an anchor actually consumed by the percentile-normalisation
    # formula (min/p5/p25/p50/p75/p95/max); 'std'/'mean' are not anchors and
    # are intentionally not validated here.
    norms["features"]["saliency_dispersion"]["p50"] = "not-a-number"
    bad_path = tmp_path / "norms_non_numeric.json"
    bad_path.write_text(json.dumps(norms))

    with pytest.raises(FeatureNormsError):
        HCEyeFeatureExtractor(feature_norms_path=str(bad_path))


def test_non_finite_required_norm_value_rejected(tmp_path):
    norms = _valid_production_norms_dict()
    # Python's json module accepts the non-standard NaN/Infinity literals on
    # both dump and load, so this round-trips to a real float('nan').
    norms["features"]["saliency_coverage"]["p50"] = float("nan")
    bad_path = tmp_path / "norms_non_finite.json"
    bad_path.write_text(json.dumps(norms))
    reloaded = json.loads(bad_path.read_text())
    assert math.isnan(reloaded["features"]["saliency_coverage"]["p50"])

    with pytest.raises(FeatureNormsError):
        HCEyeFeatureExtractor(feature_norms_path=str(bad_path))


# ---------------------------------------------------------------------------
# 6. _percentile_normalize(): required=True fails closed; required=False
#    (default, unrelated visual-feature callers) keeps the original neutral
#    behaviour unchanged.
# ---------------------------------------------------------------------------

def test_percentile_normalize_required_raises_on_missing_value():
    extractor = HCEyeFeatureExtractor()
    with pytest.raises(FeatureNormsError):
        extractor._percentile_normalize(None, "saliency_dispersion", required=True)


def test_percentile_normalize_required_raises_on_missing_block():
    extractor = HCEyeFeatureExtractor()
    # Simulate a missing block in the already-loaded in-memory norms without
    # touching any file on disk.
    del extractor.feature_norms["saliency_coverage"]
    with pytest.raises(FeatureNormsError):
        extractor._percentile_normalize(0.05, "saliency_coverage", required=True)


def test_percentile_normalize_non_required_still_returns_neutral_default():
    """Non-required callers (e.g. visual-feature concept mapping, or the
    legitimate saliency-omitted estimation mode) are unaffected by this
    contract and keep the pre-existing neutral-default behaviour."""
    extractor = HCEyeFeatureExtractor()
    assert extractor._percentile_normalize(None, "saliency_dispersion") == 0.5
    assert extractor._percentile_normalize(0.5, "no_such_feature_key") == 0.5


# ---------------------------------------------------------------------------
# 7. Consumer normalisation matches an independently computed application of
#    the existing (unchanged) formula, for a real known raw value against the
#    real production norms.
# ---------------------------------------------------------------------------

def _reference_percentile_normalize(norms_block: dict, value: float) -> float:
    """Independent re-implementation of the piecewise-linear interpolation in
    HCEyeFeatureExtractor._percentile_normalize, written separately here so a
    match is not tautological."""
    anchors = [
        (norms_block["min"], 0.0),
        (norms_block["p5"], 0.05),
        (norms_block["p25"], 0.25),
        (norms_block["p50"], 0.50),
        (norms_block["p75"], 0.75),
        (norms_block["p95"], 0.95),
        (norms_block["max"], 1.0),
    ]
    xs, ys = [], []
    for x, y in anchors:
        if not xs or float(x) > xs[-1]:
            xs.append(float(x))
            ys.append(y)
    return float(np.interp(float(value), xs, ys))


@pytest.mark.parametrize("key", ["saliency_dispersion", "saliency_coverage"])
def test_consumer_normalization_matches_independent_reference(key):
    extractor = HCEyeFeatureExtractor()
    norms_block = extractor.feature_norms[key]
    known_value = float(norms_block["mean"])  # a real, valid raw value

    consumer_result = extractor._percentile_normalize(known_value, key, required=True)
    reference_result = _reference_percentile_normalize(norms_block, known_value)

    assert consumer_result == pytest.approx(reference_result, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. The production norms file used still has the exact expected SHA-256.
# ---------------------------------------------------------------------------

def test_production_norms_file_has_expected_sha256():
    with open(PRODUCTION_NORMS_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == EXPECTED_PRODUCTION_NORMS_SHA256


def test_production_norms_file_has_all_required_saliency_blocks():
    extractor = HCEyeFeatureExtractor()
    for key in REQUIRED_SALIENCY_KEYS:
        assert key in extractor.feature_norms, f"missing block: {key}"


# ---------------------------------------------------------------------------
# 9. Green-path sanity: with a valid (test-synthetic) saliency stage and the
#    real, valid production norms, the route still returns a complete 200
#    result exercising the real mandatory-saliency + normalization code.
#    NOTE: this uses a synthetic heatmap (real weights are unavailable in
#    this worktree, see class docstring). It proves the fail-closed contract
#    does not block a *working* saliency stage; it is NOT proof of real
#    weighted-model success-path parity.
# ---------------------------------------------------------------------------

def test_working_saliency_stage_still_returns_full_result(client, monkeypatch):
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    heat = np.exp(-(((yy - 60) ** 2 + (xx - 60) ** 2) / (2 * 15.0 ** 2)))
    heat = (heat / heat.max()).astype(np.float32)
    classif = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _fake_predict(image_hash, image_path):
        return heat, classif, False

    monkeypatch.setattr(app_module, "_predict_saliency_cached", _fake_predict)

    resp = client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(_small_png_bytes()), "syn.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "error" not in data or not data.get("error")
    assert "experimental_complexity_index" in data["layout"]
    assert "hceye_proxy_features" in data
    assert "layout" in data
    assert set(REQUIRED_SALIENCY_KEYS) <= set(data["saliency_features"].keys())
