"""Canonical-analysis-resolution regression tests.

These tests validate the ROOT-CAUSE fix for the deterministic scale-sensitivity
defect: the eight Stage-1 visual features are computed on a fixed,
aspect-ratio-preserving canonical resolution, so the same layout rendered at
different native resolutions yields a materially equivalent analysis input.

What is asserted, and why:
  * The property that actually matters is the endpoint effect. All five of the
    HCEye-mapped Stage-1 features (edge_density, feature_congestion,
    interactive_element_density, layout_symmetry, visual_hierarchy — see
    HCEYE_FEATURE_MAP) feed the HCEye-derived cognitive-load index, so the tests
    assert that this index is stable across 1x / 2x / 3x renderings of the same
    layout, and additionally report every raw feature gap and every normalized
    HCEye input for transparency (a raw-feature threshold alone is not a
    faithful success criterion, because a residual can be large in raw units yet
    negligible after percentile-normalization and weighting).
  * The canonicalization helper's shape / aspect-ratio / interpolation /
    edge-case contract is covered directly.

The existing headline-level scale test in ``test_smoke.py`` is intentionally
left untouched.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))
sys.path.insert(0, os.path.join(ROOT, "stage1", "tools"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for these tests")

import visual_complexity  # noqa: E402
from visual_complexity import (  # noqa: E402
    canonicalize_for_analysis,
    CANONICAL_LONG_SIDE,
    MIN_CANONICAL_INPUT_LONG_SIDE,
    MIN_CANONICAL_INPUT_SHORT_SIDE,
    ImageTooSmallError,
    FEATURE_KEYS,
)

# Shared, single-source analysis helpers (also used by the committed evidence
# script stage1/tools/canonical_scale_eval.py), so the regression test and the
# audit artifacts measure the pipeline identically.
from canonical_scale_eval import (  # noqa: E402
    FIXTURES,
    HCEYE_MAPPED_STAGE1,
    PIXEL_SCALE_DRIVERS,
    canonical_features,
    normalized_hceye_inputs,
    hceye_rule_index_no_saliency_no_ocr,
)


# ---------------------------------------------------------------------------
# Scale-stability guards.
#
# HONEST STATUS: these thresholds are POST-HOC regression guards, not
# pre-registered materiality thresholds. They were chosen AFTER observing the
# residuals reported by the tests below, and are set roughly an order of
# magnitude above the worst observed value so the suite catches a gross
# regression without over-claiming a tighter guarantee than the fix delivers.
#
#   * ENDPOINT_ABS_GUARD — the meaningful criterion. The HCEye cognitive-load
#     index lives in [0, 1]; the worst observed 1x/2x/3x absolute gap across the
#     three fixtures is ~0.0025. A guard of 0.02 is well below any load-band
#     boundary yet an order of magnitude above the observed residual.
#   * DRIVER_NORM_GUARD — a secondary guard on the three inherently
#     pixel-scale-dependent features that the diagnosis proved caused the
#     native-resolution defect (feature_congestion, edge_density,
#     interactive_element_density). After canonicalization their percentile-
#     normalized inputs move by <=~2 % across scales; 0.05 guards that band.
#
# layout_symmetry and visual_hierarchy ALSO feed the index and are reported in
# full, but they are NOT held to DRIVER_NORM_GUARD: visual_hierarchy in
# particular retains a real sub-pixel resampling residual (up to ~20 % raw /
# ~14 % normalized on the text-heavy fixture). The tests demonstrate that this
# residual does not materially move the endpoint index (ENDPOINT_ABS_GUARD),
# rather than pretending the raw feature has a fully standardised scale.
# ---------------------------------------------------------------------------
ENDPOINT_ABS_GUARD = 0.02
DRIVER_NORM_GUARD = 0.05


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures + feature helpers are imported from the
# evidence script above, so the test and the committed audit artifacts render
# and measure the exact same layouts.
# ---------------------------------------------------------------------------
def _features_of(img_bgr):
    """Production 8-feature vector (canonicalized internally) for an array."""
    return canonical_features(img_bgr)


def _gap(vals):
    absgap = max(vals) - min(vals)
    denom = max(abs(float(np.mean(vals))), 1e-9)
    return absgap, absgap / denom


# ---------------------------------------------------------------------------
# 1. Canonicalization shape and aspect-ratio contract
# ---------------------------------------------------------------------------
def test_canonicalize_landscape_sets_long_side_and_keeps_aspect():
    img = np.zeros((600, 900, 3), np.uint8)   # landscape, aspect 3:2
    out = canonicalize_for_analysis(img)
    h, w = out.shape[:2]
    assert max(h, w) == CANONICAL_LONG_SIDE
    # Aspect ratio preserved within one pixel of rounding.
    assert abs((w / h) - (900 / 600)) < (1.0 / min(h, w))
    assert w > h  # still landscape (not stretched/rotated)


def test_canonicalize_portrait_sets_long_side_and_keeps_aspect():
    img = np.zeros((1920, 1080, 3), np.uint8)  # portrait, aspect 9:16
    out = canonicalize_for_analysis(img)
    h, w = out.shape[:2]
    assert max(h, w) == CANONICAL_LONG_SIDE
    assert abs((w / h) - (1080 / 1920)) < (1.0 / min(h, w))
    assert h > w  # still portrait


def test_canonicalize_already_canonical_is_noop_copy():
    img = np.random.default_rng(0).integers(
        0, 255, (720, CANONICAL_LONG_SIDE, 3), dtype=np.uint8)
    out = canonicalize_for_analysis(img)
    assert out.shape == img.shape
    assert np.array_equal(out, img)     # unchanged values
    assert out is not img               # but a distinct copy


def test_canonicalize_grayscale_preserves_2d_layout():
    img = np.zeros((600, 900), np.uint8)   # single-channel / grayscale
    out = canonicalize_for_analysis(img)
    assert out.ndim == 2
    assert max(out.shape) == CANONICAL_LONG_SIDE


def test_canonicalize_four_channel_preserves_channels():
    img = np.zeros((600, 900, 4), np.uint8)  # e.g. decoded-with-alpha layout
    out = canonicalize_for_analysis(img)
    assert out.shape[2] == 4
    assert max(out.shape[:2]) == CANONICAL_LONG_SIDE


def test_canonicalize_downscale_uses_area_upscale_uses_linear(monkeypatch):
    # Actually assert the interpolation KIND, not just the output size, by
    # intercepting cv2.resize inside the module under test.
    real_resize = cv2.resize
    seen = {}

    def spy(img, dsize, interpolation=None, **kw):
        seen["interpolation"] = interpolation
        return real_resize(img, dsize, interpolation=interpolation, **kw)

    monkeypatch.setattr(visual_complexity.cv2, "resize", spy)

    seen.clear()
    big = np.zeros((2000, 3000, 3), np.uint8)   # long side 3000 -> downscale
    out = canonicalize_for_analysis(big)
    assert max(out.shape[:2]) == CANONICAL_LONG_SIDE
    assert seen["interpolation"] == cv2.INTER_AREA

    seen.clear()
    small = np.zeros((300, 450, 3), np.uint8)    # long side 450 -> upscale
    out = canonicalize_for_analysis(small)
    assert max(out.shape[:2]) == CANONICAL_LONG_SIDE
    assert seen["interpolation"] == cv2.INTER_LINEAR


def test_canonicalize_rejects_too_small_and_degenerate():
    # Too small on the long side -> documented ImageTooSmallError (a ValueError
    # subclass, so callers catching ValueError still work).
    tiny = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ImageTooSmallError):
        canonicalize_for_analysis(tiny)
    assert 4 < MIN_CANONICAL_INPUT_LONG_SIDE
    assert issubclass(ImageTooSmallError, ValueError)

    # BOTH dimensions are validated: a degenerate strip whose LONG side clears
    # the minimum but whose SHORT side does not must still be rejected.
    strip = np.zeros((8, CANONICAL_LONG_SIDE, 3), np.uint8)  # 1280 long, 8 short
    assert max(strip.shape[:2]) >= MIN_CANONICAL_INPUT_LONG_SIDE
    assert min(strip.shape[:2]) < MIN_CANONICAL_INPUT_SHORT_SIDE
    with pytest.raises(ImageTooSmallError):
        canonicalize_for_analysis(strip)

    # Zero-sized dimension is also too small.
    with pytest.raises(ImageTooSmallError):
        canonicalize_for_analysis(np.zeros((0, 10, 3), np.uint8))

    # A non-array is a programming error, not a client input problem -> TypeError.
    with pytest.raises(TypeError):
        canonicalize_for_analysis("not an image")


# ---------------------------------------------------------------------------
# 2. Determinism: identical input -> identical output
# ---------------------------------------------------------------------------
def test_canonicalize_is_deterministic():
    img = np.random.default_rng(1).integers(
        0, 255, (777, 1234, 3), dtype=np.uint8)
    a = canonicalize_for_analysis(img)
    b = canonicalize_for_analysis(img)
    assert np.array_equal(a, b)


def test_feature_vector_is_deterministic():
    img = FIXTURES["hard_edged"](2)
    v1 = _features_of(img)
    v2 = _features_of(img)
    for k in FEATURE_KEYS:
        assert v1[k] == v2[k], f"{k} not deterministic: {v1[k]} vs {v2[k]}"


# ---------------------------------------------------------------------------
# 3. Scale-stability across 1x / 2x / 3x, per fixture.
#
#    Reports (for transparency) every raw feature's absolute + relative gap and
#    every normalized HCEye input, and ASSERTS on what actually matters:
#      (a) the HCEye cognitive-load index (endpoint effect) is stable, and
#      (b) the three pixel-scale-driver features are stable after normalization.
#    layout_symmetry / visual_hierarchy are reported but not held to (b): their
#    residual sub-pixel resampling drift is shown to be immaterial via (a).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fx_name", list(FIXTURES))
def test_scale_invariance_report_and_endpoint(fx_name):
    fx = FIXTURES[fx_name]
    scales = (1, 2, 3)
    vecs = {s: _features_of(fx(s)) for s in scales}
    norm = {s: normalized_hceye_inputs(vecs[s]) for s in scales}
    idx = {s: hceye_rule_index_no_saliency_no_ocr(vecs[s]) for s in scales}

    report = [f"\n[{fx_name}] scale report (D=pixel-scale driver, "
              f"M=HCEye-mapped):"]
    report.append("  raw 8 features:")
    for k in FEATURE_KEYS:
        vals = [vecs[s][k] for s in scales]
        a, r = _gap(vals)
        tag = ("D" if k in PIXEL_SCALE_DRIVERS else
               ("M" if k in HCEYE_MAPPED_STAGE1 else " "))
        report.append(
            f"    [{tag}] {k:28s} 1x={vals[0]:10.5f} 2x={vals[1]:10.5f} "
            f"3x={vals[2]:10.5f} abs={a:9.5f} rel={r:7.2%}")
    report.append("  normalized HCEye inputs (percentile in [0,1]):")
    driver_fail = []
    for k in HCEYE_MAPPED_STAGE1:
        vals = [norm[s][k] for s in scales]
        a, r = _gap(vals)
        report.append(
            f"        {k:28s} 1x={vals[0]:.4f} 2x={vals[1]:.4f} "
            f"3x={vals[2]:.4f} abs={a:.4f} rel={r:7.2%}")
        if k in PIXEL_SCALE_DRIVERS and a > DRIVER_NORM_GUARD:
            driver_fail.append((k, a))
    idx_vals = [idx[s] for s in scales]
    idx_abs, idx_rel = _gap(idx_vals)
    report.append(
        f"  HCEye rule index (no saliency, no OCR) 1x={idx_vals[0]:.5f} "
        f"2x={idx_vals[1]:.5f} 3x={idx_vals[2]:.5f} "
        f"abs={idx_abs:.5f} rel={idx_rel:.2%}")
    print("\n".join(report))

    # (a) The meaningful criterion: the HCEye rule index barely moves.
    assert idx_abs <= ENDPOINT_ABS_GUARD, (
        f"{fx_name}: HCEye rule index moved {idx_abs:.5f} "
        f"across scales (> {ENDPOINT_ABS_GUARD})\n" + "\n".join(report))
    # (b) The proven defect drivers stay stable after normalization.
    assert not driver_fail, (
        f"{fx_name}: pixel-scale drivers exceeded normalized guard "
        f"{DRIVER_NORM_GUARD}: {driver_fail}\n" + "\n".join(report))


# ---------------------------------------------------------------------------
# 4. API contract: an expected too-small input returns a documented 400, not a
#    generic 500. Uses the Flask test client (same lightweight path as the smoke
#    suite; no heavy ML stack required).
# ---------------------------------------------------------------------------
from app import app, IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def _png_bytes(h, w):
    im = np.full((h, w, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("h,w,label", [
    (8, 8, "too small on both sides"),
    (8, CANONICAL_LONG_SIDE, "degenerate strip: long side ok, short side tiny"),
])
def test_analyze_too_small_returns_400(client, h, w, label):
    import io
    data = {"image": (io.BytesIO(_png_bytes(h, w)), "tiny.png")}
    resp = client.post("/api/analyze", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 400, f"{label}: expected 400, got {resp.status_code}"
    body = resp.get_json()
    assert body == {
        "analysis_complete": False,
        "error": {
            "code": "image_resource_limit",
            "message": IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE,
        },
    }


def test_analyze_normal_image_is_not_400(client):
    import io
    data = {"image": (io.BytesIO(_png_bytes(300, 400)), "ok.png")}
    resp = client.post("/api/analyze", data=data,
                       content_type="multipart/form-data")
    # A normal image must NOT hit the too-small 400 path (200 expected; the
    # saliency stage degrades gracefully without the ML stack).
    assert resp.status_code == 200, resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# 5. Two-path endpoint WIRING test.
#
# The scale tests above prove the analysis math is scale-stable, but they call
# the helpers directly. This test drives the REAL Flask /api/cognitive-load
# route once and installs spies on the four boundary functions to prove the
# endpoint actually wires the two paths as claimed:
#
#   (A) the score-driving LAYOUT detection runs on the CANONICAL image
#       (long side == CANONICAL_LONG_SIDE),
#   (B) the score-driving OCR receives the SAME canonical image and the SAME
#       canonical element objects,
#   (C) the NATIVE element detection receives the ORIGINAL native dimensions,
#   (D) the Jokinen search model receives the NATIVE element set and the NATIVE
#       image shape,
#   (E) the returned detected_elements are the NATIVE-coordinate elements, and
#   (F) the mapped readability boxes stay inside the native image bounds.
#
# Saliency is replaced with a deterministic, valid synthetic success result
# (see _synthetic_saliency_success below) so no ML weights are needed; OCR and
# Jokinen are stubbed to keep the test fast and deterministic while still
# capturing exactly what the endpoint passed them. A raised/unavailable
# saliency stage is no longer an option here: the fail-closed contract (see
# tests/test_saliency_production_contract.py) turns that into a 503, which
# would defeat this test's 200-success-path intent.
# ---------------------------------------------------------------------------
import app as app_module  # noqa: E402
import canonical_layout as canonical_layout_mod  # noqa: E402
import cognitive.element_detector as element_detector_mod  # noqa: E402
import cognitive.text_reader as text_reader_mod  # noqa: E402
import cognitive.jokinen_model as jokinen_mod  # noqa: E402


def _synthetic_ui(h, w):
    """Deterministic light-background UI with a few dark/colored boxes so the
    real element detector returns a non-empty element set."""
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(img, (50, 50), (300, 150), (30, 30, 30), -1)
    cv2.rectangle(img, (400, 200), (700, 320), (60, 60, 200), -1)
    cv2.rectangle(img, (100, 400), (500, 520), (200, 60, 60), -1)
    cv2.rectangle(img, (900, 600), (1200, 780), (40, 160, 40), -1)
    return img


def _synthetic_saliency_success(image_hash, image_path):
    """Deterministic, valid ``(heatmap, classif, cache_hit)`` success triple
    matching the current ``app._predict_saliency_cached`` return contract
    (see stage1/app.py:_predict_saliency_cached).

    Used as a stand-in for the real (weights-gated, unavailable-in-this-
    environment) UMSI++ model so these success-path tests keep reaching the
    200 response instead of the fail-closed 503 that a raised/broken
    saliency stage now produces (see
    tests/test_saliency_production_contract.py). The same fixed heatmap is
    returned regardless of the input image, which is intentional: it removes
    saliency as a source of variance so it does not contaminate what these
    tests actually measure (endpoint wiring / scale invariance of the other
    seven visual features). Mirrors the pattern already established in
    tests/test_smoke.py:_fake_saliency_model.
    """
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    heat = np.zeros((size, size), dtype=np.float64)
    for cy, cx, sigma in ((40.0, 40.0, 18.0), (90.0, 70.0, 12.0)):
        heat += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    heat = heat / heat.max()
    heatmap = heat.astype(np.float32)
    classif = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return heatmap, classif, False


def test_cognitive_load_endpoint_wires_two_paths(client, monkeypatch):
    import io

    NATIVE_H, NATIVE_W = 1000, 1600            # long side 1600 != canonical 1280
    assert max(NATIVE_H, NATIVE_W) != CANONICAL_LONG_SIDE
    img = _synthetic_ui(NATIVE_H, NATIVE_W)
    ok, buf = cv2.imencode(".png", img)
    assert ok

    captured = {}
    real_detect = element_detector_mod.detect_elements

    # (A) score-driving canonical layout detection (called inside
    #     measure_canonical_layout via canonical_layout.detect_elements).
    def spy_canonical_detect(image, *a, **k):
        captured["canonical_img_shape"] = tuple(image.shape[:2])
        els = real_detect(image, *a, **k)
        captured["canonical_elements"] = els
        return els

    # (C) native element detection (app.py re-imports this name at call time
    #     from cognitive.element_detector).
    def spy_native_detect(image, *a, **k):
        captured["native_img_shape"] = tuple(image.shape[:2])
        els = real_detect(image, *a, **k)
        captured["native_elements"] = els
        return els

    # (B) score-driving OCR: capture what it received, return a minimal valid
    #     readability report (canonical-coordinate text box) without running
    #     the heavy OCR stack.
    def stub_ocr(image_bgr, elements, *a, **k):
        captured["ocr_img_shape"] = tuple(image_bgr.shape[:2])
        captured["ocr_elements"] = elements
        return {
            "n_elements": len(elements),
            "n_text_elements": 1 if elements else 0,
            "text_elements": (
                [{"id": 0, "bbox": [10, 10, 40, 15],
                  "center": [30, 17], "text": "x", "reading_time_s": 0.1}]
                if elements else []
            ),
        }

    # (D) Jokinen search model: capture the element set and image shape it was
    #     given, return a minimal result (empty per_element skips bottlenecks).
    def spy_jokinen(self, elements=None, saliency_map=None, image_shape=None,
                    *a, **k):
        captured["jokinen_elements"] = elements
        captured["jokinen_image_shape"] = tuple(image_shape)
        return {"mean_search_time_s": 1.0, "per_element": []}

    # Saliency stays on its mandatory success path (no ML weights needed)
    # via a deterministic synthetic heatmap; see _synthetic_saliency_success.
    monkeypatch.setattr(canonical_layout_mod, "detect_elements",
                        spy_canonical_detect)
    monkeypatch.setattr(element_detector_mod, "detect_elements",
                        spy_native_detect)
    monkeypatch.setattr(text_reader_mod, "compute_readability", stub_ocr)
    monkeypatch.setattr(jokinen_mod.JokinenSearchModel, "predict_search_times",
                        spy_jokinen)
    monkeypatch.setattr(app_module, "_predict_saliency_cached",
                        _synthetic_saliency_success)

    data = {"image": (io.BytesIO(buf.tobytes()), "wiring.png")}
    resp = client.post("/api/cognitive-load", data=data,
                       content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()

    # All four boundaries must have been exercised.
    for key in ("canonical_img_shape", "canonical_elements", "ocr_img_shape",
                "ocr_elements", "native_img_shape", "native_elements",
                "jokinen_elements", "jokinen_image_shape"):
        assert key in captured, f"boundary {key} was never called"

    # (A) canonical layout detection saw the canonical long side.
    assert max(captured["canonical_img_shape"]) == CANONICAL_LONG_SIDE

    # (B) OCR saw the canonical image AND the exact canonical element objects.
    assert max(captured["ocr_img_shape"]) == CANONICAL_LONG_SIDE
    assert captured["ocr_elements"] is captured["canonical_elements"]

    # (C) native detection saw the ORIGINAL native dimensions (not canonical).
    assert captured["native_img_shape"] == (NATIVE_H, NATIVE_W)
    assert max(captured["native_img_shape"]) != CANONICAL_LONG_SIDE

    # (D) Jokinen received the native element set and the native image shape.
    assert captured["jokinen_elements"] is captured["native_elements"]
    assert captured["jokinen_image_shape"] == (NATIVE_H, NATIVE_W)

    # (E) returned detected_elements are the native-coordinate elements.
    returned = body["detected_elements"]
    expected = [list(e["bbox"]) for e in captured["native_elements"]]
    assert [e["bbox"] for e in returned] == expected
    assert len(returned) >= 1
    # Every returned native box lies within native bounds.
    for e in returned:
        x, y, bw, bh = e["bbox"]
        assert 0 <= x and 0 <= y
        assert x + bw <= NATIVE_W and y + bh <= NATIVE_H

    # (F) mapped readability boxes stay inside the native image bounds.
    rr = body.get("readability_report")
    assert rr is not None and rr.get("coordinate_space") == "native"
    assert rr["text_elements"], "expected at least one mapped text element"
    for te in rr["text_elements"]:
        x, y, bw, bh = te["bbox"]
        assert 0 <= x and 0 <= y
        assert x + bw <= NATIVE_W and y + bh <= NATIVE_H

    # Provenance confirms the score-driving layout came from the canonical path.
    prov = body["hceye_inputs"]["analysis_provenance"]
    assert prov is not None
    assert prov["analysis_path"] == f"canonical-analysis:long{CANONICAL_LONG_SIDE}"
    assert prov["analysis_long_side"] == CANONICAL_LONG_SIDE
    assert list(prov["native_shape"]) == [NATIVE_H, NATIVE_W]


# ---------------------------------------------------------------------------
# 5. Two-path layout-scale architecture regression tests.
#
# These prove the LAYOUT-SCALE fix: the score-driving layout measurements
# (whitespace_ratio, OCR text_density) run on the CANONICAL analysis image
# (long side 1280) and in CANONICAL coordinates, while the Jokinen / target-
# selection / overlay path keeps NATIVE images and native coordinates.
# See stage1/canonical_layout.py.
# ---------------------------------------------------------------------------
import canonical_layout  # noqa: E402
from canonical_layout import (  # noqa: E402
    measure_canonical_layout,
    scale_box_to_native,
    CanonicalLayoutMeasurement,
    _whitespace_from_boxes,
    ANALYSIS_PATH,
)


@pytest.mark.parametrize("fx_name", list(FIXTURES))
def test_layout_detector_always_receives_canonical_long_side(fx_name, monkeypatch):
    """(1) For 1x/2x/3x inputs the score-driving detector always sees long=1280."""
    seen = []
    real_detect = canonical_layout.detect_elements

    def spy(img, *a, **k):
        seen.append(int(max(img.shape[:2])))
        return real_detect(img, *a, **k)

    monkeypatch.setattr(canonical_layout, "detect_elements", spy)
    fx = FIXTURES[fx_name]
    for s in (1, 2, 3):
        measure_canonical_layout(fx(s), run_ocr=False)
    assert seen, "layout detector was never called"
    assert set(seen) == {CANONICAL_LONG_SIDE}, (
        f"{fx_name}: detector saw long sides {seen}, expected all "
        f"{CANONICAL_LONG_SIDE}")


@pytest.mark.parametrize("fx_name", list(FIXTURES))
def test_whitespace_from_canonical_boxes_on_canonical_canvas(fx_name):
    """(2) Whitespace is computed from canonical boxes on the canonical canvas."""
    m = measure_canonical_layout(FIXTURES[fx_name](2), run_ocr=False)
    ah, aw = m.analysis_shape
    assert max(ah, aw) == CANONICAL_LONG_SIDE
    # It equals recomputation from the returned canonical boxes on the canonical
    # canvas...
    expected = _whitespace_from_boxes(m.analysis_elements, ah, aw)
    assert abs(m.whitespace_ratio - expected) < 1e-9
    # ...and differs from the value the same boxes would give on the native
    # canvas, proving the measurement lives in canonical space (not native).
    nh, nw = m.native_shape
    if (nh, nw) != (ah, aw):
        native_val = _whitespace_from_boxes(m.analysis_elements, nh, nw)
        assert native_val != pytest.approx(m.whitespace_ratio, abs=1e-6)


def test_score_driving_ocr_uses_canonical_image_and_elements(monkeypatch):
    """(3) text_density OCR runs on the canonical image AND canonical elements."""
    captured = {}

    def fake_readability(img, elements):
        captured["long_side"] = int(max(img.shape[:2]))
        captured["elements_id"] = id(elements)
        captured["n"] = len(elements)
        n_text = sum(1 for i in range(len(elements)) if i % 2 == 0)
        return {"n_elements": len(elements), "n_text_elements": n_text,
                "text_elements": []}

    import cognitive.text_reader as tr
    monkeypatch.setattr(tr, "compute_readability", fake_readability)

    m = measure_canonical_layout(FIXTURES["hard_edged"](2), run_ocr=True)
    assert captured["long_side"] == CANONICAL_LONG_SIDE
    # The exact canonical element list is what OCR received and what is stored.
    assert captured["elements_id"] == id(m.analysis_elements)
    assert m.text_density_source == "ocr"
    n = captured["n"]
    expected_td = (sum(1 for i in range(n) if i % 2 == 0) / n) if n else None
    assert m.text_density == pytest.approx(expected_td)


def test_native_path_uses_native_coordinates_not_canonical():
    """(4) The native path yields native-coordinate elements distinct from the
    canonical analysis elements (target selection / Jokinen / overlays)."""
    from cognitive.element_detector import detect_elements
    native = FIXTURES["hard_edged"](3)          # native 1800x2700
    m = measure_canonical_layout(native, run_ocr=False)
    native_elems = detect_elements(native)
    # Canonical elements stay within the canonical canvas.
    ah, aw = m.analysis_shape
    for e in m.analysis_elements:
        x, y, bw, bh = e["bbox"]
        assert x + bw <= aw + 1 and y + bh <= ah + 1
    # Native elements extend beyond the canonical long side (native coordinates).
    max_native_x = max((e["bbox"][0] + e["bbox"][2]) for e in native_elems)
    assert max_native_x > CANONICAL_LONG_SIDE


def test_readability_native_mapping_stays_in_bounds():
    """(5) Readability boxes mapped back to native coordinates stay in bounds."""
    native_h, native_w = 1200, 1800
    analysis_h, analysis_w = 853, 1280
    sx, sy = native_w / analysis_w, native_h / analysis_h
    report = {
        "n_elements": 2, "n_text_elements": 2,
        "text_elements": [
            {"bbox": [analysis_w - 10, analysis_h - 10, 40, 40], "center": [0, 0]},
            {"bbox": [0, 0, 5, 5], "center": [0, 0]},
        ],
    }
    m = CanonicalLayoutMeasurement(
        native_shape=(native_h, native_w),
        analysis_shape=(analysis_h, analysis_w),
        scale_x=sx, scale_y=sy, analysis_elements=[], whitespace_ratio=0.5,
        text_density=1.0, text_density_source="ocr", readability_report=report)
    nat = m.readability_report_native()
    assert nat["coordinate_space"] == "native"
    for te in nat["text_elements"]:
        x, y, w, h = te["bbox"]
        assert 0 <= x < native_w and 0 <= y < native_h
        assert x + w <= native_w and y + h <= native_h
    # The reusable helper clips an out-of-range canonical box in-bounds.
    bx, by, bw, bh = scale_box_to_native(
        (analysis_w + 500, analysis_h + 500, 100, 100), sx, sy, native_w, native_h)
    assert 0 <= bx < native_w and 0 <= by < native_h
    assert bx + bw <= native_w and by + bh <= native_h


def test_analysis_path_provenance_identifier():
    """The measurement is unmistakably tagged as the analysis path."""
    m = measure_canonical_layout(FIXTURES["gradient"](1), run_ocr=False)
    assert m.analysis_path == ANALYSIS_PATH
    d = m.as_dict()
    assert d["analysis_long_side"] == CANONICAL_LONG_SIDE
    assert d["analysis_path"] == ANALYSIS_PATH


def test_fixtures_are_independently_rendered_not_raster_enlarged():
    """(8) Each scale is drawn natively, not produced by enlarging the 1x raster."""
    for name, fx in FIXTURES.items():
        one = fx(1)
        two = fx(2)
        assert two.shape[0] == one.shape[0] * 2 and two.shape[1] == one.shape[1] * 2
        enlarged = cv2.resize(one, (one.shape[1] * 2, one.shape[0] * 2),
                              interpolation=cv2.INTER_NEAREST)
        # A true re-render is not byte-identical to a raster enlargement.
        assert not np.array_equal(two, enlarged), name


# ---------------------------------------------------------------------------
# 6. End-to-end endpoint scale-stability.
#
# The prospective guard: the HCEye cognitive-load index lives in [0, 1]; a
# maximum gap <= 0.01 across 1x/2x/3x equals <= 1.0 displayed point on the
# 0-100 scale. This is exercised through the ACTUAL Flask endpoint, not a
# formula helper.
# ---------------------------------------------------------------------------
SYNTHETIC_ENDPOINT_GUARD = 0.01   # == 1.0 displayed point on the 0-100 scale


def test_experimental_layout_index_scale_invariant_through_endpoint(client, monkeypatch):
    """(6,7,9) POST independently re-rendered 1x/2x/3x fixtures to
    /api/cognitive-load and require the bounded experimental layout-index gap
    <= 0.01.

    Explicitly mocked (and ONLY these):
      * UMSI++ saliency: app._predict_saliency_cached is replaced with
        _synthetic_saliency_success, a deterministic, valid, fixed heatmap
        (identical regardless of scale). Saliency is a nondeterministic
        external ML component and is not what this scale test measures; a
        raised/unavailable saliency stage would now hit the fail-closed 503
        contract (see tests/test_saliency_production_contract.py) instead of
        reaching this test's 200 success path, so a fixed valid result is
        used instead to remove saliency as a source of variance.
      * OCR: cognitive.text_reader.compute_readability returns a valid no-text
        report deterministically.
    Nothing else is stubbed: the eight visual features, the canonical element
    detector, whitespace_ratio and the HCEye rule all run for real.
    """
    import io
    import app as app_module
    import cognitive.text_reader as tr

    monkeypatch.setattr(app_module, "_predict_saliency_cached",
                        _synthetic_saliency_success)
    monkeypatch.setattr(
        tr,
        "compute_readability",
        lambda _image, elements: {
            "n_elements": len(elements),
            "n_text_elements": 0,
            "text_elements": [],
        },
    )

    per_fx = {}
    for name, fx in FIXTURES.items():
        idx = {}
        for s in (1, 2, 3):
            ok, buf = cv2.imencode(".png", fx(s))
            assert ok
            data = {"image": (io.BytesIO(buf.tobytes()), f"{name}_{s}x.png")}
            resp = client.post("/api/cognitive-load", data=data,
                               content_type="multipart/form-data")
            assert resp.status_code == 200, resp.get_data(as_text=True)
            body = resp.get_json()
            idx[s] = float(
                body["hceye_proxy_features"][
                    "experimental_layout_complexity_index"
                ]
            )
        gap = max(idx.values()) - min(idx.values())
        per_fx[name] = (idx, gap)

    print("\n[endpoint /api/cognitive-load experimental layout index 1x/2x/3x]")
    for name, (idx, gap) in per_fx.items():
        print(f"  {name}: 1x={idx[1]:.5f} 2x={idx[2]:.5f} 3x={idx[3]:.5f} "
              f"gap={gap:.5f} ({gap * 100:.3f} displayed pt)")

    for name, (idx, gap) in per_fx.items():
        assert gap <= SYNTHETIC_ENDPOINT_GUARD, (
            f"{name}: experimental layout-index gap {gap:.5f} exceeds "
            f"{SYNTHETIC_ENDPOINT_GUARD} (= 1.0 displayed point). idx={idx}")


# ---------------------------------------------------------------------------
# 7. Provenance is derived from the ACTUAL analysis resolution (Finding 4 / E).
#
# There must be exactly ONE provenance-building path keyed on the real long
# side: the production default reports long1280, a non-default 1024 analysis
# reports long1024, and the production canonical long side stays 1280.
# ---------------------------------------------------------------------------
from visual_complexity import (  # noqa: E402
    canonical_analysis_version,
    CANONICAL_ANALYSIS_VERSION,
)
from canonical_layout import analysis_path_for  # noqa: E402


def test_production_canonical_long_side_is_1280():
    assert CANONICAL_LONG_SIDE == 1280


def test_canonical_analysis_version_reflects_actual_long_side():
    v1280 = canonical_analysis_version(1280)
    v1024 = canonical_analysis_version(1024)
    assert ":long1280:" in v1280
    assert ":long1024:" in v1024
    assert v1280 != v1024
    # The default (no argument) and the module constant are the production value.
    assert canonical_analysis_version() == v1280
    assert CANONICAL_ANALYSIS_VERSION == v1280


def test_analysis_path_for_reflects_actual_long_side():
    assert analysis_path_for(1280) == "canonical-analysis:long1280"
    assert analysis_path_for(1024) == "canonical-analysis:long1024"
    assert analysis_path_for() == "canonical-analysis:long1280"
    assert ANALYSIS_PATH == analysis_path_for(1280)


def test_measure_canonical_layout_reports_nondefault_resolution():
    """A non-default 1024 analysis reports long1024; the default stays 1280."""
    m = measure_canonical_layout(FIXTURES["gradient"](1), run_ocr=False,
                                 canonical_long_side=1024)
    assert m.analysis_path == "canonical-analysis:long1024"
    assert m.analysis_long_side == 1024
    d = m.as_dict()
    assert d["analysis_path"] == "canonical-analysis:long1024"
    assert d["analysis_long_side"] == 1024

    m2 = measure_canonical_layout(FIXTURES["gradient"](1), run_ocr=False)
    assert m2.analysis_path == "canonical-analysis:long1280"
    assert m2.analysis_long_side == CANONICAL_LONG_SIDE


# ---------------------------------------------------------------------------
# 8. Authorized-category enforcement (Finding 2 / A).
#
# The authorized set is EXACTLY {desktop, mobile, web}. `poster` must never
# enter any boundary: corpus loading, defensive filtering, decision-image
# selection, or held-out sampling. Boundary functions filter themselves, so a
# caller passing an unfiltered dictionary containing `poster` cannot leak one.
# ---------------------------------------------------------------------------
import canonical_scale_eval as cse  # noqa: E402


def _poster_contaminated_dict():
    d = {}
    for cat in ("desktop", "mobile", "web"):
        for i in range(6):
            d[f"{cat}{i:02d}.png"] = cat
    for i in range(6):
        d[f"poster{i:02d}.png"] = "poster"
    return d


def test_authorized_category_definition_is_exactly_three():
    assert cse.AUTHORIZED_CATEGORIES == ("desktop", "mobile", "web")
    assert "poster" in cse.EXCLUDED_CATEGORIES
    assert "poster" not in cse._AUTHORIZED_SET


def test_authorized_only_drops_poster():
    filtered = cse._authorized_only(_poster_contaminated_dict())
    assert filtered, "expected authorized entries to remain"
    assert all(c in cse._AUTHORIZED_SET for c in filtered.values())
    assert not any(n.startswith("poster") for n in filtered)


def test_load_types_drops_poster(tmp_path):
    csv_path = tmp_path / "image_types.csv"
    csv_path.write_text(
        "Image Name;Category;Block;Train/Test\n"
        "a.png;desktop;0;Train\n"
        "b.png;mobile;0;Train\n"
        "c.png;web;0;Train\n"
        "p.png;poster;0;Train\n")
    cat_by_id = cse._load_types(str(csv_path))
    assert set(cat_by_id) == {"a.png", "b.png", "c.png"}
    assert "poster" not in cat_by_id.values()


def test_selection_and_heldout_cannot_admit_poster(monkeypatch):
    """Even fed an unfiltered poster-containing dict, neither boundary admits a
    poster. `_resolve` is stubbed so the check runs without the image corpus."""
    monkeypatch.setattr(cse, "_resolve", lambda images_dir, name: f"/fake/{name}")
    contaminated = _poster_contaminated_dict()

    sel = cse.selection_images("/fake", contaminated)
    assert sel, "expected a non-empty selection"
    assert all(cat in cse._AUTHORIZED_SET for _id, cat, _p in sel)
    assert not any(_id.startswith("poster") for _id, _cat, _p in sel)

    heldout = cse.heldout_sample("/fake", contaminated, exclude_ids=set(),
                                 seed=20240607, per_cat=2)
    assert heldout, "expected a non-empty held-out sample"
    assert all(cat in cse._AUTHORIZED_SET for _id, cat, _p in heldout)
    assert not any(_id.startswith("poster") for _id, _cat, _p in heldout)


@pytest.mark.skipif(
    not os.environ.get("UEYES_TYPES_CSV") or not os.environ.get("UEYES_IMAGES_DIR"),
    reason="full UEyes corpus not available (set UEYES_TYPES_CSV and "
           "UEYES_IMAGES_DIR to run the exact deterministic-ID check)")
def test_exact_selection_and_heldout_ids_are_poster_free():
    """The documented deterministic decision and held-out image IDs, when the
    corpus is available. Both sets are exactly six images with zero posters."""
    images_dir = os.environ["UEYES_IMAGES_DIR"]
    types_csv = os.environ["UEYES_TYPES_CSV"]
    cat_by_id = cse._load_types(types_csv)
    sel = cse.selection_images(images_dir, cat_by_id)
    sel_ids = {i for i, _c, _p in sel}
    assert sel_ids == {"003eb9", "7f066d", "006450", "78666b", "009085", "81627a"}
    exclude = {i for i, _c, _p in sel}
    heldout = cse.heldout_sample(images_dir, cat_by_id, exclude_ids=exclude,
                                 seed=20240607, per_cat=2)
    ho_ids = {i for i, _c, _p in heldout}
    assert ho_ids == {"1f3437", "9dd93b", "62f428", "e129e6", "1408dc", "d9baa1"}
    assert not any(c == "poster" for _i, c, _p in sel + heldout)


# ---------------------------------------------------------------------------
# 9. Canonical visual-norms generator safety contract (Section F).
#
# The dedicated generator produces ONLY the eight visual features, refuses to
# overwrite the protected reference files, rejects posters at discovery, and
# refuses to resume a native / mixed-domain / differently-versioned CSV.
# ---------------------------------------------------------------------------
import csv as _csv  # noqa: E402
import canonical_visual_norms as gen  # noqa: E402


def test_generator_schema_is_eight_visual_features_no_saliency():
    assert len(gen.VISUAL_FEATURE_KEYS) == 8
    assert tuple(gen.VISUAL_FEATURE_KEYS) == tuple(FEATURE_KEYS)
    assert not any("saliency" in k for k in gen.VISUAL_FEATURE_KEYS)
    assert gen.AUTHORIZED_CATEGORIES == ("desktop", "mobile", "web")


def test_generator_provenance_tracks_actual_long_side():
    p1280 = gen.expected_provenance(1280)
    p1024 = gen.expected_provenance(1024)
    assert p1280["canonical_long_side"] == "1280"
    assert p1024["canonical_long_side"] == "1024"
    assert ":long1280:" in p1280["canonical_analysis_version"]
    assert ":long1024:" in p1024["canonical_analysis_version"]
    assert p1280["analysis_domain"] == "canonical"


def test_generator_refuses_protected_output_paths():
    for prot in ("stage1/data/results/feature_norms.json",
                 "hceye/sensitivity_lookup.json",
                 "feature_norms.json", "sensitivity_lookup.json"):
        with pytest.raises(gen.GeneratorError):
            gen._refuse_protected(prot, "test")


def _gen_types_csv(tmp_path):
    imgd = tmp_path / "images"
    imgd.mkdir()
    for n in ("a.png", "b.png", "c.png", "p.png"):
        (imgd / n).write_bytes(b"x")
    csv_path = tmp_path / "image_types.csv"
    csv_path.write_text(
        "Image Name;Category;Block;Train/Test\n"
        "a.png;desktop;0;Train\n"
        "b.png;mobile;0;Train\n"
        "c.png;web;0;Train\n"
        "p.png;poster;0;Train\n")
    return str(csv_path), str(imgd)


def test_generator_load_authorized_drops_poster(tmp_path):
    csv_path, imgd = _gen_types_csv(tmp_path)
    imgs = gen.load_authorized_images(csv_path, imgd)
    assert {n for n, _c in imgs} == {"a.png", "b.png", "c.png"}
    assert not any(c == "poster" for _n, c in imgs)


def test_generator_rejects_duplicate_rows(tmp_path):
    csv_path, imgd = _gen_types_csv(tmp_path)
    with open(csv_path, "a") as f:
        f.write("a.png;desktop;0;Train\n")
    with pytest.raises(gen.GeneratorError):
        gen.load_authorized_images(csv_path, imgd)


def _write_gen_rows(path, long_side, domain=None, version=None):
    cols = ["filename", "category", "analysis_domain", "canonical_long_side",
            "canonical_analysis_version"] + list(gen.VISUAL_FEATURE_KEYS)
    prov = gen.expected_provenance(long_side)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        row = {
            "filename": "a.png", "category": "desktop",
            "analysis_domain": domain or prov["analysis_domain"],
            "canonical_long_side": str(long_side),
            "canonical_analysis_version":
                version or prov["canonical_analysis_version"],
        }
        for k in gen.VISUAL_FEATURE_KEYS:
            row[k] = "0.5"
        w.writerow(row)


def test_generator_resume_accepts_matching_provenance(tmp_path):
    csv_path = str(tmp_path / "rows.csv")
    _write_gen_rows(csv_path, 1280)
    rows, done = gen.load_existing_csv_for_resume(csv_path, 1280)
    assert "a.png" in done
    assert len(rows) == 1


def test_generator_resume_rejects_mismatched_resolution(tmp_path):
    csv_path = str(tmp_path / "rows.csv")
    _write_gen_rows(csv_path, 1280)
    with pytest.raises(gen.GeneratorError):
        gen.load_existing_csv_for_resume(csv_path, 1024)


def test_generator_resume_rejects_native_domain(tmp_path):
    csv_path = str(tmp_path / "rows.csv")
    _write_gen_rows(csv_path, 1280, domain="native", version="native-vX")
    with pytest.raises(gen.GeneratorError):
        gen.load_existing_csv_for_resume(csv_path, 1280)


def test_generator_resume_rejects_legacy_csv_without_provenance(tmp_path):
    csv_path = str(tmp_path / "legacy.csv")
    cols = ["filename", "category"] + list(gen.VISUAL_FEATURE_KEYS)
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        row = {"filename": "a.png", "category": "desktop"}
        for k in gen.VISUAL_FEATURE_KEYS:
            row[k] = "0.5"
        w.writerow(row)
    with pytest.raises(gen.GeneratorError):
        gen.load_existing_csv_for_resume(csv_path, 1280)
