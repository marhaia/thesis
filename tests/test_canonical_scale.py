"""Canonical-analysis-resolution regression tests.

These tests validate the ROOT-CAUSE fix for the deterministic scale-invariance
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
    hceye_load_index,
)


# ---------------------------------------------------------------------------
# Scale-invariance guards.
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
# rather than pretending the raw feature is scale-invariant.
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
# 3. Scale invariance across 1x / 2x / 3x, per fixture.
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
    idx = {s: hceye_load_index(vecs[s]) for s in scales}

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
        f"  endpoint cognitive_load_index 1x={idx_vals[0]:.5f} "
        f"2x={idx_vals[1]:.5f} 3x={idx_vals[2]:.5f} "
        f"abs={idx_abs:.5f} rel={idx_rel:.2%}")
    print("\n".join(report))

    # (a) The meaningful criterion: the endpoint index barely moves.
    assert idx_abs <= ENDPOINT_ABS_GUARD, (
        f"{fx_name}: endpoint cognitive_load_index moved {idx_abs:.5f} "
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
from app import app  # noqa: E402


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
    assert body is not None and "error" in body
    # The documented message is client-safe and explains the size requirement.
    assert "too small" in body["error"].lower()


def test_analyze_normal_image_is_not_400(client):
    import io
    data = {"image": (io.BytesIO(_png_bytes(300, 400)), "ok.png")}
    resp = client.post("/api/analyze", data=data,
                       content_type="multipart/form-data")
    # A normal image must NOT hit the too-small 400 path (200 expected; the
    # saliency stage degrades gracefully without the ML stack).
    assert resp.status_code == 200, resp.get_data(as_text=True)
