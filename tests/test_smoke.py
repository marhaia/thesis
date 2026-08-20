"""
Lightweight pytest smoke + regression suite.

Purpose
-------
Fast, dependency-light checks that (a) the Flask app imports and serves its
core endpoints, (b) input validation rejects malformed requests cleanly, and
(c) the specific bugs fixed after the July-2026 security/robustness audit stay
fixed (regression guards).

These tests do NOT require the UMSI++ weights or TensorFlow. Saliency is a
MANDATORY part of a complete /api/cognitive-load analysis (fail-closed
contract: a broken/unavailable saliency stage now returns a structured,
non-200 error rather than a partial 200 result — see
tests/test_saliency_production_contract.py). Since the real (weights-gated)
UMSI++ model is unavailable in this environment, an autouse fixture below
replaces only the model-inference step (``app._predict_saliency_cached``)
with a deterministic, plausible synthetic heatmap, so these tests still
exercise the real saliency-feature-extraction / normalization / mandatory-
gate production code. This is test scaffolding for the missing weights, not
a claim of real weighted-model parity (that remains separately verified/
NOT PROVEN elsewhere). Anything needing the heavy ML stack itself is
intentionally out of scope here.

Run: python -m pytest tests/test_smoke.py -q
"""
import io
import os
import sys

import numpy as np
import pytest

# Make the project root and stage1 importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for the smoke tests")

from app import (  # noqa: E402  (import after sys.path setup)
    app,
    _clamp_simulations,
    _resolve_target_index,
    MAX_SIMULATIONS,
    MIN_SIMULATIONS,
    MAX_SCHEDULE_LEN,
)
from stage2.screen_consistency import _normalized_occupancy_grid  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    app.config.update(TESTING=True)
    return app.test_client()


def _synthetic_heatmap() -> np.ndarray:
    """Deterministic, plausible (two Gaussian blobs) synthetic saliency map.

    Used ONLY as test scaffolding to stand in for the real (weights-gated,
    unavailable-in-this-environment) UMSI++ model output, so that tests can
    still exercise the real saliency-feature-extraction, normalization, and
    mandatory-saliency-gate production code paths. Not a claim that this
    matches real model output.
    """
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    heat = np.zeros((size, size), dtype=np.float64)
    for cy, cx, sigma in ((40.0, 40.0, 18.0), (90.0, 70.0, 12.0)):
        heat += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    heat = heat / heat.max()
    return heat.astype(np.float32)


@pytest.fixture(autouse=True)
def _fake_saliency_model(monkeypatch):
    """Autouse: patch ``app._predict_saliency_cached`` to return a synthetic,
    but valid, heatmap instead of running the real (unavailable) UMSI++
    model. Saliency is now mandatory for a complete analysis, so without
    this, every existing test that hits /api/cognitive-load in this
    weights-less environment would receive the new fail-closed error
    response instead of a full result.

    A test that specifically wants to exercise the fail-closed
    missing/broken-saliency path (see
    test_cognitive_load_scale_invariance_sanity_saliency_disabled below) can
    simply call ``monkeypatch.setattr`` again inside its own body to override
    this default, which takes effect for the remainder of that test only.
    """
    import app as app_module

    heatmap = _synthetic_heatmap()
    classif = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _fake_predict(image_hash, image_path):
        return heatmap, classif, False

    monkeypatch.setattr(app_module, "_predict_saliency_cached", _fake_predict)

    # CI intentionally excludes EasyOCR.  Supply a valid, deterministic
    # no-text OCR report so ordinary success-path smoke tests do not exercise
    # P4's unavailable-OCR failure branch by accident.
    import cognitive.text_reader as text_reader

    def _fake_readability(_image, elements):
        return {
            "n_elements": len(elements),
            "n_text_elements": 0,
            "text_elements": [],
        }

    monkeypatch.setattr(text_reader, "compute_readability", _fake_readability)


def _png_bytes(cx: int = 40) -> io.BytesIO:
    """A small synthetic screenshot with a couple of coloured boxes."""
    im = np.full((300, 400, 3), 240, np.uint8)
    cv2.rectangle(im, (cx, 40), (cx + 120, 90), (30, 30, 200), -1)
    cv2.rectangle(im, (200, 200), (320, 250), (30, 180, 30), -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return io.BytesIO(buf.tobytes())


def _multibox_png() -> bytes:
    """A synthetic screen with several separated coloured boxes so element
    detection yields multiple selectable targets."""
    im = np.full((600, 900, 3), 245, np.uint8)
    boxes = [
        ((60, 60), (180, 140), (200, 40, 40)),
        ((400, 80), (520, 160), (40, 160, 40)),
        ((700, 100), (820, 180), (40, 40, 200)),
        ((120, 360), (260, 460), (180, 120, 40)),
        ((520, 380), (660, 470), (120, 40, 160)),
    ]
    for p1, p2, col in boxes:
        cv2.rectangle(im, p1, p2, col, -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


class _FakeReq:
    """Minimal stand-in for a Flask request exposing .args.get."""

    def __init__(self, value):
        self._value = value

        class _Args:
            @staticmethod
            def get(key, default=None, type=None):
                if key == "n_simulations":
                    return value
                return default

        self.args = _Args()


# ---------------------------------------------------------------------------
# Config / limits
# ---------------------------------------------------------------------------

def test_max_content_length_configured():
    # A request-size cap must exist so uploads cannot exhaust memory.
    assert app.config.get("MAX_CONTENT_LENGTH")
    assert app.config["MAX_CONTENT_LENGTH"] >= 1 * 1024 * 1024


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, MIN_SIMULATIONS),      # zero would give an empty Monte-Carlo -> NaN
        (-5, MIN_SIMULATIONS),     # negative likewise
        (None, 100),               # non-numeric input falls back to default
        (99999, MAX_SIMULATIONS),  # unbounded is capped
        (100, 100),                # in-range passes through
    ],
)
def test_clamp_simulations(value, expected):
    assert _clamp_simulations(_FakeReq(value)) == expected


# ---------------------------------------------------------------------------
# Endpoint smoke
# ---------------------------------------------------------------------------

def test_analyze_returns_eight_features(client):
    r = client.post(
        "/api/analyze",
        data={"image": (_png_bytes(), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    for key in (
        "shannon_entropy",
        "edge_density",
        "feature_congestion",
        "subband_entropy",
        "layout_symmetry",
        "chromatic_coherence",
        "visual_hierarchy",
        "interactive_element_density",
    ):
        assert key in body
        assert not np.isnan(body[key]), f"{key} is NaN"


def test_analyze_missing_image_is_400(client):
    r = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_search_time_zero_simulations_no_nan(client):
    # Regression: n_simulations=0 must be clamped, not produce an empty result.
    r = client.post(
        "/api/search-time?n_simulations=0",
        data={"image": (_png_bytes(), "s.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)


def test_consistency_single_screen_rejected(client):
    r = client.post(
        "/api/screen-consistency",
        data={"images": (_png_bytes(), "one.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_learning_curve_bad_exposures_is_400(client):
    r = client.post(
        "/api/learning-curve?exposures=abc",
        data={"image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_learning_curve_too_many_exposures_is_400(client):
    many = ",".join(["1"] * (MAX_SCHEDULE_LEN + 1))
    r = client.post(
        f"/api/learning-curve?exposures={many}",
        data={"image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_learning_curve_nonpositive_exposures_is_400(client):
    r = client.post(
        "/api/learning-curve?exposures=0,-3",
        data={"image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Regression: occupancy union (screen-consistency metric)
# ---------------------------------------------------------------------------

def test_occupancy_full_cell_box_is_one():
    occ = _normalized_occupancy_grid([{"bbox": (0, 0, 10, 10)}], (100, 100), (10, 10))
    assert occ[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_occupancy_union_not_double_counted():
    # Two overlapping boxes must not sum past their union area.
    boxes = [{"bbox": (0, 0, 8, 8)}, {"bbox": (2, 2, 8, 8)}]
    occ = _normalized_occupancy_grid(boxes, (100, 100), (10, 10))
    assert occ[0, 0] < 1.0
    assert (occ <= 1.0).all()
    assert not np.isnan(occ).any()


def test_occupancy_union_is_idempotent():
    # A duplicated box must produce the same occupancy as a single box.
    single = _normalized_occupancy_grid([{"bbox": (0, 0, 7, 7)}], (100, 100), (10, 10))
    doubled = _normalized_occupancy_grid(
        [{"bbox": (0, 0, 7, 7)}, {"bbox": (0, 0, 7, 7)}], (100, 100), (10, 10)
    )
    assert np.allclose(single, doubled)


# ---------------------------------------------------------------------------
# Regression: entropy on degenerate inputs (F25)
# ---------------------------------------------------------------------------

def test_entropy_single_sample_no_crash():
    # A single-sample signal used to request np.histogram(x, bins=0) and raise.
    from visual_complexity import _entropy  # noqa: E402

    assert _entropy(np.array([5.0])) == 0.0
    assert _entropy(np.array([])) == 0.0
    assert _entropy(np.random.rand(50), nbins=1) == 0.0
    assert _entropy(np.random.rand(500)) > 0.0


# ---------------------------------------------------------------------------
# Regression: coverage-weighted consistency headline (F52)
# ---------------------------------------------------------------------------

def _el(cx, cy, color="gray"):
    return {"bbox": [cx - 20, cy - 20, 40, 40], "center": [cx, cy],
            "color_category": color}


def test_consistency_identical_screens_is_one():
    from stage2.screen_consistency import compute_screen_set_consistency

    screen = [_el(50, 50, "red")] + [_el(100, 200 + i * 100) for i in range(5)]
    r = compute_screen_set_consistency([screen, screen], [(1000, 1000)] * 2)
    assert r["match_coverage"] == pytest.approx(1.0, abs=1e-6)
    assert r["consistency_score"] == pytest.approx(1.0, abs=1e-6)


def test_consistency_single_match_does_not_force_high_score():
    # One persistent logo while every other control moves must NOT let the
    # near-zero displacement of that single match claim a perfect headline.
    from stage2.screen_consistency import compute_screen_set_consistency

    a = [_el(50, 50, "red")] + [_el(100, 200 + i * 100) for i in range(5)]
    b = [_el(50, 50, "red")] + [_el(900, 200 + i * 100) for i in range(5)]
    r = compute_screen_set_consistency([a, b], [(1000, 1000)] * 2)
    assert r["match_coverage"] < 0.5
    # With low coverage the headline must be pulled below a naive 1.0.
    assert r["consistency_score"] < 1.0


# ---------------------------------------------------------------------------
# Regression: empty-canvas anchoring of the HCEye headline (supervisor test)
# ---------------------------------------------------------------------------

def test_hceye_blank_screen_anchors_low():
    # A blank canvas (no edges/elements, full whitespace) must read as LOW
    # load, not the ~0.4 "medium" floor the fixed coefficients used to produce.
    from hceye.hceye_features import HCEyeFeatureExtractor

    ext = HCEyeFeatureExtractor()
    blank_vf = {
        "edge_density": 0.0,
        "feature_congestion": 0.0,
        "interactive_element_density": 0.0,
        "layout_symmetry": 1.0,
        "visual_hierarchy": 0.0,
    }
    h_blank = ext.extract_features(blank_vf, whitespace_ratio=1.0, text_density=0.0)
    # Same features but a fully-covered canvas (no whitespace) must NOT anchor
    # to zero, proving the drop is driven by the content-presence gate.
    h_full = ext.extract_features(blank_vf, whitespace_ratio=0.0, text_density=0.0)

    assert h_blank[5] < 0.1, f"blank canvas load {h_blank[5]} should be near zero"
    assert h_full[5] > h_blank[5]


# ---------------------------------------------------------------------------
# Regression: Jokinen top-down guidance steers the search (supervisor #5)
# ---------------------------------------------------------------------------

def test_jokinen_target_guidance_pops_out_unique_target():
    # A uniquely coloured target must be found faster than the typical
    # same-coloured distractor, i.e. the selected target actively steers the
    # search (W_TA guidance), not just defines the stop condition.
    from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

    def _e(i, x, y, w, h, color):
        return {"id": i, "bbox": [x, y, w, h], "center": [x + w / 2, y + h / 2],
                "color_category": color, "area": w * h}

    els = []
    k = 0
    for gx in range(6):
        for gy in range(4):
            els.append(_e(k, 100 + gx * 180, 100 + gy * 160, 40, 30, "gray"))
            k += 1
    els[5]["color_category"] = "red"  # the single unique-colour target

    m = JokinenSearchModel(JokinenParams(n_simulations=40, random_seed=1))
    res = m.predict_search_times(els, image_shape=(800, 1200))
    per = {r["id"]: r["search_time_s"] for r in res["per_element"]}
    gray_mean = np.mean([per[i] for i in range(len(els)) if i != 5])
    assert per[5] < gray_mean, (
        f"unique red target {per[5]}s should beat mean gray {gray_mean}s"
    )


def test_jokinen_scanpath_depends_on_target():
    # Changing the target must change the scanpath (task-driven search).
    from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

    def _e(i, x, y, w, h, color):
        return {"id": i, "bbox": [x, y, w, h], "center": [x + w / 2, y + h / 2],
                "color_category": color, "area": w * h}

    els = []
    k = 0
    for gx in range(6):
        for gy in range(4):
            els.append(_e(k, 100 + gx * 180, 100 + gy * 160, 40, 30, "gray"))
            k += 1
    els[5]["color_category"] = "red"

    m = JokinenSearchModel(JokinenParams(n_simulations=40, random_seed=1))
    seq_red = [f.get("element_id")
               for f in m.predict_scanpath_to_target(5, els, image_shape=(800, 1200))["fixations"]]
    seq_gray = [f.get("element_id")
                for f in m.predict_scanpath_to_target(0, els, image_shape=(800, 1200))["fixations"]]
    assert seq_red != seq_gray


# ---------------------------------------------------------------------------
# Regression: Layout vs Selected-Target construct separation (supervisor #6)
# ---------------------------------------------------------------------------

def _cognitive_load(client, img_bytes):
    return client.post(
        "/api/cognitive-load",
        data={"image": (io.BytesIO(img_bytes), "syn.png")},
        content_type="multipart/form-data",
    ).get_json()


def _scanpath(client, img_bytes, target_id, n=30):
    return client.post(
        f"/api/scanpath-to-target?target_id={target_id}&n_simulations={n}",
        data={"image": (io.BytesIO(img_bytes), "syn.png")},
        content_type="multipart/form-data",
    ).get_json()


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", ["target_x", "target_y", "target_w", "target_h"])
def test_scanpath_rejects_nonfinite_target_geometry(client, field, bad_value):
    query = {
        "target_x": "10",
        "target_y": "12",
        "target_w": "30",
        "target_h": "20",
        "use_saliency": "false",
    }
    query[field] = bad_value

    response = client.post(
        "/api/scanpath-to-target",
        query_string=query,
        data={"image": (_png_bytes(), "invalid-target.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["analysis_complete"] is False
    assert body["error"]["code"] == "invalid_target_geometry"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("target_x", "-1"),
        ("target_y", "-1"),
        ("target_w", "0"),
        ("target_h", "0"),
        ("target_w", "-1"),
        ("target_h", "-1"),
    ],
)
def test_scanpath_rejects_negative_coordinates_and_nonpositive_region_sizes(
    client, field, bad_value
):
    query = {
        "target_x": "10",
        "target_y": "12",
        "target_w": "30",
        "target_h": "20",
        "use_saliency": "false",
    }
    query[field] = bad_value

    response = client.post(
        "/api/scanpath-to-target",
        query_string=query,
        data={"image": (_png_bytes(), "invalid-target.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["analysis_complete"] is False
    assert body["error"]["code"] == "invalid_target_geometry"


def test_scanpath_rejects_incomplete_or_duplicate_target_geometry(client):
    cases = [
        [("target_x", "10")],
        [("target_x", "10"), ("target_y", "12"), ("target_w", "30")],
        [("target_x", "10"), ("target_x", "11"), ("target_y", "12")],
    ]

    for query in cases:
        response = client.post(
            "/api/scanpath-to-target",
            query_string=query,
            data={"image": (_png_bytes(), "invalid-target.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        body = response.get_json()
        assert body["analysis_complete"] is False
        assert body["error"]["code"] == "invalid_target_geometry"


def test_scanpath_accepts_repeated_identical_finite_target_geometry(client):
    response = client.post(
        "/api/scanpath-to-target",
        query_string=[
            ("target_x", "10"),
            ("target_x", "10.0"),
            ("target_y", "12"),
            ("target_y", "12.0"),
            ("target_w", "30"),
            ("target_w", "30.0"),
            ("target_h", "20"),
            ("target_h", "20.0"),
            ("use_saliency", "false"),
            ("n_simulations", "1"),
        ],
        data={"image": (_png_bytes(), "repeated-target.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["analysis_complete"] is True
    assert body["target_bbox"] == pytest.approx([10.0, 12.0, 30.0, 20.0])


def test_scanpath_glance_metrics_have_explicit_nonclaim_boundary(client):
    response = client.post(
        "/api/scanpath-to-target",
        query_string={
            "target_x": "10",
            "target_y": "12",
            "target_w": "30",
            "target_h": "20",
            "display_preset": "phone",
            "use_saliency": "false",
            "n_simulations": "1",
        },
        data={"image": (_png_bytes(), "bounded-glance.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    metrics = response.get_json()["glance_metrics"]
    assert metrics is not None
    assert "compliant" not in metrics
    assert metrics["score_bearing"] is False
    assert metrics["validated_measurement"] is False
    assert metrics["validated_behavioral_prediction"] is False
    assert metrics["regulatory_compliance_assessment"] is False
    assert "not measured eye tracking" in metrics["claim_boundary"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_target_resolver_never_selects_an_element_for_invalid_coordinates(value):
    elements = [{"id": 0, "bbox": [10, 12, 30, 20], "center": [25, 22]}]

    assert _resolve_target_index(elements, None, value, 12.0) is None
    assert _resolve_target_index(elements, None, 10.0, value) is None


def test_layout_block_present_and_named(client):
    # The layout construct must be exposed as a separate, clearly-named block.
    data = _cognitive_load(client, _multibox_png())
    layout = data.get("layout")
    assert layout is not None
    assert "experimental_complexity_index" in layout
    assert "task_modifier" not in layout
    assert "profile_modifier" not in layout


def test_target_selection_does_not_change_layout(client):
    # Selecting (any) target must never alter the stable layout value.
    img = _multibox_png()
    before = _cognitive_load(client, img)["layout"]["experimental_complexity_index"]
    els = _cognitive_load(client, img).get("detected_elements", [])
    assert len(els) >= 2
    _scanpath(client, img, els[0]["id"])
    _scanpath(client, img, els[1]["id"])
    after = _cognitive_load(client, img)["layout"]["experimental_complexity_index"]
    assert before == pytest.approx(after)


def test_selected_target_changes_with_target(client):
    # Different targets with different model predictions must yield different
    # selected_target results (the target result is genuinely target-driven).
    img = _multibox_png()
    els = _cognitive_load(client, img).get("detected_elements", [])
    assert len(els) >= 2
    st0 = _scanpath(client, img, els[0]["id"])["selected_target"]
    st1 = _scanpath(client, img, els[1]["id"])["selected_target"]
    assert st0["target_id"] != st1["target_id"]
    # At least one of the reported quantities must differ between targets.
    assert (st0["search_time_s"], st0["deviation_pct"]) != \
           (st1["search_time_s"], st1["deviation_pct"])


def test_selected_target_has_required_fields(client):
    img = _multibox_png()
    els = _cognitive_load(client, img).get("detected_elements", [])
    assert els
    st = _scanpath(client, img, els[0]["id"])["selected_target"]
    for key in ("target_id", "search_time_s", "search_time_std_s",
                "layout_mean_search_time_s", "relative_difficulty",
                "deviation_pct", "search_success"):
        assert key in st, f"selected_target missing {key}"
    assert st["relative_difficulty"] in (
        "easier than layout average",
        "around layout average",
        "harder than layout average",
    )


def test_selected_target_is_not_cognitive_load(client):
    # The target result must never be presented as absolute cognitive load:
    # no cognitive-load / mental-effort keys and no 0-100 score field.
    img = _multibox_png()
    els = _cognitive_load(client, img).get("detected_elements", [])
    assert els
    resp = _scanpath(client, img, els[0]["id"])
    st = resp["selected_target"]
    forbidden = ("cognitive_load", "cognitive_load_score", "mental_effort",
                 "overload", "safety_risk", "load_score", "score")
    for key in st:
        low = key.lower()
        assert all(bad not in low for bad in forbidden), \
            f"selected_target key {key} implies cognitive load"
    # The headline is an absolute time in seconds, not a 0-100 score.
    assert st["search_time_s"] >= 0.0


# ---------------------------------------------------------------------------
# Sanity: input-shape robustness of the headline (supervisor #10)
#   The pipeline must behave sensibly across a small crop, a large fullscreen
#   render, and multiple targets on one screen - not crash or return
#   out-of-range / resolution-dependent values.
# ---------------------------------------------------------------------------

def _crop_png() -> bytes:
    """A small cropped region of a GUI: a single control on a light panel."""
    im = np.full((140, 220, 3), 244, np.uint8)
    cv2.rectangle(im, (30, 40), (150, 90), (40, 120, 210), -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


def _scaled_multibox_png(scale: int) -> bytes:
    """The same five-box layout as ``_multibox_png`` rendered at ``scale``x,
    used to check the headline is not wildly resolution-dependent."""
    im = np.full((600 * scale, 900 * scale, 3), 245, np.uint8)
    boxes = [
        ((60, 60), (180, 140), (200, 40, 40)),
        ((400, 80), (520, 160), (40, 160, 40)),
        ((700, 100), (820, 180), (40, 40, 200)),
        ((120, 360), (260, 460), (180, 120, 40)),
        ((520, 380), (660, 470), (120, 40, 160)),
    ]
    for (x1, y1), (x2, y2), col in boxes:
        cv2.rectangle(im, (x1 * scale, y1 * scale),
                      (x2 * scale, y2 * scale), col, -1)
    ok, buf = cv2.imencode(".png", im)
    assert ok
    return buf.tobytes()


def _headline(data) -> float:
    return data["layout"]["experimental_complexity_index"]


def test_cognitive_load_small_crop_is_valid(client):
    # A tiny crop with real content must return an in-range score above the
    # blank-canvas floor (proves the pipeline handles small inputs).
    data = _cognitive_load(client, _crop_png())
    score = _headline(data)
    assert 0.0 <= score <= 100.0
    assert score > 0.0


def test_cognitive_load_fullscreen_is_valid(client):
    # A large (2x) fullscreen render must return an in-range score and must
    # not overload to the ceiling just because there are more pixels.
    data = _cognitive_load(client, _scaled_multibox_png(2))
    score = _headline(data)
    assert 0.0 <= score <= 100.0


def test_cognitive_load_scale_invariance_sanity(client, monkeypatch):
    # The SAME layout re-rendered at 1x and 2x must read within 1.0 displayed
    # point (== 0.01 on the 0-1 HCEye index): the layout score now runs on the
    # canonical analysis image (a standardised analysis scale), so its measured
    # scale sensitivity is reduced. This is a scale-stability regression guard.
    #
    # Saliency and OCR are provided by the module-level autouse fakes: the SAME
    # synthetic heatmap and a valid no-text OCR report are used regardless of
    # input scale, so neither contaminates this layout-scale check.
    # The task/profile modifiers are identical across both requests, so the
    # headline difference reflects only the scale behaviour of the score.
    s1 = _headline(_cognitive_load(client, _scaled_multibox_png(1)))
    s2 = _headline(_cognitive_load(client, _scaled_multibox_png(2)))
    assert abs(s1 - s2) <= 1.0, f"scale changed headline too much: {s1} vs {s2}"


def test_multi_target_layout_stable_and_targets_valid(client):
    # With several targets on one screen, the layout value must stay fixed
    # while every per-target result is valid and independent.
    img = _multibox_png()
    base = _cognitive_load(client, img)
    layout = _headline(base)
    els = base.get("detected_elements", [])
    assert len(els) >= 3
    seen_ids = set()
    for e in els[:3]:
        resp = _scanpath(client, img, e["id"])
        st = resp["selected_target"]
        assert _headline(_cognitive_load(client, img)) == pytest.approx(layout)
        assert st["search_time_s"] >= 0.0
        assert st["relative_difficulty"] in (
            "easier than layout average",
            "around layout average",
            "harder than layout average",
        )
        seen_ids.add(st["target_id"])
    # Distinct targets must resolve to distinct target ids.
    assert len(seen_ids) == 3


# ---------------------------------------------------------------------------
# Framing / construct-validity regression guards (fix/audit-framing)
# ---------------------------------------------------------------------------
# These guard the *scientific framing* of the standard UI. They ensure the
# tool never presents unvalidated estimates as validated cognitive-load
# measurements, overload / safety verdicts, or WCAG conformance verdicts, and
# that the required honesty caveats stay in place. They read the served
# standard UI (index.html) as plain text.

def _standard_ui_html() -> str:
    """Return the standard UI markup as served at the app root."""
    path = os.path.join(ROOT, "stage1", "ui", "index.html")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_standard_ui_has_no_overload_or_safety_verdicts():
    html = _standard_ui_html().lower()
    forbidden = [
        "cognitive overload",
        "overload risk",
        "critical cognitive",
        "cognitive load measurement",
        "safety risk",
        "wcag compliant",
        "wcag non-compliant",
        "redesign recommended",
        "elevated cognitive demand",
        "notable attentional strain",
        "cognitive load estimation",  # visible pipeline heading must be renamed
        "wcag conformance audit",  # only allowed negated ("not a WCAG conformance audit")
    ]
    for phrase in forbidden:
        if phrase == "wcag conformance audit":
            # The only allowed occurrence is the explicit negation.
            assert "not a wcag conformance audit" in html, (
                "expected the 'not a WCAG conformance audit' caveat"
            )
            assert html.count("wcag conformance audit") == html.count(
                "not a wcag conformance audit"
            ), "found an un-negated 'WCAG conformance audit' claim"
        else:
            assert phrase not in html, f"forbidden framing phrase present: {phrase!r}"


def test_standard_ui_keeps_exploratory_caveats():
    html = _standard_ui_html().lower()
    # Layout score must be flagged as project-specific and unvalidated.
    assert "exploratory, project-specific heuristic" in html
    assert "not a validated cognitive-load measurement" in html
    # Contrast section must be WCAG-informed, not a conformance verdict.
    assert "wcag-informed contrast" in html
    assert "not a wcag conformance audit" in html


def test_standard_ui_has_no_rendered_load_or_performance_verdicts():
    # Honesty sweep: the standard UI must not present the heuristic as a
    # validated cognitive-load / human-performance verdict. Internal API keys
    # (cognitive_load_score, attention_demand) and factual descriptions of the
    # HCEye study (e.g. "under high load") are allowed; only rendered
    # verdict phrases are banned.
    html = _standard_ui_html().lower()
    forbidden = [
        "low load",
        "moderate load",
        "high load impact",
        "critical load drivers",
        "easy to process for most users",
        "typical attentional capacity",
        "performance may degrade",
        "very high attentional demand",
        "estimated load",
        "cognitive load score",  # spaces = user-facing label; underscore key is allowed
        "prediction is still valid",
    ]
    for phrase in forbidden:
        assert phrase not in html, f"forbidden rendered verdict present: {phrase!r}"
    # Bare "high load" is only allowed inside the factual HCEye description
    # ("fewer fixations under high load"); it must never appear as a band label.
    assert html.count("high load") == html.count("under high load"), (
        "found a 'high load' verdict outside the factual HCEye description"
    )
    # The neutral replacements must be present.
    assert "experimental layout complexity" in html
    assert "exploratory cross-signal review" in html
    assert "not calibrated validation" in html


def test_standard_ui_removes_legacy_context_adjusted_proxy_outputs():
    html = _standard_ui_html().lower()
    assert "search efficiency" not in html
    assert "attention demand" not in html
    assert "context-adjusted experimental output" not in html


def test_standard_ui_has_no_visible_driver_glance_panel():
    html = _standard_ui_html()
    # The glance helpers must be disabled no-ops (Future Work), not renderers.
    assert "function showGlanceMetrics(_gm) { /* intentionally disabled" in html
    assert "function hideGlanceMetrics() { /* intentionally disabled" in html
    lower = html.lower()
    # No automotive compliance conclusions presented to the user.
    for phrase in ("nhtsa compliant", "iso 15008 compliant", "glance compliant"):
        assert phrase not in lower, f"automotive compliance verdict present: {phrase!r}"


def test_standard_ui_keeps_selected_target_result_separate():
    html = _standard_ui_html()
    # Selected-target search difficulty is presented as its own card and kept
    # methodologically separate from the layout value.
    assert "Selected Target Search Difficulty" in html
    assert "the Experimental Layout Complexity value above is unchanged" in html


def test_standard_ui_final_framing_acceptance():
    # Final bounded framing patch: prohibit the exact remaining causal /
    # human-performance / mislabelled-validation phrases. Internal API keys
    # and the factual HCEye publication references remain allowed; the legacy
    # source-study statistic is kept only inside a hidden HTML comment.
    html = _standard_ui_html().lower()
    forbidden = [
        "lowers search load",
        "reacts more strongly to load",
        "high = more load",
        "reduces perceptual demand",
        "efficient pre-attentive processing",
        "increase scanning effort",
        "aids visual parsing",
        "reducing search time",
        "users must scan",
        "selection demand",
        "hceye behavioral validation",
        "umsi++ saliency validation",
        "cognitive load index (hceye)",
    ]
    for phrase in forbidden:
        assert phrase not in html, f"forbidden framing phrase present: {phrase!r}"

    # Positive assertions: the neutral replacements must be present.
    assert "internal checks and source-study correspondence" in html
    assert "not independent validation" in html
    assert (
        "hceye-derived rule index (unitless project-specific proxy)" in html
    )


def test_standard_ui_removes_unverified_umsi_class_mapping():
    # The unverified six-class softmax mapping must not exist as a dormant UI
    # branch or drive any semantic label / domain-reliability decision.
    html = _standard_ui_html()
    lower = html.lower()
    for forbidden in (
        "designclassbanner",
        "data.design_classification",
        "data.predicted_class",
        "out_of_domain",
        "within the model's intended ui domain",
    ):
        assert forbidden not in lower


def test_standard_ui_consumes_only_bounded_stage2_v1_schema():
    html = _standard_ui_html()
    for required in (
        "data.stage2_scenario_proxy",
        "data.cross_signal_review",
        "data.jokinen_diagnostic",
        "data.hceye_proxy_features",
        "experimental_layout_complexity",
        "scenario_direction",
        "scenario_score_bearing",
    ):
        assert required in html
    for forbidden in (
        "data.base_experimental_outputs",
        "data.context_adjusted_experimental_outputs",
        "data.task_descriptor",
        "data.big_five_profile",
        "data.base_prediction",
        "data.adjusted_prediction",
        "data.cognitive_load_features",
        ".cognitive_load_score",
        ".cognitive_load_index",
        "row.score",
        "d.hceye_features",
    ):
        assert forbidden not in html


# ---------------------------------------------------------------------------
# Regression: content-presence gate uses DIRECT structural evidence only.
#
# feature_congestion (a Rosenholtz clutter/texture proxy) was removed from the
# empty-canvas content-presence gate so that a busy-textured but element-sparse
# region cannot masquerade as "content present" and re-inflate the headline
# index. feature_congestion is retained in the `complexity` pathway (fixation
# reduction + highlight effectiveness). These tests pin that contract without
# touching any existing threshold.
# ---------------------------------------------------------------------------

def test_congestion_alone_cannot_create_content_presence():
    # No edges, no elements, full whitespace, but MAXIMAL feature_congestion:
    # the content-presence gate must stay ~0, so the final index must stay ~0.
    # feature_congestion must not be able to conjure content presence on a
    # screen that presents no direct structural evidence.
    from hceye.hceye_features import HCEyeFeatureExtractor

    ext = HCEyeFeatureExtractor()
    congested_but_empty = {
        "edge_density": 0.0,
        "feature_congestion": 1e9,   # far above the empirical max -> norm -> 1.0
        "interactive_element_density": 0.0,
        "layout_symmetry": 1.0,
        "visual_hierarchy": 0.0,
    }
    h = ext.extract_features(congested_but_empty, whitespace_ratio=1.0,
                             text_density=0.0)
    # Full whitespace + no edges/elements => content presence gate == 0 =>
    # final index (h[5]) must be exactly 0 regardless of congestion.
    assert h[5] == 0.0, (
        f"congestion alone produced non-zero content presence/index: h[5]={h[5]}")


def test_congestion_still_moves_index_via_complexity_pathway():
    # With content presence held at 1.0 by DIRECT evidence (full elements /
    # zero whitespace), feature_congestion must still change the HCEye output
    # through the retained `complexity` term, proving it was only removed from
    # the presence gate, not from the model.
    from hceye.hceye_features import HCEyeFeatureExtractor

    ext = HCEyeFeatureExtractor()
    base = {
        "edge_density": 0.0,
        "feature_congestion": 0.0,
        "interactive_element_density": 1.0,   # direct evidence -> presence 1.0
        "layout_symmetry": 0.5,
        "visual_hierarchy": 0.5,
    }
    low = ext.extract_features({**base, "feature_congestion": 0.0},
                               whitespace_ratio=0.0, text_density=0.5)
    high = ext.extract_features({**base, "feature_congestion": 1e9},
                                whitespace_ratio=0.0, text_density=0.5)
    # content presence is 1.0 in both (max includes 1-whitespace == 1.0), so any
    # index difference is purely the retained complexity pathway.
    assert low[5] != high[5], (
        "feature_congestion no longer affects the index via complexity; "
        f"low={low[5]} high={high[5]}")
