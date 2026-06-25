#!/usr/bin/env python3
"""
Stage 1 — Local Web Interface

Simple Flask server that serves the UI and processes uploaded screenshots
through the visual complexity pipeline.

Performance notes:
    The two most expensive pipeline steps are the visual complexity feature
    extraction (``compute_complexity_vector``) and the UMSI++ saliency model
    inference. To keep repeated analyses of the *same* image fast, both results
    are cached in-memory keyed by the SHA256 hash of the uploaded image bytes
    (see ``_visual_cache`` / ``_saliency_cache``). The caches are pure runtime
    optimizations — they never change the computed values, only avoid redundant
    recomputation. The UMSI++ model is additionally warmed up once at startup
    (``_warmup_saliency_model``) so the first real request does not pay the
    TensorFlow graph-build cost.

Usage:
    python app.py
    → Open http://localhost:5001 in your browser
"""

import json
import hashlib
import os
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Add stage1 and project root to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent))
from visual_complexity import compute_complexity_vector
from stage2.coherence_check import run_coherence_check

# Lazy-load saliency model (heavy TF import — only when needed)
_saliency_model = None

# In-memory caches keyed by the SHA256 hash of the uploaded image bytes.
# OrderedDict is used as a simple LRU: on a cache hit the entry is moved to the
# end, and once the cache exceeds its max size the oldest (front) entry is
# evicted. These caches only avoid recomputation; identical inputs always yield
# identical results.
_saliency_cache = OrderedDict()   # image_hash -> {"heatmap", "classif"}
_saliency_cache_max = 32          # max distinct images kept for saliency
_visual_cache = OrderedDict()     # image_hash -> visual complexity results dict
_visual_cache_max = 64            # max distinct images kept for visual features

# Empirical GUI reference distribution (mean / std / percentiles per feature),
# computed by build_feature_norms.py over 1,485 real GUI screenshots
# (495 web + 495 mobile + 495 desktop). It lets the pipeline express each
# feature value as a neutral z-score / percentile relative to the typical GUI.
# Loaded lazily once and cached for the process lifetime.
_FEATURE_NORMS_PATH = Path(__file__).parent / "data" / "results" / "feature_norms.json"
_feature_norms = None


def _load_feature_norms():
    """Load the GUI reference distribution from disk (cached). Returns a dict
    with ``meta`` and ``features`` keys; an empty structure if unavailable."""
    global _feature_norms
    if _feature_norms is None:
        try:
            with open(_FEATURE_NORMS_PATH) as f:
                _feature_norms = json.load(f)
        except (OSError, ValueError):
            _feature_norms = {"meta": {}, "features": {}}
    return _feature_norms


def _empirical_percentile(stats, value):
    """Estimate the percentile of ``value`` via piecewise-linear interpolation
    over the stored reference quantiles (min, p5, p25, p50, p75, p95, max)."""
    anchors = [
        (stats["min"], 0.0), (stats["p5"], 5.0), (stats["p25"], 25.0),
        (stats["p50"], 50.0), (stats["p75"], 75.0), (stats["p95"], 95.0),
        (stats["max"], 100.0),
    ]
    if value <= anchors[0][0]:
        return 0.0
    if value >= anchors[-1][0]:
        return 100.0
    for (x0, p0), (x1, p1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return p1
            return p0 + (p1 - p0) * (value - x0) / (x1 - x0)
    return 50.0


def _comparison_band(z):
    """Map a z-score to a neutral, direction-aware band label."""
    az = abs(z)
    if az < 1.0:
        return "typical"
    if az < 2.0:
        return "above_typical" if z > 0 else "below_typical"
    return "far_above_typical" if z > 0 else "far_below_typical"


def compare_to_reference(feature_values):
    """Compare measured feature values against the empirical GUI reference.

    Args:
        feature_values: mapping of ``feature_key -> value``. Keys without a
            reference entry (or with zero std) are skipped.

    Returns:
        Mapping of ``feature_key -> {value, mean, std, z, percentile, band, n}``.
    """
    norms = _load_feature_norms().get("features", {})
    out = {}
    for key, value in feature_values.items():
        stats = norms.get(key)
        if not stats or not stats.get("std"):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        z = (v - stats["mean"]) / stats["std"]
        # Signed change relative to the reference baseline (the typical GUI).
        # Positive = this screen is above the reference mean, negative = below.
        # This keeps direction/sign visible, unlike a 0-100 % range mapping.
        mean = stats["mean"]
        delta_pct = ((v - mean) / mean * 100.0) if mean else None
        out[key] = {
            "value": v,
            "mean": mean,
            "std": stats["std"],
            "z": z,
            "delta_pct": delta_pct,
            "percentile": _empirical_percentile(stats, v),
            "band": _comparison_band(z),
            "n": stats.get("n"),
        }
    return out


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _get_saliency_model():
    """Lazy-load the UMSI++ saliency model (avoids TF startup penalty on every request)."""
    global _saliency_model
    if _saliency_model is None:
        from saliency.umsi_model import UMSIPlus
        weights = Path(__file__).parent.parent / "saliency" / "weights" / "model_weights" / "saliency_models" / "UMSI++" / "umsi++.hdf5"
        _saliency_model = UMSIPlus(str(weights))
    return _saliency_model


def _hash_upload(file_storage):
    """Return ``(sha256_hex, raw_bytes)`` for an uploaded image.

    Reads the full upload stream to compute a content hash (used as the cache
    key) and then rewinds the stream so the caller can still persist the file.
    """
    data = file_storage.read()
    file_storage.stream.seek(0)  # rewind so the bytes can be written to disk later
    return hashlib.sha256(data).hexdigest(), data


def _resolve_target_index(elements, target_id, target_x, target_y):
    """Resolve a target element index from selection parameters.

    Selection precedence:
      1. ``target_id`` (exact element id match), when provided.
      2. ``(target_x, target_y)`` click position in original-image pixels:
         prefer the element whose bounding box contains the point; otherwise
         fall back to the element with the nearest center.

    Returns the element index, or ``None`` if no target can be resolved.
    """
    # 1. Explicit element id.
    if target_id is not None:
        for idx, el in enumerate(elements):
            if el.get("id") == target_id:
                return idx
        return None

    if target_x is None or target_y is None:
        return None

    # 2a. Point-in-bbox hit test.
    for idx, el in enumerate(elements):
        bbox = el.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        if bx <= target_x <= bx + bw and by <= target_y <= by + bh:
            return idx

    # 2b. Nearest-center fallback.
    best_idx = None
    best_dist = None
    for idx, el in enumerate(elements):
        center = el.get("center")
        if not center or len(center) < 2:
            continue
        dist = (center[0] - target_x) ** 2 + (center[1] - target_y) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _predict_saliency_cached(image_hash, image_path):
    """Run UMSI++ saliency prediction with an LRU hash cache.

    Returns ``(heatmap, classif, cache_hit)`` where ``cache_hit`` is True when
    the result was served from cache. Caching avoids repeated (expensive)
    TensorFlow inference for identical images.
    """
    cached = _saliency_cache.get(image_hash)
    if cached is not None:
        _saliency_cache.move_to_end(image_hash)  # mark as most-recently-used
        return cached["heatmap"], cached["classif"], True

    model = _get_saliency_model()
    heatmap, classif = model.predict_saliency(str(image_path), return_classif=True)
    _saliency_cache[image_hash] = {"heatmap": heatmap, "classif": classif}
    # Evict the oldest entries once the cache grows beyond its size limit.
    while len(_saliency_cache) > _saliency_cache_max:
        _saliency_cache.popitem(last=False)
    return heatmap, classif, False


def _compute_visual_cached(image_hash, image_path):
    """Compute the 8 visual complexity features with an LRU hash cache.

    Returns ``(results, cache_hit)``. ``compute_complexity_vector`` is the
    single most expensive step in the pipeline, so caching it by image hash
    gives the largest speedup for repeated analyses of the same screenshot.
    """
    cached = _visual_cache.get(image_hash)
    if cached is not None:
        _visual_cache.move_to_end(image_hash)  # mark as most-recently-used
        return cached, True

    results = compute_complexity_vector(str(image_path))
    _visual_cache[image_hash] = results
    # Evict the oldest entries once the cache grows beyond its size limit.
    while len(_visual_cache) > _visual_cache_max:
        _visual_cache.popitem(last=False)
    return results, False


def _warmup_saliency_model():
    """Warm up TensorFlow model once at startup to reduce first-request latency."""
    if not _as_bool(os.getenv("STAGE1_WARMUP", "1"), default=True):
        print("[Warmup] Skipped (STAGE1_WARMUP=0)")
        return

    warmup_path = UPLOAD_DIR / f"_warmup_{uuid.uuid4().hex[:8]}.png"
    try:
        import cv2
        import numpy as np

        # Synthetic GUI-like image (text + blocks) to trigger model graph load.
        img = np.zeros((320, 480, 3), dtype=np.uint8)
        img[:] = (28, 30, 37)
        cv2.rectangle(img, (24, 24), (456, 86), (58, 62, 74), -1)
        cv2.rectangle(img, (24, 106), (280, 286), (45, 49, 61), -1)
        cv2.rectangle(img, (300, 106), (456, 286), (45, 49, 61), -1)
        cv2.putText(img, "Warmup", (34, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 190, 220), 2, cv2.LINE_AA)
        cv2.imwrite(str(warmup_path), img)

        image_hash = hashlib.sha256(img.tobytes()).hexdigest()
        _predict_saliency_cached(image_hash, warmup_path)
        print("[Warmup] UMSI++ model loaded and primed.")
    except Exception as e:
        print(f"[Warmup] Skipped due to error: {e}")
    finally:
        if warmup_path.exists():
            warmup_path.unlink()

app = Flask(__name__, static_folder="ui", static_url_path="")

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    from flask import make_response
    resp = make_response(send_from_directory("ui", "index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Accept an uploaded image and return the 8-feature complexity vector."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Hash the upload first so identical images reuse cached feature results.
    image_hash, image_bytes = _hash_upload(file)
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    try:
        # Copy the cached dict before mutating it so the cache stays pristine.
        results, cache_hit = _compute_visual_cached(image_hash, filepath)
        results = dict(results)
        results["filename"] = file.filename
        results["visual_cache_hit"] = cache_hit
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route("/api/features", methods=["GET"])
def features_info():
    """Return metadata about the 8 features."""
    features = [
        {
            "key": "shannon_entropy",
            "name": "Shannon Entropy",
            "description": "Global information density of the image. Higher = more visual information competing for attention.",
            "range": "[0, 8]",
            "reference": "Shannon (1948)",
            "icon": "📊"
        },
        {
            "key": "edge_density",
            "name": "Edge Density",
            "description": "Proportion of pixels classified as edges. A proxy for structural complexity — more boundaries = more parsing effort.",
            "range": "[0, 1]",
            "reference": "Canny edge detection (AIM m4)",
            "icon": "📐"
        },
        {
            "key": "feature_congestion",
            "name": "Feature Congestion",
            "description": "Multi-scale clutter combining color covariance, contrast variance, and orientation energy. Higher = more visual noise.",
            "range": "[0, ∞)",
            "reference": "Rosenholtz et al. (2007) — AIM m8",
            "icon": "🌀"
        },
        {
            "key": "subband_entropy",
            "name": "Subband Entropy",
            "description": "Redundancy-based clutter via steerable pyramid decomposition. Higher = more unpredictable spatial frequency content.",
            "range": "[0, ∞)",
            "reference": "Rosenholtz et al. (2007) — AIM m7",
            "icon": "🔬"
        },
        {
            "key": "layout_symmetry",
            "name": "Layout Symmetry",
            "description": "Degree of axial balance (vertical + horizontal). Higher = more symmetric = less visual search needed.",
            "range": "[0, 1]",
            "reference": "Miniukovich & De Angeli (2015)",
            "icon": "⚖️"
        },
        {
            "key": "chromatic_coherence",
            "name": "Chromatic Coherence",
            "description": "Color palette fragmentation combining luminance variance, colorfulness, and hue/saturation spread. Higher = more fragmented.",
            "range": "[0, 1]",
            "reference": "Hasler & Süsstrunk (2003)",
            "icon": "🎨"
        },
        {
            "key": "visual_hierarchy",
            "name": "Visual Hierarchy",
            "description": "Strength of layered visual structure (contrast gradients + size dominance). Higher = clearer hierarchy = less search effort.",
            "range": "[0, 1]",
            "reference": "Tuch et al. (2009)",
            "icon": "📏"
        },
        {
            "key": "interactive_element_density",
            "name": "Interactive Element Density",
            "description": "Estimated count of UI controls per area. Higher = more action possibilities = higher decisional load.",
            "range": "[0, ∞)",
            "reference": "Custom (contour-based)",
            "icon": "🔘"
        },
    ]
    return jsonify(features)


@app.route("/api/saliency", methods=["POST"])
def saliency():
    """Predict saliency heatmap and extract saliency features for an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    image_hash, image_bytes = _hash_upload(file)
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    try:
        import base64
        import cv2
        import numpy as np
        from saliency.saliency_features import extract_saliency_features

        heatmap, classif, cache_hit = _predict_saliency_cached(image_hash, filepath)
        model = _get_saliency_model()

        # Extract saliency features
        features = extract_saliency_features(heatmap)

        # Create colored heatmap for visualization (base64 PNG)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        _, buf = cv2.imencode(".png", heatmap_colored)
        heatmap_b64 = base64.b64encode(buf).decode("utf-8")

        # Classification results
        classif_dict = {
            cls: float(prob)
            for cls, prob in zip(model.DESIGN_CLASSES, classif)
        }

        return jsonify({
            "filename": file.filename,
            "features": features,
            "classification": classif_dict,
            "predicted_class": model.DESIGN_CLASSES[int(np.argmax(classif))],
            "heatmap_png_base64": heatmap_b64,
            "saliency_cache_hit": cache_hit,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/search-time", methods=["POST"])
def search_time():
    """
    Predict visual search time per UI element using the Jokinen 2020 model.

    This is the CENTRAL cognitive metric of the thesis:
      - Detects UI elements from the uploaded screenshot
      - Computes UMSI++ saliency per element (deep bottom-up signal)
      - Runs Monte Carlo simulation of novice visual search (EMMA + feature guidance)
      - Returns predicted search time per element + aggregate statistics

    Reference:
        Jokinen, J.P.P. et al. (2020). Adaptive feature guidance: Modelling
        visual search with graphical layouts. IJHCS, 136, 102376.

    Request:
        POST multipart/form-data with field "image" (PNG/JPG screenshot)
        Optional query params:
            n_simulations (int): Monte Carlo trials per element (default: 100)
            use_saliency (bool): Use UMSI++ saliency (default: true)

    Response JSON:
        {
            "filename": str,
            "n_elements": int,
            "mean_search_time_s": float,
            "max_search_time_s": float,
            "min_search_time_s": float,
            "predicted_difficulty": str,  # "easy"|"moderate"|"difficult"|"very_hard"
            "per_element": [
                {"id": int, "search_time_s": float, "fixation_count": float,
                 "bbox": [x,y,w,h], "center": [cx,cy], "color_category": str},
                ...
            ],
            "model_info": {...}
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Parse optional query parameters
    n_simulations = request.args.get("n_simulations", 100, type=int)
    use_saliency = request.args.get("use_saliency", "true").lower() != "false"

    image_hash, image_bytes = _hash_upload(file)
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    try:
        import cv2
        import numpy as np
        from cognitive.element_detector import detect_elements
        from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

        # Load image
        img = cv2.imread(str(filepath))
        if img is None:
            return jsonify({"error": "Cannot read image"}), 400

        # Step 1: Detect UI elements
        elements = detect_elements(img)
        if len(elements) == 0:
            return jsonify({
                "filename": file.filename,
                "error": "No UI elements detected",
                "n_elements": 0,
            }), 200

        # Step 2: Get saliency map (optional)
        saliency_map = None
        if use_saliency:
            try:
                saliency_map, _, _ = _predict_saliency_cached(image_hash, filepath)
            except Exception:
                pass  # Fall back to feature-only mode

        # Step 3: Run Jokinen model
        params = JokinenParams(
            n_simulations=min(n_simulations, 500),  # Cap at 500 for performance
            random_seed=42,
        )
        jokinen = JokinenSearchModel(params)
        results = jokinen.predict_search_times(
            elements=elements,
            saliency_map=saliency_map,
            image_shape=img.shape[:2],
        )

        # Add metadata
        results["filename"] = file.filename
        results["model_info"] = jokinen.get_model_info()
        results["saliency_used"] = saliency_map is not None

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/scanpath-to-target", methods=["POST"])
def scanpath_to_target():
    """
    Predict the visual-search scanpath toward a user-selected target element.

    The user picks a target on the screenshot (by click position or element id);
    this endpoint returns the fixation sequence the Jokinen 2020 Adaptive Feature
    Guidance model traverses while a novice searches for that target, starting
    from the screen center. This is a TASK-DRIVEN scanpath (goal-directed search),
    not a free-viewing scanpath.

    Reference:
        Jokinen, J.P.P. et al. (2020). Adaptive feature guidance: Modelling
        visual search with graphical layouts. IJHCS, 136, 102376.

    Request:
        POST multipart/form-data with field "image" (PNG/JPG screenshot).
        Target selection (one of):
            - target_x, target_y (float, query params): click position in
              ORIGINAL-image pixel coordinates. The element whose bounding box
              contains the point is used; otherwise the element with the nearest
              center is chosen.
            - target_id (int, query param): id of a previously detected element.
        Optional:
            n_simulations (int): Monte Carlo trials (default: 100).
            use_saliency (bool): use UMSI++ saliency (default: true).

    Response JSON:
        {
            "filename": str,
            "n_elements": int,
            "target_id": int,
            "target_center": [cx, cy],
            "target_bbox": [x, y, w, h],
            "scanpath": {
                "fixations": [
                    {"x","y","order","t_cumulative_s","step_time_s",
                     "element_id","is_target"}, ...
                ],
                "total_time_s": float,
                "n_fixations": int,
                "image_width": int,
                "image_height": int,
                "basis": "jokinen_search_model",
                "is_target_driven": true
            }
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Target selection params.
    target_x = request.args.get("target_x", default=None, type=float)
    target_y = request.args.get("target_y", default=None, type=float)
    target_id = request.args.get("target_id", default=None, type=int)
    if target_id is None and (target_x is None or target_y is None):
        return jsonify({
            "error": "Provide either target_id or both target_x and target_y"
        }), 400

    n_simulations = request.args.get("n_simulations", 100, type=int)
    use_saliency = request.args.get("use_saliency", "true").lower() != "false"

    image_hash, image_bytes = _hash_upload(file)
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    try:
        import cv2
        from cognitive.element_detector import detect_elements
        from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

        img = cv2.imread(str(filepath))
        if img is None:
            return jsonify({"error": "Cannot read image"}), 400

        elements = detect_elements(img)
        if len(elements) == 0:
            return jsonify({
                "filename": file.filename,
                "error": "No UI elements detected",
                "n_elements": 0,
            }), 200

        # Resolve the target element index from the selection params.
        target_idx = _resolve_target_index(elements, target_id, target_x, target_y)
        if target_idx is None:
            return jsonify({"error": "Could not resolve target element"}), 400

        # Saliency (optional) so the scanpath matches the search-time pipeline.
        saliency_map = None
        if use_saliency:
            try:
                saliency_map, _, _ = _predict_saliency_cached(image_hash, filepath)
            except Exception as e:
                print(f"[Scanpath] Saliency unavailable, feature-only mode: {e!r}")

        params = JokinenParams(
            n_simulations=min(n_simulations, 500),
            random_seed=42,
        )
        jokinen = JokinenSearchModel(params)
        scanpath = jokinen.predict_scanpath_to_target(
            target_idx=target_idx,
            elements=elements,
            saliency_map=saliency_map,
            image_shape=img.shape[:2],
        )

        target_elem = elements[target_idx]
        return jsonify({
            "filename": file.filename,
            "n_elements": len(elements),
            "target_id": target_elem.get("id"),
            "target_center": list(target_elem.get("center", [])),
            "target_bbox": list(target_elem.get("bbox", [])),
            "saliency_used": saliency_map is not None,
            "scanpath": scanpath,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/cognitive-load", methods=["POST"])
def cognitive_load():
    """
    Compute cognitive load features using HCEye-derived sensitivity model.

    Combines visual complexity (v∈ℝ⁸) + saliency (s∈ℝ⁵) + HCEye cognitive
    load sensitivity (h∈ℝ⁶) into a full feature vector for Stage 2.

    Response JSON:
        {
            "filename": str,
            "visual_features": {...},         # v∈ℝ⁸
            "saliency_features": {...},       # s∈ℝ⁵  
            "cognitive_load_features": {...},  # h∈ℝ⁶
            "cognitive_load_index": float,    # Combined CLI (0-1)
            "full_feature_vector": [...]      # ℝ¹⁹ for Stage 2
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Hash once and reuse the same key for both the visual and saliency caches.
    image_hash, image_bytes = _hash_upload(file)
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(image_bytes)

    try:
        import numpy as np
        from hceye.hceye_features import HCEyeFeatureExtractor
        from stage2.task_descriptor import TaskDescriptor
        from stage2.user_profile import get_profile
        from stage2.regression_model import Stage2Model

        use_trained_model = _as_bool(
            request.form.get("use_trained_model", request.args.get("use_trained_model")),
            default=False,
        )

        task_descriptor = TaskDescriptor(
            task_type=request.form.get("task_type", "search"),
            target_specificity=request.form.get("target_specificity", "medium"),
            time_pressure=request.form.get("time_pressure", "medium"),
            search_mode=request.form.get("search_mode", "known_item"),
        )
        profile = get_profile(request.form.get("profile_preset", "neutral"))

        # Step 1: Visual complexity (v∈ℝ⁸)
        vis_results, visual_cache_hit = _compute_visual_cached(image_hash, filepath)
        vis_results = dict(vis_results)
        v = np.array([
            vis_results["shannon_entropy"],
            vis_results["edge_density"],
            vis_results["feature_congestion"],
            vis_results["subband_entropy"],
            vis_results["layout_symmetry"],
            vis_results["chromatic_coherence"],
            vis_results["visual_hierarchy"],
            vis_results["interactive_element_density"],
        ], dtype=np.float32)

        # Step 2: Saliency features (s∈ℝ⁵) — optional
        s = None
        saliency_dict = {}
        saliency_overlay_b64 = None
        try:
            import base64, cv2
            from saliency.saliency_features import extract_saliency_features
            heatmap, _, cache_hit = _predict_saliency_cached(image_hash, filepath)
            saliency_dict = extract_saliency_features(heatmap)
            s = np.array([
                saliency_dict["saliency_dispersion"],
                saliency_dict["saliency_entropy"],
                saliency_dict["saliency_coverage"],
                saliency_dict["saliency_peak_count"],
                saliency_dict["saliency_center_bias"],
            ], dtype=np.float32)
        except Exception as e:
            # Do NOT fail silently: the cognitive-load model degrades to image-only
            # features (s=None) when saliency is missing. Log loudly so a broken
            # saliency stage is visible during the study instead of silently
            # producing a partial result that still looks "green".
            cache_hit = False
            s = None
            saliency_dict = {}
            print(f"[Saliency] Saliency features unavailable: {e!r}")

        # Build colored overlay: original image blended with JET-colormap heatmap.
        # This is purely cosmetic (visualization only) and is kept in a SEPARATE
        # try-block so that a failure here can never discard the real saliency
        # features computed above.
        # Alpha blend: 0.55 original + 0.45 heatmap (warm-on-dark, readable on
        # both light and dark UIs). JET colormap: blue=low, red=high saliency.
        # The overlay is returned as JPEG (quality 85) to keep response size
        # manageable; PNG would be ~3–5× larger for typical screenshot dimensions.
        # Reference for JET colormap in saliency visualization:
        #   Itti, L. & Koch, C. (2001). Computational modelling of visual
        #   attention. Nature Reviews Neuroscience, 2(3), 194–203.
        if s is not None:
            try:
                import base64, cv2
                orig = cv2.imread(str(filepath))
                if orig is not None:
                    heat_u8 = (heatmap * 255).astype(np.uint8)
                    heat_colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                    heat_resized = cv2.resize(
                        heat_colored, (orig.shape[1], orig.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    overlay = cv2.addWeighted(orig, 0.55, heat_resized, 0.45, 0)
                    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    saliency_overlay_b64 = base64.b64encode(buf).decode("utf-8")
            except Exception as e:
                # Cosmetic only: real features are unaffected. Log, don't crash.
                print(f"[Saliency] Overlay rendering failed (features unaffected): {e!r}")

        # Step 3: HCEye cognitive load features (h∈ℝ⁶)
        lookup_path = Path(__file__).parent.parent / "hceye" / "sensitivity_lookup.json"
        extractor = HCEyeFeatureExtractor(str(lookup_path))
        h = extractor.extract_features(v, s)
        cog_names = extractor.get_feature_names()
        cog_dict = dict(zip(cog_names, h.tolist()))

        # Step 4: Optional task/profile vectors
        t = task_descriptor.to_vector()
        t_dict = task_descriptor.as_dict()
        p = np.array(profile["vector"], dtype=np.float32)

        # Build full feature vector for Stage 2 base model (v⁸ + s⁵ + h⁶ = ℝ¹⁹)
        parts = [v]
        if s is not None:
            parts.append(s)
        else:
            parts.append(np.zeros(5, dtype=np.float32))
        parts.append(h)
        base_vector = np.concatenate(parts)

        # Extended vector for downstream experiments (base + descriptor + profile)
        extended_vector = np.concatenate([base_vector, t, p]).tolist()

        model_path = Path(__file__).parent.parent / "stage2" / "models" / "stage2_model.pkl"
        predictions = {
            "cognitive_load_score": float(h[5] * 100.0),
            "search_efficiency": float(np.clip(1.0 - h[3], 0.0, 1.0)),
            "attention_demand": float(np.clip(h[5] + 0.15, 0.0, 1.0)),
        }
        prediction_source = "hceye_rule_based"
        if use_trained_model and model_path.exists():
            stage2_model = Stage2Model(model_path=str(model_path))
            predictions = stage2_model.predict(base_vector)
            prediction_source = "stage2_trained_model"

        base_score = float(predictions["cognitive_load_score"])
        descriptor_modifier = float(t_dict["modifier"])
        profile_modifier = float(profile["modifier"])
        adjusted_score = float(np.clip(base_score + descriptor_modifier + profile_modifier, 0.0, 100.0))
        search_efficiency = float(np.clip(
            predictions["search_efficiency"] - 0.0025 * descriptor_modifier - 0.0030 * profile_modifier,
            0.0,
            1.0,
        ))
        attention_demand = float(np.clip(
            predictions["attention_demand"] + 0.0040 * descriptor_modifier + 0.0040 * profile_modifier,
            0.0,
            1.0,
        ))

        # Coherence check — validate internal consistency of pipeline outputs
        # Extract Jokinen search metrics if available (computed on-demand here).
        mean_search_time_s: float | None = None
        estimated_fixation_count: float | None = None
        try:
            import cv2
            from cognitive.jokinen_model import JokinenSearchModel, JokinenParams
            from cognitive.element_detector import detect_elements
            # detect_elements expects a BGR image array (cv2.imread), not a path.
            jokinen_img = cv2.imread(str(filepath))
            if jokinen_img is None:
                raise ValueError(f"Cannot read image for Jokinen search model: {filepath}")
            elements = detect_elements(jokinen_img)
            # Reuse the already-computed UMSI++ heatmap when saliency succeeded
            # (avoids re-running the slow saliency step). s is not None implies
            # the saliency block ran past the heatmap assignment above.
            jokinen_saliency = heatmap if s is not None else None
            jokinen_model = JokinenSearchModel(JokinenParams())
            jresult = jokinen_model.predict_search_times(
                elements=elements,
                saliency_map=jokinen_saliency,
                image_shape=jokinen_img.shape[:2],
            )
            mean_search_time_s = float(jresult["mean_search_time_s"])
            # The model returns per-element fixation counts; aggregate to a
            # layout-wide mean for the coherence check (no aggregate key exists).
            per_elem = jresult.get("per_element", [])
            if per_elem:
                estimated_fixation_count = float(
                    sum(e["fixation_count"] for e in per_elem) / len(per_elem)
                )
        except Exception as e:
            # Do NOT fail silently: the coherence check depends on these values.
            # Log loudly so a broken search model is visible during the study.
            print(f"[Jokinen] Search model unavailable for coherence check: {e!r}")

        saliency_spread = saliency_dict.get("saliency_dispersion") if saliency_dict else None
        coherence = run_coherence_check(
            saliency_spread=saliency_spread,
            estimated_fixation_count=estimated_fixation_count,
            mean_search_time_s=mean_search_time_s,
            cognitive_load_score=adjusted_score,
        )

        # Per-feature comparison against the empirical GUI reference distribution
        # (z-score / percentile vs. the typical GUI over 1,485 screenshots).
        reference_input = dict(vis_results)
        if saliency_dict:
            reference_input.update(saliency_dict)
        reference = compare_to_reference(reference_input)
        reference_meta = _load_feature_norms().get("meta", {})

        return jsonify({
            "filename": file.filename,
            "visual_features": vis_results,
            "visual_cache_hit": visual_cache_hit,
            "saliency_features": saliency_dict,
            "saliency_overlay_b64": saliency_overlay_b64,
            "saliency_cache_hit": cache_hit,
            "cognitive_load_features": cog_dict,
            "task_descriptor": t_dict,
            "big_five_profile": profile,
            "base_prediction": predictions,
            "adjusted_prediction": {
                "cognitive_load_score": adjusted_score,
                "search_efficiency": search_efficiency,
                "attention_demand": attention_demand,
            },
            "prediction_source": prediction_source,
            "trained_model_requested": use_trained_model,
            "trained_model_available": model_path.exists(),
            "cognitive_load_index": float(h[5]),
            "full_feature_vector": extended_vector,
            "vector_dimensions": f"v({len(v)}) + s(5) + h({len(h)}) + t({len(t)}) + p({len(p)}) = {len(extended_vector)}",
            "coherence": coherence,
            "reference": reference,
            "reference_meta": reference_meta,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath.exists():
            filepath.unlink()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Stage 1 — Visual Complexity Analyzer")
    print("  Endpoints:")
    print("    POST /api/analyze         → v∈ℝ⁸ visual complexity")
    print("    POST /api/saliency        → s∈ℝ⁵ saliency features")
    print("    POST /api/search-time     → Jokinen search time")
    print("    POST /api/cognitive-load   → h∈ℝ⁶ + full vector ℝ¹⁹")
    print("  Open: http://localhost:5001")
    print("=" * 50 + "\n")
    # Prime the UMSI++ model before accepting traffic so the first real
    # request doesn't pay the TensorFlow graph-build cost.
    _warmup_saliency_model()
    app.run(host="localhost", port=5001, debug=True)
