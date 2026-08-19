#!/usr/bin/env python3
"""
Stage 1 — Local Web Interface

Simple Flask server that serves the UI and processes uploaded screenshots
through the visual complexity pipeline.

Performance notes:
    The two most expensive pipeline steps are the visual complexity feature
    extraction (``compute_complexity_vector``) and the UMSI++ saliency model
    inference. To keep repeated analyses of the *same* image fast, both results
    are cached in-memory using the uploaded-image SHA256 plus the applicable P6
    model/postprocessing/norm/runtime identity (see ``_visual_cache`` /
    ``_saliency_cache``). The caches are pure runtime
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
from flask.json.provider import DefaultJSONProvider

# Add stage1 and project root to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent))
from visual_complexity import (
    compute_complexity_vector,
    CANONICAL_ANALYSIS_VERSION,
    FEATURE_KEYS,
    ImageTooSmallError,
    MIN_CANONICAL_INPUT_LONG_SIDE,
    MIN_CANONICAL_INPUT_SHORT_SIDE,
)
from stage2.coherence_check import run_cross_signal_review
from stage2.scenario_proxy import ScenarioProxyInputError, build_scenario_proxy
from hceye.hceye_features import FeatureNormsError
from saliency.checkpoint_identity import verify_umsi_checkpoint
from reproducibility import (
    saliency_cache_identity,
    study_reproducibility_metadata,
    visual_cache_identity,
)


class SaliencyUnavailableError(Exception):
    """Raised when the mandatory saliency stage (model init, weights load,
    inference, postprocessing, or saliency-feature extraction) could not be
    completed for a request that requires a full score-bearing analysis.

    Carries a stable, machine-readable ``code`` and a client-safe ``message``.
    A route that requires a complete analysis must let this propagate into a
    structured, non-200 failure response instead of degrading to a partial
    result.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ScoreInputUnavailableError(Exception):
    """Raised when mandatory score-driving layout/OCR inputs are unavailable.

    The underlying exception is retained only in the server-side exception
    chain.  ``message`` is fixed and client-safe so local paths, dependency
    details, and stack traces can never cross the API boundary.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Stage1VectorUnavailableError(Exception):
    """Raised when a mandatory Stage-1 block or x19 is non-finite.

    The public response uses only the fixed client-safe ``message``.  Invalid
    numerical data must never cross the API boundary as JSON NaN/Infinity or
    appear alongside a valid-looking Stage-1 score.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidImageUploadError(Exception):
    """Raised when supported-extension upload bytes cannot decode as an image.

    This is a client-correctable input failure, not a pipeline/model failure.
    The public response is fixed and must not expose decoder details.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class UploadStorageUnavailableError(Exception):
    """Raised when a validated upload cannot be persisted for analysis."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# Lazy-load saliency model (heavy TF import — only when needed)
_saliency_model = None

# In-memory caches keyed by the upload SHA256 plus the complete P6 analysis
# identity. A checkpoint, postprocessor, extractor, runtime, or norm change
# therefore cannot reuse a result produced under the previous identity.
# OrderedDict is used as a simple LRU: on a cache hit the entry is moved to the
# end, and once the cache exceeds its max size the oldest (front) entry is
# evicted. These caches only avoid recomputation; identical inputs always yield
# identical results.
_saliency_cache = OrderedDict()   # versioned key -> {"heatmap", "classif"}
_saliency_cache_max = 32          # max distinct images kept for saliency
_visual_cache = OrderedDict()     # versioned key -> visual results dict
_visual_cache_max = 64            # max distinct images kept for visual features

# Empirical GUI reference distribution (mean / std / percentiles per feature),
# computed by build_feature_norms.py over 1,485 real GUI screenshots
# (495 web + 495 mobile + 495 desktop). It lets the pipeline express each
# feature value as a neutral z-score / percentile relative to the typical GUI.
# Loaded lazily once and cached for the process lifetime.
_FEATURE_NORMS_PATH = Path(__file__).parent / "data" / "results" / "feature_norms.json"
_feature_norms = None

# Stable public Stage-1 boundary contract.  The saliency order deliberately
# follows the production API assembly order audited for x19; it is not inferred
# from dictionary iteration or from the different prose order in the saliency
# module documentation.
STAGE1_SALIENCY_FEATURE_NAMES = (
    "saliency_dispersion",
    "saliency_entropy",
    "saliency_coverage",
    "saliency_peak_count",
    "saliency_center_bias",
)
STAGE1_VECTOR_DIMENSIONS = 19
STAGE1_VECTOR_DTYPE = "float32"
LAYOUT_OCR_ERROR_CODE = "layout_ocr_unavailable"
LAYOUT_OCR_ERROR_MESSAGE = (
    "Required layout/OCR analysis is unavailable; no score was produced. "
    "Check the server's layout/OCR setup and retry."
)
STAGE1_VECTOR_ERROR_CODE = "stage1_vector_invalid"
STAGE1_VECTOR_ERROR_MESSAGE = (
    "A required Stage-1 feature was non-finite; no vector or score was produced."
)
INVALID_IMAGE_ERROR_CODE = "invalid_image"
INVALID_IMAGE_ERROR_MESSAGE = (
    "Uploaded file could not be decoded as a supported image; "
    "no analysis was performed."
)
IMAGE_RESOURCE_LIMIT_ERROR_CODE = "image_resource_limit"
IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE = (
    "Uploaded image dimensions or decoded size exceed the supported analysis "
    "limits; no analysis was performed."
)
UPLOAD_STORAGE_ERROR_CODE = "upload_storage_unavailable"
UPLOAD_STORAGE_ERROR_MESSAGE = (
    "Uploaded image could not be stored for analysis; no analysis was performed."
)
AUXILIARY_SALIENCY_ERROR_CODE = "saliency_unavailable"
AUXILIARY_SALIENCY_ERROR_MESSAGE = (
    "Requested saliency analysis is unavailable; no diagnostic result was produced. "
    "Retry when the saliency stage is available or explicitly request "
    "use_saliency=false for the separate feature-only contract."
)
# P5 / AG-07..09 production semantics policy.  This metadata is returned with
# every successful score-bearing response so API consumers cannot mistake the
# project-specific HCEye adaptation for a validated cognitive-load measure or
# infer semantic labels from the unverified UMSI classification-head order.
SCIENTIFIC_SEMANTICS = {
    "construct": "exploratory_project_specific_layout_proxy",
    "validated_cognitive_load_measurement": False,
    "hceye_derivation_reproducible_from_repository": False,
    "hceye_coefficient_provenance": (
        "hash_pinned_external_csv_with_repository_verifier"
    ),
    "umsi_class_label_mapping_verified": False,
}

# One production format policy for every single-image endpoint. Screen-set
# analysis intentionally adds animated GIF because it can decode multiple
# frames; WebP is not part of either public upload contract.
SINGLE_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff"})
SCREEN_SET_EXTENSIONS = SINGLE_IMAGE_EXTENSIONS | frozenset({".gif"})


def _load_feature_norms():
    """Load the production GUI reference distribution from disk (cached).

    Fail-closed: raises FeatureNormsError if the file is missing, unreadable,
    or not valid JSON. A full score-bearing analysis must never silently
    substitute an empty reference structure for the production norms file.
    """
    global _feature_norms
    if _feature_norms is None:
        try:
            with open(_FEATURE_NORMS_PATH) as f:
                _feature_norms = json.load(f)
        except (OSError, ValueError) as e:
            raise FeatureNormsError(
                f"Production feature norms file unavailable or invalid: "
                f"{_FEATURE_NORMS_PATH}"
            ) from e
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


def _assemble_stage1_vector(v, s, h, hceye_feature_names):
    """Return the authoritative ``[v8 | s5 | h6]`` Stage-1 boundary."""
    import numpy as np

    dtype = np.dtype(STAGE1_VECTOR_DTYPE)
    visual = np.asarray(v, dtype=dtype).reshape(-1)
    saliency = np.asarray(s, dtype=dtype).reshape(-1)
    hceye = np.asarray(h, dtype=dtype).reshape(-1)
    names = (
        list(FEATURE_KEYS)
        + list(STAGE1_SALIENCY_FEATURE_NAMES)
        + list(hceye_feature_names)
    )

    if visual.shape != (8,) or saliency.shape != (5,) or hceye.shape != (6,):
        raise ValueError(
            "Stage-1 vector blocks must have exact dimensions v8, s5, and h6"
        )
    if len(names) != STAGE1_VECTOR_DIMENSIONS:
        raise ValueError("Stage-1 feature-name contract must contain 19 names")
    if not all(np.isfinite(block).all() for block in (visual, saliency, hceye)):
        raise Stage1VectorUnavailableError(
            STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
        )

    vector = np.concatenate([visual, saliency, hceye]).astype(dtype, copy=False)
    if vector.shape != (STAGE1_VECTOR_DIMENSIONS,):
        raise ValueError("Stage-1 feature vector must contain exactly 19 values")
    if not np.isfinite(vector).all():
        raise Stage1VectorUnavailableError(
            STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
        )
    return vector, names


def _validated_layout_score_inputs(measurement):
    """Return finite measured layout inputs or fail before score assembly."""
    import math

    if measurement is None:
        raise _layout_ocr_unavailable()

    source = getattr(measurement, "text_density_source", None)
    if source in (None, "", "fallback_neutral", "disabled"):
        raise _layout_ocr_unavailable()

    values = (
        ("whitespace_ratio", getattr(measurement, "whitespace_ratio", None)),
        ("text_density", getattr(measurement, "text_density", None)),
    )
    validated = []
    for _name, value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise _layout_ocr_unavailable() from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise _layout_ocr_unavailable()
        validated.append(numeric)
    return validated[0], validated[1], source


def _layout_ocr_unavailable():
    """Build the one client-safe P4 failure used by every layout/OCR guard."""
    return ScoreInputUnavailableError(
        LAYOUT_OCR_ERROR_CODE, LAYOUT_OCR_ERROR_MESSAGE
    )


def _get_saliency_model():
    """Lazy-load the UMSI++ saliency model (avoids TF startup penalty on every request)."""
    global _saliency_model
    if _saliency_model is None:
        weights = Path(__file__).parent.parent / "saliency" / "weights" / "model_weights" / "saliency_models" / "UMSI++" / "umsi++.hdf5"
        # P6 fail-closed checkpoint gate: production must never load arbitrary
        # same-shaped HDF5 bytes from the expected path. TensorFlow/Keras is
        # imported only after exact filename, byte-size and SHA-256 identity.
        verify_umsi_checkpoint(weights)
        from saliency.umsi_model import UMSIPlus
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


def _validate_uploaded_image_bytes(image_bytes: bytes):
    """Decode one upload before persistence or any score-bearing analysis.

    Extension checks alone cannot establish that the payload is an image.
    OpenCV may either return ``None`` or raise for corrupt/truncated inputs; both
    cases map to the same fixed client-safe error contract. The decoded array is
    returned for direct unit verification, while production stages continue to
    consume the persisted file exactly as before.
    """
    import cv2
    import numpy as np

    # Inspect header dimensions before OpenCV allocates the full decoded array.
    # This is the first resource guard; the decoded shape is checked again
    # below so a malformed header cannot bypass the production limit.
    _inspect_encoded_image_dimensions(image_bytes)

    try:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = (
            cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if encoded.size
            else None
        )
    except Exception as exc:
        raise InvalidImageUploadError(
            INVALID_IMAGE_ERROR_CODE, INVALID_IMAGE_ERROR_MESSAGE
        ) from exc

    if (
        decoded is None
        or decoded.size == 0
        or decoded.ndim != 3
        or decoded.shape[2] != 3
    ):
        raise InvalidImageUploadError(
            INVALID_IMAGE_ERROR_CODE, INVALID_IMAGE_ERROR_MESSAGE
        )
    _validate_image_dimensions(decoded.shape[1], decoded.shape[0])
    return decoded


def _validate_image_dimensions(width: int, height: int) -> int:
    """Return pixel count or reject an image outside the analysis envelope."""
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidImageUploadError(
            INVALID_IMAGE_ERROR_CODE, INVALID_IMAGE_ERROR_MESSAGE
        ) from exc

    if width < 1 or height < 1:
        raise InvalidImageUploadError(
            INVALID_IMAGE_ERROR_CODE, INVALID_IMAGE_ERROR_MESSAGE
        )

    pixels = width * height
    short_side = min(width, height)
    long_side = max(width, height)
    aspect_ratio = long_side / short_side
    if (
        short_side < MIN_CANONICAL_INPUT_SHORT_SIDE
        or long_side < MIN_CANONICAL_INPUT_LONG_SIDE
        or width > MAX_IMAGE_WIDTH
        or height > MAX_IMAGE_HEIGHT
        or pixels > MAX_IMAGE_PIXELS
        or pixels * 3 > MAX_DECODED_BYTES_PER_IMAGE
        or aspect_ratio > MAX_IMAGE_ASPECT_RATIO
    ):
        raise InvalidImageUploadError(
            IMAGE_RESOURCE_LIMIT_ERROR_CODE,
            IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE,
        )
    return pixels


def _inspect_encoded_image_dimensions(image_bytes: bytes) -> tuple:
    """Read image header dimensions without decoding full pixel storage."""
    import io
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            width, height = probe.size
    except Image.DecompressionBombError as exc:
        raise InvalidImageUploadError(
            IMAGE_RESOURCE_LIMIT_ERROR_CODE,
            IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE,
        ) from exc
    except Exception as exc:
        raise InvalidImageUploadError(
            INVALID_IMAGE_ERROR_CODE, INVALID_IMAGE_ERROR_MESSAGE
        ) from exc

    _validate_image_dimensions(width, height)
    return int(width), int(height)


def _persist_uploaded_image_bytes(extension: str, image_bytes: bytes) -> Path:
    """Persist an upload under a generated name with fail-safe cleanup.

    The original client filename is deliberately not accepted by this helper.
    Every caller has already allowlisted ``extension``; UUID-only storage makes
    traversal, absolute-path and nested-directory filename syntax irrelevant.
    """
    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    try:
        filepath.write_bytes(image_bytes)
    except OSError as exc:
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            # The fixed public error remains safe even if cleanup itself fails.
            pass
        raise UploadStorageUnavailableError(
            UPLOAD_STORAGE_ERROR_CODE, UPLOAD_STORAGE_ERROR_MESSAGE
        ) from exc
    return filepath


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


def _build_region_target_element(
    img,
    elements,
    region_x,
    region_y,
    region_w,
    region_h,
    screen_width_cm,
    screen_height_cm,
    viewing_distance_cm,
):
    """Turn a user-drawn selection box into a synthetic target element.

    VAS-style interaction: the user drags a rectangle around the area they are
    searching for. That rectangle IS the target, independently of whether the
    geometric detector happened to find an element there. We build a full
    element dict from the drawn region -- measuring its dominant colour, WCAG
    contrast and angular size directly from the image ROI, exactly like
    ``detect_elements`` does -- append it to the detected elements as the
    target, and return the extended element list plus the target index. The
    previously detected elements stay in the list as distractors for the
    visual-search simulation.

    Returns ``(elements_with_target, target_idx)``. A new list is returned so
    the caller's original detection is left untouched. ``target_idx`` is
    ``None`` if the region is degenerate (outside the image).
    """
    import numpy as np
    from cognitive.element_detector import (
        compute_contrast_ratio,
        _dominant_color_hsv,
        _hsv_to_category,
        WCAG_AA_LARGE,
    )

    h_img, w_img = img.shape[:2]

    # Clamp the drawn box to the image bounds and enforce a minimal size so a
    # tiny drag still yields a usable region.
    x = int(max(0, min(region_x, w_img - 1)))
    y = int(max(0, min(region_y, h_img - 1)))
    w = int(max(1, min(region_w, w_img - x)))
    h = int(max(1, min(region_h, h_img - y)))

    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return list(elements), None

    dom_hsv = _dominant_color_hsv(roi)
    color_cat = _hsv_to_category(dom_hsv)
    contrast_ratio = compute_contrast_ratio(roi)

    # Angular size from the region diagonal, matching element_detector's formula.
    px_per_cm_x = w_img / screen_width_cm
    px_per_cm_y = h_img / screen_height_cm
    diag_px = float(np.sqrt(w ** 2 + h ** 2))
    diag_cm = diag_px / ((px_per_cm_x + px_per_cm_y) / 2.0)
    angular_size_deg = float(
        np.degrees(2 * np.arctan(diag_cm / (2 * viewing_distance_cm)))
    )

    # Give the synthetic target a fresh id after the detected elements.
    existing_ids = [e.get("id", -1) for e in elements]
    new_id = (max(existing_ids) + 1) if existing_ids else 0

    target_elem = {
        "id": int(new_id),
        "bbox": (x, y, w, h),
        "center": (float(x + w / 2.0), float(y + h / 2.0)),
        "area": int(w * h),
        "dominant_color_hsv": (
            float(dom_hsv[0]), float(dom_hsv[1]), float(dom_hsv[2])
        ),
        "color_category": color_cat,
        "angular_size": angular_size_deg,
        "contrast_ratio": float(contrast_ratio),
        "wcag_aa_pass": bool(contrast_ratio >= WCAG_AA_LARGE),
        # Marks this as the user-drawn region rather than a detected element.
        "is_user_region": True,
    }

    extended = list(elements) + [target_elem]
    return extended, len(extended) - 1


def _predict_saliency_cached(image_hash, image_path):
    """Run UMSI++ saliency prediction with an LRU hash cache.

    Returns ``(heatmap, classif, cache_hit)`` where ``cache_hit`` is True when
    the result was served from cache. Caching avoids repeated (expensive)
    TensorFlow inference for identical images.
    """
    cache_key = f"{image_hash}:{saliency_cache_identity()}"
    cached = _saliency_cache.get(cache_key)
    if cached is not None:
        _saliency_cache.move_to_end(cache_key)  # mark as most-recently-used
        return cached["heatmap"], cached["classif"], True

    model = _get_saliency_model()
    heatmap, classif = model.predict_saliency(str(image_path), return_classif=True)
    _saliency_cache[cache_key] = {"heatmap": heatmap, "classif": classif}
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
    # The identity digest includes the extractor bytes, canonical contract,
    # runtime freeze and reference-pack/norm identity. A changed analysis
    # implementation can therefore never reuse an earlier cached result.
    cache_key = (
        f"{image_hash}:{CANONICAL_ANALYSIS_VERSION}:{visual_cache_identity()}"
    )
    cached = _visual_cache.get(cache_key)
    if cached is not None:
        _visual_cache.move_to_end(cache_key)  # mark as most-recently-used
        return cached, True

    results = compute_complexity_vector(str(image_path))
    _visual_cache[cache_key] = results
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

class _StrictJSONProvider(DefaultJSONProvider):
    """Reject non-standard JSON NaN/Infinity tokens on every API surface."""

    def dumps(self, obj, **kwargs):
        kwargs["allow_nan"] = False
        return super().dumps(obj, **kwargs)


app = Flask(__name__, static_folder="ui", static_url_path="")
app.json = _StrictJSONProvider(app)

# --- Resource limits (guard against accidental or malicious exhaustion) ------
# Maximum accepted request body size. Flask rejects larger uploads with 413
# before reading them into memory. Override with the MAX_UPLOAD_MB env var.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Decoded-image envelope. The request-body limit alone is insufficient because
# a small compressed PNG can expand into tens or hundreds of megabytes. The
# defaults accept ordinary 4K/5K screenshots and the thesis fixtures while
# rejecting disproportionate dimensions before full OpenCV decoding.
MAX_IMAGE_WIDTH = int(os.environ.get("MAX_IMAGE_WIDTH", "8192"))
MAX_IMAGE_HEIGHT = int(os.environ.get("MAX_IMAGE_HEIGHT", "8192"))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "16000000"))
MAX_DECODED_BYTES_PER_IMAGE = int(
    os.environ.get("MAX_DECODED_BYTES_PER_IMAGE", "48000000")
)
MAX_IMAGE_ASPECT_RATIO = float(os.environ.get("MAX_IMAGE_ASPECT_RATIO", "20"))
# Maximum number of screens accepted in one screen-set request (several files
# or animated-GIF frames). Larger sets are rejected rather than processed.
MAX_SCREENS = int(os.environ.get("MAX_SCREENS", "60"))
MAX_SCREEN_SET_PIXELS = int(
    os.environ.get("MAX_SCREEN_SET_PIXELS", "32000000")
)
MAX_SCREEN_SET_DECODED_BYTES = int(
    os.environ.get("MAX_SCREEN_SET_DECODED_BYTES", "96000000")
)

# Maximum length of an exposure / total-uses schedule list.
MAX_SCHEDULE_LEN = 50

# Bounds for the Monte-Carlo trial count exposed via ?n_simulations=.
MIN_SIMULATIONS = 1
MAX_SIMULATIONS = 500

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.errorhandler(413)
def _too_large(_err):
    """Return a clean JSON 413 instead of an HTML error page for oversized uploads."""
    return jsonify({
        "error": f"Upload too large (limit {MAX_UPLOAD_MB} MB)."
    }), 413


@app.errorhandler(UploadStorageUnavailableError)
def _upload_storage_unavailable(err):
    """Return stable JSON when a validated upload cannot be persisted."""
    app.logger.exception("Validated upload could not be persisted")
    return _fail_closed_error(err.code, err.message, status=500)


@app.errorhandler(InvalidImageUploadError)
def _invalid_image_upload(err):
    """Return one structured 400 contract for invalid or excessive images."""
    return _fail_closed_error(err.code, err.message, status=400)


def _clamp_simulations(req):
    """Read ?n_simulations and clamp it to [MIN_SIMULATIONS, MAX_SIMULATIONS].

    ``type=int`` yields None for non-numeric input (e.g. ?n_simulations=abc);
    that and any out-of-range value fall back into the valid band so the Monte
    Carlo loop can never run zero (which would yield an empty result / NaN) or an
    unbounded number of trials.
    """
    val = req.args.get("n_simulations", 100, type=int)
    if val is None:
        val = 100
    return int(max(MIN_SIMULATIONS, min(val, MAX_SIMULATIONS)))


def _server_error(exc):
    """Log the full exception server-side and return a generic JSON 500.

    Prevents leaking internal details (local paths, dependency versions, model
    internals, stack traces) to the client while keeping the real cause in the
    server log for debugging. Use for unexpected faults only; expected input
    problems should still return a specific 4xx.
    """
    import traceback
    print(f"[error] {type(exc).__name__}: {exc}")
    traceback.print_exc()
    return jsonify({"error": "Internal server error"}), 500


def _too_small_error(exc):
    """Return a documented JSON 400 for an image that is too small to analyse.

    Unlike ``_server_error`` this is an EXPECTED, client-correctable condition
    (the upload is degenerate / below the minimum analysable size), so the
    message is safe to surface to the client and the status code is 400.
    """
    return jsonify({"error": str(exc)}), 400


def _fail_closed_error(code: str, message: str, status: int = 503):
    """Structured, visible failure for a request that could not produce a
    complete, score-bearing analysis (e.g. saliency model/weights/inference
    unavailable, or the production feature-norms reference invalid).

    The response never carries a ``cognitive_load_score`` or other field that
    could be mistaken for a successfully computed result — ``analysis_complete``
    is explicitly false and the failure is identified by a stable machine
    -readable ``error.code`` rather than only free-text.
    """
    return jsonify({
        "analysis_complete": False,
        "error": {"code": code, "message": message},
    }), status


def _requested_saliency_failure(route_name: str):
    """Fail closed when an auxiliary diagnostic explicitly requested saliency.

    Feature-only execution is a separate, caller-selected contract. A failed
    requested saliency stage must never be converted into an HTTP 200 result.
    """
    app.logger.exception("Requested saliency failed on %s", route_name)
    return _fail_closed_error(
        AUXILIARY_SALIENCY_ERROR_CODE,
        AUXILIARY_SALIENCY_ERROR_MESSAGE,
        status=503,
    )


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
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Hash the upload first so identical images reuse cached feature results.
    image_hash, image_bytes = _hash_upload(file)
    _validate_uploaded_image_bytes(image_bytes)
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

    try:
        # Copy the cached dict before mutating it so the cache stays pristine.
        results, cache_hit = _compute_visual_cached(image_hash, filepath)
        results = dict(results)
        results["filename"] = file.filename
        results["visual_cache_hit"] = cache_hit
        return jsonify(results)
    except ImageTooSmallError as e:
        return _too_small_error(e)
    except Exception as e:
        return _server_error(e)
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@app.route("/api/features", methods=["GET"])
def features_info():
    """Return bounded metadata about the eight visual feature proxies."""
    features = [
        {
            "key": "shannon_entropy",
            "name": "Shannon Entropy",
            "description": "Global image-information density under this metric. Higher values indicate more varied pixel information; any attention interpretation is an unvalidated project hypothesis.",
            "range": "[0, 8]",
            "reference": "Shannon (1948)"
        },
        {
            "key": "edge_density",
            "name": "Edge Density",
            "description": "Proportion of pixels classified as edges. It is a structural-complexity proxy; a link to human parsing effort is an unvalidated project hypothesis.",
            "range": "[0, 1]",
            "reference": "Canny edge detection (AIM m4)"
        },
        {
            "key": "feature_congestion",
            "name": "Feature Congestion",
            "description": "Multi-scale clutter proxy combining color covariance, contrast variance, and orientation energy. Higher values mean more congestion under this metric, not validated human difficulty.",
            "range": "[0, ∞)",
            "reference": "Rosenholtz et al. (2007) — AIM m8"
        },
        {
            "key": "subband_entropy",
            "name": "Subband Entropy",
            "description": "Redundancy-based clutter proxy via steerable-pyramid decomposition. Higher values indicate less predictable spatial-frequency content under this metric.",
            "range": "[0, ∞)",
            "reference": "Rosenholtz et al. (2007) — AIM m7"
        },
        {
            "key": "layout_symmetry",
            "name": "Layout Symmetry",
            "description": "Degree of axial balance (vertical + horizontal). Higher values mean more symmetry under this metric; reduced visual search is only an unvalidated design hypothesis.",
            "range": "[0, 1]",
            "reference": "Miniukovich & De Angeli (2015)"
        },
        {
            "key": "chromatic_coherence",
            "name": "Chromatic Coherence",
            "description": "Color-palette fragmentation proxy combining luminance variance, colorfulness, and hue/saturation spread. Higher values mean more fragmentation under this metric.",
            "range": "[0, 1]",
            "reference": "Hasler & Süsstrunk (2003)"
        },
        {
            "key": "visual_hierarchy",
            "name": "Visual Hierarchy",
            "description": "Strength of layered visual structure from contrast gradients and size dominance. Higher values mean clearer hierarchy under this metric; reduced search effort is an unvalidated design hypothesis.",
            "range": "[0, 1]",
            "reference": "Tuch et al. (2009)"
        },
        {
            "key": "interactive_element_density",
            "name": "Interactive Element Density",
            "description": "Estimated count of control-like contours per area. Higher values mean more detected candidates under this custom proxy; decisional-load implications are unvalidated.",
            "range": "[0, ∞)",
            "reference": "Custom (contour-based)"
        },
    ]
    return jsonify({
        "validated_behavioral_prediction": False,
        "claim_boundary": (
            "These are project-specific image-feature measurements and proxy "
            "interpretations, not validated predictions of attention, visual "
            "search, decisional load, or other human behavior."
        ),
        "features": features,
    })


@app.route("/api/saliency", methods=["POST"])
def saliency():
    """Predict a saliency heatmap and numeric features for an uploaded image.

    The model's six-value auxiliary head remains an internal numeric output:
    its index-to-label order is not verified, so this public route must not
    attach class names, a predicted design type, or domain judgments to it.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    image_hash, image_bytes = _hash_upload(file)
    _validate_uploaded_image_bytes(image_bytes)
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

    try:
        import base64
        import cv2
        import numpy as np
        from saliency.saliency_features import extract_saliency_features

        heatmap, _classif, cache_hit = _predict_saliency_cached(image_hash, filepath)

        # Extract saliency features
        features = extract_saliency_features(heatmap)

        # Create colored heatmap for visualization (base64 PNG)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        _, buf = cv2.imencode(".png", heatmap_colored)
        heatmap_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "filename": file.filename,
            "features": features,
            "heatmap_png_base64": heatmap_b64,
            "saliency_cache_hit": cache_hit,
        })
    except Exception as e:
        return _server_error(e)
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/search-time", methods=["POST"])
def search_time():
    """
    Simulate visual search time per UI element using the Jokinen 2020 model.

    This is a model-based diagnostic, not observed fixation or timing data:
      - Detects UI elements from the uploaded screenshot
      - Computes UMSI++ saliency per element (deep bottom-up signal)
      - Runs Monte Carlo simulation of novice visual search (EMMA + feature guidance)
      - Returns model-estimated search time and fixation count per element

    Reference:
        Jokinen, J.P.P. et al. (2020). Adaptive feature guidance: Modelling
        visual search with graphical layouts. IJHCS, 136, 102376.

    Request:
        POST multipart/form-data with field "image"
        (PNG/JPG/JPEG/BMP/TIFF screenshot)
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
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Parse optional query parameters
    n_simulations = _clamp_simulations(request)
    use_saliency = request.args.get("use_saliency", "true").lower() != "false"

    image_hash, image_bytes = _hash_upload(file)
    _validate_uploaded_image_bytes(image_bytes)
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

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
                return _requested_saliency_failure("/api/search-time")

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
        results["analysis_complete"] = True
        results["analysis_mode"] = (
            "saliency_augmented" if use_saliency else "feature_only_explicit"
        )
        results["saliency_requested"] = use_saliency
        results["saliency_used"] = saliency_map is not None

        return jsonify(results)

    except Exception as e:
        return _server_error(e)
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/scanpath-to-target", methods=["POST"])
def scanpath_to_target():
    """
    Simulate a visual-search path toward a user-selected target element.

    The user selects the area they are searching for on the screenshot. The
    primary (VAS-style) mode is a DRAWN REGION box: the drawn rectangle becomes
    a synthetic target element (its colour and contrast measured from the image
    ROI), so the target is exactly what the user selected -- independent of what
    the geometric detector happened to find. The other detected elements stay as
    distractors. A single click position or an explicit element id are also
    accepted (they select an existing detected element).

    This endpoint returns the model-generated sequence produced by the Jokinen
    2020 Adaptive Feature Guidance implementation, starting from the screen
    center. It is a task-driven simulation, not an observed eye-tracking
    sequence, a free-viewing scanpath, or validated user-performance evidence.

    Reference:
        Jokinen, J.P.P. et al. (2020). Adaptive feature guidance: Modelling
        visual search with graphical layouts. IJHCS, 136, 102376.

    Request:
        POST multipart/form-data with field "image"
        (PNG/JPG/JPEG/BMP/TIFF screenshot).
        Target selection (one of):
            - target_x, target_y, target_w, target_h (float, query params):
              VAS-style drawn region in ORIGINAL-image pixels (top-left x/y and
              size). The region becomes a synthetic target element and the other
              detected elements are distractors. This is the primary mode.
            - target_x, target_y (float, query params): a single click position
              in ORIGINAL-image pixels. The element whose bounding box contains
              the point is used; otherwise the nearest-center element is chosen.
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
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Target selection params. Primary (VAS-style) mode is a drawn region box:
    # target_x, target_y = top-left in original-image pixels, target_w/target_h =
    # its size. A single click point (target_x, target_y only) and an explicit
    # target_id remain supported for backwards compatibility.
    target_x = request.args.get("target_x", default=None, type=float)
    target_y = request.args.get("target_y", default=None, type=float)
    target_w = request.args.get("target_w", default=None, type=float)
    target_h = request.args.get("target_h", default=None, type=float)
    target_id = request.args.get("target_id", default=None, type=int)
    has_region = (
        target_x is not None and target_y is not None
        and target_w is not None and target_w > 0
        and target_h is not None and target_h > 0
    )
    has_point = target_x is not None and target_y is not None
    if target_id is None and not has_point:
        return jsonify({
            "error": "Provide a target region (target_x, target_y, target_w, "
                     "target_h), a click point (target_x, target_y), or target_id"
        }), 400

    n_simulations = _clamp_simulations(request)
    use_saliency = request.args.get("use_saliency", "true").lower() != "false"

    # Physical display geometry (affects search timing via angular size).
    (screen_w_cm, screen_h_cm,
     viewing_cm, display_preset_meta) = _resolve_display_preset(request)

    image_hash, image_bytes = _hash_upload(file)
    _validate_uploaded_image_bytes(image_bytes)
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

    try:
        import cv2
        from cognitive.element_detector import detect_elements
        from cognitive.jokinen_model import (
            JokinenSearchModel,
            JokinenParams,
            compute_glance_metrics,
        )

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

        # Resolve the target. VAS-style: a drawn region becomes a synthetic
        # target element (measured from the image ROI) so the target is exactly
        # what the user selected, independent of the geometric detector; the
        # other detected elements stay as distractors. A click point or an
        # explicit id falls back to selecting an existing detected element.
        if has_region:
            elements, target_idx = _build_region_target_element(
                img, elements,
                target_x, target_y, target_w, target_h,
                screen_width_cm=screen_w_cm,
                screen_height_cm=screen_h_cm,
                viewing_distance_cm=viewing_cm,
            )
        else:
            target_idx = _resolve_target_index(
                elements, target_id, target_x, target_y
            )
        if target_idx is None:
            return jsonify({"error": "Could not resolve target element"}), 400

        # Saliency (optional) so the scanpath matches the search-time pipeline.
        saliency_map = None
        if use_saliency:
            try:
                saliency_map, _, _ = _predict_saliency_cached(image_hash, filepath)
            except Exception:
                return _requested_saliency_failure("/api/scanpath-to-target")

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
            screen_width_cm=screen_w_cm,
            screen_height_cm=screen_h_cm,
            viewing_distance_cm=viewing_cm,
        )

        # Glance-based automotive metrics (NHTSA 2013 / ISO 15008): split the
        # predicted search scanpath into eyes-off-road glances and check it
        # against the single-glance (<=2 s) and cumulative (<=12 s) limits. Only
        # meaningful for in-vehicle displays, so we skip it for the desktop
        # preset (where the guidelines do not apply).
        glance_metrics = None
        if display_preset_meta.get("key") != "desktop":
            fixations = (scanpath or {}).get("fixations", [])
            if fixations:
                glance_metrics = compute_glance_metrics(fixations)


        # already estimates a search cost PER element; until now those costs were
        # only averaged into a layout-wide mean. Selecting a target lets us use
        # that element's specific search cost instead of the average, so the load
        # reflects a concrete top-down task ("find THIS button") rather than the
        # layout in general. We express it as a SIGNED modifier (in score points)
        # relative to the layout's mean search time: a target that is harder to
        # find than average raises the load, an easier one lowers it.
        # SELECTED-TARGET SEARCH DIFFICULTY (methodologically separate from the
        # layout-wide complexity index). The Jokinen search model already
        # estimates a search cost PER element. Selecting a target lets us report
        # that element's ABSOLUTE predicted search time (seconds) plus how it
        # compares to the layout mean — WITHOUT folding it into the layout score
        # or inventing a new 0-100 number. This is a search-difficulty construct,
        # not a cognitive-load / mental-effort / safety measure.
        selected_target = None
        target_load = None  # legacy alias, kept for backward compatibility
        search_res = jokinen.predict_search_times(
            elements=elements,
            saliency_map=saliency_map,
            image_shape=img.shape[:2],
            screen_width_cm=screen_w_cm,
            screen_height_cm=screen_h_cm,
            viewing_distance_cm=viewing_cm,
        )
        mean_time = float(search_res.get("mean_search_time_s", 0.0) or 0.0)
        per_elem = search_res.get("per_element", [])
        target_time = None
        target_time_std = None
        if 0 <= target_idx < len(per_elem):
            target_time = float(per_elem[target_idx].get("search_time_s", 0.0))
            target_time_std = float(per_elem[target_idx].get("search_time_std_s", 0.0))

        # Whether the model actually reached the target in the representative
        # scanpath (vs. running out of fixations). The last fixation is the
        # target when the search succeeded.
        fixations = (scanpath or {}).get("fixations", [])
        search_success = bool(fixations and fixations[-1].get("is_target"))

        if target_time is not None and mean_time > 0:
            deviation = (target_time - mean_time) / mean_time
            deviation_pct = round(deviation * 100.0, 1)
            # Plain-language, relative label. Deliberately about SEARCH time
            # vs the layout average, never about cognitive load / effort.
            if deviation_pct <= -15.0:
                relative_difficulty = "easier than layout average"
            elif deviation_pct >= 15.0:
                relative_difficulty = "harder than layout average"
            else:
                relative_difficulty = "around layout average"

            selected_target = {
                "target_id": elements[target_idx].get("id"),
                "search_time_s": round(target_time, 4),
                # Monte Carlo uncertainty (1 SD over the simulation trials).
                "search_time_std_s": round(target_time_std, 4)
                if target_time_std is not None else None,
                "layout_mean_search_time_s": round(mean_time, 4),
                "relative_difficulty": relative_difficulty,
                "deviation_pct": deviation_pct,
                "search_success": search_success,
            }

            # Legacy diagnostic: the old signed score-point modifier. Kept ONLY
            # as an explicitly experimental field for backward compatibility; it
            # must NOT be combined with the layout score in new UI/exports.
            POINTS_PER_UNIT = 25.0
            MODIFIER_CAP = 15.0
            search_load_modifier = float(
                max(-MODIFIER_CAP, min(MODIFIER_CAP, POINTS_PER_UNIT * deviation))
            )
            target_load = {
                "mean_search_time_s": round(mean_time, 4),
                "target_search_time_s": round(target_time, 4),
                "target_search_time_std_s": round(target_time_std, 4)
                if target_time_std is not None else None,
                "relative_difficulty": round(target_time / mean_time, 3),
                "deviation_pct": deviation_pct,
                # Experimental diagnostic only — see selected_target for the
                # canonical, non-cognitive-load search-difficulty result.
                "experimental_search_load_modifier": round(search_load_modifier, 2),
            }

        target_elem = elements[target_idx]
        return jsonify({
            "analysis_complete": True,
            "analysis_mode": (
                "saliency_augmented" if use_saliency else "feature_only_explicit"
            ),
            "filename": file.filename,
            "n_elements": len(elements),
            "target_id": target_elem.get("id"),
            "target_center": list(target_elem.get("center", [])),
            "target_bbox": list(target_elem.get("bbox", [])),
            "saliency_requested": use_saliency,
            "saliency_used": saliency_map is not None,
            "scanpath": scanpath,
            # Canonical, methodologically-separate search-difficulty result.
            "selected_target": selected_target,
            # Legacy alias (experimental diagnostic modifier inside).
            "target_load": target_load,
            "display_preset": display_preset_meta,
            "glance_metrics": glance_metrics,
        })

    except Exception as e:
        return _server_error(e)
    finally:
        if filepath.exists():
            filepath.unlink()


# Physical display presets for the Jokinen search model. Pixel-based features
# are display-independent, but visual search timing depends on PHYSICAL angular
# size (a 12 cm IVI viewed at 75 cm subtends a different angle than a 37 cm
# desktop monitor at 60 cm). Each preset gives the active screen area in cm and
# a typical viewing distance in cm. Values are nominal 16:9 active areas for the
# named diagonal; automotive viewing distances follow common HMI design guidance
# (driver eye to display). "desktop" is the model's default and matches the
# defaults in JokinenSearchModel.predict_search_times.
DISPLAY_PRESETS = {
    # General-purpose (non-automotive) displays. The validation is run on
    # normal GUIs, not domain-specific automotive HMIs, so these are the
    # presets surfaced in the UI. The automotive presets below remain only so
    # older saved requests / links still resolve; they are hidden in the UI.
    "phone":     {"label": "Phone ~6.1\"",              "width_cm": 7.0,  "height_cm": 15.0, "viewing_distance_cm": 30.0},
    "laptop":    {"label": "Laptop 14\"",               "width_cm": 30.9, "height_cm": 17.4, "viewing_distance_cm": 50.0},
    "desktop":   {"label": "Desktop monitor 17\"",      "width_cm": 37.0, "height_cm": 23.0, "viewing_distance_cm": 60.0},
    # Legacy automotive presets (kept for backward compatibility, not shown).
    "ivi_8":     {"label": "IVI 8\"",                    "width_cm": 17.7, "height_cm": 10.0, "viewing_distance_cm": 70.0},
    "ivi_10":    {"label": "IVI 10.25\"",                "width_cm": 22.7, "height_cm": 12.7, "viewing_distance_cm": 72.0},
    "ivi_12":    {"label": "IVI 12.3\"",                 "width_cm": 27.2, "height_cm": 15.3, "viewing_distance_cm": 75.0},
    "ivi_15":    {"label": "IVI 15\"",                   "width_cm": 33.2, "height_cm": 18.7, "viewing_distance_cm": 75.0},
    "cluster":   {"label": "Instrument cluster 12.3\"",  "width_cm": 27.2, "height_cm": 15.3, "viewing_distance_cm": 80.0},
    "hud":       {"label": "Head-up display",            "width_cm": 30.0, "height_cm": 12.0, "viewing_distance_cm": 220.0},
}

STAGE2_V1_UNSUPPORTED_FIELDS = (
    "target_specificity",
    "search_mode",
    "profile_preset",
    "use_trained_model",
)
STAGE2_V1_ALLOWED_FORM_FIELDS = {
    "task_type",
    "time_pressure",
    "include_jokinen_diagnostic",
    "display_preset",
}
STAGE2_V1_DISPLAY_PRESETS = {"phone", "laptop", "desktop"}


def _request_value(req, name):
    """Return one unambiguous form/query value, preserving explicit blanks."""
    values = list(req.form.getlist(name)) + list(req.args.getlist(name))
    if len(values) > 1:
        raise ScenarioProxyInputError(
            f"{name} must be supplied at most once across form and query data"
        )
    return values[0] if values else None


def _plain_finite_tree(value, path="value"):
    """Return a JSON-safe copy while rejecting non-finite/unsupported leaves."""
    import math
    import numbers

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must contain only finite numeric values")
        if isinstance(value, numbers.Integral):
            return int(value)
        return float(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} must use string object keys")
        return {
            key: _plain_finite_tree(child, f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain_finite_tree(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    # NumPy arrays are legitimate internal containers, but must be converted
    # before Flask's strict JSON provider sees them.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain_finite_tree(tolist(), path)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _finite_number(value, path):
    """Require a real finite number (booleans and numeric strings are invalid)."""
    import math
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{path} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite real number")
    return number


def _finite_integer(value, path):
    """Require a finite integral value without silently truncating decimals."""
    number = _finite_number(value, path)
    if not number.is_integer():
        raise ValueError(f"{path} must be an integer")
    return int(number)


def _finite_sequence(value, length, path):
    """Require an exact-length finite numeric sequence and return plain floats."""
    if isinstance(value, (str, bytes, dict)):
        raise TypeError(f"{path} must be a numeric sequence of length {length}")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TypeError(f"{path} must be a numeric sequence of length {length}")
    return [
        _finite_number(child, f"{path}[{index}]")
        for index, child in enumerate(value)
    ]


def _validated_native_elements(elements):
    """Validate and sanitize non-score-bearing native detector output."""
    safe = _plain_finite_tree(elements, "native_elements")
    if not isinstance(safe, list):
        raise TypeError("native_elements must be a list")
    for index, element in enumerate(safe):
        path = f"native_elements[{index}]"
        if not isinstance(element, dict):
            raise TypeError(f"{path} must be an object")
        for required in ("id", "bbox", "center"):
            if required not in element:
                raise ValueError(f"{path}.{required} is required")
        element["id"] = _finite_integer(element["id"], f"{path}.id")
        element["bbox"] = _finite_sequence(element["bbox"], 4, f"{path}.bbox")
        element["center"] = _finite_sequence(
            element["center"], 2, f"{path}.center"
        )
        for optional in ("area", "angular_size", "contrast_ratio"):
            if optional in element:
                element[optional] = _finite_number(
                    element[optional], f"{path}.{optional}"
                )
        if "dominant_color_hsv" in element:
            element["dominant_color_hsv"] = _finite_sequence(
                element["dominant_color_hsv"],
                3,
                f"{path}.dominant_color_hsv",
            )
        if "color_category" in element and not isinstance(
            element["color_category"], str
        ):
            raise TypeError(f"{path}.color_category must be text")
        if "wcag_aa_pass" in element and not isinstance(
            element["wcag_aa_pass"], bool
        ):
            raise TypeError(f"{path}.wcag_aa_pass must be boolean")
    return safe


def _validated_jokinen_result(result):
    """Validate the complete optional Jokinen result schema before use."""
    safe = _plain_finite_tree(result, "jokinen_result")
    if not isinstance(safe, dict):
        raise TypeError("jokinen_result must be an object")
    for required in ("mean_search_time_s", "per_element"):
        if required not in safe:
            raise ValueError(f"jokinen_result.{required} is required")
    safe["mean_search_time_s"] = _finite_number(
        safe["mean_search_time_s"], "jokinen_result.mean_search_time_s"
    )
    per_element = safe["per_element"]
    if not isinstance(per_element, list):
        raise TypeError("jokinen_result.per_element must be a list")
    for index, element in enumerate(per_element):
        path = f"jokinen_result.per_element[{index}]"
        if not isinstance(element, dict):
            raise TypeError(f"{path} must be an object")
        for required in (
            "id",
            "search_time_s",
            "fixation_count",
            "bbox",
            "center",
        ):
            if required not in element:
                raise ValueError(f"{path}.{required} is required")
        element["id"] = _finite_integer(element["id"], f"{path}.id")
        for numeric in ("search_time_s", "fixation_count"):
            element[numeric] = _finite_number(
                element[numeric], f"{path}.{numeric}"
            )
        if "search_time_std_s" in element:
            element["search_time_std_s"] = _finite_number(
                element["search_time_std_s"], f"{path}.search_time_std_s"
            )
        element["bbox"] = _finite_sequence(element["bbox"], 4, f"{path}.bbox")
        element["center"] = _finite_sequence(
            element["center"], 2, f"{path}.center"
        )
        if "color_category" in element and not isinstance(
            element["color_category"], str
        ):
            raise TypeError(f"{path}.color_category must be text")
    return safe


def _stage2_v1_context(req):
    """Validate the exact public Stage-2 v1 context boundary."""
    unknown = sorted(
        (set(req.form) | set(req.args)) - STAGE2_V1_ALLOWED_FORM_FIELDS
    )
    # Keep an explicit legacy list so the error remains stable and auditable,
    # while also rejecting any future undeclared field rather than ignoring it.
    unsupported = [
        name
        for name in STAGE2_V1_UNSUPPORTED_FIELDS
        if _request_value(req, name) not in (None, "")
    ]
    rejected = sorted(set(unknown) | set(unsupported))
    if rejected:
        raise ScenarioProxyInputError(
            "Stage-2 v1 accepts only task_type and time_pressure plus the "
            "explicit optional Jokinen controls; unsupported fields were "
            f"supplied: {', '.join(rejected)}"
        )

    raw_task_type = _request_value(req, "task_type")
    raw_time_pressure = _request_value(req, "time_pressure")
    proxy = build_scenario_proxy(
        task_type=(
            "search" if raw_task_type is None else str(raw_task_type).strip()
        ),
        time_pressure=(
            "medium"
            if raw_time_pressure is None
            else str(raw_time_pressure).strip()
        ),
    )

    raw_jokinen = _request_value(req, "include_jokinen_diagnostic")
    if raw_jokinen is None:
        include_jokinen = False
    elif isinstance(raw_jokinen, bool):
        include_jokinen = raw_jokinen
    else:
        normalized = str(raw_jokinen).strip()
        if normalized in {"1", "true", "yes", "on"}:
            include_jokinen = True
        elif normalized in {"0", "false", "no", "off"}:
            include_jokinen = False
        else:
            raise ScenarioProxyInputError(
                "include_jokinen_diagnostic must be true or false"
            )

    raw_display = _request_value(req, "display_preset")
    if raw_display is None:
        display_key = "desktop"
    else:
        display_key = str(raw_display).strip()
        if not display_key:
            raise ScenarioProxyInputError(
                "display_preset must be one of phone, laptop, desktop"
            )
        if not include_jokinen:
            raise ScenarioProxyInputError(
                "display_preset is accepted only when the optional Jokinen "
                "diagnostic is requested"
            )
        if display_key not in STAGE2_V1_DISPLAY_PRESETS:
            raise ScenarioProxyInputError(
                "display_preset must be one of phone, laptop, desktop"
            )
    return proxy, include_jokinen, display_key


def _display_preset_geometry(key):
    """Return physical geometry for an already validated preset key."""
    if key not in DISPLAY_PRESETS:
        raise ValueError(f"Unknown display preset: {key}")
    preset = DISPLAY_PRESETS[key]
    meta = {
        "key": key,
        "label": preset["label"],
        "width_cm": preset["width_cm"],
        "height_cm": preset["height_cm"],
        "viewing_distance_cm": preset["viewing_distance_cm"],
    }
    return (
        preset["width_cm"],
        preset["height_cm"],
        preset["viewing_distance_cm"],
        meta,
    )


def _resolve_display_preset(req):
    """Resolve the physical display geometry from the request.

    Looks for a ``display_preset`` key (form or query). Falls back to the
    ``desktop`` preset (which matches the Jokinen model defaults) when the key
    is missing or unknown. Returns ``(width_cm, height_cm, viewing_distance_cm,
    preset_meta)`` where ``preset_meta`` is a dict suitable for the API response.
    """
    key = (req.form.get("display_preset")
           or req.args.get("display_preset")
           or "desktop").strip().lower()
    if key not in DISPLAY_PRESETS:
        key = "desktop"
    return _display_preset_geometry(key)


@app.route("/api/cognitive-load", methods=["POST"])
def cognitive_load():
    """
    Compute the exploratory project-specific layout proxy.

    ``/api/cognitive-load`` is retained as the legacy route name.  The returned
    construct is not a validated cognitive-load measurement.  HCEye-derived
    values are exposed only as explicitly named proxies, and the unverified
    UMSI classification-head label mapping is never exposed.

    Combines visual complexity (v∈ℝ⁸) + saliency (s∈ℝ⁵) + HCEye-derived
    proxy features (h∈ℝ⁶) into the task/profile-independent Stage-1 x19
    boundary. Stage 2 v1 attaches a deterministic, non-score-bearing scenario
    proxy from ``task_type`` and ``time_pressure`` only. Personality and ML
    inputs are rejected. The Jokinen diagnostic is separate and opt-in.

    Response JSON:
        {
            "filename": str,
            "visual_features": {...},         # v∈ℝ⁸
            "saliency_features": {...},       # s∈ℝ⁵  
            "hceye_proxy_features": {...},     # h∈ℝ⁶
            "stage2_scenario_proxy": {...},    # qualitative; no number
            "cross_signal_review": {...},      # non-score-bearing tri-state
            "jokinen_diagnostic": {...},       # separate, off by default
            "scientific_semantics": {...},
            "stage1_feature_vector": [...],   # [v8 | s5 | h6]
            "stage1_feature_names": [...],
            "stage1_vector_dtype": "float32",
            "vector_dimensions": 19
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    try:
        (
            scenario_proxy,
            include_jokinen_diagnostic,
            display_preset_key,
        ) = _stage2_v1_context(request)
    except ScenarioProxyInputError as exc:
        return _fail_closed_error(
            "stage2_scenario_invalid",
            str(exc),
            status=400,
        )

    # Hash once and reuse the same key for both the visual and saliency caches.
    image_hash, image_bytes = _hash_upload(file)
    try:
        _validate_uploaded_image_bytes(image_bytes)
    except InvalidImageUploadError as e:
        return _fail_closed_error(e.code, e.message, status=400)

    # The user-controlled filename remains response metadata only.
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

    try:
        import numpy as np
        from hceye.hceye_features import HCEyeFeatureExtractor

        # Physical display geometry is used only when the optional, separate
        # Jokinen diagnostic is requested; pixel features are unaffected.
        (screen_w_cm, screen_h_cm,
         viewing_cm, display_preset_meta) = _display_preset_geometry(
            display_preset_key
        )

        # Step 1: Visual complexity (v∈ℝ⁸)
        vis_results, visual_cache_hit = _compute_visual_cached(image_hash, filepath)
        vis_results = dict(vis_results)
        try:
            v = np.asarray(
                [vis_results.get(name) for name in FEATURE_KEYS],
                dtype=np.float32,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise Stage1VectorUnavailableError(
                STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
            ) from exc
        if v.shape != (8,) or not np.isfinite(v).all():
            raise Stage1VectorUnavailableError(
                STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
            )

        # Step 2: Saliency features (s∈ℝ⁵) — mandatory for x19
        s = None
        saliency_dict = {}
        saliency_overlay_b64 = None
        try:
            import base64, cv2
            from saliency.saliency_features import extract_saliency_features
            heatmap, _classif, cache_hit = _predict_saliency_cached(image_hash, filepath)
            saliency_dict = extract_saliency_features(heatmap)
            s = np.array(
                [saliency_dict[name] for name in STAGE1_SALIENCY_FEATURE_NAMES],
                dtype=np.float32,
            )
        except Exception as e:
            # Saliency is MANDATORY for a complete, score-bearing analysis.
            # Do not degrade to image-only features (s=None): that would let
            # the request still return HTTP 200 with a full-looking score
            # whose scientific meaning silently changed. Log the real cause
            # server-side, then surface a structured, visible failure instead.
            print(f"[Saliency] Saliency features unavailable: {e!r}")
            raise SaliencyUnavailableError(
                "saliency_unavailable",
                "Saliency computation failed; a complete analysis could not "
                "be produced.",
            ) from e

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

        # Step 2.5: Layout measurements for the HCEye stage — TWO explicit paths.
        #
        # ANALYSIS PATH (canonical, score-driving): whitespace_ratio and OCR-
        # derived text_density are computed on a single canonical analysis image
        # (long side 1280 px) with canonical element detection, so they share the
        # SAME analysis scale as the eight visual features, which standardises
        # their analysis scale and reduces measured resolution sensitivity. These
        # are the ONLY element-derived values that feed the layout
        # experimental_complexity_index.
        #
        # NATIVE PATH (interaction only): the Jokinen search model, target
        # selection, native overlays, native contrast diagnostics and the
        # detected_elements returned to the target selector use a SEPARATE native
        # element detection on the original image. Analysis elements are never
        # reused as native elements.
        import cv2
        native_img = cv2.imread(str(filepath))

        # --- Analysis path (canonical) -----------------------------------
        if native_img is None:
            raise _layout_ocr_unavailable()
        try:
            from canonical_layout import measure_canonical_layout
            analysis_measurement = measure_canonical_layout(native_img)
            whitespace_ratio, text_density, text_density_source = (
                _validated_layout_score_inputs(analysis_measurement)
            )
            # Readability boxes are on the canonical image; map ONLY their
            # display coordinates back to native for the UI. This mapped
            # report is display-only and never feeds the layout score.
            readability_report = analysis_measurement.readability_report_native()
        except ScoreInputUnavailableError:
            raise
        except Exception as exc:
            raise _layout_ocr_unavailable() from exc

        # --- Native path (Jokinen / interaction) -------------------------
        native_elements = []
        native_elements_available = False
        if native_img is not None:
            try:
                from cognitive.element_detector import detect_elements
                native_elements = _validated_native_elements(
                    detect_elements(native_img)
                )
                native_elements_available = True
            except Exception as e:
                print(f"[HCEye] Native element detection unavailable: {e!r}")

        # Step 3: project-specific HCEye-derived proxy features (h∈ℝ⁶).
        # Their interpretation is deliberately bounded by SCIENTIFIC_SEMANTICS;
        # these values are not validated screenshot-level cognitive load.
        # Live uploads are novel screenshots, so the source-study image lookup
        # is intentionally not loaded here. Offline HCEye reproduction scripts
        # can still opt into that lookup by passing both lookup_path and
        # image_name directly to HCEyeFeatureExtractor.
        extractor = HCEyeFeatureExtractor(
            feature_norms_path=str(_FEATURE_NORMS_PATH)
        )
        h = extractor.extract_features(
            vis_results,
            saliency_features=(saliency_dict or None),
            whitespace_ratio=whitespace_ratio,
            text_density=text_density,
        )
        proxy_names = extractor.get_feature_names()
        hceye_proxy_dict = dict(zip(proxy_names, h.tolist()))

        # Build the sole public Stage-1 boundary (v⁸ + s⁵ + h⁶ = ℝ¹⁹).
        # Saliency is mandatory on this score-bearing route. The preceding
        # stage either produced a complete s5 block or raised
        # SaliencyUnavailableError; retaining a zero-substitution branch here
        # would be a latent fail-open hazard if that earlier guard changed.
        if s is None:
            raise Stage1VectorUnavailableError(
                STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
            )
        base_vector, stage1_feature_names = _assemble_stage1_vector(
            v, s, h, proxy_names
        )

        stage1_score = float(h[5] * 100.0)
        if not np.isfinite(stage1_score):
            raise Stage1VectorUnavailableError(
                STAGE1_VECTOR_ERROR_CODE, STAGE1_VECTOR_ERROR_MESSAGE
            )

        # Optional, methodologically separate Jokinen diagnostic. It never
        # changes x19, the Stage-1 layout index or the Stage-2 scenario proxy.
        mean_search_time_s = None
        estimated_fixation_count = None
        search_feedback = None
        contrast_report = None
        jokinen_diagnostic = {
            "requested": include_jokinen_diagnostic,
            "status": "not_requested",
            "score_bearing": False,
            "methodologically_separate": True,
            "validated_behavioral_prediction": False,
            "claim_boundary": (
                "Optional Jokinen model diagnostic only; not measured search, "
                "eye tracking, or a validated behavioral prediction."
            ),
            "display_preset": display_preset_meta,
            "result": None,
        }
        if include_jokinen_diagnostic:
            try:
                import cv2
                from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

                # Reuse the native image and native element boxes from Step 2.5.
                # Only re-read/re-detect if that earlier step failed.
                if native_img is None:
                    native_img = cv2.imread(str(filepath))
                if native_img is None:
                    raise ValueError(
                        f"Cannot read image for Jokinen search model: {filepath}"
                    )
                if not native_elements_available:
                    raise ValueError("Native element data is unavailable")

                jokinen_model = JokinenSearchModel(JokinenParams())
                jresult = jokinen_model.predict_search_times(
                    elements=native_elements,
                    saliency_map=heatmap,
                    image_shape=native_img.shape[:2],
                    screen_width_cm=screen_w_cm,
                    screen_height_cm=screen_h_cm,
                    viewing_distance_cm=viewing_cm,
                )
                # The diagnostic is optional, so numerically invalid model
                # output must be contained here before it can reach either the
                # Cross-Signal Review or strict JSON serialization.
                jresult = _validated_jokinen_result(jresult)
                mean_search_time_s = float(jresult["mean_search_time_s"])
                per_elem = jresult.get("per_element", [])
                if per_elem:
                    estimated_fixation_count = float(
                        sum(e["fixation_count"] for e in per_elem) / len(per_elem)
                    )

                # Generative feedback (diagnosis -> design): turn the per-element
                # search costs into a ranked list of "bottleneck" elements so the
                # designer sees WHICH elements are hardest to find, not just an
                # aggregate score. The hardest elements are the ones whose
                # predicted search time is well above the layout mean.
                mean_t = mean_search_time_s if mean_search_time_s else 0.0
                ranked = sorted(
                    per_elem, key=lambda e: e["search_time_s"], reverse=True
                )
                bottlenecks = []
                for e in ranked[:5]:
                    t_e = float(e["search_time_s"])
                    deviation_pct = (
                        round((t_e - mean_t) / mean_t * 100.0, 1)
                        if mean_t > 0 else 0.0
                    )
                    bottlenecks.append({
                        "id": e["id"],
                        "search_time_s": round(t_e, 3),
                        "search_time_std_s": round(
                            float(e.get("search_time_std_s", 0.0)), 3
                        ),
                        "fixation_count": round(float(e["fixation_count"]), 1),
                        "bbox": e["bbox"],
                        "center": e["center"],
                        "color_category": e.get("color_category", "unknown"),
                        # Signed % vs the layout mean: positive = harder to find.
                        "deviation_pct": deviation_pct,
                    })
                search_feedback = {
                    "mean_search_time_s": round(mean_t, 3),
                    "n_elements": len(per_elem),
                    # Elements ranked hardest-to-find first (top 5).
                    "bottlenecks": bottlenecks,
                }

                # Display-only contrast summary attached to this optional
                # diagnostic; it is not part of the scenario proxy.
                if native_elements:
                    wcag_threshold = 3.0
                    ratios = [
                        float(e.get("contrast_ratio", 1.0))
                        for e in native_elements
                    ]
                    failing = sorted(
                        (
                            e for e in native_elements
                            if float(e.get("contrast_ratio", 1.0))
                            < wcag_threshold
                        ),
                        key=lambda e: float(e.get("contrast_ratio", 1.0)),
                    )
                    contrast_report = {
                        "wcag_threshold": wcag_threshold,
                        "n_elements": len(native_elements),
                        "n_pass": sum(1 for ratio in ratios if ratio >= wcag_threshold),
                        "n_fail": sum(1 for ratio in ratios if ratio < wcag_threshold),
                        "min_contrast_ratio": round(min(ratios), 2),
                        "mean_contrast_ratio": round(
                            float(sum(ratios) / len(ratios)), 2
                        ),
                        "low_contrast_elements": [
                            {
                                "id": e["id"],
                                "contrast_ratio": round(
                                    float(e.get("contrast_ratio", 1.0)), 2
                                ),
                                "bbox": e["bbox"],
                                "center": e["center"],
                                "color_category": e.get(
                                    "color_category", "unknown"
                                ),
                            }
                            for e in failing[:5]
                        ],
                    }

                diagnostic_result = {
                    "mean_search_time_s": mean_search_time_s,
                    "estimated_fixation_count": estimated_fixation_count,
                    "search_feedback": search_feedback,
                    "contrast_report": contrast_report,
                }
                # Validate derived arithmetic as well as raw model output so
                # overflow or conversion in feedback assembly cannot escape
                # the optional containment boundary.
                jokinen_diagnostic["result"] = _plain_finite_tree(
                    diagnostic_result, "jokinen_diagnostic.result"
                )
                jokinen_diagnostic["status"] = "complete"
            except Exception as e:
                print(f"[Jokinen] Optional diagnostic unavailable: {e!r}")
                jokinen_diagnostic["status"] = "unavailable"
                jokinen_diagnostic["result"] = None
                mean_search_time_s = None
                estimated_fixation_count = None
                search_feedback = None
                contrast_report = None

        saliency_spread = saliency_dict.get("saliency_dispersion") if saliency_dict else None
        cross_signal_review = run_cross_signal_review(
            saliency_spread=saliency_spread,
            estimated_fixation_count=estimated_fixation_count,
            mean_search_time_s=mean_search_time_s,
            layout_proxy_value=stage1_score,
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
            "hceye_proxy_features": hceye_proxy_dict,
            "hceye_inputs": {
                # Real element-derived measurements fed into the HCEye rules.
                # These come from the CANONICAL analysis path (long side 1280),
                # so they share the analysis scale of the eight visual features.
                # text_density_source distinguishes a real OCR measurement from
                # the defined no-elements zero case. Neutral fallbacks are not
                # allowed on this score-bearing route.
                "whitespace_ratio": whitespace_ratio,
                "text_density": text_density,
                "text_density_source": text_density_source,
                "analysis_provenance": (
                    analysis_measurement.as_dict() if analysis_measurement else None
                ),
            },
            "stage2_scenario_proxy": scenario_proxy,
            "cross_signal_review": cross_signal_review,
            "jokinen_diagnostic": jokinen_diagnostic,
            # Methodologically-separate LAYOUT construct. This is the stable,
            # image-based value that must NOT change when a target is selected.
            # Named "experimental_complexity_index" to keep it distinct from the
            # per-target search-difficulty result (see /api/scanpath-to-target ->
            # selected_target) and to signal it is an exploratory heuristic.
            "layout": {
                "experimental_complexity_index": stage1_score,
            },
            "scientific_semantics": dict(SCIENTIFIC_SEMANTICS),
            # P6 study-export identity: this binds every exported score to the
            # source commit/tree state, exact UMSI and EasyOCR artifacts,
            # reference norms, schema contracts, runtime freeze and versioned
            # cache identities.
            "reproducibility": study_reproducibility_metadata(),
            "stage1_feature_vector": base_vector.tolist(),
            "stage1_feature_names": stage1_feature_names,
            "stage1_vector_dtype": STAGE1_VECTOR_DTYPE,
            "vector_dimensions": int(base_vector.size),
            "reference": reference,
            "reference_meta": reference_meta,
            "readability_report": readability_report,
            # Lightweight list of every detected element's box, so the target
            # selector in the UI can offer the detected elements as one-tap
            # suggestions (in addition to free drag-box selection). These are
            # NATIVE-coordinate elements (target selection is the native path).
            "detected_elements": [
                {"id": e["id"], "bbox": list(e["bbox"]), "center": list(e["center"])}
                for e in (native_elements or [])
            ],
        })
    except ImageTooSmallError as e:
        return _too_small_error(e)
    except SaliencyUnavailableError as e:
        return _fail_closed_error(e.code, e.message)
    except ScoreInputUnavailableError as e:
        app.logger.exception("Score-driving layout/OCR analysis failed")
        return _fail_closed_error(e.code, e.message)
    except Stage1VectorUnavailableError as e:
        app.logger.exception("Stage-1 vector validation failed")
        return _fail_closed_error(e.code, e.message)
    except FeatureNormsError as e:
        # Client-safe boundary: the underlying exception message may contain
        # a local absolute file path (e.g. the feature-norms file location)
        # and must never be forwarded to the API response. Always use a
        # fixed, generic message here regardless of str(e) -- this also
        # protects against any future, more detailed FeatureNormsError text.
        # The full cause is preserved server-side via exception chaining and
        # the log below (never sent to the client).
        app.logger.exception("Feature norms validation failed (fail-closed)")
        return _fail_closed_error(
            "saliency_norms_invalid",
            "Production feature norms reference is unavailable or invalid; "
            "a complete analysis could not be produced.",
        )
    except Exception as e:
        return _server_error(e)
    finally:
        if filepath.exists():
            filepath.unlink()


# ===========================================================================
# Screen-set input (shared by the inter-screen consistency feature)
# ===========================================================================

def _read_screen_set(req):
    """
    Read an ordered set of screens from a request.

    Two input formats are accepted (both produce an ordered list of frames):
      1. Several image files under the multipart field "images" (the upload
         order is preserved).
      2. A single animated GIF under the field "image": every frame becomes one
         screen, in playback order (PIL Image.seek per frame).

    Returns
    -------
    (frames, names) : (list of np.ndarray BGR, list of str)
        One decoded BGR image per screen and a human-readable name per screen.

    Raises
    ------
    ValueError
        If no usable screens were supplied or a file cannot be decoded.
    """
    import io
    import cv2
    import numpy as np
    from PIL import Image

    allowed = SCREEN_SET_EXTENSIONS
    frames = []
    names = []
    total_pixels = 0
    total_decoded_bytes = 0

    def reserve_frame(width, height, decoded_bytes):
        """Apply per-frame and cumulative decoded screen-set budgets."""
        nonlocal total_pixels, total_decoded_bytes
        pixels = _validate_image_dimensions(width, height)
        next_pixels = total_pixels + pixels
        next_bytes = total_decoded_bytes + int(decoded_bytes)
        if (
            next_pixels > MAX_SCREEN_SET_PIXELS
            or next_bytes > MAX_SCREEN_SET_DECODED_BYTES
        ):
            raise InvalidImageUploadError(
                IMAGE_RESOURCE_LIMIT_ERROR_CODE,
                IMAGE_RESOURCE_LIMIT_ERROR_MESSAGE,
            )
        total_pixels = next_pixels
        total_decoded_bytes = next_bytes

    # --- Format 1: multiple image files ---
    files = req.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if files:
        if len(files) > MAX_SCREENS:
            raise ValueError(
                f"Too many screens: {len(files)} (limit {MAX_SCREENS})"
            )
        for f in files:
            ext = Path(f.filename).suffix.lower()
            if ext not in allowed:
                raise ValueError(f"Unsupported format: {ext}")
            data = f.read()
            width, height = _inspect_encoded_image_dimensions(data)
            reserve_frame(width, height, width * height * 3)
            img = _validate_uploaded_image_bytes(data)
            frames.append(img)
            names.append(f.filename)
        return frames, names

    # --- Format 2: a single (possibly animated) file under "image" ---
    single = req.files.get("image")
    if single is None or not single.filename:
        raise ValueError(
            "Provide either several files under 'images' or one file under 'image'"
        )
    ext = Path(single.filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"Unsupported format: {ext}")

    data = single.read()
    try:
        pil = Image.open(io.BytesIO(data))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read image: {single.filename}") from exc

    frame_index = 0
    try:
        if int(getattr(pil, "n_frames", 1)) > MAX_SCREENS:
            raise ValueError(
                f"Too many frames: GIF exceeds the {MAX_SCREENS}-screen limit"
            )
        while True:
            try:
                pil.seek(frame_index)
            except EOFError:
                break
            if frame_index >= MAX_SCREENS:
                raise ValueError(
                    f"Too many frames: GIF exceeds the {MAX_SCREENS}-screen limit"
                )

            # Reserve capacity before conversion allocates a full RGB frame.
            width, height = pil.size
            reserve_frame(width, height, width * height * 3)
            try:
                rgb = np.array(pil.convert("RGB"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Cannot read image: {single.filename}"
                ) from exc
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            names.append(f"{single.filename}#frame{frame_index}")
            frame_index += 1
    finally:
        pil.close()

    if not frames:
        raise ValueError(f"Cannot read image: {single.filename}")
    return frames, names


@app.route("/api/screen-consistency", methods=["POST"])
def screen_consistency():
    """
    Measure how consistent a set of screens from one product is.

    Detects UI elements on each screen and reports a geometric inter-screen
    consistency score (do recurring controls stay in place across screens?).

    Input: several files under "images", OR one animated GIF under "image".

    NOTE (honesty): this is a purely geometric metric. It does NOT identify
    controls semantically and is NOT yet validated against user behaviour.
    """
    try:
        frames, names = _read_screen_set(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if len(frames) < 2:
        return jsonify({
            "error": "Need at least 2 screens to measure inter-screen consistency",
            "n_screens": len(frames),
        }), 400

    try:
        from cognitive.element_detector import detect_elements
        from stage2.screen_consistency import compute_screen_set_consistency

        element_sets = []
        image_shapes = []
        per_screen = []
        for name, img in zip(names, frames):
            elements = detect_elements(img)
            element_sets.append(elements)
            image_shapes.append(img.shape[:2])
            per_screen.append({"screen": name, "n_elements": len(elements)})

        result = compute_screen_set_consistency(element_sets, image_shapes)
        result["screens"] = per_screen
        return jsonify(result)
    except Exception as e:
        return _server_error(e)


@app.route("/api/learning-curve", methods=["POST"])
def learning_curve():
    """
    Predict how search effort on one screen drops as the user practises it.

    Runs the Jokinen search simulation at several exposure counts (default
    1 / 10 / 100 uses). The first point (1 use) equals the novice prediction.

    NOTE (honesty): only the SHAPE of the curve is meaningful; the absolute
    learning rate depends on an uncalibrated parameter (needs a user study).
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in SINGLE_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Optional exposures list, e.g. ?exposures=1,5,20,100
    exposures_arg = request.args.get("exposures", default=None)
    if exposures_arg:
        try:
            exposures = [int(x) for x in exposures_arg.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "exposures must be a comma-separated list of integers"}), 400
        if not exposures:
            return jsonify({"error": "exposures must contain at least one value"}), 400
        if len(exposures) > MAX_SCHEDULE_LEN:
            return jsonify({"error": f"exposures accepts at most {MAX_SCHEDULE_LEN} values"}), 400
        if any(x < 1 for x in exposures):
            return jsonify({"error": "exposures values must be >= 1"}), 400
    else:
        exposures = [1, 10, 100]

    n_simulations = _clamp_simulations(request)
    use_saliency = request.args.get("use_saliency", "true").lower() != "false"
    (screen_w_cm, screen_h_cm,
     viewing_cm, display_preset_meta) = _resolve_display_preset(request)

    image_hash, image_bytes = _hash_upload(file)
    _validate_uploaded_image_bytes(image_bytes)
    filepath = _persist_uploaded_image_bytes(ext, image_bytes)

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

        saliency_map = None
        if use_saliency:
            try:
                saliency_map, _, _ = _predict_saliency_cached(image_hash, filepath)
            except Exception:
                return _requested_saliency_failure("/api/learning-curve")

        params = JokinenParams(
            n_simulations=min(n_simulations, 500),
            random_seed=42,
        )
        jokinen = JokinenSearchModel(params)
        result = jokinen.predict_learning_curve(
            elements=elements,
            exposures=exposures,
            saliency_map=saliency_map,
            image_shape=img.shape[:2],
            screen_width_cm=screen_w_cm,
            screen_height_cm=screen_h_cm,
            viewing_distance_cm=viewing_cm,
        )
        result["filename"] = file.filename
        result["n_elements"] = len(elements)
        result["display_preset"] = display_preset_meta
        result["analysis_complete"] = True
        result["analysis_mode"] = (
            "saliency_augmented" if use_saliency else "feature_only_explicit"
        )
        result["saliency_requested"] = use_saliency
        result["saliency_used"] = saliency_map is not None
        return jsonify(result)
    except Exception as e:
        return _server_error(e)
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/api/product-learning", methods=["POST"])
def product_learning():
    """
    Predict steady-state learning for a whole multi-screen product.

    Couples the two axes: the per-screen learning curve (time) and the
    inter-screen consistency (space). A spatially consistent product transfers
    learning across screens, so it reaches a lower steady-state search load.

    Input: several files under "images", OR one animated GIF under "image"
    (one frame per screen). Optional ?total_uses=1,10,100.

    NOTE (honesty): first unvalidated coupling; the transfer model and the
    learning rate are declared modelling choices, not calibrated.
    """
    try:
        frames, names = _read_screen_set(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if len(frames) < 2:
        return jsonify({
            "error": "Need at least 2 screens to model a multi-screen product",
            "n_screens": len(frames),
        }), 400

    total_uses_arg = request.args.get("total_uses", default=None)
    if total_uses_arg:
        try:
            total_uses = [int(x) for x in total_uses_arg.split(",") if x.strip()]
        except ValueError:
            return jsonify({"error": "total_uses must be a comma-separated list of integers"}), 400
        if not total_uses:
            return jsonify({"error": "total_uses must contain at least one value"}), 400
        if len(total_uses) > MAX_SCHEDULE_LEN:
            return jsonify({"error": f"total_uses accepts at most {MAX_SCHEDULE_LEN} values"}), 400
        if any(x < 1 for x in total_uses):
            return jsonify({"error": "total_uses values must be >= 1"}), 400
    else:
        total_uses = [1, 10, 100]

    n_simulations = _clamp_simulations(request)
    (screen_w_cm, screen_h_cm,
     viewing_cm, display_preset_meta) = _resolve_display_preset(request)

    try:
        from cognitive.element_detector import detect_elements
        from cognitive.jokinen_model import JokinenSearchModel, JokinenParams
        from stage2.screen_consistency import compute_screen_set_consistency

        element_sets = []
        image_shapes = []
        per_screen = []
        for name, img in zip(names, frames):
            elements = detect_elements(img)
            element_sets.append(elements)
            image_shapes.append(img.shape[:2])
            per_screen.append({"screen": name, "n_elements": len(elements)})

        # Space axis: geometric inter-screen consistency.
        consistency = compute_screen_set_consistency(element_sets, image_shapes)

        # Time axis, coupled with space: product-level learning curve.
        params = JokinenParams(
            n_simulations=min(n_simulations, 500),
            random_seed=42,
        )
        jokinen = JokinenSearchModel(params)
        product = jokinen.predict_product_learning(
            element_sets=element_sets,
            consistency_score=consistency["consistency_score"],
            image_shapes=image_shapes,
            total_uses=total_uses,
            screen_width_cm=screen_w_cm,
            screen_height_cm=screen_h_cm,
            viewing_distance_cm=viewing_cm,
        )
        product["screens"] = per_screen
        product["display_preset"] = display_preset_meta
        product["inter_screen_consistency"] = consistency
        return jsonify(product)
    except Exception as e:
        return _server_error(e)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Stage 1 — Visual Complexity Analyzer")
    print("  Endpoints:")
    print("    POST /api/analyze         → v∈ℝ⁸ visual complexity")
    print("    POST /api/saliency        → s∈ℝ⁵ saliency features")
    print("    POST /api/search-time     → Jokinen search time")
    print("    POST /api/cognitive-load   → h∈ℝ⁶ + full vector ℝ¹⁹")
    print("    POST /api/screen-consistency → inter-screen consistency")
    print("    POST /api/learning-curve  → novice→expert learning curve")
    print("    POST /api/product-learning → product-level learning (time × space)")
    print("  Open: http://localhost:5001")
    print("=" * 50 + "\n")
    # Prime the UMSI++ model before accepting traffic so the first real
    # request doesn't pay the TensorFlow graph-build cost.
    _warmup_saliency_model()
    # Debug mode is OFF by default (the reloader/debugger must never be exposed
    # if the host ever changes from localhost). Enable explicitly for local
    # development with FLASK_DEBUG=1.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    host = os.environ.get("FLASK_HOST", "localhost")
    app.run(host=host, port=5001, debug=debug_mode)
