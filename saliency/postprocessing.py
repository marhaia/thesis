"""Production postprocessing for UMSI saliency predictions.

This module intentionally depends only on NumPy and OpenCV so that the exact
production postprocessor can be regression-tested without importing the
TensorFlow model stack.
"""

from __future__ import annotations

import cv2
import numpy as np


def _require_finite(arr: np.ndarray, label: str) -> None:
    """Reject non-finite saliency data."""
    try:
        finite = np.isfinite(arr)
    except TypeError as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not finite.all():
        raise ValueError(f"{label} contains non-finite values")


def _as_finite_2d(saliency_map: np.ndarray, label: str) -> np.ndarray:
    """Return a real, finite, non-empty 2D saliency map."""
    arr = np.asarray(saliency_map)
    if arr.ndim == 3:
        if arr.shape[2] != 1:
            raise ValueError(
                f"{label} must have one channel when three-dimensional"
            )
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{label} must be a non-empty 2D map")
    if np.iscomplexobj(arr):
        raise ValueError(f"{label} must contain real numeric values")
    _require_finite(arr, label)
    return arr


def normalize_saliency_map(saliency_map: np.ndarray) -> np.ndarray:
    """Apply the single production normalization policy to a saliency map.

    Finite, non-constant maps use true min-max normalization. Constant maps
    have no defined contrast and become deterministic all-zero maps. NaN and
    positive or negative infinity fail closed with ``ValueError``.
    """
    arr = _as_finite_2d(saliency_map, "saliency map")
    try:
        work = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("saliency map must contain real numeric values") from exc
    vmin = float(work.min())
    vmax = float(work.max())

    with np.errstate(over="ignore", invalid="ignore"):
        span = vmax - vmin
    if span == 0.0:
        return np.zeros(work.shape, dtype=np.float32)

    if np.isfinite(span):
        normalized = (work - vmin) / span
    else:
        # The endpoints can both be finite while their direct difference
        # overflows (for example -float64.max to +float64.max). Scaling first
        # preserves true min-max semantics for that finite input class.
        scale = max(abs(vmin), abs(vmax))
        scaled = work / scale
        scaled_min = float(scaled.min())
        scaled_span = float(scaled.max()) - scaled_min
        normalized = (scaled - scaled_min) / scaled_span
    _require_finite(normalized, "normalized saliency map")
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized.astype(np.float32)


def postprocess_saliency(
    pred: np.ndarray,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """Resize a UMSI prediction and return a finite min-max-normalized map.

    The accepted input shapes are ``(H, W)`` and ``(H, W, 1)``. Constant maps
    become all-zero maps. Invalid shapes, dimensions, NaN, and Inf raise
    ``ValueError`` so callers cannot silently publish an invalid saliency map.
    """
    if original_h <= 0 or original_w <= 0:
        raise ValueError("original image dimensions must be positive")

    pred = _as_finite_2d(pred, "saliency prediction")

    # Preserve the declared constant-map policy before interpolation. OpenCV's
    # float64 resize path can introduce tiny numerical differences even when
    # every source pixel is exactly equal; min-max normalization would then
    # amplify that numerical noise into artificial saliency structure.
    if float(np.min(pred)) == float(np.max(pred)):
        return np.zeros((original_h, original_w), dtype=np.float32)

    pred_shape = pred.shape
    rows_rate = original_h / pred_shape[0]
    cols_rate = original_w / pred_shape[1]

    if rows_rate > cols_rate:
        new_cols = (pred_shape[1] * original_h) // pred_shape[0]
        pred = cv2.resize(pred, (new_cols, original_h))
        offset = (pred.shape[1] - original_w) // 2
        img = pred[:, offset:offset + original_w]
    else:
        new_rows = (pred_shape[0] * original_w) // pred_shape[1]
        pred = cv2.resize(pred, (original_w, new_rows))
        offset = (pred.shape[0] - original_h) // 2
        img = pred[offset:offset + original_h, :]

    if img.shape != (original_h, original_w):
        raise ValueError(
            "postprocessed saliency shape mismatch: "
            f"expected {(original_h, original_w)}, got {img.shape}"
        )
    return normalize_saliency_map(img)
