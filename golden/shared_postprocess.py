"""ONE shared, explicitly-documented comparison postprocess, applied identically
to BOTH the legacy (TF1/Keras2) and the modern (TF2/Keras3) raw saliency
tensors.

The raw model output for both implementations is a (512, 512) heatmap from a
linear decoder head (no final activation), so it can contain negative values.
This module performs, with a single implementation shared by both sides:

  1. exact squeeze to 2D,
  2. identical aspect-preserving resize + unpad crop to the original fixture
     (H, W) using cv2.INTER_LINEAR (the same op both native pipelines use),
  3. identical min-max normalisation to exact [0, 1].

It intentionally does NOT rotate, flip, transpose, shift, align or otherwise
transform either result. Any orientation/shape difference is a genuine failure.
Depends only on numpy + cv2 so it runs identically in both runtimes.
"""
import numpy as np
import cv2


def squeeze_2d(raw):
    """Exact squeeze to a 2D array. Raises on anything that is not (H, W) or
    (H, W, 1)."""
    a = np.asarray(raw)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim != 2:
        raise ValueError("expected 2D (or HxWx1) raw map, got shape %r" % (a.shape,))
    return a.astype(np.float32)


def resize_unpad(raw2d, orig_h, orig_w):
    """Aspect-preserving resize + centre-crop of the (512,512) map back to the
    original (orig_h, orig_w). Mirrors the reverse-padding used by both native
    postprocess functions (sal_imp_utilities.postprocess_predictions and
    saliency.umsi_model.postprocess_saliency)."""
    ph, pw = raw2d.shape
    rows_rate = orig_h / ph
    cols_rate = orig_w / pw
    if rows_rate > cols_rate:
        new_cols = (pw * orig_h) // ph
        r = cv2.resize(raw2d, (new_cols, orig_h))
        off = (r.shape[1] - orig_w) // 2
        out = r[:, off:off + orig_w]
    else:
        new_rows = (ph * orig_w) // pw
        r = cv2.resize(raw2d, (orig_w, new_rows))
        off = (r.shape[0] - orig_h) // 2
        out = r[off:off + orig_h, :]
    return out.astype(np.float32)


def minmax01(arr):
    """Min-max normalise to exact [0, 1]. Raises on non-finite or constant."""
    a = arr.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("non-finite value in map before normalisation")
    vmin = float(a.min())
    vmax = float(a.max())
    if vmax - vmin <= 0.0:
        raise ValueError("constant map (zero dynamic range)")
    return ((a - vmin) / (vmax - vmin)).astype(np.float32)


def shared_postprocess(raw, orig_h, orig_w):
    """squeeze -> resize/unpad -> min-max[0,1]. The single shared comparison
    map used for BOTH implementations."""
    return minmax01(resize_unpad(squeeze_2d(raw), orig_h, orig_w))
