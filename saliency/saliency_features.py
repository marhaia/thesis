#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
saliency_features.py — Saliency-Derived Feature Extraction
============================================================
Extracts quantitative features from a predicted saliency heatmap.
These features form the s∈ℝ⁵ block of the authoritative, task-independent
Stage-1 vector x = [v8 | s5 | h6] ∈ ℝ¹⁹.

Features extracted (s∈ℝ⁵):
  1. saliency_dispersion    — spatial spread of attention (σ of hotspots)
  2. saliency_peak_count    — number of distinct attention peaks
  3. saliency_center_bias   — how much attention concentrates at center
  4. saliency_entropy        — Shannon entropy of the saliency distribution
  5. saliency_coverage       — fraction of image area receiving >50% max attention

The five values are numeric screenshot descriptors. They do not by themselves
measure cognitive load or observed human attention.

Mathematical Definitions
------------------------
Let S(x,y) be the normalized saliency map, S∈[0,1].

1. **Dispersion** (spatial σ):
   $\\sigma_S = \\sqrt{\\text{Var}[x \\cdot S] + \\text{Var}[y \\cdot S]}$
   where x,y are normalized pixel coordinates ∈[0,1].

2. **Peak count**:
   Number of local maxima in S after Gaussian blur (σ=5) with value ≥ 0.3·max(S).

3. **Center bias**:
   $CB = \\frac{\\sum_{(x,y) \\in C_{0.25}} S(x,y)}{\\sum_{(x,y)} S(x,y)}$
   where C₀.₂₅ is the central 25% of the image area.

4. **Entropy**:
   $H = -\\sum_b p_b \\log_2 p_b$
   where pᵢ is the probability mass in histogram bin b (32 bins).

5. **Coverage**:
   $\\text{Cov} = \\frac{|\\{(x,y) : S(x,y) > 0.5 \\cdot \\max(S)\\}|}{W \\cdot H}$

References
----------
- Rosenholtz, R., Li, Y., & Nakano, L. (2007). "Measuring visual clutter."
- Itti, L. & Koch, C. (2001). "Computational modelling of visual attention."
- Das, S. et al. (2024). "Shifting Focus with HCEye." PACM HCI / ETRA 2024.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage

from saliency.postprocessing import normalize_saliency_map


def extract_saliency_features(saliency_map: np.ndarray) -> Dict[str, float]:
    """Extract all saliency-derived features from a heatmap.

    Args:
        saliency_map: 2D array of shape (H, W), values in [0, 1].

    Returns:
        Dictionary with keys:
          - saliency_dispersion
          - saliency_peak_count
          - saliency_center_bias
          - saliency_entropy
          - saliency_coverage
    """
    # Use the same explicit policy as the UMSI production postprocessor. This
    # is intentionally idempotent for its already-normalized model output and
    # prevents feature extraction from introducing a divergent normalization.
    smap = normalize_saliency_map(saliency_map).astype(np.float64, copy=False)

    return {
        "saliency_dispersion": _compute_dispersion(smap),
        "saliency_peak_count": _compute_peak_count(smap),
        "saliency_center_bias": _compute_center_bias(smap),
        "saliency_entropy": _compute_entropy(smap),
        "saliency_coverage": _compute_coverage(smap),
    }


# ============================================================================
# Individual feature computations
# ============================================================================

def _compute_dispersion(smap: np.ndarray) -> float:
    """Spatial dispersion of saliency (weighted standard deviation).

    Low dispersion → attention concentrated in one area (focal interface).
    High dispersion → attention spread across the image (cluttered interface).

    Dispersion correlates with visual complexity: highly cluttered UIs produce
    diffuse saliency maps (Rosenholtz et al., 2007; Itti & Koch, 2001).

    Reference:
        Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
          Journal of Vision, 7(2), 17.
        Peters, R.J., et al. (2005). Components of bottom-up gaze allocation.
          Vision Research, 45(18), 2397–2416.  [saliency-weighted spread]

    Returns value in [0, 1] (normalized by image diagonal).
    """
    h, w = smap.shape
    total = smap.sum()
    if total == 0:
        return 0.0

    # Normalized coordinate grids ∈ [0, 1]
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float64) / max(h - 1, 1)
    xx = xx.astype(np.float64) / max(w - 1, 1)

    # Weighted means
    wx = (xx * smap).sum() / total
    wy = (yy * smap).sum() / total

    # Weighted variances
    var_x = ((xx - wx) ** 2 * smap).sum() / total
    var_y = ((yy - wy) ** 2 * smap).sum() / total

    # Combined spatial σ (in normalized coords → [0, ~0.7])
    sigma = np.sqrt(var_x + var_y)

    # Normalize to [0, 1]: max possible σ ≈ sqrt(0.25 + 0.25) = 0.707
    return float(min(sigma / 0.707, 1.0))


def _detect_peak_mask(smap: np.ndarray, sigma: float = 5.0,
                      threshold_ratio: float = 0.3) -> np.ndarray:
    """Return the exact significant-peak mask used by peak counting."""
    smoothed = ndimage.gaussian_filter(smap, sigma=sigma)
    vmax = smoothed.max()
    if vmax == 0:
        return np.zeros(smap.shape, dtype=bool)

    local_max = ndimage.maximum_filter(smoothed, size=int(sigma * 4 + 1))
    return (smoothed == local_max) & (smoothed >= threshold_ratio * vmax)


def _compute_peak_count(smap: np.ndarray, sigma: float = 5.0,
                         threshold_ratio: float = 0.3) -> int:
    """Count distinct attention peaks in the saliency map.

    Multiple peaks record competing regions in the model-estimated saliency
    map. The count is an exploratory image descriptor and is not a direct
    measurement of attentional switching or cognitive load.

    Steps:
      1. Gaussian blur (sigma=5 px) to suppress sub-pixel noise
      2. Find local maxima (pixels > all 8 neighbors after dilation)
      3. Filter peaks below threshold_ratio × max value

    Reference:
        Itti, L., & Koch, C. (2001). Computational modelling of visual attention.
          Nature Reviews Neuroscience, 2(3), 194–203.
        Lavie, N. (2005). Distracted and confused? Selective attention under load.
          Trends in Cognitive Sciences, 9(2), 75–82.

    Returns integer count of significant peaks.
    """
    peaks = _detect_peak_mask(smap, sigma=sigma, threshold_ratio=threshold_ratio)

    # Label connected components
    _, num_features = ndimage.label(peaks)
    return int(num_features)


def _compute_center_bias(smap: np.ndarray,
                          center_fraction: float = 0.25) -> float:
    """Fraction of total saliency falling in the central region.

    Human gaze has a well-documented center bias: fixations are
    disproportionately concentrated in the central region of a display,
    independent of image content (Tatler, 2007; Tseng et al., 2009).
    Low center-bias in a model-estimated saliency map indicates that more of
    the predicted saliency mass lies outside the central region. No direct
    human-performance conclusion is inferred from this feature alone.

    Center region: inner sqrt(center_fraction) strip per axis,
    i.e., the central 50×50% area for center_fraction=0.25.

    Reference:
        Tatler, B.W. (2007). The central fixation bias in scene viewing.
          Journal of Vision, 7(14), 4.
        Tseng, P.-H. et al. (2009). Quantifying center bias of observers
          in free viewing of dynamic natural scenes. Journal of Vision, 9(7), 4.

    Returns value in [0, 1]. Higher = more center-biased.
    """
    h, w = smap.shape
    total = smap.sum()
    if total == 0:
        return 0.0

    # Central region boundaries (inner sqrt(fraction) in each dimension)
    frac_half = np.sqrt(center_fraction) / 2.0
    r0 = int(h * (0.5 - frac_half))
    r1 = int(h * (0.5 + frac_half))
    c0 = int(w * (0.5 - frac_half))
    c1 = int(w * (0.5 + frac_half))

    center_sum = smap[r0:r1, c0:c1].sum()
    return float(center_sum / total)


def _compute_entropy(smap: np.ndarray, n_bins: int = 32) -> float:
    """Shannon entropy of the saliency distribution.

    Treats the saliency map as a discrete probability distribution
    (32 histogram bins) and computes its entropy in bits.
    High entropy → uniform/spread distribution (many competing regions).
    Low entropy  → highly peaked distribution (single dominant region).

    Saliency entropy correlates with visual complexity: interfaces with
    high clutter produce higher-entropy saliency maps
    (Rosenholtz et al., 2007).

    Reference:
        Shannon, C.E. (1948). A mathematical theory of communication.
          Bell System Technical Journal, 27(3), 379–423.
        Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
          Journal of Vision, 7(2), 17.

    Returns value in [0, 1] (normalized by log₂(n_bins)).
    """
    # Create histogram
    hist, _ = np.histogram(smap, bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)

    # Normalize to probabilities
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total

    # Shannon entropy
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize to [0, 1] by dividing by max possible entropy
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _compute_coverage(smap: np.ndarray,
                       threshold_ratio: float = 0.5) -> float:
    """Fraction of the image receiving significant saliency.

    Counts pixels where S(x,y) > threshold_ratio × max(S).

    Returns value in [0, 1]. Higher = more spread out attention.
    """
    vmax = smap.max()
    if vmax == 0:
        return 0.0

    above = (smap > threshold_ratio * vmax).sum()
    total = smap.shape[0] * smap.shape[1]
    return float(above / total)


# ============================================================================
# CLI test
# ============================================================================
if __name__ == "__main__":
    # Generate a test saliency map (Gaussian blob)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    center_y, center_x = h // 2, w // 2
    sigma = 80
    test_map = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    test_map = test_map.astype(np.float32)

    features = extract_saliency_features(test_map)
    print("Saliency features (Gaussian blob test):")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
