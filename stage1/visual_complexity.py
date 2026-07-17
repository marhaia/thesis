#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 — Task-Independent Visual Complexity Extraction

Computes a visual complexity vector v ∈ ℝ⁸ from an automotive GUI screenshot.
Each feature captures a different dimension of visual complexity that may
contribute to cognitive load.

The 8 features (as specified in the project plan):
    1. Shannon Entropy          — Global information density (Shannon, 1948)
    2. Edge Density             — Structural complexity via Canny edges
    3. Feature Congestion       — Multi-scale clutter: color + contrast + orientation
                                  (Rosenholtz et al., 2007) [AIM m8]
    4. Subband Entropy          — Redundancy via steerable pyramid decomposition
                                  (Rosenholtz et al., 2007) [AIM m7]
    5. Layout Symmetry          — Axial balance (Miniukovich & De Angeli, 2015)
    6. Chromatic Coherence      — Color palette fragmentation (Hasler & Süsstrunk, 2003)
    7. Visual Hierarchy         — Contrast gradients + size dominance (Tuch et al., 2009)
    8. Interactive Element Density — UI control density per area

Usage:
    python visual_complexity.py                           # all PNGs in data/screenshots/
    python visual_complexity.py --image path/to/img.png   # single image
    python visual_complexity.py --dir path/to/folder/     # all PNGs in folder

Output: CSV with one row per image, columns = features.
"""

# ───────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pyrtools as pt
from PIL import Image
from scipy import ndimage, signal
from skimage import transform


# ═══════════════════════════════════════════════════════════════════════════
# Shared Utility Functions
# ─────────────────────────────────────────────────────────────────────────
# These helper functions are ported 1:1 from the AIM repository:
#   Source: aim/backend/aim/metrics/image_visual_clutter_utils.py
#   Commit: branch aim2 (Aalto-UI/aim)
#
# They implement low-level signal processing primitives needed by the
# Feature Congestion (m8) and Subband Entropy (m7) algorithms from:
#   Rosenholtz, Li & Nakano (2007). "Measuring Visual Clutter."
#   Journal of Vision, 7(2):17, 1–22.
#
# Each function is prefixed with _ to mark it as module-private.
# ═══════════════════════════════════════════════════════════════════════════

def _rgb2lab(im: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image [0–255] to CIE L*a*b* color space.

    Ported from: AIM image_visual_clutter_utils.py → rgb2lab()

    Steps:
      1. Linearize sRGB via inverse gamma (IEC 61966-2-1):
         - If value >= 0.04045: ((v + 0.055) / 1.055)^2.4
         - Else: v / 12.92
      2. Convert linear RGB → CIE XYZ using the sRGB→XYZ D65 matrix.
      3. Normalize XYZ by the D65 white point (95.047, 100, 108.883).
      4. Apply the CIE nonlinear transform:
         - If t >= 0.008856: t^(1/3)
         - Else: 7.787·t + 16/116
      5. Compute L*, a*, b* from transformed XYZ.

    Why CIELab?
      Feature Congestion computes color clutter as the covariance volume
      in a perceptually uniform color space. Lab ensures that Euclidean
      distances approximate perceived color differences (ΔE).

    Args:
        im: HxWx3 uint8 RGB image
    Returns:
        HxWx3 float64 CIELab image (L in [0,100], a/b in ~[-128,127])
    """
    # Step 1: Linearize sRGB gamma
    im = im.astype(np.float64) / 255.0
    mask = im >= 0.04045
    im[mask] = ((im[mask] + 0.055) / 1.055) ** 2.4  # inverse gamma
    im[~mask] = im[~mask] / 12.92                    # linear segment

    # Step 2: sRGB → XYZ (D65 illuminant) using the standard 3×3 matrix
    matrix = np.array([
        [0.412453, 0.357580, 0.180423],   # X coefficients
        [0.212671, 0.715160, 0.072169],   # Y coefficients (≈ luminance)
        [0.019334, 0.119193, 0.950227],   # Z coefficients
    ])
    xyz = np.dot(im, matrix.T)

    # Step 3: Normalize by D65 white point reference values.
    # xyz is on the 0..1 scale (linear-sRGB white maps to Y=1.0), so the white
    # point must be on the SAME 0..1 scale (Xn=0.95047, Yn=1.0, Zn=1.08883).
    # Using the 0..100-scale values here (95.047/100/108.883) was a unit bug:
    # it drove white to L*~=8.99 instead of 100 and corrupted every downstream
    # Lab-based metric (feature congestion, subband entropy).
    xyz[:, :, 0] /= 0.95047    # Xn (D65)
    xyz[:, :, 1] /= 1.00000    # Yn (D65)
    xyz[:, :, 2] /= 1.08883    # Zn (D65)

    # Step 4: CIE nonlinear companding (cube-root or linear segment)
    mask = xyz >= 0.008856              # threshold = (6/29)^3
    xyz[mask] = xyz[mask] ** (1.0 / 3.0)
    xyz[~mask] = 7.787 * xyz[~mask] + 16.0 / 116.0

    # Step 5: XYZ → L*a*b*
    lab = np.zeros_like(xyz)
    lab[:, :, 0] = 116.0 * xyz[:, :, 1] - 16.0                 # L*
    lab[:, :, 1] = 500.0 * (xyz[:, :, 0] - xyz[:, :, 1])       # a*
    lab[:, :, 2] = 200.0 * (xyz[:, :, 1] - xyz[:, :, 2])       # b*
    return lab


def _conv2(x: np.ndarray, y: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
    """
    2D convolution matching MATLAB's conv2 behavior.

    Ported from: AIM image_visual_clutter_utils.py → conv2()

    The rot90 trick ensures the output matches MATLAB's convention
    (MATLAB does correlation, SciPy does convolution — rotating both
    inputs by 180° converts one to the other).

    Args:
        x, y: 2D arrays to convolve
        mode: 'same' for same-size output, None for full output
    """
    if mode == "same":
        return np.rot90(signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
    return signal.convolve2d(x, y)


def _RRoverlapconv(kernel: np.ndarray, in_: np.ndarray) -> np.ndarray:
    """
    Border-corrected convolution (Rosenholtz & Bhatt).

    Ported from: AIM image_visual_clutter_utils.py → RRoverlapconv()

    Standard convolution near image borders is biased because the kernel
    extends beyond the image. This corrects by dividing each pixel's
    convolution result by the sum of kernel weights that actually
    overlap with the image at that position.

    Mathematical formulation:
        out(x,y) = Σ(kernel) × (kernel ⊛ image)(x,y) / (kernel ⊛ ones)(x,y)

    This is used extensively in Feature Congestion to compute local
    means and covariances without border artifacts.
    """
    out = _conv2(in_, kernel, mode="same")
    rect = np.ones_like(in_)
    overlapsum = _conv2(rect, kernel, "same")  # effective kernel mass at each pixel
    out = np.sum(kernel) * out / overlapsum     # normalize by actual overlap
    return out


def _RRgaussfilter1D(halfsupport: int, sigma: float, center: float = 0) -> np.ndarray:
    """
    Create a 1D Gaussian kernel as a 1×N row vector.

    Ported from: AIM image_visual_clutter_utils.py → RRgaussfilter1D()

    The kernel spans [-halfsupport, +halfsupport] and is normalized
    to sum to 1. An optional 'center' parameter shifts the peak
    (used for oriented filters in _orient_filtnew).

    Formula: g(x) = exp(-(x - center)² / (2σ²))

    Args:
        halfsupport: kernel extends from -halfsupport to +halfsupport
        sigma: standard deviation of the Gaussian
        center: offset of the peak from 0 (default=0)
    Returns:
        1×(2·halfsupport+1) normalized kernel
    """
    t = list(range(-halfsupport, halfsupport + 1))
    kernel = np.array([np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) for x in t])
    kernel = kernel / sum(kernel)   # normalize to unit sum
    return kernel.reshape(1, kernel.shape[0])


def _DoG1filter(a: int, sigma: float):
    """
    Create a 1D Difference-of-Gaussians (DoG) filter pair.

    Ported from: AIM image_visual_clutter_utils.py → DoG1filter()

    Returns two 1D Gaussians:
      - inner (σ_i = 0.71·σ) — narrower, captures fine detail
      - outer (σ_o = 1.14·σ) — wider, captures surround

    Used in _RRcontrast1channel to compute local contrast as
    |inner - outer| (center-surround mechanism mimicking retinal
    ganglion cells).

    The 0.71/1.14 ratio comes from Rosenholtz's clutter model and
    approximates a Laplacian-of-Gaussian operator.

    Args:
        a: half-support of the filter (= round(3·σ))
        sigma: base scale parameter
    Returns:
        (inner_kernel, outer_kernel) — each is a 1×(2a+1) row vector
    """
    sigi = 0.71 * sigma   # inner Gaussian σ
    sigo = 1.14 * sigma   # outer Gaussian σ
    t = range(-a, a + 1)
    gi = np.array([np.exp(-(x ** 2) / (2 * sigi ** 2)) for x in t])
    gi = gi / np.sum(gi)
    go = np.array([np.exp(-(x ** 2) / (2 * sigo ** 2)) for x in t])
    go = go / np.sum(go)
    return gi.reshape(1, gi.shape[0]), go.reshape(1, go.shape[0])


def _addborder(im: np.ndarray, xbdr: int, ybdr: int, arg) -> np.ndarray:
    """
    Pad an image with a specified border style.

    Ported from: AIM image_visual_clutter_utils.py → addborder()

    Border styles:
      - int/float: constant-value padding
      - 'even': mirror reflection (ABCD → DCBA|ABCD|DCBA)
      - 'odd': reflection with negation at boundary (ABCD → DCBA|ABCD|DCBA,
               but boundary pixel not duplicated)
      - 'wrap': circular/periodic extension
    """
    if isinstance(arg, (int, float)):
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_CONSTANT, value=arg)
    elif arg == "even":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT)
    elif arg == "odd":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT_101)
    elif arg == "wrap":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_WRAP)
    raise ValueError("unknown border style")


def _filt2(kernel: np.ndarray, im1: np.ndarray, reflect_style="odd") -> np.ndarray:
    """
    2D filtering with reflected border handling.

    Ported from: AIM image_visual_clutter_utils.py → filt2()

    Extends the image by the kernel size using odd-reflection,
    convolves, then crops back to the original size. This avoids
    the boundary artifacts that scipy.signal.convolve2d would produce.
    """
    ky, kx = kernel.shape
    iy, ix = im1.shape
    imbig = _addborder(im1, kx, ky, reflect_style)   # pad
    imbig = _conv2(imbig, kernel, "same")             # convolve
    return imbig[ky:ky + iy, kx:kx + ix]             # crop to original size


def _RRcontrast1channel(pyr: Dict, DoG_sigma: float = 2) -> List:
    """
    Compute local contrast at each pyramid level using a center-surround
    Difference-of-Gaussians (DoG) mechanism.

    Ported from: AIM image_visual_clutter_utils.py → RRcontrast1channel()

    Algorithm (per pyramid level):
      1. Apply inner Gaussian (σ = 0.71·DoG_sigma) — separable H then V.
      2. Apply outer Gaussian (σ = 1.14·DoG_sigma) — separable H then V.
      3. Contrast = |inner - outer| at each pixel.

    This approximates a Laplacian-of-Gaussian and models how retinal
    ganglion cells respond to local luminance contrast.

    Args:
        pyr: Gaussian pyramid dict with keys (level, 0)
        DoG_sigma: base scale for the DoG filter (default=2)
    Returns:
        List of contrast maps, one per pyramid level
    """
    levels = len(pyr)
    contrast = [0] * levels
    innerG1, outerG1 = _DoG1filter(round(DoG_sigma * 3), DoG_sigma)
    for i in range(levels):
        # Separable 2D filtering: first horizontal, then vertical
        inner = _filt2(innerG1, pyr[(i, 0)])      # inner Gaussian, horizontal
        inner = _filt2(innerG1.T, inner)           # inner Gaussian, vertical
        outer = _filt2(outerG1, pyr[(i, 0)])      # outer Gaussian, horizontal
        outer = _filt2(outerG1.T, outer)           # outer Gaussian, vertical
        contrast[i] = abs(inner - outer)           # center-surround difference
    return contrast


def _reduce(image0: np.ndarray, kernel=None) -> np.ndarray:
    """
    Downsample image by factor 2 with anti-aliasing (Gaussian pyramid reduce).

    Ported from: AIM image_visual_clutter_utils.py → reduce()

    Uses a 5-tap binomial kernel [0.05 0.25 0.40 0.25 0.05] (Burt & Adelson,
    1983) applied separably before 2× decimation.
    """
    if kernel is None:
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])  # 5-tap binomial
    ysize, xsize = image0.shape
    image0 = _filt2(kernel, image0)            # low-pass horizontal
    image1 = image0[:, range(0, xsize, 2)]     # decimate columns by 2
    image1 = _filt2(kernel.T, image1)          # low-pass vertical
    return image1[range(0, ysize, 2), :]       # decimate rows by 2


def _RRoverlapconvexpand(in_: np.ndarray, kernel=None) -> np.ndarray:
    """
    Upsample image by factor 2 with border-corrected interpolation.

    Ported from: AIM image_visual_clutter_utils.py → RRoverlapconvexpand()

    Inserts zeros between samples (upsampling), then smooths with
    a 2× scaled kernel using border-corrected convolution.
    This is the "expand" step for collapsing multi-scale clutter maps
    back to the original resolution.
    """
    if kernel is None:
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
    ysize, xsize = in_.shape
    kernel2 = kernel * 2   # scale kernel by 2 to compensate for zero-insertion
    # Upsample horizontally: insert zeros between columns
    tmp = np.zeros([ysize, 2 * xsize])
    for k in range(xsize):
        tmp[:, k * 2] = in_[:, k]
    tmp = _RRoverlapconv(kernel2, tmp)          # smooth horizontally
    # Upsample vertically: insert zeros between rows
    out = np.zeros([2 * ysize, 2 * xsize])
    for k in range(ysize):
        out[k * 2, :] = tmp[k, :]
    out = _RRoverlapconv(kernel2.T, out)        # smooth vertically
    return out


def _imrotate(im: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a filter kernel by a given angle (in degrees).

    Ported from: AIM image_visual_clutter_utils.py → imrotate()

    Used in _orient_filtnew to create the ±45° diagonal orientation
    filters from the horizontal filter. Normalizes to [0,1] before
    rotation to avoid interpolation artifacts, then rescales back.
    """
    immin = np.min(im)
    imrange = np.max(im) - immin
    if imrange == 0:
        return im.copy()
    im_n = (im - immin) / imrange                          # normalize to [0,1]
    im_n = transform.rotate(im_n, angle, order=3, resize=False)  # bicubic rotation
    return im_n * imrange + immin                          # rescale back


def _poolnew(in_: list, sigma=None):
    """
    Pool (smooth and downsample) a list of oriented energy maps.

    Ported from: AIM image_visual_clutter_utils.py → poolnew()

    Each map is upsampled then downsampled (expand→reduce) to smooth it.
    If sigma is provided, a Gaussian of that width is used instead of
    the default 5-tap binomial kernel.

    This pooling step is applied to squared orientation filter responses
    before computing orientation clutter in Feature Congestion.
    """
    if sigma is None:
        out = tuple(_reduce(_RRoverlapconvexpand(x)) for x in in_)
    else:
        kernel = _RRgaussfilter1D(round(2 * sigma), sigma)
        out = tuple(_reduce(_RRoverlapconvexpand(x, kernel), kernel) for x in in_)
    return out


def _orient_filtnew(pyr: np.ndarray, sigma: float = 16 / 14):
    """
    Apply 4-orientation second-derivative filters to an image.

    Ported from: AIM image_visual_clutter_utils.py → orient_filtnew()

    Constructs oriented energy filters at 0° (H), 90° (V), +45° (R),
    and -45° (L) by:
      1. Build 3 shifted Gaussians Ga(+σ), Gb(0), Gc(-σ) along one axis.
      2. Horizontal filter H = -Ga + 2·Gb - Gc (second derivative approx.).
      3. Vertical filter V = H^T.
      4. Diagonal filters: rotate Ga/Gb/Gc by ±45° and form the same
         second-derivative combination.

    These approximate oriented Gabor-like filters and capture edge
    energy at 4 orientations. The squared responses are later pooled
    and used to measure orientation clutter.

    Args:
        pyr: single pyramid level (2D array, typically L channel)
        sigma: filter scale (default 16/14 ≈ 1.14)
    Returns:
        (H_response, V_response, L_response, R_response) — 4 filtered maps
    """
    halfsupport = round(3 * sigma)
    sigy = sigma
    sigx = sigma
    # Build shifted Gaussians for second-derivative approximation
    gx = _RRgaussfilter1D(halfsupport, sigx)
    gy = _RRgaussfilter1D(halfsupport, sigy, sigma)         # shifted +σ
    Ga = _conv2(gx, gy.T); Ga = Ga / np.sum(Ga)
    gy = _RRgaussfilter1D(halfsupport, sigy)                # centered
    Gb = _conv2(gx, gy.T); Gb = Gb / np.sum(Gb)
    gy = _RRgaussfilter1D(halfsupport, sigy, -sigma)        # shifted -σ
    Gc = _conv2(gx, gy.T); Gc = Gc / np.sum(Gc)
    # Horizontal & Vertical second-derivative filters
    H = -Ga + 2 * Gb - Gc     # horizontal edge detector
    V = H.T                    # vertical edge detector (transpose)
    # +45° diagonal filter (rotate all three Gaussians by 45°)
    GGa = _imrotate(Ga, 45); GGa = GGa / np.sum(GGa)
    GGb = _imrotate(Gb, 45); GGb = GGb / np.sum(GGb)
    GGc = _imrotate(Gc, 45); GGc = GGc / np.sum(GGc)
    R = -GGa + 2 * GGb - GGc  # right-diagonal edge detector
    # -45° diagonal filter (rotate by -45°)
    GGa = _imrotate(Ga, -45); GGa = GGa / np.sum(GGa)
    GGb = _imrotate(Gb, -45); GGb = GGb / np.sum(GGb)
    GGc = _imrotate(Gc, -45); GGc = GGc / np.sum(GGc)
    L_filt = -GGa + 2 * GGb - GGc  # left-diagonal edge detector
    # Apply all 4 filters to the image
    return (_filt2(H, pyr), _filt2(V, pyr), _filt2(L_filt, pyr), _filt2(R, pyr))


def _entropy(x: np.ndarray, nbins: Optional[int] = None) -> float:
    """
    Shannon entropy of a 1D signal using uniform histogram binning.

    Ported from: AIM image_visual_clutter_utils.py → entropy()

    Formula:  H(x) = -Σ p(b) · ln(p(b))   (natural log, nats)

    The number of bins defaults to √N (where N = number of samples),
    which is a standard heuristic for entropy estimation.

    Used by Subband Entropy (Feature 4 / AIM m7) to compute the
    entropy of each steerable pyramid subband.

    Note: Uses natural log (ln), not log₂. The absolute scale doesn't
    matter because all subbands use the same base, and the final
    result is a relative measure.
    """
    nsamples = x.shape[0]
    # Degenerate inputs carry no information (entropy = 0) and would otherwise
    # produce nbins <= 1, i.e. np.histogram(x, bins=0), which raises.
    if nsamples <= 1:
        return 0.0
    if nbins is None:
        nbins = int(np.ceil(np.sqrt(nsamples)))  # heuristic: √N bins
    if nbins <= 1:
        return 0.0
    edges = np.histogram(x, bins=nbins - 1)[1]   # bin edge computation
    ref_hist = np.digitize(x, edges)              # assign samples to bins
    counts = np.zeros(len(edges))
    for el in ref_hist:
        counts[el - 1] += 1
    counts = counts / float(np.sum(counts))       # normalize to probabilities
    counts = counts[counts > 0]                   # remove zeros (0·log0 = 0)
    return float(-np.sum(counts * np.log(counts))) # H = -Σ p·ln(p)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 1: Shannon Entropy
# ═══════════════════════════════════════════════════════════════════════════

def shannon_entropy(image: np.ndarray) -> float:
    """
    Feature 1: Shannon Entropy — Global Information Density

    Measures the average information content (surprise) per pixel
    of the grayscale image using the discrete Shannon entropy.

    Formula:
        H(I) = - Σₖ p(k) · log₂(p(k))

    where p(k) is the probability of grayscale intensity k ∈ [0, 255],
    estimated from the 256-bin histogram of pixel values.

    Range:     [0, 8] bits (8 = max for 256 bins, i.e. uniform distribution)
    Intuition: A cluttered GUI with many different gray levels has high
               entropy; a mostly uniform screen has low entropy.

    Reference: Shannon, C. E. (1948). "A Mathematical Theory of
               Communication." Bell System Technical Journal, 27, 379–423.

    Related AIM metric: AIM m21 (JPEG file size) also captures information
    density, but we use the direct histogram approach for interpretability.
    """
    # Step 1: Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Step 2: Compute 256-bin histogram and normalize to probabilities
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()           # p(k) = count(k) / total_pixels
    hist = hist[hist > 0]              # exclude zero-probability bins

    # Step 3: H = -Σ p(k) * log₂(p(k))
    return float(-np.sum(hist * np.log2(hist)))


# ═══════════════════════════════════════════════════════════════════════════
# Feature 2: Edge Density (AIM m4)
# ═══════════════════════════════════════════════════════════════════════════

def edge_density(image: np.ndarray) -> float:
    """
    Feature 2: Edge Density — Structural Complexity

    Measures the proportion of pixels classified as edges by the
    Canny detector. More edges indicate more boundaries between
    visual elements, which increases parsing effort.

    Algorithm:
        1. Convert to grayscale.
        2. Apply Gaussian blur (σ = 2) to suppress noise.
        3. Run Canny edge detection with thresholds from AIM m4:
           - low  = 0.11 × 255 ≈ 28
           - high = 0.27 × 255 ≈ 69
        4. Edge density = #edge_pixels / #total_pixels

    Range:     [0, 1] (typically 0.01–0.15 for GUI screenshots)
    Intuition: A simple dashboard with few elements → low edge density;
               a dense information display → high edge density.

    AIM source: aim/backend/aim/metrics/m4_edge_density.py
    The threshold values 0.11/0.27 are the AIM "Desktop" defaults.
    """
    # Step 1: Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2: Gaussian pre-smoothing (σ = 2, kernel auto-sized)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)

    # Step 3: Canny with AIM m4 thresholds
    edges = cv2.Canny(blurred, int(0.11 * 255), int(0.27 * 255))

    # Step 4: Ratio of edge pixels to total pixels
    return float(np.count_nonzero(edges) / edges.size)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 3: Feature Congestion (AIM m8 — full multi-scale)
# ═══════════════════════════════════════════════════════════════════════════

def feature_congestion(image: np.ndarray) -> float:
    """
    Feature 3: Feature Congestion — Multi-Scale Visual Clutter

    Implements the full Feature Congestion clutter model from:
        Rosenholtz, R., Li, Y. & Nakano, L. (2007).
        "Measuring Visual Clutter." Journal of Vision, 7(2):17, 1–22.

    AIM source: aim/backend/aim/metrics/m8_feature_congestion.py

    The model decomposes visual clutter into three independent channels,
    each computed at multiple scales via Gaussian pyramids:

    A) COLOR CLUTTER (Lab space)
       - At each pyramid level, compute the 3×3 covariance matrix of
         the L*, a*, b* channels within a local Gaussian window (σ=3).
       - Color clutter = det(Cov)^(1/6)  (volume of the covariance ellipsoid).
       - Small noise floors (deltaL², deltaa², deltab²) ensure numerical
         stability in uniform regions.

    B) CONTRAST CLUTTER (Luminance)
       - At each pyramid level, compute local contrast via a center-
         surround DoG mechanism (see _RRcontrast1channel).
       - Contrast clutter = local standard deviation of the contrast
         map (i.e., how variable the contrast is across the neighborhood).

    C) ORIENTATION CLUTTER
       - At each pyramid level, apply 4 oriented second-derivative
         filters at 0°, 90°, ±45° (see _orient_filtnew).
       - Square the responses (energy), pool spatially.
       - Compute normalized H-V and diagonal opponency.
       - Orientation clutter = det(Cov(hv, dd))^(1/4).

    COLLAPSE ACROSS SCALES:
       Each clutter type yields one map per pyramid level. These are
       collapsed to the finest scale by upsampling coarser levels and
       taking the per-pixel maximum.

    FINAL COMBINATION:
       clutter = color/C_color + contrast/C_contrast + orient/C_orient
       where C_color=0.2088, C_contrast=0.0660, C_orient=0.0269 are
       normalization constants from Rosenholtz (2007), ensuring each
       channel contributes roughly equally.

    The scalar output is the spatial mean of the combined clutter map.

    Range:     [0, ∞) — typical GUI values: 1–10
    Intuition: Busy, colorful interfaces with many oriented edges have
               high feature congestion. Simple, monochromatic UIs are low.

    Parameters used (matching AIM m8 defaults):
        NUM_LEVELS = 3                  pyramid depth
        COLOR_COEF = 0.2088            normalization for color channel
        CONTRAST_COEF = 0.0660         normalization for contrast channel
        ORIENT_COEF = 0.0269           normalization for orientation channel
        OPP_ENERGY_FILTER_SCALE = 2.0  scale of orientation filters
        OPP_ENERGY_POOL_SCALE = 1.75   pooling σ for orientation energy
        OPP_ENERGY_NOISE = 1.0         noise floor for orientation total
        ORIENT_NOISE = 0.001           noise floor for orientation covariance
    """
    # --- Preprocessing: Convert to CIELab ---
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    lab = _rgb2lab(rgb).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # --- Hyperparameters (from AIM m8 source code) ---
    NUM_LEVELS = 3
    COLOR_COEF, CONTRAST_COEF, ORIENT_COEF = 0.2088, 0.0660, 0.0269
    OPP_ENERGY_FILTER_SCALE = 16 / 14 * 1.75  # ≈ 2.0
    OPP_ENERGY_POOL_SCALE = 1.75
    OPP_ENERGY_NOISE = 1.0
    ORIENT_NOISE = 0.001

    # --- Build Gaussian pyramids for L, a, b channels ---
    # pyrtools GaussianPyramid: each level is half the resolution.
    # pyr_coeffs is a dict with keys (level, 0).
    L_pyr = pt.pyramids.GaussianPyramid(L, height=NUM_LEVELS).pyr_coeffs
    a_pyr = pt.pyramids.GaussianPyramid(a, height=NUM_LEVELS).pyr_coeffs
    b_pyr = pt.pyramids.GaussianPyramid(b, height=NUM_LEVELS).pyr_coeffs

    # --- Collapse utility: merge multi-scale maps to finest resolution ---
    # Strategy: upsample coarser levels by 2^k, then take per-pixel max.
    # This ensures that clutter at ANY scale is captured.
    def collapse(clutter_levels):
        kernel_1d = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])  # Burt-Adelson
        kernel_2d = _conv2(kernel_1d, kernel_1d.T)               # 2D separable
        cmap = clutter_levels[0].copy()  # start with finest level
        for scale in range(1, len(clutter_levels)):
            ch = clutter_levels[scale]
            # Upsample 'scale' times (each time doubles resolution)
            for _ in range(scale, 0, -1):
                ch = pt.upConv(image=ch, filt=kernel_2d, edge_type="reflect1",
                               step=[2, 2], start=[0, 0])
            # Take per-pixel maximum (worst-case clutter across scales)
            csz = (min(cmap.shape[0], ch.shape[0]), min(cmap.shape[1], ch.shape[1]))
            cmap[:csz[0], :csz[1]] = np.maximum(cmap[:csz[0], :csz[1]], ch[:csz[0], :csz[1]])
        return cmap

    # =====================================================================
    # A) COLOR CLUTTER
    # =====================================================================
    # Compute the covariance matrix of (L, a, b) in a local Gaussian
    # window (σ=3, support=6 pixels). The clutter is det(Cov)^(1/6),
    # which is the volume of the 3D covariance ellipsoid raised to a
    # power that makes it linearly proportional to spread.
    #
    # Noise floors (delta² values) prevent 0-variance in flat regions:
    #   ΔL² = 0.0007² = 4.9e-7    (luminance is very clean)
    #   Δa² = 0.10²   = 0.01       (a-channel has more noise)
    #   Δb² = 0.05²   = 0.0025     (b-channel intermediate)
    deltaL2, deltaa2, deltab2 = 0.0007**2, 0.1**2, 0.05**2
    bigG = _RRgaussfilter1D(round(6), 3)  # Gaussian window, σ=3, support=±6
    color_levels = []
    for i in range(NUM_LEVELS):
        # Local means E[X] via border-corrected convolution
        DL = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, L_pyr[(i, 0)]))
        Da = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, a_pyr[(i, 0)]))
        Db = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, b_pyr[(i, 0)]))
        # Covariance entries: Cov(X,Y) = E[XY] - E[X]E[Y] + noise
        cov00 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, L_pyr[(i, 0)] ** 2)) - DL ** 2 + deltaL2
        cov01 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, L_pyr[(i, 0)] * a_pyr[(i, 0)])) - DL * Da
        cov02 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, L_pyr[(i, 0)] * b_pyr[(i, 0)])) - DL * Db
        cov11 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, a_pyr[(i, 0)] ** 2)) - Da ** 2 + deltaa2
        cov12 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, a_pyr[(i, 0)] * b_pyr[(i, 0)])) - Da * Db
        cov22 = _RRoverlapconv(bigG.T, _RRoverlapconv(bigG, b_pyr[(i, 0)] ** 2)) - Db ** 2 + deltab2
        # 3×3 determinant (Sarrus' rule / cofactor expansion)
        det = (cov00 * (cov11 * cov22 - cov12 ** 2)
               - cov01 * (cov01 * cov22 - cov12 * cov02)
               + cov02 * (cov01 * cov12 - cov11 * cov02))
        # Color clutter = det^(1/6) = (sqrt(det))^(1/3)
        color_levels.append(np.sqrt(np.maximum(det, 0)) ** (1 / 3))
    color_map = collapse(color_levels)

    # =====================================================================
    # B) CONTRAST CLUTTER
    # =====================================================================
    # Local contrast is computed via a center-surround DoG mechanism
    # (see _RRcontrast1channel). Contrast clutter is the local
    # standard deviation of this contrast map:
    #   contrast_clutter = sqrt(E[c²] - E[c]²)
    # High contrast clutter = contrast varies a lot spatially.
    contrast_raw = _RRcontrast1channel(L_pyr, 1)  # DoG_sigma=1
    bigG_c = _RRgaussfilter1D(round(6), 3)        # local averaging window
    contrast_levels = []
    for scale in range(NUM_LEVELS):
        meanD = _RRoverlapconv(bigG_c.T, _RRoverlapconv(bigG_c, contrast_raw[scale]))
        meanD2 = _RRoverlapconv(bigG_c.T, _RRoverlapconv(bigG_c, contrast_raw[scale] ** 2))
        # std = sqrt(E[X²] - E[X]²)
        contrast_levels.append(np.sqrt(np.abs(meanD2 - meanD ** 2)))
    contrast_map = collapse(contrast_levels)

    # =====================================================================
    # C) ORIENTATION CLUTTER
    # =====================================================================
    # Oriented edge energy is computed at 4 orientations (0°, 90°, ±45°)
    # using second-derivative Gaussian filters (_orient_filtnew).
    # The squared responses are pooled, then normalized to opponency:
    #   hv = (H² - V²) / (H² + V² + D1² + D2² + noise)
    #   dd = (D2² - D1²) / (same total)
    # Orientation clutter = det(Cov(hv, dd))^(1/4)
    # High orientation clutter = many conflicting edge directions.
    orient_levels = []
    bigG_o = _RRgaussfilter1D(round(8 * 3.5), 4 * 3.5)  # wider window for orientation
    for scale in range(NUM_LEVELS):
        # Apply 4 oriented filters to luminance pyramid level
        hvdd = _orient_filtnew(L_pyr[(scale, 0)], OPP_ENERGY_FILTER_SCALE)
        hvdd = [x ** 2 for x in hvdd]                    # squared energy
        hvdd = _poolnew(hvdd, OPP_ENERGY_POOL_SCALE)     # spatial pooling
        # Normalize to opponency channels
        total = hvdd[0] + hvdd[1] + hvdd[2] + hvdd[3] + OPP_ENERGY_NOISE
        hv = (hvdd[0] - hvdd[1]) / total   # horizontal vs vertical
        dd = (hvdd[3] - hvdd[2]) / total    # diagonal opponency
        # 2×2 covariance of (hv, dd)
        Dc = _RRoverlapconv(bigG_o.T, _RRoverlapconv(bigG_o, hv))
        Ds = _RRoverlapconv(bigG_o.T, _RRoverlapconv(bigG_o, dd))
        cov00 = _RRoverlapconv(bigG_o.T, _RRoverlapconv(bigG_o, hv ** 2)) - Dc ** 2 + ORIENT_NOISE
        cov01 = _RRoverlapconv(bigG_o.T, _RRoverlapconv(bigG_o, hv * dd)) - Dc * Ds
        cov11 = _RRoverlapconv(bigG_o.T, _RRoverlapconv(bigG_o, dd ** 2)) - Ds ** 2 + ORIENT_NOISE
        det = cov00 * cov11 - cov01 ** 2         # 2×2 determinant
        orient_levels.append(np.maximum(det, 0) ** (1 / 4))  # det^(1/4)
    orient_map = collapse(orient_levels)

    # =====================================================================
    # COMBINE: weighted sum of the three clutter channels
    # =====================================================================
    # Each channel is divided by its normalization coefficient so that
    # all three contribute roughly equally to the final score.
    clutter_map = color_map / COLOR_COEF + contrast_map / CONTRAST_COEF + orient_map / ORIENT_COEF

    # Return the spatial mean as a single scalar summary
    return float(np.mean(clutter_map))


# ═══════════════════════════════════════════════════════════════════════════
# Feature 4: Subband Entropy (AIM m7 — steerable pyramid)
# ═══════════════════════════════════════════════════════════════════════════

def subband_entropy(image: np.ndarray) -> float:
    """
    Feature 4: Subband Entropy — Spectral Redundancy Clutter

    Implements the Subband Entropy clutter model from:
        Rosenholtz, R., Li, Y. & Nakano, L. (2007).
        "Measuring Visual Clutter." Journal of Vision, 7(2):17, 1–22.

    AIM source: aim/backend/aim/metrics/m7_subband_entropy.py

    Algorithm:
      1. Convert image to CIELab.
      2. Decompose the L* channel using a Steerable Pyramid in the
         frequency domain (SteerablePyramidFreq from pyrtools):
         - 3 spatial frequency scales (W_LEVELS = 3)
         - 4 orientations per scale (WOR = 4: 0°, 45°, 90°, 135°)
         - Plus highpass and lowpass residuals
         This yields 3×4 + 2 = 14 subband images.
      3. For each subband, compute Shannon entropy H(subband) using
         the _entropy() helper (histogram-based, √N bins, natural log).
      4. Average the entropy across all subbands for luminance.
      5. Repeat for a* and b* channels with a reduced weight:
         WGHT_CHROM = 0.0625 = 1/16 (chrominance contributes less
         to visual clutter than luminance).
      6. Final: clutter_SE = (H_L + w·H_a + w·H_b) / (1 + 2w)

    Intuition:
      A predictable image (e.g., smooth gradient) has low subband entropy
      (the pyramid coefficients are mostly zero → peaked histogram).
      A complex image with many textures at various scales and orientations
      has high subband entropy (spread-out histograms).

    Range:     [0, ∞) — typical GUI values: 1–5
    Chrominance channels that are nearly constant (range < 0.008) are
    zeroed out to avoid spurious entropy from quantization noise.

    Parameters (matching AIM m7 defaults):
        W_LEVELS = 3            number of pyramid scales
        WOR = 4                 number of orientations per scale
        WGHT_CHROM = 0.0625     chrominance weight (1/16)
        ZERO_THRESHOLD = 0.008  min range to consider a channel non-zero
    """
    W_LEVELS = 3       # number of spatial frequency scales
    WOR = 4            # number of orientations per scale
    WGHT_CHROM = 0.0625  # chrominance weight = 1/16
    ZERO_THRESHOLD = 0.008  # below this range, channel is treated as zero

    # Step 1: Convert to CIELab
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    lab = _rgb2lab(rgb).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    def band_entropy(map_: np.ndarray) -> List[float]:
        """
        Decompose a single channel into a steerable pyramid and compute
        Shannon entropy for each subband.

        pyrtools.SteerablePyramidFreq:
          - Frequency-domain implementation (FFT-based)
          - order = WOR - 1 = 3 (order of the steerable filters)
          - height = W_LEVELS = 3
          - pyr_coeffs: dict with keys like (scale, orientation),
            'residual_highpass', 'residual_lowpass'
        """
        pyr = pt.pyramids.SteerablePyramidFreq(map_, height=W_LEVELS, order=WOR - 1)
        S = pyr.pyr_coeffs
        # Compute entropy for each subband (flatten 2D → 1D for histogram)
        return [_entropy(S[ind].ravel()) for ind in S.keys()]

    # Step 2: Luminance subband entropy (main contributor)
    clutter_se = float(np.mean(band_entropy(L)))

    # Step 3: Chrominance subband entropy (weighted by 1/16)
    for ch in [a, b]:
        # Skip near-constant channels (avoid entropy from noise)
        if np.max(ch) - np.min(ch) < ZERO_THRESHOLD:
            ch = np.zeros_like(ch)
        clutter_se += WGHT_CHROM * float(np.mean(band_entropy(ch)))

    # Step 4: Normalize by total weight (1 + 2 × 0.0625 = 1.125)
    clutter_se /= (1 + 2 * WGHT_CHROM)
    return clutter_se


# ═══════════════════════════════════════════════════════════════════════════
# Feature 5: Layout Symmetry (custom)
# ═══════════════════════════════════════════════════════════════════════════

def layout_symmetry(image: np.ndarray) -> float:
    """
    Feature 5: Layout Symmetry — Axial Balance

    Measures the degree of vertical and horizontal mirror symmetry
    in the GUI layout using normalized cross-correlation (NCC).

    Algorithm:
      1. Convert to grayscale.
      2. Zero-mean the image: I' = I - mean(I)
      3. Compute vertical symmetry:
         NCC_v = Σ(I' · fliplr(I')) / Σ(I'²)
      4. Compute horizontal symmetry:
         NCC_h = Σ(I' · flipud(I')) / Σ(I'²)
      5. Symmetry = (NCC_v + NCC_h) / 2, clipped to [0, 1]

    Where fliplr = left-right mirror, flipud = top-bottom mirror.
    NCC = 1 means perfect symmetry; NCC = 0 means no correlation.

    Range:     [0, 1]
    Intuition: Symmetric layouts reduce visual search time because
               users can predict where elements are. Asymmetric
               layouts require more exploration.

    Reference: Miniukovich, A. & De Angeli, A. (2015).
               "Computation of Interface Aesthetics."
               CHI '15, pp. 1163–1172.
    """
    # Step 1: Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    # Step 2: Zero-mean the image
    gray_norm = gray - gray.mean()
    denom = np.sqrt(np.sum(gray_norm ** 2))  # L2 norm
    if denom == 0:
        return 1.0  # perfectly uniform image = perfectly symmetric

    # Step 3: Vertical symmetry (left-right mirror)
    v_sym = np.sum(gray_norm * np.fliplr(gray_norm)) / (denom ** 2)

    # Step 4: Horizontal symmetry (top-bottom mirror)
    h_sym = np.sum(gray_norm * np.flipud(gray_norm)) / (denom ** 2)

    # Step 5: Average and clip to [0, 1]
    return float(np.clip((v_sym + h_sym) / 2.0, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Feature 6: Chromatic Coherence (AIM m13 + m15 + m16 combined)
# ═══════════════════════════════════════════════════════════════════════════

def chromatic_coherence(image: np.ndarray) -> float:
    """
    Feature 6: Chromatic Coherence — Color Palette Fragmentation

    Combines four sub-metrics from AIM into a single [0,1] score:

    A) Luminance Standard Deviation (AIM m13)
       - lum = 0.2126·R + 0.7152·G + 0.0722·B  (Rec.709 luma)
       - lum_std = std(lum)
       - Normalized by /128 (half of max luma range)

    B) Colorfulness (AIM m15 / Hasler & Süsstrunk 2003)
       - rg = R - G            (red-green opponent)
       - yb = 0.5(R+G) - B    (yellow-blue opponent)
       - C = sqrt(σ_rg² + σ_yb²) + 0.3 × sqrt(μ_rg² + μ_yb²)
       - Normalized by /150

    C) Hue Circular Standard Deviation (AIM m16)
       - Hue is a circular variable (0°–360°), so we use the
         Mardia-Jupp circular std: σ_circ = sqrt(-2 · ln(R̄))
         where R̄ = |mean(e^{i·θ})| is the mean resultant length.
       - Normalized by /2.5

    D) Saturation Standard Deviation (AIM m16)
       - sat = S channel from HSV, in [0,1]
       - Normalized by /0.5

    Final: average of the 4 normalized sub-metrics.

    Range:     [0, 1]
    Intuition: A coherent color palette (few harmonious colors) scores
               low. A fragmented palette (many unrelated colors, widely
               varying brightness and saturation) scores high.

    References:
      - Hasler, D. & Süsstrunk, S. E. (2003). "Measuring Colorfulness
        in Natural Images." SPIE 5007.
      - Miniukovich & De Angeli (2015).
      - AIM metrics m13 (luminance_std), m15 (colorfulness), m16 (HSV_avg/std)
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # --- A) Luminance std (AIM m13: Rec.709 luma weights) ---
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    lum_std = float(np.std(luminance))

    # --- B) Colorfulness (AIM m15: Hasler & Süsstrunk formula) ---
    rg = R - G                          # red-green opponent channel
    yb = 0.5 * (R + G) - B             # yellow-blue opponent channel
    rgyb_std = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)    # combined std
    rgyb_mean = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2) # combined mean
    colorfulness_val = float(rgyb_std + 0.3 * rgyb_mean)     # C formula

    # --- C) & D) HSV statistics (AIM m16) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hue_deg = hsv[:, :, 0] * 2.0       # OpenCV hue is [0,180] → [0,360]°
    sat = hsv[:, :, 1] / 255.0          # saturation to [0,1]
    hue_rad = np.deg2rad(hue_deg)       # convert to radians for circular stats
    # Mean resultant length R̄ = |mean(e^{iθ})|
    R_bar = np.sqrt(np.mean(np.sin(hue_rad)) ** 2 + np.mean(np.cos(hue_rad)) ** 2)
    # Circular std (Mardia & Jupp): σ = sqrt(-2 · ln(R̄))
    hue_circular_std = float(np.sqrt(-2.0 * np.log(max(R_bar, 1e-10))))
    sat_std = float(np.std(sat))

    # --- Combine: normalize each sub-metric to [0,1], then average ---
    norm_lum = min(lum_std / 128.0, 1.0)          # 128 = half of 256
    norm_color = min(colorfulness_val / 150.0, 1.0)  # 150 = empirical max
    norm_hue = min(hue_circular_std / 2.5, 1.0)   # 2.5 = near-uniform dist
    norm_sat = min(sat_std / 0.5, 1.0)             # 0.5 = high sat variance
    return float((norm_lum + norm_color + norm_hue + norm_sat) / 4.0)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 7: Visual Hierarchy (AIM m5 + custom size gradient)
# ═══════════════════════════════════════════════════════════════════════════

def visual_hierarchy(image: np.ndarray) -> float:
    """
    Feature 7: Visual Hierarchy — Contrast Gradients & Size Dominance

    Measures how clearly the GUI is organized into visual layers.
    A strong hierarchy means users can quickly identify the most
    important elements. Combines two sub-metrics:

    A) Figure-Ground Contrast (inspired by AIM m5)
       - Apply Canny at 7 increasing threshold levels (0.1–0.7).
       - Count edge pixels at each level.
       - Compute a weighted decay score: edges that persist to high
         thresholds indicate strong contrast boundaries.
       - Formula: fg = Σₖ (count[k] - count[k+1]) × (k/6)
                       / (count[0] - count[-1])
       - A UI where all edges vanish at low thresholds has weak
         figure-ground separation (fg ≈ 0); a UI whose edges persist
         to the highest thresholds has strong separation (fg ≈ 1).

    B) Size Gradient
       - Binarize with Otsu's threshold.
       - Find connected components.
       - Compute the area ratio of the top-3 largest components
         to the total image area.
       - If a few large components dominate, the hierarchy is clear.

    Final: (fg_contrast + size_gradient) / 2, clipped to [0, 1].

    Range:     [0, 1]
    Intuition: Clear hierarchy = less cognitive effort to find targets.
               Flat/cluttered layouts score low.

    Reference: Tuch, A. N. et al. (2009). "The Role of Visual Complexity
               and Prototypicality Regarding First Impression of Websites."
               Interacting with Computers, 24(1–2), 48–59.
    AIM source (figure-ground): aim/backend/aim/metrics/m5_contour_density.py
    """
    # Step 1: Grayscale conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # =================================================================
    # A) Figure-Ground Contrast (multi-threshold edge analysis)
    # =================================================================
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 7 Canny thresholds
    edge_counts = []
    for level in levels:
        high = int(level * 255)         # upper Canny threshold
        low = int(0.4 * high)           # lower = 40% of upper (standard ratio)
        edges = cv2.Canny(blurred, low, max(high, 1))
        edge_counts.append(np.count_nonzero(edges))
    # Weighted decay: edges that survive to HIGHER thresholds (larger k, i.e.
    # stronger contrast boundaries) are weighted more. Edges that drop out at
    # step k are weighted by k/6, so the weakest edges (k=0) contribute nothing
    # and the strongest surviving edges dominate the score.
    denom = edge_counts[0] - edge_counts[-1]
    if denom > 0:
        fg_contrast = sum(
            (edge_counts[k] - edge_counts[k + 1]) * (k / 6) for k in range(6)
        ) / denom
    else:
        # All thresholds give the same count. Two degenerate cases:
        #  - edges persist even through the highest threshold -> strongest
        #    possible figure-ground separation -> fg = 1.0
        #  - there are no edges at all -> no separation -> fg = 0.0
        fg_contrast = 1.0 if edge_counts[-1] > 0 else 0.0

    # =================================================================
    # B) Size Gradient (dominant component area ratio)
    # =================================================================
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
        areas_sorted = np.sort(areas)[::-1]   # descending
        total_area = gray.shape[0] * gray.shape[1]
        top_k = min(3, len(areas_sorted))     # top-3 components
        size_gradient = float(areas_sorted[:top_k].sum() / total_area)
    else:
        size_gradient = 1.0  # single blob = trivially dominant

    # Combine and clip
    return float((np.clip(fg_contrast, 0, 1) + np.clip(size_gradient, 0, 1)) / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 8: Interactive Element Density (custom contour-based)
# ═══════════════════════════════════════════════════════════════════════════

def interactive_element_density(image: np.ndarray) -> float:
    """
    Feature 8: Interactive Element Density — UI Control Density

    Estimates the number of interactive UI elements (buttons, sliders,
    text fields, icons) per 100×100 pixel area using a contour-based
    heuristic detector.

    Algorithm:
      1. Convert to grayscale.
      2. Gaussian blur (5×5 kernel) to suppress text/noise.
      3. Canny edge detection (thresholds 50/150).
      4. Morphological closing (5×5 rect kernel) to merge nearby edges
         into closed contour regions.
      5. Find external contours.
      6. Filter contours by:
         - Size: area in (0.05% of image, 25% of image)
           Too small = noise; too large = background region.
         - Aspect ratio: max(w,h)/min(w,h) < 10
           Rejects very elongated shapes (probably borders/dividers).
         - Solidity: contour_area / bounding_rect_area > 0.3
           Rejects very irregular shapes.
      7. Density = qualifying_contours / (image_area / 10000)
         This normalizes to "elements per 100×100px patch".

    Range:     [0, ∞) — typical GUI values: 0.01–0.5
    Intuition: More interactive elements = more choices = higher
               decisional complexity (Hick's Law).

    Limitations:
      - This is a heuristic, not a trained UI element detector.
      - For production use, a YOLO/Faster R-CNN model trained on
        UI screenshots (e.g., RICO dataset) would be more accurate.
      - Sufficient for Stage 1 as a proxy; can be replaced in Stage 2.
    """
    # Step 1: Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2: Blur to suppress text and fine noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Morphological closing — bridges small gaps between edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 5: Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Filter contours to plausible UI elements
    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * 0.0005   # 0.05% of image = smallest button
    max_area = img_area * 0.25     # 25% of image = too large
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)   # aspect ratio
            solidity = area / max(w * h, 1)            # fill ratio
            if aspect < 10 and solidity > 0.3:
                count += 1  # plausible UI element

    # Step 7: Normalize to elements per 100×100px patch
    return float(count / (img_area / 10000.0))


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

# The 8 features in project-plan order
FEATURE_KEYS = [
    "shannon_entropy",
    "edge_density",
    "feature_congestion",
    "subband_entropy",
    "layout_symmetry",
    "chromatic_coherence",
    "visual_hierarchy",
    "interactive_element_density",
]


def compute_complexity_vector(image_path: str) -> Dict[str, float]:
    """
    Compute the Stage 1 visual complexity vector v in R^8 for one image.
    Returns a dictionary with all 8 feature values.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    print(f"  Processing: {os.path.basename(image_path)} "
          f"({image.shape[1]}x{image.shape[0]} px)")

    results = {}
    print("    [1/8] Shannon Entropy...")
    results["shannon_entropy"] = shannon_entropy(image)
    print("    [2/8] Edge Density...")
    results["edge_density"] = edge_density(image)
    print("    [3/8] Feature Congestion (multi-scale, may take a moment)...")
    results["feature_congestion"] = feature_congestion(image)
    print("    [4/8] Subband Entropy (steerable pyramid)...")
    results["subband_entropy"] = subband_entropy(image)
    print("    [5/8] Layout Symmetry...")
    results["layout_symmetry"] = layout_symmetry(image)
    print("    [6/8] Chromatic Coherence...")
    results["chromatic_coherence"] = chromatic_coherence(image)
    print("    [7/8] Visual Hierarchy...")
    results["visual_hierarchy"] = visual_hierarchy(image)
    print("    [8/8] Interactive Element Density...")
    results["interactive_element_density"] = interactive_element_density(image)

    return results


def process_directory(input_dir: str, output_path: str) -> None:
    """Process all images in a directory and save results as CSV."""
    input_path = Path(input_dir)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in extensions)

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\nStage 1 - Visual Complexity Extraction (v in R^8)")
    print(f"{'=' * 55}")
    print(f"Input:  {input_dir} ({len(image_files)} images)")
    print(f"Output: {output_path}\n")

    all_results = []
    for img_file in image_files:
        try:
            result = compute_complexity_vector(str(img_file))
            result["filename"] = img_file.name
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {img_file.name} - {e}")

    if not all_results:
        print("No images were processed successfully.")
        return

    fieldnames = ["filename"] + FEATURE_KEYS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n{'=' * 55}")
    print(f"Done! {len(all_results)} images -> {output_path}")
    print(f"\n{'Feature':<32} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print("-" * 64)
    for feat in FEATURE_KEYS:
        vals = [r[feat] for r in all_results]
        print(f"{feat:<32} {min(vals):>10.4f} {max(vals):>10.4f} {np.mean(vals):>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 - Visual Complexity Extraction (v in R^8)"
    )
    parser.add_argument("--image", "-i", type=str, help="Path to a single image")
    parser.add_argument("--dir", "-d", type=str, default="data/screenshots",
                        help="Directory with screenshots (default: data/screenshots/)")
    parser.add_argument("--output", "-o", type=str, default="data/results/complexity_vectors.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    if args.image:
        result = compute_complexity_vector(args.image)
        print(f"\nVisual Complexity Vector v in R^8 for: {args.image}")
        print("-" * 55)
        for key in FEATURE_KEYS:
            print(f"  {key:<32} {result[key]:.4f}")
    else:
        process_directory(args.dir, args.output)


if __name__ == "__main__":
    main()
