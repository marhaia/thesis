# Stage 1 — Visual Complexity & Saliency Extraction

## Documentation & Technical Reference

**Technical scope:** *Frozen Stage-1 Screenshot Pipeline + Qualitative Stage-2 v1 Scenario Proxy*
**Author:** Hannah Mueller (Q682780)  
**Module:** `stage1/visual_complexity.py` + `saliency/umsi_model.py`  
**Version:** 3.2 (19.08.2026)

> **Current P7 claim policy (authoritative):** A complete score-bearing
> Stage-1 analysis exposes the task-independent float32 vector
> `x = [v8 | s5 | h6] ∈ ℝ¹⁹` and the screenshot-only
> `layout.experimental_complexity_index`. HCEye supplies project-specific,
> unvalidated proxy rules; it does not calibrate a cognitive-load measurement.
> Stage 2 v1 accepts only `task_type` and `time_pressure` and attaches a
> deterministic qualitative direction (`lower` / `baseline` / `higher`). It is
> non-score-bearing, has no numeric modifier, and never changes x19 or the
> layout index. Personality inputs and trained regressors are excluded. UMSI++
> provides a numeric saliency map
> and an internal six-value auxiliary head whose semantic label order is not
> verified. Search times, fixation counts, and paths are model simulations, not
> observed user data. Historical development notes below are subordinate to
> this policy wherever wording conflicts.

---

## Change Log

Historical terminology below records implementation history and does not
override the current claim policy at the top of this document.

| Date | Version | Change |
|------|---------|--------|
| 04.05.2026 | 1.0 | Implemented Stage-1 feature vector v∈ℝ⁸ (eight visual-complexity metrics) |
| 04.05.2026 | 1.0 | Added Flask web UI and API on port 5001 |
| 04.05.2026 | 1.0 | Added inline comments and `DOCUMENTATION.md` |
| 05.05.2026 | 1.5 | Cloned the UEyes repository and inspected the UMSI++ architecture |
| 06.05.2026 | 2.0 | Implemented the UMSI++ TensorFlow 2 / Keras 3 port for the recorded M4 CPU runtime |
| 06.05.2026 | 2.0 | Implemented saliency-feature extraction s∈ℝ⁵ |
| 06.05.2026 | 2.0 | Added `/api/saliency` |
| 06.05.2026 | 2.0 | Loaded the UEyes/UMSI++ checkpoint |
| 06.05.2026 | 2.0 | Downloaded PathGAN++ and DeepGaze++ weights; neither model is part of the current pipeline |
| 06.05.2026 | 2.5 | Implemented the project Jokinen-based visual-search simulation (`cognitive/jokinen_model.py`) |
| 06.05.2026 | 2.5 | Implemented the project UI-element detector (`cognitive/element_detector.py`) |
| 06.05.2026 | 2.5 | Added project-specific UMSI++ activation to the Jokinen-based simulation |
| 06.05.2026 | 2.5 | Added `/api/search-time`; current policy classifies its output as model simulation, not measured search time |
| 06.05.2026 | 2.5 | Added Monte Carlo simulation (50–500 trials per element, EMMA, VSTM) |
| 26.05.2026 | 2.6 | Historical release used HCEye rules plus descriptor/profile modifiers on `/api/cognitive-load`; those modifiers are retired |
| 26.05.2026 | 2.6 | Historical release exposed a trained Stage-2 model as an experimental opt-in; it is excluded from Stage 2 v1 |
| 16.07.2026 | 2.7 | Revised web UI presentation and copy without changing scoring |
| 16.07.2026 | 2.7 | Expanded comparison-history labels and added selected-row CSV export |
| 16.07.2026 | 2.7 | Revised navigation, contact placement, and repository-link placeholders |
| 16.07.2026 | 2.7 | Simplified display-preset wording and removed unused emoji metadata |
| 16.07.2026 | 2.7 | Audited displayed feature references and added dataset-shift context |
| 11.08.2026 | 3.0 | Reconciled x19, task independence, canonical scale, HCEye/UMSI boundaries, simulated outputs, formats, and validation status |
| 18.08.2026 | 3.1 | Established the qualitative, non-score-bearing Stage-2 v1 scenario boundary and excluded personality, ML, and numeric task/profile modifiers |
| 19.08.2026 | 3.2 | Finalized English claim boundaries, source provenance, and release-facing documentation without changing score-bearing computation |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The 8 Features in Detail](#3-the-8-features-in-detail)
   - [F1: Shannon Entropy](#f1-shannon-entropy)
   - [F2: Edge Density](#f2-edge-density)
   - [F3: Feature Congestion](#f3-feature-congestion)
   - [F4: Subband Entropy](#f4-subband-entropy)
   - [F5: Layout Symmetry](#f5-layout-symmetry)
   - [F6: Chromatic Coherence](#f6-chromatic-coherence)
   - [F7: Visual Hierarchy](#f7-visual-hierarchy)
   - [F8: Interactive Element Density](#f8-interactive-element-density)
4. [AIM Source Mapping](#4-aim-source-mapping)
5. [Utility Functions Reference](#5-utility-functions-reference)
6. [Saliency Pipeline (UMSI++)](#6-saliency-pipeline-umsi)
7. [Optional Jokinen-Based Visual-Search Diagnostic](#7-optional-jokinen-based-visual-search-diagnostic)
8. [Dependencies](#8-dependencies)
9. [Usage](#9-usage)
10. [Example Output](#10-example-output)
11. [Normalization Ranges](#11-normalization-ranges)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Overview

Stage 1 extracts `v ∈ ℝ⁸`, mandatory score-bearing saliency features
`s ∈ ℝ⁵`, and project-specific HCEye-derived proxy features `h ∈ ℝ⁶`.
Together they form the sole public, task-independent boundary
`x = [v8 | s5 | h6] ∈ ℝ¹⁹` (`float32`).

The screenshot-only **Experimental Layout-Complexity Index** is an exploratory
project-specific heuristic, not a validated cognitive-load measurement. Stage
2 v1 attaches only a deterministic qualitative scenario proxy from
`task_type` and `time_pressure`. The proxy has `score_bearing=false` and
`numeric_modifier=null`; it never enters x19 or alters
`layout.experimental_complexity_index`. Personality/Big-Five inputs, trained
regressors, and numeric task/profile modifiers are excluded from the active
pipeline.

The Jokinen 2020 implementation supplies model-estimated per-element search
times, fixation counts, and target-driven paths as separate diagnostics. These
values are simulations, not observed eye tracking or validated human
performance.

```
Input:  GUI screenshot (PNG/JPG/JPEG/BMP/TIFF)
        ↓
   ┌──────────────────────────────────────────────────────────┐
   │  Stage 1a: visual_complexity.py          [04.05.2026]    │
   │                                                          │
   │  f₁  Shannon Entropy                                    │
   │  f₂  Edge Density                                       │
   │  f₃  Feature Congestion (AIM m8)                        │
   │  f₄  Subband Entropy   (AIM m7)                         │
   │  f₅  Layout Symmetry                                    │
   │  f₆  Chromatic Coherence                                │
   │  f₇  Visual Hierarchy                                   │
   │  f₈  Interactive Element Density                         │
   │                                                          │
   │  → v = [f₁, …, f₈] ∈ ℝ⁸                               │
   └──────────────────────────────────────────────────────────┘
        ↓
   ┌──────────────────────────────────────────────────────────┐
   │  Stage 1b: saliency/umsi_model.py        [06.05.2026]    │
   │                                                          │
   │  UMSI++ (Jiang et al., CHI 2023)                         │
   │  UEyes source: 62 participants, 1,980 screenshots,       │
   │  four UI types (webpage, desktop, mobile, poster)        │
   │                                                          │
   │  Outputs:                                                │
   │    • Saliency heatmap (512×512 → original resolution)   │
   │    • Internal numeric auxiliary head (6 values; no       │
   │      verified semantic label order)                      │
   │                                                          │
   │  Derived features (saliency/saliency_features.py):       │
   │    s₁  Saliency Dispersion                               │
   │    s₂  Saliency Peak Count                               │
   │    s₃  Saliency Center Bias                              │
   │    s₄  Saliency Entropy                                  │
   │    s₅  Saliency Coverage                                 │
   │                                                          │
   │  → s = [s₁, …, s₅] ∈ ℝ⁵                               │
   └──────────────────────────────────────────────────────────┘
        ↓
   ┌──────────────────────────────────────────────────────────┐
   │  Separate diagnostic: cognitive/jokinen_model.py         │
   │                                                          │
   │  Jokinen 2020 Adaptive Feature Guidance                  │
   │  (IJHCS, 136, 102376)                                   │
   │                                                          │
   │  Input: detected UI elements + UMSI++ saliency map       │
   │  Process: Monte Carlo search simulation (EMMA + VSTM)    │
   │  Output:                                                 │
   │    • Model-estimated search time (seconds)               │
   │    • Model-estimated fixation count                      │
   │    • Experimental search-difficulty rating               │
   │                                                          │
   │  → c = [mean_time, max_time, std_time, difficulty]       │
   └──────────────────────────────────────────────────────────┘
        ↓
Output: x = [v8 | s5 | h6] ∈ ℝ¹⁹ + separate model diagnostics
```

---

## 2. Architecture

### File Structure

```
.
├── stage1/
│   ├── visual_complexity.py           # Screenshot-derived v8 block
│   ├── app.py                         # Flask API and x19 assembly
│   ├── reproducibility.py             # Runtime and artifact verification
│   ├── reference_pack_manifest.json   # Pinned reference identities
│   ├── DOCUMENTATION.md               # This technical reference
│   ├── ui/index.html                  # Standard web interface
│   └── data/results/                  # Pinned normalization artifacts
├── saliency/
│   ├── umsi_model.py                  # UMSI++ TensorFlow 2 port
│   ├── postprocessing.py              # Finite min-max normalization policy
│   └── saliency_features.py           # Numeric s5 extraction
├── hceye/                             # Project-specific h6 proxy rules
├── cognitive/                         # Optional model-simulated diagnostics
├── stage2/
│   ├── scenario_proxy.py              # Non-score-bearing scenario proxy
│   └── coherence_check.py             # Exploratory cross-signal review
├── scripts/                           # Active evidence/replay utilities
└── tests/                             # Unit, regression, contract, and gates
```

The upstream UEyes and AIM repositories, virtual environments, downloaded
model artifacts, literature notes, and user-study materials are not part of
the active source-code architecture shown above. Their required technical
identities or source relationships are recorded separately where applicable.

### Processing Flow

**Stage 1a (Visual Complexity):**
1. Image is loaded via OpenCV (`cv2.imread`) in BGR format.
2. The image is resized once, aspect-ratio preserving, to the project-specific
   canonical analysis resolution (1280 px long side). The eight visual features
   and score-driving layout/OCR measurements use this canonical analysis path.
3. Features are computed sequentially (F1→F8).
4. Results are returned as a Python dict: `{"shannon_entropy": 6.64, ...}`
5. The Flask API serializes this to JSON for the web UI.

**Stage 1b (Saliency; added 06.05.2026):**
1. Image is loaded as BGR, resized with aspect-ratio padding to 256×256.
2. VGG mean subtraction applied (B: 103.939, G: 116.779, R: 123.68).
3. Forward pass through UMSI++ model → 512×512 heatmap + numeric six-value
   auxiliary softmax head. No semantic class-label order is asserted.
4. Heatmap is unpadded and resized to original image resolution.
5. Five scalar features are extracted from the normalized heatmap.
6. Results returned via `/api/saliency` endpoint.

---

## 3. The 8 Features in Detail

### F1: Shannon Entropy

| Property | Value |
|----------|-------|
| **Function** | `shannon_entropy(image)` |
| **Output range** | [0, 8] bits |
| **Unit** | bits per pixel |
| **Higher means** | Greater grayscale-intensity variability under this metric; any attention interpretation is an unvalidated project hypothesis |
| **Reference** | Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27, 379–423. |

**Algorithm:**

1. Convert to grayscale (8-bit, 256 levels).
2. Compute the 256-bin histogram of pixel intensities.
3. Normalize to a probability distribution: `p(k) = count(k) / total_pixels`.
4. Compute Shannon entropy:

$$H(I) = -\sum_{k=0}^{255} p(k) \cdot \log_2 p(k)$$

**Interpretation:**
- H = 0: All pixels have the same intensity (perfectly uniform).
- H = 8: All 256 possible intensities occur with equal probability (maximum entropy for 8-bit images).
- Screenshots in the project's authorized UEyes GUI reference subset commonly
  fall within approximately 5.5–7.5 bits. This is a corpus-relative observation,
  not a universal range for typical GUIs.

---

### F2: Edge Density

| Property | Value |
|----------|-------|
| **Function** | `edge_density(image)` |
| **Output range** | [0, 1] |
| **Unit** | proportion of edge pixels |
| **Higher means** | More structural boundaries under this metric; increased human parsing effort is an unvalidated project hypothesis |
| **AIM source** | `m4_edge_density.py` |

**Algorithm:**

1. Convert to grayscale.
2. Gaussian blur with σ = 2 (noise suppression).
3. Canny edge detection with AIM "Desktop" thresholds:
   - Low threshold: `0.11 × 255 ≈ 28`
   - High threshold: `0.27 × 255 ≈ 69`
4. Count edge pixels and divide by total pixels.

$$\text{edge\_density} = \frac{|\{(x,y) : \text{Canny}(x,y) = 255\}|}{W \times H}$$

**Why these thresholds?**
The 0.11/0.27 values are AIM m4 defaults adopted by this implementation. They
are deterministic engineering parameters, not independently calibrated
human-performance or universal desktop-UI thresholds.

---

### F3: Feature Congestion

| Property | Value |
|----------|-------|
| **Function** | `feature_congestion(image)` |
| **Output range** | [0, ∞), typically 1–10 for GUIs |
| **Unit** | arbitrary (combined clutter units) |
| **Higher means** | More visual clutter from color, contrast, and orientation |
| **AIM source** | `m8_feature_congestion.py` |
| **Reference** | Rosenholtz, R., Li, Y. & Nakano, L. (2007). "Measuring Visual Clutter." *Journal of Vision*, 7(2):17, 1–22. |

**This is the most complex feature.** It decomposes visual clutter into three independent channels, each computed at 3 Gaussian pyramid scales:

#### A) Color Clutter

Computed in CIE L\*a\*b\* space (perceptually uniform). At each pyramid level:

1. Compute local means `E[L]`, `E[a]`, `E[b]` within a Gaussian window (σ=3).
2. Compute the full 3×3 covariance matrix of (L, a, b):

$$\text{Cov}(X,Y) = E[XY] - E[X] \cdot E[Y]$$

3. Add noise floors to diagonal elements (δL² = 0.0007², δa² = 0.1², δb² = 0.05²) for numerical stability.
4. Color clutter = det(Cov)^(1/6) — the "volume" of the 3D covariance ellipsoid:

$$C_{\text{color}} = \left(\det(\Sigma_{3\times3})\right)^{1/6}$$

#### B) Contrast Clutter

1. Compute local contrast at each pyramid level using a center-surround Difference-of-Gaussians (DoG):
   - Inner Gaussian: σᵢ = 0.71 × σ (fine detail)
   - Outer Gaussian: σₒ = 1.14 × σ (surround)
   - Contrast = |inner − outer| (mimics retinal ganglion cells)
2. Contrast clutter = local standard deviation of the contrast map:

$$C_{\text{contrast}} = \sqrt{E[c^2] - E[c]^2}$$

#### C) Orientation Clutter

1. Apply 4 oriented second-derivative Gaussian filters at 0°, 90°, ±45° (see `_orient_filtnew`).
2. Square responses (energy), spatially pool.
3. Compute normalized opponency channels:
   - `hv = (H² − V²) / (H² + V² + D1² + D2² + noise)` (horizontal vs. vertical)
   - `dd = (D2² − D1²) / (same total)` (diagonal opponency)
4. Orientation clutter = det(Cov(hv, dd))^(1/4):

$$C_{\text{orient}} = \left(\det(\Sigma_{2\times2})\right)^{1/4}$$

#### Collapse Across Scales

Each clutter type produces one map per pyramid level. These are collapsed to the finest resolution by upsampling coarser levels and taking the **per-pixel maximum** (worst-case clutter at any scale).

#### Final Combination

$$\text{FC} = \frac{C_{\text{color}}}{0.2088} + \frac{C_{\text{contrast}}}{0.0660} + \frac{C_{\text{orient}}}{0.0269}$$

The normalization constants (0.2088, 0.0660, 0.0269) are from Rosenholtz et al. (2007) and ensure each channel contributes roughly equally. The output is the **spatial mean** of this combined map.

---

### F4: Subband Entropy

| Property | Value |
|----------|-------|
| **Function** | `subband_entropy(image)` |
| **Output range** | [0, ∞), typically 1–5 for GUIs |
| **Unit** | nats (natural log entropy) |
| **Higher means** | More unpredictable spatial frequency content |
| **AIM source** | `m7_subband_entropy.py` |
| **Reference** | Rosenholtz, R., Li, Y. & Nakano, L. (2007). "Measuring Visual Clutter." *Journal of Vision*, 7(2):17, 1–22. |

**Algorithm:**

1. Convert to CIE L\*a\*b\*.
2. Decompose each channel using a **Steerable Pyramid** (frequency-domain implementation via `pyrtools.SteerablePyramidFreq`):
   - 3 spatial frequency scales (octave spacing)
   - 4 orientations per scale (0°, 45°, 90°, 135°)
   - Plus highpass and lowpass residuals
   - Total: 3×4 + 2 = **14 subbands** per channel
3. For each subband, flatten to 1D and compute Shannon entropy using histogram binning (√N bins, natural log).
4. Average entropy across all subbands for luminance (L\*).
5. Repeat for chrominance (a\*, b\*) with weight `w = 0.0625 = 1/16`.
6. Final weighted average:

$$\text{SE} = \frac{H_L + w \cdot H_a + w \cdot H_b}{1 + 2w}$$

**Interpretation:**
A smooth gradient has low subband entropy (pyramig coefficients are mostly zero → peaked histogram). A complex texture has high subband entropy (spread-out histograms at multiple scales/orientations).

**Note on entropy base:** Uses natural log (nats), not log₂ (bits). The absolute scale doesn't matter because all subbands use the same base.

---

### F5: Layout Symmetry

| Property | Value |
|----------|-------|
| **Function** | `layout_symmetry(image)` |
| **Output range** | [0, 1] |
| **Unit** | normalized correlation |
| **Higher means** | More symmetry under this metric; reduced visual search is an unvalidated design hypothesis |
| **Reference** | Custom project metric inspired by Miniukovich, A. & De Angeli, A. (2015), "Computation of Interface Aesthetics." *CHI '15*, 1163–1172. |

**Algorithm:**

1. Convert to grayscale (float64).
2. Zero-mean: `I' = I − mean(I)`.
3. Compute vertical symmetry (left-right mirror):

$$\text{NCC}_v = \frac{\sum I'(x,y) \cdot I'(W{-}1{-}x, y)}{\sum I'(x,y)^2}$$

4. Compute horizontal symmetry (top-bottom mirror):

$$\text{NCC}_h = \frac{\sum I'(x,y) \cdot I'(x, H{-}1{-}y)}{\sum I'(x,y)^2}$$

5. Final: `symmetry = clip((NCC_v + NCC_h) / 2, 0, 1)`

**Interpretation:**
NCC = 1 means the image is identical to its mirror. NCC = 0 means no linear correlation. Most GUIs have moderate vertical symmetry (centered layouts) but low horizontal symmetry (header differs from footer).

---

### F6: Chromatic Coherence

| Property | Value |
|----------|-------|
| **Function** | `chromatic_coherence(image)` |
| **Output range** | [0, 1] |
| **Unit** | normalized composite score |
| **Higher means** | More fragmented color palette |
| **AIM source** | `m13_luminance_std.py`, `m15_colorfulness.py`, `m16_hsv_avg.py` |
| **Reference** | Custom project composite; the colorfulness submetric follows Hasler, D. & Süsstrunk, S. E. (2003), "Measuring Colorfulness in Natural Images." *SPIE 5007*. |

**Combines 4 sub-metrics:**

#### A) Luminance Standard Deviation (AIM m13)

$$\text{lum} = 0.2126R + 0.7152G + 0.0722B \quad \text{(Rec.709 luma)}$$
$$\text{norm\_lum} = \min\left(\frac{\sigma(\text{lum})}{128}, 1\right)$$

#### B) Colorfulness (AIM m15 / Hasler & Süsstrunk)

$$rg = R - G, \quad yb = 0.5(R+G) - B$$
$$C = \sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3 \cdot \sqrt{\mu_{rg}^2 + \mu_{yb}^2}$$
$$\text{norm\_color} = \min\left(\frac{C}{150}, 1\right)$$

#### C) Hue Circular Standard Deviation (AIM m16)

Hue is circular (0°–360°), so standard deviation is computed using the Mardia-Jupp formula:

$$\bar{R} = \left|\text{mean}(e^{i\theta})\right| = \sqrt{\text{mean}(\sin\theta)^2 + \text{mean}(\cos\theta)^2}$$
$$\sigma_{\text{circ}} = \sqrt{-2 \ln \bar{R}}$$
$$\text{norm\_hue} = \min\left(\frac{\sigma_{\text{circ}}}{2.5}, 1\right)$$

#### D) Saturation Standard Deviation (AIM m16)

$$\text{norm\_sat} = \min\left(\frac{\sigma(S)}{0.5}, 1\right) \quad \text{where } S \in [0,1]$$

#### Final

$$\text{CC} = \frac{\text{norm\_lum} + \text{norm\_color} + \text{norm\_hue} + \text{norm\_sat}}{4}$$

---

### F7: Visual Hierarchy

| Property | Value |
|----------|-------|
| **Function** | `visual_hierarchy(image)` |
| **Output range** | [0, 1] |
| **Unit** | normalized composite score |
| **Higher means** | Clearer layered structure under this metric; reduced search effort is an unvalidated design hypothesis |
| **AIM source** | `m5_contour_density.py` (figure-ground part) |
| **Reference** | Custom project composite; broader visual-complexity context from Tuch, A. N. et al. (2009), "Visual Complexity of Websites: Effects on Users' Experience, Physiology, Performance, and Memory." *International Journal of Human-Computer Studies*, 67(9). |

**Combines 2 sub-metrics:**

#### A) Figure-Ground Contrast (inspired by AIM m5)

Apply Canny at 7 increasing threshold levels (10%–70% of max):

| Level | High threshold | Low threshold (40% of high) |
|-------|---------------|----------------------------|
| 0.1 | 25 | 10 |
| 0.2 | 51 | 20 |
| 0.3 | 76 | 30 |
| 0.4 | 102 | 40 |
| 0.5 | 127 | 50 |
| 0.6 | 153 | 61 |
| 0.7 | 178 | 71 |

Compute weighted decay:

$$\text{fg} = \frac{\sum_{k=0}^{5} (\text{count}[k] - \text{count}[k{+}1]) \cdot (1 - k/6)}{\text{count}[0] - \text{count}[6]}$$

Edges that survive to high thresholds indicate strong figure-ground boundaries.

#### B) Size Gradient

1. Binarize with Otsu's adaptive threshold.
2. Find connected components (8-connectivity).
3. Sort by area (descending), take top-3.
4. `size_gradient = sum(top3_areas) / total_image_area`

#### Final

$$\text{VH} = \frac{\text{clip}(fg, 0, 1) + \text{clip}(sg, 0, 1)}{2}$$

---

### F8: Interactive Element Density

| Property | Value |
|----------|-------|
| **Function** | `interactive_element_density(image)` |
| **Output range** | [0, ∞), typically 0.01–0.5 |
| **Unit** | elements per 100×100 px |
| **Higher means** | More detected control-like candidates under this custom proxy; decisional-load implications are unvalidated |
| **Reference** | Custom heuristic (contour-based) |

**Algorithm:**

1. Grayscale → Gaussian blur (5×5) → Canny (50/150).
2. Morphological closing (5×5 rect kernel) to merge nearby edges.
3. Find external contours.
4. Filter by:
   - **Area**: 0.05%–25% of image area
   - **Aspect ratio**: max/min side < 10 (not too elongated)
   - **Solidity**: contour_area / bounding_rect_area > 0.3 (compact shape)
5. Density = qualifying_count / (image_area / 10000)

**Limitations:** This is a heuristic, not a trained detector. For higher accuracy, a deep learning model (e.g., YOLO trained on the RICO UI dataset) could be substituted.

---

## 4. AIM Source Mapping

Each feature is traceable to the AIM repository (Aalto-UI/aim, branch `aim2`):

| Feature | AIM Metric ID | AIM Source File | Our Function |
|---------|---------------|-----------------|--------------|
| Shannon Entropy | m21 (related) | `m21_jpeg_file_size.py` | `shannon_entropy()` — we use direct histogram instead of JPEG |
| Edge Density | m4 | `m4_edge_density.py` | `edge_density()` |
| Feature Congestion | m8 | `m8_feature_congestion.py` + `image_visual_clutter_utils.py` | `feature_congestion()` |
| Subband Entropy | m7 | `m7_subband_entropy.py` + `image_visual_clutter_utils.py` | `subband_entropy()` |
| Layout Symmetry | — | Custom (inspired by Miniukovich) | `layout_symmetry()` |
| Chromatic Coherence | m13, m15, m16 | `m13_luminance_std.py`, `m15_colorfulness.py`, `m16_hsv_avg.py` | `chromatic_coherence()` |
| Visual Hierarchy | m5 (partial) | `m5_contour_density.py` | `visual_hierarchy()` |
| Interactive Element Density | — | Custom (contour-based) | `interactive_element_density()` |

### Ported Utility Functions

These are 1:1 ports from `aim/backend/aim/metrics/image_visual_clutter_utils.py`:

| Our function | AIM original | Purpose |
|-------------|-------------|---------|
| `_rgb2lab()` | `rgb2lab()` | sRGB → CIELab color space |
| `_conv2()` | `conv2()` | MATLAB-compatible 2D convolution |
| `_RRoverlapconv()` | `RRoverlapconv()` | Border-corrected convolution |
| `_RRgaussfilter1D()` | `RRgaussfilter1D()` | 1D Gaussian kernel |
| `_DoG1filter()` | `DoG1filter()` | Difference-of-Gaussians filter pair |
| `_addborder()` | `addborder()` | Image border padding |
| `_filt2()` | `filt2()` | 2D filtering with reflected borders |
| `_RRcontrast1channel()` | `RRcontrast1channel()` | Center-surround contrast |
| `_reduce()` | `reduce()` | Gaussian pyramid downsample (2×) |
| `_RRoverlapconvexpand()` | `RRoverlapconvexpand()` | Border-corrected upsample (2×) |
| `_imrotate()` | `imrotate()` | Kernel rotation |
| `_poolnew()` | `poolnew()` | Orientation energy pooling |
| `_orient_filtnew()` | `orient_filtnew()` | 4-orientation second-derivative filters |
| `_entropy()` | `entropy()` | Histogram-based Shannon entropy |

---

## 5. Utility Functions Reference

### Signal Processing Chain for Feature Congestion

```
Image (BGR)
  │
  ├── RGB → CIELab (_rgb2lab)
  │     ├── L* ─┐
  │     ├── a* ─┼── GaussianPyramid (pyrtools, 3 levels)
  │     └── b* ─┘
  │
  ├── COLOR CLUTTER (per level):
  │     Local mean (_RRoverlapconv + _RRgaussfilter1D)
  │     → 3×3 Covariance → det^(1/6)
  │
  ├── CONTRAST CLUTTER (per level):
  │     DoG (_DoG1filter → _filt2) → |inner - outer|
  │     → Local std (_RRoverlapconv)
  │
  ├── ORIENTATION CLUTTER (per level):
  │     4 oriented filters (_orient_filtnew)
  │     → squared energy → pool (_poolnew)
  │     → opponency → 2×2 Cov → det^(1/4)
  │
  └── COLLAPSE (per-pixel max across scales)
        → COMBINE (weighted sum)
        → spatial mean → scalar
```

### Signal Processing Chain for Subband Entropy

```
Image (BGR)
  │
  ├── RGB → CIELab (_rgb2lab)
  │     ├── L* ──── SteerablePyramidFreq (pyrtools)
  │     │            ├── 3 scales × 4 orientations = 12 subbands
  │     │            ├── highpass residual
  │     │            └── lowpass residual
  │     │            → _entropy(each subband) → mean
  │     │
  │     ├── a* ──── Same pyramid → weighted by 0.0625
  │     └── b* ──── Same pyramid → weighted by 0.0625
  │
  └── Weighted average → scalar
```

---

## 6. Saliency Pipeline (UMSI++)

*Added: 06.05.2026*

### 6.1 Background and source boundary

**UMSI++** is the UEyes-adapted version of the Unified Model of Saliency and
Importance. Jiang et al. (CHI 2023) evaluated it using the UEyes eye-tracking
dataset: 62 participants viewing 1,980 screenshots across four interface types.
The broader graphic-design and natural-image classes associated with the
predecessor UMSI must not be described as UEyes categories.

| Property | Value |
|---|---|
| **Paper** | Jiang et al., "UEyes: Understanding Visual Saliency across User Interface Types", CHI 2023, doi:10.1145/3544548.3581096 |
| **Source dataset** | UEyes: 1,980 screenshots with eye-tracking logs and multi-duration saliency/scanpath derivatives |
| **Participants** | 62 |
| **UEyes interface types** | Webpage, Desktop UI, Mobile UI, Poster |
| **Model-history boundary** | Infographics, advertisements, and natural-image classes belong to predecessor-model context, not to the four UEyes categories |
| **Original framework** | TensorFlow 1.14 + Keras 2.3.1 + CUDA 9.0 + Python 3.7 |
| **Production port** | TensorFlow 2.16.2 + Keras 3.10 + Apple Silicon M4 (CPU) |
| **Checkpoint** | `saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5` (29.9M parameters) |

The checkpoint identity and port compatibility are engineering evidence. They
do not establish a new human benchmark, guarantee generalization outside the
source domain, or turn model activation into observed gaze.

### 6.2 Architecture

```
Input(256×256×3, BGR, VGG-mean-subtracted)
  │
  ├── Custom Xception Backbone
  │   • Blocks 1-3: Standard (stride 2 → 32×32 spatial)
  │   • Block 4: stride MODIFIED (1,1) statt (2,2) → bleibt 32×32
  │   • Middle Flow: 8 blocks (Blocks 5-12), Residual SepConvs
  │   • Exit Flow: stride MODIFIED (1,1) → bleibt 32×32
  │   → Output: (batch, 32, 32, 2048)
  │
  ├── ASPP Branch (Atrous Spatial Pyramid Pooling)
  │   • 1×1 Conv (256 filter)
  │   • DepthwiseConv rate=6 → Pointwise (256)
  │   • DepthwiseConv rate=12 → Pointwise (256)
  │   • DepthwiseConv rate=18 → Pointwise (256)
  │   → Concat: (batch, 32, 32, 1024)
  │
  ├── Numeric Auxiliary Head
  │   • Conv 3×3 stride 3 → BN → ReLU → Dropout
  │   • GlobalAveragePooling → Dense(256) → Dropout
  │   • Dense(6, softmax) → out_classif (six numeric values;
  │     semantic class order is not verified)
  │   • Dense(256) → Tile zu (32, 32, 256) via Lambda
  │
  ├── Concatenate [ASPP, Auxiliary Tile]
  │   → (batch, 32, 32, 1280)
  │
  └── Decoder
      • Conv 1×1(256) → BN → ReLU → Dropout
      • Conv 3×3(256) → Conv 3×3(256) → Dropout
      • UpSample ×2 → (64, 64)
      • Conv 3×3(128) → Conv 3×3(128) → Dropout
      • UpSample ×2 → (128, 128)
      • Conv 3×3(64) → Dropout
      • UpSample ×4 → (512, 512)
      • Conv 1×1(1) → out_heatmap
```

### 6.3 TensorFlow 2 port (06.05.2026)

| Aspect | Original (TF1) | Production port (TF2) |
|--------|----------------|-------------|
| Imports | `from keras.layers import ...` | `from keras import layers` (Keras 3) |
| Xception | `xception_custom.py` + `keras_applications` | Inline implementation in `_build_custom_xception()` |
| Lambda tiling | `tf.concat` loops | Equivalent TensorFlow 2-compatible operation |
| Weight format | HDF5 with `layer_names` attribute | Directly compatible; all 107 weighted layers match |
| GPU | CUDA 9.0 + TF-GPU 1.14 | Apple Silicon CPU (tf-macos 2.16.2) |

**Engineering checks for the port (not construct validation):**
- 107 weighted model layers correspond to 107 layers in the HDF5 checkpoint;
- positional loading fails on architecture or tensor-shape mismatch;
- saliency and numeric auxiliary-head outputs are checked against
  repository-pinned parity evidence; and
- these checks establish compatibility and reproducibility, not an independent
  saliency benchmark or semantic auxiliary-head labels.

### 6.4 Saliency features (s ∈ ℝ⁵)

The following features are extracted from the normalized model-activation map
$S(x,y) \in [0,1]$:

| # | Feature | Formula | Range | Bounded interpretation |
|---|---------|---------|-------|------------------------|
| s₁ | **Dispersion** | $\sigma_S = \sqrt{\text{Var}[x \cdot S] + \text{Var}[y \cdot S]}$ (normalized) | [0, 1] | Spatial spread of model-estimated saliency activation |
| s₂ | **Peak Count** | Number of local maxima after Gaussian smoothing (σ=5) with value ≥ 0.3·max | ℕ₀ | Distinct model-activation peaks, not observed attention hotspots |
| s₃ | **Center Bias** | $\frac{\sum_{(x,y) \in C_{25\%}} S(x,y)}{\sum S}$ | [0, 1] | Relative model activation in the central image region |
| s₄ | **Entropy** | $H = -\sum_b p_b \log_2 p_b$ (32 bins, normalized) | [0, 1] | Evenness of the model-activation distribution |
| s₅ | **Coverage** | $\frac{|\{(x,y): S > 0.5 \cdot \max(S)\}|}{W \cdot H}$ | [0, 1] | Image area above the declared relative-activation threshold |

### 6.5 Numeric Auxiliary Head (6 values)

The checkpoint contains a six-value auxiliary softmax head used inside the
UMSI++ architecture. The exact index-to-label mapping is not verified from a
shipped training-time label encoder. Production API, UI, CSV, and CLI surfaces
therefore attach no design-class names, confidence labels, or domain verdicts
to these values. The auxiliary head remains relevant only as a numeric internal
model output and for architecture/parity evidence; downstream scoring consumes
the saliency heatmap, not semantic class labels.

### 6.6 API endpoints

| Endpoint | Method | Input | Output | Available since |
|----------|---------|-------|--------|------|
| `/api/analyze` | POST | Image (multipart) | v∈ℝ⁸ plus metadata | 04.05.2026 |
| `/api/features` | GET | — | Metadata for eight visual features | 04.05.2026 |
| `/api/saliency` | POST | Image (multipart) | s∈ℝ⁵ plus heatmap (Base64); no semantic class labels | 06.05.2026 |

### 6.7 Relationship to the UEyes dataset

The UMSI++ checkpoint originates from the UEyes model release. Therefore:
- the source model was trained using aggregated eye-tracking data from 62
  participants;
- source saliency maps are available under
  `ueyes/saliency_models/UMSI++/saliency_gt/`; and
- saliency metrics such as KL divergence, CC, and NSS can be computed against
  explicitly selected source ground truth.

This provenance does not mean that every production screenshot belongs to the
UEyes distribution or that the five project-specific summaries reproduce the
paper's benchmark metrics.

**Current evidence boundary:**
- repository tests and frozen fixtures establish production behavior,
  checkpoint identity, postprocessing policy, and project-reference parity;
- local UEyes sample checks are engineering sanity checks, not an official
  benchmark or independent construct validation;
- no end-to-end validation against new human data has been completed;
- no HCEye correspondence result validates the screenshot-level layout index.

---

## 7. Optional Jokinen-Based Visual-Search Diagnostic

### 7.1 Methodological role and claim boundary

The project implements a Python-native adaptation informed by the Adaptive
Feature Guidance model of Jokinen et al. (2020). Its purpose is to expose a
separate, optional computational diagnostic for declared target-search
scenarios.

The diagnostic produces **model-simulated** search times, fixation counts, and
paths. These values are not observed eye tracking, measured user performance,
or validated predictions for a new user or screenshot. The diagnostic is off by
default, non-score-bearing, and cannot change the Stage-1 x19 vector, the
experimental layout-complexity index, or the Stage-2 scenario direction.

### 7.2 Model-informed simulation sequence

The implementation simulates target-directed visual search step by step:

1. The controller starts at the screen centre.
2. Each detected element receives a model activation from declared visual
   features, optional UMSI++ activation, and simulation noise.
3. A winner-take-all selection chooses the next candidate element.
4. EMMA-informed encoding time uses the Salvucci (2001) formulation:
   $T_e = K \cdot [-\ln(f)] \cdot e^{k \cdot \varepsilon}$.
5. Saccade time is represented as
   $T_s = t_{prep} + t_{exec} \cdot D + t_{sacc}$.
6. Recently visited elements are inhibited through a visual short-term-memory
   mechanism.
7. A trial terminates when the declared target is found or `max_fixations` is
   reached.

This is a project implementation informed by the cited models, not a claim of
source-code identity or parameter equivalence with the original study.

### 7.3 Key equations

| Equation | Formula | Model role |
|----------|---------|------------|
| Eq. 1 (Visual Threshold) | $\theta = a \cdot \varepsilon^2 + b \cdot \varepsilon$ | Declares feature visibility when $\theta < \alpha_{size}$ |
| Eq. 2 (Bottom-Up Activation) | $BA_i = \sum_j \sum_k \frac{dissim(v_{ik}, v_{jk})}{\sqrt{d_{ij}}}$ | Computes activation from feature dissimilarity |
| Eq. 4 (EMMA Encoding) | $T_e = K \cdot [-\ln(f)] \cdot e^{k \cdot \varepsilon}$ | Increases model encoding time with eccentricity |
| Eq. 5 (Saccade) | $T_s = t_{prep} + t_{exec} \cdot D + t_{sacc}$ | Computes model saccade duration |

### 7.4 Parameter provenance

| Parameter | Value | Provenance status |
|-----------|-------|-------------------|
| K (EMMA encoding) | 0.006 | Salvucci (2001) |
| k (EMMA exponent) | 0.4 | Salvucci (2001) |
| t_prep (saccade preparation) | 0.135 s | EMMA-informed parameter |
| t_exec (saccade per degree) | 0.002 s/° | EMMA-informed parameter |
| W_BA (bottom-up weight) | 1.1 | Nyamsuren & Taatgen (2013), as used by Jokinen et al. (2020) |
| σ_TA (activation noise) | 0.376 | Nyamsuren & Taatgen (2013), as used by Jokinen et al. (2020) |
| τ_VSTM (memory capacity) | 20 steps | Project implementation choice informed by Jokinen et al. (2020) |
| W_saliency (UMSI++ weight) | 0.8 | **Project-specific author choice; not source-paper calibrated** |
| saliency_exponent | 2.0 | **Project-specific author choice; not source-paper calibrated** |

The parameter table provides provenance, not validation. The project-specific
UMSI++ coupling and element detector would require a separate equation/parameter
audit and human-data evaluation before any real-performance claim.

### 7.5 Project adaptations relative to legacy AIM

| Aspect | Legacy (`vg2_visual_search.py`) | Project implementation |
|--------|----------------------------------|------------------------|
| Runtime | Compiled binary | Python-native |
| Saliency input | Hand-crafted categorical features | Continuous UMSI++ model activation with project-specific coupling |
| Element detection | External segmentation required | Project-specific Canny/contour detector |
| Simulation | Legacy binary behavior not inspectable here | Monte Carlo trials with declared parameters |
| Output | Heatmap image | JSON model estimates per element plus aggregate statistics |
| Platform | Legacy Linux/macOS x86 binaries | Cross-platform Python subject to the frozen runtime contract |

### 7.6 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Screenshot (PNG/JPG/JPEG/BMP/TIFF)                          │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│ Element     │  │ UMSI++       │
│ Detector    │  │ Saliency     │
│ (Canny+NMS) │  │ (TF2/Keras3) │
└──────┬──────┘  └──────┬───────┘
       │                │
       │  elements[]    │  saliency_map (H×W)
       │                │
       └────────┬───────┘
                ▼
┌─────────────────────────────────┐
│ Jokinen 2020 Search Model       │
│                                 │
│  feature_activations (Eq. 2)    │
│  + UMSI++ saliency (continuous) │
│  → combined_activation          │
│                                 │
│  Monte Carlo Simulation:        │
│  for each target element:       │
│    for N trials:                │
│      simulate_search()          │
│      → (time, fixations)        │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ Output per Element:             │
│  • search_time_s (float)        │
│  • fixation_count (float)       │
│  • bbox, center, color          │
│                                 │
│ Aggregate:                      │
│  • mean/max/min search time     │
│  • predicted_difficulty         │
│    (easy/moderate/difficult/    │
│     very_hard)                  │
└─────────────────────────────────┘
```

### 7.7 API Endpoint

```
POST /api/search-time
Content-Type: multipart/form-data
Body: image=@screenshot.png

Query params (optional):
  n_simulations=100  (Monte Carlo trials per element)
  use_saliency=true  (integrate UMSI++ saliency)
```

`use_saliency=true` is fail-closed: if the requested UMSI++ stage fails, the
route returns structured HTTP 503 with `analysis_complete=false` and no
diagnostic result. Feature-only mode is a separate caller-selected contract
available only with `use_saliency=false`; successful responses identify it as
`analysis_mode=feature_only_explicit`. The scanpath-to-target and learning-curve
diagnostics use the same policy.

**Response:**
```json
{
  "filename": "bmw_route.png",
  "n_elements": 40,
  "mean_search_time_s": 6.069,
  "max_search_time_s": 8.276,
  "min_search_time_s": 2.479,
  "search_time_std_s": 1.259,
  "predicted_difficulty": "very_hard",
  "analysis_complete": true,
  "analysis_mode": "saliency_augmented",
  "saliency_requested": true,
  "saliency_used": true,
  "per_element": [
    {"id": 0, "search_time_s": 2.479, "fixation_count": 9.6,
     "bbox": [280, 264, 985, 495], "color_category": "orange"},
    ...
  ],
  "model_info": {
    "model": "Jokinen 2020 Adaptive Feature Guidance",
    "mode": "novice_search",
    "reference": "Jokinen et al. (2020). IJHCS, 136, 102376."
  }
}
```

### 7.8 Engineering sanity check — not validation

The recorded `bmw_route.png` example produced finite outputs (mean 6.07 s,
maximum 8.28 s, minimum 2.48 s) and ranked a visually distinctive orange
element first under the project implementation. This demonstrates executable,
deterministic model behaviour for that fixture only.

It does **not** reproduce the original paper's experiment, establish parameter
equivalence, validate the absolute time or fixation scale, or show agreement
with human participants. Comparisons to example layouts from the source paper
must not be reported as validation or cross-study consistency without a formal
reproduction protocol.

### 7.9 Files

| File | Purpose |
|------|---------|
| `cognitive/__init__.py` | Package definition |
| `cognitive/jokinen_model.py` | Project simulation implementation and parameters |
| `cognitive/element_detector.py` | Project-specific UI-element detector |
| `cognitive/test_jokinen.py` | Engineering end-to-end test; not human validation |

---

## 8. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Array operations |
| opencv-python | 4.13.0 | Image I/O, Canny, morphology, connected components |
| scipy | 1.13.1 | 2D convolution (`signal.convolve2d`) |
| scikit-image | 0.24.0 | Image rotation (`transform.rotate`) |
| Pillow | 11.3.0 | (Optional, used for compatibility) |
| PyWavelets | 1.6.0 | (Indirect dependency via pyrtools) |
| pyrtools | 1.0.10 | Gaussian & Steerable pyramids |
| matplotlib | 3.9.4 | (Optional, for visualization) |
| flask | 3.1.3 | Web server (app.py only) |
| tensorflow-macos | 2.16.2 | UMSI++ model inference (Apple Silicon) |
| keras | 3.10.0 | High-level neural network API |
| h5py | 3.14.0 | HDF5 weight file I/O |

Install all:
```bash
python -m venv venv
source venv/bin/activate
pip install numpy opencv-python scipy scikit-image Pillow PyWavelets pyrtools matplotlib flask tensorflow-macos
```

---

## 9. Usage

### Command Line

```bash
# Single image
python stage1/visual_complexity.py --image path/to/screenshot.png

# Directory of images → CSV
python stage1/visual_complexity.py --dir stage1/data/screenshots/ --output results.csv
```

### Python API

```python
# === Stage 1a: Visual Complexity ===
from visual_complexity import compute_complexity_vector, FEATURE_KEYS

v = compute_complexity_vector("screenshot.png")
# v = {"shannon_entropy": 6.64, "edge_density": 0.024, ...}

for key in FEATURE_KEYS:
    print(f"{key}: {v[key]:.4f}")
```

```python
# === Stage 1b: Saliency (06.05.2026) ===
from saliency.umsi_model import UMSIPlus
from saliency.saliency_features import extract_saliency_features

# Load the model once, then reuse it for multiple images
model = UMSIPlus("saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5")

# Predict saliency
heatmap, auxiliary_head = model.predict_saliency("screenshot.png", return_classif=True)
# heatmap: np.ndarray shape (H, W), float32 in [0,1]
# auxiliary_head: np.ndarray shape (6,), numeric softmax values;
# semantic labels/order are not verified and are not used for scoring

# Extract scalar features
features = extract_saliency_features(heatmap)
# features = {"saliency_dispersion": 0.49, "saliency_peak_count": 7, ...}
```

### Web UI

```bash
python stage1/app.py
# Open http://localhost:5001
```

### cURL API test

```bash
# Stage 1a: Visual Complexity
curl -X POST -F "image=@screenshot.png" http://localhost:5001/api/analyze

# Stage 1b: Saliency
curl -X POST -F "image=@screenshot.png" http://localhost:5001/api/saliency
```

---

## 10. Example Output

**Input:** BMW Route navigation screenshot (1536 × 2050 px)

```
Visual Complexity Vector v ∈ ℝ⁸:
  shannon_entropy              6.6414
  edge_density                 0.0237
  feature_congestion           2.6120
  subband_entropy              2.5944
  layout_symmetry              0.2198
  chromatic_coherence          0.4761
  visual_hierarchy             0.5722
  interactive_element_density  0.0762
```

**Interpretation:**
- High entropy (6.64/8) — rich information content (map + text + controls)
- Low edge density (2.4%) — relatively clean boundaries
- Moderate feature congestion (2.61) — some color/contrast variability from the map
- Moderate subband entropy (2.59) — texture from map tiles
- Low symmetry (0.22) — map is inherently asymmetric
- Moderate chromatic coherence (0.48) — mix of BMW blue/dark theme + colorful map
- Moderate hierarchy (0.57) — clear separation between map/nav bar/controls
- Low element density (0.08) — few interactive elements visible

### Historical Saliency Output Example (06.05.2026)

```
Saliency Features s ∈ ℝ⁵ (UMSI++):
  saliency_dispersion          0.4931
  saliency_peak_count          7
  saliency_center_bias         0.3090
  saliency_entropy             0.7169
  saliency_coverage            0.0642

Numeric auxiliary head (semantic index order unverified):
  index_0                      0.1555
  index_1                      0.1672
  index_2                      0.1654
  index_3                      0.1640
  index_4                      0.2446
  index_5                      0.1032
```

**Interpretation:**
- Dispersion 0.49 → model-estimated saliency activation is moderately spread.
- 7 peaks → seven distinct model-activation peaks; these are not observed
  attention hotspots.
- Center bias 0.31 → 31% of the relative model activation falls within the
  declared central region.
- Entropy 0.72 → comparatively even model-activation distribution under the
  project metric.
- Coverage 0.06 → 6.4% of the image exceeds 50% of the within-image maximum
  model activation.
- The auxiliary values receive no semantic class interpretation.

---

## 11. Normalization Ranges

For the web UI radar chart and bar display, features are normalized to [0, 1] using these ranges:

| Feature | Min | Max | Basis |
|---------|-----|-----|-------|
| Shannon Entropy | 0 | 8 | Theoretical max for 256 bins |
| Edge Density | 0 | 0.15 | Empirical max for GUI screenshots |
| Feature Congestion | 0 | 120 | Empirical max (very cluttered images) |
| Subband Entropy | 0 | 5 | Empirical max |
| Layout Symmetry | 0 | 1 | NCC range |
| Chromatic Coherence | 0 | 1 | Composite score already normalized |
| Visual Hierarchy | 0 | 1 | Composite score already normalized |
| Interactive Element Density | 0 | 0.5 | Empirical max for dense UIs |

---

## 12. Limitations & Future Work

### Stage 1a (Visual Complexity)

1. **Feature Congestion** and **Subband Entropy** are computationally expensive (10–30s per image on M4 Mac due to multi-scale pyramid construction). Potential optimization: resize images to a maximum dimension before processing.

2. **Interactive Element Density** uses a heuristic contour detector, not a trained model. For production accuracy, a deep learning UI element detector (e.g., UIED, YOLO trained on RICO dataset) would be preferable.

3. **Layout Symmetry** uses raw pixel NCC, which is sensitive to small translations. A more robust version could use SSIM or feature-point matching.

4. The Stage-1 visual block contains raw and already bounded component
   measures. The public layout index applies the repository's fixed project
   heuristics; any later dataset-level calibration remains an open research
   decision outside the current production contract.

5. The normalization constants in **Feature Congestion** (0.2088, 0.0660,
   0.0269) were derived by Rosenholtz et al. for natural images. Transfer to
   any final thesis domain requires separate validation; that domain is not
   fixed by this document.

### Stage 1b (Saliency — 06.05.2026)

6. **UMSI++ currently runs on CPU in the frozen environment.** TensorFlow 2.16
   on the recorded M4 runtime uses the CPU; inference speed is an engineering
   property, not a scientific-validity claim.

7. **Auxiliary-head semantics unverified** — the six numeric values have no
   public class-label interpretation and do not drive downstream scoring.

8. **Validation boundary** — repository parity/sanity evidence is not an
   official external benchmark and does not validate cognitive load, observed
   gaze, or human performance.

### Explicit scope exclusions

Target-driven scanpath prediction, participant-level cognitive-load
measurement, personality inputs, trained Stage-2 regressors, numeric contextual
modifiers, user-study execution, and post-study analysis are not part of this
technical pipeline release. They must not be inferred from dormant prototypes
or historical project material.

---

## References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
2. Canny, J. (1986). "A Computational Approach to Edge Detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851
3. Rosenholtz, R., Li, Y., & Nakano, L. (2007). "Measuring Visual Clutter." *Journal of Vision*, 7(2):17, 1–22. https://doi.org/10.1167/7.2.17
4. Miniukovich, A., & De Angeli, A. (2015). "Computation of Interface Aesthetics." *Proceedings of CHI 2015*, 1163–1172. https://doi.org/10.1145/2702123.2702575
5. Hasler, D., & Süsstrunk, S. E. (2003). "Measuring Colourfulness in Natural Images." *Proceedings of SPIE 5007, Human Vision and Electronic Imaging VIII*. https://doi.org/10.1117/12.477378
6. Tuch, A. N., Bargas-Avila, J. A., Opwis, K., & Wilhelm, F. H. (2009). "Visual Complexity of Websites: Effects on Users' Experience, Physiology, Performance, and Memory." *International Journal of Human-Computer Studies*, 67(9), 703–715. https://doi.org/10.1016/j.ijhcs.2009.04.002
7. Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B., Kylmälä, J., & Oulasvirta, A. (2023). "UEyes: Understanding Visual Saliency across User Interface Types." *Proceedings of CHI 2023*, Article 285. https://doi.org/10.1145/3544548.3581096
8. Das, A., Wu, Z., Škrjanec, I., & Feit, A. M. (2024). "Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction." *Proceedings of the ACM on Human-Computer Interaction*, 8(ETRA), Article 236. https://doi.org/10.1145/3655610
9. Jokinen, J. P. P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X. (2020). "Adaptive Feature Guidance: Modelling Visual Search with Graphical Layouts." *International Journal of Human-Computer Studies*, 136, 102376. https://doi.org/10.1016/j.ijhcs.2019.102376
10. Salvucci, D. D. (2001). "An Integrated Model of Eye Movements and Visual Encoding." *Cognitive Systems Research*, 1(4), 201–220. https://doi.org/10.1016/S1389-0417(00)00015-2
11. Nyamsuren, E., & Taatgen, N. A. (2013). "Pre-attentive and Attentive Vision Module." *Cognitive Systems Research*, 24, 62–71. https://doi.org/10.1016/j.cogsys.2012.12.010
12. Liu, C., Liu, Y.-H., Gedeon, T., Zhao, Y., Wei, Y., & Yang, F. (2019). "The Effects of Perceived Chronic Pressure and Time Constraint on Information Search Behaviors and Experience." *Information Processing & Management*, 56(5), 1667–1679. https://doi.org/10.1016/j.ipm.2019.04.004
13. Torralba, A., & Efros, A. A. (2011). "Unbiased Look at Dataset Bias." *Proceedings of CVPR 2011*, 1521–1528. https://doi.org/10.1109/CVPR.2011.5995347
