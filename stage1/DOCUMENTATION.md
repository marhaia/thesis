# Stage 1 — Visual Complexity & Saliency Extraction

## Documentation & Technical Reference

**Thesis:** *Two-Stage Multi-Output Pipeline: Computational Estimation of Interactional Complexity from GUI Screenshots*  
**Author:** Hannah Mueller (Q682780)  
**Module:** `stage1/visual_complexity.py` + `saliency/umsi_model.py`  
**Version:** 2.7 (16.07.2026)

---

## Change Log

| Datum | Version | Änderung |
|-------|---------|----------|
| 04.05.2026 | 1.0 | Stage 1: Feature-Vektor v∈ℝ⁸ implementiert (8 visuelle Komplexitätsmetriken) |
| 04.05.2026 | 1.0 | Flask-WebUI + API auf Port 5001 |
| 04.05.2026 | 1.0 | Inline-Kommentare + DOCUMENTATION.md erstellt |
| 05.05.2026 | 1.5 | UEyes-Repo geclont, UMSI++ Architektur analysiert |
| 06.05.2026 | 2.0 | UMSI++ TF2-Port (Keras 3) — lauffähig auf M4 Mac (CPU) |
| 06.05.2026 | 2.0 | Saliency-Feature-Extraktion s∈ℝ⁵ implementiert |
| 06.05.2026 | 2.0 | Flask-Endpoint `/api/saliency` integriert |
| 06.05.2026 | 2.0 | Pretrained Weights geladen (UEyes CHI2023, ~115 MB / 120 MB Datei) |
| 06.05.2026 | 2.0 | PathGAN++ und DeepGaze++ Weights heruntergeladen (noch nicht portiert) |
| 06.05.2026 | 2.5 | **Jokinen 2020 Visual Search Model** implementiert (`cognitive/jokinen_model.py`) |
| 06.05.2026 | 2.5 | Element-Detektor für UI-Elemente (`cognitive/element_detector.py`) |
| 06.05.2026 | 2.5 | Integration: UMSI++ Saliency als Bottom-Up-Signal in Jokinen-Modell |
| 06.05.2026 | 2.5 | Flask-Endpoint `/api/search-time` — kognitive Suchzeit-Vorhersage |
| 06.05.2026 | 2.5 | Monte-Carlo-Simulation (50–500 Trials pro Element, EMMA, VSTM) |
| 26.05.2026 | 2.6 | `/api/cognitive-load` standardmäßig auf **HCEye-Regeln + Descriptor/Profile-Modifikatoren** gestellt |
| 26.05.2026 | 2.6 | Trainiertes Stage-2-Modell nur noch als **explizites experimentelles Opt-in** |
| 16.07.2026 | 2.7 | **Web-UI/UX-Überarbeitung** (nur Darstellung/Copy, keine Scoring-Änderung): konsistente Info-„i"-Icons inkl. Tooltips für Cognitive Load Score, Search Efficiency & Attention Demand; einheitliche Trennlinien zwischen Ergebnis-Blöcken |
| 16.07.2026 | 2.7 | Comparison-History: Spalte „Attn" → **„Attention"** ausgeschrieben; CSV-Export-Button dynamisch **„Export all / Export selected (N)"** und exportiert nur ausgewählte Zeilen |
| 16.07.2026 | 2.7 | Navigation: Burger-Menü durch **Einstellungs-Zahnrad-Icon** ersetzt (nur „Alerts"); Contact in den **Footer** verschoben; Footer-Platzhalter für **GitHub / OSF / DOI** ergänzt |
| 16.07.2026 | 2.7 | Display-Preset-Wording vereinfacht (Phone / Laptop 14″ / Desktop 17″, „at desk distance" entfernt); alle Emoji-Feature-Icons aus `/api/features` entfernt (waren ungenutzt) |
| 16.07.2026 | 2.7 | **Quellen-Audit:** Feature-DOIs mit dem angezeigten Referenztext abgeglichen — Edge Density → **Canny (1986)**, Visual Hierarchy → **Tuch et al. (2009)**; Out-of-Domain-Warnung mit wissenschaftlichem Beleg zur Dataset-Shift-Problematik versehen (**Torralba & Efros 2011**, **Jiang et al. 2023**) |

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
7. [Cognitive Metrics: Jokinen 2020 Visual Search](#7-cognitive-metrics-jokinen-2020-visual-search)
8. [Dependencies](#8-dependencies)
9. [Usage](#9-usage)
10. [Example Output](#10-example-output)
11. [Normalization Ranges](#11-normalization-ranges)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Overview

Stage 1 extracts a **visual complexity vector v ∈ ℝ⁸** from a single GUI screenshot, plus a **saliency feature vector s ∈ ℝ⁵** predicted by UMSI++. Together they form the **extended feature vector v̂ ∈ ℝ¹³** that feeds into the cognitive-load layer. The saliency vector is optional for the image-only `/api/analyze` route, but it is **required** by `/api/saliency` and `/api/cognitive-load`, which fail closed (HTTP 503) if the UMSI++ checkpoint is unavailable or produces an invalid map.

Additionally, the **Jokinen 2020 Cognitive Search Model** provides per-element **predicted visual search time** — the central cognitive metric that bridges the gap between AIM's existing visual metrics and actual user performance.

The pipeline is **task-independent** at the image-analysis level — it analyzes only the raw visual properties of the screenshot. Task context is then added by a **rule-based task descriptor** and an **optional coarse Big-Five preset**. A trained Stage-2 regressor may still be used experimentally, but it is not required for the default pipeline.

**Current default (26.05.2026):**
- Base score from HCEye-derived cognitive-load coefficients
- Additive adjustment from Task Descriptor
- Additive adjustment from optional Big-Five preset
- No retraining required for the default path

```
Input:  GUI screenshot (PNG/JPG)
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
   │  Pretrained on UEyes dataset (62 participants,           │
   │  1980 UI screenshots, 5 duration conditions)             │
   │                                                          │
   │  Outputs:                                                │
   │    • Saliency heatmap (512×512 → original resolution)   │
   │    • Auxiliary 6-dim head (UNVALIDATED, unused)          │
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
   │  Stage 1c: cognitive/jokinen_model.py    [06.05.2026]    │
   │                                                          │
   │  Jokinen 2020 Adaptive Feature Guidance                  │
   │  (IJHCS, 136, 102376)                                   │
   │                                                          │
   │  Input: detected UI elements + UMSI++ saliency map       │
   │  Process: Monte Carlo search simulation (EMMA + VSTM)    │
   │  Output:                                                 │
   │    • Per-element predicted search time (seconds)         │
   │    • Per-element fixation count                          │
   │    • Layout difficulty rating                            │
   │                                                          │
   │  → c = [mean_time, max_time, std_time, difficulty]       │
   └──────────────────────────────────────────────────────────┘
        ↓
Output: v̂ = [f₁, …, f₈, s₁, …, s₅] ∈ ℝ¹³ + cognitive metrics
```

---

## 2. Architecture

### File Structure

```
Thesis_G/
├── stage1/                            ← Stage 1a: Visual Complexity
│   ├── visual_complexity.py           ← Core pipeline (v∈ℝ⁸)
│   ├── app.py                         ← Flask server (Port 5001)
│   ├── DOCUMENTATION.md               ← This file
│   ├── ui/
│   │   └── index.html                 ← Web interface
│   └── data/
│       ├── screenshots/               ← Input images
│       ├── uploads/                   ← Temp upload storage (Flask)
│       └── results/                   ← Output CSVs
│
├── saliency/                          ← Stage 1b: Saliency (06.05.2026)
│   ├── __init__.py
│   ├── umsi_model.py                  ← UMSI++ TF2 port (Keras 3)
│   ├── saliency_features.py           ← Feature extraction (s∈ℝ⁵)
│   ├── inspect_weights.py             ← HDF5 weight inspector utility
│   ├── test_full_pipeline.py          ← E2E test script
│   ├── output/                        ← Generated heatmaps/overlays
│   └── weights/model_weights/
│       ├── saliency_models/UMSI++/
│       │   └── umsi++.hdf5            ← 29.9M params (pretrained)
│       └── scanpath_models/
│           ├── PathGAN++/             ← Generator + Discriminator
│           └── DeepGaze++/            ← Center bias + model
│
├── ueyes/                             ← UEyes CHI2023 original repo (reference)
│   └── saliency_models/UMSI++/src/    ← Original TF1 code (not executed)
│
├── aim/                               ← AIM repository (source for F3, F4)
│   └── backend/aim/metrics/...        
│
└── venv/                              ← Python 3.9 virtual environment
```

### Processing Flow

**Stage 1a (Visual Complexity):**
1. Image is loaded via OpenCV (`cv2.imread`) in BGR format.
2. Each feature function receives the full-resolution BGR image.
3. Features are computed sequentially (F1→F8).
4. Results are returned as a Python dict: `{"shannon_entropy": 6.64, ...}`
5. The Flask API serializes this to JSON for the web UI.

**Stage 1b (Saliency; added 06.05.2026):**
1. Image is loaded as BGR, resized with aspect-ratio padding to 256×256.
2. VGG mean subtraction applied (B: 103.939, G: 116.779, R: 123.68).
3. Forward pass through UMSI++ model → 512×512 heatmap + 6-dim auxiliary head (UNVALIDATED, unused).
4. Heatmap is unpadded and resized to original image resolution.
5. Five scalar features are extracted from the min-max normalized heatmap.
6. Results returned via `/api/saliency` endpoint.

---

## 3. The 8 Features in Detail

### F1: Shannon Entropy

| Property | Value |
|----------|-------|
| **Function** | `shannon_entropy(image)` |
| **Output range** | [0, 8] bits |
| **Unit** | bits per pixel |
| **Higher means** | More visual information competing for attention |
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
- Typical GUI screenshots: 5.5–7.5 bits.

---

### F2: Edge Density

| Property | Value |
|----------|-------|
| **Function** | `edge_density(image)` |
| **Output range** | [0, 1] |
| **Unit** | proportion of edge pixels |
| **Higher means** | More structural boundaries = more parsing effort |
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
The 0.11/0.27 values are the default AIM m4 thresholds calibrated for desktop UI screenshots. They balance sensitivity (detecting real edges) with specificity (ignoring noise/texture).

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
| **Higher means** | More symmetric = less visual search needed |
| **Reference** | Miniukovich, A. & De Angeli, A. (2015). "Computation of Interface Aesthetics." *CHI '15*, pp. 1163–1172. |

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
| **Reference** | Hasler, D. & Süsstrunk, S. E. (2003). "Measuring Colorfulness in Natural Images." *SPIE 5007*. |

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
| **Higher means** | Clearer layered structure = less search effort |
| **AIM source** | `m5_contour_density.py` (figure-ground part) |
| **Reference** | Tuch, A. N. et al. (2009). "The Role of Visual Complexity and Prototypicality." *Interacting with Computers*. |

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
| **Higher means** | More action possibilities = higher decisional load |
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

*Hinzugefügt: 06.05.2026*

### 6.1 Hintergrund

**UMSI++** (Unified Model of Saliency and Importance, extended version) ist ein Deep-Learning-Modell zur Vorhersage visueller Salienz auf UIs. Es wurde im Rahmen des UEyes-Projekts (Jiang et al., CHI 2023) auf Eye-Tracking-Daten von 62 Probanden auf 1980 UI-Screenshots trainiert.

| Eigenschaft | Wert |
|---|---|
| **Paper** | Jiang et al., "UEyes: Understanding Visual Saliency across User Interface Types", CHI 2023 |
| **Training-Daten** | UEyes: 1980 Screenshots × 5 Blickdauer-Bedingungen (0.5s, 3s, 5s, …) |
| **Probanden** | 62 Personen |
| **UI-Typen** | Poster, Infografiken, Mobile UI, Desktop UI, Webseiten, Natural Images |
| **Original-Framework** | TensorFlow 1.14 + Keras 2.3.1 + CUDA 9.0 + Python 3.7 |
| **Unser Port** | TensorFlow 2.16.2 + Keras 3.10 + Apple Silicon M4 (CPU) |
| **Gewichtsdatei** | `saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5` (29.9M Parameter) |

### 6.2 Architektur

```
Input(256×256×3, BGR, VGG-mean-subtracted)
  │
  ├── Custom Xception Backbone
  │   • Blocks 1-3: Standard (stride 2 → 32×32 spatial)
  │   • Block 4: stride MODIFIED (1,1) statt (2,2) → bleibt 32×32
  │   • Middle Flow: 8 Blöcke (Blocks 5-12), Residual SepConvs
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
  ├── Auxiliary Branch (unvalidated, not exposed)
  │   • Conv 3×3 stride 3 → BN → ReLU → Dropout
  │   • GlobalAveragePooling → Dense(256) → Dropout
  │   • Dense(6, softmax) → out_classif (6-dim auxiliary output; unvalidated, not exposed)
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

### 6.3 TF2-Portierung (06.05.2026)

| Aspekt | Original (TF1) | Port (TF2) |
|--------|----------------|-------------|
| Imports | `from keras.layers import ...` | `from keras import layers` (Keras 3) |
| Xception | `xception_custom.py` + `keras_applications` | Inline in `_build_custom_xception()` |
| Lambda Tiling | `tf.concat` Loops | Identisch (TF2-kompatibel) |
| Weight Format | HDF5 mit `layer_names` attr | Direkt kompatibel — alle 107 Layer matchen |
| GPU | CUDA 9.0 + TF-GPU 1.14 | Apple Silicon CPU (tf-macos 2.16.2) |

**Validierung der Portierung:**
- 107 gewichtete Layer im Modell ↔ 107 Layer in der HDF5-Datei
- Layer-Namen und Tensor-Shapes stimmen 1:1 überein
- Der Auxiliary-Softmax-Head (`out_classif`) summiert zu 1.0, ist aber
  UNVALIDIERT (kein Trainingssignal, siehe §6.5) und wird nicht interpretiert

### 6.4 Saliency-Features (s ∈ ℝ⁵)

Aus der normalisierten Saliency-Map $S(x,y) \in [0,1]$ werden folgende Features extrahiert:

| # | Feature | Formel | Wertebereich | Interpretation |
|---|---------|--------|--------------|----------------|
| s₁ | **Dispersion** | $\sigma_S = \sqrt{\text{Var}[x \cdot S] + \text{Var}[y \cdot S]}$ (normiert) | [0, 1] | Räumliche Streuung der Aufmerksamkeit |
| s₂ | **Peak Count** | Anzahl lokaler Maxima nach Gauß-Glättung (σ=5) mit Wert ≥ 0.3·max | ℕ₀ | Anzahl distinkter Aufmerksamkeits-Hotspots |
| s₃ | **Center Bias** | $\frac{\sum_{(x,y) \in C_{25\%}} S(x,y)}{\sum S}$ | [0, 1] | Konzentration im Bildzentrum |
| s₄ | **Entropy** | $H = -\sum_b p_b \log_2 p_b$ (32 Bins, normiert) | [0, 1] | Gleichmäßigkeit der Salienz-Verteilung |
| s₅ | **Coverage** | $\frac{|\{(x,y): S > 0.5 \cdot \max(S)\}|}{W \cdot H}$ | [0, 1] | Flächenanteil mit signifikanter Salienz |

### 6.5 Auxiliary-Output-Head (6-dim, UNVALIDIERT — nicht verwendet)

Der UMSI++-Graph besitzt einen zweiten Output-Head `out_classif` (Dense(6,
softmax)), der aus der ursprünglichen UMSI-Architektur stammt. Am gepinnten
Upstream-Commit (7bc0641) kompiliert das offizielle Training-Notebook das Modell
mit `loss_weights={'dec_c_cout': 1, 'out_classif': 0}` — der Softmax-Head erhält
also **kein Trainingssignal**. Es gibt daher **keinen** belastbaren Nachweis,
dass seine sechs Wahrscheinlichkeiten einen trainierten semantischen
UI-Typ-Klassifikator bilden.

Konsequenz in diesem Projekt:

- Der Head wird als **unvalidierter** Hilfs-Tensor behandelt.
- Es werden **keine** semantischen Labels vergeben (frühere Labels wie
  `infographic` oder `natural_image` waren erfunden bzw. aus dem alten UMSI
  übernommen und sind entfernt).
- Weder `/api/saliency` noch `/api/cognitive-load` geben eine
  Klassenwahrscheinlichkeit oder eine `predicted_class` zurück; `/api/saliency`
  meldet stattdessen `classification_used: false` und
  `classification_status: "disabled_unvalidated"`.
- Der Head bleibt ausschließlich im Graphen, damit der autoritative Checkpoint
  weiterhin strikt (alle Tensoren vorhanden) geladen werden kann.

### 6.6 API-Endpoints

| Endpoint | Methode | Input | Output | Seit |
|----------|---------|-------|--------|------|
| `/api/analyze` | POST | Image (multipart) | v∈ℝ⁸ + Metadaten | 04.05.2026 |
| `/api/features` | GET | — | Feature-Metadaten (8 Features) | 04.05.2026 |
| `/api/saliency` | POST | Image (multipart) | s∈ℝ⁵ + Heatmap (Base64); `classification_used: false` | 06.05.2026 |

### 6.7 Bezug zu UEyes-Datensatz

Das UMSI++ Modell wurde **auf dem UEyes-Datensatz trainiert**. D.h.:
- Die Gewichte kodieren das aggregierte Blickverhalten von 62 Probanden
- Ground-Truth Saliency Maps sind verfügbar in `ueyes/saliency_models/UMSI++/saliency_gt/`
- Validierungsmetriken (KL-Divergenz, CC, NSS) können gegen diese Ground Truth berechnet werden

**Noch ausstehend:**
- Formale Validierung: UMSI++ Predictions vs. UEyes Ground Truth Saliency Maps
- Cross-Validation mit HCEye-Datensatz (Das et al., ETRA 2024) — zeigt Saliency-Änderung unter kognitivem Load

---

## 7. Cognitive Metrics: Jokinen 2020 Visual Search

### 7.1 Motivation & Thesis Gap

**Forschungsfrage:** *"Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?"*

**Gap:** AIM bietet visuelle/perzeptuelle Metriken (Farbe, Kanten, Clutter), aber **keine kognitiven** Vorhersagen wie:
- Wie lange dauert es, ein UI-Element zu finden? (Search Time)
- Wie viele Fixationen braucht ein Novize? (Fixation Count)
- Welche Elemente sind besonders schwer aufzufinden? (Difficulty Rating)

**Lösung:** Implementation des Jokinen 2020 Adaptive Feature Guidance Modells als Python-nativer Ersatz für die Legacy-Binärdatei (`vg2-linux`/`vg2-macos`) im alten AIM.

### 7.2 Theoretisches Modell

Das Modell simuliert visuelle Suche Fixierung für Fixierung:

1. **Controller** startet mit Eyes am Screen-Zentrum
2. **Activation** pro Element = Bottom-Up Saliency + Noise
3. **Winner-Take-All:** Element mit höchster Activation wird fixiert
4. **EMMA Encoding Time** (Salvucci, 2001): $T_e = K \cdot [-\ln(f)] \cdot e^{k \cdot \varepsilon}$
5. **Saccade Time** (Eq. 5): $T_s = t_{prep} + t_{exec} \cdot D + t_{sacc}$
6. **VSTM Inhibition:** Bereits besuchte Elemente werden gehemmt (Inhibition of Return)
7. **Terminierung:** Wenn Zielelement gefunden oder `max_fixations` erreicht

### 7.3 Schlüsselgleichungen

| Gleichung | Formel | Bedeutung |
|-----------|--------|-----------|
| Eq. 1 (Visual Threshold) | $\theta = a \cdot \varepsilon^2 + b \cdot \varepsilon$ | Feature sichtbar wenn $\theta < \alpha_{size}$ |
| Eq. 2 (Bottom-Up Activation) | $BA_i = \sum_j \sum_k \frac{dissim(v_{ik}, v_{jk})}{\sqrt{d_{ij}}}$ | Saliency aus Feature-Dissimilarität |
| Eq. 4 (EMMA Encoding) | $T_e = K \cdot [-\ln(f)] \cdot e^{k \cdot \varepsilon}$ | Encoding-Zeit steigt mit Exzentrizität |
| Eq. 5 (Saccade) | $T_s = t_{prep} + t_{exec} \cdot D + t_{sacc}$ | Sakkaden-Dauer |

### 7.4 Parameter (aus Paper Table 1)

| Parameter | Wert | Quelle |
|-----------|------|--------|
| K (EMMA encoding) | 0.006 | Salvucci (2001) |
| k (EMMA exponent) | 0.4 | Salvucci (2001) |
| t_prep (saccade prep) | 0.135 s | EMMA |
| t_exec (saccade per deg) | 0.002 s/° | EMMA |
| W_BA (bottom-up weight) | 1.1 | Nyamsuren & Taatgen (2013) |
| σ_TA (activation noise) | 0.376 | Nyamsuren & Taatgen (2013) |
| τ_VSTM (memory capacity) | 20 steps | Fitted (Jokinen 2020) |
| W_saliency (UMSI++ weight) | 0.8 | **Eigener Beitrag** |
| saliency_exponent | 2.0 | **Eigener Beitrag** |

### 7.5 Unsere Beiträge (vs. Legacy AIM)

| Aspekt | Legacy (vg2_visual_search.py) | Unsere Implementation |
|--------|-------------------------------|----------------------|
| Sprache | Compiled C++ binary | Python-nativ |
| Saliency | Hand-crafted (3 Farben, 3 Größen) | UMSI++ Deep Saliency |
| Element-Erkennung | Extern (Segmentierung nötig) | Automatisch (Canny + Contours) |
| Simulation | Unbekannt (Black-Box) | Monte Carlo (N=50–500 Trials) |
| Output | Heatmap-Bild | JSON: per-element time + statistics |
| Plattform | Linux/macOS x86 only | Plattformunabhängig (pure Python) |

### 7.6 Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  Screenshot (PNG/JPG)                                        │
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

### 7.8 Validierung

**Erwartete Ergebnisse (basierend auf Paper):**
- BP Layout (einfach, ~10 Elemente): mean ~1.5–3.0s → Difficulty: "moderate"
- WIN10 Desktop (~15 Elemente): mean ~2.0–3.5s → Difficulty: "moderate"/"difficult"
- NYT Front Page (~30+ Elemente): mean ~5.0–8.0s → Difficulty: "very_hard"

**Unser Ergebnis (bmw_route.png, 40 Elemente):**
- Mean: 6.07s, Max: 8.28s, Min: 2.48s → Difficulty: "very_hard"
- Konsistent mit Jokinen's NYT-Daten für komplexe Layouts

**Key Insight:** Element #0 (orange, einzige Farbe im Layout) wird am schnellsten gefunden (2.48s / 9.6 Fixationen) — exakt der Pop-Out-Effekt den das Modell vorhersagt.

### 7.9 Dateien

| Datei | Zweck |
|-------|-------|
| `cognitive/__init__.py` | Package-Definition |
| `cognitive/jokinen_model.py` | Kernmodell: JokinenSearchModel, JokinenParams |
| `cognitive/element_detector.py` | UI-Element-Erkennung (detect_elements) |
| `cognitive/test_jokinen.py` | End-to-End Test (element detection + model + UMSI++) |

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

# Load model (einmalig — danach wiederholbar für viele Bilder)
model = UMSIPlus("saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5")

# Predict saliency
heatmap, aux_classif = model.predict_saliency("screenshot.png", return_classif=True)
# heatmap: np.ndarray shape (H, W), float32 in [0,1]
# aux_classif: np.ndarray shape (6,), RAW auxiliary output — UNVALIDATED
#   (zero training-loss weight in the official setup); NOT a semantic
#   UI-type classifier and never exposed via the API.

# Extract scalar features
features = extract_saliency_features(heatmap)
# features = {"saliency_dispersion": 0.49, "saliency_peak_count": 7, ...}
```

### Web UI

```bash
python stage1/app.py
# Open http://localhost:5001
```

### cURL (API testen)

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

### Saliency Output (06.05.2026)

```
Saliency Features s ∈ ℝ⁵ (UMSI++):
  saliency_dispersion          0.4931
  saliency_peak_count          7
  saliency_center_bias         0.3090
  saliency_entropy             0.7169
  saliency_coverage            0.0642

Auxiliary output head (out_classif, 6-dim):
  UNVALIDATED — zero training-loss weight in the official setup.
  Not mapped to semantic labels and not returned by the API.
```

**Interpretation:**
- Dispersion 0.49 → Aufmerksamkeit moderat gestreut (nicht nur ein Punkt)
- 7 Peaks → mehrere distrinkte Aufmerksamkeits-Hotspots (Kartenelemente, Buttons, Text)
- Center Bias 0.31 → Aufmerksamkeit nicht besonders zentrumslastig (Karte füllt Rand-zu-Rand)
- Entropy 0.72 → relativ gleichmäßig verteilte Salienz
- Coverage 0.06 → nur 6.4% der Fläche erhält >50% der maximalen Salienz (wenige dominante Punkte)
- Der 6-dim Auxiliary-Head wird nicht interpretiert (unvalidiert, siehe §6.5)

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

4. All features are currently **unnormalized** (raw scale). Stage 2 should apply z-score or min-max normalization based on a reference dataset of automotive GUI screenshots.

5. The normalization constants in **Feature Congestion** (0.2088, 0.0660, 0.0269) were derived by Rosenholtz et al. for natural images. They may need recalibration for automotive GUI screenshots specifically.

### Stage 1b (Saliency — 06.05.2026)

6. **UMSI++ auf CPU only** — TF 2.16 auf M4 Mac nutzt nur die CPU. Inference dauert ~3–5s pro Bild. Metal-Plugin (tensorflow-metal) könnte GPU-Beschleunigung bringen.

7. **Auxiliary-Head unvalidiert und ungenutzt** — Der 6-dim `out_classif`-Head erhält im offiziellen Setup kein Trainingssignal (`loss_weights={'dec_c_cout': 1, 'out_classif': 0}`) und ist daher KEIN trainierter Klassifikator. Er wird nicht interpretiert, nicht als Feature verwendet und nicht über die API ausgegeben (siehe §6.5).

8. **Keine formale Validierung** — Die UMSI++ Predictions müssen noch gegen die UEyes Ground-Truth Saliency Maps evaluiert werden (KL, CC, NSS Metriken).

### Nächste Schritte (geplant)

| Priorität | Aufgabe | Relevanz |
|-----------|---------|----------|
| Hoch | DeepGaze++ / PathGAN++ portieren | Fixation-Verteilung für Stage 2 |
| Hoch | Validierung gegen UEyes Ground Truth | Qualitätsnachweis |
| Mittel | HCEye-Datensatz integrieren | Saliency unter kognitivem Load |
| Mittel | Stage 2 Multi-Output Head aufsetzen | Hauptziel der Thesis |
| Niedrig | UMSI++ GPU-Beschleunigung (Metal) | Performance-Optimierung |

---

## References

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27, 379–423.
2. Rosenholtz, R., Li, Y., & Nakano, L. (2007). "Measuring Visual Clutter." *Journal of Vision*, 7(2):17, 1–22.
3. Miniukovich, A., & De Angeli, A. (2015). "Computation of Interface Aesthetics." *CHI 2015*, pp. 1163–1172.
4. Hasler, D., & Süsstrunk, S. E. (2003). "Measuring Colourfulness in Natural Images." *SPIE Human Vision and Electronic Imaging VIII*.
5. Jiang, Y., et al. (2023). "UEyes: Understanding Visual Saliency across User Interface Types." *CHI 2023*.
6. Das, S., et al. (2024). "Shifting Focus with HCEye: How Cognitive Load Shapes Gaze Behavior on Webpages." *PACM HCI / ETRA 2024*.
