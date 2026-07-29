#!/usr/bin/env python3
"""Reproducible evidence generator for the canonical analysis resolution.

This is an offline audit / evidence script, not runtime code. It regenerates
every number behind ``CANONICAL_LONG_SIDE`` in ``stage1/visual_complexity.py``
and behind the corrective scale re-audit:

  1. candidate-resolution comparison (1024 / 1280 / 1440) on deterministic
     synthetic fixtures plus a deterministic UEyes *selection* sample;
  2. per-fixture scale tables reporting, at 1x / 2x / 3x, the raw absolute and
     relative gap of all eight Stage-1 features, the five percentile-normalised
     HCEye inputs, and the resulting HCEye rule index computed from visual
     features + whitespace only (no saliency, no OCR, no task/profile
     modifiers), so it is deliberately NOT the full endpoint
     ``cognitive_load_index``;
  3. a held-out real-UI scale evaluation on a seeded UEyes sample that EXCLUDES
     the images used to select 1280 (variants are raster-enlarged, not natively
     re-rendered, and are reported descriptively). Percentile-normalised inputs
     and any derived index are WITHHELD here as a domain mismatch until the
     separately authorized canonical-domain norms exist.

Design notes for auditability:
  * No absolute paths are hard-coded. Inputs default to repo-relative locations
    and can be overridden with ``--images-dir`` / ``--types-csv`` / ``--out-dir``.
  * Only the interactive GUI categories (desktop / mobile / web) are used; the
    non-interactive ``poster`` category is permanently excluded at every corpus
    boundary (see ``AUTHORIZED_CATEGORIES``).
  * The UEyes images themselves are never written to git; only numeric results
    and the sample manifest (IDs + category + seed) are emitted.
  * Feature perturbation is reported as a measured *feature change*, not as an
    information-theoretic loss claim. The perturbation tie-breaker performs one
    direct native-to-candidate resampling; it does not build artificial 2x/3x
    screenshot variants. Canonicalisation still upscales originals below the
    candidate long side, so this is not a "no upscaling" measurement.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "stage1"))

import cv2  # noqa: E402

from visual_complexity import (  # noqa: E402
    canonicalize_for_analysis,
    shannon_entropy,
    edge_density,
    feature_congestion,
    subband_entropy,
    layout_symmetry,
    chromatic_coherence,
    visual_hierarchy,
    interactive_element_density,
    FEATURE_KEYS,
    CANONICAL_LONG_SIDE,
)
from hceye.hceye_features import HCEyeFeatureExtractor, HCEYE_FEATURE_MAP  # noqa: E402

# The single authoritative definition of the interactive GUI population used for
# every corpus-facing operation in this tool. ``poster`` is a non-interactive
# graphic category and is PERMANENTLY excluded from category enumeration,
# decision-image selection, held-out sampling, corpus statistics and every
# emitted manifest. Every boundary function filters against this set itself; no
# boundary relies on a caller having pre-filtered the category dictionary.
AUTHORIZED_CATEGORIES = ("desktop", "mobile", "web")
_AUTHORIZED_SET = frozenset(AUTHORIZED_CATEGORIES)
EXCLUDED_CATEGORIES = ("poster",)

# Explicit, machine-readable marker for any normalized / derived-index field
# that cannot be presented as valid because the only available reference norms
# (feature_norms.json) were built at NATIVE resolution while these features are
# measured in the CANONICAL domain. It is emitted INSTEAD of a value — never a
# neutral constant or fallback — so a consumer cannot mistake a withheld score
# for a real one.
_NORMALIZED_UNAVAILABLE = {
    "status": "unavailable",
    "reason": "domain_mismatch_native_norms_vs_canonical_features",
    "detail": (
        "Percentile normalization and any derived HCEye/layout index require "
        "canonical-domain reference norms. The only available "
        "feature_norms.json was built at NATIVE resolution, so normalizing "
        "canonical-domain features against it would be a domain mismatch. "
        "These fields are withheld until the separately authorized canonical "
        "visual-feature norms are generated."),
}

# The three inherently pixel-scale-dependent features the diagnosis proved to
# drive the native-resolution headline defect. They are a SUBSET of the five
# HCEye-mapped features (see HCEYE_FEATURE_MAP); layout_symmetry and
# visual_hierarchy also feed the index and are reported in full below.
PIXEL_SCALE_DRIVERS = ("feature_congestion", "edge_density",
                       "interactive_element_density")

# All five Stage-1 features that feed the HCEye-derived index, in HCEye order.
HCEYE_MAPPED_STAGE1 = tuple(stage1 for stage1, _ in HCEYE_FEATURE_MAP.values())

_FEATURE_FUNCS = {
    "shannon_entropy": shannon_entropy,
    "edge_density": edge_density,
    "feature_congestion": feature_congestion,
    "subband_entropy": subband_entropy,
    "layout_symmetry": layout_symmetry,
    "chromatic_coherence": chromatic_coherence,
    "visual_hierarchy": visual_hierarchy,
    "interactive_element_density": interactive_element_density,
}


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures (shared with tests/test_canonical_scale.py).
# ---------------------------------------------------------------------------
def fx_hard_edged(s: int) -> np.ndarray:
    """Hard-edged UI: separated solid colour boxes on a light ground."""
    im = np.full((600 * s, 900 * s, 3), 245, np.uint8)
    boxes = [((60, 60), (180, 140), (200, 40, 40)),
             ((400, 80), (520, 160), (40, 160, 40)),
             ((700, 100), (820, 180), (40, 40, 200)),
             ((120, 360), (260, 460), (180, 120, 40)),
             ((520, 380), (660, 470), (120, 40, 160))]
    for (x1, y1), (x2, y2), c in boxes:
        cv2.rectangle(im, (x1 * s, y1 * s), (x2 * s, y2 * s), c, -1)
    return im


def fx_text_lines(s: int) -> np.ndarray:
    """Text / line-heavy UI: many thin horizontal strokes + rule lines."""
    im = np.full((600 * s, 900 * s, 3), 250, np.uint8)
    for i, y in enumerate(range(60, 560, 26)):
        x2 = 120 + (i * 37) % 700
        cv2.line(im, (60 * s, y * s), ((60 + x2) * s, y * s), (30, 30, 30),
                 max(1, 2 * s))
    cv2.line(im, (60 * s, 40 * s), (840 * s, 40 * s), (0, 0, 0), max(1, 3 * s))
    cv2.rectangle(im, (600 * s, 400 * s), (840 * s, 540 * s), (60, 120, 200),
                  max(1, 2 * s))
    return im


def fx_gradient(s: int) -> np.ndarray:
    """Gradient / low-detail UI: smooth horizontal ramp + one soft blob."""
    h, w = 600 * s, 900 * s
    grad = np.tile(np.linspace(40, 220, w, dtype=np.uint8), (h, 1))
    im = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    cv2.circle(im, (w // 2, h // 2), min(h, w) // 6, (120, 90, 60), -1)
    return im


FIXTURES = {
    "hard_edged": fx_hard_edged,
    "text_lines": fx_text_lines,
    "gradient": fx_gradient,
}


# ---------------------------------------------------------------------------
# Core analysis helpers (also imported by the regression test).
# ---------------------------------------------------------------------------
def features_at_long_side(img_bgr: np.ndarray, long_side: int) -> Dict[str, float]:
    """Canonicalise ``img_bgr`` to ``long_side`` then compute all 8 features.

    With ``long_side == CANONICAL_LONG_SIDE`` this is exactly the production
    feature path (``compute_complexity_vector``), minus the disk read.
    """
    canon = canonicalize_for_analysis(img_bgr, long_side=long_side)
    return {k: float(_FEATURE_FUNCS[k](canon)) for k in FEATURE_KEYS}


def canonical_features(img_bgr: np.ndarray) -> Dict[str, float]:
    """Production 8-feature vector at the selected canonical resolution."""
    return features_at_long_side(img_bgr, CANONICAL_LONG_SIDE)


def _extractor() -> HCEyeFeatureExtractor:
    lookup = os.path.join(_REPO_ROOT, "hceye", "sensitivity_lookup.json")
    return HCEyeFeatureExtractor(lookup if os.path.exists(lookup) else None)


def normalized_hceye_inputs(vis: Dict[str, float],
                            extractor: Optional[HCEyeFeatureExtractor] = None
                            ) -> Dict[str, float]:
    """Percentile-normalised value of each of the five HCEye-mapped features."""
    ex = extractor or _extractor()
    return {stage1: float(ex._percentile_normalize(vis.get(stage1), stage1))
            for stage1 in HCEYE_MAPPED_STAGE1}


def hceye_rule_index_no_saliency_no_ocr(
        vis: Dict[str, float],
        whitespace_ratio: Optional[float] = None,
        extractor: Optional[HCEyeFeatureExtractor] = None) -> float:
    """HCEye rule index (h[5]) from visual features + whitespace ONLY.

    This deliberately omits the saliency term, the OCR-derived text_density and
    the task/profile modifiers, so it is NOT the full endpoint
    ``cognitive_load_index``. It isolates the visual-feature + whitespace
    contribution, which is exactly the part whose resolution behaviour this
    audit measures. It is named accordingly to avoid overclaiming.
    """
    ex = extractor or _extractor()
    h = ex.extract_features(vis, whitespace_ratio=whitespace_ratio)
    return float(h[5])


def canonical_layout_stats(img_bgr: np.ndarray) -> Dict[str, object]:
    """Score-driving layout stats via the PRODUCTION helper (analysis path).

    Calls ``measure_canonical_layout`` — the exact code the endpoint runs — so
    this evaluation cannot drift from runtime whitespace/element behaviour and
    contains no separate reimplementation of production whitespace logic. OCR is
    disabled here (``run_ocr=False``): OCR is heavy and nondeterministic and is
    not the variable under test; whitespace and element geometry are what the
    scale audit measures. Bounding boxes are normalised by the CANONICAL canvas
    (long side 1280), matching the space in which the score is computed.
    """
    from canonical_layout import measure_canonical_layout
    m = measure_canonical_layout(img_bgr, run_ocr=False)
    ah, aw = m.analysis_shape
    norm_boxes = []
    for e in m.analysis_elements:
        x, y, bw, bh = e["bbox"]
        norm_boxes.append((x / aw, y / ah, bw / aw, bh / ah))
    return {
        "element_count": int(m.element_count),
        "whitespace_ratio": float(m.whitespace_ratio),
        "norm_covered_ratio": 1.0 - float(m.whitespace_ratio),
        "norm_boxes": norm_boxes,
        "analysis_long_side": int(m.analysis_long_side),
    }


def native_whitespace_stats(img_bgr: np.ndarray) -> Dict[str, object]:
    """Old NATIVE-resolution whitespace, kept only to reproduce the baseline.

    This is the pre-fix behaviour (element detection + whitespace on the native
    image). It is retained solely so the held-out report can REPRODUCE the
    baseline decomposition numbers and show, side by side, how the canonical
    path removes the native scale defect. It no longer feeds any score.
    """
    from cognitive.element_detector import detect_elements
    elements = detect_elements(img_bgr)
    h_img, w_img = img_bgr.shape[:2]
    area = float(h_img * w_img) or 1.0
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for e in elements:
        x, y, bw, bh = e["bbox"]
        mask[int(y):int(y + bh), int(x):int(x + bw)] = 1
    whitespace = float(np.clip(1.0 - float(mask.sum()) / area, 0.0, 1.0))
    return {"element_count": int(len(elements)), "whitespace_ratio": whitespace}


def _gap(vals: List[float]) -> Tuple[float, float]:
    absgap = max(vals) - min(vals)
    denom = max(abs(float(np.mean(vals))), 1e-9)
    return absgap, absgap / denom


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def fixture_scale_report(scales: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, dict]:
    """Per-fixture raw 8-feature gaps + normalised HCEye inputs + index effect."""
    ex = _extractor()
    out: Dict[str, dict] = {}
    for name, fx in FIXTURES.items():
        feats = {s: canonical_features(fx(s)) for s in scales}
        raw = {}
        for k in FEATURE_KEYS:
            vals = [feats[s][k] for s in scales]
            a, r = _gap(vals)
            raw[k] = {"by_scale": {str(s): feats[s][k] for s in scales},
                      "abs_gap": a, "rel_gap": r,
                      "hceye_mapped": k in HCEYE_MAPPED_STAGE1,
                      "pixel_scale_driver": k in PIXEL_SCALE_DRIVERS}
        norm = {s: normalized_hceye_inputs(feats[s], ex) for s in scales}
        norm_gaps = {}
        for k in HCEYE_MAPPED_STAGE1:
            vals = [norm[s][k] for s in scales]
            a, r = _gap(vals)
            norm_gaps[k] = {"by_scale": {str(s): norm[s][k] for s in scales},
                            "abs_gap": a, "rel_gap": r}
        idx = {s: hceye_rule_index_no_saliency_no_ocr(feats[s], extractor=ex)
               for s in scales}
        ia, ir = _gap([idx[s] for s in scales])
        out[name] = {
            "raw_features": raw,
            "normalized_hceye_inputs": norm_gaps,
            "hceye_rule_index_no_saliency_no_ocr": {
                "by_scale": {str(s): idx[s] for s in scales},
                "abs_gap": ia, "rel_gap": ir},
        }
    return out


def _print_fixture_report(rep: Dict[str, dict]) -> None:
    for name, d in rep.items():
        print(f"\n=== fixture: {name} ===")
        print("  raw 8-feature 1x/2x/3x (M=HCEye-mapped, D=pixel-scale driver):")
        for k in FEATURE_KEYS:
            r = d["raw_features"][k]
            tag = ("D" if r["pixel_scale_driver"] else
                   ("M" if r["hceye_mapped"] else " "))
            bs = r["by_scale"]
            print(f"    [{tag}] {k:28s} "
                  f"1x={bs['1']:10.5f} 2x={bs['2']:10.5f} 3x={bs['3']:10.5f} "
                  f"abs={r['abs_gap']:9.5f} rel={r['rel_gap']:7.2%}")
        print("  normalized HCEye inputs (percentile in [0,1]):")
        for k in HCEYE_MAPPED_STAGE1:
            r = d["normalized_hceye_inputs"][k]
            bs = r["by_scale"]
            print(f"        {k:28s} "
                  f"1x={bs['1']:.4f} 2x={bs['2']:.4f} 3x={bs['3']:.4f} "
                  f"abs={r['abs_gap']:.4f} rel={r['rel_gap']:7.2%}")
        e = d["hceye_rule_index_no_saliency_no_ocr"]
        bs = e["by_scale"]
        print(f"  HCEye rule index (no saliency, no OCR) "
              f"1x={bs['1']:.5f} 2x={bs['2']:.5f} 3x={bs['3']:.5f} "
              f"abs={e['abs_gap']:.5f} rel={e['rel_gap']:.2%}")


def native_features(img_bgr: np.ndarray) -> Dict[str, float]:
    """Eight features computed on the image at its NATIVE size (no canonical
    resize), used as the reference for feature-perturbation measurements."""
    return {k: float(_FEATURE_FUNCS[k](img_bgr)) for k in FEATURE_KEYS}


def candidate_comparison(candidates=(1024, 1280, 1440),
                         scales=(1, 2, 3)) -> List[dict]:
    """Primary objective: pixel-scale-driver scale gap per candidate.

    Measured ONLY on the deterministic synthetic fixtures, which are *re-rendered*
    natively at each integer scale. Re-rendering is the correct model of "the
    same layout captured at a different resolution/DPI"; raster-upscaling a real
    screenshot is NOT (it is a lossy round-trip and, worse, biases the ranking
    toward candidates far below the native size because the 1x no-op disappears).
    Real UEyes images are therefore handled by ``ueyes_perturbation`` (a fair,
    single-pass perturbation/cost tie-breaker), not by this scale objective.
    """
    rows = []
    for cand in candidates:
        rels = []
        for name, fx in FIXTURES.items():
            feats = {s: features_at_long_side(fx(s), cand) for s in scales}
            for k in PIXEL_SCALE_DRIVERS:
                _, r = _gap([feats[s][k] for s in scales])
                rels.append(r)
        rows.append({"candidate": cand,
                     "mean_rel_gap": float(np.mean(rels)),
                     "worst_rel_gap": float(np.max(rels))})
    return rows


def ueyes_perturbation(sel_images: List[Tuple[str, str, str]],
                       candidates=(1024, 1280, 1440)) -> List[dict]:
    """Tie-breaker: how much does forcing each candidate perturb REAL images.

    For every UEyes selection image, compares the eight features at the image's
    NATIVE size against the features after a SINGLE direct canonicalisation to
    the candidate long side. This performs one native-to-candidate resampling
    and does NOT first build artificial 2x/3x screenshot variants. Note that
    canonicalisation DOES upscale originals whose native long side is below the
    candidate (a real resampling that perturbs features); this is not a
    "no upscaling" measurement. Reports the mean and worst relative feature
    change across the sample. Smaller = the candidate leaves real screenshots
    closer to their native measurement.
    """
    rows = []
    for cand in candidates:
        rels = []
        for img_id, cat, path in sel_images:
            base = cv2.imread(path)
            if base is None:
                continue
            nat = native_features(base)
            can = features_at_long_side(base, cand)
            for k in FEATURE_KEYS:
                _, r = _gap([nat[k], can[k]])
                rels.append(r)
        rows.append({"candidate": cand,
                     "mean_rel_perturbation": float(np.mean(rels)) if rels else 0.0,
                     "worst_rel_perturbation": float(np.max(rels)) if rels else 0.0})
    return rows


def _iou(a: Tuple[float, float, float, float],
         b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes in the same space.

    The result is clamped to [0, 1] so floating-point rounding can never
    report an IoU outside the mathematically valid range.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(min(max(inter / union, 0.0), 1.0))


def _match_boxes(ref_boxes: List[tuple], cand_boxes: List[tuple],
                 iou_thresh: float = 0.5) -> Dict[str, object]:
    """Greedy best-IoU one-to-one matching of a 1x reference box set against a
    scaled candidate box set.

    Semantics:
      * If the 1x reference set is EMPTY the comparison is NOT APPLICABLE
        (``applicable=False``); coverages and mean IoU are returned as ``None``
        rather than a misleading 0.0.
      * ``reference_coverage`` is the DIRECTIONAL fraction of 1x-reference boxes
        that found a match (it can be 1.0 even when the scaled image detects
        EXTRA boxes).
      * ``candidate_coverage`` is the directional fraction of scaled-image boxes
        that were matched, so spurious extra 2x/3x detections are visible and
        cannot be hidden behind ``reference_coverage == 1.0``.
      * ``symmetric_coverage`` = 2 * matched / (n_ref + n_cand) collapses both
        directions into a single figure.
      * ``mean_iou_matched`` averages IoU over matched pairs only.
    """
    n_ref = len(ref_boxes)
    n_cand = len(cand_boxes)
    if n_ref == 0:
        return {
            "applicable": False,
            "n_reference_boxes": n_ref,
            "n_candidate_boxes": n_cand,
            "n_matched": 0,
            "reference_coverage": None,
            "candidate_coverage": None,
            "symmetric_coverage": None,
            "mean_iou_matched": None,
        }
    used: set = set()
    ious: List[float] = []
    for a in ref_boxes:
        best, best_j = -1.0, -1
        for j, b in enumerate(cand_boxes):
            if j in used:
                continue
            v = _iou(a, b)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= iou_thresh:
            used.add(best_j)
            ious.append(best)
    n_matched = len(ious)
    candidate_coverage = (n_matched / n_cand) if n_cand else None
    symmetric_coverage = (2.0 * n_matched / (n_ref + n_cand))
    return {
        "applicable": True,
        "n_reference_boxes": n_ref,
        "n_candidate_boxes": n_cand,
        "n_matched": n_matched,
        "reference_coverage": float(n_matched / n_ref),
        "candidate_coverage": (float(candidate_coverage)
                               if candidate_coverage is not None else None),
        "symmetric_coverage": float(symmetric_coverage),
        "mean_iou_matched": float(np.mean(ious)) if ious else 0.0,
    }


def heldout_ueyes(sample: List[Tuple[str, str, str]],
                  scales=(1, 2, 3)) -> List[dict]:
    """Descriptive scale evaluation for held-out real UEyes screenshots.

    Emits RAW visual features, canonical layout measurements (whitespace,
    element geometry, normalised bounding boxes) and provenance ONLY.

    Percentile-normalised HCEye inputs and any derived HCEye/layout index are
    DELIBERATELY WITHHELD here: the only available ``feature_norms.json`` was
    built at NATIVE resolution, so percentile-normalising these CANONICAL-domain
    features against it would be a domain mismatch. Those fields are emitted as
    an explicit machine-readable ``unavailable`` marker (never a neutral
    constant or fallback value), pending the separately authorized canonical
    visual-feature norm generation.

    IMPORTANT: real screenshots exist at a single native capture only, so the
    2x/3x variants here are produced by RASTER ENLARGEMENT (cv2.resize) — one
    direct resampling, not a native re-render at a higher resolution, and NOT an
    artificial 2x/3x screenshot pipeline. Results are reported descriptively;
    the >=1x synthetic 1.0-point acceptance guard is NOT applied to them.
    """
    results = []
    for img_id, cat, path in sample:
        base = cv2.imread(path)
        if base is None:
            continue
        ws_canon: Dict[int, Optional[float]] = {}
        ws_native: Dict[int, Optional[float]] = {}
        norm_boxes_by_scale: Dict[int, List[tuple]] = {}
        per_scale = {}
        analysis_long_side = None
        for s in scales:
            if s == 1:
                v = base
            else:
                v = cv2.resize(base, (base.shape[1] * s, base.shape[0] * s),
                               interpolation=cv2.INTER_LINEAR)
            vis = canonical_features(v)
            can = canonical_layout_stats(v)
            nat = native_whitespace_stats(v)
            ws_canon[s] = can["whitespace_ratio"]
            ws_native[s] = nat["whitespace_ratio"]
            norm_boxes_by_scale[s] = [tuple(b) for b in can["norm_boxes"]]
            analysis_long_side = can["analysis_long_side"]
            per_scale[s] = {
                "raw_features": vis,
                "normalized_hceye_inputs": _NORMALIZED_UNAVAILABLE,
                "canonical_whitespace_ratio": can["whitespace_ratio"],
                "canonical_element_count": can["element_count"],
                "canonical_norm_boxes": [list(b) for b in can["norm_boxes"]],
                "native_whitespace_ratio": nat["whitespace_ratio"],
                "native_element_count": nat["element_count"],
                "hceye_rule_index_no_saliency_no_ocr_canonical": _NORMALIZED_UNAVAILABLE,
                "hceye_rule_index_no_saliency_no_ocr_native": _NORMALIZED_UNAVAILABLE,
            }

        match_12 = _match_boxes(norm_boxes_by_scale[scales[0]],
                                norm_boxes_by_scale.get(2, []))
        match_13 = _match_boxes(norm_boxes_by_scale[scales[0]],
                                norm_boxes_by_scale.get(3, []))

        ws_canon_vals = [ws_canon[s] for s in scales]
        ws_native_vals = [ws_native[s] for s in scales]
        ec_vals = [per_scale[s]["canonical_element_count"] for s in scales]
        results.append({
            "image_id": img_id,
            "category": cat,
            "scale_variant_method": "raster_enlargement_cv2_resize",
            "analysis_provenance": {
                "analysis_path": f"canonical-analysis:long{CANONICAL_LONG_SIDE}",
                "analysis_long_side": int(analysis_long_side)
                if analysis_long_side else CANONICAL_LONG_SIDE,
            },
            "normalized_scores_status": _NORMALIZED_UNAVAILABLE,
            "per_scale": {str(s): per_scale[s] for s in scales},
            "decomposition_canonical_path": _NORMALIZED_UNAVAILABLE,
            "decomposition_native_path_baseline": _NORMALIZED_UNAVAILABLE,
            "canonical_whitespace_abs_gap":
                max(ws_canon_vals) - min(ws_canon_vals),
            "native_whitespace_abs_gap":
                max(ws_native_vals) - min(ws_native_vals),
            "canonical_element_count_range": [int(min(ec_vals)), int(max(ec_vals))],
            "bbox_stability": {
                "matching_method": "greedy best-IoU one-to-one, IoU>=0.5, "
                                   "canonical-normalised boxes; 1x is the "
                                   "reference set",
                "not_applicable_reason": (
                    "1x reference detected zero boxes"
                    if not match_12["applicable"] else None),
                "match_1x_to_2x": match_12,
                "match_1x_to_3x": match_13,
            },
        })
    return results


# ---------------------------------------------------------------------------
# UEyes sampling
# ---------------------------------------------------------------------------
def _load_types(types_csv: str) -> Dict[str, str]:
    """Load ``image name -> category`` for AUTHORIZED categories only.

    ``poster`` (and any other non-authorized category) is dropped here, at the
    single corpus entry point, so no downstream selection / sampling / statistic
    can ever observe it. Boundary functions additionally re-filter defensively.
    """
    cat_by_id: Dict[str, str] = {}
    with open(types_csv, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = (row.get("Image Name") or "").strip()
            cat = (row.get("Category") or "").strip().lower()
            if name and cat in _AUTHORIZED_SET:
                cat_by_id[name] = cat
    return cat_by_id


def _authorized_only(cat_by_id: Dict[str, str]) -> Dict[str, str]:
    """Return only the entries whose category is authorized (defensive filter)."""
    return {n: c for n, c in cat_by_id.items()
            if (c or "").strip().lower() in _AUTHORIZED_SET}


def _resolve(images_dir: str, name: str) -> Optional[str]:
    p = os.path.join(images_dir, name)
    if os.path.exists(p):
        return p
    stem = os.path.splitext(name)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        alt = os.path.join(images_dir, stem + ext)
        if os.path.exists(alt):
            return alt
    return None


def selection_images(images_dir: str, cat_by_id: Dict[str, str]
                     ) -> List[Tuple[str, str, str]]:
    """The deterministic images used to SELECT 1280: first + middle per category.

    Filters to AUTHORIZED categories itself, so a caller passing an unfiltered
    dictionary (containing ``poster``) can never leak a poster into the
    selection.
    """
    by_cat: Dict[str, List[str]] = {}
    for name, cat in _authorized_only(cat_by_id).items():
        by_cat.setdefault(cat, []).append(name)
    chosen = []
    for cat in sorted(by_cat):
        names = sorted(by_cat[cat])
        for idx in (0, len(names) // 2):
            name = names[idx]
            path = _resolve(images_dir, name)
            if path:
                chosen.append((os.path.splitext(name)[0], cat, path))
    return chosen


def heldout_sample(images_dir: str, cat_by_id: Dict[str, str],
                   exclude_ids: set, seed: int, per_cat: int
                   ) -> List[Tuple[str, str, str]]:
    """Seeded random sample per category, EXCLUDING the selection images.

    Filters to AUTHORIZED categories itself before the per-category RNG draw, so
    ``poster`` can never enter the held-out sample even if an unfiltered
    dictionary is passed.
    """
    rng = np.random.default_rng(seed)
    by_cat: Dict[str, List[str]] = {}
    for name, cat in _authorized_only(cat_by_id).items():
        if os.path.splitext(name)[0] in exclude_ids:
            continue
        by_cat.setdefault(cat, []).append(name)
    sample = []
    for cat in sorted(by_cat):
        names = sorted(by_cat[cat])
        picks = rng.choice(len(names), size=min(per_cat, len(names)),
                           replace=False)
        for i in sorted(int(p) for p in picks):
            path = _resolve(images_dir, names[i])
            if path:
                sample.append((os.path.splitext(names[i])[0], cat, path))
    return sample


# ---------------------------------------------------------------------------
# Reproducible corpus statistics + environment (Findings 3 / D)
# ---------------------------------------------------------------------------
def _image_long_side(path: str) -> Optional[int]:
    """Return the native long side of an image, reading only the header.

    Uses PIL's lazy header read (no full decode) when available, falling back to
    a full OpenCV decode. Returns ``None`` if the file cannot be read.
    """
    try:
        from PIL import Image  # lazy; header read only, no pixel decode
        with Image.open(path) as im:
            w, h = im.size
            return int(max(w, h))
    except Exception:
        img = cv2.imread(path)
        if img is None:
            return None
        return int(max(img.shape[:2]))


def _percentiles(vals: List[int]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def corpus_stats(images_dir: str, types_csv: str,
                 candidates=(1024, 1280, 1440)) -> Dict[str, object]:
    """Compute reproducible authorized-corpus statistics from the actual files.

    Every number the resolution note cites (authorized counts, total,
    poster-excluded count, native long-side percentiles, and the per-candidate
    upscale/downscale fractions) is derived here from the corpus on disk, so no
    corpus claim exists that this executable output cannot reproduce.
    """
    import collections
    authorized_counts: Dict[str, int] = collections.Counter()
    excluded_counts: Dict[str, int] = collections.Counter()
    with open(types_csv, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = (row.get("Image Name") or "").strip()
            cat = (row.get("Category") or "").strip().lower()
            if not name or not cat:
                continue
            if cat in _AUTHORIZED_SET:
                authorized_counts[cat] += 1
            else:
                excluded_counts[cat] += 1

    long_sides: List[int] = []
    long_sides_by_cat: Dict[str, List[int]] = {c: [] for c in AUTHORIZED_CATEGORIES}
    cat_by_id = _load_types(types_csv)  # authorized-only
    for name, cat in sorted(cat_by_id.items()):
        path = _resolve(images_dir, name)
        if path is None:
            continue
        ls = _image_long_side(path)
        if ls is None:
            continue
        long_sides.append(ls)
        long_sides_by_cat[cat].append(ls)

    total_authorized = sum(authorized_counts.values())
    total_excluded = sum(excluded_counts.values())
    n_measured = len(long_sides)

    scale_fractions = []
    for cand in candidates:
        up = sum(1 for ls in long_sides if ls <= cand)
        down = n_measured - up
        scale_fractions.append({
            "candidate": cand,
            "upscaled_fraction": (up / n_measured) if n_measured else None,
            "downscaled_fraction": (down / n_measured) if n_measured else None,
            "note": ("fraction whose native long side <= candidate is upscaled "
                     "(resampled larger); the remainder is downscaled. Both "
                     "operations perturb feature values relative to native."),
        })

    return {
        "authorized_categories": list(AUTHORIZED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "authorized_counts_by_category": dict(authorized_counts),
        "total_authorized": total_authorized,
        "excluded_counts_by_category": dict(excluded_counts),
        "total_excluded": total_excluded,
        "native_long_side_measured_images": n_measured,
        "native_long_side_summary_px": _percentiles(long_sides) if long_sides else None,
        "native_long_side_summary_by_category_px": {
            c: (_percentiles(v) if v else None)
            for c, v in long_sides_by_cat.items()},
        "candidate_scale_fractions": scale_fractions,
    }


def environment_info() -> Dict[str, str]:
    """Environment-specific context for interpreting observed runtimes.

    Runtimes emitted by this tool are OBSERVED measurements on this machine, not
    universal benchmarks; this block records what is needed to contextualise
    them.
    """
    import platform
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "runtime_note": ("Elapsed times are environment-specific observed "
                         "measurements on the platform above, not a universal "
                         "benchmark."),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir",
                    default=os.path.join(_REPO_ROOT, "ueyes", "dataset_full",
                                         "UEyes_dataset", "images"))
    ap.add_argument("--types-csv",
                    default=os.path.join(_REPO_ROOT, "ueyes", "dataset_full",
                                         "UEyes_dataset", "image_types.csv"))
    ap.add_argument("--out-dir",
                    default=os.path.join(_REPO_ROOT, "stage1", "canonical_eval"))
    ap.add_argument("--seed", type=int, default=20240607)
    ap.add_argument("--per-cat", type=int, default=2,
                    help="held-out images sampled per category")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    have_ueyes = os.path.exists(args.images_dir) and os.path.exists(args.types_csv)

    import time
    phase_times: Dict[str, float] = {}
    env = environment_info()
    print("== Environment (context for observed runtimes; not a benchmark) ==")
    print(f"    python={env['python_version']} numpy={env['numpy_version']} "
          f"opencv={env['opencv_version']}")
    print(f"    platform={env['platform']}")

    _t = time.perf_counter()
    print("\n== Fixture scale report (synthetic, deterministic) ==")
    fx_rep = fixture_scale_report()
    _print_fixture_report(fx_rep)
    with open(os.path.join(args.out_dir, "fixture_scale_report.json"), "w") as f:
        json.dump(fx_rep, f, indent=2)
    phase_times["fixture_scale_report_s"] = time.perf_counter() - _t

    stats: Optional[Dict[str, object]] = None
    if have_ueyes:
        _t = time.perf_counter()
        print("\n== Authorized-corpus statistics (poster excluded) ==")
        stats = corpus_stats(args.images_dir, args.types_csv)
        phase_times["corpus_stats_s"] = time.perf_counter() - _t
        print(f"    authorized categories: {stats['authorized_categories']} "
              f"excluded: {stats['excluded_categories']}")
        print(f"    authorized counts: {stats['authorized_counts_by_category']} "
              f"total={stats['total_authorized']}")
        print(f"    excluded counts:   {stats['excluded_counts_by_category']} "
              f"total={stats['total_excluded']}")
        ls = stats["native_long_side_summary_px"]
        if ls:
            print(f"    native long side (n={ls['n']}): min={ls['min']:.0f} "
                  f"p25={ls['p25']:.0f} median={ls['median']:.0f} "
                  f"p75={ls['p75']:.0f} max={ls['max']:.0f}")
        for sf in stats["candidate_scale_fractions"]:
            print(f"    long={sf['candidate']:5d}  "
                  f"upscaled={sf['upscaled_fraction']:.4f}  "
                  f"downscaled={sf['downscaled_fraction']:.4f}")
        with open(os.path.join(args.out_dir, "corpus_stats.json"), "w") as f:
            json.dump({"environment": env, "corpus": stats}, f, indent=2)

    sel: List[Tuple[str, str, str]] = []
    heldout: List[Tuple[str, str, str]] = []
    if have_ueyes:
        cats = _load_types(args.types_csv)  # authorized-only (poster dropped)
        sel = selection_images(args.images_dir, cats)
        sel_ids = {i for i, _, _ in sel}
        heldout = heldout_sample(args.images_dir, cats, sel_ids, args.seed,
                                 args.per_cat)
        assert all(c in _AUTHORIZED_SET for _, c, _ in sel), "poster in selection"
        assert all(c in _AUTHORIZED_SET for _, c, _ in heldout), "poster in held-out"
        print("\n== UEyes selection images (used to choose 1280; authorized only) ==")
        for i, c, _ in sel:
            print(f"    {i}  {c}")
        print(f"== UEyes held-out sample (seed={args.seed}, excludes selection, "
              f"authorized only) ==")
        for i, c, _ in heldout:
            print(f"    {i}  {c}")
    else:
        print("\n[warn] UEyes dataset not found at --images-dir/--types-csv; "
              "skipping UEyes candidate + held-out sections.")

    _t = time.perf_counter()
    print("\n== Candidate-resolution comparison (synthetic re-render, "
          "pixel-scale drivers) ==")
    cand_rows = candidate_comparison()
    for r in cand_rows:
        print(f"    long_side={r['candidate']:5d}  "
              f"mean_rel={r['mean_rel_gap']:.4%}  worst_rel={r['worst_rel_gap']:.4%}")
    phase_times["candidate_scale_gap_s"] = time.perf_counter() - _t

    pert_rows = []
    if sel:
        _t = time.perf_counter()
        print("\n== Candidate tie-breaker: single native->candidate resampling "
              "feature perturbation on UEyes selection images ==")
        pert_rows = ueyes_perturbation(sel)
        for r in pert_rows:
            print(f"    long_side={r['candidate']:5d}  "
                  f"mean_pert={r['mean_rel_perturbation']:.4%}  "
                  f"worst_pert={r['worst_rel_perturbation']:.4%}")
        phase_times["ueyes_perturbation_s"] = time.perf_counter() - _t

    with open(os.path.join(args.out_dir, "candidate_comparison.json"), "w") as f:
        json.dump({"seed": args.seed,
                   "environment": env,
                   "selection_ids": [i for i, _, _ in sel],
                   "scale_objective_source": "synthetic re-rendered fixtures",
                   "perturbation_method": (
                       "one direct native-to-candidate resampling per image; no "
                       "artificial 2x/3x screenshot variants are created. Note "
                       "that canonicalisation DOES upscale originals whose long "
                       "side is below the candidate."),
                   "scale_gap": cand_rows,
                   "ueyes_native_to_candidate_perturbation": pert_rows}, f, indent=2)
    with open(os.path.join(args.out_dir, "candidate_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["candidate_long_side", "driver_scale_mean_rel_gap",
                    "driver_scale_worst_rel_gap", "ueyes_mean_rel_perturbation",
                    "ueyes_worst_rel_perturbation"])
        pert_by = {r["candidate"]: r for r in pert_rows}
        for r in cand_rows:
            p = pert_by.get(r["candidate"], {})
            w.writerow([r["candidate"],
                        f"{r['mean_rel_gap']:.6f}", f"{r['worst_rel_gap']:.6f}",
                        f"{p.get('mean_rel_perturbation', float('nan')):.6f}",
                        f"{p.get('worst_rel_perturbation', float('nan')):.6f}"])

    if heldout:
        _t = time.perf_counter()
        print("\n== Held-out UEyes scale evaluation "
              "(raster enlargement, descriptive; normalized scores gated) ==")
        ho = heldout_ueyes(heldout)
        for r in ho:
            bb = r["bbox_stability"]
            print(f"    {r['image_id']} ({r['category']}):  "
                  f"canonical_ws_abs_gap={r['canonical_whitespace_abs_gap']:.4f}  "
                  f"native_ws_abs_gap={r['native_whitespace_abs_gap']:.4f}  "
                  f"elem_range={r['canonical_element_count_range']}")
            print(f"        normalized/index scores: "
                  f"{r['normalized_scores_status']['status']} "
                  f"({r['normalized_scores_status']['reason']})")

            def _fmt_match(m: dict) -> str:
                if not m["applicable"]:
                    return "N/A (1x has zero boxes)"
                cc = ("n/a" if m["candidate_coverage"] is None
                      else f"{m['candidate_coverage']:.2f}")
                return (f"ref_cov={m['reference_coverage']:.2f} "
                        f"cand_cov={cc} sym_cov={m['symmetric_coverage']:.2f} "
                        f"iou={m['mean_iou_matched']:.3f} "
                        f"(ref={m['n_reference_boxes']} "
                        f"cand={m['n_candidate_boxes']})")

            print(f"        bbox 1x->2x {_fmt_match(bb['match_1x_to_2x'])}")
            print(f"        bbox 1x->3x {_fmt_match(bb['match_1x_to_3x'])}")
        phase_times["heldout_ueyes_s"] = time.perf_counter() - _t
        with open(os.path.join(args.out_dir, "heldout_ueyes_results.json"),
                  "w") as f:
            json.dump({"seed": args.seed,
                       "environment": env,
                       "scale_variant_method": "raster_enlargement_cv2_resize",
                       "note": ("Real screenshots are raster-enlarged (one "
                                "direct cv2.resize), not natively re-rendered; "
                                "the synthetic 1.0-point guard is not applied "
                                "here. Percentile-normalized HCEye inputs and "
                                "any derived index are WITHHELD (see "
                                "normalized_scores_status): the only available "
                                "feature_norms.json is native-domain, so "
                                "canonical-domain normalization is a domain "
                                "mismatch and remains pending the separately "
                                "authorized canonical-norm generation."),
                       "normalized_scores_status": _NORMALIZED_UNAVAILABLE,
                       "results": ho}, f, indent=2)
        with open(os.path.join(args.out_dir, "heldout_ueyes_manifest.csv"), "w",
                  newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["image_id", "category", "seed"])
            for i, c, _ in heldout:
                w.writerow([i, c, args.seed])

    print("\n== Phase timings (observed, environment-specific) ==")
    for k, v in phase_times.items():
        print(f"    {k}: {v:.2f}s")
    with open(os.path.join(args.out_dir, "run_environment.json"), "w") as f:
        json.dump({"environment": env, "phase_times_s": phase_times,
                   "seed": args.seed,
                   "authorized_categories": list(AUTHORIZED_CATEGORIES),
                   "excluded_categories": list(EXCLUDED_CATEGORIES)}, f, indent=2)

    print(f"\nArtifacts written to {os.path.relpath(args.out_dir, _REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    # Feature functions are quiet; only guard against accidental library prints.
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass
    raise SystemExit(main())
