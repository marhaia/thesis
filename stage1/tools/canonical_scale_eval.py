#!/usr/bin/env python3
"""Reproducible evidence generator for the canonical analysis resolution.

This is an offline audit / evidence script, not runtime code. It regenerates
every number behind ``CANONICAL_LONG_SIDE`` in ``stage1/visual_complexity.py``
and behind the corrective scale re-audit:

  1. candidate-resolution comparison (1024 / 1280 / 1440) on deterministic
     synthetic fixtures plus a deterministic UEyes *selection* sample;
  2. per-fixture scale tables reporting, at 1x / 2x / 3x, the raw absolute and
     relative gap of all eight Stage-1 features, the five percentile-normalised
     HCEye inputs, and the resulting HCEye cognitive-load index (the endpoint's
     ``cognitive_load_index`` field);
  3. a held-out real-UI scale evaluation on a seeded UEyes sample that EXCLUDES
     the images used to select 1280.

Design notes for auditability:
  * No absolute paths are hard-coded. Inputs default to repo-relative locations
    and can be overridden with ``--images-dir`` / ``--types-csv`` / ``--out-dir``.
  * The UEyes images themselves are never written to git; only numeric results
    and the sample manifest (IDs + category + seed) are emitted.
  * Feature perturbation is reported as a measured *feature change*, not as an
    information-theoretic loss claim.
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


def hceye_load_index(vis: Dict[str, float],
                     whitespace_ratio: Optional[float] = None,
                     extractor: Optional[HCEyeFeatureExtractor] = None) -> float:
    """HCEye cognitive-load index (h[5]) — the endpoint ``cognitive_load_index``.

    This is the estimate path the live app always takes for novel screenshots
    (no ``image_name`` is passed), so it faithfully reflects the endpoint's
    scale behaviour for the visual-feature contribution.
    """
    ex = extractor or _extractor()
    h = ex.extract_features(vis, whitespace_ratio=whitespace_ratio)
    return float(h[5])


def detect_stats(img_bgr: np.ndarray) -> Dict[str, object]:
    """Replicate the endpoint's native-resolution element / whitespace measures.

    Element detection and whitespace run on the ORIGINAL (native) image in the
    app, deliberately NOT on the canonical image, so this reports them at native
    scale to expose any residual scale sensitivity in that path.
    """
    from cognitive.element_detector import detect_elements
    elements = detect_elements(img_bgr)
    h_img, w_img = img_bgr.shape[:2]
    area = float(h_img * w_img) or 1.0
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    norm_boxes = []
    for e in elements:
        x, y, bw, bh = e["bbox"]
        mask[int(y):int(y + bh), int(x):int(x + bw)] = 1
        norm_boxes.append((x / w_img, y / h_img, bw / w_img, bh / h_img))
    whitespace = float(np.clip(1.0 - float(mask.sum()) / area, 0.0, 1.0))
    return {
        "element_count": int(len(elements)),
        "whitespace_ratio": whitespace,
        "norm_covered_ratio": 1.0 - whitespace,
        "norm_boxes": norm_boxes,
    }


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
        idx = {s: hceye_load_index(feats[s], extractor=ex) for s in scales}
        ia, ir = _gap([idx[s] for s in scales])
        out[name] = {
            "raw_features": raw,
            "normalized_hceye_inputs": norm_gaps,
            "endpoint_load_index": {"by_scale": {str(s): idx[s] for s in scales},
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
        e = d["endpoint_load_index"]
        bs = e["by_scale"]
        print(f"  endpoint cognitive_load_index "
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
    NATIVE size against the features after a single canonicalisation to the
    candidate long side (no artificial upscaling). Reports the mean and worst
    relative feature change across the sample. Smaller = the candidate leaves
    real screenshots closer to their native measurement.
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


def heldout_ueyes(sample: List[Tuple[str, str, str]],
                  scales=(1, 2, 3)) -> List[dict]:
    """Full scale evaluation for held-out real UEyes screenshots."""
    ex = _extractor()
    results = []
    for img_id, cat, path in sample:
        base = cv2.imread(path)
        if base is None:
            continue
        per_scale = {}
        for s in scales:
            if s == 1:
                v = base
            else:
                v = cv2.resize(base, (base.shape[1] * s, base.shape[0] * s),
                               interpolation=cv2.INTER_LINEAR)
            vis = canonical_features(v)
            det = detect_stats(v)
            idx = hceye_load_index(vis, whitespace_ratio=det["whitespace_ratio"],
                                   extractor=ex)
            per_scale[s] = {
                "raw_features": vis,
                "normalized_hceye_inputs": normalized_hceye_inputs(vis, ex),
                "whitespace_ratio": det["whitespace_ratio"],
                "element_count": det["element_count"],
                "norm_covered_ratio": det["norm_covered_ratio"],
                "endpoint_load_index": idx,
            }
        raw_gaps = {k: _gap([per_scale[s]["raw_features"][k] for s in scales])[1]
                    for k in FEATURE_KEYS}
        norm_gaps = {k: _gap([per_scale[s]["normalized_hceye_inputs"][k]
                              for s in scales])[1] for k in HCEYE_MAPPED_STAGE1}
        ws_vals = [per_scale[s]["whitespace_ratio"] for s in scales]
        ec_vals = [per_scale[s]["element_count"] for s in scales]
        idx_vals = [per_scale[s]["endpoint_load_index"] for s in scales]
        results.append({
            "image_id": img_id,
            "category": cat,
            "per_scale": {str(s): per_scale[s] for s in scales},
            "raw_rel_gaps": raw_gaps,
            "normalized_rel_gaps": norm_gaps,
            "whitespace_ratio_abs_gap": max(ws_vals) - min(ws_vals),
            "element_count_range": [int(min(ec_vals)), int(max(ec_vals))],
            "endpoint_index_abs_gap": max(idx_vals) - min(idx_vals),
        })
    return results


# ---------------------------------------------------------------------------
# UEyes sampling
# ---------------------------------------------------------------------------
def _load_types(types_csv: str) -> Dict[str, str]:
    cat_by_id: Dict[str, str] = {}
    with open(types_csv, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = (row.get("Image Name") or "").strip()
            cat = (row.get("Category") or "").strip()
            if name and cat:
                cat_by_id[name] = cat
    return cat_by_id


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
    """The deterministic images used to SELECT 1280: first + middle per category."""
    by_cat: Dict[str, List[str]] = {}
    for name, cat in cat_by_id.items():
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
    """Seeded random sample per category, EXCLUDING the selection images."""
    rng = np.random.default_rng(seed)
    by_cat: Dict[str, List[str]] = {}
    for name, cat in cat_by_id.items():
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

    print("== Fixture scale report (synthetic, deterministic) ==")
    fx_rep = fixture_scale_report()
    _print_fixture_report(fx_rep)
    with open(os.path.join(args.out_dir, "fixture_scale_report.json"), "w") as f:
        json.dump(fx_rep, f, indent=2)

    sel: List[Tuple[str, str, str]] = []
    heldout: List[Tuple[str, str, str]] = []
    if have_ueyes:
        cats = _load_types(args.types_csv)
        sel = selection_images(args.images_dir, cats)
        sel_ids = {i for i, _, _ in sel}
        heldout = heldout_sample(args.images_dir, cats, sel_ids, args.seed,
                                 args.per_cat)
        print("\n== UEyes selection images (used to choose 1280) ==")
        for i, c, _ in sel:
            print(f"    {i}  {c}")
        print(f"== UEyes held-out sample (seed={args.seed}, excludes selection) ==")
        for i, c, _ in heldout:
            print(f"    {i}  {c}")
    else:
        print("\n[warn] UEyes dataset not found at --images-dir/--types-csv; "
              "skipping UEyes candidate + held-out sections.")

    print("\n== Candidate-resolution comparison (synthetic re-render, "
          "pixel-scale drivers) ==")
    cand_rows = candidate_comparison()
    for r in cand_rows:
        print(f"    long_side={r['candidate']:5d}  "
              f"mean_rel={r['mean_rel_gap']:.4%}  worst_rel={r['worst_rel_gap']:.4%}")

    pert_rows = []
    if sel:
        print("\n== Candidate tie-breaker: native->candidate feature "
              "perturbation on UEyes selection images ==")
        pert_rows = ueyes_perturbation(sel)
        for r in pert_rows:
            print(f"    long_side={r['candidate']:5d}  "
                  f"mean_pert={r['mean_rel_perturbation']:.4%}  "
                  f"worst_pert={r['worst_rel_perturbation']:.4%}")

    with open(os.path.join(args.out_dir, "candidate_comparison.json"), "w") as f:
        json.dump({"seed": args.seed,
                   "selection_ids": [i for i, _, _ in sel],
                   "scale_objective_source": "synthetic re-rendered fixtures",
                   "scale_gap": cand_rows,
                   "ueyes_native_to_candidate_perturbation": pert_rows}, f, indent=2)
    with open(os.path.join(args.out_dir, "candidate_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
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
        print("\n== Held-out UEyes scale evaluation ==")
        ho = heldout_ueyes(heldout)
        for r in ho:
            print(f"    {r['image_id']} ({r['category']}): "
                  f"endpoint_index_abs_gap={r['endpoint_index_abs_gap']:.5f} "
                  f"whitespace_abs_gap={r['whitespace_ratio_abs_gap']:.5f} "
                  f"elem_range={r['element_count_range']}")
        with open(os.path.join(args.out_dir, "heldout_ueyes_results.json"),
                  "w") as f:
            json.dump({"seed": args.seed, "results": ho}, f, indent=2)
        with open(os.path.join(args.out_dir, "heldout_ueyes_manifest.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "category", "seed"])
            for i, c, _ in heldout:
                w.writerow([i, c, args.seed])

    print(f"\nArtifacts written to {os.path.relpath(args.out_dir, _REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    # Feature functions are quiet; only guard against accidental library prints.
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass
    raise SystemExit(main())
