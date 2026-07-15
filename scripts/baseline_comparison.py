#!/usr/bin/env python3
"""
Baseline Comparison — Two-Stage Pipeline
==========================================
Computes cognitive load estimates for a set of test screenshots under three
pipeline configurations and reports the difference in score and feature richness.

Configurations
--------------
A  Baseline-A   v∈ℝ⁸ only (visual complexity, no HCEye, no Jokinen)
B  Baseline-B   v∈ℝ⁸ + h∈ℝ⁶ (visual + HCEye rule-based, no task context)
C  Full         v∈ℝ⁸ + s∈ℝ⁵ + h∈ℝ⁶ + t∈ℝ⁶ + p∈ℝ⁵ (complete pipeline)

The comparison demonstrates that the HCEye features (h) shift the raw
visual-complexity estimate towards empirically grounded load estimates, and
that the task descriptor (t) further personalises the prediction.

Usage
-----
    python scripts/baseline_comparison.py [--images path/to/img1 path/to/img2 ...]

If no --images flag is given, the script uses all PNGs/JPGs found in
stage1/data/screenshots/ and ueyes/saliency_models/UMSI++/images/.

Output
------
    Prints a Markdown table to stdout.
    Writes a CSV to scripts/baseline_comparison_results.csv.

Scientific context (for thesis §5 — Technical Baselines)
---------------------------------------------------------
A purely visual-complexity-based estimate (Baseline-A) relies only on
pixel-level statistics (entropy, edge density, etc.) as proxies for load.
Adding HCEye coefficients (Baseline-B) grounds the estimate in observed
gaze behaviour (Das et al., 2024, ETRA), reducing dependence on indirect
proxies. The full pipeline (C) additionally models task-specific demand via
the task descriptor (Wickens, 2008; Paas et al., 2003) and optional
individual differences (Big Five presets).

References:
    Das et al. (2024). Shifting Focus with HCEye. ETRA'24.
    Wickens, C. D. (2008). Multiple resources and mental workload.
        Human Factors, 50(3), 449-455.
    Paas et al. (2003). Cognitive load theory. Educational Psychologist, 38(1).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1"))

from visual_complexity import compute_complexity_vector
from hceye.hceye_features import HCEyeFeatureExtractor
from stage2.task_descriptor import TaskDescriptor
from stage2.user_profile import get_profile


# ── Default test images ───────────────────────────────────────────────────────
DEFAULT_SEARCH_DIRS = [
    ROOT / "stage1" / "data" / "screenshots",
    ROOT / "ueyes" / "saliency_models" / "UMSI++" / "images",
]
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def find_default_images() -> List[Path]:
    imgs = []
    for d in DEFAULT_SEARCH_DIRS:
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    imgs.append(p)
    return imgs


# ── Baseline-A: visual complexity only ───────────────────────────────────────
def score_baseline_a(v: dict) -> dict:
    """
    Cognitive-load proxy from visual features only.

    Uses a linear combination of the most load-relevant v∈ℝ⁸ dimensions:
      - feature_congestion: dense visual info → higher load
      - shannon_entropy:    unpredictability → higher load
      - edge_density:       structural complexity
      - visual_hierarchy:   clear hierarchy → lower load

    Weights are ordinal heuristics, not empirically calibrated.
    This is the weakest baseline — pure pixel statistics.
    """
    score = (
        25.0 * float(v["shannon_entropy"])
        + 20.0 * float(v["edge_density"])
        + 30.0 * float(v["feature_congestion"])
        + 15.0 * float(v["subband_entropy"])
        - 10.0 * float(v["visual_hierarchy"])
    )
    score = float(np.clip(score, 0.0, 100.0))
    return {
        "cognitive_load_score": score,
        "search_efficiency": float(np.clip(1.0 - score / 100.0, 0.0, 1.0)),
        "attention_demand": float(np.clip(score / 100.0, 0.0, 1.0)),
        "feature_dims": "v(8)",
        "n_features": 8,
    }


# ── Baseline-B: visual + HCEye rule-based ────────────────────────────────────
def score_baseline_b(v: dict, extractor: HCEyeFeatureExtractor) -> dict:
    """
    Visual complexity + HCEye empirical adjustment.

    Extracts h∈ℝ⁶ from the HCEye feature extractor (rule-based, no saliency)
    and uses h[5] (cognitive_load_index) as the primary score, calibrated
    against Das et al. (2024) gaze observations.
    """
    h = extractor.extract_features(v, saliency_features=None)
    cog_idx = float(h[5])  # cognitive_load_index (0-1)
    score = float(np.clip(cog_idx * 100.0, 0.0, 100.0))
    return {
        "cognitive_load_score": score,
        "search_efficiency": float(np.clip(1.0 - float(h[3]), 0.0, 1.0)),
        "attention_demand": float(np.clip(cog_idx + 0.15, 0.0, 1.0)),
        "feature_dims": "v(8) + h(6)",
        "n_features": 14,
    }


# ── Full pipeline (C): visual + saliency + HCEye + task + profile ─────────────
def score_full(
    v: dict,
    extractor: HCEyeFeatureExtractor,
    task: TaskDescriptor,
    profile_name: str = "neutral",
    s: Optional[np.ndarray] = None,
) -> dict:
    """
    Complete two-stage pipeline prediction.

    Adds task descriptor modifier (t∈ℝ⁶) and optional Big Five preset (p∈ℝ⁵).
    Saliency features (s∈ℝ⁵) are included if available; zeroed otherwise.
    """
    h = extractor.extract_features(v, saliency_features=s)
    cog_idx = float(h[5])
    base_score = float(np.clip(cog_idx * 100.0, 0.0, 100.0))
    base_search = float(np.clip(1.0 - float(h[3]), 0.0, 1.0))
    base_attn   = float(np.clip(cog_idx + 0.15, 0.0, 1.0))

    t_mod = task.score_modifier()
    profile = get_profile(profile_name)
    p_mod = float(profile["modifier"])

    adjusted_score = float(np.clip(base_score + t_mod + p_mod, 0.0, 100.0))
    s_dim = 5 if s is not None else 0
    t_dim = len(task.to_vector())
    p_dim = len(profile["vector"])

    return {
        "cognitive_load_score": adjusted_score,
        "search_efficiency": float(np.clip(base_search - 0.0025 * t_mod - 0.003 * p_mod, 0.0, 1.0)),
        "attention_demand":  float(np.clip(base_attn  + 0.004  * t_mod + 0.004 * p_mod, 0.0, 1.0)),
        "task_modifier": t_mod,
        "profile_modifier": p_mod,
        "feature_dims": f"v(8) + s({s_dim}) + h(6) + t({t_dim}) + p({p_dim})",
        "n_features": 8 + s_dim + 6 + t_dim + p_dim,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline baseline comparison")
    parser.add_argument("--images", nargs="+", type=str, default=None,
                        help="Paths to screenshots. Default: auto-discover.")
    parser.add_argument("--task", default="search",
                        choices=["navigation", "search", "monitoring",
                                 "data_entry", "decision"],
                        help="Task type for Full pipeline (default: search)")
    parser.add_argument("--time-pressure", default="medium",
                        choices=["low", "medium", "high"],
                        help="Time pressure for Full pipeline (default: medium)")
    parser.add_argument("--search-mode", default="known_item",
                        choices=["known_item", "comparative", "exploratory"],
                        help="Search mode for Full pipeline (default: known_item)")
    parser.add_argument("--profile", default="neutral",
                        choices=["neutral", "focused", "exploratory",
                                 "social", "stress_sensitive"],
                        help="Big Five preset for Full pipeline (default: neutral)")
    args = parser.parse_args()

    # Collect images
    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = find_default_images()

    if not images:
        print("No images found. Provide --images or add screenshots to "
              "stage1/data/screenshots/", file=sys.stderr)
        sys.exit(1)

    # Shared objects
    lookup_path = ROOT / "hceye" / "sensitivity_lookup.json"
    extractor = HCEyeFeatureExtractor(str(lookup_path))
    task = TaskDescriptor(
        task_type=args.task,
        time_pressure=args.time_pressure,
        search_mode=args.search_mode,
    )

    # ── Column headers ──
    rows = []
    col_a = "A: Visual only\n(v∈ℝ⁸)"
    col_b = "B: +HCEye\n(v+h∈ℝ¹⁴)"
    col_c = f"C: Full pipeline\n(v+h+t+p∈ℝ²⁴+)"
    col_d = "B−A (HCEye Δ)"
    col_e = "C−B (Task+Profile Δ)"

    print("\n" + "=" * 90)
    print(f"  PIPELINE BASELINE COMPARISON  |  task={args.task}  "
          f"time_pressure={args.time_pressure}  profile={args.profile}")
    print("=" * 90)
    print(f"{'Image':<28} {'A (v⁸)':>9} {'B (v+h¹⁴)':>11} "
          f"{'C (full)':>10} {'Δ B−A':>8} {'Δ C−B':>8}  Coherent?")
    print("-" * 90)

    for img_path in images:
        if not img_path.exists():
            print(f"  [SKIP] {img_path.name} — file not found", file=sys.stderr)
            continue

        try:
            v = compute_complexity_vector(str(img_path))
        except Exception as e:
            print(f"  [ERR]  {img_path.name}: {e}", file=sys.stderr)
            continue

        res_a = score_baseline_a(v)
        res_b = score_baseline_b(v, extractor)
        res_c = score_full(v, extractor, task, args.profile)

        sa = res_a["cognitive_load_score"]
        sb = res_b["cognitive_load_score"]
        sc = res_c["cognitive_load_score"]
        delta_ba = sb - sa
        delta_cb = sc - sb

        # Quick coherence check (no saliency available for standalone script)
        from stage2.coherence_check import run_coherence_check
        coh = run_coherence_check(None, None, None, sc)
        coherent = "✓" if coh["is_coherent"] else "⚠ " + ",".join(coh["flags"])

        name = img_path.name[:27]
        print(f"  {name:<26} {sa:>8.1f}  {sb:>10.1f}  {sc:>9.1f}  "
              f"{delta_ba:>+7.1f}  {delta_cb:>+7.1f}  {coherent}")

        rows.append({
            "image": img_path.name,
            "baseline_a_score": round(sa, 2),
            "baseline_b_score": round(sb, 2),
            "full_pipeline_score": round(sc, 2),
            "delta_b_minus_a": round(delta_ba, 2),
            "delta_c_minus_b": round(delta_cb, 2),
            "a_features": res_a["feature_dims"],
            "b_features": res_b["feature_dims"],
            "c_features": res_c["feature_dims"],
            "task_modifier": round(res_c.get("task_modifier", 0), 2),
            "profile_modifier": round(res_c.get("profile_modifier", 0), 2),
            "task": args.task,
            "time_pressure": args.time_pressure,
            "profile": args.profile,
        })

    if not rows:
        print("No images were processed successfully.", file=sys.stderr)
        sys.exit(1)

    # Summary stats
    scores_a = [r["baseline_a_score"] for r in rows]
    scores_b = [r["baseline_b_score"] for r in rows]
    scores_c = [r["full_pipeline_score"] for r in rows]
    deltas_ba = [r["delta_b_minus_a"] for r in rows]
    deltas_cb = [r["delta_c_minus_b"] for r in rows]

    print("-" * 90)
    print(f"  {'MEAN':<26} {np.mean(scores_a):>8.1f}  {np.mean(scores_b):>10.1f}  "
          f"{np.mean(scores_c):>9.1f}  {np.mean(deltas_ba):>+7.1f}  {np.mean(deltas_cb):>+7.1f}")
    print(f"  {'STD':<26} {np.std(scores_a):>8.1f}  {np.std(scores_b):>10.1f}  "
          f"{np.std(scores_c):>9.1f}  {np.std(deltas_ba):>+7.1f}  {np.std(deltas_cb):>+7.1f}")
    print("=" * 90)

    print(f"\n  Configurations:")
    print(f"    A  v∈ℝ⁸  — visual complexity only (heuristic linear combination)")
    print(f"    B  v∈ℝ¹⁴ — + HCEye rule-based h∈ℝ⁶ (Das et al., 2024, ETRA)")
    print(f"    C  full  — + task descriptor t∈ℝ⁶ + Big Five p∈ℝ⁵")
    print(f"  Task: {args.task} | Time pressure: {args.time_pressure} | "
          f"Search mode: {args.search_mode} | Profile: {args.profile}")
    print(f"\n  Interpretation of Δ:")
    print(f"    Δ B−A > 0  → HCEye raises the raw visual estimate (more complex than pixels suggest)")
    print(f"    Δ B−A < 0  → HCEye lowers the estimate (visual stats over-estimated load)")
    print(f"    Δ C−B      → Task/profile contribution (always deterministic for same settings)")
    print()

    # ── Write CSV ──
    out_csv = Path(__file__).parent / "baseline_comparison_results.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV written → {out_csv}\n")


if __name__ == "__main__":
    main()
