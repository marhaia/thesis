"""
HCEye Condition Comparison — Behavioural Validation of Cognitive Load Signatures.

Purpose
-------
Use the HCEye dataset (Das et al., ETRA 2024) to validate that the behavioural
signatures predicted by our hceye_features.py module are consistent with the
ground-truth eye-tracking data collected under three experimentally induced
cognitive-load conditions:

  CognitiveLoad = Absent | Low | High

Design
------
The HCEye experiment used a dual-task paradigm (Brouwer et al., 2012):
  - Absent : No secondary task — baseline attention
  - Low    : Easy secondary task (e.g., single-digit arithmetic)
  - High   : Hard secondary task (e.g., multi-digit mental arithmetic)

Hypotheses (from HCEye paper, Das et al. 2024, Table 2):
  H1: Total fixation count DECREASES under higher cognitive load.
      (High < Low < Absent)
  H2: Mean fixation duration INCREASES under higher cognitive load.
      (High > Low > Absent)
  H3: Both effects are statistically significant (paired t-test per image).

The script:
  1. Loads fixation_AOI_metrics_final.csv
  2. Aggregates per-participant per-image fixation metrics by CL condition
  3. Runs paired t-tests (Absent vs High) and computes Cohen's d
  4. Compares the magnitude of the effect to the HCEye model's predictions
     in hceye_features.py (fixation_reduction_mean = 0.876, i.e. ~12.4% fewer
     fixations under load)
  5. Optionally: if --images-dir is provided, runs the full pipeline on those
     images and reports correlation of pipeline score with behavioural CL effect.

References
----------
  Das, S., Husain, S., Bhatt, U., & Oulasvirta, A. (2024).
    Highlighting under cognitive load: An eye-tracking study.
    ACM Symposium on Eye Tracking Research & Applications (ETRA).
  Brouwer, A.-M., et al. (2012). Estimating workload using EEG spectral power
    and ERPs in the n-back task. Journal of Neural Engineering, 9(4), 045008.
  Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd
    ed.). Lawrence Erlbaum Associates.

Usage
-----
  python3 scripts/hceye_condition_comparison.py
  python3 scripts/hceye_condition_comparison.py --images-dir /path/to/screenshots
  python3 scripts/hceye_condition_comparison.py --output-csv results/hceye_cl_comparison.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Paths
ROOT = Path(__file__).resolve().parent.parent
HCEYE_CSV = ROOT / "hceye" / "gaze" / "fixation_AOI_metrics_final.csv"

# HCEye model predictions (from hceye/hceye_features.py)
# Das et al. (2024): under cognitive load, participants show:
HCEYE_PREDICTED_FIXATION_REDUCTION = 0.876   # ratio (12.4% fewer fixations)
HCEYE_PREDICTED_DURATION_INCREASE  = 1.082   # ratio (8.2% longer fixations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d for two independent samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 +
                          (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """
    Load HCEye CSV and return per-image per-CL-condition mean metrics.

    Returns a DataFrame with columns:
      Image_Name, CognitiveLoad,
      mean_fixation_count, mean_fixation_duration, n_observations
    """
    df = pd.read_csv(csv_path)

    # Keep only numeric fixation metrics; drop sentinel -99.99 values
    numeric_cols = ["TotalNumFixations", "MeanFixationDuration",
                    "TotalFixationDurationOnScreen"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    agg = (
        df.groupby(["Image_Name", "CognitiveLoad"])
        .agg(
            mean_fixation_count=("TotalNumFixations", "mean"),
            mean_fixation_duration=("MeanFixationDuration", "mean"),
            mean_total_duration=("TotalFixationDurationOnScreen", "mean"),
            n_obs=("TotalNumFixations", "count"),
        )
        .reset_index()
    )
    return agg


def run_hypothesis_tests(agg: pd.DataFrame) -> dict:
    """
    Test H1 (fixation count decreases) and H2 (duration increases) for
    Absent vs High cognitive-load conditions.

    Returns a dict with t-statistics, p-values, and Cohen's d.
    """
    results = {}

    for metric, direction, h_label in [
        ("mean_fixation_count",    "decrease", "H1"),
        ("mean_fixation_duration", "increase", "H2"),
    ]:
        # Per-image means under each condition
        absent = agg[agg["CognitiveLoad"] == "Absent"].set_index("Image_Name")[metric]
        low    = agg[agg["CognitiveLoad"] == "Low"].set_index("Image_Name")[metric]
        high   = agg[agg["CognitiveLoad"] == "High"].set_index("Image_Name")[metric]

        # Align on common images
        common = absent.index.intersection(high.index)
        a_vals = absent.loc[common].values
        h_vals = high.loc[common].values

        # Paired t-test: Absent vs High
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_val = stats.ttest_rel(a_vals, h_vals)

        d = cohens_d(a_vals, h_vals)
        ratio = np.mean(h_vals) / np.mean(a_vals) if np.mean(a_vals) != 0 else float("nan")

        results[h_label] = {
            "metric":        metric,
            "direction":     direction,
            "n_images":      len(common),
            "mean_absent":   float(np.mean(a_vals)),
            "mean_high":     float(np.mean(h_vals)),
            "ratio_H_A":     float(ratio),
            "t_statistic":   float(t_stat),
            "p_value":       float(p_val),
            "cohens_d":      float(d),
            "effect_size":   effect_label(d),
            "significant":   p_val < 0.05,
        }

        # Also test Low vs High
        if len(low) > 0:
            common_lh = low.index.intersection(high.index)
            l_vals = low.loc[common_lh].values
            h_vals_lh = high.loc[common_lh].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_lh, p_lh = stats.ttest_rel(l_vals, h_vals_lh)
            results[h_label]["p_value_low_vs_high"] = float(p_lh)
            results[h_label]["t_low_vs_high"] = float(t_lh)

    return results


def compare_to_model_predictions(results: dict) -> None:
    """Print comparison between observed data and hceye_features.py predictions."""
    print("\n" + "=" * 60)
    print("  Comparison: HCEye CSV data vs. hceye_features.py predictions")
    print("=" * 60)

    h1 = results.get("H1", {})
    h2 = results.get("H2", {})

    obs_fix_ratio = h1.get("ratio_H_A", float("nan"))
    obs_dur_ratio = h2.get("ratio_H_A", float("nan"))

    print(f"\nFixation count  (High / Absent):")
    print(f"  Observed in CSV : {obs_fix_ratio:.4f}  ({(1-obs_fix_ratio)*100:.1f}% fewer fixations under high load)")
    print(f"  hceye_features  : {HCEYE_PREDICTED_FIXATION_REDUCTION:.4f}  ({(1-HCEYE_PREDICTED_FIXATION_REDUCTION)*100:.1f}% fewer)")
    dev_fix = abs(obs_fix_ratio - HCEYE_PREDICTED_FIXATION_REDUCTION)
    print(f"  Deviation       : {dev_fix:.4f}  ({'✓ within 5%' if dev_fix < 0.05 else '⚠ >5% deviation'})")

    print(f"\nFixation duration (High / Absent):")
    print(f"  Observed in CSV : {obs_dur_ratio:.4f}  ({(obs_dur_ratio-1)*100:.1f}% longer fixations under high load)")
    print(f"  hceye_features  : {HCEYE_PREDICTED_DURATION_INCREASE:.4f}  ({(HCEYE_PREDICTED_DURATION_INCREASE-1)*100:.1f}% longer)")
    dev_dur = abs(obs_dur_ratio - HCEYE_PREDICTED_DURATION_INCREASE)
    print(f"  Deviation       : {dev_dur:.4f}  ({'✓ within 5%' if dev_dur < 0.05 else '⚠ >5% deviation'})")


def pipeline_correlation(images_dir: Path, agg: pd.DataFrame) -> None:
    """
    If screenshot images are available locally, run the pipeline via the
    Flask API and correlate the score with the behavioural CL effect.

    The 'CL effect' per image is defined as:
      cl_effect = (fixation_count_High - fixation_count_Absent) / fixation_count_Absent
    A more negative effect means the image induced more cognitive disruption.
    """
    import glob, json, requests

    image_files = (
        glob.glob(str(images_dir / "*.png")) +
        glob.glob(str(images_dir / "*.jpg")) +
        glob.glob(str(images_dir / "*.jpeg"))
    )
    if not image_files:
        print(f"\n⚠  No images found in {images_dir} — skipping pipeline correlation.")
        return

    API_URL = "http://localhost:5001/api/cognitive-load"
    print(f"\nRunning pipeline on {len(image_files)} images via {API_URL} ...")

    pipeline_scores = {}
    for img_path in image_files:
        stem = Path(img_path).stem
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(API_URL, files={"image": f}, data={
                    "task_type": "search", "target_specificity": "medium",
                    "time_pressure": "medium", "search_mode": "exploratory",
                    "profile_preset": "neutral", "use_trained_model": "false",
                }, timeout=120)
            data = resp.json()
            if "error" not in data:
                pipeline_scores[stem] = data["base_prediction"]["cognitive_load_score"]
                print(f"  {stem}: score={pipeline_scores[stem]:.1f}")
        except Exception as e:
            print(f"  {stem}: ERROR — {e}")

    if len(pipeline_scores) < 3:
        print("  Too few pipeline scores for correlation — skipping.")
        return

    # Compute per-image CL effect from HCEye data
    absent = agg[agg["CognitiveLoad"] == "Absent"].set_index("Image_Name")["mean_fixation_count"]
    high   = agg[agg["CognitiveLoad"] == "High"].set_index("Image_Name")["mean_fixation_count"]

    rows = []
    for stem, score in pipeline_scores.items():
        # Try to match image name to HCEye Image_Name (partial match)
        matches = [img for img in absent.index if stem in img or img in stem]
        if matches:
            img_name = matches[0]
            if img_name in high.index:
                effect = (high[img_name] - absent[img_name]) / absent[img_name]
                rows.append({"stem": stem, "pipeline_score": score, "cl_effect": effect})

    if len(rows) < 3:
        print("  Insufficient HCEye matches for correlation — provide images from the HCEye set.")
        return

    scores  = np.array([r["pipeline_score"] for r in rows])
    effects = np.array([r["cl_effect"] for r in rows])

    # Spearman rank correlation (pipeline score vs behavioural CL effect)
    # Expected: images with high pipeline scores should show more negative CL effect
    # (i.e., greater fixation reduction under load → harder to scan under pressure)
    rho, p = stats.spearmanr(scores, effects)
    print(f"\n  Spearman ρ (pipeline score vs CL effect): {rho:.4f}  (p={p:.4f})")
    if p < 0.05:
        direction = "negative" if rho < 0 else "positive"
        print(f"  Significant {direction} correlation — consistent with H1.")
    else:
        print("  Not significant — may need more images or different task config.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HCEye CL condition comparison")
    parser.add_argument("--csv",        type=Path, default=HCEYE_CSV,
                        help="Path to fixation_AOI_metrics_final.csv")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Optional: directory of screenshot images to run through pipeline")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Optional: save aggregated results to CSV")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    print("=" * 60)
    print("  HCEye Cognitive Load — Behavioural Condition Comparison")
    print("  Das et al. (ETRA 2024)")
    print("=" * 60)

    print(f"\nLoading: {args.csv}")
    agg = load_and_aggregate(args.csv)
    print(f"  {agg['Image_Name'].nunique()} unique images")
    print(f"  CL conditions: {sorted(agg['CognitiveLoad'].unique())}")
    total_obs = agg["n_obs"].sum()
    print(f"  Total observations: {total_obs}")

    print("\nRunning hypothesis tests (paired t-test, Absent vs High) ...")
    results = run_hypothesis_tests(agg)

    print("\n" + "-" * 60)
    for h_label, r in results.items():
        sig = "✓ significant" if r["significant"] else "✗ not significant"
        print(f"\n{h_label}: {r['metric']} should {r['direction']} under high CL")
        print(f"  n_images       : {r['n_images']}")
        print(f"  Mean (Absent)  : {r['mean_absent']:.3f}")
        print(f"  Mean (High)    : {r['mean_high']:.3f}")
        print(f"  Ratio H/A      : {r['ratio_H_A']:.4f}")
        print(f"  t-statistic    : {r['t_statistic']:.4f}")
        print(f"  p-value        : {r['p_value']:.4g}  →  {sig}")
        print(f"  Cohen's d      : {r['cohens_d']:.4f}  ({r['effect_size']} effect)")
        if "p_value_low_vs_high" in r:
            print(f"  p (Low vs High): {r['p_value_low_vs_high']:.4g}")

    compare_to_model_predictions(results)

    if args.output_csv:
        agg.to_csv(args.output_csv, index=False)
        print(f"\nSaved aggregated data: {args.output_csv}")

    if args.images_dir:
        pipeline_correlation(args.images_dir, agg)

    print("\n" + "=" * 60)
    print("  Summary for thesis")
    print("=" * 60)
    h1 = results.get("H1", {})
    h2 = results.get("H2", {})
    print(f"""
  H1 (fewer fixations under high CL):
    Absent={h1.get('mean_absent', 0):.1f}  High={h1.get('mean_high', 0):.1f}
    p={h1.get('p_value', 1):.4g}  d={h1.get('cohens_d', 0):.3f} ({h1.get('effect_size','?')})

  H2 (longer fixations under high CL):
    Absent={h2.get('mean_absent', 0):.1f} ms  High={h2.get('mean_high', 0):.1f} ms
    p={h2.get('p_value', 1):.4g}  d={h2.get('cohens_d', 0):.3f} ({h2.get('effect_size','?')})

  These effects are the empirical ground truth for the HCEye coefficients
  used in hceye_features.py (fixation_reduction_mean, duration_increase_mean).
""")


if __name__ == "__main__":
    main()
