#!/usr/bin/env python3
"""
Feature Ablation + SHAP — Stage 2 Regression Model
===================================================
Answers a critical question raised in supervision: does the UMSI++ saliency
block (s in R^5) add anything to the cognitive-load prediction beyond the
classic visual-complexity block (v in R^8) and the empirical HCEye block
(h in R^6)?

Two complementary analyses are run on the Stage 2 training data:

1. Block ablation (cross-validated R^2)
   Each feature block (v = visual, s = saliency, h = HCEye) is switched on or
   off and the model is re-fitted. The change in 5-fold CV R^2 shows the
   marginal contribution of each block. If s barely moves R^2, UMSI++ does not
   help the regression (a publishable negative result).

2. Per-feature importance (permutation + SHAP)
   On the full 19-feature model we compute two model-agnostic importance
   measures for the headline output (cognitive_load_score):
     - permutation importance (sklearn, always available)
     - SHAP mean(|value|) (TreeExplainer on a Random Forest)
   Agreement between the two rankings makes the finding more trustworthy.

IMPORTANT scientific caveat (must be stated in the thesis)
----------------------------------------------------------
In the current Stage 2 training data the visual (v) and saliency (s) feature
vectors are SIMULATED from HCEye gaze proxies (see
stage2/regression_model.build_training_data), while the HCEye block (h) and the
regression targets (y) are both derived from the same empirical sensitivity
lookup. The h block is therefore expected to dominate almost by construction.
This ablation is thus a diagnostic of the CURRENT model, not yet evidence about
saliency in general; a fair test needs real image-derived v and s features.

Usage
-----
    python scripts/feature_ablation_shap.py

Output
------
    - prints an ablation table and importance rankings to stdout
    - writes scripts/feature_ablation_results.csv (ablation R^2 per block combo)
    - writes scripts/feature_importance_results.csv (per-feature importances)
    - writes scripts/feature_ablation_shap.png (importance bar chart) if
      matplotlib is available

References
----------
    Lundberg, S.M., & Lee, S.-I. (2017). A Unified Approach to Interpreting
        Model Predictions. NeurIPS 2017.  [SHAP]
    Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
        [permutation importance]
    Das, S. et al. (2024). Shifting Focus with HCEye. ETRA 2024.
"""

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# Make the project root importable so we can reuse the Stage 2 data builder.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stage2"))

from stage2.regression_model import Stage2Model, build_training_data  # noqa: E402

# Default empirical inputs (same as regression_model.py CLI defaults).
_DEFAULT_CSV = _PROJECT_ROOT / "hceye" / "gaze" / "fixation_AOI_metrics_final.csv"
_DEFAULT_LOOKUP = _PROJECT_ROOT / "hceye" / "sensitivity_lookup.json"

# Feature-block layout inside the 19-dim vector (must match Stage2Model).
BLOCKS = {
    "v": (0, 8),    # visual complexity
    "s": (8, 13),   # UMSI++ saliency
    "h": (13, 19),  # HCEye sensitivity
}
BLOCK_LABEL = {
    "v": "visual (v)",
    "s": "saliency (s)",
    "h": "hceye (h)",
}

# Headline output used for the per-feature importance analysis.
PRIMARY_OUTPUT_IDX = 0  # cognitive_load_score


def _block_columns(block_keys):
    """Return the column indices covered by the given block keys."""
    cols = []
    for key in block_keys:
        start, end = BLOCKS[key]
        cols.extend(range(start, end))
    return cols


def _cv_r2(X_sub, y, n_splits=5):
    """Mean 5-fold CV R^2 of a standardised Ridge model on the given columns.

    A scikit-learn Pipeline scales inside each fold, so no information leaks
    from the validation fold into the training fold.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("reg", MultiOutputRegressor(Ridge(alpha=1.0))),
    ])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # cross_val_score with the default 'r2' scorer averages R^2 across outputs.
    scores = cross_val_score(pipe, X_sub, y, cv=kf, scoring="r2")
    return float(np.mean(scores)), float(np.std(scores))


def run_ablation(X, y):
    """Cross-validated R^2 for every non-empty combination of feature blocks."""
    keys = list(BLOCKS.keys())
    rows = []
    for r in range(1, len(keys) + 1):
        for combo in combinations(keys, r):
            cols = _block_columns(combo)
            mean_r2, std_r2 = _cv_r2(X[:, cols], y)
            rows.append({
                "blocks": "+".join(combo),
                "n_features": len(cols),
                "cv_r2_mean": round(mean_r2, 4),
                "cv_r2_std": round(std_r2, 4),
            })
    # Sort best-first for readability.
    rows.sort(key=lambda d: d["cv_r2_mean"], reverse=True)
    return rows


def marginal_contributions(ablation_rows):
    """Marginal R^2 gain of adding each block to the other two.

    Compares the full v+s+h model against the model with that one block removed.
    A near-zero (or negative) gain means the block does not help.
    """
    by_blocks = {row["blocks"]: row["cv_r2_mean"] for row in ablation_rows}
    full_key = "v+s+h"
    full_r2 = by_blocks.get(full_key)
    gains = {}
    if full_r2 is None:
        return gains
    for drop in ("v", "s", "h"):
        remaining = [k for k in ("v", "s", "h") if k != drop]
        without_key = "+".join(remaining)
        without_r2 = by_blocks.get(without_key)
        if without_r2 is not None:
            gains[drop] = round(full_r2 - without_r2, 4)
    return gains


def per_feature_importance(X, y, feature_names):
    """Permutation + SHAP importance for the primary output on a RF model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y_primary = y[:, PRIMARY_OUTPUT_IDX]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    rf.fit(X_scaled, y_primary)

    # Permutation importance: drop in R^2 when a feature is shuffled.
    perm = permutation_importance(
        rf, X_scaled, y_primary, n_repeats=30, random_state=42, scoring="r2"
    )
    perm_importance = perm.importances_mean

    # SHAP importance: mean absolute SHAP value per feature (TreeExplainer).
    shap_importance = None
    try:
        import shap

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        # shap_values shape: (n_samples, n_features) for single-output RF.
        shap_importance = np.abs(np.asarray(shap_values)).mean(axis=0)
    except Exception as exc:  # pragma: no cover - SHAP is optional
        print(f"[warn] SHAP unavailable, using permutation only: {exc!r}")

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "feature": name,
            "block": _feature_block(i),
            "perm_importance": round(float(perm_importance[i]), 5),
            "shap_importance": (
                round(float(shap_importance[i]), 5)
                if shap_importance is not None else None
            ),
        })
    rows.sort(key=lambda d: d["perm_importance"], reverse=True)
    return rows


def _feature_block(col_idx):
    """Return the block key ('v'/'s'/'h') that owns a given column index."""
    for key, (start, end) in BLOCKS.items():
        if start <= col_idx < end:
            return key
    return "?"


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _plot_importance(importance_rows, out_path):
    """Bar chart of per-feature importance (permutation vs SHAP)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] matplotlib unavailable, skipping figure: {exc!r}")
        return

    names = [r["feature"] for r in importance_rows]
    perm = [r["perm_importance"] for r in importance_rows]
    has_shap = importance_rows and importance_rows[0]["shap_importance"] is not None
    block_color = {"v": "#3b82f6", "s": "#f59e0b", "h": "#22c55e"}
    colors = [block_color.get(r["block"], "#888") for r in importance_rows]

    fig, axes = plt.subplots(
        1, 2 if has_shap else 1, figsize=(12 if has_shap else 7, 6), squeeze=False
    )
    y_pos = np.arange(len(names))

    ax = axes[0][0]
    ax.barh(y_pos, perm, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (drop in R^2)")
    ax.set_title("Permutation importance\n(cognitive_load_score)")

    if has_shap:
        shap_vals = [r["shap_importance"] for r in importance_rows]
        ax2 = axes[0][1]
        ax2.barh(y_pos, shap_vals, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("mean(|SHAP value|)")
        ax2.set_title("SHAP importance\n(cognitive_load_score)")

    # Shared block-colour legend.
    from matplotlib.patches import Patch
    legend = [
        Patch(color=block_color["v"], label="visual (v)"),
        Patch(color=block_color["s"], label="saliency (s)"),
        Patch(color=block_color["h"], label="hceye (h)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    print("=" * 68)
    print("  Stage 2 Feature Ablation + SHAP")
    print("=" * 68)

    X, y = build_training_data(str(_DEFAULT_CSV), str(_DEFAULT_LOOKUP))
    feature_names = (
        Stage2Model.VISUAL_FEATURES
        + Stage2Model.SALIENCY_FEATURES
        + Stage2Model.COGNITIVE_FEATURES
    )

    # ---- 1. Block ablation --------------------------------------------------
    print("\n--- Block ablation (5-fold CV R^2, averaged over 3 outputs) ---")
    print(f"{'blocks':<10}{'#feat':>6}{'CV R^2':>12}{'+/- std':>10}")
    ablation_rows = run_ablation(X, y)
    for row in ablation_rows:
        print(
            f"{row['blocks']:<10}{row['n_features']:>6}"
            f"{row['cv_r2_mean']:>12.4f}{row['cv_r2_std']:>10.4f}"
        )

    gains = marginal_contributions(ablation_rows)
    print("\n--- Marginal R^2 gain of each block (full minus block-removed) ---")
    for key in ("v", "s", "h"):
        if key in gains:
            verdict = "helps" if gains[key] > 0.01 else "negligible"
            print(f"  {BLOCK_LABEL[key]:<14} +{gains[key]:+.4f}  ({verdict})")

    # ---- 2. Per-feature importance -----------------------------------------
    print("\n--- Per-feature importance (RF, cognitive_load_score) ---")
    importance_rows = per_feature_importance(X, y, feature_names)
    print(f"{'feature':<30}{'block':>6}{'perm':>10}{'shap':>10}")
    for row in importance_rows:
        shap_str = (
            f"{row['shap_importance']:>10.4f}"
            if row["shap_importance"] is not None else f"{'n/a':>10}"
        )
        print(
            f"{row['feature']:<30}{row['block']:>6}"
            f"{row['perm_importance']:>10.4f}{shap_str}"
        )

    # ---- 3. Persist results -------------------------------------------------
    _write_csv(
        _THIS_DIR / "feature_ablation_results.csv",
        ablation_rows,
        ["blocks", "n_features", "cv_r2_mean", "cv_r2_std"],
    )
    _write_csv(
        _THIS_DIR / "feature_importance_results.csv",
        importance_rows,
        ["feature", "block", "perm_importance", "shap_importance"],
    )
    _plot_importance(importance_rows, _THIS_DIR / "feature_ablation_shap.png")

    print("\nDone. Remember the caveat: v and s are currently simulated from")
    print("gaze proxies, so this ablation diagnoses the CURRENT model only.")


if __name__ == "__main__":
    main()
