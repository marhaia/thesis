"""Build the GUI reference distribution (feature norms) for Stage 2 comparison.

This standalone batch tool computes, for each Stage 1 feature, the empirical
distribution (mean, standard deviation, and percentiles) over a reference set
of real GUI screenshots. The resulting ``feature_norms.json`` lets the pipeline
tell a user whether the GUI they are testing lies above or below the typical
GUI for each feature (a neutral z-score / percentile statement).

Reference corpus:
    UEyes dataset (Jiang et al., CHI 2023; Zenodo record 8010312, CC-BY-4.0),
    located at ``ueyes/dataset_full/UEyes_dataset/``. Only the interactive UI
    types are used (web + mobile + desktop = 1,485 images); the ``poster`` type
    is excluded because posters/ads are non-interactive graphics with different
    visual-attention characteristics, which would distort a GUI baseline.

Feature blocks:
    v in R^8  - visual complexity   (compute_complexity_vector)
    s in R^5  - saliency features    (UMSI++ -> extract_saliency_features)

This script does NOT modify the live pipeline (app.py / UI). It only reads the
reference images and writes:
    - a per-image CSV  (raw feature values, for transparency / re-aggregation)
    - feature_norms.json (mean / std / percentiles per feature)

Usage:
    # Stage 1 - quick validation on a small balanced sample:
    python3 stage1/build_feature_norms.py --limit 40

    # Stage 2 - full run over all 1,485 GUI images:
    python3 stage1/build_feature_norms.py

    # Visual features only (skip the UMSI++ saliency model):
    python3 stage1/build_feature_norms.py --no-saliency
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve project paths and make the Stage 1 + project-root modules importable,
# mirroring the import setup used by app.py.
_THIS_DIR = Path(__file__).resolve().parent          # .../stage1
_PROJECT_ROOT = _THIS_DIR.parent                     # repo root
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from visual_complexity import compute_complexity_vector, FEATURE_KEYS  # noqa: E402

# Reference dataset locations.
_DATASET_DIR = _PROJECT_ROOT / "ueyes" / "dataset_full" / "UEyes_dataset"
_IMAGES_DIR = _DATASET_DIR / "images"
_TYPES_CSV = _DATASET_DIR / "image_types.csv"

# UMSI++ weights (same path app.py uses for the live saliency model).
_UMSI_WEIGHTS = (_PROJECT_ROOT / "saliency" / "weights" / "model_weights"
                 / "saliency_models" / "UMSI++" / "umsi++.hdf5")

# Interactive UI types that make up the GUI baseline (poster is excluded).
_GUI_CATEGORIES = {"web", "mobile", "desktop"}

# Saliency feature keys (order matches extract_saliency_features output).
SALIENCY_KEYS = [
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
]

# Default output locations.
_RESULTS_DIR = _THIS_DIR / "data" / "results"
_PER_IMAGE_CSV = _RESULTS_DIR / "gui_reference_features.csv"
_NORMS_JSON = _RESULTS_DIR / "feature_norms.json"


def load_gui_image_list(limit=None, categories=None):
    """Return a list of (filename, category) for GUI images in the reference set.

    Reads the semicolon-delimited ``image_types.csv`` and keeps only the
    interactive UI types (web / mobile / desktop). When ``limit`` is given, a
    type-balanced subsample of roughly ``limit`` images is returned so that the
    quick Stage 1 validation still covers all three GUI types. When
    ``categories`` is given (a subset of web/mobile/desktop), only those UI
    types are returned, which enables a folder-wise run one type at a time.
    """
    if not _TYPES_CSV.exists():
        raise FileNotFoundError(f"image_types.csv not found: {_TYPES_CSV}")

    allowed = _GUI_CATEGORIES if not categories else set(categories)

    by_category = defaultdict(list)
    with open(_TYPES_CSV, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = (row.get("Image Name") or "").strip()
            category = (row.get("Category") or "").strip().lower()
            if not name or category not in allowed:
                continue
            if (_IMAGES_DIR / name).exists():
                by_category[category].append(name)

    # Stable order within each category for reproducibility.
    for cat in by_category:
        by_category[cat].sort()

    if limit is None:
        # Full set: concatenate all selected GUI categories.
        images = []
        for cat in sorted(by_category):
            images.extend((name, cat) for name in by_category[cat])
        return images

    # Balanced subsample: take an equal share from each category.
    per_cat = max(1, limit // max(len(by_category), 1))
    images = []
    for cat in sorted(by_category):
        for name in by_category[cat][:per_cat]:
            images.append((name, cat))
    return images


def load_existing_rows(csv_path, feature_order):
    """Return (rows, done_filenames) from a previous (possibly partial) run.

    Enables crash-safe resume: a run that was killed part-way leaves a valid
    per-image CSV, and re-invoking the script skips images already processed.
    Returns an empty result when no usable CSV exists.
    """
    rows = []
    done = set()
    if not os.path.exists(csv_path):
        return rows, done
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                name = (raw.get("filename") or "").strip()
                if not name:
                    continue
                row = {"filename": name, "category": (raw.get("category") or "").strip()}
                ok = True
                for k in feature_order:
                    val = raw.get(k, "")
                    if val == "" or val is None:
                        ok = False
                        break
                    row[k] = float(val)
                if not ok:
                    continue  # incomplete line (e.g. crash mid-write) -> recompute
                rows.append(row)
                done.add(name)
    except Exception as exc:
        print(f"WARN: could not read existing CSV ({exc}); starting fresh.")
        return [], set()
    return rows, done


def aggregate(values_by_feature):
    """Compute mean / std / percentiles per feature from collected raw values."""
    norms = {}
    for feature, values in values_by_feature.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]  # drop NaN/inf defensively
        if arr.size == 0:
            continue
        norms[feature] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }
    return norms


def main():
    parser = argparse.ArgumentParser(
        description="Build GUI feature-norm reference distribution (mean/std/percentiles)."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only a balanced subsample of ~N GUI images (Stage 1 validation).")
    parser.add_argument("--category", type=str, default=None,
                        choices=sorted(_GUI_CATEGORIES),
                        help="Process only one GUI type (web/mobile/desktop) for a folder-wise run.")
    parser.add_argument("--no-saliency", action="store_true",
                        help="Skip UMSI++ saliency features (compute only the 8 visual features).")
    parser.add_argument("--output", type=str, default=str(_NORMS_JSON),
                        help="Output path for feature_norms.json.")
    parser.add_argument("--csv", type=str, default=str(_PER_IMAGE_CSV),
                        help="Output path for the per-image raw feature CSV.")
    args = parser.parse_args()

    use_saliency = not args.no_saliency

    categories = [args.category] if args.category else None
    images = load_gui_image_list(limit=args.limit, categories=categories)
    if not images:
        print("No GUI reference images found. Check the dataset path:")
        print(f"  {_IMAGES_DIR}")
        return

    cat_counts = defaultdict(int)
    for _, cat in images:
        cat_counts[cat] += 1

    print("=" * 64)
    print("GUI Feature-Norm Builder")
    print("=" * 64)
    print(f"Reference set : {_IMAGES_DIR}")
    print(f"Images        : {len(images)}  "
          f"({', '.join(f'{c}={cat_counts[c]}' for c in sorted(cat_counts))})")
    print(f"Visual (v)    : 8 features")
    print(f"Saliency (s)  : {'5 features (UMSI++)' if use_saliency else 'SKIPPED'}")
    print(f"Output JSON   : {args.output}")
    print("=" * 64)

    # Lazily load the UMSI++ model only when saliency is requested.
    saliency_model = None
    extract_saliency_features = None
    if use_saliency:
        from saliency.umsi_model import UMSIPlus
        from saliency.saliency_features import extract_saliency_features as _esf
        extract_saliency_features = _esf
        print("Loading UMSI++ model...")
        saliency_model = UMSIPlus(str(_UMSI_WEIGHTS))
        print("UMSI++ ready.\n")

    values_by_feature = defaultdict(list)
    per_image_rows = []
    feature_order = list(FEATURE_KEYS) + (SALIENCY_KEYS if use_saliency else [])

    # Crash-safe resume: reload any rows from a previous (partial) CSV so a
    # killed run can be continued instead of recomputed from scratch.
    fieldnames = ["filename", "category"] + feature_order
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    existing_rows, done_names = load_existing_rows(args.csv, feature_order)
    if done_names:
        per_image_rows.extend(existing_rows)
        for row in existing_rows:
            for k in feature_order:
                values_by_feature[k].append(row[k])
        print(f"Resuming: {len(done_names)} images already in {args.csv}; "
              f"skipping those.\n")

    # Open the per-image CSV in append mode and flush after every row, so a
    # crash never loses already-computed features.
    csv_exists = os.path.exists(args.csv) and os.path.getsize(args.csv) > 0
    csv_file = open(args.csv, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()

    t_start = time.time()
    n_ok = len(per_image_rows)
    n_new = 0
    for idx, (name, category) in enumerate(images, start=1):
        if name in done_names:
            continue
        img_path = _IMAGES_DIR / name
        t0 = time.time()
        try:
            row = {"filename": name, "category": category}

            # Visual complexity v in R^8.
            v = compute_complexity_vector(str(img_path))
            for k in FEATURE_KEYS:
                row[k] = v[k]
                values_by_feature[k].append(v[k])

            # Saliency s in R^5 (optional).
            if use_saliency:
                heatmap = saliency_model.predict_saliency(str(img_path))
                s = extract_saliency_features(heatmap)
                for k in SALIENCY_KEYS:
                    row[k] = s[k]
                    values_by_feature[k].append(s[k])

            per_image_rows.append(row)
            csv_writer.writerow(row)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            n_ok += 1
            n_new += 1
            dt = time.time() - t0
            print(f"[{idx}/{len(images)}] {name} ({category})  {dt:.1f}s")
        except Exception as exc:  # keep going on a single bad image
            print(f"[{idx}/{len(images)}] ERROR {name}: {exc}")

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\nProcessed {n_new} new images in {elapsed/60:.1f} min "
          f"({elapsed/max(n_new,1):.1f}s/image). Total usable: {n_ok}.")

    if n_ok == 0:
        print("No images processed successfully; nothing written.")
        return

    print(f"Per-image features -> {args.csv}")

    # Aggregate and write the norms.
    norms = aggregate(values_by_feature)
    processed_counts = defaultdict(int)
    for row in per_image_rows:
        processed_counts[row["category"]] += 1
    payload = {
        "meta": {
            "source": "UEyes dataset (Jiang et al., CHI 2023; Zenodo 8010312)",
            "categories_included": sorted(_GUI_CATEGORIES),
            "categories_excluded": ["poster"],
            "num_images": n_ok,
            "category_counts": dict(processed_counts),
            "includes_saliency": use_saliency,
            "is_subsample": args.limit is not None,
        },
        "features": norms,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Feature norms     -> {args.output}")

    # Console summary.
    print("\n" + "=" * 78)
    print(f"{'Feature':<30}{'mean':>10}{'std':>10}{'p5':>10}{'p50':>10}{'p95':>10}")
    print("-" * 78)
    for feat in feature_order:
        if feat in norms:
            s = norms[feat]
            print(f"{feat:<30}{s['mean']:>10.4f}{s['std']:>10.4f}"
                  f"{s['p5']:>10.4f}{s['p50']:>10.4f}{s['p95']:>10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
