"""
UEyes Saliency Validation — UMSI++ vs Human Fixation Maps.

Purpose
-------
Validate the Stage 1b saliency component of the pipeline by comparing
UMSI++ model predictions against human fixation density maps from the
UEyes dataset (Jiang et al., CHI 2023).

Metrics
-------
  NSS  — Normalised Scanpath Saliency (Peters et al., 2005):
          NSS = mean of normalised saliency map at fixation locations.
          NSS = 1.0 means fixations land exactly at the model's mean prediction.
          NSS > 1.0 means model over-predicts; NSS < 1.0 under-predicts.
          State-of-the-art models typically achieve NSS ≈ 1.5–3.0 on images.

  CC   — Pearson Linear Correlation Coefficient between predicted saliency
          map and fixation density map (Judd et al., 2012).
          CC = 1.0 means perfect linear correspondence; CC ≈ 0.3–0.7 typical.

  SIM  — Similarity (histogram intersection, Swain & Ballard 1991):
          SIM ∈ [0,1]; measures overlap after normalising both maps to sum=1.

  KL   — KL-Divergence (Kullback & Leibler, 1951):
          D_KL(GT || Pred).  Lower = better.  Sensitive to zero-probability bins.

References
----------
  Jiang, Y., Leiva, L.A., Tavakoli, H.R., Houssel, P.R.B., Kylmälä, J., &
    Oulasvirta, A. (2023). UEyes: Understanding visual saliency across user
    interface types. CHI '23. https://doi.org/10.1145/3544548.3581096

  Peters, R.J., Iyer, A., Itti, L., & Koch, C. (2005). Components of
    bottom-up gaze allocation in natural images. Vision Research, 45(18),
    2397–2416.  [NSS definition]

  Judd, T., Durand, F., & Torralba, A. (2012). A benchmark of computational
    models of saliency to predict human fixations. MIT CSAIL TR 2012-001.
    [CC metric]

  Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., & Durand, F. (2018).
    What do different evaluation metrics tell us about saliency models?
    IEEE TPAMI, 41(3), 740–757.  [Comprehensive metric overview]

Usage
-----
  python3 scripts/ueyes_saliency_validation.py
  python3 scripts/ueyes_saliency_validation.py --images-dir /path/to/images
                                                --gt-dir /path/to/gt_saliency
  python3 scripts/ueyes_saliency_validation.py --output-csv results/ueyes_nss.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Default paths (5 sample images bundled with UEyes repo clone)
DEFAULT_IMAGES_DIR = ROOT / "ueyes" / "saliency_models" / "UMSI++" / "images"
DEFAULT_GT_DIR     = ROOT / "ueyes" / "saliency_models" / "UMSI++" / "saliency_gt"

# UMSI++ saliency model
UMSI_WEIGHTS = (ROOT / "saliency" / "weights" / "model_weights" / "saliency_models"
                / "UMSI++" / "umsi++.hdf5")


# ---------------------------------------------------------------------------
# Metric functions  (Bylinskii et al., 2018)
# ---------------------------------------------------------------------------

def _normalize_map(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize map to [0,1]."""
    mn, mx = m.min(), m.max()
    if mx - mn < eps:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def _to_prob(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize map to probability distribution (sum=1)."""
    m = m.astype(np.float64)
    s = m.sum()
    if s < eps:
        return np.full_like(m, 1.0 / m.size)
    return m / s


def nss(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """
    Normalised Scanpath Saliency (Peters et al., 2005).

    Parameters
    ----------
    saliency_map : float array, any range — model predicted saliency.
    fixation_map : binary or density array — 1 at fixation locations, 0 elsewhere.
    """
    # Standardize saliency map (zero mean, unit std)
    sm = saliency_map.astype(np.float64)
    std = sm.std()
    if std < 1e-12:
        return 0.0
    sm_norm = (sm - sm.mean()) / std

    fix = (fixation_map > 0).astype(np.float64)
    n_fix = fix.sum()
    if n_fix == 0:
        return 0.0
    return float((sm_norm * fix).sum() / n_fix)


def cc(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """Pearson correlation coefficient between predicted and GT saliency."""
    s = saliency_map.astype(np.float64).ravel()
    f = fixation_map.astype(np.float64).ravel()
    if s.std() < 1e-12 or f.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(s, f)[0, 1])


def sim(saliency_map: np.ndarray, fixation_map: np.ndarray) -> float:
    """
    Similarity (histogram intersection, Swain & Ballard 1991).
    Both maps normalized to sum=1 before comparison.
    """
    s = _to_prob(saliency_map.astype(np.float64).ravel())
    f = _to_prob(fixation_map.astype(np.float64).ravel())
    return float(np.minimum(s, f).sum())


def kl_divergence(saliency_map: np.ndarray, fixation_map: np.ndarray,
                  eps: float = 1e-8) -> float:
    """
    KL-Divergence D_KL(GT || Pred).  Lower = better.
    Both maps converted to probability distributions.
    """
    q = _to_prob(saliency_map.astype(np.float64).ravel()) + eps
    p = _to_prob(fixation_map.astype(np.float64).ravel()) + eps
    q /= q.sum()
    p /= p.sum()
    return float((p * np.log(p / q)).sum())


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB numpy array."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        import cv2
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gt_map(path: Path) -> np.ndarray:
    """Load ground-truth saliency/fixation map as float32 grayscale."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32) / 255.0


def resize_map(m: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize map to target_shape (H, W) using bilinear interpolation."""
    try:
        from PIL import Image
        img = Image.fromarray((m * 255).astype(np.uint8))
        img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        import cv2
        return cv2.resize(m, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ---------------------------------------------------------------------------
# UMSI++ prediction
# ---------------------------------------------------------------------------

def predict_saliency(image_rgb: np.ndarray, image_path: Path) -> np.ndarray:
    """
    Run UMSI++ on an image file and return a saliency map (H x W float32).

    Uses UMSIPlus from saliency/umsi_model.py (same path as the Flask API).
    predict_saliency() takes a file path and returns a (H, W) float32 map
    already resized to the input image dimensions.
    """
    sys.path.insert(0, str(ROOT))
    from saliency.umsi_model import UMSIPlus

    model = UMSIPlus(weights_path=str(UMSI_WEIGHTS), verbose=False)
    sal_map = model.predict_saliency(str(image_path))
    return sal_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_pair(img_path: Path, gt_path: Path, verbose: bool = True) -> dict:
    """Evaluate one image: run UMSI++ and compute all metrics vs GT map."""
    image = load_image_rgb(img_path)
    gt    = load_gt_map(gt_path)

    if verbose:
        print(f"  {img_path.stem}: image={image.shape} gt={gt.shape}")

    # Run UMSI++ (takes file path)
    sal = predict_saliency(image, img_path)

    # Resize GT to saliency map resolution
    if gt.shape != sal.shape:
        gt = resize_map(gt, sal.shape)

    return {
        "image":       img_path.stem,
        "nss":         nss(sal, gt),
        "cc":          cc(sal, gt),
        "sim":         sim(sal, gt),
        "kl_div":      kl_divergence(sal, gt),
        "img_w":       image.shape[1],
        "img_h":       image.shape[0],
    }


def print_results(rows: list[dict]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("  UEyes Saliency Validation — UMSI++ vs Human Fixation Maps")
    print("=" * 70)
    print(f"\n{'Image':<20} {'NSS':>8} {'CC':>8} {'SIM':>8} {'KL':>8}")
    print("-" * 58)
    for r in rows:
        print(f"{r['image']:<20} {r['nss']:8.4f} {r['cc']:8.4f} "
              f"{r['sim']:8.4f} {r['kl_div']:8.4f}")

    nss_vals = np.array([r["nss"] for r in rows])
    cc_vals  = np.array([r["cc"]  for r in rows])
    sim_vals = np.array([r["sim"] for r in rows])
    kl_vals  = np.array([r["kl_div"] for r in rows])

    print("-" * 58)
    print(f"{'MEAN':<20} {nss_vals.mean():8.4f} {cc_vals.mean():8.4f} "
          f"{sim_vals.mean():8.4f} {kl_vals.mean():8.4f}")
    print(f"{'STD':<20} {nss_vals.std():8.4f} {cc_vals.std():8.4f} "
          f"{sim_vals.std():8.4f} {kl_vals.std():8.4f}")

    print("\nInterpretation:")
    print(f"  NSS = {nss_vals.mean():.3f}  "
          f"(NOTE: NSS requires binary fixation maps; GT maps here are blurred\n"
          f"         density maps → NSS score is not interpretable in this context)")
    print(f"  CC  = {cc_vals.mean():.3f}  (typical SOTA: 0.40–0.70 — our value: "
          f"{'✓ ABOVE SOTA' if cc_vals.mean() > 0.70 else '✗ below SOTA'})")
    print(f"  SIM = {sim_vals.mean():.3f}  (typical SOTA: 0.35–0.55 — our value: "
          f"{'✓ ABOVE SOTA' if sim_vals.mean() > 0.55 else '✗ below SOTA'})")
    print(f"  KL  = {kl_vals.mean():.3f}  (lower is better; < 0.5 considered good)")

    print("\nBenchmark context (Bylinskii et al., 2018; Judd et al., 2012):")
    print("  CC and SIM are the primary valid metrics for continuous GT saliency maps.")
    print("  NSS requires binary fixation point maps (not available in this sample).")
    print("  Full validation with binary fixation maps requires the complete UEyes")
    print("  dataset (Zenodo:8010312) — use paths_1s/ or paths_3s/ scanpath files.")
    _cc = cc_vals.mean()
    _verdict = ("exceeds" if _cc > 0.70 else "is within" if _cc >= 0.40
                else "is below") + " the typical published CC range (0.40–0.70)"
    print(f"  CC = {_cc:.3f} indicates the spatial correspondence between UMSI++")
    print(f"  predictions and human fixation density {_verdict}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="UEyes UMSI++ saliency validation")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--gt-dir",     type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.images_dir.exists():
        sys.exit(f"Images directory not found: {args.images_dir}")
    if not args.gt_dir.exists():
        sys.exit(f"GT directory not found: {args.gt_dir}")

    # Match image files to GT files by stem
    img_exts = {".png", ".jpg", ".jpeg"}
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in img_exts)

    pairs = []
    for img_path in images:
        # Try matching GT by same stem, any extension
        for ext in img_exts:
            gt_path = args.gt_dir / (img_path.stem + ext)
            if gt_path.exists():
                pairs.append((img_path, gt_path))
                break

    if not pairs:
        sys.exit(f"No matching (image, GT) pairs found in:\n  {args.images_dir}\n  {args.gt_dir}")

    print(f"Found {len(pairs)} image/GT pairs. Running UMSI++ ...")
    print("(First run loads model weights — may take 20–40 s)\n")

    rows = []
    for img_path, gt_path in pairs:
        try:
            result = evaluate_pair(img_path, gt_path)
            rows.append(result)
        except Exception as e:
            print(f"  ERROR on {img_path.stem}: {e}")

    if not rows:
        sys.exit("No results computed.")

    print_results(rows)

    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
