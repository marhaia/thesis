#!/usr/bin/env python3
"""Manual UMSI++ end-to-end diagnostic with scratch-only output.

This intentionally remains outside the authoritative pytest contract because
it loads the full checkpoint. It never writes into tracked ``saliency/output``:
callers may supply ``--output-dir`` or receive a newly created temporary
directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saliency.saliency_features import extract_saliency_features
from saliency.umsi_model import UMSIPlus


DEFAULT_IMAGE = ROOT / "stage1" / "data" / "screenshots" / "bmw_route.png"
DEFAULT_WEIGHTS = (
    ROOT
    / "saliency"
    / "weights"
    / "model_weights"
    / "saliency_models"
    / "UMSI++"
    / "umsi++.hdf5"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Caller-owned scratch directory; defaults to a new temporary directory.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(tempfile.mkdtemp(prefix="umsi-full-pipeline-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMSI++ End-to-End Saliency Prediction Diagnostic")
    print("=" * 60)
    print(f"Scratch output directory: {output_dir}")

    print("\n[1] Loading UMSI++ model...")
    model = UMSIPlus(str(args.weights))

    print(f"\n[2] Predicting saliency for: {args.image}")
    heatmap, classif = model.predict_saliency(str(args.image), return_classif=True)
    print(f"    Heatmap shape: {heatmap.shape}")
    print(f"    Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print("    Auxiliary head (semantic index order unverified):")
    for index, probability in enumerate(classif):
        marker = " <--" if probability == classif.max() else ""
        print(f"      index_{index}: {probability:.4f}{marker}")

    print("\n[3] Extracting saliency-derived features...")
    features = extract_saliency_features(heatmap)
    for key, value in features.items():
        print(f"    {key:30s}: {value:.4f}")

    heatmap_path = output_dir / "bmw_route_saliency.png"
    cv2.imwrite(str(heatmap_path), (heatmap * 255).astype(np.uint8))
    print(f"\n[4] Saved heatmap: {heatmap_path}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Cannot read diagnostic image: {args.image}")
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    overlay_path = output_dir / "bmw_route_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"    Saved overlay:  {overlay_path}")

    print("\n" + "=" * 60)
    print("SUCCESS - UMSI++ diagnostic completed without repository writes")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
