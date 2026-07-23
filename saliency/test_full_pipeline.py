#!/usr/bin/env python3
"""Full end-to-end diagnostic: UMSI++ saliency prediction + feature extraction.

Manual diagnostic script, NOT a pytest test. All heavy imports and all model
execution live inside ``main()`` behind an ``if __name__ == "__main__"`` guard,
so pytest collection never triggers inference or writes output files. Warning
suppression is deliberately NOT applied, so real runtime warnings stay visible.
A real model/runtime failure propagates and yields a non-zero exit code.
Run manually with: python saliency/test_full_pipeline.py
"""
import os
import sys


def main() -> int:
    # Resolve the repository root relative to this file so the script is portable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    import cv2
    from saliency.umsi_model import UMSIPlus
    from saliency.saliency_features import extract_saliency_features

    # Test image
    test_img = "stage1/data/screenshots/bmw_route.png"
    weights_path = "saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5"

    print("=" * 60)
    print("UMSI++ End-to-End Saliency Prediction Test")
    print("=" * 60)

    # Initialize model. Any load/runtime error is intentionally NOT caught here,
    # so a real failure propagates and the process exits non-zero.
    print("\n[1] Loading UMSI++ model...")
    model = UMSIPlus(weights_path)

    # Predict saliency
    print(f"\n[2] Predicting saliency for: {test_img}")
    heatmap, aux_classif = model.predict_saliency(test_img, return_classif=True)

    print(f"    Heatmap shape: {heatmap.shape}")
    print(f"    Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    # aux_classif is the RAW auxiliary output head: UNVALIDATED and NON-SEMANTIC
    # (zero training-loss weight upstream). Print the raw vector only; do NOT map
    # it to UI-type class labels.
    print(f"    aux_classif (RAW, UNVALIDATED, non-semantic): "
          f"{np.asarray(aux_classif).ravel().tolist()}")

    # Extract saliency features
    print(f"\n[3] Extracting saliency-derived features...")
    features = extract_saliency_features(heatmap)
    for k, v in features.items():
        print(f"    {k:30s}: {v:.4f}")

    # Save heatmap visualization
    output_dir = "saliency/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save raw heatmap
    heatmap_path = os.path.join(output_dir, "bmw_route_saliency.png")
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    cv2.imwrite(heatmap_path, heatmap_uint8)
    print(f"\n[4] Saved heatmap: {heatmap_path}")

    # Save colored heatmap overlay
    img_orig = cv2.imread(test_img)
    heatmap_resized = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_orig, 0.5, heatmap_colored, 0.5, 0)
    overlay_path = os.path.join(output_dir, "bmw_route_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"    Saved overlay:  {overlay_path}")

    print("\n" + "=" * 60)
    # No numerical-parity or "operational" completeness claim: this exercises
    # the port end-to-end but does NOT establish TF1/Keras2 golden parity.
    print("UMSI++ saliency pipeline ran end-to-end (source-matched port; "
          "TF1/Keras2 numerical parity NOT established).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
