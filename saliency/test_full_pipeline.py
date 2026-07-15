#!/usr/bin/env python3
"""Full end-to-end test: UMSI++ saliency prediction + feature extraction."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import sys

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

# Initialize model
print("\n[1] Loading UMSI++ model...")
model = UMSIPlus(weights_path)

# Predict saliency
print(f"\n[2] Predicting saliency for: {test_img}")
heatmap, classif = model.predict_saliency(test_img, return_classif=True)

print(f"    Heatmap shape: {heatmap.shape}")
print(f"    Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
print(f"    Classification:")
for cls, prob in zip(UMSIPlus.DESIGN_CLASSES, classif):
    marker = " <--" if prob == classif.max() else ""
    print(f"      {cls:20s}: {prob:.4f}{marker}")

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
img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
heatmap_resized = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_orig, 0.5, heatmap_colored, 0.5, 0)
overlay_path = os.path.join(output_dir, "bmw_route_overlay.png")
cv2.imwrite(overlay_path, overlay)
print(f"    Saved overlay:  {overlay_path}")

print("\n" + "=" * 60)
print("SUCCESS - UMSI++ saliency pipeline fully operational!")
print("=" * 60)
