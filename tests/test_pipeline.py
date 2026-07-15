"""
End-to-end test: Full pipeline from screenshot → cognitive load prediction.

Tests the complete flow:
  1. Visual complexity extraction (v ∈ ℝ⁸)
  2. HCEye cognitive load features (h ∈ ℝ⁶)
  3. Stage 2 regression → cognitive load score
"""
import sys
import os
import numpy as np

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stage1.visual_complexity import compute_complexity_vector
from hceye.hceye_features import HCEyeFeatureExtractor
from stage2.regression_model import Stage2Model

def test_full_pipeline(image_path: str):
    """Run the full pipeline on a single image."""
    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE TEST: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Step 1: Visual complexity
    print("\n[1/3] Visual Complexity (v ∈ ℝ⁸)...")
    vis = compute_complexity_vector(image_path)
    v = np.array([
        vis["shannon_entropy"],
        vis["edge_density"],
        vis["feature_congestion"],
        vis["subband_entropy"],
        vis["layout_symmetry"],
        vis["chromatic_coherence"],
        vis["visual_hierarchy"],
        vis["interactive_element_density"],
    ], dtype=np.float32)
    print(f"  v = {v}")
    
    # Step 2: HCEye cognitive load features
    print("\n[2/3] HCEye Cognitive Load Features (h ∈ ℝ⁶)...")
    lookup_path = "hceye/sensitivity_lookup.json"
    extractor = HCEyeFeatureExtractor(lookup_path)
    # extract_features now takes the named visual-feature dict directly.
    h = extractor.extract_features(vis)
    feature_names = extractor.get_feature_names()
    print(f"  h = {h}")
    for name, val in zip(feature_names, h):
        print(f"    {name}: {val:.4f}")
    
    # Step 3: Combine features (without saliency for this test)
    # v⁸ + padding⁵ (no saliency) + h⁶ = ℝ¹⁹
    s_placeholder = np.zeros(5, dtype=np.float32)
    x = np.concatenate([v, s_placeholder, h])
    print(f"\n  Full vector: ℝ{len(x)} (v⁸ + s⁵ + h⁶)")
    
    # Step 4: Stage 2 prediction
    print("\n[3/3] Stage 2 Regression...")
    model_path = "stage2/models/stage2_model.pkl"
    if os.path.exists(model_path):
        model = Stage2Model(model_path=model_path)
        pred = model.predict(x)
        print(f"  Predictions:")
        print(f"    Cognitive Load Score:  {pred['cognitive_load_score']:.1f} / 100")
        print(f"    Search Efficiency:     {pred['search_efficiency']:.3f}")
        print(f"    Attention Demand:      {pred['attention_demand']:.3f}")
        
        # Interpret
        score = pred['cognitive_load_score']
        if score < 25:
            level = "LOW"
        elif score < 50:
            level = "MODERATE"
        elif score < 75:
            level = "HIGH"
        else:
            level = "VERY HIGH"
        print(f"\n  → Overall Cognitive Load: {level} ({score:.0f}/100)")
    else:
        print(f"  Model not found at {model_path}")
        print(f"  Raw CLI from HCEye: {h[5]:.3f}")
    
    return pred if os.path.exists(model_path) else None


if __name__ == '__main__':
    # Find test images
    test_dirs = [
        'stage1/data/screenshots',
        'stage1/data/uploads',
    ]
    
    images = []
    for d in test_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(d, f))
    
    if not images:
        print("No test images found. Generating synthetic test...")
        # Synthetic test
        extractor = HCEyeFeatureExtractor("hceye/sensitivity_lookup.json")
        model = Stage2Model(model_path="stage2/models/stage2_model.pkl")
        
        _v_keys = [
            "shannon_entropy", "edge_density", "feature_congestion",
            "subband_entropy", "layout_symmetry", "chromatic_coherence",
            "visual_hierarchy", "interactive_element_density",
        ]
        for name, v in [
            ("Complex Dashboard", np.array([7.2, 0.35, 0.75, 3.2, 0.3, 0.7, 0.4, 65.0])),
            ("Simple Settings",   np.array([4.5, 0.08, 0.15, 0.8, 0.85, 0.2, 0.8, 8.0])),
            ("Medium Form",       np.array([5.8, 0.18, 0.40, 1.5, 0.6, 0.4, 0.6, 25.0])),
        ]:
            vis = dict(zip(_v_keys, v.tolist()))
            h = extractor.extract_features(vis)
            x = np.concatenate([v, np.zeros(5), h])
            pred = model.predict(x)
            print(f"  {name:25s}  CLI={pred['cognitive_load_score']:5.1f}  "
                  f"SearchEff={pred['search_efficiency']:.3f}  "
                  f"AttnDem={pred['attention_demand']:.3f}")
    else:
        for img in images[:3]:
            test_full_pipeline(img)
