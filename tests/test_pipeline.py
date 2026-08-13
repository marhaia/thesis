"""HISTORICAL / INCOMPLETE DEMO — not the current Stage-1 contract.

This manually invoked development script predates the audited Stage-1 x19
boundary. It extracts v8 and h6 but does not run UMSI++; it substitutes five
zeros where real s5 values would be required. The resulting 19 numeric slots
are therefore only an incomplete historical Stage-2 scaffold input and must
never be identified as a Stage-1 x19 vector, a production pipeline test, or
validation evidence.

The script is retained to document earlier development. Current Stage-1
behavior and evidence are defined by ``stage1/app.py``, ``README.md``, and the
automated tests selected in ``pytest.ini``.
"""
import sys
import os
import numpy as np

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stage1.visual_complexity import compute_complexity_vector
from hceye.hceye_features import HCEyeFeatureExtractor
from stage2.regression_model import Stage2Model


HISTORICAL_DEMO_BANNER = (
    "HISTORICAL / INCOMPLETE DEMO — NOT THE CURRENT STAGE-1 CONTRACT"
)


def _print_historical_warning():
    print(f"\n{'!'*72}")
    print(f"  {HISTORICAL_DEMO_BANNER}")
    print("  UMSI++/s5 is not computed; five zeros are only placeholders.")
    print("  The combined values are not Stage-1 x19 and are not validation evidence.")
    print(f"{'!'*72}")


def run_historical_incomplete_demo(image_path: str):
    """Run the retained incomplete scaffold on one image."""
    _print_historical_warning()
    print(f"\n{'='*60}")
    print(f"  HISTORICAL SCAFFOLD DEMO: {os.path.basename(image_path)}")
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
    
    # Step 2: Project-specific HCEye-derived screenshot proxies.
    print("\n[2/3] HCEye-Derived Screenshot Proxies (h ∈ ℝ⁶)...")
    lookup_path = "hceye/sensitivity_lookup.json"
    extractor = HCEyeFeatureExtractor(lookup_path)
    # extract_features now takes the named visual-feature dict directly.
    h = extractor.extract_features(vis)
    feature_names = extractor.get_feature_names()
    print(f"  h = {h}")
    for name, val in zip(feature_names, h):
        print(f"    {name}: {val:.4f}")
    
    # Step 3: Assemble the historical scaffold input. This demo does not run
    # UMSI++: five zeros are placeholders, not saliency features.
    # A vector with placeholder s5 values is not the audited Stage-1 x19.
    s_placeholder = np.zeros(5, dtype=np.float32)
    historical_scaffold_input = np.concatenate([v, s_placeholder, h])
    print(
        f"\n  Incomplete scaffold input: {len(historical_scaffold_input)} numeric slots "
        "(v8 + zero placeholders5 + h6; NOT Stage-1 x19)"
    )
    
    # Step 4: Stage 2 prediction
    print("\n[3/3] Historical Stage-2 Regression Scaffold...")
    model_path = "stage2/models/stage2_model.pkl"
    if os.path.exists(model_path):
        model = Stage2Model(model_path=model_path)
        pred = model.predict(historical_scaffold_input)
        print("  Retired Stage-2 scaffold outputs (not Stage-1 measurements/evidence):")
        print(f"    Retired score field:      {pred['cognitive_load_score']:.1f} / 100")
        print(f"    Retired search field:     {pred['search_efficiency']:.3f}")
        print(f"    Retired attention field:  {pred['attention_demand']:.3f}")
    else:
        print(f"  Historical Stage-2 model not found at {model_path}; no prediction produced.")
        print(f"  HCEye-derived proxy h[5]: {h[5]:.3f} (not a measured load score)")
    
    return pred if os.path.exists(model_path) else None


if __name__ == '__main__':
    _print_historical_warning()

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
        print("No demo images found. Building synthetic historical scaffold inputs...")
        extractor = HCEyeFeatureExtractor("hceye/sensitivity_lookup.json")
        model_path = "stage2/models/stage2_model.pkl"
        model = Stage2Model(model_path=model_path) if os.path.exists(model_path) else None
        
        _v_keys = [
            "shannon_entropy", "edge_density", "feature_congestion",
            "subband_entropy", "layout_symmetry", "chromatic_coherence",
            "visual_hierarchy", "interactive_element_density",
        ]
        # Synthetic values use REALISTIC Stage-1 ranges (matching feature_norms:
        # feature_congestion ~1.3-9.7, edge_density ~0.01-0.12,
        # interactive_element_density ~0-1.9). Out-of-range values would clamp to
        # the percentile floor/ceiling and mask the demo's intended contrast.
        for name, v in [
            ("Complex Dashboard", np.array([7.2, 0.09, 7.5, 3.2, 0.30, 0.7, 0.30, 1.6])),
            ("Simple Settings",   np.array([4.5, 0.02, 2.0, 0.8, 0.85, 0.2, 0.80, 0.2])),
            ("Medium Form",       np.array([5.8, 0.05, 4.0, 1.5, 0.60, 0.4, 0.55, 0.7])),
        ]:
            vis = dict(zip(_v_keys, v.tolist()))
            h = extractor.extract_features(vis)
            historical_scaffold_input = np.concatenate([v, np.zeros(5), h])
            if model is None:
                print(
                    f"  {name:25s}  scaffold slots={len(historical_scaffold_input)}; "
                    "no historical model, no prediction"
                )
                continue
            pred = model.predict(historical_scaffold_input)
            print(
                f"  {name:25s}  retired_score={pred['cognitive_load_score']:5.1f}  "
                f"retired_search={pred['search_efficiency']:.3f}  "
                f"retired_attention={pred['attention_demand']:.3f}"
            )
    else:
        for img in images[:3]:
            run_historical_incomplete_demo(img)
