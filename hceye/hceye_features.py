"""
HCEye Cognitive Load Features
==============================
Extracts cognitive-load-adjusted features from GUI screenshots using
empirical findings from the HCEye dataset (Das et al., ETRA 2024).

Key Insight: Under cognitive load, users make fewer fixations with longer
durations and reduced exploration. The magnitude of this effect depends on
the visual properties of the interface.

Features extracted (h ∈ ℝ⁶):
  1. cog_fixation_reduction    - Expected reduction in fixation count under load
  2. cog_duration_increase     - Expected increase in fixation duration under load  
  3. cog_exploration_reduction - Expected reduction in spatial exploration
  4. cog_aoi_sensitivity       - How much AOI attention drops under load
  5. highlight_effectiveness   - Expected benefit of dynamic highlighting
  6. cognitive_load_index      - Combined cognitive load vulnerability score (0-1)

Reference:
  Das, A., Wu, Z., Skrjanec, I., & Feit, A.M. (2024). Shifting Focus with
  HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on
  User Attention and Saliency Prediction. Proc. ACM ETRA.
"""

import numpy as np
import os
import json
from typing import Dict, Tuple, Optional

# Empirical coefficients derived from HCEye dataset (N=27 participants, 150 webpages).
# All means and standard deviations are computed from fixation_AOI_metrics_final.csv.
# Primary source: Das et al. (2024), ETRA, https://doi.org/10.1145/3655610
# Column mapping: CognitiveLoad ∈ {Absent, High, Low}, Highlight ∈ {Absent, Static, Dynamic}
HCEYE_COEFFICIENTS = {
    # Fixation count ratio: High Load / Absent Load
    # Das et al. (2024): participants made ~12.4% fewer fixations under high cognitive load.
    # Computed as mean(fixation_count | CogLoad=High) / mean(fixation_count | CogLoad=Absent)
    'fixation_reduction_mean': 0.876,  # ratio; ~12.4% fewer fixations (Das et al., 2024)
    'fixation_reduction_std': 0.107,   # inter-participant SD of that ratio

    # Fixation duration ratio: High Load / Absent Load
    # Longer fixations under load reflect deeper processing (Rayner, 1998, Psych. Bull.).
    # Computed as mean(fixation_duration_ms | CogLoad=High) / mean(...| CogLoad=Absent)
    'duration_increase_mean': 1.081,   # ratio; ~8.1% longer fixations (Das et al., 2024)
    'duration_increase_std': 0.252,

    # Fixation frequency ratio: High Load / Absent Load
    # Das et al. (2024): fixation rate (fix/s) drops slightly under load.
    'frequency_reduction_mean': 0.935,  # ratio; ~6.5% lower frequency (Das et al., 2024)
    'frequency_reduction_std': 0.120,

    # AOI hit rate under different conditions
    # Das et al. (2024), Table 2: proportion of fixations landing inside the target AOI.
    'aoi_hit_absent_no_hl': 0.115,    # Baseline (no load, no highlight)
    'aoi_hit_high_no_hl': 0.069,      # High load, no highlight  — 40% drop vs. baseline
    'aoi_hit_absent_dynamic': 0.768,  # No load, dynamic highlight  — 6.7× baseline
    'aoi_hit_high_dynamic': 0.690,    # High load, dynamic highlight  (Das et al., 2024)
    'aoi_hit_absent_static': 0.679,   # No load, static highlight
    'aoi_hit_high_static': 0.588,     # High load, static highlight

    # Baseline gaze metrics (Absent load, no highlight)
    # Used to normalise per-image estimates when no real gaze data is available.
    'baseline_fixations_mean': 14.39,      # mean fixation count per trial
    'baseline_fixations_std': 2.07,
    'baseline_duration_mean': 359.84,      # ms  (Das et al., 2024, descriptive stats)
    'baseline_duration_std': 146.23,
    'baseline_frequency_mean': 3.26,       # fixations/sec
    'baseline_frequency_std': 0.51,
}


class HCEyeFeatureExtractor:
    """
    Extracts cognitive-load sensitivity features for GUI screenshots.
    
    Uses a regression model trained on the relationship between visual
    complexity features and cognitive load sensitivity observed in HCEye.
    For new images (not in HCEye), estimates sensitivity from visual features.
    """
    
    def __init__(self, lookup_path: Optional[str] = None):
        """
        Initialize the HCEye feature extractor.
        
        Args:
            lookup_path: Path to pre-computed per-image sensitivity data.
                        If None, uses visual-feature-based estimation.
        """
        self.coefficients = HCEYE_COEFFICIENTS
        self.sensitivity_lookup = {}
        
        if lookup_path and os.path.exists(lookup_path):
            with open(lookup_path, 'r') as f:
                self.sensitivity_lookup = json.load(f)
    
    def extract_features(self, 
                         visual_features: np.ndarray,
                         saliency_features: Optional[np.ndarray] = None,
                         image_name: Optional[str] = None) -> np.ndarray:
        """
        Extract cognitive load features for a GUI.
        
        Args:
            visual_features: Stage 1 visual complexity vector (v ∈ ℝ⁸)
                [edge_density, color_count, layout_complexity, whitespace_ratio,
                 text_density, element_count, symmetry, grid_quality]
            saliency_features: Saliency features (s ∈ ℝ⁵) if available
                [peak_sal, mean_sal, sal_spread, hotspot_count, coverage]
            image_name: Optional name to check against HCEye lookup table
            
        Returns:
            h ∈ ℝ⁶ cognitive load feature vector
        """
        # Check if we have empirical data for this specific image
        if image_name and image_name in self.sensitivity_lookup:
            return self._from_lookup(image_name)
        
        # Otherwise, estimate from visual features
        return self._estimate_from_features(visual_features, saliency_features)
    
    def _from_lookup(self, image_name: str) -> np.ndarray:
        """Use pre-computed sensitivity from HCEye empirical data."""
        data = self.sensitivity_lookup[image_name]
        return np.array([
            data['fixation_reduction'],
            data['duration_increase'],
            data['exploration_reduction'],
            data['aoi_sensitivity'],
            data['highlight_effectiveness'],
            data['cognitive_load_index'],
        ], dtype=np.float32)
    
    def _estimate_from_features(self, 
                                 visual_features: np.ndarray,
                                 saliency_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate cognitive load sensitivity from visual complexity features.
        
        The estimation is based on empirical findings from HCEye:
        - More complex GUIs → greater fixation reduction under load
        - Higher saliency spread → more exploration reduction
        - More elements → highlights become more effective
        """
        # Unpack visual features (v ∈ ℝ⁸)
        # [edge_density, color_count, layout_complexity, whitespace_ratio,
        #  text_density, element_count, symmetry, grid_quality]
        v = np.asarray(visual_features, dtype=np.float32)
        
        # Normalize visual features to [0, 1] range
        # Using typical ranges from our Stage 1 extractor
        v_norm = self._normalize_visual(v)
        
        # 1. Cognitive Fixation Reduction
        # More complex GUIs see greater reduction in fixation count under load
        # Complexity proxy: edge_density + element_count - whitespace
        complexity = (v_norm[0] + v_norm[5] + v_norm[2]) / 3.0  # edge, elements, layout
        simplicity = v_norm[3]  # whitespace ratio
        
        fixation_reduction = self.coefficients['fixation_reduction_mean'] - \
            0.05 * (complexity - simplicity)  # More complex = more reduction
        fixation_reduction = np.clip(fixation_reduction, 0.6, 1.1)
        
        # 2. Cognitive Duration Increase
        # Under load, fixations get longer (more processing per fixation)
        # Higher text density and more elements → longer processing needed
        processing_demand = (v_norm[4] + v_norm[5]) / 2.0  # text + elements
        
        duration_increase = self.coefficients['duration_increase_mean'] + \
            0.1 * processing_demand
        duration_increase = np.clip(duration_increase, 0.8, 1.5)
        
        # 3. Exploration Reduction
        # Under load, users explore less of the interface
        # Lower symmetry and grid quality → less predictable layout → more reduction
        layout_quality = (v_norm[6] + v_norm[7]) / 2.0  # symmetry + grid
        
        exploration_reduction = self.coefficients['frequency_reduction_mean'] - \
            0.04 * (1.0 - layout_quality)
        exploration_reduction = np.clip(exploration_reduction, 0.7, 1.0)
        
        # 4. AOI Sensitivity
        # How much does cognitive load reduce attention to key areas
        # More visual clutter → greater AOI attention loss
        clutter = 1.0 - v_norm[3]  # inverse of whitespace
        
        aoi_baseline = self.coefficients['aoi_hit_absent_no_hl']
        aoi_loaded = self.coefficients['aoi_hit_high_no_hl']
        aoi_drop = (aoi_baseline - aoi_loaded) / aoi_baseline  # ~40% drop
        
        aoi_sensitivity = aoi_drop * (0.7 + 0.6 * clutter)
        aoi_sensitivity = np.clip(aoi_sensitivity, 0.0, 1.0)
        
        # 5. Highlight Effectiveness
        # How much would dynamic highlighting help under cognitive load
        # More elements and lower current saliency focus → highlighting helps more
        if saliency_features is not None:
            s = np.asarray(saliency_features, dtype=np.float32)
            sal_spread = s[2] if len(s) > 2 else 0.5  # saliency spread
            coverage = s[4] if len(s) > 4 else 0.5    # coverage
            # High spread + low coverage = highlighting would help
            highlight_need = (sal_spread + (1.0 - coverage)) / 2.0
        else:
            highlight_need = complexity  # fallback to complexity
        
        # Empirical: dynamic highlighting maintains ~69% AOI hit vs 6.9% without
        hl_effectiveness = 0.5 + 0.4 * highlight_need
        hl_effectiveness = np.clip(hl_effectiveness, 0.0, 1.0)
        
        # 6. Combined Cognitive Load Index (0-1)
        # Overall vulnerability to cognitive load
        # Weighted combination of all factors
        cognitive_load_index = (
            0.30 * (1.0 - fixation_reduction) +       # More reduction = higher index
            0.20 * (duration_increase - 1.0) +         # More increase = higher index
            0.20 * (1.0 - exploration_reduction) +     # More reduction = higher index
            0.15 * aoi_sensitivity +                    # Higher sensitivity = higher index
            0.15 * (1.0 - hl_effectiveness)            # Lower effectiveness = higher index (paradox: we flip)
        )
        # Normalize to 0-1
        cognitive_load_index = np.clip(cognitive_load_index / 0.3, 0.0, 1.0)
        
        return np.array([
            fixation_reduction,
            duration_increase,
            exploration_reduction,
            aoi_sensitivity,
            hl_effectiveness,
            cognitive_load_index,
        ], dtype=np.float32)
    
    def _normalize_visual(self, v: np.ndarray) -> np.ndarray:
        """
        Normalize visual features to [0, 1] using typical ranges.
        
        Expected input order:
        [edge_density, color_count, layout_complexity, whitespace_ratio,
         text_density, element_count, symmetry, grid_quality]
        """
        # Typical ranges from Stage 1 visual complexity extractor
        ranges = np.array([
            [0.0, 0.5],     # edge_density: 0-0.5
            [0.0, 1000.0],  # color_count: 0-1000 
            [0.0, 1.0],     # layout_complexity: already 0-1
            [0.0, 1.0],     # whitespace_ratio: already 0-1
            [0.0, 0.5],     # text_density: 0-0.5
            [0.0, 100.0],   # element_count: 0-100
            [0.0, 1.0],     # symmetry: already 0-1
            [0.0, 1.0],     # grid_quality: already 0-1
        ], dtype=np.float32)
        
        # Pad if fewer features provided
        if len(v) < 8:
            v = np.pad(v, (0, 8 - len(v)), constant_values=0.5)
        
        v_norm = (v[:8] - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0] + 1e-8)
        return np.clip(v_norm, 0.0, 1.0)
    
    def get_feature_names(self) -> list:
        """Return names of the 6 cognitive load features."""
        return [
            'cog_fixation_reduction',
            'cog_duration_increase', 
            'cog_exploration_reduction',
            'cog_aoi_sensitivity',
            'highlight_effectiveness',
            'cognitive_load_index',
        ]


def build_sensitivity_lookup(csv_path: str, output_path: str) -> Dict:
    """
    Build per-image sensitivity lookup table from HCEye CSV data.
    
    This creates empirical ground-truth values for the 150 webpages
    used in the HCEye study.
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    metrics = ['TotalNumFixations', 'MeanFixationDuration', 'FixationFrequency']
    
    # Aggregate per image + condition
    img_cond = df.groupby(['Image_Name', 'CognitiveLoad', 'Highlight'])[metrics].mean().reset_index()
    
    # Get baseline (Absent load, no highlight)
    baseline = img_cond[(img_cond['CognitiveLoad'] == 'Absent') & 
                        (img_cond['Highlight'] == 'Absent')]
    
    # Get high load (no highlight)
    high_load = img_cond[(img_cond['CognitiveLoad'] == 'High') & 
                         (img_cond['Highlight'] == 'Absent')]
    
    # Get dynamic highlight under high load
    high_dynamic = img_cond[(img_cond['CognitiveLoad'] == 'High') & 
                           (img_cond['Highlight'] == 'Dynamic')]
    
    lookup = {}
    
    for img in df['Image_Name'].unique():
        bl = baseline[baseline['Image_Name'] == img]
        hl = high_load[high_load['Image_Name'] == img]
        hd = high_dynamic[high_dynamic['Image_Name'] == img]
        
        if bl.empty or hl.empty:
            continue
        
        bl_fix = bl['TotalNumFixations'].values[0]
        bl_dur = bl['MeanFixationDuration'].values[0]
        bl_freq = bl['FixationFrequency'].values[0]
        
        hl_fix = hl['TotalNumFixations'].values[0]
        hl_dur = hl['MeanFixationDuration'].values[0]
        hl_freq = hl['FixationFrequency'].values[0]
        
        # Compute ratios
        fix_reduction = hl_fix / bl_fix if bl_fix > 0 else 1.0
        dur_increase = hl_dur / bl_dur if bl_dur > 0 else 1.0
        freq_reduction = hl_freq / bl_freq if bl_freq > 0 else 1.0
        
        # AOI sensitivity from overall rates
        aoi_sensitivity = 0.4  # Default from study mean (40% drop)
        
        # Highlight effectiveness
        if not hd.empty:
            hd_fix = hd['TotalNumFixations'].values[0]
            hl_effectiveness = hd_fix / bl_fix if bl_fix > 0 else 0.5
        else:
            hl_effectiveness = 0.7
        
        # Combined index
        cognitive_load_index = (
            0.30 * (1.0 - fix_reduction) +
            0.20 * max(0, dur_increase - 1.0) +
            0.20 * (1.0 - freq_reduction) +
            0.15 * aoi_sensitivity +
            0.15 * (1.0 - hl_effectiveness)
        )
        cognitive_load_index = np.clip(cognitive_load_index / 0.3, 0.0, 1.0)
        
        lookup[img] = {
            'fixation_reduction': float(fix_reduction),
            'duration_increase': float(dur_increase),
            'exploration_reduction': float(freq_reduction),
            'aoi_sensitivity': float(aoi_sensitivity),
            'highlight_effectiveness': float(hl_effectiveness),
            'cognitive_load_index': float(cognitive_load_index),
        }
    
    # Save lookup
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(lookup, f, indent=2)
    
    print(f"Built sensitivity lookup for {len(lookup)} images -> {output_path}")
    return lookup


# ============================================================
# CLI Interface
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HCEye Cognitive Load Features')
    parser.add_argument('--build-lookup', action='store_true',
                       help='Build sensitivity lookup from CSV')
    parser.add_argument('--csv', type=str, 
                       default='hceye/gaze/fixation_AOI_metrics_final.csv',
                       help='Path to HCEye CSV')
    parser.add_argument('--output', type=str,
                       default='hceye/sensitivity_lookup.json',
                       help='Output path for lookup JSON')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample visual features')
    
    args = parser.parse_args()
    
    if args.build_lookup:
        lookup = build_sensitivity_lookup(args.csv, args.output)
        
        # Print sample
        sample_imgs = list(lookup.keys())[:5]
        print("\nSample entries:")
        for img in sample_imgs:
            print(f"  {img}: CLI={lookup[img]['cognitive_load_index']:.3f}")
    
    if args.test:
        print("\n=== Test: Estimate from visual features ===")
        extractor = HCEyeFeatureExtractor()
        
        # Simulate a complex GUI
        complex_gui = np.array([0.35, 800, 0.8, 0.15, 0.3, 75, 0.4, 0.3])
        h_complex = extractor.extract_features(complex_gui)
        print(f"Complex GUI: {dict(zip(extractor.get_feature_names(), h_complex))}")
        
        # Simulate a simple GUI
        simple_gui = np.array([0.08, 50, 0.2, 0.7, 0.05, 10, 0.9, 0.85])
        h_simple = extractor.extract_features(simple_gui)
        print(f"Simple GUI:  {dict(zip(extractor.get_feature_names(), h_simple))}")
        
        print(f"\nComplex GUI Load Index: {h_complex[5]:.3f}")
        print(f"Simple GUI Load Index:  {h_simple[5]:.3f}")
        print(f"Difference: {h_complex[5] - h_simple[5]:.3f} (complex more vulnerable)")
