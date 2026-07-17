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


# Mapping from the concept each HCEye rule needs to the Stage-1 feature that
# supplies it. "equal" = same construct; "proxy" = closest available stand-in.
# whitespace_ratio and text_density are measured separately (element bounding
# boxes / OCR); color_count is intentionally omitted because no rule uses it.
# See the thesis methods section "Feature Mapping: Justification and Limits".
HCEYE_FEATURE_MAP = {
    "edge_density":      ("edge_density", "equal"),
    "layout_complexity": ("feature_congestion", "proxy"),   # both measure clutter
    "element_count":     ("interactive_element_density", "equal"),
    "symmetry":          ("layout_symmetry", "equal"),
    "grid_quality":      ("visual_hierarchy", "proxy"),      # layout order; valid after Bug-2 fix
}


class HCEyeFeatureExtractor:
    """
    Extracts cognitive-load sensitivity features for GUI screenshots.
    
    Uses a regression model trained on the relationship between visual
    complexity features and cognitive load sensitivity observed in HCEye.
    For new images (not in HCEye), estimates sensitivity from visual features.
    """
    
    def __init__(self, lookup_path: Optional[str] = None,
                 feature_norms_path: Optional[str] = None):
        """
        Initialize the HCEye feature extractor.
        
        Args:
            lookup_path: Path to pre-computed per-image sensitivity data.
                        If None, uses visual-feature-based estimation.
            feature_norms_path: Path to feature_norms.json (the empirical
                        reference distribution over 1,485 GUI screenshots).
                        Used to percentile-normalise each Stage-1 feature so
                        that no slot saturates. If None, the default location
                        stage1/data/results/feature_norms.json is used.
        """
        self.coefficients = HCEYE_COEFFICIENTS
        self.sensitivity_lookup = {}
        
        if lookup_path and os.path.exists(lookup_path):
            with open(lookup_path, 'r') as f:
                self.sensitivity_lookup = json.load(f)

        # Load the empirical reference distribution (percentiles per feature).
        # Percentile normalisation replaces the old fixed hand-ranges, which
        # saturated features whose real range did not match the guessed range.
        if feature_norms_path is None:
            feature_norms_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "stage1", "data", "results", "feature_norms.json",
            )
        self.feature_norms = {}
        if os.path.exists(feature_norms_path):
            with open(feature_norms_path, 'r') as f:
                self.feature_norms = json.load(f).get("features", {})

    
    def extract_features(self,
                         visual_features: Dict[str, float],
                         saliency_features: Optional[Dict[str, float]] = None,
                         whitespace_ratio: Optional[float] = None,
                         text_density: Optional[float] = None,
                         image_name: Optional[str] = None) -> np.ndarray:
        """
        Extract cognitive-load sensitivity features (h ∈ ℝ⁶) for a GUI.

        Named-field interface (not array positions): the caller passes the
        Stage-1 feature dicts directly, which makes the feature→concept mapping
        explicit and structurally rules out the position-mismatch bug this
        method previously had.

        Args:
            visual_features: Stage-1 visual complexity dict with keys
                {shannon_entropy, edge_density, feature_congestion,
                 subband_entropy, layout_symmetry, chromatic_coherence,
                 visual_hierarchy, interactive_element_density}.
            saliency_features: Optional saliency dict with keys
                {saliency_dispersion, saliency_coverage, ...}.
            whitespace_ratio: Optional real whitespace ratio in [0, 1]
                (1 − covered_area / image_area) measured from the element
                bounding boxes. If None, it is derived from element density.
            text_density: Optional real text density in [0, 1] from OCR. If
                None, a neutral value (0.5) is used (the caller flags the
                source in the API response).
            image_name: Optional name to check against the HCEye lookup table.
                NOTE: the lookup only contains HCEye's own study images, so this
                branch is used exclusively by offline analysis scripts on those
                images. The live Flask app never passes image_name (a user's
                novel screenshot can never be in the lookup), so it always takes
                the feature-estimate path below. This is intended behaviour, not
                a missing wiring.

        Returns:
            h ∈ ℝ⁶ cognitive load feature vector.
        """
        # Lookup path: only reachable for HCEye study images via offline scripts.
        if image_name and image_name in self.sensitivity_lookup:
            return self._from_lookup(image_name)
        # Estimate path: always taken by the live app (novel screenshots).
        return self._estimate_from_features(
            visual_features, saliency_features, whitespace_ratio, text_density,
        )
    
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
                                visual_features: Dict[str, float],
                                saliency_features: Optional[Dict[str, float]] = None,
                                whitespace_ratio: Optional[float] = None,
                                text_density: Optional[float] = None) -> np.ndarray:
        """
        Estimate cognitive-load sensitivity from Stage-1 features.

        Each Stage-1 feature is percentile-normalised against the empirical
        reference distribution (feature_norms.json), so its value becomes
        "where this screen sits relative to a typical GUI" in [0, 1]. This
        removes the saturation that fixed hand-ranges caused.

        The HCEye effect coefficients (Das et al., 2024) are unchanged; only
        the mapping of features onto them is a purpose-built, literature-guided
        heuristic (direction motivated by Rosenholtz 2007 / Rayner 1998; the
        exact weighting is not empirically calibrated and is examined by the
        user study).
        """
        vf = visual_features or {}

        def concept(name: str) -> float:
            """Percentile-normalised value of the Stage-1 feature mapped onto a
            given HCEye concept (see HCEYE_FEATURE_MAP)."""
            stage1_key, _kind = HCEYE_FEATURE_MAP[name]
            return self._percentile_normalize(vf.get(stage1_key), stage1_key)

        edge = concept("edge_density")
        layout_complexity = concept("layout_complexity")
        element_count = concept("element_count")
        symmetry = concept("symmetry")
        grid_quality = concept("grid_quality")

        # whitespace_ratio: real measurement in [0, 1]; fall back to the inverse
        # of element density only if it was not supplied.
        if whitespace_ratio is None:
            whitespace = float(np.clip(1.0 - element_count, 0.0, 1.0))
        else:
            whitespace = float(np.clip(whitespace_ratio, 0.0, 1.0))

        # text_density: real OCR measurement in [0, 1]; neutral fallback = 0.5.
        text = 0.5 if text_density is None else float(np.clip(text_density, 0.0, 1.0))

        # 1. Cognitive Fixation Reduction — more complex GUIs reduce fixations more.
        complexity = (edge + element_count + layout_complexity) / 3.0
        simplicity = whitespace
        fixation_reduction = self.coefficients['fixation_reduction_mean'] - \
            0.05 * (complexity - simplicity)
        fixation_reduction = float(np.clip(fixation_reduction, 0.6, 1.1))

        # 2. Cognitive Duration Increase — more text/elements → longer processing.
        processing_demand = (text + element_count) / 2.0
        duration_increase = self.coefficients['duration_increase_mean'] + \
            0.1 * processing_demand
        duration_increase = float(np.clip(duration_increase, 0.8, 1.5))

        # 3. Exploration Reduction — less predictable layout → more reduction.
        layout_quality = (symmetry + grid_quality) / 2.0
        exploration_reduction = self.coefficients['frequency_reduction_mean'] - \
            0.04 * (1.0 - layout_quality)
        exploration_reduction = float(np.clip(exploration_reduction, 0.7, 1.0))

        # 4. AOI Sensitivity — more visual clutter → greater AOI attention loss.
        clutter = 1.0 - whitespace
        aoi_baseline = self.coefficients['aoi_hit_absent_no_hl']
        aoi_loaded = self.coefficients['aoi_hit_high_no_hl']
        aoi_drop = (aoi_baseline - aoi_loaded) / aoi_baseline  # ~40% drop
        aoi_sensitivity = float(np.clip(aoi_drop * (0.7 + 0.6 * clutter), 0.0, 1.0))

        # 5. Highlight Effectiveness — high saliency spread + low coverage → helps.
        if saliency_features:
            sal_spread = self._percentile_normalize(
                saliency_features.get("saliency_dispersion"), "saliency_dispersion")
            coverage = self._percentile_normalize(
                saliency_features.get("saliency_coverage"), "saliency_coverage")
            highlight_need = (sal_spread + (1.0 - coverage)) / 2.0
        else:
            highlight_need = complexity  # fallback to complexity
        hl_effectiveness = float(np.clip(0.5 + 0.4 * highlight_need, 0.0, 1.0))

        # 6. Combined Cognitive Load Index (0–1) — weighted combination.
        cognitive_load_index = (
            0.30 * (1.0 - fixation_reduction) +       # More reduction = higher index
            0.20 * (duration_increase - 1.0) +         # More increase = higher index
            0.20 * (1.0 - exploration_reduction) +     # More reduction = higher index
            0.15 * aoi_sensitivity +                    # Higher sensitivity = higher index
            0.15 * (1.0 - hl_effectiveness)            # Lower effectiveness = higher index
        )
        cognitive_load_index = float(np.clip(cognitive_load_index / 0.3, 0.0, 1.0))

        # --- Empty-canvas anchoring (supervisor sanity test) ----------------
        # The five terms above carry non-zero floors: highlight effectiveness
        # has a fixed 0.5 base and AOI sensitivity a constant ~0.28 offset, so
        # even a blank canvas would score ~0.4 ("medium load"). Interaction
        # load is only meaningful to the degree the screen actually presents
        # content to process, so we scale the index by a measured content-
        # presence factor. A near-empty screen (no edges, no elements, near-
        # total whitespace) -> presence ~0 -> low load; a content-rich screen
        # -> presence ~1 -> index effectively unchanged. This anchors the
        # headline at the bottom of its range instead of a fixed floor.
        content_presence = float(np.clip(
            max(edge, element_count, layout_complexity, 1.0 - whitespace),
            0.0, 1.0,
        ))
        cognitive_load_index *= content_presence

        return np.array([
            fixation_reduction,
            duration_increase,
            exploration_reduction,
            aoi_sensitivity,
            hl_effectiveness,
            cognitive_load_index,
        ], dtype=np.float32)

    def _percentile_normalize(self, value: Optional[float], feature_key: str) -> float:
        """
        Map a raw feature value to [0, 1] by its position in the empirical
        reference distribution (feature_norms.json).

        Piecewise-linear interpolation across the reference anchors
        min < p5 < p25 < p50 < p75 < p95 < max, mapped to
        0.0 < 0.05 < 0.25 < 0.50 < 0.75 < 0.95 < 1.0.

        Including the empirical min/max (not only p5/p95) matters for screens
        that sit in the extreme tails of the reference distribution: automotive
        HMIs are much sparser than the web/mobile GUIs the reference was built
        from, so several of their features fall below p5. Anchoring on min/max
        keeps those tail values ordered (a slightly cleaner screen still scores
        lower) instead of hard-clamping every tail value to the same 0.05 floor.
        Values below min / above max clamp to 0.0 / 1.0. If the feature has no
        reference entry, or the value is missing, returns a neutral 0.5.
        """
        if value is None:
            return 0.5
        norms = self.feature_norms.get(feature_key)
        if not norms:
            return 0.5
        anchors = [
            (norms.get("min"), 0.0),
            (norms.get("p5"), 0.05),
            (norms.get("p25"), 0.25),
            (norms.get("p50"), 0.50),
            (norms.get("p75"), 0.75),
            (norms.get("p95"), 0.95),
            (norms.get("max"), 1.0),
        ]
        # Keep only strictly increasing x-anchors so np.interp is well defined
        # (some features have degenerate anchors, e.g. min == p5 == 0).
        xs: list = []
        ys: list = []
        for x, y in anchors:
            if x is None:
                continue
            if not xs or float(x) > xs[-1]:
                xs.append(float(x))
                ys.append(y)
        if not xs:
            return 0.5
        return float(np.interp(float(value), xs, ys))

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

        # Simulate a complex GUI (named Stage-1 features).
        complex_gui = {
            "shannon_entropy": 6.5, "edge_density": 0.09, "feature_congestion": 7.0,
            "subband_entropy": 3.8, "layout_symmetry": 0.15,
            "chromatic_coherence": 0.6, "visual_hierarchy": 0.30,
            "interactive_element_density": 0.9,
        }
        h_complex = extractor.extract_features(complex_gui, whitespace_ratio=0.15)
        print(f"Complex GUI: {dict(zip(extractor.get_feature_names(), h_complex))}")

        # Simulate a simple GUI.
        simple_gui = {
            "shannon_entropy": 2.5, "edge_density": 0.02, "feature_congestion": 2.5,
            "subband_entropy": 1.8, "layout_symmetry": 0.7,
            "chromatic_coherence": 0.4, "visual_hierarchy": 0.85,
            "interactive_element_density": 0.1,
        }
        h_simple = extractor.extract_features(simple_gui, whitespace_ratio=0.7)
        print(f"Simple GUI:  {dict(zip(extractor.get_feature_names(), h_simple))}")

        print(f"\nComplex GUI Load Index: {h_complex[5]:.3f}")
        print(f"Simple GUI Load Index:  {h_simple[5]:.3f}")
        print(f"Difference: {h_complex[5] - h_simple[5]:.3f} (complex more vulnerable)")
