"""
Stage 2 — Multi-Output Regression Model
=========================================
Predicts cognitive load indices from the combined feature vector produced
by Stage 1. Uses classical ML (Ridge Regression, Random Forest, XGBoost)
on pre-extracted feature vectors — no CNN/ViT.

Input: Feature vector x ∈ ℝ¹⁹ (or ℝ¹⁴ without saliency)
    - v ∈ ℝ⁸: Visual complexity features
    - s ∈ ℝ⁵: Saliency features (optional)
    - h ∈ ℝ⁶: HCEye cognitive load sensitivity features

Output: y ∈ ℝ³ (multi-output)
    - cognitive_load_score: Overall cognitive load index (0-100)
    - search_efficiency: Expected visual search efficiency (0-1)
    - attention_demand: Attentional demand level (0-1)

Training Data:
    Uses HCEye empirical data (150 webpages × 27 participants × 3 load conditions)
    as ground truth for how cognitive load affects gaze behavior.

    NOTE: the current `build_training_data` helper is a SCAFFOLD only. Its (X, y)
    pairs are circular (the targets leak from the h block that is part of x) and
    its v/s features are fabricated from gaze statistics rather than extracted
    from images. See the warning in that function: no R^2/SHAP result from it may
    be reported as evidence. The trained path is off by default and no model file
    is shipped; the deployed cognitive-load score comes from the HCEye rule block
    (h[5]), not from this regressor.

Architecture Decision:
    Classical regression per pipeline_guide.md §4.2 (adapted from neural multi-head).
    Ridge as baseline, XGBoost as primary model. Cross-validated.

References:
    Das, S., Husain, S., Bhatt, U., & Oulasvirta, A. (2024).
        Shifting Focus with HCEye. PACM HCI / ETRA 2024.
    Jokinen, J.P.P. et al. (2020). Adaptive Feature Guidance. IJHCS, 136.
    Hoerl, A.E., & Kennard, R.W. (1970). Ridge regression: Biased estimation
        for nonorthogonal problems. Technometrics, 12(1), 55–67.  [Ridge]
    Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.  [RF]
    Borchani, H. et al. (2015). A survey on multi-output regression.
        WIREs Data Mining & Knowledge Discovery, 5(5), 216–233.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for sklearn/xgboost (not loaded until training)
_SKLEARN_AVAILABLE = False
_XGBOOST_AVAILABLE = False


def _check_deps():
    global _SKLEARN_AVAILABLE, _XGBOOST_AVAILABLE
    try:
        import sklearn
        _SKLEARN_AVAILABLE = True
    except ImportError:
        pass
    try:
        import xgboost
        _XGBOOST_AVAILABLE = True
    except Exception:
        pass


class Stage2Model:
    """
    Multi-output regression model for cognitive load prediction.
    
    Supports multiple backends:
        - 'ridge': Ridge Regression (sklearn)
        - 'rf': Random Forest (sklearn)
        - 'xgb': XGBoost (xgboost)
    """
    
    # Feature names for the full vector (v∈ℝ⁸ + s∈ℝ⁵ + h∈ℝ⁶ = ℝ¹⁹)
    VISUAL_FEATURES = [
        'shannon_entropy', 'edge_density', 'feature_congestion',
        'subband_entropy', 'layout_symmetry', 'chromatic_coherence',
        'visual_hierarchy', 'interactive_element_density'
    ]
    # Saliency features from saliency/saliency_features.py → extract_saliency_features()
    # Keys must match exactly what that function returns.
    SALIENCY_FEATURES = [
        'saliency_dispersion',   # spatial σ of attention distribution
        'saliency_entropy',      # Shannon entropy of saliency map
        'saliency_coverage',     # fraction of image with S > 0.5·max(S)
        'saliency_peak_count',   # number of distinct attention peaks
        'saliency_center_bias',  # proportion of attention in central 25% of image
    ]
    COGNITIVE_FEATURES = [
        'cog_fixation_reduction', 'cog_duration_increase',
        'cog_exploration_reduction', 'cog_aoi_sensitivity',
        'highlight_effectiveness', 'cognitive_load_index'
    ]
    
    OUTPUT_NAMES = [
        'cognitive_load_score',   # 0-100 composite score
        'search_efficiency',      # 0-1 (how easy to find elements)
        'attention_demand',       # 0-1 (how much attentional resource needed)
    ]
    
    def __init__(self, model_type: str = 'ridge', model_path: Optional[str] = None):
        """
        Initialize Stage 2 model.
        
        Args:
            model_type: 'ridge', 'rf', or 'xgb'
            model_path: Path to load a pre-trained model from
        """
        _check_deps()
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = self.VISUAL_FEATURES + self.SALIENCY_FEATURES + self.COGNITIVE_FEATURES
        self.n_features = len(self.feature_names)  # 19
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def _create_model(self):
        """
        Create the underlying ML model.

        Backend choices and references:
          Ridge  — L2-regularized linear regression; closed-form solution.
                   Hoerl & Kennard (1970), Technometrics, 12(1), 55–67.
                   alpha=1.0 is the default regularization strength.
          RF     — Ensemble of randomized decision trees; robust to collinear
                   features and high-dimensional inputs (our x∈ℝ¹⁹ is small,
                   but feature correlations exist between v and s components).
                   Breiman (2001), Machine Learning, 45(1), 5–32.
          XGBoost— Gradient-boosted trees; typically outperforms RF on tabular
                   data with sufficient samples. Requires xgboost package.
                   Chen & Guestrin (2016), KDD'16.
          MultiOutputRegressor — sklearn wrapper that fits one estimator per
                   output independently.
                   Borchani et al. (2015), WIREs Data Mining, 5(5), 216–233.
        """
        if self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            from sklearn.multioutput import MultiOutputRegressor
            # alpha=1.0: default ridge penalty (Hoerl & Kennard, 1970)
            self.model = MultiOutputRegressor(Ridge(alpha=1.0))

        elif self.model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            # n_estimators=100, max_depth=10: conservative settings for
            # small datasets (N≈150 HCEye images); Breiman (2001)
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            )
            
        elif self.model_type == 'xgb':
            if not _XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Use: pip install xgboost")
            from sklearn.multioutput import MultiOutputRegressor
            import xgboost as xgb
            self.model = MultiOutputRegressor(
                xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2) -> Dict:
        """
        Train the Stage 2 model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target matrix (n_samples, 3) — [cog_load, search_eff, attn_demand]
            validation_split: Fraction for validation
            
        Returns:
            Dict with training metrics
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        
        # Create and train model
        self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        metrics = {
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_outputs': y.shape[1],
            'train': {
                'r2': float(r2_score(y_train, y_pred_train, multioutput='uniform_average')),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
            },
            'val': {
                'r2': float(r2_score(y_val, y_pred_val, multioutput='uniform_average')),
                'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                'mae': float(mean_absolute_error(y_val, y_pred_val)),
            },
            'per_output': {}
        }
        
        for i, name in enumerate(self.OUTPUT_NAMES):
            metrics['per_output'][name] = {
                'val_r2': float(r2_score(y_val[:, i], y_pred_val[:, i])),
                'val_rmse': float(np.sqrt(mean_squared_error(y_val[:, i], y_pred_val[:, i]))),
            }
        
        # Cross-validation (on full data)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, test_idx in kf.split(X_scaled):
            self._create_model()
            self.model.fit(X_scaled[train_idx], y[train_idx])
            y_pred_cv = self.model.predict(X_scaled[test_idx])
            cv_scores.append(r2_score(y[test_idx], y_pred_cv, multioutput='uniform_average'))
        
        metrics['cv_r2_mean'] = float(np.mean(cv_scores))
        metrics['cv_r2_std'] = float(np.std(cv_scores))
        
        # Retrain on full data for final model
        self._create_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return metrics
    
    def predict(self, x: np.ndarray) -> Dict:
        """
        Predict cognitive load outputs for a feature vector.
        
        Args:
            x: Feature vector (n_features,) or matrix (n_samples, n_features)
            
        Returns:
            Dict with predictions for each output
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Handle missing features (pad to expected size)
        if x.shape[1] < self.n_features:
            padding = np.zeros((x.shape[0], self.n_features - x.shape[1]))
            x = np.hstack([x, padding])
        
        x_scaled = self.scaler.transform(x)
        y_pred = self.model.predict(x_scaled)
        
        # Clip to valid ranges
        y_pred[:, 0] = np.clip(y_pred[:, 0], 0, 100)  # cognitive_load_score
        y_pred[:, 1] = np.clip(y_pred[:, 1], 0, 1)    # search_efficiency
        y_pred[:, 2] = np.clip(y_pred[:, 2], 0, 1)    # attention_demand
        
        if y_pred.shape[0] == 1:
            return {
                name: float(y_pred[0, i])
                for i, name in enumerate(self.OUTPUT_NAMES)
            }
        
        return {
            name: y_pred[:, i].tolist()
            for i, name in enumerate(self.OUTPUT_NAMES)
        }
    
    def save(self, path: str):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a pre-trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        print(f"Model loaded from {path} (type={self.model_type})")


def build_training_data(hceye_csv_path: str, 
                        sensitivity_lookup_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training data from HCEye empirical measurements.

    Creates (X, y) pairs where:
        X = simulated feature vectors based on HCEye image properties
        y = ground-truth cognitive load effects from the study

    Returns:
        X: (n_samples, 19) feature matrix
        y: (n_samples, 3) target matrix

    WARNING — this training set is CIRCULAR and must not be used as evidence.
    ---------------------------------------------------------------------------
    The audit (2026-07) confirmed two problems that make any metric computed on
    this data scientifically invalid:

    1. Target leakage. The feature vector is x = [v(8), s(5), h(6)], so the
       HCEye block h occupies x[13..18]. The targets are direct functions of
       that same block:
           y1 = sens['cognitive_load_index'] * 100        == x[18] * 100
           y2 = affine(sens['fixation_reduction'])         from x[13]
           y3 = affine(sens['duration_increase'])          from x[14]
       The model is therefore trained to predict values that are already part
       of its own input. Any R^2, RMSE, cross-validation score, feature
       ablation or SHAP result from this data is near-tautological.

    2. Fabricated inputs. The v and s blocks are NOT extracted from images here;
       they are synthesised from per-image gaze statistics (see below), so they
       do not correspond to the pixel-based features used at inference time.

    Consequently this function exists only as a scaffold/placeholder. It is off
    by default in the pipeline (use_trained_model=False, no model file shipped)
    and NO number derived from it may appear in the thesis as evidence about the
    pipeline's predictive validity. A valid Stage-2 model needs real per-image
    v/s features and targets that are independent of the h block.
    """
    import pandas as pd

    print(
        "[Stage2] WARNING: build_training_data() produces CIRCULAR training data "
        "(targets leak from the h block in x, v/s are fabricated from gaze stats). "
        "Any R^2/RMSE/SHAP from it is tautological and must NOT be reported as "
        "evidence. Scaffold only."
    )
    
    df = pd.read_csv(hceye_csv_path)
    
    with open(sensitivity_lookup_path, 'r') as f:
        sensitivity = json.load(f)
    
    # Aggregate metrics per image
    metrics = ['TotalNumFixations', 'MeanFixationDuration', 'FixationFrequency']
    
    # Get baseline (absent load, absent highlight) per image
    baseline = df[(df['CognitiveLoad'] == 'Absent') & (df['Highlight'] == 'Absent')]
    baseline_agg = baseline.groupby('Image_Name')[metrics].mean()
    
    # Get high load per image
    high_load = df[(df['CognitiveLoad'] == 'High') & (df['Highlight'] == 'Absent')]
    high_agg = high_load.groupby('Image_Name')[metrics].mean()
    
    X_list = []
    y_list = []
    
    for img_name in sensitivity.keys():
        if img_name not in baseline_agg.index:
            continue
        
        sens = sensitivity[img_name]
        bl = baseline_agg.loc[img_name]
        
        # Create synthetic visual features from baseline gaze behavior
        # (In production, these come from actual image analysis)
        # Use gaze metrics as proxy for visual complexity
        fix_count_norm = bl['TotalNumFixations'] / 20.0  # normalize
        fix_dur_norm = bl['MeanFixationDuration'] / 1000.0
        fix_freq_norm = bl['FixationFrequency'] / 5.0
        
        # Simulate v∈ℝ⁸ from gaze-inferred complexity
        v = np.array([
            5.0 + fix_count_norm * 2.0,    # shannon_entropy (5-7)
            0.05 + fix_freq_norm * 0.3,     # edge_density
            fix_count_norm * 0.8,            # feature_congestion
            fix_dur_norm * 3.0,              # subband_entropy
            1.0 - fix_count_norm * 0.3,      # layout_symmetry (inverse)
            fix_freq_norm * 0.6,             # chromatic_coherence
            1.0 - fix_dur_norm * 0.5,        # visual_hierarchy (inverse)
            fix_count_norm * 50.0,           # interactive_element_density
        ], dtype=np.float32)
        
        # Simulate s∈ℝ⁵ 
        s = np.array([
            0.3 + fix_count_norm * 0.5,     # peak_saliency
            0.1 + fix_freq_norm * 0.2,      # mean_saliency
            fix_count_norm * 0.7,            # saliency_spread
            max(1, fix_count_norm * 8),      # hotspot_count
            fix_freq_norm * 0.5,             # coverage
        ], dtype=np.float32)
        
        # h∈ℝ⁶ from sensitivity lookup (empirical)
        h = np.array([
            sens['fixation_reduction'],
            sens['duration_increase'],
            sens['exploration_reduction'],
            sens['aoi_sensitivity'],
            sens['highlight_effectiveness'],
            sens['cognitive_load_index'],
        ], dtype=np.float32)
        
        # Full feature vector
        x = np.concatenate([v, s, h])
        
        # Target: derive from empirical cognitive load effects
        # y1: cognitive_load_score (0-100)
        # Based on how much behavior degrades under load
        cog_score = sens['cognitive_load_index'] * 100.0
        
        # y2: search_efficiency (0-1) - how easy to find targets
        # Inverse of fixation reduction (more reduction = harder to find)  
        search_eff = sens['fixation_reduction']  # 0.6-1.1, higher = easier
        search_eff = np.clip((search_eff - 0.5) / 0.6, 0, 1)  # normalize to 0-1
        
        # y3: attention_demand (0-1) - attentional resource required
        # Based on duration increase (longer fixations = more processing needed)
        attn_demand = (sens['duration_increase'] - 0.8) / 0.7  # normalize
        attn_demand = np.clip(attn_demand, 0, 1)
        
        y = np.array([cog_score, search_eff, attn_demand], dtype=np.float32)
        
        X_list.append(x)
        y_list.append(y)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"Built training data: X={X.shape}, y={y.shape}")
    print(f"  Cognitive Load Score range: [{y[:,0].min():.1f}, {y[:,0].max():.1f}]")
    print(f"  Search Efficiency range:    [{y[:,1].min():.3f}, {y[:,1].max():.3f}]")
    print(f"  Attention Demand range:     [{y[:,2].min():.3f}, {y[:,2].max():.3f}]")
    
    return X, y


# ============================================================
# CLI Interface
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2 — Cognitive Load Regression')
    parser.add_argument('--train', action='store_true', help='Train model on HCEye data')
    parser.add_argument('--model-type', type=str, default='ridge',
                       choices=['ridge', 'rf', 'xgb'], help='Model type')
    parser.add_argument('--csv', type=str,
                       default='hceye/gaze/fixation_AOI_metrics_final.csv')
    parser.add_argument('--lookup', type=str,
                       default='hceye/sensitivity_lookup.json')
    parser.add_argument('--output', type=str,
                       default='stage2/models/stage2_model.pkl')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all model types')
    
    args = parser.parse_args()
    
    if args.train or args.compare:
        # Build training data
        X, y = build_training_data(args.csv, args.lookup)
        
        if args.compare:
            print("\n" + "=" * 60)
            print("  MODEL COMPARISON (5-fold CV)")
            print("=" * 60)
            
            for mt in ['ridge', 'rf', 'xgb']:
                try:
                    model = Stage2Model(model_type=mt)
                    metrics = model.train(X, y)
                    print(f"\n--- {mt.upper()} ---")
                    print(f"  CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
                    print(f"  Val R²: {metrics['val']['r2']:.4f}")
                    print(f"  Val RMSE: {metrics['val']['rmse']:.4f}")
                    for name, m in metrics['per_output'].items():
                        print(f"    {name}: R²={m['val_r2']:.4f}, RMSE={m['val_rmse']:.4f}")
                except Exception as e:
                    print(f"\n--- {mt.upper()} --- FAILED: {e}")
        else:
            model = Stage2Model(model_type=args.model_type)
            metrics = model.train(X, y)
            
            print(f"\n{'=' * 60}")
            print(f"  TRAINING RESULTS ({args.model_type.upper()})")
            print(f"{'=' * 60}")
            print(f"  CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            print(f"  Val R²: {metrics['val']['r2']:.4f}")
            print(f"  Val RMSE: {metrics['val']['rmse']:.4f}")
            print(f"  Val MAE: {metrics['val']['mae']:.4f}")
            
            # Save model
            model.save(args.output)
            
            # Test prediction
            print(f"\n  Sample prediction (first training sample):")
            pred = model.predict(X[0])
            print(f"    {pred}")
