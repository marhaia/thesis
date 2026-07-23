"""Shared constants + helpers for the UMSI++ TF1/Keras2 <-> TF2/Keras3 golden run.

Deliberately depends ONLY on the Python standard library + numpy so that it
imports cleanly in BOTH runtimes:
  * legacy: Python 3.7.9 / TF 1.14.0 / Keras 2.3.1 / numpy 1.19.2
  * modern: Python 3.9.6 / TF 2.16.2 / Keras 3.10.0 / numpy 1.26.4

Nothing here is model-specific; the two runners and the comparator all import
the same fixture registry, feature/HCEye key order, and pre-registered
thresholds from this single source of truth.
"""
import hashlib

# ---------------------------------------------------------------------------
# The five pinned fixtures (canonical name -> required H, W, SHA-256).
# The audit fails BLOCKED if any on-disk fixture does not match these exactly.
# ---------------------------------------------------------------------------
FIXTURES = {
    "lowcontrast": {"h": 800, "w": 1280,
                    "sha256": "4a467bdd72d41b9fd490573468db171a254e7cbf4aa8727ac8fb86bef05a3562"},
    "ui1":         {"h": 800, "w": 1280,
                    "sha256": "b0741b0660e3cede9a8f11db92c3973e4336e4c8c859315ba69c8677963ece3b"},
    "ui2":         {"h": 768, "w": 1024,
                    "sha256": "1e693f150402276b819aaa133c17ef5e9540e66789518a47856b17892d7266a6"},
    "ui3":         {"h": 600, "w": 800,
                    "sha256": "38c4348cbf93da4e95f07d5e2bf7b7a315eec57b9841a8dfc174b8a3bab6cab0"},
    "uniform":     {"h": 800, "w": 1280,
                    "sha256": "2987ab872f4a798686ce3f975c7a8a3ad41bbf619fae113678b5afc53a634db6"},
}

# Deterministic fixture ordering (used everywhere for reproducible output).
FIXTURE_ORDER = ["lowcontrast", "ui1", "ui2", "ui3", "uniform"]

# The authoritative checkpoint.
CHECKPOINT_SHA256 = "f4290c3f11f18befbb47de50d81e4555ec8e7a63066c71c343a32fe32799e9fe"
CHECKPOINT_BYTES = 120093896

# The pinned upstream (legacy) reference.
UPSTREAM_REPO = "YueJiang-nj/UEyes-CHI2023"
UPSTREAM_COMMIT = "7bc064175310284c74e64e0c3c2ef264bfbe66e2"

# Order of the 5 saliency-derived features (as returned by
# saliency.saliency_features.extract_saliency_features).
FEAT_KEYS = [
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
]

# Order of the 6 HCEye outputs (as returned by
# hceye.hceye_features.HCEyeFeatureExtractor.extract_features).
HCE_KEYS = [
    "fixation_reduction",
    "duration_increase",
    "exploration_reduction",
    "aoi_sensitivity",
    "highlight_effectiveness",
    "cognitive_load_index",
]

# ---------------------------------------------------------------------------
# PRE-REGISTERED pass criteria (Phase 0). Frozen BEFORE any prediction is run.
# Do not change these after observing results.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "pearson_min": 0.999,
    "spearman_min": 0.99,
    "ssim_min": 0.99,
    "feature_abs_max": 0.02,        # per continuous 0-1 saliency feature
    "peak_count_exact": True,       # saliency_peak_count must match exactly
    "cli_abs_max": 0.01,            # provisional cognitive_load_index abs diff
    "preproc_abs_max": 0.0,         # preprocessing tensors must be identical
    "repeatability_abs_max": 0.0,   # bit-identical within each environment
}

# saliency_peak_count is an integer count; treat it as the "exact match"
# feature and exclude it from the <=0.02 continuous-feature band.
CONTINUOUS_FEATURES = [
    "saliency_dispersion",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr):
    import numpy as np
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()
