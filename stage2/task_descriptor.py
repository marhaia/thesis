"""Task descriptor encoding for the pipeline concept.

Encodes a lightweight task descriptor vector `t` from structured UI inputs.
This is intentionally simple and explicit so it can be used before a
descriptor-conditioned regression model is retrained.

Weight calibration basis:
    Task-type weights reflect the relative attentional demand of each category
    as documented in the GUI complexity and visual search literature:
    - Decision tasks carry the highest weight (0.80): multiple alternatives
      must be evaluated simultaneously, maximising decisional load
      (Wickens, 2008, Multiple Resource Theory; Fitts & Posner, 1967).
    - Data entry (0.70) and search (0.65) require sustained focused attention
      (Salvucci & Taatgen, 2008, Threaded Cognition).
    - Monitoring (0.55) is largely peripheral and pre-attentive.
    - Navigation (0.45) relies on spatial memory and procedural schemata,
      reducing active attentional demand (Paas et al., 2003, CLT).
    Weights are ordinal approximations, not empirically calibrated scalars.
    They should be replaced by regression coefficients once user-study data
    are available.

    Specificity and time-pressure weights follow the same ordinal logic:
    high specificity = narrow search → lower load (fewer candidates);
    high time pressure = reduced processing time → higher load
    (Kahneman, 1973, Attention and Effort).

References:
    Fitts, P. M., & Posner, M. I. (1967). Human performance. Brooks/Cole.
    Kahneman, D. (1973). Attention and effort. Prentice-Hall.
    Paas, F., Renkl, A., & Sweller, J. (2003). Cognitive load theory.
        Educational Psychologist, 38(1), 1-4.
    Salvucci, D. D., & Taatgen, N. A. (2008). Threaded cognition.
        Psychological Review, 115(1), 101-130.
    Wickens, C. D. (2008). Multiple resources and mental workload.
        Human Factors, 50(3), 449-455.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# Task-type attentional demand weights (ordinal, not empirically calibrated).
# Ordering justified by Multiple Resource Theory (Wickens, 2008) and
# Cognitive Load Theory (Paas et al., 2003). See module docstring.
TASK_TYPE_WEIGHTS = {
    "navigation": 0.45,   # spatial/procedural — reduced active attention
    "search": 0.65,       # sustained focused attention
    "monitoring": 0.55,   # largely pre-attentive / peripheral
    "data_entry": 0.70,   # sustained attention + motor execution
    "decision": 0.80,     # highest: simultaneous evaluation of alternatives
}

# Target specificity: high specificity narrows the search set → lower load.
# Basis: visual search set-size effects (Treisman & Gelade, 1980, Cogn. Psych.).
SPECIFICITY_WEIGHTS = {"high": 0.25, "medium": 0.55, "low": 0.85}

# Time pressure: reduced processing time increases cognitive load
# (Kahneman, 1973; Wickens, 2008, Multiple Resource Theory).
TIME_PRESSURE_WEIGHTS = {"low": 0.20, "medium": 0.55, "high": 0.90}

# Search mode: known-item search is fastest (direct memory retrieval);
# exploratory search requires exhaustive scanning → highest load.
# Basis: Marchionini (1997) information-seeking model.
SEARCH_MODE_WEIGHTS = {"known_item": 0.30, "comparative": 0.70, "exploratory": 0.90}


@dataclass
class TaskDescriptor:
    task_type: str = "search"
    target_specificity: str = "medium"
    time_pressure: str = "medium"
    search_mode: str = "known_item"

    def to_vector(self) -> np.ndarray:
        task_weight = TASK_TYPE_WEIGHTS.get(self.task_type, TASK_TYPE_WEIGHTS["search"])
        specificity = SPECIFICITY_WEIGHTS.get(self.target_specificity, SPECIFICITY_WEIGHTS["medium"])
        time_pressure = TIME_PRESSURE_WEIGHTS.get(self.time_pressure, TIME_PRESSURE_WEIGHTS["medium"])
        search_mode = SEARCH_MODE_WEIGHTS.get(self.search_mode, SEARCH_MODE_WEIGHTS["known_item"])

        visual_search_intensity = (task_weight + search_mode) / 2.0
        decisional_demand = (task_weight + specificity + time_pressure) / 3.0

        return np.array(
            [
                task_weight,
                specificity,
                time_pressure,
                search_mode,
                visual_search_intensity,
                decisional_demand,
            ],
            dtype=np.float32,
        )

    def score_modifier(self) -> float:
        """Return an additive modifier on a 0-100 cognitive-load score.

        The intercept is chosen so the neutral default configuration
        (search / medium specificity / medium time pressure / known-item)
        adds exactly 0. Relative to that baseline, the highest-demand
        combination adds up to ~+11 points and the lowest-demand combination
        subtracts up to ~-7 points (further clipped to [-8, +12]). These bounds
        are heuristic; ordinal relationships follow the CLT and Multiple
        Resource Theory references in the module docstring.
        """
        vec = self.to_vector()
        modifier = (
            8.0 * vec[0] +   # task_type_weight   — largest single driver
            6.0 * vec[1] +   # specificity        — narrows/widens search set
            10.0 * vec[2] +  # time_pressure      — strongest acute load driver
            7.0 * vec[3] -   # search_mode        — exploration penalty
            16.1             # intercept: centres the neutral default at exactly 0
        )
        return float(np.clip(modifier, -8.0, 12.0))

    def as_dict(self) -> Dict[str, object]:
        vec = self.to_vector()
        return {
            "task_type": self.task_type,
            "target_specificity": self.target_specificity,
            "time_pressure": self.time_pressure,
            "search_mode": self.search_mode,
            "vector": vec.tolist(),
            "modifier": self.score_modifier(),
        }


def feature_names() -> List[str]:
    return [
        "task_type_weight",
        "target_specificity_weight",
        "time_pressure_weight",
        "search_mode_weight",
        "visual_search_intensity",
        "decisional_demand",
    ]


def estimate_search_efficiency(
    visual_hierarchy: float,
    saliency_dispersion: float,
    saliency_peak_count: float,
    task: "TaskDescriptor",
    max_peaks: int = 8,
) -> float:
    """Estimate visual search efficiency from visual features + task context.

    Implements the task-conditioned visual search efficiency concept from
    SeekUI (Guo et al., ACM CHI 2026, doi:10.1145/3772318.3791178):
    search efficiency on a GUI is jointly determined by three factors —
    (1) layout predictability (visual hierarchy), (2) attentional competition
    (number/spread of salient regions), and (3) task-layout alignment (how
    well the target specificity and search mode match the visual structure).

    This function provides an *analytical* fallback for the ``search_efficiency``
    regression output before a trained Stage 2 model is available.  It uses
    features from the existing Stage 1 pipeline and the TaskDescriptor without
    any additional inference step.

    Factor weights (sum to 1.0):
        0.40 — visual_hierarchy: strongest single predictor; clear hierarchy
               reduces search path length (Jokinen et al., 2020, IJHCS 136).
        0.25 — saliency focus (1 - normalized_peaks): fewer competing
               attentional peaks → fewer distractor fixations
               (Rosenholtz et al., 2007, J. Vision 7:2:17; SeekUI §4.2).
        0.20 — saliency concentration (1 - dispersion): concentrated saliency
               guides the eye towards the target region
               (SeekUI reward function: spatial scanpath efficiency).
        0.10 — target specificity: high specificity → narrower search set
               (Treisman & Gelade, 1980, Cognitive Psychology 12).
        0.05 — search mode: known-item search is most efficient (direct
               memory retrieval); exploratory scan is least efficient
               (Marchionini, 1997, information-seeking model).

    Args:
        visual_hierarchy:    v[6] from Stage 1 feature vector ∈ [0, 1].
        saliency_dispersion: s[0] from saliency feature vector ∈ [0, 1].
        saliency_peak_count: s[3] from saliency feature vector (raw integer
                             or float, clipped to [0, max_peaks]).
        task:                TaskDescriptor instance for the current analysis.
        max_peaks:           Normalisation ceiling for peak count (default 8).

    Returns:
        float ∈ [0, 1].  1.0 = maximally efficient; 0.0 = exhaustive search.

    References:
        Guo, Z. et al. (2026). SeekUI. ACM CHI 2026.
            https://doi.org/10.1145/3772318.3791178
        Jokinen, J. P. P. et al. (2020). Adaptive feature guidance.
            IJHCS, 136, 102376. https://doi.org/10.1016/j.ijhcs.2019.102376
        Marchionini, G. (1997). Information seeking in electronic environments.
            Cambridge University Press.
        Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
            Journal of Vision, 7(2), 17. https://doi.org/10.1167/7.2.17
        Treisman, A. M., & Gelade, G. (1980). A feature-integration theory
            of attention. Cognitive Psychology, 12(1), 97–136.
    """
    # Factor 1: layout predictability (already normalised by Stage 1)
    hierarchy_contribution = float(np.clip(visual_hierarchy, 0.0, 1.0))

    # Factor 2: attentional competition — fewer peaks = cleaner search
    normalized_peaks = float(np.clip(saliency_peak_count / max(max_peaks, 1), 0.0, 1.0))
    focus_contribution = 1.0 - normalized_peaks

    # Factor 3: saliency concentration — low dispersion = eye guided to target
    concentration_contribution = 1.0 - float(np.clip(saliency_dispersion, 0.0, 1.0))

    # Factor 4: target specificity — high specificity → narrow search set.
    # SPECIFICITY_WEIGHTS are *load* weights (high specificity = 0.25 load),
    # so invert: efficiency ∝ (1 - load_weight).
    specificity_load = SPECIFICITY_WEIGHTS.get(task.target_specificity, SPECIFICITY_WEIGHTS["medium"])
    specificity_contribution = 1.0 - specificity_load

    # Factor 5: search mode — SEARCH_MODE_WEIGHTS are *load* weights; invert.
    mode_load = SEARCH_MODE_WEIGHTS.get(task.search_mode, SEARCH_MODE_WEIGHTS["known_item"])
    mode_contribution = 1.0 - mode_load

    efficiency = (
        0.40 * hierarchy_contribution
        + 0.25 * focus_contribution
        + 0.20 * concentration_contribution
        + 0.10 * specificity_contribution
        + 0.05 * mode_contribution
    )
    return float(np.clip(efficiency, 0.0, 1.0))