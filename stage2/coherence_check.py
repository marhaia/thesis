"""
Mutual Coherence Check for the Two-Stage Pipeline
==================================================
Verifies that the three primary pipeline outputs are internally consistent.
Incoherent output combinations are flagged as warnings — they do not invalidate
the prediction but indicate that the inputs are at the boundary of the model's
calibration range.

Scientific basis for each rule:

Rule 1 — Saliency spread vs. fixation count:
    High feature congestion (many competing saliency peaks) forces the visual
    system to perform more fixations to resolve attentional competition.
    Rosenholtz et al. (2007, J. Vision 7:2:17) show that clutter directly
    increases the number of fixations needed to locate a target. Jokinen et al.
    (2020, IJHCS 136:102376) model this as increased visual search steps under
    high element density. A high spread with a low fixation estimate is therefore
    internally inconsistent.

Rule 2 — Concentrated saliency vs. high cognitive load:
    When saliency is concentrated (clear attentional guidance), cognitive load
    should be reduced because the visual system is efficiently directed to the
    relevant region. Das et al. (2024, ETRA, doi:10.1145/3655610) show
    empirically that dynamic highlighting — which concentrates saliency —
    maintains AOI hit rate at ~69% even under high cognitive load (vs. 6.9%
    without highlighting). Tuch et al. (2009, IJHCS 67:703-715) document the
    same effect via visual hierarchy: a well-defined hierarchy reduces perceived
    visual complexity and cognitive load.

Rule 3 — Visual search time vs. cognitive load:
    Predicted visual search time (Jokinen et al., 2020) is a direct
    operationalization of the effort required to locate interface elements. High
    search time implies high attentional demand, which is a primary component of
    cognitive load (Hart & Staveland, 1988, NASA-TLX). A high search time
    coexisting with a low load index is therefore logically inconsistent.

References:
    Das, A., Wu, Z., Skrjanec, I., & Feit, A. M. (2024). Shifting Focus with
        HCEye. Proc. ACM ETRA. https://doi.org/10.1145/3655610
    Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX. In
        Human Mental Workload (pp. 139-183). North-Holland.
    Jokinen, J. P. P., et al. (2020). Adaptive feature guidance. IJHCS, 136,
        102376. https://doi.org/10.1016/j.ijhcs.2019.102376
    Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter.
        Journal of Vision, 7(2), 17. https://doi.org/10.1167/7.2.17
    Tuch, A. N., et al. (2009). Visual complexity of websites. IJHCS, 67(9),
        703-715. https://doi.org/10.1016/j.ijhcs.2009.04.002
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────
# Rule 1: saliency spread vs. fixation estimate
# "High spread" threshold derived from the s₃ (saliency_spread) feature range
# observed across 150 HCEye webpages. Values > 0.55 place a screen in the upper
# quartile of attentional dispersion — Rosenholtz et al. (2007) associate this
# range with measurable clutter effects.
SPREAD_HIGH_THRESHOLD = 0.55

# "Low fixation count" threshold: Jokinen et al. (2020) show that visual search
# on typical web GUIs requires 8–20 fixations; below 6 implies implausibly
# efficient search given high attentional competition.
FIXATION_COUNT_LOW_THRESHOLD = 6.0

# Rule 2: concentrated saliency vs. high load
# "Concentrated" = spread below the 25th percentile of HCEye distribution.
SPREAD_LOW_THRESHOLD = 0.25

# "High load" cutoff: score ≥ 60/100 places the screen in the high-load
# category, consistent with the interpretation thresholds in the UI.
LOAD_HIGH_THRESHOLD = 60.0

# Rule 3: search time vs. load
# Jokinen et al. (2020) report mean search times of 1.2–3.5 s for standard GUIs.
# Times > 4.0 s indicate a difficult layout; combined with a low load score
# (< 35) this is inconsistent.
SEARCH_TIME_HIGH_THRESHOLD = 4.0  # seconds
LOAD_LOW_THRESHOLD = 35.0


def run_coherence_check(
    saliency_spread: Optional[float],
    estimated_fixation_count: Optional[float],
    mean_search_time_s: Optional[float],
    cognitive_load_score: float,
) -> Dict:
    """
    Run all three coherence rules and return a structured result.

    Args:
        saliency_spread:         s₃ from saliency feature vector (0–1).
                                 None if saliency was not computed.
        estimated_fixation_count: Mean fixation count from Jokinen model.
                                 None if search-time endpoint was not called.
        mean_search_time_s:      Mean predicted search time in seconds.
                                 None if search-time endpoint was not called.
        cognitive_load_score:    Final adjusted cognitive load score (0–100).

    Returns:
        Dict with keys:
            is_coherent (bool)
            flags       (list of str  — machine-readable flag names)
            warnings    (list of str  — human-readable explanations)
            rules_checked (int        — how many rules could be evaluated)
    """
    flags: List[str] = []
    warnings: List[str] = []
    rules_checked = 0

    # ── Rule 1: Saliency spread vs. fixation count ───────────────────────────
    # Basis: Rosenholtz et al. (2007); Jokinen et al. (2020)
    if saliency_spread is not None and estimated_fixation_count is not None:
        rules_checked += 1
        if (saliency_spread > SPREAD_HIGH_THRESHOLD
                and estimated_fixation_count < FIXATION_COUNT_LOW_THRESHOLD):
            flags.append("saliency_fixation_mismatch")
            warnings.append(
                f"Saliency spread is high ({saliency_spread:.2f} > {SPREAD_HIGH_THRESHOLD}) "
                f"but estimated fixation count is low ({estimated_fixation_count:.1f} < "
                f"{FIXATION_COUNT_LOW_THRESHOLD}). High attentional dispersion should "
                f"require more fixations to resolve competing regions "
                f"(Rosenholtz et al., 2007; Jokinen et al., 2020)."
            )

    # ── Rule 2: Concentrated saliency vs. high load ──────────────────────────
    # Basis: Das et al. (2024, HCEye); Tuch et al. (2009)
    if saliency_spread is not None:
        rules_checked += 1
        if (saliency_spread < SPREAD_LOW_THRESHOLD
                and cognitive_load_score >= LOAD_HIGH_THRESHOLD):
            flags.append("concentrated_saliency_high_load")
            warnings.append(
                f"Saliency is highly concentrated (spread {saliency_spread:.2f} < "
                f"{SPREAD_LOW_THRESHOLD}) yet the cognitive load score is high "
                f"({cognitive_load_score:.1f} ≥ {LOAD_HIGH_THRESHOLD}). Clear "
                f"attentional guidance typically reduces cognitive load — HCEye data "
                f"show dynamic highlighting maintains ~69% AOI hit rate even under "
                f"load (Das et al., 2024). Visual features may be driving the load "
                f"estimate independently of the saliency signal."
            )

    # ── Rule 3: Search time vs. load ─────────────────────────────────────────
    # Basis: Jokinen et al. (2020); Hart & Staveland (1988, NASA-TLX)
    if mean_search_time_s is not None:
        rules_checked += 1
        if (mean_search_time_s > SEARCH_TIME_HIGH_THRESHOLD
                and cognitive_load_score < LOAD_LOW_THRESHOLD):
            flags.append("search_time_load_mismatch")
            warnings.append(
                f"Mean predicted search time is high ({mean_search_time_s:.1f} s > "
                f"{SEARCH_TIME_HIGH_THRESHOLD} s) but cognitive load score is low "
                f"({cognitive_load_score:.1f} < {LOAD_LOW_THRESHOLD}). High visual "
                f"search effort is a primary driver of cognitive load "
                f"(Jokinen et al., 2020; Hart & Staveland, 1988). These outputs "
                f"are inconsistent."
            )

    return {
        "is_coherent": len(flags) == 0,
        "flags": flags,
        "warnings": warnings,
        "rules_checked": rules_checked,
        "thresholds": {
            "spread_high": SPREAD_HIGH_THRESHOLD,
            "spread_low": SPREAD_LOW_THRESHOLD,
            "fixation_count_low": FIXATION_COUNT_LOW_THRESHOLD,
            "load_high": LOAD_HIGH_THRESHOLD,
            "load_low": LOAD_LOW_THRESHOLD,
            "search_time_high_s": SEARCH_TIME_HIGH_THRESHOLD,
        },
    }
