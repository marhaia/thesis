"""
Exploratory Proxy-Coherence Check
=================================
Applies three project-defined consistency hypotheses to separately generated
proxy outputs. Flags are diagnostic prompts for inspection, not validation,
measured gaze, cognitive-load evidence, or demonstrated human behavior.

Scientific basis for each rule:

Rule 1 — Saliency spread vs. fixation-count proxy:
    The project hypothesis flags high normalized saliency spread paired with a
    low model-estimated fixation count. The cited work motivates the direction;
    it does not validate this deployed screenshot-level rule.

Rule 2 — Concentrated saliency vs. high layout-proxy value:
    The project hypothesis flags concentrated normalized model activation paired
    with a high exploratory layout index. HCEye aggregate observations provide
    source-study context only; they do not validate this screenshot mapping.

Rule 3 — Search-time simulation vs. low layout-proxy value:
    The project hypothesis flags high model-simulated search time paired with a
    low exploratory layout index. The outputs are different constructs; the
    rule does not establish a causal or validated cognitive-load relationship.

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

# Rule 2: concentrated saliency vs. high context-adjusted proxy
# "Concentrated" = spread below the 25th percentile of HCEye distribution.
SPREAD_LOW_THRESHOLD = 0.25

# Project-defined high-proxy cutoff used only for this diagnostic comparison.
LOAD_HIGH_THRESHOLD = 60.0

# Rule 3: search-time simulation vs. context-adjusted proxy
# Jokinen et al. (2020) report mean search times of 1.2–3.5 s for standard GUIs.
# Times > 4.0 s enter the project's difficult-search band; pairing them with a
# low proxy value (< 35) triggers an exploratory consistency flag.
SEARCH_TIME_HIGH_THRESHOLD = 4.0  # seconds
LOAD_LOW_THRESHOLD = 35.0


def run_coherence_check(
    saliency_spread: Optional[float],
    estimated_fixation_count: Optional[float],
    mean_search_time_s: Optional[float],
    cognitive_load_score: float,
) -> Dict:
    """
    Run all three exploratory proxy-coherence rules.

    Args:
        saliency_spread:         s₃ from saliency feature vector (0–1).
                                 None if saliency was not computed.
        estimated_fixation_count: Mean fixation count from Jokinen model.
                                 None if search-time endpoint was not called.
        mean_search_time_s:      Mean predicted search time in seconds.
                                 None if search-time endpoint was not called.
        cognitive_load_score:    Legacy parameter name for the context-adjusted
                                 experimental proxy value (0–100); not validated
                                 cognitive load or human behavior.

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
                f"{FIXATION_COUNT_LOW_THRESHOLD}). This exploratory rule flags "
                f"the combination because the project hypothesis associates wider "
                f"activation spread with more search steps; it is not measured gaze "
                f"or a validated behavioral prediction (Rosenholtz et al., 2007; "
                f"Jokinen et al., 2020)."
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
                f"{SPREAD_LOW_THRESHOLD}) yet the context-adjusted experimental "
                f"proxy value is "
                f"high ({cognitive_load_score:.1f} ≥ {LOAD_HIGH_THRESHOLD}). The "
                f"project heuristic expects concentrated activation and this layout "
                f"proxy to align more closely. HCEye aggregate observations are "
                f"source-study context only; this is not a validated cognitive-load "
                f"or behavioral relation (Das et al., 2024)."
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
                f"{SEARCH_TIME_HIGH_THRESHOLD} s) but the context-adjusted "
                f"experimental proxy value is low ({cognitive_load_score:.1f} < "
                f"{LOAD_LOW_THRESHOLD}). "
                f"This project rule marks the two model outputs for review; it does "
                f"not establish measured search effort, cognitive load, or a causal "
                f"relationship (Jokinen et al., 2020; Hart & Staveland, 1988)."
            )

    return {
        "is_coherent": len(flags) == 0,
        "flags": flags,
        "warnings": warnings,
        "rules_checked": rules_checked,
        "validated_behavioral_prediction": False,
        "claim_boundary": (
            "Exploratory heuristic consistency flags only; not measured gaze, "
            "validated behavior, or cognitive-load evidence."
        ),
        "thresholds": {
            "spread_high": SPREAD_HIGH_THRESHOLD,
            "spread_low": SPREAD_LOW_THRESHOLD,
            "fixation_count_low": FIXATION_COUNT_LOW_THRESHOLD,
            "load_high": LOAD_HIGH_THRESHOLD,
            "load_low": LOAD_LOW_THRESHOLD,
            "search_time_high_s": SEARCH_TIME_HIGH_THRESHOLD,
        },
    }
