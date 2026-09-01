"""Exploratory Cross-Signal Review for Stage 2 v1.

The review compares outputs produced by separate heuristics and models. Its
author-selected thresholds are inspection triggers only. A result never changes
the Stage-1 index, never contributes to another score and does not validate the
signals against each other or against human behaviour.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


# Author-selected review triggers retained for deterministic diagnostics. They
# are not calibrated cut-offs, source-study effect sizes or validation limits.
SPREAD_HIGH_THRESHOLD = 0.55
FIXATION_COUNT_LOW_THRESHOLD = 6.0
SPREAD_LOW_THRESHOLD = 0.25
LAYOUT_PROXY_HIGH_THRESHOLD = 60.0
SEARCH_TIME_HIGH_THRESHOLD = 4.0
LAYOUT_PROXY_LOW_THRESHOLD = 35.0

TRI_STATE_STATUSES = (
    "not_evaluable",
    "no_review_flag",
    "review_recommended",
)


def _optional_finite(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite real number or None")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def run_cross_signal_review(
    saliency_spread: Optional[float],
    estimated_fixation_count: Optional[float],
    mean_search_time_s: Optional[float],
    layout_proxy_value: Optional[float],
) -> Dict[str, object]:
    """Return a non-score-bearing tri-state review of available signals.

    ``not_evaluable`` means no rule had all of its required inputs.
    ``no_review_flag`` means at least one rule ran and no trigger fired.
    ``review_recommended`` means one or more author-selected triggers fired.
    None of the states means valid/invalid, coherent/incoherent or measured.
    """
    spread = _optional_finite(saliency_spread, "saliency_spread")
    fixation_count = _optional_finite(
        estimated_fixation_count, "estimated_fixation_count"
    )
    search_time = _optional_finite(mean_search_time_s, "mean_search_time_s")
    layout_value = _optional_finite(layout_proxy_value, "layout_proxy_value")

    flags: List[str] = []
    notes: List[str] = []
    rules_checked = 0

    if spread is not None and fixation_count is not None:
        rules_checked += 1
        if (
            spread > SPREAD_HIGH_THRESHOLD
            and fixation_count < FIXATION_COUNT_LOW_THRESHOLD
        ):
            flags.append("saliency_fixation_review")
            notes.append(
                "High model-estimated saliency spread and a low model-simulated "
                "fixation count crossed an author-selected review trigger. Inspect "
                "the two diagnostics separately; this is not measured gaze or a "
                "validated relation."
            )

    if spread is not None and layout_value is not None:
        rules_checked += 1
        if (
            spread < SPREAD_LOW_THRESHOLD
            and layout_value >= LAYOUT_PROXY_HIGH_THRESHOLD
        ):
            flags.append("saliency_layout_review")
            notes.append(
                "Concentrated model-estimated saliency and a high experimental "
                "layout-proxy value crossed an author-selected review trigger. "
                "The signals are separate heuristics, not mutual validation or "
                "cognitive-load evidence."
            )

    if search_time is not None and layout_value is not None:
        rules_checked += 1
        if (
            search_time > SEARCH_TIME_HIGH_THRESHOLD
            and layout_value < LAYOUT_PROXY_LOW_THRESHOLD
        ):
            flags.append("search_layout_review")
            notes.append(
                "High model-simulated search time and a low experimental layout-"
                "proxy value crossed an author-selected review trigger. Inspect "
                "the outputs separately; no behavioral or causal relation is "
                "asserted."
            )

    if rules_checked == 0:
        status = "not_evaluable"
    elif flags:
        status = "review_recommended"
    else:
        status = "no_review_flag"

    return {
        "status": status,
        "review_flags": flags,
        "review_notes": notes,
        "rules_checked": rules_checked,
        "score_bearing": False,
        "validated_behavioral_prediction": False,
        "threshold_status": "author_selected_review_triggers_not_calibrated",
        "claim_boundary": (
            "Exploratory Cross-Signal Review only. Tri-state output is an "
            "inspection cue, not validation, measurement, or evidence of "
            "cognitive load or human behavior."
        ),
        "thresholds": {
            "saliency_spread_high": SPREAD_HIGH_THRESHOLD,
            "saliency_spread_low": SPREAD_LOW_THRESHOLD,
            "fixation_count_low": FIXATION_COUNT_LOW_THRESHOLD,
            "layout_proxy_high": LAYOUT_PROXY_HIGH_THRESHOLD,
            "layout_proxy_low": LAYOUT_PROXY_LOW_THRESHOLD,
            "search_time_high_s": SEARCH_TIME_HIGH_THRESHOLD,
        },
    }
