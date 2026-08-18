"""Stage-2 v1 deterministic scenario proxy.

The proxy adds a declared scenario label to the frozen Stage-1 screenshot
result. It never changes x19 or the Stage-1 layout index and intentionally has
no numeric modifier. Task categories are labels, not simulated processes. The
only directional hypothesis is the ordinal, author-declared time-pressure
context; it is not an empirical effect size or measurement.
"""

from __future__ import annotations

from typing import Dict


SCENARIO_PROXY_VERSION = "stage2-scenario-proxy-v1"

TASK_CATEGORIES = {
    "navigation": "Navigation",
    "search": "Search",
    "monitoring": "Monitoring",
    "data_entry": "Data Entry",
    "decision": "Decision",
}

TIME_PRESSURE_LEVELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

TIME_PRESSURE_DIRECTIONS = {
    "low": "lower",
    "medium": "baseline",
    "high": "higher",
}

# Freeze-time compatibility matrix. Every v1 task category accepts exactly the
# same two fields; task_type is a label only and time_pressure supplies the
# qualitative direction. No hidden category-specific fields or effects exist.
TASK_CONTEXT_COMPATIBILITY = {
    task_type: {
        "supported_fields": ("task_type", "time_pressure"),
        "task_type_role": "scenario_label_only",
        "time_pressure_role": "ordinal_directional_hypothesis",
    }
    for task_type in TASK_CATEGORIES
}


class ScenarioProxyInputError(ValueError):
    """Raised when a Stage-2 v1 scenario field is missing or unsupported."""


def build_scenario_proxy(task_type: str, time_pressure: str) -> Dict[str, object]:
    """Return the deterministic, non-score-bearing Stage-2 v1 proxy."""
    if not isinstance(task_type, str) or task_type not in TASK_CATEGORIES:
        raise ScenarioProxyInputError(
            f"task_type must be one of {tuple(TASK_CATEGORIES)}"
        )
    if not isinstance(time_pressure, str) or time_pressure not in TIME_PRESSURE_LEVELS:
        raise ScenarioProxyInputError(
            f"time_pressure must be one of {tuple(TIME_PRESSURE_LEVELS)}"
        )

    compatibility = TASK_CONTEXT_COMPATIBILITY[task_type]
    return {
        "version": SCENARIO_PROXY_VERSION,
        "task_type": task_type,
        "task_label": TASK_CATEGORIES[task_type],
        "time_pressure": time_pressure,
        "time_pressure_label": TIME_PRESSURE_LEVELS[time_pressure],
        "direction": TIME_PRESSURE_DIRECTIONS[time_pressure],
        "compatibility": {
            "compatible": True,
            "supported_fields": list(compatibility["supported_fields"]),
            "task_type_role": compatibility["task_type_role"],
            "time_pressure_role": compatibility["time_pressure_role"],
        },
        "score_bearing": False,
        "numeric_modifier": None,
        "simulated_process": False,
        "validated_measurement": False,
        "empirical_effect_size": False,
        "direction_basis": (
            "Pre-specified ordinal hypothesis from the declared time-pressure "
            "label only; no magnitude or category-specific task effect is asserted."
        ),
        "claim_boundary": (
            "Deterministic scenario proxy only; not a measurement, calibrated "
            "effect, simulated task process, or validated behavioral prediction."
        ),
    }
