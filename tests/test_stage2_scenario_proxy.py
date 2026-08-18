from __future__ import annotations

import math
from pathlib import Path

import pytest

from stage2.coherence_check import run_cross_signal_review
from stage2.scenario_proxy import (
    SCENARIO_PROXY_VERSION,
    TASK_CATEGORIES,
    TASK_CONTEXT_COMPATIBILITY,
    TIME_PRESSURE_DIRECTIONS,
    TIME_PRESSURE_LEVELS,
    ScenarioProxyInputError,
    build_scenario_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "stage1" / "app.py").read_text(encoding="utf-8")
UI_SOURCE = (ROOT / "stage1" / "ui" / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("task_type", TASK_CATEGORIES)
@pytest.mark.parametrize("time_pressure,direction", TIME_PRESSURE_DIRECTIONS.items())
def test_all_fifteen_scenarios_are_deterministic_and_non_score_bearing(
    task_type, time_pressure, direction
):
    first = build_scenario_proxy(task_type, time_pressure)
    second = build_scenario_proxy(task_type, time_pressure)

    assert first == second
    assert first["version"] == SCENARIO_PROXY_VERSION
    assert first["task_type"] == task_type
    assert first["direction"] == direction
    assert first["score_bearing"] is False
    assert first["numeric_modifier"] is None
    assert first["simulated_process"] is False
    assert first["validated_measurement"] is False
    assert first["empirical_effect_size"] is False


def test_task_category_is_a_label_and_cannot_change_direction():
    for time_pressure, expected_direction in TIME_PRESSURE_DIRECTIONS.items():
        directions = {
            build_scenario_proxy(task_type, time_pressure)["direction"]
            for task_type in TASK_CATEGORIES
        }
        assert directions == {expected_direction}


def test_compatibility_matrix_is_explicit_and_uniform():
    assert set(TASK_CONTEXT_COMPATIBILITY) == set(TASK_CATEGORIES)
    for entry in TASK_CONTEXT_COMPATIBILITY.values():
        assert entry == {
            "supported_fields": ("task_type", "time_pressure"),
            "task_type_role": "scenario_label_only",
            "time_pressure_role": "ordinal_directional_hypothesis",
        }


@pytest.mark.parametrize(
    "task_type,time_pressure",
    [
        ("", "medium"),
        ("SEARCH", "medium"),
        ("driving", "medium"),
        ("search", ""),
        ("search", "urgent"),
        (None, "medium"),
        ("search", None),
    ],
)
def test_invalid_scenario_fields_fail_closed(task_type, time_pressure):
    with pytest.raises(ScenarioProxyInputError):
        build_scenario_proxy(task_type, time_pressure)


def test_cross_signal_review_tri_state_and_non_score_bearing_contract():
    not_evaluable = run_cross_signal_review(
        saliency_spread=None,
        estimated_fixation_count=None,
        mean_search_time_s=None,
        layout_proxy_value=None,
    )
    assert not_evaluable["status"] == "not_evaluable"

    no_flag = run_cross_signal_review(
        saliency_spread=0.4,
        estimated_fixation_count=10.0,
        mean_search_time_s=2.0,
        layout_proxy_value=40.0,
    )
    assert no_flag["status"] == "no_review_flag"
    assert no_flag["review_flags"] == []

    flagged = run_cross_signal_review(
        saliency_spread=0.1,
        estimated_fixation_count=10.0,
        mean_search_time_s=5.0,
        layout_proxy_value=70.0,
    )
    assert flagged["status"] == "review_recommended"
    assert flagged["review_flags"]

    for result in (not_evaluable, no_flag, flagged):
        assert result["score_bearing"] is False
        assert result["validated_behavioral_prediction"] is False
        assert result["threshold_status"] == "author_selected_review_triggers_not_calibrated"
        assert set(result) >= {
            "status",
            "review_flags",
            "review_notes",
            "rules_checked",
            "score_bearing",
            "threshold_status",
            "claim_boundary",
        }


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_cross_signal_review_rejects_supplied_nonfinite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        run_cross_signal_review(
            saliency_spread=bad,
            estimated_fixation_count=None,
            mean_search_time_s=None,
            layout_proxy_value=40.0,
        )


def test_active_v1_route_and_ui_exclude_legacy_personality_and_ml_paths():
    for forbidden_import in (
        "from stage2.task_descriptor import",
        "from stage2.user_profile import",
        "from stage2.regression_model import",
    ):
        assert forbidden_import not in APP_SOURCE

    for forbidden_ui_contract in (
        "profilePreset",
        "targetSpecificity",
        "searchMode",
        "use_trained_model",
        "data.big_five_profile",
        "data.context_adjusted_experimental_outputs",
    ):
        assert forbidden_ui_contract not in UI_SOURCE

    for required_ui_contract in (
        "data.stage2_scenario_proxy",
        "data.cross_signal_review",
        "data.jokinen_diagnostic",
        "score_bearing=false",
        "numeric_modifier=null",
    ):
        assert required_ui_contract in UI_SOURCE
