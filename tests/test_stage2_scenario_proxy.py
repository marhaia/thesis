from __future__ import annotations

import json
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
PIPELINE_FIGURE_SOURCE = (
    ROOT / "scripts" / "generate_pipeline_figure.py"
).read_text(encoding="utf-8")
STAGE1_DOCUMENTATION = (ROOT / "stage1" / "DOCUMENTATION.md").read_text(
    encoding="utf-8"
)
STAGE2_DOCUMENTATION = (ROOT / "stage2" / "README.md").read_text(
    encoding="utf-8"
)
ENDPOINT_SCALE_TOOL = (
    ROOT / "stage1" / "tools" / "endpoint_scale_matrix.py"
).read_text(encoding="utf-8")
ENDPOINT_SCALE_REPORT = json.loads(
    (ROOT / "stage1" / "canonical_eval" / "endpoint_scale_report.json").read_text(
        encoding="utf-8"
    )
)
CANONICAL_SCALE_TOOL = (
    ROOT / "stage1" / "tools" / "canonical_scale_eval.py"
).read_text(encoding="utf-8")
CANONICAL_SCALE_DOCUMENTATION = (
    ROOT / "stage1" / "canonical_resolution_evaluation.md"
).read_text(encoding="utf-8")


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


def test_time_pressure_literature_is_motivation_not_calibration():
    compact = " ".join(STAGE2_DOCUMENTATION.split())
    assert "pre-specified ordinal project hypothesis" in compact
    assert "10.1016/j.ipm.2019.04.004" in compact
    assert "does not calibrate" in compact
    assert "not evidence for a universal" in compact


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


def test_pipeline_figure_generator_matches_the_active_stage2_v1_boundary():
    from PIL import Image

    for required_claim in (
        "Current Technical Pipeline — Stage 1 + Stage 2 v1",
        "Deterministic scenario proxy",
        "score_bearing=false · numeric_modifier=null",
        "Optional Jokinen diagnostic",
        "Exploratory Cross-Signal Review",
        "No personality input · no trained regressor · no numeric task modifier",
    ):
        assert required_claim in PIPELINE_FIGURE_SOURCE

    for retired_claim in (
        "Big Five (optional)",
        "Task-Conditioned Multi-Output Prediction",
        "Cognitive Load Index 0–100",
        "Corr. NASA-TLX",
        "jointly optimised",
    ):
        assert retired_claim not in PIPELINE_FIGURE_SOURCE

    image_path = ROOT / "scripts" / "pipeline_figure.png"
    assert image_path.is_file()
    with Image.open(image_path) as image:
        assert image.info["Title"] == (
            "Current Technical Pipeline — Stage 1 + Stage 2 v1"
        )
        assert "qualitative, non-score-bearing Stage 2 v1" in image.info[
            "Description"
        ]


def test_authoritative_stage1_documentation_matches_the_active_v1_contract():
    current_policy = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("> **Current P7 claim policy"):
        STAGE1_DOCUMENTATION.index("---")
    ]
    overview = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("## 1. Overview"):
        STAGE1_DOCUMENTATION.index("```", STAGE1_DOCUMENTATION.index("## 1. Overview"))
    ]
    active_copy = current_policy + overview

    for required in (
        "deterministic qualitative direction",
        "non-score-bearing",
        "has no numeric modifier",
        "Personality inputs and trained regressors are excluded",
        "score_bearing=false",
        "numeric_modifier=null",
    ):
        assert required in active_copy

    for retired_claim in (
        "Task/profile modifiers remain",
        "optional coarse Big-Five preset",
        "context-adjusted experimental outputs",
    ):
        assert retired_claim not in active_copy


def test_legacy_endpoint_scale_tool_and_export_are_unmistakably_inactive():
    assert ENDPOINT_SCALE_TOOL.startswith('"""ARCHIVED LEGACY TOOL')
    assert "must not be executed or cited as current pipeline evidence" in (
        ENDPOINT_SCALE_TOOL
    )
    assert "return 2" in ENDPOINT_SCALE_TOOL
    for removed_active_schema in (
        'body["cognitive_load_index"]',
        'body["task_descriptor"]',
        'body["big_five_profile"]',
        "fallback_neutral",
        "TRUE endpoint scale matrix",
    ):
        assert removed_active_schema not in ENDPOINT_SCALE_TOOL

    assert ENDPOINT_SCALE_REPORT == {
        "status": "historical_superseded_do_not_use",
        "active_pipeline_evidence": False,
        "claim_boundary": (
            "This legacy endpoint-scale export depended on retired fail-open "
            "saliency, neutral OCR fallbacks, task/profile modifiers, and "
            "response fields that are not part of the frozen Stage-1 plus "
            "Stage-2-v1 contract."
        ),
        "replacement_evidence": [
            "maintained canonical-scale regression tests",
            "real-weight Stage-1 and Stage-2-v1 acceptance replay",
        ],
        "legacy_measurements_removed": True,
    }


def test_active_scale_evidence_uses_the_current_layout_output_name():
    for active_scale_surface in (
        CANONICAL_SCALE_TOOL,
        CANONICAL_SCALE_DOCUMENTATION,
    ):
        assert "`cognitive_load_index`" not in active_scale_surface
        assert "layout.experimental_complexity_index" in active_scale_surface
