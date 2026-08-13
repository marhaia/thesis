"""P7 / AG-12 regression tests for public claim reconciliation."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "stage1" / "ui" / "index.html").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYTEST_CONFIG = (ROOT / "pytest.ini").read_text(encoding="utf-8")
CITATION_MATRIX = (
    ROOT / "Literature" / "notes" / "media" / "citation_matrix.html"
).read_text(encoding="utf-8")
HISTORICAL_PIPELINE_DEMO = (ROOT / "tests" / "test_pipeline.py").read_text(
    encoding="utf-8"
)
TEST_CONFTEST = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
STAGE2_REGRESSION = (ROOT / "stage2" / "regression_model.py").read_text(
    encoding="utf-8"
)


def _ui_slice(start: str, end: str) -> str:
    return UI[UI.index(start):UI.index(end, UI.index(start))]


def test_ui_headline_and_history_use_task_independent_stage1_layout_value():
    assert "${layout.experimental_complexity_index.toFixed(1)}" in UI
    assert (
        "experimental_layout_complexity: "
        "data.layout.experimental_complexity_index"
    ) in UI
    assert (
        "context_adjusted_experimental_output: "
        "data.context_adjusted_experimental_outputs.layout_complexity_score"
    ) in UI


def test_ui_declares_exact_x19_and_keeps_context_outside_boundary():
    assert "x = [v, s, h] ∈ ℝ¹⁹" in UI
    assert "x = [v, s, h, t, p] ∈ ℝ³⁰" not in UI
    assert "They never enter or alter x19" in UI


def test_ui_and_csv_labels_do_not_mislabel_context_adjusted_output():
    assert "stage1_experimental_layout_complexity_index" in UI
    assert "context_adjusted_experimental_output" in UI
    assert 'accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff"' in UI
    assert "image/webp" not in UI


def test_umsi_heatmap_is_presented_only_as_relative_normalized_activation():
    panel = _ui_slice('<div id="saliencyPanel"', '<div id="radarPanel"')

    assert "Relative Model-Estimated Saliency Map" in panel
    assert "within-image min–max normalized model activation on [0,1]" in panel
    assert "not observed gaze, a calibrated probability, or a timing prediction" in panel
    assert "Low relative activation" in panel
    assert "High relative activation" in panel
    assert "not gaze probability or viewing order" in panel
    for forbidden in (
        "predicted probability",
        "fixation probability",
        "first few seconds",
        "rarely looked at",
        "looked at first",
    ):
        assert forbidden not in panel.lower()

    method_node = _ui_slice(
        '<div class="pipeline-node-title">Saliency Features</div>',
        '<div class="pipeline-node-title">HCEye-Derived Proxies</div>',
    )
    assert "relative model-estimated saliency map" in method_node
    assert "not observed gaze, calibrated probabilities, or timing predictions" in method_node


def test_screenshot_hceye_proxies_are_never_converted_to_effect_percentages():
    interpretation = _ui_slice(
        "function renderInterpretation(data)",
        "function renderRadarInterpretation(vf)",
    )

    assert "unitless, project-specific screenshot proxy" in interpretation
    assert "not a predicted fixation count" in interpretation
    assert "not a predicted or observed fixation duration" in interpretation
    assert "not measured AOI retention or a percentage uplift" in interpretation
    assert "Screenshot proxy values are not source-study effect estimates" in interpretation
    for forbidden in (
        "const fixPct",
        "const durPct",
        "% fewer fixations",
        "% longer",
        "Math.round(hlEff*100)",
        "higher AOI retention",
    ):
        assert forbidden not in interpretation


def test_hceye_source_study_effect_is_explicitly_separate_from_screen_proxies():
    evidence = _ui_slice(
        '<div class="pipeline-node-title">HCEye Aggregate-Ratio Reproduction</div>',
        '<div class="pipeline-node-title">UEyes Sample Sanity Check</div>',
    )

    assert "Source-study context only" in evidence
    assert "separate from the screenshot proxy values" in evidence
    assert "not independent validation of the screenshot-level index" in evidence


def test_active_pdf_labels_keep_umsi_and_hceye_claim_boundaries():
    export = _ui_slice("async function exportPDF(d)", "// ─── CSV Export")

    assert "within-image normalized; not gaze probability" in export
    assert "HCEye-derived Rule Index (unitless project-specific proxy)" in export
    assert "HCEye-Derived Unitless Proxies (h, 6 dims)" in export


def test_readme_uses_bounded_construct_and_current_test_contract():
    lower = README.lower()
    compact = " ".join(lower.split())
    assert "exploratory, project-specific layout-complexity index" in compact
    assert "not a validated cognitive-load measurement" in compact
    assert "[v8 | s5 | h6]" in README
    assert "pytest -q" in README
    assert "deployed load score" not in lower
    assert "tests/test_claim_reconciliation.py" in PYTEST_CONFIG


def test_stage2_scaffold_never_describes_circular_targets_as_ground_truth():
    lower = STAGE2_REGRESSION.lower()
    assert "experimental multi-output regression scaffold" in lower
    assert "not screenshot-level ground truth" in lower
    assert "deployed cognitive-load score" not in lower
    assert "as ground truth for how cognitive load affects" not in lower
    assert "y = ground-truth cognitive load effects" not in lower
    assert "predicts cognitive load indices" not in lower


def test_target_scanpath_prototype_is_outside_basic_stage1_ui_and_contract():
    compact_readme = " ".join(README.split())
    assert (
        'id="targetScanpathPanel" class="result-section" hidden '
        'aria-hidden="true"'
    ) in UI
    assert "renderTargetScanpath(data.detected_elements)" not in UI
    assert "FUTURE WORK ONLY" in UI
    assert "Stage-1 acceptance boundary" in README
    assert "are not rendered by the standard Stage-1 UI" in compact_readme


def test_latest_expose_is_self_identified_as_historical():
    path = ROOT / "Literature" / "notes" / "expose" / "expose_final_v16_24jun.docx"
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "HISTORICAL WORKING DRAFT" in xml
    assert "not the current methodology or claim basis" in xml


def test_citation_matrix_is_prominently_self_identified_as_superseded():
    assert '<meta name="robots" content="noindex,nofollow">' in CITATION_MATRIX
    assert '<title>HISTORICAL / SUPERSEDED — Citation Matrix</title>' in CITATION_MATRIX
    assert '<body data-artifact-status="historical-superseded">' in CITATION_MATRIX
    assert 'id="artifactStatus" role="alert"' in CITATION_MATRIX
    assert "Historical / Superseded Research Note" in CITATION_MATRIX
    assert "must not be cited as current claims" in CITATION_MATRIX

    banner_end = CITATION_MATRIX.index("</aside>")
    interactive_header = CITATION_MATRIX.index("<header>")
    legacy_claim = CITATION_MATRIX.index(
        "the Cognitive Load Index quantifies how far a GUI exhausts"
    )
    assert banner_end < interactive_header < legacy_claim


def test_citation_matrix_banner_states_current_claim_boundaries_and_authority():
    banner = CITATION_MATRIX[
        CITATION_MATRIX.index('<aside class="artifact-status"'):
        CITATION_MATRIX.index("</aside>")
    ]

    assert "not the current Stage-1 methodology" in banner
    assert "exploratory," in banner
    assert "project-specific layout-complexity index" in banner
    assert "not a validated cognitive-load" in banner
    assert "relative within-image normalized model activation" in banner
    assert "not observed gaze or calibrated fixation probability" in banner
    assert "unitless screenshot proxies" in banner
    assert "not source-study effect estimates" in banner
    assert 'href="../../../README.md"' in banner
    assert 'href="../../../stage1/DOCUMENTATION.md"' in banner


def test_retired_pipeline_demo_cannot_present_placeholder_values_as_stage1_x19():
    assert "HISTORICAL / INCOMPLETE DEMO" in HISTORICAL_PIPELINE_DEMO
    assert "not the current Stage-1 contract" in HISTORICAL_PIPELINE_DEMO
    assert "does not run UMSI++" in HISTORICAL_PIPELINE_DEMO
    assert "five zeros are placeholders, not saliency features" in HISTORICAL_PIPELINE_DEMO
    assert "NOT Stage-1 x19" in HISTORICAL_PIPELINE_DEMO
    assert "not Stage-1 measurements/evidence" in HISTORICAL_PIPELINE_DEMO

    for retired_unqualified_output in (
        "FULL PIPELINE TEST",
        "Full vector:",
        "Cognitive Load Score:",
        "Overall Cognitive Load:",
        "Raw CLI",
        "CLI=",
    ):
        assert retired_unqualified_output not in HISTORICAL_PIPELINE_DEMO


def test_readme_and_pytest_collection_identify_pipeline_demo_as_historical_only():
    assert "**historical, incomplete development demo**" in README
    assert "five explicit zero" in README
    assert "**not Stage-1 x19**" in README
    assert "not current pipeline or" in README
    assert "validation evidence" in README

    assert "HISTORICAL / INCOMPLETE demo" in TEST_CONFTEST
    assert "not an automated" in TEST_CONFTEST
    assert "current pipeline evidence" in TEST_CONFTEST
    assert 'collect_ignore = ["test_pipeline.py"]' in TEST_CONFTEST
