"""P7 / AG-12 regression tests for public claim reconciliation."""

from __future__ import annotations

from pathlib import Path
import subprocess
from xml.etree import ElementTree
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
COHERENCE = (ROOT / "stage2" / "coherence_check.py").read_text(encoding="utf-8")
STAGE1_DOCUMENTATION = (ROOT / "stage1" / "DOCUMENTATION.md").read_text(
    encoding="utf-8"
)
PLANNING_STATUS_HTML = {
    path.name: path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "planning" / "status").glob("*.html"))
}
CLAIM_ARTIFACT_ROOTS = (
    ROOT / "Audit",
    ROOT / "Literature" / "notes",
    ROOT / "planning",
)
REQUIRED_HISTORICAL_CLAIM_ARTIFACTS = (
    ROOT / "Audit" / "audit_prompt.md",
    ROOT / "Audit" / "audit_prompt_v2.md",
    ROOT / "Audit" / "audit_prompt_v2_chatgpt.md",
    ROOT / "Audit" / "audit_prompt_v2_live.md",
    ROOT / "Audit" / "audit_report.md",
    ROOT / "Audit" / "framing_todo.md",
    ROOT / "Audit" / "thesis_audit_report.md",
    ROOT / "Literature" / "notes" / "_projektplan.md",
    ROOT / "Literature" / "notes" / "_gem_context.md",
    ROOT / "Literature" / "notes" / "_overview.md",
    ROOT / "Literature" / "notes" / "_overview_obsidian.md",
    ROOT / "Literature" / "notes" / "_reading_plan.md",
    ROOT / "Literature" / "notes" / "expose" / "README.md",
    ROOT / "Literature" / "notes" / "media" / "presentation_literature.html",
    ROOT / "Literature" / "notes" / "media" / "presentation_research_methodik.html",
    ROOT / "planning" / "README.md",
    ROOT / "planning" / "daily" / "2026-06-03.md",
    ROOT / "planning" / "weekly" / "2026-KW23.md",
)
PLANNING_README = (ROOT / "planning" / "README.md").read_text(encoding="utf-8")


def _ui_slice(start: str, end: str) -> str:
    return UI[UI.index(start):UI.index(end, UI.index(start))]


def _claim_risk_signals(document: str) -> set[str]:
    """Identify current-authority markers and the superseded R8/Stage-2 plan."""
    lower = document.lower()
    signals = set()
    if "status: active" in lower or "**status:** aktiv" in lower:
        signals.add("active-authority")
    if "aktueller entwicklungsstand" in lower:
        signals.add("current-status")
    if "pipeline bereits implementiert" in lower:
        signals.add("implemented-pipeline")
    if "alles läuft" in lower:
        signals.add("all-components-running")
    old_stage1 = any(marker in lower for marker in ("ℝ⁸", "r8", "8d feature", "8d-feature"))
    if old_stage1 and "stage 1" in lower and "stage 2" in lower and "cognitive load index" in lower:
        signals.add("superseded-r8-stage2-architecture")
    return signals


def _assert_prominent_historical_boundary(path: Path, document: str) -> None:
    if path.suffix == ".html":
        assert '<meta name="robots" content="noindex,nofollow">' in document, path
        assert "<title>HISTORICAL / SUPERSEDED" in document, path
        assert '<body data-artifact-status="historical-superseded">' in document, path
        assert 'class="artifact-status" id="artifactStatus" role="alert"' in document, path
        assert "HISTORICAL / SUPERSEDED" in document, path
        banner = document.index('id="artifactStatus"')
        content = min(
            marker for marker in (
                document.find('<div class="slideshow"'),
                document.find('<div class="deck"'),
            ) if marker >= 0
        )
        assert banner < content, path
    else:
        opening = document[:1500].upper()
        assert "HISTORICAL" in opening and "SUPERSEDED" in opening, path


def _tracked_claim_artifacts() -> tuple[Path, ...]:
    """Enumerate the real tracked inventory, with archive-export fallback."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout:
        candidates = [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    else:
        candidates = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]

    suffixes = {".md", ".html", ".docx", ".xlsx"}
    return tuple(
        sorted(
            path
            for path in candidates
            if path.suffix.lower() in suffixes
            and any(path.is_relative_to(directory) for directory in CLAIM_ARTIFACT_ROOTS)
        )
    )


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return " ".join(text for text in root.itertext() if text)


def _xlsx_sheet_names_and_text(path: Path) -> tuple[list[str], str]:
    with ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        names = [element.attrib["name"] for element in workbook.iter() if element.tag.endswith("}sheet")]
        text_parts = []
        for name in archive.namelist():
            if name == "xl/sharedStrings.xml" or name.startswith("xl/worksheets/sheet"):
                root = ElementTree.fromstring(archive.read(name))
                text_parts.extend(text for text in root.itertext() if text)
    return names, " ".join(text_parts)


def test_repository_wide_claim_inventory_requires_historical_boundaries():
    inventory = _tracked_claim_artifacts()
    assert inventory
    assert any(path.parent == ROOT / "Audit" for path in inventory)
    assert any(path.suffix == ".docx" for path in inventory)
    assert any(path.suffix == ".xlsx" for path in inventory)

    discovered = set()
    for path in inventory:
        if path.suffix.lower() not in {".md", ".html"}:
            continue
        document = path.read_text(encoding="utf-8", errors="replace")
        if _claim_risk_signals(document):
            discovered.add(path)
            _assert_prominent_historical_boundary(path, document)

    for path in REQUIRED_HISTORICAL_CLAIM_ARTIFACTS:
        document = path.read_text(encoding="utf-8")
        _assert_prominent_historical_boundary(path, document)

    assert set(REQUIRED_HISTORICAL_CLAIM_ARTIFACTS).issubset(set(inventory))


def test_every_tracked_expose_docx_has_a_point_of_use_historical_boundary():
    docx_paths = [
        path for path in _tracked_claim_artifacts()
        if path.parent == ROOT / "Literature" / "notes" / "expose"
        and path.suffix.lower() == ".docx"
    ]
    assert len(docx_paths) == 24
    for path in docx_paths:
        opening = _docx_text(path)[:1500].upper()
        assert "HISTORICAL WORKING DRAFT" in opening, path
        assert "SUPERSEDED" in opening, path
        assert "NOT THE CURRENT METHODOLOGY OR CLAIM BASIS" in opening, path


def test_tracked_literature_workbook_is_historical_at_workbook_and_sheet_level():
    path = ROOT / "Literature" / "notes" / "Literatur_Research.xlsx"
    assert path in _tracked_claim_artifacts()
    names, workbook_text = _xlsx_sheet_names_and_text(path)
    assert "STATUS - READ FIRST" in names
    original_sheets = [name for name in names if name != "STATUS - READ FIRST"]
    assert len(original_sheets) == 8
    assert all(name.startswith("HIST - ") for name in original_sheets)
    upper = workbook_text.upper()
    assert "HISTORICAL / SUPERSEDED RESEARCH WORKBOOK" in upper
    assert "NOT THE CURRENT STAGE-1 METHODOLOGY" in upper
    assert "DO NOT CITE THEM AS CURRENT EVIDENCE" in upper


def test_superseded_claim_artifacts_do_not_present_current_authority_or_validation():
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in REQUIRED_HISTORICAL_CLAIM_ARTIFACTS
    ).lower()
    for forbidden in (
        "**status:** aktiv — zentrale architektur-referenz",
        "status: active",
        "must treat these findings as ground truth",
        "pipeline bereits implementiert",
        "die kopplung dieser outputs ist nicht nur theoretisch sinnvoll — sie ist empirisch belegt",
        "stage 1, stage 2, umsi++, jokinen afg, hceye-features, flask api</strong> — alles läuft",
    ):
        assert forbidden not in combined

    assert "historical context only — do not use as current project instructions" in combined
    assert "nicht als stage 2 akzeptiert oder validiert" in combined
    assert "aktueller entwicklungsstand" not in PLANNING_README.lower()


def test_ui_headline_and_history_use_task_independent_stage1_layout_value():
    assert "${layout.experimental_complexity_index.toFixed(1)}" in UI
    assert (
        "experimental_layout_complexity: "
        "data.layout.experimental_complexity_index"
    ) in UI
    assert "scenario_direction: data.stage2_scenario_proxy.direction" in UI
    assert "scenario_score_bearing: data.stage2_scenario_proxy.score_bearing" in UI
    assert "context_adjusted_experimental_output" not in UI


def test_ui_declares_exact_x19_and_keeps_context_outside_boundary():
    assert "x = [v, s, h] ∈ ℝ¹⁹" in UI
    assert "x = [v, s, h, t, p] ∈ ℝ³⁰" not in UI
    assert "never changes either value" in UI


def test_ui_and_csv_labels_expose_only_qualitative_stage2_v1_proxy():
    assert "stage1_experimental_layout_complexity_index" in UI
    assert "stage2_scenario_direction" in UI
    assert "stage2_score_bearing" in UI
    assert "numeric_modifier=null" in UI
    assert "context_adjusted_experimental_output" not in UI
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
    assert "archived legacy research scaffold" in lower
    assert "excluded from stage 2 v1" in lower
    assert "not screenshot-level ground truth" in lower
    assert "deployed cognitive-load score" not in lower
    assert "as ground truth for how cognitive load affects" not in lower
    assert "y = ground-truth cognitive load effects" not in lower
    assert "predicts cognitive load indices" not in lower

    output_schema = STAGE2_REGRESSION[
        STAGE2_REGRESSION.index("OUTPUT_NAMES = ["):
        STAGE2_REGRESSION.index("    ]", STAGE2_REGRESSION.index("OUTPUT_NAMES = ["))
    ]
    required_target_qualifiers = {
        "cognitive_load_score": ("circular", "proxy"),
        "search_efficiency": ("simulated", "proxy"),
        "attention_demand": ("simulated", "proxy"),
    }
    for target, qualifiers in required_target_qualifiers.items():
        target_line = next(
            line for line in output_schema.splitlines() if f"'{target}'" in line
        ).lower()
        assert all(qualifier in target_line for qualifier in qualifiers)
        assert "ground truth" not in target_line


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


def test_design_diagnosis_describes_the_task_independent_heuristic_truthfully():
    diagnosis = _ui_slice("function buildDesignDiagnosis", "// Normalization ranges")
    unconditional_scope = diagnosis[
        diagnosis.index("const bullets = [];"):
        diagnosis.index("// Reconciled summary")
    ]
    for required in (
        "task-independent Design Diagnosis",
        "hand-weighted project heuristic",
        "selected normalized saliency descriptors",
        "measured layout/OCR inputs",
        "not an eye-tracking or behavioral measurement",
        "does not use target-search simulation",
        "is not changed by task or profile selections",
    ):
        assert required in unconditional_scope

    scope_push_index = unconditional_scope.index("bullets.push({")
    scope_push_line_start = unconditional_scope.rfind(
        "\n", 0, scope_push_index
    ) + 1
    scope_push_line_end = unconditional_scope.index("\n", scope_push_index)
    scope_push_line = unconditional_scope[
        scope_push_line_start:scope_push_line_end
    ]
    assert "if (" not in unconditional_scope[:scope_push_index]
    assert scope_push_line.strip() == "bullets.push({"

    assert "Combined feature signal:" in diagnosis
    assert "uncalibrated design heuristic" in diagnosis
    for forbidden in (
        "increase scan path length",
        "interaction patterns",
        "predicted saliency + eye-movement model",
        "shaped by the task context",
    ):
        assert forbidden not in diagnosis


def test_stage2_v1_copy_is_qualitative_and_excludes_profile_and_ml():
    controls = _ui_slice("<!-- Step 2: Task Context -->", "<!-- Analyze action")
    loading = _ui_slice('<div class="loading" id="loading">', '<div class="results"')
    interpretation = _ui_slice(
        "function renderInterpretation(data)",
        "function renderRadarInterpretation(vf)",
    )

    assert "qualitative direction" in controls
    assert "There is no numeric modifier" in controls
    assert "Attaching deterministic qualitative scenario proxy" in loading
    assert "data.stage2_scenario_proxy" in UI
    assert "score_bearing=false" in UI
    assert "numeric_modifier=null" in UI
    assert "Stage-1 x19 vector and layout index remain unchanged" in UI

    for forbidden in (
        "User Profile",
        "Big Five",
        "use_trained_model",
        "context-adjusted simulation",
        "Original context-adjusted output",
        "Simulated context-adjusted output",
        "shift the score to reflect how a user is actually interacting",
        "modifier to the score",
        "Task &amp; profile modifiers → final score",
        "task type raises the index",
        "task type lowers the index",
        "higher adjusted index",
        "Original adjusted score",
        "Simulated adjusted score",
        "simulated adjusted score",
    ):
        assert forbidden not in UI


def test_ocr_text_is_escaped_before_readability_inner_html_rendering():
    readability = _ui_slice(
        "function renderReadabilityReport(report)",
        "function renderTargetScanpath",
    )
    escaping = _ui_slice("function escapeHtml(value)", "function renderHistory")

    assert "const label = escapeHtml((e.text || '').slice(0, 40));" in readability
    assert "const label = (e.text || '').slice(0, 40);" not in readability
    assert "list.innerHTML = elOverviewBlock()" in readability
    for required_replacement in (
        '.replace(/&/g, "&amp;")',
        '.replace(/</g, "&lt;")',
        '.replace(/>/g, "&gt;")',
        '.replace(/"/g, "&quot;")',
        ".replace(/'/g, \"&#39;\")",
    ):
        assert required_replacement in escaping


def test_visible_cross_signal_claims_are_bounded_tri_state_review_cues():
    from stage2.coherence_check import run_cross_signal_review

    result = run_cross_signal_review(
        saliency_spread=0.1,
        estimated_fixation_count=None,
        mean_search_time_s=5.0,
        layout_proxy_value=70.0,
    )
    joined = " ".join(result["review_notes"]).lower()
    assert result["status"] == "review_recommended"
    assert result["score_bearing"] is False
    assert result["validated_behavioral_prediction"] is False
    assert "not mutual validation" in joined
    assert "typically reduces cognitive load" not in joined
    assert "primary driver of cognitive load" not in joined
    assert "exploratory cross-signal review" in UI.lower()
    assert "not calibrated validation" in UI.lower()
    assert "cognitive load should be reduced" not in COHERENCE.lower()


def test_feature_metadata_endpoint_exposes_an_explicit_behavioral_boundary():
    from stage1.app import app

    response = app.test_client().get("/api/features")
    assert response.status_code == 200
    body = response.get_json()
    assert body["validated_behavioral_prediction"] is False
    assert "not validated predictions" in body["claim_boundary"]
    assert [item["key"] for item in body["features"]] == [
        "shannon_entropy",
        "edge_density",
        "feature_congestion",
        "subband_entropy",
        "layout_symmetry",
        "chromatic_coherence",
        "visual_hierarchy",
        "interactive_element_density",
    ]
    descriptions = " ".join(item["description"] for item in body["features"])
    for forbidden in (
        "less visual search needed",
        "less search effort",
        "higher decisional load",
    ):
        assert forbidden not in descriptions.lower()


def test_current_feature_documentation_uses_the_same_hypothesis_boundary():
    lower = STAGE1_DOCUMENTATION.lower()
    assert "unvalidated project hypothesis" in lower
    assert "unvalidated design hypothesis" in lower
    assert "decisional-load implications are unvalidated" in lower
    for forbidden in (
        "more structural boundaries = more parsing effort",
        "more symmetric = less visual search needed",
        "clearer layered structure = less search effort",
        "more action possibilities = higher decisional load",
    ):
        assert forbidden not in lower


def test_readme_documents_both_screen_set_routes_and_shared_limits():
    compact = " ".join(README.split())
    assert "`/api/screen-consistency` and `/api/product-learning`" in compact
    assert "both screen-set endpoints" in compact
    assert "16 px minimum for each dimension" in compact
    assert "| `/api/product-learning` | POST |" in README
    assert "shares the screen-set cumulative limits" in README


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


def test_planning_status_presentations_are_prominently_historical():
    assert PLANNING_STATUS_HTML
    for filename, document in PLANNING_STATUS_HTML.items():
        assert '<meta name="robots" content="noindex,nofollow">' in document
        assert "<title>HISTORICAL / SUPERSEDED" in document
        assert '<body data-artifact-status="historical-superseded">' in document
        assert 'class="artifact-status" id="artifactStatus" role="alert"' in document
        assert "Historical / Superseded" in document
        assert "not the current methodology" in document
        assert "does not ship a validated cognitive-load score" in document

        banner_end = document.index("</aside>")
        content_start = min(
            marker
            for marker in (
                document.find('<div class="deck">'),
                document.find('<div class="slideshow"'),
            )
            if marker >= 0
        )
        assert banner_end < content_start, filename

    supervisor = PLANNING_STATUS_HTML["supervisor_presentation.html"].lower()
    assert "beide über dem publizierten sota-bereich" not in supervisor
    assert "pipeline liegt deutlich darüber" not in supervisor
    assert "kein unabhängiger benchmark oder validierungsnachweis" in supervisor
    assert "kein sota-überlegenheitsnachweis" in supervisor


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
