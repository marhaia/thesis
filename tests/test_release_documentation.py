"""Regression boundaries for release-facing technical documentation."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
STAGE1_DOCUMENTATION = (ROOT / "stage1" / "DOCUMENTATION.md").read_text(
    encoding="utf-8"
)
REPRODUCIBILITY = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
AUDIT_TRAIL = (ROOT / "AUDIT_TRAIL.md").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
APP_SOURCE = (ROOT / "stage1" / "app.py").read_text(encoding="utf-8")
SALIENCY_SOURCE = (ROOT / "saliency" / "saliency_features.py").read_text(
    encoding="utf-8"
)
UI = (ROOT / "stage1" / "ui" / "index.html").read_text(encoding="utf-8")


def _literal_assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name} was not found")


def test_readme_reports_release_state_without_overclaiming_stage2_freeze():
    compact = " ".join(README.split())
    assert "## Release status" in README
    assert "Stage 1 is frozen" in README
    assert "Stage 2 v1 remains a release candidate" in README
    assert "two independent clean-room technical audits" in compact
    assert "an untagged checkout must be treated as development state" in compact


def test_release_architecture_excludes_local_workspace_and_upstream_checkouts():
    section = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("### File Structure"):
        STAGE1_DOCUMENTATION.index("### Processing Flow")
    ]
    assert "Thesis_G/" not in section
    assert "├── ueyes/" not in section
    assert "├── aim/" not in section
    assert "└── venv/" not in section
    assert "saliency/postprocessing.py" not in section  # represented as a tree leaf
    assert "postprocessing.py" in section
    assert "scenario_proxy.py" in section
    assert "coherence_check.py" in section


def test_reproducibility_policy_requires_exact_output_parity_after_cleanup():
    compact = " ".join(REPRODUCIBILITY.split())
    assert "Any x19 or layout byte change rejects the cleanup." in REPRODUCIBILITY
    assert "all 15 Stage-2 combinations" in REPRODUCIBILITY
    assert "97 distributions" in REPRODUCIBILITY
    assert "A local skip is not a failure" in REPRODUCIBILITY
    assert "final release record" in REPRODUCIBILITY
    assert "Candidate-bound successor evidence" in REPRODUCIBILITY
    assert "cannot substitute for a fresh replay" in REPRODUCIBILITY
    assert "source-to-release blob manifest" in REPRODUCIBILITY
    assert "Commit/tree strings without resolvable objects" in compact


def test_audit_trail_distinguishes_technical_evidence_from_validation():
    assert "not a scientific validation report" in AUDIT_TRAIL
    assert "They do not establish peer review" in AUDIT_TRAIL
    assert "Stage 2 v1:** release candidate" in AUDIT_TRAIL
    assert "two independent auditors" in AUDIT_TRAIL
    assert "do not validate" in AUDIT_TRAIL
    assert "Technical v1.0.0-rc.2 (2026-08-19): rejected" in AUDIT_TRAIL
    assert "this was not a release PASS" in AUDIT_TRAIL


def test_ci_covers_active_stage2_and_release_branches():
    assert '- "stage2/**"' in CI
    assert '- "release/**"' in CI


def test_active_saliency_specification_matches_the_frozen_x19_order():
    expected = (
        "saliency_dispersion",
        "saliency_entropy",
        "saliency_coverage",
        "saliency_peak_count",
        "saliency_center_bias",
    )
    assert _literal_assignment(APP_SOURCE, "STAGE1_SALIENCY_FEATURE_NAMES") == expected

    module_order = [
        SALIENCY_SOURCE.index(f"{index}. {name}")
        for index, name in enumerate(expected, start=1)
    ]
    assert module_order == sorted(module_order)

    section = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("### 6.4 Saliency features"):
        STAGE1_DOCUMENTATION.index("### 6.5 Numeric Auxiliary Head")
    ]
    display_names = (
        "Dispersion",
        "Entropy",
        "Coverage",
        "Peak Count",
        "Center Bias",
    )
    table_order = [section.index(f"**{name}**") for name in display_names]
    assert table_order == sorted(table_order)
    assert "indices 8–12" in section
    assert "positions 9–13" in section


def test_active_dispersion_specification_states_the_production_equation():
    assert "Var}[x" not in SALIENCY_SOURCE
    assert "variance_x = sum((x-mu_x)^2*S)/T" in SALIENCY_SOURCE
    assert "min(sqrt(variance_x + variance_y) / 0.707, 1)" in SALIENCY_SOURCE

    section = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("### 6.4 Saliency features"):
        STAGE1_DOCUMENTATION.index("### 6.5 Numeric Auxiliary Head")
    ]
    assert "var_x=sum((x-mu_x)^2S)/T" in section
    assert "dispersion=min(sqrt(var_x+var_y)/0.707, 1)" in section
    assert "for T=0: dispersion=0" in section


def test_active_h6_layout_specification_states_the_complete_production_formula():
    section = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("### 6.6 Project-specific h6"):
        STAGE1_DOCUMENTATION.index("### 6.7 API endpoints")
    ]
    compact = " ".join(section.split())
    for required in (
        "min, p5, p25, p50, p75, p95, max",
        "0.876-0.05(C-W)",
        "1.081+0.1(T+N)/2",
        "0.935-0.04[1-(S+G)/2]",
        "0.30(1-h_1)+0.20(h_2-1)+0.20(1-h_3)+0.15h_4+0.15(1-h_5)",
        "{0.3}",
        "max(E,N,1-W)",
        "h_6=L_0P_{content}",
        "author-defined and uncalibrated",
        "not a cognitive-load measurement",
    ):
        assert required in compact


def test_dependency_and_csv_surfaces_do_not_reintroduce_retired_claims():
    dependency_section = STAGE1_DOCUMENTATION[
        STAGE1_DOCUMENTATION.index("## 8. Dependencies"):
        STAGE1_DOCUMENTATION.index("## 9. Usage")
    ]
    assert "opencv-python-headless | 4.10.0.84" in dependency_section
    assert "pip install numpy opencv-python" not in dependency_section
    assert "verify-runtime --strict" in dependency_section

    assert "complexity_vector.csv" not in UI
    assert "complexity_vectors_all.csv" not in UI
    assert "complexity_vectors_selected.csv" not in UI
    assert "complexity_analysis.csv" in UI
    assert "complexity_analyses_all.csv" in UI
    assert "complexity_analyses_selected.csv" in UI
