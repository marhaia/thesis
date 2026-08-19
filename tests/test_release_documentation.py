"""Regression boundaries for release-facing technical documentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
STAGE1_DOCUMENTATION = (ROOT / "stage1" / "DOCUMENTATION.md").read_text(
    encoding="utf-8"
)
REPRODUCIBILITY = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
AUDIT_TRAIL = (ROOT / "AUDIT_TRAIL.md").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


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
    assert "Any x19 or layout byte change rejects the cleanup." in REPRODUCIBILITY
    assert "all 15 Stage-2 combinations" in REPRODUCIBILITY
    assert "97 distributions" in REPRODUCIBILITY
    assert "A local skip is not a failure" in REPRODUCIBILITY
    assert "final release record" in REPRODUCIBILITY


def test_audit_trail_distinguishes_technical_evidence_from_validation():
    assert "not a scientific validation report" in AUDIT_TRAIL
    assert "They do not establish peer review" in AUDIT_TRAIL
    assert "Stage 2 v1:** release candidate" in AUDIT_TRAIL
    assert "two independent auditors" in AUDIT_TRAIL
    assert "do not validate" in AUDIT_TRAIL


def test_ci_covers_active_stage2_and_release_branches():
    assert '- "stage2/**"' in CI
    assert '- "release/**"' in CI
