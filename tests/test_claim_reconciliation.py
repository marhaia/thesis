"""P7 / AG-12 regression tests for public claim reconciliation."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
UI = (ROOT / "stage1" / "ui" / "index.html").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYTEST_CONFIG = (ROOT / "pytest.ini").read_text(encoding="utf-8")


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


def test_readme_uses_bounded_construct_and_current_test_contract():
    lower = README.lower()
    compact = " ".join(lower.split())
    assert "exploratory, project-specific layout-complexity index" in compact
    assert "not a validated cognitive-load measurement" in compact
    assert "[v8 | s5 | h6]" in README
    assert "pytest -q" in README
    assert "deployed load score" not in lower
    assert "tests/test_claim_reconciliation.py" in PYTEST_CONFIG


def test_latest_expose_is_self_identified_as_historical():
    path = ROOT / "Literature" / "notes" / "expose" / "expose_final_v16_24jun.docx"
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "HISTORICAL WORKING DRAFT" in xml
    assert "not the current methodology or claim basis" in xml
