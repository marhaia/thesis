"""P8 / AG-15..16 cleanup and portability regression contracts."""

from __future__ import annotations

import io
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _parser_actions(parser):
    return {action.dest: action for action in parser._actions}


def test_evidence_tools_have_no_developer_machine_path_defaults():
    files = (
        "stage1/tools/umsi_boundary_input_probe.py",
        "stage1/tools/umsi_step2b_gate_runner.py",
    )
    for relative in files:
        source = _source(relative)
        assert "/Users/Q682780" not in source
        assert "Thesis_G_umsi_legacy_resize_fix" not in source


def test_boundary_probe_requires_external_artifacts_and_defaults_to_checkout():
    from stage1.tools.umsi_boundary_input_probe import build_parser

    actions = _parser_actions(build_parser())
    assert Path(actions["target_repo"].default).resolve() == ROOT
    for name in ("fixture", "evidence_npz", "weights"):
        assert actions[name].required is True
        assert actions[name].default is None


def test_step2b_runner_requires_external_artifacts_and_uses_checkout_paths():
    from stage1.tools.umsi_step2b_gate_runner import _build_parser

    actions = _parser_actions(_build_parser())
    assert Path(actions["repo"].default).resolve() == ROOT
    assert Path(actions["contract"].default).resolve() == (
        ROOT / "stage1/evidence/umsi_raw_output_gate_contract.json"
    )
    assert Path(actions["output_dir"].default).resolve() == ROOT / "stage1/evidence"
    for name in ("weights", "evidence_npz", "fixture"):
        assert actions[name].required is True
        assert actions[name].default is None


def test_single_image_and_screen_set_formats_have_one_production_policy():
    from stage1 import app

    expected = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff"})
    assert app.SINGLE_IMAGE_EXTENSIONS == expected
    assert app.SCREEN_SET_EXTENSIONS == expected | {".gif"}
    assert ".webp" not in app.SCREEN_SET_EXTENSIONS

    source = _source("stage1/app.py")
    assert source.count("frozenset({\".png\", \".jpg\", \".jpeg\", \".bmp\", \".tiff\"})") == 1
    assert source.count("if ext not in SINGLE_IMAGE_EXTENSIONS") == 6
    assert "allowed = SCREEN_SET_EXTENSIONS" in source


def test_live_route_does_not_load_unreachable_hceye_study_lookup():
    source = _source("stage1/app.py")
    assert 'extractor = HCEyeFeatureExtractor()' in source
    assert 'HCEyeFeatureExtractor(str(lookup_path))' not in source


def test_text_reader_documents_score_bearing_ocr_as_mandatory_and_fail_closed():
    source = _source("cognitive/text_reader.py")
    compact = " ".join(source.split())

    for required in (
        "does not authorize score-bearing analysis to continue",
        "score-bearing route returns structured",
        "HTTP 503 with no x19 or score",
        "is fatal on the score-bearing canonical-layout path",
        "must fail closed on this sentinel",
        "never substituted with zero text or another neutral value",
        "Non-score/offline callers",
    ):
        assert required in compact

    for misleading in (
        "reading costs disabled",
        "OCR is optional",
        "continue without OCR",
    ):
        assert misleading not in source


def test_known_silent_saliency_fallbacks_are_absent():
    model_source = _source("saliency/umsi_model.py")
    app_source = _source("stage1/app.py")
    assert model_source.count("load_weights(") == 1
    assert "skip_mismatch=False" in model_source
    assert "skip_mismatch=True" not in model_source
    assert "except: pass" not in app_source
    assert 'results["saliency_used"] = saliency_map is not None' in app_source


def test_ueyes_sample_script_has_no_benchmark_or_sota_verdict():
    source = _source("scripts/ueyes_saliency_validation.py")
    assert "ABOVE SOTA" not in source
    assert "typical SOTA" not in source
    assert "considered good" not in source
    assert "engineering comparison on the supplied sample only" in source


def test_default_pytest_collection_and_ci_share_one_configured_contract():
    pytest_config = _source("pytest.ini")
    ci = _source(".github/workflows/ci.yml")
    assert "tests/test_stage1_cleanup.py" in pytest_config
    assert "tests/test_claim_reconciliation.py" in pytest_config
    assert "run: python -m pytest -q" in ci


class _SanityResponse:
    def __init__(self, status_code=200, body=None, is_json=True):
        self.status_code = status_code
        self._body = body
        self.is_json = is_json

    def get_json(self, silent=False):
        return self._body


class _SanityClient:
    def __init__(self, response):
        self.response = response

    def post(self, *_args, **_kwargs):
        return self.response


def _sanity_index(response):
    from stage1.sanity_empty_screen import _index

    return _index(_SanityClient(response), io.BytesIO(b"image"), "case.png")


def test_empty_screen_sanity_uses_current_finite_layout_index_contract():
    value = _sanity_index(
        _SanityResponse(
            body={"layout": {"experimental_complexity_index": 23.75}}
        )
    )

    assert value == 23.75

    source = _source("stage1/sanity_empty_screen.py")
    assert 'body.get("cognitive_load_index")' not in source
    assert "idx * 100" not in source
    assert "layout.experimental_complexity_index" in source
    assert "raise SystemExit(main())" in source


@pytest.mark.parametrize(
    "response, expected_message",
    [
        (_SanityResponse(status_code=503, body={}), "expected HTTP 200"),
        (_SanityResponse(body={}, is_json=False), "expected a JSON response"),
        (_SanityResponse(body=None), "JSON response must be an object"),
        (_SanityResponse(body={}), "missing object 'layout'"),
        (
            _SanityResponse(body={"layout": []}),
            "missing object 'layout'",
        ),
        (
            _SanityResponse(body={"layout": {}}),
            "must be numeric",
        ),
        (
            _SanityResponse(
                body={"layout": {"experimental_complexity_index": "23.75"}}
            ),
            "must be numeric",
        ),
        (
            _SanityResponse(
                body={"layout": {"experimental_complexity_index": True}}
            ),
            "must be numeric",
        ),
        (
            _SanityResponse(
                body={"layout": {"experimental_complexity_index": float("nan")}}
            ),
            "must be finite",
        ),
        (
            _SanityResponse(
                body={"layout": {"experimental_complexity_index": float("inf")}}
            ),
            "must be finite",
        ),
        (
            _SanityResponse(
                body={"layout": {"experimental_complexity_index": float("-inf")}}
            ),
            "must be finite",
        ),
    ],
    ids=[
        "non-200",
        "non-json",
        "non-object-json",
        "missing-layout",
        "non-object-layout",
        "missing-index",
        "string-index",
        "boolean-index",
        "nan-index",
        "positive-infinity-index",
        "negative-infinity-index",
    ],
)
def test_empty_screen_sanity_fails_loudly_on_response_or_schema_drift(
    response, expected_message
):
    from stage1.sanity_empty_screen import SanityCheckError

    with pytest.raises(SanityCheckError, match=expected_message):
        _sanity_index(response)


def test_empty_screen_sanity_main_returns_nonzero_and_reports_failure(
    monkeypatch, capsys
):
    from stage1 import sanity_empty_screen as sanity

    monkeypatch.setattr(sanity, "_UEYES", "/definitely/not/a/corpus")

    class _App:
        @staticmethod
        def test_client():
            return object()

    monkeypatch.setattr(sanity, "app", _App())
    monkeypatch.setattr(
        sanity,
        "_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sanity.SanityCheckError("schema drift")
        ),
    )

    assert sanity.main() == 1
    captured = capsys.readouterr()
    assert "SANITY CHECK FAILED: schema drift" in captured.err
