"""Tests for stage1/tools/canonical_saliency_norms.py (Stage 1 Step 3A-R1).

All tests use synthetic, temp-directory-only data, fake/injected predictors,
and (where feature-value computation is exercised) the REAL, TF-free
production ``extract_saliency_features`` function. No real UMSI/TensorFlow/
Keras import, no real weight loading, no real UEyes image and no real
inference occurs anywhere in this file.

Synthetic tests monkeypatch ``m.FROZEN_P9_REPORT_SHA256`` to a synthetic
report's own hash for success-path testing. Production code always defaults
to the real frozen hash (verified exactly by
``test_frozen_p9_report_sha_constant_is_exact``).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import stage1.tools.canonical_saliency_norms as m  # noqa: E402

_REAL_FEATURE_NORMS_JSON = (
    _REPO_ROOT / "stage1" / "data" / "results" / "feature_norms.json")
_REAL_VISUAL_NORMS_JSON = (
    _REPO_ROOT / "stage1" / "data" / "results"
    / "canonical_visual_feature_norms.json")
_REAL_FROZEN_P9_REPORT = (
    _REPO_ROOT / "stage1" / "evidence" / "step2b_results_p9"
    / "umsi_step2b_p9_corrigendum_20260804T122435Z" / "gate_report_p9.json")

_VISUAL_FEATURE_KEYS = (
    "shannon_entropy", "edge_density", "feature_congestion",
    "subband_entropy", "layout_symmetry", "chromatic_coherence",
    "visual_hierarchy", "interactive_element_density",
)


# ---------------------------------------------------------------------------
# Corpus / image builders
# ---------------------------------------------------------------------------
def _write_tiny_image(path: Path, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"failed to write test image {path}"


def _build_corpus(root: Path, counts: Dict[str, int],
                  extra_rows: Optional[List[tuple]] = None,
                  duplicate_content_pair: Optional[tuple] = None) -> Path:
    """Build a synthetic UEyes-shaped corpus: root/images + root/image_types.csv."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[tuple] = []
    seed = 0
    for cat, n in counts.items():
        for i in range(n):
            name = f"{cat}_{i:04d}.png"
            w = 6 + (i % 3)
            h = 4 + (i % 2)
            _write_tiny_image(images_dir / name, w, h, seed)
            seed += 1
            rows.append((name, cat))
    if extra_rows:
        for name, cat in extra_rows:
            _write_tiny_image(images_dir / name, 6, 4, seed)
            seed += 1
            rows.append((name, cat))
    if duplicate_content_pair:
        name_a, name_b, cat = duplicate_content_pair
        _write_tiny_image(images_dir / name_a, 6, 4, seed)
        seed += 1
        (images_dir / name_b).write_bytes((images_dir / name_a).read_bytes())
        rows.append((name_a, cat))
        rows.append((name_b, cat))

    types_csv = root / "image_types.csv"
    with open(types_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Image Name", "Category"])
        for name, cat in rows:
            writer.writerow([name, cat])
    return root


@pytest.fixture(scope="module")
def full_corpus_root(tmp_path_factory):
    """A genuine 495/495/495 = 1,485 synthetic corpus, built once and reused."""
    root = tmp_path_factory.mktemp("ueyes_full_corpus")
    _build_corpus(root, {"desktop": 495, "mobile": 495, "web": 495})
    return root


@pytest.fixture
def small_corpus_contract(monkeypatch):
    """Monkeypatch the corpus-count contract down to 2+2+2=6 for fast tests."""
    monkeypatch.setattr(m, "FULL_RUN_TOTAL", 6)
    monkeypatch.setattr(m, "FULL_RUN_PER_CATEGORY", 2)
    monkeypatch.setattr(m, "EXPECTED_AUTHORIZED_COUNTS",
                        {"desktop": 2, "mobile": 2, "web": 2})
    return {"desktop": 2, "mobile": 2, "web": 2}


# ---------------------------------------------------------------------------
# P9 report / identity builders
# ---------------------------------------------------------------------------
def _default_gate(name: str, passed: bool = True,
                  reason: Optional[str] = None,
                  observed: Optional[dict] = None) -> dict:
    return {"name": name, "passed": passed, "reason": reason,
           "observed": observed if observed is not None else {}}


def _build_p9_report(path: Path, *, source_sha: str, weights_sha: str,
                    conclusion: str = m.P9_REQUIRED_CONCLUSION,
                    verdict_conclusion: Optional[str] = None,
                    verdict_norms_status: str = m.P9_REQUIRED_NORMS_STATUS,
                    meta_norms_status: Optional[str] = None,
                    negative_control_status: str = m.P9_REQUIRED_NEGATIVE_CONTROL_STATUS,
                    verdict_overrides: Optional[dict] = None,
                    meta_overrides: Optional[dict] = None,
                    identities_overrides: Optional[dict] = None,
                    acceptance_gates: Optional[list] = None,
                    diagnostic_gates: Optional[list] = None,
                    omit_sections: Optional[list] = None) -> str:
    meta_norms_status = (verdict_norms_status if meta_norms_status is None
                         else meta_norms_status)
    verdict_conclusion = (conclusion if verdict_conclusion is None
                          else verdict_conclusion)
    verdict = {
        "conclusion": verdict_conclusion,
        "saliency_norms_release_status": verdict_norms_status,
        "independent_inference_performed": False,
        "negative_control_status": negative_control_status,
        "source_sha_match": True,
        "weights_sha_match": True,
        "p8_historical_result_preserved": True,
    }
    if verdict_overrides:
        verdict.update(verdict_overrides)
    meta = {
        "saliency_norms_release_status": meta_norms_status,
        "independent_inference_performed": False,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    identities = {"p8_source_sha256": source_sha, "weights_sha256": weights_sha}
    if identities_overrides:
        identities.update(identities_overrides)
    if acceptance_gates is None:
        acceptance_gates = [_default_gate("gate_1"), _default_gate("gate_2")]
    if diagnostic_gates is None:
        diagnostic_gates = [_default_gate("diag_1", observed={"alarm": False})]
    report = {
        "conclusion": conclusion,
        "verdict": verdict,
        "assessment_metadata": meta,
        "identities": identities,
        "acceptance_gate_results": acceptance_gates,
        "diagnostic_gate_results": diagnostic_gates,
    }
    for section in (omit_sections or []):
        report.pop(section, None)
    data = json.dumps(report, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake_identity(tmp_path_factory, monkeypatch):
    """Fake production source + weights files, wired into the module globals."""
    d = tmp_path_factory.mktemp("identity")
    source_path = d / "umsi_model.py"
    source_path.write_text("# fake production source for testing\n")
    weights_path = d / "umsi_weights.hdf5"
    weights_path.write_bytes(b"FAKE-UMSI-WEIGHTS-BYTES-0123456789")
    monkeypatch.setattr(m, "_PRODUCTION_SOURCE_PATH", source_path)
    monkeypatch.setattr(m, "_PREDICTOR_MODULE_PATH", source_path)
    return {
        "source_path": source_path,
        "weights_path": weights_path,
        "source_sha": m._sha256_file(source_path),
        "weights_sha": m._sha256_file(weights_path),
    }


@pytest.fixture
def frozen_synthetic_p9(tmp_path, fake_identity, monkeypatch):
    """A complete, synthetic PASS P9 report, with FROZEN_P9_REPORT_SHA256
    monkeypatched to equal it, for success-path tests only."""
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha=fake_identity["weights_sha"])
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    return {"path": p9_path, "sha": p9_sha}


# ---------------------------------------------------------------------------
# Fake predictor
# ---------------------------------------------------------------------------
class _FakePredictor:
    """Records (path, width, height) per call; never imports/loads UMSI/TF."""

    def __init__(self, fail_at_call: Optional[int] = None):
        self.fail_at_call = fail_at_call
        self.calls: List[tuple] = []
        self._n = 0

    def predict_saliency(self, image_path):
        self._n += 1
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        self.calls.append((str(image_path), w, h))
        if self.fail_at_call is not None and self._n == self.fail_at_call:
            raise RuntimeError(f"synthetic induced failure at call {self._n}")
        rng = np.random.default_rng(self._n)
        return rng.random((h, w)).astype(np.float32)


def _factory_with_flag(fake_predictor, load_flag: dict):
    def factory(weights_path):
        load_flag["loaded"] = True
        load_flag["weights_path"] = weights_path
        return fake_predictor
    return factory


def _extractor_with_nan_at(call_index: int):
    from saliency.saliency_features import extract_saliency_features as real
    state = {"n": 0}

    def fn(heatmap):
        state["n"] += 1
        feats = dict(real(heatmap))
        if state["n"] == call_index:
            feats["saliency_entropy"] = float("nan")
        return feats
    return fn


def _run_main(dataset_root, output_dir, identity, p9_report_path, p9_sha,
             fake_predictor, load_flag, extra_argv=None, extractor_fn=None):
    argv = [
        "--p9-report", str(p9_report_path),
        "--p9-report-sha", p9_sha,
        "--dataset-root", str(dataset_root),
        "--weights", str(identity["weights_path"]),
        "--output-dir", str(output_dir),
    ]
    if extra_argv:
        argv.extend(extra_argv)
    return m.main(argv, predictor_factory=_factory_with_flag(fake_predictor, load_flag),
                 extractor_fn=extractor_fn)


# ===========================================================================
# 1 + 2. Feature-key discipline
# ===========================================================================
def test_aggregate_emits_only_five_saliency_keys():
    values = {k: [1.0, 2.0, 3.0] for k in m.SALIENCY_FEATURE_KEYS}
    norms = m.aggregate(values)
    assert set(norms.keys()) == set(m.SALIENCY_FEATURE_KEYS)


def test_no_visual_feature_key_anywhere():
    assert set(m.SALIENCY_FEATURE_KEYS).isdisjoint(_VISUAL_FEATURE_KEYS)
    values = {k: [1.0, 2.0] for k in m.SALIENCY_FEATURE_KEYS}
    values["shannon_entropy"] = [9.0, 9.0]
    norms = m.aggregate(values)
    assert "shannon_entropy" not in norms
    assert set(norms.keys()) == set(m.SALIENCY_FEATURE_KEYS)


def test_generator_has_no_alternative_feature_formula_implementation():
    source = (_REPO_ROOT / "stage1" / "tools"
             / "canonical_saliency_norms.py").read_text(encoding="utf-8")
    assert "from saliency.saliency_features import extract_saliency_features" in source
    forbidden_defs = [
        "_compute_dispersion", "_compute_peak_count", "_compute_center_bias",
        "_compute_entropy", "_compute_coverage",
    ]
    for name in forbidden_defs:
        assert f"def {name}" not in source


# ===========================================================================
# 4. Frozen P9 report hash constant is exact
# ===========================================================================
def test_frozen_p9_report_sha_constant_is_exact():
    assert m.FROZEN_P9_REPORT_SHA256 == (
        "a71b68fc00c2be2e0dc6220054e47fb35a56f3e0bbe6bad311a7f7975ecc530d")
    if _REAL_FROZEN_P9_REPORT.exists():
        actual = m._sha256_file(_REAL_FROZEN_P9_REPORT)
        assert actual == m.FROZEN_P9_REPORT_SHA256


# ===========================================================================
# 1. Self-consistent but non-frozen P9 report/hash is rejected
# ===========================================================================
def test_self_consistent_non_frozen_p9_report_rejected(tmp_path, fake_identity):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha=fake_identity["weights_sha"])
    assert p9_sha != m.FROZEN_P9_REPORT_SHA256
    with pytest.raises(m.SaliencyNormsError, match="FROZEN_P9_REPORT_SHA256"):
        m.run_p9_release_preflight(
            p9_report_path=p9_path, p9_report_sha=p9_sha,
            production_source_path=fake_identity["source_path"],
            weights_path=fake_identity["weights_path"],
            predictor_module_path=fake_identity["source_path"],
            extractor_module_path=m._EXTRACTOR_MODULE_PATH,
        )


# ===========================================================================
# 2. Minimal PASS-looking report without Gate 1/2/Diag-1 rejected
# ===========================================================================
def test_minimal_report_without_gates_rejected(tmp_path, fake_identity, monkeypatch):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha=fake_identity["weights_sha"],
                              omit_sections=["acceptance_gate_results",
                                            "diagnostic_gate_results"])
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    with pytest.raises(m.SaliencyNormsError):
        m.run_p9_release_preflight(
            p9_report_path=p9_path, p9_report_sha=p9_sha,
            production_source_path=fake_identity["source_path"],
            weights_path=fake_identity["weights_path"],
            predictor_module_path=fake_identity["source_path"],
            extractor_module_path=m._EXTRACTOR_MODULE_PATH,
        )


# ===========================================================================
# 3. Malformed / alarmed Diag-1 rejected
# ===========================================================================
@pytest.mark.parametrize("diagnostic_gates", [
    [],
    [_default_gate("diag_1"), _default_gate("diag_1")],
    [_default_gate("diag_1", passed=False)],
    [_default_gate("diag_1", reason="something went wrong")],
    [_default_gate("diag_1", observed={"alarm": True})],
    [_default_gate("diag_1", observed={})],
])
def test_malformed_or_alarmed_diag1_rejected(tmp_path, fake_identity, monkeypatch,
                                             diagnostic_gates):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha=fake_identity["weights_sha"],
                              diagnostic_gates=diagnostic_gates)
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    with pytest.raises(m.SaliencyNormsError):
        m.run_p9_release_preflight(
            p9_report_path=p9_path, p9_report_sha=p9_sha,
            production_source_path=fake_identity["source_path"],
            weights_path=fake_identity["weights_path"],
            predictor_module_path=fake_identity["source_path"],
            extractor_module_path=m._EXTRACTOR_MODULE_PATH,
        )


# ===========================================================================
# 5. TensorFlow/Keras/h5py versions fingerprint-bound without importing them
# ===========================================================================
def test_runtime_versions_resolved_without_importing():
    before = set(sys.modules)
    versions = m.resolve_runtime_versions()
    after = set(sys.modules)
    assert "tensorflow" not in after - before
    assert "keras" not in after - before
    assert set(versions.keys()) == {
        "tensorflow_distribution_name", "tensorflow_version",
        "keras_version", "h5py_version"}
    assert versions["tensorflow_version"]
    assert versions["keras_version"]
    assert versions["h5py_version"]


def test_runtime_versions_missing_blocks(monkeypatch):
    monkeypatch.setattr(m, "_TENSORFLOW_DISTRIBUTION_CANDIDATES", ("no_such_tf_dist",))
    with pytest.raises(m.SaliencyNormsError):
        m.resolve_runtime_versions()


def test_full_run_manifest_contains_runtime_versions(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}
    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)
    assert rc == 0
    manifest = json.loads(
        (output_dir / "saliency_feature_norms_manifest.json").read_text())
    env = manifest["environment"]
    assert env["tensorflow_version"]
    assert env["keras_version"]
    assert env["h5py_version"]
    assert env["tensorflow_distribution_name"]


# ===========================================================================
# 6. Changed runtime version rejects resume before model loading
# ===========================================================================
def test_changed_runtime_version_rejects_resume(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9, monkeypatch):
    output_dir = tmp_path / "out"
    fake1 = _FakePredictor(fail_at_call=5)
    load_flag1: dict = {"loaded": False}
    _run_main(full_corpus_root, output_dir, fake_identity,
             frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
             fake1, load_flag1)

    real_resolve = m.resolve_runtime_versions

    def _tampered():
        v = dict(real_resolve())
        v["tensorflow_version"] = "999.999.999"
        return v
    monkeypatch.setattr(m, "resolve_runtime_versions", _tampered)

    fake2 = _FakePredictor()
    load_flag2: dict = {"loaded": False}
    rc2 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake2, load_flag2, extra_argv=["--resume"])
    assert rc2 != 0
    assert load_flag2["loaded"] is False


# ===========================================================================
# 3/17. Native dimensions reach the predictor -- via the REAL generator path
# ===========================================================================
def test_native_dimensions_reach_predictor_through_generator(
        tmp_path, fake_identity, small_corpus_contract, frozen_synthetic_p9):
    root = _build_corpus(tmp_path / "corpus", small_corpus_contract)
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc == 0
    assert load_flag["loaded"] is True
    images_dir = root / "images"
    with open(output_dir / "saliency_feature_rows.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(small_corpus_contract.values())
    assert len(fake.calls) == len(rows)

    by_path = {Path(p).name: (w, h) for p, w, h in fake.calls}
    for row in rows:
        rel_path = row["relative_path"]
        expected_w, expected_h = by_path[rel_path]
        assert int(row["native_width"]) == expected_w
        assert int(row["native_height"]) == expected_h
        actual_img = cv2.imread(str(images_dir / rel_path))
        actual_h, actual_w = actual_img.shape[:2]
        assert expected_w == actual_w
        assert expected_h == actual_h
    for call_path, _w, _h in fake.calls:
        assert Path(call_path).parent == images_dir
    assert len({(w, h) for _p, w, h in fake.calls}) > 1


# ===========================================================================
# 4/16. Poster excluded / unknown category / malformed rows / traversal
# ===========================================================================
def test_poster_excluded_from_population(tmp_path):
    root = _build_corpus(
        tmp_path, {"desktop": 2, "mobile": 2, "web": 2},
        extra_rows=[("poster_0000.png", "poster"), ("poster_0001.png", "poster")])
    images_dir = root / "images"
    types_csv = root / "image_types.csv"
    population, counts = m.discover_authorized_corpus(types_csv, images_dir)
    assert "poster" not in counts
    assert all(cat != "poster" for _name, cat in population)


def test_unknown_category_rejected(tmp_path):
    root = _build_corpus(
        tmp_path, {"desktop": 2, "mobile": 2, "web": 2},
        extra_rows=[("infographic_0000.png", "infographic")])
    with pytest.raises(m.SaliencyNormsError, match="unknown|unauthorized"):
        m.discover_authorized_corpus(root / "image_types.csv", root / "images")


def test_malformed_corpus_row_rejected(tmp_path):
    root = tmp_path
    images_dir = root / "images"
    images_dir.mkdir()
    _write_tiny_image(images_dir / "desktop_0000.png", 6, 4, 0)
    types_csv = root / "image_types.csv"
    with open(types_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Image Name", "Category"])
        w.writerow(["", "desktop"])
    with pytest.raises(m.SaliencyNormsError):
        m.discover_authorized_corpus(types_csv, images_dir)

    with open(types_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Image Name", "Category"])
        w.writerow(["desktop_0000.png", ""])
    with pytest.raises(m.SaliencyNormsError):
        m.discover_authorized_corpus(types_csv, images_dir)


def test_path_traversal_in_corpus_row_rejected(tmp_path):
    root = tmp_path
    images_dir = root / "images"
    images_dir.mkdir()
    _write_tiny_image(images_dir / "desktop_0000.png", 6, 4, 0)
    types_csv = root / "image_types.csv"
    with open(types_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Image Name", "Category"])
        w.writerow(["../desktop_0000.png", "desktop"])
    with pytest.raises(m.SaliencyNormsError):
        m.discover_authorized_corpus(types_csv, images_dir)

    with open(types_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Image Name", "Category"])
        w.writerow(["/etc/passwd", "desktop"])
    with pytest.raises(m.SaliencyNormsError):
        m.discover_authorized_corpus(types_csv, images_dir)


# ===========================================================================
# 15. Unlisted image file on disk rejected
# ===========================================================================
def test_unlisted_image_file_on_disk_rejected(tmp_path):
    root = _build_corpus(tmp_path, {"desktop": 2, "mobile": 2, "web": 2})
    images_dir = root / "images"
    _write_tiny_image(images_dir / "mystery_extra.png", 6, 4, 999)
    with pytest.raises(m.SaliencyNormsError, match="not listed"):
        m.discover_authorized_corpus(root / "image_types.csv", images_dir)


# ===========================================================================
# 5. Exact 495+495+495 = 1,485 inventory passes
# ===========================================================================
def test_exact_495_495_495_inventory_passes(full_corpus_root):
    images_dir = full_corpus_root / "images"
    types_csv = full_corpus_root / "image_types.csv"
    population, counts = m.discover_authorized_corpus(types_csv, images_dir)
    assert len(population) == 1485
    assert counts == {"desktop": 495, "mobile": 495, "web": 495}
    m.enforce_corpus_count_contract(counts, len(population))
    records, _sha = m.build_ordered_corpus_manifest(population, images_dir)
    assert [r["category"] for r in records[:495]] == ["desktop"] * 495
    assert [r["category"] for r in records[495:990]] == ["mobile"] * 495
    assert [r["category"] for r in records[990:1485]] == ["web"] * 495
    assert records[0]["global_index"] == 1
    assert records[-1]["global_index"] == 1485


# ===========================================================================
# 6. Count mismatch stops before model loading
# ===========================================================================
def test_count_mismatch_stops_before_model_loading(
        tmp_path, fake_identity, frozen_synthetic_p9):
    root = _build_corpus(tmp_path / "corpus", {"desktop": 2, "mobile": 2, "web": 2})
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc != 0
    assert load_flag["loaded"] is False
    assert not output_dir.exists()


# ===========================================================================
# 7. Duplicate path / duplicate hash rejected
# ===========================================================================
def test_duplicate_row_in_csv_rejected(tmp_path):
    root = tmp_path
    images_dir = root / "images"
    images_dir.mkdir()
    _write_tiny_image(images_dir / "desktop_0000.png", 6, 4, 0)
    types_csv = root / "image_types.csv"
    with open(types_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Image Name", "Category"])
        w.writerow(["desktop_0000.png", "desktop"])
        w.writerow(["desktop_0000.png", "desktop"])
    with pytest.raises(m.SaliencyNormsError, match="Duplicate row"):
        m.discover_authorized_corpus(types_csv, images_dir)


def test_duplicate_image_content_hash_rejected(tmp_path):
    root = _build_corpus(
        tmp_path, {"desktop": 2, "mobile": 2, "web": 2},
        duplicate_content_pair=("desktop_dupA.png", "desktop_dupB.png", "desktop"))
    images_dir = root / "images"
    types_csv = root / "image_types.csv"
    population, _counts = m.discover_authorized_corpus(types_csv, images_dir)
    with pytest.raises(m.SaliencyNormsError, match="Duplicate image content"):
        m.build_ordered_corpus_manifest(population, images_dir)


# ===========================================================================
# 8. Non-finite feature value rejected
# ===========================================================================
def test_non_finite_feature_value_rejected(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag, extractor_fn=_extractor_with_nan_at(1))

    assert rc != 0
    assert not (output_dir / "saliency_feature_norms.json").exists()


# ===========================================================================
# 9. One image failure prevents aggregation
# ===========================================================================
def test_one_image_failure_prevents_aggregation(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor(fail_at_call=1)
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc != 0
    assert load_flag["loaded"] is True
    assert not (output_dir / "saliency_feature_norms.json").exists()
    assert not (output_dir / "saliency_feature_norms_manifest.json").exists()
    sidecar = json.loads((output_dir / "saliency_feature_rows.csv.provenance.json").read_text())
    assert sidecar["status"] == "failed"
    assert sidecar["last_failure"]["global_index"] == 1


# ===========================================================================
# 10 / 18. Full successful run + fresh/partial-resume still succeed
# ===========================================================================
def test_full_run_success_writes_exactly_1485_rows(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc == 0
    assert load_flag["loaded"] is True
    norms_path = output_dir / "saliency_feature_norms.json"
    manifest_path = output_dir / "saliency_feature_norms_manifest.json"
    assert norms_path.exists()
    assert manifest_path.exists()

    norms = json.loads(norms_path.read_text())
    assert set(norms["features"].keys()) == set(m.SALIENCY_FEATURE_KEYS)
    assert norms["num_images"] == 1485
    for k in m.SALIENCY_FEATURE_KEYS:
        assert norms["features"][k]["n"] == 1485

    manifest = json.loads(manifest_path.read_text())
    assert manifest["authorized_total"] == 1485
    assert manifest["authorized_category_counts"] == {
        "desktop": 495, "mobile": 495, "web": 495}
    assert manifest["feature_order"] == list(m.SALIENCY_FEATURE_KEYS)
    assert "run_fingerprint_sha256" in manifest

    with open(output_dir / "saliency_feature_rows.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1485


# ===========================================================================
# 11. Statistical result matches a known synthetic fixture
# ===========================================================================
def test_aggregate_statistics_match_hand_computed_fixture():
    values = {"saliency_dispersion": [10.0, 20.0, 30.0, 40.0]}
    for k in m.SALIENCY_FEATURE_KEYS:
        values.setdefault(k, [1.0])
    norms = m.aggregate(values)
    d = norms["saliency_dispersion"]
    assert d["n"] == 4
    assert d["mean"] == pytest.approx(25.0)
    assert d["std"] == pytest.approx(12.909944487358056)
    assert d["min"] == pytest.approx(10.0)
    assert d["max"] == pytest.approx(40.0)
    assert d["p50"] == pytest.approx(25.0)


# ===========================================================================
# 12. Fresh vs. resume directory discipline
# ===========================================================================
def test_fresh_run_refuses_nonempty_output_dir(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stray_file.txt").write_text("preexisting")
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc != 0
    assert load_flag["loaded"] is False


def test_resume_requires_explicit_flag(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake1 = _FakePredictor()
    load_flag1: dict = {"loaded": False}
    rc1 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake1, load_flag1)
    assert rc1 == 0

    fake2 = _FakePredictor()
    load_flag2: dict = {"loaded": False}
    rc2 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake2, load_flag2)
    assert rc2 != 0
    assert load_flag2["loaded"] is False


# ===========================================================================
# 14/15/17/18. Resume fingerprint semantics
# ===========================================================================
def test_resume_with_identical_fingerprint_continues(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"

    fake1 = _FakePredictor(fail_at_call=800)
    load_flag1: dict = {"loaded": False}
    rc1 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake1, load_flag1)
    assert rc1 != 0
    with open(output_dir / "saliency_feature_rows.csv") as f:
        n_partial = sum(1 for _ in csv.DictReader(f))
    assert n_partial == 799

    fake2 = _FakePredictor()
    load_flag2: dict = {"loaded": False}
    rc2 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake2, load_flag2, extra_argv=["--resume"])
    assert rc2 == 0
    assert len(fake2.calls) == 1485 - n_partial
    norms = json.loads((output_dir / "saliency_feature_norms.json").read_text())
    assert norms["num_images"] == 1485


def test_resume_rejected_on_changed_weights(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9, monkeypatch):
    output_dir = tmp_path / "out"
    fake1 = _FakePredictor(fail_at_call=5)
    load_flag1: dict = {"loaded": False}
    _run_main(full_corpus_root, output_dir, fake_identity,
             frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
             fake1, load_flag1)

    fake_identity["weights_path"].write_bytes(b"DIFFERENT-WEIGHTS-BYTES-9876543210")
    new_weights_sha = m._sha256_file(fake_identity["weights_path"])
    p9_path = frozen_synthetic_p9["path"]
    p9_sha2 = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                               weights_sha=new_weights_sha)
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha2)

    fake2 = _FakePredictor()
    load_flag2: dict = {"loaded": False}
    rc2 = _run_main(full_corpus_root, output_dir, fake_identity, p9_path, p9_sha2,
                    fake2, load_flag2, extra_argv=["--resume"])
    assert rc2 != 0
    assert load_flag2["loaded"] is False


def test_resume_rejected_on_changed_dataset(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake1 = _FakePredictor(fail_at_call=5)
    load_flag1: dict = {"loaded": False}
    _run_main(full_corpus_root, output_dir, fake_identity,
             frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
             fake1, load_flag1)

    changed_path = full_corpus_root / "images" / "desktop_0000.png"
    original_bytes = changed_path.read_bytes()
    try:
        _write_tiny_image(changed_path, 6, 4, seed=999999)
        fake2 = _FakePredictor()
        load_flag2: dict = {"loaded": False}
        rc2 = _run_main(full_corpus_root, output_dir, fake_identity,
                        frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                        fake2, load_flag2, extra_argv=["--resume"])
        assert rc2 != 0
        assert load_flag2["loaded"] is False
    finally:
        changed_path.write_bytes(original_bytes)


# ===========================================================================
# 7/8/9/10/11/12. Sidecar structural fail-closed checks
# ===========================================================================
def _write_valid_checkpoint(csv_path: Path, sidecar_path: Path,
                           run_fingerprint: dict, fingerprint_sha: str,
                           manifest_entry: dict, row_overrides: Optional[dict] = None,
                           sidecar_overrides: Optional[dict] = None,
                           status: str = m.STATUS_RUNNING,
                           header_override: Optional[list] = None):
    fieldnames = ["global_index", "category", "relative_path",
                 m.ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                 *m.SALIENCY_FEATURE_KEYS, m.ROW_FEATURE_ORDER_COLUMN,
                 m.ROW_FINGERPRINT_COLUMN]
    row = {
        "global_index": str(manifest_entry["global_index"]),
        "category": manifest_entry["category"], "relative_path": "a.png",
        m.ROW_IMAGE_SHA_COLUMN: manifest_entry["image_sha256"],
        "native_width": str(manifest_entry["native_width"]),
        "native_height": str(manifest_entry["native_height"]),
        m.ROW_FEATURE_ORDER_COLUMN: "|".join(m.SALIENCY_FEATURE_KEYS),
        m.ROW_FINGERPRINT_COLUMN: fingerprint_sha,
    }
    for k in m.SALIENCY_FEATURE_KEYS:
        row[k] = "1.0"
    if row_overrides:
        row.update(row_overrides)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_override or fieldnames)
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in (header_override or fieldnames)})

    segments = [{"invocation_index": 0, "started_at": "2026-01-01T00:00:00Z",
                "status": m.STATUS_RUNNING}]
    sidecar = {
        "schema_version": m.SCHEMA_VERSION,
        "status": status,
        "recovered_from_interruption": False,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint_sha,
        "csv_basename": csv_path.name,
        "segments": segments,
        "invocation_count": 1,
        "last_failure": None,
    }
    if sidecar_overrides:
        sidecar.update(sidecar_overrides)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))


def _minimal_manifest_and_fingerprint(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_tiny_image(images_dir / "a.png", 6, 4, 0)
    manifest_entry = {
        "global_index": 1, "category": "desktop",
        "relative_path": "a.png",
        "image_sha256": m._sha256_file(images_dir / "a.png"),
        "native_width": 6, "native_height": 4,
    }
    manifest_by_path = {"a.png": manifest_entry}
    run_fingerprint = {"dummy": True, "n": 1}
    fingerprint_sha = hashlib.sha256(
        m._canonical_json(run_fingerprint).encode("utf-8")).hexdigest()
    return images_dir, manifest_by_path, manifest_entry, run_fingerprint, fingerprint_sha


def test_tampered_sidecar_fingerprint_object_rejected(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    tampered_fp = dict(run_fp)
    tampered_fp["n"] = 999
    _write_valid_checkpoint(csv_path, sidecar_path, tampered_fp, fp_sha, entry)
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_wrong_csv_basename_rejected(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                            sidecar_overrides={"csv_basename": "wrong_name.csv"})
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_completed_sidecar_cannot_resume(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                            status=m.STATUS_COMPLETED)
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_malformed_segments_rejected(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                            sidecar_overrides={"segments": "not-a-list"})
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)

    _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                            sidecar_overrides={"segments": [{"invocation_index": 0}]})
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_additional_checkpoint_columns_rejected(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    fieldnames = ["global_index", "category", "relative_path",
                 m.ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                 *m.SALIENCY_FEATURE_KEYS, m.ROW_FEATURE_ORDER_COLUMN,
                 m.ROW_FINGERPRINT_COLUMN, "extra_bogus_column"]
    _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                            header_override=fieldnames)
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_blank_checkpoint_row_rejected(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    csv_path = tmp_path / "rows.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    fieldnames = ["global_index", "category", "relative_path",
                 m.ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                 *m.SALIENCY_FEATURE_KEYS, m.ROW_FEATURE_ORDER_COLUMN,
                 m.ROW_FINGERPRINT_COLUMN]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({k: "" for k in fieldnames})
    segments = [{"invocation_index": 0, "started_at": "2026-01-01T00:00:00Z",
                "status": m.STATUS_RUNNING}]
    sidecar = {
        "schema_version": m.SCHEMA_VERSION, "status": m.STATUS_RUNNING,
        "recovered_from_interruption": False, "run_fingerprint": run_fp,
        "run_fingerprint_sha256": fp_sha, "csv_basename": csv_path.name,
        "segments": segments, "invocation_count": 1, "last_failure": None,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    with pytest.raises(m.SaliencyNormsError):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


def test_load_checkpoint_rejects_malformed_rows(tmp_path):
    images_dir, manifest_by_path, entry, run_fp, fp_sha = \
        _minimal_manifest_and_fingerprint(tmp_path)
    scenarios = {
        "wrong_fingerprint": {m.ROW_FINGERPRINT_COLUMN: "0" * 64},
        "wrong_feature_order": {m.ROW_FEATURE_ORDER_COLUMN: "wrong|order"},
        "not_in_manifest": {"relative_path": "not_present.png"},
        "wrong_category": {"category": "mobile"},
        "wrong_global_index": {"global_index": "999"},
        "non_finite_feature": {"saliency_entropy": "nan"},
        "missing_feature": {"saliency_entropy": ""},
    }
    for label, overrides in scenarios.items():
        csv_path = tmp_path / f"{label}.csv"
        sidecar_path = m.sidecar_path_for(csv_path)
        _write_valid_checkpoint(csv_path, sidecar_path, run_fp, fp_sha, entry,
                                row_overrides=overrides)
        with pytest.raises(m.SaliencyNormsError):
            m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                         manifest_by_path, images_dir)

    csv_path = tmp_path / "duplicate.csv"
    sidecar_path = m.sidecar_path_for(csv_path)
    fieldnames = ["global_index", "category", "relative_path",
                 m.ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                 *m.SALIENCY_FEATURE_KEYS, m.ROW_FEATURE_ORDER_COLUMN,
                 m.ROW_FINGERPRINT_COLUMN]
    good_row = {
        "global_index": "1", "category": "desktop", "relative_path": "a.png",
        m.ROW_IMAGE_SHA_COLUMN: entry["image_sha256"],
        "native_width": "6", "native_height": "4",
        m.ROW_FEATURE_ORDER_COLUMN: "|".join(m.SALIENCY_FEATURE_KEYS),
        m.ROW_FINGERPRINT_COLUMN: fp_sha,
    }
    for k in m.SALIENCY_FEATURE_KEYS:
        good_row[k] = "1.0"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(good_row)
        w.writerow(good_row)
    segments = [{"invocation_index": 0, "started_at": "2026-01-01T00:00:00Z",
                "status": m.STATUS_RUNNING}]
    sidecar = {
        "schema_version": m.SCHEMA_VERSION, "status": m.STATUS_RUNNING,
        "recovered_from_interruption": False, "run_fingerprint": run_fp,
        "run_fingerprint_sha256": fp_sha, "csv_basename": csv_path.name,
        "segments": segments, "invocation_count": 1, "last_failure": None,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    with pytest.raises(m.SaliencyNormsError, match="duplicate"):
        m.load_checkpoint_for_resume(csv_path, sidecar_path, run_fp, fp_sha,
                                     manifest_by_path, images_dir)


# ===========================================================================
# 13/14. Existing final artifacts / resume-after-completion
# ===========================================================================
def test_existing_final_artifacts_block_resume(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake1 = _FakePredictor()
    load_flag1: dict = {"loaded": False}
    rc1 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake1, load_flag1)
    assert rc1 == 0

    norms_bytes_before = (output_dir / "saliency_feature_norms.json").read_bytes()
    manifest_bytes_before = (output_dir / "saliency_feature_norms_manifest.json").read_bytes()
    csv_bytes_before = (output_dir / "saliency_feature_rows.csv").read_bytes()

    fake2 = _FakePredictor()
    load_flag2: dict = {"loaded": False}
    rc2 = _run_main(full_corpus_root, output_dir, fake_identity,
                    frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                    fake2, load_flag2, extra_argv=["--resume"])

    assert rc2 != 0
    assert load_flag2["loaded"] is False
    assert (output_dir / "saliency_feature_norms.json").read_bytes() == norms_bytes_before
    assert (output_dir / "saliency_feature_norms_manifest.json").read_bytes() == manifest_bytes_before
    assert (output_dir / "saliency_feature_rows.csv").read_bytes() == csv_bytes_before


# ===========================================================================
# 17. Protected output paths rejected
# ===========================================================================
def test_protected_output_paths_rejected(tmp_path):
    protected_examples = [
        _REPO_ROOT / "stage1" / "data" / "results" / "feature_norms.json",
        _REPO_ROOT / "stage1" / "data" / "results" / "feature_norms_visual.json",
        _REPO_ROOT / "hceye" / "sensitivity_lookup.json",
        _REPO_ROOT / "stage1" / "evidence" / "step2b_results_p8" / "x.json",
        _REPO_ROOT / "stage1" / "evidence" / "step2b_results_p9" / "x.json",
        _REPO_ROOT / "saliency" / "weights" / "some_checkpoint.hdf5",
        _REPO_ROOT / "saliency" / "umsi_model.py",
        _REPO_ROOT / "saliency" / "saliency_features.py",
        tmp_path / "some_dir" / "feature_norms.json",
    ]
    for p in protected_examples:
        with pytest.raises(m.SaliencyNormsError):
            m._refuse_protected_output(p, "test artifact")
    m._refuse_protected_output(tmp_path / "saliency_feature_norms.json", "norms JSON")


# ===========================================================================
# 18. Atomic write
# ===========================================================================
def test_atomic_write_leaves_no_temp_file(tmp_path):
    target = tmp_path / "out.json"
    m._atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert target.exists()
    assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []


# ===========================================================================
# Wrong P9 hash / status / source / weights mismatches
# ===========================================================================
def test_wrong_p9_report_hash_stops_before_model_import(tmp_path, fake_identity):
    p9_path = tmp_path / "gate_report_p9.json"
    _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                     weights_sha=fake_identity["weights_sha"])
    wrong_sha = "0" * 64
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(tmp_path / "nonexistent_dataset_root", tmp_path / "out",
                   fake_identity, p9_path, wrong_sha, fake, load_flag)

    assert rc != 0
    assert load_flag["loaded"] is False
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kwargs", [
    {"conclusion": "STEP_2B_P9_CORRIGENDUM_FAIL_GATE_1"},
    {"verdict_conclusion": "SOMETHING_ELSE"},
    {"verdict_norms_status": "BLOCKED"},
    {"meta_norms_status": "BLOCKED"},
    {"negative_control_status": "NON_DISCRIMINATIVE"},
    {"verdict_overrides": {"independent_inference_performed": True}},
    {"verdict_overrides": {"source_sha_match": False}},
    {"verdict_overrides": {"weights_sha_match": False}},
    {"verdict_overrides": {"p8_historical_result_preserved": False}},
])
def test_p9_status_other_than_pass_unblocked_stops(tmp_path, fake_identity,
                                                   monkeypatch, kwargs):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha=fake_identity["weights_sha"], **kwargs)
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    with pytest.raises(m.SaliencyNormsError):
        m.run_p9_release_preflight(
            p9_report_path=p9_path, p9_report_sha=p9_sha,
            production_source_path=fake_identity["source_path"],
            weights_path=fake_identity["weights_path"],
            predictor_module_path=fake_identity["source_path"],
            extractor_module_path=m._EXTRACTOR_MODULE_PATH,
        )


def test_source_mismatch_stops_before_model_import(tmp_path, fake_identity, monkeypatch):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(
        p9_path, source_sha="a" * 64, weights_sha=fake_identity["weights_sha"])
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    with pytest.raises(m.SaliencyNormsError, match="source"):
        m.run_p9_release_preflight(
            p9_report_path=p9_path, p9_report_sha=p9_sha,
            production_source_path=fake_identity["source_path"],
            weights_path=fake_identity["weights_path"],
            predictor_module_path=fake_identity["source_path"],
            extractor_module_path=m._EXTRACTOR_MODULE_PATH,
        )


def test_weights_mismatch_stops_before_model_import_at_main_level(
        tmp_path, fake_identity, full_corpus_root, monkeypatch):
    p9_path = tmp_path / "gate_report_p9.json"
    p9_sha = _build_p9_report(p9_path, source_sha=fake_identity["source_sha"],
                              weights_sha="b" * 64)
    monkeypatch.setattr(m, "FROZEN_P9_REPORT_SHA256", p9_sha)
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity, p9_path, p9_sha,
                   fake, load_flag)

    assert rc != 0
    assert load_flag["loaded"] is False
    assert not output_dir.exists()


# ===========================================================================
# --preflight-only imports nothing and writes nothing
# ===========================================================================
def test_preflight_only_imports_nothing_and_writes_nothing(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}

    before_modules = set(sys.modules)
    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag, extra_argv=["--preflight-only"])
    after_modules = set(sys.modules)
    newly_loaded_modules = after_modules - before_modules

    assert rc == 0
    assert load_flag["loaded"] is False
    assert not output_dir.exists()
    assert fake.calls == []
    forbidden = [
        name for name in newly_loaded_modules
        if name in ("tensorflow", "keras", "tf_keras", "saliency.umsi_model")
        or name.startswith("tensorflow.")
        or name.startswith("keras.")
        or name.startswith("tf_keras.")
    ]
    assert not forbidden, (
        f"--preflight-only newly loaded forbidden modules: {forbidden}")


# ===========================================================================
# An interrupted row is not marked complete
# ===========================================================================
def test_interrupted_row_not_marked_complete(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    output_dir = tmp_path / "out"
    fake = _FakePredictor(fail_at_call=3)
    load_flag: dict = {"loaded": False}

    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)

    assert rc != 0
    with open(output_dir / "saliency_feature_rows.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["global_index"] for r in rows} == {"1", "2"}
    sidecar = json.loads((output_dir / "saliency_feature_rows.csv.provenance.json").read_text())
    assert sidecar["last_failure"]["global_index"] == 3


# ===========================================================================
# Existing visual / combined norm files remain byte-identical
# ===========================================================================
def test_existing_visual_and_combined_norm_files_unchanged(
        tmp_path, fake_identity, full_corpus_root, frozen_synthetic_p9):
    def _hash_if_exists(p: Path) -> Optional[str]:
        return m._sha256_file(p) if p.exists() else None

    before_feature_norms = _hash_if_exists(_REAL_FEATURE_NORMS_JSON)
    before_visual_norms = _hash_if_exists(_REAL_VISUAL_NORMS_JSON)

    output_dir = tmp_path / "out"
    fake = _FakePredictor()
    load_flag: dict = {"loaded": False}
    rc = _run_main(full_corpus_root, output_dir, fake_identity,
                   frozen_synthetic_p9["path"], frozen_synthetic_p9["sha"],
                   fake, load_flag)
    assert rc == 0

    after_feature_norms = _hash_if_exists(_REAL_FEATURE_NORMS_JSON)
    after_visual_norms = _hash_if_exists(_REAL_VISUAL_NORMS_JSON)
    assert before_feature_norms == after_feature_norms
    assert before_visual_norms == after_visual_norms


# ===========================================================================
# No real UMSI, TensorFlow or Keras import/inference occurs in the suite
# ===========================================================================
def test_no_real_umsi_tensorflow_keras_imported_by_this_suite():
    """Importing the canonical module alone must not activate TensorFlow/Keras.

    Checking ``sys.modules`` in the shared pytest process is order-dependent:
    other test files in the same session (for example
    ``test_umsi_boundary_input_probe.py``) legitimately import
    ``saliency.umsi_model`` (and therefore TensorFlow/Keras) in-process. This
    check therefore runs in a fresh subprocess with a pristine ``sys.modules``
    and verifies only that importing this file's module under test
    (``stage1.tools.canonical_saliency_norms``) does not, by itself, load any
    real TensorFlow/Keras/tf_keras module or ``saliency.umsi_model``.
    """
    script = (
        "import sys\n"
        "import stage1.tools.canonical_saliency_norms\n"
        "forbidden = [m for m in sys.modules "
        "if m in ('tensorflow', 'keras', 'tf_keras', 'saliency.umsi_model') "
        "or m.startswith('tensorflow.') "
        "or m.startswith('keras.') "
        "or m.startswith('tf_keras.')]\n"
        "assert not forbidden, forbidden\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}")
    assert "OK" in result.stdout
