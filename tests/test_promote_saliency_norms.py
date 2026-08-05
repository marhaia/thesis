"""Tests for ``stage1/tools/promote_saliency_norms.py``.

These tests use small synthetic fixtures (6 images: 2 desktop, 2 mobile,
2 web) built entirely in a temp directory -- never the real 1,485-image
production corpus, never real UMSI weights, never TensorFlow. The module's
``EXPECTED_TOTAL`` / ``EXPECTED_CATEGORY_COUNTS`` contract constants are
monkeypatched down to this small size for every test in this file.

Golden statistics (``GOLDEN_FEATURE_VALUE`` / ``GOLDEN_NORMS`` below) are
constructed WITHOUT calling ``psn.recompute_norms`` -- the production
tool's own aggregation function. Every synthetic image is assigned the
SAME constant value per feature, which makes every statistic (mean, std,
min, max, all five percentiles) trivially and exactly hand-verifiable:
mean == the constant, std == 0.0 exactly (zero variance), and every
percentile == the constant exactly (no floating-point interpolation ever
occurs between two different values). This sidesteps any risk of the
golden fixture disagreeing with ``promote_saliency_norms.py``'s exact
(``==``) floating-point comparison due to summation-order or library
differences, while remaining fully independent of the code under test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

import stage1.tools.promote_saliency_norms as psn

FEATURES: List[str] = list(psn.SALIENCY_FEATURE_KEYS)
CATS: List[str] = ["desktop", "mobile", "web"]
SMALL_COUNTS: Dict[str, int] = {"desktop": 2, "mobile": 2, "web": 2}
SMALL_TOTAL = 6

# Independent golden expectation (see module docstring): one constant,
# exactly-representable-in-binary value per feature, identical across all
# 6 synthetic images. This is NOT computed via psn.recompute_norms().
GOLDEN_FEATURE_VALUE: Dict[str, float] = {
    "saliency_dispersion": 0.5,
    "saliency_peak_count": 4.0,
    "saliency_center_bias": 0.25,
    "saliency_entropy": 0.625,
    "saliency_coverage": 0.03125,
}
GOLDEN_NORMS: Dict[str, Dict[str, float]] = {
    feat: {
        "n": SMALL_TOTAL, "mean": val, "std": 0.0, "min": val, "max": val,
        "p5": val, "p25": val, "p50": val, "p75": val, "p95": val,
    }
    for feat, val in GOLDEN_FEATURE_VALUE.items()
}

# ---------------------------------------------------------------------------
# Independent, non-constant golden expectation (Phase B hardening).
#
# A constant-value fixture (GOLDEN_FEATURE_VALUE above) cannot distinguish
# ddof=0 from ddof=1 (variance of a constant is 0 either way) and cannot
# distinguish the "linear" percentile/quantile interpolation method from any
# other method (every percentile of a constant array equals that constant
# regardless of interpolation). The six per-row values below are chosen to
# be non-constant and heavily skewed (one outlier per feature) specifically
# so that:
#   * ddof=0 and ddof=1 sample standard deviation differ substantially, and
#   * the "linear" quantile method disagrees with alternative methods
#     (e.g. "lower") at p25 and p75.
#
# These six-value lists (in row/global_index order 1..6: desktop, desktop,
# mobile, mobile, web, web) and every statistic below were precomputed ONCE,
# offline, directly with numpy (never via psn.recompute_norms() or
# canonical_saliency_norms.py's aggregate() at test-run time) and are
# hardcoded here as fixed literals:
#
#   saliency_dispersion:   [1, 2, 3, 4, 5, 100]
#   saliency_peak_count:   [2, 4, 6, 8, 10, 200]
#   saliency_center_bias:  [0.1, 0.2, 0.3, 0.4, 0.5, 10.0]
#   saliency_entropy:      [10, 20, 30, 40, 50, 1000]
#   saliency_coverage:     [0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
#
# For n=6, mean=115/6=19.1666..., the ddof=1 sample variance is
# sum((x-mean)**2)/5 = 1570.1666...; ddof=0 (population) variance divides
# by 6 instead of 5 -- std differs (39.625... vs 36.173... for the first
# feature) at every decimal place actually written below, so a norms
# artifact using ddof=0 can never accidentally match.
NONCONSTANT_FEATURE_VALUES: Dict[str, List[float]] = {
    "saliency_dispersion": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
    "saliency_peak_count": [2.0, 4.0, 6.0, 8.0, 10.0, 200.0],
    "saliency_center_bias": [0.1, 0.2, 0.3, 0.4, 0.5, 10.0],
    "saliency_entropy": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0],
    "saliency_coverage": [0.01, 0.02, 0.03, 0.04, 0.05, 1.0],
}

# Independent golden expectation: mean, ddof=1 std, min/max, and the 5
# "linear"-interpolation percentiles, precomputed by hand/offline (see
# above), for each of the 5 non-constant feature arrays.
GOLDEN_NONCONSTANT_NORMS: Dict[str, Dict[str, float]] = {
    "saliency_dispersion": {
        "n": SMALL_TOTAL, "mean": 19.166666666666668, "std": 39.62532860010963,
        "min": 1.0, "max": 100.0,
        "p5": 1.25, "p25": 2.25, "p50": 3.5, "p75": 4.75, "p95": 76.25,
    },
    "saliency_peak_count": {
        "n": SMALL_TOTAL, "mean": 38.333333333333336, "std": 79.25065720021927,
        "min": 2.0, "max": 200.0,
        "p5": 2.5, "p25": 4.5, "p50": 7.0, "p75": 9.5, "p95": 152.5,
    },
    "saliency_center_bias": {
        "n": SMALL_TOTAL, "mean": 1.9166666666666667, "std": 3.9625328600109637,
        "min": 0.1, "max": 10.0,
        "p5": 0.125, "p25": 0.225, "p50": 0.35, "p75": 0.475, "p95": 7.625,
    },
    "saliency_entropy": {
        "n": SMALL_TOTAL, "mean": 191.66666666666666, "std": 396.2532860010964,
        "min": 10.0, "max": 1000.0,
        "p5": 12.5, "p25": 22.5, "p50": 35.0, "p75": 47.5, "p95": 762.5,
    },
    "saliency_coverage": {
        "n": SMALL_TOTAL, "mean": 0.19166666666666665, "std": 0.3962532860010964,
        "min": 0.01, "max": 1.0,
        "p5": 0.0125, "p25": 0.0225, "p50": 0.035, "p75": 0.0475, "p95": 0.7625,
    },
}

# WRONG variant #1: every field identical to the golden expectation above
# EXCEPT "std", which uses the ddof=0 (population) standard deviation
# instead of the required ddof=1 (sample) standard deviation. A norms
# artifact carrying this value must be rejected by norms_math_gate.
GOLDEN_NONCONSTANT_NORMS_WRONG_DDOF0: Dict[str, Dict[str, float]] = {
    "saliency_dispersion": {**GOLDEN_NONCONSTANT_NORMS["saliency_dispersion"], "std": 36.17281053805775},
    "saliency_peak_count": {**GOLDEN_NONCONSTANT_NORMS["saliency_peak_count"], "std": 72.3456210761155},
    "saliency_center_bias": {**GOLDEN_NONCONSTANT_NORMS["saliency_center_bias"], "std": 3.617281053805776},
    "saliency_entropy": {**GOLDEN_NONCONSTANT_NORMS["saliency_entropy"], "std": 361.7281053805776},
    "saliency_coverage": {**GOLDEN_NONCONSTANT_NORMS["saliency_coverage"], "std": 0.36172810538057754},
}

# WRONG variant #2: every field identical to the golden expectation above
# EXCEPT "p25"/"p75", which use the "lower" quantile-interpolation method
# instead of the required "linear" method. A norms artifact carrying these
# values must be rejected by norms_math_gate.
GOLDEN_NONCONSTANT_NORMS_WRONG_QUANTILE: Dict[str, Dict[str, float]] = {
    "saliency_dispersion": {**GOLDEN_NONCONSTANT_NORMS["saliency_dispersion"], "p25": 2.0, "p75": 4.0},
    "saliency_peak_count": {**GOLDEN_NONCONSTANT_NORMS["saliency_peak_count"], "p25": 4.0, "p75": 8.0},
    "saliency_center_bias": {**GOLDEN_NONCONSTANT_NORMS["saliency_center_bias"], "p25": 0.2, "p75": 0.4},
    "saliency_entropy": {**GOLDEN_NONCONSTANT_NORMS["saliency_entropy"], "p25": 20.0, "p75": 40.0},
    "saliency_coverage": {**GOLDEN_NONCONSTANT_NORMS["saliency_coverage"], "p25": 0.02, "p75": 0.04},
}


@pytest.fixture(autouse=True)
def small_contract(monkeypatch):
    """Shrink the production corpus-size contract to a fast, synthetic size."""
    monkeypatch.setattr(psn, "EXPECTED_TOTAL", SMALL_TOTAL)
    monkeypatch.setattr(psn, "EXPECTED_CATEGORY_COUNTS", dict(SMALL_COUNTS))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_good_environment(tmp_path: Path, *,
                            per_row_feature_values: Optional[Dict[str, List[float]]] = None,
                            norms_features_override: Optional[Dict[str, Dict[str, float]]] = None
                            ) -> Dict[str, Any]:
    """Build a fully self-consistent, valid set of promotion inputs.

    Includes a real on-disk ``dataset_root/images/`` directory with 6 real
    (arbitrary-content) files, so the real corpus verification (Phase 4b of
    the production tool) has real files to check. Returns a dict of paths/
    objects individual tests mutate to construct FAIL scenarios.

    By default every row gets the same constant ``GOLDEN_FEATURE_VALUE`` per
    feature (see module docstring). ``per_row_feature_values`` (a mapping of
    feature name -> list of 6 values, in row order matching global_index
    1..6) overrides this with non-constant, independently precomputed
    values; ``norms_features_override`` supplies the matching independent
    golden ``features`` block for the saliency norms artifact.
    """
    generator_path = tmp_path / "gen.py"
    extractor_path = tmp_path / "extractor.py"
    predictor_path = tmp_path / "predictor.py"
    weights_path = tmp_path / "weights.hdf5"
    _write_text(generator_path, "# dummy generator script\n")
    _write_text(extractor_path, "# dummy extractor script\n")
    _write_text(predictor_path, "# dummy predictor/source script\n")
    _write_bytes(weights_path, b"\x00fake-weights-content\x00" * 8)
    generator_sha = psn._sha256_file(generator_path)
    extractor_sha = psn._sha256_file(extractor_path)
    predictor_sha = psn._sha256_file(predictor_path)
    weights_sha = psn._sha256_file(weights_path)
    weights_size = weights_path.stat().st_size

    p9_report_path = tmp_path / "gate_report_p9.json"
    _write_text(p9_report_path, json.dumps({"conclusion": "STEP_2B_P9_CORRIGENDUM_PASS"}))
    p9_sha = psn._sha256_file(p9_report_path)

    env_info = {"python_version": "3.9.6", "platform": "test"}

    dataset_root = tmp_path / "dataset_root"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    corpus_records: List[Dict[str, Any]] = []
    gi = 1
    for cat in CATS:
        for j in range(SMALL_COUNTS[cat]):
            rel_path = f"{cat}_{j}.png"
            content = f"fake-image-bytes-{cat}-{j}-{gi}".encode()
            _write_bytes(images_dir / rel_path, content)
            image_sha = _sha256_bytes(content)
            native_width = 100 + gi
            native_height = 200 + gi

            row: Dict[str, Any] = {
                "global_index": gi,
                "category": cat,
                "relative_path": rel_path,
                "image_sha256": image_sha,
                "native_width": native_width,
                "native_height": native_height,
            }
            for feat in FEATURES:
                if per_row_feature_values is not None:
                    row[feat] = per_row_feature_values[feat][gi - 1]
                else:
                    row[feat] = GOLDEN_FEATURE_VALUE[feat]
            rows.append(row)

            corpus_records.append({
                "global_index": gi,
                "category": cat,
                "relative_path": rel_path,
                "image_sha256": image_sha,
                "native_width": native_width,
                "native_height": native_height,
            })
            gi += 1

    corpus_manifest_sha256 = hashlib.sha256(
        psn._canonical_json(corpus_records).encode("utf-8")).hexdigest()

    feature_order_str = "|".join(FEATURES)

    run_fingerprint = {
        "schema_version": psn.EXPECTED_SCHEMA_VERSION,
        "feature_order": list(FEATURES),
        "resolution_policy": psn.EXPECTED_RESOLUTION_POLICY,
        "authorized_categories": list(CATS),
        "excluded_categories": ["poster"],
        "authorized_category_counts": dict(SMALL_COUNTS),
        "authorized_total": SMALL_TOTAL,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "production_source_sha256": predictor_sha,
        "predictor_module_sha256": predictor_sha,
        "extractor_module_sha256": extractor_sha,
        "weights_sha256": weights_sha,
        "weights_byte_size": weights_size,
        "generator_sha256": generator_sha,
        "p9_report_sha256": p9_sha,
        "environment": env_info,
    }
    run_fp_sha = hashlib.sha256(psn._canonical_json(run_fingerprint).encode()).hexdigest()

    for row in rows:
        row["feature_order"] = feature_order_str
        row["run_fingerprint_sha256"] = run_fp_sha

    manifest = {
        "authorized_category_counts": dict(SMALL_COUNTS),
        "authorized_total": SMALL_TOTAL,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "environment": env_info,
        "feature_extractor_module_sha256": extractor_sha,
        "feature_order": list(FEATURES),
        "generator_sha256": generator_sha,
        "invocation_count": 1,
        "n_failed_images": 0,
        "p9_conclusion": "STEP_2B_P9_CORRIGENDUM_PASS",
        "p9_report_path": str(p9_report_path),
        "p9_report_sha256": p9_sha,
        "p9_saliency_norms_release_status": "UNBLOCKED",
        "predictor_module_sha256": predictor_sha,
        "production_umsi_source_sha256": predictor_sha,
        "recovered_from_interruption": False,
        "resolution_policy": psn.EXPECTED_RESOLUTION_POLICY,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": run_fp_sha,
        "schema_version": psn.EXPECTED_SCHEMA_VERSION,
        "segments": [{"status": "completed", "incomplete": False}],
        "source_git_commit": "test",
        "weights_byte_size": weights_size,
        "weights_sha256": weights_sha,
    }

    sidecar = {
        "status": "completed",
        "last_failure": None,
        "run_fingerprint_sha256": run_fp_sha,
        "run_fingerprint": run_fingerprint,
    }

    norms_artifact = {
        "feature_order": list(FEATURES),
        "resolution_policy": psn.EXPECTED_RESOLUTION_POLICY,
        "schema_version": psn.EXPECTED_SCHEMA_VERSION,
        "num_images": SMALL_TOTAL,
        "features": {
            feat: dict(stats) for feat, stats in
            (norms_features_override if norms_features_override is not None else GOLDEN_NORMS).items()
        },
    }

    base_feature_norms = {
        "meta": {
            "source": "test corpus",
            "categories_included": list(CATS),
            "categories_excluded": ["poster"],
            "num_images": SMALL_TOTAL,
            "category_counts": dict(SMALL_COUNTS),
            "includes_saliency": True,
            "is_subsample": False,
            "regenerated_note": "prior note",
            "visual_norms_provenance": {
                "authorized_corpus_aggregate_sha256": "d" * 64,
                "canonical_long_side": 1280,
            },
        },
        "features": {
            vf: {"n": SMALL_TOTAL, "mean": 1.0, "std": 0.1, "min": 0.0, "max": 2.0,
                 "p5": 0.1, "p25": 0.5, "p50": 1.0, "p75": 1.5, "p95": 1.9}
            for vf in psn.VISUAL_FEATURE_KEYS
        },
    }

    manifest_path = tmp_path / "saliency_feature_norms_manifest.json"
    sidecar_path = tmp_path / "saliency_feature_rows.csv.provenance.json"
    norms_path = tmp_path / "saliency_feature_norms.json"
    csv_path = tmp_path / "saliency_feature_rows.csv"
    base_path = tmp_path / "feature_norms.base.json"

    _write_text(manifest_path, json.dumps(manifest))
    _write_text(sidecar_path, json.dumps(sidecar))
    _write_text(norms_path, json.dumps(norms_artifact))
    _write_text(base_path, json.dumps(base_feature_norms))
    write_csv(csv_path, rows)

    return {
        "manifest_path": manifest_path,
        "sidecar_path": sidecar_path,
        "norms_path": norms_path,
        "csv_path": csv_path,
        "base_path": base_path,
        "generator_path": generator_path,
        "extractor_path": extractor_path,
        "predictor_path": predictor_path,
        "weights_path": weights_path,
        "p9_report_path": p9_report_path,
        "dataset_root": dataset_root,
        "images_dir": images_dir,
        "manifest": manifest,
        "sidecar": sidecar,
        "norms_artifact": norms_artifact,
        "base_feature_norms": base_feature_norms,
        "rows": rows,
        "corpus_manifest_sha256": corpus_manifest_sha256,
    }


def write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["global_index", "category", "relative_path", "image_sha256",
                  "native_width", "native_height", *FEATURES,
                  "feature_order", "run_fingerprint_sha256"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_argv(env: Dict[str, Any], output_path: Path, *,
               apply_path: Optional[Path] = None,
               dataset_root: Optional[Path] = None,
               weights_path: Optional[Path] = None,
               p9_report_path: Optional[Path] = None,
               omit: Optional[List[str]] = None) -> List[str]:
    """Build a full, valid argv list for ``psn.main()``.

    ``omit`` names CLI flags (without the leading ``--``, e.g. ``"weights"``)
    to leave out entirely -- used to test that required identity paths
    cannot be silently skipped.
    """
    omit_set = set(omit or [])
    parts: List[Tuple[str, str]] = [
        ("base-feature-norms", str(env["base_path"])),
        ("saliency-norms", str(env["norms_path"])),
        ("saliency-manifest", str(env["manifest_path"])),
        ("saliency-csv", str(env["csv_path"])),
        ("saliency-sidecar", str(env["sidecar_path"])),
        ("dataset-root", str(dataset_root if dataset_root is not None else env["dataset_root"])),
        ("output", str(output_path)),
        ("generator-script", str(env["generator_path"])),
        ("extractor-module", str(env["extractor_path"])),
        ("predictor-module", str(env["predictor_path"])),
        ("weights", str(weights_path if weights_path is not None else env["weights_path"])),
        ("p9-report", str(p9_report_path if p9_report_path is not None else env["p9_report_path"])),
    ]
    argv: List[str] = []
    for name, value in parts:
        if name in omit_set:
            continue
        argv += [f"--{name}", value]
    if apply_path is not None:
        argv += ["--apply", str(apply_path)]
    return argv


def run_tool(env: Dict[str, Any], tmp_path: Path, output_name: str = "candidate.json",
             apply_path: Optional[Path] = None, **kwargs):
    output_path = tmp_path / output_name
    argv = build_argv(env, output_path, apply_path=apply_path, **kwargs)
    exit_code = psn.main(argv)
    return exit_code, output_path


# ---------------------------------------------------------------------------
# 1. Full successful candidate run (also exercises the real weights-hash
#    verification path -- --weights is always mandatory now, no [SKIP]).
# ---------------------------------------------------------------------------
def test_full_successful_run(tmp_path, capsys):
    env = build_good_environment(tmp_path)
    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    assert output_path.is_file()

    captured = capsys.readouterr()
    assert "weights_sha256" in captured.out
    assert "[SKIP]" not in captured.out

    candidate = json.loads(output_path.read_text())
    for feat in FEATURES:
        assert candidate["features"][feat] == env["norms_artifact"]["features"][feat]

    validation = candidate["meta"]["saliency_norms_provenance"]["validation"]
    assert validation["corpus_identity_gate"] == "PASS"
    assert validation["norms_math_gate"] == "PASS"
    assert validation["exact_stat_matches"] == "50/50"
    assert validation["tolerance_mismatches"] == 0
    assert candidate["meta"]["saliency_norms_provenance"]["processed_count"] == SMALL_TOTAL
    assert candidate["meta"]["saliency_norms_provenance"]["processed_category_counts"] == SMALL_COUNTS


# ---------------------------------------------------------------------------
# 1b. Honest hash logging: source_artifact_sha256 / source_csv_sha256 have no
#     independent historical anchor (documented schema limitation) and must
#     therefore be logged as [INFO] observed/recomputed-only, NEVER as
#     [PASS]/"verified" -- that wording is reserved for checks that compare
#     against an independent, pre-existing expectation.
# ---------------------------------------------------------------------------
def test_source_artifact_and_csv_hashes_logged_as_info_not_pass(tmp_path, capsys):
    env = build_good_environment(tmp_path)
    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    assert output_path.is_file()

    captured = capsys.readouterr()
    assert "[INFO] source_artifact_sha256 observed" in captured.out
    assert "[INFO] source_csv_sha256 observed" in captured.out
    assert "[PASS] source_artifact_sha256" not in captured.out
    assert "[PASS] source_csv_sha256" not in captured.out
    assert "recomputed only" in captured.out or "not independently verified" in captured.out

    # The candidate's provenance block may still store the hashes (as plain
    # observed values), but nothing in the candidate itself may claim they
    # were "PASS"/"verified" against an independent anchor.
    candidate = json.loads(output_path.read_text())
    validation = candidate["meta"]["saliency_norms_provenance"]["validation"]
    assert "source_artifact_sha256" not in validation
    assert "source_csv_sha256" not in validation
    assert candidate["meta"]["saliency_norms_provenance"]["source_artifact_sha256"]
    assert candidate["meta"]["saliency_norms_provenance"]["source_csv_sha256"]


# ---------------------------------------------------------------------------
# 2. Hash mismatch (generator script tampered after manifest was built)
# ---------------------------------------------------------------------------
def test_generator_hash_mismatch_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    env["generator_path"].write_text("# tampered generator content\n")

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 2b. Extractor / predictor / weights / P9-report hash mismatch (dedicated,
#     independent tests for each of the four other mandatory identities).
# ---------------------------------------------------------------------------
def test_extractor_hash_mismatch_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    env["extractor_path"].write_text("# tampered extractor content\n")

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_predictor_hash_mismatch_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    env["predictor_path"].write_text("# tampered predictor content\n")

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_weights_hash_mismatch_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    env["weights_path"].write_bytes(b"\x00tampered-weights-content\x00" * 8)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_p9_report_hash_mismatch_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    env["p9_report_path"].write_text(json.dumps({"conclusion": "TAMPERED"}))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 2c. Missing required identity path: --weights, --p9-report and
#     --dataset-root have no fallback default and are declared
#     ``required=True`` -- omitting them can never be silently accepted,
#     argparse itself enforces this with a SystemExit.
#     (--generator-script / --extractor-module / --predictor-module instead
#     fall back to real, non-None default paths inside the repository --
#     see test_generator_hash_mismatch_fails_closed et al. for the fail-
#     closed behaviour exercised via the ordinary hash-mismatch path.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["weights", "p9-report", "dataset-root"])
def test_missing_required_identity_path_rejected(tmp_path, flag):
    env = build_good_environment(tmp_path)
    output_path = tmp_path / "candidate.json"
    argv = build_argv(env, output_path, omit=[flag])

    with pytest.raises(SystemExit) as exc_info:
        psn.main(argv)
    assert exc_info.value.code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 3. Missing CSV row
# ---------------------------------------------------------------------------
def test_missing_csv_row_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    rows = env["rows"][:-1]  # drop the last row -> only 5 of 6
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 4. Duplicate CSV row (duplicate global_index / relative_path)
# ---------------------------------------------------------------------------
def test_duplicate_csv_row_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    # Overwrite row[1]'s identity fields with row[0]'s -- still 6 physical
    # rows, but a genuine duplicate global_index/relative_path/image_sha256.
    rows[1]["global_index"] = rows[0]["global_index"]
    rows[1]["relative_path"] = rows[0]["relative_path"]
    rows[1]["image_sha256"] = rows[0]["image_sha256"]
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 5. Wrong category count / extra category / poster row
# ---------------------------------------------------------------------------
def test_wrong_category_count_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    # Relabel one "web" row as "desktop" -> desktop=3, web=1, total still 6.
    for row in rows:
        if row["category"] == "web":
            row["category"] = "desktop"
            break
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_poster_row_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    rows[0]["category"] = "poster"
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_extra_unknown_category_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    rows[0]["category"] = "banner"
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 6. NaN / Inf rejected (CSV) and JSON (norms artifact statistics)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_non_finite_feature_value_in_csv_fails_closed(tmp_path, bad_value):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    rows[0][FEATURES[0]] = bad_value
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_json_nonfinite_literal_in_norms_artifact_rejected(tmp_path, literal):
    env = build_good_environment(tmp_path)
    # Inject a raw (unquoted) NaN/Infinity/-Infinity JSON token -- Python's
    # json module accepts these non-standard literals by default; the
    # strict loader must reject them via parse_constant.
    raw = env["norms_path"].read_text()
    target = f'"mean": {GOLDEN_FEATURE_VALUE[FEATURES[0]]}'
    replacement = f'"mean": {literal}'
    assert target in raw, "test setup: expected substring not found"
    tampered = raw.replace(target, replacement, 1)
    env["norms_path"].write_text(tampered)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 6b. bool instead of a numeric statistic
# ---------------------------------------------------------------------------
def test_bool_instead_of_number_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    raw = env["norms_path"].read_text()
    target = f'"mean": {GOLDEN_FEATURE_VALUE[FEATURES[0]]}'
    assert target in raw, "test setup: expected substring not found"
    tampered = raw.replace(target, '"mean": true', 1)
    env["norms_path"].write_text(tampered)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 7. Missing / unexpected feature in the saliency norms artifact
# ---------------------------------------------------------------------------
def test_missing_feature_in_norms_artifact_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    norms_artifact = json.loads(env["norms_path"].read_text())
    del norms_artifact["features"][FEATURES[0]]
    env["norms_path"].write_text(json.dumps(norms_artifact))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_unexpected_feature_in_norms_artifact_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    norms_artifact = json.loads(env["norms_path"].read_text())
    norms_artifact["features"]["saliency_bogus_extra"] = dict(
        norms_artifact["features"][FEATURES[0]])
    env["norms_path"].write_text(json.dumps(norms_artifact))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 8. Duplicate JSON object keys (top-level and nested) must be rejected at
#    parse time -- NOT silently collapsed by json.load.
# ---------------------------------------------------------------------------
def test_duplicate_json_key_top_level_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    # Hand-craft raw JSON text with a duplicated top-level key; json.dumps()
    # can never produce this, so we write it directly as text.
    tampered = (
        '{"feature_order": ["a"], "feature_order": ["b"], '
        '"resolution_policy": "NATIVE_INPUT", "schema_version": "x", '
        '"num_images": 6, "features": {}}'
    )
    env["norms_path"].write_text(tampered)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_duplicate_json_key_nested_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    parsed = json.loads(env["norms_path"].read_text())
    first_feat = FEATURES[0]
    # Hand-craft a norms artifact whose nested per-feature stats object has
    # a duplicated "mean" key -- explicit, unambiguous text construction
    # (does not depend on json.dumps' key-ordering behavior).
    tampered = (
        '{"feature_order": ' + json.dumps(parsed["feature_order"]) + ', '
        '"resolution_policy": ' + json.dumps(parsed["resolution_policy"]) + ', '
        '"schema_version": ' + json.dumps(parsed["schema_version"]) + ', '
        '"num_images": ' + json.dumps(parsed["num_images"]) + ', '
        '"features": {"' + first_feat + '": {"mean": 1.0, "mean": 2.0}}}'
    )
    env["norms_path"].write_text(tampered)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 9. Manipulated norm statistic
# ---------------------------------------------------------------------------
def test_manipulated_statistic_fails_closed(tmp_path):
    env = build_good_environment(tmp_path)
    norms_artifact = json.loads(env["norms_path"].read_text())
    norms_artifact["features"][FEATURES[0]]["mean"] += 1.0
    env["norms_path"].write_text(json.dumps(norms_artifact))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 9b. Independent, non-constant golden statistic test (Phase B hardening).
#
# Unlike GOLDEN_FEATURE_VALUE/GOLDEN_NORMS (constant per feature -- variance
# is trivially 0 regardless of ddof, and every percentile trivially equals
# the single value regardless of interpolation method), this test uses six
# genuinely different, skewed per-row values per feature
# (NONCONSTANT_FEATURE_VALUES) and an independently, offline-precomputed
# golden expectation (GOLDEN_NONCONSTANT_NORMS -- see the module-level
# comment above those constants for the exact input values and how each
# statistic was derived). None of these expected values were produced by
# calling psn.recompute_norms() or canonical_saliency_norms.py's aggregate()
# at test-run time.
#
# This test runs the real CLI/integration entry point (psn.main(), via
# run_tool()) end-to-end, not a bare call to an internal helper function.
# ---------------------------------------------------------------------------
def test_nonconstant_golden_statistics_full_successful_run(tmp_path):
    env = build_good_environment(
        tmp_path,
        per_row_feature_values=NONCONSTANT_FEATURE_VALUES,
        norms_features_override=GOLDEN_NONCONSTANT_NORMS,
    )

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    assert output_path.is_file()

    candidate = json.loads(output_path.read_text())
    for feat in FEATURES:
        golden = GOLDEN_NONCONSTANT_NORMS[feat]
        got = candidate["features"][feat]
        assert got["n"] == golden["n"]
        for stat in ("mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95"):
            assert got[stat] == golden[stat], (
                f"{feat}.{stat}: candidate={got[stat]!r} independent golden={golden[stat]!r}")

    validation = candidate["meta"]["saliency_norms_provenance"]["validation"]
    assert validation["norms_math_gate"] == "PASS"
    assert validation["exact_stat_matches"] == "50/50"
    assert validation["tolerance_mismatches"] == 0


# ---------------------------------------------------------------------------
# 9c. Negative variants: a norms artifact using ddof=0 instead of ddof=1, or
#     an alternative quantile/interpolation method instead of "linear", must
#     both be rejected by norms_math_gate -- the non-constant input values
#     make these two mistakes actually observable (see 9b above), unlike a
#     constant-value fixture where they would be indistinguishable.
# ---------------------------------------------------------------------------
def test_nonconstant_golden_wrong_ddof0_std_rejected(tmp_path):
    env = build_good_environment(
        tmp_path,
        per_row_feature_values=NONCONSTANT_FEATURE_VALUES,
        norms_features_override=GOLDEN_NONCONSTANT_NORMS_WRONG_DDOF0,
    )

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_nonconstant_golden_wrong_quantile_method_rejected(tmp_path):
    env = build_good_environment(
        tmp_path,
        per_row_feature_values=NONCONSTANT_FEATURE_VALUES,
        norms_features_override=GOLDEN_NONCONSTANT_NORMS_WRONG_QUANTILE,
    )

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 10. The eight visual feature blocks are protected (byte-for-byte untouched)
# ---------------------------------------------------------------------------
def test_visual_feature_blocks_are_left_untouched(tmp_path):
    env = build_good_environment(tmp_path)
    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    candidate = json.loads(output_path.read_text())
    for vf in psn.VISUAL_FEATURE_KEYS:
        assert candidate["features"][vf] == env["base_feature_norms"]["features"][vf]
    assert candidate["meta"]["visual_norms_provenance"] == \
        env["base_feature_norms"]["meta"]["visual_norms_provenance"]


# ---------------------------------------------------------------------------
# 11. No TensorFlow / UMSI import activation (subprocess-isolated: sys.modules
#     state in the shared pytest process is order-dependent and cannot be
#     trusted in-process).
# ---------------------------------------------------------------------------
def test_no_tensorflow_or_umsi_import_activation():
    script = (
        "import sys\n"
        "import stage1.tools.promote_saliency_norms\n"
        "forbidden = [m for m in sys.modules "
        "if m == 'tensorflow' or m.startswith('tensorflow.') "
        "or m == 'saliency.umsi_model' or m == 'cv2']\n"
        "assert not forbidden, forbidden\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}")
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 12. No write at all without --apply (beyond the explicit --output candidate)
# ---------------------------------------------------------------------------
def test_no_target_write_without_apply(tmp_path):
    env = build_good_environment(tmp_path)
    apply_target = tmp_path / "would_be_target.json"
    sentinel = json.dumps({"sentinel": True})
    apply_target.write_text(sentinel)

    # Note: apply_path is intentionally NOT passed to run_tool.
    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    assert output_path.is_file()
    # The untouched sentinel file must remain byte-identical.
    assert apply_target.read_text() == sentinel


# ---------------------------------------------------------------------------
# 13. Atomic write on explicit --apply
# ---------------------------------------------------------------------------
def test_atomic_apply_writes_target(tmp_path):
    env = build_good_environment(tmp_path)
    apply_target = tmp_path / "feature_norms.target.json"
    apply_target.write_text(json.dumps({"sentinel": "old-content"}))

    exit_code, output_path = run_tool(env, tmp_path, apply_path=apply_target)

    assert exit_code == 0
    candidate = json.loads(output_path.read_text())
    applied = json.loads(apply_target.read_text())
    assert applied == candidate

    leftover_tmp_files = list(tmp_path.glob(".tmp_promote_*"))
    assert leftover_tmp_files == []


# ---------------------------------------------------------------------------
# 14. Fail-closed: apply target is never partially written on failure
# ---------------------------------------------------------------------------
def test_fail_closed_leaves_apply_target_untouched(tmp_path):
    env = build_good_environment(tmp_path)
    # Break the run via a manipulated statistic (norms_math_gate failure).
    norms_artifact = json.loads(env["norms_path"].read_text())
    norms_artifact["features"][FEATURES[0]]["mean"] += 1.0
    env["norms_path"].write_text(json.dumps(norms_artifact))

    apply_target = tmp_path / "feature_norms.target.json"
    original_content = json.dumps({"sentinel": "untouched-original"})
    apply_target.write_text(original_content)

    exit_code, output_path = run_tool(env, tmp_path, apply_path=apply_target)

    assert exit_code != 0
    assert not output_path.exists()
    assert apply_target.read_text() == original_content

    leftover_tmp_files = list(tmp_path.glob(".tmp_promote_*"))
    assert leftover_tmp_files == []


# ---------------------------------------------------------------------------
# 15. Real corpus verification: missing image file, tampered image content,
#     unsafe relative_path, and corpus_manifest_sha256 mismatch.
# ---------------------------------------------------------------------------
def test_missing_real_image_file_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    victim_rel_path = env["rows"][0]["relative_path"]
    (env["images_dir"] / victim_rel_path).unlink()

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_tampered_image_content_hash_mismatch_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    victim_rel_path = env["rows"][0]["relative_path"]
    (env["images_dir"] / victim_rel_path).write_bytes(b"tampered-image-bytes-do-not-match-csv")

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_unsafe_relative_path_traversal_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    rows[0]["relative_path"] = "../escape.png"
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_unsafe_absolute_relative_path_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    rows = [dict(r) for r in env["rows"]]
    rows[0]["relative_path"] = "/etc/passwd"
    write_csv(env["csv_path"], rows)

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_corpus_manifest_hash_mismatch_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    manifest = json.loads(env["manifest_path"].read_text())
    manifest["corpus_manifest_sha256"] = "f" * 64
    env["manifest_path"].write_text(json.dumps(manifest))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 15b. Corpus-hash triple-consistency (manifest.corpus_manifest_sha256 /
#      manifest.run_fingerprint.corpus_manifest_sha256 /
#      sidecar.run_fingerprint.corpus_manifest_sha256): all three must be
#      present and identical BEFORE the real, file-backed reconstruction is
#      even attempted. In particular, tampering ONLY the nested
#      sidecar.run_fingerprint.corpus_manifest_sha256 (without touching
#      sidecar.run_fingerprint_sha256) is otherwise invisible to the
#      run_fingerprint_sha256 self-consistency check -- these dedicated
#      tests close exactly that gap.
# ---------------------------------------------------------------------------
def test_manifest_run_fingerprint_corpus_hash_tampered_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    manifest = json.loads(env["manifest_path"].read_text())
    manifest["run_fingerprint"]["corpus_manifest_sha256"] = "1" * 64
    env["manifest_path"].write_text(json.dumps(manifest))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_sidecar_run_fingerprint_corpus_hash_tampered_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    sidecar = json.loads(env["sidecar_path"].read_text())
    sidecar["run_fingerprint"]["corpus_manifest_sha256"] = "2" * 64
    env["sidecar_path"].write_text(json.dumps(sidecar))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


@pytest.mark.parametrize("remove_from", [
    "manifest_top_level", "manifest_run_fingerprint", "sidecar_run_fingerprint",
])
def test_missing_corpus_hash_field_rejected(tmp_path, remove_from):
    env = build_good_environment(tmp_path)
    if remove_from == "manifest_top_level":
        manifest = json.loads(env["manifest_path"].read_text())
        del manifest["corpus_manifest_sha256"]
        env["manifest_path"].write_text(json.dumps(manifest))
    elif remove_from == "manifest_run_fingerprint":
        manifest = json.loads(env["manifest_path"].read_text())
        del manifest["run_fingerprint"]["corpus_manifest_sha256"]
        env["manifest_path"].write_text(json.dumps(manifest))
    else:
        sidecar = json.loads(env["sidecar_path"].read_text())
        del sidecar["run_fingerprint"]["corpus_manifest_sha256"]
        env["sidecar_path"].write_text(json.dumps(sidecar))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


def test_corpus_hash_triple_consistency_successful_case(tmp_path, capsys):
    """Dedicated successful-case test: the baseline fixture's three copies of
    corpus_manifest_sha256 (manifest top-level, manifest.run_fingerprint,
    sidecar.run_fingerprint) are identical by construction, and the tool
    must PASS this specific check and log it explicitly.
    """
    env = build_good_environment(tmp_path)
    manifest = json.loads(env["manifest_path"].read_text())
    sidecar = json.loads(env["sidecar_path"].read_text())
    top_level = manifest["corpus_manifest_sha256"]
    manifest_rf = manifest["run_fingerprint"]["corpus_manifest_sha256"]
    sidecar_rf = sidecar["run_fingerprint"]["corpus_manifest_sha256"]
    assert top_level == manifest_rf == sidecar_rf == env["corpus_manifest_sha256"]

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code == 0
    assert output_path.is_file()
    captured = capsys.readouterr()
    assert ("[PASS] corpus_manifest_sha256 identical across manifest / "
            "run_fingerprint / sidecar") in captured.out


# ---------------------------------------------------------------------------
# 16. Manifest/sidecar run_fingerprint_sha256 mismatch (independent of any
#     specific identity file -- tampers the sidecar's own recorded value).
# ---------------------------------------------------------------------------
def test_manifest_sidecar_fingerprint_mismatch_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    sidecar = json.loads(env["sidecar_path"].read_text())
    sidecar["run_fingerprint_sha256"] = "e" * 64
    env["sidecar_path"].write_text(json.dumps(sidecar))

    exit_code, output_path = run_tool(env, tmp_path)

    assert exit_code != 0
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# 17. Path-collision protection: --output must never equal an input path or
#     --apply.
# ---------------------------------------------------------------------------
def test_output_equals_base_feature_norms_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    original_content = env["base_path"].read_bytes()

    argv = build_argv(env, env["base_path"])  # --output == --base-feature-norms
    exit_code = psn.main(argv)

    assert exit_code != 0
    assert env["base_path"].read_bytes() == original_content


def test_output_equals_apply_rejected(tmp_path):
    env = build_good_environment(tmp_path)
    shared_path = tmp_path / "shared.json"
    sentinel = json.dumps({"sentinel": "pre-existing"})
    shared_path.write_text(sentinel)

    argv = build_argv(env, shared_path, apply_path=shared_path)
    exit_code = psn.main(argv)

    assert exit_code != 0
    assert shared_path.read_text() == sentinel


# ---------------------------------------------------------------------------
# 18. Unchanged inputs and apply file after a LATE failure (real corpus
#     verification failing after CSV/JSON validation has already passed).
# ---------------------------------------------------------------------------
def test_inputs_and_apply_unchanged_after_late_real_corpus_failure(tmp_path):
    env = build_good_environment(tmp_path)
    victim_rel_path = env["rows"][0]["relative_path"]
    victim_image_path = env["images_dir"] / victim_rel_path
    original_image_bytes = victim_image_path.read_bytes()
    original_csv_bytes = env["csv_path"].read_bytes()
    original_manifest_bytes = env["manifest_path"].read_bytes()
    original_norms_bytes = env["norms_path"].read_bytes()
    original_base_bytes = env["base_path"].read_bytes()

    victim_image_path.write_bytes(b"tampered-late-in-the-pipeline")

    apply_target = tmp_path / "feature_norms.target.json"
    original_apply_content = json.dumps({"sentinel": "untouched-original"})
    apply_target.write_text(original_apply_content)

    exit_code, output_path = run_tool(env, tmp_path, apply_path=apply_target)

    assert exit_code != 0
    assert not output_path.exists()
    assert apply_target.read_text() == original_apply_content
    # The CSV/manifest/norms/base inputs themselves are never written to by
    # this tool at all -- confirm they are still byte-identical.
    assert env["csv_path"].read_bytes() == original_csv_bytes
    assert env["manifest_path"].read_bytes() == original_manifest_bytes
    assert env["norms_path"].read_bytes() == original_norms_bytes
    assert env["base_path"].read_bytes() == original_base_bytes
    # The only file this test itself modified (the image) is left exactly
    # as this test set it -- i.e. the tool did not touch it either.
    assert victim_image_path.read_bytes() == b"tampered-late-in-the-pipeline"
    assert victim_image_path.read_bytes() != original_image_bytes  # sanity: test setup worked
