"""Corrective regression tests for the canonical VISUAL-norm generator.

These tests target the proven generator + checkpoint-integrity defects:

  1/2. Configured resolution must reach the COMPUTATION, not only provenance
       strings: ``compute_complexity_vector`` canonicalizes at the actual
       ``long_side`` and the generator forwards ``--long-side`` into it.
  3.   Resume verifies the COMPLETE run fingerprint (schema, feature order,
       resolution, extractor + generator hashes, environment, types CSV and
       corpus contents) and each row's image identity, not three columns.
  4.   A non-finite value can never be counted as usable while being dropped
       from a feature distribution.
  5.   An incomplete / torn checkpoint row stops the resume with a nonzero
       error instead of being silently recomputed and appended behind.
  6.   ``authorized_corpus_aggregate_sha256`` describes the full authorized
       population; a limited run reports the processed subset separately.
  7.   Runtime is accumulated across checkpoint segments.
  8.   The legacy protected-file refusal exits nonzero.

Fast by design: the expensive per-image feature computation is monkeypatched to
a deterministic stub for the generator-flow tests, while two dedicated tests run
the REAL extractor to prove the actual 1280 and 1024 analysis inputs.
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stage1"))
sys.path.insert(0, os.path.join(ROOT, "stage1", "tools"))

cv2 = pytest.importorskip("cv2", reason="opencv is required for these tests")

import visual_complexity as vc  # noqa: E402
from visual_complexity import FEATURE_KEYS  # noqa: E402
import canonical_visual_norms as gen  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_img(path, h=300, w=200, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    assert cv2.imwrite(path, img)


def make_corpus(tmp_path, per_cat=2, with_poster=False):
    """Create a tiny types CSV + images dir. Returns (types_csv, images_dir)."""
    imgd = tmp_path / "images"
    imgd.mkdir()
    rows = [("Image Name", "Category", "Block", "Train/Test")]
    seed = 0
    for cat in ("desktop", "mobile", "web"):
        for i in range(per_cat):
            name = f"{cat}{i}.png"
            _write_img(str(imgd / name), seed=seed)
            rows.append((name, cat, "0", "Train"))
            seed += 1
    if with_poster:
        for i in range(per_cat):
            name = f"poster{i}.png"
            _write_img(str(imgd / name), seed=seed)
            rows.append((name, "poster", "0", "Train"))
            seed += 1
    tc = tmp_path / "image_types.csv"
    with open(tc, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerows(rows)
    return str(tc), str(imgd)


def _stub_compute(record=None):
    """A deterministic finite 8-vector stub; optionally records long_side."""
    def fake(path, long_side=vc.CANONICAL_LONG_SIDE):
        if record is not None:
            record.append(long_side)
        return {k: 0.1 * (i + 1) for i, k in enumerate(FEATURE_KEYS)}
    return fake


def _run_main(tmp_path, args):
    return gen.main(args)


# ---------------------------------------------------------------------------
# B. Computation / provenance agreement (defects 1 + 2)
# ---------------------------------------------------------------------------
def test_compute_complexity_vector_accepts_long_side():
    import inspect
    sig = inspect.signature(vc.compute_complexity_vector)
    assert "long_side" in sig.parameters
    assert sig.parameters["long_side"].default == vc.CANONICAL_LONG_SIDE


def test_compute_rejects_non_positive_long_side(tmp_path):
    img = str(tmp_path / "x.png")
    _write_img(img)
    for bad in (0, -1, -1280):
        with pytest.raises(ValueError):
            vc.compute_complexity_vector(img, long_side=bad)


@pytest.mark.parametrize("long_side", [1280, 1024])
def test_feature_functions_receive_actual_long_side(tmp_path, monkeypatch,
                                                    long_side):
    """Every one of the eight feature functions is handed an image whose long
    side equals the configured resolution (proves computation, not strings)."""
    img = str(tmp_path / "big.png")
    _write_img(img, h=1400, w=2000, seed=3)
    seen = {}

    def spy(name):
        def f(image, *a, **k):
            seen.setdefault(name, int(max(image.shape[:2])))
            return 0.0
        return f

    for fn in ("shannon_entropy", "edge_density", "feature_congestion",
               "subband_entropy", "layout_symmetry", "chromatic_coherence",
               "visual_hierarchy", "interactive_element_density"):
        monkeypatch.setattr(vc, fn, spy(fn))

    vc.compute_complexity_vector(img, long_side=long_side)
    assert len(seen) == 8
    assert set(seen.values()) == {long_side}, seen


def test_end_to_end_resolution_paths_differ_on_scale_sensitive_feature(tmp_path):
    """A small REAL feature calculation: a nontrivial image analysed at 1024 vs
    1280 differs on at least one scale-sensitive feature."""
    img = str(tmp_path / "lines.png")
    canvas = np.full((1400, 2000, 3), 255, np.uint8)
    for x in range(0, 2000, 6):           # fine vertical lines -> edges
        canvas[:, x:x + 2] = 0
    for y in range(0, 1400, 40):
        canvas[y:y + 3, :] = 0
    assert cv2.imwrite(img, canvas)

    v1280 = vc.compute_complexity_vector(img, long_side=1280)
    v1024 = vc.compute_complexity_vector(img, long_side=1024)
    for k in FEATURE_KEYS:
        assert np.isfinite(v1280[k]) and np.isfinite(v1024[k])
    scale_sensitive = ("edge_density", "feature_congestion",
                       "interactive_element_density")
    assert any(v1280[k] != v1024[k] for k in scale_sensitive), (
        f"expected a scale-sensitive difference; 1280={v1280} 1024={v1024}")


def test_generator_forwards_long_side_into_computation(tmp_path, monkeypatch):
    """A 1024 generator run calls compute_complexity_vector with long_side=1024
    (not the default 1280), and the emitted metadata agrees."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    rec = []
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute(rec))
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--long-side", "1024", "--limit", "3"])
    assert rec and set(rec) == {1024}
    d = json.load(open(out))
    assert d["canonical_long_side"] == 1024
    assert ":long1024:" in d["canonical_analysis_version"]
    # CSV rows carry the same resolution provenance.
    with open(csvp) as f:
        for row in csv.DictReader(f):
            assert row["canonical_long_side"] == "1024"


# ---------------------------------------------------------------------------
# C. Complete run fingerprint (defect 3)
# ---------------------------------------------------------------------------
def test_run_fingerprint_covers_all_provenance():
    env = {"python_version": "3.9", "numpy_version": "1", "opencv_version": "4",
           "platform": "p", "machine": "m"}
    counts = {"desktop": 495, "mobile": 495, "web": 495}
    fp, sha = gen.build_run_fingerprint(1280, counts, "TYPES", "CORPUS",
                                        "EXTRACT", "GENER", env)
    for key in ("schema_version", "feature_list", "analysis_domain",
                "canonical_long_side", "canonical_analysis_version",
                "interpolation_contract", "min_input_dimension_contract",
                "authorized_categories", "excluded_categories",
                "authorized_counts", "image_types_csv_sha256",
                "authorized_corpus_aggregate_sha256",
                "production_extractor_sha256", "generator_sha256",
                "environment"):
        assert key in fp
    assert len(sha) == 64
    # Any change to any input changes the fingerprint.
    _, sha_res = gen.build_run_fingerprint(1024, counts, "TYPES", "CORPUS",
                                           "EXTRACT", "GENER", env)
    _, sha_types = gen.build_run_fingerprint(1280, counts, "OTHER", "CORPUS",
                                             "EXTRACT", "GENER", env)
    _, sha_corpus = gen.build_run_fingerprint(1280, counts, "TYPES", "OTHER",
                                              "EXTRACT", "GENER", env)
    _, sha_ext = gen.build_run_fingerprint(1280, counts, "TYPES", "CORPUS",
                                           "OTHER", "GENER", env)
    _, sha_gen = gen.build_run_fingerprint(1280, counts, "TYPES", "CORPUS",
                                           "EXTRACT", "OTHER", env)
    _, sha_env = gen.build_run_fingerprint(
        1280, counts, "TYPES", "CORPUS", "EXTRACT", "GENER",
        {**env, "numpy_version": "2"})
    assert len({sha, sha_res, sha_types, sha_corpus, sha_ext, sha_gen,
                sha_env}) == 7


def _fresh_run(tmp_path, monkeypatch, per_cat=2, limit=3, long_side=1280):
    tc, imgd = make_corpus(tmp_path, per_cat=per_cat)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    argv = ["--images-dir", imgd, "--types-csv", tc, "--out", out,
            "--csv", csvp, "--limit", str(limit)]
    if long_side != 1280:
        argv += ["--long-side", str(long_side)]
    gen.main(argv)
    return tc, imgd, out, csvp


def test_resume_matching_fingerprint_accepts(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    rc = gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                   "--csv", csvp, "--limit", "3", "--resume"])
    assert rc == 0
    side = json.load(open(csvp + ".provenance.json"))
    assert side["invocation_count"] == 2


def test_resume_rejects_changed_resolution(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--long-side", "1024", "--limit", "3",
                  "--resume"])


def test_resume_rejects_missing_sidecar(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    os.remove(csvp + ".provenance.json")
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_resume_rejects_changed_image_content(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    # Overwrite one already-processed image so its content hash changes.
    with open(csvp) as f:
        first = next(csv.DictReader(f))
    _write_img(os.path.join(imgd, first["filename"]), seed=999)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_resume_rejects_unknown_filename(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    _tamper_csv(csvp, lambda rows: _rename_first(rows, "not_in_corpus.png"))
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_resume_rejects_wrong_category(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)

    def flip(rows):
        rows[0]["category"] = "web" if rows[0]["category"] != "web" else "mobile"
        return rows
    _tamper_csv(csvp, flip)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


# ---------------------------------------------------------------------------
# E. Non-finite / incomplete / duplicate rows (defects 4 + 5)
# ---------------------------------------------------------------------------
def test_resume_rejects_non_finite_value(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)

    def nanify(rows):
        rows[0][FEATURE_KEYS[0]] = "nan"
        return rows
    _tamper_csv(csvp, nanify)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_resume_rejects_incomplete_row(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)

    def blank(rows):
        rows[0][FEATURE_KEYS[-1]] = ""
        return rows
    _tamper_csv(csvp, blank)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_resume_rejects_duplicate_row(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    with open(csvp) as f:
        rows = list(csv.DictReader(f))
        header = rows[0].keys()
    rows.append(dict(rows[0]))  # duplicate first row
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3", "--resume"])


def test_failed_image_leaves_no_partial_values(tmp_path, monkeypatch):
    """A compute that returns a non-finite value must not be written or counted;
    the aggregate distribution count equals the number of written rows."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)

    calls = {"n": 0}

    def sometimes_bad(path, long_side=vc.CANONICAL_LONG_SIDE):
        calls["n"] += 1
        vals = {k: 0.2 * (i + 1) for i, k in enumerate(FEATURE_KEYS)}
        if calls["n"] == 2:                      # one image yields inf
            vals[FEATURE_KEYS[0]] = float("inf")
        return vals

    monkeypatch.setattr(gen, "compute_complexity_vector", sometimes_bad)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3"])
    d = json.load(open(out))
    n_rows = sum(1 for _ in csv.DictReader(open(csvp)))
    assert d["processed_count"] == n_rows
    assert d["n_failed_images"] >= 1
    for k, stats in d["features"].items():
        assert stats["n"] == d["processed_count"]


# ---------------------------------------------------------------------------
# D. Full-population vs subset semantics (defect 6)
# ---------------------------------------------------------------------------
def test_limited_run_reports_full_population_and_subset_separately(
        tmp_path, monkeypatch):
    tc, imgd = make_corpus(tmp_path, per_cat=2)   # 6 authorized images
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3"])
    d = json.load(open(out))
    assert d["is_subsample"] is True
    assert d["authorized_total"] == 6
    assert d["authorized_category_counts"] == {"desktop": 2, "mobile": 2,
                                               "web": 2}
    assert d["processed_count"] == 3
    # Full-population aggregate differs from the processed-subset manifest hash.
    assert (d["authorized_corpus_aggregate_sha256"]
            != d["processed_subset_sha256"])


def test_full_run_aborts_before_compute_on_corpus_mismatch(tmp_path,
                                                           monkeypatch):
    tc, imgd = make_corpus(tmp_path, per_cat=2)   # NOT the 1485 corpus
    spy = {"called": False}

    def must_not_run(*a, **k):
        spy["called"] = True
        return {k2: 0.0 for k2 in FEATURE_KEYS}

    monkeypatch.setattr(gen, "compute_complexity_vector", must_not_run)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp])          # no --limit -> full run
    assert spy["called"] is False
    assert not os.path.exists(out)
    assert not os.path.exists(csvp)


# ---------------------------------------------------------------------------
# F. Runtime accumulation + output safety (defects 7 + 8)
# ---------------------------------------------------------------------------
def test_runtime_accumulates_across_segments(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    d1 = json.load(open(out))
    assert d1["invocation_count"] == 1
    assert "runtime_current_invocation_seconds" in d1
    assert "runtime_cumulative_seconds" in d1
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3", "--resume"])
    d2 = json.load(open(out))
    assert d2["invocation_count"] == 2
    assert len(d2["segments"]) == 2
    assert d2["runtime_cumulative_seconds"] >= d2[
        "runtime_current_invocation_seconds"]


def test_fresh_run_refuses_existing_output(tmp_path, monkeypatch):
    tc, imgd, out, csvp = _fresh_run(tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):   # no --resume, outputs exist
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "3"])


def test_filename_only_output_paths_work(tmp_path, monkeypatch):
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    monkeypatch.chdir(tmp_path)
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", "norms.json",
              "--csv", "rows.csv", "--limit", "3"])
    assert os.path.exists(tmp_path / "norms.json")
    assert os.path.exists(tmp_path / "rows.csv.provenance.json")


def test_poster_excluded_everywhere(tmp_path, monkeypatch):
    tc, imgd = make_corpus(tmp_path, per_cat=2, with_poster=True)
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3"])
    d = json.load(open(out))
    assert d["authorized_total"] == 6         # posters excluded from population
    assert "poster" in d["excluded_categories"]
    with open(csvp) as f:
        for row in csv.DictReader(f):
            assert not row["filename"].startswith("poster")
            assert row["category"] in ("desktop", "mobile", "web")


def test_generator_protected_output_exits_nonzero(tmp_path):
    """Subprocess: targeting a protected file exits nonzero (not 0)."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "stage1", "tools", "canonical_visual_norms.py"),
         "--images-dir", imgd, "--types-csv", tc,
         "--out", "feature_norms.json",
         "--csv", str(tmp_path / "rows.csv"), "--limit", "3"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert proc.returncode != 0
    assert "protected" in (proc.stdout + proc.stderr).lower()


def test_legacy_builder_protected_refusal_exits_nonzero(tmp_path):
    """Subprocess: legacy build_feature_norms.py refuses a protected write with
    a NONZERO exit. It must not reach the Saliency path."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "stage1", "build_feature_norms.py"),
         "--output", "feature_norms.json",
         "--csv", str(tmp_path / "rows.csv"), "--limit", "1"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert proc.returncode != 0
    assert "refusing" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# CSV tamper helpers
# ---------------------------------------------------------------------------
def _tamper_csv(csvp, transform):
    with open(csvp) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    rows = transform(rows)
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _rename_first(rows, new_name):
    rows[0]["filename"] = new_name
    return rows


# ===========================================================================
# SCOPED FINAL PATCH — recoverability, processed-subset provenance, deps.
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1. Crash-safe interruption recovery.
# ---------------------------------------------------------------------------
def test_interrupted_fresh_run_leaves_usable_checkpoint_and_resumes(
        tmp_path, monkeypatch):
    """A fresh run interrupted after its first successful row leaves a CSV row
    plus a usable running checkpoint, resumes cleanly, and produces the correct
    unique final rows and aggregates."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)      # 6 authorized images
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    sidecar = csvp + ".provenance.json"

    # Interrupt (BaseException, not caught by the per-image except) on the 2nd
    # image, after the 1st row has been durably written.
    calls = {"n": 0}

    def interrupt_after_first(path, long_side=vc.CANONICAL_LONG_SIDE):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated crash")
        return {k: 0.1 * (i + 1) for i, k in enumerate(FEATURE_KEYS)}

    monkeypatch.setattr(gen, "compute_complexity_vector", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", "6"])

    # CSV + running checkpoint exist; exactly one data row; aggregate not yet.
    assert os.path.exists(csvp)
    assert os.path.exists(sidecar)
    assert not os.path.exists(out)
    with open(csvp) as f:
        rows_after_crash = list(csv.DictReader(f))
    assert len(rows_after_crash) == 1
    side = json.load(open(sidecar))
    assert side["status"] == gen.STATUS_RUNNING
    assert side["segments"] and side["segments"][-1]["incomplete"] is True

    # Resume with the normal finite stub -> completes.
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    rc = gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                   "--csv", csvp, "--limit", "6", "--resume"])
    assert rc == 0

    with open(csvp) as f:
        final_rows = list(csv.DictReader(f))
    names = [r["filename"] for r in final_rows]
    assert len(names) == 6
    assert len(set(names)) == 6                       # unique, no duplicates
    d = json.load(open(out))
    assert d["processed_count"] == 6
    for stats in d["features"].values():
        assert stats["n"] == 6
    # Honest runtime: the interrupted segment was never measured -> lower bound.
    assert d["recovered_from_interruption"] is True
    assert d["runtime_cumulative_is_lower_bound"] is True
    fside = json.load(open(sidecar))
    assert fside["status"] == gen.STATUS_COMPLETED


def test_running_checkpoint_written_before_processing(tmp_path, monkeypatch):
    """The fingerprint-bearing sidecar exists as soon as processing starts."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    sidecar = csvp + ".provenance.json"
    seen = {}

    def check_sidecar_first(path, long_side=vc.CANONICAL_LONG_SIDE):
        # On the very first image the running checkpoint must already exist.
        seen.setdefault("sidecar_at_first_image", os.path.exists(sidecar))
        return {k: 0.1 * (i + 1) for i, k in enumerate(FEATURE_KEYS)}

    monkeypatch.setattr(gen, "compute_complexity_vector", check_sidecar_first)
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3"])
    assert seen["sidecar_at_first_image"] is True


# ---------------------------------------------------------------------------
# Fix 2. Processed-subset provenance from actual stored rows.
# ---------------------------------------------------------------------------
def test_six_row_checkpoint_resumed_with_limit3_reports_six(tmp_path,
                                                            monkeypatch):
    """A six-row checkpoint resumed with --limit 3 must report six processed
    rows and a six-image processed hash, never a three-image hash."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)      # 6 images
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "6"])           # full 6-row checkpoint
    d6 = json.load(open(out))
    assert d6["processed_count"] == 6
    six_hash = d6["processed_subset_sha256"]

    # Resume asking for only 3 -> all already done; processed stays 6.
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3", "--resume"])
    d = json.load(open(out))
    assert d["processed_count"] == 6
    assert d["processed_subset_sha256"] == six_hash
    assert d["requested_limit"] == 3
    assert d["requested_selection_count"] == 3
    # The requested-3 hash must NOT be presented as the processed hash.
    assert d["requested_selection_sha256"] != d["processed_subset_sha256"]


def test_failed_image_excluded_from_processed_hash(tmp_path, monkeypatch):
    """A selected image that fails is absent from the processed hash, and the
    processed hash equals the hash of exactly the stored CSV rows."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")

    def fail_one(path, long_side=vc.CANONICAL_LONG_SIDE):
        if os.path.basename(path) == "desktop0.png":
            raise RuntimeError("boom")
        return {k: 0.1 * (i + 1) for i, k in enumerate(FEATURE_KEYS)}

    monkeypatch.setattr(gen, "compute_complexity_vector", fail_one)
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "6"])
    d = json.load(open(out))
    with open(csvp) as f:
        stored = list(csv.DictReader(f))
    stored_names = [r["filename"] for r in stored]
    assert "desktop0.png" not in stored_names
    assert d["processed_count"] == len(stored)
    assert d["n_failed_images"] >= 1
    # Recompute the processed hash from the actual stored rows and compare.
    pairs = [(r["filename"], r["category"]) for r in stored]
    assert d["processed_subset_sha256"] == gen.authorized_aggregate_hash(
        pairs, imgd)


def test_processed_metadata_agrees_with_csv(tmp_path, monkeypatch):
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "6"])
    d = json.load(open(out))
    with open(csvp) as f:
        stored = list(csv.DictReader(f))
    pairs = [(r["filename"], r["category"]) for r in stored]
    counts = {}
    for _n, c in pairs:
        counts[c] = counts.get(c, 0) + 1
    assert d["processed_count"] == len(stored)
    assert d["processed_category_counts"] == counts
    assert d["processed_subset_sha256"] == gen.authorized_aggregate_hash(
        pairs, imgd)


@pytest.mark.parametrize("bad_limit", ["0", "-1", "-3"])
def test_zero_or_negative_limit_fails(tmp_path, monkeypatch, bad_limit):
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    with pytest.raises(gen.GeneratorError):
        gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
                  "--csv", csvp, "--limit", bad_limit])


# ---------------------------------------------------------------------------
# Fix 3. Complete calculation-environment provenance.
# ---------------------------------------------------------------------------
def test_environment_info_records_all_calculation_libraries():
    env = gen.environment_info()
    for key in ("python_version", "numpy_version", "opencv_version",
                "scipy_version", "scikit_image_version", "pyrtools_version",
                "pillow_version"):
        assert key in env, key
        assert env[key] and env[key] != "unknown", (key, env.get(key))


def test_each_dependency_version_changes_run_fingerprint():
    counts = {"desktop": 495, "mobile": 495, "web": 495}
    base_env = gen.environment_info()
    _, base_sha = gen.build_run_fingerprint(
        1280, counts, "TYPES", "CORPUS", "EXTRACT", "GENER", base_env)
    dep_keys = ("python_version", "numpy_version", "opencv_version",
                "scipy_version", "scikit_image_version", "pyrtools_version",
                "pillow_version")
    seen = {base_sha}
    for key in dep_keys:
        mutated = dict(base_env)
        mutated[key] = base_env[key] + ".changed"
        _, sha = gen.build_run_fingerprint(
            1280, counts, "TYPES", "CORPUS", "EXTRACT", "GENER", mutated)
        assert sha != base_sha, f"changing {key} did not change fingerprint"
        seen.add(sha)
    assert len(seen) == len(dep_keys) + 1     # every change is distinct


def test_run_fingerprint_embeds_full_environment(tmp_path, monkeypatch):
    """The emitted run fingerprint and every segment carry the full env dict."""
    tc, imgd = make_corpus(tmp_path, per_cat=2)
    out = str(tmp_path / "norms.json")
    csvp = str(tmp_path / "rows.csv")
    monkeypatch.setattr(gen, "compute_complexity_vector", _stub_compute())
    gen.main(["--images-dir", imgd, "--types-csv", tc, "--out", out,
              "--csv", csvp, "--limit", "3"])
    d = json.load(open(out))
    for key in ("scipy_version", "scikit_image_version", "pyrtools_version",
                "pillow_version"):
        assert key in d["run_fingerprint"]["environment"]
        assert key in d["environment"]
        assert key in d["segments"][-1]["environment"]

