#!/usr/bin/env python3
"""Dedicated, SAFE generator for the canonical VISUAL feature-norm distribution.

This tool builds the future canonical reference distribution for exactly the
EIGHT Stage-1 visual features, computed with the production canonical
preprocessing contract (``compute_complexity_vector`` -> ``canonicalize_for_analysis``).

It is deliberately narrow and defensive:

  * VISUAL-ONLY. It never imports, loads or runs UMSI++ or any saliency model.
    The aggregate contains only the eight visual norms — no saliency values are
    copied in.
  * It writes ONLY to a dedicated canonical artifact
    (``canonical_visual_feature_norms.json``) plus a provenance-carrying
    per-image CSV. It hard-refuses to write to the protected
    ``feature_norms.json`` or ``hceye/sensitivity_lookup.json``.
  * Only the authorized interactive GUI categories (desktop / mobile / web) are
    accepted; ``poster`` is rejected at discovery.
  * Every emitted row carries provenance (analysis domain, canonical long side,
    canonical-analysis version). Resume must be explicitly requested and is
    allowed ONLY when every provenance field of the existing CSV matches the
    current run; a native, mixed-resolution or differently-versioned CSV is
    rejected.
  * Duplicate filenames, duplicate categories-per-filename, and duplicate rows
    are rejected.
  * A FULL run (no ``--limit``) requires exactly 1,485 usable rows and 495 per
    authorized category.
  * Outputs contain no personal absolute paths (filenames and hashes only).

This script does NOT modify the live pipeline. The expensive full 1,485-image
generation is a SEPARATELY AUTHORIZED run; nothing here runs it automatically.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths / imports. Only the VISUAL extractor is imported. No saliency imports.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parent
_STAGE1_DIR = _TOOLS_DIR.parent
_PROJECT_ROOT = _STAGE1_DIR.parent
for _p in (str(_STAGE1_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visual_complexity import (  # noqa: E402
    compute_complexity_vector,
    canonicalize_for_analysis,  # noqa: F401 (documents the production contract)
    canonical_analysis_version,
    FEATURE_KEYS,
    CANONICAL_LONG_SIDE,
    MIN_CANONICAL_INPUT_LONG_SIDE,
    MIN_CANONICAL_INPUT_SHORT_SIDE,
)

# The eight canonical VISUAL features, in production order. This is the ONLY
# feature set this generator ever produces (no saliency).
VISUAL_FEATURE_KEYS: Tuple[str, ...] = tuple(FEATURE_KEYS)

# The single authoritative authorized GUI population. ``poster`` is excluded.
AUTHORIZED_CATEGORIES = ("desktop", "mobile", "web")
_AUTHORIZED_SET = frozenset(AUTHORIZED_CATEGORIES)
EXCLUDED_CATEGORIES = ("poster",)

# Full-run expectations.
FULL_RUN_TOTAL = 1485
FULL_RUN_PER_CATEGORY = 495

# Schema version for the emitted artifact.
SCHEMA_VERSION = "canonical-visual-feature-norms-v1"
ANALYSIS_DOMAIN = "canonical"

# Protected files this generator must NEVER write.
_PROTECTED_BASENAMES = {"feature_norms.json", "sensitivity_lookup.json"}
_PROTECTED_PATHS = {
    (_STAGE1_DIR / "data" / "results" / "feature_norms.json").resolve(),
    (_PROJECT_ROOT / "hceye" / "sensitivity_lookup.json").resolve(),
}

# Provenance columns embedded per row so a mixed / native / differently-versioned
# CSV is detectable at the row level and can never be silently resumed.
_PROVENANCE_COLUMNS = ("analysis_domain", "canonical_long_side",
                       "canonical_analysis_version")

# Per-row integrity columns. Each row is bound to the complete run fingerprint
# and to the exact image content it was computed from, so a resume can detect a
# changed corpus / code / environment (via the fingerprint) or a changed image
# file (via the content hash) at the granularity of an individual row.
ROW_FINGERPRINT_COLUMN = "run_fingerprint_sha256"
ROW_IMAGE_SHA_COLUMN = "image_sha256"

# Deterministic, human-readable contracts folded into the run fingerprint.
INTERPOLATION_CONTRACT = ("INTER_AREA when downscaling, INTER_LINEAR when "
                          "upscaling (aspect preserved)")

# ---------------------------------------------------------------------------
# Verified authorized-corpus identity (from the audited UEyes GUI corpus).
# A FULL run (no --limit) MUST reproduce these exact counts and hashes BEFORE
# the feature extractor is invoked even once; a mismatching corpus aborts with
# no computation and no output. These describe the complete 1,485-image
# authorized population (desktop/mobile/web, 495 each) — poster is excluded.
# ---------------------------------------------------------------------------
EXPECTED_TYPES_CSV_SHA256 = (
    "fc22d679849862c0f0476b1d9c270a50383c62304c7635b87cb18a2559a25c68")
EXPECTED_AUTHORIZED_AGGREGATE_SHA256 = (
    "785c4e37e95ecd36903c05a280f4db3733d378328a8deec8fa4258c3bb67e200")
EXPECTED_AUTHORIZED_COUNTS = {"desktop": 495, "mobile": 495, "web": 495}


class GeneratorError(RuntimeError):
    """Fatal, expected generator condition (bad provenance, poster, etc.)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refuse_protected(path: str, label: str) -> None:
    """Abort if ``path`` targets a protected file (by realpath or basename)."""
    rp = Path(path).resolve()
    if rp in _PROTECTED_PATHS or rp.name in _PROTECTED_BASENAMES:
        raise GeneratorError(
            f"Refusing to write {label} to protected file '{path}'. This "
            f"generator must never overwrite feature_norms.json or "
            f"sensitivity_lookup.json.")


def load_authorized_images(types_csv: str, images_dir: str,
                           limit: Optional[int] = None
                           ) -> List[Tuple[str, str]]:
    """Return ``(filename, category)`` for AUTHORIZED images only.

    Rejects ``poster`` at discovery, and rejects duplicate filenames, duplicate
    category-per-filename and duplicate ``(filename, category)`` rows in the
    source CSV.
    """
    if not os.path.exists(types_csv):
        raise GeneratorError(f"types-csv not found: {types_csv}")
    if not os.path.isdir(images_dir):
        raise GeneratorError(f"images-dir not found: {images_dir}")

    seen_pairs = set()
    name_to_cats: Dict[str, set] = collections.defaultdict(set)
    by_category: Dict[str, List[str]] = collections.defaultdict(list)

    with open(types_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = (row.get("Image Name") or "").strip()
            cat = (row.get("Category") or "").strip().lower()
            if not name or not cat:
                continue
            # Reject duplicate identical rows in the source manifest.
            pair = (name, cat)
            if pair in seen_pairs:
                raise GeneratorError(
                    f"Duplicate row in types-csv: {name!r} / {cat!r}")
            seen_pairs.add(pair)
            name_to_cats[name].add(cat)
            if cat in _AUTHORIZED_SET:
                by_category[cat].append(name)
            # poster (and anything else) is silently dropped at discovery.

    # A filename mapped to more than one category is a corpus integrity error.
    conflicting = [n for n, cs in name_to_cats.items() if len(cs) > 1]
    if conflicting:
        raise GeneratorError(
            f"{len(conflicting)} filename(s) map to multiple categories, "
            f"e.g. {conflicting[:3]}")

    images: List[Tuple[str, str]] = []
    for cat in sorted(by_category):
        names = sorted(set(by_category[cat]))
        # Duplicate filename within a category (post-dedup) would shrink names;
        # detect it explicitly.
        if len(names) != len(by_category[cat]):
            raise GeneratorError(
                f"Duplicate filename within category {cat!r}")
        chosen = names if limit is None else names[:max(1, limit // len(AUTHORIZED_CATEGORIES))]
        for name in chosen:
            if (Path(images_dir) / name).exists():
                images.append((name, cat))
    return images


def expected_provenance(long_side: int) -> Dict[str, str]:
    return {
        "analysis_domain": ANALYSIS_DOMAIN,
        "canonical_long_side": str(int(long_side)),
        "canonical_analysis_version": canonical_analysis_version(long_side),
    }


def load_existing_csv_for_resume(csv_path: str, long_side: int
                                 ) -> Tuple[List[dict], set]:
    """Load a prior CSV for resume, enforcing matching provenance on EVERY row.

    Rejects the resume (raises) if any row's provenance columns differ from the
    current run (native / mixed-resolution / differently-versioned CSV), if the
    header lacks provenance columns, or if duplicate filenames/rows appear.
    """
    exp = expected_provenance(long_side)
    rows: List[dict] = []
    done: set = set()
    if not os.path.exists(csv_path):
        return rows, done

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in _PROVENANCE_COLUMNS if c not in header]
        if missing:
            raise GeneratorError(
                f"Refusing to resume: CSV '{csv_path}' lacks provenance "
                f"columns {missing}. It is not a canonical provenance-carrying "
                f"CSV and may be native/legacy.")
        for c in ("filename", "category", *VISUAL_FEATURE_KEYS):
            if c not in header:
                raise GeneratorError(
                    f"Refusing to resume: CSV '{csv_path}' missing column {c!r}.")
        seen_pairs = set()
        for raw in reader:
            name = (raw.get("filename") or "").strip()
            cat = (raw.get("category") or "").strip().lower()
            if not name:
                continue
            # Row-level provenance must match the current run exactly.
            for col in _PROVENANCE_COLUMNS:
                if (raw.get(col) or "").strip() != exp[col]:
                    raise GeneratorError(
                        f"Refusing to resume: row {name!r} has {col}="
                        f"{raw.get(col)!r}, expected {exp[col]!r}. The existing "
                        f"CSV is a different domain/resolution/version and must "
                        f"not be mixed.")
            if cat not in _AUTHORIZED_SET:
                raise GeneratorError(
                    f"Refusing to resume: row {name!r} has non-authorized "
                    f"category {cat!r}.")
            pair = (name, cat)
            if pair in seen_pairs or name in done:
                raise GeneratorError(
                    f"Refusing to resume: duplicate row for {name!r}.")
            seen_pairs.add(pair)
            row = {"filename": name, "category": cat}
            ok = True
            for k in VISUAL_FEATURE_KEYS:
                v = raw.get(k, "")
                if v == "" or v is None:
                    ok = False
                    break
                row[k] = float(v)
            if not ok:
                continue  # incomplete line -> recompute
            rows.append(row)
            done.add(name)
    return rows, done


def aggregate(values_by_feature: Dict[str, List[float]]) -> Dict[str, dict]:
    norms = {}
    for feature in VISUAL_FEATURE_KEYS:
        vals = values_by_feature.get(feature, [])
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        norms[feature] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }
    return norms


def authorized_aggregate_hash(images: List[Tuple[str, str]],
                              images_dir: str) -> str:
    """Deterministic SHA-256 over sorted ``filename|category|content-sha256``."""
    recs = []
    for name, cat in sorted(images):
        p = os.path.join(images_dir, name)
        recs.append(f"{name}|{cat}|{_sha256_file(p)}")
    return hashlib.sha256("\n".join(sorted(recs)).encode()).hexdigest()


def environment_info() -> Dict[str, str]:
    import cv2
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def load_full_authorized_population(types_csv: str, images_dir: str
                                    ) -> List[Tuple[str, str]]:
    """Return the COMPLETE authorized ``(filename, category)`` population.

    Never applies ``--limit``: this is the full-population reference used for
    authorized counts and both corpus hashes. Subsetting happens separately.
    """
    return load_authorized_images(types_csv, images_dir, limit=None)


def subsample_balanced(population: List[Tuple[str, str]],
                       limit: Optional[int]) -> List[Tuple[str, str]]:
    """Return a deterministic per-category balanced subset (or the full list)."""
    if limit is None:
        return list(population)
    by_cat: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for name, cat in population:
        by_cat[cat].append((name, cat))
    per = max(1, limit // len(AUTHORIZED_CATEGORIES))
    out: List[Tuple[str, str]] = []
    for cat in sorted(by_cat):
        out.extend(sorted(by_cat[cat])[:per])
    return out


def subset_manifest_hash(images: List[Tuple[str, str]],
                         images_dir: str) -> str:
    """Deterministic hash over the PROCESSED subset (distinct from full corpus)."""
    return authorized_aggregate_hash(images, images_dir)


def _canonical_json(obj) -> str:
    """Canonical, deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def build_run_fingerprint(long_side: int,
                          full_counts: Dict[str, int],
                          types_hash: str,
                          corpus_hash: str,
                          extractor_hash: str,
                          generator_hash: str,
                          env: Dict[str, str]) -> Tuple[Dict[str, object], str]:
    """Build the complete calculation-provenance object and its SHA-256.

    The fingerprint captures every input that changes what is computed or what
    it means: schema, exact feature order, analysis domain, actual canonical
    long side + version, interpolation and minimum-dimension contracts,
    authorized/excluded categories, complete authorized counts, both corpus
    hashes, extractor + generator hashes, and the environment versions. Runtime,
    ``--limit`` and any subset detail are deliberately EXCLUDED so a legitimate
    resume of the same corpus/code/environment matches exactly.
    """
    fp = {
        "schema_version": SCHEMA_VERSION,
        "feature_list": list(VISUAL_FEATURE_KEYS),
        "analysis_domain": ANALYSIS_DOMAIN,
        "canonical_long_side": int(long_side),
        "canonical_analysis_version": canonical_analysis_version(long_side),
        "interpolation_contract": INTERPOLATION_CONTRACT,
        "min_input_dimension_contract": {
            "min_long_side_px": int(MIN_CANONICAL_INPUT_LONG_SIDE),
            "min_short_side_px": int(MIN_CANONICAL_INPUT_SHORT_SIDE),
        },
        "authorized_categories": list(AUTHORIZED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "authorized_counts": {k: int(full_counts.get(k, 0))
                              for k in AUTHORIZED_CATEGORIES},
        "image_types_csv_sha256": types_hash,
        "authorized_corpus_aggregate_sha256": corpus_hash,
        "production_extractor_sha256": extractor_hash,
        "generator_sha256": generator_hash,
        "environment": dict(env),
    }
    fp_sha = hashlib.sha256(_canonical_json(fp).encode("utf-8")).hexdigest()
    return fp, fp_sha


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON atomically (temp file in the same directory, then replace)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def sidecar_path_for(csv_path: str) -> str:
    return csv_path + ".provenance.json"


def load_checkpoint_for_resume(csv_path: str, sidecar_path: str,
                               fingerprint_sha: str,
                               population: Dict[str, str],
                               images_dir: str
                               ) -> Tuple[List[dict], set, List[dict]]:
    """Validate and load a resumable checkpoint (CSV + provenance sidecar).

    Resume is allowed ONLY when the complete stored run fingerprint equals the
    freshly calculated fingerprint AND every stored row remains consistent with
    the current authorized population and on-disk image content. Any of the
    rejection classes below raises ``GeneratorError`` (nonzero exit):

      * missing checkpoint provenance (sidecar absent / unreadable);
      * stored run fingerprint != current fingerprint (folds in schema /
        feature-order / resolution / extractor / generator / environment /
        types-CSV / corpus-content changes);
      * a row bound to a different run fingerprint;
      * a filename absent from the current authorized population;
      * a category that disagrees with the current manifest;
      * changed image contents (row image hash != current file hash);
      * a poster / unauthorized category;
      * a duplicate filename / row;
      * a missing, NaN or infinite feature value (incomplete / torn record).
    """
    if not os.path.exists(sidecar_path):
        raise GeneratorError(
            f"Refusing to resume: checkpoint provenance sidecar "
            f"'{os.path.basename(sidecar_path)}' is missing. The checkpoint set "
            f"(CSV + sidecar) is not mutually consistent.")
    try:
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    except Exception as exc:
        raise GeneratorError(
            f"Refusing to resume: cannot read checkpoint provenance sidecar: "
            f"{exc}")

    stored_sha = sidecar.get("run_fingerprint_sha256")
    if stored_sha != fingerprint_sha:
        raise GeneratorError(
            f"Refusing to resume: stored run fingerprint {stored_sha!r} does "
            f"not equal the freshly calculated fingerprint {fingerprint_sha!r}. "
            f"The corpus, code, resolution or environment has changed.")

    rows: List[dict] = []
    done: set = set()
    seen_pairs: set = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = ["filename", "category", *_PROVENANCE_COLUMNS,
                    ROW_FINGERPRINT_COLUMN, ROW_IMAGE_SHA_COLUMN,
                    *VISUAL_FEATURE_KEYS]
        missing = [c for c in required if c not in header]
        if missing:
            raise GeneratorError(
                f"Refusing to resume: checkpoint CSV lacks required columns "
                f"{missing}.")
        for raw in reader:
            name = (raw.get("filename") or "").strip()
            cat = (raw.get("category") or "").strip().lower()
            if not name:
                continue
            if (raw.get(ROW_FINGERPRINT_COLUMN) or "").strip() != fingerprint_sha:
                raise GeneratorError(
                    f"Refusing to resume: row {name!r} is bound to a different "
                    f"run fingerprint.")
            if name not in population:
                raise GeneratorError(
                    f"Refusing to resume: row {name!r} is absent from the "
                    f"current authorized population.")
            if population[name] != cat:
                raise GeneratorError(
                    f"Refusing to resume: row {name!r} category {cat!r} "
                    f"disagrees with the current manifest "
                    f"({population[name]!r}).")
            if cat not in _AUTHORIZED_SET:
                raise GeneratorError(
                    f"Refusing to resume: row {name!r} has non-authorized "
                    f"category {cat!r}.")
            pair = (name, cat)
            if pair in seen_pairs or name in done:
                raise GeneratorError(
                    f"Refusing to resume: duplicate row for {name!r}.")
            seen_pairs.add(pair)
            # Image content must be unchanged since the row was computed.
            img_path = os.path.join(images_dir, name)
            if not os.path.exists(img_path):
                raise GeneratorError(
                    f"Refusing to resume: image for row {name!r} no longer "
                    f"exists on disk.")
            current_img_sha = _sha256_file(img_path)
            if (raw.get(ROW_IMAGE_SHA_COLUMN) or "").strip() != current_img_sha:
                raise GeneratorError(
                    f"Refusing to resume: image content for row {name!r} has "
                    f"changed since it was computed.")
            # Every feature must be present, numeric and finite. A missing /
            # NaN / infinite value is a torn or corrupt record -> stop and
            # require operator action (never count it or append behind it).
            row = {"filename": name, "category": cat}
            for k in VISUAL_FEATURE_KEYS:
                v = raw.get(k, "")
                if v is None or str(v).strip() == "":
                    raise GeneratorError(
                        f"Refusing to resume: row {name!r} has an incomplete "
                        f"(missing) value for {k!r}. Remove the torn record and "
                        f"re-run.")
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    raise GeneratorError(
                        f"Refusing to resume: row {name!r} has a non-numeric "
                        f"value for {k!r}: {v!r}.")
                if not np.isfinite(fv):
                    raise GeneratorError(
                        f"Refusing to resume: row {name!r} has a non-finite "
                        f"value for {k!r}: {fv!r}. Remove the corrupt record "
                        f"and re-run.")
                row[k] = fv
            rows.append(row)
            done.add(name)

    prior_segments = list(sidecar.get("segments", []))
    return rows, done, prior_segments


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    default_images = os.path.join(_PROJECT_ROOT, "ueyes", "dataset_full",
                                  "UEyes_dataset", "images")
    default_types = os.path.join(_PROJECT_ROOT, "ueyes", "dataset_full",
                                 "UEyes_dataset", "image_types.csv")
    default_out = os.path.join(_STAGE1_DIR, "data", "results",
                               "canonical_visual_feature_norms.json")
    default_csv = os.path.join(_STAGE1_DIR, "data", "results",
                               "canonical_visual_feature_rows.csv")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", default=default_images)
    ap.add_argument("--types-csv", default=default_types)
    ap.add_argument("--out", default=default_out,
                    help="canonical visual-norm JSON (never a protected file)")
    ap.add_argument("--csv", default=default_csv,
                    help="provenance-carrying per-image CSV")
    ap.add_argument("--long-side", type=int, default=CANONICAL_LONG_SIDE,
                    help="canonical long side (default = production value)")
    ap.add_argument("--limit", type=int, default=None,
                    help="balanced subsample of ~N images (marks is_subsample)")
    ap.add_argument("--resume", action="store_true",
                    help="explicitly resume from a mutually consistent, "
                         "fingerprint-matching checkpoint (CSV + sidecar)")
    args = ap.parse_args(argv)

    # (F/5) Hard refusal to target protected files (out, CSV and sidecar).
    sidecar = sidecar_path_for(args.csv)
    _refuse_protected(args.out, "aggregate JSON")
    _refuse_protected(args.csv, "per-image CSV")
    _refuse_protected(sidecar, "provenance sidecar")

    # (B) Reject a non-positive / invalid configured resolution up front.
    if args.long_side is None or int(args.long_side) <= 0:
        raise GeneratorError(
            f"--long-side must be a positive integer, got {args.long_side!r}.")
    long_side = int(args.long_side)
    exp = expected_provenance(long_side)
    is_subsample = args.limit is not None

    # (D) Load and validate the COMPLETE authorized population BEFORE --limit.
    # Authorized counts and both corpus hashes always describe the full 1,485
    # population, never the processed subset.
    full_population = load_full_authorized_population(args.types_csv,
                                                      args.images_dir)
    if not full_population:
        raise GeneratorError("No authorized images discovered.")
    assert all(c in _AUTHORIZED_SET for _, c in full_population), "poster leaked"

    full_counts = collections.Counter(c for _, c in full_population)
    types_hash = _sha256_file(args.types_csv)
    corpus_hash = authorized_aggregate_hash(full_population, args.images_dir)
    extractor_hash = _sha256_file(str(_STAGE1_DIR / "visual_complexity.py"))
    generator_hash = _sha256_file(str(_THIS_FILE))
    env = environment_info()
    population_map = {name: cat for name, cat in full_population}

    # (D/9) A FULL run must reproduce the verified corpus identity BEFORE the
    # feature extractor is invoked even once. A mismatching corpus aborts here
    # with no computation and no output.
    if not is_subsample:
        actual_counts = {k: int(full_counts.get(k, 0))
                         for k in AUTHORIZED_CATEGORIES}
        if actual_counts != EXPECTED_AUTHORIZED_COUNTS:
            raise GeneratorError(
                f"Full-run corpus mismatch: authorized counts {actual_counts} "
                f"!= expected {EXPECTED_AUTHORIZED_COUNTS}. Aborting before any "
                f"feature computation.")
        if len(full_population) != FULL_RUN_TOTAL:
            raise GeneratorError(
                f"Full-run corpus mismatch: {len(full_population)} authorized "
                f"images != expected {FULL_RUN_TOTAL}. Aborting before compute.")
        if types_hash != EXPECTED_TYPES_CSV_SHA256:
            raise GeneratorError(
                f"Full-run corpus mismatch: image_types.csv hash {types_hash} "
                f"!= expected {EXPECTED_TYPES_CSV_SHA256}. Aborting before "
                f"compute.")
        if corpus_hash != EXPECTED_AUTHORIZED_AGGREGATE_SHA256:
            raise GeneratorError(
                f"Full-run corpus mismatch: authorized aggregate hash "
                f"{corpus_hash} != expected "
                f"{EXPECTED_AUTHORIZED_AGGREGATE_SHA256}. Aborting before "
                f"compute.")

    # (C) One complete, deterministic calculation-provenance fingerprint.
    run_fingerprint, fingerprint_sha = build_run_fingerprint(
        long_side, full_counts, types_hash, corpus_hash, extractor_hash,
        generator_hash, env)

    # Processed subset (never affects full-population hashes / counts).
    images = subsample_balanced(full_population, args.limit)

    feature_order = list(VISUAL_FEATURE_KEYS)
    fieldnames = ["filename", "category", *_PROVENANCE_COLUMNS,
                  ROW_FINGERPRINT_COLUMN, ROW_IMAGE_SHA_COLUMN, *feature_order]

    values_by_feature: Dict[str, List[float]] = collections.defaultdict(list)
    per_image_rows: List[dict] = []
    prior_segments: List[dict] = []
    done_names: set = set()

    csv_exists = os.path.exists(args.csv) and os.path.getsize(args.csv) > 0
    sidecar_exists = os.path.exists(sidecar)
    out_exists = os.path.exists(args.out)

    if args.resume:
        # (F) Resume requires a mutually consistent checkpoint set.
        if not csv_exists or not sidecar_exists:
            raise GeneratorError(
                "Refusing to resume: a consistent checkpoint set (CSV + "
                "provenance sidecar) was not found.")
        existing, done_names, prior_segments = load_checkpoint_for_resume(
            args.csv, sidecar, fingerprint_sha, population_map, args.images_dir)
        per_image_rows.extend(existing)
        for row in existing:
            for k in feature_order:
                values_by_feature[k].append(row[k])
        print(f"Resuming: {len(done_names)} rows validated against the run "
              f"fingerprint in {os.path.basename(args.csv)}.")
    else:
        # (F) No implicit overwrite of any existing output.
        for label, pth, present in (("CSV", args.csv, csv_exists),
                                    ("provenance sidecar", sidecar,
                                     sidecar_exists),
                                    ("aggregate JSON", args.out, out_exists)):
            if present:
                raise GeneratorError(
                    f"Output {label} '{pth}' already exists. Refusing to "
                    f"overwrite implicitly. Pass --resume for a matching "
                    f"checkpoint or choose fresh output paths.")

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t_start = time.perf_counter()
    n_new = 0
    n_failed = 0
    with open(args.csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
            csv_file.flush()
        for idx, (name, category) in enumerate(images, start=1):
            if name in done_names:
                continue
            img_path = os.path.join(args.images_dir, name)
            # (E) Build ALL eight values, require every one finite, and only
            # THEN mutate the aggregate collections / append the row. A failed
            # image leaves no partial values anywhere.
            try:
                v = compute_complexity_vector(img_path, long_side=long_side)
                staged = {}
                for k in feature_order:
                    fv = float(v[k])
                    if not np.isfinite(fv):
                        raise ValueError(f"non-finite {k}={fv!r}")
                    staged[k] = fv
                img_sha = _sha256_file(img_path)
            except Exception as exc:  # keep going, but stage nothing
                n_failed += 1
                print(f"[{idx}/{len(images)}] ERROR {name}: {exc}")
                continue

            row = {"filename": name, "category": category,
                   "analysis_domain": exp["analysis_domain"],
                   "canonical_long_side": exp["canonical_long_side"],
                   "canonical_analysis_version": exp["canonical_analysis_version"],
                   ROW_FINGERPRINT_COLUMN: fingerprint_sha,
                   ROW_IMAGE_SHA_COLUMN: img_sha}
            for k in feature_order:
                row[k] = staged[k]
                values_by_feature[k].append(staged[k])
            per_image_rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            n_new += 1
            print(f"[{idx}/{len(images)}] {name} ({category})")
    elapsed = time.perf_counter() - t_start
    ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    processed_counts = collections.Counter(r["category"] for r in per_image_rows)
    total_usable = len(per_image_rows)

    # (E/12) Full-run integrity, enforced on the assembled result.
    if not is_subsample:
        if total_usable != FULL_RUN_TOTAL:
            raise GeneratorError(
                f"Full run requires exactly {FULL_RUN_TOTAL} usable rows; got "
                f"{total_usable}. Refusing to write aggregate.")
        for cat in AUTHORIZED_CATEGORIES:
            if processed_counts.get(cat, 0) != FULL_RUN_PER_CATEGORY:
                raise GeneratorError(
                    f"Full run requires {FULL_RUN_PER_CATEGORY} in {cat}; got "
                    f"{processed_counts.get(cat, 0)}.")
        produced = {r["filename"] for r in per_image_rows}
        if produced != set(population_map):
            raise GeneratorError(
                "Full run produced a filename set that differs from the "
                "authorized population.")
        for k in feature_order:
            vals = values_by_feature.get(k, [])
            if len(vals) != FULL_RUN_TOTAL or not all(np.isfinite(vals)):
                raise GeneratorError(
                    f"Full run feature {k!r} has {len(vals)} finite values; "
                    f"expected {FULL_RUN_TOTAL}.")

    norms = aggregate(values_by_feature)
    # Every feature distribution must have exactly one finite value per usable
    # row (guards defect: a non-finite value counting as usable but dropped from
    # the distribution).
    for k in feature_order:
        n_dist = norms.get(k, {}).get("n", 0)
        if n_dist != total_usable:
            raise GeneratorError(
                f"Aggregate integrity error: feature {k!r} distribution has "
                f"n={n_dist} but {total_usable} rows are counted as usable.")

    # (F) Runtime accumulation across checkpoint segments.
    this_segment = {
        "invocation_index": len(prior_segments),
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": round(elapsed, 3),
        "n_new_rows": n_new,
        "n_failed": n_failed,
        "environment": env,
    }
    all_segments = prior_segments + [this_segment]
    cumulative = round(sum(float(s.get("elapsed_seconds", 0.0))
                           for s in all_segments), 3)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_list": list(VISUAL_FEATURE_KEYS),
        "analysis_domain": ANALYSIS_DOMAIN,
        "canonical_long_side": long_side,
        "canonical_analysis_version": exp["canonical_analysis_version"],
        "interpolation_behavior": INTERPOLATION_CONTRACT,
        "input_dimension_contract": {
            "min_long_side_px": int(MIN_CANONICAL_INPUT_LONG_SIDE),
            "min_short_side_px": int(MIN_CANONICAL_INPUT_SHORT_SIDE),
        },
        "authorized_categories": list(AUTHORIZED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        # Full-population identity (always the complete 1,485 authorized corpus).
        "authorized_category_counts": {k: int(full_counts.get(k, 0))
                                       for k in AUTHORIZED_CATEGORIES},
        "authorized_total": int(sum(full_counts.get(k, 0)
                                    for k in AUTHORIZED_CATEGORIES)),
        "image_types_csv_sha256": types_hash,
        "image_types_csv_basename": os.path.basename(args.types_csv),
        "authorized_corpus_aggregate_sha256": corpus_hash,
        # Processed-subset identity (distinct from the full population).
        "is_subsample": is_subsample,
        "processed_count": total_usable,
        "processed_category_counts": dict(processed_counts),
        "processed_subset_sha256": subset_manifest_hash(images, args.images_dir),
        "n_failed_images": n_failed,
        "production_extractor_sha256": extractor_hash,
        "generator_sha256": generator_hash,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint_sha,
        "environment": env,
        # Runtime is an environment-specific observed measurement.
        "runtime_current_invocation_seconds": round(elapsed, 3),
        "runtime_cumulative_seconds": cumulative,
        "invocation_count": len(all_segments),
        "segments": all_segments,
        "runtime_note": ("Observed on the environment above; not a universal "
                         "benchmark. Cumulative time sums all checkpoint "
                         "segments; the current invocation is reported "
                         "separately."),
        "includes_saliency": False,
        "num_images": total_usable,
        "features": norms,
    }

    # (F) Write the provenance sidecar and the aggregate JSON atomically.
    sidecar_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint_sha,
        "csv_basename": os.path.basename(args.csv),
        "segments": all_segments,
        "invocation_count": len(all_segments),
        "runtime_cumulative_seconds": cumulative,
    }
    _refuse_protected(sidecar, "provenance sidecar")
    _atomic_write_json(sidecar, sidecar_payload)

    _refuse_protected(args.out, "aggregate JSON")
    _atomic_write_json(args.out, payload)

    print(f"\nWrote {total_usable} rows -> {os.path.basename(args.csv)}")
    print(f"Wrote provenance sidecar -> {os.path.basename(sidecar)}")
    print(f"Wrote canonical visual norms -> {os.path.basename(args.out)}")
    print(f"  schema={SCHEMA_VERSION} domain={ANALYSIS_DOMAIN} "
          f"long_side={long_side} subsample={is_subsample}")
    print(f"  run_fingerprint_sha256={fingerprint_sha}")
    print(f"  processed_category_counts={dict(processed_counts)} "
          f"failed={n_failed}")
    print(f"  runtime current={round(elapsed, 3)}s cumulative={cumulative}s "
          f"invocations={len(all_segments)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GeneratorError as exc:
        print(f"GeneratorError: {exc}", file=sys.stderr)
        raise SystemExit(2)
