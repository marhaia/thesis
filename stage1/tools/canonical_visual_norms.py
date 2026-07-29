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
import hashlib
import json
import os
import platform
import sys
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
                    help="explicitly resume from an existing matching-provenance CSV")
    args = ap.parse_args(argv)

    # (5) Hard refusal to target protected files.
    _refuse_protected(args.out, "aggregate JSON")
    _refuse_protected(args.csv, "per-image CSV")

    long_side = int(args.long_side)
    exp = expected_provenance(long_side)
    is_subsample = args.limit is not None

    images = load_authorized_images(args.types_csv, args.images_dir, args.limit)
    if not images:
        raise GeneratorError("No authorized images discovered.")
    # Poster can never appear post-discovery; assert defensively.
    assert all(c in _AUTHORIZED_SET for _, c in images), "poster leaked"

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    feature_order = list(VISUAL_FEATURE_KEYS)
    fieldnames = ["filename", "category", *_PROVENANCE_COLUMNS, *feature_order]

    values_by_feature: Dict[str, List[float]] = collections.defaultdict(list)
    per_image_rows: List[dict] = []

    # Resume handling: only when explicitly requested AND provenance matches.
    done_names: set = set()
    if os.path.exists(args.csv) and os.path.getsize(args.csv) > 0:
        if not args.resume:
            raise GeneratorError(
                f"Output CSV '{args.csv}' already exists. Refusing to append "
                f"implicitly. Pass --resume (provenance is verified) or choose "
                f"a fresh --csv path.")
        existing, done_names = load_existing_csv_for_resume(args.csv, long_side)
        per_image_rows.extend(existing)
        for row in existing:
            for k in feature_order:
                values_by_feature[k].append(row[k])
        print(f"Resuming: {len(done_names)} rows with matching provenance in "
              f"{os.path.basename(args.csv)}.")

    csv_exists = os.path.exists(args.csv) and os.path.getsize(args.csv) > 0
    t_start = time.time()
    with open(args.csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
            csv_file.flush()
        n_new = 0
        for idx, (name, category) in enumerate(images, start=1):
            if name in done_names:
                continue
            img_path = os.path.join(args.images_dir, name)
            try:
                # (3) Exact production canonical preprocessing + 8 visual feats.
                v = compute_complexity_vector(img_path)
                row = {"filename": name, "category": category,
                       "analysis_domain": exp["analysis_domain"],
                       "canonical_long_side": exp["canonical_long_side"],
                       "canonical_analysis_version": exp["canonical_analysis_version"]}
                for k in feature_order:
                    row[k] = float(v[k])
                    values_by_feature[k].append(row[k])
                per_image_rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                n_new += 1
                print(f"[{idx}/{len(images)}] {name} ({category})")
            except Exception as exc:  # keep going on a single bad image
                print(f"[{idx}/{len(images)}] ERROR {name}: {exc}")
    elapsed = time.time() - t_start

    # Recompute per-category counts from what we actually have.
    processed_counts = collections.Counter(r["category"] for r in per_image_rows)
    total_usable = len(per_image_rows)

    # (12) Full-run integrity: exactly 1485 usable & 495 per authorized category.
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

    norms = aggregate(values_by_feature)

    # (13) No personal absolute paths in outputs: identify the corpus by hash and
    # basenames only.
    types_hash = _sha256_file(args.types_csv)
    corpus_hash = authorized_aggregate_hash(images, args.images_dir)
    extractor_hash = _sha256_file(str(_STAGE1_DIR / "visual_complexity.py"))
    generator_hash = _sha256_file(str(_THIS_FILE))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_list": list(VISUAL_FEATURE_KEYS),
        "analysis_domain": ANALYSIS_DOMAIN,
        "canonical_long_side": long_side,
        "canonical_analysis_version": exp["canonical_analysis_version"],
        "interpolation_behavior": "INTER_AREA when downscaling, "
                                  "INTER_LINEAR when upscaling (aspect preserved)",
        "input_dimension_contract": {
            "min_long_side_px": int(MIN_CANONICAL_INPUT_LONG_SIDE),
            "min_short_side_px": int(MIN_CANONICAL_INPUT_SHORT_SIDE),
        },
        "authorized_categories": list(AUTHORIZED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "category_counts": dict(processed_counts),
        "num_images": total_usable,
        "is_subsample": is_subsample,
        "image_types_csv_sha256": types_hash,
        "image_types_csv_basename": os.path.basename(args.types_csv),
        "authorized_corpus_aggregate_sha256": corpus_hash,
        "production_extractor_sha256": extractor_hash,
        "generator_sha256": generator_hash,
        "environment": environment_info(),
        "runtime_seconds_observed": round(elapsed, 3),
        "runtime_note": ("Observed on the environment above; not a universal "
                         "benchmark."),
        "includes_saliency": False,
        "features": norms,
    }

    # Final protected-path guard immediately before writing.
    _refuse_protected(args.out, "aggregate JSON")
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {total_usable} rows -> {os.path.basename(args.csv)}")
    print(f"Wrote canonical visual norms -> {os.path.basename(args.out)}")
    print(f"  schema={SCHEMA_VERSION} domain={ANALYSIS_DOMAIN} "
          f"long_side={long_side} subsample={is_subsample}")
    print(f"  category_counts={dict(processed_counts)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GeneratorError as exc:
        print(f"GeneratorError: {exc}", file=sys.stderr)
        raise SystemExit(2)
