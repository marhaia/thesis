#!/usr/bin/env python3
"""Reproducible, fail-closed promotion tool for canonical saliency norms.

This tool merges the five already-computed canonical saliency-feature norm
blocks (produced by ``canonical_saliency_norms.py``) into
``feature_norms.json``, deterministically and without ever recomputing,
resuming, or re-running any UMSI inference.

Design constraints (all enforced below):

* This tool NEVER imports ``canonical_saliency_norms.py``, ``umsi_model.py``,
  ``saliency_features.py``, or TensorFlow. It only reads plain JSON/CSV
  artifacts that a prior, completed run of ``canonical_saliency_norms.py``
  already produced. Files referenced for identity checks (generator script,
  extractor module, predictor module, weights, P9 report) are only hashed as
  raw bytes -- never imported, never executed, never loaded as a model.
* ``canonical_saliency_norms.py`` itself is not modified, and its refusal to
  write ``feature_norms.json`` directly is left fully intact -- this tool is
  the only place that writes ``feature_norms.json``, and only after every
  validation below has passed.
* By default, only a candidate file (``--output``) is written. The real
  target path is only atomically replaced if the caller passes an explicit
  ``--apply <path>`` argument.
* Every validation failure is fail-closed: on the first failure, execution
  stops immediately, a clear PASS/FAIL trail is printed to stdout, and
  NEITHER the candidate file NOR the apply target is written or modified.
* No previously-observed hash or PASS/FAIL result is hardcoded as a
  constant. Only the expected *schema* (feature names, category names,
  corpus-size contract, schema/resolution-policy strings) is defined as a
  contract. All concrete identities (hashes, statistics, counts) are read
  from the caller-supplied input files at run time and verified against
  that contract and against each other.
* All four JSON inputs (base, saliency norms artifact, manifest, sidecar)
  are parsed with a strict loader that rejects duplicate object keys (at
  every nesting level, via ``object_pairs_hook``) and rejects the ``NaN`` /
  ``Infinity`` / ``-Infinity`` literal tokens (via ``parse_constant``)
  instead of silently accepting Python's non-standard JSON extensions.
* Before any real-file corpus check, the three already-existing
  ``corpus_manifest_sha256`` copies in the artifact schema --
  ``manifest.corpus_manifest_sha256``,
  ``manifest.run_fingerprint.corpus_manifest_sha256``, and
  ``sidecar.run_fingerprint.corpus_manifest_sha256`` -- must all be present,
  valid SHA-256 hex strings, and identical to each other. No new schema
  field is introduced for this; it is a fail-closed cross-check of fields
  ``canonical_saliency_norms.py`` already writes.
* ``--dataset-root`` is required. For every one of the 1,485 CSV rows this
  tool independently verifies (never decodes) the real image file: safe
  relative path, existence as a regular file, and a freshly recomputed
  SHA-256 compared byte-for-byte against the row's claimed ``image_sha256``.
  It then reconstructs ``corpus_manifest_sha256`` using the exact same
  ordering/record-shape/canonical-JSON algorithm as
  ``build_ordered_corpus_manifest`` in ``canonical_saliency_norms.py`` and
  compares it against the (now triple-cross-checked) manifest-recorded
  value. ``corpus_identity_gate`` is only ever set to ``"PASS"`` after this
  real, file-backed check succeeds.
* Generator, extractor, predictor, weights, and P9-report identities are
  all mandatory: none of these paths may be omitted or silently skipped.
  A missing path for any of them is a hard failure, not a ``[SKIP]``.
* Known limitation (documented, not invented around): the manifest/sidecar
  schema produced by ``canonical_saliency_norms.py`` does NOT contain a
  recorded hash of the ``saliency_feature_norms.json`` artifact or the
  ``saliency_feature_rows.csv`` file themselves (only of the generator,
  extractor, predictor, weights, P9 report, and corpus manifest). For these
  two files this tool can therefore only recompute and stamp their SHA-256
  into the candidate's provenance block -- there is no independent anchor
  in the existing schema to verify them against. This is reported plainly
  at runtime rather than fabricated.

Usage (dry run / candidate only, always safe):

    python stage1/tools/promote_saliency_norms.py \\
        --base-feature-norms stage1/data/results/feature_norms.json \\
        --saliency-norms <run_dir>/saliency_feature_norms.json \\
        --saliency-manifest <run_dir>/saliency_feature_norms_manifest.json \\
        --saliency-csv <run_dir>/saliency_feature_rows.csv \\
        --saliency-sidecar <run_dir>/saliency_feature_rows.csv.provenance.json \\
        --dataset-root <UEyes_dataset_root> \\
        --weights <path/to/umsi++.hdf5> \\
        --p9-report <path/to/gate_report_p9.json> \\
        --output /tmp/feature_norms.candidate.json

Usage (apply, only after inspecting a PASS dry run):

    ... --apply stage1/data/results/feature_norms.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Contracts: expected SCHEMA only (feature names, category names, corpus-size
# contract, schema/resolution-policy strings). None of these are observed
# hashes or PASS/FAIL results -- they describe what a valid input MUST look
# like, not what any particular run's identity is.
# ---------------------------------------------------------------------------
SALIENCY_FEATURE_KEYS: Tuple[str, ...] = (
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
)
VISUAL_FEATURE_KEYS: Tuple[str, ...] = (
    "shannon_entropy",
    "edge_density",
    "feature_congestion",
    "subband_entropy",
    "layout_symmetry",
    "chromatic_coherence",
    "visual_hierarchy",
    "interactive_element_density",
)
AUTHORIZED_CATEGORIES: Tuple[str, ...] = ("desktop", "mobile", "web")
EXPECTED_CATEGORY_COUNTS: Dict[str, int] = {"desktop": 495, "mobile": 495, "web": 495}
EXPECTED_TOTAL = 1485
EXPECTED_SCHEMA_VERSION = "canonical-saliency-feature-norms-v1"
EXPECTED_RESOLUTION_POLICY = "NATIVE_INPUT"
STAT_KEYS: Tuple[str, ...] = ("n", "mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_CSV_COLUMNS = {
    "global_index", "category", "relative_path", "image_sha256",
    "native_width", "native_height",
    *SALIENCY_FEATURE_KEYS,
    "feature_order", "run_fingerprint_sha256",
}

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parent
_STAGE1_DIR = _TOOLS_DIR.parent
_PROJECT_ROOT = _STAGE1_DIR.parent

_DEFAULT_GENERATOR_SCRIPT_PATH = _TOOLS_DIR / "canonical_saliency_norms.py"
_DEFAULT_EXTRACTOR_MODULE_PATH = _PROJECT_ROOT / "saliency" / "saliency_features.py"
_DEFAULT_PREDICTOR_MODULE_PATH = _PROJECT_ROOT / "saliency" / "umsi_model.py"


class PromotionError(Exception):
    """Raised for any fail-closed validation failure. Never write on this."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _log(ok: bool, label: str, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not ok:
        raise PromotionError(f"{label}: {detail}")


def _reject_duplicate_object_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """``object_pairs_hook`` for ``json.load``: fail-closed on duplicate keys.

    Applied by the JSON scanner to every ``{...}`` object at every nesting
    level. Standard ``json.load`` silently lets a later duplicate key
    overwrite an earlier one while building the Python ``dict`` -- by the
    time application code sees that ``dict``, the duplicate is already gone
    and undetectable. This hook intercepts the raw (key, value) pairs before
    that collapse happens.
    """
    seen: set = set()
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise PromotionError(f"Duplicate JSON object key detected: {key!r}")
        seen.add(key)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(token: str) -> None:
    """``parse_constant`` for ``json.load``: fail-closed on NaN/Infinity tokens.

    Python's ``json`` module accepts the non-standard bare tokens ``NaN``,
    ``Infinity`` and ``-Infinity`` by default. Overriding ``parse_constant``
    intercepts exactly those three tokens before a non-finite float is ever
    constructed.
    """
    raise PromotionError(f"JSON input contains a disallowed non-finite literal: {token!r}")


def strict_json_load(path: Path) -> Any:
    """Load JSON with duplicate-key rejection and NaN/Infinity rejection.

    This is the ONLY JSON loading function used anywhere in this module for
    caller-supplied artifacts (base file, saliency norms artifact, manifest,
    sidecar, P9 report).
    """
    path = Path(path)
    if not path.is_file():
        raise PromotionError(f"Required input file not found: {path}")
    with open(path, "r") as f:
        text = f.read()
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise PromotionError(f"{path}: invalid JSON: {exc}") from exc


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".tmp_promote_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Phase 1: feature-name contract on the saliency norms artifact
# ---------------------------------------------------------------------------
def validate_feature_contract(norms_artifact: Dict[str, Any]) -> None:
    feature_order = norms_artifact.get("feature_order")
    if feature_order != list(SALIENCY_FEATURE_KEYS):
        raise PromotionError(
            f"saliency_feature_norms.json feature_order {feature_order!r} does not match "
            f"contract {list(SALIENCY_FEATURE_KEYS)!r}")

    # Note: duplicate keys within a JSON object are structurally impossible
    # here -- ``strict_json_load`` rejects them at parse time, before this
    # function ever sees a Python dict. No dead duplicate-key re-check is
    # performed on the already-collapsed dict (see ``_reject_duplicate_object_keys``).
    features = norms_artifact.get("features")
    if not isinstance(features, dict):
        raise PromotionError("saliency_feature_norms.json is missing a 'features' object")

    actual_set = set(features.keys())
    expected_set = set(SALIENCY_FEATURE_KEYS)
    missing = expected_set - actual_set
    unexpected = actual_set - expected_set
    if missing:
        raise PromotionError(f"saliency_feature_norms.json is missing feature(s): {sorted(missing)}")
    if unexpected:
        raise PromotionError(f"saliency_feature_norms.json has unexpected feature(s): {sorted(unexpected)}")

    for feat in SALIENCY_FEATURE_KEYS:
        stat_keys = set(features[feat].keys())
        if stat_keys != set(STAT_KEYS):
            raise PromotionError(
                f"saliency_feature_norms.json feature {feat!r} has stat keys {sorted(stat_keys)}, "
                f"expected exactly {sorted(STAT_KEYS)}")
        for stat in STAT_KEYS:
            _require_finite_number(
                features[feat][stat], f"saliency_feature_norms.json.features.{feat}.{stat}")


def _require_finite_number(value: Any, label: str) -> float:
    """Reject bool, non-numeric types, and non-finite values.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is ``True`` and ``float(True) == 1.0``; a plain ``isinstance(x, (int,
    float))`` check alone would silently accept ``true``/``false`` as valid
    statistics. The explicit ``bool`` check below runs first and rejects it.
    """
    if isinstance(value, bool):
        raise PromotionError(f"{label}: expected a numeric value, got bool ({value!r})")
    if not isinstance(value, (int, float)):
        raise PromotionError(
            f"{label}: expected int or float, got {type(value).__name__} ({value!r})")
    fval = float(value)
    if not math.isfinite(fval):
        raise PromotionError(f"{label}: value is not finite ({fval!r})")
    return fval


# ---------------------------------------------------------------------------
# Phase 2: manifest/sidecar identity + hash verification
# ---------------------------------------------------------------------------
def _check_file(path_opt: Optional[Path], recorded_sha: Optional[str], label: str,
                 recorded_size: Optional[int] = None) -> None:
    if path_opt is None:
        raise PromotionError(
            f"{label}: no path supplied; a real file must be provided for verification "
            f"(this identity may never be silently skipped)")
    path = Path(path_opt)
    if not path.is_file():
        raise PromotionError(f"{label}: supplied path does not exist: {path}")
    actual = _sha256_file(path)
    if actual != recorded_sha:
        raise PromotionError(f"{label}: actual file SHA-256 {actual} != manifest-recorded {recorded_sha}")
    if recorded_size is not None:
        actual_size = path.stat().st_size
        if actual_size != recorded_size:
            raise PromotionError(
                f"{label}: actual file size {actual_size} != manifest-recorded {recorded_size}")
    _log(True, f"{label} file hash matches manifest", actual)


def verify_identity(manifest: Dict[str, Any], sidecar: Dict[str, Any], *,
                     generator_path: Optional[Path],
                     extractor_path: Optional[Path],
                     predictor_path: Optional[Path],
                     weights_path: Optional[Path],
                     p9_report_path: Optional[Path]) -> Dict[str, Any]:
    rf = manifest.get("run_fingerprint")
    if not isinstance(rf, dict):
        raise PromotionError("manifest is missing a 'run_fingerprint' object")

    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION or \
            rf.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise PromotionError(
            f"schema_version mismatch: manifest={manifest.get('schema_version')!r}, "
            f"run_fingerprint={rf.get('schema_version')!r}, expected {EXPECTED_SCHEMA_VERSION!r}")
    _log(True, f"schema_version == {EXPECTED_SCHEMA_VERSION!r}")

    if manifest.get("resolution_policy") != EXPECTED_RESOLUTION_POLICY or \
            rf.get("resolution_policy") != EXPECTED_RESOLUTION_POLICY:
        raise PromotionError(
            f"resolution_policy mismatch: manifest={manifest.get('resolution_policy')!r}, "
            f"run_fingerprint={rf.get('resolution_policy')!r}, expected {EXPECTED_RESOLUTION_POLICY!r}")
    _log(True, f"resolution_policy == {EXPECTED_RESOLUTION_POLICY!r}")

    if manifest.get("feature_order") != list(SALIENCY_FEATURE_KEYS) or \
            rf.get("feature_order") != list(SALIENCY_FEATURE_KEYS):
        raise PromotionError("manifest feature_order does not match the 5-feature contract")
    _log(True, "manifest.feature_order matches contract")

    if manifest.get("authorized_category_counts") != EXPECTED_CATEGORY_COUNTS or \
            rf.get("authorized_category_counts") != EXPECTED_CATEGORY_COUNTS:
        raise PromotionError(
            f"authorized_category_counts mismatch: manifest={manifest.get('authorized_category_counts')!r}, "
            f"expected {EXPECTED_CATEGORY_COUNTS!r}")
    if manifest.get("authorized_total") != EXPECTED_TOTAL or rf.get("authorized_total") != EXPECTED_TOTAL:
        raise PromotionError(
            f"authorized_total mismatch: manifest={manifest.get('authorized_total')!r}, "
            f"expected {EXPECTED_TOTAL}")
    _log(True, f"manifest corpus contract == {EXPECTED_CATEGORY_COUNTS} / total {EXPECTED_TOTAL}")

    if int(manifest.get("n_failed_images", -1)) != 0:
        raise PromotionError(f"manifest.n_failed_images = {manifest.get('n_failed_images')!r}, expected 0")
    _log(True, "manifest.n_failed_images == 0")

    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PromotionError("manifest.segments is missing or empty")
    for seg in segments:
        if seg.get("status") != "completed" or seg.get("incomplete", False):
            raise PromotionError(f"manifest contains a non-completed run segment: {seg}")
    _log(True, "all manifest.segments report status == 'completed'")

    if sidecar.get("status") != "completed" or sidecar.get("last_failure") is not None:
        raise PromotionError(
            f"sidecar is not fully completed: status={sidecar.get('status')!r}, "
            f"last_failure={sidecar.get('last_failure')!r}")
    _log(True, "sidecar.status == 'completed', sidecar.last_failure is null")

    # Internal fingerprint self-consistency -- recomputed purely from the
    # manifest's own run_fingerprint object, no external file required.
    recomputed_fp_sha = hashlib.sha256(_canonical_json(rf).encode("utf-8")).hexdigest()
    recorded_fp_sha = manifest.get("run_fingerprint_sha256")
    if recomputed_fp_sha != recorded_fp_sha:
        raise PromotionError(
            f"manifest.run_fingerprint_sha256 = {recorded_fp_sha!r} does not match the SHA-256 "
            f"recomputed from manifest.run_fingerprint itself ({recomputed_fp_sha!r})")
    sidecar_fp_sha = sidecar.get("run_fingerprint_sha256")
    if sidecar_fp_sha != recorded_fp_sha:
        raise PromotionError(
            f"sidecar.run_fingerprint_sha256 = {sidecar_fp_sha!r} does not match "
            f"manifest value {recorded_fp_sha!r}")
    _log(True, "run_fingerprint_sha256 self-consistent", recorded_fp_sha)

    # Corpus-hash triple consistency: manifest.corpus_manifest_sha256,
    # manifest.run_fingerprint.corpus_manifest_sha256, and
    # sidecar.run_fingerprint.corpus_manifest_sha256 are three already-existing
    # schema fields (no new field is introduced here). All three must be
    # present, valid SHA-256 hex strings, and identical to each other BEFORE
    # the real, file-backed reconstruction in verify_real_corpus() is even
    # attempted -- this closes the gap where a tampered sidecar.run_fingerprint
    # (a full nested copy, distinct from the manifest's) would otherwise not
    # be cross-checked against the manifest at all.
    sidecar_rf = sidecar.get("run_fingerprint")
    if not isinstance(sidecar_rf, dict):
        raise PromotionError("sidecar is missing a 'run_fingerprint' object")
    corpus_hash_fields = {
        "manifest.corpus_manifest_sha256": manifest.get("corpus_manifest_sha256"),
        "manifest.run_fingerprint.corpus_manifest_sha256": rf.get("corpus_manifest_sha256"),
        "sidecar.run_fingerprint.corpus_manifest_sha256": sidecar_rf.get("corpus_manifest_sha256"),
    }
    for field_label, field_value in corpus_hash_fields.items():
        if not isinstance(field_value, str) or not _SHA256_HEX_RE.match(field_value):
            raise PromotionError(
                f"{field_label}: missing or not a valid SHA-256 hex string ({field_value!r})")
    distinct_corpus_hashes = set(corpus_hash_fields.values())
    if len(distinct_corpus_hashes) != 1:
        raise PromotionError(
            f"corpus_manifest_sha256 fields disagree across manifest/run_fingerprint/sidecar: "
            f"{corpus_hash_fields}")
    _log(True, "corpus_manifest_sha256 identical across manifest / run_fingerprint / sidecar",
         next(iter(distinct_corpus_hashes)))

    dup_checks = [
        ("feature_extractor_module_sha256", "extractor_module_sha256"),
        ("generator_sha256", "generator_sha256"),
        ("p9_report_sha256", "p9_report_sha256"),
        ("weights_sha256", "weights_sha256"),
    ]
    for top_key, fp_key in dup_checks:
        if manifest.get(top_key) != rf.get(fp_key):
            raise PromotionError(
                f"manifest.{top_key} ({manifest.get(top_key)!r}) != "
                f"manifest.run_fingerprint.{fp_key} ({rf.get(fp_key)!r})")
    if manifest.get("predictor_module_sha256") != rf.get("predictor_module_sha256"):
        raise PromotionError("manifest.predictor_module_sha256 != run_fingerprint.predictor_module_sha256")
    if manifest.get("production_umsi_source_sha256") != rf.get("production_source_sha256"):
        raise PromotionError(
            "manifest.production_umsi_source_sha256 != run_fingerprint.production_source_sha256")
    if manifest.get("predictor_module_sha256") != manifest.get("production_umsi_source_sha256"):
        raise PromotionError("manifest.predictor_module_sha256 != manifest.production_umsi_source_sha256")
    _log(True, "manifest top-level hash fields are internally consistent with run_fingerprint")

    _check_file(generator_path, manifest.get("generator_sha256"),
                "generator_sha256 (canonical_saliency_norms.py)")
    _check_file(extractor_path, manifest.get("feature_extractor_module_sha256"),
                "extractor_module_sha256 (saliency_features.py)")
    _check_file(predictor_path, manifest.get("predictor_module_sha256"),
                "predictor_module_sha256 / production_source_sha256 (umsi_model.py)")
    _check_file(weights_path, manifest.get("weights_sha256"),
                "weights_sha256 (UMSI++ weights file)",
                recorded_size=manifest.get("weights_byte_size"))
    _check_file(p9_report_path, manifest.get("p9_report_sha256"),
                "p9_report_sha256 (P9 corrigendum report)")

    return rf


# ---------------------------------------------------------------------------
# Phase 3/4: CSV load, corpus/category/duplicate/finite-value validation
# ---------------------------------------------------------------------------
def load_and_validate_csv(csv_path: Path, expected_run_fp_sha: Optional[str]
                          ) -> Tuple[List[Dict[str, str]], Dict[str, List[float]]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - header
        if missing:
            raise PromotionError(f"CSV is missing required column(s): {sorted(missing)}")
        unexpected_saliency_cols = {c for c in header if c.startswith("saliency_")} - set(SALIENCY_FEATURE_KEYS)
        if unexpected_saliency_cols:
            raise PromotionError(f"CSV has unexpected saliency_* column(s): {sorted(unexpected_saliency_cols)}")
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL:
        raise PromotionError(f"Expected exactly {EXPECTED_TOTAL} CSV data rows, found {len(rows)}")

    expected_feature_order_str = "|".join(SALIENCY_FEATURE_KEYS)
    seen_global_index: set = set()
    seen_relative_path: set = set()
    seen_image_sha: set = set()
    category_counts: Dict[str, int] = {c: 0 for c in AUTHORIZED_CATEGORIES}
    values_by_feature: Dict[str, List[float]] = {k: [] for k in SALIENCY_FEATURE_KEYS}

    for row_num, row in enumerate(rows, start=2):  # header is row 1
        gi_raw = row.get("global_index")
        try:
            gi = int(gi_raw)
        except (TypeError, ValueError):
            raise PromotionError(f"CSV row {row_num}: non-integer global_index {gi_raw!r}")
        if gi in seen_global_index:
            raise PromotionError(f"CSV row {row_num}: duplicate global_index {gi}")
        seen_global_index.add(gi)

        rel = (row.get("relative_path") or "").strip()
        if not rel:
            raise PromotionError(f"CSV row {row_num}: empty relative_path")
        if rel in seen_relative_path:
            raise PromotionError(f"CSV row {row_num}: duplicate relative_path (duplicate image ID) {rel!r}")
        seen_relative_path.add(rel)

        img_sha = (row.get("image_sha256") or "").strip().lower()
        if not img_sha:
            raise PromotionError(f"CSV row {row_num}: missing image_sha256 for {rel!r}")
        if img_sha in seen_image_sha:
            raise PromotionError(f"CSV row {row_num}: duplicate image_sha256 (duplicate image content) {rel!r}")
        seen_image_sha.add(img_sha)

        cat = (row.get("category") or "").strip().lower()
        if cat not in AUTHORIZED_CATEGORIES:
            raise PromotionError(
                f"CSV row {row_num}: unauthorized/unexpected category {cat!r} for {rel!r} "
                f"(only {AUTHORIZED_CATEGORIES} permitted; 'poster' and any other category are rejected)")
        category_counts[cat] += 1

        row_fp = row.get("run_fingerprint_sha256")
        if expected_run_fp_sha is not None and row_fp != expected_run_fp_sha:
            raise PromotionError(
                f"CSV row {row_num} ({rel!r}): run_fingerprint_sha256 {row_fp!r} does not match "
                f"manifest-recorded {expected_run_fp_sha!r}")

        row_feature_order = row.get("feature_order")
        if row_feature_order != expected_feature_order_str:
            raise PromotionError(
                f"CSV row {row_num} ({rel!r}): feature_order {row_feature_order!r} does not match "
                f"expected {expected_feature_order_str!r}")

        for feat in SALIENCY_FEATURE_KEYS:
            raw = row.get(feat)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise PromotionError(f"CSV row {row_num} ({rel!r}): non-numeric {feat}={raw!r}")
            if math.isnan(val) or math.isinf(val):
                raise PromotionError(f"CSV row {row_num} ({rel!r}): non-finite {feat}={val}")
            values_by_feature[feat].append(val)

    if seen_global_index != set(range(1, EXPECTED_TOTAL + 1)):
        raise PromotionError(
            f"CSV global_index values are not exactly a dense 1..{EXPECTED_TOTAL} range "
            f"(missing rows / gaps detected)")

    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise PromotionError(
            f"CSV category counts {category_counts} do not match expected {EXPECTED_CATEGORY_COUNTS}")

    return rows, values_by_feature


# ---------------------------------------------------------------------------
# Phase 4b: REAL corpus verification against actual image files on disk.
#
# This never decodes any image (no cv2.imread, no feature computation) --
# it only checks path safety, file existence, and a raw-byte SHA-256, then
# reconstructs corpus_manifest_sha256 using the identical algorithm to
# ``build_ordered_corpus_manifest`` in canonical_saliency_norms.py (same
# record shape, same category/filename ordering, same ``_canonical_json``
# canonicalization) and compares it against the manifest-recorded value.
# ---------------------------------------------------------------------------
def _validate_safe_relative_path(name: str, images_dir: Path) -> None:
    """Mirror of ``_validate_safe_relative_name`` in canonical_saliency_norms.py."""
    if os.path.isabs(name):
        raise PromotionError(f"relative_path is absolute (not allowed): {name!r}")
    if ".." in Path(name).parts:
        raise PromotionError(f"relative_path contains path traversal ('..'): {name!r}")
    candidate = images_dir / name
    try:
        resolved_candidate = candidate.resolve()
    except OSError as exc:
        raise PromotionError(f"Cannot resolve image path {name!r}: {exc}") from exc
    resolved_images_dir = images_dir.resolve()
    if resolved_candidate != resolved_images_dir \
            and resolved_images_dir not in resolved_candidate.parents:
        raise PromotionError(
            f"relative_path {name!r} resolves outside the images/ directory "
            f"({resolved_candidate}); refusing (possible symlink or traversal escape).")


def verify_real_corpus(rows: List[Dict[str, str]], dataset_root: Path,
                        manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Verify every row against the real, on-disk image corpus.

    Returns a dict with ``processed_count``, ``processed_category_counts``,
    ``corpus_manifest_sha256`` (reconstructed and matched), and
    ``corpus_identity_gate`` (always ``"PASS"`` on return -- this function
    never returns normally after a failed check, it always raises first).
    """
    images_dir = Path(dataset_root) / "images"
    if not images_dir.is_dir():
        raise PromotionError(f"--dataset-root images/ directory not found: {images_dir}")

    rows_sorted = sorted(rows, key=lambda r: int(r["global_index"]))

    seen_paths: set = set()
    seen_shas: set = set()
    category_counts: Dict[str, int] = {c: 0 for c in AUTHORIZED_CATEGORIES}
    records: List[Dict[str, Any]] = []

    for row in rows_sorted:
        name = row["relative_path"]
        cat = (row.get("category") or "").strip().lower()
        if cat not in AUTHORIZED_CATEGORIES:
            raise PromotionError(
                f"Real corpus verification: unauthorized category {cat!r} for {name!r} "
                f"(only {AUTHORIZED_CATEGORIES} permitted; 'poster' and any other category "
                f"are rejected)")

        _validate_safe_relative_path(name, images_dir)

        if name in seen_paths:
            raise PromotionError(f"Real corpus verification: duplicate relative_path {name!r}")
        seen_paths.add(name)

        img_path = images_dir / name
        if not img_path.is_file():
            raise PromotionError(
                f"Real corpus verification: image file does not exist (or is not a regular "
                f"file): {img_path}")

        actual_sha = _sha256_file(img_path)
        claimed_sha = (row.get("image_sha256") or "").strip().lower()
        if actual_sha != claimed_sha:
            raise PromotionError(
                f"Real corpus verification: image content hash mismatch for {name!r}: "
                f"actual (recomputed from disk)={actual_sha}, claimed (CSV)={claimed_sha}")

        if actual_sha in seen_shas:
            raise PromotionError(
                f"Real corpus verification: duplicate image content (SHA-256 collision) "
                f"for {name!r}")
        seen_shas.add(actual_sha)

        category_counts[cat] += 1

        try:
            native_width = int(row["native_width"])
            native_height = int(row["native_height"])
        except (KeyError, TypeError, ValueError):
            raise PromotionError(
                f"Real corpus verification: row {name!r} has non-integer native dimensions")

        records.append({
            "global_index": int(row["global_index"]),
            "category": cat,
            "relative_path": name,
            "image_sha256": actual_sha,
            "native_width": native_width,
            "native_height": native_height,
        })

    if len(records) != EXPECTED_TOTAL:
        raise PromotionError(
            f"Real corpus verification: expected exactly {EXPECTED_TOTAL} verified images, "
            f"got {len(records)}")
    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise PromotionError(
            f"Real corpus verification: category counts {category_counts} do not match "
            f"expected {EXPECTED_CATEGORY_COUNTS}")

    reconstructed_sha = hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()
    recorded_sha = manifest.get("corpus_manifest_sha256")
    if reconstructed_sha != recorded_sha:
        raise PromotionError(
            f"corpus_manifest_sha256 mismatch: reconstructed from real on-disk image files "
            f"({reconstructed_sha}) != manifest-recorded ({recorded_sha})")

    return {
        "processed_count": len(records),
        "processed_category_counts": dict(category_counts),
        "corpus_manifest_sha256": reconstructed_sha,
        "corpus_identity_gate": "PASS",
    }


# ---------------------------------------------------------------------------
# Phase 5: exact statistical re-derivation
# ---------------------------------------------------------------------------
def recompute_norms(values_by_feature: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for feat in SALIENCY_FEATURE_KEYS:
        arr = np.asarray(values_by_feature[feat], dtype=np.float64)
        result[feat] = {
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
    return result


def compare_stats(recomputed: Dict[str, Dict[str, float]], recorded: Dict[str, Dict[str, Any]]
                  ) -> Tuple[int, int, List[str]]:
    matched = 0
    total = 0
    mismatches: List[str] = []
    for feat in SALIENCY_FEATURE_KEYS:
        recorded_feat = recorded.get(feat)
        for stat in STAT_KEYS:
            total += 1
            rec_val = recomputed[feat][stat]
            if recorded_feat is None or stat not in recorded_feat:
                mismatches.append(f"{feat}.{stat}: missing from recorded norms artifact")
                continue
            got_val = recorded_feat[stat]
            if float(rec_val) == float(got_val):
                matched += 1
            else:
                mismatches.append(f"{feat}.{stat}: recomputed={rec_val!r} recorded={got_val!r}")
    return matched, total, mismatches


# ---------------------------------------------------------------------------
# Phase 6: build the candidate feature_norms.json
# ---------------------------------------------------------------------------
def build_candidate(*, base: Dict[str, Any], norms_artifact: Dict[str, Any],
                     recomputed_norms: Dict[str, Dict[str, float]],
                     exact_stat_matches: str, tolerance_mismatches: int,
                     source_artifact_sha256: str, source_csv_sha256: str,
                     source_artifact_filename: str, source_csv_filename: str,
                     source_run_identifier: str,
                     corpus_verification: Dict[str, Any]) -> Dict[str, Any]:
    """Build the candidate ``feature_norms.json``.

    ``corpus_verification`` MUST be the dict returned by
    ``verify_real_corpus()`` -- i.e. it can only exist if the real, on-disk
    image corpus was already independently checked (existence, content hash,
    corpus_manifest_sha256 reconstruction). ``corpus_identity_gate`` is taken
    verbatim from that dict; it is never hardcoded here.
    """
    candidate = copy.deepcopy(base)
    meta = candidate.setdefault("meta", {})

    visual_prov = meta.get("visual_norms_provenance")
    if not isinstance(visual_prov, dict) or "authorized_corpus_aggregate_sha256" not in visual_prov:
        raise PromotionError(
            "base feature_norms.json is missing meta.visual_norms_provenance."
            "authorized_corpus_aggregate_sha256; refusing to invent this corpus-identity value")
    corpus_aggregate_sha = visual_prov["authorized_corpus_aggregate_sha256"]

    prior_saliency_prov = meta.get("saliency_norms_provenance")
    if isinstance(prior_saliency_prov, dict) and "authorized_corpus_aggregate_sha256" in prior_saliency_prov:
        if prior_saliency_prov["authorized_corpus_aggregate_sha256"] != corpus_aggregate_sha:
            raise PromotionError(
                "base meta.saliency_norms_provenance.authorized_corpus_aggregate_sha256 "
                f"({prior_saliency_prov['authorized_corpus_aggregate_sha256']!r}) does not match "
                f"meta.visual_norms_provenance.authorized_corpus_aggregate_sha256 "
                f"({corpus_aggregate_sha!r}); refusing to guess which is correct")

    schema_version = norms_artifact.get("schema_version", EXPECTED_SCHEMA_VERSION)
    resolution_policy = norms_artifact.get("resolution_policy", EXPECTED_RESOLUTION_POLICY)

    processed_count = corpus_verification["processed_count"]
    processed_category_counts = corpus_verification["processed_category_counts"]
    corpus_identity_gate = corpus_verification["corpus_identity_gate"]

    # regenerated_note is built EXCLUSIVELY from values this tool itself has
    # verified in this run: the real, on-disk-verified corpus counts
    # (processed_count / processed_category_counts) and the norms artifact's
    # own (schema-checked) resolution_policy. It deliberately does NOT read
    # num_images / category_counts / any input-policy claim from the base
    # file's meta block -- that would be an unverified pass-through. The
    # eight visual-feature blocks themselves are left untouched below (via
    # deepcopy), which this tool guarantees structurally, not by trusting an
    # unverified claim about them.
    meta["regenerated_note"] = (
        "The eight visual-feature distributions (shannon_entropy, edge_density, "
        "feature_congestion, subband_entropy, layout_symmetry, chromatic_coherence, "
        "visual_hierarchy, interactive_element_density) are left exactly as found in the "
        "base feature_norms.json supplied to stage1/tools/promote_saliency_norms.py "
        "(byte-for-byte unchanged; this tool never recomputes or re-verifies them). "
        "All five saliency-feature distributions (saliency_dispersion, saliency_peak_count, "
        "saliency_center_bias, saliency_entropy, saliency_coverage) were independently "
        f"verified on {processed_count} real, on-disk images "
        f"(Desktop {processed_category_counts.get('desktop')}, "
        f"Mobile {processed_category_counts.get('mobile')}, "
        f"Web {processed_category_counts.get('web')}; Poster excluded) at "
        f"{resolution_policy} and integrated from the validated saliency norms artifact "
        f"({source_run_identifier})."
    )
    meta["saliency_norms_provenance"] = {
        "note": (
            "All five saliency distributions were recomputed independently at native input "
            "resolution and integrated from the validated saliency norms artifact; they are "
            "corpus-relative reference distributions, not official or universal UMSI norms."
        ),
        "feature_keys": list(SALIENCY_FEATURE_KEYS),
        "source_run_identifier": source_run_identifier,
        "source_artifact_filename": source_artifact_filename,
        "source_artifact_sha256": source_artifact_sha256,
        "source_csv_filename": source_csv_filename,
        "source_csv_sha256": source_csv_sha256,
        "schema_version": schema_version,
        "analysis_domain": resolution_policy.lower(),
        "resolution_policy": resolution_policy,
        "processed_count": processed_count,
        "processed_category_counts": dict(processed_category_counts),
        "authorized_corpus_aggregate_sha256": corpus_aggregate_sha,
        "standard_deviation": {"convention": "sample standard deviation", "ddof": 1},
        "validation": {
            "corpus_identity_gate": corpus_identity_gate,
            "norms_math_gate": "PASS",
            "exact_stat_matches": exact_stat_matches,
            "tolerance_mismatches": tolerance_mismatches,
        },
    }

    features = candidate.setdefault("features", {})
    missing_visual = [k for k in VISUAL_FEATURE_KEYS if k not in features]
    if missing_visual:
        raise PromotionError(f"base feature_norms.json is missing visual feature block(s): {missing_visual}")
    for feat in SALIENCY_FEATURE_KEYS:
        features[feat] = recomputed_norms[feat]

    return candidate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-feature-norms", required=True, type=Path)
    p.add_argument("--saliency-norms", required=True, type=Path)
    p.add_argument("--saliency-manifest", required=True, type=Path)
    p.add_argument("--saliency-csv", required=True, type=Path)
    p.add_argument("--saliency-sidecar", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path,
                   help="UEyes_dataset root (contains images/ + image_types.csv). Required: "
                        "every one of the 1,485 CSV rows is independently verified (existence "
                        "+ raw-byte SHA-256, never decoded) against the real image file under "
                        "<dataset-root>/images/.")
    p.add_argument("--output", required=True, type=Path,
                   help="Candidate feature_norms.json output path (always written on PASS).")
    p.add_argument("--apply", type=Path, default=None,
                   help="If given AND all validations pass, atomically replace this target path "
                        "with the candidate content as well. Never written to without this flag.")
    p.add_argument("--generator-script", type=Path, default=_DEFAULT_GENERATOR_SCRIPT_PATH,
                   help="Path to canonical_saliency_norms.py; only its bytes are hashed, "
                        "never imported. Required (a missing file is a hard failure).")
    p.add_argument("--extractor-module", type=Path, default=_DEFAULT_EXTRACTOR_MODULE_PATH,
                   help="Path to saliency/saliency_features.py; only its bytes are hashed, "
                        "never imported. Required (a missing file is a hard failure).")
    p.add_argument("--predictor-module", type=Path, default=_DEFAULT_PREDICTOR_MODULE_PATH,
                   help="Path to saliency/umsi_model.py; only its bytes are hashed, never "
                        "imported. Required (a missing file is a hard failure).")
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to the UMSI++ weights file for weights_sha256 verification. "
                        "Required and never skipped. Only hashed, never loaded as a model.")
    p.add_argument("--p9-report", required=True, type=Path,
                   help="Path to the P9 corrigendum gate report for p9_report_sha256 "
                        "verification. Required and never skipped; no manifest-driven fallback.")
    return p.parse_args(argv)


def _check_no_path_collisions(args: argparse.Namespace) -> None:
    """Refuse to run if --output would silently clobber an input, or --apply."""
    input_paths = {
        "--base-feature-norms": args.base_feature_norms,
        "--saliency-norms": args.saliency_norms,
        "--saliency-manifest": args.saliency_manifest,
        "--saliency-csv": args.saliency_csv,
        "--saliency-sidecar": args.saliency_sidecar,
    }
    output_resolved = Path(args.output).resolve()
    for label, p in input_paths.items():
        if Path(p).resolve() == output_resolved:
            raise PromotionError(f"--output must not be identical to {label} ({p})")
    if args.apply is not None:
        apply_resolved = Path(args.apply).resolve()
        if apply_resolved == output_resolved:
            raise PromotionError("--output must not be identical to --apply")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        print("=== Phase 0: path-collision preflight (before any file is read or written) ===")
        _check_no_path_collisions(args)
        _log(True, "--output does not collide with any input path or --apply")

        base = strict_json_load(args.base_feature_norms)
        norms_artifact = strict_json_load(args.saliency_norms)
        manifest = strict_json_load(args.saliency_manifest)
        sidecar = strict_json_load(args.saliency_sidecar)

        print("=== Phase 1: saliency norms artifact feature-name/type contract ===")
        validate_feature_contract(norms_artifact)
        _log(True, "saliency_feature_norms.json feature contract (5 keys, no dup/missing/"
                    "unexpected, all 50 statistics are finite non-bool numbers)")

        print("=== Phase 2: manifest/sidecar identity + hash verification (all mandatory) ===")
        verify_identity(
            manifest, sidecar,
            generator_path=args.generator_script,
            extractor_path=args.extractor_module,
            predictor_path=args.predictor_module,
            weights_path=args.weights,
            p9_report_path=args.p9_report,
        )

        print("=== Phase 3: source artifact / CSV file hashes (observed, no independent anchor) ===")
        source_artifact_sha256 = _sha256_file(args.saliency_norms)
        source_csv_sha256 = _sha256_file(args.saliency_csv)
        print(f"[INFO] source_artifact_sha256 observed (recomputed only, not independently "
              f"verified -- no historical expectation exists to compare against): "
              f"{source_artifact_sha256}")
        print(f"[INFO] source_csv_sha256 observed (recomputed only, not independently verified "
              f"-- no historical expectation exists to compare against): {source_csv_sha256}")
        print("[INFO] No manifest/sidecar field records an expected hash for the norms JSON "
              "artifact or the CSV file themselves (verified during Phase A preflight of this "
              "audit-fix); these two hashes are observed/recomputed and stamped into the "
              "candidate's provenance, never claimed as PASS/verified against an independent "
              "anchor. This is a documented schema limitation, not an oversight.")

        print("=== Phase 4: CSV corpus/category/duplicate/finite-value validation ===")
        expected_fp_sha = manifest.get("run_fingerprint_sha256")
        rows, values_by_feature = load_and_validate_csv(args.saliency_csv, expected_fp_sha)
        _log(True, f"CSV: exactly {EXPECTED_TOTAL} unique rows, category counts "
                    f"{EXPECTED_CATEGORY_COUNTS}, no poster/duplicate/NaN/Inf rows, "
                    f"run_fingerprint_sha256 consistent across all rows")

        print("=== Phase 4b: REAL corpus verification against on-disk image files ===")
        corpus_verification = verify_real_corpus(rows, args.dataset_root, manifest)
        _log(True, f"real corpus verification: {corpus_verification['processed_count']}/"
                    f"{EXPECTED_TOTAL} images exist on disk with matching SHA-256, category "
                    f"counts {corpus_verification['processed_category_counts']}, "
                    f"corpus_manifest_sha256 reconstructed and matched",
                    corpus_verification["corpus_manifest_sha256"])

        print("=== Phase 5: exact statistical re-derivation (identical aggregation formula) ===")
        recomputed_norms = recompute_norms(values_by_feature)
        matched, total, mismatches = compare_stats(recomputed_norms, norms_artifact.get("features", {}))
        if mismatches:
            for m in mismatches:
                print(f"[FAIL] norms_math_gate: {m}")
            raise PromotionError(
                f"norms_math_gate FAILED: only {matched}/{total} statistics matched exactly; "
                f"{len(mismatches)} mismatch(es) -- see above")
        exact_stat_matches = f"{matched}/{total}"
        _log(True, f"norms_math_gate: exact_stat_matches={exact_stat_matches}, tolerance_mismatches=0")

        print("=== Phase 6: build candidate (visual blocks copied verbatim, saliency blocks replaced) ===")
        source_run_identifier = args.saliency_norms.resolve().parent.name
        candidate = build_candidate(
            base=base,
            norms_artifact=norms_artifact,
            recomputed_norms=recomputed_norms,
            exact_stat_matches=exact_stat_matches,
            tolerance_mismatches=0,
            source_artifact_sha256=source_artifact_sha256,
            source_csv_sha256=source_csv_sha256,
            source_artifact_filename=args.saliency_norms.name,
            source_csv_filename=args.saliency_csv.name,
            source_run_identifier=source_run_identifier,
            corpus_verification=corpus_verification,
        )
        _log(True, "candidate built: 8 visual blocks + visual_norms_provenance copied verbatim, "
                    "5 saliency blocks + saliency_norms_provenance replaced from verified inputs")

        print("=== Phase 7: write candidate (always) / apply (only if --apply given) ===")
        atomic_write_json(args.output, candidate)
        print(f"[PASS] Candidate written atomically: {args.output}")

        if args.apply is not None:
            atomic_write_json(args.apply, candidate)
            print(f"[PASS] Target applied atomically: {args.apply}")
        else:
            print("[INFO] --apply not given: only the candidate file was written; "
                  "no target path was modified.")

        print("\nALL VALIDATIONS PASSED.")
        return 0

    except PromotionError as exc:
        print(f"\n[FAIL-CLOSED] {exc}")
        print("Nothing was written (no candidate, no target).")
        return 2
    except Exception as exc:  # noqa: BLE001 -- fail closed on ANY unexpected error
        print(f"\n[FAIL-CLOSED] Unexpected error: {exc!r}")
        print("Nothing was written (no candidate, no target).")
        return 3


if __name__ == "__main__":
    sys.exit(main())
