#!/usr/bin/env python3
"""Dedicated, SAFE generator for the canonical SALIENCY feature-norm distribution.

This tool builds the future canonical reference distribution for exactly the
FIVE Stage-1 saliency-derived features, computed with the existing production
UMSI++ predictor (``saliency.umsi_model.UMSIPlus.predict_saliency``) and the
existing production saliency-feature extractor
(``saliency.saliency_features.extract_saliency_features``). It implements NO
alternative scientific formula of its own.

It is deliberately narrow, fail-closed and defensive:

  * SALIENCY-ONLY. It never computes or emits any of the eight Stage-1 visual
    features. The aggregate contains only the five saliency norms.
  * It writes ONLY to two dedicated, non-protected artifacts:
    ``saliency_feature_norms.json`` (the five norm distributions, in a shape
    compatible with later selective integration into ``feature_norms.json``)
    and ``saliency_feature_norms_manifest.json`` (all provenance / execution
    metadata, kept separate so the norm artifact itself stays minimal). It
    hard-refuses to write to, or under, any protected file or directory:
    ``feature_norms.json``, ``feature_norms_visual.json``,
    ``sensitivity_lookup.json``, the P8 evidence tree, the P9 evidence tree,
    the UMSI weights tree, or the production source files.
  * NATIVE-RESOLUTION saliency semantics. Every image is passed to
    ``predict_saliency`` by path, unresized, uncropped and unpadded, exactly
    as the live pipeline (``stage1/app.py``) already does. ``saliency_peak_count``
    and ``saliency_coverage`` are pixel-scale dependent, so no canonical-width
    scaling is ever applied here.
  * A hard, fail-closed P9 release-gate preflight runs BEFORE any dataset scan
    or model import: the declared P9 report content hash, its ``conclusion``
    and ``saliency_norms_release_status``, and the current production UMSI
    source / weights identities must all agree with the frozen P9 result.
    Any missing field or mismatch aborts before the model is ever imported.
  * TensorFlow / Keras / UMSI are imported LAZILY, inside a factory function,
    only after every other preflight (release gate, corpus contract, output
    path protection) has already passed. ``--preflight-only`` never reaches
    that import.
  * Only the authorized interactive GUI categories (desktop / mobile / web)
    are accepted; ``poster`` is excluded, and any other category is a hard
    corpus-contract violation (unlike a lenient silent-drop).
  * A run requires exactly the frozen corpus contract: 495 desktop + 495
    mobile + 495 web = 1,485 total. Any mismatch aborts before the feature
    extractor -- and before the model -- is ever invoked.
  * Every completed row is durably checkpointed (CSV + JSON provenance
    sidecar) so an interrupted run can be resumed. Resume is only performed
    when explicitly requested via ``--resume`` and only when the complete run
    fingerprint (dataset identity, generator, predictor module, extractor
    module, weights, environment, feature order, resolution policy) matches
    exactly; any mismatch aborts rather than mixing runs.
  * On the FIRST image failure, the run stops immediately: it returns
    nonzero, preserves the valid completed checkpoint rows, records the
    precise failed index/path/reason, and produces no aggregate artifact.
  * The final aggregate is written only once exactly 1,485 valid, unique rows
    have passed validation, and is written atomically (temp file, flush,
    fsync, ``os.replace``).

This script does NOT run the real 1,485-image generation as a side effect of
being authored, imported or tested. The expensive full run is a SEPARATELY
AUTHORIZED, manually invoked operation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths. Only the production SALIENCY-FEATURE EXTRACTOR is imported eagerly
# (it has no TensorFlow / Keras / UMSI dependency). The production PREDICTOR
# (``saliency.umsi_model``) is imported LAZILY -- see ``_default_predictor_factory``
# -- and only ever reached after every preflight below has already passed.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parent
_STAGE1_DIR = _TOOLS_DIR.parent
_PROJECT_ROOT = _STAGE1_DIR.parent
for _p in (str(_STAGE1_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from saliency.saliency_features import extract_saliency_features  # noqa: E402

# The five canonical SALIENCY features, in production order. This is the ONLY
# feature set this generator ever produces (no visual features).
SALIENCY_FEATURE_KEYS: Tuple[str, ...] = (
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
)

# The single authoritative authorized GUI population. ``poster`` is excluded.
AUTHORIZED_CATEGORIES: Tuple[str, ...] = ("desktop", "mobile", "web")
_AUTHORIZED_SET = frozenset(AUTHORIZED_CATEGORIES)
EXCLUDED_CATEGORIES: Tuple[str, ...] = ("poster",)

# Frozen full-run corpus contract.
FULL_RUN_TOTAL = 1485
FULL_RUN_PER_CATEGORY = 495
EXPECTED_AUTHORIZED_COUNTS = {"desktop": 495, "mobile": 495, "web": 495}

# Schema version / resolution policy for the emitted artifacts.
SCHEMA_VERSION = "canonical-saliency-feature-norms-v1"
RESOLUTION_POLICY = "NATIVE_INPUT"

# Frozen P9 release-gate literals (Stage 1, Step 2B corrigendum protocol).
P9_REQUIRED_CONCLUSION = "STEP_2B_P9_CORRIGENDUM_PASS"
P9_REQUIRED_NORMS_STATUS = "UNBLOCKED"
P9_REQUIRED_NEGATIVE_CONTROL_STATUS = "DISCRIMINATIVE"

# Immutable production constant: the ONLY P9 report this generator's default
# (non-test) code path will ever accept. A caller-supplied ``--p9-report-sha``
# for a different, self-consistent, even PASS-looking report is never
# sufficient by itself -- it must additionally equal this frozen value.
# Synthetic tests may monkeypatch this module attribute to a synthetic
# report's hash for success-path testing; production code always defaults to
# the real frozen hash below.
FROZEN_P9_REPORT_SHA256 = (
    "a71b68fc00c2be2e0dc6220054e47fb35a56f3e0bbe6bad311a7f7975ecc530d")

# Candidate installed-distribution names for the TensorFlow package, tried in
# order via ``importlib.metadata`` (NEVER via ``import tensorflow``). Some
# environments (notably Apple-Silicon macOS) install ``tensorflow-macos`` as
# the concrete distribution backing the ``tensorflow`` import name; both (and
# a couple of other known variants) are supported.
_TENSORFLOW_DISTRIBUTION_CANDIDATES: Tuple[str, ...] = (
    "tensorflow", "tensorflow-macos", "tensorflow-cpu", "tensorflow-intel",
    "tensorflow-aarch64", "tf-nightly",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Production module paths whose identity this generator verifies and records.
# ``predictor_module`` and ``production_source`` are literally the same file
# in this codebase (saliency/umsi_model.py); both names are kept because the
# release-gate compares the file against P9's recorded *source* identity while
# the run fingerprint records the *predictor module* identity -- they are
# always numerically identical here but conceptually distinct roles.
_PRODUCTION_SOURCE_PATH = _PROJECT_ROOT / "saliency" / "umsi_model.py"
_PREDICTOR_MODULE_PATH = _PRODUCTION_SOURCE_PATH
_EXTRACTOR_MODULE_PATH = _PROJECT_ROOT / "saliency" / "saliency_features.py"

# Protected files/directories this generator must NEVER write to or under.
_PROTECTED_BASENAMES = {
    "feature_norms.json",
    "feature_norms_visual.json",
    "sensitivity_lookup.json",
}
_PROTECTED_FILES = {
    (_STAGE1_DIR / "data" / "results" / "feature_norms.json").resolve(),
    (_STAGE1_DIR / "data" / "results" / "feature_norms_visual.json").resolve(),
    (_PROJECT_ROOT / "hceye" / "sensitivity_lookup.json").resolve(),
    _PRODUCTION_SOURCE_PATH.resolve(),
    _EXTRACTOR_MODULE_PATH.resolve(),
}
_PROTECTED_DIRS = {
    (_STAGE1_DIR / "evidence" / "step2b_results_p8").resolve(),
    (_STAGE1_DIR / "evidence" / "step2b_results_p9").resolve(),
    (_PROJECT_ROOT / "saliency" / "weights").resolve(),
}

ROW_FINGERPRINT_COLUMN = "run_fingerprint_sha256"
ROW_IMAGE_SHA_COLUMN = "image_sha256"
ROW_FEATURE_ORDER_COLUMN = "feature_order"

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class SaliencyNormsError(RuntimeError):
    """Fatal, expected fail-closed condition (bad release gate, corpus, etc.)."""


# ---------------------------------------------------------------------------
# Small, dependency-free helpers
# ---------------------------------------------------------------------------
def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_write_json(path, obj) -> None:
    """Write JSON atomically: temp file in the same directory, flush, fsync, replace."""
    path = Path(path)
    directory = path.parent
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _pkg_version(module_name: str, dist_name: Optional[str] = None) -> str:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        return str(importlib_metadata.version(dist_name or module_name))
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, str]:
    """Installed versions of every calculation-relevant library.

    The five saliency features are computed by ``saliency_features.py`` using
    cv2, NumPy and SciPy; TensorFlow/Keras/h5py versions (the actual model
    runtime) are resolved separately via ``resolve_runtime_versions`` -- see
    that function's docstring for why they are deliberately NOT obtained here.
    """
    return {
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "opencv_version": _pkg_version("cv2", "opencv-python"),
        "scipy_version": _pkg_version("scipy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def resolve_runtime_versions() -> Dict[str, str]:
    """Resolve the installed TensorFlow/Keras/h5py runtime versions.

    This is the runtime that actually executes UMSI (``predict_saliency``),
    so its identity must be bound into the run fingerprint just like the
    predictor/extractor module hashes -- but without ever importing
    TensorFlow, Keras or UMSI during preflight (``--preflight-only`` must
    leave them absent from ``sys.modules``). Versions are therefore obtained
    exclusively through ``importlib.metadata``, which reads installed
    distribution metadata without executing any package code.

    Raises ``SaliencyNormsError`` if any of the three required runtime
    identities cannot be resolved -- this must block before ANY output is
    created, exactly like every other preflight check.
    """
    tf_dist_name: Optional[str] = None
    tf_version: Optional[str] = None
    for candidate in _TENSORFLOW_DISTRIBUTION_CANDIDATES:
        try:
            tf_version = importlib_metadata.version(candidate)
            tf_dist_name = candidate
            break
        except importlib_metadata.PackageNotFoundError:
            continue
    if tf_dist_name is None or not tf_version:
        raise SaliencyNormsError(
            "Cannot resolve an installed TensorFlow distribution via "
            "importlib.metadata (tried: "
            f"{list(_TENSORFLOW_DISTRIBUTION_CANDIDATES)}). The production "
            "model runtime identity is required and unresolvable; refusing "
            "to proceed before any output is created.")

    try:
        keras_version = importlib_metadata.version("keras")
    except importlib_metadata.PackageNotFoundError as exc:
        raise SaliencyNormsError(
            "Cannot resolve the installed Keras distribution version via "
            "importlib.metadata. The production model runtime identity is "
            "required and unresolvable; refusing to proceed before any "
            "output is created.") from exc

    try:
        h5py_version = importlib_metadata.version("h5py")
    except importlib_metadata.PackageNotFoundError as exc:
        raise SaliencyNormsError(
            "Cannot resolve the installed h5py distribution version via "
            "importlib.metadata. The production weights-loading runtime "
            "identity is required and unresolvable; refusing to proceed "
            "before any output is created.") from exc

    if not tf_version.strip() or not keras_version.strip() or not h5py_version.strip():
        raise SaliencyNormsError(
            "Resolved an empty TensorFlow/Keras/h5py version string; "
            "refusing to proceed before any output is created.")

    return {
        "tensorflow_distribution_name": tf_dist_name,
        "tensorflow_version": tf_version,
        "keras_version": keras_version,
        "h5py_version": h5py_version,
    }


def _git_commit_sha(project_root: Path) -> str:
    """Best-effort current commit SHA, read directly from ``.git`` (no subprocess)."""
    try:
        git_dir = project_root / ".git"
        head_path = git_dir / "HEAD"
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
            packed = git_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.strip().endswith(" " + ref):
                        return line.split()[0]
            return "unknown"
        return head
    except Exception:
        return "unknown"


def _refuse_protected_output(path, label: str) -> None:
    """Abort if ``path`` targets, or is nested under, a protected location."""
    rp = Path(path).resolve()
    if rp in _PROTECTED_FILES or rp.name in _PROTECTED_BASENAMES:
        raise SaliencyNormsError(
            f"Refusing to write {label} to protected path '{path}'. This "
            f"generator must never overwrite feature_norms.json, "
            f"feature_norms_visual.json, sensitivity_lookup.json, P8/P9 "
            f"evidence, UMSI weights or production source files.")
    for protected_dir in _PROTECTED_DIRS:
        if rp == protected_dir or protected_dir in rp.parents:
            raise SaliencyNormsError(
                f"Refusing to write {label} to '{path}': it is inside the "
                f"protected path '{protected_dir}'.")


# ---------------------------------------------------------------------------
# P9 release-gate preflight
# ---------------------------------------------------------------------------
def run_p9_release_preflight(*, p9_report_path: Path, p9_report_sha: str,
                             production_source_path: Path, weights_path: Path,
                             predictor_module_path: Path,
                             extractor_module_path: Path) -> Dict[str, Any]:
    """Verify the frozen P9 release gate BEFORE any dataset scan or model import.

    Raises ``SaliencyNormsError`` fail-closed on any missing field or mismatch.
    Returns a dict of verified identities for inclusion in the run fingerprint
    and manifest.
    """
    # 0. Required files must exist before anything else.
    for label, p in (("P9 report", p9_report_path),
                     ("production UMSI source", production_source_path),
                     ("UMSI weights", weights_path),
                     ("predictor module", predictor_module_path),
                     ("feature-extractor module", extractor_module_path)):
        if not Path(p).is_file():
            raise SaliencyNormsError(f"Required {label} file not found: {p}")

    # 1. THREE-WAY hash pinning (fail-closed hash-of-truth check). The actual
    #    report file SHA, the caller-supplied --p9-report-sha, and the
    #    immutable FROZEN_P9_REPORT_SHA256 production constant must ALL agree.
    #    A caller-supplied hash of some other (even internally self-consistent
    #    and PASS-looking) report is never sufficient by itself.
    if not isinstance(p9_report_sha, str) or not _SHA256_RE.match(p9_report_sha):
        raise SaliencyNormsError(
            f"--p9-report-sha is not 64 lowercase hex chars: {p9_report_sha!r}")
    report_bytes = Path(p9_report_path).read_bytes()
    actual_report_sha = _sha256_bytes(report_bytes)
    if actual_report_sha != p9_report_sha:
        raise SaliencyNormsError(
            f"P9 report SHA mismatch: expected {p9_report_sha!r}, got "
            f"{actual_report_sha!r}. Refusing to proceed before model import.")
    if p9_report_sha != FROZEN_P9_REPORT_SHA256:
        raise SaliencyNormsError(
            f"P9 report SHA {p9_report_sha!r} does not equal the immutable "
            f"FROZEN_P9_REPORT_SHA256 production constant "
            f"{FROZEN_P9_REPORT_SHA256!r}. A self-consistent hash for a "
            f"different report is never sufficient. Refusing to proceed "
            f"before model import.")

    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except Exception as exc:
        raise SaliencyNormsError(f"P9 report is not valid JSON: {exc}") from exc

    # 2. Required top-level sections.
    for section in ("conclusion", "verdict", "assessment_metadata", "identities",
                    "acceptance_gate_results", "diagnostic_gate_results"):
        if section not in report:
            raise SaliencyNormsError(
                f"P9 report missing required section: {section!r}")

    verdict = report["verdict"]
    meta = report["assessment_metadata"]
    identities = report["identities"]
    acceptance_gates = report["acceptance_gate_results"]
    diagnostic_gates = report["diagnostic_gate_results"]

    if not isinstance(verdict, dict) or not isinstance(meta, dict) \
            or not isinstance(identities, dict):
        raise SaliencyNormsError(
            "P9 report verdict/assessment_metadata/identities must be objects.")
    if not isinstance(acceptance_gates, list) or not isinstance(diagnostic_gates, list):
        raise SaliencyNormsError(
            "P9 report acceptance_gate_results/diagnostic_gate_results must "
            "be lists.")

    required_verdict_fields = (
        "conclusion", "saliency_norms_release_status",
        "independent_inference_performed", "negative_control_status",
        "source_sha_match", "weights_sha_match", "p8_historical_result_preserved",
    )
    missing = [f for f in required_verdict_fields if f not in verdict]
    if missing:
        raise SaliencyNormsError(f"P9 report verdict missing fields: {missing}")
    required_meta_fields = ("saliency_norms_release_status",
                            "independent_inference_performed")
    missing = [f for f in required_meta_fields if f not in meta]
    if missing:
        raise SaliencyNormsError(
            f"P9 report assessment_metadata missing fields: {missing}")
    required_identity_fields = ("p8_source_sha256", "weights_sha256")
    missing = [f for f in required_identity_fields if f not in identities]
    if missing:
        raise SaliencyNormsError(f"P9 report identities missing fields: {missing}")

    # 3. conclusion must be exactly the frozen PASS conclusion, at BOTH the
    #    top level and inside verdict.
    conclusion = report["conclusion"]
    if conclusion != P9_REQUIRED_CONCLUSION:
        raise SaliencyNormsError(
            f"P9 conclusion is {conclusion!r}, not {P9_REQUIRED_CONCLUSION!r}. "
            f"Refusing to proceed before model import.")
    if verdict.get("conclusion") != P9_REQUIRED_CONCLUSION:
        raise SaliencyNormsError(
            f"P9 verdict.conclusion is {verdict.get('conclusion')!r}, not "
            f"{P9_REQUIRED_CONCLUSION!r}. Refusing to proceed before model "
            f"import.")

    # 3b. Required acceptance/diagnostic gate semantics: gate_1, gate_2 and
    #     diag_1 must each be present exactly once, passed, with no reason,
    #     and diag_1's negative control must not be alarmed.
    def _find_gate(gates: List[Any], name: str, section_label: str) -> Dict[str, Any]:
        matches = [g for g in gates
                  if isinstance(g, dict) and g.get("name") == name]
        if len(matches) != 1:
            raise SaliencyNormsError(
                f"P9 {section_label} must contain exactly one {name!r} entry; "
                f"found {len(matches)}. Refusing to proceed before model import.")
        return matches[0]

    gate_1 = _find_gate(acceptance_gates, "gate_1", "acceptance_gate_results")
    gate_2 = _find_gate(acceptance_gates, "gate_2", "acceptance_gate_results")
    diag_1 = _find_gate(diagnostic_gates, "diag_1", "diagnostic_gate_results")

    if gate_1.get("passed") is not True or gate_1.get("reason") is not None:
        raise SaliencyNormsError(
            f"P9 gate_1 must have passed=True and reason=None; got "
            f"passed={gate_1.get('passed')!r}, reason={gate_1.get('reason')!r}.")
    if gate_2.get("passed") is not True or gate_2.get("reason") is not None:
        raise SaliencyNormsError(
            f"P9 gate_2 must have passed=True and reason=None; got "
            f"passed={gate_2.get('passed')!r}, reason={gate_2.get('reason')!r}.")
    if diag_1.get("passed") is not True or diag_1.get("reason") is not None:
        raise SaliencyNormsError(
            f"P9 diag_1 must have passed=True and reason=None; got "
            f"passed={diag_1.get('passed')!r}, reason={diag_1.get('reason')!r}.")
    diag_1_observed = diag_1.get("observed")
    if not isinstance(diag_1_observed, dict) or diag_1_observed.get("alarm") is not False:
        raise SaliencyNormsError(
            "P9 diag_1.observed.alarm must be False (negative control must "
            "be discriminative, not alarmed). Refusing to proceed before "
            "model import.")

    # 3c. Negative-control status must be DISCRIMINATIVE.
    negative_control_status = verdict.get("negative_control_status")
    if negative_control_status != P9_REQUIRED_NEGATIVE_CONTROL_STATUS:
        raise SaliencyNormsError(
            f"P9 verdict.negative_control_status is "
            f"{negative_control_status!r}, not "
            f"{P9_REQUIRED_NEGATIVE_CONTROL_STATUS!r}. Refusing to proceed "
            f"before model import.")

    # 4. saliency_norms_release_status must be UNBLOCKED in BOTH recorded copies.
    for label, obj in (("verdict", verdict), ("assessment_metadata", meta)):
        status = obj.get("saliency_norms_release_status")
        if status != P9_REQUIRED_NORMS_STATUS:
            raise SaliencyNormsError(
                f"P9 {label}.saliency_norms_release_status is {status!r}, not "
                f"{P9_REQUIRED_NORMS_STATUS!r}. Refusing to proceed before "
                f"model import.")

    # 5. Integrity fields must all be successful (not merely present).
    if verdict.get("independent_inference_performed") is not False:
        raise SaliencyNormsError(
            "P9 verdict.independent_inference_performed must be false.")
    if meta.get("independent_inference_performed") is not False:
        raise SaliencyNormsError(
            "P9 assessment_metadata.independent_inference_performed must be false.")
    if verdict.get("source_sha_match") is not True:
        raise SaliencyNormsError("P9 verdict.source_sha_match must be true.")
    if verdict.get("weights_sha_match") is not True:
        raise SaliencyNormsError("P9 verdict.weights_sha_match must be true.")
    if verdict.get("p8_historical_result_preserved") is not True:
        raise SaliencyNormsError(
            "P9 verdict.p8_historical_result_preserved must be true.")

    # 6. Current production source SHA must match the source identity P9 recorded.
    p9_source_sha = identities["p8_source_sha256"]
    if not isinstance(p9_source_sha, str) or not _SHA256_RE.match(p9_source_sha):
        raise SaliencyNormsError(
            f"P9 identities.p8_source_sha256 is not 64 lowercase hex chars: "
            f"{p9_source_sha!r}")
    actual_source_sha = _sha256_file(production_source_path)
    if actual_source_sha != p9_source_sha:
        raise SaliencyNormsError(
            f"Current production UMSI source SHA {actual_source_sha!r} does "
            f"not match the identity P9 recorded ({p9_source_sha!r}). "
            f"Refusing to proceed before model import.")

    # 7. Current weights SHA (and byte size, recorded for our own provenance)
    #    must match the weights identity P9 recorded.
    p9_weights_sha = identities["weights_sha256"]
    if not isinstance(p9_weights_sha, str) or not _SHA256_RE.match(p9_weights_sha):
        raise SaliencyNormsError(
            f"P9 identities.weights_sha256 is not 64 lowercase hex chars: "
            f"{p9_weights_sha!r}")
    actual_weights_sha = _sha256_file(weights_path)
    actual_weights_size = int(Path(weights_path).stat().st_size)
    if actual_weights_sha != p9_weights_sha:
        raise SaliencyNormsError(
            f"Current UMSI weights SHA {actual_weights_sha!r} does not match "
            f"the identity P9 recorded ({p9_weights_sha!r}). Refusing to "
            f"proceed before model import.")

    predictor_module_sha = _sha256_file(predictor_module_path)
    extractor_module_sha = _sha256_file(extractor_module_path)

    return {
        "p9_report_sha256": actual_report_sha,
        "p9_conclusion": conclusion,
        "p9_saliency_norms_release_status": verdict["saliency_norms_release_status"],
        "production_source_sha256": actual_source_sha,
        "predictor_module_sha256": predictor_module_sha,
        "extractor_module_sha256": extractor_module_sha,
        "weights_sha256": actual_weights_sha,
        "weights_byte_size": actual_weights_size,
    }


# ---------------------------------------------------------------------------
# Corpus discovery + strict contract enforcement
# ---------------------------------------------------------------------------
# Image extensions the production pipeline treats as decodable corpus images.
# Used ONLY to decide which files under images/ must appear somewhere in
# image_types.csv (authorized or poster); it does not affect decoding itself
# (cv2.imread is always the actual decode authority).
_SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


def _validate_safe_relative_name(name: str, images_dir: Path) -> None:
    """Reject absolute paths, path traversal, or any path/symlink escaping
    ``images_dir`` -- BEFORE the name is trusted for any filesystem or CSV
    bookkeeping purpose.
    """
    if os.path.isabs(name):
        raise SaliencyNormsError(
            f"image_types.csv row has an absolute image name (not allowed): "
            f"{name!r}")
    if ".." in Path(name).parts:
        raise SaliencyNormsError(
            f"image_types.csv row contains path traversal ('..'): {name!r}")
    candidate = (images_dir / name)
    resolved_images_dir = images_dir.resolve()
    try:
        resolved_candidate = candidate.resolve()
    except OSError as exc:
        raise SaliencyNormsError(
            f"Cannot resolve image path {name!r}: {exc}") from exc
    if resolved_candidate != resolved_images_dir \
            and resolved_images_dir not in resolved_candidate.parents:
        raise SaliencyNormsError(
            f"image_types.csv row {name!r} resolves outside the images/ "
            f"directory ({resolved_candidate}); refusing (possible symlink "
            f"or traversal escape).")


def discover_authorized_corpus(types_csv: Path, images_dir: Path
                               ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Return the ordered ``(relative_path, category)`` population and counts.

    Ordering is deterministic: category order is exactly
    ``AUTHORIZED_CATEGORIES`` (desktop, mobile, web) and, within each
    category, filenames are lexicographically sorted -- i.e. desktop occupies
    global indices 1-495, mobile 496-990, web 991-1485 for the complete
    corpus.

    Rejects (raises ``SaliencyNormsError``):
      * a types-csv or images-dir that does not exist;
      * duplicate ``(name, category)`` rows in the source manifest;
      * a filename mapped to more than one category;
      * any category other than the three authorized ones or ``poster``
        (poster is excluded; anything else is a hard corpus violation, never
        silently dropped);
      * a missing (zero-count) authorized category;
      * a listed image file that does not exist or cannot be decoded.
    """
    if not types_csv.exists():
        raise SaliencyNormsError(f"image_types.csv not found: {types_csv}")
    if not images_dir.is_dir():
        raise SaliencyNormsError(f"images directory not found: {images_dir}")

    seen_pairs = set()
    name_to_cats: Dict[str, set] = collections.defaultdict(set)
    by_category: Dict[str, List[str]] = collections.defaultdict(list)
    unknown_categories: set = set()
    all_listed_names: set = set()

    with open(types_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        required_header = {"Image Name", "Category"}
        header = set(reader.fieldnames or [])
        if not required_header.issubset(header):
            raise SaliencyNormsError(
                f"image_types.csv header must contain {sorted(required_header)}; "
                f"got {sorted(header)}.")
        for row_num, row in enumerate(reader, start=2):  # header is row 1
            raw_name = row.get("Image Name")
            raw_cat = row.get("Category")
            # A wholly blank row (csv yields all-None or all-empty values).
            if raw_name is None and raw_cat is None:
                raise SaliencyNormsError(
                    f"image_types.csv row {row_num} is blank/malformed.")
            name = (raw_name or "").strip()
            cat = (raw_cat or "").strip().lower()
            if not name:
                raise SaliencyNormsError(
                    f"image_types.csv row {row_num} has an empty image name.")
            if not cat:
                raise SaliencyNormsError(
                    f"image_types.csv row {row_num} ({name!r}) has an empty "
                    f"category.")
            _validate_safe_relative_name(name, images_dir)
            pair = (name, cat)
            if pair in seen_pairs:
                raise SaliencyNormsError(
                    f"Duplicate row in image_types.csv: {name!r} / {cat!r}")
            seen_pairs.add(pair)
            all_listed_names.add(name)
            name_to_cats[name].add(cat)
            if cat in _AUTHORIZED_SET:
                by_category[cat].append(name)
            elif cat in EXCLUDED_CATEGORIES:
                pass  # poster: excluded by design, not an error
            else:
                unknown_categories.add(cat)

    if unknown_categories:
        raise SaliencyNormsError(
            f"Corpus contains unknown/unauthorized categor{'y' if len(unknown_categories)==1 else 'ies'} "
            f"{sorted(unknown_categories)!r}. Only {list(AUTHORIZED_CATEGORIES)} "
            f"(plus excluded {list(EXCLUDED_CATEGORIES)}) are permitted.")

    conflicting = [n for n, cs in name_to_cats.items() if len(cs) > 1]
    if conflicting:
        raise SaliencyNormsError(
            f"{len(conflicting)} filename(s) map to multiple categories, "
            f"e.g. {conflicting[:3]}")

    missing_categories = [c for c in AUTHORIZED_CATEGORIES if not by_category.get(c)]
    if missing_categories:
        raise SaliencyNormsError(
            f"Corpus is missing authorized categor{'y' if len(missing_categories)==1 else 'ies'}: "
            f"{missing_categories}")

    # Disk <-> CSV completeness: every supported image file on disk must be
    # listed in the CSV (authorized OR poster), and every CSV-listed image
    # must exist on disk (checked again per-file below).
    on_disk_supported = {
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
    }
    unlisted_on_disk = sorted(on_disk_supported - all_listed_names)
    if unlisted_on_disk:
        raise SaliencyNormsError(
            f"{len(unlisted_on_disk)} supported image file(s) exist under "
            f"images/ but are not listed in image_types.csv, e.g. "
            f"{unlisted_on_disk[:3]}.")

    population: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    for cat in AUTHORIZED_CATEGORIES:
        names = sorted(set(by_category[cat]))
        if len(names) != len(by_category[cat]):
            raise SaliencyNormsError(f"Duplicate filename within category {cat!r}")
        for name in names:
            if not (images_dir / name).exists():
                raise SaliencyNormsError(
                    f"Authorized image listed but not found on disk: "
                    f"{cat}/{name}")
            population.append((name, cat))
        counts[cat] = len(names)

    return population, counts


def enforce_corpus_count_contract(counts: Dict[str, int], total: int) -> None:
    """Reject any category-count mismatch BEFORE any decode/model work."""
    actual = {k: int(counts.get(k, 0)) for k in AUTHORIZED_CATEGORIES}
    if actual != EXPECTED_AUTHORIZED_COUNTS or total != FULL_RUN_TOTAL:
        raise SaliencyNormsError(
            f"Corpus contract violation: expected exactly "
            f"{EXPECTED_AUTHORIZED_COUNTS} ({FULL_RUN_TOTAL} total), got "
            f"{actual} ({total} total). Aborting before any feature "
            f"computation or model load.")


def _decode_native_dims(path: Path) -> Tuple[int, int]:
    """Return ``(width, height)`` via the same cv2-based decode path production uses."""
    img = cv2.imread(str(path))
    if img is None:
        raise SaliencyNormsError(f"Cannot decode image: {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def build_ordered_corpus_manifest(population: List[Tuple[str, str]],
                                  images_dir: Path
                                  ) -> Tuple[List[Dict[str, Any]], str]:
    """Build the deterministic, ordered, content- and dimension-bound corpus manifest.

    Rejects (raises ``SaliencyNormsError``):
      * a duplicate image path (should be impossible given discovery, checked
        defensively anyway);
      * a duplicate image SHA-256 (two different filenames with identical
        content);
      * an unreadable/undecodable image.
    """
    records: List[Dict[str, Any]] = []
    seen_paths: set = set()
    seen_shas: set = set()
    for idx, (name, cat) in enumerate(population, start=1):
        if name in seen_paths:
            raise SaliencyNormsError(f"Duplicate image path in corpus: {name!r}")
        seen_paths.add(name)
        path = images_dir / name
        sha = _sha256_file(path)
        if sha in seen_shas:
            raise SaliencyNormsError(
                f"Duplicate image content (SHA-256 collision) for {name!r}")
        seen_shas.add(sha)
        width, height = _decode_native_dims(path)
        records.append({
            "global_index": idx,
            "category": cat,
            "relative_path": name,
            "image_sha256": sha,
            "native_width": width,
            "native_height": height,
        })
    manifest_sha = hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()
    return records, manifest_sha


# ---------------------------------------------------------------------------
# Run fingerprint
# ---------------------------------------------------------------------------
def build_run_fingerprint(*, category_counts: Dict[str, int],
                          corpus_manifest_sha256: str,
                          identity: Dict[str, Any],
                          generator_sha256: str,
                          env: Dict[str, str]) -> Tuple[Dict[str, Any], str]:
    """Build the complete calculation-provenance object and its SHA-256.

    Binds: schema, exact feature order, resolution policy, authorized /
    excluded categories, complete authorized counts, the corpus manifest hash,
    production-source / predictor-module / feature-extractor-module /
    weights (hash + byte size) / generator / P9-report hashes, and the
    environment versions. Runtime is deliberately excluded so a legitimate
    resume of the same corpus/code/environment matches exactly.
    """
    fp = {
        "schema_version": SCHEMA_VERSION,
        "feature_order": list(SALIENCY_FEATURE_KEYS),
        "resolution_policy": RESOLUTION_POLICY,
        "authorized_categories": list(AUTHORIZED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "authorized_category_counts": {k: int(category_counts.get(k, 0))
                                       for k in AUTHORIZED_CATEGORIES},
        "authorized_total": int(sum(category_counts.get(k, 0)
                                    for k in AUTHORIZED_CATEGORIES)),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "production_source_sha256": identity["production_source_sha256"],
        "predictor_module_sha256": identity["predictor_module_sha256"],
        "extractor_module_sha256": identity["extractor_module_sha256"],
        "weights_sha256": identity["weights_sha256"],
        "weights_byte_size": identity["weights_byte_size"],
        "generator_sha256": generator_sha256,
        "p9_report_sha256": identity["p9_report_sha256"],
        "environment": dict(env),
    }
    fp_sha = hashlib.sha256(_canonical_json(fp).encode("utf-8")).hexdigest()
    return fp, fp_sha


# ---------------------------------------------------------------------------
# Aggregation (identical statistical schema/formulas to the existing norms)
# ---------------------------------------------------------------------------
def aggregate(values_by_feature: Dict[str, List[float]]) -> Dict[str, dict]:
    norms = {}
    for feature in SALIENCY_FEATURE_KEYS:
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


# ---------------------------------------------------------------------------
# Checkpoint (CSV rows + JSON provenance sidecar)
# ---------------------------------------------------------------------------
def sidecar_path_for(csv_path: Path) -> Path:
    return Path(str(csv_path) + ".provenance.json")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _new_segment(invocation_index: int, env: Dict[str, str]) -> dict:
    return {
        "invocation_index": invocation_index,
        "started_at": _now_iso(),
        "ended_at": None,
        "elapsed_seconds": None,
        "n_new_rows": 0,
        "environment": env,
        "incomplete": True,
        "status": STATUS_RUNNING,
    }


def _finalize_segment(seg: dict, elapsed: float, n_new: int) -> dict:
    seg["ended_at"] = _now_iso()
    seg["elapsed_seconds"] = round(float(elapsed), 3)
    seg["n_new_rows"] = int(n_new)
    seg["incomplete"] = False
    seg["status"] = STATUS_COMPLETED
    return seg


def write_checkpoint_sidecar(sidecar_path: Path, run_fingerprint: dict,
                             fingerprint_sha: str, csv_basename: str,
                             segments: List[dict], status: str,
                             recovered: bool,
                             last_failure: Optional[dict] = None) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "recovered_from_interruption": bool(recovered),
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint_sha,
        "csv_basename": csv_basename,
        "segments": segments,
        "invocation_count": len(segments),
        "last_failure": last_failure,
    }
    _atomic_write_json(sidecar_path, payload)


def load_checkpoint_for_resume(csv_path: Path, sidecar_path: Path,
                               run_fingerprint: dict, fingerprint_sha: str,
                               manifest_by_path: Dict[str, Dict[str, Any]],
                               images_dir: Path
                               ) -> Tuple[List[dict], set, List[dict]]:
    """Validate and load a resumable checkpoint (CSV + provenance sidecar).

    Resume is allowed ONLY when:
      * the stored run-fingerprint OBJECT equals the freshly calculated one
        (deep equality, not just the hash);
      * ``sha256(canonical(stored run_fingerprint))`` equals the stored
        ``run_fingerprint_sha256`` AND the freshly calculated fingerprint SHA
        (i.e. the sidecar's own hash is internally consistent, not merely
        copy-pasted);
      * the sidecar's ``csv_basename`` equals the actual checkpoint CSV's
        basename;
      * the sidecar's ``schema_version`` equals the current schema version;
      * the sidecar's ``status`` is ``running`` or ``failed`` (never
        ``completed``);
      * the sidecar's ``segments`` is a list of structurally valid dicts;
      * every stored CSV row remains consistent with the current authorized
        corpus manifest and on-disk image content, using the EXACT expected
        header (no missing or additional columns) and rejecting every
        blank/malformed row rather than silently skipping it.
    """
    if not sidecar_path.exists():
        raise SaliencyNormsError(
            f"Refusing to resume: checkpoint provenance sidecar "
            f"'{sidecar_path.name}' is missing.")
    try:
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    except Exception as exc:
        raise SaliencyNormsError(
            f"Refusing to resume: cannot read checkpoint provenance sidecar: {exc}")
    if not isinstance(sidecar, dict):
        raise SaliencyNormsError(
            "Refusing to resume: checkpoint provenance sidecar is not a JSON object.")

    # -- Structural fail-closed checks on the sidecar itself. --
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise SaliencyNormsError(
            f"Refusing to resume: sidecar schema_version "
            f"{sidecar.get('schema_version')!r} does not match the current "
            f"schema version {SCHEMA_VERSION!r}.")
    status = sidecar.get("status")
    if status not in (STATUS_RUNNING, STATUS_FAILED):
        raise SaliencyNormsError(
            f"Refusing to resume: sidecar status is {status!r}; only "
            f"{STATUS_RUNNING!r} or {STATUS_FAILED!r} checkpoints can be "
            f"resumed (a {STATUS_COMPLETED!r} run is finished, not resumable).")
    if sidecar.get("csv_basename") != csv_path.name:
        raise SaliencyNormsError(
            f"Refusing to resume: sidecar csv_basename "
            f"{sidecar.get('csv_basename')!r} does not match the actual "
            f"checkpoint CSV basename {csv_path.name!r}.")
    segments = sidecar.get("segments")
    if not isinstance(segments, list):
        raise SaliencyNormsError(
            "Refusing to resume: sidecar 'segments' is not a list.")
    required_segment_fields = ("invocation_index", "started_at", "status")
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise SaliencyNormsError(
                f"Refusing to resume: sidecar segments[{i}] is not an object.")
        missing_seg_fields = [f for f in required_segment_fields if f not in seg]
        if missing_seg_fields:
            raise SaliencyNormsError(
                f"Refusing to resume: sidecar segments[{i}] is missing "
                f"required fields {missing_seg_fields}.")

    stored_fingerprint_obj = sidecar.get("run_fingerprint")
    if not isinstance(stored_fingerprint_obj, dict):
        raise SaliencyNormsError(
            "Refusing to resume: sidecar 'run_fingerprint' is not an object.")
    if stored_fingerprint_obj != run_fingerprint:
        raise SaliencyNormsError(
            "Refusing to resume: stored run_fingerprint object does not "
            "deep-equal the freshly calculated run fingerprint. The dataset, "
            "generator, model, weights, extractor, environment/runtime or "
            "feature order has changed.")

    recomputed_stored_sha = hashlib.sha256(
        _canonical_json(stored_fingerprint_obj).encode("utf-8")).hexdigest()
    stored_sha = sidecar.get("run_fingerprint_sha256")
    if stored_sha != fingerprint_sha:
        raise SaliencyNormsError(
            f"Refusing to resume: stored run fingerprint hash {stored_sha!r} "
            f"does not equal the freshly calculated fingerprint "
            f"{fingerprint_sha!r}.")
    if recomputed_stored_sha != stored_sha:
        raise SaliencyNormsError(
            f"Refusing to resume: sidecar run_fingerprint_sha256 "
            f"{stored_sha!r} does not equal sha256(canonical(sidecar's own "
            f"run_fingerprint)) ({recomputed_stored_sha!r}). The sidecar is "
            f"internally inconsistent (tampered or corrupted).")

    expected_feature_order = "|".join(SALIENCY_FEATURE_KEYS)
    rows: List[dict] = []
    done: set = set()
    seen_paths: set = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = ["global_index", "category", "relative_path",
                    ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                    *SALIENCY_FEATURE_KEYS, ROW_FEATURE_ORDER_COLUMN,
                    ROW_FINGERPRINT_COLUMN]
        if list(header) != required:
            missing_cols = [c for c in required if c not in header]
            extra_cols = [c for c in header if c not in required]
            raise SaliencyNormsError(
                f"Refusing to resume: checkpoint CSV header does not exactly "
                f"match the expected columns. missing={missing_cols} "
                f"extra={extra_cols}.")
        for row_num, raw in enumerate(reader, start=2):  # header is row 1
            if raw is None or all(
                    (v is None or str(v).strip() == "") for v in raw.values()):
                raise SaliencyNormsError(
                    f"Refusing to resume: checkpoint CSV row {row_num} is "
                    f"blank/malformed.")
            rel_path = (raw.get("relative_path") or "").strip()
            if not rel_path:
                raise SaliencyNormsError(
                    f"Refusing to resume: checkpoint CSV row {row_num} has "
                    f"an empty relative_path; refusing to silently skip a "
                    f"malformed row.")
            if (raw.get(ROW_FINGERPRINT_COLUMN) or "").strip() != fingerprint_sha:
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} is bound to a "
                    f"different run fingerprint.")
            if (raw.get(ROW_FEATURE_ORDER_COLUMN) or "").strip() != expected_feature_order:
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} has a different "
                    f"feature order than the current generator.")
            manifest_entry = manifest_by_path.get(rel_path)
            if manifest_entry is None:
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} is not present in "
                    f"the current authorized corpus manifest.")
            if rel_path in seen_paths or rel_path in done:
                raise SaliencyNormsError(
                    f"Refusing to resume: duplicate row for {rel_path!r}.")
            seen_paths.add(rel_path)
            cat = (raw.get("category") or "").strip().lower()
            if cat != manifest_entry["category"]:
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} category {cat!r} "
                    f"disagrees with the current manifest "
                    f"({manifest_entry['category']!r}).")
            try:
                global_index = int(raw.get("global_index"))
            except (TypeError, ValueError):
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} has a non-integer "
                    f"global_index.")
            if global_index != manifest_entry["global_index"]:
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} global_index "
                    f"{global_index} disagrees with the manifest "
                    f"({manifest_entry['global_index']}).")

            img_path = images_dir / rel_path
            if not img_path.exists():
                raise SaliencyNormsError(
                    f"Refusing to resume: image for row {rel_path!r} no "
                    f"longer exists on disk.")
            current_sha = _sha256_file(img_path)
            row_sha = (raw.get(ROW_IMAGE_SHA_COLUMN) or "").strip()
            if row_sha != current_sha or row_sha != manifest_entry["image_sha256"]:
                raise SaliencyNormsError(
                    f"Refusing to resume: image content or hash for row "
                    f"{rel_path!r} has changed since it was computed or "
                    f"differs from the corpus manifest.")

            try:
                nw = int(raw.get("native_width"))
                nh = int(raw.get("native_height"))
            except (TypeError, ValueError):
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} has non-integer "
                    f"native dimensions.")
            if (nw, nh) != (manifest_entry["native_width"],
                           manifest_entry["native_height"]):
                raise SaliencyNormsError(
                    f"Refusing to resume: row {rel_path!r} native dimensions "
                    f"({nw}x{nh}) differ from the corpus manifest "
                    f"({manifest_entry['native_width']}x"
                    f"{manifest_entry['native_height']}).")

            row = {"global_index": global_index, "category": cat,
                  "relative_path": rel_path, ROW_IMAGE_SHA_COLUMN: row_sha,
                  "native_width": nw, "native_height": nh}
            for k in SALIENCY_FEATURE_KEYS:
                v = raw.get(k, "")
                if v is None or str(v).strip() == "":
                    raise SaliencyNormsError(
                        f"Refusing to resume: row {rel_path!r} has an "
                        f"incomplete (missing) value for {k!r}.")
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    raise SaliencyNormsError(
                        f"Refusing to resume: row {rel_path!r} has a "
                        f"non-numeric value for {k!r}: {v!r}.")
                if not np.isfinite(fv):
                    raise SaliencyNormsError(
                        f"Refusing to resume: row {rel_path!r} has a "
                        f"non-finite value for {k!r}: {fv!r}.")
                row[k] = fv
            rows.append(row)
            done.add(rel_path)

    prior_segments = list(sidecar.get("segments", []))
    return rows, done, prior_segments


# ---------------------------------------------------------------------------
# Lazy predictor factory (the ONLY place TensorFlow/Keras/UMSI may be imported)
# ---------------------------------------------------------------------------
def _default_predictor_factory(weights_path):
    from saliency.umsi_model import UMSIPlus  # noqa: WPS433 (intentional lazy import)
    return UMSIPlus(str(weights_path))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None,
        predictor_factory: Optional[Callable[[str], Any]] = None,
        extractor_fn: Optional[Callable[[np.ndarray], Dict[str, float]]] = None
        ) -> int:
    try:
        return _run(argv, predictor_factory, extractor_fn)
    except SaliencyNormsError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


def _run(argv: Optional[List[str]],
        predictor_factory: Optional[Callable[[str], Any]],
        extractor_fn: Optional[Callable[[np.ndarray], Dict[str, float]]]) -> int:
    if extractor_fn is None:
        extractor_fn = extract_saliency_features
    if predictor_factory is None:
        predictor_factory = _default_predictor_factory

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p9-report", required=True,
                    help="path to the frozen gate_report_p9.json")
    ap.add_argument("--p9-report-sha", required=True,
                    help="expected SHA-256 of the P9 report file")
    ap.add_argument("--dataset-root", required=True,
                    help="UEyes_dataset root (contains images/ + image_types.csv)")
    ap.add_argument("--weights", required=True,
                    help="path to the UMSI++ weights checkpoint")
    ap.add_argument("--output-dir", required=True,
                    help="output directory for checkpoint + final artifacts")
    ap.add_argument("--resume", action="store_true",
                    help="explicitly resume from a fingerprint-matching checkpoint")
    ap.add_argument("--preflight-only", action="store_true",
                    help="validate everything but never import the model, "
                         "predict, or create the output directory/artifacts")
    args = ap.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    norms_path = output_dir / "saliency_feature_norms.json"
    manifest_path = output_dir / "saliency_feature_norms_manifest.json"
    csv_path = output_dir / "saliency_feature_rows.csv"
    sidecar_path = sidecar_path_for(csv_path)

    for label, p in (("output directory", output_dir),
                     ("norms JSON", norms_path),
                     ("manifest JSON", manifest_path),
                     ("per-image CSV", csv_path),
                     ("provenance sidecar", sidecar_path)):
        _refuse_protected_output(p, label)

    # ---- 1. P9 release-gate preflight -- BEFORE any dataset scan or model import.
    identity = run_p9_release_preflight(
        p9_report_path=Path(args.p9_report).resolve(),
        p9_report_sha=args.p9_report_sha,
        production_source_path=_PRODUCTION_SOURCE_PATH,
        weights_path=Path(args.weights).resolve(),
        predictor_module_path=_PREDICTOR_MODULE_PATH,
        extractor_module_path=_EXTRACTOR_MODULE_PATH,
    )

    # ---- 2. Corpus discovery + strict count contract (before any decode-heavy work).
    dataset_root = Path(args.dataset_root).resolve()
    images_dir = dataset_root / "images"
    types_csv = dataset_root / "image_types.csv"
    population, category_counts = discover_authorized_corpus(types_csv, images_dir)
    enforce_corpus_count_contract(category_counts, len(population))

    # ---- 3. Deterministic ordered corpus manifest (content hash + native dims).
    corpus_records, corpus_manifest_sha = build_ordered_corpus_manifest(
        population, images_dir)
    manifest_by_path = {r["relative_path"]: r for r in corpus_records}

    # ---- 4. Run fingerprint.
    env = environment_info()
    runtime_versions = resolve_runtime_versions()
    env.update(runtime_versions)
    generator_sha = _sha256_file(_THIS_FILE)
    git_commit = _git_commit_sha(_PROJECT_ROOT)
    run_fingerprint, fingerprint_sha = build_run_fingerprint(
        category_counts=category_counts,
        corpus_manifest_sha256=corpus_manifest_sha,
        identity=identity,
        generator_sha256=generator_sha,
        env=env,
    )

    print(f"Authorized corpus: {dict(category_counts)} total={len(population)}")
    print(f"Corpus manifest sha256: {corpus_manifest_sha}")
    print(f"Production source sha256: {identity['production_source_sha256']}")
    print(f"Weights sha256: {identity['weights_sha256']} "
          f"size_bytes={identity['weights_byte_size']}")
    print(f"Run fingerprint sha256: {fingerprint_sha}")

    if args.preflight_only:
        print("Preflight-only mode: stopping before model import, prediction, "
              "output-directory creation or artifact write.")
        return 0

    # ---- 5. Fresh vs. resume directory handling.
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    sidecar_exists = sidecar_path.exists()
    dir_has_content = output_dir.exists() and any(output_dir.iterdir())

    done_rows: List[dict] = []
    done_paths: set = set()
    prior_segments: List[dict] = []

    if args.resume:
        if not csv_exists or not sidecar_exists:
            raise SaliencyNormsError(
                "Refusing to resume: a consistent checkpoint set (CSV + "
                "provenance sidecar) was not found.")
        if norms_path.exists() or manifest_path.exists():
            raise SaliencyNormsError(
                "Refusing to resume: a final artifact (saliency_feature_norms.json "
                "or saliency_feature_norms_manifest.json) already exists in "
                "the output directory. A completed run must not be resumed or "
                "overwritten.")
        allowed_names = {csv_path.name, sidecar_path.name}
        unexpected = [p.name for p in output_dir.iterdir()
                     if p.name not in allowed_names]
        if unexpected:
            raise SaliencyNormsError(
                f"Refusing to resume: output directory contains unexpected "
                f"entries besides the partial checkpoint CSV and its "
                f"provenance sidecar: {sorted(unexpected)}.")
        done_rows, done_paths, prior_segments = load_checkpoint_for_resume(
            csv_path, sidecar_path, run_fingerprint, fingerprint_sha,
            manifest_by_path, images_dir)
        print(f"Resuming: {len(done_paths)} rows validated against the run "
              f"fingerprint.")
    else:
        if dir_has_content:
            raise SaliencyNormsError(
                f"Output directory '{output_dir}' is not empty. Refusing to "
                f"overwrite implicitly. Pass --resume for a matching "
                f"checkpoint, or choose an empty output directory.")

    os.makedirs(output_dir, exist_ok=True)

    recovered = any(s.get("incomplete") for s in prior_segments)
    current_segment = _new_segment(len(prior_segments), env)
    all_segments = prior_segments + [current_segment]
    csv_basename = csv_path.name

    write_checkpoint_sidecar(sidecar_path, run_fingerprint, fingerprint_sha,
                             csv_basename, all_segments, STATUS_RUNNING, recovered)

    # ---- 6. Lazy model load -- only now, after every other check has passed.
    predictor = predictor_factory(str(Path(args.weights).resolve()))

    # ---- 7. Sequential, deterministic per-image processing loop.
    feature_order = list(SALIENCY_FEATURE_KEYS)
    feature_order_str = "|".join(feature_order)
    fieldnames = ["global_index", "category", "relative_path",
                 ROW_IMAGE_SHA_COLUMN, "native_width", "native_height",
                 *feature_order, ROW_FEATURE_ORDER_COLUMN, ROW_FINGERPRINT_COLUMN]

    values_by_feature: Dict[str, List[float]] = collections.defaultdict(list)
    per_image_rows: List[dict] = list(done_rows)
    for row in done_rows:
        for k in feature_order:
            values_by_feature[k].append(row[k])

    t_start = time.perf_counter()
    n_new = 0
    failure_info: Optional[dict] = None

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
            csv_file.flush()
            os.fsync(csv_file.fileno())

        for rec in corpus_records:
            rel_path = rec["relative_path"]
            if rel_path in done_paths:
                continue
            img_path = images_dir / rel_path

            try:
                current_sha = _sha256_file(img_path)
                if current_sha != rec["image_sha256"]:
                    raise SaliencyNormsError(
                        f"Image content changed since corpus discovery: "
                        f"{rel_path!r}")
                heatmap = predictor.predict_saliency(str(img_path))
                feats = extractor_fn(heatmap)
                if set(feats.keys()) != set(feature_order):
                    raise SaliencyNormsError(
                        f"Extractor returned unexpected feature keys for "
                        f"{rel_path!r}: {sorted(feats.keys())}")
                staged: Dict[str, float] = {}
                for k in feature_order:
                    fv = float(feats[k])
                    if not np.isfinite(fv):
                        raise SaliencyNormsError(
                            f"Non-finite feature {k}={fv!r} for {rel_path!r}")
                    staged[k] = fv
            except Exception as exc:
                failure_info = {
                    "global_index": rec["global_index"],
                    "relative_path": rel_path,
                    "category": rec["category"],
                    "reason": str(exc),
                }
                print(f"FAILED [{rec['global_index']}/{len(corpus_records)}] "
                      f"{rel_path}: {exc}", file=sys.stderr)
                break

            row = {"global_index": rec["global_index"], "category": rec["category"],
                  "relative_path": rel_path, ROW_IMAGE_SHA_COLUMN: rec["image_sha256"],
                  "native_width": rec["native_width"],
                  "native_height": rec["native_height"],
                  ROW_FEATURE_ORDER_COLUMN: feature_order_str,
                  ROW_FINGERPRINT_COLUMN: fingerprint_sha}
            for k in feature_order:
                row[k] = staged[k]
                values_by_feature[k].append(staged[k])
            per_image_rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            os.fsync(csv_file.fileno())
            n_new += 1
            done_paths.add(rel_path)

            current_segment["n_new_rows"] = n_new
            write_checkpoint_sidecar(sidecar_path, run_fingerprint, fingerprint_sha,
                                     csv_basename, all_segments, STATUS_RUNNING,
                                     recovered)
            print(f"[{rec['global_index']}/{len(corpus_records)}] {rel_path} "
                  f"({rec['category']})")

    elapsed = time.perf_counter() - t_start

    if failure_info is not None:
        # First-failure fail-closed policy: stop immediately, preserve valid
        # completed rows, record the precise failure, produce no aggregate.
        write_checkpoint_sidecar(sidecar_path, run_fingerprint, fingerprint_sha,
                                 csv_basename, all_segments, STATUS_FAILED,
                                 recovered, last_failure=failure_info)
        raise SaliencyNormsError(
            f"Image failure at global_index={failure_info['global_index']} "
            f"({failure_info['relative_path']}): {failure_info['reason']}. "
            f"Stopping immediately; {len(per_image_rows)} valid row(s) "
            f"preserved; no aggregate artifact written.")

    _finalize_segment(current_segment, elapsed, n_new)

    # ---- 8. Fail-closed full-run integrity, then aggregation.
    total_usable = len(per_image_rows)
    if total_usable != FULL_RUN_TOTAL:
        write_checkpoint_sidecar(sidecar_path, run_fingerprint, fingerprint_sha,
                                 csv_basename, all_segments, STATUS_RUNNING,
                                 recovered)
        raise SaliencyNormsError(
            f"Run requires exactly {FULL_RUN_TOTAL} usable rows; got "
            f"{total_usable}. Refusing to write aggregate.")
    processed_counts = collections.Counter(r["category"] for r in per_image_rows)
    for cat in AUTHORIZED_CATEGORIES:
        if processed_counts.get(cat, 0) != FULL_RUN_PER_CATEGORY:
            raise SaliencyNormsError(
                f"Run requires {FULL_RUN_PER_CATEGORY} in {cat}; got "
                f"{processed_counts.get(cat, 0)}.")
    produced_paths = {r["relative_path"] for r in per_image_rows}
    if produced_paths != set(manifest_by_path):
        raise SaliencyNormsError(
            "Produced row set differs from the authorized corpus manifest.")

    norms = aggregate(values_by_feature)
    for k in feature_order:
        n_dist = norms.get(k, {}).get("n", 0)
        if n_dist != total_usable:
            raise SaliencyNormsError(
                f"Aggregate integrity error: feature {k!r} distribution has "
                f"n={n_dist} but {total_usable} rows are counted as usable.")

    write_checkpoint_sidecar(sidecar_path, run_fingerprint, fingerprint_sha,
                             csv_basename, all_segments, STATUS_COMPLETED, recovered)

    norms_payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_order": feature_order,
        "resolution_policy": RESOLUTION_POLICY,
        "num_images": total_usable,
        "features": norms,
    }
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_git_commit": git_commit,
        "generator_sha256": generator_sha,
        "production_umsi_source_sha256": identity["production_source_sha256"],
        "weights_sha256": identity["weights_sha256"],
        "weights_byte_size": identity["weights_byte_size"],
        "predictor_module_sha256": identity["predictor_module_sha256"],
        "feature_extractor_module_sha256": identity["extractor_module_sha256"],
        "p9_report_sha256": identity["p9_report_sha256"],
        "p9_report_path": str(Path(args.p9_report).resolve()),
        "p9_conclusion": identity["p9_conclusion"],
        "p9_saliency_norms_release_status": identity["p9_saliency_norms_release_status"],
        "environment": env,
        "corpus_manifest_sha256": corpus_manifest_sha,
        "feature_order": feature_order,
        "resolution_policy": RESOLUTION_POLICY,
        "authorized_category_counts": {k: int(category_counts.get(k, 0))
                                       for k in AUTHORIZED_CATEGORIES},
        "authorized_total": int(sum(category_counts.get(k, 0)
                                    for k in AUTHORIZED_CATEGORIES)),
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_sha256": fingerprint_sha,
        "n_failed_images": 0,
        "recovered_from_interruption": bool(recovered),
        "invocation_count": len(all_segments),
        "segments": all_segments,
    }

    _refuse_protected_output(norms_path, "norms JSON")
    _refuse_protected_output(manifest_path, "manifest JSON")
    _atomic_write_json(norms_path, norms_payload)
    _atomic_write_json(manifest_path, manifest_payload)

    print(f"\nWrote {total_usable} rows -> {csv_path.name}")
    print(f"Wrote saliency norms -> {norms_path.name}")
    print(f"Wrote manifest -> {manifest_path.name}")
    print(f"  run_fingerprint_sha256={fingerprint_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
