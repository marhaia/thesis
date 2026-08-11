"""Pinned P6 runtime, reference-pack, cache, and study-export identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from saliency.checkpoint_identity import load_umsi_checkpoint_identity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE1_ROOT = Path(__file__).resolve().parent
RUNTIME_MANIFEST_PATH = STAGE1_ROOT / "runtime_environment_manifest.json"
RUNTIME_MANIFEST_HASH_PATH = STAGE1_ROOT / "runtime_environment_manifest.sha256"
REFERENCE_MANIFEST_PATH = STAGE1_ROOT / "reference_pack_manifest.json"
REFERENCE_MANIFEST_HASH_PATH = STAGE1_ROOT / "reference_pack_manifest.sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReproducibilityError(RuntimeError):
    """Raised when a frozen P6 identity is missing, changed, or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(path: Path) -> str:
    try:
        fields = Path(path).read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise ReproducibilityError("Pinned manifest hash is unavailable.") from exc
    if not fields or not _SHA256_RE.fullmatch(fields[0]):
        raise ReproducibilityError("Pinned manifest hash is malformed.")
    return fields[0]


def _load_pinned_manifest(
    manifest_path: Path,
    hash_path: Path,
    schema_version: str,
) -> tuple[dict, str]:
    expected = _expected_sha256(hash_path)
    try:
        actual = sha256_file(manifest_path)
        with Path(manifest_path).open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, ValueError) as exc:
        raise ReproducibilityError("Pinned manifest is unavailable or invalid.") from exc
    if actual != expected:
        raise ReproducibilityError(
            f"Pinned manifest SHA-256 mismatch: expected {expected}, found {actual}."
        )
    if manifest.get("schema_version") != schema_version:
        raise ReproducibilityError("Pinned manifest schema version mismatch.")
    return manifest, actual


def _verify_artifact(spec: Mapping[str, object]) -> None:
    try:
        relative_path = str(spec["path"])
        expected = str(spec["sha256"])
    except KeyError as exc:
        raise ReproducibilityError("Pinned artifact identity is incomplete.") from exc
    if not _SHA256_RE.fullmatch(expected):
        raise ReproducibilityError("Pinned artifact SHA-256 is malformed.")
    path = PROJECT_ROOT / relative_path
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ReproducibilityError(
            f"Pinned artifact is unavailable: {relative_path}."
        ) from exc
    if actual != expected:
        raise ReproducibilityError(
            f"Pinned artifact SHA-256 mismatch for {relative_path}."
        )


@lru_cache(maxsize=1)
def load_runtime_environment_manifest() -> tuple[dict, str]:
    manifest, manifest_sha = _load_pinned_manifest(
        RUNTIME_MANIFEST_PATH,
        RUNTIME_MANIFEST_HASH_PATH,
        "stage1-runtime-environment-v1",
    )
    freeze = manifest.get("dependency_freeze", {})
    for path_key, sha_key in (
        ("lock_file", "lock_file_sha256"),
        ("requirements_file", "requirements_file_sha256"),
        ("ci_requirements_file", "ci_requirements_file_sha256"),
    ):
        _verify_artifact({"path": freeze.get(path_key), "sha256": freeze.get(sha_key)})
    return manifest, manifest_sha


def _locked_distributions(lock_path: Path) -> dict:
    locked = {}
    try:
        lines = Path(lock_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReproducibilityError("Runtime lock file is unavailable.") from exc
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ReproducibilityError("Runtime lock file contains an invalid entry.")
        name, version = line.split("==", 1)
        if not name or not version:
            raise ReproducibilityError("Runtime lock file contains an invalid entry.")
        locked[name] = version
    if not locked:
        raise ReproducibilityError("Runtime lock file is empty.")
    return locked


@lru_cache(maxsize=1)
def verify_runtime_environment() -> dict:
    """Verify the executing environment or explicitly report a CI bypass."""
    manifest, _ = load_runtime_environment_manifest()
    mode = os.environ.get("STAGE1_RUNTIME_VERIFICATION", "strict").strip().lower()
    if mode == "metadata_only":
        return {
            "status": "not_verified",
            "mode": mode,
            "reason": "explicit lightweight-CI bypass",
        }
    if mode != "strict":
        raise ReproducibilityError("STAGE1_RUNTIME_VERIFICATION is invalid.")

    expected_platform = manifest["platform"]
    observed_platform = {
        "operating_system": f"macOS {platform.mac_ver()[0]}",
        "machine": platform.machine(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
    }
    if observed_platform != expected_platform:
        raise ReproducibilityError(
            "Runtime platform does not match the frozen Stage-1 environment."
        )

    lock_relative = manifest["dependency_freeze"]["lock_file"]
    locked = _locked_distributions(PROJECT_ROOT / lock_relative)
    mismatches = []
    for name, expected_version in locked.items():
        try:
            observed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed_version = None
        if observed_version != expected_version:
            mismatches.append((name, expected_version, observed_version))
    if mismatches:
        name, expected, observed = mismatches[0]
        raise ReproducibilityError(
            "Runtime dependency lock mismatch: "
            f"{name} expected {expected}, found {observed or 'missing'} "
            f"({len(mismatches)} mismatch(es))."
        )
    return {
        "status": "verified",
        "mode": mode,
        "locked_distributions": len(locked),
        "platform": observed_platform,
    }


@lru_cache(maxsize=1)
def load_reference_pack_manifest() -> tuple[dict, str]:
    manifest, manifest_sha = _load_pinned_manifest(
        REFERENCE_MANIFEST_PATH,
        REFERENCE_MANIFEST_HASH_PATH,
        "stage1-reference-pack-v1",
    )
    reference_pack = manifest.get("reference_pack")
    if not isinstance(reference_pack, dict) or not reference_pack:
        raise ReproducibilityError("Pinned reference pack is malformed.")
    for spec in reference_pack.values():
        if not isinstance(spec, dict):
            raise ReproducibilityError("Pinned reference-pack entry is malformed.")
        _verify_artifact(spec)
    return manifest, manifest_sha


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def saliency_cache_identity() -> str:
    verify_runtime_environment()
    runtime, runtime_sha = load_runtime_environment_manifest()
    reference, reference_sha = load_reference_pack_manifest()
    checkpoint = load_umsi_checkpoint_identity()
    feature_norms = reference["reference_pack"]["feature_norms"]
    return _canonical_digest({
        "schema": "stage1-saliency-cache-v1",
        "checkpoint_sha256": checkpoint["sha256"],
        "umsi_model_sha256": sha256_file(PROJECT_ROOT / "saliency/umsi_model.py"),
        "postprocessing_sha256": sha256_file(
            PROJECT_ROOT / "saliency/postprocessing.py"
        ),
        "saliency_features_sha256": sha256_file(
            PROJECT_ROOT / "saliency/saliency_features.py"
        ),
        "feature_norms_sha256": feature_norms["sha256"],
        "reference_pack_manifest_sha256": reference_sha,
        "runtime_environment_manifest_sha256": runtime_sha,
        "runtime_schema": runtime["schema_version"],
    })


@lru_cache(maxsize=1)
def visual_cache_identity() -> str:
    verify_runtime_environment()
    _, runtime_sha = load_runtime_environment_manifest()
    reference, reference_sha = load_reference_pack_manifest()
    return _canonical_digest({
        "schema": "stage1-visual-cache-v1",
        "visual_complexity_sha256": sha256_file(
            PROJECT_ROOT / "stage1/visual_complexity.py"
        ),
        "feature_norms_sha256": reference["reference_pack"]["feature_norms"][
            "sha256"
        ],
        "reference_pack_manifest_sha256": reference_sha,
        "runtime_environment_manifest_sha256": runtime_sha,
    })


@lru_cache(maxsize=1)
def resolve_source_state() -> dict:
    override = os.environ.get("STAGE1_SOURCE_COMMIT", "").strip().lower()
    if override:
        if not _COMMIT_RE.fullmatch(override):
            raise ReproducibilityError("STAGE1_SOURCE_COMMIT must be a full SHA-1.")
        tree_state = os.environ.get("STAGE1_SOURCE_TREE_STATE", "exported").strip()
        if tree_state not in {"clean", "dirty", "exported"}:
            raise ReproducibilityError("STAGE1_SOURCE_TREE_STATE is invalid.")
        return {"commit": override, "tree_state": tree_state, "source": "environment"}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReproducibilityError(
            "Source commit is unavailable; set STAGE1_SOURCE_COMMIT for an exported build."
        ) from exc
    if not _COMMIT_RE.fullmatch(commit):
        raise ReproducibilityError("Resolved source commit is invalid.")
    return {
        "commit": commit,
        "tree_state": "dirty" if status.strip() else "clean",
        "source": "git",
    }


def study_reproducibility_metadata() -> dict:
    runtime, runtime_sha = load_runtime_environment_manifest()
    reference, reference_sha = load_reference_pack_manifest()
    checkpoint = load_umsi_checkpoint_identity()
    schemas = reference["stage1_schema_identifiers"]
    return {
        "schema_id": schemas["study_export"],
        "source": dict(resolve_source_state()),
        "runtime_environment_manifest_sha256": runtime_sha,
        "runtime_environment_schema": runtime["schema_version"],
        "runtime_verification": dict(verify_runtime_environment()),
        "reference_pack_manifest_sha256": reference_sha,
        "feature_norms_sha256": reference["reference_pack"]["feature_norms"][
            "sha256"
        ],
        "umsi_checkpoint_sha256": checkpoint["sha256"],
        "stage1_vector_schema": schemas["feature_vector"],
        "visual_norms_schema": schemas["canonical_visual"],
        "saliency_norms_schema": schemas["canonical_saliency"],
        "saliency_cache_identity": saliency_cache_identity(),
        "visual_cache_identity": visual_cache_identity(),
    }
