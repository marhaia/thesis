"""Exact external-artifact identity for score-driving Stage-1 EasyOCR models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IDENTITY_PATH = Path(__file__).with_name("easyocr_model_identity.json")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


class EasyOCRModelIntegrityError(RuntimeError):
    """Raised when EasyOCR code or weights do not match the Stage-1 contract."""


def _repository_path(relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise EasyOCRModelIntegrityError(f"EasyOCR {label} path is invalid.")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {label} path is not repository-relative."
        )
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {label} path escapes the repository."
        ) from exc
    return resolved


def _validate_artifact(role: str, artifact: object) -> None:
    if not isinstance(artifact, dict) or set(artifact) != {
        "filename",
        "bytes",
        "sha256",
        "upstream_md5",
        "acquisition_url",
    }:
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {role} artifact identity has an invalid schema."
        )
    if not isinstance(artifact["filename"], str) or not artifact["filename"].endswith(
        ".pth"
    ):
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} filename is invalid.")
    if Path(artifact["filename"]).name != artifact["filename"]:
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} filename is unsafe.")
    if not isinstance(artifact["bytes"], int) or artifact["bytes"] <= 0:
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} byte size is invalid.")
    if not isinstance(artifact["sha256"], str) or not _SHA256_RE.fullmatch(
        artifact["sha256"]
    ):
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} SHA-256 is invalid.")
    if not isinstance(artifact["upstream_md5"], str) or not _MD5_RE.fullmatch(
        artifact["upstream_md5"]
    ):
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} upstream MD5 is invalid.")
    if not isinstance(artifact["acquisition_url"], str) or not artifact[
        "acquisition_url"
    ].startswith("https://github.com/JaidedAI/EasyOCR/releases/download/"):
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {role} acquisition URL is invalid."
        )


def load_easyocr_model_identity(path: Path = IDENTITY_PATH) -> dict:
    """Load and validate the exact EasyOCR model and reader policy manifest."""
    try:
        with Path(path).open(encoding="utf-8") as source:
            identity = json.load(source)
    except (OSError, ValueError) as exc:
        raise EasyOCRModelIntegrityError(
            "EasyOCR model identity manifest is unavailable or invalid."
        ) from exc

    required = {
        "schema_version",
        "easyocr_version",
        "repository_relative_model_directory",
        "repository_relative_user_network_directory",
        "reader",
        "artifacts",
        "claim_boundary",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise EasyOCRModelIntegrityError(
            "EasyOCR model identity manifest has an invalid schema."
        )
    if identity["schema_version"] != 1:
        raise EasyOCRModelIntegrityError(
            "Unsupported EasyOCR model identity schema version."
        )
    if identity["easyocr_version"] != "1.7.2":
        raise EasyOCRModelIntegrityError("EasyOCR version identity is invalid.")

    _repository_path(
        identity["repository_relative_model_directory"], label="model-directory"
    )
    _repository_path(
        identity["repository_relative_user_network_directory"],
        label="user-network-directory",
    )

    reader = identity["reader"]
    expected_reader = {
        "languages": ["en"],
        "gpu": False,
        "detect_network": "craft",
        "recog_network": "standard",
        "download_enabled": False,
        "verbose": False,
        "quantize": True,
        "cudnn_benchmark": False,
    }
    if reader != expected_reader:
        raise EasyOCRModelIntegrityError(
            "EasyOCR strict Production reader policy is invalid."
        )

    artifacts = identity["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "detector",
        "recognizer",
    }:
        raise EasyOCRModelIntegrityError("EasyOCR artifact set is invalid.")
    for role, artifact in artifacts.items():
        _validate_artifact(role, artifact)
    if not isinstance(identity["claim_boundary"], str) or not identity[
        "claim_boundary"
    ]:
        raise EasyOCRModelIntegrityError("EasyOCR claim boundary is invalid.")
    return identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_artifact(
    artifact_path: Path,
    *,
    role: str,
    identity: Mapping[str, object],
) -> dict:
    """Fail closed unless one model file matches its declared exact identity."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise EasyOCRModelIntegrityError(
            f"Required EasyOCR {role} model is unavailable."
        )
    if artifact_path.name != identity["filename"]:
        raise EasyOCRModelIntegrityError(f"EasyOCR {role} filename mismatch.")

    expected_bytes = int(identity["bytes"])
    actual_bytes = artifact_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {role} byte-size mismatch: expected {expected_bytes}, "
            f"found {actual_bytes}."
        )

    expected_sha = str(identity["sha256"])
    actual_sha = sha256_file(artifact_path)
    if actual_sha != expected_sha:
        raise EasyOCRModelIntegrityError(
            f"EasyOCR {role} SHA-256 mismatch: expected {expected_sha}, "
            f"found {actual_sha}."
        )
    return {
        "filename": artifact_path.name,
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def verify_easyocr_models(
    model_directory: Optional[Path] = None,
    *,
    identity_path: Optional[Path] = None,
) -> dict:
    """Verify package and both model files before any EasyOCR reader is built."""
    identity = load_easyocr_model_identity(
        IDENTITY_PATH if identity_path is None else identity_path
    )
    try:
        installed_version = importlib.metadata.version("easyocr")
    except importlib.metadata.PackageNotFoundError as exc:
        raise EasyOCRModelIntegrityError(
            "Required EasyOCR package is unavailable."
        ) from exc
    if installed_version != identity["easyocr_version"]:
        raise EasyOCRModelIntegrityError(
            "EasyOCR package version does not match the Stage-1 identity."
        )

    resolved_model_directory = (
        _repository_path(
            identity["repository_relative_model_directory"],
            label="model-directory",
        )
        if model_directory is None
        else Path(model_directory).resolve()
    )
    verified = {}
    for role, artifact in identity["artifacts"].items():
        verified[role] = verify_model_artifact(
            resolved_model_directory / artifact["filename"],
            role=role,
            identity=artifact,
        )
    return {
        "identity": identity,
        "model_directory": resolved_model_directory,
        "user_network_directory": _repository_path(
            identity["repository_relative_user_network_directory"],
            label="user-network-directory",
        ),
        "artifacts": verified,
    }
