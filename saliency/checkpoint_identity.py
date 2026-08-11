"""Authoritative identity and fail-closed verification for the UMSI checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional


IDENTITY_PATH = Path(__file__).with_name("umsi_checkpoint_identity.json")


class CheckpointIntegrityError(RuntimeError):
    """Raised when checkpoint bytes do not match the frozen P6 identity."""


def load_umsi_checkpoint_identity(path: Path = IDENTITY_PATH) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as source:
            identity = json.load(source)
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            "UMSI checkpoint identity manifest is unavailable or invalid."
        ) from exc

    required = {
        "schema_version",
        "artifact",
        "repository_relative_path",
        "bytes",
        "sha256",
        "acquisition_url",
        "evidence_contract",
        "claim_boundary",
    }
    if set(identity) != required:
        raise CheckpointIntegrityError(
            "UMSI checkpoint identity manifest has an invalid schema."
        )
    if identity["schema_version"] != 1:
        raise CheckpointIntegrityError(
            "Unsupported UMSI checkpoint identity schema version."
        )
    if not isinstance(identity["artifact"], str) or not identity["artifact"]:
        raise CheckpointIntegrityError("UMSI checkpoint artifact name is invalid.")
    if not isinstance(identity["bytes"], int) or identity["bytes"] <= 0:
        raise CheckpointIntegrityError("UMSI checkpoint byte size is invalid.")
    sha256 = identity["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise CheckpointIntegrityError("UMSI checkpoint SHA-256 is invalid.")
    return identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    checkpoint_path: Path,
    *,
    identity: Mapping[str, object],
    require_filename: bool = True,
) -> dict:
    """Fail closed unless a checkpoint matches the supplied frozen identity."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise CheckpointIntegrityError("Required UMSI checkpoint is unavailable.")
    if require_filename and checkpoint_path.name != identity["artifact"]:
        raise CheckpointIntegrityError("UMSI checkpoint filename mismatch.")

    expected_bytes = int(identity["bytes"])
    actual_bytes = checkpoint_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise CheckpointIntegrityError(
            f"UMSI checkpoint byte-size mismatch: expected {expected_bytes}, "
            f"found {actual_bytes}."
        )

    expected_sha = str(identity["sha256"])
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise CheckpointIntegrityError(
            f"UMSI checkpoint SHA-256 mismatch: expected {expected_sha}, "
            f"found {actual_sha}."
        )
    return {
        "artifact": str(identity["artifact"]),
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def verify_umsi_checkpoint(
    checkpoint_path: Path,
    *,
    require_filename: bool = True,
    identity_path: Optional[Path] = None,
) -> dict:
    identity = load_umsi_checkpoint_identity(
        IDENTITY_PATH if identity_path is None else identity_path
    )
    return verify_checkpoint(
        checkpoint_path,
        identity=identity,
        require_filename=require_filename,
    )
