#!/usr/bin/env python3
"""Install the two exact external EasyOCR model files required by Stage 1."""

from __future__ import annotations

import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cognitive.easyocr_identity import (  # noqa: E402
    EasyOCRModelIntegrityError,
    load_easyocr_model_identity,
    verify_easyocr_models,
    verify_model_artifact,
)


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _download_artifact(role: str, artifact: dict, model_directory: Path) -> None:
    target = model_directory / artifact["filename"]
    if target.exists():
        try:
            verified = verify_model_artifact(
                target, role=role, identity=artifact
            )
        except EasyOCRModelIntegrityError as exc:
            print(f"Existing {role} model failed verification: {exc}")
        else:
            print(
                f"Verified {role}: {target} "
                f"({_human(verified['bytes'])}, {verified['sha256']})"
            )
            return

    print(f"Downloading exact {role} model from {artifact['acquisition_url']}")
    model_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"easyocr-{role}-", dir=model_directory
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        archive_path = temporary / "model.zip"
        request = urllib.request.Request(
            artifact["acquisition_url"],
            headers={"User-Agent": "Thesis-Stage1-EasyOCR-Installer/1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open(
            "wb"
        ) as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)

        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                member
                for member in archive.infolist()
                if Path(member.filename).name == artifact["filename"]
                and not member.is_dir()
            ]
            if len(matches) != 1:
                raise EasyOCRModelIntegrityError(
                    f"EasyOCR {role} archive does not contain exactly one "
                    f"{artifact['filename']}."
                )
            extracted = temporary / artifact["filename"]
            with archive.open(matches[0]) as source, extracted.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)

        verified = verify_model_artifact(extracted, role=role, identity=artifact)
        extracted.replace(target)
        print(
            f"Installed {role}: {target} "
            f"({_human(verified['bytes'])}, {verified['sha256']})"
        )


def main() -> int:
    identity = load_easyocr_model_identity()
    model_directory = (
        PROJECT_ROOT / identity["repository_relative_model_directory"]
    )
    try:
        for role, artifact in identity["artifacts"].items():
            _download_artifact(role, artifact, model_directory)
        result = verify_easyocr_models(model_directory)
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        EasyOCRModelIntegrityError,
    ) as exc:
        print(f"EasyOCR model installation failed: {exc}")
        return 1

    print("EasyOCR strict Production model identity verified:")
    for role, artifact in result["artifacts"].items():
        print(f"  {role}: {artifact['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
