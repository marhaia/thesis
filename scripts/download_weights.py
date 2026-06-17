#!/usr/bin/env python3
"""
Download the UMSI++ saliency model weights.

The weights file (`umsi++.hdf5`, ~433 MB) is too large to store in the git
repository. This script downloads it to the expected location.

Usage:
    python scripts/download_weights.py

You can override the download URL via an environment variable:
    WEIGHTS_URL="https://.../umsi++.hdf5" python scripts/download_weights.py

Where to host the file:
    The original UMSI++ project does not publish a direct download link. To make
    this script work, upload `umsi++.hdf5` somewhere that allows large files and
    put the direct URL below (or pass it via WEIGHTS_URL). A free option is a
    GitHub Release asset (up to 2 GB per file).
"""

import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Direct download URL for the weights file.
# TODO: replace with your hosted URL (e.g. a GitHub Release asset), or set the
#       WEIGHTS_URL environment variable when running this script.
WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")

# Target location (must match what saliency/umsi_model.py expects).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = (
    PROJECT_ROOT
    / "saliency"
    / "weights"
    / "model_weights"
    / "saliency_models"
    / "UMSI++"
    / "umsi++.hdf5"
)

# Roughly the expected file size (used as a sanity check). ~433 MB.
MIN_EXPECTED_BYTES = 400 * 1024 * 1024


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    sys.stdout.write(
        f"\r  Downloading... {pct:5.1f}%  "
        f"({_human(downloaded)} / {_human(total_size)})"
    )
    sys.stdout.flush()


def main() -> int:
    # 1. Already present?
    if TARGET_PATH.exists() and TARGET_PATH.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"✅ Weights already present: {TARGET_PATH}")
        print(f"   Size: {_human(TARGET_PATH.stat().st_size)}")
        return 0

    # 2. Do we have a URL?
    if not WEIGHTS_URL:
        print("❌ No download URL configured.")
        print()
        print("   The UMSI++ weights are not publicly hosted at a fixed URL.")
        print("   To use this script:")
        print("     1. Obtain `umsi++.hdf5` (see README / UEyes CHI2023 project).")
        print("     2. Upload it somewhere that allows large files,")
        print("        e.g. a GitHub Release asset (free, up to 2 GB).")
        print("     3. Set the URL in this script (WEIGHTS_URL) or run:")
        print('        WEIGHTS_URL="https://.../umsi++.hdf5" \\')
        print("            python scripts/download_weights.py")
        print()
        print(f"   Alternatively, place the file manually at:")
        print(f"     {TARGET_PATH}")
        return 1

    # 3. Download.
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from:\n  {WEIGHTS_URL}")
    print(f"Saving to:\n  {TARGET_PATH}")
    tmp_path = TARGET_PATH.with_suffix(".hdf5.part")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp_path, _progress)
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"\n❌ Download failed: {exc}")
        return 1

    # 4. Sanity check + finalize.
    if tmp_path.stat().st_size < MIN_EXPECTED_BYTES:
        size = _human(tmp_path.stat().st_size)
        tmp_path.unlink()
        print(f"❌ Downloaded file is too small ({size}). The URL may be wrong.")
        return 1

    tmp_path.replace(TARGET_PATH)
    print(f"✅ Done. Saved {_human(TARGET_PATH.stat().st_size)} to {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
