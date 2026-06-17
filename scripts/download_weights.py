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
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Public download URL (works once the repository is public).
# GitHub strips the "++" from filenames, so the asset is named "umsi.hdf5".
WEIGHTS_URL = os.environ.get(
    "WEIGHTS_URL",
    "https://github.com/marhaia/thesis/releases/download/weights-v1/umsi.hdf5",
)

# Authenticated GitHub API asset endpoint (works while the repository is still
# private). Used automatically when a token is available.
ASSET_API_URL = (
    "https://api.github.com/repos/marhaia/thesis/releases/assets/450177038"
)

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

# Roughly the expected file size (used as a sanity check). The file is ~115 MB;
# we require at least 100 MB so a small error page is never mistaken for it.
MIN_EXPECTED_BYTES = 100 * 1024 * 1024


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _get_github_token() -> str:
    """Return a GitHub token, or an empty string if none is available.

    Tries, in order:
      1. The GITHUB_TOKEN environment variable.
      2. The macOS git credential helper (osxkeychain) used for github.com.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token
    try:
        out = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return ""
    for line in out.splitlines():
        if line.startswith("password="):
            return line[len("password="):]
    return ""


def _build_request() -> "urllib.request.Request":
    """Build the download request, using an authenticated API endpoint if the
    public URL is not reachable (e.g. while the repository is private)."""
    token = _get_github_token()
    if token:
        # Authenticated asset endpoint works for both private and public repos.
        req = urllib.request.Request(ASSET_API_URL)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/octet-stream")
        return req
    # No token: fall back to the public URL (works once the repo is public).
    return urllib.request.Request(WEIGHTS_URL)


def main() -> int:
    # 1. Already present?
    if TARGET_PATH.exists() and TARGET_PATH.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"✅ Weights already present: {TARGET_PATH}")
        print(f"   Size: {_human(TARGET_PATH.stat().st_size)}")
        return 0

    # 2. Build the request (authenticated if a token is available).
    req = _build_request()
    using_auth = any(h.lower() == "authorization" for h in req.headers)
    print("Downloading UMSI++ weights" + (" (authenticated)" if using_auth else ""))
    print(f"Saving to:\n  {TARGET_PATH}")

    # 3. Download (streamed, with progress).
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TARGET_PATH.with_suffix(".hdf5.part")
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 256
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(downloaded / total * 100, 100)
                        sys.stdout.write(
                            f"\r  {pct:5.1f}%  "
                            f"({_human(downloaded)} / {_human(total)})"
                        )
                        sys.stdout.flush()
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"\n❌ Download failed: {exc}")
        if not using_auth:
            print("   The repository may still be private. Set a token via")
            print('   GITHUB_TOKEN="<your token>" and try again, or place the')
            print(f"   file manually at:\n     {TARGET_PATH}")
        return 1

    # 4. Sanity check + finalize.
    if tmp_path.stat().st_size < MIN_EXPECTED_BYTES:
        size = _human(tmp_path.stat().st_size)
        tmp_path.unlink()
        print(f"❌ Downloaded file is too small ({size}). The URL may be wrong")
        print("   or the repository is private and no token was provided.")
        return 1

    tmp_path.replace(TARGET_PATH)
    print(f"✅ Done. Saved {_human(TARGET_PATH.stat().st_size)} to {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
