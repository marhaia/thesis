"""Verify the P5 HCEye source-coefficient matrix against a pinned CSV.

This is an audit-evidence utility, not a model-training or validation script.
It verifies only that the repository's aggregate HCEye source values can be
re-derived from the exact external CSV bytes identified by the manifest.  It
does not validate the project-specific screenshot mapping or combined index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "hceye" / "hceye_coefficient_provenance.json"
sys.path.insert(0, str(ROOT))

from hceye.hceye_features import HCEYE_COEFFICIENTS  # noqa: E402


class HCEyeProvenanceError(RuntimeError):
    """Raised when source identity or coefficient derivation does not match."""


REQUIRED_COLUMNS = {
    "Participant_ID",
    "Image_Name",
    "CognitiveLoad",
    "Highlight",
    "AOI_Hit",
    "TotalNumFixations",
    "MeanFixationDuration",
    "FixationFrequency",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def derive_source_coefficients(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], dict, dict]:
    """Apply only the aggregation policies frozen in the P5 manifest."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise HCEyeProvenanceError(
            "HCEye CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    metrics = [
        "TotalNumFixations",
        "MeanFixationDuration",
        "FixationFrequency",
    ]
    image_condition = (
        df.groupby(["Image_Name", "CognitiveLoad"])[metrics]
        .mean()
        .unstack("CognitiveLoad")
    )

    def image_ratios(metric: str) -> pd.Series:
        try:
            ratios = (
                image_condition[(metric, "High")]
                / image_condition[(metric, "Absent")]
            )
        except KeyError as exc:
            raise HCEyeProvenanceError(
                f"HCEye CSV cannot form High/Absent ratios for {metric}."
            ) from exc
        ratios = ratios.dropna()
        if len(ratios) != 150 or not (ratios.map(math.isfinite).all()):
            raise HCEyeProvenanceError(
                f"Expected 150 finite per-image ratios for {metric}; "
                f"found {len(ratios)}."
            )
        return ratios

    fixation_ratios = image_ratios("TotalNumFixations")
    duration_ratios = image_ratios("MeanFixationDuration")
    frequency_ratios = image_ratios("FixationFrequency")

    condition_means = df.groupby("CognitiveLoad")[metrics].mean()
    try:
        frequency_ratio = (
            condition_means.loc["High", "FixationFrequency"]
            / condition_means.loc["Absent", "FixationFrequency"]
        )
    except KeyError as exc:
        raise HCEyeProvenanceError(
            "HCEye CSV cannot form the global High/Absent frequency ratio."
        ) from exc

    aoi_rates = df.groupby(["CognitiveLoad", "Highlight"])["AOI_Hit"].mean()

    baseline = df[
        (df["CognitiveLoad"] == "Absent") & (df["Highlight"] == "Absent")
    ]
    baseline_images = baseline.groupby("Image_Name")[metrics].mean()
    if len(baseline_images) != 150:
        raise HCEyeProvenanceError(
            "Expected 150 Absent-load/Absent-highlight baseline images; "
            f"found {len(baseline_images)}."
        )

    def aoi(load: str, highlight: str) -> float:
        try:
            return float(aoi_rates.loc[(load, highlight)])
        except KeyError as exc:
            raise HCEyeProvenanceError(
                f"HCEye CSV is missing AOI condition {load}/{highlight}."
            ) from exc

    coefficients = {
        "fixation_reduction_mean": float(fixation_ratios.mean()),
        "fixation_reduction_std": float(fixation_ratios.std(ddof=1)),
        "duration_increase_mean": float(duration_ratios.mean()),
        "duration_increase_std": float(duration_ratios.std(ddof=1)),
        "frequency_reduction_mean": float(frequency_ratio),
        "aoi_hit_absent_no_hl": aoi("Absent", "Absent"),
        "aoi_hit_high_no_hl": aoi("High", "Absent"),
        "aoi_hit_absent_dynamic": aoi("Absent", "Dynamic"),
        "aoi_hit_high_dynamic": aoi("High", "Dynamic"),
        "aoi_hit_absent_static": aoi("Absent", "Static"),
        "aoi_hit_high_static": aoi("High", "Static"),
        "baseline_fixations_mean": float(
            baseline_images["TotalNumFixations"].mean()
        ),
        "baseline_fixations_std": float(
            baseline_images["TotalNumFixations"].std(ddof=1)
        ),
        "baseline_duration_mean": float(
            baseline_images["MeanFixationDuration"].mean()
        ),
        "baseline_duration_std": float(
            baseline_images["MeanFixationDuration"].std(ddof=1)
        ),
        "baseline_frequency_mean": float(
            baseline_images["FixationFrequency"].mean()
        ),
        "baseline_frequency_std": float(
            baseline_images["FixationFrequency"].std(ddof=1)
        ),
    }
    identity = {
        "rows": int(len(df)),
        "participants": int(df["Participant_ID"].nunique()),
        "images": int(df["Image_Name"].nunique()),
        "cognitive_load_conditions": sorted(
            str(value) for value in df["CognitiveLoad"].dropna().unique()
        ),
        "highlight_conditions": sorted(
            str(value) for value in df["Highlight"].dropna().unique()
        ),
    }
    diagnostics = {
        "frequency_reduction_std": float(frequency_ratios.std(ddof=1)),
    }
    return coefficients, identity, diagnostics


def verify_csv(csv_path: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Fail closed unless source bytes, dataset identity and values all match."""
    manifest = load_manifest(manifest_path)
    source = manifest["source"]
    expected_size = int(source["bytes"])
    actual_size = csv_path.stat().st_size
    if actual_size != expected_size:
        raise HCEyeProvenanceError(
            f"HCEye CSV byte-size mismatch: expected {expected_size}, "
            f"found {actual_size}."
        )

    actual_sha = _sha256(csv_path)
    if actual_sha != source["sha256"]:
        raise HCEyeProvenanceError(
            f"HCEye CSV SHA-256 mismatch: expected {source['sha256']}, "
            f"found {actual_sha}."
        )

    derived, identity, diagnostics = derive_source_coefficients(
        pd.read_csv(csv_path)
    )
    if identity != manifest["dataset_identity"]:
        raise HCEyeProvenanceError(
            f"HCEye dataset identity mismatch: expected "
            f"{manifest['dataset_identity']}, found {identity}."
        )

    matrix = manifest["coefficients"]
    expected_keys = set(matrix)
    if set(derived) != expected_keys:
        raise HCEyeProvenanceError("Derived coefficient keys do not match manifest.")
    if set(HCEYE_COEFFICIENTS) != expected_keys:
        raise HCEyeProvenanceError(
            "Production HCEYE_COEFFICIENTS keys do not match provenance matrix."
        )

    for name, spec in matrix.items():
        actual_source = derived[name]
        frozen_source = float(spec["source_value"])
        if not math.isclose(actual_source, frozen_source, rel_tol=0.0, abs_tol=1e-12):
            raise HCEyeProvenanceError(
                f"{name} source derivation mismatch: expected {frozen_source!r}, "
                f"found {actual_source!r}."
            )
        stored = float(spec["stored_value"])
        if round(actual_source, int(spec["round_digits"])) != stored:
            raise HCEyeProvenanceError(
                f"{name} rounding mismatch: derived {actual_source!r} does not "
                f"round to stored value {stored!r}."
            )
        if float(HCEYE_COEFFICIENTS[name]) != stored:
            raise HCEyeProvenanceError(
                f"{name} production mismatch: manifest stores {stored!r}, "
                f"code stores {HCEYE_COEFFICIENTS[name]!r}."
            )

    removed_frequency_std = manifest["removed_legacy_metadata"][
        "frequency_reduction_std"
    ]
    expected_diagnostic = float(
        removed_frequency_std["per_image_ratio_sample_std"]
    )
    if not math.isclose(
        diagnostics["frequency_reduction_std"],
        expected_diagnostic,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise HCEyeProvenanceError(
            "Removed frequency_reduction_std diagnostic mismatch: expected "
            f"{expected_diagnostic!r}, found "
            f"{diagnostics['frequency_reduction_std']!r}."
        )

    return {
        "status": "PASS",
        "source_sha256": actual_sha,
        "rows": identity["rows"],
        "participants": identity["participants"],
        "images": identity["images"],
        "verified_coefficients": len(matrix),
        "removed_unreproducible_metadata_values": 1,
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify hash-pinned HCEye source-coefficient provenance."
    )
    parser.add_argument(
        "--csv", required=True, type=Path, help="External HCEye fixation CSV"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Repository-pinned provenance manifest",
    )
    args = parser.parse_args()
    try:
        report = verify_csv(args.csv, args.manifest)
    except (OSError, ValueError, KeyError, HCEyeProvenanceError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
