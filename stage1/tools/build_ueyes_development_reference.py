#!/usr/bin/env python3
"""Build the UEyes development reference and held-out study candidate manifest.

The official UEyes split is treated as a strict development/evaluation boundary:

* ``Train``: 468 desktop + 468 mobile + 468 web images. These 1,404 images
  define the production reference distributions.
* ``Test``: 27 images per GUI category. These 81 images never contribute to
  reference statistics. A deterministic, block-balanced policy selects 20 per
  category as study candidates and orders the remaining seven as reserves.

This tool does not train UMSI++, re-run feature extraction, or inspect human
outcomes. It reaggregates the independently generated, per-image canonical
visual and native-input saliency rows after verifying their identities against
the on-disk UEyes corpus. Outputs are written to a caller-provided directory and
never overwrite production artifacts implicitly.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "ueyes-development-reference-v1"
VISUAL_SCHEMA_VERSION = "canonical-visual-feature-norms-v2"
SALIENCY_SCHEMA_VERSION = "canonical-saliency-feature-norms-v2"
SELECTION_SCHEMA_VERSION = "ueyes-heldout-study-candidates-v1"

AUTHORIZED_CATEGORIES = ("desktop", "mobile", "web")
AUTHORIZED_SET = frozenset(AUTHORIZED_CATEGORIES)
TRAIN_PER_CATEGORY = 468
TEST_PER_CATEGORY = 27
TRAIN_TOTAL = TRAIN_PER_CATEGORY * len(AUTHORIZED_CATEGORIES)
TEST_TOTAL = TEST_PER_CATEGORY * len(AUTHORIZED_CATEGORIES)
FULL_TOTAL = TRAIN_TOTAL + TEST_TOTAL
TEST_BLOCKS = (53, 54, 55)
TEST_PER_CATEGORY_BLOCK = 9
SELECTED_PER_CATEGORY = 20

# Rotating the six-image block keeps the aggregate selection exactly balanced:
# each official test block contributes 20 images across the three categories.
SELECTION_QUOTAS = {
    "desktop": {53: 6, 54: 7, 55: 7},
    "mobile": {53: 7, 54: 6, 55: 7},
    "web": {53: 7, 54: 7, 55: 6},
}

DEFAULT_SELECTION_SEED = "ueyes-heldout-study-v1"

VISUAL_FEATURE_KEYS = (
    "shannon_entropy",
    "edge_density",
    "feature_congestion",
    "subband_entropy",
    "layout_symmetry",
    "chromatic_coherence",
    "visual_hierarchy",
    "interactive_element_density",
)

SALIENCY_FEATURE_KEYS = (
    "saliency_dispersion",
    "saliency_peak_count",
    "saliency_center_bias",
    "saliency_entropy",
    "saliency_coverage",
)


class DevelopmentReferenceError(RuntimeError):
    """Raised when the split, source rows, or generated artifacts are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_block(raw: object) -> int:
    text = str(raw or "").strip().replace(",", ".")
    try:
        value = float(text)
    except ValueError as exc:
        raise DevelopmentReferenceError(f"Invalid UEyes block value: {raw!r}.") from exc
    if not math.isfinite(value) or value != int(value):
        raise DevelopmentReferenceError(f"Invalid UEyes block value: {raw!r}.")
    return int(value)


def load_official_split(types_csv: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with Path(types_csv).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, delimiter=";")
        required = {"Image Name", "Category", "Block", "Train/Test"}
        if not required.issubset(reader.fieldnames or []):
            raise DevelopmentReferenceError(
                f"UEyes split CSV must contain {sorted(required)}."
            )
        for row in reader:
            filename = (row.get("Image Name") or "").strip()
            category = (row.get("Category") or "").strip().lower()
            partition = (row.get("Train/Test") or "").strip().title()
            if not filename:
                raise DevelopmentReferenceError("UEyes split contains an empty filename.")
            if filename in records:
                raise DevelopmentReferenceError(
                    f"UEyes split contains duplicate filename {filename!r}."
                )
            if category not in AUTHORIZED_SET and category != "poster":
                raise DevelopmentReferenceError(
                    f"UEyes split contains unsupported category {category!r}."
                )
            if partition not in {"Train", "Test"}:
                raise DevelopmentReferenceError(
                    f"UEyes split contains unsupported partition {partition!r}."
                )
            records[filename] = {
                "filename": filename,
                "image_id": Path(filename).stem,
                "category": category,
                "block": _parse_block(row.get("Block")),
                "partition": partition,
            }
    return records


def validate_official_split(records: Mapping[str, Mapping[str, object]]) -> None:
    gui_records = [r for r in records.values() if r["category"] in AUTHORIZED_SET]
    if len(gui_records) != FULL_TOTAL:
        raise DevelopmentReferenceError(
            f"Expected {FULL_TOTAL} authorized GUI rows, found {len(gui_records)}."
        )
    counts = collections.Counter(
        (str(r["category"]), str(r["partition"])) for r in gui_records
    )
    for category in AUTHORIZED_CATEGORIES:
        if counts[(category, "Train")] != TRAIN_PER_CATEGORY:
            raise DevelopmentReferenceError(
                f"Expected {TRAIN_PER_CATEGORY} {category} Train rows, found "
                f"{counts[(category, 'Train')]}."
            )
        if counts[(category, "Test")] != TEST_PER_CATEGORY:
            raise DevelopmentReferenceError(
                f"Expected {TEST_PER_CATEGORY} {category} Test rows, found "
                f"{counts[(category, 'Test')]}."
            )
        for block in TEST_BLOCKS:
            block_count = sum(
                1
                for r in gui_records
                if r["category"] == category
                and r["partition"] == "Test"
                and int(r["block"]) == block
            )
            if block_count != TEST_PER_CATEGORY_BLOCK:
                raise DevelopmentReferenceError(
                    f"Expected {TEST_PER_CATEGORY_BLOCK} {category} Test rows in "
                    f"block {block}, found {block_count}."
                )
    unexpected_test_blocks = {
        int(r["block"])
        for r in gui_records
        if r["partition"] == "Test" and int(r["block"]) not in TEST_BLOCKS
    }
    if unexpected_test_blocks:
        raise DevelopmentReferenceError(
            f"Unexpected UEyes Test blocks: {sorted(unexpected_test_blocks)}."
        )


def _load_feature_rows(
    path: Path,
    *,
    filename_column: str,
    feature_keys: Sequence[str],
) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {filename_column, "category", "image_sha256", *feature_keys}
        if not required.issubset(reader.fieldnames or []):
            raise DevelopmentReferenceError(
                f"{path.name} is missing required columns: "
                f"{sorted(required - set(reader.fieldnames or []))}."
            )
        for raw in reader:
            filename = (raw.get(filename_column) or "").strip()
            category = (raw.get("category") or "").strip().lower()
            image_sha = (raw.get("image_sha256") or "").strip().lower()
            if not filename or filename in rows:
                raise DevelopmentReferenceError(
                    f"{path.name} contains an empty or duplicate filename {filename!r}."
                )
            if category not in AUTHORIZED_SET:
                raise DevelopmentReferenceError(
                    f"{path.name} contains unauthorized category {category!r}."
                )
            if len(image_sha) != 64 or any(c not in "0123456789abcdef" for c in image_sha):
                raise DevelopmentReferenceError(
                    f"{path.name} contains invalid image SHA-256 for {filename!r}."
                )
            row = {
                "filename": filename,
                "category": category,
                "image_sha256": image_sha,
            }
            for key in feature_keys:
                try:
                    value = float(raw[key])
                except (TypeError, ValueError) as exc:
                    raise DevelopmentReferenceError(
                        f"{path.name} contains invalid {key!r} for {filename!r}."
                    ) from exc
                if not math.isfinite(value):
                    raise DevelopmentReferenceError(
                        f"{path.name} contains non-finite {key!r} for {filename!r}."
                    )
                row[key] = value
            rows[filename] = row
    return rows


def validate_source_rows(
    split: Mapping[str, Mapping[str, object]],
    visual_rows: Mapping[str, Mapping[str, object]],
    saliency_rows: Mapping[str, Mapping[str, object]],
    images_dir: Path,
) -> None:
    expected = {
        name for name, r in split.items() if r["category"] in AUTHORIZED_SET
    }
    if set(visual_rows) != expected:
        missing = sorted(expected - set(visual_rows))[:3]
        extra = sorted(set(visual_rows) - expected)[:3]
        raise DevelopmentReferenceError(
            f"Visual row set mismatch (missing={missing}, extra={extra})."
        )
    if set(saliency_rows) != expected:
        missing = sorted(expected - set(saliency_rows))[:3]
        extra = sorted(set(saliency_rows) - expected)[:3]
        raise DevelopmentReferenceError(
            f"Saliency row set mismatch (missing={missing}, extra={extra})."
        )
    for filename in sorted(expected):
        split_category = str(split[filename]["category"])
        visual = visual_rows[filename]
        saliency = saliency_rows[filename]
        if visual["category"] != split_category or saliency["category"] != split_category:
            raise DevelopmentReferenceError(
                f"Category mismatch for {filename!r} across source artifacts."
            )
        if visual["image_sha256"] != saliency["image_sha256"]:
            raise DevelopmentReferenceError(
                f"Image SHA-256 mismatch for {filename!r} across source artifacts."
            )
        image_path = Path(images_dir) / filename
        if not image_path.is_file():
            raise DevelopmentReferenceError(f"UEyes image is missing: {filename!r}.")
        if sha256_file(image_path) != visual["image_sha256"]:
            raise DevelopmentReferenceError(
                f"On-disk UEyes image SHA-256 mismatch for {filename!r}."
            )


def aggregate_rows(
    rows: Iterable[Mapping[str, object]], feature_keys: Sequence[str]
) -> Dict[str, dict]:
    materialized = list(rows)
    result: Dict[str, dict] = {}
    for key in feature_keys:
        values = np.asarray([float(r[key]) for r in materialized], dtype=np.float64)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise DevelopmentReferenceError(
                f"Feature {key!r} does not contain at least two finite values."
            )
        result[key] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p5": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def _selection_key(seed: str, record: Mapping[str, object], image_sha: str) -> str:
    payload = "|".join(
        (
            seed,
            str(record["category"]),
            str(record["block"]),
            str(record["filename"]),
            image_sha,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manual_exclusions(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    exclusions: Dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"filename", "exclusion_reason"}
        if not required.issubset(reader.fieldnames or []):
            raise DevelopmentReferenceError(
                "Manual exclusions CSV requires filename and exclusion_reason."
            )
        for row in reader:
            filename = (row.get("filename") or "").strip()
            reason = (row.get("exclusion_reason") or "").strip()
            if not filename or not reason or filename in exclusions:
                raise DevelopmentReferenceError(
                    "Manual exclusions contain an empty or duplicate entry."
                )
            exclusions[filename] = reason
    return exclusions


def build_selection(
    split: Mapping[str, Mapping[str, object]],
    visual_rows: Mapping[str, Mapping[str, object]],
    *,
    seed: str,
    exclusions: Optional[Mapping[str, str]] = None,
) -> List[dict]:
    exclusions = dict(exclusions or {})
    test_names = {
        name
        for name, r in split.items()
        if r["category"] in AUTHORIZED_SET and r["partition"] == "Test"
    }
    unknown_exclusions = sorted(set(exclusions) - test_names)
    if unknown_exclusions:
        raise DevelopmentReferenceError(
            f"Manual exclusions are not official GUI Test images: {unknown_exclusions[:3]}."
        )

    output: List[dict] = []
    for category in AUTHORIZED_CATEGORIES:
        selected_category: List[dict] = []
        reserve_category: List[dict] = []
        excluded_category: List[dict] = []
        for block in TEST_BLOCKS:
            candidates = []
            for filename, record in split.items():
                if (
                    record["category"] == category
                    and record["partition"] == "Test"
                    and int(record["block"]) == block
                ):
                    image_sha = str(visual_rows[filename]["image_sha256"])
                    candidates.append(
                        {
                            **record,
                            "image_sha256": image_sha,
                            "selection_key_sha256": _selection_key(seed, record, image_sha),
                        }
                    )
            candidates.sort(key=lambda r: (r["selection_key_sha256"], r["filename"]))
            eligible = [r for r in candidates if r["filename"] not in exclusions]
            rejected = [r for r in candidates if r["filename"] in exclusions]
            quota = SELECTION_QUOTAS[category][block]
            if len(eligible) < quota:
                raise DevelopmentReferenceError(
                    f"Not enough eligible {category} images in block {block}: "
                    f"need {quota}, found {len(eligible)}."
                )
            for block_rank, record in enumerate(eligible, start=1):
                item = dict(record)
                item["block_rank"] = block_rank
                item["status"] = "selected_candidate" if block_rank <= quota else "reserve_candidate"
                item["exclusion_reason"] = ""
                if block_rank <= quota:
                    selected_category.append(item)
                else:
                    reserve_category.append(item)
            for record in rejected:
                item = dict(record)
                item["block_rank"] = ""
                item["status"] = "excluded"
                item["exclusion_reason"] = exclusions[item["filename"]]
                excluded_category.append(item)

        selected_category.sort(key=lambda r: (int(r["block"]), int(r["block_rank"])))
        reserve_category.sort(key=lambda r: (r["selection_key_sha256"], r["filename"]))
        if len(selected_category) != SELECTED_PER_CATEGORY:
            raise DevelopmentReferenceError(
                f"Selection produced {len(selected_category)} {category} candidates; "
                f"expected {SELECTED_PER_CATEGORY}."
            )
        for rank, item in enumerate(selected_category, start=1):
            item["selected_rank_within_category"] = rank
            item["reserve_rank_within_category"] = ""
        for rank, item in enumerate(reserve_category, start=1):
            item["selected_rank_within_category"] = ""
            item["reserve_rank_within_category"] = rank
        for item in excluded_category:
            item["selected_rank_within_category"] = ""
            item["reserve_rank_within_category"] = ""
        output.extend(selected_category + reserve_category + excluded_category)

    selected = [r for r in output if r["status"] == "selected_candidate"]
    block_counts = collections.Counter(int(r["block"]) for r in selected)
    if block_counts != collections.Counter({53: 20, 54: 20, 55: 20}):
        raise DevelopmentReferenceError(
            f"Selected block totals are not balanced: {dict(block_counts)}."
        )
    return output


def _population_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    lines = [
        f"{r['filename']}|{r['category']}|{r['image_sha256']}"
        for r in rows
    ]
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_selection_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "image_id",
        "filename",
        "category",
        "block",
        "official_split",
        "status",
        "selected_rank_within_category",
        "reserve_rank_within_category",
        "block_rank",
        "eligibility_status",
        "exclusion_reason",
        "selection_key_sha256",
        "image_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                AUTHORIZED_CATEGORIES.index(str(r["category"])),
                {"selected_candidate": 0, "reserve_candidate": 1, "excluded": 2}[str(r["status"])],
                int(r.get("selected_rank_within_category") or r.get("reserve_rank_within_category") or 999),
                str(r["filename"]),
            ),
        ):
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "filename": row["filename"],
                    "category": row["category"],
                    "block": row["block"],
                    "official_split": row["partition"],
                    "status": row["status"],
                    "selected_rank_within_category": row["selected_rank_within_category"],
                    "reserve_rank_within_category": row["reserve_rank_within_category"],
                    "block_rank": row["block_rank"],
                    "eligibility_status": (
                        "excluded" if row["status"] == "excluded"
                        else "technical_pass_manual_review_pending"
                    ),
                    "exclusion_reason": row["exclusion_reason"],
                    "selection_key_sha256": row["selection_key_sha256"],
                    "image_sha256": row["image_sha256"],
                }
            )
    os.replace(tmp, path)


def build_drift(old_features: Mapping[str, Mapping[str, object]], new_features: Mapping[str, Mapping[str, object]]) -> dict:
    if set(old_features) != set(new_features):
        raise DevelopmentReferenceError("Old and new feature sets differ.")
    details = {}
    for key in sorted(new_features):
        old = old_features[key]
        new = new_features[key]
        feature = {}
        for stat in ("mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95"):
            old_value = float(old[stat])
            new_value = float(new[stat])
            delta = new_value - old_value
            feature[stat] = {
                "old": old_value,
                "new": new_value,
                "absolute_delta": delta,
                "relative_delta_percent": (delta / old_value * 100.0) if old_value else None,
            }
        details[key] = feature
    return {
        "schema_version": "ueyes-development-reference-drift-v1",
        "comparison": "all-1485 reference versus official-Train-1404 reference",
        "old_n": 1485,
        "new_n": TRAIN_TOTAL,
        "features": details,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types-csv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--visual-rows", required=True, type=Path)
    parser.add_argument("--saliency-rows", required=True, type=Path)
    parser.add_argument("--current-feature-norms", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--manual-exclusions", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    split = load_official_split(args.types_csv)
    validate_official_split(split)
    visual_rows = _load_feature_rows(
        args.visual_rows,
        filename_column="filename",
        feature_keys=VISUAL_FEATURE_KEYS,
    )
    saliency_rows = _load_feature_rows(
        args.saliency_rows,
        filename_column="relative_path",
        feature_keys=SALIENCY_FEATURE_KEYS,
    )
    validate_source_rows(split, visual_rows, saliency_rows, args.images_dir)

    development_names = sorted(
        name
        for name, record in split.items()
        if record["category"] in AUTHORIZED_SET and record["partition"] == "Train"
    )
    test_names = sorted(
        name
        for name, record in split.items()
        if record["category"] in AUTHORIZED_SET and record["partition"] == "Test"
    )
    if len(development_names) != TRAIN_TOTAL or len(test_names) != TEST_TOTAL:
        raise DevelopmentReferenceError("Development/evaluation partition sizes are invalid.")

    visual_stats = aggregate_rows(
        (visual_rows[name] for name in development_names), VISUAL_FEATURE_KEYS
    )
    saliency_stats = aggregate_rows(
        (saliency_rows[name] for name in development_names), SALIENCY_FEATURE_KEYS
    )
    combined_stats = {**visual_stats, **saliency_stats}
    if any(int(stats["n"]) != TRAIN_TOTAL for stats in combined_stats.values()):
        raise DevelopmentReferenceError("A generated norm block has the wrong sample size.")

    selection = build_selection(
        split,
        visual_rows,
        seed=args.selection_seed,
        exclusions=load_manual_exclusions(args.manual_exclusions),
    )
    selected_count = collections.Counter(
        r["category"] for r in selection if r["status"] == "selected_candidate"
    )
    reserve_count = collections.Counter(
        r["category"] for r in selection if r["status"] == "reserve_candidate"
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "ueyes_stimulus_selection_v1.csv"
    _write_selection_csv(selection_path, selection)
    selection_sha = sha256_file(selection_path)

    split_sha = sha256_file(args.types_csv)
    visual_rows_sha = sha256_file(args.visual_rows)
    saliency_rows_sha = sha256_file(args.saliency_rows)
    development_population_rows = [
        {
            "filename": name,
            "category": split[name]["category"],
            "image_sha256": visual_rows[name]["image_sha256"],
        }
        for name in development_names
    ]
    test_population_rows = [
        {
            "filename": name,
            "category": split[name]["category"],
            "image_sha256": visual_rows[name]["image_sha256"],
        }
        for name in test_names
    ]
    development_population_sha = _population_sha256(development_population_rows)
    test_population_sha = _population_sha256(test_population_rows)
    common = {
        "official_split_csv_sha256": split_sha,
        "reference_partition": "Train",
        "evaluation_partition": "Test",
        "reference_category_counts": {c: TRAIN_PER_CATEGORY for c in AUTHORIZED_CATEGORIES},
        "reference_total": TRAIN_TOTAL,
        "evaluation_category_counts": {c: TEST_PER_CATEGORY for c in AUTHORIZED_CATEGORIES},
        "evaluation_total": TEST_TOTAL,
        "excluded_categories": ["poster"],
        "development_population_sha256": development_population_sha,
        "heldout_test_population_sha256": test_population_sha,
        "standard_deviation": {"convention": "sample standard deviation", "ddof": 1},
        "stored_quantiles": {
            "implementation": "numpy.percentile",
            "numpy_version": np.__version__,
            "method": "linear",
            "probabilities_percent": [5, 25, 50, 75, 95],
        },
    }

    visual_artifact = {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "analysis_domain": "canonical",
        "canonical_long_side": 1280,
        "canonical_analysis_version": "canonical-analysis-v1.1:long1280:both-dims-min16:area-down/linear-up",
        "feature_list": list(VISUAL_FEATURE_KEYS),
        "includes_saliency": False,
        "num_images": TRAIN_TOTAL,
        **common,
        "source_rows": {
            "filename": args.visual_rows.name,
            "sha256": visual_rows_sha,
            "row_count": FULL_TOTAL,
        },
        "features": visual_stats,
    }
    visual_path = out / "canonical_visual_feature_norms.json"
    _write_json(visual_path, visual_artifact)

    saliency_artifact = {
        "schema_version": SALIENCY_SCHEMA_VERSION,
        "analysis_domain": "native_input",
        "resolution_policy": "NATIVE_INPUT",
        "feature_order": list(SALIENCY_FEATURE_KEYS),
        "num_images": TRAIN_TOTAL,
        **common,
        "source_rows": {
            "filename": args.saliency_rows.name,
            "sha256": saliency_rows_sha,
            "row_count": FULL_TOTAL,
        },
        "features": saliency_stats,
    }
    saliency_path = out / "development_saliency_feature_norms.json"
    _write_json(saliency_path, saliency_artifact)

    combined_artifact = {
        "features": combined_stats,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "source": "UEyes dataset (Jiang et al., CHI 2023; Zenodo 8010312)",
            "categories_included": list(AUTHORIZED_CATEGORIES),
            "categories_excluded": ["poster"],
            "reference_partition": "Train",
            "evaluation_partition_excluded": "Test",
            "category_counts": {c: TRAIN_PER_CATEGORY for c in AUTHORIZED_CATEGORIES},
            "num_images": TRAIN_TOTAL,
            "includes_saliency": True,
            "is_subsample": False,
            "reference_population_note": (
                "Development-reference distributions computed exclusively from the "
                "official UEyes Train partition. The complete official Test partition "
                "is excluded from all reference statistics and reserved for evaluation."
            ),
            "official_split_csv_sha256": split_sha,
            "development_population_sha256": development_population_sha,
            "heldout_test_population_sha256": test_population_sha,
            "visual_norms_provenance": {
                "source_artifact_filename": visual_path.name,
                "source_artifact_sha256": sha256_file(visual_path),
                "source_csv_filename": args.visual_rows.name,
                "source_csv_sha256": visual_rows_sha,
                "analysis_domain": "canonical",
                "canonical_long_side": 1280,
                "processed_count": TRAIN_TOTAL,
                "processed_category_counts": {c: TRAIN_PER_CATEGORY for c in AUTHORIZED_CATEGORIES},
            },
            "saliency_norms_provenance": {
                "source_artifact_filename": saliency_path.name,
                "source_artifact_sha256": sha256_file(saliency_path),
                "source_csv_filename": args.saliency_rows.name,
                "source_csv_sha256": saliency_rows_sha,
                "analysis_domain": "native_input",
                "resolution_policy": "NATIVE_INPUT",
                "processed_count": TRAIN_TOTAL,
                "processed_category_counts": {c: TRAIN_PER_CATEGORY for c in AUTHORIZED_CATEGORIES},
            },
            "stimulus_selection_provenance": {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "selection_manifest_filename": selection_path.name,
                "selection_manifest_sha256": selection_sha,
                "selection_seed": args.selection_seed,
                "selected_category_counts": dict(selected_count),
                "reserve_category_counts": dict(reserve_count),
                "manual_eligibility_review": "pending",
            },
        },
    }
    feature_norms_path = out / "feature_norms.json"
    _write_json(feature_norms_path, combined_artifact)

    with args.current_feature_norms.open(encoding="utf-8") as source:
        old_feature_norms = json.load(source)
    drift = build_drift(old_feature_norms.get("features", {}), combined_stats)
    drift_path = out / "development_reference_norm_drift.json"
    _write_json(drift_path, drift)

    outputs = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (
            visual_path,
            saliency_path,
            feature_norms_path,
            selection_path,
            drift_path,
        )
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "These are corpus-relative development references, not universal GUI "
            "norms or human cognitive-load measurements. The selected Test images "
            "are held-out UEyes candidates, not an external dataset."
        ),
        **common,
        "selection": {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "seed": args.selection_seed,
            "policy": "SHA-256 rank within category and official Test block",
            "block_quotas": SELECTION_QUOTAS,
            "selected_category_counts": dict(selected_count),
            "reserve_category_counts": dict(reserve_count),
            "manual_eligibility_review": "pending",
        },
        "inputs": {
            "official_split_csv": {"sha256": split_sha, "bytes": args.types_csv.stat().st_size},
            "visual_rows_csv": {"sha256": visual_rows_sha, "bytes": args.visual_rows.stat().st_size},
            "saliency_rows_csv": {"sha256": saliency_rows_sha, "bytes": args.saliency_rows.stat().st_size},
            "current_feature_norms": {
                "sha256": sha256_file(args.current_feature_norms),
                "bytes": args.current_feature_norms.stat().st_size,
            },
        },
        "outputs": outputs,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    manifest_path = out / "development_reference_manifest.json"
    _write_json(manifest_path, manifest)

    print(f"PASS: official Train reference = {TRAIN_TOTAL} images (468/category)")
    print(f"PASS: held-out official Test set = {TEST_TOTAL} images (27/category)")
    print("PASS: deterministic candidates = 60 (20/category); reserves = 21 (7/category)")
    print(f"Wrote outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
