#!/usr/bin/env python3
"""Run the real Stage-1 API for the frozen 75-stimulus study pool.

The export is fail-closed: it writes no final output unless every packaged
image, model identity, runtime identity, response schema, feature count and
numeric value validates. Saliency overlays and non-score-bearing element boxes
are intentionally not retained in the analysis artifact.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "stage1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE1))

STIMULUS_ZIP_SHA256 = (
    "6bbf0fcfdb32e00181a8a7c05a3c84a7fb95e018a1344e2e27ff6d48b8e6c154"
)
STIMULUS_MANIFEST_SHA256 = (
    "e77faface786f339d28bf9683cf4b6b0981ce4abe3c7787f0bf48d40c21628e8"
)
FEATURE_NORMS_SHA256 = (
    "2e9861f04ea80ad03ac14d0a8f5c61522f8150b814dcb31de54a36042f340e3d"
)
REFERENCE_PACK_SHA256 = (
    "00537e81365bde48e830e111de74bab16e6c54fccbfaf5dc7ebcfb6641f40581"
)
RUNTIME_MANIFEST_SHA256 = (
    "846e84a3cd35dedd054e05df49fba1081114ff04a66f328a9b3bfb744f5abe02"
)
REFERENCE_POPULATION_SHA256 = (
    "9ef889847f63be8fa2a41653e3a06d4aa992384594d814c57ca1a8f82000b923"
)
EXPECTED_COUNTS = {
    "total": 75,
    "main": 72,
    "practice": 3,
    "desktop": 25,
    "mobile": 25,
    "web": 25,
}

PIPELINE_SOURCE_FILES = (
    "stage1/app.py",
    "stage1/visual_complexity.py",
    "stage1/canonical_layout.py",
    "stage1/reproducibility.py",
    "stage1/data/results/feature_norms.json",
    "stage1/data/results/canonical_visual_feature_norms.json",
    "stage1/data/results/development_saliency_feature_norms.json",
    "stage1/data/results/study_isolated_reference_manifest.json",
    "stage1/reference_pack_manifest.json",
    "stage1/reference_pack_manifest.sha256",
    "stage1/runtime_environment_manifest.json",
    "stage1/runtime_environment_manifest.sha256",
    "stage1/environment/requirements-macos-arm64-python3.9.lock.txt",
    "stage1/tools/export_study_scores.py",
    "saliency/umsi_model.py",
    "saliency/postprocessing.py",
    "saliency/saliency_features.py",
    "saliency/umsi_checkpoint_identity.json",
    "cognitive/element_detector.py",
    "cognitive/text_reader.py",
    "cognitive/easyocr_identity.py",
    "cognitive/easyocr_model_identity.json",
    "hceye/hceye_features.py",
    "hceye/hceye_coefficient_provenance.json",
)


class ExportError(RuntimeError):
    """Raised when an export gate fails."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def require_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise ExportError(f"{message}: expected {expected!r}, found {actual!r}")


def require_finite_tree(value, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ExportError(f"Non-finite value at {path}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"Non-text mapping key at {path}")
            require_finite_tree(child, f"{path}.{key}")
        return
    raise ExportError(f"Unsupported value type at {path}: {type(value).__name__}")


def canonical_digest(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_split_rows(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    require_equal(len(rows), 75, "Study split row count")
    by_id = {row["stimulus_id"]: row for row in rows}
    require_equal(len(by_id), 75, "Unique study split IDs")
    return by_id


def load_stimulus_manifest(archive: zipfile.ZipFile) -> tuple[str, dict, bytes]:
    candidates = [
        name
        for name in archive.namelist()
        if name.endswith("/04_documentation/final_pool_manifest.json")
        and not name.startswith("__MACOSX/")
    ]
    require_equal(len(candidates), 1, "Final pool manifest count")
    name = candidates[0]
    raw = archive.read(name)
    require_equal(
        sha256_bytes(raw), STIMULUS_MANIFEST_SHA256, "Final pool manifest SHA-256"
    )
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise ExportError("Final pool manifest is invalid JSON") from exc
    require_equal(manifest.get("schema_version"), "final-stimulus-pool-v2", "Manifest schema")
    counts = manifest.get("counts", {})
    for key in ("total", "main", "practice"):
        require_equal(counts.get(key), EXPECTED_COUNTS[key], f"Manifest {key} count")
    require_equal(counts.get("by_platform"), {
        "mobile": 25,
        "web": 25,
        "desktop": 25,
    }, "Manifest platform counts")
    return name, manifest, raw


def pipeline_source_hashes() -> dict:
    hashes = {}
    for relative in PIPELINE_SOURCE_FILES:
        path = ROOT / relative
        require(path.is_file(), f"Required pipeline source file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def validate_response(body: Mapping[str, object], reproducibility: Mapping[str, object]) -> None:
    require_equal(body.get("vector_dimensions"), 19, "Stage-1 vector dimensions")
    names = body.get("stage1_feature_names")
    vector = body.get("stage1_feature_vector")
    visual = body.get("visual_features")
    saliency = body.get("saliency_features")
    hceye = body.get("hceye_proxy_features")
    require(isinstance(names, list), "Stage-1 feature names are missing")
    require(isinstance(vector, list), "Stage-1 feature vector is missing")
    require(isinstance(visual, dict), "Visual feature block is missing")
    require(isinstance(saliency, dict), "Saliency feature block is missing")
    require(isinstance(hceye, dict), "HCEye feature block is missing")
    require_equal(len(names), 19, "Stage-1 feature-name count")
    require_equal(len(vector), 19, "Stage-1 feature-value count")
    require_equal(len(set(names)), 19, "Unique Stage-1 feature-name count")
    require_equal(len(visual), 8, "Visual feature count")
    require_equal(len(saliency), 5, "Saliency feature count")
    require_equal(len(hceye), 6, "HCEye feature count")
    combined = {**visual, **saliency, **hceye}
    require_equal(len(combined), 19, "Declared x19 block size")
    require_equal(set(combined), set(names), "Named x19 feature set")
    declared_values = [combined[name] for name in names]
    for observed, declared, name in zip(vector, declared_values, names):
        expected_float32 = struct.unpack(
            "!f", struct.pack("!f", float(declared))
        )[0]
        require_equal(
            float(observed), expected_float32,
            f"x19 value disagrees with float32-cast named feature {name}",
        )
    layout = body.get("layout")
    require(isinstance(layout, dict), "Layout block is missing")
    layout_index = layout.get("experimental_complexity_index")
    require(isinstance(layout_index, (int, float)), "Layout index is missing")
    require(
        math.isclose(float(layout_index), float(declared_values[-1]) * 100.0,
                     rel_tol=0.0, abs_tol=1e-5),
        "Layout index disagrees with h6[5]",
    )
    reference_meta = body.get("reference_meta")
    require(isinstance(reference_meta, dict), "Reference metadata is missing")
    require_equal(reference_meta.get("num_images"), 1380, "Reference image count")
    require_equal(
        reference_meta.get("development_population_sha256"),
        REFERENCE_POPULATION_SHA256,
        "Reference population SHA-256",
    )
    reference = body.get("reference")
    require(isinstance(reference, dict), "Reference comparisons are missing")
    require_equal(len(reference), 13, "Reference comparison count")
    require(
        all(item.get("n") == 1380 for item in reference.values()),
        "Reference comparison counts are not uniformly 1,380",
    )
    response_repro = body.get("reproducibility")
    require_equal(response_repro, reproducibility, "Per-screen reproducibility identity")
    require_finite_tree({
        "visual_features": visual,
        "saliency_features": saliency,
        "hceye_proxy_features": hceye,
        "stage1_feature_vector": vector,
        "layout": layout,
        "reference": reference,
        "hceye_inputs": body.get("hceye_inputs"),
    }, "score_payload")


def build_csv(records: Sequence[Mapping[str, object]]) -> bytes:
    feature_names = list(records[0]["analysis"]["stage1_feature_names"])
    reference_features = list(records[0]["analysis"]["reference"])
    identity_columns = [
        "stimulus_order", "stimulus_id", "role", "platform", "package_path",
        "original_ueyes_filename", "original_split", "original_block",
        "image_sha256", "task_prompt_en", "text_density_source",
        "whitespace_ratio", "text_density", "layout_index_0_100",
    ]
    reference_columns = []
    for feature in reference_features:
        reference_columns.extend((
            f"reference_percentile__{feature}",
            f"reference_z__{feature}",
            f"reference_delta_pct__{feature}",
        ))
    provenance_columns = [
        "source_commit", "source_tree_state", "stage1_vector_schema",
        "feature_norms_sha256", "reference_pack_manifest_sha256",
        "runtime_environment_manifest_sha256", "umsi_checkpoint_sha256",
        "easyocr_detector_sha256", "easyocr_recognizer_sha256",
        "reference_population_sha256", "reference_num_images",
    ]
    columns = identity_columns + feature_names + reference_columns + provenance_columns
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        analysis = record["analysis"]
        repro = record["reproducibility"]
        hceye_inputs = analysis["hceye_inputs"]
        row = {
            key: record.get(key)
            for key in (
                "stimulus_order", "stimulus_id", "role", "platform", "package_path",
                "original_ueyes_filename", "original_split", "original_block",
                "image_sha256", "task_prompt_en",
            )
        }
        row.update({
            "text_density_source": hceye_inputs["text_density_source"],
            "whitespace_ratio": hceye_inputs["whitespace_ratio"],
            "text_density": hceye_inputs["text_density"],
            "layout_index_0_100": analysis["layout"]["experimental_complexity_index"],
        })
        row.update(zip(feature_names, analysis["stage1_feature_vector"]))
        for feature, comparison in analysis["reference"].items():
            row[f"reference_percentile__{feature}"] = comparison["percentile"]
            row[f"reference_z__{feature}"] = comparison["z"]
            row[f"reference_delta_pct__{feature}"] = comparison["delta_pct"]
        row.update({
            "source_commit": repro["source"]["commit"],
            "source_tree_state": repro["source"]["tree_state"],
            "stage1_vector_schema": repro["stage1_vector_schema"],
            "feature_norms_sha256": repro["feature_norms_sha256"],
            "reference_pack_manifest_sha256": repro["reference_pack_manifest_sha256"],
            "runtime_environment_manifest_sha256": repro["runtime_environment_manifest_sha256"],
            "umsi_checkpoint_sha256": repro["umsi_checkpoint_sha256"],
            "easyocr_detector_sha256": repro["easyocr_detector_sha256"],
            "easyocr_recognizer_sha256": repro["easyocr_recognizer_sha256"],
            "reference_population_sha256": REFERENCE_POPULATION_SHA256,
            "reference_num_images": 1380,
        })
        writer.writerow(row)
    return target.getvalue().encode("utf-8")


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def run(stimuli_zip: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        require(not any(output_dir.iterdir()), f"Output directory is not empty: {output_dir}")
    require(stimuli_zip.is_file(), f"Stimulus ZIP is missing: {stimuli_zip}")
    require_equal(sha256_file(stimuli_zip), STIMULUS_ZIP_SHA256, "Stimulus ZIP SHA-256")

    os.environ.pop("STAGE1_RUNTIME_VERIFICATION", None)
    import app as app_module
    import reproducibility
    from cognitive.easyocr_identity import verify_easyocr_models
    from saliency.checkpoint_identity import verify_umsi_checkpoint

    runtime = reproducibility.verify_runtime_environment()
    require_equal(runtime.get("status"), "verified", "Strict runtime status")
    require_equal(runtime.get("mode"), "strict", "Runtime verification mode")
    reproducibility_meta = reproducibility.study_reproducibility_metadata()
    require_equal(
        reproducibility_meta["feature_norms_sha256"],
        FEATURE_NORMS_SHA256,
        "Active feature norms SHA-256",
    )
    require_equal(
        reproducibility_meta["reference_pack_manifest_sha256"],
        REFERENCE_PACK_SHA256,
        "Active reference pack SHA-256",
    )
    require_equal(
        reproducibility_meta["runtime_environment_manifest_sha256"],
        RUNTIME_MANIFEST_SHA256,
        "Active runtime manifest SHA-256",
    )
    checkpoint_path = (
        ROOT / "saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5"
    )
    checkpoint = verify_umsi_checkpoint(checkpoint_path)
    easyocr = verify_easyocr_models()
    split_by_id = load_split_rows(
        ROOT / "stage1/data/results/study_stimulus_split_75.csv"
    )
    source_hashes = pipeline_source_hashes()

    started = time.monotonic()
    records = []
    with zipfile.ZipFile(stimuli_zip) as archive, tempfile.TemporaryDirectory(
        prefix="vicop-score-uploads-"
    ) as upload_root:
        manifest_name, manifest, _manifest_raw = load_stimulus_manifest(archive)
        stimuli = manifest.get("stimuli")
        require(isinstance(stimuli, list), "Final pool stimuli are missing")
        require_equal(len(stimuli), 75, "Final pool stimulus count")
        require_equal(len({item.get("stimulus_id") for item in stimuli}), 75,
                      "Unique final pool stimulus IDs")
        app_module.UPLOAD_DIR = Path(upload_root)
        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()

        for index, stimulus in enumerate(stimuli, start=1):
            stimulus_id = stimulus.get("stimulus_id")
            role = stimulus.get("role")
            platform_name = stimulus.get("platform")
            package_path = stimulus.get("package_path")
            require(stimulus_id in split_by_id, f"Unknown stimulus ID: {stimulus_id}")
            split = split_by_id[stimulus_id]
            require_equal(role, split["role"], f"Role for {stimulus_id}")
            require_equal(platform_name, split["category"], f"Platform for {stimulus_id}")
            require_equal(package_path, split["packaged_path"], f"Package path for {stimulus_id}")
            require_equal(stimulus.get("source_sha256"), stimulus.get("packaged_sha256"),
                          f"Source/package identity for {stimulus_id}")
            archive_path = manifest_name.rsplit("/04_documentation/", 1)[0] + "/" + package_path
            try:
                image_bytes = archive.read(archive_path)
            except KeyError as exc:
                raise ExportError(f"Missing packaged image for {stimulus_id}") from exc
            image_sha = sha256_bytes(image_bytes)
            require_equal(image_sha, stimulus.get("packaged_sha256"),
                          f"Packaged image SHA-256 for {stimulus_id}")
            require_equal(image_sha, split["image_sha256"],
                          f"Study split image SHA-256 for {stimulus_id}")

            filename = Path(package_path).name
            response = client.post(
                "/api/cognitive-load",
                data={"image": (io.BytesIO(image_bytes), filename)},
                content_type="multipart/form-data",
            )
            body = response.get_json(silent=True)
            if response.status_code != 200 or not isinstance(body, dict):
                error = body.get("error") if isinstance(body, dict) else None
                raise ExportError(
                    f"API failure for {stimulus_id}: HTTP {response.status_code}, error={error!r}"
                )
            validate_response(body, reproducibility_meta)
            analysis = {
                "stage1_feature_names": body["stage1_feature_names"],
                "stage1_feature_vector": body["stage1_feature_vector"],
                "stage1_vector_dtype": body["stage1_vector_dtype"],
                "vector_dimensions": body["vector_dimensions"],
                "visual_features": body["visual_features"],
                "saliency_features": body["saliency_features"],
                "hceye_proxy_features": body["hceye_proxy_features"],
                "hceye_inputs": body["hceye_inputs"],
                "layout": body["layout"],
                "reference": body["reference"],
            }
            records.append({
                "stimulus_order": index,
                "stimulus_id": stimulus_id,
                "role": role,
                "platform": platform_name,
                "package_path": package_path,
                "original_ueyes_filename": split["original_ueyes_filename"],
                "original_split": split["original_split"],
                "original_block": split["original_block"],
                "image_sha256": image_sha,
                "task_prompt_en": stimulus.get("task_prompt_en", ""),
                "analysis": analysis,
                "reproducibility": reproducibility_meta,
            })
            layout_index = analysis["layout"]["experimental_complexity_index"]
            print(
                f"[{index:02d}/75] PASS {stimulus_id} ({platform_name}, {role}) "
                f"layout={layout_index:.6f}",
                flush=True,
            )

    role_counts = collections.Counter(record["role"] for record in records)
    platform_counts = collections.Counter(record["platform"] for record in records)
    require_equal(role_counts, collections.Counter({"main": 72, "practice": 3}),
                  "Export role counts")
    require_equal(platform_counts,
                  collections.Counter({"desktop": 25, "mobile": 25, "web": 25}),
                  "Export platform counts")
    require_equal(len({record["image_sha256"] for record in records}), 75,
                  "Unique exported image hashes")
    score_rows_sha256 = canonical_digest(records)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "vicop-study-scores-75-v1",
        "status": "PASS",
        "created_at_utc": created_at,
        "api_endpoint": "/api/cognitive-load",
        "execution_mode": "real UMSI++ + real EasyOCR + strict runtime verification",
        "claim_boundary": (
            "Project-specific v8+s5+h6=x19 descriptors and experimental layout "
            "index; not a validated human cognitive-load measure."
        ),
        "counts": {
            "total": len(records),
            "main": role_counts["main"],
            "practice": role_counts["practice"],
            "by_platform": dict(sorted(platform_counts.items())),
        },
        "input_identity": {
            "stimulus_zip": str(stimuli_zip),
            "stimulus_zip_sha256": STIMULUS_ZIP_SHA256,
            "stimulus_manifest_sha256": STIMULUS_MANIFEST_SHA256,
            "study_split_csv_sha256": sha256_file(
                ROOT / "stage1/data/results/study_stimulus_split_75.csv"
            ),
            "reference_population_sha256": REFERENCE_POPULATION_SHA256,
            "reference_num_images": 1380,
        },
        "model_identity": {
            "umsi_checkpoint": checkpoint,
            "easyocr": {
                "version": easyocr["identity"]["easyocr_version"],
                "artifacts": easyocr["artifacts"],
            },
        },
        "reproducibility": reproducibility_meta,
        "pipeline_source_sha256": source_hashes,
        "score_rows_sha256": score_rows_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "records": records,
    }
    require_finite_tree(payload, "export")
    json_bytes = (json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")
    csv_bytes = build_csv(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vicop_study_scores_75.csv"
    json_path = output_dir / "vicop_study_scores_75.json"
    write_atomic(csv_path, csv_bytes)
    write_atomic(json_path, json_bytes)
    manifest_payload = {
        "schema_version": "vicop-study-score-export-manifest-v1",
        "status": "PASS",
        "created_at_utc": created_at,
        "score_rows_sha256": score_rows_sha256,
        "counts": payload["counts"],
        "inputs": payload["input_identity"],
        "reproducibility": reproducibility_meta,
        "outputs": {
            csv_path.name: {"bytes": len(csv_bytes), "sha256": sha256_bytes(csv_bytes)},
            json_path.name: {"bytes": len(json_bytes), "sha256": sha256_bytes(json_bytes)},
        },
    }
    manifest_bytes = (json.dumps(
        manifest_payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")
    export_manifest_path = output_dir / "vicop_study_scores_75_manifest.json"
    write_atomic(export_manifest_path, manifest_bytes)
    checksums = {
        csv_path.name: sha256_file(csv_path),
        json_path.name: sha256_file(json_path),
        export_manifest_path.name: sha256_file(export_manifest_path),
    }
    checksum_bytes = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
    ).encode("ascii")
    write_atomic(output_dir / "SHA256SUMS", checksum_bytes)
    print(f"PASS: wrote 75 validated study scores to {output_dir}", flush=True)
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.stimuli_zip.resolve(), args.output_dir.resolve())
    except (ExportError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"FAIL CLOSED: {exc}")


if __name__ == "__main__":
    main()
