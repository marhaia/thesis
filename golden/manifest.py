"""Phase 0 — freeze the comparison protocol BEFORE any prediction runs.

Writes golden_protocol_manifest.json capturing: the audit branch/commit, the
container images + immutable digests (from the workflow), the pinned upstream
commit, the checkpoint expectation, the five fixtures (name/dims/required SHA),
and the pre-registered comparison stages, metrics and acceptance thresholds.

The thresholds themselves live in common.THRESHOLDS (committed source), so they
are fixed before the workflow ever runs; this manifest simply records them
alongside the runtime provenance as a timestamped, machine-readable artifact.
"""
import os
import json
import time
import argparse

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="golden_protocol_manifest.json")
    args = ap.parse_args()

    manifest = {
        "phase": "0_frozen_protocol",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_branch_ref": os.environ.get("GITHUB_REF", ""),
        "audit_commit": os.environ.get("GITHUB_SHA", ""),
        "for_commit": "6a17288bbac848bb023fccc98aee606c034ea54b",
        "workflow_run": {
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "server_url": os.environ.get("GITHUB_SERVER_URL", ""),
        },
        "containers": {
            "legacy_image": os.environ.get("LEGACY_IMAGE", ""),
            "legacy_image_digest": os.environ.get("LEGACY_IMAGE_DIGEST", ""),
            "modern_image": os.environ.get("MODERN_IMAGE", ""),
            "modern_image_digest": os.environ.get("MODERN_IMAGE_DIGEST", ""),
        },
        "target_legacy_environment": {
            "python": "3.7.9", "tensorflow": "1.14.0", "keras": "2.3.1",
            "numpy": "1.19.2", "arch": "x86_64", "device": "CPU",
        },
        "target_modern_environment": {
            "python": "3.9.6", "tensorflow": "2.16.2", "keras": "3.10.0",
            "numpy": "1.26.4", "arch": "x86_64", "device": "CPU",
        },
        "upstream": {
            "repo": common.UPSTREAM_REPO,
            "commit": common.UPSTREAM_COMMIT,
            "model_builder": "singleduration_models.py::UMSI",
            "preprocessing": "sal_imp_utilities.py",
        },
        "checkpoint": {
            "path": "saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5",
            "sha256": common.CHECKPOINT_SHA256,
            "byte_size": common.CHECKPOINT_BYTES,
        },
        "fixtures": {
            name: {
                "dims_hw": [common.FIXTURES[name]["h"], common.FIXTURES[name]["w"]],
                "required_sha256": common.FIXTURES[name]["sha256"],
            } for name in common.FIXTURE_ORDER
        },
        "comparison_stages": [
            "preprocessing_tensor (shape/dtype/range/sha256/max_abs_diff; expect 0.0)",
            "raw_saliency_tensor (512x512, pre-resize/normalise) stats + metrics",
            "shared_postprocess (squeeze->identical resize->identical min-max[0,1]) "
            "applied to BOTH raw tensors; all metrics + thresholds evaluated here",
            "genuine end-to-end path of each implementation, compared separately",
            "5 saliency features + 6 HCEye outputs (PROVISIONAL) on shared maps",
            "same-process (3) and fresh-process (3) repeatability per environment",
        ],
        "metrics": ["shape", "dtype", "min", "max", "mean", "std",
                    "nonfinite_pct", "sha256", "mae", "rmse", "max_abs_diff",
                    "pearson", "spearman", "ssim"],
        "pre_registered_thresholds": common.THRESHOLDS,
        "continuous_features": common.CONTINUOUS_FEATURES,
        "feature_keys": common.FEAT_KEYS,
        "hceye_keys": common.HCE_KEYS,
        "provisional_note": ("All HCEye and cognitive_load_index values are "
                             "PROVISIONAL: the saliency norm distributions are "
                             "stale. Used only to measure downstream numerical "
                             "impact, not to validate the score scientifically."),
        "scope_disclaimer": ("An x86_64 CPU comparison is a source-matched "
                             "legacy numerical golden run, NOT a claim of "
                             "bitwise CUDA 9 reproduction."),
    }
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("FROZEN PROTOCOL written to %s" % args.out)
    print(json.dumps(manifest["pre_registered_thresholds"], indent=2))


if __name__ == "__main__":
    main()
