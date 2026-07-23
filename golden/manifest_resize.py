"""Phase 0 — freeze the resize causality test protocol BEFORE any prediction.

Writes resize_causality_manifest.json capturing: the audit base + experiment
commit, both container images + immutable digests, both target environments,
the pinned upstream reference, the checkpoint expectation, the five fixtures,
the identified decoder resize layers with source/target dimensions, the candidate
legacy resize operator, all conditions/ablations, the windowed-SSIM parameters,
the metrics and the UNCHANGED pre-registered thresholds + decision rules.

Thresholds live in common.THRESHOLDS (committed source) so they are fixed before
the workflow runs. This manifest records them with runtime provenance and MUST
NOT be altered after results are known (only results are appended downstream).
"""
import os
import json
import time
import argparse

import common
import resize_common as RC
from windowed_ssim import WINDOWED_SSIM_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resize_causality_manifest.json")
    args = ap.parse_args()

    manifest = {
        "phase": "0_frozen_protocol",
        "experiment": "umsi_resize_causality",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_base_commit": "77241fc6e635949dada502e6c2d902eac1e8b030",
        "audit_experiment_branch_ref": os.environ.get("GITHUB_REF", ""),
        "audit_experiment_commit": os.environ.get("GITHUB_SHA", ""),
        "for_commit": "6a17288bbac848bb023fccc98aee606c034ea54b",
        "prior_golden_run": "30028284848",
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
        "upstream": {"repo": common.UPSTREAM_REPO, "commit": common.UPSTREAM_COMMIT,
                     "model_builder": "singleduration_models.py::UMSI",
                     "preprocessing": "sal_imp_utilities.py"},
        "checkpoint": {
            "path": "saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5",
            "sha256": common.CHECKPOINT_SHA256,
            "byte_size": common.CHECKPOINT_BYTES},
        "fixtures": {
            name: {"dims_hw": [common.FIXTURES[name]["h"],
                              common.FIXTURES[name]["w"]],
                   "required_sha256": common.FIXTURES[name]["sha256"]}
            for name in common.FIXTURE_ORDER},
        "decoder_resize_layers": RC.DECODER_RESIZE,
        "candidate_legacy_resize_operator": (
            "tf.raw_ops.ResizeBilinear(align_corners=False, "
            "half_pixel_centers=False)  # align_corners=True is NOT used"),
        "modern_resize_operator": (
            "keras UpSampling2D(interpolation='bilinear')  # TF2/Keras3 "
            "half-pixel resize semantics"),
        "oracle_acceptance_max_abs_diff": RC.ORACLE_MAX_ABS_DIFF,
        "conditions": RC.CONDITIONS,
        "cumulative_ablations": RC.ABLATION_SETS,
        "byte_identical_input_control": (
            "legacy preprocessing tensor saved as uncompressed float32 .npy, "
            "loaded in the modern job with allow_pickle=False; controlled tensor "
            "difference must be exactly 0.0. The genuine modern preprocessing "
            "result is preserved separately (its ~e-6 difference is NOT hidden)."),
        "metrics": ["shape", "dtype", "min", "max", "mean", "std",
                    "nonfinite_pct", "sha256", "mae", "rmse", "max_abs_diff",
                    "pearson", "spearman", "legacy_audit_global_ssim",
                    "windowed_ssim"],
        "windowed_ssim_params": WINDOWED_SSIM_PARAMS,
        "pre_registered_thresholds": common.THRESHOLDS,
        "thresholds_note": ("Identical to the frozen golden thresholds; BOTH "
                            "reported SSIM values (legacy_audit_global_ssim and "
                            "windowed_ssim) must be >= ssim_min."),
        "continuous_features": common.CONTINUOUS_FEATURES,
        "feature_keys": common.FEAT_KEYS,
        "hceye_keys": common.HCE_KEYS,
        "decision_rules": {
            "CONFIRMED": ("oracle passes AND M_LC still reproduces a meaningful "
                          "parity failure AND M_LL passes every unchanged golden "
                          "threshold for all five fixtures AND the activation "
                          "comparison shows the resize replacement removes the "
                          "divergence from the first affected resize layer"),
            "PARTIAL": ("legacy resize substantially improves results but M_LL "
                        "still fails any threshold"),
            "NOT_CONFIRMED": ("controlled results do not causally support the "
                              "resize hypothesis"),
            "BLOCKED": ("the authentic experiment cannot be executed / a required "
                        "runtime or artifact is unavailable"),
        },
        "provisional_note": ("All HCEye and cognitive_load_index values are "
                             "PROVISIONAL: the saliency norm distributions are "
                             "stale."),
        "forbidden": ("No production edit; no threshold change after results; no "
                      "rotate/shift/align of maps; no norms rebuild; no parity "
                      "claim from source inspection alone."),
        "scope_disclaimer": ("An x86_64 CPU comparison is a source-matched legacy "
                             "numerical golden run, NOT a claim of bitwise CUDA 9 "
                             "reproduction."),
    }
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("FROZEN PROTOCOL written to %s" % args.out)
    print(json.dumps(manifest["pre_registered_thresholds"], indent=2))
    print(json.dumps(manifest["decision_rules"], indent=2))


if __name__ == "__main__":
    main()
