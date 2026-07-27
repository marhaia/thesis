"""Phase 0 — pre-register the machine-readable validation protocol for the
UMSI++ legacy-resize PRODUCTION fix, BEFORE the substantive Golden workflow runs.

This freezes: the production base commit, the audit source-of-truth commit whose
verified operator this fix reproduces, the exact three decoder call sites, the
single allowed changed production file (diff allowlist), the checkpoint and
fixture hashes, the container digests, every validation stage, the frozen
metrics/formulas/thresholds (imported from committed ``common.THRESHOLDS`` so
they cannot move after results are seen), the repeatability and downstream
requirements, the scope/diff rules, and the mechanical verdict logic.

Writes ``umsi_legacy_resize_fix_protocol.json``. Committed together with the
production fix, the operator oracle, the structural call-site check, the focused
unit tests and the final validator BEFORE any Golden prediction is run.
"""
import os
import json
import time
import argparse

import common


# The single production file this milestone is permitted to change.
DIFF_ALLOWLIST = ["saliency/umsi_model.py"]

# The three decoder resize call sites, mapped to the audit boundaries whose
# TF1.14-equivalent operator was verified (max_abs_diff = 0.0) by the resize
# causality experiment.
CALL_SITES = [
    {"name": "dec_ups1", "size": [2, 2],
     "in_hw_c": [32, 32, 256], "out_hw": [64, 64]},
    {"name": "dec_ups2", "size": [2, 2],
     "in_hw_c": [64, 64, 128], "out_hw": [128, 128]},
    {"name": "dec_ups3", "size": [4, 4],
     "in_hw_c": [128, 128, 64], "out_hw": [512, 512]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="umsi_legacy_resize_fix_protocol.json")
    args = ap.parse_args()

    protocol = {
        "phase": "0_frozen_fix_protocol",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "umsi_legacy_resize_production_fix",
        "fix_branch_ref": os.environ.get("GITHUB_REF", ""),
        "fix_commit": os.environ.get("GITHUB_SHA", ""),
        "production_base_commit":
            "6a17288bbac848bb023fccc98aee606c034ea54b",
        "audit_source_of_truth": {
            "branch": "audit/umsi-resize-causality",
            "commit": "a03c42a8ac508d61f8b7fd1cffde3c0e0df8d5f3",
            "verified_operator": ("tf.raw_ops.ResizeBilinear("
                                  "align_corners=False, "
                                  "half_pixel_centers=False), out=in*factor"),
            "operator_evidence": ("resize causality run 30034973775: oracle "
                                  "max_abs_diff = 0.0 vs genuine TF1.14 on "
                                  "ramp+checkerboard at all three decoder "
                                  "boundaries"),
        },
        "production_change": {
            "kind": ("replace three modern half-pixel bilinear UpSampling2D "
                     "decoder resizes with a LegacyBilinearUpSampling2D layer "
                     "that calls the verified legacy operator"),
            "new_layer": "saliency.umsi_model.LegacyBilinearUpSampling2D",
            "call_sites": CALL_SITES,
            "diff_allowlist": DIFF_ALLOWLIST,
            "must_not_change": [
                "public Python interfaces (UMSIPlus, build_umsi_model, "
                "preprocess_image, postprocess_saliency)",
                "model input/output contract (input (None,256,256,3); "
                "outputs [(None,512,512,1),(None,6)])",
                "trainable/total parameter counts",
                "checkpoint key mapping / by-name load compatibility",
                "feature_norms.json, sensitivity_lookup.json, any thresholds",
            ],
        },
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
        "checkpoint": {
            "path": ("saliency/weights/model_weights/saliency_models/"
                     "UMSI++/umsi++.hdf5"),
            "sha256": common.CHECKPOINT_SHA256,
            "byte_size": common.CHECKPOINT_BYTES,
        },
        "upstream": {
            "repo": common.UPSTREAM_REPO,
            "commit": common.UPSTREAM_COMMIT,
        },
        "fixtures": {
            name: {
                "dims_hw": [common.FIXTURES[name]["h"],
                            common.FIXTURES[name]["w"]],
                "required_sha256": common.FIXTURES[name]["sha256"],
            } for name in common.FIXTURE_ORDER
        },
        "validation_stages": [
            "operator_oracle: production helper reproduces the verified legacy "
            "operator bit-for-bit at the three decoder shapes and differs from "
            "the half-pixel op it replaces",
            "structural_callsite: exactly dec_ups1/dec_ups2/dec_ups3 use the "
            "compat layer; no stock UpSampling2D remains",
            "checkpoint_param_compat: strict by-name checkpoint load with zero "
            "warnings; total+trainable param counts unchanged vs base; I/O "
            "contract unchanged",
            "focused_unit_tests: non-square, odd dims, boundary pixels, dtype, "
            "determinism, equality with legacy operator, actual decoder shapes",
            "golden_compare: same five frozen fixtures, same checkpoint, same "
            "TF1.14 reference impl, same preprocessing, same shared postprocess, "
            "same metrics/formulas/thresholds/containers as the frozen golden "
            "protocol; per fixture capture inputs, raw tensors, decoder-boundary "
            "activations, shared maps, 5 features, 6 PROVISIONAL HCEye, "
            "provisional cognitive_load_index, all threshold results",
            "repeatability: same-process and fresh-process array hashes + "
            "metric/downstream stability",
            "regression_interface: public interfaces, checkpoint loading, "
            "shapes/dtypes, downstream feature extraction, HCEye field "
            "availability, non-golden tests",
        ],
        "metrics": ["shape", "dtype", "min", "max", "mean", "std",
                    "nonfinite_pct", "sha256", "mae", "rmse", "max_abs_diff",
                    "pearson", "spearman", "ssim"],
        "formulas": {
            "sha256_array": ("hashlib.sha256(np.ascontiguousarray(arr, "
                             "dtype=float32).tobytes())"),
            "ssim": ("global single-window SSIM, c1=0.01^2, c2=0.03^2 "
                     "(golden/compare.py::ssim)"),
            "shared_postprocess": ("squeeze_2d -> aspect-preserving cv2 resize + "
                                   "centre-crop to fixture (H,W) -> exact "
                                   "min-max to [0,1]"),
        },
        "pre_registered_thresholds": common.THRESHOLDS,
        "continuous_features": common.CONTINUOUS_FEATURES,
        "feature_keys": common.FEAT_KEYS,
        "hceye_keys": common.HCE_KEYS,
        "repeatability_requirement": {
            "same_process_unique_hashes": 1,
            "fresh_process_unique_hashes_incl_main": 1,
            "metric_downstream_abs_max": 0.0,
        },
        "verdict_logic": {
            "pass_string": ("UMSI LEGACY-RESIZE PRODUCTION FIX VALIDATED — "
                            "ALL FROZEN GOLDEN CRITERIA PASSED"),
            "block_string": ("UMSI LEGACY-RESIZE PRODUCTION FIX BLOCKED — "
                             "GOLDEN OR SCOPE VALIDATION FAILED"),
            "pass_requires_all": [
                "diff within allowlist (only saliency/umsi_model.py)",
                "operator faithful (helper == legacy operator, max_abs 0.0)",
                "exactly three call sites use the compat layer",
                "checkpoint loads by-name with zero warnings",
                "total+trainable param counts unchanged",
                "all five fixtures pass every pre-registered threshold",
                "downstream (feature + HCEye + CLI) comparisons pass",
                "same-process and fresh-process repeatability pass",
                "regression/interface checks pass",
                "provenance complete",
                "protected refs unchanged",
                "no norms rebuild executed",
            ],
        },
        "scope_disclaimer": ("An x86_64 CPU comparison is a source-matched "
                             "legacy numerical golden run, NOT a claim of "
                             "bitwise CUDA 9 reproduction."),
        "provisional_note": ("All HCEye and cognitive_load_index values are "
                             "PROVISIONAL: saliency norm distributions are "
                             "stale and are NOT rebuilt in this milestone; they "
                             "measure downstream numerical impact only."),
    }
    with open(args.out, "w") as fh:
        json.dump(protocol, fh, indent=2, sort_keys=True)
    print("FROZEN FIX PROTOCOL written to %s" % args.out, flush=True)
    print(json.dumps(protocol["pre_registered_thresholds"], indent=2),
          flush=True)


if __name__ == "__main__":
    main()
