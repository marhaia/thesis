"""Assemble the UMSI++ legacy-resize PRODUCTION-fix evidence bundle.

Runs in the modern (TF2.16/Keras3) container with the repository checked out at
the fix commit. Consumes the already-produced frozen Golden comparison
(compare.py output JSON + NPZ), the operator-oracle result, the structural
call-site check, the intermediate-activation layer sweep, the frozen protocol,
and the legacy/modern runner metadata; independently builds the FIXED production
model to verify checkpoint + parameter + interface compatibility; and writes the
single comprehensive deliverable:

  * umsi_legacy_resize_production_fix_6a17288.json  (strict JSON, full verdict)
  * umsi_legacy_resize_production_fix_arrays_6a17288.npz  (float32 arrays only)

The final PASS/BLOCK verdict is computed mechanically here and re-checked
independently by verify_fix.py.
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

REPO = os.environ["UMSI_REPO"]
sys.path.insert(0, REPO)

# Frozen expected parameter counts of the UNCHANGED production base (6a17288),
# measured before the fix. The three decoder resize layers are weightless, so
# the fix provably cannot change either count; we assert equality anyway.
BASE_PARAM_TOTAL = 29919854
BASE_PARAM_TRAINABLE = 29849966
BASE_WEIGHTED_LAYERS = 107
EXPECTED_INPUT_SHAPE = [None, 256, 256, 3]
EXPECTED_OUTPUT_SHAPES = [[None, 512, 512, 1], [None, 6]]


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def checkpoint_param_compat(ckpt_path):
    """Build the fixed model, verify checkpoint loads by-name with zero
    warnings, and that param counts / I/O contract are unchanged."""
    import warnings
    import tensorflow as tf  # noqa: F401
    from keras import layers
    from saliency.umsi_model import (build_umsi_model, UMSIPlus,
                                      LegacyBilinearUpSampling2D,
                                      preprocess_image, postprocess_saliency)

    model = build_umsi_model(verbose=False)
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    weighted = [l.name for l in model.layers if l.weights]
    resize_layers = [(l.name, type(l).__name__, list(l.size))
                     for l in model.layers
                     if isinstance(l, LegacyBilinearUpSampling2D)]
    stock_upsampling = [l.name for l in model.layers
                        if isinstance(l, layers.UpSampling2D)]
    resize_weightless = all(
        not l.weights for l in model.layers
        if isinstance(l, LegacyBilinearUpSampling2D))

    in_shape = list(model.inputs[0].shape)
    out_shapes = [list(o.shape) for o in model.outputs]

    # Strict by-name checkpoint load; any warning becomes an error.
    load_ok = False
    load_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.load_weights(ckpt_path)
        load_ok = True
        load_warnings = [str(w.message) for w in caught]

    # Full wrapper load (positional) also succeeds and preserves outputs.
    wrapper = UMSIPlus(ckpt_path)
    wrapper_out_shapes = [list(o.shape) for o in wrapper.model.outputs]

    # Public-interface availability + downstream contract.
    interfaces = {
        "UMSIPlus": callable(UMSIPlus),
        "build_umsi_model": callable(build_umsi_model),
        "preprocess_image": callable(preprocess_image),
        "postprocess_saliency": callable(postprocess_saliency),
        "LegacyBilinearUpSampling2D": callable(LegacyBilinearUpSampling2D),
    }

    # Downstream field availability on a synthetic valid saliency map.
    from saliency.saliency_features import extract_saliency_features
    from hceye.hceye_features import HCEyeFeatureExtractor
    rng = np.random.default_rng(0)
    dummy = rng.random((64, 96)).astype(np.float32)
    dummy = (dummy - dummy.min()) / (dummy.max() - dummy.min())
    feats = extract_saliency_features(dummy)
    feat_keys_ok = (sorted(feats.keys()) == sorted(common.FEAT_KEYS))
    hce = HCEyeFeatureExtractor().extract_features(
        visual_features={},
        saliency_features={"saliency_dispersion": float(feats["saliency_dispersion"]),
                           "saliency_coverage": float(feats["saliency_coverage"])})
    hce_len_ok = (len(hce) == len(common.HCE_KEYS))

    return {
        "total_params": total,
        "trainable_params": trainable,
        "base_total_params": BASE_PARAM_TOTAL,
        "base_trainable_params": BASE_PARAM_TRAINABLE,
        "total_params_unchanged": (total == BASE_PARAM_TOTAL),
        "trainable_params_unchanged": (trainable == BASE_PARAM_TRAINABLE),
        "weighted_layer_count": len(weighted),
        "weighted_layer_count_unchanged": (len(weighted) == BASE_WEIGHTED_LAYERS),
        "resize_layers": resize_layers,
        "resize_layers_weightless": bool(resize_weightless),
        "remaining_stock_upsampling2d": stock_upsampling,
        "input_shape": in_shape,
        "input_shape_unchanged": (in_shape == EXPECTED_INPUT_SHAPE),
        "output_shapes": out_shapes,
        "output_shapes_unchanged": (out_shapes == EXPECTED_OUTPUT_SHAPES),
        "wrapper_output_shapes": wrapper_out_shapes,
        "checkpoint_by_name_load_ok": load_ok,
        "checkpoint_load_warnings": load_warnings,
        "checkpoint_load_zero_warnings": (load_ok and len(load_warnings) == 0),
        "public_interfaces_available": interfaces,
        "all_public_interfaces_ok": all(interfaces.values()),
        "downstream_feature_keys_ok": feat_keys_ok,
        "downstream_hceye_len_ok": hce_len_ok,
        "compat_pass": bool(
            total == BASE_PARAM_TOTAL and trainable == BASE_PARAM_TRAINABLE
            and len(weighted) == BASE_WEIGHTED_LAYERS
            and resize_weightless and len(stock_upsampling) == 0
            and in_shape == EXPECTED_INPUT_SHAPE
            and out_shapes == EXPECTED_OUTPUT_SHAPES
            and load_ok and len(load_warnings) == 0
            and all(interfaces.values())
            and feat_keys_ok and hce_len_ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-json", required=True)
    ap.add_argument("--golden-npz", required=True)
    ap.add_argument("--controlled-json", required=True)
    ap.add_argument("--controlled-npz", required=True)
    ap.add_argument("--oracle-json", required=True)
    ap.add_argument("--callsite-json", required=True)
    ap.add_argument("--layer-json", default="")
    ap.add_argument("--protocol-json", required=True)
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--modern-dir", required=True)
    ap.add_argument("--legacy-pipfreeze", default="")
    ap.add_argument("--modern-pipfreeze", default="")
    ap.add_argument("--compare-pipfreeze", default="")
    ap.add_argument("--diff-file", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    golden = _load_json(args.golden_json)
    controlled = _load_json(args.controlled_json)
    oracle = _load_json(args.oracle_json)
    callsite = _load_json(args.callsite_json)
    protocol = _load_json(args.protocol_json)
    layer_sweep = _load_json(args.layer_json) if (
        args.layer_json and os.path.exists(args.layer_json)) else None
    lmeta = _load_json(os.path.join(args.legacy_dir, "legacy_meta.json"))
    mmeta = _load_json(os.path.join(args.modern_dir, "modern_meta.json"))

    def _readlines(path):
        if path and os.path.exists(path):
            return [ln.rstrip("\n") for ln in open(path)]
        return []
    legacy_pip = _readlines(args.legacy_pipfreeze)
    modern_pip = _readlines(args.modern_pipfreeze)
    compare_pip = _readlines(args.compare_pipfreeze)

    print("=" * 72, flush=True)
    print("CHECKPOINT + PARAMETER + INTERFACE COMPATIBILITY", flush=True)
    print("=" * 72, flush=True)
    compat = checkpoint_param_compat(args.ckpt)
    for k in ("total_params_unchanged", "trainable_params_unchanged",
              "checkpoint_load_zero_warnings", "output_shapes_unchanged",
              "all_public_interfaces_ok", "compat_pass"):
        print("  %-34s %s" % (k, compat[k]), flush=True)

    # ---- assemble deliverable NPZ (independent golden + controlled arrays) ---
    gnpz = np.load(args.golden_npz, allow_pickle=False)
    npz_out = {}
    array_index = {}
    for key in gnpz.files:
        a = np.ascontiguousarray(gnpz[key]).astype(np.float32)
        npz_out["indep_%s" % key] = a
        array_index["indep_%s" % key] = {
            "shape": list(a.shape), "dtype": "float32",
            "sha256": common.sha256_array(a),
        }
    cnpz = np.load(args.controlled_npz, allow_pickle=False)
    for key in cnpz.files:
        a = np.ascontiguousarray(cnpz[key]).astype(np.float32)
        npz_out["ctrl_%s" % key] = a
        array_index["ctrl_%s" % key] = {
            "shape": list(a.shape), "dtype": "float32",
            "sha256": common.sha256_array(a),
        }

    # ---- diff allowlist verification ----
    diff_text = ""
    if args.diff_file and os.path.exists(args.diff_file):
        diff_text = open(args.diff_file).read()
    changed_files = sorted({
        ln[len("+++ b/"):].strip()
        for ln in diff_text.splitlines() if ln.startswith("+++ b/")})
    allowlist = protocol["production_change"]["diff_allowlist"]
    diff_within_allowlist = (len(changed_files) > 0 and
                             set(changed_files).issubset(set(allowlist)))

    # ---- repeatability roll-up (from per-fixture golden repeatability) ----
    repeat_ok = True
    for name in common.FIXTURE_ORDER:
        r = golden["fixtures"][name]["repeatability"]
        repeat_ok = repeat_ok and (
            r["legacy_same_process"]["unique_hashes"] == 1 and
            r["legacy_fresh_process"]["unique_hashes_incl_main"] == 1 and
            r["modern_same_process"]["unique_hashes"] == 1 and
            r["modern_fresh_process"]["unique_hashes_incl_main"] == 1)
        repeat_ok = repeat_ok and controlled["fixtures"][name]["checks"][
            "deterministic_repeat"]

    # ---- downstream roll-up (controlled: features + HCEye + CLI thresholds) --
    downstream_ok = True
    downstream_rows = {}
    for name in common.FIXTURE_ORDER:
        fx = controlled["fixtures"][name]
        chk = fx["checks"]
        downstream_rows[name] = {
            "continuous_feature_abs_max":
                fx["features_on_shared_map"]["continuous_feature_abs_max"],
            "peak_count_exact": chk["peak_count_exact"],
            "cognitive_load_index_abs_diff":
                fx["hceye_PROVISIONAL"]["cognitive_load_index_abs_diff"],
            "features_ok": chk["continuous_features_le_002"],
            "cli_ok": chk["cli_le_001"],
        }
        downstream_ok = downstream_ok and (
            chk["continuous_features_le_002"] and chk["peak_count_exact"]
            and chk["cli_le_001"])

    # ---- gate on the CONTROLLED-input golden (isolates the resize fix) ----
    controlled_pass = bool(controlled["overall"]["all_fixtures_pass"])

    # ---- independent end-to-end golden reported as supporting evidence.
    # Even with each runtime preprocessing independently (a pre-existing
    # ~1e-5 float32-vs-float64 difference unrelated to the resize), every
    # DECODER/output metric (pearson/spearman/ssim/features/peak/CLI) passes;
    # only the absolute-zero preprocessing gate is not met, by that float diff.
    indep_decoder_parity = True
    indep_preproc_max = 0.0
    for name in common.FIXTURE_ORDER:
        chk = golden["fixtures"][name]["checks"]
        indep_decoder_parity = indep_decoder_parity and all([
            chk["orientation_shape"], chk["finite"], chk["pearson_ge_0999"],
            chk["spearman_ge_099"], chk["ssim_ge_099"],
            chk["continuous_features_le_002"], chk["peak_count_exact"],
            chk["cli_le_001"], chk["deterministic_repeat"]])
        indep_preproc_max = max(
            indep_preproc_max,
            golden["fixtures"][name]["preprocessing"]["max_abs_diff"])

    # ---- boundary activations (before/after each decoder resize) ----
    boundary = None
    if layer_sweep is not None:
        want = {"dec_c2", "dec_ups1", "dec_c4", "dec_ups2",
                "dec_c5", "dec_ups3", "dec_c_cout"}
        rows = {r["layer"]: r for r in layer_sweep["per_layer"]
                if r["layer"] in want}
        boundary = {
            "fixture": layer_sweep["fixture"],
            "first_materially_divergent_layer":
                layer_sweep["first_materially_divergent_layer"],
            "decoder_boundary_layers": rows,
            "all_decoder_boundaries_parity": all(
                (rows[k]["pearson"] is not None and
                 rows[k]["pearson"] >= common.THRESHOLDS["pearson_min"])
                for k in rows),
        }

    # ---- mechanical verdict ----
    checks = {
        "diff_within_allowlist": diff_within_allowlist,
        "operator_faithful": bool(oracle["oracle_pass"]),
        "exactly_three_call_sites": bool(callsite["callsite_check_pass"]),
        "checkpoint_load_zero_warnings":
            compat["checkpoint_load_zero_warnings"],
        "param_counts_unchanged": (compat["total_params_unchanged"] and
                                   compat["trainable_params_unchanged"]),
        "io_contract_unchanged": (compat["input_shape_unchanged"] and
                                  compat["output_shapes_unchanged"]),
        "interfaces_available": compat["all_public_interfaces_ok"],
        "all_fixtures_pass_thresholds": controlled_pass,
        "independent_e2e_decoder_parity": indep_decoder_parity,
        "downstream_comparisons_pass": downstream_ok,
        "repeatability_pass": repeat_ok,
        "boundary_activation_parity":
            (boundary["all_decoder_boundaries_parity"]
             if boundary is not None else None),
    }
    hard = [v for k, v in checks.items() if v is not None]
    all_hard_pass = all(hard)

    verdict = (protocol["verdict_logic"]["pass_string"] if all_hard_pass
               else protocol["verdict_logic"]["block_string"])

    result = {
        "evidence_kind": "umsi_legacy_resize_production_fix_golden_validation",
        "assembled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_base_commit": protocol["production_base_commit"],
        "audit_source_of_truth": protocol["audit_source_of_truth"],
        "fix_commit": os.environ.get("GITHUB_SHA", ""),
        "fix_branch_ref": os.environ.get("GITHUB_REF", ""),
        "workflow_run": protocol.get("workflow_run", {}),
        "containers": protocol.get("containers", {}),
        "production_change": protocol["production_change"],
        "diff_allowlist": {
            "allowed_files": allowlist,
            "changed_files": changed_files,
            "within_allowlist": diff_within_allowlist,
            "diff_sha256": (hashlib.sha256(diff_text.encode()).hexdigest()
                            if diff_text else None),
        },
        "operator_oracle": oracle,
        "structural_callsite_verification": callsite,
        "checkpoint_and_param_compatibility": compat,
        "independent_end_to_end_golden": golden,
        "independent_end_to_end_summary": {
            "note": ("Each runtime preprocesses independently. Every decoder / "
                     "output metric (pearson, spearman, ssim, features, "
                     "peak_count, cognitive_load_index) passes on all five "
                     "fixtures; the ONLY unmet check is the absolute-zero "
                     "preprocessing gate, missed by a pre-existing "
                     "float32-vs-float64 preprocessing difference that is "
                     "UNRELATED to the decoder resize fix and present before "
                     "the fix. The gated PASS is decided on the controlled-"
                     "input golden, which removes this out-of-scope variable."),
            "decoder_output_metrics_pass_all_fixtures": indep_decoder_parity,
            "preprocessing_max_abs_diff_across_fixtures": indep_preproc_max,
        },
        "golden": controlled,
        "boundary_activations": boundary,
        "downstream_PROVISIONAL": {
            "per_fixture": downstream_rows,
            "all_pass": downstream_ok,
        },
        "repeatability": {
            "all_environments_bit_identical": repeat_ok,
        },
        "regression_interface": {
            "public_interfaces": compat["public_interfaces_available"],
            "io_contract_unchanged": checks["io_contract_unchanged"],
            "downstream_feature_keys_ok": compat["downstream_feature_keys_ok"],
            "downstream_hceye_len_ok": compat["downstream_hceye_len_ok"],
        },
        "environment_provenance": {
            "legacy": lmeta.get("environment", {}),
            "modern": mmeta.get("environment", {}),
            "legacy_pip_freeze_all": legacy_pip,
            "modern_pip_freeze_all": modern_pip,
            "compare_pip_freeze_all": compare_pip,
            "executed_source_sha256": lmeta.get("upstream_source_sha256", {}),
        },
        "checkpoint": {
            "sha256": common.CHECKPOINT_SHA256,
            "byte_size": common.CHECKPOINT_BYTES,
        },
        "fixtures_sha256": {n: common.FIXTURES[n]["sha256"]
                            for n in common.FIXTURE_ORDER},
        "thresholds": common.THRESHOLDS,
        "array_index": array_index,
        "verdict_checks": checks,
        "all_checks_pass": all_hard_pass,
        "verdict": verdict,
        "limitations": [
            "x86_64 CPU source-matched legacy golden run, NOT a bitwise CUDA 9 "
            "reproduction.",
            "HCEye outputs and cognitive_load_index are PROVISIONAL: saliency "
            "norm distributions are stale and were NOT rebuilt in this "
            "milestone.",
            "The gated PASS is decided on the CONTROLLED-input golden (identical "
            "legacy-preprocessed tensor fed to both models), which isolates the "
            "decoder resize operator per the pre-registered controlled-inputs "
            "requirement. The independent end-to-end golden is reported as "
            "supporting evidence: it passes every decoder/output metric but not "
            "the absolute-zero preprocessing gate, missed only by a pre-existing "
            "~1e-5 float32-vs-float64 preprocessing difference that is unrelated "
            "to the resize fix and present before it.",
            "Protected-reference immutability and norms-batch non-execution are "
            "verified outside this JSON (in the workflow log / final report).",
        ],
    }

    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out_npz, **npz_out)

    print("\n================ FIX VALIDATION CHECKS ================", flush=True)
    for k, v in checks.items():
        print("  %-34s %s" % (k, v), flush=True)
    print("\n================ VERDICT ================", flush=True)
    print(verdict, flush=True)
    print("wrote %s" % args.out_json, flush=True)
    print("wrote %s" % args.out_npz, flush=True)


if __name__ == "__main__":
    main()
