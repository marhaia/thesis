"""Controlled-input golden comparison for the legacy-resize production fix.

This is the pre-registered isolation of the decoder resize fix, matching the
milestone requirement to compare on CONTROLLED (byte-identical) inputs and the
audit's established byte-identical-control methodology (a03c42a): instead of
letting each runtime preprocess independently (which differs by a pre-existing
~3.8e-6 float32-vs-float64 amount, unrelated to the resize), we feed the SAME
legacy-produced preprocessed tensor into BOTH the legacy reference output and
the FIXED modern model, so the ONLY remaining variable is the decoder resize
operator being validated.

Runs in the modern (TF2.16/Keras3) container with the FIXED model. The legacy
raw outputs and legacy preprocessed tensors come from the legacy runner
artifact; the modern-controlled raw output is produced here by running the fixed
model on the legacy preprocessed tensor.

Preprocessing parity is therefore identical (max_abs_diff = 0.0) by construction;
every other frozen threshold is evaluated exactly as in the frozen golden
protocol.
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from shared_postprocess import shared_postprocess  # noqa: E402
# Reuse the exact scoring helpers from the frozen comparator.
from compare import (ssim, metrics, arr_stats,  # noqa: E402,F401
                     compute_feats, compute_hce)

REPO = os.environ["UMSI_REPO"]
sys.path.insert(0, REPO)


def fixed_raw_on(model, preproc):
    """Run the FIXED modern model on a supplied preprocessed tensor.
    Returns (raw (512,512) float32, classif (6,) float64)."""
    x = np.ascontiguousarray(preproc).astype(np.float32)
    preds = model.model.predict(x, verbose=0)
    raw = np.asarray(preds[0])[0, :, :, 0].astype(np.float32)
    classif = np.asarray(preds[1])[0].astype(np.float64)
    return raw, classif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    from saliency.umsi_model import UMSIPlus
    model = UMSIPlus(args.ckpt)

    lmeta = json.load(open(os.path.join(args.legacy_dir, "legacy_meta.json")))
    lnpz = np.load(os.path.join(args.legacy_dir, "legacy_arrays.npz"),
                   allow_pickle=False)

    T = common.THRESHOLDS
    fixtures_out = {}
    npz_out = {}
    all_pass = True

    print("\n=========== CONTROLLED-INPUT GOLDEN (per-fixture) ===========",
          flush=True)
    print("%-12s %8s %8s %8s %8s %8s %6s" % (
        "fixture", "pearson", "spearman", "ssim", "featMax", "dCLI", "PASS"),
        flush=True)

    for name in common.FIXTURE_ORDER:
        h = common.FIXTURES[name]["h"]
        w = common.FIXTURES[name]["w"]
        lraw = lnpz["%s_raw" % name].astype(np.float32)
        lpre = lnpz["%s_preproc" % name]
        lclass = lnpz["%s_classif" % name].astype(np.float64)

        # Feed the identical legacy preprocessed tensor to the fixed model.
        mraw, mclass = fixed_raw_on(model, lpre)

        # Preprocessing is identical by construction: we verify the tensor we
        # fed is bit-for-bit the legacy preprocessed tensor.
        fed = np.ascontiguousarray(lpre).astype(np.float32)
        preproc_diff = float(np.abs(
            fed.astype(np.float64)
            - np.ascontiguousarray(lpre).astype(np.float32).astype(np.float64)
        ).max())
        preproc_ok = preproc_diff <= T["preproc_abs_max"]

        failed = []
        if not preproc_ok:
            failed.append("controlled_preproc_max_abs_diff=%.3e" % preproc_diff)

        orientation_ok = (lraw.shape == mraw.shape)
        if not orientation_ok:
            failed.append("raw_shape_mismatch")
        finite_ok = bool(np.isfinite(lraw).all() and np.isfinite(mraw).all())
        if not finite_ok:
            failed.append("nonfinite_raw")

        lshared = shared_postprocess(lraw, h, w)
        mshared = shared_postprocess(mraw, h, w)
        sm = metrics(lshared, mshared)
        pearson_ok = (sm["pearson"] is not None and sm["pearson"] >= T["pearson_min"])
        spearman_ok = (sm["spearman"] is not None and sm["spearman"] >= T["spearman_min"])
        ssim_ok = (sm["ssim"] >= T["ssim_min"])
        if not pearson_ok:
            failed.append("pearson=%s" % sm["pearson"])
        if not spearman_ok:
            failed.append("spearman=%s" % sm["spearman"])
        if not ssim_ok:
            failed.append("ssim=%.6f" % sm["ssim"])

        feats_l = compute_feats(lshared)
        feats_m = compute_feats(mshared)
        feat_max = max(abs(feats_l[k] - feats_m[k])
                       for k in common.CONTINUOUS_FEATURES)
        features_ok = feat_max <= T["feature_abs_max"]
        if not features_ok:
            failed.append("feature_abs_max=%.4f" % feat_max)
        peak_ok = (feats_l["saliency_peak_count"] == feats_m["saliency_peak_count"])
        if not peak_ok:
            failed.append("peak_count %s vs %s" % (
                feats_l["saliency_peak_count"], feats_m["saliency_peak_count"]))

        hce_l = compute_hce(feats_l)
        hce_m = compute_hce(feats_m)
        dcli = abs(hce_l["cognitive_load_index"] - hce_m["cognitive_load_index"])
        cli_ok = dcli <= T["cli_abs_max"]
        if not cli_ok:
            failed.append("dCLI=%.4f" % dcli)

        # Same-process determinism of the controlled modern prediction.
        raw2, _ = fixed_raw_on(model, lpre)
        repeat_ok = (common.sha256_array(mraw) == common.sha256_array(raw2))
        if not repeat_ok:
            failed.append("nondeterministic_controlled_repeat")

        fixture_pass = (preproc_ok and orientation_ok and finite_ok
                        and pearson_ok and spearman_ok and ssim_ok
                        and features_ok and peak_ok and cli_ok and repeat_ok)
        all_pass = all_pass and fixture_pass

        fixtures_out[name] = {
            "dims_hw": [h, w],
            "controlled_input": {
                "source": "legacy_preprocessed_tensor",
                "preproc_max_abs_diff": preproc_diff,
                "identical_by_construction": preproc_ok,
                "input_sha256": common.sha256_array(lpre.astype(np.float32)),
            },
            "raw_tensor": {
                "legacy": arr_stats(lraw),
                "modern_controlled": arr_stats(mraw),
                "metrics": metrics(lraw, mraw),
            },
            "shared_map": {
                "legacy_stats": arr_stats(lshared),
                "modern_controlled_stats": arr_stats(mshared),
                "metrics": sm,
            },
            "features_on_shared_map": {
                "legacy": feats_l, "modern_controlled": feats_m,
                "abs_delta": {k: abs(feats_l[k] - feats_m[k])
                              for k in common.FEAT_KEYS},
                "continuous_feature_abs_max": feat_max,
            },
            "hceye_PROVISIONAL": {
                "legacy": hce_l, "modern_controlled": hce_m,
                "cognitive_load_index_abs_diff": dcli,
            },
            "classif_aux_numeric_only": {
                "legacy": [float(v) for v in lclass],
                "modern_controlled": [float(v) for v in mclass],
                "max_abs_diff": float(np.abs(lclass - mclass).max()),
            },
            "checks": {
                "controlled_preprocessing_identical": preproc_ok,
                "orientation_shape": orientation_ok,
                "finite": finite_ok,
                "pearson_ge_0999": pearson_ok,
                "spearman_ge_099": spearman_ok,
                "ssim_ge_099": ssim_ok,
                "continuous_features_le_002": features_ok,
                "peak_count_exact": peak_ok,
                "cli_le_001": cli_ok,
                "deterministic_repeat": repeat_ok,
                "all_pass": fixture_pass,
            },
            "failed_checks": failed,
        }
        npz_out["%s_legacy_raw" % name] = lraw
        npz_out["%s_modern_controlled_raw" % name] = mraw
        npz_out["%s_legacy_shared" % name] = lshared
        npz_out["%s_modern_controlled_shared" % name] = mshared

        def _f(x):
            return ("%.6f" % x) if isinstance(x, float) else str(x)
        print("%-12s %8s %8s %8s %8s %8s %6s" % (
            name, _f(sm["pearson"]), _f(sm["spearman"]), _f(sm["ssim"]),
            _f(feat_max), _f(dcli), "PASS" if fixture_pass else "FAIL"),
            flush=True)

    verdict = ("CONTROLLED-INPUT GOLDEN PARITY PASSED" if all_pass
               else "CONTROLLED-INPUT GOLDEN PARITY FAILED")
    result = {
        "evidence_kind": "controlled_input_golden_isolating_decoder_resize",
        "methodology": ("identical legacy-preprocessed tensor fed to the "
                        "legacy reference and the FIXED modern model; the only "
                        "variable is the decoder resize operator"),
        "for_commit": "6a17288bbac848bb023fccc98aee606c034ea54b",
        "thresholds": T,
        "environments": {"legacy": lmeta.get("environment", {})},
        "fixtures": fixtures_out,
        "overall": {"all_fixtures_pass": all_pass, "verdict": verdict},
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out_npz, **npz_out)
    print("\n================ CONTROLLED VERDICT ================", flush=True)
    print(verdict, flush=True)
    print("wrote %s" % args.out_json, flush=True)
    print("wrote %s" % args.out_npz, flush=True)


if __name__ == "__main__":
    main()
