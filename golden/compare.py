"""Phase 1/2 comparator + verdict for the UMSI++ TF1/Keras2 <-> TF2/Keras3
golden run.

Runs in the modern (Python 3.9) container with the repository checked out at the
audit commit, so it can use the repo's OWN feature and HCEye code
(saliency.saliency_features.extract_saliency_features and
hceye.hceye_features.HCEyeFeatureExtractor) to score both sides identically.

Loads the two runner artifacts (legacy_arrays.npz + legacy_meta.json and
modern_arrays.npz + modern_meta.json), applies the single shared postprocess to
both raw tensors, computes every metric per fixture, checks each pre-registered
threshold, and writes:
  * umsi_tf1_tf2_golden_6a17288.json  (strict JSON, full provenance + verdict)
  * umsi_tf1_tf2_golden_arrays_6a17288.npz  (raw + shared maps, both envs)
Plus a human-readable results table to stdout (captured into the .log).
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from shared_postprocess import shared_postprocess  # noqa: E402

REPO = os.environ["UMSI_REPO"]
sys.path.insert(0, REPO)
from saliency.saliency_features import extract_saliency_features  # noqa: E402
from hceye.hceye_features import HCEyeFeatureExtractor  # noqa: E402


def ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def metrics(a, b):
    fa = a.ravel().astype(np.float64)
    fb = b.ravel().astype(np.float64)
    d = fa - fb
    out = {
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "max_abs_diff": float(np.abs(d).max()),
    }
    if fa.std() > 0 and fb.std() > 0:
        out["pearson"] = float(pearsonr(fa, fb)[0])
        out["spearman"] = float(spearmanr(fa, fb)[0])
    else:
        out["pearson"] = None
        out["spearman"] = None
    out["ssim"] = ssim(a, b)
    return out


def arr_stats(a):
    fin = np.isfinite(a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(a[fin].min()) if fin.any() else None,
        "max": float(a[fin].max()) if fin.any() else None,
        "mean": float(a[fin].mean()) if fin.any() else None,
        "std": float(a[fin].std()) if fin.any() else None,
        "nonfinite_pct": float((~fin).mean() * 100.0),
        "sha256": common.sha256_array(a),
    }


def compute_feats(shared_map):
    f = {k: float(v) for k, v in extract_saliency_features(shared_map).items()}
    return f


def compute_hce(feat):
    hce = HCEyeFeatureExtractor()
    sf = {"saliency_dispersion": feat["saliency_dispersion"],
          "saliency_coverage": feat["saliency_coverage"]}
    h = hce.extract_features(visual_features={}, saliency_features=sf)
    return {k: float(v) for k, v in zip(common.HCE_KEYS, h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--modern-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    lmeta = json.load(open(os.path.join(args.legacy_dir, "legacy_meta.json")))
    mmeta = json.load(open(os.path.join(args.modern_dir, "modern_meta.json")))
    lnpz = np.load(os.path.join(args.legacy_dir, "legacy_arrays.npz"))
    mnpz = np.load(os.path.join(args.modern_dir, "modern_arrays.npz"))

    T = common.THRESHOLDS
    fixtures_out = {}
    npz_out = {}
    all_pass = True

    print("\n================ GOLDEN COMPARISON (per-fixture) ================", flush=True)
    hdr = "%-12s %8s %8s %8s %8s %8s %6s" % (
        "fixture", "pearson", "spearman", "ssim", "featMax", "dCLI", "PASS")
    print(hdr, flush=True)

    for name in common.FIXTURE_ORDER:
        h = common.FIXTURES[name]["h"]
        w = common.FIXTURES[name]["w"]
        lraw = lnpz["%s_raw" % name].astype(np.float32)
        mraw = mnpz["%s_raw" % name].astype(np.float32)
        lpre = lnpz["%s_preproc" % name].astype(np.float64)
        mpre = mnpz["%s_preproc" % name].astype(np.float64)
        le2e = lnpz["%s_e2e" % name].astype(np.float32)
        me2e = mnpz["%s_e2e" % name].astype(np.float32)
        lclass = lnpz["%s_classif" % name].astype(np.float64)
        mclass = mnpz["%s_classif" % name].astype(np.float64)

        failed = []

        # --- preprocessing tensor parity (expect 0.0) ---
        preproc_diff = float(np.abs(lpre - mpre).max())
        preproc_ok = preproc_diff <= T["preproc_abs_max"]
        if not preproc_ok:
            failed.append("preprocessing_max_abs_diff=%.3e" % preproc_diff)

        # --- orientation / shape ---
        orientation_ok = (lraw.shape == mraw.shape)
        if not orientation_ok:
            failed.append("raw_shape_mismatch %r vs %r" % (lraw.shape, mraw.shape))

        # --- finite ---
        finite_ok = bool(np.isfinite(lraw).all() and np.isfinite(mraw).all())
        if not finite_ok:
            failed.append("nonfinite_raw_output")

        # --- shared postprocessed maps (thresholds evaluated here) ---
        shared_metrics = None
        feats_l = feats_m = hce_l = hce_m = None
        pearson_ok = spearman_ok = ssim_ok = features_ok = peak_ok = cli_ok = False
        feat_max = None
        dcli = None
        lshared = mshared = None
        if orientation_ok and finite_ok:
            try:
                lshared = shared_postprocess(lraw, h, w)
                mshared = shared_postprocess(mraw, h, w)
                shared_metrics = metrics(lshared, mshared)
                p = shared_metrics["pearson"]
                s = shared_metrics["spearman"]
                ss = shared_metrics["ssim"]
                pearson_ok = (p is not None and p >= T["pearson_min"])
                spearman_ok = (s is not None and s >= T["spearman_min"])
                ssim_ok = (ss >= T["ssim_min"])
                if not pearson_ok:
                    failed.append("pearson=%s" % p)
                if not spearman_ok:
                    failed.append("spearman=%s" % s)
                if not ssim_ok:
                    failed.append("ssim=%.6f" % ss)

                feats_l = compute_feats(lshared)
                feats_m = compute_feats(mshared)
                feat_max = 0.0
                for k in common.CONTINUOUS_FEATURES:
                    feat_max = max(feat_max, abs(feats_l[k] - feats_m[k]))
                features_ok = feat_max <= T["feature_abs_max"]
                if not features_ok:
                    failed.append("feature_abs_max=%.4f" % feat_max)

                peak_ok = (feats_l["saliency_peak_count"]
                           == feats_m["saliency_peak_count"])
                if not peak_ok:
                    failed.append("peak_count %s vs %s" % (
                        feats_l["saliency_peak_count"],
                        feats_m["saliency_peak_count"]))

                hce_l = compute_hce(feats_l)
                hce_m = compute_hce(feats_m)
                dcli = abs(hce_l["cognitive_load_index"]
                           - hce_m["cognitive_load_index"])
                cli_ok = dcli <= T["cli_abs_max"]
                if not cli_ok:
                    failed.append("dCLI=%.4f" % dcli)
            except Exception as exc:  # shared postprocess / feature failure
                failed.append("shared_stage_error:%s:%s" % (
                    type(exc).__name__, exc))

        # --- repeatability (bit-identical within each environment) ---
        lsame = lmeta["fixtures"][name]["same_process"]
        lfresh = lmeta["fixtures"][name]["fresh_process"]
        msame = mmeta["fixtures"][name]["same_process"]
        mfresh = mmeta["fixtures"][name]["fresh_process"]
        repeat_ok = (lsame["unique_hashes"] == 1
                     and lfresh["unique_hashes_incl_main"] == 1
                     and msame["unique_hashes"] == 1
                     and mfresh["unique_hashes_incl_main"] == 1)
        if not repeat_ok:
            failed.append("nondeterministic_repeat")

        fixture_pass = (preproc_ok and orientation_ok and finite_ok
                        and pearson_ok and spearman_ok and ssim_ok
                        and features_ok and peak_ok and cli_ok and repeat_ok)
        all_pass = all_pass and fixture_pass

        # genuine end-to-end comparison (each on its own scale -> min-max first)
        def _mm(x):
            x = x.astype(np.float64)
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 0 else np.zeros_like(x)
        e2e_metrics = metrics(_mm(le2e), _mm(me2e))

        fixtures_out[name] = {
            "dims_hw": [h, w],
            "file_sha256": common.FIXTURES[name]["sha256"],
            "preprocessing": {
                "legacy": {"shape": list(lpre.shape),
                           "sha256": common.sha256_array(lpre)},
                "modern": {"shape": list(mpre.shape),
                           "sha256": common.sha256_array(mpre)},
                "max_abs_diff": preproc_diff,
                "identical": preproc_ok,
            },
            "raw_tensor": {
                "legacy": arr_stats(lraw),
                "modern": arr_stats(mraw),
                "metrics": metrics(lraw, mraw) if orientation_ok else None,
            },
            "shared_map": {
                "legacy_stats": arr_stats(lshared) if lshared is not None else None,
                "modern_stats": arr_stats(mshared) if mshared is not None else None,
                "metrics": shared_metrics,
            },
            "genuine_e2e": {
                "legacy": {"shape": list(le2e.shape),
                           "min": float(le2e.min()), "max": float(le2e.max()),
                           "sha256": common.sha256_array(le2e)},
                "modern": {"shape": list(me2e.shape),
                           "min": float(me2e.min()), "max": float(me2e.max()),
                           "sha256": common.sha256_array(me2e)},
                "metrics_after_minmax": e2e_metrics,
            },
            "features_on_shared_map": {
                "legacy": feats_l, "modern": feats_m,
                "abs_delta": ({k: abs(feats_l[k] - feats_m[k])
                               for k in common.FEAT_KEYS}
                              if feats_l and feats_m else None),
                "continuous_feature_abs_max": feat_max,
            },
            "hceye_PROVISIONAL": {
                "legacy": hce_l, "modern": hce_m,
                "abs_delta": ({k: abs(hce_l[k] - hce_m[k])
                               for k in common.HCE_KEYS}
                              if hce_l and hce_m else None),
                "cognitive_load_index_abs_diff": dcli,
            },
            "classif_aux_numeric_only": {
                "legacy": [float(v) for v in lclass],
                "modern": [float(v) for v in mclass],
                "max_abs_diff": float(np.abs(lclass - mclass).max()),
            },
            "repeatability": {
                "legacy_same_process": lsame,
                "legacy_fresh_process": lfresh,
                "modern_same_process": msame,
                "modern_fresh_process": mfresh,
            },
            "checks": {
                "preprocessing_identical": preproc_ok,
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
        npz_out["%s_modern_raw" % name] = mraw
        if lshared is not None:
            npz_out["%s_legacy_shared" % name] = lshared
        if mshared is not None:
            npz_out["%s_modern_shared" % name] = mshared

        def _fmt(x):
            return ("%.6f" % x) if isinstance(x, float) else str(x)
        p = shared_metrics["pearson"] if shared_metrics else None
        s = shared_metrics["spearman"] if shared_metrics else None
        ss = shared_metrics["ssim"] if shared_metrics else None
        print("%-12s %8s %8s %8s %8s %8s %6s" % (
            name,
            _fmt(p) if p is not None else "n/a",
            _fmt(s) if s is not None else "n/a",
            _fmt(ss) if ss is not None else "n/a",
            _fmt(feat_max) if feat_max is not None else "n/a",
            _fmt(dcli) if dcli is not None else "n/a",
            "PASS" if fixture_pass else "FAIL"), flush=True)

    if all_pass:
        verdict = "TF1/KERAS2 GOLDEN PARITY PASSED"
    else:
        verdict = "TF1/KERAS2 GOLDEN PARITY FAILED — NUMERICAL DIFFERENCE"

    manifest = {}
    if args.manifest and os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest))

    result = {
        "evidence_kind": "tf1_keras2_vs_tf2_keras3_golden_numerical_comparison",
        "for_commit": "6a17288bbac848bb023fccc98aee606c034ea54b",
        "audit_branch_ref": os.environ.get("GITHUB_REF", ""),
        "audit_commit": os.environ.get("GITHUB_SHA", ""),
        "workflow_run": manifest.get("workflow_run", {}),
        "containers": manifest.get("containers", {}),
        "scope_disclaimer": ("An x86_64 CPU comparison is a source-matched "
                             "legacy numerical golden run, NOT a claim of "
                             "bitwise CUDA 9 reproduction."),
        "provisional_note": ("All HCEye and cognitive_load_index values are "
                             "PROVISIONAL (saliency norms are stale); used only "
                             "to measure downstream numerical impact."),
        "thresholds": T,
        "environments": {
            "legacy": lmeta.get("environment", {}),
            "modern": mmeta.get("environment", {}),
        },
        "checkpoint": {
            "legacy": lmeta.get("checkpoint", {}),
            "modern": mmeta.get("checkpoint", {}),
            "expected_sha256": common.CHECKPOINT_SHA256,
            "expected_byte_size": common.CHECKPOINT_BYTES,
        },
        "upstream": {"repo": common.UPSTREAM_REPO, "commit": common.UPSTREAM_COMMIT},
        "executed_source_sha256": lmeta.get("upstream_source_sha256", {}),
        "legacy_captured_warnings": lmeta.get("captured_warnings", []),
        "modern_captured_warnings": mmeta.get("captured_warnings", []),
        "fixtures": fixtures_out,
        "overall": {
            "all_fixtures_pass": all_pass,
            "verdict": verdict,
        },
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out_npz, **npz_out)

    print("\n================ VERDICT ================", flush=True)
    print(verdict, flush=True)
    print("wrote %s" % args.out_json, flush=True)
    print("wrote %s" % args.out_npz, flush=True)


if __name__ == "__main__":
    main()
