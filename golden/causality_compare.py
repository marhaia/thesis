"""Comparator + verdict for the UMSI++ resize causality experiment.

Runs in the modern (Python 3.9) container with the repository checked out at the
audit commit, so it scores every controlled condition with the repo's OWN
feature / HCEye code, exactly as the golden comparator did.

Inputs (from the two runner jobs):
  legacy_causality_meta.json, legacy_causality_arrays.npz,
  legacy_oracle_inputs.npz, legacy_oracle.npz, legacy_boundary_<fx>.npz
  modern_causality_meta.json, modern_causality_arrays.npz,
  modern_oracle_candidate.npz, modern_boundary_<tag>_<fx>.npz

It computes, against the L_REF legacy reference:
  * the resize-operator ORACLE result (legacy vs candidate, <=1e-6);
  * per fixture, for every condition (M_CC/M_LC/M_CL/M_LL), the shared-map
    metrics (both legacy_audit_global_ssim and windowed_ssim), saliency
    features, HCEye (PROVISIONAL) and every pre-registered threshold;
  * the per-resize-boundary activation comparison for the boundary fixture;
  * the causality decision + exact verdict line.

Writes umsi_resize_causality_6a17288.json + _arrays.npz. No suppression.
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
import resize_common as RC  # noqa: E402
from shared_postprocess import shared_postprocess  # noqa: E402
from windowed_ssim import windowed_ssim, WINDOWED_SSIM_PARAMS  # noqa: E402

REPO = os.environ["UMSI_REPO"]
sys.path.insert(0, REPO)
from saliency.saliency_features import extract_saliency_features  # noqa: E402
from hceye.hceye_features import HCEyeFeatureExtractor  # noqa: E402


def global_ssim(a, b):
    """The previous audit's global SSIM formula (kept for continuity)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def full_metrics(a, b):
    fa = a.ravel().astype(np.float64)
    fb = b.ravel().astype(np.float64)
    d = fa - fb
    out = {"shape": list(a.shape), "dtype": str(a.dtype),
           "min_a": float(np.nanmin(a)), "max_a": float(np.nanmax(a)),
           "mean_a": float(np.nanmean(a)), "std_a": float(np.nanstd(a)),
           "nonfinite_pct": float((~np.isfinite(a)).mean() * 100.0),
           "sha256": common.sha256_array(a),
           "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())),
           "max_abs_diff": float(np.abs(d).max())}
    if fa.std() > 0 and fb.std() > 0:
        out["pearson"] = float(pearsonr(fa, fb)[0])
        out["spearman"] = float(spearmanr(fa, fb)[0])
    else:
        out["pearson"] = None
        out["spearman"] = None
    out["legacy_audit_global_ssim"] = global_ssim(a, b)
    out["windowed_ssim"] = windowed_ssim(a, b)
    return out


def feats_of(shared_map):
    return {k: float(v) for k, v in extract_saliency_features(shared_map).items()}


def hce_of(feat):
    hce = HCEyeFeatureExtractor()
    sf = {"saliency_dispersion": feat["saliency_dispersion"],
          "saliency_coverage": feat["saliency_coverage"]}
    h = hce.extract_features(visual_features={}, saliency_features=sf)
    return {k: float(v) for k, v in zip(common.HCE_KEYS, h)}


def evaluate_condition(ref_shared, cond_raw, h, w, T):
    """Compare a condition's shared map to the L_REF shared map; return the
    metrics/features/HCEye block + per-threshold pass flags + failed list."""
    failed = []
    if cond_raw.shape != (512, 512):
        return ({"error": "unexpected raw shape %r" % (cond_raw.shape,)},
                ["shape"], False, np.zeros((h, w), np.float32))
    if not np.isfinite(cond_raw).all():
        failed.append("nonfinite_raw")
    cond_shared = shared_postprocess(cond_raw, h, w)
    m = full_metrics(ref_shared, cond_shared)
    p, s = m["pearson"], m["spearman"]
    gssim, wssim = m["legacy_audit_global_ssim"], m["windowed_ssim"]
    fr = feats_of(ref_shared)
    fc = feats_of(cond_shared)
    feat_max = max(abs(fr[k] - fc[k]) for k in common.CONTINUOUS_FEATURES)
    peak_ok = (fr["saliency_peak_count"] == fc["saliency_peak_count"])
    hr, hc = hce_of(fr), hce_of(fc)
    dcli = abs(hr["cognitive_load_index"] - hc["cognitive_load_index"])

    pearson_ok = (p is not None and p >= T["pearson_min"])
    spearman_ok = (s is not None and s >= T["spearman_min"])
    gssim_ok = (gssim >= T["ssim_min"])
    wssim_ok = (wssim >= T["ssim_min"])
    features_ok = (feat_max <= T["feature_abs_max"])
    cli_ok = (dcli <= T["cli_abs_max"])
    finite_ok = bool(np.isfinite(cond_shared).all())
    if not pearson_ok:
        failed.append("pearson=%s" % p)
    if not spearman_ok:
        failed.append("spearman=%s" % s)
    if not gssim_ok:
        failed.append("global_ssim=%.6f" % gssim)
    if not wssim_ok:
        failed.append("windowed_ssim=%.6f" % wssim)
    if not features_ok:
        failed.append("feature_abs_max=%.4f" % feat_max)
    if not peak_ok:
        failed.append("peak_count %s vs %s" % (
            fr["saliency_peak_count"], fc["saliency_peak_count"]))
    if not cli_ok:
        failed.append("dCLI=%.4f" % dcli)
    cond_pass = (pearson_ok and spearman_ok and gssim_ok and wssim_ok
                 and features_ok and peak_ok and cli_ok and finite_ok)
    block = {
        "shared_map_metrics": m,
        "features_ref": fr, "features_cond": fc,
        "continuous_feature_abs_max": feat_max,
        "peak_count_ref": fr["saliency_peak_count"],
        "peak_count_cond": fc["saliency_peak_count"],
        "hceye_PROVISIONAL_ref": hr, "hceye_PROVISIONAL_cond": hc,
        "cognitive_load_index_abs_diff": dcli,
        "checks": {
            "pearson_ge_0999": pearson_ok, "spearman_ge_099": spearman_ok,
            "legacy_audit_global_ssim_ge_099": gssim_ok,
            "windowed_ssim_ge_099": wssim_ok,
            "continuous_features_le_002": features_ok,
            "peak_count_exact": peak_ok, "cli_le_001": cli_ok,
            "finite": finite_ok, "all_pass": cond_pass},
        "failed_checks": failed,
    }
    return block, failed, cond_pass, cond_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--modern-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    lmeta = json.load(open(os.path.join(args.legacy_dir, "legacy_causality_meta.json")))
    mmeta = json.load(open(os.path.join(args.modern_dir, "modern_causality_meta.json")))
    lnpz = np.load(os.path.join(args.legacy_dir, "legacy_causality_arrays.npz"))
    mnpz = np.load(os.path.join(args.modern_dir, "modern_causality_arrays.npz"))
    lora = np.load(os.path.join(args.legacy_dir, "legacy_oracle.npz"))
    mora = np.load(os.path.join(args.modern_dir, "modern_oracle_candidate.npz"))

    T = common.THRESHOLDS
    npz_out = {}

    # ---------- Phase 1: resize-operator oracle ----------
    print("\n============ RESIZE OPERATOR ORACLE (legacy vs candidate) ============", flush=True)
    print("%-20s %10s %12s %8s" % ("tensor", "max_abs", "shapes_match", "finite"), flush=True)
    oracle = {}
    oracle_pass = True
    for key in sorted(lora.files):
        la = lora[key].astype(np.float64)
        ca = mora[key].astype(np.float64)
        shapes_match = (la.shape == ca.shape)
        finite = bool(np.isfinite(ca).all() and np.isfinite(la).all())
        maxabs = float(np.abs(la - ca).max()) if shapes_match else float("inf")
        ok = (shapes_match and finite and maxabs <= RC.ORACLE_MAX_ABS_DIFF)
        oracle_pass = oracle_pass and ok
        oracle[key] = {"max_abs_diff": maxabs, "shapes_match": shapes_match,
                       "finite": finite, "mae": float(np.abs(la - ca).mean())
                       if shapes_match else None, "pass": ok}
        print("%-20s %10.3e %12s %8s  %s" % (
            key, maxabs, shapes_match, finite, "OK" if ok else "FAIL"), flush=True)
    print("ORACLE PASS = %s (threshold max_abs<=%.0e)" % (
        oracle_pass, RC.ORACLE_MAX_ABS_DIFF), flush=True)

    # ---------- Phase 2: byte-identical input control ----------
    print("\n============ BYTE-IDENTICAL INPUT CONTROL ============", flush=True)
    input_control = {}
    control_ok = True
    for name in common.FIXTURE_ORDER:
        lsha = lmeta["fixtures"][name]["preproc"]["array_sha256"]
        mctrl = mmeta["fixtures"][name]["legacy_controlled_input"]["array_sha256"]
        # The modern job loaded the SAME .npy; the byte-identical control diff
        # is the difference between the legacy preproc array and the tensor the
        # modern model consumed under the legacy-controlled condition.
        lctrl = lnpz["%s_preproc" % name].astype(np.float64)
        mctrl_arr = mnpz["legctrl_%s_input" % name].astype(np.float64)
        ctrl_diff = float(np.abs(lctrl - mctrl_arr).max())
        modern_vs_legacy = mmeta["fixtures"][name]["modern_vs_legacy_preproc_max_abs_diff"]
        same = (lsha == mctrl and ctrl_diff == 0.0)
        control_ok = control_ok and same
        input_control[name] = {
            "legacy_preproc_sha256": lsha,
            "modern_controlled_sha256": mctrl,
            "controlled_tensor_max_abs_diff": ctrl_diff,
            "byte_identical": same,
            "modern_vs_legacy_preproc_max_abs_diff": modern_vs_legacy}
        print("  %-12s controlled_diff=%.1e byte_identical=%s  (modern_preproc "
              "vs legacy_preproc=%.3e)" % (
                  name, ctrl_diff, same, modern_vs_legacy), flush=True)

    # ---------- Phase 3/4: conditions vs L_REF ----------
    conditions = ["M_CC", "M_LC", "M_CL", "M_LL"]
    per_fixture = {}
    cond_all_pass = {c: True for c in conditions}
    print("\n============ CONDITIONS vs L_REF (shared map) ============", flush=True)
    print("%-8s %-12s %9s %9s %9s %9s %8s %7s %6s" % (
        "cond", "fixture", "pearson", "spearman", "gSSIM", "wSSIM",
        "featMax", "dCLI", "PASS"), flush=True)
    for name in common.FIXTURE_ORDER:
        h, w = common.FIXTURES[name]["h"], common.FIXTURES[name]["w"]
        lraw = lnpz["%s_raw" % name].astype(np.float32)
        ref_shared = shared_postprocess(lraw, h, w)
        npz_out["%s_L_REF_shared" % name] = ref_shared
        per_fixture[name] = {"dims_hw": [h, w], "conditions": {}}
        for cond in conditions:
            craw = mnpz["%s_%s_raw" % (cond, name)].astype(np.float32)
            block, failed, cpass, cshared = evaluate_condition(
                ref_shared, craw, h, w, T)
            per_fixture[name]["conditions"][cond] = block
            cond_all_pass[cond] = cond_all_pass[cond] and cpass
            npz_out["%s_%s_shared" % (name, cond)] = cshared
            m = block["shared_map_metrics"]
            print("%-8s %-12s %9s %9s %9.6f %9.6f %8.4f %7.4f %6s" % (
                cond, name,
                ("%.6f" % m["pearson"]) if m["pearson"] is not None else "n/a",
                ("%.6f" % m["spearman"]) if m["spearman"] is not None else "n/a",
                m["legacy_audit_global_ssim"], m["windowed_ssim"],
                block["continuous_feature_abs_max"],
                block["cognitive_load_index_abs_diff"],
                "PASS" if cpass else "FAIL"), flush=True)

    # ---------- cumulative ablations (raw vs L_REF raw) ----------
    print("\n============ CUMULATIVE ABLATIONS (legacy-controlled input, %s raw) ============"
          % RC.BOUNDARY_FIXTURE, flush=True)
    ablation_summary = {}
    for name in common.FIXTURE_ORDER:
        h, w = common.FIXTURES[name]["h"], common.FIXTURES[name]["w"]
        lraw = lnpz["%s_raw" % name].astype(np.float32)
        ref_shared = shared_postprocess(lraw, h, w)
        ablation_summary[name] = {}
        for abl in ["abl_ups1", "abl_ups1_ups2", "abl_all"]:
            araw = mnpz["%s_%s_raw" % (abl, name)].astype(np.float32)
            ashared = shared_postprocess(araw, h, w)
            fa = full_metrics(ref_shared, ashared)
            ablation_summary[name][abl] = {
                "pearson": fa["pearson"], "windowed_ssim": fa["windowed_ssim"],
                "max_abs_diff": fa["max_abs_diff"]}
        row = ablation_summary[name]
        print("  %-12s ups1 P=%s | ups1+ups2 P=%s | all P=%s" % (
            name,
            "%.6f" % row["abl_ups1"]["pearson"],
            "%.6f" % row["abl_ups1_ups2"]["pearson"],
            "%.6f" % row["abl_all"]["pearson"]), flush=True)

    # ---------- per-resize-boundary activations (boundary fixture) ----------
    print("\n============ RESIZE-BOUNDARY ACTIVATIONS (%s) ============"
          % RC.BOUNDARY_FIXTURE, flush=True)
    lbnd = np.load(os.path.join(
        args.legacy_dir, "legacy_boundary_%s.npz" % RC.BOUNDARY_FIXTURE))
    bnames = [str(x) for x in lbnd["__names__"]]
    boundary = {}
    for tag in ["M_LC_prod", "abl_ups1", "abl_ups1_ups2", "abl_all"]:
        p = os.path.join(args.modern_dir,
                         "modern_boundary_%s_%s.npz" % (tag, RC.BOUNDARY_FIXTURE))
        if not os.path.exists(p):
            continue
        mb = np.load(p)
        boundary[tag] = {}
        print("  --- variant %s ---" % tag, flush=True)
        for bn in bnames:
            la = lbnd[bn].astype(np.float64).ravel()
            ca = mb[bn].astype(np.float64).ravel()
            if la.shape != ca.shape:
                boundary[tag][bn] = {"pearson": None, "max_abs_diff": None,
                                     "shape_mismatch": True}
                continue
            per = (float(pearsonr(la, ca)[0])
                   if la.std() > 0 and ca.std() > 0 else None)
            mad = float(np.abs(la - ca).max())
            boundary[tag][bn] = {"pearson": per, "max_abs_diff": mad}
            print("    %-16s pearson=%s max_abs=%.4g" % (
                bn, ("%.6f" % per) if per is not None else "n/a", mad),
                flush=True)

    # first boundary at which M_LC_prod (all-modern resize) diverges from legacy
    first_div = None
    if "M_LC_prod" in boundary:
        for bn in bnames:
            per = boundary["M_LC_prod"][bn].get("pearson")
            if per is not None and per < T["pearson_min"]:
                first_div = bn
                break

    # ---------- DECISION LOGIC ----------
    m_lc_fails = not cond_all_pass["M_LC"]
    m_ll_passes = cond_all_pass["M_LL"]
    m_cl_passes = cond_all_pass["M_CL"]
    # activation demonstrates resize removes divergence: all-modern diverges at
    # a resize output, and abl_all matches legacy at dec_c_cout.
    abl_all_final_ok = (boundary.get("abl_all", {})
                        .get("dec_c_cout", {}).get("pearson") is not None
                        and boundary["abl_all"]["dec_c_cout"]["pearson"]
                        >= T["pearson_min"])
    activation_supports = (first_div is not None
                           and first_div.startswith("dec_ups")
                           and abl_all_final_ok)

    if oracle_pass and m_lc_fails and m_ll_passes and activation_supports:
        verdict = "UMSI RESIZE CAUSE CONFIRMED — CONTROLLED PARITY RESTORED"
    elif (m_ll_passes is False) and any(
            per_fixture[n]["conditions"]["M_LL"]["shared_map_metrics"]["pearson"]
            is not None and
            per_fixture[n]["conditions"]["M_LL"]["shared_map_metrics"]["pearson"]
            > per_fixture[n]["conditions"]["M_LC"]["shared_map_metrics"]["pearson"]
            for n in common.FIXTURE_ORDER):
        verdict = "UMSI RESIZE CAUSE PARTIAL — FURTHER READ-ONLY DIAGNOSIS REQUIRED"
    else:
        verdict = "UMSI RESIZE CAUSE NOT CONFIRMED — FURTHER READ-ONLY DIAGNOSIS REQUIRED"

    manifest = {}
    if args.manifest and os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest))

    result = {
        "experiment": "umsi_resize_causality",
        "audit_commit": os.environ.get("AUDIT_COMMIT", ""),
        "frozen_manifest": manifest,
        "environments": {"legacy": lmeta["environment"],
                         "modern": mmeta["environment"]},
        "checkpoint": mmeta["checkpoint"],
        "model_info": mmeta["model_info"],
        "thresholds": T,
        "windowed_ssim_params": WINDOWED_SSIM_PARAMS,
        "phase1_resize_operator_oracle": {"per_tensor": oracle,
                                          "oracle_pass": oracle_pass},
        "phase2_byte_identical_input_control": {
            "per_fixture": input_control, "all_byte_identical": control_ok},
        "phase3_conditions_vs_L_REF": per_fixture,
        "phase3_condition_all_pass": cond_all_pass,
        "phase3_cumulative_ablations": ablation_summary,
        "phase3_resize_boundary_activations": boundary,
        "first_divergent_boundary_all_modern_resize": first_div,
        "decision": {
            "oracle_pass": oracle_pass,
            "M_LC_reproduces_failure": m_lc_fails,
            "M_LL_passes_all_fixtures": m_ll_passes,
            "M_CL_passes_all_fixtures": m_cl_passes,
            "activation_supports_hypothesis": activation_supports,
            "abl_all_final_layer_matches_legacy": abl_all_final_ok},
        "verdict": verdict,
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    np.savez_compressed(args.out_npz, **npz_out)

    print("\n================ DECISION ================", flush=True)
    print("oracle_pass                 :", oracle_pass, flush=True)
    print("M_LC reproduces failure     :", m_lc_fails, flush=True)
    print("M_LL passes all 5 fixtures  :", m_ll_passes, flush=True)
    print("M_CL passes all 5 fixtures  :", m_cl_passes, flush=True)
    print("first divergent boundary    :", first_div, flush=True)
    print("activation supports         :", activation_supports, flush=True)
    print("\nVERDICT:", verdict, flush=True)
    print("\nSAVED %s" % args.out_json, flush=True)
    print("SAVED %s" % args.out_npz, flush=True)


if __name__ == "__main__":
    main()
