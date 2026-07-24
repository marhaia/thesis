"""Evidence consolidator for the UMSI++ resize causality experiment.

EVIDENCE CONSOLIDATION ONLY. This script performs NO model inference: it never
imports TensorFlow/Keras and never instantiates UMSI++. It reads the immutable
Actions artifacts of the source run (30034973775), corrects three documented
evidence-packaging defects, and emits ONE self-contained corrected bundle.

Defects corrected (see the report and the correction_history block):
  1. shared_map_metrics.sha256 stored the L_REF *reference* array hash instead
     of the corresponding *condition* array hash. This module recomputes both
     hashes directly from the stored arrays and replaces the single ambiguous
     field with reference_array_data_sha256 + condition_array_data_sha256.
  2. Raw (512x512) per-condition tensor metrics live only in the intermediate
     legacy/modern meta artifacts; they are consolidated here.
  3. Complete pip freeze records (from the job logs) and executed-source SHA-256
     (upstream UMSI source from the legacy meta + audit/production Python files
     as immutable git objects at the audit commit) are consolidated here.

The scientific values are NOT recomputed to replace the originals. Every metric
is copied from the immutable source artifacts and ALSO independently recomputed
from the stored arrays with the exact original comparison implementation
(golden.causality_compare) purely as an integrity check; both values plus their
absolute difference and a consistency flag are stored.

Inputs (environment):
  RC_SRC_DIR      directory with extracted artifacts:
                    manifest/ legacy/ modern/ final/
  RC_LOG_DIR      directory with the 4 raw Actions job-step logs (*.txt)
  RC_ARTIFACT_META optional JSON: [{id,name,archive_sha256,archive_bytes,
                    expires_at,source_job}] for archive-level provenance
  UMSI_REPO       repository root at the audit commit (feature/HCEye code +
                    git source hashing); no model is loaded from it
  AUDIT_COMMIT    a03c42a...  (immutable experiment commit)
  SOURCE_RUN_ID   30034973775
  RC_OUT_JSON / RC_OUT_NPZ  output paths
"""
import os
import sys
import json
import glob
import hashlib
import subprocess
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common          # noqa: E402
import resize_common as RC   # noqa: E402
# The exact original comparison implementation (numpy/scipy/cv2 + repo
# feature/HCEye code only; NO TensorFlow, NO UMSI model). Reused for the
# integrity recompute so the corrected bundle is checked against itself.
import causality_compare as CC   # noqa: E402

SRC = os.environ["RC_SRC_DIR"]
LOGDIR = os.environ["RC_LOG_DIR"]
REPO = os.environ["UMSI_REPO"]
AUDIT_COMMIT = os.environ["AUDIT_COMMIT"]
SOURCE_RUN_ID = os.environ["SOURCE_RUN_ID"]
OUT_JSON = os.environ["RC_OUT_JSON"]
OUT_NPZ = os.environ["RC_OUT_NPZ"]
ARTIFACT_META = os.environ.get("RC_ARTIFACT_META", "")

FIXTURES = ["lowcontrast", "ui1", "ui2", "ui3", "uniform"]
CONDS = ["M_CC", "M_CL", "M_LC", "M_LL"]
ALL_MAPS = ["L_REF"] + CONDS
BOUND_NAMES = ["dec_ups1__in", "dec_ups1__out", "dec_ups2__in", "dec_ups2__out",
               "dec_ups3__in", "dec_ups3__out", "dec_c_cout"]
BOUND_TAGS = ["M_LC_prod", "abl_all", "abl_ups1", "abl_ups1_ups2"]

# The Python source files executed by each job, tracked in the immutable audit
# commit. Their SHA-256 are read from git objects at AUDIT_COMMIT (not the
# working tree) so they reflect exactly what CI checked out and ran.
AUDIT_EXECUTED_SOURCE = {
    "legacy_job": [
        "golden/common.py", "golden/resize_common.py", "golden/windowed_ssim.py",
        "golden/legacy_causality.py", "golden/legacy_runner.py",
    ],
    "modern_job": [
        "golden/common.py", "golden/resize_common.py", "golden/windowed_ssim.py",
        "golden/shared_postprocess.py", "golden/audit_resize_model.py",
        "golden/modern_causality.py", "saliency/umsi_model.py",
    ],
    "compare_job": [
        "golden/common.py", "golden/resize_common.py", "golden/windowed_ssim.py",
        "golden/shared_postprocess.py", "golden/causality_compare.py",
        "golden/verify_causality.py",
        "saliency/saliency_features.py", "hceye/hceye_features.py",
    ],
    "manifest_job": ["golden/manifest_resize.py", "golden/resize_common.py",
                     "golden/windowed_ssim.py", "golden/common.py"],
}

LOG = []


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG.append(line)


# ---------------------------------------------------------------- hash contract
def sha_file(p):
    return common.sha256_file(p)


def array_data_sha256(a):
    """SHA-256 of the UNCHANGED, C-contiguous array bytes (no cast/reorder).
    For the float32 shared maps this coincides with the original
    common.sha256_array (which casts an already-float32 array to float32)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def arr_desc(a):
    return {
        "shape": list(a.shape),
        "dtype_str": a.dtype.str,          # e.g. '<f4' (endianness + kind + size)
        "byteorder": a.dtype.byteorder,     # '<', '>', '=' or '|'
        "array_data_sha256": array_data_sha256(a),
    }


def load_npz(path):
    z = np.load(path, allow_pickle=False)   # rejects pickled object arrays
    out = {}
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "O":
            raise ValueError("object array rejected: %s in %s" % (k, path))
        out[k] = a
    return out


def git_blob_sha(relpath):
    """SHA-256 of the file's content as stored in the immutable audit commit."""
    blob = subprocess.check_output(
        ["git", "-C", REPO, "show", "%s:%s" % (AUDIT_COMMIT, relpath)])
    return hashlib.sha256(blob).hexdigest(), len(blob)


# ---------------------------------------------------------------- source loading
log("=" * 78)
log("UMSI++ RESIZE CAUSALITY — EVIDENCE CONSOLIDATION")
log("source_run_id =", SOURCE_RUN_ID, " audit_commit =", AUDIT_COMMIT)
log("generated_utc =", datetime.datetime.utcnow().isoformat() + "Z")
log("=" * 78)

paths = {
    "manifest_json": SRC + "/manifest/resize_causality_manifest.json",
    "legacy_meta": SRC + "/legacy/legacy_causality_meta.json",
    "modern_meta": SRC + "/modern/modern_causality_meta.json",
    "final_json": SRC + "/final/umsi_resize_causality_6a17288.json",
    "final_arrays": SRC + "/final/umsi_resize_causality_arrays_6a17288.npz",
    "legacy_arrays": SRC + "/legacy/legacy_causality_arrays.npz",
    "modern_arrays": SRC + "/modern/modern_causality_arrays.npz",
    "legacy_oracle": SRC + "/legacy/legacy_oracle.npz",
    "candidate_oracle": SRC + "/modern/modern_oracle_candidate.npz",
    "oracle_inputs": SRC + "/legacy/legacy_oracle_inputs.npz",
    "legacy_boundary": SRC + "/legacy/legacy_boundary_ui1.npz",
}
for t in BOUND_TAGS:
    paths["mbnd_" + t] = SRC + "/modern/modern_boundary_%s_ui1.npz" % t

manifest = json.load(open(paths["manifest_json"]))
lmeta = json.load(open(paths["legacy_meta"]))
mmeta = json.load(open(paths["modern_meta"]))
final = json.load(open(paths["final_json"]))

log("\n[load] final arrays + intermediate arrays (allow_pickle=False)")
final_arr = load_npz(paths["final_arrays"])
legacy_arr = load_npz(paths["legacy_arrays"])
modern_arr = load_npz(paths["modern_arrays"])
legacy_oracle = load_npz(paths["legacy_oracle"])
candidate_oracle = load_npz(paths["candidate_oracle"])
oracle_inputs = load_npz(paths["oracle_inputs"])
legacy_bnd = load_npz(paths["legacy_boundary"])
modern_bnd = {t: load_npz(paths["mbnd_" + t]) for t in BOUND_TAGS}
log("[load] OK — no object arrays present in any NPZ")


# ---------------------------------------------------- 1. source artifact inventory
def inventory():
    inv = {}
    for art in ["manifest", "legacy", "modern", "final"]:
        files = []
        for f in sorted(glob.glob(SRC + "/" + art + "/*")):
            if os.path.isfile(f):
                files.append({"filename": os.path.basename(f),
                              "bytes": os.path.getsize(f),
                              "sha256": sha_file(f)})
        inv[art] = files
    return inv


artifact_files = inventory()
log("\n[inventory] extracted-file provenance:")
for art, files in artifact_files.items():
    for fdesc in files:
        log("   %-8s %-42s %10d  %s" % (art, fdesc["filename"], fdesc["bytes"],
                                        fdesc["sha256"]))

archive_meta = json.load(open(ARTIFACT_META)) if ARTIFACT_META and \
    os.path.exists(ARTIFACT_META) else None

# job-step logs
job_logs = []
for f in sorted(glob.glob(LOGDIR + "/*.txt")):
    job_logs.append({"filename": os.path.basename(f),
                     "bytes": os.path.getsize(f),
                     "sha256": sha_file(f)})
log("\n[inventory] source job-step logs:")
for j in job_logs:
    log("   %-58s %9d  %s" % (j["filename"], j["bytes"], j["sha256"]))


# ------------------------------------------------------------ 2. pip freeze parse
def extract_pip_freeze(logfile):
    """Return the ordered list of 'name==version' lines emitted by the job's
    `python -m pip freeze` block (pure freeze output, not 'Collecting' lines)."""
    txt = open(logfile, encoding="utf-8", errors="replace").read().splitlines()
    stripped = []
    for ln in txt:
        # drop the leading ISO timestamp that Actions prepends
        parts = ln.split(" ", 1)
        stripped.append(parts[1] if len(parts) == 2 and
                        parts[0][:4].isdigit() and "T" in parts[0] else ln)
    freeze = []
    started = False
    for ln in stripped:
        s = ln.strip()
        if s == "----- pip freeze -----":
            started = True
            continue
        if not started:
            continue
        if "==" in s and " " not in s and not s.startswith("Collecting") \
                and not s.startswith("#") and "/" not in s:
            name = s.split("==")[0]
            if name and all(c.isalnum() or c in "-_.[]" for c in name):
                freeze.append(s)
        elif freeze and (s == "" or "----" in s or s.startswith("+ ")
                         or s.startswith("=====") or "python" in s.lower()):
            break
    return freeze


LEGACY_LOG = LOGDIR + "/2_Legacy TF1.14_Keras2.3.1 (L_REF + inputs + oracle).txt"
MODERN_LOG = LOGDIR + "/1_Modern TF2.16_Keras3 (conditions + ablations).txt"
COMPARE_LOG = LOGDIR + "/0_Compare + causality verdict + evidence.txt"
legacy_freeze = extract_pip_freeze(LEGACY_LOG)
modern_freeze = extract_pip_freeze(MODERN_LOG)
compare_freeze = extract_pip_freeze(COMPARE_LOG)
log("\n[pip freeze] legacy=%d modern=%d compare=%d packages" % (
    len(legacy_freeze), len(modern_freeze), len(compare_freeze)))
assert len(legacy_freeze) >= 20, "legacy pip freeze incomplete"
assert len(modern_freeze) >= 20, "modern pip freeze incomplete"


# ---------------------------------------------------- 3. executed source hashes
executed_source = {"upstream_umsi_model_source": lmeta["executed_source_sha256"],
                   "upstream_provenance": "legacy_causality_meta.json "
                   "(immutable legacy artifact); SHA-256 of the cloned "
                   "YueJiang-nj/UEyes-CHI2023 UMSI++ builder source",
                   "audit_repo_source_git_objects": {}}
log("\n[source hashes] audit/production Python files (git objects @ %s):"
    % AUDIT_COMMIT[:12])
for job, files in AUDIT_EXECUTED_SOURCE.items():
    d = {}
    for rel in files:
        sha, n = git_blob_sha(rel)
        d[rel] = {"sha256": sha, "bytes": n}
        log("   %-13s %-42s %8d  %s" % (job, rel, n, sha))
    executed_source["audit_repo_source_git_objects"][job] = d


# ---------------------------------------------- consolidated NPZ + array index
npz_out = {}
array_index = []


def add_array(cons_key, a, source_artifact, source_file, source_key):
    if cons_key in npz_out:
        raise KeyError("duplicate consolidated key: %s" % cons_key)
    npz_out[cons_key] = np.ascontiguousarray(a)
    array_index.append({
        "consolidated_key": cons_key,
        "source_artifact": source_artifact,
        "source_filename": source_file,
        "source_key": source_key,
        "shape": list(a.shape),
        "dtype_str": a.dtype.str,
        "source_array_data_sha256": array_data_sha256(a),
        # consolidated hash filled in after reload (verified equal)
    })


# shared postprocessed maps (5 fixtures x {L_REF + 4 conditions})
for fx in FIXTURES:
    for cond in ALL_MAPS:
        k = "%s_%s_shared" % (fx, cond)
        add_array("shared/%s_%s" % (fx, cond), final_arr[k], "final",
                  "umsi_resize_causality_arrays_6a17288.npz", k)
# legacy L_REF raw pipeline arrays
for k, a in legacy_arr.items():
    add_array("legacy_raw/%s" % k, a, "legacy",
              "legacy_causality_arrays.npz", k)
# modern raw arrays for every condition + ablation (85 keys)
for k, a in modern_arr.items():
    add_array("modern_raw/%s" % k, a, "modern",
              "modern_causality_arrays.npz", k)
# boundary activations (legacy + 4 modern variants)
for nm in BOUND_NAMES:
    add_array("boundary_legacy/%s" % nm, legacy_bnd[nm], "legacy",
              "legacy_boundary_ui1.npz", nm)
for t in BOUND_TAGS:
    for nm in BOUND_NAMES:
        add_array("boundary_%s/%s" % (t, nm), modern_bnd[t][nm], "modern",
                  "modern_boundary_%s_ui1.npz" % t, nm)
# oracle arrays
for k, a in legacy_oracle.items():
    add_array("oracle_legacy/%s" % k, a, "legacy", "legacy_oracle.npz", k)
for k, a in candidate_oracle.items():
    add_array("oracle_candidate/%s" % k, a, "modern",
              "modern_oracle_candidate.npz", k)
for k, a in oracle_inputs.items():
    add_array("oracle_inputs/%s" % k, a, "legacy", "legacy_oracle_inputs.npz", k)

log("\n[arrays] consolidated %d arrays into the NPZ" % len(npz_out))
np.savez_compressed(OUT_NPZ, **npz_out)

# reload consolidated NPZ and verify every array hash round-trips exactly
recon = load_npz(OUT_NPZ)
mismatches = 0
for entry in array_index:
    ck = entry["consolidated_key"]
    ch = array_data_sha256(recon[ck])
    entry["consolidated_array_data_sha256"] = ch
    entry["hash_roundtrip_ok"] = bool(ch == entry["source_array_data_sha256"])
    if not entry["hash_roundtrip_ok"]:
        mismatches += 1
        log("   !! HASH MISMATCH", ck)
log("[arrays] hash round-trip verified: %d/%d exact, %d mismatch"
    % (len(array_index) - mismatches, len(array_index), mismatches))
assert mismatches == 0, "consolidated NPZ array hash mismatch"


# ----------------------------------- 1) CORRECT shared_map_metrics.sha256 schema
correction_history = []
phase3 = json.loads(json.dumps(final["phase3_conditions_vs_L_REF"]))  # deep copy
integrity_shared = {}

for fx in FIXTURES:
    ref = final_arr["%s_L_REF_shared" % fx]
    ref_h = array_data_sha256(ref)
    integrity_shared[fx] = {}
    for cond in CONDS:
        cond_arr = final_arr["%s_%s_shared" % (fx, cond)]
        cond_h = array_data_sha256(cond_arr)
        blk = phase3[fx]["conditions"][cond]["shared_map_metrics"]
        orig = blk.pop("sha256")
        blk["reference_array_data_sha256"] = ref_h
        blk["condition_array_data_sha256"] = cond_h
        blk["reference_array_key"] = "shared/%s_L_REF" % fx
        blk["condition_array_key"] = "shared/%s_%s" % (fx, cond)
        blk["bitwise_equal_to_reference"] = bool(cond_h == ref_h)
        correction_history.append({
            "defect": "shared_map_metrics.sha256 held the L_REF reference hash "
                      "instead of the condition array hash",
            "field_path": "phase3_conditions_vs_L_REF.%s.conditions.%s."
                          "shared_map_metrics.sha256" % (fx, cond),
            "original_recorded_value": orig,
            "original_value_equals_reference_hash": bool(orig == ref_h),
            "corrected_reference_array_data_sha256": ref_h,
            "corrected_condition_array_data_sha256": cond_h,
            "source_npz": "final/umsi_resize_causality_arrays_6a17288.npz",
            "reference_array_key": "%s_L_REF_shared" % fx,
            "condition_array_key": "%s_%s_shared" % (fx, cond),
            "verified_against_consolidated_npz": bool(
                array_data_sha256(recon["shared/%s_%s" % (fx, cond)]) == cond_h
                and array_data_sha256(recon["shared/%s_L_REF" % fx]) == ref_h),
        })

n_corr = len(correction_history)
all_orig_were_ref = all(c["original_value_equals_reference_hash"]
                        for c in correction_history)
all_corr_verified = all(c["verified_against_consolidated_npz"]
                        for c in correction_history)
log("\n[correction] shared_map sha256: %d fields corrected; "
    "every original == reference hash: %s; every correction verified: %s"
    % (n_corr, all_orig_were_ref, all_corr_verified))
assert all_orig_were_ref and all_corr_verified


# ------------------------------------ integrity recompute of shared-map metrics
def close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


for fx in FIXTURES:
    ref = final_arr["%s_L_REF_shared" % fx]
    for cond in CONDS:
        cond_arr = final_arr["%s_%s_shared" % (fx, cond)]
        rec = CC.full_metrics(ref, cond_arr)   # exact original implementation
        src = final["phase3_conditions_vs_L_REF"][fx]["conditions"][cond][
            "shared_map_metrics"]
        checks = {}
        for key, tol in [("pearson", 1e-9), ("spearman", 1e-9),
                         ("mae", 1e-9), ("rmse", 1e-9), ("max_abs_diff", 1e-6),
                         ("legacy_audit_global_ssim", 1e-9),
                         ("windowed_ssim", 1e-9)]:
            checks[key] = {
                "source_recorded": src.get(key),
                "recomputed": rec.get(key),
                "abs_diff": (None if src.get(key) is None or rec.get(key) is None
                             else abs(float(src[key]) - float(rec[key]))),
                "consistent": close(src.get(key), rec.get(key), tol),
            }
        integrity_shared[fx][cond] = checks

incons = [(fx, c, k) for fx in integrity_shared for c in integrity_shared[fx]
          for k in integrity_shared[fx][c]
          if not integrity_shared[fx][c][k]["consistent"]]
log("[integrity] shared-map metric recompute: %d inconsistencies"
    % len(incons))
for t in incons:
    log("   !! inconsistent", t)
assert not incons, "integrity recompute disagreed with source-recorded metrics"


# --------------------------------------------- 2) raw-tensor metrics (defect #2)
raw_metrics = {}
for fx in FIXTURES:
    leg_raw = legacy_arr["%s_raw" % fx]           # L_REF raw (512,512)
    leg_stats = lmeta["fixtures"][fx]["raw"]
    fxout = {"L_REF_raw_stats": leg_stats,
             "L_REF_raw_array_data_sha256": array_data_sha256(leg_raw),
             "conditions": {}}
    for cond in CONDS:
        craw = modern_arr["%s_%s_raw" % (cond, fx)]
        cstats = mmeta["fixtures"][fx]["conditions"][cond]["raw"]
        fa = leg_raw.ravel().astype(np.float64)
        fb = craw.ravel().astype(np.float64)
        d = fa - fb
        cmp_vs_legacy = {
            "pearson": (float(CC.pearsonr(fa, fb)[0])
                        if fa.std() > 0 and fb.std() > 0 else None),
            "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "max_abs_diff": float(np.abs(d).max()),
        }
        fxout["conditions"][cond] = {
            "input": mmeta["fixtures"][fx]["conditions"][cond]["input"],
            "resize": mmeta["fixtures"][fx]["conditions"][cond]["resize"],
            "raw_stats_source_recorded": cstats,
            "raw_array_data_sha256": array_data_sha256(craw),
            "raw_512x512_vs_L_REF_raw": cmp_vs_legacy,
        }
    raw_metrics[fx] = fxout
log("[raw] consolidated raw 512x512 tensor metrics for %d fixtures x %d conditions"
    % (len(FIXTURES), len(CONDS)))


# --------------------------------------------------------- 3) repeatability
repeatability = {"L_REF_legacy": {}, "modern_M_CC_M_LL": {}}
for fx in FIXTURES:
    repeatability["L_REF_legacy"][fx] = {
        "same_process": lmeta["fixtures"][fx]["same_process"],
        "fresh_process": lmeta["fixtures"][fx]["fresh_process"],
    }
    repeatability["modern_M_CC_M_LL"][fx] = {
        "same_process": mmeta["fixtures"][fx]["same_process"],
        "fresh_process": mmeta["fixtures"][fx]["fresh_process"],
    }
# summary
rep_ok = True
for fx in FIXTURES:
    if lmeta["fixtures"][fx]["same_process"]["max_abs_diff"] != 0.0:
        rep_ok = False
    if lmeta["fixtures"][fx]["same_process"]["unique_hashes"] != 1:
        rep_ok = False
    if lmeta["fixtures"][fx]["fresh_process"]["unique_hashes_incl_main"] != 1:
        rep_ok = False
    sp = mmeta["fixtures"][fx]["same_process"]
    if sp["M_CC_max_abs_diff"] != 0.0 or sp["M_LL_max_abs_diff"] != 0.0:
        rep_ok = False
    if sp["M_CC_unique"] != 1 or sp["M_LL_unique"] != 1:
        rep_ok = False
    for c in ("M_CC", "M_LL"):
        if mmeta["fixtures"][fx]["fresh_process"][c][
                "unique_hashes_incl_main"] != 1:
            rep_ok = False
repeatability["all_deterministic"] = bool(rep_ok)
log("[repeat] L_REF/M_CC/M_LL same+fresh process determinism (all runs unique=1,"
    " max_abs_diff=0.0): %s" % rep_ok)


# ------------------------------------------------ oracle integrity recompute
oracle_recompute = {}
oracle_ok = True
for k in sorted(legacy_oracle):
    lg, cd = legacy_oracle[k], candidate_oracle[k]
    d = np.abs(lg.astype(np.float64) - cd.astype(np.float64))
    mad = float(d.max())
    finite = bool(np.isfinite(cd).all() and np.isfinite(lg).all())
    shapes = list(lg.shape) == list(cd.shape)
    src = final["phase1_resize_operator_oracle"]["per_tensor"][k]
    oracle_recompute[k] = {
        "source_recorded_max_abs_diff": src["max_abs_diff"],
        "recomputed_max_abs_diff": mad,
        "shapes_match": shapes,
        "finite": finite,
        "legacy_array_data_sha256": array_data_sha256(lg),
        "candidate_array_data_sha256": array_data_sha256(cd),
        "bitwise_identical": bool(array_data_sha256(lg) == array_data_sha256(cd)),
        "pass_le_1e6": bool(shapes and finite and mad <= RC.ORACLE_MAX_ABS_DIFF),
        "consistent_with_source": close(src["max_abs_diff"], mad, 1e-12),
    }
    oracle_ok = oracle_ok and oracle_recompute[k]["pass_le_1e6"] \
        and oracle_recompute[k]["consistent_with_source"]
log("[oracle] recompute over %d tensors: all pass<=1e-6 & consistent: %s "
    "(all bitwise_identical: %s)"
    % (len(oracle_recompute), oracle_ok,
       all(v["bitwise_identical"] for v in oracle_recompute.values())))
assert oracle_ok


# ------------------------------------------------------------ assemble JSON
verdict = final["verdict"]
decision = final["decision"]
# corrected wording distinction
abl_all_maxabs = max(final["phase3_cumulative_ablations"][fx]["abl_all"][
    "max_abs_diff"] for fx in FIXTURES)
any_cond_bitwise = any(
    correction_history[i]["corrected_condition_array_data_sha256"] ==
    correction_history[i]["corrected_reference_array_data_sha256"]
    for i in range(len(correction_history)))

consolidated = {
    "experiment": "umsi_resize_causality_evidence_consolidation",
    "consolidation_of_source_run_id": SOURCE_RUN_ID,
    "source_run_url":
        "https://github.com/marhaia/thesis/actions/runs/" + SOURCE_RUN_ID,
    "source_branch": "audit/umsi-resize-causality",
    "source_commit": final["audit_commit"],
    "consolidation_branch": "audit/umsi-resize-evidence-consolidation",
    "audit_base_commit": manifest["audit_base_commit"],
    "prior_golden_run": manifest.get("prior_golden_run"),
    "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",

    "scientific_verdict": verdict,
    "scientific_verdict_unchanged": True,
    "bitwise_equal": False,
    "numerical_parity_within_preregistered_thresholds": True,
    "max_abl_all_shared_map_abs_diff": abl_all_maxabs,
    "any_passing_condition_bitwise_equal_to_reference": bool(any_cond_bitwise),
    "causal_conclusion": (
        "The modern resize semantics were the sole material cause of the "
        "pre-registered golden-threshold failures in this controlled "
        "experiment."),
    "causal_conclusion_scope": (
        "This is a controlled resize-operator causality result, NOT a claim "
        "that the resize is the only possible difference anywhere in the "
        "complete pipeline."),

    "decision": decision,
    "first_divergent_boundary_all_modern_resize":
        final["first_divergent_boundary_all_modern_resize"],

    "frozen_manifest": manifest,
    "thresholds": final["thresholds"],
    "thresholds_match_frozen_manifest": bool(
        all(final["thresholds"].get(k) == manifest[
            "pre_registered_thresholds"].get(k)
            for k in final["thresholds"])),
    "windowed_ssim_params": final["windowed_ssim_params"],

    "environments": {
        "legacy": lmeta["environment"],
        "modern": mmeta["environment"],
        "containers": {
            "legacy_image": manifest["containers"]["legacy_image"],
            "legacy_image_digest": manifest["containers"]["legacy_image_digest"],
            "modern_image": manifest["containers"]["modern_image"],
            "modern_image_digest": manifest["containers"]["modern_image_digest"],
        },
    },
    "pip_freeze": {
        "legacy_tf1_keras2": legacy_freeze,
        "modern_tf2_keras3": modern_freeze,
        "compare_env": compare_freeze,
        "provenance": "python -m pip freeze blocks parsed from the immutable "
                      "Actions job-step logs",
    },
    "executed_source_sha256": executed_source,
    "checkpoint": final["checkpoint"],
    "model_info": {**final.get("model_info", {}),
                   "variants": mmeta["model_info"]},
    "captured_warnings_modern": mmeta.get("captured_warnings", []),

    "phase1_resize_operator_oracle": final["phase1_resize_operator_oracle"],
    "phase1_oracle_integrity_recompute": oracle_recompute,
    "phase2_byte_identical_input_control":
        final["phase2_byte_identical_input_control"],

    "phase3_conditions_vs_L_REF": phase3,          # corrected hash schema
    "phase3_condition_all_pass": final["phase3_condition_all_pass"],
    "phase3_shared_map_metric_integrity_recompute": integrity_shared,
    "phase3_raw_tensor_metrics": raw_metrics,       # defect #2 consolidated
    "phase3_cumulative_ablations": final["phase3_cumulative_ablations"],
    "phase3_resize_boundary_activations":
        final["phase3_resize_boundary_activations"],
    "repeatability": repeatability,                 # L_REF/M_CC/M_LL

    "source_artifact_inventory": {
        "extracted_files": artifact_files,
        "archive_level": archive_meta,
        "job_step_logs": job_logs,
    },
    "consolidated_array_index": array_index,
    "correction_history": correction_history,

    "limitations": [
        "x86_64 CPU source-matched legacy numerical golden run; NOT a claim of "
        "bitwise TF1.14/TF2.16 or CUDA-9 GPU reproduction.",
        "bitwise_equal is false: passing conditions (M_CL, M_LL) match L_REF as "
        "numerical parity within the pre-registered thresholds (max abl_all "
        "shared-map abs diff = %.3e), not as identical bytes." % abl_all_maxabs,
        "HCEye outputs and cognitive_load_index are PROVISIONAL "
        "(computed against feature norms that predate the corrected saliency "
        "map interpretation).",
        "This bundle is evidence consolidation only: no model inference was run "
        "during consolidation; no scientific value was recomputed to replace "
        "the source-recorded originals.",
        "The causal conclusion is scoped to the controlled resize operator; it "
        "does not assert the resize is the only possible pipeline difference.",
    ],
    "consolidation_no_model_inference": True,
}

with open(OUT_JSON, "w") as fh:
    json.dump(consolidated, fh, indent=2, sort_keys=True,
              allow_nan=False)     # strict: no NaN/Infinity tokens
log("\n[write] %s (%d bytes)" % (OUT_JSON, os.path.getsize(OUT_JSON)))
log("[write] %s (%d bytes)" % (OUT_NPZ, os.path.getsize(OUT_NPZ)))
log("\nSCIENTIFIC VERDICT (unchanged):", verdict)
log("bitwise_equal = False ; numerical_parity_within_preregistered_thresholds = True")
log("=" * 78)
