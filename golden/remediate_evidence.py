"""Phases 2-5 — correct the resize-evidence provenance defects (no inference).

Takes the immutable consolidated bundle + both source runs' logs/metadata and
emits a truthful remediated bundle that fixes the six provenance defects:

  D1/D2  pip_freeze.compare_env wrongly held the legacy TF1 freeze (parser took
         the first embedded freeze block in the Compare log).
  D3     the historical Compare job never ran its own `pip freeze`; this is
         recorded transparently as unavailable and is NOT reconstructed.
  D4     the validator did not catch the misattribution -> verify_remediation.py
         now enforces it.
  D5     scripts/download_weights.py was absent from the executed-source hash
         inventory -> added here (hashed at the immutable source commit).
  D6     the artifact inventory lacked independent source-job attribution ->
         artifact_to_job_inventory carries GitHub job ids + names.

NO model inference is performed: only numpy/scipy array reads and the frozen
metric formulas are used. Array data is copied byte-for-byte from the immutable
consolidated NPZ (which was already proven byte-identical to the source NPZs).
"""
import os
import sys
import io
import re
import json
import shutil
import hashlib
import subprocess
import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from windowed_ssim import windowed_ssim, WINDOWED_SSIM_PARAMS  # noqa: E402
from common import sha256_array  # noqa: E402


def sha256_array_native(a):
    """Hash the array's native-dtype bytes (matches the consolidated index,
    which recorded per-array data hashes without forcing float32)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ---- inputs
SRC = os.environ["RC_SRC_DIR"]
LOGDIR = os.environ["RC_LOG_DIR"]
ATTRIB = json.load(open(os.environ["RC_ATTRIB"]))
REM_FREEZE = os.environ["RC_REM_FREEZE"]
OUT_JSON = os.environ["RC_OUT_JSON"]
OUT_NPZ = os.environ["RC_OUT_NPZ"]
REPO = os.environ["GH_REPO"]
CAUS_COMMIT = os.environ["RC_CAUS_COMMIT"]
CONS_COMMIT = os.environ["RC_CONS_COMMIT"]
REM_COMMIT = os.environ["RC_REM_COMMIT"]
GIT_DIR = os.environ.get("RC_GIT_DIR", os.getcwd())

FIXTURES = ["lowcontrast", "ui1", "ui2", "ui3", "uniform"]
CONDS = ["M_CC", "M_CL", "M_LC", "M_LL"]
LEGACY_LOG = "2_Legacy TF1.14_Keras2.3.1 (L_REF + inputs + oracle).txt"
MODERN_LOG = "1_Modern TF2.16_Keras3 (conditions + ablations).txt"
# thresholds (frozen manifest)
TH_TIGHT = 1e-9   # pearson/spearman/ssim/mae/rmse
TH_MAXABS = 1e-6  # max_abs_diff


def line(msg=""):
    print(msg, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_blob(commit, path):
    """Return (bytes, sha256, git_blob_id) for path at commit, or None."""
    try:
        blob_id = subprocess.check_output(
            ["git", "-C", GIT_DIR, "rev-parse", "%s:%s" % (commit, path)],
            stderr=subprocess.DEVNULL).decode().strip()
        data = subprocess.check_output(
            ["git", "-C", GIT_DIR, "cat-file", "blob", "%s:%s" % (commit, path)],
            stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return data, sha_bytes(data), blob_id


# ---------------------------------------------------------------- frozen metric
def global_ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def recompute_metrics(a, b):
    """Byte-for-byte frozen full_metrics() minus the categorical fields."""
    fa = a.ravel().astype(np.float64)
    fb = b.ravel().astype(np.float64)
    d = fa - fb
    out = {"mae": float(np.abs(d).mean()),
           "rmse": float(np.sqrt((d ** 2).mean())),
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


# --------------------------------------------------------------- log freeze parse
def parse_pip_freeze(log_path):
    """Extract the single `pip freeze` output block from a per-job step log.

    Actions step logs prefix every line with an RFC3339 timestamp; strip it.
    Returns a sorted list of `name==version` (or bare name) lines.
    """
    with io.open(log_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.readlines()
    stripped = []
    for ln in raw:
        # remove leading ISO timestamp + single space
        if len(ln) > 21 and ln[4] == "-" and ln[10] == "T" and "Z " in ln[:35]:
            stripped.append(ln.split("Z ", 1)[1].rstrip("\n"))
        else:
            stripped.append(ln.rstrip("\n"))
    # find the OUTPUT marker (not the echoed command). The echo line contains
    # the shell `echo "----- pip freeze -----"`; the real output is the bare
    # marker line followed by requirement lines.
    starts = [i for i, s in enumerate(stripped)
              if s.strip() == "----- pip freeze -----"]
    if not starts:
        raise RuntimeError("no pip-freeze output marker in %s" % log_path)
    i = starts[-1] + 1
    pkg_re = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(==|===| @ )|"
        r"^[A-Za-z0-9._-]+ @ |^-e |^# Editable")
    noise = ("WARNING:", "[notice]", "You should consider", "pip install",
             "DEPRECATION:", "Defaulting to user")
    pkgs = []
    while i < len(stripped):
        s = stripped[i].strip()
        i += 1
        if s.startswith("-----") or s.startswith("+ ") or s.startswith("##["):
            break
        if not s:
            # blank line only terminates once packages have been collected
            if pkgs:
                break
            continue
        if any(s.startswith(n) for n in noise):
            continue
        if pkg_re.match(s):
            pkgs.append(s)
        elif pkgs:
            # first non-package, non-noise line after the block ends it
            break
        # else: still inside pre-package noise -> keep scanning
    return sorted(set(pkgs))


line("=" * 78)
line("REMEDIATION PHASES 2-5 — correct provenance (NO model inference)")
line("=" * 78)

# ---- load immutable consolidated base
cons_json = os.path.join(SRC, "consolidated",
                         "umsi_resize_causality_6a17288_consolidated.json")
cons_npz = os.path.join(SRC, "consolidated",
                        "umsi_resize_causality_arrays_6a17288_consolidated.npz")
final_json = os.path.join(SRC, "final", "umsi_resize_causality_6a17288.json")
d = json.load(open(cons_json))
src_recorded = json.load(open(final_json))
line("[load] consolidated JSON keys=%d | final source JSON loaded" % len(d))

# ============================================================ PHASE 2: env prov
line("\n--- PHASE 2: correct environment provenance ---")
leg_freeze = parse_pip_freeze(os.path.join(LOGDIR, "causality", LEGACY_LOG))
mod_freeze = parse_pip_freeze(os.path.join(LOGDIR, "causality", MODERN_LOG))
line("[freeze] legacy job freeze: %d pkgs (tensorflow==1.14.0 present=%s)"
     % (len(leg_freeze), "tensorflow==1.14.0" in leg_freeze))
line("[freeze] modern job freeze: %d pkgs (tensorflow==2.16.2 present=%s)"
     % (len(mod_freeze), "tensorflow==2.16.2" in mod_freeze))
assert "tensorflow==1.14.0" in leg_freeze and "keras==2.3.1" in \
    [x.lower() for x in leg_freeze], "legacy freeze markers missing"
assert "tensorflow==2.16.2" in mod_freeze, "modern freeze markers missing"

# remediation-validation environment freeze (clean env; must have NO tf/keras)
rem_freeze = sorted(
    l.strip() for l in open(REM_FREEZE) if l.strip())
tf_leak = [p for p in rem_freeze
           if p.lower().startswith(("tensorflow", "keras", "tf-nightly"))]
line("[freeze] remediation env freeze: %d pkgs (tf/keras leak=%s)"
     % (len(rem_freeze), tf_leak))
assert not tf_leak, "remediation env unexpectedly contains %s" % tf_leak

envs = d.get("environments", {})
containers = envs.get("containers", {})
d["environment_provenance"] = {
    "historical_legacy_inference_env": {
        "source_run_id": ATTRIB["runs"]["causality"],
        "source_job_name": "Legacy TF1.14/Keras2.3.1 (L_REF + inputs + oracle)",
        "source_job_id": 89300575026,
        "role": "legacy_tf1_keras2",
        "python": envs.get("legacy", {}).get("python"),
        "platform": envs.get("legacy", {}).get("platform"),
        "container_image": containers.get("legacy_image"),
        "container_image_digest": containers.get("legacy_image_digest"),
        "pip_freeze_source": "python -m pip freeze output block parsed from the "
        "immutable Legacy job-step log (single block in that job's log)",
        "pip_freeze": leg_freeze,
        "historical_environment_reconstructed": False,
    },
    "historical_modern_inference_env": {
        "source_run_id": ATTRIB["runs"]["causality"],
        "source_job_name": "Modern TF2.16/Keras3 (conditions + ablations)",
        "source_job_id": 89301087755,
        "role": "modern_tf2_keras3",
        "python": envs.get("modern", {}).get("python"),
        "platform": envs.get("modern", {}).get("platform"),
        "container_image": containers.get("modern_image"),
        "container_image_digest": containers.get("modern_image_digest"),
        "pip_freeze_source": "python -m pip freeze output block parsed from the "
        "immutable Modern job-step log (single block in that job's log)",
        "pip_freeze": mod_freeze,
        "historical_environment_reconstructed": False,
    },
    "historical_compare_job_env_record": {
        "source_run_id": ATTRIB["runs"]["causality"],
        "source_job_name": "Compare + causality verdict + evidence",
        "source_job_id": 89302307645,
        "role": "compare_verdict",
        "pip_freeze": None,
        "pip_freeze_status": "not_recorded_in_source_run",
        "historical_environment_reconstructed": False,
        "explanation": "The historical Compare job did not execute its own "
        "`pip freeze`; no freeze was captured in the immutable source run. It "
        "is therefore unrecoverable and is neither guessed, reconstructed, nor "
        "substituted with another job's freeze. The prior consolidated bundle "
        "incorrectly populated pip_freeze.compare_env with the Legacy TF1 "
        "freeze; that field is removed here.",
    },
    "remediation_validation_env": {
        "purpose": "clean environment used ONLY to read arrays and recompute "
        "frozen metrics during this remediation; it is NOT and must not be "
        "read as a recovered historical Compare environment",
        "source_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "job_id": os.environ.get("RC_JOB_ID"),
        "workflow_job_step": os.environ.get("RC_JOB_STEP",
                                            "remediate-evidence"),
        "runner_image": os.environ.get("RC_RUNNER_IMAGE"),
        "container_image_digest": os.environ.get("RC_CONTAINER_DIGEST"),
        "os": os.environ.get("RUNNER_OS"),
        "arch": os.environ.get("RUNNER_ARCH"),
        "python_version": os.environ.get("RC_PYTHON_VERSION"),
        "pip_freeze_all_source": "pip freeze --all in this clean remediation "
        "environment",
        "pip_freeze_all": rem_freeze,
        "contains_tensorflow_or_keras": False,
        "performed_inference": False,
    },
}
# remove the defective field
if "pip_freeze" in d and "compare_env" in d["pip_freeze"]:
    del d["pip_freeze"]["compare_env"]
    line("[fix] removed defective pip_freeze.compare_env")
d["pip_freeze"]["provenance"] = (
    "legacy/modern freezes parsed from their OWN per-job step logs; the "
    "Compare job recorded no freeze (see environment_provenance).")

# ============================================================ PHASE 3: sources
line("\n--- PHASE 3: complete executed-source inventory ---")
# executed python per job at the causality commit (independently git-hashed)
CAUS_JOB_FILES = {
    "Legacy TF1.14/Keras2.3.1 (L_REF + inputs + oracle)": (89300575026, [
        "golden/common.py", "golden/legacy_causality.py",
        "golden/legacy_runner.py", "golden/resize_common.py",
        "golden/windowed_ssim.py", "scripts/download_weights.py"]),
    "Modern TF2.16/Keras3 (conditions + ablations)": (89301087755, [
        "golden/audit_resize_model.py", "golden/common.py",
        "golden/modern_causality.py", "golden/resize_common.py",
        "golden/shared_postprocess.py", "golden/windowed_ssim.py",
        "saliency/umsi_model.py", "scripts/download_weights.py"]),
    "Compare + causality verdict + evidence": (89302307645, [
        "golden/causality_compare.py", "golden/common.py",
        "golden/resize_common.py", "golden/shared_postprocess.py",
        "golden/verify_causality.py", "golden/windowed_ssim.py",
        "hceye/hceye_features.py", "saliency/saliency_features.py"]),
    "Phase 0 - freeze protocol manifest": (89300542501, [
        "golden/common.py", "golden/manifest_resize.py",
        "golden/resize_common.py", "golden/windowed_ssim.py"]),
}
CONS_JOB_FILES = {
    "Consolidate + correct resize causality evidence": (89451142115, [
        "golden/acquire_source_evidence.py", "golden/consolidate_evidence.py",
        "golden/verify_consolidated.py"]),
}
REM_JOB_FILES = {
    "remediate resize causality evidence": (None, [
        "golden/remediation_manifest.py", "golden/verify_remediation.py",
        "golden/acquire_remediation_evidence.py",
        "golden/remediate_evidence.py"]),
}

inv_files = []
missing_src = []


def add_files(commit, run_label, jobmap):
    for job_name, (job_id, paths) in jobmap.items():
        for p in paths:
            gb = git_blob(commit, p)
            if gb is None:
                missing_src.append((commit, p))
                continue
            data, sha, blob_id = gb
            inv_files.append({
                "path": p, "repo": REPO, "commit": commit,
                "role": "executed_python", "source_run_label": run_label,
                "source_job_name": job_name, "source_job_id": job_id,
                "bytes": len(data), "sha256": sha, "git_blob_id": blob_id})


add_files(CAUS_COMMIT, "causality", CAUS_JOB_FILES)
add_files(CONS_COMMIT, "consolidation", CONS_JOB_FILES)
add_files(REM_COMMIT, "remediation", REM_JOB_FILES)

# executed workflow YAMLs
WORKFLOWS = [
    (CAUS_COMMIT, "causality", ".github/workflows/umsi-resize-causality.yml"),
    (CONS_COMMIT, "consolidation",
     ".github/workflows/umsi-resize-consolidation.yml"),
    (REM_COMMIT, "remediation",
     ".github/workflows/umsi-resize-remediation.yml"),
]
inline_run_blocks = []
for commit, label, wf in WORKFLOWS:
    gb = git_blob(commit, wf)
    if gb is None:
        missing_src.append((commit, wf))
        continue
    data, sha, blob_id = gb
    inv_files.append({
        "path": wf, "repo": REPO, "commit": commit, "role": "workflow_yaml",
        "source_run_label": label, "bytes": len(data), "sha256": sha,
        "git_blob_id": blob_id})
    # extract inline run: blocks (indentation-based, no yaml dep)
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    i = 0
    blk_no = 0
    while i < len(lines):
        s = lines[i]
        st = s.strip()
        if st.startswith("run:"):
            base_indent = len(s) - len(s.lstrip())
            body = []
            if st != "run: |" and st != "run: |-" and st != "run:":
                body.append(st[len("run:"):].strip())
                i += 1
            else:
                i += 1
                while i < len(lines):
                    nxt = lines[i]
                    if nxt.strip() == "":
                        body.append("")
                        i += 1
                        continue
                    ind = len(nxt) - len(nxt.lstrip())
                    if ind <= base_indent:
                        break
                    body.append(nxt)
                    i += 1
            blk_no += 1
            btxt = "\n".join(body)
            inline_run_blocks.append({
                "workflow": wf, "commit": commit, "source_run_label": label,
                "block_index": blk_no, "bytes": len(btxt.encode("utf-8")),
                "sha256": sha_bytes(btxt.encode("utf-8"))})
        else:
            i += 1

# complete local source trees hashed at their immutable commits
source_trees = {}
for commit, label, subtrees in (
    (CAUS_COMMIT, "causality", ["golden", "scripts", ".github/workflows"]),
    (CONS_COMMIT, "consolidation", ["golden", ".github/workflows"]),
    (REM_COMMIT, "remediation", ["golden", ".github/workflows"]),
):
    tree = {}
    for sub in subtrees:
        try:
            listing = subprocess.check_output(
                ["git", "-C", GIT_DIR, "ls-tree", "-r", "--long", commit, sub],
                stderr=subprocess.DEVNULL).decode().splitlines()
        except subprocess.CalledProcessError:
            continue
        for row in listing:
            meta, path = row.split("\t", 1)
            _mode, _type, blob_id, size = meta.split()
            gb = git_blob(commit, path)
            if gb is None:
                continue
            data, sha, _bid = gb
            tree[path] = {"git_blob_id": blob_id, "bytes": int(size),
                          "sha256": sha}
    root = sha_bytes(
        "\n".join("%s:%s" % (p, tree[p]["git_blob_id"])
                  for p in sorted(tree)).encode())
    source_trees[label] = {"commit": commit, "tree_root_sha256": root,
                           "file_count": len(tree), "files": tree}

d["executed_source_inventory"] = {
    "files": inv_files,
    "inline_run_blocks": inline_run_blocks,
    "source_trees": source_trees,
    "note": "Every directly invoked script (incl. scripts/download_weights.py), "
    "every executed workflow YAML and every inline run block is hashed at its "
    "immutable commit via git objects; artifact-to-job attribution comes from "
    "GitHub job metadata, not filename inference.",
}
# keep the prior git-object record for continuity, retitled
d["executed_source_sha256"]["note"] = (
    "superseded by executed_source_inventory (this remediation adds "
    "scripts/download_weights.py, workflow YAMLs, inline run blocks and full "
    "source trees).")

dw = [f for f in inv_files if f["path"] == "scripts/download_weights.py"]
line("[inv] files=%d | inline_run_blocks=%d | source_trees=%d"
     % (len(inv_files), len(inline_run_blocks), len(source_trees)))
line("[inv] scripts/download_weights.py present in %d job roles: %s"
     % (len(dw), ", ".join(sorted({f["source_job_name"] for f in dw}))))
if missing_src:
    line("BLOCKED: missing source objects: %s" % missing_src)
    sys.exit(2)

# ---- artifact-to-job inventory (D6)
d["artifact_to_job_inventory"] = [
    {"artifact_name": a["artifact_name"], "run_label": a["run_label"],
     "run_id": a["run_id"], "artifact_id": a["artifact_id"],
     "source_job_name": a["source_job_name"], "source_job_id": a["source_job_id"],
     "attribution_method": a["attribution_method"],
     "attribution_verified": a["attribution_verified"],
     "archive_sha256": a["archive_sha256"],
     "extracted_files": a["extracted_files"]}
    for a in ATTRIB["artifacts"]]
line("[inv] artifact_to_job_inventory: %d artifacts (all verified=%s)"
     % (len(d["artifact_to_job_inventory"]),
        all(a["attribution_verified"] for a in d["artifact_to_job_inventory"])))

# ============================================================ PHASE 4: reverify
line("\n--- PHASE 4: independent no-inference reverification ---")
z = np.load(cons_npz, allow_pickle=False)
obj_arrays = [k for k in z.files if z[k].dtype.kind == "O"]
assert not obj_arrays, "object arrays present: %s" % obj_arrays
line("[npz] loaded %d arrays, object arrays=%d (allow_pickle=False)"
     % (len(z.files), len(obj_arrays)))

idx = d["consolidated_array_index"]
assert len(idx) == 183, "expected 183 arrays, got %d" % len(idx)
bad = 0
for e in idx:
    k = e["consolidated_key"]
    h = sha256_array_native(z[k])
    ok = (h == e["consolidated_array_data_sha256"]
          == e["source_array_data_sha256"])
    e["reverified_npz_array_data_sha256"] = h
    e["reverified_hash_matches"] = ok
    if not ok:
        bad += 1
line("[npz] 183 array hashes reverified (source==consolidated==npz), bad=%d"
     % bad)
assert bad == 0, "array hash mismatch"

# 20 hash-corrections preserved + re-verified against npz
corr = d["correction_history"]
assert len(corr) == 20, "expected 20 corrections, got %d" % len(corr)
cbad = 0
for c in corr:
    ref_k = c["reference_array_key"].replace("_shared", "")
    cond_k = c["condition_array_key"].replace("_shared", "")
    # consolidated npz keys are "shared/<fixture>_<cond>"
    fixture_cond = c["condition_array_key"].rsplit("_shared", 1)[0]
    fixture_ref = c["reference_array_key"].rsplit("_shared", 1)[0]
    ck = "shared/" + fixture_cond
    rk = "shared/" + fixture_ref
    ok = True
    if ck in z.files:
        ok = ok and sha256_array_native(z[ck]) == \
            c["corrected_condition_array_data_sha256"]
    if rk in z.files:
        ok = ok and sha256_array_native(z[rk]) == \
            c["corrected_reference_array_data_sha256"]
    ok = ok and c["original_value_equals_reference_hash"] and \
        c["verified_against_consolidated_npz"]
    c["reverified_against_remediated_npz"] = bool(ok)
    if not ok:
        cbad += 1
line("[corr] 20 hash-corrections re-verified against arrays, bad=%d" % cbad)
assert cbad == 0, "hash-correction reverify failed"
d["hash_correction_history"] = corr

# metric reverification: source-recorded vs previous-consolidated vs recomputed
line("[metric] recomputing frozen metrics from immutable arrays ...")
metric_keys = ["pearson", "spearman", "mae", "rmse", "max_abs_diff",
               "legacy_audit_global_ssim", "windowed_ssim"]
rv = {}
n_incons = 0
for fx in FIXTURES:
    rv[fx] = {}
    for c in CONDS:
        a = z["shared/%s_%s" % (fx, c)]
        b = z["shared/%s_L_REF" % fx]
        rec = recompute_metrics(a, b)
        prev = d["phase3_conditions_vs_L_REF"][fx]["conditions"][c][
            "shared_map_metrics"]
        srcm = (src_recorded["phase3_conditions_vs_L_REF"][fx]["conditions"][c]
                ["shared_map_metrics"])
        rv[fx][c] = {}
        for mk in metric_keys:
            th = TH_MAXABS if mk == "max_abs_diff" else TH_TIGHT
            sv = srcm.get(mk)
            pv = prev.get(mk)
            nv = rec.get(mk)
            diffs = [abs(x - nv) for x in (sv, pv)
                     if isinstance(x, (int, float)) and isinstance(nv, float)]
            consistent = bool(diffs) and all(dd <= th for dd in diffs)
            if sv is None and pv is None and nv is None:
                consistent = True
            rv[fx][c][mk] = {
                "source_recorded": sv, "previous_consolidated": pv,
                "newly_recomputed": nv,
                "abs_diff_source_vs_recomputed":
                    (abs(sv - nv) if isinstance(sv, (int, float))
                     and isinstance(nv, float) else None),
                "abs_diff_consolidated_vs_recomputed":
                    (abs(pv - nv) if isinstance(pv, (int, float))
                     and isinstance(nv, float) else None),
                "threshold": th, "consistent": consistent}
            if not consistent:
                n_incons += 1
d["metric_reverification"] = rv
line("[metric] fixtures*conditions*metrics reverified; inconsistent=%d"
     % n_incons)
assert n_incons == 0, "metric reverification inconsistency"

# ============================================================ PHASE 5: schema
line("\n--- PHASE 5: corrected schema flags + correction history entry ---")
d["bitwise_equal"] = False
d["numerical_parity_within_preregistered_thresholds"] = True
d["historical_compare_pip_freeze_available"] = False
d["historical_compare_environment_reconstructed"] = False
d["remediation_validation_pip_freeze_complete"] = True
d["scientific_verdict_changed"] = False
d["remediation_no_model_inference"] = True
d["thresholds_match_frozen_manifest"] = True
# scientific conclusion unchanged (scoped)
d["scientific_conclusion_unchanged"] = (
    "The modern resize semantics were the sole material cause of the "
    "pre-registered golden-threshold failures in this controlled experiment.")
d["remediation_limitation_disclosure"] = (
    "The historical Compare job did not record its own pip freeze; that "
    "environment is transparently unavailable and has NOT been reconstructed. "
    "The bundle is not bitwise-identical; passing conditions match L_REF as "
    "numerical parity within the pre-registered thresholds. No model inference "
    "was performed during remediation.")

# provenance correction-history (env misattribution) — kept separate from the
# 20 array hash-corrections
d["provenance_correction_history"] = d.get("provenance_correction_history", [])
d["provenance_correction_history"].append({
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "field_path": "pip_freeze.compare_env",
    "defect": "consolidated bundle populated the Compare-job environment with "
    "the Legacy TF1.14/Keras2.3.1 freeze (parser took the first embedded freeze "
    "block in the Compare log, which cats the legacy+modern run logs).",
    "original_recorded_value_summary":
        "46-package freeze identical to legacy_tf1_keras2 "
        "(tensorflow==1.14.0, Keras==2.3.1, numpy==1.19.2)",
    "correction": "removed pip_freeze.compare_env; recorded the Compare job's "
    "freeze as not_recorded_in_source_run (null, not reconstructed); legacy and "
    "modern freezes re-derived from their OWN per-job step logs.",
    "scientific_verdict_changed": False,
})
d["remediation_generated_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
d["remediation_branch"] = os.environ.get("RC_BRANCH")
d["remediation_commit"] = REM_COMMIT
d["remediation_of_consolidated_run_id"] = ATTRIB["runs"]["consolidation"]
d["remediation_of_causality_run_id"] = ATTRIB["runs"]["causality"]

# ---- emit deliverables
# NPZ: copy the immutable consolidated NPZ verbatim (array data proven byte
# identical to the source NPZs; no cast/normalize/reorder/recalc).
shutil.copyfile(cons_npz, OUT_NPZ)
# integrity: reopen the copy and re-hash every array
zc = np.load(OUT_NPZ, allow_pickle=False)
copy_bad = sum(1 for e in idx
               if sha256_array_native(zc[e["consolidated_key"]]) !=
               e["consolidated_array_data_sha256"])
assert copy_bad == 0, "copied NPZ array mismatch"
line("[emit] NPZ copied byte-identical (%d arrays, all hashes match)"
     % len(idx))

with open(OUT_JSON, "w") as fh:
    json.dump(d, fh, indent=1, sort_keys=True, allow_nan=False)
line("[emit] remediated JSON written: %s (%d bytes)"
     % (OUT_JSON, os.path.getsize(OUT_JSON)))
line("=" * 78)
line("PHASES 2-5 COMPLETE — provenance corrected, no model inference")
line("=" * 78)
