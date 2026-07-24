"""Phase 6 validator for the remediated UMSI++ resize evidence bundle.

Strictly validates the remediated bundle with NO model inference. Exits
non-zero on any failure. Enforces the provenance-defect fixes so they cannot
recur:
  * strict JSON (no NaN/Infinity);
  * the four environment objects use the correct attribution;
  * compare_env carries NO legacy/modern freeze (must be null +
    not_recorded_in_source_run);
  * the remediation-validation env freeze is embedded, complete and free of
    tensorflow/keras;
  * every artifact has a verified source-job mapping;
  * scripts/download_weights.py and every directly invoked file are hashed;
  * every consolidated array reference exists and every hash matches exactly;
  * all 183 arrays present + unchanged; all 20 hash corrections preserved;
  * metric recomputation consistent across source/consolidated/recomputed;
  * thresholds/conditions unchanged; scientific verdict unchanged;
  * no model inference occurred.
"""
import os
import sys
import json
import math
import hashlib

import numpy as np

JSON = os.environ["RC_OUT_JSON"]
NPZ = os.environ["RC_OUT_NPZ"]
FIXTURES = ["lowcontrast", "ui1", "ui2", "ui3", "uniform"]
CONDS = ["M_CC", "M_CL", "M_LC", "M_LL"]
LEGACY_MARKERS = {"tensorflow==1.14.0", "keras==2.3.1", "numpy==1.19.2"}
MODERN_MARKERS = {"tensorflow==2.16.2", "keras==3.10.0", "numpy==1.26.4"}

fails = []


def check(cond, msg):
    print(("  OK  " if cond else " FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def a_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def has_any(freeze, markers):
    low = {x.lower() for x in (freeze or [])}
    return bool(low & markers)


print("=" * 78)
print("PHASE 6 VALIDATION — remediated resize evidence bundle")
print("=" * 78)

# 1. strict JSON
raw = open(JSON).read()
try:
    d = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("non-finite: %s" % x)))
    check(True, "strict JSON parse (no NaN/Infinity constants)")
except Exception as e:  # noqa: BLE001
    check(False, "strict JSON parse failed: %s" % e)
    print("BLOCKED"); sys.exit(1)
for tok in ("NaN", "Infinity", "-Infinity"):
    check(tok not in raw.replace('"', ''), "no bare %s token" % tok)


def walk(o, p=""):
    if isinstance(o, float) and not math.isfinite(o):
        fails.append("non-finite float @%s" % p)
    elif isinstance(o, dict):
        for k, v in o.items():
            walk(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, p + "[%d]" % i)


walk(d)
check(not any("non-finite" in f for f in fails), "all JSON floats finite")

# 2. four environment objects with correct attribution
env = d["environment_provenance"]
for k in ("historical_legacy_inference_env", "historical_modern_inference_env",
          "historical_compare_job_env_record", "remediation_validation_env"):
    check(k in env, "environment object present: %s" % k)

leg = env["historical_legacy_inference_env"]
mod = env["historical_modern_inference_env"]
check(leg["source_job_name"] ==
      "Legacy TF1.14/Keras2.3.1 (L_REF + inputs + oracle)",
      "legacy env attributed to Legacy job")
check(has_any(leg["pip_freeze"], LEGACY_MARKERS),
      "legacy env freeze carries TF1.14/Keras2.3.1/numpy1.19.2")
check(mod["source_job_name"] == "Modern TF2.16/Keras3 (conditions + ablations)",
      "modern env attributed to Modern job")
check(has_any(mod["pip_freeze"], MODERN_MARKERS),
      "modern env freeze carries TF2.16/Keras3.10/numpy1.26.4")

# 3. compare-job env record: truthful, no substituted freeze
comp = env["historical_compare_job_env_record"]
check(comp["pip_freeze"] is None, "compare-job pip_freeze is null")
check(comp["pip_freeze_status"] == "not_recorded_in_source_run",
      "compare-job pip_freeze_status = not_recorded_in_source_run")
check(comp["historical_environment_reconstructed"] is False,
      "compare-job historical_environment_reconstructed = false")
check("compare_env" not in d.get("pip_freeze", {}),
      "no residual pip_freeze.compare_env field")
# ensure the compare record does NOT smuggle a legacy/modern freeze anywhere
comp_txt = json.dumps(comp).lower()
check("tensorflow==1.14.0" not in comp_txt and
      "tensorflow==2.16.2" not in comp_txt,
      "compare-job record contains no legacy/modern freeze list")

# 4. remediation-validation env: complete freeze, no TF/keras
rem = env["remediation_validation_env"]
rf = rem["pip_freeze_all"]
check(isinstance(rf, list) and len(rf) >= 5,
      "remediation env pip_freeze_all embedded (%d pkgs)" % len(rf or []))
check(d["remediation_validation_pip_freeze_complete"] is True,
      "remediation_validation_pip_freeze_complete flag true")
tf_present = any(x.lower().startswith(("tensorflow", "keras", "tf-nightly"))
                 for x in rf)
check(not tf_present, "remediation env has NO tensorflow/keras package")
for f in ("runner_image", "os", "arch", "python_version",
          "workflow_run_id", "job_id"):
    check(f in rem and rem[f], "remediation env records %s" % f)
check(rem.get("performed_inference") is False,
      "remediation env performed_inference = false")

# 5. artifact -> job mapping verified
inv = d["artifact_to_job_inventory"]
unresolved = [a["artifact_name"] for a in inv
              if not a.get("source_job_id") or not a.get("source_job_name")
              or not a.get("attribution_verified")]
check(len(inv) >= 5, "artifact inventory has >=5 artifacts (%d)" % len(inv))
check(not unresolved, "every artifact has verified source-job mapping")

# 6. executed-source completeness incl download_weights.py
src = d["executed_source_inventory"]
allpaths = {e["path"] for e in src["files"]}
check("scripts/download_weights.py" in allpaths,
      "scripts/download_weights.py present in source inventory")
nohash = [e["path"] for e in src["files"] if not e.get("sha256")]
nocommit = [e["path"] for e in src["files"] if not e.get("commit")]
check(not nohash, "every inventory file has a sha256")
check(not nocommit, "every inventory file has a commit")
check(bool(src.get("inline_run_blocks")),
      "inline workflow run blocks recorded + hashed")
check(bool(src.get("source_trees")),
      "complete local source trees hashed at their commits")

# 7-8. arrays present + hashes match; 183 arrays; corrections preserved
z = np.load(NPZ, allow_pickle=False)
check(sum(1 for k in z.files if z[k].dtype.kind == "O") == 0,
      "NPZ has no object arrays")
idx = d["consolidated_array_index"]
check(len(idx) == 183, "183 arrays indexed (%d)" % len(idx))
miss = bad = 0
for e in idx:
    k = e["consolidated_key"]
    if k not in z.files:
        miss += 1
        continue
    h = a_sha(z[k])
    if h != e["consolidated_array_data_sha256"] or \
            h != e["source_array_data_sha256"]:
        bad += 1
check(miss == 0, "all indexed arrays present in NPZ")
check(bad == 0, "all array hashes match exactly (source==consolidated==NPZ)")
check(len(d["hash_correction_history"]) == 20, "20 hash corrections preserved")
cbad = sum(1 for c in d["hash_correction_history"]
           if not c.get("verified_against_consolidated_npz")
           or not c.get("original_value_equals_reference_hash"))
check(cbad == 0, "all 20 corrections verified + original==reference-hash")

# 9. metric reverification consistency (source/consolidated/recomputed)
rv = d["metric_reverification"]
inconsistent = 0
for fx in FIXTURES:
    for c in CONDS:
        for key, rec in rv[fx][c].items():
            if not rec["consistent"]:
                inconsistent += 1
check(inconsistent == 0,
      "all metric recomputations consistent (source/consolidated/recomputed)")

# 10. thresholds/conditions unchanged; verdict unchanged; no inference
check(d["thresholds_match_frozen_manifest"] is True,
      "thresholds match frozen manifest")
check(d["phase3_condition_all_pass"] == {"M_CC": False, "M_CL": True,
                                         "M_LC": False, "M_LL": True},
      "condition pass/fail unchanged (M_CC/M_LC fail, M_CL/M_LL pass)")
check(d["scientific_verdict_changed"] is False, "scientific_verdict_changed=false")
check(d["bitwise_equal"] is False and
      d["numerical_parity_within_preregistered_thresholds"] is True,
      "bitwise_equal=false, numerical_parity=true")
check(d["historical_compare_pip_freeze_available"] is False,
      "historical_compare_pip_freeze_available=false")
check(d["historical_compare_environment_reconstructed"] is False,
      "historical_compare_environment_reconstructed=false")
check(d["remediation_no_model_inference"] is True,
      "remediation_no_model_inference=true")

print("=" * 78)
if fails:
    print("VALIDATION FAILED — %d checks:" % len(fails))
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("ALL REMEDIATION VALIDATION CHECKS PASSED")
print("=" * 78)
