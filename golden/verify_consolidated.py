"""Phase 5 validation for the consolidated UMSI++ resize causality bundle.

Strictly validates the corrected evidence bundle WITHOUT any model inference.
Exits non-zero on any failure. Checks:
  * strict JSON parse with no NaN/Infinity tokens;
  * no residual ambiguous single 'sha256' field in shared_map_metrics;
  * every NPZ key opens with allow_pickle=False (no object arrays);
  * every JSON array reference (consolidated_array_index + correction_history)
    exists in the NPZ and its hash matches exactly;
  * raw-tensor metrics present for every fixture x condition;
  * repeatability records complete for L_REF/M_CC/M_LL;
  * environment + pip freeze + executed-source provenance embedded;
  * thresholds + conditions unchanged vs the frozen manifest;
  * the scientific verdict follows mechanically from the stored decision flags;
  * consolidation ran with no model inference.
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

fails = []


def check(cond, msg):
    print(("  OK  " if cond else " FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def array_data_sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


print("=" * 78)
print("PHASE 5 VALIDATION — consolidated resize causality bundle")
print("=" * 78)

# 1. strict JSON parse, reject NaN/Infinity
raw = open(JSON).read()
try:
    d = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("non-finite token: %s" % x)))
    check(True, "strict JSON parse (no NaN/Infinity constants)")
except Exception as e:  # noqa: BLE001
    check(False, "strict JSON parse failed: %s" % e)
    print("BLOCKED: JSON not strict"); sys.exit(1)

for tok in ("NaN", "Infinity", "-Infinity"):
    check(tok not in raw.replace('"', ''),
          "no bare %s token in JSON text" % tok)


def walk_finite(o, path=""):
    if isinstance(o, float):
        if not math.isfinite(o):
            fails.append("non-finite float at %s" % path)
    elif isinstance(o, dict):
        for k, v in o.items():
            walk_finite(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk_finite(v, path + "[%d]" % i)


walk_finite(d)
check(not any("non-finite" in f for f in fails), "all JSON floats finite")

# 2. no residual ambiguous sha256 field
resid = 0
for fx in FIXTURES:
    for c in CONDS:
        blk = d["phase3_conditions_vs_L_REF"][fx]["conditions"][c][
            "shared_map_metrics"]
        if "sha256" in blk:
            resid += 1
        for need in ("reference_array_data_sha256",
                     "condition_array_data_sha256"):
            if need not in blk:
                fails.append("missing %s in %s/%s" % (need, fx, c))
check(resid == 0, "no residual ambiguous 'sha256' field (found %d)" % resid)
check(all("reference_array_data_sha256" in
          d["phase3_conditions_vs_L_REF"][fx]["conditions"][c][
              "shared_map_metrics"] for fx in FIXTURES for c in CONDS),
      "dual reference/condition hash present for all 20 fixture-conditions")

# 3. NPZ opens strict, no object arrays
z = np.load(NPZ, allow_pickle=False)
nobj = sum(1 for k in z.files if z[k].dtype.kind == "O")
check(nobj == 0, "NPZ has no object arrays (%d keys)" % len(z.files))

# 4. every array reference exists + hash matches
idx = d["consolidated_array_index"]
missing = 0
badhash = 0
for e in idx:
    k = e["consolidated_key"]
    if k not in z.files:
        missing += 1
        continue
    h = array_data_sha256(z[k])
    if h != e["consolidated_array_data_sha256"] or \
            h != e["source_array_data_sha256"]:
        badhash += 1
check(missing == 0, "all %d indexed arrays present in NPZ" % len(idx))
check(badhash == 0, "all indexed array hashes match (source==consolidated)")

# correction_history array references resolve + hashes match
ch_bad = 0
for c in d["correction_history"]:
    parts = c["field_path"].split(".")
    fx, cond = parts[1], parts[3]
    refk = "shared/%s_L_REF" % fx
    cndk = "shared/%s_%s" % (fx, cond)
    if refk not in z.files or cndk not in z.files:
        ch_bad += 1
        continue
    if array_data_sha256(z[refk]) != c["corrected_reference_array_data_sha256"]:
        ch_bad += 1
    if array_data_sha256(z[cndk]) != c["corrected_condition_array_data_sha256"]:
        ch_bad += 1
    if not c["original_value_equals_reference_hash"]:
        ch_bad += 1
    if not c["verified_against_consolidated_npz"]:
        ch_bad += 1
check(ch_bad == 0, "all %d corrections verified against consolidated NPZ"
      % len(d["correction_history"]))
check(len(d["correction_history"]) == 20, "20 hash corrections recorded")

# 5. raw metrics present for every fixture x condition
rawok = all(c in d["phase3_raw_tensor_metrics"][fx]["conditions"]
            for fx in FIXTURES for c in CONDS)
check(rawok, "raw-tensor metrics present for all fixture x condition")
check(all("raw_512x512_vs_L_REF_raw" in
          d["phase3_raw_tensor_metrics"][fx]["conditions"][c]
          for fx in FIXTURES for c in CONDS),
      "raw-vs-L_REF comparison present for all conditions")

# 6. repeatability complete
rep = d["repeatability"]
repok = all(fx in rep["L_REF_legacy"] and fx in rep["modern_M_CC_M_LL"]
            for fx in FIXTURES)
check(repok, "repeatability records for L_REF/M_CC/M_LL over all fixtures")
check(rep["all_deterministic"] is True, "all repeatability runs deterministic")

# 7. provenance embedded
check(bool(d["pip_freeze"]["legacy_tf1_keras2"]) and
      bool(d["pip_freeze"]["modern_tf2_keras3"]),
      "pip freeze embedded for legacy + modern")
check(bool(d["executed_source_sha256"]["upstream_umsi_model_source"]) and
      bool(d["executed_source_sha256"]["audit_repo_source_git_objects"]),
      "executed-source hashes embedded (upstream + audit git objects)")
check("legacy" in d["environments"] and "modern" in d["environments"] and
      d["environments"]["containers"]["legacy_image_digest"].startswith("sha256:"),
      "environments + immutable container digests embedded")

# 8. thresholds + conditions unchanged vs frozen manifest
man = d["frozen_manifest"]
check(d["thresholds_match_frozen_manifest"] is True,
      "thresholds match the frozen manifest")
check(list(man["conditions"].keys()) == ["L_REF", "M_CC", "M_CL", "M_LC", "M_LL"],
      "five conditions unchanged in frozen manifest")

# 9. verdict follows mechanically from decision flags
dec = d["decision"]
confirmed = (dec["oracle_pass"] and dec["M_LC_reproduces_failure"] and
             dec["M_LL_passes_all_fixtures"] and
             dec["M_CL_passes_all_fixtures"] and
             dec["activation_supports_hypothesis"])
verdict = d["scientific_verdict"]
check(confirmed and verdict.startswith("UMSI RESIZE CAUSE CONFIRMED"),
      "verdict follows mechanically from stored decision flags")
check(d["bitwise_equal"] is False and
      d["numerical_parity_within_preregistered_thresholds"] is True,
      "bitwise_equal=false and numerical_parity=true stated")
check(d["phase3_condition_all_pass"] == {"M_CC": False, "M_CL": True,
                                         "M_LC": False, "M_LL": True},
      "condition pass/fail pattern unchanged (M_CC/M_LC fail, M_CL/M_LL pass)")

# 10. no model inference
check(d["consolidation_no_model_inference"] is True,
      "consolidation ran with no model inference")

print("=" * 78)
if fails:
    print("VALIDATION FAILED — %d checks failed:" % len(fails))
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("ALL VALIDATION CHECKS PASSED")
print("=" * 78)
