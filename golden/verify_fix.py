"""Independent final validator for the UMSI++ legacy-resize production-fix
evidence bundle.

Re-opens the deliverable JSON + NPZ produced by build_fix_evidence.py and
mechanically re-checks every acceptance criterion WITHOUT trusting the recorded
verdict: it recomputes the verdict from the individual checks, re-verifies every
NPZ array (opened with allow_pickle=False, object arrays rejected) against the
JSON array_index by SHA-256, confirms all required sections are present, and
confirms the verdict string is exactly the pre-registered PASS string only when
every hard check passes.

Exits non-zero (BLOCKED) on any failure.
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

PASS_STRING = ("UMSI LEGACY-RESIZE PRODUCTION FIX VALIDATED — "
               "ALL FROZEN GOLDEN CRITERIA PASSED")
BLOCK_STRING = ("UMSI LEGACY-RESIZE PRODUCTION FIX BLOCKED — "
                "GOLDEN OR SCOPE VALIDATION FAILED")

REQUIRED_SECTIONS = [
    "operator_oracle", "structural_callsite_verification",
    "checkpoint_and_param_compatibility", "golden", "boundary_activations",
    "downstream_PROVISIONAL", "repeatability", "regression_interface",
    "environment_provenance", "diff_allowlist", "array_index",
    "verdict_checks", "limitations",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    failures = []
    d = json.load(open(args.json))

    # 1) required sections present + non-null.
    for sec in REQUIRED_SECTIONS:
        if d.get(sec) in (None, {}, []):
            failures.append("missing_or_empty_section:%s" % sec)

    # 2) NPZ integrity: allow_pickle=False, no object arrays, hashes match.
    z = np.load(args.npz, allow_pickle=False)
    idx = d.get("array_index", {})
    if set(z.files) != set(idx.keys()):
        failures.append("npz_key_mismatch")
    for k in z.files:
        a = z[k]
        if a.dtype == object:
            failures.append("object_array:%s" % k)
            continue
        want = idx.get(k, {})
        got_sha = common.sha256_array(a)
        if want.get("sha256") != got_sha:
            failures.append("array_hash_mismatch:%s" % k)
        if want.get("shape") != list(a.shape):
            failures.append("array_shape_mismatch:%s" % k)

    # 3) recompute verdict from individual checks (ignore stored verdict).
    checks = d.get("verdict_checks", {})
    hard = {k: v for k, v in checks.items() if v is not None}
    recomputed_pass = bool(hard) and all(hard.values())
    for k, v in hard.items():
        if not v:
            failures.append("check_failed:%s" % k)

    # 4) five fixtures each pass every threshold.
    golden = d.get("golden", {})
    fx = golden.get("fixtures", {})
    if sorted(fx.keys()) != sorted(common.FIXTURE_ORDER):
        failures.append("golden_fixture_set_mismatch")
    for name in common.FIXTURE_ORDER:
        chk = fx.get(name, {}).get("checks", {})
        if not chk.get("all_pass"):
            failures.append("fixture_not_pass:%s" % name)
    if not golden.get("overall", {}).get("all_fixtures_pass"):
        failures.append("golden_overall_not_pass")

    # 5) operator oracle bit-exact + faithful.
    if not d["operator_oracle"].get("oracle_pass"):
        failures.append("operator_oracle_not_pass")
    if d["operator_oracle"].get("worst_helper_vs_legacy_max_abs_diff") != 0.0:
        failures.append("operator_helper_not_bit_exact")

    # 6) exactly three call sites.
    if not d["structural_callsite_verification"].get("callsite_check_pass"):
        failures.append("callsite_not_exactly_three")

    # 7) checkpoint + param compatibility.
    compat = d["checkpoint_and_param_compatibility"]
    if not compat.get("compat_pass"):
        failures.append("checkpoint_param_compat_failed")
    if not compat.get("checkpoint_load_zero_warnings"):
        failures.append("checkpoint_load_had_warnings")

    # 8) diff within allowlist.
    if not d["diff_allowlist"].get("within_allowlist"):
        failures.append("diff_not_within_allowlist")

    # 9) verdict string consistency.
    stored_verdict = d.get("verdict", "")
    expect = PASS_STRING if (recomputed_pass and not failures) else BLOCK_STRING
    if recomputed_pass and stored_verdict != PASS_STRING:
        failures.append("verdict_string_not_pass_despite_checks")
    if (not recomputed_pass) and stored_verdict == PASS_STRING:
        failures.append("verdict_string_pass_despite_failed_checks")

    ok = (len(failures) == 0 and recomputed_pass)
    print("=" * 72, flush=True)
    print("FINAL VALIDATOR — verify_fix.py", flush=True)
    print("=" * 72, flush=True)
    print("recomputed_pass:", recomputed_pass, flush=True)
    print("npz arrays verified:", len(z.files), flush=True)
    if failures:
        print("FAILURES:", flush=True)
        for f in failures:
            print("  -", f, flush=True)
    print("\nVERDICT:", PASS_STRING if ok else BLOCK_STRING, flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
