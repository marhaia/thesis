"""Phase 0 — freeze the UMSI++ resize EVIDENCE REMEDIATION protocol.

Emits a machine-readable protocol (locked refs, both source runs, the six exact
provenance defects, acquisition/attribution/inventory/reverification rules,
validation checks and the exact verdict logic) BEFORE the substantive
remediation run. Standard-library only so it imports on a bare runner.

This is evidence remediation only: no model inference, no production fix, no
norms rebuild.
"""
import os
import json
import datetime

OUT = os.environ.get("RC_MANIFEST_OUT", "resize_remediation_manifest.json")

LOCKED_REFS = {
    "main": "12ba939dd4ead3eba15b98f5501ebabd888a6677",
    "fix/umsi-saliency-integrity": "6a17288bbac848bb023fccc98aee606c034ea54b",
    "fix/layout-score-canonical-path":
        "6b02ef648975d7902b5d03041634d6acad891fbf",
    "audit/umsi-resize-causality":
        "a03c42a8ac508d61f8b7fd1cffde3c0e0df8d5f3",
    "audit/umsi-resize-evidence-consolidation":
        "048aba5c33057681009087a7d06ec555b2990d20",
    "audit/umsi-tf1-keras2-golden":
        "77241fc6e635949dada502e6c2d902eac1e8b030",
}

DEFECTS = [
    {"id": 1,
     "summary": "pip_freeze.compare_env incorrectly contains the TF1 legacy "
                "environment (tensorflow==1.14.0, standalone Keras 2.3.1, "
                "numpy==1.19.2).",
     "fixable": True},
    {"id": 2,
     "summary": "The parser encountered embedded legacy and modern freeze "
                "blocks in the Compare log and assigned the first one to "
                "compare_env.",
     "fixable": True},
    {"id": 3,
     "summary": "The historical Compare job did not execute its own pip "
                "freeze; its exact historical package set cannot be recovered "
                "from the source run.",
     "fixable": False,
     "resolution": "record truthfully as not_recorded_in_source_run; do NOT "
                   "guess, reconstruct or substitute an embedded freeze."},
    {"id": 4,
     "summary": "The existing validator did not detect the environment "
                "misattribution.",
     "fixable": True},
    {"id": 5,
     "summary": "At least one directly executed source file, "
                "scripts/download_weights.py, is absent from the "
                "executed-source hash inventory.",
     "fixable": True},
    {"id": 6,
     "summary": "The artifact inventory lacks explicit, independently "
                "verifiable source-job attribution.",
     "fixable": True},
]

SOURCE_RUNS = {
    "causality": {"run_id": "30034973775",
                  "branch": "audit/umsi-resize-causality",
                  "commit": "a03c42a8ac508d61f8b7fd1cffde3c0e0df8d5f3",
                  "workflow": ".github/workflows/umsi-resize-causality.yml"},
    "consolidation": {"run_id": "30083793554",
                      "branch": "audit/umsi-resize-evidence-consolidation",
                      "commit": "048aba5c33057681009087a7d06ec555b2990d20",
                      "workflow":
                          ".github/workflows/umsi-resize-consolidation.yml"},
}

# Authoritative artifact -> job mapping, declared by the workflow upload steps.
ARTIFACT_JOB_MAP = {
    "manifest": {"run": "causality",
                 "job_name": "Phase 0 - freeze protocol manifest"},
    "legacy": {"run": "causality",
               "job_name": "Legacy TF1.14/Keras2.3.1 (L_REF + inputs + oracle)"},
    "modern": {"run": "causality",
               "job_name": "Modern TF2.16/Keras3 (conditions + ablations)"},
    "umsi-resize-causality-evidence-6a17288":
        {"run": "causality",
         "job_name": "Compare + causality verdict + evidence"},
    "umsi-resize-causality-evidence-consolidated-a03c42a":
        {"run": "consolidation",
         "job_name": "Consolidate + correct resize causality evidence"},
}

ENV_ATTRIBUTION_RULES = {
    "historical_legacy_inference_env":
        "authentic legacy pip freeze from the Legacy job log only",
    "historical_modern_inference_env":
        "authentic modern pip freeze from the Modern job log only",
    "historical_compare_job_env_record": {
        "pip_freeze_status": "not_recorded_in_source_run",
        "pip_freeze": None,
        "historical_environment_reconstructed": False,
        "rule": "the Compare job ran no pip freeze; embedded Legacy/Modern "
                "freezes in the assembled Compare log are the cat'd runner "
                "logs, NOT the Compare job's own environment, and must NOT be "
                "assigned to compare_env.",
    },
    "remediation_validation_env":
        "the NEW remediation job's own fully documented evidence-only "
        "environment (pip freeze --all); must contain no tensorflow/keras and "
        "must never be labelled the recovered historical Compare environment.",
}

SOURCE_INVENTORY_RULES = {
    "must_include": [
        "scripts/download_weights.py",
        "every directly invoked golden/ Python script",
        "every workflow YAML involved in producing/consolidating evidence",
        "every directly invoked repository shell script",
        "relevant production/upstream local Python files used by the historical "
        "inference jobs",
        "every new remediation and validation script",
    ],
    "per_file_fields": ["repository", "commit", "path", "role",
                        "source_job_or_remediation_job", "bytes", "sha256",
                        "git_blob_id"],
    "also": "hash the complete relevant local source trees at their immutable "
            "commits; record + hash inline workflow run blocks with job/step, "
            "not as separate files.",
}

METRIC_REVERIFICATION_RULES = {
    "open_npz_allow_pickle_false": True,
    "reject_object_arrays": True,
    "expected_arrays": 183,
    "hash_corrections": 20,
    "recompute": "exact frozen formulas/params from golden.causality_compare "
                 "(full_metrics: pearson/spearman/mae/rmse/max_abs_diff, "
                 "legacy_audit_global_ssim, windowed_ssim data_range=1.0)",
    "store_per_value": ["source_recorded", "previous_consolidated",
                        "newly_recomputed", "abs_diff", "consistent",
                        "source_arrays_keys", "impl_env_provenance"],
    "numeric_consistency_rule": {
        "array_data_sha256": "must match EXACTLY",
        "categorical_results": "must match EXACTLY",
        "pearson_spearman_ssim_mae_rmse": "abs_diff <= 1e-9",
        "max_abs_diff": "abs_diff <= 1e-6",
    },
    "provisional_preserved": ["hceye_*", "cognitive_load_index"],
    "no_inference": ["no tensorflow/keras package present",
                     "no checkpoint load", "no model construction",
                     "no prediction call", "no new saliency tensors"],
}

VALIDATION_CHECKS = [
    "strict JSON parse without NaN/Infinity",
    "four environment objects use correct attribution",
    "no false historical compare_env freeze remains",
    "complete remediation-environment freeze embedded",
    "every artifact has a verified source-job mapping",
    "scripts/download_weights.py and all directly invoked files hashed",
    "every JSON array reference exists and every array hash matches",
    "all 183 arrays remain numerically unchanged",
    "all 20 hash corrections preserved",
    "metric recomputation consistent (source/consolidated/recomputed)",
    "thresholds and conditions unchanged",
    "no model inference occurred",
    "protected refs and production files unchanged",
]

VERDICT_LOGIC = {
    "REMEDIATED": "every fixable provenance defect corrected AND the new "
                  "independent no-inference reverification passes (all array "
                  "hashes exact, all categorical results exact, all numeric "
                  "recomputations within the Phase-0 consistency rule, all 20 "
                  "corrections preserved, verdict supported) AND the bundle is "
                  "accessible -> 'UMSI RESIZE CAUSALITY EVIDENCE REMEDIATED — "
                  "SCIENTIFIC VERDICT UNCHANGED; HISTORICAL COMPARE FREEZE "
                  "TRANSPARENTLY UNAVAILABLE'",
    "BLOCKED": "any required artifact unmappable/missing/expired, any "
               "reverification inconsistency, or any fixable defect uncorrected "
               "-> 'UMSI RESIZE CAUSALITY EVIDENCE REMEDIATION BLOCKED — "
               "PROVENANCE OR REVERIFICATION INCOMPLETE'",
    "never": "do not describe the missing historical Compare freeze as "
             "recovered under either verdict.",
}

manifest = {
    "phase": "remediation_phase_0_protocol",
    "experiment": "umsi_resize_causality_evidence_remediation",
    "frozen_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "scope": "evidence remediation only; no model inference, no production "
             "resize fix, no norms rebuild.",
    "locked_refs": LOCKED_REFS,
    "source_runs": SOURCE_RUNS,
    "defects": DEFECTS,
    "artifact_job_map": ARTIFACT_JOB_MAP,
    "env_attribution_rules": ENV_ATTRIBUTION_RULES,
    "source_inventory_rules": SOURCE_INVENTORY_RULES,
    "metric_reverification_rules": METRIC_REVERIFICATION_RULES,
    "validation_checks": VALIDATION_CHECKS,
    "verdict_logic": VERDICT_LOGIC,
    "scientific_conclusion_unchanged":
        "The modern resize semantics were the sole material cause of the "
        "pre-registered golden-threshold failures in this controlled "
        "experiment.",
    "conclusion_scope": "NOT a claim that the complete TF1/TF2 pipelines are "
                        "bitwise identical.",
    "forbidden": "no TF1/TF2 inference; no checkpoint model load; no UMSI++ "
                 "instantiation; no raw-array regeneration; no stored-array "
                 "modification; no production edits; no rewrite of prior audit "
                 "branches; no assigning Legacy/Modern freeze to compare_env; "
                 "no threshold/formula/fixture/condition change; no norms "
                 "rebuild; no PR/merge.",
}

with open(OUT, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
print("locked refs:", len(LOCKED_REFS), "| defects:", len(DEFECTS),
      "| artifact_job_map:", len(ARTIFACT_JOB_MAP))
