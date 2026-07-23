"""Strict-parse the resize causality result JSON and print a summary.

Usage:
    python verify_causality.py <path-to-json>   # strict parse; always exit 0

The causality experiment reports its verdict regardless of outcome, so this
helper never gates the job; it only proves the deliverable is strict-parseable
and echoes the verdict + decision flags into the assembled log.
"""
import sys
import json


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    print("strict_json_parse: OK")
    print("experiment:", d.get("experiment"))
    print("verdict:", d["verdict"])
    print("oracle_pass:", d["decision"]["oracle_pass"])
    print("M_LC_reproduces_failure:", d["decision"]["M_LC_reproduces_failure"])
    print("M_LL_passes_all_fixtures:", d["decision"]["M_LL_passes_all_fixtures"])
    print("M_CL_passes_all_fixtures:", d["decision"]["M_CL_passes_all_fixtures"])
    print("activation_supports_hypothesis:",
          d["decision"]["activation_supports_hypothesis"])
    print("first_divergent_boundary_all_modern_resize:",
          d.get("first_divergent_boundary_all_modern_resize"))
    print("condition_all_pass:", json.dumps(d["phase3_condition_all_pass"]))


if __name__ == "__main__":
    main()
