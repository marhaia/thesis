#!/usr/bin/env python3
"""XR-04 current launcher for the five-fixture P2/AG-05 evidence gate.

The comparison and inference logic remains in the existing successor gate
library. This launcher adds only the current-v2 lifecycle and prospective exact
production-output identity checks. Historical contracts and reports are never
modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from stage1.tools import umsi_production_postprocess_gate as base_gate


def validate_current_report(
    contract: Mapping[str, Any], report: Mapping[str, Any]
) -> dict:
    """Fail closed unless the real report satisfies the current-v2 additions."""
    if contract.get("lifecycle") != "CURRENT_SUCCESSOR":
        raise base_gate.ContractError("current contract lifecycle mismatch")
    if contract.get("remediation_finding") != "XR-04":
        raise base_gate.ContractError("current contract remediation finding mismatch")
    if report.get("overall_verdict") != "PASS":
        raise base_gate.ContractError("base five-fixture gate did not pass")

    required = contract.get("required_current_execution")
    expected_hashes = contract.get("expected_production_e2e_sha256")
    if not isinstance(required, dict) or not isinstance(expected_hashes, dict):
        raise base_gate.ContractError("current execution requirements are missing")
    expected_order = required.get("fixture_order")
    if report.get("fixture_order") != expected_order:
        raise base_gate.ContractError("current fixture order mismatch")
    if list(expected_hashes) != expected_order:
        raise base_gate.ContractError("expected output identity order mismatch")

    observed_hashes = {}
    for fixture in expected_order:
        try:
            result = report["per_fixture"][fixture]
            observed_sha = result["observed"]["production_e2e_sha256"]
        except (KeyError, TypeError) as exc:
            raise base_gate.ContractError(
                f"current report output identity missing: {fixture}"
            ) from exc
        if result.get("status") != "PASS" or result.get("passed") is not True:
            raise base_gate.ContractError(
                f"current fixture did not pass: {fixture}"
            )
        if observed_sha != expected_hashes.get(fixture):
            raise base_gate.ContractError(
                f"current production E2E SHA-256 mismatch: {fixture}"
            )
        observed_hashes[fixture] = observed_sha

    return {
        "passed": True,
        "lifecycle": "CURRENT_SUCCESSOR",
        "remediation_finding": "XR-04",
        "exact_production_e2e_sha256": observed_hashes,
        "historical_artifacts_mutated": False,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--reference-contract", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--evidence-npz", required=True, type=Path)
    parser.add_argument("--fixtures-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    base_argv = [
        "--contract", str(args.contract),
        "--reference-contract", str(args.reference_contract),
        "--weights", str(args.weights),
        "--evidence-npz", str(args.evidence_npz),
        "--fixtures-root", str(args.fixtures_root),
        "--output", str(args.output),
    ]
    base_result = base_gate.main(base_argv)
    if base_result != 0:
        return base_result

    contract = base_gate.load_and_validate_contract(args.contract)
    report = json.loads(args.output.read_text(encoding="utf-8"))
    try:
        report["current_contract_checks"] = validate_current_report(
            contract, report
        )
    except base_gate.ContractError as exc:
        report["current_contract_checks"] = {
            "passed": False,
            "reason": str(exc),
        }
        report["overall_verdict"] = "FAILED"
        _write_report(args.output, report)
        print(json.dumps({"overall_verdict": "FAILED", "reason": str(exc)}))
        return 1

    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "overall_verdict": "PASS",
                "current_contract_checks": "PASS",
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
