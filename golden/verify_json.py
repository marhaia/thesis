"""Tiny helper to strict-parse the golden result JSON and optionally gate on it.

Usage:
    python verify_json.py <path-to-json>          # print summary, always exit 0
    python verify_json.py <path-to-json> --gate   # exit 0 iff all fixtures pass
"""
import sys
import json


def main():
    path = sys.argv[1]
    gate = "--gate" in sys.argv[2:]
    d = json.load(open(path))
    print("strict_json_parse: OK")
    print("verdict:", d["overall"]["verdict"])
    print("all_fixtures_pass:", d["overall"]["all_fixtures_pass"])
    print("fixtures:", list(d["fixtures"].keys()))
    if gate:
        sys.exit(0 if d["overall"]["all_fixtures_pass"] else 1)


if __name__ == "__main__":
    main()
