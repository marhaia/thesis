"""Structural call-site verification for the legacy-resize production fix.

Mechanically proves — by building the actual production model and inspecting the
instantiated layer objects (not by reading source text) — that EXACTLY the three
intended decoder boundaries (dec_ups1, dec_ups2, dec_ups3) use the
``LegacyBilinearUpSampling2D`` compatibility layer and that NO other layer in the
graph does, and that no stock ``UpSampling2D`` remains.

Exits non-zero if the call sites are not exactly the three approved ones.
"""
import os
import sys
import json
import argparse

from keras import layers

REPO = os.environ.get("UMSI_REPO",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from saliency.umsi_model import (build_umsi_model,  # noqa: E402
                                 LegacyBilinearUpSampling2D)

EXPECTED = {"dec_ups1", "dec_ups2", "dec_ups3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    model = build_umsi_model(verbose=False)
    legacy_sites = []
    stock_upsampling = []
    for lyr in model.layers:
        if isinstance(lyr, LegacyBilinearUpSampling2D):
            legacy_sites.append({"name": lyr.name, "size": list(lyr.size)})
        elif isinstance(lyr, layers.UpSampling2D):
            stock_upsampling.append(lyr.name)

    legacy_names = {s["name"] for s in legacy_sites}
    exactly_three = (legacy_names == EXPECTED)
    no_stock = (len(stock_upsampling) == 0)
    ok = exactly_three and no_stock and len(legacy_sites) == 3

    result = {
        "check_kind": "structural_call_site_verification",
        "method": "instantiated keras layer object inspection",
        "expected_legacy_sites": sorted(EXPECTED),
        "found_legacy_sites": sorted(legacy_names),
        "legacy_site_details": legacy_sites,
        "remaining_stock_upsampling2d": stock_upsampling,
        "exactly_three_expected_sites": exactly_three,
        "no_stock_upsampling_remaining": no_stock,
        "callsite_check_pass": ok,
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print("legacy sites  :", sorted(legacy_names), flush=True)
    print("stock UpSampling2D remaining:", stock_upsampling, flush=True)
    print("CALLSITE CHECK PASS:", ok, flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
