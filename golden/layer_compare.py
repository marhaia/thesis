"""On-failure intermediate-activation comparison.

Loads the per-layer activation captures from both runtimes (one fixture) and
computes, for each shared layer along the network spine, the Pearson
correlation and max/relative absolute difference, in depth order. Identifies the
FIRST layer whose Pearson drops below the parity threshold — i.e. the first
materially divergent layer. Read-only; does not modify or "fix" any code.
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def _pearson(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-npz", required=True)
    ap.add_argument("--modern-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fixture", default=common.LAYER_DIAG_FIXTURE)
    args = ap.parse_args()

    L = np.load(args.legacy_npz, allow_pickle=True)
    M = np.load(args.modern_npz, allow_pickle=True)
    names_l = [str(x) for x in L["__names__"]]
    names_m = [str(x) for x in M["__names__"]]
    common_names = [n for n in common.LAYER_SPINE
                    if n in names_l and n in names_m]

    thr = common.THRESHOLDS["pearson_min"]
    rows = []
    first_divergent = None
    for n in common_names:
        a = L[n]
        b = M[n]
        shape_match = (a.shape == b.shape)
        p = _pearson(a, b) if shape_match else None
        if shape_match:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = float(d.max())
            denom = float(np.abs(a.astype(np.float64)).max()) or 1.0
            rel = max_abs / denom
        else:
            max_abs = None
            rel = None
        diverged = (p is None) or (p < thr) or (not shape_match)
        rows.append({
            "layer": n,
            "shape_legacy": list(a.shape),
            "shape_modern": list(b.shape),
            "shape_match": shape_match,
            "pearson": p,
            "max_abs_diff": max_abs,
            "rel_max_abs_diff": rel,
            "below_threshold": diverged,
        })
        if diverged and first_divergent is None:
            first_divergent = n

    result = {
        "evidence_kind": "intermediate_activation_first_divergent_layer",
        "fixture": args.fixture,
        "pearson_threshold": thr,
        "compared_layers": common_names,
        "first_materially_divergent_layer": first_divergent,
        "per_layer": rows,
        "note": ("Depth-ordered spine shared by both graphs. The first layer "
                 "with Pearson < %.3f marks where the TF1.14/Keras2.3.1 and "
                 "TF2.16/Keras3 activations first materially diverge." % thr),
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print("\n========== INTERMEDIATE-ACTIVATION LAYER SWEEP (%s) ==========" % args.fixture, flush=True)
    print("%-24s %10s %14s %10s %s" % (
        "layer", "pearson", "max_abs_diff", "rel", "shape"), flush=True)
    for r in rows:
        p = ("%.6f" % r["pearson"]) if r["pearson"] is not None else "n/a"
        ma = ("%.4g" % r["max_abs_diff"]) if r["max_abs_diff"] is not None else "n/a"
        rl = ("%.4g" % r["rel_max_abs_diff"]) if r["rel_max_abs_diff"] is not None else "n/a"
        flag = "  <-- first divergent" if r["layer"] == first_divergent else ""
        print("%-24s %10s %14s %10s %s%s" % (
            r["layer"], p, ma, rl, r["shape_legacy"], flag), flush=True)
    print("\nfirst_materially_divergent_layer:", first_divergent, flush=True)
    print("wrote %s" % args.out_json, flush=True)


if __name__ == "__main__":
    main()
