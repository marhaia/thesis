"""Operator oracle for the legacy-resize production fix.

Proves, WITHOUT any TF1 runtime, that the production
``saliency.umsi_model.LegacyBilinearUpSampling2D`` helper reproduces the
verified legacy bilinear operator
(``tf.raw_ops.ResizeBilinear(align_corners=False, half_pixel_centers=False)``)
bit-for-bit at exactly the three decoder boundaries, and that this differs from
the TF2/Keras3 half-pixel ``UpSampling2D`` it replaces.

The genuine-TF1.14 equivalence of that operator was established (max_abs_diff =
0.0 on ramp+checkerboard at all three decoder shapes) by the resize causality
experiment (run 30034973775, commit a03c42a); this oracle re-verifies the
production helper against that same operator on the same deterministic inputs
and cross-checks the stored oracle result.

Writes a JSON report and exits non-zero on any mismatch.
"""
import os
import sys
import json
import argparse

import numpy as np
import tensorflow as tf
from keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resize_common as RC  # noqa: E402

REPO = os.environ.get("UMSI_REPO", os.path.dirname(HERE))
sys.path.insert(0, REPO)
from saliency.umsi_model import LegacyBilinearUpSampling2D  # noqa: E402


def _legacy_rawop(x, fh, fw):
    sh = tf.shape(x)
    tgt = tf.stack([sh[1] * fh, sh[2] * fw])
    return tf.raw_ops.ResizeBilinear(
        images=x, size=tgt, align_corners=False,
        half_pixel_centers=False).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--stored-oracle", default="",
                    help="optional resize-causality JSON with the stored "
                         "phase1_resize_operator_oracle to cross-check")
    args = ap.parse_args()

    oi = RC.oracle_inputs()
    per_tensor = {}
    worst_vs_legacy = 0.0
    worst_vs_halfpixel = 0.0
    all_pass = True

    print("=" * 72, flush=True)
    print("OPERATOR ORACLE — production helper vs verified legacy operator",
          flush=True)
    print("=" * 72, flush=True)
    for key in sorted(oi):
        arr = oi[key]
        layer_name = key.split("_", 1)[1]
        d = next(x for x in RC.DECODER_RESIZE if x["name"] == layer_name)
        fh, fw = d["fh"], d["fw"]
        x = tf.constant(arr)
        helper = LegacyBilinearUpSampling2D(size=(fh, fw))(x).numpy()
        legacy = _legacy_rawop(x, fh, fw)
        halfpix = layers.UpSampling2D(size=(fh, fw),
                                      interpolation="bilinear")(x).numpy()
        vs_legacy = float(np.abs(helper - legacy).max())
        vs_hp = float(np.abs(helper - halfpix).max())
        shapes_ok = (helper.shape == legacy.shape ==
                     (1, d["out_h"], d["out_w"], arr.shape[3]))
        finite = bool(np.isfinite(helper).all())
        ok = (vs_legacy == 0.0) and shapes_ok and finite
        all_pass = all_pass and ok
        worst_vs_legacy = max(worst_vs_legacy, vs_legacy)
        worst_vs_halfpixel = max(worst_vs_halfpixel, vs_hp)
        per_tensor[key] = {
            "layer": layer_name,
            "factor": [fh, fw],
            "out_shape": list(helper.shape),
            "helper_vs_legacy_max_abs_diff": vs_legacy,
            "helper_vs_halfpixel_max_abs_diff": vs_hp,
            "shapes_match": shapes_ok,
            "finite": finite,
            "pass": ok,
        }
        print("%-20s out=%s helper==legacy maxabs=%.3e  vs_halfpixel=%.4f  %s"
              % (key, helper.shape, vs_legacy, vs_hp,
                 "PASS" if ok else "FAIL"), flush=True)

    # differs-from-halfpixel is required so we prove the fix changes behaviour.
    changed_behaviour = worst_vs_halfpixel > 0.0

    stored = None
    stored_ok = None
    if args.stored_oracle and os.path.exists(args.stored_oracle):
        sd = json.load(open(args.stored_oracle))
        stored = sd.get("phase1_resize_operator_oracle", {})
        pt = stored.get("per_tensor", {})
        stored_ok = bool(stored.get("oracle_pass")) and all(
            v.get("max_abs_diff") == 0.0 for v in pt.values())
        print("\nstored resize-causality oracle_pass=%s (all max_abs_diff==0: %s)"
              % (stored.get("oracle_pass"), stored_ok), flush=True)

    result = {
        "oracle_kind": "production_helper_vs_verified_legacy_operator",
        "operator": ("tf.raw_ops.ResizeBilinear(align_corners=False, "
                     "half_pixel_centers=False)"),
        "decoder_boundaries": RC.DECODER_RESIZE,
        "per_tensor": per_tensor,
        "worst_helper_vs_legacy_max_abs_diff": worst_vs_legacy,
        "worst_helper_vs_halfpixel_max_abs_diff": worst_vs_halfpixel,
        "helper_reproduces_legacy_operator_bitwise": all_pass,
        "helper_differs_from_halfpixel": changed_behaviour,
        "stored_resize_causality_oracle_pass": stored_ok,
        "oracle_pass": bool(all_pass and changed_behaviour and
                            (stored_ok is None or stored_ok)),
    }
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print("\nworst helper-vs-legacy  : %.3e (must be 0.0)" % worst_vs_legacy,
          flush=True)
    print("worst helper-vs-halfpix : %.4f (must be > 0)" % worst_vs_halfpixel,
          flush=True)
    print("ORACLE PASS: %s" % result["oracle_pass"], flush=True)
    if not result["oracle_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
