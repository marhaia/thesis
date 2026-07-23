"""Modern causality runner — executes every controlled condition of the UMSI++
resize causality experiment on the reviewed TF2.16 / Keras 3.10 stack.

Runs inside the x86_64 Linux modern container (Python 3.9.6). It:

  1. Loads each fixture's byte-identical legacy preprocessing tensor from the
     legacy .npy files (allow_pickle=False) — the Phase-2 input control — and
     also computes the genuine modern preprocessing tensor (kept separate).
  2. Builds the FROZEN production model and, via the audit-only variant builder
     (weights copied by name), the legacy-resize and cumulative-ablation
     variants. Only the decoder bilinear resize operator differs.
  3. Runs the model conditions M_CC / M_LC / M_CL / M_LL and the cumulative
     ablations (dec_ups1 / dec_ups1+dec_ups2 / all) under the byte-identical
     input, recording raw tensors, aux classif, genuine e2e maps and
     per-resize-boundary activations.
  4. Runs the resize-operator ORACLE candidate
     tf.raw_ops.ResizeBilinear(align_corners=False, half_pixel_centers=False)
     on the byte-identical legacy oracle inputs.
  5. Records same-process (x3) and fresh-process (x3) repeatability for M_CC
     and M_LL.

No output suppression; warnings captured, not silenced.
"""
import os
import sys
import json
import time
import platform
import argparse
import warnings
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FIXTURE_ORDER, FIXTURES, sha256_file, sha256_array  # noqa: E402
import resize_common as RC  # noqa: E402

REPO = os.environ["UMSI_REPO"]
FIXTURES_DIR = os.environ["UMSI_FIXTURES_DIR"]
CKPT = os.environ["UMSI_CKPT"]
LEGACY_DIR = os.environ["UMSI_LEGACY_DIR"]   # dir with legacy_preproc_*.npy etc
sys.path.insert(0, REPO)

warnings.simplefilter("always")
_CAPTURED = []
_orig = warnings.showwarning


def _cap(message, category, filename, lineno, file=None, line=None):
    _CAPTURED.append("%s: %s (%s:%d)" % (
        category.__name__, message, os.path.basename(filename), lineno))
    _orig(message, category, filename, lineno, file, line)


warnings.showwarning = _cap


def _fixture_path(name):
    return os.path.join(FIXTURES_DIR, name + ".png")


def modern_preproc(name):
    from saliency.umsi_model import preprocess_image
    return np.ascontiguousarray(
        np.asarray(preprocess_image(_fixture_path(name))).astype(np.float32))


def legacy_controlled(name):
    npy = os.path.join(LEGACY_DIR, "legacy_preproc_%s.npy" % name)
    return np.ascontiguousarray(np.load(npy, allow_pickle=False).astype(np.float32))


def raw_of(model, x):
    preds = model.predict(x, verbose=0)
    raw = np.asarray(preds[0])[0, :, :, 0].astype(np.float32)
    classif = np.asarray(preds[1])[0].astype(np.float64)
    return raw, classif


def e2e_of(raw, h, w):
    from saliency.umsi_model import postprocess_saliency
    return postprocess_saliency(raw, h, w).astype(np.float32)


def boundary_activations(model, x):
    import keras
    outs, names = [], []
    for rn in RC.DECODER_RESIZE_NAMES:
        lyr = model.get_layer(rn)
        outs += [lyr.input, lyr.output]
        names += [rn + "__in", rn + "__out"]
    outs += [model.get_layer("dec_c_cout").output]
    names += ["dec_c_cout"]
    sub = keras.Model(inputs=model.inputs, outputs=outs)
    vals = sub.predict(x, verbose=0)
    if not isinstance(vals, list):
        vals = [vals]
    saved = {"__names__": np.array(names)}
    for n, a in zip(names, vals):
        saved[n] = np.asarray(a)[0].astype(np.float32)
    return saved


def build_all_models():
    from audit_resize_model import (load_production_model,
                                    build_variant_with_weights)
    prod = load_production_model(CKPT)
    variants = {}
    var_all, c_all, p_all = build_variant_with_weights(
        CKPT, RC.ABLATION_SETS["abl_all"], src_model=prod)
    var_1, c1, p1 = build_variant_with_weights(
        CKPT, RC.ABLATION_SETS["abl_ups1"], src_model=prod)
    var_12, c12, p12 = build_variant_with_weights(
        CKPT, RC.ABLATION_SETS["abl_ups1_ups2"], src_model=prod)
    variants["abl_all"] = var_all
    variants["abl_ups1"] = var_1
    variants["abl_ups1_ups2"] = var_12
    info = {"prod_params": int(prod.count_params()),
            "abl_all": {"copied": c_all, "params": p_all},
            "abl_ups1": {"copied": c1, "params": p1},
            "abl_ups1_ups2": {"copied": c12, "params": p12}}
    return prod, variants, info


def run_worker(out_json):
    """Fresh-process: raw-heatmap SHA per fixture for M_CC and M_LL.

    Builds only the two models it needs (production + legacy-all variant)."""
    from audit_resize_model import (load_production_model,
                                    build_variant_with_weights)
    prod = load_production_model(CKPT)
    var_all, _, _ = build_variant_with_weights(
        CKPT, RC.ABLATION_SETS["abl_all"], src_model=prod)
    hashes = {"M_CC": {}, "M_LL": {}}
    for name in FIXTURE_ORDER:
        xm = modern_preproc(name)
        xl = legacy_controlled(name)
        r_cc, _ = raw_of(prod, xm)
        r_ll, _ = raw_of(var_all, xl)
        hashes["M_CC"][name] = sha256_array(r_cc)
        hashes["M_LL"][name] = sha256_array(r_ll)
    with open(out_json, "w") as fh:
        json.dump(hashes, fh)


def resize_oracle_candidate(out_dir):
    """Run the candidate legacy operator on byte-identical legacy oracle inputs."""
    import tensorflow as tf
    oin = np.load(os.path.join(LEGACY_DIR, "legacy_oracle_inputs.npz"))
    out_arrays = {}
    meta = {}
    for key in sorted(oin.files):
        x = np.ascontiguousarray(oin[key].astype(np.float32))
        oh, ow = RC.oracle_target(key)
        y = tf.raw_ops.ResizeBilinear(
            images=tf.constant(x), size=tf.constant([oh, ow], dtype=tf.int32),
            align_corners=False, half_pixel_centers=False)
        y = np.asarray(y).astype(np.float32)
        out_arrays[key] = y
        meta[key] = {"in_shape": list(x.shape), "out_shape": list(y.shape),
                     "in_sha256": sha256_array(x), "out_sha256": sha256_array(y),
                     "finite": bool(np.isfinite(y).all())}
        print("  candidate %-18s in %s -> out %s finite=%s" % (
            key, x.shape, y.shape, bool(np.isfinite(y).all())), flush=True)
    np.savez_compressed(os.path.join(out_dir, "modern_oracle_candidate.npz"),
                        **out_arrays)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker)
        return

    t0 = time.time()
    print("=== MODERN causality runner: TF2.16 / Keras 3 ===", flush=True)
    import tensorflow as tf
    import keras
    print("python  :", platform.python_version(), flush=True)
    print("platform:", platform.platform(), flush=True)
    print("tf      :", tf.__version__, flush=True)
    print("keras   :", keras.__version__, flush=True)
    print("numpy   :", np.__version__, flush=True)
    ckpt_sha = sha256_file(CKPT)
    ckpt_bytes = os.path.getsize(CKPT)
    print("ckpt    : %d bytes  %s" % (ckpt_bytes, ckpt_sha), flush=True)

    prod, variants, model_info = build_all_models()
    var_all = variants["abl_all"]
    print("model params: prod=%d  variants copied 107 layers each" %
          model_info["prod_params"], flush=True)

    # Model per condition (input chosen per fixture below).
    cond_model = {"M_CC": prod, "M_LC": prod, "M_CL": var_all, "M_LL": var_all}
    cond_input = {"M_CC": "modern", "M_LC": "legacy", "M_CL": "modern",
                  "M_LL": "legacy"}

    per_fixture = {}
    npz = {}
    for name in FIXTURE_ORDER:
        h, w = FIXTURES[name]["h"], FIXTURES[name]["w"]
        xm = modern_preproc(name)
        xl = legacy_controlled(name)
        # Byte-identity proof (control diff must be exactly 0.0 vs the .npy).
        legacy_npy_sha = sha256_file(
            os.path.join(LEGACY_DIR, "legacy_preproc_%s.npy" % name))
        ctrl_input_sha = sha256_array(xl)
        modern_vs_legacy_preproc = float(np.abs(
            xm.astype(np.float64) - xl.astype(np.float64)).max())
        print("\n--- fixture %s (%dx%d) ---" % (name, h, w), flush=True)
        print("  legacy-controlled input sha=%s (npy file sha=%s)" % (
            ctrl_input_sha, legacy_npy_sha), flush=True)
        print("  modern_preproc vs legacy_preproc max_abs_diff=%.3e" %
              modern_vs_legacy_preproc, flush=True)

        fx = {"dims_hw": [h, w],
              "legacy_controlled_input": {
                  "array_sha256": ctrl_input_sha,
                  "npy_file_sha256": legacy_npy_sha},
              "modern_preproc": {
                  "shape": list(xm.shape), "array_sha256": sha256_array(xm),
                  "min": float(xm.min()), "max": float(xm.max())},
              "modern_vs_legacy_preproc_max_abs_diff": modern_vs_legacy_preproc,
              "conditions": {}}

        npz["M_CC_%s_preproc" % name] = xm
        npz["legctrl_%s_input" % name] = xl

        for cond in ["M_CC", "M_LC", "M_CL", "M_LL"]:
            x = xm if cond_input[cond] == "modern" else xl
            raw, classif = raw_of(cond_model[cond], x)
            e2e = e2e_of(raw, h, w)
            fx["conditions"][cond] = {
                "input": cond_input[cond],
                "resize": RC.CONDITIONS[cond]["resize"],
                "raw": {"shape": list(raw.shape), "min": float(raw.min()),
                        "max": float(raw.max()), "mean": float(raw.mean()),
                        "std": float(raw.std()),
                        "neg_pixel_pct": float((raw < 0).mean() * 100.0),
                        "sha256": sha256_array(raw)},
                "classif": [float(v) for v in classif],
                "e2e": {"shape": list(e2e.shape), "min": float(e2e.min()),
                        "max": float(e2e.max()), "sha256": sha256_array(e2e)},
            }
            npz["%s_%s_raw" % (cond, name)] = raw
            npz["%s_%s_e2e" % (cond, name)] = e2e
            npz["%s_%s_classif" % (cond, name)] = classif.astype(np.float64)
            print("  %-5s raw[min %.3g max %.3g] neg%%=%.3f" % (
                cond, raw.min(), raw.max(), (raw < 0).mean() * 100.0),
                flush=True)

        # Cumulative ablations under byte-identical (legacy-controlled) input.
        for abl, mdl in (("abl_ups1", variants["abl_ups1"]),
                         ("abl_ups1_ups2", variants["abl_ups1_ups2"]),
                         ("abl_all", var_all)):
            raw, _ = raw_of(mdl, xl)
            fx["conditions"][abl] = {
                "input": "legacy_controlled", "resize_replaced":
                    RC.ABLATION_SETS[abl],
                "raw": {"shape": list(raw.shape), "min": float(raw.min()),
                        "max": float(raw.max()), "sha256": sha256_array(raw)}}
            npz["%s_%s_raw" % (abl, name)] = raw

        # Same-process repeatability for M_CC and M_LL (3 reps).
        cc_h = [npz["M_CC_%s_raw" % name]]
        ll_h = [npz["M_LL_%s_raw" % name]]
        for _ in range(2):
            r_cc, _ = raw_of(prod, xm)
            r_ll, _ = raw_of(var_all, xl)
            cc_h.append(r_cc)
            ll_h.append(r_ll)
        fx["same_process"] = {
            "M_CC_unique": len(set(sha256_array(a) for a in cc_h)),
            "M_CC_max_abs_diff": max(float(np.abs(cc_h[i] - cc_h[0]).max())
                                     for i in range(1, 3)),
            "M_LL_unique": len(set(sha256_array(a) for a in ll_h)),
            "M_LL_max_abs_diff": max(float(np.abs(ll_h[i] - ll_h[0]).max())
                                     for i in range(1, 3))}
        per_fixture[name] = fx

    # Per-boundary activations for the boundary fixture, every variant, under
    # byte-identical legacy-controlled input (plus production for reference).
    print("\n=== resize-boundary activations (%s, legacy-controlled input) ==="
          % RC.BOUNDARY_FIXTURE, flush=True)
    xb = legacy_controlled(RC.BOUNDARY_FIXTURE)
    for tag, mdl in (("M_LC_prod", prod), ("abl_ups1", variants["abl_ups1"]),
                     ("abl_ups1_ups2", variants["abl_ups1_ups2"]),
                     ("abl_all", var_all)):
        bnd = boundary_activations(mdl, xb)
        np.savez_compressed(os.path.join(
            args.out_dir, "modern_boundary_%s_%s.npz" % (
                tag, RC.BOUNDARY_FIXTURE)), **bnd)
        print("  saved boundary tensors for %s" % tag, flush=True)

    # Resize oracle candidate.
    print("\n=== resize operator oracle candidate (TF2 ResizeBilinear) ===",
          flush=True)
    oracle_meta = resize_oracle_candidate(args.out_dir)

    # Fresh-process repeatability for M_CC and M_LL.
    print("\n=== fresh-process repeatability (3 processes) ===", flush=True)
    fresh = {"M_CC": {n: [] for n in FIXTURE_ORDER},
             "M_LL": {n: [] for n in FIXTURE_ORDER}}
    for k in range(3):
        wj = os.path.join(args.out_dir, "modern_worker_%d.json" % k)
        subprocess.check_call([sys.executable, os.path.abspath(__file__),
                               "--worker", wj])
        wh = json.load(open(wj))
        for cond in ("M_CC", "M_LL"):
            for name in FIXTURE_ORDER:
                fresh[cond][name].append(wh[cond][name])
    for name in FIXTURE_ORDER:
        for cond in ("M_CC", "M_LL"):
            main_h = per_fixture[name]["conditions"][cond]["raw"]["sha256"]
            allh = [main_h] + fresh[cond][name]
            per_fixture[name].setdefault("fresh_process", {})[cond] = {
                "hashes": fresh[cond][name],
                "unique_hashes_incl_main": len(set(allh))}
        print("  %-12s M_CC uniq=%d  M_LL uniq=%d" % (
            name,
            per_fixture[name]["fresh_process"]["M_CC"]["unique_hashes_incl_main"],
            per_fixture[name]["fresh_process"]["M_LL"]["unique_hashes_incl_main"]),
            flush=True)

    meta = {
        "environment": {
            "role": "modern_tf2_keras3", "python": platform.python_version(),
            "platform": platform.platform(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__},
        "checkpoint": {"sha256": ckpt_sha, "byte_size": ckpt_bytes, "path": CKPT},
        "model_info": model_info,
        "fixtures": per_fixture,
        "oracle_candidate": oracle_meta,
        "captured_warnings": _CAPTURED,
        "total_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "modern_causality_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    np.savez_compressed(os.path.join(args.out_dir, "modern_causality_arrays.npz"),
                        **npz)
    print("\nSAVED modern_causality_meta.json + modern_causality_arrays.npz",
          flush=True)
    print("captured_warnings=%d" % len(_CAPTURED), flush=True)


if __name__ == "__main__":
    main()
