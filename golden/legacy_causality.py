"""Legacy causality runner — produces the L_REF reference and the legacy-side
inputs for the UMSI++ resize causality experiment.

Runs inside the x86_64 Linux legacy container (Python 3.7.9 / TF 1.14.0 /
Keras 2.3.1). Reuses the unmodified upstream model + preprocessing via the
existing legacy_runner helpers. It additionally:

  1. Saves each fixture's GENUINE legacy preprocessing tensor as an
     uncompressed float32 .npy (the byte-identical input control for Phase 2).
  2. Runs the resize-operator ORACLE: the genuine Keras 2.3.1
     UpSampling2D(interpolation='bilinear') on deterministic ramp/checkerboard
     tensors for every decoder source/target shape.
  3. Captures per-resize-boundary activations (input + output of each
     dec_upsN, plus dec_c_cout) for the boundary fixture.

No output suppression; warnings captured, not silenced.
"""
import os
import sys
import json
import time
import platform
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (FIXTURE_ORDER, FIXTURES, sha256_file,  # noqa: E402
                    sha256_array)
import resize_common as RC  # noqa: E402
# Reuse the proven legacy model/preproc helpers (module reads env at import).
from legacy_runner import (build_and_load, raw_predict, genuine_e2e,  # noqa: E402
                           _fixture_path, EXECUTED_SOURCE, LEGACY_SRC, CKPT,
                           run_worker)


def resize_oracle():
    """Genuine Keras 2.3.1 bilinear UpSampling2D on deterministic tensors."""
    import keras
    from keras.layers import Input, UpSampling2D
    from keras.models import Model as KModel
    inputs = RC.oracle_inputs()
    out_arrays = {}
    meta = {}
    for key in sorted(inputs):
        x = inputs[key].astype(np.float32)
        layer = key.split("_", 1)[1]
        d = next(v for v in RC.DECODER_RESIZE if v["name"] == layer)
        inp = Input(shape=x.shape[1:])
        up = UpSampling2D(size=(d["fh"], d["fw"]),
                          interpolation="bilinear")(inp)
        m = KModel(inp, up)
        y = np.asarray(m.predict(x, verbose=0)).astype(np.float32)
        out_arrays[key] = y
        meta[key] = {
            "in_shape": list(x.shape), "out_shape": list(y.shape),
            "in_sha256": sha256_array(x), "out_sha256": sha256_array(y),
            "finite": bool(np.isfinite(y).all()),
        }
        print("  oracle %-18s in %s -> out %s finite=%s" % (
            key, x.shape, y.shape, bool(np.isfinite(y).all())), flush=True)
    return inputs, out_arrays, meta


def boundary_activations(model, x):
    """Capture input+output of each decoder resize, plus dec_c_cout."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker)          # reuse legacy fresh-process worker
        return

    t0 = time.time()
    print("=== LEGACY causality runner: TF1.14 / Keras 2.3.1 ===", flush=True)
    import tensorflow as tf
    import keras
    print("python  :", platform.python_version(), flush=True)
    print("platform:", platform.platform(), flush=True)
    print("tf      :", tf.__version__, flush=True)
    print("keras   :", keras.__version__, flush=True)
    print("numpy   :", np.__version__, flush=True)

    source_sha = {fn: sha256_file(os.path.join(LEGACY_SRC, fn))
                  for fn in EXECUTED_SOURCE}
    for fn in EXECUTED_SOURCE:
        print("source  : %s  %s" % (fn, source_sha[fn]), flush=True)
    ckpt_sha = sha256_file(CKPT)
    ckpt_bytes = os.path.getsize(CKPT)
    print("ckpt    : %d bytes  %s" % (ckpt_bytes, ckpt_sha), flush=True)

    model = build_and_load()

    per_fixture = {}
    npz = {}
    for name in FIXTURE_ORDER:
        h, w = FIXTURES[name]["h"], FIXTURES[name]["w"]
        path = _fixture_path(name)
        fx_sha = sha256_file(path)
        print("\n--- L_REF fixture %s (%dx%d) sha=%s ---" % (
            name, h, w, fx_sha), flush=True)

        preproc, raw1, classif = raw_predict(model, path)
        same_hashes = [sha256_array(raw1)]
        maps = [raw1]
        for _ in range(2):
            _, r, _ = raw_predict(model, path)
            same_hashes.append(sha256_array(r))
            maps.append(r)
        same_max_diff = max(
            [0.0] + [float(np.abs(maps[i] - maps[0]).max())
                     for i in range(1, len(maps))])
        e2e = genuine_e2e(raw1, h, w)

        # Phase 2 control: save legacy preprocessing tensor as uncompressed .npy
        preproc32 = np.ascontiguousarray(preproc.astype(np.float32))
        npy_path = os.path.join(args.out_dir, "legacy_preproc_%s.npy" % name)
        np.save(npy_path, preproc32, allow_pickle=False)
        npy_sha = sha256_file(npy_path)

        per_fixture[name] = {
            "dims_hw": [h, w], "file_sha256": fx_sha,
            "preproc": {
                "shape": list(preproc32.shape), "dtype": str(preproc32.dtype),
                "min": float(preproc32.min()), "max": float(preproc32.max()),
                "array_sha256": sha256_array(preproc32),
                "npy_file": os.path.basename(npy_path), "npy_sha256": npy_sha,
            },
            "raw": {
                "shape": list(raw1.shape), "dtype": str(raw1.dtype),
                "min": float(raw1.min()), "max": float(raw1.max()),
                "mean": float(raw1.mean()), "std": float(raw1.std()),
                "neg_pixel_pct": float((raw1 < 0).mean() * 100.0),
                "sha256": same_hashes[0],
            },
            "classif": [float(v) for v in classif],
            "e2e": {"shape": list(e2e.shape), "min": float(e2e.min()),
                    "max": float(e2e.max()), "sha256": sha256_array(e2e)},
            "same_process": {"hashes": same_hashes,
                             "unique_hashes": len(set(same_hashes)),
                             "max_abs_diff": same_max_diff},
        }
        npz["%s_preproc" % name] = preproc32
        npz["%s_raw" % name] = raw1.astype(np.float32)
        npz["%s_classif" % name] = classif.astype(np.float64)
        npz["%s_e2e" % name] = e2e.astype(np.float32)
        print("  saved %s sha=%s" % (os.path.basename(npy_path), npy_sha),
              flush=True)

    # Fresh-process repeatability for L_REF (3 independent processes).
    print("\n=== L_REF fresh-process repeatability (3 processes) ===", flush=True)
    import subprocess
    fresh = {name: [] for name in FIXTURE_ORDER}
    for k in range(3):
        wj = os.path.join(args.out_dir, "legacy_worker_%d.json" % k)
        subprocess.check_call([sys.executable, os.path.abspath(
            os.path.join(HERE, "legacy_runner.py")), "--worker", wj])
        wh = json.load(open(wj))
        for name in FIXTURE_ORDER:
            fresh[name].append(wh[name])
    for name in FIXTURE_ORDER:
        allh = [per_fixture[name]["same_process"]["hashes"][0]] + fresh[name]
        per_fixture[name]["fresh_process"] = {
            "hashes": fresh[name], "unique_hashes_incl_main": len(set(allh))}
        print("  %-12s fresh unique(incl main)=%d" % (
            name, len(set(allh))), flush=True)

    # Resize operator oracle (genuine Keras 2.3.1 bilinear).
    print("\n=== resize operator oracle (genuine legacy UpSampling2D) ===",
          flush=True)
    oin, oout, ometa = resize_oracle()
    np.savez_compressed(os.path.join(args.out_dir, "legacy_oracle_inputs.npz"),
                        **oin)
    np.savez_compressed(os.path.join(args.out_dir, "legacy_oracle.npz"), **oout)

    # Per-boundary activations for the boundary fixture.
    print("\n=== L_REF resize-boundary activations (%s) ===" % RC.BOUNDARY_FIXTURE,
          flush=True)
    from sal_imp_utilities import preprocess_images
    xb = np.asarray(preprocess_images(
        [_fixture_path(RC.BOUNDARY_FIXTURE)], 256, 256, pad=True))
    bnd = boundary_activations(model, xb)
    np.savez_compressed(os.path.join(
        args.out_dir, "legacy_boundary_%s.npz" % RC.BOUNDARY_FIXTURE), **bnd)
    print("  saved %d boundary tensors" % (len(bnd) - 1), flush=True)

    meta = {
        "environment": {
            "role": "legacy_tf1_keras2", "python": platform.python_version(),
            "platform": platform.platform(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__},
        "executed_source_sha256": source_sha,
        "checkpoint": {"sha256": ckpt_sha, "byte_size": ckpt_bytes, "path": CKPT},
        "fixtures": per_fixture,
        "oracle": ometa,
        "total_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "legacy_causality_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    np.savez_compressed(os.path.join(args.out_dir, "legacy_causality_arrays.npz"),
                        **npz)
    print("\nSAVED legacy_causality_meta.json + legacy_causality_arrays.npz",
          flush=True)


if __name__ == "__main__":
    main()
