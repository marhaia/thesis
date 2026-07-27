"""Modern UMSI++ runner — executes the reviewed TF2.16 / Keras 3.10 stack.

Runs inside the x86_64 Linux modern container (Python 3.9.6). It imports the
repository's own saliency code at commit 6a17288 (saliency.umsi_model) and
records, per fixture, exactly the same quantities as legacy_runner.py so the
comparator can line them up: preprocessing tensor, RAW 512x512 heatmap, aux
classif (6,), the implementation's GENUINE end-to-end map
(postprocess_saliency), plus 3 same-process and 3 fresh-process repeats.

No output suppression; warnings captured (not silenced); stdout/stderr via tee.
Read-only w.r.t. the repository production code.
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
from common import (FIXTURE_ORDER, FIXTURES,  # noqa: E402
                    sha256_file, sha256_array)

REPO = os.environ["UMSI_REPO"]
FIXTURES_DIR = os.environ["UMSI_FIXTURES_DIR"]
CKPT = os.environ["UMSI_CKPT"]
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


def build_and_load():
    from saliency.umsi_model import UMSIPlus
    return UMSIPlus(CKPT)


def raw_predict(model, path):
    """Modern preprocessing + raw model output. Returns
    (preproc (1,256,256,3), raw (512,512), classif (6,))."""
    from saliency.umsi_model import preprocess_image
    x = preprocess_image(path)
    x = np.asarray(x)
    preds = model.model.predict(x, verbose=0)
    heatmap = np.asarray(preds[0])[0, :, :, 0].astype(np.float32)
    classif = np.asarray(preds[1])[0].astype(np.float64)
    return x, heatmap, classif


def genuine_e2e(raw, h, w):
    """The modern implementation's own end-to-end map: postprocess_saliency
    (min-max [0,1], its production path)."""
    from saliency.umsi_model import postprocess_saliency
    return postprocess_saliency(raw, h, w).astype(np.float32)


def run_worker(out_json):
    model = build_and_load()
    hashes = {}
    for name in FIXTURE_ORDER:
        _, raw, _ = raw_predict(model, _fixture_path(name))
        hashes[name] = sha256_array(raw)
    with open(out_json, "w") as fh:
        json.dump(hashes, fh)


def run_layers(fixture, out_npz):
    """Capture intermediate activations for the on-failure layer sweep."""
    import keras
    from common import LAYER_SPINE
    from saliency.umsi_model import preprocess_image
    wrapper = build_and_load()
    km = wrapper.model               # underlying keras model
    names = [n for n in LAYER_SPINE
             if n in [l.name for l in km.layers]]
    sub = keras.Model(inputs=km.inputs,
                      outputs=[km.get_layer(n).output for n in names])
    x = np.asarray(preprocess_image(_fixture_path(fixture)))
    outs = sub.predict(x, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]
    saved = {"__names__": np.array(names)}
    for n, a in zip(names, outs):
        # float32 (not float16): decoder activations reach ~1e6 and would
        # overflow float16 (max 65504) to +inf, corrupting the comparison.
        saved[n] = np.asarray(a)[0].astype(np.float32)
    np.savez_compressed(out_npz, **saved)
    print("SAVED layers %s (%d layers)" % (out_npz, len(names)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--layers-out", default="")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker)
        return
    if args.layers:
        run_layers(args.layers, args.layers_out)
        return

    t0 = time.time()
    print("=== MODERN runner: TF2.16 / Keras 3 ===", flush=True)
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

    model = build_and_load()

    per_fixture = {}
    npz = {}
    for name in FIXTURE_ORDER:
        h = FIXTURES[name]["h"]
        w = FIXTURES[name]["w"]
        path = _fixture_path(name)
        fx_sha = sha256_file(path)
        print("\n--- fixture %s (%dx%d) sha=%s ---" % (name, h, w, fx_sha), flush=True)

        preproc, raw1, classif = raw_predict(model, path)
        same_hashes = [sha256_array(raw1)]
        maps = [raw1]
        for _ in range(2):
            _, r, _ = raw_predict(model, path)
            same_hashes.append(sha256_array(r))
            maps.append(r)
        same_max_diff = 0.0
        for i in range(1, len(maps)):
            same_max_diff = max(same_max_diff,
                                float(np.abs(maps[i] - maps[0]).max()))

        e2e = genuine_e2e(raw1, h, w)

        per_fixture[name] = {
            "dims_hw": [h, w],
            "file_sha256": fx_sha,
            "preproc": {
                "shape": list(preproc.shape),
                "dtype": str(preproc.dtype),
                "min": float(preproc.min()),
                "max": float(preproc.max()),
                "sha256": sha256_array(preproc),
            },
            "raw": {
                "shape": list(raw1.shape),
                "dtype": str(raw1.dtype),
                "min": float(raw1.min()),
                "max": float(raw1.max()),
                "mean": float(raw1.mean()),
                "std": float(raw1.std()),
                "neg_pixel_pct": float((raw1 < 0).mean() * 100.0),
                "sha256": same_hashes[0],
            },
            "classif": [float(v) for v in classif],
            "e2e": {
                "shape": list(e2e.shape),
                "min": float(e2e.min()),
                "max": float(e2e.max()),
                "sha256": sha256_array(e2e),
            },
            "same_process": {
                "hashes": same_hashes,
                "unique_hashes": len(set(same_hashes)),
                "max_abs_diff": same_max_diff,
            },
        }
        npz["%s_preproc" % name] = preproc.astype(np.float32)
        npz["%s_raw" % name] = raw1.astype(np.float32)
        npz["%s_classif" % name] = classif.astype(np.float64)
        npz["%s_e2e" % name] = e2e.astype(np.float32)

    print("\n=== fresh-process repeatability (3 independent processes) ===", flush=True)
    fresh_hashes = {name: [] for name in FIXTURE_ORDER}
    for k in range(3):
        wj = os.path.join(args.out_dir, "modern_worker_%d.json" % k)
        subprocess.check_call(
            [sys.executable, os.path.abspath(__file__), "--worker", wj])
        with open(wj) as fh:
            wh = json.load(fh)
        for name in FIXTURE_ORDER:
            fresh_hashes[name].append(wh[name])
    for name in FIXTURE_ORDER:
        allh = [per_fixture[name]["same_process"]["hashes"][0]] + fresh_hashes[name]
        per_fixture[name]["fresh_process"] = {
            "hashes": fresh_hashes[name],
            "unique_hashes_incl_main": len(set(allh)),
        }
        print("  %-12s fresh unique(incl main)=%d" % (
            name, len(set(allh))), flush=True)

    meta = {
        "environment": {
            "role": "modern_tf2_keras3",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
        },
        "checkpoint": {"sha256": ckpt_sha, "byte_size": ckpt_bytes, "path": CKPT},
        "fixtures": per_fixture,
        "captured_warnings": _CAPTURED,
        "total_seconds": time.time() - t0,
    }
    out_meta = os.path.join(args.out_dir, "modern_meta.json")
    out_npz = os.path.join(args.out_dir, "modern_arrays.npz")
    with open(out_meta, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    np.savez_compressed(out_npz, **npz)
    print("\nSAVED %s" % out_meta, flush=True)
    print("SAVED %s" % out_npz, flush=True)
    print("captured_warnings=%d" % len(_CAPTURED), flush=True)


if __name__ == "__main__":
    main()
