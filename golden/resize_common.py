"""Shared constants + deterministic oracle inputs for the UMSI++ resize
causality experiment.

Depends ONLY on the Python standard library + numpy so it imports cleanly in
BOTH runtimes (legacy TF1.14/Keras2.3.1/numpy1.19.2 and modern
TF2.16/Keras3.10/numpy1.26.4).

The three bilinear resize operations in the UMSI++ decoder (from the frozen
production graph saliency/umsi_model.py at 6a17288) are, in execution order:

    dec_ups1 : UpSampling2D(size=(2,2), bilinear)  in (32,32,256)  -> (64,64,256)
    dec_ups2 : UpSampling2D(size=(2,2), bilinear)  in (64,64,128)  -> (128,128,128)
    dec_ups3 : UpSampling2D(size=(4,4), bilinear)  in (128,128,64) -> (512,512,64)

The modern candidate "legacy resize" operator (audit-only; NOT a production
edit) is:

    tf.raw_ops.ResizeBilinear(images=x, size=[out_h, out_w],
                              align_corners=False, half_pixel_centers=False)

which reproduces the TF1.14 / Keras 2.3.1 UpSampling2D(interpolation='bilinear')
semantics. align_corners=True is deliberately NOT used or tested.
"""
import numpy as np

# Ordered decoder resize layers with (name, src_h, src_w, factor_h, factor_w).
# Channel count is irrelevant to bilinear resize (applied per channel); the
# oracle probes the SPATIAL source/target shapes actually used by the decoder.
DECODER_RESIZE = [
    {"name": "dec_ups1", "src_h": 32,  "src_w": 32,  "fh": 2, "fw": 2,
     "out_h": 64,  "out_w": 64},
    {"name": "dec_ups2", "src_h": 64,  "src_w": 64,  "fh": 2, "fw": 2,
     "out_h": 128, "out_w": 128},
    {"name": "dec_ups3", "src_h": 128, "src_w": 128, "fh": 4, "fw": 4,
     "out_h": 512, "out_w": 512},
]

DECODER_RESIZE_NAMES = [d["name"] for d in DECODER_RESIZE]

# The single fixture used for the per-boundary intermediate-activation
# comparison (kept small; matches the golden layer sweep fixture).
BOUNDARY_FIXTURE = "ui1"

# Cumulative ablation sets, in execution order (Phase 3).
ABLATION_SETS = {
    "abl_ups1":      ["dec_ups1"],
    "abl_ups1_ups2": ["dec_ups1", "dec_ups2"],
    "abl_all":       ["dec_ups1", "dec_ups2", "dec_ups3"],
}

# The five model conditions (Phase 3). "input" = which preprocessing tensor;
# "resize" = which resize operator the decoder uses.
CONDITIONS = {
    "L_REF": {"input": "legacy_preproc",   "resize": "legacy_native"},
    "M_CC":  {"input": "modern_preproc",   "resize": "modern"},
    "M_LC":  {"input": "legacy_controlled", "resize": "modern"},
    "M_CL":  {"input": "modern_preproc",   "resize": "legacy"},
    "M_LL":  {"input": "legacy_controlled", "resize": "legacy"},
}

# Oracle acceptance threshold for the legacy-vs-candidate resize operator.
ORACLE_MAX_ABS_DIFF = 1e-6

# Small channel count for the oracle probe tensors (resize is per-channel).
ORACLE_CHANNELS = 3


def oracle_inputs():
    """Deterministic float32 ramp + checkerboard tensors for every decoder
    source shape. Returns a dict name -> (1, H, W, C) float32 array. The exact
    same bytes are consumed by the legacy operator and the modern candidate."""
    out = {}
    for d in DECODER_RESIZE:
        h, w, c = d["src_h"], d["src_w"], ORACLE_CHANNELS
        # Ramp: separable linear gradient in H and W, per-channel offset.
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w, 1)
        cc = np.arange(c, dtype=np.float32).reshape(1, 1, c) * 0.1
        ramp = (yy + xx + cc).astype(np.float32)[None, ...]
        out["ramp_%s" % d["name"]] = np.ascontiguousarray(ramp)
        # Checkerboard: high-frequency pattern to stress interpolation.
        iy = np.arange(h).reshape(h, 1, 1)
        ix = np.arange(w).reshape(1, w, 1)
        chk = (((iy + ix) % 2) * 1.0).astype(np.float32)
        chk = np.repeat(chk, c, axis=2)[None, ...]
        out["checker_%s" % d["name"]] = np.ascontiguousarray(
            chk.astype(np.float32))
    return out


def oracle_target(name):
    """Return (out_h, out_w) for an oracle input tensor keyed by
    'ramp_<layer>' or 'checker_<layer>'."""
    layer = name.split("_", 1)[1]
    d = next(x for x in DECODER_RESIZE if x["name"] == layer)
    return d["out_h"], d["out_w"]
