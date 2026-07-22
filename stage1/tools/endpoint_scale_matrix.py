"""Endpoint scale matrix — the TRUE two-path endpoint evidence.

This script drives the REAL Flask ``/api/cognitive-load`` route (via the Flask
test client) for every deterministic synthetic fixture at 1x/2x/3x and records
the two production scores the route actually returns:

  * ``cognitive_load_index``                 (HCEye combined index, h[5], 0..1)
  * ``layout.experimental_complexity_index`` (adjusted layout score, 0..100)

For each metric the per-scale values and the 1x/2x/3x gap are saved in both the
raw units and in *displayed points* (the index shown on a 0..100 scale, i.e.
``cognitive_load_index * 100``; the complexity index is already 0..100).

Two heavy ML stages are deterministically neutralised so the artifact can be
regenerated without model weights, and every neutralised component is written
into the report under ``mocked_components`` (exactly as required):

  * UMSI++ saliency is disabled -> the endpoint degrades to ``s = None``.
  * EasyOCR ``compute_readability`` is disabled -> ``text_density`` uses the
    neutral fallback (``text_density_source = "fallback_neutral"``).

Everything else (the canonical layout path, the eight visual features, the
HCEye rule mapping, the task/profile modifiers and the native element / Jokinen
path) runs through the real route unchanged.

This is DISTINCT from ``fixture_scale_report.json`` (the reduced HCEye-rule
calculation produced by ``canonical_scale_eval.py``): that file is the
feature-only rule index and must never be substituted for this endpoint matrix.

Usage:
    python stage1/tools/endpoint_scale_matrix.py [--out-dir DIR]
"""

import argparse
import io
import json
import os
import platform
import sys

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1_DIR = os.path.dirname(_TOOLS_DIR)
_REPO_ROOT = os.path.dirname(_STAGE1_DIR)
for _p in (_STAGE1_DIR, _REPO_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

from canonical_scale_eval import FIXTURES  # noqa: E402


SCALES = (1, 2, 3)

# Human-readable declaration of exactly which components were neutralised for a
# deterministic, weight-free regeneration of this matrix.
MOCKED_COMPONENTS = [
    {
        "component": "saliency_umsi_plus_plus",
        "route_symbol": "app._predict_saliency_cached",
        "effect": "raises -> endpoint sets s=None (saliency vector = zeros(5))",
    },
    {
        "component": "ocr_easyocr_compute_readability",
        "route_symbol": "cognitive.text_reader.compute_readability",
        "effect": ("returns None -> text_density stays neutral fallback "
                   "(text_density_source='fallback_neutral')"),
    },
]


def _install_mocks():
    """Neutralise saliency and OCR on the real route (deterministic, no weights).

    The native element detector and the Jokinen search model are left REAL:
    they are deterministic and do not affect either saved score.
    """
    import app as app_module
    import cognitive.text_reader as text_reader_mod

    def _no_saliency(*_a, **_k):
        raise RuntimeError("saliency disabled for endpoint scale matrix")

    def _no_ocr(*_a, **_k):
        return None

    app_module._predict_saliency_cached = _no_saliency
    text_reader_mod.compute_readability = _no_ocr


def _client():
    from app import app
    app.config.update(TESTING=True)
    return app.test_client()


def _post_fixture(client, img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("failed to PNG-encode fixture")
    data = {"image": (io.BytesIO(buf.tobytes()), "fixture.png")}
    resp = client.post("/api/cognitive-load", data=data,
                       content_type="multipart/form-data")
    if resp.status_code != 200:
        raise RuntimeError(
            f"endpoint returned {resp.status_code}: {resp.get_data(as_text=True)}")
    return resp.get_json()


def _gaps(by_scale, display_factor):
    """Return raw and displayed-point gap for a metric's per-scale values.

    display_factor scales the raw metric onto the 0..100 displayed range
    (1.0 for a value already in points, 100.0 for an index in [0, 1]).
    """
    vals = [by_scale[str(s)] for s in SCALES]
    raw_gap = float(max(vals) - min(vals))
    denom = max(abs(float(np.mean(vals))), 1e-9)
    return {
        "by_scale": {str(s): float(by_scale[str(s)]) for s in SCALES},
        "raw_abs_gap": raw_gap,
        "raw_rel_gap": raw_gap / denom,
        "displayed_point_gap": raw_gap * display_factor,
    }


def build_matrix() -> dict:
    _install_mocks()
    client = _client()

    fixtures_out = {}
    for name, fx in FIXTURES.items():
        cli_by_scale = {}
        cx_by_scale = {}
        task_vals = None
        profile_vals = None
        provenance_by_scale = {}
        for s in SCALES:
            body = _post_fixture(client, fx(s))
            cli_by_scale[str(s)] = float(body["cognitive_load_index"])
            cx_by_scale[str(s)] = float(
                body["layout"]["experimental_complexity_index"])
            provenance_by_scale[str(s)] = body["hceye_inputs"][
                "analysis_provenance"]
            if task_vals is None:
                td = body["task_descriptor"]
                task_vals = {
                    "task_type": td.get("task_type"),
                    "target_specificity": td.get("target_specificity"),
                    "time_pressure": td.get("time_pressure"),
                    "search_mode": td.get("search_mode"),
                    "modifier": float(td.get("modifier", 0.0)),
                }
                prof = body["big_five_profile"]
                profile_vals = {
                    "preset": prof.get("preset", prof.get("name")),
                    "modifier": float(prof.get("modifier", 0.0)),
                }

        fixtures_out[name] = {
            "cognitive_load_index": _gaps(cli_by_scale, display_factor=100.0),
            "layout_experimental_complexity_index": _gaps(
                cx_by_scale, display_factor=1.0),
            "task_descriptor": task_vals,
            "profile": profile_vals,
            "analysis_provenance_by_scale": provenance_by_scale,
        }

    return {
        "description": (
            "TRUE endpoint scale matrix produced through the real Flask "
            "/api/cognitive-load route. NOT the reduced HCEye-rule "
            "fixture_scale_report.json."),
        "route": "/api/cognitive-load",
        "scales": list(SCALES),
        "mocked_components": MOCKED_COMPONENTS,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "fixtures": fixtures_out,
    }


def _print_matrix(matrix: dict) -> None:
    print("Endpoint scale matrix (real /api/cognitive-load route)")
    print(f"  python={matrix['environment']['python']} "
          f"numpy={matrix['environment']['numpy']} "
          f"opencv={matrix['environment']['opencv']}")
    print(f"  mocked: {[m['component'] for m in matrix['mocked_components']]}")
    for name, d in matrix["fixtures"].items():
        cli = d["cognitive_load_index"]
        cx = d["layout_experimental_complexity_index"]
        print(f"\n=== fixture: {name} ===")
        bs = cli["by_scale"]
        print(f"  cognitive_load_index        "
              f"1x={bs['1']:.5f} 2x={bs['2']:.5f} 3x={bs['3']:.5f} "
              f"raw_gap={cli['raw_abs_gap']:.5f} "
              f"disp_pts={cli['displayed_point_gap']:.4f}")
        bs = cx["by_scale"]
        print(f"  experimental_complexity_idx "
              f"1x={bs['1']:.4f} 2x={bs['2']:.4f} 3x={bs['3']:.4f} "
              f"raw_gap={cx['raw_abs_gap']:.4f} "
              f"disp_pts={cx['displayed_point_gap']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",
                    default=os.path.join(_STAGE1_DIR, "canonical_eval"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    matrix = build_matrix()
    out_path = os.path.join(args.out_dir, "endpoint_scale_report.json")
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2, sort_keys=True)
    _print_matrix(matrix)
    print(f"\nEndpoint matrix written to "
          f"{os.path.relpath(out_path, _REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
