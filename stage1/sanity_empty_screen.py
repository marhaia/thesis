"""Empty-/sparse-screen sanity check for the Stage-1 layout index.

Runs controlled screenshots through the live ``/api/cognitive-load`` route and
prints ``layout.experimental_complexity_index``. The route name is retained for
compatibility; the value is the exploratory, project-specific layout-complexity
index on its public 0–100 scale, not a validated cognitive-load measurement.

This is a strict contract probe: every case must return HTTP 200, JSON, and a
finite numeric layout index. Any endpoint or schema drift stops the run with a
clear error and a non-zero exit status; it is never rendered as ``n/a``.

Run: venv/bin/python stage1/sanity_empty_screen.py
"""
import io
import math
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from app import app  # noqa: E402

_UEYES = os.path.join(
    _ROOT, "ueyes", "dataset_full", "UEyes_dataset", "images"
)


class SanityCheckError(RuntimeError):
    """Raised when a sanity case does not satisfy the public API contract."""


def _synthetic(kind: str) -> io.BytesIO:
    im = np.full((800, 1200, 3), 255, np.uint8)
    if kind == "gray":
        im[:] = 128
    elif kind == "terminal":
        im[:] = 20
        for y in range(40, 520, 26):
            cv2.putText(im, "user@host:~$ ls -la /var/log/system",
                        (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    ok, buf = cv2.imencode(".png", im)
    return io.BytesIO(buf.tobytes())


def _index(client, payload, name):
    response = client.post(
        "/api/cognitive-load",
        data={"image": (payload, name)},
        content_type="multipart/form-data",
    )
    if response.status_code != 200:
        raise SanityCheckError(
            f"{name}: expected HTTP 200, received HTTP {response.status_code}"
        )
    if not response.is_json:
        raise SanityCheckError(f"{name}: expected a JSON response")

    body = response.get_json(silent=True)
    if not isinstance(body, dict):
        raise SanityCheckError(f"{name}: JSON response must be an object")
    layout = body.get("layout")
    if not isinstance(layout, dict):
        raise SanityCheckError(f"{name}: response is missing object 'layout'")
    value = layout.get("experimental_complexity_index")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SanityCheckError(
            f"{name}: layout.experimental_complexity_index must be numeric"
        )
    index = float(value)
    if not math.isfinite(index):
        raise SanityCheckError(
            f"{name}: layout.experimental_complexity_index must be finite"
        )
    return index


def main():
    client = app.test_client()
    cases = [
        ("white  (blank)", _synthetic("white"), "white.png"),
        ("gray   (blank)", _synthetic("gray"), "gray.png"),
        ("terminal(text)", _synthetic("terminal"), "terminal.png"),
    ]
    # Add up to three real UEyes GUIs as content-rich controls.
    if os.path.isdir(_UEYES):
        reals = [f for f in sorted(os.listdir(_UEYES))
                 if not f.startswith(".")][:3]
        for fn in reals:
            with open(os.path.join(_UEYES, fn), "rb") as fh:
                cases.append((f"real   {fn}", io.BytesIO(fh.read()), fn))

    print(f"{'case':22s}  layout_index_0_100")
    print("-" * 44)
    try:
        for label, payload, name in cases:
            index = _index(client, payload, name)
            print(f"{label:22s}  {index:18.3f}")
    except SanityCheckError as exc:
        print(f"SANITY CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
