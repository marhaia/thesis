"""
Empty-/sparse-screen sanity check for the HCEye cognitive-load headline.

Runs a set of controlled screenshots through the live /api/cognitive-load path
and prints the resulting cognitive_load_index (headline / 100). Used to verify
the supervisor's sanity test: a blank white/gray canvas and an empty terminal
must NOT read as medium/high load, while genuinely content-rich GUIs still do.

Run: venv/bin/python stage1/sanity_empty_screen.py
"""
import io
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
    r = client.post(
        "/api/cognitive-load",
        data={"image": (payload, name)},
        content_type="multipart/form-data",
    )
    b = r.get_json() or {}
    return b.get("cognitive_load_index")


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

    print(f"{'case':22s}  CL_index  ~score")
    print("-" * 42)
    for label, payload, name in cases:
        idx = _index(client, payload, name)
        score = f"{idx * 100:5.1f}" if idx is not None else "  n/a"
        print(f"{label:22s}  {idx!r:>8}  {score}")


if __name__ == "__main__":
    main()
