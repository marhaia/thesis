"""Canonical-resolution layout measurements (analysis path).

Production helper that computes the score-driving *layout* measurements
(``whitespace_ratio`` and OCR-derived ``text_density``) on a single canonical
analysis image whose long side is ``CANONICAL_LONG_SIDE`` (1280 px). Running
these on a fixed resolution standardises their analysis scale, matching the
eight canonical visual features, so the layout ``experimental_complexity_index``
has substantially reduced measured resolution sensitivity (see the reproducible
canonical evaluation for the residual raster-stress gaps).

This module is the ANALYSIS path. It is deliberately separate from the NATIVE
path used by the Jokinen search model, target selection, scanpaths and native
overlays: those keep native-resolution element detection and native
coordinates. Analysis elements must NEVER be passed to code that expects the
native detector's output.

The helper exposes provenance (native/analysis shapes, canonical scale factors,
canonical elements, whitespace ratio, text density and its source) so tests and
the evaluation script can import and reuse the exact production logic instead of
reimplementing whitespace/OCR.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the Stage-1 feature code and the project root importable regardless of
# how this module is imported (as ``canonical_layout`` with stage1 on the path,
# or as ``stage1.canonical_layout`` with the project root on the path).
_STAGE1_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_STAGE1_DIR)
for _p in (_STAGE1_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visual_complexity import (  # noqa: E402
    canonicalize_for_analysis,
    CANONICAL_LONG_SIDE,
)
from cognitive.element_detector import detect_elements  # noqa: E402

# Identifier for the analysis path, embedded in the provenance so consumers can
# assert that a measurement came from the canonical layout path.
ANALYSIS_PATH = f"canonical-analysis:long{CANONICAL_LONG_SIDE}"


def scale_box_to_native(
    bbox: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    native_w: int,
    native_h: int,
) -> Tuple[int, int, int, int]:
    """Map a canonical-space box to native pixels, clipped to native bounds.

    This is the one reusable coordinate transform for turning canonical
    analysis coordinates into displayable native coordinates. It is used ONLY
    for display (e.g. readability boxes drawn on the uploaded image); mapped
    coordinates must never be fed back into the layout score.

    Args:
        bbox: ``(x, y, w, h)`` in canonical-analysis pixels.
        scale_x: native_w / analysis_w.
        scale_y: native_h / analysis_h.
        native_w: native image width.
        native_h: native image height.

    Returns:
        ``(x, y, w, h)`` in native pixels, guaranteed inside the native image.
    """
    x, y, w, h = bbox
    nx = int(round(x * scale_x))
    ny = int(round(y * scale_y))
    nw = max(1, int(round(w * scale_x)))
    nh = max(1, int(round(h * scale_y)))
    # Clip so a rounded box never leaves the native image.
    nx = min(max(nx, 0), max(native_w - 1, 0))
    ny = min(max(ny, 0), max(native_h - 1, 0))
    nw = min(nw, native_w - nx)
    nh = min(nh, native_h - ny)
    return (nx, ny, max(1, nw), max(1, nh))


@dataclass
class CanonicalLayoutMeasurement:
    """Provenance-carrying result of the canonical layout analysis path.

    All geometric fields (``analysis_elements``) are in CANONICAL coordinates.
    Use :meth:`readability_report_native` to obtain native-coordinate boxes for
    display only.
    """

    native_shape: Tuple[int, int]      # (height, width) of the native image
    analysis_shape: Tuple[int, int]    # (height, width) of the canonical image
    scale_x: float                     # native_w / analysis_w (canonical->native)
    scale_y: float                     # native_h / analysis_h (canonical->native)
    analysis_elements: List[Dict]      # canonical-coordinate element dicts
    whitespace_ratio: float            # from canonical boxes on canonical canvas
    text_density: Optional[float]      # OCR-derived, canonical; None if no OCR
    text_density_source: str           # "ocr" | "fallback_neutral"
    readability_report: Optional[Dict] = None   # canonical-coordinate report
    analysis_path: str = ANALYSIS_PATH

    @property
    def analysis_long_side(self) -> int:
        return int(max(self.analysis_shape))

    @property
    def element_count(self) -> int:
        return len(self.analysis_elements)

    def readability_report_native(self) -> Optional[Dict]:
        """Return the readability report with display boxes mapped to native.

        Only the coordinate fields (``bbox``, ``center``) of each listed text
        element are mapped; all counts / reading times are unchanged. Returns
        ``None`` if no readability report is available.
        """
        if not self.readability_report:
            return None
        nh, nw = self.native_shape
        out = dict(self.readability_report)
        mapped = []
        for te in self.readability_report.get("text_elements", []):
            te2 = dict(te)
            nb = scale_box_to_native(
                tuple(te["bbox"]), self.scale_x, self.scale_y, nw, nh)
            te2["bbox"] = list(nb)
            te2["center"] = [nb[0] + nb[2] / 2.0, nb[1] + nb[3] / 2.0]
            mapped.append(te2)
        out["text_elements"] = mapped
        out["coordinate_space"] = "native"
        return out

    def as_dict(self) -> Dict:
        return {
            "analysis_path": self.analysis_path,
            "native_shape": list(self.native_shape),
            "analysis_shape": list(self.analysis_shape),
            "analysis_long_side": self.analysis_long_side,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "element_count": self.element_count,
            "whitespace_ratio": self.whitespace_ratio,
            "text_density": self.text_density,
            "text_density_source": self.text_density_source,
        }


def _whitespace_from_boxes(elements: List[Dict], h: int, w: int) -> float:
    """1 - (union area of element boxes / image area) on the given canvas.

    A binary union mask is used so overlapping boxes are not double-counted.
    """
    area = float(h * w) or 1.0
    mask = np.zeros((h, w), dtype=np.uint8)
    for e in elements:
        x, y, bw, bh = e["bbox"]
        mask[int(y):int(y + bh), int(x):int(x + bw)] = 1
    return float(np.clip(1.0 - float(mask.sum()) / area, 0.0, 1.0))


def measure_canonical_layout(
    native_bgr: np.ndarray,
    run_ocr: bool = True,
    canonical_long_side: Optional[int] = None,
) -> CanonicalLayoutMeasurement:
    """Compute the canonical (analysis-path) layout measurements.

    Steps:
      1. Build ONE canonical analysis image (long side = 1280 px) from the
         decoded native BGR image.
      2. Detect UI elements on that canonical image (canonical coordinates).
      3. Compute ``whitespace_ratio`` from the union of those canonical boxes on
         the canonical canvas (NOT on native coordinates).
      4. If ``run_ocr``, run readability/OCR on the SAME canonical image and
         canonical element set, and derive ``text_density`` from it.

    The single canonical resize is reused for detection and OCR, so no
    additional canonical resizing happens inside this helper.

    Args:
        native_bgr: decoded native image (H x W x 3, BGR).
        run_ocr: if False, skip OCR (``text_density`` stays None); useful for
            fast/deterministic evaluation where OCR is not the variable under
            test.
        canonical_long_side: override the canonical long side (defaults to
            ``CANONICAL_LONG_SIDE``).

    Returns:
        A :class:`CanonicalLayoutMeasurement` with canonical-coordinate
        elements and provenance.
    """
    long_side = canonical_long_side or CANONICAL_LONG_SIDE
    nh, nw = native_bgr.shape[:2]

    analysis_img = canonicalize_for_analysis(native_bgr, long_side=long_side)
    ah, aw = analysis_img.shape[:2]

    analysis_elements = detect_elements(analysis_img)
    whitespace_ratio = _whitespace_from_boxes(analysis_elements, ah, aw)

    text_density: Optional[float] = None
    text_density_source = "fallback_neutral"
    readability_report: Optional[Dict] = None
    if run_ocr and analysis_elements:
        try:
            from cognitive.text_reader import compute_readability
            readability_report = compute_readability(analysis_img, analysis_elements)
            if readability_report and readability_report.get("n_elements"):
                text_density = float(readability_report["n_text_elements"]) / float(
                    max(readability_report["n_elements"], 1)
                )
                text_density_source = "ocr"
        except Exception as exc:  # OCR is optional; degrade gracefully.
            print(f"[canonical_layout] OCR unavailable (neutral fallback): {exc!r}")

    # Scale factors map canonical coordinates back to native (display only).
    scale_x = nw / float(aw)
    scale_y = nh / float(ah)

    return CanonicalLayoutMeasurement(
        native_shape=(nh, nw),
        analysis_shape=(ah, aw),
        scale_x=scale_x,
        scale_y=scale_y,
        analysis_elements=analysis_elements,
        whitespace_ratio=whitespace_ratio,
        text_density=text_density,
        text_density_source=text_density_source,
        readability_report=readability_report,
    )
