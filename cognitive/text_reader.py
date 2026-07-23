# ===========================================================================
# cognitive/text_reader.py
# ===========================================================================
# Purpose:
#   Optional OCR layer that estimates the READING COST of textual UI elements.
#   Without this, a labelled control such as a "Collision Warning" button is
#   treated by the search model like a plain icon, ignoring the time a user
#   needs to actually read the label.
#
# Method:
#   1. Run EasyOCR once on the whole screenshot -> a list of text boxes.
#   2. Assign each text box to the detected UI element whose bounding box
#      contains the text-box centre.
#   3. Per element: concatenate its text, count words, and estimate a reading
#      time from an empirical silent-reading rate.
#
# Reading-time model:
#   reading_time_s = n_words / (WORDS_PER_MINUTE / 60)
#   WORDS_PER_MINUTE = 238 (silent reading of English prose).
#   Reference: Brysbaert, M. (2019). How many words do we read per minute?
#   A review and meta-analysis of reading rate. Journal of Memory and
#   Language, 109, 104047.
#
#   NOTE (thesis honesty): 238 wpm is a prose-reading average; short glanceable
#   UI labels are read differently (often a single fixation). This is therefore
#   a first-order engineering estimate, not a validated per-label reading model.
#
# This module is OPTIONAL. If EasyOCR (and its torch backend) is not installed,
# every public function degrades gracefully and returns None, so the rest of
# the pipeline keeps working (text_density then uses its neutral fallback).
#
# References:
#   Smith, R. (2007). An overview of the Tesseract OCR engine. ICDAR.
#   Brysbaert, M. (2019). Reading rate meta-analysis. JML, 109.
# ===========================================================================

from typing import Dict, List, Optional

import numpy as np

# Empirical silent-reading rate (Brysbaert, 2019).
WORDS_PER_MINUTE = 238.0

# Minimum OCR confidence to accept a text box (EasyOCR score in [0, 1]).
MIN_OCR_CONFIDENCE = 0.3

# Lazily-initialised, process-wide EasyOCR reader (model load is expensive).
_READER = None
_READER_FAILED = False


def _get_reader():
    """Return a shared EasyOCR reader, or None if OCR is unavailable.

    The reader is created on first use and cached for the process lifetime,
    because loading the detection/recognition models takes several seconds.
    """
    global _READER, _READER_FAILED
    if _READER is not None:
        return _READER
    if _READER_FAILED:
        return None
    try:
        import easyocr

        # English only, CPU. gpu=False keeps it deterministic and dependency-light.
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _READER
    except Exception as exc:
        # Remember the failure so we do not retry the slow import every call.
        _READER_FAILED = True
        print(f"[text_reader] OCR unavailable, reading costs disabled: {exc!r}")
        return None


def _point_in_bbox(px: float, py: float, bbox) -> bool:
    """True if point (px, py) lies inside bbox = (x, y, w, h)."""
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def _reading_time_s(n_words: int) -> float:
    """Estimate silent reading time for n_words at WORDS_PER_MINUTE."""
    if n_words <= 0:
        return 0.0
    return n_words / (WORDS_PER_MINUTE / 60.0)


def compute_readability(image_bgr: np.ndarray, elements: List[Dict]) -> Optional[Dict]:
    """
    Estimate per-element text reading costs for a screenshot.

    Runs OCR once on the full image, maps each recognised text box to the UI
    element that contains it, and augments those elements in place with:
        - ``text``           : the concatenated label text (str)
        - ``n_words``        : word count (int)
        - ``reading_time_s`` : estimated silent reading time (float, seconds)
        - ``is_text``        : whether the element carries any text (bool)

    Parameters
    ----------
    image_bgr : np.ndarray
        The screenshot in BGR (OpenCV) order.
    elements : list of dict
        Detected UI elements (from element_detector.detect_elements). Each must
        have a 'bbox' = (x, y, w, h).

    Returns
    -------
    Optional[Dict]
        A readability report, or None if OCR is unavailable. The report has:
            words_per_minute, n_elements, n_text_elements, total_words,
            total_reading_time_s,
            text_elements: list of the most text-heavy elements (worst first),
                each {id, text, n_words, reading_time_s, bbox, center,
                      color_category}.
    """
    reader = _get_reader()
    if reader is None or image_bgr is None or not elements:
        return None

    try:
        raw = reader.readtext(image_bgr)
    except Exception as exc:
        print(f"[text_reader] OCR read failed: {exc!r}")
        return None

    # Collect accepted (centre, text) pairs above the confidence threshold.
    boxes = []
    for quad, text, conf in raw:
        if conf < MIN_OCR_CONFIDENCE or not str(text).strip():
            continue
        pts = np.asarray(quad, dtype=np.float64)  # 4 corner points
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        boxes.append((cx, cy, str(text).strip()))

    # Initialise text fields on every element.
    for e in elements:
        e["text"] = ""
        e["n_words"] = 0
        e["reading_time_s"] = 0.0
        e["is_text"] = False

    # Assign each text box to the first element that contains its centre.
    for cx, cy, text in boxes:
        for e in elements:
            if _point_in_bbox(cx, cy, e["bbox"]):
                e["text"] = (e["text"] + " " + text).strip()
                break

    # Finalise per-element word counts and reading times.
    for e in elements:
        if e["text"]:
            n_words = len(e["text"].split())
            e["n_words"] = n_words
            e["reading_time_s"] = round(_reading_time_s(n_words), 3)
            e["is_text"] = True

    text_elems = [e for e in elements if e["is_text"]]
    total_words = int(sum(e["n_words"] for e in text_elems))
    total_reading_time = round(sum(e["reading_time_s"] for e in text_elems), 3)

    # Rank the most text-heavy elements (longest reading time first).
    ranked = sorted(text_elems, key=lambda e: e["reading_time_s"], reverse=True)
    text_elements = [{
        "id": e["id"],
        "text": e["text"],
        "n_words": e["n_words"],
        "reading_time_s": e["reading_time_s"],
        "bbox": e["bbox"],
        "center": e["center"],
        "color_category": e.get("color_category", "unknown"),
    } for e in ranked[:5]]

    return {
        "words_per_minute": WORDS_PER_MINUTE,
        "n_elements": len(elements),
        "n_text_elements": len(text_elems),
        "total_words": total_words,
        "total_reading_time_s": total_reading_time,
        "text_elements": text_elements,
    }
