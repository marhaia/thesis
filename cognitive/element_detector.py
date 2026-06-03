# ===========================================================================
# cognitive/element_detector.py
# ===========================================================================
# Purpose:
#   Detect UI elements (bounding boxes) from a GUI screenshot.
#   These elements serve as "visual objects" for the Jokinen 2020 model.
#
# Method:
#   1. Convert to grayscale, apply adaptive threshold
#   2. Edge detection (Canny) + morphological closing to merge nearby edges
#   3. Find contours → bounding boxes
#   4. Filter: remove too-small, too-large, or overlapping boxes
#   5. Extract per-element features: position, size, dominant colour
#
# The output is a list of Element dicts, each with:
#   - id: int
#   - bbox: (x, y, w, h) in pixels
#   - center: (cx, cy) in pixels
#   - area: int (pixels²)
#   - dominant_color_hsv: (H, S, V) — dominant colour cluster
#   - angular_size: float (degrees visual angle, assuming 60cm viewing dist)
#
# Reference:
#   Jokinen et al. (2020), Section 3.1: "We assume the modeller defines the
#   visual representation of the layout as a set of visual elements with
#   their locations, sizes, shapes, and colours."
#
# ===========================================================================

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Assumed viewing parameters (Jokinen 2020 experiment: 60cm, 17" 1680×1050)
VIEWING_DISTANCE_CM = 60.0      # Distance from eyes to screen
SCREEN_WIDTH_CM = 37.0          # Physical width of a typical 17" screen
SCREEN_HEIGHT_CM = 23.0         # Physical height

# Element filtering thresholds
MIN_ELEMENT_AREA_PX = 400       # Ignore elements smaller than 20×20 px
MAX_ELEMENT_AREA_RATIO = 0.5    # Ignore elements > 50% of image area
MIN_ASPECT_RATIO = 0.05         # Ignore very thin slivers
MAX_ASPECT_RATIO = 20.0         # (width/height must be in [0.05, 20])

# Morphological kernel size for merging nearby edges
MORPH_KERNEL_SIZE = 7

# Maximum number of elements (cap for performance)
MAX_ELEMENTS = 80


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def detect_elements(
    image: np.ndarray,
    min_area: int = MIN_ELEMENT_AREA_PX,
    max_elements: int = MAX_ELEMENTS,
    screen_width_cm: float = SCREEN_WIDTH_CM,
    screen_height_cm: float = SCREEN_HEIGHT_CM,
    viewing_distance_cm: float = VIEWING_DISTANCE_CM,
) -> List[Dict]:
    """
    Detect UI elements from a BGR screenshot image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3), as loaded by cv2.imread().
    min_area : int
        Minimum bounding box area in pixels to keep.
    max_elements : int
        Cap on number of detected elements (keep largest).
    screen_width_cm : float
        Physical screen width for visual angle calculation.
    screen_height_cm : float
        Physical screen height for visual angle calculation.
    viewing_distance_cm : float
        Viewing distance in cm for visual angle calculation.

    Returns
    -------
    List[Dict]
        Each dict has keys: id, bbox, center, area, dominant_color_hsv,
        angular_size, color_category.
    """
    h_img, w_img = image.shape[:2]

    # ------------------------------------------------------------------
    # Step 1: Preprocessing — grayscale + edge detection
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to detect regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    # Morphological closing: merge nearby edges into solid regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Additional dilation to fill small gaps
    closed = cv2.dilate(closed, kernel, iterations=1)

    # ------------------------------------------------------------------
    # Step 2: Find contours → bounding boxes
    # ------------------------------------------------------------------
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filter by size
        if area < min_area:
            continue
        if area > MAX_ELEMENT_AREA_RATIO * (h_img * w_img):
            continue

        # Filter by aspect ratio
        aspect = w / max(h, 1)
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue

        raw_boxes.append((x, y, w, h, area))

    # ------------------------------------------------------------------
    # Step 3: Non-maximum suppression (remove highly overlapping boxes)
    # ------------------------------------------------------------------
    raw_boxes.sort(key=lambda b: b[4], reverse=True)  # Sort by area descending
    filtered_boxes = _non_max_suppression(raw_boxes, iou_threshold=0.5)

    # Cap at max_elements (keep largest)
    filtered_boxes = filtered_boxes[:max_elements]

    # ------------------------------------------------------------------
    # Step 4: Extract per-element features
    # ------------------------------------------------------------------
    # Pixels-per-cm conversion
    px_per_cm_x = w_img / screen_width_cm
    px_per_cm_y = h_img / screen_height_cm

    elements = []
    for idx, (x, y, w, h, area) in enumerate(filtered_boxes):
        # Center of element in pixels
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Dominant colour (HSV) via k-means on the element ROI
        roi = image[y:y+h, x:x+w]
        dom_hsv = _dominant_color_hsv(roi)

        # Colour category (Jokinen: "red", "green", "blue" — simplified)
        color_cat = _hsv_to_category(dom_hsv)

        # Angular size in degrees visual angle
        # Use the diagonal as "size" of the element
        diag_px = np.sqrt(w**2 + h**2)
        diag_cm = diag_px / ((px_per_cm_x + px_per_cm_y) / 2.0)
        angular_size_deg = np.degrees(
            2 * np.arctan(diag_cm / (2 * viewing_distance_cm))
        )

        elements.append({
            "id": idx,
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(cx), float(cy)),
            "area": int(area),
            "dominant_color_hsv": (float(dom_hsv[0]), float(dom_hsv[1]), float(dom_hsv[2])),
            "color_category": color_cat,
            "angular_size": float(angular_size_deg),
        })

    return elements


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _non_max_suppression(
    boxes: List[Tuple], iou_threshold: float = 0.5
) -> List[Tuple]:
    """
    Simple greedy NMS on bounding boxes.
    boxes: list of (x, y, w, h, area) tuples, sorted by area descending.
    """
    keep = []
    suppressed = set()

    for i, box_i in enumerate(boxes):
        if i in suppressed:
            continue
        keep.append(box_i)
        xi, yi, wi, hi, _ = box_i

        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            xj, yj, wj, hj, _ = boxes[j]

            # Compute IoU
            inter_x1 = max(xi, xj)
            inter_y1 = max(yi, yj)
            inter_x2 = min(xi + wi, xj + wj)
            inter_y2 = min(yi + hi, yj + hj)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            union = wi * hi + wj * hj - intersection
            iou = intersection / max(union, 1)

            if iou > iou_threshold:
                suppressed.add(j)

    return keep


def _dominant_color_hsv(roi: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute dominant colour of an ROI using k-means (k=3).
    Returns (H, S, V) with H in [0, 360], S in [0, 1], V in [0, 1].
    """
    if roi.size == 0:
        return (0.0, 0.0, 0.5)

    # Downsample for speed
    max_px = 500
    h, w = roi.shape[:2]
    if h * w > max_px:
        scale = np.sqrt(max_px / (h * w))
        roi = cv2.resize(roi, (max(1, int(w * scale)), max(1, int(h * scale))))

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # K-means with k=3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(3, len(pixels))
    if k < 1:
        return (0.0, 0.0, 0.5)

    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )

    # Pick the cluster with most pixels
    counts = np.bincount(labels.flatten(), minlength=k)
    dominant_idx = np.argmax(counts)
    dom = centers[dominant_idx]

    # OpenCV HSV: H in [0,180], S in [0,255], V in [0,255]
    return (float(dom[0] * 2), float(dom[1] / 255.0), float(dom[2] / 255.0))


def _hsv_to_category(hsv: Tuple[float, float, float]) -> str:
    """
    Map HSV to a categorical colour label.
    Following Jokinen's legacy approach (red/green/blue) but extended.

    Jokinen 2020 uses qualitative colour comparison: elements are
    "dissimilar" if their colour category differs.
    """
    h, s, v = hsv

    # Low saturation → achromatic (gray/white/black)
    if s < 0.15:
        if v < 0.3:
            return "black"
        elif v > 0.8:
            return "white"
        else:
            return "gray"

    # Chromatic - classify by hue (H in [0, 360])
    if h < 15 or h >= 345:
        return "red"
    elif 15 <= h < 45:
        return "orange"
    elif 45 <= h < 75:
        return "yellow"
    elif 75 <= h < 165:
        return "green"
    elif 165 <= h < 195:
        return "cyan"
    elif 195 <= h < 270:
        return "blue"
    elif 270 <= h < 345:
        return "purple"

    return "gray"


# ---------------------------------------------------------------------------
# Utility: Convert pixel coordinates to visual angle
# ---------------------------------------------------------------------------

def pixels_to_degrees(
    px_distance: float,
    image_width_px: int,
    screen_width_cm: float = SCREEN_WIDTH_CM,
    viewing_distance_cm: float = VIEWING_DISTANCE_CM,
) -> float:
    """
    Convert a pixel distance to degrees of visual angle.

    Uses the small-angle approximation:
        angle_deg ≈ arctan(distance_cm / viewing_distance_cm) * (180/π)

    Parameters
    ----------
    px_distance : float
        Distance in pixels.
    image_width_px : int
        Width of the image in pixels (for px→cm conversion).
    screen_width_cm : float
        Physical screen width in cm.
    viewing_distance_cm : float
        Viewing distance in cm.

    Returns
    -------
    float
        Visual angle in degrees.
    """
    px_per_cm = image_width_px / screen_width_cm
    distance_cm = px_distance / px_per_cm
    angle_rad = np.arctan(distance_cm / viewing_distance_cm)
    return float(np.degrees(angle_rad))
