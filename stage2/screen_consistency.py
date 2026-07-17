"""
Inter-Screen Spatial Consistency
================================
A screen-SET level metric: given an ordered set of screens from ONE product
(e.g. the successive views of an in-vehicle infotainment system), it quantifies
how spatially STABLE the layout is across those screens.

Motivation (automotive relevance)
----------------------------------
A driver uses the same on-board UI every day. What matters for the settled
(expert) state is not how hard a single screen is in isolation, but whether the
system keeps recurring controls (e.g. a "Back" button, a status bar) in the same
place from screen to screen. When a control moves between screens, the returning
user's spatial memory is violated, forcing a fresh visual search each time.

This module measures that internal spatial (in)consistency for a whole product,
not a single view. To our knowledge no screenshot-based tool computes it.

Scientific grounding
--------------------
- Spatial memory in visual search: users learn and reuse the location of
  targets; consistent placement lets them look directly rather than search
  (Chun & Jiang, 1998, "Contextual cueing", Cognitive Psychology 36:28-71).
- Change blindness: layout changes between views are easily missed and costly
  to reconcile (Rensink, O'Regan & Clark, 1997, Psychological Science 8:368-373).
- Consistency principle in UI design: internal consistency reduces learning cost
  (Nielsen, 1994, "Usability Engineering", heuristic "Consistency and standards").

Scope and honest limitations
----------------------------
- This is a GEOMETRIC metric computed from detected element bounding boxes. It
  does NOT identify a control semantically (it cannot know a box "is the Back
  button") - that would need a reliable semantic detector, which is out of scope
  (see the thesis' Item 4 / UIED discussion). It therefore measures overall
  spatial layout stability, and, as a secondary diagnostic, the displacement of
  heuristically matched persistent elements between consecutive screens.
- The score is a first-order, unweighted geometric measure. It is NOT yet
  validated against human data; that requires the user study (thesis Item 1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Occupancy grid resolution (rows, cols). A coarse grid captures "which region
# of the screen is used" while staying robust to small detector jitter.
DEFAULT_GRID: Tuple[int, int] = (10, 10)

# Cost weights for the (secondary) element-matching diagnostic. All three terms
# are normalised to [0, 1] before weighting, so the weights are directly
# comparable. Position dominates because it is what spatial memory encodes.
MATCH_W_POSITION: float = 1.0
MATCH_W_SIZE: float = 0.3
MATCH_W_COLOR: float = 0.3

# A matched pair is only accepted as "the same persistent element" if its total
# normalised cost is below this threshold. Above it, the elements are treated as
# appeared/disappeared rather than moved. Declared as a heuristic (needs no
# calibration against human data, but is a modelling choice).
MATCH_MAX_COST: float = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized_occupancy_grid(
    elements: List[Dict],
    image_shape: Tuple[int, int],
    grid: Tuple[int, int],
) -> np.ndarray:
    """
    Build a normalised occupancy grid for one screen.

    Each grid cell holds the fraction of its area covered by the UNION of the
    element bounding boxes, in [0, 1]. Coordinates are normalised by the image
    size, so screens of different resolutions are directly comparable.

    The union is computed by rasterising the boxes onto a supersampled BINARY
    mask and then averaging per cell. A binary mask is what makes this a true
    union: overlapping boxes cover the same sub-samples only once, so two
    overlapping half-cell boxes can never sum to full occupancy (the previous
    additive-per-box implementation double-counted overlaps before clipping).
    The supersampling factor bounds the raster size independently of the input
    resolution.

    Parameters
    ----------
    elements : list of element dicts with 'bbox' = (x, y, w, h) in pixels.
    image_shape : (H, W) of the screen the elements were detected on.
    grid : (rows, cols) of the occupancy grid.

    Returns
    -------
    np.ndarray of shape (rows, cols) with values in [0, 1].
    """
    rows, cols = grid
    img_h, img_w = float(image_shape[0]), float(image_shape[1])
    occ = np.zeros((rows, cols), dtype=np.float64)
    if img_h <= 0 or img_w <= 0 or rows <= 0 or cols <= 0:
        return occ

    # Supersamples per cell edge: S=20 -> a 200x200 raster for a 10x10 grid,
    # regardless of the screenshot's pixel resolution.
    S = 20
    rr, cc = rows * S, cols * S
    mask = np.zeros((rr, cc), dtype=bool)

    for e in elements:
        x, y, w, h = e["bbox"]
        x0, y0 = float(x), float(y)
        x1, y1 = float(x + w), float(y + h)
        # Clip the box to the image bounds.
        x0, x1 = max(0.0, x0), min(img_w, x1)
        y0, y1 = max(0.0, y0), min(img_h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        # Map the box to raster indices (normalised by the image size). floor on
        # the near edge and ceil on the far edge so a box always marks at least
        # one sub-sample it touches.
        ci0 = max(0, int(np.floor(x0 / img_w * cc)))
        ci1 = min(cc, int(np.ceil(x1 / img_w * cc)))
        ri0 = max(0, int(np.floor(y0 / img_h * rr)))
        ri1 = min(rr, int(np.ceil(y1 / img_h * rr)))
        if ci1 <= ci0 or ri1 <= ri0:
            continue
        mask[ri0:ri1, ci0:ci1] = True

    # Per-cell occupancy = fraction of covered sub-samples in that cell.
    occ = mask.reshape(rows, S, cols, S).mean(axis=(1, 3))
    return np.clip(occ, 0.0, 1.0)


def _match_elements(
    elems_a: List[Dict],
    shape_a: Tuple[int, int],
    elems_b: List[Dict],
    shape_b: Tuple[int, int],
) -> List[float]:
    """
    Greedily match elements between two screens and return the normalised
    centre displacement of confidently matched (persistent) elements.

    Matching cost combines normalised position distance, size difference and
    colour-category mismatch. Greedy nearest-neighbour is used (not the optimal
    assignment) for robustness and to avoid over-fitting the match; pairs above
    MATCH_MAX_COST are treated as appeared/disappeared, not moved.

    Returns
    -------
    list of float : normalised centre displacements (0 = same position, larger =
    moved further) for each accepted persistent match. Empty if none match.
    """
    if not elems_a or not elems_b:
        return []

    ha, wa = float(shape_a[0]), float(shape_a[1])
    hb, wb = float(shape_b[0]), float(shape_b[1])
    if min(ha, wa, hb, wb) <= 0:
        return []

    diag_a = float(np.hypot(wa, ha))
    diag_b = float(np.hypot(wb, hb))

    def feats(e: Dict, w: float, h: float, diag: float) -> Dict:
        cx, cy = e["center"]
        bw, bh = e["bbox"][2], e["bbox"][3]
        return {
            "pos": np.array([cx / w, cy / h], dtype=np.float64),
            "size": (bw * bh) / (w * h),
            "diag_norm": float(np.hypot(bw, bh)) / diag,
            "color": e.get("color_category", "unknown"),
        }

    fa = [feats(e, wa, ha, diag_a) for e in elems_a]
    fb = [feats(e, wb, hb, diag_b) for e in elems_b]

    # Build all candidate pairs with cost, then greedily accept lowest first.
    candidates: List[Tuple[float, int, int, float]] = []
    for i, a in enumerate(fa):
        for j, b in enumerate(fb):
            pos_dist = float(np.linalg.norm(a["pos"] - b["pos"]))  # ~[0, sqrt(2)]
            pos_dist /= np.sqrt(2.0)  # -> [0, 1]
            size_diff = abs(a["size"] - b["size"])  # [0, 1]
            color_diff = 0.0 if a["color"] == b["color"] else 1.0
            cost = (
                MATCH_W_POSITION * pos_dist
                + MATCH_W_SIZE * size_diff
                + MATCH_W_COLOR * color_diff
            ) / (MATCH_W_POSITION + MATCH_W_SIZE + MATCH_W_COLOR)
            candidates.append((cost, i, j, pos_dist))

    candidates.sort(key=lambda t: t[0])
    used_a: set = set()
    used_b: set = set()
    displacements: List[float] = []
    for cost, i, j, pos_dist in candidates:
        if cost > MATCH_MAX_COST:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        displacements.append(pos_dist)

    return displacements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_screen_set_consistency(
    element_sets: Sequence[List[Dict]],
    image_shapes: Sequence[Tuple[int, int]],
    grid: Tuple[int, int] = DEFAULT_GRID,
) -> Dict:
    """
    Compute inter-screen spatial consistency for an ordered set of screens.

    Parameters
    ----------
    element_sets : sequence of element lists, one per screen. Each element is a
        dict as returned by ``cognitive.element_detector.detect_elements``
        (keys 'bbox', 'center', 'color_category', ...).
    image_shapes : sequence of (H, W) tuples, one per screen, matching
        ``element_sets``.
    grid : (rows, cols) occupancy grid resolution.

    Returns
    -------
    Dict with keys:
        'n_screens' : int
        'consistency_score' : float in [0, 1] (1 = recurring controls never
            move; 0 = maximally unstable). A coverage-weighted blend of the
            mean displacement of persistent elements matched between
            consecutive screens (the direct "does the Back button move?"
            signal) and the coarser occupancy measure. When few elements match
            (low match_coverage) the score leans on the occupancy measure so a
            single lucky match cannot claim high consistency on its own; it
            falls back fully to occupancy when no elements can be matched.
        'layout_instability' : float in [0, 1] (= 1 - consistency_score).
        'mean_element_displacement' : float or None — mean normalised centre
            displacement of persistent elements matched between consecutive
            screens (0 = controls never move; larger = controls move more).
            None if fewer than two screens or no confident matches.
        'max_element_displacement' : float or None — the single worst matched
            control movement (highlights the least stable control).
        'occupancy_consistency' : float in [0, 1] — a coarser, matching-free
            supporting measure of how stably screen REGIONS are used across the
            set (from the per-cell occupancy std).
        'match_coverage' : float in [0, 1] — fraction of the (smaller) element
            set that could be confidently matched across consecutive screens.
            Low values mean the displacement signal is based on few elements,
            so the headline leans more on 'occupancy_consistency'.
        'per_cell_instability' : list[list[float]] — the (rows x cols) grid of
            per-cell occupancy std, for visualisation/inspection.
        'interpretation' : str — a short human-readable verdict.

    Notes
    -----
    Requires at least ONE screen. With a single screen the consistency is
    trivially 1.0 (nothing can be inconsistent) and displacement is None.

    The headline score maps mean displacement via ``1 - clip(2 * mean_disp)``:
    a mean control movement of half the screen diagonal is treated as maximal
    inconsistency. The factor 2 is a DECLARED scaling choice, not a calibrated
    constant; the mapping from displacement to a perceived "consistency cost"
    must be validated in the user study (thesis Item 1).
    """
    n = len(element_sets)
    if n != len(image_shapes):
        raise ValueError(
            "element_sets and image_shapes must have the same length "
            f"({n} vs {len(image_shapes)})."
        )
    if n == 0:
        raise ValueError("At least one screen is required.")

    # --- Occupancy grids per screen ---
    grids = np.stack([
        _normalized_occupancy_grid(element_sets[s], image_shapes[s], grid)
        for s in range(n)
    ], axis=0)  # shape (n, rows, cols)

    if n == 1:
        per_cell_std = np.zeros(grid, dtype=np.float64)
        occupancy_consistency = 1.0
    else:
        # Per-cell std across screens. For a value in [0, 1], the maximum
        # possible std is 0.5 (half the screens occupy the cell, half do not),
        # so 2 * mean_std maps the instability into [0, 1].
        per_cell_std = grids.std(axis=0)
        # Average only over cells that are EVER occupied, so large empty margins
        # (std = 0 everywhere) do not dilute the signal toward "perfectly
        # consistent". If nothing is ever occupied, treat as fully consistent.
        ever_occupied = grids.max(axis=0) > 0.0
        if ever_occupied.any():
            mean_std = float(per_cell_std[ever_occupied].mean())
        else:
            mean_std = 0.0
        occupancy_consistency = float(np.clip(1.0 - 2.0 * mean_std, 0.0, 1.0))

    # --- Primary signal: persistent-element displacement ---
    # This is the direct "does a recurring control move between screens?" signal
    # (Gigi's Back-button example). It is the mean over all confident matches
    # across every consecutive screen pair.
    mean_disp: Optional[float] = None
    max_disp: Optional[float] = None
    # Match coverage = fraction of the (smaller) element set that could be
    # confidently matched across consecutive screens. A headline built on only
    # a couple of matched elements must not claim high consistency on its own.
    match_coverage: float = 0.0
    if n >= 2:
        all_disp: List[float] = []
        n_matched = 0
        n_possible = 0
        for s in range(n - 1):
            disp = _match_elements(
                element_sets[s], image_shapes[s],
                element_sets[s + 1], image_shapes[s + 1],
            )
            all_disp.extend(disp)
            n_matched += len(disp)
            n_possible += min(len(element_sets[s]), len(element_sets[s + 1]))
        if all_disp:
            mean_disp = float(np.mean(all_disp))
            max_disp = float(np.max(all_disp))
        if n_possible > 0:
            match_coverage = n_matched / float(n_possible)

    # --- Headline consistency score ---
    # Prefer the direct displacement signal, but weight it by how many elements
    # actually persist (match_coverage). When only a handful of elements match
    # (low coverage), a near-zero mean displacement would otherwise falsely
    # report a highly consistent layout while most elements appeared/moved/
    # disappeared; in that regime we lean on the coarser occupancy measure.
    # See the docstring for the factor 2.
    if mean_disp is not None:
        disp_consistency = float(np.clip(1.0 - 2.0 * mean_disp, 0.0, 1.0))
        consistency = float(
            match_coverage * disp_consistency
            + (1.0 - match_coverage) * occupancy_consistency
        )
    else:
        consistency = occupancy_consistency

    instability = 1.0 - consistency
    if consistency >= 0.85:
        verdict = "highly consistent layout across screens"
    elif consistency >= 0.6:
        verdict = "moderately consistent layout"
    else:
        verdict = "low layout consistency - recurring controls likely move between screens"

    return {
        "n_screens": n,
        "consistency_score": round(consistency, 4),
        "layout_instability": round(instability, 4),
        "mean_element_displacement": (
            round(mean_disp, 4) if mean_disp is not None else None
        ),
        "max_element_displacement": (
            round(max_disp, 4) if max_disp is not None else None
        ),
        "occupancy_consistency": round(occupancy_consistency, 4),
        "match_coverage": round(match_coverage, 4),
        "per_cell_instability": per_cell_std.round(4).tolist(),
        "interpretation": verdict,
    }
