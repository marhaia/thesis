# ===========================================================================
# cognitive/ — Cognitive Predictive Metrics for GUI Evaluation
# ===========================================================================
#
# This package implements computational cognitive models that predict
# user performance metrics (e.g., visual search time) from static GUI
# screenshots. The central model is based on:
#
#   Jokinen, J.P.P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X. (2020).
#   Adaptive feature guidance: Modelling visual search with graphical layouts.
#   International Journal of Human-Computer Studies, 136, 102376.
#
# Architecture:
#   1. Element Detection  — segment GUI into UI elements (bounding boxes)
#   2. Saliency Mapping   — UMSI++ deep saliency per element
#   3. Search Simulation  — Jokinen 2020 visual search model (EMMA + saliency)
#   4. Predicted Metrics  — per-element search time, fixation count
#
# ===========================================================================

from cognitive.jokinen_model import JokinenSearchModel
from cognitive.element_detector import detect_elements

__all__ = ["JokinenSearchModel", "detect_elements"]
