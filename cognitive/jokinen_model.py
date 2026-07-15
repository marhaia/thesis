# ===========================================================================
# cognitive/jokinen_model.py
# ===========================================================================
# Purpose:
#   Python implementation of the Adaptive Feature Guidance model for
#   predicting visual search time on graphical user interfaces.
#
# Reference:
#   Jokinen, J.P.P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X. (2020).
#   Adaptive feature guidance: Modelling visual search with graphical layouts.
#   International Journal of Human-Computer Studies, 136, 102376.
#
# Architecture:
#   The model simulates fixation-by-fixation visual search:
#   1. Start with eyes at a default position (e.g., center of screen)
#   2. Compute activation for each element (bottom-up saliency + noise)
#   3. Attend element with highest activation (winner-take-all)
#   4. Compute encoding time via EMMA model (Salvucci, 2001)
#   5. If saccade needed, add saccade duration
#   6. Mark attended element in VSTM (inhibition of return)
#   7. Repeat until target found
#
# This implementation focuses on NOVICE search (no LTM, no utility learning)
# because a static screenshot evaluation cannot assume prior user experience.
# This gives our model the prediction: "how long would a first-time user
# take to find element X?"
#
# Integration with UMSI++:
#   Instead of computing bottom-up activation purely from feature dissimilarity
#   (which requires the modeller to define discrete features), we ALSO
#   incorporate the UMSI++ saliency map as a continuous bottom-up signal.
#   This is the key contribution: deep-learning saliency replaces/augments
#   hand-crafted feature heuristics from the legacy approach.
#
# Parameters:
#   All parameters from Jokinen 2020 Table 1, with literature-based defaults.
#   Fitted parameters use the values from the paper's grid search.
#
# ===========================================================================

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ===========================================================================
# Model Parameters (from Jokinen 2020, Table 1)
# ===========================================================================

@dataclass
class JokinenParams:
    """
    All parameters of the Adaptive Feature Guidance model.
    Defaults are from Jokinen et al. (2020), Table 1.

    IMPLEMENTATION SCOPE (honest note): the current pipeline implements a
    bottom-up, novice-mode simulation — feature dissimilarity (colour + size),
    a size-based visual acuity gate, UMSI++ saliency, EMMA timing, and
    inhibition of return. Three blocks are declared with their literature
    citations for reference / a possible future extension but are NOT read
    anywhere in the current code and are marked "[NOT USED]" below: the top-down
    guidance weight (W_TA), the ACT-R memory block, and the utility-learning
    block. In particular there is no top-down (goal-directed) guidance yet, so
    the model ranks bottom-up conspicuity rather than simulating "search for X";
    see the module docstring / thesis for the resulting limitation. The colour
    and shape acuity coefficients are also unused — the acuity gate uses only the
    size coefficients (a_size, b_size).

    Literature-based (fixed):
        K           : EMMA encoding constant (Salvucci, 2001)          [USED]
        k           : EMMA encoding exponent                           [USED]
        t_prep      : Saccade preparation time (s)                     [USED]
        t_exec      : Saccade execution time per degree (s/deg)        [USED]
        t_sacc      : Additional saccade constant (s)                  [USED]
        W_BA        : Bottom-up activation weight                      [USED]
        W_TA        : Top-down activation weight                       [NOT USED]
        sigma_TA    : Noise SD for activation (logistic)              [USED]
        a_color     : Visual threshold param a for colour             [NOT USED]
        b_color     : Visual threshold param b for colour             [NOT USED]
        a_shape     : Visual threshold param a for shape              [NOT USED]
        b_shape     : Visual threshold param b for shape              [NOT USED]
        a_size      : Visual threshold param a for size               [USED]
        b_size      : Visual threshold param b for size               [USED]
        freq        : Element frequency (default 0.1)                  [USED]

    Fitted (from grid search in paper):
        B_bl        : Base-level activation constant                  [NOT USED]
        F_mem       : Memory retrieval scaling constant               [NOT USED]
        f_mem       : Memory retrieval exponent                       [NOT USED]
        sigma_M     : Memory noise SD                                 [NOT USED]
        alpha       : Utility learning rate                           [NOT USED]
        sigma_U     : Utility noise SD                                [NOT USED]
        B_sa        : Source activation for LTM                       [NOT USED]
        tau_vstm    : VSTM decay time (number of steps)               [USED]
    """
    # === EMMA Eye Movement Model (Salvucci, 2001) ===
    # Salvucci, D.D. (2001). An integrated model of eye movements and visual
    # encoding. Cognitive Systems Research, 1(4), 201–220.
    # Parameter values from Jokinen et al. (2020), Table 1 (literature-fixed).
    K: float = 0.006            # Encoding time constant (s); EMMA Eq. 4
    k: float = 0.4              # Eccentricity exponent; EMMA Eq. 4
    t_prep: float = 0.135       # Saccade preparation time (s); EMMA Eq. 5
    t_exec: float = 0.002       # Saccade execution per degree (s/deg); EMMA Eq. 5
    t_sacc: float = 0.07        # Saccade constant overhead (s); EMMA Eq. 5
    # Element frequency: proportion of elements of this type on a typical screen.
    # Default 0.1 (≈1 in 10 elements); used in EMMA encoding: T_e = K·[-log(f)]·e^(k·ε)
    freq: float = 0.1

    # === Activation Weights (Jokinen 2020, Table 1 — fitted) ===
    W_BA: float = 1.1           # Bottom-up (saliency) activation weight
    W_TA: float = 0.45          # [NOT USED] Top-down (task relevance) weight —
                                # no top-down guidance is implemented (see class docstring)
    sigma_TA: float = 0.376     # Logistic noise SD on activation (fitted)

    # === Visual Threshold Parameters (Jokinen 2020, Table 1; Eq. 1) ===
    # A feature at eccentricity ε (deg) is visible if its angular size exceeds:
    #   threshold(ε) = a · ε² + b · ε
    # Values from perceptual acuity literature, adapted in Jokinen 2020 Table 1.
    # NOTE: the acuity gate uses ONLY the size coefficients; the colour and
    # shape coefficients below are declared for reference but never read.
    a_color: float = 0.104      # [NOT USED] Quadratic term — colour (deg⁻¹)
    b_color: float = 0.85       # [NOT USED] Linear term  — colour (dimensionless)
    a_shape: float = 0.14       # [NOT USED] Quadratic term — shape
    b_shape: float = 0.96       # [NOT USED] Linear term  — shape
    a_size: float = 0.142       # Quadratic term — size (used by the acuity gate)
    b_size: float = 0.96        # Linear term  — size (used by the acuity gate)

    # Penalty applied to the activation of an element that is below the visual
    # acuity threshold (Eq. 1) at its current eccentricity. It is a finite
    # penalty (not -inf) so a below-threshold element can still eventually be
    # found once the eye moves closer or all other elements are inhibited,
    # which guarantees the search terminates. Chosen large relative to the
    # normalised activations ([0,1]) + logistic noise (sigma_TA=0.376) so a
    # peripherally invisible element is very unlikely to be selected.
    visibility_penalty: float = 5.0

    # === Memory (ACT-R) — [NOT USED] declared for a future expert-mode extension ===
    # Based on Anderson et al. (2004). An integrated theory of the mind.
    # Psychological Review, 111(4), 1036–1060.  Fitted in Jokinen 2020 grid search.
    # None of these are read anywhere in the current novice-mode implementation.
    B_bl: float = 6.0           # [NOT USED] Base-level activation constant (fitted)
    F_mem: float = 0.65         # [NOT USED] Retrieval time scaling factor (fitted)
    f_mem: float = 1.0          # [NOT USED] Retrieval time exponent (fitted)
    sigma_M: float = 0.4        # [NOT USED] Memory retrieval noise SD (fitted)

    # === Utility Learning — [NOT USED] declared for a future expert-mode extension ===
    # Reinforcement-learning component for expert users (Jokinen 2020, §3.3).
    # Not used in novice mode (our pipeline assumes novice / first-time users).
    alpha: float = 0.2          # [NOT USED] Learning rate (fitted)
    sigma_U: float = 0.4        # [NOT USED] Utility noise SD (fitted)
    B_sa: float = 2.0           # [NOT USED] Source activation for LTM spreading (fitted)

    # === Visual Short-Term Memory (Inhibition of Return) ===
    # VSTM capacity ~4 items (Cowan, 2001; Luck & Vogel, 1997).
    # tau_vstm controls how many fixation steps an attended element is suppressed
    # before it can re-enter the search set (inhibition of return).
    # Jokinen 2020 Table 1: tau = 20 fixation steps.
    tau_vstm: int = 20

    # === Saliency Integration (our contribution — not in Jokinen 2020) ===
    # UMSI++ deep-saliency map replaces / augments the hand-crafted bottom-up
    # feature dissimilarity signal.  W_saliency scales its contribution relative
    # to W_BA.  saliency_exponent sharpens the contrast between high- and
    # low-salience regions (power-law normalisation).
    W_saliency: float = 0.8
    saliency_exponent: float = 2.0

    # === Simulation ===
    max_fixations: int = 50     # Hard cap: abort search after this many fixations
    n_simulations: int = 100    # Monte Carlo runs per target (reduces noise)
    random_seed: Optional[int] = 42


# ===========================================================================
# Main Model Class
# ===========================================================================

class JokinenSearchModel:
    """
    Jokinen 2020 Adaptive Feature Guidance — Novice Visual Search.

    Predicts per-element search time for a GUI layout, combining:
      - Feature-based bottom-up saliency (Eq. 2, dissimilarity)
      - Deep saliency from UMSI++ heatmap (our contribution)
      - EMMA eye-movement timing (Eqs. 4-5, Salvucci 2001)
      - Visual short-term memory (inhibition of return)
      - Stochastic noise (logistic distribution)

    Usage:
        model = JokinenSearchModel(params=JokinenParams())
        results = model.predict_search_times(elements, saliency_map, image_shape)
    """

    def __init__(self, params: Optional[JokinenParams] = None):
        """
        Initialize the Jokinen search model.

        Parameters
        ----------
        params : JokinenParams, optional
            Model parameters. Uses literature defaults if not provided.
        """
        self.params = params or JokinenParams()
        self.rng = np.random.default_rng(self.params.random_seed)

    # ===================================================================
    # PUBLIC API
    # ===================================================================

    def predict_search_times(
        self,
        elements: List[Dict],
        saliency_map: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        screen_width_cm: float = 37.0,
        screen_height_cm: float = 23.0,
        viewing_distance_cm: float = 60.0,
    ) -> Dict:
        """
        Predict visual search time for each element in the layout.

        This simulates a novice user searching for each element as a target,
        starting from the center of the screen. The simulation runs
        n_simulations Monte Carlo trials and returns the mean search time.

        Parameters
        ----------
        elements : List[Dict]
            List of detected UI elements, each with keys:
            'id', 'bbox' (x,y,w,h), 'center' (cx,cy), 'area',
            'dominant_color_hsv', 'color_category', 'angular_size'.
        saliency_map : np.ndarray, optional
            UMSI++ saliency heatmap (H, W), values in [0, 1].
            If provided, used as continuous bottom-up signal.
        image_shape : tuple (H, W), optional
            Shape of the original image. Required if saliency_map given.
        screen_width_cm : float
            Physical screen width in cm.
        screen_height_cm : float
            Physical screen height in cm.
        viewing_distance_cm : float
            Viewing distance in cm.

        Returns
        -------
        Dict with keys:
            'per_element': List[Dict] — per-element results:
                'id', 'search_time_s', 'search_time_std_s', 'fixation_count',
                'bbox', 'center'
            'mean_search_time_s': float — layout-wide average
            'max_search_time_s': float — worst-case element
            'min_search_time_s': float — best-case element
            'search_time_std_s': float — standard deviation across elements
            'predicted_difficulty': str — categorical rating
        """
        if len(elements) == 0:
            return {
                "per_element": [],
                "mean_search_time_s": 0.0,
                "max_search_time_s": 0.0,
                "min_search_time_s": 0.0,
                "search_time_std_s": 0.0,
                "predicted_difficulty": "trivial",
            }

        # --- Precompute element saliency from UMSI++ map ---
        element_saliency = self._compute_element_saliency(
            elements, saliency_map, image_shape
        )

        # --- Precompute feature-based bottom-up activations (Eq. 2) ---
        feature_activations = self._compute_feature_activations(elements)

        # --- Combined activation per element ---
        combined_activations = self._combine_activations(
            feature_activations, element_saliency
        )

        # --- Image dimensions for pixel→degree conversion ---
        if image_shape is not None:
            img_h, img_w = image_shape
        elif saliency_map is not None:
            img_h, img_w = saliency_map.shape[:2]
        else:
            # Estimate from element bounding boxes
            max_x = max(e["bbox"][0] + e["bbox"][2] for e in elements)
            max_y = max(e["bbox"][1] + e["bbox"][3] for e in elements)
            img_w, img_h = int(max_x * 1.1), int(max_y * 1.1)

        # --- Pixel-to-degree conversion factor ---
        px_per_cm = img_w / screen_width_cm
        deg_per_px = np.degrees(np.arctan(1.0 / (px_per_cm * viewing_distance_cm)))

        # --- Starting fixation: center of screen ---
        start_fixation = (img_w / 2.0, img_h / 2.0)

        # --- Run Monte Carlo simulation for each target ---
        per_element_results = []

        for target_idx, target_elem in enumerate(elements):
            search_times = []
            fixation_counts = []

            for _ in range(self.params.n_simulations):
                t, n_fix = self._simulate_single_search(
                    target_idx=target_idx,
                    elements=elements,
                    combined_activations=combined_activations,
                    start_fixation=start_fixation,
                    deg_per_px=deg_per_px,
                )
                search_times.append(t)
                fixation_counts.append(n_fix)

            mean_time = float(np.mean(search_times))
            mean_fix = float(np.mean(fixation_counts))
            # Per-target Monte Carlo uncertainty: the standard deviation across
            # the n_simulations trials for THIS target. This quantifies how much
            # the predicted search time varies from run to run (model noise +
            # stochastic saccade selection), and is the genuine uncertainty of
            # the per-element estimate (distinct from "search_time_std_s" below,
            # which is the spread of mean search times ACROSS different targets).
            std_time = float(np.std(search_times))

            per_element_results.append({
                "id": target_elem["id"],
                "search_time_s": round(mean_time, 4),
                "search_time_std_s": round(std_time, 4),
                "fixation_count": round(mean_fix, 2),
                "bbox": target_elem["bbox"],
                "center": target_elem["center"],
                "color_category": target_elem.get("color_category", "unknown"),
            })

        # --- Aggregate statistics ---
        all_times = [r["search_time_s"] for r in per_element_results]
        mean_st = float(np.mean(all_times))
        max_st = float(np.max(all_times))
        min_st = float(np.min(all_times))
        std_st = float(np.std(all_times))

        # --- Difficulty rating (based on mean novice search time) ---
        difficulty = self._rate_difficulty(mean_st)

        return {
            "per_element": per_element_results,
            "mean_search_time_s": round(mean_st, 4),
            "max_search_time_s": round(max_st, 4),
            "min_search_time_s": round(min_st, 4),
            "search_time_std_s": round(std_st, 4),
            "predicted_difficulty": difficulty,
            "n_elements": len(elements),
            "n_simulations": self.params.n_simulations,
        }

    def predict_scanpath_to_target(
        self,
        target_idx: int,
        elements: List[Dict],
        saliency_map: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        screen_width_cm: float = 37.0,
        screen_height_cm: float = 23.0,
        viewing_distance_cm: float = 60.0,
    ) -> Dict:
        """
        Predict a single, representative visual-search scanpath toward a target.

        Unlike predict_search_times (which only keeps aggregate timing), this
        returns the actual fixation sequence the Adaptive Feature Guidance model
        traverses while a novice searches for elements[target_idx], starting from
        the screen center. This is a TASK-DRIVEN scanpath (goal-directed search to
        a chosen target), NOT a free-viewing scanpath.

        Because the simulation is stochastic (logistic activation noise), we run
        several Monte Carlo trials and return the one whose fixation count is the
        median, so the displayed path is typical rather than an outlier.

        Parameters
        ----------
        target_idx : int
            Index into `elements` of the element the user selected as the target.
        (other parameters as in predict_search_times)

        Returns
        -------
        Dict with keys:
            'target_id'        : id of the target element
            'target_index'     : target_idx
            'fixations'        : List[Dict] — the scanpath, each fixation having
                                 'x', 'y', 'order', 't_cumulative_s',
                                 'step_time_s', 'element_id', 'is_target'
            'total_time_s'     : total predicted search time for this path
            'n_fixations'      : number of fixations (excluding the start point)
            'image_height'     : height in px used for coordinates
            'image_width'      : width in px used for coordinates
            'basis'            : 'jokinen_search_model'
            'is_target_driven' : True (goal-directed search, not free-viewing)
        """
        if len(elements) == 0 or target_idx < 0 or target_idx >= len(elements):
            return {}

        # Reuse the exact activation pipeline of predict_search_times so the
        # scanpath is consistent with the reported search times.
        element_saliency = self._compute_element_saliency(
            elements, saliency_map, image_shape
        )
        feature_activations = self._compute_feature_activations(elements)
        combined_activations = self._combine_activations(
            feature_activations, element_saliency
        )

        if image_shape is not None:
            img_h, img_w = image_shape
        elif saliency_map is not None:
            img_h, img_w = saliency_map.shape[:2]
        else:
            max_x = max(e["bbox"][0] + e["bbox"][2] for e in elements)
            max_y = max(e["bbox"][1] + e["bbox"][3] for e in elements)
            img_w, img_h = int(max_x * 1.1), int(max_y * 1.1)

        px_per_cm = img_w / screen_width_cm
        deg_per_px = np.degrees(np.arctan(1.0 / (px_per_cm * viewing_distance_cm)))
        start_fixation = (img_w / 2.0, img_h / 2.0)

        # Run several trials and keep the median-length one as representative.
        trials = []
        for _ in range(self.params.n_simulations):
            t, n_fix, path = self._simulate_single_search(
                target_idx=target_idx,
                elements=elements,
                combined_activations=combined_activations,
                start_fixation=start_fixation,
                deg_per_px=deg_per_px,
                return_path=True,
            )
            trials.append((n_fix, t, path))

        # Median by fixation count (ties broken by closeness to median time).
        trials.sort(key=lambda tr: tr[0])
        median_trial = trials[len(trials) // 2]
        rep_n_fix, rep_time, rep_path = median_trial

        target_elem = elements[target_idx]
        return {
            "target_id": target_elem.get("id"),
            "target_index": target_idx,
            "fixations": rep_path,
            "total_time_s": round(float(rep_time), 4),
            "n_fixations": int(rep_n_fix),
            "image_height": int(img_h),
            "image_width": int(img_w),
            "basis": "jokinen_search_model",
            "is_target_driven": True,
        }

    # ===================================================================
    # SIMULATION CORE
    # ===================================================================

    def _simulate_single_search(
        self,
        target_idx: int,
        elements: List[Dict],
        combined_activations: np.ndarray,
        start_fixation: Tuple[float, float],
        deg_per_px: float,
        return_path: bool = False,
    ):
        """
        Simulate a single visual search trial.

        The model starts at start_fixation and searches for elements[target_idx].
        At each step:
            1. Compute total activation = base_activation + noise - VSTM inhibition
            2. Select element with highest total activation (winner-take-all)
            3. If saccade needed (eccentricity > 2°): execute saccade, then encode
               at residual eccentricity ≈ 0.5° (saccade landing noise)
            4. If no saccade needed: encode at current eccentricity
            5. If selected element == target → done
            6. Else → add to VSTM, continue

        IMPORTANT: The EMMA model (Salvucci, 2001) dictates that after a saccade,
        the eye lands on/near the target. Encoding then occurs at the POST-saccade
        eccentricity (near 0), NOT the pre-saccade distance. This is why search
        times are typically 0.2-0.5s per fixation, not 10+ seconds.

        Parameters
        ----------
        return_path : bool
            When True, also return the fixation sequence (the scanpath) the
            model traversed while searching for the target. Each fixation is a
            dict with pixel coordinates and cumulative timing. Defaults to False
            so existing callers keep the (time, n_fixations) two-tuple contract.

        Returns
        -------
        (total_time_s, n_fixations)                         if return_path is False
        (total_time_s, n_fixations, path: List[Dict])       if return_path is True
            where each path item is:
                {'x', 'y', 'order', 't_cumulative_s', 'step_time_s',
                 'element_id', 'is_target'}
        """
        p = self.params
        n_elements = len(elements)
        current_fix = np.array(start_fixation, dtype=np.float64)
        vstm = set()  # Indices of recently visited elements
        vstm_order = []  # FIFO queue for VSTM decay

        # Precompute element geometry once (does not change during the search).
        # Centers as an array (for vectorised eccentricity), and each element's
        # angular size (deg) from its bbox diagonal using the SAME deg_per_px as
        # the eccentricity, so the visibility gate respects the display preset
        # (a small in-vehicle display makes elements angularly smaller).
        centers_arr = np.array(
            [e["center"] for e in elements], dtype=np.float64
        )
        elem_diag_px = np.array(
            [np.hypot(e["bbox"][2], e["bbox"][3]) for e in elements],
            dtype=np.float64,
        )
        elem_angular_size = elem_diag_px * deg_per_px

        total_time = 0.0
        n_fixations = 0

        # Record the scanpath (only populated when return_path is True). The
        # first fixation is the starting gaze position (screen center), which
        # carries no encoding time.
        path: List[Dict] = []
        if return_path:
            path.append({
                "x": float(start_fixation[0]),
                "y": float(start_fixation[1]),
                "order": 0,
                "t_cumulative_s": 0.0,
                "step_time_s": 0.0,
                "element_id": None,
                "is_target": False,
            })

        for step in range(p.max_fixations):
            # --- Compute activation with noise ---
            activations = combined_activations.copy()

            # Add logistic noise (Jokinen 2020: σ_TA = 0.376)
            noise = self.rng.logistic(0, p.sigma_TA, size=n_elements)
            activations += noise

            # --- Visual acuity gate (Jokinen 2020, Eq. 1) ---
            # An element at eccentricity ε (deg) from the current fixation is
            # only peripherally visible if its angular size exceeds the acuity
            # threshold a·ε² + b·ε (using the size-feature coefficients). Elements
            # below this limit are hard to detect from where the eye currently
            # is, so their activation is penalised; they become selectable again
            # once a later fixation lands closer to them (smaller ε).
            ecc_deg = np.linalg.norm(centers_arr - current_fix, axis=1) * deg_per_px
            acuity_threshold = p.a_size * ecc_deg ** 2 + p.b_size * ecc_deg
            below_threshold = elem_angular_size < acuity_threshold
            activations[below_threshold] -= p.visibility_penalty

            # VSTM inhibition: set visited elements to -∞
            for visited_idx in vstm:
                activations[visited_idx] = -np.inf

            # --- Winner-take-all: select highest activation ---
            selected_idx = int(np.argmax(activations))

            # --- Compute timing (EMMA model) ---
            selected_elem = elements[selected_idx]
            sel_center = np.array(selected_elem["center"], dtype=np.float64)

            # Pre-saccade eccentricity: distance from current fixation (degrees)
            px_dist = np.linalg.norm(sel_center - current_fix)
            eccentricity_deg = px_dist * deg_per_px

            # --- EMMA timing logic (Salvucci, 2001) ---
            # If target is within fovea (< 2°): encode without saccade
            # If target is beyond fovea: saccade first, then encode at ~0°
            saccade_time = 0.0
            if eccentricity_deg > 2.0:
                # Saccade needed (Eq. 5): T_s = t_prep + t_exec * D + t_sacc
                saccade_time = p.t_prep + p.t_exec * eccentricity_deg + p.t_sacc

                # After saccade, eye lands near target with small error
                # Residual eccentricity ~ 0.5° (saccade landing noise)
                residual_ecc = abs(self.rng.normal(0.0, 0.5))
                encoding_ecc = residual_ecc
            else:
                # No saccade: encode at current eccentricity
                encoding_ecc = eccentricity_deg

            # EMMA encoding time (Eq. 4): T_e = K * [-log(f)] * e^(k*ε)
            encoding_time = p.K * (-np.log(p.freq)) * np.exp(p.k * encoding_ecc)

            # Total step time = saccade + encoding
            step_time = encoding_time + saccade_time
            total_time += step_time
            n_fixations += 1

            # --- Update fixation position (eye now on selected element) ---
            current_fix = sel_center.copy()

            is_target = (selected_idx == target_idx)
            if return_path:
                path.append({
                    "x": float(sel_center[0]),
                    "y": float(sel_center[1]),
                    "order": n_fixations,
                    "t_cumulative_s": float(total_time),
                    "step_time_s": float(step_time),
                    "element_id": selected_elem.get("id"),
                    "is_target": is_target,
                })

            # --- Check if target found ---
            if is_target:
                if return_path:
                    return (total_time, n_fixations, path)
                return (total_time, n_fixations)

            # --- Update VSTM (inhibition of return) ---
            vstm.add(selected_idx)
            vstm_order.append(selected_idx)

            # VSTM decay: remove oldest if exceeds capacity
            if len(vstm_order) > p.tau_vstm:
                oldest = vstm_order.pop(0)
                vstm.discard(oldest)

        # If we reach max_fixations without finding target
        if return_path:
            return (total_time, n_fixations, path)
        return (total_time, n_fixations)

    # ===================================================================
    # ACTIVATION COMPUTATION
    # ===================================================================

    def _compute_element_saliency(
        self,
        elements: List[Dict],
        saliency_map: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Extract per-element saliency from the UMSI++ heatmap.

        For each element, compute the mean saliency within its bounding box.
        This replaces the purely feature-based bottom-up activation with
        a data-driven, deep-learning-based saliency signal.

        Returns
        -------
        np.ndarray of shape (n_elements,), values in [0, 1].
        """
        n = len(elements)
        saliency_values = np.zeros(n, dtype=np.float64)

        if saliency_map is None:
            # No saliency map available → return zeros (feature-only mode)
            return saliency_values

        sal_h, sal_w = saliency_map.shape[:2]

        # If image_shape differs from saliency_map shape, we need to scale
        if image_shape is not None:
            img_h, img_w = image_shape
            scale_x = sal_w / img_w
            scale_y = sal_h / img_h
        else:
            scale_x, scale_y = 1.0, 1.0

        for i, elem in enumerate(elements):
            x, y, w, h = elem["bbox"]

            # Scale bbox to saliency map coordinates
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            sw = max(1, int(w * scale_x))
            sh = max(1, int(h * scale_y))

            # Clip to map bounds
            sx = max(0, min(sx, sal_w - 1))
            sy = max(0, min(sy, sal_h - 1))
            ex = min(sx + sw, sal_w)
            ey = min(sy + sh, sal_h)

            # Mean saliency in the element's region
            region = saliency_map[sy:ey, sx:ex]
            if region.size > 0:
                saliency_values[i] = float(np.mean(region))

        # Normalize to [0, 1]
        max_sal = saliency_values.max()
        if max_sal > 0:
            saliency_values /= max_sal

        return saliency_values

    def _compute_feature_activations(
        self, elements: List[Dict]
    ) -> np.ndarray:
        """
        Compute feature-based bottom-up activation (Eq. 2 from paper).

        BA_i = Σ_j Σ_k dissim(v_ik, v_jk) / d_ij

        The paper sums the dissimilarity over several feature dimensions k.
        We use two dimensions here:
          - colour   : categorical, dissim = 1 if the colour categories differ,
                       else 0 (as in the legacy AIM approach).
          - size     : continuous, dissim = |A_i - A_j| / (A_i + A_j) using the
                       element areas. This is 0 for equally sized elements and
                       approaches 1 when one element is far larger than the other.
                       A large, unique element "pops out" and is easier to find,
                       which raises its bottom-up activation.

        The per-pair dissimilarity is the (equally weighted) sum of the colour
        and size terms, consistent with the sum over feature dimensions k.

        Returns
        -------
        np.ndarray of shape (n_elements,), unnormalized activation values.
        """
        n = len(elements)
        if n <= 1:
            return np.ones(n, dtype=np.float64)

        activations = np.zeros(n, dtype=np.float64)

        # Extract centers, colours and areas
        centers = np.array([e["center"] for e in elements], dtype=np.float64)
        colors = [e.get("color_category", "gray") for e in elements]
        # Area falls back to the bbox area when the detector did not provide one.
        areas = np.array([
            float(e.get("area") or (e["bbox"][2] * e["bbox"][3]))
            for e in elements
        ], dtype=np.float64)

        for i in range(n):
            ba_i = 0.0
            for j in range(n):
                if i == j:
                    continue

                # Distance between elements (in pixels)
                d_ij = np.linalg.norm(centers[i] - centers[j])
                d_ij = max(d_ij, 1.0)  # Avoid division by zero

                # Feature dissimilarity: colour (categorical, 0 or 1)
                dissim_color = 0.0 if colors[i] == colors[j] else 1.0

                # Feature dissimilarity: size (continuous, in [0, 1]).
                # Symmetric relative area difference; 0 when equal, ->1 when very
                # different. Guard against two zero-area elements.
                area_sum = areas[i] + areas[j]
                dissim_size = (
                    abs(areas[i] - areas[j]) / area_sum if area_sum > 0 else 0.0
                )

                # Accumulate (Eq. 2): sum of feature dissimilarities / sqrt(dist)
                ba_i += (dissim_color + dissim_size) / np.sqrt(d_ij)

            activations[i] = ba_i

        # Normalize to [0, 1]
        max_act = activations.max()
        if max_act > 0:
            activations /= max_act

        return activations

    def _combine_activations(
        self,
        feature_activations: np.ndarray,
        saliency_values: np.ndarray,
    ) -> np.ndarray:
        """
        Combine feature-based and saliency-based activations.

        Combined = W_BA * feature_activation + W_saliency * saliency^exponent

        The saliency exponent sharpens the distribution, making high-saliency
        elements stand out more (analogous to how bottom-up pop-out works).

        Returns
        -------
        np.ndarray of shape (n_elements,), combined activation values.
        """
        p = self.params

        # Apply exponent to sharpen saliency differences
        sal_sharpened = np.power(saliency_values, p.saliency_exponent)

        # Weighted combination
        combined = (p.W_BA * feature_activations) + (p.W_saliency * sal_sharpened)

        return combined

    # ===================================================================
    # UTILITY METHODS
    # ===================================================================

    def _rate_difficulty(self, mean_search_time_s: float) -> str:
        """
        Categorize layout difficulty based on mean novice search time.

        Based on the empirical data in Jokinen 2020:
        - BP layout (simple): mean ~1.5s in novice phase
        - WIN10 (medium): mean ~2.5s in novice phase
        - NYT (complex): mean ~5-8s in novice phase

        Categories:
            easy      : < 1.0s (simple layout, few salient elements)
            moderate  : 1.0-2.5s (typical desktop UI)
            difficult : 2.5-5.0s (complex web page)
            very_hard : > 5.0s (information-dense, like NYT front page)
        """
        if mean_search_time_s < 1.0:
            return "easy"
        elif mean_search_time_s < 2.5:
            return "moderate"
        elif mean_search_time_s < 5.0:
            return "difficult"
        else:
            return "very_hard"

    def get_model_info(self) -> Dict:
        """Return model metadata for documentation/API responses."""
        return {
            "model": "Jokinen 2020 Adaptive Feature Guidance",
            "mode": "novice_search",
            "reference": "Jokinen et al. (2020). IJHCS, 136, 102376.",
            "parameters": {
                "K": self.params.K,
                "k": self.params.k,
                "t_prep": self.params.t_prep,
                "W_BA": self.params.W_BA,
                "W_saliency": self.params.W_saliency,
                "sigma_TA": self.params.sigma_TA,
                "max_fixations": self.params.max_fixations,
                "n_simulations": self.params.n_simulations,
                "tau_vstm": self.params.tau_vstm,
            },
            "contribution": (
                "Replaces legacy binary-based search simulation with "
                "Python-native implementation. Integrates UMSI++ deep "
                "saliency as continuous bottom-up signal instead of "
                "hand-crafted colour/size heuristics."
            ),
        }


# ===========================================================================
# Convenience function for quick prediction
# ===========================================================================

def predict_search_time(
    elements: List[Dict],
    saliency_map: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    n_simulations: int = 100,
    random_seed: int = 42,
) -> Dict:
    """
    Quick convenience function to predict search times.

    Parameters
    ----------
    elements : List[Dict]
        Detected UI elements (from element_detector.detect_elements).
    saliency_map : np.ndarray, optional
        UMSI++ saliency heatmap (H, W), values in [0, 1].
    image_shape : tuple (H, W), optional
        Original image dimensions.
    n_simulations : int
        Number of Monte Carlo trials per element.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict with search time predictions per element and aggregate statistics.
    """
    params = JokinenParams(n_simulations=n_simulations, random_seed=random_seed)
    model = JokinenSearchModel(params=params)
    return model.predict_search_times(
        elements=elements,
        saliency_map=saliency_map,
        image_shape=image_shape,
    )


# ===========================================================================
# Glance-based metrics (automotive: NHTSA 2013 / ISO 15008)
# ===========================================================================

def compute_glance_metrics(
    fixations: List[Dict],
    glance_budget_s: float = 1.5,
    single_glance_limit_s: float = 2.0,
    cumulative_limit_s: float = 12.0,
) -> Dict:
    """
    Break a predicted search scanpath into in-vehicle "glances" and check it
    against automotive eyes-off-road guidelines.

    Rationale
    ---------
    When a GUI is an in-vehicle display, a driver cannot keep looking at it: they
    time-share attention, glancing at the display for a short period and then
    returning their eyes to the road. We approximate this by packing the model's
    predicted fixations into glances of at most ``glance_budget_s`` seconds each
    (a driver ends a glance and looks back once the budget is used up). The
    resulting glance profile is then compared to the accepted limits:

      - single glance <= 2.0 s   (NHTSA Visual-Manual Guidelines, 2013;
                                   also AAM/ISO 15007 practice)
      - cumulative eyes-off-road time <= 12.0 s for the whole task (NHTSA, 2013)

    These are DESIGN guidelines applied to a MODEL estimate of goal-directed
    search, not measured eye-tracking; they flag layouts whose predicted search
    would push a driver over the safe glance budget.

    Parameters
    ----------
    fixations : list of dict
        Scanpath fixations, each with a numeric 'step_time_s' (as produced by
        predict_scanpath_to_target). The leading start point (order 0,
        step_time_s = 0) is ignored.
    glance_budget_s : float
        Maximum duration of a single modelled glance before the driver looks
        back to the road (default 1.5 s, a common design target below the 2 s
        limit).
    single_glance_limit_s : float
        Regulatory single-glance limit (default 2.0 s, NHTSA 2013).
    cumulative_limit_s : float
        Regulatory cumulative eyes-off-road limit for the task (default 12.0 s).

    Returns
    -------
    Dict with keys:
        'glances'                 : list of per-glance durations (s)
        'n_glances'               : number of glances
        'max_single_glance_s'     : longest single glance (s)
        'total_eyes_off_road_s'   : sum of all glance durations (s)
        'mean_glance_s'           : mean glance duration (s)
        'glance_budget_s'         : the budget used for chunking
        'single_glance_limit_s'   : the applied single-glance limit
        'cumulative_limit_s'      : the applied cumulative limit
        'exceeds_single_glance'   : bool, any glance > single_glance_limit_s
        'exceeds_cumulative'      : bool, total > cumulative_limit_s
        'compliant'               : bool, neither limit exceeded
    """
    # Keep only real fixations with a positive encoding/saccade time.
    steps = [
        float(f.get("step_time_s", 0.0))
        for f in fixations
        if float(f.get("step_time_s", 0.0)) > 0.0
    ]

    if not steps:
        return {
            "glances": [],
            "n_glances": 0,
            "max_single_glance_s": 0.0,
            "total_eyes_off_road_s": 0.0,
            "mean_glance_s": 0.0,
            "glance_budget_s": round(glance_budget_s, 3),
            "single_glance_limit_s": round(single_glance_limit_s, 3),
            "cumulative_limit_s": round(cumulative_limit_s, 3),
            "exceeds_single_glance": False,
            "exceeds_cumulative": False,
            "compliant": True,
        }

    # Greedy packing: fill the current glance until adding the next fixation
    # would exceed the budget, then start a new glance (eyes back to road).
    # A single fixation longer than the budget forms its own (over-budget) glance.
    glances: List[float] = []
    current = 0.0
    for step in steps:
        if current > 0.0 and current + step > glance_budget_s:
            glances.append(current)
            current = step
        else:
            current += step
    if current > 0.0:
        glances.append(current)

    total = float(sum(glances))
    max_single = float(max(glances))
    exceeds_single = max_single > single_glance_limit_s
    exceeds_cumulative = total > cumulative_limit_s

    return {
        "glances": [round(g, 3) for g in glances],
        "n_glances": len(glances),
        "max_single_glance_s": round(max_single, 3),
        "total_eyes_off_road_s": round(total, 3),
        "mean_glance_s": round(total / len(glances), 3),
        "glance_budget_s": round(glance_budget_s, 3),
        "single_glance_limit_s": round(single_glance_limit_s, 3),
        "cumulative_limit_s": round(cumulative_limit_s, 3),
        "exceeds_single_glance": bool(exceeds_single),
        "exceeds_cumulative": bool(exceeds_cumulative),
        "compliant": bool(not exceeds_single and not exceeds_cumulative),
    }
