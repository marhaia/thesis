"""Optional Big Five profile presets for the pipeline concept.

The dropdown uses coarse presets rather than questionnaire-grade scores.
This keeps the feature optional and avoids overclaiming psychometric validity.

Scientific basis:
  - Big Five (OCEAN) personality dimensions: Costa & McCrae (1992).
    NEO PI-R manual. Psychological Assessment Resources.
  - Neuroticism → cognitive load sensitivity: Serapio-García et al. (2023).
    Personality traits in large language models. arXiv:2307.00184.
    Also: Matthews et al. (2000). Cognitive science of attention and stress.
  - Conscientiousness → structured, focused processing under load:
    Taubman-Ben-Ari et al. (2004). The multidimensional driving style inventory.
    Accident Analysis & Prevention, 36(3), 323–332.  [conscientiousness as
    protective factor against cognitive overload in complex tasks]
  - Openness → exploratory scanning, tolerance for interface complexity:
    Chamorro-Premuzic & Furnham (2003). Personality predicts academic
    performance: Evidence from two longitudinal studies. J. Research in
    Personality, 37(4), 319–338.
  - Resilience = 1 − N (neuroticism complement): Fredrickson et al. (2003).
    What good are positive emotions in crises? Psychol. Science, 14(3), 237–245.

Preset vectors [O, C, A, E, N] are normalised to [0,1] and represent
stereotypical trait profiles — not empirically sampled individuals.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# Preset vectors: [Openness, Conscientiousness, Agreeableness, Extraversion, Neuroticism]
# Normalised to [0, 1].  Trait directions follow Costa & McCrae (1992).
PROFILE_PRESETS = {
    "neutral": {
        "label": "Neutral",
        "vector": [0.50, 0.50, 0.50, 0.50, 0.50],  # mid-point baseline; no CL adjustment
    },
    "focused": {
        "label": "Focused / Conscientious",
        # High C (0.82): structured, planful processing reduces redundant fixations
        # Low N (0.25): low anxiety → lower susceptibility to stress-induced CL
        # (Taubman-Ben-Ari et al., 2004; Matthews et al., 2000)
        "vector": [0.40, 0.82, 0.35, 0.55, 0.25],
    },
    "exploratory": {
        "label": "Exploratory / Open",
        # High O (0.72): broader attentional scope, comfortable with interface novelty
        # Moderate N (0.32): resilient baseline (Chamorro-Premuzic & Furnham, 2003)
        "vector": [0.72, 0.48, 0.82, 0.58, 0.32],
    },
    "social": {
        "label": "Social / Extraverted",
        # High E (0.84): socially oriented; moderate CL sensitivity
        # Low-moderate N (0.34): mild stress resilience
        "vector": [0.68, 0.52, 0.60, 0.84, 0.34],
    },
    "stress_sensitive": {
        "label": "Stress-Sensitive",
        # High N (0.82): neuroticism is the strongest personality predictor of
        # increased perceived cognitive load under demanding conditions
        # (Serapio-García et al., 2023; Matthews et al., 2000)
        # Low C (0.46): less structured processing amplifies load further
        "vector": [0.35, 0.46, 0.42, 0.40, 0.82],
    },
}


def get_profile(profile_key: str = "neutral") -> Dict[str, object]:
    profile = PROFILE_PRESETS.get(profile_key, PROFILE_PRESETS["neutral"])
    vec = np.array(profile["vector"], dtype=np.float32)

    openness, conscientiousness, agreeableness, extraversion, neuroticism = vec

    # Resilience = complement of neuroticism (Fredrickson et al., 2003)
    resilience = 1.0 - neuroticism
    structure_preference = conscientiousness   # Costa & McCrae (1992)
    exploration_preference = openness          # Costa & McCrae (1992)

    # Cognitive-load modifier formula:
    #
    #   modifier = 4·N  − 2.5·C  − 1.5·(1−N)  + 1·O  − 1
    #            = 5.5·N − 2.5·C + O − 2.5
    #
    #   Rationale:
    #     +4·N   : Neuroticism is the dominant predictor of CL sensitivity;
    #              high-N users perceive significantly higher load on complex UIs
    #              (Serapio-García et al., 2023; Matthews et al., 2000).
    #     −2.5·C : Conscientiousness exerts a protective effect — structured,
    #              goal-directed processing reduces redundant cognitive work
    #              (Taubman-Ben-Ari et al., 2004).
    #     −1.5·R : Resilience (1−N) further dampens load amplification;
    #              emotionally stable users recover faster from demanding UIs
    #              (Fredrickson et al., 2003).
    #     +1·O   : High Openness correlates with exploratory scanning, which
    #              can slightly increase time-on-task but rarely causes overload
    #              (Chamorro-Premuzic & Furnham, 2003).
    #     −1.0   : Intercept to centre neutral profile at modifier ≈ 0.
    #
    #   Clipped to [−5, +6] to prevent outlier profiles from dominating.
    modifier = (
        4.0 * neuroticism -
        2.5 * conscientiousness -
        1.5 * resilience +
        1.0 * exploration_preference -
        1.0
    )
    modifier = float(np.clip(modifier, -5.0, 6.0))

    return {
        "key": profile_key if profile_key in PROFILE_PRESETS else "neutral",
        "label": profile["label"],
        "traits": {
            "openness": float(openness),
            "conscientiousness": float(conscientiousness),
            "agreeableness": float(agreeableness),
            "extraversion": float(extraversion),
            "neuroticism": float(neuroticism),
        },
        "vector": vec.tolist(),
        "modifier": modifier,
        "derived": {
            "resilience": float(resilience),
            "structure_preference": float(structure_preference),
            "exploration_preference": float(exploration_preference),
        },
    }


def feature_names() -> List[str]:
    return [
        "openness",
        "conscientiousness",
        "agreeableness",
        "extraversion",
        "neuroticism",
    ]