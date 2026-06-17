# EyeFormer: Predicting Personalized Scanpaths with Transformer-Guided Reinforcement Learning (2024)

**Autoren:** Yue Jiang, Zixin Guo, Hamed R. Tavakoli, Luis A. Leiva, Antti Oulasvirta  
**Quelle:** arXiv:2404.10163 (Aalto University)  
**PDF:** `Tier2_Sorgfaeltig/JIE24_EyeFormer.pdf`

---

## Kernfrage
Wie kann man personalisierte Scanpaths vorhersagen — also nicht die durchschnittliche Population, sondern das spezifische Blickverhalten einer Einzelperson?

## Methode
- Transformer-Architektur (ViT Vision Encoder + Fixation Decoder) als Policy Network
- RL (REINFORCE) für sequentielle Fixationsplanung — optimiert nicht-differenzierbare Objectives
- Personalisierung via **Viewer Encoder** (Section 3.5): 2-Layer Transformer, fine-getuned mit **n_path = 50 Scanpaths** pro Nutzer durch Backpropagation
- Reward: DTWD (räumlich + zeitlich) + Salient-Value-Reward mit Inhibition-of-Return
- Validiert auf UEyes (GUIs: Poster, Desktop, Mobile, Web) + OSIE (Naturszenen)

## Wichtigste Ergebnisse
- **Erstmals:** Population-Level UND individuelles Scanpath-Modell in einem System
- Personalisierung auf n_path=50 Scanpaths — praktisch anspruchsvoll, nicht trivial
- Räumliche UND zeitliche Charakteristik abgedeckt (Fixationsort + -dauer via DTWD)
- Generalisiert über UI-Typen (Web, Mobile, Desktop, Poster, Naturszenen)
- **Eigenes Gap-Statement (Section 2):** *"All previous research has concentrated on modeling scanpath patterns at the population level [...] and no prior work has focused on the prediction of personalized scanpaths."*
- **Limitations selbst benannt:** *"It remains a practical challenge, however, how to obtain such data on an individual"* — direkt zitierbar für Kap. 3 Begründung warum du Presets verwendest
- Baut auf TOD19 (Ref [51]) auf — explizit für personalisierte GUI-Layout-Optimierung

## Verwendung in der Thesis

### Kapitel 2: Related Work (★★ relevant)
- Abschnitt "Personalized Gaze Prediction", nach GUO26
- Positioniert deinen Personality-Preset-Ansatz als bewussten Trade-off: Interpretierbarkeit + kein Nutzerdaten-Bedarf vs. maximale Präzision
- *"EyeFormer (Jiang et al., 2024) demonstrates that individual-level scanpath prediction is technically feasible using a viewer encoder fine-tuned on per-user gaze samples. The present pipeline approximates inter-individual variation through personality presets rather than learned gaze profiles, prioritizing deployability over personalization precision."*

### Kapitel 3: Methodik (★ marginal)
- 1 Satz bei Einführung des Personality Presets
- *"While individual-level scanpath personalization is technically feasible (Jiang et al., 2024), it requires 50+ user-specific gaze samples unavailable in pre-deployment evaluation contexts — a limitation the authors themselves acknowledge (ibid., Section 7); personality presets provide a deployable approximation."*

### Kapitel 6: Diskussion / Future Work (★★★ zentral)
- Stärkste Stelle — konkreter Future-Work-Anker
- *"EyeFormer (Jiang et al., 2024) demonstrates that few-shot personalization with ~50 gaze samples per viewer is sufficient to capture individual scanpath characteristics; integrating a similar viewer encoder into Stage 2 — replacing static personality presets with dynamically calibrated gaze profiles — represents a concrete direction for future work."*

> **Was du lesen musst:** Abstract + Figure 1 (Population vs. Personalized) + **Section 3.5** (Personalized Scanpaths — nicht 4.2!) + Section 7 Limitations — ca. 20 Min.

## Kritik / Offene Fragen
- ⚠️ **DOI ist Placeholder** (`10.1145/nnnnnnn.nnnnnnn`) — reiner arXiv-Preprint (arXiv:2404.10163v2), noch kein veröffentlichtes ACM Paper. Bei Abgabe (Dez. 2026) prüfen ob publiziert
- 50 Scanpaths pro Nutzer = praktisch nicht trivial; "few-shot" ist irreführend
- Keine Cognitive Load Modellierung
- Dataset = UEyes (JIA23) — gleicher Datensatz wie in deiner Pipeline ✅

## Verbindungen zu anderen Papern
- Gleiche Gruppe wie → JIA23 (UEyes-Dataset verwendet), GUO26 (SeekUI)
- Baut auf → TOD19 (Ref [51] im Paper — personalisierte GUI-Layout-Optimierung)
- Ergänzt → GUO26 (task-conditioned) + JIE24 (personalisiert) = volles Bild individualisierter Gaze-Vorhersage

---

**Tags:** #personality-layer #scanpath #personalization #transformer #RL #few-shot
