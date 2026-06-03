# EyeFormer: Predicting Personalized Scanpaths with Transformer-Guided Reinforcement Learning (2024)

**Autoren:** Yue Jiang, Zixin Guo, Hamed R. Tavakoli, Luis A. Leiva, Antti Oulasvirta  
**Quelle:** arXiv:2404.10163 (Aalto University)  
**PDF:** `Tier2_Sorgfaeltig/JIE24_EyeFormer.pdf`

---

## Kernfrage
Wie kann man personalisierte Scanpaths vorhersagen — also nicht die durchschnittliche Population, sondern das spezifische Blickverhalten einer Einzelperson?

## Methode
- Transformer-Architektur für Scanpath-Repräsentation
- RL für sequentielle Fixations-Planung
- Few-shot Personalization: Wenige Scanpath-Samples eines Nutzers → personalisiertes Modell
- Validiert auf natürlichen Szenen und GUIs (inkl. Poster, Web, Mobile)

## Wichtigste Ergebnisse
- Erstmals: Population-Level UND individuelles Scanpath-Modell in einem System
- Few-shot Personalisierung funktioniert mit sehr wenigen Beispiel-Scanpaths
- Räumliche UND zeitliche Charakteristik wird abgedeckt (Fixationsort + -dauer)
- Generalisiert über UI-Typen (Web, Mobile, Desktop, Poster, Naturszenen)

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Personality Layer / Optionaler Stage 2 Extension)
- **Argument das es stützt:** Personalisierung ist technisch machbar — rechtfertigt den "optionalen Personality Layer" als zukünftige Erweiterung
- **Direkt zitierbar für:** "EyeFormer (Jiang et al., 2024) zeigt, dass Few-shot Personalisierung von Scanpaths möglich ist — als potentielle Erweiterung unseres Personality Layers"

> **Was du lesen musst:** Abstract + Figure 1 (Population vs. Personalized) + Section 4.2 (Few-shot) — ca. 20 Min.

## Kritik / Offene Fragen
- ArXiv-Preprint — peer-review Status prüfen bei Abgabe
- Personalisierung braucht echte Nutzer-Scanpath-Samples — nicht immer verfügbar
- Keine Cognitive Load Modellierung

## Verbindungen zu anderen Papern
- Gleiche Gruppe wie → JIA23 (UEyes), GUO26 (SeekUI)
- Ergänzt → GUO26 (task-conditioned) + Personalisierung = volles Bild

---

**Tags:** #personality-layer #scanpath #personalization #transformer #RL #few-shot
