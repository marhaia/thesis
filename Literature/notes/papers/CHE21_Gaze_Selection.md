# An Adaptive Model of Gaze-based Selection (2021)

**Autoren:** Xiuli Chen, Aditya Acharya, Antti Oulasvirta, Andrew Howes  
**Quelle:** CHI 2021, ACM  
**DOI:** 10.1145/3411764.3445177  
**PDF:** `Tier2_Sorgfaeltig/CHE21_Gaze_Selection.pdf`

---

## Kernfrage
Wie kann man die adaptive Steuerung von Augenbewegungen bei der Gaze-basierten Auswahl (Selektion durch Blick + Dwell-Zeit) computational modellieren?

## Methode
- RL-Modell: Sequentielle Planung als optimales Entscheidungsproblem
- Bounds: Visuelle und motorische Systemgrenzen (Sehschärfe, Sakkadendauer, Motorikrauschen)
- Validierung: Repliziert bekannte Befunde (Zielgröße, Distanz-Effekte)

## Wichtigste Ergebnisse
- Modell sagt Anzahl der Fixierungen und Dwell-Dauer für Gaze-Selektion vorher
- Adaptive Strategie entsteht emergent aus RL — nicht hand-kodiert
- Erfasst individuelle Unterschiede in Sakkaden-Planung
- Validiert an empirischen Datensätzen mit Zielgröße und Distanz als Variablen

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2), Methode
- **Argument das es stützt:** Stage 2 kann Fixationsanzahl und -dauer aus UI-Eigenschaften vorhersagen — direkte Parallele zu deinem Head 2
- **Direkt zitierbar für:** "Ähnlich wie Chen et al. (2021) modellieren wir visuelle Selektion als bounded optimal planning problem"

> **Was du lesen musst:** Abstract + Section 3 (Model) + Table 1 (Ergebnisse) — ca. 20 Min.

## Kritik / Offene Fragen
- Fokus auf Gaze-Selektion (Dwell), nicht auf freie visuelle Suche
- Kein Task-Kontext — reine Selektion, kein Dashboard-Szenario

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR Framework), JOK20 (Fixation Modelling)
- Ergänzt → SHI24 (supervisory control als nächster Schritt)

---

**Tags:** #stage2 #RL #fixation #gaze #computational-model #CR
