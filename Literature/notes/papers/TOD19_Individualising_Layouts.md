# Individualising Graphical Layouts with Predictive Visual Search Models (2019)

**Autoren:** Kashyap Todi, Jussi Jokinen, Kris Luyten, Antti Oulasvirta  
**Quelle:** ACM TOCHI 2019  
**DOI:** 10.1145/3241381  
**PDF:** `Tier2_Sorgfaeltig/TOD19_Individualising_Layouts.pdf`

---

## Kernfrage
Wie kann man Layout-Redesign individualisieren, sodass Nutzer häufige Elemente auf unbekannten Interfaces schneller finden?

## Methode
- 4 technische Prinzipien (inspiriert vom Human Visual System):
  1. Frequenz-basiertes Template
  2. Recall-Wahrscheinlichkeit (Serial Position Curve)
  3. Visual Statistical Learning (wahrscheinlichste Positionen)
  4. Generatives kognitives Modell (Visual Sampling Modelling)
- Nutzerstudie zur Validierung der Suchzeit-Reduktion
- Baseline: unmodifiziertes Layout

## Wichtigste Ergebnisse
- Individualisierte Layouts reduzieren Suchzeiten signifikant
- Visual Sampling Modelling (Prinzip IV) schneidet am besten ab
- HVS-Prinzipien können rechnerisch angewendet werden — kein User-Testing nötig
- Personalisierung ist effektiver als generische Optimierung

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2 — Predicted Search Time)
- **Argument das es stützt:** Predicted Search Time ist als Metrik validiert und individualisierbar — dein Stage 2 Head 2 baut direkt darauf auf
- **Direkt zitierbar für:** "Todi et al. (2019) validieren Predicted Visual Search Time als individualisierbare Metrik für Layout-Optimierung"

> **Was du lesen musst:** Abstract + Section 3 (4 Prinzipien) + Table 2 (Ergebnisse) — ca. 30 Min.  
> **Besonders wichtig:** Prinzip IV (Visual Sampling) — das ist die engste Verbindung zu deinem Modell.

## Kritik / Offene Fragen
- Fokus auf Individualisierung (bekannte User-History) — dein Modell evaluiert unbekannte Nutzer
- Web-/App-Layouts, nicht explizit Automotive HMI

## Verbindungen zu anderen Papern
- Baut auf → JOK20 (Adaptive Feature Guidance), TOD18 (Familiarisation)
- Direkte Weiterentwicklung von → TOD18
- Ergänzt → GUO26 (task-conditioned statt user-conditioned)

---

**Tags:** #stage2 #search-time #visual-search #HVS #layout-optimization #individualization
