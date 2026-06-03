# A Comparative Study of Scanpath Models in Graph-Based Visualization (2025)

**Autoren:** Angela Lopez-Cardona, Parvin Emami, Sebastian Idesis, Saravanakumar Duraisamy, Luis A. Leiva, Ioannis Arapakis  
**Quelle:** *CHI 2025* (Telefónica Research / University of Luxembourg)  
**PDF:** `Tier3_Ueberfliegen/LOP25_Scanpath_Comparison.pdf`

---

## Kernaussage
Systematischer Vergleich von Scanpath-Vorhersagemodellen auf Graph-Visualisierungen (Kriminalfall-Graphen in Digital Forensics). 40 Versuchspersonen, Eye-Tracking. Modelle: **DeepGaze**, **UMSS**, **Gazeformer** — verglichen nach Vorhersagegenauigkeit bei verschiedenen Frage-Komplexitätsstufen und Graph-Größen.

## Methodik
- **40 Teilnehmer** analysieren Kriminalfall-Graphen, beantworten Fragen unterschiedlicher Komplexität
- Synthetische Scanpaths (von Modellen) vs. menschliche Scanpaths
- Verglichene Modelle:
  - **DeepGaze** (Kümmerer et al.) — saliency-basiert
  - **UMSS** — uncertainty-modellierend
  - **Gazeformer** (Mondal et al.) — Transformer, goal-directed, SOTA für object-search
- Variablen: Fragekomplexität (einfach/komplex) + Knotenanzahl im Graph

## Evaluationsmetriken
- Saliency-basierte Metriken (wo wird fixiert)
- Temporale Metriken (Reihenfolge der Fixationen) — entscheidend für Scanpath-Qualität
- Unterschied Saliency vs. Scanpath: gleiche Coverage, aber andere Sequenz → verschiedene Metriken nötig

## Hauptbefunde
- Gazeformer performt am besten für goal-directed Suche
- Alle Modelle degradieren bei höherer Fragekomplexität — task-conditioning bleibt schwieriges Problem
- **InfoVis-spezifische Modelle** nötig (allgemeine Naturszenen-Modelle übertragen schlecht)

## Relevanz für meine Thesis
- **Kapitel:** Evaluation → Stage 2 Benchmarking-Methodik
- **Argument:** Zeigt welche **Evaluationsmetriken** für Scanpath-Vergleiche etabliert sind und wie man saliency-basierte vs. temporale Metriken trennt
- **Zitierbar für:** "Für die Evaluation von Stage 2 Head 1 (Scanpath) folgen wir der Methodik von Lopez-Cardona et al. (2025): temporale Metriken zusätzlich zu saliency-basierten Maßen"
- **Wichtig:** InfoVis ≠ GUI — Übertragbarkeit eingeschränkt; nur Evaluationsmethodik übernehmen, nicht die Ergebnisse

> **Lesen nötig?** Ja (Tier 3 must-read) — Evaluationsmetriken für Stage 2 Abschnitt nachschlagen.

---

**Tags:** #scanpath #evaluation #benchmarking #stage2 #InfoVis #metrics #gazeformer
