# Aalto Interface Metrics (AIM): A Service and Codebase for Computational GUI Evaluation (2018)

**Autoren:** Oulasvirta, A., De Pascale, S., Koch, J., Langerak, T., Jokinen, J., Todi, K., Laine, M., Kristhombuge, M., Zhu, Y., Miniukovich, A., Palmas, G., & Weinkauf, T.  
**Quelle:** UIST '18 Adjunct (Poster), Berlin, Germany — pp. 16–19  
**DOI:** 10.1145/3266037.3266087  
**PDF:** OUL2018.pdf  
**ID:** OUL18 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Wie kann GUI-Evaluation automatisiert und zugänglich gemacht werden — über eine einheitliche Webservice-Plattform mit berechneten Metriken?

## Methode
- Entwicklung des AIM-Webservices: API + Codebase für computational GUI evaluation
- Implementierung verschiedener Wahrnehmungsmodelle als Metriken (Clutter, Saliency, Visual Complexity etc.)
- Poster-Paper / System-Paper — kein klassisches Experiment

## Wichtigste Ergebnisse
- **17 Metriken** in 4 Kategorien (Table 1): Color Perception / Perceptual Fluency / Visual Guidance / Accessibility
  - Visual Search Performance [Ref 7 = Jokinen et al. 2017] + Itti-Koch Saliency = direkte Vorläufer deiner Stage 1 Features
  - `Visual Complexity`, `Edge Density`, `Contour Congestion`, `White Space`, `Symmetry`, `Colorfulness` → alle im Stage 1 Feature-Vektor ℝ¹⁹ enthalten
  - ⚠️ **Kein Cognitive Load, kein Task-Kontext** → die Lücke ist aus Table 1 direkt ablesbar
- Architecture: URL → Headless Chrome Screenshot → Segmentation → Metrics (parallel) → JSON → das ist exakt deine Stage 1 Pipeline-Architektur
- Open-source Python API, uniform metric interface → dein `aim/metrics/`-Ordner folgt demselben Pattern
- **Key motivation quote (Introduction):** *"evaluation in interface and interaction design practice relies on personal experience and empirical testing, and less so on computational modeling"* — wörtliche Motivation für deine Pipeline
- **Key gap quote (Introduction):** *"previous work on automated evaluation has had limited scope (in terms of number of metrics) or have not been easily extendable"* — dein Beitrag = erweitert um kognitive Load-Dimension
- **Key research-goal quote (Introduction):** *"secondary goal of AIM is to facilitate research efforts centered around computational models of human-computer interaction"* — deine Thesis ist genau diese Forschungsarbeit

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★★ zentral — Ausgangspunkt der gesamten Arbeit)
- Introduction-Quote = wörtliche Motivation für deine Pipeline → direkt im ersten Absatz zitierbar
- *"Oulasvirta et al. (2018) establish that computational GUI evaluation is feasible through AIM's 17 image-based metrics — the present pipeline extends this foundation by adding task-contextualised cognitive load estimation, closing the gap that AIM's Table 1 makes explicit: no cognitive load metric exists."*
- Abstract: *"enabling designers and practitioners to detect shortcomings and possible improvements"* → dein Versprechen erweitert das um CL und Task-Kontext

### Kapitel 2: Related Work (★★★ Hauptreferenz — Systembaseline)
- Table 1: 17 Metriken explizit aufführen → zeigt welche Features du übernimmst vs. was du ergänzt
- Kategorie "Visual Guidance" (Visual Search Performance + Itti-Koch Saliency) = direkter Vorläufer von Stage 1 `search_efficiency` und `saliency`-Feature
- Fehlende 18. Metrik = Cognitive Load → das ist dein Beitrag, explizit so benennen
- ⚠️ Abgrenzung: AIM ist vollständig bottom-up (kein Nutzermodell, kein Task, keine Load-Schätzung) — Stage 2 + Task Descriptor = systematische Erweiterung nach oben

### Kapitel 3: Methodik (★★★ direkte technische Basis)
- Implementation-Abschnitt: Python Backend + Segmentation Script → deine Stage 1 (`app.py`, Flask, `aim/`-Modul) ist der direkte Nachfolger
- "Metrics are independent from each other, and therefore can be run in parallel" → deine Parallelisierung in Stage 1 ist AIM-konform und explizit begründbar
- Uniform API-Pattern (Screenshot + Segmentation → numerische Scores) → identisch mit deinem Feature-Extraction-Layer
- Jokinen et al. 2017 [Ref 7] als Visual Search Performance-Basis → Brücke zu JOK20

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Keine explizite Limitations-Section (Poster) — Table 1 ist die implizite Limitation: keine Load-Metrik
- *"previous work has had limited scope or have not been easily extendable"* → deine Erweiterung ist genau die Antwort darauf
- OUL18 nennt open-source als Ziel → deine Pipeline folgt demselben Prinzip

> **Was du lesen musst:** Abstract + Introduction (5 Min.) + **Table 1** (Metrics-Übersicht, 5 Min.) + **Implementation-Abschnitt** (Architektur, 5 Min.)  
> ⚠️ Nur 4 Seiten gesamt — komplett lesen, dauert 15 Min. Kein Experiment, keine Results-Section zum Überfliegen.

## Kritik / Offene Fragen
- Nur Poster-Paper (4 Seiten) → keine empirische Validierung der Metrik-Kombinationen
- Alle 17 Metriken bottom-up, bildbasiert — kein Nutzermodell, kein Task, kein Load
- Visual Search Performance [Ref 7] basiert auf Jokinen et al. 2017 (Keyboard-Layout) — nicht direkt auf JOK20 (2020), aber derselbe Modelltyp
- Web-Service (interfacemetrics.aalto.fi) — Verfügbarkeit langfristig nicht garantiert; Python-Codebase ist der robustere Anker

## Verbindungen zu anderen Papern
- Erweitert durch → Stage 2 + Task Descriptor: die fehlende 18. Metrik (Cognitive Load) wird ergänzt
- Theoretisch untermauert durch → OUL22: CR liefert das "Warum" hinter AIM's Saliency/Search-Metriken
- Konkret erweitert durch → JOK20 Section 8.1: "an automated segmenter can be used" = AIM ist dieser Segmenter
- Lücke belegt durch → DAS24: zeigt empirisch dass AIM's Saliency ohne CL-Kontext unzureichend ist
- Explizit zitiert in → DAS24 Section 2.3 [Ref 36] + Section 3.2 (FC Score) → AIM ist im Forschungskontext verankert
- Visual Search Performance [Ref 7] → Jokinen et al. 2017 = Vorgänger von JOK20 (2020)

---

**Tags:** #AIM #GUI-evaluation #computational-metrics #baseline #platform #bottom-up #saliency #clutter #visual-complexity #UIST2018 #stage1 #feature-extraction #open-source
