# UiLab: A Workbench for Conducting and Reproducing Experiments in GUI Visual Design (2021)

**Autoren:** Burny, N., & Vanderdonckt, J.  
**Quelle:** ACM (Université catholique de Louvain)  
**DOI:** —  
**PDF:** BUR21.pdf  
**ID:** BUR21 | **Status:** 🟢 Analyzed (Screening: KEEP)  

---

## Kernfrage
Wie kann das Durchführen und Reproduzieren von Experimenten zur visuellen Gestaltung von GUIs systematisiert und werkzeuggestützt werden?

## Methode
- Entwicklung von UiLab: ein Workbench-System zur GUI-Experiment-Infrastruktur
- Unterstützt verschiedene Geräte/Auflösungen, visuelle Design-Maße, Analyse-Workflows
- System-/Tool-Paper

## Wichtigste Ergebnisse
- Manuelle GUI-Designexperimente: 74–91h pro Studie → UiLab reduziert auf <3h (Tab. 3, S. 25)
- **7-Level Replicability Framework** (Fig. 4, S. 10): definiert Standards von Level A (Beschreibung) bis F (vollständige Reproduzierbarkeit)
  - **Level D** = automatisierte Metrikberechnung — das ist das Ziel von Stage 1
  - **Level E** = Software-Zugänglichkeit — rechtfertigt den Bau eines eigenen Tools
- Tab. 2 (S. 9): Analyse bestehender Tools (inkl. AIM) — prüfen ob AIM dort als nicht-skalierbar (1 Input gleichzeitig) gelistet ist → wenn ja: direkter Gap-Beleg für deine Batch-Pipeline
- Unterscheidung Evaluation vs. Experimental Research (Tab. 1, S. 5): deine Thesis = Research (Ziel: generalisierbare Regeln für Automotive UIs)
- S. 2: *"Science is a cumulative process"* — ohne Automatisierung + Datensharing kein wissenschaftlicher Fortschritt
- Cloud-basierte Architektur (service-oriented): Screenshot-Capture → Metrikberechnung → Analyse vollständig automatisiert
- **Kernproblem laut Paper:** Manuelle Experimente = Bottleneck für "Cumulative Science" in HCI

## Verwendung in der Thesis

### Kapitel 1: Einleitung (optional)
- Motivationszahl: *"Manual GUI evaluation is not scalable for pre-deployment use — existing workflows require 74–91 hours per experiment (Burny & Vanderdonckt, 2021)."*

### Kapitel 2: Related Work (★ marginal)
- Nur wenn Tab. 2 AIM als nicht-skalierbar belegt — dann 1 Satz als Gap-Beleg
- *"Existing computational GUI evaluation tools process single inputs sequentially (Burny & Vanderdonckt, 2021), lacking the batch-processing capability required for pre-deployment screening of multiple UI variants."*

### Kapitel 3: Methodik (★ marginal)
- Optional: Replicability Level D/E als externer Standard
- *"The pipeline is designed to meet Level D replicability (Burny & Vanderdonckt, 2021), producing deterministic, automatable metric outputs for identical inputs."*

> **Was du lesen musst:** S. 9 Tab. 2 (Skalierbarkeit bestehender Tools — prüfen ob AIM gelistet) + S. 10 Fig. 4 (Replicability Levels D/E) — 15 Min.  
> **Entscheidend:** Wenn Tab. 2 AIM als nicht-skalierbar zeigt → Kap. 2 Rolle wird stärker. Wenn nicht → BUR21 bleibt Footnote-Zitat.

## Kritik / Offene Fragen
- Tool-Paper ohne theoretischen Beitrag — begrenzte Tiefe
- Kein DOI vorhanden — Zitierbarkeit prüfen
- Fokus auf visuelle Design-Maße, nicht auf kognitive Metriken
- ⚠️ Schwächstes Paper im Set — Kandidat zum Rausschmeißen wenn Kap. 2 zu dicht wird. OUL18 deckt dieselbe Argumentation stärker ab.

## Verbindungen zu anderen Papern
- Komplementär → OUL18 (AIM): ähnlicher Ansatz, beide schaffen Plattformen für computational GUI evaluation
- Ergänzt → BUR22 (Consistency): gleiche Autorengruppe, Folgearbeit

---

**Tags:** #GUI-evaluation #reproducibility #workbench #visual-design #metrics #methodology
