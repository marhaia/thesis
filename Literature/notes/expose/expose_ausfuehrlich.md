# Exposé (Ausführliche Version) — ~6 Seiten mit Gliederungsentwurf

**Arbeitstitel:** Integrating Cognitive Predictive Metrics into the AIM Platform for Automated GUI Evaluation  
**Verfasserin:** Hannah  
**Studiengang:** Master HCI / [Studiengang eintragen]  
**Betreuer:in:** [Name eintragen]  
**Datum:** Juni 2026

---

## 1. Einleitung und Motivation

Die automatisierte Bewertung von GUIs verspricht skalierbare, reproduzierbare Qualitätsurteile ohne den Zeit- und Ressourcenaufwand klassischer Nutzerstudien. Bestehende Plattformen wie AIM (Oulasvirta et al., 2018) operieren dabei auf Basis rein visueller, taskagnostischer Metriken: Sie analysieren Pixel, nicht Verhalten. Diese Abstraktion ist einerseits eine Stärke — sie erlaubt den Vergleich beliebiger GUIs ohne spezifische Nutzungsannahmen. Andererseits ist sie eine fundamentale Schwäche: Das gleiche Interface, das unter ruhiger Betrachtung "sauber" wirkt, kann unter realen Bedingungen — Zeitdruck, parallele Aufgaben, individuell unterschiedliche Kapazitäten — eine inakzeptable kognitive Belastung erzeugen.

Das ist keine Spekulation. Das et al. (2024, HCEye) zeigen empirisch: Kognitive Doppelbelastung im Fahrszenario führt zu messbarem Tunnelblick — Nutzer:innen übersehen peripherere UI-Elemente, bilden kürzere und ineffizientere Scanpaths und verpassen informationsrelevante Bereiche, die bei taskloser Betrachtung problemlos wahrgenommen würden. AIM sieht das nicht. Jiang et al. (2023, UEyes) ergänzen: Das domänen-spezifische Aufmerksamkeitsmuster von UI-Nutzer:innen (starker Oben-Links-Bias) weicht systematisch von dem ab, was allgemeine Salienzmodelle vorhersagen.

Gleichzeitig hat das Feld der computationalen Verhaltensmodellierung erhebliche Fortschritte gemacht. Auf Basis der Computational Rationality (CR, Oulasvirta et al., 2022) entstanden in den letzten Jahren task-konditionierte Scanpath-Modelle (Guo et al., 2026; Shi et al., 2025), Multi-Output-Performance-Prädiktoren (Miazga et al., 2026) und personalisierbare Fixationsmodelle (Jiang et al., 2024). Diese Methoden liefern verhaltensnahe, mechanistisch erklärbare Vorhersagen — sind aber bisher nicht in eine bestehende GUI-Evaluierungsinfrastruktur integriert.

Die vorliegende Masterarbeit schließt diese Lücke durch die Entwicklung einer **zweistufigen kognitiv-prädiktiven Erweiterungspipeline** für AIM.

**Forschungsfrage:** *Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?*

---

## 2. Forschungsstand

### 2.1 Computationale GUI-Evaluation: Stand und Grenzen

AIM (Oulasvirta et al., 2018) aggregiert ca. 20 Metriken — darunter visuelle Unordnung, Salienz, Symmetrie, Farbkontrast — zu einem vergleichbaren Interface-Qualitätsprofil. Ergänzende Werkzeuge wie UiLab (Burny & Vanderdonckt, 2021) ermöglichen reproduzierbare Experimente, BUR22 quantifiziert Multi-Screen-Konsistenz, und Criticmate (Ko et al., 2026) strukturiert die Evaluation als mehrstufigen Human-AI-Prozess (Perception → Comprehension → Projection). Black-Box-Ansätze wie UIClip (Wu et al., 2024) lernen UI-Qualitätsurteile datengetrieben — liefern aber keinen mechanistischen Erklärungsrahmen, kein Nutzerverhalten und keine Verhaltensvorhersage.

Keine dieser Arbeiten integriert kognitive Belastungsvorhersage in die Plattform.

### 2.2 Computational Rationality

CR (Oulasvirta, Jokinen, Howes, 2022) modelliert Interaktion als optimale sequentielle Entscheidung unter Unsicherheit und kognitiven Grenzen — formalisiert als POMDP, gelöst via RL. Das Rahmenwerk erklärt, warum Nutzer:innen bestimmte Pfade durch Interfaces nehmen (nicht nur welche), und erlaubt Vorhersagen aus Interface-Eigenschaften heraus. Aktuelle Anwendungen reichen von Gaze-basierter Selektion (Chen et al., 2021) über Touchscreen-Tippen (Shi et al., 2024) und Attention-Switching im Dual-Task (Lingler et al., 2024; Bai et al., 2024) bis zu audiovisuellem Suchverhalten (Cho et al., 2026). Besonders relevant: Lingler et al. (2024) zeigen, dass RL-Modelle, die auf CR-Nutzermodellen trainiert wurden, reale Nutzerperformance besser verbessern als unconstrained Modelle — ein Beleg für CR-basiertes User Modeling als valides Trainings-Proxy.

### 2.3 Scanpath- und Salienzmodelle

Die Entwicklung geht von taskagnostischer Salienzvorhersage (DeepGaze II, Kümmerer et al., 2017; SALICON, Jiang et al., 2015) zu task-konditionierten, sequentiellen Modellen: SeekUI (Guo et al., 2026) kombiniert reward-augmented VLMs mit RL für GUI-spezifische Scanpaths; Chartist (Shi et al., 2025) zeigt eine hierarchische LLM+RL-Architektur für Diagramme; EyeFormer (Jiang et al., 2024) ermöglicht personalisierte Scanpaths mit wenigen Beispielen. Lopez-Cardona et al. (2025) etablieren Benchmarking-Methodik für Scanpath-Modelle und betonen die Notwendigkeit temporaler (DTW, TDE) neben lokationsbasierten Metriken.

### 2.4 Ability-based und persönlichkeitsbasierte Nutzermodellierung

Gajos et al. (2008) und Sarcar et al. (2016) legen die Grundlage für ability-based Interface-Anpassung: Nutzerprofile auf Basis motorischer und kognitiver Fähigkeiten verbessern Performance-Vorhersagen. Auf LLM-Seite zeigen Argyle et al. (2023) und Serapio-García et al. (2023), dass Big-Five-Conditioning psychometrisch validierbare Persönlichkeitssimulation ermöglicht — während Kapania et al. (2025) die Grenzen dieser Simulation (fehlende Tiefe, Surrogate Effect) klar benennen.

### 2.5 Theoretische Lücke

Kein existierendes System kombiniert: (a) task-unabhängige, erklärbare Komplexitätsextraktion als erste Stufe, (b) task- und nutzerkonditionierte Multi-Output-Verhaltensvorhersage mit (c) einem expliziten Coherence-Constraint zwischen Salienz, Fixationsverteilung und Cognitive Load — eingebettet in eine bestehende, öffentlich zugängliche Evaluierungsplattform.

---

## 3. Forschungsdesign

### 3.1 Pipeline-Architektur

**Stage 1:** Task-unabhängige visuelle Komplexitätsextraktion aus GUI-Screenshot → $v \in \mathbb{R}^8$ (Shannon Entropie, Edge Density, Information Clutter nach Rosenholtz et al. 2007, Layout Symmetry nach Miniukovich & De Angeli 2015, Chromatic Coherence, Visual Hierarchy, Element Density). Direkt in AIM integrierbar.

**Stage 2:** Input: $v$ + Task Descriptor + User Profile (optional) → 3 simultane Outputs:

$$L = \lambda_1 L_\text{sal} + \lambda_2 L_\text{fix} + \lambda_3 L_\text{load} + \lambda_\text{coh} \cdot L_\text{coherence}$$

- **Head 1:** Saliency Map (pixel-level) — Validierung: AUC-Judd, NSS, SIM, CC, KL-Divergenz
- **Head 2:** Fixationsverteilung / Scanpath-Entropie — Validierung: DTW, TDE
- **Head 3:** Cognitive Load Index 0–100 — Validierung: Korrelation mit NASA-TLX (kein Ersatz)

Der Coherence-Term $L_\text{coherence}$ koppelt die drei Heads und bestraft physikalisch inkonsistente Kombinationen. Die Kopplung ist das methodische Novum gegenüber bestehenden Arbeiten.

### 3.2 Empirische Validierung

**Datenbasis 1 (Pilotdatensatz, vorhanden):** N=32, Eye-Tracking + NASA-TLX, automotive HMI. Verwendung: Kalibrierung Stage 1, explorative Stage 2 Analyse.

**Datenbasis 2 (Geplante Studie):** N≈30–35, within-subjects, 8–10 automotive GUIs (Komplexitätsvarianz), 2–3 Tasks pro Screenshot, Highway-Szenario. Ziel: Varianzdekomposition Stage 1 vs. Stage 2.

**Baselines:** B1 (Stage 1 + lineare Regression), B2 (Stage 2 ohne Coherence), B3 (Single-Output Load).

---

## 4. Gliederungsentwurf

```
1. Einleitung
   1.1 Motivation: Lücke zwischen visueller und kognitiver GUI-Bewertung
   1.2 Forschungsfrage und Zielsetzung
   1.3 Beitrag und Eingrenzung
   1.4 Aufbau der Arbeit

2. Hintergrund und verwandte Arbeiten
   2.1 Computationale GUI-Evaluation
       2.1.1 AIM: Architektur, Metriken, Grenzen
       2.1.2 Verwandte Plattformen (UiLab, UIClip, Criticmate)
   2.2 Kognitive Belastung in der Mensch-Computer-Interaktion
       2.2.1 Kognitive Belastungstheorie und NASA-TLX
       2.2.2 Visuelle Komplexität und Cognitive Load (HCEye, LIA24)
       2.2.3 Dual-Task in automotive HMI (LOR24, BAI24, LIN24)
   2.3 Computational Rationality
       2.3.1 Theorie und Formalisierung (OUL22, JOK20)
       2.3.2 Anwendungen: Gaze, Typing, Task-Switching
   2.4 Salienz- und Scanpathmodelle
       2.4.1 Taskagnostische Salienz (DeepGaze II, SALICON)
       2.4.2 Task-konditionierte Modelle (SeekUI, Chartist)
       2.4.3 Personalisierung (EyeFormer, Ability-based)
   2.5 LLM-basierte Nutzersimulation
       2.5.1 Möglichkeiten (ARG23, SER23)
       2.5.2 Grenzen (KAP25) — Surrogate Effect
   2.6 Forschungslücke und Positionierung

3. Systemarchitektur
   3.1 Überblick: Zweistufige Pipeline
   3.2 Stage 1: Visuelle Komplexitätsextraktion
       3.2.1 Feature-Definitionen und Berechnungsgrundlagen
       3.2.2 Integration in AIM
   3.3 Stage 2: Multi-Output-Vorhersage
       3.3.1 Input Layer (Task Descriptor, User Profile)
       3.3.2 Drei Prediction Heads
       3.3.3 Coherence Loss-Funktion
   3.4 Implementierung (Web-App / Figma-Plugin)

4. Studie und Validierung
   4.1 Studiendesign
       4.1.1 Pilotdatensatz (Kalibrierung)
       4.1.2 Geplante Hauptstudie (N≈30–35)
   4.2 Stimulusmaterial: Automotive GUI Screenshots
   4.3 Prozedur und Messinstrumente
       4.3.1 Eye-Tracking-Protokoll
       4.3.2 NASA-TLX Erhebung
   4.4 Auswertungsstrategie
       4.4.1 Varianzdekomposition Stage 1 vs. Stage 2
       4.4.2 Baseline-Vergleiche (B1, B2, B3)
       4.4.3 Coherence-Term-Analyse

5. Ergebnisse
   5.1 Stage 1: Komplexitätsvektor und Validierung
   5.2 Stage 2: Multi-Output-Performance
   5.3 Baseline-Vergleiche und Coherence-Gewinn
   5.4 Null-Ergebnisse und ungeplante Befunde

6. Diskussion
   6.1 Beantwortung der Forschungsfrage
   6.2 Theoretische Implikationen (CR als Rahmen)
   6.3 Praktische Implikationen (Screening-Tool)
   6.4 Limitationen
       6.4.1 Domänenbeschränkung (automotive only)
       6.4.2 Cognitive Load Index ≠ NASA-TLX
       6.4.3 LLM-Simulation: Trait vs. State
       6.4.4 Pilotdaten: Single-GUI-Design

7. Fazit
   7.1 Zusammenfassung der Beiträge
   7.2 Ausblick: Erweiterungen und Generalisierung

Literaturverzeichnis
Anhang
   A. Stimulusmaterial (GUI Screenshots)
   B. Studienprotokoll
   C. Feature-Berechnungsformeln
   D. Rohdaten-Übersicht
```

---

## 5. Zeitplan

> *Hinweis: Implementierung (Stage 1+2, Flask-API auf Port 5001, UMSI++ Saliency-Modell mit pretrained Weights, Jokinen 2020 AFG, HCEye-Features, Coherence-Check, User Profile, Baseline-Scripts, End-to-End-Tests) ist bereits vollständig abgeschlossen. Die verbleibende Arbeit ist Datenerhebung, Auswertung und Schreiben.*

| Phase | Inhalt | Zeitraum | Deliverable |
|-------|--------|----------|-------------|
| ~~Implementierung~~ | Stage 1+2, Flask-API, AIM-Integration | ✅ abgeschlossen | — |
| Literatur | Tier 1+2 Paper lesen, fehlende PDFs beschaffen | Juni 2026 | Vollständige Notizen |
| Studie (Vorbereitung) | Rekrutierung, Ethikantrag, Stimulusmaterial, Pilottest | Juli 2026 | Studienprotokoll |
| Studie (Erhebung) | Haupterhebung N≈30–35, Eye-Tracking + NASA-TLX | Aug 2026 | Rohdatensatz |
| Auswertung | Eye-Tracking-Prozessierung, NASA-TLX, Varianzdekomposition Stage 1 vs. 2 | Sep 2026 | Auswertungsdatei |
| Modell-Training | Training auf echten Daten, Baseline-Vergleiche (B1–B3), Coherence-Analyse | Okt 2026 | Modell + Ergebnisse |
| Schreiben | Kapitel 1–7 verfassen | Nov 2026 | Thesis-Draft |
| Abgabe | — | **Dez 2026** | Masterarbeit |

---

## 6. Limitationen (explizit)

1. **Domäne:** Automotive HMI-Screenshots; Generalisierung auf Mobile, Web oder Desktop erfordert eigene Validierung
2. **Cognitive Load Index:** Berechneter Komplexitätsindikator — kein Ersatz für empirische NASA-TLX-Erhebungen; NASA-TLX-Subskalen ohne visuelles Korrelat (Physical Demand, Frustration) werden nicht modelliert
3. **LLM-Simulation:** Nur stabile Persönlichkeitsmerkmale (Big Five Trait-Level); State-based Simulation (akuter Stress, Fatigue, Müdigkeit) ist nicht validiert und wird als harte Limitation deklariert (Kapania et al., 2025)
4. **Pilotdatensatz:** Single-GUI-Design ohne Varianzdekomposition; GTA-imputierte TLX-Daten nur exploratorisch verwertbar
5. **Stichprobengröße:** N≈30–35 für within-subjects Design; Power-Analyse ausstehend

---

## 7. Literatur (Auswahl)

- Argyle, L.P. et al. (2023). Out of One, Many. *Political Analysis*, 31(3)
- Bai, Y. et al. (2024). Heads-Up Multitasker. *CHI 2024*
- Burny, N. & Vanderdonckt, J. (2021). UiLab. *EICS 2021*
- Burny, N. & Vanderdonckt, J. (2022). UI Consistency. *EICS 2022*
- Chen, X. et al. (2021). Adaptive Model of Gaze-based Selection. *CHI 2021*
- Das, D. et al. (2024). HCEye. *CHI 2024*
- Gajos, K.Z. et al. (2008). Ability-Based Interfaces. *UIST 2008*
- Guo, Z. et al. (2026). SeekUI. *CHI 2026*
- Jiang, M. et al. (2015). SALICON. *CVPR 2015*
- Jiang, M. et al. (2023). UEyes. *CHI 2023*
- Jiang, Y. et al. (2024). EyeFormer. *CHI 2024*
- Jokinen, J. et al. (2020). Adaptive Feature Guidance. *CHI 2020*
- Kangasrääsiö, A. et al. (2017). ABC Cognitive Models. *CHI 2017*
- Kapania, S. et al. (2025). Simulacrum of Stories. *CHI 2025*
- Ko, J. et al. (2026). Criticmate. *CHI 2026*
- Kümmerer, M. et al. (2017). DeepGaze II. *ICCV 2017*
- Lingler, A. et al. (2024). Task Switching with RL. *CHI 2024*
- Lopez-Cardona, A. et al. (2025). Scanpath Models Comparison. *CHI 2025*
- Miazga, M.P. et al. (2026). Log2Motion. *CHI 2026*
- Miniukovich, A. & De Angeli, A. (2015). Interface Aesthetics. *CHI 2015*
- Oulasvirta, A. et al. (2018). AIM. *UIST 2018*
- Oulasvirta, A. et al. (2022). Computational Rationality. *ACM TOCHI*
- Rosenholtz, R. et al. (2007). Visual Clutter. *Journal of Vision*
- Sarcar, S. et al. (2016). Ability-Based Optimization. *ITAP 2016*
- Serapio-García, G. et al. (2023). Personality Traits in LLMs. *arXiv:2307.00184*
- Shannon, C.E. (1948). Mathematical Theory of Communication. *Bell System Technical Journal*
- Shi, D. et al. (2024). CRTypist. *CHI 2024*
- Shi, D. et al. (2025). Chartist. *CHI 2025*
- Todi, K. et al. (2016). Sketchplore. *CHI 2016*
- Todi, K. et al. (2018). Familiarisation. *IUI 2018*
- Todi, K. et al. (2019). Individualising Layouts. *IUI 2019*
- Wu, J. et al. (2024). UIClip. *UIST 2024*
