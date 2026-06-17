# Exposé (Standardversion) — ~4 Seiten

**Arbeitstitel:** Integrating Cognitive Predictive Metrics into the AIM Platform for Automated GUI Evaluation  
**Verfasserin:** Hannah  
**Studiengang:** User Experience Design  
**Betreuer:** Gerhard Graf und Prof. Dr. Christian Sturm  
**Datum:** Juni 2026

---

## 1. Einleitung und Problemstellung

Die automatisierte Bewertung von graphischen Benutzeroberflächen (GUIs) ist ein zentrales Problem der HCI-Forschung: Designer:innen benötigen skalierbare, reproduzierbare Qualitätsindikatoren, ohne für jede Designentscheidung eine aufwändige Nutzerstudie durchführen zu müssen. Mit der Aalto Interface Metrics (AIM) Plattform existiert ein etablierter Ansatz, der aus statischen Screenshots wahrnehmungsbasierte Metriken wie visuelle Unordnung (Clutter), Salienz und Symmetrie berechnet (Oulasvirta et al., 2018). Diese Metriken sind reproducierbar und objektiv — sie erfassen jedoch ausschließlich die physikalischen Eigenschaften des Bildes, nicht das Zusammenspiel aus Interfacestruktur, Nutzungskontext und kognitiver Kapazität der Nutzerin.

Diese Lücke ist nicht trivial. Das et al. (2024) zeigen empirisch, dass kognitive Doppelbelastung (z.B. gleichzeitiges Fahren und Ablesen eines Dashboards) das Blickverhalten fundamental verändert: Nutzer:innen zeigen ausgeprägte Tunnelblick-Phänomene, übersehen periphere UI-Elemente und bilden verkürzte, ineffiziente Scanpaths. AIM kann dieses verhaltensrelevante Phänomen nicht vorhersagen, weil es keinen Task-Kontext und kein kognitives Modell enthält. Jiang et al. (2023) ergänzen, dass UI-spezifische Aufmerksamkeitsmuster — insbesondere ein starker Oben-Links-Bias in mobilen Interfaces — von taskagnostischen Salienzmodellen systematisch unterschätzt werden.

Die vorliegende Masterarbeit fragt daher: **Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?**

---

## 2. Forschungsstand und theoretischer Rahmen

### 2.1 AIM und computationale GUI-Evaluation

AIM (Oulasvirta et al., 2018) ist eine Open-Source-Plattform, die ca. 20 wahrnehmungsbasierte Metriken aus GUI-Screenshots berechnet. Ergänzende Arbeiten wie UiLab (Burny & Vanderdonckt, 2021) und Criticmate (Ko et al., 2026) erweitern den Evaluierungsrahmen um Konsistenz- und mehrstufige Kritikprozesse. UIClip (Wu et al., 2024) zeigt, dass datengetriebene CLIP-Modelle GUI-Qualitätsurteile erlernen können — liefern aber keinen mechanistischen Erklärungsrahmen und keine Verhaltensvorhersage.

### 2.2 Computational Rationality als theoretische Grundlage

Das Rahmenwerk der Computational Rationality (CR) nach Oulasvirta, Jokinen und Howes (2022) ersetzt regelbasierte Heuristiken (GOMS, ACT-R-Rezepte) durch sequentielle Entscheidungstheorie: Interaktion wird als optimale Kontrollpolitik modelliert, die kognitive, wahrnehmungsbasierte und motorische Grenzen balanciert. Formalisiert als Partially Observable Markov Decision Process (POMDP), gelöst via Reinforcement Learning, erklärt CR den Übergang von Novizen-geleiteter, bottom-up-Suche zu expert-geführtem, top-down-Memorabruf (Jokinen et al., 2020). Aktuelle CR-Anwendungen umfassen Gaze-basierte Selektion (Chen et al., 2021), Touchscreen-Tippen (Shi et al., 2024), task-konditionierte Scanpaths (Shi et al., 2025), Attention-Switching in Dual-Task-Szenarien (Lingler et al., 2024) sowie Adaptive UI via Multi-Agent RL (Langerak et al., 2024).

### 2.3 Salienz, Scanpaths und Cognitive Load

Scanpath-Vorhersage hat sich von taskagnostischen Salienzmodellen (DeepGaze II, Kümmerer et al., 2017) zu task-konditionierten Ansätzen entwickelt: SeekUI (Guo et al., 2026) und Chartist (Shi et al., 2025) zeigen, dass task-spezifische Konditionierung die Vorhersagequalität signifikant verbessert. EyeFormer (Jiang et al., 2024) demonstriert die Machbarkeit personalisierter Scanpaths durch Few-Shot-Konditionierung. Für die simultane Vorhersage mehrerer Outputs (Geschwindigkeit, Genauigkeit, Aufwand aus einem Modell) liefert Miazga et al. (2026) methodische Grundlagen.

### 2.4 Lücke im Forschungsstand

Keine existierende Arbeit kombiniert: (a) task-unabhängige Komplexitätsextraktion als erklärbare Eingangsschicht, (b) task-konditionierte Verhaltensvorhersage mit (c) einem expliziten Coherence-Term zwischen Salienz, Scanpaths und Cognitive Load — integriert in eine bestehende GUI-Evaluierungsplattform (AIM).

---

## 3. Forschungsdesign und Methodik

### 3.1 Architektur: Zweistufige Pipeline

**Stage 1 — Task-unabhängige Komplexitätsextraktion**

Input: GUI-Screenshot (automotive HMI). Output: Visueller Feature-Vektor $v \in \mathbb{R}^8$.

| Feature | Beschreibung | Referenz |
|---------|-------------|---------|
| Shannon Entropie | Informationsdichte (Graustufenhistogramm) | Shannon (1948) |
| Edge Density | Anteil Kantenpixel via Canny | — |
| Information Clutter | Feature Congestion + Subband Entropy | Rosenholtz et al. (2007) |
| Layout Symmetry | Vertikale/horizontale Balance | Miniukovich & De Angeli (2015) |
| Chromatic Coherence | Farbpaletten-Varianz | — |
| Visual Hierarchy | Größengradient, Kontrast, Gruppenstruktur | Tuch et al. (2009) |
| Element Density | Anzahl interaktiver Elemente pro Fläche | — |

Stage 1 ist taskagnostisch und direkt in die bestehende AIM-Infrastruktur integrierbar.

**Stage 2 — Task-konditionierte Multi-Output-Vorhersage**

Input: $v$ (Stage 1) + Task Descriptor (kategorial) + User Profile (Big Five, optional).  
Output: Drei simultane Heads mit Mutual Coherence Constraints.

$$L = \lambda_1 L_\text{sal} + \lambda_2 L_\text{fix} + \lambda_3 L_\text{load} + \lambda_\text{coh} \cdot L_\text{coherence}$$

| Head | Output | Validierungsmetrik |
|------|--------|-------------------|
| Head 1 | Saliency Map | AUC-Judd, NSS, SIM, CC |
| Head 2 | Fixationsverteilung / Scanpath-Entropie | DTW, TDE |
| Head 3 | Cognitive Load Index (0–100) | Korrelation mit NASA-TLX |

Der **Coherence-Term** $L_\text{coherence}$ bestraft physikalisch inkonsistente Ausgaben (z.B. hohe Salienz-Dispersion bei gleichzeitig niedriger Fixationsanzahl). Die Kopplung der drei Heads ist das methodische Novum der Arbeit.

**User Profile:** LLM-basierte Simulation auf Big-Five-Trait-Level ist methodisch vertretbar (Serapio-García et al., 2023; Argyle et al., 2023). State-based Simulation (akuter Stress, Fatigue) wird explizit als nicht-validiert deklariert (Kapania et al., 2025).

### 3.2 Empirische Validierung

**Datenbasis 1 — Pilotdatensatz (bereits vorhanden):**
- N=32, Eye-Tracking + NASA-TLX
- Automotive HMI, zwei Fahraufgaben (Lane-Change, Emergency Braking)
- Verwendung: Kalibrierung Stage 1; explorativ für Stage 2

**Datenbasis 2 — Geplante Nutzerstudie:**
- N≈30–35, within-subjects Design
- 8–10 automotive GUI Screenshots (Komplexitätsvariation: minimalistisch → informationsdicht)
- 2–3 Tasks pro Screenshot (Visual Search, Monitoring, Navigation)
- Highway-Szenario bei Konstantgeschwindigkeit (reduzierter Driving-Load)
- ~24 Zellen/Teilnehmer

**Baseline-Vergleiche:**

| Baseline | Beschreibung |
|---------|-------------|
| B1 | Stage 1 allein + lineare Regression |
| B2 | Stage 2 ohne Coherence-Term ($\lambda_\text{coh} = 0$) |
| B3 | Single-Output-Modell (nur Head 3: Load) |

**Kernauswertung:**
- Varianzdekomposition: Welchen Anteil erklärt Stage 1 allein? Was bringt Stage 2 zusätzlich?
- Coherence-Gewinn: B2 vs. vollständiges Stage 2
- Null-Ergebnisse werden explizit publiziert

---

## 4. Erwartete Beiträge

**Technisch:** Erste Integration eines Cognitive Load Predictors in die AIM-Pipeline — als Open-Source-Erweiterung, Web-App oder Figma-Plugin.

**Empirisch:** Varianzdekomposition task-unabhängiger Komplexität (Stage 1) vs. task-konditionierter Verhaltensvorhersage (Stage 2) in einem kontrollierten Automotive-HMI-Setting.

**Theoretisch:** Überprüfung der CR-These, dass kognitive Belastung als Outcome einer optimalen Kontrollpolitik über visuelle Interface-Eigenschaften vorhergesagt werden kann.

**Praktisch:** Screening-Werkzeug für frühe Designphasen — kein Ersatz für Nutzerstudien, aber skalierbare erste Einschätzung.

---

## 5. Zeitplan

> *Hinweis: Die gesamte Pipeline (Stage 1, Stage 2, API-Integration, Salienzmodell, kognitive Nutzermodelle, Coherence-Check, User Profile) ist bereits vollständig implementiert. Die verbleibende Arbeit umfasst Datenerhebung, Modelltraining auf echten Daten und das Schreiben der Thesis.*

| Phase | Inhalt | Zeitraum |
|-------|--------|----------|
| ~~Implementierung~~ | Stage 1+2, Flask-API, AIM-Integration | ✅ abgeschlossen |
| Literatur | Tier 1+2 Paper lesen, Lücken schließen | Juni 2026 |
| Studie (Vorbereitung) | Rekrutierung, Ethikantrag, Stimulusmaterial, Pilottest | Juli 2026 |
| Studie (Erhebung) | Haupterhebung N≈30–35, Eye-Tracking + NASA-TLX | Aug 2026 |
| Auswertung | Eye-Tracking-Prozessierung, Varianzdekomposition Stage 1 vs. Stage 2 | Sep 2026 |
| Modell-Training | Training auf echten Daten, Baseline-Vergleiche (B1–B3) | Okt 2026 |
| Schreiben | Thesis-Kapitel 1–7 verfassen | Nov 2026 |
| Abgabe | — | **Dez 2026** |

---

## 6. Eingrenzungen und Limitationen

- Domänenbeschränkung: ausschließlich automotive HMI-Screenshots; keine Generalisierung auf andere UI-Typen ohne eigene Validierung
- Cognitive Load Index ≠ NASA-TLX; der Output ist ein berechneter Interaktionskomplexitäts-Index
- LLM User Simulation: nur Trait-Level (Big Five), kein State-Level (Fatigue, Stress)
- Pilotdatensatz: Single-GUI-Design ohne Varianzdekomposition; nur für Kalibrierung geeignet

---

## 7. Literatur (Auswahl)

- Oulasvirta, A. et al. (2018). Aalto Interface Metrics (AIM). *UIST 2018*
- Oulasvirta, A. et al. (2022). Computational Rationality as a Theory of Interaction. *ACM TOCHI*
- Das, D. et al. (2024). HCEye: Dynamics of Visual Highlighting and Cognitive Load. *CHI 2024*
- Jiang, M. et al. (2023). UEyes: Understanding Visual Saliency across UI Types. *CHI 2023*
- Jokinen, J. et al. (2020). Adaptive Feature Guidance. *CHI 2020*
- Guo, Z. et al. (2026). SeekUI. *CHI 2026*
- Ko, J. et al. (2026). Criticmate. *CHI 2026*
- Wu, J. et al. (2024). UIClip. *UIST 2024*
- Shi, D. et al. (2024). CRTypist. *CHI 2024*
- Shi, D. et al. (2025). Chartist. *CHI 2025*
- Kapania, S. et al. (2025). Simulacrum of Stories. *CHI 2025*
- Langerak, T. et al. (2024). MARLUI: Multi-Agent Reinforcement Learning for Adaptive Point-and-Click UIs. *ACM TOCHI 2024*
