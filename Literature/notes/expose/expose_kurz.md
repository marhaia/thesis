# Exposé (Kurzversion) — ~2 Seiten

**Arbeitstitel:** Integrating Cognitive Predictive Metrics into the AIM Platform for Automated GUI Evaluation  
**Verfasserin:** Hannah  
**Studiengang:** User Experience Design  
**Betreuer:** Gerhard Graf und Prof. Dr. Christian Sturm  
**Datum:** Juni 2026

---

## 1. Forschungsfrage

> *Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?*

Graphische Benutzeroberflächen (GUIs) werden heute primär durch visuelle und wahrnehmungsbasierte Metriken bewertet — etwa durch die Aalto Interface Metrics (AIM) Plattform (Oulasvirta et al., 2018), die Clutter, Salienz und Symmetrie aus statischen Screenshots berechnet. Solche Werkzeuge liefern skalierbare, reproduzierbare Bewertungen, blenden jedoch zentrale Einflussfaktoren aus: den konkreten Nutzungskontext (Task), individuelle kognitive Profile (Nutzerin) und die dynamischen Wechselwirkungen zwischen visueller Wahrnehmung und kognitiver Belastung. Das führt zu einer systematischen Lücke: Eine GUI kann visuell "sauber" sein und trotzdem unter echten Nutzungsbedingungen hohe kognitive Belastung erzeugen.

---

## 2. Motivation & Problemstellung

Empirische Belege für diese Lücke liefern Das et al. (2024): Unter kognitiver Doppelbelastung verändert sich das Blickverhalten fundamental — Nutzer:innen zeigen Tunnelblick, übersehen periphere UI-Elemente und bilden kürzere Scanpaths. AIM kann dieses Phänomen nicht vorhersagen. Gleichzeitig zeigen Jiang et al. (2023), dass UI-spezifische Aufmerksamkeitsmuster (z.B. starker Oben-Links-Bias) durch taskagnostische Salienzmodelle systematisch unterschätzt werden.

Die vorliegende Arbeit adressiert dieses Problem durch eine **zweistufige Erweiterungspipeline** für AIM, die kognitive Vorhersagemetriken integriert.

---

## 3. Ansatz

Die Pipeline gliedert sich in zwei Stufen:

**Stage 1 — Task-unabhängige visuelle Komplexitätsextraktion:** Aus dem GUI-Screenshot wird ein 8-dimensionaler Feature-Vektor berechnet (Shannon Entropie, Edge Density, Information Clutter, Layout Symmetry, Chromatic Coherence, Visual Hierarchy, Interactive Element Density). Diese Stufe ist task- und nutzungsunabhängig und direkt in die AIM-Infrastruktur integrierbar.

**Stage 2 — Task-konditionierte Multi-Output-Vorhersage:** Der Feature-Vektor wird mit einem Task Descriptor und einem optionalen User Profile (Big Five Persönlichkeitsmerkmale) zu drei simultanen Outputs kombiniert: (1) Saliency Map, (2) Fixationsverteilung, (3) Cognitive Load Index (0–100). Die drei Heads werden durch einen Coherence-Term gekoppelt, der physikalisch inkonsistente Kombinationen bestraft.

Das theoretische Fundament bildet die **Computational Rationality** (CR) nach Oulasvirta et al. (2022): Nutzerverhalten als optimale Kontrollpolitik unter kognitiven, wahrnehmungsbasierten und motorischen Grenzen — modelliert als POMDP, gelöst via Reinforcement Learning.

---

## 4. Empirische Validierung

Zur Validierung werden zwei Datengrundlagen genutzt: (1) ein bestehender Pilotdatensatz (N=32, Eye-Tracking + NASA-TLX, automotive HMI, zwei Fahraufgaben) zur Kalibrierung von Stage 1; (2) eine geplante neue Studie (N≈30, within-subjects, 8–10 automotive GUI Screenshots, 2–3 Tasks pro Screenshot). Verglichen werden drei Baselines: Stage 1 allein (lineare Regression), Stage 2 ohne Coherence-Term, und ein Single-Output-Modell (nur Load).

---

## 5. Erwartete Beiträge

1. **Technisch:** Erste Integration von Cognitive Load Prediction in die AIM-Pipeline — als offener Web-Service oder Figma-Plugin.
2. **Theoretisch:** Empirische Überprüfung der CR-These, dass visuelle Komplexität (Stage 1) einen spezifischen, von Task und Nutzer unabhängigen Vorhersagebeitrag leistet.
3. **Methodisch:** Varianzdekomposition zwischen task-unabhängiger Komplexität (Stage 1) und task-konditionierter Verhaltensvorhersage (Stage 2).

---

## 6. Zeitplan

> *Hinweis: Die gesamte Pipeline (Stage 1, Stage 2, API-Integration, Salienzmodell, kognitive Nutzermodelle) ist bereits vollständig implementiert. Die verbleibende Arbeit umfasst Datenerhebung, Modelltraining auf echten Daten und das Schreiben der Thesis.*

| Phase | Inhalt | Zeitraum |
|-------|--------|----------|
| Literatur | Tier 1+2 Paper lesen, Lücken schließen | Juni 2026 |
| Studie (Vorbereitung) | Rekrutierung, Ethikantrag, Stimulusmaterial, Pilottest | Juli 2026 |
| Studie (Erhebung) | Haupterhebung N≈30–35 | Aug 2026 |
| Auswertung | Eye-Tracking, NASA-TLX, Varianzdekomposition Stage 1 vs. Stage 2 | Sep 2026 |
| Modell-Training | Training auf echten Daten, Baseline-Vergleiche (B1–B3) | Okt 2026 |
| Schreiben | Thesis-Kapitel 1–7 verfassen | Nov 2026 |
| Abgabe | — | **Dez 2026** |

---

## 7. Eingrenzungen

Die Arbeit beschränkt sich auf automotive HMI-Screenshots. Der Cognitive Load Index ist ein berechneter Komplexitätsindikator — kein Ersatz für empirische NASA-TLX-Erhebungen. LLM-basierte Persönlichkeitssimulation (User Profile) ist auf stabile Trait-Level-Eigenschaften (Big Five) beschränkt; state-based Simulation (akuter Stress, Fatigue) wird als nicht-validiert deklariert.

---

*Schlüsselreferenzen: Oulasvirta et al. (2018, 2022), Das et al. (2024), Jiang et al. (2023), Jokinen et al. (2020), Langerak et al. (2024)*
