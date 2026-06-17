# Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction (2024)

**Autoren:** Das, A., Wu, Z., Škrjanec, I., & Feit, A.M.  
**Quelle:** Proc. ACM Hum.-Comput. Interact., Vol. 8, No. ETRA, Article 236 — ETRA '24, Mai 2024  
**DOI:** 10.1145/3655610  
**PDF:** DAS_2024.pdf  
**ID:** DAS24 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Wie beeinflusst hohe kognitive Last (durch Dual-Task) die visuelle Aufmerksamkeit auf UI-Elemente — und wie gut sagen aktuelle Saliency-Modelle dieses veränderte Verhalten voraus?

## Methode
- Eye-Tracking-Studie mit **27 Teilnehmern**
- **150 einzigartige Webseiten** als Stimuli
- Dual-Task-Paradigma zur Induktion kognitiver Last
- Zwei Arten von Highlighting: permanent und dynamisch
- Analyse: Gaze-Daten, Saliency-Model-Vergleich (State-of-the-Art Modelle)
- Datensatz: offen verfügbar (HCEye Dataset)

## Wichtigste Ergebnisse
- Hohe kognitive Last → **„Tunnel Vision"**: Fixation Count ↓ (z=3.732, p<0.001), Fixation Duration ↑, Anzahl explorierter Regionen ↓
- Dynamisches Highlighting bleibt auch unter High CL aufmerksamkeitsstark — kein signifikanter Unterschied in Time-to-First-Fixation zwischen Absent CL und High CL bei Dynamic HT
- Kognitive Last **verändert fundamental was als „salient" gilt** — Standard-Saliency (AIM/SALICON) ≠ Load-adjustiertes Saliency
- Fine-tuned CL-Modell vs. SALICON-Baseline: CC +26% (High CL: 0.349→0.440), NSS +8% (1.301→1.411)
- **Key motivation quote (Section 2.2):** *"As our focus on a task intensifies, 'tunnel vision' may occur, impeding our ability to perceive changes in peripheral vision"* — direkt zitierbar für Kap. 1
- **Key gap quote (Abstract):** *"The presence of these factors significantly alters what people attend to and thus what is salient"* — empirische Rechtfertigung für CL-Integration in Stage 2
- **Key pre-deployment quote (Section 6.2):** *"predictive models of visual attention could serve to help designers... assess the impact on user attention already at design time"* — wörtlich dein Pipeline-Versprechen
- **Key future-work quote (Section 6.1):** *"the need for saliency models to explicitly incorporate the specific cognitive state of the viewer, to achieve more accurate and robust saliency predictions"* — Lücke, die dein Task Descriptor schließt
- ⚠️ Section 2.3: zitiert OUL18/AIM [Ref 36] + JIA23/UEyes [Ref 18] explizit → DAS24 verknüpft deinen Stack methodisch

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★★ zentral)
- Section 2.2 Tunnel-Vision-Quote = stärkste empirische Motivation für Load-Kontext in Automotive HMI in deinem gesamten Set
- Section 6.2 Pre-Deployment-Quote = wörtlich dein Pipeline-Versprechen → direkt im Einleitungsabsatz einsetzbar
- *"Das et al. (2024) demonstrate empirically that cognitive load fundamentally alters what users attend to in interfaces — the present pipeline operationalises this insight by incorporating task context (via Task Descriptor) into pre-deployment estimates, without requiring observed user behaviour."*

### Kapitel 2: Related Work (★★★ Hauptreferenz für CL-Saliency-Gap)
- Section 2.2: General Interference Model + Tunnel Vision = theoretische Basis warum `attention_demand` und `cognitive_load_score` separate Outputs sein müssen
- Section 2.3: OUL18/AIM [Ref 36] + JIA23/UEyes [Ref 18] explizit zitiert → DAS24 positioniert deinen Stack im Forschungskontext
- Section 3.2: Feature Congestion (FC Score, Oulasvirta et al.) als Stimulus-Selektionskriterium → selbe Metrik wie Stage 1 clutter-Feature — methodische Kontinuität explizit nennen
- Abstract: *"The presence of these factors significantly alters what people attend to and thus what is salient"* → empirische Rechtfertigung dass AIM-Saliency allein (ohne CL-Kontext) unzureichend ist
- ⚠️ Abgrenzung: DAS24 beschreibt das Problem empirisch (Webseiten, gemessene CL) — deine Pipeline liefert die Lösung (CL als a-priori Input via Task Descriptor, kein gemessenes Nutzerverhalten nötig)

### Kapitel 3: Methodik (★★ relevant)
- Section 3.2: FC Score als Stimulus-Selektionskriterium validiert `feature_congestion` als Feature in Stage 1
- Section 4.1 GLMM-Spezifikation: Random Intercepts by participant + image + block index, Random Slopes für CL + HT by participant → exaktes Template für LMM-Analyse August 2026 (N≈30–35)
- ⚠️ Abgrenzung: DAS24 löst CL durch condition-spezifisches Finetuning (getrennte Modelle pro Load-Level) — dein Ansatz: CL als Eingabeparameter (Task Descriptor), kein Retraining → explizit als Design-Entscheidung begründen

### Kapitel 5: Validierung (★★ relevant)
- Table 4: High-CL Fine-tuned vs. SALICON: CC 0.349→0.440 (+26%), NSS 1.301→1.411 (+8%) → Format für deine Verbesserungs-Tabellen
- Section 4.2: Fixation Count ↓ + Duration ↑ unter High CL = validierbares Muster: deine Pipeline sollte bei hohem `cognitive_load_score` längere `search_time`-Schätzungen ausgeben
- N=27, 45 Min., within-subjects → Größenordnung deiner Studie realistisch

### Kapitel 6: Diskussion / Future Work (★★★ zentral)
- Section 6.1: *"the need for saliency models to explicitly incorporate the specific cognitive state of the viewer"* → Future-Work-Lücke, die dein Task Descriptor schließt — stärkster Gap-to-Solution-Satz in DAS24
- Section 6.2: *"assess the impact on user attention already at design time"* = wörtlich dein Beitrag → Kap. 6 Schlussabsatz
- Limitation Section 6.2: nur Webseiten, kein Automotive, kleiner N → deine Arbeit erweitert genau das auf automotive HMI
- Limitation: CL nur experimentell induziert (counting backwards) — deine Pipeline schätzt CL ohne Experiment aus Screenshot + Task → genuiner Fortschritt

> **Was du lesen musst:** Abstract + Section 1 (10 Min.) + **Section 2.2** (Tunnel Vision, 5 Min.) + **Section 4.2** (Hauptbefunde, 10 Min.) + **Section 6.1–6.2** (Discussion, 10 Min.)  
> ⚠️ Section 3 (Studie) + Section 5 (Saliency Models) = überfliegen — Table 4 (CL-Modell-Vergleich) ist das Einzige was du in Kap. 5 brauchst.

## Kritik / Offene Fragen
- Fokussiert auf Highlighting — kein allgemeines UI-Suchmodell; Tunnel-Vision-Befund ist aber unabhängig vom Highlighting robust
- Webseiten als Stimuli → Automotive-HMI explizit als Transfer-Limitation nennen (Section 6.2 fordert selbst mehr UI-Varietät)
- CL nur durch Counting-Backward induziert — kein realer Fahrkontext; dein Task Descriptor geht darüber hinaus
- N=27, 13 Nationalitäten, aber enge Altersspanne (20–37) → demographische Limitation

## Verbindungen zu anderen Papern
- Ergänzt → OUL18 (AIM): DAS24 zeigt empirisch die Lücke in AIM's statischer Saliency; Section 2.3 zitiert OUL18 [Ref 36] explizit
- Ergänzt → JOK20: Tunnel Vision verlängert Search Time — JOK20 modelliert Search Time, DAS24 erklärt warum sie unter Last steigt
- Verknüpft mit → JIA23/UEyes: Section 2.3 zitiert UEyes [Ref 18] als Prior-Work; wenn Load Saliency verändert, ist UEyes-Benchmark nur unter Absent-CL valide
- Automotive-Analogie → JOK21: Tunnel Vision in Driving-Kontext vgl. Lee et al. 2023 [Ref 25] (DAS24 selbst zitiert)
- Methodisch → TOD19/LMM: GLMM-Struktur identisch mit TOD19 und JOK20 → konsistente Analysemethode

---

**Tags:** #cognitive-load #saliency #eye-tracking #dual-task #tunnel-vision #highlighting #HCEye #load-adjusted-saliency #AIM #ETRA2024 #pre-deployment #task-descriptor
