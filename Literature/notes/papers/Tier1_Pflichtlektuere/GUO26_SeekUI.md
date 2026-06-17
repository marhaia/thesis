# SeekUI: Predicting Visual Search Behavior on GUIs with a Reward-Augmented Vision Language Model (2026)

**Autoren:** Guo, Z., Jiang, Y., Leiva, L.A., Oulasvirta, A. et al.  
**Quelle:** Proc. ACM CHI '26, Barcelona, Spain, April 13–17, 2026 — 23 pages  
**DOI:** 10.1145/3772318.3791178  
**PDF:** GUO2026.pdf  
**ID:** GUO26 | **Status:** 🟢 Analyzed (Missing Rabbit: KEEP — Stage 2/SOTA)  

---

## Kernfrage
Kann ein Reward-augmented Vision Language Model (VLM) visuelle Suchpfade auf GUIs vorhersagen — aus einem Screenshot + einem Textziel (Task Descriptor)?

## Methode
- Kombination: Vision Language Model + Reward-Augmentation
- Input: GUI-Screenshot + Text-Cue (Suchziel / Task Descriptor)
- Output: vorhergesagter Scanpfad (Sequenz von Fixationen)
- Validierung gegen menschliche Gaze-Daten

## Methode
- **SeekUI**: Qwen2.5-VL + Instruction Tuning (Stage 1) + Reinforcement Learning mit ScanMatch-Reward (Stage 2)
- Input: GUI-Screenshot + Text-Cue (Suchziel) → Output: Explanation + Scanpath (Fixationssequenz)
- Dataset: VSGUI10K [Putkonen et al. 2025], N=84 Teilnehmer, 730 GUIs (Mobile/Desktop/Web), 1.616 Scanpaths
- Training-Split: 85/15 (1.366 Scanpaths Training, 250 Test)
- Vergleich gegen: EyeFormer++ (Free-Viewing), GazeFormer (Natural Images), Chen et al. (Natural Images)

## Wichtigste Ergebnisse
- **ScanMatch**: SeekUI 0.347 vs. best baseline 0.236 (GazeFormer) → **+47% Verbesserung**
- **NSS**: SeekUI 1.394 vs. GazeFormer 0.659 → **mehr als verdoppelt**
- **Success Rate**: SeekUI **62%** vs. Human 70% vs. GazeFormer 14% / EyeFormer++ 10%
- **Guess–Scan–Confirm Strategy** reproduziert: Upper-Left-Bias in Guess-Phase (Correlation >0.85), systematisches Scanning, Confirm-Phase CC >0.89 in allen Quadranten
- Clutter → Search Time Korrelation korrekt reproduziert (MSE=10.437 vs. Chen et al. MSE=93.490)
- ⚠️ Kein kognitives Modell dahinter: rein statistisch gelernte Regularitäten
- **Key explainability-gap quote (Section 7.4.5):** *"it lacks intrinsic explanatory power about human cognitive architecture"* — deine Pipeline hat diese Erklärbarkeit
- **Key application quote (Section 6.1):** *"designers can reposition components and immediately generate predicted scanpaths... without requiring eye-tracking hardware or repeated user studies"* — wörtlich dein Pre-Deployment-Versprechen
- **Key positioning quote (Section 7.4.5):** *"future work could explore hybrid approaches that combine VLM semantic reasoning with biologically constrained cognitive models"* — dein Ansatz ist genau diese Hybridisierung (CR-Theorie + ML)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 1: Einleitung (★★★ zentral — SOTA-Kontextualisierung)
- Section 6.1: *"designers can reposition components and immediately generate predicted scanpaths... without requiring eye-tracking hardware or repeated user studies"* → wörtlich dein Pre-Deployment-Versprechen → direkt im ersten Absatz einsetzbar
- GUO26 zeigt: das Feld bewegt sich 2026 zu task-conditioned, screenshot-basierten Modellen → deine Pipeline ist in diesem Moment relevant

### Kapitel 2: Related Work / Stand der Forschung (★★★ Hauptreferenz — SOTA-Abgrenzung)
- GUO26 als aktuellster SOTA für task-conditioned scanpath prediction auf GUIs
- Guess–Scan–Confirm Strategy: Section 5.3.5 = direkte empirische Bestätigung für JOK20's Novice→Expert-Transition im Kontext von Suchstrategien → Brücke JOK20 → GUO26
- Direkte Abgrenzung: *"SeekUI (Guo et al., 2026) demonstriert, dass die Kopplung von visuellem GUI-Input mit einem textbasierten Task Descriptor ausreicht, um visuelle Suchpfade über einem menschlichen Baseline zu prädizieren — der vorliegende Ansatz teilt diese Grundannahme, unterscheidet sich jedoch fundamental im Ziel: anstelle von Scanpath-Accuracy steht die Schätzung des Cognitive Load Index als interpretierbare Designmetrik."*
- ⚠️ Abgrenzung: GUO26 = Black-Box VLM (keine kognitive Fundierung, kein CR-Theorie-Bezug) → dein Ansatz = interpretierbar + CR-fundiert

### Kapitel 3: Methodik (★★ relevant — Feature-Validierung)
- Section 5.3.4: Clutter (Rosenholtz) → Fixation Count Korrelation direkt belegt → validiert `feature_congestion` in Stage 1
- Guess–Scan–Confirm + Upper-Left-Bias (Section 5.3.5) → validiert `attention_demand`-Output: UI-Designs ohne klare Hierarchie erzwingen mehr Scanning
- Section 7.4.5: *"future work could explore hybrid approaches that combine VLM semantic reasoning with biologically constrained cognitive models"* — dein Ansatz ist diese Hybridisierung

### Kapitel 5: Validierung (★★ relevant)
- GUO26 operationalisiert search efficiency als kürzester Pfad von Fixation 1 zum Ziel (Section 5.3.1: "last three fixations within foveal radius")
- Nach August 2026 Studie: dieselbe Metrik aus Gaze-Daten berechnen → Spearman-Korrelation gegen `search_efficiency`-Score → externes SOTA-Validierungsargument
- Table 1 Metriken (ScanMatch, SED, SS, MultiMatch, CC, AUC, NSS) → vollständigster Metrik-Stack in deinem Paper-Set, als Referenz für Kap. 5

### Kapitel 6: Diskussion (★★★ zentral — Positionierung)
- Section 7.4.5: *"it lacks intrinsic explanatory power about human cognitive architecture"* → stärkster Abgrenzungssatz im gesamten Paper-Set
- Positionierungsabsatz (direkt verwendbar):
  > *"SeekUI (Guo et al., 2026) erreicht state-of-the-art Scanpath-Accuracy durch ein Reward-augmented Vision Language Model, verzichtet jedoch auf interpretierbare Ausgaben und kognitive Fundierung. Der vorliegende Ansatz verfolgt ein komplementäres Ziel: die Kopplung von task-conditioned visual search mit einem Cognitive Load Index und Coherence Constraints (Jokinen et al., 2020; Das et al., 2024), die gezielt für Design-Feedback konzipiert sind."*
- Limitation: *"Im Gegensatz zu Black-Box-Ansätzen wie SeekUI (Guo et al., 2026) verzichtet der vorliegende Ansatz auf Deep Learning zugunsten von Interpretierbarkeit — auf Kosten potentiell geringerer Scanpath-Accuracy."*

> **Was du lesen musst:** Abstract + Section 1 (10 Min.) + **Section 5.3.5** (Guess–Scan–Confirm, 5 Min.) + **Section 6.1–6.2** (Applications, 10 Min.) + **Section 7.4.5** (Limitations, 5 Min.)  
> ⚠️ Section 3 (Architektur-Details) + Section 5.2 (Tabellen) = überfliegen — Table 1 Gesamtergebnis-Zeile reicht

## Kritik / Offene Fragen
- VLM/Reward-Architektur nicht portierbar auf deinen Ansatz — kein RL, kein VLM, kein Transformer
- Section 7.4.5 explizit: *"it lacks intrinsic explanatory power about human cognitive architecture"* → das ist dein Wettbewerbsvorteil
- GUO26 braucht Ground-Truth Scanpaths (VSGUI10K) für Training — deine Pipeline braucht keine Nutzerdaten
- Kein Automotive HMI im VSGUI10K-Dataset (Mobile/Desktop/Web) → gleiche Limitation wie JIA23
- Section 7.4.2: Individual differences nicht modelliert → dein Task Descriptor adressiert genau das

## Verbindungen zu anderen Papern
- Gleiche Autoren-Gruppe → JIA23 (UEyes): Jiang, Leiva, Oulasvirta; GUO26 = direkte Weiterentwicklung von JIA23
- Ergänzt/konkurriert → JOK20: beide sagen Search Behavior vorher; JOK20 = kognitive Basis, GUO26 = SOTA-Vergleich
- Kontextualisiert → OUL18 (AIM): zeigt wohin sich das Feld seit AIM 2018 bewegt hat
- Validierungs-Brücke → DAS24: GUO26-Success-Rate-Metrik kann gegen HCEye-CL-Scores validiert werden
- Section 7.4.5 "hybrid approaches" → OUL22: dein CR-fundierter Ansatz ist diese Hybridisierung

---

**Tags:** #SOTA #scanpath #task-descriptor #VLM #visual-search #stage2 #reward-augmentation #CHI2026 #guess-scan-confirm #interpretability #cognitive-architecture #pre-deployment
