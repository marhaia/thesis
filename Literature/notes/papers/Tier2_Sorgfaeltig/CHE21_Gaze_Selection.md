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
- Fixationsanzahl und Dwell-Dauer emergieren als optimale Strategie aus RL — nicht hand-kodiert
- Repliziert 5 empirische Befunde: Target-Größe → Selektionszeit, Dwell-Zeit, Sakkadenzahl, Sakkadendauer, ID → Sakkadenzahl
- RMSE EMT = 29.35 ms, Selektionszeit = 54.50 ms (Figure 5, Vergleich mit Zhang 2010)
- **Key methodological quote (Section 3.3.1):** *"the agent is not trained on human data. Rather, it is trained on tasks sampled from an artificial environment"* — direkt zitierbar
- **Key design quote (Section 5):** *"Predictive models in HCI have traditionally had a role in guiding the evaluation of prototypes"* — direkt zitierbar für Kap. 1
- Kalibrierung auf einen einzigen Datenpunkt → Vorhersage aller anderen (Section 5)

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★ relevant)
- Section 5 Discussion hat die stärkste Motivations-Formulierung im gesamten Paper
- *"Chen et al. (2021) establish that computational models of gaze behavior have 'traditionally had a role in guiding the evaluation of prototypes' — the present pipeline extends this principle from gaze-selection to the cognitive load domain of in-vehicle HMI."*

### Kapitel 2: Related Work (★★ relevant)
- Bounded optimality + POMDP → emergente Fixationsstrategie → dein theoretisches Fundament für Stage 2
- "not trained on human data... trained on tasks sampled from an artificial environment" → methodisches Vorbild für deine Kombination aus Jokinen/HCEye (künstliche Umgebung) + Nutzerstudie
- *"Chen et al. (2021) demonstrate that fixation count and selection time emerge as optimal strategies under bounded rationality without training on human data — a methodological principle instantiated in the present pipeline, where Stage 2 predictions are derived from computational models prior to empirical user validation."*

### Kapitel 3: Methodik (★ marginal)
- POMDP Section 3.2: Fixationsanzahl ist direkte Funktion von target size + distance + motor noise → rechtfertigt Fixation Count als Stage-2-Feature
- Footnote-Niveau, kein langer Absatz nötig

### Kapitel 5: Validierung (★ relevant)
- "calibrated to just a single data point and then the others can be predicted" → Argument für warum N≈30 Nutzerstudie trotzdem generalisierende Aussagen erlaubt
- Model überprediziert Fixationen bei kleinen Targets (Section 4.3, Figure 6) → zeigt dass Computational Models systematische edge cases haben → dein Kohärenz-Check schließt diese Lücke

### Kapitel 6: Diskussion / Future Work (★ relevant)
- Section 5: "ability-based optimization... computationally improve designs for users with impairments" + ABC-Parameter-Fitting für Individuen → direkte Future-Work-Verbindung zu JIE24 (Personalisierung) und KANK17 (ABC)
- *"Chen et al. (2021) point toward ability-based optimization through individual parameter fitting as a future direction — the present pipeline's user study design (August 2026) lays the empirical foundation for such personalization in the automotive domain."*

> **Was du lesen musst:** Abstract + **Section 3.3.1** ("not trained on human data", 3 Min.) + **Section 5 Discussion** (2 Paragraphen, 5 Min.)  
> ⚠️ Table 1 zeigt nur Hyperparameter (motor noise, spatial noise, jitter) — NICHT Ergebnisse. Ergebnisse sind Figures 5–9.

## Kritik / Offene Fragen
- Fokus auf Gaze-Selektion (Dwell) ≠ freie visuelle Suche im Dashboard
- Kein Task-Kontext — reine Selektion, kein Fahrszenario
- Überprädiktion bei kleinen Targets (Section 4.3) — zeigt Grenzen des Ansatzes

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR Framework), JOK20 (Fixation Modelling)
- Ergänzt → SHI24 (supervisory control als nächster Schritt über reine Selektion hinaus)
- Future Work Parallele → JIE24 (ABC-Personalisierung), KANK17 (Parameter Inference)
- Methodisch → MIA26 (beide: kein Training auf echten Nutzerdaten, synthetische Umgebung)

---

**Tags:** #stage2 #RL #fixation #gaze #computational-model #CR #bounded-optimality #POMDP #pre-deployment
