# Efficient Human-in-the-Loop Optimization via Priors Learned from User Models (2026)

**Autoren:** Yi-Chi Liao, João Marcelo Evangelista Belo, Hee-Seung Moon, Jürgen Steimle, Anna Maria Feit  
**Quelle:** CHI 2026, ACM  
**DOI:** 10.1145/3772318.3791976  
**PDF:** `Tier2_Sorgfaeltig/LIA26_HOMI.pdf`

---

## Kernfrage
Wie kann Human-in-the-Loop Optimierung durch Priors aus synthetischen User Models beschleunigt werden — ohne echte Nutzerdaten für das Training zu benötigen?

## Methode
- **HOMI Framework** (Human-in-the-Loop Optimization with Model-Informed Priors): 4 Schritte
  1. Model selection (z.B. Fitts' Law)
  2. Synthetic user generation (Parameter-Sampling)
  3. Meta-BO training mit synthetischen Interaktionen (offline)
  4. Human-in-the-loop deployment mit echten Nutzern (online)
- **NAF+** (Neural Acquisition Function+): Bayesian Optimization mit RL-trainierter Neural Acquisition Function
- Novelty Detector: erkennt out-of-distribution Nutzer → fällt auf Standard-EI zurück
- Validiert an VR Mid-Air Keyboard Adaptation (N=12)

## Wichtigste Ergebnisse
- NAF+ übertrifft TAF und ConBO in frühen Iterationen (Iteration 2–3 signifikant besser)
- **Kernthese Section 6:** *"HOMI repositions the role of user models from being optimization targets to training resources"* — direkt zitierbar
- Novelty Detector zeigt: reale Nutzer weichen von synthetischen ab → EI-Fallback nötig
- In-silico Training → In-situ Deployment funktioniert als valider Workflow

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★ marginal)
- Optional: Legitimierung des "simulate first, validate later" Paradigmas
- *"Liao et al. (2026) demonstrate that computational user models can serve as training resources for interface optimization rather than as fixed optimization targets — a paradigm that informs the present pipeline's use of predictive models prior to empirical validation."*

### Kapitel 2: Related Work (★★ relevant)
- Abschnitt "Computational Interaction + Empirical Validation", nach OUL22
- Liefert methodologischen Rahmen für deine Kombination aus computational Screening (Stage 1+2) und Nutzerstudie
- *"The HOMI framework (Liao et al., 2026) demonstrates the validity of bridging in-silico optimization — using synthetic users generated from computational models — with in-situ validation through real user studies. The present pipeline follows an analogous workflow: Stage 1/2 generates model-based predictions as prior estimates, which the subsequent user study validates against real gaze data."*
- ⚠️ Abgrenzung: LIA26 optimiert Interfaces zur Laufzeit (Bayesian Opt. + RL), du evaluierst vorab (kein Deployment-Loop, kein RL)

### Kapitel 5: Validierung (★★ relevant)
- Novelty Detector Logik → übertragbar auf dein Validierungsdesign
- Nutzer deren Gaze-Daten stark von Modellvorhersagen abweichen = "novel users" → müssen separat analysiert werden
- Argument: Kohärenz-Check + Nutzerstudie dienen als dein "EI-Fallback" für out-of-distribution Fälle

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Section 6 "Re-purposing user models" direkt zitierbar
- *"Liao et al. (2026) argue that user models should be treated not as passive analytic tools but as 'active enablers of interface optimization' — the present pipeline instantiates this principle by using computational rationality models to generate pre-deployment cognitive load estimates that guide design decisions prior to user testing."*

> **Was du lesen musst:** Abstract + Figure 1 (HOMI 4 Schritte) + **Section 6 Discussion** ("Re-purposing user models" + "Rethinking personalization") — ca. 20 Min.  
> Section 4 (NAF+ Technisches) und Appendix weglassen.

## Kritik / Offene Fragen
- LIA26 optimiert zur Laufzeit (RL + Bayesian Opt.) — deine Pipeline evaluiert vorab. Transfer ist konzeptuell, nicht methodisch direkt
- Domäne: VR Keyboard ≠ Automotive HMI
- "User models als Training Resources" Argument trägt am stärksten für Kap. 6

## Verbindungen zu anderen Papern
- Gleicher Erstautor → LIA25 (Affordance via CR) — beide Yi-Chi Liao
- Thematisch → OUL22 (CR als Grundlage für User Models)
- Methodisch → DAS24 (beide verbinden Simulation mit empirischer Validierung)
- Parallele → KANK17 (beide nutzen Priors aus Modellen für effizientere Inferenz)

---

**Tags:** #human-in-the-loop #optimization #user-models #in-silico #in-situ #priors #bayesian-optimization #meta-learning
