# Computational Rationality as a Theory of Interaction (2022)

**Autoren:** Oulasvirta, A., Jokinen, J.P.P., & Howes, A.  
**Quelle:** CHI Conference on Human Factors in Computing Systems (CHI '22), New Orleans  
**DOI:** https://doi.org/10.1145/3491102.3517739  
**PDF:** OUL2022.pdf  
**ID:** OUL22 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Warum interagieren Menschen so wie sie es tun? — Kann Interaktion als **rationale Adaptation** an kognitive Grenzen und Umgebungsbedingungen erklärt werden?

## Methode
- Theoretisches Review-Paper (kein eigenes Experiment)
- Synthese des Frameworks der **Computational Rationality** (CR)
- Verknüpfung mit Reinforcement Learning, kognitiver Architektur, HCI-Modellen

## Wichtigste Ergebnisse
- Kernthese: Nutzer handeln optimal gegeben ihre kognitiven Grenzen (bounded rationality) — policy emerges from bounds
- Interaktion = Adaptation an: kognitive Architektur + Task-Umgebung + Gerätedesign
- CR ermöglicht es, Interaktion in computational models auszudrücken, die Verhalten erklären und vorhersagen können
- **Table 1 (Section 5):** Listet JOK20 [Ref 63] + TOD19 [Ref 118] als CR-Instanziierungen — deine Pipeline = nächster Eintrag, automotive domain
- **Key design quote (Section 1):** *"Design does not directly determine behavior but, rather, modifies the external environment of the user and thereby influences the actions that a rational user will take"* — deine Pipeline modelliert exakt diesen Schritt
- **Key pre-deployment quote (Section 6.2):** *"the designer can explore the consequences of hypothetical interventions only if it is possible to model and predict how the user will adapt to these changes before they are actually made"* — stärkste Pre-Deployment-Begründung in Tier 1
- **Key summary quote (Section 7):** *"people do what is best for them, given what they can do"* — kompakteste CR-Formel
- Section 4.2 Four Bounds: Time, Noise, Uncertainty, Capacity → Mapping auf Feature-Vektor ℝ¹⁹
- Section 3.6 Reward: "rewards are a function of internal states" → cognitive load ist interner Zustand, nicht direkt messbar → Stage 2 Regression löst das

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★★ zentral)
- Section 6.2 "What if" = stärkste theoretische Pre-Deployment-Rechtfertigung der gesamten Arbeit
- *"Oulasvirta et al. (2022) establish that 'the designer can explore the consequences of hypothetical interventions only if it is possible to model and predict how the user will adapt to these changes before they are actually made' — the present pipeline operationalises this principle for automotive HMI by providing pre-deployment cognitive load estimates from interface screenshots alone."*
- Section 1: "Design modifies external environment → influences rational user actions" → Screenshot = external environment representation

### Kapitel 2: Related Work (★★★ Hauptreferenz — Theoriefundament)
- Section 3 Theoretical Commitments (1–5) = Gerüst für den Kap. 2 Theorieabschnitt
- Figure 2 (POMDP internal/external environment) → Stage 1 = external→percept, Stage 2 = internal state→policy
- Section 3.6 Reward = "rewards are a function of internal states" → cognitive load ist interner Zustand → Stage 2 Regression approximiert ihn aus beobachtbaren Proxies
- Section 4.2 Modeling Bounds (Time, Noise, Uncertainty, Capacity) → dein Feature-Vektor bildet exakt diese vier Bound-Typen ab:
  - Time → search_time, task_duration
  - Noise → clutter, visual_complexity
  - Uncertainty → fixation_count, entropy
  - Capacity → cognitive_load_score (Zielgröße)
- Table 1: JOK20 + TOD19 explizit als CR-Instanziierungen gelistet → argumentative Linie zu deiner Pipeline

### Kapitel 3: Methodik (★★ relevant)
- Section 4.2 + Eq. 5 (Parameter Inference θ* = argmax p(y|M_θ,ϕ)) → das ist exakt was Ridge/RF/XGB in Stage 2 tut: optimale Parameter unter gegebenen Bounds finden
- Section 6.1 "Why?" questions = Parameter Inference → dein Modell beantwortet "warum ist dieses Interface kognitiv belastend?"
- Section 6.2 Eq. 6 (i* = argmax V(h)) = Counterfactual Design → deine Pipeline bewertet alternative Designs vor dem User Test

### Kapitel 5: Validierung (★★ relevant)
- Table 1 Validierungsstruktur: jedes Paper zeigt "policy adapts to bounds" → Vorlage für deine Hypothesenstruktur
- Section 6.2: Counterfactual reasoning als Validierungslogik → zeige dass Pipeline andere Outputs für verschiedene Task Descriptors produziert

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Section 7 "Computational design": *"Today's training times for computationally rational models are prohibitively long for large design spaces"* → deine Regression = die praktische Lösung für genau dieses Problem
- Section 7 Context 1 vs. 2: Task Descriptor = Context 1 (stabil, a priori definierbar) — ehrliche Limitation für dynamischen Context 2 (sich verändernder situativer Kontext)
- Section 7 "Adaptive and cooperative interfaces" → Future Work: deine Pipeline als Basis für adaptive Automotive HMI

> **Was du lesen musst:** Section 1 (letzter Absatz, 5 Min.) + **Section 3** (Theoretical Commitments + Figure 2, 15 Min.) + **Section 4.2** (Modeling Bounds, 10 Min.) + **Section 6.2** (What If + Eq. 6, 10 Min.) + **Section 7** (Research Agenda, 10 Min.)  
> ⚠️ Section 5 (Literature Review) = Table 1 überfliegen (JOK20 + TOD19 suchen) — Rest weglassen.

## Kritik / Offene Fragen
- Rein theoretisches Review-Paper — kein eigenes Experiment, keine neuen Daten
- Section 7: "Training times prohibitively long for large design spaces" → deine Regression ist die Lösung, aber das bedeutet auch: du verlässt die strenge CR-Formalisierung
- Reward-Funktion bleibt subjektiv definiert — wie genau TLX-Scores auf CR-Utility mappen, ist nicht trivial
- Context 2 (dynamisch, sozial konstruiert) = explizite Limitation des gesamten Frameworks, nicht nur deiner Pipeline

## Verbindungen zu anderen Papern
- Baut auf → Card, Moran & Newell 1983 [Ref 19] (GOMS — erwähnt im Paper), Lewis et al. 2014 [Ref 74] (CR Definition)
- Explizit zitiert als CR-Instanziierungen → JOK20 [Ref 63] (Adaptive Feature Guidance), TOD19 [Ref 118] (Individualising Layouts)
- Konkretisiert durch → JOK20 (CR in Praxis als Search Time Modell), SHI24 (CR für Typing)
- Kontextualisiert durch → OUL18 (AIM: das System, das CR noch nicht implementiert)
- Affordanz-Theoriebasis → LIA25 (baut explizit auf OUL22 auf, Ref [60] im LIA25-Paper)
- Automotive CR → JOK21 (Drivers Adaptation, direkte Anwendung auf Fahrzeugkontext)
- KANK17 [Ref 65] = Parameter Inference Methodik (ABC) → in Table 1 referenziert

---

**Tags:** #computational-rationality #theory #cognitive-bounds #expected-utility #HCI-theory #adaptation #cognitive-modeling #POMDP #RL #bounds #pre-deployment #CHI22
