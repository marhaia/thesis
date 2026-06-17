# MARLUI: Multi-Agent Reinforcement Learning for Adaptive Point-and-Click UIs (2024)

**Autoren:** Thomas Langerak, Sammy Christen, Mert Albaba, Christoph Gebhardt, Christian Holz, Otmar Hilliges  
**Quelle:** ACM TOCHI 2024  
**DOI:** 10.1145/3661147  
**PDF:** `Tier2_Sorgfaeltig/LAN24_MARLUI.pdf`

---

## Kernfrage
Wie kann man UI-Adaptierung (welche Elemente anzeigen?) als Multi-Agent RL Problem formulieren — ohne hand-crafted Regeln oder echte Nutzerdaten für Training?

## Methode
- Zwei Agenten: User-Agent (simuliert Nutzungsverhalten via point-and-click) + Interface-Agent (lernt Adaptierungspolitik)
- Turn-based Multi-Agent RL in geteilter Umgebung
- Trainiert ohne echte Nutzerdaten — Transfer auf reale Nutzer im gleichen Task

## Wichtigste Ergebnisse
- Interface-Agent lernt relevante Elemente anzuzeigen durch Beobachtung des User-Agenten
- Überträgt erfolgreich auf reale Nutzer in gleichen Tasks
- Ohne Heuristiken oder domain-spezifische Regeln generalisierbar
- Kernproblem: Intent-Inferenz (Was will der Nutzer?) als RL-gelöste Herausforderung

## Wichtigste Ergebnisse
- Interface-Agent lernt relevante Elemente anzuzeigen durch Beobachtung des User-Agenten — ohne Heuristiken oder Nutzerdaten
- Sim-to-real Transfer erfolgreich: N=12 Teilnehmer, signifikant weniger Actions als statische Baseline (3.34 vs. 5.73, p<.001) — aber kein signifikanter Unterschied in Task Completion Time
- **Key quote (Section 2.2):** *"More recent work extends these models and, for instance, predicts... cognitive load"* [Ref 24/32] — LAN24 zitiert CL-Modellierung als verwandtes Feld, Ref [32] = Duchowski 2018 (Pupillary Activity als CL-Maß) = dein Feld
- Section 1: *"The presence of more items requires users to process more information... lead to an increased cognitive load"* — direkt zitierbar für UI-Komplexität → CL-Verbindung
- Kernproblem: Intent-Inferenz als RL-gelöste Herausforderung — du löst es anders (expliziter Task Descriptor)

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★ marginal)
- Section 1: UI-Komplexität → Cognitive Load als 1-Satz-Kontext
- *"As the number of interface elements increases, users face greater cognitive demands (Langerak et al., 2024) — a relationship the present pipeline quantifies computationally for pre-deployment HMI evaluation."*

### Kapitel 2: Related Work (★★ relevant)
- Section 2.2: MARLUI ordnet sich in Tradition computationaler User Models ein die Cognitive Load vorhersagen → deine Pipeline = nächste Stufe (nicht nur vorhersagen, sondern evaluieren vor Deployment)
- Intent-Inferenz ohne Nutzerdaten = methodisches Vorbild für Task Descriptor — aber gegensätzliche Strategie
- *"Where MARLUI (Langerak et al., 2024) implicitly infers user intent from interaction patterns without real user data, the present pipeline adopts an alternative strategy: intent is explicitly encoded via the Task Descriptor, trading inference flexibility for interpretability and determinism — a deliberate design choice for pre-deployment evaluation."*
- ⚠️ Abgrenzung: LAN24 = online, reactive, adaptive (Laufzeit); deine Pipeline = offline, static, evaluative (vor Deployment)

### Kapitel 3: Methodik (★ relevant)
- Section 9 Limitations: *"it does not adapt to the user themselves (e.g., different levels of expertise)"* → dein Task Descriptor löst exakt dieses Problem durch explizites User-Profile Encoding (SEARCH_MODE_WEIGHTS, TIME_PRESSURE_WEIGHTS)
- User Agent: CR-bounds für point-and-click → selbes theoretisches Fundament wie Stage 1 (Jokinen-Modell als CR-basierter User Agent)

### Kapitel 6: Diskussion / Future Work (★ marginal)
- Section 9: "increasing realism... modeling human-like search or motor control with a biomechanical model" → dein Jokinen/HCEye-Modell = dieser Schritt für die visuelle Suchdimension bereits realisiert

> **Was du lesen musst:** Abstract + **Section 1** (Motivation CL, 3 Min.) + **Section 2.2** (Computational User Models, 5 Min.) + **Section 9** (Limitations/Future Work, 5 Min.)  
> ⚠️ Section 4 ("Background") = POMDP/RL Theorie — weglassen. Evaluation ist Section 7.

## Kritik / Offene Fragen
- Point-and-click Paradigma — kein visueller Suchkontext, kein Fahrszenario
- Kein Cognitive Load Modell — reine Interaktionsoptimierung
- Intent-Inferenz = binär/sequentiell; dein Task Descriptor ist dimensionaler (5 Kategorien)
- Kein signifikanter Unterschied in Task Completion Time trotz weniger Actions → zeigt dass reine Action-Reduktion nicht ausreicht

## Verbindungen zu anderen Papern
- Methodisch → CHE21, JOK21 (beide: CR-basierte User Agents ohne echte Nutzerdaten)
- Task Descriptor Kontext → KO26 (beide: explizite Task-Repräsentation vs. implizite Intent-Inferenz)
- CL-Modellierung Verweis → KRE16 (Duchowski Ref [32] im Paper = Pupillary Activity als CL-Maß)
- Adaptive UI Tradition → SHI24 (supervisory control als verwandte Architektur)

---

**Tags:** #stage2 #RL #multi-agent #intent #adaptive-ui #task-descriptor #cognitive-load #sim-to-real #CR
