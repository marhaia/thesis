# Modelling Drivers' Adaptation to Assistance Systems (2021)

**Autoren:** Jokinen, J.P.P., & Kujala, T.  
**Quelle:** AutomotiveUI 2021, ACM (Woodstock '18 Proceedings)  
**DOI:** 10.1145/3409118.3475150  
**PDF:** JOK21.pdf (auch als JOK.pdf — Duplikat im Ordner)  
**ID:** JOK21 | **Status:** 🟢 Analyzed  

> ⚠️ **Hinweis:** JOK.pdf und JOK21.pdf sind identisch — gleicher Inhalt, doppelt abgelegt.

---

## Kernfrage
Wie adaptieren Fahrer ihre Interaktionsstrategien wenn ein Fahrassistenzsystem (ADAS) eingeführt wird — und kann dieses Adaptationsverhalten computational modelliert werden?

## Methode
- Simulationsmodell basierend auf **Computational Rationality**
- Task-Interleaving: Modellierung wie Fahrer zwischen Primärtask (Fahren) und ADAS-Interaktion wechseln
- Reinforcement Learning Grundlage

## Wichtigste Ergebnisse
- Mit zunehmendem UI-Elementanzahl (4→12 Items) steigt Glance-Dauer UND Lane-Excursion-Wahrscheinlichkeit (Figure 2 + 4) — direktes Ablenkungsrisiko durch UI-Komplexität
- Modell mit Lane-Assist-Adaptation: signifikant mehr Lane Excursions wenn Assist plötzlich ausfällt (Figure 4)
- **Key design quote (Section 4):** *"predicting how humans adapt their goal-oriented behaviour to changes in the task environment is far from trivial"* — rechtfertigt Computational Pipeline statt simpler Guidelines
- **Key motivation quote (Section 1):** *"design and testing would benefit immensely from well-established model-based methodologies"* — direkt zitierbar für Kap. 1
- Supervisory Control Modell: Supervisor observiert Q-Values beider Subtasks → entscheidet Attention-Allocation → emergiert als RL-Policy (keine Hand-Kodierung)

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★ relevant)
- Section 1 liefert den stärksten Automotive-Motivationssatz im gesamten Tier-2-Set
- *"Jokinen & Kujala (2021) argue that HMI design 'would benefit immensely from well-established model-based methodologies' for predicting driver adaptation — the present pipeline operationalizes this vision by providing pre-deployment cognitive load estimates before empirical user studies."*

### Kapitel 2: Related Work (★★★ zentral)
- **Wichtigstes Automotive-Computational-Paper in deinem Set** — liefert Domänen-Validierung und theoretisches Fundament gleichzeitig
- Figure 4: UI-Komplexität (Elementanzahl) → Lane Excursion Wahrscheinlichkeit — das ist dein kognitiver Load Argument in Zahlen
- Supervisory Control = theoretischer Vorläufer für SHI24 und für deine Stage-2-Architektur
- *"Jokinen & Kujala (2021) demonstrate computationally that increasing in-vehicle UI complexity — measured as the number of interface elements — directly increases lane excursion probability, corroborating the theoretical motivation for automated cognitive load estimation prior to deployment."*
- ⚠️ Abgrenzung: JOK21 modelliert Adaptation über Zeit (RL-Training), du evaluierst einen statischen Screenshot — kein dynamischer Adaptationsprozess

### Kapitel 3: Methodik (★★ relevant)
- State space des Modells: lateral road position + UI-Elemente + gaze position → das sind exakt deine Feature-Kategorien (visual complexity, element count, task type)
- Reward function: Lane Excursion = negativer Reward → dein cognitive_load_score ist die prädiktive Version dieser Metrik
- *"Following the modelling tradition of Jokinen & Kujala (2021), the present pipeline encodes task environment properties — UI element count, visual complexity, task type — as state variables from which cognitive load outcomes are predicted."*

### Kapitel 6: Diskussion / Future Work (★ relevant)
- Section 4 letzter Absatz: "transparent adaptive automation, where the driver's situational needs and limitations are recognised" → deine Pipeline = Schritt in diese Richtung (statische Evaluation als Vorstufe)
- "models of human adaptation can be used in tandem with algorithms for adaptive automation" → Future Work: deine Pipeline + Echtzeit-Adaptation (HOMI-Logik aus LIA26)

> **Was du lesen musst:** Abstract + **Figure 2 + Figure 4** (Kernbefunde, 5 Min.) + **Section 1 Motivation** (2 Paragraphen, 5 Min.) + Section 4 Discussion letzter Absatz (3 Min.)  
> Section 2 (MDP Formalism) weglassen — kennst du aus JOK20.

## Kritik / Offene Fragen
- Keine Validierung gegen echte Fahrerdaten in diesem Paper (nur Modell-Predictions) — Section 4 gibt das explizit zu
- Fokus auf Lane-Keeping + Visual Search; kein Touch-Interaction-Szenario
- UI-Complexity = nur Elementanzahl (4/6/9/12) — nicht Salienz, Farbe, Hierarchie wie in Stage 1

## Verbindungen zu anderen Papern
- Baut auf → JOK20 (Multitasking Driving, selber Erstautor, validiertes Basismodell)
- Baut auf → OUL22 (CR Framework als theoretische Grundlage)
- Direkte Domänen-Verbindung → LOR24 (In-Vehicle UI Review, Automotive-Kontext)
- Supervisory Control Vorläufer → SHI24 (CRTypist, selbe Architektur)
- Future Work Parallelarbeit → LIA26 (HOMI: Adaptation + Computational Models)

---

**Tags:** #automotive #ADAS #supervisory-control #computational-rationality #adaptation #driving #cognitive-load #UI-complexity #lane-excursion #pre-deployment
