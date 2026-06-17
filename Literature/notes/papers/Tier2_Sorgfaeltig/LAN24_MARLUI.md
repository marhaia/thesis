# MARLUI: Multi-Agent Reinforcement Learning for Adaptive Point-and-Click UIs (2024)

**Autoren:** Thomas Langerak, Sammy Christen, Mert Albaba, Christoph Gebhardt, Christian Holz, Otmar Hilliges  
**Quelle:** ACM TOCHI 2024  
**DOI:** 10.1145/3661147  
**PDF:** `Tier2_Sorgfaeltig/LAN24_MARLUI.pdf`

---

## Kernfrage
Wie kann man UI-Adaptierung (welche Elemente anzeigen?) als Multi-Agent RL Problem formulieren — ohne hand-crafted Regeln oder echte Nutzerdaten für Training?

## Methode
MARLUI formuliert Online-UI-Adaptation als kooperatives Multi-Agent-RL-Problem. Zwei Agenten agieren turn-based in einer geteilten Umgebung (der adaptierten UI), ohne echte Nutzerdaten zu benötigen.

**User Agent (hierarchisch, 2 Ebenen):**
- *High-level Decision-Making Policy πd:* Wählt nächstes Target-Slot; State Sd = (p, m, x, g) — Position, UI-Encoding, aktueller Zustand, Zielzustand; Reward: Rd = α·Egd − (1−α)·(TD + TM) + 1_success; TD = Hick-Hyman-Entscheidungszeit (SDP-Modell); TM = Bewegungszeit
- *Low-level Motor Policy πm:* WHo-Modell (Fitts'-Law-basiert); gibt Endpoint-Verteilung (μp, σp) aus; σp = 1/6 Slot-Breite → 96% Hit-Rate; Position gesamplet aus p ∼ N(μp, σp)
- CR-Annahme [Ref 83] = OUL22: Nutzer verhält sich bounded-rational → MDP-Formulierung zulässig

**Interface Agent (flat RL Policy πI):**
- Observiert: (p, x, m, o) — Position, UI-Zustand, Item-Encoding, Interaktionshistory (Stack)
- Kennt NICHT das Ziel g des User Agents → POMDP-Setting
- Reward: RI = RD (direkt an User-Agent-Performance gekoppelt)
- Lernt implizite Verteilung über wahrscheinliche Nutzerziele durch Beobachtung der Interaktionssequenz

**Training:** Simultanes Training beider Agenten via PPO (RLLib); Curriculum Learning (Schwierigkeit steigt wenn Completion Rate >90%); ∼36h Training (Intel Xeon + NVIDIA TITAN Xp). **Keine echten Nutzerdaten benötigt.**

**Evaluation:** N=12 Probanden (ETH Zürich), Character-Creation-Task (5 Attribute × 3 Items = 243 Konfigurationen), 30 Trials/Condition (Latin Square), Oculus Quest 2. Baselines: Static (keine Adaptation) + SVM (trainiert auf 6 Pilotnutzern, >3000 Interaktionen, 91% Top-3-Accuracy).

## Wichtigste Ergebnisse
- **Actions:** MARLUI = 3.34 vs. Static = 5.73 vs. SVM = 3.87 Aktionen; F(2,22)=209.68, p<.001; Holm-korrigierte Post-hoc: MARLUI signifikant besser als SVM (p=0.006) und Static (p<.001)
- **Task Completion Time:** MARLUI = 12.14s vs. Static = 12.36s vs. SVM = 12.71s — **kein signifikanter Unterschied**: F(1.568,17.243)=0.86, p=0.42 → Nutzer bildeten Strategien mit der statischen Reihenfolge (Familiarity-Effekt)
- **Sim-to-Real Transfer:** Policies die ohne echte Nutzerdaten trainiert wurden, übertragen sich auf reale Nutzer in gleichen Tasks — MARLUI competitive mit SVM, das echte Nutzerdaten benötigt
- **Generalisierung:** 50% der Goals im Training genügen für vollständige Generalisierung (Figure 5); Framework anwendbar auf 5 verschiedene Interface-Typen (Toolbar, Keypad, Block Tower, Out-of-reach Grabbing, Hierarchical Menu)
- **Section 7.4:** *"familiarity is not captured by our current cognitive model"* — Limitation explizit benannt; direkte Verbindung zu TOD18 Familiarisierungs-Modell

## Direkt zitierbare Schlüsselsätze

> *"The presence of more items requires users to process more information and, thus, to evaluate more options, both of which lead to an increased cognitive load."* (Section 1)

> *"it does not adapt to the user themselves (e.g., different levels of expertise). Such personalization is an interesting direction for future research."* (Section 9)

> *"increasing realism in the model of the simulated user is an interesting future research direction, for instance, modeling human-like search or motor control with a biomechanical model."* (Section 9)

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
- OUL22 [Ref 83] = CR-Theorie: MARLUI baut explizit auf CR auf ("users behave rationally within their bounded resources [83]") — direkte Verbindung zur Thesis-Theoriebasis
- **AIM [Ref 81]** = OUL18 explizit zitiert: "Aalto interface metrics (AIM) a service and codebase for computational GUI evaluation" — LAN24 kennt AIM; Stage 1 ist direkte AIM-Pipeline-Erweiterung
- JOK21 [Ref 52] + JOK21 Typing [Ref 51] beide in LAN24 zitiert — bestätigt Jokinen-Gruppe als State-of-Art in CR-basierter User Modeling
- **Familiarity-Lücke → TOD18:** Section 7.4 benennt explizit: "familiarity is not captured by our current cognitive model" — MARLUI identifiziert genau die Lücke, die TOD18 für Webseiten-Familiarisierung schließt; für Stage 2 (experience_level) ist dieser Befund legitimierend
- Methodisch → CHE21, JOK21 (beide: CR-basierte User Agents ohne echte Nutzerdaten)
- Task Descriptor Kontext → KO26 (beide: explizite Task-Repräsentation vs. implizite Intent-Inferenz)
- Adaptive UI Tradition → SHI24 (supervisory control als verwandte Architektur)

---

**Tags:** #stage2 #RL #multi-agent #intent #adaptive-ui #task-descriptor #cognitive-load #sim-to-real #CR #TOCHI2024 #ETH-Zurich #WHo-model #Fitts #AIM-zitiert #OUL22-zitiert #JOK21-zitiert #familiarity-gap
