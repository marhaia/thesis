# Simulating Emotions With an Integrated Computational Model of Appraisal and Reinforcement Learning (2024)

**Autoren:** Jiayi Eurus Zhang, Joost Broekens, Bernard Hilpert, Jussi P.P. Jokinen  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3641908  
**PDF:** `Tier2_Sorgfaeltig/ZHA24_Simulating_Emotions.pdf`

---

## Kernfrage
Wie kann man emotionale Zustände (Freude, Langeweile, Frustration) während der Interaktion computational modellieren — durch Integration von Appraisal-Theorie und RL?

## Methode
ZHA24 adaptiert das RL-Appraisal-Modell von [Ref 54] (Broekens et al. 2021, erstmals nur an Vignetten evaluiert) auf reale interaktive HCI-Aufgaben. Die Architektur kombiniert zwei Theorieebenen in einem einheitlichen Framework.

**Formales Interaktionsmodell:** Jede Interaktionsepisode wird als MDP (Markov Decision Process) formalisiert: Zustandsmenge S, Aktionsmenge A, Übergangsfunktion T(s,a,s'), Reward-Funktion R(s,a,s'), Diskontfaktor γ. TD/Q-Learning aktualisiert Value-Schätzungen: Q(s,a) ← Q(s,a) + α[R(s,a) + γ·max_a'Q(s',a') − Q(s,a)].

**4 Appraisal-Dimensionen (Section 3.2):**
- **Suddenness:** S = 1 − n̂(s,a,s') / Σ n̂(s,a,s'') — wie unerwartet ist ein Zustandsübergang? Gelernt über Weltmodell n̂
- **Goal Relevance:** GR = min(1, |ΔTD|) — Magnitude des TD-Errors als Proxy für Zielrelevanz
- **Conduciveness:** C = σ(Δ,−1)×0.5 + 0.5 — Richtung+Magnitude der TD-Abweichung, normalisiert auf [0,1]; 0.5 = neutral, >0.5 = konduktiv, <0.5 = obstruktiv
- **Power:** ∝ |max_a'Q(s') − avg_a'Q(s')| — Fähigkeit des Agenten, gute von schlechten Aktionen zu unterscheiden

**Klassifikator:** Linearer SVM mappt Appraisal-Vektor → modal emotion; Table 1: Happiness = low Suddenness / medium GR / high Conduciveness; Boredom = very low Suddenness / low GR / open Conduciveness; Irritation = low Suddenness / medium GR / obstructive Conduciveness / medium Power.

**Experiment:** Proband löst interaktive Computing-Aufgabe (einfaches Problemlösen am Computer); self-reports nach Aufgabenende; Modell-Vorhersagen werden gegen menschliche Selbstberichte verglichen (Figure 1). Jokinen als Co-Autor = direkte Jyväskylä-Lab-Linie (OUL22/JOK20/TOD18/ZHA24).

## Wichtigste Ergebnisse
- **Modell-Human-Alignment:** Vorhersagen stimmen mit self-reported Emotionen überein (Figure 1): bei einem einfachen, belohnenden Task → Happiness dominant, schwache Boredom (wegen geringer Goal Relevance durch Simplizität), keine Irritation — Modell trifft genau dieses Profil
- **Happiness:** Entsteht aus hoher Conduciveness (positive TD-Updates) + mittlerer Goal Relevance — erfahrene, erfolgreiche Aktionen mit klaren Zielen
- **Boredom:** Entsteht aus sehr niedriger Suddenness und niedriger Goal Relevance — Routineaufgaben mit vorhersehbaren Outcomes; niedrige TD-Error-Magnitude trotz positiver Outcomes
- **Irritation:** Obstruktive Conduciveness bei mittlerer Goal Relevance und mittlerem Power — Hindernisse bei wichtigen Zielen, aber mit gewissem Handlungsspielraum; Power-Appraisal erklärt, warum dieselbe Error-Message bei Novizen Angst, bei Experten Irritation auslöst
- **Hauptbeitrag:** Erste Evaluation des RL-Appraisal-Modells [54] auf interaktiven (nicht nur Vignetten-)Tasks in HCI

## Direkt zitierbare Schlüsselsätze

> *"Our approach opens the possibility of designing interactive systems that adapt to users' emotional states, thereby improving user experience and engagement."* (Abstract)

> *"Power appraisal provides a means to explain why a particular event, such as an error message, might cause widely different emotions in different users (e.g. confusion or even fear in novice users, and irritation in experienced users)."* (Section 3.2)

> *"Predicting users' emotional states during interaction is a longstanding goal of affective computing. However, traditional methods based on sensory data alone fall short due to the interplay between users' latent cognitive states and emotional responses."* (Abstract)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 2: Related Work / Stand der Forschung (★★ relevant — CR↔Emotion-Brücke)
ZHA24 liefert die theoretische Brücke zwischen Computational Rationality und affektiven Zuständen: Boredom und Irritation emergieren nicht aus Zufall, sondern direkt aus der Reward-Goal-Dynamik eines RL-Agenten. Das begründet, warum CL-Optimierung in Stage 2 auch affektiv relevant ist — ein Interface mit hohem cognitive load erhöht die Wahrscheinlichkeit von Conduciveness-Disruption und damit Irritations-Emergence. Zitierbar: *„Zhang et al. (2024) zeigen, dass affektive Zustände wie Irritation und Boredom direkt aus der Reward-Goal-Dynamik interaktiver Episoden emergieren — eine CR-theoretische Grundlage dafür, warum die vorliegende Pipeline Cognitive Load als Design-Proxy für affektive Negativerfahrungen minimiert."* ZHA24 schließt die Argumentationskette: niedrige CL → hohe Conduciveness → Happiness; hohe CL → obstruktive Conduciveness → Irritation.

### Kapitel 3: Methodik / Stage 2 (★★ relevant — Appraisal-Dimensionen auf Stage 2 abbildbar)
Die 4 Appraisal-Dimensionen von ZHA24 sind strukturell auf Stage 2-Outputs abbildbar: Conduciveness ↔ `search_efficiency` (positive TD-Updates entsprechen erfolgreicher Suche); Power ↔ `experience_level` in `user_profile.py` (hohe Power = Q-Werte spreizen sich = Nutzer kann gute von schlechten Aktionen unterscheiden); Goal Relevance ↔ `TASK_TYPE_WEIGHTS` in `task_descriptor.py` (Goal-Relevanz ist task-type-abhängig). Diese Abbildung ist keine direkte Implementierung, sondern eine konzeptuelle Analogie — sie zeigt, dass Stage 2 implizit Appraisal-relevante Variablen modelliert, ohne selbst ein Emotionsmodell zu sein.

### Kapitel 6: Diskussion (★★ relevant — Affektive Extension als Future Work)
ZHA24 benennt explizit als Möglichkeit: Systeme, die sich an emotionale Zustände der Nutzer adaptieren. Stage 2 ist genau die Vorstufe: der CL-Score identifiziert Interfaces mit hohem Irritations-Potenzial *vor dem Deployment*, während ZHA24 beschreibt, *wie* die Emotion laufzeitadaptiv adressiert werden könnte. Zitierbar: *„Während Zhang et al. (2024) zeigen, wie laufzeitadaptive Systeme auf emergente Nutzeremotionen reagieren können, adressiert die vorliegende Pipeline dasselbe Problem präventiv durch vordeployment-seitige Cognitive-Load-Optimierung."*

> **Was du lesen musst:** Abstract + Figure 1 + Table 1 (Appraisal-Profile, 5 Min.) + Section 3.2 (Formeln, 10 Min.)  
> Kap. 6 Affektive-Extension-Argument braucht nur Abstract + die Power-Appraisal-Passage (Section 3.2, 2 Absätze)

## Kritik / Offene Fragen
- Validierung auf einem einzigen, sehr einfachen Computing-Task — Generalisierbarkeit auf komplexe, multimodale HCI-Szenarien (z. B. Automotive HMI) nicht gezeigt; die drei emulierten Emotionen (Happiness/Boredom/Irritation) decken nicht das volle emotionale Spektrum relevanter Fahrsituationen ab
- Modell benötigt laufenden RL-Agenten mit vollständiger Interaktionshistorie (sequenzielle MDP-Episoden) — nicht anwendbar auf Screenshot-basierte Pre-Deployment-Evaluation; Stage 2 kann diesen Mechanismus konzeptuell nutzen, aber nicht direkt implementieren
- SVM-Klassifikator wurde auf simulierten (nicht auf echten menschlichen) Appraisal-Daten trainiert (Section 3.3) — die Übertragung auf reale Nutzer ist eine empirische Annahme, nicht formal validiert
- Emotion ≠ Cognitive Load: ZHA24 modelliert affektive Valenz (positiv/negativ/neutral), Stage 2 modelliert kognitive Beanspruchung — die Verbindung ist theoretisch plausibel (hohe CL → obstruktive Conduciveness → Irritation), aber in ZHA24 nicht direkt empirisch belegt

## Verbindungen zu anderen Papern
- Baut explizit auf → OUL22 (Computational Rationality als theoretisches Fundament; Jokinen als Co-Autor = direkte Jyväskylä-Linie)
- Jyväskylä-Lab-Kette → OUL22 → JOK20 → TOD18 → ZHA24: alle aus demselben Forschungskontext; ZHA24 ist die emotionale Extension der CR-Modellierungslinie
- Direkt komplementär → BAI24: beide RL-basiert in HCI, aber verschiedene Outputs — BAI24 modelliert Verhalten (Reading/Scrolling/Obstruction Costs), ZHA24 modelliert Emotionsemergenz; gemeinsame theoretische Basis ist Q-Learning + Reward-Maximierung
- Appraisal-Power → `user_profile.py` (experience_level): das Power-Appraisal in ZHA24 (hohe Power = Q-Werte spreizen sich = Experte erkennt gute Aktionen) ist die emotionale Kehrseite des experience_level-Parameters; dasselbe Error-Message-Ereignis erzeugt Irritation beim Experten, Angst beim Novizen
- CL-Emotion-Brücke → KRE16/18: Mikrosakkaden-basiertes CL-Maß (KRE18) + Emotionsemergenz (ZHA24) können sequenziell kombiniert werden: KRE18 detektiert CL-Peak → ZHA24 modelliert daraus die affektive Konsequenz

---

**Tags:** #emotion #appraisal #reinforcement-learning #computational-rationality #CR #affective-computing #CHI2024 #jyvaskylalab #jokinen #power-appraisal #boredom #irritation #happiness #affective-extension #user-profile
