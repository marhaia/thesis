# Towards Ability-Based Optimization for Aging Users (2016)

**Autoren:** Sayan Sarcar, Jussi Jokinen, Antti Oulasvirta, Chaklam Silpasuwanchai, Zhenxin Wang, Xiangshi Ren  
**Quelle:** ITAP 2016 (International Conference on Information Technology and Aging), ACM  
**DOI:** 10.1145/2996267.2996275  
**PDF:** `Tier2_Sorgfaeltig/SAR16_Ability_Based_Aging.pdf`

---

## Kernfrage
Wie kann man UI-Design für ältere Nutzer optimieren, indem man altersbedingte Veränderungen in motorischen und kognitiven Fähigkeiten als Modellierungsgrundlage nutzt?

## Methode
SAR16 präsentiert **erste Ergebnisse** (Workshop-Paper, kein Vollstudie) aus einem parametrischen Ability-Based-Optimization-Ansatz für ältere Nutzer. Der Kern ist das Modell **Touch-WLM** (Touch Word-Level Model) — ein prädiktives Modell für Texteingabe auf Smartphones, das individuelle Unterschiede als Parametervektor θ ausdrückt und mit einem Optimizer kombiniert wird, der den besten Design-Konfiguration für gegebene θ sucht.

**Design Space:** 4 Faktoren für ein Smartphone-Keyboard — Anzahl Zeilen in der Wortvorschlagsliste (1–5), Zeilenhöhe (0.03–0.07% Bildschirmhöhe), Elemente pro Zeile (fest 3), Zeilen im Textdisplay-Bereich (2–7). QWERTY + Gruppierlayouts (3×3, 5×2, 10×1) für Englisch und Finnisch.

**Touch-WLM Parameter (Table 1):** Okulomotorisch (EMMA: eK, ek, tprep, texec, tsacc), motorisch (WHo-Modell: mk = Gesamtressource, mα = Speed-Accuracy-Bias, WHomin/max), kognitiv-strategisch (ma = Fingergenauigkeit, l = Buchstaben vor Proofreading, skey = visuelle Suchzeit, tconfirm = Backspace-Entscheidungszeit). Optimale Strategie wird über Rational Analysis ermittelt (exhaustive search über Strategie-Raum).

**Emulierte Nutzergruppen (Figure 6):** (a) Tremor/Parkinson (↓mk, ↓Fingergenauigkeit), (b) eingeschränkte visuelle Suche (↑skey), (c)+(d) langsameres Proofreading (↑l, verschiedene SAT-Konfigurationen). Junge vs. ältere Erwachsene: empirisch erhobene Parameterwerte aus Table 3 (z.B. EMMA tprep: YA=0.292s vs. OA=0.326s; WHo-max: YA=0.0753 vs. OA=0.0538 — ältere Nutzer haben niedrigere Maximalgeschwindigkeit).

**Kein N, kein between-subjects-Vergleich** — das ist ein Simulationspaper mit empirisch abgeleiteten Parametersets, keine empirische Studie mit Probanden.

## Wichtigste Ergebnisse
- **Tremor/Parkinson-Layout:** +13.83% WPM über QWERTY-Baseline (2.21 vs. 1.94 WPM simuliert) durch großflächiges Gruppierlayout; mk=1.0 (niedrige SAT-Ressourcen) erzwingt Larger-Target-Präferenz
- **Visuelle-Suche-Layout:** +7.5% WPM (8.1 vs. 7.5 WPM) durch erweiterte Wortvorschlagsliste; bei skey=1.0s fällt Baseline-WPM von 15.7 auf 7.5 — visuelle Suche ist dominant leistungsbegrenzend
- **Proofreading-Layouts:** +2.64% und +8.43% WPM über Baseline (10.73 und 10.82 WPM) durch Gruppierlayout + angepassten l-Schwellwert; hoher l-Wert führt zur Wahl von accurate SAT → paradoxerweise profitiert Baseline dann wieder
- **Kernbefund:** "One design for all" versagt, weil Trade-offs zwischen Design-Faktoren kontextsensitiv sind — was für Tremor-Nutzer optimal ist (große Targets) schadet Nutzern mit visuellem Suchproblem (mehr Scrollaufwand); nur parametrische Modellierung kann diese Trade-offs auflösen
- **CR-Verbindung:** Das Paper führt erstmals Computational Rationality (Rational Analysis für Strategy Adaptation) in die Ability-Based-Optimization ein — SAR16 + JOK20 + OUL22 sind explizit miteinander verbunden; SAR16 [Ref 35 in JOK20] ist die unmittelbare Vorgängerpublikation des Oulasvirta-Labs zu JOK20

## Direkt zitierbare Schlüsselsätze

> *"We argue that, to be successful, predictive models must take into account how users adapt their behavioral strategies as a function of their abilities."* (Abstract)

> *"We note tremendeously important to investigate methodology that allows designers to go beyond the failed 'one design fits all' approach."* (Discussion)

> *"[The parametric approach] allows for optimizing for a single objective (e.g., task completion time) without heuristics."* (Related Work — SUPPLE-Abgrenzung)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 2: Related Work / Stand der Forschung (★★★ zentral — Intellectual Lineage + CR-Brücke)
SAR16 ist das entscheidende Verbindungsglied in der intellectual lineage von GAJ08 → SAR16 → JOK20 → Stage 2, weil es als erstes Paper im Oulasvirta-Lab Computational Rationality (Strategy Adaptation via Rational Analysis) mit Ability-Based Design verbindet. GAJ08 hatte motorische Fähigkeiten über Fitts-Law modelliert, aber keine emergente Strategie — SAR16 ergänzt, dass Nutzer ihre Strategie optimal an ihre Fähigkeiten anpassen (Speed-Accuracy Trade-off als Funktion von mk und mα). Das ist der theoretische Sprung, der JOK20's Adaptive Feature Guidance vorbereitet und in OUL22's CR-Framework mündet. Direkt zitierbar als Mittelpunkt dieser Kette: *„Sarcar et al. (2016) erweiterten das Ability-Based-Design-Paradigma von Gajos et al. (2008) durch Einbeziehung von Computational Rationality: Nutzerstrategien werden nicht als fest angenommen, sondern als optimale Adaptation an gegebene motorische und kognitive Fähigkeiten modelliert — eine Grundannahme, die die vorliegende Pipeline für den User Profile Input übernimmt."*

### Kapitel 3: Methodik / Stage 2 (★★★ zentral — User Profile Parametrisierung)
Table 1 von SAR16 ist die direkteste Vorlage für die Parametrisierungslogik in `stage2/user_profile.py`. SAR16 zeigt, wie ein Nutzermodell als Parametervektor θ = {okulomotorisch, motorisch, kognitiv-strategisch} aufgebaut ist, wobei jede Dimension einen anderen Aspekt der Nutzer-Capabilities abdeckt. In Stage 2 übernimmt der User Profile dieselbe Logik — experience_level, workload_tolerance, search_strategy — als kognitive Parameter, die den CL-Score skalieren. SAR16's explizite Aufteilung in Eye Movements / Motor Performance / Strategy entspricht direkt der Drei-Schichten-Struktur, die der vorliegende Ansatz für kognitive Traits verwendet. Zitierbar: *„Die Parametrisierung des User Profiles in drei Dimensionen (kognitiv, strategisch, erfahrungsbasiert) folgt dem von Sarcar et al. (2016) eingeführten parametrischen Nutzermodell, das motorische und kognitive Fähigkeiten als separate, kombinierbare Optimierungsgrößen darstellt."*

### Kapitel 6: Diskussion (★★ relevant — Limitation: Simulation ohne Nutzerstudie)
SAR16 ist explizit ein „first results"-Paper ohne empirische Validierungsstudie — die Autoren selbst benennen als Future Work, dass Nutzerstudien mit Probanden mit spezifischen Einschränkungen nötig sind, um die optimierten Layouts gegen QWERTY zu testen. Das ist dieselbe Limitation, die Stage 2 hat: der CL-Score ist bisher modellbasiert-simuliert, nicht gegen echte Performance-Daten validiert. Für die Diskussion: *„Wie Sarcar et al. (2016) beschreiben die vorliegenden Ergebnisse zunächst eine modellbasierte Exploration ohne empirische Validierung der optimierten Designs — die August-2026-Nutzerstudie schließt diese Lücke für das Automotive-HMI-Szenario analog zur geplanten Validierungsstudie in SAR16."*

> **Was du lesen musst:** Abstract (5 Min.) + **Section „Approach"** (parametric optimization overview, Figure 2, 10 Min.) + **Table 1** (Parameter-Inventar, 5 Min.) + **Table 3** (YA vs. OA Parameter-Werte, 5 Min.) + **Discussion** (10 Min.)  
> ⚠️ Section Touch-WLM (Algorithmus-Details zu EMMA, WHo-Modell) = überfliegen — nur die Parameterliste und die Ergebnisse der vier Nutzergruppen braucht es

## Kritik / Offene Fragen
- Kein N, keine empirische Nutzerstudie — reine Simulation mit empirisch abgeleiteten Parameterwerten aus separaten Vortests; die WPM-Zahlen (z.B. +13.83%) sind Modellvorhersagen, keine gemessenen Performance-Werte
- Text Entry auf Smartphone ≠ visuelle Suche im Automotive Dashboard: die motorischen Parameter (Fingergenauigkeit, Tremor, Scrollgeschwindigkeit) sind für HMI-Touchscreens im Auto teilweise übertragbar, aber die kognitive Dual-Task-Struktur (Fahren + Ablesen) fehlt vollständig
- Die vier emulierten Nutzergruppen sind sehr schematisch (je ein Parameter erhöht) — reale Alterseffekte sind multidimensional und korreliert (z.B. Tremor + verlangsamte Sakkaden + schlechteres Working Memory gleichzeitig); SAR16 adressiert das als bekannte Limitation
- Workshop-Paper (6 Seiten, ITAP '16 Kochi) — Peer-Review-Tiefe ist geringer als CHI; die Ergebnisse sind als Proof-of-Concept gedacht, nicht als abschließende Validierung

## Verbindungen zu anderen Papern
- Intellectual-Lineage-Mittelpunkt: **GAJ08** → **SAR16** → **JOK20** → Stage 2 (`stage2/user_profile.py`) — das ist die sauberste, komplett belegbare Entwicklungslinie im Paper-Set
- Explizit zitiert → GAJ08 [Ref 14] (SUPPLE als direkter Vorgänger, aber mit Heuristik-Limitation) + JOK20-Vorgänger implizit durch Oulasvirta-Ko-Autorenschaft
- CR-Brücke → OUL22: SAR16's Rational Analysis für Strategy Adaptation ist der direkte Vorläufer von OUL22's CR-Framework; beide sagen, dass Menschen optimal unter Constraints handeln
- Table-1-Entsprechung → `stage2/user_profile.py`: SAR16 θ = {okulomotorisch / motorisch / strategisch} ↔ Stage 2 User Profile = {experience_level / workload_tolerance / search_strategy}
- Skalierbarkeits-Argument → OUL18 (AIM): SAR16 braucht Parameter-Elicitation pro Nutzergruppe; OUL18 + Stage 1 brauchen nur Screenshot — das ist der Fortschritt vom nutzertest-basierten zum screenshot-basierten Paradigma

---

**Tags:** #ability-based #aging #user-profile #parametric-model #rational-analysis #strategy-adaptation #Touch-WLM #ITAP2016 #oulasvirta-lab #CR-precursor #user_profile_py #intellectual-lineage
