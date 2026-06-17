# Familiarisation: Restructuring Layouts with Visual Learning Models (2018)

**Autoren:** Kashyap Todi, Jussi Jokinen, Kris Luyten, Antti Oulasvirta  
**Quelle:** IUI 2018 (ACM IUI Conference), Tokyo, Japan — **nicht CHI 2018**  
**DOI:** 10.1145/3172944.3172949  
**PDF:** `Tier2_Sorgfaeltig/TOD18_Familiarisation.pdf`

---

## Kernfrage
Wie kann man Layouts automatisch so umstrukturieren, dass Nutzer häufig gesuchte Elemente auf unbekannten Interfaces schneller finden — basierend auf ihrer Nutzungshistorie?

## Methode
**Familiariser** ist ein browser-basiertes System (Swift, MacOS 10.13), das beim Besuch einer neuen Webseite deren Layout automatisch auf Basis der Nutzungshistorie umstrukturiert — Element-Positionen werden so verschoben, dass die neue Seite dem ähnelt, was der Nutzer bereits kennt.

**4 Familiarisierungs-Prinzipien (Figure 1):**
- **I. Frequenz:** Die meistbesuchte Seite aus der History wird als Basis-Design gewählt; direkte 1:1-Retargetierung `fpage = nvisits × taverage`
- **II. Serial Position Curve:** Kombiniert Frequenz (Sv), Recency (Sr) und Primacy (Sp) zu einem Familiarity Score `Fpage = α·Sv + β·Sr + γ·Sp` (Gewichte α=β=0.4, γ=0.2); berücksichtigt Vergessen — kürzlich besuchte und zuerst besuchte Seiten werden besser erinnert
- **III. Visual Statistical Learning (VSL):** Baut Wahrscheinlichkeitsverteilungen über Feature-Positionen aus der gesamten History auf; Resultat ist ein Feature-Mesh aus häufigsten Positionen aller erkannten UI-Elemente (Logo, Navigation, Search Box, etc.)
- **IV. Cognitive Model:** Integriert EMMA-Augenmovementmodell [Salvucci 2001] + ACT-R Activation Memory [Anderson et al. 1998]; berechnet Aktivierungsstärken `Bi = ln(Σ t_j^{-d})` und Recall-Zeit `Ti = F·e^{-f·Bi}` — simuliert echtes Lern- und Vergessverhalten; dieses Modell ist [Ref 18] = JOK17/JOK20-Vorgänger

**Overlap Resolution:** Integer-Linear-Programming (IBM CPLEX) für überlappungsfreie Layouts nach Repositionierung.

**Nutzerstudie:** N=16 (21–36 Jahre, kein Sehfehler), within-subject. Learning Phase: 25 Trials auf 5 zufällig ausgewählten Shopping-Websites. Test Phase: 100 Trials auf unbekannten Seiten, Original vs. Familiarisiert (Prinzip III — VSL). Aufgabe: Feature-Element (Logo, Nav, Search Box) anklicken so schnell wie möglich. EyeTribe Eye-Tracker für Fixationsdaten, Cursor-Logging für Klickzeiten. 30 Shopping-Websites im Dataset.

## Wichtigste Ergebnisse
- **Visuelle Suchzeit:** Original = 2.8s vs. Familiarisiert = 2.5s — Reduktion um >10%; t(663)=5.3, p<0.001, F(1,663)=28.5, p<0.001, β=0.35 (standardisiert)
- **Fixationsanzahl:** Original = 3.4 vs. Familiarisiert = 2.6 — Reduktion um >23%; t(596)=4.7, p<0.001, F(1,596)=22.3, p<0.001, β=0.35
- **Prinzip IV (Cognitive Model) > alle anderen:** Direkter Vergleich aller 4 Prinzipien wurde als Future Work zurückgestellt — nur Prinzip III (VSL) wurde in der Studie getestet; aber Diskussion argumentiert Prinzip IV als theoretisch leistungsfähigstes wegen explizitem Forgetting-Mechanismus
- **Aktivierung via Geschichte:** Längere Nutzungshistorie → höhere Aktivierungsstärken → schnellere, expertenhaftere Feature-Suche (Gleichung 8+9)
- **Familiarisierungszeitpunkt:** 25 Page-Visits / maximal 5 einzigartige Seiten in History = empirisch bestimmter optimaler Aktivierungszeitpunkt

## Direkt zitierbare Schlüsselsätze

> *"Familiarisation improved user performance by reducing both visual search time and number of gaze fixations."* (Discussion)

> *"Familiarisation significantly improves visual search time by over 10%, and reduces the number of fixations by over 23%, while searching for features on a given design."* (Discussion)

> *"By understanding how this happens, we could design and adapt interfaces automatically such that elements can be more easily found."* (Introduction)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 2: Related Work / Stand der Forschung (★★★ zentral — Empirische Basis für History-Effekt)
TOD18 liefert die direkteste empirische Evidenz dafür, dass **Nutzungshistorie** die visuelle Suchzeit auf GUIs messbar beeinflusst: +10% Suchzeit-Reduktion und +23% weniger Fixationen durch layout-Familiarisierung. Das ist der empirische Baustein, der die `search_experience`-Variable im User Profile von Stage 2 legitimiert — erfahrene Nutzer eines bestimmten HMI-Layouts benötigen weniger kognitive Ressourcen für visuelle Suche, weil Aktivierungsstärken (Bi) für häufig gesehene Elemente hoch sind. Direkt zitierbar: *„Todi et al. (2018) zeigen empirisch, dass Nutzungshistorie die visuelle Suchzeit auf unbekannten Interfaces um über 10% und die Fixationsanzahl um über 23% reduziert — eine Grundlage für die History-gewichtete Sucheffizienz-Komponente in Stage 2."* TOD18 ist die erste Evidenz; TOD19 repliziert und erweitert mit vollständiger Studie.

### Kapitel 3: Methodik / Stage 2 (★★ relevant — Cognitive Model als Referenzarchitektur)
Prinzip IV (Cognitive Model) verwendet dieselbe EMMA-basierte Augenmovementarchitektur wie JOK20 [Ref 18 in TOD18 = JOK17, direkter JOK20-Vorgänger] und denselben ACT-R-Activation-Memory-Mechanismus. Die Aktivierungsformel `Bi = ln(Σ t_j^{-d})` modelliert, wie häufige + zeitnahe Exponierung die Recall-Wahrscheinlichkeit eines Layout-Elements erhöht — strukturell analog zu dem, was `search_experience` in `user_profile.py` approximiert: ein höherer experience_level reduziert den erwarteten CL für visuelle Suche, weil Elementierten aktivierter sind. TOD18 liefert die formale Basis für diesen Mechanismus.

### Kapitel 6: Diskussion (★★ relevant — Adaptive UI Kostenargument)
TOD18 Discussion benennt explizit das Hauptproblem adaptiver UIs: Unpredictability und Adaptation-Cost können die Vorteile zunichtemachen. TOD18 umgeht das, indem nur *neu besuchte* Interfaces familiarisiert werden — bestehende bekannte Layouts werden nicht verändert. Dasselbe Argument gilt für Stage 2: der CL-Score und die Coherence-Checks greifen *vor* dem Deployment, nicht während der Nutzung — damit entsteht keine Adaptation-Cost für den Endnutzer. Zitierbar: *„Anders als laufzeitadaptive Ansätze (Todi et al., 2018) operiert die vorliegende Pipeline ausschließlich in der Design-Phase und vermeidet damit die von Lavie & Meyer (2010) beschriebenen Kosten unvorhergesehener Interface-Änderungen."*

> **Was du lesen musst:** Abstract (5 Min.) + **Tabelle 1** (Ergebnisse, 2 Min.) + **Discussion** (10 Min.)  
> TOD18 bekannt wenn TOD19 gelesen — TOD19 ist die vollständige Studie; TOD18 nur für die 4-Prinzipien-Taxonomie und die JOK17-Verbindung (Ref 18) relevant

## Kritik / Offene Fragen
- Nur Prinzip III (VSL) in der Nutzerstudie getestet — Prinzip IV (Cognitive Model, das theoretisch stärkste) wurde nicht gegen menschliche Daten validiert; Vergleich aller vier Prinzipien ist explizit als Future Work benannt
- N=16 für erste Evidenz akzeptabel, aber die 100 Test-Trials pro Proband (mit Fatigue-Risiko bei ~30 Min. Gesamtdauer) könnten Learning-Effekte und Reihenfolgeeffekte unzureichend kontrolliert haben
- Shopping-Websites Domain: Auf Automotive HMI nicht direkt übertragbar — Automotive-Interfaces haben fixierte Layout-Strukturen (kein dynamisches Parsing nötig), aber die kognitive Mechanik (Aktivierungsstärke durch Wiederholung → schnellere Suche) gilt kontextunabhängig
- Familiarisierung erfordert Nutzungshistorie und DOM-Parsing — beides ist für Pre-Deployment-Evaluation (Screenshot-only) nicht verfügbar; Stage 2 approximiert den History-Effekt über den User Profile Parameter, nicht durch echte Historiendaten

## Verbindungen zu anderen Papern
- Direkter Vorgänger zu → TOD19 (TOCHI 2019): TOD18 = IUI-Paper mit Pilot-Evidenz; TOD19 = vollständige TOCHI-Studie mit N=40 und formeller Validierung; in der Thesis immer TOD18 + TOD19 zusammen zitieren
- Explizit zitiert → JOK17 [Ref 18] = Jokinen et al. CHI 2017 "Modelling Learning of New Keyboard Layouts" = unmittelbarer Vorgänger von JOK20; das Cognitive Model (Prinzip IV) implementiert dieselbe EMMA+ACT-R Architektur
- Suchmodell-Brücke → JOK20: Prinzip IV von TOD18 ist strukturell identisch mit JOK20's Visual Search Model — beide nutzen EMMA-Kodierzeit + ACT-R-Aktivierungen; TOD18 fügt nur den Layout-Restrukturierungs-Layer obendrauf
- User-Profile-Verbindung → `stage2/user_profile.py`: TOD18 Aktivierungsformeln (Bi) sind die theoretische Basis für den experience_level-Parameter
- Adaptive-UI-Kontext → GAJ08 + SAR16: TOD18 ist der dritte Baustein im Adaptive-UI-Panorama; GAJ08 = statische Ability-Based Generation, SAR16 = parametrische Optimization, TOD18 = runtime-adaptive Familiarisierung

---

**Tags:** #visual-search #layout #familiarisation #history-effect #adaptive-UI #cognitive-model #EMMA #ACT-R #IUI2018 #user-profile #search-experience #TOD19-precursor #oulasvirta-lab
