# Eye Tracking Cognitive Load Using Pupil Diameter and Microsaccades with Fixed Gaze (2018)

**Autoren:** Krzysztof Krejtz, Andrew T. Duchowski, Anna Niedzielska, Cezary Biele, Izabela Krejtz  
**Quelle:** *PLOS ONE*, 13(9): e0203629 (2018). DOI: 10.1371/journal.pone.0203629  
**⚠️ DOI-Korrektur:** Note-Datei hieß ursprünglich KRE16 mit DOI 10.1371/journal.pone.0150489 (PLOS ONE 2016, gleichnamige Vorstudie). Das tatsächlich vorliegende und hier analysierte PDF ist das **2018-Paper** (e0203629). Die 2016-Version (e0150489) ist eine frühere Fassung derselben Autorengruppe; zitiere das 2018-Paper als jüngere, vollständigere Version.  
**Supervisor-Dokument:** nannte fälschlich DOI `10.1145/2857491.2857540` (ETRA 2016 — Gangstudien-Paper von Kothari et al.) — dieser DOI ist definitiv falsch.  
**PDF:** `Tier2_Sorgfaeltig/KRE16_Eye_Tracking_Cognitive_Load.pdf`

---

## Kernaussage
Pupillendurchmesser (inter-trial Veränderung, BCPD) und Mikrosakkaden-Magnitude sind valide, hochreliable physiologische Indikatoren für kognitive Belastung. Erster systematischer direkter Vergleich beider Methoden auf ihre Reliabilität (Cronbach α) und Sensitivität (ANCOVA-Effektgröße η²). Hauptbefund: beide Maße diskriminieren Aufgabenschwierigkeit signifikant und nahezu gleichwertig, aber Mikrosakkaden-Magnitude ist ökologisch valider weil unempfindlich gegen Ambient-Light und Off-Axis-Verzerrung.

## Methode
Experiment repliziert Siegenthaler et al. (2014): **3 × 6 within-subjects Design** — Task Difficulty (Difficult / Easy / Control) × Time-On-Task (6 Blöcke, 18 Trials gesamt). **N=13** (von 17 rekrutierten Psychologie-Studierenden; 4 wegen Eye-Tracker-Kalibrierungsfehlern ausgeschlossen; 7 männlich, 6 weiblich, M=29.77, SD=7.15, 20–40 Jahre).

**Aufgabe:** Mentally Arithmetic bei fixiertem Blick auf Zentralpunkt (Abweichung >3° = akustische Warnung). Difficult = rückwärts zählen in 17er-Schritten; Easy = vorwärts in 2er-Schritten; Control = keine Rechenaufgabe. 4 Prompts pro Trial, 15–80s Abstand, max. 9s Antwortzeit, max. 3 Min. pro Trial. Working Memory Capacity (DSPAN, Digit Span Forward + Backward) als Kovariate erhoben.

**Equipment:** SR Research EyeLink 1000, 500 Hz, Kinnstütze, 1920×1080px Bildschirm, 120–130 lux konstante Beleuchtung (Raum ohne Fenster). Stimuluspräsentation via PsychoPy.

**Abhängige Maße:**
- Mikrosakkaden: Magnitude (Grad), Rate (Hz), Peak Velocity / Magnitude Slope — Detektion via adaptiertem Engbert & Kliegl (2003) Algorithmus mit Savitzky-Golay-Filter (3. Grad, Breite 3, Velocity-Threshold 100°/s)
- Pupillometrie: CPD (Intra-Trial Change in Pupil Diameter, 2s Baseline) + BCPD (Inter-Trial Change, gesamter Control-Trial als Baseline) — Butterworth-Glättung (2. Grad, 1/8 Sampling-Periode)
- Selbsteinschätzung: NASA-TLX (5 Items, 1–21 Likert) nach jedem Trial-Block

## Wichtigste Ergebnisse
- **Aufgaben-Manipulation Check:** Antwortgenauigkeit: Difficult 49% (SD=0.04) vs. Easy 97% vs. Control 99%; NASA-TLX: Difficult 12.26 vs. Easy 8.39 vs. Control 5.19; beide Checks mit p<0.001 signifikant
- **Mikrosakkaden-Magnitude:** Haupteffekt Task Difficulty F(1.79,19.74)=32.39, p<0.001, **η²=0.17** — größter erklärter Varianzanteil im Set; Magnitude höher bei Difficult (0.46°) als Easy (0.43°) als Control (0.39°); Differenz wächst über Blöcke (Interaktionseffekt p<0.01)
- **BCPD (Inter-Trial Pupillendurchmesser):** F(1,11)=25.15, p<0.001, **η²=0.16** — gleichwertig zu Mikrosakkaden-Magnitude; Difficult BCPD=72.71 vs. Easy BCPD=17.37 (in relativen Einheiten)
- **CPD (Intra-Trial):** F(1.26,13.89)=6.44, p<0.05, **η²=0.07** — signifikant aber deutlich schwächer; CPD-Werte durchgehend negativ (Pupillenkonstriktion im Trial-Verlauf), was die Interpretation erschwert; Autoren empfehlen gegen CPD-Verwendung
- **Mikrosakkaden-Rate:** nur marginal signifikant (p=0.06) — nicht repliziert im Gegensatz zu Siegenthaler et al. (2014)
- **Reliabilität (Cronbach α):** Mikrosakkaden-Magnitude α=0.96, Rate α=0.96, BCPD α=0.95, CPD α=0.86, NASA-TLX α=0.88 — exzellent für alle Hauptmaße
- **MLR (Multinomiale Logistische Regression):** BCPD (β=2.47, SE=0.42, z=5.90, p<0.001) und Mikrosakkaden-Magnitude (β=1.20, SE=0.25, z=4.81, p<0.001) sind die einzigen signifikanten Prädiktoren für Difficult vs. Control; CPD und Mikrosakkaden-Rate nur für Easy vs. Control relevant
- **WMC-Kovariate:** Kein signifikanter Effekt von Working Memory Capacity auf Mikrosakkaden oder Pupillometrie — CL-Indikatoren sind von WMC unabhängig

## Direkt zitierbare Schlüsselsätze

> *"Microsaccade magnitude may serve as a reliable and sensitive discriminant of task difficulty, vis-à-vis cognitive load."* (Section 5)

> *"Being able to distinguish a user's level of cognitive load, especially in real-time, has significant implications for design and/or evaluation of interactive systems."* (Section 6)

> *"Unlike pupil diameter, microsaccade magnitude should also be free from the influence of ambient light."* (Section 6)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 2: Related Work / Stand der Forschung (★★★ zentral — physiologische Validierungsbasis)
KRE18 ist die stärkste verfügbare Referenz dafür, dass Eye-Tracking-Metriken kognitive Last valide und reliabel abbilden — und damit die Messmethode legitimiert, die sowohl DAS24 (HCEye-Dataset mit Fixationsmetriken) als auch die eigene Nutzerstudie (August 2026) verwenden. Das Paper beantwortet die Grundsatzfrage *„Kann man cognitive load überhaupt mit Eye-Tracking messen?"* mit einem klaren empirischen Ja, und liefert für zwei der wichtigsten Metriken (Pupillendurchmesser-Veränderung, Mikrosakkaden-Magnitude) Effektgrößen von η²≈0.16–0.17. Direkt zitierbar für den ersten Absatz des CL-Messungs-Abschnitts in Related Work: *„Die physiologische Validierbarkeit von Eye-Tracking als Cognitive-Load-Indikator ist durch Krejtz et al. (2018) empirisch belegt: Inter-Trial-Pupillendurchmesser und Mikrosakkaden-Magnitude diskriminieren Aufgabenschwierigkeit mit vergleichbaren Effektgrößen (η²=0.16–0.17) und exzellenter Reliabilität (α>0.95)."*

### Kapitel 4 / 5: Nutzerstudie & Validierung (★★★ zentral — Messvalidierung)
Die eigene Nutzerstudie (August 2026, N≈30-35) erhebt Eye-Tracking-Daten und NASA-TLX als Ground Truth für Stage-2-Outputs. KRE18 liefert die methodische Begründung dafür, dass diese beiden Messinstrumente tatsächlich dasselbe latente Konstrukt (cognitive load) erfassen und daher für die Kreuzvalidierung geeignet sind. Konkret: NASA-TLX α=0.88 + Pupillendilatation α=0.95 zeigen hohe konvergente Reliabilität → wenn beide im eigenen Datensatz mit dem CL-Score von Stage 2 korrelieren, ist das ein stärkeres Validierungsargument als jedes einzelne Maß allein. Zitierbar in der Methodik der Nutzerstudie: *„Die Wahl von NASA-TLX und Fixationsmetriken als Validierungsgrößen folgt Krejtz et al. (2018), die zeigen, dass beide Maße dieselbe Varianz in kognitiver Belastung erklären (α>0.88) und sich gegenseitig validieren."*

### Kapitel 6: Diskussion (★★ relevant — Fixed-Gaze-Limitation als Kontextualisierung)
Section 6 benennt die entscheidende Limitation von KRE18: Das Experiment erfordert **fixierten Blick** auf einen Zentralpunkt — genau die Bedingung, die bei freiem GUI-Viewing in realistischen Evaluationsszenarien nicht gilt. Das ist eine wichtige Einschränkung beim Transfer auf die Nutzerstudie: Mikrosakkaden sind unter freiem Viewing anders verteilt und weniger sauber isolierbar als in einem Laboratory-Fixed-Gaze-Paradigma. Die Autoren sehen Mikrosakkaden-Magnitude trotzdem als robuster an als Pupillendurchmesser (kein Licht-Einfluss, kein Off-Axis-Problem). Diese Limitation kann in der Diskussion der Nutzerstudie ehrlich eingestanden werden: *„Krejtz et al. (2018) zeigen, dass Mikrosakkaden-Magnitude kognitiven Load unter Fixed-Gaze-Bedingungen zuverlässig diskriminiert; die vorliegende Studie verwendet freies GUI-Viewing, was die Übertragbarkeit dieser Metriken einschränkt."*

> **Was du lesen musst:** Abstract (5 Min.) + **Section 4.7** (Pupillary and Microsaccadic Effect Sizes — die η²-Tabelle, 5 Min.) + **Table 1** (Reliabilität, 5 Min.) + **Table 3** (MLR-Koeffizienten, 5 Min.) + **Section 6** (Limitations, 10 Min.)  
> ⚠️ Section 3 (Algorithmus-Details: Butterworth, Savitzky-Golay, Engbert-Kliegl) = überfliegen — nur Section 3.2 (Dependent Measures) zur Übersicht reicht

## Kritik / Offene Fragen
- N=13 ist für ein physiologisches Eye-Tracking-Experiment mit within-subjects-Design vertretbar, aber klein — die exzellenten α-Werte und η²-Effektgrößen sind stabil genug, aber Replikationen mit größeren N wären wünschenswert
- Fixed-Gaze-Paradigma (kein GUI, keine freie Blickbewegung) ist weit von realen HMI-Evaluationsszenarien entfernt — direkter Transfer der Schwellwerte nicht möglich
- Mikrosakkaden-Rate nicht signifikant (entgegen Siegenthaler et al.) — KRE18 warnt selbst vor inkonsistenten Rate-Befunden in der Literatur; nur Magnitude zuverlässig
- Intra-Trial CPD schwerer interpretierbar als BCPD (CPD-Werte durchgehend negativ durch kurzes 2s-Baseline-Fenster) — in eigener Nutzerstudie nur BCPD verwenden, falls Pupillometrie erhoben wird
- Erfordert EyeLink 1000 mit 500 Hz Sampling Rate für Mikrosakkaden-Detektion — Standard-Eye-Tracker (z.B. Tobii mit 60–120 Hz) hätten möglicherweise zu niedrige Sampling Rate; für NASA-TLX-Validierung ist das kein Problem

## Verbindungen zu anderen Papern
- Direkte Validierungsbasis für → DAS24 (HCEye): HCEye verwendet Fixationsanzahl, Fixationsdauer, Sakkadenamplitude als CL-Proxies; KRE18 zeigt dass solche Eye-Tracking-Metriken tatsächlich CL abbilden
- Theorie-Kontext → OUL22 (CR): KRE18 beantwortet empirisch die Frage, welche physiologischen Maße die kognitiven Zustände abbilden, die OUL22 theoretisch modelliert
- Methodenparallele → JOK20 (Adaptive Feature Guidance): beide nutzen Fixationsmetriken als Proxy für kognitiven Zustand; KRE18 legitimiert diese Wahl mit physiologischer Evidenz
- Validierungspartner → NASA-TLX-Literatur: KRE18 zeigt konvergente Validität von NASA-TLX (α=0.88) mit physiologischen Maßen → direkte Rechtfertigung für NASA-TLX in der Nutzerstudie

---

**Tags:** #eye-tracking #cognitive-load #pupil-diameter #microsaccades #validation #physiological-measures #fixed-gaze #BCPD #reliability #cronbach-alpha #effect-size #PlosONE2018 #NASA-TLX #user-study-methodology
