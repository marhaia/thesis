# Inferring Cognitive Models from Data using Approximate Bayesian Computation (2017)

**Autoren:** Kangasrääsiö, A., Athukorala, K., Howes, A., Corander, J., Kaski, S., & Oulasvirta, A.  
**Quelle:** Proc. ACM CHI 2017, Denver CO, USA — pp. 1295–1306  
**DOI:** 10.1145/3025453.3025576  
**PDF:** KANK17.pdf  
**ID:** KANK17 | **Status:** 🟢 Analyzed (Auswahl: KEEP — Stage 2/Method)  

---

## Kernfrage
Wie können die Parameter eines kognitiven Modells aus realen Verhaltensdaten (z.B. Klick-Daten) inferiert werden — ohne direkte Messung?

## Methode
- **Approximate Bayesian Computation (ABC)**: statistisches Inferenzverfahren
- Keine analytische Likelihood nötig — Simulation + Vergleich mit Daten
- Anwendung: Inferenz von Fixationsdauer und Recall-Wahrscheinlichkeit aus Klick-Daten

## Wichtigste Ergebnisse
- ABC verbessert Parameterschätzung gegenüber manueller Literatur-Kalibrierung: TCT Manual 1.49s → ABC 0.93s (Ground Truth: 0.92s) — Study 1
- Individual models (Study 3) outperform population-level model, besonders für Ausreißer-Nutzer
- 4 Hauptparameter inferiert: `fdur` (Fixationsdauer), `dsel` (Selection Latency), `prec` (Recall-Wahrscheinlichkeit), `psem` (Periphere Wahrnehmung) — Table 1
- Variant 3 beste Vorhersage: fdur=280ms, dsel=290ms, prec=69%, psem=93%
- **Key inference-problem quote (Introduction):** *"how to set the parameter values of the model, such that the values agree with literature and prior knowledge, and that the resulting predictions match with the observations we have"* — exakt das Problem das Stage 2 löst
- **Key manual-tuning critique (Introduction):** *"the traditional way in HCI has been to set the model parameters based on past models and existing literature... this process generally has no guarantees that the final parameters will be close to the most likely values"* — direkte Kritik an JOK20's Parameterherleitung
- **Key future-work quote (Discussion):** *"inverse modeling might provide a general framework for implementing adaptive interfaces that are able to interpret user behavior so as to determine individual preferences, capabilities, and intentions"* — Stage 2 ist diese Instanziierung
- ⚠️ Study 3: Individual models ±10% um Population-Level-Parameter → individuelle Variation messbar und modellierbar

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★ relevant)
- Introduction-Quote: "how to set the parameter values of the model, such that the values agree with literature and prior knowledge" → exakter Problemrahmen deiner Thesis
- Abstract: *"ABC (i) improves estimates of model parameter values, (ii) enables meaningful comparisons between model variants, and (iii) supports fitting models to individual users"* → deine Stage 2 erfüllt alle drei Punkte ohne ABC's Rechenaufwand

### Kapitel 2: Related Work (★★ relevant)
- "Introduction to Computational Rationality"-Abschnitt = kompakteste CR-Definition im Paper-Set → Fußnote-Zitat neben OUL22 für Kap. 2 CR-Einführung
- Explizite Darstellung dass Standard-Praxis ("set from literature, tune by hand") keine Gütegarantien hat → direkte Kritik an JOK20's Herangehensweise → motiviert Stage 2
- Verbindet Chen et al. 2015 [Ref 13] (Menü-CR-Modell) mit JOK20 (derselber Modelltyp) → zeigt die methodische Linie OUL22 → JOK20 → KANK17 → deine Stage 2

### Kapitel 3: Methodik (★★★ zentral — die explizite Brücke)
- KANK17 ist das Paper, das JOK20 Section 8.1 explizit ankündigt: *"machine learning methods could be used to find model parameters for individuals (Kangasrääsiö et al., 2019)"*
- Study 1: Manual Tuning → ABC: TCT 1.49s → 0.93s → empirischer Nachweis dass datengetriebene Inferenz besser als Literaturwerte ist → dein Ridge/RF/XGB macht dasselbe, schneller
- Study 3: Individual models outperform population model → dein Task Descriptor ermöglicht pseudo-individuelle Adaptation ohne User-History
- Table 1 (4 Parameter): `fdur`, `dsel`, `prec`, `psem` — alle abbildbar auf dein Feature-Set (`search_time`, `cognitive_load_score`, `search_efficiency`, `attention_demand`)
- ⚠️ Abgrenzung: KANK17 nutzt ABC (Bayesian, 20h+ per Experiment-Run) — Stage 2 nutzt Ridge/RF/XGB (ms-schnell, pre-deployment tauglich) → explizit als praktische Weiterentwicklung begründen
- ⚠️ Abgrenzung: KANK17 inferiert aus Interaktionsdaten (Klick-Zeiten, gemessene User-Daten) — Stage 2 inferiert aus visuellen Features (Screenshot + Task Descriptor) ohne jede Nutzerdaten → grundlegend andere Einsatzklasse

### Kapitel 5: Validierung (★★ relevant)
- Study 1 Ergebnis-Format: "Manual vs. ABC vs. Ground Truth" in Figure 5 → Vorlage für deine "Pipeline vs. Baseline vs. Ground Truth"-Tabelle
- Study 3 Figure 10: Decrease in prediction error Individual vs. Population → Vorlage für per-Nutzer-Fehleranalyse
- N=21 (Bailly et al. Dataset), within-subjects → Größenordnung vergleichbar

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Discussion: *"inverse modeling might provide a general framework for implementing adaptive interfaces"* → Stage 2 ist diese Instanziierung — explizit so benennen
- Limitation: ABC ist computationally prohibitive (20h+, Cluster-Compute) → dein Ridge/RF/XGB = praktische Alternative mit akzeptablem Trade-off
- KANK17 zeigt dass Individual-Modelle ±10% von Population abweichen → validiert dass Task Descriptor-basierte Adaptation sinnvoll ist

> **Was du lesen musst:** Introduction (10 Min.) + **"Inverse Modeling Approaches"** + **"Approximate Bayesian Computation"-Abschnitt** (10 Min.) + **Study 1 Results** (Figure 5, 5 Min.) + **Study 3 Results** (Figure 10, 5 Min.) + Discussion (5 Min.)  
> ⚠️ BOLFI-Implementierungsdetails (Appendix) = überspringen — nur relevant wenn du ABC selbst implementieren willst (tust du nicht)

## Kritik / Offene Fragen
- ABC computationally prohibitive (20h+ per Study, Cluster-Compute) → explizit als Limitation nennen, die Stage 2 durch ML-Regression löst
- Nur Menu-Interaction als Case Study — kein Automotive, kein Screenshot-Input
- Inferiert aus Klick-Zeiten (aggregate behavioral data) — braucht also immer Nutzerdaten → deine Pipeline braucht keine
- Study 3: User S8 (Ausreißer) schlecht modellierbar → individuelle Extremnutzer bleiben schwierig

## Verbindungen zu anderen Papern
- Explizit angekündigt in → JOK20 Section 8.1: "machine learning methods could be used to find model parameters for individuals (Kangasrääsiö et al., 2019)" — stärkste direkte Verknüpfung im gesamten Set
- Selbe Autoren-Gruppe → OUL22 (CR-Fundament), OUL18 (AIM als Segmenter), JOK20 (das Modell, das KANK17 kalibriert)
- Baut auf → Chen et al. 2015 [Ref 13] (Menü-CR-Modell) — derselbe Modelltyp wie JOK20
- Methodisch übertroffen durch → Stage 2 Ridge/RF/XGB: gleiche Inferenz-Idee, aber ohne Nutzerdaten und in ms statt 20h
- Verbindet mit → OUL22 Table 1: KANK17 = Instanz der "parameter inference for CR models" — deine Stage 2 = nächste Zeile in dieser Tabelle

---

**Tags:** #ABC #Bayesian #cognitive-modeling #parameter-inference #inverse-modeling #method #fixation #recall #CHI2017 #stage2 #individual-differences #computational-rationality
