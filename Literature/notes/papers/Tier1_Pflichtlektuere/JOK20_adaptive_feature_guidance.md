# Adaptive Feature Guidance: Modelling Visual Search with Graphical Layouts (2020)

**Autoren:** Jokinen, J.P.P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X.  
**Quelle:** International Journal of Human-Computer Studies, Vol. 136 (2020), Article 102376  
**DOI:** 10.1016/j.ijhcs.2019.102376  
**PDF:** JOK2020.pdf  
**ID:** JOK20 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Wie wechseln Nutzer beim Umgang mit GUIs von aktivem visuellen Suchen zu gedächtnisgesteuerter Navigation — und wie lässt sich das computational modellieren?

## Methode
- Computational model der visuellen Suche auf grafischen Layouts
- Grundlage: **Expected Utility Maximization** (jede Fixation wird optimal gewählt)
- 3 Utility-Schätzungen pro Suchziel: ungeleitete Wahrnehmung / Langzeitgedächtnis (Ort) / Langzeitgedächtnis (Feature)
- Adaptives System: je besser das Gedächtnis, desto weniger Suche nötig
- Validierung gegen reale menschliche Daten in Studie mit grafischen Layouts

## Wichtigste Ergebnisse
- Nutzer lernen Layouts und wechseln von Wahrnehmungs- zu Gedächtnissteuerung (Novice → Expert)
- Visuell homogene Layouts schwerer zu lernen + anfälliger für Änderungen
- Visuell saliente Elemente leichter zu finden + robuster nach Layout-Änderung
- Nicht-saliente Elemente weit verschoben = besonders schädlich (Section 8.2 Guideline 8)
- Modell-Fit: Table 3 — R²=0.94–0.96 (Search Times), RMSE=0.45–1.98 über 3 Layouts (BP/NYT/WIN10), N=20, 24.514 Trials
- **Key counterfactual quote (Section 1):** *"The importance of such models is in their capacity to entertain counterfactual scenarios, allowing the modeller to change the task and user parameters to investigate various types of 'what ifs'"* — direkt zitierbar für Kap. 1
- **Key screenshot-gap quote (Section 8.1):** *"it is not possible to directly use the model to analyse images, such as screenshots of interfaces. However, an automated segmenter can be used"* — Stage 1 ist genau dieser Segmenter
- **Key ML bridge quote (Section 8.1):** *"machine learning methods could be used to find model parameters for individuals (Kangasrääsiö et al., 2019)"* — explizite Brücke zu KANK17 + Stage 2 ML
- **Key conclusion quote (Section 8.2):** *"models exploiting reinforcement or utility learning under the idea of bounded rationality offer exciting avenues for applied modelling in HCI"*
- ⚠️ Table 1: **19 Parameter** im Modell — selbe Zahl wie dein Feature-Vektor ℝ¹⁹

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★★ zentral)
- Section 1 counterfactual quote = eines der stärksten Pre-Deployment-Argumente in deinem gesamten Set
- Abstract: "tool for practitioners to evaluate how easy it is to find an item for a novice or an expert, and what happens if a layout is changed" → direkt zitierbar
- *"Jokinen et al. (2020) establish that computational visual search models enable practitioners to 'investigate various types of what ifs' regarding layout changes — the present pipeline extends this principle to automotive HMI by providing pre-deployment estimates of cognitive load and search efficiency from screenshots alone, without requiring user history."*

### Kapitel 2: Related Work (★★★ Hauptreferenz — methodisches Fundament)
- Drei Utility-Estimates (unguided / LTM-location / LTM-feature) = theoretische Basis für Stage 1 Feature-Typen:
  - Unguided ≈ bottom-up saliency (visual_complexity, clutter)
  - LTM-location ≈ familiarity proxy (kein direktes Äquivalent → Task Descriptor SEARCH_MODE)
  - LTM-feature ≈ top-down feature guidance (saliency map, contrast)
- Expected Utility Maximization = formale Basis für `search_efficiency = 1 - h[3]` in Stage 1
- Section 8.1: explizite Verbindung zu OUL18 als Screenshot-Segmenter → Stage 1 ist genau das
- ⚠️ Abgrenzung: JOK20 braucht User-History (bekannte Layouts) — deine Pipeline evaluiert unbekannte Designs ohne History via Task Descriptor als Proxy

### Kapitel 3: Methodik (★★★ direkte methodische Basis)
- Eq. (4)+(5): EMMA encoding time + saccade time → formale Basis für search_time-Berechnung in Stage 1
- Section 8.1: "machine learning methods could be used to find model parameters for individuals" → explizite theoretische Rechtfertigung für Ridge/RF/XGB in Stage 2
- 5 von 19 Parametern gefittet (Rest aus Literatur) → Methodologisches Vorbild für deine Kalibrierungsstrategie (Stage 2 Parameter vs. Literatur-basierte Features)
- Utility Learning Eq. (8) (Delta Learning Rule) → Analogie zu TASK_TYPE_WEIGHTS + TIME_PRESSURE_WEIGHTS im Task Descriptor

### Kapitel 5: Validierung (★★ relevant)
- Table 2 LMM (trial + layout + version als Fixed, participant als Random Intercept) → Methodologisches Vorbild für User Study August 2026 (N≈30–35)
- Table 3 R²+RMSE über 3 Layouts → Format für deine Kap. 5 Validierungsmetriken
- Section 5: N=20, within-subjects, 70 Min. → Größenordnung deiner Studie realistisch

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Section 8.1 Limitation: "not possible to directly use the model to analyse images" → Stage 1 löst genau das — stärkster "Gap → Solution"-Satz in Tier 1
- Section 8.2 Conclusion: "models exploiting reinforcement or utility learning... offer exciting avenues" → Kap. 6 Schlusssatz
- Section 7 Design Guidelines (1–8) → deine Pipeline validieren: produãziert sie höhere Load-Scores für homogene Layouts? (Section 7 Guideline 1+3)

> **Was du lesen musst:** Section 1 (Introduction, 10 Min.) + **Section 3** (Modelling, Eq. 4–8, 20 Min.) + **Table 1** (19 Parameter, 5 Min.) + **Section 8.1–8.2** (Limitations + Conclusion, 10 Min.)  
> ⚠️ Section 4 (Walkthrough) + Section 6 (Applications) = überfliegen — nur Figure 4 (Novice→Expert Transition) ist Kap. 2 relevant.

## Kritik / Offene Fragen
- Braucht vordefinierte Suchziele + User-History — deine Pipeline braucht beides nicht (Task Descriptor als Proxy)
- Symbolische Repräsentation der Layouts (keine Bilddaten) → Section 8.1 nennt das explizit als Limitation
- N=20, 3 Layouts (Web/Ticket/OS) — kein Automotive HMI, keine Fahrsituation
- Fixation Count wird systematisch unterschätzt (Section 5.3: "model strictly refixates only if this takes less time")
- R² für modifiziertes BP sehr niedrig (0.48) — Modell lernt schneller als Menschen bei hoch-überlernten Layouts

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (Computational Rationality als theoretisches Fundament), Lewis et al. 2014 (CR Definition)
- Konkretisiert → OUL18 (AIM: Section 8.1 zitiert OUL18 explizit als Screenshot-Segmenter)
- Brücke zu → KANK17 (Section 8.1: "machine learning methods... Kangasrääsiö et al." explizit zitiert)
- Ergänzt durch → DAS24 (Search Time ändert sich unter kognitiver Last — Tunnel Vision)
- Weiterentwickelt in → TOD19 (Section 6.3.3: TOD18 als praktische Anwendung des Modells zitiert)
- Selbe EMMA Basis → SHI24/CRTypist (beide nutzen EMMA für Eye-Movement-Modeling)
- Driving-Kontext → JOK21 (Jokinen & Kujala 2021 — Multitasking in OUL22 Table 1 [Ref 60])

---

**Tags:** #visual-search #cognitive-modeling #search-time #long-term-memory #expected-utility #layout #novice-expert #AIM #EMMA #utility-learning #LTM #screenshot-gap #IJHCS2020
