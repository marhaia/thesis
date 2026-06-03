# Adaptive Feature Guidance: Modelling Visual Search with Graphical Layouts (2020)

**Autoren:** Jokinen, J.P.P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X.  
**Quelle:** International Journal of Human-Computer Studies  
**DOI:** (Elsevier — IJHCS)  
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
- Nutzer lernen Layouts und wechseln von Wahrnehmungs- zu Gedächtnissteuerung
- Visuell homogene Layouts sind **schwerer zu lernen** und anfälliger für Änderungen
- Visuell saliente Elemente sind leichter zu finden und robuster gegenüber Änderungen
- Nicht-saliente Elemente, die weit verschoben werden → besonders schädlich für Wiedererkennbarkeit
- Modell passt gut zu menschlichen Daten

## Relevanz für meine Thesis
> Adds "Search Time" as a metric — allows AIM to predict not just where people look, but how long they need, depending on their experience with the layout.

- **Rolle:** Konkrete kognitive Metrik-Erweiterung für AIM — Search Time
- **Kapitel:** Related Work / Kognitive Metriken / Methodik
- **Argument:** Zeigt, dass Erfahrung des Nutzers mit dem Layout die Evaluation beeinflusst — rein bildbasierte Metriken ignorieren diesen Faktor
- **Direkt zitierbar für:** Begründung warum Search Time als Metrik relevant ist; Unterschied Novize vs. Experte

## Kritik / Offene Fragen
- Benötigt vordefinierte Suchziele — für dynamische UIs schwer anwendbar
- Computational heavy für komplexe GUIs
- Meine Frage: Wie integriere ich Search Time konkret in die AIM-API?

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (Computational Rationality als theoretisches Fundament)
- Konkretisiert → OUL18 (AIM: fügt Search Time als Metrik hinzu)
- Ergänzt durch → DAS24 (Search Time ändert sich unter kognitiver Last)

## Meine Notizen (aus Excel)
- **Cognitive Metric:** Predicted Search Time
- **Gap filled:** Integrates long-term memory into automated visual search models
- **Relation to AIM:** Improvement — adds Search Time as metric; predicts not just where people look but how long they need depending on layout experience
- **Cluster:** Cognitive Modeling

---

**Tags:** #visual-search #cognitive-modeling #search-time #long-term-memory #expected-utility #layout #novice-expert #AIM
