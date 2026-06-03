# Inferring Cognitive Models from Data using Approximate Bayesian Computation (2017)

**Autoren:** Kangasrääsiö, A., Athukorala, K., Howes, A., Corander, J., Kaski, S., & Oulasvirta, A.  
**Quelle:** ACM CHI 2017 (Aalto University / University of Birmingham / University of Oslo)  
**DOI:** —  
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
- ABC ermöglicht Parameterschätzung für komplexe kognitive Modelle
- Kann auf Verhaltensdaten angewendet werden, für die keine exakten Modellformeln existieren
- Zeigt wie kognitive Modelle aus realen HCI-Daten rückgeschlossen werden können

## Relevanz für meine Thesis
> Stage 2 / Method: Uses ABC to infer fixation duration and recall probability from click data. Strong methodological baseline for Stage 2.

- **Rolle:** Methodologisches Fundament für das Inferieren kognitiver Parameter aus Daten
- **Kapitel:** Methodik / Stage 2 Grundlagen
- **Argument:** Wenn ich kognitive Parameter nicht direkt messen kann, zeigt ABC einen validierten Weg diese zu schätzen — relevant für die Kalibrierung meiner Stage 2 Modelle
- **Direkt zitierbar für:** Methodologische Absicherung der Parameter-Inferenz

## Kritik / Offene Fragen
- ABC ist computational aufwendig — praktische Einschränkungen für Echtzeit-Evaluation?
- Wie übertrage ich ABC auf meinen konkreten Stage 2 Kontext (automotive GUIs, TLX-Scores)?

## Verbindungen zu anderen Papern
- Gleiche Autoren-Gruppe → OUL22, OUL18, JOK20: Oulasvirta-Lab Methodik
- Methodisch relevant → JOK20 (Adaptive Feature Guidance): ABC kann Feature-Gewichtungen inferieren

---

**Tags:** #ABC #Bayesian #cognitive-modeling #parameter-inference #method #fixation #recall #HCI
