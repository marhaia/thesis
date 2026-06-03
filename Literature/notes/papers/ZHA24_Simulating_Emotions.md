# Simulating Emotions With an Integrated Computational Model of Appraisal and Reinforcement Learning (2024)

**Autoren:** Jiayi Eurus Zhang, Joost Broekens, Bernard Hilpert, Jussi P.P. Jokinen  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3641908  
**PDF:** `Tier2_Sorgfaeltig/ZHA24_Simulating_Emotions.pdf`

---

## Kernfrage
Wie kann man emotionale Zustände (Freude, Langeweile, Frustration) während der Interaktion computational modellieren — durch Integration von Appraisal-Theorie und RL?

## Methode
- Temporal Difference RL + Cognitive Appraisal Theory kombiniert
- Emotion als Bewertung von interaktiven Events (Reward Processing + Kognitive Bewertung)
- Illustration: Nutzer versucht ein Computerproblem zu lösen → wiederholte Erfolge → Glücksgefühl
- Validierung: Modell-Vorhersagen vs. menschlich berichtete Emotionen

## Wichtigste Ergebnisse
- Emotionen (Freude, Langeweile, Frustration, Angst) emergieren aus Reward-Prozessierung
- Appraisal-Dimension: Neuartigkeit, Ziel-Kongruenz, Bewältigbarkeit
- Boredom entsteht wenn Reward-Raten unter Erwartung fallen
- Frustration wenn Ziel-Kongruenz trotz Aufwand nicht erreicht wird

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Personality Layer — Affektive Extension)
- **Argument das es stützt:** Emotionale Zustände als CR-basierte Emergenz rechtfertigt Emotion als optionaler 4. Output deines Systems (Affective Load)
- **Direkt zitierbar für:** "Zhang et al. (2024) zeigen, dass Langeweile und Frustration als emergente CR-Outcomes modellierbar sind — Grundlage für eine affektive Erweiterung"

> **Was du lesen musst:** Abstract + Figure 1 (Modell-Übersicht) + Section 5 (Ergebnisse) — ca. 20 Min.  
> **Achtung:** Dieses Paper ist für die optionale Erweiterung, nicht für den Core. Nur lesen wenn du Affective Extension ausbauen willst.

## Kritik / Offene Fragen
- Emotion ≠ Cognitive Load — konzeptuell verwandt, aber nicht gleichzusetzen
- Validierung auf sehr vereinfachtem Task (Computerproblem lösen)
- Automotive-Kontext: Frustration beim Fahren sehr anders als im Labor

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR), Jokinen (Jyväskylä Gruppe)
- Ergänzt → BAI24 (Attention Switching + Emotion), LIN24 (Task Switching)

---

**Tags:** #personality-layer #emotion #appraisal #RL #affective-extension #optional
