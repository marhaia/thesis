# Shifting Focus with HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on User Attention and Saliency Prediction (2024)

**Autoren:** Das, A., Wu, Z., Škrjanec, I., & Feit, A.M.  
**Quelle:** Proc. ACM Human-Computer Interaction (ETRA), Vol. 8, Article 236  
**DOI:** (ACM ETRA 2024)  
**PDF:** DAS_2024.pdf  
**ID:** DAS24 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Wie beeinflusst hohe kognitive Last (durch Dual-Task) die visuelle Aufmerksamkeit auf UI-Elemente — und wie gut sagen aktuelle Saliency-Modelle dieses veränderte Verhalten voraus?

## Methode
- Eye-Tracking-Studie mit **27 Teilnehmern**
- **150 einzigartige Webseiten** als Stimuli
- Dual-Task-Paradigma zur Induktion kognitiver Last
- Zwei Arten von Highlighting: permanent und dynamisch
- Analyse: Gaze-Daten, Saliency-Model-Vergleich (State-of-the-Art Modelle)
- Datensatz: offen verfügbar (HCEye Dataset)

## Wichtigste Ergebnisse
- Hohe kognitive Last → **"Tunnel Vision"**: Aufmerksamkeit konzentriert sich, Highlights werden weniger beachtet
- Dynamisches Highlighting (animiert) bleibt auch unter Last aufmerksamkeitsstark
- Kognitive Last **verändert fundamental was als "salient" gilt** — aktuelles Bild-Saliency ≠ Load-adjustiertes Saliency
- State-of-the-Art Saliency-Modelle verbessern sich **signifikant**, wenn sie kognitive Last berücksichtigen

## Relevanz für meine Thesis
> Provides a "Load-Adjuster" — shows that AIM's saliency maps need to change when the user is under high cognitive load (e.g., multitasking), making the evaluation more realistic.

- **Rolle:** Empirischer Beweis dass kognitive Last Saliency-Evaluation verändert
- **Kapitel:** Related Work / Motivation / Methodik (Begründung für load-adjusted Metriken)
- **Argument:** AIM-Saliency-Maps sind nur unter "idle" Bedingungen valide — sobald der Nutzer unter Last steht (realer Kontext), braucht es angepasste Metriken
- **Direkt zitierbar für:** Begründung warum cognitive load in Saliency-Modelle integriert werden muss; Tunnel Vision Effekt

## Kritik / Offene Fragen
- Fokussiert auf Highlighting — kein allgemeines UI-Suchmodell
- Webseiten als Stimuli → Übertragbarkeit auf automotive GUIs zu diskutieren
- Meine Frage: Wie genau modelliere ich den Load-Adjuster in meinem Pipeline?

## Verbindungen zu anderen Papern
- Ergänzt → OUL18 (AIM): zeigt konkret die Lücke in AIM's Saliency-Modellen
- Ergänzt → JOK20: Search Time verlängert sich unter kognitiver Last
- Validierungsgrundlage → JIA23 (UEyes): wenn Load Saliency verändert, muss auch der Benchmark das berücksichtigen

## Meine Notizen (aus Excel)
- **Cognitive Metric:** Saliency under Load
- **Gap filled:** Empirical proof that cognitive load alters saliency maps in UIs
- **Relation to AIM:** Improvement — "Load-Adjuster": AIM's saliency maps need to change when user is under high cognitive load
- **Cluster:** Cognitive Load / Saliency

---

**Tags:** #cognitive-load #saliency #eye-tracking #dual-task #tunnel-vision #highlighting #HCEye #load-adjusted-saliency #AIM
