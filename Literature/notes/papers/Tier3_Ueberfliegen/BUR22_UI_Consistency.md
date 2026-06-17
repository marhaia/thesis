# (Semi-)Automatic Computation of User Interface Consistency (2022)

**Autoren:** Burny, N., & Vanderdonckt, J.  
**Quelle:** ACM (Université catholique de Louvain)  
**DOI:** —  
**PDF:** BUR22.pdf  
**ID:** BUR22 | **Status:** 🟢 Analyzed (Missing Rabbit: C5 Coherence)  

---

## Beim Ueberfliegen - gezielt lesen
> Note reicht fuer Evaluation/Coherence-Abschnitt. PDF nur wenn du die formale Konsistenz-Formel brauchst.

**Direkt springen zu:**
- **Section 3 Formal Approach (ca. S. 4-6):** Die mathematische Konsistenz-Formel
- **Table 1 / Metrik-Uebersicht (ca. S. 5-7):** Welche Konsistenz-Dimensionen gemessen werden
- **Figure 1-2:** Visuelle Beispiele konsistente vs. inkonsistente UIs

**Ueberspringen:** Section 5 (Tool-Implementierung), Evaluation-Details

Kernunterschied zur Thesis: BUR22 = visuelle Konsistenz zwischen Screens.
Dein Coherence-Term = Konsistenz zwischen Outputs (Saliency, Fixation, Load).

---

## Kernfrage
Wie kann die Konsistenz von GUIs — innerhalb einer App (intra) und zwischen Apps (inter) — automatisch berechnet werden?

## Methode
- Formale Formel + Methode für Konsistenz-Berechnung
- Anwendung auf einzelne Screens und App-übergreifend
- (Semi-)automatischer Ansatz

## Wichtigste Ergebnisse
- Bisherige Metriken wirken immer auf einem einzelnen GUI — Konsistenz erfordert Vergleich mehrerer Screens
- Präsentiert erste formalisierte Methode zur automatischen Konsistenzberechnung für UIs

## Relevanz für meine Thesis
> C5: Coherence — relevant für die Abschluss-Evaluation der Pipeline.

- **Rolle:** Unterstützt die Kohärenz-Dimension der Evaluation — wenn meine Pipeline mehrere GUIs bewertet, muss sie konsistente Ergebnisse liefern
- **Kapitel:** Evaluation / Methodik
- **Argument:** Konsistenz als Qualitätsmerkmal für automatisierte GUI-Evaluatoren

## Kritik / Offene Fragen
- Beschränkt auf formale Konsistenz (visuell), nicht auf kognitive Konsistenz
- Wie unterscheidet sich ihr Konsistenz-Begriff von meinem Multi-Output Coherence Term?

## Verbindungen zu anderen Papern
- Gleiche Autoren → BUR21 (UiLab): methodologische Fortsetzung
- Ergänzt → OUL18 (AIM): liefert eine Dimension (Konsistenz) die AIM fehlt

---

**Tags:** #GUI-consistency #semi-automatic #metrics #intra-app #inter-app #coherence
