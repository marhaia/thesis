# Context-Aware Online Adaptation of Mixed Reality Interfaces (2019)

**Autoren:** David Lindlbauer, Anna Maria Feit, Otmar Hilliges (ETH Zürich)  
**Quelle:** *UIST 2019*  
**Anmerkung:** REF_ Prefix = nur als Referenz, nicht in Literatursuche gefunden  
**PDF:** `Tier3_Ueberfliegen/REF_LIN19_MixedReality.pdf`

---

## Kernaussage
Optimierungsbasierter Ansatz für MR-Systeme: entscheidet automatisch **wann**, **wo** und **wie viel** jede Anwendung angezeigt wird — basierend auf aktuellem Cognitive Load (gemessen via Index of Pupillary Activity), Task und Umgebung.

## Methodik
- **Input:** Kognitive Last (Pupillen-Index), Task-Kontext, Umgebungsgeometrie
- **3-Schritt-Prozess:**
  1. Rule-based: View-anchored vs. World-anchored Platzierung
  2. **Integer Linear Program** → optimiert Sichtbarkeit (visible/hidden) und Level-of-Detail (LOD) für alle Elemente gleichzeitig
  3. Greedy-Algorithmus → Platzierungsoptimierung für view-anchored Elemente
- Inhalt-Designer spezifizieren Elemente + LOD-Varianten; System adaptiert zur Laufzeit

## Hauptbefunde
- Low CL → mehr Elemente, höheres LOD
- High CL → minimales UI (nur kritische Elemente, niedrigstes LOD)
- Manuelles Anpassen von 12 Anwendungen bei jedem Context-Switch wäre für Nutzer unzumutbar → Automation notwendig

## Relevanz für meine Thesis
- **Kapitel:** Related Work → Adaptive UI / Cognitive Load als Steuerungsgröße
- **Argument:** Zeigt dass Cognitive Load als **Output-Metrik** direkt für UI-Adaptation verwendbar ist — nicht nur als theoretisches Konstrukt. Unterstützt Stage 2 Output Head "Cognitive Load Estimate"
- **Methodischer Brücken-Beitrag:** Kombinatorische Optimierung mit CL als Constraint = konzeptuell verwandt mit unserer Optimierungsformulierung in Stage 2
- **Zitierbar für:** "Lindlbauer et al. (2019) demonstrieren, dass Cognitive Load als Echtzeitgröße UI-Adaptation steuern kann — ein Beleg für den praktischen Nutzen von CL-Vorhersagen, wie sie Stage 2 liefert"

> **Lesen nötig?** Nein — Note reicht als unterstützende Referenz.

---

**Tags:** #adaptive-UI #cognitive-load #MR #optimization #ILP #reference-only
