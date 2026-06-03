# SeekUI: Predicting Visual Search Behavior on GUIs with a Reward-Augmented Vision Language Model (2026)

**Autoren:** Guo, Z., Jiang, Y., Leiva, L.A., Oulasvirta, A. et al.  
**Quelle:** ACM CHI 2026 (Aalto University / University of Luxembourg)  
**DOI:** https://dl.acm.org/doi/full/10.1145/3772318.3791178  
**PDF:** GUO2026.pdf  
**ID:** GUO26 | **Status:** 🟢 Analyzed (Missing Rabbit: KEEP — Stage 2/SOTA)  

---

## Kernfrage
Kann ein Reward-augmented Vision Language Model (VLM) visuelle Suchpfade auf GUIs vorhersagen — aus einem Screenshot + einem Textziel (Task Descriptor)?

## Methode
- Kombination: Vision Language Model + Reward-Augmentation
- Input: GUI-Screenshot + Text-Cue (Suchziel / Task Descriptor)
- Output: vorhergesagter Scanpfad (Sequenz von Fixationen)
- Validierung gegen menschliche Gaze-Daten

## Wichtigste Ergebnisse
- VLM kann visuelle Suchpfade aus Screenshot + Textbeschreibung des Ziels vorhersagen
- Reward-Augmentation verbessert die Genauigkeit gegenüber reinen VLM-Ansätzen
- Direkte Modellierung des Task-Descriptors als Eingabe

## Relevanz für meine Thesis
> Critical Hit: Directly predicts scanpaths from a screenshot + text cue (Task Descriptor). Validates Stage 2 logic perfectly.

- **Rolle:** SOTA-Vergleich — zeigt was der aktuelle Stand der Technik für Stage 2 ist
- **Kapitel:** Related Work / SOTA / Diskussion
- **Argument:** Belegt, dass die Kombination von visuellem Input + Task Descriptor für Scanpath-Prediction funktioniert (= mein Stage 2 Ansatz); mein Beitrag ist die explizite Kopplung mit Cognitive Load Index und Coherence Constraints
- **Direkt zitierbar für:** Task-conditioned scanpath prediction als validierter Ansatz

## Kritik / Offene Fragen
- VLM/Deep Learning Ansatz — ich nutze keinen CNN/VLM, sondern berechnete Features
- Wie verhält sich mein Ansatz im Vergleich zu diesem SOTA?
- Muss ich argumentieren warum mein Feature-basierter Ansatz trotzdem sinnvoll ist (Interpretierbarkeit, kognitive Fundierung)

## Verbindungen zu anderen Papern
- Direkte Verbindung → JIA23 (UEyes): gleiche Autoren-Gruppe (Jiang, Leiva, Oulasvirta)
- Ergänzt/konkurriert → JOK20 (Adaptive Feature Guidance): beide sagen Search Behavior vorher, aber unterschiedliche Methode
- Kontextualisiert → OUL18 (AIM): zeigt wohin sich das Feld seit AIM 2018 bewegt hat

---

**Tags:** #SOTA #scanpath #task-descriptor #VLM #visual-search #stage2 #reward-augmentation #2026
