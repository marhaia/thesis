# UIClip: A Data-driven Model for Assessing User Interface Design (2024)

**Autoren:** Jason Wu, Yi-Hao Peng, Xin Yue Li, Amanda Swearngin, Jeffrey P. Bigham, Jeffrey Nichols (CMU / Apple)  
**Quelle:** *UIST 2024*, Pittsburgh. DOI: 10.1145/3654777  
**PDF:** `Tier3_Ueberfliegen/WU24_UIClip.pdf`

---

## Kernaussage
**UIClip**: CLIP-Modell, fine-getuned auf UI-Screenshots + Qualitätslabels. Bewertet UI-Design-Qualität und visuelle Relevanz aus Screenshot + natürlichsprachlicher Beschreibung. Übertrifft alle getesteten Large Vision-Language Models (LVLMs) — trotz kleinerer Modellgröße.

## Methodik
- **Zwei Trainingsdatensätze:**
  1. **JitterWeb** (synthetisch): Webseiten-Screenshots + automatisch generierte Designfehler (Color Swap, Spacing Noise, Layout-Änderungen, Complexity-Reduktion) + LLM-generierte Beschreibungen
  2. **BetterApp** (human-rated): iOS/Android App Screenshots, echte Designfehler, menschliche Designer-Ratings — aus VINS-Dataset
- **Training:** Contrastive Learning (CLIP-Paradigma) auf Qualitäts-Rankings
- **Evaluation:** 3 Tasks — Design Quality, Improvement Suggestions, Design Relevance
- Vergleich gegen: CLIP (off-the-shelf), diverse LVLMs

## Hauptbefunde
- UIClip **schlägt alle anderen Modelle** in allen 3 Tasks
- Off-the-shelf CLIP scheitert an UI-spezifischer Qualitätsbewertung → Domain Fine-Tuning essentiell
- Anwendungen: (1) Quality-aware UI Code Generation, (2) Design Suggestion Generation, (3) Quality-aware UI Retrieval

## Relevanz für meine Thesis
- **Kapitel:** Related Work → Kontrastpaper (Black-Box vs. CR)
- **Argument:** UIClip repräsentiert den rein datengetriebenen Gegenansatz zu unserer CR-Pipeline:
  - UIClip: 1 Qualitätsscore (was?), keine Verhaltensvorhersage, kein Task, kein User-Profil
  - Unsere Pipeline: erklärbare Prädiktionen (wie/warum), task-konditioniert, nutzerabhängig
- **Direkte Konkurrenz/Baseline für Stage 1:** UIClip + unser Feature-Vektor sollten verglichen werden
- **Zitierbar für:** "Im Gegensatz zu Black-Box-Ansätzen wie UIClip (Wu et al., 2024), die einen einzelnen Qualitätsscore ohne Verhaltensmodell liefern, erzeugt unsere Pipeline erklärbare, nutzer- und taskspezifische Prädiktionen"

> **Lesen nötig?** Ja (Tier 3 must-read) — für Related Work Abgrenzung und als mögliche Baseline in der Evaluation.

---

**Tags:** #stage1 #black-box #CLIP #baseline #contrast #data-driven #UIST
