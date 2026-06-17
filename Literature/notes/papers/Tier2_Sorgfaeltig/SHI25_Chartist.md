# Chartist: Task-driven Eye Movement Control for Chart Reading (2025)

**Autoren:** Danqing Shi, Yao Wang, Yunpeng Bai, Andreas Bulling, Antti Oulasvirta  
**Quelle:** CHI 2025, ACM  
**DOI:** 10.1145/3706598.3713285 (Konferenz-Version, arXiv: 2502.03575)  
**PDF:** `Tier2_Sorgfaeltig/SHI25_Chartist.pdf`

---

## Kernfrage
Wie kann man task-getriebene Augenbewegungen beim Lesen von Diagrammen computational modellieren — über verschiedene Analyse-Tasks (Wert abrufen, Filtern, Extremwerte finden)?

## Methode
- Zweistufige hierarchische Architektur:
  - High-Level: LLM wählt Ziel basierend auf bisherigem Informationsgewinn
  - Low-Level: RL-Policy steuert Augenbewegungen (Sampling Policy)
- Task als expliziter Input — kein task-agnostisches Modell
- Validiert an menschlichen Scanpath-Daten auf Diagrammen

## Wichtigste Ergebnisse
- Modell sagt task-abhängige Scanpaths vorher — gleiche Visualisierung, unterschiedliche Tasks → unterschiedliche Gaze-Muster
- LLM als High-Level Goal Selection funktioniert ohne domain-spezifisches Training
- Generalisiert über verschiedene Chart-Typen (bar, line, scatter)
- Limitierung: noch nicht auf andere UI-Typen übertragen (selbst so benannt)
- **Empirischer Schlüsselbefund:** Task-agnostische Modelle (UMSS, DeepGaze III) scheitern bei task-abhängigen AOI-Ratios — nur Chartist repliziert die human-like Verteilung → task-Konditionierung ist nicht optional
- **Wichtig für dich:** Weniger als 20% der Fixationen landen auf task-relevanten AOIs — der Rest ist Informationssammlung/Bestätigung. Das bedeutet: Fixationsanzahl allein ist kein gutes Maß für kognitive Last → stärkt deinen separaten cognitive_load_score Ansatz

## Verwendung in der Thesis

### Kapitel 1: Einleitung (optional)
- 1 Satz als empirische Motivation für task-Konditionierung
- *"Recent work demonstrates that identical visual interfaces produce systematically different gaze patterns depending on the active task (Shi et al., 2025), motivating the task-conditioned cognitive load assessment developed in this thesis."*

### Kapitel 2: Related Work (★★★ zentral)
- SOTA-Block für task-konditionierte Modelle, zusammen mit GUO26
- Chartist + GUO26 zusammen = domänenübergreifender Beleg dass task-Konditionierung notwendig ist
- *"Shi et al. (2025) demonstrate in the chart reading domain that task-conditioned models outperform task-agnostic baselines across all scanpath similarity metrics; the same principle of task-conditioned prediction underlies the Task Descriptor mechanism in Stage 2 of the present pipeline."*
- Abgrenzung: SHI25 = online, LLM zur Laufzeit, Black Box; du = offline, expliziter Task Descriptor, interpretierbar — bewusster Trade-off für Safety-Critical Context
- ⚠️ Architektur-Parallele nur konzeptuell formulieren ("conceptually analogous") — SHI25's Hierarchie ist echtes HRL (Eppe et al. 2022), dein Task Descriptor ist ein statischer Input, kein lernender Controller

### Kapitel 3: Methodik (★★ relevant)
- 1 Satz bei Einführung des Task Descriptors
- *"Empirical evidence from task-conditioned eye movement models (Shi et al., 2025; Guo et al., 2026) confirms that task context is a necessary input for accurate behavioral prediction, motivating the explicit Task Descriptor in Stage 2."*

### Kapitel 6: Diskussion (★★ relevant)
- Limitation: LLM vs. expliziter Descriptor
- *"Shi et al. (2025) employ an LLM for high-level goal selection, enabling flexible generalization at the cost of interpretability. The present pipeline foregoes LLM-based inference in favor of an explicit, auditable Task Descriptor — a necessary constraint for safety-critical automotive deployment."*

> **Was du lesen musst:** Abstract + Figure 2 (Architektur) + Table 1 Ergebnisse + Limitations letzter Absatz — ca. 25 Min.  
> **Besonders wichtig:** Table 1 Zeile "Task AOIs ratio" — hier sieht man direkt warum task-agnostische Modelle versagen.

## Kritik / Offene Fragen
- Chart-Visualisierungen ≠ Automotive HMI Dashboards
- LLM für Goal Selection = Black Box — nicht erklärbar, nicht safety-zertifizierbar
- Hierarchische Architektur basiert auf HRL (Eppe et al. 2022) — kein klassischer Supervisory Control Ansatz, "supervisory control" in SHI25 ist metaphorisch verwendet
- ✅ CHI 2025 DOI bestätigt: peer-reviewed (arXiv 2502.03575 ist Preprint-Version, inhaltlich identisch prüfen bei Abgabe)

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR), GUO26 (task-conditioned GUI)
- Ergänzt → SHI24 (gleiche Gruppe, Typing→Charts)
- Parallele zu → GUO26 (beide task-conditioned, beide RL+LLM)

---

**Tags:** #stage2 #SOTA #task-conditioned #scanpath #RL #LLM #chart
