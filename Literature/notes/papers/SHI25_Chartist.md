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
- Limitierung: noch nicht auf andere UI-Typen übertragen

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2 SOTA), Diskussion
- **Argument das es stützt:** Task-Konditionierung ist nicht optional — gleiche visuelle Komplexität → unterschiedliche Kognitiver Aufwand je nach Task. Chartist zeigt das empirisch für Charts, du zeigst es für Automotive HMIs.
- **Direkt zitierbar für:** "Chartist (Shi et al., 2025) demonstriert, dass task-konditionierte Scanpath-Modelle task-agnostischen überlegen sind"

> **Was du lesen musst:** Abstract + Figure 2 (Architektur) + Ergebnistabellen — ca. 25 Min.  
> **Besonders wichtig:** Section auf hierarchische Architektur — das ist die engste strukturelle Parallele zu deiner Pipeline.

## Kritik / Offene Fragen
- Chart-Visualisierungen ≠ Automotive HMI Dashboards
- LLM für Goal Selection = Black Box — nicht erklärbar
- ArXiv-Version (2502.03575) — Status peer-review prüfen bei Abgabe

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR), GUO26 (task-conditioned GUI)
- Ergänzt → SHI24 (gleiche Gruppe, Typing→Charts)
- Parallele zu → GUO26 (beide task-conditioned, beide RL+LLM)

---

**Tags:** #stage2 #SOTA #task-conditioned #scanpath #RL #LLM #chart
