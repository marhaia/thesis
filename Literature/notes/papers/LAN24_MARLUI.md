# MARLUI: Multi-Agent Reinforcement Learning for Adaptive Point-and-Click UIs (2024)

**Autoren:** Thomas Langerak, Sammy Christen, Mert Albaba, Christoph Gebhardt, Christian Holz, Otmar Hilliges  
**Quelle:** ACM TOCHI 2024  
**DOI:** 10.1145/3661147  
**PDF:** `Tier2_Sorgfaeltig/LAN24_MARLUI.pdf`

---

## Kernfrage
Wie kann man UI-Adaptierung (welche Elemente anzeigen?) als Multi-Agent RL Problem formulieren — ohne hand-crafted Regeln oder echte Nutzerdaten für Training?

## Methode
- Zwei Agenten: User-Agent (simuliert Nutzungsverhalten via point-and-click) + Interface-Agent (lernt Adaptierungspolitik)
- Turn-based Multi-Agent RL in geteilter Umgebung
- Trainiert ohne echte Nutzerdaten — Transfer auf reale Nutzer im gleichen Task

## Wichtigste Ergebnisse
- Interface-Agent lernt relevante Elemente anzuzeigen durch Beobachtung des User-Agenten
- Überträgt erfolgreich auf reale Nutzer in gleichen Tasks
- Ohne Heuristiken oder domain-spezifische Regeln generalisierbar
- Kernproblem: Intent-Inferenz (Was will der Nutzer?) als RL-gelöste Herausforderung

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2 — Task Descriptor / Intent)
- **Argument das es stützt:** Task Descriptor als Input für Stage 2 ist nicht trivial — MARLUI zeigt, dass Intent-Inferenz ein eigenständiges Forschungsproblem ist
- **Direkt zitierbar für:** "Die Herausforderung der Intent-Inferenz wird auch in MARLUI (Langerak et al., 2024) als zentrales Problem adaptiver UIs adressiert"

> **Was du lesen musst:** Abstract + Figure 1 (Architektur) + Section 4 (Evaluation) — ca. 20 Min.

## Kritik / Offene Fragen
- Point-and-click Paradigma — kein visueller Suchkontext
- Kein Cognitive Load Modell — reine Interaktionsoptimierung
- Intent ist binär (hat Ziel / hat kein Ziel) — dein Task Descriptor ist dimensionaler

## Verbindungen zu anderen Papern
- Ergänzt → SHI24 (supervisory control), GUO26 (task-conditioned scanpath)
- Parallele zu → LIA25 (Affordance + Intent)

---

**Tags:** #stage2 #RL #multi-agent #intent #adaptive-ui #task-descriptor
