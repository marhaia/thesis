# Supporting Task Switching with Reinforcement Learning (2024)

**Autoren:** Alexander Lingler, Dinara Talypova, Jussi P.P. Jokinen, Antti Oulasvirta, Philipp Wintersberger  
**Quelle:** *CHI 2024*, Honolulu. DOI: 10.1145/3613904.3642063 | 18 Seiten  
**PDF:** `Tier3_Ueberfliegen/LIN24_Task_Switching_RL.pdf`

---

## Beim Ueberfliegen - gezielt lesen
> Note reicht fuer Related Work. PDF nur wenn du das CR-User-Model-Argument detailliert ausarbeitest.

**Direkt springen zu:**
- **Figure 3 AMS Results (ca. S. 8-10):** Vergleich CR-trainiert vs. unconstrained vs. kein AMS
- **Table 2 (ca. S. 9):** Quantitative Performance-Unterschiede
- **Section 4.2 Study Results:** CR-Model > unconstrained - das entscheidende Ergebnis

**Ueberspringen:** Section 2 (Balancing-Game-Details), Appendix

---

## Kernaussage
RL-basiertes Attention Management System (AMS) für Dual-Task-Szenarien. Schlüsselentscheidung: Das RL-Modell wird **nicht auf echten Nutzerdaten** trainiert, sondern auf **User Models** — davon eines CR-basiert (mit kognitiven Constraints wie WM-Limitierungen und Reaktionszeiten), eines ohne Constraints.

## Methodik
- Balancing-Game als Dual-Task-Testumgebung (zwei parallele Aufgaben, nur eine gleichzeitig kontrollierbar)
- AMS lernt Wechsel-Policy ohne manuelle Regeln
- **Zwei User Model Typen für Training:**
  1. Unconstrained Model — perfekte Wahrnehmung, sofortige Reaktion
  2. **Cognitive Rationality Model** — kognitive Constraints (Wahrnehmungsverzögerung, WM-Begrenzung, Reaktionszeit)
- Nutzerstudie: Vergleich AMS (unconstrained), AMS (CR-Model), Selbst-Switching, Fixiertes Switching

## Hauptbefunde
- CR-trainiertes AMS > unconstrained AMS für echte Nutzer
- Rein unconstrained Training → Frustration durch schlechte Synchronisation mit menschlichem Tempo
- CR-User-Models sind bessere Proxies für menschliches Verhalten beim RL-Training

## Relevanz für meine Thesis
- **Kapitel:** Related Work → Stage 2 Dual-Task / Methodology
- **Argument 1:** Direkte Bestätigung dass **CR-User-Models für RL-Training** geeignet sind — stärkt die Methodik der Pipeline
- **Argument 2:** Dual-Task (Fahren + Dashboard) ist strukturell identisch mit dem Balancing-Game; Transfer legitimiert den Automotive-Use-Case
- **Zitierbar für:** "Lingler et al. (2024) zeigen, dass RL-Modelle, die auf CR-basierten Nutzermodellen trainiert wurden, reale Nutzerperformance besser verbessern als unconstrained Modelle — ein Beleg für CR-User-Models als Trainings-Proxy"

> **Lesen nötig?** Nein — Note reicht. Wichtige methodische Unterstützung für Stage 2.

---

**Tags:** #dual-task #attention-management #RL #CR-user-model #training-proxy #automotive #oulasvirta
