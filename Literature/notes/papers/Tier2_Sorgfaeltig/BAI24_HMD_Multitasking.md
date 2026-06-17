# Heads-Up Multitasker: Simulating Attention Switching on Optical Head-Mounted Displays (2024)

**Autoren:** Yunpeng Bai, Aleksi Ikkala, Antti Oulasvirta, Shengdong Zhao, Lucia J Wang, Pengzhi Yang, Peisen Xu  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3642540  
**PDF:** `Tier2_Sorgfaeltig/BAI24_HMD_Multitasking.pdf`

---

## Kernfrage
Wie kann man Aufmerksamkeitswechsel zwischen zwei simultanen Tasks (Lesen auf OHMD + Gehen) computational modellieren?

## Methode
- Hierarchisches RL: Supervisory Controller optimiert Aufmerksamkeitsallokation
- High-Level: Wann zwischen Reading und Walking wechseln?
- Low-Level: Wie Augen/Körper bewegen?
- Validierung: Modell-Vorhersagen vs. empirische Gaze-Daten beim Gehen

## Wichtigste Ergebnisse
- Aufmerksamkeitswechsel emergiert optimal aus hierarchischem RL
- Supervisory Control erklärt wann Nutzer zwischen Tasks wechseln
- Modell repliziert menschliche Multitasking-Patterns (Timing, Frequenz der Switches)
- Kognitive Kosten des Wechsels sind implizit im Reward kodiert

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2 — Task Interleaving, Sekundäraufgabe beim Fahren)
- **Argument das es stützt:** Automotive Dual-Task Szenario (Fahren + Dashboard lesen) ist strukturell identisch mit OHMD-Multitasking — Supervisory Control erklärt Glance-Verhalten
- **Direkt zitierbar für:** "Die Dual-Task Struktur im Fahren (Primäraufgabe: Fahren, Sekundäraufgabe: Dashboard) entspricht dem von Bai et al. (2024) modellierten Attention-Switching-Problem"

> **Was du lesen musst:** Abstract + Figure 1 (Architektur) + Section 5.1 (Ergebnisse Attention Switching) — ca. 20 Min.

## Kritik / Offene Fragen
- OHMD ≠ Automotive Dashboard (andere Modalität, anderer physischer Kontext)
- Gehen ≠ Fahren als Primäraufgabe (Automatisierungsgrad unterschiedlich)

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR), JOK21 (Fahrer Adaptation — direktere Automotive-Referenz)
- Parallele zu → JOK21 (gleiches Multitasking-Problem, anderer Kontext)
- Ergänzt → SHI24 (supervisory control), LIN24 (task switching)

---

**Tags:** #stage2 #multitasking #attention-switching #supervisory-control #RL #automotive-parallel
