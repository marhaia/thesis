# CRTypist: Simulating Touchscreen Typing Behavior via Computational Rationality (2024)

**Autoren:** Danqing Shi, Yujun Zhu, Jussi P.P. Jokinen, Aditya Acharya, Aini Putkonen, Shumin Zhai, Antti Oulasvirta  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3642918  
**PDF:** `Tier2_Sorgfaeltig/SHI24_CRTypist.pdf`

---

## Kernfrage
Wie kann man menschliches Tippverhalten auf Touchscreens durch CR modellieren — über verschiedene Keyboards und Nutzer hinweg?

## Methode
- Supervisory Control Reformulierung: Visuelle Aufmerksamkeit + Motorsystem werden über Working Memory gesteuert
- Arbeitet direkt auf Pixeln (kein Feature Engineering)
- RL für Bewegungsplanung (asymptotisch optimal unter kognitiven Bounds)
- Validiert an Tipp-Daten: Bewegungen, Performance, individuelle Unterschiede

## Wichtigste Ergebnisse
- Modell generiert human-like Tippverhalten ohne Keyboard-spezifisches Hand-Crafting
- Deckt individuelle Unterschiede ab (kein one-size-fits-all)
- Generalisiert auf verschiedene Keyboard-Designs
- Working Memory als zentrales Koordinationsmedium zwischen Sehen und Motorik

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Stage 2 Architektur)
- **Argument das es stützt:** Supervisory Control Architektur (Auge + Motor = koordiniertes System) rechtfertigt deine Multi-Head Kopplung in Stage 2
- **Direkt zitierbar für:** "CRTypist (Shi et al., 2024) zeigt, dass supervisory control von visueller Aufmerksamkeit und Motorik durch ein gemeinsames Working-Memory-Modell koordiniert werden kann"

> **Was du lesen musst:** Abstract + Section 2 (Supervisory Control Formulation) + Figure 2 (Architektur) — ca. 25 Min.

## Kritik / Offene Fragen
- Nur skilled typists — kein Novice-Modell
- Fokus auf Typing, nicht auf visuelle Suche in Dashboards
- Working Memory Parameter schwer auf Automotive-Kontext übertragbar

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR), JOK20 (supervisory control)
- Ergänzt → LAN24 (Intent Inferenz), BAI24 (Attention Switching)
- Parallele zu → CHE21 (RL für Augen-Motor-Koordination)

---

**Tags:** #stage2 #CR #RL #supervisory-control #typing #computational-model
