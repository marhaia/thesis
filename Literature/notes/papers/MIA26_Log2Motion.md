# Log2Motion: Biomechanical Motion Synthesis from Touch Logs (2026)

**Autoren:** Michał Patryk Miazga, Hannah Bussmann, Antti Oulasvirta, Patrick Ebel  
**Quelle:** CHI 2026, ACM  
**DOI:** 10.1145/3772318.3790773  
**PDF:** `Tier2_Sorgfaeltig/MIA26_Log2Motion.pdf`

---

## Kernfrage
Wie kann man aus reinen Interaktionslogs (touch events) biomechanisch plausible Bewegungssequenzen synthetisieren — um Performance-Maße wie Speed, Accuracy und Effort zu schätzen?

## Methode
- RL-getriebene muskuloskelettale Vorwärtssimulation
- Software-Emulator integriert in Physik-Simulator
- Biomechanische Modelle manipulieren echte Apps in Echtzeit
- Validierung: Vergleich synthetisierter Bewegungen mit echten Nutzerbewegungen

## Wichtigste Ergebnisse
- Aus reinen Logs können Speed, Accuracy und Effort geschätzt werden
- Biomechanisch plausible Bewegungssequenzen ohne echte Motion-Capture-Daten
- Multi-Output: ein Modell → drei simultane Performance-Indikatoren
- Brücke zwischen digitalen Traces und physischen Handlungen

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Multi-Output Kohärenz, Stage 2 Architektur)
- **Argument das es stützt:** Multi-Output Vorhersage (Speed + Accuracy + Effort gleichzeitig) ist methodisch etabliert — unterstützt deine 3-Head-Architektur (Saliency + Fixation + CL-Index)
- **Direkt zitierbar für:** "Miazga et al. (2026) zeigen, dass ein einziges Modell simultane Performance-Indikatoren schätzen kann — analog zu unserer Multi-Head Stage 2"

> **Was du lesen musst:** Abstract + Figure 1 (Architektur) + Section 6 (Results) — ca. 20 Min.

## Kritik / Offene Fragen
- Touch-Interaktion ≠ Automotive Dashboard (kein physisches Touching beim Fahren)
- Biomechanisch sehr detailliert — dein Modell braucht diese Granularität nicht

## Verbindungen zu anderen Papern
- Parallele zu → BAI24 (multi-task attention switching), CHO26 (search time + effort)
- Ergänzt → SHI24 (supervisory control auch in Touching-Domäne)

---

**Tags:** #stage2 #multi-output #biomechanics #RL #performance-estimation #CHI2026
