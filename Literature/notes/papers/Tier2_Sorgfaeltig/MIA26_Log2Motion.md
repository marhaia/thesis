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

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★ relevant)
- Section 7.2 "Downstream Applications" direkt zitierbar als Motivation für Pre-Deployment Ansatz
- *"Miazga et al. (2026) identify automated ergonomic evaluation 'long before user studies are feasible' as a key downstream application of computational user modeling — the present pipeline operationalizes this vision specifically for the cognitive domain of in-vehicle HMI evaluation."*
- Stärker als die meisten anderen Papers für diesen Punkt — weil explizit "long before user studies" formuliert

### Kapitel 2: Related Work (★★ relevant)
- Section 2.2 "Forward Models of Task Performance" — Log2Motion ordnet sich in GOMS/KLM/RL-Tradition ein → deine Pipeline tut dasselbe für kognitive Evaluation
- Multi-Output: Speed + Accuracy + Effort aus einem Modell → stützt Stage 2 Multi-Output-Architektur
- *"Building on the forward modeling tradition in HCI (Miazga et al., 2026; Card et al., 1980), the present pipeline applies the same principle to the cognitive domain: instead of synthesizing motor behavior, Stage 2 generates simultaneous estimates of cognitive load, search efficiency, and attention demand from a single feature vector."*
- ⚠️ Abgrenzung: Biomechanik (motor control) ≠ Kognition (gaze-basiert) — verschiedene Modalitäten, gleiches epistemologisches Prinzip
- Kontextuell: Patrick Ebel (Letztautor) = Automotive HCI Domäne (Refs [22][23][24] im Paper) → Transfer ist nicht arbiträr

### Kapitel 5: Validierung (★★ relevant)
- Section 5.5 (DTW-Vergleich Simulation vs. echte Nutzer) als methodisches Vorbild für Validierungsdesign
- DTW-Abstände zwischen Log2Motion und echten Nutzern liegen innerhalb der Between-User-Variabilität → das ist das Argument das du für deine Pipeline-Validierung brauchst
- *"In a motion-capture comparison (N=6), Miazga et al. (2026) demonstrate that synthesized trajectories fall within the natural variability observed between human participants — an analogous validation criterion is adopted in the present study, where computational load estimates are assessed against within-subject gaze variability in the user study."*
- ⚠️ Argument-Struktur übernehmen, keine Zahlenwerte direkt übertragen

### Kapitel 6: Diskussion / Future Work (★ marginal)
- Section 7.3 Limitations-Struktur ("single-finger, table posture, no vision") → Vorlage für deine Limitations ("static screenshot, no real driving context, no individual calibration")
- Ref [33] im Paper: Fleig et al. (2025) "Mind & Motion" (arXiv) — verbindet biomechanische + kognitive Modelle direkt → als zusätzliche Future-Work-Referenz prüfen

> **Was du lesen musst:** Abstract + **Section 7.2** (Downstream Applications, 5 Min.) + **Section 5.5** (DTW-Vergleich, 10 Min.) + Section 7.3 Limitations überfliegen (3 Min.)  
> Sections 3–5.4 (technisches RL + Reward Design) weglassen.

## Kritik / Offene Fragen
- Touch-Interaktion ≠ Automotive Dashboard (kein physisches Touching beim Fahren)
- Biomechanisch sehr detailliert — dein Modell braucht diese Granularität nicht
- Transfer von motor → kognitive Evaluation ist konzeptuell, nicht methodisch direkt

## Verbindungen zu anderen Papern
- Methodisch → LIA26 (beide: computational simulation → in-situ Validierung)
- Domäne → EBE23 (Ebel Automotive HCI, selber Autor-Kontext)
- Multi-Output Parallele → BAI24, CHO26 (simultane Performance-Maße)
- Forward-Modeling-Tradition → OUL22, JOK20 (CR als Grundlage)

---

**Tags:** #stage2 #multi-output #biomechanics #RL #performance-estimation #forward-modeling #pre-deployment #CHI2026
