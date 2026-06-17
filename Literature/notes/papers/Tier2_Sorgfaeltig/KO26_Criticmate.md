# Criticmate: Stagewise Human–AI Co-Critique in Single-Screen UI Evaluation (2026)

**Autoren:** Jisu Ko, Jinyoung Choi, Cielo Morales, Dajung Kim, Minsam Ko  
**Quelle:** CHI 2026, ACM  
**DOI:** 10.1145/3772318.3790929  
**PDF:** `Tier2_Sorgfaeltig/KO26_Criticmate.pdf`

---

## Kernfrage
Wie kann man KI-gestützte UI-Evaluation so strukturieren, dass sie menschliche Kritik qualitativ verbessert — statt Evaluation als Black-Box Single-Pass zu behandeln?

## Methode
- Situation Awareness (SA) Theory als Rahmen
- 3 Stufen: Perception (was ist da?) → Comprehension (was bedeutet es?) → Projection (was folgt daraus?)
- System Criticmate: jede Stufe editierbar, KI produziert Zwischen-Artefakte
- Evaluation: Offline Benchmark + Controlled User Study (Expert-Ratings)

## Wichtigste Ergebnisse
- Stagewise Co-Critique produziert expert-ähnlichere Kritiken als Single-Pass KI
- Bessere Balance zwischen Stärken und Problemen
- Höheres Nutzervertrauen ohne Verlust von wahrgenommener Autonomie
- Struktur Perception → Comprehension entspricht Bottom-Up → Top-Down

## Relevanz für meine Thesis
- **Kapitel:** Related Work (Pipeline-Justifikation), Diskussion
- **Argument das es stützt:** Die Aufteilung in Perception-Stage (= dein Stage 1: was ist visuell salient?) und Comprehension-Stage (= dein Stage 2: was bedeutet das kognitiv?) ist unabhängig validiert
- **Direkt zitierbar für:** "Criticmate (Ko et al., 2026) bestätigt unabhängig den zweistufigen Ansatz — Perception zuerst, Comprehension danach"

> **Was du lesen musst:** Abstract + Section 2 (SA Theory) + Figure 2 (3 Stufen) — ca. 20 Min.

## Kritik / Offene Fragen
- Fokus auf heuristische Evaluation (Usability-Probleme), nicht auf quantitative Metriken
- Mobile UIs, nicht Automotive

## Verbindungen zu anderen Papern
- Strukturelle Parallele → deine Two-Stage Pipeline
- Ergänzt → OUL18 (AIM als Stage 1), DAS24 (Cognitive Load als Stage 2)

---

**Tags:** #pipeline-justification #two-stage #UI-evaluation #situation-awareness #human-AI
