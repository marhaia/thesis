# Projektplan: Two-Stage Multi-Output Pipeline
**Typ:** Projektplan / Gemini-Guidedokument  
**Status:** Aktiv — zentrale Architektur-Referenz

---

## Überblick / Research Question
> Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?

Das System schätzt **kognitive Belastungsindikatoren** aus Automotive-GUI-Screenshots — **ohne** Gleichsetzung mit NASA-TLX. Kernarchitektur: zwei Stufen + optionales Persönlichkeits-Conditioning.

**Wichtige Einschränkung:** Kein CNN wird aufgebaut. Stage 1 basiert auf berechneten Feature-Vektoren, kein End-to-End Deep Learning.

---

## Stage 1 — Task-Independent Visual Complexity Extraction

**Rationale:** Was trägt das Bild allein zur kognitiven Last bei — ohne Wissen über Task oder User?

**Output:** Visueller Komplexitätsvektor v ∈ ℝ⁸

| Feature | Beschreibung | Referenz |
|---------|-------------|---------|
| Shannon Entropy | Globale Informationsdichte (Graustufenhistogramm) | Shannon (1948) |
| Edge Density | Anteil Kantenpixel via Canny/Sobel | — |
| Information Clutter | Feature Congestion + Subband Entropy | Rosenholtz et al. (2007) |
| Layout Symmetry | Vertikale/horizontale Balance der Elemente | Miniukovich & De Angeli (2015) |
| Chromatic Coherence | Varianz Farbhistogramm, Palette-Konsistenz | — |
| Visual Hierarchy | Größengradient, Kontrast, Gruppierungsstruktur | Tuch et al. (2009) |
| Interactive Element Density | Anzahl Bedienelemente pro Fläche | — |

**Positionierung:** Features nicht neu — Neuheit liegt in ihrer Rolle als quantifizierte Baseline für Stage 2.

---

## Stage 2 — Task-Conditioned Multi-Output Prediction

**Input:** v (Stage 1) + Task Descriptor + User Profile (optional)  
**Output:** 3 simultane Outputs mit Mutual Coherence Constraints

| Head | Output | Validierung |
|------|--------|------------|
| Head 1 | Saliency Map (pixel-level) | DeepGaze II, SALICON |
| Head 2 | Fixation Distribution (Scanpath-Entropie) | Krejtz et al. (2016) |
| Head 3 | Cognitive Load Index 0–100 (Skalar) | NASA-TLX korreliert, NICHT gleichgesetzt |

**Loss-Funktion:**  
`L = λ₁L_sal + λ₂L_fix + λ₃L_load + λ_coh · L_coherence`

**Coherence-Term** bestraft inkonsistente Kombinationen (z.B. hohe Saliency-Dispersion + geringe Fixationsanzahl).

**Novelty:** Standard-Saliency-Modelle ignorieren Load; Standard-Load-Modelle ignorieren Attention. Die Kopplung ist der Beitrag.

---

## Input Layer

| Input | Wann | Format |
|-------|------|--------|
| GUI Screenshot | Stage 1 | Statisches Bild (automotive HMI) |
| Task Descriptor | Stage 2 | Kategorial oder dimensional |
| User Profile (Big Five) | Stage 2, optional | Normierter Vektor |

**Kritische Unterscheidung (User Profile):**  
- Trait-based Simulation (stabile Persönlichkeit) → methodologisch vertretbar  
- State-based Simulation (akuter Stress, Fatigue) → **NICHT validiert**, muss als Limitation deklariert werden  
→ Referenz: Kapania et al. (2025)

---

## Empirische Validierung

### Bestehendes Datensatz (Pilot)
- 32 Teilnehmer, Eye-Tracking + NASA-TLX
- Zwei Fahraufgaben (Lane-Change, Emergency Braking) unter zwei Bedingungen (mit/ohne Predictive Corridor)
- **Limitation:** Single GUI → keine Varianzdekomposition möglich
- Verwendung: Kalibrierung Stage 1, explorativ für Stage 2

### Neue Studie (Proposed)
- 30–35 Teilnehmer, within-subjects
- 8–10 automotive GUI Screenshots (variiert: minimalistisch bis informationsdicht)
- 2–3 Tasks pro GUI (Visual Search, Monitoring, Navigation)
- Highway-Szenario bei Konstantgeschwindigkeit
- ~24 Zellen/Teilnehmer, ~3–4h pro Person

### Baseline-Vergleiche (Pflicht)
| Baseline | Beschreibung |
|---------|-------------|
| Baseline 1 | Stage 1 allein (lineare Regression) |
| Baseline 2 | Stage 2 ohne Coherence Constraints (λ_coh = 0) |
| Baseline 3 | Single-Output (nur Load-Head) |

### Key Results
- **Result 1:** Varianzdekomposition Stage 1 vs. Stage 2 → primärer Beitrag
- **Result 2:** Coherence Constraints vs. Baseline 2 & 3
- **Result 3:** Null-Ergebnisse explizit publizierbar

---

## Epistemic Framing — Wichtige Grenzen

### Safe Claims
- Output = "computational index of expected interactional complexity"
- Korreliert mit NASA-TLX unter definierten Bedingungen
- Screening-Tool, kein Ersatz für empirische Evaluation

### Overclaim-Risiken (vermeiden!)
- ❌ "predicted NASA-TLX" — falsche Gleichsetzung
- ❌ "misst kognitive Last" — misst nur Proxy
- ❌ Cross-domain Generalisierung ohne eigene Validierung
- ❌ Als Ersatz für User Studies darstellen

### Explizite Limitationen
- State-based Stress-Simulation nicht validiert
- Single-domain (Automotive)
- TLX-Subskalen ohne visuellen Korrelat (Physical Demand, Frustration) werden nicht modelliert
- GTA-imputierte TLX-Daten nur exploratorisch, nicht als primäre Validierung

---

## Referenzen aus dem Dokument
- Argyle et al. (2023) — LLMs simulieren demografische Profile
- Chen et al. (2018) — Tappability-Prediction (Single-Output Baseline)
- Hart & Staveland (1988) — NASA-TLX
- Jiang et al. (2015) — SALICON Dataset
- Kapania et al. (2025) — Epistemische Grenzen LLM-User-Simulation
- Krejtz et al. (2016) — Transition Entropy, Eye Tracking + Cognitive Load
- Kümmerer et al. (2017) — DeepGaze II
- Li et al. (2020) — Automated Usability Evaluation
- Miniukovich & De Angeli (2015) — Interface Aesthetics / Symmetry
- Oulasvirta et al. (2022) — Computational Rationality
- Rosenholtz et al. (2007) — Visual Clutter
- Serapio-García et al. (2023) — Big Five in LLMs
- Shannon (1948) — Informationstheorie
- Taubman-Ben-Ari et al. (2004) — Driving Style & Personality
- Tuch et al. (2009) — Visual Complexity of Websites
