# Development Status — Two-Stage GUI Complexity Pipeline
**Thesis:** *Two-Stage Multi-Output Pipeline: Computational Estimation of Interactional Complexity from GUI Screenshots*  
**Stand:** 03.06.2026 · Version 3.0  
**Für:** Supervisor-Meeting

---

## 1. Was ist gebaut (aktueller Stand)

### Gesamtarchitektur

```
Screenshot (PNG/JPG)
       │
       ▼
┌─────────────────────────────────┐
│  STAGE 1a — Visual Complexity   │  ← FERTIG ✅
│  visual_complexity.py           │
│  v ∈ ℝ⁸                        │
│  8 Metriken (Entropy, Edge,     │
│  Congestion, Symmetry, ...)     │
└────────────┬────────────────────┘
             │
       ┌─────┴──────┐
       ▼            ▼
┌─────────────┐  ┌──────────────────────────────────┐
│  STAGE 1b   │  │  STAGE 1c — Jokinen 2020          │
│  UMSI++     │  │  Cognitive Search Model            │  ← FERTIG ✅
│  Saliency   │  │  jokinen_model.py                  │
│  s ∈ ℝ⁵    │  │  Monte Carlo (EMMA + VSTM)          │
│  (FERTIG ✅)│  │  → predicted search time / element │
└──────┬──────┘  └──────────────┬───────────────────┘
       │                        │
       └───────────┬────────────┘
                   ▼
       ┌───────────────────────────┐
       │  HCEye Feature Layer      │  ← FERTIG ✅
       │  hceye_features.py        │
       │  h ∈ ℝ⁶                   │
       │  (Sensitivitäts-Gewichte  │
       │   aus Das et al. 2024)    │
       └───────────┬───────────────┘
                   │
       ┌───────────┴───────────────┐
       │  Task Descriptor          │  ← FERTIG ✅
       │  task_descriptor.py       │
       │  t ∈ ℝ⁵                   │
       │  (Aufgabentyp, Spezifität,│
       │   Zeitdruck)              │
       └───────────┬───────────────┘
                   │
       ┌───────────┴──────────────────────────┐
       │  STAGE 2 — Cognitive Load Output     │
       │                                       │
       │  DEFAULT PATH (aktuell aktiv):        │  ← FERTIG ✅
       │  HCEye-Koeffizienten +               │
       │  Task-Descriptor-Adjustierung        │
       │  → cognitive_load_score (0–100)      │
       │                                       │
       │  EXPERIMENTAL:                        │  ← TEILWEISE ⚠️
       │  Ridge / XGBoost Regression          │
       │  (training data noch dünn)           │
       └──────────────────────────────────────┘
```

---

## 2. Fertige Komponenten ✅

| Modul | Beschreibung | Status |
|-------|-------------|--------|
| `stage1/visual_complexity.py` | 8 visuelle Komplexitätsmetriken aus Screenshot | ✅ Fertig |
| `stage1/app.py` | Flask-API (Port 5001) mit allen Endpoints | ✅ Fertig |
| `saliency/umsi_model.py` | UMSI++ Saliency-Modell (TF2/Keras3, CPU) | ✅ Fertig |
| `saliency/saliency_features.py` | 5 Saliency-Feature-Aggregationen | ✅ Fertig |
| `cognitive/jokinen_model.py` | Kognitives Such-Modell (Jokinen 2020) | ✅ Fertig |
| `cognitive/element_detector.py` | UI-Element-Detektor (Basis) | ✅ Fertig |
| `hceye/hceye_features.py` | HCEye Sensitivitäts-Schicht | ✅ Fertig |
| `stage2/task_descriptor.py` | Aufgaben-Kontext-Kodierung | ✅ Fertig |
| `stage2/user_profile.py` | Big-Five-Persönlichkeits-Preset | ✅ Fertig |
| `stage2/regression_model.py` | Ridge/RF Regressor (Grundgerüst, korrekte Feature-Keys) | ⚠️ Gerüst da, kein Training |
| `stage2/coherence_check.py` | Mutual Coherence Check (3 Regeln, wissenschaftlich referenziert) | ✅ Fertig |
| `tests/test_pipeline.py` | End-to-End Pipeline Test | ✅ Vorhanden |
| `stage1/ui/index.html` | Web-UI: Steps, Score Breakdown, Saliency Heatmap, Coherence Banner | ✅ Fertig |
| `scripts/baseline_comparison.py` | Konfigurationsvergleich A vs B vs C | ✅ Fertig |
| `scripts/hceye_condition_comparison.py` | HCEye Bedingungsvergleich (H1 p=2.55e-28, d=1.47) | ✅ Fertig |
| `scripts/ueyes_saliency_validation.py` | UMSI++ vs UEyes GT (CC=0.896, SIM=0.806) | ✅ Fertig |

### Aktive API-Endpoints

| Endpoint | Funktion |
|----------|----------|
| `POST /api/analyze` | Visual Complexity v ∈ ℝ⁸ |
| `POST /api/saliency` | UMSI++ Saliency Map + s ∈ ℝ⁵ + heatmap_png_base64 |
| `POST /api/search-time` | Jokinen Kognitive Suchzeit |
| `POST /api/cognitive-load` | Gesamt-Cognitive-Load-Score + saliency_overlay_b64 + coherence |

---

## 3. Was noch fehlt / In Arbeit ⚠️

### Kritisch (für Thesis-Abgabe nötig)

| Was | Warum wichtig | Aufwand-Schätzung |
|-----|--------------|-------------------|
| **Stage 2 Training** | Regression läuft im Default-HCEye-Modus; kein empirisch trainiertes Modell | M (1 Woche) |
| **Jokinen-Modell Validierung** | Monte-Carlo-Parameter sind Literaturwerte, nicht empirisch kalibriert | M (1 Woche) |
| **Experten-Validierung (Option 3)** | Spearman-Rankkorrelation gegen 3–5 UX-Experten | S (2–3 Tage) |

### Wichtig (für Qualität / Vollständigkeit)

| Was | Warum wichtig | Aufwand-Schätzung |
|-----|--------------|-------------------|
| **Element-Detektor verbessern** | Aktuell regelbasiert; ML-basierter Detektor würde Jokinen verbessern | L |
| **User Study Integration** | Task-Descriptor-Gewichte sind Literatur-Ordinalwerte, keine empirischen | XL |
| **Test-Coverage erhöhen** | Nur 1 End-to-End-Test, keine Unit-Tests | S |
| **UEyes NSS mit binären Fixationskarten** | Vollständiger Datensatz von Zenodo:8010312 benötigt | M |

### Nice-to-have (wenn Zeit bleibt)

| Was | Anmerkung |
|-----|-----------|
| PathGAN++ TF2-Port | Weights sind da, Architektur fehlt |
| Eye-Tracking-Integration | `hceye/gaze/` hat Daten, aber keine direkte Pipeline-Anbindung |
| Docker-Compose vollständig | AIM-Backend läuft, aber nicht mit Stage1/2 zusammen |

---

## 4. Aktuelle Limitations (ehrlich)

1. **Kein empirisches Training:** Stage 2 läuft im rule-based Modus. Die Regression ist ein Gerüst, aber ohne Ground-Truth-Daten nicht trainiert.
2. **Element-Detektor ist limitiert:** Jokinen-Modell-Genauigkeit hängt stark von der Qualität der UI-Element-Erkennung ab.
3. **Task-Gewichte sind ordinal, nicht kalib.:** `task_descriptor.py` verwendet Literatur-basierte Ordinalwerte — sollten durch Nutzerstudie kalibriert werden.
4. **UMSI++ läuft auf CPU:** Inferenz ist langsam (~5–15 Sek/Bild). GPU-Support möglich aber nicht eingerichtet.
5. **UEyes-NSS nicht vollständig:** CC=0.896 auf 5 Sample-Bildern; NSS-Validierung erfordert binäre Fixationsmaps (Zenodo:8010312).

---

## 5. Validierungsstand (neu)

| Komponente | Methode | Ergebnis |
|---|---|---|
| HCEye Koeffizienten (hceye_features.py) | Bedingungsvergleich H1/H2 (scripts/hceye_condition_comparison.py) | H1: p=2.55e-28, d=1.47 (**large**); Vorhersage 0.876 vs. empirisch 0.870 (Δ=0.6%) ✅ |
| UMSI++ Saliency (saliency/umsi_model.py) | UEyes CC/SIM (scripts/ueyes_saliency_validation.py) | CC=0.896, SIM=0.806 (weit über SOTA 0.40–0.70) ✅ |
| Gesamt-Pipeline CL-Score | HCEye Bedingungsvergleich (empirisch) | Rule-based; kein trainiertes Modell, Validierung indirekt |
| Big Five Modifier (user_profile.py) | Literaturbasiert | Serapio-García 2023, Taubman-Ben-Ari 2004 — kein direkter empirischer Test |

---

## 6. Nächste konkrete Schritte (priorisiert)

1. Supervisor-Meeting: Validierungsstrategie klären (Option 2+3 kombiniert)
2. Experten-Validierung (3–5 UX-Experten, Spearman ρ)
3. UEyes vollständigen Datensatz herunterladen → NSS mit binären Fixationskarten
4. Stage 2 mit HCEye-Verhaltensdaten trainieren (auch wenn klein)

---

## 7. Fragen / Diskussionspunkte für Supervisor

- [ ] Reicht HCEye-Bedingungsvergleich + UEyes-CC als Validierungsnachweis?
- [ ] Experten-Validierung als ergänzende externe Validierung ausreichend?
- [ ] Ist UMSI++ als Saliency-Modell ausreichend, oder soll PathGAN++ noch portiert werden?
- [ ] Zeitplan: Ist Abgabe bis [DATUM] realistisch?

---

*Zuletzt aktualisiert: 03.06.2026*
