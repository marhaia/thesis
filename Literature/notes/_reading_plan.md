# Leseplan — Priorisierung nach RQ & Projektplan

**RQ:** Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?  
**Pipeline:** Stage 1 (Visual Complexity Vector) → Stage 2 (Saliency + Fixation + Cognitive Load Index)

---

## Tagesplan — Tier 1 + Tier 2 (27 Paper, 5/Tag)

> **Ziel:** 6 Tage · Tier 1 zuerst vollständig, dann Tier 2 thematisch gebündelt  
> **Tempo:** Tier 1 = ca. 45–60 min/Paper · Tier 2 = ca. 30–40 min/Paper  
> **Beim Lesen:** Abstract → Methodik → Ergebnisse → Discussion/Limitations → Note ergänzen

| Tag | Paper | Thematischer Fokus |
|-----|-------|--------------------|
| **Tag 1** | OUL18 · OUL22 · JOK20 · DAS24 · JIA23 | Tier 1 komplett Block 1 — AIM + Theorie + Kernempirie |
| **Tag 2** | GUO26 · KANK17 · JOK21 · CHE21 · SHI24 | Tier 1 abschließen + Stage 2 CR-Kern |
| **Tag 3** | LAN24 · SHI25 · LIA25 · TOD19 · TOD16 | Stage 2 Architektur + Layout-Optimierung |
| **Tag 4** | LOR24 · BUR21 · LIA26 · MIA26 · KO26 | Validierung + Pipeline-Methodik |
| **Tag 5** | JIE24 · SAR16 · GAJ08 · ZHA24 · BAI24 | Personality Layer + Ability-Based Modeling |
| **Tag 6** | TOD18 · KRE16 | Tier 2 abschließen — Lernkurve + Eye Tracking CL |

---

### Tag 1 — AIM-Plattform + Theoretisches Fundament
**Lesefokus:** Was genau macht AIM? Was ist CR? Warum verändert CL die Saliency?

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **OUL18** AIM | Welche Metriken hat AIM aktuell? Welche fehlen? Was ist die API-Schnittstelle? |
| 2 | **OUL22** Computational Rationality | Wie funktioniert das POMDP-Modell? Was ist der Unterschied zu GOMS/ACT-R? |
| 3 | **JOK20** Adaptive Feature Guidance | Wie wird Predicted Search Time berechnet? Welche Parameter? |
| 4 | **DAS24** HCEye | Welches Experiment? Welche CL-Stufen? Welche Saliency-Shifts? Statistik? |
| 5 | **JIA23** UEyes | Welche UI-Typen? Wie werden Saliency-Maps validiert? N=? |

---

### Tag 2 — SOTA + CR in Automotive
**Lesefokus:** Was kann GUO26 was AIM nicht kann? Wie ist KANK17 auf Stage 2 übertragbar?

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **GUO26** SeekUI | VLM-Architektur? Reward-Signal? Benchmark? Direkt vergleichbar mit deiner Pipeline? |
| 2 | **KANK17** ABC | Wie wird das Modell kalibriert? Was sind die freien Parameter? Wie auf Stage 2 übertragen? |
| 3 | **JOK21** Drivers Adaptation | Wie wird CR auf Automotive angewendet? Gleiche Struktur wie JOK20? |
| 4 | **CHE21** Gaze Selection | RL-Modell: Welche States/Actions? Wie modelliert es Fixationsdauer? |
| 5 | **SHI24** CRTypist | Supervisory Control: Wie wird Vision + Motor entkoppelt? |

---

### Tag 3 — Stage 2 Architektur + Layout-Optimierung
**Lesefokus:** Wie bauen andere Multi-Output-Systeme? Was ist die Gewichtungsformel?

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **LAN24** MARLUI | Multi-Agent RL für UI: Wie wird Intent inferiert? Verbindung zu Task Descriptor? |
| 2 | **SHI25** Chartist | Task-konditionierte Scanpaths: Wie werden Tasks kodiert? LLM-Integration? |
| 3 | **LIA25** Affordance CR | Wie verbindet das Paper Stage 1 Features mit Stage 2 Actions? |
| 4 | **TOD19** Individualising Layouts | Predictive search model: Formeln? Validierung gegen Eye Tracking? |
| 5 | **TOD16** Sketchplore | Gewichtungsschema U = Σwᵢsᵢ: Welche Metriken? Wie werden Gewichte gelernt? |

---

### Tag 4 — Validierung + Pipeline-Methodik
**Lesefokus:** Wie validieren andere ihre Pipelines? Was ist methodisch übertragbar?

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **LOR24** In-Vehicle UI Review | Welche Gaps benennt der Review? Welche Metriken fehlen laut Autoren? |
| 2 | **BUR21** UiLab | Wie ist das Benchmark-Setup? Reproduzierbarkeit? Vergleichbar mit deiner Pipeline? |
| 3 | **LIA26** Human-in-the-Loop | Wie werden User Model Priors integriert? Optimierungsschleife? |
| 4 | **MIA26** Log2Motion | Multi-Output-Kohärenz: Wie werden Speed + Accuracy + Effort gemeinsam modelliert? |
| 5 | **KO26** Criticmate | Zwei-Phasen-Evaluation (Perception → Comprehension): Übertragbar auf deine Heads? |

---

### Tag 5 — Personality Layer + Ability-Based Modeling
**Lesefokus:** Wie werden Nutzerprofile in kognitive Modelle integriert?

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **JIE24** EyeFormer | Personalisierte Scanpaths: Wie werden Nutzereigenschaften kodiert? Transformer-Architektur? |
| 2 | **SAR16** Ability-Based Aging | Wie werden motorische/kognitive Fähigkeiten als Modell-Input genutzt? |
| 3 | **GAJ08** SUPPLE++ | Fundamentalpaper: Wie werden User Traits direkt in UI-Parameter übersetzt? |
| 4 | **ZHA24** Simulating Emotions | Appraisal-Modell + RL: Wie wird Emotion als Zustandsvariable modelliert? |
| 5 | **BAI24** HMD Multitasking | Attention Switching: Wie wird zwischen Tasks umgeschaltet? Dual-Task-Metriken? |

---

### Tag 6 — Tier 2 abschließen
**Lesefokus:** Lernkurve + Eye-Tracking-Validierung

| # | Paper | Kernfrage beim Lesen |
|---|-------|---------------------|
| 1 | **TOD18** Familiarisation | Layout-Lernkurve: Wie wird Vorwissen als Variable modelliert? History-Variable? |
| 2 | **KRE16** Eye Tracking CL | Pupillendurchmesser + Mikrosakkaden: Konkrete r-Werte? Verbindung zu HCEye-Metriken? |

---

---

## Screening-Kriterien
- **Tier 1** — Direkt zitierpflichtig: Ohne dieses Paper kann kein Kernargument der Thesis belegt werden
- **Tier 2** — Wichtig für konkrete Abschnitte: Related Work, Methode, Diskussion
- **Tier 3** — Überfliegen reicht: Abstract + Ergebnisse lesen, als Hintergrundquelle
- **Tier 4** — Peripheral / Context: Optional für Outlook oder Diskussion

---

## Tier 1 — Vollständig lesen (Pflicht, vor dem Schreiben)

Diese 7 Paper sind das Fundament der Thesis. Jedes Argument in Kapitel 2–4 stützt sich darauf.

| ID | PDF | Warum unverzichtbar |
|----|-----|---------------------|
| **OUL18** | `OUL18_AIM.pdf` | Das System das du erweiterst — Baseline, Metriken, Architektur |
| **OUL22** | `OUL22_computational_rationality.pdf` | Theoretisches Fundament für Stage 2 — ohne das kein „warum" |
| **JOK20** | `JOK20_adaptive_feature_guidance.pdf` | Predicted Search Time Metrik — Stage 2 Head 2 Kernreferenz |
| **DAS24** | `DAS24_HCEye.pdf` | Empirischer Beweis: Cognitive Load verändert Saliency → Hauptargument für die Kopplung |
| **JIA23** | `JIA23_UEyes.pdf` | Stage 1 Baseline — Dataset, Metriken, UI-spezifische Saliency |
| **GUO26** | `GUO26_SeekUI.pdf` | SOTA Stage 2 — direkter Vergleichskandidat für deine Pipeline |
| **KANK17** | `KANK17_ABC_cognitive_models.pdf` | Methode: Wie du Stage 2 kalibrierst (ABC / inverse modeling) |

---

## Tier 2 — Sorgfältig lesen (für spezifische Abschnitte)

### Stage 2 — Modellierung & CR-Theorie
| ID | PDF | Wofür lesen |
|----|-----|-------------|
| **JOK21** | `JOK21_drivers_adaptation.pdf` | Automotive CR-Modell — Stage 2 Fahrkontext |
| **CHE21** | `CHE21_Gaze_Selection.pdf` | RL-Modell für Fixationsanzahl/-dauer — Head 2 |
| **SHI24** | `SHI24_CRTypist.pdf` | Supervisory control vision+motor — Stage 2 Architektur |
| **LAN24** | `LAN24_MARLUI.pdf` | Intent-Inferenz via Multi-Agent RL — Task Descriptor Logik |
| **SHI25** | `SHI25_Chartist.pdf` | Task-konditionierte Scanpaths via LLM+RL — direkte SOTA |
| **LIA25** | `LIA25_Affordance_CR.pdf` | Brücke Stage 1→2: Features → Affordance → Aktion |
| **TOD19** | `TOD19_Individualising_Layouts.pdf` | Predictive visual search — Stage 2 Goldstandard |
| **TOD16** | `TOD16_Sketchplore.pdf` | Gewichtete Metrikformel U = Σwᵢsᵢ — Stage 1 Aggregation |

### Validierung & Methode
| ID | PDF | Wofür lesen |
|----|-----|-------------|
| **LOR24** | `LOR24_in_vehicle_UI_review.pdf` | Gap-Analyse: Warum keine Modelle in der Praxis → RQ-Begründung |
| **BUR21** | `BUR21_UiLab.pdf` | Reproduzierbarkeit & Methodik-Anker für deine Pipeline |
| **LIA26** | `LIA26_human_in_loop_optimization.pdf` | In-silico Optimierung — Verbesserungsaspekt der RQ |
| **MIA26** | `MIA26_Log2Motion.pdf` | Multi-Output-Kohärenz: Speed + Accuracy + Effort — Head-Architektur |
| **KO26** | `KO26_Criticmate.pdf` | Strukturvalidierung: Zweistufige Evaluation (Perception→Comprehension) |

### Personality Layer (optional, aber vorbereitet)
| ID | PDF | Wofür lesen |
|----|-----|-------------|
| **JIE24** | `JIE24_EyeFormer.pdf` | Personalized Scanpaths — optionaler Personality-Head |
| **SAR16** | `SAR16_Ability_Based_Aging.pdf` | Ability-basierte Modellierung als Personality-Baseline |
| **GAJ08** | `GAJ08_Ability_Based_Interfaces.pdf` | Fundamentalpaper: User-Traits → UI-Performance |
| **ZHA24** | `ZHA24_Simulating_Emotions.pdf` | Emotion + RL — Affective Extension |
| **BAI24** | `BAI24_HMD_Multitasking.pdf` | Attention Switching zwischen Tasks — Stage 2 multi-task |
| **TOD18** | `TOD18_Familiarisation.pdf` | Layout-Lernkurve — History-Variable Stage 2 |

---

## Tier 3 — Überfliegen (Abstract + Ergebnisse)

### Technische Grundlagen (Stage 1 Formeln)
| ID | PDF | Was entnehmen |
|----|-----|---------------|
| **SHA48** | `SHA48_Information_Theory.pdf` | Formel Shannon Entropy H = -Σp·log(p) — einmal nachschlagen |
| **ROS07** | `ROS07_Visual_Clutter.pdf` | Feature Congestion Definition — Stage 1 Formel |
| **MIN15** | `MIN15_Interface_Aesthetics.pdf` | Symmetrie-/Ordnungsmaße für GUIs — Stage 1 Feature |
| **KUM17** | `KUM17_DeepGaze_Fixation.pdf` | DeepGaze II Architektur — Stage 2 Head 1 Baseline |
| **JIA15** | `JIA15_SALICON.pdf` | SALICON Dataset — Head 1 Trainings-Baseline |
| **SAL00** | `SAL00_EMMA_Eye_Movements.pdf` | EMMA-Modell Grundlagen — historischer Kontext CR |

### LLM & Persönlichkeit (Limitation-Abschnitt)
| ID | PDF | Was entnehmen |
|----|-----|---------------|
| **ARG23** | `ARG23_Out_of_One_Many.pdf` | LLMs simulieren demografische Gruppen — Personality Layer Rechtfertigung |
| **SER23** | `SER23_Personality_LLMs.pdf` | Big Five in LLMs — Limitation deklarieren |
| **KAP25** | `KAP25_Simulacrum_Stories.pdf` | Epistemische Grenzen LLM-Simulation — Limitations-Kapitel |

### Theoretischer Hintergrund
| ID | PDF | Was entnehmen |
|----|-----|---------------|
| **OUL16** | `OUL16_HCI_Problem_Solving.pdf` | HCI als Problem-Solving — theoretische Rahmung Intro |
| **LOP25** | `LOP25_Scanpath_Comparison.pdf` | Scanpath-Metriken im Vergleich — Methodenauswahl |
| **WU24** | `WU24_UIClip.pdf` | CLIP-basiertes UI-Assessment — Kontrastpaper (Blackbox vs. CR) |
| **LIN24** | `LIN24_Task_Switching_RL.pdf` | Task-Switching Policies — Stage 2 Kontext |
| **CHO26** | `CHO26_Audiovisual_Search.pdf` | Search Time vs. Effort Multi-Output — Kohärenz |
| **LIA24** | `LIA24_Cognitive_Load_Web.pdf` | Cognitive Load + visuelle Komplexität — Hintergrund Stage 1 |
| **REF_LIN19** | `REF_LIN19_MixedReality.pdf` | Adaptive UI bei steigendem CL — Stage 2 Motivation |

### Nicht in Excel (Snowball / Grundlagen)
| ID | PDF | Was entnehmen |
|----|-----|---------------|
| **BOD06** | `BOD06_Second_Wave_HCI.pdf` | HCI-Wellen — nur wenn Einleitung historisch rahmt |
| **BOD15** | `BOD15_Third_Wave_HCI.pdf` | HCI-Wellen — nur wenn Einleitung historisch rahmt |
| **BUR22** | `BUR22_UI_Consistency.pdf` | UI Consistency Metriken — Stage 1 ergänzend |

---

## Tier 4 — Peripheral (nur bei Bedarf)

Vorhanden, aber nicht für Core-Argument nötig. Für Outlook oder Diskussionskapitel:

| ID | PDF | Wann relevant |
|----|-----|---------------|
| **CHE26** | `CHE26_Adversarial_UI.pdf` | Outlook: Warum Saliency-Genauigkeit sicherheitskritisch |
| **KIM26** | `KIM26_Adaptive_Interfaces.pdf` | Outlook: Adaptive Interfaces als nächster Schritt |
| **COR26** | `COR26_Goal_Driven_Interfaces.pdf` | Diskussion: LLM-Ansätze als Gegenpol |
| **CAO25** | `CAO25_Task_Driven_UI.pdf` | Diskussion: Task-Driven UI Design als Kontext |
| **ANON26** | `ANON26_Temporal_AV_Takeover.pdf` | Supervisor-Paper — wird separat besprochen |

---

## PDFs — Offene Punkte (Stand 4. Juni 2026)

| ID | Status | Anmerkung |
|----|--------|-----------|
| **KRE16** | ✅ `Tier2_Sorgfaeltig/` | Note erstellt |
| **TUC09** | ✅ `Tier3_Ueberfliegen/` | Note erstellt |
| **HAR88** | ⚠️ fehlt | Nur in Hochschul-Bibliothek zugänglich — Zotero-Eintrag reicht |
| **TAU16** | ✅ `Tier4_Peripheral/` | 2016-Review als Ersatz für TAU04 (Original noch offen) |
| **CHE18** | ❌ entfernt | Titel + DOI nicht verifizierbar (KI-Halluzination) — durch MIN15 + WU24 ersetzt |
| **LI20** | ❌ entfernt | Titel + DOI nicht verifizierbar (KI-Halluzination) — durch WU24 ersetzt |

---

## Empfohlene Lesereihenfolge

```
Woche 1 (Fundament):
  OUL18 → OUL22 → JOK20 → DAS24

Woche 2 (Stage 2 verstehen):
  GUO26 → KANK17 → JIA23 → JOK21

Woche 3 (Related Work auffüllen):
  TOD19 → TOD16 → CHE21 → SHI24 → LAN24

Woche 4 (Methode & Validierung):
  LOR24 → BUR21 → LIA26 → MIA26 → KO26

Parallel (Formeln nachschlagen, kein vollständiges Lesen):
  SHA48, ROS07, MIN15, KUM17, JIA15
```
