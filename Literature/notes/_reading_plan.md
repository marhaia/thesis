# Leseplan — Priorisierung nach RQ & Projektplan

**RQ:** Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?  
**Pipeline:** Stage 1 (Visual Complexity Vector) → Stage 2 (Saliency + Fixation + Cognitive Load Index)

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

## ⏳ Noch fehlende PDFs (kommen im Laufe des Tages)

| ID | Titel | Wofür |
|----|-------|-------|
| **CHE18** | Chen 2018 — Predicting Tappability | Tier 3 — Single-Output Baseline, Kontrastpaper |
| **KRE16** | Krejtz 2016 — Transition Entropy + Pupil | Tier 2 — Head 2 Validierung (Fixation Entropy) |
| **TUC09** | Tuch 2009 — Visual Complexity of Websites | Tier 3 — Stage 1 Feature Validierung |
| **LI20** | Li 2020 — Automated Usability Evaluation | Tier 3 — Kontrastpaper Deep Learning |
| **HAR88** | Hart & Staveland 1988 — NASA-TLX | Tier 2 — Head 3 Validierung (Zotero-Eintrag reicht) |
| **TAU04** | Taubman-Ben-Ari 2004 — Driving Style | Tier 3 — Personality + Automotive Kontext |

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
