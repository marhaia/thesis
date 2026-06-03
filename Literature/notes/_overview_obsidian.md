---
title: Literature Overview
tags:
  - thesis
  - literature
  - overview
created: 2026-06-02
updated: 2026-06-02
rq: "Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?"
papers_total: 50
papers_with_notes: 39
status: active
---

# Literature Overview

> [!abstract] Forschungsfrage
> **Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?**
> 
> Pipeline: Stage 1 (Visual Complexity $v \in \mathbb{R}^8$) → Stage 2 (Saliency + Fixation + Cognitive Load Index)

---

## Status-Legende

| Symbol | Bedeutung |
|--------|-----------|
| ✅ | **Selbst gelesen** — du hast das Paper persönlich gelesen |
| 📝 | **Notes vorhanden** — Notes existieren, noch nicht selbst gelesen |
| 📒 | **Nur nachschlagen** — Formel/Definition in [[_nachschlagen]], kein vollständiges Lesen |
| 📥 | **Kein PDF** — fehlt noch (steht in [[_luecken]]) |
| ⬛ | **Peripheral** — nur bei Bedarf |
| ❌ | **Ausgeschlossen** |

---

## Cluster 1 — AIM Platform & GUI Evaluation Tools
#cluster/aim-evaluation

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| OUL18 | Aalto Interface Metrics (AIM) | 2018 | 📝 | [[OUL18_AIM]] |
| BUR21 | UiLab: Workbench for GUI Experiments | 2021 | 📝 | [[BUR21_UiLab]] |
| BUR22 | (Semi-)Automatic Computation of UI Consistency | 2022 | 📝 | [[BUR22_UI_Consistency]] |
| KO26 | Criticmate: Stagewise Human–AI Co-Critique | 2026 | 📝 | [[KO26_Criticmate]] |
| WU24 | UIClip: Data-driven Model for Assessing UI Design | 2024 | 📝 | [[WU24_UIClip]] |
| MIN15 | Computation of Interface Aesthetics | 2015 | 📒 | [[_nachschlagen]] |
| LI20 | Automated usability evaluation of mobile apps | 2020 | 📥 | — |
| CHE18 | Predicting tappability of mobile interface elements | 2018 | 📥 | — |

> [!tip] Kernaussage
> AIM ist die Baseline und das System das erweitert wird. WU24/KO26 zeigen SOTA — keines integriert kognitive Belastungsvorhersage.

---

## Cluster 2 — Cognitive Models & Computational Rationality
#cluster/cognitive-models #cluster/cr

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| OUL22 | Computational Rationality as a Theory of Interaction | 2022 | 📝 | [[OUL22_computational_rationality]] |
| JOK20 | Adaptive Feature Guidance: Visual Search with Graphical Layouts | 2020 | 📝 | [[JOK20_adaptive_feature_guidance]] |
| JOK21 | Modelling Drivers' Adaptation to Assistance Systems | 2021 | 📝 | [[JOK21_drivers_adaptation]] |
| KANK17 | Inferring Cognitive Models via ABC | 2017 | 📝 | [[KANK17_ABC_cognitive_models]] |
| LIA25 | Redefining Affordance via Computational Rationality | 2025 | 📝 | [[LIA25_Affordance_CR]] |
| SHI24 | CRTypist: Simulating Touchscreen Typing via CR | 2024 | 📝 | [[SHI24_CRTypist]] |
| ZHA24 | Simulating Emotions with Appraisal + RL | 2024 | 📝 | [[ZHA24_Simulating_Emotions]] |
| SAR16 | Towards Ability-Based Optimization for Aging Users | 2016 | 📝 | [[SAR16_Ability_Based_Aging]] |
| GAJ08 | Improving Performance of Motor-Impaired Users (SUPPLE++) | 2008 | 📝 | [[GAJ08_Ability_Based_Interfaces]] |
| HAR88 | Development of NASA-TLX | 1988 | 📥 | — |

> [!tip] Kernaussage
> OUL22 liefert das theoretische Fundament. JOK20 = direkte Implementierungsreferenz (AFG-Modell im Code). KANK17 = Methode für Kalibrierung (ABC/Inversion).

---

## Cluster 3 — Saliency, Scanpath & Visual Attention
#cluster/saliency #cluster/scanpath

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| JIA23 | UEyes: Visual Saliency across UI Types | 2023 | 📝 | [[JIA23_UEyes]] |
| DAS24 | HCEye: Visual Highlighting and Cognitive Load | 2024 | 📝 | [[DAS24_HCEye]] |
| GUO26 | SeekUI: Predicting Visual Search with Reward-Augmented VLM | 2026 | 📝 | [[GUO26_SeekUI]] |
| CHE21 | An Adaptive Model of Gaze-based Selection | 2021 | 📝 | [[CHE21_Gaze_Selection]] |
| JIE24 | EyeFormer: Personalized Scanpaths with Transformer + RL | 2024 | 📝 | [[JIE24_EyeFormer]] |
| SHI25 | Chartist: Task-driven Eye Movement Control | 2025 | 📝 | [[SHI25_Chartist]] |
| LOP25 | Comparative Study of Scanpath Models (InfoVis) | 2025 | 📝 | [[LOP25_Scanpath_Comparison]] |
| KUM17 | DeepGaze II | 2017 | 📒 | [[_nachschlagen]] |
| JIA15 | SALICON: Saliency in Context | 2015 | 📒 | [[_nachschlagen]] |
| ROS07 | Measuring Visual Clutter | 2007 | 📒 | [[_nachschlagen]] |
| TUC09 | Visual complexity of websites | 2009 | 📥 | — |

> [!tip] Kernaussage
> DAS24 = empirischer Hauptbeweis für die Kopplung von Saliency und kognitiver Belastung. GUO26 = direkter SOTA-Vergleichskandidat.

---

## Cluster 4 — Automotive HMI & In-Vehicle UI
#cluster/automotive

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| LOR24 | Computational Models for In-Vehicle UI Design (Review) | 2024 | 📝 | [[LOR24_in_vehicle_UI_review]] |
| BAI24 | Heads-Up Multitasker: Attention Switching on Optical HMDs | 2024 | 📝 | [[BAI24_HMD_Multitasking]] |
| LIN24 | Supporting Task Switching with Reinforcement Learning | 2024 | 📝 | [[LIN24_Task_Switching_RL]] |
| ANON26 | [Supervisor Paper — vertraulich] | 2026 | 📝 | [[_projektplan]] |
| KRE16 | Eye tracking cognitive load (pupil diameter & microsaccades) | 2016 | 📥 | — |
| TAU04 | The multidimensional driving style inventory | 2004 | 📥 | — |

> [!tip] Kernaussage
> LOR24 = Review für Domäneneinordnung. BAI24/LIN24 = CR-Anwendungen in Dual-Task-Szenarien (methodisch verwandt mit Stage 2).

---

## Cluster 5 — Pipeline, Layout-Optimierung & UI-Adaptation
#cluster/pipeline #cluster/layout-optimization

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| TOD16 | Sketchplore: Sketch and Explore Layout Designs | 2016 | 📝 | [[TOD16_Sketchplore]] |
| TOD18 | Familiarisation: Restructuring Layouts with Visual Learning | 2018 | 📝 | [[TOD18_Familiarisation]] |
| TOD19 | Individualising Layouts with Predictive Visual Search | 2019 | 📝 | [[TOD19_Individualising_Layouts]] |
| LIA26 | Human-in-the-Loop Optimization via User Model Priors | 2026 | 📝 | [[LIA26_human_in_loop_optimization]] |
| LAN24 | MARLUI: Multi-Agent RL for Adaptive UIs | 2024 | 📝 | [[LAN24_MARLUI]] |
| MIA26 | Log2Motion: Biomechanical Motion from Touch Logs | 2026 | 📝 | [[MIA26_Log2Motion]] |
| REF_LIN19 | Context-Aware Online Adaptation of MR Interfaces | 2019 | 📝 | [[REF_LIN19_MixedReality]] |
| LIA24 | Cognitive Load in Web Interface Design | 2024 | 📝 ⚠️ sim. | [[LIA24_Cognitive_Load_Web]] |

> [!warning] LIA24 — Simulierte Daten
> LIA24 verwendet **simulierte** Nutzerdaten (keine echten Probanden). Nur für methodische Einordnung verwenden, nicht als empirischen Beleg zitieren.

---

## Cluster 6 — LLM-basierte Nutzersimulation & Persönlichkeit
#cluster/llm-simulation

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| ARG23 | Out of One, Many: LLMs to Simulate Human Samples | 2023 | 📝 | [[ARG23_Out_of_One_Many]] |
| SER23 | Personality Traits in Large Language Models | 2023 | 📝 ⚠️ Preprint | [[SER23_Personality_LLMs]] |
| KAP25 | Simulacrum of Stories: LLMs as Research Participants | 2025 | 📝 | [[KAP25_Simulacrum_Stories]] |

> [!warning] SER23 — Nur arXiv-Preprint
> SER23 ist ein reines arXiv-Preprint ohne Peer Review. Im Fließtext als "Preprint" deklarieren.

> [!tip] Kernaussage
> ARG23/SER23 = Grundlage für Big Five LLM-Simulation im User Profile. KAP25 = Grenzen (Surrogate Effect) — immer zusammen zitieren.

---

## Cluster 7 — HCI-Theorie & Grundlagen
#cluster/hci-theory

| ID | Titel | Jahr | Status | Note |
|----|-------|------|--------|------|
| OUL16 | HCI Research as Problem-Solving | 2016 | 📝 | [[OUL16_HCI_Problem_Solving]] |
| BOD06 | When Second Wave HCI Meets Third Wave Challenges | 2006 | 📝 | [[BOD06_Second_Wave_HCI]] |
| BOD15 | Third Wave HCI, 10 Years Later | 2015 | 📝 | [[BOD15_Third_Wave_HCI]] |
| CHO26 | Simulating Human Audiovisual Search (Sensonaut) | 2026 | 📝 | [[CHO26_Audiovisual_Search]] |
| SHA48 | A Mathematical Theory of Communication | 1948 | 📒 | [[_nachschlagen]] |
| SAL00 | EMMA: Eye Movements and Visual Attention | 2000 | 📒 | [[_nachschlagen]] |

---

## Cluster 8 — Peripheral (Tier 4)
#cluster/peripheral

| ID | Titel | Jahr | Status |
|----|-------|------|--------|
| CHE26 | Adversarial UI Examples | 2026 | ⬛ |
| KIM26 | Adaptive Interfaces | 2026 | ⬛ |
| COR26 | Goal-Driven Interfaces | 2026 | ⬛ |
| CAO25 | Task-Driven UI | 2025 | ⬛ |

---

## Cluster 9 — Ausgeschlossen
#cluster/excluded

| ID | Titel | Begründung |
|----|-------|------------|
| EXCL_CAM20 | Safety-Critical Systems | Scope zu weit |
| EXCL_DUA24 | UICrit: UI Critique Dataset | Kein prädiktiver Ansatz |
| EXCL_MAH25 | User Preferences & Personality (Recommender) | Falscher Kontext |

---

## Fortschritt

> [!check] Abgeschlossen
> - ✅ **7/7** Tier 1 Notes vollständig
> - ✅ **19/19** Tier 2 Notes vollständig
> - ✅ **12/12** Tier 3 Notes vollständig
> - ✅ **39/39** Papiere mit Notes
> - ✅ Exposé in 3 Längen (siehe [[expose/expose_kurz|kurz]], [[expose/expose_standard|standard]], [[expose/expose_ausfuehrlich|ausführlich]])
> - ✅ Implementierung vollständig (Stage 1+2, Flask API :5001, UMSI++, Jokinen AFG, HCEye)

> [!todo] Noch ausstehend
> - 📥 Fehlende PDFs: CHE18, KRE16, TUC09, LI20, HAR88, TAU04 → [[_luecken]]
> - 📖 Selbst-Lesen: alle 📝-Einträge (Tier 1 Pflicht vor Schreibstart)
> - 🔬 Nutzerstudie N≈30–35 (Aug 2026)
> - ✍️ Thesis-Kapitel (Nov 2026)
> - **Abgabe: Dez 2026**

---

## Querverweise

| Datei | Inhalt |
|-------|--------|
| [[_nachschlagen]] | Formel-Lookup (SHA48, ROS07, MIN15, KUM17, JIA15, SAL00) |
| [[_luecken]] | Fehlende PDFs, inhaltliche Lücken, Supervisor-Paper |
| [[_reading_plan]] | Priorisierter 2-Wochen-Leseplan |
| [[_search_methodology]] | ACM DL Suchstrategie, Screening-Protokoll |
| [[_projektplan]] | Zwei-Stufen-Pipeline-Architektur |
| [[_gem_context]] | Gemini Gem System-Prompt |
