# Search Methodology — Systematic Literature Search
**Thesis:** Integrating Cognitive Predictive Metrics into the AIM Platform for Automated GUI Evaluation  
**Zeitraum:** April 2026  
**Zweck:** Dokumentation der systematischen Literatursuche für die Methodenbegründung in der Thesis

---

## 1. Suchdatenbanken

| Datenbank | Begründung |
|-----------|-----------|
| ACM Digital Library | Einzige Suchdatenbank — HCI-Kernliteratur (CHI, UIST, ETRA etc.) |
| ResearchRabbit | Snowballing — Paper aus Zitationsnetzwerken der Kernliteratur |

---

## 2. Suchcluster & Queries

### Cluster 1: GAP — AIM Platform Limitations
**Ziel:** Defizite aktueller computational Modelle identifizieren (fehlende Task-Awareness, Ungenauigkeiten bei komplexen Layouts) → Rechtfertigung der RQ

| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|-----------|
| Q1 | ACM DL | `"Aalto Interface Metrics"` | Research Article | 7 | 7 | 4 |


### Cluster 2: COGNITIVE METRICS — Predicted Search Time & Mental Workload
**Ziel:** Quantifizierbare kognitive Metriken identifizieren (z.B. Predicted Search Time) die über visuelle Saliency hinausgehen → geeignet für AIM-Integration

| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|-----------|
| Q2 | ACM DL | `"visual search time" AND GUI AND model*` | 2020–2026; Research Article | 25 | 20 | 6 |


### Cluster 3: TASK & COGNITIVE LOAD — Top-Down Guidance & Saliency
**Ziel:** Empirische Belege für die Unzulänglichkeit statischer Bottom-Up-Modelle → Notwendigkeit task-kontextualisierter Metriken

| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|-----------|
| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|----------|
| Q3a | ACM DL | `(task* OR goal*) AND saliency AND "user interface"` | — (zu breit, nicht weiter gescreent) | 5530 | — | — |
| Q3b | ACM DL | `saliency AND "user interface" AND ("task-driven" OR "goal-directed")` | 2020–2026; Research Article | 54 | 21 | 7 |
| Q3c | ACM DL | `"task-driven saliency" OR "goal-directed saliency" OR "task-aware saliency"` | — | 1 | — | — |

### Cluster 4: COMPUTATIONAL RATIONALITY — Theoretisches Fundament
**Ziel:** Theoretischer Rahmen: Interaktion als mathematisches Optimierungsproblem → Rechtfertigung des kognitiven Metrik-Ansatzes

| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|-----------|
| Query | Datenbank | Suchstring | Filter | Ergebnisse | Gescreent | Ausgewählt |
|-------|-----------|-----------|--------|-----------|-----------|----------|
| Q4.1 | ACM DL | `"computational rationality"` | Research Article | 36 | 20 | 5 |
| Q4.2 | ACM DL | `"computational rationality" AND optimiz*` | Research Article | 20 | 20 | 11 |

### Zusatz: ResearchRabbit (Snowballing)
Paper aus Zitationsnetzwerken der Kernliteratur — nicht durch reguläre Queries gefunden:
- GUO26 (SeekUI), KO26 (Criticmate), CHE26 (Adversarial UI), LIA25 (Human-in-the-Loop) u.a.

### Peripheral / Context Papers (behalten, aber nicht Core)
Diese Paper sind thematisch angrenzend und könnten in einem Outlook- oder Related-Work-Abschnitt erwähnt werden, sind aber kein direkter Beleg für die Pipeline:

| ID | Autoren | Titel (Kurzform) | Einordnung |
|----|---------|-----------------|------------|
| CHE26 | Chen et al. 2026 | Adversarial UI Manipulations (LLM Agents) | Zeigt warum Saliency-Genauigkeit sicherheitskritisch ist — Outlook |
| KIM26 | Kim et al. 2026 | In-Situ Adaptive Interfaces (Online Browsing) | Adaptive UI Design Dimensions — Context |
| COR26 | Cordioli et al. 2026 | Goal-Driven Interfaces / Mirage (Web) | LLM-Web-UI, nicht CR-basiert — Context |
| CAO25 | Cao et al. 2025 | Generative Task-Driven UI | Task-driven data models — Context |

---

### Reference Only (nicht analyzed, aber konzeptuell relevant)
| ID | Autoren | Jahr | Titel | Begründung |
|----|---------|------|-------|------------|
| LIN19 | Lindlbauer et al. | 2019 | Context-Aware Online Adaptation of Mixed Reality Interfaces | MR-Kontext nicht direkt übertragbar, aber konzeptuell relevant: System reduziert UI-Komplexität automatisch bei steigendem Cognitive Load — stützt das "warum" von Stage 2. PDF: `REF_LIN19_MixedReality.pdf` |

---

## 3. Statistik

| Schritt | Anzahl |
|---------|--------|
| Gefunden (vor Duplikaten) | 117 |
| Nach Duplikat-Entfernung | 59 |
| Gescreent | 59 |
| Ausgewählt (für Analyse) | 33 |

---

## 4. Inclusion / Exclusion Kriterien

### Inclusion (KEEP)
- Direkte Relevanz für AIM-Plattform oder computational GUI Evaluation
- Liefert kognitive Metriken, empirische Evidenz oder theoretischen Rahmen für Stage 1 oder Stage 2
- Peer-reviewed (ACM, IEEE, Elsevier) oder prominenter arXiv-Preprint (2024–2026)
- Zeitraum: bevorzugt 2016–2026, Ausnahmen für Grundlagenwerke (Shannon 1948, Hart 1988)

### Exclusion (EXCLUDE)
| Begründung | Beispiele |
|-----------|---------|
| Out of Scope: Nicht auf 2D-Screenshot-basierte GUI-Evaluation anwendbar | VR/XR-Ergonomie, tangible interfaces |
| Motor Only: Fokus auf motorische Gesetze ohne kognitiven Load-Bezug | Fitts' Law Varianten, Lasso-Steering |
| Educational/Pedagogical: Kein computational Vorhersagemodell | Mentoring, Apprenticeship-LLMs |
| Generic LLM Focus: LLM-Feedback ohne predictive Metriken | UICrit (Duan et al. 2024) |
| Conversational/Recommender Systems: Kein GUI-Evaluations-Kontext | Mahmud et al. 2025 |
| Domain too narrow: Nicht generalisierbar auf automotive HMIs | T&C-Design, free-form web curation |
| Formal Methods: Logic/Code Verification, kein kognitives Modell | Campos et al. 2020 |

---

## 5. Ausgeschlossene Paper (vollständige Liste)

| ID | Autoren | Jahr | Titel (Kurzform) | Begründung |
|----|---------|------|-----------------|-----------|
| DUA | Duan et al. | 2024 | UICrit: Enhancing Automated Design Evaluation | Generic LLM Focus — keine computational/predictiven Metriken. Datensatz (3.059 Critiques, 983 Mobile UIs) potenziell als Benchmark, aber LLM-Evaluation ≠ kognitives Vorhersagemodell. PDF: `EXCL_DUA24_UICrit.pdf` |
| CAM | Campos et al. | 2020 | Supporting the Analysis of Safety Critical UIs | Formal Methods (CIRCUS, PVSio-web, IVY) — model-checking auf Code-Ebene, kein kognitives Vorhersagemodell. Domäne passt, Methode nicht. PDF: `EXCL_CAM20_SafetyCritical.pdf` |
| JAI | Jain et al. | 2021 | Recognizing creative visual design | Out of Context — free-form web curation |
| AHN | Ahn et al. | 2026 | From Answer Givers to Design Mentors | Educational/Pedagogical |
| BOU | Bouzit et al. | 2017 | Polymodal Menus | Out of Scope — Voice/Gesture Multimodalität |
| BOU | Bouzit et al. | 2014 | From appearing to disappearing ephemeral adaptation | Too narrow — Fading Menus Trick |
| VAN | Vanderdonckt et al. | 2019 | Exploring a Design Space of Graphical Adaptive Menus | Widget Focus — Cloud Menus |
| YAM | Yamanaka et al. | 2022 | Path-Segmentation for Modeling Lasso Times | Motor Only — Steering Laws |
| SUN | Sun et al. | 2019 | Low-Occlusion Qwerty Soft Keyboard | Motor Only — Text Entry |
| WEN | Wentzel et al. | 2020 | Improving VR Ergonomics Through Non-Linear Input | Hardware/VR |
| KIT | Kitkowska | 2022 | Online Terms and Conditions: Improving User Engagement | User Behavior — zu spezifisch |
| ROS | Ross et al. | 2023 | The Programmer's Assistant | LLM/Coding — kein GUI-Kontext |
| BER | Bertolo et al. | 2025 | CalMe: Tangible Environment for Pupils | Tangible/Education |
| REE | Reed et al. | 2024 | Shifting Ambiguity, Collapsing Indeterminacy | Philosophie/Theorie — zu abstrakt |
| BAZ | Bazzana et al. | 2025 | GlideRX: Situation Awareness in Glider Flight | XR/Aviation — 3D, nicht 2D GUI |
| MAH | Mahmud et al. | 2025 | User Preferences for Interaction Styles in Conversational RS | Conversational Recommender System — komplett anderer Kontext (NLP-Dialogsystem), kein Bezug zu GUI-Metriken oder kognitiven Vorhersagemodellen. PDF: `EXCL_MAH25_User_Preferences_Traits.pdf` |

---

## 6. Hinweise für das Methodik-Kapitel der Thesis

- Der Suchprozess folgt dem **PRISMA-ähnlichen Protokoll** (Identification → Screening → Eligibility → Included)
- Snowballing via ResearchRabbit ergänzt die systematische Suche und ist als solches zu deklarieren
- Alle Queries sind reproduzierbar (ACM DL Strings dokumentiert)
- Die Excel-Datei `Literatur_Research.xlsx` dient als vollständiges Such-Protokoll und Backup
- Peripheral Papers (CHE26, KIM26, COR26, CAO25) sind als PDF vorhanden — bei Bedarf für Outlook oder Diskussionskapitel verwendbar
