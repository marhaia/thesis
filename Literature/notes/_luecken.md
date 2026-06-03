# Literaturlücken — Offene Punkte

**Stand:** 2. Juni 2026  
**Zweck:** Tracking offener Referenzen und inhaltlicher Lücken — aktualisieren sobald gelöst

---

## 1. Fehlende PDFs (kommen noch)

| ID | Autoren | Titel | Wo zu finden | Priorität |
|----|---------|-------|-------------|-----------|
| **KRE16** | Krejtz et al. 2016 | Eye Tracking Cognitive Load Using Pupil Diameter and Microsaccades | ACM ETRA, DOI: 10.1145/2857491.2857540 | 🔴 Hoch — Head 2 Validierung |
| **TUC09** | Tuch et al. 2009 | Visual Complexity of Websites: Effects on Users' Experience | IJHCS, DOI: 10.1016/j.ijhcs.2009.04.002 | 🔴 Hoch — Stage 1 Feature |
| **HAR88** | Hart & Staveland 1988 | Development of NASA-TLX | Buchkapitel North-Holland — Zotero-Eintrag reicht, kein PDF nötig | 🔴 Hoch — Head 3 Grundlage |
| **CHE18** | Chen et al. 2018 | Predicting Tappability of Mobile Interface Elements | ACM CHI, DOI: 10.1145/3173574.3174121 | 🟡 Mittel — Single-Output Baseline |
| **LI20** | Li et al. 2020 | Automated Usability Evaluation of Mobile Apps Using Deep Learning | ACM CHI, DOI: 10.1145/3313831.3376224 | 🟡 Mittel — Kontrastpaper |
| **TAU04** | Taubman-Ben-Ari et al. 2004 | Driving Style & Personality | Journal of Safety Research | 🟢 Niedrig — Peripheral |

---

## 2. Stage 1 Features ohne eigene Referenz

| Feature | Problem | Lösung |
|---------|---------|--------|
| **Edge Density** (Canny/Sobel) | Standard-CV, kein HCI-Paper nötig | Als "standard feature extraction" formulieren → OUL18 als Gesamtreferenz |
| **Chromatic Coherence** | Kein dediziertes Paper vorhanden | Option A: Paper suchen (Farb-Komplexität in UIs) — Option B: Unter OUL18 + ROS07 subsumieren |
| **Interactive Element Density** | Kein Paper zugewiesen | OUL18 als Gesamtreferenz reicht |

---

## 3. Inhaltliche Lücken — kommen beim Schreiben

| Wann relevant | Lücke | Empfehlung |
|--------------|-------|------------|
| **Head 3 (CL-Index)** | Wie wird Skalar 0–100 normiert? Skalierungsreferenz fehlt | Erst beim Schreiben klären — ggf. eigene Definition ohne Referenz |
| **Result 1 (Varianzdekomposition)** | Statistische Methode R²-Zerlegung braucht Referenz | Standardlehrbuch (Field 2013 o.ä.) — noch nicht gesucht |
| **Personality Layer (nicht-LLM)** | Big Five → kognitive Leistung: Nur LLM-Paper vorhanden (ARG23, SER23), aber keine klassische Persönlichkeitspsychologie | Suche: "Big Five cognitive performance" — erst relevant wenn Personality Layer ausgearbeitet wird |
| **State-based Simulation (Stress/Fatigue)** | Als Limitation deklariert, aber keine Referenz für "warum nicht validiert" | KAP25 + SER23 reichen für LLM-Grenzen; für State-based braucht man evtl. noch ein Workload-Paper |

---

## 4. Supervisor-Paper

| ID | Status | Anmerkung |
|----|--------|-----------|
| **ANON26** | PDF vorhanden (`ANON26_Temporal_AV_Takeover.pdf`) | Anonymes Paper vom Supervisor — separat besprechen wie/ob zitieren |

---

## 5. arXiv-Paper — Qualitätsprüfung

Faustregel: arXiv ok als Ergänzung, nie als einzige Quelle für Kernargument. Bei Abgabe prüfen ob inzwischen peer-reviewed erschienen.

| ID | arXiv-Nr. | Risiko | Maßnahme |
|----|-----------|--------|----------|
| **SER23** | 2307.00184 | 🟡 Nur arXiv, nie in Konferenz | Als "preprint" kennzeichnen — nur für Limitation genutzt, vertretbar |
| **JIE24** | 2404.10163 | 🟢 Gering | Bei Abgabe prüfen ob CHI/UIST/ETRA-Version existiert |
| **SHI25** | 2502.03575 | 🟢 Gering | Wahrscheinlich CHI 2026 — finalen Venue nachschlagen |
| **LOP25** | 2503.24160 | 🟡 Noch unveröffentlicht | Nur Tier 3 (peripher) — kein Kernargument darauf stützen |
| **CHO26** | 2602.02790 | 🟢 Gering | Aalto-Gruppe, etabliert — bei Abgabe Status prüfen |

**Alle Kernargumente stützen sich auf peer-reviewed Paper (ACM/IEEE/Elsevier) ✓**

---

## Erledigte Lücken ✓

*(Hier eintragen sobald eine Lücke geschlossen ist)*
