# Literaturlücken — Offene Punkte

**Stand:** 2. Juni 2026  
**Zweck:** Tracking offener Referenzen und inhaltlicher Lücken — aktualisieren sobald gelöst

---

## 1. Fehlende PDFs — Stand 4. Juni 2026

### Gefunden ✅

| ID | Titel | Ablage |
|----|-------|--------|
| **KRE16** | Krejtz et al. — Eye Tracking Cognitive Load (Pupil + Microsaccades) | `pdfs/Tier2_Sorgfaeltig/KRE16_Eye_Tracking_Cognitive_Load.pdf` — PLOS ONE DOI: 10.1371/journal.pone.0150489 ✅. Supervisor-DOI `10.1145/2857491.2857540` ist falsch (zeigt auf Gangstudien-Paper). |
| **TUC09** | Tuch et al. — Visual Complexity of Websites (IJHCS 2009) | `pdfs/Tier3_Ueberfliegen/TUC09_Visual_Complexity_Websites.pdf` |
| **TAU16** | Taubman-Ben-Ari & Skvirsky 2016 — Driving Style Inventory (Review, 2016er-Version statt 2004) | `pdfs/Tier4_Peripheral/TAU16_Driving_Style_Inventory_Review.pdf` |
| **HOE70** | Hoerl & Kennard 1970 — Ridge Regression | `pdfs/Entwicklungsreferenzen/HOE70_Ridge_Regression.pdf` |
| **BRE01** | Breiman 2001 — Random Forests | `pdfs/Entwicklungsreferenzen/BRE01_Random_Forests.pdf` |
| **BOR15** | Borchani et al. 2015 — Multi-Output Regression Survey | `pdfs/Entwicklungsreferenzen/BOR15_Multi_Output_Regression.pdf` |

### Noch ausstehend

| ID | Autoren | Titel | Priorität | Anmerkung |
|----|---------|-------|-----------|-----------|
| **HAR88** | Hart & Staveland 1988 | Development of NASA-TLX | 🔴 Hoch | Nur in Hochschul-Bibliothek zugänglich — Zotero-Eintrag reicht als Platzhalter |

### Entfernt — Referenzen waren nicht verifiziert ❌

| ID | Begründung |
|----|------------|
| **CHE18** | DOI `10.1145/3173574.3174121` existiert **aber zeigt auf ein anderes Paper**: "User-Driven Design Principles for Gesture Representations" (McAweeney, Zhang, Nebeling, CHI 2018). Titel "Predicting tappability" und Autoren "Chen, J. et al." sind **frei erfunden**. Durch MIN15 + WU24 ersetzt. |
| **LI20** | DOI `10.1145/3313831.3376224` existiert **aber zeigt auf ein anderes Paper**: "Decoding Intent With Control Theory: Comparing Muscle Versus Manual Interface Performance" (Yamagami, Steele, Burden, CHI 2020). Titel "Automated usability evaluation" und Autoren "Li, G. et al." sind **frei erfunden**. Durch WU24 ersetzt. |

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
