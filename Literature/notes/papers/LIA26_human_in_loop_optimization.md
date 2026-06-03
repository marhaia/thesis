# Efficient Human-in-the-Loop Optimization via Priors Learned from User Models (2025/2026)

**Autoren:** Liao, Y-C., Belo, J.M.E., Moon, H-S., Steimle, J. et al.  
**Quelle:** Saarland University / ETH Zürich / Chung-Ang University  
**DOI:** arxiv  
**PDF:** LIA25.pdf  
**ID:** LIA25 | **Status:** 🟢 Analyzed (Auswahl: KEEP — Optimization, C3: Task Interaction)  

---

## Kernfrage
Wie kann menschliche Interaktion (Human-in-the-Loop) mit simulierten User Models kombiniert werden, um Optimierungsprozesse effizienter zu machen?

## Methode
- Human-in-the-Loop Optimierung: echte Nutzerfeedback + User-Model-Simulationen
- Priors werden aus Nutzungsmodellen gelernt (in-silico)
- Anschließend mit echten Daten (in-situ) verfeinert

## Wichtigste Ergebnisse
- Kombination von Simulation (User Models) + echtem Feedback reduziert Evaluationsaufwand
- Gelernte Priors aus User Models beschleunigen die Optimierung erheblich

## Relevanz für meine Thesis
> Optimization: Bridges in-situ and in-silico optimization. Relevant for the "Improvement" aspect of the thesis.

- **Rolle:** Verbindet simulationsbasierte Vorhersage (mein Pipeline-Ansatz) mit empirischer Validierung
- **Kapitel:** Diskussion / Ausblick / Methodologie
- **Argument:** Zeigt dass mein Ansatz (computational Vorhersage als Prior) + empirische Validierung ein wissenschaftlich valider Workflow ist
- **Direkt zitierbar für:** Begründung warum computational Screening + User Study sinnvoll kombinierbar sind

## Kritik / Offene Fragen
- Eher Optimierungs-Kontext — weniger direkt auf GUI-Evaluation bezogen
- Wie konkret ist der Transfer auf meinen Stage 1 → Stage 2 Workflow?

## Verbindungen zu anderen Papern
- Thematisch → DAS24 (HCEye): beide verbinden Simulation mit empirischer Nutzerdaten
- Methodisch → KANK17 (ABC): beide nutzen Priors aus Modellen für effizientere Inferenz

---

**Tags:** #human-in-the-loop #optimization #user-models #in-silico #in-situ #priors #efficiency
