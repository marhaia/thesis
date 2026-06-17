# A Survey on Multi-Output Regression (2015)

**Autoren:** Hanen Borchani, Gherardo Varando, Concha Bielza, Pedro Larrañaga  
**Quelle:** *WIREs Data Mining and Knowledge Discovery*, 5(5), 216–233 (2015)  
**PDF:** `Entwicklungsreferenzen/BOR15_Multi_Output_Regression.pdf`  
**Typ:** Methodik-Referenz (im Code zitiert)

---

## Wozu diese Referenz

Wird in `stage2/regression_model.py` als methodische Grundlage für simultane Multi-Output-Regression zitiert. Stage 2 prediziert 3 Outputs gleichzeitig (cognitive_load_score, search_efficiency, attention_demand) — das ist klassische Multi-Output-Regression.

## Kernaussage

Survey über Multi-Output-Regression-Methoden, kategorisiert in:
1. **Problem Transformation:** Ein-Output-Modell pro Output, unabhängig trainiert
2. **Algorithm Adaptation:** Ein Modell für alle Outputs gleichzeitig — nutzt Korrelationen zwischen Outputs

Evaluationsmetriken für Multi-Output: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² pro Output sowie gemittelt.

## Relevanz für Thesis

- **Kapitel:** Methodik → Stage 2 Architektur-Entscheidung
- **Argument:** Begründet, warum simultane Multi-Output-Prediction (Algorithm Adaptation) gegenüber getrennten Modellen bevorzugt wird — Outputs sind inhaltlich korreliert (hohe Saliency-Dispersion → niedrige Fixationsanzahl → niedrigerer Load-Score)
- **Zitierbar für:** "Stage 2 folgt dem Algorithm-Adaptation-Paradigma der Multi-Output-Regression (Borchani et al., 2015): Ein Modell wird auf alle drei Outputs gleichzeitig trainiert, um Korrelationsstruktur zwischen den Outputs auszunutzen"
- **Verbindung zu Coherence-Term:** Der Coherence-Term L_coh ist eine explizite Form der Output-Korrelation, die BOR15 implizit beschreibt

> **Lesen nötig?** Nein — beim Schreiben als Methodenreferenz zitieren. Die Kategorisierung (Problem Transformation vs. Algorithm Adaptation) ist die einzige relevante Stelle.

---

**Tags:** #stage2 #multi-output-regression #methodik #ML-Grundlage #survey
