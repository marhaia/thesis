# Ridge Regression: Applications to Nonorthogonal Problems (1970)

**Autoren:** Arthur E. Hoerl, Robert W. Kennard  
**Quelle:** *Technometrics*, Vol. 12, No. 1 (Feb. 1970), pp. 69–82  
**PDF:** `Entwicklungsreferenzen/HOE70_Ridge_Regression.pdf`  
**Typ:** Statistik-Grundlagenreferenz (im Code zitiert)

---

## Wozu diese Referenz

Wird in `stage2/regression_model.py` als Grundlagenreferenz für Ridge Regression (L2-Regularisierung) zitiert. Ridge Regression ist das **Baseline-Modell** in Stage 2.

## Kernaussage

Einführung des Ridge-Schätzers als Alternative zur OLS-Regression bei multikollinearen Prädiktoren. Fügt einen Regularisierungsterm $\lambda \|\beta\|^2$ zur Verlustfunktion hinzu:

$$\hat{\beta}_\text{ridge} = (X^TX + \lambda I)^{-1} X^T y$$

Der Parameter $\lambda > 0$ reduziert Varianz auf Kosten eines kleinen Bias — stabiler bei korrelierten Features.

## Relevanz für Thesis

- **Kapitel:** Methodik → Stage 2 Modell-Architektur
- **Verwendung:** Ridge als interpretierbarste Baseline. Falls Stage 2 mit Ridge-Baseline ähnliche Performance erreicht wie XGBoost → Hinweis auf lineare Trennbarkeit der Feature-Räume
- **Zitierbar für:** "Als Baseline-Modell verwenden wir Ridge Regression (Hoerl & Kennard, 1970) — ein L2-regularisierter linearer Schätzer"

> **Lesen nötig?** Nein — beim Schreiben einfach als Standardreferenz zitieren.

---

**Tags:** #stage2 #ridge-regression #baseline #ML-Grundlage #regularisierung
