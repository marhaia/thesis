# Random Forests (2001)

**Autor:** Leo Breiman  
**Quelle:** *Machine Learning*, 45(1), 5–32 (2001). Kluwer Academic Publishers  
**PDF:** `Entwicklungsreferenzen/BRE01_Random_Forests.pdf`  
**Typ:** ML-Grundlagenreferenz (im Code zitiert)

---

## Wozu diese Referenz

Wird in `stage2/regression_model.py` als Grundlagenreferenz für Random Forest Regression zitiert. RF ist das **mittlere Modell** in Stage 2 zwischen Ridge (linear) und XGBoost (boosted trees).

## Kernaussage

Random Forests kombinieren viele Decision Trees durch:
1. **Bagging:** Jeder Baum trainiert auf Bootstrap-Sample der Daten
2. **Feature Randomization:** Jeder Split wählt aus zufälligem Feature-Subset

Ergebnis: Geringere Varianz als einzelne Bäume, robuster gegen Overfitting. Out-of-bag Error als eingebautes Validierungsmaß.

## Relevanz für Thesis

- **Kapitel:** Methodik → Stage 2 Modell-Architektur
- **Verwendung:** RF als Zwischenmodell — falls RF >> Ridge, sind nicht-lineare Interaktionen zwischen Features vorhanden
- **Zitierbar für:** "Für nicht-lineare Zusammenhänge setzen wir Random Forest Regression ein (Breiman, 2001)"

> **Lesen nötig?** Nein — beim Schreiben als Standardreferenz zitieren.

---

**Tags:** #stage2 #random-forest #ML-Grundlage #ensemble
