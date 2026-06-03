# UEyes: Understanding Visual Saliency across User Interface Types (2023)

**Autoren:** Jiang, Y., Leiva, L.A., Houssel, P.R.B., Tavakoli, H.R., Kylmälä, J., & Oulasvirta, A.  
**Quelle:** CHI Conference on Human Factors in Computing Systems (CHI '23)  
**DOI:** (ACM CHI 2023)  
**PDF:** JIA_2023.pdf  
**ID:** JIA23 | **Prio:** 2 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Gibt es universelle vs. UI-typ-spezifische Blickmuster — und wie unterscheidet sich visuelle Saliency je nach Interface-Typ (Web, Mobile, Desktop)?

## Methode
- Großangelegte Eye-Tracking-Studie
- Mehrere UI-Typen: Web, Mobile, Desktop
- Erhebung von Fixationswahrscheinlichkeiten
- Analyse von Layout-Biases (z.B. Upper-Left Bias)
- Ergebnis: **UEyes Dataset** — öffentlich verfügbar, große Benchmark-Grundlage

## Wichtigste Ergebnisse
- UI-Typen haben **spezifische Layout-Biases** (z.B. Web: Upper-Left Bias)
- Manche Blickmuster sind universell, andere UI-typ-spezifisch
- Beschreibend/empirisch: erklärt das **Was**, nicht das **Warum** (kein kognitives Modell)
- Liefert massiven Benchmark-Datensatz für UI-spezifische Saliency-Evaluation

## Relevanz für meine Thesis
> Validation benchmark — helps verify if new cognitive metrics in AIM still respect fundamental UI-biases found in large datasets.

- **Rolle:** Moderner Benchmark zur Validierung meiner erweiterten AIM-Metriken
- **Kapitel:** Related Work / Evaluation / Methodik (Validierungsstrategie)
- **Argument:** Wenn ich neue kognitive Metriken in AIM integriere, müssen diese die empirisch belegten Grundmuster (Upper-Left Bias etc.) noch reproduzieren können
- **Direkt zitierbar für:** Baseline-Saliency-Patterns, Validierungsdatensatz, UI-typ-spezifische Unterschiede

## Kritik / Offene Fragen
- Rein deskriptiv/empirisch — erklärt nicht warum diese Biases entstehen
- Kein kognitives Modell dahinter (wäre OUL22 / JOK20)
- Domäne: Web/Mobile — automotive GUIs fehlen → muss ich als Limitation ansprechen
- Meine Frage: Ist der UEyes-Datensatz direkt für automotive Kontext übertragbar?

## Verbindungen zu anderen Papern
- Validiert → OUL18 (AIM): zeigt was Saliency-Modelle korrekt vorhersagen müssen
- Ergänzt durch → DAS24: wenn kognitive Last Saliency verändert, reicht dieser Benchmark allein nicht
- Verbindung → JOK20: Upper-Left Bias sollte sich in Search-Time-Modellen widerspiegeln

## Meine Notizen (aus Excel)
- **Cognitive Metric:** Fixation Probability
- **Gap filled:** Provides a massive benchmark dataset for UI-specific saliency
- **Relation to AIM:** Validation — acts as modern benchmark; helps verify if new cognitive metrics in AIM still respect fundamental UI-biases (like Upper-Left-Bias) found in large datasets
- **Cluster:** Empirical Saliency

---

**Tags:** #saliency #eye-tracking #benchmark #UEyes #upper-left-bias #UI-types #fixation #empirical #validation
