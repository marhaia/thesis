# Formeln & Definitionen — Nachschlag-Referenz

**Zweck:** Schnelles Nachschlagen von Formeln und Definitionen beim Schreiben.  
Kein vollständiges Lesen dieser Paper nötig — nur hier nachschlagen.

---

## SHA48 — Shannon Entropy
**Quelle:** Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3).  
**PDF:** `Tier3_Ueberfliegen/SHA48_Information_Theory.pdf`

**Formel:**
$$H = -\sum_{i} p_i \log_2 p_i$$

- $H$ = Entropie (in Bits)
- $p_i$ = Wahrscheinlichkeit des $i$-ten Symbols
- **In Stage 1:** Auf Graustufenhistogramm angewendet → misst globale Informationsdichte des UI-Screenshots
- **Zitieren als:** "Die Shannon Entropie (Shannon, 1948) quantifiziert die Informationsdichte als $H = -\sum p_i \log_2 p_i$"

---

## ROS07 — Visual Clutter (Feature Congestion + Subband Entropy)
**Quelle:** Rosenholtz, R., Li, Y., & Nakano, L. (2007). Measuring visual clutter. *Journal of Vision*, 7(2), Article 17.  
**PDF:** `Tier3_Ueberfliegen/ROS07_Visual_Clutter.pdf`

**Zwei Maße:**
1. **Feature Congestion (FC):** Lokale Dichte von Farbe, Kontrast, Orientierung → aggregiert zu globalem Clutter-Wert
2. **Subband Entropy (SE):** Entropie über Subbänder einer Wavelet-Zerlegung → misst strukturelle Unordnung

- **In Stage 1:** Beide Maße als "Information Clutter" Feature
- **Zitieren als:** "Visual Clutter wird nach Rosenholtz et al. (2007) über Feature Congestion und Subband Entropy quantifiziert"

---

## MIN15 — Interface Aesthetics (8 Metriken)
**Quelle:** Miniukovich, A., & De Angeli, A. (2015). Computation of Interface Aesthetics. *CHI 2015*.  
**PDF:** `Tier3_Ueberfliegen/MIN15_Interface_Aesthetics.pdf`

**8 GUI Aesthetics Metriken:**
1. Visual Clutter
2. Color Range
3. Number of Dominant Colors
4. Figure-Ground Contrast
5. Contour Congestion
6. **Symmetry** ← Stage 1 relevant
7. **Grid Quality** ← Stage 1 relevant
8. White Space

- **In Stage 1:** Symmetry + Grid Quality als "Layout Symmetry" Feature
- **Zitieren als:** "Layout-Symmetrie wird nach Miniukovich & De Angeli (2015) über Symmetrie- und Grid-Qualitätsmaße berechnet"

---

## KUM17 — DeepGaze II (Saliency Baseline)
**Quelle:** Kümmerer, M., Wallis, T.S.A., Gatys, L.A., & Bethge, M. (2017). Understanding Low- and High-Level Contributions to Fixation Prediction. *ICCV 2017*.  
**PDF:** `Tier3_Ueberfliegen/KUM17_DeepGaze_Fixation.pdf`

**Kernaussage:**
- DeepGaze II trennt Low-Level (Kontrast, Helligkeit) von High-Level (Gesichter, Text) Beiträgen zur Fixationsvorhersage
- High-Level dominiert bei Text und Gesichtern → UI-Kontext relevant
- Low-Level dominiert bei abstrakten Texturen

- **In Stage 2:** Head 1 (Saliency Map) nutzt DeepGaze II als Baseline
- **Zitieren als:** "Als Saliency-Baseline verwenden wir DeepGaze II (Kümmerer et al., 2017), das Low- und High-Level Features für Fixationsvorhersage integriert"

---

## JIA15 — SALICON Dataset
**Quelle:** Jiang, M., Huang, S., Duan, J., & Zhao, Q. (2015). SALICON: Saliency in Context. *CVPR 2015*.  
**PDF:** `Tier3_Ueberfliegen/JIA15_SALICON.pdf`

**Dataset-Kennzahlen:**
- Große-Scale Mouse-Tracking als Proxy für Eye-Tracking
- Natürliche Szenen (kein UI-Fokus)
- Oft für Saliency-Modell Pre-Training verwendet

- **In Stage 2:** Head 1 Pre-Training Baseline (vor Fine-Tuning auf UI-Daten via JIA23/UEyes)
- **Zitieren als:** "Initiales Pre-Training auf SALICON (Jiang et al., 2015), danach Fine-Tuning auf UI-spezifische Daten (UEyes, Jiang et al., 2023)"

---

## SAL00 — EMMA (Eye Movements and Mental Acts)
**Quelle:** Salvucci, D.D. (2000). A Model of Eye Movements and Visual Attention. *ICCM 2000*.  
**PDF:** `Tier3_Ueberfliegen/SAL00_EMMA_Eye_Movements.pdf`

**Kernaussage:**
- EMMA: formales Modell der zeitlichen und räumlichen Aspekte von Augenbewegungen
- Integriert in ACT-R/PM Kognitionsarchitektur
- Sakkaden-Dauer als Funktion von Amplitude und Verarbeitungszeit

- **In Thesis:** Historischer Kontext — zeigt dass Augenbewegungsmodellierung in HCI seit 2000 etabliert ist
- **Zitieren als:** "Formale Modelle von Augenbewegungen gehen auf Salvucci (2000) zurück, der EMMA als Teil der ACT-R Architektur einführte"
