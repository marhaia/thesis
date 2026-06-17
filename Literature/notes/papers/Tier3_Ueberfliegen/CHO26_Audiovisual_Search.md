# Simulating Human Audiovisual Search Behavior (2026)

**Autoren:** Hyunsung Cho, Xuejing Luo, Byungjoo Lee, David Lindlbauer, Antti Oulasvirta  
**Quelle:** *CHI 2026*, Barcelona (Aalto / CMU / Yonsei)  
**PDF:** `Tier3_Ueberfliegen/CHO26_Audiovisual_Search.pdf`

---

## Beim Ueberfliegen - gezielt lesen
> Note reicht - zu weit von GUI-Evaluation entfernt. PDF nur wenn du die POMDP-Formel brauchst.

**Direkt springen zu:**
- **Figure 1 (S. 3-4):** POMDP-Modellarchitektur - strukturell identisch mit Stage 2
- **Die Policy-Formel (S. 5):** Resource-Rational Policy - als Einordnung fuer deine Stage-2-Formel
- **Abstract + Conclusion:** Reichen fuer Kontext-Zitat (5 Min)

**Ueberspringen:** Section 3-4 (audiovisuelle Sensorik, verkhoerperte Bewegung), Appendix

---

## Kernaussage
**Sensonaut**: Computational Model für verkörpertes audiovisuelles Suchverhalten. Formalisiert als **POMDP** (Partially Observable Markov Decision Process). Nutzer balancieren Aufwand, Zeit und Genauigkeit unter Unsicherheit durch Kopfrotation und Fortbewegung.

## Methodik & Modell
- **POMDP-Framework:** Agent hält Belief-State über Zielposition aufrecht
- Kombination von Audio- und Visual-Likelihood + Prior → Belief-Update
- **Resource-Rational Policy:**
  
  $$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^T \gamma^t (U(b_t, a_t) - C(a_t))\right]$$
  
  mit $U$ = Utility (Treffsicherheit), $C$ = Kosten (Zeit, physischer Aufwand)
- Aktionen: Kopf drehen, vorwärts gehen, verharren, committen

## Hauptbefunde
- Menschen suchen nicht bis zur Gewissheit — sie stoppen wenn weiteres Suchen den Aufwand nicht rechtfertigt
- Resource-Rationality erklärt bounded Search-Strategien besser als optimale Suche
- Modell reproduziert menschliches Such-Timing und Bewegungsmuster

## Relevanz für meine Thesis
- **Kapitel:** Related Work → Breite des CR/RL-Simulationsparadigmas
- **Argument:** Zeigt dass Oulasvirtas CR-Ansatz weit über GUI-Evaluation hinausgeht — audiovisuelle, verkörperte Szenarien, POMDP-Framework. Unterstreicht Generalität des Paradigmas
- **Formel:** Die Resource-Rational Policy-Formel ist strukturell dieselbe wie in Stage 2 — kann als Einordnung der eigenen Pipeline genutzt werden
- **Zitierbar für:** Kurzreferenz als Breitenbeleg für das CR-Simulationsparadigma

> **Lesen nötig?** Nein — zu weit von GUI-Evaluation entfernt. Note reicht als Kontext-Zitat.

---

**Tags:** #simulation #POMDP #resource-rationality #audiovisual #embodied #oulasvirta
