# Redefining Affordance via Computational Rationality (2025)

**Autoren:** Yi-Chi Liao, Christian Holz  
**Quelle:** IUI '25 (30th Int. Conference on Intelligent User Interfaces), Cagliari, Italy, ACM  
**DOI:** 10.1145/3708359.3712114  
**PDF:** `Tier2_Sorgfaeltig/LIA25_Affordance_CR.pdf`

---

## Kernfrage
Wie kann das klassische Affordanz-Konzept (Gibson) durch Computational Rationality neu definiert werden — um zu erklären wie Nutzer Handlungsmöglichkeiten wahrnehmen, lernen und auswählen?

## Methode
- Theoretisches Paper (keine Nutzerstudie)
- Argumentation via Thought Experiments
- Zwei Mechanismen: Feature Recognition (was sehe ich?) + Hypothetical Motion Trajectories (was könnte ich tun?)
- Affordanz-Entscheidung = Balance aus Confidence (Erfolgswahrscheinlichkeit) + Predicted Utility

## Wichtigste Ergebnisse
- Affordanz-Wahrnehmung ist ein Entscheidungsprozess, kein direktes Wahrnehmen (Section 1: *"affordance perception is constructed, and can be computationally modeled"*)
- Zwei Kernmechanismen: Feature Recognition (vertraute Muster) + Hypothetical Motion Trajectories (Simulation neuer Situationen) (Section 3.5)
- Affordanz-Entscheidung = ⟨action, confidence, predicted utility⟩ Tripel (Section 3.4)
- Kontinuierlich gelernt und verfeinert durch Reinforcement Learning (Section 3.6)
- Erklärt false/hidden affordances als Misalignment zwischen internal und external affordance (Section 3.3)
- **Key theory quote (Section 1):** *"affordance perception is constructed, and can be computationally modeled"* — Grundlage für gesamte Pipeline
- **Key design quote (Section 6.1):** *"Adaptive systems can leverage... adjusting feedback or interface complexity based on a user's proficiency and familiarity"* — direkt zitierbar für Task Descriptor

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★ relevant)
- Section 1: "affordance perception is constructed, can be computationally modeled" → erklärt warum Screenshot-basierte Pipeline überhaupt theoretisch fundiert ist
- Section 6.1 "Improving Affordance Clarity" → Ziel deiner Pipeline = affordance clarity für Automotive-Designer vor dem Deployment

### Kapitel 2: Related Work (★★★ zentral)
- Confidence + Predicted Utility (Section 3.4) = theoretische Grundlage für deine zwei Outputs: cognitive_load_score ≈ Confidence-Kehrwert, search_efficiency ≈ Predicted Utility
- Internal Environment (Section 3.3) = direkte Parallele zu Stage 1: Screenshot-Features rekonstruieren das "internal model" des Nutzers
- Figure 1 (External vs. Internal Affordance Diagram) → könnte als Rahmendiagramm in Kap. 2 erscheinen
- *"Liao & Holz (2025) propose that affordance perception is not directly picked up from the environment but inferred within an internally constructed representation shaped by bounded sensory inputs. Stage 1 of the present pipeline operationalizes this: raw screenshots are processed to derive cognitive features that approximate the internal model a driver constructs when perceiving an automotive HMI."*
- ⚠️ Abgrenzung: LIA25 = reine Theorie (Thought Experiments, N=8), keine empirische Validierung — du bist die empirische Instantiierung

### Kapitel 3: Methodik (★★ relevant)
- Feature Recognition (Stage 1 = visual feature extraction) + Hypothetical Motion Trajectories (Stage 2 = behavioral prediction) = die zwei Mechanismen von Section 3.5, in deiner Pipeline konkret umgesetzt
- Section 6.1 Adaptive Systems = direkte Rechtfertigung für Task Descriptor als Personalisierungsebene (Novice vs. Expert, TIME_PRESSURE, SEARCH_MODE)
- *"Our theory frames affordances as dynamic, continuously learned, and refined through reinforcement and feedback"* → dein Task Descriptor ermöglicht genau diese Anpassung an individuelle Nutzerprofile

### Kapitel 5: Validierung (★ marginal)
- Thought Experiments als Validierungsformat (Section 4) → zeigt, dass theoretische Arbeiten nicht zwingend Nutzerstudien brauchen → hilft Kap. 5 framen wenn Pilotdaten begrenzt
- Section 6.2 "Further Empirical Evaluations" → dein August 2026 User Study ist genau diese Validation

### Kapitel 6: Diskussion / Future Work (★★ relevant)
- Section 6.2 "Future Computational Affordance Models": VLM + MBRL → deine Stage 1 (Vision) + Stage 2 (Behavioral Model) ist exakt diese vorgeschlagene Architektur — stärkstes Argument für Kap. 6
- Section 6.2 "Extending CR-based Affordance with other Cognitive Architectures" (ACT-R, Bayesian Brain) → Future Work Richtung für deine Pipeline

> **Was du lesen musst:** Section 1 (letzter Absatz, 3 Min.) + **Section 3.3–3.5** (Internal Environment + Inference-Mechanismen + Figure 1, 15 Min.) + **Section 6.1–6.2** (Design Implications + Future Work, 10 Min.)  
> ⚠️ Section 4 (Thought Experiments) = nur überfliegen — N=8, keine statistische Validierung. Section 5 = Unifying Framework für andere Affordanz-Typen — weglassen.

## Kritik / Offene Fragen
- Reine Theorie — Thought Experiments mit N=8 (age 25–30), keine statistische Validierung
- Anwendungsbeispiele sehr breit (physisch, digital, sozial, kulturell) — Automotive-Spezifik fehlt vollständig
- "Confidence" und "Predicted Utility" bleiben konzeptuell — keine Operationalisierung oder Messmethodik angegeben
- Section 6.2: Eigene Future Work (VLM + MBRL) wird vorgeschlagen aber nicht implementiert — du implementierst es

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR als Basis), Gibson [32]/[33] (Affordance Ursprung)
- Selber Erstautor → LIA22 (Rediscovering Affordance, CHI 2022 [Ref 45 im Paper] — RL-basiertes Affordanz-Modell, Vorläufer)
- Brücke → SHI24/CRTypist (beide: CR + RL + internal representation), MIA26 (beide: pixel-basiert)
- Task Descriptor Parallele → LAN24 (beide: User-Profile als explizite Anpassungsebene)
- Empirische Instantiierung von → Section 6.2 Future Work (VLM + MBRL = Stage 1 + Stage 2)

---

**Tags:** #stage1-to-stage2 #CR #affordance #theory #decision-making #internal-representation #confidence #predicted-utility #adaptive-systems #IUI25
