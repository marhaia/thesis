# Out of One, Many: Using Language Models to Simulate Human Samples (2023)

**Autoren:** Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, David Wingate (BYU)  
**Quelle:** *Political Analysis*, 31(3), 337–351. DOI: 10.1017/pan.2023.2  
**PDF:** `Tier3_Ueberfliegen/ARG23_Out_of_One_Many.pdf`

---

## Kernaussage
GPT-3 besitzt **"Algorithmic Fidelity"** — durch Conditioning auf soziodemografische Backstories erzeugt es Antwortverteilungen, die reale menschliche Subgruppen (Frauen, Männer, Altersgruppen, Ethnizitäten, politische Gruppen) zuverlässig widerspiegeln. Bias ist nicht uniform, sondern demografisch fein-granular.

## Methodik
- "Silicon Samples": GPT-3 wird auf tausende echte Befragungs-Backstories konditioniert
- Vergleich silicon-generierter Antworten mit realen Surveydaten (US-Politik, public opinion)
- 4 Kriterien für Algorithmic Fidelity:
  1. **Social Science Turing Test** — generierte Antworten ununterscheidbar von menschlichen
  2. **Backward Continuity** — Antworten konsistent mit soziodem. Input
  3. **Forward Continuity** — Antworten folgen natürlich aus dem Kontext
  4. **Pattern Correspondence** — reflektiert empirische Muster aus echten Daten

## Hauptbefunde
- Gleiche LLM-Instanz kann Pro- und Anti-Standpunkte zuverlässig simulieren — je nach Conditioning
- Politische Einstellungen, Wertvorstellungen, demografische Interaktionen werden repliziert
- Fein-granulare Konditionierbarkeit (Black immigrants, female Republicans etc.)

## Relevanz für meine Thesis
- **Kapitel:** Limitations (Personality Layer)
- **Argument:** Konzeptuelle Grundlage für LLM-Persönlichkeitssimulation — aber: die Arbeit zeigt demografische Subgruppen, keine individuellen kognitiven Zustände. Kein Beleg für Fatigue/Stress-Simulation
- **Zitierbar für:** "Argyle et al. (2023) zeigen, dass LLMs demographische Subgruppen mit hoher Fidelity simulieren können. Jedoch ist nicht geklärt, ob dieses Prinzip auf akute kognitive Zustände wie Stress oder Fatigue übertragbar ist"
- **Limitation meiner Arbeit:** Personality Prompting ist konzeptuell gerechtfertigt für stabile Traits (Big Five), aber nicht für dynamische Zustände

> **Lesen nötig?** Nein — Note reicht für Limitations.

---

**Tags:** #limitations #personality-layer #LLM-simulation #algorithmic-fidelity #demographics
