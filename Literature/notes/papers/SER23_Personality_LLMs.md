# Personality Traits in Large Language Models (2023)

**Autoren:** Gregory Serapio-García, Mustafa Safdari, Clément Crepy et al. (Google DeepMind / Cambridge)  
**Quelle:** arXiv preprint (arXiv:2307.00184) ⚠️ Preprint — peer review ausstehend  
**PDF:** `Tier3_Ueberfliegen/SER23_Personality_LLMs.pdf`

---

## Kernaussage
Umfassende psychometrische Validierung von Persönlichkeit in LLMs. 18 LLMs mit Big-Five-Tests (1.250 resampelte Konfigurationen pro Modell) getestet. Persönlichkeit in LLMs ist mess-, validier- und steuerbar — aber nur zuverlässig bei größeren, instruction-fine-getuned Modellen.

## Methodik
- Psychometrische Validierung nach Konstruktvaliditäts-Kriterien (Reliabilität, Konvergenzvalidität, Diskriminanzvalidität, Kriteriumsvalidität)
- 18 LLMs getestet (GPT-Familie, PaLM etc.)
- 1.250 Resampling-Durchläufe pro Modell — um Output-Variabilität zu erfassen
- Personality Shaping: gezielte Prompts steuern Big-Five-Dimensionen in gewünschte Richtung

## Hauptbefunde
1. Personality-Messungen in LLMs sind **reliabel und valide** — unter spezifischen Prompting-Konfigurationen
2. Reliabilität ist stärker bei **größeren und instruction-fine-getuned** Modellen
3. Persönlichkeit in LLM-Outputs kann **entlang gewünschter Dimensionen geformt** werden
4. Synthetische Personality-Profiles imitieren spezifische menschliche Gruppen zuverlässig

## Relevanz für meine Thesis
- **Kapitel:** Related Work → Personality Layer / Limitations
- **Argument:** Stärkste technische Grundlage für Big-Five-Prompting im User Profile — zeigt dass dies psychometrisch fundiert und nicht willkürlich ist
- **Zitierbar für:** "Serapio-García et al. (2023) demonstrieren, dass Persönlichkeitsmerkmale in LLMs reliabel und valide messbar sind und durch Prompting entlang der Big-Five-Dimensionen gesteuert werden können"
- **Einschränkung:** arXiv-Preprint — als solches deklarieren. Empfehlung: parallel auf eine peer-reviewed Quelle verweisen (z.B. ARG23)

> **Lesen nötig?** Nein — Note reicht für Personality Layer Abschnitt.

---

**Tags:** #personality-layer #LLM-simulation #Big-Five #psychometrics #preprint
