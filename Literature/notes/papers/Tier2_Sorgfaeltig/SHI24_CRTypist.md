# CRTypist: Simulating Touchscreen Typing Behavior via Computational Rationality (2024)

**Autoren:** Danqing Shi, Yujun Zhu, Jussi P.P. Jokinen, Aditya Acharya, Aini Putkonen, Shumin Zhai, Antti Oulasvirta  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3642918  
**PDF:** `Tier2_Sorgfaeltig/SHI24_CRTypist.pdf`

---

## Kernfrage
Wie kann man menschliches Tippverhalten auf Touchscreens durch CR modellieren — über verschiedene Keyboards und Nutzer hinweg?

## Methode
- Supervisory Control Reformulierung: Visuelle Aufmerksamkeit + Motorsystem werden über Working Memory gesteuert
- Arbeitet direkt auf Pixeln (kein Feature Engineering)
- RL für Bewegungsplanung (asymptotisch optimal unter kognitiven Bounds)
- Validiert an Tipp-Daten: Bewegungen, Performance, individuelle Unterschiede

## Wichtigste Ergebnisse
- Modell generiert human-like Tippverhalten ohne Keyboard-spezifisches Hand-Crafting — direkt von Pixeln
- Deckt 97% der Tipp-Geschwindigkeitsverteilung ab (7.6–64.8 WPM, Table 1)
- Working Memory Parameter λ korreliert r=-0.96 (p<.01) mit Alters-bedingtem WPM-Rückgang (Table 1, Task 2.2)
- **Key motivation quote (Section 1):** *"models... provide insights into usability of prototypes before user testing, and thereby lower the costs of human-centric engineering"* — direkt zitierbar für Kap. 1
- **Key evaluation quote (Section 6):** *"simulation-based evaluation can yield immediate insights related to usability before any user testing"* — stärkste Pre-Deployment-Formulierung im Paper
- Supervisory Control Hierarchie: Supervisor → Vision/Finger/Working Memory → Pixel (Section 3.2, Figure 2)

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★★ relevant)
- Zwei direkt zitierbare Pre-Deployment-Sätze (Section 1 + Section 6) — nach JOK21 die stärksten in deinem Set
- *"Shi et al. (2024) demonstrate that simulation-based evaluation 'can yield immediate insights related to usability before any user testing' — the present pipeline instantiates this principle for the cognitive load domain of automotive HMI, providing pre-deployment estimates that complement rather than replace empirical user studies."*

### Kapitel 2: Related Work (★★ relevant)
- Supervisory Control Hierarchie = theoretischer Vorläufer für Stage-2-Architektur: übergeordnetes Modell koordiniert verschiedene kognitive Subsysteme (Gaze + Motor + Memory)
- Pixel-basiert ohne Hand-Crafting → analog zu Screenshot-Input ohne domain-spezifisches Feature Engineering in Stage 1
- *"CRTypist (Shi et al., 2024) demonstrates that supervisory control of visual attention and motor behavior — coordinated through a shared internal representation — can be learned directly from pixels without hand-crafted feature engineering. The present pipeline adopts an analogous principle: Stage 1 operates on raw screenshots, deriving cognitive load features without domain-specific manual encoding."*
- ⚠️ Abgrenzung: SHI24 = Typing (Motor + Gaze koordiniert), deine Pipeline = visuelle Suche im statischen Dashboard

### Kapitel 3: Methodik (★★ relevant)
- Section 3.3.3 Parameter Fitting: Bayesian Optimization für kognitive Parameter (EK, FK, λ) → KANK17-Methodik direkt angewendet → stützt deine Begründung für User-Profile im Task Descriptor
- λ (Working Memory Decay) als individueller Parameter → analog zu SEARCH_MODE_WEIGHTS + TIME_PRESSURE_WEIGHTS in deinem Task Descriptor

### Kapitel 5: Validierung (★ relevant)
- Table 1 Validierungsstruktur (Accuracy + Individual Differences + Generalizability) → Vorlage für deine Kap. 5 Dreiteilung
- "Model cannot cover the tails of the distribution" (Section 5.2.1) → Limitation-Formulierungsmuster für deine Kap. 5/6

### Kapitel 6: Diskussion / Future Work (★ marginal)
- Section 6 Limitations: Working Memory → Future Work (longer sequences, reading) → analog zu deiner Limitation (kein dynamischer Fahrtkontext)
- "potential extends to simulating user behavior in other interactive tasks, such as visual search" (Section 6) → direkte Verbindung zu deiner Pipeline

> **Was du lesen musst:** Section 1 (letzter Absatz, 3 Min.) + **Section 3.2** (Supervisor-Architektur + Figure 2, 10 Min.) + **Section 6** (Discussion, Pre-Deployment-Zitat, 5 Min.)  
> ⚠️ Section 2 = Human Typing Background — weglassen. Modell-Architektur ist Section 3, NICHT Section 2.

## Kritik / Offene Fragen
- Nur skilled typists (after practice) — kein Novice-Modell, kein Lernprozess
- Fokus auf Typing (Motor + Gaze koordiniert) ≠ visuelle Suche im Dashboard
- Working Memory Parameter schwer direkt auf Automotive-Kontext übertragbar
- Table 1: Modell überschätzt Gaze Shifts bei single-finger typing (5.5 vs. 3.9 human) — zeigt systematische Grenzen

## Verbindungen zu anderen Papern
- Baut auf → OUL22 (CR Framework), JOK20 (Supervisory Control Grundmodell [Ref 43 im Paper])
- Selbe Autorengruppe → SHI25/Chartist (Danqing Shi, Oulasvirta — ein Jahr später)
- Parameter Fitting → KANK17 (ABC-Methodik, Section 3.3.3)
- Pixel-basiert → MIA26 (beide: kein Hand-Crafting, direkt aus Interface-Repräsentation)
- Architektur-Parallele → LAN24 (beide: hierarchische CR-basierte User Agents)

---

**Tags:** #stage2 #CR #RL #supervisory-control #typing #computational-model #pre-deployment #working-memory #individual-differences #pixel-based
