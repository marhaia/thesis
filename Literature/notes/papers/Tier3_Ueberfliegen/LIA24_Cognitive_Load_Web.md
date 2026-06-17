# Cognitive Load in Web Interface Design (2024)

**Autoren:** Qian Liao (Lanzhou University of Technology)  
**Quelle:** *IEEE IAEAC 2024* (Conference on Advanced Electronic Communication)  
**PDF:** `Tier3_Ueberfliegen/LIA24_Cognitive_Load_Web.pdf`  
⚠️ **Methodische Einschränkung: Simulierte Daten** (keine echten Versuchspersonen)

---

## Beim Ueberfliegen - gezielt lesen
> Note reicht - alle relevanten Zahlen stehen schon unten. PDF nur zur eigenen Verifikation.

**Direkt springen zu:**
- **Table 1 / ANOVA-Ergebnisse (ca. S. 4-5):** F-Werte und p-Werte fuer Komplexitaet + Task
- **Figure CL vs. Complexity (ca. S. 5):** Visualisierung des Hauptbefunds
- **Section 4 Results - erste 2 Absaetze:** Zusammenfassung der Hauptbefunde

**Ueberspringen:** Methods (nur Simulation), Related Work, Discussion

!!! ACHTUNG: Simulierte Daten - kein echter empirischer Beleg!
Nur als ergaenzende Referenz zitieren, primaer auf DAS24 und MIN15 verweisen.

---

## Kernaussage
Zweiweg-ANOVA auf simulierten Daten (225 Teilnehmer, 3×3 Design): Visuelle Komplexität UND Task-Schwierigkeit haben unabhängige, signifikante Effekte auf Cognitive Load und Reaktionszeit — **kein signifikanter Interaktionseffekt**.

## Methodik
- **3 × 3 Design:** 3 Komplexitätsstufen (low/medium/high) × 3 Schwierigkeitsstufen (simple/medium/complex), N=25 je Zelle
- **Messgrößen:** Reaktionszeit, Cognitive Load Scores, Fehlerrate, NASA-TLX Subskalen
- Path Analysis via OLS-Regressionen für direkte Effekte

## Statistische Hauptbefunde

| Effekt | Reaktionszeit | Cognitive Load |
|--------|--------------|----------------|
| Komplexität | F(2,216)=68.86, p<0.001 | F(2,216)=115.03, p<0.001 |
| Schwierigkeit | F(2,216)=36.39, p<0.001 | F(2,216)=67.96, p<0.001 |
| Interaktion | n.s. | n.s. |

- Reaktionszeit: ~2s (low complexity) → ~4s (high complexity)
- Korrelation RT ↔ Cognitive Load: r=0.55, p<0.001

## Relevanz für meine Thesis
- **Kapitel:** Stage 1 / Theoretische Grundlage
- **Argument:** Empirische Bestätigung (mit Vorsicht wegen simulierter Daten): Visuelle Komplexität (Stage 1 Feature) beeinflusst Cognitive Load messbar und signifikant
- **Zitierbar für:** "Liao (2024) zeigt, dass visuelle Komplexität Cognitive Load signifikant erhöht (F=115.03, p<0.001) — ein Befund, der die Relevanz von Stage 1 Komplexitätsmetriken empirisch stützt"
- **Wichtige Einschränkung beim Zitieren:** Simulierte Daten → nur als ergänzender Beleg, nicht als Hauptreferenz. Primär auf echte Studien (z.B. MIN15, ROS07) verweisen

> **Lesen nötig?** Nein — Note reicht. Zahlen bei Bedarf hier nachschlagen.

---

**Tags:** #stage1 #visual-complexity #cognitive-load #ANOVA #empirical #simulated-data
