# Supervisor-Meeting — Validierungsstrategie
**Erstellt:** 02.06.2026

---

## Hintergrund (kurz)

Die Pipeline ist technisch fertig (Stage 1 + 2), aber der Cognitive Load Score ist noch nicht empirisch validiert — d.h. es gibt keine Überprüfung, ob die Ausgaben mit realem kognitivem Aufwand von Nutzern übereinstimmen.

---

## Was ich analysiert habe

### Vorhandene Datensätze im Workspace:

| Datensatz | Was drin | Taugt als CL-Ground-Truth? |
|---|---|---|
| **HCEye** (Das et al. 2024, ETRA'24) | Eye-Tracking von 27 Probanden auf 150 Webseiten unter verschiedenen CL-Bedingungen (Dual-Task) | ⚠️ Bedingt — CL war exp. Bedingung, kein Score pro Screenshot |
| **UEyes** (Jiang et al. 2023, CHI'23) | Eye-Tracking von 62 Probanden auf 1.980 UI-Screenshots, Fixation Maps, Heatmaps | ❌ für CL — misst Saliency/Aufmerksamkeit, nicht kognitiven Aufwand |

**UEyes kann jedoch die Saliency-Komponente (Stage 1b / UMSI++) validieren** — das ist trotzdem wertvoll.

---

## Die 3 Validierungsoptionen

### Option 1: Nutzerstudie *(stark, aufwendig)*
- Probanden schauen auf diverse GUIs + füllen NASA-TLX aus
- Vergleich: NASA-TLX-Score ↔ Modell-Output
- Problem: Zeit, Ethikantrag, Rekrutierung

### Option 2: Bestehende Datensätze *(realistisch, aber eingeschränkt)*
- HCEye: Prüfen ob Modell unter "hoher Last"-Bedingung höher scored als unter "niedriger Last"
- UEyes: Saliency-Validierung (nicht CL)
- Limitation: Kein echter CL-Score pro Screenshot verfügbar

### Option 3: Experten-Validierung *(schnell, schwach)*
- 3–5 UX-Experten ranken dieselben Screenshots nach kognitivem Aufwand
- Vergleich via Spearman-Rangkorrelation (ρ) oder Kendall's τ
- Realistisch in 2–3 Tagen umsetzbar
- Allein nicht ausreichend, aber gut als Ergänzung

### Option 2+3 kombiniert *(empfohlene Strategie?)*
- Option 2 als **interne Validierung** (Datensatz-Korrelation)
- Option 3 als **externe Validierung** (Experten-Ranking)

---

## Fragen für den Supervisor

1. **Reicht Option 2 + 3 kombiniert** als Validierungsstrategie, oder brauchen wir zwingend NASA-TLX-Daten (Option 1)?
2. **Ist Option 1 noch zeitlich machbar?** (Ethikantrag, Probanden-Rekrutierung)
3. **Zählt UEyes-Saliency-Validierung** als Teil der Validierung — oder muss es explizit Cognitive Load sein?
4. **HCEye-Bedingungsvergleich:** Reicht es zu zeigen, dass mein Score unter der Dual-Task-Bedingung systematisch höher ist?
5. **Scope-Abklärung:** Was erwartest du als Mindestanforderung für die wissenschaftliche Aussagekraft der Thesis?

---

## Meine Einschätzung (zur Diskussion)

Die methodisch sauberste Lösung wäre **Option 1 + 2** (Nutzerstudie + Datensatz-Korrelation). Wenn das zeitlich nicht mehr realistisch ist, wäre **Option 2 + 3** ein vertretbarer Kompromiss — mit expliziter Limitation in der Thesis-Diskussion.
