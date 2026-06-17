# Sketchplore: Sketch and Explore Layout Designs with an Optimiser (2016)

**Autoren:** Todi, K., Weir, D., & Oulasvirta, A.  
**Quelle:** CHI '16 Extended Abstracts (Demo Paper), San Jose CA, ACM  
**DOI:** 10.1145/2851581.2890236  
**PDF:** TOD16.pdf  
**ID:** TOD16 | **Status:** 🟢 Analyzed (Auswahl: KEEP — Methodology)  

---

## Kernfrage
Wie kann ein Optimierungswerkzeug Designern helfen, GUI-Layouts interaktiv zu erkunden — durch Kombination von Skizzen-Input und automatischer Optimierung?

## Methode
- Interaktives Design-Tool: Nutzer skizziert, Optimierer verfeinert
- Kombiniert visuelle (Stage 1) und sensomotorische (Stage 2) Performance-Metriken
- Layout-Optimierung basierend auf wahrnehmungsbasierten Kriterien

## Wichtigste Ergebnisse
- Demo-Paper (1 Seite, CHI Extended Abstracts) — keine empirische Evaluation, keine Zahlen
- **Key quote (Abstract):** *"Using several predictive models of user performance and perception, its suggestions steer designers toward more usable and aesthetic layouts"* — direkt zitierbar
- Architekturprinzip: Sketching-Input → Predictive Models → Local/Global Alternatives = konzeptuelle Basis der Two-Stage-Trennung
- "automatically infers the designer's task" → früheste explizite Task-Inference in der Oulasvirta-Gruppe

## Verwendung in der Thesis

### Kapitel 1: Einleitung (★ marginal)
- "steer designers toward more usable and aesthetic layouts" = kurze Pre-Deployment-Parallele, Fußnote als Entwicklungslinie (2016 → 2019 → deine Pipeline)

### Kapitel 2: Related Work (★★ relevant)
- "predictive models of user performance and perception" = frühester expliziter Beleg für die Perception-Performance-Trennung in der Oulasvirta-Gruppe
- TOD16 → TOD18 → TOD19 = Entwicklungslinie der Two-Stage-Idee, die deine Pipeline weiterführt
- *"Todi et al. (2016) introduced the concept of combining perceptual and performance-predictive models to guide layout exploration — a principle the present pipeline extends to the automotive domain through a pre-deployment cognitive load estimation architecture."*
- ⚠️ Demo-Paper: nicht als Hauptreferenz — als Beleg für die Architekturidee in der Oulasvirta-Gruppe, mit TOD19 als Hauptreferenz

### Kapitel 3: Methodik (★★ relevant)
- "automatically infers the designer's task and searches for local improvements, and global alternatives" → deine Pipeline: Task Descriptor inferiert Kontext, Stage 2 sucht im Vorhersageraum
- Sketchplore = Two-Stage-Logik (Visual Input → Task Inference → Performance Model) als methodisches Vorbild

> **Was du lesen musst:** Nur Abstract (1 Min.) — kein Full Paper vorhanden, nur Extended Abstract.  
> ⚠️ Kein Full Paper — immer zusammen mit TOD19 zitieren, nie als Einzelbeleg.

## Kritik / Offene Fragen
- Demo-Paper (1 Seite) — keine Evaluation, keine Ergebnisse, keine Methodik-Details
- Fokus auf Desktop-Design, kein Automotive HMI
- Cognitive Load nicht explizit adressiert
- Nur als Teil der TOD16→TOD18→TOD19 Entwicklungslinie zitierbar — nicht als eigenständige Referenz

## Verbindungen zu anderen Papern
- Direkte Weiterentwicklung → TOD18/Familiarisation + TOD19/Individualising Layouts (selbe Autorengruppe)
- Gleiche Autoren → OUL18/AIM (Oulasvirta/Todi-Lab, AIM ist die Weiterentwicklung des Optimierungsansatzes)
- Architektur-Parallele → Two-Stage Pipeline: Sketchplore = früheste Two-Stage-Logik in der Gruppe
- Methodik verwandt → BUR21/UiLab (beide: interaktive Design-Optimierung mit Predictive Models)

---

**Tags:** #layout-optimization #two-stage #tool #demo-paper #visual #sensorimotor #methodology #design-tool #CHI2016 #extended-abstract
