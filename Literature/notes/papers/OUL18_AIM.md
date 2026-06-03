# Aalto Interface Metrics (AIM): A Service and Codebase for Computational GUI Evaluation (2018)

**Autoren:** Oulasvirta, A., De Pascale, S., Koch, J., Langerak, T., Jokinen, J., Todi, K., Laine, M., Kristhombuge, M., Zhu, Y., Miniukovich, A., Palmas, G., & Weinkauf, T.  
**Quelle:** ACM Symposium on User Interface Software and Technology (UIST), Berlin  
**DOI:** https://doi.org/10.1145/3266037.3266087  
**PDF:** OUL2018.pdf  
**ID:** OUL18 | **Prio:** 1 | **Status:** 🟢 Analyzed  

---

## Kernfrage
Wie kann GUI-Evaluation automatisiert und zugänglich gemacht werden — über eine einheitliche Webservice-Plattform mit berechneten Metriken?

## Methode
- Entwicklung des AIM-Webservices: API + Codebase für computational GUI evaluation
- Implementierung verschiedener Wahrnehmungsmodelle als Metriken (Clutter, Saliency, Visual Complexity etc.)
- Poster-Paper / System-Paper — kein klassisches Experiment

## Wichtigste Ergebnisse
- GUI-Evaluation kann vollständig automatisiert werden via Webservice
- Metriken basieren auf **Wahrnehmungsmodellen** (bottom-up, image-based)
- Erste offene Plattform dieser Art für automatisierte Metrik-Evaluation
- Zeigt: aktuelle Metriken sind **rein bildbasiert** — kein Task-Kontext, keine dynamische Load-Vorhersage

## Relevanz für meine Thesis
> Defines the technical architecture (Webservice/API) that we want to improve.

- **Rolle:** Baseline / Ausgangspunkt — das ist das System das ich erweitere
- **Kapitel:** Hintergrund / Related Work + Methodik (Systembeschreibung)
- **Argument:** AIM zeigt, dass computational GUI evaluation möglich ist, aber kognitive Task-Kontextualisierung fehlt → das ist die Lücke meiner RQ
- **Direkt zitierbar für:** Definition von AIM, Beschreibung des Status Quo, Motivation der Erweiterung

## Kritik / Offene Fragen
- Nur Poster-Paper → wenig methodologische Tiefe
- Metriken sind bottom-up: kein Nutzermodell, kein Task
- Fehlendes Cognitive Metric: Clutter ≠ Cognitive Load

## Verbindungen zu anderen Papern
- Ergänzt durch → OUL22 (Computational Rationality: liefert das theoretische "Warum")
- Ergänzt durch → JOK20 (konkrete kognitive Metrik: Search Time)
- Lücke belegt durch → DAS24 (zeigt: Saliency-Maps müssen load-adjustiert sein)

## Meine Notizen (aus Excel)
- **Computational Model:** Various (Clutter, Saliency)
- **Gap filled:** Established the first open platform for automated metric evaluation
- **Relation to AIM:** Baseline — defines the technical architecture we want to improve. Shows current metrics are mostly image-based (bottom-up)
- **Cluster:** Evaluation Tools / Gap Evidence

---

**Tags:** #AIM #GUI-evaluation #computational-metrics #baseline #platform #bottom-up #saliency #clutter
