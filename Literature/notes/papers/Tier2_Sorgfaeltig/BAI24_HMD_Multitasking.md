# Heads-Up Multitasker: Simulating Attention Switching on Optical Head-Mounted Displays (2024)

**Autoren:** Yunpeng Bai, Aleksi Ikkala, Antti Oulasvirta, Shengdong Zhao, Lucia J Wang, Pengzhi Yang, Peisen Xu  
**Quelle:** CHI 2024, ACM  
**DOI:** 10.1145/3613904.3642540  
**PDF:** `Tier2_Sorgfaeltig/BAI24_HMD_Multitasking.pdf`

---

## Kernfrage
Wie kann man Aufmerksamkeitswechsel zwischen zwei simultanen Tasks (Lesen auf OHMD + Gehen) computational modellieren?

## Methode
BAI24 modelliert OHMD-Multitasking als **dreistufiges hierarchisches POMDP-System** in MuJoCo. Die oberste Ebene ist der **Supervisory Controller (SC)**, der bei jedem Long-Range-Timestep binär entscheidet, ob der Agent auf dem OHMD liest oder die Umgebung scannt. Die mittlere Ebene besteht aus einem **Read-Modell** (sequentielle Wortfixation inkl. Bayesianischer Reading-Resumption nach Unterbrechung, Section 4.2) und einem **Scan-Modell** (Umgebungsscan nach Ereignissen, Section 4.3). Die unterste Ebene enthält **Oculomotor Control** (pixelbasiertes Auge in MuJoCo, 90° FoV, 80×80px RGB-D) und **Locomotion Control** (Spurwechsel oder Geschwindigkeitsregelung). Die Reward-Funktion des SC lautet `R = w_r × R_r + w_w × R_w − w_s × C_s − P_t`, wobei die drei Gewichte `w_r / w_w / w_s` direkt die individuelle Aufgabenpriorität repräsentieren. Drei Agenten-Typen wurden mit unterschiedlichen Gewichtungen trainiert: Shakespeare (Lese-Priorität), Norman (ausgeglichen), Olaf (Geh-Sicherheit). Vier Studien validieren das Modell, davon zwei mit N=12 Probanden gegen empirische Eye-Tracking-Daten (Study 2+4).

## Wichtigste Ergebnisse
- **Study 2 (Reading Speed Ratio, N=12):** Das Modell repliziert den Einfluss von Gehperturbationen auf die Leserate präzise: Mensch 94.38% (SD=4.47) vs. Simulation 94.12% (SD=4.45); Permutationstest p=0.365 — die Nullhypothese gleicher Mittelwerte kann nicht abgelehnt werden
- **Study 3 (Reading Resumption, N=12):** Über drei OHMD-Layouts (L0 / L50 / L100) beträgt der durchschnittliche RMSE 0.58s für Zeitkosten und 3.03% für Fehlerrate; Kruskal-Wallis H=570.8 (Zeitkosten) und H=753.5 (Fehlerrate), jeweils p<0.001 — der Trend „mehr Abstand = weniger Resumption-Fehler" wird modellseitig korrekt repliziert
- **Study 4 (Gesamtmodell, N=12):** Alle fünf Metriken stimmen mit menschlichen Daten überein: Aufmerksamkeitsallokation 13.19% vs. 13.33%, Gehgeschwindigkeit 51.85% vs. 51.88% (%PWS), Leserate 95.52% vs. 93.13%, Resumption-Zeit 1.91s vs. 1.90s, Fehlerrate 1.53% vs. 1.52%; nur die Reading Speed Ratio zeigt einen signifikanten Unterschied (p<0.001 nach Bonferroni), was die Autoren auf hohe Varianz in menschlichen Rohdaten zurückführen
- **Study 1 (Simulation):** Aufmerksamkeitswechsel emergiert ohne explizite Regeln allein aus den Reward-Gewichten — Shakespeare wechselt seltener und präziser an Satzgrenzen; Olaf wechselt häufiger und akkumuliert weniger Gehfehler; höhere Gehgeschwindigkeit erzwingt bei allen Agenten mehr Switches
- **Ablation (Section 8.2.2):** Ohne Position Memory Module (PMM) steigt RMSE auf 1.12s / 13.63%; ohne Re-entry Position Module (RPM) auf 1.11s / 18.51% — beide Bayes-Komponenten sind für korrekte Reading-Resumption-Simulation zwingend notwendig

## Direkt zitierbare Schlüsselsätze

> *"This reward function formulation provides flexibility for modelers to design agents with different task preferences by modifying the reward component weights."* (Section 4.1)

> *"Our model offers a more interpretable alternative with little dependence on human data."* (Section 10)

> *"The model generates behaviors by estimating the optimal policy that maximizes rewards under its beliefs about the world and its internal capacities."* (Section 10)

## Verwendung in der Thesis — nach Kapitel

### Kapitel 2: Related Work / Stand der Forschung (★★★ zentral — Dual-Task-Brücke)
BAI24 ist die stärkste verfügbare Referenz, um das **strukturelle Isomorphismus-Argument** zwischen OHMD-Multitasking und Automotive HMI zu machen: Das Dual-Task-Problem „Lesen auf OHMD + Gehen" und das Problem „Fahren + Dashboard-Interaktion" teilen dieselbe kognitive Grundstruktur — ein Supervisory Controller wägt zwischen einer primären Sicherheitsaufgabe und einer sekundären Informationsaufgabe ab, wobei die relative Gewichtung die Häufigkeit und den Zeitpunkt von Aufmerksamkeitswechseln bestimmt. Section 4.1 zeigt explizit, dass diese Abwägung über parametrisierte Prioritätsgewichte vollständig modellierbar ist, ohne den Wechsel als Regel hart zu kodieren. Direkt zitierbar für ein Argument wie: *„Die von Bai et al. (2024) gezeigte Emergenz adaptiver Aufmerksamkeitswechsel aus einem POMDP-basierten Supervisory Controller bestätigt, dass Kognitive Last im Dual-Task-Kontext eine Funktion von Aufgabenpriorität, Interface-Design und den Wechselkosten ist — eine Grundannahme, die der vorliegende Ansatz für das Automotive-HMI-Szenario operationalisiert."* Gemeinsam mit JOK21 bildet BAI24 die empirische Klammer zwischen Mobilitätsliteratur und Automotive-Kontext.

### Kapitel 3: Methodik / Stage 2 (★★ relevant — Reward-Gewichte als Validierungsargument)
Die Reward-Funktion `R = w_r × R_r + w_w × R_w − w_s × C_s − P_t` aus Section 4.1 ist strukturell identisch mit der Gewichtungslogik in `stage2/task_descriptor.py` (`TASK_TYPE_WEIGHTS`, `TIME_PRESSURE_WEIGHTS`): Beide Ansätze modellieren Aufgabenpriorität als skalaren Multiplikator auf eine Basis-Reward- oder Feature-Kombination, und beide generieren dadurch unterschiedliche Verhaltensmuster aus demselben Basismodell heraus. Study 1 zeigt, dass drei verschiedene Gewichtungskonfigurationen (Shakespeare / Norman / Olaf) drei empirisch unterscheidbare Multitasking-Stile produzieren, die intuitiv plausibel sind. Das ist eine externe empirische Rechtfertigung dafür, dass die parametrisierte Task-Priority in Stage 2 kein willkürliches Design-Feature, sondern ein kognitiv fundiertes Prinzip ist — auch ohne eigene RL-Implementierung.

### Kapitel 6: Diskussion (★★ relevant — Abgrenzung + Limitation-Parallele)
Section 10 enthält den Satz *„our model offers a more interpretable alternative with little dependence on human data"* — denselben Anspruch, den die vorliegende Thesis für den feature-basierten Stage-2-Ansatz erhebt. Damit entsteht eine wichtige Positionierungsstruktur: BAI24 ist interpretierbar *innerhalb* des RL-Paradigmas (durch explizite Reward-Gewichte und beobachtbare Policy), erfordert aber trotzdem ein physik-simuliertes 3D-Environment, Policy-Training pro Szenario und pixelbasierte Wahrnehmung — der vorliegende Ansatz erzielt Interpretierbarkeit ohne jegliche Simulation, allein auf Basis eines Screenshots. BAI24 nennt in Section 11 als Limitation ausdrücklich, dass nur aggregierte Metriken validiert wurden und keine Moment-zu-Moment-Evaluation möglich war. Dieselbe Limitation gilt für Stage 2 (CL-Score ist ein skalarer Index, kein Scanpfad), und sie kann in der Diskussion offen eingestanden und in einen breiteren Kontext eingebettet werden: *„Diese Einschränkung ist kein Spezifikum des vorliegenden Ansatzes, sondern charakteristisch für CR-basierte Verhaltensmodelle, die auf Durchschnittsvorhersagen optimiert sind (Bai et al., 2024; Oulasvirta et al., 2022)."*

> **Was du lesen musst:** Abstract (5 Min.) + **Figure 2** (Architektur-Overview, 5 Min.) + **Section 4.1** (Reward-Funktion + Agenten-Konfiguration, 10 Min.) + **Figure 12** (Study 4 Zahlentabelle, 5 Min.) + **Section 10** (Discussion, 10 Min.)  
> ⚠️ Sections 4.2–4.5 (Modell-Details zu Okulomotorik und Lokomotion) + Sections 6–9 (Einzelstudien) = überfliegen — die aggregierten Zahlen aus Study 4 und der Interpretierbarkeits-Claim aus Section 10 reichen

## Kritik / Offene Fragen
- Der Kontext-Transfer OHMD→Automotive ist argumentativ belastbar (gleiche CR-Grundstruktur), aber nicht trivial: Fahren ist eine hochautomatisierte Sensomotor-Aufgabe mit variablem Unfallrisiko, Gehen ist kognitiv wenig belastend — der Safety-Koeffizient `w_w` des SC müsste im Automotive-Kontext erheblich höher kalibriert werden als in BAI24's Standard-Setup
- Das Modell operiert in einer stark vereinfachten MuJoCo-Welt (3×4 Wort-Gitter statt echter Wörter, begrenzte Umgebungsobjekte) und verallgemeinert durch Parameter-Inferenz aus N=12 — die externe Validität für komplexe Automotive-HMI-Layouts mit variabler Informationsdichte ist nicht belegt
- Study 4 testet ausschließlich das L100-Layout in einer einzigen Umgebung (rechteckiger Pfad mit 8 fixen Schildern), was die ökologische Validität der Zahlenwerte einschränkt
- Section 11 konzediert selbst die Limitation aggregierter Metriken: das Modell kann keine Einzelereignisse, sondern nur Durchschnittstrends validieren — individuelle Unterschiede in Aufmerksamkeitsstilen werden durch die drei Agenten-Typen nur grob approximiert

## Verbindungen zu anderen Papern
- Explizit zitiert → OUL22 [Ref 54] (CR-Theorie als Fundament), JOK20 [Ref 36] (Adaptive Feature Guidance als Vorgänger-Modell), JOK21 [Ref 35] (Multitasking beim Fahren = direkteste Automotive-Parallele im gesamten Paper-Set)
- Theorie-Kette → BAI24 ist damit eine dritte empirische Instantiierung von OUL22's CR-Framework neben JOK20 und JOK21 — das Oulasvirta-Lab hat alle drei gebaut
- Automotive-Brücke mit → JOK21: beide modellieren Dual-Task mit Supervisory Control + POMDP in verschiedenen Mobilitätskontexten (Fahren vs. Gehen); gemeinsam zeigen sie die Generalität der CR-Architektur über Kontexte hinweg
- Reward-Struktur → `stage2/task_descriptor.py`: BAI24 Section 4.1 `w_r / w_w / w_s` ↔ `TASK_TYPE_WEIGHTS / TIME_PRESSURE_WEIGHTS` — BAI24 liefert empirische Legitimation für diese Designentscheidung ohne dass Stage 2 RL implementieren muss
- Abgrenzung gegen → GUO26: BAI24 = RL + Physik-Simulation + Training; GUO26 = VLM + Screenshot; vorliegender Ansatz = Feature-Extraktion + Ridge/RF/XGB auf Screenshot — alle drei sind interpretierbar auf unterschiedlichen Ebenen, aber nur der vorliegende braucht keine Simulation und kein Training

---

**Tags:** #stage2 #multitasking #attention-switching #supervisory-control #POMDP #HRL #dual-task #automotive-parallel #task-priority-weights #reading-resumption #CHI2024 #oulasvirta-lab
