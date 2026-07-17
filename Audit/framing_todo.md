# Framing / honesty todo (construct validity)

Deferred deliberately: these are thesis-prose + UI-label changes, not code. No
regression risk, so they can be done at the end. Each item needs a decision
(keep / rename / soften / delete). Findings F05-F14 from the July 2026 audit.

Working principle: the tool computes rule-weighted composites and pixel-level
heuristics. It does NOT measure cognitive load, does NOT run a WCAG conformance
audit, and did NOT run any driving/glance study. Wording should match that.

## 1. "Cognitive Load Score" -> exploratory heuristic
The headline is a rule-weighted composite, not a validated CL measure.
Decide: rename (e.g. "Interaction-Complexity Score" / "Visual-Complexity
Heuristic") or keep the name but add an explicit "exploratory, not a validated
cognitive-load measurement" caveat.
- stage1/ui/index.html:3918 - summary-label + tooltip ("Overall estimated mental effort")
- stage1/ui/index.html:2049 - section title "Stage 2 - Cognitive Load Estimation"
- stage1/ui/index.html:2081 - "Cognitive Load Score (0-100)"
- stage1/ui/index.html:1811, 2745 - "HCEye cognitive load model"
- stage1/ui/index.html:2901, 2939, 2987 - internal labels/comments
- Thesis prose: wherever the headline is described as measuring cognitive load.

## 2. "Time constraints directly increase cognitive load (Kahneman, 1973)"
"directly" overstates a modelled ordinal weight.
- stage1/ui/index.html:1744 - soften to "are modelled to increase estimated load".
- stage1/ui/index.html:1738 - tooltip "More pressure raises cognitive load".

## 3. WCAG "compliance" -> "WCAG-informed check"
The tool checks a contrast ratio against a threshold; it is not a conformance audit.
- stage1/ui/index.html:1946, 1951, 1956 - "Contrast & Legibility (WCAG 2.1)" section
- stage1/ui/index.html:2104-2106 - pipeline node "WCAG Contrast"
- stage1/ui/index.html:3258-3259 - contrast_report rendering comment
Decide: relabel to "WCAG 2.1-informed contrast check (not a full conformance audit)".

## 4. Driver-glance / automotive check -> untested analogy
No driving or glance study was run; the 2s/12s budgets are borrowed thresholds
applied to a modelled search time.
- stage1/ui/index.html:1921-1931 - glance metrics block (NHTSA 2013 / ISO 15008)
- stage1/ui/index.html:2998-3038 - showGlanceMetrics / "compliant" badge
- stage1/ui/index.html:3667, 3695, 3445 - show/hide wiring
Decide: either (a) clearly mark as "illustrative analogy, not validated for
automotive use", or (b) hide this panel for the thesis version.

## 5. "above published SOTA" saliency phrasing
Per repo memory (citations_verified.md) this was already softened. Verify no
stray "above SOTA" / "CC=0.896" claims remain.
- grep: "SOTA", "0.896", "0.806" across stage1/ui/index.html and thesis prose.

## 6. General sweep
- Search thesis .tex/.md + UI for: "measures cognitive load", "compliant",
  "guarantees", "proven", "validated" - replace with modelled/estimated/
  exploratory wording where the claim is not backed by a study.
