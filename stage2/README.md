# Stage 2 v1 contract

The active Stage-2 v1 implementation consists of:

- `scenario_proxy.py`: deterministic, non-score-bearing scenario proxy using
  exactly `task_type` and `time_pressure`.
- `coherence_check.py`: Exploratory Cross-Signal Review with tri-state status
  and uncalibrated inspection triggers.

Task categories (`navigation`, `search`, `monitoring`, `data_entry`,
`decision`) are scenario labels only. Time pressure maps deterministically to
`lower`, `baseline`, or `higher`. The proxy has no numeric modifier, simulated
task process, empirical effect size, or measurement claim and cannot change
the frozen Stage-1 x19 vector or layout index.

The time-pressure field is a pre-specified ordinal project hypothesis. Evidence
that time constraints can influence interactive-search behaviour, with
person- and task-dependent responses, motivates retaining the context field
(Liu et al., 2019, https://doi.org/10.1016/j.ipm.2019.04.004). That study does
not calibrate this repository's `lower` / `baseline` / `higher` mapping and is
not evidence for a universal, category-specific, or participant-level effect.

The Jokinen model is an optional, methodologically separate technical API
diagnostic exposed through the main route only when explicitly requested. It is
not exposed by the study UI, is non-score-bearing, and is not measured search,
eye tracking, or a validated behavioral prediction.
Public context fields are case-sensitive and may be supplied at most once
across form and query data. Explicit blank values and ambiguous duplicate
values fail with HTTP 400. `display_preset` accepts only `phone`, `laptop`, or
`desktop`, and only when the Jokinen diagnostic is explicitly enabled.

`task_descriptor.py`, `user_profile.py`, and `regression_model.py` are archived
legacy research scaffolds. Personality-derived effects and ML are excluded
from Stage 2 v1. The study UI may record optional study-coded Big Five
categories locally beside its export, but does not send them to the production
route, score them, infer them, or use them to modify x19 or the layout index.
