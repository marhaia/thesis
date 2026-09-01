# Technical Audit Trail

This file records the verification policy for the thesis pipeline. It is a
technical audit trail, not a scientific validation report and not evidence of
measured human cognitive load.

## Scope

The audits assess:

- immutable candidate identity and source-tree reconstruction;
- runtime, dependency, checkpoint, OCR-model, corpus, and reference provenance;
- deterministic x19 and layout-score execution;
- input validation and fail-closed score-bearing behavior;
- separation of Stage-1 outputs, the Stage-2 scenario proxy, and optional
  diagnostics;
- API, UI, export, and documentation claim consistency; and
- regression coverage beyond named happy-path tests.

They do not establish peer review, end-to-end construct validity, empirical
effect sizes, participant-level prediction accuracy, or generalization to an
unmeasured application domain.

## Release gates

| Gate | Required evidence |
| --- | --- |
| Identity | Tag, commit, parent, tree, archive reconstruction, and SHA-256 manifest agree |
| Runtime | Recorded platform and all locked distributions verify |
| Artifacts | UMSI++, EasyOCR, reference pack, and external corpus identities match |
| Regression | Authoritative, retained heavy, evidence, and conditional tests are classified |
| Production | Real model startup and public score-bearing route complete |
| Determinism | Same image yields one x19 byte identity and one layout-score byte identity across all 15 contexts |
| Boundaries | Invalid inputs and mandatory-stage failures produce no score-bearing result |
| Claims | UI, API, exports, README, and technical documentation use the bounded terminology |
| Independence | Each clean-room auditor starts from the immutable handoff without reading the other report |

## Recorded status

- **Stage 1:** frozen at `stage1-technical-freeze-v1`; accepted by the final
  Stage-1 technical clean-room audit process.
- **Stage 2 v1:** release candidate. Its final row must not be marked PASS until
  two independent auditors have assessed the same immutable release tree after
  repository finalization.

### Candidate history

- **Technical v1.0.0-rc.2 (2026-08-19): rejected.** The immutable candidate is
  retained as evidence. Cross-audit confirmed three acceptance blockers: an
  incomplete optional-Jokinen geometry boundary, dtype-dependent artificial
  structure for a resized constant float64 saliency input, and the absence of
  the active contract's candidate-bound production-postprocess replay. The real
  UMSI++/EasyOCR baseline, x19, layout value, and all 15 deterministic Stage-2
  combinations remained stable; this was not a release PASS.

## Thesis and supplementary material

The thesis appendix should summarize each candidate by tag, commit/tree,
environment, executed gates, findings, resolution commit, and verdict. Sanitized
full reports and their SHA-256 hashes may be supplied as supplementary material.
Local user paths, tokens, account data, and unrelated workspace information must
not be published.

The correct bounded conclusion is that the audits support implementation
integrity, reproducibility, deterministic behavior, failure containment, and
claim consistency. They do not validate the exploratory layout-complexity
index as a measure of human cognitive load.
