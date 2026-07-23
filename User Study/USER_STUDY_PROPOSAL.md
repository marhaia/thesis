# User Study Proposal — Empirical Validation of the Combined Pipeline

**Status:** proposal / discussion draft
**Audience:** thesis student (implementation), supervisor (sign-off)
**Depends on:** the two core bugs listed in `AUDIT_REPORT.md` (§6, "Top 3 priority
issues") must be fixed *before* this study is run — otherwise the study would
validate noise, not the intended construct.

---

## 1. Framing and rationale

Several core components of the pipeline are calibrated on **non-automotive**
data:

| Layer | Calibrated on |
|---|---|
| Saliency model (UMSI++) | UEyes (web/mobile/desktop/poster) — the auxiliary 6-way softmax head is unvalidated and not used |
| Feature-norms percentiles (`feature_norms.json`) | 1,485 UEyes images (web+mobile+desktop) |
| HCEye cognitive-load coefficients (the score that is actually reported) | Das et al., ETRA 2024 — 27 participants, 150 **webpages** |

None of these layers have ever been validated end-to-end against real human
data, in **any** domain — not web, not automotive. The pipeline is also a
combination of several independently-designed components (Stage 1 image
statistics, UMSI++ saliency, HCEye rule-based scoring, Jokinen visual-search
simulation) whose *combination* has never been tested for validity either.

**Proposed approach:**

> Run the user study in the domain where the pipeline's components are
> already calibrated (web/mobile/desktop) — which is also logistically far
> more feasible than an in-vehicle study — and validate against real human
> measures (behavioral + subjective + eye-tracking). This becomes the
> thesis's central and genuinely new contribution: **the first empirical
> validation of the combined pipeline.** Automotive transfer is then framed
> explicitly as future work, with the item 4 (UIED) finding already serving
> as a first, partial, negative data point on that transfer question.

This reframing is deliberate: it does **not** claim the tool measures
automotive cognitive load (unvalidated), and it does **not** waste effort
building/validating on a domain where several sub-components are already
known to be out of scope. It converts the domain-gap concern into the
thesis's actual contribution instead of a caveat.

---

## 2. Prerequisite (before running the study)

Fix, in this order, before data collection:
1. HCEye feature-order/semantics mismatch (`stage1/app.py` → `hceye/hceye_features.py`) — the deployed score is currently computed from saturated, misassigned inputs.
2. Inverted `visual_hierarchy` figure-ground metric (`stage1/visual_complexity.py`).

Re-run/regenerate `feature_norms.json` after the fix (see §3 for the
held-out split needed anyway).

---

## 3. Stimuli — reuse existing GUIs, no need to source new images

UEyes and HCEye stimuli can be reused directly (saves the effort of finding
or creating new GUI images), under three rules to avoid circular validation:

1. **Never reuse the original human data** (HCEye fixations, UEyes
   eye-tracking) — only the stimulus images. Ground-truth measures must come
   from newly recruited participants.
2. **Exclude the 150 HCEye lookup images** (`hceye/sensitivity_lookup.json`,
   keys like `101-UEyes-9466ef`) from the test stimulus pool, or force
   `HCEyeFeatureExtractor` to always use its "estimate from visual features"
   path rather than the per-image lookup shortcut.
3. **Rebuild `feature_norms.json` on an 80/20 split** of the UEyes images
   currently used (1,485 total, web+mobile+desktop). Draw test stimuli only
   from the held-out 20%, so no test image contributed to the reference
   norms.

Note: HCEye's 150 stimuli are themselves UEyes images (same naming
convention) — the two "independent" calibration layers are not actually
independent. Deduplicate accordingly when building the held-out pool.

**Target stimulus set:** 30–40 images per participant session, stratified to
cover a wide range of pipeline-predicted complexity/load (not just typical
images) so there is real variance to correlate against.

---

## 4. Measures

Given eye-tracking is available, validate the pipeline at multiple levels,
not just the final score:

| Pipeline output | File | Validation measure |
|---|---|---|
| Saliency map (`s`) | `saliency/saliency_features.py` | NSS / CC / KLD / AUC vs. observed fixation map (reuse `scripts/ueyes_saliency_validation.py`, feed it real fixation data instead of the 5-sample demo) |
| HCEye vector (`h`): fixation_reduction, duration_increase, exploration_reduction, aoi_sensitivity | `hceye/hceye_features.py` | Directly computable equivalents from real gaze data (fixation count, mean duration, AOI-exploration entropy) per stimulus, correlated 1:1 with predicted values |
| Jokinen scanpath + predicted search time | `cognitive/jokinen_model.py` | Scanpath similarity (edit-distance / ScanMatch or MultiMatch) between predicted and observed AOI sequence, in addition to search-time correlation |
| Glance metrics (NHTSA 2013 / ISO 15008) | `cognitive/jokinen_model.py` | Real single-glance and total-glance durations vs. predicted glance packing |
| OCR reading-time estimate | `cognitive/text_reader.py` | Real dwell time on text AOIs vs. predicted reading time (238 wpm) |
| WCAG contrast per element | `cognitive/element_detector.py` | Fixation/dwell time before detection vs. contrast ratio (tests the implicit "low contrast → harder to notice" assumption) |
| Interactive element density (`v[7]`) + element bounding boxes | `cognitive/element_detector.py` | Sanity check against manual element count (no participant needed) |
| Final cognitive-load score | `stage1/app.py` (`h[5] * 100`) | NASA-TLX (or single-item effort/difficulty rating) per trial |
| Search time (behavioral) | — | Visual search task: "find element X", measured time-to-click + accuracy |

Use the same bounding boxes from `element_detector.py` as AOIs for fixation
assignment — reuses existing code instead of manual AOI annotation.

---

## 5. Design

- **Type:** correlational/regression validation, not a manipulated-condition
  experiment. Stimuli vary naturally in complexity; no experimental
  conditions to compare.
- **Structure:** rotated/partial block design — each participant sees a
  subset of the full stimulus pool, such that across participants every
  stimulus receives enough independent responses.
- **Per trial:** free viewing/search task (eye-tracked) → click on target →
  brief subjective rating (NASA-TLX or 1-item).
- **Session length:** ~25–35 minutes for 30–40 trials.

---

## 6. Sample size

With eye-tracking, N = 20–30 participants is adequate, because each trial
yields a rich multi-fixation signal (not a single behavioral/subjective data
point), and the unit of analysis for the core validation is the *stimulus*,
analyzed via mixed-effects models with fixation/AOI as the nested unit
(rather than a single per-trial average). This is a meaningfully richer
signal than a behavioral+subjective-only design, which would need more
participants (~40–50) to reach comparable power.

---

## 7. Analysis plan

- Mixed-effects regression (random intercepts for participant and stimulus)
  relating pipeline-predicted values to observed human measures, for each
  row of the table in §4.
- Partial correlations across pipeline sub-components (v, s, h, Jokinen
  search time) to identify which carry real signal post-bugfix, vs. which
  remain noise.
- Saliency validation metrics (NSS/CC/KLD/AUC) computed per stimulus,
  aggregated and compared against published UEyes benchmarks for context.

### 7.1 Bayesian multilevel modeling (recommended over plain frequentist tests)

Given the small-to-moderate sample (N=20-30 participants, ~30-40 stimuli),
the nested participant × stimulus structure, and the exploratory nature of
this validation (first-ever empirical test of this pipeline, not a single
confirmatory hypothesis), a Bayesian multilevel approach is a better fit
than plain correlation or NHST-only mixed models:

- **Partial pooling**: with a rotated block design, some stimuli will
  receive fewer ratings/fixation samples than others. Bayesian multilevel
  models shrink noisy per-stimulus estimates toward the population mean,
  giving more stable per-stimulus estimates than an unpooled or purely
  frequentist mixed model, especially where coverage is uneven.
- **Credible intervals over p-values**: for an exploratory validation study
  with many correlations to report (saliency, HCEye sub-features, Jokinen
  scanpath/search time, glance metrics, OCR reading time, WCAG contrast,
  final score), a posterior probability statement (e.g. "92% probability
  the predicted vs. observed fixation-reduction correlation is positive and
  non-negligible") communicates the actual evidence better than a binary
  significant/non-significant read of many separate tests.
- **Practical implementation**: use `brms` (R) if already familiar with
  `lme4`-style formulas — the model syntax is nearly identical
  (`brm(score ~ predicted + (1|participant) + (1|stimulus))` vs.
  `lmer(...)`), so the added cost is mainly conceptual (priors, MCMC
  diagnostics), not syntactic. Weakly-informative default priors are
  sufficient; no need for elaborate prior elicitation.
- **Scope trade-off**: full Bayesian modeling for every row in §4 may not be
  worth the added time given the rest of the study's workload (bugfixes,
  IRB, data collection). A reasonable compromise: Bayesian multilevel
  modeling for the **central claim** (final cognitive-load score vs.
  NASA-TLX/behavioral measures), and simpler frequentist mixed models /
  point estimates with confidence intervals for the secondary
  sub-component checks (saliency NSS/CC, scanpath similarity, glance
  metrics, etc.).

---

## 8. Practical prerequisites

- IRB/ethics approval for a study involving human behavioral, subjective,
  and eye-tracking data — apply early, before scheduling.
- Confirm UEyes CC-BY-4.0 licensing terms are compatible with reuse in a
  human-subjects study (attribution required).
- Lab eye-tracker availability/calibration procedure.

---

## 9. Scope and future work

This study validates the pipeline **in the domain its components were
calibrated on** (web/mobile/desktop). It deliberately does **not** claim
automotive validity. Automotive transfer remains an open question, framed
as future work, with the item 4 (UIED/Rico) experiment already providing a
first, partial, negative data point: an off-the-shelf detector trained on a
different domain (Rico/Android) fails to recognize automotive interactive
controls once applied out of its training domain — the same category of
risk this study is designed to address for saliency, HCEye, and Jokinen.
