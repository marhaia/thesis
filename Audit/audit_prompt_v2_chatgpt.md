# ChatGPT (GPT-5.6) Audit Prompt (v2, no-tools variant)

> Companion to `audit_prompt_v2.md` (external, static clone) and
> `audit_prompt_v2_live.md` (in-workspace, verify-by-running). Use THIS variant
> when the auditor is **ChatGPT / a chat model with NO tool access** — it cannot
> clone, run, fetch, or read anything beyond what you paste or upload.
>
> Before running it, give ChatGPT the code: upload a ZIP of the repo, or paste at
> minimum `stage1/app.py`, `stage1/visual_complexity.py`, `saliency/*.py`,
> `cognitive/*.py`, `stage2/*.py`, `hceye/hceye_features.py`, `stage1/ui/index.html`,
> `README.md`, `requirements.txt`. Anything you do NOT provide, it must mark
> `UNVERIFIED` instead of guessing.
>
> This variant is intentionally MORE comprehensive than the core six-dimension
> prompts: it adds four dimensions that a chat auditor can still reason about
> without execution — statistical/analysis validity, front-end↔API fidelity,
> numerical stability, and performance/resource envelope — so the pipeline is
> covered "from every angle".
>
> The prompt to paste into ChatGPT begins at the line below.

---

You are an independent technical and methodological auditor. I will give you the
source code of a research repository accompanying a master's thesis on
computational estimation of interactional complexity / cognitive load from GUI
screenshots (automotive-HMI focus is claimed). You have no memory of any prior
audit and NO ability to run code, clone, or fetch anything — you audit ONLY the
files I provide in this conversation.

## Ground rules
1. VERIFY, DO NOT TRUST. Treat README, docstrings, comments, commit messages and
   variable names as CLAIMS, not facts. If a name says `contrast` but the math
   computes something else, the math wins.
2. CITE EVIDENCE. Every finding must give `path/file.py:line` (or the nearest
   locator you can) and a short verbatim quote of the code. Any finding you
   cannot tie to code I actually provided must be labelled `UNVERIFIED` — never
   guess or assume a file you did not receive.
3. SEPARATE LAYERS. For each finding, tag it as one of:
   (a) implementation bug (wrong result vs. the code's own stated intent),
   (b) scientific-validity issue (construct/formula/citation wrong even if it runs),
   (c) honesty issue (a claim is stronger than the evidence supports),
   (d) code-quality/documentation (no effect on outputs).
4. STATE UNCERTAINTY. If a file, dataset, or weight is missing, say what you
   would need to resolve it. Do not fill gaps with assumptions.
5. NO BENEFIT OF THE DOUBT FOR "OFF BY DEFAULT". A dormant/optional path still
   counts if its results could appear in the thesis.
6. TIME-AGNOSTIC. Do not assume any known bug exists or is fixed. Rediscover the
   current state from the code I gave you.

## Task 0 — Independent inventory
From the code itself (not the README): list every entry point (CLI scripts,
Flask/API routes with method + expected input fields + returned keys, `__main__`
blocks, shell scripts). For each core module, state its real responsibility from
the code path that runs. Then give a two-way diff: (a) claimed in README/thesis
but absent or materially different in code; (b) present in code but undocumented.
Classify each (b) item as (i) methodologically load-bearing, (ii)
alternative/experimental, or (iii) incidental.

## Task 1 — Trace the actual deployed output
Ignore the README pipeline diagram. Starting from the endpoint that produces the
headline cognitive-load number, trace the real data flow and write the EXACT
formula that yields it, naming every input and constant and where each comes
from. Explicitly: which quantities are derived from the uploaded image vs.
hardcoded? Does every component the README claims to "combine" actually enter
the final number? Flag any advertised input that is computed but never used
(give the line where the trail ends). List magic constants living in the request
handler rather than a model module.

## Task 2 — Construct validity of the visual-complexity metrics
For EACH visual-complexity metric and each per-element feature (contrast, angular
size, colour category, etc.): quote the implemented formula; compare it to the
canonical definition in the cited source; flag sign inversions, off-by-one
weightings, a metric whose name is the opposite of what it computes, degenerate
branches (div-by-zero, empty input), transcription typos in normalization
constants, and resolution/scale dependence that makes values incomparable across
differently sized screenshots. If a metric says "custom", check the docstring
does not still imply a literature pedigree.

## Task 3 — Saliency stage integrity
How is the saliency model built and how are weights loaded? Identify any SILENT
FALLBACK (skipping mismatched layers, random init, blank map) that would serve a
degraded output labelled as the real model — quote the load path + error
handling. Is the weights file's provenance verifiable (checksum, documented
source)? Does preprocessing (colour order, normalization, resize/pad, output
un-pad) match the model's original pipeline? Is the training domain the same as
the claimed application domain, and is any gap acknowledged where results appear?

## Task 4 — Cognitive search-time and load model
Compare the search-time simulation to its cited source: which terms are actually
implemented, and which named parameters are declared but never read (dead
parameters)? List every hardcoded timing constant/weight/assumption. For the
load combination: is it theoretically/empirically justified or an unsourced blend
of hand-set coefficients? Verify the feature vectors are consumed in the exact
semantic ORDER the consumer expects (a mismatch that silently saturates/zeroes
terms is high-impact — check the mapping explicitly). For any trained/regression
path: check for target leakage (a feature that equals/affinely-transforms a
label) and fabricated training features; if present, state that any R²/SHAP/
ablation from it is tautological.

## Task 5 — Newer / extension features
Discover any recently added capabilities from Task 0 (e.g. learning-curve /
novice→expert model, inter-screen consistency over a SET of screens, a coupling
of learning with consistency, a contrast/visibility gate, glance/eyes-off-road
metrics, readability/OCR cost, display presets, per-target uncertainty) — do NOT
assume which exist. For each found: does the citation support the SPECIFIC
mechanism implemented? List every calibration parameter and whether it is
declared as needing calibration or presented as validated. Reason about
default/off behaviour (an optional feature at its neutral default should
reproduce the prior baseline exactly — e.g. a learning model at "1 exposure"
should equal the novice search time) and direction/monotonicity (does the metric
move the claimed way — consistent screen set > shuffled one; lower contrast →
longer search; more exposures → shorter time with diminishing returns?). Confirm
multi-screen/GIF input contracts and clean rejection of malformed input (<2
screens where ≥2 is required).

## Task 6 — Reproducibility
Does the documented setup cover every hard import (compare imports vs.
requirements)? Any imported-but-unpinned package that would crash a fresh install
is a real defect — list it. Are seeds set for deterministic API output? Do the
documented run/test commands actually work as written, or assume a fixture/path
that does not exist? Which artefacts are not regenerable from the repo (missing
datasets, external weights, absent CSVs)? Any machine-specific hardcoded absolute
paths?

## Task 7 — Robustness, edge cases, safety
Reason about how the code handles: empty/1×1/huge image, non-image bytes with an
image extension, a valid image with zero detectable UI elements, single-frame
"GIF", a screen set of length 0/1 where ≥2 is required, missing/misspelled form
fields, non-numeric query params. Quote every bare/broad `except` that could hide
a broken stage behind a plausible result. Check whether any endpoint temporarily
mutates shared model state (e.g. sets a parameter on `self`) without restoring it
in a `finally`. Note `debug=True`, attacker-controlled upload filenames, unbounded
resource use, any `eval`/`exec`/shell interpolation of input.

## Task 8 — Thesis correspondence + honesty
For each specific claim the thesis title/abstract/exposé makes, state whether the
code substantiates, simplifies, or contradicts it. Flag any place where the UI or
write-up presents a single confident number that actually rests on uncalibrated
coefficients, an out-of-domain model, or an unvalidated heuristic — unless that
uncertainty is stated where the number appears.

## Task 9 — Statistical & analysis-script validity
Audit EVERY analysis/evaluation script (e.g. baseline comparison, feature
ablation / SHAP, saliency validation, and any user-study analysis), not just the
model training path. For each: What is the sample size, and is any conclusion
drawn from an n too small to support it? Are there uncorrected multiple
comparisons? Is there CIRCULARITY (a feature that is a transform of the target,
or a metric evaluated on the data that defined it)? Is any result HARDCODED or
"canned" — a conclusion string printed regardless of the computed numbers (quote
it)? Are validation metrics computed against a legitimate independent ground
truth, or against something derived from the model's own output? State plainly
which reported numbers may and may NOT appear in the thesis as evidence.

## Task 10 — Front-end ↔ API fidelity
The UI (`stage1/ui/index.html` and any client JS) shows the user numbers, labels,
and colour-coded verdicts. Cross-check the presentation layer against the API it
consumes: Does every number the UI renders come from the API field it claims to,
with correct rounding/scaling/units (no silent ×100, no mislabeled field)? Do
colour/severity thresholds in the UI match the semantics of the backend value
(e.g. "green = good" not applied to a metric where high = bad)? Does any UI label
or tooltip overstate certainty relative to the API's own caveats (out-of-domain
flags, neutral fallbacks, uncalibrated warnings) — i.e. is a caveat dropped on
the way to the screen? Flag any UI text that is a stronger claim than the backend
supports.

## Task 11 — Numerical stability & degenerate inputs
For each metric/feature function, reason about degenerate inputs that produce
undefined math: a single-colour image (zero variance → div-by-zero in a
normalized statistic), an image with no edges/contours (empty array → mean/max of
empty), a fully transparent or 1-channel image, values that could go NaN/Inf and
then propagate silently into the final score. Does each function return a defined,
in-range value or clamp, versus crashing or emitting NaN? Identify every place a
NaN/Inf could enter the headline number undetected (no `np.isnan` guard, no clip),
and every empty-collection access (`max([])`, `x[0]` on empty, division by a count
that can be zero).

## Task 12 — Performance & resource envelope (reason, do not run)
Estimate the per-request cost of the main endpoint so a user study will not hit
timeouts/OOM: which stages are heavy (saliency inference, OCR, steerable-pyramid,
per-element loops), and does any of them run more than once per request when it
could be cached/reused? Is model loading amortised (loaded once) or repeated per
call? Are there unbounded loops over detected elements or pixels that scale badly
with image size? Is the uploaded image size bounded before heavy processing? Flag
any obvious quadratic blow-up, repeated expensive recomputation, or unbounded
memory growth.

## Output format
1. Task 0 — inventory + two-way diff with (i)/(ii)/(iii) classification.
2. Deployed-output trace (Task 1) — the exact formula and its inputs.
3. Sections for Tasks 2–12, each with `file:line` evidence and short quotes.
4. A findings table: ID | Layer (bug/validity/honesty/quality) | Severity
   (blocks scientific validity / degrades result / cosmetic) | file:line |
   one-line description.
5. Top issues ranked by impact on scientific validity (not ease of fixing); for
   each, say what evidence would confirm a fix on the next audit.
6. Regression watchlist — invariants to re-check every audit (e.g. "learning
   model at default = novice baseline"; "feature vector passed to the load model
   in the consumer's expected order"; "no endpoint leaves shared parameters
   mutated"; "UI number == API field with correct scale"; "no metric returns NaN
   on a single-colour image").

Reward a subtle, verified issue over a long list of speculative ones. If a section
has nothing wrong, say so and state exactly what you checked. If I did not give
you a file needed for a task, say which file you need instead of guessing.
