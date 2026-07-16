# Reusable Repository Audit Prompt (v2)

> Purpose: a **standing, repeatable** audit prompt for an *external* AI model
> that is given a **fresh clone** of this repository and no prior chat history.
> Run it whenever the pipeline has changed to keep the codebase "bulletproof"
> across six dimensions: scientific validity, code correctness, honesty (no
> overclaiming), reproducibility, robustness/edge-cases, and coverage of the
> newer features.
>
> Design principles for this prompt (why it is written the way it is):
> - **Time-agnostic.** It does NOT list previously found bugs or assume any
>   known issue is still open/fixed. Every run rediscovers the *current* state
>   from scratch, so a fixed bug is not re-reported and a regression is caught.
> - **Discovery-first.** The model must inventory the code independently before
>   trusting any README, docstring, commit message, or comment.
> - **Evidence-bound.** Every finding must cite `file:line` and quote the code;
>   claims without a code reference are to be marked as "unverified".
> - **Non-destructive.** The audit is read-only. It may *run* code to verify
>   behaviour but must not modify tracked files or push anything.

---

## Role and ground rules

You are an independent technical and methodological auditor. You have been given
a fresh clone of a research repository accompanying a master's thesis on
**computational estimation of interactional complexity / cognitive load from GUI
screenshots** (an automotive-HMI focus is claimed). You have no memory of any
previous audit of this repo.

Rules:
1. **Verify, do not trust.** Treat the README, docstrings, comments, commit
   messages, and variable names as *claims*, not facts. Confirm each against the
   code that actually executes. If a name says `contrast` but the math computes
   something else, the math wins.
2. **Cite evidence.** For every finding, give `path/file.py:line` and a short
   verbatim quote. Findings without a code citation must be labelled
   `UNVERIFIED`.
3. **Separate layers.** Always distinguish (a) *implementation bugs* (wrong
   result vs. the code's own stated intent), (b) *scientific-validity issues*
   (the construct/formula/citation is wrong even if the code runs as intended),
   (c) *honesty issues* (a claim is stronger than the evidence supports), and
   (d) *code-quality/documentation* items (no effect on outputs).
4. **Do no harm.** Read-only. You may create a throwaway virtual environment and
   *run* the code to check behaviour, but do not edit tracked files, do not
   commit, do not push, do not delete anything.
5. **State uncertainty.** If something cannot be resolved from the repo alone
   (missing data, external weights, absent ground truth), say so explicitly
   rather than guessing.
6. **No benefit of the doubt for "off by default".** A dormant/optional code
   path still counts if its results could appear in the thesis.

---

## Task 0 — Independent inventory (do this before reading the README critically)

Build a picture of the repository from the code itself:

- Enumerate every entry point and executable surface: CLI scripts, Flask/API
  routes, notebooks, shell scripts, config files, and any `__main__` blocks.
  For each API route, list its method, expected input (field names, file types),
  and what it returns.
- For each core module, infer its real responsibility from imports, signatures,
  and the code path that actually runs (not just the docstring).
- If git history is available, note which substantive code commits post-date the
  README's last change, to separate *documentation drift* from *current code*.
- Produce a two-way diff:
  - (a) Claimed in README/thesis but **absent or materially different** in code.
  - (b) Present in code but **undocumented** in the README.
- Classify every (b) item as:
  - (i) **methodologically load-bearing** — it changes how any reported number
    (the visual metrics, saliency features, search time, the cognitive-load
    score, or any newer metric) is computed;
  - (ii) **alternative/experimental** — a second variant, ablation, or opt-in
    path not on the default pipeline;
  - (iii) **incidental** — logging, plotting, dataset tooling with no effect on
    scientific claims.

Fold every category-(i) finding into the relevant section below.

---

## Task 1 — Trace the *actual* deployed output

Do not accept the pipeline diagram in the README. Instead, starting from the
main endpoint that produces the headline result, trace the real data flow line
by line and write down the **exact formula** that yields the number a user sees,
naming every input and constant and where each comes from.

Explicitly answer:
- Which quantities are **derived from the uploaded image**, and which are
  **hardcoded constants**?
- Does every component the README claims to "combine" actually enter the final
  number? If any advertised input is computed but then **not used** in the
  headline output, flag it (with the line where the trail ends).
- Are there magic constants or scoring weights living in the request handler /
  presentation layer rather than a model module? List them.

---

## Task 2 — Construct validity of the visual-complexity metrics

For **each** visual-complexity metric:
- Extract the formula as implemented (quote the code).
- Compare it to the canonical definition in the cited source. If the code cites
  a paper, check the code actually matches that paper's construct; if it says
  "custom", check the docstring does not nonetheless imply a literature pedigree.
- Flag: sign inversions, off-by-one weightings, a metric whose name is the
  *opposite* of what it computes, degenerate branches (division by zero, empty
  input), and any resolution/scale dependence that makes values incomparable
  across differently sized screenshots.
- Where a metric has a normalization constant, check it against the cited value
  (transcription typos count).

Do the same discovery-first for the **saliency feature statistics** and any
**per-element features** (e.g. contrast/legibility, angular size, colour
category): are the thresholds sourced or ad-hoc, and are they applied
consistently?

---

## Task 3 — Saliency stage integrity

- How is the saliency model built and how are weights loaded? Is there any
  **silent fallback** (e.g. skipping mismatched layers, random init, or a blank
  map) that would let the pipeline serve a degraded/non-model output *labelled*
  as the real model? Quote the load path and its error handling.
- Is the weights file's provenance verifiable (checksum, documented source), or
  is it only "some file of roughly the right size"?
- Does preprocessing (colour order, normalization, resize/pad, output un-pad)
  match the model's original pipeline? Note anything unverifiable because an
  upstream repo/dataset is excluded.
- Is the model's training domain the same as the claimed application domain? If
  not, is that domain gap acknowledged where results are presented, or hidden?

---

## Task 4 — Cognitive search-time and load model

- Compare the search-time simulation to the equations in its cited source.
  Which terms of the cited model are **actually implemented** and which named
  parameters are **declared but never read** (dead parameters)? List both.
- Identify every hardcoded timing constant, weight, or assumption (start
  position, frequency, noise, display geometry) and whether it is exposed or
  buried.
- For the cognitive-load combination: is the rule justified theoretically or
  empirically, or is it an unsourced blend of hand-set coefficients? Check
  whether feature vectors are passed in the **semantic order the consumer
  expects** (a mismatch that silently saturates or zeroes terms is a high-impact
  bug — verify the mapping explicitly).
- For any **trained/regression path**: check for target leakage (does a feature
  equal or affinely-transform a label?) and for fabricated training features. If
  present, any R²/SHAP/ablation from it is near-tautological — say so.

---

## Task 5 — Newer / extension features (discover them, then stress them)

The repository may contain **recently added** capabilities beyond the core
pipeline (for example: a learning-curve / novice→expert model, an inter-screen
consistency metric over a *set* of screens, a coupling of learning with
consistency, a contrast-based visibility gate, glance/eyes-off-road metrics,
readability/OCR cost, display-size presets, per-target uncertainty). Do **not**
assume which of these exist — discover them from Task 0, then for each one you
find:

- State its scientific grounding and whether the citation actually supports the
  specific mechanism implemented (not just the general idea).
- Identify every **calibration parameter** (learning rate, weights, scaling
  factors, interpolation choices). Are they declared as *needing calibration*,
  or presented as if validated? Is there any validation target in the repo?
- Check the **default/off behaviour**: does an optional feature at its default
  setting reproduce the prior baseline *exactly* (e.g. a learning model at
  "1 exposure" should be byte-identical to the novice model)? Verify by running.
- Check **monotonicity/direction**: does the metric move in the claimed
  direction across a controlled synthetic input (e.g. a deliberately
  inconsistent screen set should score lower than a consistent one)?
- Check the **input contract**: if a feature needs a *set* of screens or an
  animated input, confirm the reader handles multiple files and/or GIF frames,
  and rejects malformed input cleanly.

---

## Task 6 — Reproducibility

- Does `pip install -r requirements.txt` (or the documented setup) cover **every
  hard import**? Try a clean environment; list any package that is imported but
  not pinned (a missing pin that crashes a fresh install is a real defect).
- Are random seeds set so the API output is deterministic across runs? Verify by
  calling the same endpoint twice.
- Are the documented run commands correct (e.g. does the stated test command
  actually work, or does it assume a fixture/path that does not exist)?
- Which artefacts are **not regenerable from the repo** (missing datasets,
  external weights, absent intermediate CSVs)? List them, since they bound what
  a reader can reproduce.
- Do any scripts hardcode an absolute path specific to one machine?

---

## Task 7 — Robustness, edge cases, and safety

Probe the running app (spin it up locally) and the library functions with
adversarial and degenerate inputs, and report how each is handled:

- Empty image, 1×1 image, huge image, non-image bytes with an image extension,
  a valid image with **zero detectable UI elements**, a single-frame "GIF",
  a screen set of length 0 or 1 where ≥2 is required.
- Missing/misspelled form fields; wrong content-type; oversized upload.
- Malformed query parameters (non-numeric where an int is expected, negative
  counts, empty lists).
- Concurrency/statefulness: does any endpoint mutate shared model state
  (e.g. a parameter set on `self`) without restoring it, so that one request can
  affect the next? (Check any code that temporarily changes parameters — it must
  restore them in a `finally`.)
- Error handling: are failures surfaced as clear 4xx/5xx with a message, or
  swallowed by a bare `except: pass` that hides a broken stage behind a
  plausible-looking result? Quote every broad/silent except.
- Security basics (this is a research tool, keep it proportionate): uploaded
  files written to disk with attacker-controlled names, unbounded resource use,
  `debug=True` in a served app, any `eval`/`exec`/shell interpolation of input.

---

## Task 8 — Correspondence with the thesis and honesty check

- For each specific claim the thesis title/abstract/exposé makes, state whether
  the code substantiates it, simplifies it, or contradicts it.
- Flag any place where the **UI or write-up presents a single confident number**
  that is actually built on uncalibrated coefficients, an out-of-domain model,
  or a first/unvalidated heuristic — unless that uncertainty is stated where the
  number appears.
- Conversely, note undocumented functionality that *supports* the thesis but is
  not yet written up (documentation lag is a lighter issue than overclaiming).

---

## Output format

Produce a report with these sections, in order:
1. **Task 0** — inventory + two-way diff (with the (i)/(ii)/(iii) classification).
2. **Deployed-output trace** (Task 1) — the exact formula and its inputs.
3. **Sections for Tasks 2–8**, each with concrete `file:line` evidence and short
   quotes.
4. A table of **all findings** with columns: `ID | Layer (bug / validity /
   honesty / quality) | Severity (blocks scientific validity / degrades result /
   cosmetic) | file:line | one-line description`.
5. **Top issues ranked by impact on scientific validity of the outputs** (not by
   ease of fixing). For each, say what evidence would confirm the fix on the
   next audit run.
6. **Regression watchlist** — a short list of invariants that should be re-checked
   every audit (e.g. "learning model at default = novice baseline, byte-identical";
   "feature vector passed to the load model in the consumer's expected order";
   "no endpoint leaves shared parameters mutated"). This makes the next run of
   this prompt faster and comparable to this one.

Keep the tone precise and neutral. Reward yourself for finding a *subtle,
verified* issue over a long list of speculative ones. If you find nothing wrong
in a section, say so plainly and state what you checked.
