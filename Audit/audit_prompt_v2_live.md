# Live In-Workspace Audit Prompt (v2, companion)

> Companion to `audit_prompt_v2.md`. Use THIS variant when the auditor is an
> assistant working **inside the open workspace with tool access** (can read
> files, grep, run the venv, hit the Flask app), rather than an external model
> reasoning over a static clone.
>
> It reuses the same six audit dimensions and the same discovery-first,
> evidence-bound, read-only philosophy as the external prompt — the only
> difference is that here you are expected to **actively run things to verify**,
> not just reason about the code.

---

## Role and ground rules (same spirit as the external audit)

You are auditing this repository from inside the workspace. You have tools to
read files, search, and execute code. Apply the same rules:

1. **Verify by running, not by trusting.** Where the external auditor can only
   reason, you must actually execute: import every edited module, call the
   endpoints, feed synthetic inputs, and compare the observed numbers to the
   claim. The runtime result outranks any docstring or comment.
2. **Cite evidence.** Every finding gets `path/file.py:line` + a quote, plus the
   command/output that demonstrates it. No citation → label `UNVERIFIED`.
3. **Separate layers:** implementation bug / scientific-validity issue / honesty
   (overclaim) / code-quality. Keep them apart in the report.
4. **Read-only + safe.** You may create a scratch venv and run code, but do not
   edit tracked files, commit, or push as part of the audit. If you find a fix,
   propose it separately — do not fold it into the audit run.
5. **Guard against this repo's known failure mode.** This codebase has a history
   of files being reverted/emptied **out-of-band** between sessions. So, as part
   of every audit: run `git status` and `git diff --stat HEAD`, confirm the
   working tree matches `origin/main` (or explain every difference), and verify
   no tracked file has silently lost content. This check is mandatory here.

---

## Environment notes specific to this workspace (verify, then use)

These are starting hints, not gospel — confirm each before relying on it:
- There are two virtualenvs. Only **`venv/`** has the heavy stack (OpenCV,
  TensorFlow, Torch, etc.); **`.venv/`** is the language-server env and lacks
  them, which is why Pylance shows benign "cv2/flask/PIL could not be resolved"
  warnings that are **not** real errors. Always run image/model code with
  `venv/bin/python`.
- The Flask app is under `stage1/`, serves on port 5001, and its upload field is
  `image` for single-image endpoints. Multi-screen endpoints also accept several
  files under `images` or one animated GIF under `image`.
- macOS: there is no `timeout` command; do not rely on it in test commands.
- Confirm the real deployed load score's formula from the code before trusting
  any summary of it.

Do the six audit tasks below. Where the external prompt says "check", you should
"check **by executing**".

---

## Task 0 — Live inventory + integrity

- List every Flask route (enumerate from the app's URL map, not from the README)
  with method, expected input, and returned keys.
- List the core modules and run `import` on each with `venv/bin/python` to prove
  they load with no error.
- **Integrity gate (mandatory):** `git fetch`, then confirm `HEAD == origin/main`
  and the working tree is clean, OR enumerate and justify every deviation. Run
  `git diff --stat HEAD` and confirm no tracked file shows an unexpected large
  deletion (the out-of-band-loss check).

## Task 1 — Trace and *reproduce* the deployed output

Call the main endpoint on a real screenshot from the repo and capture the
headline number. Then, from the code, write the exact formula that produced it
and confirm each input's source. Flag any advertised input that is computed but
does not enter the final number (show the line where the trail ends).

## Task 2 — Metric construct validity (spot-check by execution)

For each visual-complexity metric and each per-element feature, quote the
implemented formula and compare to its cited source. Where feasible, feed a
**controlled synthetic input** and confirm the metric responds in the claimed
direction and range (e.g. a high-contrast vs. low-contrast patch; a symmetric
vs. asymmetric layout). Flag sign inversions, degenerate branches, and
scale/resolution dependence.

## Task 3 — Saliency integrity (run the loader)

Actually load the saliency model and predict on one image. Confirm the load path
does not silently fall back to skipped/random layers, and that a missing-weights
situation fails loudly rather than serving a blank map labelled as the model.
Note the training-domain vs. application-domain gap and whether it is surfaced.

## Task 4 — Search-time and load model (run + inspect)

Run the search-time and cognitive-load endpoints. Verify the feature vectors are
consumed in the **order the consumer expects** (execute a check that a permuted
input changes the output as expected). List implemented vs. dead model
parameters. For any trained/regression path, check for target leakage and
fabricated features and state that any metric derived from it is tautological.

## Task 5 — Newer features (discover, then run the invariants)

Discover the extension features from Task 0 (e.g. learning curve, inter-screen
consistency, product-level time×space coupling, contrast gate, glance metrics,
readability, display presets, uncertainty). For each, **run** these checks:
- **Default = baseline, exactly.** An optional feature at its off/neutral default
  must reproduce the prior baseline byte-identically (verify numerically; e.g.
  learning model at 1 exposure == novice search time to full precision).
- **Direction/monotonicity.** Build a controlled synthetic case and confirm the
  metric moves the claimed way (consistent screen set scores higher than a
  shuffled one; lower contrast → longer search; more exposures → shorter time
  with diminishing returns).
- **Input contract.** Confirm multi-screen readers accept both several files and
  a GIF, and reject <2 screens / malformed input with a clean 4xx.
- **Calibration honesty.** Confirm every uncalibrated constant is declared as
  such in code/docs, not presented as validated.

## Task 6 — Reproducibility (prove it)

Confirm every hard import is pinned (grep imports vs. requirements). Call one
endpoint twice and confirm identical output (seed determinism). Verify the
documented run/test commands actually work. List artefacts not regenerable from
the repo and any machine-specific hardcoded paths.

## Task 7 — Robustness / edge cases (attack it)

Spin up the app and fire degenerate inputs: empty/1×1/huge image, non-image
bytes with an image extension, zero detected elements, single-frame GIF, screen
set of length 0/1, missing/misspelled fields, non-numeric query params. Report
the status code and behaviour for each. Grep for bare/broad `except` that could
hide a broken stage. Check that any endpoint which temporarily mutates shared
model state restores it in a `finally` (run a before/after check). Note
`debug=True`, attacker-controlled filenames, and any `eval`/`exec`/shell use.

## Task 8 — Thesis correspondence + honesty

For each specific thesis claim, state substantiated / simplified / contradicted,
with evidence. Flag any single confident number in the UI or write-up that rests
on uncalibrated coefficients, an out-of-domain model, or an unvalidated
heuristic without a stated caveat.

---

## Output format

Same as the external prompt: Task 0 inventory + integrity, deployed-output
trace, Tasks 2–8 with `file:line` evidence **and the command/output that proves
each finding**, a findings table (ID | layer | severity | file:line | one-line),
a ranked top-issues list by impact on scientific validity, and a
**regression watchlist** of invariants to re-run next time. Because you executed
the checks, include the actual observed numbers so the next audit can compare.
