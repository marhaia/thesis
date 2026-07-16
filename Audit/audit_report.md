# Technical & Methodological Audit
## Two-Stage Interactional Complexity Pipeline

**Repository purpose:** Computational estimation of interactional complexity and cognitive load from automotive GUI screenshots, accompanying the master's thesis *"Two-Stage Multi-Output Pipeline: Computational Estimation of Interactional Complexity from GUI Screenshots."*

**Audit method:** Every pipeline module was read in full; claims were checked against running code (the UMSI++ model was rebuilt and executed against local weights; the `visual_hierarchy` inversion was demonstrated with a synthetic-image test). The README was treated as one source among several, not as ground truth, and cross-checked against git history to separate documentation drift from current code.

---

## Task 0 — Pre-audit inventory and diff

### 0.1 Inventory (independent of README)

**Entry points / apps**

| Item | Function (from code) |
|---|---|
| `stage1/app.py` (1,234 lines) | Flask server, **7 routes**: `/`, `/api/analyze`, `/api/features`, `/api/saliency`, `/api/search-time`, `/api/scanpath-to-target`, `/api/cognitive-load`. Contains substantial scoring logic, not just plumbing. |
| `stage1/visual_complexity.py` (1,252 lines) | 8 visual-complexity metrics + CLI (`--image`, `--dir`). |
| `stage1/ui/index.html` (3,983 lines) | Entire frontend as one file, all client logic inline. |

**Core modules**

- `saliency/umsi_model.py` — TF2/Keras3 re-implementation of the UMSI++ architecture; loads `umsi++.hdf5` (present locally, 120,093,896 bytes, git-ignored).
- `saliency/saliency_features.py` — 5 hand-crafted statistics of the predicted heatmap (dispersion, peak count, center bias, entropy, coverage).
- `cognitive/jokinen_model.py` — Monte-Carlo visual-search simulator (partial Jokinen 2020 + EMMA) plus `compute_glance_metrics` (NHTSA/ISO glance packing).
- `cognitive/element_detector.py` — contour-based UI element detection + WCAG 2.1 contrast ratio per element.
- `cognitive/text_reader.py` — optional EasyOCR reading-time estimate (238 wpm), degrades to `None`.
- `hceye/hceye_features.py` — rule-based h∈ℝ⁶ "cognitive-load sensitivity" features from HCEye condition-effect coefficients; **this, not the regression model, produces the deployed load score**.
- `stage2/regression_model.py` — Ridge/RF/XGBoost multi-output model + `build_training_data`; **no trained model exists in the repo** (`stage2/models/` absent, and `*.pkl` is git-ignored, so one could never be committed).
- `stage2/task_descriptor.py`, `stage2/user_profile.py` — additive score modifiers (task: −8…+12 pts; Big-Five preset: −5…+6 pts). `estimate_search_efficiency` (`task_descriptor.py:141`) is **never called anywhere** — dead code.
- `stage2/coherence_check.py` — three post-hoc consistency rules that emit warnings only.
- `stage1/build_feature_norms.py` + `feature_norms.json` — z-score/percentile reference distribution over 1,485 UEyes GUI screenshots (dataset itself excluded).

**Scripts** — `download_weights.py` (GitHub release asset, size check only, no checksum), `ueyes_saliency_validation.py` (NSS/CC/SIM/KL vs. 5 sample images), `baseline_comparison.py`, `feature_ablation_shap.py`, `smoke_test_endpoint.py`, `hceye_analysis.py`, `hceye_condition_comparison.py`, `osf_explore.py`/`osf_query.py` (OSF data download), `generate_expose_docx.py`, `generate_pipeline_figure.py`.

**Tests** — `tests/test_pipeline.py` and `cognitive/test_jokinen.py` are demo scripts, not a test suite (see §1).

**Non-code** — `Literature/` (notes, exposés), `VAS/` (VAS-style report screenshots, referenced by no code), `hceye/gaze/` and `hceye/saliency_pred/` (vendored upstream HCEye code; the required `fixation_AOI_metrics_final.csv` is absent — `*.csv` is git-ignored), `_input/` (untracked, contains `Patent.md`).

### 0.2 Documentation-drift boundary

README last modified in commit `79228e2` (2026-06-17). **Substantive code commits after that date**: UI/input-flow redesign (`a0ce751`), task-driven scanpath visualization (`27319c0`), task-relative load modifier (`c92decf`), per-target uncertainty + UMSI design-type warning + display presets (`55a0e1c`), ranked search bottlenecks (`ea3d755`), size dissimilarity + acuity threshold in Jokinen (`b68d5b6`), glance-based distraction metrics (`ea83a6f`), SHAP/ablation analysis (`6bc7ce9`), WCAG contrast report (`f680441`), OCR reading cost (`2c3ec4f`). So roughly **half of the cognitive/analysis functionality postdates the README**.

### 0.3 Two-way diff

**(a) Claimed in README but not found / materially different**

1. `python3 -m pytest tests/` (README.md:160) — **does not work**: pytest is neither in requirements.txt nor in the venv (verified), and `test_full_pipeline(image_path)` takes a positional argument, which pytest would reject as a missing fixture.
2. Setup instructions are broken: `pip install -r requirements.txt` omits **`pyrtools`**, a hard import at `visual_complexity.py:42` — a fresh install fails on the first Stage-1 call. (It's installed in the committed-developer venv, which masked this.)
3. Weight size is stated as "~433 MB" (README.md:44) and "~115 MB" (README.md:86); the actual file is 114.5 MiB; `DOCUMENTATION.md` says 421 MB. The `umsi_model.py:32` docstring gives a wrong weights path (`saliency/weights/umsi++.hdf5`).
4. The API table omits `/api/scanpath-to-target` entirely, and describes `/api/cognitive-load` as a simple estimate when it is now the orchestrator of nearly everything.
5. "A cognitive-load estimate — combining the above" — as shown in §4, **predicted search time is never combined into the load estimate**.

**(b) In code but absent/vague in README, classified**

*(i) Methodologically load-bearing:*
- The **HCEye h∈ℝ⁶ block** — the actual source of the deployed cognitive-load score (`score = 100·h[5] + modifiers`, `app.py:1019`, `app.py:1032`). The README's one folder comment ("HCEye-based cognitive-load rules") does not convey that this *is* Stage 2 in practice.
- Task-descriptor and Big-Five **additive point modifiers** with hand-set coefficients.
- **Display presets** (`app.py:799`) — change pixel→degree conversion and therefore all search times.
- Post-README Jokinen changes: acuity gate, size dissimilarity, glance metrics.
- The **feature-norms reference distribution** (affects interpretation shown to users).
- The **coherence check** (post-hoc warnings, not the "coherence term" the exposé describes — see §5).

*(ii) Alternative/experimental paths:*
- The trained `Stage2Model` (opt-in via `use_trained_model`, defaults off; no model file can exist in-repo). `estimate_search_efficiency` (dead). Dead Jokinen parameters: `W_TA`, `a_color/b_color`, `a_shape/b_shape`, `B_bl`, `F_mem`, `f_mem`, `sigma_M`, `alpha`, `sigma_U`, `B_sa` — declared with citations, never used. `hceye/saliency_pred/` (vendored, unused). Analysis scripts (`baseline_comparison`, `feature_ablation_shap`, `ueyes_saliency_validation`).

*(iii) Incidental:* OSF downloaders, figure/docx generators, `create_test_screenshot.py`, `inspect_weights.py`, shell wrappers, VAS material.

---

## 1. Architecture and code quality

**Actual pipeline vs. described pipeline.** The "two-stage" story maps only loosely onto the code. The real orchestration lives inside one Flask route handler (`app.py:835-1218`): v (visual) → s (saliency stats) → h (HCEye rules) → `score = 100·h[5]` → additive task/profile modifiers → coherence warnings. The Jokinen search model — which `app.py:474` itself calls "the CENTRAL cognitive metric of the thesis" — is computed in that route **only to feed the coherence check and the "bottleneck" feedback list; it contributes nothing to the load score**. The per-target `search_load_modifier` (`app.py:749`) exists only in the `/api/scanpath-to-target` response and is likewise never folded into a stored score. So the composition claimed in README/exposé (complexity + saliency + search time → load) is not what executes.

**Modularity.** Module boundaries are otherwise sensible (stage1/saliency/cognitive/stage2 as importable packages), but inference logic leaks into the presentation layer: magic scoring constants (`POINTS_PER_UNIT=25`, `MODIFIER_CAP=15` at `app.py:749-750`, slopes 0.0025/0.0030/0.0040 at `app.py:1033-1042`) live in the route handler, which is why `scripts/baseline_comparison.py` had to re-implement its own scoring. The 3,983-line single-file UI is unmaintainable but scientifically harmless.

**Error handling.** Mixed. `/api/cognitive-load` logs saliency failures loudly and degrades explicitly (good, `app.py:954-962`). But `/api/search-time` swallows saliency errors with a bare `except Exception: pass` (`app.py:551-552`), and all endpoints catch broad `Exception` into a 500 JSON.

**Reproducibility.**
- Dependencies pinned — but the pin list is incomplete (`pyrtools` missing = fatal; `pytest` missing; `xgboost` missing but guarded).
- Python 3.9.6 documented and matches the venv; CPU-only Apple Silicon is documented; no CUDA claim.
- Jokinen model is seeded (`random_seed=42`, re-seeded per request) — deterministic API output. UMSI++ inference is deterministic.
- **Not reproducible from the repo**: `hceye/sensitivity_lookup.json` (its source CSV is git-ignored and absent), `feature_norms.json` (needs the excluded 12.9 GB UEyes dataset), the saliency validation (needs the excluded `ueyes/` repo), and any trained Stage-2 model (`*.pkl` git-ignored).
- "Tests": no assertions anywhere; both test scripts hardcode a different user's machine path (`/Users/Q682780/Thesis_G`, `cognitive/test_jokinen.py:6`, `saliency/test_full_pipeline.py:8`), i.e. they were never run from a clean checkout.

---

## 2. Construct validity of the 8 visual complexity metrics

| # | Metric | Verdict |
|---|---|---|
| 1 | Shannon entropy | **Canonical.** 256-bin grayscale histogram, log₂ (`visual_complexity.py:473-508`). Fine, though luminance-only (standard). |
| 2 | Edge density | **Plausible AIM m4 port.** Canny at 0.11/0.27×255 after σ=2 blur. Caveat: OpenCV Canny thresholds operate on raw gradient magnitude, MATLAB's on normalized gradients — whether AIM's own port matches cannot be checked because `aim/` is excluded. |
| 3 | Feature congestion | **Faithful Rosenholtz et al. (2007) port** — the constants match the published model (channel normalizers 0.2088/0.0660/0.0269, det^(1/6) color, contrast std, det^(1/4) orientation, DoG σ ratios 0.71/1.14). One transcription typo: the Z white point is divided by **108.833** at `visual_complexity.py:107` while the docstring (and CIE D65) says 108.883 — numerically negligible, but it's an error, possibly inherited from AIM. |
| 4 | Subband entropy | **Structurally matches** Rosenholtz SE (3 scales × 4 orientations steerable pyramid, chroma weight 1/16, zero-threshold 0.008, √N-bin entropy in nats). Whether the final `/(1+2w)` normalization (`visual_complexity.py:839`) matches AIM m7 exactly is unverifiable in-repo. |
| 5 | Layout symmetry | **Citation ≠ implementation.** Implemented as NCC of the zero-meaned grayscale image with its mirrored self (`visual_complexity.py:888-894`). Miniukovich & De Angeli (2015) compute symmetry from mirrored *edge/contour* structure, not raw pixel correlation. The code header honestly says "(custom)" but the docstring cites M&DA as if it were their metric. |
| 6 | Chromatic coherence | **Custom composite, name inverted.** Averages four sub-metrics (luma std, Hasler–Süsstrunk colorfulness [formula correct], circular hue std, saturation std) with arbitrary normalizers 128/150/2.5/0.5 (`visual_complexity.py:969-973`). Higher = *more fragmented*, i.e. the feature measures chromatic **in**coherence. Also, hue circular std includes hue values of near-achromatic pixels (pure noise) unweighted by saturation. No single cited source defines this composite. |
| 7 | Visual hierarchy | **Confirmed inverted — empirically demonstrated.** The figure-ground sub-metric weights edge drop-outs by `(1 − k/6)`, i.e. edges that vanish at the *lowest* threshold get the *highest* weight (`visual_complexity.py:1036-1038`). Test performed: black-on-white boxes (strong contrast) → fg = 0.0, VH = 0.33; barely-visible gray-on-gray boxes → fg = 1.0, VH = 0.83. This is the exact opposite of the docstring ("A UI where all edges vanish at low thresholds has weak figure-ground separation (fg ≈ 0)") and of the construct (Tuch et al. is cited but supplies no formula). The degenerate branch compounds it: when edges persist through *all* thresholds (maximal contrast), `denom=0` and fg is set to 0. Downstream, VH is consumed with a load-*reducing* sign in `baseline_comparison.py:106` and a 0.40 positive weight in the (dead) `estimate_search_efficiency` — so wherever it's used, the direction of the effect is wrong. |
| 8 | Interactive element density | **Honest heuristic.** Contour counting with size/aspect/fill filters; docstring correctly flags it as a proxy. Minor: the quantity called "solidity" (`visual_complexity.py:1128`) is actually *extent* (area/bbox), not solidity (area/convex hull). |

Cross-cutting: none of the metrics normalizes for input resolution, so entropy/edge-density/element-density are not comparable across screenshots of different sizes; the feature-norms percentile layer partially mitigates this for display, not for the score.

---

## 3. Saliency stage

**Loading and weights — verified live.** The ported model was built and the local checkpoint loaded: the positional (topological) load path succeeds, and prediction on `bmw_route.png` yields a non-degenerate heatmap (std ≈ 0.18); the checkpoint's 181 layer names match the port's naming exactly. The architecture port (`umsi_model.py`) is careful work.

Concerns:

1. **Provenance is unverifiable.** The weights are a re-upload on the author's own GitHub release; `download_weights.py` checks only "≥100 MB" — no checksum, no link to the specific UEyes artifact. There is no in-repo evidence the file is the UEyes-trained UMSI++ checkpoint beyond layer-name compatibility.
2. **Silent-fallback risk in loading.** On any positional-load failure, `umsi_model.py:517-520` retries with `skip_mismatch=True` — a model with silently skipped (random-initialized) layers would still serve predictions labeled as UMSI++. Currently the positional path works, so this is latent, but it should hard-fail instead.
3. **The 6-class design head is close to noise.** On the BMW screenshot the softmax was ≈ [0.156, 0.167, 0.165, 0.164, 0.245, 0.103] — near-uniform. The `DESIGN_CLASSES` label order (`umsi_model.py:486`) is asserted, not verifiable from the repo, and the "out_of_domain" warning (`app.py:938-948`) is built on this head. Note the deeper irony: *automotive* UI is not a UEyes category at all, so the entire saliency stage is out-of-domain in a way the warning cannot express (it treats `desktop_ui` as in-domain).
4. **Preprocessing** (BGR, VGG mean subtraction, aspect-preserving pad to 256×256, 512×512 output un-padding) matches the documented original utilities; a diff against the actual UEyes code was not possible because `ueyes/` is excluded from the repo.
5. **Validation is thin and contains a canned claim.** `ueyes_saliency_validation.py` runs on only 5 bundled sample images (in the excluded repo — not runnable here), correctly disclaims NSS on blurred ground truth, but then **prints a hardcoded conclusion "CC = 0.896 indicates very strong spatial correspondence … exceeds published baselines" regardless of the computed values** (line 278). If the numbers ever change, the script still prints this sentence.
6. The s∈ℝ⁵ features are *statistics of the heatmap*, not model outputs; their thresholds (peak ≥ 0.3·max, coverage > 0.5·max, 32-bin entropy) are unsourced conventions — acceptable as engineered features, but not "from UMSI++" in the strong sense the README implies.

Runtime fallbacks are mostly explicit (`saliency_used` flag, loud logs in `/api/cognitive-load`), except the silent `except: pass` in `/api/search-time` noted in §1.

---

## 4. Search-time and cognitive-load stages

### Jokinen (2020) implementation

What matches the cited sources: the EMMA timing layer is correct as cited — `T_enc = K·(−ln f)·e^{k·ε}` with K=0.006, k=0.4, and saccade time `0.135 + 0.002·ε + 0.07` s (`jokinen_model.py:560-577`), including the post-saccade residual-eccentricity encoding.

What diverges, in decreasing severity:

1. **There is no top-down guidance at all.** `W_TA = 0.45` is declared "Top-down (task relevance) activation weight" (`jokinen_model.py:98`) and never used. The simulation is winner-take-all over *bottom-up* activation + logistic noise + inhibition of return + an acuity gate. The mechanism that gives Jokinen et al. (2020) its name — *adaptive feature guidance*, activation of elements sharing features with the target — is absent. Consequently the model does not simulate "searching for X"; it simulates "how many bottom-up pops occur before X happens to win". Predicted search time is a conspicuity ranking, not guided search, and will systematically overestimate times for low-salience targets.
2. **Bottom-up activation deviates from its own docstring**: the documented formula is `Σ dissim / d_ij` (`jokinen_model.py:690`) but the code divides by `√d_ij` (`jokinen_model.py:746`); which one matches the paper's Eq. 2 is unstated. Dissimilarity uses only two feature dimensions (categorical color, relative size); the acuity gate uses only the size coefficients, leaving the color/shape threshold parameters dead.
3. **Author's own extensions are declared but uncalibrated**: `W_saliency=0.8`, `saliency_exponent=2.0`, `visibility_penalty=5.0` are hand-picked with no sensitivity analysis in the repo.
4. `tau_vstm=20` is attributed to "Jokinen 2020 Table 1" while the adjacent comment cites VSTM capacity ~4 items (Cowan) — internally inconsistent, unverifiable here.
5. `max_fixations=50` truncation: censored trials are averaged as if complete — biases hard-target times downward, unflagged.
6. Hardcoded vs. derived: derived from the image are element geometry/colors and per-element saliency; hardcoded are all timing constants, weights, `freq=0.1` for every element, center start position, 0.5° landing noise, and screen geometry (unless a display preset is passed).

The difficulty bands (<1 s "easy" … >5 s "very_hard", `jokinen_model.py:788`) and the glance packing (1.5 s budget vs. NHTSA 2 s/12 s limits) are design conventions layered on model output — the glance docstring is commendably honest about this.

### The cognitive-load combination — the weakest link

The deployed formula is: `score = 100 · h[5] + task_modifier + profile_modifier` (`app.py:1019`, `app.py:1032`), where h comes from `hceye_features.py`.

**Bug (confirmed): the v and s vectors are passed in the wrong semantic order.** `HCEyeFeatureExtractor` documents its expected input as `[edge_density, color_count, layout_complexity, whitespace_ratio, text_density, element_count, symmetry, grid_quality]` with ranges `[0.5, 1000, 1, 1, 0.5, 100, 1, 1]` (`hceye_features.py:104-110`, `239-249`). The app passes `[shannon_entropy, edge_density, feature_congestion, subband_entropy, layout_symmetry, chromatic_coherence, visual_hierarchy, interactive_element_density]` (`app.py:898-907`). Concrete consequences:

- Position 0 ("edge_density", /0.5): receives Shannon entropy (typ. 4–7) → **saturates to 1.0 for every image**.
- Position 2 ("layout_complexity", /1): receives feature congestion (typ. 1–10) → saturates to 1.0.
- Position 3 ("whitespace_ratio"): receives subband entropy (typ. 1–5) → saturates to 1.0, so "simplicity" ≡ 1 and "clutter" = 1 − 1 ≡ 0 — the AOI-sensitivity term becomes a constant.
- Position 5 ("element_count", /100): receives chromatic coherence (0–1) → ≈ 0.
- The saliency block is misread the same way: `s[2]` is consumed as "spread" but carries *coverage*; `s[4]` is consumed as "coverage" but carries *center bias* (`hceye_features.py:196-199` vs. `app.py:919-925`).

Because most terms saturate or vanish, **h — and therefore the headline score — is nearly constant across images**, modulated mainly by artifacts rather than the measured metrics. The lookup-table branch that would use real per-image HCEye values is also unreachable: `image_name` is never passed from the app.

**Even with correct wiring, the rule is conceptually shaky.** The HCEye coefficients describe *how experimentally induced load changes gaze* (group-level condition ratios); the code relabels a "vulnerability to external load" index as "cognitive load caused by this UI" — a category shift with no in-repo justification. The blend weights (0.30/0.20/0.20/0.15/0.15, then `/0.3` renormalization, `hceye_features.py:212-220`) and the derived outputs (`search_efficiency = 1 − h[3]`, `attention_demand = h[5] + 0.15`, `app.py:1020-1021`) are unsourced.

**Task description / user profile**: they are genuinely used, but only as additive point offsets on the score (`task_descriptor.py:99-116`, `user_profile.py:103-110`), with coefficients justified ordinally by citation but calibrated by hand ("~+12 points max"). The t and p vectors are appended to the returned `full_feature_vector` but consumed by nothing.

**The trained Stage-2 path is circular.** `build_training_data` (`regression_model.py:318-428`) (a) *fabricates* v and s from gaze statistics rather than images, and (b) **leaks the targets into the features**: `x[18]` is `sens['cognitive_load_index']` while `y₁ = sens['cognitive_load_index'] × 100`; y₂ and y₃ are affine transforms of `x[13]` and `x[14]`. Any R² from this setup is near-tautological. To its credit, `feature_ablation_shap.py:25-33` explicitly flags the h-dominance-by-construction caveat — though it stops short of naming the exact identity leakage. The path is off by default and unshippable (no `.pkl`, git-ignored), which limits the practical damage.

---

## 5. Correspondence with the thesis

Judged against the exposés in `Literature/notes/expose/` (the thesis text itself is not in the repo):

1. **"Two-Stage Multi-Output" pipeline**: the standard exposé honestly states the current implementation is "training-free … HCEye-calibrated empirical coefficients", which matches the code. But the short exposé claims Stage 2 produces "(1) Saliency Map, (2) Fixation distribution, (3) Cognitive Load Index" with heads **"coupled by a coherence term that penalizes physically inconsistent combinations"**. The implementation has no fixation-distribution output, the saliency map comes from Stage 1's UMSI++, and the "coherence term" is a post-hoc rule checker emitting warnings (`coherence_check.py`) — not a coupling term in any model. This is a substantive claim/implementation gap.
2. **"Integrating … into the AIM platform"** (working title): nothing is integrated into AIM. The repo ports selected AIM metrics standalone; `aim/` is excluded; there is no integration artifact.
3. **Search time as the central cognitive metric**: asserted in code comments and exposé figure narrative, but as established, it never enters the load estimate — the thesis architecture diagram and the executing data flow disagree.
4. **Automotive focus**: the saliency model's training domain (UEyes: poster/mobile/desktop/web) does not include automotive UI, and the reference-norms baseline is web/mobile/desktop. Post-README code (`55a0e1c`) added the out-of-domain warning, which *partially supersedes* the exposé's silence on this — but as shown in §3 the warning stands on a near-uniform classifier head.
5. **"Interactional complexity"** (title term): no construct by that name is computed anywhere; the closest referents are the 19-dim feature vector and the 0–100 score.
6. Undocumented functionality that *supports* the thesis exists too: glance metrics, WCAG contrast, OCR reading cost, and per-target uncertainty all postdate the README and strengthen the automotive story — the documentation simply lags them.

---

## 6. Critique

### Empirical / implementation bugs (verifiable in code)

1. HCEye feature-order/semantics mismatch (v and s) — §4; the deployed score is computed from saturated, misassigned inputs.
2. `visual_hierarchy` figure-ground sub-metric inverted — demonstrated numerically in §2; one of the 8 headline metrics points the wrong way, with wrong-signed downstream uses.
3. Stage-2 training-data target leakage + fabricated features — §4 (default-off mitigates).
4. Reproducibility breaks: `pyrtools` missing from requirements; pytest claim false; test scripts hardcode a foreign machine path; sensitivity lookup, feature norms, and validation unregenerable from the repo.
5. Hardcoded "CC = 0.896 … exceeds published baselines" printed unconditionally in the validation script.
6. Silent-fallback hazards: `skip_mismatch=True` weight-load fallback; `except: pass` on saliency in `/api/search-time`.
7. Minor: Z white-point typo (108.833 vs 108.883); docstring/code mismatch `d_ij` vs `√d_ij`; large blocks of dead cited parameters; the lookup path unreachable.

### Conceptual / methodological (independent of correct implementation)

- Reinterpreting HCEye *condition-effect ratios* as a per-UI *load* score is a construct-level leap with no validation target anywhere in the repo; the coherence check tests internal consistency, never validity.
- A bottom-up-only search simulator cannot ground claims about goal-directed search time; adding UMSI++ saliency into Jokinen's activation is a plausible idea, but its weights are uncalibrated and no comparison against the paper's reported human data exists in-repo.
- Additive point modifiers for task and personality assert quantitative effects (±12, ±6 points) from ordinal literature — the docstrings admit this, but the UI presents a single blended number.
- Several Stage-1 metrics are custom composites presented under literature citations (symmetry, chromatic coherence, hierarchy); "chromatic coherence" measures its own opposite.
- Resolution sensitivity of the pixel metrics vs. a fixed-256×256 saliency model makes the combined vector scale-inconsistent.

### Open questions not resolvable from the code alone

- Whether the hosted `umsi++.hdf5` is byte-identical to the UEyes-published checkpoint (no hash), and whether the 6-class label order is correct — the near-uniform classification output makes both worth checking against the upstream repo.
- Whether AIM's m4/m7 implementations match these ports exactly (`aim/` excluded).
- Whether Jokinen 2020 Table 1 actually specifies `tau_vstm=20`, the `√d_ij` denominator, and the acuity coefficients as used.
- What the `VAS/` folder (VAS-style reports) is for — no code references it; presumably a planned convergent-validity comparison.
- Whether the final thesis text repeats the short exposé's "coherence-term-coupled multi-head" description, which the code contradicts.
- Purpose of untracked `_input/Patent.md`.

---

## Top 3 priority issues (ranked by impact on scientific validity)

1. **HCEye feature misalignment** (`app.py:898-925` → `hceye_features.py:144-207`) — *scientific validity, primary output.* The headline cognitive-load score is currently a function of saturation artifacts, nearly invariant to the actual measured features. Every number this pipeline has ever reported from `/api/cognitive-load` is affected. Fix the mapping (or make the extractor consume named features) and re-run everything.

2. **Inverted `visual_hierarchy` figure-ground metric** (`visual_complexity.py:1036-1040`) — *scientific validity, Stage-1 vector.* One of the eight thesis-level metrics rewards weak contrast; the fix is a one-line weight change (`k/6` instead of `1−k/6`, plus the `denom=0` branch), but published feature values, the feature-norms JSON, and any analysis using v must be regenerated.

3. **Stage-2 training circularity** (`regression_model.py:367-415`) — *scientific validity of any "trained model" claim.* Off by default and partially self-acknowledged, so it ranks below the two live bugs — but no R², ablation, or SHAP result from this data may appear in the thesis as evidence about the pipeline.

Below these three sit the reproducibility/documentation items (missing `pyrtools`, false pytest claim, stale README, canned validation output), which are about code quality and staleness rather than the validity of the outputs.
