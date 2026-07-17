# Independent Technical and Methodological Audit

Repository: marhaia/thesis  
Audited commit: f8a5e186672185b5f051ef8122bbbf7c78fd8d94  
Commit date: 2026-07-16 16:38:34 +02:00  
Audit date: 2026-07-17  
Auditor posture: implementation, scientific validity, claim honesty, reproducibility, robustness, and performance

## Executive verdict

The repository is a substantial and unusually well-commented research prototype. It has real value as an exploratory screenshot-analysis workbench: it computes eight visual descriptors, can run a saliency model, detects geometric elements, simulates a bottom-up novice search process, and presents several explicitly experimental extensions.

It is not presently scientifically valid as a screenshot-to-cognitive-load estimator, nor as an automotive-HMI screening instrument. The 0–100 headline is a hand-built vulnerability index whose image-to-behaviour mapping has not been fitted or validated against independent cognitive-load outcomes. Several user-facing and exposé claims are materially stronger than the implementation and evidence. Two implementation defects also compromise core inputs:

1. The RGB-to-CIELab conversion divides XYZ values in the 0–1 scale by white points expressed in the 0–100 scale. A white pixel therefore produces L* ≈ 8.99 instead of 100. This corrupts feature congestion, subband entropy, and the committed reference norms that normalize them.
2. The local UMSI++ decoder does not match the official UEyes decoder: the local decoder omits ReLU activations and dilation and disables biases. The exact pretrained checkpoint is absent, so strict load and output parity could not be demonstrated.

The most consequential methodological problem is construct reversal. HCEye measured how experimentally imposed cognitive load changed gaze. This repository takes screenshot properties, heuristically predicts those gaze-change ratios, and then relabels their weighted combination as the load caused by the screenshot. That inverse mapping is not identified by the cited experiment and has no independent calibration.

Recommended decision: do not use the current score as a dependent variable, ground truth, validated screening result, or safety/compliance decision. Do not begin a confirmatory automotive user study with this implementation as the frozen predictor. It is defensible to continue development if the output is renamed to an exploratory interaction-complexity heuristic, the two core implementation defects are fixed, the full pipeline is locked and tested, and validation is explicitly prospective and independent.

## Scope, evidence, and uncertainty

The audit covered all 200 tracked files, the public GitHub repository state, all Flask routes, all Python and shell entry points, the visible UI, the exposé generator, the user-study proposal, model and analysis code, requirements, committed reference norms, and legacy experimental code.

At the audited GitHub state:

- Default branch: main.
- Open pull requests: none.
- Open issues: none.
- GitHub status/check contexts on the audited commit: none.
- GitHub Actions workflow runs at the audited commit: none.
- Local worktree after inspection: clean.

Checks performed:

- Python byte-compilation completed successfully for the repository.
- Pure NumPy probes confirmed:
  - white maps to L* = 8.991442404 in the local conversion;
  - default task modifier = +1.10000038 points;
  - “neutral” profile modifier = −0.5 points;
  - a screen pair with one unchanged matched element and five replacements receives headline consistency 1.0 while occupancy consistency is 0.5;
  - two empty element sets receive consistency 1.0.
- A dependency-resolution check failed because requirements.txt pins numpy==1.26.4 while opencv-python==4.13.0.92 requires NumPy 2 or newer.
- Full pytest collection was not available in the base audit environment. Independently, the repository’s requirements cannot currently resolve, and the nominal test file is a demo with a required positional argument rather than an assertion suite.

Unverified because the required artifact or data is absent:

- Whether the shipped/released UMSI++ HDF5 checkpoint can load into the local architecture.
- Whether local UMSI++ predictions numerically match the official implementation.
- Any reported end-to-end predictive accuracy against independent cognitive-load, search-time, scanpath, or automotive data.
- Reproduction of HCEye condition analyses, because the raw HCEye CSV is absent.
- Reproduction of the UEyes validation figures, because the UEyes image/ground-truth folders and exact checkpoint are absent.

External source comparison was limited to primary sources, including the official UEyes repository, the HCEye paper, the Jokinen paper record, WCAG guidance, and NHTSA guidance.

# Task 0 — Independent inventory

## 0.1 Real module responsibilities

| Module | Responsibility established from executable code |
|---|---|
| stage1/app.py | Flask orchestration, upload handling, in-memory caching, visual and saliency execution, element/OCR extraction, task/profile adjustment, coherence checks, search feedback, contrast/readability reports, screen-set endpoints, and UI serving. |
| stage1/visual_complexity.py | Eight screenshot descriptors: Shannon entropy, Canny edge density, Rosenholtz-inspired feature congestion and subband entropy, custom pixel symmetry, custom palette fragmentation, custom hierarchy, and contour-based element density. |
| saliency/umsi_model.py | Local TensorFlow/Keras reconstruction of a UMSI++-like encoder/ASPP/classifier/decoder; strict HDF5 weight loading; resize/pad preprocessing and output unpadding. |
| saliency/saliency_features.py | Five hand-authored summaries of a saliency map: weighted spatial dispersion, local-maximum count, central mass, intensity-histogram entropy, and thresholded coverage. |
| cognitive/element_detector.py | Contour-based geometric element boxes plus dominant colour category, approximate internal contrast, and angular size. It is not a semantic UI-control detector. |
| cognitive/jokinen_model.py | A bottom-up stochastic novice search heuristic with EMMA timing, acuity and contrast penalties, saliency fusion, inhibition of return, optional target memory boost, product learning, and synthetic glance partitioning. It is not the full Adaptive Feature Guidance model. |
| cognitive/text_reader.py | Optional EasyOCR-based text detection/readability diagnostics and a text-element share used by the headline heuristic. |
| hceye/hceye_features.py | Percentile-normalizes selected screenshot proxies, transforms them through hand-set rules around HCEye aggregate ratios, and produces the default six-value “HCEye” vector and CLI. |
| stage2/task_descriptor.py | Maps categorical task selections to ordinal numbers and an additive score modifier. |
| stage2/user_profile.py | Maps coarse Big Five presets to an additive score modifier. |
| stage2/regression_model.py | Optional multi-output regression scaffold. No trained model is shipped; its current data builder is explicitly circular and fabricated. |
| stage2/coherence_check.py | Three threshold rules that compare saliency spread, search output, and the headline score. It flags logical disagreements; it does not calibrate or validate predictions. |
| stage2/screen_consistency.py | Geometric occupancy and greedy element-matching diagnostics over ordered screens. |
| scripts/*.py | Dataset inspection, descriptive analyses, invalid/scaffold baselines and ablations, UEyes/HCEye checks, download helpers, document/figure generation, and endpoint smoke testing. |
| hceye/gaze/data_processing_pipeline.py | Legacy gaze preprocessing, event marking, smoothing, fixation aggregation, and plotting. Several algorithms are defective and the external helper is absent. |
| hceye/saliency_pred/* | Legacy PyTorch saliency training/testing code. It is incomplete in this repository and is not used by the Flask application. |

Evidence: stage1/app.py:959–965 says “Combines visual complexity ... + saliency ... + HCEye cognitive load sensitivity”; cognitive/jokinen_model.py:56–67 explicitly says the live scope is “bottom-up, novice-mode” and that there is “no top-down (goal-directed) guidance yet”; hceye/hceye_features.py:201–205 calls the mapping a “purpose-built, literature-guided heuristic” whose “exact weighting is not empirically calibrated.”

## 0.2 Flask/API entry points

| Route | Method and accepted input | Returned keys / behavior |
|---|---|---|
| / | GET | Serves stage1/ui/index.html with no-cache headers. Evidence: stage1/app.py:377–384, “send_from_directory('ui', 'index.html')”. |
| /api/analyze | POST multipart image; PNG/JPG/JPEG/BMP/TIFF | Eight visual keys plus filename and visual_cache_hit; 400 for missing/empty/unsupported extension; other exceptions become 500 with error text. Evidence: stage1/app.py:387–416, “return jsonify(results)” and “return jsonify({'error': str(e)}), 500”. |
| /api/features | GET | Array of eight feature metadata records. Evidence: stage1/app.py:423–484, “Return metadata about the 8 features.” |
| /api/saliency | POST multipart image; same static formats | filename, five features, six-class probability dictionary, predicted_class, heatmap_png_base64, saliency_cache_hit; strict saliency failure becomes 500. Evidence: stage1/app.py:487–540. |
| /api/search-time | POST multipart image; query n_simulations and use_saliency | Search result with per_element, aggregate search times, difficulty, filename, model_info, and saliency_used. Zero detected elements returns HTTP 200 with error and n_elements=0. Saliency failure falls back to feature-only. Evidence: stage1/app.py:546–651. |
| /api/scanpath-to-target | POST image; target box x/y/w/h, point x/y, or target_id; optional n_simulations, use_saliency, display_preset | filename, n_elements, target_id/center/bbox, saliency_used, scanpath, target_load, display_preset, glance_metrics. Evidence: stage1/app.py:660–716 and 887–899. |
| /api/cognitive-load | POST image; form/query use_trained_model; form task_type, target_specificity, time_pressure, search_mode, profile_preset, display_preset | filename; visual/saliency/HCEye fields; source flags; task/profile; base and adjusted predictions; full vector; coherence; reference; display; search, contrast, readability, and detected-element diagnostics. Evidence: stage1/app.py:959–1017 and 1349–1394. |
| /api/screen-consistency | POST several files under images, or one file under image; GIF supported | Requires at least two decoded screens; returns consistency result and per-screen element counts. Evidence: stage1/app.py:1480–1521, “Need at least 2 screens”. |
| /api/learning-curve | POST static image; query exposures, n_simulations, use_saliency, display preset | curve plus novice/expert summaries, filename, n_elements, and display metadata. Evidence: stage1/app.py:1524–1612. |
| /api/product-learning | POST screen set/GIF; query total_uses, n_simulations, display preset | product curve, screen metadata, display, and nested inter_screen_consistency. Requires at least two screens. Evidence: stage1/app.py:1615–1689. |

The API accepts filenames controlled by the caller and returns the original filename. stage1/app.py:412 says “results['filename'] = file.filename”; this later matters for the UI injection finding.

## 0.3 Python entry points

| Entry point | Actual mode |
|---|---|
| stage1/app.py:1694 | Starts Flask on localhost:5001, warms UMSI++, debug mode enabled. |
| stage1/visual_complexity.py:1239–1259 | CLI for one image or a directory/CSV. |
| stage1/build_feature_norms.py:188–347 | CLI for UEyes reference-norm generation, optional category/limit/no-saliency. |
| stage1/create_test_screenshot.py:73 | Generates a synthetic test screenshot. |
| saliency/umsi_model.py:590–594 | UMSI++ prediction CLI. |
| saliency/inspect_weights.py:61 | HDF5 checkpoint inspection. |
| saliency/saliency_features.py:270 | Synthetic Gaussian-map self-demo. |
| saliency/test_full_pipeline.py:1–68 | Top-level saliency demo that writes images; no assertions. |
| cognitive/test_jokinen.py:1–112 | Top-level search demo; catches saliency failure but still prints success. |
| hceye/hceye_features.py:438–464 | Build lookup from CSV or run a feature demo. |
| hceye/gaze/data_processing_pipeline.py:463 | Hardcoded local gaze CSV/plot demo. |
| stage2/regression_model.py:475–510 | Train/compare the optional circular regression scaffold. |
| tests/test_pipeline.py:87 | Direct end-to-end demo over bundled screenshots; not a valid pytest test. |
| scripts/baseline_comparison.py:180–323 | Three-config heuristic comparison and CSV output. |
| scripts/feature_ablation_shap.py:285–345 | Circular-data ablation/permutation/SHAP report. |
| scripts/hceye_analysis.py:1–53 | Top-level descriptive HCEye CSV analysis. |
| scripts/hceye_condition_comparison.py:302–372 | HCEye condition tests and optional API correlation. |
| scripts/ueyes_saliency_validation.py:285–336 | UEyes map comparison. |
| scripts/download_weights.py:109–169 | Download released checkpoint with only a size check. |
| scripts/osf_explore.py and scripts/osf_query.py | Top-level OSF API exploration. |
| scripts/generate_expose_docx.py | Top-level DOCX generation. |
| scripts/generate_pipeline_figure.py | Top-level pipeline-figure generation. |
| scripts/smoke_test_endpoint.py | Top-level live endpoint smoke request. |
| hceye/saliency_pred/train.py, dynamic_train.py | Legacy PyTorch trainers; parse arguments at import time. |
| hceye/saliency_pred/test.py, dynamic_test.py, debug.py | Legacy PyTorch evaluation/debug entry points; incomplete dependencies and local modules. |

## 0.4 Shell entry points

| Script | Responsibility |
|---|---|
| stage1/run_feature_norms.sh | Launches norm generation. |
| stage1/run_feature_norms_full.sh | Launches the full reference run. |
| stage1/status_feature_norms.sh | Reports norm-generation status. |
| planning/scripts/morning.sh and evening.sh | Planning/log workflow helpers. |
| planning/scripts/auto_push.sh and git_push_check.sh | Automated repository status/commit/push workflow. |
| planning/scripts/setup_autopush.sh, setup_cron.sh, setup_git_hook.sh | Installs automation hooks/cron behavior. |

These planning scripts are operational tooling, not part of the scientific pipeline, and contain machine-specific assumptions.

## 0.5 Two-way documentation diff

### Claimed in README/exposé/UI but absent or materially different

| Claim | Code reality | Assessment |
|---|---|---|
| “Integrating ... into the AIM Platform” | The repository is a standalone Flask app; there is no AIM package dependency, AIM service adapter, plugin manifest, pull request, or integration test. The title is generated at scripts/generate_expose_docx.py:97, “Integrating Cognitive Predictive Metrics into the AIM Platform”. | Contradicted. AIM methods inspired some metrics, but integration is absent. |
| Full vector drives the default multi-output prediction | stage1/app.py:1193–1202 constructs default outputs directly from h[5], h[3], then uses the 19-vector only if an optional model file exists and is requested. | Materially different. |
| Search model is goal-directed / task-driven | cognitive/jokinen_model.py:63–65 says “there is no top-down (goal-directed) guidance yet”; target identity is only used for a memory boost after exposure 1 and as the stopping condition. | Contradicted for novice/default search. |
| HCEye-calibrated screenshot prediction | hceye/hceye_features.py:201–205 says the exact mapping is “not empirically calibrated.” | Contradicted. Aggregate HCEye ratios are embedded, but the screenshot-to-ratio mapping is heuristic. |
| “Cognitive Load Score” is estimated mental effort | UI stage1/ui/index.html:3918 calls it “Overall estimated mental effort”; code produces a heuristic index with no load ground truth. | Overclaim. |
| Automotive screening/compliance | Exposé generator lines 432–437 calls it a “pre-deployment screening tool for automotive HMI designers”; the saliency/reference data are not automotive and the glance construction is synthetic. | Not substantiated. |
| HCEye precedent is automotive | scripts/generate_expose_docx.py:190–197 says “in automotive dual-task conditions” and describes a peripheral saliency filter. HCEye used web pages; the local code does not filter the spatial map by load. | Contradicted. |
| CR/POMDP/RL optimal-control implementation | Exposé lines 162–178 and 413–414 frame the contribution as optimal control. The deployed code is hand-authored winner-take-all rules; no POMDP, learned policy, reward optimization, or utility learning executes. | Contradicted. |
| Majority of visual metrics are AIM implementations | Exposé lines 248–266 makes that claim, but symmetry, hierarchy, chromatic composite, and interactive density are custom; only portions of congestion/subband/edge follow AIM-style code. | Overstated. |
| README API inventory is complete | README.md:165–174 omits /api/product-learning. | Documentation gap. |

README.md:31–35 is notably honest about the deployed formula and says search time is not summed into the score. README.md:44–55 also discloses domain transfer and unvalidated extensions. Those disclosures reduce, but do not cure, stronger claims elsewhere and in the result UI.

### Present in code but undocumented in the README

| Present capability | Class | Why |
|---|---|---|
| /api/product-learning and consistency-to-learning transfer | (i) methodologically load-bearing | Produces user-facing longitudinal/product claims and can appear in the thesis. Evidence: stage1/app.py:1615–1628. |
| Per-element contrast gate and “WCAG/ISO” report | (i) methodologically load-bearing | Alters search activation and supports accessibility/automotive claims. Evidence: cognitive/jokinen_model.py:137–151 and stage1/app.py:1295–1325. |
| OCR text density/readability feeding the CLI | (i) methodologically load-bearing | Text density enters duration increase and therefore the headline. Evidence: stage1/app.py:1149–1171; hceye/hceye_features.py:238–242. |
| Physical display presets, including hidden legacy IVI/cluster/HUD presets | (i) methodologically load-bearing | Change angular geometry/search time and enable glance output. Evidence: stage1/app.py:908–930. |
| Per-target score modifier and Monte Carlo spread | (i) methodologically load-bearing | Generates target-relative load/uncertainty language. Evidence: stage1/app.py:863–870. |
| Reference percentile comparison and coherence thresholds | (i) methodologically load-bearing | Drives result bands/diagnostics and calibration language. Evidence: stage1/app.py:1341–1382 and stage2/coherence_check.py:55–81. |
| Saliency-to-feature-only fallback in search and headline routes | (i) methodologically load-bearing | Changes model composition without a user-visible error. Evidence: stage1/app.py:626–632 and 1078–1086. |
| Optional trained Stage-2 model and circular data builder | (ii) alternative/experimental | Off by default, but explicitly available for thesis experiments. Evidence: stage2/regression_model.py:22–28. |
| Big Five presets | (ii) alternative/experimental | Optional, weakly grounded, and not calibrated. Evidence: stage2/user_profile.py:23–24. |
| Legacy PyTorch saliency pipeline and gaze preprocessing | (ii) alternative/experimental | Not invoked by the Flask path; incomplete. |
| Analysis, document, OSF, figure, and smoke scripts | (iii) incidental, except where used as evidence | Tooling around the core. Their results become load-bearing only if cited as validation. |
| In-memory LRU caches and startup warmup | (iii) incidental | Runtime optimization. Evidence: stage1/app.py:43–51 and 151–158. |
| Planning auto-push/cron/hook scripts | (iii) incidental | Repository workflow, not scientific computation. |

# Task 1 — Actual deployed-output trace

## 1.1 Data path

The headline endpoint is /api/cognitive-load at stage1/app.py:959. It:

1. Reads the uploaded image bytes and hashes them.
2. Computes the eight visual features in this exact order at stage1/app.py:1022–1031:

   \[
   v=[H_{\text{gray}}, E, C, SBE, Sym, Chrom, Hier, Dens].
   \]

3. Attempts UMSI++ and constructs the five saliency values in this exact order at stage1/app.py:1043–1049:

   \[
   s=[Disp, Ent_s, Cov, Peaks, Center].
   \]

4. Detects contour elements; derives a union-box whitespace ratio; optionally runs OCR and derives the fraction of detected elements containing text.
5. Passes named dictionaries, whitespace, and text density into HCEyeFeatureExtractor at stage1/app.py:1163–1171. This named interface fixes a previous positional-order risk.
6. Builds v+s+h, plus task and profile vectors, but the default prediction ignores the concatenated vector.
7. Sets default outputs from h alone at stage1/app.py:1193–1197:

   “cognitive_load_score”: h[5] × 100,  
   “search_efficiency”: clip(1 − h[3]),  
   “attention_demand”: clip(h[5] + 0.15).

8. Adds task and profile modifiers to the headline at stage1/app.py:1204–1217.
9. Runs the separate Jokinen search model afterward for coherence and element feedback. Search time never enters the headline.

## 1.2 Exact default formula

Let \(P_k(x)\) be piecewise-linear percentile interpolation against the committed reference anchors min, p5, p25, p50, p75, p95, max, mapped to 0, .05, .25, .50, .75, .95, 1. The implementation is hceye/hceye_features.py:287–331. Missing values/norms return 0.5; non-finite values are not guarded.

Define:

\[
e=P_{\text{edge}}(E),\quad
c=P_{\text{congestion}}(C),\quad
d=P_{\text{density}}(Dens),
\]

\[
sy=P_{\text{symmetry}}(Sym),\quad
g=P_{\text{hierarchy}}(Hier).
\]

Whitespace \(w\) is image-derived from the union of contour boxes if detection succeeds; otherwise:

\[
w=\operatorname{clip}(1-d,0,1).
\]

Text share \(txt\) is OCR-derived when EasyOCR succeeds; otherwise \(txt=0.5\).

The six HCEye values are:

\[
complexity=(e+d+c)/3
\]

\[
F=\operatorname{clip}\left(0.876-0.05(complexity-w),0.6,1.1\right)
\]

\[
D=\operatorname{clip}\left(1.081+0.1(txt+d)/2,0.8,1.5\right)
\]

\[
E_x=\operatorname{clip}\left(0.935-0.04\left[1-(sy+g)/2\right],0.7,1.0\right)
\]

\[
A=\operatorname{clip}\left(\frac{0.115-0.069}{0.115}\,[0.7+0.6(1-w)],0,1\right).
\]

If saliency succeeds:

\[
need=(P_{\text{dispersion}}(Disp)+1-P_{\text{coverage}}(Cov))/2.
\]

Otherwise \(need=complexity\). Then:

\[
HL=\operatorname{clip}(0.5+0.4\,need,0,1).
\]

Finally:

\[
CLI=\operatorname{clip}\left(
\frac{
0.30(1-F)+0.20(D-1)+0.20(1-E_x)+0.15A+0.15(1-HL)
}{0.3},0,1\right).
\]

Evidence for these equations is hceye/hceye_features.py:231–276, including “complexity = (edge + element_count + layout_complexity) / 3.0” and the five weighted terms at lines 269–275.

For task values \(T,S,P,M\) from the categorical lookup tables:

\[
task=\operatorname{clip}(8T+6S+10P+7M-15,-8,12).
\]

Evidence: stage2/task_descriptor.py:99–116.

For profile openness \(O\), conscientiousness \(C_p\), and neuroticism \(N\):

\[
profile=\operatorname{clip}(4N-2.5C_p-1.5(1-N)+O-1,-5,6).
\]

Evidence: stage2/user_profile.py:82–110.

The deployed default headline is therefore:

\[
\boxed{Score=\operatorname{clip}(100\,CLI+task+profile,0,100)}.
\]

Evidence: stage1/app.py:1193–1207.

## 1.3 Image-derived versus hardcoded

Image-derived, subject to detector/model validity:

- Eight raw visual descriptors.
- UMSI++ map and class head, if the model loads.
- Saliency dispersion and coverage used by the headline; other saliency summaries are returned only.
- Element boxes, whitespace, OCR text share.

Hardcoded or hand-selected:

- HCEye aggregate constants 0.876, 1.081, 0.935, 0.115, 0.069.
- All screenshot-to-HCEye slopes: 0.05, 0.1, 0.04, 0.7, 0.6, 0.5, 0.4.
- CLI weights .30/.20/.20/.15/.15 and divisor .3.
- Task tables, task slopes, intercept, and clipping.
- Profile preset trait values, slopes, intercept, and clipping.
- Search-efficiency and attention-demand transforms.
- Reference percentile interpolation and its external distribution.

The comments correctly disclose part of this. hceye/hceye_features.py:201–205 says the “exact weighting is not empirically calibrated”; task_descriptor.py:18–20 calls the task weights “ordinal approximations.”

## 1.4 Computed or advertised inputs that do not enter the default headline

- Shannon entropy, subband entropy, and chromatic coherence are computed and returned but not read by the HCEye rules. Their trail ends in v construction at stage1/app.py:1022–1031 and the unused vector at 1180–1190.
- Saliency entropy, peak count, and center bias are computed and returned but not used by the default HCEye rule. Only dispersion and coverage are read at hceye/hceye_features.py:257–266.
- The full 19/30-dimensional vector is constructed at stage1/app.py:1180–1190, but default outputs are assigned directly from h at 1193–1198.
- Predicted search time is computed after the score, at stage1/app.py:1219 onward, and is used for coherence/bottleneck feedback, not the score.
- Display geometry affects search feedback and glance metrics, not the headline image score. stage1/app.py:1014–1017 explicitly says “pixel features are unaffected.”
- Agreeableness and extraversion appear in the returned profile vector but do not enter profile_modifier. Evidence: stage2/user_profile.py:75–108.
- Task vector’s derived visual_search_intensity and decisional_demand are returned but score_modifier reads only vec[0] through vec[3]. Evidence: stage2/task_descriptor.py:84–116.

## 1.5 Request-handler magic constants

Constants that should be model configuration/versioned calibration, but live in orchestration or UI:

- Target-load conversion: POINTS_PER_UNIT=25 and MODIFIER_CAP=15 at stage1/app.py:863–870.
- Rule-based search efficiency/attention transforms and task/profile slopes at stage1/app.py:1193–1217.
- Contrast threshold 3.0 for all detected elements at stage1/app.py:1295–1323.
- Default/cap for simulations: 100 and 500 at stage1/app.py:594–637, 1556–1592, and 1650–1674.
- UI load bands 25/50/75 and redesign advice at stage1/ui/index.html:4177–4186.

# Task 2 — Construct validity of visual and per-element metrics

## 2.1 Visual metrics

| Metric | Implemented definition and evidence | Canonical/construct audit |
|---|---|---|
| Shannon entropy | \(H=-\sum p(k)\log_2p(k)\), 256-bin grayscale histogram. stage1/visual_complexity.py:480–508: “H(I) = - Σ p(k) · log₂(p(k))”. | This is a valid intensity-histogram entropy, not semantic information or cognitive load. A non-empty OpenCV image is well-defined; an empty array would divide by zero at line 504. It is not used by the deployed headline despite being advertised in the combined vector. |
| Edge density | Canny after Gaussian sigma 2, thresholds 28/69; edge pixels divided by all pixels. stage1/visual_complexity.py:523–551. | A defensible image descriptor and close to the named AIM-style proxy. Fixed pixel sigma/threshold behavior is resolution-dependent: resizing the same UI changes edge survival and edge count. It is not scale invariant. |
| Feature congestion | Multi-scale color covariance, local contrast variation, orientation covariance, then normalized channel sum. stage1/visual_complexity.py:562–603. | The structural design resembles Rosenholtz/AIM, but its Lab input is invalid. stage1/visual_complexity.py:91–107 first scales RGB/XYZ to 0–1, then divides by 95.047/100/108.883. The docstring promises L in [0,100] at line 88; white actually returns L*≈8.99. This changes color covariance and downstream percentiles. Blocks use as a valid Rosenholtz reproduction. |
| Subband entropy | Mean entropy across steerable-pyramid subbands, chroma weighted 1/16. stage1/visual_complexity.py:793–840. | Also corrupted by Lab units. Its helper has a degenerate bin bug: when N=1 and nbins is inferred, nbins becomes 1 but the explicit “elif nbins == 1” is skipped; line 459 calls np.histogram with bins=0. It uses natural-log histogram entropy, which is acceptable if consistent, but the implementation computes edges with nbins−1, contradicting “√N bins.” |
| Layout symmetry | Mean grayscale NCC with left-right and top-bottom flips, clipped. stage1/visual_complexity.py:847–894. | This is a custom raw-pixel mirror correlation, not a verified implementation of Miniukovich & De Angeli’s interface-aesthetics symmetry metric. A uniform screen returns 1.0 by fiat at lines 883–885. The function heading says “custom,” but the docstring and API still cite the paper in a way that implies pedigree. |
| Chromatic coherence | Mean of normalized luminance SD, Hasler colourfulness, hue circular SD, saturation SD. stage1/visual_complexity.py:901–973. | The calculation measures palette fragmentation/variation: high is less coherent. The docstring is honest (“Color Palette Fragmentation” and “fragmented ... scores high”), but the name “coherence” points in the opposite intuitive direction. Achromatic pixels receive OpenCV hue zero and are included unweighted in circular hue statistics. The metric does not enter the headline. |
| Visual hierarchy | Mean of multi-threshold Canny persistence and top-three Otsu component area. stage1/visual_complexity.py:980–1065. | Current edge-persistence direction is internally consistent; the previously suspected inversion is not present. However, this custom construction is not Tuch et al.’s experimentally measured visual complexity/hierarchy construct. Otsu’s foreground can select a large white background component, making size dominance an artifact. Uniform-image handling assigns a single blob size_gradient=1 at line 1062. |
| Interactive element density | Filtered external contours divided by image area/10,000. stage1/visual_complexity.py:1072–1141. | This is explicitly a heuristic, not an interactive-control count. The value is “per 100×100 pixels,” not per physical/visual area, while fixed 5-pixel morphology changes detection after resizing. The local variable “solidity” is actually contour area / bounding-rectangle area (extent), not canonical convex-hull solidity. |

The committed norms remain contaminated by the Lab defect. stage1/data/results/feature_norms.json:20 says “visual_hierarchy recomputed ... all other features reused unchanged.” Since feature congestion and subband entropy use the defective conversion, every percentile-based use of those values inherits the error.

## 2.2 Per-element features

| Feature | Implementation | Audit |
|---|---|---|
| Bounding boxes / element identity | External contours after fixed Canny, 7×7 closing/dilation; min 400 px², max 50% of image, cap 80. cognitive/element_detector.py:46–56 and 95–147. | Not semantic controls. Fixed pixel thresholds make the set resolution-dependent; cap silently removes smaller elements. A “target” may be a merged panel, text block, or decorative contour. |
| Dominant colour | OpenCV k-means with k≤3 and random centers; largest cluster; HSV bucket. cognitive/element_detector.py:238–273 and 361–396. | A custom coarse category, not the paper’s full feature representation. No OpenCV RNG seed is set, so identical API inputs can yield different categories and search activation. |
| Colour category | Hue/saturation/value thresholds for black/white/gray/red/orange/yellow/green/cyan/blue/purple. cognitive/element_detector.py:361–396. | Explicitly “extended”; no empirical confusion/perceptual-distance model. Treats any category difference as equally dissimilar. |
| Angular size | Bounding-box diagonal converted with the average of x and y pixels/cm. cognitive/element_detector.py:152–177. | Incorrect for non-square pixel-to-physical scaling: x and y should be converted separately before taking the physical diagonal. screen_height_cm is otherwise unused by the search model’s deg_per_px, which uses only screen width at cognitive/jokinen_model.py:337–340. |
| Contrast | Otsu-split mean light/dark luminance ratio. cognitive/element_detector.py:311–358. | The WCAG luminance formula itself is correct, but the foreground/background pair is inferred from all pixels inside a detector box. It need not correspond to adjacent text/icon versus its background, so it is not a WCAG conformance measurement or ISO 15008 test. The 3:1 threshold is applied to every object, although WCAG normal text requires 4.5:1 and 3:1 has narrower large-text/non-text meanings. |
| “WCAG pass” | contrast_ratio >= 3.0. cognitive/element_detector.py:280–283 and 188–190. | Over-broad label. It cannot determine text size, boldness, component boundary, essential graphics, state, or adjacency. |
| Whitespace | One minus union mask of detected boxes. stage1/app.py:1117–1145. | Correctly avoids double-counting overlaps, but inherits a non-semantic, capped, resolution-sensitive detector. |
| Text density | OCR text-bearing detected elements / detected elements. stage1/app.py:1149–1159. | This is a share of contour boxes, not text area, word count, reading time, or density per area. With OCR unavailable it becomes exactly 0.5 and still affects the headline. |

# Task 3 — Saliency-stage integrity

## 3.1 Architecture parity is not established

The local file claims “Faithful TF2 / Keras 3 port” at saliency/umsi_model.py:6. The local decoder at lines 321–347 uses Conv2D layers with no activation, no dilation_rate, and use_bias=False, including the final heatmap convolution.

The official UEyes source defines decoder_block with:

- activation='relu' on dec_c1 through dec_c5;
- dilation_rate=(2,2);
- default Conv2D bias behavior;
- activation='relu' on the final one-channel output.

The official source is available at:
https://github.com/YueJiang-nj/UEyes-CHI2023/blob/main/saliency_models/UMSI%2B%2B/src/classif_capable_models.py

The official UMSI function also uses decoder_with_skip in the retrieved source, whereas the local architecture summary describes a plain Conv/upsampling chain. The repository therefore has not established that its layer graph or checkpoint tensor layout corresponds to the checkpoint it tells users to download.

Impact: if the exact checkpoint expects biases, different dilation semantics, skip layers, or ReLU-constrained outputs, strict loading should fail or predictions will not be equivalent. Because the checkpoint is absent, the exact outcome is UNVERIFIED. Required evidence: exact checkpoint SHA-256, a layer-by-layer configuration diff, a successful strict load, and golden predictions against the official environment on fixed images.

## 3.2 Weight loading is strict, but the endpoint can still degrade

Positive finding: the wrapper refuses partial/random loading. saliency/umsi_model.py:511–527 says “skip_mismatch=False” and raises rather than “fall back to a partial load.” There is no silent mismatched-layer initialization inside UMSIPlus.

However, the main cognitive route catches every saliency exception, sets s=None, and continues. stage1/app.py:1078–1086 says it “degrades to image-only features.” The search route does the same at lines 626–632. The server console gets a warning, but the result UI has no salient “saliency unavailable” status; it simply lacks the map and still renders a confident Cognitive Load Score. This is a behaviorally silent fallback for an end user, even if it is not silent in stdout.

## 3.3 Provenance and integrity

- saliency/umsi_model.py:30–32 identifies an Aalto URL.
- scripts/download_weights.py:33–36 defaults to a GitHub release asset in this repository.
- scripts/download_weights.py:155–164 validates only that the file is larger than 100 MB.
- No checksum, release digest, model card, signed manifest, or committed layer inventory binds that asset to the official UEyes checkpoint.
- Documentation conflicts on size: scripts/download_weights.py:5 says approximately 433 MB; lines 56–58 say approximately 115 MB.

Size is not provenance. A correct next audit needs a SHA-256 published in the repository and an upstream-source mapping.

## 3.4 Pre/post-processing

The BGR VGG means and aspect-preserving zero padding at saliency/umsi_model.py:364–425 broadly match the official style. Output unpadding mirrors the input branch.

Defects:

- Extremely narrow/tall images can make new_cols or new_rows zero at lines 410–419, leading cv2.resize to fail.
- postprocess_saliency divides only by a positive maximum at lines 463–468. It does not subtract a negative minimum or clip. Because the local decoder lacks final ReLU, negative saliency can survive while the docstring promises [0,1].
- Negative maps can then cause invalid weighted variance and inconsistent histograms. Rendering with uint8 can wrap negative values.

## 3.5 Domain

The official UEyes study reports 62 participants and 1,980 screenshots spanning four UI types: webpage, desktop, mobile, and poster. The repository’s class list at saliency/umsi_model.py:483–493 asserts six training categories including infographic and natural_image. A six-way auxiliary head may exist in an augmented model, but the comment conflates classifier classes with the published UEyes study domain.

Automotive HMI is absent from UEyes. README.md:44–49 correctly calls use on cars out-of-domain transfer. But stage1/app.py:1050–1073 treats desktop_ui as within an “automotive/desktop-style” target, and the UI says desktop is “within the model's intended UI domain” at stage1/ui/index.html:3966–3968. Desktop recognition is not evidence of automotive validity.

# Task 4 — Cognitive search-time and load model

## 4.1 Implemented search terms

Implemented:

- bottom-up color-category and relative-area dissimilarity;
- UMSI++ mean saliency per element;
- logistic activation noise;
- size/eccentricity acuity penalty;
- uncalibrated contrast penalty;
- winner-take-all selection;
- EMMA encoding and saccade timing;
- inhibition of return;
- optional target-location memory boost for repeated exposures.

Evidence: cognitive/jokinen_model.py:885–950 implements acuity, contrast, memory, and EMMA; lines 1053–1135 define bottom-up and combined activation.

Dead parameters are explicitly acknowledged at cognitive/jokinen_model.py:56–94:

- W_TA;
- a_color, b_color, a_shape, b_shape;
- B_bl, F_mem, f_mem, sigma_M;
- alpha, sigma_U, B_sa.

The model therefore is not the full Adaptive Feature Guidance model from Jokinen et al. It lacks default goal-feature guidance, long-term memory retrieval, and utility learning.

## 4.2 Target semantics are not goal-directed at default

For n_exposures=1, target_idx does not change activation. The only target-specific boost is inside “if p.n_exposures > 1” at cognitive/jokinen_model.py:905–912. At default, every target gets the same stochastic bottom-up ranking; target identity determines only when the loop stops at lines 956–972.

This directly contradicts stage1/app.py:673–676, which calls the returned path a “TASK-DRIVEN scanpath (goal-directed search).” A user drawing a red search target versus a blue target does not supply a desired feature template to the novice search policy.

## 4.3 Formula differences and constants

The code docstring states:

\[
BA_i=\sum_j\sum_k dissim(v_{ik},v_{jk})/d_{ij}.
\]

But cognitive/jokinen_model.py:1114–1115 implements division by \(\sqrt{d_{ij}}\). This is a material formula mismatch.

Hardcoded assumptions include:

- EMMA K=.006, exponent=.4, preparation=.135 s, execution=.002 s/degree, overhead=.07 s, frequency=.1;
- W_BA=1.1; logistic noise .376;
- size acuity .142 and .96; visibility penalty 5;
- contrast weight 3 and threshold 3:1;
- saliency weight .8 and exponent 2;
- max fixations 50, 100 simulations, seed 42;
- width-based 37 cm/60 cm degree conversion.

Evidence: cognitive/jokinen_model.py:96–150 and 207–218.

W_contrast can dominate the normalized feature activation (maximum 1.1) and saliency contribution (maximum .8): a maximum penalty of 3 is larger than the positive modeled signal. The comment at lines 147–150 correctly says it needs calibration.

Searches that hit max_fixations return accumulated time/count with no “target_not_found” or censoring flag at cognitive/jokinen_model.py:983–986. Those censored trials are averaged as ordinary successful search times.

## 4.4 Load combination

The weighted CLI is not derived from a theory that identifies cognitive load from screenshots, and its weights are not fitted. hceye/hceye_features.py:201–205 explicitly admits the exact weighting is not calibrated. HCEye’s empirical condition effects do not justify assigning per-image differences in edge density, whitespace, text share, or symmetry to different load-effect ratios.

Highlight effectiveness is particularly problematic. It predicts the hypothetical benefit of an intervention, then contributes protectively as \(1-HL\) to the present screen’s load even when no highlight is present. Evidence: hceye/hceye_features.py:257–275. A screen that would benefit from highlighting is not therefore less demanding before highlighting.

The “element_count” mapping labels interactive_element_density as “equal” at hceye/hceye_features.py:77–83, even though one is a contour-per-pixel heuristic and the HCEye construct is not established as that same measurement.

## 4.5 Feature order

The live HCEye call is now semantically safe: stage1/app.py:1166–1171 passes named dictionaries, and hceye/hceye_features.py:139–142 says this “structurally rules out the position-mismatch bug.”

The optional regression’s base vector order matches Stage2Model’s declared order:

- visual: eight keys at app.py:1022–1031 and regression_model.py:83–87;
- saliency: dispersion, entropy, coverage, peaks, center at app.py:1043–1049 and regression_model.py:90–96;
- HCEye: six values at hceye_features.py:278–285 and regression_model.py:97–101.

No current live-order mismatch was found.

## 4.6 Optional regression is circular

stage2/regression_model.py:339–362 already admits:

- y1 equals x[18]×100;
- y2 is a transform of x[13];
- y3 is a transform of x[14];
- v and s are fabricated from gaze statistics rather than extracted from images.

The exact synthetic blocks appear at lines 399–425, and targets at 440–455. Any R², RMSE, SHAP, feature importance, or ablation from this data is tautological and may not be used as scientific evidence.

There is also preprocessing leakage in Stage2Model.train: the scaler is fit on all X before the validation split at lines 198–205, and the same globally scaled data is used in CV at lines 240–248.

# Task 5 — Newer and extension features

## 5.1 Learning curve

Mechanism:

\[
memory(n)=\ln n,\qquad activation[target]\mathrel{+}=W_{memory}\ln n.
\]

Evidence: cognitive/jokinen_model.py:169–198 and 905–912.

Calibration:

- W_memory=1.0 is explicitly a placeholder needing repeated-use data.
- n=1 produces zero boost. The implementation resets the RNG before each curve point and restores both n_exposures and RNG in a finally block at lines 465–491. This is good state hygiene and makes the first curve point equal the novice path when exposure 1 is included.

Validity:

- ACT-R/base-level literature supports practice and memory effects generally, not this exact target-only additive activation, its scale, or its transfer to arbitrary screenshots.
- The code claims monotonic decreasing/diminishing returns at lines 425–429 and 505–508. Log boost is monotonic, but stochastic winner-take-all, finite runs, rounding, clipping, and discrete ranks do not mathematically guarantee each reported point strictly decreases or has diminishing marginal change. This requires property tests and empirical calibration.

## 5.2 Inter-screen consistency

Calibration constants:

- 10×10 occupancy grid;
- match weights 1.0/.3/.3;
- match cost threshold .35;
- displacement-to-score factor 2.

Evidence: stage2/screen_consistency.py:52–67 and 265–269. The file appropriately says the mapping is unvalidated.

Implementation defects:

- The occupancy docstring says “union” at lines 82–84, but lines 108–138 add every rectangle’s overlap then clip. Overlapping boxes are double-counted until saturation, not unioned.
- Headline consistency uses matched displacement whenever even one match exists, ignoring unmatched replacements and occupancy: lines 323–330.
- A verified synthetic case with one stable element plus five replacements returned consistency_score=1.0 while occupancy_consistency=0.5.
- Two empty screen element sets return 1.0, even though no evidence of stable controls exists.

Thus “consistent screen set > shuffled one” is not a reliable invariant under replacements/detector failures.

## 5.3 Product learning

The coupling is:

\[
effective=total\,[C+(1-C)/N].
\]

Evidence: cognitive/jokinen_model.py:539–553.

The file is honest that this is a first unvalidated interpolation. Citations about contextual cueing/transfer do not establish this linear formula. Rounding at lines 607–610 makes nearby conditions identical, and empty screens are skipped at lines 612–626, changing the effective product without a warning.

## 5.4 Contrast and visibility gate

The deficit is multiplied by W_contrast=3 at cognitive/jokinen_model.py:897–903. The code declares the weight uncalibrated. Direction is correct in code: lower inferred contrast lowers activation and generally delays selection. The specific magnitude, the Otsu proxy, and the use of a universal 3:1 threshold are not supported as human detection-time calibration or standards conformance.

## 5.5 Glance / eyes-off-road metrics

compute_glance_metrics greedily packs modeled fixation step times into artificial glances capped by a chosen 1.5-second budget, then compares the resulting chunks to 2- and 12-second values. Evidence: cognitive/jokinen_model.py:1251–1276 and 1330–1360.

Because the algorithm itself ends each glance at 1.5 seconds, most multi-step glances are designed to be below the 2-second check. Only a single modeled fixation longer than the budget can exceed it. NHTSA guidance describes test procedures and acceptance criteria for measured task performance; it does not validate deriving glances by partitioning a screenshot search simulation. The returned “compliant” and UI “within limits” are therefore scientifically and regulatorily misleading.

## 5.6 Readability/OCR

OCR executes once and is reused, which is good for performance. But text share feeds the CLI while OCR absence substitutes a neutral .5. An endpoint result can therefore change depending on whether an optional local package/model happens to load. The API returns text_density_source at stage1/app.py:1358–1365; the main score UI does not foreground that uncertainty.

## 5.7 Display presets

Nominal phone/laptop/desktop geometry is shown; automotive presets remain callable but hidden. Evidence: stage1/app.py:916–930. Unknown values silently become desktop at lines 934–947. Physical geometry changes width-based search timing, but the vertical conversion is not correctly used. These are assumptions, not calibrated displays.

## 5.8 Per-target modifiers and uncertainty

Target relative load converts deviation from average search time into 25 points per relative unit, capped at 15, at stage1/app.py:863–870. These constants are not calibrated.

The per-target standard deviation is across model noise simulations, as cognitive/jokinen_model.py:364–369 correctly describes. UI stage1/ui/index.html:2975–2981 calls it “model uncertainty,” which invites a wider epistemic interpretation. It excludes model-form, detector, coefficient, checkpoint, domain-transfer, and data uncertainty.

## 5.9 Screen-set/GIF contract

The endpoints cleanly reject fewer than two decoded screens at stage1/app.py:1498–1502 and 1635–1639. A single-frame GIF therefore receives a correct 400.

Contract inconsistency: multiple files supplied under images are decoded with cv2.imdecode, so an animated GIF contributes only its first frame; a single GIF under image is expanded frame-by-frame with PIL at lines 1435–1473. The same GIF produces different screen counts depending on field name.

# Task 6 — Reproducibility

## 6.1 Requirements do not resolve

requirements.txt:5 pins numpy==1.26.4. requirements.txt:10 pins opencv-python==4.13.0.92. The latter release requires NumPy 2 or newer, producing an unsatisfiable environment. The README setup command at README.md:99–106 therefore cannot complete as written.

The comment at requirements.txt:28–31 explicitly insists on NumPy 1.26.4 to protect TensorFlow, which confirms this is not an accidental loose constraint. A reproducible solution needs a compatible OpenCV version or a redesigned TensorFlow/NumPy stack, then a lock file tested on the declared Python/platform matrix.

## 6.2 Hard imports missing from requirements

Examples:

- h5py: saliency/inspect_weights.py:15, “import h5py”.
- python-docx: scripts/generate_expose_docx.py:3, “from docx import Document”.
- requests: scripts/osf_explore.py:2 and scripts/osf_query.py:2; also an in-function import at hceye_condition_comparison.py:230.
- cleanETcode: hceye/gaze/data_processing_pipeline.py:4.
- Legacy stack: torch, torchvision, tqdm, torchsummary, pytorch_ssim, fvcore, ptflops throughout hceye/saliency_pred.
- The legacy saliency scripts also import local model.py/utils.py modules that are not present.
- xgboost is optional and correctly guarded for the default path, but a user selecting xgb must install it separately.

These make “every entry point” unreproducible even if the core environment were fixed.

## 6.3 Determinism

Positive:

- Jokinen uses np.random.default_rng(seed=42).
- scikit-learn and XGBoost configurations use random_state=42.
- Learning resets and restores RNG in a finally block.

Negative:

- OpenCV k-means uses cv2.KMEANS_RANDOM_CENTERS at cognitive/element_detector.py:263–265 without cv2.setRNGSeed. This can change element colour categories, feature dissimilarity, matching, and search output.
- TensorFlow determinism settings are absent.
- Exact dependency/build/platform state is not locked.

## 6.4 Tests and commands

README.md:184–195 now honestly describes tests/test_pipeline.py as a direct demo. However:

- requirements.txt:37 still says the README documents pytest, which is stale.
- tests/test_pipeline.py:20 declares test_full_pipeline(image_path), so pytest will treat image_path as a fixture and fail collection unless such a fixture is supplied.
- No assertions occur in that file.
- Its no-image branch constructs Stage2Model with an absent model and then calls predict at lines 101–125.
- saliency/test_full_pipeline.py is top-level and prints “SUCCESS” at lines 66–68 if it reaches the end, but has no numeric golden assertions.
- cognitive/test_jokinen.py catches saliency errors at lines 104–109 and still prints “SUCCESS” at 111–112.

There is no CI workflow to enforce installation, unit tests, API tests, numerical invariants, or artifact integrity.

## 6.5 Missing artifacts

- UMSI++ weights.
- UEyes images, fixation density/binary fixation data, and per-image feature CSV.
- HCEye raw CSV.
- Any trained Stage-2 model.
- Independent automotive labels, search times, scanpaths, NASA-TLX, or gaze data.
- Complete legacy saliency source modules and environment.

The norm JSON is committed, but provenance is insufficient to reproduce it byte-for-byte: the raw dataset and per-image CSV are absent, the checkpoint is unverified, and line 20 says most values were reused from an earlier run.

## 6.6 Norm builder failure behavior

stage1/build_feature_norms.py:209–212 returns normally when no dataset exists; lines 307–309 also return normally if every image failed. Shell automation can therefore report success while writing nothing. Metadata at line 321 always reports sorted _GUI_CATEGORIES even when a single --category run was requested, rather than the actual processed set.

## 6.7 Machine-specific paths

- hceye/gaze/data_processing_pipeline.py:449–474 hardcodes ./analysis_data/anna_debug paths.
- The gaze display is fixed at 883×682 at lines 7–14.
- Planning automation assumes a specific local thesis checkout/home layout.
- UEyes scripts default to a separate excluded repository path at ueyes/saliency_models/UMSI++.

# Task 7 — Robustness, edge cases, and safety

## 7.1 Image edge cases

| Input | Behavior |
|---|---|
| Missing image / empty filename | Clean 400 in main routes. Evidence: stage1/app.py:977–986. |
| Non-image bytes with valid extension | Search/learning explicitly test cv2.imread and return 400. Analyze and headline enter broad exception paths and can return a 500 containing internal error text. |
| Empty/1×1 image | Image files cannot be zero-pixel in normal codecs, but a 1×1 valid file can fail Gaussian/pyramid/Canny/model resizing. No minimum dimension check exists. UMSI padding can calculate a zero resized dimension. |
| Huge image | Entire upload is read into memory before validation; visual pyramids, OCR, element detection, output overlay, and full-resolution saliency caching then amplify memory. No byte or pixel limit. |
| Zero detected elements | /api/search-time and /api/learning-curve return HTTP 200 with an error field. /api/cognitive-load still produces a headline using fallbacks. |
| Single-frame GIF for set endpoint | Decodes one frame, then cleanly rejected with 400 because fewer than two screens. |
| 0/1 screen set | Clean 400 at endpoint layer. The lower-level consistency function permits one screen and calls it 1.0, but endpoint requires two. |
| Missing/misspelled categorical form | Unknown task selections silently use default numeric weights while the response retains the supplied invalid string. Unknown profile returns neutral and reports key neutral. Unknown display silently becomes desktop. |
| Non-numeric n_simulations | Werkzeug typed get falls back to the default; negative and zero integers pass through. |
| n_simulations≤0 | Empty trial lists reach np.mean/np.std at cognitive/jokinen_model.py:351–370, producing NaN; scanpath representative indexing can fail. No lower bound exists. |

## 7.2 Broad exception paths

Broad catches that can hide a broken stage behind a plausible result:

- stage1/app.py:1078–1086: saliency failure becomes feature-only headline.
- stage1/app.py:1133–1161: detector/OCR failures become whitespace/text fallbacks.
- stage1/app.py:1225–1331: search/coherence failure leaves the headline intact.
- stage1/app.py:1074–1077: classifier failure is discarded.
- stage1/app.py:1113–1115: overlay failure is cosmetic and appropriately isolated.
- cognitive/test_jokinen.py:104–109: demo hides saliency failure and still announces success.
- stage1/build_feature_norms.py:299–300: continues over all failed images.

The code logs several of these, which is better than a completely silent branch, but the API/UI does not consistently expose degradation.

## 7.3 Shared state

Positive: learning and product learning restore n_exposures and RNG in finally blocks at cognitive/jokinen_model.py:488–491 and 649–651.

Risk:

- Global UMSI singleton and two OrderedDict caches in stage1/app.py:40–51 have no lock. Multi-threaded Flask requests can race on initialization/move/insert/eviction.
- Global EasyOCR/model internals may also have framework-specific concurrency constraints.
- Debug reloader can instantiate state twice.

## 7.4 Resource limits

No Flask MAX_CONTENT_LENGTH is configured. _hash_upload reads the complete stream at stage1/app.py:161–169. Screen sets and GIF frame counts are unbounded. Compressed-image decompression and pixel count are not limited. The cache count is bounded, but each saliency entry stores a full-resolution float map; 32 attacker-selected very large images can exhaust memory.

## 7.5 Debug and information exposure

stage1/app.py:1710 runs “debug=True.” It binds only localhost, reducing remote exposure, but debug mode should not be used in a user study or shared deployment.

Most route-level exceptions return str(e) to the caller, for example stage1/app.py:415–416 and 1395–1396. Framework, file-path, shape, and model details can leak.

## 7.6 DOM injection

The server returns file.filename. The UI history then constructs HTML directly. stage1/ui/index.html:4828–4845 interpolates row.filename into both an HTML title attribute and a table cell before assigning content.innerHTML=html.

A crafted filename containing HTML/attribute syntax can execute DOM content in the session. Task/profile text is also interpolated. Use textContent/createElement or robust contextual escaping. This is a real implementation bug even for a localhost tool because study participants may supply files and exported/session history is trusted by the page.

## 7.7 No eval/exec or user-controlled shell execution found

No Python eval/exec path or shell command interpolation of upload/request fields was found. The weight downloader invokes git credential fill with a fixed argv list, not a shell. This specific class of risk was checked and is not present.

# Task 8 — Thesis correspondence and honesty

| Thesis/exposé/UI claim | Status | Evidence |
|---|---|---|
| AIM integration improves computational GUI evaluation | Contradicted/UNVERIFIED | Title at scripts/generate_expose_docx.py:97 and RQ at 138–139; no AIM integration code or comparative evaluation exists. |
| Pipeline estimates interactional complexity | Simplified but defensible only as exploratory heuristic | README.md:31–35 accurately describes the score composition. There is no external criterion validation, so “index” or “heuristic” is necessary. |
| Pipeline estimates cognitive load / mental effort | Not substantiated | UI stage1/ui/index.html:3918 says “Overall estimated mental effort”; hceye_features.py:201–205 says weights are not calibrated. |
| HCEye-calibrated empirical coefficients map screenshots to load | Contradicted | Exposé lines 293–296 says “HCEye-calibrated”; local code calls the mapping purpose-built and uncalibrated. |
| HCEye provides automotive dual-task precedent and shorter fixations | Contradicted | Exposé lines 190–197. HCEye stimuli were web pages, and local constants/code say duration increases under load: hceye_features.py:41–45. |
| HCEye layer suppresses peripheral saliency mass | Absent | Exposé lines 193–197 says it “act[s] as a load-proportional filter”; no code mutates the spatial saliency map based on CLI. |
| Full [v,s,h,t,p] vector is passed to multi-output prediction | Default path contradicted | Exposé lines 217–225 and UI lines 2078–2082; app.py:1193–1202 uses h-derived rules unless an unavailable optional model is requested. |
| Search efficiency describes target findability | Contradicted | app.py:1195 defines it as 1−h[3], not measured/predicted target search. UI lines 3923–3925 call it “How easily a user can find a specific target.” |
| Attention demand describes capacity required | Contradicted | app.py:1196 defines it as CLI+.15. UI lines 3928–3930 gives it an independent construct label. |
| CR/POMDP/RL optimal-control policy is tested | Absent | Exposé lines 162–178 and 413–414; cognitive/jokinen_model.py:23–26 says novice and no utility learning. |
| Saliency transfer is validated on automotive HMI | Not demonstrated | Exposé lines 271–279 says it is validated against automotive screenshots; the validation script defaults to UEyes sample paths and no committed results/data support automotive transfer. |
| Coherence flags identify calibration boundary but prediction remains valid | Unsupported | UI stage1/ui/index.html:4487–4491. The rules are logical thresholds; no calibration range or validity study exists. |
| Practical automotive screening tool | Overclaim | Exposé lines 432–437. The README’s out-of-domain caveat is incompatible with safety-style result bands and “compliance.” |
| Domain restriction is automotive only | Internally inconsistent | Exposé limitation lines 472–477 while the actual training/reference domain is non-automotive. |
| Majority of Stage 1 is established/AIM | Overstated | Exposé lines 248–266; several metrics are custom or only inspired by cited work. |

The exposé also contains a bibliographic error: scripts/generate_expose_docx.py:496 gives UEyes DOI ending 3581113; the paper/repository use 3581096.

The user-study proposal is comparatively candid about missing calibration, but lines 49–54 focus on previously identified ordering/hierarchy bugs that are no longer the main blockers. The current Lab and architecture-parity defects must be incorporated before freezing a study version.

# Task 9 — Statistical and analysis-script validity

## 9.1 Baseline comparison

scripts/baseline_comparison.py:101–108 multiplies raw Shannon entropy by 25 and congestion by 30 before clipping to 100. Typical entropy alone can exceed 100, so Baseline A saturates and is not a meaningful comparator.

Baseline B’s docstring at lines 123–125 calls h[5] “calibrated against” HCEye, which is false for per-image mapping.

The “full” call at lines 247–249 does not pass saliency. It therefore compares v+h+t+p while headers/documentation advertise v+s+h+t+p. The coherence call supplies all None inputs at lines 257–260, so no rules are checked and “coherent” is vacuous.

Allowed use: debugging that configurations execute.  
Not allowed: evidence that HCEye or the full pipeline improves accuracy, validity, or feature richness.

## 9.2 Feature ablation and SHAP

The script’s caveat at scripts/feature_ablation_shap.py:25–40 correctly recognizes direct target leakage and fabricated features.

Cross-validation scaling is correctly inside a Pipeline at lines 107–126. But the underlying X/y remain circular. Per-feature permutation/SHAP fits and evaluates the same full sample at lines 178–198, adding optimistic in-sample importance even beyond circularity.

Allowed use: demonstrating why the current scaffold is invalid.  
Not allowed: any R², block contribution, SHAP rank, or saliency-importance claim.

## 9.3 UEyes saliency validation

- Defaults point to five sample images in an excluded external repo at scripts/ueyes_saliency_validation.py:65–67.
- The model is constructed inside predict_saliency for each image at lines 195–208, wasting time and making failures repetitive.
- The NSS definition in the header is wrong: lines 14–15 say NSS=1 means fixations land at the model mean and >1 means over-prediction. NSS is the mean z-scored saliency at fixation points; 1 means one standard deviation above the map mean, not “mean prediction” or overprediction.
- The script later acknowledges at lines 263–277 that blurred density maps make its NSS uninterpretable.
- Generic “typical SOTA” cutoffs at lines 267–271 are not a valid benchmark without identical dataset, split, preprocessing, center-bias treatment, and metrics.
- Validation on UEyes samples is same-domain consistency, not automotive transfer.
- No result CSV/provenance is committed.

Allowed use: a labeled engineering sanity check on the exact same-domain sample, once architecture/checkpoint parity is established.  
Not allowed: automotive validity, independent generalization, or SOTA evidence.

## 9.4 HCEye condition comparison

The docstring claims participant/image aggregation at lines 27–31, but load_and_aggregate groups only Image_Name and CognitiveLoad at lines 124–133. The paired t-test therefore treats images as independent units, not participants. Repeated participant/image observations are collapsed without a mixed model.

cohens_d at lines 80–89 is the pooled independent-samples effect size even though the test is paired. A paired standardized mean difference should be used and reported with its convention.

The script runs at least Absent–High and Low–High tests for two outcomes without multiplicity control. It compares embedded coefficients to the same data used to derive them at lines 196–218, which is circular verification, not validation.

The optional correlation:

- accepts n≥3 at lines 261–280, far too small for stable inference;
- uses the first partial filename match at lines 271–277;
- expects negative correlation at lines 286–289;
- nevertheless prints any significant positive correlation as “consistent with H1” at lines 291–293.

Allowed use: rechecking descriptive aggregate constants, with correct caveat.  
Not allowed: validation of screenshot prediction, participant-level inference, or evidence of causal image susceptibility.

## 9.5 HCEye descriptive analysis

scripts/hceye_analysis.py:7–53 is descriptive only. Ratios per image can be useful exploratory summaries, but there is no inferential design, missing-data audit, participant clustering, uncertainty interval, or independent holdout.

## 9.6 Gaze-processing pipeline

This pipeline cannot currently establish trustworthy ground truth:

- missing cleanETcode hard import at line 4;
- fixed display 883×682 at lines 7–14;
- actualX/actualY may be unbound or carry a prior row’s value when only one coordinate of an eye is NaN at lines 150–171;
- when a fixation ends, the boundary sample is discarded instead of seeding the next fixation at lines 115–127;
- velocity divides displacement by time interval times sampling frequency at lines 305–314, reducing a velocity by an extra factor of 250;
- “if i < 0” can never run and assigning i=i+1 inside a Python for loop does not advance iteration at lines 333–355;
- main uses hardcoded debug paths at lines 463–474.

Any gaze features produced by this version may not be used as study ground truth.

## 9.7 Hardcoded/canned UI validation numbers

stage1/ui/index.html:2089–2096 embeds HCEye p, d, deviation, CC, and SIM values in static HTML. These do not come from the current API or a versioned result artifact. Even with caveat language, hardcoded values can outlive code/data changes and cannot be audited from the page.

# Task 10 — Front-end ↔ API fidelity

## 10.1 Field mapping and scaling

The three main cards map to:

- adjusted_prediction.cognitive_load_score, one decimal;
- adjusted_prediction.search_efficiency, three decimals;
- adjusted_prediction.attention_demand, three decimals.

Evidence: stage1/ui/index.html:3910–3930. No silent ×100 or swapped API key was found in those cards.

The what-if modifier functions at UI lines 4165–4174 numerically mirror the backend task/profile formulas. This is a positive fidelity finding, although both formulas are scientifically uncalibrated.

## 10.2 Semantic mislabeling

- Backend search_efficiency is 1−AOI sensitivity, app.py:1195. UI lines 3923–3925 describes actual target acquisition.
- Backend attention_demand is CLI+.15, app.py:1196. UI lines 3928–3930 describes attentional capacity.
- Backend score is a heuristic vulnerability blend. UI line 3918 labels overall mental effort and lines 4510–4524 assign confident load/overload verdicts.
- UI diagnosis lines 2478–2483 says the headline is “predicted saliency + eye-movement model”; default search-time output does not enter the score.
- Pipeline diagram line 2042 says ACT-R memory decay is integrated, although default novice mode has no ACT-R memory block.
- Pipeline diagram lines 2078–2082 says the full vector is passed to regression by default, which is false.

## 10.3 Dropped caveats

- When saliency fails, the backend gives only empty saliency fields and a server log. The result page hides the saliency panel and still renders the same confident score bands; it does not tell the user the model degraded.
- OCR source is returned, but the score card does not warn when a neutral text fallback changed the HCEye rule.
- UMSI domain caveat appears only for classes the code labels out-of-domain; desktop_ui is treated as automotive-compatible without evidence.
- Coherence UI says “prediction is still valid” and invokes a calibration range at lines 4487–4491, neither of which the backend establishes.
- Per-target stochastic spread is called model uncertainty at lines 2975–2981 without delimiting it to Monte Carlo noise.

## 10.4 Thresholds and verdicts

UI thresholds 25/50/75 at lines 4177–4186 are not calibrated to human outcomes. Phrases including “well within typical attentional capacity,” “notable attentional strain,” and “critical cognitive overload risk” at lines 4509–4524 turn arbitrary score bands into clinical/safety-like conclusions.

The glance badge says “within limits” at lines 3012–3032 even though the backend created the glances algorithmically. This is not compliance fidelity.

## 10.5 Neutral defaults are not neutral

The UI repeatedly calls defaults neutral at lines 1704–1756.

Actual default task:

\[
8(.65)+6(.55)+10(.55)+7(.30)-15=+1.1.
\]

Actual neutral profile:

\[
4(.5)-2.5(.5)-1.5(.5)+.5-1=-.5.
\]

Combined default adjustment is +0.6, not zero. The profile code comment itself says “no CL adjustment” at stage2/user_profile.py:39, contradicting the formula.

# Task 11 — Numerical stability and degenerate inputs

## 11.1 Headline NaN/Inf paths

1. Local UMSI output can be negative because the decoder lacks ReLU. postprocess_saliency only divides by a positive max. Weighted saliency dispersion can then compute negative variances before sqrt at saliency/saliency_features.py:123–135, yielding NaN.
2. HCEye percentile normalization passes float(value) to np.interp without np.isfinite at hceye/hceye_features.py:305–331. NaN survives interpolation.
3. CLI uses np.clip, which does not convert NaN to a finite default. h[5], base score, and adjusted score can remain NaN.
4. Flask’s JSON path may serialize non-standard NaN or fail depending on configuration/client, and the UI toFixed call can display NaN or throw downstream.
5. compare_to_reference at stage1/app.py:120–140 checks numeric conversion but not finite/std-zero conditions; NaN can enter z, percentile, and bands.

Required invariant: every public metric and response number must be finite before JSON serialization; non-finite stage output must produce an explicit error/degradation record, not a score.

## 11.2 Function-level degeneracies

| Function | Degenerate behavior |
|---|---|
| _rgb2lab | Numerically defined but wrong units; white L*≈8.99. |
| _entropy | Inferred N=1 produces bins=0 because the guard is only in an elif for an explicitly passed nbins. stage1/visual_complexity.py:454–459. |
| shannon_entropy | hist.sum=0 on empty input; normal decoded images non-empty. |
| layout_symmetry | Uniform image returns 1.0 intentionally at lines 883–885. Defined, but construct choice is debatable. |
| visual_hierarchy | No edges returns fg=0; a single binary blob returns size_gradient=1. Defined but can call blank/uniform layout hierarchical. |
| interactive_element_density | Divides by image area; invalid zero-area arrays fail. |
| saliency normalization | Negative-only map is not normalized; all-zero map returns zeros. |
| saliency intensity entropy | Uniform map yields entropy 0, contradicting its docstring claim “High entropy → uniform/spread distribution.” It measures the diversity of intensity values, not spatial probability entropy. |
| element detector | Assumes BGR in cvtColor; function-level grayscale input fails, though cv2.imread routes force BGR. |
| k-means | Non-deterministic but k≥1 guarded. |
| search model | n_simulations≤0 gives means of empty lists; max_fixations censoring unlabeled. |
| screen consistency | Empty lower-level set raises; endpoint prevents it. Empty element sets are treated as perfectly consistent. |
| product learning | Skips empty screens and returns zero if all screens empty, without identifying detector failure. |

## 11.3 Transparent and one-channel images

cv2.imread/imdecode with color mode discards alpha and expands standard grayscale to BGR in the route paths, so a fully transparent PNG is analyzed by its stored RGB channels rather than transparency/compositing semantics. The system does not specify a background for alpha, so visually identical transparent assets can yield arbitrary metrics based on hidden RGB.

Direct metric functions inconsistently accept 2-D arrays: visual functions often convert grayscale, while detect_elements assumes three-channel BGR.

# Task 12 — Performance and resource envelope

## 12.1 Main headline cost

A cache miss can perform:

1. full upload buffering and SHA-256;
2. eight visual features, including Gaussian/steerable pyramids and FFT work;
3. TensorFlow UMSI++ inference and full-resolution output resize;
4. full-resolution overlay encoding;
5. contour detection and OpenCV k-means per element;
6. optional EasyOCR;
7. Jokinen simulation for every detected target: up to 80 targets × 100 simulations × up to 50 fixations;
8. O(E²) bottom-up feature activations;
9. reference/coherence/serialization.

Search time is not part of the headline formula, yet it is still computed on every headline request for feedback/coherence. This can dominate latency without improving the displayed score.

## 12.2 Repeated work

Positive:

- UMSI model is a lazy process singleton at stage1/app.py:151–158.
- Visual and saliency results are cached by upload hash.
- The headline route reuses the loaded image, detected elements, and OCR report where possible at lines 1117–1159 and 1229–1235.

Negative:

- /api/scanpath-to-target runs representative scanpath simulations and then a full all-target search for target-load comparison, duplicating costly stochastic work.
- scripts/ueyes_saliency_validation.py:206 constructs UMSIPlus for each image rather than once per run.
- Product learning can run multiple use points across an unbounded number of screens, each with many all-target simulations.
- The main score computes search feedback even when the UI/user does not need it.

## 12.3 Scaling and memory

- Element feature activation and screen matching are quadratic in detected elements. The cap of 80 limits a pair to roughly 6,400 comparisons, but repeated simulations multiply work.
- Saliency map, overlay, input bytes, decoded image, pyramid arrays, OCR tensors, and cached float maps can coexist.
- _saliency_cache holds 32 full-resolution maps. Count-bounding does not bound bytes.
- Screen-set/GIF endpoints have no screen/frame count or total-pixel limit.
- Product learning simulations scale with screens × use points × targets × simulations × fixations.
- There is no request timeout, queue, concurrency guard, or per-stage deadline.

For a user study, add upload byte/pixel/frame limits, a bounded-work request schema, explicit low-cost/full-analysis modes, timing telemetry, and load testing on the actual deployment hardware before collecting data.

# Consolidated findings table

| ID | Layer | Severity | File:line | Finding with evidence quote |
|---|---|---|---|---|
| F01 | bug | blocks scientific validity | stage1/visual_complexity.py:91–107 | XYZ is produced after “/ 255.0” but then divided by “95.047 ... 100.000 ... 108.883”; Lab scale is wrong. |
| F02 | bug | blocks scientific validity | stage1/data/results/feature_norms.json:20 | Norms say “all other features reused unchanged,” so Lab-dependent reference anchors remain corrupted. |
| F03 | bug | blocks scientific validity | saliency/umsi_model.py:321–347 | Local decoder Conv2D layers omit official ReLU/dilation and set “use_bias=False”; “Faithful ... port” is unverified. |
| F04 | quality | degrades result | scripts/download_weights.py:155–164 | Checkpoint integrity is only a size check; no checksum binds it to UEyes. |
| F05 | validity | blocks scientific validity | hceye/hceye_features.py:201–205 | Screenshot-to-HCEye mapping is a “purpose-built ... heuristic” and “not empirically calibrated.” |
| F06 | validity | blocks scientific validity | hceye/hceye_features.py:268–276 | Hand-set .30/.20/.20/.15/.15 blend is relabeled cognitive load without independent criterion validation. |
| F07 | validity | blocks scientific validity | hceye/hceye_features.py:257–275 | Hypothetical “Highlight Effectiveness” reduces present load through “1.0 - hl_effectiveness.” |
| F08 | honesty | blocks scientific validity | stage1/ui/index.html:3918 | UI calls the heuristic “Overall estimated mental effort.” |
| F09 | honesty | blocks scientific validity | stage1/ui/index.html:4518–4524 | Arbitrary bands become “notable attentional strain” and “Critical cognitive overload risk.” |
| F10 | honesty | blocks scientific validity | scripts/generate_expose_docx.py:293–296 | Exposé calls coefficients “HCEye-calibrated” although the code declares no calibration. |
| F11 | honesty | blocks scientific validity | scripts/generate_expose_docx.py:190–197 | HCEye is described as automotive and as a peripheral saliency filter; neither matches source/code. |
| F12 | honesty | blocks scientific validity | scripts/generate_expose_docx.py:97,138–139 | Thesis claims AIM integration but repository implements no AIM integration. |
| F13 | honesty | blocks scientific validity | scripts/generate_expose_docx.py:162–178,413–414 | Contribution invokes POMDP/RL optimal control while deployed code is rule-based novice WTA. |
| F14 | validity | blocks scientific validity | cognitive/jokinen_model.py:56–67 | Search has “no top-down (goal-directed) guidance yet.” |
| F15 | bug | degrades result | cognitive/jokinen_model.py:1057–1115 | Doc formula divides by distance, implementation uses “/ np.sqrt(d_ij)”. |
| F16 | bug | degrades result | cognitive/jokinen_model.py:983–986 | Max-fixation failures return ordinary times with no censoring/success flag. |
| F17 | validity | degrades result | cognitive/jokinen_model.py:137–151 | Contrast penalty W_contrast=3 is explicitly “NEEDS CALIBRATION” and can dominate activation. |
| F18 | validity | degrades result | cognitive/element_detector.py:311–358 | Otsu group means are called foreground/background WCAG contrast without semantic adjacency. |
| F19 | honesty | degrades result | stage1/app.py:1295–1323 | Universal 3:1 detector-box report is labeled WCAG/ISO accessibility/legibility. |
| F20 | validity | degrades result | cognitive/element_detector.py:152–177 | Angular diagonal divides by average x/y pixels-per-cm, distorting non-square physical geometry. |
| F21 | bug | degrades result | cognitive/jokinen_model.py:337–340 | Search degree conversion uses screen width only; screen height parameter does not affect y distances. |
| F22 | validity | degrades result | stage1/visual_complexity.py:847–894 | Custom grayscale NCC is cited as interface-aesthetics symmetry. |
| F23 | validity | degrades result | stage1/visual_complexity.py:980–1065 | Custom Canny/Otsu hierarchy is presented with Tuch pedigree. |
| F24 | validity | degrades result | stage1/visual_complexity.py:1072–1141 | “Interactive” density is contour count per pixel area, not semantic controls. |
| F25 | bug | degrades result | stage1/visual_complexity.py:454–459 | Single-sample inferred entropy requests zero histogram bins. |
| F26 | validity | degrades result | saliency/saliency_features.py:211–247 | “Saliency entropy” is intensity-histogram entropy; a uniform spatial map returns zero, opposite its stated interpretation. |
| F27 | bug | degrades result | saliency/umsi_model.py:463–468 | Negative model outputs are not min-shifted/clipped before downstream use. |
| F28 | bug | degrades result | hceye/hceye_features.py:305–331 | Percentile normalization has no finite-value guard; NaN can reach headline. |
| F29 | honesty | degrades result | stage1/app.py:1078–1086 | Saliency failure serves a feature-only headline; only server stdout exposes degradation. |
| F30 | honesty | blocks scientific validity | stage1/app.py:1193–1197 | “Search efficiency” is 1−h[3] and “attention demand” is h[5]+.15, not independent validated outputs. |
| F31 | honesty | degrades result | stage1/ui/index.html:2478–2483 | UI says headline is saliency plus eye-movement model although search output does not enter it. |
| F32 | bug | degrades result | stage2/user_profile.py:39,82–110 | Neutral profile says “no CL adjustment” but formula returns −0.5. |
| F33 | bug | degrades result | stage2/task_descriptor.py:73–116 | Default task labeled neutral returns +1.1; combined defaults shift score +0.6. |
| F34 | validity | blocks scientific validity | stage2/regression_model.py:339–362 | Current training X contains direct transforms of y and fabricated v/s; all metrics are tautological. |
| F35 | bug | degrades result | stage2/regression_model.py:198–205,240–248 | Scaler is fit before split and reused across CV, leaking fold information. |
| F36 | validity | blocks scientific validity | scripts/baseline_comparison.py:101–108 | Raw entropy×25 makes Baseline A saturate near 100. |
| F37 | bug | blocks scientific validity | scripts/baseline_comparison.py:247–260 | “Full” baseline omits saliency and coherence checks zero rules. |
| F38 | validity | blocks scientific validity | scripts/feature_ablation_shap.py:25–40 | Script itself admits target leakage makes R²/SHAP/ablation near-tautological. |
| F39 | validity | blocks scientific validity | scripts/ueyes_saliency_validation.py:12–16 | NSS interpretation is mathematically wrong. |
| F40 | validity | degrades result | scripts/ueyes_saliency_validation.py:263–277 | Script acknowledges density-map NSS is uninterpretable but still reports it. |
| F41 | validity | blocks scientific validity | scripts/hceye_condition_comparison.py:124–165 | Claims participant/image analysis but aggregates/tests image means only. |
| F42 | validity | degrades result | scripts/hceye_condition_comparison.py:80–89 | Uses independent pooled Cohen’s d for paired observations. |
| F43 | bug | blocks scientific validity | scripts/hceye_condition_comparison.py:286–295 | Significant positive correlation is still printed “consistent with H1,” despite expected negative direction. |
| F44 | bug | blocks scientific validity | hceye/gaze/data_processing_pipeline.py:150–171 | Partial-eye NaNs can leave actualX/actualY unbound or stale. |
| F45 | bug | blocks scientific validity | hceye/gaze/data_processing_pipeline.py:305–314 | Velocity divides by time and sampling frequency, introducing an extra factor. |
| F46 | bug | blocks scientific validity | hceye/gaze/data_processing_pipeline.py:115–127 | Fixation boundary sample is discarded instead of starting the next fixation. |
| F47 | bug | blocks scientific validity | requirements.txt:5,10 | NumPy 1.26.4 conflicts with OpenCV 4.13.0.92’s NumPy 2 requirement. |
| F48 | quality | degrades result | tests/test_pipeline.py:20–84 | Nominal test has required positional argument and no assertions; it is a demo. |
| F49 | quality | degrades result | cognitive/test_jokinen.py:104–112 | Saliency errors are caught, then script still prints “SUCCESS.” |
| F50 | quality | degrades result | stage1/build_feature_norms.py:209–212,307–309 | Missing/all-failed norm runs return success instead of nonzero failure. |
| F51 | bug | blocks scientific validity | stage2/screen_consistency.py:82–84,108–138 | Occupancy claims box union but sums overlaps and clips. |
| F52 | bug | blocks scientific validity | stage2/screen_consistency.py:323–330 | One accepted match overrides occupancy/unmatched replacements in headline consistency. |
| F53 | validity | degrades result | cognitive/jokinen_model.py:539–553 | Product transfer interpolation is explicitly an unvalidated modeling choice. |
| F54 | honesty | blocks scientific validity | cognitive/jokinen_model.py:1251–1276 | Synthetic packed fixations are compared to NHTSA/ISO limits as if they were glances. |
| F55 | honesty | blocks scientific validity | stage1/ui/index.html:3012–3032 | UI labels synthetic glance result “within limits.” |
| F56 | honesty | degrades result | stage1/ui/index.html:2975–2981 | Monte Carlo variation is presented broadly as “model uncertainty.” |
| F57 | bug | degrades result | stage1/app.py:1435–1473 | Animated GIF is one frame under images but all frames under image. |
| F58 | bug | degrades result | stage1/app.py:161–169 | Entire upload is buffered; no byte/pixel/frame limit exists. |
| F59 | bug | degrades result | stage1/ui/index.html:4828–4845 | Attacker-controlled filename is interpolated into innerHTML, enabling DOM injection. |
| F60 | quality | degrades result | stage1/app.py:1710 | Flask launches with debug=True. |
| F61 | honesty | degrades result | stage1/ui/index.html:4487–4491 | UI says incoherent prediction “is still valid” and invokes nonexistent calibration range. |
| F62 | honesty | blocks scientific validity | stage1/ui/index.html:2089–2096 | Validation statistics are hardcoded UI text, not API/versioned result artifacts. |
| F63 | validity | blocks scientific validity | stage1/app.py:1050–1073 | desktop_ui is treated as evidence of automotive in-domain use. |
| F64 | quality | cosmetic | scripts/generate_expose_docx.py:496 | UEyes DOI is mistyped as 3581113 instead of 3581096. |

# Top issues ranked by impact on scientific validity

## 1. No identified or independently validated cognitive-load construct

Why first: even perfectly implemented image metrics cannot establish that the headline estimates cognitive load. The code maps HCEye condition effects backward—from load-caused gaze changes to screenshot-caused load—through uncalibrated rules, then gives the result mental-effort and overload labels.

Evidence that would confirm a fix:

- A preregistered construct definition distinguishing interaction complexity, cognitive load, workload, distraction, and search difficulty.
- Independent outcomes not used in feature construction: e.g. trial-level NASA-TLX subscales with appropriate caveats, secondary-task performance, pupil/fixation measures, task completion/error outcomes.
- A frozen formula/model fit only on training participants/screens.
- Participant- and screen-grouped held-out validation with confidence intervals, calibration plots, and comparisons to meaningful baselines.
- Convergent validity with load measures and discriminant validity from pure visual complexity and task difficulty.
- If evidence is insufficient, renamed UI/output everywhere to “exploratory interaction-complexity heuristic.”

## 2. Lab conversion and downstream reference norms are invalid

Why second: the defect corrupts two core image metrics and every percentile-based HCEye use of their committed norms.

Evidence that would confirm a fix:

- Unit tests for black, white, D65 gray, and standard color patches against a trusted implementation such as skimage.color.rgb2lab within stated tolerance.
- Either use white points .95047/1/.1.08883 with XYZ in 0–1, or scale XYZ to 0–100 before current divisors.
- Regenerate every Lab-dependent per-image feature and all norms from a pinned dataset manifest.
- Commit generation command, dependency lock, per-image CSV or durable hash/provenance, and before/after distribution audit.
- Golden tests that a fixed screenshot’s congestion/subband values and final score are stable.

## 3. UMSI++ architecture/checkpoint equivalence is unproven and likely false

Why third: saliency affects the headline, search, UI overlay, classification, norms, and claimed validation. The local decoder materially differs from official code.

Evidence that would confirm a fix:

- Exact upstream commit and checkpoint SHA-256.
- Automated layer-by-layer comparison including names, shapes, bias, dilation, activation, skip connections, and output ordering.
- Successful strict load with zero missing/unexpected weights.
- Golden heatmaps/class probabilities versus the official environment on fixed fixtures, with explicit numerical tolerance.
- Nonnegative finite [0,1] postcondition tests.
- Clear separation between published four-type dataset domain and any auxiliary classifier categories.

## 4. Search, glance, and automotive claims exceed the implemented model/domain

Why fourth: the default search is not goal-guided, geometry is incomplete, failures are censored, and “glance compliance” is manufactured from a partition rule. These are central to the automotive framing.

Evidence that would confirm a fix:

- Implemented and tested target-feature/top-down guidance, or relabeling as bottom-up discovery order.
- Held-out target search-time and scanpath validation with success/censoring modeled explicitly.
- Correct two-axis physical geometry and calibrated display metadata.
- Automotive data from actual tasks/screens/participants, not desktop-class predictions.
- Glance metrics derived from measured or validated time-sharing behavior under a protocol aligned with the relevant standard; remove “compliant/within limits” until then.

## 5. Evidence and reproducibility pipeline cannot support reported results

Why fifth: the environment does not resolve, CI is absent, tests do not assert behavior, data are missing, regression/ablation are circular, and HCEye analysis uses the wrong unit/effect-size design.

Evidence that would confirm a fix:

- Satisfiable lock file and clean installation in CI on supported Python/platforms.
- Unit, property, golden, API, and end-to-end tests with finite-value and default-invariance assertions.
- Versioned data manifests, licenses/access instructions, exact sample sizes/exclusions, and result artifacts tied to commit/checkpoint hashes.
- Participant/screen-aware analysis plan, paired/mixed effects, multiplicity control, effect intervals, and independent test data.
- Removal or hard failure of invalid scaffold analyses; no canned validation values in UI.

# Regression watchlist

Re-check these invariants on every audit/release:

1. White converts to L*=100 within tolerance; black to 0; Lab matches trusted fixtures.
2. All Lab-dependent reference norms were regenerated by the current code and exact dependency/data/checkpoint versions.
3. UMSI layer graph matches the declared upstream commit; exact checkpoint hash loads strictly; golden output parity passes.
4. Every saliency map is finite, nonnegative, and within [0,1] before feature extraction/rendering.
5. A saliency failure is explicit in API and UI; no degraded score is presented as the full model.
6. The HCEye named-field mapping remains semantically explicit; no positional reordering.
7. The default headline formula is documented exactly, and every user-facing label matches the construct actually calculated.
8. Default task + profile adjustment equals exactly zero if called neutral.
9. Feature vectors are consumed in the exact order declared by the trained model artifact, with schema/version checks.
10. No feature equal to or derived from the target is present in training X.
11. All training/validation transforms fit inside folds; participants and screens are grouped appropriately.
12. Search target identity changes the policy through a valid top-down mechanism, not only the stopping condition.
13. Failed/censored searches carry an explicit success flag and are analyzed as censored/failures.
14. n_simulations has a positive lower and safe upper bound on every endpoint.
15. screen_width_cm and screen_height_cm both affect physical geometry correctly.
16. Learning at exposure 1 is byte/numerically equal to the novice baseline under the same RNG stream.
17. More exposure never worsens the reported central tendency beyond defined stochastic tolerance; diminishing-return claims are tested rather than assumed.
18. Screen consistency penalizes unmatched additions/removals; one stable match cannot yield perfect consistency after wholesale replacement.
19. Occupancy is a true rectangle union, not sum-and-clip.
20. Product learning handles empty screens explicitly and states the effective screen denominator.
21. The same GIF has the same frame semantics under image and images, with frame/total-pixel limits.
22. Contrast reports never claim WCAG/ISO conformance without semantic foreground/background and correct criterion selection.
23. Glance output is never labeled compliance unless produced by a validated standards-aligned protocol.
24. Every public numeric field is finite and within its declared range for uniform, single-color, 1×1, no-edge, no-element, negative-map, and huge-but-allowed fixtures.
25. No endpoint leaves shared model parameters/RNG mutated, including exceptions.
26. OpenCV/TensorFlow randomness is controlled or declared; identical input/settings give reproducible output.
27. UI main numbers equal the intended API fields at the same scale and units.
28. UI warnings preserve backend degradation/domain/calibration caveats.
29. User-controlled filenames/text are inserted with textContent or contextual escaping, never raw innerHTML.
30. Upload byte, pixel, frame, element, simulation, and request-time budgets are enforced before heavy computation.
31. Cache limits are byte-aware and concurrency-safe.
32. README route/entry-point inventory includes product learning and matches actual defaults.
33. No hardcoded validation statistic appears without a versioned result artifact and provenance link.
34. Dependency installation and tests pass in CI from a clean checkout.
35. The repository contains or precisely identifies every data/weight artifact needed to regenerate each thesis figure/table.

# Primary-source references used for comparison

- Official UEyes repository and UMSI source: https://github.com/YueJiang-nj/UEyes-CHI2023
- Official UMSI/classification architecture file: https://github.com/YueJiang-nj/UEyes-CHI2023/blob/main/saliency_models/UMSI%2B%2B/src/classif_capable_models.py
- UEyes paper: https://dl.acm.org/doi/10.1145/3544548.3581096
- HCEye paper: https://dl.acm.org/doi/10.1145/3655610
- HCEye open version: https://arxiv.org/html/2404.14232v3
- Jokinen et al. record: https://research.aalto.fi/en/publications/adaptive-feature-guidance-modelling-visual-search-with-graphical-
- Jokinen et al. DOI: https://doi.org/10.1016/j.ijhcs.2019.102376
- WCAG contrast minimum: https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
- WCAG non-text contrast: https://www.w3.org/WAI/WCAG21/Understanding/non-text-contrast.html
- NHTSA visual-manual guidelines PDF: https://www.nhtsa.gov/sites/nhtsa.gov/files/distraction_npfg-02162012.pdf
- Federal Register NHTSA guideline notice: https://www.federalregister.gov/documents/2013/04/26/2013-09883/visual-manual-nhtsa-driver-distraction-guidelines-for-in-vehicle-electronic-devices
- Rosenholtz et al., Measuring Visual Clutter: https://www.mit.edu/~yzli/clutter.pdf
- AIM repository: https://github.com/aalto-ui/aim

## Final disposition

Current status: exploratory research prototype; not a validated cognitive-load or automotive-safety estimator.

Permissible near-term use:

- descriptive screenshot metrics with metric-specific caveats;
- engineering demonstrations;
- hypothesis generation;
- prospective validation planning;
- a relabeled exploratory interaction-complexity index.

Not permissible as scientific evidence in the current state:

- accuracy/validity claims for the 0–100 cognitive-load score;
- automotive generalization or compliance;
- regression R²/SHAP/ablation from the scaffold data;
- HCEye “validation” of the screenshot-to-load mapping;
- same-domain UEyes sample results as automotive transfer evidence;
- hardcoded UI validation statistics;
- claims of AIM integration, POMDP/RL implementation, or HCEye-driven spatial saliency filtering.
