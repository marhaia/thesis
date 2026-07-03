# Exposé (Standard Version) — ~4 Pages

**Working Title:** Integrating Cognitive Predictive Metrics into the AIM Platform for Automated GUI Evaluation  
**Author:** Hannah  
**Programme:** User Experience Design  
**Supervisors:** Gerhard Graf and Prof. Dr. Christian Sturm  
**Date:** June 2026

---

## 1. Introduction and Problem Statement

Automated evaluation of graphical user interfaces (GUIs) is a central challenge in HCI research: designers need scalable, reproducible quality indicators without having to conduct a costly user study for every design decision. This is particularly critical in safety-relevant contexts such as automotive HMI, where a poorly designed dashboard can increase cognitive load and compromise driver safety. The Aalto Interface Metrics (AIM) platform provides an established approach that computes perception-based metrics such as visual clutter, salience, and symmetry from static screenshots (Oulasvirta et al., 2018). These metrics are reproducible and objective — but they capture only the physical properties of the image, not the interplay between interface structure, usage context, and the user's cognitive capacity. Das et al. (2024) show empirically that cognitive dual-task load (e.g., simultaneously driving and reading a dashboard) fundamentally alters gaze behaviour: users exhibit pronounced tunnel vision, miss peripheral UI elements, and form shorter, less efficient scanpaths. AIM cannot predict this behaviourally relevant phenomenon because it contains neither task context nor a cognitive model. Jiang et al. (2023) further demonstrate that UI-specific attention patterns across interface types are systematically underestimated by task-agnostic saliency models — and propose UMSI++, an improved CNN-based saliency model trained specifically on diverse UI types, which the present pipeline adopts as its spatial attention module.

The present master's thesis therefore asks: **Does the integration of cognitive predictive metrics into the AIM platform improve the computational evaluation of GUIs?**

---

## 2. Related Work and Theoretical Framework

### 2.1 AIM and Computational GUI Evaluation

AIM (Oulasvirta et al., 2018) is an open-source platform that computes a range of perception-based metrics from GUI screenshots. Related tools such as UiLab (Burny & Vanderdonckt, 2021) and Criticmate (Ko et al., 2026) extend the evaluation framework with consistency checks and multi-stage critique processes. UIClip (Wu et al., 2024) demonstrates that data-driven CLIP models can learn GUI quality judgements — but provide no mechanistic explanatory framework and no behavioural prediction. A systematic review of computational models specifically for automotive HMI design (Lorenz et al., 2024) finds that none of the 34 surveyed approaches operates on generic screenshots, and only one accounts for individual user characteristics — precisely the two gaps the present pipeline addresses.

### 2.2 Computational Rationality as Theoretical Foundation

The Computational Rationality (CR) framework by Oulasvirta, Jokinen, and Howes (2022) replaces rule-based engineering models (such as GOMS or ACT-R, which describe interaction as fixed rule sequences) with sequential decision theory: interaction is modelled as an optimal control policy that balances cognitive, perceptual, and motor constraints. Formalised as a Partially Observable Markov Decision Process (POMDP) and solved via Reinforcement Learning, CR explains the transition from novice-driven, bottom-up search to expert-guided, top-down memory retrieval (Jokinen et al., 2020). Current CR applications include gaze-based selection (Chen et al., 2021), touchscreen typing (Shi et al., 2024), task-conditioned scanpaths (Shi et al., 2025), attention switching in dual-task scenarios (Lingler et al., 2024), adaptive UI via multi-agent RL (Langerak et al., 2024), driver adaptation to assistance systems (Jokinen & Kujala, 2021), and dual-task attention allocation on head-mounted displays (Bai et al., 2024) — the latter two directly establishing the automotive dual-task context as a CR-tractable problem.

### 2.3 Saliency, Scanpaths, and Cognitive Load

Scanpath prediction has evolved from task-agnostic saliency models (DeepGaze II, Kümmerer et al., 2017) towards task-conditioned approaches: SeekUI (Guo et al., 2026) and Chartist (Shi et al., 2025) show that task-specific conditioning significantly improves prediction quality. EyeFormer (Jiang et al., 2024) demonstrates the feasibility of personalised scanpaths through few-shot conditioning. That cognitive load measurably reshapes scanpaths is empirically established by Krejtz et al. (2018), who show that fixation-based pupillometry and microsaccade magnitude reliably discriminate task difficulty (η²≈0.16–0.17) — validating eye-tracking as a cognitive load measure for both the HCEye dataset and the planned user study. Das et al. (2024) provide the most direct empirical precedent for the pipeline's target domain: in automotive dual-task conditions, cognitive load produces tunnel vision, shorter fixations, and missed peripheral elements — the exact behavioural pattern Stage 2 is designed to estimate. Cognitive load itself is most commonly operationalised via the NASA Task Load Index (Hart & Staveland, 1988), a multi-dimensional subjective workload measure that serves as the validation target for the Cognitive Load Index output of Stage 2.

### 2.4 Research Gap

No existing work combines: (a) task-independent complexity extraction as an interpretable input layer, (b) task-conditioned behavioural prediction with (c) an explicit post-prediction cross-validation layer to check the behavioural coherence between structural, attentional, and cognitive load outputs — integrated into an established GUI evaluation platform (AIM).

---

## 3. Research Design and Methodology

### 3.1 Architecture: Two-Stage Pipeline

The pipeline converts a GUI screenshot into three behavioural estimates. Figure 1 shows the overall data flow. A screenshot is processed by two parallel modules — a task-agnostic visual complexity extractor (Stage 1) and a deep saliency model (UMSI++) — whose outputs are combined into an extended feature vector. This vector is then passed to Stage 2, which uses task context and an optional user profile to predict three interaction quality indicators. In the current training-free implementation, Stage 2 predictions are derived from HCEye-calibrated empirical coefficients; once validation study data are available, a trained regression model replaces this component. The decoupled architecture ensures that Stage 1 outputs are reusable across task contexts and that each module can be updated independently.

**Stage 1 — Task-Independent Visual Complexity Extraction**

Stage 1 computes eight perceptual features from a raw screenshot using established image-analysis models from the GUI evaluation and visual perception literature; the majority are implemented within the open-source AIM framework (Oulasvirta et al., 2018). Output: visual complexity vector $v \in \mathbb{R}^8$.

| Feature | Description | Source |
|---------|-------------|--------|
| Shannon Entropy | Information density (greyscale histogram) | Shannon (1948) |
| Edge Density | Proportion of edge pixels via Canny | Standard image processing |
| Feature Congestion | Multi-channel clutter: colour, contrast, orientation | Rosenholtz et al. (2007) |
| Subband Entropy | Redundancy via steerable pyramid decomposition | Rosenholtz et al. (2007) |
| Layout Symmetry | Vertical/horizontal balance of elements | Miniukovich & De Angeli (2015) |
| Chromatic Coherence | Colour palette variance | Hasler & Süsstrunk (2003) |
| Visual Hierarchy | Size gradient, contrast, grouping structure | Tuch et al. (2009) |
| Element Density | Number of interactive controls per area | AIM framework |

**Saliency Module (UMSI++) — Spatial Attention Prediction**

In parallel, the original screenshot is passed through the UMSI++ deep saliency model (Jiang et al., 2023; CNN-based architecture), which produces a 2D saliency map. Five scalar features are extracted from this map ($s \in \mathbb{R}^5$: dispersion, peak count, centre bias, entropy, coverage) and appended to $v$. The saliency map is delivered directly by UMSI++ (zero-shot, no retraining); it constitutes the pipeline's spatial attention output and is validated against eye-tracking fixation densities to assess how well the pre-trained model transfers to automotive HMI screenshots. Stage 2 does not retrain or modify UMSI++; it operates on the derived scalar features only.

**Stage 2 — Task-Conditioned Prediction**

Stage 2 converts $x \in \mathbb{R}^{19}$ = [$v \in \mathbb{R}^8$ | $s \in \mathbb{R}^5$ | $h \in \mathbb{R}^6$] + Task Descriptor + User Profile into three interaction quality scores. Note: $h$ is computed directly from the screenshot image using empirical sensitivity coefficients derived from the HCEye dataset — no participant eye-tracking data is required at inference time.

**Current implementation (training-free):** Predictions are derived directly from the HCEye-calibrated sensitivity features $h \in \mathbb{R}^6$ via empirically grounded coefficient mappings (Das et al., 2024). This requires no training data and is fully functional as a screening tool from day one. Task Descriptor and User Profile apply ordinal modifiers on top of the base prediction.

**Planned extension (requires validation study data):** Once empirical ground-truth labels are available from the validation study (Option A or C), the coefficient-based predictions are replaced by a trained multi-output regression model (Ridge Regression as baseline, Random Forest and XGBoost as primary estimators, cross-validated). The model architecture is already implemented and the feature vector $x \in \mathbb{R}^{19}$ serves as its input unchanged.

| Output | Source | Validation Metric |
|--------|--------|-------------------|
| Saliency Map (2D, spatial attention) | UMSI++ — pre-trained, zero-shot | AUC-Judd, NSS, SIM, CC (domain transfer) |
| Search Efficiency + Attention Demand (scalars) | HCEye rule-based (current) / Stage 2 regression (planned) | Correlation with eye-tracking AOI metrics |
| Cognitive Load Index (0–100) | HCEye rule-based (current) / Stage 2 regression (planned) | Correlation with NASA-TLX |

*UMSI++ is applied without retraining. Stage 2 regression is activated only after validation study data are collected.*

**Coherence Check**

After prediction, a rule-based coherence check flags physically inconsistent output combinations. Three rules are implemented, each grounded in empirical literature: (1) high saliency dispersion with implausibly low fixation count (Rosenholtz et al., 2007; Jokinen et al., 2020); (2) concentrated saliency with high cognitive load (Das et al., 2024; Tuch et al., 2009); (3) high predicted search time with low cognitive load index (Hart & Staveland, 1988). Incoherent combinations are flagged as warnings; they do not invalidate the prediction but indicate the input is at the boundary of the model's calibration range. This rule-based consistency layer is the structural novelty of the pipeline: no existing GUI evaluation tool cross-validates its outputs across multiple behavioural dimensions.

**User Profile:** Big Five trait-level presets (Neutral, Focused/Conscientious, Exploratory/Open, Social/Extraverted, Stress-Sensitive) modulate the Stage 2 prediction via ordinal weight adjustments. LLM-based simulation at trait level is methodologically defensible (Serapio-García et al., 2023; Argyle et al., 2023). State-based simulation (acute stress, fatigue) is explicitly declared as non-validated (Kapania et al., 2025).

### 3.2 Empirical Validation

**Data Source 1 — HCEye Dataset (Das et al., 2024; publicly available):**
- N=27, eye-tracking + dual-task cognitive load manipulation, 150 web UI screenshots
- Use: exploratory calibration of Stage 1 feature norms; benchmark for UMSI++ zero-shot saliency transfer validation
- Limitation: web UIs, not automotive HMI — transferability to the target domain is an open question

**Data Source 2 — Planned Empirical Validation:**

> **Open question — supervisor input requested:** The form of empirical validation has not yet been decided. The following options are under consideration; the final choice depends on feasibility, lab access, and timing, which I would like to discuss.
>
> - **Option A — User Study** (N≈15–30, within-subjects, eye-tracking + NASA-TLX, automotive HMI simulator): provides quantitative validation of predictive accuracy and enables the variance decomposition between Stage 1 and Stage 2.
> - **Option B — Expert Interviews** (N≈5–12 automotive UX/HMI practitioners, structured walkthrough protocol): assesses plausibility and practical utility of the pipeline outputs from a design perspective.
> - **Option C — Sequential Mixed Methods** (reduced user study + focused expert panel): combines quantitative prediction validation with qualitative practitioner assessment, at the cost of greater organisational effort.
>
> No decision has been made. I would welcome your guidance on which approach is most appropriate given the timeline and available resources.

**Baseline Comparisons** *(require Option A or C — quantitative data collection):*

| Baseline | Description |
|---------|-------------|
| B1 | Stage 1 alone + linear regression |
| B2 | Stage 2 without coherence check |
| B3 | Single-output model (Cognitive Load Index only) |

**Core Analysis** *(require Option A or C):*
- Variance decomposition: how much does Stage 1 explain alone? What does Stage 2 add?
- Coherence gain: B2 vs. full Stage 2
- Null results will be reported explicitly

If Option B (expert interviews) is chosen, baseline comparisons and quantitative model evaluation are replaced by a structured practitioner assessment of prediction plausibility and design utility.

---

## 4. Expected Contributions

### Academic Contributions

**Theoretical:** Empirical test of the Computational Rationality hypothesis that cognitive load can be predicted as the outcome of an optimal control policy over visual interface properties.

**Empirical** *(contingent on validation Option A or C):* Variance decomposition of task-independent visual complexity (Stage 1) vs. task-conditioned behavioural prediction (Stage 2) in a controlled automotive HMI setting — establishing which layer of the pipeline carries explanatory weight. If Option B (expert interviews) is chosen instead, this contribution is replaced by a qualitative plausibility assessment of the pipeline outputs from a practitioner perspective.

**Methodological:** First integration of a multi-dimensional GUI evaluation pipeline that cross-validates structural, attentional, and cognitive load estimates through a post-prediction, rule-based coherence layer — making divergent model outputs visible as design warnings rather than silently absorbing them into a joint loss.

### Practical Deliverables

**Two-Stage Pipeline:** The pipeline itself is the primary deliverable — functioning as a pre-deployment screening tool for automotive HMI designers. It provides a scalable first estimate of cognitive load risk at the layout stage, without requiring eye-tracking infrastructure or a user study.

**Open-Source AIM Extension:** The implementation is released as an AIM-compatible module, enabling adoption as a web service or design tool integration.

---

## 5. Timeline

> *Implementation will make use of AI-assisted development (large language model coding tools) throughout, in line with current practice in computational HCI research.*

| Phase | Content | Period |
|-------|---------|--------|
| Exposé | Finalise and submit exposé; supervisor meeting | June 2026 |
| Literature | Finalize reading, close remaining gaps | June 2026 |
| Registration | Official thesis registration | July 2026 |
| Implementation (finalization) | Complete remaining 10%, integration testing, AIM compatibility | July 2026 |
| Study (Preparation) | Ethics approval (parallel), recruitment, stimuli, pilot test | July–Aug 2026 |
| Study (Data Collection) | Main study N≈15–30 (form TBD, see Section 3.2) | Aug–Sep 2026 |
| Analysis & Training | Eye-tracking processing, model training, baseline comparisons B1–B3 *(if Option A or C; replaced by expert interview analysis if Option B)* | Sep–Oct 2026 |
| Writing | Thesis chapters 1–7 (begins in parallel with analysis) | Oct–Nov 2026 |
| Submission | — | **End of Dec 2026** |

---

## 6. Scope and Limitations

- Domain restriction: automotive HMI screenshots only; generalisation to other UI types requires separate validation
- Cognitive Load Index ≠ NASA-TLX; the output is a computed interaction complexity indicator
- LLM user simulation: trait-level only (Big Five); state-based simulation (fatigue, acute stress) is not validated and should not be interpreted as such — the User Profile feature is exploratory; core validation rests on the structural and task-based inputs to Stage 2 (Kapania et al., 2025)
- Pilot dataset: single-GUI design without complexity variation; suitable for calibration only

---

## 7. References

- Oulasvirta, A. et al. (2018). Aalto Interface Metrics (AIM). *UIST 2018 Adjunct*.
- Oulasvirta, A. et al. (2022). Computational Rationality as a Theory of Interaction. *ACM TOCHI*.
- Das, A., Wu, Z., Škrjanec, I., & Feit, A. M. (2024). HCEye: Exploring the relationship between cognitive load, saliency, and visual highlighting. *Proceedings of ETRA 2024*. https://doi.org/10.1145/3655610
- Jiang, M. et al. (2023). UEyes: Understanding Visual Saliency across UI Types. *CHI 2023*.
- Jokinen, J. P. P., Sarcar, S., Oulasvirta, A., Silpasuwanchai, C., Wang, Z., & Ren, X. (2020). Adaptive feature guidance: Modelling visual search with graphical layouts. *International Journal of Human-Computer Studies, 136*, 102376. https://doi.org/10.1016/j.ijhcs.2019.102376
- Jokinen, J.P.P. & Kujala, T. (2021). Modelling Drivers' Adaptation to Assistance Systems. *AutomotiveUI 2021*.
- Lorenz, M. et al. (2024). Computational Models for In-Vehicle UI Design. *AutomotiveUI 2024*.
- Krejtz, K. et al. (2018). Eye Tracking Cognitive Load Using Pupil Diameter and Microsaccades. *PLOS ONE*.
- Guo, Z., Jiang, Y., Leiva, L. A., & Oulasvirta, A. (2026). SeekUI: Predicting visual search with reward-augmented vision-language models. *Proceedings of CHI 2026*. https://doi.org/10.1145/3772318.3791178
- Shi, D., Zhu, Y., Jokinen, J. P. P., Acharya, A., Putkonen, A., Zhai, S., & Oulasvirta, A. (2024). CRTypist: Simulating touchscreen typing behavior via computational rationality. *Proceedings of CHI 2024*. https://doi.org/10.1145/3613904.3642918
- Shi, D., Wang, Y., Bai, Y., Bulling, A., & Oulasvirta, A. (2025). Chartist: Task-driven eye movement control for chart reading. *Proceedings of CHI 2025*. https://doi.org/10.1145/3706598.3713285
- Bai, Y. et al. (2024). Heads-Up Multitasker: Simulating Attention Switching on Optical HMDs. *CHI 2024*.
- Langerak, T., Christen, S., Albaba, M., Gebhardt, C., Holz, C., & Hilliges, O. (2024). MARLUI: Multi-agent reinforcement learning for adaptive user interfaces. *ACM Transactions on Computer-Human Interaction*. https://doi.org/10.1145/3661147
- Wu, J., Peng, Y.-H., Li, X. Y., Swearngin, A., Bigham, J. P., & Nichols, J. (2024). UIClip: A data-driven model for assessing user interface design. *Proceedings of UIST 2024*. https://doi.org/10.1145/3654777
- Ko, J., Choi, J., Morales, C., Kim, D., & Ko, M. (2026). Criticmate: Stagewise human–AI co-critique for interface design. *Proceedings of CHI 2026*. https://doi.org/10.1145/3772318.3790929
- Kapania, S., Agnew, W., Eslami, M., Heidari, H., & Fox, S. E. (2025). Simulacrum of stories: Evaluating the proximity of LLM-generated qualitative data to human experiences. *Proceedings of CHI 2025*. https://doi.org/10.1145/3706598
- Hart, S.G. & Staveland, L.E. (1988). Development of NASA-TLX. *Human Mental Workload*.
- Hasler, D. & Süsstrunk, S.E. (2003). Measuring Colourfulness in Natural Images. *SPIE HVEI VIII*.
- Miniukovich, A. & De Angeli, A. (2015). Computation of Interface Aesthetics. *CHI 2015*.
- Rosenholtz, R. et al. (2007). Measuring Visual Clutter. *Journal of Vision*.
- Tuch, A.N. et al. (2009). Visual Complexity of Websites. *CHI 2009*.
- Serapio-García, G. et al. (2023). Personality Traits in LLMs. *arXiv*.
- Argyle, L.P. et al. (2023). Out of One, Many. *Political Analysis*.
- Burny, W. & Vanderdonckt, J. (2021). UiLab: A Research Instrument for UI Evaluation. *EICS 2021*.
- Chen, X. et al. (2021). An Adaptive Model of Gaze-Based Interaction. *CHI 2021*.
- Kümmerer, M. et al. (2017). DeepGaze II: Reading Fixations from Deep Features. *ICCV 2017*.
- Lingler, A., Talypova, D., Jokinen, J. P. P., Oulasvirta, A., & Wintersberger, P. (2024). Supporting task switching with reinforcement learning in user interfaces. *Proceedings of CHI 2024*. https://doi.org/10.1145/3613904.3642063
- Jiang, Y. et al. (2024). EyeFormer: Predicting Personalised Scanpaths. *CHI 2024*.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(3), 379–423.
