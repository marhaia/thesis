You are conducting a rigorous technical and methodological audit of a GitHub research repository. Do not summarize the README uncritically; verify claims against the actual code. Treat the README as one source among several, not as ground truth, since it may be outdated relative to the current state of the codebase.

<repository_context>
Name: Two-Stage Interactional Complexity Pipeline
Purpose: computational estimation of interactional complexity and cognitive load from automotive GUI screenshots, accompanying the master's thesis "Two-Stage Multi-Output Pipeline: Computational Estimation of Interactional Complexity from GUI Screenshots."
Claimed components:
- Visual complexity vector v ∈ ℝ⁸ (entropy, edge density, feature congestion, layout symmetry, color coherence, visual hierarchy, plus two more metrics)
- Saliency features s ∈ ℝ⁵ from UMSI++, trained on UEyes (CHI 2023)
- Visual search time per UI element via the Jokinen (2020) cognitive model
- A cognitive-load estimate combining the above with optional task description and user profile
- A Flask web app for screenshot upload and inspection
</repository_context>

<task_0_pre_audit_inventory>
Before comparing the repository to its README, build an independent inventory of the codebase:
- List all modules, scripts, and entry points (including CLI tools, notebooks, config files, and Flask routes) regardless of whether they are mentioned in the README.
- For each, infer its function from the code itself (imports, function signatures, docstrings, commit messages if available via git log).
- If git history is accessible, identify commits made after the README's last modification date, to isolate documentation drift from the rest of the codebase rather than treating all code as undifferentiated.
- Produce a two-way diff:
  (a) Claimed in README but not found in code, or found in a materially different form.
  (b) Present in code but absent from README, unmentioned, or only vaguely gestured at.
- For (b), classify each undocumented item as:
  (i) methodologically load-bearing (affects how the 8 metrics, saliency stage, search-time model, or cognitive-load estimate are computed),
  (ii) alternative/experimental path (e.g., a second model variant, an ablation, a different combination rule) not used in the "official" pipeline,
  (iii) incidental tooling (logging, visualization, dataset preprocessing scripts) with no bearing on the scientific claims.
Only after this inventory, proceed to the sections below, folding any category (i) findings into the relevant methodological section rather than treating them as a side note.
</task_0_pre_audit_inventory>

<task>
1. Architecture and code quality
   - Map the pipeline stages to the actual modules/files, using the inventory from task 0 rather than the README's own description. Note any mismatch between the two-stage description (visual complexity + saliency → search time → cognitive load) and how the code is actually organized.
   - Assess modularity, dependency management, error handling, and whether the Flask app separates inference logic from presentation.
   - Check reproducibility: pinned dependencies, seed control for the saliency model, documented environment (Python version, CUDA/CPU requirements), presence of tests.

2. Construct validity of the 8 visual complexity metrics
   - For each metric, identify the formula implemented in code and compare it against its canonical definition in the visual complexity literature (e.g., entropy as Shannon entropy of a luminance or edge-orientation histogram; feature congestion per Rosenholtz et al.; layout symmetry and color coherence as used in Reinecke & Gajos or Miniukovich & De Angeli).
   - Flag any metric where the implementation diverges from its cited source, or where the source is unclear/unstated.
   - If task 0 revealed an alternative or newer implementation of any metric not reflected in the README, evaluate that version instead of (or alongside) the documented one.

3. Saliency stage
   - Verify how UMSI++ is loaded and invoked, whether pretrained weights match the UEyes-trained checkpoint claimed, and whether preprocessing (resizing, normalization) matches the original UMSI/UEyes pipeline.
   - Note any silent fallback behavior (e.g., if weights are missing) that would misrepresent output as model-derived.

4. Cognitive search-time and cognitive-load stages
   - Compare the implementation of the visual search time estimation against the equations in Jokinen (2020). Identify parameters that are hardcoded versus derived from the image.
   - Assess how the final cognitive-load estimate combines the complexity vector, saliency features, and search time: is the combination rule justified (theoretically or empirically), and how is the optional task description/user profile actually used, if at all?

5. Correspondence with the thesis
   - Where the thesis title or abstract makes a specific methodological claim, check whether the repository substantiates it or whether the implementation is a simplification.
   - Note explicitly if undocumented functionality (from task 0) contradicts or supersedes a claim made in the thesis.

6. Critique
   - Distinguish, explicitly, between issues that are empirical/implementation bugs (verifiable in code) and issues that are conceptual/methodological (the validity of combining these constructs at all, independent of correct implementation).
   - State open questions you cannot resolve from the code alone (e.g., missing validation data, missing citations, undocumented functionality whose purpose remains unclear even after inspection).
</task>

<output_format>
Structure the report as: task 0 inventory and diff, followed by the six sections above. Under each, give specific file/line references where possible. End with a short list of the three highest-priority issues, ranked by whether they affect the scientific validity of the pipeline's output versus mere code quality or documentation staleness.
</output_format>
