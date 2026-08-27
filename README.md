# GUI Layout Analysis Pipeline — Stage 1 + Stage 2 v1

> Task-independent screenshot analysis with an exploratory, project-specific
> layout-complexity index and explicit claim boundaries.

This research pipeline analyzes a GUI screenshot and returns an **exploratory,
project-specific layout-complexity index** together with visual, saliency, and
reproducibility evidence. The study UI deliberately excludes the repository's
separate model-based experimental diagnostics. It does **not** provide a
validated cognitive-load measurement or observed user-performance data.

This repository defines the technical pipeline used by the master's-thesis
project. User-study execution, participant data, and any post-study analysis
are deliberately outside this repository and outside the current technical
release. Historical exposé, literature, and planning artifacts are not the
current method specification.

---

## Release status

Stage 1 is frozen at the annotated Git tag `stage1-technical-freeze-v1`.
Stage 2 v1 remains a release candidate until the exact final repository tree
has passed the complete regression suite and two independent clean-room
technical audits. A Git tag and its bound commit/tree identity are authoritative
for a released version; an untagged checkout must be treated as development
state.

---

## Current Stage-1 contract

For every successful score-bearing analysis, the public Stage-1 boundary is:

```text
x = [v8 | s5 | h6] ∈ R19, dtype float32
```

- `v ∈ R8`: eight screenshot-derived visual-complexity features.
- `s ∈ R5`: five numeric features extracted from a UMSI++ model-estimated
  saliency map.
- `h ∈ R6`: six project-specific HCEye-derived screenshot proxies.

The Stage-1 vector and `layout.experimental_complexity_index` are computed from
the screenshot only. Stage 2 v1 cannot change either value.

The route name `/api/cognitive-load` is retained for compatibility. Its
successful response explicitly reports that the construct is an exploratory
project-specific layout proxy and **not a validated cognitive-load
measurement**.

---

## Current Stage-2 v1 contract

The active v1 context boundary accepts exactly:

- `task_type`: `navigation`, `search`, `monitoring`, `data_entry`, or
  `decision`; category label only.
- `time_pressure`: `low`, `medium`, or `high`; deterministically mapped to
  `lower`, `baseline`, or `higher`.

These public context values are case-sensitive and may be supplied at most
once across form and query data. Explicit blanks and ambiguous duplicates fail
with HTTP 400. The optional `display_preset` accepts only `phone`, `laptop`, or
`desktop` and is valid only when the separate Jokinen diagnostic is requested.

All 15 combinations are declared compatible. The response is
`score_bearing=false`, `numeric_modifier=null`, `simulated_process=false`, and
`validated_measurement=false`. Legacy profile, search-mode, target-specificity,
and trained-model fields are rejected rather than silently interpreted.

The time-pressure field is included as a pre-specified contextual hypothesis.
Primary research shows that time constraints can affect interactive-search
behaviour and that the response depends on the person and task (Liu et al.,
2019, doi:10.1016/j.ipm.2019.04.004). This literature motivates retaining the
context field; it does not calibrate the `lower` / `baseline` / `higher` mapping,
establish a universal monotonic effect, or validate any category-specific
prediction in this repository.

---

## Outputs

- **Experimental Layout-Complexity Index (0–100):** task-independent Stage-1
  screenshot heuristic, exposed under `layout.experimental_complexity_index`.
- **Deterministic Stage-2 v1 scenario proxy:** task type is one of five declared
  scenario labels and time pressure maps to the qualitative direction
  `lower`, `baseline`, or `higher`. It is explicitly non-score-bearing and has
  no numeric modifier, task-process simulation, calibrated effect size, or
  measurement claim.
- **Study-only context metadata:** the browser UI can record task, time pressure,
  and optional study-coded Big Five categories beside an export. Big Five is not
  sent to the score API and cannot modify any numerical output.
- **Model-estimated saliency:** a UMSI++ heatmap and five numeric descriptors.
  The auxiliary six-value head remains numeric because its semantic class order
  has not been verified.
- **Technical API-only Jokinen diagnostic:** methodologically separate and not
  exposed by the study UI. When directly requested through the technical API,
  its model-simulated search outputs are not score-bearing, eye-tracking
  observations, or validated behavioral predictions.
- **Technical API-only Cross-Signal Review:** tri-state review cue
  (`not_evaluable`, `no_review_flag`, `review_recommended`) using
  author-selected, uncalibrated inspection triggers. It never changes a score
  and does not validate one signal against another.
- **Reference comparisons and reproducibility identity:** corpus-relative
  feature comparisons plus source/checkpoint/norm/runtime hashes.

---

## Scientific boundaries

- **HCEye is an input to a project-specific heuristic, not calibration of a
  screenshot-level cognitive-load measure.** Aggregate source-study values,
  repository-pinned derivation rules, and the external CSV hash provide
  provenance; they do not provide independent validation of the final index.
- **UMSI++ is transferred from UEyes.** The model and reference norms originate
  from general UI/image data. The production port has repository evidence and
  fail-closed checkpoint identity, but no semantic class-label mapping is
  asserted.
- **Canonical 1280 is project-specific.** Score-driving screenshot analysis is
  performed at an aspect-ratio-preserving 1280 px long side. This is an
  exploratory, evidence-based engineering parameter selected from declared
  candidates, not a universal or provably optimal resolution.
- **No independent end-to-end human validation has been completed.** Internal
  correspondence, parity, invariance, and sanity checks establish engineering
  behavior and reproducibility, not construct validity.
- **Personality-derived effects and machine learning are excluded from Stage 2
  v1.** No profile input, trained-model selection, numeric profile effect, or ML
  prediction is accepted or returned by the active v1 route. Optional Big Five
  categories in the study UI are browser-side metadata only and require a
  predeclared validated study instrument before substantive interpretation.

---

## Project structure

```text
.
├── stage1/            # Canonical screenshot analysis, Flask API, web UI
├── saliency/          # UMSI++ port, postprocessing, numeric s5 extraction
├── hceye/             # Project-specific HCEye-derived proxy rules (h6)
├── cognitive/         # Jokinen-based model simulations and element detection
├── stage2/            # Stage-2 v1 scenario proxy and cross-signal review
├── scripts/           # Reproducible analysis and evidence utilities
├── tests/             # Pytest unit, regression, contract, and smoke tests
├── REPRODUCIBILITY.md # Frozen runtime, artifacts, and replay procedure
├── AUDIT_TRAIL.md     # Technical-audit scope and release evidence policy
└── requirements.txt
```

The `ueyes/` and `aim/` folders are separate upstream repositories and are not
part of this repository.

---

## Requirements and setup

- Python 3.9 (the repository also records a frozen runtime identity for study
  exports)
- About 2 GB free disk space, primarily for the saliency checkpoint
- macOS or Linux

```bash
git clone https://github.com/marhaia/thesis.git
cd thesis
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the external UMSI++ and EasyOCR model artifacts:

```bash
python3 scripts/download_weights.py
python3 scripts/download_easyocr_models.py
```

Alternatively, place the exact files at:

```text
saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5
cognitive/weights/easyocr/model/craft_mlt_25k.pth
cognitive/weights/easyocr/model/english_g2.pth
```

The checkpoint and EasyOCR model files are not committed to Git. Their exact
filenames, byte sizes and SHA-256 hashes are tracked in repository identity
manifests and are verified before model construction. EasyOCR uses a fixed
repository-relative model directory and automatic downloads are disabled in
the Production reader. A score-bearing x19 analysis requires both saliency and
OCR and fails closed when either exact model identity or its processing is
unavailable. The standalone visual-only endpoint can run without them.

---

## Usage

Start the local web app:

```bash
source venv/bin/activate
python3 stage1/app.py
```

Then open <http://localhost:5001>.

Visual-only command-line analysis:

```bash
python3 stage1/visual_complexity.py --image path/to/screenshot.png
python3 stage1/visual_complexity.py --dir path/to/folder/
```

### Supported image formats

- Single-image analysis routes: PNG, JPG/JPEG, BMP, TIFF.
- `/api/screen-consistency` and `/api/product-learning`: the same static
  formats plus animated GIF.
- WebP is not accepted by the production API.
- The default decoded-image envelope is 8,192 px per dimension, 16 million
  pixels, a 16 px minimum for each dimension, and a 20:1 maximum aspect ratio
  per analysable image; both screen-set endpoints additionally have a
  32-million-pixel / 96-MB cumulative decoded budget.

### Stage-1 acceptance boundary

The accepted basic Stage-1 function is the task-independent screenshot path
that produces the finite `x19 = [v8 | s5 | h6]` contract and the explicitly
exploratory layout-complexity index. Target selection, predicted scanpaths and
multi-target comparison are dormant experimental prototypes: they are not
rendered by the standard Stage-1 UI, do not affect x19 or the layout value, and
are reserved for Future Work rather than Stage-1 acceptance.

---

## API

| Endpoint | Method | Bounded description |
| --- | --- | --- |
| `/api/analyze` | POST | Visual-only `v8` screenshot features |
| `/api/saliency` | POST | Numeric UMSI++ heatmap features; no class labels |
| `/api/search-time` | POST | Model-estimated per-element search diagnostics |
| `/api/scanpath-to-target` | POST | Dormant Future Work prototype; model-simulated target-driven path outside Stage-1 acceptance |
| `/api/cognitive-load` | POST | Legacy route name; returns task-independent x19/layout proxy, non-score-bearing Stage-2 v1 scenario proxy, tri-state cross-signal review, and an optional separate Jokinen diagnostic |
| `/api/screen-consistency` | POST | Exploratory inter-screen consistency diagnostic |
| `/api/product-learning` | POST | Exploratory multi-screen/GIF product-learning simulation outside x19; shares the screen-set cumulative limits |
| `/api/learning-curve` | POST | Exploratory novice-to-expert model simulation |
| `/api/features` | GET | Metadata for the eight visual features |

For `/api/search-time`, `/api/scanpath-to-target`, and `/api/learning-curve`,
saliency is requested by default. A failed requested saliency stage returns a
structured HTTP 503 with `analysis_complete=false` and no score/path/curve.
Feature-only execution remains available only through the explicit
`use_saliency=false` request and identifies itself as
`analysis_mode=feature_only_explicit`; it is never entered as a fallback.

Example:

```bash
curl -F "image=@screenshot.png" http://localhost:5001/api/analyze
```

---

## Testing

The authoritative lightweight suite is pinned in `pytest.ini`; GitHub CI runs
the same collection with `STAGE1_RUNTIME_VERIFICATION=metadata_only`:

```bash
source venv/bin/activate
pytest -q
```

An extracted source archive has no `.git` directory from which the source
commit can be resolved. Bind that archive explicitly to the audited commit
before running the same suite; `exported` records that the files came from an
immutable export rather than a Git working tree:

```bash
export STAGE1_SOURCE_COMMIT=<full-40-character-commit-sha>
export STAGE1_SOURCE_TREE_STATE=exported
STAGE1_RUNTIME_VERIFICATION=metadata_only pytest -q
```

`STAGE1_SOURCE_COMMIT` remains mandatory outside a Git working tree. When the
tree-state variable is omitted for an explicit source commit, the runtime uses
the conservative default `exported`.

Repository-root files such as `saliency/test_full_pipeline.py` and the upstream
HCEye `dynamic_test.py` are manual/heavy integration scripts, not part of the
lightweight CI collection. Additional heavy/evidence tests must be invoked by
their explicit path so they run in the required isolated process. The
Step-2b/P9 corrigendum evaluator, for example, must run alone before
TensorFlow/Keras is imported. `tests/test_pipeline.py` is retained only as a
**historical, incomplete development demo** and is not collected as an ordinary
pytest test. It never computes UMSI++/s5 and instead uses five explicit zero
placeholders; consequently its combined values are **not Stage-1 x19**, its
retired Stage-2 fields are not measurements, and it is not current pipeline or
validation evidence. It can be inspected directly with:

```bash
python3 tests/test_pipeline.py
```

---

## Core methodological references and license

The current technical and scientific claim boundary relies on these primary
sources. Their results motivate individual components; they do not independently
validate the repository's combined layout index or Stage-2 scenario proxy.

- Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B.,
  Kylmälä, J., & Oulasvirta, A. (2023). *UEyes: Understanding Visual Saliency
  across User Interface Types*. CHI 2023.
  https://doi.org/10.1145/3544548.3581096
- Das, A., Wu, Z., Škrjanec, I., & Feit, A. M. (2024). *Shifting Focus with
  HCEye: Exploring the Dynamics of Visual Highlighting and Cognitive Load on
  User Attention and Saliency Prediction*. Proceedings of the ACM on
  Human-Computer Interaction, 8(ETRA), Article 236.
  https://doi.org/10.1145/3655610
- Jokinen, J. P. P., Wang, Z., Sarcar, S., Oulasvirta, A., & Ren, X. (2020).
  *Adaptive feature guidance: Modelling visual search with graphical layouts*.
  International Journal of Human-Computer Studies, 136, 102376.
  https://doi.org/10.1016/j.ijhcs.2019.102376
- Liu, C., Liu, Y.-H., Gedeon, T., Zhao, Y., Wei, Y., & Yang, F. (2019). *The
  effects of perceived chronic pressure and time constraint on information
  search behaviors and experience*. Information Processing & Management,
  56(5), 1667–1679. https://doi.org/10.1016/j.ipm.2019.04.004
- Torralba, A., & Efros, A. A. (2011). *Unbiased look at dataset bias*. CVPR
  2011, 1521–1528. https://doi.org/10.1109/CVPR.2011.5995347

This is academic research code. A license will be added before a public
open-source release; contact the author before reuse in the meantime.
