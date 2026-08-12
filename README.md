# Stage-1 GUI Layout Analysis Pipeline

> Task-independent screenshot analysis with an exploratory, project-specific
> layout-complexity index and explicit claim boundaries.

This research pipeline analyzes a GUI screenshot and returns an **exploratory,
project-specific layout-complexity index** together with visual, saliency, and
model-based search diagnostics. It does **not** provide a validated
cognitive-load measurement or observed user-performance data.

The repository accompanies an evolving master's-thesis project. The final
research domain, post-study machine-learning path, and final thesis claims are
still open decisions; historical exposé drafts must not be treated as the
current method specification.

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
the screenshot only. Task and user-profile selections are kept outside x19 and
cannot change that layout value. They may change only separately labelled
`context_adjusted_experimental_outputs`.

The route name `/api/cognitive-load` is retained for compatibility. Its
successful response explicitly reports that the construct is an exploratory
project-specific layout proxy and **not a validated cognitive-load
measurement**.

---

## Outputs

- **Experimental Layout-Complexity Index (0–100):** task-independent Stage-1
  screenshot heuristic, exposed under `layout.experimental_complexity_index`.
- **Context-adjusted experimental outputs:** separate, unvalidated task/profile
  simulations. They are not part of x19 and are not layout measurements.
- **Model-estimated saliency:** a UMSI++ heatmap and five numeric descriptors.
  The auxiliary six-value head remains numeric because its semantic class order
  has not been verified.
- **Model-simulated search diagnostics:** Jokinen-based search times, fixation
  counts, and target paths. These are simulations, not eye-tracking
  observations or validated behavioral predictions.
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
- **No trained Stage-2 model ships.** The optional regression path is off by
  default and no trained model file is included.

---

## Project structure

```text
.
├── stage1/            # Canonical screenshot analysis, Flask API, web UI
├── saliency/          # UMSI++ port, postprocessing, numeric s5 extraction
├── hceye/             # Project-specific HCEye-derived proxy rules (h6)
├── cognitive/         # Jokinen-based model simulations and element detection
├── stage2/            # Separate experimental task/profile and regressor scaffolds
├── scripts/           # Reproducible analysis and evidence utilities
├── tests/             # Pytest unit, regression, contract, and smoke tests
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

Download the UMSI++ checkpoint:

```bash
python3 scripts/download_weights.py
```

or place it at:

```text
saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5
```

The checkpoint is not committed to Git. A score-bearing x19 analysis requires
saliency and fails closed when the checkpoint or saliency processing is
unavailable. The standalone visual-only endpoint can run without it.

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
- `/api/screen-consistency`: the same static formats plus animated GIF.
- WebP is not accepted by the production API.

---

## API

| Endpoint | Method | Bounded description |
| --- | --- | --- |
| `/api/analyze` | POST | Visual-only `v8` screenshot features |
| `/api/saliency` | POST | Numeric UMSI++ heatmap features; no class labels |
| `/api/search-time` | POST | Model-estimated per-element search diagnostics |
| `/api/scanpath-to-target` | POST | Model-simulated target-driven path |
| `/api/cognitive-load` | POST | Legacy route name; returns task-independent x19/layout proxy plus separately labelled context outputs |
| `/api/screen-consistency` | POST | Exploratory inter-screen consistency diagnostic |
| `/api/learning-curve` | POST | Exploratory novice-to-expert model simulation |
| `/api/features` | GET | Metadata for the eight visual features |

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

Repository-root files such as `saliency/test_full_pipeline.py` and the upstream
HCEye `dynamic_test.py` are manual/heavy integration scripts, not part of the
lightweight CI collection. Additional heavy/evidence tests must be invoked by
their explicit path so they run in the required isolated process. The
Step-2b/P9 corrigendum evaluator, for example, must run alone before
TensorFlow/Keras is imported. `tests/test_pipeline.py` likewise remains a direct
end-to-end demo that accepts an image path and is therefore not collected as an
ordinary zero-argument pytest test:

```bash
python3 tests/test_pipeline.py
```

---

## Citation and license

The final thesis citation is not yet frozen. The saliency component builds on
Jiang et al. (2023), *UEyes: Understanding Visual Saliency across User Interface
Types*, CHI 2023.

This is academic research code. A license will be added before a public
open-source release; contact the author before reuse in the meantime.
