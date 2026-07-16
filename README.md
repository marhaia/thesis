# Two-Stage Interactional Complexity Pipeline

> Computational estimation of **interactional complexity** and **cognitive load** from automotive GUI screenshots.

A research pipeline that takes a single GUI screenshot and predicts how visually
complex and cognitively demanding it is for a user. It combines classic visual
complexity metrics, deep-learning saliency prediction, and a cognitive visual
search model into one interactive tool.

This is the implementation accompanying the master's thesis
*"Two-Stage Multi-Output Pipeline: Computational Estimation of Interactional
Complexity from GUI Screenshots"*.

---

## What it does

Given a GUI screenshot, the pipeline produces:

- **Visual complexity vector** `v ∈ ℝ⁸` — 8 interpretable metrics (entropy, edge
  density, feature congestion, layout symmetry, color coherence, visual
  hierarchy, …)
- **Saliency features** `s ∈ ℝ⁵` — predicted by a deep saliency model (UMSI++,
  trained on the UEyes CHI2023 eye-tracking dataset)
- **Predicted visual search time** — per UI element, via the Jokinen (2020)
  cognitive search model
- **A cognitive-load estimate** — a 0–100 score derived from the HCEye
  cognitive-load rules (`h ∈ ℝ⁶`), adjusted by optional additive task and
  user-profile modifiers

> **How the score is actually composed.** The deployed load score is
> `100 · h[5] + task_modifier + profile_modifier`. The predicted search time is
> **not** summed into this score; it feeds the coherence check and the
> "bottleneck" element feedback instead. The visual and saliency vectors enter
> the score through the HCEye rules, not as a direct weighted sum.

Everything is exposed through a small **Flask web app**: upload a screenshot in
the browser and inspect the results visually.

---

## Scope & honest limitations

- **Domain transfer, not automotive-native.** The saliency model (UMSI++) and
  the feature-norms reference distribution are trained/computed on **general
  web, mobile and desktop UIs** (UEyes CHI2023), **not** on automotive HMIs.
  Applying the pipeline to automotive screenshots is therefore an
  *out-of-domain transfer* that the thesis sets out to test — it is not a
  car-calibrated measurement.
- **First implementations, not yet validated.** The inter-screen consistency
  metric and the novice→expert learning curve are geometric/first-pass models
  and are **not** validated against user behaviour. Their calibration constants
  are declared placeholders that need an empirical user study.
- **No trained Stage-2 model ships.** The optional regression path is off by
  default and no model file is included (see `stage2/`).

---

## Project structure

```
.
├── stage1/            # Visual complexity extraction + Flask web app
│   ├── app.py             → web server (entry point, port 5001)
│   └── visual_complexity.py
├── stage2/            # Task descriptor, user profile, inter-screen consistency
├── saliency/          # UMSI++ deep saliency model (TensorFlow/Keras)
│   └── weights/           → model weights (~115 MB, NOT in git — see Setup)
├── cognitive/         # Jokinen 2020 visual search model + element detector
├── hceye/             # HCEye-based cognitive-load rules (deployed load score)
├── scripts/           # Analysis & validation scripts
├── tests/             # End-to-end demo script (not a pytest suite — see Testing)
└── requirements.txt
```

> **Note:** The `ueyes/` and `aim/` folders are separate upstream repositories
> and are intentionally excluded from this repository.

---

## Requirements

- **Python 3.9** (developed and tested on 3.9.6)
- ~2 GB free disk space (mostly for the saliency model weights)
- macOS / Linux (developed on Apple Silicon — `tensorflow-macos` is pulled in
  automatically on Apple hardware)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/marhaia/thesis.git
cd thesis
```

### 2. Create a virtual environment & install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Get the saliency model weights

The UMSI++ weights (~115 MB on disk, 120 MB file) are **not** included in the
repository.

**Option A — download script (recommended):**

```bash
python3 scripts/download_weights.py
```

**Option B — place the file manually** at:

```
saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5
```

The weights are derived from the **UEyes CHI2023** model. See the original
project page for the source dataset and weights:
<https://userinterfaces.aalto.fi/ueyeschi23>.

> The visual complexity metrics (Stage 1) run **without** the weights. Only the
> saliency endpoints require them.

---

## Usage

### Web app (recommended)

```bash
source venv/bin/activate
python3 stage1/app.py
```

Then open <http://localhost:5001> in your browser and upload a screenshot.

### Command line

Analyze a single image and print the 8-feature complexity vector:

```bash
python3 stage1/visual_complexity.py --image path/to/screenshot.png
```

Analyze a whole folder (writes a CSV):

```bash
python3 stage1/visual_complexity.py --dir path/to/folder/
```

---

## API

The Flask app exposes a small JSON API (default port `5001`):

| Endpoint                    | Method | Description                                          |
| --------------------------- | ------ | ---------------------------------------------------- |
| `/api/analyze`              | POST   | Upload an image → returns the 8-feature complexity vector |
| `/api/saliency`             | POST   | Upload an image → returns saliency features (needs weights) |
| `/api/search-time`          | POST   | Predicted visual search time per element             |
| `/api/scanpath-to-target`   | POST   | Predicted scanpath to a selected target element + glance metrics |
| `/api/cognitive-load`       | POST   | Main orchestrator: runs v → s → h, task/profile modifiers, coherence check, reference norms and feedback; returns the deployed 0–100 load score |
| `/api/screen-consistency`   | POST   | Inter-screen consistency across several screens or one animated GIF |
| `/api/learning-curve`       | POST   | Predicted novice→expert search-effort curve for a screen |
| `/api/features`             | GET    | Metadata describing the 8 features                   |

Example:

```bash
curl -F "image=@screenshot.png" http://localhost:5001/api/analyze
```

---

## Testing

`tests/test_pipeline.py` is an **end-to-end demo script**, not a `pytest`
suite: its `test_full_pipeline(image_path)` takes a positional image path, so it
is meant to be run directly rather than collected by pytest. It runs on any
screenshots found under `stage1/data/screenshots/`, or falls back to a synthetic
demo if none are present.

```bash
source venv/bin/activate
python3 tests/test_pipeline.py
```

---

## Citation

If you use this work, please cite the thesis. The saliency component builds on:

> Jiang et al. (2023). *UEyes: Understanding Visual Saliency across User Interface
> Types.* CHI 2023.

---

## License

This is academic research code. A license will be added before a public
open-source release. Until then, please contact the author before reuse.
