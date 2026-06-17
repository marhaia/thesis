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
- **A cognitive-load estimate** — combining the above with an optional task
  description and user profile

Everything is exposed through a small **Flask web app**: upload a screenshot in
the browser and inspect the results visually.

---

## Project structure

```
.
├── stage1/            # Visual complexity extraction + Flask web app
│   ├── app.py             → web server (entry point, port 5001)
│   └── visual_complexity.py
├── stage2/            # Task descriptor, user profile, cognitive-load logic
├── saliency/          # UMSI++ deep saliency model (TensorFlow/Keras)
│   └── weights/           → model weights (~433 MB, NOT in git — see Setup)
├── cognitive/         # Jokinen 2020 visual search model + element detector
├── hceye/             # HCEye-based cognitive-load rules
├── scripts/           # Analysis & validation scripts
├── tests/             # Test suite
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

The UMSI++ weights (~433 MB) are **not** included in the repository. Place the
weights file here:

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

| Endpoint              | Method | Description                                          |
| --------------------- | ------ | ---------------------------------------------------- |
| `/api/analyze`        | POST   | Upload an image → returns the 8-feature complexity vector |
| `/api/saliency`       | POST   | Upload an image → returns saliency features (needs weights) |
| `/api/search-time`    | POST   | Predicted visual search time per element             |
| `/api/cognitive-load` | POST   | Cognitive-load estimate                              |
| `/api/features`       | GET    | Metadata describing the 8 features                   |

Example:

```bash
curl -F "image=@screenshot.png" http://localhost:5001/api/analyze
```

---

## Testing

```bash
source venv/bin/activate
python3 -m pytest tests/
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
