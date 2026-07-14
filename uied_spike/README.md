# UIED / Rico spike: does an off-the-shelf UI detector transfer to automotive HMIs?

This folder lets you **reproduce** the item 4 experiment: testing whether the
UIED tool with its Rico/Android-trained CNN classifier can label GUI element
**types** (Button, Text, Icon, ...) on automotive HMI screenshots.

**Short answer from our run:** the CV-based *region detection* works fine (it
even finds more boxes than our own detector), but the Rico/Android-trained
*type classifier* collapses on automotive screens - it predicts almost
everything as `ImageView` with ~0.3 confidence (near chance). This is an
evidence-based negative result: off-the-shelf UI classifiers do not transfer to
automotive HMIs, which motivates domain-specific training data (thesis item 1).

Everything here runs in an **isolated** clone + venv and never touches the
thesis pipeline or its environment.

---

## What is ours vs. UIED's

- `spike_cv_only.py`  - our script: runs ONLY UIED's OpenCV region detection.
- `spike_classify.py` - our script: runs UIED's CNN type classifier + prints
  the predicted-class distribution and mean confidence.
- Everything else (`detect_compo/`, `cnn/`, `models/`, ...) comes from the
  public UIED repo: https://github.com/MulongXie/UIED

---

## Reproduction recipe

### 1. Clone UIED (the fremd code)

```bash
git clone https://github.com/MulongXie/UIED.git uied_spike
cd uied_spike
```

### 2. Copy our two scripts into the clone

Copy `spike_cv_only.py` and `spike_classify.py` (from this folder) into the
root of the `uied_spike` clone. Put your test screenshots next to them
(e.g. `test_hmi.png`, `bmw_route.png`), or edit the `TEST_IMAGES` paths.

### 3. Create an isolated environment

```bash
python3 -m venv .uied_venv
.uied_venv/bin/pip install --upgrade pip
.uied_venv/bin/pip install "numpy==1.26.4" "opencv-python==4.8.1.78" pandas \
    "tensorflow==2.16.2" "tf-keras==2.16.0" gdown
```

### 4. One-line compatibility fix (UIED uses a Python-2-era call)

`detect_compo/ip_region_proposal.py` calls `time.clock()`, which was removed in
Python 3.8. Replace it with `time.perf_counter()`:

```bash
sed -i '' 's/time\.clock()/time.perf_counter()/g' detect_compo/ip_region_proposal.py
```

(On Linux use `sed -i` without the empty `''`.)

### 5. Download the Rico classifier model (~101 MB)

The only publicly available semantic Rico classifier is `cnn-rico-1.h5`. Get it
from the UIED author's Google Drive:

```bash
mkdir -p models
.uied_venv/bin/gdown --folder 1MK0Om7Lx0wRXGDfNcyj21B0FL1T461v5 -O models
# ensure the file ends up at models/cnn-rico-1.h5
```

### 6. Run the two spikes

```bash
# Step A: CV detection only (writes data/output/ip/<name>.json)
.uied_venv/bin/python spike_cv_only.py

# Step B: run the Rico type classifier on the detected boxes
.uied_venv/bin/python spike_classify.py
```

---

## What we observed (for comparison)

**CV detection (step A):** runs fine, finer-grained than our detector
(e.g. 21 boxes on `test_hmi` vs. 7 in our pipeline; 44 on `bmw_route`).
But CV alone only yields generic "Compo"/"Block", not a semantic type.

**Type classifier (step B):** collapses on automotive HMIs -

| screenshot | classifier output                       | mean confidence |
|------------|-----------------------------------------|-----------------|
| test_hmi   | 20/21 -> `ImageView`, 1 -> `TextView`   | 0.42            |
| bmw_route  | 43/44 -> `ImageView`                    | 0.30            |

On its own Android/Rico data this classifier reaches ~0.8+ confidence. The
near-chance confidence and collapse to a single class = clear domain mismatch.

**Why this is the fair test:** `cnn-rico-1.h5` is the *only* publicly
downloadable semantic Rico classifier. UIED's other classifiers are binary
yes/no detectors only (Text/Noise/Image). The preprocessing in
`spike_classify.py` matches UIED's own `cnn/CNN.py` exactly (resize 64x64,
divide by 255, BGR), so the negative result is not a preprocessing bug.

---

## Conclusion

An off-the-shelf, Android/Rico-trained UI type classifier does **not** transfer
to automotive HMIs. Getting semantic element types to work would require
retraining on labelled automotive screenshots - i.e. domain-specific data
collection (thesis item 1). Item 4 is therefore documented as future work,
backed by this reproducible evidence.
