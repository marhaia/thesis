# UIED / Rico spike: can an off-the-shelf UI detector label element types on our screenshots?

This folder lets you **reproduce** the item 4 experiment: testing whether the
UIED tool with its Rico/Android-trained CNN classifier can label GUI element
**types** (Button, Text, Icon, ...) on automotive HMI screenshots.

**Short answer from our runs (after fixing a bug, see below):**

1. UIED's CV *region detection* works fine and is finer-grained than our own
   detector.
2. The Rico *type classifier*, once ROIs are cropped correctly, is **stable and
   confident** (mean confidence ~0.79 across 12 automotive screens, ~0.72 on
   web/mobile UIs) - it does **not** collapse to noise.
3. BUT a **visual check** shows the labels are **coarse and often wrong**: it
   reliably flags large graphics/tiles/covers as `ImageView`, but frequently
   mislabels text as `ImageView` and - most importantly - labels buttons,
   sliders and switches as `ImageView` too. In practice it acts as a rough
   "graphic vs text" separator, not the fine 15-type widget classifier we hoped
   for. High confidence here does **not** mean high correctness.

So: UIED+Rico is usable only as a coarse hint, not as a reliable semantic
element-type detector for automotive HMIs. A dependable type detector would need
a model trained on automotive data.

> **Honesty note - we found and fixed a bug.** Our first run showed the
> classifier "collapsing" to `ImageView` at ~0.3 confidence, which looked like a
> domain failure. It was actually a **coordinate-space bug**: UIED detects on a
> resized image, but we cropped the boxes from the original full-resolution
> image, so the ROIs were misaligned. After cropping from the correctly resized
> image, confidence rose to ~0.79 and varied types appeared. The corrected
> scripts and results are what this README now documents.

Everything here runs in an **isolated** clone + venv and never touches the
thesis pipeline or its environment.

---

## What is ours vs. UIED's

- `spike_cv_only.py`          - our script: runs ONLY UIED's OpenCV region detection.
- `spike_classify.py`         - our script: runs UIED's CNN type classifier on the
  automotive screenshots (correct ROI cropping) + prints the class distribution.
- `spike_positive_control.py` - our script: same classifier on real web/mobile
  UIs (UEyes sample), a positive control to rule out "it's just the automotive
  image".
- `spike_automotive_batch.py` - our script: runs detection + classification over
  a whole folder of automotive screenshots and prints a per-image + overall
  summary.
- `spike_visualize.py`        - our script: draws the predicted type + confidence
  on each detected box, so a human can eyeball whether the labels are correct.
- `annotated_examples/`       - three annotated CID180 screens produced by
  `spike_visualize.py`, so you can see the labels without running anything.
- Everything else (`detect_compo/`, `cnn/`, `models/`, ...) comes from the
  public UIED repo: https://github.com/MulongXie/UIED

---

## Reproduction recipe

### 1. Clone UIED (the fremd code)

```bash
git clone https://github.com/MulongXie/UIED.git uied_spike
cd uied_spike
```

### 2. Copy our scripts into the clone

Copy all `spike_*.py` scripts (from this folder) into the root of the
`uied_spike` clone.

- For `spike_classify.py`: put your automotive screenshots next to the scripts
  (e.g. `test_hmi.png`, `bmw_route.png`), or edit the `TEST_IMAGES` paths.
- For `spike_automotive_batch.py`: create a folder `automotive_sample/` and drop
  your automotive screenshots in it.
- For `spike_positive_control.py`: create a folder `ueyes_sample/` and drop in a
  handful of web/mobile UI images (we used 10 images from the UEyes dataset).

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

### 6. Run the spikes

```bash
# Step A: CV detection only (writes data/output/ip/<name>.json)
.uied_venv/bin/python spike_cv_only.py

# Step B: run the Rico type classifier on the automotive boxes
.uied_venv/bin/python spike_classify.py

# Step C: positive control on web/mobile UIs (UEyes sample)
.uied_venv/bin/python spike_positive_control.py

# Step D: batch over a whole automotive_sample/ folder (per-image + summary)
.uied_venv/bin/python spike_automotive_batch.py

# Step E: draw predicted types on one image to eyeball correctness
.uied_venv/bin/python spike_visualize.py IDX_CID180_LHD_Media_Browser
```

---

## What we observed (corrected, after the ROI bug fix)

**CV detection:** runs fine, finer-grained than our detector
(e.g. 21 boxes on `test_hmi` vs. 7 in our pipeline; 44 on `bmw_route`).
But CV alone only yields generic "Compo"/"Block", not a semantic type.

**Type classifier - 12 automotive screens (8 CID180 infotainment, 2 PHUD,
`bmw_route`, `test_hmi`):**

| metric                        | value                                   |
|-------------------------------|-----------------------------------------|
| overall mean confidence       | **0.79** (11/12 images in 0.67-0.85)    |
| total components classified   | 524                                     |
| class distribution            | ImageView 460, TextView 46, ImageButton 17, Button 1 |
| distinct classes ever used    | 4 / 15                                   |

**Type classifier - positive control, 10 web/mobile UIs (UEyes):**

| metric                  | value                                                     |
|-------------------------|-----------------------------------------------------------|
| overall mean confidence | 0.72                                                      |
| class distribution      | ImageView 515, TextView 124, ImageButton 24, Button 6, EditText 6, SeekBar 1 |

**Visual check (see `annotated_examples/`):** on the CID180 screens the model
correctly marks large album covers / preview tiles as `ImageView`, but it
labels most on-screen **text** and virtually all **buttons / climate controls /
menu icons** as `ImageView` too. So the high confidence is misleading: the model
is sure, but the fine type is frequently wrong. It effectively distinguishes
"large graphic" from "text" only roughly, and does not recognise interactive
controls as such.

**Why this is a fair test:** `cnn-rico-1.h5` is the *only* publicly downloadable
semantic Rico classifier. UIED's other classifiers are binary yes/no detectors
only (Text/Noise/Image). The preprocessing in our scripts matches UIED's own
`cnn/CNN.py` exactly (resize 64x64, divide by 255, BGR).

**Limitation:** we have no hand-labelled ground truth, so we cannot report a
numeric type-accuracy - the visual check is a qualitative correctness judgement.

---

## Conclusion

After fixing the ROI bug, the off-the-shelf UIED+Rico classifier is stable and
confident, but it does **not** provide reliable semantic element types on
automotive HMIs. It behaves as a coarse "graphic vs text" separator and, most
importantly, does not recognise interactive controls (buttons, sliders,
switches) - which are exactly the elements that matter most for driver
distraction. A dependable type detector would need a stronger / retrained model
(e.g. a modern zero-shot model such as CLIP, or a model trained on labelled
automotive screenshots - i.e. thesis item 1).

Item 4 with this specific tool is therefore a coarse hint at best. The value of
this spike is the reproducible, honestly-scoped evidence (including the
bug-fix lesson: high confidence is not correctness).
