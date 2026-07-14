# UIED / Rico spike: can an off-the-shelf UI detector label element types on our screenshots?

This folder lets you **reproduce** the item 4 experiment: testing whether the
UIED tool with its Rico/Android-trained CNN classifier can label GUI element
**types** (Button, Text, Icon, ...) on our screenshots.

**Short answer from our runs:** the CV-based *region detection* works fine (it
even finds more boxes than our own detector), but the Rico/Android-trained
*type classifier* does not produce usable types. On automotive HMIs it predicts
almost everything as `ImageView` with ~0.3 confidence (near chance). Crucially,
a **positive control** on ordinary web/mobile UIs (UEyes) shows the *same*
collapse (~97% `ImageView`, conf ~0.4). So the failure is **not specific to
automotive** - the old UIED+Rico classifier is simply too weak to give reliable
element types on our data. A usable semantic detector would need a
retrained/stronger model.

Everything here runs in an **isolated** clone + venv and never touches the
thesis pipeline or its environment.

---

## What is ours vs. UIED's

- `spike_cv_only.py`          - our script: runs ONLY UIED's OpenCV region detection.
- `spike_classify.py`         - our script: runs UIED's CNN type classifier on the
  automotive screenshots + prints the predicted-class distribution.
- `spike_positive_control.py` - our script: same classifier on real web/mobile
  UIs (UEyes sample) as a positive control, to rule out "it's just the
  automotive image".
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

Copy `spike_cv_only.py`, `spike_classify.py` and `spike_positive_control.py`
(from this folder) into the root of the `uied_spike` clone.

- For `spike_classify.py`: put your automotive screenshots next to the scripts
  (e.g. `test_hmi.png`, `bmw_route.png`), or edit the `TEST_IMAGES` paths.
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
```

---

## What we observed (for comparison)

**CV detection (step A):** runs fine, finer-grained than our detector
(e.g. 21 boxes on `test_hmi` vs. 7 in our pipeline; 44 on `bmw_route`).
But CV alone only yields generic "Compo"/"Block", not a semantic type.

**Type classifier - automotive (step B):**

| screenshot | classifier output                       | mean confidence |
|------------|-----------------------------------------|-----------------|
| test_hmi   | 20/21 -> `ImageView`, 1 -> `TextView`   | 0.42            |
| bmw_route  | 43/44 -> `ImageView`                    | 0.30            |

**Type classifier - positive control on web/mobile UIs (step C):**

| sample                 | result                          | mean confidence |
|------------------------|---------------------------------|-----------------|
| 10 UEyes UIs (~640 el) | 618/639 (~97%) -> `ImageView`   | 0.40            |

On its own Android/Rico data this classifier reaches ~0.8+ confidence. Here it
collapses to a single class at near-chance confidence on **both** automotive and
ordinary web/mobile UIs.

**Why this is a fair test:** `cnn-rico-1.h5` is the *only* publicly downloadable
semantic Rico classifier. UIED's other classifiers are binary yes/no detectors
only (Text/Noise/Image). The preprocessing in our scripts matches UIED's own
`cnn/CNN.py` exactly (resize 64x64, divide by 255, BGR), so the negative result
is not a preprocessing bug.

Note on the positive control: UEyes is not the exact Rico training distribution
either, so it is an *approximate* positive control. It still demonstrates the
key point - the classifier does not give usable types outside its narrow
training data, automotive or not.

---

## Conclusion

The off-the-shelf UIED+Rico classifier does **not** produce usable semantic
element types on our data - and the failure is general, not automotive-specific.
Getting semantic element types to work would require a stronger / retrained
model (e.g. a modern zero-shot model such as CLIP, or training on labelled
automotive screenshots - i.e. thesis item 1). Item 4 is therefore open: it needs
a different approach, backed by this reproducible evidence.
