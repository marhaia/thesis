"""Debug: does the Rico classifier expect RGB or BGR input?

UIED (and our scripts) feed cv2.imread's BGR straight into the model. But CNNs
are usually trained on RGB. If the model expects RGB, the BGR feed is a hidden
bug that could bias predictions. This script runs the SAME boxes through the
classifier twice - BGR vs RGB - and compares the class distribution and mean
confidence, to see whether the channel order matters.

Isolated: runs inside the uied_spike clone + its own venv.
"""
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np
import tf_keras

MODEL_PATH = "models/cnn-rico-1.h5"
CLASS_MAP = [
    "Button", "CheckBox", "Chronometer", "EditText", "ImageButton",
    "ImageView", "ProgressBar", "RadioButton", "RatingBar", "SeekBar",
    "Spinner", "Switch", "ToggleButton", "VideoView", "TextView",
]
SAMPLE_DIR = "automotive_sample"
OUTPUT_ROOT = "data/output_automotive"


def classify(model, img, compos, to_rgb):
    labels, confs = [], []
    for c in compos:
        x1, y1 = c["column_min"], c["row_min"]
        x2, y2 = c["column_max"], c["row_max"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if to_rgb:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
        pred = model.predict(roi_resized[np.newaxis, ...], verbose=0)[0]
        idx = int(np.argmax(pred))
        labels.append(CLASS_MAP[idx])
        confs.append(float(pred[idx]))
    return labels, confs


def summarize(tag, labels, confs):
    print(f"\n=== {tag} ===")
    print(f"  components: {len(labels)}  mean conf: {np.mean(confs):.3f}  "
          f"distinct classes: {len(set(labels))}/15")
    for label, count in Counter(labels).most_common():
        cls_conf = np.mean([c for l, c in zip(labels, confs) if l == label])
        print(f"    {label:<14} {count:>4}  (mean conf {cls_conf:.2f})")


def main():
    model = tf_keras.models.load_model(MODEL_PATH)
    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))

    bgr_labels, bgr_confs = [], []
    rgb_labels, rgb_confs = [], []
    for img_path in images:
        name = os.path.basename(img_path).rsplit(".", 1)[0]
        json_path = os.path.join(OUTPUT_ROOT, "ip", f"{name}.json")
        if not os.path.exists(json_path):
            continue
        data = json.load(open(json_path))
        h, w = data["img_shape"][:2]
        img = cv2.resize(cv2.imread(img_path), (w, h))
        bl, bc = classify(model, img, data["compos"], to_rgb=False)
        rl, rc = classify(model, img, data["compos"], to_rgb=True)
        bgr_labels += bl; bgr_confs += bc
        rgb_labels += rl; rgb_confs += rc

    summarize("BGR (current, as UIED feeds it)", bgr_labels, bgr_confs)
    summarize("RGB (channel-swapped)", rgb_labels, rgb_confs)

    # How often does the predicted label change between BGR and RGB?
    changed = sum(1 for a, b in zip(bgr_labels, rgb_labels) if a != b)
    print(f"\nLabel changed BGR->RGB on {changed}/{len(bgr_labels)} boxes "
          f"({100*changed/max(len(bgr_labels),1):.0f}%)")


if __name__ == "__main__":
    main()
