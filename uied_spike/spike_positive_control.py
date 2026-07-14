"""Positive-control spike: run UIED's CV detection + Rico/Android CNN type
classifier on real web/mobile/desktop UIs (UEyes sample), the domain the
classifier WAS trained for.

Purpose: rule out "it's just the automotive image". If the classifier produces
varied types with clearly higher confidence here than on automotive HMIs, that
proves the earlier collapse is a domain gap, not a broken setup or one bad image.

Isolated: runs inside the uied_spike clone + its own venv, never touches the
thesis project.
"""
import glob
import os
from collections import Counter

import cv2
import numpy as np
import tf_keras

import detect_compo.ip_region_proposal as ip

MODEL_PATH = "models/cnn-rico-1.h5"
CLASS_MAP = [
    "Button", "CheckBox", "Chronometer", "EditText", "ImageButton",
    "ImageView", "ProgressBar", "RadioButton", "RatingBar", "SeekBar",
    "Spinner", "Switch", "ToggleButton", "VideoView", "TextView",
]

SAMPLE_DIR = "ueyes_sample"
OUTPUT_ROOT = "data/output_ueyes"

KEY_PARAMS = {
    "min-grad": 10,
    "ffl-block": 5,
    "min-ele-area": 50,
    "merge-contained-ele": True,
    "merge-line-to-paragraph": False,
    "remove-bar": True,
}


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    return int(resize_length * (height / width))


def detect(img_path):
    """Run UIED CV region proposal; return path to the components JSON."""
    resized_height = resize_height_by_longest_edge(img_path)
    os.makedirs(os.path.join(OUTPUT_ROOT, "ip"), exist_ok=True)
    ip.compo_detection(
        img_path, OUTPUT_ROOT, KEY_PARAMS,
        classifier=None, resize_by_height=resized_height, show=False,
    )
    name = os.path.basename(img_path).rsplit(".", 1)[0]
    return os.path.join(OUTPUT_ROOT, "ip", f"{name}.json")


def classify(model, img, compos):
    results = []
    for c in compos:
        x1, y1 = c["column_min"], c["row_min"]
        x2, y2 = c["column_max"], c["row_max"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # Same preprocessing as UIED's own cnn/CNN.py: resize 64x64, /255, BGR.
        roi_resized = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
        pred = model.predict(roi_resized[np.newaxis, ...], verbose=0)[0]
        idx = int(np.argmax(pred))
        results.append((CLASS_MAP[idx], float(pred[idx])))
    return results


def main():
    import json

    print("Loading classifier ...")
    model = tf_keras.models.load_model(MODEL_PATH)

    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png"))
                    + glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    print(f"Found {len(images)} UEyes sample images\n")

    all_conf = []
    all_labels = []
    per_image_unique = []

    for img_path in images:
        json_path = detect(img_path)
        if not os.path.exists(json_path):
            print(f"[skip] {img_path}: no detection json")
            continue
        img = cv2.imread(img_path)
        compos = json.load(open(json_path))["compos"]
        results = classify(model, img, compos)
        if not results:
            continue
        labels = [r[0] for r in results]
        confs = [r[1] for r in results]
        all_conf.extend(confs)
        all_labels.extend(labels)
        per_image_unique.append(len(set(labels)))
        dist = Counter(labels)
        top = dist.most_common(1)[0]
        print(f"{os.path.basename(img_path):<14} "
              f"{len(results):>3} compos | {len(set(labels))} distinct types | "
              f"mean conf {np.mean(confs):.2f} | top: {top[0]} ({top[1]})")

    print("\n=== POSITIVE-CONTROL SUMMARY (web/mobile/desktop UIs) ===")
    print(f"  total components classified: {len(all_conf)}")
    print(f"  overall mean confidence:     {np.mean(all_conf):.2f}")
    print(f"  distinct classes overall:    {len(set(all_labels))} / 15")
    print(f"  avg distinct types per image:{np.mean(per_image_unique):.1f}")
    print("  class distribution:")
    for label, count in Counter(all_labels).most_common():
        cls_conf = np.mean([c for l, c in zip(all_labels, all_conf) if l == label])
        print(f"    {label:<14} {count:>4}  (mean conf {cls_conf:.2f})")


if __name__ == "__main__":
    main()
