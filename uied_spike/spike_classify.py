"""Decisive spike test: run UIED's CNN element-type classifier on the
automotive screenshots and report the predicted-class distribution.

Answers the key question: does the Rico/Android-trained classifier produce
meaningful UI types on automotive HMIs, or does the domain gap make it
unusable (which would justify item 4 as future work)?

Isolated: uses the uied_spike clone + its own venv, never touches the thesis.
"""
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

TEST_IMAGES = {
    "test_hmi": "/Users/Q682780/Thesis_G/stage1/data/screenshots/test_hmi.png",
    "bmw_route": "/Users/Q682780/Thesis_G/stage1/data/screenshots/bmw_route.png",
}


def classify_components(model, img, compos):
    """Classify each detected component ROI; return (label, confidence) list."""
    results = []
    for c in compos:
        x1, y1 = c["column_min"], c["row_min"]
        x2, y2 = c["column_max"], c["row_max"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
        pred = model.predict(roi_resized[np.newaxis, ...], verbose=0)[0]
        idx = int(np.argmax(pred))
        results.append((CLASS_MAP[idx], float(pred[idx])))
    return results


def main():
    print("Loading classifier ...")
    model = tf_keras.models.load_model(MODEL_PATH)

    for name, img_path in TEST_IMAGES.items():
        json_path = f"data/output/ip/{name}.json"
        if not os.path.exists(json_path) or not os.path.exists(img_path):
            print(f"[skip] {name}: missing detection json or image")
            continue
        img = cv2.imread(img_path)
        data = json.load(open(json_path))
        # Detection ran on a resized image; coords are in that space.
        # Resize the original to the detection shape before cropping ROIs.
        h, w = data["img_shape"][:2]
        img = cv2.resize(img, (w, h))
        compos = data["compos"]
        results = classify_components(model, img, compos)

        labels = [r[0] for r in results]
        confs = [r[1] for r in results]
        dist = Counter(labels)
        mean_conf = float(np.mean(confs)) if confs else 0.0

        print(f"\n=== {name} ({len(results)} components) ===")
        print(f"  mean confidence: {mean_conf:.2f}")
        for label, count in dist.most_common():
            # Mean confidence for this class.
            cls_conf = np.mean([c for l, c in results if l == label])
            print(f"  {label:<14} {count:>3}  (mean conf {cls_conf:.2f})")


if __name__ == "__main__":
    main()
