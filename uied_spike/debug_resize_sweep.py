"""Resolution-sweep debug: test whether UIED's aggressive downscaling (longest
edge -> 800 px) is a reason item 4 gives only coarse graphic-vs-text labels.

Hypothesis: the automotive screenshots are very large (CID180 up to 3340 px
wide). UIED shrinks each image to a longest edge of 800 px before detection, so
small controls/text become tiny and lose the detail the 64x64 classifier needs.
If that is the cause, running at a larger resize length should yield more
distinct types and shift labels away from a near-total ImageView collapse.

Runs the same detection + Rico classifier at several resize lengths and prints
the class distribution for each. Isolated: uied_spike clone + its own venv.
"""
import glob
import json
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

SAMPLE_DIR = "automotive_sample"
OUTPUT_ROOT = "data/output_resize_sweep"

KEY_PARAMS = {
    "min-grad": 10,
    "ffl-block": 5,
    "min-ele-area": 50,
    "merge-contained-ele": True,
    "merge-line-to-paragraph": False,
    "remove-bar": True,
}

RESIZE_LENGTHS = [800, 1600, 2400]


def resize_height_by_longest_edge(img_path, resize_length):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    return int(resize_length * (height / width))


def detect(img_path, resize_length):
    resized_height = resize_height_by_longest_edge(img_path, resize_length)
    out_dir = os.path.join(OUTPUT_ROOT, str(resize_length))
    os.makedirs(os.path.join(out_dir, "ip"), exist_ok=True)
    ip.compo_detection(
        img_path, out_dir, KEY_PARAMS,
        classifier=None, resize_by_height=resized_height, show=False,
    )
    name = os.path.basename(img_path).rsplit(".", 1)[0]
    return os.path.join(out_dir, "ip", f"{name}.json")


def classify(model, img, compos):
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

    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png"))
                    + glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))
    print(f"Found {len(images)} automotive images\n")

    for resize_length in RESIZE_LENGTHS:
        all_conf = []
        all_labels = []
        for img_path in images:
            json_path = detect(img_path, resize_length)
            if not os.path.exists(json_path):
                continue
            img = cv2.imread(img_path)
            data = json.load(open(json_path))
            h, w = data["img_shape"][:2]
            img = cv2.resize(img, (w, h))
            results = classify(model, img, data["compos"])
            all_conf.extend(r[1] for r in results)
            all_labels.extend(r[0] for r in results)

        dist = Counter(all_labels)
        print(f"=== resize_length = {resize_length} px "
              f"(longest edge) ===")
        print(f"  total components: {len(all_labels)} | "
              f"mean conf {np.mean(all_conf):.2f} | "
              f"distinct classes {len(dist)}/15")
        for k, v in dist.most_common():
            share = 100.0 * v / len(all_labels)
            print(f"    {k:<12} {v:>4}  ({share:4.1f}%)")
        print()


if __name__ == "__main__":
    main()
