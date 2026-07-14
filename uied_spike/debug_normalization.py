"""Debug: is the classifier sensitive to the input NORMALIZATION?

UIED feeds (image / 255) in [0,1]. But the Rico model uses a ResNet50 backbone,
and ResNet models are often trained with a different preprocessing (ImageNet
mean subtraction, or Keras' resnet50.preprocess_input). If the model expects a
different normalization, our /255 feed would be a hidden bug.

This runs the SAME boxes through the classifier under several normalizations and
compares class distribution + mean confidence. If /255 is wrong, another scheme
should give clearly higher confidence and more varied classes.

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

# ImageNet BGR means (Caffe-style, as used by many ResNet ports).
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")


def prep(roi, scheme):
    roi = cv2.resize(roi, (64, 64)).astype("float32")
    if scheme == "div255":              # UIED's own: [0,1]
        return roi / 255.0
    if scheme == "raw":                 # [0,255], no scaling
        return roi
    if scheme == "caffe_mean":          # subtract ImageNet mean (BGR)
        return roi - IMAGENET_MEAN_BGR
    if scheme == "tf_pm1":              # scale to [-1,1] (tf/mobilenet style)
        return roi / 127.5 - 1.0
    raise ValueError(scheme)


def classify(model, img, compos, scheme):
    labels, confs = [], []
    for c in compos:
        x1, y1 = c["column_min"], c["row_min"]
        x2, y2 = c["column_max"], c["row_max"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        x = prep(roi, scheme)
        pred = model.predict(x[np.newaxis, ...], verbose=0)[0]
        idx = int(np.argmax(pred))
        labels.append(CLASS_MAP[idx])
        confs.append(float(pred[idx]))
    return labels, confs


def main():
    model = tf_keras.models.load_model(MODEL_PATH)
    images = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))

    for scheme in ["div255", "raw", "caffe_mean", "tf_pm1"]:
        all_labels, all_confs = [], []
        for img_path in images:
            name = os.path.basename(img_path).rsplit(".", 1)[0]
            json_path = os.path.join(OUTPUT_ROOT, "ip", f"{name}.json")
            if not os.path.exists(json_path):
                continue
            data = json.load(open(json_path))
            h, w = data["img_shape"][:2]
            img = cv2.resize(cv2.imread(img_path), (w, h))
            l, c = classify(model, img, data["compos"], scheme)
            all_labels += l; all_confs += c
        dist = Counter(all_labels)
        top = ", ".join(f"{k}:{v}" for k, v in dist.most_common(4))
        print(f"{scheme:<12} mean conf {np.mean(all_confs):.3f} | "
              f"{len(set(all_labels))}/15 classes | {top}")


if __name__ == "__main__":
    main()
