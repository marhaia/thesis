"""Visual reality-check: draw the classifier's predicted type on each detected
box, so a human can eyeball whether the labels are actually correct.

Colours: TextView=green, ImageView=blue, ImageButton=orange, Button=red,
everything else=magenta. Writes an annotated PNG.

Isolated: runs inside the uied_spike clone + its own venv.
"""
import json
import os
import sys

import cv2
import numpy as np
import tf_keras

MODEL_PATH = "models/cnn-rico-1.h5"
CLASS_MAP = [
    "Button", "CheckBox", "Chronometer", "EditText", "ImageButton",
    "ImageView", "ProgressBar", "RadioButton", "RatingBar", "SeekBar",
    "Spinner", "Switch", "ToggleButton", "VideoView", "TextView",
]
COLORS = {
    "TextView": (0, 200, 0),
    "ImageView": (255, 120, 0),
    "ImageButton": (0, 165, 255),
    "Button": (0, 0, 255),
}
DEFAULT_COLOR = (255, 0, 255)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "IDX_CID180_LHD_Media_Browser"
    img_path = f"automotive_sample/{name}.png"
    json_path = f"data/output_automotive/ip/{name}.json"

    model = tf_keras.models.load_model(MODEL_PATH)
    data = json.load(open(json_path))
    h, w = data["img_shape"][:2]
    img = cv2.resize(cv2.imread(img_path), (w, h))

    for c in data["compos"]:
        x1, y1 = c["column_min"], c["row_min"]
        x2, y2 = c["column_max"], c["row_max"]
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
        pred = model.predict(roi_resized[np.newaxis, ...], verbose=0)[0]
        label = CLASS_MAP[int(np.argmax(pred))]
        conf = float(pred[int(np.argmax(pred))])
        color = COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out = f"annotated_{name}.png"
    cv2.imwrite(out, img)
    print(f"wrote {out}  ({w}x{h})")


if __name__ == "__main__":
    main()
