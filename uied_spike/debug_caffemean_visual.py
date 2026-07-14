"""Debug reality-check: annotate one image using the alternative caffe_mean
normalization, which gave suspiciously high (~1.0) confidence and flipped most
boxes to TextView. If icons/buttons are wrongly labelled TextView, that proves
the high confidence is saturation garbage, not correctness - so UIED's /255 is
the right choice.

Isolated: runs inside the uied_spike clone + its own venv.
"""
import json
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
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")
COLORS = {"TextView": (0, 200, 0), "ImageView": (255, 120, 0),
          "ImageButton": (0, 165, 255), "Button": (0, 0, 255)}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "IDX_CID180_LHD_MyMode_Menu_PersonalDark"
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
        x = cv2.resize(roi, (64, 64)).astype("float32") - IMAGENET_MEAN_BGR
        pred = model.predict(x[np.newaxis, ...], verbose=0)[0]
        label = CLASS_MAP[int(np.argmax(pred))]
        conf = float(pred[int(np.argmax(pred))])
        color = COLORS.get(label, (255, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    out = f"annotated_caffemean_{name}.png"
    cv2.imwrite(out, img)
    print(f"wrote {out}  ({w}x{h})")


if __name__ == "__main__":
    main()
