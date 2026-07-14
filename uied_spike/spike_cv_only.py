"""Isolated spike test: run ONLY UIED's CV-based region proposal (no CNN
classifier, no Google OCR) on the automotive screenshots, to check whether the
base detector even runs on a modern OpenCV/numpy stack.

Run this from inside a local UIED clone (see README.md in this folder).
It never touches the thesis project or its venv.
"""
import os
from os.path import join as pjoin

import cv2

# UIED's own CV detection module (pure OpenCV).
import detect_compo.ip_region_proposal as ip


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    return int(resize_length * (height / width))


KEY_PARAMS = {
    "min-grad": 10,
    "ffl-block": 5,
    "min-ele-area": 50,
    "merge-contained-ele": True,
    "merge-line-to-paragraph": False,
    "remove-bar": True,
}

# Point these at any screenshots you want to test.
TEST_IMAGES = [
    "test_hmi.png",
    "bmw_route.png",
]

OUTPUT_ROOT = "data/output"


def main():
    for img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"[skip] missing {img_path}")
            continue
        print(f"\n=== {img_path} ===")
        resized_height = resize_height_by_longest_edge(img_path)
        os.makedirs(pjoin(OUTPUT_ROOT, "ip"), exist_ok=True)
        # classifier=None => CV detection only, no CNN type labels yet.
        ip.compo_detection(
            img_path, OUTPUT_ROOT, KEY_PARAMS,
            classifier=None, resize_by_height=resized_height, show=False,
        )
        name = os.path.basename(img_path)[:-4]
        json_path = pjoin(OUTPUT_ROOT, "ip", f"{name}.json")
        print(f"  -> wrote {json_path}, exists={os.path.exists(json_path)}")


if __name__ == "__main__":
    main()
