"""Put the stage stills of two headless runs side by side, one row per stage, with the run names above each column.

Usage: python compose_ab.py <run A dir> <run B dir> <output png> <label A> <label B> <stage> [<stage> ...]
"""
import sys

import cv2
import numpy as np

run_a, run_b, output, label_a, label_b, *stages = sys.argv[1:]
rows = []
for stage in stages:
    images = [cv2.imread(f"{run}/{stage}.png") for run in (run_a, run_b)]
    assert all(image is not None for image in images), stage
    scaled = [cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA) for image in images]
    row = np.hstack(scaled)
    cv2.putText(row, stage, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    rows.append(row)
header = np.zeros((48, rows[0].shape[1], 3), dtype=np.uint8)
for column, label in enumerate((label_a, label_b)):
    cv2.putText(header, label, (12 + column * rows[0].shape[1] // 2, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(output, np.vstack([header, *rows]))
print(output)
