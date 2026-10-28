"""Generate branching vessel textures for the kidney capsule and tumor, tiled 2x2, for a look."""
import sys

import cv2
import numpy as np
from PIL import Image


def tileable_noise(rng, size, scale):
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"))
    noise = np.fft.ifft2(np.fft.fft2(rng.normal(size=(size, size))) * np.exp(-(k * scale) ** 2)).real
    return noise / noise.std()


def vessel_tree(rng, size, roots, width):
    """Draw branching vessels that wrap around the edges of a `size` square, return coverage in [0, 1]."""
    canvas = np.zeros((size * 3, size * 3), np.uint8)
    branches = [(rng.uniform(0, size, 2), rng.uniform(0, 2 * np.pi), width, size * .6) for _ in range(roots)]
    while branches:
        point, angle, w, length = branches.pop()
        if w < .5:
            continue
        for _ in range(int(length / 4)):
            angle += rng.normal(0, .18)
            nxt = point + 4 * np.array((np.cos(angle), np.sin(angle)))
            a, b = point % size + size, (point % size) + (nxt - point) + size
            cv2.line(canvas, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)),
                     int(min(255, 120 + 80 * w)), max(1, int(round(w))), cv2.LINE_AA, shift=4)
            point = nxt
            if rng.random() < .05:
                branches.append((point, angle + rng.choice((-1, 1)) * rng.uniform(.5, 1.1), w * .72, length * .6))
    tiles = canvas.reshape(3, size, 3, size).astype(np.float32)
    return np.clip(tiles.sum(axis=(0, 2)) / 255, 0, 1)


rng = np.random.default_rng(11)
size = 512
vessels = np.maximum(vessel_tree(rng, size, 5, 3.2), .6 * vessel_tree(rng, size, 14, 1.4))
grain = tileable_noise(rng, size, 2)
capsule = (.92 + .035 * grain)[..., None] * (1 - vessels[..., None] * np.array((.3, .62, .58)))
out = sys.argv[1]
for name, tint in (("capsule2", np.array((.50, .16, .13))), ("tumor2", np.array((.90, .62, .25)))):
    image = np.clip(np.tile(capsule, (2, 2, 1)) * tint / .92, 0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(f"{out}/tex-{name}.png")
print("ok", vessels.mean())
