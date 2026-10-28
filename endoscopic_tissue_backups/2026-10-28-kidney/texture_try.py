"""Generate candidate kidney detail textures and save them, tiled 2x2, for a look."""
import sys

import numpy as np
from PIL import Image


def tileable_noise(rng, size, scale):
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"))
    noise = np.fft.ifft2(np.fft.fft2(rng.normal(size=(size, size))) * np.exp(-(k * scale) ** 2)).real
    return noise / noise.std()


rng = np.random.default_rng(5)
size = 512
vessels = np.maximum(np.exp(-(tileable_noise(rng, size, 40) / .07) ** 2),
                     .5 * np.exp(-(tileable_noise(rng, size, 14) / .09) ** 2))
grain = tileable_noise(rng, size, 2)
capsule = (.9 + .04 * grain)[..., None] * (1 - vessels[..., None] * np.array((.25, .6, .55)))
cut_grain = tileable_noise(rng, size, 3)
speckles = np.clip(tileable_noise(rng, size, 4) - 2.2, 0, .6) / .6
cut = (.88 + .08 * cut_grain)[..., None] * (1 - speckles[..., None] * np.array((.3, .7, .7)))
tint_capsule = np.array((.50, .16, .13)) / .5
tint_tumor = np.array((.90, .62, .25)) / .5
out = sys.argv[1]
for name, albedo, tint in (("capsule", capsule, tint_capsule), ("tumor", capsule, tint_tumor), ("cut", cut, np.array((.6, .17, .14)) / .5)):
    image = np.clip(np.tile(albedo, (2, 2, 1)) * tint * .5, 0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(f"{out}/tex-{name}.png")
print(capsule.min(), capsule.max(), cut.min(), cut.max())
