"""Settle the kidney scene for a few steps and render the endoscope view, timing the view update."""
import sys
import time
from dataclasses import replace

import cv2
import numpy as np

from endoscopic_tissue.native_backend import NativeTissue
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import KIDNEY_SCENE, scissors_settings

scene = KIDNEY_SCENE
settings = replace(scissors_settings(), contact_model=scene.contact_model, young_pa=scene.young_pa)
grid = scene.build_grid(settings.density_kg_m3)
command = scene.stages[0].command
with NativeTissue(grid, settings, initial_command=command) as tissue:
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 10):
        obs = tissue.step(command)
    surface = tissue.surface()
view = EndoscopicView(settings, 1280, 960, kidney=scene.shape, camera=scene.shot)
for attempt in range(3):
    start = time.perf_counter()
    view.update(surface, obs)
    updated = time.perf_counter()
    frame, _ = view.render()
    print(f"update {1000 * (updated - start):.1f} ms, render {1000 * (time.perf_counter() - updated):.1f} ms", flush=True)
cv2.imwrite(sys.argv[1], frame)
# A close-up on the tumor.
tumor = np.array((0, 0, .036))
view.camera.set_pose(pos=tumor + (-.010, -.030, .022), lookat=tumor)
view.update(surface, obs)
cv2.imwrite(sys.argv[1].replace(".png", "-close.png"), view.render()[0])
view.close()
