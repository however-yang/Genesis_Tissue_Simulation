"""Settle the whole kidney for a few steps with the instruments parked far above it, timing each step."""
import time

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.kidney import Kidney
from endoscopic_tissue.native_backend import Command, NativeTissue
from endoscopic_tissue.scenario import scissors_settings

settings = scissors_settings()
grid = build_grid(.1, density=settings.density_kg_m3, vertical_spacing=.1, shape=Kidney())
park = Command(hinge=(-3.0, 7.0, 0.0), blade_center=(3.0, 7.0, 0.0))
with NativeTissue(grid, settings, initial_command=park) as tissue:
    start = tissue.observe().nodes.copy()
    times = []
    for step in range(60):
        t0 = time.perf_counter()
        obs = tissue.step(park)
        times.append(time.perf_counter() - t0)
    surface = tissue.surface()
    moved = np.linalg.norm(obs.nodes - start, axis=1)
    print(f"steps 60: median {np.median(times) * 1000:.1f} ms, last 20 {np.mean(times[-20:]) * 1000:.1f} ms; "
          f"surface {len(surface.vertices)} vertices {len(surface.triangles)} triangles; "
          f"max settle {moved.max() * 10:.2f} mm, mean {moved[grid.active != 0].mean() * 10:.3f} mm")
