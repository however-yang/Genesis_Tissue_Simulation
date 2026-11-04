"""Time LCS_Step, LCS_Finalize and the contact refine per step on the whole kidney at rest, plus the native GPU phases."""
import ctypes as ct
import time

import numpy as np

from endoscopic_tissue import native_backend
from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.kidney import Kidney
from endoscopic_tissue.native_backend import Command, NativeTissue
from endoscopic_tissue.scenario import scissors_settings

settings = scissors_settings()
grid = build_grid(.1, density=settings.density_kg_m3, vertical_spacing=.1, shape=Kidney())
park = Command(hinge=(-3.0, 7.0, 0.0), blade_center=(3.0, 7.0, 0.0))
rows = []
with NativeTissue(grid, settings, initial_command=park) as tissue:
    tissue.dll.LCS_ProfileEnable(ct.c_int(1))
    for step in range(40):
        tissue.command(park)
        ramp = min(1.0, (tissue.tick + 1) * settings.dt / 0.5)
        tissue.check("LCS_SetGravity", 0, -9.81 / native_backend.METERS_PER_UNIT * ramp, 0)
        t0 = time.perf_counter()
        tissue.check("LCS_Step", settings.dt)
        t1 = time.perf_counter()
        tissue.check("LCS_Finalize")
        t2 = time.perf_counter()
        tissue.check("LCS_RefineContactSurface", settings.contact_refine_passes)
        t3 = time.perf_counter()
        phases = (ct.c_float * 3)()
        tissue.dll.LCS_ProfileRead(ct.c_int(settings.substeps), phases)
        t4 = time.perf_counter()
        tissue.observe()
        t5 = time.perf_counter()
        tissue.tick += 1
        rows.append((t1 - t0, t2 - t1, t3 - t2, t5 - t4, *phases))
rows = np.array(rows[10:]) * (1000, 1000, 1000, 1000, 1, 1, 1)
print("ms per step: LCS_Step %.1f  Finalize %.1f  Refine %.1f  observe %.1f | GPU per step: elastic %.1f  grasp/contact %.1f  velocity %.1f" % tuple(rows.mean(axis=0)))
