"""Grasp the tumor top, pull it up in steps and report the cross-section of the tissue at candidate cut heights.

Usage: python traction_probe.py <orientation: anterior|rim> <young_pa> [lift direction x y z]
"""
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from endoscopic_tissue.grid import build_grid
from endoscopic_tissue.kidney import Kidney, ellipsoid_distance, smooth_min
from endoscopic_tissue.native_backend import Command, NativeTissue
from endoscopic_tissue.scenario import interpolate, scissors_settings


@dataclass(frozen=True)
class RimKidney(Kidney):
    """The kidney standing on its medial border, the lateral rim up, the tumor on the rim."""
    half_extents: tuple = (5.5, 3.0, 1.75)

    def distance(self, points):
        body = ellipsoid_distance(points, np.array(self.half_extents))
        hilum = ellipsoid_distance(points - (0, -self.half_extents[1], 0), np.array((1.5, 1.2, .9)))
        kidney = -smooth_min(-body, hilum, self.blend)
        tumor = np.linalg.norm(points - self.tumor_center, axis=1) - self.tumor_radius
        return smooth_min(kidney, tumor, self.blend)


orientation, young = sys.argv[1], float(sys.argv[2])
pull = np.array([float(v) for v in sys.argv[3:6]]) if len(sys.argv) > 5 else np.array((0, 1.0, 0))
pull /= np.linalg.norm(pull)
kidney = Kidney() if orientation == "anterior" else RimKidney()
settings = replace(scissors_settings(), young_pa=young)
grid = build_grid(.1, density=settings.density_kg_m3, vertical_spacing=.1, shape=kidney)
top = kidney.tumor_center + (0, kidney.tumor_radius, 0)
axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
# Tips 2 mm below and 3 mm right of the tumor top, on the jaw line.
tips = top + (.3, -.2, 0)
bite = tips - settings.jaw_length * axis
ready = bite - .8 * axis
park = Command(hinge=tuple(ready + (-1.0, 1.2, 0)), blade_center=(7.0, top[1] + 1.5, 3.0), scissors_roll_deg=90.0,
               scissors_pitch_deg=0.0)
stages = [("settle", 20, park), ("approach", 90, replace(park, hinge=tuple(ready))),
          ("bite", 45, replace(park, hinge=tuple(bite))), ("close", 45, replace(park, hinge=tuple(bite), closure=1))]
for lift in (.4, .8, 1.2):
    stages.append((f"lift {lift:.1f}", 45, replace(park, hinge=tuple(bite + lift * pull), closure=1)))

rest = grid.positions
inside = grid.inside != 0
up = grid.neighbors[:, 3]
links = np.flatnonzero((up >= 0) & (inside | inside[np.maximum(up, 0)]))
near_axis = np.linalg.norm(rest[links][:, (0, 2)] - kidney.tumor_center[[0, 2]], axis=1) < 3.0
links = links[near_axis]
tumor_nodes = np.flatnonzero(inside & (np.linalg.norm(rest - kidney.tumor_center, axis=1) < kidney.tumor_radius))
print(f"{orientation}: E={young:.0f} Pa, active {np.count_nonzero(grid.active)}, tumor nodes {len(tumor_nodes)}, "
      f"top {top.round(3)}, bite hinge {bite.round(3)}, pull {pull.round(2)}", flush=True)
with NativeTissue(grid, settings, initial_command=park) as tissue:
    obs = tissue.observe()
    previous = park
    times = []
    for name, steps, command in stages:
        for k in range(steps):
            t0 = time.perf_counter()
            obs = tissue.step(interpolate(previous, command, (k + 1) / steps))
            times.append(time.perf_counter() - t0)
        previous = command
        nodes = obs.nodes
        rise = (nodes[tumor_nodes, 1] - rest[tumor_nodes, 1])
        print(f"{name:9s} grip {obs.grip_points:4d} jaw gap {obs.jaw_gap:.3f} hinge {obs.hinge.round(3)} "
              f"tumor rise median {np.median(rise) * 10:.2f} mm (min {rise.min() * 10:.2f}, max {rise.max() * 10:.2f}); "
              f"{np.mean(times[-steps:]) * 1000:.0f} ms/step", flush=True)
        if name.startswith("lift"):
            ya, yb = nodes[links, 1], nodes[up[links], 1]
            base = kidney.half_extents[1]
            for h in base + np.arange(-.1, 1.21, .1):
                crossing = (ya < h) & (yb >= h)
                if not np.any(crossing):
                    continue
                f = (h - ya[crossing]) / (yb[crossing] - ya[crossing])
                points = nodes[links[crossing]] + f[:, None] * (nodes[up[links[crossing]]] - nodes[links[crossing]])
                print(f"    y {h:.2f}: {np.count_nonzero(crossing):5d} columns, x {points[:, 0].min():+.2f}..{points[:, 0].max():+.2f}"
                      f", z {points[:, 2].min():+.2f}..{points[:, 2].max():+.2f}", flush=True)
    np.savez_compressed(sys.argv[0].replace(".py", f"-{orientation}.npz"), nodes=obs.nodes, rest=rest,
                        neighbors=obs.neighbors, inside=grid.inside)
