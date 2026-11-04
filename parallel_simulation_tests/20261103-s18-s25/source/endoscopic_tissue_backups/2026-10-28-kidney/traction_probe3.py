"""Grasp the tumor top, pull it, and report the neck section, the tumor left below each cut height, and wrist clearance.

Usage: python traction_probe3.py <blend> <young_pa> <tumor offset> <tumor half height> <lift cm> [pull x y z]
"""
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from endoscopic_tissue.grid import build_grid, implicit_boundary
from endoscopic_tissue.kidney import ellipsoid_distance, smooth_min
from endoscopic_tissue.native_backend import Command, NativeTissue
from endoscopic_tissue.scenario import interpolate, scissors_settings


@dataclass(frozen=True)
class Shape:
    half_extents: tuple
    medial: int
    tumor_half: tuple
    tumor_offset: float
    blend: float = .2
    bed_depth: float = .3

    @property
    def tumor_center(self):
        return np.array((0, self.half_extents[1] + self.tumor_offset, 0))

    @property
    def low(self):
        return -np.array(self.half_extents, dtype=np.float32)

    @property
    def high(self):
        a, b, c = self.half_extents
        return np.array((a, b + self.tumor_offset + self.tumor_half[1], c), dtype=np.float32)

    def distance(self, points):
        body = ellipsoid_distance(points, np.array(self.half_extents))
        hilum_center = np.zeros(3)
        hilum_center[self.medial] = -self.half_extents[self.medial]
        hilum_half = np.array((1.5, 1.2, 1.2))
        hilum_half[self.medial] = 1.2
        hilum_half[3 - self.medial] = .9
        hilum = ellipsoid_distance(points - hilum_center, hilum_half)
        kidney = -smooth_min(-body, hilum, self.blend)
        tumor = ellipsoid_distance(points - self.tumor_center, np.array(self.tumor_half))
        return smooth_min(kidney, tumor, self.blend)

    def inside(self, positions):
        return self.distance(positions.astype(np.float64)) < 0

    def boundary(self, positions, a, b, inside):
        return implicit_boundary(self.distance, positions, a, b)

    def bed(self, positions):
        return positions[:, 1] < self.low[1] + self.bed_depth


blend, young, offset, tumor_height, lift = float(sys.argv[1]), float(sys.argv[2]), *map(float, sys.argv[3:6])
orientation = "rim"
pull = np.array([float(v) for v in sys.argv[6:9]]) if len(sys.argv) > 8 else np.array((0, 1.0, 0))
pull /= np.linalg.norm(pull)
if orientation == "anterior":
    shape = Shape((5.5, 1.75, 3.0), 2, (.75, tumor_height, .75), offset)
else:
    shape = Shape((5.5, 3.0, 1.75), 1, (.75, tumor_height, .75), offset, blend=blend, bed_depth=1.5)
settings = replace(scissors_settings(), young_pa=young)
grid = build_grid(.1, density=settings.density_kg_m3, vertical_spacing=.1, shape=shape)
top = shape.tumor_center + (0, tumor_height, 0)
axis = np.array(settings.jaw_axis) / np.linalg.norm(settings.jaw_axis)
tips = top + (.3, -.2, 0)
bite = tips - settings.jaw_length * axis
ready = bite - .8 * axis
park = Command(hinge=tuple(ready + (-1.0, 1.2, 0)), blade_center=(7.0, top[1] + 1.5, 3.0), scissors_roll_deg=90.0,
               scissors_pitch_deg=0.0)
stages = [("settle", 20, park), ("approach", 90, replace(park, hinge=tuple(ready))),
          ("bite", 45, replace(park, hinge=tuple(bite))), ("close", 45, replace(park, hinge=tuple(bite), closure=1))]
for fraction in (.5, .75, 1):
    stages.append((f"lift {lift * fraction:.2f}", 60, replace(park, hinge=tuple(bite + lift * fraction * pull), closure=1)))

rest = grid.positions
inside = grid.inside != 0
up = grid.neighbors[:, 3]
links = np.flatnonzero((up >= 0) & (inside | inside[np.maximum(up, 0)]))
links = links[np.linalg.norm(rest[links][:, (0, 2)], axis=1) < 3.0]
base_y = shape.half_extents[1]
tumor_nodes = np.flatnonzero(inside & (rest[:, 1] > base_y))
in_tumor = np.linalg.norm((rest - shape.tumor_center) / shape.tumor_half, axis=1) < 1
tumor_ellipsoid = np.flatnonzero(in_tumor & inside)
print(f"{orientation}: E={young:.0f} Pa, tumor offset {offset}, half height {tumor_height}, active "
      f"{np.count_nonzero(grid.active)}, protrusion nodes {len(tumor_nodes)}, top {top.round(3)}, pull {pull.round(2)}",
      flush=True)
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
        rise = nodes[tumor_nodes, 1] - rest[tumor_nodes, 1]
        print(f"{name:10s} grip {obs.grip_points:4d} gap {obs.jaw_gap:.3f} protrusion rise median {np.median(rise) * 10:.2f}"
              f" mm (min {rise.min() * 10:.2f}, max {rise.max() * 10:.2f}); {np.mean(times[-steps:]) * 1000:.0f} ms/step",
              flush=True)
        if not name.startswith("lift"):
            continue
        ya, yb = nodes[links, 1], nodes[up[links], 1]
        for h in base_y + np.arange(.1, 1.31, .05):
            crossing = (ya < h) & (yb >= h)
            if not np.any(crossing):
                continue
            f = (h - ya[crossing]) / (yb[crossing] - ya[crossing])
            a_, b_ = links[crossing], up[links[crossing]]
            p = nodes[a_] + f[:, None] * (nodes[b_] - nodes[a_])
            r = rest[a_] + f[:, None] * (rest[b_] - rest[a_])
            through_tumor = np.count_nonzero(np.linalg.norm((r - shape.tumor_center) / shape.tumor_half, axis=1) < 1)
            left_below = np.count_nonzero(nodes[tumor_ellipsoid, 1] < h)
            print(f"    cut y {h:.2f}: {np.count_nonzero(crossing):5d} columns, x {p[:, 0].min():+.2f}..{p[:, 0].max():+.2f}"
                  f", z {p[:, 2].min():+.2f}..{p[:, 2].max():+.2f}; rest y {r[:, 1].min():.2f}..{r[:, 1].max():.2f}; "
                  f"through tumor {through_tumor}, tumor nodes below {left_below}", flush=True)
        # Highest tissue under a wrist capsule (radius 0.42 cm, from 0.15 cm in front of the hinge back 1 cm) for a
        # hinge at distance d from the tumor axis, approaching along +z (front) or +x (right).
        material = nodes[inside]
        for label, along, across in (("front", 2, 0), ("right", 0, 2)):
            row = []
            for d in (.8, 1.0, 1.2, 1.4, 1.6):
                band = ((material[:, along] > d - .15) & (material[:, along] < d + 1.0)
                        & (np.abs(material[:, across]) < .42))
                row.append(f"d {d:.1f}: {material[band, 1].max():.2f}")
            print(f"    {label} wrist band top: " + ", ".join(row), flush=True)
    np.savez_compressed(sys.argv[0].replace(".py", f"-{blend}-{offset}-{tumor_height}.npz"), nodes=obs.nodes,
                        rest=rest, neighbors=obs.neighbors, inside=grid.inside)
