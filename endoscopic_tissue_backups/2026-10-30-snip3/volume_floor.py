"""Count compressed lattice tetrahedra around the flap at the end of chosen stages of runs probed by flap_probe.py.

Usage: python volume_floor.py <stages comma separated> <run folder>...

Each lattice cell with a material corner splits into the six Kuhn tetrahedra along its main diagonal. For every stage
the line gives, over the cells with rest x from 2.5 to 3.6 cm and y above 0.2 cm, the smallest volume ratio of any
tetrahedron and the count of tetrahedra below 0.2 and 0.3 of their rest volume (the solver's volume floor is 0.15),
and where along x the compressed ones lie.
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

stages = sys.argv[1].split(",")
for folder in map(Path, sys.argv[2:]):
    data = np.load(str(folder) + ".npz")
    rest, inside, names = data["rest"], data["inside"] != 0, list(data["names"])
    xs, ys, zs = (np.unique(np.round(rest[:, a], 4)) for a in range(3))
    nx, ny, nz = len(xs), len(ys), len(zs)
    assert nx * ny * nz == len(rest)
    index = lambda i, j, k: i + nx * (j + ny * k)
    i, j, k = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    i, j, k = i.reshape(-1), j.reshape(-1), k.reshape(-1)
    base = index(i, j, k)
    corners = {(a, b, c): index(i + a, j + b, k + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    material = np.zeros(len(base), dtype=bool)
    for ids in corners.values():
        material |= inside[ids]
    region = material & (rest[base, 0] > 2.5 - 1e-3) & (rest[base, 0] < 3.6) & (rest[base, 1] > .2)
    tets = []
    for order in permutations(range(3)):
        step = [np.zeros(3, dtype=int)]
        for axis in order:
            nxt = step[-1].copy()
            nxt[axis] = 1
            step.append(nxt)
        tets.append([corners[tuple(s)][region] for s in step])
    cell_x = rest[base[region], 0]

    def ratios(positions):
        out = []
        for a, b, c, d in tets:
            pa = positions[a]
            out.append(np.einsum("ij,ij->i", np.cross(positions[b] - pa, positions[c] - pa), positions[d] - pa))
        return np.stack(out, axis=1)

    rest_volume = ratios(rest)
    print(f"== {folder.name}: {json.load(open(folder / 'report.json'))['status']}")
    for stage in stages:
        if stage not in names:
            continue
        ratio = ratios(data["stage_nodes"][names.index(stage)]) / rest_volume
        low = ratio < .3
        where = np.unique(np.round(np.repeat(cell_x[:, None], 6, axis=1)[low], 1), return_counts=True)
        print(f"   {stage:13s} min {ratio.min():.3f}  below 0.2: {int((ratio < .2).sum()):3d}  below 0.3: {int(low.sum()):3d}  "
              f"at x {dict(zip(where[0].tolist(), where[1].tolist()))}")
