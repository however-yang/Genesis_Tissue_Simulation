"""Summarise probe_cut_vs_edges records: is each severed material link where the cutting edges were when it parted?

Usage: analyze_probe.py <records.json> <column axis> [...more pairs]. Distances are printed in mm. 'Before' is the
solved state of the step before the link parted, 'after' the step it parted in. Links along the column axis (the
shear plane normal) carry the cut path; links across it are the steps between columns.
"""
import json
import sys

import numpy as np


def stats(values):
    values = np.abs(np.asarray(values, dtype=float)) * 10
    if not len(values):
        return "none"
    return (f"median {np.median(values):.2f}, 95% {np.percentile(values, 95):.2f}, max {values.max():.2f} "
            f"(n={len(values)})")


for path, axis in zip(sys.argv[1::2], sys.argv[2::2]):
    axis = int(axis)
    rows = json.load(open(path))
    material = [r for r in rows if r["material"]]
    vertical = [r for r in material if r["axis"] == axis]
    steps = [r for r in material if r["axis"] != axis]
    print(f"== {path.split('/')[-1]}: {len(rows)} severed links, {len(material)} material "
          f"({len(vertical)} along the column axis, {len(steps)} across it), {len(rows) - len(material)} skin")
    for when in ("before", "after"):
        straddle = np.mean([r[when]["straddles"] for r in vertical]) if vertical else float("nan")
        print(f"  {when:6s}: links across the blade plane {straddle * 100:5.1f}%;"
              f" crack point to blade plane {stats([r[when]['crack_off_plane'] for r in vertical])}")
    residual = max(r["after"]["plane_residual"] for r in rows) * 10 if rows else float("nan")
    print(f"  edge segments' fit residual to one plane: max {residual:.3f} mm")
    clamped = [r for r in vertical if r["before"]["geometric_fraction"] is not None
               and r["before"]["straddles"] and abs(r["before"]["geometric_fraction"] - r["fraction"]) > 1e-3]
    shifts = [(r["before"]["geometric_fraction"] - r["fraction"]) * 1.0 for r in clamped]
    print(f"  native crack fraction differs from the geometric crossing in {len(clamped)} of {len(vertical)} links"
          + (f", by up to {max(abs(s) for s in shifts):.2f} of a link" if shifts else ""))
    ahead = np.array([max(r["after"]["past_edges"]) for r in vertical]) * 10
    if len(ahead):
        swept = np.mean(ahead >= 0) * 100
        print(f"  in the plane: swept by an edge at the step it parted {swept:.0f}%; most ahead of both edges "
              f"{-ahead.min():.2f} mm" if ahead.min() < 0 else f"  in the plane: swept by an edge {swept:.0f}%")
