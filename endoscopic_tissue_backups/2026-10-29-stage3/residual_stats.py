"""Compare the end-of-substep residual of debug runs (debug_vertex_residual.py): per stage group, the distribution of the
RMS and largest Newton step a further vertex sweep would take, and the share of nodes whose step exceeds 1 um.

Usage: python residual_stats.py <label>=<run log>[,<run log>...] ...
"""
import re
import sys

import numpy as np

GROUPS = {"settle": ("settle", "approach", "bite"), "held": ("close", "lift", "blade_approach"),
          "cutting": ("incise_1", "open_1", "reposition_2", "incise_2"),
          "post-excision": ("withdraw", "release", "extract", "hold")}
for argument in sys.argv[1:]:
    label, paths = argument.split("=", 1)
    rows = []
    for path in paths.split(","):
        stage = "initial"
        for line in open(path, errors="replace"):
            match = re.match(r"\[\d+\] (\w+)", line)
            if match:
                stage = match.group(1)
            elif line.startswith("RESID "):
                _, _, rms, largest, nodes, above = line.split()
                rows.append((stage, float(rms), float(largest), float(above) / max(float(nodes), 1)))
    print(f"{label}: {len(rows)} substeps")
    for group, stages in GROUPS.items():
        values = np.array([row[1:] for row in rows if row[0] in stages])
        if not len(values):
            continue
        print(f"  {group:14s} rms step p50/p95 {np.median(values[:, 0]) * 1e4:7.2f}/{np.percentile(values[:, 0], 95) * 1e4:7.2f} um"
              f"  max step p50/p95 {np.median(values[:, 1]) * 1e4:7.1f}/{np.percentile(values[:, 1], 95) * 1e4:7.1f} um"
              f"  nodes above 1 um p50/p95 {np.median(values[:, 2]):6.1%}/{np.percentile(values[:, 2], 95):6.1%}")
