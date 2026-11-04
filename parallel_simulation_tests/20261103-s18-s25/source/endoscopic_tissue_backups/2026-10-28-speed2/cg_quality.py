"""Share of the PCG's quadratic-model decrease reached by each iteration, from CGTRACE lines.

In PCG the quadratic model drops by rho_k^2/(2 p_k'Ap_k) at iteration k. Usage: python cg_quality.py <log>
"""
import sys
from collections import defaultdict

import numpy as np

solves = defaultdict(lambda: defaultdict(dict))
for line in open(sys.argv[1], encoding="utf8", errors="replace"):
    if not line.startswith("CGTRACE"):
        continue
    _, solve, root, it, rho, nxt, den = line.split()
    solves[int(solve)][int(root)][int(it)] = (float(rho), float(nxt), float(den))
rows = []
for solve, roots in sorted(solves.items()):
    root = max(roots, key=lambda r: roots[r][0][0] if 0 in roots[r] else 0)
    its = roots[root]
    if 0 not in its:
        continue
    n = max(its) + 1
    drop = np.array([its[k][0] ** 2 / its[k][2] if k in its and its[k][2] > 0 else 0.0 for k in range(n)])
    share = np.cumsum(drop) / max(drop.sum(), 1e-300)
    rows.append((solve, n, share))
for n_iter in sorted({r[1] for r in rows}):
    group = np.array([r[2] for r in rows if r[1] == n_iter])
    marks = [k for k in (4, 8, 12, 16, 20, 24, 32, 48, 64) if k <= n_iter]
    print(f"{len(group)} solves with {n_iter} iterations: share of the final model decrease reached by iteration k (median / 5th percentile)")
    print("   " + "  ".join(f"k={k}: {np.median(group[:, k - 1]):.4f}/{np.percentile(group[:, k - 1], 5):.4f}" for k in marks))
