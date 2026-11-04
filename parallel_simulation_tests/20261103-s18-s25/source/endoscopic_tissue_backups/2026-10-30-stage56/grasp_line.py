"""Print one line per probed kidney run (realism_probe.py records) through the blade approach: the jaw gap and pad load
at the end of close, lift and blade_approach, the largest pad load during the blade approach, and the share of held
steps whose contact repair started deeper than its 0.1 mm trigger.

Usage: python grasp_line.py <probe jsonl> [<probe jsonl> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

for path in map(Path, sys.argv[1:]):
    rows = [json.loads(line) for line in open(path)]
    ends = {r["stage"]: r for r in rows}
    held = [r for r in rows if r["stage"] in ("close", "lift", "blade_approach")]
    approach = [sum(r["jaw_load_n"]) for r in rows if r["stage"] == "blade_approach"]
    repaired = np.mean([r["depth_before_mm"] > .1 for r in held]) if held else float("nan")
    print(f"{path.stem:28s} " + "  ".join(f"{s} {ends[s]['jaw_gap_mm']:.3f} mm {sum(ends[s]['jaw_load_n']):5.2f} N"
                                          for s in ("close", "lift", "blade_approach") if s in ends)
          + f"  approach max {max(approach, default=0):5.2f} N  held repaired {repaired:.1%}")
