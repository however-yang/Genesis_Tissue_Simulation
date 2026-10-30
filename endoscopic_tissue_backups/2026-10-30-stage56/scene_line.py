"""Print one line per run folder of a scissors flow: status, blade-back overlap, uncut columns, grip points at the end
and the most at any step, and each snip's fracture loads (left, right, N) and two-blade squeeze (mm).

Usage: python scene_line.py <run folder> [<run folder> ...]
"""
import json
import sys
from pathlib import Path

for folder in map(Path, sys.argv[1:]):
    report = json.load(open(folder / "report.json"))
    metrics = report["metrics"]
    grips = [json.loads(line)["grip_points"] for line in open(folder / "trajectory.jsonl")]
    snips = [(round(s["fracture_left_n"], 2), round(s["fracture_right_n"], 2), round(s["two_blade_squeeze_m"] * 1000, 2))
             for s in metrics.get("snips", [])]
    print(f"{folder.name}: {report['status']} at {report.get('last_stage')}, overlap "
          f"{metrics['max_knife_body_penetration_m'] * 1000:.3f} mm, uncut {metrics.get('uncut_columns')}, grip "
          f"{metrics.get('final_grip_points')} of max {max(grips)}, snips {snips}, elapsed {report['elapsed_s']:.0f} s")
