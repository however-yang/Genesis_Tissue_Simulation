"""Summarize the scene runs of a validate.sh batch: pass counts, knife overlap, per-blade fracture loads, two-blade
squeeze, stray skin cuts and uncut columns.

Usage: python batch_summary.py <prefix> [<prefix> ...]  (from the speed2-runs output folder)
"""
import glob
import json
import os
import sys

for prefix in sys.argv[1:]:
    for scene in ("wall", "flap", "kidney"):
        runs = []
        for folder in sorted(glob.glob(f"{prefix}-{scene}-*")):
            path = os.path.join(folder, "report.json")
            if os.path.exists(path):
                report = json.load(open(path))
                runs.append((report["status"], report["metrics"]))
        if not runs:
            continue
        overlap = sorted(m["max_knife_body_penetration_m"] * 1000 for _, m in runs)
        snips = [snip for _, m in runs for snip in m.get("snips", [])]
        loads = [snip[side] for snip in snips for side in ("fracture_left_n", "fracture_right_n")]
        squeeze = [snip["two_blade_squeeze_m"] * 1000 for snip in snips]
        print(f"{prefix}-{scene}: passed {sum(status == 'passed' for status, _ in runs)}/{len(runs)}  "
              f"overlap {overlap[0]:.3f}-{overlap[-1]:.3f} median {overlap[len(overlap) // 2]:.3f} mm  "
              f"loads {min(loads):.2f}-{max(loads):.2f} N  squeeze {min(squeeze):.1f}-{max(squeeze):.1f} mm  "
              f"stray {[m.get('stray_skin_cuts') for _, m in runs]}  uncut {[m.get('uncut_columns') for _, m in runs]}  "
              f"pieces {[m.get('lattice_pieces') for _, m in runs]}  tumor {[m.get('tumor_nodes_left') for _, m in runs]}")
