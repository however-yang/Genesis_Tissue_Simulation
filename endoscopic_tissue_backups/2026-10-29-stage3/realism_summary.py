"""Summarize probed kidney runs (probe_runs.sh) side by side: the run's acceptance metrics, the grasp equilibrium, how
the kidney moves when the lift stops and when the tumor is cut free, how the displacement spreads through the kidney,
the crease of the excised tumor, and the contact repair.

Usage: python realism_summary.py <run name> [<run name> ...]   (names under endoscopic_tissue/outputs/stage3-runs)

Displacements are taken from the stage snapshots relative to the end of settle. The lift profile bins the material
nodes by their distance from the tumor axis, the close profile by their distance from the nearest node the jaws grip.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUTPUT = Path(os.environ.get("PROBE_OUTPUT", "D:/Simulation/genesis-world/endoscopic_tissue/outputs/stage3-runs"))
CREASE = Path(__file__).resolve().parents[1] / "2026-10-29-speed3" / "tumor_crease.py"
GROUPS = {"held": ("close", "lift", "blade_approach"), "cutting": ("incise_1", "open_1", "reposition_2", "incise_2"),
          "post-excision": ("withdraw", "release", "extract", "hold")}


def stage_end(rows, stage):
    return [r for r in rows if r["stage"] == stage][-1]


def binned(values, distances, edges):
    return [float(np.median(values[(distances >= lo) & (distances < hi)])) for lo, hi in zip(edges[:-1], edges[1:])]


for name in sys.argv[1:]:
    rows = [json.loads(line) for line in open(OUTPUT / f"{name}.jsonl")]
    report = json.load(open(OUTPUT / name / "report.json"))
    m = report["metrics"]
    print(f"== {name}: {report['status']}, overlap {m['max_knife_body_penetration_m'] * 1000:.3f} mm, pieces "
          f"{m.get('material_components')}/{m.get('lattice_pieces')}, tumor left {m.get('tumor_nodes_left')}, stray "
          f"{m.get('stray_skin_cuts')}, snips (L N, R N, squeeze mm) "
          f"{[(round(x['fracture_left_n'], 3), round(x['fracture_right_n'], 3), round(x['two_blade_squeeze_m'] * 1000, 2)) for x in m.get('snips', [])]}")
    ends = {stage: stage_end(rows, stage) for stage in ("close", "lift", "blade_approach", "hold")}
    print("  jaw gap/load: " + ", ".join(f"{stage} {r['jaw_gap_mm']:.3f} mm {sum(r['jaw_load_n']):.2f} N" for stage, r in ends.items()))
    tick = {r["tick"]: r for r in rows}
    lift_end = ends["lift"]["tick"]
    stump = np.array([r["stump_y_cm"] for r in rows]) * 10
    ticks = np.array([r["tick"] for r in rows])
    rise = (ends["lift"]["stump_y_cm"] - ends["close"]["stump_y_cm"]) * 10
    tumor_rise = (ends["lift"]["tumor_com_cm"][1] - ends["close"]["tumor_com_cm"][1]) * 10
    after = stump[(ticks > lift_end) & (ticks <= lift_end + 60)]
    rest = stump[(ticks > lift_end + 50) & (ticks <= lift_end + 90)].mean()
    print(f"  lift: stump rise {rise:.2f} mm, tumor rise {tumor_rise:.2f} mm; after the stop kidney rms speed "
          f"+5 {tick[lift_end + 5]['kidney_rms_speed_mm_s']:.2f}, +10 {tick[lift_end + 10]['kidney_rms_speed_mm_s']:.2f} mm/s, "
          f"stump overshoot {after.max() - rest:.4f} mm")
    drop = int(np.argmin(np.diff(stump))) + 1
    final = stump[drop + 30:drop + 50].mean()
    settled = next(k for k in range(drop, len(rows)) if rows[k]["kidney_rms_speed_mm_s"] < .2) - drop
    recoil = (np.array(rows[drop + 50]["tumor_com_cm"]) - np.array(rows[drop + 50]["hinge_cm"])
              - np.array(rows[drop - 3]["tumor_com_cm"]) + np.array(rows[drop - 3]["hinge_cm"])) * 10
    print(f"  excision at {rows[drop]['tick']}: stump drop {stump[drop - 3] - final:.2f} mm, undershoot "
          f"{final - stump[drop:drop + 10].min():.4f} mm, kidney rms below 0.2 mm/s after {settled} steps, peak "
          f"{max(r['kidney_rms_speed_mm_s'] for r in rows[drop:drop + 10]):.1f} mm/s; tumor recoil towards the jaws "
          f"{np.round(recoil, 2)} mm")
    held = (np.array(ends["hold"]["tumor_com_cm"]) - np.array(ends["hold"]["hinge_cm"])) * 10
    print(f"  held tumor centre from the hinge at the end {np.round(held, 2)} mm")
    masks = np.load(OUTPUT / f"{name}-masks.npz")
    material = masks["tumor"] | masks["kidney"]
    settle, close, lift = (np.load(OUTPUT / name / f"{stage}.npz") for stage in ("settle", "close", "lift"))
    base = settle["nodes_cm"][material]
    axis = np.hypot(base[:, 0], base[:, 2])
    lifted = np.linalg.norm(lift["nodes_cm"][material] - base, axis=1) * 10
    print("  lift displacement (mm) by distance from the tumor axis 0-1/1-2/2-3/3-4/4-6 cm: "
          + "/".join(f"{v:.2f}" for v in binned(lifted, axis, [0, 1, 2, 3, 4, 6])))
    gripped = (close["grip_flags"] != 0)[material]
    distance = cKDTree(base[gripped]).query(base)[0] * 10
    squeezed = np.linalg.norm(close["nodes_cm"][material] - base, axis=1) * 10
    print(f"  close: {gripped.sum()} gripped nodes, their displacement median/max {np.median(squeezed[gripped]):.2f}/"
          f"{squeezed[gripped].max():.2f} mm; displacement by distance from them 0-1/1-2/2-4/4-8/8-15 mm: "
          + "/".join(f"{v:.3f}" for v in binned(squeezed, distance, [0, 1, 2, 4, 8, 15])))
    crease = subprocess.run([sys.executable, str(CREASE), str(OUTPUT / name / "hold.npz")], capture_output=True, text=True)
    print("  crease:", crease.stdout.strip().split(": ", 1)[-1] or crease.stderr.strip()[-200:])
    for group, stages in GROUPS.items():
        group_rows = [r for r in rows if r["stage"] in stages]
        before = np.array([r["depth_before_mm"] for r in group_rows])
        refine = np.array([r["refine_ms"] for r in group_rows])
        print(f"  repair {group:13s} repaired {(before > .1).mean():6.1%}, depth p50/p95/max {np.median(before):.3f}/"
              f"{np.percentile(before, 95):.3f}/{before.max():.3f} mm, refine {refine.mean():.2f} ms/step")
