"""Compare recorded motion against baseline, without imposing invented realism thresholds."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

left, right, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True, exist_ok=True)
runs = [[json.loads(line) for line in source.read_text().splitlines()] for source in (left, right)]
if any(len(rows) != 1111 or [r["step"] for r in rows] != list(range(1111)) for rows in runs):
    raise ValueError("Expected initial state plus all 1110 simulation steps")
if [r["stage"] for r in runs[0]] != [r["stage"] for r in runs[1]]:
    raise ValueError("Stage mismatch")
stages = list(dict.fromkeys(r["stage"] for r in runs[0]))
keys = ["jaw_gap_mm", "scissors_gap_mm", "grip_error_um", "grip_points", "scissors_contacts"] + [f"{region}_{metric}" for region in ("kidney", "tumor", "stump")
        for metric in ("rms_speed_mm_s", "relative_rms_speed_mm_s", "p99_speed_mm_s")]
summary = {"sources": [str(left), str(right)], "frames": 1111,
           "note": "Motion comparison only; baseline equivalence does not establish clinical realism.", "stages": {}}
for stage in stages[1:]:
    groups = [[r for r in rows if r["stage"] == stage] for rows in runs]
    summary["stages"][stage] = {}
    for key in keys:
        values = [np.array([r[key] for r in rows]) for rows in groups]
        summary["stages"][stage][key] = [dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                                                    maximum=float(a.max())) for a in values]
    summary["stages"][stage]["end_com_difference_mm"] = {
        region: float(np.linalg.norm(np.array(groups[1][-1][region + "_com_cm"]) -
                                    groups[0][-1][region + "_com_cm"]) * 10)
        for region in ("kidney", "tumor", "stump")}
(output / "motion-comparison.json").write_text(json.dumps(summary, indent=2))
fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
for rows, label, color in zip(runs, ("Baseline", "Candidate"), ("#31688e", "#e07a32")):
    seconds = np.array([r["step"] / 30 for r in rows])
    for ax, key, ylabel in zip(axes[:3], ("kidney_relative_rms_speed_mm_s", "tumor_rms_speed_mm_s", "jaw_gap_mm"),
                              ("Kidney relative RMS\nspeed (mm/s)", "Tumor RMS speed\n(mm/s)", "Grasper jaw gap\n(mm)")):
        ax.plot(seconds, [r.get(key, 0) for r in rows], label=label, color=color, linewidth=1, alpha=.85)
        ax.set_ylabel(ylabel)
    axes[3].plot(seconds, [r["stump_com_cm"][1] * 10 for r in rows], color=color, linewidth=1, alpha=.85)
    axes[4].plot(seconds, [r["scissors_gap_mm"] for r in rows], color=color, linewidth=1, alpha=.85)
    axes[5].plot(seconds, [r["scissors_load_n"][0] for r in rows], color=color, linewidth=1, alpha=.85)
    axes[6].plot(seconds, [r["scissors_load_n"][1] for r in rows], color=color, linewidth=1, alpha=.85)
axes[3].set_ylabel("Stump centre height\n(mm)")
axes[4].set_ylabel("Scissors jaw gap\n(mm)")
axes[5].set_ylabel("Left blade load\n(N)")
axes[6].set_ylabel("Right blade load\n(N)")
axes[6].set_xlabel("Simulation time (s); playback time is not execution speed")
axes[0].legend(loc="upper right")
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[["right", "top"]].set_visible(False)
    for rows in (runs[0],):
        for i, row in enumerate(rows):
            if i and row["stage"] != rows[i - 1]["stage"]:
                ax.axvline(row["step"] / 30, color="black", linewidth=.45, alpha=.2)
fig.suptitle("Whole-trajectory motion comparison, unchanged material and solver settings")
fig.savefig(output / "motion-comparison.png", dpi=150)
plt.close(fig)
print(json.dumps({"frames": 1111, "stages": len(stages) - 1, "output": str(output)}))
