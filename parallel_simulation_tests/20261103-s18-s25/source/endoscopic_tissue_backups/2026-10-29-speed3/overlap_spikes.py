"""Find single-step blade-back overlap spikes in recorded wall trajectories, per DLL.

A spike is a step whose overlap exceeds both neighbours by more than 0.05 mm; its height is the excess over the larger
neighbour. A sustained penetration rises over several steps and is not a spike.
"""
import json
from pathlib import Path

folder = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\outputs\speed2-runs")
by_dll = {}
for trajectory in sorted(folder.glob("*-wall-*/trajectory.jsonl")):
    report = json.loads((trajectory.parent / "report.json").read_text())
    dll = (report.get("dll_sha256") or "")[:8]
    values = [json.loads(line)["knife_body_penetration_mm"] for line in trajectory.read_text().splitlines()]
    spikes = [(values[k] - max(values[k - 1], values[k + 1]), k) for k in range(1, len(values) - 1)
              if values[k] - max(values[k - 1], values[k + 1]) > 0.05]
    worst = max(spikes) if spikes else (0.0, None)
    by_dll.setdefault(dll, []).append((trajectory.parent.name, len(spikes), worst[0], max(values)))
for dll, runs in by_dll.items():
    print(dll, len(runs), "runs; runs with a spike:", sum(1 for r in runs if r[1]), "; spikes:", sum(r[1] for r in runs))
    for name, count, height, peak in sorted(runs, key=lambda r: -r[2])[:6]:
        print("   %-22s spikes %d  largest excess %.3f mm  run max %.3f mm" % (name, count, height, peak))
