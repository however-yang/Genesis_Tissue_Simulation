"""List the wall runs in the speed outputs that report stray skin cuts, with the DLL that produced them."""
import glob
import json
from pathlib import Path

folder = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\outputs\speed2-runs")
reports = sorted(folder.glob("*-wall-*/report.json"))
print(len(reports), "wall reports")
for report in reports:
    try:
        r = json.loads(report.read_text())
    except (OSError, ValueError):
        continue
    m = r["metrics"]
    if m.get("stray_skin_cuts"):
        print(report.parent.name, "stray", m["stray_skin_cuts"], r["status"], (r.get("dll_sha256") or "")[:8])
