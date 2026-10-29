"""List every recorded wall run in the outputs with its blade-back overlap and DLL, worst first."""
import json
from pathlib import Path

outputs = Path(r"D:\Simulation\genesis-world\endoscopic_tissue\outputs")
rows = []
for report in outputs.rglob("report.json"):
    try:
        r = json.loads(report.read_text())
    except (OSError, ValueError):
        continue
    if r.get("scene") != "wall" or "metrics" not in r:
        continue
    m = r["metrics"]
    overlap = m.get("max_knife_body_penetration_m")
    if overlap is None:
        continue
    rows.append((overlap * 1000, report.parent.relative_to(outputs).as_posix(), (r.get("dll_sha256") or "")[:8],
                 m.get("uncut_columns"), m.get("stray_skin_cuts"), r.get("status")))
rows.sort(reverse=True)
print(len(rows), "wall runs")
for row in rows[:15]:
    print("%.3f mm  %-40s dll %s uncut %s stray %s %s" % row)
by_dll = {}
for overlap, _, dll, *_ in rows:
    by_dll.setdefault(dll, []).append(overlap)
for dll, values in sorted(by_dll.items(), key=lambda kv: -len(kv[1])):
    values.sort()
    print(dll, len(values), "runs, median %.3f, max %.3f, above 0.3: %d" % (values[len(values) // 2], values[-1], sum(v > 0.3 for v in values)))
