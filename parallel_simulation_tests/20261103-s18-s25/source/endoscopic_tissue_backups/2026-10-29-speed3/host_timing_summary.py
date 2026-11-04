"""Summarize the host-timing debug log (debug_host_timing.py build) per stage.

Usage: python host_timing_summary.py <log>
"""
import collections
import re
import sys

stage = "initial"
steps = collections.Counter()
step_parts = collections.defaultdict(collections.Counter)
refine = collections.defaultdict(collections.Counter)
graphs = collections.defaultdict(collections.Counter)
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.match(r"^\[(\d+)\] (\w+)", line)
    if m:
        stage = m.group(2)
        continue
    if line.startswith("STEPHOST"):
        v = line.split()
        steps[stage] += 1
        part = step_parts[stage]
        part["begin"] += float(v[2]); part["substeps"] += float(v[4]); part["ticks"] += float(v[6])
        part["rebuild"] += float(v[8]); part["rebuilds"] += int(v[9]); part["sync"] += float(v[11]); part["total"] += float(v[13])
    elif line.startswith("REFINE"):
        v = line.split()
        r = refine[stage]
        r["calls"] += 1; r["passes"] += int(v[2])
        for key, index in (("measure", 7), ("prepare", 9), ("relax", 11), ("rotations", 13), ("rebuild", 15), ("total", 17)):
            r[key] += float(v[index])
    elif line.startswith("GRAPH"):
        v = line.split()
        graphs[stage][v[1]] += float(v[2]); graphs[stage][v[1] + " count"] += 1
print("per step (ms):  begin substeps ticks rebuild(count) sync step-total | refine calls passes measure prepare relax rot rebuild total | graph captures")
for name in steps:
    n = steps[name]
    p, r, g = step_parts[name], refine[name], graphs[name]
    print(f"{name:15s} {n:4d} {p['begin']/n:6.2f} {p['substeps']/n:7.2f} {p['ticks']/n:6.2f} {p['rebuild']/n:6.2f}({p['rebuilds']/n:.2f}) {p['sync']/n:5.2f} {p['total']/n:7.2f} |"
          f" {r['calls']/n:5.2f} {r['passes']/n:5.2f} {r['measure']/n:6.2f} {r['prepare']/n:6.2f} {r['relax']/n:6.2f} {r['rotations']/n:5.2f} {r['rebuild']/n:6.2f} {r['total']/n:6.2f} |"
          f" unified {g['unified count']/n:.2f}x {g['unified']/n:.2f} ms, contact {g['contact count']/n:.2f}x {g['contact']/n:.2f} ms")
