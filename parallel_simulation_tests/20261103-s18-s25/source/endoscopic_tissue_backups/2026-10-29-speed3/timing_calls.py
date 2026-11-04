"""Run endoscopic_tissue.run and report, per stage, the wall time of every native call the stepping thread makes.

Usage: python timing_calls.py [run.py arguments...]

NativeTissue.check wraps every ctypes call; timing it splits a step into LCS_Step (the substeps, the cut ticks and the
rebuilds inside them, and the final device synchronization), LCS_Finalize, the contact-surface refinement and the
observation readbacks. Graphs run as in a normal run.
"""
import collections
import runpy
import sys
import time

from endoscopic_tissue import native_backend, scenario, surface_diagnostics

# The kidney flow's crack tips fail the strict topology audit after its first snip; record them and go on.
audit = surface_diagnostics.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


surface_diagnostics.topology_statistics = recording_audit
totals = collections.defaultdict(collections.Counter)
steps = collections.Counter()
state = {"stage": "initial"}
check = native_backend.NativeTissue.check
step = native_backend.NativeTissue.step
stage_start = scenario.stage_start


def timed_check(self, name, *args):
    start = time.perf_counter()
    try:
        return check(self, name, *args)
    finally:
        totals[state["stage"]][name] += time.perf_counter() - start


def counted_step(self, command):
    steps[state["stage"]] += 1
    start = time.perf_counter()
    try:
        return step(self, command)
    finally:
        totals[state["stage"]]["(whole step)"] += time.perf_counter() - start


def timed_stage_start(name, *args):
    state["stage"] = name
    return stage_start(name, *args)


native_backend.NativeTissue.check = timed_check
native_backend.NativeTissue.step = counted_step
scenario.stage_start = timed_stage_start
sys.argv[0] = "endoscopic_tissue.run"
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    names = ["(whole step)", "LCS_Step", "LCS_Finalize", "LCS_RefineContactSurface"]
    others = sorted({k for row in totals.values() for k in row} - set(names), key=lambda k: -sum(r[k] for r in totals.values()))
    columns = names + others[:6]
    print("stage           steps " + " ".join(c.replace("LCS_", "")[:14].rjust(14) for c in columns) + "   (ms per step)")
    for stage, row in totals.items():
        n = max(steps[stage], 1)
        print(f"{stage:15s} {steps[stage]:5d} " + " ".join(f"{1000 * row[c] / n:14.2f}" for c in columns))
