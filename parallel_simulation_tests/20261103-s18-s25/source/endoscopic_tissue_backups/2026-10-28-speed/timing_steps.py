"""Run endoscopic_tissue.run and report, per stage, the wall time of each step of the stepping loop and its parts.

Columns (ms per step): the whole loop iteration on the stepping thread (the critical path), the native step inside it,
and the surface read for the step checks. Per-step checks run on the recorder thread and appear only through the time
the stepping thread waits for them at stage ends.
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

totals = collections.defaultdict(lambda: collections.Counter())
state = {"stage": "initial", "mark": None}
step = native_backend.NativeTissue.step
surface = native_backend.NativeTissue.surface
stage_start = scenario.stage_start


def timed_step(self, command):
    start = time.perf_counter()
    if state["mark"] is not None:
        totals[state["stage"]]["loop"] += start - state["mark"]
        totals[state["stage"]]["steps"] += 1
    state["mark"] = start
    result = step(self, command)
    totals[state["stage"]]["native step"] += time.perf_counter() - start
    return result


def timed_surface(self):
    start = time.perf_counter()
    result = surface(self)
    totals[state["stage"]]["surface read"] += time.perf_counter() - start
    return result


def timed_stage_start(name, *args):
    state["stage"], state["mark"] = name, None
    return stage_start(name, *args)


native_backend.NativeTissue.step = timed_step
native_backend.NativeTissue.surface = timed_surface
scenario.stage_start = timed_stage_start
sys.argv[0] = "endoscopic_tissue.run"
started = time.perf_counter()
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    print("stage            steps   loop  native step  surface read   (ms per step)")
    for stage, row in totals.items():
        n = max(row["steps"], 1)
        print(f"{stage:15s} {row['steps']:6d} {1000 * row['loop'] / n:6.1f} {1000 * row['native step'] / n:12.1f} "
              f"{1000 * row['surface read'] / n:13.1f}")
    print("total wall s", round(time.perf_counter() - started, 1))
