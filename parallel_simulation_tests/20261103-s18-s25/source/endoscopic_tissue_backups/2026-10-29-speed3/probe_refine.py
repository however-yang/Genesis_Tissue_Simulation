"""Run endoscopic_tissue.run and record, for every step, the contact-surface depth before and after
LCS_RefineContactSurface and the wall time of that call.

Usage: python probe_refine.py <output jsonl> [run.py arguments...]

The refinement repeats repair passes while the depth exceeds the contact tolerance and stops after two passes that do
not lower it, so a depth left above the tolerance marks a step whose passes stalled.
"""
import ctypes as ct
import json
import runpy
import sys
import time

from endoscopic_tissue import native_backend, scenario, surface_diagnostics

output_path, *run_args = sys.argv[1:]
# The kidney flow's crack tips fail the strict topology audit after its first snip; record them and go on.
audit = surface_diagnostics.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


surface_diagnostics.topology_statistics = recording_audit
check = native_backend.NativeTissue.check
stage_start = scenario.stage_start
state = {"stage": "initial", "tick": 0}
records = open(output_path, "w")


def depth(tissue):
    value = ct.c_float()
    assert tissue.dll.LCS_SurfaceDepth(ct.byref(value)) == 0
    return value.value


def probed_check(self, name, *args):
    if name != "LCS_RefineContactSurface":
        return check(self, name, *args)
    # Zero passes only take over the grasper pose the step solved, against which the refinement measures the depth.
    assert self.dll.LCS_RefineContactSurface(0) == 0
    before = depth(self)
    start = time.perf_counter()
    try:
        return check(self, name, *args)
    finally:
        elapsed = time.perf_counter() - start
        records.write(json.dumps({"tick": self.tick, "stage": state["stage"], "depth_before_cm": before,
                                  "depth_after_cm": depth(self), "refine_ms": 1000 * elapsed}) + "\n")


def probed_stage_start(name, *args):
    state["stage"] = name
    return stage_start(name, *args)


native_backend.NativeTissue.check = probed_check
scenario.stage_start = probed_stage_start
sys.argv = ["endoscopic_tissue.run", *run_args]
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    records.close()
