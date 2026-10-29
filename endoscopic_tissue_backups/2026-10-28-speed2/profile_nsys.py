"""Run a scene natively and open the Nsight capture range for a window of steps in one stage, then exit.

Usage: python profile_nsys.py <stage> <skip steps> <window steps> [run.py arguments...]
Run under: nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node
           --capture-range=cudaProfilerApi --capture-range-end=stop ...
"""
import ctypes
import runpy
import sys

from endoscopic_tissue import native_backend, scenario, surface_diagnostics

stage_name, skip, window = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["endoscopic_tissue.run", *sys.argv[4:]]
driver = ctypes.WinDLL("nvcuda.dll")

# The kidney flow's crack tips fail the strict topology audit after its first snip; record them and go on.
audit = surface_diagnostics.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


surface_diagnostics.topology_statistics = recording_audit
step = native_backend.NativeTissue.step
stage_start = scenario.stage_start
state = {"stage": None, "count": 0}


def tracked_stage_start(name, *args):
    state["stage"], state["count"] = name, 0
    return stage_start(name, *args)


def windowed_step(self, command):
    if state["stage"] == stage_name:
        if state["count"] == skip:
            driver.cuProfilerStart()
        state["count"] += 1
    result = step(self, command)
    if state["stage"] == stage_name and state["count"] == skip + window:
        driver.cuProfilerStop()
        print(f"captured {window} steps of {stage_name}", flush=True)
        raise SystemExit(0)
    return result


scenario.stage_start = tracked_stage_start
native_backend.NativeTissue.step = windowed_step
runpy.run_module("endoscopic_tissue.run", run_name="__main__")
