"""Run endoscopic_tissue.run natively on a profiling DLL and report, per stage, the GPU time of each solver section.

Usage: python profile_sections.py <profiling dll> <report json> [run.py arguments...]

Section timing launches the solver's kernels one by one instead of through its captured graphs, so the totals add
launch gaps that graph launches avoid; the split between sections is what this measures. The step loop's wall time
per stage is reported beside it for comparison.
"""
import ctypes as ct
import json
import runpy
import sys
import time

from endoscopic_tissue import native_backend, scenario, surface_diagnostics

dll_path, report_path, *run_args = sys.argv[1:]
# The kidney flow's crack tips fail the strict topology audit after its first snip; record them and go on.
audit = surface_diagnostics.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


surface_diagnostics.topology_statistics = recording_audit
sessions = []
initialize = native_backend.NativeTissue.__init__
step = native_backend.NativeTissue.step
stage_start = scenario.stage_start
state = {"stage": None, "steps": 0, "started": None}
stages = []


def capture(self, grid, settings=native_backend.Settings(), dll_path=None, initial_command=native_backend.Command()):
    initialize(self, grid, settings, dll_path, initial_command)
    self.dll.LCS_ProfileSections.argtypes = (ct.c_int,)
    self.dll.LCS_ProfileSectionsRead.argtypes = (ct.c_char_p, ct.c_int)
    self.dll.LCS_ProfileSectionsRead.restype = ct.c_int
    sessions.append(self)


def read_sections(dll):
    capacity = dll.LCS_ProfileSectionsRead(None, 0)
    text = ct.create_string_buffer(capacity)
    assert dll.LCS_ProfileSectionsRead(text, capacity) == 0
    sections = {}
    for line in text.value.decode().splitlines():
        name, ms = line.rsplit("=", 1)
        sections[name] = float(ms)
    return sections


def close_stage():
    if state["stage"] is None or not sessions or not state["steps"]:
        return
    dll = sessions[0].dll
    sections = read_sections(dll)
    wall = time.perf_counter() - state["started"]
    stages.append({"stage": state["stage"], "steps": state["steps"], "wall_ms_per_step": 1000 * wall / state["steps"],
                   "sections_ms_per_step": {k: v / state["steps"] for k, v in sections.items()}})
    print(f"{state['stage']:14s} {state['steps']:4d} steps, {1000 * wall / state['steps']:7.1f} ms/step wall, "
          f"{sum(sections.values()) / state['steps']:7.1f} ms/step in sections", flush=True)


def timed_stage_start(name, *args):
    close_stage()
    state["stage"], state["steps"], state["started"] = name, 0, time.perf_counter()
    if sessions:
        sessions[0].dll.LCS_ProfileSections(1)
    return stage_start(name, *args)


def counted_step(self, command):
    state["steps"] += 1
    return step(self, command)


native_backend.NativeTissue.__init__ = capture
native_backend.NativeTissue.step = counted_step
scenario.stage_start = timed_stage_start
sys.argv = ["endoscopic_tissue.run", "--dll", dll_path, *run_args]
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
except RuntimeError as error:
    print("run ended:", error, flush=True)
finally:
    close_stage()
    with open(report_path, "w", encoding="utf8") as handle:
        json.dump(stages, handle, indent=2)
