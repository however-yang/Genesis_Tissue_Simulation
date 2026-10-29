"""Run endoscopic_tissue.run on a debug_material_dump.py build and dump the material PCG system after chosen steps.

Usage: python dump_material_run.py <dump folder> <step,step,...> [run.py arguments...]

Each dump holds the PCG system of the last substep of that step (debug_material_dump.py), in the file
<dump folder>/step<step>.bin. The kidney flow's crack tips fail the strict topology audit; they are recorded instead.
"""
import ctypes as ct
import runpy
import sys
from pathlib import Path

from endoscopic_tissue import native_backend, surface_diagnostics

folder, steps, *run_args = sys.argv[1:]
folder = Path(folder)
folder.mkdir(parents=True, exist_ok=True)
wanted = {int(s) for s in steps.split(",")}
audit = surface_diagnostics.topology_statistics


def recording_audit(surface):
    statistics = audit(surface)
    statistics["recorded_nonmanifold_edges"] = statistics["positional_nonmanifold_edges"]
    statistics["positional_nonmanifold_edges"] = 0
    return statistics


surface_diagnostics.topology_statistics = recording_audit
step = native_backend.NativeTissue.step


def dumping_step(self, command):
    observation = step(self, command)
    if self.tick in wanted:
        function = self.dll["LCS_DebugDumpMaterial"]
        function.argtypes, function.restype = (ct.c_char_p,), ct.c_int
        code = function(str(folder / f"step{self.tick}.bin").encode())
        print(f"DUMP step {self.tick} -> {code}", flush=True)
    return observation


native_backend.NativeTissue.step = dumping_step
sys.argv = ["endoscopic_tissue.run", *run_args]
runpy.run_module("endoscopic_tissue.run", run_name="__main__")
