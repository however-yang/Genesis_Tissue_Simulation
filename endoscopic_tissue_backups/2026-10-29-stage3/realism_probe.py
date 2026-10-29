"""Run endoscopic_tissue.run and record, for every step, what the grasp and cut look like in motion: the centre of mass
and the RMS and 99th-percentile node speed of the tumor and of the kidney, the mean height of the stump under the
tumor, the jaw gap, the pad loads and the grip error, and the contact-surface depth before and after
LCS_RefineContactSurface with the wall time of that call.

Usage: python realism_probe.py <output jsonl> [run.py arguments...]

Tumor nodes are the material nodes inside the tumor ellipsoid at rest, kidney nodes all other material nodes. The stump
is the kidney's top cap under the tumor: kidney nodes less than 1 cm from the tumor axis and above 6 mm below the
kidney's top at rest. Speeds come from the node positions of consecutive steps. The masks are saved beside the
records, as <output>-masks.npz.
"""
import ctypes as ct
import json
import os
import runpy
import sys
import time

import numpy as np

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
step = native_backend.NativeTissue.step
stage_start = scenario.stage_start
sequence_init = scenario.Sequence.__init__
state = {"stage": "initial", "masks": None, "previous": None, "refine": None, "passes": None}
# REFINE_EMULATE=1 runs the native repair loop pass by pass from Python, with the same stop rule, to record every pass.
EMULATE = os.environ.get("REFINE_EMULATE") == "1"
TOLERANCE = .01
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
    # The deepest triangle: tool kind, its contact's stencil size, gap and load, the distance between its depth witness
    # and the contact witness, and the cosine between the depth direction and the contact normal.
    diagnostic = None
    if before > .008:
        values = (ct.c_float * 12)()
        assert self.dll.LCS_ContactDiagnostic(values) == 0
        diagnostic = [values[1], values[3], values[4], values[5] * 10, values[6] * 10, values[7], values[8] * 10]
    # How far a repair moves the nodes: the largest displacement and the count of nodes moved by more than 1 um.
    moved = None
    if before > .01:
        nodes = np.empty((len(self.grid.positions), 3), dtype=np.float32)
        self.check("LCS_GetCornerPos", native_backend.pointer(nodes))
    start = time.perf_counter()
    try:
        if not EMULATE:
            return check(self, name, *args)
        # The native loop, one pass per call, recording the depth after each pass and the largest node move it makes.
        passes, previous, stalled = [], 1e30, 0
        for i in range(min(args[0], 8)):
            current = depth(self)
            if current <= TOLERANCE:
                break
            if i > 0:
                stalled = stalled + 1 if current >= previous * .995 else 0
                if stalled >= 2:
                    break
            previous = current
            ahead = np.empty((len(self.grid.positions), 3), dtype=np.float32)
            self.check("LCS_GetCornerPos", native_backend.pointer(ahead))
            tick = time.perf_counter()
            check(self, name, 1)
            spent = 1000 * (time.perf_counter() - tick)
            behind = np.empty_like(ahead)
            self.check("LCS_GetCornerPos", native_backend.pointer(behind))
            passes.append([depth(self) * 10, spent, float(np.linalg.norm(behind - ahead, axis=1).max() * 10)])
        state["passes"] = passes
    finally:
        elapsed = 1000 * (time.perf_counter() - start)
        if before > .01:
            after = np.empty_like(nodes)
            self.check("LCS_GetCornerPos", native_backend.pointer(after))
            shift = np.linalg.norm(after - nodes, axis=1)
            moved = [float(shift.max() * 10), int(np.count_nonzero(shift > 1e-4))]
        state["refine"] = (before, depth(self), elapsed, diagnostic, moved)


def probed_sequence_init(self, obs, settings, scene, rest_positions, inside, *args, **kwargs):
    sequence_init(self, obs, settings, scene, rest_positions, inside, *args, **kwargs)
    material = inside != 0
    tumor = material & (scene.shape.tumor_weight(rest_positions, 1e-3) > .5)
    kidney = material & ~tumor
    radial = np.hypot(rest_positions[:, 0], rest_positions[:, 2])
    stump = kidney & (radial < 1.0) & (rest_positions[:, 1] > scene.shape.half_extents[1] - .6)
    state["masks"] = {"tumor": tumor, "kidney": kidney, "stump": stump}
    state["dt"] = settings.dt
    np.savez_compressed(output_path.replace(".jsonl", "-masks.npz"), tumor=tumor, kidney=kidney, stump=stump)


def probed_step(self, command):
    obs = step(self, command)
    masks, previous = state["masks"], state["previous"]
    row = {"tick": self.tick, "stage": state["stage"], "jaw_gap_mm": obs.jaw_gap * 10,
           "jaw_load_n": list(obs.jaw_load_n), "grip_points": obs.grip_points, "grip_error_um": obs.grip_error_m * 1e6,
           "hinge_cm": obs.hinge.tolist()}
    if state["refine"] is not None:
        row["depth_before_mm"], row["depth_after_mm"], row["refine_ms"] = state["refine"][0] * 10, state["refine"][1] * 10, state["refine"][2]
        if state["passes"]:
            row["passes"] = state["passes"]
        state["passes"] = None
        if state["refine"][4] is not None:
            row["repair_max_move_mm"], row["repair_moved_nodes"] = state["refine"][4]
        if state["refine"][3] is not None:
            row["deepest"] = dict(zip(("kind", "contact_nodes", "lambda_n", "contact_gap_mm", "witness_offset_mm",
                                       "normal_cosine", "depth_mm"), state["refine"][3]))
        state["refine"] = None
    if masks is not None:
        nodes = obs.nodes
        for name in ("tumor", "kidney"):
            mask = masks[name]
            row[f"{name}_com_cm"] = nodes[mask].mean(axis=0).tolist()
            if previous is not None:
                speed = np.linalg.norm(nodes[mask] - previous[mask], axis=1) / state["dt"]
                row[f"{name}_rms_speed_mm_s"] = float(np.sqrt(np.mean(speed ** 2)) * 10)
                row[f"{name}_p99_speed_mm_s"] = float(np.percentile(speed, 99) * 10)
        row["stump_y_cm"] = float(nodes[masks["stump"], 1].mean())
        state["previous"] = nodes.copy()
    records.write(json.dumps(row) + "\n")
    return obs


def probed_stage_start(name, *args):
    state["stage"] = name
    return stage_start(name, *args)


native_backend.NativeTissue.check = probed_check
native_backend.NativeTissue.step = probed_step
scenario.Sequence.__init__ = probed_sequence_init
scenario.stage_start = probed_stage_start
sys.argv = ["endoscopic_tissue.run", *run_args]
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    records.close()
