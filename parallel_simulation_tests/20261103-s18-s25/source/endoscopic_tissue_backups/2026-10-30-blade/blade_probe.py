"""Run endoscopic_tissue.run with a DLL built by build_debug.sh and save the probe records of
../2026-10-30-snip3/flap_probe.py plus every scissors contact at every step with blade contact.

Usage: python blade_probe.py <output npz> [run.py arguments...]

The contacts come from LCS_GetBladeContacts (debug_blade_contacts.py): contact_steps gives the step of each row of
contacts, whose 11 columns are the kind, the material witness, the normal, the energy gradient term and the gap.

The npz holds the rest positions, the material mask and the initial neighbors of the lattice, the stage names in order
and the node positions at the end of each stage (stage_nodes), every severed link with the step that severed it, its
node and its slot (cut_steps, cut_nodes, cut_slots), the positions at every step of the lattice nodes around the flap
(track_ids, track_nodes) and the grasper hinge (hinges), and the contact-surface depth in mm before and after
LCS_RefineContactSurface at every step (repair_steps, repair_depths), whose passes run only while that depth exceeds the
repair tolerance.

Two environment variables change the run. SCISSORS_DZ (cm) moves every scissors pose of the flap scene along z, and the
scene's face offset with it, so the blades meet the flap off its rest mid-plane. REPAIR_PASSES sets the contact-surface
repair passes per step of the scissors flows instead of Settings.contact_refine_passes.
"""
import ctypes as ct
import os
import runpy
import sys
from dataclasses import replace

import numpy as np

from endoscopic_tissue import native_backend, scenario

output_path, *run_args = sys.argv[1:]
offset = float(os.environ.get("SCISSORS_DZ", "0"))
flap = scenario.SCISSORS_SCENES["flap"]
stages = tuple(replace(stage, command=replace(stage.command, blade_center=tuple(np.add(stage.command.blade_center,
                                                                                         (0, 0, offset)))))
               for stage in flap.stages)
scenario.SCISSORS_SCENES["flap"] = replace(flap, stages=stages, face_offset=flap.face_offset + offset)
if "REPAIR_PASSES" in os.environ:
    settings = scenario.scissors_settings
    scenario.scissors_settings = lambda: replace(settings(), contact_refine_passes=int(os.environ["REPAIR_PASSES"]))
step = native_backend.NativeTissue.step
check = native_backend.NativeTissue.check
stage_start = scenario.stage_start
sequence_init = scenario.Sequence.__init__
state = {"stage": None, "tick": 0, "last": None, "severed": None, "names": [], "nodes": [], "cuts": [], "rest": None,
         "inside": None, "initial": None, "track": None, "tracked": [], "hinges": [], "repairs": {},
         "contact_steps": [], "contacts": []}
CAPACITY = 8192


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
    result = check(self, name, *args)
    state["repairs"][state["tick"] + 1] = (before * 10, depth(self) * 10)
    return result


def probed_sequence_init(self, obs, settings, scene, rest_positions, inside, *args, **kwargs):
    sequence_init(self, obs, settings, scene, rest_positions, inside, *args, **kwargs)
    state["rest"], state["inside"], state["initial"] = rest_positions.copy(), inside.copy(), obs.neighbors.copy()
    state["severed"] = np.zeros(obs.neighbors.shape, dtype=bool)
    near = (rest_positions[:, 0] > 2.75) & (rest_positions[:, 1] > .2) & (np.abs(rest_positions[:, 2] - .05) < .5)
    state["track"] = np.flatnonzero(near)


def probed_step(self, command):
    obs = step(self, command)
    state["tick"] += 1
    state["last"] = obs
    if state["severed"] is not None:
        severed = (state["initial"] >= 0) & (obs.neighbors < 0)
        for node, slot in zip(*np.nonzero(severed & ~state["severed"])):
            state["cuts"].append((state["tick"], node, slot))
        state["severed"] = severed
        state["tracked"].append(obs.nodes[state["track"]].astype(np.float32))
        state["hinges"].append(np.array(obs.hinge, dtype=np.float32))
        if obs.scissors is not None and obs.scissors.contacts:
            values, count = np.empty((CAPACITY, 11), dtype=np.float32), ct.c_int()
            assert self.dll.LCS_GetBladeContacts(native_backend.pointer(values), CAPACITY, ct.byref(count)) == 0
            n = min(count.value, CAPACITY)
            state["contact_steps"].append(np.full(n, state["tick"], dtype=np.int32))
            state["contacts"].append(values[:n].copy())
    return obs


def close_stage():
    if state["stage"] is not None and state["last"] is not None:
        state["names"].append(state["stage"])
        state["nodes"].append(state["last"].nodes.astype(np.float32))


def probed_stage_start(name, *args):
    close_stage()
    state["stage"] = name
    return stage_start(name, *args)


native_backend.NativeTissue.step = probed_step
native_backend.NativeTissue.check = probed_check
scenario.Sequence.__init__ = probed_sequence_init
scenario.stage_start = probed_stage_start
sys.argv = ["endoscopic_tissue.run", *run_args]
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    close_stage()
    cuts = np.array(state["cuts"], dtype=np.int64).reshape(-1, 3)
    repairs = sorted(state["repairs"])
    np.savez_compressed(output_path, rest=state["rest"], inside=state["inside"], initial=state["initial"],
                        names=np.array(state["names"]), stage_nodes=np.stack(state["nodes"]), cut_steps=cuts[:, 0],
                        cut_nodes=cuts[:, 1], cut_slots=cuts[:, 2], track_ids=state["track"],
                        track_nodes=np.stack(state["tracked"]), hinges=np.stack(state["hinges"]),
                        repair_steps=np.array(repairs),
                        repair_depths=np.array([state["repairs"][k] for k in repairs]).reshape(-1, 2),
                        contact_steps=np.concatenate(state["contact_steps"] or [np.zeros(0, dtype=np.int32)]),
                        contacts=np.concatenate(state["contacts"] or [np.zeros((0, 11), dtype=np.float32)]))
