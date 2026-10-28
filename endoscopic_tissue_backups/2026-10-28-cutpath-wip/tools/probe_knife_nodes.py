"""Run the knife flow and report material nodes that lose four or more of their links.

A planar incision severs at most the links of a node that cross the blade plane, so a node losing four or more links,
or all six, is the anomaly to explain. For every such node the report gives each severed link: the step and stage it
parted in, the native cut fraction along it, and the solved blade centre then, plus the node's rest position relative to
the blade segment (along its axis and off it). Records go to the JSON file named by PROBE_OUT when the run ends.
"""
import ctypes as ct
import json
import os
import runpy
import sys

import numpy as np

from endoscopic_tissue import native_backend, scenario
from endoscopic_tissue.grid import build_grid

state = {"tissue": None, "previous": None, "step": 0, "cuts": {}, "isolated_at": {}}
step = native_backend.NativeTissue.step
record_step = scenario.Sequence.record_step


def traced_step(self, command):
    state["tissue"] = self
    return step(self, command)


def traced_record(self, stage, obs, *args, **kwargs):
    state["step"] += 1
    previous = state["previous"]
    if previous is not None:
        ids, slots = np.nonzero((previous.neighbors >= 0) & (obs.neighbors < 0))
        if len(ids):
            fractions = np.zeros(6 * len(obs.nodes), dtype=np.float32)
            state["tissue"].dll.LCS_GetCutEdgeFractions(fractions.ctypes.data_as(ct.POINTER(ct.c_float)))
            for a, slot in zip(ids.tolist(), slots.tolist()):
                state["cuts"][(a, slot)] = {"step": state["step"], "stage": stage,
                                            "fraction": float(fractions[6 * a + slot]),
                                            "blade_centre": obs.blade[:3].tolist(),
                                            "node_now": obs.nodes[a].tolist()}
            alive = (obs.neighbors >= 0).sum(axis=1)
            for a in np.unique(ids).tolist():
                if alive[a] == 0 and a not in state["isolated_at"]:
                    state["isolated_at"][a] = {"step": state["step"], "stage": stage}
    state["previous"] = obs
    return record_step(self, stage, obs, *args, **kwargs)


native_backend.NativeTissue.step = traced_step
scenario.Sequence.record_step = traced_record
sys.argv[0] = "endoscopic_tissue.run"
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    settings = native_backend.Settings()
    grid = build_grid(.2, density=settings.density_kg_m3, vertical_spacing=.05, tab=None)
    inside = grid.inside != 0
    axis = np.array(settings.blade_axis) / np.linalg.norm(settings.blade_axis)
    severed = {}
    # Every severed link appears from both of its nodes.
    for (a, slot) in state["cuts"]:
        severed.setdefault(a, []).append(slot)
    report = []
    for node, slots in severed.items():
        if not inside[node] or len(slots) < 4:
            continue
        links = []
        for slot in slots:
            info = dict(state["cuts"][(node, slot)])
            info["slot"] = slot
            links.append(info)
        centre = np.array(links[0].get("blade_centre", [0, 0, 0]))
        offset = grid.positions[node] - centre
        report.append({"node": node, "rest": grid.positions[node].tolist(), "severed_links": len(slots),
                       "isolated": state["isolated_at"].get(node), "along_blade_cm": float(offset @ axis),
                       "off_blade_xz_cm": float(np.linalg.norm((offset - axis * (offset @ axis))[[0, 2]])),
                       "half_length_cm": settings.blade_length / 2, "links": links})
    with open(os.environ["PROBE_OUT"], "w") as f:
        json.dump(report, f, indent=1)
    print("PROBE nodes with >= 4 severed links:", len(report), "isolated:", len(state["isolated_at"]), flush=True)
