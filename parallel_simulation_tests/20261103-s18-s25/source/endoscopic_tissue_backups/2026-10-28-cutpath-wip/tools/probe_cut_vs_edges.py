"""Check, link by link, that the scissors cut where their edges are.

Runs endoscopic_tissue.run with the given arguments. At every step that severs links, each newly severed link is
measured against the solved scissors pose of the steps just before and at the cut:

- whether its two nodes lie on opposite sides of the shear plane through both cutting edges,
- how far the crack point, placed at the native cut fraction along the link, lies from that plane,
- where the crossing point lies in the plane relative to each cutting edge: the in-plane distance past the edge
  (positive once the edge has swept over it, negative while it is still ahead of the edge).

The shear plane is fitted to the four endpoints of the two edge segments; their fit residual is reported. Records go to
the JSON file named by the environment variable PROBE_OUT, written when the run ends.
"""
import ctypes as ct
import json
import os
import runpy
import sys

import numpy as np

from endoscopic_tissue import native_backend, scenario

records, state = [], {"tissue": None, "previous": None, "step": 0}
step = native_backend.NativeTissue.step
record_step = scenario.Sequence.record_step


def plane_of(scissors):
    points = np.asarray(scissors.edges, dtype=float).reshape(-1, 3)
    center = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - center)
    return center, vt[-1], float(singular[-1] / 2)


def past_edge(point, scissors, normal):
    """In-plane distance of `point` past each cutting edge, positive on the side the edge has already swept."""
    edges = np.asarray(scissors.edges, dtype=float)
    result = []
    for i in range(2):
        a, b = edges[i]
        other = edges[1 - i].mean(axis=0)
        along = (b - a) / np.linalg.norm(b - a)
        side = np.cross(normal, along)
        # The open wedge, not yet swept by this edge, lies toward the other blade's edge.
        sign = -np.sign((other - a) @ side) or 1.0
        result.append(float(sign * ((point - a) @ side)))
    return result


def traced_step(self, command):
    state["tissue"] = self
    return step(self, command)


def traced_record(self, stage, obs, *args, **kwargs):
    state["step"] += 1
    previous = state["previous"]
    if previous is not None and obs.scissors is not None and previous.scissors is not None:
        severed = (previous.neighbors[:, 1::2] >= 0) & (obs.neighbors[:, 1::2] < 0)
        ids, axes = np.nonzero(severed)
        if len(ids):
            fractions = np.zeros(6 * len(obs.nodes), dtype=np.float32)
            state["tissue"].dll.LCS_GetCutEdgeFractions(fractions.ctypes.data_as(ct.POINTER(ct.c_float)))
            for a, axis in zip(ids.tolist(), axes.tolist()):
                b = int(previous.neighbors[a, 2 * axis + 1])
                t = float(fractions[6 * a + 2 * axis + 1])
                entry = {"step": state["step"], "stage": stage, "a": a, "b": b, "axis": axis,
                         "material": bool(self.inside[a] or self.inside[b]), "fraction": t}
                for label, o in (("before", previous), ("after", obs)):
                    center, normal, residual = plane_of(o.scissors)
                    pa, pb = o.nodes[a].astype(float), o.nodes[b].astype(float)
                    da, db = (pa - center) @ normal, (pb - center) @ normal
                    crack = pa + t * (pb - pa)
                    entry[label] = {"straddles": bool(da * db < 0), "da": float(da), "db": float(db),
                                    "crack_off_plane": float((crack - center) @ normal), "plane_residual": residual,
                                    "geometric_fraction": float(da / (da - db)) if da != db else None,
                                    "past_edges": past_edge(crack - normal * ((crack - center) @ normal), o.scissors,
                                                            normal),
                                    "gap": float(o.scissors.gap)}
                records.append(entry)
    state["previous"] = obs
    return record_step(self, stage, obs, *args, **kwargs)


native_backend.NativeTissue.step = traced_step
scenario.Sequence.record_step = traced_record
sys.argv[0] = "endoscopic_tissue.run"
try:
    runpy.run_module("endoscopic_tissue.run", run_name="__main__")
finally:
    with open(os.environ["PROBE_OUT"], "w") as f:
        json.dump(records, f)
    print("PROBE records", len(records), flush=True)
