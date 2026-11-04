"""Group the per-kernel profile of the kidney flow into soft-tissue mechanics, grasping and contact, and cutting.

Usage: python bottleneck_groups.py <sections json> [<sections json> ...]

The profile comes from profile_sections.py on a fine_marks.py build: kernels run one by one, so the totals sit a few
ms above the graph-launch timing, and host gaps between steps appear under "step start".
"""
import json
import sys

GROUPS = ("mechanics", "grasp and contact", "cutting", "host gap")
CONTACT_KERNELS = ("k_CoarseContactGrip", "k_CoarseContactBound", "k_PressureLineContacts", "k_RigidContactKinds",
                   "k_UnifiedPatchBasis", "k_PredictorInteractionEnergy", "k_UnifiedTool", "k_AssembleUnifiedInteractions")


def group(name):
    if name == "step start":
        return "host gap", "host and Python between steps"
    if "k_BladeCoarse" in name:
        return "cutting", "blade terms in the coarse modes"
    if name == "fine: discovery k_MarkCoarseContact" or name == "loop: manifold, contact marks":
        # findBlade runs just before these marks (host function, not marked itself).
        return "cutting", "blade contact search (findBlade) and contact flags"
    if name == "begin frame, cut tick and rebuild between substeps":
        return "cutting", "cut tick, fracture, release, surface rebuild, frame setup"
    if name.startswith("fine: fragment") and "finishFeasible" not in name or name.startswith("fine: prologue") \
            or name in ("post: fragment contact", "substep prologue: liveness, fragment contact, predict"):
        return "cutting", "separated pieces: component labels, fragment contact"
    if any(k in name for k in CONTACT_KERNELS) or name.startswith("fine: global"):
        return "grasp and contact", "contact terms, tool solve, grasp patch block"
    if name.startswith("fine: discovery") or name in ("pre: tool, contact discovery, manifold", "loop: contact discovery"):
        return "grasp and contact", "grasper contact discovery and manifold"
    if name == "fine: sweeps k_UnifiedVertex witness":
        return "grasp and contact", "witness sweeps (contact and anchor nodes)"
    if name == "post: blade and contact reactions, velocity":
        return "grasp and contact", "contact and blade reactions, velocity"
    if name.startswith("fine: cg") or name.startswith("cg: iterations"):
        return "mechanics", "PCG iterations"
    if name.startswith("fine: assembly") or name.startswith("cg assembly") or name == "cg: assembly":
        return "mechanics", "stiffness assembly and residual"
    if name.startswith("fine: coarse") or name == "loop: coarse modes":
        return "mechanics", "coarse modes"
    if name.startswith("fine: regional") or name == "loop: regional mode":
        return "mechanics", "regional mode"
    if name.startswith("fine: rigid") or name == "loop: rigid modes" or name == "post: free-component nullspace":
        return "mechanics", "rigid and nullspace modes"
    if name.startswith("fine: sweeps") or name == "loop: local sweeps":
        return "mechanics", "full vertex sweeps"
    if name.startswith("fine: pre") or name == "pre: predictor, recovery query":
        return "mechanics", "predictor, tet cache, recovery query"
    if name.startswith("fine: line") or name.startswith("fine: feasible") or name in ("cg: line search", "cg: feasible", "cg: trust, projection") \
            or "finishFeasible" in name:
        return "mechanics", "line search and volume safeguards"
    if name == "loop: tool, safe and global coarse":
        return "grasp and contact", "contact terms, tool solve, grasp patch block"
    return "mechanics", "other: " + name


PHASES = {"held (bite to blade approach)": ("bite", "close", "lift", "blade_approach"),
          "cutting (incisions)": ("incise_1", "open_1", "reposition_2", "incise_2"),
          "after the excision": ("withdraw", "release", "extract", "hold")}
for path in sys.argv[1:]:
    stages = {s["stage"]: s for s in json.load(open(path))}
    print("==", path)
    for phase, names in PHASES.items():
        totals, parts, steps = {}, {}, 0
        for name in names:
            if name not in stages:
                continue
            n = stages[name]["steps"]
            steps += n
            for key, value in stages[name]["sections_ms_per_step"].items():
                g, part = group(key)
                totals[g] = totals.get(g, 0) + value * n
                parts[(g, part)] = parts.get((g, part), 0) + value * n
        if not steps:
            continue
        overall = sum(totals.values()) / steps
        print(f"{phase}: {overall:.1f} ms per step ({steps} steps)")
        for g in GROUPS:
            if g not in totals:
                continue
            print(f"   {g:18s} {totals[g] / steps:5.1f} ms  {100 * totals[g] / steps / overall:4.0f}%")
            for (pg, part), value in sorted(parts.items(), key=lambda kv: -kv[1]):
                if pg == g and value / steps >= 0.3:
                    print(f"      {value / steps:5.1f}  {part}")
