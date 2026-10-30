"""Variant for a test build: project the material direction of every component with a tet at the volume floor into
the tangent cone of the floor (k_ProjectRecoveryDirection), grasped components included, in every solve graph.
Without it a grasped component only restricts the direction node by node, which zeroes the nodes of floor tets.

Usage: python variant_grasped_projection.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
patch("solver/material/material_line_search.cuh",
      "int id[4];unifiedTetIds(s,t,id);int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8)||(flags&7))return;",
      "int id[4];unifiedTetIds(s,t,id);int flags=s.graspComponents[s.component[id[0]]];if(!(flags&8))return;")
patch("xpbd/xpbd_solver.cu", "for(int projection=0;recoveryRequested && projection<64;projection++){",
      "for(int projection=0;projection<64;projection++){")
print("applied")
