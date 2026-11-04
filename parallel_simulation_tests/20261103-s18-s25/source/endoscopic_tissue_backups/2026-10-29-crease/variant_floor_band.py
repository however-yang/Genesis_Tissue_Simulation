"""Variant for a test build: the material step (its restriction and its line search) keeps every tet at or above
0.95 of the volume floor instead of the floor itself, while the vertex sweeps still bound compression at the floor.

Usage: python variant_floor_band.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path

assert sys.argv[1] == "apply"
path = Path("solver/material/material_line_search.cuh")
text = path.read_text(encoding="utf-8")
for old, new in (("    if(!materialLineFeasible(1.f,fminf(s.minimum,j0),j0,j1,j2,j3)){\n        float lo=0,hi=1,floor=fminf(s.minimum,j0);",
                  "    if(!materialLineFeasible(1.f,fminf(.95f*s.minimum,j0),j0,j1,j2,j3)){\n        float lo=0,hi=1,floor=fminf(.95f*s.minimum,j0);"),
                 ("if(!materialLineFeasible(a,fminf(s.minimum,J.j0),J.j0,J.j1,J.j2,J.j3))",
                  "if(!materialLineFeasible(a,fminf(.95f*s.minimum,J.j0),J.j0,J.j1,J.j2,J.j3))")):
    assert text.count(old) == 1, old
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("applied")
