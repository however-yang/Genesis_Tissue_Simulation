"""Variant for a test build: the PCG stencil gives each tet within twice the volume floor an extra volumetric
curvature lambda*((floor/(J-floor))^2-1), zero at twice the floor and capped at 1e4*lambda next to it, so the
material direction slides along the floor instead of pressing into it. The energy, its gradient and the line search
stay as they are: only the search direction changes.

Usage: python variant_floor_curvature.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
patch("solver/material/material_stencil.cuh",
      "        float barrierCurvature=unifiedCompressionBarrier(dot3(F[0],cof[0]),s.minimum,unifiedKuhn(s,t).bulkV+unifiedKuhn(s,t).muV/3.f).curvature;\n",
      """        float barrierJ=dot3(F[0],cof[0]),barrierLambda=unifiedKuhn(s,t).bulkV+unifiedKuhn(s,t).muV/3.f;
        float barrierCurvature=unifiedCompressionBarrier(barrierJ,s.minimum,barrierLambda).curvature;
        if(s.minimum>0 && barrierJ<2*s.minimum){
            float r=s.minimum/fmaxf(barrierJ-s.minimum,1e-2f*s.minimum);barrierCurvature+=barrierLambda*(r*r-1.f);
        }
""")
print("applied")
