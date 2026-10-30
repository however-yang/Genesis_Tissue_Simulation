"""Variant for a test build: the volume-floor restriction of the material direction scales every component as a whole
by its largest feasible fraction, grasped components included, instead of scaling a grasped component node by node.

Usage: python variant_uniform_restriction.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path


def patch(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old[:80])
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


assert sys.argv[1] == "apply"
SEARCH = "solver/material/material_line_search.cuh"
patch(SEARCH, "        int iterations=(iterationFlags&8)&&!(iterationFlags&7)?24:12;", "        int iterations=24;(void)iterationFlags;")
patch(SEARCH, """        if((flags&8)&&!(flags&7))atomicMin((unsigned*)(blocked+root),__float_as_uint(lo));
        else for(int j=0;j<4;j++)atomicMin((unsigned*)(blocked+id[j]),__float_as_uint(lo));""",
      """        (void)flags;atomicMin((unsigned*)(blocked+root),__float_as_uint(lo));""")
patch(SEARCH, "    float a=blocked[(flags&8)&&!(flags&7)?root:i];", "    (void)flags;float a=state?blocked[root]:blocked[i];")
print("applied")
