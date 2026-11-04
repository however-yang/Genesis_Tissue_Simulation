"""Run the outer iterations of the unified substep as a given sequence of pass types, as a convergence ablation.

The types 0-3 are the four passes of the default sequence 0123: 0 the first (coarse modes 1-4, PCG, discovery, regional
mode), 1 a plain pass (manifold refresh, local sweep, tool solve, tool/safe/global coarse, rigid modes), 2 a pass that
adds coarse mode 1, discovery and the regional mode, 3 the last (as 1, with the coupled-ready flag cleared). Types 4
and 5 leave out the tool/safe/global coarse and the rigid modes: 4 keeps the manifold refresh, 5 leaves it out too.

Usage: python variant_pass_pattern.py apply <sequence>, e.g. 01211113   (from endoscopic_tissue/native/vendor/cuda; undo
with git checkout)
"""
import sys
from pathlib import Path


def patch(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


assert sys.argv[1] == "apply"
sequence = sys.argv[2]
assert sequence[0] == "0" and sequence[-1] == "3" and set(sequence) <= set("012345")
path = Path("xpbd/xpbd_solver.cu")
text = path.read_text(encoding="utf-8")
text = patch(text, "                    };\n        for(int it=0;it<4;it++){\n            if(it==3)",
             f"                    }};\n        static const int passPattern[]={{{','.join(sequence)}}};\n"
             f"        for(int pass=0;pass<{len(sequence)};pass++){{const int it=passPattern[pass];\n            if(it==3)")
text = patch(text, "            if(it==3)k_MaterialReady<<<1,1,0,g_captureStream>>>(d_unified,false);\n"
             "            if(it>0 && g_surfaceCount>0){",
             "            if(it==3)k_MaterialReady<<<1,1,0,g_captureStream>>>(d_unified,false);\n"
             "            if(it>0 && it!=5 && g_surfaceCount>0){")
text = patch(text, "            // The coupled tool/patch direction needs the same feasible basis\n",
             "            if(it<4){\n            // The coupled tool/patch direction needs the same feasible basis\n")
text = patch(text, '            xpbd_section_mark("loop: rigid modes");\n        }\n        if(direct)g_captureStream=captureStream;',
             '            xpbd_section_mark("loop: rigid modes");\n            }\n        }\n        if(direct)g_captureStream=captureStream;')
path.write_text(text, encoding="utf-8")
print("applied")
