"""Drop the second outer iteration of the unified substep (a plain manifold refresh, local sweep, tool solve and global
coarse and rigid modes), keeping the first (coarse modes 1-4, PCG, discovery, regional mode), the third (coarse mode 1,
discovery, regional mode) and the last (with the full vertex sweep).

Usage: python variant_three_iterations.py apply   (from endoscopic_tissue/native/vendor/cuda; undo with git checkout)
"""
import sys
from pathlib import Path

assert sys.argv[1] == "apply"
path = Path("xpbd/xpbd_solver.cu")
text = path.read_text(encoding="utf-8")
old = "                    };\n        for(int it=0;it<4;it++){\n            if(it==3)"
assert text.count(old) == 1
path.write_text(text.replace(old, "                    };\n        for(int it=0;it<4;it+=it==0?2:1){\n            if(it==3)"), encoding="utf-8")
print("applied")
