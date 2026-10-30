"""Run more outer iterations in the unified substep, as a convergence reference: <count> passes instead of 4, the
first and the last as now (it 0 and it 3) and the passes between alternating the second (a plain manifold refresh,
local sweep, tool solve, global coarse and rigid modes) and the third (coarse mode 1, discovery, regional mode).

Usage: python variant_more_iterations.py apply <count>   (from endoscopic_tissue/native/vendor/cuda; undo with git
checkout)
"""
import sys
from pathlib import Path

assert sys.argv[1] == "apply"
count = int(sys.argv[2])
assert count >= 4 and count % 2 == 0
path = Path("xpbd/xpbd_solver.cu")
text = path.read_text(encoding="utf-8")
old = "                    };\n        for(int it=0;it<4;it++){\n            if(it==3)"
new = (f"                    }};\n        for(int pass=0;pass<{count};pass++){{const int it=pass==0?0:pass=={count - 1}?3:2-(pass&1);\n"
       "            if(it==3)")
assert text.count(old) == 1
path.write_text(text.replace(old, new), encoding="utf-8")
print("applied")
